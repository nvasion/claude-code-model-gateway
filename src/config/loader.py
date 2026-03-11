"""Extended configuration loading utilities for claude-code-model-gateway.

Builds on the base :mod:`src.config` loader with support for:

- Multi-file configuration merging (overlay patterns)
- Environment variable interpolation inside config values
- Configuration file watching and auto-reload callbacks
- Safe config loading with fallback defaults
- Config file discovery with extended search paths

Example usage::

    from src.config.loader import (
        load_and_merge,
        load_with_defaults,
        ConfigWatcher,
    )

    # Merge base + override configs
    config = load_and_merge(["base.yaml", "override.yaml"])

    # Load with safe fallback
    config = load_with_defaults("gateway.yaml")

    # Watch for changes
    watcher = ConfigWatcher("gateway.yaml", on_change=my_callback)
    watcher.start()
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from src.config import (
    ConfigError,
    ConfigValidationError,
    load_config,
    load_config_file,
    validate_config,
)
from src.models import GatewayConfig


# ---------------------------------------------------------------------------
# Multi-file merging
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *overlay* into *base* (non-destructive).

    For nested dictionaries, values are merged recursively.  For all
    other types, the overlay value replaces the base value.

    Args:
        base: The base configuration dictionary.
        overlay: Values to merge on top.

    Returns:
        A new merged dictionary (neither input is mutated).
    """
    result = copy.deepcopy(base)
    _deep_merge_in_place(result, overlay)
    return result


def _deep_merge_in_place(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Recursively merge source into target in-place."""
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge_in_place(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def load_and_merge(
    paths: list[str | Path],
    validate: bool = True,
) -> GatewayConfig:
    """Load and merge multiple configuration files in order.

    Later files take precedence over earlier ones.  This enables
    a base + overlay pattern (e.g., ``base.yaml`` + ``production.yaml``).

    Args:
        paths: Ordered list of config file paths to merge.
        validate: Whether to validate the merged result.

    Returns:
        The merged GatewayConfig.

    Raises:
        ConfigError: If any file cannot be loaded.
        ConfigValidationError: If validation fails.
    """
    if not paths:
        raise ConfigError("No configuration files specified.")

    merged: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        data = load_config_file(path)
        merged = deep_merge(merged, data)

    config = GatewayConfig.from_dict(merged)

    if validate:
        errors = validate_config(config)
        if errors:
            raise ConfigValidationError(errors)

    return config


# ---------------------------------------------------------------------------
# Environment variable interpolation
# ---------------------------------------------------------------------------

# Pattern: ${VAR_NAME} or ${VAR_NAME:-default_value}
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def interpolate_env_vars(
    data: dict[str, Any],
    env: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Replace ``${VAR}`` and ``${VAR:-default}`` placeholders in config values.

    Only string values are interpolated; nested dictionaries are
    processed recursively.

    Args:
        data: Configuration dictionary with potential placeholders.
        env: Environment mapping (defaults to ``os.environ``).

    Returns:
        A new dictionary with placeholders resolved.
    """
    if env is None:
        env = dict(os.environ)
    return _interpolate_recursive(data, env)


def _interpolate_recursive(
    data: Any, env: dict[str, str]
) -> Any:
    """Recursively interpolate environment variables."""
    if isinstance(data, dict):
        return {k: _interpolate_recursive(v, env) for k, v in data.items()}
    if isinstance(data, list):
        return [_interpolate_recursive(item, env) for item in data]
    if isinstance(data, str):
        return _interpolate_string(data, env)
    return data


def _interpolate_string(value: str, env: dict[str, str]) -> str:
    """Interpolate ${VAR} patterns in a single string value."""

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_value = env.get(var_name)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        return match.group(0)  # Leave unresolved

    return _ENV_VAR_PATTERN.sub(replacer, value)


def load_with_interpolation(
    path: str | Path,
    validate: bool = True,
    env: Optional[dict[str, str]] = None,
) -> GatewayConfig:
    """Load a config file with environment variable interpolation.

    Args:
        path: Path to the configuration file.
        validate: Whether to validate the loaded configuration.
        env: Optional environment mapping (defaults to os.environ).

    Returns:
        The loaded and interpolated GatewayConfig.

    Raises:
        ConfigError: If the file cannot be loaded.
        ConfigValidationError: If validation fails.
    """
    data = load_config_file(Path(path))
    data = interpolate_env_vars(data, env=env)
    config = GatewayConfig.from_dict(data)

    if validate:
        errors = validate_config(config)
        if errors:
            raise ConfigValidationError(errors)

    return config


# ---------------------------------------------------------------------------
# Safe loading with defaults
# ---------------------------------------------------------------------------


def load_with_defaults(
    path: Optional[str | Path] = None,
    validate: bool = True,
) -> GatewayConfig:
    """Load configuration with safe fallback to defaults.

    If the config file doesn't exist or fails to load, returns the
    default configuration instead of raising an error.

    Args:
        path: Path to the configuration file.  If None, uses auto-discovery.
        validate: Whether to validate the loaded configuration.

    Returns:
        The loaded configuration, or defaults on failure.
    """
    try:
        path_obj = Path(path) if path else None
        return load_config(path=path_obj, validate=validate)
    except (ConfigError, ConfigValidationError):
        from src.config import get_default_config

        return get_default_config()


# ---------------------------------------------------------------------------
# Extended config file discovery
# ---------------------------------------------------------------------------


def discover_config_files(
    search_dirs: Optional[list[str | Path]] = None,
    filenames: Optional[list[str]] = None,
) -> list[Path]:
    """Discover all configuration files in the given directories.

    Args:
        search_dirs: Directories to search.  Defaults to current directory
                     and ``~/.config/claude-code-model-gateway/``.
        filenames: Filenames to look for.  Defaults to
                   ``gateway.yaml``, ``gateway.yml``, ``gateway.json``.

    Returns:
        List of found configuration file paths (sorted by priority).
    """
    if search_dirs is None:
        search_dirs = [
            Path.cwd(),
            Path.home() / ".config" / "claude-code-model-gateway",
        ]
    if filenames is None:
        filenames = ["gateway.yaml", "gateway.yml", "gateway.json"]

    found: list[Path] = []
    for dir_path in search_dirs:
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            continue
        for name in filenames:
            candidate = dir_path / name
            if candidate.is_file():
                found.append(candidate)

    return found


# ---------------------------------------------------------------------------
# Configuration file watcher
# ---------------------------------------------------------------------------


class ConfigWatcher:
    """Watch a configuration file for changes and invoke a callback.

    Uses a simple polling approach that works on all platforms.
    The watcher runs in a daemon thread and can be stopped gracefully.

    Attributes:
        path: The file being watched.
        interval: Polling interval in seconds.
        on_change: Callback invoked with the new GatewayConfig on change.
        on_error: Callback invoked with the exception on load failure.

    Example::

        def handle_change(config: GatewayConfig) -> None:
            print(f"Config reloaded! Timeout is now {config.timeout}")

        watcher = ConfigWatcher("gateway.yaml", on_change=handle_change)
        watcher.start()
        # ... later ...
        watcher.stop()
    """

    def __init__(
        self,
        path: str | Path,
        on_change: Callable[[GatewayConfig], None],
        on_error: Optional[Callable[[Exception], None]] = None,
        interval: float = 2.0,
        validate: bool = True,
    ) -> None:
        self.path = Path(path)
        self.on_change = on_change
        self.on_error = on_error
        self.interval = interval
        self.validate = validate

        self._last_mtime: Optional[float] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        """Whether the watcher is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start watching the configuration file in a background thread."""
        if self.is_running:
            return

        self._stop_event.clear()
        self._last_mtime = self._get_mtime()
        self._thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name=f"config-watcher-{self.path.name}",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the watcher thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
            self._thread = None

    def check_now(self) -> bool:
        """Perform an immediate change check.

        Returns:
            True if the file changed and the callback was invoked.
        """
        return self._check_for_changes()

    def _watch_loop(self) -> None:
        """Background polling loop."""
        while not self._stop_event.is_set():
            self._check_for_changes()
            self._stop_event.wait(timeout=self.interval)

    def _check_for_changes(self) -> bool:
        """Check if the file has changed since the last check."""
        current_mtime = self._get_mtime()
        if current_mtime is None:
            return False

        if self._last_mtime is None or current_mtime != self._last_mtime:
            self._last_mtime = current_mtime
            try:
                config = load_config(
                    path=self.path,
                    validate=self.validate,
                    use_cache=False,
                )
                self.on_change(config)
                return True
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
        return False

    def _get_mtime(self) -> Optional[float]:
        """Get the file modification time, or None if unavailable."""
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None


# ---------------------------------------------------------------------------
# Config export helpers
# ---------------------------------------------------------------------------


def export_config(
    config: GatewayConfig,
    fmt: str = "yaml",
) -> str:
    """Export a GatewayConfig to a string in the specified format.

    Args:
        config: The configuration to export.
        fmt: Output format ('yaml' or 'json').

    Returns:
        The serialized configuration string.

    Raises:
        ValueError: If the format is unsupported.
    """
    data = config.to_dict()

    if fmt in ("yaml", "yml"):
        return yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    elif fmt == "json":
        return json.dumps(data, indent=2) + "\n"
    else:
        raise ValueError(
            f"Unsupported export format: {fmt}. Use 'yaml' or 'json'."
        )


def import_config(
    content: str,
    fmt: str = "yaml",
    validate: bool = True,
) -> GatewayConfig:
    """Import a GatewayConfig from a string.

    Args:
        content: The serialized configuration string.
        fmt: Input format ('yaml' or 'json').
        validate: Whether to validate the configuration.

    Returns:
        The deserialized GatewayConfig.

    Raises:
        ConfigError: If parsing fails.
        ConfigValidationError: If validation fails.
    """
    try:
        if fmt in ("yaml", "yml"):
            data = yaml.safe_load(content)
        elif fmt == "json":
            data = json.loads(content)
        else:
            raise ConfigError(
                f"Unsupported import format: {fmt}. Use 'yaml' or 'json'."
            )
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigError(f"Failed to parse configuration: {e}")

    if not isinstance(data, dict):
        raise ConfigError("Configuration must be a mapping/object.")

    config = GatewayConfig.from_dict(data)

    if validate:
        errors = validate_config(config)
        if errors:
            raise ConfigValidationError(errors)

    return config
