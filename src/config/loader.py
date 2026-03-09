"""Extended configuration loading utilities for the model gateway.

Provides additional functions for:
- Deep merging multiple config files
- Environment variable interpolation
- Config file discovery
- Import/export to different formats
- Hot-reload via file watching
"""

from __future__ import annotations

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
    load_config_file,
    validate_config,
)
from src.models import GatewayConfig


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


def deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge *overlay* into *base* without mutating either.

    Rules:
    - Keys present only in *overlay* are added.
    - Scalar values in *overlay* replace those in *base*.
    - Both dicts at the same key are merged recursively.
    - Lists in *overlay* replace lists in *base* (no concatenation).

    Args:
        base: The base dictionary.
        overlay: The dictionary to merge on top.

    Returns:
        A new merged dictionary.
    """
    result = dict(base)
    for key, overlay_val in overlay.items():
        base_val = result.get(key)
        if isinstance(base_val, dict) and isinstance(overlay_val, dict):
            result[key] = deep_merge(base_val, overlay_val)
        else:
            result[key] = overlay_val
    return result


# ---------------------------------------------------------------------------
# Environment variable interpolation
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def interpolate_env_vars(
    data: Any,
    env: Optional[dict[str, str]] = None,
) -> Any:
    """Recursively replace ${VAR} and ${VAR:-default} placeholders.

    Args:
        data: The data structure to interpolate (dict, list, str, etc.).
        env: Environment mapping to use. Defaults to os.environ.

    Returns:
        A new data structure with placeholders replaced.
    """
    if env is None:
        env = dict(os.environ)

    if isinstance(data, dict):
        return {k: interpolate_env_vars(v, env) for k, v in data.items()}
    elif isinstance(data, list):
        return [interpolate_env_vars(item, env) for item in data]
    elif isinstance(data, str):
        return _interpolate_string(data, env)
    else:
        return data


def _interpolate_string(s: str, env: dict[str, str]) -> str:
    """Replace ${VAR} and ${VAR:-default} in a string."""

    def _replace(match: re.Match) -> str:
        expr = match.group(1)
        if ":-" in expr:
            var_name, default_val = expr.split(":-", 1)
            return env.get(var_name.strip(), default_val)
        else:
            var_name = expr.strip()
            if var_name in env:
                return env[var_name]
            return match.group(0)  # Leave unchanged if not found

    return _ENV_VAR_PATTERN.sub(_replace, s)


# ---------------------------------------------------------------------------
# Config file discovery
# ---------------------------------------------------------------------------

_DEFAULT_FILENAMES = ["gateway.yaml", "gateway.yml", "gateway.json"]


def discover_config_files(
    search_dirs: Optional[list] = None,
    filenames: Optional[list[str]] = None,
) -> list[Path]:
    """Discover configuration files in the given directories.

    Args:
        search_dirs: Directories to search. Defaults to [Path.cwd()].
        filenames: File names to look for. Defaults to gateway.yaml, .yml, .json.

    Returns:
        List of Path objects for found config files (in search order).
    """
    if search_dirs is None:
        search_dirs = [Path.cwd()]
    if filenames is None:
        filenames = _DEFAULT_FILENAMES

    found: list[Path] = []
    for directory in search_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for filename in filenames:
            candidate = directory / filename
            if candidate.exists():
                found.append(candidate)
    return found


# ---------------------------------------------------------------------------
# load_and_merge
# ---------------------------------------------------------------------------


def load_and_merge(
    paths: list,
    validate: bool = True,
) -> GatewayConfig:
    """Load and merge multiple configuration files.

    Files are merged in order — later files override earlier ones.

    Args:
        paths: List of file paths to load and merge.
        validate: Whether to validate the merged configuration.

    Returns:
        The merged GatewayConfig.

    Raises:
        ConfigError: If no paths given, or a file is missing/unreadable.
        ConfigValidationError: If validate=True and the result is invalid.
    """
    if not paths:
        raise ConfigError("No configuration files provided to load_and_merge")

    merged: dict[str, Any] = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        data = load_config_file(path)
        merged = deep_merge(merged, data)

    config = GatewayConfig.from_dict(merged)

    if validate:
        errors = validate_config(config)
        if errors:
            raise ConfigValidationError(errors)

    return config


# ---------------------------------------------------------------------------
# load_with_interpolation
# ---------------------------------------------------------------------------


def load_with_interpolation(
    path: Path,
    validate: bool = True,
    env: Optional[dict[str, str]] = None,
) -> GatewayConfig:
    """Load a config file and resolve ${VAR} placeholders from the environment.

    Args:
        path: Path to the config file.
        validate: Whether to validate the resulting config.
        env: Environment mapping. Defaults to os.environ.

    Returns:
        The loaded and interpolated GatewayConfig.

    Raises:
        ConfigError: If the file cannot be read.
        ConfigValidationError: If validation is enabled and fails.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    data = load_config_file(path)
    data = interpolate_env_vars(data, env=env)
    config = GatewayConfig.from_dict(data)

    if validate:
        errors = validate_config(config)
        if errors:
            raise ConfigValidationError(errors)

    return config


# ---------------------------------------------------------------------------
# load_with_defaults
# ---------------------------------------------------------------------------


def load_with_defaults(
    path: Optional[Path] = None,
    validate: bool = True,
) -> GatewayConfig:
    """Load config from *path*, falling back to defaults on any error.

    If *path* doesn't exist, or if the loaded config is invalid (when
    *validate* is True), returns the default configuration.

    Args:
        path: Optional path to a config file. If None, auto-discovers.
        validate: Whether to validate the loaded config.

    Returns:
        The loaded GatewayConfig, or a default config on failure.
    """
    from src.config import get_default_config, find_config_file

    if path is None:
        path = find_config_file()
        if path is None:
            return get_default_config()

    path = Path(path)
    if not path.exists():
        return get_default_config()

    try:
        data = load_config_file(path)
        config = GatewayConfig.from_dict(data)
        if validate:
            errors = validate_config(config)
            if errors:
                return get_default_config()
        return config
    except Exception:
        return get_default_config()


# ---------------------------------------------------------------------------
# export_config / import_config
# ---------------------------------------------------------------------------


def export_config(config: GatewayConfig, fmt: str = "yaml") -> str:
    """Export a GatewayConfig to a string in the specified format.

    Args:
        config: The configuration to export.
        fmt: Output format: 'yaml', 'yml', or 'json'.

    Returns:
        String representation of the configuration.

    Raises:
        ValueError: If the format is unsupported.
    """
    data = config.to_dict()
    fmt_lower = fmt.lower()

    if fmt_lower in ("yaml", "yml"):
        return yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    elif fmt_lower == "json":
        return json.dumps(data, indent=2) + "\n"
    else:
        raise ValueError(
            f"Unsupported export format: {fmt!r}. Use 'yaml', 'yml', or 'json'."
        )


def import_config(
    content: str,
    fmt: str = "yaml",
    validate: bool = True,
) -> GatewayConfig:
    """Import a GatewayConfig from a string in the specified format.

    Args:
        content: String content to parse.
        fmt: Format of the content: 'yaml', 'yml', or 'json'.
        validate: Whether to validate the parsed configuration.

    Returns:
        The parsed GatewayConfig.

    Raises:
        ConfigError: If parsing fails or format is unsupported.
        ConfigValidationError: If validation is enabled and fails.
    """
    fmt_lower = fmt.lower()

    if fmt_lower in ("yaml", "yml"):
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML: {e}")
    elif fmt_lower == "json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse JSON: {e}")
    else:
        raise ConfigError(
            f"Unsupported format: {fmt!r}. Use 'yaml', 'yml', or 'json'."
        )

    if not isinstance(data, dict):
        raise ConfigError(
            "Configuration must be a mapping (dict), not a list or scalar."
        )

    config = GatewayConfig.from_dict(data)

    if validate:
        errors = validate_config(config)
        if errors:
            raise ConfigValidationError(errors)

    return config


# ---------------------------------------------------------------------------
# ConfigWatcher
# ---------------------------------------------------------------------------


class ConfigWatcher:
    """Watches a config file for changes and calls a callback on reload.

    Polls the file's modification time at a configured interval.

    Args:
        path: Path to the config file to watch.
        on_change: Callback invoked with the new GatewayConfig on change.
        on_error: Optional callback invoked on reload error.
        validate: Whether to validate on reload.
        interval: Polling interval in seconds.
    """

    def __init__(
        self,
        path: Path,
        on_change: Callable[[GatewayConfig], None],
        on_error: Optional[Callable[[Exception], None]] = None,
        validate: bool = True,
        interval: float = 5.0,
    ) -> None:
        self._path = Path(path)
        self._on_change = on_change
        self._on_error = on_error
        self._validate = validate
        self._interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_mtime: Optional[float] = self._get_mtime()

    @property
    def is_running(self) -> bool:
        """Whether the watcher thread is running."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _get_mtime(self) -> Optional[float]:
        """Get the file's modification time, or None if not accessible."""
        try:
            return self._path.stat().st_mtime
        except OSError:
            return None

    def check_now(self) -> bool:
        """Check for changes immediately.

        Returns:
            True if the file changed and callback was invoked.
        """
        current_mtime = self._get_mtime()
        if current_mtime is None:
            return False
        if current_mtime == self._last_mtime:
            return False

        self._last_mtime = current_mtime
        try:
            if self._validate:
                data = load_config_file(self._path)
                config = GatewayConfig.from_dict(data)
                errors = validate_config(config)
                if errors:
                    raise ConfigValidationError(errors)
            else:
                data = load_config_file(self._path)
                config = GatewayConfig.from_dict(data)
            self._on_change(config)
            return True
        except Exception as e:
            if self._on_error:
                self._on_error(e)
            return True  # There was a change attempt, even if it errored

    def start(self) -> None:
        """Start the background watching thread (no-op if already running)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="config-watcher"
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the background watching thread."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=self._interval + 1)
            with self._lock:
                self._thread = None

    def _run(self) -> None:
        """Background thread: poll for changes."""
        while not self._stop_event.wait(timeout=self._interval):
            self.check_now()
