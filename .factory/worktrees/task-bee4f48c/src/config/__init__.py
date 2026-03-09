"""Configuration loading, saving, and validation for the model gateway."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from src.models import GatewayConfig, ProviderConfig

# Default config file locations (searched in order)
DEFAULT_CONFIG_PATHS = [
    Path("gateway.yaml"),
    Path("gateway.yml"),
    Path("gateway.json"),
    Path.home() / ".config" / "claude-code-model-gateway" / "config.yaml",
]

# Environment variable to override config path
CONFIG_ENV_VAR = "GATEWAY_CONFIG"


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Configuration validation failed: {'; '.join(errors)}")


def find_config_file() -> Optional[Path]:
    """Find the configuration file.

    Searches in order:
    1. GATEWAY_CONFIG environment variable
    2. Default config file paths (gateway.yaml, gateway.yml, gateway.json)

    Returns:
        Path to the config file if found, None otherwise.
    """
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        return None

    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            return path

    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ConfigError: If the file cannot be read or parsed.
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(f"Config file {path} must contain a YAML mapping")
        return data
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config {path}: {e}")
    except OSError as e:
        raise ConfigError(f"Failed to read config file {path}: {e}")


def _load_json(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content as a dictionary.

    Raises:
        ConfigError: If the file cannot be read or parsed.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ConfigError(f"Config file {path} must contain a JSON object")
        return data
    except json.JSONDecodeError as e:
        raise ConfigError(f"Failed to parse JSON config {path}: {e}")
    except OSError as e:
        raise ConfigError(f"Failed to read config file {path}: {e}")


def load_config_file(path: Path) -> dict[str, Any]:
    """Load configuration from a file (YAML or JSON).

    Args:
        path: Path to the configuration file.

    Returns:
        Configuration data as a dictionary.

    Raises:
        ConfigError: If the file format is unsupported or cannot be read.
    """
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    elif suffix == ".json":
        return _load_json(path)
    else:
        raise ConfigError(
            f"Unsupported config file format: {suffix}. " "Use .yaml, .yml, or .json"
        )


def save_config_file(config: GatewayConfig, path: Path) -> None:
    """Save configuration to a file (YAML or JSON).

    Args:
        config: The gateway configuration to save.
        path: Path to write the configuration file.

    Raises:
        ConfigError: If the file format is unsupported or cannot be written.
    """
    data = config.to_dict()
    suffix = path.suffix.lower()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        if suffix in (".yaml", ".yml"):
            with open(path, "w") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
        elif suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
        else:
            raise ConfigError(
                f"Unsupported config file format: {suffix}. "
                "Use .yaml, .yml, or .json"
            )
    except OSError as e:
        raise ConfigError(f"Failed to write config file {path}: {e}")


def validate_config(config: GatewayConfig) -> list[str]:
    """Validate a gateway configuration.

    Checks for:
    - Default provider exists in providers list
    - Each provider has required fields
    - Default model exists in provider's model list
    - Valid log level
    - Positive timeout and retry values

    Args:
        config: The gateway configuration to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    # Validate default provider
    if config.default_provider and config.default_provider not in config.providers:
        errors.append(
            f"Default provider '{config.default_provider}' "
            "is not in the providers list"
        )

    # Validate log level
    valid_log_levels = {"debug", "info", "warning", "error", "critical"}
    if config.log_level.lower() not in valid_log_levels:
        errors.append(
            f"Invalid log level '{config.log_level}'. "
            f"Must be one of: {', '.join(sorted(valid_log_levels))}"
        )

    # Validate timeout
    if config.timeout <= 0:
        errors.append(f"Timeout must be positive, got {config.timeout}")

    # Validate max_retries
    if config.max_retries < 0:
        errors.append(f"Max retries must be non-negative, got {config.max_retries}")

    # Validate model_aliases: alias keys must not collide with real provider
    # model names because the alias lookup takes priority over model routing
    # and a collision would make the original model unreachable.
    all_provider_models = {
        model_name
        for provider in config.get_enabled_providers().values()
        for model_name in provider.models
    }
    for alias_name in config.model_aliases:
        if alias_name in all_provider_models:
            errors.append(
                f"model_aliases: alias key '{alias_name}' collides with a "
                "real provider model name; this makes the provider model "
                "unreachable — rename the alias or remove the model entry"
            )

    # Validate each provider
    for name, provider in config.providers.items():
        prefix = f"Provider '{name}'"

        if not provider.api_base:
            errors.append(f"{prefix}: missing api_base URL")

        if provider.default_model and provider.models:
            if provider.default_model not in provider.models:
                errors.append(
                    f"{prefix}: default model '{provider.default_model}' "
                    "is not in the models list"
                )

        # Validate model configs
        for model_name, model in provider.models.items():
            model_prefix = f"{prefix}, model '{model_name}'"
            if model.max_tokens <= 0:
                errors.append(
                    f"{model_prefix}: max_tokens must be positive, "
                    f"got {model.max_tokens}"
                )

    return errors


def load_config(
    path: Optional[Path] = None,
    validate: bool = True,
    use_cache: bool = True,
) -> GatewayConfig:
    """Load and optionally validate the gateway configuration.

    When *use_cache* is ``True`` (the default), previously loaded configs
    are returned from an in-memory cache keyed on the resolved file path.
    The cache has a short TTL so file-system changes are picked up quickly.

    Args:
        path: Path to config file. If None, searches default locations.
        validate: Whether to validate the configuration after loading.
        use_cache: Whether to use the in-memory config cache.

    Returns:
        The loaded gateway configuration.

    Raises:
        ConfigError: If no config file is found or cannot be loaded.
        ConfigValidationError: If validation is enabled and fails.
    """
    if path is None:
        path = find_config_file()
        if path is None:
            # Return empty config if no file found
            return GatewayConfig()

    # Build a cache key from the resolved path + validation flag
    cache_key = f"config:{path.resolve()}:validate={validate}"

    if use_cache:
        from src.cache import get_config_cache

        cached_value = get_config_cache().get(cache_key)
        if cached_value is not None:
            return cached_value

    data = load_config_file(path)
    config = GatewayConfig.from_dict(data)

    if validate:
        errors = validate_config(config)
        if errors:
            raise ConfigValidationError(errors)

    if use_cache:
        from src.cache import get_config_cache

        get_config_cache().set(cache_key, config)

    return config


def merge_env_overrides(config: GatewayConfig) -> GatewayConfig:
    """Apply environment variable overrides to configuration.

    Supported environment variables:
    - GATEWAY_DEFAULT_PROVIDER: Override default provider
    - GATEWAY_LOG_LEVEL: Override log level
    - GATEWAY_TIMEOUT: Override timeout
    - GATEWAY_MAX_RETRIES: Override max retries

    Args:
        config: The base configuration to apply overrides to.

    Returns:
        The configuration with environment overrides applied.
    """
    default_provider = os.environ.get("GATEWAY_DEFAULT_PROVIDER")
    if default_provider:
        config.default_provider = default_provider

    log_level = os.environ.get("GATEWAY_LOG_LEVEL")
    if log_level:
        config.log_level = log_level

    timeout = os.environ.get("GATEWAY_TIMEOUT")
    if timeout:
        try:
            config.timeout = int(timeout)
        except ValueError:
            pass

    max_retries = os.environ.get("GATEWAY_MAX_RETRIES")
    if max_retries:
        try:
            config.max_retries = int(max_retries)
        except ValueError:
            pass

    return config


def get_default_config() -> GatewayConfig:
    """Create a default configuration with common providers pre-configured.

    Returns:
        A GatewayConfig with standard provider templates.
    """
    from src.providers import get_builtin_providers

    providers = get_builtin_providers()
    config = GatewayConfig(
        default_provider="openai",
        providers=providers,
        log_level="info",
        timeout=30,
        max_retries=3,
    )
    return config
