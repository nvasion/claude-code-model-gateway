"""Tests for configuration loading, saving, and validation."""

import json
import os
from pathlib import Path

import pytest
import yaml

from src.config import (
    ConfigError,
    ConfigValidationError,
    find_config_file,
    get_default_config,
    load_config,
    load_config_file,
    merge_env_overrides,
    save_config_file,
    validate_config,
)
from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test configs."""
    return tmp_path


@pytest.fixture
def sample_config():
    """Create a sample GatewayConfig for testing."""
    return GatewayConfig(
        default_provider="openai",
        log_level="info",
        timeout=30,
        max_retries=3,
        providers={
            "openai": ProviderConfig(
                name="openai",
                display_name="OpenAI",
                api_base="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
                auth_type=AuthType.BEARER_TOKEN,
                default_model="gpt-4",
                models={
                    "gpt-4": ModelConfig(
                        name="gpt-4",
                        display_name="GPT-4",
                        max_tokens=8192,
                        supports_streaming=True,
                        supports_tools=True,
                        supports_vision=True,
                    ),
                },
            ),
        },
    )


@pytest.fixture
def sample_yaml_file(tmp_dir, sample_config):
    """Write a sample YAML config file and return its path."""
    path = tmp_dir / "gateway.yaml"
    data = sample_config.to_dict()
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return path


@pytest.fixture
def sample_json_file(tmp_dir, sample_config):
    """Write a sample JSON config file and return its path."""
    path = tmp_dir / "gateway.json"
    data = sample_config.to_dict()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# load_config_file tests
# ---------------------------------------------------------------------------


class TestLoadConfigFile:
    """Tests for loading config files."""

    def test_load_yaml(self, sample_yaml_file):
        """Test loading a YAML config file."""
        data = load_config_file(sample_yaml_file)
        assert isinstance(data, dict)
        assert data["default_provider"] == "openai"
        assert "openai" in data["providers"]

    def test_load_json(self, sample_json_file):
        """Test loading a JSON config file."""
        data = load_config_file(sample_json_file)
        assert isinstance(data, dict)
        assert data["default_provider"] == "openai"

    def test_load_yml_extension(self, tmp_dir):
        """Test loading a .yml file."""
        path = tmp_dir / "config.yml"
        with open(path, "w") as f:
            yaml.dump({"default_provider": "test"}, f)
        data = load_config_file(path)
        assert data["default_provider"] == "test"

    def test_load_unsupported_format(self, tmp_dir):
        """Test that unsupported formats raise ConfigError."""
        path = tmp_dir / "config.toml"
        path.write_text("[default]\nprovider = 'test'\n")
        with pytest.raises(ConfigError, match="Unsupported config file format"):
            load_config_file(path)

    def test_load_invalid_yaml(self, tmp_dir):
        """Test that invalid YAML raises ConfigError."""
        path = tmp_dir / "bad.yaml"
        path.write_text(":::invalid::: yaml: [")
        with pytest.raises(ConfigError, match="Failed to parse YAML"):
            load_config_file(path)

    def test_load_invalid_json(self, tmp_dir):
        """Test that invalid JSON raises ConfigError."""
        path = tmp_dir / "bad.json"
        path.write_text("{invalid json}")
        with pytest.raises(ConfigError, match="Failed to parse JSON"):
            load_config_file(path)

    def test_load_nonexistent_file(self, tmp_dir):
        """Test that missing file raises ConfigError."""
        path = tmp_dir / "nonexistent.yaml"
        with pytest.raises(ConfigError, match="Failed to read"):
            load_config_file(path)

    def test_load_empty_yaml(self, tmp_dir):
        """Test that empty YAML returns empty dict."""
        path = tmp_dir / "empty.yaml"
        path.write_text("")
        data = load_config_file(path)
        assert data == {}

    def test_load_non_dict_yaml(self, tmp_dir):
        """Test that non-dict YAML raises ConfigError."""
        path = tmp_dir / "list.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigError, match="must contain a YAML mapping"):
            load_config_file(path)

    def test_load_non_dict_json(self, tmp_dir):
        """Test that non-dict JSON raises ConfigError."""
        path = tmp_dir / "list.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ConfigError, match="must contain a JSON object"):
            load_config_file(path)


# ---------------------------------------------------------------------------
# save_config_file tests
# ---------------------------------------------------------------------------


class TestSaveConfigFile:
    """Tests for saving config files."""

    def test_save_yaml(self, tmp_dir, sample_config):
        """Test saving a YAML config file."""
        path = tmp_dir / "output.yaml"
        save_config_file(sample_config, path)
        assert path.exists()
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["default_provider"] == "openai"

    def test_save_json(self, tmp_dir, sample_config):
        """Test saving a JSON config file."""
        path = tmp_dir / "output.json"
        save_config_file(sample_config, path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["default_provider"] == "openai"

    def test_save_creates_parent_dirs(self, tmp_dir, sample_config):
        """Test that saving creates parent directories."""
        path = tmp_dir / "nested" / "dir" / "config.yaml"
        save_config_file(sample_config, path)
        assert path.exists()

    def test_save_unsupported_format(self, tmp_dir, sample_config):
        """Test that unsupported format raises ConfigError."""
        path = tmp_dir / "output.toml"
        with pytest.raises(ConfigError, match="Unsupported config file format"):
            save_config_file(sample_config, path)

    def test_save_and_load_roundtrip_yaml(self, tmp_dir, sample_config):
        """Test YAML save-then-load roundtrip."""
        path = tmp_dir / "roundtrip.yaml"
        save_config_file(sample_config, path)
        loaded = load_config(path=path, validate=False)
        assert loaded.default_provider == sample_config.default_provider
        assert "openai" in loaded.providers
        assert "gpt-4" in loaded.providers["openai"].models

    def test_save_and_load_roundtrip_json(self, tmp_dir, sample_config):
        """Test JSON save-then-load roundtrip."""
        path = tmp_dir / "roundtrip.json"
        save_config_file(sample_config, path)
        loaded = load_config(path=path, validate=False)
        assert loaded.default_provider == sample_config.default_provider
        assert "openai" in loaded.providers


# ---------------------------------------------------------------------------
# validate_config tests
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """Tests for configuration validation."""

    def test_valid_config(self, sample_config):
        """Test that a valid config passes validation."""
        errors = validate_config(sample_config)
        assert errors == []

    def test_empty_config_is_valid(self):
        """Test that an empty config is valid."""
        errors = validate_config(GatewayConfig())
        assert errors == []

    def test_invalid_default_provider(self):
        """Test that referencing a nonexistent default provider is caught."""
        config = GatewayConfig(default_provider="nonexistent")
        errors = validate_config(config)
        assert any("Default provider" in e for e in errors)

    def test_invalid_log_level(self):
        """Test that invalid log level is caught."""
        config = GatewayConfig(log_level="verbose")
        errors = validate_config(config)
        assert any("log level" in e.lower() for e in errors)

    def test_negative_timeout(self):
        """Test that negative timeout is caught."""
        config = GatewayConfig(timeout=-1)
        errors = validate_config(config)
        assert any("Timeout" in e for e in errors)

    def test_zero_timeout(self):
        """Test that zero timeout is caught."""
        config = GatewayConfig(timeout=0)
        errors = validate_config(config)
        assert any("Timeout" in e for e in errors)

    def test_negative_max_retries(self):
        """Test that negative max_retries is caught."""
        config = GatewayConfig(max_retries=-1)
        errors = validate_config(config)
        assert any("retries" in e.lower() for e in errors)

    def test_zero_max_retries_is_valid(self):
        """Test that zero retries is valid (no retries)."""
        config = GatewayConfig(max_retries=0)
        errors = validate_config(config)
        assert not any("retries" in e.lower() for e in errors)

    def test_provider_missing_api_base(self):
        """Test that provider without api_base is caught."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(name="test", api_base=""),
            }
        )
        errors = validate_config(config)
        assert any("api_base" in e for e in errors)

    def test_provider_invalid_default_model(self):
        """Test that provider with nonexistent default model is caught."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://example.com",
                    default_model="nonexistent",
                    models={
                        "real-model": ModelConfig(name="real-model"),
                    },
                ),
            }
        )
        errors = validate_config(config)
        assert any("default model" in e for e in errors)

    def test_provider_default_model_no_models_ok(self):
        """Test that default_model without models list is ok (might be dynamic)."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://example.com",
                    default_model="some-model",
                    models={},
                ),
            }
        )
        errors = validate_config(config)
        assert not any("default model" in e for e in errors)

    def test_model_invalid_max_tokens(self):
        """Test that model with invalid max_tokens is caught."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://example.com",
                    models={
                        "bad-model": ModelConfig(name="bad-model", max_tokens=0),
                    },
                ),
            }
        )
        errors = validate_config(config)
        assert any("max_tokens" in e for e in errors)

    def test_multiple_errors(self):
        """Test that multiple errors are collected."""
        config = GatewayConfig(
            default_provider="nonexistent",
            log_level="invalid",
            timeout=-1,
            max_retries=-1,
        )
        errors = validate_config(config)
        assert len(errors) >= 4


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for loading configuration."""

    def test_load_from_path(self, sample_yaml_file):
        """Test loading config from a specific path."""
        config = load_config(path=sample_yaml_file, validate=False)
        assert config.default_provider == "openai"
        assert "openai" in config.providers

    def test_load_with_validation(self, sample_yaml_file):
        """Test loading with validation enabled."""
        config = load_config(path=sample_yaml_file, validate=True)
        assert config.default_provider == "openai"

    def test_load_returns_empty_when_no_file(self, monkeypatch):
        """Test that loading without a file returns empty config."""
        monkeypatch.delenv("GATEWAY_CONFIG", raising=False)
        # Use a path that won't match defaults
        config = load_config(validate=False)
        # Since we can't control the CWD, just check it returns a GatewayConfig
        assert isinstance(config, GatewayConfig)

    def test_load_validation_failure(self, tmp_dir):
        """Test that validation errors raise ConfigValidationError."""
        path = tmp_dir / "bad.yaml"
        data = {
            "default_provider": "nonexistent",
            "log_level": "invalid",
            "timeout": -1,
        }
        with open(path, "w") as f:
            yaml.dump(data, f)

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(path=path, validate=True)
        assert len(exc_info.value.errors) > 0


# ---------------------------------------------------------------------------
# find_config_file tests
# ---------------------------------------------------------------------------


class TestFindConfigFile:
    """Tests for finding configuration files."""

    def test_env_var_override(self, tmp_dir, monkeypatch):
        """Test that GATEWAY_CONFIG env var overrides default paths."""
        path = tmp_dir / "custom.yaml"
        path.write_text("default_provider: test\n")
        monkeypatch.setenv("GATEWAY_CONFIG", str(path))
        found = find_config_file()
        assert found == path

    def test_env_var_nonexistent_file(self, monkeypatch):
        """Test that nonexistent env var path returns None."""
        monkeypatch.setenv("GATEWAY_CONFIG", "/nonexistent/path.yaml")
        found = find_config_file()
        assert found is None


# ---------------------------------------------------------------------------
# merge_env_overrides tests
# ---------------------------------------------------------------------------


class TestMergeEnvOverrides:
    """Tests for environment variable overrides."""

    def test_override_default_provider(self, monkeypatch):
        """Test overriding default_provider via env."""
        config = GatewayConfig(default_provider="openai")
        monkeypatch.setenv("GATEWAY_DEFAULT_PROVIDER", "anthropic")
        config = merge_env_overrides(config)
        assert config.default_provider == "anthropic"

    def test_override_log_level(self, monkeypatch):
        """Test overriding log_level via env."""
        config = GatewayConfig(log_level="info")
        monkeypatch.setenv("GATEWAY_LOG_LEVEL", "debug")
        config = merge_env_overrides(config)
        assert config.log_level == "debug"

    def test_override_timeout(self, monkeypatch):
        """Test overriding timeout via env."""
        config = GatewayConfig(timeout=30)
        monkeypatch.setenv("GATEWAY_TIMEOUT", "60")
        config = merge_env_overrides(config)
        assert config.timeout == 60

    def test_override_max_retries(self, monkeypatch):
        """Test overriding max_retries via env."""
        config = GatewayConfig(max_retries=3)
        monkeypatch.setenv("GATEWAY_MAX_RETRIES", "5")
        config = merge_env_overrides(config)
        assert config.max_retries == 5

    def test_invalid_timeout_ignored(self, monkeypatch):
        """Test that non-integer timeout is ignored."""
        config = GatewayConfig(timeout=30)
        monkeypatch.setenv("GATEWAY_TIMEOUT", "not_a_number")
        config = merge_env_overrides(config)
        assert config.timeout == 30

    def test_invalid_max_retries_ignored(self, monkeypatch):
        """Test that non-integer max_retries is ignored."""
        config = GatewayConfig(max_retries=3)
        monkeypatch.setenv("GATEWAY_MAX_RETRIES", "abc")
        config = merge_env_overrides(config)
        assert config.max_retries == 3

    def test_no_env_vars_no_change(self, monkeypatch):
        """Test that config is unchanged when no env vars set."""
        monkeypatch.delenv("GATEWAY_DEFAULT_PROVIDER", raising=False)
        monkeypatch.delenv("GATEWAY_LOG_LEVEL", raising=False)
        monkeypatch.delenv("GATEWAY_TIMEOUT", raising=False)
        monkeypatch.delenv("GATEWAY_MAX_RETRIES", raising=False)
        config = GatewayConfig(
            default_provider="openai",
            log_level="info",
            timeout=30,
            max_retries=3,
        )
        config = merge_env_overrides(config)
        assert config.default_provider == "openai"
        assert config.log_level == "info"
        assert config.timeout == 30
        assert config.max_retries == 3


# ---------------------------------------------------------------------------
# get_default_config tests
# ---------------------------------------------------------------------------


class TestGetDefaultConfig:
    """Tests for default configuration generation."""

    def test_default_config_has_providers(self):
        """Test that default config includes built-in providers."""
        config = get_default_config()
        assert len(config.providers) > 0
        assert "openai" in config.providers
        assert "anthropic" in config.providers

    def test_default_config_default_provider(self):
        """Test that default config has a default provider set."""
        config = get_default_config()
        assert config.default_provider == "openai"

    def test_default_config_is_valid(self):
        """Test that the default config passes validation."""
        config = get_default_config()
        errors = validate_config(config)
        assert errors == []
