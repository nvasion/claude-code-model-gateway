"""Tests for CLI config and provider commands."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from src.cli import main


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config file with sample data."""
    config_data = {
        "default_provider": "openai",
        "log_level": "info",
        "timeout": 30,
        "max_retries": 3,
        "providers": {
            "openai": {
                "name": "openai",
                "display_name": "OpenAI",
                "api_base": "https://api.openai.com/v1",
                "api_key_env_var": "OPENAI_API_KEY",
                "auth_type": "bearer_token",
                "default_model": "gpt-4o",
                "enabled": True,
                "models": {
                    "gpt-4o": {
                        "name": "gpt-4o",
                        "display_name": "GPT-4o",
                        "max_tokens": 16384,
                        "supports_streaming": True,
                        "supports_tools": True,
                        "supports_vision": True,
                    }
                },
            },
            "anthropic": {
                "name": "anthropic",
                "display_name": "Anthropic",
                "api_base": "https://api.anthropic.com/v1",
                "api_key_env_var": "ANTHROPIC_API_KEY",
                "auth_type": "api_key",
                "default_model": "claude-sonnet-4-20250514",
                "enabled": True,
                "models": {
                    "claude-sonnet-4-20250514": {
                        "name": "claude-sonnet-4-20250514",
                        "display_name": "Claude Sonnet 4",
                        "max_tokens": 8192,
                        "supports_streaming": True,
                        "supports_tools": True,
                        "supports_vision": True,
                    }
                },
            },
        },
    }
    path = tmp_path / "gateway.yaml"
    with open(path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)
    return str(path)


# ---------------------------------------------------------------------------
# Config group tests
# ---------------------------------------------------------------------------


class TestConfigGroup:
    """Tests for the config command group."""

    def test_config_help(self, runner):
        """Test config --help."""
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "Manage gateway configuration" in result.output

    def test_config_init_help(self, runner):
        """Test config init --help."""
        result = runner.invoke(main, ["config", "init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.output


class TestConfigInit:
    """Tests for the config init command."""

    def test_init_creates_yaml(self, runner, tmp_path):
        """Test creating a default YAML config."""
        output = str(tmp_path / "gateway.yaml")
        result = runner.invoke(main, ["config", "init", "-o", output])
        assert result.exit_code == 0
        assert "Configuration file created" in result.output
        assert Path(output).exists()

        with open(output) as f:
            data = yaml.safe_load(f)
        assert "providers" in data
        assert data["default_provider"] == "openai"

    def test_init_creates_json(self, runner, tmp_path):
        """Test creating a default JSON config."""
        output = str(tmp_path / "gateway.json")
        result = runner.invoke(
            main, ["config", "init", "-o", output, "--format", "json"]
        )
        assert result.exit_code == 0
        assert Path(output).exists()

        with open(output) as f:
            data = json.load(f)
        assert "providers" in data

    def test_init_refuses_overwrite(self, runner, tmp_path):
        """Test that init refuses to overwrite existing file."""
        output = str(tmp_path / "gateway.yaml")
        Path(output).write_text("existing content")
        result = runner.invoke(main, ["config", "init", "-o", output])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_init_force_overwrite(self, runner, tmp_path):
        """Test that --force allows overwriting."""
        output = str(tmp_path / "gateway.yaml")
        Path(output).write_text("existing content")
        result = runner.invoke(main, ["config", "init", "-o", output, "--force"])
        assert result.exit_code == 0
        assert "Configuration file created" in result.output

    def test_init_reports_provider_count(self, runner, tmp_path):
        """Test that init reports how many providers were configured."""
        output = str(tmp_path / "gateway.yaml")
        result = runner.invoke(main, ["config", "init", "-o", output])
        assert result.exit_code == 0
        assert "providers configured" in result.output


class TestConfigShow:
    """Tests for the config show command."""

    def test_show_text_format(self, runner, config_file):
        """Test showing config in text format."""
        result = runner.invoke(main, ["config", "show", "-c", config_file])
        assert result.exit_code == 0
        assert "Gateway Configuration" in result.output
        assert "openai" in result.output.lower()

    def test_show_yaml_format(self, runner, config_file):
        """Test showing config in YAML format."""
        result = runner.invoke(
            main, ["config", "show", "-c", config_file, "--format", "yaml"]
        )
        assert result.exit_code == 0
        # Should be valid YAML
        data = yaml.safe_load(result.output)
        assert data["default_provider"] == "openai"

    def test_show_json_format(self, runner, config_file):
        """Test showing config in JSON format."""
        result = runner.invoke(
            main, ["config", "show", "-c", config_file, "--format", "json"]
        )
        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert data["default_provider"] == "openai"


class TestConfigValidate:
    """Tests for the config validate command."""

    def test_validate_valid_config(self, runner, config_file):
        """Test validating a valid configuration."""
        result = runner.invoke(main, ["config", "validate", "-c", config_file])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_invalid_config(self, runner, tmp_path):
        """Test validating an invalid configuration."""
        path = tmp_path / "bad.yaml"
        data = {
            "default_provider": "nonexistent",
            "log_level": "invalid_level",
            "timeout": -1,
        }
        with open(path, "w") as f:
            yaml.dump(data, f)

        result = runner.invoke(main, ["config", "validate", "-c", str(path)])
        assert result.exit_code != 0
        assert "failed" in result.output.lower()


class TestConfigSet:
    """Tests for the config set command."""

    def test_set_default_provider(self, runner, config_file):
        """Test setting default_provider."""
        result = runner.invoke(
            main, ["config", "set", "default_provider", "anthropic", "-c", config_file]
        )
        assert result.exit_code == 0
        assert "Set default_provider = anthropic" in result.output

        # Verify the change was persisted
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["default_provider"] == "anthropic"

    def test_set_log_level(self, runner, config_file):
        """Test setting log_level."""
        result = runner.invoke(
            main, ["config", "set", "log_level", "debug", "-c", config_file]
        )
        assert result.exit_code == 0
        assert "Set log_level = debug" in result.output

    def test_set_timeout(self, runner, config_file):
        """Test setting timeout."""
        result = runner.invoke(
            main, ["config", "set", "timeout", "60", "-c", config_file]
        )
        assert result.exit_code == 0
        assert "Set timeout = 60" in result.output

    def test_set_max_retries(self, runner, config_file):
        """Test setting max_retries."""
        result = runner.invoke(
            main, ["config", "set", "max_retries", "5", "-c", config_file]
        )
        assert result.exit_code == 0
        assert "Set max_retries = 5" in result.output

    def test_set_invalid_key(self, runner, config_file):
        """Test setting an invalid key."""
        result = runner.invoke(
            main, ["config", "set", "invalid_key", "value", "-c", config_file]
        )
        assert result.exit_code != 0
        assert "Unknown key" in result.output

    def test_set_invalid_timeout_value(self, runner, config_file):
        """Test setting timeout to non-integer."""
        result = runner.invoke(
            main, ["config", "set", "timeout", "not_a_number", "-c", config_file]
        )
        assert result.exit_code != 0
        assert "not a valid integer" in result.output


# ---------------------------------------------------------------------------
# Provider group tests
# ---------------------------------------------------------------------------


class TestProviderGroup:
    """Tests for the provider command group."""

    def test_provider_help(self, runner):
        """Test provider --help."""
        result = runner.invoke(main, ["provider", "--help"])
        assert result.exit_code == 0
        assert "Manage model providers" in result.output


class TestProviderList:
    """Tests for the provider list command."""

    def test_list_configured_providers(self, runner, config_file):
        """Test listing configured providers."""
        result = runner.invoke(main, ["provider", "list", "-c", config_file])
        assert result.exit_code == 0
        assert "openai" in result.output.lower()
        assert "anthropic" in result.output.lower()

    def test_list_builtin_providers(self, runner):
        """Test listing built-in provider templates."""
        result = runner.invoke(main, ["provider", "list", "--builtins"])
        assert result.exit_code == 0
        assert "openai" in result.output.lower()
        assert "anthropic" in result.output.lower()
        assert "azure" in result.output.lower()
        assert "google" in result.output.lower()


class TestProviderShow:
    """Tests for the provider show command."""

    def test_show_provider(self, runner, config_file):
        """Test showing a specific provider."""
        result = runner.invoke(
            main, ["provider", "show", "openai", "-c", config_file]
        )
        assert result.exit_code == 0
        assert "OpenAI" in result.output
        assert "api.openai.com" in result.output
        assert "gpt-4o" in result.output

    def test_show_nonexistent_provider(self, runner, config_file):
        """Test showing a nonexistent provider."""
        result = runner.invoke(
            main, ["provider", "show", "nonexistent", "-c", config_file]
        )
        assert result.exit_code != 0
        assert "not found" in result.output


class TestProviderAdd:
    """Tests for the provider add command."""

    def test_add_custom_provider(self, runner, config_file):
        """Test adding a custom provider."""
        result = runner.invoke(
            main,
            [
                "provider",
                "add",
                "local-llm",
                "--api-base",
                "http://localhost:8000/v1",
                "--api-key-env",
                "LOCAL_KEY",
                "--default-model",
                "llama3",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0
        assert "added successfully" in result.output

        # Verify persisted
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert "local-llm" in data["providers"]

    def test_add_duplicate_provider(self, runner, config_file):
        """Test that adding a duplicate provider fails."""
        result = runner.invoke(
            main,
            [
                "provider",
                "add",
                "openai",
                "--api-base",
                "http://example.com",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "already exists" in result.output


class TestProviderRemove:
    """Tests for the provider remove command."""

    def test_remove_provider(self, runner, config_file):
        """Test removing a provider."""
        result = runner.invoke(
            main, ["provider", "remove", "anthropic", "-c", config_file]
        )
        assert result.exit_code == 0
        assert "removed" in result.output

        # Verify persisted
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert "anthropic" not in data["providers"]

    def test_remove_nonexistent_provider(self, runner, config_file):
        """Test removing a nonexistent provider."""
        result = runner.invoke(
            main, ["provider", "remove", "nonexistent", "-c", config_file]
        )
        assert result.exit_code != 0
        assert "not found" in result.output


class TestProviderSetDefault:
    """Tests for the provider set-default command."""

    def test_set_default(self, runner, config_file):
        """Test setting default provider."""
        result = runner.invoke(
            main, ["provider", "set-default", "anthropic", "-c", config_file]
        )
        assert result.exit_code == 0
        assert "Default provider set to 'anthropic'" in result.output

        # Verify persisted
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["default_provider"] == "anthropic"

    def test_set_default_nonexistent(self, runner, config_file):
        """Test setting nonexistent provider as default."""
        result = runner.invoke(
            main, ["provider", "set-default", "nonexistent", "-c", config_file]
        )
        assert result.exit_code != 0
        assert "not found" in result.output


class TestProviderEnableDisable:
    """Tests for the provider enable/disable commands."""

    def test_disable_provider(self, runner, config_file):
        """Test disabling a provider."""
        result = runner.invoke(
            main, ["provider", "disable", "openai", "-c", config_file]
        )
        assert result.exit_code == 0
        assert "disabled" in result.output

        # Verify persisted
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["enabled"] is False

    def test_enable_provider(self, runner, config_file):
        """Test enabling a provider."""
        # First disable it
        runner.invoke(main, ["provider", "disable", "openai", "-c", config_file])
        # Then enable it
        result = runner.invoke(
            main, ["provider", "enable", "openai", "-c", config_file]
        )
        assert result.exit_code == 0
        assert "enabled" in result.output

        # Verify persisted
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["enabled"] is True

    def test_enable_nonexistent_provider(self, runner, config_file):
        """Test enabling a nonexistent provider."""
        result = runner.invoke(
            main, ["provider", "enable", "nonexistent", "-c", config_file]
        )
        assert result.exit_code != 0
        assert "not found" in result.output
