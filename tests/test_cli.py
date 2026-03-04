"""Tests for CLI commands."""

import json
import tempfile
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
def two_provider_config_file(tmp_path):
    """Write a minimal two-provider gateway config to a temp YAML file."""
    config = {
        "default_provider": "openai",
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
                    },
                    "gpt-4o-mini": {
                        "name": "gpt-4o-mini",
                        "display_name": "GPT-4o Mini",
                        "max_tokens": 16384,
                        "supports_streaming": True,
                        "supports_tools": True,
                        "supports_vision": True,
                    },
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
                    },
                },
            },
        },
    }
    config_path = tmp_path / "gateway.yaml"
    config_path.write_text(yaml.dump(config))
    return str(config_path)


def test_main_group(runner):
    """Test that main command group works."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "claude-code-model-gateway" in result.output


def test_hello_command(runner):
    """Test the hello command."""
    result = runner.invoke(main, ["hello"])
    assert result.exit_code == 0
    assert "Hello from claude-code-model-gateway" in result.output


def test_greet_default(runner):
    """Test greet command with default name."""
    result = runner.invoke(main, ["greet"])
    assert result.exit_code == 0
    assert "Hello, World!" in result.output


def test_greet_with_name(runner):
    """Test greet command with custom name."""
    result = runner.invoke(main, ["greet", "Alice"])
    assert result.exit_code == 0
    assert "Hello, Alice!" in result.output


def test_version_command(runner):
    """Test the version command."""
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    assert "claude-code-model-gateway version" in result.output


def test_version_option(runner):
    """Test the --version option."""
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "claude-code-model-gateway" in result.output


# ---------------------------------------------------------------------------
# models commands
# ---------------------------------------------------------------------------


class TestModelsCommand:
    """Tests for the ``models`` command group."""

    def test_models_group_in_main_help(self, runner):
        """The models command appears in the main help output."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "models" in result.output

    def test_models_help(self, runner):
        """The models group help is accessible."""
        result = runner.invoke(main, ["models", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output

    def test_models_list_help(self, runner):
        """models list --help works."""
        result = runner.invoke(main, ["models", "list", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "--provider" in result.output

    def test_models_show_help(self, runner):
        """models show --help works."""
        result = runner.invoke(main, ["models", "show", "--help"])
        assert result.exit_code == 0

    def test_models_list_text_output(self, runner, two_provider_config_file):
        """models list shows models from all providers in text format."""
        result = runner.invoke(
            main,
            ["models", "list", "--config-file", two_provider_config_file],
        )
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "gpt-4o-mini" in result.output
        assert "claude-sonnet-4-20250514" in result.output

    def test_models_list_shows_provider_names(self, runner, two_provider_config_file):
        """models list shows provider names in the output."""
        result = runner.invoke(
            main,
            ["models", "list", "--config-file", two_provider_config_file],
        )
        assert result.exit_code == 0
        assert "OpenAI" in result.output or "openai" in result.output
        assert "Anthropic" in result.output or "anthropic" in result.output

    def test_models_list_json_output(self, runner, two_provider_config_file):
        """models list --format json returns valid JSON with model list."""
        result = runner.invoke(
            main,
            ["models", "list", "--config-file", two_provider_config_file, "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "models" in data
        assert "total" in data
        model_ids = [m["id"] for m in data["models"]]
        assert "gpt-4o" in model_ids
        assert "gpt-4o-mini" in model_ids
        assert "claude-sonnet-4-20250514" in model_ids

    def test_models_list_json_model_fields(self, runner, two_provider_config_file):
        """Each JSON model entry contains required fields."""
        result = runner.invoke(
            main,
            ["models", "list", "--config-file", two_provider_config_file, "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        for model in data["models"]:
            assert "id" in model
            assert "display_name" in model
            assert "provider" in model
            assert "max_tokens" in model

    def test_models_list_provider_filter(self, runner, two_provider_config_file):
        """models list --provider filters to a single provider's models."""
        result = runner.invoke(
            main,
            [
                "models", "list",
                "--config-file", two_provider_config_file,
                "--provider", "openai",
                "--format", "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        providers = {m["provider"] for m in data["models"]}
        assert providers == {"openai"}
        assert data["total"] == 2  # gpt-4o and gpt-4o-mini

    def test_models_list_invalid_provider_filter(self, runner, two_provider_config_file):
        """models list with unknown --provider exits with error."""
        result = runner.invoke(
            main,
            [
                "models", "list",
                "--config-file", two_provider_config_file,
                "--provider", "nonexistent-provider",
            ],
        )
        assert result.exit_code != 0

    def test_models_show_known_model(self, runner, two_provider_config_file):
        """models show displays details for a known model."""
        result = runner.invoke(
            main,
            ["models", "show", "gpt-4o", "--config-file", two_provider_config_file],
        )
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "OpenAI" in result.output or "openai" in result.output
        assert "16,384" in result.output or "16384" in result.output

    def test_models_show_anthropic_model(self, runner, two_provider_config_file):
        """models show works for a model from a non-default provider."""
        result = runner.invoke(
            main,
            [
                "models", "show",
                "claude-sonnet-4-20250514",
                "--config-file", two_provider_config_file,
            ],
        )
        assert result.exit_code == 0
        assert "claude-sonnet-4-20250514" in result.output
        assert "Anthropic" in result.output or "anthropic" in result.output

    def test_models_show_unknown_model_exits_error(self, runner, two_provider_config_file):
        """models show with an unknown model ID exits with a non-zero code."""
        result = runner.invoke(
            main,
            [
                "models", "show",
                "does-not-exist-12345",
                "--config-file", two_provider_config_file,
            ],
        )
        assert result.exit_code != 0

    def test_models_list_count(self, runner, two_provider_config_file):
        """models list --format json total matches expected model count."""
        result = runner.invoke(
            main,
            ["models", "list", "--config-file", two_provider_config_file, "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 3  # gpt-4o, gpt-4o-mini, claude-sonnet-4-20250514


# ---------------------------------------------------------------------------
# third-party-models command
# ---------------------------------------------------------------------------


class TestThirdPartyModelsCommand:
    """Tests for the installable ``third-party-models`` command."""

    def test_command_in_main_help(self, runner):
        """third-party-models appears in the main help."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "third-party-models" in result.output

    def test_help_text(self, runner):
        """third-party-models --help is accessible."""
        result = runner.invoke(main, ["third-party-models", "--help"])
        assert result.exit_code == 0
        assert "--config-file" in result.output
        assert "--host" in result.output
        assert "--port" in result.output

    def test_no_config_exits_zero_with_hint(self, runner, tmp_path):
        """Without a config file the command exits 0 and prints a hint."""
        # Run from a temp dir with no gateway.yaml present
        result = runner.invoke(
            main,
            ["third-party-models"],
            env={"GATEWAY_CONFIG": ""},
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "config init" in result.output

    def test_shows_models_from_config(self, runner, two_provider_config_file):
        """Command lists all models from the gateway config."""
        result = runner.invoke(
            main,
            ["third-party-models", "--config-file", two_provider_config_file],
        )
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "gpt-4o-mini" in result.output
        assert "claude-sonnet-4-20250514" in result.output

    def test_shows_provider_names(self, runner, two_provider_config_file):
        """Command includes provider display names in the output."""
        result = runner.invoke(
            main,
            ["third-party-models", "--config-file", two_provider_config_file],
        )
        assert result.exit_code == 0
        assert "OpenAI" in result.output or "openai" in result.output
        assert "Anthropic" in result.output or "anthropic" in result.output

    def test_shows_claude_code_instructions(self, runner, two_provider_config_file):
        """Command prints Claude Code integration instructions."""
        result = runner.invoke(
            main,
            ["third-party-models", "--config-file", two_provider_config_file],
        )
        assert result.exit_code == 0
        assert "ANTHROPIC_BASE_URL" in result.output
        assert "/model" in result.output

    def test_custom_port_in_url_hint(self, runner, two_provider_config_file):
        """Custom --port is reflected in the gateway URL hint."""
        result = runner.invoke(
            main,
            [
                "third-party-models",
                "--config-file", two_provider_config_file,
                "--port", "9090",
            ],
        )
        assert result.exit_code == 0
        assert "9090" in result.output

    def test_gateway_command_hint_shown(self, runner, two_provider_config_file):
        """Output includes the gateway start command."""
        result = runner.invoke(
            main,
            ["third-party-models", "--config-file", two_provider_config_file],
        )
        assert result.exit_code == 0
        assert "gateway" in result.output
