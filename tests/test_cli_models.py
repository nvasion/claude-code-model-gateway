"""Tests for CLI provider model and provider update commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from src.cli import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config file with two providers and several models."""
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
    path = tmp_path / "gateway.yaml"
    with open(path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)
    return str(path)


# ---------------------------------------------------------------------------
# provider update
# ---------------------------------------------------------------------------


class TestProviderUpdate:
    """Tests for 'provider update' command."""

    def test_update_help(self, runner):
        """provider update --help exits cleanly."""
        result = runner.invoke(main, ["provider", "update", "--help"])
        assert result.exit_code == 0
        assert "update" in result.output.lower()

    def test_update_api_base(self, runner, config_file):
        """provider update --api-base changes the api_base."""
        result = runner.invoke(
            main,
            [
                "provider",
                "update",
                "openai",
                "--api-base",
                "https://proxy.example.com/v1",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0
        assert "api_base" in result.output

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["api_base"] == "https://proxy.example.com/v1"

    def test_update_display_name(self, runner, config_file):
        """provider update --display-name changes the display_name."""
        result = runner.invoke(
            main,
            [
                "provider",
                "update",
                "openai",
                "--display-name",
                "My OpenAI",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["display_name"] == "My OpenAI"

    def test_update_default_model(self, runner, config_file):
        """provider update --default-model changes the default_model."""
        result = runner.invoke(
            main,
            [
                "provider",
                "update",
                "openai",
                "--default-model",
                "gpt-4o-mini",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["default_model"] == "gpt-4o-mini"

    def test_update_api_key_env(self, runner, config_file):
        """provider update --api-key-env changes api_key_env_var."""
        result = runner.invoke(
            main,
            [
                "provider",
                "update",
                "openai",
                "--api-key-env",
                "MY_OPENAI_KEY",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["api_key_env_var"] == "MY_OPENAI_KEY"

    def test_update_auth_type(self, runner, config_file):
        """provider update --auth-type changes auth_type."""
        result = runner.invoke(
            main,
            [
                "provider",
                "update",
                "openai",
                "--auth-type",
                "api_key",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["auth_type"] == "api_key"

    def test_update_multiple_fields(self, runner, config_file):
        """provider update can change several fields at once."""
        result = runner.invoke(
            main,
            [
                "provider",
                "update",
                "openai",
                "--api-base",
                "https://proxy.example.com/v1",
                "--display-name",
                "Proxied OpenAI",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["api_base"] == "https://proxy.example.com/v1"
        assert data["providers"]["openai"]["display_name"] == "Proxied OpenAI"

    def test_update_nonexistent_provider(self, runner, config_file):
        """provider update fails for an unknown provider."""
        result = runner.invoke(
            main,
            [
                "provider",
                "update",
                "nonexistent",
                "--api-base",
                "http://x.com",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_update_no_options_produces_message(self, runner, config_file):
        """provider update with no options prints a message and exits 0."""
        result = runner.invoke(
            main,
            ["provider", "update", "openai", "-c", config_file],
        )
        assert result.exit_code == 0
        assert "No changes" in result.output

    def test_update_invalid_auth_type(self, runner, config_file):
        """provider update rejects invalid auth_type values."""
        result = runner.invoke(
            main,
            [
                "provider",
                "update",
                "openai",
                "--auth-type",
                "invalid_value",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# provider model list
# ---------------------------------------------------------------------------


class TestProviderModelList:
    """Tests for 'provider model list' command."""

    def test_list_models_text(self, runner, config_file):
        """provider model list shows models in text format."""
        result = runner.invoke(
            main, ["provider", "model", "list", "openai", "-c", config_file]
        )
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "gpt-4o-mini" in result.output

    def test_list_models_json(self, runner, config_file):
        """provider model list --format json returns valid JSON."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "list",
                "openai",
                "-c",
                config_file,
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["provider"] == "openai"
        model_names = [m["name"] for m in data["models"]]
        assert "gpt-4o" in model_names
        assert "gpt-4o-mini" in model_names

    def test_list_models_shows_default_marker(self, runner, config_file):
        """provider model list marks the default model."""
        result = runner.invoke(
            main, ["provider", "model", "list", "openai", "-c", config_file]
        )
        assert result.exit_code == 0
        assert "(default)" in result.output

    def test_list_models_nonexistent_provider(self, runner, config_file):
        """provider model list fails for unknown provider."""
        result = runner.invoke(
            main, ["provider", "model", "list", "nonexistent", "-c", config_file]
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_list_models_empty_provider(self, runner, tmp_path):
        """provider model list reports no models for empty provider."""
        path = tmp_path / "gateway.yaml"
        data = {
            "providers": {
                "empty": {
                    "name": "empty",
                    "api_base": "http://example.com",
                }
            }
        }
        with open(path, "w") as f:
            yaml.dump(data, f)

        result = runner.invoke(
            main, ["provider", "model", "list", "empty", "-c", str(path)]
        )
        assert result.exit_code == 0
        assert "No models" in result.output


# ---------------------------------------------------------------------------
# provider model add
# ---------------------------------------------------------------------------


class TestProviderModelAdd:
    """Tests for 'provider model add' command."""

    def test_add_basic_model(self, runner, config_file):
        """provider model add creates a model with defaults."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "add",
                "openai",
                "gpt-5",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0
        assert "added" in result.output.lower()

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert "gpt-5" in data["providers"]["openai"]["models"]

    def test_add_model_with_all_options(self, runner, config_file):
        """provider model add accepts all model configuration options."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "add",
                "openai",
                "o4-mini",
                "--display-name",
                "o4 Mini",
                "--max-tokens",
                "65536",
                "--tools",
                "--vision",
                "--streaming",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0

        with open(config_file) as f:
            data = yaml.safe_load(f)
        model = data["providers"]["openai"]["models"]["o4-mini"]
        assert model["display_name"] == "o4 Mini"
        assert model["max_tokens"] == 65536
        assert model["supports_tools"] is True
        assert model["supports_vision"] is True

    def test_add_model_set_default(self, runner, config_file):
        """provider model add --set-default makes the model the provider default."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "add",
                "openai",
                "gpt-5",
                "--set-default",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["default_model"] == "gpt-5"

    def test_add_duplicate_model_fails(self, runner, config_file):
        """provider model add fails when model already exists."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "add",
                "openai",
                "gpt-4o",  # already exists
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "already exists" in result.output.lower()

    def test_add_model_nonexistent_provider(self, runner, config_file):
        """provider model add fails for an unknown provider."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "add",
                "nonexistent",
                "my-model",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_add_model_no_streaming(self, runner, config_file):
        """provider model add --no-streaming creates a non-streaming model."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "add",
                "openai",
                "batch-model",
                "--no-streaming",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["models"]["batch-model"]["supports_streaming"] is False


# ---------------------------------------------------------------------------
# provider model remove
# ---------------------------------------------------------------------------


class TestProviderModelRemove:
    """Tests for 'provider model remove' command."""

    def test_remove_model(self, runner, config_file):
        """provider model remove deletes the model."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "remove",
                "openai",
                "gpt-4o-mini",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0
        assert "removed" in result.output.lower()

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert "gpt-4o-mini" not in data["providers"]["openai"]["models"]

    def test_remove_nonexistent_model(self, runner, config_file):
        """provider model remove fails for a model that doesn't exist."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "remove",
                "openai",
                "nonexistent-model",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_remove_nonexistent_provider(self, runner, config_file):
        """provider model remove fails for unknown provider."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "remove",
                "nonexistent",
                "gpt-4o",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_remove_default_model_updates_default(self, runner, config_file):
        """Removing the default model updates default_model to another model."""
        # gpt-4o is the default
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "remove",
                "openai",
                "gpt-4o",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0

        with open(config_file) as f:
            data = yaml.safe_load(f)
        # The remaining model should be gpt-4o-mini
        assert data["providers"]["openai"]["default_model"] != "gpt-4o"


# ---------------------------------------------------------------------------
# provider model set-default
# ---------------------------------------------------------------------------


class TestProviderModelSetDefault:
    """Tests for 'provider model set-default' command."""

    def test_set_default_model(self, runner, config_file):
        """provider model set-default changes the default_model."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "set-default",
                "openai",
                "gpt-4o-mini",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0
        assert "gpt-4o-mini" in result.output

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["default_model"] == "gpt-4o-mini"

    def test_set_default_nonexistent_model(self, runner, config_file):
        """provider model set-default fails for unknown model."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "set-default",
                "openai",
                "nonexistent-model",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_set_default_nonexistent_provider(self, runner, config_file):
        """provider model set-default fails for unknown provider."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "set-default",
                "nonexistent",
                "gpt-4o",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# provider model update
# ---------------------------------------------------------------------------


class TestProviderModelUpdate:
    """Tests for 'provider model update' command."""

    def test_update_max_tokens(self, runner, config_file):
        """provider model update --max-tokens changes max_tokens."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "update",
                "openai",
                "gpt-4o",
                "--max-tokens",
                "8192",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0
        assert "max_tokens = 8192" in result.output

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["models"]["gpt-4o"]["max_tokens"] == 8192

    def test_update_display_name(self, runner, config_file):
        """provider model update --display-name changes display_name."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "update",
                "openai",
                "gpt-4o",
                "--display-name",
                "GPT-4o (Updated)",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["providers"]["openai"]["models"]["gpt-4o"]["display_name"] == "GPT-4o (Updated)"

    def test_update_capability_flags(self, runner, config_file):
        """provider model update can toggle capability flags via true/false values."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "update",
                "openai",
                "gpt-4o",
                "--tools",
                "false",
                "--vision",
                "false",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code == 0

        with open(config_file) as f:
            data = yaml.safe_load(f)
        model = data["providers"]["openai"]["models"]["gpt-4o"]
        assert model["supports_tools"] is False
        assert model["supports_vision"] is False

    def test_update_no_options_produces_message(self, runner, config_file):
        """provider model update with no options prints a message."""
        result = runner.invoke(
            main,
            ["provider", "model", "update", "openai", "gpt-4o", "-c", config_file],
        )
        assert result.exit_code == 0
        assert "No changes" in result.output

    def test_update_nonexistent_model(self, runner, config_file):
        """provider model update fails for an unknown model."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "update",
                "openai",
                "nonexistent-model",
                "--max-tokens",
                "1000",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_update_nonexistent_provider(self, runner, config_file):
        """provider model update fails for an unknown provider."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "update",
                "nonexistent",
                "gpt-4o",
                "--max-tokens",
                "1000",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# provider model show
# ---------------------------------------------------------------------------


class TestProviderModelShow:
    """Tests for 'provider model show' command."""

    def test_show_model_text(self, runner, config_file):
        """provider model show displays model details in text format."""
        result = runner.invoke(
            main,
            ["provider", "model", "show", "openai", "gpt-4o", "-c", config_file],
        )
        assert result.exit_code == 0
        assert "GPT-4o" in result.output
        assert "openai" in result.output
        assert "16384" in result.output

    def test_show_model_json(self, runner, config_file):
        """provider model show --format json returns valid JSON."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "show",
                "openai",
                "gpt-4o",
                "-c",
                config_file,
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "gpt-4o"
        assert data["provider"] == "openai"
        assert data["max_tokens"] == 16384

    def test_show_marks_default_model(self, runner, config_file):
        """provider model show indicates when a model is the default."""
        result = runner.invoke(
            main,
            ["provider", "model", "show", "openai", "gpt-4o", "-c", config_file],
        )
        assert result.exit_code == 0
        assert "Default" in result.output

    def test_show_nonexistent_model(self, runner, config_file):
        """provider model show fails for an unknown model."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "show",
                "openai",
                "nonexistent-model",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_show_nonexistent_provider(self, runner, config_file):
        """provider model show fails for an unknown provider."""
        result = runner.invoke(
            main,
            [
                "provider",
                "model",
                "show",
                "nonexistent",
                "gpt-4o",
                "-c",
                config_file,
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# provider model help
# ---------------------------------------------------------------------------


class TestProviderModelHelp:
    """Tests for the 'provider model' group help."""

    def test_model_group_help(self, runner):
        """provider model --help shows the group description."""
        result = runner.invoke(main, ["provider", "model", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()

    def test_model_add_help(self, runner):
        """provider model add --help exits cleanly."""
        result = runner.invoke(main, ["provider", "model", "add", "--help"])
        assert result.exit_code == 0

    def test_model_remove_help(self, runner):
        """provider model remove --help exits cleanly."""
        result = runner.invoke(main, ["provider", "model", "remove", "--help"])
        assert result.exit_code == 0

    def test_model_list_help(self, runner):
        """provider model list --help exits cleanly."""
        result = runner.invoke(main, ["provider", "model", "list", "--help"])
        assert result.exit_code == 0

    def test_model_set_default_help(self, runner):
        """provider model set-default --help exits cleanly."""
        result = runner.invoke(main, ["provider", "model", "set-default", "--help"])
        assert result.exit_code == 0

    def test_model_update_help(self, runner):
        """provider model update --help exits cleanly."""
        result = runner.invoke(main, ["provider", "model", "update", "--help"])
        assert result.exit_code == 0

    def test_model_show_help(self, runner):
        """provider model show --help exits cleanly."""
        result = runner.invoke(main, ["provider", "model", "show", "--help"])
        assert result.exit_code == 0
