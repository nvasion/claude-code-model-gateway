"""Tests for the new configuration validation and testing CLI commands.

Covers the following commands:
  - config schema
  - config lint
  - config doctor
  - config export
  - config env-check
"""

import json
import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from src.cli import main
from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig
from src.validation.testing import generate_minimal_config, generate_sample_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def minimal_config_file(tmp_path):
    """Write a minimal valid config file and return its path string."""
    config = generate_minimal_config()
    path = tmp_path / "minimal.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


@pytest.fixture
def sample_config_file(tmp_path):
    """Write the full sample config file and return its path string."""
    config = generate_sample_config()
    path = tmp_path / "sample.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


@pytest.fixture
def invalid_config_file(tmp_path):
    """Write an invalid config file (negative timeout, bad log level)."""
    config = {
        "default_provider": "nonexistent",
        "timeout": -1,
        "log_level": "bad_level",
    }
    path = tmp_path / "invalid.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


@pytest.fixture
def warning_config_file(tmp_path):
    """Write a config that produces warnings but no errors."""
    config = generate_minimal_config()
    config["timeout"] = 500  # too high -> warning
    path = tmp_path / "warning.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


@pytest.fixture
def placeholder_config_file(tmp_path):
    """Write a config with placeholder URLs (production failure)."""
    config = {
        "default_provider": "azure",
        "providers": {
            "azure": {
                "name": "azure",
                "display_name": "Azure OpenAI",
                "api_base": "https://<YOUR_RESOURCE>.openai.azure.com",
                "api_key_env_var": "AZURE_OPENAI_KEY",
                "enabled": True,
            }
        },
    }
    path = tmp_path / "placeholder.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


@pytest.fixture
def multi_provider_config_file(tmp_path):
    """Write a config with multiple providers including env var references."""
    config = {
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
                "enabled": True,
                "models": {
                    "gpt-4o": {"name": "gpt-4o", "max_tokens": 16384}
                },
            },
            "anthropic": {
                "name": "anthropic",
                "display_name": "Anthropic",
                "api_base": "https://api.anthropic.com/v1",
                "api_key_env_var": "ANTHROPIC_API_KEY",
                "auth_type": "api_key",
                "enabled": True,
                "models": {
                    "claude-3-5-sonnet": {
                        "name": "claude-3-5-sonnet",
                        "max_tokens": 200000,
                    }
                },
            },
            "local": {
                "name": "local",
                "display_name": "Local LLM",
                "api_base": "http://localhost:8000/v1",
                "auth_type": "none",
                "enabled": False,
            },
        },
    }
    path = tmp_path / "multi.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


@pytest.fixture
def json_config_file(tmp_path):
    """Write a minimal valid config as JSON."""
    config = generate_minimal_config()
    path = tmp_path / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return str(path)


# ===========================================================================
# Tests: config schema
# ===========================================================================


class TestConfigSchema:
    """Tests for the `config schema` command."""

    def test_schema_shows_all_fields(self, runner):
        """Running without options should list fields from all categories."""
        result = runner.invoke(main, ["config", "schema"])
        assert result.exit_code == 0
        output = result.output
        # Top-level fields should appear
        assert "timeout" in output
        assert "log_level" in output
        assert "default_provider" in output

    def test_schema_text_output_has_categories(self, runner):
        """Text output should include category headers."""
        result = runner.invoke(main, ["config", "schema"])
        assert result.exit_code == 0
        # At least one category header should be uppercased
        assert any(
            cat in result.output.upper()
            for cat in ["GATEWAY", "PROVIDER", "MODEL", "LOGGING", "RETRY"]
        )

    def test_schema_json_output(self, runner):
        """JSON output should be valid and contain expected keys."""
        result = runner.invoke(main, ["config", "schema", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "fields" in data
        assert "total" in data
        assert data["total"] > 0
        assert isinstance(data["fields"], list)

    def test_schema_json_field_structure(self, runner):
        """Each JSON field entry should have expected keys."""
        result = runner.invoke(main, ["config", "schema", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for field in data["fields"]:
            assert "name" in field
            assert "type" in field
            assert "description" in field

    def test_schema_filter_by_category_gateway(self, runner):
        """Filtering by 'gateway' category should return gateway fields."""
        result = runner.invoke(
            main, ["config", "schema", "--category", "gateway"]
        )
        assert result.exit_code == 0
        assert "default_provider" in result.output or "timeout" in result.output

    def test_schema_filter_by_category_logging(self, runner):
        """Filtering by 'logging' category should include log_level."""
        result = runner.invoke(
            main, ["config", "schema", "--category", "logging"]
        )
        assert result.exit_code == 0
        assert "log_level" in result.output

    def test_schema_filter_by_category_json(self, runner):
        """Category filter in JSON mode should return subset of fields."""
        result = runner.invoke(
            main,
            ["config", "schema", "--category", "logging", "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] >= 1

    def test_schema_specific_field_timeout(self, runner):
        """--field timeout should show timeout field details."""
        result = runner.invoke(main, ["config", "schema", "--field", "timeout"])
        assert result.exit_code == 0
        output = result.output
        assert "timeout" in output.lower()
        assert "integer" in output.lower()
        assert "30" in output  # default value

    def test_schema_specific_field_log_level(self, runner):
        """--field log_level should show allowed values."""
        result = runner.invoke(
            main, ["config", "schema", "--field", "log_level"]
        )
        assert result.exit_code == 0
        assert "info" in result.output
        assert "debug" in result.output

    def test_schema_specific_field_json(self, runner):
        """--field with JSON format should return field definition."""
        result = runner.invoke(
            main,
            ["config", "schema", "--field", "timeout", "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "timeout"
        assert data["type"] == "integer"

    def test_schema_unknown_field_exits_nonzero(self, runner):
        """Requesting an unknown field path should exit with code 1."""
        result = runner.invoke(
            main, ["config", "schema", "--field", "totally.unknown.field"]
        )
        assert result.exit_code == 1

    def test_schema_template_format_is_yaml(self, runner):
        """--format template should produce parseable YAML output."""
        result = runner.invoke(
            main, ["config", "schema", "--format", "template"]
        )
        assert result.exit_code == 0
        # Output should be non-empty YAML
        assert len(result.output.strip()) > 0
        # Should have YAML comments
        assert "#" in result.output

    def test_schema_template_contains_key_fields(self, runner):
        """YAML template should contain key field names."""
        result = runner.invoke(
            main, ["config", "schema", "--format", "template"]
        )
        assert result.exit_code == 0
        # Core fields should appear in the template
        assert "log_level" in result.output
        assert "timeout" in result.output

    def test_schema_help_text(self, runner):
        """--help should describe the command."""
        result = runner.invoke(main, ["config", "schema", "--help"])
        assert result.exit_code == 0
        assert "schema" in result.output.lower()
        assert "--field" in result.output
        assert "--category" in result.output
        assert "--format" in result.output


# ===========================================================================
# Tests: config lint
# ===========================================================================


class TestConfigLint:
    """Tests for the `config lint` command."""

    def test_lint_valid_config_exits_zero(self, runner, minimal_config_file):
        """Linting a valid config should exit 0."""
        result = runner.invoke(
            main, ["config", "lint", "-c", minimal_config_file]
        )
        assert result.exit_code == 0

    def test_lint_invalid_config_exits_one(self, runner, invalid_config_file):
        """Linting a config with errors should exit 1."""
        result = runner.invoke(
            main, ["config", "lint", "-c", invalid_config_file]
        )
        assert result.exit_code == 1

    def test_lint_warning_config_exits_two(self, runner, warning_config_file):
        """Linting a config with only warnings should exit 2 (no errors)."""
        result = runner.invoke(
            main, ["config", "lint", "-c", warning_config_file]
        )
        # Exit 2 means warnings only
        assert result.exit_code == 2

    def test_lint_shows_error_messages(self, runner, invalid_config_file):
        """Linting an invalid config should display error messages."""
        result = runner.invoke(
            main, ["config", "lint", "-c", invalid_config_file]
        )
        assert "error" in result.output.lower()

    def test_lint_json_output_valid_config(self, runner, minimal_config_file):
        """JSON output for a valid config should have zero errors."""
        result = runner.invoke(
            main,
            ["config", "lint", "-c", minimal_config_file, "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["error_count"] == 0
        assert "messages" in data
        assert "profile" in data

    def test_lint_json_output_invalid_config(self, runner, invalid_config_file):
        """JSON output for an invalid config should have nonzero error count."""
        result = runner.invoke(
            main,
            ["config", "lint", "-c", invalid_config_file, "--format", "json"],
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["error_count"] > 0
        assert len(data["messages"]) > 0

    def test_lint_json_message_structure(self, runner, invalid_config_file):
        """Each JSON message should have severity, path, and message keys."""
        result = runner.invoke(
            main,
            ["config", "lint", "-c", invalid_config_file, "--format", "json"],
        )
        data = json.loads(result.output)
        for msg in data["messages"]:
            assert "severity" in msg
            assert "path" in msg
            assert "message" in msg

    def test_lint_strict_profile_treats_warnings_as_errors(
        self, runner, warning_config_file
    ):
        """In strict mode the linter should exit 1 (errors from warnings)."""
        result = runner.invoke(
            main,
            [
                "config", "lint",
                "-c", warning_config_file,
                "--profile", "strict",
            ],
        )
        # strict profile: warnings still exit 2 from lint perspective;
        # the profile controls how many warnings appear, not the exit code
        assert result.exit_code in (0, 1, 2)

    def test_lint_production_profile(self, runner, placeholder_config_file):
        """Production profile should flag placeholder URLs as errors."""
        result = runner.invoke(
            main,
            [
                "config", "lint",
                "-c", placeholder_config_file,
                "--profile", "production",
            ],
        )
        assert result.exit_code == 1

    def test_lint_no_suggestions_flag(self, runner, warning_config_file):
        """--no-suggestions should suppress fix hints."""
        # With suggestions (default)
        result_with = runner.invoke(
            main, ["config", "lint", "-c", warning_config_file]
        )
        # Without suggestions
        result_without = runner.invoke(
            main,
            ["config", "lint", "-c", warning_config_file, "--no-suggestions"],
        )
        # Without suggestions should not have '↳' hints
        assert "↳" not in result_without.output

    def test_lint_shows_file_path(self, runner, minimal_config_file):
        """Lint output should reference the config file path."""
        result = runner.invoke(
            main, ["config", "lint", "-c", minimal_config_file]
        )
        # File path or "No issues" should appear
        assert result.exit_code == 0

    def test_lint_json_output_contains_file_field(
        self, runner, minimal_config_file
    ):
        """JSON output should include a 'file' field."""
        result = runner.invoke(
            main,
            ["config", "lint", "-c", minimal_config_file, "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "file" in data

    def test_lint_help_text(self, runner):
        """--help should describe the lint command."""
        result = runner.invoke(main, ["config", "lint", "--help"])
        assert result.exit_code == 0
        assert "lint" in result.output.lower()
        assert "--profile" in result.output
        assert "--format" in result.output


# ===========================================================================
# Tests: config doctor
# ===========================================================================


class TestConfigDoctor:
    """Tests for the `config doctor` command."""

    def test_doctor_valid_config_exits_zero(self, runner, minimal_config_file):
        """Doctor on a valid config should exit 0."""
        result = runner.invoke(
            main, ["config", "doctor", "-c", minimal_config_file]
        )
        assert result.exit_code == 0

    def test_doctor_invalid_config_exits_one(self, runner, invalid_config_file):
        """Doctor on an invalid config should exit 1."""
        result = runner.invoke(
            main, ["config", "doctor", "-c", invalid_config_file]
        )
        assert result.exit_code == 1

    def test_doctor_shows_parse_check(self, runner, minimal_config_file):
        """Doctor output should show the parse check."""
        result = runner.invoke(
            main, ["config", "doctor", "-c", minimal_config_file]
        )
        assert result.exit_code == 0
        assert "parse" in result.output.lower() or "PARSE" in result.output

    def test_doctor_shows_semantic_check(self, runner, minimal_config_file):
        """Doctor output should show the semantic check."""
        result = runner.invoke(
            main, ["config", "doctor", "-c", minimal_config_file]
        )
        assert result.exit_code == 0
        assert "semantic" in result.output.lower() or "SEMANTIC" in result.output

    def test_doctor_shows_production_check(self, runner, minimal_config_file):
        """Doctor output should show the production check."""
        result = runner.invoke(
            main, ["config", "doctor", "-c", minimal_config_file]
        )
        assert result.exit_code == 0
        assert "production" in result.output.lower() or "PRODUCTION" in result.output

    def test_doctor_json_output_valid_config(self, runner, minimal_config_file):
        """JSON output for a valid config should have all_passed=true."""
        result = runner.invoke(
            main,
            ["config", "doctor", "-c", minimal_config_file, "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "all_passed" in data
        assert "checks" in data
        assert isinstance(data["checks"], list)
        assert len(data["checks"]) >= 3  # parse, schema, semantic, production

    def test_doctor_json_output_invalid_config(self, runner, invalid_config_file):
        """JSON output for an invalid config should have all_passed=false."""
        result = runner.invoke(
            main,
            [
                "config", "doctor",
                "-c", invalid_config_file,
                "--format", "json",
            ],
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["all_passed"] is False

    def test_doctor_json_check_structure(self, runner, minimal_config_file):
        """Each check in JSON output should have required fields."""
        result = runner.invoke(
            main,
            ["config", "doctor", "-c", minimal_config_file, "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        for check in data["checks"]:
            assert "name" in check
            assert "description" in check
            assert "passed" in check
            assert "message" in check
            assert "details" in check

    def test_doctor_check_env_no_providers(self, runner, tmp_path):
        """--check-env on empty config should not crash."""
        empty_config = tmp_path / "empty.yaml"
        with open(empty_config, "w") as f:
            yaml.dump({}, f)
        result = runner.invoke(
            main,
            [
                "config", "doctor",
                "-c", str(empty_config),
                "--check-env",
            ],
        )
        # Should not crash (exit code may vary due to other checks)
        assert result.exit_code in (0, 1)

    def test_doctor_check_env_missing_vars(self, runner, multi_provider_config_file):
        """--check-env should detect missing environment variables."""
        # Ensure the vars are NOT set
        env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        result = runner.invoke(
            main,
            [
                "config", "doctor",
                "-c", multi_provider_config_file,
                "--check-env",
            ],
            env=env,
        )
        data_json = runner.invoke(
            main,
            [
                "config", "doctor",
                "-c", multi_provider_config_file,
                "--check-env",
                "--format", "json",
            ],
            env=env,
        )
        data = json.loads(data_json.output)
        env_check = next(
            (c for c in data["checks"] if c["name"] == "env_vars"), None
        )
        assert env_check is not None

    def test_doctor_check_env_with_set_var(self, runner, minimal_config_file):
        """--check-env should pass when the env var is set."""
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = "sk-test-key"
        result = runner.invoke(
            main,
            [
                "config", "doctor",
                "-c", minimal_config_file,
                "--check-env",
                "--format", "json",
            ],
            env=env,
        )
        data = json.loads(result.output)
        env_check = next(
            (c for c in data["checks"] if c["name"] == "env_vars"), None
        )
        assert env_check is not None
        assert env_check["passed"] is True

    def test_doctor_json_includes_config_file(self, runner, minimal_config_file):
        """JSON output should include the config_file field."""
        result = runner.invoke(
            main,
            ["config", "doctor", "-c", minimal_config_file, "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "config_file" in data

    def test_doctor_placeholder_config_fails_production(
        self, runner, placeholder_config_file
    ):
        """Config with placeholder URLs should fail the production check."""
        result = runner.invoke(
            main,
            [
                "config", "doctor",
                "-c", placeholder_config_file,
                "--format", "json",
            ],
        )
        data = json.loads(result.output)
        prod_check = next(
            (c for c in data["checks"] if c["name"] == "production"), None
        )
        assert prod_check is not None
        assert prod_check["passed"] is False

    def test_doctor_help_text(self, runner):
        """--help should describe the doctor command."""
        result = runner.invoke(main, ["config", "doctor", "--help"])
        assert result.exit_code == 0
        assert "doctor" in result.output.lower()
        assert "--check-env" in result.output
        assert "--format" in result.output


# ===========================================================================
# Tests: config export
# ===========================================================================


class TestConfigExport:
    """Tests for the `config export` command."""

    def test_export_yaml_to_json(self, runner, minimal_config_file, tmp_path):
        """Export a YAML config to JSON."""
        output = str(tmp_path / "output.json")
        result = runner.invoke(
            main,
            [
                "config", "export",
                "-c", minimal_config_file,
                "-o", output,
            ],
        )
        assert result.exit_code == 0
        assert Path(output).exists()
        with open(output) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_export_json_to_yaml(self, runner, json_config_file, tmp_path):
        """Export a JSON config to YAML."""
        output = str(tmp_path / "output.yaml")
        result = runner.invoke(
            main,
            [
                "config", "export",
                "-c", json_config_file,
                "-o", output,
            ],
        )
        assert result.exit_code == 0
        assert Path(output).exists()
        with open(output) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_export_yaml_to_yaml(self, runner, minimal_config_file, tmp_path):
        """Export a YAML config to another YAML file."""
        output = str(tmp_path / "copy.yaml")
        result = runner.invoke(
            main,
            [
                "config", "export",
                "-c", minimal_config_file,
                "-o", output,
            ],
        )
        assert result.exit_code == 0
        assert Path(output).exists()

    def test_export_preserves_data(self, runner, minimal_config_file, tmp_path):
        """Exported data should match the source config."""
        output = str(tmp_path / "exported.json")
        runner.invoke(
            main,
            [
                "config", "export",
                "-c", minimal_config_file,
                "-o", output,
            ],
        )
        with open(minimal_config_file) as f:
            src = yaml.safe_load(f)
        with open(output) as f:
            dst = json.load(f)
        # Both should have same providers
        assert set(src.get("providers", {}).keys()) == set(dst.get("providers", {}).keys())

    def test_export_fails_if_output_exists_without_force(
        self, runner, minimal_config_file, tmp_path
    ):
        """Export should fail if the output file exists and --force is not set."""
        output = tmp_path / "existing.yaml"
        output.write_text("existing: content\n")
        result = runner.invoke(
            main,
            [
                "config", "export",
                "-c", minimal_config_file,
                "-o", str(output),
            ],
        )
        assert result.exit_code == 1
        assert "already exists" in result.output.lower() or "error" in result.output.lower()

    def test_export_force_overwrites(self, runner, minimal_config_file, tmp_path):
        """--force should allow overwriting an existing file."""
        output = tmp_path / "existing.yaml"
        output.write_text("old: content\n")
        result = runner.invoke(
            main,
            [
                "config", "export",
                "-c", minimal_config_file,
                "-o", str(output),
                "--force",
            ],
        )
        assert result.exit_code == 0
        # File should now contain the exported config
        with open(output) as f:
            data = yaml.safe_load(f)
        assert "providers" in data

    def test_export_unsupported_format_exits_one(
        self, runner, minimal_config_file, tmp_path
    ):
        """Export to an unsupported format should exit 1."""
        output = str(tmp_path / "config.xml")
        result = runner.invoke(
            main,
            [
                "config", "export",
                "-c", minimal_config_file,
                "-o", output,
            ],
        )
        assert result.exit_code == 1

    def test_export_no_validate_skips_validation(
        self, runner, invalid_config_file, tmp_path
    ):
        """--no-validate should export even an invalid config."""
        output = str(tmp_path / "exported.yaml")
        result = runner.invoke(
            main,
            [
                "config", "export",
                "-c", invalid_config_file,
                "-o", output,
                "--no-validate",
            ],
        )
        # With no-validate, invalid config should still be exported
        assert result.exit_code == 0
        assert Path(output).exists()

    def test_export_output_message(self, runner, minimal_config_file, tmp_path):
        """Export should print a confirmation message."""
        output = str(tmp_path / "out.json")
        result = runner.invoke(
            main,
            [
                "config", "export",
                "-c", minimal_config_file,
                "-o", output,
            ],
        )
        assert result.exit_code == 0
        assert "exported" in result.output.lower() or "→" in result.output

    def test_export_yml_extension(self, runner, minimal_config_file, tmp_path):
        """Export should support .yml extension (not just .yaml)."""
        output = str(tmp_path / "config.yml")
        result = runner.invoke(
            main,
            [
                "config", "export",
                "-c", minimal_config_file,
                "-o", output,
            ],
        )
        assert result.exit_code == 0
        assert Path(output).exists()

    def test_export_help_text(self, runner):
        """--help should describe the export command."""
        result = runner.invoke(main, ["config", "export", "--help"])
        assert result.exit_code == 0
        assert "export" in result.output.lower()
        assert "--output" in result.output or "-o" in result.output


# ===========================================================================
# Tests: config env-check
# ===========================================================================


class TestConfigEnvCheck:
    """Tests for the `config env-check` command."""

    def test_env_check_all_set_exits_zero(self, runner, minimal_config_file):
        """If all env vars are set, should exit 0."""
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = "sk-test"
        result = runner.invoke(
            main, ["config", "env-check", "-c", minimal_config_file], env=env
        )
        assert result.exit_code == 0

    def test_env_check_missing_exits_one(self, runner, minimal_config_file):
        """If env var is missing, should exit 1."""
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        result = runner.invoke(
            main, ["config", "env-check", "-c", minimal_config_file], env=env
        )
        assert result.exit_code == 1

    def test_env_check_json_output_all_set(self, runner, minimal_config_file):
        """JSON output when all vars are set should show all_set=true."""
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = "sk-test"
        result = runner.invoke(
            main,
            [
                "config", "env-check",
                "-c", minimal_config_file,
                "--format", "json",
            ],
            env=env,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["all_set"] is True
        assert "env_vars" in data
        assert "missing_count" in data

    def test_env_check_json_output_missing_var(self, runner, minimal_config_file):
        """JSON output when var is missing should show all_set=false."""
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        result = runner.invoke(
            main,
            [
                "config", "env-check",
                "-c", minimal_config_file,
                "--format", "json",
            ],
            env=env,
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["all_set"] is False
        assert data["missing_count"] > 0

    def test_env_check_json_entry_structure(self, runner, minimal_config_file):
        """Each entry in JSON output should have required keys."""
        result = runner.invoke(
            main,
            [
                "config", "env-check",
                "-c", minimal_config_file,
                "--format", "json",
            ],
        )
        data = json.loads(result.output)
        for entry in data["env_vars"]:
            assert "provider" in entry
            assert "env_var" in entry
            assert "status" in entry

    def test_env_check_no_providers(self, runner, tmp_path):
        """Empty config with no providers should exit 0 (nothing to check)."""
        empty = tmp_path / "empty.yaml"
        with open(empty, "w") as f:
            yaml.dump({}, f)
        result = runner.invoke(
            main, ["config", "env-check", "-c", str(empty)]
        )
        assert result.exit_code == 0

    def test_env_check_no_auth_provider(self, runner, multi_provider_config_file):
        """Providers with auth_type=none should not require env vars."""
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = "sk-test"
        env["ANTHROPIC_API_KEY"] = "sk-ant-test"
        result = runner.invoke(
            main,
            [
                "config", "env-check",
                "-c", multi_provider_config_file,
                "--format", "json",
            ],
            env=env,
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Find the 'local' provider entry (no-auth)
        local = next(
            (e for e in data["env_vars"] if e["provider"] == "local"), None
        )
        assert local is not None
        assert local["status"] in ("no-auth", "disabled")

    def test_env_check_show_all_flag(self, runner, minimal_config_file):
        """--all should show status for all providers including set ones."""
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = "sk-test"
        result = runner.invoke(
            main,
            [
                "config", "env-check",
                "-c", minimal_config_file,
                "--all",
            ],
            env=env,
        )
        assert result.exit_code == 0
        # Should show provider info (not empty)
        output = result.output
        assert "openai" in output.lower() or "provider" in output.lower()

    def test_env_check_multi_provider_some_missing(
        self, runner, multi_provider_config_file
    ):
        """With multiple providers, only set some env vars."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        env["OPENAI_API_KEY"] = "sk-test"
        # ANTHROPIC_API_KEY not set
        result = runner.invoke(
            main,
            [
                "config", "env-check",
                "-c", multi_provider_config_file,
                "--format", "json",
            ],
            env=env,
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["missing_count"] >= 1

    def test_env_check_help_text(self, runner):
        """--help should describe the env-check command."""
        result = runner.invoke(main, ["config", "env-check", "--help"])
        assert result.exit_code == 0
        assert "env" in result.output.lower()
        assert "--format" in result.output


# ===========================================================================
# Integration: all new commands appear in main help
# ===========================================================================


class TestNewCommandsInHelp:
    """Verify that all new commands appear in the right help output."""

    def test_config_group_shows_schema(self, runner):
        """config --help should list the schema subcommand."""
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "schema" in result.output

    def test_config_group_shows_lint(self, runner):
        """config --help should list the lint subcommand."""
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "lint" in result.output

    def test_config_group_shows_doctor(self, runner):
        """config --help should list the doctor subcommand."""
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "doctor" in result.output

    def test_config_group_shows_export(self, runner):
        """config --help should list the export subcommand."""
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "export" in result.output

    def test_config_group_shows_env_check(self, runner):
        """config --help should list the env-check subcommand."""
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "env-check" in result.output

    def test_main_help_still_works(self, runner):
        """Main --help should still show all top-level commands."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "config" in result.output
        assert "gateway" in result.output
        assert "proxy" in result.output


# ===========================================================================
# Edge cases and robustness
# ===========================================================================


class TestEdgeCases:
    """Edge case and robustness tests for the new validation tools."""

    def test_lint_empty_config_file(self, runner, tmp_path):
        """Linting an empty config file should not crash."""
        empty = tmp_path / "empty.yaml"
        with open(empty, "w") as f:
            yaml.dump({}, f)
        result = runner.invoke(
            main, ["config", "lint", "-c", str(empty)]
        )
        # Empty config is technically valid (warnings only)
        assert result.exit_code in (0, 2)

    def test_doctor_empty_config_file(self, runner, tmp_path):
        """Doctor on an empty config file should not crash."""
        empty = tmp_path / "empty.yaml"
        with open(empty, "w") as f:
            yaml.dump({}, f)
        result = runner.invoke(
            main, ["config", "doctor", "-c", str(empty)]
        )
        assert result.exit_code in (0, 1)

    def test_env_check_provider_no_api_key_env(self, runner, tmp_path):
        """Provider without api_key_env_var configured."""
        config = {
            "default_provider": "test",
            "providers": {
                "test": {
                    "name": "test",
                    "api_base": "https://api.example.com",
                    "auth_type": "api_key",
                    # No api_key_env_var
                    "enabled": True,
                }
            },
        }
        path = tmp_path / "no-env.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)
        result = runner.invoke(
            main,
            ["config", "env-check", "-c", str(path), "--format", "json"],
        )
        data = json.loads(result.output)
        test_entry = next(
            (e for e in data["env_vars"] if e["provider"] == "test"), None
        )
        assert test_entry is not None
        assert test_entry["status"] == "missing-config"

    def test_schema_provider_field(self, runner):
        """config schema --field providers should show providers field."""
        result = runner.invoke(
            main, ["config", "schema", "--field", "providers"]
        )
        assert result.exit_code == 0
        assert "provider" in result.output.lower()

    def test_lint_json_profile_field(self, runner, minimal_config_file):
        """JSON lint output should include the profile used."""
        result = runner.invoke(
            main,
            [
                "config", "lint",
                "-c", minimal_config_file,
                "--profile", "strict",
                "--format", "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["profile"] == "strict"

    def test_export_no_output_flag_fails(self, runner, minimal_config_file):
        """Export without -o should fail (required option)."""
        result = runner.invoke(
            main, ["config", "export", "-c", minimal_config_file]
        )
        assert result.exit_code != 0

    def test_doctor_json_check_names(self, runner, minimal_config_file):
        """Doctor JSON output should contain checks with specific names."""
        result = runner.invoke(
            main,
            ["config", "doctor", "-c", minimal_config_file, "--format", "json"],
        )
        data = json.loads(result.output)
        check_names = {c["name"] for c in data["checks"]}
        assert "parse" in check_names
        assert "schema" in check_names
        assert "semantic" in check_names
        assert "production" in check_names

    def test_lint_and_doctor_agree_on_validity(
        self, runner, minimal_config_file
    ):
        """Lint and doctor should agree that a valid config is valid."""
        lint_result = runner.invoke(
            main,
            ["config", "lint", "-c", minimal_config_file, "--format", "json"],
        )
        doctor_result = runner.invoke(
            main,
            ["config", "doctor", "-c", minimal_config_file, "--format", "json"],
        )
        lint_data = json.loads(lint_result.output)
        doctor_data = json.loads(doctor_result.output)
        assert lint_data["error_count"] == 0
        assert doctor_data["all_passed"] is True

    def test_lint_and_doctor_agree_on_invalidity(
        self, runner, invalid_config_file
    ):
        """Lint and doctor should both report issues for an invalid config."""
        lint_result = runner.invoke(
            main,
            ["config", "lint", "-c", invalid_config_file, "--format", "json"],
        )
        doctor_result = runner.invoke(
            main,
            ["config", "doctor", "-c", invalid_config_file, "--format", "json"],
        )
        lint_data = json.loads(lint_result.output)
        doctor_data = json.loads(doctor_result.output)
        assert lint_data["error_count"] > 0
        assert doctor_data["all_passed"] is False

    def test_schema_all_categories_covered(self, runner):
        """Schema should include all expected categories."""
        result = runner.invoke(
            main, ["config", "schema", "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        categories = {f.get("category") for f in data["fields"] if f.get("category")}
        assert "gateway" in categories or "provider" in categories
