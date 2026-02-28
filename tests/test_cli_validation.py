"""Tests for configuration validation and testing CLI commands."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from src.cli import main
from src.validation.testing import generate_minimal_config, generate_sample_config


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def valid_config_file(tmp_path):
    """Create a valid YAML config file."""
    config = generate_minimal_config()
    path = tmp_path / "valid.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


@pytest.fixture
def valid_json_file(tmp_path):
    """Create a valid JSON config file."""
    config = generate_minimal_config()
    path = tmp_path / "valid.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return str(path)


@pytest.fixture
def invalid_config_file(tmp_path):
    """Create an invalid YAML config file."""
    config = {
        "default_provider": "nonexistent",
        "timeout": -1,
        "log_level": "invalid_level",
    }
    path = tmp_path / "invalid.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


@pytest.fixture
def warning_config_file(tmp_path):
    """Create a config file that produces warnings but no errors."""
    config = generate_minimal_config()
    config["timeout"] = 500  # triggers high-timeout warning
    path = tmp_path / "warning.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


@pytest.fixture
def config_file_a(tmp_path):
    """Create first config file for diff testing."""
    config = generate_minimal_config()
    config["timeout"] = 30
    path = tmp_path / "a.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


@pytest.fixture
def config_file_b(tmp_path):
    """Create second config file for diff testing."""
    config = generate_minimal_config()
    config["timeout"] = 60
    path = tmp_path / "b.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


# ---------------------------------------------------------------------------
# validate-config command tests
# ---------------------------------------------------------------------------


class TestValidateConfigCmd:
    """Tests for the validate-config CLI command."""

    def test_validate_valid_config(self, runner, valid_config_file):
        """Test validating a valid config file."""
        result = runner.invoke(
            main, ["validate-config", "-c", valid_config_file]
        )
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_invalid_config(self, runner, invalid_config_file):
        """Test validating an invalid config file."""
        result = runner.invoke(
            main, ["validate-config", "-c", invalid_config_file]
        )
        assert result.exit_code == 1
        assert "error" in result.output.lower()

    def test_validate_json_output(self, runner, valid_config_file):
        """Test JSON output format."""
        result = runner.invoke(
            main,
            ["validate-config", "-c", valid_config_file, "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["valid"] is True
        assert "error_count" in data
        assert "warning_count" in data
        assert "messages" in data

    def test_validate_json_output_invalid(self, runner, invalid_config_file):
        """Test JSON output for invalid config."""
        result = runner.invoke(
            main,
            [
                "validate-config",
                "-c",
                invalid_config_file,
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["valid"] is False
        assert data["error_count"] > 0

    def test_validate_strict_mode_warnings(self, runner, warning_config_file):
        """Test that strict mode treats warnings as errors."""
        # Without strict: should pass
        result = runner.invoke(
            main, ["validate-config", "-c", warning_config_file]
        )
        assert result.exit_code == 0

        # With strict: should fail on warnings
        result = runner.invoke(
            main,
            ["validate-config", "-c", warning_config_file, "--strict"],
        )
        assert result.exit_code == 1

    def test_validate_show_info(self, runner, valid_config_file):
        """Test --show-info flag includes info messages."""
        result = runner.invoke(
            main,
            [
                "validate-config",
                "-c",
                valid_config_file,
                "--show-info",
            ],
        )
        assert result.exit_code == 0

    def test_validate_nonexistent_file(self, runner):
        """Test validating a nonexistent file."""
        result = runner.invoke(
            main, ["validate-config", "-c", "/nonexistent/path.yaml"]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# test-config command tests
# ---------------------------------------------------------------------------


class TestTestConfigCmd:
    """Tests for the test-config CLI command."""

    def test_builtin_tests_pass(self, runner):
        """Test that built-in test suite passes."""
        result = runner.invoke(main, ["test-config"])
        assert result.exit_code == 0
        assert "passed" in result.output.lower()
        assert "All tests passed" in result.output

    def test_builtin_tests_json_output(self, runner):
        """Test JSON output format for test suite."""
        result = runner.invoke(
            main, ["test-config", "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total" in data
        assert "passed" in data
        assert "failed" in data
        assert data["failed"] == 0
        assert "results" in data

    def test_with_file(self, runner, valid_config_file):
        """Test running tests with a specific config file."""
        result = runner.invoke(
            main, ["test-config", "-c", valid_config_file]
        )
        assert result.exit_code == 0
        assert "passed" in result.output.lower()

    def test_with_file_json(self, runner, valid_config_file):
        """Test JSON output with a specific config file."""
        result = runner.invoke(
            main,
            [
                "test-config",
                "-c",
                valid_config_file,
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["failed"] == 0

    def test_no_builtin(self, runner, valid_config_file):
        """Test running only file tests (no built-in)."""
        result = runner.invoke(
            main,
            [
                "test-config",
                "-c",
                valid_config_file,
                "--no-builtin",
            ],
        )
        assert result.exit_code == 0

    def test_no_tests_to_run(self, runner):
        """Test with no file and no builtin."""
        result = runner.invoke(
            main, ["test-config", "--no-builtin"]
        )
        assert result.exit_code == 0
        assert "No tests to run" in result.output


# ---------------------------------------------------------------------------
# config-diff command tests
# ---------------------------------------------------------------------------


class TestConfigDiffCmd:
    """Tests for the config-diff CLI command."""

    def test_identical_files(self, runner, valid_config_file):
        """Test diffing identical files."""
        result = runner.invoke(
            main,
            [
                "config-diff",
                valid_config_file,
                valid_config_file,
            ],
        )
        assert result.exit_code == 0
        assert "identical" in result.output.lower()

    def test_different_files(self, runner, config_file_a, config_file_b):
        """Test diffing different files."""
        result = runner.invoke(
            main, ["config-diff", config_file_a, config_file_b]
        )
        assert result.exit_code == 2  # differences found
        assert "timeout" in result.output.lower()

    def test_diff_json_output(self, runner, config_file_a, config_file_b):
        """Test JSON output for diff."""
        result = runner.invoke(
            main,
            [
                "config-diff",
                config_file_a,
                config_file_b,
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 2
        data = json.loads(result.output)
        assert data["identical"] is False
        assert data["difference_count"] > 0

    def test_diff_json_identical(self, runner, valid_config_file):
        """Test JSON output for identical files."""
        result = runner.invoke(
            main,
            [
                "config-diff",
                valid_config_file,
                valid_config_file,
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["identical"] is True
        assert data["difference_count"] == 0

    def test_diff_nonexistent_file(self, runner, valid_config_file):
        """Test diffing with a nonexistent file."""
        result = runner.invoke(
            main,
            [
                "config-diff",
                valid_config_file,
                "/nonexistent/path.yaml",
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Help text tests
# ---------------------------------------------------------------------------


class TestHelpText:
    """Tests for help text of new commands."""

    def test_validate_config_help(self, runner):
        """Test that validate-config --help works."""
        result = runner.invoke(main, ["validate-config", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.output.lower()
        assert "--strict" in result.output
        assert "--format" in result.output

    def test_test_config_help(self, runner):
        """Test that test-config --help works."""
        result = runner.invoke(main, ["test-config", "--help"])
        assert result.exit_code == 0
        assert "test" in result.output.lower()
        assert "--builtin" in result.output

    def test_config_diff_help(self, runner):
        """Test that config-diff --help works."""
        result = runner.invoke(main, ["config-diff", "--help"])
        assert result.exit_code == 0
        assert "diff" in result.output.lower()

    def test_main_help_shows_new_commands(self, runner):
        """Test that main --help lists the new commands."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "validate-config" in result.output
        assert "test-config" in result.output
        assert "config-diff" in result.output
