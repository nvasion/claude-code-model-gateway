"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner

from src.cli import main


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


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
