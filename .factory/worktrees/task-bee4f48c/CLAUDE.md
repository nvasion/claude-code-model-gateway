# CLAUDE.md

This file provides guidance to Claude Code when working on this project.

## Project Overview

claude-code-model-gateway is a command-line application built with Python using the Click framework.

### Persona 
Senior software engineer/architect who reads and refactors existing code before creating new                                                                                                                                                                      
#### Core Principles:
  - Read Before You Write — understand existing code, check for duplicates, refactor over recreate
  - Test Everything — tests for every feature, run the full suite, regression tests for bug fixes
  - Keep Documentation Current — README.md always reflects current state, document all changes
  - Code Quality — simple readable code, follow established conventions, lint after every change, explicit error handling
  - No Unnecessary Complexity — no over-engineering, no premature abstractions, no unrequested features

## README Maintenance (MANDATORY)

**After every code change, always update `README.md` to reflect the current state.**

This is non-negotiable. Do not complete any task without verifying `README.md` is accurate:

- New CLI command or option added → update CLI Reference table and Quick Start examples
- New/changed module → update Project Structure section
- New/changed configuration key or env var → update Configuration section
- New dependency → update Requirements section
- Behaviour change → update any affected examples or descriptions

The README is the source of truth for users and must never lag behind the code.

## Development Commands

### Installation
```bash
pip install -e .              # Install in editable mode
pip install -e ".[dev]"       # Install with dev dependencies
pip install -r requirements.txt  # Install from requirements
```

### Testing
```bash
pytest                        # Run all tests
pytest -v                     # Run with verbose output
pytest --cov=src              # Run with coverage
pytest -k test_name           # Run specific test
```

### Code Quality
```bash
black src tests               # Format code
ruff check src tests          # Lint code
ruff check --fix src tests    # Auto-fix lint issues
```

### Running
```bash
python -m src.main            # Run from module
claude-code-model-gateway              # Run installed command
```

## Architecture

Full module layout — read all relevant files before editing:

```
src/
├── __init__.py              # Package version
├── main.py                  # CLI entry point (calls cli.main)
├── cli.py                   # Click command definitions (gateway, config, validate, …)
├── service.py               # Service daemon (SIGTERM/SIGHUP handling)
├── proxy.py                 # HTTP proxy / reverse-proxy server
├── anthropic_passthrough.py # Anthropic-specific passthrough logic
├── providers.py             # Built-in provider definitions (Anthropic, OpenAI, Azure)
├── models.py                # GatewayConfig, ProviderConfig, ModelConfig dataclasses
├── errors.py                # Typed error hierarchy with retryable classification
├── retry.py                 # Retry logic & backoff strategies
├── cache.py                 # Thread-safe LRU cache with TTL
├── logging_config.py        # Structured logging (standard/detailed/json/colored)
├── config/                  # Configuration loading & validation sub-package
│   ├── __init__.py
│   ├── loader.py
│   ├── schema.py
│   ├── validator.py
│   └── testing.py
└── validation/              # Request validation sub-package
    ├── __init__.py
    ├── validator.py
    └── testing.py
tests/                       # pytest test suite (mirrors src/ structure)
service/                     # systemd / init.d service definitions + conf templates
scripts/                     # install.sh, uninstall.sh, healthcheck.sh
```

## Code Style

- Follow PEP 8 conventions
- Use `black` for formatting (line length: 88)
- Use `ruff` for linting
- Type hints are encouraged
- Add docstrings to all functions and commands
- Keep functions small and focused
- Add tests for new functionality

## Adding New Commands

1. Open `src/cli.py`
2. Add a new function with `@main.command()` decorator
3. Use Click decorators for arguments and options
4. Add tests in `tests/test_cli.py`

Example:
```python
@main.command()
@click.argument("filename")
@click.option("--output", "-o", default="output.txt", help="Output file")
def process(filename: str, output: str):
    """Process a file and write results.

    FILENAME is the input file to process.
    """
    click.echo(f"Processing {filename} -> {output}")
```

## Dependencies

Runtime (`pyproject.toml > dependencies`):
- `click >= 8.1.0` - CLI framework
- `pyyaml >= 6.0` - YAML configuration parsing

Dev (`pyproject.toml > optional-dependencies > dev`):
- `pytest` / `pytest-cov` - Testing framework & coverage
- `black` - Code formatter (line length: 88)
- `ruff` - Fast Python linter
- `build` - Package build tool

Add new dependencies:
```bash
pip install package-name
# Then add to pyproject.toml [project.dependencies] (runtime)
# or [project.optional-dependencies] dev (dev-only)
```

## Testing

Write tests in `tests/` directory with `test_` prefix.

Example:
```python
from click.testing import CliRunner
from src.cli import main

def test_my_command():
    runner = CliRunner()
    result = runner.invoke(main, ["mycommand", "arg"])
    assert result.exit_code == 0
    assert "expected output" in result.output
```

## Generated by AI Project Factory

This project was initialized by AI Project Factory.
Modify this file to add project-specific guidance for Claude Code.
