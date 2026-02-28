# claude-code-model-gateway

An HTTP model gateway and proxy service with multi-provider AI backend support. Routes requests to AI providers (Anthropic, OpenAI, Azure OpenAI, and custom endpoints) with production-grade reliability features: retry logic, caching, structured logging, and graceful error handling.

## Features

- **Multi-provider support** — Built-in configurations for Anthropic, OpenAI, and Azure OpenAI; add custom providers via YAML
- **Retry & resilience** — Exponential, linear, and constant backoff strategies with circuit breaker support
- **Request caching** — Thread-safe LRU cache with TTL for upstream responses
- **Structured logging** — Standard, detailed, JSON, and colored log formats with per-request correlation IDs
- **Configuration management** — YAML/JSON config files with environment variable overrides
- **System service integration** — Ships with systemd and SysV init.d service definitions
- **Comprehensive error handling** — Typed error hierarchy with retryable classification and HTTP status mapping

## Requirements

- Python 3.11+
- `click >= 8.1.0`
- `pyyaml >= 6.0`

## Installation

```bash
# Install in editable mode (development)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install from requirements file
pip install -r requirements.txt
```

## Quick Start

```bash
# Start the gateway on the default address (127.0.0.1:8080)
claude-code-model-gateway gateway

# Start on a custom port
claude-code-model-gateway gateway --port 9090

# Start with an explicit API key
claude-code-model-gateway gateway --api-key sk-ant-...

# Show help
claude-code-model-gateway --help
```

## Using with Claude Code

The gateway acts as a local reverse proxy to the Anthropic API, letting Claude Code's traffic flow through it for logging, auditing, retry handling, and more.

### 1. Start the gateway

```bash
# Uses ANTHROPIC_API_KEY from environment automatically
claude-code-model-gateway gateway

# Or pass the key explicitly
claude-code-model-gateway gateway --api-key sk-ant-...

# Verbose mode to see every request/response
claude-code-model-gateway gateway --verbose

# Custom port if 8080 is taken
claude-code-model-gateway gateway --port 9090
```

### 2. Point Claude Code at the gateway

Set `ANTHROPIC_BASE_URL` to the gateway address before launching Claude Code:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude
```

Or export it for your shell session so every `claude` invocation uses the gateway:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
claude
```

### What gets proxied

The gateway transparently forwards all Anthropic API endpoints Claude Code uses:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/messages` | POST | Create a message (streaming and non-streaming) |
| `/v1/messages/count_tokens` | POST | Count tokens for a request |
| `/v1/models` | GET | List available models |

### One-liner (start gateway + Claude Code together)

```bash
claude-code-model-gateway gateway & ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude
```

### Useful gateway options for Claude Code

| Option | Purpose |
|---|---|
| `--verbose` / `-v` | Log every request/response for debugging |
| `--timeout 600` | Increase timeout for long-running tasks (default: 300s) |
| `--max-retries 5` | More retries on transient API errors |
| `--log-format json` | Structured JSON logs for log aggregation |
| `--log-file gateway.log` | Write logs to a file alongside console output |

Example with all options relevant to Claude Code:

```bash
claude-code-model-gateway gateway \
  --port 8080 \
  --timeout 600 \
  --max-retries 3 \
  --verbose \
  --log-format colored \
  --log-file ~/.local/share/claude-gateway/gateway.log
```

## CLI Reference

### `gateway` — Start the Anthropic pass-through server

```
claude-code-model-gateway gateway [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--host` | `-H` | `127.0.0.1` | Host address to bind |
| `--port` | `-p` | `8080` | Port to listen on |
| `--timeout` | `-t` | `300` | Upstream connection timeout (seconds) |
| `--api-key` | | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--anthropic-version` | | `2023-06-01` | Anthropic API version header |
| `--max-retries` | | `2` | Retry attempts for failed non-streaming requests |
| `--retry-delay` | | `1.0` | Base delay (seconds) between retries |
| `--verbose` | `-v` | `false` | Enable DEBUG logging |
| `--log-format` | | `standard` | Log format: `standard`, `detailed`, `json`, `colored`, `minimal` |
| `--log-file` | | | Path to log file (enables file logging) |

### `validate` — Validate a configuration file

```bash
claude-code-model-gateway validate --config gateway.yaml
```

### `config` — Manage configuration

```bash
# Initialise a new configuration file
claude-code-model-gateway config init

# Display the active configuration
claude-code-model-gateway config show

# Validate the active configuration
claude-code-model-gateway config validate

# List available providers
claude-code-model-gateway config providers

# List models for a specific provider
claude-code-model-gateway config models --provider anthropic
```

### Utility commands

```bash
# Print a greeting
claude-code-model-gateway hello

# Greet a specific name
claude-code-model-gateway greet Alice

# Show application version
claude-code-model-gateway version
claude-code-model-gateway --version
```

## Configuration

The gateway looks for a configuration file in the following locations (in order):

1. Path passed via `--config` / `GATEWAY_CONFIG` environment variable
2. `./gateway.yaml` / `./gateway.yml` / `./gateway.json`
3. `~/.config/claude-code-model-gateway/gateway.yaml`

### Example `gateway.yaml`

```yaml
# Default provider when none is specified in the request
default_provider: anthropic

log_level: info
timeout: 300
max_retries: 3

providers:
  anthropic:
    enabled: true
    api_base: "https://api.anthropic.com"
    api_key_env_var: "ANTHROPIC_API_KEY"
    models:
      - name: "claude-sonnet-4-20250514"
        display_name: "Claude Sonnet 4"
        max_tokens: 8192
        supports_streaming: true
        supports_tools: true
        supports_vision: true

  openai:
    enabled: true
    api_base: "https://api.openai.com/v1"
    api_key_env_var: "OPENAI_API_KEY"

  # Custom / self-hosted endpoint
  custom-provider:
    enabled: true
    api_base: "https://my-endpoint.example.com/v1"
    api_key_env_var: "CUSTOM_API_KEY"
    auth_type: "bearer_token"
    models:
      - name: "my-model"
        display_name: "My Custom Model"
        max_tokens: 4096
```

### Environment variable overrides

All configuration values can be overridden at runtime with environment variables:

| Variable | Description |
|---|---|
| `GATEWAY_HOST` | Bind address |
| `GATEWAY_PORT` | Listen port |
| `GATEWAY_TIMEOUT` | Upstream timeout (seconds) |
| `GATEWAY_LOG_LEVEL` | Log level |
| `GATEWAY_LOG_FORMAT` | Log format |
| `GATEWAY_LOG_FILE` | Log file path |
| `GATEWAY_CONFIG` | Path to configuration file |
| `GATEWAY_DEFAULT_PROVIDER` | Default provider name |
| `GATEWAY_MAX_RETRIES` | Max retry attempts |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |

## Built-in Providers

## Example Models
| Provider | Models |
|---|---|
| **Anthropic** | Claude Sonnet 4.6, Claude 4.6 Opus |
| **OpenAI** | gpt-4o, gpt-4-turbo, o1, o3-mini |
| **Azure OpenAI** | Configurable Azure deployment endpoint |

## Running as a System Service

The package ships with service definitions for both systemd and SysV init.d.

### systemd

```bash
# Copy service files
sudo cp service/systemd/claude-code-model-gateway.service /etc/systemd/system/
sudo cp service/systemd/claude-code-model-gateway.sysusers /usr/lib/sysusers.d/
sudo cp service/systemd/claude-code-model-gateway.tmpfiles /usr/lib/tmpfiles.d/

# Copy configuration
sudo mkdir -p /etc/claude-code-model-gateway
sudo cp service/conf/gateway.yaml /etc/claude-code-model-gateway/gateway.yaml
sudo cp service/conf/environment  /etc/claude-code-model-gateway/environment

# Add API keys (protect this file — it contains secrets)
sudo chmod 640 /etc/claude-code-model-gateway/environment
sudo nano /etc/claude-code-model-gateway/environment

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now claude-code-model-gateway
```

The service can also be installed and uninstalled with the provided scripts:

```bash
sudo bash scripts/install.sh
sudo bash scripts/uninstall.sh
```

### Service entry point

A dedicated service entry point is registered separately from the CLI:

```bash
claude-code-model-gateway-service   # runs src/service.py
```

It reads all settings from environment variables and handles `SIGTERM`, `SIGINT`, and `SIGHUP` for graceful shutdown and configuration reload.

## Project Structure

```
claude-code-model-gateway/
├── src/
│   ├── __init__.py              # Package version
│   ├── main.py                  # CLI entry point
│   ├── cli.py                   # Click command definitions
│   ├── service.py               # Service daemon (SIGTERM/SIGHUP handling)
│   ├── proxy.py                 # HTTP proxy server
│   ├── providers.py             # Built-in provider definitions
│   ├── models.py                # GatewayConfig, ProviderConfig, ModelConfig
│   ├── errors.py                # Typed error hierarchy
│   ├── retry.py                 # Retry logic & backoff strategies
│   ├── cache.py                 # LRU cache with TTL
│   ├── logging_config.py        # Structured logging setup
│   ├── anthropic_passthrough.py # Anthropic-specific passthrough logic
│   ├── config/                  # Configuration loading & validation
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── schema.py
│   │   ├── validator.py
│   │   └── testing.py
│   └── validation/              # Request validation
│       ├── __init__.py
│       ├── validator.py
│       └── testing.py
├── tests/                       # pytest test suite
│   ├── test_cli.py
│   ├── test_cli_config.py
│   ├── test_cli_validation.py
│   ├── test_cache.py
│   ├── test_config.py
│   ├── test_errors.py
│   ├── test_logging.py
│   ├── test_models.py
│   ├── test_proxy.py
│   ├── test_providers.py
│   ├── test_retry.py
│   ├── test_service.py
│   ├── test_validator.py
│   └── ...
├── service/
│   ├── conf/
│   │   ├── gateway.yaml         # Configuration template
│   │   └── environment          # Environment variable template
│   ├── systemd/                 # systemd unit files
│   └── initd/                   # SysV init.d script
├── scripts/
│   ├── install.sh
│   ├── uninstall.sh
│   └── healthcheck.sh
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src

# Run a specific test
pytest -k test_name

# Format code
black src tests

# Lint
ruff check src tests

# Auto-fix lint issues
ruff check --fix src tests
```

### Writing tests

Tests use `pytest` and Click's `CliRunner`:

```python
from click.testing import CliRunner
from src.cli import main

def test_my_command():
    runner = CliRunner()
    result = runner.invoke(main, ["gateway", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
```

## License

MIT
