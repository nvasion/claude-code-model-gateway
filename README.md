# claude-code-model-gateway

An HTTP model gateway and proxy service with multi-provider AI backend support. Routes and proxies requests to Anthropic, OpenAI, Google Gemini, and AWS Bedrock, with caching, retries, routing rules, health monitoring, and flexible configuration management.

## Installation

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

```bash
# Start the Anthropic API gateway (reads ANTHROPIC_API_KEY from environment)
claude-code-model-gateway gateway

# Start the HTTP forward proxy
claude-code-model-gateway proxy

# Show help
claude-code-model-gateway --help

# Show version
claude-code-model-gateway --version
```

## Commands

### `gateway` — Anthropic API Gateway

Starts an HTTP server that proxies requests to the Anthropic Messages API. Supports response caching, token count caching, connection pooling, and automatic retries.

```bash
# Start on default host/port (127.0.0.1:8080)
claude-code-model-gateway gateway

# Custom host and port
claude-code-model-gateway gateway --host 0.0.0.0 --port 9000

# With API key and pass-through mode (forwards all /v1/* paths)
claude-code-model-gateway gateway --api-key sk-ant-... --passthrough-mode

# Enable response caching (5-minute TTL, 256 entries)
claude-code-model-gateway gateway --response-cache --cache-ttl 300 --cache-maxsize 256

# Enable token count caching (1-hour TTL)
claude-code-model-gateway gateway --token-count-cache

# With connection pooling and retries
claude-code-model-gateway gateway --pool-size 20 --max-retries 3 --retry-delay 0.5

# With verbose JSON logging to a file
claude-code-model-gateway gateway --verbose --log-format json --log-file gateway.log
```

**Key options:**

| Option | Default | Description |
|---|---|---|
| `--host` / `-H` | `127.0.0.1` | Bind address |
| `--port` / `-p` | `8080` | Listen port |
| `--timeout` / `-t` | `300` | Upstream timeout (seconds) |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--anthropic-version` | `2023-06-01` | API version header |
| `--passthrough-mode` | off | Forward all `/v1/*` paths |
| `--max-request-size` | `10485760` | Max request body (bytes) |
| `--response-cache` | off | Cache GET/HEAD responses |
| `--cache-ttl` | `300` | Response cache TTL (seconds) |
| `--token-count-cache` | off | Cache token count responses |
| `--pool-size` | `10` | Connection pool size |
| `--max-retries` | `2` | Retry count (non-streaming) |
| `--log-format` | `standard` | `standard`, `detailed`, `json`, `colored`, `minimal` |

---

### `proxy` — HTTP Forward Proxy

Starts a simple HTTP forward proxy that relays requests to upstream servers with optional retry logic.

```bash
# Start on default host/port (127.0.0.1:3000)
claude-code-model-gateway proxy

# Custom host and port with retries
claude-code-model-gateway proxy --host 0.0.0.0 --port 8888 --max-retries 3 --retry-delay 0.5
```

---

### `config` — Configuration Management

Manage the gateway YAML/JSON configuration file.

```bash
claude-code-model-gateway config init                  # Create a default config file
claude-code-model-gateway config show                  # Display current config
claude-code-model-gateway config validate              # Validate config file
claude-code-model-gateway config set <key> <value>     # Set a top-level config value
claude-code-model-gateway config schema                # Show full config schema
claude-code-model-gateway config schema --field providers  # Show schema for a specific field
claude-code-model-gateway config lint                  # Lint config for issues
claude-code-model-gateway config doctor                # Run comprehensive health check
claude-code-model-gateway config export --format json  # Export config to JSON
claude-code-model-gateway config env-check             # Check required environment variables
```

---

### `provider` — Provider Management

List, add, remove, and configure AI model providers.

```bash
claude-code-model-gateway provider list                # List configured providers
claude-code-model-gateway provider list --builtins     # Show built-in provider templates
claude-code-model-gateway provider show anthropic      # Show provider details
claude-code-model-gateway provider add openai          # Add a provider from built-in template
claude-code-model-gateway provider remove openai       # Remove a provider
claude-code-model-gateway provider set-default anthropic  # Set default provider
claude-code-model-gateway provider enable anthropic    # Enable a provider
claude-code-model-gateway provider disable openai      # Disable a provider
claude-code-model-gateway provider update anthropic --priority 1  # Update provider settings
```

Supported built-in providers: `anthropic`, `openai`, `gemini`, `bedrock`.

---

### `cache` — Cache Management

View statistics and manage the response and token count caches.

```bash
claude-code-model-gateway cache stats                  # Show cache hit/miss statistics
claude-code-model-gateway cache stats --format json    # Output as JSON
claude-code-model-gateway cache clear                  # Clear all caches
claude-code-model-gateway cache purge                  # Purge expired cache entries
claude-code-model-gateway cache warmup                 # Pre-warm the cache
claude-code-model-gateway cache info                   # Show cache configuration
claude-code-model-gateway cache response-stats         # Response cache statistics
claude-code-model-gateway cache token-count-stats      # Token count cache statistics
```

---

### `logging` — Logging Configuration

Inspect and configure the application logging system at runtime.

```bash
claude-code-model-gateway logging status               # Show logging configuration
claude-code-model-gateway logging status --format json
claude-code-model-gateway logging test                 # Emit a test log message
claude-code-model-gateway logging formats              # List available log formats
claude-code-model-gateway logging levels               # List available log levels
claude-code-model-gateway logging metrics              # Show logging metrics
claude-code-model-gateway logging health               # Check logging system health
claude-code-model-gateway logging test-redaction       # Test log redaction of sensitive data
claude-code-model-gateway logging set-level debug      # Change log level at runtime
claude-code-model-gateway logging configure --format json --level info  # Reconfigure logging
claude-code-model-gateway logging files                # List active log files
```

---

### `health` — Health Monitoring

Monitor system health, provider status, error rates, and circuit breakers.

```bash
claude-code-model-gateway health status                # Overall system health
claude-code-model-gateway health status --format json
claude-code-model-gateway health errors                # Recent error log
claude-code-model-gateway health categories            # Errors grouped by category
claude-code-model-gateway health circuit-breakers      # Circuit breaker states
claude-code-model-gateway health reset                 # Reset error counters
claude-code-model-gateway health retry-policies        # Show retry policy configuration
```

---

### `route` — Request Routing

Manage and inspect routing rules that map model names to providers.

```bash
claude-code-model-gateway route list                   # List all routing rules
claude-code-model-gateway route list --format json
claude-code-model-gateway route resolve --model claude-sonnet-4-20250514  # Resolve a model
claude-code-model-gateway route add "claude-*" --provider anthropic       # Add a routing rule
claude-code-model-gateway route remove "claude-*"                         # Remove a rule
claude-code-model-gateway route test --model gpt-4o                       # Test routing
claude-code-model-gateway route stats                  # Show routing statistics
claude-code-model-gateway route serve                  # Start route-aware gateway
```

---

### Utility Commands

```bash
# Validate a config file directly
claude-code-model-gateway validate-config --config gateway.yaml

# Test a config (dry run, validate + connectivity checks)
claude-code-model-gateway test-config --config gateway.yaml

# Diff two config files
claude-code-model-gateway config-diff config-a.yaml config-b.yaml
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key (used by `gateway` command) |

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src

# Format code
black src tests

# Lint code
ruff check src tests

# Auto-fix lint issues
ruff check --fix src tests
```

## Project Structure

```
claude-code-model-gateway/
├── src/
│   ├── __init__.py                  # Package init with version
│   ├── main.py                      # Entry point
│   ├── cli.py                       # All CLI commands
│   ├── gateway.py                   # Gateway implementation
│   ├── proxy.py                     # HTTP proxy implementation
│   ├── anthropic_passthrough.py     # Anthropic API pass-through
│   ├── router.py                    # Request routing
│   ├── interceptor.py               # Request/response interception
│   ├── providers.py                 # Provider management
│   ├── models.py                    # Data models
│   ├── cache.py                     # Caching layer
│   ├── response_cache.py            # Response caching
│   ├── token_count_cache.py         # Token count caching
│   ├── retry.py                     # Retry logic
│   ├── retry_budget.py              # Retry budget management
│   ├── error_handling.py            # Error handling & health tracking
│   ├── error_recovery_strategies.py # Recovery strategies
│   ├── errors.py                    # Custom exceptions
│   ├── logging_config.py            # Logging configuration
│   ├── service.py                   # Daemon/service entry point
│   ├── config/                      # Configuration management
│   │   ├── loader.py
│   │   ├── manager.py
│   │   ├── validator.py
│   │   └── schema.py
│   ├── translators/                 # Per-provider API translators
│   │   ├── anthropic.py
│   │   ├── openai.py
│   │   ├── gemini.py
│   │   └── bedrock.py
│   └── validation/                  # Request validation utilities
├── tests/                           # pytest test suite (42 files)
├── service/                         # Service files (systemd, initd, openrc, launchd)
├── scripts/                         # Install/update/healthcheck scripts
├── examples/                        # Usage examples
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Service Installation

Pre-built service files are included for common init systems:

```bash
# Install as a system service (Linux/macOS)
sudo bash scripts/install.sh

# Uninstall
sudo bash scripts/uninstall.sh
```

Service files are located in `service/` for systemd, init.d, OpenRC, and launchd.

## Docker

```bash
# Build and start with Docker Compose
docker-compose up

# Or build the image directly
docker build -t claude-code-model-gateway .
docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8080:8080 claude-code-model-gateway gateway --host 0.0.0.0
```
