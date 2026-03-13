# Claude Code Model Gateway

A drop-in proxy that sits between **Claude Code** and AI providers, giving you cost control, provider redundancy, and unified routing — without changing a line of your Claude Code workflow.

## Why Use This with Claude Code?

Claude Code communicates with the Anthropic Messages API. This gateway speaks the same protocol, so you point Claude Code at the gateway instead of directly at Anthropic. From there, the gateway can:

- **Reduce costs** — route lower-priority requests to cheaper providers (GPT-4o, Gemini) while keeping critical work on Claude
- **Eliminate rate-limit disruptions** — automatically fail over to a secondary provider when Anthropic returns 429s
- **Cache repeated calls** — response and token-count caches absorb duplicate requests at zero API cost
- **Unify audit logging** — every request flowing through Claude Code lands in one structured log, regardless of provider
- **Enforce budgets** — set per-client rate limits so a runaway agent doesn't burn your quota

Supported upstream providers: **Anthropic**, **OpenAI**, **Google Gemini**, **AWS Bedrock**.

---

## Quick Start with Claude Code

### 1. Install the gateway

```bash
pip install -e .
```

### 2. Configure providers

```bash
# Create a starter config
claude-code-model-gateway config init

# Add your providers
claude-code-model-gateway provider add anthropic   # reads ANTHROPIC_API_KEY
claude-code-model-gateway provider add openai      # reads OPENAI_API_KEY
claude-code-model-gateway provider add gemini      # reads GEMINI_API_KEY

# Set Anthropic as default, OpenAI as fallback
claude-code-model-gateway provider set-default anthropic
```

### 3. Add routing rules

```bash
# All claude-* models go to Anthropic; everything else to OpenAI
claude-code-model-gateway route add "claude-*" --provider anthropic
claude-code-model-gateway route add "*"        --provider openai
```

### 4. Start the gateway

```bash
# Listens on 127.0.0.1:8080 by default
claude-code-model-gateway gateway --response-cache --token-count-cache
```

### 5. Point Claude Code at the gateway

Set the `ANTHROPIC_BASE_URL` environment variable before launching Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
claude  # Claude Code now routes through the gateway
```

Or add it to your shell profile so it's always active:

```bash
echo 'export ANTHROPIC_BASE_URL=http://127.0.0.1:8080' >> ~/.zshrc
```

Claude Code's requests arrive at the gateway in Anthropic API format. The gateway resolves the target provider from your routing rules, translates the request, forwards it, translates the response back, and returns it to Claude Code — all transparently.

---

## Typical Developer Workflows

### Cost optimisation: expensive model for code, cheap model for summaries

```bash
claude-code-model-gateway route add "claude-opus-*"   --provider anthropic
claude-code-model-gateway route add "claude-sonnet-*" --provider anthropic
claude-code-model-gateway route add "gpt-4o-mini"     --provider openai
claude-code-model-gateway route add "*"               --provider gemini
```

### Resilience: fall back to OpenAI when Anthropic is rate-limited

The gateway retries on 429/502/503/504 before surfacing an error:

```bash
claude-code-model-gateway gateway --max-retries 3 --retry-delay 1.0
```

### Caching: avoid re-paying for identical requests

```bash
claude-code-model-gateway gateway \
  --response-cache --cache-ttl 600 --cache-maxsize 512 \
  --token-count-cache
```

### Observability: structured JSON logs for every Claude Code request

```bash
claude-code-model-gateway gateway \
  --log-format json --log-file /var/log/claude-gateway.log --verbose
```

### Running as a persistent background service

```bash
# Install as a systemd / launchd / OpenRC service
sudo bash scripts/install.sh

# Or run in Docker
docker-compose up -d
```

---

## Configuration Reference

### Environment variables

| Variable | Required for | Description |
|---|---|---|
| `ANTHROPIC_BASE_URL` | Claude Code | Point Claude Code at the gateway (`http://127.0.0.1:8080`) |
| `ANTHROPIC_API_KEY` | Anthropic provider | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI provider | OpenAI API key |
| `GEMINI_API_KEY` | Gemini provider | Google Gemini API key |
| `AWS_ACCESS_KEY_ID` | Bedrock provider | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Bedrock provider | AWS secret key |
| `AWS_DEFAULT_REGION` | Bedrock provider | AWS region (e.g. `us-east-1`) |
| `GATEWAY_DEFAULT_PROVIDER` | Gateway runtime | Override default provider |
| `GATEWAY_LOG_LEVEL` | Gateway runtime | Override log level |
| `GATEWAY_TIMEOUT` | Gateway runtime | Override request timeout (seconds) |

Run `claude-code-model-gateway config env-check` to verify all required variables are set for your configured providers.

### Example `gateway.yaml`

```yaml
default_provider: anthropic
log_level: info
timeout: 120
max_retries: 3

providers:
  anthropic:
    name: anthropic
    api_base: https://api.anthropic.com/v1
    api_key_env_var: ANTHROPIC_API_KEY
    enabled: true

  openai:
    name: openai
    api_base: https://api.openai.com/v1
    api_key_env_var: OPENAI_API_KEY
    enabled: true

  gemini:
    name: gemini
    api_base: https://generativelanguage.googleapis.com/v1beta
    api_key_env_var: GEMINI_API_KEY
    enabled: true
```

Pass the config file to any command with `--config gateway.yaml`.

---

## All Commands

### `gateway` — Multi-provider routing gateway

Starts an HTTP server that accepts Anthropic Messages API requests and routes them to the configured upstream providers.

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

Internal endpoints (answered by the gateway itself, not forwarded):

- `GET /health` — returns `{"status": "ok"}`
- `GET /status` — returns gateway statistics and provider counts

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

```bash
claude-code-model-gateway config init                       # Create a default config file
claude-code-model-gateway config show                       # Display current config
claude-code-model-gateway config validate                   # Validate config file
claude-code-model-gateway config set <key> <value>          # Set a top-level config value
claude-code-model-gateway config schema                     # Show full config schema
claude-code-model-gateway config schema --field providers   # Show schema for a specific field
claude-code-model-gateway config lint                       # Lint config for issues
claude-code-model-gateway config doctor                     # Run comprehensive health check
claude-code-model-gateway config export --format json       # Export config to JSON
claude-code-model-gateway config env-check                  # Check required environment variables
```

---

### `provider` — Provider Management

```bash
claude-code-model-gateway provider list                        # List configured providers
claude-code-model-gateway provider list --builtins             # Show built-in provider templates
claude-code-model-gateway provider show anthropic              # Show provider details
claude-code-model-gateway provider add openai                  # Add a provider from built-in template
claude-code-model-gateway provider remove openai               # Remove a provider
claude-code-model-gateway provider set-default anthropic       # Set default provider
claude-code-model-gateway provider enable anthropic            # Enable a provider
claude-code-model-gateway provider disable openai              # Disable a provider
claude-code-model-gateway provider update anthropic --priority 1  # Update provider settings
```

Supported built-in providers: `anthropic`, `openai`, `gemini`, `bedrock`.

---

### `route` — Request Routing

```bash
claude-code-model-gateway route list                                       # List all routing rules
claude-code-model-gateway route list --format json
claude-code-model-gateway route resolve --model claude-sonnet-4-20250514   # Resolve a model to a provider
claude-code-model-gateway route add "claude-*" --provider anthropic        # Add a routing rule
claude-code-model-gateway route remove "claude-*"                          # Remove a rule
claude-code-model-gateway route test --model gpt-4o                        # Test routing
claude-code-model-gateway route stats                                       # Show routing statistics
claude-code-model-gateway route serve                                       # Start route-aware gateway
```

---

### `cache` — Cache Management

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

### `health` — Health Monitoring

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

### `logging` — Logging Configuration

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

## Installation

```bash
# Install in development mode (recommended for local use)
pip install -e .

# Install with development/test dependencies
pip install -e ".[dev]"

# Or install runtime dependencies only
pip install -r requirements.txt
```

> **Already covered in Quick Start?** Steps 1–4 of [Quick Start with Claude Code](#quick-start-with-claude-code) walk through a complete first-time install. This section is a condensed reference.

## Docker

```bash
# Build and start with Docker Compose
docker-compose up

# Or build the image directly
docker build -t claude-code-model-gateway .
docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8080:8080 claude-code-model-gateway gateway --host 0.0.0.0
```

## Service Installation

Pre-built service files are included for the four most common init systems:

| Init system | Platform | File location |
|---|---|---|
| **systemd** | Linux (Debian, Ubuntu, Fedora, Arch…) | `service/systemd/` |
| **init.d** | Linux (SysV / older distros) | `service/initd/` |
| **OpenRC** | Linux (Alpine, Gentoo…) | `service/openrc/` |
| **launchd** | macOS | `service/launchd/` |

```bash
# Detect the init system and install automatically
sudo bash scripts/install.sh

# Uninstall
sudo bash scripts/uninstall.sh
```

---

## Troubleshooting

### Claude Code is not using the gateway

Verify the environment variable is exported in the same shell where you launch Claude Code:

```bash
echo $ANTHROPIC_BASE_URL   # should print http://127.0.0.1:8080
curl http://127.0.0.1:8080/health  # should return {"status":"ok"}
```

If `ANTHROPIC_BASE_URL` is empty, re-export it or add it to your shell profile (see Quick Start step 5).

### Gateway returns `connection refused`

The gateway process is not running. Start it with:

```bash
claude-code-model-gateway gateway
```

Or check that the port is not already in use:

```bash
lsof -i :8080
```

### Provider returns 401 Unauthorized

The API key for the target provider is missing or incorrect. Verify it is set:

```bash
claude-code-model-gateway config env-check
```

### Requests are not being cached

Caching must be explicitly enabled when starting the gateway:

```bash
claude-code-model-gateway gateway --response-cache --token-count-cache
```

Check cache hit rates with `claude-code-model-gateway cache stats`.

### Rate limits / 429 errors still surfacing

Increase the retry count and ensure a fallback provider is configured:

```bash
claude-code-model-gateway gateway --max-retries 5 --retry-delay 2.0
claude-code-model-gateway route add "*" --provider openai   # fallback
```

### Config file not found

By default the gateway looks for `gateway.yaml` in the current directory. Pass an explicit path with `--config`:

```bash
claude-code-model-gateway gateway --config /path/to/gateway.yaml
```

Or generate a starter config in the current directory:

```bash
claude-code-model-gateway config init
```

---

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
│   ├── gateway.py                   # Multi-provider routing gateway
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
