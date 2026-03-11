# ===========================================================================
# Makefile — claude-code-model-gateway build & service management
# ===========================================================================
#
# Targets:
#   Development:
#     make install-dev    Install package in dev/editable mode
#     make test           Run test suite
#     make lint           Run linters
#     make format         Auto-format code
#     make build          Build distribution packages (sdist + wheel)
#     make package        Build a self-contained distributable .tar.gz
#     make clean          Remove build artifacts
#
#   Service management (Linux — requires sudo):
#     make install        Install as system service (auto-detects systemd/SysV/OpenRC)
#     make install-launchd  Install as macOS launchd service (macOS, requires sudo)
#     make install-openrc   Install as OpenRC service (Alpine/Gentoo, requires sudo)
#     make uninstall      Remove system service (requires sudo)
#     make update         Upgrade an existing installation (requires sudo)
#     make start          Start the service
#     make stop           Stop the service
#     make restart        Restart the service
#     make reload         Reload service configuration (SIGHUP)
#     make status         Show service status
#     make logs           Tail service logs (Ctrl+C to stop)
#     make healthcheck    Run health check against the running service
#
#   Docker:
#     make docker-build   Build the Docker image
#     make docker-run     Run the gateway in a Docker container
#     make docker-stop    Stop and remove the gateway container
#     make docker-logs    Tail Docker container logs
#
# ===========================================================================

.PHONY: help install-dev test test-cov lint format build package clean \
        install install-launchd install-openrc uninstall update \
        start stop restart reload status logs healthcheck \
        docker-build docker-run docker-stop docker-logs

SHELL := /bin/bash
PYTHON  ?= python3
PIP     ?= pip
PROJECT_NAME  := claude-code-model-gateway
SERVICE_NAME  := claude-code-model-gateway
INSTALL_PREFIX ?= /opt/claude-code-model-gateway
CONFIG_DIR    ?= /etc/claude-code-model-gateway
LOG_DIR       ?= /var/log/claude-code-model-gateway
DOCKER_IMAGE  ?= claude-code-model-gateway
DOCKER_TAG    ?= latest
GATEWAY_PORT  ?= 8080

# Detect init system (Linux)
HAS_SYSTEMD := $(shell command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ] && echo true || echo false)
HAS_OPENRC  := $(shell command -v rc-service >/dev/null 2>&1 && echo true || echo false)
IS_MACOS    := $(shell [ "$$(uname -s)" = "Darwin" ] && echo true || echo false)

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------

help:
	@echo ""
	@echo "claude-code-model-gateway — Build & Service Management"
	@echo "======================================================="
	@echo ""
	@echo "Development:"
	@echo "  make install-dev      Install in editable mode with dev deps"
	@echo "  make test             Run test suite"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Format code with black"
	@echo "  make build            Build sdist + wheel"
	@echo "  make package          Build distributable .tar.gz archive"
	@echo "  make clean            Remove build artifacts"
	@echo ""
	@echo "Service — Linux (requires sudo):"
	@echo "  make install          Install as system service (auto-detects init)"
	@echo "  make install-openrc   Install as OpenRC service (Alpine/Gentoo)"
	@echo "  make uninstall        Remove system service"
	@echo "  make update           Upgrade an existing installation"
	@echo "  make start            Start the service"
	@echo "  make stop             Stop the service"
	@echo "  make restart          Restart the service"
	@echo "  make reload           Reload configuration (SIGHUP)"
	@echo "  make status           Show service status"
	@echo "  make logs             Tail service logs (Ctrl+C to stop)"
	@echo "  make healthcheck      Run health check against the service"
	@echo ""
	@echo "Service — macOS (requires sudo):"
	@echo "  make install-launchd  Install as launchd daemon"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker image ($(DOCKER_IMAGE):$(DOCKER_TAG))"
	@echo "  make docker-run       Run gateway container on port $(GATEWAY_PORT)"
	@echo "  make docker-stop      Stop and remove gateway container"
	@echo "  make docker-logs      Tail container logs"
	@echo ""

# ---------------------------------------------------------------------------
# Development targets
# ---------------------------------------------------------------------------

install-dev:
	$(PIP) install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check src tests

format:
	black src tests

build: clean
	$(PYTHON) -m build

# Build a self-contained distributable .tar.gz archive
package:
	bash scripts/build-package.sh

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Service targets — Linux (require root)
# ---------------------------------------------------------------------------

install:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Error: 'make install' must be run as root (use sudo)."; \
		exit 1; \
	fi
	bash scripts/install.sh

install-openrc:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Error: 'make install-openrc' must be run as root (use sudo)."; \
		exit 1; \
	fi
	bash scripts/install.sh --openrc

uninstall:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Error: 'make uninstall' must be run as root (use sudo)."; \
		exit 1; \
	fi
	bash scripts/uninstall.sh

update:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Error: 'make update' must be run as root (use sudo)."; \
		exit 1; \
	fi
	bash scripts/update.sh

# ---------------------------------------------------------------------------
# Service targets — macOS (require root)
# ---------------------------------------------------------------------------

install-launchd:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Error: 'make install-launchd' must be run as root (use sudo)."; \
		exit 1; \
	fi
	bash scripts/install.sh --launchd

# ---------------------------------------------------------------------------
# Start / stop / status (works for systemd, OpenRC, SysV, and macOS)
# ---------------------------------------------------------------------------

start:
ifeq ($(HAS_SYSTEMD),true)
	systemctl start $(SERVICE_NAME)
else ifeq ($(IS_MACOS),true)
	launchctl kickstart system/com.anthropic.$(SERVICE_NAME)
else ifeq ($(HAS_OPENRC),true)
	rc-service $(SERVICE_NAME) start
else
	service $(SERVICE_NAME) start
endif

stop:
ifeq ($(HAS_SYSTEMD),true)
	systemctl stop $(SERVICE_NAME)
else ifeq ($(IS_MACOS),true)
	launchctl kill TERM system/com.anthropic.$(SERVICE_NAME)
else ifeq ($(HAS_OPENRC),true)
	rc-service $(SERVICE_NAME) stop
else
	service $(SERVICE_NAME) stop
endif

restart:
ifeq ($(HAS_SYSTEMD),true)
	systemctl restart $(SERVICE_NAME)
else ifeq ($(IS_MACOS),true)
	launchctl kickstart -k system/com.anthropic.$(SERVICE_NAME)
else ifeq ($(HAS_OPENRC),true)
	rc-service $(SERVICE_NAME) restart
else
	service $(SERVICE_NAME) restart
endif

reload:
ifeq ($(HAS_SYSTEMD),true)
	systemctl reload $(SERVICE_NAME)
else ifeq ($(IS_MACOS),true)
	launchctl kill HUP system/com.anthropic.$(SERVICE_NAME)
else ifeq ($(HAS_OPENRC),true)
	rc-service $(SERVICE_NAME) reload
else
	service $(SERVICE_NAME) reload
endif

status:
ifeq ($(HAS_SYSTEMD),true)
	systemctl status $(SERVICE_NAME) --no-pager
else ifeq ($(IS_MACOS),true)
	launchctl print system/com.anthropic.$(SERVICE_NAME)
else ifeq ($(HAS_OPENRC),true)
	rc-service $(SERVICE_NAME) status
else
	service $(SERVICE_NAME) status
endif

logs:
ifeq ($(HAS_SYSTEMD),true)
	journalctl -u $(SERVICE_NAME) -f
else
	tail -f $(LOG_DIR)/gateway.log
endif

healthcheck:
	bash scripts/healthcheck.sh

# ---------------------------------------------------------------------------
# Docker targets
# ---------------------------------------------------------------------------

docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker run -d \
		--name $(SERVICE_NAME) \
		--restart unless-stopped \
		-p $(GATEWAY_PORT):8080 \
		-e GATEWAY_HOST=0.0.0.0 \
		-e GATEWAY_PORT=8080 \
		-e GATEWAY_LOG_FORMAT=json \
		-e GATEWAY_LOG_LEVEL=info \
		-e ANTHROPIC_API_KEY=$${ANTHROPIC_API_KEY:-} \
		-v $(SERVICE_NAME)-logs:/var/log/claude-code-model-gateway \
		-v $(SERVICE_NAME)-data:/var/lib/claude-code-model-gateway \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "Gateway running on http://localhost:$(GATEWAY_PORT)"
	@echo "Stop with: make docker-stop"

docker-stop:
	docker stop $(SERVICE_NAME) 2>/dev/null || true
	docker rm   $(SERVICE_NAME) 2>/dev/null || true

docker-logs:
	docker logs -f $(SERVICE_NAME)
