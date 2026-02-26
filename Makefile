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
#     make build          Build distribution packages
#     make clean          Remove build artifacts
#
#   Service management:
#     make install        Install as system service (requires sudo)
#     make uninstall      Remove system service (requires sudo)
#     make start          Start the service
#     make stop           Stop the service
#     make restart        Restart the service
#     make status         Show service status
#     make logs           Tail service logs
#     make healthcheck    Run health check
#
# ===========================================================================

.PHONY: help install-dev test lint format build clean \
        install uninstall start stop restart status logs healthcheck

SHELL := /bin/bash
PYTHON ?= python3
PIP ?= pip
PROJECT_NAME := claude-code-model-gateway
SERVICE_NAME := claude-code-model-gateway
INSTALL_PREFIX ?= /opt/claude-code-model-gateway
CONFIG_DIR ?= /etc/claude-code-model-gateway
LOG_DIR ?= /var/log/claude-code-model-gateway

# Detect init system
HAS_SYSTEMD := $(shell command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ] && echo true || echo false)

# Default target
help:
	@echo ""
	@echo "claude-code-model-gateway — Build & Service Management"
	@echo "======================================================="
	@echo ""
	@echo "Development:"
	@echo "  make install-dev    Install in editable mode with dev deps"
	@echo "  make test           Run test suite"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make lint           Run ruff linter"
	@echo "  make format         Format code with black"
	@echo "  make build          Build sdist + wheel"
	@echo "  make clean          Remove build artifacts"
	@echo ""
	@echo "Service (requires sudo):"
	@echo "  make install        Install as system service"
	@echo "  make uninstall      Remove system service"
	@echo "  make start          Start the service"
	@echo "  make stop           Stop the service"
	@echo "  make restart        Restart the service"
	@echo "  make reload         Reload service configuration"
	@echo "  make status         Show service status"
	@echo "  make logs           Tail service logs (Ctrl+C to stop)"
	@echo "  make healthcheck    Run health check against the service"
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

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Service targets (require root)
# ---------------------------------------------------------------------------

install:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Error: 'make install' must be run as root (use sudo)."; \
		exit 1; \
	fi
	bash scripts/install.sh

uninstall:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Error: 'make uninstall' must be run as root (use sudo)."; \
		exit 1; \
	fi
	bash scripts/uninstall.sh

start:
ifeq ($(HAS_SYSTEMD),true)
	systemctl start $(SERVICE_NAME)
else
	service $(SERVICE_NAME) start
endif

stop:
ifeq ($(HAS_SYSTEMD),true)
	systemctl stop $(SERVICE_NAME)
else
	service $(SERVICE_NAME) stop
endif

restart:
ifeq ($(HAS_SYSTEMD),true)
	systemctl restart $(SERVICE_NAME)
else
	service $(SERVICE_NAME) restart
endif

reload:
ifeq ($(HAS_SYSTEMD),true)
	systemctl reload $(SERVICE_NAME)
else
	service $(SERVICE_NAME) reload
endif

status:
ifeq ($(HAS_SYSTEMD),true)
	systemctl status $(SERVICE_NAME) --no-pager
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
