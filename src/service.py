"""Service entry point for running claude-code-model-gateway as a daemon.

This module provides a dedicated entry point for running the gateway as a
long-lived system service.  It handles:

  - Signal management (SIGTERM, SIGINT, SIGHUP)
  - PID file creation and cleanup
  - Graceful shutdown
  - Health-check endpoint
  - Structured startup logging

Usage (direct):
    python -m src.service

Usage (via setuptools entry point):
    claude-code-model-gateway-service

The service reads configuration from environment variables and/or the
standard gateway configuration file.
"""

import os
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from src import __version__
from src.logging_config import (
    get_logger,
    setup_logging,
)


logger = get_logger("service")


class ServiceManager:
    """Manages the lifecycle of the gateway service daemon.

    Handles signal registration, PID file management, and coordinated
    startup/shutdown of the gateway and health-check components.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        timeout: int = 300,
        log_level: str = "info",
        log_format: str = "json",
        log_file: Optional[str] = None,
        pid_file: Optional[str] = None,
        config_file: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.log_level = log_level
        self.log_format = log_format
        self.log_file = log_file
        self.pid_file = pid_file
        self.config_file = config_file
        self._shutdown_event = threading.Event()
        self._server = None

    # ------------------------------------------------------------------ #
    # PID file
    # ------------------------------------------------------------------ #

    def _write_pid_file(self) -> None:
        """Write the current PID to the configured PID file."""
        if not self.pid_file:
            return
        try:
            pid_path = Path(self.pid_file)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text(str(os.getpid()))
            logger.info("PID file written: %s (pid=%d)", self.pid_file, os.getpid())
        except OSError as exc:
            logger.warning("Failed to write PID file %s: %s", self.pid_file, exc)

    def _remove_pid_file(self) -> None:
        """Remove the PID file on shutdown."""
        if not self.pid_file:
            return
        try:
            Path(self.pid_file).unlink(missing_ok=True)
            logger.debug("PID file removed: %s", self.pid_file)
        except OSError as exc:
            logger.warning("Failed to remove PID file %s: %s", self.pid_file, exc)

    # ------------------------------------------------------------------ #
    # Signal handlers
    # ------------------------------------------------------------------ #

    def _register_signals(self) -> None:
        """Register handlers for SIGTERM, SIGINT, and SIGHUP."""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGHUP, self._handle_reload)
        logger.debug("Signal handlers registered.")

    def _handle_shutdown(self, signum: int, frame) -> None:
        """Handle SIGTERM/SIGINT — initiate graceful shutdown."""
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating graceful shutdown ...", sig_name)
        self._shutdown_event.set()
        if self._server:
            # Shut down in a thread to avoid blocking the signal handler
            threading.Thread(target=self._server.shutdown, daemon=True).start()

    def _handle_reload(self, signum: int, frame) -> None:
        """Handle SIGHUP — reload configuration."""
        logger.info("Received SIGHUP — reloading configuration ...")
        # Re-read environment file if present
        env_file = os.environ.get(
            "GATEWAY_ENV_FILE", "/etc/claude-code-model-gateway/environment"
        )
        if os.path.isfile(env_file):
            self._load_env_file(env_file)
            logger.info("Environment reloaded from %s", env_file)

    @staticmethod
    def _load_env_file(path: str) -> None:
        """Read a KEY=VALUE environment file and inject into os.environ."""
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key:
                            os.environ[key] = value
        except OSError as exc:
            logger.warning("Failed to load env file %s: %s", path, exc)

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #

    def run(self) -> int:
        """Start the service and block until shutdown.

        Returns:
            Exit code (0 for clean shutdown, 1 for error).
        """
        # Configure logging
        output = "both" if self.log_file else "console"
        setup_logging(
            level=self.log_level,
            log_format=self.log_format,
            output=output,
            log_file=self.log_file,
        )

        logger.info(
            "Starting claude-code-model-gateway service v%s "
            "(pid=%d, host=%s, port=%d)",
            __version__,
            os.getpid(),
            self.host,
            self.port,
        )

        self._register_signals()
        self._write_pid_file()

        try:
            return self._start_gateway()
        except Exception as exc:
            logger.critical("Service failed to start: %s", exc, exc_info=True)
            return 1
        finally:
            self._remove_pid_file()
            logger.info("Service stopped.")

    def _start_gateway(self) -> int:
        """Import and start the gateway/proxy server."""
        from src.proxy import create_proxy_server

        self._server = create_proxy_server(
            host=self.host,
            port=self.port,
            timeout=self.timeout,
        )

        logger.info(
            "Gateway listening on %s:%d (timeout=%ds)",
            self.host,
            self.port,
            self.timeout,
        )

        try:
            self._server.serve_forever()
        except Exception as exc:
            if not self._shutdown_event.is_set():
                logger.error("Server error: %s", exc, exc_info=True)
                return 1
        finally:
            self._server.server_close()

        logger.info("Gateway server shut down cleanly.")
        return 0


def _env(name: str, default: str) -> str:
    """Read an environment variable with a fallback default."""
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback default."""
    raw = os.environ.get(name)
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


def main() -> None:
    """Service entry point — reads settings from environment variables."""
    manager = ServiceManager(
        host=_env("GATEWAY_HOST", "0.0.0.0"),
        port=_env_int("GATEWAY_PORT", 8080),
        timeout=_env_int("GATEWAY_TIMEOUT", 300),
        log_level=_env("GATEWAY_LOG_LEVEL", "info"),
        log_format=_env("GATEWAY_LOG_FORMAT", "json"),
        log_file=_env("GATEWAY_LOG_FILE", ""),
        pid_file=_env("GATEWAY_PID_FILE", ""),
        config_file=_env("GATEWAY_CONFIG", ""),
    )
    sys.exit(manager.run())


if __name__ == "__main__":
    main()
