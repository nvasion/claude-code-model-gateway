"""Comprehensive logging system for claude-code-model-gateway.

Provides centralized logging configuration with support for:
- Multiple output formats (standard, JSON structured, colored)
- Console and rotating file handlers
- Request correlation IDs for tracing
- Performance timing decorators and context managers
- Log filtering by module, level, and custom predicates
- Integration with gateway configuration (GatewayConfig.log_level)

Typical usage:

    from src.logging_config import setup_logging, get_logger

    # Setup once at application start
    setup_logging(level="info", log_format="standard")

    # Get a module-level logger
    logger = get_logger(__name__)
    logger.info("Server started on port %d", port)

    # Use structured logging
    logger.info("request_processed", extra={
        "request_id": "abc-123",
        "duration_ms": 42,
        "status_code": 200,
    })

    # Use performance timing
    from src.logging_config import log_duration

    @log_duration("config_load")
    def load_config():
        ...

    with log_duration("db_query"):
        ...
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging
import logging.handlers
import os
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Application root logger name
ROOT_LOGGER_NAME = "gateway"

# Default log format strings
STANDARD_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s "
    "[%(filename)s:%(lineno)d] %(funcName)s: %(message)s"
)
MINIMAL_FORMAT = "%(levelname)s: %(message)s"

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default rotating log file settings
DEFAULT_LOG_DIR = "logs"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5

# Valid log level names
VALID_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}

# Context variable for request correlation ID
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LogFormat(str, Enum):
    """Supported log output formats."""

    STANDARD = "standard"
    DETAILED = "detailed"
    MINIMAL = "minimal"
    JSON = "json"
    COLORED = "colored"


class LogOutput(str, Enum):
    """Supported log output destinations."""

    CONSOLE = "console"
    FILE = "file"
    BOTH = "both"


# ---------------------------------------------------------------------------
# Correlation ID management
# ---------------------------------------------------------------------------


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID for request tracing.

    Returns:
        The current correlation ID, or None if not set.
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set the correlation ID for the current context.

    Args:
        correlation_id: A specific ID to use. If None, generates a UUID.

    Returns:
        The correlation ID that was set.
    """
    cid = correlation_id or uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current context."""
    _correlation_id.set(None)


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager that sets and clears a correlation ID.

    Args:
        correlation_id: A specific ID to use. If None, generates a UUID.

    Yields:
        The active correlation ID.

    Example::

        with correlation_context() as cid:
            logger.info("Processing request %s", cid)
    """
    cid = set_correlation_id(correlation_id)
    try:
        yield cid
    finally:
        clear_correlation_id()


# ---------------------------------------------------------------------------
# Custom Formatters
# ---------------------------------------------------------------------------


class StandardFormatter(logging.Formatter):
    """Standard text log formatter with optional correlation ID.

    Extends the default formatter to append the correlation ID when present.
    """

    def __init__(
        self,
        fmt: str = STANDARD_FORMAT,
        datefmt: str = DATE_FORMAT,
        include_correlation_id: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.include_correlation_id = include_correlation_id

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with optional correlation ID."""
        if self.include_correlation_id:
            cid = get_correlation_id()
            if cid:
                record.msg = f"[{cid}] {record.msg}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter.

    Produces one JSON object per log line containing standard fields
    plus any extra attributes passed to the logger.
    """

    def __init__(
        self,
        include_extras: bool = True,
        include_stack_info: bool = True,
    ) -> None:
        super().__init__()
        self.include_extras = include_extras
        self.include_stack_info = include_stack_info

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON object."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if present
        cid = get_correlation_id()
        if cid:
            log_entry["correlation_id"] = cid

        # Add thread info
        log_entry["thread"] = record.thread
        log_entry["thread_name"] = record.threadName

        # Add process info
        log_entry["process"] = record.process

        # Add exception info
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add stack info
        if self.include_stack_info and record.stack_info:
            log_entry["stack_info"] = record.stack_info

        # Add extra fields (skip standard LogRecord attributes)
        if self.include_extras:
            standard_attrs = {
                "name", "msg", "args", "created", "relativeCreated",
                "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "levelno", "levelname", "pathname", "filename", "module",
                "thread", "threadName", "process", "processName", "msecs",
                "message", "taskName",
            }
            for key, value in record.__dict__.items():
                if key not in standard_attrs and not key.startswith("_"):
                    try:
                        json.dumps(value)  # Ensure serializable
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)

        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored terminal log formatter.

    Applies ANSI color codes based on log level for improved
    readability in terminal output.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"

    def __init__(
        self,
        fmt: str = STANDARD_FORMAT,
        datefmt: str = DATE_FORMAT,
        use_colors: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colors = use_colors and self._supports_color()

    @staticmethod
    def _supports_color() -> bool:
        """Check if the terminal supports color output."""
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("FORCE_COLOR"):
            return True
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with color codes."""
        # Add correlation ID
        cid = get_correlation_id()
        if cid:
            record.msg = f"[{cid}] {record.msg}"

        if not self.use_colors:
            return super().format(record)

        # Apply color to the level name
        color = self.COLORS.get(record.levelname, "")
        original_levelname = record.levelname

        record.levelname = f"{color}{record.levelname}{self.RESET}"

        # Dim the timestamp
        formatted = super().format(record)

        # Restore original levelname for other handlers
        record.levelname = original_levelname

        return formatted


# ---------------------------------------------------------------------------
# Custom Filters
# ---------------------------------------------------------------------------


class CorrelationFilter(logging.Filter):
    """Filter that adds correlation ID to log records.

    This filter adds the correlation_id attribute to every log record
    so it can be used in format strings.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id attribute to the record."""
        record.correlation_id = get_correlation_id() or "-"  # type: ignore[attr-defined]
        return True


class ModuleFilter(logging.Filter):
    """Filter log records by module name patterns.

    Args:
        allowed_modules: List of module name prefixes to allow.
            If empty, all modules are allowed.
        denied_modules: List of module name prefixes to deny.
    """

    def __init__(
        self,
        allowed_modules: Optional[list[str]] = None,
        denied_modules: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.allowed_modules = allowed_modules or []
        self.denied_modules = denied_modules or []

    def filter(self, record: logging.LogRecord) -> bool:
        """Check if the record's module is allowed."""
        # Check deny list first
        for prefix in self.denied_modules:
            if record.name.startswith(prefix):
                return False

        # If allow list is empty, allow everything
        if not self.allowed_modules:
            return True

        # Check allow list
        for prefix in self.allowed_modules:
            if record.name.startswith(prefix):
                return True

        return False


class RateLimitFilter(logging.Filter):
    """Filter that rate-limits repeated log messages.

    Suppresses duplicate messages that occur more frequently than
    the configured interval.

    Args:
        rate_seconds: Minimum interval between identical messages.
        max_suppressed_report: After this many suppressions, log a
            summary message.
    """

    def __init__(
        self,
        rate_seconds: float = 5.0,
        max_suppressed_report: int = 100,
    ) -> None:
        super().__init__()
        self.rate_seconds = rate_seconds
        self.max_suppressed_report = max_suppressed_report
        self._lock = threading.Lock()
        self._last_seen: dict[str, float] = {}
        self._suppressed_count: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Rate-limit repeated messages."""
        key = f"{record.name}:{record.levelno}:{record.msg}"
        now = time.monotonic()

        with self._lock:
            last = self._last_seen.get(key, 0.0)
            elapsed = now - last

            if elapsed < self.rate_seconds:
                self._suppressed_count[key] = (
                    self._suppressed_count.get(key, 0) + 1
                )

                # Periodically report suppression count
                if (
                    self._suppressed_count[key] >= self.max_suppressed_report
                ):
                    count = self._suppressed_count[key]
                    self._suppressed_count[key] = 0
                    record.msg = (
                        f"[suppressed {count} similar messages] {record.msg}"
                    )
                    self._last_seen[key] = now
                    return True

                return False

            # Report any previous suppressions
            suppressed = self._suppressed_count.pop(key, 0)
            if suppressed > 0:
                record.msg = (
                    f"[suppressed {suppressed} similar messages] {record.msg}"
                )

            self._last_seen[key] = now
            return True

    def reset(self) -> None:
        """Reset all rate-limiting state."""
        with self._lock:
            self._last_seen.clear()
            self._suppressed_count.clear()


# ---------------------------------------------------------------------------
# Log configuration data class
# ---------------------------------------------------------------------------


@dataclass
class LoggingConfig:
    """Configuration for the logging system.

    Attributes:
        level: The minimum log level (debug, info, warning, error, critical).
        log_format: Output format (standard, detailed, minimal, json, colored).
        output: Output destination (console, file, both).
        log_file: Path to the log file (used when output includes file).
        max_bytes: Maximum log file size before rotation.
        backup_count: Number of rotated log files to keep.
        include_correlation_id: Whether to include correlation IDs.
        rate_limit_seconds: Rate limiting interval for repeated messages.
            Set to 0 to disable.
        module_filters: Optional list of module prefixes to include.
        json_extras: Whether to include extra fields in JSON output.
    """

    level: str = "info"
    log_format: str = "standard"
    output: str = "console"
    log_file: str = ""
    max_bytes: int = DEFAULT_MAX_BYTES
    backup_count: int = DEFAULT_BACKUP_COUNT
    include_correlation_id: bool = True
    rate_limit_seconds: float = 0.0
    module_filters: list[str] = field(default_factory=list)
    json_extras: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "level": self.level,
            "log_format": self.log_format,
            "output": self.output,
            "log_file": self.log_file,
            "max_bytes": self.max_bytes,
            "backup_count": self.backup_count,
            "include_correlation_id": self.include_correlation_id,
            "rate_limit_seconds": self.rate_limit_seconds,
            "module_filters": self.module_filters,
            "json_extras": self.json_extras,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoggingConfig:
        """Deserialize configuration from dictionary."""
        return cls(
            level=data.get("level", "info"),
            log_format=data.get("log_format", "standard"),
            output=data.get("output", "console"),
            log_file=data.get("log_file", ""),
            max_bytes=data.get("max_bytes", DEFAULT_MAX_BYTES),
            backup_count=data.get("backup_count", DEFAULT_BACKUP_COUNT),
            include_correlation_id=data.get("include_correlation_id", True),
            rate_limit_seconds=data.get("rate_limit_seconds", 0.0),
            module_filters=data.get("module_filters", []),
            json_extras=data.get("json_extras", True),
        )


# ---------------------------------------------------------------------------
# Logger state tracking
# ---------------------------------------------------------------------------

_setup_lock = threading.Lock()
_is_configured = False
_active_config: Optional[LoggingConfig] = None
_managed_handlers: list[logging.Handler] = []


def is_configured() -> bool:
    """Check whether the logging system has been configured.

    Returns:
        True if setup_logging() has been called.
    """
    return _is_configured


def get_active_config() -> Optional[LoggingConfig]:
    """Get the active logging configuration.

    Returns:
        The current LoggingConfig, or None if not configured.
    """
    return _active_config


# ---------------------------------------------------------------------------
# Formatter factory
# ---------------------------------------------------------------------------


def create_formatter(
    log_format: Union[str, LogFormat] = LogFormat.STANDARD,
    include_correlation_id: bool = True,
    json_extras: bool = True,
) -> logging.Formatter:
    """Create a log formatter based on the specified format type.

    Args:
        log_format: The format type to use.
        include_correlation_id: Whether to include correlation IDs.
        json_extras: Whether to include extra fields in JSON format.

    Returns:
        A configured logging.Formatter instance.
    """
    fmt_str = getattr(log_format, "value", str(log_format)).lower()

    if fmt_str == "json":
        return JSONFormatter(
            include_extras=json_extras,
            include_stack_info=True,
        )
    elif fmt_str == "colored":
        return ColoredFormatter(
            fmt=STANDARD_FORMAT,
            datefmt=DATE_FORMAT,
            use_colors=True,
        )
    elif fmt_str == "detailed":
        return StandardFormatter(
            fmt=DETAILED_FORMAT,
            datefmt=DATE_FORMAT,
            include_correlation_id=include_correlation_id,
        )
    elif fmt_str == "minimal":
        return StandardFormatter(
            fmt=MINIMAL_FORMAT,
            datefmt=DATE_FORMAT,
            include_correlation_id=include_correlation_id,
        )
    else:
        # Default: standard
        return StandardFormatter(
            fmt=STANDARD_FORMAT,
            datefmt=DATE_FORMAT,
            include_correlation_id=include_correlation_id,
        )


# ---------------------------------------------------------------------------
# Core setup function
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "info",
    log_format: str = "standard",
    output: str = "console",
    log_file: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    include_correlation_id: bool = True,
    rate_limit_seconds: float = 0.0,
    module_filters: Optional[list[str]] = None,
    json_extras: bool = True,
    config: Optional[LoggingConfig] = None,
) -> logging.Logger:
    """Configure the application-wide logging system.

    This is the primary entry point for configuring logging. It sets up
    handlers, formatters, and filters on the root application logger.

    Can be called multiple times; previous configuration is replaced.

    Args:
        level: Minimum log level (debug, info, warning, error, critical).
        log_format: Output format (standard, detailed, minimal, json, colored).
        output: Output destination (console, file, both).
        log_file: Path to the log file. Auto-generated if output includes
            file and this is not set.
        max_bytes: Maximum log file size before rotation.
        backup_count: Number of rotated log files to keep.
        include_correlation_id: Whether to include correlation IDs.
        rate_limit_seconds: Rate limiting interval. 0 disables.
        module_filters: Optional list of module name prefixes to include.
        json_extras: Whether to include extra fields in JSON format.
        config: Optional LoggingConfig to use instead of individual params.

    Returns:
        The configured root application logger.

    Example::

        # Simple setup
        setup_logging(level="debug", log_format="colored")

        # Production setup with file rotation and JSON
        setup_logging(
            level="info",
            log_format="json",
            output="both",
            log_file="gateway.log",
            max_bytes=50_000_000,
            backup_count=10,
        )

        # From a LoggingConfig object
        cfg = LoggingConfig(level="debug", log_format="json")
        setup_logging(config=cfg)
    """
    global _is_configured, _active_config, _managed_handlers

    # Use config object if provided
    if config is not None:
        level = config.level
        log_format = config.log_format
        output = config.output
        log_file = config.log_file or None
        max_bytes = config.max_bytes
        backup_count = config.backup_count
        include_correlation_id = config.include_correlation_id
        rate_limit_seconds = config.rate_limit_seconds
        module_filters = config.module_filters or None
        json_extras = config.json_extras

    with _setup_lock:
        # Resolve log level
        log_level = _resolve_log_level(level)

        # Get or create the root application logger
        root_logger = logging.getLogger(ROOT_LOGGER_NAME)

        # Remove previously managed handlers
        for handler in _managed_handlers:
            root_logger.removeHandler(handler)
            handler.close()
        _managed_handlers = []

        root_logger.setLevel(log_level)

        # Create formatter
        formatter = create_formatter(
            log_format=log_format,
            include_correlation_id=include_correlation_id,
            json_extras=json_extras,
        )

        # Build filters
        filters: list[logging.Filter] = []
        filters.append(CorrelationFilter())

        if module_filters:
            filters.append(ModuleFilter(allowed_modules=module_filters))

        if rate_limit_seconds > 0:
            filters.append(RateLimitFilter(rate_seconds=rate_limit_seconds))

        # Setup console handler
        output_lower = output.lower()
        if output_lower in ("console", "both"):
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            for f in filters:
                console_handler.addFilter(f)
            root_logger.addHandler(console_handler)
            _managed_handlers.append(console_handler)

        # Setup file handler
        if output_lower in ("file", "both"):
            if log_file is None:
                # Auto-generate log file path
                log_dir = Path(DEFAULT_LOG_DIR)
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = str(log_dir / "gateway.log")

            # Ensure parent directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(log_level)

            # Use JSON format for files if format is standard/colored
            if log_format in ("colored", "standard", "minimal"):
                file_formatter = create_formatter(
                    log_format="detailed",
                    include_correlation_id=include_correlation_id,
                    json_extras=json_extras,
                )
            else:
                file_formatter = formatter

            file_handler.setFormatter(file_formatter)
            for f in filters:
                file_handler.addFilter(f)
            root_logger.addHandler(file_handler)
            _managed_handlers.append(file_handler)

        # Also configure the Python root logger to propagate our level
        # for modules that use logging.getLogger(__name__) directly
        root_py_logger = logging.getLogger()
        if not root_py_logger.handlers:
            # Only set up root if it has no handlers to avoid duplicates
            root_py_logger.setLevel(log_level)

        # Make src.* loggers propagate to our gateway logger
        for module_name in ("src", "src.proxy", "src.anthropic_passthrough",
                            "src.cache", "src.config", "src.validation"):
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(log_level)

        # Store active configuration
        _active_config = LoggingConfig(
            level=level,
            log_format=log_format,
            output=output,
            log_file=log_file or "",
            max_bytes=max_bytes,
            backup_count=backup_count,
            include_correlation_id=include_correlation_id,
            rate_limit_seconds=rate_limit_seconds,
            module_filters=module_filters or [],
            json_extras=json_extras,
        )
        _is_configured = True

    return root_logger


def reset_logging() -> None:
    """Reset the logging system to its unconfigured state.

    Removes all managed handlers and resets state tracking.
    Primarily useful for testing.
    """
    global _is_configured, _active_config, _managed_handlers

    with _setup_lock:
        root_logger = logging.getLogger(ROOT_LOGGER_NAME)
        for handler in _managed_handlers:
            root_logger.removeHandler(handler)
            handler.close()
        _managed_handlers = []
        _is_configured = False
        _active_config = None


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the gateway namespace.

    This creates a logger with the name ``gateway.<name>`` so it
    inherits the configuration set up by :func:`setup_logging`.

    For module-level loggers, pass ``__name__``::

        logger = get_logger(__name__)

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        A configured logger instance.
    """
    # If the name already starts with the root logger name, don't double-prefix
    if name.startswith(f"{ROOT_LOGGER_NAME}.") or name == ROOT_LOGGER_NAME:
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


# ---------------------------------------------------------------------------
# Performance timing utilities
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    """Result of a timed operation.

    Attributes:
        operation: Name of the operation.
        duration_ms: Duration in milliseconds.
        start_time: Start timestamp.
        end_time: End timestamp.
        success: Whether the operation completed without error.
        error: Error message if the operation failed.
    """

    operation: str
    duration_ms: float
    start_time: float
    end_time: float
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 3),
            "success": self.success,
        }
        if self.error:
            result["error"] = self.error
        return result


class log_duration:
    """Log the duration of an operation.

    Can be used as both a decorator and a context manager.

    As a decorator::

        @log_duration("load_config")
        def load_config():
            ...

    As a context manager::

        with log_duration("db_query") as timer:
            result = db.query(...)
        print(f"Query took {timer.result.duration_ms}ms")

    Args:
        operation: Name of the operation being timed.
        level: Log level for the timing message.
        logger: Logger to use. If None, uses the gateway root logger.
        threshold_ms: Only log if duration exceeds this threshold.
            Set to 0 to always log.
    """

    def __init__(
        self,
        operation: str,
        level: int = logging.DEBUG,
        logger: Optional[logging.Logger] = None,
        threshold_ms: float = 0.0,
    ) -> None:
        self.operation = operation
        self.level = level
        self.logger = logger or get_logger("timing")
        self.threshold_ms = threshold_ms
        self.result: Optional[TimingResult] = None

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as a decorator."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.monotonic()
            error_msg = None
            success = True
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                success = False
                error_msg = str(exc)
                raise
            finally:
                end = time.monotonic()
                duration_ms = (end - start) * 1000.0
                result = TimingResult(
                    operation=self.operation,
                    duration_ms=duration_ms,
                    start_time=start,
                    end_time=end,
                    success=success,
                    error=error_msg,
                )
                if duration_ms >= self.threshold_ms:
                    self.logger.log(
                        self.level,
                        "%s completed in %.3fms (success=%s)",
                        self.operation,
                        duration_ms,
                        success,
                    )

        return wrapper

    def __enter__(self) -> "log_duration":
        """Enter context manager."""
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and log timing."""
        end = time.monotonic()
        duration_ms = (end - self._start) * 1000.0
        success = exc_type is None
        error_msg = str(exc_val) if exc_val else None

        self.result = TimingResult(
            operation=self.operation,
            duration_ms=duration_ms,
            start_time=self._start,
            end_time=end,
            success=success,
            error=error_msg,
        )

        if duration_ms >= self.threshold_ms:
            self.logger.log(
                self.level,
                "%s completed in %.3fms (success=%s)",
                self.operation,
                duration_ms,
                success,
            )


# ---------------------------------------------------------------------------
# Request logging middleware-style helper
# ---------------------------------------------------------------------------


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    client_ip: Optional[str] = None,
    request_id: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log an HTTP request with structured data.

    Intended as a helper for proxy/gateway request handlers.

    Args:
        method: HTTP method (GET, POST, etc.).
        path: Request path.
        status_code: Response status code.
        duration_ms: Request duration in milliseconds.
        client_ip: Client IP address.
        request_id: Unique request identifier.
        extra: Additional structured data to include.
        logger: Logger to use. Defaults to the gateway request logger.
    """
    log = logger or get_logger("request")

    log_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 3),
    }
    if client_ip:
        log_data["client_ip"] = client_ip
    if request_id:
        log_data["request_id"] = request_id
    if extra:
        log_data.update(extra)

    # Choose level based on status code
    if status_code >= 500:
        level = logging.ERROR
    elif status_code >= 400:
        level = logging.WARNING
    else:
        level = logging.INFO

    log.log(
        level,
        "%s %s -> %d (%.1fms)",
        method,
        path,
        status_code,
        duration_ms,
        extra=log_data,
    )


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


_audit_logger: Optional[logging.Logger] = None


def get_audit_logger() -> logging.Logger:
    """Get the dedicated audit logger.

    The audit logger is separate from the main application logger and
    is intended for recording security-relevant events (config changes,
    authentication events, etc.).

    Returns:
        The audit logger instance.
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = get_logger("audit")
    return _audit_logger


def audit_log(
    event: str,
    details: Optional[dict[str, Any]] = None,
    level: int = logging.INFO,
) -> None:
    """Record an audit event.

    Args:
        event: Short description of the event (e.g., "config_changed").
        details: Structured details about the event.
        level: Log level for the audit event.
    """
    logger = get_audit_logger()
    extra_data = {"audit_event": event}
    if details:
        extra_data.update(details)

    logger.log(
        level,
        "AUDIT: %s",
        event,
        extra=extra_data,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _resolve_log_level(level: str) -> int:
    """Convert a string log level to its numeric value.

    Args:
        level: Log level name (case-insensitive).

    Returns:
        The numeric log level.

    Raises:
        ValueError: If the level name is invalid.
    """
    level_upper = level.upper()
    numeric = getattr(logging, level_upper, None)
    if numeric is None or not isinstance(numeric, int):
        valid = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ValueError(
            f"Invalid log level '{level}'. Must be one of: {valid}"
        )
    return numeric


def get_log_level_name(level: int) -> str:
    """Convert a numeric log level to its name.

    Args:
        level: Numeric log level.

    Returns:
        The log level name (e.g., 'INFO').
    """
    return logging.getLevelName(level)


def configure_from_gateway_config(gateway_config: Any) -> logging.Logger:
    """Configure logging from a GatewayConfig object.

    Reads the ``log_level`` attribute from the gateway configuration
    and applies it. Supports the ``logging`` extra configuration key
    for advanced settings.

    Args:
        gateway_config: A GatewayConfig instance.

    Returns:
        The configured root logger.
    """
    level = getattr(gateway_config, "log_level", "info")

    # Check for extended logging config in the 'extra' dict
    extra = getattr(gateway_config, "extra", {})
    logging_extra = extra.get("logging", {}) if isinstance(extra, dict) else {}

    if logging_extra:
        log_config = LoggingConfig.from_dict(logging_extra)
        log_config.level = level  # Gateway level takes precedence
        return setup_logging(config=log_config)

    return setup_logging(level=level)


def get_logging_status() -> dict[str, Any]:
    """Get the current status of the logging system.

    Returns:
        Dictionary with logging system status information.
    """
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)

    status: dict[str, Any] = {
        "configured": _is_configured,
        "root_logger": ROOT_LOGGER_NAME,
        "effective_level": get_log_level_name(root_logger.getEffectiveLevel()),
        "handler_count": len(root_logger.handlers),
        "handlers": [],
    }

    for handler in root_logger.handlers:
        handler_info: dict[str, Any] = {
            "type": type(handler).__name__,
            "level": get_log_level_name(handler.level),
            "formatter": type(handler.formatter).__name__ if handler.formatter else None,
            "filters": [type(f).__name__ for f in handler.filters],
        }
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            handler_info["file"] = handler.baseFilename
            handler_info["max_bytes"] = handler.maxBytes
            handler_info["backup_count"] = handler.backupCount
        status["handlers"].append(handler_info)

    if _active_config:
        status["config"] = _active_config.to_dict()

    return status


# ---------------------------------------------------------------------------
# SensitiveDataFilter
# ---------------------------------------------------------------------------

_DEFAULT_SENSITIVE_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9\-_]+"),
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"(?i)api[_\-]?key\s*[=:]\s*\S+"),
    re.compile(r"(?i)Bearer\s+\S+"),
    re.compile(r"(?i)password\s*[=:]\s*\S+"),
    re.compile(r"(?i)x-api-key\s*[=:]\s*\S+"),
    re.compile(r"(?i)secret\s*[=:]\s*\S+"),
    re.compile(r"(?i)token\s*[=:]\s*\S+"),
]

_SENSITIVE_EXTRA_KEYS = frozenset(
    {"api_key", "password", "secret", "token", "auth", "bearer", "key"}
)


class SensitiveDataFilter(logging.Filter):
    """Filter that redacts sensitive data from log records.

    Scans log message text (and optionally extra fields) for patterns
    matching API keys, passwords, Bearer tokens, and similar secrets,
    replacing them with a configurable placeholder.

    Args:
        patterns: List of compiled regex patterns (or raw strings) to redact.
            Defaults to the built-in set of common secret patterns.
        replacement: Replacement text used in the message.  Defaults to
            ``"REDACTED"``.
        redact_extras: Whether to also redact extra fields whose *names*
            match common sensitive-key names.  Defaults to ``True``.
    """

    def __init__(
        self,
        patterns: Optional[list] = None,
        replacement: str = "REDACTED",
        redact_extras: bool = True,
    ) -> None:
        super().__init__()
        if patterns is None:
            self._patterns = _DEFAULT_SENSITIVE_PATTERNS
        else:
            compiled: list = []
            for p in patterns:
                if isinstance(p, str):
                    compiled.append(re.compile(p))
                else:
                    compiled.append(p)
            self._patterns = compiled
        self._replacement = replacement
        self._redact_extras = redact_extras

    def _redact(self, text: str) -> str:
        """Apply all redaction patterns to *text*."""
        for pattern in self._patterns:
            text = pattern.sub(self._replacement, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from the record in-place."""
        # Redact message
        record.msg = self._redact(str(record.msg))

        # Redact args
        if isinstance(record.args, tuple):
            record.args = tuple(
                self._redact(str(a)) if isinstance(a, str) else a
                for a in record.args
            )
        elif isinstance(record.args, dict):
            record.args = {
                k: self._redact(str(v)) if isinstance(v, str) else v
                for k, v in record.args.items()
            }

        # Redact sensitive extra fields by key name
        if self._redact_extras:
            for attr_name in list(record.__dict__.keys()):
                if attr_name.lower() in _SENSITIVE_EXTRA_KEYS:
                    setattr(record, attr_name, "***REDACTED***")

        return True


# ---------------------------------------------------------------------------
# SamplingFilter
# ---------------------------------------------------------------------------


class SamplingFilter(logging.Filter):
    """Filter that probabilistically samples log records by level.

    Messages at levels in *always_pass_levels* (defaults to ERROR and
    CRITICAL) are always passed through.  For other levels a random
    draw against the configured *rate* decides whether the record is
    passed or suppressed.

    Args:
        rates: Mapping of level name → probability in [0.0, 1.0].
        default_rate: Rate for levels not in *rates*.  Defaults to 1.0.
        always_pass_levels: Set of level names that are never sampled.
            Defaults to ``{"ERROR", "CRITICAL"}``.
    """

    def __init__(
        self,
        rates: Optional[dict[str, float]] = None,
        default_rate: float = 1.0,
        always_pass_levels: Optional[set] = None,
    ) -> None:
        super().__init__()
        self._rates: dict[str, float] = rates or {}
        self._default_rate = default_rate
        self._always_pass: set = (
            always_pass_levels
            if always_pass_levels is not None
            else {"ERROR", "CRITICAL"}
        )
        self._lock = threading.Lock()
        self._total_count = 0
        self._passed_count = 0
        self._suppressed_by_level: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Sample the record according to its level's configured rate."""
        import random

        level_name = record.levelname
        with self._lock:
            self._total_count += 1

            if level_name in self._always_pass:
                self._passed_count += 1
                return True

            rate = self._rates.get(level_name, self._default_rate)
            if random.random() < rate:
                self._passed_count += 1
                return True
            else:
                self._suppressed_by_level[level_name] = (
                    self._suppressed_by_level.get(level_name, 0) + 1
                )
                return False

    def get_stats(self) -> dict[str, Any]:
        """Return sampling statistics."""
        with self._lock:
            return {
                "total_count": self._total_count,
                "passed_count": self._passed_count,
                "suppressed_by_level": dict(self._suppressed_by_level),
            }

    def reset_stats(self) -> None:
        """Reset all sampling statistics."""
        with self._lock:
            self._total_count = 0
            self._passed_count = 0
            self._suppressed_by_level = {}


# ---------------------------------------------------------------------------
# LogMetrics
# ---------------------------------------------------------------------------


class LogMetrics:
    """Collects aggregate metrics about log output.

    Tracks message counts by level and module, recent error details,
    and provides an error-rate calculation over a configurable window.

    Args:
        window_seconds: Size of the sliding time-window used for the
            error-rate calculation.  Defaults to 60 seconds.
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._total_count = 0
        self._levels: dict[str, int] = {}
        self._modules: dict[str, int] = {}
        self._recent_errors: list[dict[str, Any]] = []
        self._max_recent_errors = 100
        self._error_timestamps: list[float] = []
        self._start_time = time.monotonic()

    def record(self, record: logging.LogRecord) -> None:
        """Record a log message in the metrics."""
        with self._lock:
            self._total_count += 1
            level_name = record.levelname
            self._levels[level_name] = self._levels.get(level_name, 0) + 1
            self._modules[record.name] = self._modules.get(record.name, 0) + 1

            if record.levelno >= logging.ERROR:
                now = time.monotonic()
                self._error_timestamps.append(now)
                # Trim timestamps outside the window
                cutoff = now - self._window_seconds
                self._error_timestamps = [
                    t for t in self._error_timestamps if t >= cutoff
                ]

                error_entry: dict[str, Any] = {
                    "message": record.getMessage(),
                    "module": record.name,
                    "timestamp": record.created,
                }
                if record.exc_info and record.exc_info[0] is not None:
                    error_entry["exception_type"] = record.exc_info[0].__name__
                self._recent_errors.append(error_entry)
                if len(self._recent_errors) > self._max_recent_errors:
                    self._recent_errors = self._recent_errors[
                        -self._max_recent_errors:
                    ]

    def get_report(self) -> dict[str, Any]:
        """Return a snapshot of current metrics."""
        with self._lock:
            return {
                "total_count": self._total_count,
                "levels": dict(self._levels),
                "top_modules": dict(self._modules),
                "recent_errors": list(self._recent_errors),
                "uptime_seconds": time.monotonic() - self._start_time,
            }

    def get_error_rate(self) -> float:
        """Return the error rate (errors per minute) over the time window."""
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._window_seconds
            recent = [t for t in self._error_timestamps if t >= cutoff]
            return len(recent) * 60.0 / self._window_seconds

    def reset(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._total_count = 0
            self._levels = {}
            self._modules = {}
            self._recent_errors = []
            self._error_timestamps = []
            self._start_time = time.monotonic()


class LogMetricsHandler(logging.Handler):
    """Logging handler that feeds records into a :class:`LogMetrics` instance.

    Args:
        metrics: The LogMetrics instance to record into.
    """

    def __init__(self, metrics: LogMetrics) -> None:
        super().__init__()
        self.metrics = metrics

    def emit(self, record: logging.LogRecord) -> None:
        """Record the log entry in the metrics."""
        self.metrics.record(record)


# Global LogMetrics instance
_global_metrics: Optional["LogMetrics"] = None


def get_log_metrics() -> LogMetrics:
    """Return (or lazily create) the global LogMetrics instance.

    Returns:
        The singleton LogMetrics for the process.
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = LogMetrics()
    return _global_metrics


def reset_log_metrics() -> None:
    """Discard the current global LogMetrics instance.

    The next call to :func:`get_log_metrics` will create a fresh instance.
    Primarily useful for testing.
    """
    global _global_metrics
    _global_metrics = None


# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------


class StructuredLogger:
    """A structured-logging wrapper around a standard :class:`logging.Logger`.

    Allows default fields to be bound once and injected into every subsequent
    log call without repeating them at every call site.

    Args:
        name: Logger name (passed to :func:`get_logger`).
        default_fields: Key/value pairs injected into every log record's
            ``extra`` dict.  Not required at construction time.

    Example::

        log = StructuredLogger("my_module")
        req_log = log.bind(request_id="abc-123", user="alice")
        req_log.info("Processing request", endpoint="/v1/messages")
    """

    def __init__(
        self,
        name: str,
        default_fields: Optional[dict[str, Any]] = None,
    ) -> None:
        self._module_name = name
        self._logger = get_logger(name)
        self._default_fields: dict[str, Any] = dict(default_fields or {})

    @property
    def name(self) -> str:
        """The fully-qualified logger name."""
        return self._logger.name

    def bind(self, **kwargs: Any) -> "StructuredLogger":
        """Return a new StructuredLogger with additional default fields.

        Args:
            **kwargs: Fields to add (or override) in the new logger.

        Returns:
            A child StructuredLogger with the merged default fields.
        """
        new_fields = {**self._default_fields, **kwargs}
        child = StructuredLogger.__new__(StructuredLogger)
        child._module_name = self._module_name
        child._logger = self._logger
        child._default_fields = new_fields
        return child

    def unbind(self, *keys: str) -> "StructuredLogger":
        """Return a new StructuredLogger with specified fields removed.

        Args:
            *keys: Field names to remove from the default fields.

        Returns:
            A child StructuredLogger without the specified fields.
        """
        new_fields = {
            k: v for k, v in self._default_fields.items() if k not in keys
        }
        child = StructuredLogger.__new__(StructuredLogger)
        child._module_name = self._module_name
        child._logger = self._logger
        child._default_fields = new_fields
        return child

    def _extra(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {**self._default_fields, **kwargs}

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._logger.debug(msg, extra=self._extra(kwargs))

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._logger.info(msg, extra=self._extra(kwargs))

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._logger.warning(msg, extra=self._extra(kwargs))

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._logger.error(msg, extra=self._extra(kwargs))

    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._logger.critical(msg, extra=self._extra(kwargs))

    def exception(self, msg: str, **kwargs: Any) -> None:
        """Log at ERROR level, capturing current exception info."""
        self._logger.exception(msg, extra=self._extra(kwargs))

    def log(self, level: int, msg: str, **kwargs: Any) -> None:
        """Log at an arbitrary numeric level."""
        self._logger.log(level, msg, extra=self._extra(kwargs))


# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------

_request_context_var: contextvars.ContextVar[Optional[dict[str, Any]]] = (
    contextvars.ContextVar("request_context", default=None)
)


def set_request_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """Set the request context for the current async/thread context.

    Args:
        ctx: Dictionary of request metadata (e.g. method, path, request_id).

    Returns:
        The context dictionary that was set.
    """
    _request_context_var.set(ctx)
    return ctx


def get_request_context() -> Optional[dict[str, Any]]:
    """Get the current request context, or None if not set.

    Returns:
        The current request context dictionary, or None.
    """
    return _request_context_var.get()


def clear_request_context() -> None:
    """Clear the current request context."""
    _request_context_var.set(None)


def update_request_context(**kwargs: Any) -> dict[str, Any]:
    """Update (or create) the current request context with new fields.

    Args:
        **kwargs: Fields to add or update.

    Returns:
        The updated request context dictionary.
    """
    existing = _request_context_var.get() or {}
    updated = {**existing, **kwargs}
    _request_context_var.set(updated)
    return updated


@contextmanager
def request_context(**kwargs: Any):
    """Context manager that sets up a correlated request context.

    Sets a correlation ID, stores all kwargs as the request context
    (adding ``correlation_id`` automatically), and cleans up on exit.

    Args:
        **kwargs: Arbitrary request metadata.  A ``correlation_id`` key
            can be passed to use a specific ID instead of a generated one.

    Yields:
        Tuple of ``(correlation_id, context_dict)``.

    Example::

        with request_context(method="POST", path="/v1/messages") as (cid, ctx):
            logger.info("Handling %s %s", ctx["method"], ctx["path"])
    """
    correlation_id = kwargs.pop("correlation_id", None)
    cid = set_correlation_id(correlation_id)
    ctx: dict[str, Any] = dict(kwargs)
    ctx["correlation_id"] = cid
    set_request_context(ctx)
    try:
        yield cid, ctx
    finally:
        clear_request_context()
        clear_correlation_id()


# ---------------------------------------------------------------------------
# RequestContextFilter
# ---------------------------------------------------------------------------

_STANDARD_LOG_ATTRS: frozenset = frozenset(
    {
        "name", "msg", "args", "created", "relativeCreated", "exc_info",
        "exc_text", "stack_info", "lineno", "funcName", "levelno", "levelname",
        "pathname", "filename", "module", "thread", "threadName", "process",
        "processName", "msecs", "message", "taskName",
    }
)


class RequestContextFilter(logging.Filter):
    """Filter that injects current request context into log records.

    Any fields present in the active request context (set via
    :func:`set_request_context`) are attached to each log record so
    they are available to formatters and downstream handlers.

    Standard :class:`logging.LogRecord` attributes are never overwritten.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Inject request context fields into *record*."""
        ctx = get_request_context()
        if ctx:
            for key, value in ctx.items():
                if key not in _STANDARD_LOG_ATTRS:
                    setattr(record, key, value)
        return True


# ---------------------------------------------------------------------------
# LogBuffer / BufferedHandler
# ---------------------------------------------------------------------------

_STANDARD_RECORD_DICT_ATTRS: frozenset = frozenset(
    {
        "name", "msg", "args", "created", "relativeCreated", "exc_info",
        "exc_text", "stack_info", "lineno", "funcName", "levelno", "levelname",
        "pathname", "filename", "module", "thread", "threadName", "process",
        "processName", "msecs", "message", "taskName",
    }
)


class LogBuffer:
    """Thread-safe buffer for log records.

    Accumulates records as serialisable dictionaries.  When the buffer
    reaches *max_size* it either auto-flushes (if *on_flush* is set) or
    drops subsequent records.

    Args:
        max_size: Maximum number of entries before auto-flush or drop.
        on_flush: Callback invoked with a list of entry dicts on flush.
    """

    def __init__(
        self,
        max_size: int = 1000,
        on_flush: Optional[Callable[[list], None]] = None,
    ) -> None:
        self._max_size = max_size
        self._on_flush = on_flush
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._total_flushed = 0
        self._total_dropped = 0

    @property
    def size(self) -> int:
        """Current number of buffered entries."""
        with self._lock:
            return len(self._entries)

    def add(self, record: logging.LogRecord) -> bool:
        """Add a log record to the buffer.

        Returns:
            ``True`` if the record was buffered (or the buffer was
            auto-flushed to make room), ``False`` if it was dropped.
        """
        entries_to_flush: Optional[list] = None
        with self._lock:
            if len(self._entries) >= self._max_size:
                if self._on_flush:
                    entries_to_flush = self._entries[:]
                    self._total_flushed += len(entries_to_flush)
                    self._entries = [self._record_to_dict(record)]
                else:
                    self._total_dropped += 1
                    return False
            else:
                self._entries.append(self._record_to_dict(record))

        if entries_to_flush is not None and self._on_flush:
            self._on_flush(entries_to_flush)

        return True

    def flush(self) -> list[dict[str, Any]]:
        """Flush and return all buffered entries.

        Invokes the *on_flush* callback (if set) and clears the buffer.

        Returns:
            List of entry dictionaries.
        """
        with self._lock:
            entries = self._entries[:]
            self._total_flushed += len(entries)
            self._entries = []

        if entries and self._on_flush:
            self._on_flush(entries)

        return entries

    def stop(self) -> list[dict[str, Any]]:
        """Flush and stop the buffer.  Returns remaining entries."""
        return self.flush()

    def get_stats(self) -> dict[str, Any]:
        """Return buffer statistics."""
        with self._lock:
            return {
                "current_size": len(self._entries),
                "max_size": self._max_size,
                "total_flushed": self._total_flushed,
                "total_dropped": self._total_dropped,
            }

    def _record_to_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        """Convert a log record to a serialisable dictionary."""
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": getattr(record, "module", ""),
            "function": record.funcName,
            "line": record.lineno,
        }
        cid = get_correlation_id()
        if cid:
            entry["correlation_id"] = cid
        # Append extra fields
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_DICT_ATTRS and not key.startswith("_"):
                entry[key] = value
        return entry


class BufferedHandler(logging.Handler):
    """Logging handler that routes records to a :class:`LogBuffer`.

    Args:
        buffer: The LogBuffer to write records into.
    """

    def __init__(self, buffer: LogBuffer) -> None:
        super().__init__()
        self._buffer = buffer

    @property
    def buffer(self) -> LogBuffer:
        """The underlying :class:`LogBuffer`."""
        return self._buffer

    def emit(self, record: logging.LogRecord) -> None:
        """Add the record to the buffer."""
        self._buffer.add(record)

    def close(self) -> None:
        """Flush the buffer and close the handler."""
        self._buffer.stop()
        super().close()


# ---------------------------------------------------------------------------
# Exception formatting helpers
# ---------------------------------------------------------------------------


def format_exception_context(
    exc: BaseException,
    include_chain: bool = True,
    max_depth: int = 10,
) -> dict[str, Any]:
    """Format an exception as a rich context dictionary.

    Includes the exception type, module, message, and traceback.  For
    :class:`~src.errors.GatewayError` subclasses the structured error
    context is also included.  Exception chains (``__cause__``) are
    walked up to *max_depth* levels.

    Args:
        exc: The exception to describe.
        include_chain: Whether to include the ``__cause__`` chain.
        max_depth: Maximum chain depth to traverse.

    Returns:
        Dictionary with keys ``type``, ``module``, ``message``,
        ``traceback``, and optionally ``chain``, ``is_retryable``,
        ``error_context``.
    """
    import traceback as _traceback

    result: dict[str, Any] = {
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "message": str(exc),
        "traceback": _traceback.format_tb(exc.__traceback__),
    }

    # Enrich GatewayError instances
    try:
        from src.errors import GatewayError

        if isinstance(exc, GatewayError):
            result["is_retryable"] = exc.is_retryable
            result["error_context"] = exc.context.to_dict()
    except ImportError:
        pass

    # Walk the exception chain
    if include_chain:
        chain: list[dict[str, Any]] = []
        cause = exc.__cause__
        depth = 0
        while cause is not None and depth < max_depth:
            chain.append(
                {"type": type(cause).__name__, "message": str(cause)}
            )
            cause = cause.__cause__
            depth += 1
        if chain:
            result["chain"] = chain

    return result


def log_exception(
    logger: logging.Logger,
    message: str,
    exc: BaseException,
    level: int = logging.ERROR,
    **kwargs: Any,
) -> None:
    """Log an exception with structured context fields.

    Attaches ``exception_type`` and ``exception_message`` to the log
    record, plus any additional :class:`~src.errors.GatewayError` fields
    (``is_retryable``, ``error_context``) when applicable.

    Args:
        logger: The logger to emit to.
        message: Human-readable log message.
        exc: The exception to log.
        level: Log level (defaults to ERROR).
        **kwargs: Additional fields to include in the record's ``extra``.
    """
    extra: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }
    extra.update(kwargs)

    try:
        from src.errors import GatewayError

        if isinstance(exc, GatewayError):
            extra["is_retryable"] = exc.is_retryable
            extra["error_context"] = exc.context.to_dict()
    except ImportError:
        pass

    logger.log(level, message, exc_info=exc, extra=extra)


# ---------------------------------------------------------------------------
# HealthLogger
# ---------------------------------------------------------------------------


class HealthLogger:
    """Periodically emits health-check log messages.

    Runs a background timer that calls :meth:`emit_now` at each
    *interval*.  Custom health-check functions can be registered to
    include component-level status in each emission.

    Args:
        interval: Seconds between automatic health-check emissions.
        include_metrics: Whether to include :class:`LogMetrics` data.
    """

    def __init__(
        self,
        interval: float = 60.0,
        include_metrics: bool = False,
    ) -> None:
        self._interval = interval
        self._include_metrics = include_metrics
        self._logger = get_logger("health")
        self._custom_checks: dict[str, Callable[[], Any]] = {}
        self._running = False
        self._heartbeat_count = 0
        self._start_time: Optional[float] = None
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def register_check(self, name: str, check_fn: Callable[[], Any]) -> None:
        """Register a custom health-check function.

        Args:
            name: Unique identifier for the check.
            check_fn: Zero-argument callable returning a status dict.
        """
        with self._lock:
            self._custom_checks[name] = check_fn

    def unregister_check(self, name: str) -> None:
        """Remove a previously registered health check.

        Args:
            name: The check identifier to remove.
        """
        with self._lock:
            self._custom_checks.pop(name, None)

    def emit_now(self) -> dict[str, Any]:
        """Execute all checks and emit a health log entry immediately.

        Returns:
            The health-status dictionary that was logged.
        """
        self._heartbeat_count += 1

        uptime = (
            time.monotonic() - self._start_time
            if self._start_time is not None
            else 0.0
        )

        data: dict[str, Any] = {
            "status": "healthy",
            "heartbeat_count": self._heartbeat_count,
            "uptime_seconds": uptime,
        }

        # Run custom checks
        with self._lock:
            custom_checks = dict(self._custom_checks)

        checks: dict[str, Any] = {}
        all_ok = True
        for name, check_fn in custom_checks.items():
            try:
                result = check_fn()
                checks[name] = result
            except Exception as e:
                checks[name] = {"status": "error", "error": str(e)}
                all_ok = False

        if checks:
            data["checks"] = checks

        if not all_ok:
            data["status"] = "degraded"

        if self._include_metrics:
            data["log_metrics"] = get_log_metrics().get_report()

        self._logger.info("Health check", extra=data)
        return data

    def start(self) -> None:
        """Start the background health-check timer."""
        with self._lock:
            if self._running:
                return
            self._running = True
            if self._start_time is None:
                self._start_time = time.monotonic()
        self._schedule_next()

    def stop(self) -> None:
        """Stop the background health-check timer."""
        with self._lock:
            self._running = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _schedule_next(self) -> None:
        """Schedule the next health-check tick."""
        with self._lock:
            if not self._running:
                return
            self._timer = threading.Timer(self._interval, self._tick)
            self._timer.daemon = True
            self._timer.start()

    def _tick(self) -> None:
        """Background timer callback."""
        if self._running:
            self.emit_now()
            self._schedule_next()


# ---------------------------------------------------------------------------
# setup_logging_advanced
# ---------------------------------------------------------------------------


def setup_logging_advanced(
    level: str = "info",
    log_format: str = "standard",
    output: str = "console",
    log_file: Optional[str] = None,
    enable_metrics: bool = False,
    enable_sensitive_filter: bool = False,
    sampling_rates: Optional[dict[str, float]] = None,
    enable_request_context: bool = False,
    buffer_config: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> logging.Logger:
    """Configure logging with advanced features on top of the standard setup.

    Calls :func:`setup_logging` first, then optionally attaches:

    * :class:`LogMetricsHandler` for metrics collection
    * :class:`SensitiveDataFilter` for secret redaction
    * :class:`SamplingFilter` for probabilistic sampling
    * :class:`RequestContextFilter` for request-context injection
    * :class:`BufferedHandler` for in-memory buffering

    Args:
        level: Log level string.
        log_format: Output format string.
        output: Output destination (console/file/both).
        log_file: Path to log file.
        enable_metrics: Attach a :class:`LogMetricsHandler`.
        enable_sensitive_filter: Attach a :class:`SensitiveDataFilter`.
        sampling_rates: If given, attach a :class:`SamplingFilter`.
        enable_request_context: Attach a :class:`RequestContextFilter`.
        buffer_config: If given, create and attach a :class:`BufferedHandler`.
            Recognised keys: ``max_size`` (int), ``on_flush`` (callable).
        **kwargs: Extra keyword arguments forwarded to :func:`setup_logging`.

    Returns:
        The configured root gateway :class:`logging.Logger`.
    """
    global _managed_handlers

    root_logger = setup_logging(
        level=level,
        log_format=log_format,
        output=output,
        log_file=log_file,
        **kwargs,
    )

    root = logging.getLogger(ROOT_LOGGER_NAME)

    # Metrics handler (attached as a handler, not a filter)
    if enable_metrics:
        metrics = get_log_metrics()
        metrics_handler = LogMetricsHandler(metrics)
        root.addHandler(metrics_handler)
        _managed_handlers.append(metrics_handler)

    # Per-handler filters
    additional_filters: list[logging.Filter] = []
    if enable_sensitive_filter:
        additional_filters.append(SensitiveDataFilter())
    if sampling_rates:
        additional_filters.append(SamplingFilter(rates=sampling_rates))
    if enable_request_context:
        additional_filters.append(RequestContextFilter())

    if additional_filters:
        for handler in root.handlers:
            for f in additional_filters:
                handler.addFilter(f)

    # Buffered handler
    if buffer_config is not None:
        buf = LogBuffer(
            max_size=buffer_config.get("max_size", 1000),
            on_flush=buffer_config.get("on_flush"),
        )
        buf_handler = BufferedHandler(buf)
        root.addHandler(buf_handler)
        _managed_handlers.append(buf_handler)

    return root_logger
