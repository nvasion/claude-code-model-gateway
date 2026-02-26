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
    fmt_str = str(log_format).lower()

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
