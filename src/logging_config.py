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
        max_bytes: Maximum log file size before rotation (size-based rotation).
        backup_count: Number of rotated log files to keep.
        include_correlation_id: Whether to include correlation IDs.
        rate_limit_seconds: Rate limiting interval for repeated messages.
            Set to 0 to disable.
        module_filters: Optional list of module prefixes to include.
        json_extras: Whether to include extra fields in JSON output.
        rotation_mode: File rotation mode: "size" (default) or "time".
        rotation_when: When to rotate for time-based rotation.
            One of 'S', 'M', 'H', 'D', 'midnight', 'W0'-'W6'.
        rotation_interval: Interval between rotations (for time-based).
        use_queue_handler: Whether to use QueueHandler for async logging.
        queue_max_size: Maximum size of the async logging queue.
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
    rotation_mode: str = "size"
    rotation_when: str = "midnight"
    rotation_interval: int = 1
    use_queue_handler: bool = False
    queue_max_size: int = 10000

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
            "rotation_mode": self.rotation_mode,
            "rotation_when": self.rotation_when,
            "rotation_interval": self.rotation_interval,
            "use_queue_handler": self.use_queue_handler,
            "queue_max_size": self.queue_max_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
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
            rotation_mode=data.get("rotation_mode", "size"),
            rotation_when=data.get("rotation_when", "midnight"),
            rotation_interval=data.get("rotation_interval", 1),
            use_queue_handler=data.get("use_queue_handler", False),
            queue_max_size=data.get("queue_max_size", 10000),
        )

    @classmethod
    def from_env(
        cls,
        prefix: str = "GATEWAY_LOG",
    ) -> "LoggingConfig":
        """Load logging configuration from environment variables.

        Reads configuration from environment variables with the given prefix.
        Environment variables take the form ``{PREFIX}_{FIELD}`` where FIELD
        is the uppercase field name.

        Supported variables:
            GATEWAY_LOG_LEVEL         - Log level (debug/info/warning/error/critical)
            GATEWAY_LOG_FORMAT        - Log format (standard/detailed/json/colored/minimal)
            GATEWAY_LOG_OUTPUT        - Output destination (console/file/both)
            GATEWAY_LOG_FILE          - Path to log file
            GATEWAY_LOG_MAX_BYTES     - Max file size before rotation
            GATEWAY_LOG_BACKUP_COUNT  - Number of backup files
            GATEWAY_LOG_RATE_LIMIT    - Rate limiting interval in seconds
            GATEWAY_LOG_ROTATION_MODE - Rotation mode (size/time)
            GATEWAY_LOG_ROTATION_WHEN - When to rotate (midnight/H/D/W0-W6)
            GATEWAY_LOG_QUEUE         - Use queue handler (1/true/yes)
            GATEWAY_LOG_JSON_EXTRAS   - Include JSON extras (1/true/yes)

        Args:
            prefix: Environment variable prefix (default: GATEWAY_LOG).

        Returns:
            A LoggingConfig populated from environment variables.
            Fields not set in the environment use default values.

        Example::

            # With GATEWAY_LOG_LEVEL=debug GATEWAY_LOG_FORMAT=json set:
            cfg = LoggingConfig.from_env()
            # cfg.level == "debug", cfg.log_format == "json"

            # Custom prefix:
            cfg = LoggingConfig.from_env("MY_APP_LOG")
        """

        def _bool(val: str) -> bool:
            return val.lower() in ("1", "true", "yes", "on")

        def _env(suffix: str) -> Optional[str]:
            return os.environ.get(f"{prefix}_{suffix}")

        cfg = cls()

        if (v := _env("LEVEL")) is not None:
            cfg.level = v.lower()
        if (v := _env("FORMAT")) is not None:
            cfg.log_format = v.lower()
        if (v := _env("OUTPUT")) is not None:
            cfg.output = v.lower()
        if (v := _env("FILE")) is not None:
            cfg.log_file = v
        if (v := _env("MAX_BYTES")) is not None:
            try:
                cfg.max_bytes = int(v)
            except ValueError:
                pass
        if (v := _env("BACKUP_COUNT")) is not None:
            try:
                cfg.backup_count = int(v)
            except ValueError:
                pass
        if (v := _env("RATE_LIMIT")) is not None:
            try:
                cfg.rate_limit_seconds = float(v)
            except ValueError:
                pass
        if (v := _env("ROTATION_MODE")) is not None:
            cfg.rotation_mode = v.lower()
        if (v := _env("ROTATION_WHEN")) is not None:
            cfg.rotation_when = v
        if (v := _env("ROTATION_INTERVAL")) is not None:
            try:
                cfg.rotation_interval = int(v)
            except ValueError:
                pass
        if (v := _env("QUEUE")) is not None:
            cfg.use_queue_handler = _bool(v)
        if (v := _env("QUEUE_MAX_SIZE")) is not None:
            try:
                cfg.queue_max_size = int(v)
            except ValueError:
                pass
        if (v := _env("JSON_EXTRAS")) is not None:
            cfg.json_extras = _bool(v)
        if (v := _env("CORRELATION_ID")) is not None:
            cfg.include_correlation_id = _bool(v)
        if (v := _env("MODULES")) is not None:
            cfg.module_filters = [m.strip() for m in v.split(",") if m.strip()]

        return cfg


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
    fmt_str = (log_format.value if hasattr(log_format, "value") else str(log_format)).lower()

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
    rotation_mode: str = "size",
    rotation_when: str = "midnight",
    rotation_interval: int = 1,
    use_queue_handler: bool = False,
    queue_max_size: int = 10000,
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
        max_bytes: Maximum log file size before rotation (size mode).
        backup_count: Number of rotated log files to keep.
        include_correlation_id: Whether to include correlation IDs.
        rate_limit_seconds: Rate limiting interval. 0 disables.
        module_filters: Optional list of module name prefixes to include.
        json_extras: Whether to include extra fields in JSON format.
        config: Optional LoggingConfig to use instead of individual params.
        rotation_mode: File rotation mode: "size" (default) or "time".
        rotation_when: When to rotate (for time-based rotation).
            One of 'S', 'M', 'H', 'D', 'midnight', 'W0'-'W6'.
        rotation_interval: How often to rotate (for time-based rotation).
        use_queue_handler: Whether to wrap file handler in QueueHandler
            for non-blocking async logging.
        queue_max_size: Maximum size of the async logging queue.

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

        # Time-based rotation (daily at midnight)
        setup_logging(
            level="info",
            output="file",
            log_file="gateway.log",
            rotation_mode="time",
            rotation_when="midnight",
            backup_count=30,
        )

        # Non-blocking async logging with QueueHandler
        setup_logging(
            level="info",
            output="file",
            log_file="gateway.log",
            use_queue_handler=True,
        )

        # From a LoggingConfig object
        cfg = LoggingConfig(level="debug", log_format="json")
        setup_logging(config=cfg)

        # From environment variables
        cfg = LoggingConfig.from_env()
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
        rotation_mode = config.rotation_mode
        rotation_when = config.rotation_when
        rotation_interval = config.rotation_interval
        use_queue_handler = config.use_queue_handler
        queue_max_size = config.queue_max_size

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

            # Create the appropriate file handler based on rotation mode
            if rotation_mode == "time":
                file_handler: logging.Handler = (
                    logging.handlers.TimedRotatingFileHandler(
                        filename=log_file,
                        when=rotation_when,
                        interval=rotation_interval,
                        backupCount=backup_count,
                        encoding="utf-8",
                    )
                )
                # Store original rotation_interval (handler.interval is in seconds)
                file_handler._rotation_interval = rotation_interval  # type: ignore[attr-defined]
            else:
                file_handler = logging.handlers.RotatingFileHandler(
                    filename=log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )

            file_handler.setLevel(log_level)

            # Use detailed format for files if format is standard/colored
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

            # Optionally wrap in QueueHandler for non-blocking async logging
            if use_queue_handler:
                import queue as _queue_mod
                log_queue = _queue_mod.Queue(maxsize=queue_max_size)  # type: ignore[var-annotated]
                queue_handler = logging.handlers.QueueHandler(log_queue)
                queue_handler.setLevel(log_level)
                for f in filters:
                    queue_handler.addFilter(f)
                queue_listener = logging.handlers.QueueListener(
                    log_queue,
                    file_handler,
                    respect_handler_level=True,
                )
                queue_listener.start()
                # Store listener reference on handler for cleanup
                queue_handler._queue_listener = queue_listener  # type: ignore[attr-defined]
                root_logger.addHandler(queue_handler)
                _managed_handlers.append(queue_handler)
                # Keep file_handler alive via the listener (not added to root directly)
                _managed_handlers.append(file_handler)
            else:
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
            rotation_mode=rotation_mode,
            rotation_when=rotation_when,
            rotation_interval=rotation_interval,
            use_queue_handler=use_queue_handler,
            queue_max_size=queue_max_size,
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
            # Stop QueueListener if this is a QueueHandler with one attached
            listener = getattr(handler, "_queue_listener", None)
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass
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
        if isinstance(handler, logging.handlers.TimedRotatingFileHandler):
            handler_info["file"] = handler.baseFilename
            handler_info["rotation_when"] = handler.when
            handler_info["rotation_interval"] = getattr(handler, "_rotation_interval", handler.interval)
            handler_info["backup_count"] = handler.backupCount
            handler_info["rotation_mode"] = "time"
        elif isinstance(handler, logging.handlers.RotatingFileHandler):
            handler_info["file"] = handler.baseFilename
            handler_info["max_bytes"] = handler.maxBytes
            handler_info["backup_count"] = handler.backupCount
            handler_info["rotation_mode"] = "size"
        elif isinstance(handler, logging.handlers.QueueHandler):
            handler_info["queue_maxsize"] = handler.queue.maxsize
            listener = getattr(handler, "_queue_listener", None)
            handler_info["queue_listener_active"] = listener is not None
        status["handlers"].append(handler_info)

    if _active_config:
        status["config"] = _active_config.to_dict()

    return status


def set_log_level(level: str) -> int:
    """Change the active log level without reconfiguring logging.

    Updates the root logger and all managed handlers to the new level.
    The active configuration is also updated.

    Args:
        level: The new log level (debug, info, warning, error, critical).

    Returns:
        The numeric log level that was applied.

    Raises:
        ValueError: If the level name is invalid.

    Example::

        # Temporarily increase verbosity at runtime
        set_log_level("debug")
        logger.debug("Verbose diagnostics enabled")
        set_log_level("info")
    """
    global _active_config

    numeric = _resolve_log_level(level)

    with _setup_lock:
        root_logger = logging.getLogger(ROOT_LOGGER_NAME)
        root_logger.setLevel(numeric)

        for handler in _managed_handlers:
            # Don't lower the level below the handler's explicit level
            if handler.level == logging.NOTSET or handler.level >= 0:
                handler.setLevel(numeric)

        if _active_config is not None:
            _active_config.level = level

    return numeric


def get_log_files() -> list[dict[str, Any]]:
    """List all active log files and their rotation metadata.

    Scans managed file handlers to report current log file paths,
    sizes, and rotation settings.

    Returns:
        List of dictionaries with file information. Each dict contains:
        - path: Absolute path to the log file
        - size_bytes: Current file size
        - rotation_mode: "size" or "time"
        - backup_count: Number of backup files kept
        - backup_files: List of existing backup file paths

    Example::

        files = get_log_files()
        for f in files:
            print(f["path"], f["size_bytes"])
    """
    results: list[dict[str, Any]] = []

    for handler in _managed_handlers:
        if isinstance(handler, (
            logging.handlers.RotatingFileHandler,
            logging.handlers.TimedRotatingFileHandler,
        )):
            file_path = Path(handler.baseFilename)
            size = file_path.stat().st_size if file_path.exists() else 0

            if isinstance(handler, logging.handlers.TimedRotatingFileHandler):
                rotation_mode = "time"
                extra: dict[str, Any] = {
                    "rotation_when": handler.when,
                    "rotation_interval": getattr(handler, "_rotation_interval", handler.interval),
                }
            else:
                rotation_mode = "size"
                extra = {"max_bytes": handler.maxBytes}

            # Find existing backup files
            parent = file_path.parent
            base = file_path.name
            backup_files = sorted(
                str(p) for p in parent.glob(f"{base}.*")
                if p != file_path
            )

            results.append({
                "path": str(file_path),
                "size_bytes": size,
                "rotation_mode": rotation_mode,
                "backup_count": handler.backupCount,
                "backup_files": backup_files,
                **extra,
            })

    return results


# ---------------------------------------------------------------------------
# Sensitive Data Redaction Filter
# ---------------------------------------------------------------------------


class SensitiveDataFilter(logging.Filter):
    """Filter that redacts sensitive data from log messages.

    Automatically detects and masks API keys, tokens, passwords,
    and other secrets that might appear in log output.

    Args:
        patterns: Additional regex patterns to redact (compiled or strings).
        replacement: Replacement text for redacted content.
        redact_extras: Whether to also redact sensitive keys in extra fields.
    """

    import re as _re

    # Default patterns for common sensitive data
    DEFAULT_PATTERNS = [
        # API keys (various formats)
        _re.compile(r"(sk-ant-[a-zA-Z0-9\-]{20,})", _re.IGNORECASE),
        _re.compile(r"(sk-[a-zA-Z0-9]{20,})", _re.IGNORECASE),
        _re.compile(r"(api[_-]?key\s*[:=]\s*)[\"']?([a-zA-Z0-9\-_]{20,})[\"']?", _re.IGNORECASE),
        # Bearer tokens
        _re.compile(r"(Bearer\s+)([a-zA-Z0-9\-_.]+)", _re.IGNORECASE),
        # Authorization headers
        _re.compile(r"(Authorization\s*[:=]\s*)[\"']?([^\s\"']+)[\"']?", _re.IGNORECASE),
        # Password patterns
        _re.compile(r"(password\s*[:=]\s*)[\"']?([^\s\"',}]+)[\"']?", _re.IGNORECASE),
        # x-api-key header values
        _re.compile(r"(x-api-key\s*[:=]\s*)[\"']?([a-zA-Z0-9\-_]{8,})[\"']?", _re.IGNORECASE),
        # Generic secret/token patterns
        _re.compile(r"(secret\s*[:=]\s*)[\"']?([^\s\"',}]+)[\"']?", _re.IGNORECASE),
        _re.compile(r"(token\s*[:=]\s*)[\"']?([a-zA-Z0-9\-_.]{20,})[\"']?", _re.IGNORECASE),
    ]

    # Keys in extra fields that should be redacted
    SENSITIVE_KEYS = {
        "api_key", "apikey", "api-key",
        "token", "access_token", "refresh_token",
        "password", "passwd", "secret",
        "authorization", "x-api-key",
        "credentials", "private_key",
    }

    def __init__(
        self,
        patterns: Optional[list] = None,
        replacement: str = "***REDACTED***",
        redact_extras: bool = True,
    ) -> None:
        super().__init__()
        import re

        self.replacement = replacement
        self.redact_extras = redact_extras
        self._patterns = list(self.DEFAULT_PATTERNS)
        if patterns:
            for p in patterns:
                if isinstance(p, str):
                    self._patterns.append(re.compile(p, re.IGNORECASE))
                else:
                    self._patterns.append(p)

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from the log record."""
        # Redact the message
        if isinstance(record.msg, str):
            record.msg = self._redact_string(record.msg)

        # Redact args if they're strings
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact_string(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._redact_string(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )

        # Redact sensitive extra fields
        if self.redact_extras:
            for key in self.SENSITIVE_KEYS:
                if hasattr(record, key):
                    setattr(record, key, self.replacement)

        return True

    def _redact_string(self, text: str) -> str:
        """Apply all redaction patterns to a string."""
        for pattern in self._patterns:
            groups = pattern.groups if hasattr(pattern, 'groups') else 0
            if groups >= 2:
                # Pattern has groups — replace the second group (the secret)
                text = pattern.sub(
                    lambda m: m.group(1) + self.replacement, text
                )
            else:
                # Full match pattern — replace entire match
                text = pattern.sub(self.replacement, text)
        return text


# ---------------------------------------------------------------------------
# Sampling Filter (for high-throughput scenarios)
# ---------------------------------------------------------------------------


class SamplingFilter(logging.Filter):
    """Filter that samples log messages for high-throughput scenarios.

    Only passes through a configurable percentage of messages at each level.
    ERROR and CRITICAL messages always pass through.

    Args:
        rates: Dictionary mapping level names to sample rates (0.0 to 1.0).
            Levels not in the dictionary default to 1.0 (pass all).
        default_rate: Default sampling rate for unspecified levels.
        always_pass_levels: Set of level names that always pass through.
    """

    import random as _random

    def __init__(
        self,
        rates: Optional[dict[str, float]] = None,
        default_rate: float = 1.0,
        always_pass_levels: Optional[set[str]] = None,
    ) -> None:
        super().__init__()
        self.rates = {
            k.upper(): v for k, v in (rates or {}).items()
        }
        self.default_rate = default_rate
        self.always_pass_levels = {
            level.upper() for level in (always_pass_levels or {"ERROR", "CRITICAL"})
        }
        self._lock = threading.Lock()
        self._total_count = 0
        self._passed_count = 0
        self._suppressed_by_level: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Apply sampling to determine if the record passes."""
        import random

        level_name = record.levelname

        with self._lock:
            self._total_count += 1

        # Always pass ERROR/CRITICAL
        if level_name in self.always_pass_levels:
            with self._lock:
                self._passed_count += 1
            return True

        rate = self.rates.get(level_name, self.default_rate)

        if rate >= 1.0:
            with self._lock:
                self._passed_count += 1
            return True

        if rate <= 0.0:
            with self._lock:
                self._suppressed_by_level[level_name] = (
                    self._suppressed_by_level.get(level_name, 0) + 1
                )
            return False

        if random.random() < rate:
            with self._lock:
                self._passed_count += 1
            return True

        with self._lock:
            self._suppressed_by_level[level_name] = (
                self._suppressed_by_level.get(level_name, 0) + 1
            )
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get sampling statistics."""
        with self._lock:
            return {
                "total_count": self._total_count,
                "passed_count": self._passed_count,
                "suppressed_by_level": dict(self._suppressed_by_level),
                "pass_rate": (
                    self._passed_count / self._total_count
                    if self._total_count > 0
                    else 0.0
                ),
            }

    def reset_stats(self) -> None:
        """Reset sampling statistics."""
        with self._lock:
            self._total_count = 0
            self._passed_count = 0
            self._suppressed_by_level.clear()


# ---------------------------------------------------------------------------
# Log Metrics Collector
# ---------------------------------------------------------------------------


class LogMetrics:
    """Collects metrics about log messages.

    Tracks counts by level, module, and time window. Thread-safe.

    Example::

        metrics = LogMetrics()
        handler = LogMetricsHandler(metrics)
        logger.addHandler(handler)

        # Later:
        report = metrics.get_report()
        print(f"Errors in last minute: {report['levels']['ERROR']}")
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._lock = threading.Lock()
        self._window_seconds = window_seconds
        self._level_counts: dict[str, int] = {}
        self._module_counts: dict[str, int] = {}
        self._total_count = 0
        self._error_timestamps: list[float] = []
        self._recent_errors: list[dict[str, Any]] = []
        self._max_recent_errors = 50
        self._start_time = time.monotonic()

    def record(self, record: logging.LogRecord) -> None:
        """Record a log event."""
        now = time.monotonic()
        with self._lock:
            self._total_count += 1

            # Count by level
            level = record.levelname
            self._level_counts[level] = self._level_counts.get(level, 0) + 1

            # Count by module
            module = record.name
            self._module_counts[module] = self._module_counts.get(module, 0) + 1

            # Track error timestamps for rate calculation
            if record.levelno >= logging.ERROR:
                self._error_timestamps.append(now)
                # Keep only timestamps within the window
                cutoff = now - self._window_seconds
                self._error_timestamps = [
                    t for t in self._error_timestamps if t > cutoff
                ]
                # Store recent error details
                error_info: dict[str, Any] = {
                    "timestamp": record.created,
                    "level": level,
                    "module": record.name,
                    "message": record.getMessage()[:200],
                }
                if record.exc_info and record.exc_info[0]:
                    error_info["exception_type"] = record.exc_info[0].__name__
                self._recent_errors.append(error_info)
                if len(self._recent_errors) > self._max_recent_errors:
                    self._recent_errors.pop(0)

    def get_report(self) -> dict[str, Any]:
        """Get a metrics report."""
        now = time.monotonic()
        with self._lock:
            uptime = now - self._start_time
            cutoff = now - self._window_seconds
            recent_errors = len([
                t for t in self._error_timestamps if t > cutoff
            ])

            return {
                "total_count": self._total_count,
                "uptime_seconds": round(uptime, 1),
                "levels": dict(self._level_counts),
                "top_modules": dict(
                    sorted(
                        self._module_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]
                ),
                "error_rate_per_minute": round(
                    recent_errors * (60.0 / self._window_seconds), 2
                ),
                "recent_errors": list(self._recent_errors[-10:]),
            }

    def get_error_rate(self) -> float:
        """Get the current error rate (errors per minute)."""
        now = time.monotonic()
        with self._lock:
            cutoff = now - self._window_seconds
            recent_errors = len([
                t for t in self._error_timestamps if t > cutoff
            ])
            return recent_errors * (60.0 / self._window_seconds)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._level_counts.clear()
            self._module_counts.clear()
            self._total_count = 0
            self._error_timestamps.clear()
            self._recent_errors.clear()
            self._start_time = time.monotonic()


class LogMetricsHandler(logging.Handler):
    """Handler that feeds log records into a LogMetrics collector.

    Add this handler to any logger to collect metrics without
    affecting other handlers.

    Args:
        metrics: The LogMetrics instance to record into.
    """

    def __init__(self, metrics: LogMetrics) -> None:
        super().__init__()
        self.metrics = metrics

    def emit(self, record: logging.LogRecord) -> None:
        """Record the log event in the metrics collector."""
        self.metrics.record(record)


# Module-level default metrics instance
_default_metrics: Optional[LogMetrics] = None


def get_log_metrics() -> LogMetrics:
    """Get or create the default LogMetrics instance.

    Returns:
        The global LogMetrics instance.
    """
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = LogMetrics()
    return _default_metrics


def reset_log_metrics() -> None:
    """Reset the default LogMetrics instance."""
    global _default_metrics
    if _default_metrics is not None:
        _default_metrics.reset()
    _default_metrics = None


# ---------------------------------------------------------------------------
# Structured Logger Wrapper
# ---------------------------------------------------------------------------


class StructuredLogger:
    """Wrapper around a standard logger that simplifies structured logging.

    Provides a fluent interface for attaching structured metadata
    to log messages without needing to manually build ``extra`` dicts.

    Example::

        slog = StructuredLogger("my_module")
        slog.info("request processed",
            method="POST", path="/v1/messages",
            status_code=200, duration_ms=42.5,
        )

        # With binding (creates a child logger with default fields)
        req_log = slog.bind(request_id="abc-123", client_ip="1.2.3.4")
        req_log.info("handling request")
        req_log.info("request complete", status=200)
    """

    def __init__(
        self,
        name: str,
        logger: Optional[logging.Logger] = None,
        default_fields: Optional[dict[str, Any]] = None,
    ) -> None:
        self._logger = logger or get_logger(name)
        self._default_fields: dict[str, Any] = default_fields or {}

    @property
    def name(self) -> str:
        """The underlying logger name."""
        return self._logger.name

    def bind(self, **fields: Any) -> "StructuredLogger":
        """Create a child logger with additional default fields.

        Args:
            **fields: Key-value pairs to include in every log message.

        Returns:
            A new StructuredLogger with the combined default fields.
        """
        combined = {**self._default_fields, **fields}
        return StructuredLogger(
            name=self._logger.name,
            logger=self._logger,
            default_fields=combined,
        )

    def unbind(self, *keys: str) -> "StructuredLogger":
        """Create a child logger without specified default fields.

        Args:
            *keys: Field names to remove from defaults.

        Returns:
            A new StructuredLogger without the specified fields.
        """
        remaining = {
            k: v for k, v in self._default_fields.items() if k not in keys
        }
        return StructuredLogger(
            name=self._logger.name,
            logger=self._logger,
            default_fields=remaining,
        )

    def _log(self, level: int, msg: str, **fields: Any) -> None:
        """Internal log method that merges default and provided fields."""
        extra = {**self._default_fields, **fields}
        self._logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **fields: Any) -> None:
        """Log a DEBUG message with structured fields."""
        self._log(logging.DEBUG, msg, **fields)

    def info(self, msg: str, **fields: Any) -> None:
        """Log an INFO message with structured fields."""
        self._log(logging.INFO, msg, **fields)

    def warning(self, msg: str, **fields: Any) -> None:
        """Log a WARNING message with structured fields."""
        self._log(logging.WARNING, msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        """Log an ERROR message with structured fields."""
        self._log(logging.ERROR, msg, **fields)

    def critical(self, msg: str, **fields: Any) -> None:
        """Log a CRITICAL message with structured fields."""
        self._log(logging.CRITICAL, msg, **fields)

    def exception(self, msg: str, **fields: Any) -> None:
        """Log an ERROR message with exception info and structured fields."""
        extra = {**self._default_fields, **fields}
        self._logger.exception(msg, extra=extra)

    def log(self, level: int, msg: str, **fields: Any) -> None:
        """Log at an arbitrary level with structured fields."""
        self._log(level, msg, **fields)


# ---------------------------------------------------------------------------
# Enhanced Request Context
# ---------------------------------------------------------------------------


_request_context: contextvars.ContextVar[Optional[dict[str, Any]]] = (
    contextvars.ContextVar("request_context", default=None)
)


def get_request_context() -> Optional[dict[str, Any]]:
    """Get the current request context metadata.

    Returns:
        Dictionary of request-scoped metadata, or None if not set.
    """
    return _request_context.get()


def set_request_context(context: dict[str, Any]) -> dict[str, Any]:
    """Set the request context metadata.

    Args:
        context: Dictionary of request-scoped metadata.

    Returns:
        The context that was set.
    """
    _request_context.set(context)
    return context


def clear_request_context() -> None:
    """Clear the request context."""
    _request_context.set(None)


def update_request_context(**fields: Any) -> dict[str, Any]:
    """Update the current request context with additional fields.

    Creates a new context if none exists.

    Args:
        **fields: Key-value pairs to add to the context.

    Returns:
        The updated context.
    """
    current = _request_context.get() or {}
    current.update(fields)
    _request_context.set(current)
    return current


@contextmanager
def request_context(
    correlation_id: Optional[str] = None,
    **initial_fields: Any,
):
    """Context manager for request-scoped logging context.

    Sets up both the correlation ID and request context metadata,
    and cleans up on exit.

    Args:
        correlation_id: Optional correlation ID (auto-generated if None).
        **initial_fields: Initial request context fields.

    Yields:
        Tuple of (correlation_id, context_dict).

    Example::

        with request_context(method="POST", path="/v1/messages") as (cid, ctx):
            logger.info("Processing request %s", cid)
            update_request_context(status_code=200)
    """
    cid = set_correlation_id(correlation_id)
    ctx = set_request_context({"correlation_id": cid, **initial_fields})
    try:
        yield cid, ctx
    finally:
        clear_correlation_id()
        clear_request_context()


class RequestContextFilter(logging.Filter):
    """Filter that injects request context fields into log records.

    Adds all fields from the current request context as attributes
    on the log record, making them available to formatters.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request context fields to the record."""
        ctx = get_request_context()
        if ctx:
            for key, value in ctx.items():
                if not hasattr(record, key):
                    setattr(record, key, value)
        return True


# ---------------------------------------------------------------------------
# Log Buffer (for batching log entries)
# ---------------------------------------------------------------------------


class LogBuffer:
    """Thread-safe buffer for collecting and batching log records.

    Useful for aggregating log entries before sending to external
    services or for deferred output.

    Args:
        max_size: Maximum number of entries to buffer.
        flush_interval: Seconds between automatic flushes (0 = manual only).
        on_flush: Callback invoked when the buffer is flushed.
            Receives a list of formatted log entries.
    """

    def __init__(
        self,
        max_size: int = 1000,
        flush_interval: float = 0.0,
        on_flush: Optional[Callable[[list[dict[str, Any]]], None]] = None,
    ) -> None:
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.on_flush = on_flush
        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._flush_timer: Optional[threading.Timer] = None
        self._total_flushed = 0
        self._total_dropped = 0

        if flush_interval > 0 and on_flush:
            self._start_flush_timer()

    def add(self, record: logging.LogRecord) -> bool:
        """Add a log record to the buffer.

        Args:
            record: The log record to buffer.

        Returns:
            True if added, False if the buffer is full and the entry was dropped.
        """
        entry = {
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

        cid = get_correlation_id()
        if cid:
            entry["correlation_id"] = cid

        # Add extra fields
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
                    json.dumps(value)
                    entry[key] = value
                except (TypeError, ValueError):
                    entry[key] = str(value)

        with self._lock:
            if len(self._buffer) >= self.max_size:
                self._total_dropped += 1
                return False
            self._buffer.append(entry)

            # Auto-flush if buffer is at capacity
            if len(self._buffer) >= self.max_size and self.on_flush:
                self._do_flush()

        return True

    def flush(self) -> list[dict[str, Any]]:
        """Flush the buffer and return all entries.

        Returns:
            List of buffered log entries.
        """
        with self._lock:
            return self._do_flush()

    def _do_flush(self) -> list[dict[str, Any]]:
        """Internal flush (must be called with lock held)."""
        entries = list(self._buffer)
        self._buffer.clear()
        self._total_flushed += len(entries)

        if entries and self.on_flush:
            try:
                self.on_flush(entries)
            except Exception:
                pass  # Don't let flush callbacks break logging

        return entries

    def _start_flush_timer(self) -> None:
        """Start the periodic flush timer."""
        if self.flush_interval <= 0:
            return

        def _timer_flush():
            with self._lock:
                if self._buffer:
                    self._do_flush()
            self._start_flush_timer()

        self._flush_timer = threading.Timer(self.flush_interval, _timer_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def stop(self) -> list[dict[str, Any]]:
        """Stop the buffer and flush remaining entries.

        Returns:
            Any remaining buffered entries.
        """
        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None
        return self.flush()

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "current_size": len(self._buffer),
                "max_size": self.max_size,
                "total_flushed": self._total_flushed,
                "total_dropped": self._total_dropped,
            }

    @property
    def size(self) -> int:
        """Current number of buffered entries."""
        with self._lock:
            return len(self._buffer)


class BufferedHandler(logging.Handler):
    """Handler that writes log records into a LogBuffer.

    Args:
        buffer: The LogBuffer to write into.
    """

    def __init__(self, buffer: LogBuffer) -> None:
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        """Add the record to the buffer."""
        self.buffer.add(record)

    def close(self) -> None:
        """Flush the buffer on handler close."""
        self.buffer.flush()
        super().close()


# ---------------------------------------------------------------------------
# Exception Formatting Helpers
# ---------------------------------------------------------------------------


def format_exception_context(
    exc: Exception,
    include_chain: bool = True,
    max_depth: int = 5,
) -> dict[str, Any]:
    """Format an exception with full context for structured logging.

    Captures the exception type, message, traceback, and any chained
    exceptions (cause or context).

    Args:
        exc: The exception to format.
        include_chain: Whether to include chained exceptions.
        max_depth: Maximum depth for chained exceptions.

    Returns:
        Dictionary with structured exception information.
    """
    import traceback as tb_module

    result: dict[str, Any] = {
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "message": str(exc),
        "traceback": tb_module.format_exception(type(exc), exc, exc.__traceback__),
    }

    # Include GatewayError context if available
    from src.errors import GatewayError

    if isinstance(exc, GatewayError):
        result["error_context"] = exc.context.to_dict()
        result["is_retryable"] = exc.is_retryable

    # Include chained exceptions
    if include_chain and max_depth > 0:
        chain = []
        current = exc.__cause__ or exc.__context__
        depth = 0
        while current and depth < max_depth:
            chain_entry: dict[str, Any] = {
                "type": type(current).__name__,
                "message": str(current),
                "explicit": exc.__cause__ is not None,
            }
            if isinstance(current, GatewayError):
                chain_entry["error_context"] = current.context.to_dict()
            chain.append(chain_entry)
            current = current.__cause__ or current.__context__
            depth += 1
        if chain:
            result["chain"] = chain

    return result


def log_exception(
    logger: logging.Logger,
    msg: str,
    exc: Exception,
    level: int = logging.ERROR,
    **extra_fields: Any,
) -> None:
    """Log an exception with structured context.

    Combines a log message with detailed exception information
    as extra structured data.

    Args:
        logger: Logger to use.
        msg: Log message.
        exc: The exception to log.
        level: Log level (default: ERROR).
        **extra_fields: Additional structured data.
    """
    exc_context = format_exception_context(exc)
    extra = {
        "exception_type": exc_context["type"],
        "exception_message": exc_context["message"],
        **extra_fields,
    }
    if "error_context" in exc_context:
        extra["error_context"] = exc_context["error_context"]
    if "is_retryable" in exc_context:
        extra["is_retryable"] = exc_context["is_retryable"]
    if "chain" in exc_context:
        extra["exception_chain"] = exc_context["chain"]

    logger.log(level, msg, extra=extra, exc_info=(type(exc), exc, exc.__traceback__))


# ---------------------------------------------------------------------------
# Health / Heartbeat Logging
# ---------------------------------------------------------------------------


class HealthLogger:
    """Periodic health/heartbeat logger.

    Emits periodic log entries with system health information.
    Useful for monitoring that the service is alive and healthy.

    Args:
        interval: Seconds between heartbeat messages.
        logger: Logger to use. Defaults to gateway.health.
        include_metrics: Whether to include LogMetrics in heartbeats.
    """

    def __init__(
        self,
        interval: float = 60.0,
        logger: Optional[logging.Logger] = None,
        include_metrics: bool = True,
    ) -> None:
        self.interval = interval
        self._logger = logger or get_logger("health")
        self.include_metrics = include_metrics
        self._running = False
        self._timer: Optional[threading.Timer] = None
        self._start_time: Optional[float] = None
        self._heartbeat_count = 0
        self._custom_checks: dict[str, Callable[[], dict[str, Any]]] = {}

    def register_check(
        self,
        name: str,
        check: Callable[[], dict[str, Any]],
    ) -> None:
        """Register a custom health check.

        Args:
            name: Name of the health check.
            check: Callable that returns a dict with health information.
        """
        self._custom_checks[name] = check

    def unregister_check(self, name: str) -> None:
        """Remove a registered health check."""
        self._custom_checks.pop(name, None)

    def start(self) -> None:
        """Start periodic heartbeat logging."""
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._heartbeat_count = 0
        self._logger.info("Health logger started (interval=%ss)", self.interval)
        self._schedule_heartbeat()

    def stop(self) -> None:
        """Stop periodic heartbeat logging."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
        self._logger.info(
            "Health logger stopped (heartbeats=%d)", self._heartbeat_count
        )

    def _schedule_heartbeat(self) -> None:
        """Schedule the next heartbeat."""
        if not self._running:
            return
        self._timer = threading.Timer(self.interval, self._emit_heartbeat)
        self._timer.daemon = True
        self._timer.start()

    def _emit_heartbeat(self) -> None:
        """Emit a heartbeat log entry."""
        if not self._running:
            return

        self._heartbeat_count += 1
        uptime = time.monotonic() - self._start_time if self._start_time else 0

        health_data: dict[str, Any] = {
            "heartbeat_count": self._heartbeat_count,
            "uptime_seconds": round(uptime, 1),
            "status": "healthy",
        }

        # Include log metrics if available
        if self.include_metrics and _default_metrics is not None:
            metrics_report = _default_metrics.get_report()
            health_data["log_metrics"] = {
                "total_messages": metrics_report["total_count"],
                "error_rate_per_minute": metrics_report["error_rate_per_minute"],
                "levels": metrics_report["levels"],
            }

        # Run custom health checks
        checks_status: dict[str, Any] = {}
        for name, check in self._custom_checks.items():
            try:
                checks_status[name] = check()
            except Exception as e:
                checks_status[name] = {
                    "status": "error",
                    "error": str(e),
                }
                health_data["status"] = "degraded"

        if checks_status:
            health_data["checks"] = checks_status

        self._logger.info(
            "heartbeat #%d (uptime=%ss, status=%s)",
            self._heartbeat_count,
            round(uptime, 0),
            health_data["status"],
            extra=health_data,
        )

        # Schedule next heartbeat
        self._schedule_heartbeat()

    def emit_now(self) -> dict[str, Any]:
        """Emit a heartbeat immediately and return the health data.

        Returns:
            Dictionary with current health information.
        """
        self._heartbeat_count += 1
        uptime = time.monotonic() - self._start_time if self._start_time else 0

        health_data: dict[str, Any] = {
            "heartbeat_count": self._heartbeat_count,
            "uptime_seconds": round(uptime, 1),
            "status": "healthy",
        }

        if self.include_metrics and _default_metrics is not None:
            health_data["log_metrics"] = _default_metrics.get_report()

        for name, check in self._custom_checks.items():
            try:
                health_data.setdefault("checks", {})[name] = check()
            except Exception as e:
                health_data.setdefault("checks", {})[name] = {
                    "status": "error",
                    "error": str(e),
                }
                health_data["status"] = "degraded"

        return health_data


# ---------------------------------------------------------------------------
# Enhanced setup_logging with new features
# ---------------------------------------------------------------------------


def setup_logging_advanced(
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
    enable_metrics: bool = False,
    enable_sensitive_filter: bool = False,
    sensitive_patterns: Optional[list[str]] = None,
    sampling_rates: Optional[dict[str, float]] = None,
    enable_request_context: bool = False,
    buffer_config: Optional[dict[str, Any]] = None,
    config: Optional[LoggingConfig] = None,
) -> logging.Logger:
    """Advanced logging setup with all extended features.

    Extends :func:`setup_logging` with metrics collection, sensitive data
    redaction, sampling, request context, and buffered output.

    Args:
        level: Minimum log level.
        log_format: Output format.
        output: Output destination.
        log_file: Path to log file.
        max_bytes: Max log file size before rotation.
        backup_count: Number of rotated files to keep.
        include_correlation_id: Include correlation IDs.
        rate_limit_seconds: Rate limiting interval.
        module_filters: Module name prefixes to include.
        json_extras: Include extras in JSON output.
        enable_metrics: Enable LogMetrics collection.
        enable_sensitive_filter: Enable sensitive data redaction.
        sensitive_patterns: Additional regex patterns for redaction.
        sampling_rates: Dict of level -> sample rate for SamplingFilter.
        enable_request_context: Enable RequestContextFilter.
        buffer_config: Optional config for LogBuffer (keys: max_size,
            flush_interval, on_flush).
        config: Optional LoggingConfig object.

    Returns:
        The configured root application logger.
    """
    # First, do the standard setup
    root_logger = setup_logging(
        level=level,
        log_format=log_format,
        output=output,
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
        include_correlation_id=include_correlation_id,
        rate_limit_seconds=rate_limit_seconds,
        module_filters=module_filters,
        json_extras=json_extras,
        config=config,
    )

    # Add metrics handler
    if enable_metrics:
        metrics = get_log_metrics()
        metrics_handler = LogMetricsHandler(metrics)
        metrics_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(metrics_handler)
        _managed_handlers.append(metrics_handler)

    # Add sensitive data filter to all handlers
    if enable_sensitive_filter:
        sensitive_filter = SensitiveDataFilter(patterns=sensitive_patterns)
        for handler in root_logger.handlers:
            handler.addFilter(sensitive_filter)

    # Add sampling filter
    if sampling_rates:
        sampling_filter = SamplingFilter(rates=sampling_rates)
        for handler in root_logger.handlers:
            handler.addFilter(sampling_filter)

    # Add request context filter
    if enable_request_context:
        ctx_filter = RequestContextFilter()
        for handler in root_logger.handlers:
            handler.addFilter(ctx_filter)

    # Add buffered handler
    if buffer_config:
        log_buffer = LogBuffer(
            max_size=buffer_config.get("max_size", 1000),
            flush_interval=buffer_config.get("flush_interval", 0.0),
            on_flush=buffer_config.get("on_flush"),
        )
        buffered_handler = BufferedHandler(log_buffer)
        buffered_handler.setLevel(
            _resolve_log_level(level if config is None else config.level)
        )
        root_logger.addHandler(buffered_handler)
        _managed_handlers.append(buffered_handler)

    return root_logger
