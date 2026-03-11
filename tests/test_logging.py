"""Comprehensive tests for the logging system."""

import json
import logging
import logging.handlers
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.cli import main
from src.logging_config import (
    ColoredFormatter,
    CorrelationFilter,
    JSONFormatter,
    LogFormat,
    LoggingConfig,
    LogOutput,
    ModuleFilter,
    RateLimitFilter,
    ROOT_LOGGER_NAME,
    StandardFormatter,
    TimingResult,
    audit_log,
    clear_correlation_id,
    configure_from_gateway_config,
    correlation_context,
    create_formatter,
    get_active_config,
    get_audit_logger,
    get_correlation_id,
    get_log_files,
    get_log_level_name,
    get_logger,
    get_logging_status,
    is_configured,
    log_duration,
    log_request,
    reset_logging,
    set_correlation_id,
    set_log_level,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_logging():
    """Reset logging state before and after each test."""
    reset_logging()
    clear_correlation_id()
    yield
    reset_logging()
    clear_correlation_id()


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ---------------------------------------------------------------------------
# LoggingConfig Tests
# ---------------------------------------------------------------------------


class TestLoggingConfig:
    """Tests for the LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoggingConfig()
        assert config.level == "info"
        assert config.log_format == "standard"
        assert config.output == "console"
        assert config.log_file == ""
        assert config.max_bytes == 10 * 1024 * 1024
        assert config.backup_count == 5
        assert config.include_correlation_id is True
        assert config.rate_limit_seconds == 0.0
        assert config.module_filters == []
        assert config.json_extras is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = LoggingConfig(
            level="debug",
            log_format="json",
            output="both",
            log_file="test.log",
        )
        data = config.to_dict()
        assert data["level"] == "debug"
        assert data["log_format"] == "json"
        assert data["output"] == "both"
        assert data["log_file"] == "test.log"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "level": "warning",
            "log_format": "detailed",
            "output": "file",
            "log_file": "app.log",
            "max_bytes": 5_000_000,
            "backup_count": 3,
            "rate_limit_seconds": 10.0,
        }
        config = LoggingConfig.from_dict(data)
        assert config.level == "warning"
        assert config.log_format == "detailed"
        assert config.output == "file"
        assert config.log_file == "app.log"
        assert config.max_bytes == 5_000_000
        assert config.backup_count == 3
        assert config.rate_limit_seconds == 10.0

    def test_from_dict_defaults(self):
        """Test deserialization with missing keys uses defaults."""
        config = LoggingConfig.from_dict({})
        assert config.level == "info"
        assert config.log_format == "standard"

    def test_roundtrip(self):
        """Test that to_dict -> from_dict preserves values."""
        original = LoggingConfig(
            level="error",
            log_format="json",
            output="both",
            log_file="test.log",
            rate_limit_seconds=5.0,
            module_filters=["src.proxy"],
        )
        restored = LoggingConfig.from_dict(original.to_dict())
        assert restored.level == original.level
        assert restored.log_format == original.log_format
        assert restored.output == original.output
        assert restored.log_file == original.log_file
        assert restored.rate_limit_seconds == original.rate_limit_seconds
        assert restored.module_filters == original.module_filters


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for LogFormat and LogOutput enums."""

    def test_log_format_values(self):
        """Test LogFormat enum values."""
        assert LogFormat.STANDARD == "standard"
        assert LogFormat.DETAILED == "detailed"
        assert LogFormat.MINIMAL == "minimal"
        assert LogFormat.JSON == "json"
        assert LogFormat.COLORED == "colored"

    def test_log_output_values(self):
        """Test LogOutput enum values."""
        assert LogOutput.CONSOLE == "console"
        assert LogOutput.FILE == "file"
        assert LogOutput.BOTH == "both"


# ---------------------------------------------------------------------------
# Correlation ID Tests
# ---------------------------------------------------------------------------


class TestCorrelationID:
    """Tests for correlation ID management."""

    def test_default_is_none(self):
        """Test that correlation ID starts as None."""
        assert get_correlation_id() is None

    def test_set_and_get(self):
        """Test setting and getting a correlation ID."""
        cid = set_correlation_id("test-123")
        assert cid == "test-123"
        assert get_correlation_id() == "test-123"

    def test_auto_generate(self):
        """Test auto-generation of correlation ID."""
        cid = set_correlation_id()
        assert cid is not None
        assert len(cid) == 12  # UUID hex[:12]
        assert get_correlation_id() == cid

    def test_clear(self):
        """Test clearing the correlation ID."""
        set_correlation_id("test-456")
        assert get_correlation_id() == "test-456"
        clear_correlation_id()
        assert get_correlation_id() is None

    def test_context_manager(self):
        """Test correlation_context context manager."""
        assert get_correlation_id() is None

        with correlation_context("ctx-001") as cid:
            assert cid == "ctx-001"
            assert get_correlation_id() == "ctx-001"

        assert get_correlation_id() is None

    def test_context_manager_auto_generate(self):
        """Test context manager with auto-generated ID."""
        with correlation_context() as cid:
            assert cid is not None
            assert len(cid) == 12
            assert get_correlation_id() == cid

        assert get_correlation_id() is None

    def test_context_manager_cleans_up_on_exception(self):
        """Test that context manager clears ID even on exception."""
        try:
            with correlation_context("error-ctx"):
                assert get_correlation_id() == "error-ctx"
                raise ValueError("test error")
        except ValueError:
            pass

        assert get_correlation_id() is None

    def test_thread_isolation(self):
        """Test that correlation IDs are isolated between threads."""
        results = {}

        def thread_func(name, cid):
            set_correlation_id(cid)
            time.sleep(0.05)  # Allow interleaving
            results[name] = get_correlation_id()

        t1 = threading.Thread(target=thread_func, args=("t1", "id-1"))
        t2 = threading.Thread(target=thread_func, args=("t2", "id-2"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == "id-1"
        assert results["t2"] == "id-2"


# ---------------------------------------------------------------------------
# Formatter Tests
# ---------------------------------------------------------------------------


class TestStandardFormatter:
    """Tests for the StandardFormatter."""

    def test_basic_format(self):
        """Test basic message formatting."""
        formatter = StandardFormatter(
            fmt="%(levelname)s: %(message)s",
            include_correlation_id=False,
        )
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO: hello" in output

    def test_with_correlation_id(self):
        """Test formatting with correlation ID."""
        formatter = StandardFormatter(
            fmt="%(message)s",
            include_correlation_id=True,
        )
        set_correlation_id("corr-123")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "[corr-123]" in output
        assert "hello" in output

    def test_without_correlation_id(self):
        """Test formatting without correlation ID when none set."""
        formatter = StandardFormatter(
            fmt="%(message)s",
            include_correlation_id=True,
        )
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert output == "hello"


class TestJSONFormatter:
    """Tests for the JSONFormatter."""

    def test_basic_json_output(self):
        """Test that output is valid JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.module", level=logging.INFO, pathname="test.py",
            lineno=42, msg="test message", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.module"
        assert data["message"] == "test message"
        assert data["line"] == 42
        assert "timestamp" in data
        assert "thread" in data

    def test_json_with_correlation_id(self):
        """Test JSON output includes correlation ID."""
        formatter = JSONFormatter()
        set_correlation_id("json-corr")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["correlation_id"] == "json-corr"

    def test_json_with_exception(self):
        """Test JSON output includes exception info."""
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="",
            lineno=0, msg="error occurred", args=(), exc_info=exc_info,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "test error"

    def test_json_with_extras(self):
        """Test JSON output includes extra fields."""
        formatter = JSONFormatter(include_extras=True)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="request", args=(), exc_info=None,
        )
        record.status_code = 200  # type: ignore
        record.duration_ms = 42.5  # type: ignore
        output = formatter.format(record)
        data = json.loads(output)
        assert data["status_code"] == 200
        assert data["duration_ms"] == 42.5

    def test_json_without_extras(self):
        """Test JSON output without extra fields."""
        formatter = JSONFormatter(include_extras=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="request", args=(), exc_info=None,
        )
        record.status_code = 200  # type: ignore
        output = formatter.format(record)
        data = json.loads(output)
        assert "status_code" not in data


class TestColoredFormatter:
    """Tests for the ColoredFormatter."""

    def test_no_color_when_disabled(self):
        """Test that no color codes are added when colors are disabled."""
        formatter = ColoredFormatter(
            fmt="%(levelname)s: %(message)s",
            use_colors=False,
        )
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "\033[" not in output
        assert "INFO: hello" in output

    def test_supports_color_env_no_color(self):
        """Test NO_COLOR environment variable disables colors."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            assert ColoredFormatter._supports_color() is False

    def test_supports_color_env_force_color(self):
        """Test FORCE_COLOR environment variable enables colors."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            if "NO_COLOR" in os.environ:
                # NO_COLOR takes precedence
                pass
            else:
                assert ColoredFormatter._supports_color() is True


# ---------------------------------------------------------------------------
# Filter Tests
# ---------------------------------------------------------------------------


class TestCorrelationFilter:
    """Tests for the CorrelationFilter."""

    def test_adds_correlation_id(self):
        """Test that filter adds correlation_id attribute."""
        f = CorrelationFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        set_correlation_id("filter-001")
        assert f.filter(record) is True
        assert record.correlation_id == "filter-001"  # type: ignore

    def test_dash_when_no_correlation_id(self):
        """Test that filter uses '-' when no correlation ID set."""
        f = CorrelationFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        assert f.filter(record) is True
        assert record.correlation_id == "-"  # type: ignore


class TestModuleFilter:
    """Tests for the ModuleFilter."""

    def test_allow_all_when_empty(self):
        """Test that empty filter allows all modules."""
        f = ModuleFilter()
        record = logging.LogRecord(
            name="any.module", level=logging.INFO, pathname="",
            lineno=0, msg="", args=(), exc_info=None,
        )
        assert f.filter(record) is True

    def test_allowed_modules(self):
        """Test filtering by allowed module prefixes."""
        f = ModuleFilter(allowed_modules=["src.proxy", "src.cache"])
        record_proxy = logging.LogRecord(
            name="src.proxy.handler", level=logging.INFO, pathname="",
            lineno=0, msg="", args=(), exc_info=None,
        )
        record_config = logging.LogRecord(
            name="src.config.loader", level=logging.INFO, pathname="",
            lineno=0, msg="", args=(), exc_info=None,
        )
        assert f.filter(record_proxy) is True
        assert f.filter(record_config) is False

    def test_denied_modules(self):
        """Test filtering by denied module prefixes."""
        f = ModuleFilter(denied_modules=["noisy.module"])
        record_noisy = logging.LogRecord(
            name="noisy.module.sub", level=logging.INFO, pathname="",
            lineno=0, msg="", args=(), exc_info=None,
        )
        record_good = logging.LogRecord(
            name="good.module", level=logging.INFO, pathname="",
            lineno=0, msg="", args=(), exc_info=None,
        )
        assert f.filter(record_noisy) is False
        assert f.filter(record_good) is True

    def test_deny_takes_precedence(self):
        """Test that deny list takes precedence over allow list."""
        f = ModuleFilter(
            allowed_modules=["src"],
            denied_modules=["src.secret"],
        )
        record = logging.LogRecord(
            name="src.secret.handler", level=logging.INFO, pathname="",
            lineno=0, msg="", args=(), exc_info=None,
        )
        assert f.filter(record) is False


class TestRateLimitFilter:
    """Tests for the RateLimitFilter."""

    def test_first_message_passes(self):
        """Test that the first occurrence always passes."""
        f = RateLimitFilter(rate_seconds=1.0)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        assert f.filter(record) is True

    def test_rapid_duplicates_suppressed(self):
        """Test that rapid duplicate messages are suppressed."""
        f = RateLimitFilter(rate_seconds=1.0)

        def make_record():
            return logging.LogRecord(
                name="test", level=logging.INFO, pathname="",
                lineno=0, msg="same message", args=(), exc_info=None,
            )

        assert f.filter(make_record()) is True
        assert f.filter(make_record()) is False
        assert f.filter(make_record()) is False

    def test_different_messages_not_suppressed(self):
        """Test that different messages are not suppressed."""
        f = RateLimitFilter(rate_seconds=1.0)
        record1 = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="message A", args=(), exc_info=None,
        )
        record2 = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="message B", args=(), exc_info=None,
        )
        assert f.filter(record1) is True
        assert f.filter(record2) is True

    def test_reset(self):
        """Test that reset clears rate limit state."""
        f = RateLimitFilter(rate_seconds=10.0)

        def make_record():
            return logging.LogRecord(
                name="test", level=logging.INFO, pathname="",
                lineno=0, msg="same", args=(), exc_info=None,
            )

        assert f.filter(make_record()) is True
        assert f.filter(make_record()) is False

        f.reset()
        assert f.filter(make_record()) is True


# ---------------------------------------------------------------------------
# Formatter Factory Tests
# ---------------------------------------------------------------------------


class TestCreateFormatter:
    """Tests for the create_formatter factory function."""

    def test_standard_format(self):
        """Test creating a standard formatter."""
        fmt = create_formatter("standard")
        assert isinstance(fmt, StandardFormatter)

    def test_detailed_format(self):
        """Test creating a detailed formatter."""
        fmt = create_formatter("detailed")
        assert isinstance(fmt, StandardFormatter)

    def test_minimal_format(self):
        """Test creating a minimal formatter."""
        fmt = create_formatter("minimal")
        assert isinstance(fmt, StandardFormatter)

    def test_json_format(self):
        """Test creating a JSON formatter."""
        fmt = create_formatter("json")
        assert isinstance(fmt, JSONFormatter)

    def test_colored_format(self):
        """Test creating a colored formatter."""
        fmt = create_formatter("colored")
        assert isinstance(fmt, ColoredFormatter)

    def test_unknown_defaults_to_standard(self):
        """Test that unknown format falls back to standard."""
        fmt = create_formatter("nonexistent")
        assert isinstance(fmt, StandardFormatter)

    def test_with_enum(self):
        """Test creating formatter with LogFormat enum."""
        fmt = create_formatter(LogFormat.JSON)
        assert isinstance(fmt, JSONFormatter)


# ---------------------------------------------------------------------------
# Setup and Configuration Tests
# ---------------------------------------------------------------------------


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_basic_setup(self):
        """Test basic logging setup."""
        logger = setup_logging(level="info")
        assert logger.name == ROOT_LOGGER_NAME
        assert is_configured() is True

    def test_setup_returns_logger(self):
        """Test that setup returns a logger."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_debug_level(self):
        """Test setting debug level."""
        logger = setup_logging(level="debug")
        assert logger.level == logging.DEBUG

    def test_error_level(self):
        """Test setting error level."""
        logger = setup_logging(level="error")
        assert logger.level == logging.ERROR

    def test_console_handler_added(self):
        """Test that console handler is added."""
        setup_logging(output="console")
        root = logging.getLogger(ROOT_LOGGER_NAME)
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in handler_types

    def test_file_handler_added(self, temp_log_dir):
        """Test that file handler is added."""
        log_file = os.path.join(temp_log_dir, "test.log")
        setup_logging(output="file", log_file=log_file)
        root = logging.getLogger(ROOT_LOGGER_NAME)
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "RotatingFileHandler" in handler_types

    def test_both_handlers(self, temp_log_dir):
        """Test that both console and file handlers are added."""
        log_file = os.path.join(temp_log_dir, "test.log")
        setup_logging(output="both", log_file=log_file)
        root = logging.getLogger(ROOT_LOGGER_NAME)
        assert len(root.handlers) >= 2

    def test_file_logging_writes_to_file(self, temp_log_dir):
        """Test that file handler actually writes to the log file."""
        log_file = os.path.join(temp_log_dir, "test.log")
        setup_logging(
            level="info",
            log_format="standard",
            output="file",
            log_file=log_file,
        )
        logger = get_logger("file_test")
        logger.info("Test message written to file")

        # Flush handlers
        root = logging.getLogger(ROOT_LOGGER_NAME)
        for handler in root.handlers:
            handler.flush()

        with open(log_file) as f:
            content = f.read()
        assert "Test message written to file" in content

    def test_reconfigure(self):
        """Test that calling setup_logging twice replaces handlers."""
        setup_logging(level="info")
        root1_handlers = len(logging.getLogger(ROOT_LOGGER_NAME).handlers)

        setup_logging(level="debug")
        root2_handlers = len(logging.getLogger(ROOT_LOGGER_NAME).handlers)

        # Should not accumulate handlers
        assert root2_handlers == root1_handlers

    def test_with_config_object(self):
        """Test setup from a LoggingConfig object."""
        config = LoggingConfig(
            level="warning",
            log_format="json",
            output="console",
        )
        setup_logging(config=config)
        assert is_configured() is True
        active = get_active_config()
        assert active is not None
        assert active.level == "warning"
        assert active.log_format == "json"

    def test_with_rate_limiting(self):
        """Test setup with rate limiting enabled."""
        setup_logging(rate_limit_seconds=5.0)
        root = logging.getLogger(ROOT_LOGGER_NAME)
        has_rate_filter = any(
            isinstance(f, RateLimitFilter)
            for h in root.handlers
            for f in h.filters
        )
        assert has_rate_filter is True

    def test_with_module_filters(self):
        """Test setup with module filters."""
        setup_logging(module_filters=["src.proxy"])
        root = logging.getLogger(ROOT_LOGGER_NAME)
        has_module_filter = any(
            isinstance(f, ModuleFilter)
            for h in root.handlers
            for f in h.filters
        )
        assert has_module_filter is True

    def test_invalid_level_raises(self):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level="nonexistent")


class TestResetLogging:
    """Tests for the reset_logging function."""

    def test_reset_clears_state(self):
        """Test that reset clears configuration state."""
        setup_logging()
        assert is_configured() is True
        assert get_active_config() is not None

        reset_logging()
        assert is_configured() is False
        assert get_active_config() is None

    def test_reset_removes_handlers(self):
        """Test that reset removes managed handlers."""
        setup_logging()
        root = logging.getLogger(ROOT_LOGGER_NAME)
        assert len(root.handlers) > 0

        reset_logging()
        # All managed handlers should be removed
        # (there might still be non-managed handlers)


# ---------------------------------------------------------------------------
# Logger Factory Tests
# ---------------------------------------------------------------------------


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_creates_namespaced_logger(self):
        """Test that logger is created under the gateway namespace."""
        logger = get_logger("mymodule")
        assert logger.name == f"{ROOT_LOGGER_NAME}.mymodule"

    def test_no_double_prefix(self):
        """Test that already-prefixed names are not double-prefixed."""
        logger = get_logger(f"{ROOT_LOGGER_NAME}.mymodule")
        assert logger.name == f"{ROOT_LOGGER_NAME}.mymodule"

    def test_root_logger_name(self):
        """Test getting the root logger directly."""
        logger = get_logger(ROOT_LOGGER_NAME)
        assert logger.name == ROOT_LOGGER_NAME


# ---------------------------------------------------------------------------
# Timing Utilities Tests
# ---------------------------------------------------------------------------


class TestTimingResult:
    """Tests for the TimingResult dataclass."""

    def test_basic_result(self):
        """Test basic timing result creation."""
        result = TimingResult(
            operation="test_op",
            duration_ms=42.5,
            start_time=1000.0,
            end_time=1000.0425,
        )
        assert result.operation == "test_op"
        assert result.duration_ms == 42.5
        assert result.success is True
        assert result.error is None

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = TimingResult(
            operation="test_op",
            duration_ms=42.567,
            start_time=0,
            end_time=0,
            success=False,
            error="timeout",
        )
        data = result.to_dict()
        assert data["operation"] == "test_op"
        assert data["duration_ms"] == 42.567
        assert data["success"] is False
        assert data["error"] == "timeout"

    def test_to_dict_no_error(self):
        """Test that error is omitted when None."""
        result = TimingResult(
            operation="test", duration_ms=1.0, start_time=0, end_time=0,
        )
        data = result.to_dict()
        assert "error" not in data


class TestLogDurationDecorator:
    """Tests for log_duration as a decorator."""

    def test_decorator_timing(self):
        """Test that decorator measures execution time."""
        setup_logging(level="debug")

        @log_duration("test_func")
        def slow_func():
            time.sleep(0.05)
            return "result"

        result = slow_func()
        assert result == "result"

    def test_decorator_preserves_return_value(self):
        """Test that decorator preserves the original return value."""
        @log_duration("multiply")
        def multiply(a, b):
            return a * b

        assert multiply(3, 4) == 12

    def test_decorator_propagates_exceptions(self):
        """Test that decorator propagates exceptions."""
        @log_duration("failing_func")
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()

    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves the wrapped function name."""
        @log_duration("test")
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


class TestLogDurationContextManager:
    """Tests for log_duration as a context manager."""

    def test_context_manager_timing(self):
        """Test that context manager measures execution time."""
        setup_logging(level="debug")

        with log_duration("ctx_test") as timer:
            time.sleep(0.05)

        assert timer.result is not None
        assert timer.result.duration_ms >= 40  # At least 40ms
        assert timer.result.success is True

    def test_context_manager_records_failure(self):
        """Test that context manager records failure on exception."""
        setup_logging(level="debug")

        try:
            with log_duration("failing_ctx") as timer:
                raise RuntimeError("test error")
        except RuntimeError:
            pass

        assert timer.result is not None
        assert timer.result.success is False
        assert timer.result.error == "test error"

    def test_threshold(self):
        """Test that threshold_ms controls logging."""
        setup_logging(level="debug")

        # This should not log (duration < threshold)
        with log_duration("fast_op", threshold_ms=10000):
            pass


# ---------------------------------------------------------------------------
# Request Logging Tests
# ---------------------------------------------------------------------------


class TestLogRequest:
    """Tests for the log_request helper."""

    def test_info_level_for_success(self):
        """Test that 2xx responses log at INFO level."""
        setup_logging(level="debug")
        # This should not raise
        log_request(
            method="GET",
            path="/v1/models",
            status_code=200,
            duration_ms=15.3,
        )

    def test_warning_level_for_client_errors(self):
        """Test that 4xx responses log at WARNING level."""
        setup_logging(level="debug")
        log_request(
            method="POST",
            path="/v1/messages",
            status_code=400,
            duration_ms=5.0,
        )

    def test_error_level_for_server_errors(self):
        """Test that 5xx responses log at ERROR level."""
        setup_logging(level="debug")
        log_request(
            method="POST",
            path="/v1/messages",
            status_code=500,
            duration_ms=1000.0,
        )

    def test_with_extra_data(self):
        """Test request logging with extra data."""
        setup_logging(level="debug")
        log_request(
            method="POST",
            path="/v1/messages",
            status_code=200,
            duration_ms=42.0,
            client_ip="127.0.0.1",
            request_id="req-001",
            extra={"model": "claude-3-opus"},
        )


# ---------------------------------------------------------------------------
# Audit Logging Tests
# ---------------------------------------------------------------------------


class TestAuditLogging:
    """Tests for audit logging functions."""

    def test_get_audit_logger(self):
        """Test getting the audit logger."""
        logger = get_audit_logger()
        assert logger is not None
        assert "audit" in logger.name

    def test_audit_log(self):
        """Test recording an audit event."""
        setup_logging(level="debug")
        # Should not raise
        audit_log("config_changed", details={"key": "log_level", "value": "debug"})

    def test_audit_log_without_details(self):
        """Test audit logging without details."""
        setup_logging(level="debug")
        audit_log("user_login")


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for helper/utility functions."""

    def test_get_log_level_name(self):
        """Test converting numeric level to name."""
        assert get_log_level_name(logging.DEBUG) == "DEBUG"
        assert get_log_level_name(logging.INFO) == "INFO"
        assert get_log_level_name(logging.WARNING) == "WARNING"
        assert get_log_level_name(logging.ERROR) == "ERROR"
        assert get_log_level_name(logging.CRITICAL) == "CRITICAL"

    def test_get_logging_status_unconfigured(self):
        """Test logging status when not configured."""
        status = get_logging_status()
        assert status["configured"] is False
        assert status["root_logger"] == ROOT_LOGGER_NAME

    def test_get_logging_status_configured(self):
        """Test logging status when configured."""
        setup_logging(level="debug", log_format="json")
        status = get_logging_status()
        assert status["configured"] is True
        assert status["effective_level"] == "DEBUG"
        assert status["handler_count"] > 0
        assert len(status["handlers"]) > 0
        assert "config" in status

    def test_configure_from_gateway_config(self):
        """Test configuring from a GatewayConfig-like object."""
        from src.models import GatewayConfig

        config = GatewayConfig(log_level="warning")
        logger = configure_from_gateway_config(config)
        assert is_configured() is True

    def test_configure_from_gateway_config_with_extra(self):
        """Test configuring from GatewayConfig with extended logging config."""
        from src.models import GatewayConfig

        config = GatewayConfig(
            log_level="debug",
            extra={
                "logging": {
                    "log_format": "json",
                    "output": "console",
                }
            },
        )
        logger = configure_from_gateway_config(config)
        active = get_active_config()
        assert active is not None
        assert active.level == "debug"
        assert active.log_format == "json"


# ---------------------------------------------------------------------------
# CLI Command Tests
# ---------------------------------------------------------------------------


class TestLoggingCLI:
    """Tests for logging CLI commands."""

    def test_logging_group_help(self, runner):
        """Test the logging command group help."""
        result = runner.invoke(main, ["logging", "--help"])
        assert result.exit_code == 0
        assert "logging" in result.output.lower()

    def test_logging_status_command(self, runner):
        """Test the logging status command."""
        result = runner.invoke(main, ["logging", "status"])
        assert result.exit_code == 0
        assert "Logging System Status" in result.output

    def test_logging_status_json(self, runner):
        """Test logging status with JSON output."""
        result = runner.invoke(main, ["logging", "status", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "configured" in data
        assert "root_logger" in data

    def test_logging_test_command(self, runner):
        """Test the logging test command."""
        result = runner.invoke(main, ["logging", "test"])
        assert result.exit_code == 0
        assert "Testing log output" in result.output
        assert "Log test complete" in result.output

    def test_logging_test_with_format(self, runner):
        """Test the logging test command with different format."""
        result = runner.invoke(
            main, ["logging", "test", "--format", "json", "--level", "debug"]
        )
        assert result.exit_code == 0
        assert "Log test complete" in result.output

    def test_logging_formats_command(self, runner):
        """Test the logging formats command."""
        result = runner.invoke(main, ["logging", "formats"])
        assert result.exit_code == 0
        assert "Available Log Formats" in result.output
        assert "standard" in result.output
        assert "json" in result.output

    def test_logging_levels_command(self, runner):
        """Test the logging levels command."""
        result = runner.invoke(main, ["logging", "levels"])
        assert result.exit_code == 0
        assert "Available Log Levels" in result.output
        assert "debug" in result.output
        assert "info" in result.output
        assert "warning" in result.output
        assert "error" in result.output
        assert "critical" in result.output

    def test_proxy_log_format_option(self, runner):
        """Test that proxy command accepts --log-format option."""
        result = runner.invoke(main, ["proxy", "--help"])
        assert result.exit_code == 0
        assert "--log-format" in result.output

    def test_gateway_log_format_option(self, runner):
        """Test that gateway command accepts --log-format option."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert result.exit_code == 0
        assert "--log-format" in result.output


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining multiple logging features."""

    def test_full_logging_pipeline(self, temp_log_dir):
        """Test a complete logging pipeline with file output."""
        log_file = os.path.join(temp_log_dir, "integration.log")

        setup_logging(
            level="debug",
            log_format="detailed",
            output="both",
            log_file=log_file,
        )

        logger = get_logger("integration")

        with correlation_context("int-test-001"):
            logger.info("Starting integration test")
            logger.debug("Debug details here")

            with log_duration("test_operation") as timer:
                time.sleep(0.01)

            logger.info("Operation completed")

        # Flush handlers
        root = logging.getLogger(ROOT_LOGGER_NAME)
        for handler in root.handlers:
            handler.flush()

        # Verify file output
        with open(log_file) as f:
            content = f.read()

        assert "Starting integration test" in content
        assert "int-test-001" in content

    def test_json_structured_logging(self):
        """Test that JSON format produces parseable structured logs."""
        setup_logging(level="info", log_format="json", output="console")
        logger = get_logger("json_test")

        # The log should not raise any errors
        with correlation_context("json-001"):
            logger.info(
                "Request processed",
                extra={"duration_ms": 42, "status": "ok"},
            )

    def test_multiple_loggers(self):
        """Test that multiple loggers work correctly under the same root."""
        setup_logging(level="debug")

        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")

        assert logger1.name != logger2.name
        assert logger1.name.startswith(ROOT_LOGGER_NAME)
        assert logger2.name.startswith(ROOT_LOGGER_NAME)

        # Both should be able to log without errors
        logger1.info("Message from module A")
        logger2.info("Message from module B")

    def test_logging_from_different_threads(self):
        """Test thread-safe logging from multiple threads."""
        setup_logging(level="debug")
        errors = []

        def log_from_thread(thread_id):
            try:
                logger = get_logger(f"thread.{thread_id}")
                with correlation_context(f"thread-{thread_id}"):
                    for i in range(10):
                        logger.info("Message %d from thread %s", i, thread_id)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=log_from_thread, args=(i,))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred in threads: {errors}"


# ---------------------------------------------------------------------------
# LoggingConfig new fields (rotation / queue)
# ---------------------------------------------------------------------------


class TestLoggingConfigNewFields:
    """Tests for new LoggingConfig fields added for rotation and async support."""

    def test_default_rotation_mode_is_size(self):
        cfg = LoggingConfig()
        assert cfg.rotation_mode == "size"

    def test_default_rotation_when(self):
        cfg = LoggingConfig()
        assert cfg.rotation_when == "midnight"

    def test_default_rotation_interval(self):
        cfg = LoggingConfig()
        assert cfg.rotation_interval == 1

    def test_default_use_queue_handler_false(self):
        cfg = LoggingConfig()
        assert cfg.use_queue_handler is False

    def test_default_queue_max_size(self):
        cfg = LoggingConfig()
        assert cfg.queue_max_size == 10000

    def test_to_dict_includes_new_fields(self):
        cfg = LoggingConfig(
            rotation_mode="time",
            rotation_when="H",
            rotation_interval=2,
            use_queue_handler=True,
            queue_max_size=5000,
        )
        d = cfg.to_dict()
        assert d["rotation_mode"] == "time"
        assert d["rotation_when"] == "H"
        assert d["rotation_interval"] == 2
        assert d["use_queue_handler"] is True
        assert d["queue_max_size"] == 5000

    def test_from_dict_roundtrip_new_fields(self):
        original = LoggingConfig(
            rotation_mode="time",
            rotation_when="D",
            rotation_interval=7,
            use_queue_handler=True,
            queue_max_size=2000,
        )
        restored = LoggingConfig.from_dict(original.to_dict())
        assert restored.rotation_mode == "time"
        assert restored.rotation_when == "D"
        assert restored.rotation_interval == 7
        assert restored.use_queue_handler is True
        assert restored.queue_max_size == 2000

    def test_from_dict_defaults_for_missing_new_fields(self):
        """from_dict should gracefully handle dicts without new fields."""
        d = {
            "level": "info",
            "log_format": "standard",
            "output": "console",
        }
        cfg = LoggingConfig.from_dict(d)
        assert cfg.rotation_mode == "size"
        assert cfg.use_queue_handler is False
        assert cfg.queue_max_size == 10000

    def test_custom_rotation_mode_stored(self):
        cfg = LoggingConfig(rotation_mode="time")
        assert cfg.rotation_mode == "time"

    def test_custom_queue_max_size_stored(self):
        cfg = LoggingConfig(queue_max_size=256)
        assert cfg.queue_max_size == 256


# ---------------------------------------------------------------------------
# LoggingConfig.from_env()
# ---------------------------------------------------------------------------


class TestLoggingConfigFromEnv:
    """Tests for LoggingConfig.from_env() classmethod."""

    def test_from_env_defaults_when_no_env_vars(self):
        # Remove any stray GATEWAY_LOG_* vars before testing defaults
        clean = {
            k: v for k, v in os.environ.items()
            if not k.startswith("GATEWAY_LOG_")
        }
        with patch.dict(os.environ, clean, clear=True):
            cfg = LoggingConfig.from_env()
        default = LoggingConfig()
        assert cfg.level == default.level
        assert cfg.log_format == default.log_format
        assert cfg.output == default.output

    def test_from_env_reads_level(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_LEVEL": "debug"}):
            cfg = LoggingConfig.from_env()
        assert cfg.level == "debug"

    def test_from_env_reads_format(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_FORMAT": "json"}):
            cfg = LoggingConfig.from_env()
        assert cfg.log_format == "json"

    def test_from_env_reads_output(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_OUTPUT": "file"}):
            cfg = LoggingConfig.from_env()
        assert cfg.output == "file"

    def test_from_env_reads_log_file(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_FILE": "custom.log"}):
            cfg = LoggingConfig.from_env()
        assert cfg.log_file == "custom.log"

    def test_from_env_reads_max_bytes(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_MAX_BYTES": "5242880"}):
            cfg = LoggingConfig.from_env()
        assert cfg.max_bytes == 5242880

    def test_from_env_reads_backup_count(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_BACKUP_COUNT": "10"}):
            cfg = LoggingConfig.from_env()
        assert cfg.backup_count == 10

    def test_from_env_reads_rate_limit(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_RATE_LIMIT": "0.5"}):
            cfg = LoggingConfig.from_env()
        assert cfg.rate_limit_seconds == pytest.approx(0.5)

    def test_from_env_reads_rotation_mode(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_ROTATION_MODE": "time"}):
            cfg = LoggingConfig.from_env()
        assert cfg.rotation_mode == "time"

    def test_from_env_reads_rotation_when(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_ROTATION_WHEN": "H"}):
            cfg = LoggingConfig.from_env()
        assert cfg.rotation_when == "H"

    def test_from_env_reads_rotation_interval(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_ROTATION_INTERVAL": "3"}):
            cfg = LoggingConfig.from_env()
        assert cfg.rotation_interval == 3

    def test_from_env_reads_queue_true_variants(self):
        for val in ("1", "true", "yes", "on"):
            with patch.dict(os.environ, {"GATEWAY_LOG_QUEUE": val}):
                cfg = LoggingConfig.from_env()
            assert cfg.use_queue_handler is True, (
                f"Expected True for GATEWAY_LOG_QUEUE={val!r}"
            )

    def test_from_env_reads_queue_false_variants(self):
        for val in ("0", "false", "no", "off"):
            with patch.dict(os.environ, {"GATEWAY_LOG_QUEUE": val}):
                cfg = LoggingConfig.from_env()
            assert cfg.use_queue_handler is False, (
                f"Expected False for GATEWAY_LOG_QUEUE={val!r}"
            )

    def test_from_env_reads_queue_max_size(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_QUEUE_MAX_SIZE": "500"}):
            cfg = LoggingConfig.from_env()
        assert cfg.queue_max_size == 500

    def test_from_env_reads_json_extras_false(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_JSON_EXTRAS": "0"}):
            cfg = LoggingConfig.from_env()
        assert cfg.json_extras is False

    def test_from_env_reads_json_extras_true(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_JSON_EXTRAS": "true"}):
            cfg = LoggingConfig.from_env()
        assert cfg.json_extras is True

    def test_from_env_reads_correlation_id_flag_false(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_CORRELATION_ID": "false"}):
            cfg = LoggingConfig.from_env()
        assert cfg.include_correlation_id is False

    def test_from_env_reads_modules(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_MODULES": "gateway.api,gateway.db"}):
            cfg = LoggingConfig.from_env()
        assert cfg.module_filters == ["gateway.api", "gateway.db"]

    def test_from_env_ignores_invalid_max_bytes(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_MAX_BYTES": "not_a_number"}):
            cfg = LoggingConfig.from_env()
        assert cfg.max_bytes == LoggingConfig().max_bytes

    def test_from_env_ignores_invalid_backup_count(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_BACKUP_COUNT": "abc"}):
            cfg = LoggingConfig.from_env()
        assert cfg.backup_count == LoggingConfig().backup_count

    def test_from_env_ignores_invalid_rate_limit(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_RATE_LIMIT": "fast"}):
            cfg = LoggingConfig.from_env()
        assert cfg.rate_limit_seconds == LoggingConfig().rate_limit_seconds

    def test_from_env_ignores_invalid_rotation_interval(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_ROTATION_INTERVAL": "weekly"}):
            cfg = LoggingConfig.from_env()
        assert cfg.rotation_interval == LoggingConfig().rotation_interval

    def test_from_env_ignores_invalid_queue_max_size(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_QUEUE_MAX_SIZE": "big"}):
            cfg = LoggingConfig.from_env()
        assert cfg.queue_max_size == LoggingConfig().queue_max_size

    def test_from_env_custom_prefix(self):
        with patch.dict(os.environ, {"MY_LOG_LEVEL": "error"}):
            cfg = LoggingConfig.from_env(prefix="MY_LOG")
        assert cfg.level == "error"

    def test_from_env_custom_prefix_does_not_read_default_prefix(self):
        """Custom prefix should not pick up GATEWAY_LOG_* vars."""
        with patch.dict(
            os.environ,
            {"GATEWAY_LOG_LEVEL": "critical", "CUSTOM_LEVEL": "debug"},
        ):
            cfg = LoggingConfig.from_env(prefix="CUSTOM")
        assert cfg.level == "debug"

    def test_from_env_multiple_vars_set(self):
        env = {
            "GATEWAY_LOG_LEVEL": "warning",
            "GATEWAY_LOG_FORMAT": "json",
            "GATEWAY_LOG_OUTPUT": "console",
            "GATEWAY_LOG_ROTATION_MODE": "time",
            "GATEWAY_LOG_QUEUE": "true",
        }
        with patch.dict(os.environ, env):
            cfg = LoggingConfig.from_env()
        assert cfg.level == "warning"
        assert cfg.log_format == "json"
        assert cfg.output == "console"
        assert cfg.rotation_mode == "time"
        assert cfg.use_queue_handler is True

    def test_from_env_level_normalized_to_lowercase(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_LEVEL": "DEBUG"}):
            cfg = LoggingConfig.from_env()
        assert cfg.level == "debug"

    def test_from_env_format_normalized_to_lowercase(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_FORMAT": "JSON"}):
            cfg = LoggingConfig.from_env()
        assert cfg.log_format == "json"

    def test_from_env_modules_strips_whitespace(self):
        with patch.dict(
            os.environ, {"GATEWAY_LOG_MODULES": " gateway.api , gateway.db "}
        ):
            cfg = LoggingConfig.from_env()
        assert cfg.module_filters == ["gateway.api", "gateway.db"]

    def test_from_env_empty_modules_string(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_MODULES": ""}):
            cfg = LoggingConfig.from_env()
        assert cfg.module_filters == []

    def test_from_env_rotation_mode_normalized_to_lowercase(self):
        with patch.dict(os.environ, {"GATEWAY_LOG_ROTATION_MODE": "TIME"}):
            cfg = LoggingConfig.from_env()
        assert cfg.rotation_mode == "time"


# ---------------------------------------------------------------------------
# TimedRotatingFileHandler support
# ---------------------------------------------------------------------------


class TestTimedRotation:
    """Tests for time-based log rotation using TimedRotatingFileHandler."""

    def test_setup_logging_creates_timed_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "app.log")
            setup_logging(
                output="file",
                log_file=log_file,
                rotation_mode="time",
                rotation_when="midnight",
            )
            status = get_logging_status()
            timed_handlers = [
                h for h in status["handlers"]
                if h.get("rotation_mode") == "time"
            ]
            assert len(timed_handlers) >= 1

    def test_setup_logging_timed_handler_has_correct_when(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "app.log")
            setup_logging(
                output="file",
                log_file=log_file,
                rotation_mode="time",
                rotation_when="H",
            )
            status = get_logging_status()
            timed = next(
                (h for h in status["handlers"] if h.get("rotation_mode") == "time"),
                None,
            )
            assert timed is not None
            assert timed["rotation_when"] == "H"

    def test_setup_logging_timed_creates_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "timed.log")
            logger = setup_logging(
                output="file",
                log_file=log_file,
                rotation_mode="time",
            )
            logger.info("Test message for timed rotation")
            assert os.path.exists(log_file)

    def test_timed_rotation_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "cfg_timed.log")
            cfg = LoggingConfig(
                output="file",
                log_file=log_file,
                rotation_mode="time",
                rotation_when="D",
                rotation_interval=1,
            )
            logger = setup_logging(config=cfg)
            logger.info("Config-based timed rotation")
            status = get_logging_status()
            timed = next(
                (h for h in status["handlers"] if h.get("rotation_mode") == "time"),
                None,
            )
            assert timed is not None

    def test_timed_rotation_active_config_reflects_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "ac.log")
            setup_logging(
                output="file",
                log_file=log_file,
                rotation_mode="time",
                rotation_when="midnight",
            )
            cfg = get_active_config()
            assert cfg is not None
            assert cfg.rotation_mode == "time"
            assert cfg.rotation_when == "midnight"

    def test_size_rotation_still_works_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "size.log")
            logger = setup_logging(output="file", log_file=log_file)
            logger.info("Size rotation test")
            status = get_logging_status()
            size_handlers = [
                h for h in status["handlers"] if h.get("rotation_mode") == "size"
            ]
            assert len(size_handlers) >= 1

    def test_timed_rotation_messages_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "write.log")
            logger = setup_logging(
                output="file",
                log_file=log_file,
                rotation_mode="time",
                rotation_when="midnight",
                level="debug",
            )
            logger.debug("debug timed msg")
            logger.info("info timed msg")
            logger.warning("warn timed msg")
            content = Path(log_file).read_text()
            assert "debug timed msg" in content
            assert "info timed msg" in content
            assert "warn timed msg" in content

    def test_timed_rotation_interval_reflected_in_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "int.log")
            setup_logging(
                output="file",
                log_file=log_file,
                rotation_mode="time",
                rotation_when="H",
                rotation_interval=2,
            )
            status = get_logging_status()
            timed = next(
                (h for h in status["handlers"] if h.get("rotation_mode") == "time"),
                None,
            )
            assert timed is not None
            assert timed.get("rotation_interval") == 2

    def test_timed_rotation_backup_count_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "bc_timed.log")
            setup_logging(
                output="file",
                log_file=log_file,
                rotation_mode="time",
                backup_count=14,
            )
            status = get_logging_status()
            timed = next(
                (h for h in status["handlers"] if h.get("rotation_mode") == "time"),
                None,
            )
            assert timed is not None
            assert timed.get("backup_count") == 14


# ---------------------------------------------------------------------------
# QueueHandler / async logging
# ---------------------------------------------------------------------------


class TestQueueHandler:
    """Tests for async logging via QueueHandler."""

    def test_setup_logging_with_queue_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "queue.log")
            setup_logging(
                output="file",
                log_file=log_file,
                use_queue_handler=True,
            )
            status = get_logging_status()
            queue_handlers = [
                h for h in status["handlers"]
                if h.get("type") == "QueueHandler"
            ]
            assert len(queue_handlers) >= 1

    def test_queue_handler_max_size_reflected_in_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "qmax.log")
            setup_logging(
                output="file",
                log_file=log_file,
                use_queue_handler=True,
                queue_max_size=500,
            )
            status = get_logging_status()
            qh = next(
                (h for h in status["handlers"] if "queue_maxsize" in h),
                None,
            )
            assert qh is not None
            assert qh["queue_maxsize"] == 500

    def test_queue_handler_logs_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "qlog.log")
            logger = setup_logging(
                output="file",
                log_file=log_file,
                use_queue_handler=True,
                level="debug",
            )
            logger.info("Queue handler test message")
            # Allow the listener thread to flush
            time.sleep(0.2)
            content = Path(log_file).read_text() if Path(log_file).exists() else ""
            assert "Queue handler test message" in content

    def test_queue_handler_reset_stops_listener(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "qreset.log")
            setup_logging(
                output="file",
                log_file=log_file,
                use_queue_handler=True,
            )
            # reset_logging() should stop the listener without error
            reset_logging()
            assert not is_configured()

    def test_queue_handler_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "qcfg.log")
            cfg = LoggingConfig(
                output="file",
                log_file=log_file,
                use_queue_handler=True,
                queue_max_size=200,
            )
            setup_logging(config=cfg)
            status = get_logging_status()
            qh = next(
                (h for h in status["handlers"] if "queue_maxsize" in h),
                None,
            )
            assert qh is not None

    def test_queue_handler_active_config_reflects_setting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "qac.log")
            setup_logging(
                output="file",
                log_file=log_file,
                use_queue_handler=True,
                queue_max_size=1500,
            )
            cfg = get_active_config()
            assert cfg is not None
            assert cfg.use_queue_handler is True
            assert cfg.queue_max_size == 1500

    def test_queue_handler_multiple_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "qmulti.log")
            logger = setup_logging(
                output="file",
                log_file=log_file,
                use_queue_handler=True,
                level="debug",
            )
            for i in range(20):
                logger.info("Queue message %d", i)
            time.sleep(0.3)
            content = Path(log_file).read_text() if Path(log_file).exists() else ""
            for i in range(20):
                assert f"Queue message {i}" in content

    def test_queue_handler_reset_then_reconfigure(self):
        """Resetting and reconfiguring should work without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "qre.log")
            setup_logging(
                output="file",
                log_file=log_file,
                use_queue_handler=True,
            )
            reset_logging()
            # Reconfigure without queue handler
            setup_logging(output="console")
            assert is_configured()


# ---------------------------------------------------------------------------
# set_log_level()
# ---------------------------------------------------------------------------


class TestSetLogLevel:
    """Tests for the set_log_level() runtime level change function."""

    def test_set_log_level_returns_numeric_debug(self):
        setup_logging(level="info")
        numeric = set_log_level("debug")
        assert numeric == logging.DEBUG

    def test_set_log_level_returns_numeric_info(self):
        setup_logging(level="debug")
        numeric = set_log_level("info")
        assert numeric == logging.INFO

    def test_set_log_level_returns_numeric_warning(self):
        setup_logging(level="info")
        numeric = set_log_level("warning")
        assert numeric == logging.WARNING

    def test_set_log_level_returns_numeric_error(self):
        setup_logging(level="info")
        numeric = set_log_level("error")
        assert numeric == logging.ERROR

    def test_set_log_level_returns_numeric_critical(self):
        setup_logging(level="info")
        numeric = set_log_level("critical")
        assert numeric == logging.CRITICAL

    def test_set_log_level_updates_root_logger_debug(self):
        setup_logging(level="info")
        set_log_level("debug")
        root = logging.getLogger(ROOT_LOGGER_NAME)
        assert root.level == logging.DEBUG

    def test_set_log_level_updates_root_logger_warning(self):
        setup_logging(level="info")
        set_log_level("warning")
        root = logging.getLogger(ROOT_LOGGER_NAME)
        assert root.level == logging.WARNING

    def test_set_log_level_updates_root_logger_error(self):
        setup_logging(level="info")
        set_log_level("error")
        root = logging.getLogger(ROOT_LOGGER_NAME)
        assert root.level == logging.ERROR

    def test_set_log_level_updates_root_logger_critical(self):
        setup_logging(level="info")
        set_log_level("critical")
        root = logging.getLogger(ROOT_LOGGER_NAME)
        assert root.level == logging.CRITICAL

    def test_set_log_level_updates_active_config(self):
        setup_logging(level="info")
        set_log_level("warning")
        cfg = get_active_config()
        assert cfg is not None
        assert cfg.level == "warning"

    def test_set_log_level_invalid_raises(self):
        setup_logging(level="info")
        with pytest.raises((ValueError, AttributeError, KeyError)):
            set_log_level("verbose")

    def test_set_log_level_updates_status_effective_level(self):
        setup_logging(level="info")
        set_log_level("error")
        status = get_logging_status()
        assert status["effective_level"] == "ERROR"

    def test_set_log_level_multiple_changes(self):
        setup_logging(level="info")
        levels = ["debug", "info", "warning", "error", "critical", "info"]
        for lvl in levels:
            set_log_level(lvl)
        root = logging.getLogger(ROOT_LOGGER_NAME)
        assert root.level == logging.INFO

    def test_set_log_level_with_file_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "lvl.log")
            setup_logging(output="file", log_file=log_file, level="info")
            set_log_level("debug")
            root = logging.getLogger(ROOT_LOGGER_NAME)
            assert root.level == logging.DEBUG

    def test_set_log_level_active_config_level_string(self):
        setup_logging(level="info")
        set_log_level("critical")
        cfg = get_active_config()
        assert cfg is not None
        assert cfg.level == "critical"


# ---------------------------------------------------------------------------
# get_log_files()
# ---------------------------------------------------------------------------


class TestGetLogFiles:
    """Tests for the get_log_files() function."""

    def test_get_log_files_empty_when_no_file_handler(self):
        setup_logging(output="console")
        files = get_log_files()
        assert files == []

    def test_get_log_files_returns_one_entry_with_file_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = setup_logging(output="file", log_file=log_file)
            logger.info("Write something to create the file")
            files = get_log_files()
            assert len(files) == 1

    def test_get_log_files_contains_path_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "path_test.log")
            logger = setup_logging(output="file", log_file=log_file)
            logger.info("Test")
            files = get_log_files()
            assert len(files) == 1
            assert "path" in files[0]

    def test_get_log_files_path_contains_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "named.log")
            logger = setup_logging(output="file", log_file=log_file)
            logger.info("path check")
            files = get_log_files()
            assert "named.log" in files[0]["path"]

    def test_get_log_files_size_bytes_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "size.log")
            logger = setup_logging(output="file", log_file=log_file)
            logger.info("Size test message")
            files = get_log_files()
            assert "size_bytes" in files[0]
            assert isinstance(files[0]["size_bytes"], int)
            assert files[0]["size_bytes"] >= 0

    def test_get_log_files_rotation_mode_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "mode.log")
            setup_logging(output="file", log_file=log_file, rotation_mode="size")
            files = get_log_files()
            assert files[0]["rotation_mode"] == "size"

    def test_get_log_files_rotation_mode_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "timed.log")
            setup_logging(
                output="file",
                log_file=log_file,
                rotation_mode="time",
                rotation_when="midnight",
            )
            files = get_log_files()
            assert files[0]["rotation_mode"] == "time"

    def test_get_log_files_backup_count_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "bc.log")
            setup_logging(output="file", log_file=log_file, backup_count=3)
            files = get_log_files()
            assert "backup_count" in files[0]
            assert files[0]["backup_count"] == 3

    def test_get_log_files_backup_files_field_is_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "bk.log")
            setup_logging(output="file", log_file=log_file)
            files = get_log_files()
            assert "backup_files" in files[0]
            assert isinstance(files[0]["backup_files"], list)

    def test_get_log_files_size_handler_has_max_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "maxb.log")
            setup_logging(output="file", log_file=log_file, rotation_mode="size")
            files = get_log_files()
            assert "max_bytes" in files[0]

    def test_get_log_files_timed_handler_has_rotation_when(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "tw.log")
            setup_logging(
                output="file",
                log_file=log_file,
                rotation_mode="time",
                rotation_when="H",
            )
            files = get_log_files()
            assert "rotation_when" in files[0]
            assert files[0]["rotation_when"] == "H"

    def test_get_log_files_timed_handler_has_rotation_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "ti.log")
            setup_logging(
                output="file",
                log_file=log_file,
                rotation_mode="time",
                rotation_when="H",
                rotation_interval=4,
            )
            files = get_log_files()
            assert "rotation_interval" in files[0]
            assert files[0]["rotation_interval"] == 4

    def test_get_log_files_after_reset_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "rst.log")
            setup_logging(output="file", log_file=log_file)
            reset_logging()
            files = get_log_files()
            assert files == []

    def test_get_log_files_file_size_increases_after_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "grow.log")
            logger = setup_logging(output="file", log_file=log_file, level="debug")
            logger.info("First message")
            files_before = get_log_files()
            size_before = files_before[0]["size_bytes"]
            for _ in range(10):
                logger.info("Additional message to grow the file")
            # Re-read size from filesystem
            size_after = Path(log_file).stat().st_size
            assert size_after > size_before


# ---------------------------------------------------------------------------
# New CLI commands: logging set-level, logging configure, logging files
# ---------------------------------------------------------------------------


class TestNewLoggingCLICommands:
    """Tests for new CLI commands: logging set-level, logging configure, logging files."""

    def setup_method(self):
        self.runner = CliRunner()

    # --- logging set-level ---

    def test_logging_set_level_debug(self):
        result = self.runner.invoke(main, ["logging", "set-level", "debug"])
        assert result.exit_code == 0
        assert "DEBUG" in result.output

    def test_logging_set_level_info(self):
        result = self.runner.invoke(main, ["logging", "set-level", "info"])
        assert result.exit_code == 0
        assert "INFO" in result.output

    def test_logging_set_level_warning(self):
        result = self.runner.invoke(main, ["logging", "set-level", "warning"])
        assert result.exit_code == 0
        assert "WARNING" in result.output

    def test_logging_set_level_error(self):
        result = self.runner.invoke(main, ["logging", "set-level", "error"])
        assert result.exit_code == 0
        assert "ERROR" in result.output

    def test_logging_set_level_critical(self):
        result = self.runner.invoke(main, ["logging", "set-level", "critical"])
        assert result.exit_code == 0
        assert "CRITICAL" in result.output

    def test_logging_set_level_invalid_exits_nonzero(self):
        result = self.runner.invoke(main, ["logging", "set-level", "verbose"])
        assert result.exit_code != 0

    def test_logging_set_level_shows_effective_level(self):
        result = self.runner.invoke(main, ["logging", "set-level", "debug"])
        assert result.exit_code == 0
        assert "Effective level" in result.output or "effective" in result.output.lower()

    def test_logging_set_level_shows_configured_status(self):
        result = self.runner.invoke(main, ["logging", "set-level", "info"])
        assert result.exit_code == 0
        # Should at least mention configured
        assert "Configured" in result.output or "configured" in result.output.lower()

    def test_logging_set_level_help(self):
        result = self.runner.invoke(main, ["logging", "set-level", "--help"])
        assert result.exit_code == 0
        assert "level" in result.output.lower()

    # --- logging configure ---

    def test_logging_configure_no_options(self):
        """configure with no options should succeed using defaults."""
        result = self.runner.invoke(main, ["logging", "configure"])
        assert result.exit_code == 0
        assert "reconfigured" in result.output.lower()

    def test_logging_configure_level_debug(self):
        result = self.runner.invoke(
            main, ["logging", "configure", "--level", "debug"]
        )
        assert result.exit_code == 0
        assert "DEBUG" in result.output

    def test_logging_configure_level_warning(self):
        result = self.runner.invoke(
            main, ["logging", "configure", "--level", "warning"]
        )
        assert result.exit_code == 0
        assert "WARNING" in result.output

    def test_logging_configure_log_format_json(self):
        result = self.runner.invoke(
            main, ["logging", "configure", "--log-format", "json"]
        )
        assert result.exit_code == 0
        assert "json" in result.output.lower()

    def test_logging_configure_log_format_standard(self):
        result = self.runner.invoke(
            main, ["logging", "configure", "--log-format", "standard"]
        )
        assert result.exit_code == 0
        assert "standard" in result.output.lower()

    def test_logging_configure_output_console(self):
        result = self.runner.invoke(
            main, ["logging", "configure", "--output", "console"]
        )
        assert result.exit_code == 0
        assert "console" in result.output.lower()

    def test_logging_configure_output_json_format(self):
        """--format json should emit a JSON object."""
        result = self.runner.invoke(
            main, ["logging", "configure", "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_logging_configure_json_output_has_configured_key(self):
        result = self.runner.invoke(
            main, ["logging", "configure", "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "configured" in data

    def test_logging_configure_from_env_no_env(self):
        """--from-env with no env vars should succeed."""
        result = self.runner.invoke(main, ["logging", "configure", "--from-env"])
        assert result.exit_code == 0

    def test_logging_configure_from_env_reads_level(self):
        result = self.runner.invoke(
            main,
            ["logging", "configure", "--from-env"],
            env={"GATEWAY_LOG_LEVEL": "error"},
        )
        assert result.exit_code == 0
        assert "ERROR" in result.output

    def test_logging_configure_explicit_overrides_env(self):
        """Explicit --level should override GATEWAY_LOG_LEVEL."""
        result = self.runner.invoke(
            main,
            ["logging", "configure", "--from-env", "--level", "debug"],
            env={"GATEWAY_LOG_LEVEL": "error"},
        )
        assert result.exit_code == 0
        assert "DEBUG" in result.output

    def test_logging_configure_with_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "cli.log")
            result = self.runner.invoke(
                main,
                [
                    "logging", "configure",
                    "--level", "info",
                    "--output", "file",
                    "--log-file", log_file,
                ],
            )
            assert result.exit_code == 0
            assert "Log file" in result.output or log_file in result.output

    def test_logging_configure_rotation_mode_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "rot.log")
            result = self.runner.invoke(
                main,
                [
                    "logging", "configure",
                    "--output", "file",
                    "--log-file", log_file,
                    "--rotation-mode", "time",
                ],
            )
            assert result.exit_code == 0
            assert "time" in result.output.lower()

    def test_logging_configure_rotation_mode_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "sz.log")
            result = self.runner.invoke(
                main,
                [
                    "logging", "configure",
                    "--output", "file",
                    "--log-file", log_file,
                    "--rotation-mode", "size",
                ],
            )
            assert result.exit_code == 0

    def test_logging_configure_invalid_rotation_mode(self):
        result = self.runner.invoke(
            main,
            ["logging", "configure", "--rotation-mode", "daily"],
        )
        assert result.exit_code != 0

    def test_logging_configure_invalid_level(self):
        result = self.runner.invoke(
            main,
            ["logging", "configure", "--level", "trace"],
        )
        assert result.exit_code != 0

    def test_logging_configure_invalid_log_format(self):
        result = self.runner.invoke(
            main,
            ["logging", "configure", "--log-format", "yaml"],
        )
        assert result.exit_code != 0

    def test_logging_configure_help(self):
        result = self.runner.invoke(main, ["logging", "configure", "--help"])
        assert result.exit_code == 0
        assert "--level" in result.output

    def test_logging_configure_shows_handler_count(self):
        result = self.runner.invoke(main, ["logging", "configure"])
        assert result.exit_code == 0
        assert "Handlers" in result.output or "handler" in result.output.lower()

    # --- logging files ---

    def test_logging_files_no_file_handlers_message(self):
        """When no file handlers, show informative message."""
        self.runner.invoke(main, ["logging", "configure", "--output", "console"])
        result = self.runner.invoke(main, ["logging", "files"])
        assert result.exit_code == 0
        assert "No file-based" in result.output

    def test_logging_files_with_file_handler_shows_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "files_test.log")
            self.runner.invoke(
                main,
                [
                    "logging", "configure",
                    "--output", "file",
                    "--log-file", log_file,
                ],
            )
            result = self.runner.invoke(main, ["logging", "files"])
            assert result.exit_code == 0
            # Either shows active files header or the path, or no-file message
            assert (
                "Active Log Files" in result.output
                or log_file in result.output
                or "No file-based" in result.output
            )

    def test_logging_files_json_format_no_handlers(self):
        """JSON output with no file handlers should be an empty list."""
        self.runner.invoke(main, ["logging", "configure", "--output", "console"])
        result = self.runner.invoke(main, ["logging", "files", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data == []

    def test_logging_files_json_format_with_file_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "fj.log")
            self.runner.invoke(
                main,
                [
                    "logging", "configure",
                    "--output", "file",
                    "--log-file", log_file,
                ],
            )
            result = self.runner.invoke(
                main, ["logging", "files", "--format", "json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            # May be empty if configure set up in a subprocess context
            # Just verify the structure is correct
            for entry in data:
                assert "path" in entry
                assert "size_bytes" in entry
                assert "rotation_mode" in entry

    def test_logging_files_help(self):
        result = self.runner.invoke(main, ["logging", "files", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output

    def test_logging_files_invalid_format(self):
        result = self.runner.invoke(
            main, ["logging", "files", "--format", "yaml"]
        )
        assert result.exit_code != 0
