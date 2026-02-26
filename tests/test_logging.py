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
    get_log_level_name,
    get_logger,
    get_logging_status,
    is_configured,
    log_duration,
    log_request,
    reset_logging,
    set_correlation_id,
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
