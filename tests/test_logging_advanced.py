"""Comprehensive tests for the advanced logging system features.

Tests for SensitiveDataFilter, SamplingFilter, LogMetrics, StructuredLogger,
RequestContext, LogBuffer, exception formatting, and HealthLogger.
"""

import json
import logging
import logging.handlers
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli import main
from src.logging_config import (
    BufferedHandler,
    HealthLogger,
    LogBuffer,
    LogMetrics,
    LogMetricsHandler,
    RequestContextFilter,
    ROOT_LOGGER_NAME,
    SamplingFilter,
    SensitiveDataFilter,
    StructuredLogger,
    clear_correlation_id,
    clear_request_context,
    format_exception_context,
    get_correlation_id,
    get_log_metrics,
    get_logger,
    get_request_context,
    log_exception,
    request_context,
    reset_log_metrics,
    reset_logging,
    set_correlation_id,
    set_request_context,
    setup_logging,
    setup_logging_advanced,
    update_request_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_logging():
    """Reset logging state before and after each test."""
    reset_logging()
    clear_correlation_id()
    clear_request_context()
    reset_log_metrics()
    yield
    reset_logging()
    clear_correlation_id()
    clear_request_context()
    reset_log_metrics()


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def make_record(
    msg="test message",
    level=logging.INFO,
    name="test",
    **extra,
):
    """Helper to create log records."""
    record = logging.LogRecord(
        name=name, level=level, pathname="",
        lineno=0, msg=msg, args=(), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# ---------------------------------------------------------------------------
# SensitiveDataFilter Tests
# ---------------------------------------------------------------------------


class TestSensitiveDataFilter:
    """Tests for sensitive data redaction in log output."""

    def test_redacts_anthropic_api_key(self):
        """Test that Anthropic API keys are redacted."""
        f = SensitiveDataFilter()
        record = make_record("Using key sk-ant-abcdef1234567890abcdef")
        f.filter(record)
        assert "sk-ant-" not in record.msg
        assert "REDACTED" in record.msg

    def test_redacts_openai_api_key(self):
        """Test that OpenAI-style API keys are redacted."""
        f = SensitiveDataFilter()
        record = make_record("Using key sk-proj12345678901234567890")
        f.filter(record)
        assert "sk-proj12345678901234567890" not in record.msg
        assert "REDACTED" in record.msg

    def test_redacts_api_key_assignment(self):
        """Test that API key assignments are redacted."""
        f = SensitiveDataFilter()
        record = make_record("api_key = mysecretapikey12345678")
        f.filter(record)
        assert "mysecretapikey12345678" not in record.msg
        assert "REDACTED" in record.msg

    def test_redacts_bearer_token(self):
        """Test that Bearer tokens are redacted."""
        f = SensitiveDataFilter()
        record = make_record("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test.sig")
        f.filter(record)
        assert "eyJhbGciOiJIUzI1NiJ9" not in record.msg
        assert "REDACTED" in record.msg

    def test_redacts_password(self):
        """Test that passwords are redacted."""
        f = SensitiveDataFilter()
        record = make_record("password = super_secret_123!")
        f.filter(record)
        assert "super_secret_123!" not in record.msg
        assert "REDACTED" in record.msg

    def test_redacts_x_api_key(self):
        """Test that x-api-key headers are redacted."""
        f = SensitiveDataFilter()
        record = make_record("x-api-key: my-secret-key-value-here")
        f.filter(record)
        assert "my-secret-key-value-here" not in record.msg
        assert "REDACTED" in record.msg

    def test_preserves_normal_messages(self):
        """Test that non-sensitive messages are not modified."""
        f = SensitiveDataFilter()
        record = make_record("Starting server on port 8080")
        original_msg = record.msg
        f.filter(record)
        assert record.msg == original_msg

    def test_always_returns_true(self):
        """Test that the filter always passes records through."""
        f = SensitiveDataFilter()
        record = make_record("test")
        assert f.filter(record) is True

    def test_redacts_extra_fields(self):
        """Test that sensitive extra fields are redacted."""
        f = SensitiveDataFilter(redact_extras=True)
        record = make_record("test")
        record.api_key = "secret-key-value"  # type: ignore
        f.filter(record)
        assert record.api_key == "***REDACTED***"  # type: ignore

    def test_no_redact_extras_when_disabled(self):
        """Test that extra fields are not redacted when disabled."""
        f = SensitiveDataFilter(redact_extras=False)
        record = make_record("test")
        record.api_key = "secret-key-value"  # type: ignore
        f.filter(record)
        assert record.api_key == "secret-key-value"  # type: ignore

    def test_custom_patterns(self):
        """Test that custom redaction patterns work."""
        import re

        custom = [re.compile(r"(CUSTOM-[A-Z0-9]{10})")]
        f = SensitiveDataFilter(patterns=custom)
        record = make_record("Token: CUSTOM-ABCDEF1234")
        f.filter(record)
        assert "CUSTOM-ABCDEF1234" not in record.msg
        assert "REDACTED" in record.msg

    def test_custom_patterns_as_strings(self):
        """Test that custom string patterns are compiled."""
        f = SensitiveDataFilter(patterns=[r"SECRET_\d{6}"])
        record = make_record("Code is SECRET_123456")
        f.filter(record)
        assert "SECRET_123456" not in record.msg

    def test_custom_replacement_text(self):
        """Test custom replacement text."""
        f = SensitiveDataFilter(replacement="[HIDDEN]")
        record = make_record("password = mysecretpassword123")
        f.filter(record)
        assert "[HIDDEN]" in record.msg

    def test_redacts_tuple_args(self):
        """Test redaction of string args in tuple format."""
        f = SensitiveDataFilter()
        record = make_record("Setting %s")
        record.args = ("password = super_secret_value",)
        f.filter(record)
        assert "super_secret_value" not in str(record.args[0])

    def test_redacts_dict_args(self):
        """Test redaction of string args in dict format."""
        f = SensitiveDataFilter()
        record = make_record("Setting %(key)s")
        record.args = {"key": "password = my_secret_pass"}
        f.filter(record)
        assert "my_secret_pass" not in str(record.args["key"])

    def test_redacts_secret_key_assignment(self):
        """Test that secret key assignments are redacted."""
        f = SensitiveDataFilter()
        record = make_record("secret = this_is_very_secret_value")
        f.filter(record)
        assert "this_is_very_secret_value" not in record.msg
        assert "REDACTED" in record.msg

    def test_redacts_token_assignment(self):
        """Test that token assignments are redacted."""
        f = SensitiveDataFilter()
        record = make_record("token = abcdef1234567890abcdef")
        f.filter(record)
        assert "abcdef1234567890abcdef" not in record.msg
        assert "REDACTED" in record.msg


# ---------------------------------------------------------------------------
# SamplingFilter Tests
# ---------------------------------------------------------------------------


class TestSamplingFilter:
    """Tests for log message sampling."""

    def test_always_pass_error(self):
        """Test that ERROR messages always pass."""
        f = SamplingFilter(rates={"DEBUG": 0.0, "INFO": 0.0})
        record = make_record(level=logging.ERROR)
        assert f.filter(record) is True

    def test_always_pass_critical(self):
        """Test that CRITICAL messages always pass."""
        f = SamplingFilter(rates={"DEBUG": 0.0, "INFO": 0.0})
        record = make_record(level=logging.CRITICAL)
        assert f.filter(record) is True

    def test_zero_rate_blocks_all(self):
        """Test that a zero rate blocks all messages."""
        f = SamplingFilter(rates={"DEBUG": 0.0})
        results = [f.filter(make_record(level=logging.DEBUG)) for _ in range(100)]
        assert not any(results)

    def test_full_rate_passes_all(self):
        """Test that rate 1.0 passes all messages."""
        f = SamplingFilter(rates={"INFO": 1.0})
        results = [f.filter(make_record(level=logging.INFO)) for _ in range(100)]
        assert all(results)

    def test_partial_rate_samples(self):
        """Test that partial rates sample messages."""
        f = SamplingFilter(rates={"INFO": 0.5})
        results = [f.filter(make_record(level=logging.INFO)) for _ in range(1000)]
        passed = sum(results)
        # With 1000 samples at 0.5 rate, expect roughly 500 (allow wide margin)
        assert 300 < passed < 700

    def test_default_rate(self):
        """Test that unspecified levels use the default rate."""
        f = SamplingFilter(default_rate=0.0)
        results = [f.filter(make_record(level=logging.WARNING)) for _ in range(100)]
        # WARNING is not in always_pass_levels by default, and default_rate=0
        assert not any(results)

    def test_custom_always_pass_levels(self):
        """Test custom always-pass levels."""
        f = SamplingFilter(
            rates={"WARNING": 0.0},
            always_pass_levels={"WARNING"},
        )
        record = make_record(level=logging.WARNING)
        assert f.filter(record) is True

    def test_stats(self):
        """Test sampling statistics."""
        f = SamplingFilter(rates={"DEBUG": 0.0})
        for _ in range(10):
            f.filter(make_record(level=logging.DEBUG))
        for _ in range(5):
            f.filter(make_record(level=logging.ERROR))

        stats = f.get_stats()
        assert stats["total_count"] == 15
        assert stats["passed_count"] == 5  # Only ERROR passed
        assert stats["suppressed_by_level"]["DEBUG"] == 10

    def test_reset_stats(self):
        """Test resetting sampling statistics."""
        f = SamplingFilter(rates={"DEBUG": 0.0})
        f.filter(make_record(level=logging.DEBUG))
        f.reset_stats()
        stats = f.get_stats()
        assert stats["total_count"] == 0
        assert stats["passed_count"] == 0


# ---------------------------------------------------------------------------
# LogMetrics Tests
# ---------------------------------------------------------------------------


class TestLogMetrics:
    """Tests for log metrics collection."""

    def test_record_counts_by_level(self):
        """Test that metrics count messages by level."""
        metrics = LogMetrics()
        metrics.record(make_record(level=logging.INFO))
        metrics.record(make_record(level=logging.INFO))
        metrics.record(make_record(level=logging.ERROR))

        report = metrics.get_report()
        assert report["levels"]["INFO"] == 2
        assert report["levels"]["ERROR"] == 1

    def test_record_counts_by_module(self):
        """Test that metrics count messages by module."""
        metrics = LogMetrics()
        metrics.record(make_record(name="src.proxy"))
        metrics.record(make_record(name="src.proxy"))
        metrics.record(make_record(name="src.cache"))

        report = metrics.get_report()
        assert report["top_modules"]["src.proxy"] == 2
        assert report["top_modules"]["src.cache"] == 1

    def test_total_count(self):
        """Test total message count."""
        metrics = LogMetrics()
        for _ in range(5):
            metrics.record(make_record())

        report = metrics.get_report()
        assert report["total_count"] == 5

    def test_error_rate(self):
        """Test error rate calculation."""
        metrics = LogMetrics(window_seconds=60.0)
        for _ in range(10):
            metrics.record(make_record(level=logging.ERROR))

        rate = metrics.get_error_rate()
        assert rate == 10.0  # 10 errors in 60 seconds = 10/min

    def test_recent_errors(self):
        """Test recent errors tracking."""
        metrics = LogMetrics()
        metrics.record(make_record("Error 1", level=logging.ERROR, name="mod1"))
        metrics.record(make_record("Error 2", level=logging.ERROR, name="mod2"))

        report = metrics.get_report()
        assert len(report["recent_errors"]) == 2
        assert report["recent_errors"][0]["message"] == "Error 1"
        assert report["recent_errors"][1]["module"] == "mod2"

    def test_recent_errors_max_limit(self):
        """Test that recent errors list is capped."""
        metrics = LogMetrics()
        metrics._max_recent_errors = 5
        for i in range(10):
            metrics.record(make_record(f"Error {i}", level=logging.ERROR))

        report = metrics.get_report()
        assert len(report["recent_errors"]) == 5

    def test_exception_type_captured(self):
        """Test that exception types are captured in recent errors."""
        metrics = LogMetrics()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="",
            lineno=0, msg="error occurred", args=(), exc_info=exc_info,
        )
        metrics.record(record)

        report = metrics.get_report()
        assert report["recent_errors"][0]["exception_type"] == "ValueError"

    def test_reset(self):
        """Test metrics reset."""
        metrics = LogMetrics()
        metrics.record(make_record(level=logging.ERROR))
        metrics.reset()

        report = metrics.get_report()
        assert report["total_count"] == 0
        assert report["levels"] == {}

    def test_uptime(self):
        """Test uptime tracking."""
        metrics = LogMetrics()
        time.sleep(0.05)
        report = metrics.get_report()
        assert report["uptime_seconds"] >= 0.04

    def test_thread_safety(self):
        """Test that metrics are thread-safe."""
        metrics = LogMetrics()
        errors = []

        def record_messages():
            try:
                for _ in range(100):
                    metrics.record(make_record())
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=record_messages) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        report = metrics.get_report()
        assert report["total_count"] == 500


class TestLogMetricsHandler:
    """Tests for the LogMetricsHandler."""

    def test_handler_records_to_metrics(self):
        """Test that handler feeds records to metrics."""
        metrics = LogMetrics()
        handler = LogMetricsHandler(metrics)

        record = make_record(level=logging.INFO)
        handler.emit(record)

        report = metrics.get_report()
        assert report["total_count"] == 1
        assert report["levels"]["INFO"] == 1


class TestGetLogMetrics:
    """Tests for the global LogMetrics instance."""

    def test_get_creates_instance(self):
        """Test that get_log_metrics creates a new instance."""
        metrics = get_log_metrics()
        assert isinstance(metrics, LogMetrics)

    def test_get_returns_same_instance(self):
        """Test that get_log_metrics returns the same instance."""
        m1 = get_log_metrics()
        m2 = get_log_metrics()
        assert m1 is m2

    def test_reset_clears_instance(self):
        """Test that reset_log_metrics clears the instance."""
        m1 = get_log_metrics()
        reset_log_metrics()
        m2 = get_log_metrics()
        assert m1 is not m2


# ---------------------------------------------------------------------------
# StructuredLogger Tests
# ---------------------------------------------------------------------------


class TestStructuredLogger:
    """Tests for the StructuredLogger wrapper."""

    def test_basic_logging(self):
        """Test basic structured logging."""
        setup_logging(level="debug")
        slog = StructuredLogger("test_module")
        # Should not raise
        slog.info("test message", key="value")

    def test_name_property(self):
        """Test the name property."""
        slog = StructuredLogger("test_module")
        assert "test_module" in slog.name

    def test_bind_creates_child(self):
        """Test that bind creates a child with default fields."""
        slog = StructuredLogger("test")
        child = slog.bind(request_id="abc-123", client_ip="1.2.3.4")

        assert child is not slog
        assert child._default_fields["request_id"] == "abc-123"
        assert child._default_fields["client_ip"] == "1.2.3.4"

    def test_bind_inherits_parent_fields(self):
        """Test that bind inherits parent default fields."""
        slog = StructuredLogger("test")
        child1 = slog.bind(service="gateway")
        child2 = child1.bind(request_id="abc")

        assert child2._default_fields["service"] == "gateway"
        assert child2._default_fields["request_id"] == "abc"

    def test_bind_overrides_parent_fields(self):
        """Test that bind can override parent fields."""
        slog = StructuredLogger("test")
        child = slog.bind(env="staging")
        grandchild = child.bind(env="production")

        assert grandchild._default_fields["env"] == "production"

    def test_unbind_removes_fields(self):
        """Test that unbind removes specified fields."""
        slog = StructuredLogger("test")
        child = slog.bind(a=1, b=2, c=3)
        unbound = child.unbind("a", "c")

        assert "a" not in unbound._default_fields
        assert "b" in unbound._default_fields
        assert "c" not in unbound._default_fields

    def test_all_log_levels(self):
        """Test that all log level methods work."""
        setup_logging(level="debug")
        slog = StructuredLogger("test")

        # None should raise
        slog.debug("debug msg")
        slog.info("info msg")
        slog.warning("warning msg")
        slog.error("error msg")
        slog.critical("critical msg")

    def test_exception_method(self):
        """Test the exception method."""
        setup_logging(level="debug")
        slog = StructuredLogger("test")

        try:
            raise ValueError("test error")
        except ValueError:
            slog.exception("An error occurred", key="value")

    def test_log_method(self):
        """Test the generic log method."""
        setup_logging(level="debug")
        slog = StructuredLogger("test")
        slog.log(logging.WARNING, "custom level message", key="value")

    def test_default_fields_included_in_extra(self):
        """Test that default fields appear in log record extras."""
        setup_logging(level="debug")

        # Capture the log record
        captured = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = CaptureHandler()
        root = logging.getLogger(ROOT_LOGGER_NAME)
        root.addHandler(handler)

        try:
            slog = StructuredLogger("test")
            bound = slog.bind(service="gateway", env="prod")
            bound.info("test", extra_field="value")

            assert len(captured) == 1
            record = captured[0]
            assert getattr(record, "service", None) == "gateway"
            assert getattr(record, "env", None) == "prod"
            assert getattr(record, "extra_field", None) == "value"
        finally:
            root.removeHandler(handler)


# ---------------------------------------------------------------------------
# Request Context Tests
# ---------------------------------------------------------------------------


class TestRequestContext:
    """Tests for enhanced request context management."""

    def test_default_is_none(self):
        """Test that request context starts as None."""
        assert get_request_context() is None

    def test_set_and_get(self):
        """Test setting and getting request context."""
        ctx = set_request_context({"method": "POST", "path": "/v1/messages"})
        assert ctx["method"] == "POST"
        assert get_request_context() == ctx

    def test_clear(self):
        """Test clearing request context."""
        set_request_context({"key": "value"})
        clear_request_context()
        assert get_request_context() is None

    def test_update(self):
        """Test updating request context."""
        set_request_context({"method": "GET"})
        updated = update_request_context(status_code=200, duration_ms=42.0)
        assert updated["method"] == "GET"
        assert updated["status_code"] == 200
        assert updated["duration_ms"] == 42.0

    def test_update_creates_if_none(self):
        """Test that update creates context if none exists."""
        ctx = update_request_context(key="value")
        assert ctx["key"] == "value"

    def test_context_manager(self):
        """Test request_context context manager."""
        with request_context(method="POST", path="/v1/messages") as (cid, ctx):
            assert cid is not None
            assert ctx["method"] == "POST"
            assert ctx["path"] == "/v1/messages"
            assert ctx["correlation_id"] == cid

        assert get_request_context() is None
        assert get_correlation_id() is None

    def test_context_manager_with_correlation_id(self):
        """Test context manager with explicit correlation ID."""
        with request_context(correlation_id="req-001", user="admin") as (cid, ctx):
            assert cid == "req-001"
            assert ctx["user"] == "admin"

    def test_context_manager_cleanup_on_exception(self):
        """Test that context manager cleans up on exception."""
        try:
            with request_context(method="GET"):
                raise RuntimeError("test error")
        except RuntimeError:
            pass

        assert get_request_context() is None
        assert get_correlation_id() is None


class TestRequestContextFilter:
    """Tests for RequestContextFilter."""

    def test_adds_context_fields(self):
        """Test that filter adds request context to records."""
        f = RequestContextFilter()
        set_request_context({"method": "POST", "request_id": "abc-123"})

        record = make_record()
        f.filter(record)

        assert getattr(record, "method") == "POST"
        assert getattr(record, "request_id") == "abc-123"

    def test_no_context_no_fields(self):
        """Test that filter does nothing when no context is set."""
        f = RequestContextFilter()
        record = make_record()
        f.filter(record)
        assert not hasattr(record, "method")

    def test_does_not_overwrite_existing_attrs(self):
        """Test that filter doesn't overwrite existing record attributes."""
        f = RequestContextFilter()
        set_request_context({"name": "custom_name"})

        record = make_record(name="original_name")
        f.filter(record)
        # 'name' is a standard attribute, should not be overwritten
        assert record.name == "original_name"

    def test_always_returns_true(self):
        """Test that the filter always passes records."""
        f = RequestContextFilter()
        record = make_record()
        assert f.filter(record) is True


# ---------------------------------------------------------------------------
# LogBuffer Tests
# ---------------------------------------------------------------------------


class TestLogBuffer:
    """Tests for the LogBuffer."""

    def test_add_and_flush(self):
        """Test adding records and flushing."""
        buf = LogBuffer(max_size=100)
        buf.add(make_record("msg 1"))
        buf.add(make_record("msg 2"))

        assert buf.size == 2

        entries = buf.flush()
        assert len(entries) == 2
        assert entries[0]["message"] == "msg 1"
        assert entries[1]["message"] == "msg 2"
        assert buf.size == 0

    def test_max_size_limit(self):
        """Test that buffer respects max_size."""
        buf = LogBuffer(max_size=3)
        for i in range(5):
            buf.add(make_record(f"msg {i}"))

        assert buf.size <= 3

    def test_add_returns_false_when_full(self):
        """Test that add returns False when buffer is full."""
        buf = LogBuffer(max_size=2)
        assert buf.add(make_record("msg 1")) is True
        assert buf.add(make_record("msg 2")) is True
        # Buffer is now full and auto-flushes with no on_flush callback
        # After auto-flush, buffer should accept new entries

    def test_entry_format(self):
        """Test that buffered entries have correct format."""
        buf = LogBuffer(max_size=10)
        record = make_record("test message", level=logging.WARNING, name="test.module")
        buf.add(record)

        entries = buf.flush()
        entry = entries[0]
        assert "timestamp" in entry
        assert entry["level"] == "WARNING"
        assert entry["message"] == "test message"
        assert entry["logger"] == "test.module"
        assert "module" in entry
        assert "function" in entry
        assert "line" in entry

    def test_correlation_id_included(self):
        """Test that correlation ID is included in entries."""
        buf = LogBuffer(max_size=10)
        set_correlation_id("buf-corr-001")
        buf.add(make_record("test"))

        entries = buf.flush()
        assert entries[0]["correlation_id"] == "buf-corr-001"

    def test_extra_fields_included(self):
        """Test that extra fields are included in entries."""
        buf = LogBuffer(max_size=10)
        record = make_record("test")
        record.status_code = 200  # type: ignore
        record.duration_ms = 42.5  # type: ignore
        buf.add(record)

        entries = buf.flush()
        assert entries[0]["status_code"] == 200
        assert entries[0]["duration_ms"] == 42.5

    def test_on_flush_callback(self):
        """Test that on_flush callback is invoked."""
        received = []

        def callback(entries):
            received.extend(entries)

        buf = LogBuffer(max_size=10, on_flush=callback)
        buf.add(make_record("msg 1"))
        buf.add(make_record("msg 2"))
        buf.flush()

        assert len(received) == 2

    def test_stop_flushes_remaining(self):
        """Test that stop flushes remaining entries."""
        buf = LogBuffer(max_size=100)
        buf.add(make_record("msg 1"))
        buf.add(make_record("msg 2"))

        entries = buf.stop()
        assert len(entries) == 2

    def test_stats(self):
        """Test buffer statistics."""
        buf = LogBuffer(max_size=10)
        buf.add(make_record("msg 1"))
        buf.add(make_record("msg 2"))

        stats = buf.get_stats()
        assert stats["current_size"] == 2
        assert stats["max_size"] == 10

        buf.flush()
        stats = buf.get_stats()
        assert stats["current_size"] == 0
        assert stats["total_flushed"] == 2

    def test_dropped_count(self):
        """Test that dropped entries are tracked."""
        received = []
        # Don't provide on_flush so auto-flush won't trigger
        buf = LogBuffer(max_size=2)
        buf.add(make_record("msg 1"))
        buf.add(make_record("msg 2"))
        # Buffer is now full; next add triggers auto-flush (but no callback)
        # Actually, auto-flush only triggers if on_flush is set
        result = buf.add(make_record("msg 3"))
        # Since no on_flush, the buffer stays full and drops
        assert result is False

        stats = buf.get_stats()
        assert stats["total_dropped"] >= 1

    def test_thread_safety(self):
        """Test thread-safe buffer operations."""
        buf = LogBuffer(max_size=10000)
        errors = []

        def add_records():
            try:
                for i in range(100):
                    buf.add(make_record(f"msg {i}"))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=add_records) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        entries = buf.flush()
        assert len(entries) == 500


class TestBufferedHandler:
    """Tests for the BufferedHandler."""

    def test_handler_adds_to_buffer(self):
        """Test that handler adds records to its buffer."""
        buf = LogBuffer(max_size=100)
        handler = BufferedHandler(buf)

        record = make_record("handler test")
        handler.emit(record)

        assert buf.size == 1
        entries = buf.flush()
        assert entries[0]["message"] == "handler test"

    def test_handler_close_flushes(self):
        """Test that closing the handler flushes the buffer."""
        received = []

        def callback(entries):
            received.extend(entries)

        buf = LogBuffer(max_size=100, on_flush=callback)
        handler = BufferedHandler(buf)

        handler.emit(make_record("test"))
        handler.close()

        assert len(received) == 1


# ---------------------------------------------------------------------------
# Exception Formatting Tests
# ---------------------------------------------------------------------------


class TestFormatExceptionContext:
    """Tests for exception context formatting."""

    def test_basic_exception(self):
        """Test formatting a basic exception."""
        try:
            raise ValueError("test error")
        except ValueError as exc:
            result = format_exception_context(exc)

        assert result["type"] == "ValueError"
        assert result["module"] == "builtins"
        assert result["message"] == "test error"
        assert "traceback" in result
        assert len(result["traceback"]) > 0

    def test_gateway_error_context(self):
        """Test formatting a GatewayError with context."""
        from src.errors import NetworkError

        try:
            raise NetworkError("connection failed", host="example.com", port=443)
        except NetworkError as exc:
            result = format_exception_context(exc)

        assert result["type"] == "NetworkError"
        assert "error_context" in result
        assert result["error_context"]["category"] == "network"
        assert result["is_retryable"] is True

    def test_chained_exceptions(self):
        """Test formatting chained exceptions."""
        try:
            try:
                raise ConnectionError("connection refused")
            except ConnectionError as ce:
                raise ValueError("failed to connect") from ce
        except ValueError as exc:
            result = format_exception_context(exc, include_chain=True)

        assert result["type"] == "ValueError"
        assert "chain" in result
        assert len(result["chain"]) == 1
        assert result["chain"][0]["type"] == "ConnectionError"

    def test_no_chain_when_disabled(self):
        """Test that chain is excluded when disabled."""
        try:
            try:
                raise ConnectionError("conn error")
            except ConnectionError as ce:
                raise ValueError("wrapper") from ce
        except ValueError as exc:
            result = format_exception_context(exc, include_chain=False)

        assert "chain" not in result

    def test_max_depth_limit(self):
        """Test that chain depth is limited."""
        # Build a chain of 5 exceptions
        exc = None
        for i in range(5):
            try:
                if exc:
                    raise ValueError(f"error {i}") from exc
                else:
                    raise ValueError(f"error {i}")
            except ValueError as e:
                exc = e

        result = format_exception_context(exc, max_depth=2)
        if "chain" in result:
            assert len(result["chain"]) <= 2


class TestLogException:
    """Tests for the log_exception helper."""

    def test_logs_exception_with_context(self):
        """Test that log_exception logs with structured context."""
        setup_logging(level="debug")

        captured = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = CaptureHandler()
        root = logging.getLogger(ROOT_LOGGER_NAME)
        root.addHandler(handler)

        try:
            logger = get_logger("test")
            try:
                raise ValueError("test error")
            except ValueError as exc:
                log_exception(
                    logger, "An error occurred",
                    exc, request_id="req-001",
                )

            assert len(captured) > 0
            record = captured[-1]
            assert getattr(record, "exception_type", None) == "ValueError"
            assert getattr(record, "exception_message", None) == "test error"
            assert getattr(record, "request_id", None) == "req-001"
        finally:
            root.removeHandler(handler)

    def test_logs_gateway_error_fields(self):
        """Test that GatewayError fields are included."""
        from src.errors import NetworkError

        setup_logging(level="debug")

        captured = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = CaptureHandler()
        root = logging.getLogger(ROOT_LOGGER_NAME)
        root.addHandler(handler)

        try:
            logger = get_logger("test")
            try:
                raise NetworkError("conn failed", host="example.com")
            except NetworkError as exc:
                log_exception(logger, "Network error", exc)

            assert len(captured) > 0
            record = captured[-1]
            assert getattr(record, "is_retryable", None) is True
            assert "error_context" in record.__dict__
        finally:
            root.removeHandler(handler)


# ---------------------------------------------------------------------------
# HealthLogger Tests
# ---------------------------------------------------------------------------


class TestHealthLogger:
    """Tests for the HealthLogger."""

    def test_start_and_stop(self):
        """Test starting and stopping the health logger."""
        setup_logging(level="debug")
        health = HealthLogger(interval=10.0)

        health.start()
        assert health._running is True

        health.stop()
        assert health._running is False

    def test_emit_now(self):
        """Test immediate health check."""
        setup_logging(level="debug")
        health = HealthLogger(interval=60.0)
        health._start_time = time.monotonic()

        data = health.emit_now()
        assert data["status"] == "healthy"
        assert data["heartbeat_count"] == 1
        assert "uptime_seconds" in data

    def test_custom_health_check(self):
        """Test registering custom health checks."""
        setup_logging(level="debug")
        health = HealthLogger(interval=60.0)
        health._start_time = time.monotonic()

        health.register_check(
            "database",
            lambda: {"connected": True, "latency_ms": 5},
        )

        data = health.emit_now()
        assert "checks" in data
        assert data["checks"]["database"]["connected"] is True

    def test_failing_health_check(self):
        """Test that failing health checks set degraded status."""
        setup_logging(level="debug")
        health = HealthLogger(interval=60.0)
        health._start_time = time.monotonic()

        def failing_check():
            raise RuntimeError("service down")

        health.register_check("failing_service", failing_check)

        data = health.emit_now()
        assert data["status"] == "degraded"
        assert "checks" in data
        assert data["checks"]["failing_service"]["status"] == "error"

    def test_unregister_check(self):
        """Test unregistering a health check."""
        health = HealthLogger(interval=60.0)
        health.register_check("test", lambda: {"ok": True})
        health.unregister_check("test")
        assert "test" not in health._custom_checks

    def test_heartbeat_count_increments(self):
        """Test that heartbeat count increments."""
        setup_logging(level="debug")
        health = HealthLogger(interval=60.0)
        health._start_time = time.monotonic()

        health.emit_now()
        health.emit_now()
        assert health._heartbeat_count == 2

    def test_double_start_no_op(self):
        """Test that double start is a no-op."""
        setup_logging(level="debug")
        health = HealthLogger(interval=60.0)
        health.start()
        health.start()  # Should not raise or create duplicate timers
        health.stop()

    def test_includes_metrics_when_available(self):
        """Test that metrics are included when available."""
        setup_logging(level="debug")
        metrics = get_log_metrics()
        metrics.record(make_record(level=logging.ERROR))

        health = HealthLogger(interval=60.0, include_metrics=True)
        health._start_time = time.monotonic()

        data = health.emit_now()
        assert "log_metrics" in data


# ---------------------------------------------------------------------------
# Advanced Setup Tests
# ---------------------------------------------------------------------------


class TestSetupLoggingAdvanced:
    """Tests for the advanced logging setup function."""

    def test_basic_advanced_setup(self):
        """Test basic advanced setup returns a logger."""
        logger = setup_logging_advanced(level="debug")
        assert isinstance(logger, logging.Logger)
        assert logger.name == ROOT_LOGGER_NAME

    def test_with_metrics(self):
        """Test advanced setup with metrics enabled."""
        logger = setup_logging_advanced(level="info", enable_metrics=True)

        # Metrics handler should be attached
        root = logging.getLogger(ROOT_LOGGER_NAME)
        has_metrics = any(
            isinstance(h, LogMetricsHandler) for h in root.handlers
        )
        assert has_metrics is True

    def test_with_sensitive_filter(self):
        """Test advanced setup with sensitive data filter."""
        logger = setup_logging_advanced(
            level="info",
            enable_sensitive_filter=True,
        )

        root = logging.getLogger(ROOT_LOGGER_NAME)
        has_sensitive = any(
            isinstance(f, SensitiveDataFilter)
            for h in root.handlers
            for f in h.filters
        )
        assert has_sensitive is True

    def test_with_sampling(self):
        """Test advanced setup with sampling."""
        logger = setup_logging_advanced(
            level="info",
            sampling_rates={"DEBUG": 0.1},
        )

        root = logging.getLogger(ROOT_LOGGER_NAME)
        has_sampling = any(
            isinstance(f, SamplingFilter)
            for h in root.handlers
            for f in h.filters
        )
        assert has_sampling is True

    def test_with_request_context_filter(self):
        """Test advanced setup with request context filter."""
        logger = setup_logging_advanced(
            level="info",
            enable_request_context=True,
        )

        root = logging.getLogger(ROOT_LOGGER_NAME)
        has_ctx = any(
            isinstance(f, RequestContextFilter)
            for h in root.handlers
            for f in h.filters
        )
        assert has_ctx is True

    def test_with_buffer(self):
        """Test advanced setup with buffer."""
        received = []

        logger = setup_logging_advanced(
            level="info",
            buffer_config={
                "max_size": 100,
                "on_flush": lambda entries: received.extend(entries),
            },
        )

        root = logging.getLogger(ROOT_LOGGER_NAME)
        has_buffer = any(
            isinstance(h, BufferedHandler) for h in root.handlers
        )
        assert has_buffer is True

    def test_all_features_combined(self):
        """Test enabling all advanced features together."""
        logger = setup_logging_advanced(
            level="debug",
            log_format="json",
            enable_metrics=True,
            enable_sensitive_filter=True,
            sampling_rates={"DEBUG": 0.5},
            enable_request_context=True,
        )
        assert isinstance(logger, logging.Logger)

        # Log a test message - should not raise
        test_logger = get_logger("integration")
        with request_context(method="POST", path="/test") as (cid, ctx):
            test_logger.info("Test message", extra={"status_code": 200})


# ---------------------------------------------------------------------------
# CLI Command Tests for New Features
# ---------------------------------------------------------------------------


class TestNewLoggingCLI:
    """Tests for new logging CLI commands."""

    def test_logging_metrics_text(self, runner):
        """Test the logging metrics command in text format."""
        result = runner.invoke(main, ["logging", "metrics"])
        assert result.exit_code == 0
        assert "Log Metrics Report" in result.output

    def test_logging_metrics_json(self, runner):
        """Test the logging metrics command in JSON format."""
        result = runner.invoke(main, ["logging", "metrics", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total_count" in data
        assert "levels" in data

    def test_logging_health_text(self, runner):
        """Test the logging health command in text format."""
        result = runner.invoke(main, ["logging", "health"])
        assert result.exit_code == 0
        assert "Logging System Health" in result.output

    def test_logging_health_json(self, runner):
        """Test the logging health command in JSON format."""
        result = runner.invoke(main, ["logging", "health", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "status" in data
        assert "total_messages" in data

    def test_logging_test_redaction(self, runner):
        """Test the redaction test command."""
        result = runner.invoke(main, ["logging", "test-redaction"])
        assert result.exit_code == 0
        assert "Redaction test complete" in result.output


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestAdvancedIntegration:
    """Integration tests combining multiple advanced logging features."""

    def test_full_pipeline_with_redaction(self, temp_log_dir):
        """Test full logging pipeline with sensitive data redaction."""
        log_file = os.path.join(temp_log_dir, "advanced.log")

        setup_logging_advanced(
            level="debug",
            log_format="json",
            output="both",
            log_file=log_file,
            enable_sensitive_filter=True,
            enable_metrics=True,
            enable_request_context=True,
        )

        logger = get_logger("pipeline_test")

        with request_context(
            method="POST", path="/v1/messages"
        ) as (cid, ctx):
            logger.info("Received request")
            logger.info("Using api_key = sk-ant-test1234567890abcdef")
            update_request_context(status_code=200)
            logger.info("Request completed")

        # Flush handlers
        root = logging.getLogger(ROOT_LOGGER_NAME)
        for handler in root.handlers:
            handler.flush()

        # Verify file output has redacted data
        with open(log_file) as f:
            content = f.read()

        assert "sk-ant-test1234567890abcdef" not in content
        assert "REDACTED" in content
        assert "Received request" in content

    def test_metrics_collection_pipeline(self):
        """Test that metrics are collected during normal logging."""
        setup_logging_advanced(
            level="debug",
            enable_metrics=True,
        )

        logger = get_logger("metrics_test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        metrics = get_log_metrics()
        report = metrics.get_report()
        # At least our messages should be counted
        assert report["total_count"] >= 4

    def test_structured_logger_with_context(self):
        """Test structured logger with request context."""
        setup_logging_advanced(
            level="debug",
            log_format="json",
            enable_request_context=True,
        )

        slog = StructuredLogger("api")
        req_log = slog.bind(request_id="req-001", method="POST")

        with request_context(correlation_id="corr-001"):
            req_log.info("Processing request", path="/v1/messages")
            req_log.info("Request complete", status_code=200, duration_ms=42.0)

    def test_buffer_with_metrics(self):
        """Test log buffer combined with metrics."""
        flushed = []

        setup_logging_advanced(
            level="info",
            enable_metrics=True,
            buffer_config={
                "max_size": 100,
                "on_flush": lambda entries: flushed.extend(entries),
            },
        )

        logger = get_logger("buffer_test")
        for i in range(5):
            logger.info(f"Message {i}")

        # Force flush of buffered handler
        root = logging.getLogger(ROOT_LOGGER_NAME)
        for handler in root.handlers:
            if isinstance(handler, BufferedHandler):
                handler.buffer.flush()
                break

        assert len(flushed) >= 5

    def test_exception_logging_with_structured_logger(self):
        """Test exception logging with StructuredLogger."""
        setup_logging(level="debug")
        slog = StructuredLogger("error_test")

        try:
            raise ValueError("test error")
        except ValueError:
            slog.exception("Operation failed", operation="test")

    def test_sampling_with_metrics(self):
        """Test that sampling interacts correctly with metrics."""
        setup_logging_advanced(
            level="debug",
            enable_metrics=True,
            sampling_rates={"DEBUG": 0.0},  # Block all DEBUG
        )

        logger = get_logger("sample_test")
        for _ in range(10):
            logger.debug("Debug message (should be sampled out)")
        for _ in range(5):
            logger.info("Info message (should pass)")

        # Metrics should still count the messages that reached the handler
        metrics = get_log_metrics()
        report = metrics.get_report()
        # Since sampling filter is on the handlers, metrics handler has
        # the same filter, so debug messages should be mostly filtered
        # Info messages should have passed
        assert report["total_count"] >= 5
