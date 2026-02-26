"""Tests for the retry logic module."""

import threading
import time

import pytest

from src.errors import (
    CircuitOpenError,
    GatewayError,
    NetworkError,
    RateLimitError,
    RetryExhaustedError,
    AuthenticationError,
    ErrorContext,
    ErrorCategory,
)
from src.retry import (
    BackoffStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
    RetryContext,
    RetryStats,
    clear_breaker_registry,
    get_circuit_breaker,
    list_circuit_breakers,
    reset_all_circuit_breakers,
    retry_call,
    retry_with_circuit_breaker,
    with_retry,
)


# --------------------------------------------------------------------------- #
# RetryConfig tests
# --------------------------------------------------------------------------- #


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_config(self):
        """Default config has sensible values."""
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 60.0
        assert cfg.jitter is True

    def test_exponential_backoff(self):
        """Exponential backoff doubles with each attempt."""
        cfg = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False,
        )
        # attempt=1: 1.0 * 2^0 = 1.0
        assert cfg.compute_delay(1) == 1.0
        # attempt=2: 1.0 * 2^1 = 2.0
        assert cfg.compute_delay(2) == 2.0
        # attempt=3: 1.0 * 2^2 = 4.0
        assert cfg.compute_delay(3) == 4.0

    def test_linear_backoff(self):
        """Linear backoff increases linearly."""
        cfg = RetryConfig(
            backoff_strategy=BackoffStrategy.LINEAR,
            base_delay=2.0,
            jitter=False,
        )
        assert cfg.compute_delay(1) == 2.0
        assert cfg.compute_delay(2) == 4.0
        assert cfg.compute_delay(3) == 6.0

    def test_constant_backoff(self):
        """Constant backoff returns the same delay."""
        cfg = RetryConfig(
            backoff_strategy=BackoffStrategy.CONSTANT,
            base_delay=5.0,
            jitter=False,
        )
        assert cfg.compute_delay(1) == 5.0
        assert cfg.compute_delay(2) == 5.0
        assert cfg.compute_delay(5) == 5.0

    def test_max_delay_cap(self):
        """Delay is capped at max_delay."""
        cfg = RetryConfig(
            base_delay=10.0,
            max_delay=15.0,
            exponential_base=10.0,
            jitter=False,
        )
        # attempt=3: 10 * 10^2 = 1000, capped to 15
        assert cfg.compute_delay(3) == 15.0

    def test_jitter_varies_delay(self):
        """Jitter produces varying delays."""
        cfg = RetryConfig(
            base_delay=10.0,
            jitter=True,
            jitter_range=(0.5, 1.5),
        )
        delays = {cfg.compute_delay(1) for _ in range(20)}
        # With jitter, we should get different values
        assert len(delays) > 1

    def test_should_retry_retryable_error(self):
        """should_retry returns True for retryable errors."""
        cfg = RetryConfig()
        err = NetworkError("fail")
        assert cfg.should_retry(err) is True

    def test_should_retry_non_retryable_error(self):
        """should_retry returns False for non-retryable errors."""
        cfg = RetryConfig()
        err = AuthenticationError("bad key")
        assert cfg.should_retry(err) is False

    def test_should_retry_with_retry_on_types(self):
        """should_retry checks retry_on exception types."""
        cfg = RetryConfig(retry_on=(ValueError, TypeError))
        assert cfg.should_retry(ValueError("fail")) is True
        assert cfg.should_retry(TypeError("fail")) is True
        assert cfg.should_retry(RuntimeError("fail")) is False

    def test_should_retry_with_status_codes(self):
        """should_retry checks retry_on_status codes."""
        cfg = RetryConfig(retry_on_status={429, 503})
        err429 = RateLimitError(status_code=429)
        err503 = GatewayError(
            "fail",
            context=ErrorContext(category=ErrorCategory.PROVIDER, status_code=503),
        )
        err400 = GatewayError(
            "fail",
            context=ErrorContext(category=ErrorCategory.VALIDATION, status_code=400),
        )
        assert cfg.should_retry(err429) is True
        assert cfg.should_retry(err503) is True
        assert cfg.should_retry(err400) is False


# --------------------------------------------------------------------------- #
# retry_call tests
# --------------------------------------------------------------------------- #


class TestRetryCall:
    """Tests for the retry_call function."""

    def test_success_on_first_try(self):
        """Function succeeds on first try — no retries needed."""
        result = retry_call(lambda: 42, config=RetryConfig(max_attempts=3))
        assert result == 42

    def test_success_after_retries(self):
        """Function succeeds after a few retries."""
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("transient failure")
            return "success"

        result = retry_call(
            flaky,
            config=RetryConfig(
                max_attempts=5,
                base_delay=0.01,
                jitter=False,
            ),
        )
        assert result == "success"
        assert call_count == 3

    def test_exhausted_retries_raises(self):
        """RetryExhaustedError is raised when all attempts fail."""

        def always_fail():
            raise NetworkError("always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            retry_call(
                always_fail,
                config=RetryConfig(max_attempts=3, base_delay=0.01, jitter=False),
            )

        err = exc_info.value
        assert err.attempts == 3
        assert len(err.all_errors) == 3
        assert err.total_elapsed > 0

    def test_non_retryable_error_raises_immediately(self):
        """Non-retryable errors are raised immediately without retry."""
        call_count = 0

        def bad_auth():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("bad key")

        with pytest.raises(AuthenticationError):
            retry_call(
                bad_auth,
                config=RetryConfig(max_attempts=5, base_delay=0.01),
            )

        assert call_count == 1  # Only one attempt

    def test_total_timeout_respected(self):
        """Retries stop when total_timeout is exceeded."""

        def slow_fail():
            time.sleep(0.05)
            raise NetworkError("slow")

        with pytest.raises(RetryExhaustedError):
            retry_call(
                slow_fail,
                config=RetryConfig(
                    max_attempts=100,
                    base_delay=0.01,
                    total_timeout=0.15,
                    jitter=False,
                ),
            )

    def test_on_retry_callback(self):
        """on_retry callback is invoked before each retry."""
        retries_seen = []

        def on_retry(attempt, delay, exc):
            retries_seen.append((attempt, type(exc).__name__))

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("fail")
            return "ok"

        retry_call(
            flaky,
            config=RetryConfig(
                max_attempts=5,
                base_delay=0.01,
                jitter=False,
                on_retry=on_retry,
            ),
        )

        assert len(retries_seen) == 2
        assert retries_seen[0] == (1, "NetworkError")
        assert retries_seen[1] == (2, "NetworkError")

    def test_retry_after_respected(self):
        """Retry-After hint from error context is respected."""
        call_count = 0

        def rate_limited():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(retry_after=0.05)
            return "ok"

        start = time.monotonic()
        result = retry_call(
            rate_limited,
            config=RetryConfig(
                max_attempts=3,
                base_delay=0.01,
                jitter=False,
            ),
        )
        elapsed = time.monotonic() - start

        assert result == "ok"
        assert elapsed >= 0.04  # Should have waited for retry_after

    def test_args_and_kwargs_passed(self):
        """Arguments are forwarded to the function."""
        def add(a, b, extra=0):
            return a + b + extra

        result = retry_call(
            add,
            args=(1, 2),
            kwargs={"extra": 10},
            config=RetryConfig(max_attempts=1),
        )
        assert result == 13

    def test_default_config(self):
        """retry_call works with no explicit config."""
        result = retry_call(lambda: "ok")
        assert result == "ok"


# --------------------------------------------------------------------------- #
# with_retry decorator tests
# --------------------------------------------------------------------------- #


class TestWithRetryDecorator:
    """Tests for the @with_retry decorator."""

    def test_decorator_success(self):
        """Decorated function succeeds normally."""

        @with_retry(max_attempts=3, base_delay=0.01)
        def greet(name):
            return f"hello {name}"

        assert greet("world") == "hello world"

    def test_decorator_retries(self):
        """Decorated function retries on failure."""
        call_count = 0

        @with_retry(max_attempts=5, base_delay=0.01, jitter=False)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("fail")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3

    def test_decorator_exhausted(self):
        """Decorated function raises RetryExhaustedError when exhausted."""

        @with_retry(max_attempts=2, base_delay=0.01)
        def always_fail():
            raise NetworkError("nope")

        with pytest.raises(RetryExhaustedError):
            always_fail()

    def test_decorator_has_retry_config(self):
        """Decorated function exposes retry_config attribute."""

        @with_retry(max_attempts=5, base_delay=2.0)
        def func():
            pass

        assert func.retry_config.max_attempts == 5
        assert func.retry_config.base_delay == 2.0

    def test_decorator_preserves_name(self):
        """Decorator preserves function name and docstring."""

        @with_retry(max_attempts=3)
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


# --------------------------------------------------------------------------- #
# RetryContext tests
# --------------------------------------------------------------------------- #


class TestRetryContext:
    """Tests for the RetryContext context manager."""

    def test_success_on_first_try(self):
        """Context succeeds on first try."""
        ctx = RetryContext(max_attempts=3, base_delay=0.01, jitter=False)
        for attempt in ctx:
            with attempt:
                result = "ok"
                break

        assert result == "ok"
        assert ctx.stats.successful is True
        assert ctx.stats.total_attempts == 1

    def test_success_after_retries(self):
        """Context retries and eventually succeeds."""
        count = 0
        ctx = RetryContext(max_attempts=5, base_delay=0.01, jitter=False)

        for attempt in ctx:
            with attempt:
                count += 1
                if count < 3:
                    raise NetworkError("fail")
                result = "ok"

        assert result == "ok"
        assert ctx.stats.successful is True
        assert ctx.stats.total_attempts == 3

    def test_exhausted_raises(self):
        """Context raises RetryExhaustedError when exhausted."""
        ctx = RetryContext(max_attempts=2, base_delay=0.01, jitter=False)

        with pytest.raises(RetryExhaustedError):
            for attempt in ctx:
                with attempt:
                    raise NetworkError("always fails")

    def test_non_retryable_propagates(self):
        """Non-retryable errors propagate immediately."""
        ctx = RetryContext(max_attempts=5, base_delay=0.01)

        with pytest.raises(AuthenticationError):
            for attempt in ctx:
                with attempt:
                    raise AuthenticationError("bad key")

        assert ctx.stats.total_attempts == 1


# --------------------------------------------------------------------------- #
# RetryStats tests
# --------------------------------------------------------------------------- #


class TestRetryStats:
    """Tests for RetryStats dataclass."""

    def test_default_stats(self):
        """Default stats are zeroed out."""
        stats = RetryStats()
        assert stats.total_attempts == 0
        assert stats.successful is False
        assert stats.errors == []
        assert stats.delays == []
        assert stats.total_elapsed == 0.0

    def test_to_dict(self):
        """to_dict() serializes stats."""
        stats = RetryStats(
            total_attempts=3,
            successful=True,
            errors=[ValueError("a"), ValueError("b")],
            delays=[0.5, 1.0],
            total_elapsed=2.5,
        )
        d = stats.to_dict()
        assert d["total_attempts"] == 3
        assert d["successful"] is True
        assert d["error_count"] == 2
        assert d["delays"] == [0.5, 1.0]
        assert d["total_elapsed"] == 2.5
        assert d["error_types"] == ["ValueError", "ValueError"]


# --------------------------------------------------------------------------- #
# CircuitBreaker tests
# --------------------------------------------------------------------------- #


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Circuit starts in CLOSED state."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_stays_closed_on_success(self):
        """Circuit stays closed after successes."""
        cb = CircuitBreaker("test")
        for _ in range(10):
            cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        """Circuit opens after failure_threshold failures."""
        cfg = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config=cfg)

        for _ in range(3):
            cb.record_failure(NetworkError("fail"))

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    def test_open_rejects_requests(self):
        """Open circuit rejects requests."""
        cfg = CircuitBreakerConfig(failure_threshold=1, reset_timeout=100.0)
        cb = CircuitBreaker("test", config=cfg)

        cb.record_failure(NetworkError("fail"))
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_transitions_to_half_open(self):
        """Circuit transitions to HALF_OPEN after reset_timeout."""
        cfg = CircuitBreakerConfig(
            failure_threshold=1,
            reset_timeout=0.05,
        )
        cb = CircuitBreaker("test", config=cfg)

        cb.record_failure(NetworkError("fail"))
        assert cb.state == CircuitState.OPEN

        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True

    def test_half_open_closes_on_success(self):
        """Circuit closes when half-open requests succeed."""
        cfg = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            reset_timeout=0.01,
        )
        cb = CircuitBreaker("test", config=cfg)

        cb.record_failure(NetworkError("fail"))
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """Circuit re-opens if half-open request fails."""
        cfg = CircuitBreakerConfig(
            failure_threshold=1,
            reset_timeout=0.01,
        )
        cb = CircuitBreaker("test", config=cfg)

        cb.record_failure(NetworkError("fail"))
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure(NetworkError("fail again"))
        assert cb.state == CircuitState.OPEN

    def test_reset(self):
        """reset() forces circuit to CLOSED."""
        cfg = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config=cfg)

        cb.record_failure(NetworkError("fail"))
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_context_manager_success(self):
        """Circuit breaker works as a context manager."""
        cb = CircuitBreaker("test")

        with cb:
            pass  # Success

        assert cb.state == CircuitState.CLOSED

    def test_context_manager_raises_when_open(self):
        """Context manager raises CircuitOpenError when open."""
        cfg = CircuitBreakerConfig(failure_threshold=1, reset_timeout=100.0)
        cb = CircuitBreaker("test", config=cfg)
        cb.record_failure(NetworkError("fail"))

        with pytest.raises(CircuitOpenError):
            with cb:
                pass

    def test_context_manager_records_failure(self):
        """Context manager records failures from exceptions."""
        cfg = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config=cfg)

        with pytest.raises(NetworkError):
            with cb:
                raise NetworkError("fail")

        assert cb.failure_count == 1

    def test_decorator_usage(self):
        """Circuit breaker works as a decorator."""
        cb = CircuitBreaker("test")
        call_count = 0

        @cb
        def my_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert my_func() == "ok"
        assert call_count == 1
        assert cb.state == CircuitState.CLOSED

    def test_decorator_has_circuit_breaker_attr(self):
        """Decorated function exposes circuit_breaker attribute."""
        cb = CircuitBreaker("test")

        @cb
        def my_func():
            pass

        assert my_func.circuit_breaker is cb

    def test_get_stats(self):
        """get_stats() returns circuit state."""
        cfg = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config=cfg)
        cb.record_failure(NetworkError("fail"))

        stats = cb.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1
        assert stats["failure_threshold"] == 5

    def test_half_open_max_calls(self):
        """Only half_open_max_calls are allowed through in HALF_OPEN."""
        cfg = CircuitBreakerConfig(
            failure_threshold=1,
            reset_timeout=0.01,
            half_open_max_calls=1,
        )
        cb = CircuitBreaker("test", config=cfg)

        cb.record_failure(NetworkError("fail"))
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        # First call allowed
        assert cb.allow_request() is True
        # Second call rejected (max 1 in half-open)
        assert cb.allow_request() is False

    def test_success_resets_failure_count(self):
        """Success in CLOSED state resets failure count."""
        cfg = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config=cfg)

        cb.record_failure(NetworkError("fail"))
        cb.record_failure(NetworkError("fail"))
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0


# --------------------------------------------------------------------------- #
# retry_with_circuit_breaker tests
# --------------------------------------------------------------------------- #


class TestRetryWithCircuitBreaker:
    """Tests for retry_with_circuit_breaker()."""

    def test_success(self):
        """Successful call works through both layers."""
        cb = CircuitBreaker("test")
        result = retry_with_circuit_breaker(
            lambda: "ok",
            retry_config=RetryConfig(max_attempts=3, base_delay=0.01),
            circuit_breaker=cb,
        )
        assert result == "ok"

    def test_retry_with_circuit_breaker_retries(self):
        """Function retries through the circuit breaker."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=10))
        count = 0

        def flaky():
            nonlocal count
            count += 1
            if count < 3:
                raise NetworkError("transient")
            return "ok"

        result = retry_with_circuit_breaker(
            flaky,
            retry_config=RetryConfig(max_attempts=5, base_delay=0.01, jitter=False),
            circuit_breaker=cb,
        )
        assert result == "ok"
        assert count == 3

    def test_circuit_open_stops_retries(self):
        """CircuitOpenError is non-retryable, stopping retries."""
        cfg = CircuitBreakerConfig(failure_threshold=1, reset_timeout=100.0)
        cb = CircuitBreaker("test", config=cfg)
        cb.record_failure(NetworkError("fail"))  # Open the circuit

        with pytest.raises(CircuitOpenError):
            retry_with_circuit_breaker(
                lambda: "ok",
                retry_config=RetryConfig(max_attempts=3, base_delay=0.01),
                circuit_breaker=cb,
            )


# --------------------------------------------------------------------------- #
# Circuit breaker registry tests
# --------------------------------------------------------------------------- #


class TestCircuitBreakerRegistry:
    """Tests for the global circuit breaker registry."""

    def setup_method(self):
        """Clear the registry before each test."""
        clear_breaker_registry()

    def test_get_creates_new_breaker(self):
        """get_circuit_breaker creates a new breaker."""
        cb = get_circuit_breaker("test-service")
        assert cb.name == "test-service"

    def test_get_returns_same_breaker(self):
        """get_circuit_breaker returns the same breaker for same name."""
        cb1 = get_circuit_breaker("test-service")
        cb2 = get_circuit_breaker("test-service")
        assert cb1 is cb2

    def test_list_breakers(self):
        """list_circuit_breakers returns all registered breakers."""
        get_circuit_breaker("service-a")
        get_circuit_breaker("service-b")
        breakers = list_circuit_breakers()
        assert "service-a" in breakers
        assert "service-b" in breakers

    def test_reset_all(self):
        """reset_all_circuit_breakers resets all to CLOSED."""
        cb = get_circuit_breaker(
            "test",
            CircuitBreakerConfig(failure_threshold=1),
        )
        cb.record_failure(NetworkError("fail"))
        assert cb.state == CircuitState.OPEN

        reset_all_circuit_breakers()
        assert cb.state == CircuitState.CLOSED

    def test_clear_registry(self):
        """clear_breaker_registry removes all breakers."""
        get_circuit_breaker("service-a")
        get_circuit_breaker("service-b")
        clear_breaker_registry()
        assert list_circuit_breakers() == {}


# --------------------------------------------------------------------------- #
# BackoffStrategy enum tests
# --------------------------------------------------------------------------- #


class TestBackoffStrategy:
    """Tests for BackoffStrategy enum."""

    def test_values(self):
        """All expected strategies exist."""
        assert BackoffStrategy.EXPONENTIAL == "exponential"
        assert BackoffStrategy.LINEAR == "linear"
        assert BackoffStrategy.CONSTANT == "constant"

    def test_string_comparison(self):
        """Strategies compare as strings."""
        assert BackoffStrategy.EXPONENTIAL == "exponential"


# --------------------------------------------------------------------------- #
# Edge case tests
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_zero_delay(self):
        """Zero base_delay produces zero delay."""
        cfg = RetryConfig(base_delay=0.0, jitter=False)
        assert cfg.compute_delay(1) == 0.0

    def test_single_attempt_no_retry(self):
        """max_attempts=1 means no retries."""
        call_count = 0

        def fail():
            nonlocal call_count
            call_count += 1
            raise NetworkError("fail")

        with pytest.raises(RetryExhaustedError):
            retry_call(
                fail,
                config=RetryConfig(max_attempts=1, base_delay=0.01),
            )

        assert call_count == 1

    def test_on_retry_callback_error_ignored(self):
        """Errors in on_retry callback don't break the retry loop."""
        def bad_callback(attempt, delay, exc):
            raise RuntimeError("callback boom")

        count = 0

        def flaky():
            nonlocal count
            count += 1
            if count < 3:
                raise NetworkError("fail")
            return "ok"

        result = retry_call(
            flaky,
            config=RetryConfig(
                max_attempts=5,
                base_delay=0.01,
                jitter=False,
                on_retry=bad_callback,
            ),
        )
        assert result == "ok"

    def test_thread_safety_of_circuit_breaker(self):
        """Circuit breaker is thread-safe under concurrent access."""
        cfg = CircuitBreakerConfig(failure_threshold=100)
        cb = CircuitBreaker("thread-test", config=cfg)
        errors = []

        def record_many():
            try:
                for _ in range(50):
                    cb.record_failure(NetworkError("fail"))
                    cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0  # No exceptions during concurrent access
