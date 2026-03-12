"""Retry logic with configurable backoff strategies and circuit breaker.

Provides a production-grade retry framework for the model gateway:
- Multiple backoff strategies (exponential, linear, constant)
- Jitter to prevent thundering herd
- Selective retry based on exception type / retryability
- Circuit breaker pattern to fail fast when a service is down
- Decorator and context-manager interfaces
- Structured logging and metrics
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from src.errors import (
    CircuitOpenError,
    GatewayError,
    RetryExhaustedError,
    is_retryable_exception,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# --------------------------------------------------------------------------- #
# Backoff strategies
# --------------------------------------------------------------------------- #


class BackoffStrategy(str, Enum):
    """Available backoff strategies for retry delays."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"


@dataclass
class RetryConfig:
    """Configuration for retry behaviour.

    Attributes:
        max_attempts: Maximum number of attempts (including the initial one).
            A value of 1 means no retries. 0 means unlimited retries
            (use with caution and always pair with a total timeout).
        backoff_strategy: How to calculate the delay between retries.
        base_delay: Base delay in seconds for the first retry.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Multiplier for exponential backoff (default 2).
        jitter: Whether to add random jitter to the delay.
        jitter_range: Tuple (min_factor, max_factor) applied to the
            computed delay. E.g. (0.5, 1.5) means the actual delay
            is between 50% and 150% of the computed delay.
        retry_on: Optional tuple of exception types to retry on.
            If None, uses :func:`is_retryable_exception` to decide.
        retry_on_status: Optional set of HTTP status codes that should
            trigger a retry (checked via GatewayError.context.status_code).
        total_timeout: Optional total timeout in seconds across all
            attempts. If exceeded, retries stop regardless of max_attempts.
        on_retry: Optional callback invoked before each retry. Receives
            (attempt, delay, exception).
    """

    max_attempts: int = 3
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: tuple[float, float] = (0.5, 1.5)
    retry_on: Optional[tuple[type[Exception], ...]] = None
    retry_on_status: Optional[set[int]] = None
    total_timeout: Optional[float] = None
    on_retry: Optional[Callable[[int, float, Exception], None]] = None

    def compute_delay(self, attempt: int) -> float:
        """Calculate the delay before the next retry attempt.

        Args:
            attempt: The attempt number (1-based, so attempt=1 is
                the delay before the second try).

        Returns:
            Delay in seconds (with jitter applied if enabled).
        """
        if self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay * attempt
        else:  # CONSTANT
            delay = self.base_delay

        # Cap
        delay = min(delay, self.max_delay)

        # Jitter
        if self.jitter:
            lo, hi = self.jitter_range
            delay *= random.uniform(lo, hi)

        return max(0.0, delay)

    def should_retry(self, exc: Exception) -> bool:
        """Decide whether an exception is retryable under this config.

        Args:
            exc: The exception that was raised.

        Returns:
            True if the exception should trigger a retry.
        """
        # Check status-code based retries
        if self.retry_on_status and isinstance(exc, GatewayError):
            status = exc.context.status_code
            if status is not None and status in self.retry_on_status:
                return True

        # Check type-based retries
        if self.retry_on is not None:
            return isinstance(exc, self.retry_on)

        # Default: use the exception's retryable flag
        return is_retryable_exception(exc)


# --------------------------------------------------------------------------- #
# Retry statistics
# --------------------------------------------------------------------------- #


@dataclass
class RetryStats:
    """Statistics gathered during a retry sequence.

    Attributes:
        total_attempts: Number of attempts made.
        successful: Whether the operation eventually succeeded.
        errors: List of exceptions encountered (in order).
        delays: List of delays (seconds) between attempts.
        total_elapsed: Total wall-clock time across all attempts.
        final_result: The return value of the operation (if successful).
    """

    total_attempts: int = 0
    successful: bool = False
    errors: list[Exception] = field(default_factory=list)
    delays: list[float] = field(default_factory=list)
    total_elapsed: float = 0.0
    final_result: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for logging."""
        return {
            "total_attempts": self.total_attempts,
            "successful": self.successful,
            "error_count": len(self.errors),
            "delays": self.delays,
            "total_elapsed": round(self.total_elapsed, 3),
            "error_types": [type(e).__name__ for e in self.errors],
        }


# --------------------------------------------------------------------------- #
# Core retry executor
# --------------------------------------------------------------------------- #


def retry_call(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: Optional[dict[str, Any]] = None,
    config: Optional[RetryConfig] = None,
) -> Any:
    """Execute *func* with automatic retry on failure.

    This is the core retry engine. It repeatedly calls *func* until
    it succeeds, the maximum number of attempts is reached, or the
    total timeout expires.

    Args:
        func: The callable to execute.
        args: Positional arguments for *func*.
        kwargs: Keyword arguments for *func*.
        config: Retry configuration (uses defaults if not provided).

    Returns:
        The return value of *func* on success.

    Raises:
        RetryExhaustedError: If all retries are exhausted.
        Exception: If a non-retryable exception is raised.
    """
    if kwargs is None:
        kwargs = {}
    if config is None:
        config = RetryConfig()

    stats = RetryStats()
    start_time = time.monotonic()
    attempt = 0
    unlimited = config.max_attempts == 0

    while unlimited or attempt < config.max_attempts:
        attempt += 1
        stats.total_attempts = attempt

        try:
            result = func(*args, **kwargs)
            stats.successful = True
            stats.final_result = result
            stats.total_elapsed = time.monotonic() - start_time

            if attempt > 1:
                logger.info(
                    "Retry succeeded on attempt %d/%s after %.2fs",
                    attempt,
                    "∞" if unlimited else config.max_attempts,
                    stats.total_elapsed,
                )
            return result

        except Exception as exc:
            stats.errors.append(exc)
            elapsed = time.monotonic() - start_time
            stats.total_elapsed = elapsed

            # Non-retryable? Raise immediately
            if not config.should_retry(exc):
                logger.debug(
                    "Non-retryable error on attempt %d: %s",
                    attempt, exc,
                )
                raise

            # Respect Retry-After from the error context
            retry_after_hint: Optional[float] = None
            if isinstance(exc, GatewayError) and exc.context.retry_after:
                retry_after_hint = exc.context.retry_after

            # Last attempt?
            if not unlimited and attempt >= config.max_attempts:
                break

            # Total timeout exceeded?
            if config.total_timeout and elapsed >= config.total_timeout:
                logger.warning(
                    "Total retry timeout (%.1fs) exceeded after %d attempts",
                    config.total_timeout,
                    attempt,
                )
                break

            # Calculate delay
            delay = config.compute_delay(attempt)
            if retry_after_hint is not None:
                delay = max(delay, retry_after_hint)

            # Check that waiting won't exceed total timeout
            if config.total_timeout:
                remaining = config.total_timeout - elapsed
                if delay > remaining:
                    delay = max(0.0, remaining)

            stats.delays.append(delay)

            logger.warning(
                "Attempt %d/%s failed: %s — retrying in %.2fs",
                attempt,
                "∞" if unlimited else config.max_attempts,
                exc,
                delay,
            )

            # Callback
            if config.on_retry:
                try:
                    config.on_retry(attempt, delay, exc)
                except Exception:
                    pass  # Don't let callback failures break the retry loop

            if delay > 0:
                time.sleep(delay)

    # All attempts exhausted
    stats.total_elapsed = time.monotonic() - start_time
    last_error = stats.errors[-1] if stats.errors else None

    raise RetryExhaustedError(
        f"All {attempt} retry attempts exhausted after {stats.total_elapsed:.2f}s",
        attempts=attempt,
        total_elapsed=stats.total_elapsed,
        last_error=last_error,
        errors=stats.errors,
    )


# --------------------------------------------------------------------------- #
# Decorator
# --------------------------------------------------------------------------- #


def with_retry(
    max_attempts: int = 3,
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retry_on: Optional[tuple[type[Exception], ...]] = None,
    retry_on_status: Optional[set[int]] = None,
    total_timeout: Optional[float] = None,
    on_retry: Optional[Callable[[int, float, Exception], None]] = None,
) -> Callable[[F], F]:
    """Decorator that adds automatic retry to a function.

    Usage::

        @with_retry(max_attempts=3, base_delay=1.0)
        def call_api(url: str) -> dict:
            ...

    All parameters map to :class:`RetryConfig` attributes.

    Args:
        max_attempts: Maximum number of attempts.
        backoff: Backoff strategy.
        base_delay: Base delay in seconds.
        max_delay: Maximum delay cap.
        jitter: Whether to apply jitter.
        retry_on: Exception types to retry on.
        retry_on_status: HTTP status codes to retry on.
        total_timeout: Total timeout across all attempts.
        on_retry: Callback before each retry.

    Returns:
        A decorator that wraps the function with retry logic.
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        backoff_strategy=backoff,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter,
        retry_on=retry_on,
        retry_on_status=retry_on_status,
        total_timeout=total_timeout,
        on_retry=on_retry,
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return retry_call(func, args=args, kwargs=kwargs, config=config)

        # Attach config for introspection
        wrapper.retry_config = config  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


# --------------------------------------------------------------------------- #
# Context manager
# --------------------------------------------------------------------------- #


class RetryContext:
    """Context manager that retries its body on failure.

    Usage::

        ctx = RetryContext(max_attempts=3, base_delay=0.5)
        for attempt in ctx:
            with attempt:
                result = call_api()

    The context manager iterates through attempts and catches
    retryable exceptions, sleeping between attempts.

    Args:
        max_attempts: Maximum number of attempts.
        config: Full retry config (overrides other params).
        **kwargs: Passed to :class:`RetryConfig`.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        config: Optional[RetryConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config:
            self.config = config
        else:
            self.config = RetryConfig(max_attempts=max_attempts, **kwargs)
        self.stats = RetryStats()
        self._start_time = 0.0

    def __iter__(self):
        """Iterate through retry attempts."""
        self._start_time = time.monotonic()
        attempt = 0
        unlimited = self.config.max_attempts == 0

        while unlimited or attempt < self.config.max_attempts:
            attempt += 1
            self.stats.total_attempts = attempt
            yield _Attempt(self, attempt)

            if self.stats.successful:
                return

            # Check total timeout
            elapsed = time.monotonic() - self._start_time
            if self.config.total_timeout and elapsed >= self.config.total_timeout:
                break

        # If we get here, all attempts were exhausted
        self.stats.total_elapsed = time.monotonic() - self._start_time
        if self.stats.errors:
            last_error = self.stats.errors[-1]
            raise RetryExhaustedError(
                f"All {attempt} retry attempts exhausted "
                f"after {self.stats.total_elapsed:.2f}s",
                attempts=attempt,
                total_elapsed=self.stats.total_elapsed,
                last_error=last_error,
                errors=self.stats.errors,
            )


class _Attempt:
    """Represents a single retry attempt within a RetryContext."""

    def __init__(self, ctx: RetryContext, number: int) -> None:
        self._ctx = ctx
        self.number = number

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            self._ctx.stats.successful = True
            return False

        if not self._ctx.config.should_retry(exc_val):
            return False  # Propagate non-retryable exceptions

        self._ctx.stats.errors.append(exc_val)
        elapsed = time.monotonic() - self._ctx._start_time

        unlimited = self._ctx.config.max_attempts == 0
        if not unlimited and self.number >= self._ctx.config.max_attempts:
            return True  # Last attempt — suppress so __iter__ raises RetryExhaustedError

        if self._ctx.config.total_timeout and elapsed >= self._ctx.config.total_timeout:
            return True  # Timeout — suppress so __iter__ raises RetryExhaustedError

        # Compute delay
        delay = self._ctx.config.compute_delay(self.number)

        # Respect Retry-After
        if isinstance(exc_val, GatewayError) and exc_val.context.retry_after:
            delay = max(delay, exc_val.context.retry_after)

        if self._ctx.config.total_timeout:
            remaining = self._ctx.config.total_timeout - elapsed
            delay = min(delay, max(0.0, remaining))

        self._ctx.stats.delays.append(delay)

        logger.warning(
            "Attempt %d/%s failed: %s — retrying in %.2fs",
            self.number,
            "∞" if unlimited else self._ctx.config.max_attempts,
            exc_val,
            delay,
        )

        if self._ctx.config.on_retry:
            try:
                self._ctx.config.on_retry(self.number, delay, exc_val)
            except Exception:
                pass

        if delay > 0:
            time.sleep(delay)

        return True  # Suppress exception and continue loop


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


class CircuitState(str, Enum):
    """States of a circuit breaker."""

    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening the circuit.
        success_threshold: Number of successes in half-open state before
            closing the circuit.
        reset_timeout: Seconds to wait in open state before transitioning
            to half-open.
        half_open_max_calls: Maximum concurrent calls in half-open state.
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    reset_timeout: float = 30.0
    half_open_max_calls: int = 1


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Monitors failures and opens the circuit to prevent cascading
    failures when a downstream service is unhealthy.

    Usage::

        breaker = CircuitBreaker("anthropic-api")

        @breaker
        def call_api():
            ...

    Or as a context manager::

        with breaker:
            call_api()

    Args:
        name: Identifier for the circuit (used in logging / errors).
        config: Circuit breaker configuration.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may transition from OPEN → HALF_OPEN)."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.config.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0
                    logger.info(
                        "Circuit '%s' transitioning to HALF_OPEN after %.1fs",
                        self.name,
                        elapsed,
                    )
            return self._state

    @property
    def failure_count(self) -> int:
        """Current failure count."""
        return self._failure_count

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit '%s' CLOSED (recovered)", self.name)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self, exc: Optional[Exception] = None) -> None:
        """Record a failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit '%s' re-OPENED (half-open test failed: %s)",
                    self.name,
                    exc,
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit '%s' OPENED after %d failures",
                    self.name,
                    self._failure_count,
                )

    def allow_request(self) -> bool:
        """Check if a request is allowed through the circuit.

        Returns:
            True if the request should proceed.
        """
        state = self.state  # May trigger OPEN → HALF_OPEN
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
            return False
        # OPEN
        return False

    def reset(self) -> None:
        """Force-reset the circuit to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            logger.info("Circuit '%s' force-reset to CLOSED", self.name)

    def get_stats(self) -> dict[str, Any]:
        """Return circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.config.failure_threshold,
            "reset_timeout": self.config.reset_timeout,
        }

    def __enter__(self):
        if not self.allow_request():
            raise CircuitOpenError(
                self.name, reset_timeout=self.config.reset_timeout
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions

    def __call__(self, func: F) -> F:
        """Use the circuit breaker as a decorator."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        wrapper.circuit_breaker = self  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Combined retry + circuit breaker
# --------------------------------------------------------------------------- #


def retry_with_circuit_breaker(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: Optional[dict[str, Any]] = None,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> Any:
    """Execute *func* with both retry logic and circuit breaker protection.

    The circuit breaker is checked before each attempt. If the circuit
    is open, a :class:`CircuitOpenError` is raised immediately (which
    is non-retryable, so retries stop).

    Args:
        func: The callable to execute.
        args: Positional arguments.
        kwargs: Keyword arguments.
        retry_config: Retry configuration.
        circuit_breaker: Circuit breaker instance.

    Returns:
        The return value of *func* on success.

    Raises:
        CircuitOpenError: If the circuit breaker is open.
        RetryExhaustedError: If all retries are exhausted.
    """
    if kwargs is None:
        kwargs = {}
    if retry_config is None:
        retry_config = RetryConfig()

    def guarded_call(*a: Any, **kw: Any) -> Any:
        if circuit_breaker and not circuit_breaker.allow_request():
            raise CircuitOpenError(
                circuit_breaker.name,
                reset_timeout=circuit_breaker.config.reset_timeout,
            )
        try:
            result = func(*a, **kw)
            if circuit_breaker:
                circuit_breaker.record_success()
            return result
        except Exception as exc:
            if circuit_breaker:
                circuit_breaker.record_failure(exc)
            raise

    return retry_call(guarded_call, args=args, kwargs=kwargs, config=retry_config)


# --------------------------------------------------------------------------- #
# Global circuit breaker registry
# --------------------------------------------------------------------------- #

_breaker_registry: dict[str, CircuitBreaker] = {}
_breaker_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create a named circuit breaker from the global registry.

    Args:
        name: Unique circuit breaker name.
        config: Configuration (only used on first creation).

    Returns:
        The circuit breaker instance.
    """
    with _breaker_lock:
        if name not in _breaker_registry:
            _breaker_registry[name] = CircuitBreaker(name, config)
        return _breaker_registry[name]


def list_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Return all registered circuit breakers."""
    with _breaker_lock:
        return dict(_breaker_registry)


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers in the registry to CLOSED state."""
    with _breaker_lock:
        for breaker in _breaker_registry.values():
            breaker.reset()


def clear_breaker_registry() -> None:
    """Remove all circuit breakers from the global registry."""
    with _breaker_lock:
        _breaker_registry.clear()


# --------------------------------------------------------------------------- #
# Adaptive retry configuration
# --------------------------------------------------------------------------- #


@dataclass
class AdaptiveRetryConfig:
    """Retry configuration that adapts based on observed error patterns.

    Adjusts retry behaviour dynamically:
    - Increases delay when rate-limited (respects Retry-After).
    - Reduces max attempts when a provider is consistently failing.
    - Switches to linear backoff under sustained load.
    - Resets to defaults when the provider recovers.

    The adapter wraps a base :class:`RetryConfig` and produces a
    new config for each request based on current conditions.

    Attributes:
        base_config: The starting retry configuration.
        min_attempts: Minimum number of attempts (never go below this).
        max_attempts_ceiling: Absolute maximum attempts.
        rate_limit_multiplier: Multiply base delay by this when rate-limited.
        error_rate_threshold: Error rate (per second) above which to
            reduce max attempts.
        success_rate_threshold: Success rate below which to reduce retries.
        recovery_success_count: Consecutive successes needed to restore
            original config.
    """

    base_config: RetryConfig = field(default_factory=RetryConfig)
    min_attempts: int = 1
    max_attempts_ceiling: int = 10
    rate_limit_multiplier: float = 2.0
    error_rate_threshold: float = 0.5
    success_rate_threshold: float = 0.7
    recovery_success_count: int = 5

    def __post_init__(self) -> None:
        """Initialize mutable state."""
        self._lock = threading.Lock()
        self._consecutive_successes: int = 0
        self._consecutive_rate_limits: int = 0
        self._current_delay_multiplier: float = 1.0
        self._current_max_attempts: int = self.base_config.max_attempts
        self._current_strategy: BackoffStrategy = self.base_config.backoff_strategy

    def get_config(
        self,
        error_rate: float = 0.0,
        success_rate: float = 1.0,
    ) -> RetryConfig:
        """Produce a RetryConfig adapted to current conditions.

        Args:
            error_rate: Current errors per second for the provider.
            success_rate: Current success rate (0.0–1.0) for the provider.

        Returns:
            An adapted RetryConfig.
        """
        with self._lock:
            max_attempts = self._current_max_attempts
            base_delay = self.base_config.base_delay * self._current_delay_multiplier
            strategy = self._current_strategy

            # Reduce attempts if error rate is high
            if error_rate > self.error_rate_threshold:
                max_attempts = max(self.min_attempts, max_attempts - 1)

            # Reduce attempts if success rate is low
            if success_rate < self.success_rate_threshold:
                max_attempts = max(self.min_attempts, max_attempts - 1)

            # Switch to linear backoff under sustained errors
            if error_rate > self.error_rate_threshold * 2:
                strategy = BackoffStrategy.LINEAR

            return RetryConfig(
                max_attempts=max_attempts,
                backoff_strategy=strategy,
                base_delay=min(base_delay, self.base_config.max_delay),
                max_delay=self.base_config.max_delay,
                exponential_base=self.base_config.exponential_base,
                jitter=self.base_config.jitter,
                jitter_range=self.base_config.jitter_range,
                retry_on=self.base_config.retry_on,
                retry_on_status=self.base_config.retry_on_status,
                total_timeout=self.base_config.total_timeout,
                on_retry=self.base_config.on_retry,
            )

    def record_rate_limit(self) -> None:
        """Record a rate-limit event, increasing delay for future retries."""
        with self._lock:
            self._consecutive_rate_limits += 1
            self._consecutive_successes = 0
            # Increase delay multiplier (caps at 8x)
            self._current_delay_multiplier = min(
                8.0,
                self._current_delay_multiplier * self.rate_limit_multiplier,
            )
            logger.debug(
                "Adaptive retry: rate limit recorded, delay multiplier=%.1f",
                self._current_delay_multiplier,
            )

    def record_overload(self) -> None:
        """Record an overloaded/unavailable event, reducing aggressiveness."""
        with self._lock:
            self._consecutive_successes = 0
            # Reduce max attempts
            self._current_max_attempts = max(
                self.min_attempts,
                self._current_max_attempts - 1,
            )
            # Increase delay
            self._current_delay_multiplier = min(
                8.0,
                self._current_delay_multiplier * 1.5,
            )
            logger.debug(
                "Adaptive retry: overload recorded, max_attempts=%d, delay_mult=%.1f",
                self._current_max_attempts,
                self._current_delay_multiplier,
            )

    def record_success(self) -> None:
        """Record a success, potentially restoring original config."""
        with self._lock:
            self._consecutive_successes += 1
            self._consecutive_rate_limits = 0

            if self._consecutive_successes >= self.recovery_success_count:
                # Gradually restore to defaults
                self._current_delay_multiplier = max(
                    1.0, self._current_delay_multiplier * 0.5
                )
                self._current_max_attempts = min(
                    self.base_config.max_attempts,
                    self._current_max_attempts + 1,
                )
                self._current_strategy = self.base_config.backoff_strategy

                # Full reset if we're back to normal
                if (
                    self._current_delay_multiplier <= 1.0
                    and self._current_max_attempts >= self.base_config.max_attempts
                ):
                    self._current_delay_multiplier = 1.0
                    self._consecutive_successes = 0
                    logger.debug("Adaptive retry: fully recovered to defaults")

    def record_error(self, error: Exception) -> None:
        """Record an error and adapt retry behaviour accordingly.

        Dispatches to the appropriate handler based on error type.

        Args:
            error: The exception that occurred.
        """
        from src.errors import RateLimitError, OverloadedError, ProviderUnavailableError

        if isinstance(error, RateLimitError):
            self.record_rate_limit()
        elif isinstance(error, (OverloadedError, ProviderUnavailableError)):
            self.record_overload()
        else:
            # Generic error: reset success counter
            with self._lock:
                self._consecutive_successes = 0

    def reset(self) -> None:
        """Reset adaptive state to defaults."""
        with self._lock:
            self._consecutive_successes = 0
            self._consecutive_rate_limits = 0
            self._current_delay_multiplier = 1.0
            self._current_max_attempts = self.base_config.max_attempts
            self._current_strategy = self.base_config.backoff_strategy

    def get_state(self) -> dict[str, Any]:
        """Return the current adaptive state for inspection.

        Returns:
            Dictionary with current adaptive parameters.
        """
        with self._lock:
            return {
                "current_max_attempts": self._current_max_attempts,
                "current_delay_multiplier": self._current_delay_multiplier,
                "current_strategy": self._current_strategy.value,
                "consecutive_successes": self._consecutive_successes,
                "consecutive_rate_limits": self._consecutive_rate_limits,
                "base_max_attempts": self.base_config.max_attempts,
                "base_delay": self.base_config.base_delay,
            }


# --------------------------------------------------------------------------- #
# Per-provider retry policies
# --------------------------------------------------------------------------- #


@dataclass
class RetryPolicy:
    """Named retry policy for a specific provider or use case.

    Combines a :class:`RetryConfig` with a descriptive name and optional
    adaptive behaviour.

    Attributes:
        name: Policy identifier (e.g. ``"anthropic-messages"``).
        retry_config: The static retry configuration.
        adaptive: Optional adaptive config that modulates the static config.
        circuit_breaker_config: Optional circuit breaker for this policy.
    """

    name: str
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    adaptive: Optional[AdaptiveRetryConfig] = None
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None

    def get_effective_config(
        self,
        error_rate: float = 0.0,
        success_rate: float = 1.0,
    ) -> RetryConfig:
        """Get the effective retry config, applying adaptive adjustments.

        Args:
            error_rate: Current errors per second.
            success_rate: Current success rate (0.0–1.0).

        Returns:
            The effective RetryConfig.
        """
        if self.adaptive:
            return self.adaptive.get_config(
                error_rate=error_rate,
                success_rate=success_rate,
            )
        return self.retry_config

    def record_outcome(self, error: Optional[Exception] = None) -> None:
        """Record the outcome of a request for adaptive tuning.

        Args:
            error: The exception if the request failed, None for success.
        """
        if self.adaptive:
            if error is None:
                self.adaptive.record_success()
            else:
                self.adaptive.record_error(error)


# --------------------------------------------------------------------------- #
# Retry policy registry
# --------------------------------------------------------------------------- #

_policy_registry: dict[str, RetryPolicy] = {}
_policy_lock = threading.Lock()


def register_retry_policy(policy: RetryPolicy) -> None:
    """Register a named retry policy in the global registry.

    Args:
        policy: The retry policy to register.
    """
    with _policy_lock:
        _policy_registry[policy.name] = policy


def get_retry_policy(name: str) -> Optional[RetryPolicy]:
    """Look up a named retry policy.

    Args:
        name: Policy name.

    Returns:
        The retry policy, or None if not registered.
    """
    with _policy_lock:
        return _policy_registry.get(name)


def list_retry_policies() -> dict[str, RetryPolicy]:
    """Return all registered retry policies.

    Returns:
        Dictionary mapping policy names to RetryPolicy instances.
    """
    with _policy_lock:
        return dict(_policy_registry)


def clear_retry_policies() -> None:
    """Remove all registered retry policies."""
    with _policy_lock:
        _policy_registry.clear()
