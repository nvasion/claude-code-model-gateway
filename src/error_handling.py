"""Centralized error handling middleware for claude-code-model-gateway.

Provides a unified error handling layer that ties together the error
hierarchy, retry logic, circuit breakers, and provider failover into
a cohesive request-processing pipeline.

Key components:

- **ErrorTracker**: System-wide error statistics and rate tracking.
  Monitors error rates per provider, category, and endpoint to
  support operational dashboards and alerting.

- **ProviderFailover**: Automatic failover between configured
  providers when the primary provider is unhealthy.  Uses circuit
  breaker state and error rates to make routing decisions.

- **HealthStatus**: Computes overall system and per-provider health
  from error tracker metrics and circuit breaker states.

- **ErrorRecoveryMiddleware**: Request-level error handling that wraps
  upstream calls with retry, circuit-breaker, failover, and structured
  error responses.

Typical usage::

    from src.error_handling import (
        ErrorTracker,
        ProviderFailover,
        ErrorRecoveryMiddleware,
        get_health_status,
    )

    # Create a global tracker
    tracker = get_error_tracker()

    # Record errors as they occur
    tracker.record_error("anthropic", error)

    # Get system health
    health = get_health_status()

    # Use the middleware for upstream calls
    middleware = ErrorRecoveryMiddleware(tracker=tracker)
    result = middleware.execute(call_provider, provider="anthropic")
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from src.errors import (
    CircuitOpenError,
    ErrorCategory,
    ErrorSeverity,
    GatewayError,
    ProviderError,
    ProviderUnavailableError,
    RetryExhaustedError,
    is_retryable_exception,
)
from src.retry import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
    RetryStats,
    get_circuit_breaker,
    retry_call,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Health status
# --------------------------------------------------------------------------- #


class HealthState(str, Enum):
    """Overall health state of a component or the system."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ProviderHealth:
    """Health status for a single provider.

    Attributes:
        name: Provider identifier.
        state: Current health state.
        circuit_state: Circuit breaker state (if applicable).
        error_rate: Errors per second over the tracking window.
        total_errors: Total errors recorded in the tracking window.
        total_requests: Total requests recorded in the tracking window.
        success_rate: Success rate as a float 0.0–1.0.
        avg_latency_ms: Average response latency in milliseconds.
        last_error_time: Timestamp of the most recent error (or 0).
        last_error_message: Message from the most recent error.
        consecutive_failures: Count of consecutive failures.
    """

    name: str
    state: HealthState = HealthState.HEALTHY
    circuit_state: Optional[str] = None
    error_rate: float = 0.0
    total_errors: int = 0
    total_requests: int = 0
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
    last_error_time: float = 0.0
    last_error_message: str = ""
    consecutive_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON responses."""
        d: dict[str, Any] = {
            "name": self.name,
            "state": self.state.value,
            "error_rate": round(self.error_rate, 4),
            "total_errors": self.total_errors,
            "total_requests": self.total_requests,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "consecutive_failures": self.consecutive_failures,
        }
        if self.circuit_state is not None:
            d["circuit_state"] = self.circuit_state
        if self.last_error_time > 0:
            d["last_error_time"] = self.last_error_time
        if self.last_error_message:
            d["last_error_message"] = self.last_error_message
        return d


@dataclass
class SystemHealth:
    """Overall system health status.

    Attributes:
        state: Aggregate health state.
        providers: Per-provider health details.
        total_errors: Total errors across all providers.
        total_requests: Total requests across all providers.
        uptime_seconds: Time since the tracker was initialized.
        error_rate: System-wide errors per second.
    """

    state: HealthState = HealthState.HEALTHY
    providers: dict[str, ProviderHealth] = field(default_factory=dict)
    total_errors: int = 0
    total_requests: int = 0
    uptime_seconds: float = 0.0
    error_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON responses."""
        return {
            "state": self.state.value,
            "total_errors": self.total_errors,
            "total_requests": self.total_requests,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "error_rate": round(self.error_rate, 4),
            "providers": {
                name: ph.to_dict() for name, ph in self.providers.items()
            },
        }


# --------------------------------------------------------------------------- #
# Error event
# --------------------------------------------------------------------------- #


@dataclass
class ErrorEvent:
    """A single recorded error event.

    Attributes:
        timestamp: When the error occurred (monotonic).
        wall_time: Wall-clock time for display purposes.
        provider: Provider that produced the error.
        category: Error category.
        severity: Error severity.
        status_code: HTTP status code (if applicable).
        message: Short error description.
        retryable: Whether the error was retryable.
        latency_ms: Request latency in milliseconds.
    """

    timestamp: float
    wall_time: float
    provider: str
    category: ErrorCategory
    severity: ErrorSeverity
    status_code: Optional[int] = None
    message: str = ""
    retryable: bool = False
    latency_ms: float = 0.0


@dataclass
class RequestEvent:
    """A single recorded request event (success or failure).

    Attributes:
        timestamp: When the request completed (monotonic).
        provider: Provider that handled the request.
        success: Whether the request succeeded.
        latency_ms: Request latency in milliseconds.
        status_code: HTTP status code.
    """

    timestamp: float
    provider: str
    success: bool
    latency_ms: float = 0.0
    status_code: Optional[int] = None


# --------------------------------------------------------------------------- #
# Error tracker
# --------------------------------------------------------------------------- #


class ErrorTracker:
    """System-wide error statistics and rate tracking.

    Maintains a sliding window of error and request events per provider
    to compute real-time error rates, success rates, and latency stats.

    The tracker is thread-safe and designed for concurrent access from
    multiple request handler threads.

    Args:
        window_seconds: Size of the sliding window for rate calculations.
        max_events: Maximum number of events to retain per provider.

    Example::

        tracker = ErrorTracker(window_seconds=300)
        tracker.record_error("anthropic", some_error, latency_ms=150.0)
        tracker.record_success("anthropic", latency_ms=42.0)
        health = tracker.get_provider_health("anthropic")
    """

    def __init__(
        self,
        window_seconds: float = 300.0,
        max_events: int = 10000,
    ) -> None:
        self.window_seconds = window_seconds
        self.max_events = max_events
        self._start_time = time.monotonic()

        self._lock = threading.Lock()
        # Per-provider error events
        self._error_events: dict[str, deque[ErrorEvent]] = defaultdict(
            lambda: deque(maxlen=max_events)
        )
        # Per-provider request events (both success and failure)
        self._request_events: dict[str, deque[RequestEvent]] = defaultdict(
            lambda: deque(maxlen=max_events)
        )
        # Per-provider consecutive failure counters
        self._consecutive_failures: dict[str, int] = defaultdict(int)
        # Per-provider last error details
        self._last_errors: dict[str, ErrorEvent] = {}
        # Category-level aggregation
        self._category_counts: dict[ErrorCategory, int] = defaultdict(int)

    def record_error(
        self,
        provider: str,
        error: Exception,
        latency_ms: float = 0.0,
    ) -> None:
        """Record an error event.

        Args:
            provider: The provider that produced the error.
            error: The exception that occurred.
            latency_ms: Request latency in milliseconds.
        """
        now = time.monotonic()
        wall = time.time()

        # Extract metadata from GatewayError subclasses
        if isinstance(error, GatewayError):
            category = error.context.category
            severity = error.context.severity
            status_code = error.context.status_code
            retryable = error.is_retryable
        else:
            category = ErrorCategory.INTERNAL
            severity = ErrorSeverity.MEDIUM
            status_code = None
            retryable = is_retryable_exception(error)

        event = ErrorEvent(
            timestamp=now,
            wall_time=wall,
            provider=provider,
            category=category,
            severity=severity,
            status_code=status_code,
            message=str(error)[:500],
            retryable=retryable,
            latency_ms=latency_ms,
        )

        request_event = RequestEvent(
            timestamp=now,
            provider=provider,
            success=False,
            latency_ms=latency_ms,
            status_code=status_code,
        )

        with self._lock:
            self._error_events[provider].append(event)
            self._request_events[provider].append(request_event)
            self._consecutive_failures[provider] += 1
            self._last_errors[provider] = event
            self._category_counts[category] += 1

        logger.debug(
            "Error recorded for provider '%s': %s (category=%s, severity=%s)",
            provider,
            str(error)[:100],
            category.value,
            severity.value,
        )

    def record_success(
        self,
        provider: str,
        latency_ms: float = 0.0,
        status_code: Optional[int] = None,
    ) -> None:
        """Record a successful request.

        Args:
            provider: The provider that handled the request.
            latency_ms: Request latency in milliseconds.
            status_code: HTTP status code.
        """
        now = time.monotonic()
        event = RequestEvent(
            timestamp=now,
            provider=provider,
            success=True,
            latency_ms=latency_ms,
            status_code=status_code,
        )

        with self._lock:
            self._request_events[provider].append(event)
            self._consecutive_failures[provider] = 0

    def get_provider_health(self, provider: str) -> ProviderHealth:
        """Compute the health status for a single provider.

        Args:
            provider: The provider name.

        Returns:
            ProviderHealth with current metrics.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            # Count errors in window
            errors_in_window = [
                e for e in self._error_events.get(provider, [])
                if e.timestamp >= cutoff
            ]
            # Count requests in window
            requests_in_window = [
                r for r in self._request_events.get(provider, [])
                if r.timestamp >= cutoff
            ]

            total_errors = len(errors_in_window)
            total_requests = len(requests_in_window)
            consecutive = self._consecutive_failures.get(provider, 0)
            last_error = self._last_errors.get(provider)

        # Compute rates
        if self.window_seconds > 0:
            error_rate = total_errors / self.window_seconds
        else:
            error_rate = 0.0

        if total_requests > 0:
            success_count = sum(1 for r in requests_in_window if r.success)
            success_rate = success_count / total_requests
        else:
            success_rate = 1.0

        # Compute average latency
        latencies = [r.latency_ms for r in requests_in_window if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Determine health state
        state = self._compute_health_state(
            error_rate=error_rate,
            success_rate=success_rate,
            consecutive_failures=consecutive,
        )

        # Get circuit breaker state if available
        circuit_state = None
        try:
            from src.retry import list_circuit_breakers
            breakers = list_circuit_breakers()
            if provider in breakers:
                circuit_state = breakers[provider].state.value
        except Exception:
            pass

        return ProviderHealth(
            name=provider,
            state=state,
            circuit_state=circuit_state,
            error_rate=error_rate,
            total_errors=total_errors,
            total_requests=total_requests,
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            last_error_time=last_error.wall_time if last_error else 0.0,
            last_error_message=last_error.message if last_error else "",
            consecutive_failures=consecutive,
        )

    def get_system_health(self) -> SystemHealth:
        """Compute the overall system health.

        Returns:
            SystemHealth with aggregate metrics and per-provider details.
        """
        now = time.monotonic()
        uptime = now - self._start_time
        cutoff = now - self.window_seconds

        with self._lock:
            providers = set(self._error_events.keys()) | set(
                self._request_events.keys()
            )

        provider_health = {}
        total_errors = 0
        total_requests = 0

        for name in providers:
            ph = self.get_provider_health(name)
            provider_health[name] = ph
            total_errors += ph.total_errors
            total_requests += ph.total_requests

        # System-wide error rate
        if self.window_seconds > 0:
            error_rate = total_errors / self.window_seconds
        else:
            error_rate = 0.0

        # Aggregate health state
        if not provider_health:
            system_state = HealthState.HEALTHY
        elif all(
            ph.state == HealthState.UNHEALTHY for ph in provider_health.values()
        ):
            system_state = HealthState.UNHEALTHY
        elif any(
            ph.state in (HealthState.UNHEALTHY, HealthState.DEGRADED)
            for ph in provider_health.values()
        ):
            system_state = HealthState.DEGRADED
        else:
            system_state = HealthState.HEALTHY

        return SystemHealth(
            state=system_state,
            providers=provider_health,
            total_errors=total_errors,
            total_requests=total_requests,
            uptime_seconds=uptime,
            error_rate=error_rate,
        )

    def get_error_counts_by_category(self) -> dict[str, int]:
        """Get error counts grouped by category.

        Returns:
            Dictionary mapping category names to counts.
        """
        with self._lock:
            return {
                cat.value: count
                for cat, count in self._category_counts.items()
            }

    def get_recent_errors(
        self,
        provider: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get recent error events.

        Args:
            provider: Filter by provider (None for all providers).
            limit: Maximum number of errors to return.

        Returns:
            List of error event dictionaries, most recent first.
        """
        with self._lock:
            if provider:
                events = list(self._error_events.get(provider, []))
            else:
                events = []
                for provider_events in self._error_events.values():
                    events.extend(provider_events)

        # Sort by timestamp descending
        events.sort(key=lambda e: e.timestamp, reverse=True)

        return [
            {
                "wall_time": e.wall_time,
                "provider": e.provider,
                "category": e.category.value,
                "severity": e.severity.value,
                "status_code": e.status_code,
                "message": e.message,
                "retryable": e.retryable,
                "latency_ms": round(e.latency_ms, 2),
            }
            for e in events[:limit]
        ]

    def reset(self) -> None:
        """Reset all tracked statistics."""
        with self._lock:
            self._error_events.clear()
            self._request_events.clear()
            self._consecutive_failures.clear()
            self._last_errors.clear()
            self._category_counts.clear()
            self._start_time = time.monotonic()

    @staticmethod
    def _compute_health_state(
        error_rate: float,
        success_rate: float,
        consecutive_failures: int,
    ) -> HealthState:
        """Determine health state from metrics.

        Args:
            error_rate: Errors per second.
            success_rate: Success ratio 0.0–1.0.
            consecutive_failures: Number of consecutive failures.

        Returns:
            The computed HealthState.
        """
        # Critical thresholds
        if consecutive_failures >= 10 or success_rate < 0.5:
            return HealthState.UNHEALTHY

        # Warning thresholds
        if consecutive_failures >= 3 or success_rate < 0.9 or error_rate > 1.0:
            return HealthState.DEGRADED

        return HealthState.HEALTHY


# --------------------------------------------------------------------------- #
# Provider failover
# --------------------------------------------------------------------------- #


@dataclass
class FailoverConfig:
    """Configuration for provider failover behaviour.

    Attributes:
        enabled: Whether failover is active.
        max_failover_attempts: Maximum providers to try before giving up.
        failover_on_status: HTTP status codes that trigger failover.
        failover_on_categories: Error categories that trigger failover.
        prefer_healthy: Whether to prefer providers with HEALTHY state.
        cooldown_seconds: Minimum time before retrying a failed provider.
    """

    enabled: bool = True
    max_failover_attempts: int = 3
    failover_on_status: set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504, 529}
    )
    failover_on_categories: set[ErrorCategory] = field(
        default_factory=lambda: {
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.PROVIDER,
            ErrorCategory.RATE_LIMIT,
        }
    )
    prefer_healthy: bool = True
    cooldown_seconds: float = 30.0


class ProviderFailover:
    """Automatic failover between configured providers.

    When the primary provider fails with a retryable error, the failover
    mechanism routes the request to an alternative provider based on
    health status and circuit breaker state.

    Args:
        providers: Ordered list of provider names (first = preferred).
        tracker: ErrorTracker for health-based routing decisions.
        config: Failover configuration.

    Example::

        failover = ProviderFailover(
            providers=["anthropic", "openai"],
            tracker=tracker,
        )

        # Get the next provider to try
        for provider in failover.get_providers():
            try:
                result = call_provider(provider)
                failover.record_success(provider)
                break
            except Exception as e:
                failover.record_failure(provider, e)
    """

    def __init__(
        self,
        providers: list[str],
        tracker: Optional[ErrorTracker] = None,
        config: Optional[FailoverConfig] = None,
    ) -> None:
        self.providers = list(providers)
        self.tracker = tracker
        self.config = config or FailoverConfig()

        self._lock = threading.Lock()
        self._last_failure: dict[str, float] = {}

    def should_failover(self, error: Exception) -> bool:
        """Determine whether an error should trigger failover.

        Args:
            error: The exception that occurred.

        Returns:
            True if failover should be attempted.
        """
        if not self.config.enabled:
            return False

        if isinstance(error, GatewayError):
            # Check status code
            if (
                error.context.status_code is not None
                and error.context.status_code in self.config.failover_on_status
            ):
                return True
            # Check category
            if error.context.category in self.config.failover_on_categories:
                return True

        # Check circuit open
        if isinstance(error, CircuitOpenError):
            return True

        # Check retry exhausted (the underlying errors triggered retries)
        if isinstance(error, RetryExhaustedError):
            return True

        # Fallback: any retryable exception
        return is_retryable_exception(error)

    def get_providers(
        self,
        exclude: Optional[set[str]] = None,
    ) -> list[str]:
        """Get an ordered list of providers to try.

        Providers are ordered by preference:
        1. Healthy providers first (if prefer_healthy is enabled)
        2. Degraded providers next
        3. Unhealthy providers last
        4. Within each tier, original order is preserved
        5. Providers in cooldown are deprioritized

        Args:
            exclude: Provider names to exclude from the list.

        Returns:
            Ordered list of provider names to try.
        """
        exclude = exclude or set()
        now = time.monotonic()
        candidates = [p for p in self.providers if p not in exclude]

        if not self.config.prefer_healthy or self.tracker is None:
            return candidates[: self.config.max_failover_attempts]

        # Sort by health tier, respecting original order within tiers
        tier_order = {
            HealthState.HEALTHY: 0,
            HealthState.DEGRADED: 1,
            HealthState.UNHEALTHY: 2,
        }

        def sort_key(provider: str) -> tuple[int, bool, int]:
            health = self.tracker.get_provider_health(provider)
            tier = tier_order.get(health.state, 3)
            # Deprioritize providers in cooldown
            with self._lock:
                last_fail = self._last_failure.get(provider, 0.0)
            in_cooldown = (now - last_fail) < self.config.cooldown_seconds
            # Use original index for stable ordering within tier
            idx = self.providers.index(provider) if provider in self.providers else 999
            return (tier, in_cooldown, idx)

        candidates.sort(key=sort_key)
        return candidates[: self.config.max_failover_attempts]

    def record_success(self, provider: str) -> None:
        """Record a successful request to a provider.

        Args:
            provider: The provider that succeeded.
        """
        if self.tracker:
            self.tracker.record_success(provider)

    def record_failure(
        self,
        provider: str,
        error: Exception,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a failed request to a provider.

        Args:
            provider: The provider that failed.
            error: The exception that occurred.
            latency_ms: Request latency in milliseconds.
        """
        with self._lock:
            self._last_failure[provider] = time.monotonic()

        if self.tracker:
            self.tracker.record_error(provider, error, latency_ms=latency_ms)


# --------------------------------------------------------------------------- #
# Error recovery middleware
# --------------------------------------------------------------------------- #


@dataclass
class RecoveryConfig:
    """Configuration for the error recovery middleware.

    Attributes:
        retry_config: Base retry configuration for individual attempts.
        circuit_breaker_config: Circuit breaker settings per provider.
        failover_config: Provider failover settings.
        use_circuit_breaker: Whether to enable circuit breakers.
        use_failover: Whether to enable provider failover.
        record_metrics: Whether to track error metrics.
        on_error: Optional callback invoked on each error
            (provider, error, attempt).
        on_failover: Optional callback invoked on provider failover
            (from_provider, to_provider, error).
    """

    retry_config: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker_config: CircuitBreakerConfig = field(
        default_factory=CircuitBreakerConfig
    )
    failover_config: FailoverConfig = field(default_factory=FailoverConfig)
    use_circuit_breaker: bool = True
    use_failover: bool = True
    record_metrics: bool = True
    on_error: Optional[Callable[[str, Exception, int], None]] = None
    on_failover: Optional[Callable[[str, str, Exception], None]] = None


class ErrorRecoveryMiddleware:
    """Request-level error handling with retry, circuit-breaker, and failover.

    Wraps upstream provider calls with a layered error recovery pipeline:

    1. **Circuit breaker check** — fail fast if the provider is known-bad.
    2. **Retry with backoff** — retry transient failures with exponential backoff.
    3. **Error recording** — track errors for health monitoring.
    4. **Provider failover** — route to an alternative provider if the
       primary fails.

    Args:
        tracker: ErrorTracker instance for metrics.
        providers: List of available provider names.
        config: Recovery configuration.

    Example::

        middleware = ErrorRecoveryMiddleware(
            tracker=tracker,
            providers=["anthropic", "openai"],
        )

        def call_provider(provider: str) -> dict:
            # Make the actual API call
            ...

        result = middleware.execute(call_provider, provider="anthropic")
    """

    def __init__(
        self,
        tracker: Optional[ErrorTracker] = None,
        providers: Optional[list[str]] = None,
        config: Optional[RecoveryConfig] = None,
    ) -> None:
        self.tracker = tracker or ErrorTracker()
        self.config = config or RecoveryConfig()
        self.failover = (
            ProviderFailover(
                providers=providers or [],
                tracker=self.tracker,
                config=self.config.failover_config,
            )
            if providers
            else None
        )

    def _get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider.

        Args:
            provider: Provider name.

        Returns:
            CircuitBreaker instance.
        """
        return get_circuit_breaker(
            provider, config=self.config.circuit_breaker_config
        )

    def execute(
        self,
        func: Callable[..., T],
        *,
        provider: str,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> T:
        """Execute a function with full error recovery pipeline.

        Attempts the call with the specified provider first, then fails
        over to alternative providers if configured and necessary.

        Args:
            func: Callable to execute. Must accept a ``provider``
                keyword argument (or it will be passed via kwargs).
            provider: Primary provider name.
            args: Positional arguments for *func*.
            kwargs: Additional keyword arguments for *func*.
            retry_config: Override retry config for this call.

        Returns:
            The return value of *func* on success.

        Raises:
            GatewayError: If all providers and retries are exhausted.
        """
        if kwargs is None:
            kwargs = {}
        retry_cfg = retry_config or self.config.retry_config

        # Build ordered list of providers to try
        if self.config.use_failover and self.failover:
            providers_to_try = self.failover.get_providers()
            # Ensure the requested provider is first if not already
            if provider in providers_to_try:
                providers_to_try.remove(provider)
            providers_to_try.insert(0, provider)
        else:
            providers_to_try = [provider]

        last_error: Optional[Exception] = None
        errors: list[Exception] = []

        for idx, current_provider in enumerate(providers_to_try):
            if idx > 0:
                # This is a failover attempt
                logger.warning(
                    "Failing over from '%s' to '%s' (attempt %d/%d)",
                    provider if idx == 1 else providers_to_try[idx - 1],
                    current_provider,
                    idx + 1,
                    len(providers_to_try),
                )
                if self.config.on_failover and last_error:
                    try:
                        self.config.on_failover(
                            providers_to_try[idx - 1],
                            current_provider,
                            last_error,
                        )
                    except Exception:
                        pass

            try:
                result = self._execute_with_provider(
                    func=func,
                    provider=current_provider,
                    args=args,
                    kwargs=kwargs,
                    retry_config=retry_cfg,
                )
                return result

            except CircuitOpenError as exc:
                last_error = exc
                errors.append(exc)
                logger.warning(
                    "Circuit open for provider '%s', trying next",
                    current_provider,
                )
                continue

            except (RetryExhaustedError, GatewayError) as exc:
                last_error = exc
                errors.append(exc)

                # Check if failover is warranted
                if (
                    self.failover
                    and self.config.use_failover
                    and self.failover.should_failover(exc)
                    and idx < len(providers_to_try) - 1
                ):
                    continue

                # No more providers to try
                raise

            except Exception as exc:
                last_error = exc
                errors.append(exc)

                if (
                    self.failover
                    and self.config.use_failover
                    and idx < len(providers_to_try) - 1
                ):
                    continue

                raise

        # All providers exhausted
        if last_error:
            raise last_error
        raise ProviderUnavailableError(provider=provider)

    def _execute_with_provider(
        self,
        func: Callable[..., T],
        provider: str,
        args: tuple,
        kwargs: dict[str, Any],
        retry_config: RetryConfig,
    ) -> T:
        """Execute a function against a single provider with retry + circuit breaker.

        Args:
            func: The callable to execute.
            provider: Provider name.
            args: Positional arguments.
            kwargs: Keyword arguments.
            retry_config: Retry configuration.

        Returns:
            The return value of *func*.
        """
        cb = self._get_circuit_breaker(provider) if self.config.use_circuit_breaker else None

        attempt_count = 0

        def guarded_call(*a: Any, **kw: Any) -> T:
            nonlocal attempt_count
            attempt_count += 1
            start = time.monotonic()

            # Circuit breaker check
            if cb and not cb.allow_request():
                raise CircuitOpenError(
                    provider, reset_timeout=cb.config.reset_timeout
                )

            try:
                result = func(*a, **kw)

                # Record success
                latency_ms = (time.monotonic() - start) * 1000.0
                if cb:
                    cb.record_success()
                if self.config.record_metrics:
                    self.tracker.record_success(provider, latency_ms=latency_ms)

                return result

            except Exception as exc:
                latency_ms = (time.monotonic() - start) * 1000.0

                # Record failure
                if cb:
                    cb.record_failure(exc)
                if self.config.record_metrics:
                    self.tracker.record_error(
                        provider, exc, latency_ms=latency_ms
                    )

                # Invoke error callback
                if self.config.on_error:
                    try:
                        self.config.on_error(provider, exc, attempt_count)
                    except Exception:
                        pass

                raise

        return retry_call(
            guarded_call,
            args=args,
            kwargs=kwargs,
            config=retry_config,
        )


# --------------------------------------------------------------------------- #
# Global tracker singleton
# --------------------------------------------------------------------------- #

_global_tracker: Optional[ErrorTracker] = None
_tracker_lock = threading.Lock()


def get_error_tracker(
    window_seconds: float = 300.0,
    max_events: int = 10000,
) -> ErrorTracker:
    """Get or create the global error tracker singleton.

    Args:
        window_seconds: Sliding window size (only used on first call).
        max_events: Maximum events per provider (only used on first call).

    Returns:
        The global ErrorTracker instance.
    """
    global _global_tracker
    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = ErrorTracker(
                window_seconds=window_seconds,
                max_events=max_events,
            )
        return _global_tracker


def reset_error_tracker() -> None:
    """Reset the global error tracker (primarily for testing)."""
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None


def get_health_status() -> SystemHealth:
    """Get the current system health status.

    Convenience function that reads from the global error tracker.

    Returns:
        SystemHealth with current metrics.
    """
    tracker = get_error_tracker()
    return tracker.get_system_health()
