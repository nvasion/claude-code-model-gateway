"""Provider-specific error recovery strategies for claude-code-model-gateway.

Defines a composable *strategy* interface for reacting to specific error
types *before* (or instead of) retrying.  Strategies sit on top of the
existing retry / circuit-breaker infrastructure and handle concerns like:

- Refreshing stale credentials before the next attempt.
- Respecting deadline budgets across retry loops.
- Emitting structured audit events on authentication failures.
- Applying provider-specific back-off hints (e.g. ``Retry-After``).
- Aggregating errors from multiple attempts for richer diagnostics.

Key components
--------------
- **RecoveryAction** — enumeration of possible strategy outcomes.
- **RecoveryContext** — structured state passed through the strategy chain.
- **BaseRecoveryStrategy** — abstract base with a single ``recover()`` hook.
- Concrete strategies: :class:`RateLimitRecoveryStrategy`,
  :class:`AuthRefreshStrategy`, :class:`DeadlineAwareStrategy`,
  :class:`CircuitBreakerResetStrategy`, :class:`ProviderRotationStrategy`,
  :class:`ErrorAggregationStrategy`.
- **RecoveryChain** — runs strategies in order, stopping at the first
  ``ABORT`` or ``SKIP_RETRY`` decision.

Typical usage::

    from src.error_recovery_strategies import (
        RecoveryChain,
        RateLimitRecoveryStrategy,
        DeadlineAwareStrategy,
        ErrorAggregationStrategy,
        RecoveryContext,
    )

    chain = RecoveryChain([
        DeadlineAwareStrategy(deadline_seconds=30.0),
        RateLimitRecoveryStrategy(),
        ErrorAggregationStrategy(max_errors=10),
    ])

    ctx = RecoveryContext(provider="anthropic", start_time=time.monotonic())
    action = chain.recover(error, ctx)
    if action == RecoveryAction.ABORT:
        raise error
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from src.errors import (
    AuthenticationError,
    CircuitOpenError,
    ErrorCategory,
    GatewayError,
    RateLimitError,
    RetryExhaustedError,
    is_retryable_exception,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# RecoveryAction — outcome of a strategy
# --------------------------------------------------------------------------- #


class RecoveryAction(str, Enum):
    """The decision produced by a :class:`BaseRecoveryStrategy`.

    Attributes:
        RETRY: Continue with the next retry attempt.
        ABORT: Stop retrying and propagate the error immediately.
        SKIP_RETRY: Skip this particular retry attempt (treat as consumed)
            but allow the loop to continue if attempts remain.
        WAIT_AND_RETRY: Sleep for the duration specified in the context
            before the next attempt.
    """

    RETRY = "retry"
    ABORT = "abort"
    SKIP_RETRY = "skip_retry"
    WAIT_AND_RETRY = "wait_and_retry"


# --------------------------------------------------------------------------- #
# RecoveryContext — shared mutable state for the strategy chain
# --------------------------------------------------------------------------- #


@dataclass
class RecoveryContext:
    """Mutable context threaded through the recovery strategy chain.

    Strategies read this context to make decisions and write to it to
    communicate with the retry loop and downstream strategies.

    Attributes:
        provider: Name of the provider that produced the error.
        start_time: Monotonic timestamp when the overall request began.
        attempt: Current attempt number (1-based).
        max_attempts: Maximum number of attempts allowed.
        deadline_seconds: Optional absolute deadline (monotonic time) by
            which all attempts must complete.  A value of 0 means no
            deadline.
        extra_delay: Additional delay (seconds) to apply before the next
            retry, beyond the computed backoff.  Strategies can set this
            to honour ``Retry-After`` hints.
        errors: All exceptions encountered so far (most-recent last).
        metadata: Arbitrary key/value store for strategy communication.
        aborted: Whether a strategy has decided to abort.
        abort_reason: Human-readable abort reason.
        credentials_refreshed: Whether credentials were refreshed this
            cycle (prevents duplicate refreshes).
    """

    provider: str = ""
    start_time: float = field(default_factory=time.monotonic)
    attempt: int = 0
    max_attempts: int = 3
    deadline_seconds: float = 0.0   # 0 = no deadline
    extra_delay: float = 0.0
    errors: list[Exception] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str = ""
    credentials_refreshed: bool = False

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since *start_time*."""
        return time.monotonic() - self.start_time

    @property
    def deadline_remaining(self) -> float:
        """Seconds until the deadline, or *float('inf')* if no deadline."""
        if self.deadline_seconds <= 0:
            return float("inf")
        remaining = self.deadline_seconds - time.monotonic()
        return max(0.0, remaining)

    @property
    def is_past_deadline(self) -> bool:
        """True if the deadline has been exceeded."""
        return self.deadline_seconds > 0 and time.monotonic() >= self.deadline_seconds

    def record_error(self, error: Exception) -> None:
        """Append *error* to the error list and update attempt counter."""
        self.errors.append(error)
        self.attempt = len(self.errors)

    def abort(self, reason: str) -> None:
        """Signal that retries should stop."""
        self.aborted = True
        self.abort_reason = reason


# --------------------------------------------------------------------------- #
# BaseRecoveryStrategy — abstract interface
# --------------------------------------------------------------------------- #


class BaseRecoveryStrategy(ABC):
    """Abstract base class for recovery strategies.

    Implementations react to a specific error type or condition and
    return a :class:`RecoveryAction` that directs the retry loop.

    Subclasses should:
    1. Override :meth:`applies_to` to declare which errors they handle.
    2. Override :meth:`recover` to implement the recovery logic.
    """

    @property
    def name(self) -> str:
        """Human-readable strategy name."""
        return type(self).__name__

    def applies_to(self, error: Exception, ctx: RecoveryContext) -> bool:
        """Return *True* if this strategy should handle *error*.

        The default implementation matches all exceptions.  Override to
        narrow the scope.

        Args:
            error: The exception that triggered recovery.
            ctx: Current recovery context.

        Returns:
            True if this strategy should be invoked.
        """
        return True

    @abstractmethod
    def recover(
        self,
        error: Exception,
        ctx: RecoveryContext,
    ) -> RecoveryAction:
        """Execute the recovery action for *error*.

        Args:
            error: The exception to recover from.
            ctx: Mutable recovery context shared across the chain.

        Returns:
            A :class:`RecoveryAction` directing the retry loop.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# --------------------------------------------------------------------------- #
# Concrete strategies
# --------------------------------------------------------------------------- #


class RateLimitRecoveryStrategy(BaseRecoveryStrategy):
    """Handle rate-limit errors by honouring the Retry-After hint.

    When the upstream returns a 429 / rate-limit response the strategy:
    1. Reads ``context.retry_after`` from the error.
    2. Stores the hint in ``ctx.extra_delay`` so the retry loop sleeps
       for at least that long.
    3. Optionally caps the hint at *max_wait_seconds* to avoid extremely
       long waits.

    If rate limits have been hit more than *abort_after_consecutive*
    times in a row, the strategy aborts to prevent livelock.

    Args:
        max_wait_seconds: Cap the Retry-After hint at this value.
        abort_after_consecutive: Abort after this many consecutive
            rate-limit errors.
    """

    def __init__(
        self,
        max_wait_seconds: float = 120.0,
        abort_after_consecutive: int = 5,
    ) -> None:
        self.max_wait_seconds = max_wait_seconds
        self.abort_after_consecutive = abort_after_consecutive

    def applies_to(self, error: Exception, ctx: RecoveryContext) -> bool:
        """Match rate-limit errors."""
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, GatewayError):
            return error.context.category == ErrorCategory.RATE_LIMIT
        return False

    def recover(
        self,
        error: Exception,
        ctx: RecoveryContext,
    ) -> RecoveryAction:
        # Count consecutive rate-limits
        consecutive = sum(
            1 for e in reversed(ctx.errors)
            if isinstance(e, GatewayError)
            and e.context.category == ErrorCategory.RATE_LIMIT
        )
        if consecutive >= self.abort_after_consecutive:
            ctx.abort(
                f"Aborted after {consecutive} consecutive rate-limit errors "
                f"from provider '{ctx.provider}'"
            )
            logger.error(
                "Rate-limit recovery: aborting after %d consecutive rate limits "
                "(provider=%s)",
                consecutive,
                ctx.provider,
            )
            return RecoveryAction.ABORT

        # Extract Retry-After hint
        retry_after: float = 0.0
        if isinstance(error, GatewayError) and error.context.retry_after:
            retry_after = error.context.retry_after

        if retry_after > 0:
            wait = min(retry_after, self.max_wait_seconds)
            ctx.extra_delay = max(ctx.extra_delay, wait)
            logger.info(
                "Rate-limit recovery: will wait %.1fs before retry (provider=%s)",
                wait,
                ctx.provider,
            )
            return RecoveryAction.WAIT_AND_RETRY

        logger.debug(
            "Rate-limit recovery: no Retry-After hint, using standard backoff "
            "(provider=%s)",
            ctx.provider,
        )
        return RecoveryAction.RETRY


class AuthRefreshStrategy(BaseRecoveryStrategy):
    """Attempt credential refresh on authentication failures.

    When an :class:`~src.errors.AuthenticationError` is encountered the
    strategy invokes the user-supplied *refresh_credentials* callback.
    If the refresh succeeds the retry is allowed to proceed; if it fails
    (or if it has already been attempted this cycle) the strategy aborts.

    Args:
        refresh_credentials: Callable that refreshes credentials for the
            given provider.  Should raise on failure.  Signature:
            ``(provider: str) -> None``.
        max_refresh_attempts: Maximum number of credential refreshes to
            attempt per request lifecycle.
    """

    def __init__(
        self,
        refresh_credentials: Optional[Callable[[str], None]] = None,
        max_refresh_attempts: int = 1,
    ) -> None:
        self._refresh = refresh_credentials
        self.max_refresh_attempts = max_refresh_attempts

    def applies_to(self, error: Exception, ctx: RecoveryContext) -> bool:
        """Match authentication errors."""
        return isinstance(error, AuthenticationError)

    def recover(
        self,
        error: Exception,
        ctx: RecoveryContext,
    ) -> RecoveryAction:
        # Auth errors are normally not retryable — only allow refresh once
        refresh_count = ctx.metadata.get("auth_refresh_count", 0)

        if self._refresh is None:
            logger.warning(
                "Authentication error from '%s': no refresh callback configured, "
                "aborting",
                ctx.provider,
            )
            ctx.abort("No credential refresh callback configured")
            return RecoveryAction.ABORT

        if refresh_count >= self.max_refresh_attempts:
            logger.error(
                "Authentication error from '%s': already refreshed credentials "
                "%d time(s), aborting",
                ctx.provider,
                refresh_count,
            )
            ctx.abort(
                f"Authentication failed after {refresh_count} credential refresh(es)"
            )
            return RecoveryAction.ABORT

        try:
            logger.info(
                "Authentication error from '%s': attempting credential refresh "
                "(attempt %d/%d)",
                ctx.provider,
                refresh_count + 1,
                self.max_refresh_attempts,
            )
            self._refresh(ctx.provider)
            ctx.metadata["auth_refresh_count"] = refresh_count + 1
            ctx.credentials_refreshed = True
            logger.info(
                "Credential refresh successful for '%s', will retry",
                ctx.provider,
            )
            return RecoveryAction.RETRY

        except Exception as exc:
            logger.error(
                "Credential refresh for '%s' failed: %s, aborting",
                ctx.provider,
                exc,
            )
            ctx.abort(f"Credential refresh failed: {exc}")
            return RecoveryAction.ABORT


class DeadlineAwareStrategy(BaseRecoveryStrategy):
    """Abort retries when the request deadline has been exceeded.

    Checks both the deadline stored in *ctx* and an optional hard
    ``deadline_seconds`` provided at construction time.  If either is
    exceeded the strategy aborts immediately to avoid wasting time on
    doomed retries.

    Args:
        deadline_seconds: Optional absolute deadline (monotonic seconds).
            If 0, only the context deadline is checked.
        safety_margin_seconds: Abort if remaining time is less than this
            many seconds (to account for scheduling latency).
    """

    def __init__(
        self,
        deadline_seconds: float = 0.0,
        safety_margin_seconds: float = 0.1,
    ) -> None:
        self.deadline_seconds = deadline_seconds
        self.safety_margin_seconds = safety_margin_seconds

    def applies_to(self, error: Exception, ctx: RecoveryContext) -> bool:
        """Apply to all errors (deadline check is universal)."""
        return True

    def recover(
        self,
        error: Exception,
        ctx: RecoveryContext,
    ) -> RecoveryAction:
        now = time.monotonic()

        # Check strategy-level deadline
        if self.deadline_seconds > 0:
            remaining = self.deadline_seconds - now
            if remaining <= self.safety_margin_seconds:
                ctx.abort(
                    f"Request deadline exceeded (remaining={remaining:.2f}s, "
                    f"margin={self.safety_margin_seconds}s)"
                )
                logger.warning(
                    "Deadline-aware strategy: aborting '%s' retry (%.2fs remaining)",
                    ctx.provider,
                    max(0.0, remaining),
                )
                return RecoveryAction.ABORT

        # Check context deadline
        if ctx.deadline_seconds > 0:
            remaining = ctx.deadline_seconds - now
            if remaining <= self.safety_margin_seconds:
                ctx.abort(
                    f"Request deadline exceeded (remaining={remaining:.2f}s)"
                )
                logger.warning(
                    "Deadline-aware strategy: context deadline exceeded for '%s' "
                    "(%.2fs remaining)",
                    ctx.provider,
                    max(0.0, remaining),
                )
                return RecoveryAction.ABORT

        return RecoveryAction.RETRY


class CircuitBreakerResetStrategy(BaseRecoveryStrategy):
    """Handle open-circuit errors by optionally requesting a circuit reset.

    When a :class:`~src.errors.CircuitOpenError` is encountered the
    strategy checks whether a manual reset function is registered.  If
    so, and if the circuit has been open for longer than
    *force_reset_after_seconds*, the circuit is force-reset and the
    request is retried.  Otherwise the error is aborted.

    This strategy is intended for test/admin scenarios — in production
    prefer letting the circuit reset naturally via its ``reset_timeout``.

    Args:
        force_reset_func: Optional callable ``(service: str) -> None``
            that force-resets the named circuit breaker.
        force_reset_after_seconds: Only force-reset if the circuit has
            been open for at least this many seconds.
    """

    def __init__(
        self,
        force_reset_func: Optional[Callable[[str], None]] = None,
        force_reset_after_seconds: float = 60.0,
    ) -> None:
        self._reset_func = force_reset_func
        self.force_reset_after_seconds = force_reset_after_seconds

    def applies_to(self, error: Exception, ctx: RecoveryContext) -> bool:
        """Match circuit-open errors."""
        return isinstance(error, CircuitOpenError)

    def recover(
        self,
        error: Exception,
        ctx: RecoveryContext,
    ) -> RecoveryAction:
        service = getattr(error, "args", [""])[0] if error.args else ctx.provider
        last_reset = ctx.metadata.get("last_circuit_reset", 0.0)
        now = time.monotonic()

        if (
            self._reset_func is not None
            and (now - last_reset) >= self.force_reset_after_seconds
        ):
            try:
                logger.warning(
                    "Circuit open for '%s': force-resetting (last reset %.1fs ago)",
                    service,
                    now - last_reset if last_reset else float("inf"),
                )
                self._reset_func(service)
                ctx.metadata["last_circuit_reset"] = now
                return RecoveryAction.RETRY
            except Exception as exc:
                logger.error(
                    "Circuit reset for '%s' failed: %s, aborting",
                    service,
                    exc,
                )

        # No reset function or not yet time — abort
        reset_timeout = (
            error.context.details.get("reset_timeout", 0)
            if isinstance(error, GatewayError)
            else 0
        )
        ctx.abort(
            f"Circuit breaker open for '{service}' "
            f"(reset_timeout={reset_timeout}s)"
        )
        logger.info(
            "Circuit-breaker strategy: aborting retry for '%s' "
            "(circuit is open)",
            service,
        )
        return RecoveryAction.ABORT


class ProviderRotationStrategy(BaseRecoveryStrategy):
    """Record provider-level failures and suggest provider rotation.

    Tracks per-provider failure counts within a single request lifecycle.
    When a provider accumulates more than *max_provider_failures* errors
    the strategy signals that retries should be skipped for this provider
    (allowing upstream failover to kick in).

    This strategy does **not** perform the actual failover — it simply
    surfaces the recommendation via :attr:`RecoveryContext.metadata`
    under the key ``"suggest_failover"``.

    Args:
        max_provider_failures: Maximum tolerated failures per provider
            before suggesting failover.
    """

    def __init__(self, max_provider_failures: int = 2) -> None:
        self.max_provider_failures = max_provider_failures

    def applies_to(self, error: Exception, ctx: RecoveryContext) -> bool:
        """Apply to all retryable errors from a specific provider."""
        return bool(ctx.provider) and is_retryable_exception(error)

    def recover(
        self,
        error: Exception,
        ctx: RecoveryContext,
    ) -> RecoveryAction:
        key = f"provider_failures.{ctx.provider}"
        count = ctx.metadata.get(key, 0) + 1
        ctx.metadata[key] = count

        if count >= self.max_provider_failures:
            ctx.metadata["suggest_failover"] = ctx.provider
            logger.warning(
                "Provider rotation: '%s' has failed %d time(s), "
                "suggesting failover",
                ctx.provider,
                count,
            )
            return RecoveryAction.SKIP_RETRY

        return RecoveryAction.RETRY


class ErrorAggregationStrategy(BaseRecoveryStrategy):
    """Aggregate errors from multiple retry attempts for richer diagnostics.

    Maintains a concise summary of all errors seen during the retry
    lifecycle in :attr:`RecoveryContext.metadata` under ``"error_summary"``.
    The summary includes category distributions, unique messages, and
    the first / last errors for debugging.

    This strategy never aborts — it only enriches context and always
    returns :class:`RecoveryAction.RETRY`.

    Args:
        max_errors: Maximum number of full error records to retain.
    """

    def __init__(self, max_errors: int = 20) -> None:
        self.max_errors = max_errors

    def applies_to(self, error: Exception, ctx: RecoveryContext) -> bool:
        """Apply to all errors."""
        return True

    def recover(
        self,
        error: Exception,
        ctx: RecoveryContext,
    ) -> RecoveryAction:
        summary = ctx.metadata.setdefault(
            "error_summary",
            {
                "total": 0,
                "categories": {},
                "first_error": None,
                "last_error": None,
                "unique_messages": [],
            },
        )

        summary["total"] += 1

        # Category tracking
        if isinstance(error, GatewayError):
            cat = error.context.category.value
        else:
            cat = "unknown"
        summary["categories"][cat] = summary["categories"].get(cat, 0) + 1

        # First / last error
        error_repr = {
            "type": type(error).__name__,
            "message": str(error)[:200],
            "category": cat,
        }
        if summary["first_error"] is None:
            summary["first_error"] = error_repr
        summary["last_error"] = error_repr

        # Unique messages (capped)
        msg = str(error)[:100]
        if len(summary["unique_messages"]) < self.max_errors:
            if msg not in summary["unique_messages"]:
                summary["unique_messages"].append(msg)

        logger.debug(
            "Error aggregation: attempt %d total=%d categories=%s",
            ctx.attempt,
            summary["total"],
            summary["categories"],
        )

        return RecoveryAction.RETRY


class ExponentialDelayStrategy(BaseRecoveryStrategy):
    """Add exponential extra delay to the recovery context.

    This strategy augments the backoff computed by the retry engine with
    an additional delay derived from the attempt number.  Useful when
    you want a *minimum* back-off irrespective of what the retry config
    computes.

    Args:
        base_extra_delay: Base extra delay (seconds) for the first retry.
        multiplier: Exponential multiplier (applied per attempt).
        max_extra_delay: Cap for the extra delay.
    """

    def __init__(
        self,
        base_extra_delay: float = 0.5,
        multiplier: float = 2.0,
        max_extra_delay: float = 30.0,
    ) -> None:
        self.base_extra_delay = base_extra_delay
        self.multiplier = multiplier
        self.max_extra_delay = max_extra_delay

    def applies_to(self, error: Exception, ctx: RecoveryContext) -> bool:
        """Apply to retryable errors."""
        return is_retryable_exception(error)

    def recover(
        self,
        error: Exception,
        ctx: RecoveryContext,
    ) -> RecoveryAction:
        attempt = max(1, ctx.attempt)
        extra = min(
            self.base_extra_delay * (self.multiplier ** (attempt - 1)),
            self.max_extra_delay,
        )
        ctx.extra_delay = max(ctx.extra_delay, extra)
        logger.debug(
            "Exponential delay: adding %.2fs extra delay (attempt=%d)",
            extra,
            attempt,
        )
        return RecoveryAction.WAIT_AND_RETRY


# --------------------------------------------------------------------------- #
# RecoveryChain — composite strategy runner
# --------------------------------------------------------------------------- #


class RecoveryChain:
    """Run multiple recovery strategies in sequence.

    Strategies are evaluated in registration order.  The chain stops at
    the first strategy that returns :class:`RecoveryAction.ABORT` or
    :class:`RecoveryAction.SKIP_RETRY`.  If no strategy matches, the
    default action is :class:`RecoveryAction.RETRY`.

    Args:
        strategies: Ordered list of :class:`BaseRecoveryStrategy`
            instances to run.
        default_action: Action to return when no strategy matches.
    """

    def __init__(
        self,
        strategies: Optional[list[BaseRecoveryStrategy]] = None,
        default_action: RecoveryAction = RecoveryAction.RETRY,
    ) -> None:
        self.strategies: list[BaseRecoveryStrategy] = strategies or []
        self.default_action = default_action

    def add(self, strategy: BaseRecoveryStrategy) -> "RecoveryChain":
        """Append a strategy to the chain and return *self* for chaining.

        Args:
            strategy: Strategy to add.

        Returns:
            This :class:`RecoveryChain` instance.
        """
        self.strategies.append(strategy)
        return self

    def recover(
        self,
        error: Exception,
        ctx: RecoveryContext,
    ) -> RecoveryAction:
        """Run all applicable strategies and return the final action.

        The chain runs every applicable strategy in order, stopping
        early only on ABORT or SKIP_RETRY.  Among RETRY /
        WAIT_AND_RETRY actions it keeps the *most significant* one
        seen (WAIT_AND_RETRY takes precedence over RETRY) so that
        callers can tell when they need to respect ``ctx.extra_delay``.

        Args:
            error: The exception to recover from.
            ctx: Mutable recovery context.

        Returns:
            * :data:`RecoveryAction.ABORT` — if any strategy aborted.
            * :data:`RecoveryAction.SKIP_RETRY` — if any strategy
              requested a skip.
            * :data:`RecoveryAction.WAIT_AND_RETRY` — if any strategy
              set extra delay and no abort was requested.
            * :data:`RecoveryAction.RETRY` — default when all
              strategies pass without extra delay.
        """
        ctx.record_error(error)

        # Track the most significant non-abort action seen so far.
        # Priority: WAIT_AND_RETRY > RETRY.
        best_action: RecoveryAction = self.default_action

        for strategy in self.strategies:
            if not strategy.applies_to(error, ctx):
                continue

            action = strategy.recover(error, ctx)
            logger.debug(
                "Recovery chain: strategy=%s action=%s attempt=%d provider=%s",
                strategy.name,
                action.value,
                ctx.attempt,
                ctx.provider,
            )

            if action in (RecoveryAction.ABORT, RecoveryAction.SKIP_RETRY):
                return action

            # Upgrade best_action if this strategy requests a wait
            if action == RecoveryAction.WAIT_AND_RETRY:
                best_action = RecoveryAction.WAIT_AND_RETRY

        # If context was flagged for abort by any strategy, honour it
        if ctx.aborted:
            return RecoveryAction.ABORT

        # Also treat non-zero extra_delay as implying WAIT_AND_RETRY
        if ctx.extra_delay > 0 and best_action == RecoveryAction.RETRY:
            best_action = RecoveryAction.WAIT_AND_RETRY

        return best_action

    def make_on_retry_callback(
        self,
        ctx: RecoveryContext,
    ) -> Callable[[int, float, Exception], None]:
        """Create an ``on_retry`` callback wired to this chain.

        The returned callable can be used as
        :attr:`~src.retry.RetryConfig.on_retry`.  It invokes the full
        recovery chain on each retry attempt and sleeps for
        ``ctx.extra_delay`` (set by strategies) in addition to the
        normal backoff.

        Args:
            ctx: Shared recovery context.

        Returns:
            A callable ``(attempt, delay, exc) -> None``.
        """

        def on_retry(attempt: int, delay: float, exc: Exception) -> None:
            ctx.attempt = attempt
            action = self.recover(exc, ctx)

            if action == RecoveryAction.ABORT:
                # Raise a non-retryable exception to short-circuit the loop
                raise RecoveryAbortedError(
                    f"Recovery chain aborted: {ctx.abort_reason}",
                    original_error=exc,
                    context=ctx,
                )

            if ctx.extra_delay > 0:
                logger.debug(
                    "Recovery chain: sleeping extra %.2fs before retry",
                    ctx.extra_delay,
                )
                time.sleep(ctx.extra_delay)
                ctx.extra_delay = 0.0  # Reset for next attempt

        return on_retry

    def __len__(self) -> int:
        return len(self.strategies)

    def __repr__(self) -> str:
        names = [s.name for s in self.strategies]
        return f"RecoveryChain([{', '.join(names)}])"


# --------------------------------------------------------------------------- #
# RecoveryAbortedError
# --------------------------------------------------------------------------- #


class RecoveryAbortedError(BaseException):
    """Raised when a recovery strategy decides to abort retrying.

    Inherits from :class:`BaseException` (not :class:`Exception`) so that it
    bypasses the ``except Exception: pass`` guard that
    :func:`~src.retry.retry_call` places around ``on_retry`` callbacks.
    This ensures that a strategy's abort decision actually stops the retry
    loop rather than being silently swallowed.

    This exception is non-retryable and carries the original error and
    the recovery context for debugging.

    Attributes:
        original_error: The exception that triggered the abort decision.
        context: The :class:`RecoveryContext` at time of abort.
    """

    def __init__(
        self,
        message: str,
        *,
        original_error: Optional[Exception] = None,
        context: Optional[RecoveryContext] = None,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error
        self.context = context


# --------------------------------------------------------------------------- #
# Preset chain factories
# --------------------------------------------------------------------------- #


def make_standard_recovery_chain(
    provider: Optional[str] = None,
    refresh_credentials: Optional[Callable[[str], None]] = None,
    deadline_seconds: float = 0.0,
    force_circuit_reset: Optional[Callable[[str], None]] = None,
    max_rate_limit_wait: float = 120.0,
    abort_after_consecutive_rate_limits: int = 5,
) -> RecoveryChain:
    """Build a standard recovery chain suitable for most providers.

    Includes: deadline enforcement → rate-limit handling →
    credential refresh → circuit-breaker handling → error aggregation.

    Args:
        provider: Provider name (used in log messages).
        refresh_credentials: Optional credential refresh callback.
        deadline_seconds: Hard deadline (monotonic time).
        force_circuit_reset: Optional circuit reset callback.
        max_rate_limit_wait: Maximum wait for rate-limit Retry-After.
        abort_after_consecutive_rate_limits: Abort threshold for rate limits.

    Returns:
        A configured :class:`RecoveryChain`.
    """
    strategies: list[BaseRecoveryStrategy] = []

    # 1. Always check the deadline first
    strategies.append(
        DeadlineAwareStrategy(deadline_seconds=deadline_seconds)
    )

    # 2. Handle rate limits (most common transient error)
    strategies.append(
        RateLimitRecoveryStrategy(
            max_wait_seconds=max_rate_limit_wait,
            abort_after_consecutive=abort_after_consecutive_rate_limits,
        )
    )

    # 3. Credential refresh (if callback provided)
    if refresh_credentials is not None:
        strategies.append(AuthRefreshStrategy(refresh_credentials=refresh_credentials))

    # 4. Circuit-breaker handling
    strategies.append(
        CircuitBreakerResetStrategy(force_reset_func=force_circuit_reset)
    )

    # 5. Error aggregation (always last — never aborts)
    strategies.append(ErrorAggregationStrategy())

    return RecoveryChain(strategies)


def make_aggressive_recovery_chain(
    max_provider_failures: int = 1,
    deadline_seconds: float = 0.0,
) -> RecoveryChain:
    """Build an aggressive chain that quickly rotates providers.

    Suitable for latency-sensitive paths where failover speed is more
    important than exhaustive retries.

    Args:
        max_provider_failures: Failures per provider before suggesting
            rotation.
        deadline_seconds: Hard deadline (monotonic time).

    Returns:
        A configured :class:`RecoveryChain`.
    """
    return RecoveryChain(
        [
            DeadlineAwareStrategy(deadline_seconds=deadline_seconds),
            ProviderRotationStrategy(max_provider_failures=max_provider_failures),
            ErrorAggregationStrategy(),
        ]
    )
