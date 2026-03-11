"""Shared retry budget management for claude-code-model-gateway.

Implements a token-bucket algorithm for system-wide retry rate limiting.
Under high load — when many concurrent requests are all retrying — the
aggregate retry rate can overwhelm a struggling upstream provider and
prevent recovery.  A *retry budget* provides a shared pool that limits
the **total** number of retries across all in-flight requests.

Key components:

- **RetryBudgetConfig** — configurable bucket size, refill rate, and
  per-provider sub-limits.
- **RetryBudget** — thread-safe token bucket with per-provider tracking,
  stats export, and sliding-window rate enforcement.
- **RetryBudgetMiddleware** — drop-in wrapper around :func:`retry_call`
  that checks (and consumes) the budget before each retry attempt.
- **BudgetAwareRetryConfig** — a :class:`~src.retry.RetryConfig` factory
  that integrates a ``RetryBudget`` via the ``on_retry`` callback.

Typical usage::

    from src.retry_budget import RetryBudget, RetryBudgetConfig, BudgetAwareRetryConfig
    from src.retry import retry_call

    budget = RetryBudget(RetryBudgetConfig(
        max_retries_per_second=5.0,
        max_total_retries=500,
        per_provider_limit=100,
    ))

    config = BudgetAwareRetryConfig(budget=budget, provider="anthropic")
    result = retry_call(my_api_call, config=config)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.retry import RetryConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class RetryBudgetConfig:
    """Configuration for a shared retry budget.

    Attributes:
        max_retries_per_second: Maximum retry attempts allowed per second
            across all requests (token-bucket refill rate).
        window_seconds: Sliding time window (seconds) used for rate
            measurement.  Also controls how long individual retry
            timestamps are retained.
        max_total_retries: Hard cap on the total number of retries
            processed since the budget was created (or last reset).
            Use 0 for unlimited.
        per_provider_limit: Optional maximum retries per provider within
            *window_seconds*.  When set, each provider has its own
            sliding-window counter in addition to the global one.
        burst_capacity: Maximum number of retries allowed in a single
            burst (tokens in the bucket).  Defaults to
            ``max_retries_per_second * 10`` if 0.
        warn_at_percent: Log a WARNING when budget consumption exceeds
            this fraction (0.0–1.0) of ``max_total_retries``.
    """

    max_retries_per_second: float = 10.0
    window_seconds: float = 60.0
    max_total_retries: int = 1000
    per_provider_limit: Optional[int] = None
    burst_capacity: int = 0
    warn_at_percent: float = 0.8

    def __post_init__(self) -> None:
        """Derive burst capacity when not explicitly set."""
        if self.burst_capacity == 0:
            self.burst_capacity = max(1, int(self.max_retries_per_second * 10))


# --------------------------------------------------------------------------- #
# Core retry budget
# --------------------------------------------------------------------------- #


class RetryBudget:
    """Thread-safe shared retry budget using a token-bucket algorithm.

    The token bucket is refilled at *max_retries_per_second* and holds
    up to *burst_capacity* tokens.  Each retry attempt consumes one
    token.  When the bucket is empty (or the per-provider or total-cap
    limits are hit) :meth:`consume` returns *False* and the caller
    should **skip** the retry rather than sleeping.

    Stats are exported via :meth:`get_stats` for use in health checks
    and dashboards.

    Args:
        config: Budget configuration (defaults used if not provided).

    Example::

        budget = RetryBudget(RetryBudgetConfig(
            max_retries_per_second=5.0,
            max_total_retries=500,
        ))

        # In your retry loop:
        if budget.consume(provider="anthropic"):
            time.sleep(delay)
            result = call_api()
        else:
            raise BudgetExhaustedError("Retry budget exhausted")
    """

    def __init__(self, config: Optional[RetryBudgetConfig] = None) -> None:
        self.config = config or RetryBudgetConfig()
        self._lock = threading.Lock()

        # Token bucket state
        self._tokens: float = float(self.config.burst_capacity)
        self._last_refill: float = time.monotonic()

        # Counters
        self._total_retries: int = 0
        self._total_consumed: int = 0  # attempts that actually consumed a token
        self._total_rejected: int = 0  # attempts that were rejected

        # Sliding-window retry timestamps (global + per-provider)
        self._global_times: deque[float] = deque()
        self._provider_times: dict[str, deque[float]] = defaultdict(deque)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def can_retry(self, provider: Optional[str] = None) -> bool:
        """Non-destructive check — would the next consume() succeed?

        Args:
            provider: Optional provider name for per-provider limit check.

        Returns:
            True if a retry is currently allowed.
        """
        with self._lock:
            return self._check_budget(provider, consume=False)

    def consume(self, provider: Optional[str] = None) -> bool:
        """Attempt to consume one retry token from the budget.

        Args:
            provider: Optional provider name for per-provider tracking.

        Returns:
            True if the token was consumed successfully (retry is
            allowed), False if the budget is exhausted.
        """
        with self._lock:
            allowed = self._check_budget(provider, consume=True)
            if allowed:
                self._total_consumed += 1
                logger.debug(
                    "Retry budget consumed (provider=%s, total=%d, remaining_tokens=%.1f)",
                    provider or "global",
                    self._total_consumed,
                    self._tokens,
                )
                # Warn when approaching total cap
                if (
                    self.config.max_total_retries > 0
                    and self._total_consumed
                    >= self.config.max_total_retries * self.config.warn_at_percent
                ):
                    logger.warning(
                        "Retry budget at %.0f%% capacity (%d / %d total retries consumed)",
                        self.config.warn_at_percent * 100,
                        self._total_consumed,
                        self.config.max_total_retries,
                    )
            else:
                self._total_rejected += 1
                logger.warning(
                    "Retry budget rejected (provider=%s, consumed=%d, rejected=%d)",
                    provider or "global",
                    self._total_consumed,
                    self._total_rejected,
                )
            return allowed

    def reset(self) -> None:
        """Reset the budget to its initial state."""
        with self._lock:
            self._tokens = float(self.config.burst_capacity)
            self._last_refill = time.monotonic()
            self._total_retries = 0
            self._total_consumed = 0
            self._total_rejected = 0
            self._global_times.clear()
            self._provider_times.clear()
        logger.info("Retry budget reset")

    def get_stats(self) -> dict[str, Any]:
        """Return current budget statistics for monitoring.

        Returns:
            Dictionary with token counts, rates, and counters.
        """
        now = time.monotonic()
        with self._lock:
            self._refill_tokens(now)
            cutoff = now - self.config.window_seconds
            recent_global = sum(1 for t in self._global_times if t >= cutoff)
            rate = recent_global / max(self.config.window_seconds, 1.0)

            provider_stats = {
                name: sum(1 for t in times if t >= cutoff)
                for name, times in self._provider_times.items()
            }

            return {
                "tokens_available": round(self._tokens, 2),
                "burst_capacity": self.config.burst_capacity,
                "total_consumed": self._total_consumed,
                "total_rejected": self._total_rejected,
                "retries_in_window": recent_global,
                "rate_per_second": round(rate, 4),
                "max_rate_per_second": self.config.max_retries_per_second,
                "max_total_retries": self.config.max_total_retries,
                "budget_remaining": (
                    max(0, self.config.max_total_retries - self._total_consumed)
                    if self.config.max_total_retries > 0
                    else -1
                ),
                "per_provider": provider_stats,
            }

    # ------------------------------------------------------------------ #
    # Private helpers (must be called with self._lock held)
    # ------------------------------------------------------------------ #

    def _refill_tokens(self, now: float) -> None:
        """Refill the token bucket based on elapsed time."""
        elapsed = now - self._last_refill
        if elapsed > 0:
            new_tokens = elapsed * self.config.max_retries_per_second
            self._tokens = min(
                float(self.config.burst_capacity),
                self._tokens + new_tokens,
            )
            self._last_refill = now

    def _purge_old(self, now: float) -> None:
        """Remove sliding-window entries older than window_seconds."""
        cutoff = now - self.config.window_seconds
        while self._global_times and self._global_times[0] < cutoff:
            self._global_times.popleft()
        for times in self._provider_times.values():
            while times and times[0] < cutoff:
                times.popleft()

    def _check_budget(self, provider: Optional[str], *, consume: bool) -> bool:
        """Core check / consume logic.  Lock must be held by caller."""
        now = time.monotonic()
        self._refill_tokens(now)
        self._purge_old(now)

        # 1. Token bucket check
        if self._tokens < 1.0:
            return False

        # 2. Total cap check
        if self.config.max_total_retries > 0:
            if self._total_consumed >= self.config.max_total_retries:
                return False

        # 3. Per-provider limit
        if provider and self.config.per_provider_limit is not None:
            times = self._provider_times[provider]
            if len(times) >= self.config.per_provider_limit:
                return False

        # All checks passed → optionally consume
        if consume:
            self._tokens -= 1.0
            self._global_times.append(now)
            if provider:
                self._provider_times[provider].append(now)

        return True


# --------------------------------------------------------------------------- #
# Budget-exhausted error
# --------------------------------------------------------------------------- #


class BudgetExhaustedError(Exception):
    """Raised when the retry budget is exhausted.

    This is a non-retryable condition — the caller should propagate the
    *original* error rather than retrying.

    Attributes:
        provider: The provider that was being retried.
        stats: Budget statistics at the time of rejection.
    """

    def __init__(
        self,
        message: str = "Retry budget exhausted",
        *,
        provider: Optional[str] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.stats = stats or {}


# --------------------------------------------------------------------------- #
# Budget-aware RetryConfig subclass
# --------------------------------------------------------------------------- #


class _BudgetAwareRetryConfig(RetryConfig):
    """Internal :class:`~src.retry.RetryConfig` subclass that aborts the retry
    loop when the shared budget is exhausted.

    :func:`~src.retry.retry_call` silently swallows exceptions raised from the
    ``on_retry`` callback, so raising :class:`BudgetExhaustedError` there has no
    effect.  Instead, an ``_exhausted_flag`` (a single-element ``list[bool]``) is
    shared between the ``on_retry`` callback and this subclass.  When the callback
    sets the flag to ``True``, the next call to :meth:`should_retry` raises
    :class:`BudgetExhaustedError` directly — which *is* visible to the caller
    because ``retry_call`` does not protect that call path.

    This class is an internal implementation detail; use
    :func:`make_budget_aware_retry_config` or :class:`RetryBudgetMiddleware`.
    """

    def __init__(
        self,
        budget: RetryBudget,
        provider: Optional[str],
        exhausted_flag: list,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # These are regular instance attributes, not dataclass fields.
        self._budget = budget
        self._provider = provider
        self._exhausted_flag = exhausted_flag

    def should_retry(self, exc: Exception) -> bool:
        """Raise :class:`BudgetExhaustedError` if the shared budget was exhausted.

        This is called by :func:`~src.retry.retry_call` before each retry.
        When the ``on_retry`` callback has signalled budget exhaustion via
        ``_exhausted_flag``, we raise here so that the error propagates out of
        the retry loop to the caller rather than being silently swallowed.
        """
        if self._exhausted_flag[0]:
            raise BudgetExhaustedError(
                "Retry budget exhausted",
                provider=self._provider,
                stats=self._budget.get_stats(),
            )
        return super().should_retry(exc)


# --------------------------------------------------------------------------- #
# BudgetAwareRetryConfig factory
# --------------------------------------------------------------------------- #


def make_budget_aware_on_retry(
    budget: RetryBudget,
    provider: Optional[str] = None,
    on_budget_exhausted: Optional[Callable[[str, int, Exception], None]] = None,
    _exhausted_flag: Optional[list] = None,
) -> Callable[[int, float, Exception], None]:
    """Create an ``on_retry`` callback that gates retries against a budget.

    Returns a callable suitable for use as
    :attr:`~src.retry.RetryConfig.on_retry`.  Before each retry attempt
    the callback consumes one token from *budget*.

    .. important::

       :func:`~src.retry.retry_call` silently swallows exceptions raised
       by ``on_retry`` callbacks to protect production retry loops.
       Therefore this callback does **not** raise directly.  Instead it
       sets an ``_exhausted_flag`` (a single-element list) which must be
       checked by a guarded wrapper function at the top of each attempt.
       Use :func:`make_budget_aware_retry_config` or
       :class:`RetryBudgetMiddleware` which handle this automatically.

    Args:
        budget: The shared :class:`RetryBudget` to consume from.
        provider: Provider name for per-provider tracking.
        on_budget_exhausted: Optional callback invoked when the budget is
            exhausted.  Receives (provider, attempt, exception).
        _exhausted_flag: Optional list[bool] used to signal exhaustion to
            a guarded wrapper.  If not provided the callback raises
            :class:`BudgetExhaustedError` directly (which may be swallowed
            by ``retry_call``).

    Returns:
        A callable ``(attempt: int, delay: float, exc: Exception) -> None``.
    """

    def on_retry(attempt: int, delay: float, exc: Exception) -> None:
        if not budget.consume(provider=provider):
            stats = budget.get_stats()
            logger.warning(
                "Retry budget exhausted for provider '%s' on attempt %d: %s",
                provider or "global",
                attempt,
                stats,
            )
            if on_budget_exhausted:
                try:
                    on_budget_exhausted(provider or "global", attempt, exc)
                except Exception:
                    pass

            if _exhausted_flag is not None:
                # Signal the guarded wrapper to raise on the next call
                _exhausted_flag[0] = True
            else:
                # Fallback: raise directly (may be swallowed by retry_call)
                raise BudgetExhaustedError(
                    f"Retry budget exhausted after {attempt} attempts",
                    provider=provider,
                    stats=stats,
                )

    return on_retry


# --------------------------------------------------------------------------- #
# BudgetAwareRetryConfig — convenience wrapper
# --------------------------------------------------------------------------- #


def make_budget_aware_retry_config(
    budget: RetryBudget,
    *,
    provider: Optional[str] = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    on_budget_exhausted: Optional[Callable[[str, int, Exception], None]] = None,
) -> RetryConfig:
    """Create a :class:`~src.retry.RetryConfig` wired to a shared budget.

    The returned config will check the budget before every retry attempt.
    If the budget is exhausted, the retry loop is aborted immediately.

    Args:
        budget: The shared :class:`RetryBudget`.
        provider: Provider name (for per-provider accounting).
        max_attempts: Maximum number of attempts.
        base_delay: Base retry delay in seconds.
        max_delay: Maximum retry delay in seconds.
        jitter: Whether to apply jitter to delays.
        on_budget_exhausted: Optional callback when budget runs out.

    Returns:
        A :class:`~src.retry.RetryConfig` with budget checking enabled.

    Example::

        budget = RetryBudget()
        config = make_budget_aware_retry_config(
            budget,
            provider="anthropic",
            max_attempts=5,
        )
        result = retry_call(my_func, config=config)
    """
    # Single-element list used as a mutable boolean flag.  The on_retry
    # callback sets this to True when the budget is exhausted.
    # _BudgetAwareRetryConfig.should_retry() checks the flag and raises
    # BudgetExhaustedError, which propagates out of retry_call to the caller.
    exhausted_flag: list = [False]

    return _BudgetAwareRetryConfig(
        budget=budget,
        provider=provider,
        exhausted_flag=exhausted_flag,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter,
        on_retry=make_budget_aware_on_retry(
            budget,
            provider=provider,
            on_budget_exhausted=on_budget_exhausted,
            _exhausted_flag=exhausted_flag,
        ),
    )


# --------------------------------------------------------------------------- #
# Global budget singleton
# --------------------------------------------------------------------------- #

_global_budget: Optional[RetryBudget] = None
_budget_lock = threading.Lock()


def get_global_budget(
    config: Optional[RetryBudgetConfig] = None,
) -> RetryBudget:
    """Get or create the global retry budget singleton.

    Args:
        config: Configuration (only used on first call).

    Returns:
        The global :class:`RetryBudget` instance.
    """
    global _global_budget
    with _budget_lock:
        if _global_budget is None:
            _global_budget = RetryBudget(config)
        return _global_budget


def reset_global_budget() -> None:
    """Discard the global budget singleton (primarily for testing)."""
    global _global_budget
    with _budget_lock:
        _global_budget = None


# --------------------------------------------------------------------------- #
# RetryBudgetMiddleware — standalone executor
# --------------------------------------------------------------------------- #


class RetryBudgetMiddleware:
    """Execute functions with retry logic governed by a shared budget.

    Wraps :func:`~src.retry.retry_call` with budget-gating.  Each retry
    attempt consumes one token; when the budget runs dry the last error
    is re-raised without further retrying.

    Args:
        budget: Shared :class:`RetryBudget`.
        provider: Default provider name for budget accounting.
        max_attempts: Default maximum attempts per call.
        base_delay: Default base retry delay.

    Example::

        middleware = RetryBudgetMiddleware(budget=budget, provider="anthropic")
        result = middleware.execute(call_api)
    """

    def __init__(
        self,
        budget: Optional[RetryBudget] = None,
        provider: Optional[str] = None,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        self.budget = budget or get_global_budget()
        self.provider = provider
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def execute(
        self,
        func: Callable[..., Any],
        *,
        provider: Optional[str] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Execute *func* with budget-gated retries.

        Args:
            func: The callable to execute.
            provider: Override provider name for this call.
            args: Positional arguments for *func*.
            kwargs: Keyword arguments for *func*.

        Returns:
            The return value of *func*.

        Raises:
            BudgetExhaustedError: When the shared budget is consumed.
            RetryExhaustedError: When max_attempts are exhausted.
            Exception: Non-retryable exceptions are re-raised immediately.
        """
        from src.retry import retry_call

        effective_provider = provider or self.provider
        config = make_budget_aware_retry_config(
            self.budget,
            provider=effective_provider,
            max_attempts=self.max_attempts,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            jitter=self.jitter,
        )
        return retry_call(func, args=args, kwargs=kwargs or {}, config=config)
