"""Tests for the retry budget module."""

from __future__ import annotations

import threading
import time

import pytest

from src.retry_budget import (
    BudgetExhaustedError,
    RetryBudget,
    RetryBudgetConfig,
    RetryBudgetMiddleware,
    get_global_budget,
    make_budget_aware_on_retry,
    make_budget_aware_retry_config,
    reset_global_budget,
)
from src.errors import NetworkError, AuthenticationError, RetryExhaustedError
from src.retry import RetryConfig, retry_call


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before and after each test."""
    reset_global_budget()
    yield
    reset_global_budget()


@pytest.fixture
def unlimited_budget():
    """Budget with very generous limits — effectively unlimited for tests."""
    return RetryBudget(
        RetryBudgetConfig(
            max_retries_per_second=10000.0,
            window_seconds=60.0,
            max_total_retries=0,  # unlimited
            burst_capacity=10000,
        )
    )


@pytest.fixture
def tight_budget():
    """Budget with tight limits for exhaustion testing."""
    return RetryBudget(
        RetryBudgetConfig(
            max_retries_per_second=10000.0,
            window_seconds=60.0,
            max_total_retries=3,
            burst_capacity=10,
        )
    )


# --------------------------------------------------------------------------- #
# RetryBudgetConfig tests
# --------------------------------------------------------------------------- #


class TestRetryBudgetConfig:
    """Tests for RetryBudgetConfig."""

    def test_defaults(self):
        """Default config has sensible values."""
        cfg = RetryBudgetConfig()
        assert cfg.max_retries_per_second == 10.0
        assert cfg.window_seconds == 60.0
        assert cfg.max_total_retries == 1000
        assert cfg.per_provider_limit is None
        assert cfg.burst_capacity > 0  # auto-computed
        assert cfg.warn_at_percent == 0.8

    def test_burst_capacity_auto_computed(self):
        """burst_capacity is computed from rate when not explicitly set."""
        cfg = RetryBudgetConfig(max_retries_per_second=5.0, burst_capacity=0)
        assert cfg.burst_capacity == 50  # 5.0 * 10

    def test_burst_capacity_explicit(self):
        """Explicit burst_capacity is preserved."""
        cfg = RetryBudgetConfig(burst_capacity=42)
        assert cfg.burst_capacity == 42

    def test_warn_at_percent_custom(self):
        """Custom warn_at_percent is stored."""
        cfg = RetryBudgetConfig(warn_at_percent=0.5)
        assert cfg.warn_at_percent == 0.5


# --------------------------------------------------------------------------- #
# RetryBudget — basic operations
# --------------------------------------------------------------------------- #


class TestRetryBudgetBasic:
    """Basic consume / can_retry functionality."""

    def test_initial_budget_allows_retry(self):
        """A fresh budget allows the first retry."""
        budget = RetryBudget()
        assert budget.can_retry() is True

    def test_consume_returns_true_when_available(self):
        """consume() returns True when tokens are available."""
        budget = RetryBudget(
            RetryBudgetConfig(max_total_retries=10, burst_capacity=10)
        )
        assert budget.consume() is True

    def test_consume_decrements_tokens(self):
        """Each consume() uses one token."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=1000.0,
                max_total_retries=0,
                burst_capacity=5,
            )
        )
        for _ in range(5):
            assert budget.consume() is True

    def test_budget_exhausted_by_total_cap(self):
        """Budget returns False once max_total_retries is reached."""
        budget = RetryBudget(
            RetryBudgetConfig(max_total_retries=3, burst_capacity=100)
        )
        assert budget.consume() is True
        assert budget.consume() is True
        assert budget.consume() is True
        # 4th should be rejected
        assert budget.consume() is False

    def test_can_retry_returns_false_when_exhausted(self, tight_budget):
        """can_retry() returns False after budget is exhausted."""
        for _ in range(3):
            tight_budget.consume()
        assert tight_budget.can_retry() is False

    def test_can_retry_is_non_destructive(self, unlimited_budget):
        """can_retry() does not consume a token."""
        stats_before = unlimited_budget.get_stats()
        unlimited_budget.can_retry()
        unlimited_budget.can_retry()
        unlimited_budget.can_retry()
        stats_after = unlimited_budget.get_stats()
        assert stats_before["total_consumed"] == stats_after["total_consumed"]

    def test_reset_restores_initial_state(self):
        """reset() brings the budget back to its initial state."""
        budget = RetryBudget(RetryBudgetConfig(max_total_retries=2, burst_capacity=10))
        budget.consume()
        budget.consume()
        assert budget.consume() is False

        budget.reset()
        assert budget.consume() is True

    def test_unlimited_budget_never_exhausted(self):
        """max_total_retries=0 means no total cap."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=100000.0,
                max_total_retries=0,
                burst_capacity=100000,
            )
        )
        for _ in range(10000):
            assert budget.consume() is True


# --------------------------------------------------------------------------- #
# RetryBudget — token bucket rate limiting
# --------------------------------------------------------------------------- #


class TestRetryBudgetTokenBucket:
    """Token bucket rate-limiting behaviour."""

    def test_rate_limited_after_burst(self):
        """Budget rejects retries once the burst capacity is exhausted."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=1.0,   # very slow refill
                window_seconds=60.0,
                max_total_retries=0,
                burst_capacity=2,
            )
        )
        assert budget.consume() is True
        assert budget.consume() is True
        # Bucket now empty; refill is slow
        assert budget.consume() is False

    def test_tokens_refill_over_time(self):
        """Tokens refill at the configured rate after the bucket empties."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=100.0,   # refills quickly
                window_seconds=60.0,
                max_total_retries=0,
                burst_capacity=1,
            )
        )
        assert budget.consume() is True
        assert budget.consume() is False  # Bucket empty

        time.sleep(0.05)  # Wait for at least 5 tokens to refill
        assert budget.consume() is True

    def test_burst_capacity_limits_simultaneous_retries(self):
        """burst_capacity limits how many tokens can accumulate."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=1000.0,
                window_seconds=60.0,
                max_total_retries=0,
                burst_capacity=3,
            )
        )
        # Burst: 3 should succeed, 4th should fail
        results = [budget.consume() for _ in range(4)]
        assert results[:3] == [True, True, True]
        assert results[3] is False


# --------------------------------------------------------------------------- #
# RetryBudget — per-provider limits
# --------------------------------------------------------------------------- #


class TestRetryBudgetPerProvider:
    """Per-provider limit behaviour."""

    def test_per_provider_limit_enforced(self):
        """Per-provider limit is respected independently of global cap."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=10000.0,
                window_seconds=60.0,
                max_total_retries=0,
                per_provider_limit=2,
                burst_capacity=10000,
            )
        )
        assert budget.consume(provider="anthropic") is True
        assert budget.consume(provider="anthropic") is True
        # 3rd for same provider should be rejected
        assert budget.consume(provider="anthropic") is False
        # But a different provider is still allowed
        assert budget.consume(provider="openai") is True

    def test_global_limit_applies_across_providers(self):
        """Global cap applies even if per-provider limits are not hit."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=10000.0,
                window_seconds=60.0,
                max_total_retries=3,
                per_provider_limit=10,
                burst_capacity=10000,
            )
        )
        assert budget.consume(provider="anthropic") is True
        assert budget.consume(provider="openai") is True
        assert budget.consume(provider="gemini") is True
        # Global cap hit
        assert budget.consume(provider="bedrock") is False

    def test_no_provider_name_uses_global_counter(self):
        """Calls without a provider name count against the global budget."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=10000.0,
                window_seconds=60.0,
                max_total_retries=2,
                per_provider_limit=5,
                burst_capacity=10000,
            )
        )
        assert budget.consume() is True
        assert budget.consume() is True
        assert budget.consume() is False


# --------------------------------------------------------------------------- #
# RetryBudget — stats
# --------------------------------------------------------------------------- #


class TestRetryBudgetStats:
    """get_stats() output."""

    def test_initial_stats(self):
        """Fresh budget stats are zeroed."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_total_retries=100, burst_capacity=50
            )
        )
        stats = budget.get_stats()
        assert stats["total_consumed"] == 0
        assert stats["total_rejected"] == 0
        assert stats["retries_in_window"] == 0
        assert stats["budget_remaining"] == 100
        assert stats["tokens_available"] == 50

    def test_stats_after_consumption(self):
        """Stats reflect consumed tokens."""
        budget = RetryBudget(
            RetryBudgetConfig(max_total_retries=10, burst_capacity=10)
        )
        budget.consume()
        budget.consume()
        stats = budget.get_stats()
        assert stats["total_consumed"] == 2
        assert stats["budget_remaining"] == 8
        assert stats["total_rejected"] == 0

    def test_stats_after_rejection(self, tight_budget):
        """Rejected retries are counted."""
        for _ in range(3):
            tight_budget.consume()
        tight_budget.consume()  # rejected

        stats = tight_budget.get_stats()
        assert stats["total_consumed"] == 3
        assert stats["total_rejected"] == 1

    def test_stats_unlimited_budget_remaining(self):
        """Unlimited budget shows -1 for remaining."""
        budget = RetryBudget(
            RetryBudgetConfig(max_total_retries=0, burst_capacity=100)
        )
        stats = budget.get_stats()
        assert stats["budget_remaining"] == -1

    def test_stats_per_provider_tracking(self):
        """Per-provider counts appear in stats."""
        budget = RetryBudget(
            RetryBudgetConfig(max_total_retries=0, burst_capacity=100)
        )
        budget.consume(provider="anthropic")
        budget.consume(provider="anthropic")
        budget.consume(provider="openai")

        stats = budget.get_stats()
        assert "anthropic" in stats["per_provider"]
        assert "openai" in stats["per_provider"]
        assert stats["per_provider"]["anthropic"] == 2
        assert stats["per_provider"]["openai"] == 1


# --------------------------------------------------------------------------- #
# RetryBudget — thread safety
# --------------------------------------------------------------------------- #


class TestRetryBudgetThreadSafety:
    """Thread-safety under concurrent access."""

    def test_concurrent_consume(self):
        """Concurrent consume calls do not race."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=100000.0,
                max_total_retries=0,
                burst_capacity=100000,
            )
        )
        exceptions: list[Exception] = []
        success_count: list[int] = [0]
        lock = threading.Lock()

        def worker():
            try:
                for _ in range(100):
                    if budget.consume():
                        with lock:
                            success_count[0] += 1
            except Exception as e:
                exceptions.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0
        assert success_count[0] == 1000

    def test_total_cap_enforced_under_concurrency(self):
        """Total cap is never exceeded under concurrent access."""
        cap = 50
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=100000.0,
                max_total_retries=cap,
                burst_capacity=1000,
            )
        )
        results: list[bool] = []
        lock = threading.Lock()

        def worker():
            for _ in range(20):
                result = budget.consume()
                with lock:
                    results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        consumed = sum(1 for r in results if r)
        assert consumed <= cap

    def test_concurrent_stats_access(self, unlimited_budget):
        """get_stats() is safe to call concurrently."""
        exceptions: list[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    unlimited_budget.consume()
                    unlimited_budget.get_stats()
            except Exception as e:
                exceptions.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 0


# --------------------------------------------------------------------------- #
# make_budget_aware_on_retry
# --------------------------------------------------------------------------- #


class TestMakeBudgetAwareOnRetry:
    """Tests for the on_retry callback factory."""

    def test_callback_allows_retry_when_budget_available(self):
        """Callback does not raise when budget is available."""
        budget = RetryBudget(RetryBudgetConfig(max_total_retries=10, burst_capacity=10))
        callback = make_budget_aware_on_retry(budget, provider="test")
        # Should not raise
        callback(1, 0.0, NetworkError("fail"))

    def test_callback_raises_when_budget_exhausted(self, tight_budget):
        """Callback raises BudgetExhaustedError when budget runs out."""
        # Exhaust the budget
        for _ in range(3):
            tight_budget.consume()

        callback = make_budget_aware_on_retry(tight_budget, provider="test")

        with pytest.raises(BudgetExhaustedError):
            callback(4, 0.0, NetworkError("fail"))

    def test_callback_invokes_on_budget_exhausted(self):
        """on_budget_exhausted callback is invoked when budget is depleted."""
        budget = RetryBudget(RetryBudgetConfig(max_total_retries=1, burst_capacity=10))
        budget.consume()  # exhaust

        exhausted_calls: list[tuple] = []

        def on_exhausted(provider, attempt, exc):
            exhausted_calls.append((provider, attempt, type(exc).__name__))

        callback = make_budget_aware_on_retry(
            budget,
            provider="anthropic",
            on_budget_exhausted=on_exhausted,
        )

        with pytest.raises(BudgetExhaustedError):
            callback(1, 0.0, NetworkError("fail"))

        assert len(exhausted_calls) == 1
        assert exhausted_calls[0][0] == "anthropic"

    def test_callback_consumes_token_on_each_call(self):
        """Each invocation consumes one token."""
        budget = RetryBudget(
            RetryBudgetConfig(max_total_retries=0, burst_capacity=100)
        )
        callback = make_budget_aware_on_retry(budget)

        for _ in range(5):
            callback(1, 0.0, NetworkError("fail"))

        stats = budget.get_stats()
        assert stats["total_consumed"] == 5

    def test_budget_exhausted_error_carries_stats(self):
        """BudgetExhaustedError carries budget stats."""
        budget = RetryBudget(RetryBudgetConfig(max_total_retries=1, burst_capacity=5))
        budget.consume()  # exhaust

        callback = make_budget_aware_on_retry(budget, provider="test")

        try:
            callback(1, 0.0, NetworkError("fail"))
            pytest.fail("Expected BudgetExhaustedError")
        except BudgetExhaustedError as e:
            assert e.provider == "test"
            assert isinstance(e.stats, dict)
            assert "total_consumed" in e.stats


# --------------------------------------------------------------------------- #
# make_budget_aware_retry_config
# --------------------------------------------------------------------------- #


class TestMakeBudgetAwareRetryConfig:
    """Tests for the RetryConfig factory."""

    def test_creates_valid_retry_config(self):
        """Factory returns a valid RetryConfig."""
        budget = RetryBudget()
        config = make_budget_aware_retry_config(budget, provider="test", max_attempts=5)
        assert config.max_attempts == 5
        assert config.on_retry is not None

    def test_budget_stops_retries(self, tight_budget):
        """Exhausted budget causes retry_call to stop early."""
        config = make_budget_aware_retry_config(
            tight_budget,
            provider="test",
            max_attempts=10,
            base_delay=0.0,
        )

        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise NetworkError("fail")

        with pytest.raises((RetryExhaustedError, BudgetExhaustedError)):
            retry_call(always_fail, config=config)

        # Should not have run 10 attempts because budget was exhausted
        # (tight_budget allows 3 total retries, so max calls = 1 initial + 3 retries)
        assert call_count <= 5  # generous upper bound

    def test_success_before_budget_exhaustion(self, unlimited_budget):
        """Successful call proceeds normally without touching budget cap."""
        config = make_budget_aware_retry_config(
            unlimited_budget,
            provider="test",
            max_attempts=5,
            base_delay=0.0,
        )

        count = 0

        def flaky():
            nonlocal count
            count += 1
            if count < 3:
                raise NetworkError("transient")
            return "ok"

        result = retry_call(flaky, config=config)
        assert result == "ok"
        assert count == 3

    def test_non_retryable_error_not_consumed_from_budget(self, unlimited_budget):
        """Non-retryable errors skip the on_retry callback entirely."""
        stats_before = unlimited_budget.get_stats()

        config = make_budget_aware_retry_config(
            unlimited_budget,
            provider="test",
            max_attempts=5,
            base_delay=0.0,
        )

        def raise_auth_error():
            raise AuthenticationError("bad key")

        with pytest.raises(AuthenticationError):
            retry_call(raise_auth_error, config=config)

        stats_after = unlimited_budget.get_stats()
        # No tokens consumed because the error is non-retryable
        assert stats_before["total_consumed"] == stats_after["total_consumed"]


# --------------------------------------------------------------------------- #
# BudgetExhaustedError
# --------------------------------------------------------------------------- #


class TestBudgetExhaustedError:
    """Tests for the BudgetExhaustedError exception."""

    def test_basic_construction(self):
        """BudgetExhaustedError can be constructed with just a message."""
        err = BudgetExhaustedError("exhausted")
        assert str(err) == "exhausted"
        assert err.provider is None
        assert err.stats == {}

    def test_with_provider_and_stats(self):
        """BudgetExhaustedError carries provider and stats."""
        err = BudgetExhaustedError(
            "out of budget",
            provider="anthropic",
            stats={"total_consumed": 100, "budget_remaining": 0},
        )
        assert err.provider == "anthropic"
        assert err.stats["total_consumed"] == 100

    def test_is_not_retryable_gateway_error(self):
        """BudgetExhaustedError is not a GatewayError (correct — it's intentional)."""
        from src.errors import GatewayError

        err = BudgetExhaustedError("out")
        assert not isinstance(err, GatewayError)


# --------------------------------------------------------------------------- #
# Global budget singleton
# --------------------------------------------------------------------------- #


class TestGlobalBudget:
    """Tests for the global budget singleton."""

    def test_get_global_budget_returns_singleton(self):
        """get_global_budget returns the same instance."""
        b1 = get_global_budget()
        b2 = get_global_budget()
        assert b1 is b2

    def test_reset_global_budget_creates_new_instance(self):
        """reset_global_budget discards the old singleton."""
        b1 = get_global_budget()
        reset_global_budget()
        b2 = get_global_budget()
        assert b1 is not b2

    def test_custom_config_on_first_call(self):
        """Custom config is applied on first call."""
        cfg = RetryBudgetConfig(max_total_retries=42)
        budget = get_global_budget(config=cfg)
        assert budget.config.max_total_retries == 42

    def test_custom_config_ignored_on_subsequent_calls(self):
        """Config is only applied on first call; subsequent calls ignore it."""
        cfg1 = RetryBudgetConfig(max_total_retries=42)
        cfg2 = RetryBudgetConfig(max_total_retries=999)

        budget1 = get_global_budget(config=cfg1)
        budget2 = get_global_budget(config=cfg2)

        assert budget1 is budget2
        assert budget1.config.max_total_retries == 42


# --------------------------------------------------------------------------- #
# RetryBudgetMiddleware
# --------------------------------------------------------------------------- #


class TestRetryBudgetMiddleware:
    """Tests for RetryBudgetMiddleware."""

    def test_execute_success(self, unlimited_budget):
        """Middleware passes through a successful call."""
        middleware = RetryBudgetMiddleware(
            budget=unlimited_budget,
            provider="test",
            max_attempts=3,
            base_delay=0.0,
            jitter=False,
        )
        result = middleware.execute(lambda: "ok")
        assert result == "ok"

    def test_execute_with_retries(self, unlimited_budget):
        """Middleware retries on transient failures."""
        count = 0

        def flaky():
            nonlocal count
            count += 1
            if count < 3:
                raise NetworkError("transient")
            return "recovered"

        middleware = RetryBudgetMiddleware(
            budget=unlimited_budget,
            max_attempts=5,
            base_delay=0.0,
            jitter=False,
        )
        result = middleware.execute(flaky)
        assert result == "recovered"
        assert count == 3

    def test_execute_exhausted_budget_stops_retries(self, tight_budget):
        """Exhausted budget stops retries early."""
        middleware = RetryBudgetMiddleware(
            budget=tight_budget,
            max_attempts=20,
            base_delay=0.0,
            jitter=False,
        )

        def always_fail():
            raise NetworkError("fail")

        with pytest.raises((RetryExhaustedError, BudgetExhaustedError)):
            middleware.execute(always_fail)

    def test_execute_uses_default_global_budget(self):
        """Middleware uses global budget when none is provided."""
        middleware = RetryBudgetMiddleware(max_attempts=1, base_delay=0.0, jitter=False)
        # Should succeed without raising
        result = middleware.execute(lambda: "ok")
        assert result == "ok"

    def test_execute_provider_override(self, unlimited_budget):
        """Provider name can be overridden per call."""
        stats_before = unlimited_budget.get_stats()

        middleware = RetryBudgetMiddleware(
            budget=unlimited_budget,
            provider="default",
            max_attempts=2,
            base_delay=0.0,
            jitter=False,
        )

        count = 0

        def flaky():
            nonlocal count
            count += 1
            if count < 2:
                raise NetworkError("fail")
            return "ok"

        result = middleware.execute(flaky, provider="anthropic")
        assert result == "ok"

    def test_execute_forwards_args_and_kwargs(self, unlimited_budget):
        """Middleware forwards positional and keyword arguments."""
        def add(a, b, extra=0):
            return a + b + extra

        middleware = RetryBudgetMiddleware(
            budget=unlimited_budget,
            max_attempts=1,
            base_delay=0.0,
        )
        result = middleware.execute(add, args=(3, 4), kwargs={"extra": 5})
        assert result == 12

    def test_non_retryable_error_propagates_immediately(self, unlimited_budget):
        """Non-retryable errors bypass the retry loop."""
        call_count = 0

        def auth_fail():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("bad key")

        middleware = RetryBudgetMiddleware(
            budget=unlimited_budget,
            max_attempts=5,
            base_delay=0.0,
        )

        with pytest.raises(AuthenticationError):
            middleware.execute(auth_fail)

        assert call_count == 1


# --------------------------------------------------------------------------- #
# Integration: budget + retry_call
# --------------------------------------------------------------------------- #


class TestRetryBudgetIntegration:
    """Integration tests combining the budget with the retry engine."""

    def test_budget_shared_across_calls(self):
        """Multiple calls share the same token pool."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=100000.0,
                max_total_retries=3,
                burst_capacity=10,
            )
        )
        config = make_budget_aware_retry_config(
            budget, max_attempts=10, base_delay=0.0
        )

        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise NetworkError("fail")

        # Run first "request" — should exhaust the budget
        with pytest.raises((RetryExhaustedError, BudgetExhaustedError)):
            retry_call(always_fail, config=config)

        # Budget should now be exhausted — second request gets rejected quickly
        call_count_second = 0

        def second_fail():
            nonlocal call_count_second
            call_count_second += 1
            raise NetworkError("fail again")

        with pytest.raises((RetryExhaustedError, BudgetExhaustedError)):
            retry_call(second_fail, config=config)

        # Second request should get very few (if any) retries
        assert call_count_second <= 2

    def test_successful_calls_dont_consume_budget(self):
        """Successful first-attempt calls do not consume any retry tokens."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=100000.0,
                max_total_retries=5,
                burst_capacity=10,
            )
        )
        config = make_budget_aware_retry_config(budget, max_attempts=3, base_delay=0.0)

        # Many successful calls
        for _ in range(20):
            retry_call(lambda: "ok", config=config)

        # Budget is only consumed on retries, not initial calls
        stats = budget.get_stats()
        assert stats["total_consumed"] == 0

    def test_retry_with_rate_limit_respects_budget(self):
        """Retries due to rate limits consume budget tokens."""
        budget = RetryBudget(
            RetryBudgetConfig(
                max_retries_per_second=100000.0,
                max_total_retries=2,
                burst_capacity=10,
            )
        )
        config = make_budget_aware_retry_config(budget, max_attempts=5, base_delay=0.0)

        call_count = 0

        def rate_limited():
            nonlocal call_count
            call_count += 1
            from src.errors import RateLimitError
            raise RateLimitError("too many")

        with pytest.raises((RetryExhaustedError, BudgetExhaustedError)):
            retry_call(rate_limited, config=config)

        # At most 2 retries should have been consumed from budget
        assert budget.get_stats()["total_consumed"] <= 2
