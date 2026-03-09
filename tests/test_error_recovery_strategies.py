"""Tests for the error recovery strategies module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call

import pytest

from src.error_recovery_strategies import (
    AuthRefreshStrategy,
    BaseRecoveryStrategy,
    CircuitBreakerResetStrategy,
    DeadlineAwareStrategy,
    ErrorAggregationStrategy,
    ExponentialDelayStrategy,
    ProviderRotationStrategy,
    RateLimitRecoveryStrategy,
    RecoveryAbortedError,
    RecoveryAction,
    RecoveryChain,
    RecoveryContext,
    make_aggressive_recovery_chain,
    make_standard_recovery_chain,
)
from src.errors import (
    AuthenticationError,
    CircuitOpenError,
    ErrorCategory,
    ErrorContext,
    NetworkError,
    OverloadedError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    RetryExhaustedError,
    TimeoutError_,
)
from src.retry import RetryConfig, retry_call


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_ctx(provider="anthropic", attempt=0, **kwargs) -> RecoveryContext:
    """Create a RecoveryContext with defaults suitable for tests."""
    ctx = RecoveryContext(provider=provider, attempt=attempt, **kwargs)
    return ctx


def make_rate_limit_error(retry_after: float = 0.0) -> RateLimitError:
    return RateLimitError("rate limited", retry_after=retry_after, provider="anthropic")


# --------------------------------------------------------------------------- #
# RecoveryContext tests
# --------------------------------------------------------------------------- #


class TestRecoveryContext:
    """Tests for the RecoveryContext dataclass."""

    def test_defaults(self):
        """Default context is well-formed."""
        ctx = RecoveryContext()
        assert ctx.provider == ""
        assert ctx.attempt == 0
        assert ctx.errors == []
        assert ctx.metadata == {}
        assert ctx.aborted is False
        assert ctx.credentials_refreshed is False

    def test_elapsed_increases(self):
        """elapsed property grows over time."""
        ctx = RecoveryContext()
        time.sleep(0.05)
        assert ctx.elapsed >= 0.04

    def test_deadline_remaining_no_deadline(self):
        """No deadline means infinite remaining."""
        ctx = RecoveryContext(deadline_seconds=0.0)
        assert ctx.deadline_remaining == float("inf")

    def test_deadline_remaining_with_deadline(self):
        """deadline_remaining reflects seconds until deadline."""
        ctx = RecoveryContext(deadline_seconds=time.monotonic() + 100.0)
        remaining = ctx.deadline_remaining
        assert 99.0 < remaining <= 100.0

    def test_is_past_deadline_false_when_no_deadline(self):
        """No deadline means not past deadline."""
        ctx = RecoveryContext(deadline_seconds=0.0)
        assert ctx.is_past_deadline is False

    def test_is_past_deadline_false_before_deadline(self):
        """Not past deadline when deadline is in the future."""
        ctx = RecoveryContext(deadline_seconds=time.monotonic() + 60.0)
        assert ctx.is_past_deadline is False

    def test_is_past_deadline_true_after_deadline(self):
        """Past deadline when deadline has elapsed."""
        ctx = RecoveryContext(deadline_seconds=time.monotonic() - 1.0)
        assert ctx.is_past_deadline is True

    def test_record_error_appends(self):
        """record_error appends to error list."""
        ctx = RecoveryContext()
        err = NetworkError("fail")
        ctx.record_error(err)
        assert len(ctx.errors) == 1
        assert ctx.errors[0] is err
        assert ctx.attempt == 1

    def test_record_error_increments_attempt(self):
        """record_error increments attempt counter."""
        ctx = RecoveryContext()
        ctx.record_error(NetworkError("a"))
        ctx.record_error(NetworkError("b"))
        assert ctx.attempt == 2

    def test_abort_sets_flag(self):
        """abort() sets aborted flag and stores reason."""
        ctx = RecoveryContext()
        ctx.abort("too many failures")
        assert ctx.aborted is True
        assert ctx.abort_reason == "too many failures"


# --------------------------------------------------------------------------- #
# RecoveryAction enum
# --------------------------------------------------------------------------- #


class TestRecoveryAction:
    """Tests for RecoveryAction enum values."""

    def test_values_exist(self):
        """All expected actions exist."""
        assert RecoveryAction.RETRY == "retry"
        assert RecoveryAction.ABORT == "abort"
        assert RecoveryAction.SKIP_RETRY == "skip_retry"
        assert RecoveryAction.WAIT_AND_RETRY == "wait_and_retry"


# --------------------------------------------------------------------------- #
# RateLimitRecoveryStrategy tests
# --------------------------------------------------------------------------- #


class TestRateLimitRecoveryStrategy:
    """Tests for RateLimitRecoveryStrategy."""

    def test_applies_to_rate_limit_error(self):
        """Strategy applies to RateLimitError."""
        strategy = RateLimitRecoveryStrategy()
        ctx = make_ctx()
        assert strategy.applies_to(RateLimitError(), ctx) is True

    def test_does_not_apply_to_network_error(self):
        """Strategy does not apply to NetworkError."""
        strategy = RateLimitRecoveryStrategy()
        ctx = make_ctx()
        assert strategy.applies_to(NetworkError("fail"), ctx) is False

    def test_applies_to_gateway_error_with_rate_limit_category(self):
        """Strategy applies to GatewayError with rate_limit category."""
        strategy = RateLimitRecoveryStrategy()
        ctx = make_ctx()
        err = ProviderError(
            "rate limited",
            status_code=429,
        )
        # ProviderError with 429 uses PROVIDER category, so this should be False
        assert strategy.applies_to(err, ctx) is False

        # But a GatewayError with explicit RATE_LIMIT category should match
        from src.errors import GatewayError
        err2 = GatewayError(
            "rate limited",
            context=ErrorContext(category=ErrorCategory.RATE_LIMIT),
        )
        assert strategy.applies_to(err2, ctx) is True

    def test_sets_extra_delay_from_retry_after(self):
        """Strategy sets extra_delay from Retry-After hint."""
        strategy = RateLimitRecoveryStrategy(max_wait_seconds=120.0)
        ctx = make_ctx()
        ctx.record_error(make_rate_limit_error(retry_after=30.0))
        action = strategy.recover(make_rate_limit_error(retry_after=30.0), ctx)

        assert action == RecoveryAction.WAIT_AND_RETRY
        assert ctx.extra_delay >= 30.0

    def test_caps_extra_delay_at_max_wait(self):
        """Extra delay is capped at max_wait_seconds."""
        strategy = RateLimitRecoveryStrategy(max_wait_seconds=10.0)
        ctx = make_ctx()
        ctx.record_error(make_rate_limit_error(retry_after=500.0))
        action = strategy.recover(make_rate_limit_error(retry_after=500.0), ctx)

        assert ctx.extra_delay <= 10.0

    def test_returns_retry_when_no_retry_after(self):
        """Strategy returns RETRY when no Retry-After hint."""
        strategy = RateLimitRecoveryStrategy()
        ctx = make_ctx()
        ctx.record_error(RateLimitError())
        action = strategy.recover(RateLimitError(), ctx)
        assert action == RecoveryAction.RETRY

    def test_aborts_after_consecutive_limit(self):
        """Strategy aborts after too many consecutive rate limits."""
        strategy = RateLimitRecoveryStrategy(abort_after_consecutive=3)
        ctx = make_ctx()
        # Record 3 rate-limit errors
        for _ in range(3):
            ctx.record_error(RateLimitError())
        action = strategy.recover(RateLimitError(), ctx)
        assert action == RecoveryAction.ABORT
        assert ctx.aborted is True

    def test_does_not_abort_below_threshold(self):
        """Strategy does not abort before reaching the threshold."""
        strategy = RateLimitRecoveryStrategy(abort_after_consecutive=5)
        ctx = make_ctx()
        for _ in range(2):
            ctx.record_error(RateLimitError())
        action = strategy.recover(RateLimitError(), ctx)
        assert action != RecoveryAction.ABORT

    def test_name_property(self):
        """name returns the class name."""
        assert RateLimitRecoveryStrategy().name == "RateLimitRecoveryStrategy"


# --------------------------------------------------------------------------- #
# AuthRefreshStrategy tests
# --------------------------------------------------------------------------- #


class TestAuthRefreshStrategy:
    """Tests for AuthRefreshStrategy."""

    def test_applies_to_auth_error(self):
        """Strategy applies to AuthenticationError."""
        strategy = AuthRefreshStrategy()
        ctx = make_ctx()
        assert strategy.applies_to(AuthenticationError("bad key"), ctx) is True

    def test_does_not_apply_to_network_error(self):
        """Strategy does not apply to NetworkError."""
        strategy = AuthRefreshStrategy()
        ctx = make_ctx()
        assert strategy.applies_to(NetworkError("fail"), ctx) is False

    def test_retries_after_successful_refresh(self):
        """Strategy returns RETRY when refresh succeeds."""
        refresh = MagicMock()
        strategy = AuthRefreshStrategy(refresh_credentials=refresh)
        ctx = make_ctx(provider="anthropic")
        ctx.record_error(AuthenticationError("expired"))
        action = strategy.recover(AuthenticationError("expired"), ctx)

        assert action == RecoveryAction.RETRY
        refresh.assert_called_once_with("anthropic")
        assert ctx.credentials_refreshed is True

    def test_aborts_when_no_refresh_callback(self):
        """Strategy aborts when no refresh callback is configured."""
        strategy = AuthRefreshStrategy(refresh_credentials=None)
        ctx = make_ctx()
        ctx.record_error(AuthenticationError("bad key"))
        action = strategy.recover(AuthenticationError("bad key"), ctx)
        assert action == RecoveryAction.ABORT
        assert ctx.aborted is True

    def test_aborts_after_max_refresh_attempts(self):
        """Strategy aborts after max_refresh_attempts is reached."""
        refresh = MagicMock()
        strategy = AuthRefreshStrategy(
            refresh_credentials=refresh, max_refresh_attempts=1
        )
        ctx = make_ctx()
        ctx.metadata["auth_refresh_count"] = 1  # already refreshed once

        ctx.record_error(AuthenticationError("still bad"))
        action = strategy.recover(AuthenticationError("still bad"), ctx)
        assert action == RecoveryAction.ABORT

    def test_aborts_when_refresh_raises(self):
        """Strategy aborts when the refresh callback fails."""
        def failing_refresh(provider):
            raise RuntimeError("token service unavailable")

        strategy = AuthRefreshStrategy(refresh_credentials=failing_refresh)
        ctx = make_ctx(provider="openai")
        ctx.record_error(AuthenticationError("bad token"))
        action = strategy.recover(AuthenticationError("bad token"), ctx)
        assert action == RecoveryAction.ABORT
        assert ctx.aborted is True

    def test_increments_refresh_count_in_metadata(self):
        """Successful refresh increments auth_refresh_count in metadata."""
        refresh = MagicMock()
        strategy = AuthRefreshStrategy(refresh_credentials=refresh)
        ctx = make_ctx()
        ctx.record_error(AuthenticationError("expired"))
        strategy.recover(AuthenticationError("expired"), ctx)
        assert ctx.metadata.get("auth_refresh_count") == 1


# --------------------------------------------------------------------------- #
# DeadlineAwareStrategy tests
# --------------------------------------------------------------------------- #


class TestDeadlineAwareStrategy:
    """Tests for DeadlineAwareStrategy."""

    def test_applies_to_all_errors(self):
        """Strategy applies to any exception."""
        strategy = DeadlineAwareStrategy()
        ctx = make_ctx()
        assert strategy.applies_to(NetworkError("fail"), ctx) is True
        assert strategy.applies_to(ValueError("bad"), ctx) is True

    def test_returns_retry_when_no_deadline(self):
        """Returns RETRY when no deadline is set."""
        strategy = DeadlineAwareStrategy(deadline_seconds=0.0)
        ctx = make_ctx(deadline_seconds=0.0)
        ctx.record_error(NetworkError("fail"))
        action = strategy.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.RETRY

    def test_returns_retry_when_deadline_is_far(self):
        """Returns RETRY when deadline is in the distant future."""
        deadline = time.monotonic() + 3600.0  # 1 hour from now
        strategy = DeadlineAwareStrategy(
            deadline_seconds=deadline, safety_margin_seconds=0.1
        )
        ctx = make_ctx()
        ctx.record_error(NetworkError("fail"))
        action = strategy.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.RETRY

    def test_aborts_when_deadline_exceeded(self):
        """Aborts when the strategy-level deadline has passed."""
        past_deadline = time.monotonic() - 1.0  # already in the past
        strategy = DeadlineAwareStrategy(
            deadline_seconds=past_deadline, safety_margin_seconds=0.0
        )
        ctx = make_ctx()
        ctx.record_error(NetworkError("fail"))
        action = strategy.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.ABORT
        assert ctx.aborted is True

    def test_aborts_when_context_deadline_exceeded(self):
        """Aborts when the context deadline has passed."""
        strategy = DeadlineAwareStrategy(deadline_seconds=0.0, safety_margin_seconds=0.0)
        ctx = make_ctx(deadline_seconds=time.monotonic() - 1.0)  # past
        ctx.record_error(NetworkError("fail"))
        action = strategy.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.ABORT

    def test_aborts_within_safety_margin(self):
        """Aborts when remaining time is within the safety margin."""
        # Deadline 0.05 seconds in the future, margin 0.5s
        deadline = time.monotonic() + 0.05
        strategy = DeadlineAwareStrategy(
            deadline_seconds=deadline, safety_margin_seconds=0.5
        )
        ctx = make_ctx()
        ctx.record_error(NetworkError("fail"))
        action = strategy.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.ABORT


# --------------------------------------------------------------------------- #
# CircuitBreakerResetStrategy tests
# --------------------------------------------------------------------------- #


class TestCircuitBreakerResetStrategy:
    """Tests for CircuitBreakerResetStrategy."""

    def test_applies_to_circuit_open_error(self):
        """Strategy applies to CircuitOpenError."""
        strategy = CircuitBreakerResetStrategy()
        ctx = make_ctx()
        assert strategy.applies_to(CircuitOpenError("service"), ctx) is True

    def test_does_not_apply_to_network_error(self):
        """Strategy does not apply to NetworkError."""
        strategy = CircuitBreakerResetStrategy()
        ctx = make_ctx()
        assert strategy.applies_to(NetworkError("fail"), ctx) is False

    def test_aborts_with_no_reset_function(self):
        """Strategy aborts when no reset function is configured."""
        strategy = CircuitBreakerResetStrategy(force_reset_func=None)
        ctx = make_ctx()
        ctx.record_error(CircuitOpenError("service"))
        action = strategy.recover(CircuitOpenError("service"), ctx)
        assert action == RecoveryAction.ABORT
        assert ctx.aborted is True

    def test_resets_and_retries_when_reset_function_available(self):
        """Strategy resets the circuit and allows retry."""
        reset = MagicMock()
        strategy = CircuitBreakerResetStrategy(
            force_reset_func=reset,
            force_reset_after_seconds=0.0,  # allow immediately
        )
        ctx = make_ctx()
        ctx.record_error(CircuitOpenError("my-service"))
        action = strategy.recover(CircuitOpenError("my-service"), ctx)
        assert action == RecoveryAction.RETRY
        reset.assert_called_once()

    def test_does_not_reset_before_threshold(self):
        """Strategy does not reset if not enough time has elapsed."""
        reset = MagicMock()
        strategy = CircuitBreakerResetStrategy(
            force_reset_func=reset,
            force_reset_after_seconds=3600.0,  # 1 hour
        )
        ctx = make_ctx()
        ctx.metadata["last_circuit_reset"] = time.monotonic()  # just reset
        ctx.record_error(CircuitOpenError("service"))
        action = strategy.recover(CircuitOpenError("service"), ctx)
        assert action == RecoveryAction.ABORT
        reset.assert_not_called()

    def test_aborts_when_reset_raises(self):
        """Strategy aborts when the reset function raises."""
        def failing_reset(service):
            raise RuntimeError("cannot reset circuit")

        strategy = CircuitBreakerResetStrategy(
            force_reset_func=failing_reset,
            force_reset_after_seconds=0.0,
        )
        ctx = make_ctx()
        ctx.record_error(CircuitOpenError("service"))
        action = strategy.recover(CircuitOpenError("service"), ctx)
        assert action == RecoveryAction.ABORT

    def test_stores_reset_time_in_metadata(self):
        """Successful reset stores timestamp in metadata."""
        reset = MagicMock()
        strategy = CircuitBreakerResetStrategy(
            force_reset_func=reset,
            force_reset_after_seconds=0.0,
        )
        ctx = make_ctx()
        ctx.record_error(CircuitOpenError("service"))
        strategy.recover(CircuitOpenError("service"), ctx)
        assert "last_circuit_reset" in ctx.metadata


# --------------------------------------------------------------------------- #
# ProviderRotationStrategy tests
# --------------------------------------------------------------------------- #


class TestProviderRotationStrategy:
    """Tests for ProviderRotationStrategy."""

    def test_applies_to_retryable_errors(self):
        """Strategy applies to retryable errors with a provider."""
        strategy = ProviderRotationStrategy()
        ctx = make_ctx(provider="anthropic")
        assert strategy.applies_to(NetworkError("fail"), ctx) is True

    def test_does_not_apply_without_provider(self):
        """Strategy skips when no provider is set."""
        strategy = ProviderRotationStrategy()
        ctx = make_ctx(provider="")
        assert strategy.applies_to(NetworkError("fail"), ctx) is False

    def test_does_not_apply_to_non_retryable(self):
        """Strategy skips for non-retryable errors."""
        strategy = ProviderRotationStrategy()
        ctx = make_ctx(provider="anthropic")
        assert strategy.applies_to(AuthenticationError("bad key"), ctx) is False

    def test_increments_failure_count(self):
        """Each recover() call increments the provider failure count."""
        strategy = ProviderRotationStrategy(max_provider_failures=5)
        ctx = make_ctx(provider="anthropic")
        ctx.record_error(NetworkError("fail"))
        strategy.recover(NetworkError("fail"), ctx)
        assert ctx.metadata.get("provider_failures.anthropic") == 1

    def test_suggests_failover_after_threshold(self):
        """Strategy suggests failover after max_provider_failures."""
        strategy = ProviderRotationStrategy(max_provider_failures=2)
        ctx = make_ctx(provider="anthropic")

        # First failure: RETRY
        ctx.record_error(NetworkError("fail"))
        action1 = strategy.recover(NetworkError("fail"), ctx)
        assert action1 == RecoveryAction.RETRY
        assert "suggest_failover" not in ctx.metadata

        # Second failure: suggest failover
        ctx.record_error(NetworkError("fail"))
        action2 = strategy.recover(NetworkError("fail"), ctx)
        assert action2 == RecoveryAction.SKIP_RETRY
        assert ctx.metadata.get("suggest_failover") == "anthropic"

    def test_independent_tracking_per_provider(self):
        """Failure counts are tracked independently per provider."""
        strategy = ProviderRotationStrategy(max_provider_failures=2)
        ctx = make_ctx(provider="anthropic")

        ctx.record_error(NetworkError("fail"))
        strategy.recover(NetworkError("fail"), ctx)

        # Switch provider
        ctx.provider = "openai"
        ctx.record_error(NetworkError("fail"))
        action = strategy.recover(NetworkError("fail"), ctx)
        # openai has only 1 failure — should still RETRY
        assert action == RecoveryAction.RETRY


# --------------------------------------------------------------------------- #
# ErrorAggregationStrategy tests
# --------------------------------------------------------------------------- #


class TestErrorAggregationStrategy:
    """Tests for ErrorAggregationStrategy."""

    def test_applies_to_all_errors(self):
        """Strategy applies universally."""
        strategy = ErrorAggregationStrategy()
        ctx = make_ctx()
        assert strategy.applies_to(NetworkError("fail"), ctx) is True
        assert strategy.applies_to(ValueError("bad"), ctx) is True

    def test_always_returns_retry(self):
        """Strategy always returns RETRY."""
        strategy = ErrorAggregationStrategy()
        ctx = make_ctx()
        ctx.record_error(NetworkError("fail"))
        action = strategy.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.RETRY

    def test_initializes_error_summary(self):
        """First error initialises the summary in metadata."""
        strategy = ErrorAggregationStrategy()
        ctx = make_ctx()
        ctx.record_error(NetworkError("first"))
        strategy.recover(NetworkError("first"), ctx)
        assert "error_summary" in ctx.metadata
        summary = ctx.metadata["error_summary"]
        assert summary["total"] == 1

    def test_tracks_category_counts(self):
        """Strategy tracks error categories."""
        strategy = ErrorAggregationStrategy()
        ctx = make_ctx()

        for err in [NetworkError("a"), NetworkError("b"), RateLimitError()]:
            ctx.record_error(err)
            strategy.recover(err, ctx)

        cats = ctx.metadata["error_summary"]["categories"]
        assert cats.get("network", 0) == 2
        assert cats.get("rate_limit", 0) == 1

    def test_tracks_first_and_last_error(self):
        """Summary records first and last errors."""
        strategy = ErrorAggregationStrategy()
        ctx = make_ctx()

        first = NetworkError("first error")
        ctx.record_error(first)
        strategy.recover(first, ctx)

        last = RateLimitError("last error")
        ctx.record_error(last)
        strategy.recover(last, ctx)

        summary = ctx.metadata["error_summary"]
        assert summary["first_error"]["type"] == "NetworkError"
        assert summary["last_error"]["type"] == "RateLimitError"

    def test_deduplicates_unique_messages(self):
        """Identical messages are not duplicated in unique_messages."""
        strategy = ErrorAggregationStrategy(max_errors=20)
        ctx = make_ctx()
        err = NetworkError("same message")

        for _ in range(5):
            ctx.record_error(err)
            strategy.recover(err, ctx)

        unique_msgs = ctx.metadata["error_summary"]["unique_messages"]
        assert unique_msgs.count("same message") == 1

    def test_respects_max_errors_limit(self):
        """Unique messages list is capped at max_errors."""
        strategy = ErrorAggregationStrategy(max_errors=3)
        ctx = make_ctx()

        for i in range(10):
            err = NetworkError(f"unique message {i}")
            ctx.record_error(err)
            strategy.recover(err, ctx)

        assert len(ctx.metadata["error_summary"]["unique_messages"]) <= 3


# --------------------------------------------------------------------------- #
# ExponentialDelayStrategy tests
# --------------------------------------------------------------------------- #


class TestExponentialDelayStrategy:
    """Tests for ExponentialDelayStrategy."""

    def test_applies_to_retryable_errors(self):
        """Strategy applies to retryable exceptions."""
        strategy = ExponentialDelayStrategy()
        ctx = make_ctx()
        assert strategy.applies_to(NetworkError("fail"), ctx) is True

    def test_does_not_apply_to_non_retryable(self):
        """Strategy does not apply to non-retryable errors."""
        strategy = ExponentialDelayStrategy()
        ctx = make_ctx()
        assert strategy.applies_to(AuthenticationError("bad key"), ctx) is False

    def test_returns_wait_and_retry(self):
        """Strategy returns WAIT_AND_RETRY."""
        strategy = ExponentialDelayStrategy(base_extra_delay=0.1, multiplier=2.0)
        ctx = make_ctx(attempt=1)
        ctx.record_error(NetworkError("fail"))
        action = strategy.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.WAIT_AND_RETRY

    def test_extra_delay_grows_exponentially(self):
        """Extra delay increases exponentially with attempt number."""
        strategy = ExponentialDelayStrategy(
            base_extra_delay=1.0, multiplier=2.0, max_extra_delay=1000.0
        )
        ctx1 = make_ctx(attempt=1)
        ctx2 = make_ctx(attempt=2)
        ctx3 = make_ctx(attempt=3)

        err = NetworkError("fail")
        ctx1.record_error(err)
        ctx2.record_error(err)
        ctx3.record_error(err)

        strategy.recover(err, ctx1)
        strategy.recover(err, ctx2)
        strategy.recover(err, ctx3)

        assert ctx1.extra_delay <= ctx2.extra_delay <= ctx3.extra_delay

    def test_extra_delay_capped_at_max(self):
        """Extra delay is capped at max_extra_delay."""
        strategy = ExponentialDelayStrategy(
            base_extra_delay=100.0, multiplier=10.0, max_extra_delay=30.0
        )
        ctx = make_ctx(attempt=5)
        ctx.record_error(NetworkError("fail"))
        strategy.recover(NetworkError("fail"), ctx)
        assert ctx.extra_delay <= 30.0

    def test_does_not_reduce_existing_delay(self):
        """Strategy only increases extra_delay, never reduces it."""
        strategy = ExponentialDelayStrategy(base_extra_delay=0.01, multiplier=2.0)
        ctx = make_ctx(attempt=1)
        ctx.extra_delay = 100.0  # already very large
        ctx.record_error(NetworkError("fail"))
        strategy.recover(NetworkError("fail"), ctx)
        assert ctx.extra_delay >= 100.0  # should not be reduced


# --------------------------------------------------------------------------- #
# RecoveryChain tests
# --------------------------------------------------------------------------- #


class TestRecoveryChain:
    """Tests for RecoveryChain composition."""

    def test_empty_chain_returns_default(self):
        """Empty chain returns the default action."""
        chain = RecoveryChain(default_action=RecoveryAction.RETRY)
        ctx = make_ctx()
        action = chain.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.RETRY

    def test_chain_stops_on_abort(self):
        """Chain stops at the first ABORT decision."""
        abort_strategy = MagicMock(spec=BaseRecoveryStrategy)
        abort_strategy.name = "AbortStrategy"
        abort_strategy.applies_to.return_value = True
        abort_strategy.recover.return_value = RecoveryAction.ABORT

        second_strategy = MagicMock(spec=BaseRecoveryStrategy)
        second_strategy.name = "SecondStrategy"
        second_strategy.applies_to.return_value = True
        second_strategy.recover.return_value = RecoveryAction.RETRY

        chain = RecoveryChain([abort_strategy, second_strategy])
        ctx = make_ctx()
        action = chain.recover(NetworkError("fail"), ctx)

        assert action == RecoveryAction.ABORT
        second_strategy.recover.assert_not_called()

    def test_chain_stops_on_skip_retry(self):
        """Chain stops at the first SKIP_RETRY decision."""
        skip_strategy = MagicMock(spec=BaseRecoveryStrategy)
        skip_strategy.name = "SkipStrategy"
        skip_strategy.applies_to.return_value = True
        skip_strategy.recover.return_value = RecoveryAction.SKIP_RETRY

        second = MagicMock(spec=BaseRecoveryStrategy)
        second.name = "SecondStrategy"
        second.applies_to.return_value = True
        second.recover.return_value = RecoveryAction.RETRY

        chain = RecoveryChain([skip_strategy, second])
        ctx = make_ctx()
        action = chain.recover(NetworkError("fail"), ctx)

        assert action == RecoveryAction.SKIP_RETRY
        second.recover.assert_not_called()

    def test_chain_continues_through_retry_and_wait(self):
        """Chain continues through RETRY and WAIT_AND_RETRY decisions."""
        strategy1 = MagicMock(spec=BaseRecoveryStrategy)
        strategy1.name = "Strategy1"
        strategy1.applies_to.return_value = True
        strategy1.recover.return_value = RecoveryAction.RETRY

        strategy2 = MagicMock(spec=BaseRecoveryStrategy)
        strategy2.name = "Strategy2"
        strategy2.applies_to.return_value = True
        strategy2.recover.return_value = RecoveryAction.WAIT_AND_RETRY

        strategy3 = MagicMock(spec=BaseRecoveryStrategy)
        strategy3.name = "Strategy3"
        strategy3.applies_to.return_value = True
        strategy3.recover.return_value = RecoveryAction.RETRY

        chain = RecoveryChain([strategy1, strategy2, strategy3])
        ctx = make_ctx()
        chain.recover(NetworkError("fail"), ctx)

        # All three strategies should have been called
        strategy1.recover.assert_called_once()
        strategy2.recover.assert_called_once()
        strategy3.recover.assert_called_once()

    def test_chain_skips_non_applicable_strategies(self):
        """Strategies whose applies_to returns False are skipped."""
        non_applicable = MagicMock(spec=BaseRecoveryStrategy)
        non_applicable.name = "NonApplicable"
        non_applicable.applies_to.return_value = False
        non_applicable.recover.return_value = RecoveryAction.ABORT

        applicable = MagicMock(spec=BaseRecoveryStrategy)
        applicable.name = "Applicable"
        applicable.applies_to.return_value = True
        applicable.recover.return_value = RecoveryAction.RETRY

        chain = RecoveryChain([non_applicable, applicable])
        ctx = make_ctx()
        action = chain.recover(NetworkError("fail"), ctx)

        non_applicable.recover.assert_not_called()
        applicable.recover.assert_called_once()
        assert action == RecoveryAction.RETRY

    def test_add_returns_self_for_chaining(self):
        """add() returns the chain for method chaining."""
        chain = RecoveryChain()
        result = chain.add(DeadlineAwareStrategy())
        assert result is chain

    def test_chain_length(self):
        """len() reflects the number of strategies."""
        chain = RecoveryChain([DeadlineAwareStrategy(), ErrorAggregationStrategy()])
        assert len(chain) == 2

    def test_record_error_called_per_recover(self):
        """recover() calls ctx.record_error() for each invocation."""
        chain = RecoveryChain([ErrorAggregationStrategy()])
        ctx = make_ctx()
        chain.recover(NetworkError("a"), ctx)
        chain.recover(NetworkError("b"), ctx)
        assert ctx.attempt == 2
        assert len(ctx.errors) == 2

    def test_aborted_context_returns_abort(self):
        """Chain returns ABORT if context.aborted is set by a strategy."""
        # Strategy that sets ctx.aborted but returns RETRY
        class AbortingStrategy(BaseRecoveryStrategy):
            def recover(self, error, ctx):
                ctx.abort("set by strategy")
                return RecoveryAction.RETRY  # returns RETRY but sets aborted

        chain = RecoveryChain([AbortingStrategy()])
        ctx = make_ctx()
        action = chain.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.ABORT

    def test_repr(self):
        """repr includes strategy names."""
        chain = RecoveryChain(
            [DeadlineAwareStrategy(), ErrorAggregationStrategy()]
        )
        r = repr(chain)
        assert "DeadlineAwareStrategy" in r
        assert "ErrorAggregationStrategy" in r


# --------------------------------------------------------------------------- #
# RecoveryChain.make_on_retry_callback tests
# --------------------------------------------------------------------------- #


class TestRecoveryChainCallback:
    """Tests for the on_retry callback generated by RecoveryChain."""

    def test_callback_allows_retry_on_retry_action(self):
        """Callback does not raise when action is RETRY."""
        chain = RecoveryChain([ErrorAggregationStrategy()])
        ctx = make_ctx()
        callback = chain.make_on_retry_callback(ctx)
        # Should not raise
        callback(1, 0.0, NetworkError("fail"))

    def test_callback_raises_recovery_aborted_on_abort(self):
        """Callback raises RecoveryAbortedError when action is ABORT."""
        # Deadline in the past → DeadlineAwareStrategy aborts
        strategy = DeadlineAwareStrategy(
            deadline_seconds=time.monotonic() - 1.0,
            safety_margin_seconds=0.0,
        )
        chain = RecoveryChain([strategy])
        ctx = make_ctx()
        callback = chain.make_on_retry_callback(ctx)

        with pytest.raises(RecoveryAbortedError) as exc_info:
            callback(1, 0.0, NetworkError("fail"))

        err = exc_info.value
        assert err.original_error is not None
        assert err.context is ctx

    def test_callback_sleeps_extra_delay(self):
        """Callback sleeps the extra_delay set by strategies."""
        class DelayingStrategy(BaseRecoveryStrategy):
            def recover(self, error, ctx):
                ctx.extra_delay = 0.05
                return RecoveryAction.WAIT_AND_RETRY

        chain = RecoveryChain([DelayingStrategy()])
        ctx = make_ctx()
        callback = chain.make_on_retry_callback(ctx)

        start = time.monotonic()
        callback(1, 0.0, NetworkError("fail"))
        elapsed = time.monotonic() - start

        assert elapsed >= 0.04  # Slept at least 0.05s
        assert ctx.extra_delay == 0.0  # Reset after sleep

    def test_callback_resets_extra_delay_after_sleeping(self):
        """extra_delay is reset to 0 after the sleep."""
        chain = RecoveryChain([RateLimitRecoveryStrategy()])
        ctx = make_ctx()
        ctx.extra_delay = 0.01  # pre-set a small delay

        callback = chain.make_on_retry_callback(ctx)
        callback(1, 0.0, RateLimitError("rate limited"))

        assert ctx.extra_delay == 0.0


# --------------------------------------------------------------------------- #
# RecoveryAbortedError tests
# --------------------------------------------------------------------------- #


class TestRecoveryAbortedError:
    """Tests for RecoveryAbortedError."""

    def test_basic_construction(self):
        """RecoveryAbortedError can be constructed with just a message."""
        err = RecoveryAbortedError("aborted")
        assert str(err) == "aborted"
        assert err.original_error is None
        assert err.context is None

    def test_with_context(self):
        """RecoveryAbortedError carries original_error and context."""
        original = NetworkError("network failure")
        ctx = make_ctx()
        err = RecoveryAbortedError("aborted", original_error=original, context=ctx)
        assert err.original_error is original
        assert err.context is ctx


# --------------------------------------------------------------------------- #
# Preset factory tests
# --------------------------------------------------------------------------- #


class TestPresetFactories:
    """Tests for make_standard_recovery_chain and make_aggressive_recovery_chain."""

    def test_standard_chain_has_expected_strategies(self):
        """Standard chain includes deadline, rate-limit, circuit, aggregation."""
        chain = make_standard_recovery_chain()
        # At minimum: deadline, rate-limit, circuit-breaker, aggregation
        assert len(chain) >= 3

    def test_standard_chain_with_refresh_callback(self):
        """Standard chain includes auth refresh when callback is provided."""
        refresh = MagicMock()
        chain = make_standard_recovery_chain(refresh_credentials=refresh)
        # Includes auth refresh strategy
        strategy_types = [type(s).__name__ for s in chain.strategies]
        assert "AuthRefreshStrategy" in strategy_types

    def test_standard_chain_without_refresh_excludes_auth(self):
        """Standard chain excludes auth refresh when no callback."""
        chain = make_standard_recovery_chain(refresh_credentials=None)
        strategy_types = [type(s).__name__ for s in chain.strategies]
        assert "AuthRefreshStrategy" not in strategy_types

    def test_standard_chain_handles_rate_limit(self):
        """Standard chain handles RateLimitError correctly."""
        chain = make_standard_recovery_chain(max_rate_limit_wait=5.0)
        ctx = make_ctx(provider="anthropic")
        err = RateLimitError(retry_after=2.0)
        action = chain.recover(err, ctx)
        assert action == RecoveryAction.WAIT_AND_RETRY
        assert ctx.extra_delay >= 2.0

    def test_standard_chain_aborts_on_exceeded_deadline(self):
        """Standard chain aborts immediately when deadline is exceeded."""
        chain = make_standard_recovery_chain(
            deadline_seconds=time.monotonic() - 1.0,
        )
        ctx = make_ctx()
        action = chain.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.ABORT

    def test_aggressive_chain_rotates_quickly(self):
        """Aggressive chain suggests failover after 1 failure."""
        chain = make_aggressive_recovery_chain(max_provider_failures=1)
        ctx = make_ctx(provider="anthropic")

        # First failure should already suggest failover
        action = chain.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.SKIP_RETRY
        assert ctx.metadata.get("suggest_failover") == "anthropic"

    def test_aggressive_chain_with_deadline(self):
        """Aggressive chain respects deadline."""
        chain = make_aggressive_recovery_chain(
            deadline_seconds=time.monotonic() - 1.0,  # past
        )
        ctx = make_ctx()
        action = chain.recover(NetworkError("fail"), ctx)
        assert action == RecoveryAction.ABORT


# --------------------------------------------------------------------------- #
# Integration: RecoveryChain with retry_call
# --------------------------------------------------------------------------- #


class TestRecoveryChainIntegration:
    """Integration tests combining RecoveryChain with the retry engine."""

    def test_chain_callback_with_retry_call_success(self):
        """Chain callback integrates with retry_call for a successful call."""
        chain = RecoveryChain([ErrorAggregationStrategy()])
        ctx = RecoveryContext(provider="anthropic", max_attempts=5)
        callback = chain.make_on_retry_callback(ctx)

        count = 0

        def flaky():
            nonlocal count
            count += 1
            if count < 3:
                raise NetworkError("transient")
            return "ok"

        config = RetryConfig(
            max_attempts=5,
            base_delay=0.01,
            jitter=False,
            on_retry=callback,
        )
        result = retry_call(flaky, config=config)
        assert result == "ok"
        assert count == 3

    def test_chain_aborts_retry_call_via_callback(self):
        """Chain abort causes retry_call to propagate the abort error."""
        # Strategy that aborts immediately
        class ImmediateAbort(BaseRecoveryStrategy):
            def recover(self, error, ctx):
                ctx.abort("immediate abort")
                return RecoveryAction.ABORT

        chain = RecoveryChain([ImmediateAbort()])
        ctx = RecoveryContext(provider="anthropic")
        callback = chain.make_on_retry_callback(ctx)

        config = RetryConfig(
            max_attempts=5,
            base_delay=0.01,
            jitter=False,
            on_retry=callback,
        )

        def always_fail_net():
            raise NetworkError("fail")

        with pytest.raises(RecoveryAbortedError):
            retry_call(always_fail_net, config=config)

    def test_error_summary_accumulated_across_retries(self):
        """Error aggregation strategy accumulates across retry_call iterations."""
        chain = RecoveryChain([ErrorAggregationStrategy()])
        ctx = RecoveryContext(provider="test", max_attempts=5)
        callback = chain.make_on_retry_callback(ctx)

        count = 0

        def flaky():
            nonlocal count
            count += 1
            if count < 4:
                raise NetworkError(f"error {count}")
            return "recovered"

        config = RetryConfig(
            max_attempts=5,
            base_delay=0.0,
            jitter=False,
            on_retry=callback,
        )
        retry_call(flaky, config=config)

        # 3 errors should have been aggregated
        summary = ctx.metadata.get("error_summary", {})
        assert summary.get("total", 0) == 3

    def test_rate_limit_strategy_applies_extra_delay(self):
        """Rate-limit strategy delays are observed during retry_call."""
        chain = make_standard_recovery_chain()
        ctx = RecoveryContext(provider="anthropic")
        callback = chain.make_on_retry_callback(ctx)

        count = 0

        def rate_limited():
            nonlocal count
            count += 1
            if count == 1:
                raise RateLimitError(retry_after=0.05)
            return "ok"

        config = RetryConfig(
            max_attempts=3,
            base_delay=0.0,
            jitter=False,
            on_retry=callback,
        )

        start = time.monotonic()
        result = retry_call(rate_limited, config=config)
        elapsed = time.monotonic() - start

        assert result == "ok"
        assert elapsed >= 0.04  # Slept for the retry_after hint

    def test_deadline_strategy_stops_long_running_retry(self):
        """Deadline strategy prevents retry_call from running past the deadline."""
        deadline = time.monotonic() + 0.1  # 100ms deadline
        strategy = DeadlineAwareStrategy(
            deadline_seconds=deadline, safety_margin_seconds=0.0
        )
        chain = RecoveryChain([strategy])
        ctx = RecoveryContext(provider="test")
        callback = chain.make_on_retry_callback(ctx)

        config = RetryConfig(
            max_attempts=100,
            base_delay=0.05,  # Each retry sleeps 50ms
            jitter=False,
            on_retry=callback,
        )

        def always_fail_deadline():
            raise NetworkError("fail")

        with pytest.raises((RecoveryAbortedError, Exception)):
            retry_call(always_fail_deadline, config=config)

        # Should have been stopped well before 100 attempts
        assert ctx.attempt < 10
