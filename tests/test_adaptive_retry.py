"""Tests for adaptive retry and retry policy features."""

import threading
import time

import pytest

from src.errors import (
    NetworkError,
    OverloadedError,
    ProviderUnavailableError,
    RateLimitError,
    RetryExhaustedError,
)
from src.retry import (
    AdaptiveRetryConfig,
    BackoffStrategy,
    CircuitBreakerConfig,
    RetryConfig,
    RetryPolicy,
    clear_retry_policies,
    get_retry_policy,
    list_retry_policies,
    register_retry_policy,
    retry_call,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up retry policy registry after each test."""
    clear_retry_policies()
    yield
    clear_retry_policies()


# --------------------------------------------------------------------------- #
# AdaptiveRetryConfig tests
# --------------------------------------------------------------------------- #


class TestAdaptiveRetryConfig:
    """Tests for AdaptiveRetryConfig."""

    def test_default_config(self):
        """Default adaptive config uses the base config."""
        base = RetryConfig(max_attempts=3, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(base_config=base)

        effective = adaptive.get_config()
        assert effective.max_attempts == 3
        assert effective.base_delay == 1.0

    def test_rate_limit_increases_delay(self):
        """Recording rate limits increases the delay multiplier."""
        base = RetryConfig(max_attempts=3, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(
            base_config=base, rate_limit_multiplier=2.0
        )

        adaptive.record_rate_limit()
        effective = adaptive.get_config()
        assert effective.base_delay == 2.0  # 1.0 * 2.0

        adaptive.record_rate_limit()
        effective = adaptive.get_config()
        assert effective.base_delay == 4.0  # 1.0 * 4.0

    def test_rate_limit_caps_delay(self):
        """Delay multiplier caps at 8x."""
        base = RetryConfig(max_attempts=3, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(
            base_config=base, rate_limit_multiplier=3.0
        )

        # 3x -> 9x, but caps at 8x
        adaptive.record_rate_limit()
        adaptive.record_rate_limit()
        adaptive.record_rate_limit()

        effective = adaptive.get_config()
        assert effective.base_delay == 8.0  # Capped at 8x

    def test_overload_reduces_attempts(self):
        """Recording overload reduces max attempts."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(base_config=base, min_attempts=1)

        adaptive.record_overload()
        effective = adaptive.get_config()
        assert effective.max_attempts == 4

        adaptive.record_overload()
        effective = adaptive.get_config()
        assert effective.max_attempts == 3

    def test_overload_respects_min_attempts(self):
        """Max attempts never goes below min_attempts."""
        base = RetryConfig(max_attempts=2, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(base_config=base, min_attempts=1)

        adaptive.record_overload()
        adaptive.record_overload()
        adaptive.record_overload()

        effective = adaptive.get_config()
        assert effective.max_attempts >= 1

    def test_success_restores_defaults(self):
        """Consecutive successes restore default configuration."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(
            base_config=base,
            recovery_success_count=3,
        )

        # Degrade the config
        adaptive.record_rate_limit()
        adaptive.record_overload()

        # Record enough successes to recover
        for _ in range(10):
            adaptive.record_success()

        effective = adaptive.get_config()
        # Should be back to or near defaults
        assert effective.max_attempts >= base.max_attempts
        assert effective.base_delay <= base.base_delay

    def test_high_error_rate_reduces_attempts(self):
        """High error rate reduces effective max attempts."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(
            base_config=base,
            error_rate_threshold=0.5,
            min_attempts=1,
        )

        effective = adaptive.get_config(error_rate=1.0)
        assert effective.max_attempts < 5

    def test_low_success_rate_reduces_attempts(self):
        """Low success rate reduces effective max attempts."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(
            base_config=base,
            success_rate_threshold=0.7,
            min_attempts=1,
        )

        effective = adaptive.get_config(success_rate=0.5)
        assert effective.max_attempts < 5

    def test_very_high_error_rate_switches_to_linear(self):
        """Very high error rate switches to linear backoff."""
        base = RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            jitter=False,
        )
        adaptive = AdaptiveRetryConfig(
            base_config=base,
            error_rate_threshold=0.5,
        )

        effective = adaptive.get_config(error_rate=2.0)
        assert effective.backoff_strategy == BackoffStrategy.LINEAR

    def test_record_error_dispatches_to_rate_limit(self):
        """record_error detects RateLimitError and calls record_rate_limit."""
        base = RetryConfig(max_attempts=3, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(
            base_config=base, rate_limit_multiplier=2.0
        )

        adaptive.record_error(RateLimitError())
        effective = adaptive.get_config()
        assert effective.base_delay == 2.0

    def test_record_error_dispatches_to_overload(self):
        """record_error detects OverloadedError and calls record_overload."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(base_config=base, min_attempts=1)

        adaptive.record_error(OverloadedError(provider="test"))
        effective = adaptive.get_config()
        assert effective.max_attempts < 5

    def test_record_error_dispatches_to_provider_unavailable(self):
        """record_error detects ProviderUnavailableError."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(base_config=base, min_attempts=1)

        adaptive.record_error(ProviderUnavailableError(provider="test"))
        effective = adaptive.get_config()
        assert effective.max_attempts < 5

    def test_record_error_generic_resets_success(self):
        """Generic errors reset the success counter."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(
            base_config=base,
            recovery_success_count=3,
        )

        # Build up successes
        for _ in range(3):
            adaptive.record_success()
        # A generic error resets
        adaptive.record_error(NetworkError("fail"))

        state = adaptive.get_state()
        assert state["consecutive_successes"] == 0

    def test_reset(self):
        """reset() restores all adaptive state to defaults."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(base_config=base)

        adaptive.record_rate_limit()
        adaptive.record_overload()
        adaptive.reset()

        state = adaptive.get_state()
        assert state["current_max_attempts"] == 5
        assert state["current_delay_multiplier"] == 1.0
        assert state["current_strategy"] == "exponential"

    def test_get_state(self):
        """get_state() returns current adaptive parameters."""
        base = RetryConfig(max_attempts=5, base_delay=2.0, jitter=False)
        adaptive = AdaptiveRetryConfig(base_config=base)

        state = adaptive.get_state()
        assert state["current_max_attempts"] == 5
        assert state["current_delay_multiplier"] == 1.0
        assert state["base_max_attempts"] == 5
        assert state["base_delay"] == 2.0

    def test_delay_capped_at_max_delay(self):
        """Adaptive delay is capped at base_config.max_delay."""
        base = RetryConfig(
            max_attempts=3, base_delay=10.0, max_delay=20.0, jitter=False
        )
        adaptive = AdaptiveRetryConfig(
            base_config=base, rate_limit_multiplier=3.0
        )

        # 10.0 * 3.0 = 30.0, but should be capped at max_delay
        adaptive.record_rate_limit()
        effective = adaptive.get_config()
        assert effective.base_delay == 20.0  # Capped at max_delay

    def test_thread_safety(self):
        """AdaptiveRetryConfig is thread-safe."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(base_config=base)
        errors = []

        def mutate():
            try:
                for _ in range(50):
                    adaptive.record_rate_limit()
                    adaptive.record_success()
                    adaptive.record_overload()
                    adaptive.get_config()
                    adaptive.get_state()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mutate) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_preserves_base_config_options(self):
        """Adaptive config preserves non-adapted options from base."""
        base = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            jitter=True,
            jitter_range=(0.8, 1.2),
            retry_on_status={429, 503},
            total_timeout=30.0,
        )
        adaptive = AdaptiveRetryConfig(base_config=base)

        effective = adaptive.get_config()
        assert effective.jitter is True
        assert effective.jitter_range == (0.8, 1.2)
        assert effective.retry_on_status == {429, 503}
        assert effective.total_timeout == 30.0


# --------------------------------------------------------------------------- #
# RetryPolicy tests
# --------------------------------------------------------------------------- #


class TestRetryPolicy:
    """Tests for the RetryPolicy class."""

    def test_basic_policy(self):
        """Basic policy returns static retry config."""
        policy = RetryPolicy(
            name="test",
            retry_config=RetryConfig(max_attempts=5, base_delay=2.0),
        )
        assert policy.name == "test"
        assert policy.get_effective_config().max_attempts == 5
        assert policy.get_effective_config().base_delay == 2.0

    def test_policy_with_adaptive(self):
        """Policy with adaptive config adapts based on conditions."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        policy = RetryPolicy(
            name="adaptive-test",
            retry_config=base,
            adaptive=AdaptiveRetryConfig(
                base_config=base,
                error_rate_threshold=0.5,
                min_attempts=1,
            ),
        )

        # Normal conditions
        config = policy.get_effective_config(error_rate=0.0, success_rate=1.0)
        assert config.max_attempts == 5

        # High error rate
        config = policy.get_effective_config(error_rate=1.0, success_rate=0.5)
        assert config.max_attempts < 5

    def test_record_outcome_success(self):
        """Recording a success updates adaptive state."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(
            base_config=base, recovery_success_count=2
        )
        policy = RetryPolicy(
            name="test", retry_config=base, adaptive=adaptive
        )

        policy.record_outcome()  # success
        assert adaptive.get_state()["consecutive_successes"] == 1

    def test_record_outcome_error(self):
        """Recording an error updates adaptive state."""
        base = RetryConfig(max_attempts=5, base_delay=1.0, jitter=False)
        adaptive = AdaptiveRetryConfig(
            base_config=base, rate_limit_multiplier=2.0
        )
        policy = RetryPolicy(
            name="test", retry_config=base, adaptive=adaptive
        )

        policy.record_outcome(error=RateLimitError())
        assert adaptive.get_state()["current_delay_multiplier"] == 2.0

    def test_record_outcome_no_adaptive(self):
        """Recording outcome with no adaptive config is a no-op."""
        policy = RetryPolicy(
            name="test",
            retry_config=RetryConfig(max_attempts=3),
        )
        # Should not raise
        policy.record_outcome()
        policy.record_outcome(error=NetworkError("fail"))

    def test_policy_with_circuit_breaker_config(self):
        """Policy can carry circuit breaker config."""
        policy = RetryPolicy(
            name="test",
            retry_config=RetryConfig(max_attempts=3),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=10, reset_timeout=60.0
            ),
        )
        assert policy.circuit_breaker_config is not None
        assert policy.circuit_breaker_config.failure_threshold == 10


# --------------------------------------------------------------------------- #
# Retry policy registry tests
# --------------------------------------------------------------------------- #


class TestRetryPolicyRegistry:
    """Tests for the global retry policy registry."""

    def test_register_and_get(self):
        """Register and retrieve a policy."""
        policy = RetryPolicy(
            name="anthropic-messages",
            retry_config=RetryConfig(max_attempts=5),
        )
        register_retry_policy(policy)

        retrieved = get_retry_policy("anthropic-messages")
        assert retrieved is policy

    def test_get_nonexistent(self):
        """Getting a non-existent policy returns None."""
        assert get_retry_policy("does-not-exist") is None

    def test_list_policies(self):
        """list_retry_policies returns all registered policies."""
        register_retry_policy(
            RetryPolicy(name="a", retry_config=RetryConfig())
        )
        register_retry_policy(
            RetryPolicy(name="b", retry_config=RetryConfig())
        )

        policies = list_retry_policies()
        assert "a" in policies
        assert "b" in policies

    def test_clear_policies(self):
        """clear_retry_policies removes all policies."""
        register_retry_policy(
            RetryPolicy(name="a", retry_config=RetryConfig())
        )
        clear_retry_policies()
        assert list_retry_policies() == {}

    def test_overwrite_policy(self):
        """Registering with the same name overwrites."""
        register_retry_policy(
            RetryPolicy(name="test", retry_config=RetryConfig(max_attempts=3))
        )
        register_retry_policy(
            RetryPolicy(name="test", retry_config=RetryConfig(max_attempts=7))
        )

        policy = get_retry_policy("test")
        assert policy.retry_config.max_attempts == 7


# --------------------------------------------------------------------------- #
# Integration: Adaptive retry with retry_call
# --------------------------------------------------------------------------- #


class TestAdaptiveRetryIntegration:
    """Integration tests for adaptive retry with the core retry engine."""

    def test_adaptive_config_with_retry_call(self):
        """Adaptive config produces valid configs for retry_call."""
        base = RetryConfig(
            max_attempts=3, base_delay=0.01, jitter=False
        )
        adaptive = AdaptiveRetryConfig(base_config=base)

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("transient")
            return "ok"

        effective = adaptive.get_config()
        result = retry_call(flaky, config=effective)
        assert result == "ok"
        assert call_count == 3

    def test_adaptive_reduces_attempts_under_load(self):
        """Under high error rate, adaptive config reduces retry attempts."""
        base = RetryConfig(
            max_attempts=5, base_delay=0.01, jitter=False
        )
        adaptive = AdaptiveRetryConfig(
            base_config=base,
            error_rate_threshold=0.1,
            min_attempts=1,
        )

        # Get config under high error rate
        effective = adaptive.get_config(error_rate=1.0, success_rate=0.3)
        # Should have reduced attempts
        assert effective.max_attempts < 5

        # Should still work with retry_call
        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise NetworkError("fail")

        with pytest.raises(RetryExhaustedError):
            retry_call(always_fail, config=effective)

        assert call_count == effective.max_attempts

    def test_policy_get_effective_and_retry(self):
        """RetryPolicy.get_effective_config works with retry_call."""
        policy = RetryPolicy(
            name="test",
            retry_config=RetryConfig(
                max_attempts=3, base_delay=0.01, jitter=False
            ),
        )

        config = policy.get_effective_config()
        result = retry_call(lambda: "ok", config=config)
        assert result == "ok"

    def test_adaptive_recovery_flow(self):
        """Full flow: degrade under load, then recover."""
        base = RetryConfig(max_attempts=5, base_delay=0.01, jitter=False)
        adaptive = AdaptiveRetryConfig(
            base_config=base,
            rate_limit_multiplier=2.0,
            recovery_success_count=3,
            min_attempts=1,
        )

        # Phase 1: Rate limiting degrades the config
        adaptive.record_rate_limit()
        adaptive.record_rate_limit()
        state = adaptive.get_state()
        assert state["current_delay_multiplier"] == 4.0

        # Phase 2: Successes trigger recovery
        for _ in range(10):
            adaptive.record_success()

        state = adaptive.get_state()
        # Should be recovering toward defaults
        assert state["current_delay_multiplier"] < 4.0
