"""Tests for the src.config.testing helper module."""

import pytest

from src.config.testing import (
    assert_config_invalid,
    assert_config_valid,
    assert_has_error_at,
    assert_has_warning_at,
    make_model,
    make_provider,
    make_valid_config,
    run_builtin_suite,
)
from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig
from src.validation.validator import ConfigValidator, ValidationResult


# ---------------------------------------------------------------------------
# make_model tests
# ---------------------------------------------------------------------------


class TestMakeModel:
    """Tests for the make_model factory."""

    def test_default_model(self):
        """Test creating a model with default values."""
        model = make_model()
        assert model.name == "test-model"
        assert model.max_tokens == 4096
        assert model.supports_streaming is True
        assert model.supports_tools is False
        assert model.supports_vision is False

    def test_custom_name(self):
        """Test creating a model with a custom name."""
        model = make_model(name="gpt-4o")
        assert model.name == "gpt-4o"

    def test_custom_max_tokens(self):
        """Test creating a model with custom max_tokens."""
        model = make_model(max_tokens=16384)
        assert model.max_tokens == 16384

    def test_supports_all_features(self):
        """Test creating a model with all features enabled."""
        model = make_model(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        )
        assert model.supports_streaming is True
        assert model.supports_tools is True
        assert model.supports_vision is True

    def test_display_name_defaults_to_name(self):
        """Test that display_name defaults to name when not specified."""
        model = make_model(name="my-model")
        assert model.display_name == "my-model"

    def test_custom_display_name(self):
        """Test setting an explicit display_name."""
        model = make_model(name="gpt-4", display_name="GPT-4")
        assert model.display_name == "GPT-4"

    def test_extra_config(self):
        """Test setting extra model configuration."""
        model = make_model(extra={"context_window": 128000})
        assert model.extra == {"context_window": 128000}

    def test_is_model_config_instance(self):
        """Test that make_model returns a ModelConfig."""
        assert isinstance(make_model(), ModelConfig)


# ---------------------------------------------------------------------------
# make_provider tests
# ---------------------------------------------------------------------------


class TestMakeProvider:
    """Tests for the make_provider factory."""

    def test_default_provider(self):
        """Test creating a provider with default values."""
        provider = make_provider()
        assert provider.name == "test-provider"
        assert provider.api_base == "https://api.example.com/v1"
        assert provider.enabled is True
        assert "test-model" in provider.models

    def test_custom_name_and_api_base(self):
        """Test creating a provider with custom name and api_base."""
        provider = make_provider(name="openai", api_base="https://api.openai.com/v1")
        assert provider.name == "openai"
        assert provider.api_base == "https://api.openai.com/v1"

    def test_default_model_auto_created(self):
        """Test that a model is created automatically for default_model."""
        provider = make_provider(default_model="my-model")
        assert "my-model" in provider.models
        assert provider.default_model == "my-model"

    def test_custom_models(self):
        """Test providing custom model configs."""
        custom_models = {
            "gpt-4": ModelConfig(name="gpt-4", max_tokens=8192),
            "gpt-3.5": ModelConfig(name="gpt-3.5", max_tokens=4096),
        }
        provider = make_provider(
            name="openai",
            default_model="gpt-4",
            models=custom_models,
        )
        assert len(provider.models) == 2
        assert "gpt-4" in provider.models

    def test_auth_type(self):
        """Test setting a specific auth type."""
        provider = make_provider(auth_type=AuthType.BEARER_TOKEN)
        assert provider.auth_type == AuthType.BEARER_TOKEN

    def test_disabled_provider(self):
        """Test creating a disabled provider."""
        provider = make_provider(enabled=False)
        assert provider.enabled is False

    def test_display_name_defaults_to_name(self):
        """Test display_name defaults to name."""
        provider = make_provider(name="my-prov")
        assert provider.display_name == "my-prov"

    def test_custom_headers(self):
        """Test setting custom headers."""
        provider = make_provider(headers={"X-Custom": "value"})
        assert provider.headers == {"X-Custom": "value"}

    def test_is_provider_config_instance(self):
        """Test that make_provider returns a ProviderConfig."""
        assert isinstance(make_provider(), ProviderConfig)

    def test_passes_validation(self):
        """Test that the default make_provider output passes validation."""
        provider = make_provider()
        config = GatewayConfig(
            default_provider="test-provider",
            providers={"test-provider": provider},
        )
        result = ConfigValidator.validate(config)
        assert result.is_valid


# ---------------------------------------------------------------------------
# make_valid_config tests
# ---------------------------------------------------------------------------


class TestMakeValidConfig:
    """Tests for the make_valid_config factory."""

    def test_default_config(self):
        """Test creating a config with default values."""
        config = make_valid_config()
        assert config.default_provider == "test-provider"
        assert config.log_level == "info"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert "test-provider" in config.providers

    def test_custom_default_provider(self):
        """Test setting a custom default provider name."""
        config = make_valid_config(default_provider="myapi")
        assert config.default_provider == "myapi"
        assert "myapi" in config.providers

    def test_custom_providers(self):
        """Test providing custom providers dict."""
        providers = {
            "prov1": make_provider(name="prov1"),
            "prov2": make_provider(name="prov2"),
        }
        config = make_valid_config(
            default_provider="prov1",
            providers=providers,
        )
        assert len(config.providers) == 2

    def test_is_gateway_config(self):
        """Test that make_valid_config returns a GatewayConfig."""
        assert isinstance(make_valid_config(), GatewayConfig)

    def test_passes_validation_by_default(self):
        """Test that the default output passes validation."""
        config = make_valid_config()
        result = ConfigValidator.validate(config)
        assert result.is_valid, [str(e) for e in result.errors]

    def test_custom_timeout(self):
        """Test setting a custom timeout."""
        config = make_valid_config(timeout=60)
        assert config.timeout == 60

    def test_custom_log_level(self):
        """Test setting a custom log level."""
        config = make_valid_config(log_level="debug")
        assert config.log_level == "debug"


# ---------------------------------------------------------------------------
# assert_config_valid tests
# ---------------------------------------------------------------------------


class TestAssertConfigValid:
    """Tests for the assert_config_valid helper."""

    def test_passes_on_valid_config(self):
        """Test that a valid config does not raise."""
        config = make_valid_config()
        result = assert_config_valid(config)
        assert result.is_valid

    def test_returns_validation_result(self):
        """Test that the function returns a ValidationResult."""
        config = make_valid_config()
        result = assert_config_valid(config)
        assert isinstance(result, ValidationResult)

    def test_raises_on_invalid_config(self):
        """Test that an invalid config raises AssertionError."""
        config = GatewayConfig(timeout=-1, log_level="bad_level")
        with pytest.raises(AssertionError) as exc_info:
            assert_config_valid(config)
        assert "valid" in str(exc_info.value).lower()

    def test_error_message_includes_errors(self):
        """Test that the AssertionError includes the error messages."""
        config = GatewayConfig(timeout=-1)
        with pytest.raises(AssertionError) as exc_info:
            assert_config_valid(config)
        assert "timeout" in str(exc_info.value).lower()

    def test_custom_message_included(self):
        """Test that the custom message is included in the error."""
        config = GatewayConfig(timeout=-1)
        with pytest.raises(AssertionError) as exc_info:
            assert_config_valid(config, msg="my custom context")
        assert "my custom context" in str(exc_info.value)


# ---------------------------------------------------------------------------
# assert_config_invalid tests
# ---------------------------------------------------------------------------


class TestAssertConfigInvalid:
    """Tests for the assert_config_invalid helper."""

    def test_passes_on_invalid_config(self):
        """Test that an invalid config does not raise."""
        config = GatewayConfig(timeout=-1, log_level="bad")
        result = assert_config_invalid(config)
        assert not result.is_valid

    def test_returns_validation_result(self):
        """Test that the function returns a ValidationResult."""
        config = GatewayConfig(timeout=-1)
        result = assert_config_invalid(config)
        assert isinstance(result, ValidationResult)

    def test_raises_on_valid_config(self):
        """Test that a valid config raises AssertionError."""
        config = make_valid_config()
        with pytest.raises(AssertionError) as exc_info:
            assert_config_invalid(config)
        assert "invalid" in str(exc_info.value).lower()

    def test_custom_message_included(self):
        """Test that the custom message is included in the error."""
        config = make_valid_config()
        with pytest.raises(AssertionError) as exc_info:
            assert_config_invalid(config, msg="test scenario")
        assert "test scenario" in str(exc_info.value)


# ---------------------------------------------------------------------------
# assert_has_error_at tests
# ---------------------------------------------------------------------------


class TestAssertHasErrorAt:
    """Tests for the assert_has_error_at helper."""

    def test_passes_when_error_at_path(self):
        """Test that it passes when the expected error path is present."""
        config = GatewayConfig(timeout=-1)
        result = ConfigValidator.validate(config)
        # Should not raise
        assert_has_error_at(result, "timeout")

    def test_raises_when_no_error_at_path(self):
        """Test that it raises when the expected error path is absent."""
        result = ValidationResult()
        result.add_error("other.path", "Some error.")
        with pytest.raises(AssertionError) as exc_info:
            assert_has_error_at(result, "missing.path")
        assert "missing.path" in str(exc_info.value)

    def test_raises_on_empty_result(self):
        """Test that it raises when the result has no errors."""
        result = ValidationResult()
        with pytest.raises(AssertionError):
            assert_has_error_at(result, "timeout")

    def test_custom_message_included(self):
        """Test that the custom message is included in the error."""
        result = ValidationResult()
        with pytest.raises(AssertionError) as exc_info:
            assert_has_error_at(result, "field", msg="checking timeout error")
        assert "checking timeout error" in str(exc_info.value)

    def test_error_message_lists_actual_paths(self):
        """Test that the error message shows where actual errors are."""
        result = ValidationResult()
        result.add_error("real.path", "An error.")
        with pytest.raises(AssertionError) as exc_info:
            assert_has_error_at(result, "expected.path")
        assert "real.path" in str(exc_info.value)


# ---------------------------------------------------------------------------
# assert_has_warning_at tests
# ---------------------------------------------------------------------------


class TestAssertHasWarningAt:
    """Tests for the assert_has_warning_at helper."""

    def test_passes_when_warning_at_path(self):
        """Test that it passes when the expected warning path is present."""
        config = GatewayConfig(timeout=500)  # triggers high-timeout warning
        result = ConfigValidator.validate(config)
        # Should not raise
        assert_has_warning_at(result, "timeout")

    def test_raises_when_no_warning_at_path(self):
        """Test that it raises when the expected warning path is absent."""
        result = ValidationResult()
        result.add_warning("other.path", "A warning.")
        with pytest.raises(AssertionError) as exc_info:
            assert_has_warning_at(result, "missing.path")
        assert "missing.path" in str(exc_info.value)

    def test_raises_on_empty_result(self):
        """Test that it raises when the result has no warnings."""
        result = ValidationResult()
        with pytest.raises(AssertionError):
            assert_has_warning_at(result, "timeout")

    def test_custom_message_included(self):
        """Test that the custom message is included in the error."""
        result = ValidationResult()
        with pytest.raises(AssertionError) as exc_info:
            assert_has_warning_at(result, "field", msg="checking warning")
        assert "checking warning" in str(exc_info.value)

    def test_error_message_lists_actual_paths(self):
        """Test that the error message shows where actual warnings are."""
        result = ValidationResult()
        result.add_warning("actual.path", "A warning.")
        with pytest.raises(AssertionError) as exc_info:
            assert_has_warning_at(result, "expected.path")
        assert "actual.path" in str(exc_info.value)

    def test_does_not_confuse_errors_and_warnings(self):
        """Test that errors at a path don't satisfy warning assertion."""
        result = ValidationResult()
        result.add_error("timeout", "An error.")
        # Warning assertion should fail even though there's an error at same path
        with pytest.raises(AssertionError):
            assert_has_warning_at(result, "timeout")


# ---------------------------------------------------------------------------
# run_builtin_suite tests
# ---------------------------------------------------------------------------


class TestRunBuiltinSuite:
    """Tests for the run_builtin_suite convenience function."""

    def test_returns_suite_result(self):
        """Test that run_builtin_suite returns a ConfigTestSuiteResult."""
        from src.validation.testing import ConfigTestSuiteResult

        result = run_builtin_suite()
        assert isinstance(result, ConfigTestSuiteResult)

    def test_suite_has_tests(self):
        """Test that the built-in suite has at least one test."""
        result = run_builtin_suite()
        assert result.total > 0

    def test_all_builtin_tests_pass(self):
        """Test that all built-in tests pass."""
        result = run_builtin_suite()
        failed = [r.test_case.name for r in result.results if not r.passed]
        assert result.all_passed, f"Failed built-in tests: {failed}"


# ---------------------------------------------------------------------------
# Integration tests using helpers together
# ---------------------------------------------------------------------------


class TestHelperIntegration:
    """Integration tests demonstrating helper usage together."""

    def test_build_and_validate_provider_config(self):
        """Test building and validating a provider config end-to-end."""
        model = make_model(
            name="claude-3",
            max_tokens=100000,
            supports_streaming=True,
            supports_tools=True,
        )
        provider = make_provider(
            name="anthropic",
            api_base="https://api.anthropic.com/v1",
            api_key_env_var="ANTHROPIC_API_KEY",
            auth_type=AuthType.API_KEY,
            default_model="claude-3",
            models={"claude-3": model},
        )
        config = make_valid_config(
            default_provider="anthropic",
            providers={"anthropic": provider},
        )
        result = assert_config_valid(config)
        assert result.error_count == 0

    def test_invalid_provider_triggers_correct_errors(self):
        """Test that a misconfigured provider triggers the expected errors."""
        # Provider with no api_base and bad model
        provider = ProviderConfig(
            name="broken",
            api_base="",  # missing
            models={
                "bad-model": ModelConfig(name="bad-model", max_tokens=0)
            },
        )
        config = GatewayConfig(
            default_provider="broken",
            providers={"broken": provider},
        )
        result = assert_config_invalid(config)
        assert_has_error_at(result, "providers.broken.api_base")
        assert_has_error_at(result, "providers.broken.models.bad-model.max_tokens")

    def test_high_timeout_triggers_warning(self):
        """Test that a high timeout produces a warning at the expected path."""
        config = make_valid_config(timeout=400)
        result = ConfigValidator.validate(config)
        assert_has_warning_at(result, "timeout")

    def test_no_default_provider_triggers_warning(self):
        """Test that a missing default provider triggers a warning."""
        provider = make_provider(name="test")
        config = GatewayConfig(
            default_provider="",  # not set
            providers={"test": provider},
        )
        result = ConfigValidator.validate(config)
        assert_has_warning_at(result, "default_provider")

    def test_multiple_providers_all_valid(self):
        """Test a config with multiple providers passes validation."""
        providers = {
            "openai": make_provider(
                name="openai",
                api_base="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
                auth_type=AuthType.BEARER_TOKEN,
            ),
            "anthropic": make_provider(
                name="anthropic",
                api_base="https://api.anthropic.com/v1",
                api_key_env_var="ANTHROPIC_API_KEY",
                auth_type=AuthType.API_KEY,
            ),
        }
        config = make_valid_config(
            default_provider="openai",
            providers=providers,
        )
        assert_config_valid(config)
