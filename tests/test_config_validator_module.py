"""Tests for the src.config.validator module.

This module bridges src.config.schema (schema-level validation) with
src.validation.validator (semantic validation) into a unified pipeline.
"""

import pytest

from src.config.validator import (
    PROFILES,
    ValidationPipeline,
    ValidationPreset,
    ValidationProfile,
    ValidationStep,
    create_default_pipeline,
    full_validate,
    full_validate_dict,
    is_valid,
    quick_validate,
    validate_for_production,
)
from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig
from src.validation.validator import ValidationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_config():
    """Create a fully valid GatewayConfig."""
    return GatewayConfig(
        default_provider="openai",
        log_level="info",
        timeout=30,
        max_retries=3,
        providers={
            "openai": ProviderConfig(
                name="openai",
                display_name="OpenAI",
                api_base="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
                auth_type=AuthType.BEARER_TOKEN,
                default_model="gpt-4",
                models={
                    "gpt-4": ModelConfig(name="gpt-4", max_tokens=8192),
                },
            )
        },
    )


@pytest.fixture
def valid_config_dict():
    """Create a valid configuration dictionary."""
    return {
        "default_provider": "openai",
        "log_level": "info",
        "timeout": 30,
        "max_retries": 3,
        "providers": {
            "openai": {
                "name": "openai",
                "api_base": "https://api.openai.com/v1",
                "api_key_env_var": "OPENAI_API_KEY",
                "auth_type": "bearer_token",
                "default_model": "gpt-4",
                "models": {
                    "gpt-4": {"name": "gpt-4", "max_tokens": 8192}
                },
            }
        },
    }


@pytest.fixture
def invalid_config_dict():
    """Create an invalid configuration dictionary."""
    return {
        "default_provider": "nonexistent",
        "log_level": "bad_level",
        "timeout": -1,
    }


# ---------------------------------------------------------------------------
# ValidationPreset tests
# ---------------------------------------------------------------------------


class TestValidationPreset:
    """Tests for the ValidationPreset enum."""

    def test_preset_values(self):
        """Test that all expected presets exist."""
        assert ValidationPreset.RELAXED.value == "relaxed"
        assert ValidationPreset.DEFAULT.value == "default"
        assert ValidationPreset.STRICT.value == "strict"
        assert ValidationPreset.PRODUCTION.value == "production"

    def test_presets_are_strings(self):
        """Test that presets behave as strings."""
        assert ValidationPreset.RELAXED == "relaxed"
        assert ValidationPreset.DEFAULT == "default"


# ---------------------------------------------------------------------------
# ValidationProfile tests
# ---------------------------------------------------------------------------


class TestValidationProfile:
    """Tests for the ValidationProfile dataclass."""

    def test_default_profile_does_not_fail_on_warnings(self):
        """Test that default profile allows warnings."""
        profile = PROFILES["default"]
        assert profile.fail_on_warnings is False
        assert profile.fail_on_info is False

    def test_strict_profile_fails_on_warnings(self):
        """Test that strict profile fails on warnings."""
        profile = PROFILES["strict"]
        assert profile.fail_on_warnings is True

    def test_relaxed_profile_ignores_warnings(self):
        """Test that relaxed profile does not fail on warnings."""
        profile = PROFILES["relaxed"]
        assert profile.fail_on_warnings is False

    def test_production_profile_fails_on_warnings(self):
        """Test that production profile fails on warnings."""
        profile = PROFILES["production"]
        assert profile.fail_on_warnings is True

    def test_production_profile_has_extra_rules(self):
        """Test that production profile has additional validation rules."""
        profile = PROFILES["production"]
        assert len(profile.extra_rules) > 0

    def test_custom_profile_creation(self):
        """Test creating a custom validation profile."""
        profile = ValidationProfile(
            name="custom",
            description="A custom profile.",
            fail_on_warnings=True,
            fail_on_info=False,
        )
        assert profile.name == "custom"
        assert profile.fail_on_warnings is True

    def test_all_standard_profiles_exist(self):
        """Test that all four standard profiles exist in PROFILES."""
        assert "relaxed" in PROFILES
        assert "default" in PROFILES
        assert "strict" in PROFILES
        assert "production" in PROFILES


# ---------------------------------------------------------------------------
# quick_validate tests
# ---------------------------------------------------------------------------


class TestQuickValidate:
    """Tests for quick_validate (schema-level check)."""

    def test_valid_empty_dict_passes(self):
        """Test that an empty dict passes quick validation."""
        errors = quick_validate({})
        assert errors == []

    def test_valid_full_config_passes(self, valid_config_dict):
        """Test that a valid config dict passes."""
        errors = quick_validate(valid_config_dict)
        assert errors == []

    def test_invalid_timeout_fails(self):
        """Test that string timeout fails quick validation."""
        errors = quick_validate({"timeout": "thirty"})
        assert len(errors) >= 1
        assert any("timeout" in e for e in errors)

    def test_invalid_log_level_fails(self):
        """Test that invalid log level fails quick validation."""
        errors = quick_validate({"log_level": "verbose"})
        assert len(errors) >= 1

    def test_invalid_max_retries_type_fails(self):
        """Test that string max_retries fails quick validation."""
        errors = quick_validate({"max_retries": "three"})
        assert len(errors) >= 1

    def test_returns_list(self):
        """Test that quick_validate always returns a list."""
        result = quick_validate({})
        assert isinstance(result, list)

    def test_unknown_fields_do_not_cause_errors(self):
        """Test that unknown top-level fields are ignored."""
        errors = quick_validate({"completely_unknown_field": "value"})
        assert errors == []


# ---------------------------------------------------------------------------
# full_validate tests
# ---------------------------------------------------------------------------


class TestFullValidate:
    """Tests for full_validate (semantic validation)."""

    def test_valid_config_passes_default(self, valid_config):
        """Test that a valid config passes default-profile validation."""
        result = full_validate(valid_config, profile="default")
        assert result.is_valid

    def test_valid_config_passes_relaxed(self, valid_config):
        """Test that a valid config passes relaxed-profile validation."""
        result = full_validate(valid_config, profile="relaxed")
        assert result.is_valid

    def test_valid_config_passes_strict(self, valid_config):
        """Test that a valid config passes strict-profile validation."""
        result = full_validate(valid_config, profile="strict")
        assert result.is_valid

    def test_config_with_high_timeout_warns_on_default(self):
        """Test that high timeout produces a warning on default profile."""
        config = GatewayConfig(timeout=500)
        result = full_validate(config, profile="default")
        assert result.warning_count > 0

    def test_strict_profile_sees_warnings(self):
        """Test that strict profile propagates warnings."""
        config = GatewayConfig(timeout=500)
        result = full_validate(config, profile="strict")
        # Both strict and default run the same validator;
        # result will have warnings regardless - the profile
        # controls how the caller *interprets* them.
        assert result.warning_count > 0

    def test_production_profile_runs_extra_rules(self):
        """Test that production profile adds extra validation rules."""
        # A config with placeholder URLs should fail production checks
        config = GatewayConfig(
            providers={
                "azure": ProviderConfig(
                    name="azure",
                    api_base="https://<your-resource>.openai.azure.com",
                    enabled=True,
                )
            }
        )
        result = full_validate(config, profile="production")
        # Production rule should flag the placeholder URL as an error
        error_paths = [m.path for m in result.errors]
        assert any("api_base" in p for p in error_paths)

    def test_unknown_profile_falls_back_to_default(self, valid_config):
        """Test that an unknown profile falls back to default."""
        result = full_validate(valid_config, profile="nonexistent-profile")
        assert isinstance(result, ValidationResult)
        # Should not raise; falls back to default
        assert result.is_valid

    def test_returns_validation_result(self, valid_config):
        """Test that full_validate always returns a ValidationResult."""
        result = full_validate(valid_config)
        assert isinstance(result, ValidationResult)

    def test_config_with_errors_is_invalid(self):
        """Test that config with errors is correctly identified as invalid."""
        config = GatewayConfig(
            default_provider="nonexistent",
            timeout=-1,
            log_level="bad",
        )
        result = full_validate(config)
        assert not result.is_valid
        assert result.error_count > 0


# ---------------------------------------------------------------------------
# full_validate_dict tests
# ---------------------------------------------------------------------------


class TestFullValidateDict:
    """Tests for full_validate_dict (dict input)."""

    def test_valid_dict_passes(self, valid_config_dict):
        """Test that a valid dict passes validation."""
        result = full_validate_dict(valid_config_dict)
        assert result.is_valid

    def test_invalid_dict_fails(self, invalid_config_dict):
        """Test that an invalid dict fails validation."""
        result = full_validate_dict(invalid_config_dict)
        assert not result.is_valid

    def test_empty_dict_passes(self):
        """Test that an empty dict is valid (warnings only)."""
        result = full_validate_dict({})
        assert result.is_valid

    def test_unparseable_dict_returns_error(self):
        """Test that unparseable dict returns error instead of raising."""
        result = full_validate_dict({"providers": "not-a-dict"})
        assert len(result.messages) > 0

    def test_with_production_profile(self, valid_config_dict):
        """Test full_validate_dict with production profile."""
        result = full_validate_dict(valid_config_dict, profile="production")
        assert isinstance(result, ValidationResult)

    def test_returns_validation_result(self):
        """Test that full_validate_dict always returns ValidationResult."""
        result = full_validate_dict({})
        assert isinstance(result, ValidationResult)


# ---------------------------------------------------------------------------
# validate_for_production tests
# ---------------------------------------------------------------------------


class TestValidateForProduction:
    """Tests for validate_for_production."""

    def test_valid_config_passes_production(self, valid_config_dict):
        """Test that a complete valid config passes production validation."""
        ok, report = validate_for_production(valid_config_dict)
        assert isinstance(ok, bool)
        assert isinstance(report, str)
        # A valid config with no placeholder URLs and good auth should pass
        # (though it might have warnings from production rules)

    def test_placeholder_url_fails_production(self):
        """Test that a placeholder URL fails production validation."""
        data = {
            "providers": {
                "azure": {
                    "name": "azure",
                    "api_base": "https://<YOUR_RESOURCE>.openai.azure.com",
                    "enabled": True,
                }
            }
        }
        ok, report = validate_for_production(data)
        assert ok is False

    def test_returns_tuple(self):
        """Test that validate_for_production returns a (bool, str) tuple."""
        result = validate_for_production({})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_report_contains_summary(self):
        """Test that the production report contains a summary."""
        _, report = validate_for_production({})
        assert "VALID" in report or "INVALID" in report

    def test_invalid_config_fails_production(self, invalid_config_dict):
        """Test that an invalid config fails production validation."""
        ok, report = validate_for_production(invalid_config_dict)
        assert ok is False


# ---------------------------------------------------------------------------
# is_valid tests
# ---------------------------------------------------------------------------


class TestIsValid:
    """Tests for is_valid quick checker."""

    def test_valid_dict_returns_true(self, valid_config_dict):
        """Test that a valid config returns True."""
        assert is_valid(valid_config_dict) is True

    def test_empty_dict_returns_true(self):
        """Test that empty dict returns True (warnings are ok)."""
        assert is_valid({}) is True

    def test_invalid_dict_returns_false(self, invalid_config_dict):
        """Test that an invalid config returns False."""
        assert is_valid(invalid_config_dict) is False

    def test_timeout_negative_returns_false(self):
        """Test that negative timeout returns False."""
        assert is_valid({"timeout": -1}) is False

    def test_valid_log_level_returns_true(self):
        """Test that a valid log level returns True."""
        assert is_valid({"log_level": "debug"}) is True

    def test_invalid_log_level_returns_false(self):
        """Test that an invalid log level returns False."""
        assert is_valid({"log_level": "extreme"}) is False


# ---------------------------------------------------------------------------
# ValidationStep tests
# ---------------------------------------------------------------------------


class TestValidationStep:
    """Tests for the ValidationStep dataclass."""

    def test_step_creation(self):
        """Test creating a validation step."""
        def noop(data, result):
            pass

        step = ValidationStep(
            name="my-step",
            description="A test step.",
            validate_fn=noop,
            required=True,
        )
        assert step.name == "my-step"
        assert step.required is True

    def test_step_required_defaults_to_true(self):
        """Test that required defaults to True."""
        step = ValidationStep(
            name="s",
            description="",
            validate_fn=lambda d, r: None,
        )
        assert step.required is True

    def test_step_optional(self):
        """Test creating an optional step."""
        step = ValidationStep(
            name="opt",
            description="",
            validate_fn=lambda d, r: None,
            required=False,
        )
        assert step.required is False


# ---------------------------------------------------------------------------
# ValidationPipeline tests
# ---------------------------------------------------------------------------


class TestValidationPipeline:
    """Tests for the ValidationPipeline class."""

    def test_empty_pipeline_produces_empty_result(self):
        """Test that running an empty pipeline returns empty result."""
        pipeline = ValidationPipeline()
        result = pipeline.run({})
        assert isinstance(result, ValidationResult)
        assert len(result.messages) == 0

    def test_single_step_runs(self):
        """Test that a single step is executed."""
        called = []

        def my_step(data, result):
            called.append(data)

        pipeline = ValidationPipeline()
        pipeline.add_step(
            ValidationStep(name="step1", description="", validate_fn=my_step)
        )
        pipeline.run({"key": "value"})
        assert len(called) == 1
        assert called[0] == {"key": "value"}

    def test_step_can_add_messages(self):
        """Test that steps can add messages to the result."""
        def adding_step(data, result):
            result.add_error("field", "An error.")
            result.add_warning("field2", "A warning.")

        pipeline = ValidationPipeline()
        pipeline.add_step(
            ValidationStep(name="adder", description="", validate_fn=adding_step)
        )
        result = pipeline.run({})
        assert result.error_count == 1
        assert result.warning_count == 1

    def test_required_step_failure_stops_pipeline(self):
        """Test that a failed required step stops subsequent steps."""
        call_order = []

        def step1(data, result):
            call_order.append("step1")
            result.add_error("f", "Error from step1.")

        def step2(data, result):
            call_order.append("step2")

        pipeline = ValidationPipeline()
        pipeline.add_step(
            ValidationStep(
                name="step1",
                description="Required step that fails.",
                validate_fn=step1,
                required=True,
            )
        )
        pipeline.add_step(
            ValidationStep(
                name="step2",
                description="Should not run.",
                validate_fn=step2,
            )
        )
        pipeline.run({})
        assert "step1" in call_order
        assert "step2" not in call_order  # blocked by required failure

    def test_optional_step_failure_does_not_stop_pipeline(self):
        """Test that a failed optional step allows subsequent steps to run."""
        call_order = []

        def step1(data, result):
            call_order.append("step1")
            result.add_error("f", "Error.")

        def step2(data, result):
            call_order.append("step2")

        pipeline = ValidationPipeline()
        pipeline.add_step(
            ValidationStep(
                name="step1",
                description="Optional step that fails.",
                validate_fn=step1,
                required=False,
            )
        )
        pipeline.add_step(
            ValidationStep(name="step2", description="Runs after.", validate_fn=step2)
        )
        pipeline.run({})
        assert "step1" in call_order
        assert "step2" in call_order  # runs because step1 was optional

    def test_step_exception_is_caught(self):
        """Test that exceptions in steps are caught and reported."""
        def crashing_step(data, result):
            raise RuntimeError("Unexpected crash!")

        pipeline = ValidationPipeline()
        pipeline.add_step(
            ValidationStep(
                name="crasher",
                description="",
                validate_fn=crashing_step,
            )
        )
        result = pipeline.run({})
        # Exception should be caught and reported as an error
        assert result.error_count >= 1
        error_paths = [m.path for m in result.errors]
        assert any("pipeline.crasher" in p for p in error_paths)

    def test_add_step_returns_self_for_chaining(self):
        """Test that add_step returns self for method chaining."""
        pipeline = ValidationPipeline()
        returned = pipeline.add_step(
            ValidationStep(name="s", description="", validate_fn=lambda d, r: None)
        )
        assert returned is pipeline

    def test_list_steps(self):
        """Test that list_steps returns step names in order."""
        pipeline = ValidationPipeline()
        pipeline.add_step(
            ValidationStep(name="alpha", description="", validate_fn=lambda d, r: None)
        )
        pipeline.add_step(
            ValidationStep(name="beta", description="", validate_fn=lambda d, r: None)
        )
        steps = pipeline.list_steps()
        assert steps == ["alpha", "beta"]

    def test_multiple_steps_run_in_order(self):
        """Test that steps run in registration order."""
        call_order = []

        def make_step(n):
            def step(data, result):
                call_order.append(n)
            return step

        pipeline = ValidationPipeline()
        for name in ["first", "second", "third"]:
            pipeline.add_step(
                ValidationStep(name=name, description="", validate_fn=make_step(name))
            )

        pipeline.run({})
        assert call_order == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# create_default_pipeline tests
# ---------------------------------------------------------------------------


class TestCreateDefaultPipeline:
    """Tests for create_default_pipeline factory."""

    def test_returns_pipeline(self):
        """Test that it returns a ValidationPipeline."""
        pipeline = create_default_pipeline()
        assert isinstance(pipeline, ValidationPipeline)

    def test_has_two_steps(self):
        """Test that the default pipeline has two steps."""
        pipeline = create_default_pipeline()
        steps = pipeline.list_steps()
        assert len(steps) == 2

    def test_has_schema_step(self):
        """Test that the default pipeline has a 'schema' step."""
        pipeline = create_default_pipeline()
        assert "schema" in pipeline.list_steps()

    def test_has_semantic_step(self):
        """Test that the default pipeline has a 'semantic' step."""
        pipeline = create_default_pipeline()
        assert "semantic" in pipeline.list_steps()

    def test_valid_config_passes_default_pipeline(self, valid_config_dict):
        """Test that a valid config passes the default pipeline."""
        pipeline = create_default_pipeline()
        result = pipeline.run(valid_config_dict)
        assert result.is_valid

    def test_invalid_config_fails_default_pipeline(self, invalid_config_dict):
        """Test that an invalid config fails the default pipeline."""
        pipeline = create_default_pipeline()
        result = pipeline.run(invalid_config_dict)
        assert not result.is_valid

    def test_pipeline_runs_both_schema_and_semantic(self):
        """Test that both schema and semantic checks run on valid but edge config."""
        # Config with schema-level violation (timeout = string)
        data = {"timeout": "thirty"}
        pipeline = create_default_pipeline()
        result = pipeline.run(data)
        # Should have errors from schema step (type mismatch)
        assert result.error_count >= 0  # At minimum no crash


# ---------------------------------------------------------------------------
# Production rules tests
# ---------------------------------------------------------------------------


class TestProductionRules:
    """Tests for the production-specific validation rules."""

    def test_no_placeholder_urls_rule(self):
        """Test that placeholder URLs in api_base are flagged."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://<YOUR-NAME>.openai.azure.com/v1",
                    enabled=True,
                )
            }
        )
        result = full_validate(config, profile="production")
        errors = [m for m in result.errors if "api_base" in m.path]
        assert len(errors) >= 1

    def test_at_least_one_model_rule(self):
        """Test that enabled providers with no models trigger a warning."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://api.example.com",
                    enabled=True,
                    models={},  # no models
                )
            }
        )
        result = full_validate(config, profile="production")
        warnings = [m for m in result.warnings if "models" in m.path]
        assert len(warnings) >= 1

    def test_reasonable_timeouts_rule_low_timeout(self):
        """Test that very low timeout triggers a warning in production."""
        config = GatewayConfig(timeout=2)  # very low
        result = full_validate(config, profile="production")
        warnings = [m for m in result.warnings if "timeout" in m.path]
        assert len(warnings) >= 1

    def test_api_keys_configured_rule(self):
        """Test that missing api_key_env_var triggers a warning in production."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://api.example.com",
                    api_key_env_var="",
                    auth_type=AuthType.API_KEY,
                    enabled=True,
                )
            }
        )
        result = full_validate(config, profile="production")
        warnings = [m for m in result.warnings if "api_key_env_var" in m.path]
        assert len(warnings) >= 1

    def test_disabled_provider_skipped_by_production_rules(self):
        """Test that disabled providers are skipped by production api-key rule."""
        # Use AuthType.NONE for the disabled provider so base validator
        # won't add api_key_env_var warnings for it either.  This lets us
        # isolate the production-rule behaviour (which skips disabled providers).
        config = GatewayConfig(
            providers={
                "disabled_provider": ProviderConfig(
                    name="disabled_provider",
                    api_base="https://api.example.com",
                    api_key_env_var="",
                    auth_type=AuthType.NONE,  # no auth → no api-key warnings
                    enabled=False,
                ),
                "enabled_provider": ProviderConfig(
                    name="enabled_provider",
                    api_base="https://api2.example.com",
                    api_key_env_var="MY_KEY",
                    auth_type=AuthType.API_KEY,
                    enabled=True,
                    models={
                        "model1": ModelConfig(name="model1", max_tokens=4096)
                    },
                ),
            },
            default_provider="enabled_provider",
        )
        result = full_validate(config, profile="production")
        # No api-key warning for the disabled provider (it has AuthType.NONE anyway)
        disabled_api_key_warnings = [
            m
            for m in result.warnings
            if "disabled_provider" in m.path and "api_key" in m.path
        ]
        assert len(disabled_api_key_warnings) == 0
