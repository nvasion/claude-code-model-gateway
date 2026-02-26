"""Tests for the comprehensive configuration validator."""

import pytest

from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig
from src.validation.validator import (
    ConfigValidator,
    ExtendedValidator,
    Severity,
    ValidationMessage,
    ValidationResult,
    ValidationRule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_config():
    """Create a fully valid GatewayConfig for testing."""
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
                    "gpt-4": ModelConfig(
                        name="gpt-4",
                        display_name="GPT-4",
                        max_tokens=8192,
                    ),
                },
            ),
        },
    )


@pytest.fixture
def multi_provider_config():
    """Create a config with multiple providers."""
    return GatewayConfig(
        default_provider="openai",
        log_level="info",
        timeout=30,
        max_retries=3,
        providers={
            "openai": ProviderConfig(
                name="openai",
                api_base="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
                auth_type=AuthType.BEARER_TOKEN,
                default_model="gpt-4",
                models={
                    "gpt-4": ModelConfig(name="gpt-4", max_tokens=8192),
                },
            ),
            "anthropic": ProviderConfig(
                name="anthropic",
                api_base="https://api.anthropic.com/v1",
                api_key_env_var="ANTHROPIC_API_KEY",
                auth_type=AuthType.API_KEY,
                default_model="claude-3",
                models={
                    "claude-3": ModelConfig(name="claude-3", max_tokens=4096),
                },
            ),
        },
    )


# ---------------------------------------------------------------------------
# ValidationMessage tests
# ---------------------------------------------------------------------------


class TestValidationMessage:
    """Tests for ValidationMessage."""

    def test_str_format_basic(self):
        """Test basic string formatting."""
        msg = ValidationMessage(
            severity=Severity.ERROR,
            path="timeout",
            message="Timeout must be positive.",
        )
        text = str(msg)
        assert "[ERROR]" in text
        assert "timeout" in text
        assert "Timeout must be positive" in text

    def test_str_format_with_value(self):
        """Test string formatting with value."""
        msg = ValidationMessage(
            severity=Severity.WARNING,
            path="max_retries",
            message="High retry count.",
            value=15,
        )
        text = str(msg)
        assert "[WARNING]" in text
        assert "Got: 15" in text

    def test_str_format_with_suggestion(self):
        """Test string formatting with suggestion."""
        msg = ValidationMessage(
            severity=Severity.ERROR,
            path="log_level",
            message="Invalid log level.",
            suggestion="Use one of: debug, info.",
        )
        text = str(msg)
        assert "Suggestion:" in text

    def test_to_dict(self):
        """Test dictionary serialization."""
        msg = ValidationMessage(
            severity=Severity.ERROR,
            path="timeout",
            message="Bad timeout.",
            value=-1,
            suggestion="Use positive value.",
        )
        d = msg.to_dict()
        assert d["severity"] == "error"
        assert d["path"] == "timeout"
        assert d["message"] == "Bad timeout."
        assert d["value"] == "-1"
        assert d["suggestion"] == "Use positive value."

    def test_to_dict_none_value(self):
        """Test that None value serializes as None."""
        msg = ValidationMessage(
            severity=Severity.INFO,
            path="test",
            message="Info msg.",
        )
        d = msg.to_dict()
        assert d["value"] is None
        assert d["suggestion"] is None


# ---------------------------------------------------------------------------
# ValidationResult tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_empty_result_is_valid(self):
        """Test that empty result is valid."""
        result = ValidationResult()
        assert result.is_valid
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_add_error_makes_invalid(self):
        """Test that adding error makes result invalid."""
        result = ValidationResult()
        result.add_error("field", "Error message")
        assert not result.is_valid
        assert result.error_count == 1

    def test_add_warning_still_valid(self):
        """Test that warnings don't affect validity."""
        result = ValidationResult()
        result.add_warning("field", "Warning message")
        assert result.is_valid
        assert result.warning_count == 1

    def test_add_info_still_valid(self):
        """Test that info messages don't affect validity."""
        result = ValidationResult()
        result.add_info("field", "Info message")
        assert result.is_valid
        assert len(result.infos) == 1

    def test_errors_property(self):
        """Test errors filter."""
        result = ValidationResult()
        result.add_error("f1", "err1")
        result.add_warning("f2", "warn1")
        result.add_error("f3", "err2")
        assert len(result.errors) == 2
        assert all(m.severity == Severity.ERROR for m in result.errors)

    def test_warnings_property(self):
        """Test warnings filter."""
        result = ValidationResult()
        result.add_error("f1", "err1")
        result.add_warning("f2", "warn1")
        result.add_warning("f3", "warn2")
        assert len(result.warnings) == 2

    def test_summary_valid(self):
        """Test summary for valid config."""
        result = ValidationResult()
        result.add_warning("f1", "warn")
        summary = result.summary()
        assert "VALID" in summary
        assert "0 error(s)" in summary
        assert "1 warning(s)" in summary

    def test_summary_invalid(self):
        """Test summary for invalid config."""
        result = ValidationResult()
        result.add_error("f1", "err")
        summary = result.summary()
        assert "INVALID" in summary

    def test_format_report(self):
        """Test report formatting."""
        result = ValidationResult()
        result.add_error("field1", "Error here")
        result.add_warning("field2", "Warning here")
        result.add_info("field3", "Info here")

        report = result.format_report(show_info=False)
        assert "[ERROR]" in report
        assert "[WARNING]" in report
        assert "[INFO]" not in report

        report_with_info = result.format_report(show_info=True)
        assert "[INFO]" in report_with_info


# ---------------------------------------------------------------------------
# ConfigValidator tests - top level
# ---------------------------------------------------------------------------


class TestConfigValidatorTopLevel:
    """Tests for top-level validation."""

    def test_valid_config_passes(self, valid_config):
        """Test that a valid config has no errors."""
        result = ConfigValidator.validate(valid_config)
        assert result.is_valid

    def test_invalid_log_level(self):
        """Test that invalid log level is caught."""
        config = GatewayConfig(log_level="verbose")
        result = ConfigValidator.validate(config)
        errors = [m for m in result.errors if "log_level" in m.path]
        assert len(errors) >= 1

    def test_negative_timeout_error(self):
        """Test that negative timeout produces an error."""
        config = GatewayConfig(timeout=-1)
        result = ConfigValidator.validate(config)
        errors = [m for m in result.errors if "timeout" in m.path]
        assert len(errors) >= 1

    def test_zero_timeout_error(self):
        """Test that zero timeout produces an error."""
        config = GatewayConfig(timeout=0)
        result = ConfigValidator.validate(config)
        errors = [m for m in result.errors if "timeout" in m.path]
        assert len(errors) >= 1

    def test_high_timeout_warning(self):
        """Test that very high timeout produces a warning."""
        config = GatewayConfig(timeout=500)
        result = ConfigValidator.validate(config)
        warnings = [m for m in result.warnings if "timeout" in m.path]
        assert len(warnings) >= 1

    def test_negative_retries_error(self):
        """Test that negative max_retries produces an error."""
        config = GatewayConfig(max_retries=-1)
        result = ConfigValidator.validate(config)
        errors = [m for m in result.errors if "max_retries" in m.path]
        assert len(errors) >= 1

    def test_high_retries_warning(self):
        """Test that high retry count produces a warning."""
        config = GatewayConfig(max_retries=15)
        result = ConfigValidator.validate(config)
        warnings = [m for m in result.warnings if "max_retries" in m.path]
        assert len(warnings) >= 1

    def test_no_providers_warning(self):
        """Test warning when no providers configured."""
        config = GatewayConfig()
        result = ConfigValidator.validate(config)
        warnings = [m for m in result.warnings if "providers" in m.path]
        assert len(warnings) >= 1


# ---------------------------------------------------------------------------
# ConfigValidator tests - providers
# ---------------------------------------------------------------------------


class TestConfigValidatorProviders:
    """Tests for provider-level validation."""

    def test_missing_api_base(self):
        """Test that missing api_base produces an error."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(name="test", api_base=""),
            }
        )
        result = ConfigValidator.validate(config)
        errors = [m for m in result.errors if "api_base" in m.path]
        assert len(errors) >= 1

    def test_invalid_url(self):
        """Test that invalid URL produces an error."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(name="test", api_base="not-a-url"),
            }
        )
        result = ConfigValidator.validate(config)
        errors = [m for m in result.errors if "api_base" in m.path]
        assert len(errors) >= 1

    def test_template_url_warns(self):
        """Test that template URL produces a warning."""
        config = GatewayConfig(
            providers={
                "azure": ProviderConfig(
                    name="azure",
                    api_base="https://<your-resource>.openai.azure.com",
                ),
            }
        )
        result = ConfigValidator.validate(config)
        warnings = [m for m in result.warnings if "api_base" in m.path]
        assert len(warnings) >= 1

    def test_no_api_key_env_warns(self):
        """Test warning when auth-enabled provider has no api_key_env_var."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://api.example.com",
                    api_key_env_var="",
                    auth_type=AuthType.BEARER_TOKEN,
                ),
            }
        )
        result = ConfigValidator.validate(config)
        warnings = [m for m in result.warnings if "api_key_env_var" in m.path]
        assert len(warnings) >= 1

    def test_name_mismatch_warns(self):
        """Test warning when provider name doesn't match key."""
        config = GatewayConfig(
            providers={
                "mykey": ProviderConfig(
                    name="different-name",
                    api_base="https://api.example.com",
                ),
            }
        )
        result = ConfigValidator.validate(config)
        warnings = [m for m in result.warnings if ".name" in m.path]
        assert len(warnings) >= 1

    def test_all_providers_disabled(self):
        """Test that all-disabled providers produce an error."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://api.example.com",
                    enabled=False,
                ),
            }
        )
        result = ConfigValidator.validate(config)
        errors = [m for m in result.errors if m.path == "providers"]
        assert len(errors) >= 1

    def test_bad_model_max_tokens(self):
        """Test that model with 0 max_tokens errors."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://api.example.com",
                    models={
                        "bad": ModelConfig(name="bad", max_tokens=0),
                    },
                ),
            }
        )
        result = ConfigValidator.validate(config)
        errors = [m for m in result.errors if "max_tokens" in m.path]
        assert len(errors) >= 1

    def test_very_high_max_tokens_warns(self):
        """Test warning for very high max_tokens."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://api.example.com",
                    models={
                        "big": ModelConfig(name="big", max_tokens=5_000_000),
                    },
                ),
            }
        )
        result = ConfigValidator.validate(config)
        warnings = [m for m in result.warnings if "max_tokens" in m.path]
        assert len(warnings) >= 1

    def test_default_model_not_in_models(self):
        """Test error when default_model references non-existent model."""
        config = GatewayConfig(
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://api.example.com",
                    default_model="nonexistent",
                    models={
                        "real": ModelConfig(name="real", max_tokens=4096),
                    },
                ),
            }
        )
        result = ConfigValidator.validate(config)
        errors = [m for m in result.errors if "default_model" in m.path]
        assert len(errors) >= 1


# ---------------------------------------------------------------------------
# ConfigValidator tests - cross references
# ---------------------------------------------------------------------------


class TestConfigValidatorCrossReferences:
    """Tests for cross-reference validation."""

    def test_default_provider_missing(self):
        """Test error when default provider not in providers."""
        config = GatewayConfig(
            default_provider="nonexistent",
            providers={
                "real": ProviderConfig(
                    name="real",
                    api_base="https://api.example.com",
                ),
            },
        )
        result = ConfigValidator.validate(config)
        errors = [m for m in result.errors if "default_provider" in m.path]
        assert len(errors) >= 1

    def test_default_provider_disabled_warns(self):
        """Test warning when default provider is disabled."""
        config = GatewayConfig(
            default_provider="off",
            providers={
                "off": ProviderConfig(
                    name="off",
                    api_base="https://api.example.com",
                    enabled=False,
                ),
                "on": ProviderConfig(
                    name="on",
                    api_base="https://api2.example.com",
                    enabled=True,
                ),
            },
        )
        result = ConfigValidator.validate(config)
        warnings = [m for m in result.warnings if "default_provider" in m.path]
        assert len(warnings) >= 1

    def test_no_default_provider_warns(self):
        """Test warning when no default provider set."""
        config = GatewayConfig(
            default_provider="",
            providers={
                "test": ProviderConfig(
                    name="test",
                    api_base="https://api.example.com",
                ),
            },
        )
        result = ConfigValidator.validate(config)
        warnings = [m for m in result.warnings if "default_provider" in m.path]
        assert len(warnings) >= 1


# ---------------------------------------------------------------------------
# ConfigValidator.validate_dict tests
# ---------------------------------------------------------------------------


class TestValidateDict:
    """Tests for validating raw dictionaries."""

    def test_valid_dict(self):
        """Test validating a valid dictionary."""
        data = {
            "default_provider": "openai",
            "log_level": "info",
            "timeout": 30,
            "max_retries": 3,
            "providers": {
                "openai": {
                    "name": "openai",
                    "api_base": "https://api.openai.com/v1",
                }
            },
        }
        result = ConfigValidator.validate_dict(data)
        assert result.is_valid

    def test_empty_dict(self):
        """Test validating empty dictionary."""
        result = ConfigValidator.validate_dict({})
        assert result.is_valid  # valid but with warnings

    def test_invalid_dict(self):
        """Test validating dictionary with errors."""
        data = {
            "default_provider": "nonexistent",
            "timeout": -1,
        }
        result = ConfigValidator.validate_dict(data)
        assert not result.is_valid

    def test_parse_error(self):
        """Test that unparseable data produces an error."""
        # GatewayConfig.from_dict should handle this, but we test
        # the error path
        result = ConfigValidator.validate_dict({"providers": "not-a-dict"})
        # This should either fail to parse or produce validation errors
        assert len(result.messages) > 0


# ---------------------------------------------------------------------------
# ExtendedValidator tests
# ---------------------------------------------------------------------------


class TestExtendedValidator:
    """Tests for the ExtendedValidator with custom rules."""

    def test_custom_rule_runs(self, valid_config):
        """Test that custom rules are executed."""
        called = []

        def my_check(config, result):
            called.append(True)
            result.add_info("custom", "Custom rule ran.")

        validator = ExtendedValidator()
        validator.add_rule(
            ValidationRule("my-rule", "A custom rule", my_check)
        )
        result = validator.validate(valid_config)
        assert len(called) == 1
        assert any("Custom rule ran" in m.message for m in result.infos)

    def test_custom_rule_adds_errors(self, valid_config):
        """Test that custom rules can add errors."""

        def strict_check(config, result):
            if config.timeout > 10:
                result.add_error(
                    "timeout", "Timeout too high for strict mode."
                )

        validator = ExtendedValidator()
        validator.add_rule(
            ValidationRule("strict-timeout", "Strict timeout check", strict_check)
        )
        result = validator.validate(valid_config)
        assert not result.is_valid  # valid_config has timeout=30

    def test_rule_exception_caught(self, valid_config):
        """Test that exceptions in rules are caught and reported."""

        def bad_rule(config, result):
            raise ValueError("Intentional error")

        validator = ExtendedValidator()
        validator.add_rule(
            ValidationRule("bad-rule", "Will fail", bad_rule)
        )
        result = validator.validate(valid_config)
        errors = [m for m in result.errors if "bad-rule" in m.path]
        assert len(errors) == 1
        assert "exception" in errors[0].message.lower()

    def test_list_rules(self):
        """Test listing registered rules."""
        validator = ExtendedValidator()
        validator.add_rule(
            ValidationRule("rule-a", "First rule", lambda c, r: None)
        )
        validator.add_rule(
            ValidationRule("rule-b", "Second rule", lambda c, r: None)
        )
        assert validator.list_rules() == ["rule-a", "rule-b"]

    def test_multiple_rules_combined(self, valid_config):
        """Test that multiple rules combine their results."""

        def rule1(config, result):
            result.add_warning("custom.r1", "Warning from rule 1")

        def rule2(config, result):
            result.add_info("custom.r2", "Info from rule 2")

        validator = ExtendedValidator()
        validator.add_rule(ValidationRule("r1", "", rule1))
        validator.add_rule(ValidationRule("r2", "", rule2))
        result = validator.validate(valid_config)

        assert any("rule 1" in m.message for m in result.warnings)
        assert any("rule 2" in m.message for m in result.infos)
