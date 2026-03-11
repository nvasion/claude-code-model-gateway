"""Configuration validation module for claude-code-model-gateway.

Provides validation presets, rule-based validation, and a unified
validation pipeline that combines schema-level checks with semantic
validation from :mod:`src.validation.validator`.

This module acts as the config-package entry point for validation,
bridging the schema definitions in :mod:`src.config.schema` with the
full validation engine in :mod:`src.validation.validator`.

Example usage::

    from src.config.validator import (
        quick_validate,
        full_validate,
        validate_for_production,
    )

    # Quick schema-only check
    errors = quick_validate(config_dict)

    # Full validation with warnings
    result = full_validate(config_object)

    # Strict production check
    ok, report = validate_for_production(config_dict)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from src.config.schema import GATEWAY_SCHEMA, validate_against_schema
from src.models import GatewayConfig
from src.validation.validator import (
    ConfigValidator,
    ExtendedValidator,
    Severity,
    ValidationMessage,
    ValidationResult,
    ValidationRule,
)


# ---------------------------------------------------------------------------
# Validation presets
# ---------------------------------------------------------------------------


class ValidationPreset(str, Enum):
    """Pre-configured validation strictness levels."""

    RELAXED = "relaxed"      # Only errors, ignore warnings
    DEFAULT = "default"      # Errors and warnings
    STRICT = "strict"        # Errors, warnings, and info
    PRODUCTION = "production"  # Everything + extra production checks


@dataclass
class ValidationProfile:
    """A named validation configuration.

    Attributes:
        name: Profile name.
        description: What this profile checks.
        fail_on_warnings: Whether warnings cause validation failure.
        fail_on_info: Whether info messages cause validation failure.
        extra_rules: Additional validation rules for this profile.
        skip_rules: Rule names to skip in this profile.
    """

    name: str
    description: str = ""
    fail_on_warnings: bool = False
    fail_on_info: bool = False
    extra_rules: list[ValidationRule] = field(default_factory=list)
    skip_rules: list[str] = field(default_factory=list)


# Built-in profiles
PROFILES: dict[str, ValidationProfile] = {
    "relaxed": ValidationProfile(
        name="relaxed",
        description="Only fail on errors. Ignore warnings and info.",
        fail_on_warnings=False,
        fail_on_info=False,
    ),
    "default": ValidationProfile(
        name="default",
        description="Fail on errors. Report warnings.",
        fail_on_warnings=False,
        fail_on_info=False,
    ),
    "strict": ValidationProfile(
        name="strict",
        description="Fail on errors and warnings.",
        fail_on_warnings=True,
        fail_on_info=False,
    ),
    "production": ValidationProfile(
        name="production",
        description="Strictest: fail on errors and warnings with extra checks.",
        fail_on_warnings=True,
        fail_on_info=False,
    ),
}


# ---------------------------------------------------------------------------
# Production-specific validation rules
# ---------------------------------------------------------------------------


def _check_no_placeholder_urls(
    config: GatewayConfig, result: ValidationResult
) -> None:
    """Check that no provider has placeholder URLs."""
    for name, provider in config.providers.items():
        if "<" in provider.api_base and ">" in provider.api_base:
            result.add_error(
                f"providers.{name}.api_base",
                "API base URL contains placeholder(s). "
                "Replace with actual values before deploying.",
                provider.api_base,
            )


def _check_api_keys_configured(
    config: GatewayConfig, result: ValidationResult
) -> None:
    """Check that enabled providers with auth have API key env vars."""
    import os
    from src.models import AuthType

    for name, provider in config.providers.items():
        if not provider.enabled:
            continue
        if provider.auth_type == AuthType.NONE:
            continue
        if not provider.api_key_env_var:
            result.add_warning(
                f"providers.{name}.api_key_env_var",
                "No API key environment variable specified.",
                suggestion="Set api_key_env_var for authenticated providers.",
            )


def _check_at_least_one_model(
    config: GatewayConfig, result: ValidationResult
) -> None:
    """Check that enabled providers have at least one model defined."""
    for name, provider in config.providers.items():
        if not provider.enabled:
            continue
        if not provider.models:
            result.add_warning(
                f"providers.{name}.models",
                "Enabled provider has no models defined.",
                suggestion="Add model definitions for explicit configuration.",
            )


def _check_reasonable_timeouts(
    config: GatewayConfig, result: ValidationResult
) -> None:
    """Check that timeout values are reasonable for production."""
    if config.timeout < 5:
        result.add_warning(
            "timeout",
            "Very low timeout may cause premature request failures.",
            config.timeout,
            "Production systems typically use 15-120 seconds.",
        )


_PRODUCTION_RULES: list[ValidationRule] = [
    ValidationRule(
        name="no-placeholder-urls",
        description="Ensure no placeholder URLs remain in provider configs.",
        check_fn=_check_no_placeholder_urls,
    ),
    ValidationRule(
        name="api-keys-configured",
        description="Ensure API key env vars are set for authenticated providers.",
        check_fn=_check_api_keys_configured,
    ),
    ValidationRule(
        name="at-least-one-model",
        description="Ensure enabled providers have model definitions.",
        check_fn=_check_at_least_one_model,
    ),
    ValidationRule(
        name="reasonable-timeouts",
        description="Ensure timeouts are reasonable for production use.",
        check_fn=_check_reasonable_timeouts,
    ),
]

# Add production rules to the production profile
PROFILES["production"].extra_rules = _PRODUCTION_RULES


# ---------------------------------------------------------------------------
# Public validation functions
# ---------------------------------------------------------------------------


def quick_validate(data: dict[str, Any]) -> list[str]:
    """Perform a quick schema-level validation on raw config data.

    This is a lightweight check that validates types and constraints
    without loading the full model. Useful for pre-flight checks
    before expensive operations.

    Args:
        data: Raw configuration dictionary.

    Returns:
        List of error messages (empty if valid).
    """
    return validate_against_schema(data)


def full_validate(
    config: GatewayConfig,
    profile: str = "default",
) -> ValidationResult:
    """Run full validation using the specified profile.

    Args:
        config: The gateway configuration to validate.
        profile: Validation profile name ('relaxed', 'default',
                 'strict', 'production').

    Returns:
        The complete validation result.
    """
    vp = PROFILES.get(profile, PROFILES["default"])

    if vp.extra_rules:
        validator = ExtendedValidator()
        for rule in vp.extra_rules:
            if rule.name not in vp.skip_rules:
                validator.add_rule(rule)
        result = validator.validate(config)
    else:
        result = ConfigValidator.validate(config)

    return result


def full_validate_dict(
    data: dict[str, Any],
    profile: str = "default",
) -> ValidationResult:
    """Run full validation on a raw config dictionary.

    Args:
        data: Raw configuration dictionary.
        profile: Validation profile name.

    Returns:
        The complete validation result.
    """
    try:
        config = GatewayConfig.from_dict(data)
    except Exception as e:
        result = ValidationResult()
        result.add_error("", f"Failed to parse configuration: {e}")
        return result
    return full_validate(config, profile=profile)


def validate_for_production(
    data: dict[str, Any],
) -> tuple[bool, str]:
    """Validate configuration for production readiness.

    Runs the strictest validation profile and returns a pass/fail
    result with a formatted report.

    Args:
        data: Raw configuration dictionary.

    Returns:
        Tuple of (is_valid, report_string).
    """
    result = full_validate_dict(data, profile="production")
    profile = PROFILES["production"]

    is_valid = result.error_count == 0
    if profile.fail_on_warnings:
        is_valid = is_valid and result.warning_count == 0

    report = result.format_report(show_info=True)
    return is_valid, report


def is_valid(data: dict[str, Any]) -> bool:
    """Quick check whether a config dictionary is valid.

    Args:
        data: Raw configuration dictionary.

    Returns:
        True if no errors are found.
    """
    result = ConfigValidator.validate_dict(data)
    return result.is_valid


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------


@dataclass
class ValidationStep:
    """A single step in a validation pipeline.

    Attributes:
        name: Step identifier.
        description: What this step checks.
        validate_fn: Function that takes (data_dict, ValidationResult).
        required: Whether pipeline aborts if this step fails.
    """

    name: str
    description: str
    validate_fn: Callable[[dict[str, Any], ValidationResult], None]
    required: bool = True


class ValidationPipeline:
    """An ordered pipeline of validation steps.

    Steps are executed in order.  If a required step produces errors,
    subsequent steps are skipped.  This enables fast-fail validation
    for expensive downstream checks.

    Example::

        pipeline = ValidationPipeline()
        pipeline.add_step(ValidationStep(
            name="schema",
            description="Schema-level validation",
            validate_fn=schema_check,
            required=True,
        ))
        pipeline.add_step(ValidationStep(
            name="semantic",
            description="Cross-reference checks",
            validate_fn=semantic_check,
        ))
        result = pipeline.run(config_dict)
    """

    def __init__(self) -> None:
        self._steps: list[ValidationStep] = []

    def add_step(self, step: ValidationStep) -> "ValidationPipeline":
        """Add a validation step to the pipeline.

        Args:
            step: The validation step to add.

        Returns:
            Self for method chaining.
        """
        self._steps.append(step)
        return self

    def run(self, data: dict[str, Any]) -> ValidationResult:
        """Execute the validation pipeline.

        Args:
            data: Raw configuration dictionary.

        Returns:
            Aggregated validation result from all executed steps.
        """
        result = ValidationResult()

        for step in self._steps:
            pre_error_count = result.error_count
            try:
                step.validate_fn(data, result)
            except Exception as e:
                result.add_error(
                    f"pipeline.{step.name}",
                    f"Validation step '{step.name}' raised an exception: {e}",
                )

            # If a required step produced errors, stop the pipeline
            if step.required and result.error_count > pre_error_count:
                result.add_info(
                    f"pipeline.{step.name}",
                    f"Pipeline stopped: required step '{step.name}' failed.",
                )
                break

        return result

    def list_steps(self) -> list[str]:
        """Return names of all registered steps.

        Returns:
            Ordered list of step names.
        """
        return [s.name for s in self._steps]


def create_default_pipeline() -> ValidationPipeline:
    """Create the default validation pipeline.

    The default pipeline performs:
    1. Schema-level type and constraint checks
    2. Full semantic validation via ConfigValidator

    Returns:
        A configured ValidationPipeline.
    """

    def schema_step(data: dict[str, Any], result: ValidationResult) -> None:
        """Run schema-level validation."""
        errors = validate_against_schema(data)
        for err in errors:
            path, _, message = err.partition(": ")
            result.add_error(path.strip(), message.strip() if message else err)

    def semantic_step(data: dict[str, Any], result: ValidationResult) -> None:
        """Run full semantic validation."""
        semantic_result = ConfigValidator.validate_dict(data)
        result.messages.extend(semantic_result.messages)

    pipeline = ValidationPipeline()
    pipeline.add_step(
        ValidationStep(
            name="schema",
            description="Schema-level type and constraint validation",
            validate_fn=schema_step,
            required=False,  # Continue even with schema issues
        )
    )
    pipeline.add_step(
        ValidationStep(
            name="semantic",
            description="Semantic cross-reference and best-practice checks",
            validate_fn=semantic_step,
            required=True,
        )
    )
    return pipeline
