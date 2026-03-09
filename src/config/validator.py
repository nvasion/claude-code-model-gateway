"""Unified configuration validation pipeline for the model gateway.

Combines schema-level validation (src.config.schema) with semantic
validation (src.validation.validator) into a configurable pipeline.

Provides validation profiles (relaxed, default, strict, production)
and top-level convenience functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from src.models import AuthType, GatewayConfig, ProviderConfig
from src.validation.validator import ConfigValidator, ValidationResult


# ---------------------------------------------------------------------------
# ValidationPreset
# ---------------------------------------------------------------------------


class ValidationPreset(str, Enum):
    """Named validation preset levels."""

    RELAXED = "relaxed"
    DEFAULT = "default"
    STRICT = "strict"
    PRODUCTION = "production"


# ---------------------------------------------------------------------------
# ValidationProfile
# ---------------------------------------------------------------------------


@dataclass
class ValidationProfile:
    """A named validation profile with specific rules and behaviour.

    Attributes:
        name: Profile identifier.
        description: Human-readable description.
        fail_on_warnings: Whether warnings cause validation to fail.
        fail_on_info: Whether info messages cause validation to fail.
        extra_rules: Additional validation callables (config, result) -> None.
    """

    name: str
    description: str
    fail_on_warnings: bool = False
    fail_on_info: bool = False
    extra_rules: list[Callable[[GatewayConfig, ValidationResult], None]] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Production extra rules
# ---------------------------------------------------------------------------


def _rule_no_placeholder_urls(config: GatewayConfig, result: ValidationResult) -> None:
    """Flag api_base URLs that contain placeholder text like <YOUR-NAME>."""
    for name, provider in config.providers.items():
        if provider.api_base and "<" in provider.api_base:
            result.add_error(
                f"providers.{name}.api_base",
                "API base URL contains placeholder text. Replace before production use.",
                provider.api_base,
            )


def _rule_at_least_one_model(config: GatewayConfig, result: ValidationResult) -> None:
    """Warn if any enabled provider has no models configured."""
    for name, provider in config.providers.items():
        if provider.enabled and not provider.models:
            result.add_warning(
                f"providers.{name}.models",
                "Enabled provider has no models configured.",
                suggestion="Add at least one model definition.",
            )


def _rule_reasonable_timeouts(config: GatewayConfig, result: ValidationResult) -> None:
    """Warn if timeout is unreasonably low for production use."""
    if config.timeout < 5:
        result.add_warning(
            "timeout",
            f"Timeout of {config.timeout}s is very low for production use.",
            config.timeout,
            "Consider at least 30 seconds for LLM API calls.",
        )


def _rule_api_keys_configured(config: GatewayConfig, result: ValidationResult) -> None:
    """Warn if enabled providers requiring auth have no api_key_env_var."""
    for name, provider in config.providers.items():
        if not provider.enabled:
            continue
        if provider.auth_type in (AuthType.API_KEY, AuthType.BEARER_TOKEN):
            if not provider.api_key_env_var:
                result.add_warning(
                    f"providers.{name}.api_key_env_var",
                    "No API key environment variable set for authenticated provider.",
                    suggestion=f"Set api_key_env_var to the env var name for {name}'s API key.",
                )


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------


PROFILES: dict[str, ValidationProfile] = {
    "relaxed": ValidationProfile(
        name="relaxed",
        description="Minimal validation; only errors block deployment.",
        fail_on_warnings=False,
        fail_on_info=False,
    ),
    "default": ValidationProfile(
        name="default",
        description="Standard validation with errors and warnings.",
        fail_on_warnings=False,
        fail_on_info=False,
    ),
    "strict": ValidationProfile(
        name="strict",
        description="Strict validation; warnings are treated as errors.",
        fail_on_warnings=True,
        fail_on_info=False,
    ),
    "production": ValidationProfile(
        name="production",
        description="Production-readiness checks including security and reliability.",
        fail_on_warnings=True,
        fail_on_info=False,
        extra_rules=[
            _rule_no_placeholder_urls,
            _rule_at_least_one_model,
            _rule_reasonable_timeouts,
            _rule_api_keys_configured,
        ],
    ),
}


# ---------------------------------------------------------------------------
# ValidationStep
# ---------------------------------------------------------------------------


@dataclass
class ValidationStep:
    """A single step in a validation pipeline.

    Attributes:
        name: Step identifier.
        description: Human-readable description.
        validate_fn: Callable (data, result) -> None that adds messages.
        required: If True, errors in this step stop subsequent steps.
    """

    name: str
    description: str
    validate_fn: Callable[[Any, ValidationResult], None]
    required: bool = True


# ---------------------------------------------------------------------------
# ValidationPipeline
# ---------------------------------------------------------------------------


class ValidationPipeline:
    """A configurable chain of validation steps.

    Steps run in registration order. If a required step produces errors,
    subsequent steps are skipped.
    """

    def __init__(self) -> None:
        self._steps: list[ValidationStep] = []

    def add_step(self, step: ValidationStep) -> "ValidationPipeline":
        """Add a step to the pipeline.

        Returns:
            self (for method chaining).
        """
        self._steps.append(step)
        return self

    def list_steps(self) -> list[str]:
        """Return step names in order."""
        return [s.name for s in self._steps]

    def run(self, data: Any) -> ValidationResult:
        """Execute all steps in order.

        Args:
            data: The data to validate (dict or GatewayConfig).

        Returns:
            A ValidationResult containing all messages.
        """
        result = ValidationResult()

        for step in self._steps:
            try:
                step.validate_fn(data, result)
            except Exception as exc:
                result.add_error(
                    f"pipeline.{step.name}",
                    f"Step '{step.name}' raised an unexpected error: {exc}",
                )

            # If required step produced errors, stop
            if step.required and result.error_count > 0:
                break

        return result


# ---------------------------------------------------------------------------
# Schema-level quick validation
# ---------------------------------------------------------------------------


def quick_validate(data: dict[str, Any]) -> list[str]:
    """Run schema-level validation on a raw configuration dict.

    Args:
        data: Configuration dictionary to validate.

    Returns:
        List of error strings (empty if valid).
    """
    from src.config.schema import validate_against_schema

    return validate_against_schema(data)


# ---------------------------------------------------------------------------
# Full semantic validation
# ---------------------------------------------------------------------------


def full_validate(
    config: GatewayConfig,
    profile: str = "default",
) -> ValidationResult:
    """Run full semantic validation on a GatewayConfig.

    Args:
        config: The configuration to validate.
        profile: Profile name (default, relaxed, strict, production).
                 Falls back to 'default' if the name is unknown.

    Returns:
        ValidationResult with all discovered issues.
    """
    active_profile = PROFILES.get(profile, PROFILES["default"])
    result = ConfigValidator.validate(config)

    # Run extra rules for this profile
    for rule in active_profile.extra_rules:
        try:
            rule(config, result)
        except Exception as exc:
            result.add_error(
                "profile_rule",
                f"Profile rule raised an error: {exc}",
            )

    return result


def full_validate_dict(
    data: dict[str, Any],
    profile: str = "default",
) -> ValidationResult:
    """Run full semantic validation on a configuration dictionary.

    Parses the dict into a GatewayConfig first; on parse failure,
    returns a result with a parse error.

    Args:
        data: Raw configuration dictionary.
        profile: Profile name.

    Returns:
        ValidationResult with all discovered issues.
    """
    try:
        config = GatewayConfig.from_dict(data)
    except Exception as exc:
        result = ValidationResult()
        result.add_error("", f"Failed to parse configuration: {exc}")
        return result
    return full_validate(config, profile=profile)


def validate_for_production(data: dict[str, Any]) -> tuple[bool, str]:
    """Validate a configuration dict for production readiness.

    Uses the 'production' profile which treats warnings as failures.

    Args:
        data: Raw configuration dictionary.

    Returns:
        Tuple of (ok: bool, report: str).
        ok is True only if there are no errors AND no warnings (production
        profile has fail_on_warnings=True).
    """
    result = full_validate_dict(data, profile="production")
    production_profile = PROFILES["production"]

    ok = result.is_valid
    if production_profile.fail_on_warnings and result.warning_count > 0:
        ok = False

    report = result.format_report(show_info=False)
    return ok, report


def is_valid(data: dict[str, Any]) -> bool:
    """Quick check: does the configuration have any errors?

    Args:
        data: Raw configuration dictionary.

    Returns:
        True if the configuration has no errors.
    """
    result = full_validate_dict(data, profile="default")
    return result.is_valid


# ---------------------------------------------------------------------------
# Default pipeline factory
# ---------------------------------------------------------------------------


def _schema_step(data: Any, result: ValidationResult) -> None:
    """Pipeline step: schema-level type/constraint checks."""
    if isinstance(data, dict):
        errors = quick_validate(data)
        for error in errors:
            result.add_error("schema", error)


def _semantic_step(data: Any, result: ValidationResult) -> None:
    """Pipeline step: semantic validation via ConfigValidator."""
    if isinstance(data, dict):
        try:
            config = GatewayConfig.from_dict(data)
        except Exception as exc:
            result.add_error("semantic", f"Failed to parse configuration: {exc}")
            return
    elif isinstance(data, GatewayConfig):
        config = data
    else:
        result.add_error("semantic", f"Expected dict or GatewayConfig, got {type(data)}")
        return

    semantic_result = ConfigValidator.validate(config)
    for msg in semantic_result.messages:
        result.messages.append(msg)


def create_default_pipeline() -> ValidationPipeline:
    """Create the default two-step validation pipeline.

    Steps:
    1. 'schema' — schema-level type/constraint checks (required).
    2. 'semantic' — semantic validation via ConfigValidator (optional).

    Returns:
        A configured ValidationPipeline.
    """
    pipeline = ValidationPipeline()
    pipeline.add_step(
        ValidationStep(
            name="schema",
            description="Schema-level type and constraint checks.",
            validate_fn=_schema_step,
            required=True,
        )
    )
    pipeline.add_step(
        ValidationStep(
            name="semantic",
            description="Semantic validation for business rules.",
            validate_fn=_semantic_step,
            required=False,
        )
    )
    return pipeline
