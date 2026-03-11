"""Configuration validator for claude-code-model-gateway.

Provides comprehensive validation of gateway configuration with detailed
error, warning, and informational messages. Validates types, value
constraints, cross-field dependencies, and semantic consistency against
the existing GatewayConfig / ProviderConfig / ModelConfig data models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig


class Severity(str, Enum):
    """Severity level for validation messages."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationMessage:
    """A single validation message with path and context.

    Attributes:
        severity: The severity level (error, warning, info).
        path: Dotted path to the config field (e.g., 'providers.openai.api_base').
        message: Human-readable description of the issue.
        value: The actual value that caused the issue, if applicable.
        suggestion: Suggested fix, if applicable.
    """

    severity: Severity
    path: str
    message: str
    value: Any = None
    suggestion: str = ""

    def __str__(self) -> str:
        """Format the message for human-readable display."""
        prefix = self.severity.value.upper()
        parts = [f"[{prefix}] {self.path}: {self.message}"]
        if self.value is not None:
            parts.append(f"  Got: {self.value!r}")
        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON output."""
        return {
            "severity": self.severity.value,
            "path": self.path,
            "message": self.message,
            "value": str(self.value) if self.value is not None else None,
            "suggestion": self.suggestion or None,
        }


@dataclass
class ValidationResult:
    """Aggregated result of configuration validation.

    Attributes:
        messages: List of all validation messages.
    """

    messages: list[ValidationMessage] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationMessage]:
        """Return only error-level messages."""
        return [m for m in self.messages if m.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationMessage]:
        """Return only warning-level messages."""
        return [m for m in self.messages if m.severity == Severity.WARNING]

    @property
    def infos(self) -> list[ValidationMessage]:
        """Return only info-level messages."""
        return [m for m in self.messages if m.severity == Severity.INFO]

    @property
    def is_valid(self) -> bool:
        """Whether the configuration has no errors (warnings are allowed)."""
        return len(self.errors) == 0

    @property
    def error_count(self) -> int:
        """Number of error-level messages."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of warning-level messages."""
        return len(self.warnings)

    def add_error(
        self,
        path: str,
        message: str,
        value: Any = None,
        suggestion: str = "",
    ) -> None:
        """Add an error message."""
        self.messages.append(
            ValidationMessage(Severity.ERROR, path, message, value, suggestion)
        )

    def add_warning(
        self,
        path: str,
        message: str,
        value: Any = None,
        suggestion: str = "",
    ) -> None:
        """Add a warning message."""
        self.messages.append(
            ValidationMessage(Severity.WARNING, path, message, value, suggestion)
        )

    def add_info(
        self,
        path: str,
        message: str,
        value: Any = None,
        suggestion: str = "",
    ) -> None:
        """Add an info message."""
        self.messages.append(
            ValidationMessage(Severity.INFO, path, message, value, suggestion)
        )

    def summary(self) -> str:
        """Return a summary string of the validation result."""
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"Configuration is {status}\n"
            f"  {self.error_count} error(s), "
            f"{self.warning_count} warning(s), "
            f"{len(self.infos)} info(s)"
        )

    def format_report(self, show_info: bool = False) -> str:
        """Format a full validation report.

        Args:
            show_info: Whether to include info-level messages.

        Returns:
            Formatted multi-line report string.
        """
        lines = [self.summary(), ""]
        for msg in self.messages:
            if msg.severity == Severity.INFO and not show_info:
                continue
            lines.append(str(msg))
        return "\n".join(lines)


# URL pattern for basic URL validation
_URL_PATTERN = re.compile(r"^https?://\S+$")

# Valid log levels
VALID_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


class ConfigValidator:
    """Validates GatewayConfig instances against constraints and best practices.

    Performs structural validation, value range checks, cross-field
    consistency checks, and best-practice warnings against the existing
    GatewayConfig / ProviderConfig / ModelConfig models.
    """

    @classmethod
    def validate(cls, config: GatewayConfig) -> ValidationResult:
        """Run all validation checks on a GatewayConfig.

        Args:
            config: The gateway configuration to validate.

        Returns:
            A ValidationResult with all discovered issues.
        """
        result = ValidationResult()
        cls._validate_top_level(config, result)
        cls._validate_providers(config, result)
        cls._validate_cross_references(config, result)
        return result

    @classmethod
    def validate_dict(cls, data: dict[str, Any]) -> ValidationResult:
        """Validate a raw configuration dictionary.

        Parses the dict into a GatewayConfig and validates it.
        Catches parsing errors and reports them as validation errors.

        Args:
            data: Raw configuration dictionary.

        Returns:
            A ValidationResult with all discovered issues.
        """
        try:
            config = GatewayConfig.from_dict(data)
        except Exception as e:
            result = ValidationResult()
            result.add_error("", f"Failed to parse configuration: {e}")
            return result
        return cls.validate(config)

    @classmethod
    def _validate_top_level(
        cls, config: GatewayConfig, result: ValidationResult
    ) -> None:
        """Validate top-level gateway settings."""
        # Log level
        if config.log_level.lower() not in VALID_LOG_LEVELS:
            result.add_error(
                "log_level",
                f"Invalid log level '{config.log_level}'.",
                config.log_level,
                f"Use one of: {', '.join(sorted(VALID_LOG_LEVELS))}.",
            )

        # Timeout
        if config.timeout <= 0:
            result.add_error(
                "timeout",
                "Timeout must be a positive integer.",
                config.timeout,
                "Use a value like 30 or 60.",
            )
        elif config.timeout > 300:
            result.add_warning(
                "timeout",
                "Very high default timeout.",
                config.timeout,
                "Typical API timeouts are 30-120 seconds.",
            )

        # Max retries
        if config.max_retries < 0:
            result.add_error(
                "max_retries",
                "Max retries must be non-negative.",
                config.max_retries,
            )
        elif config.max_retries > 10:
            result.add_warning(
                "max_retries",
                "High retry count may cause long delays.",
                config.max_retries,
                "Typical values are 1-5.",
            )

        # Providers presence
        if not config.providers:
            result.add_warning(
                "providers",
                "No providers configured.",
                suggestion="Add at least one provider with 'config init' "
                "or 'provider add'.",
            )

    @classmethod
    def _validate_providers(
        cls, config: GatewayConfig, result: ValidationResult
    ) -> None:
        """Validate individual provider configurations."""
        enabled_count = 0

        for name, provider in config.providers.items():
            prefix = f"providers.{name}"

            # Name consistency
            if provider.name != name:
                result.add_warning(
                    f"{prefix}.name",
                    f"Provider name '{provider.name}' does not match "
                    f"its key '{name}'.",
                    provider.name,
                    f"Set the provider name to '{name}'.",
                )

            # API base
            if not provider.api_base:
                result.add_error(
                    f"{prefix}.api_base",
                    "API base URL is required.",
                    suggestion="Provide a URL like 'https://api.openai.com/v1'.",
                )
            elif "<" in provider.api_base:
                # Template URL with <placeholders> — warn regardless of URL pattern match
                result.add_warning(
                    f"{prefix}.api_base",
                    "API base contains placeholder(s). "
                    "Replace before deployment.",
                    provider.api_base,
                )
            elif not _URL_PATTERN.match(provider.api_base):
                result.add_error(
                    f"{prefix}.api_base",
                    "API base must be a valid HTTP(S) URL.",
                    provider.api_base,
                    "Use a URL like 'https://api.openai.com/v1'.",
                )

            # Auth type
            try:
                AuthType(provider.auth_type) if isinstance(
                    provider.auth_type, str
                ) else None
            except ValueError:
                if isinstance(provider.auth_type, str):
                    result.add_error(
                        f"{prefix}.auth_type",
                        f"Unknown auth type '{provider.auth_type}'.",
                        provider.auth_type,
                        f"Use one of: "
                        f"{', '.join(t.value for t in AuthType)}.",
                    )

            # API key env var
            if (
                provider.auth_type != AuthType.NONE
                and not provider.api_key_env_var
            ):
                result.add_warning(
                    f"{prefix}.api_key_env_var",
                    "No API key environment variable specified for "
                    f"authenticated provider.",
                    suggestion="Set api_key_env_var to the env var name "
                    "containing the API key.",
                )

            # Models
            cls._validate_provider_models(provider, prefix, result)

            if provider.enabled:
                enabled_count += 1

        if config.providers and enabled_count == 0:
            result.add_error(
                "providers",
                "All providers are disabled. At least one must be enabled.",
                suggestion="Enable a provider with 'provider enable <name>'.",
            )

    @classmethod
    def _validate_provider_models(
        cls,
        provider: ProviderConfig,
        prefix: str,
        result: ValidationResult,
    ) -> None:
        """Validate models within a provider."""
        if not provider.models:
            result.add_info(
                f"{prefix}.models",
                "No models defined for this provider.",
                suggestion="Add model definitions for better validation.",
            )
            return

        for model_name, model in provider.models.items():
            model_prefix = f"{prefix}.models.{model_name}"

            # Name consistency
            if model.name != model_name:
                result.add_warning(
                    f"{model_prefix}.name",
                    f"Model name '{model.name}' does not match "
                    f"its key '{model_name}'.",
                    model.name,
                )

            # Max tokens
            if model.max_tokens <= 0:
                result.add_error(
                    f"{model_prefix}.max_tokens",
                    "max_tokens must be a positive integer.",
                    model.max_tokens,
                )
            elif model.max_tokens > 2_000_000:
                result.add_warning(
                    f"{model_prefix}.max_tokens",
                    "Very high max_tokens value.",
                    model.max_tokens,
                    "Most models support up to 200K tokens.",
                )

        # Default model reference
        if provider.default_model and provider.models:
            if provider.default_model not in provider.models:
                result.add_error(
                    f"{prefix}.default_model",
                    f"Default model '{provider.default_model}' is not "
                    f"in the models list.",
                    provider.default_model,
                    f"Available: {', '.join(sorted(provider.models.keys()))}.",
                )

    @classmethod
    def _validate_cross_references(
        cls, config: GatewayConfig, result: ValidationResult
    ) -> None:
        """Validate cross-field references and consistency."""
        provider_names = set(config.providers.keys())
        enabled_names = {
            name
            for name, p in config.providers.items()
            if p.enabled
        }

        # Default provider
        if config.default_provider:
            if config.default_provider not in provider_names:
                result.add_error(
                    "default_provider",
                    f"Default provider '{config.default_provider}' "
                    f"is not in the providers list.",
                    config.default_provider,
                    f"Available: {', '.join(sorted(provider_names)) or 'none'}.",
                )
            elif config.default_provider not in enabled_names:
                result.add_warning(
                    "default_provider",
                    f"Default provider '{config.default_provider}' is disabled.",
                    suggestion="Enable it or choose an enabled provider.",
                )
        elif config.providers:
            result.add_warning(
                "default_provider",
                "No default provider specified.",
                suggestion="Set default_provider to one of: "
                f"{', '.join(sorted(provider_names))}.",
            )


class ValidationRule:
    """A named, reusable validation rule.

    Allows registering custom validation checks that integrate with
    the standard ValidationResult reporting.

    Attributes:
        name: A short identifier for this rule.
        description: What this rule checks.
        check_fn: Callable that takes (GatewayConfig, ValidationResult).
    """

    def __init__(
        self,
        name: str,
        description: str,
        check_fn: Callable[[GatewayConfig, ValidationResult], None],
    ) -> None:
        self.name = name
        self.description = description
        self.check_fn = check_fn

    def run(self, config: GatewayConfig, result: ValidationResult) -> None:
        """Execute this validation rule.

        Args:
            config: The configuration to validate.
            result: The result to append messages to.
        """
        self.check_fn(config, result)


class ExtendedValidator:
    """Extended validator that supports custom rules.

    Combines the built-in ConfigValidator checks with user-defined
    ValidationRule instances.
    """

    def __init__(self) -> None:
        self._rules: list[ValidationRule] = []

    def add_rule(self, rule: ValidationRule) -> None:
        """Register a custom validation rule.

        Args:
            rule: The validation rule to add.
        """
        self._rules.append(rule)

    def validate(self, config: GatewayConfig) -> ValidationResult:
        """Run built-in and custom validation.

        Args:
            config: The gateway configuration to validate.

        Returns:
            Combined ValidationResult.
        """
        result = ConfigValidator.validate(config)
        for rule in self._rules:
            try:
                rule.run(config, result)
            except Exception as e:
                result.add_error(
                    f"rule.{rule.name}",
                    f"Validation rule '{rule.name}' raised an exception: {e}",
                )
        return result

    def list_rules(self) -> list[str]:
        """Return names of all registered custom rules."""
        return [r.name for r in self._rules]
