"""Test helpers for configuration validation testing.

Provides factory functions and assertion helpers to simplify writing
tests for gateway configuration.
"""

from __future__ import annotations

from typing import Optional

from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig
from src.validation.validator import ConfigValidator, ValidationResult


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def make_model(
    name: str = "test-model",
    max_tokens: int = 4096,
    supports_streaming: bool = True,
    supports_tools: bool = False,
    supports_vision: bool = False,
    display_name: str = "",
    extra: Optional[dict] = None,
) -> ModelConfig:
    """Create a ModelConfig for testing.

    Args:
        name: Model identifier.
        max_tokens: Maximum token limit.
        supports_streaming: Whether streaming is supported.
        supports_tools: Whether tools are supported.
        supports_vision: Whether vision is supported.
        display_name: Human-readable name (defaults to name).
        extra: Additional model configuration.

    Returns:
        A configured ModelConfig instance.
    """
    return ModelConfig(
        name=name,
        display_name=display_name or name,
        max_tokens=max_tokens,
        supports_streaming=supports_streaming,
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        extra=extra or {},
    )


def make_provider(
    name: str = "test-provider",
    api_base: str = "https://api.example.com/v1",
    api_key_env_var: str = "TEST_API_KEY",
    auth_type: AuthType = AuthType.API_KEY,
    default_model: str = "test-model",
    models: Optional[dict[str, ModelConfig]] = None,
    enabled: bool = True,
    display_name: str = "",
    headers: Optional[dict[str, str]] = None,
    extra: Optional[dict] = None,
) -> ProviderConfig:
    """Create a ProviderConfig for testing.

    If *models* is None, a default model is created using *default_model* as key.

    Args:
        name: Provider identifier.
        api_base: API base URL.
        api_key_env_var: Env var for API key.
        auth_type: Authentication type.
        default_model: Default model name.
        models: Model configs dict. Auto-generated if None.
        enabled: Whether the provider is enabled.
        display_name: Human-readable name (defaults to name).
        headers: Additional HTTP headers.
        extra: Additional provider configuration.

    Returns:
        A configured ProviderConfig instance.
    """
    if models is None:
        models = {default_model: make_model(name=default_model)}

    return ProviderConfig(
        name=name,
        display_name=display_name or name,
        api_base=api_base,
        api_key_env_var=api_key_env_var,
        auth_type=auth_type,
        default_model=default_model,
        models=models,
        enabled=enabled,
        headers=headers or {},
        extra=extra or {},
    )


def make_valid_config(
    default_provider: str = "test-provider",
    providers: Optional[dict[str, ProviderConfig]] = None,
    log_level: str = "info",
    timeout: int = 30,
    max_retries: int = 3,
) -> GatewayConfig:
    """Create a valid GatewayConfig for testing.

    Args:
        default_provider: Name of the default provider.
        providers: Provider configs. Auto-generated if None.
        log_level: Logging level.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.

    Returns:
        A valid GatewayConfig instance.
    """
    if providers is None:
        providers = {default_provider: make_provider(name=default_provider)}

    return GatewayConfig(
        default_provider=default_provider,
        providers=providers,
        log_level=log_level,
        timeout=timeout,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_config_valid(
    config: GatewayConfig,
    msg: str = "",
) -> ValidationResult:
    """Assert that a GatewayConfig is valid (has no errors).

    Args:
        config: The configuration to validate.
        msg: Optional context message to include in AssertionError.

    Returns:
        The ValidationResult if valid.

    Raises:
        AssertionError: If the configuration has any errors.
    """
    result = ConfigValidator.validate(config)
    if not result.is_valid:
        error_msgs = [str(e) for e in result.errors]
        context = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{context}Expected config to be valid, but it has "
            f"{result.error_count} error(s):\n"
            + "\n".join(f"  - {m}" for m in error_msgs)
        )
    return result


def assert_config_invalid(
    config: GatewayConfig,
    msg: str = "",
) -> ValidationResult:
    """Assert that a GatewayConfig is invalid (has at least one error).

    Args:
        config: The configuration to validate.
        msg: Optional context message to include in AssertionError.

    Returns:
        The ValidationResult if invalid.

    Raises:
        AssertionError: If the configuration is valid (no errors).
    """
    result = ConfigValidator.validate(config)
    if result.is_valid:
        context = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{context}Expected config to be invalid, but it has no errors."
        )
    return result


def assert_has_error_at(
    result: ValidationResult,
    path: str,
    msg: str = "",
) -> None:
    """Assert that a ValidationResult has at least one error at *path*.

    Args:
        result: The validation result to check.
        path: The dotted path to check for errors.
        msg: Optional context message.

    Raises:
        AssertionError: If there's no error at the given path.
    """
    matching = [e for e in result.errors if path in e.path]
    if not matching:
        actual_paths = [e.path for e in result.errors]
        context = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{context}Expected error at path '{path}', but none found.\n"
            f"  Actual error paths: {actual_paths}"
        )


def assert_has_warning_at(
    result: ValidationResult,
    path: str,
    msg: str = "",
) -> None:
    """Assert that a ValidationResult has at least one warning at *path*.

    Args:
        result: The validation result to check.
        path: The dotted path to check for warnings.
        msg: Optional context message.

    Raises:
        AssertionError: If there's no warning at the given path.
    """
    matching = [w for w in result.warnings if path in w.path]
    if not matching:
        actual_paths = [w.path for w in result.warnings]
        context = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{context}Expected warning at path '{path}', but none found.\n"
            f"  Actual warning paths: {actual_paths}"
        )


# ---------------------------------------------------------------------------
# Built-in test suite
# ---------------------------------------------------------------------------


def run_builtin_suite():
    """Run the built-in configuration test suite.

    Returns:
        A ConfigTestSuiteResult from src.validation.testing.
    """
    from src.validation.testing import (
        ConfigTestCase,
        ConfigTestSuiteResult,
        run_test_suite,
    )

    # Define built-in test cases using GatewayConfig objects directly
    test_cases = [
        ConfigTestCase(
            name="valid_empty_config",
            description="Empty GatewayConfig should be valid (no providers = warning only).",
            config=GatewayConfig(),
            expect_valid=True,
        ),
        ConfigTestCase(
            name="valid_full_config",
            description="A fully configured gateway should be valid.",
            config=make_valid_config(),
            expect_valid=True,
        ),
        ConfigTestCase(
            name="invalid_negative_timeout",
            description="Negative timeout should be invalid.",
            config=GatewayConfig(timeout=-1),
            expect_valid=False,
        ),
        ConfigTestCase(
            name="invalid_bad_log_level",
            description="Invalid log level should be invalid.",
            config=GatewayConfig(log_level="extreme"),
            expect_valid=False,
        ),
        ConfigTestCase(
            name="invalid_missing_default_provider",
            description="Non-existent default provider should be invalid.",
            config=GatewayConfig(
                default_provider="nonexistent",
                providers={"other": make_provider(name="other")},
            ),
            expect_valid=False,
        ),
    ]

    return run_test_suite(test_cases)
