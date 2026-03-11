"""Configuration testing utilities for the config package.

Re-exports and augments the core testing tools from
:mod:`src.validation.testing` with config-package–specific helpers,
making it easy to write configuration tests without deep import paths.

Example usage::

    from src.config.testing import (
        make_valid_config,
        make_provider,
        make_model,
        assert_config_valid,
        assert_config_invalid,
        assert_has_error_at,
        assert_has_warning_at,
    )

    def test_my_provider():
        provider = make_provider("myprovider", api_base="https://api.example.com")
        config = make_valid_config(providers={"myprovider": provider})
        assert_config_valid(config)
"""

from __future__ import annotations

from typing import Any, Optional

from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig
from src.validation.testing import (
    ConfigDiff,
    ConfigTestCase,
    ConfigTestResult,
    ConfigTestRunner,
    ConfigTestSuiteResult,
    create_test_config_with_overrides,
    generate_minimal_config,
    generate_sample_config,
    get_builtin_test_cases,
)
from src.validation.validator import ConfigValidator, ValidationResult

__all__ = [
    # Re-exports from validation.testing
    "ConfigDiff",
    "ConfigTestCase",
    "ConfigTestResult",
    "ConfigTestRunner",
    "ConfigTestSuiteResult",
    "create_test_config_with_overrides",
    "generate_minimal_config",
    "generate_sample_config",
    "get_builtin_test_cases",
    # Helpers defined here
    "make_model",
    "make_provider",
    "make_valid_config",
    "assert_config_valid",
    "assert_config_invalid",
    "assert_has_error_at",
    "assert_has_warning_at",
    "run_builtin_suite",
]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_model(
    name: str = "test-model",
    max_tokens: int = 4096,
    supports_streaming: bool = True,
    supports_tools: bool = False,
    supports_vision: bool = False,
    display_name: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> ModelConfig:
    """Create a ModelConfig with sensible defaults for testing.

    Args:
        name: Model identifier.
        max_tokens: Maximum token limit (must be > 0).
        supports_streaming: Whether streaming is supported.
        supports_tools: Whether tool calling is supported.
        supports_vision: Whether vision input is supported.
        display_name: Human-readable name.
        extra: Additional model-specific config.

    Returns:
        A ModelConfig instance ready for use in tests.
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
    extra: Optional[dict[str, Any]] = None,
) -> ProviderConfig:
    """Create a ProviderConfig with sensible defaults for testing.

    If *models* is None, a single model (named *default_model*) is
    created automatically so the provider passes validation.

    Args:
        name: Provider identifier.
        api_base: Base URL for the provider API.
        api_key_env_var: Environment variable holding the API key.
        auth_type: Authentication type.
        default_model: Name of the default model.
        models: Model configs keyed by name.  Auto-populated if None.
        enabled: Whether the provider is enabled.
        display_name: Human-readable provider name.
        headers: Additional HTTP headers.
        extra: Extra provider-specific config.

    Returns:
        A ProviderConfig instance ready for use in tests.
    """
    if models is None:
        models = {
            default_model: make_model(name=default_model),
        }
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
    """Create a GatewayConfig that passes validation out of the box.

    If *providers* is None, a single provider (named *default_provider*)
    is created automatically using :func:`make_provider`.

    Args:
        default_provider: Name of the default provider.
        providers: Provider configs keyed by name.  Auto-populated if None.
        log_level: Logging level string.
        timeout: Request timeout in seconds.
        max_retries: Max number of retries.

    Returns:
        A GatewayConfig that passes :func:`src.config.validate_config`.
    """
    if providers is None:
        providers = {
            default_provider: make_provider(name=default_provider),
        }
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


def assert_config_valid(config: GatewayConfig, msg: str = "") -> ValidationResult:
    """Assert that a GatewayConfig passes comprehensive validation.

    Args:
        config: The configuration to validate.
        msg: Additional message to include in the assertion error.

    Returns:
        The ValidationResult for further inspection.

    Raises:
        AssertionError: If the configuration has any errors.
    """
    result = ConfigValidator.validate(config)
    if not result.is_valid:
        error_messages = "\n".join(str(e) for e in result.errors)
        prefix = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{prefix}Configuration expected to be valid, but has "
            f"{result.error_count} error(s):\n{error_messages}"
        )
    return result


def assert_config_invalid(config: GatewayConfig, msg: str = "") -> ValidationResult:
    """Assert that a GatewayConfig fails validation (has at least one error).

    Args:
        config: The configuration to validate.
        msg: Additional message to include in the assertion error.

    Returns:
        The ValidationResult for further inspection.

    Raises:
        AssertionError: If the configuration unexpectedly passes validation.
    """
    result = ConfigValidator.validate(config)
    if result.is_valid:
        prefix = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{prefix}Configuration expected to be invalid, but passed validation "
            f"with {result.warning_count} warning(s) and no errors."
        )
    return result


def assert_has_error_at(
    result: ValidationResult,
    path: str,
    msg: str = "",
) -> None:
    """Assert that a ValidationResult has at least one error at *path*.

    Args:
        result: The validation result to inspect.
        path: Dotted config path expected to have an error
              (e.g., 'providers.openai.api_base').
        msg: Additional message to include in the assertion error.

    Raises:
        AssertionError: If no error is found at the specified path.
    """
    error_paths = {m.path for m in result.errors}
    if path not in error_paths:
        prefix = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{prefix}Expected an error at '{path}', but none found. "
            f"Actual error paths: {sorted(error_paths) or 'none'}"
        )


def assert_has_warning_at(
    result: ValidationResult,
    path: str,
    msg: str = "",
) -> None:
    """Assert that a ValidationResult has at least one warning at *path*.

    Args:
        result: The validation result to inspect.
        path: Dotted config path expected to have a warning.
        msg: Additional message to include in the assertion error.

    Raises:
        AssertionError: If no warning is found at the specified path.
    """
    warning_paths = {m.path for m in result.warnings}
    if path not in warning_paths:
        prefix = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{prefix}Expected a warning at '{path}', but none found. "
            f"Actual warning paths: {sorted(warning_paths) or 'none'}"
        )


# ---------------------------------------------------------------------------
# Suite runner convenience
# ---------------------------------------------------------------------------


def run_builtin_suite() -> ConfigTestSuiteResult:
    """Run the complete built-in configuration test suite.

    Convenience wrapper that fetches the built-in test cases and runs
    them all, returning the aggregated suite result.

    Returns:
        A ConfigTestSuiteResult summarising all test outcomes.
    """
    cases = get_builtin_test_cases()
    return ConfigTestRunner.run_suite(cases)
