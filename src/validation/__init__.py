"""Configuration validation and testing tools for claude-code-model-gateway.

Provides comprehensive validation with detailed error/warning reporting,
configuration testing utilities, diff tools, and built-in test suites.
"""

from src.validation.validator import (
    ConfigValidator,
    Severity,
    ValidationMessage,
    ValidationResult,
)
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

__all__ = [
    "ConfigDiff",
    "ConfigTestCase",
    "ConfigTestResult",
    "ConfigTestRunner",
    "ConfigTestSuiteResult",
    "ConfigValidator",
    "Severity",
    "ValidationMessage",
    "ValidationResult",
    "create_test_config_with_overrides",
    "generate_minimal_config",
    "generate_sample_config",
    "get_builtin_test_cases",
]
