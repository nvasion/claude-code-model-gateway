"""Configuration testing utilities for claude-code-model-gateway.

Provides tools for running automated configuration tests, generating
test configs, comparing configurations, and creating test fixtures
based on the existing GatewayConfig / ProviderConfig data models.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import ConfigError, load_config_file
from src.models import GatewayConfig, ProviderConfig, ModelConfig, AuthType
from src.validation.validator import ConfigValidator, Severity, ValidationResult


@dataclass
class ConfigTestCase:
    """A single configuration test case.

    Attributes:
        name: Short descriptive name for the test case.
        description: Longer explanation of what this test checks.
        config_data: The configuration dictionary to test.
        expect_valid: Whether this config is expected to pass validation.
        expect_errors_at: List of config paths expected to have errors.
        expect_warnings_at: List of config paths expected to have warnings.
    """

    name: str
    description: str = ""
    config_data: dict[str, Any] = field(default_factory=dict)
    expect_valid: bool = True
    expect_errors_at: list[str] = field(default_factory=list)
    expect_warnings_at: list[str] = field(default_factory=list)


@dataclass
class ConfigTestResult:
    """Result of running a single config test case.

    Attributes:
        test_case: The test case that was run.
        passed: Whether the test passed.
        validation_result: The full validation result.
        failure_reason: Human-readable description of why it failed.
    """

    test_case: ConfigTestCase
    passed: bool
    validation_result: ValidationResult
    failure_reason: str = ""


@dataclass
class ConfigTestSuiteResult:
    """Result of running an entire test suite.

    Attributes:
        results: List of individual test results.
    """

    results: list[ConfigTestResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of tests run."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Number of tests that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        """Number of tests that failed."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_passed(self) -> bool:
        """Whether all tests passed."""
        return self.failed == 0

    def format_report(self) -> str:
        """Format a full test suite report.

        Returns:
            Multi-line formatted report string.
        """
        lines = [
            f"Configuration Test Results: {self.passed}/{self.total} passed",
            "=" * 60,
        ]
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"  [{status}] {result.test_case.name}")
            if not result.passed:
                lines.append(f"         Reason: {result.failure_reason}")
        lines.append("=" * 60)
        if self.all_passed:
            lines.append("All tests passed!")
        else:
            lines.append(f"{self.failed} test(s) failed.")
        return "\n".join(lines)


class ConfigTestRunner:
    """Runs configuration test cases and reports results."""

    @classmethod
    def run_test(cls, test_case: ConfigTestCase) -> ConfigTestResult:
        """Run a single configuration test case.

        Args:
            test_case: The test case to execute.

        Returns:
            The test result.
        """
        validation = ConfigValidator.validate_dict(test_case.config_data)

        passed = True
        reasons: list[str] = []

        # Check validity expectation
        if test_case.expect_valid and not validation.is_valid:
            passed = False
            error_paths = [m.path for m in validation.errors]
            reasons.append(
                f"Expected valid, but got {validation.error_count} error(s) "
                f"at: {', '.join(error_paths)}"
            )
        elif not test_case.expect_valid and validation.is_valid:
            passed = False
            reasons.append("Expected invalid, but config passed validation.")

        # Check expected error paths
        error_paths_set = {m.path for m in validation.errors}
        for expected_path in test_case.expect_errors_at:
            if expected_path not in error_paths_set:
                passed = False
                reasons.append(
                    f"Expected error at '{expected_path}', but none found. "
                    f"Actual errors at: {', '.join(sorted(error_paths_set)) or 'none'}"
                )

        # Check expected warning paths
        warning_paths_set = {m.path for m in validation.warnings}
        for expected_path in test_case.expect_warnings_at:
            if expected_path not in warning_paths_set:
                passed = False
                reasons.append(
                    f"Expected warning at '{expected_path}', but none found. "
                    f"Actual warnings at: "
                    f"{', '.join(sorted(warning_paths_set)) or 'none'}"
                )

        return ConfigTestResult(
            test_case=test_case,
            passed=passed,
            validation_result=validation,
            failure_reason="; ".join(reasons),
        )

    @classmethod
    def run_suite(
        cls, test_cases: list[ConfigTestCase]
    ) -> ConfigTestSuiteResult:
        """Run a list of test cases and return aggregated results.

        Args:
            test_cases: List of test cases to execute.

        Returns:
            Aggregated test suite results.
        """
        suite_result = ConfigTestSuiteResult()
        for test_case in test_cases:
            result = cls.run_test(test_case)
            suite_result.results.append(result)
        return suite_result

    @classmethod
    def run_file_tests(cls, config_path: str | Path) -> ConfigTestResult:
        """Run validation tests on a configuration file.

        Args:
            config_path: Path to the configuration file to test.

        Returns:
            The test result.
        """
        path = Path(config_path)
        test_case = ConfigTestCase(
            name=f"File: {path.name}",
            description=f"Validate configuration file: {path}",
            expect_valid=True,
        )

        try:
            data = load_config_file(path)
            test_case.config_data = data
        except ConfigError as e:
            validation = ValidationResult()
            validation.add_error("", f"Failed to load config: {e}")
            return ConfigTestResult(
                test_case=test_case,
                passed=False,
                validation_result=validation,
                failure_reason=str(e),
            )

        return cls.run_test(test_case)


class ConfigDiff:
    """Compare two configurations and report differences."""

    @classmethod
    def diff(
        cls,
        config_a: dict[str, Any],
        config_b: dict[str, Any],
        path: str = "",
    ) -> list[str]:
        """Compare two config dictionaries and return a list of differences.

        Args:
            config_a: First configuration (labeled 'A').
            config_b: Second configuration (labeled 'B').
            path: Current path prefix for nested diffs.

        Returns:
            List of human-readable difference strings.
        """
        diffs: list[str] = []
        all_keys = sorted(set(list(config_a.keys()) + list(config_b.keys())))

        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            in_a = key in config_a
            in_b = key in config_b

            if in_a and not in_b:
                diffs.append(
                    f"  - {current_path}: removed (was {config_a[key]!r})"
                )
            elif in_b and not in_a:
                diffs.append(
                    f"  + {current_path}: added ({config_b[key]!r})"
                )
            elif config_a[key] != config_b[key]:
                val_a = config_a[key]
                val_b = config_b[key]
                if isinstance(val_a, dict) and isinstance(val_b, dict):
                    diffs.extend(cls.diff(val_a, val_b, current_path))
                elif isinstance(val_a, list) and isinstance(val_b, list):
                    diffs.extend(cls._diff_lists(val_a, val_b, current_path))
                else:
                    diffs.append(
                        f"  ~ {current_path}: {val_a!r} -> {val_b!r}"
                    )

        return diffs

    @classmethod
    def _diff_lists(
        cls, list_a: list, list_b: list, path: str
    ) -> list[str]:
        """Compare two lists and return differences."""
        diffs: list[str] = []
        max_len = max(len(list_a), len(list_b))

        for i in range(max_len):
            item_path = f"{path}[{i}]"
            if i >= len(list_a):
                diffs.append(f"  + {item_path}: added ({list_b[i]!r})")
            elif i >= len(list_b):
                diffs.append(f"  - {item_path}: removed (was {list_a[i]!r})")
            elif list_a[i] != list_b[i]:
                if isinstance(list_a[i], dict) and isinstance(list_b[i], dict):
                    diffs.extend(cls.diff(list_a[i], list_b[i], item_path))
                else:
                    diffs.append(
                        f"  ~ {item_path}: {list_a[i]!r} -> {list_b[i]!r}"
                    )

        return diffs

    @classmethod
    def format_diff(
        cls,
        config_a: dict[str, Any],
        config_b: dict[str, Any],
        label_a: str = "A",
        label_b: str = "B",
    ) -> str:
        """Format a diff report between two configurations.

        Args:
            config_a: First configuration.
            config_b: Second configuration.
            label_a: Label for the first configuration.
            label_b: Label for the second configuration.

        Returns:
            Formatted diff report string.
        """
        diffs = cls.diff(config_a, config_b)
        if not diffs:
            return f"Configurations '{label_a}' and '{label_b}' are identical."

        lines = [
            f"Differences between '{label_a}' and '{label_b}':",
            f"  (- = only in {label_a}, + = only in {label_b}, ~ = changed)",
        ]
        lines.extend(diffs)
        lines.append(f"  Total: {len(diffs)} difference(s)")
        return "\n".join(lines)


def generate_sample_config() -> dict[str, Any]:
    """Generate a complete sample configuration dictionary.

    Uses the existing provider infrastructure to create a realistic
    configuration that passes validation.

    Returns:
        A dictionary containing a fully populated sample configuration.
    """
    from src.providers import get_builtin_providers

    providers = get_builtin_providers()
    config = GatewayConfig(
        default_provider="openai",
        providers=providers,
        log_level="info",
        timeout=30,
        max_retries=3,
    )
    return config.to_dict()


def generate_minimal_config() -> dict[str, Any]:
    """Generate a minimal valid configuration dictionary.

    Returns:
        A minimal configuration that passes validation.
    """
    config = GatewayConfig(
        default_provider="openai",
        providers={
            "openai": ProviderConfig(
                name="openai",
                display_name="OpenAI",
                api_base="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
                auth_type=AuthType.BEARER_TOKEN,
                default_model="gpt-4o",
                models={
                    "gpt-4o": ModelConfig(
                        name="gpt-4o",
                        max_tokens=16384,
                    ),
                },
            ),
        },
        log_level="info",
        timeout=30,
        max_retries=3,
    )
    return config.to_dict()


def create_test_config_with_overrides(
    base: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a test configuration by merging overrides onto a base.

    Args:
        base: Base configuration dict. Uses minimal config if None.
        overrides: Dict of values to merge on top.

    Returns:
        Merged configuration dictionary.
    """
    if base is None:
        base = generate_minimal_config()

    result = copy.deepcopy(base)

    if overrides:
        _deep_merge(result, overrides)

    return result


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override dict into base dict in-place."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_builtin_test_cases() -> list[ConfigTestCase]:
    """Return a set of built-in test cases for configuration validation.

    These test cases exercise the validator against known-good and
    known-bad configurations to verify validation logic.

    Returns:
        List of ConfigTestCase instances covering common scenarios.
    """
    minimal = generate_minimal_config()

    return [
        # --- Valid configurations ---
        ConfigTestCase(
            name="valid-full-config",
            description="The default sample configuration should be valid.",
            config_data=generate_sample_config(),
            expect_valid=True,
        ),
        ConfigTestCase(
            name="valid-minimal-config",
            description="A minimal configuration should be valid.",
            config_data=minimal,
            expect_valid=True,
        ),
        ConfigTestCase(
            name="valid-empty-config",
            description="An empty config is technically valid (just warnings).",
            config_data={},
            expect_valid=True,
            expect_warnings_at=["providers"],
        ),
        # --- Invalid: bad top-level values ---
        ConfigTestCase(
            name="invalid-negative-timeout",
            description="Negative timeout should fail validation.",
            config_data=create_test_config_with_overrides(
                overrides={"timeout": -1}
            ),
            expect_valid=False,
            expect_errors_at=["timeout"],
        ),
        ConfigTestCase(
            name="invalid-negative-retries",
            description="Negative max_retries should fail validation.",
            config_data=create_test_config_with_overrides(
                overrides={"max_retries": -5}
            ),
            expect_valid=False,
            expect_errors_at=["max_retries"],
        ),
        ConfigTestCase(
            name="invalid-log-level",
            description="An invalid log level should fail validation.",
            config_data=create_test_config_with_overrides(
                overrides={"log_level": "verbose"}
            ),
            expect_valid=False,
            expect_errors_at=["log_level"],
        ),
        # --- Invalid: bad provider values ---
        ConfigTestCase(
            name="invalid-provider-no-api-base",
            description="A provider without api_base should fail.",
            config_data={
                "default_provider": "broken",
                "providers": {
                    "broken": {
                        "name": "broken",
                        "api_base": "",
                    }
                },
            },
            expect_valid=False,
            expect_errors_at=["providers.broken.api_base"],
        ),
        ConfigTestCase(
            name="invalid-provider-bad-url",
            description="A provider with an invalid URL should fail.",
            config_data={
                "default_provider": "bad",
                "providers": {
                    "bad": {
                        "name": "bad",
                        "api_base": "not-a-url",
                    }
                },
            },
            expect_valid=False,
            expect_errors_at=["providers.bad.api_base"],
        ),
        ConfigTestCase(
            name="invalid-provider-bad-model-tokens",
            description="A model with 0 max_tokens should fail.",
            config_data={
                "default_provider": "test",
                "providers": {
                    "test": {
                        "name": "test",
                        "api_base": "https://api.example.com",
                        "models": {
                            "badmodel": {
                                "name": "badmodel",
                                "max_tokens": 0,
                            }
                        },
                    }
                },
            },
            expect_valid=False,
            expect_errors_at=["providers.test.models.badmodel.max_tokens"],
        ),
        # --- Invalid: cross-reference errors ---
        ConfigTestCase(
            name="invalid-default-provider-missing",
            description="Referencing a non-existent default provider should fail.",
            config_data=create_test_config_with_overrides(
                overrides={"default_provider": "nonexistent"}
            ),
            expect_valid=False,
            expect_errors_at=["default_provider"],
        ),
        ConfigTestCase(
            name="invalid-default-model-missing",
            description=(
                "Provider default_model not in models list should fail."
            ),
            config_data={
                "default_provider": "test",
                "providers": {
                    "test": {
                        "name": "test",
                        "api_base": "https://api.example.com",
                        "default_model": "nonexistent",
                        "models": {
                            "real-model": {
                                "name": "real-model",
                                "max_tokens": 4096,
                            }
                        },
                    }
                },
            },
            expect_valid=False,
            expect_errors_at=["providers.test.default_model"],
        ),
        # --- Invalid: all providers disabled ---
        ConfigTestCase(
            name="invalid-all-providers-disabled",
            description="All providers disabled should fail.",
            config_data={
                "default_provider": "test",
                "providers": {
                    "test": {
                        "name": "test",
                        "api_base": "https://api.example.com",
                        "enabled": False,
                    }
                },
            },
            expect_valid=False,
            expect_errors_at=["providers"],
        ),
        # --- Warnings ---
        ConfigTestCase(
            name="warning-high-timeout",
            description="Very high timeout should warn.",
            config_data=create_test_config_with_overrides(
                overrides={"timeout": 500}
            ),
            expect_valid=True,
            expect_warnings_at=["timeout"],
        ),
        ConfigTestCase(
            name="warning-high-retries",
            description="High retry count should warn.",
            config_data=create_test_config_with_overrides(
                overrides={"max_retries": 15}
            ),
            expect_valid=True,
            expect_warnings_at=["max_retries"],
        ),
        ConfigTestCase(
            name="warning-disabled-default-provider",
            description="Disabled default provider should warn.",
            config_data={
                "default_provider": "off",
                "providers": {
                    "off": {
                        "name": "off",
                        "api_base": "https://api.example.com",
                        "enabled": False,
                    },
                    "on": {
                        "name": "on",
                        "api_base": "https://api2.example.com",
                        "enabled": True,
                    },
                },
            },
            expect_valid=True,
            expect_warnings_at=["default_provider"],
        ),
    ]
