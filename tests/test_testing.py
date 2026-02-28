"""Tests for configuration testing utilities."""

import json
from pathlib import Path

import pytest
import yaml

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


# ---------------------------------------------------------------------------
# Config generation tests
# ---------------------------------------------------------------------------


class TestGenerateSampleConfig:
    """Tests for sample config generation."""

    def test_generates_dict(self):
        """Test that sample config is a dictionary."""
        config = generate_sample_config()
        assert isinstance(config, dict)

    def test_has_providers(self):
        """Test that sample config includes providers."""
        config = generate_sample_config()
        assert "providers" in config
        assert len(config["providers"]) > 0

    def test_has_default_provider(self):
        """Test that sample config has a default provider."""
        config = generate_sample_config()
        assert config.get("default_provider")

    def test_passes_validation(self):
        """Test that the generated sample passes validation."""
        from src.validation.validator import ConfigValidator

        config = generate_sample_config()
        result = ConfigValidator.validate_dict(config)
        assert result.is_valid


class TestGenerateMinimalConfig:
    """Tests for minimal config generation."""

    def test_generates_dict(self):
        """Test that minimal config is a dictionary."""
        config = generate_minimal_config()
        assert isinstance(config, dict)

    def test_has_one_provider(self):
        """Test that minimal config has exactly one provider."""
        config = generate_minimal_config()
        assert len(config.get("providers", {})) == 1

    def test_passes_validation(self):
        """Test that minimal config passes validation."""
        from src.validation.validator import ConfigValidator

        config = generate_minimal_config()
        result = ConfigValidator.validate_dict(config)
        assert result.is_valid


class TestCreateTestConfigWithOverrides:
    """Tests for override-based config creation."""

    def test_returns_base_when_no_overrides(self):
        """Test that base config returned unchanged without overrides."""
        config = create_test_config_with_overrides()
        assert isinstance(config, dict)
        assert "providers" in config

    def test_applies_simple_override(self):
        """Test simple top-level override."""
        config = create_test_config_with_overrides(
            overrides={"timeout": 999}
        )
        assert config["timeout"] == 999

    def test_applies_nested_override(self):
        """Test nested dictionary override."""
        base = {"gateway": {"name": "test", "port": 8080}}
        config = create_test_config_with_overrides(
            base=base, overrides={"gateway": {"port": 9090}}
        )
        assert config["gateway"]["port"] == 9090
        assert config["gateway"]["name"] == "test"  # preserved

    def test_custom_base(self):
        """Test providing a custom base config."""
        base = {"custom_key": "custom_value"}
        config = create_test_config_with_overrides(
            base=base, overrides={"extra": True}
        )
        assert config["custom_key"] == "custom_value"
        assert config["extra"] is True

    def test_does_not_mutate_base(self):
        """Test that the original base dict is not mutated."""
        base = {"timeout": 30}
        create_test_config_with_overrides(
            base=base, overrides={"timeout": 999}
        )
        assert base["timeout"] == 30  # unchanged


# ---------------------------------------------------------------------------
# ConfigTestCase / ConfigTestResult / ConfigTestSuiteResult
# ---------------------------------------------------------------------------


class TestConfigTestCase:
    """Tests for ConfigTestCase dataclass."""

    def test_defaults(self):
        """Test default values."""
        tc = ConfigTestCase(name="test")
        assert tc.name == "test"
        assert tc.expect_valid is True
        assert tc.config_data == {}
        assert tc.expect_errors_at == []
        assert tc.expect_warnings_at == []


class TestConfigTestSuiteResult:
    """Tests for ConfigTestSuiteResult."""

    def test_empty_suite(self):
        """Test empty suite."""
        suite = ConfigTestSuiteResult()
        assert suite.total == 0
        assert suite.passed == 0
        assert suite.failed == 0
        assert suite.all_passed

    def test_all_passed(self):
        """Test suite with all passing tests."""
        from src.validation.validator import ValidationResult

        suite = ConfigTestSuiteResult(
            results=[
                ConfigTestResult(
                    test_case=ConfigTestCase(name="t1"),
                    passed=True,
                    validation_result=ValidationResult(),
                ),
                ConfigTestResult(
                    test_case=ConfigTestCase(name="t2"),
                    passed=True,
                    validation_result=ValidationResult(),
                ),
            ]
        )
        assert suite.total == 2
        assert suite.passed == 2
        assert suite.failed == 0
        assert suite.all_passed

    def test_mixed_results(self):
        """Test suite with mixed pass/fail."""
        from src.validation.validator import ValidationResult

        suite = ConfigTestSuiteResult(
            results=[
                ConfigTestResult(
                    test_case=ConfigTestCase(name="pass"),
                    passed=True,
                    validation_result=ValidationResult(),
                ),
                ConfigTestResult(
                    test_case=ConfigTestCase(name="fail"),
                    passed=False,
                    validation_result=ValidationResult(),
                    failure_reason="Something went wrong",
                ),
            ]
        )
        assert suite.total == 2
        assert suite.passed == 1
        assert suite.failed == 1
        assert not suite.all_passed

    def test_format_report(self):
        """Test report formatting."""
        from src.validation.validator import ValidationResult

        suite = ConfigTestSuiteResult(
            results=[
                ConfigTestResult(
                    test_case=ConfigTestCase(name="passing-test"),
                    passed=True,
                    validation_result=ValidationResult(),
                ),
                ConfigTestResult(
                    test_case=ConfigTestCase(name="failing-test"),
                    passed=False,
                    validation_result=ValidationResult(),
                    failure_reason="Bad config",
                ),
            ]
        )
        report = suite.format_report()
        assert "1/2 passed" in report
        assert "[PASS] passing-test" in report
        assert "[FAIL] failing-test" in report
        assert "Bad config" in report
        assert "1 test(s) failed." in report


# ---------------------------------------------------------------------------
# ConfigTestRunner tests
# ---------------------------------------------------------------------------


class TestConfigTestRunner:
    """Tests for the configuration test runner."""

    def test_run_valid_test_passes(self):
        """Test that a valid config test case passes."""
        tc = ConfigTestCase(
            name="valid-test",
            config_data=generate_minimal_config(),
            expect_valid=True,
        )
        result = ConfigTestRunner.run_test(tc)
        assert result.passed
        assert result.failure_reason == ""

    def test_run_expected_invalid_passes(self):
        """Test that expected-invalid test case passes when errors found."""
        tc = ConfigTestCase(
            name="invalid-test",
            config_data={"timeout": -1},
            expect_valid=False,
            expect_errors_at=["timeout"],
        )
        result = ConfigTestRunner.run_test(tc)
        assert result.passed

    def test_run_valid_but_has_errors_fails(self):
        """Test that expected-valid test fails when errors present."""
        tc = ConfigTestCase(
            name="should-fail",
            config_data={"timeout": -1},
            expect_valid=True,
        )
        result = ConfigTestRunner.run_test(tc)
        assert not result.passed
        assert "Expected valid" in result.failure_reason

    def test_run_expected_invalid_but_valid_fails(self):
        """Test that expected-invalid test fails when config is actually valid."""
        tc = ConfigTestCase(
            name="should-fail",
            config_data=generate_minimal_config(),
            expect_valid=False,
        )
        result = ConfigTestRunner.run_test(tc)
        assert not result.passed
        assert "Expected invalid" in result.failure_reason

    def test_expected_error_path_missing_fails(self):
        """Test failure when expected error path not found."""
        tc = ConfigTestCase(
            name="missing-error",
            config_data=generate_minimal_config(),
            expect_valid=True,
            expect_errors_at=["nonexistent.path"],
        )
        result = ConfigTestRunner.run_test(tc)
        assert not result.passed
        assert "nonexistent.path" in result.failure_reason

    def test_expected_warning_path_missing_fails(self):
        """Test failure when expected warning path not found."""
        tc = ConfigTestCase(
            name="missing-warning",
            config_data=generate_minimal_config(),
            expect_valid=True,
            expect_warnings_at=["nonexistent.path"],
        )
        result = ConfigTestRunner.run_test(tc)
        assert not result.passed
        assert "nonexistent.path" in result.failure_reason

    def test_run_suite(self):
        """Test running a suite of test cases."""
        test_cases = [
            ConfigTestCase(
                name="good",
                config_data=generate_minimal_config(),
                expect_valid=True,
            ),
            ConfigTestCase(
                name="bad",
                config_data={"timeout": -1},
                expect_valid=False,
                expect_errors_at=["timeout"],
            ),
        ]
        suite = ConfigTestRunner.run_suite(test_cases)
        assert suite.total == 2
        assert suite.passed == 2
        assert suite.all_passed

    def test_run_file_tests_valid(self, tmp_path):
        """Test running file-based validation on a valid YAML config."""
        config = generate_minimal_config()
        path = tmp_path / "gateway.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        result = ConfigTestRunner.run_file_tests(path)
        assert result.passed
        assert result.test_case.name == "File: gateway.yaml"

    def test_run_file_tests_invalid_file(self, tmp_path):
        """Test running file-based validation on a bad YAML file."""
        path = tmp_path / "bad.yaml"
        path.write_text(":::invalid yaml:::")

        result = ConfigTestRunner.run_file_tests(path)
        assert not result.passed
        assert "Failed to load" in result.failure_reason

    def test_run_file_tests_nonexistent(self, tmp_path):
        """Test running file-based validation on nonexistent file."""
        path = tmp_path / "nope.yaml"
        result = ConfigTestRunner.run_file_tests(path)
        assert not result.passed

    def test_run_file_tests_json(self, tmp_path):
        """Test running file-based validation on a JSON config."""
        config = generate_minimal_config()
        path = tmp_path / "gateway.json"
        with open(path, "w") as f:
            json.dump(config, f)

        result = ConfigTestRunner.run_file_tests(path)
        assert result.passed


# ---------------------------------------------------------------------------
# Built-in test cases
# ---------------------------------------------------------------------------


class TestBuiltinTestCases:
    """Tests for the built-in test suite."""

    def test_builtin_cases_exist(self):
        """Test that built-in test cases are returned."""
        cases = get_builtin_test_cases()
        assert len(cases) > 0

    def test_builtin_cases_all_have_names(self):
        """Test that all built-in cases have names."""
        for tc in get_builtin_test_cases():
            assert tc.name
            assert isinstance(tc.name, str)

    def test_builtin_suite_passes(self):
        """Test that all built-in test cases pass the test runner."""
        cases = get_builtin_test_cases()
        suite = ConfigTestRunner.run_suite(cases)
        failed_names = [
            r.test_case.name for r in suite.results if not r.passed
        ]
        assert suite.all_passed, (
            f"Failed built-in tests: {failed_names}"
        )


# ---------------------------------------------------------------------------
# ConfigDiff tests
# ---------------------------------------------------------------------------


class TestConfigDiff:
    """Tests for configuration diffing."""

    def test_identical_configs(self):
        """Test that identical configs produce no diffs."""
        config = {"key": "value", "nested": {"a": 1}}
        diffs = ConfigDiff.diff(config, config)
        assert diffs == []

    def test_added_key(self):
        """Test detecting an added key."""
        a = {"key1": "value1"}
        b = {"key1": "value1", "key2": "value2"}
        diffs = ConfigDiff.diff(a, b)
        assert len(diffs) == 1
        assert "+" in diffs[0]
        assert "key2" in diffs[0]

    def test_removed_key(self):
        """Test detecting a removed key."""
        a = {"key1": "value1", "key2": "value2"}
        b = {"key1": "value1"}
        diffs = ConfigDiff.diff(a, b)
        assert len(diffs) == 1
        assert "-" in diffs[0]
        assert "key2" in diffs[0]

    def test_changed_value(self):
        """Test detecting a changed value."""
        a = {"key": "old"}
        b = {"key": "new"}
        diffs = ConfigDiff.diff(a, b)
        assert len(diffs) == 1
        assert "~" in diffs[0]
        assert "'old'" in diffs[0]
        assert "'new'" in diffs[0]

    def test_nested_diff(self):
        """Test detecting nested differences."""
        a = {"outer": {"inner": 1}}
        b = {"outer": {"inner": 2}}
        diffs = ConfigDiff.diff(a, b)
        assert len(diffs) == 1
        assert "outer.inner" in diffs[0]

    def test_list_diff_added(self):
        """Test detecting items added to a list."""
        a = {"items": [1, 2]}
        b = {"items": [1, 2, 3]}
        diffs = ConfigDiff.diff(a, b)
        assert len(diffs) == 1
        assert "+" in diffs[0]

    def test_list_diff_removed(self):
        """Test detecting items removed from a list."""
        a = {"items": [1, 2, 3]}
        b = {"items": [1, 2]}
        diffs = ConfigDiff.diff(a, b)
        assert len(diffs) == 1
        assert "-" in diffs[0]

    def test_list_diff_changed(self):
        """Test detecting changed items in a list."""
        a = {"items": ["a", "b"]}
        b = {"items": ["a", "c"]}
        diffs = ConfigDiff.diff(a, b)
        assert len(diffs) == 1
        assert "~" in diffs[0]

    def test_list_of_dicts_diff(self):
        """Test detecting changes in lists of dictionaries."""
        a = {"items": [{"name": "a", "value": 1}]}
        b = {"items": [{"name": "a", "value": 2}]}
        diffs = ConfigDiff.diff(a, b)
        assert len(diffs) == 1
        assert "value" in diffs[0]

    def test_format_diff_identical(self):
        """Test format_diff for identical configs."""
        config = {"key": "value"}
        report = ConfigDiff.format_diff(config, config)
        assert "identical" in report

    def test_format_diff_with_differences(self):
        """Test format_diff shows differences."""
        a = {"key": "old", "only_a": True}
        b = {"key": "new", "only_b": True}
        report = ConfigDiff.format_diff(a, b, label_a="Dev", label_b="Prod")
        assert "Dev" in report
        assert "Prod" in report
        assert "difference(s)" in report

    def test_format_diff_custom_labels(self):
        """Test format_diff with custom labels."""
        a = {"x": 1}
        b = {"x": 2}
        report = ConfigDiff.format_diff(a, b, label_a="File A", label_b="File B")
        assert "File A" in report
        assert "File B" in report
