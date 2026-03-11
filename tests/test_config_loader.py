"""Tests for the extended configuration loader module."""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import ConfigError, ConfigValidationError
from src.config.loader import (
    ConfigWatcher,
    deep_merge,
    discover_config_files,
    export_config,
    import_config,
    interpolate_env_vars,
    load_and_merge,
    load_with_defaults,
    load_with_interpolation,
)
from src.models import GatewayConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def minimal_yaml(tmp_dir):
    """A minimal valid YAML config file."""
    data = {
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
                "default_model": "gpt-4o",
                "models": {
                    "gpt-4o": {
                        "name": "gpt-4o",
                        "max_tokens": 16384,
                    }
                },
            }
        },
    }
    path = tmp_dir / "gateway.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


@pytest.fixture
def minimal_json(tmp_dir):
    """A minimal valid JSON config file."""
    data = {
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
                "default_model": "gpt-4o",
                "models": {
                    "gpt-4o": {
                        "name": "gpt-4o",
                        "max_tokens": 16384,
                    }
                },
            }
        },
    }
    path = tmp_dir / "gateway.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# deep_merge tests
# ---------------------------------------------------------------------------


class TestDeepMerge:
    """Tests for the deep_merge function."""

    def test_merge_empty_dicts(self):
        """Test merging two empty dictionaries."""
        result = deep_merge({}, {})
        assert result == {}

    def test_merge_disjoint_keys(self):
        """Test merging dicts with no common keys."""
        a = {"key1": "value1"}
        b = {"key2": "value2"}
        result = deep_merge(a, b)
        assert result == {"key1": "value1", "key2": "value2"}

    def test_overlay_replaces_scalar(self):
        """Test that overlay value replaces base scalar."""
        a = {"key": "old"}
        b = {"key": "new"}
        result = deep_merge(a, b)
        assert result["key"] == "new"

    def test_nested_dicts_merged_recursively(self):
        """Test that nested dicts are merged recursively."""
        a = {"outer": {"a": 1, "b": 2}}
        b = {"outer": {"b": 99, "c": 3}}
        result = deep_merge(a, b)
        assert result["outer"]["a"] == 1    # preserved from base
        assert result["outer"]["b"] == 99   # overridden
        assert result["outer"]["c"] == 3    # added from overlay

    def test_does_not_mutate_base(self):
        """Test that the base dict is not mutated."""
        a = {"key": "original"}
        b = {"key": "override"}
        deep_merge(a, b)
        assert a["key"] == "original"

    def test_does_not_mutate_overlay(self):
        """Test that the overlay dict is not mutated."""
        a = {"key": "original"}
        b = {"key": "override", "nested": {"x": 1}}
        deep_merge(a, b)
        assert b["key"] == "override"  # overlay unchanged

    def test_list_values_replaced_not_merged(self):
        """Test that list values are replaced, not merged."""
        a = {"items": [1, 2, 3]}
        b = {"items": [4, 5]}
        result = deep_merge(a, b)
        assert result["items"] == [4, 5]

    def test_deep_nesting(self):
        """Test deep nesting of three levels."""
        a = {"l1": {"l2": {"l3": "original"}}}
        b = {"l1": {"l2": {"l3": "override", "new_key": "added"}}}
        result = deep_merge(a, b)
        assert result["l1"]["l2"]["l3"] == "override"
        assert result["l1"]["l2"]["new_key"] == "added"

    def test_overlay_with_none_value(self):
        """Test merging with None value in overlay."""
        a = {"key": "value"}
        b = {"key": None}
        result = deep_merge(a, b)
        assert result["key"] is None

    def test_base_nested_dict_overlay_scalar(self):
        """Test that overlay scalar replaces base dict entirely."""
        a = {"key": {"nested": "value"}}
        b = {"key": "scalar_replacement"}
        result = deep_merge(a, b)
        assert result["key"] == "scalar_replacement"


# ---------------------------------------------------------------------------
# load_and_merge tests
# ---------------------------------------------------------------------------


class TestLoadAndMerge:
    """Tests for load_and_merge function."""

    def test_merge_single_file(self, minimal_yaml):
        """Test loading a single file."""
        config = load_and_merge([minimal_yaml], validate=False)
        assert isinstance(config, GatewayConfig)
        assert config.default_provider == "openai"

    def test_merge_two_files(self, tmp_dir):
        """Test merging two config files."""
        base_data = {
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
                }
            },
        }
        overlay_data = {"timeout": 60, "log_level": "debug"}

        base_path = tmp_dir / "base.yaml"
        overlay_path = tmp_dir / "overlay.yaml"
        with open(base_path, "w") as f:
            yaml.dump(base_data, f)
        with open(overlay_path, "w") as f:
            yaml.dump(overlay_data, f)

        config = load_and_merge([base_path, overlay_path], validate=False)
        assert config.timeout == 60        # from overlay
        assert config.log_level == "debug"  # from overlay
        assert "openai" in config.providers  # from base

    def test_later_file_takes_precedence(self, tmp_dir):
        """Test that later files override earlier ones."""
        file1 = tmp_dir / "first.yaml"
        file2 = tmp_dir / "second.yaml"
        file1.write_text("timeout: 10\n")
        file2.write_text("timeout: 20\n")

        config = load_and_merge([file1, file2], validate=False)
        assert config.timeout == 20

    def test_no_files_raises_error(self):
        """Test that passing no files raises ConfigError."""
        with pytest.raises(ConfigError, match="No configuration files"):
            load_and_merge([])

    def test_missing_file_raises_error(self, tmp_dir):
        """Test that a missing file raises ConfigError."""
        with pytest.raises(ConfigError):
            load_and_merge([tmp_dir / "nonexistent.yaml"])

    def test_validation_enabled_raises_on_bad_config(self, tmp_dir):
        """Test that validation errors are raised when validate=True."""
        path = tmp_dir / "bad.yaml"
        path.write_text("timeout: -1\nlog_level: invalid\n")
        with pytest.raises(ConfigValidationError):
            load_and_merge([path], validate=True)

    def test_merge_yaml_and_json(self, tmp_dir, minimal_yaml):
        """Test merging YAML and JSON files together."""
        overlay_data = {"timeout": 90}
        json_path = tmp_dir / "overlay.json"
        with open(json_path, "w") as f:
            json.dump(overlay_data, f)

        config = load_and_merge([minimal_yaml, json_path], validate=False)
        assert config.timeout == 90


# ---------------------------------------------------------------------------
# interpolate_env_vars tests
# ---------------------------------------------------------------------------


class TestInterpolateEnvVars:
    """Tests for interpolate_env_vars function."""

    def test_simple_substitution(self):
        """Test ${VAR} substitution."""
        data = {"key": "${MY_VAR}"}
        result = interpolate_env_vars(data, env={"MY_VAR": "hello"})
        assert result["key"] == "hello"

    def test_default_value_used_when_var_missing(self):
        """Test ${VAR:-default} falls back to default."""
        data = {"key": "${MISSING_VAR:-fallback}"}
        result = interpolate_env_vars(data, env={})
        assert result["key"] == "fallback"

    def test_env_value_takes_precedence_over_default(self):
        """Test that env value takes precedence over default."""
        data = {"key": "${MY_VAR:-fallback}"}
        result = interpolate_env_vars(data, env={"MY_VAR": "env_value"})
        assert result["key"] == "env_value"

    def test_unresolved_var_left_as_is(self):
        """Test that unresolvable ${VAR} without default is left unchanged."""
        data = {"key": "${MISSING_NO_DEFAULT}"}
        result = interpolate_env_vars(data, env={})
        assert result["key"] == "${MISSING_NO_DEFAULT}"

    def test_nested_dict_interpolated(self):
        """Test that nested dicts are also interpolated."""
        data = {"outer": {"inner": "${VAR}"}}
        result = interpolate_env_vars(data, env={"VAR": "value"})
        assert result["outer"]["inner"] == "value"

    def test_list_values_interpolated(self):
        """Test that list items are also interpolated."""
        data = {"items": ["${VAR1}", "${VAR2}"]}
        result = interpolate_env_vars(data, env={"VAR1": "a", "VAR2": "b"})
        assert result["items"] == ["a", "b"]

    def test_non_string_values_unchanged(self):
        """Test that non-string values are passed through unchanged."""
        data = {"count": 42, "flag": True}
        result = interpolate_env_vars(data, env={})
        assert result["count"] == 42
        assert result["flag"] is True

    def test_uses_os_environ_by_default(self, monkeypatch):
        """Test that os.environ is used when env is not provided."""
        monkeypatch.setenv("TEST_INTERP_VAR", "from_environ")
        data = {"key": "${TEST_INTERP_VAR}"}
        result = interpolate_env_vars(data)  # no env kwarg
        assert result["key"] == "from_environ"

    def test_empty_default(self):
        """Test ${VAR:-} with empty default."""
        data = {"key": "${MISSING:-}"}
        result = interpolate_env_vars(data, env={})
        assert result["key"] == ""

    def test_does_not_mutate_input(self):
        """Test that the input dict is not mutated."""
        data = {"key": "${VAR}"}
        interpolate_env_vars(data, env={"VAR": "value"})
        assert data["key"] == "${VAR}"  # original unchanged

    def test_multiple_vars_in_one_string(self):
        """Test multiple ${VAR} in a single string value."""
        data = {"key": "${FIRST}-${SECOND}"}
        result = interpolate_env_vars(
            data, env={"FIRST": "hello", "SECOND": "world"}
        )
        assert result["key"] == "hello-world"


# ---------------------------------------------------------------------------
# load_with_interpolation tests
# ---------------------------------------------------------------------------


class TestLoadWithInterpolation:
    """Tests for load_with_interpolation function."""

    def test_loads_and_interpolates(self, tmp_dir):
        """Test that env vars in config file values are resolved."""
        data = {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "name": "openai",
                    "api_base": "${MY_API_BASE:-https://api.openai.com/v1}",
                    "api_key_env_var": "OPENAI_API_KEY",
                    "auth_type": "bearer_token",
                }
            },
        }
        path = tmp_dir / "interpolated.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)

        config = load_with_interpolation(
            path,
            validate=False,
            env={"MY_API_BASE": "https://custom.example.com"},
        )
        assert config.providers["openai"].api_base == "https://custom.example.com"

    def test_default_interpolation_used(self, tmp_dir):
        """Test that default value is used when env var is not set."""
        # Use a string field (log_level) to avoid int coercion issues
        data = {
            "log_level": "${CUSTOM_LOG_LEVEL:-debug}",
        }
        path = tmp_dir / "defaults.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)

        # YAML loads "${CUSTOM_LOG_LEVEL:-debug}" as a string.
        # Interpolation resolves to "debug" since the env var is not set.
        config = load_with_interpolation(path, validate=False, env={})
        assert config.log_level == "debug"

    def test_missing_file_raises_error(self, tmp_dir):
        """Test that a missing file raises ConfigError."""
        with pytest.raises(ConfigError):
            load_with_interpolation(
                tmp_dir / "nonexistent.yaml",
                validate=False,
            )


# ---------------------------------------------------------------------------
# load_with_defaults tests
# ---------------------------------------------------------------------------


class TestLoadWithDefaults:
    """Tests for load_with_defaults function."""

    def test_loads_valid_file(self, minimal_yaml):
        """Test loading a valid file returns the config."""
        config = load_with_defaults(path=minimal_yaml, validate=False)
        assert isinstance(config, GatewayConfig)
        assert config.default_provider == "openai"

    def test_nonexistent_file_returns_defaults(self, tmp_dir):
        """Test that nonexistent file returns default config."""
        config = load_with_defaults(
            path=tmp_dir / "does_not_exist.yaml",
            validate=False,
        )
        assert isinstance(config, GatewayConfig)
        # Default config should have some providers
        assert len(config.providers) > 0

    def test_invalid_config_returns_defaults(self, tmp_dir):
        """Test that invalid config (validation fails) returns default."""
        path = tmp_dir / "invalid.yaml"
        path.write_text("timeout: -1\nlog_level: bad_level\n")
        config = load_with_defaults(path=path, validate=True)
        assert isinstance(config, GatewayConfig)
        # Should have returned defaults
        assert config.timeout > 0

    def test_no_path_uses_discovery(self, tmp_dir, monkeypatch):
        """Test that no path falls back to auto-discovery."""
        monkeypatch.delenv("GATEWAY_CONFIG", raising=False)
        # Just verify it doesn't crash and returns a GatewayConfig
        config = load_with_defaults(validate=False)
        assert isinstance(config, GatewayConfig)


# ---------------------------------------------------------------------------
# discover_config_files tests
# ---------------------------------------------------------------------------


class TestDiscoverConfigFiles:
    """Tests for discover_config_files function."""

    def test_finds_yaml_file(self, tmp_dir):
        """Test discovering a gateway.yaml file."""
        (tmp_dir / "gateway.yaml").write_text("default_provider: test\n")
        found = discover_config_files(search_dirs=[tmp_dir])
        assert any("gateway.yaml" in str(p) for p in found)

    def test_finds_yml_file(self, tmp_dir):
        """Test discovering a gateway.yml file."""
        (tmp_dir / "gateway.yml").write_text("default_provider: test\n")
        found = discover_config_files(search_dirs=[tmp_dir])
        assert any("gateway.yml" in str(p) for p in found)

    def test_finds_json_file(self, tmp_dir):
        """Test discovering a gateway.json file."""
        (tmp_dir / "gateway.json").write_text('{"default_provider": "test"}')
        found = discover_config_files(search_dirs=[tmp_dir])
        assert any("gateway.json" in str(p) for p in found)

    def test_empty_dir_returns_empty_list(self, tmp_dir):
        """Test that an empty directory returns an empty list."""
        found = discover_config_files(search_dirs=[tmp_dir])
        assert found == []

    def test_custom_filenames(self, tmp_dir):
        """Test using custom filenames."""
        (tmp_dir / "custom.yaml").write_text("key: value\n")
        found = discover_config_files(
            search_dirs=[tmp_dir], filenames=["custom.yaml"]
        )
        assert any("custom.yaml" in str(p) for p in found)

    def test_nonexistent_dir_ignored(self, tmp_dir):
        """Test that nonexistent directories are silently ignored."""
        found = discover_config_files(
            search_dirs=[tmp_dir / "does_not_exist"]
        )
        assert found == []

    def test_multiple_dirs_searched(self, tmp_dir):
        """Test that multiple directories are all searched."""
        dir_a = tmp_dir / "a"
        dir_b = tmp_dir / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "gateway.yaml").write_text("key: a\n")
        (dir_b / "gateway.yaml").write_text("key: b\n")
        found = discover_config_files(search_dirs=[dir_a, dir_b])
        assert len(found) == 2

    def test_returns_path_objects(self, tmp_dir):
        """Test that discovered files are Path objects."""
        (tmp_dir / "gateway.yaml").write_text("key: value\n")
        found = discover_config_files(search_dirs=[tmp_dir])
        assert all(isinstance(p, Path) for p in found)


# ---------------------------------------------------------------------------
# ConfigWatcher tests
# ---------------------------------------------------------------------------


class TestConfigWatcher:
    """Tests for the ConfigWatcher class."""

    def test_watcher_starts_and_stops(self, minimal_yaml):
        """Test that watcher starts and stops cleanly."""
        changes = []
        watcher = ConfigWatcher(
            path=minimal_yaml,
            on_change=lambda c: changes.append(c),
            interval=0.1,
        )
        watcher.start()
        assert watcher.is_running
        watcher.stop()
        assert not watcher.is_running

    def test_watcher_not_started_by_default(self, minimal_yaml):
        """Test that watcher is not running before start() is called."""
        watcher = ConfigWatcher(
            path=minimal_yaml,
            on_change=lambda c: None,
        )
        assert not watcher.is_running

    def test_start_twice_is_safe(self, minimal_yaml):
        """Test that calling start() twice does not create duplicate threads."""
        watcher = ConfigWatcher(
            path=minimal_yaml,
            on_change=lambda c: None,
            interval=0.1,
        )
        watcher.start()
        thread1 = watcher._thread
        watcher.start()  # second start
        thread2 = watcher._thread
        assert thread1 is thread2  # same thread
        watcher.stop()

    def test_check_now_returns_false_for_unchanged(self, minimal_yaml):
        """Test that check_now returns False when file unchanged."""
        watcher = ConfigWatcher(
            path=minimal_yaml,
            on_change=lambda c: None,
            interval=60,
        )
        # Record initial mtime
        watcher._last_mtime = watcher._get_mtime()
        changed = watcher.check_now()
        assert changed is False

    def test_check_now_detects_change(self, minimal_yaml):
        """Test that check_now returns True when file changes."""
        changes = []
        watcher = ConfigWatcher(
            path=minimal_yaml,
            on_change=lambda c: changes.append(c),
            validate=False,
            interval=60,
        )
        # Simulate "file changed by setting last_mtime to something old"
        watcher._last_mtime = 0.0  # very old mtime
        changed = watcher.check_now()
        assert changed is True
        assert len(changes) == 1

    def test_on_error_callback_invoked(self, tmp_dir):
        """Test that on_error callback is invoked when file becomes invalid."""
        # Create a valid file
        path = tmp_dir / "watcher_test.yaml"
        path.write_text("timeout: 30\n")

        errors = []
        watcher = ConfigWatcher(
            path=path,
            on_change=lambda c: None,
            on_error=lambda e: errors.append(e),
            validate=True,
            interval=60,
        )

        # Now write invalid YAML to trigger an error on next check
        path.write_text("[unclosed bracket\n")
        watcher._last_mtime = 0.0  # force change detection
        watcher.check_now()
        assert len(errors) >= 1

    def test_watcher_with_nonexistent_file_does_not_crash(self, tmp_dir):
        """Test that watcher on nonexistent file handles gracefully."""
        path = tmp_dir / "does_not_exist.yaml"
        watcher = ConfigWatcher(
            path=path,
            on_change=lambda c: None,
            interval=0.1,
        )
        # Checking mtime on nonexistent file should return None
        mtime = watcher._get_mtime()
        assert mtime is None

        # check_now should not crash
        result = watcher.check_now()
        assert result is False


# ---------------------------------------------------------------------------
# export_config tests
# ---------------------------------------------------------------------------


class TestExportConfig:
    """Tests for export_config function."""

    @pytest.fixture
    def sample_config(self):
        """Create a simple GatewayConfig for export tests."""
        from src.models import AuthType, ModelConfig, ProviderConfig

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
                        "gpt-4": ModelConfig(name="gpt-4", max_tokens=8192)
                    },
                )
            },
        )

    def test_export_yaml(self, sample_config):
        """Test exporting to YAML format."""
        result = export_config(sample_config, fmt="yaml")
        assert isinstance(result, str)
        data = yaml.safe_load(result)
        assert data["default_provider"] == "openai"

    def test_export_yml(self, sample_config):
        """Test exporting to yml (alias for yaml)."""
        result = export_config(sample_config, fmt="yml")
        assert isinstance(result, str)
        data = yaml.safe_load(result)
        assert data["default_provider"] == "openai"

    def test_export_json(self, sample_config):
        """Test exporting to JSON format."""
        result = export_config(sample_config, fmt="json")
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["default_provider"] == "openai"

    def test_export_json_ends_with_newline(self, sample_config):
        """Test that JSON export ends with a newline."""
        result = export_config(sample_config, fmt="json")
        assert result.endswith("\n")

    def test_export_unsupported_format_raises(self, sample_config):
        """Test that unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            export_config(sample_config, fmt="toml")

    def test_roundtrip_yaml(self, sample_config):
        """Test YAML export → import roundtrip."""
        exported = export_config(sample_config, fmt="yaml")
        imported = import_config(exported, fmt="yaml", validate=False)
        assert imported.default_provider == sample_config.default_provider
        assert imported.timeout == sample_config.timeout

    def test_roundtrip_json(self, sample_config):
        """Test JSON export → import roundtrip."""
        exported = export_config(sample_config, fmt="json")
        imported = import_config(exported, fmt="json", validate=False)
        assert imported.default_provider == sample_config.default_provider
        assert "openai" in imported.providers


# ---------------------------------------------------------------------------
# import_config tests
# ---------------------------------------------------------------------------


class TestImportConfig:
    """Tests for import_config function."""

    VALID_YAML = """\
default_provider: openai
log_level: info
timeout: 30
max_retries: 3
providers:
  openai:
    name: openai
    api_base: https://api.openai.com/v1
    api_key_env_var: OPENAI_API_KEY
    auth_type: bearer_token
    default_model: gpt-4o
    models:
      gpt-4o:
        name: gpt-4o
        max_tokens: 16384
"""

    VALID_JSON = json.dumps(
        {
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
                    "default_model": "gpt-4o",
                    "models": {
                        "gpt-4o": {"name": "gpt-4o", "max_tokens": 16384}
                    },
                }
            },
        }
    )

    def test_import_valid_yaml(self):
        """Test importing valid YAML string."""
        config = import_config(self.VALID_YAML, fmt="yaml", validate=False)
        assert config.default_provider == "openai"

    def test_import_valid_json(self):
        """Test importing valid JSON string."""
        config = import_config(self.VALID_JSON, fmt="json", validate=False)
        assert config.default_provider == "openai"

    def test_import_with_validation(self):
        """Test importing with validation enabled."""
        config = import_config(self.VALID_YAML, fmt="yaml", validate=True)
        assert isinstance(config, GatewayConfig)

    def test_import_invalid_yaml_raises(self):
        """Test that invalid YAML raises ConfigError."""
        with pytest.raises(ConfigError, match="Failed to parse"):
            import_config("[unclosed bracket", fmt="yaml")

    def test_import_invalid_json_raises(self):
        """Test that invalid JSON raises ConfigError."""
        with pytest.raises(ConfigError, match="Failed to parse"):
            import_config("{bad json", fmt="json")

    def test_import_list_yaml_raises(self):
        """Test that a YAML list raises ConfigError (not a dict)."""
        with pytest.raises(ConfigError, match="mapping"):
            import_config("- item1\n- item2\n", fmt="yaml")

    def test_import_unsupported_format_raises(self):
        """Test that unsupported format raises ConfigError."""
        with pytest.raises(ConfigError, match="Unsupported"):
            import_config("data: value", fmt="toml")

    def test_import_validates_by_default(self):
        """Test that validation is run by default on import."""
        invalid_content = "timeout: -1\nlog_level: bad_level\n"
        with pytest.raises(ConfigValidationError):
            import_config(invalid_content, fmt="yaml")
