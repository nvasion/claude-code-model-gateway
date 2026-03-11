"""Tests for src.config.manager.ConfigManager."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.manager import (
    ConfigManager,
    ConfigManagerError,
    ModelExistsError,
    ModelNotFoundError,
    ProviderExistsError,
    ProviderNotFoundError,
)
from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_config():
    """A GatewayConfig with two providers and a handful of models."""
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
                default_model="gpt-4o",
                models={
                    "gpt-4o": ModelConfig(
                        name="gpt-4o",
                        display_name="GPT-4o",
                        max_tokens=16384,
                        supports_streaming=True,
                        supports_tools=True,
                        supports_vision=True,
                    ),
                    "gpt-4o-mini": ModelConfig(
                        name="gpt-4o-mini",
                        display_name="GPT-4o Mini",
                        max_tokens=16384,
                        supports_streaming=True,
                        supports_tools=True,
                        supports_vision=True,
                    ),
                },
            ),
            "anthropic": ProviderConfig(
                name="anthropic",
                display_name="Anthropic",
                api_base="https://api.anthropic.com/v1",
                api_key_env_var="ANTHROPIC_API_KEY",
                auth_type=AuthType.API_KEY,
                default_model="claude-sonnet-4-20250514",
                models={
                    "claude-sonnet-4-20250514": ModelConfig(
                        name="claude-sonnet-4-20250514",
                        display_name="Claude Sonnet 4",
                        max_tokens=8192,
                        supports_streaming=True,
                        supports_tools=True,
                        supports_vision=True,
                    ),
                },
            ),
        },
    )


@pytest.fixture
def manager(simple_config):
    """A ConfigManager wrapping simple_config (no file path)."""
    return ConfigManager(config=simple_config)


@pytest.fixture
def config_file(tmp_path, simple_config):
    """Write simple_config to a YAML file and return its path."""
    path = tmp_path / "gateway.yaml"
    data = simple_config.to_dict()
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConfigManagerConstruction:
    """Tests for ConfigManager initialisation."""

    def test_from_config(self, simple_config):
        """ConfigManager wraps an in-memory config."""
        mgr = ConfigManager(config=simple_config)
        assert mgr.config is simple_config
        assert mgr.path is None

    def test_from_file(self, config_file):
        """ConfigManager loads from a file when auto_load is True."""
        mgr = ConfigManager(path=config_file)
        assert mgr.config is not None
        assert "openai" in mgr.config.providers

    def test_empty_when_no_path_no_config(self):
        """ConfigManager without path or config starts with empty config."""
        mgr = ConfigManager(auto_load=False)
        assert mgr.config.providers == {}
        assert mgr.path is None

    def test_auto_load_false_skips_file(self, config_file):
        """auto_load=False does not read the file."""
        mgr = ConfigManager(path=config_file, auto_load=False)
        assert mgr.config.providers == {}

    def test_from_file_classmethod(self, config_file):
        """from_file() factory loads the file."""
        mgr = ConfigManager.from_file(config_file)
        assert "openai" in mgr.config.providers

    def test_from_default_classmethod(self):
        """from_default() includes all built-in providers."""
        mgr = ConfigManager.from_default()
        assert "openai" in mgr.config.providers
        assert "anthropic" in mgr.config.providers
        assert "google" in mgr.config.providers

    def test_load_nonexistent_file_raises(self, tmp_path):
        """ConfigManager raises ConfigManagerError for missing file."""
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(ConfigManagerError):
            ConfigManager(path=missing, auto_load=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestConfigManagerPersistence:
    """Tests for load() and save()."""

    def test_save_creates_file(self, tmp_path, manager):
        """save() writes the config to disk."""
        out = tmp_path / "out.yaml"
        manager.save(out)
        assert out.exists()
        with open(out) as f:
            data = yaml.safe_load(f)
        assert "openai" in data["providers"]

    def test_save_then_load_roundtrip(self, tmp_path, manager):
        """Configuration survives a save→load roundtrip."""
        path = tmp_path / "roundtrip.yaml"
        manager.save(path)

        loaded = ConfigManager.from_file(path)
        assert loaded.config.default_provider == manager.config.default_provider
        assert set(loaded.config.providers.keys()) == set(manager.config.providers.keys())
        assert "gpt-4o" in loaded.config.providers["openai"].models

    def test_load_refreshes_in_memory_config(self, config_file, manager):
        """load() replaces the in-memory config with the file's content."""
        manager.path = config_file
        # Mutate in-memory before loading
        manager.config.default_provider = "mutated"
        manager.load()
        assert manager.config.default_provider == "openai"

    def test_load_without_path_raises(self, manager):
        """load() raises ConfigManagerError when path is None."""
        assert manager.path is None
        with pytest.raises(ConfigManagerError):
            manager.load()

    def test_save_without_path_raises(self, manager):
        """save() raises ConfigManagerError when no path is set."""
        assert manager.path is None
        with pytest.raises(ConfigManagerError):
            manager.save()

    def test_save_uses_attribute_path(self, config_file, simple_config):
        """save() without argument uses self.path."""
        mgr = ConfigManager(path=config_file, config=simple_config)
        mgr.config.log_level = "debug"
        mgr.save()

        reloaded = ConfigManager.from_file(config_file)
        assert reloaded.config.log_level == "debug"


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------


class TestConfigManagerSettings:
    """Tests for get_setting() / set_setting()."""

    def test_get_existing_setting(self, manager):
        """get_setting() returns the current value."""
        assert manager.get_setting("log_level") == "info"
        assert manager.get_setting("timeout") == 30

    def test_set_string_setting(self, manager):
        """set_setting() updates a string field."""
        manager.set_setting("log_level", "debug")
        assert manager.config.log_level == "debug"

    def test_set_integer_setting(self, manager):
        """set_setting() coerces integer fields."""
        manager.set_setting("timeout", "60")
        assert manager.config.timeout == 60

    def test_set_integer_setting_accepts_int(self, manager):
        """set_setting() accepts native int for integer fields."""
        manager.set_setting("max_retries", 5)
        assert manager.config.max_retries == 5

    def test_set_invalid_integer_raises(self, manager):
        """set_setting() raises ConfigManagerError for non-numeric integer field."""
        with pytest.raises(ConfigManagerError, match="integer"):
            manager.set_setting("timeout", "not-a-number")

    def test_get_unknown_key_raises(self, manager):
        """get_setting() raises ConfigManagerError for unknown keys."""
        with pytest.raises(ConfigManagerError):
            manager.get_setting("unknown_key")

    def test_set_unknown_key_raises(self, manager):
        """set_setting() raises ConfigManagerError for unknown keys."""
        with pytest.raises(ConfigManagerError):
            manager.set_setting("unknown_key", "value")


# ---------------------------------------------------------------------------
# Provider CRUD
# ---------------------------------------------------------------------------


class TestConfigManagerProviders:
    """Tests for provider-level operations."""

    def test_list_providers(self, manager):
        """list_providers() returns sorted names."""
        names = manager.list_providers()
        assert names == ["anthropic", "openai"]

    def test_has_provider_true(self, manager):
        """has_provider() returns True for existing providers."""
        assert manager.has_provider("openai") is True

    def test_has_provider_false(self, manager):
        """has_provider() returns False for unknown providers."""
        assert manager.has_provider("nonexistent") is False

    def test_get_provider(self, manager):
        """get_provider() returns the correct ProviderConfig."""
        p = manager.get_provider("openai")
        assert p.display_name == "OpenAI"

    def test_get_provider_not_found(self, manager):
        """get_provider() raises ProviderNotFoundError for missing providers."""
        with pytest.raises(ProviderNotFoundError):
            manager.get_provider("missing")

    def test_add_provider_basic(self, manager):
        """add_provider() creates a new provider."""
        p = manager.add_provider(
            "local-llm",
            api_base="http://localhost:8000/v1",
            api_key_env_var="LOCAL_KEY",
        )
        assert p.name == "local-llm"
        assert "local-llm" in manager.config.providers

    def test_add_provider_sets_default_when_first(self):
        """add_provider() auto-sets the default when it's the first provider."""
        mgr = ConfigManager(config=GatewayConfig())
        mgr.add_provider("first", api_base="http://example.com")
        assert mgr.config.default_provider == "first"

    def test_add_provider_does_not_override_existing_default(self, manager):
        """add_provider() does not change an existing default provider."""
        manager.add_provider("new", api_base="http://example.com")
        assert manager.config.default_provider == "openai"

    def test_add_duplicate_provider_raises(self, manager):
        """add_provider() raises ProviderExistsError when name is taken."""
        with pytest.raises(ProviderExistsError):
            manager.add_provider("openai", api_base="http://example.com")

    def test_add_provider_from_builtin(self, manager):
        """add_provider_from_builtin() copies a template."""
        p = manager.add_provider_from_builtin("google")
        assert p.name == "google"
        assert "google" in manager.config.providers
        assert len(p.models) > 0

    def test_add_provider_from_builtin_with_custom_name(self, manager):
        """add_provider_from_builtin() supports a custom identifier."""
        p = manager.add_provider_from_builtin("openai", name="my-openai")
        assert p.name == "my-openai"
        assert "my-openai" in manager.config.providers

    def test_add_provider_from_invalid_builtin(self, manager):
        """add_provider_from_builtin() raises ConfigManagerError for unknown names."""
        with pytest.raises(ConfigManagerError, match="Unknown built-in provider"):
            manager.add_provider_from_builtin("does-not-exist")

    def test_add_provider_from_builtin_duplicate_raises(self, manager):
        """add_provider_from_builtin() raises ProviderExistsError when name is taken."""
        with pytest.raises(ProviderExistsError):
            manager.add_provider_from_builtin("openai")

    def test_update_provider_api_base(self, manager):
        """update_provider() changes api_base."""
        manager.update_provider("openai", api_base="https://my-proxy.example.com/v1")
        assert manager.config.providers["openai"].api_base == "https://my-proxy.example.com/v1"

    def test_update_provider_multiple_fields(self, manager):
        """update_provider() can change multiple fields at once."""
        manager.update_provider(
            "openai",
            display_name="My OpenAI",
            api_key_env_var="MY_KEY",
            default_model="gpt-4o-mini",
        )
        p = manager.config.providers["openai"]
        assert p.display_name == "My OpenAI"
        assert p.api_key_env_var == "MY_KEY"
        assert p.default_model == "gpt-4o-mini"

    def test_update_provider_none_args_unchanged(self, manager):
        """update_provider() leaves fields unchanged when None is passed."""
        original_base = manager.config.providers["openai"].api_base
        manager.update_provider("openai", display_name="New Name")
        assert manager.config.providers["openai"].api_base == original_base

    def test_update_provider_not_found(self, manager):
        """update_provider() raises ProviderNotFoundError for missing providers."""
        with pytest.raises(ProviderNotFoundError):
            manager.update_provider("missing", api_base="http://x.com")

    def test_update_provider_auth_type(self, manager):
        """update_provider() can change auth_type."""
        manager.update_provider("openai", auth_type=AuthType.API_KEY)
        assert manager.config.providers["openai"].auth_type == AuthType.API_KEY

    def test_update_provider_headers(self, manager):
        """update_provider() replaces headers when supplied."""
        manager.update_provider("openai", headers={"X-Custom": "value"})
        assert manager.config.providers["openai"].headers == {"X-Custom": "value"}

    def test_remove_provider(self, manager):
        """remove_provider() deletes the provider."""
        manager.remove_provider("anthropic")
        assert "anthropic" not in manager.config.providers

    def test_remove_provider_updates_default(self, manager):
        """Removing the default provider promotes the next one."""
        assert manager.config.default_provider == "openai"
        manager.remove_provider("openai")
        assert manager.config.default_provider == "anthropic"

    def test_remove_nonexistent_provider_raises(self, manager):
        """remove_provider() raises ProviderNotFoundError for missing providers."""
        with pytest.raises(ProviderNotFoundError):
            manager.remove_provider("nonexistent")

    def test_enable_disable_provider(self, manager):
        """enable_provider() and disable_provider() toggle the enabled flag."""
        manager.disable_provider("openai")
        assert manager.config.providers["openai"].enabled is False

        manager.enable_provider("openai")
        assert manager.config.providers["openai"].enabled is True

    def test_disable_nonexistent_provider_raises(self, manager):
        """disable_provider() raises ProviderNotFoundError for missing providers."""
        with pytest.raises(ProviderNotFoundError):
            manager.disable_provider("nonexistent")

    def test_enable_nonexistent_provider_raises(self, manager):
        """enable_provider() raises ProviderNotFoundError for missing providers."""
        with pytest.raises(ProviderNotFoundError):
            manager.enable_provider("nonexistent")

    def test_set_default_provider(self, manager):
        """set_default_provider() updates the default."""
        manager.set_default_provider("anthropic")
        assert manager.config.default_provider == "anthropic"

    def test_set_default_provider_not_found(self, manager):
        """set_default_provider() raises ProviderNotFoundError for missing providers."""
        with pytest.raises(ProviderNotFoundError):
            manager.set_default_provider("nonexistent")


# ---------------------------------------------------------------------------
# Model CRUD
# ---------------------------------------------------------------------------


class TestConfigManagerModels:
    """Tests for model-level operations."""

    def test_list_models(self, manager):
        """list_models() returns sorted model names."""
        models = manager.list_models("openai")
        assert models == ["gpt-4o", "gpt-4o-mini"]

    def test_list_models_provider_not_found(self, manager):
        """list_models() raises ProviderNotFoundError for missing providers."""
        with pytest.raises(ProviderNotFoundError):
            manager.list_models("nonexistent")

    def test_get_model(self, manager):
        """get_model() returns the correct ModelConfig."""
        m = manager.get_model("openai", "gpt-4o")
        assert m.display_name == "GPT-4o"
        assert m.max_tokens == 16384

    def test_get_model_provider_not_found(self, manager):
        """get_model() raises ProviderNotFoundError for missing provider."""
        with pytest.raises(ProviderNotFoundError):
            manager.get_model("nonexistent", "gpt-4o")

    def test_get_model_model_not_found(self, manager):
        """get_model() raises ModelNotFoundError for missing model."""
        with pytest.raises(ModelNotFoundError):
            manager.get_model("openai", "nonexistent-model")

    def test_has_model_true(self, manager):
        """has_model() returns True for existing models."""
        assert manager.has_model("openai", "gpt-4o") is True

    def test_has_model_false(self, manager):
        """has_model() returns False for missing models."""
        assert manager.has_model("openai", "nonexistent") is False

    def test_has_model_provider_not_found(self, manager):
        """has_model() raises ProviderNotFoundError for missing providers."""
        with pytest.raises(ProviderNotFoundError):
            manager.has_model("nonexistent", "model")

    def test_add_model(self, manager):
        """add_model() creates a new model on a provider."""
        m = manager.add_model(
            "openai",
            "gpt-4-turbo",
            max_tokens=4096,
            supports_tools=True,
        )
        assert m.name == "gpt-4-turbo"
        assert "gpt-4-turbo" in manager.config.providers["openai"].models

    def test_add_model_auto_sets_default_when_empty(self):
        """add_model() auto-sets default_model when the provider has none."""
        provider = ProviderConfig(name="empty", api_base="http://example.com")
        mgr = ConfigManager(config=GatewayConfig(providers={"empty": provider}))
        mgr.add_model("empty", "first-model")
        assert provider.default_model == "first-model"

    def test_add_model_does_not_override_existing_default(self, manager):
        """add_model() does not change an existing default_model."""
        assert manager.config.providers["openai"].default_model == "gpt-4o"
        manager.add_model("openai", "new-model")
        assert manager.config.providers["openai"].default_model == "gpt-4o"

    def test_add_model_display_name_defaults_to_name(self, manager):
        """add_model() uses model_name as display_name when not provided."""
        m = manager.add_model("openai", "my-model")
        assert m.display_name == "my-model"

    def test_add_model_duplicate_raises(self, manager):
        """add_model() raises ModelExistsError when model already exists."""
        with pytest.raises(ModelExistsError):
            manager.add_model("openai", "gpt-4o")

    def test_add_model_provider_not_found(self, manager):
        """add_model() raises ProviderNotFoundError for missing provider."""
        with pytest.raises(ProviderNotFoundError):
            manager.add_model("nonexistent", "my-model")

    def test_update_model_single_field(self, manager):
        """update_model() changes a single field."""
        manager.update_model("openai", "gpt-4o", max_tokens=8192)
        assert manager.config.providers["openai"].models["gpt-4o"].max_tokens == 8192

    def test_update_model_multiple_fields(self, manager):
        """update_model() can update multiple fields at once."""
        manager.update_model(
            "openai",
            "gpt-4o",
            display_name="GPT-4o Updated",
            supports_tools=False,
            supports_vision=False,
        )
        m = manager.config.providers["openai"].models["gpt-4o"]
        assert m.display_name == "GPT-4o Updated"
        assert m.supports_tools is False
        assert m.supports_vision is False

    def test_update_model_none_args_unchanged(self, manager):
        """update_model() leaves fields unchanged when None is passed."""
        original_tokens = manager.config.providers["openai"].models["gpt-4o"].max_tokens
        manager.update_model("openai", "gpt-4o", display_name="Changed")
        assert manager.config.providers["openai"].models["gpt-4o"].max_tokens == original_tokens

    def test_update_model_provider_not_found(self, manager):
        """update_model() raises ProviderNotFoundError for missing provider."""
        with pytest.raises(ProviderNotFoundError):
            manager.update_model("nonexistent", "gpt-4o", max_tokens=100)

    def test_update_model_not_found(self, manager):
        """update_model() raises ModelNotFoundError for missing model."""
        with pytest.raises(ModelNotFoundError):
            manager.update_model("openai", "nonexistent-model", max_tokens=100)

    def test_update_model_extra(self, manager):
        """update_model() replaces extra when supplied."""
        manager.update_model("openai", "gpt-4o", extra={"note": "test"})
        assert manager.config.providers["openai"].models["gpt-4o"].extra == {"note": "test"}

    def test_remove_model(self, manager):
        """remove_model() deletes a model from a provider."""
        manager.remove_model("openai", "gpt-4o-mini")
        assert "gpt-4o-mini" not in manager.config.providers["openai"].models

    def test_remove_default_model_promotes_next(self, manager):
        """Removing the default model promotes the next alphabetically."""
        # Add a second model first and set it as default
        manager.config.providers["anthropic"].models["claude-3-opus"] = ModelConfig(
            name="claude-3-opus", max_tokens=4096
        )
        manager.config.providers["anthropic"].default_model = "claude-3-opus"
        manager.remove_model("anthropic", "claude-3-opus")
        # The remaining model should become default
        assert manager.config.providers["anthropic"].default_model == "claude-sonnet-4-20250514"

    def test_remove_last_model_clears_default(self):
        """Removing the last model clears the provider's default_model."""
        provider = ProviderConfig(
            name="solo",
            api_base="http://example.com",
            default_model="only-model",
            models={"only-model": ModelConfig(name="only-model")},
        )
        mgr = ConfigManager(config=GatewayConfig(providers={"solo": provider}))
        mgr.remove_model("solo", "only-model")
        assert provider.default_model == ""

    def test_remove_model_provider_not_found(self, manager):
        """remove_model() raises ProviderNotFoundError for missing provider."""
        with pytest.raises(ProviderNotFoundError):
            manager.remove_model("nonexistent", "gpt-4o")

    def test_remove_model_not_found(self, manager):
        """remove_model() raises ModelNotFoundError for missing model."""
        with pytest.raises(ModelNotFoundError):
            manager.remove_model("openai", "nonexistent-model")

    def test_set_default_model(self, manager):
        """set_default_model() updates the provider's default_model."""
        manager.set_default_model("openai", "gpt-4o-mini")
        assert manager.config.providers["openai"].default_model == "gpt-4o-mini"

    def test_set_default_model_provider_not_found(self, manager):
        """set_default_model() raises ProviderNotFoundError for missing provider."""
        with pytest.raises(ProviderNotFoundError):
            manager.set_default_model("nonexistent", "gpt-4o")

    def test_set_default_model_not_found(self, manager):
        """set_default_model() raises ModelNotFoundError for missing model."""
        with pytest.raises(ModelNotFoundError):
            manager.set_default_model("openai", "nonexistent-model")


# ---------------------------------------------------------------------------
# Exception properties
# ---------------------------------------------------------------------------


class TestConfigManagerExceptions:
    """Verify exception attributes are set correctly."""

    def test_provider_not_found_error_name(self):
        """ProviderNotFoundError carries the provider name."""
        exc = ProviderNotFoundError("my-provider")
        assert exc.provider_name == "my-provider"
        assert "my-provider" in str(exc)

    def test_provider_exists_error_name(self):
        """ProviderExistsError carries the provider name."""
        exc = ProviderExistsError("openai")
        assert exc.provider_name == "openai"
        assert "openai" in str(exc)

    def test_model_not_found_error_fields(self):
        """ModelNotFoundError carries provider and model names."""
        exc = ModelNotFoundError("openai", "gpt-99")
        assert exc.provider_name == "openai"
        assert exc.model_name == "gpt-99"
        assert "gpt-99" in str(exc)

    def test_model_exists_error_fields(self):
        """ModelExistsError carries provider and model names."""
        exc = ModelExistsError("openai", "gpt-4o")
        assert exc.provider_name == "openai"
        assert exc.model_name == "gpt-4o"


# ---------------------------------------------------------------------------
# End-to-end: mutate then persist
# ---------------------------------------------------------------------------


class TestConfigManagerEndToEnd:
    """Integration tests that combine multiple operations."""

    def test_add_provider_add_model_save_reload(self, tmp_path):
        """Add a provider, add a model, save, reload — all changes survive."""
        path = tmp_path / "e2e.yaml"

        mgr = ConfigManager.from_default()
        mgr.add_provider(
            "local",
            api_base="http://localhost:11434/v1",
            display_name="Ollama",
        )
        mgr.add_model("local", "llama3", display_name="LLaMA 3", max_tokens=4096)
        mgr.save(path)

        reloaded = ConfigManager.from_file(path)
        assert "local" in reloaded.config.providers
        assert "llama3" in reloaded.config.providers["local"].models
        assert reloaded.config.providers["local"].models["llama3"].max_tokens == 4096

    def test_update_provider_update_model_save_reload(self, tmp_path, simple_config):
        """Update a provider and one of its models, save, reload."""
        path = tmp_path / "update.yaml"
        mgr = ConfigManager(path=path, config=simple_config)

        mgr.update_provider("openai", api_base="https://proxy.example.com/v1")
        mgr.update_model("openai", "gpt-4o", max_tokens=8192, supports_tools=False)
        mgr.save()

        reloaded = ConfigManager.from_file(path)
        p = reloaded.config.providers["openai"]
        assert p.api_base == "https://proxy.example.com/v1"
        assert p.models["gpt-4o"].max_tokens == 8192
        assert p.models["gpt-4o"].supports_tools is False

    def test_remove_provider_save_reload(self, tmp_path, simple_config):
        """Remove a provider, save, reload — provider is absent."""
        path = tmp_path / "remove.yaml"
        mgr = ConfigManager(path=path, config=simple_config)

        mgr.remove_provider("anthropic")
        mgr.save()

        reloaded = ConfigManager.from_file(path)
        assert "anthropic" not in reloaded.config.providers

    def test_remove_model_save_reload(self, tmp_path, simple_config):
        """Remove a model, save, reload — model is absent."""
        path = tmp_path / "rm_model.yaml"
        mgr = ConfigManager(path=path, config=simple_config)

        mgr.remove_model("openai", "gpt-4o-mini")
        mgr.save()

        reloaded = ConfigManager.from_file(path)
        assert "gpt-4o-mini" not in reloaded.config.providers["openai"].models

    def test_set_default_model_save_reload(self, tmp_path, simple_config):
        """Set default model, save, reload — default is persisted."""
        path = tmp_path / "default.yaml"
        mgr = ConfigManager(path=path, config=simple_config)

        mgr.set_default_model("openai", "gpt-4o-mini")
        mgr.save()

        reloaded = ConfigManager.from_file(path)
        assert reloaded.config.providers["openai"].default_model == "gpt-4o-mini"

    def test_settings_persisted(self, tmp_path, simple_config):
        """Top-level setting changes survive save→reload."""
        path = tmp_path / "settings.yaml"
        mgr = ConfigManager(path=path, config=simple_config)

        mgr.set_setting("log_level", "warning")
        mgr.set_setting("timeout", 120)
        mgr.save()

        reloaded = ConfigManager.from_file(path)
        assert reloaded.config.log_level == "warning"
        assert reloaded.config.timeout == 120
