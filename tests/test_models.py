"""Tests for data models."""

import pytest

from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig


# ---------------------------------------------------------------------------
# ModelConfig tests
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_defaults(self):
        """Test ModelConfig default values."""
        model = ModelConfig(name="test-model")
        assert model.name == "test-model"
        assert model.display_name == "test-model"  # defaults to name
        assert model.max_tokens == 4096
        assert model.supports_streaming is True
        assert model.supports_tools is False
        assert model.supports_vision is False
        assert model.extra == {}

    def test_custom_display_name(self):
        """Test ModelConfig with custom display_name."""
        model = ModelConfig(name="gpt-4", display_name="GPT-4")
        assert model.display_name == "GPT-4"

    def test_to_dict(self):
        """Test ModelConfig serialization."""
        model = ModelConfig(
            name="gpt-4o",
            display_name="GPT-4o",
            max_tokens=16384,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        )
        data = model.to_dict()
        assert data["name"] == "gpt-4o"
        assert data["display_name"] == "GPT-4o"
        assert data["max_tokens"] == 16384
        assert data["supports_tools"] is True
        assert data["supports_vision"] is True

    def test_to_dict_no_display_name_when_same(self):
        """Test that display_name is omitted when same as name."""
        model = ModelConfig(name="gpt-4")
        data = model.to_dict()
        assert "display_name" not in data

    def test_to_dict_extra_included(self):
        """Test that extra fields are serialized."""
        model = ModelConfig(name="test", extra={"foo": "bar"})
        data = model.to_dict()
        assert data["extra"] == {"foo": "bar"}

    def test_to_dict_extra_omitted_when_empty(self):
        """Test that extra is not serialized when empty."""
        model = ModelConfig(name="test")
        data = model.to_dict()
        assert "extra" not in data

    def test_from_dict(self):
        """Test ModelConfig deserialization."""
        data = {
            "name": "claude-3",
            "display_name": "Claude 3",
            "max_tokens": 8192,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": True,
            "extra": {"key": "value"},
        }
        model = ModelConfig.from_dict(data)
        assert model.name == "claude-3"
        assert model.display_name == "Claude 3"
        assert model.max_tokens == 8192
        assert model.supports_tools is True
        assert model.extra == {"key": "value"}

    def test_from_dict_defaults(self):
        """Test deserialization with minimal data."""
        data = {"name": "test-model"}
        model = ModelConfig.from_dict(data)
        assert model.name == "test-model"
        assert model.max_tokens == 4096
        assert model.supports_streaming is True
        assert model.supports_tools is False

    def test_roundtrip(self):
        """Test that serialization and deserialization are consistent."""
        original = ModelConfig(
            name="test",
            display_name="Test Model",
            max_tokens=2048,
            supports_streaming=False,
            supports_tools=True,
            supports_vision=True,
            extra={"version": "2"},
        )
        data = original.to_dict()
        restored = ModelConfig.from_dict(data)
        assert restored.name == original.name
        assert restored.display_name == original.display_name
        assert restored.max_tokens == original.max_tokens
        assert restored.supports_streaming == original.supports_streaming
        assert restored.supports_tools == original.supports_tools
        assert restored.supports_vision == original.supports_vision
        assert restored.extra == original.extra


# ---------------------------------------------------------------------------
# ProviderConfig tests
# ---------------------------------------------------------------------------


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_defaults(self):
        """Test ProviderConfig default values."""
        provider = ProviderConfig(name="test")
        assert provider.name == "test"
        assert provider.display_name == "test"
        assert provider.api_base == ""
        assert provider.api_key_env_var == ""
        assert provider.auth_type == AuthType.API_KEY
        assert provider.default_model == ""
        assert provider.models == {}
        assert provider.headers == {}
        assert provider.extra == {}
        assert provider.enabled is True

    def test_get_model(self):
        """Test getting a model by name."""
        model = ModelConfig(name="gpt-4")
        provider = ProviderConfig(
            name="openai",
            models={"gpt-4": model},
            default_model="gpt-4",
        )
        assert provider.get_model("gpt-4") == model
        assert provider.get_model() == model  # uses default
        assert provider.get_model("nonexistent") is None

    def test_list_models(self):
        """Test listing model names."""
        provider = ProviderConfig(
            name="test",
            models={
                "b-model": ModelConfig(name="b-model"),
                "a-model": ModelConfig(name="a-model"),
            },
        )
        assert provider.list_models() == ["a-model", "b-model"]

    def test_to_dict(self):
        """Test ProviderConfig serialization."""
        provider = ProviderConfig(
            name="openai",
            display_name="OpenAI",
            api_base="https://api.openai.com/v1",
            api_key_env_var="OPENAI_API_KEY",
            auth_type=AuthType.BEARER_TOKEN,
            default_model="gpt-4",
            headers={"X-Custom": "value"},
            extra={"org_id": "123"},
            enabled=True,
        )
        data = provider.to_dict()
        assert data["name"] == "openai"
        assert data["display_name"] == "OpenAI"
        assert data["api_base"] == "https://api.openai.com/v1"
        assert data["auth_type"] == "bearer_token"
        assert data["headers"] == {"X-Custom": "value"}
        assert data["extra"] == {"org_id": "123"}

    def test_from_dict(self):
        """Test ProviderConfig deserialization."""
        data = {
            "name": "anthropic",
            "display_name": "Anthropic",
            "api_base": "https://api.anthropic.com/v1",
            "api_key_env_var": "ANTHROPIC_API_KEY",
            "auth_type": "api_key",
            "default_model": "claude-3",
            "models": {
                "claude-3": {
                    "name": "claude-3",
                    "max_tokens": 8192,
                }
            },
            "enabled": True,
        }
        provider = ProviderConfig.from_dict(data)
        assert provider.name == "anthropic"
        assert provider.display_name == "Anthropic"
        assert provider.auth_type == AuthType.API_KEY
        assert "claude-3" in provider.models
        assert provider.models["claude-3"].max_tokens == 8192

    def test_from_dict_invalid_auth_type(self):
        """Test that invalid auth_type defaults to API_KEY."""
        data = {"name": "test", "auth_type": "invalid_type"}
        provider = ProviderConfig.from_dict(data)
        assert provider.auth_type == AuthType.API_KEY

    def test_from_dict_model_name_inferred(self):
        """Test that model name is inferred from dict key."""
        data = {
            "name": "test",
            "models": {
                "my-model": {"max_tokens": 2048}
            },
        }
        provider = ProviderConfig.from_dict(data)
        assert "my-model" in provider.models
        assert provider.models["my-model"].name == "my-model"

    def test_from_dict_models_as_list(self):
        """Test that models can be provided as a list instead of a dict."""
        data = {
            "name": "test",
            "models": [
                {"name": "model-a", "max_tokens": 1024},
                {"name": "model-b", "max_tokens": 2048, "supports_tools": True},
            ],
        }
        provider = ProviderConfig.from_dict(data)
        assert "model-a" in provider.models
        assert "model-b" in provider.models
        assert provider.models["model-a"].max_tokens == 1024
        assert provider.models["model-b"].supports_tools is True

    def test_from_dict_models_as_list_skips_entries_without_name(self):
        """Test that list items without a 'name' key are skipped gracefully."""
        data = {
            "name": "test",
            "models": [
                {"max_tokens": 512},          # no name – should be skipped
                {"name": "valid", "max_tokens": 4096},
            ],
        }
        provider = ProviderConfig.from_dict(data)
        assert "valid" in provider.models
        assert len(provider.models) == 1

    def test_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = ProviderConfig(
            name="test",
            display_name="Test Provider",
            api_base="https://example.com",
            api_key_env_var="TEST_KEY",
            auth_type=AuthType.BEARER_TOKEN,
            default_model="model-a",
            models={
                "model-a": ModelConfig(name="model-a", max_tokens=1024),
            },
            headers={"X-Version": "1"},
            extra={"note": "test"},
            enabled=False,
        )
        data = original.to_dict()
        restored = ProviderConfig.from_dict(data)
        assert restored.name == original.name
        assert restored.display_name == original.display_name
        assert restored.api_base == original.api_base
        assert restored.auth_type == original.auth_type
        assert restored.enabled == original.enabled
        assert "model-a" in restored.models


# ---------------------------------------------------------------------------
# GatewayConfig tests
# ---------------------------------------------------------------------------


class TestGatewayConfig:
    """Tests for GatewayConfig dataclass."""

    def test_defaults(self):
        """Test GatewayConfig default values."""
        config = GatewayConfig()
        assert config.default_provider == ""
        assert config.providers == {}
        assert config.log_level == "info"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.extra == {}

    def test_get_provider(self):
        """Test getting a provider by name."""
        provider = ProviderConfig(name="openai")
        config = GatewayConfig(
            default_provider="openai",
            providers={"openai": provider},
        )
        assert config.get_provider("openai") == provider
        assert config.get_provider() == provider  # uses default
        assert config.get_provider("nonexistent") is None

    def test_get_enabled_providers(self):
        """Test getting only enabled providers."""
        config = GatewayConfig(
            providers={
                "a": ProviderConfig(name="a", enabled=True),
                "b": ProviderConfig(name="b", enabled=False),
                "c": ProviderConfig(name="c", enabled=True),
            }
        )
        enabled = config.get_enabled_providers()
        assert "a" in enabled
        assert "b" not in enabled
        assert "c" in enabled

    def test_list_providers(self):
        """Test listing provider names."""
        config = GatewayConfig(
            providers={
                "beta": ProviderConfig(name="beta"),
                "alpha": ProviderConfig(name="alpha"),
            }
        )
        assert config.list_providers() == ["alpha", "beta"]

    def test_add_provider(self):
        """Test adding a provider."""
        config = GatewayConfig()
        provider = ProviderConfig(name="openai")
        config.add_provider(provider)
        assert "openai" in config.providers
        assert config.default_provider == "openai"  # auto-set as first

    def test_add_provider_does_not_override_default(self):
        """Test that adding a second provider doesn't change default."""
        config = GatewayConfig(default_provider="openai")
        config.providers["openai"] = ProviderConfig(name="openai")
        config.add_provider(ProviderConfig(name="anthropic"))
        assert config.default_provider == "openai"

    def test_remove_provider(self):
        """Test removing a provider."""
        config = GatewayConfig(
            default_provider="openai",
            providers={
                "openai": ProviderConfig(name="openai"),
                "anthropic": ProviderConfig(name="anthropic"),
            },
        )
        assert config.remove_provider("openai") is True
        assert "openai" not in config.providers
        assert config.default_provider == "anthropic"

    def test_remove_nonexistent_provider(self):
        """Test removing a provider that doesn't exist."""
        config = GatewayConfig()
        assert config.remove_provider("nonexistent") is False

    def test_remove_last_provider_clears_default(self):
        """Test that removing the last provider clears default."""
        config = GatewayConfig(
            default_provider="only",
            providers={"only": ProviderConfig(name="only")},
        )
        config.remove_provider("only")
        assert config.default_provider == ""

    def test_to_dict(self):
        """Test GatewayConfig serialization."""
        config = GatewayConfig(
            default_provider="openai",
            log_level="debug",
            timeout=60,
            max_retries=5,
            providers={
                "openai": ProviderConfig(
                    name="openai",
                    api_base="https://api.openai.com/v1",
                ),
            },
            extra={"feature_flag": True},
        )
        data = config.to_dict()
        assert data["default_provider"] == "openai"
        assert data["log_level"] == "debug"
        assert data["timeout"] == 60
        assert data["max_retries"] == 5
        assert "openai" in data["providers"]
        assert data["extra"] == {"feature_flag": True}

    def test_from_dict(self):
        """Test GatewayConfig deserialization."""
        data = {
            "default_provider": "anthropic",
            "log_level": "warning",
            "timeout": 45,
            "max_retries": 2,
            "providers": {
                "anthropic": {
                    "name": "anthropic",
                    "api_base": "https://api.anthropic.com/v1",
                }
            },
        }
        config = GatewayConfig.from_dict(data)
        assert config.default_provider == "anthropic"
        assert config.log_level == "warning"
        assert config.timeout == 45
        assert config.max_retries == 2
        assert "anthropic" in config.providers

    def test_from_dict_provider_name_inferred(self):
        """Test that provider name is inferred from dict key."""
        data = {
            "providers": {
                "my-provider": {"api_base": "https://example.com"}
            }
        }
        config = GatewayConfig.from_dict(data)
        assert "my-provider" in config.providers
        assert config.providers["my-provider"].name == "my-provider"

    def test_from_dict_providers_as_list(self):
        """Test that providers can be given as a list instead of a dict."""
        data = {
            "default_provider": "prov-a",
            "providers": [
                {"name": "prov-a", "api_base": "https://a.example.com"},
                {"name": "prov-b", "api_base": "https://b.example.com"},
            ],
        }
        config = GatewayConfig.from_dict(data)
        assert "prov-a" in config.providers
        assert "prov-b" in config.providers
        assert config.providers["prov-a"].api_base == "https://a.example.com"

    def test_from_dict_providers_as_list_with_models_as_list(self):
        """Test nested list format: providers list containing models list."""
        data = {
            "providers": [
                {
                    "name": "openai",
                    "api_base": "https://api.openai.com/v1",
                    "models": [
                        {"name": "gpt-4", "max_tokens": 8192},
                        {"name": "gpt-3.5-turbo", "max_tokens": 4096},
                    ],
                }
            ]
        }
        config = GatewayConfig.from_dict(data)
        assert "openai" in config.providers
        provider = config.providers["openai"]
        assert "gpt-4" in provider.models
        assert "gpt-3.5-turbo" in provider.models
        assert provider.models["gpt-4"].max_tokens == 8192

    def test_roundtrip(self):
        """Test full serialization/deserialization roundtrip."""
        original = GatewayConfig(
            default_provider="openai",
            log_level="debug",
            timeout=60,
            max_retries=5,
            providers={
                "openai": ProviderConfig(
                    name="openai",
                    api_base="https://api.openai.com/v1",
                    models={
                        "gpt-4": ModelConfig(name="gpt-4", max_tokens=8192),
                    },
                ),
            },
        )
        data = original.to_dict()
        restored = GatewayConfig.from_dict(data)
        assert restored.default_provider == original.default_provider
        assert restored.log_level == original.log_level
        assert restored.timeout == original.timeout
        assert "openai" in restored.providers
        assert "gpt-4" in restored.providers["openai"].models

    # ------------------------------------------------------------------
    # find_provider_for_model tests
    # ------------------------------------------------------------------

    def _make_multi_provider_config(self) -> GatewayConfig:
        """Build a GatewayConfig with two providers for routing tests."""
        openai_provider = ProviderConfig(
            name="openai",
            display_name="OpenAI",
            api_base="https://api.openai.com/v1",
            models={
                "gpt-4o": ModelConfig(name="gpt-4o"),
                "gpt-4o-mini": ModelConfig(name="gpt-4o-mini"),
            },
        )
        anthropic_provider = ProviderConfig(
            name="anthropic",
            display_name="Anthropic",
            api_base="https://api.anthropic.com/v1",
            models={
                "claude-sonnet-4-20250514": ModelConfig(name="claude-sonnet-4-20250514"),
            },
        )
        return GatewayConfig(
            default_provider="anthropic",
            providers={
                "openai": openai_provider,
                "anthropic": anthropic_provider,
            },
        )

    def test_find_provider_for_model_returns_correct_provider(self):
        """find_provider_for_model returns the provider that owns the model."""
        config = self._make_multi_provider_config()
        provider = config.find_provider_for_model("gpt-4o")
        assert provider is not None
        assert provider.name == "openai"

    def test_find_provider_for_model_second_provider(self):
        """find_provider_for_model finds models in the second provider."""
        config = self._make_multi_provider_config()
        provider = config.find_provider_for_model("claude-sonnet-4-20250514")
        assert provider is not None
        assert provider.name == "anthropic"

    def test_find_provider_for_model_unknown_falls_back_to_default(self):
        """Unknown model name falls back to the default provider."""
        config = self._make_multi_provider_config()
        provider = config.find_provider_for_model("some-unknown-model")
        assert provider is not None
        assert provider.name == "anthropic"  # the default provider

    def test_find_provider_for_model_skips_disabled_providers(self):
        """Disabled providers are not considered during model lookup."""
        disabled_openai = ProviderConfig(
            name="openai",
            models={"gpt-4o": ModelConfig(name="gpt-4o")},
            enabled=False,
        )
        default_provider = ProviderConfig(
            name="anthropic",
            models={"claude-sonnet-4-20250514": ModelConfig(name="claude-sonnet-4-20250514")},
            enabled=True,
        )
        config = GatewayConfig(
            default_provider="anthropic",
            providers={
                "openai": disabled_openai,
                "anthropic": default_provider,
            },
        )
        # gpt-4o is in the disabled openai provider, should fall back to default
        provider = config.find_provider_for_model("gpt-4o")
        assert provider is not None
        assert provider.name == "anthropic"

    def test_find_provider_for_model_no_providers_returns_none(self):
        """Returns None when no providers are configured."""
        config = GatewayConfig()
        provider = config.find_provider_for_model("gpt-4o")
        assert provider is None

    def test_find_provider_for_model_all_providers_disabled_returns_none(self):
        """Returns None when all providers are disabled and no default matches."""
        config = GatewayConfig(
            default_provider="nonexistent",
            providers={
                "openai": ProviderConfig(
                    name="openai",
                    models={"gpt-4o": ModelConfig(name="gpt-4o")},
                    enabled=False,
                ),
            },
        )
        provider = config.find_provider_for_model("gpt-4o")
        assert provider is None  # default_provider 'nonexistent' not in providers


# ---------------------------------------------------------------------------
# AuthType tests
# ---------------------------------------------------------------------------


class TestAuthType:
    """Tests for AuthType enum."""

    def test_values(self):
        """Test AuthType enum values."""
        assert AuthType.API_KEY.value == "api_key"
        assert AuthType.BEARER_TOKEN.value == "bearer_token"
        assert AuthType.NONE.value == "none"

    def test_from_string(self):
        """Test creating AuthType from string."""
        assert AuthType("api_key") == AuthType.API_KEY
        assert AuthType("bearer_token") == AuthType.BEARER_TOKEN
        assert AuthType("none") == AuthType.NONE

    def test_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            AuthType("invalid")
