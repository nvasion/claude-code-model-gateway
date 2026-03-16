"""Tests for built-in provider definitions and provider registry."""

import pytest

from src.models import AuthType, ModelConfig, ProviderConfig
from src.providers import (
    create_custom_provider,
    get_builtin_provider,
    get_builtin_providers,
    list_builtin_providers,
)


# ---------------------------------------------------------------------------
# Built-in providers tests
# ---------------------------------------------------------------------------


class TestBuiltinProviders:
    """Tests for the built-in provider registry."""

    def test_list_builtin_providers(self):
        """Test listing available built-in providers (canonical names only)."""
        names = list_builtin_providers()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "openai" in names
        assert "anthropic" in names
        assert "azure" in names
        assert "google" in names
        assert "bedrock" in names
        assert "openrouter" in names
        assert "local" in names
        # "gemini" is an alias resolved via get_builtin_provider(); it is
        # not a canonical name so it does not appear in the default list.
        assert "gemini" not in names

    def test_list_builtin_providers_with_aliases(self):
        """Test listing built-in providers including alias names."""
        names = list_builtin_providers(include_aliases=True)
        assert "google" in names
        assert "gemini" in names

    def test_list_is_sorted(self):
        """Test that the provider list is sorted."""
        names = list_builtin_providers()
        assert names == sorted(names)

    def test_get_all_builtin_providers(self):
        """Test getting all built-in provider configs."""
        providers = get_builtin_providers()
        assert isinstance(providers, dict)
        # get_builtin_providers() returns canonical names only (no aliases)
        assert len(providers) == len(list_builtin_providers())
        for name, provider in providers.items():
            assert isinstance(provider, ProviderConfig)
            # The provider's internal name must match its registry key
            assert provider.name == name

    def test_get_specific_builtin_provider(self):
        """Test getting a specific built-in provider."""
        provider = get_builtin_provider("openai")
        assert provider is not None
        assert provider.name == "openai"
        assert isinstance(provider, ProviderConfig)

    def test_get_nonexistent_provider(self):
        """Test that nonexistent provider returns None."""
        assert get_builtin_provider("nonexistent") is None

    def test_providers_are_fresh_instances(self):
        """Test that each call returns a fresh instance."""
        p1 = get_builtin_provider("openai")
        p2 = get_builtin_provider("openai")
        assert p1 is not p2
        assert p1.name == p2.name


# ---------------------------------------------------------------------------
# OpenAI provider tests
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    """Tests for the OpenAI built-in provider."""

    def test_basic_config(self):
        """Test OpenAI provider basic configuration."""
        p = get_builtin_provider("openai")
        assert p.display_name == "OpenAI"
        assert p.api_base == "https://api.openai.com/v1"
        assert p.api_key_env_var == "OPENAI_API_KEY"
        assert p.auth_type == AuthType.BEARER_TOKEN
        assert p.enabled is True

    def test_has_models(self):
        """Test that OpenAI provider has models configured."""
        p = get_builtin_provider("openai")
        assert len(p.models) > 0
        assert "gpt-4o" in p.models

    def test_default_model(self):
        """Test OpenAI default model."""
        p = get_builtin_provider("openai")
        assert p.default_model != ""
        assert p.default_model in p.models

    def test_gpt4o_features(self):
        """Test GPT-4o model features."""
        p = get_builtin_provider("openai")
        model = p.models.get("gpt-4o")
        assert model is not None
        assert model.supports_streaming is True
        assert model.supports_tools is True
        assert model.supports_vision is True
        assert model.max_tokens > 0


# ---------------------------------------------------------------------------
# Anthropic provider tests
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    """Tests for the Anthropic built-in provider."""

    def test_basic_config(self):
        """Test Anthropic provider basic configuration."""
        p = get_builtin_provider("anthropic")
        assert p.display_name == "Anthropic"
        assert p.api_base == "https://api.anthropic.com/v1"
        assert p.api_key_env_var == "ANTHROPIC_API_KEY"
        assert p.auth_type == AuthType.API_KEY

    def test_has_models(self):
        """Test that Anthropic provider has models configured."""
        p = get_builtin_provider("anthropic")
        assert len(p.models) > 0

    def test_has_anthropic_version_header(self):
        """Test that Anthropic provider has version header."""
        p = get_builtin_provider("anthropic")
        assert "anthropic-version" in p.headers

    def test_default_model_exists(self):
        """Test that default model exists in models list."""
        p = get_builtin_provider("anthropic")
        assert p.default_model in p.models


# ---------------------------------------------------------------------------
# Azure OpenAI provider tests
# ---------------------------------------------------------------------------


class TestAzureProvider:
    """Tests for the Azure OpenAI built-in provider."""

    def test_basic_config(self):
        """Test Azure provider basic configuration."""
        p = get_builtin_provider("azure")
        assert p.display_name == "Azure OpenAI"
        assert p.api_key_env_var == "AZURE_OPENAI_API_KEY"

    def test_has_extra_config(self):
        """Test that Azure provider has extra config for API version."""
        p = get_builtin_provider("azure")
        assert "api_version" in p.extra
        assert "deployment_name" in p.extra


# ---------------------------------------------------------------------------
# Google provider tests
# ---------------------------------------------------------------------------


class TestGoogleProvider:
    """Tests for the Google Gemini built-in provider."""

    def test_basic_config(self):
        """Test Google provider basic configuration."""
        p = get_builtin_provider("google")
        assert p.display_name == "Google Gemini"
        assert p.api_key_env_var == "GOOGLE_API_KEY"
        assert len(p.models) > 0


# ---------------------------------------------------------------------------
# Google / Gemini alias tests
# ---------------------------------------------------------------------------


class TestGeminiAlias:
    """Tests that 'gemini' works as an alias for the Google Gemini provider."""

    def test_gemini_alias_resolves(self):
        """Test that 'gemini' alias returns a valid provider config."""
        p = get_builtin_provider("gemini")
        assert p is not None

    def test_gemini_alias_has_models(self):
        """Test that gemini alias has models configured."""
        p = get_builtin_provider("gemini")
        assert len(p.models) > 0

    def test_gemini_alias_is_separate_instance(self):
        """Test that gemini and google return separate (but equivalent) instances."""
        pg = get_builtin_provider("google")
        pge = get_builtin_provider("gemini")
        assert pg is not pge
        assert pg.api_base == pge.api_base


# ---------------------------------------------------------------------------
# Bedrock provider tests
# ---------------------------------------------------------------------------


class TestBedrockProvider:
    """Tests for the AWS Bedrock built-in provider."""

    def test_basic_config(self):
        """Test Bedrock provider basic configuration."""
        p = get_builtin_provider("bedrock")
        assert p.display_name == "AWS Bedrock"
        assert p.auth_type == AuthType.NONE  # uses AWS credentials
        assert "region" in p.extra


# ---------------------------------------------------------------------------
# OpenRouter provider tests
# ---------------------------------------------------------------------------


class TestOpenRouterProvider:
    """Tests for the OpenRouter built-in provider."""

    def test_basic_config(self):
        """Test OpenRouter provider basic configuration."""
        p = get_builtin_provider("openrouter")
        assert p is not None
        assert p.name == "openrouter"
        assert p.display_name == "OpenRouter"
        assert p.api_base == "https://openrouter.ai/api/v1"
        assert p.api_key_env_var == "OPENROUTER_API_KEY"
        assert p.auth_type == AuthType.BEARER_TOKEN

    def test_has_models(self):
        """Test that OpenRouter provider has models configured."""
        p = get_builtin_provider("openrouter")
        assert len(p.models) > 0

    def test_has_claude_models(self):
        """Test that OpenRouter includes Anthropic Claude models."""
        p = get_builtin_provider("openrouter")
        claude_models = [m for m in p.models if m.startswith("anthropic/")]
        assert len(claude_models) > 0

    def test_has_openai_models(self):
        """Test that OpenRouter includes OpenAI models."""
        p = get_builtin_provider("openrouter")
        openai_models = [m for m in p.models if m.startswith("openai/")]
        assert len(openai_models) > 0

    def test_has_google_models(self):
        """Test that OpenRouter includes Google models."""
        p = get_builtin_provider("openrouter")
        google_models = [m for m in p.models if m.startswith("google/")]
        assert len(google_models) > 0

    def test_default_model_in_models(self):
        """Test that the default model is in the models list."""
        p = get_builtin_provider("openrouter")
        assert p.default_model in p.models

    def test_is_fresh_instance(self):
        """Test that each call returns a fresh instance."""
        p1 = get_builtin_provider("openrouter")
        p2 = get_builtin_provider("openrouter")
        assert p1 is not p2
        assert p1.name == p2.name


# ---------------------------------------------------------------------------
# Custom provider tests
# ---------------------------------------------------------------------------


class TestCustomProvider:
    """Tests for creating custom providers."""

    def test_create_custom_provider(self):
        """Test creating a custom provider."""
        p = create_custom_provider(
            name="local-llm",
            api_base="http://localhost:8000/v1",
            api_key_env_var="LOCAL_API_KEY",
            default_model="llama3",
            display_name="Local LLM",
        )
        assert p.name == "local-llm"
        assert p.display_name == "Local LLM"
        assert p.api_base == "http://localhost:8000/v1"
        assert p.api_key_env_var == "LOCAL_API_KEY"
        assert p.default_model == "llama3"
        assert p.auth_type == AuthType.BEARER_TOKEN  # has API key

    def test_create_custom_provider_no_auth(self):
        """Test creating a custom provider without authentication."""
        p = create_custom_provider(
            name="local",
            api_base="http://localhost:8000/v1",
        )
        assert p.auth_type == AuthType.NONE
        assert p.api_key_env_var == ""

    def test_create_custom_provider_default_display_name(self):
        """Test that display_name defaults to name."""
        p = create_custom_provider(
            name="my-provider",
            api_base="http://localhost:8000/v1",
        )
        assert p.display_name == "my-provider"

    def test_create_custom_provider_empty_models(self):
        """Test that custom provider starts with no models."""
        p = create_custom_provider(
            name="test",
            api_base="http://localhost:8000/v1",
        )
        assert p.models == {}

    def test_custom_provider_is_enabled(self):
        """Test that custom provider is enabled by default."""
        p = create_custom_provider(
            name="test",
            api_base="http://localhost:8000/v1",
        )
        assert p.enabled is True


# ---------------------------------------------------------------------------
# All providers validation tests
# ---------------------------------------------------------------------------


class TestAllProvidersValid:
    """Validate that all built-in providers have consistent configurations."""

    def test_all_providers_have_names(self):
        """Test all providers have non-empty names."""
        for name, provider in get_builtin_providers().items():
            assert provider.name == name
            assert provider.name != ""

    def test_all_providers_have_api_base(self):
        """Test all providers have API base URLs."""
        for name, provider in get_builtin_providers().items():
            assert provider.api_base != "", f"{name} missing api_base"

    def test_all_providers_have_display_names(self):
        """Test all providers have display names."""
        for name, provider in get_builtin_providers().items():
            assert provider.display_name != "", f"{name} missing display_name"

    def test_all_providers_have_default_model(self):
        """Test all providers have a default model."""
        for name, provider in get_builtin_providers().items():
            assert provider.default_model != "", f"{name} missing default_model"

    def test_all_default_models_exist(self):
        """Test all default models exist in their provider's model list."""
        for name, provider in get_builtin_providers().items():
            if provider.models:
                assert provider.default_model in provider.models, (
                    f"{name}: default_model '{provider.default_model}' "
                    f"not in models {list(provider.models.keys())}"
                )

    def test_all_models_have_positive_max_tokens(self):
        """Test all models have positive max_tokens."""
        for pname, provider in get_builtin_providers().items():
            for mname, model in provider.models.items():
                assert model.max_tokens > 0, (
                    f"{pname}/{mname} has non-positive max_tokens"
                )

    def test_all_providers_serializable(self):
        """Test that all providers can be serialized and deserialized."""
        for name, provider in get_builtin_providers().items():
            data = provider.to_dict()
            restored = ProviderConfig.from_dict(data)
            assert restored.name == provider.name
            assert len(restored.models) == len(provider.models)
