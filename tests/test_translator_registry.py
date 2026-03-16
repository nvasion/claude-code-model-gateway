"""Tests for TranslatorRegistry and module-level helpers."""

from __future__ import annotations

import pytest

from src.translators.base import BaseTranslator, TranslationError
from src.translators.registry import (
    TranslatorRegistry,
    get_registry,
    reset_registry,
)
from src.translators import (
    AnthropicTranslator,
    AzureOpenAITranslator,
    BedrockTranslator,
    GeminiTranslator,
    OpenAITranslator,
    translate_request,
    translate_response,
    translate_stream_chunk,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Ensure the global registry singleton is reset between tests."""
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def registry() -> TranslatorRegistry:
    return TranslatorRegistry()


@pytest.fixture
def simple_request() -> dict:
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
    }


# ---------------------------------------------------------------------------
# Built-in provider resolution
# ---------------------------------------------------------------------------


class TestRegistryBuiltinProviders:
    def test_get_openai(self, registry):
        t = registry.get("openai")
        assert t is not None
        assert isinstance(t, OpenAITranslator)

    def test_get_azure(self, registry):
        t = registry.get("azure")
        assert t is not None
        assert isinstance(t, AzureOpenAITranslator)

    def test_get_anthropic(self, registry):
        t = registry.get("anthropic")
        assert t is not None
        assert isinstance(t, AnthropicTranslator)

    def test_get_google(self, registry):
        t = registry.get("google")
        assert t is not None
        assert isinstance(t, GeminiTranslator)

    def test_get_gemini_alias(self, registry):
        t = registry.get("gemini")
        assert t is not None
        assert isinstance(t, GeminiTranslator)

    def test_get_bedrock(self, registry):
        t = registry.get("bedrock")
        assert t is not None
        assert isinstance(t, BedrockTranslator)

    def test_get_openrouter(self, registry):
        """OpenRouter uses the OpenAI-compatible translator."""
        t = registry.get("openrouter")
        assert t is not None
        assert isinstance(t, OpenAITranslator)

    def test_get_local(self, registry):
        """Local / Ollama uses the OpenAI-compatible translator."""
        t = registry.get("local")
        assert t is not None
        assert isinstance(t, OpenAITranslator)

    def test_openrouter_in_list_providers(self, registry):
        """openrouter should appear in the list of known providers."""
        assert "openrouter" in registry.list_providers()

    def test_get_unknown_returns_none(self, registry):
        assert registry.get("nonexistent-provider") is None

    def test_require_unknown_raises(self, registry):
        with pytest.raises(TranslationError, match="nonexistent"):
            registry.require("nonexistent-provider")

    def test_has_builtin_provider(self, registry):
        assert registry.has("openai") is True
        assert registry.has("anthropic") is True

    def test_has_unknown_provider(self, registry):
        assert registry.has("nonexistent") is False

    def test_contains_syntax(self, registry):
        assert "openai" in registry
        assert "nonexistent" not in registry


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistryRegistration:
    def test_register_custom_translator(self, registry):
        """A custom translator can be registered and retrieved."""

        class MyTranslator(BaseTranslator):
            PROVIDER_NAME = "my-provider"

            def translate_request(self, request, *, model=None):
                return {}

            def translate_response(self, response, *, model=None):
                return {}  # type: ignore

            def translate_stream_chunk(self, chunk, *, model=None):
                return None

        t = MyTranslator()
        registry.register("my-provider", t)
        assert registry.get("my-provider") is t

    def test_register_with_aliases(self, registry):
        """Aliases resolve to the same translator instance."""

        class AliasTranslator(BaseTranslator):
            PROVIDER_NAME = "primary"

            def translate_request(self, request, *, model=None):
                return {}

            def translate_response(self, response, *, model=None):
                return {}  # type: ignore

            def translate_stream_chunk(self, chunk, *, model=None):
                return None

        t = AliasTranslator()
        registry.register("primary", t, aliases=["alias1", "alias2"])
        assert registry.get("alias1") is t
        assert registry.get("alias2") is t

    def test_overwrite_false_raises(self, registry):
        """Registering a duplicate with overwrite=False raises ValueError."""
        t = OpenAITranslator()
        registry.register("custom", t)
        with pytest.raises(ValueError):
            registry.register("custom", t, overwrite=False)

    def test_overwrite_true_replaces(self, registry):
        """Registering a duplicate with overwrite=True (default) replaces."""
        t1 = OpenAITranslator()
        t2 = AzureOpenAITranslator()
        registry.register("custom", t1)
        registry.register("custom", t2)  # overwrite=True is default
        assert registry.get("custom") is t2

    def test_unregister_removes_translator(self, registry):
        """unregister() removes a registered translator."""
        t = OpenAITranslator()
        registry.register("temp", t)
        assert registry.has("temp")
        removed = registry.unregister("temp")
        assert removed is True
        assert registry.get("temp") is None

    def test_unregister_nonexistent_returns_false(self, registry):
        assert registry.unregister("does-not-exist") is False

    def test_cached_builtins(self, registry):
        """Built-ins are cached after first access."""
        t1 = registry.get("openai")
        t2 = registry.get("openai")
        assert t1 is t2


# ---------------------------------------------------------------------------
# list_providers
# ---------------------------------------------------------------------------


class TestRegistryListProviders:
    def test_includes_all_builtins(self, registry):
        providers = registry.list_providers()
        for name in ("openai", "azure", "anthropic", "google", "gemini", "bedrock"):
            assert name in providers

    def test_includes_custom(self, registry):
        t = OpenAITranslator()
        registry.register("my-custom", t)
        assert "my-custom" in registry.list_providers()

    def test_list_is_sorted(self, registry):
        providers = registry.list_providers()
        assert providers == sorted(providers)

    def test_iter_returns_providers(self, registry):
        providers = list(registry)
        assert "openai" in providers


# ---------------------------------------------------------------------------
# len()
# ---------------------------------------------------------------------------


class TestRegistryLen:
    def test_len_increases_on_access(self, registry):
        initial = len(registry)
        _ = registry.get("openai")
        assert len(registry) >= initial


# ---------------------------------------------------------------------------
# Convenience translation methods
# ---------------------------------------------------------------------------


class TestRegistryConvenienceMethods:
    def test_translate_request(self, registry, simple_request):
        result = registry.translate_request("openai", simple_request)
        assert result["model"] == "gpt-4o"

    def test_translate_response(self, registry):
        response = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1712345678,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        result = registry.translate_response("openai", response)
        assert result["choices"][0]["message"]["content"] == "Hi"

    def test_translate_stream_chunk(self, registry):
        chunk = {
            "id": "x",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "Hey"}}],
        }
        result = registry.translate_stream_chunk("openai", chunk)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hey"

    def test_get_api_path(self, registry):
        path = registry.get_api_path("openai")
        assert path == "/chat/completions"

    def test_get_api_path_gemini_with_model(self, registry):
        path = registry.get_api_path("google", "gemini-2.0-flash")
        assert "gemini-2.0-flash" in path

    def test_get_extra_headers_anthropic(self, registry, simple_request):
        headers = registry.get_extra_headers("anthropic", simple_request)
        assert "anthropic-version" in headers

    def test_translate_request_unknown_provider_raises(self, registry, simple_request):
        with pytest.raises(TranslationError):
            registry.translate_request("nonexistent", simple_request)

    def test_translate_response_unknown_provider_raises(self, registry):
        with pytest.raises(TranslationError):
            registry.translate_response("nonexistent", {})


# ---------------------------------------------------------------------------
# Eager loading
# ---------------------------------------------------------------------------


class TestRegistryEagerLoad:
    def test_eager_load_populates_registry(self):
        reg = TranslatorRegistry(eager_load=True)
        # All built-ins should be pre-loaded
        assert reg.get("openai") is not None
        assert reg.get("anthropic") is not None
        assert reg.get("bedrock") is not None

    def test_eager_load_len(self):
        reg = TranslatorRegistry(eager_load=True)
        # At least 6 built-ins
        assert len(reg) >= 6


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestGlobalRegistry:
    def test_get_registry_returns_instance(self):
        reg = get_registry()
        assert isinstance(reg, TranslatorRegistry)

    def test_get_registry_same_instance(self):
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_reset_creates_new_instance(self):
        reg1 = get_registry()
        reset_registry()
        reg2 = get_registry()
        assert reg1 is not reg2


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


class TestModuleLevelHelpers:
    def test_translate_request_openai(self, simple_request):
        result = translate_request("openai", simple_request)
        assert result["model"] == "gpt-4o"

    def test_translate_request_anthropic(self):
        req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = translate_request("anthropic", req)
        assert "messages" in result
        assert result["messages"][0]["role"] == "user"

    def test_translate_response_openai(self):
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1712345678,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        result = translate_response("openai", response)
        assert result["choices"][0]["message"]["content"] == "Hello!"

    def test_translate_stream_chunk_openai(self):
        chunk = {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "Hi"}}],
        }
        result = translate_stream_chunk("openai", chunk)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hi"

    def test_translate_request_unknown_raises(self, simple_request):
        with pytest.raises(TranslationError):
            translate_request("unknown-provider", simple_request)
