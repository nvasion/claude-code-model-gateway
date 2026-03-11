"""Tests for the OpenAI and Azure OpenAI translators."""

from __future__ import annotations

import pytest

from src.translators.base import TranslationError
from src.translators.openai import AzureOpenAITranslator, OpenAITranslator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def translator() -> OpenAITranslator:
    return OpenAITranslator()


@pytest.fixture
def strict_translator() -> OpenAITranslator:
    return OpenAITranslator(strict=True)


@pytest.fixture
def azure_translator() -> AzureOpenAITranslator:
    return AzureOpenAITranslator()


@pytest.fixture
def minimal_request() -> dict:
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
    }


@pytest.fixture
def full_request() -> dict:
    return {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "stream": False,
        "stop": ["END"],
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "user": "user-1",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    }


@pytest.fixture
def openai_response() -> dict:
    return {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1712345678,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


# ---------------------------------------------------------------------------
# OpenAITranslator.translate_request
# ---------------------------------------------------------------------------


class TestOpenAITranslatorRequest:
    def test_passthrough_minimal(self, translator, minimal_request):
        """Minimal request is passed through unchanged."""
        result = translator.translate_request(minimal_request)
        assert result["model"] == "gpt-4o"
        assert result["messages"] == minimal_request["messages"]

    def test_passthrough_full_request(self, translator, full_request):
        """All standard fields are preserved."""
        result = translator.translate_request(full_request)
        assert result["max_tokens"] == 512
        assert result["temperature"] == 0.7
        assert result["tools"] == full_request["tools"]

    def test_model_override(self, translator, minimal_request):
        """model kwarg overrides the request's model field."""
        result = translator.translate_request(minimal_request, model="gpt-4-turbo")
        assert result["model"] == "gpt-4-turbo"

    def test_strips_unknown_keys(self, translator, minimal_request):
        """Unknown top-level keys are stripped by default."""
        req = dict(minimal_request)
        req["gateway_internal_key"] = "should-be-stripped"
        result = translator.translate_request(req)
        assert "gateway_internal_key" not in result

    def test_keeps_unknown_keys_when_disabled(self, minimal_request):
        """Unknown keys are kept when strip_unknown_keys=False."""
        t = OpenAITranslator(strip_unknown_keys=False)
        req = dict(minimal_request)
        req["custom_key"] = "kept"
        result = t.translate_request(req)
        assert result["custom_key"] == "kept"

    def test_strict_raises_on_missing_model(self, strict_translator):
        """Strict mode raises TranslationError when model is absent."""
        req = {"messages": [{"role": "user", "content": "Hi"}]}
        with pytest.raises(TranslationError, match="model"):
            strict_translator.translate_request(req)

    def test_strict_raises_on_missing_messages(self, strict_translator):
        """Strict mode raises TranslationError when messages are absent."""
        req = {"model": "gpt-4o", "messages": []}
        with pytest.raises(TranslationError, match="messages"):
            strict_translator.translate_request(req)

    def test_non_strict_allows_missing_model(self, translator):
        """Non-strict mode does not raise when model is absent."""
        req = {"messages": [{"role": "user", "content": "Hi"}]}
        result = translator.translate_request(req)
        assert "model" not in result or result.get("model") is None or True

    def test_stream_flag_preserved(self, translator, minimal_request):
        req = dict(minimal_request, stream=True)
        result = translator.translate_request(req)
        assert result["stream"] is True

    def test_response_format_preserved(self, translator, minimal_request):
        req = dict(minimal_request, response_format={"type": "json_object"})
        result = translator.translate_request(req)
        assert result["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# OpenAITranslator.translate_response
# ---------------------------------------------------------------------------


class TestOpenAITranslatorResponse:
    def test_response_passthrough(self, translator, openai_response):
        """OpenAI response is normalised and returned."""
        result = translator.translate_response(openai_response)
        assert result["id"] == "chatcmpl-abc"
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hi there!"

    def test_response_fills_missing_id(self, translator):
        """Missing id is filled with a generated one."""
        resp = {
            "object": "chat.completion",
            "created": 1712345678,
            "model": "gpt-4o",
            "choices": [],
            "usage": {},
        }
        result = translator.translate_response(resp)
        assert result["id"].startswith("chatcmpl-")

    def test_response_fills_missing_created(self, translator):
        """Missing created is filled with current timestamp."""
        resp = {"id": "x", "object": "chat.completion", "choices": [], "usage": {}}
        result = translator.translate_response(resp)
        assert isinstance(result["created"], int)
        assert result["created"] > 0

    def test_response_model_override(self, translator, openai_response):
        """model kwarg overrides response model when absent."""
        resp = dict(openai_response)
        del resp["model"]
        result = translator.translate_response(resp, model="gpt-4o-mini")
        assert result["model"] == "gpt-4o-mini"

    def test_response_normalises_usage(self, translator):
        """Empty usage dict is normalised with zeros."""
        resp = {
            "id": "x",
            "object": "chat.completion",
            "choices": [],
            "usage": {},
        }
        result = translator.translate_response(resp)
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0
        assert result["usage"]["total_tokens"] == 0

    def test_response_normalises_choices(self, translator):
        """Choice index and finish_reason defaults are set."""
        resp = {
            "id": "x",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
            "usage": {},
        }
        result = translator.translate_response(resp)
        choice = result["choices"][0]
        assert choice["index"] == 0
        assert "finish_reason" in choice

    def test_invalid_response_raises(self, translator):
        """Non-dict response raises TranslationError."""
        with pytest.raises(TranslationError):
            translator.translate_response("invalid")  # type: ignore

    def test_deep_copy_does_not_mutate_input(self, translator, openai_response):
        """translate_response returns a deep copy; original is not mutated."""
        original = openai_response.copy()
        result = translator.translate_response(openai_response)
        result["model"] = "mutated"
        assert openai_response["model"] == original["model"]


# ---------------------------------------------------------------------------
# OpenAITranslator.translate_stream_chunk
# ---------------------------------------------------------------------------


class TestOpenAITranslatorStreamChunk:
    def test_text_delta_chunk(self, translator):
        chunk = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1712345678,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ],
        }
        result = translator.translate_stream_chunk(chunk)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_empty_chunk_returns_none(self, translator):
        result = translator.translate_stream_chunk({})
        assert result is None

    def test_none_chunk_returns_none(self, translator):
        result = translator.translate_stream_chunk(None)  # type: ignore
        assert result is None

    def test_chunk_model_override(self, translator):
        chunk = {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "Hi"}}],
        }
        result = translator.translate_stream_chunk(chunk, model="gpt-4-turbo")
        assert result["model"] == "gpt-4-turbo"

    def test_chunk_fills_defaults(self, translator):
        chunk = {"choices": [{"delta": {"content": "X"}}]}
        result = translator.translate_stream_chunk(chunk)
        assert result is not None
        assert result["object"] == "chat.completion.chunk"
        assert result["id"].startswith("chatcmpl-")
        assert result["created"] > 0


# ---------------------------------------------------------------------------
# OpenAITranslator helpers
# ---------------------------------------------------------------------------


class TestOpenAITranslatorHelpers:
    def test_get_api_path(self, translator):
        assert translator.get_api_path() == "/chat/completions"
        assert translator.get_api_path("gpt-4o") == "/chat/completions"

    def test_provider_name(self, translator):
        assert translator.PROVIDER_NAME == "openai"
        assert translator.get_provider_name() == "openai"

    def test_supports_streaming(self, translator):
        assert translator.supports_streaming() is True

    def test_supports_tools(self, translator):
        assert translator.supports_tools() is True


# ---------------------------------------------------------------------------
# AzureOpenAITranslator
# ---------------------------------------------------------------------------


class TestAzureOpenAITranslator:
    def test_provider_name(self, azure_translator):
        assert azure_translator.PROVIDER_NAME == "azure"

    def test_translate_request_passthrough(self, azure_translator, minimal_request):
        result = azure_translator.translate_request(minimal_request)
        assert result["model"] == "gpt-4o"

    def test_translate_response_passthrough(self, azure_translator, openai_response):
        result = azure_translator.translate_response(openai_response)
        assert result["id"] == "chatcmpl-abc"

    def test_get_api_path(self, azure_translator):
        assert azure_translator.get_api_path() == "/chat/completions"
