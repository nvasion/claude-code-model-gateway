"""Tests for the AWS Bedrock translator."""

from __future__ import annotations

import json
import pytest

from src.translators.base import TranslationError
from src.translators.bedrock import BedrockTranslator, _detect_family


# ---------------------------------------------------------------------------
# _detect_family helper
# ---------------------------------------------------------------------------


class TestDetectFamily:
    def test_anthropic_prefix(self):
        assert _detect_family("anthropic.claude-sonnet-4-20250514-v1:0") == "anthropic"

    def test_titan_prefix(self):
        assert _detect_family("amazon.titan-text-express-v1") == "titan"

    def test_llama_prefix(self):
        assert _detect_family("meta.llama2-70b-chat-v1") == "llama"

    def test_cohere_prefix(self):
        assert _detect_family("cohere.command-r-v1:0") == "cohere"

    def test_ai21_prefix(self):
        assert _detect_family("ai21.j2-ultra-v1") == "ai21"

    def test_mistral_prefix(self):
        assert _detect_family("mistral.mistral-7b-instruct-v0:2") == "mistral"

    def test_unknown(self):
        assert _detect_family("unknown-model") == "unknown"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def translator() -> BedrockTranslator:
    return BedrockTranslator()


@pytest.fixture
def claude_request() -> dict:
    return {
        "model": "anthropic.claude-sonnet-4-20250514-v1:0",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 256,
    }


@pytest.fixture
def titan_request() -> dict:
    return {
        "model": "amazon.titan-text-express-v1",
        "messages": [{"role": "user", "content": "Hello Titan!"}],
        "max_tokens": 200,
        "temperature": 0.7,
    }


@pytest.fixture
def llama_request() -> dict:
    return {
        "model": "meta.llama2-70b-chat-v1",
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "max_tokens": 100,
    }


# ---------------------------------------------------------------------------
# translate_request: Anthropic family
# ---------------------------------------------------------------------------


class TestBedrockTranslatorClaudeRequest:
    def test_anthropic_version_in_body(self, translator, claude_request):
        """Bedrock Claude requires anthropic_version in the body."""
        result = translator.translate_request(claude_request)
        assert result.get("anthropic_version") == "bedrock-2023-05-31"

    def test_custom_anthropic_version(self, claude_request):
        t = BedrockTranslator(anthropic_version="bedrock-custom-v1")
        result = t.translate_request(claude_request)
        assert result["anthropic_version"] == "bedrock-custom-v1"

    def test_stream_flag_removed(self, translator, claude_request):
        """stream flag must not appear in Bedrock body (path controls streaming)."""
        req = dict(claude_request, stream=True)
        result = translator.translate_request(req)
        assert "stream" not in result

    def test_messages_translated(self, translator, claude_request):
        result = translator.translate_request(claude_request)
        assert "messages" in result
        assert result["messages"][0]["role"] == "user"

    def test_max_tokens_in_body(self, translator, claude_request):
        result = translator.translate_request(claude_request)
        assert result["max_tokens"] == 256

    def test_system_extracted(self, translator):
        req = {
            "model": "anthropic.claude-sonnet-4-20250514-v1:0",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = translator.translate_request(req)
        assert result["system"] == "Be concise."

    def test_model_override(self, translator, claude_request):
        result = translator.translate_request(
            claude_request, model="anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        assert result["model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"


# ---------------------------------------------------------------------------
# translate_request: Titan family
# ---------------------------------------------------------------------------


class TestBedrockTranslatorTitanRequest:
    def test_input_text_field(self, translator, titan_request):
        result = translator.translate_request(titan_request)
        assert "inputText" in result
        assert "Hello Titan!" in result["inputText"]

    def test_generation_config(self, translator, titan_request):
        result = translator.translate_request(titan_request)
        config = result.get("textGenerationConfig", {})
        assert config.get("maxTokenCount") == 200
        assert config.get("temperature") == 0.7

    def test_system_included_in_prompt(self, translator):
        req = {
            "model": "amazon.titan-text-express-v1",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = translator.translate_request(req)
        assert "System:" in result["inputText"]
        assert "You are helpful." in result["inputText"]


# ---------------------------------------------------------------------------
# translate_request: Llama family
# ---------------------------------------------------------------------------


class TestBedrockTranslatorLlamaRequest:
    def test_prompt_field(self, translator, llama_request):
        result = translator.translate_request(llama_request)
        assert "prompt" in result
        assert "France" in result["prompt"]

    def test_max_gen_len(self, translator, llama_request):
        result = translator.translate_request(llama_request)
        assert result.get("max_gen_len") == 100

    def test_system_included(self, translator, llama_request):
        result = translator.translate_request(llama_request)
        assert "Be helpful." in result["prompt"]


# ---------------------------------------------------------------------------
# translate_response: Anthropic family
# ---------------------------------------------------------------------------


class TestBedrockTranslatorClaudeResponse:
    def test_basic_response(self, translator):
        model = "anthropic.claude-sonnet-4-20250514-v1:0"
        resp = {
            "id": "msg_abc",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "Hello from Bedrock!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = translator.translate_response(resp, model=model)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello from Bedrock!"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_tool_use_response(self, translator):
        model = "anthropic.claude-sonnet-4-20250514-v1:0"
        resp = {
            "id": "msg_xyz",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "search",
                    "input": {"query": "Paris"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 15, "output_tokens": 8},
        }
        result = translator.translate_response(resp, model=model)
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# translate_response: Titan family
# ---------------------------------------------------------------------------


class TestBedrockTranslatorTitanResponse:
    def test_basic_titan_response(self, translator):
        model = "amazon.titan-text-express-v1"
        resp = {
            "inputTextTokenCount": 10,
            "results": [
                {
                    "tokenCount": 15,
                    "outputText": "Paris is the capital of France.",
                    "completionReason": "FINISH",
                }
            ],
        }
        result = translator.translate_response(resp, model=model)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Paris is the capital of France."
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 15

    def test_length_stop_reason(self, translator):
        model = "amazon.titan-text-express-v1"
        resp = {
            "inputTextTokenCount": 5,
            "results": [
                {"tokenCount": 20, "outputText": "...", "completionReason": "LENGTH"}
            ],
        }
        result = translator.translate_response(resp, model=model)
        assert result["choices"][0]["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# translate_response: Llama family
# ---------------------------------------------------------------------------


class TestBedrockTranslatorLlamaResponse:
    def test_basic_llama_response(self, translator):
        model = "meta.llama2-70b-chat-v1"
        resp = {
            "generation": "Paris is the capital of France.",
            "prompt_token_count": 25,
            "generation_token_count": 10,
            "stop_reason": "stop",
        }
        result = translator.translate_response(resp, model=model)
        assert result["choices"][0]["message"]["content"] == "Paris is the capital of France."
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 25

    def test_eos_token_stop_reason(self, translator):
        model = "meta.llama2-70b-chat-v1"
        resp = {
            "generation": "Done",
            "prompt_token_count": 5,
            "generation_token_count": 2,
            "stop_reason": "eos_token",
        }
        result = translator.translate_response(resp, model=model)
        assert result["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# translate_response: generic fallback
# ---------------------------------------------------------------------------


class TestBedrockTranslatorGenericResponse:
    def test_generic_fallback_text(self, translator):
        resp = {"text": "Some output"}
        result = translator.translate_response(resp, model="custom.unknown-model")
        assert result["choices"][0]["message"]["content"] == "Some output"

    def test_generic_fallback_completion(self, translator):
        resp = {"completion": "Generated text"}
        result = translator.translate_response(resp, model="custom.unknown")
        assert result["choices"][0]["message"]["content"] == "Generated text"

    def test_invalid_response_raises(self, translator):
        with pytest.raises(TranslationError):
            translator.translate_response("not a dict")  # type: ignore


# ---------------------------------------------------------------------------
# translate_stream_chunk
# ---------------------------------------------------------------------------


class TestBedrockTranslatorStreamChunk:
    def test_claude_text_chunk(self, translator):
        model = "anthropic.claude-sonnet-4-20250514-v1:0"
        chunk = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        result = translator.translate_stream_chunk(chunk, model=model)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_titan_text_chunk(self, translator):
        model = "amazon.titan-text-express-v1"
        chunk = {"outputText": "Hello from Titan", "index": "0"}
        result = translator.translate_stream_chunk(chunk, model=model)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hello from Titan"

    def test_titan_final_chunk_with_stop_reason(self, translator):
        model = "amazon.titan-text-express-v1"
        chunk = {
            "outputText": "",
            "index": "0",
            "completionReason": "FINISH",
        }
        result = translator.translate_stream_chunk(chunk, model=model)
        assert result is not None
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_empty_chunk_returns_none(self, translator):
        result = translator.translate_stream_chunk({})
        assert result is None

    def test_none_chunk_returns_none(self, translator):
        result = translator.translate_stream_chunk(None)  # type: ignore
        assert result is None


# ---------------------------------------------------------------------------
# API path
# ---------------------------------------------------------------------------


class TestBedrockTranslatorApiPath:
    def test_invoke_path(self, translator):
        path = translator.get_api_path("anthropic.claude-sonnet-4-20250514-v1:0")
        assert path == "/model/anthropic.claude-sonnet-4-20250514-v1:0/invoke"

    def test_streaming_path(self, translator):
        path = translator.get_streaming_api_path("amazon.titan-text-express-v1")
        assert (
            path
            == "/model/amazon.titan-text-express-v1/invoke-with-response-stream"
        )

    def test_default_path_no_model(self, translator):
        path = translator.get_api_path()
        assert path == "/model/invoke"

    def test_provider_name(self, translator):
        assert translator.PROVIDER_NAME == "bedrock"
