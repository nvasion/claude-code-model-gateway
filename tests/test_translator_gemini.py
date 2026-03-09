"""Tests for the Google Gemini translator."""

from __future__ import annotations

import json
import pytest

from src.translators.base import TranslationError
from src.translators.gemini import GeminiTranslator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def translator() -> GeminiTranslator:
    return GeminiTranslator()


@pytest.fixture
def simple_request() -> dict:
    return {
        "model": "gemini-2.0-flash",
        "messages": [{"role": "user", "content": "Hello, Gemini!"}],
    }


@pytest.fixture
def gemini_response() -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello! How can I help?"}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18,
        },
    }


# ---------------------------------------------------------------------------
# translate_request: basic
# ---------------------------------------------------------------------------


class TestGeminiTranslatorRequest:
    def test_minimal_request(self, translator, simple_request):
        result = translator.translate_request(simple_request)
        assert "contents" in result
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"][0]["text"] == "Hello, Gemini!"

    def test_user_role_preserved(self, translator, simple_request):
        result = translator.translate_request(simple_request)
        assert result["contents"][0]["role"] == "user"

    def test_assistant_role_becomes_model(self, translator):
        req = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        result = translator.translate_request(req)
        assert result["contents"][1]["role"] == "model"

    def test_system_extracted_to_system_instruction(self, translator):
        req = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Tell me a joke"},
            ],
        }
        result = translator.translate_request(req)
        assert "systemInstruction" in result
        assert result["systemInstruction"]["parts"][0]["text"] == "Be concise."
        # System message should not appear in contents
        for content in result["contents"]:
            assert content["role"] != "system"

    def test_no_system_no_system_instruction(self, translator, simple_request):
        result = translator.translate_request(simple_request)
        assert "systemInstruction" not in result

    def test_empty_messages_raises(self, translator):
        req = {"model": "gemini-2.0-flash", "messages": []}
        with pytest.raises(TranslationError):
            translator.translate_request(req)

    def test_only_system_messages_raises(self, translator):
        req = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "system", "content": "Be helpful"}],
        }
        with pytest.raises(TranslationError):
            translator.translate_request(req)

    def test_generation_config_max_tokens(self, translator, simple_request):
        req = dict(simple_request, max_tokens=512)
        result = translator.translate_request(req)
        assert result["generationConfig"]["maxOutputTokens"] == 512

    def test_generation_config_temperature(self, translator, simple_request):
        req = dict(simple_request, temperature=0.5)
        result = translator.translate_request(req)
        assert result["generationConfig"]["temperature"] == 0.5

    def test_generation_config_top_p(self, translator, simple_request):
        req = dict(simple_request, top_p=0.9)
        result = translator.translate_request(req)
        assert result["generationConfig"]["topP"] == 0.9

    def test_generation_config_candidate_count(self, translator, simple_request):
        req = dict(simple_request, n=2)
        result = translator.translate_request(req)
        assert result["generationConfig"]["candidateCount"] == 2

    def test_generation_config_seed(self, translator, simple_request):
        req = dict(simple_request, seed=42)
        result = translator.translate_request(req)
        assert result["generationConfig"]["seed"] == 42

    def test_stop_string_becomes_list(self, translator, simple_request):
        req = dict(simple_request, stop="END")
        result = translator.translate_request(req)
        assert result["generationConfig"]["stopSequences"] == ["END"]

    def test_stop_list_preserved(self, translator, simple_request):
        req = dict(simple_request, stop=["END", "STOP"])
        result = translator.translate_request(req)
        assert result["generationConfig"]["stopSequences"] == ["END", "STOP"]

    def test_json_response_format(self, translator, simple_request):
        req = dict(simple_request, response_format={"type": "json_object"})
        result = translator.translate_request(req)
        assert result["generationConfig"]["responseMimeType"] == "application/json"

    def test_no_gen_config_when_no_params(self, translator, simple_request):
        """No generationConfig emitted if no sampling params set."""
        result = translator.translate_request(simple_request)
        assert "generationConfig" not in result


# ---------------------------------------------------------------------------
# translate_request: tools
# ---------------------------------------------------------------------------


class TestGeminiTranslatorTools:
    def test_tools_conversion(self, translator, simple_request):
        req = dict(
            simple_request,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
        )
        result = translator.translate_request(req)
        assert "tools" in result
        decls = result["tools"][0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "search"
        assert decls[0]["description"] == "Search the web"
        assert "parameters" in decls[0]

    def test_tool_choice_none(self, translator, simple_request):
        req = dict(simple_request, tool_choice="none")
        result = translator.translate_request(req)
        assert result["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"

    def test_tool_choice_auto(self, translator, simple_request):
        req = dict(simple_request, tool_choice="auto")
        result = translator.translate_request(req)
        assert result["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"

    def test_tool_choice_required(self, translator, simple_request):
        req = dict(simple_request, tool_choice="required")
        result = translator.translate_request(req)
        assert result["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"

    def test_tool_choice_specific_function(self, translator, simple_request):
        req = dict(
            simple_request,
            tool_choice={"type": "function", "function": {"name": "search"}},
        )
        result = translator.translate_request(req)
        config = result["toolConfig"]["functionCallingConfig"]
        assert config["mode"] == "ANY"
        assert "search" in config["allowedFunctionNames"]


# ---------------------------------------------------------------------------
# translate_request: multimodal
# ---------------------------------------------------------------------------


class TestGeminiTranslatorMultimodal:
    def test_base64_image(self, translator):
        req = {
            "model": "gemini-2.0-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/abc="
                            },
                        },
                    ],
                }
            ],
        }
        result = translator.translate_request(req)
        parts = result["contents"][0]["parts"]
        assert parts[0]["text"] == "Describe"
        assert "inlineData" in parts[1]
        assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"

    def test_url_image(self, translator):
        req = {
            "model": "gemini-2.0-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/img.jpg"},
                        }
                    ],
                }
            ],
        }
        result = translator.translate_request(req)
        part = result["contents"][0]["parts"][0]
        assert "fileData" in part
        assert part["fileData"]["fileUri"] == "https://example.com/img.jpg"


# ---------------------------------------------------------------------------
# translate_request: tool messages
# ---------------------------------------------------------------------------


class TestGeminiTranslatorToolMessages:
    def test_tool_message_becomes_function_response(self, translator):
        req = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": "Use search"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"q": "test"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"result": "found"}',
                    "tool_call_id": "search",
                },
            ],
        }
        result = translator.translate_request(req)
        tool_msg = result["contents"][2]
        assert tool_msg["role"] == "user"
        assert "functionResponse" in tool_msg["parts"][0]


# ---------------------------------------------------------------------------
# translate_response
# ---------------------------------------------------------------------------


class TestGeminiTranslatorResponse:
    def test_basic_response(self, translator, gemini_response):
        result = translator.translate_response(gemini_response)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello! How can I help?"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_usage_mapping(self, translator, gemini_response):
        result = translator.translate_response(gemini_response)
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 8
        assert result["usage"]["total_tokens"] == 18

    def test_finish_reason_stop(self, translator, gemini_response):
        result = translator.translate_response(gemini_response)
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_finish_reason_max_tokens(self, translator):
        resp = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "..."}], "role": "model"},
                    "finishReason": "MAX_TOKENS",
                }
            ],
            "usageMetadata": {},
        }
        result = translator.translate_response(resp)
        assert result["choices"][0]["finish_reason"] == "length"

    def test_finish_reason_safety(self, translator):
        resp = {
            "candidates": [
                {
                    "content": {"parts": [{"text": ""}], "role": "model"},
                    "finishReason": "SAFETY",
                }
            ],
            "usageMetadata": {},
        }
        result = translator.translate_response(resp)
        assert result["choices"][0]["finish_reason"] == "content_filter"

    def test_response_with_function_call(self, translator):
        resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "NYC"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {},
        }
        result = translator.translate_response(resp)
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "NYC"}

    def test_multiple_candidates(self, translator):
        resp = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Answer 1"}], "role": "model"},
                    "finishReason": "STOP",
                },
                {
                    "content": {"parts": [{"text": "Answer 2"}], "role": "model"},
                    "finishReason": "STOP",
                },
            ],
            "usageMetadata": {},
        }
        result = translator.translate_response(resp)
        assert len(result["choices"]) == 2
        assert result["choices"][0]["message"]["content"] == "Answer 1"
        assert result["choices"][1]["message"]["content"] == "Answer 2"

    def test_model_override(self, translator, gemini_response):
        result = translator.translate_response(gemini_response, model="gemini-1.5-pro")
        assert result["model"] == "gemini-1.5-pro"

    def test_invalid_response_raises(self, translator):
        with pytest.raises(TranslationError):
            translator.translate_response("bad")  # type: ignore

    def test_empty_candidates(self, translator):
        resp = {"candidates": [], "usageMetadata": {}}
        result = translator.translate_response(resp)
        assert result["choices"] == []


# ---------------------------------------------------------------------------
# translate_stream_chunk
# ---------------------------------------------------------------------------


class TestGeminiTranslatorStreamChunk:
    def test_text_chunk(self, translator):
        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello"}],
                        "role": "model",
                    },
                    "finishReason": None,
                }
            ]
        }
        result = translator.translate_stream_chunk(chunk)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_final_chunk_with_finish_reason(self, translator):
        chunk = {
            "candidates": [
                {
                    "content": {"parts": [{"text": ""}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8,
            },
        }
        result = translator.translate_stream_chunk(chunk)
        assert result is not None
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 8

    def test_empty_chunk_returns_none(self, translator):
        result = translator.translate_stream_chunk({})
        assert result is None

    def test_none_chunk_returns_none(self, translator):
        result = translator.translate_stream_chunk(None)  # type: ignore
        assert result is None

    def test_chunk_without_candidates_returns_none(self, translator):
        chunk = {"usageMetadata": {}}
        result = translator.translate_stream_chunk(chunk)
        assert result is None


# ---------------------------------------------------------------------------
# API path
# ---------------------------------------------------------------------------


class TestGeminiTranslatorApiPath:
    def test_get_api_path_with_model(self, translator):
        path = translator.get_api_path("gemini-2.0-flash")
        assert path == "/models/gemini-2.0-flash:generateContent"

    def test_get_api_path_default(self, translator):
        path = translator.get_api_path()
        assert ":generateContent" in path

    def test_get_streaming_api_path(self, translator):
        path = translator.get_streaming_api_path("gemini-1.5-pro")
        assert path == "/models/gemini-1.5-pro:streamGenerateContent"

    def test_provider_name(self, translator):
        assert translator.PROVIDER_NAME == "google"
