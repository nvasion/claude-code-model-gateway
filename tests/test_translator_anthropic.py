"""Tests for the Anthropic Messages API translator."""

from __future__ import annotations

import json
import pytest

from src.translators.anthropic import AnthropicTranslator
from src.translators.base import TranslationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def translator() -> AnthropicTranslator:
    return AnthropicTranslator()


@pytest.fixture
def simple_request() -> dict:
    return {
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Hello, Claude!"}],
    }


@pytest.fixture
def request_with_system() -> dict:
    return {
        "model": "claude-sonnet-4-20250514",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
    }


@pytest.fixture
def anthropic_response() -> dict:
    return {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "2 + 2 = 4"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 10},
    }


# ---------------------------------------------------------------------------
# translate_request: basic
# ---------------------------------------------------------------------------


class TestAnthropicTranslatorRequest:
    def test_minimal_request(self, translator, simple_request):
        """Simple user message translates correctly."""
        result = translator.translate_request(simple_request)
        assert result["model"] == "claude-sonnet-4-20250514"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello, Claude!"

    def test_model_override(self, translator, simple_request):
        result = translator.translate_request(
            simple_request, model="claude-3-5-sonnet-20241022"
        )
        assert result["model"] == "claude-3-5-sonnet-20241022"

    def test_system_extracted(self, translator, request_with_system):
        """System message is extracted to top-level 'system' field."""
        result = translator.translate_request(request_with_system)
        assert result["system"] == "You are a helpful assistant."
        # System message should NOT appear in messages list
        for msg in result["messages"]:
            assert msg["role"] != "system"

    def test_multiple_system_messages_concatenated(self, translator):
        """Multiple system messages are concatenated."""
        req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = translator.translate_request(req)
        assert "Be helpful." in result["system"]
        assert "Be concise." in result["system"]

    def test_no_system_message(self, translator, simple_request):
        """Request without system message has no 'system' field."""
        result = translator.translate_request(simple_request)
        assert "system" not in result

    def test_max_tokens_preserved(self, translator, simple_request):
        req = dict(simple_request, max_tokens=512)
        result = translator.translate_request(req)
        assert result["max_tokens"] == 512

    def test_default_max_tokens_injected(self, translator, simple_request):
        """When max_tokens is absent, a default is injected (Anthropic requires it)."""
        result = translator.translate_request(simple_request)
        assert "max_tokens" in result
        assert result["max_tokens"] > 0

    def test_custom_default_max_tokens(self, simple_request):
        t = AnthropicTranslator(default_max_tokens=2048)
        result = t.translate_request(simple_request)
        assert result["max_tokens"] == 2048

    def test_temperature_preserved(self, translator, simple_request):
        req = dict(simple_request, temperature=0.5)
        result = translator.translate_request(req)
        assert result["temperature"] == 0.5

    def test_top_p_preserved(self, translator, simple_request):
        req = dict(simple_request, top_p=0.9)
        result = translator.translate_request(req)
        assert result["top_p"] == 0.9

    def test_stop_string_becomes_list(self, translator, simple_request):
        req = dict(simple_request, stop="STOP")
        result = translator.translate_request(req)
        assert result["stop_sequences"] == ["STOP"]

    def test_stop_list_preserved(self, translator, simple_request):
        req = dict(simple_request, stop=["END", "STOP"])
        result = translator.translate_request(req)
        assert result["stop_sequences"] == ["END", "STOP"]

    def test_stream_flag_forwarded(self, translator, simple_request):
        req = dict(simple_request, stream=True)
        result = translator.translate_request(req)
        assert result["stream"] is True

    def test_user_goes_to_metadata(self, translator, simple_request):
        req = dict(simple_request, user="user-42")
        result = translator.translate_request(req)
        assert result.get("metadata", {}).get("user_id") == "user-42"

    def test_empty_messages_raises(self, translator):
        req = {"model": "claude-sonnet-4-20250514", "messages": []}
        with pytest.raises(TranslationError):
            translator.translate_request(req)

    def test_only_system_messages_raises(self, translator):
        req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "system", "content": "Be helpful"}],
        }
        with pytest.raises(TranslationError):
            translator.translate_request(req)


# ---------------------------------------------------------------------------
# translate_request: message roles
# ---------------------------------------------------------------------------


class TestAnthropicTranslatorMessageRoles:
    def test_user_message(self, translator):
        req = {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = translator.translate_request(req)
        assert result["messages"][0]["role"] == "user"

    def test_assistant_message(self, translator):
        req = {
            "model": "x",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Follow-up"},
            ],
        }
        result = translator.translate_request(req)
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["content"] == "Hello!"

    def test_tool_message_becomes_user_with_tool_result(self, translator):
        """Tool messages are converted to user messages with tool_result block."""
        req = {
            "model": "x",
            "messages": [
                {"role": "user", "content": "Use the tool"},
                {
                    "role": "tool",
                    "content": '{"result": 42}',
                    "tool_call_id": "call_abc",
                },
            ],
        }
        result = translator.translate_request(req)
        tool_msg = result["messages"][1]
        assert tool_msg["role"] == "user"
        assert isinstance(tool_msg["content"], list)
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "call_abc"


# ---------------------------------------------------------------------------
# translate_request: multimodal content
# ---------------------------------------------------------------------------


class TestAnthropicTranslatorMultimodal:
    def test_base64_image_url(self, translator):
        req = {
            "model": "x",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgo="
                            },
                        },
                    ],
                }
            ],
        }
        result = translator.translate_request(req)
        parts = result["messages"][0]["content"]
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image"
        assert parts[1]["source"]["type"] == "base64"
        assert parts[1]["source"]["media_type"] == "image/png"
        assert parts[1]["source"]["data"] == "iVBORw0KGgo="

    def test_url_image(self, translator):
        req = {
            "model": "x",
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
        part = result["messages"][0]["content"][0]
        assert part["type"] == "image"
        assert part["source"]["type"] == "url"
        assert part["source"]["url"] == "https://example.com/img.jpg"


# ---------------------------------------------------------------------------
# translate_request: tools
# ---------------------------------------------------------------------------


class TestAnthropicTranslatorTools:
    def test_tool_conversion(self, translator, simple_request):
        req = dict(
            simple_request,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        )
        result = translator.translate_request(req)
        assert "tools" in result
        tool = result["tools"][0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather"
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"

    def test_tool_choice_auto(self, translator, simple_request):
        req = dict(simple_request, tool_choice="auto")
        result = translator.translate_request(req)
        assert result["tool_choice"] == {"type": "auto"}

    def test_tool_choice_none(self, translator, simple_request):
        req = dict(simple_request, tool_choice="none")
        result = translator.translate_request(req)
        assert result["tool_choice"] == {"type": "none"}

    def test_tool_choice_required(self, translator, simple_request):
        req = dict(simple_request, tool_choice="required")
        result = translator.translate_request(req)
        assert result["tool_choice"] == {"type": "any"}

    def test_tool_choice_specific_function(self, translator, simple_request):
        req = dict(
            simple_request,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        result = translator.translate_request(req)
        assert result["tool_choice"] == {"type": "tool", "name": "get_weather"}


# ---------------------------------------------------------------------------
# translate_request: assistant with tool calls
# ---------------------------------------------------------------------------


class TestAnthropicTranslatorAssistantToolCalls:
    def test_assistant_message_with_tool_calls(self, translator):
        req = {
            "model": "x",
            "messages": [
                {"role": "user", "content": "Search for Python"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "Python"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "Python is a programming language.",
                    "tool_call_id": "call_abc",
                },
            ],
        }
        result = translator.translate_request(req)
        assert result["messages"][1]["role"] == "assistant"
        content = result["messages"][1]["content"]
        assert isinstance(content, list)
        tool_use_block = next(b for b in content if b["type"] == "tool_use")
        assert tool_use_block["name"] == "search"
        assert tool_use_block["input"] == {"query": "Python"}

    def test_assistant_with_text_and_tool_calls(self, translator):
        req = {
            "model": "x",
            "messages": [
                {"role": "user", "content": "Help me"},
                {
                    "role": "assistant",
                    "content": "I'll search for you.",
                    "tool_calls": [
                        {
                            "id": "call_xyz",
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
                    "content": "Results found",
                    "tool_call_id": "call_xyz",
                },
            ],
        }
        result = translator.translate_request(req)
        assistant_content = result["messages"][1]["content"]
        # Should have text + tool_use blocks
        assert any(b["type"] == "text" for b in assistant_content)
        assert any(b["type"] == "tool_use" for b in assistant_content)


# ---------------------------------------------------------------------------
# translate_response
# ---------------------------------------------------------------------------


class TestAnthropicTranslatorResponse:
    def test_basic_response(self, translator, anthropic_response):
        result = translator.translate_response(anthropic_response)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "2 + 2 = 4"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_usage_mapping(self, translator, anthropic_response):
        result = translator.translate_response(anthropic_response)
        usage = result["usage"]
        assert usage["prompt_tokens"] == 20
        assert usage["completion_tokens"] == 10
        assert usage["total_tokens"] == 30

    def test_stop_reason_mapping(self, translator, anthropic_response):
        """end_turn → stop."""
        result = translator.translate_response(anthropic_response)
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_max_tokens_stop_reason(self, translator, anthropic_response):
        resp = dict(anthropic_response, stop_reason="max_tokens")
        result = translator.translate_response(resp)
        assert result["choices"][0]["finish_reason"] == "length"

    def test_tool_use_stop_reason(self, translator, anthropic_response):
        resp = dict(anthropic_response, stop_reason="tool_use")
        result = translator.translate_response(resp)
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_response_with_tool_use_blocks(self, translator):
        resp = {
            "id": "msg_abc",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"location": "NYC"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 25, "output_tokens": 12},
        }
        result = translator.translate_response(resp)
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "NYC"}

    def test_response_with_text_and_tool_use(self, translator):
        resp = {
            "id": "msg_abc",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "I'll look that up."},
                {
                    "type": "tool_use",
                    "id": "toolu_02",
                    "name": "search",
                    "input": {"query": "weather"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 20},
        }
        result = translator.translate_response(resp)
        msg = result["choices"][0]["message"]
        assert msg["content"] == "I'll look that up."
        assert len(msg["tool_calls"]) == 1

    def test_response_model_override(self, translator, anthropic_response):
        resp = dict(anthropic_response)
        del resp["model"]
        result = translator.translate_response(resp, model="claude-3-5-sonnet-20241022")
        assert result["model"] == "claude-3-5-sonnet-20241022"

    def test_invalid_response_raises(self, translator):
        with pytest.raises(TranslationError):
            translator.translate_response("bad input")  # type: ignore

    def test_response_has_id(self, translator, anthropic_response):
        result = translator.translate_response(anthropic_response)
        assert result["id"] == anthropic_response["id"]


# ---------------------------------------------------------------------------
# translate_stream_chunk
# ---------------------------------------------------------------------------


class TestAnthropicTranslatorStreamChunk:
    def test_message_start_event(self, translator):
        chunk = {"type": "message_start", "message": {"id": "msg_x"}}
        result = translator.translate_stream_chunk(chunk)
        assert result is not None
        assert result["choices"][0]["delta"]["role"] == "assistant"

    def test_text_delta_event(self, translator):
        chunk = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        result = translator.translate_stream_chunk(chunk)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_tool_arg_delta_event(self, translator):
        chunk = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"loc'},
        }
        result = translator.translate_stream_chunk(chunk)
        assert result is not None
        tool_calls = result["choices"][0]["delta"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["arguments"] == '{"loc'

    def test_content_block_start_tool_use(self, translator):
        chunk = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "search",
            },
        }
        result = translator.translate_stream_chunk(chunk)
        assert result is not None
        tool_calls = result["choices"][0]["delta"]["tool_calls"]
        assert tool_calls[0]["id"] == "toolu_abc"
        assert tool_calls[0]["function"]["name"] == "search"

    def test_message_delta_with_stop_reason(self, translator):
        chunk = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = translator.translate_stream_chunk(chunk)
        assert result is not None
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_message_stop_event_dropped(self, translator):
        chunk = {"type": "message_stop"}
        result = translator.translate_stream_chunk(chunk)
        assert result is None

    def test_ping_event_dropped(self, translator):
        chunk = {"type": "ping"}
        result = translator.translate_stream_chunk(chunk)
        assert result is None

    def test_empty_chunk_dropped(self, translator):
        result = translator.translate_stream_chunk({})
        assert result is None

    def test_content_block_stop_dropped(self, translator):
        chunk = {"type": "content_block_stop", "index": 0}
        result = translator.translate_stream_chunk(chunk)
        assert result is None


# ---------------------------------------------------------------------------
# Extra headers
# ---------------------------------------------------------------------------


class TestAnthropicTranslatorHeaders:
    def test_anthropic_version_header(self, translator, simple_request):
        headers = translator.get_extra_headers(simple_request)
        assert headers["anthropic-version"] == "2023-06-01"

    def test_custom_anthropic_version(self, simple_request):
        t = AnthropicTranslator(anthropic_version="2024-01-01")
        headers = t.get_extra_headers(simple_request)
        assert headers["anthropic-version"] == "2024-01-01"

    def test_api_path(self, translator):
        assert translator.get_api_path() == "/messages"
