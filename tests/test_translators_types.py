"""Tests for the canonical type definitions in src.translators.types."""

from __future__ import annotations

import pytest

from src.translators.types import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamChunk,
    Choice,
    FinishReason,
    FunctionDefinition,
    FunctionParameters,
    ImageUrl,
    ImageUrlContentPart,
    ResponseMessage,
    StreamChoice,
    StreamDelta,
    TextContentPart,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolMessage,
    UsageInfo,
)


class TestCanonicalRequest:
    """Tests for CanonicalRequest structure."""

    def test_minimal_request(self):
        """CanonicalRequest can be created with only required fields."""
        req: CanonicalRequest = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        assert req["model"] == "gpt-4o"
        assert len(req["messages"]) == 1

    def test_full_request(self):
        """CanonicalRequest can contain all optional fields."""
        req: CanonicalRequest = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stream": False,
            "stop": ["END", "STOP"],
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "user": "user-123",
            "seed": 42,
            "tools": [],
            "tool_choice": "auto",
            "response_format": {"type": "text"},
        }
        assert req["max_tokens"] == 1024
        assert req["temperature"] == 0.7
        assert req["stream"] is False

    def test_request_with_string_stop(self):
        """stop can be a string."""
        req: CanonicalRequest = {
            "model": "gpt-4o",
            "messages": [],
            "stop": "END",
        }
        assert req["stop"] == "END"

    def test_request_with_tool_choice_dict(self):
        """tool_choice can be a dict specifying a function."""
        req: CanonicalRequest = {
            "model": "gpt-4o",
            "messages": [],
            "tool_choice": {"type": "function", "function": {"name": "search"}},
        }
        assert isinstance(req["tool_choice"], dict)

    def test_request_with_multimodal_content(self):
        """Messages can contain multimodal content."""
        req: CanonicalRequest = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/img.png"},
                        },
                    ],
                }
            ],
        }
        msg = req["messages"][0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"


class TestCanonicalResponse:
    """Tests for CanonicalResponse structure."""

    def test_minimal_response(self):
        """CanonicalResponse with required fields."""
        resp: CanonicalResponse = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1712345678,
            "model": "gpt-4o",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        assert resp["id"] == "chatcmpl-abc"
        assert resp["object"] == "chat.completion"

    def test_response_with_choices(self):
        """CanonicalResponse can have multiple choices."""
        resp: CanonicalResponse = {
            "id": "chatcmpl-abc",
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
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert resp["choices"][0]["message"]["content"] == "Hello!"

    def test_response_with_tool_calls(self):
        """CanonicalResponse can include tool_calls in the message."""
        resp: CanonicalResponse = {
            "id": "chatcmpl-xyz",
            "object": "chat.completion",
            "created": 1712345678,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35,
            },
        }
        tool_calls = resp["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"


class TestCanonicalStreamChunk:
    """Tests for CanonicalStreamChunk structure."""

    def test_text_delta_chunk(self):
        """CanonicalStreamChunk for a text delta."""
        chunk: CanonicalStreamChunk = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1712345678,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }
        assert chunk["choices"][0]["delta"]["content"] == "Hello"

    def test_final_chunk_with_finish_reason(self):
        """CanonicalStreamChunk with finish_reason='stop'."""
        chunk: CanonicalStreamChunk = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1712345678,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_chunk_with_usage(self):
        """Final chunk can include usage info."""
        chunk: CanonicalStreamChunk = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1712345678,
            "model": "gpt-4o",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        assert chunk["usage"]["total_tokens"] == 15


class TestFinishReason:
    """Tests for FinishReason constants."""

    def test_constants(self):
        assert FinishReason.STOP == "stop"
        assert FinishReason.LENGTH == "length"
        assert FinishReason.TOOL_CALLS == "tool_calls"
        assert FinishReason.CONTENT_FILTER == "content_filter"
        assert FinishReason.NULL is None


class TestToolTypes:
    """Tests for tool/function calling type structures."""

    def test_tool_definition(self):
        tool: Tool = {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
        }
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "search_web"

    def test_tool_call(self):
        tc: ToolCall = {
            "id": "call_xyz",
            "type": "function",
            "function": {"name": "search_web", "arguments": '{"query": "Python"}'},
        }
        assert tc["id"] == "call_xyz"
        assert tc["function"]["name"] == "search_web"

    def test_tool_message(self):
        msg: ToolMessage = {
            "role": "tool",
            "content": '{"results": ["Python is a language"]}',
            "tool_call_id": "call_xyz",
        }
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_xyz"
