"""Anthropic Messages API request/response translator.

Converts between the canonical OpenAI Chat Completions format and the
Anthropic Messages API format (``POST /v1/messages``).

Key differences from OpenAI format
-----------------------------------

**Request**

+-----------------------------+----------------------------------------+
| OpenAI (canonical)          | Anthropic                              |
+=============================+========================================+
| ``messages[].role="system"``| Extracted → top-level ``system``       |
+-----------------------------+----------------------------------------+
| ``messages[].role="tool"``  | ``role="user"`` + ``content`` list     |
|                             | with ``{"type":"tool_result", ...}``   |
+-----------------------------+----------------------------------------+
| ``tools[].function.parameters`` | ``tools[].input_schema``           |
+-----------------------------+----------------------------------------+
| ``tool_choice="auto"``      | ``{"type":"auto"}``                    |
+-----------------------------+----------------------------------------+
| ``tool_choice="none"``      | ``{"type":"none"}``                    |
+-----------------------------+----------------------------------------+
| ``tool_choice="required"``  | ``{"type":"any"}``                     |
+-----------------------------+----------------------------------------+
| ``tool_choice={"function":…}`` | ``{"type":"tool","name":"…"}``      |
+-----------------------------+----------------------------------------+
| ``stop`` (str or list)      | ``stop_sequences`` (list)              |
+-----------------------------+----------------------------------------+
| ``max_tokens`` (optional)   | ``max_tokens`` (**required**)          |
+-----------------------------+----------------------------------------+

**Response**

+--------------------------------------------+-----------------------------+
| Anthropic                                  | OpenAI (canonical)          |
+============================================+=============================+
| ``content`` list of blocks                 | ``choices[0].message``      |
+--------------------------------------------+-----------------------------+
| ``{"type":"text","text":"…"}``             | ``message.content``         |
+--------------------------------------------+-----------------------------+
| ``{"type":"tool_use","id":"…","name":"…",``| ``message.tool_calls``      |
| ``"input":{…}}``                           |                             |
+--------------------------------------------+-----------------------------+
| ``stop_reason="end_turn"``                 | ``finish_reason="stop"``    |
+--------------------------------------------+-----------------------------+
| ``stop_reason="max_tokens"``               | ``finish_reason="length"``  |
+--------------------------------------------+-----------------------------+
| ``stop_reason="tool_use"``                 | ``finish_reason="tool_calls"``|
+--------------------------------------------+-----------------------------+
| ``stop_reason="stop_sequence"``            | ``finish_reason="stop"``    |
+--------------------------------------------+-----------------------------+
| ``usage.input_tokens``                     | ``usage.prompt_tokens``     |
+--------------------------------------------+-----------------------------+
| ``usage.output_tokens``                    | ``usage.completion_tokens`` |
+--------------------------------------------+-----------------------------+

**Streaming**

Anthropic SSE emits typed events such as ``message_start``,
``content_block_start``, ``content_block_delta``, ``content_block_stop``,
``message_delta``, and ``message_stop``.  This translator maps:

* ``content_block_delta`` with ``delta.type="text_delta"`` → text chunk
* ``content_block_delta`` with ``delta.type="input_json_delta"`` → tool-call
  argument chunk
* ``message_stop`` → final chunk with ``finish_reason``
* All other events → ``None`` (dropped)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.translators.base import BaseTranslator, TranslationError
from src.translators.types import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamChunk,
)

# Default max_tokens when the canonical request omits it (Anthropic requires it)
_DEFAULT_MAX_TOKENS = 4096

# Anthropic stop_reason → canonical finish_reason
_STOP_REASON_MAP: Dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}


class AnthropicTranslator(BaseTranslator):
    """Translator for the Anthropic Messages API.

    Handles bidirectional conversion between the canonical OpenAI Chat
    Completions schema and the Anthropic ``/v1/messages`` schema.

    Args:
        default_max_tokens: Fallback ``max_tokens`` value injected into the
            Anthropic request when the canonical request does not specify one.
            Anthropic requires this field; OpenAI does not.
        anthropic_version: Value of the ``anthropic-version`` header to
            include in extra headers.
    """

    PROVIDER_NAME = "anthropic"

    def __init__(
        self,
        *,
        default_max_tokens: int = _DEFAULT_MAX_TOKENS,
        anthropic_version: str = "2023-06-01",
    ) -> None:
        self._default_max_tokens = default_max_tokens
        self._anthropic_version = anthropic_version

    # -- translate_request -------------------------------------------------- #

    def translate_request(
        self,
        request: CanonicalRequest,
        *,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert a canonical OpenAI request to an Anthropic Messages request.

        Args:
            request: Canonical request dict.
            model: Optional model override.

        Returns:
            Anthropic-formatted request body.

        Raises:
            TranslationError: If messages are absent or malformed.
        """
        messages: List[Dict[str, Any]] = list(request.get("messages") or [])
        if not messages:
            raise TranslationError(
                "Request must contain at least one message",
                provider=self.PROVIDER_NAME,
                direction="request",
            )

        # --- Extract system message ---
        system_content: Optional[str] = None
        non_system: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                # Anthropic only supports a single system string
                text = self._extract_text_content(msg.get("content", ""))
                if text:
                    if system_content:
                        system_content = system_content + "\n" + text
                    else:
                        system_content = text
            else:
                non_system.append(msg)

        # --- Convert messages ---
        anthropic_messages: List[Dict[str, Any]] = []
        for msg in non_system:
            converted = self._convert_message(msg)
            if converted:
                anthropic_messages.append(converted)

        if not anthropic_messages:
            raise TranslationError(
                "Request must contain at least one non-system message",
                provider=self.PROVIDER_NAME,
                direction="request",
            )

        # --- Build body ---
        body: Dict[str, Any] = {
            "model": model or request.get("model", ""),
            "messages": anthropic_messages,
            "max_tokens": request.get("max_tokens") or self._default_max_tokens,
        }

        if system_content:
            body["system"] = system_content

        # Sampling parameters
        if "temperature" in request:
            body["temperature"] = request["temperature"]
        if "top_p" in request:
            body["top_p"] = request["top_p"]

        # stop → stop_sequences (must be a list)
        stop = request.get("stop")
        if stop is not None:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)

        # Tools
        tools = request.get("tools")
        if tools:
            body["tools"] = self._convert_tools(tools)

        # tool_choice
        tool_choice = request.get("tool_choice")
        if tool_choice is not None:
            body["tool_choice"] = self._convert_tool_choice(tool_choice)

        # stream flag
        if request.get("stream"):
            body["stream"] = True

        # user → metadata.user_id
        user = request.get("user")
        if user:
            body.setdefault("metadata", {})["user_id"] = user

        return body

    # -- translate_response ------------------------------------------------- #

    def translate_response(
        self,
        response: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> CanonicalResponse:
        """Convert an Anthropic Messages response to canonical format.

        Args:
            response: Raw Anthropic response dict.
            model: Optional model name to embed in the canonical response.

        Returns:
            Canonical response dict.

        Raises:
            TranslationError: If the response cannot be parsed.
        """
        if not isinstance(response, dict):
            raise TranslationError(
                f"Expected dict response, got {type(response).__name__}",
                provider=self.PROVIDER_NAME,
                direction="response",
            )

        # Extract content blocks
        content_blocks: List[Dict[str, Any]] = response.get("content") or []
        message_content: Optional[str] = None
        tool_calls: List[Dict[str, Any]] = []

        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                if message_content is None:
                    message_content = text
                else:
                    message_content += text
            elif btype == "tool_use":
                tool_calls.append(self._convert_tool_use_block(block))

        # Build message dict
        response_message: Dict[str, Any] = {"role": "assistant"}
        if message_content is not None:
            response_message["content"] = message_content
        else:
            response_message["content"] = None
        if tool_calls:
            response_message["tool_calls"] = tool_calls

        # finish_reason
        stop_reason: str = response.get("stop_reason") or "end_turn"
        finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")

        # Usage
        usage_raw: Dict[str, Any] = response.get("usage") or {}
        usage = {
            "prompt_tokens": self._safe_int(usage_raw.get("input_tokens")),
            "completion_tokens": self._safe_int(usage_raw.get("output_tokens")),
            "total_tokens": (
                self._safe_int(usage_raw.get("input_tokens"))
                + self._safe_int(usage_raw.get("output_tokens"))
            ),
        }

        result: Dict[str, Any] = {
            "id": response.get("id") or self._make_response_id(),
            "object": "chat.completion",
            "created": self._now(),
            "model": model or response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": usage,
        }

        return result  # type: ignore[return-value]

    # -- translate_stream_chunk --------------------------------------------- #

    def translate_stream_chunk(
        self,
        chunk: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> Optional[CanonicalStreamChunk]:
        """Convert an Anthropic SSE event dict to a canonical stream chunk.

        Anthropic emits several event types.  Only content-bearing events are
        converted; all others return ``None`` and are dropped by the caller.

        Args:
            chunk: Parsed Anthropic SSE event dict (with ``type`` field).
            model: Optional model name.

        Returns:
            Canonical stream chunk or ``None``.
        """
        if not chunk or not isinstance(chunk, dict):
            return None

        event_type: str = chunk.get("type", "")

        # -- content_block_delta: text or tool-arg chunk --
        if event_type == "content_block_delta":
            delta: Dict[str, Any] = chunk.get("delta") or {}
            delta_type: str = delta.get("type", "")
            index: int = self._safe_int(chunk.get("index"))

            if delta_type == "text_delta":
                text = delta.get("text", "")
                return self._make_text_chunk(text, index=index, model=model)

            if delta_type == "input_json_delta":
                # Partial JSON arguments for a tool call
                partial_json = delta.get("partial_json", "")
                return self._make_tool_arg_chunk(
                    partial_json, index=index, model=model
                )

            return None  # other delta types ignored

        # -- message_delta: finish_reason and final usage --
        if event_type == "message_delta":
            delta = chunk.get("delta") or {}
            stop_reason = delta.get("stop_reason") or "end_turn"
            finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")
            usage_raw = chunk.get("usage") or {}
            usage = None
            if usage_raw:
                usage = {
                    "prompt_tokens": self._safe_int(usage_raw.get("input_tokens")),
                    "completion_tokens": self._safe_int(
                        usage_raw.get("output_tokens")
                    ),
                    "total_tokens": (
                        self._safe_int(usage_raw.get("input_tokens"))
                        + self._safe_int(usage_raw.get("output_tokens"))
                    ),
                }
            result: Dict[str, Any] = {
                "id": self._make_response_id(),
                "object": "chat.completion.chunk",
                "created": self._now(),
                "model": model or "",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            if usage:
                result["usage"] = usage
            return result  # type: ignore[return-value]

        # -- content_block_start: tool_use start (emit tool_call id/name) --
        if event_type == "content_block_start":
            block: Dict[str, Any] = chunk.get("content_block") or {}
            index = self._safe_int(chunk.get("index"))
            if block.get("type") == "tool_use":
                tool_call_start: Dict[str, Any] = {
                    "index": index,
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": ""},
                }
                return {
                    "id": self._make_response_id(),
                    "object": "chat.completion.chunk",
                    "created": self._now(),
                    "model": model or "",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [tool_call_start]},
                            "finish_reason": None,
                        }
                    ],
                }  # type: ignore[return-value]
            return None  # text block start → no chunk needed

        # -- message_start: emit role delta --
        if event_type == "message_start":
            return {
                "id": self._make_response_id(),
                "object": "chat.completion.chunk",
                "created": self._now(),
                "model": model or "",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }  # type: ignore[return-value]

        # All other events (content_block_stop, message_stop, ping, …) → drop
        return None

    # -- Extra headers ------------------------------------------------------- #

    def get_extra_headers(
        self,
        request: CanonicalRequest,
        *,
        model: Optional[str] = None,
    ) -> Dict[str, str]:
        """Return Anthropic-specific required headers."""
        return {"anthropic-version": self._anthropic_version}

    def get_api_path(self, model: Optional[str] = None) -> str:
        """Return the Anthropic chat completions endpoint."""
        return "/messages"

    # -- Private helpers ---------------------------------------------------- #

    def _convert_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a single canonical message to Anthropic format."""
        role: str = msg.get("role", "")

        if role == "user":
            return {"role": "user", "content": self._convert_user_content(msg)}

        if role == "assistant":
            return {
                "role": "assistant",
                "content": self._convert_assistant_content(msg),
            }

        if role == "tool":
            # Tool results are wrapped in a user message with type=tool_result
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }
                ],
            }

        # Unknown roles → skip
        return None

    def _convert_user_content(self, msg: Dict[str, Any]) -> Any:
        """Convert a user message's content to Anthropic format."""
        content = msg.get("content")

        # Plain string → return as-is
        if isinstance(content, str):
            return content

        # List of content parts → convert each part
        if isinstance(content, list):
            anthropic_parts: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    anthropic_parts.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "image_url":
                    image_url = part.get("image_url") or {}
                    url: str = image_url.get("url", "")
                    anthropic_parts.append(
                        self._convert_image_url_to_anthropic(url)
                    )
            return anthropic_parts if anthropic_parts else ""

        return content or ""

    def _convert_image_url_to_anthropic(self, url: str) -> Dict[str, Any]:
        """Convert an OpenAI image_url to an Anthropic image block."""
        if url.startswith("data:"):
            # Data URI: data:image/png;base64,<data>
            try:
                header, data = url.split(",", 1)
                media_type = header.split(":")[1].split(";")[0]
            except (ValueError, IndexError):
                media_type = "image/png"
                data = url
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }
        else:
            # URL source
            return {
                "type": "image",
                "source": {"type": "url", "url": url},
            }

    def _convert_assistant_content(self, msg: Dict[str, Any]) -> Any:
        """Convert an assistant message's content to Anthropic format."""
        tool_calls: List[Dict[str, Any]] = msg.get("tool_calls") or []
        content = msg.get("content")

        if not tool_calls:
            # Simple text assistant turn
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return self._extract_text_content(content) or ""
            return content or ""

        # Mix of text and tool_use blocks
        blocks: List[Dict[str, Any]] = []

        # Text content first (if any)
        text = self._extract_text_content(content)
        if text:
            blocks.append({"type": "text", "text": text})

        # Tool use blocks
        for tc in tool_calls:
            fn = tc.get("function") or {}
            arguments_raw = fn.get("arguments", "{}")
            try:
                arguments = json.loads(arguments_raw) if arguments_raw else {}
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", self._make_response_id("toolu")),
                    "name": fn.get("name", ""),
                    "input": arguments,
                }
            )

        return blocks

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tools list to Anthropic format."""
        result: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function") or {}
            anthropic_tool: Dict[str, Any] = {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters")
                or {"type": "object", "properties": {}},
            }
            result.append(anthropic_tool)
        return result

    def _convert_tool_choice(self, tool_choice: Any) -> Dict[str, Any]:
        """Convert OpenAI tool_choice to Anthropic format."""
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "none":
            return {"type": "none"}
        if tool_choice == "required":
            return {"type": "any"}
        if isinstance(tool_choice, dict):
            fn = tool_choice.get("function") or {}
            name = fn.get("name", "")
            if name:
                return {"type": "tool", "name": name}
        return {"type": "auto"}

    def _convert_tool_use_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an Anthropic tool_use content block to a canonical tool_call."""
        input_data = block.get("input") or {}
        try:
            arguments = json.dumps(input_data)
        except (TypeError, ValueError):
            arguments = "{}"
        return {
            "id": block.get("id", self._make_response_id("toolu")),
            "type": "function",
            "function": {
                "name": block.get("name", ""),
                "arguments": arguments,
            },
        }

    def _make_text_chunk(
        self,
        text: str,
        *,
        index: int = 0,
        model: Optional[str] = None,
    ) -> CanonicalStreamChunk:
        """Build a canonical text-delta stream chunk."""
        return {
            "id": self._make_response_id(),
            "object": "chat.completion.chunk",
            "created": self._now(),
            "model": model or "",
            "choices": [
                {
                    "index": index,
                    "delta": {"content": text},
                    "finish_reason": None,
                }
            ],
        }  # type: ignore[return-value]

    def _make_tool_arg_chunk(
        self,
        partial_json: str,
        *,
        index: int = 0,
        model: Optional[str] = None,
    ) -> CanonicalStreamChunk:
        """Build a canonical tool-argument delta stream chunk."""
        return {
            "id": self._make_response_id(),
            "object": "chat.completion.chunk",
            "created": self._now(),
            "model": model or "",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": index,
                                "function": {"arguments": partial_json},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }  # type: ignore[return-value]
