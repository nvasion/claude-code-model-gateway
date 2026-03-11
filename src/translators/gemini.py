"""Google Gemini (Generative Language API) request/response translator.

Converts between the canonical OpenAI Chat Completions format and the
Google Gemini ``generateContent`` API format.

Key differences
---------------

**Request**

+-------------------------------+------------------------------------------+
| OpenAI (canonical)            | Gemini ``generateContent``               |
+===============================+==========================================+
| ``messages``                  | ``contents`` (list of turns)             |
+-------------------------------+------------------------------------------+
| ``role="system"``             | ``systemInstruction`` top-level field    |
+-------------------------------+------------------------------------------+
| ``role="user"``               | ``role="user"``                          |
+-------------------------------+------------------------------------------+
| ``role="assistant"``          | ``role="model"``                         |
+-------------------------------+------------------------------------------+
| ``role="tool"``               | ``role="user"`` with ``functionResponse``|
+-------------------------------+------------------------------------------+
| ``content`` (str)             | ``parts=[{"text": "…"}]``                |
+-------------------------------+------------------------------------------+
| ``content`` (list)            | ``parts=[…]`` with type mapping          |
+-------------------------------+------------------------------------------+
| ``image_url``                 | ``inlineData`` or ``fileData``           |
+-------------------------------+------------------------------------------+
| ``max_tokens``                | ``generationConfig.maxOutputTokens``     |
+-------------------------------+------------------------------------------+
| ``temperature``               | ``generationConfig.temperature``         |
+-------------------------------+------------------------------------------+
| ``top_p``                     | ``generationConfig.topP``                |
+-------------------------------+------------------------------------------+
| ``stop`` (str or list)        | ``generationConfig.stopSequences``       |
+-------------------------------+------------------------------------------+
| ``n``                         | ``generationConfig.candidateCount``      |
+-------------------------------+------------------------------------------+
| ``response_format.type=json`` | ``generationConfig.responseMimeType``    |
+-------------------------------+------------------------------------------+
| ``tools``                     | ``tools[].functionDeclarations``         |
+-------------------------------+------------------------------------------+
| ``tool_choice``               | ``toolConfig.functionCallingConfig``     |
+-------------------------------+------------------------------------------+

**Response**

+------------------------------------------+-------------------------------+
| Gemini                                   | OpenAI (canonical)            |
+==========================================+===============================+
| ``candidates[0].content.parts``          | ``choices[0].message``        |
+------------------------------------------+-------------------------------+
| ``{"text": "…"}``                        | ``message.content``           |
+------------------------------------------+-------------------------------+
| ``{"functionCall": {…}}``                | ``message.tool_calls``        |
+------------------------------------------+-------------------------------+
| ``finishReason="STOP"``                  | ``finish_reason="stop"``      |
+------------------------------------------+-------------------------------+
| ``finishReason="MAX_TOKENS"``            | ``finish_reason="length"``    |
+------------------------------------------+-------------------------------+
| ``finishReason="SAFETY"``                | ``finish_reason="content_filter"``|
+------------------------------------------+-------------------------------+
| ``finishReason="RECITATION"``            | ``finish_reason="content_filter"``|
+------------------------------------------+-------------------------------+
| ``usageMetadata.promptTokenCount``       | ``usage.prompt_tokens``       |
+------------------------------------------+-------------------------------+
| ``usageMetadata.candidatesTokenCount``   | ``usage.completion_tokens``   |
+------------------------------------------+-------------------------------+

**API Path**

Gemini uses a model-specific path:
``/models/{model}:generateContent`` (or ``:streamGenerateContent``).

**Streaming**

Gemini streaming responses contain a JSON array or newline-delimited JSON
objects, each shaped like a non-streaming ``generateContent`` response.
This translator converts each candidate chunk into a canonical stream chunk.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional, Union

from src.translators.base import BaseTranslator, TranslationError
from src.translators.types import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamChunk,
)

# Gemini finishReason → canonical finish_reason
_FINISH_REASON_MAP: Dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
    "FINISH_REASON_UNSPECIFIED": "stop",
    "LANGUAGE": "stop",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
}


class GeminiTranslator(BaseTranslator):
    """Translator for the Google Gemini ``generateContent`` API.

    Args:
        api_version: Gemini API version prefix (default ``"v1beta"``).
            Controls the URL path used by :meth:`get_api_path`.
    """

    PROVIDER_NAME = "google"

    def __init__(self, *, api_version: str = "v1beta") -> None:
        self._api_version = api_version

    # -- translate_request -------------------------------------------------- #

    def translate_request(
        self,
        request: CanonicalRequest,
        *,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert a canonical OpenAI request to a Gemini generateContent request.

        Args:
            request: Canonical request dict.
            model: Optional model override.

        Returns:
            Gemini-formatted request body dict.

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

        # --- Extract system instruction ---
        system_parts: List[Dict[str, Any]] = []
        non_system: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                text = self._extract_text_content(msg.get("content", ""))
                if text:
                    system_parts.append({"text": text})
            else:
                non_system.append(msg)

        # --- Convert messages to Gemini contents ---
        contents: List[Dict[str, Any]] = []
        for msg in non_system:
            converted = self._convert_message(msg)
            if converted:
                contents.append(converted)

        if not contents:
            raise TranslationError(
                "Request must contain at least one non-system message",
                provider=self.PROVIDER_NAME,
                direction="request",
            )

        body: Dict[str, Any] = {"contents": contents}

        # System instruction
        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}

        # Generation config
        gen_config: Dict[str, Any] = {}
        if "max_tokens" in request and request["max_tokens"] is not None:
            gen_config["maxOutputTokens"] = request["max_tokens"]
        if "temperature" in request:
            gen_config["temperature"] = request["temperature"]
        if "top_p" in request:
            gen_config["topP"] = request["top_p"]
        if "n" in request:
            gen_config["candidateCount"] = request["n"]
        if "seed" in request:
            gen_config["seed"] = request["seed"]

        # stop → stopSequences
        stop = request.get("stop")
        if stop is not None:
            gen_config["stopSequences"] = (
                [stop] if isinstance(stop, str) else list(stop)
            )

        # response_format → responseMimeType
        response_format = request.get("response_format") or {}
        if response_format.get("type") == "json_object":
            gen_config["responseMimeType"] = "application/json"

        if gen_config:
            body["generationConfig"] = gen_config

        # Tools → functionDeclarations
        tools = request.get("tools")
        if tools:
            body["tools"] = [{"functionDeclarations": self._convert_tools(tools)}]

        # tool_choice → toolConfig
        tool_choice = request.get("tool_choice")
        if tool_choice is not None:
            body["toolConfig"] = self._convert_tool_choice(tool_choice)

        return body

    # -- translate_response ------------------------------------------------- #

    def translate_response(
        self,
        response: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> CanonicalResponse:
        """Convert a Gemini generateContent response to canonical format.

        Args:
            response: Raw Gemini response dict.
            model: Optional model name.

        Returns:
            Canonical response dict.

        Raises:
            TranslationError: If the response is structurally invalid.
        """
        if not isinstance(response, dict):
            raise TranslationError(
                f"Expected dict response, got {type(response).__name__}",
                provider=self.PROVIDER_NAME,
                direction="response",
            )

        candidates: List[Dict[str, Any]] = response.get("candidates") or []
        choices: List[Dict[str, Any]] = []

        for idx, candidate in enumerate(candidates):
            content_block: Dict[str, Any] = candidate.get("content") or {}
            parts: List[Dict[str, Any]] = content_block.get("parts") or []

            message_text: Optional[str] = None
            tool_calls: List[Dict[str, Any]] = []

            for part in parts:
                if "text" in part:
                    if message_text is None:
                        message_text = part["text"]
                    else:
                        message_text += part["text"]
                elif "functionCall" in part:
                    tool_calls.append(
                        self._convert_function_call(part["functionCall"])
                    )

            finish_raw: str = candidate.get("finishReason", "STOP")
            finish_reason = _FINISH_REASON_MAP.get(finish_raw, "stop")

            msg: Dict[str, Any] = {"role": "assistant"}
            msg["content"] = message_text
            if tool_calls:
                msg["tool_calls"] = tool_calls

            choices.append(
                {
                    "index": idx,
                    "message": msg,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            )

        # Usage metadata
        usage_raw: Dict[str, Any] = response.get("usageMetadata") or {}
        prompt_tokens = self._safe_int(usage_raw.get("promptTokenCount"))
        completion_tokens = self._safe_int(usage_raw.get("candidatesTokenCount"))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": self._safe_int(
                usage_raw.get("totalTokenCount", prompt_tokens + completion_tokens)
            ),
        }

        result: Dict[str, Any] = {
            "id": self._make_response_id("gemini"),
            "object": "chat.completion",
            "created": self._now(),
            "model": model or "",
            "choices": choices,
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
        """Convert a Gemini streaming response chunk to canonical format.

        Gemini streaming sends full ``generateContent``-shaped objects for
        each chunk (not deltas).  We extract only the content delta from the
        first candidate.

        Args:
            chunk: Parsed Gemini streaming JSON object.
            model: Optional model name.

        Returns:
            Canonical stream chunk or ``None``.
        """
        if not chunk or not isinstance(chunk, dict):
            return None

        candidates: List[Dict[str, Any]] = chunk.get("candidates") or []
        if not candidates:
            return None

        candidate = candidates[0]
        content_block: Dict[str, Any] = candidate.get("content") or {}
        parts: List[Dict[str, Any]] = content_block.get("parts") or []

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                tool_calls.append(
                    self._convert_function_call(part["functionCall"])
                )

        text = "".join(text_parts) if text_parts else None

        finish_raw: Optional[str] = candidate.get("finishReason")
        finish_reason: Optional[str] = None
        if finish_raw:
            finish_reason = _FINISH_REASON_MAP.get(finish_raw)

        delta: Dict[str, Any] = {}
        if text is not None:
            delta["content"] = text
        if tool_calls:
            delta["tool_calls"] = [
                {
                    "index": i,
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": tc.get("function", {}),
                }
                for i, tc in enumerate(tool_calls)
            ]

        # Usage in final chunk
        usage_raw: Dict[str, Any] = chunk.get("usageMetadata") or {}
        usage = None
        if usage_raw:
            prompt_tokens = self._safe_int(usage_raw.get("promptTokenCount"))
            completion_tokens = self._safe_int(
                usage_raw.get("candidatesTokenCount")
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": self._safe_int(
                    usage_raw.get(
                        "totalTokenCount", prompt_tokens + completion_tokens
                    )
                ),
            }

        result: Dict[str, Any] = {
            "id": self._make_response_id("gemini"),
            "object": "chat.completion.chunk",
            "created": self._now(),
            "model": model or "",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage:
            result["usage"] = usage

        return result  # type: ignore[return-value]

    # -- API path ------------------------------------------------------------ #

    def get_api_path(self, model: Optional[str] = None) -> str:
        """Return the Gemini generateContent path for *model*.

        Args:
            model: The Gemini model ID (e.g. ``"gemini-2.0-flash"``).

        Returns:
            Path string, e.g. ``"/models/gemini-2.0-flash:generateContent"``.
        """
        if model:
            return f"/models/{model}:generateContent"
        return "/models/gemini-2.0-flash:generateContent"

    def get_streaming_api_path(self, model: Optional[str] = None) -> str:
        """Return the Gemini streaming endpoint path for *model*."""
        if model:
            return f"/models/{model}:streamGenerateContent"
        return "/models/gemini-2.0-flash:streamGenerateContent"

    # -- Private helpers ---------------------------------------------------- #

    def _convert_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a single canonical message to a Gemini content object."""
        role: str = msg.get("role", "")

        if role == "user":
            parts = self._convert_user_content_to_parts(msg.get("content"))
            return {"role": "user", "parts": parts}

        if role == "assistant":
            parts = self._convert_assistant_content_to_parts(msg)
            return {"role": "model", "parts": parts}

        if role == "tool":
            # Function response wrapped in a user turn
            tool_call_id = msg.get("tool_call_id") or msg.get("name", "")
            content = msg.get("content", "")
            try:
                response_data = json.loads(content) if content else {}
            except (json.JSONDecodeError, TypeError):
                response_data = {"result": content}
            return {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": tool_call_id,
                            "response": response_data,
                        }
                    }
                ],
            }

        return None

    def _convert_user_content_to_parts(
        self, content: Any
    ) -> List[Dict[str, Any]]:
        """Convert user message content to Gemini parts list."""
        if content is None:
            return [{"text": ""}]
        if isinstance(content, str):
            return [{"text": content}]
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                ptype = item.get("type")
                if ptype == "text":
                    parts.append({"text": item.get("text", "")})
                elif ptype == "image_url":
                    image_url = item.get("image_url") or {}
                    url: str = image_url.get("url", "")
                    parts.append(self._convert_image_url_to_gemini(url))
            return parts or [{"text": ""}]
        return [{"text": str(content)}]

    def _convert_assistant_content_to_parts(
        self, msg: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert an assistant message to Gemini parts (text + functionCalls)."""
        parts: List[Dict[str, Any]] = []
        content = msg.get("content")
        tool_calls: List[Dict[str, Any]] = msg.get("tool_calls") or []

        if content:
            text = self._extract_text_content(content)
            if text:
                parts.append({"text": text})

        for tc in tool_calls:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if args_raw else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            parts.append(
                {
                    "functionCall": {
                        "name": fn.get("name", ""),
                        "args": args,
                    }
                }
            )

        return parts or [{"text": ""}]

    def _convert_image_url_to_gemini(self, url: str) -> Dict[str, Any]:
        """Convert an image URL / data URI to a Gemini inline part."""
        if url.startswith("data:"):
            try:
                header, data = url.split(",", 1)
                media_type = header.split(":")[1].split(";")[0]
            except (ValueError, IndexError):
                media_type = "image/png"
                data = url
            return {
                "inlineData": {
                    "mimeType": media_type,
                    "data": data,
                }
            }
        else:
            return {"fileData": {"mimeType": "image/jpeg", "fileUri": url}}

    def _convert_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI tools to Gemini functionDeclarations."""
        result: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function") or {}
            decl: Dict[str, Any] = {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
            }
            params = fn.get("parameters")
            if params:
                decl["parameters"] = params
            result.append(decl)
        return result

    def _convert_tool_choice(self, tool_choice: Any) -> Dict[str, Any]:
        """Convert OpenAI tool_choice to Gemini toolConfig."""
        if tool_choice == "none":
            return {
                "functionCallingConfig": {"mode": "NONE"}
            }
        if tool_choice == "auto":
            return {
                "functionCallingConfig": {"mode": "AUTO"}
            }
        if tool_choice == "required":
            return {
                "functionCallingConfig": {"mode": "ANY"}
            }
        if isinstance(tool_choice, dict):
            fn = tool_choice.get("function") or {}
            name = fn.get("name", "")
            if name:
                return {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [name],
                    }
                }
        return {"functionCallingConfig": {"mode": "AUTO"}}

    def _convert_function_call(
        self, function_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert a Gemini functionCall part to a canonical tool_call."""
        name = function_call.get("name", "")
        args = function_call.get("args") or {}
        try:
            arguments = json.dumps(args)
        except (TypeError, ValueError):
            arguments = "{}"
        return {
            "id": self._make_response_id("call"),
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
