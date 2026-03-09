"""AWS Bedrock request/response translator.

AWS Bedrock is a managed AI service that exposes multiple model families
through a unified invocation API.  Each model family has its own native
request/response schema; Bedrock wraps the provider-native body in a Bedrock
invocation envelope.

This translator supports two Bedrock-hosted model families:

1. **Anthropic Claude via Bedrock** (``anthropic.claude-*``) – Uses the
   Anthropic Messages API format wrapped in Bedrock's invocation envelope.
   The ``anthropic-version`` field is placed inside the request body (not in
   HTTP headers, unlike the direct Anthropic API).

2. **Amazon Titan** (``amazon.titan-*``) – Uses Amazon's ``textGenerationConfig``
   format.

For other model families hosted on Bedrock (Cohere, AI21, Meta Llama, etc.)
the translator falls back to an OpenAI-like JSON body and logs a warning;
callers should extend or subclass :class:`BedrockTranslator` to add bespoke
support for additional families.

Bedrock API paths
-----------------

* Synchronous: ``POST /model/{modelId}/invoke``
* Streaming:   ``POST /model/{modelId}/invoke-with-response-stream``

The ``modelId`` is the Bedrock model identifier such as
``anthropic.claude-sonnet-4-20250514-v1:0``.

Authentication
--------------

Bedrock uses AWS Signature Version 4 (SigV4).  API-key-based auth is not
used; the gateway must be configured with IAM credentials (typically via the
environment variables ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and
``AWS_SESSION_TOKEN``).  Authentication header generation is outside the scope
of this translator and is handled by the HTTP transport layer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.translators.anthropic import AnthropicTranslator
from src.translators.base import BaseTranslator, TranslationError
from src.translators.types import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamChunk,
)

# Default Anthropic API version embedded in the body for Bedrock
_BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Model-family prefixes
_ANTHROPIC_PREFIX = "anthropic."
_TITAN_PREFIX = "amazon.titan"
_LLAMA_PREFIX = "meta.llama"
_COHERE_PREFIX = "cohere."
_AI21_PREFIX = "ai21."
_MISTRAL_PREFIX = "mistral."


def _detect_family(model_id: str) -> str:
    """Return the model family string for *model_id*."""
    lower = model_id.lower()
    if lower.startswith(_ANTHROPIC_PREFIX):
        return "anthropic"
    if lower.startswith(_TITAN_PREFIX):
        return "titan"
    if lower.startswith(_LLAMA_PREFIX):
        return "llama"
    if lower.startswith(_COHERE_PREFIX):
        return "cohere"
    if lower.startswith(_AI21_PREFIX):
        return "ai21"
    if lower.startswith(_MISTRAL_PREFIX):
        return "mistral"
    return "unknown"


class BedrockTranslator(BaseTranslator):
    """Translator for AWS Bedrock model invocation.

    Detects the model family from the model ID and delegates request/response
    translation to the appropriate family-specific logic.

    Currently supported families:

    * ``anthropic`` – Anthropic Claude models (full support incl. streaming,
      tools, vision)
    * ``titan`` – Amazon Titan text models (basic text generation only)
    * Other families fall back to a minimal JSON body

    Args:
        anthropic_version: The ``anthropic-version`` value to embed in the
            request body for Anthropic-family models.  Bedrock expects
            ``"bedrock-2023-05-31"`` (not the direct-API header format).
        default_max_tokens: Default ``max_tokens`` to inject for Anthropic
            models when the canonical request does not specify one.
    """

    PROVIDER_NAME = "bedrock"

    def __init__(
        self,
        *,
        anthropic_version: str = _BEDROCK_ANTHROPIC_VERSION,
        default_max_tokens: int = 4096,
    ) -> None:
        self._anthropic_version = anthropic_version
        self._default_max_tokens = default_max_tokens

        # Reuse the Anthropic translator for Claude-family request/response
        # translation, but customise the anthropic-version.
        self._anthropic = AnthropicTranslator(
            default_max_tokens=default_max_tokens,
            # Bedrock doesn't want this in headers – translator methods don't
            # add headers for Bedrock; we embed the version in the body.
            anthropic_version=anthropic_version,
        )

    # -- translate_request -------------------------------------------------- #

    def translate_request(
        self,
        request: CanonicalRequest,
        *,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert a canonical request to a Bedrock invocation body.

        The body format depends on the model family:

        * Anthropic Claude → Anthropic Messages format + ``anthropic_version``
        * Amazon Titan → ``inputText`` + ``textGenerationConfig``
        * Other → passthrough-style with ``prompt`` extracted from messages

        Args:
            request: Canonical request dict.
            model: Optional model ID override.

        Returns:
            Bedrock model invocation request body dict.

        Raises:
            TranslationError: If the request cannot be translated.
        """
        model_id = model or request.get("model", "")
        family = _detect_family(model_id)

        if family == "anthropic":
            return self._translate_request_anthropic(request, model=model_id)
        if family == "titan":
            return self._translate_request_titan(request)
        if family == "llama":
            return self._translate_request_llama(request)
        if family in ("cohere", "ai21", "mistral"):
            return self._translate_request_generic(request)

        # Unknown family – best effort generic translation
        return self._translate_request_generic(request)

    # -- translate_response ------------------------------------------------- #

    def translate_response(
        self,
        response: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> CanonicalResponse:
        """Convert a Bedrock invocation response to canonical format.

        Args:
            response: Raw Bedrock response body dict.
            model: Optional model ID to embed in the canonical response.

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

        model_id = model or ""
        family = _detect_family(model_id)

        if family == "anthropic":
            return self._translate_response_anthropic(response, model=model_id)
        if family == "titan":
            return self._translate_response_titan(response, model=model_id)
        if family == "llama":
            return self._translate_response_llama(response, model=model_id)

        return self._translate_response_generic(response, model=model_id)

    # -- translate_stream_chunk --------------------------------------------- #

    def translate_stream_chunk(
        self,
        chunk: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> Optional[CanonicalStreamChunk]:
        """Convert a Bedrock streaming event to canonical format.

        Bedrock streaming responses use a binary framing protocol at the HTTP
        layer (EventStream).  The gateway's transport layer is responsible for
        decoding the EventStream frames and extracting the payload JSON before
        calling this method.

        The payload format depends on the model family:

        * Anthropic Claude → same events as the direct Anthropic SSE API
        * Titan / others → chunk with ``outputText`` field

        Args:
            chunk: Decoded Bedrock streaming event payload dict.
            model: Optional model ID.

        Returns:
            Canonical stream chunk or ``None``.
        """
        if not chunk or not isinstance(chunk, dict):
            return None

        model_id = model or ""
        family = _detect_family(model_id)

        if family == "anthropic":
            # Bedrock Claude streaming events mirror the direct Anthropic SSE
            # event format; delegate to the Anthropic translator.
            return self._anthropic.translate_stream_chunk(chunk, model=model_id)
        if family == "titan":
            return self._translate_stream_chunk_titan(chunk, model=model_id)

        return self._translate_stream_chunk_generic(chunk, model=model_id)

    # -- API path ------------------------------------------------------------ #

    def get_api_path(self, model: Optional[str] = None) -> str:
        """Return the Bedrock invocation path for *model*.

        Args:
            model: The full Bedrock model ID.

        Returns:
            Path string, e.g. ``"/model/anthropic.claude-sonnet-4-20250514-v1:0/invoke"``.
        """
        if model:
            return f"/model/{model}/invoke"
        return "/model/invoke"

    def get_streaming_api_path(self, model: Optional[str] = None) -> str:
        """Return the Bedrock streaming invocation path for *model*."""
        if model:
            return f"/model/{model}/invoke-with-response-stream"
        return "/model/invoke-with-response-stream"

    def supports_streaming(self) -> bool:  # pragma: no cover
        """Bedrock supports streaming via EventStream."""
        return True

    # -----------------------------------------------------------------------
    # Private: Anthropic-family
    # -----------------------------------------------------------------------

    def _translate_request_anthropic(
        self,
        request: CanonicalRequest,
        *,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Translate to Anthropic Messages format for Bedrock Claude models."""
        body = self._anthropic.translate_request(request, model=model)
        # Bedrock requires anthropic_version in the body, NOT as a header
        body["anthropic_version"] = self._anthropic_version
        # Remove stream flag from body – Bedrock streaming is controlled by
        # the endpoint path (/invoke vs /invoke-with-response-stream)
        body.pop("stream", None)
        return body

    def _translate_response_anthropic(
        self,
        response: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> CanonicalResponse:
        """Translate an Anthropic-family Bedrock response to canonical format."""
        return self._anthropic.translate_response(response, model=model)

    # -----------------------------------------------------------------------
    # Private: Amazon Titan family
    # -----------------------------------------------------------------------

    def _translate_request_titan(
        self,
        request: CanonicalRequest,
    ) -> Dict[str, Any]:
        """Translate to Amazon Titan ``InvokeModel`` format."""
        # Titan expects a single ``inputText`` string
        messages: List[Dict[str, Any]] = list(request.get("messages") or [])
        prompt_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = self._extract_text_content(msg.get("content", ""))
            if text:
                if role == "system":
                    prompt_parts.append(f"System: {text}")
                elif role == "user":
                    prompt_parts.append(f"User: {text}")
                elif role == "assistant":
                    prompt_parts.append(f"Bot: {text}")

        prompt = "\n".join(prompt_parts)

        config: Dict[str, Any] = {}
        if "max_tokens" in request and request["max_tokens"]:
            config["maxTokenCount"] = request["max_tokens"]
        if "temperature" in request:
            config["temperature"] = request["temperature"]
        if "top_p" in request:
            config["topP"] = request["top_p"]
        stop = request.get("stop")
        if stop is not None:
            config["stopSequences"] = (
                [stop] if isinstance(stop, str) else list(stop)
            )

        body: Dict[str, Any] = {"inputText": prompt}
        if config:
            body["textGenerationConfig"] = config
        return body

    def _translate_response_titan(
        self,
        response: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> CanonicalResponse:
        """Translate an Amazon Titan response to canonical format."""
        results: List[Dict[str, Any]] = response.get("results") or []
        output_text = ""
        if results:
            output_text = results[0].get("outputText", "")

        completion_reason = (
            results[0].get("completionReason", "FINISH") if results
            else response.get("completionReason", "FINISH")
        )
        finish_reason = "stop" if completion_reason == "FINISH" else "length"

        input_tokens = self._safe_int(
            response.get("inputTextTokenCount", 0)
        )
        output_tokens = self._safe_int(
            results[0].get("tokenCount", 0) if results else 0
        )

        return {
            "id": self._make_response_id("titan"),
            "object": "chat.completion",
            "created": self._now(),
            "model": model or "",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }  # type: ignore[return-value]

    def _translate_stream_chunk_titan(
        self,
        chunk: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> Optional[CanonicalStreamChunk]:
        """Translate a Titan streaming chunk to canonical format."""
        output_text: str = chunk.get("outputText", "")
        index: str = chunk.get("index", "0")

        completion_reason = chunk.get("completionReason")
        finish_reason: Optional[str] = None
        if completion_reason == "FINISH":
            finish_reason = "stop"
        elif completion_reason == "LENGTH":
            finish_reason = "length"

        return {
            "id": self._make_response_id("titan"),
            "object": "chat.completion.chunk",
            "created": self._now(),
            "model": model or "",
            "choices": [
                {
                    "index": int(index) if str(index).isdigit() else 0,
                    "delta": {"content": output_text},
                    "finish_reason": finish_reason,
                }
            ],
        }  # type: ignore[return-value]

    # -----------------------------------------------------------------------
    # Private: Meta Llama family
    # -----------------------------------------------------------------------

    def _translate_request_llama(
        self,
        request: CanonicalRequest,
    ) -> Dict[str, Any]:
        """Translate to Meta Llama ``InvokeModel`` format on Bedrock."""
        messages: List[Dict[str, Any]] = list(request.get("messages") or [])
        # Llama 2/3 use a simple prompt string
        prompt_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = self._extract_text_content(msg.get("content", ""))
            if text:
                if role == "system":
                    prompt_parts.append(f"<|system|>\n{text}")
                elif role == "user":
                    prompt_parts.append(f"<|user|>\n{text}")
                elif role == "assistant":
                    prompt_parts.append(f"<|assistant|>\n{text}")
        prompt_parts.append("<|assistant|>")

        body: Dict[str, Any] = {"prompt": "\n".join(prompt_parts)}
        if "max_tokens" in request and request["max_tokens"]:
            body["max_gen_len"] = request["max_tokens"]
        if "temperature" in request:
            body["temperature"] = request["temperature"]
        if "top_p" in request:
            body["top_p"] = request["top_p"]
        return body

    def _translate_response_llama(
        self,
        response: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> CanonicalResponse:
        """Translate a Meta Llama Bedrock response to canonical format."""
        generation = response.get("generation", "")
        stop_reason = response.get("stop_reason", "stop")
        finish_reason = "stop" if stop_reason in ("stop", "eos_token") else "length"
        prompt_tokens = self._safe_int(response.get("prompt_token_count", 0))
        completion_tokens = self._safe_int(response.get("generation_token_count", 0))

        return {
            "id": self._make_response_id("llama"),
            "object": "chat.completion",
            "created": self._now(),
            "model": model or "",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generation},
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }  # type: ignore[return-value]

    # -----------------------------------------------------------------------
    # Private: Generic / fallback
    # -----------------------------------------------------------------------

    def _translate_request_generic(
        self,
        request: CanonicalRequest,
    ) -> Dict[str, Any]:
        """Generic fallback translation for unsupported Bedrock model families."""
        messages: List[Dict[str, Any]] = list(request.get("messages") or [])
        prompt = "\n".join(
            self._extract_text_content(m.get("content", "")) or ""
            for m in messages
        )
        body: Dict[str, Any] = {"prompt": prompt}
        if "max_tokens" in request and request["max_tokens"]:
            body["maxTokens"] = request["max_tokens"]
        if "temperature" in request:
            body["temperature"] = request["temperature"]
        return body

    def _translate_response_generic(
        self,
        response: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> CanonicalResponse:
        """Generic fallback response translation for unknown Bedrock families."""
        # Try common field names
        text = (
            response.get("text")
            or response.get("completion")
            or response.get("generation")
            or response.get("outputText")
            or ""
        )
        return {
            "id": self._make_response_id("bedrock"),
            "object": "chat.completion",
            "created": self._now(),
            "model": model or "",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }  # type: ignore[return-value]

    def _translate_stream_chunk_generic(
        self,
        chunk: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> Optional[CanonicalStreamChunk]:
        """Generic fallback streaming chunk translation."""
        text = (
            chunk.get("text")
            or chunk.get("completion")
            or chunk.get("generation")
            or chunk.get("outputText")
            or ""
        )
        if not text:
            return None
        return {
            "id": self._make_response_id("bedrock"),
            "object": "chat.completion.chunk",
            "created": self._now(),
            "model": model or "",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }
            ],
        }  # type: ignore[return-value]
