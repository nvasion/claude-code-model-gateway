"""OpenAI and Azure OpenAI request/response translator.

OpenAI's Chat Completions API is the *canonical* format used throughout this
gateway, so this translator is essentially a passthrough with light
normalisation:

* **Request** – strips unknown / gateway-internal keys and ensures required
  fields are present.
* **Response** – re-uses the provider response almost verbatim; fills in any
  missing optional fields with sensible defaults.
* **Stream chunk** – returns the chunk unchanged after minor normalisation.

The same translator handles **Azure OpenAI** because Azure's Chat Completions
endpoint accepts exactly the same request/response schema.  The only
Azure-specific differences (deployment name in the URL, ``api-version``
query parameter, ``api-key`` header) are handled at the HTTP transport layer,
not in the body translation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set

from src.translators.base import BaseTranslator, TranslationError
from src.translators.types import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamChunk,
)

# Keys that are valid in an OpenAI Chat Completions request body.
# Anything else is stripped during normalisation to avoid 400 errors.
_VALID_REQUEST_KEYS: Set[str] = {
    "model",
    "messages",
    "max_tokens",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stream_options",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "seed",
    "user",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "metadata",
}


class OpenAITranslator(BaseTranslator):
    """Translator for the OpenAI (and Azure OpenAI) Chat Completions API.

    Since OpenAI format *is* the canonical format, translation is a
    near-passthrough.  The translator:

    1. Strips any gateway-internal or unrecognised keys from the request body
       so that the upstream API does not reject the request.
    2. Ensures the response has the expected structure and fills in sensible
       defaults for optional fields.
    3. Returns stream chunks with minor structural normalisation.

    Args:
        strict: When ``True``, :meth:`translate_request` raises
            :class:`TranslationError` if any required field (``model``,
            ``messages``) is missing.  When ``False`` (default), missing fields
            are silently accepted and the caller is responsible for ensuring
            the request is valid.
        strip_unknown_keys: When ``True`` (default), unrecognised top-level
            request keys are stripped before forwarding to the API.
    """

    PROVIDER_NAME = "openai"

    def __init__(
        self,
        *,
        strict: bool = False,
        strip_unknown_keys: bool = True,
    ) -> None:
        self._strict = strict
        self._strip_unknown_keys = strip_unknown_keys

    # -- translate_request -------------------------------------------------- #

    def translate_request(
        self,
        request: CanonicalRequest,
        *,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalise a canonical request for the OpenAI API.

        Args:
            request: Canonical request dict.
            model: Optional model override.

        Returns:
            Cleaned request dict ready to POST to OpenAI.

        Raises:
            TranslationError: If ``strict=True`` and required fields are absent.
        """
        body: Dict[str, Any] = dict(request)  # shallow copy

        # Apply model override
        if model is not None:
            body["model"] = model

        # Validate required fields in strict mode
        if self._strict:
            if "model" not in body or not body["model"]:
                raise TranslationError(
                    "Request is missing required field 'model'",
                    provider=self.PROVIDER_NAME,
                    direction="request",
                )
            if "messages" not in body or not body["messages"]:
                raise TranslationError(
                    "Request is missing required field 'messages'",
                    provider=self.PROVIDER_NAME,
                    direction="request",
                )

        # Strip unrecognised keys to prevent 400 errors
        if self._strip_unknown_keys:
            body = {k: v for k, v in body.items() if k in _VALID_REQUEST_KEYS}

        return body

    # -- translate_response ------------------------------------------------- #

    def translate_response(
        self,
        response: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> CanonicalResponse:
        """Normalise an OpenAI response into the canonical format.

        The OpenAI response is already in canonical format; this method
        makes a deep copy and fills in defaults for any missing optional
        fields.

        Args:
            response: Raw OpenAI response dict.
            model: Optional model name to embed if absent from the response.

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

        result: Dict[str, Any] = copy.deepcopy(response)

        # Fill in defaults for optional top-level fields
        result.setdefault("id", self._make_response_id())
        result.setdefault("object", "chat.completion")
        result.setdefault("created", self._now())

        if model and not result.get("model"):
            result["model"] = model

        # Normalise choices
        choices: List[Dict[str, Any]] = result.get("choices") or []
        for i, choice in enumerate(choices):
            choice.setdefault("index", i)
            choice.setdefault("finish_reason", None)
            msg: Dict[str, Any] = choice.get("message") or {}
            msg.setdefault("role", "assistant")
            choice["message"] = msg

        result["choices"] = choices

        # Normalise usage
        usage: Dict[str, Any] = result.get("usage") or {}
        usage.setdefault("prompt_tokens", 0)
        usage.setdefault("completion_tokens", 0)
        usage.setdefault("total_tokens", 0)
        result["usage"] = usage

        return result  # type: ignore[return-value]

    # -- translate_stream_chunk --------------------------------------------- #

    def translate_stream_chunk(
        self,
        chunk: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> Optional[CanonicalStreamChunk]:
        """Normalise an OpenAI streaming chunk into canonical format.

        Args:
            chunk: Parsed SSE data dict from the OpenAI streaming response.
            model: Optional model name override.

        Returns:
            Canonical stream chunk dict, or ``None`` if the chunk should be
            discarded.
        """
        if not chunk or not isinstance(chunk, dict):
            return None

        result: Dict[str, Any] = copy.deepcopy(chunk)

        result.setdefault("id", self._make_response_id())
        result.setdefault("object", "chat.completion.chunk")
        result.setdefault("created", self._now())

        if model and not result.get("model"):
            result["model"] = model

        choices: List[Dict[str, Any]] = result.get("choices") or []
        for i, choice in enumerate(choices):
            choice.setdefault("index", i)
            choice.setdefault("finish_reason", None)
            delta: Dict[str, Any] = choice.get("delta") or {}
            choice["delta"] = delta

        result["choices"] = choices

        return result  # type: ignore[return-value]

    # -- Extra helpers ------------------------------------------------------ #

    def get_api_path(self, model: Optional[str] = None) -> str:
        """Return the OpenAI chat completions endpoint path."""
        return "/chat/completions"


# ---------------------------------------------------------------------------
# Azure OpenAI translator (inherits OpenAI behaviour)
# ---------------------------------------------------------------------------


class AzureOpenAITranslator(OpenAITranslator):
    """Translator for Azure OpenAI.

    Azure uses exactly the same request/response body as OpenAI.  The only
    differences are at the HTTP transport level:

    * **Base URL** – ``https://<resource>.openai.azure.com/openai/deployments/<deployment>``
    * **API version** – ``?api-version=<version>`` query parameter
    * **Authentication** – ``api-key`` header instead of ``Authorization: Bearer``

    All of these are handled by the gateway's HTTP transport / provider config,
    not by body translation.  This subclass therefore provides no additional
    translation logic, but exposes a distinct :attr:`PROVIDER_NAME` so the
    translator registry can map ``"azure"`` to this class.
    """

    PROVIDER_NAME = "azure"

    def get_api_path(self, model: Optional[str] = None) -> str:
        """Azure path is assembled by the transport layer; return a placeholder."""
        return "/chat/completions"
