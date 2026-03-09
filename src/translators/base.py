"""Abstract base class for model provider translators.

Every provider translator must extend :class:`BaseTranslator` and implement
the three core translation methods:

* :meth:`translate_request` – canonical → provider request format
* :meth:`translate_response` – provider response → canonical format
* :meth:`translate_stream_chunk` – provider SSE chunk → canonical chunk

The canonical format is the OpenAI Chat Completions API schema (see
:mod:`src.translators.types`).

Usage example::

    from src.translators.anthropic import AnthropicTranslator

    translator = AnthropicTranslator()

    # Translate outgoing request
    provider_body = translator.translate_request(canonical_request)

    # Translate incoming response
    canonical_resp = translator.translate_response(raw_provider_response)

    # Translate a single streaming chunk
    chunk = translator.translate_stream_chunk(raw_sse_event_dict)
"""

from __future__ import annotations

import abc
import time
from typing import Any, Dict, List, Optional

from src.translators.types import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamChunk,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TranslationError(Exception):
    """Raised when a request or response cannot be translated.

    Args:
        message: Human-readable description of what went wrong.
        provider: Name of the provider the translator targets.
        direction: Either ``"request"`` or ``"response"`` to indicate
            which translation direction failed.
        details: Optional dictionary with additional diagnostic data.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        direction: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.direction = direction
        self.details: Dict[str, Any] = details or {}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TranslationError(provider={self.provider!r}, "
            f"direction={self.direction!r}, message={str(self)!r})"
        )


class UnsupportedFeatureError(TranslationError):
    """Raised when a canonical feature is not supported by a provider.

    For example, Google Gemini does not support ``frequency_penalty``; if a
    caller passes it the translator raises this error so the caller can
    decide whether to strip the field or abort.

    Args:
        feature: The unsupported feature name (e.g. ``"frequency_penalty"``).
        provider: The target provider name.
    """

    def __init__(self, feature: str, provider: str = "") -> None:
        super().__init__(
            f"Feature '{feature}' is not supported by provider '{provider}'",
            provider=provider,
            direction="request",
            details={"unsupported_feature": feature},
        )
        self.feature = feature


# ---------------------------------------------------------------------------
# Abstract base translator
# ---------------------------------------------------------------------------


class BaseTranslator(abc.ABC):
    """Abstract base class that all provider translators must extend.

    Responsibilities
    ----------------
    * **translate_request** – converts a :class:`CanonicalRequest` dict into
      the provider's native request body.
    * **translate_response** – converts the provider's raw response dict back
      into a :class:`CanonicalResponse`.
    * **translate_stream_chunk** – converts a single provider streaming event
      dict into a :class:`CanonicalStreamChunk`.  Returns ``None`` for events
      that should be dropped (e.g. keep-alives, non-content events).

    Subclasses may also override:

    * :meth:`get_api_path` – the URL path for the chat completions endpoint
      (some providers use model-specific paths).
    * :meth:`get_extra_headers` – additional HTTP headers required by the
      provider.
    * :meth:`supports_streaming` – whether the provider supports SSE streaming.
    * :meth:`supports_tools` – whether the provider supports tool calling.

    Class attributes
    ----------------
    PROVIDER_NAME : str
        Canonical provider identifier (e.g. ``"anthropic"``).  Must be set by
        each concrete subclass.
    """

    PROVIDER_NAME: str = ""

    # -- Abstract methods --------------------------------------------------- #

    @abc.abstractmethod
    def translate_request(
        self,
        request: CanonicalRequest,
        *,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Translate a canonical request to provider-specific format.

        Args:
            request: A :class:`CanonicalRequest` dict in OpenAI format.
            model: Optional model name override; if omitted the value from
                ``request["model"]`` is used.

        Returns:
            Provider-specific request body dict, ready to be JSON-serialised
            and posted to the provider's API.

        Raises:
            TranslationError: If the request cannot be translated.
            UnsupportedFeatureError: If the request contains a feature that
                this provider does not support.
        """

    @abc.abstractmethod
    def translate_response(
        self,
        response: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> CanonicalResponse:
        """Translate a provider response to canonical format.

        Args:
            response: The raw provider response dict (parsed from JSON).
            model: Optional model name to embed in the canonical response.

        Returns:
            A :class:`CanonicalResponse` dict in OpenAI Chat Completions
            format.

        Raises:
            TranslationError: If the response cannot be translated.
        """

    @abc.abstractmethod
    def translate_stream_chunk(
        self,
        chunk: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> Optional[CanonicalStreamChunk]:
        """Translate a single provider SSE chunk to canonical format.

        Args:
            chunk: A parsed provider streaming event dict.  The caller is
                responsible for stripping the ``data:`` prefix and parsing
                the JSON payload before passing it here.
            model: Optional model name to embed in the canonical chunk.

        Returns:
            A :class:`CanonicalStreamChunk` dict, or ``None`` if this chunk
            should be silently discarded (e.g. keep-alive pings, metadata
            events with no content delta).

        Raises:
            TranslationError: If the chunk cannot be translated.
        """

    # -- Concrete helpers --------------------------------------------------- #

    def get_api_path(self, model: Optional[str] = None) -> str:
        """Return the API endpoint path for chat completions.

        Most OpenAI-compatible providers use ``/chat/completions``.
        Override this for providers that use model-specific paths (e.g.
        Google Gemini uses ``/models/{model}:generateContent``).

        Args:
            model: The model name (used by some providers).

        Returns:
            URL path string.
        """
        return "/chat/completions"

    def get_extra_headers(
        self, request: CanonicalRequest, *, model: Optional[str] = None
    ) -> Dict[str, str]:
        """Return additional HTTP headers required for this provider.

        For example, Anthropic requires ``anthropic-version``.  These headers
        are *merged* with any headers already in the provider's config.

        Args:
            request: The canonical request (may be needed to set
                request-specific headers).
            model: The target model name.

        Returns:
            Dictionary of header name → value.
        """
        return {}

    def supports_streaming(self) -> bool:
        """Return ``True`` if this provider supports SSE streaming."""
        return True

    def supports_tools(self) -> bool:
        """Return ``True`` if this provider supports tool / function calling."""
        return True

    def get_provider_name(self) -> str:
        """Return the canonical provider name."""
        return self.PROVIDER_NAME

    # -- Utility helpers ---------------------------------------------------- #

    @staticmethod
    def _make_response_id(prefix: str = "chatcmpl") -> str:
        """Generate a simple unique response ID."""
        import uuid

        return f"{prefix}-{uuid.uuid4().hex[:24]}"

    @staticmethod
    def _now() -> int:
        """Return the current Unix timestamp as an integer."""
        return int(time.time())

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Safely cast *value* to int, returning *default* on failure."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_str(value: Any, default: str = "") -> str:
        """Safely cast *value* to str, returning *default* on failure."""
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _extract_text_content(content: Any) -> Optional[str]:
        """Extract a plain text string from a (possibly multimodal) content value.

        If *content* is a list of content parts, returns the concatenation of
        all ``text``-typed parts.  If *content* is already a string, returns it
        directly.  Returns ``None`` if no text can be extracted.

        Args:
            content: A message content value (str or list of dicts).

        Returns:
            The extracted text, or ``None``.
        """
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if text:
                        parts.append(text)
            return "\n".join(parts) if parts else None
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(provider={self.PROVIDER_NAME!r})"
