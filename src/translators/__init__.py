"""Request/response translation layer for model providers.

This package converts between the **canonical** OpenAI Chat Completions API
format and the native request/response schemas used by each supported AI model
provider.

Supported providers
-------------------

+------------------+----------------------------------------------------------+
| Provider key     | Description                                              |
+==================+==========================================================+
| ``"openai"``     | OpenAI Chat Completions API (canonical passthrough)      |
+------------------+----------------------------------------------------------+
| ``"azure"``      | Azure OpenAI Service (same body as OpenAI)               |
+------------------+----------------------------------------------------------+
| ``"anthropic"``  | Anthropic Messages API (``POST /v1/messages``)           |
+------------------+----------------------------------------------------------+
| ``"google"``     | Google Gemini generateContent API                        |
+------------------+----------------------------------------------------------+
| ``"gemini"``     | Alias for ``"google"``                                   |
+------------------+----------------------------------------------------------+
| ``"bedrock"``    | AWS Bedrock (Claude, Titan, Llama model families)        |
+------------------+----------------------------------------------------------+

Quick start
-----------

**Using the global registry (recommended)**::

    from src.translators import get_registry, translate_request, translate_response

    # Translate a canonical request for Anthropic
    provider_body = translate_request("anthropic", canonical_request)

    # Translate the Anthropic response back to canonical
    canonical_resp = translate_response("anthropic", raw_provider_response)

**Using a translator directly**::

    from src.translators import AnthropicTranslator

    t = AnthropicTranslator()
    body = t.translate_request(canonical_request)
    resp = t.translate_response(raw_anthropic_response)

**Registering a custom translator**::

    from src.translators import get_registry
    from my_pkg import MyTranslator

    get_registry().register("my-provider", MyTranslator())

Module layout
-------------

.. code-block:: text

    src/translators/
    ├── __init__.py      ← this file; public API
    ├── types.py         ← canonical TypedDict types
    ├── base.py          ← abstract BaseTranslator + TranslationError
    ├── openai.py        ← OpenAI & Azure OpenAI translators
    ├── anthropic.py     ← Anthropic Messages API translator
    ├── gemini.py        ← Google Gemini translator
    ├── bedrock.py       ← AWS Bedrock translator (Claude, Titan, Llama)
    └── registry.py      ← TranslatorRegistry + global singleton helpers
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# -- Public types ---------------------------------------------------------
from src.translators.types import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamChunk,
    Choice,
    ContentPart,
    FinishReason,
    FunctionDefinition,
    FunctionParameters,
    ImageUrl,
    ImageUrlContentPart,
    Message,
    MessageContent,
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

# -- Base / errors --------------------------------------------------------
from src.translators.base import (
    BaseTranslator,
    TranslationError,
    UnsupportedFeatureError,
)

# -- Provider translators -------------------------------------------------
from src.translators.openai import AzureOpenAITranslator, OpenAITranslator
from src.translators.anthropic import AnthropicTranslator
from src.translators.gemini import GeminiTranslator
from src.translators.bedrock import BedrockTranslator

# -- Registry -------------------------------------------------------------
from src.translators.registry import (
    TranslatorRegistry,
    get_registry,
    reset_registry,
)


# ---------------------------------------------------------------------------
# Convenience module-level helpers
# ---------------------------------------------------------------------------


def translate_request(
    provider: str,
    request: CanonicalRequest,
    *,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Translate *request* for the named *provider*.

    Delegates to :meth:`TranslatorRegistry.translate_request` on the global
    singleton registry.

    Args:
        provider: Provider identifier (e.g. ``"anthropic"``).
        request: Canonical OpenAI-format request dict.
        model: Optional model name override.

    Returns:
        Provider-specific request body dict.

    Raises:
        TranslationError: If the provider is unknown or translation fails.

    Example::

        body = translate_request("anthropic", {"model": "claude-sonnet-4-20250514",
                                               "messages": [{"role": "user",
                                                             "content": "Hello"}]})
    """
    return get_registry().translate_request(provider, request, model=model)


def translate_response(
    provider: str,
    response: Dict[str, Any],
    *,
    model: Optional[str] = None,
) -> CanonicalResponse:
    """Translate a provider *response* back to canonical format.

    Delegates to :meth:`TranslatorRegistry.translate_response` on the global
    singleton registry.

    Args:
        provider: Provider identifier.
        response: Raw provider response dict.
        model: Optional model name to embed in the canonical response.

    Returns:
        Canonical OpenAI Chat Completions response dict.

    Raises:
        TranslationError: If the provider is unknown or translation fails.
    """
    return get_registry().translate_response(provider, response, model=model)


def translate_stream_chunk(
    provider: str,
    chunk: Dict[str, Any],
    *,
    model: Optional[str] = None,
) -> Optional[CanonicalStreamChunk]:
    """Translate a single provider SSE *chunk* to canonical format.

    Delegates to :meth:`TranslatorRegistry.translate_stream_chunk` on the
    global singleton registry.

    Args:
        provider: Provider identifier.
        chunk: Parsed provider streaming event dict.
        model: Optional model name.

    Returns:
        Canonical stream chunk or ``None`` if the chunk should be dropped.

    Raises:
        TranslationError: If the provider is unknown or translation fails.
    """
    return get_registry().translate_stream_chunk(provider, chunk, model=model)


__all__ = [
    # Types
    "CanonicalRequest",
    "CanonicalResponse",
    "CanonicalStreamChunk",
    "Choice",
    "ContentPart",
    "FinishReason",
    "FunctionDefinition",
    "FunctionParameters",
    "ImageUrl",
    "ImageUrlContentPart",
    "Message",
    "MessageContent",
    "ResponseMessage",
    "StreamChoice",
    "StreamDelta",
    "TextContentPart",
    "Tool",
    "ToolCall",
    "ToolCallFunction",
    "ToolMessage",
    "UsageInfo",
    # Base
    "BaseTranslator",
    "TranslationError",
    "UnsupportedFeatureError",
    # Translators
    "OpenAITranslator",
    "AzureOpenAITranslator",
    "AnthropicTranslator",
    "GeminiTranslator",
    "BedrockTranslator",
    # Registry
    "TranslatorRegistry",
    "get_registry",
    "reset_registry",
    # Convenience functions
    "translate_request",
    "translate_response",
    "translate_stream_chunk",
]
