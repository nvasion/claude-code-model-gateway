"""Translator registry for model provider request/response translation.

The registry maps provider names to :class:`~src.translators.base.BaseTranslator`
instances and provides a unified interface for:

* Looking up a translator by provider name
* Registering custom or override translators
* Listing available providers
* Translating requests and responses without knowing the provider in advance

Built-in mappings
-----------------

+------------------+-------------------------------------------------+
| Provider name    | Translator class                                |
+==================+=================================================+
| ``"openai"``     | :class:`~src.translators.openai.OpenAITranslator`|
+------------------+-------------------------------------------------+
| ``"azure"``      | :class:`~src.translators.openai.AzureOpenAITranslator`|
+------------------+-------------------------------------------------+
| ``"openrouter"`` | :class:`~src.translators.openai.OpenAITranslator`|
+------------------+-------------------------------------------------+
| ``"local"``      | :class:`~src.translators.openai.OpenAITranslator`|
+------------------+-------------------------------------------------+
| ``"anthropic"``  | :class:`~src.translators.anthropic.AnthropicTranslator`|
+------------------+-------------------------------------------------+
| ``"google"``     | :class:`~src.translators.gemini.GeminiTranslator`|
+------------------+-------------------------------------------------+
| ``"gemini"``     | :class:`~src.translators.gemini.GeminiTranslator`|
+------------------+-------------------------------------------------+
| ``"bedrock"``    | :class:`~src.translators.bedrock.BedrockTranslator`|
+------------------+-------------------------------------------------+

Usage example::

    from src.translators.registry import TranslatorRegistry

    registry = TranslatorRegistry()

    # Translate outbound request
    body = registry.translate_request("anthropic", canonical_request)

    # Translate inbound response
    canonical_resp = registry.translate_response("anthropic", raw_response)

    # Register a custom translator
    from my_pkg import MyCustomTranslator
    registry.register("my-provider", MyCustomTranslator())

    # Use the global singleton
    from src.translators.registry import get_registry
    translator = get_registry().get("openai")
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, List, Optional

from src.translators.base import BaseTranslator, TranslationError
from src.translators.types import (
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStreamChunk,
)


class TranslatorRegistry:
    """Thread-safe registry that maps provider names to translator instances.

    Translators are registered by their string provider name (e.g.
    ``"openai"``, ``"anthropic"``).  Multiple aliases can point to the same
    translator instance (e.g. ``"gemini"`` and ``"google"`` both resolve to
    :class:`~src.translators.gemini.GeminiTranslator`).

    The registry is populated lazily: built-in translators are instantiated
    on first access to avoid importing all provider modules at startup.

    Args:
        eager_load: If ``True``, all built-in translators are instantiated
            immediately in ``__init__``.  Defaults to ``False`` (lazy).
    """

    def __init__(self, *, eager_load: bool = False) -> None:
        self._lock = threading.Lock()
        # Maps provider name → translator instance
        self._translators: Dict[str, BaseTranslator] = {}

        if eager_load:
            self._load_all_builtins()

    # -----------------------------------------------------------------------
    # Built-in provider factory
    # -----------------------------------------------------------------------

    def _get_builtin_translator(self, name: str) -> Optional[BaseTranslator]:
        """Instantiate a built-in translator by provider name.

        Returns ``None`` if *name* is not a known built-in.
        """
        # Import lazily to avoid circular imports and startup overhead
        if name in ("openai", "openrouter", "local"):
            from src.translators.openai import OpenAITranslator

            return OpenAITranslator()

        if name in ("azure",):
            from src.translators.openai import AzureOpenAITranslator

            return AzureOpenAITranslator()

        if name in ("anthropic",):
            from src.translators.anthropic import AnthropicTranslator

            return AnthropicTranslator()

        if name in ("google", "gemini"):
            from src.translators.gemini import GeminiTranslator

            return GeminiTranslator()

        if name in ("bedrock",):
            from src.translators.bedrock import BedrockTranslator

            return BedrockTranslator()

        return None

    def _load_all_builtins(self) -> None:
        """Eagerly instantiate all built-in translators."""
        for name in (
            "openai",
            "azure",
            "openrouter",
            "local",
            "anthropic",
            "google",
            "gemini",
            "bedrock",
        ):
            if name not in self._translators:
                translator = self._get_builtin_translator(name)
                if translator is not None:
                    self._translators[name] = translator

    # -----------------------------------------------------------------------
    # Registration / lookup
    # -----------------------------------------------------------------------

    def register(
        self,
        provider_name: str,
        translator: BaseTranslator,
        *,
        aliases: Optional[List[str]] = None,
        overwrite: bool = True,
    ) -> None:
        """Register a translator for a provider name.

        Args:
            provider_name: The canonical provider identifier (e.g. ``"openai"``).
            translator: The translator instance to register.
            aliases: Optional list of additional names that resolve to the
                same translator.
            overwrite: If ``False`` and *provider_name* is already registered,
                raises :class:`ValueError`.

        Raises:
            ValueError: If *overwrite* is ``False`` and the name is taken.
        """
        with self._lock:
            if not overwrite and provider_name in self._translators:
                raise ValueError(
                    f"Translator for provider '{provider_name}' is already registered. "
                    "Pass overwrite=True to replace it."
                )
            self._translators[provider_name] = translator
            for alias in aliases or []:
                self._translators[alias] = translator

    def unregister(self, provider_name: str) -> bool:
        """Remove a translator registration.

        Args:
            provider_name: The provider name to unregister.

        Returns:
            ``True`` if the translator was found and removed, ``False`` if not.
        """
        with self._lock:
            if provider_name in self._translators:
                del self._translators[provider_name]
                return True
        return False

    def get(self, provider_name: str) -> Optional[BaseTranslator]:
        """Return the translator for *provider_name*, or ``None``.

        Checks the registry first; if not found, tries the built-in factory.
        Successfully resolved built-in translators are cached for subsequent
        calls.

        Args:
            provider_name: Provider identifier (case-sensitive).

        Returns:
            :class:`BaseTranslator` instance or ``None`` if not found.
        """
        with self._lock:
            if provider_name in self._translators:
                return self._translators[provider_name]

        # Try built-in (outside lock to avoid import-time deadlocks)
        translator = self._get_builtin_translator(provider_name)
        if translator is not None:
            with self._lock:
                # Double-check: another thread may have loaded it already
                if provider_name not in self._translators:
                    self._translators[provider_name] = translator
                return self._translators[provider_name]

        return None

    def require(self, provider_name: str) -> BaseTranslator:
        """Return the translator for *provider_name*, raising if not found.

        Args:
            provider_name: Provider identifier.

        Returns:
            :class:`BaseTranslator` instance.

        Raises:
            TranslationError: If no translator is registered for *provider_name*.
        """
        translator = self.get(provider_name)
        if translator is None:
            available = ", ".join(sorted(self.list_providers()))
            raise TranslationError(
                f"No translator registered for provider '{provider_name}'. "
                f"Available providers: {available or '(none)'}",
                provider=provider_name,
            )
        return translator

    def has(self, provider_name: str) -> bool:
        """Return ``True`` if a translator is registered for *provider_name*."""
        return self.get(provider_name) is not None

    def list_providers(self) -> List[str]:
        """Return a sorted list of all registered provider names."""
        # Combine explicitly registered names with known built-ins
        builtin_names = {
            "openai",
            "azure",
            "openrouter",
            "local",
            "anthropic",
            "google",
            "gemini",
            "bedrock",
        }
        with self._lock:
            registered = set(self._translators.keys())
        return sorted(registered | builtin_names)

    def __iter__(self) -> Iterator[str]:
        """Iterate over all provider names (including built-ins)."""
        return iter(self.list_providers())

    def __contains__(self, provider_name: str) -> bool:
        """Support ``"openai" in registry`` syntax."""
        return self.has(provider_name)

    def __len__(self) -> int:
        """Return number of registered translators (including loaded built-ins)."""
        with self._lock:
            return len(self._translators)

    # -----------------------------------------------------------------------
    # Convenience translation methods
    # -----------------------------------------------------------------------

    def translate_request(
        self,
        provider_name: str,
        request: CanonicalRequest,
        *,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Translate a canonical request for *provider_name*.

        Args:
            provider_name: Target provider identifier.
            request: Canonical request dict.
            model: Optional model name override.

        Returns:
            Provider-specific request body dict.

        Raises:
            TranslationError: If no translator is found or translation fails.
        """
        return self.require(provider_name).translate_request(request, model=model)

    def translate_response(
        self,
        provider_name: str,
        response: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> CanonicalResponse:
        """Translate a provider response back to canonical format.

        Args:
            provider_name: Source provider identifier.
            response: Raw provider response dict.
            model: Optional model name to embed in the canonical response.

        Returns:
            Canonical response dict.

        Raises:
            TranslationError: If no translator is found or translation fails.
        """
        return self.require(provider_name).translate_response(response, model=model)

    def translate_stream_chunk(
        self,
        provider_name: str,
        chunk: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> Optional[CanonicalStreamChunk]:
        """Translate a provider streaming chunk to canonical format.

        Args:
            provider_name: Source provider identifier.
            chunk: Parsed provider SSE event dict.
            model: Optional model name.

        Returns:
            Canonical stream chunk or ``None`` if the chunk should be dropped.

        Raises:
            TranslationError: If no translator is found or translation fails.
        """
        return self.require(provider_name).translate_stream_chunk(
            chunk, model=model
        )

    def get_api_path(
        self,
        provider_name: str,
        model: Optional[str] = None,
    ) -> str:
        """Return the API endpoint path for *provider_name* and *model*.

        Args:
            provider_name: Provider identifier.
            model: Optional model name (required by some providers like Gemini).

        Returns:
            URL path string.

        Raises:
            TranslationError: If no translator is found.
        """
        return self.require(provider_name).get_api_path(model)

    def get_extra_headers(
        self,
        provider_name: str,
        request: CanonicalRequest,
        *,
        model: Optional[str] = None,
    ) -> Dict[str, str]:
        """Return additional HTTP headers required by *provider_name*.

        Args:
            provider_name: Provider identifier.
            request: The canonical request (may affect headers).
            model: Optional model name.

        Returns:
            Dict of header name → value.

        Raises:
            TranslationError: If no translator is found.
        """
        return self.require(provider_name).get_extra_headers(request, model=model)

    # -----------------------------------------------------------------------
    # Repr
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TranslatorRegistry(registered={len(self._translators)}, "
            f"providers={self.list_providers()})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_registry: Optional[TranslatorRegistry] = None
_global_registry_lock = threading.Lock()


def get_registry() -> TranslatorRegistry:
    """Return the module-level singleton :class:`TranslatorRegistry`.

    The registry is created on first call (lazy initialisation).  The same
    instance is returned on subsequent calls.

    Returns:
        The global :class:`TranslatorRegistry` instance.
    """
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = TranslatorRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry singleton (primarily for testing).

    After calling this, :func:`get_registry` will create a fresh instance.
    """
    global _global_registry
    with _global_registry_lock:
        _global_registry = None
