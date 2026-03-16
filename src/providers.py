"""Built-in provider definitions and provider registry."""

from __future__ import annotations

import copy
from collections.abc import Callable

from src.models import AuthType, ModelConfig, ProviderConfig


def _openai_provider() -> ProviderConfig:
    """Create the default OpenAI provider configuration."""
    return ProviderConfig(
        name="openai",
        display_name="OpenAI",
        api_base="https://api.openai.com/v1",
        api_key_env_var="OPENAI_API_KEY",
        auth_type=AuthType.BEARER_TOKEN,
        default_model="gpt-4o",
        models={
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                display_name="GPT-4o",
                max_tokens=16384,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "gpt-4o-mini": ModelConfig(
                name="gpt-4o-mini",
                display_name="GPT-4o Mini",
                max_tokens=16384,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "gpt-4-turbo": ModelConfig(
                name="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                max_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "o1": ModelConfig(
                name="o1",
                display_name="o1",
                max_tokens=100000,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "o1-mini": ModelConfig(
                name="o1-mini",
                display_name="o1 Mini",
                max_tokens=65536,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=False,
            ),
            "o3-mini": ModelConfig(
                name="o3-mini",
                display_name="o3 Mini",
                max_tokens=100000,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=False,
            ),
        },
    )


def _anthropic_provider() -> ProviderConfig:
    """Create the default Anthropic provider configuration."""
    return ProviderConfig(
        name="anthropic",
        display_name="Anthropic",
        api_base="https://api.anthropic.com/v1",
        api_key_env_var="ANTHROPIC_API_KEY",
        auth_type=AuthType.API_KEY,
        default_model="claude-sonnet-4-20250514",
        headers={"anthropic-version": "2023-06-01"},
        models={
            "claude-sonnet-4-20250514": ModelConfig(
                name="claude-sonnet-4-20250514",
                display_name="Claude Sonnet 4",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "claude-3-5-sonnet-20241022": ModelConfig(
                name="claude-3-5-sonnet-20241022",
                display_name="Claude 3.5 Sonnet",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "claude-3-5-haiku-20241022": ModelConfig(
                name="claude-3-5-haiku-20241022",
                display_name="Claude 3.5 Haiku",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=False,
            ),
            "claude-3-opus-20240229": ModelConfig(
                name="claude-3-opus-20240229",
                display_name="Claude 3 Opus",
                max_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
        },
    )


def _azure_openai_provider() -> ProviderConfig:
    """Create the default Azure OpenAI provider configuration."""
    return ProviderConfig(
        name="azure",
        display_name="Azure OpenAI",
        api_base="https://<your-resource>.openai.azure.com",
        api_key_env_var="AZURE_OPENAI_API_KEY",
        auth_type=AuthType.API_KEY,
        default_model="gpt-4o",
        extra={
            "api_version": "2024-06-01",
            "deployment_name": "<your-deployment>",
        },
        models={
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                display_name="GPT-4o (Azure)",
                max_tokens=16384,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "gpt-4o-mini": ModelConfig(
                name="gpt-4o-mini",
                display_name="GPT-4o Mini (Azure)",
                max_tokens=16384,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
        },
    )


def _google_provider() -> ProviderConfig:
    """Create the default Google (Gemini) provider configuration."""
    return ProviderConfig(
        name="google",
        display_name="Google Gemini",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        api_key_env_var="GOOGLE_API_KEY",
        auth_type=AuthType.API_KEY,
        default_model="gemini-2.0-flash",
        models={
            "gemini-2.0-flash": ModelConfig(
                name="gemini-2.0-flash",
                display_name="Gemini 2.0 Flash",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "gemini-2.0-pro": ModelConfig(
                name="gemini-2.0-pro",
                display_name="Gemini 2.0 Pro",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "gemini-1.5-pro": ModelConfig(
                name="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "gemini-1.5-flash": ModelConfig(
                name="gemini-1.5-flash",
                display_name="Gemini 1.5 Flash",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
        },
    )


def _bedrock_provider() -> ProviderConfig:
    """Create the default AWS Bedrock provider configuration."""
    return ProviderConfig(
        name="bedrock",
        display_name="AWS Bedrock",
        api_base="https://bedrock-runtime.<region>.amazonaws.com",
        api_key_env_var="",
        auth_type=AuthType.NONE,
        default_model="anthropic.claude-sonnet-4-20250514-v1:0",
        extra={
            "region": "us-east-1",
            "auth_method": "aws_credentials",
        },
        models={
            "anthropic.claude-sonnet-4-20250514-v1:0": ModelConfig(
                name="anthropic.claude-sonnet-4-20250514-v1:0",
                display_name="Claude Sonnet 4 (Bedrock)",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelConfig(
                name="anthropic.claude-3-5-sonnet-20241022-v2:0",
                display_name="Claude 3.5 Sonnet (Bedrock)",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
        },
    )


def _openrouter_provider() -> ProviderConfig:
    """Create the default OpenRouter provider configuration.

    OpenRouter exposes an OpenAI-compatible Chat Completions endpoint that
    routes requests to hundreds of underlying models from Anthropic, OpenAI,
    Google, Meta, Mistral, and others.  The gateway uses the OpenAI translator
    for all OpenRouter traffic.

    The OpenRouter-specific headers ``HTTP-Referer`` and ``X-Title`` are
    optional but encouraged; users can set ``OPENROUTER_HTTP_REFERER`` and
    ``OPENROUTER_TITLE`` environment variables if they want them forwarded.
    """
    return ProviderConfig(
        name="openrouter",
        display_name="OpenRouter",
        api_base="https://openrouter.ai/api/v1",
        api_key_env_var="OPENROUTER_API_KEY",
        auth_type=AuthType.BEARER_TOKEN,
        default_model="anthropic/claude-sonnet-4",
        models={
            # Claude models via OpenRouter
            "anthropic/claude-sonnet-4": ModelConfig(
                name="anthropic/claude-sonnet-4",
                display_name="Claude Sonnet 4 (OpenRouter)",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "anthropic/claude-3-5-sonnet": ModelConfig(
                name="anthropic/claude-3-5-sonnet",
                display_name="Claude 3.5 Sonnet (OpenRouter)",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "anthropic/claude-3-5-haiku": ModelConfig(
                name="anthropic/claude-3-5-haiku",
                display_name="Claude 3.5 Haiku (OpenRouter)",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=False,
            ),
            "anthropic/claude-3-opus": ModelConfig(
                name="anthropic/claude-3-opus",
                display_name="Claude 3 Opus (OpenRouter)",
                max_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            # OpenAI models via OpenRouter
            "openai/gpt-4o": ModelConfig(
                name="openai/gpt-4o",
                display_name="GPT-4o (OpenRouter)",
                max_tokens=16384,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            "openai/gpt-4o-mini": ModelConfig(
                name="openai/gpt-4o-mini",
                display_name="GPT-4o Mini (OpenRouter)",
                max_tokens=16384,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            # Google models via OpenRouter
            "google/gemini-2.0-flash": ModelConfig(
                name="google/gemini-2.0-flash",
                display_name="Gemini 2.0 Flash (OpenRouter)",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
            ),
            # Meta models via OpenRouter
            "meta-llama/llama-3.3-70b-instruct": ModelConfig(
                name="meta-llama/llama-3.3-70b-instruct",
                display_name="LLaMA 3.3 70B (OpenRouter)",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=False,
            ),
            # Free tier models for cost-conscious routing
            "mistralai/mistral-7b-instruct:free": ModelConfig(
                name="mistralai/mistral-7b-instruct:free",
                display_name="Mistral 7B (OpenRouter Free)",
                max_tokens=4096,
                supports_streaming=True,
                supports_tools=False,
                supports_vision=False,
            ),
        },
    )


def _local_provider() -> ProviderConfig:
    """Create the default local/Ollama provider configuration."""
    return ProviderConfig(
        name="local",
        display_name="Ollama",
        api_base="http://localhost:11434/v1",
        api_key_env_var="",
        auth_type=AuthType.BEARER_TOKEN,
        default_model="llama3",
        models={
            "llama3": ModelConfig(
                name="llama3",
                display_name="LLaMA 3",
                max_tokens=4096,
                supports_streaming=True,
                supports_tools=False,
                supports_vision=False,
            ),
        },
    )


# Registry of all built-in provider factory functions
_BUILTIN_PROVIDERS: dict[str, Callable[[], ProviderConfig]] = {
    "openai": _openai_provider,
    "anthropic": _anthropic_provider,
    "azure": _azure_openai_provider,
    "google": _google_provider,
    "bedrock": _bedrock_provider,
    "openrouter": _openrouter_provider,
    "local": _local_provider,
}

# Aliases map alternative names to canonical provider names.
# These allow users to run `provider add gemini` instead of `provider add google`.
_PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google",
}


def get_builtin_providers(use_cache: bool = True) -> dict[str, ProviderConfig]:
    """Get all built-in provider configurations.

    When *use_cache* is ``True`` (the default), provider configs are
    returned from an in-memory cache to avoid repeated object construction.

    Args:
        use_cache: Whether to use the in-memory provider cache.

    Returns:
        Dictionary of provider name to ProviderConfig for all built-in providers.
    """
    cache_key = "builtin_providers:all"

    if use_cache:
        from src.cache import get_provider_cache

        cached_value = get_provider_cache().get(cache_key)
        if cached_value is not None:
            return cached_value

    result = {name: factory() for name, factory in _BUILTIN_PROVIDERS.items()}

    if use_cache:
        from src.cache import get_provider_cache

        get_provider_cache().set(cache_key, result)

    return result


def get_builtin_provider(name: str, use_cache: bool = True) -> ProviderConfig | None:
    """Get a specific built-in provider configuration.

    Resolves aliases (e.g. ``"gemini"`` → ``"google"``) transparently.
    When *use_cache* is ``True`` (the default), the provider config is
    returned from an in-memory cache.

    Args:
        name: The provider name or alias (e.g., 'openai', 'gemini').
        use_cache: Whether to use the in-memory provider cache.

    Returns:
        The ProviderConfig if found, None otherwise.
    """
    # Resolve alias first
    canonical_name = _PROVIDER_ALIASES.get(name, name)
    cache_key = f"builtin_provider:{canonical_name}"

    if use_cache:
        from src.cache import get_provider_cache

        cached_value = get_provider_cache().get(cache_key)
        if cached_value is not None:
            return copy.deepcopy(cached_value)

    factory = _BUILTIN_PROVIDERS.get(canonical_name)
    if factory is None:
        return None

    result = factory()

    if use_cache:
        from src.cache import get_provider_cache

        get_provider_cache().set(cache_key, result)

    return copy.deepcopy(result)


def list_builtin_providers(include_aliases: bool = False) -> list[str]:
    """List all available built-in provider names.

    Args:
        include_aliases: If ``True``, also include alias names (e.g. ``"gemini"``
            for ``"google"``).  Defaults to ``False`` so that only canonical
            provider names are returned.

    Returns:
        Sorted list of built-in provider names.
    """
    names = set(_BUILTIN_PROVIDERS.keys())
    if include_aliases:
        names |= set(_PROVIDER_ALIASES.keys())
    return sorted(names)


def create_custom_provider(
    name: str,
    api_base: str,
    api_key_env_var: str = "",
    default_model: str = "",
    display_name: str = "",
) -> ProviderConfig:
    """Create a custom OpenAI-compatible provider configuration.

    This is useful for self-hosted or third-party OpenAI-compatible APIs
    (e.g., vLLM, Ollama, LiteLLM, etc.).

    Args:
        name: Unique identifier for this provider.
        api_base: Base URL for the provider's API.
        api_key_env_var: Environment variable containing the API key.
        default_model: Default model name to use.
        display_name: Human-readable name for the provider.

    Returns:
        A new ProviderConfig for the custom provider.
    """
    return ProviderConfig(
        name=name,
        display_name=display_name or name,
        api_base=api_base,
        api_key_env_var=api_key_env_var,
        auth_type=AuthType.BEARER_TOKEN if api_key_env_var else AuthType.NONE,
        default_model=default_model,
        models={},
    )
