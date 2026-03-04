"""Data models for model provider configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AuthType(str, Enum):
    """Authentication type for a provider."""

    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    NONE = "none"


@dataclass
class ModelConfig:
    """Configuration for a specific model.

    Attributes:
        name: The model identifier (e.g., 'gpt-4', 'claude-3-opus').
        display_name: Human-readable name for the model.
        max_tokens: Maximum token limit for this model.
        supports_streaming: Whether the model supports streaming responses.
        supports_tools: Whether the model supports tool/function calling.
        supports_vision: Whether the model supports vision/image inputs.
        extra: Additional model-specific configuration.
    """

    name: str
    display_name: str = ""
    max_tokens: int = 4096
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set display_name to name if not provided."""
        if not self.display_name:
            self.display_name = self.name

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        data: dict[str, Any] = {
            "name": self.name,
            "max_tokens": self.max_tokens,
            "supports_streaming": self.supports_streaming,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
        }
        if self.display_name != self.name:
            data["display_name"] = self.display_name
        if self.extra:
            data["extra"] = self.extra
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            display_name=data.get("display_name", ""),
            max_tokens=data.get("max_tokens", 4096),
            supports_streaming=data.get("supports_streaming", True),
            supports_tools=data.get("supports_tools", False),
            supports_vision=data.get("supports_vision", False),
            extra=data.get("extra", {}),
        )


@dataclass
class ProviderConfig:
    """Configuration for a model provider.

    Attributes:
        name: Unique identifier for this provider (e.g., 'openai', 'anthropic').
        display_name: Human-readable provider name.
        api_base: Base URL for the provider's API.
        api_key_env_var: Environment variable name containing the API key.
        auth_type: Authentication method used by this provider.
        default_model: Default model to use for this provider.
        models: Dictionary of available models keyed by model name.
        headers: Additional HTTP headers to include in requests.
        extra: Additional provider-specific configuration.
        enabled: Whether this provider is currently enabled.
    """

    name: str
    display_name: str = ""
    api_base: str = ""
    api_key_env_var: str = ""
    auth_type: AuthType = AuthType.API_KEY
    default_model: str = ""
    models: dict[str, ModelConfig] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        """Set display_name to name if not provided."""
        if not self.display_name:
            self.display_name = self.name

    def get_model(self, model_name: Optional[str] = None) -> Optional[ModelConfig]:
        """Get a model config by name, or the default model.

        Args:
            model_name: The model name to look up. If None, returns default.

        Returns:
            The ModelConfig if found, None otherwise.
        """
        name = model_name or self.default_model
        return self.models.get(name)

    def list_models(self) -> list[str]:
        """Return list of available model names."""
        return sorted(self.models.keys())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        data: dict[str, Any] = {
            "name": self.name,
            "api_base": self.api_base,
            "api_key_env_var": self.api_key_env_var,
            "auth_type": self.auth_type.value,
            "default_model": self.default_model,
            "enabled": self.enabled,
        }
        if self.display_name != self.name:
            data["display_name"] = self.display_name
        if self.models:
            data["models"] = {
                name: model.to_dict() for name, model in self.models.items()
            }
        if self.headers:
            data["headers"] = self.headers
        if self.extra:
            data["extra"] = self.extra
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderConfig:
        """Deserialize from dictionary."""
        models = {}
        raw_models = data.get("models", {})
        if isinstance(raw_models, list):
            for model_data in raw_models:
                if isinstance(model_data, dict) and "name" in model_data:
                    models[model_data["name"]] = ModelConfig.from_dict(model_data)
        else:
            for name, model_data in raw_models.items():
                if isinstance(model_data, dict):
                    if "name" not in model_data:
                        model_data["name"] = name
                    models[name] = ModelConfig.from_dict(model_data)

        auth_type_val = data.get("auth_type", "api_key")
        try:
            auth_type = AuthType(auth_type_val)
        except ValueError:
            auth_type = AuthType.API_KEY

        return cls(
            name=data["name"],
            display_name=data.get("display_name", ""),
            api_base=data.get("api_base", ""),
            api_key_env_var=data.get("api_key_env_var", ""),
            auth_type=auth_type,
            default_model=data.get("default_model", ""),
            models=models,
            headers=data.get("headers", {}),
            extra=data.get("extra", {}),
            enabled=data.get("enabled", True),
        )


@dataclass
class GatewayConfig:
    """Top-level gateway configuration.

    Attributes:
        default_provider: Name of the default provider to use.
        providers: Dictionary of configured providers keyed by name.
        log_level: Logging level (debug, info, warning, error).
        timeout: Default request timeout in seconds.
        max_retries: Default number of retries for failed requests.
        extra: Additional gateway-wide configuration.
    """

    default_provider: str = ""
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    log_level: str = "info"
    timeout: int = 30
    max_retries: int = 3
    extra: dict[str, Any] = field(default_factory=dict)

    def get_provider(
        self, provider_name: Optional[str] = None
    ) -> Optional[ProviderConfig]:
        """Get a provider config by name, or the default provider.

        Args:
            provider_name: The provider name to look up. If None, returns default.

        Returns:
            The ProviderConfig if found, None otherwise.
        """
        name = provider_name or self.default_provider
        return self.providers.get(name)

    def get_enabled_providers(self) -> dict[str, ProviderConfig]:
        """Return only enabled providers."""
        return {
            name: provider
            for name, provider in self.providers.items()
            if provider.enabled
        }

    def list_providers(self) -> list[str]:
        """Return list of all provider names."""
        return sorted(self.providers.keys())

    def find_provider_for_model(self, model_name: str) -> Optional[ProviderConfig]:
        """Find which enabled provider owns the given model name.

        Searches all enabled providers for a model matching *model_name*.
        Falls back to the default provider if no enabled provider claims
        the model, so that unknown / custom model IDs are still forwarded
        somewhere useful rather than being dropped.

        Args:
            model_name: The model ID to look up (e.g., ``'gpt-4o'``).

        Returns:
            The :class:`ProviderConfig` that owns the model, or the default
            provider when no enabled provider lists the model.
        """
        for provider in self.get_enabled_providers().values():
            if model_name in provider.models:
                return provider
        # No match found — fall back to the default provider.
        return self.get_provider()

    def add_provider(self, provider: ProviderConfig) -> None:
        """Add or update a provider configuration.

        Args:
            provider: The provider configuration to add.
        """
        self.providers[provider.name] = provider
        if not self.default_provider:
            self.default_provider = provider.name

    def remove_provider(self, name: str) -> bool:
        """Remove a provider by name.

        Args:
            name: The provider name to remove.

        Returns:
            True if the provider was removed, False if not found.
        """
        if name not in self.providers:
            return False
        del self.providers[name]
        if self.default_provider == name:
            remaining = self.list_providers()
            self.default_provider = remaining[0] if remaining else ""
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        data: dict[str, Any] = {
            "default_provider": self.default_provider,
            "log_level": self.log_level,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.providers:
            data["providers"] = {
                name: provider.to_dict()
                for name, provider in self.providers.items()
            }
        if self.extra:
            data["extra"] = self.extra
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GatewayConfig:
        """Deserialize from dictionary."""
        providers = {}
        raw_providers = data.get("providers", {})
        if isinstance(raw_providers, list):
            for provider_data in raw_providers:
                if isinstance(provider_data, dict) and "name" in provider_data:
                    providers[provider_data["name"]] = ProviderConfig.from_dict(provider_data)
        else:
            for name, provider_data in raw_providers.items():
                if isinstance(provider_data, dict):
                    if "name" not in provider_data:
                        provider_data["name"] = name
                    providers[name] = ProviderConfig.from_dict(provider_data)

        return cls(
            default_provider=data.get("default_provider", ""),
            providers=providers,
            log_level=data.get("log_level", "info"),
            timeout=data.get("timeout", 30),
            max_retries=data.get("max_retries", 3),
            extra=data.get("extra", {}),
        )
