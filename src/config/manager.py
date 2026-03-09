"""High-level configuration manager for model providers.

Provides a clean, transactional API for reading and mutating gateway
configuration (providers, models, top-level settings) while handling
all file-system persistence transparently.

Usage example::

    from pathlib import Path
    from src.config.manager import ConfigManager

    mgr = ConfigManager(Path("gateway.yaml"))

    # Provider CRUD
    mgr.add_provider("my-llm", api_base="http://localhost:8000/v1")
    mgr.update_provider("my-llm", default_model="llama3")
    mgr.enable_provider("my-llm")

    # Model CRUD
    mgr.add_model("my-llm", "llama3", max_tokens=4096)
    mgr.set_default_model("my-llm", "llama3")
    mgr.remove_model("my-llm", "llama3")

    # Persist all changes in a single write
    mgr.save()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig


class ConfigManagerError(Exception):
    """Raised when a ConfigManager operation fails."""


class ProviderNotFoundError(ConfigManagerError):
    """Raised when a referenced provider does not exist."""

    def __init__(self, name: str) -> None:
        self.provider_name = name
        super().__init__(f"Provider '{name}' not found.")


class ProviderExistsError(ConfigManagerError):
    """Raised when trying to add a provider that already exists."""

    def __init__(self, name: str) -> None:
        self.provider_name = name
        super().__init__(f"Provider '{name}' already exists.")


class ModelNotFoundError(ConfigManagerError):
    """Raised when a referenced model does not exist on a provider."""

    def __init__(self, provider: str, model: str) -> None:
        self.provider_name = provider
        self.model_name = model
        super().__init__(f"Model '{model}' not found on provider '{provider}'.")


class ModelExistsError(ConfigManagerError):
    """Raised when trying to add a model that already exists on a provider."""

    def __init__(self, provider: str, model: str) -> None:
        self.provider_name = provider
        self.model_name = model
        super().__init__(f"Model '{model}' already exists on provider '{provider}'.")


class ConfigManager:
    """High-level interface for managing gateway configuration.

    Wraps a :class:`~src.models.GatewayConfig` and provides named
    operations for providers and models.  Changes are buffered in memory
    and written to disk only when :meth:`save` is called.

    Attributes:
        path: The config file path used for loading and saving, or ``None``
            when the manager was created from an in-memory config.
        config: The current in-memory :class:`~src.models.GatewayConfig`.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        config: Optional[GatewayConfig] = None,
        auto_load: bool = True,
    ) -> None:
        """Create a ConfigManager.

        Args:
            path: Path to the configuration file.  Used for :meth:`load`
                and :meth:`save`.  May be ``None`` when working with a
                purely in-memory config.
            config: An already-loaded :class:`~src.models.GatewayConfig`.
                When supplied together with *path* the in-memory config is
                used as-is and *auto_load* is ignored.
            auto_load: When ``True`` (default) and *path* is set and no
                *config* was provided, the file is loaded immediately.

        Raises:
            ConfigManagerError: If *auto_load* is ``True`` and the file
                cannot be read or parsed.
        """
        self.path = path
        if config is not None:
            self.config = config
        elif path is not None and auto_load:
            self.config = self._load_from_path(path)
        else:
            self.config = GatewayConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_from_path(path: Path) -> GatewayConfig:
        """Load a GatewayConfig from *path*.

        Args:
            path: Path to a YAML or JSON config file.

        Returns:
            Parsed :class:`~src.models.GatewayConfig`.

        Raises:
            ConfigManagerError: On parse or I/O error.
        """
        from src.config import ConfigError, load_config

        try:
            return load_config(path=path, validate=False, use_cache=False)
        except ConfigError as exc:
            raise ConfigManagerError(f"Failed to load config from {path}: {exc}") from exc

    def _require_provider(self, name: str) -> ProviderConfig:
        """Return a provider by name or raise :exc:`ProviderNotFoundError`.

        Args:
            name: The provider identifier to look up.

        Returns:
            The :class:`~src.models.ProviderConfig` for *name*.

        Raises:
            ProviderNotFoundError: If *name* is not in the config.
        """
        provider = self.config.providers.get(name)
        if provider is None:
            raise ProviderNotFoundError(name)
        return provider

    def _require_model(self, provider_name: str, model_name: str) -> ModelConfig:
        """Return a model from a provider or raise :exc:`ModelNotFoundError`.

        Args:
            provider_name: The provider identifier.
            model_name: The model identifier.

        Returns:
            The :class:`~src.models.ModelConfig` for *model_name*.

        Raises:
            ProviderNotFoundError: If *provider_name* is not in the config.
            ModelNotFoundError: If *model_name* is not in the provider.
        """
        provider = self._require_provider(provider_name)
        model = provider.models.get(model_name)
        if model is None:
            raise ModelNotFoundError(provider_name, model_name)
        return model

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Reload configuration from :attr:`path`.

        Replaces the current in-memory config with the contents of the
        file on disk.  Any unsaved in-memory changes are discarded.

        Raises:
            ConfigManagerError: If :attr:`path` is ``None`` or the file
                cannot be read.
        """
        if self.path is None:
            raise ConfigManagerError("No config file path set; cannot load.")
        self.config = self._load_from_path(self.path)

    def save(self, path: Optional[Path] = None) -> None:
        """Persist the current in-memory config to disk.

        Args:
            path: Override the save destination.  Falls back to
                :attr:`path` when not provided.

        Raises:
            ConfigManagerError: If no path is available or the write fails.
        """
        from src.config import ConfigError, save_config_file

        dst = path or self.path
        if dst is None:
            raise ConfigManagerError("No config file path set; cannot save.")
        try:
            save_config_file(self.config, dst)
        except ConfigError as exc:
            raise ConfigManagerError(f"Failed to save config to {dst}: {exc}") from exc

    # ------------------------------------------------------------------
    # Top-level settings
    # ------------------------------------------------------------------

    def get_setting(self, key: str) -> Any:
        """Get a top-level gateway setting.

        Args:
            key: One of ``default_provider``, ``log_level``, ``timeout``,
                ``max_retries``.

        Returns:
            The current value.

        Raises:
            ConfigManagerError: If *key* is not a valid setting.
        """
        valid = {"default_provider", "log_level", "timeout", "max_retries"}
        if key not in valid:
            raise ConfigManagerError(
                f"Unknown setting '{key}'. Valid keys: {', '.join(sorted(valid))}"
            )
        return getattr(self.config, key)

    def set_setting(self, key: str, value: Any) -> None:
        """Update a top-level gateway setting.

        Args:
            key: One of ``default_provider``, ``log_level``, ``timeout``,
                ``max_retries``.
            value: The new value.  Integer coercion is performed for
                ``timeout`` and ``max_retries``.

        Raises:
            ConfigManagerError: If *key* is not valid or *value* cannot
                be coerced to the expected type.
        """
        valid = {"default_provider", "log_level", "timeout", "max_retries"}
        if key not in valid:
            raise ConfigManagerError(
                f"Unknown setting '{key}'. Valid keys: {', '.join(sorted(valid))}"
            )
        if key in ("timeout", "max_retries"):
            try:
                value = int(value)
            except (TypeError, ValueError) as exc:
                raise ConfigManagerError(
                    f"Setting '{key}' requires an integer value, got {value!r}."
                ) from exc
        setattr(self.config, key, value)

    # ------------------------------------------------------------------
    # Provider operations
    # ------------------------------------------------------------------

    def list_providers(self) -> list[str]:
        """Return all provider names (sorted).

        Returns:
            Sorted list of provider identifiers.
        """
        return self.config.list_providers()

    def get_provider(self, name: str) -> ProviderConfig:
        """Return a provider by name.

        Args:
            name: Provider identifier.

        Returns:
            The :class:`~src.models.ProviderConfig`.

        Raises:
            ProviderNotFoundError: If the provider does not exist.
        """
        return self._require_provider(name)

    def has_provider(self, name: str) -> bool:
        """Return ``True`` if a provider with *name* exists.

        Args:
            name: Provider identifier to check.
        """
        return name in self.config.providers

    def add_provider(
        self,
        name: str,
        *,
        api_base: str = "",
        api_key_env_var: str = "",
        display_name: str = "",
        auth_type: AuthType = AuthType.BEARER_TOKEN,
        default_model: str = "",
        headers: Optional[dict[str, str]] = None,
        extra: Optional[dict[str, Any]] = None,
        enabled: bool = True,
    ) -> ProviderConfig:
        """Add a new provider to the configuration.

        Args:
            name: Unique identifier for the provider.
            api_base: Base URL for the provider's API.
            api_key_env_var: Environment variable holding the API key.
            display_name: Human-readable provider name (defaults to *name*).
            auth_type: Authentication scheme.
            default_model: Default model to use with this provider.
            headers: Additional HTTP request headers.
            extra: Provider-specific extra configuration.
            enabled: Whether to enable the provider immediately.

        Returns:
            The newly created :class:`~src.models.ProviderConfig`.

        Raises:
            ProviderExistsError: If a provider with *name* already exists.
        """
        if name in self.config.providers:
            raise ProviderExistsError(name)
        provider = ProviderConfig(
            name=name,
            display_name=display_name or name,
            api_base=api_base,
            api_key_env_var=api_key_env_var,
            auth_type=auth_type,
            default_model=default_model,
            headers=headers or {},
            extra=extra or {},
            enabled=enabled,
        )
        self.config.add_provider(provider)
        return provider

    def add_provider_from_builtin(
        self,
        builtin_name: str,
        *,
        name: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key_env_var: Optional[str] = None,
        display_name: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> ProviderConfig:
        """Add a provider initialised from a built-in template.

        Args:
            builtin_name: One of ``openai``, ``anthropic``, ``azure``,
                ``google``, ``bedrock``.
            name: Override the provider's identifier (defaults to
                *builtin_name*).
            api_base: Override the built-in API base URL.
            api_key_env_var: Override the built-in env var name.
            display_name: Override the built-in display name.
            default_model: Override the built-in default model.

        Returns:
            The newly created :class:`~src.models.ProviderConfig`.

        Raises:
            ConfigManagerError: If *builtin_name* is not recognised.
            ProviderExistsError: If a provider with the resolved *name*
                already exists.
        """
        from src.providers import get_builtin_provider

        template = get_builtin_provider(builtin_name, use_cache=False)
        if template is None:
            raise ConfigManagerError(
                f"Unknown built-in provider '{builtin_name}'. "
                "Use list_builtin_providers() to see available names."
            )

        resolved_name = name or builtin_name
        if resolved_name in self.config.providers:
            raise ProviderExistsError(resolved_name)

        template.name = resolved_name
        if api_base:
            template.api_base = api_base
        if api_key_env_var:
            template.api_key_env_var = api_key_env_var
        if display_name:
            template.display_name = display_name
        if default_model:
            template.default_model = default_model

        self.config.add_provider(template)
        return template

    def update_provider(
        self,
        name: str,
        *,
        api_base: Optional[str] = None,
        api_key_env_var: Optional[str] = None,
        display_name: Optional[str] = None,
        auth_type: Optional[AuthType] = None,
        default_model: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> ProviderConfig:
        """Update properties of an existing provider.

        Only the keyword arguments that are explicitly passed (not ``None``)
        are modified; others are left unchanged.

        Args:
            name: Identifier of the provider to update.
            api_base: New API base URL.
            api_key_env_var: New API key environment variable name.
            display_name: New human-readable name.
            auth_type: New authentication scheme.
            default_model: New default model.
            headers: Replacement headers dict (replaces existing headers).
            extra: Replacement extra dict (replaces existing extra).

        Returns:
            The updated :class:`~src.models.ProviderConfig`.

        Raises:
            ProviderNotFoundError: If *name* does not exist.
        """
        provider = self._require_provider(name)
        if api_base is not None:
            provider.api_base = api_base
        if api_key_env_var is not None:
            provider.api_key_env_var = api_key_env_var
        if display_name is not None:
            provider.display_name = display_name
        if auth_type is not None:
            provider.auth_type = auth_type
        if default_model is not None:
            provider.default_model = default_model
        if headers is not None:
            provider.headers = headers
        if extra is not None:
            provider.extra = extra
        return provider

    def remove_provider(self, name: str) -> None:
        """Remove a provider from the configuration.

        If the removed provider was the default, the next available provider
        (alphabetically) is promoted to default automatically.

        Args:
            name: Provider identifier to remove.

        Raises:
            ProviderNotFoundError: If *name* does not exist.
        """
        if not self.config.remove_provider(name):
            raise ProviderNotFoundError(name)

    def enable_provider(self, name: str) -> None:
        """Enable a provider.

        Args:
            name: Provider identifier.

        Raises:
            ProviderNotFoundError: If *name* does not exist.
        """
        self._require_provider(name).enabled = True

    def disable_provider(self, name: str) -> None:
        """Disable a provider.

        Args:
            name: Provider identifier.

        Raises:
            ProviderNotFoundError: If *name* does not exist.
        """
        self._require_provider(name).enabled = False

    def set_default_provider(self, name: str) -> None:
        """Set the default provider.

        Args:
            name: Provider identifier to make the default.

        Raises:
            ProviderNotFoundError: If *name* does not exist.
        """
        self._require_provider(name)  # ensure it exists
        self.config.default_provider = name

    # ------------------------------------------------------------------
    # Model operations
    # ------------------------------------------------------------------

    def list_models(self, provider_name: str) -> list[str]:
        """Return model names for a provider (sorted).

        Args:
            provider_name: Provider identifier.

        Returns:
            Sorted list of model identifiers.

        Raises:
            ProviderNotFoundError: If *provider_name* does not exist.
        """
        return self._require_provider(provider_name).list_models()

    def get_model(self, provider_name: str, model_name: str) -> ModelConfig:
        """Return a model config from a provider.

        Args:
            provider_name: Provider identifier.
            model_name: Model identifier.

        Returns:
            The :class:`~src.models.ModelConfig`.

        Raises:
            ProviderNotFoundError: If *provider_name* does not exist.
            ModelNotFoundError: If *model_name* does not exist on the provider.
        """
        return self._require_model(provider_name, model_name)

    def has_model(self, provider_name: str, model_name: str) -> bool:
        """Return ``True`` if *model_name* exists on *provider_name*.

        Args:
            provider_name: Provider identifier.
            model_name: Model identifier.

        Raises:
            ProviderNotFoundError: If *provider_name* does not exist.
        """
        return model_name in self._require_provider(provider_name).models

    def add_model(
        self,
        provider_name: str,
        model_name: str,
        *,
        display_name: str = "",
        max_tokens: int = 4096,
        supports_streaming: bool = True,
        supports_tools: bool = False,
        supports_vision: bool = False,
        extra: Optional[dict[str, Any]] = None,
    ) -> ModelConfig:
        """Add a new model to a provider.

        Args:
            provider_name: Provider to add the model to.
            model_name: Unique model identifier within the provider.
            display_name: Human-readable model name (defaults to *model_name*).
            max_tokens: Maximum output token limit.
            supports_streaming: Whether the model supports SSE streaming.
            supports_tools: Whether the model supports function/tool calling.
            supports_vision: Whether the model accepts image inputs.
            extra: Model-specific extra configuration.

        Returns:
            The newly created :class:`~src.models.ModelConfig`.

        Raises:
            ProviderNotFoundError: If *provider_name* does not exist.
            ModelExistsError: If *model_name* already exists on the provider.
        """
        provider = self._require_provider(provider_name)
        if model_name in provider.models:
            raise ModelExistsError(provider_name, model_name)
        model = ModelConfig(
            name=model_name,
            display_name=display_name or model_name,
            max_tokens=max_tokens,
            supports_streaming=supports_streaming,
            supports_tools=supports_tools,
            supports_vision=supports_vision,
            extra=extra or {},
        )
        provider.models[model_name] = model
        # Auto-set as default if the provider has no default model yet.
        if not provider.default_model:
            provider.default_model = model_name
        return model

    def update_model(
        self,
        provider_name: str,
        model_name: str,
        *,
        display_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        supports_streaming: Optional[bool] = None,
        supports_tools: Optional[bool] = None,
        supports_vision: Optional[bool] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> ModelConfig:
        """Update properties of an existing model.

        Only keyword arguments that are explicitly passed (not ``None``)
        are modified.

        Args:
            provider_name: Provider identifier.
            model_name: Model identifier.
            display_name: New human-readable name.
            max_tokens: New token limit.
            supports_streaming: New streaming capability flag.
            supports_tools: New tools capability flag.
            supports_vision: New vision capability flag.
            extra: Replacement extra dict.

        Returns:
            The updated :class:`~src.models.ModelConfig`.

        Raises:
            ProviderNotFoundError: If *provider_name* does not exist.
            ModelNotFoundError: If *model_name* does not exist on the provider.
        """
        model = self._require_model(provider_name, model_name)
        if display_name is not None:
            model.display_name = display_name
        if max_tokens is not None:
            model.max_tokens = max_tokens
        if supports_streaming is not None:
            model.supports_streaming = supports_streaming
        if supports_tools is not None:
            model.supports_tools = supports_tools
        if supports_vision is not None:
            model.supports_vision = supports_vision
        if extra is not None:
            model.extra = extra
        return model

    def remove_model(self, provider_name: str, model_name: str) -> None:
        """Remove a model from a provider.

        If the removed model was the provider's default, the default is
        cleared (set to ``""``).

        Args:
            provider_name: Provider identifier.
            model_name: Model identifier to remove.

        Raises:
            ProviderNotFoundError: If *provider_name* does not exist.
            ModelNotFoundError: If *model_name* does not exist on the provider.
        """
        provider = self._require_provider(provider_name)
        if model_name not in provider.models:
            raise ModelNotFoundError(provider_name, model_name)
        del provider.models[model_name]
        if provider.default_model == model_name:
            remaining = provider.list_models()
            provider.default_model = remaining[0] if remaining else ""

    def set_default_model(self, provider_name: str, model_name: str) -> None:
        """Set the default model for a provider.

        Args:
            provider_name: Provider identifier.
            model_name: Model identifier to make the default.

        Raises:
            ProviderNotFoundError: If *provider_name* does not exist.
            ModelNotFoundError: If *model_name* does not exist on the provider.
        """
        self._require_model(provider_name, model_name)  # ensure it exists
        self._require_provider(provider_name).default_model = model_name

    # ------------------------------------------------------------------
    # Class-level factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_default(cls) -> "ConfigManager":
        """Create a ConfigManager pre-loaded with all built-in providers.

        The manager has no file path and changes must be saved explicitly
        by passing a *path* to :meth:`save`.

        Returns:
            A :class:`ConfigManager` wrapping the default
            :class:`~src.models.GatewayConfig`.
        """
        from src.config import get_default_config

        return cls(config=get_default_config())

    @classmethod
    def from_file(cls, path: Path) -> "ConfigManager":
        """Create a ConfigManager by loading an existing file.

        Args:
            path: Path to a YAML or JSON config file.

        Returns:
            A :class:`ConfigManager` backed by *path*.

        Raises:
            ConfigManagerError: If the file cannot be loaded.
        """
        return cls(path=path, auto_load=True)
