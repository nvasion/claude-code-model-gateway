"""API request routing engine for the model gateway.

Provides configurable request routing that maps incoming API requests
to the appropriate upstream provider based on model name, request headers,
path patterns, routing rules, and provider health. Supports multiple
routing strategies including model-based, round-robin, priority-based,
and weighted routing.

Typical usage:

    from src.router import Router, RouteRule, RequestContext

    # Create router from gateway config
    router = Router.from_config(gateway_config)

    # Add custom routing rules
    router.add_rule(RouteRule(
        pattern="claude-*",
        provider="anthropic",
        priority=10,
    ))

    # Route a request
    ctx = RequestContext(method="POST", path="/v1/messages", model="claude-sonnet-4-20250514")
    match = router.resolve(ctx)
    print(match.provider_name)  # "anthropic"
"""

from __future__ import annotations

import fnmatch
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.errors import (
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    GatewayError,
)
from src.logging_config import get_logger
from src.models import GatewayConfig, ProviderConfig

logger = get_logger("router")


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class RoutingStrategy(str, Enum):
    """Strategy for selecting a provider when multiple match a request."""

    MODEL_BASED = "model_based"
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"
    WEIGHTED = "weighted"
    SINGLE = "single"


class RouteRuleType(str, Enum):
    """Type of pattern matching used in a routing rule."""

    MODEL_PATTERN = "model_pattern"
    PATH_PATTERN = "path_pattern"
    HEADER_MATCH = "header_match"
    CATCH_ALL = "catch_all"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class RoutingError(GatewayError):
    """No suitable provider could be found for a request.

    This is a non-retryable error: re-sending the exact same request
    will produce the same routing failure.
    """

    def __init__(
        self,
        message: str,
        *,
        model: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        details: dict[str, Any] = {}
        if path:
            details["path"] = path
        ctx = ErrorContext(
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.MEDIUM,
            retryable=False,
            model=model,
            details=details,
        )
        super().__init__(message, context=ctx)


class NoRouteError(RoutingError):
    """No route matched the incoming request."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        parts = ["No route found"]
        if model:
            parts.append(f"for model '{model}'")
        if path:
            parts.append(f"on path '{path}'")
        super().__init__(" ".join(parts), model=model, path=path)


class ProviderDisabledError(RoutingError):
    """The resolved provider is disabled."""

    def __init__(self, provider: str) -> None:
        super().__init__(f"Provider '{provider}' is disabled")
        self.context.provider = provider


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class RequestContext:
    """Captured metadata from an incoming API request used for routing.

    Attributes:
        method: HTTP method (GET, POST, etc.).
        path: Request path (e.g., ``/v1/messages``).
        model: Model name extracted from the request body or header.
        headers: Dictionary of request headers (lowercased keys).
        body: Parsed request body (usually a dict from JSON).
        client_ip: IP address of the client.
        timestamp: When the request was received.
    """

    method: str = "POST"
    path: str = ""
    model: Optional[str] = None
    headers: dict[str, str] = field(default_factory=dict)
    body: Optional[dict[str, Any]] = None
    client_ip: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def get_header(self, name: str, default: str = "") -> str:
        """Get a header value by name (case-sensitive; keys must be pre-lowered).

        Args:
            name: Header name. Keys in the headers dict must already be
                lowercase; this method does not normalise the lookup key.
            default: Default value if header is not present.

        Returns:
            The header value or *default*.
        """
        return self.headers.get(name, default)


@dataclass
class RouteRule:
    """A single routing rule that maps requests to a provider.

    Rules are evaluated in priority order (higher priority first). The
    first matching rule wins.

    Attributes:
        pattern: Pattern to match against (glob for models, regex for
            paths, exact for headers). Meaning depends on *rule_type*.
        provider: Name of the target provider.
        rule_type: What kind of matching this rule performs.
        priority: Higher values are evaluated first (default 0).
        header_name: For header_match rules, which header to check.
        header_value: For header_match rules, expected header value.
        weight: For weighted routing, relative weight (default 1).
        enabled: Whether this rule is active.
        name: Optional human-readable name for this rule.
        description: Optional description of the rule's purpose.
    """

    pattern: str = "*"
    provider: str = ""
    rule_type: RouteRuleType = RouteRuleType.MODEL_PATTERN
    priority: int = 0
    header_name: str = ""
    header_value: str = ""
    weight: int = 1
    enabled: bool = True
    name: str = ""
    description: str = ""

    def matches(self, ctx: RequestContext) -> bool:
        """Test whether this rule matches the given request context.

        Args:
            ctx: The incoming request context.

        Returns:
            True if this rule matches.
        """
        if not self.enabled:
            return False

        if self.rule_type == RouteRuleType.MODEL_PATTERN:
            if ctx.model is None:
                return False
            return fnmatch.fnmatch(ctx.model, self.pattern)

        if self.rule_type == RouteRuleType.PATH_PATTERN:
            if not ctx.path:
                return False
            return fnmatch.fnmatch(ctx.path, self.pattern)

        if self.rule_type == RouteRuleType.HEADER_MATCH:
            if not self.header_name:
                return False
            actual = ctx.get_header(self.header_name)
            if self.header_value:
                return actual == self.header_value
            # If no value specified, just check presence
            return bool(actual)

        if self.rule_type == RouteRuleType.CATCH_ALL:
            return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        data: dict[str, Any] = {
            "pattern": self.pattern,
            "provider": self.provider,
            "rule_type": self.rule_type.value,
            "priority": self.priority,
            "weight": self.weight,
            "enabled": self.enabled,
        }
        if self.header_name:
            data["header_name"] = self.header_name
        if self.header_value:
            data["header_value"] = self.header_value
        if self.name:
            data["name"] = self.name
        if self.description:
            data["description"] = self.description
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouteRule:
        """Deserialize from dictionary."""
        rule_type_str = data.get("rule_type", "model_pattern")
        try:
            rule_type = RouteRuleType(rule_type_str)
        except ValueError:
            rule_type = RouteRuleType.MODEL_PATTERN

        return cls(
            pattern=data.get("pattern", "*"),
            provider=data.get("provider", ""),
            rule_type=rule_type,
            priority=data.get("priority", 0),
            header_name=data.get("header_name", ""),
            header_value=data.get("header_value", ""),
            weight=data.get("weight", 1),
            enabled=data.get("enabled", True),
            name=data.get("name", ""),
            description=data.get("description", ""),
        )


@dataclass
class RouteMatch:
    """Result of a successful route resolution.

    Attributes:
        provider_name: Name of the matched provider.
        provider_config: The full ProviderConfig for the provider.
        rule: The rule that produced this match (None for auto-routing).
        model: The resolved model name.
        strategy: The routing strategy that was used.
        resolution_time_ms: Time spent resolving the route.
        fallback_used: Whether a fallback provider was used.
        metadata: Additional routing metadata.
    """

    provider_name: str
    provider_config: ProviderConfig
    rule: Optional[RouteRule] = None
    model: Optional[str] = None
    strategy: RoutingStrategy = RoutingStrategy.MODEL_BASED
    resolution_time_ms: float = 0.0
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        data: dict[str, Any] = {
            "provider_name": self.provider_name,
            "model": self.model,
            "strategy": self.strategy.value,
            "resolution_time_ms": round(self.resolution_time_ms, 3),
            "fallback_used": self.fallback_used,
        }
        if self.rule:
            data["rule"] = self.rule.to_dict()
        if self.metadata:
            data["metadata"] = self.metadata
        return data


# --------------------------------------------------------------------------- #
# Model-to-provider mapping
# --------------------------------------------------------------------------- #


def build_model_provider_map(
    config: GatewayConfig,
) -> dict[str, str]:
    """Build a mapping from model names to provider names.

    Scans all enabled providers in the config and creates a flat
    model-name -> provider-name lookup table.

    Args:
        config: The gateway configuration.

    Returns:
        Dictionary mapping model names to provider names.
    """
    mapping: dict[str, str] = {}
    for provider_name, provider in config.providers.items():
        if not provider.enabled:
            continue
        for model_name in provider.models:
            # First provider to claim a model wins
            if model_name not in mapping:
                mapping[model_name] = provider_name
    return mapping


def find_provider_for_model(
    model: str,
    config: GatewayConfig,
) -> Optional[str]:
    """Find the provider that supports a given model.

    First checks for an exact model name match across all enabled
    providers. If no exact match, tries glob-style matching against
    model names.

    Args:
        model: The model name to look up.
        config: The gateway configuration.

    Returns:
        Provider name if found, None otherwise.
    """
    # Exact match
    for provider_name, provider in config.providers.items():
        if not provider.enabled:
            continue
        if model in provider.models:
            return provider_name

    # Glob/prefix match (e.g., "claude-*" matches in anthropic)
    for provider_name, provider in config.providers.items():
        if not provider.enabled:
            continue
        for model_name in provider.models:
            # Check if model is a prefix match
            if model.startswith(model_name.split("-")[0]):
                return provider_name

    return None


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


class Router:
    """API request routing engine.

    Maintains a set of routing rules and resolves incoming requests
    to the appropriate upstream provider. Supports model-based lookup,
    rule-based routing, round-robin selection, and fallback chains.

    Thread-safe: all mutable state is protected by a lock.

    Args:
        config: Gateway configuration with provider definitions.
        strategy: Default routing strategy.
        rules: Initial routing rules.
        fallback_provider: Provider to use when no rule matches.
    """

    def __init__(
        self,
        config: GatewayConfig,
        strategy: RoutingStrategy = RoutingStrategy.MODEL_BASED,
        rules: Optional[list[RouteRule]] = None,
        fallback_provider: Optional[str] = None,
    ) -> None:
        self._config = config
        self._strategy = strategy
        # Sort rules by priority descending so higher priority rules are checked first
        self._rules: list[RouteRule] = sorted(
            list(rules) if rules else [], key=lambda r: r.priority, reverse=True
        )
        self._fallback_provider = fallback_provider or config.default_provider
        self._model_map: dict[str, str] = build_model_provider_map(config)
        self._lock = threading.Lock()

        # Round-robin state
        self._rr_index = 0
        self._rr_providers: list[str] = sorted(
            config.get_enabled_providers().keys()
        )

        # Statistics
        self._stats = RouterStats()

        logger.info(
            "Router initialised: strategy=%s, rules=%d, providers=%d, "
            "models_mapped=%d, fallback=%s",
            strategy.value,
            len(self._rules),
            len(self._rr_providers),
            len(self._model_map),
            self._fallback_provider or "(none)",
        )

    # -- Properties --------------------------------------------------------- #

    @property
    def strategy(self) -> RoutingStrategy:
        """Current default routing strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: RoutingStrategy) -> None:
        self._strategy = value

    @property
    def fallback_provider(self) -> Optional[str]:
        """Name of the fallback provider."""
        return self._fallback_provider

    @fallback_provider.setter
    def fallback_provider(self, value: Optional[str]) -> None:
        self._fallback_provider = value

    @property
    def rules(self) -> list[RouteRule]:
        """Return a copy of the current routing rules."""
        with self._lock:
            return list(self._rules)

    @property
    def model_map(self) -> dict[str, str]:
        """Return a copy of the model-to-provider map."""
        with self._lock:
            return dict(self._model_map)

    # -- Rule management ---------------------------------------------------- #

    def add_rule(self, rule: RouteRule) -> None:
        """Add a routing rule.

        Rules are kept sorted by priority (descending) so the highest
        priority rule is evaluated first.

        Args:
            rule: The routing rule to add.
        """
        with self._lock:
            self._rules.append(rule)
            self._rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(
            "Added route rule: %s -> %s (priority=%d, type=%s)",
            rule.pattern,
            rule.provider,
            rule.priority,
            rule.rule_type.value,
        )

    def remove_rule(self, pattern: str, provider: str) -> bool:
        """Remove a routing rule by pattern and provider.

        Args:
            pattern: The rule pattern to match.
            provider: The rule provider to match.

        Returns:
            True if a rule was removed.
        """
        with self._lock:
            before = len(self._rules)
            self._rules = [
                r
                for r in self._rules
                if not (r.pattern == pattern and r.provider == provider)
            ]
            removed = len(self._rules) < before
        if removed:
            logger.info("Removed route rule: %s -> %s", pattern, provider)
        return removed

    def clear_rules(self) -> int:
        """Remove all routing rules.

        Returns:
            Number of rules removed.
        """
        with self._lock:
            count = len(self._rules)
            self._rules.clear()
        logger.info("Cleared %d route rules", count)
        return count

    # -- Model map management ----------------------------------------------- #

    def refresh_model_map(self) -> None:
        """Rebuild the model-to-provider map from current config."""
        with self._lock:
            self._model_map = build_model_provider_map(self._config)
            self._rr_providers = sorted(
                self._config.get_enabled_providers().keys()
            )
        logger.debug(
            "Refreshed model map: %d models, %d providers",
            len(self._model_map),
            len(self._rr_providers),
        )

    def update_config(self, config: GatewayConfig) -> None:
        """Update the gateway configuration and refresh internal state.

        Args:
            config: New gateway configuration.
        """
        with self._lock:
            self._config = config
            self._model_map = build_model_provider_map(config)
            self._rr_providers = sorted(
                config.get_enabled_providers().keys()
            )
            if not self._fallback_provider:
                self._fallback_provider = config.default_provider
        logger.info("Router config updated: %d providers", len(self._rr_providers))

    # -- Core resolution ---------------------------------------------------- #

    def resolve(self, ctx: RequestContext) -> RouteMatch:
        """Resolve a request context to a provider.

        Evaluation order:
        1. Explicit routing rules (sorted by priority)
        2. Model-based lookup (model name -> provider)
        3. Strategy-based selection (round-robin, default, etc.)
        4. Fallback provider

        Args:
            ctx: The incoming request context.

        Returns:
            A RouteMatch describing where to send the request.

        Raises:
            NoRouteError: If no provider could be found.
            ProviderDisabledError: If the resolved provider is disabled.
        """
        start = time.monotonic()

        with self._lock:
            # Phase 1: Check explicit rules
            match = self._match_rules(ctx)
            if match is not None:
                match.resolution_time_ms = (time.monotonic() - start) * 1000.0
                self._stats.rule_matches += 1
                self._stats.total_requests += 1
                logger.debug(
                    "Route resolved by rule: %s -> %s (%.2fms)",
                    ctx.model or ctx.path,
                    match.provider_name,
                    match.resolution_time_ms,
                )
                return match

            # Phase 2: Model-based lookup
            if ctx.model:
                match = self._match_by_model(ctx)
                if match is not None:
                    match.resolution_time_ms = (time.monotonic() - start) * 1000.0
                    self._stats.model_matches += 1
                    self._stats.total_requests += 1
                    logger.debug(
                        "Route resolved by model map: %s -> %s (%.2fms)",
                        ctx.model,
                        match.provider_name,
                        match.resolution_time_ms,
                    )
                    return match

            # Phase 3: Strategy-based selection
            match = self._match_by_strategy(ctx)
            if match is not None:
                match.resolution_time_ms = (time.monotonic() - start) * 1000.0
                self._stats.strategy_matches += 1
                self._stats.total_requests += 1
                logger.debug(
                    "Route resolved by strategy (%s): -> %s (%.2fms)",
                    self._strategy.value,
                    match.provider_name,
                    match.resolution_time_ms,
                )
                return match

            # Phase 4: Fallback
            match = self._match_fallback(ctx)
            if match is not None:
                match.resolution_time_ms = (time.monotonic() - start) * 1000.0
                match.fallback_used = True
                self._stats.fallback_matches += 1
                self._stats.total_requests += 1
                logger.debug(
                    "Route resolved by fallback: -> %s (%.2fms)",
                    match.provider_name,
                    match.resolution_time_ms,
                )
                return match

        # No route found
        self._stats.no_route += 1
        self._stats.total_requests += 1
        raise NoRouteError(model=ctx.model, path=ctx.path)

    def _match_rules(self, ctx: RequestContext) -> Optional[RouteMatch]:
        """Try to match the request against explicit routing rules.

        Rules are already sorted by priority (descending).
        """
        for rule in self._rules:
            if rule.matches(ctx):
                provider = self._config.get_provider(rule.provider)
                if provider is None:
                    logger.warning(
                        "Rule matched provider '%s' but it is not configured",
                        rule.provider,
                    )
                    continue
                if not provider.enabled:
                    logger.debug(
                        "Rule matched disabled provider '%s', skipping",
                        rule.provider,
                    )
                    continue
                return RouteMatch(
                    provider_name=rule.provider,
                    provider_config=provider,
                    rule=rule,
                    model=ctx.model,
                    strategy=RoutingStrategy.PRIORITY,
                )
        return None

    def _match_by_model(self, ctx: RequestContext) -> Optional[RouteMatch]:
        """Try to match by model name in the model-provider map."""
        if not ctx.model:
            return None

        provider_name = self._model_map.get(ctx.model)
        if provider_name is None:
            # Try glob-style matching against known model patterns
            provider_name = self._glob_match_model(ctx.model)

        if provider_name is None:
            return None

        provider = self._config.get_provider(provider_name)
        if provider is None or not provider.enabled:
            return None

        return RouteMatch(
            provider_name=provider_name,
            provider_config=provider,
            model=ctx.model,
            strategy=RoutingStrategy.MODEL_BASED,
        )

    def _glob_match_model(self, model: str) -> Optional[str]:
        """Try matching a model name using prefix/glob against provider models."""
        for provider_name, provider in self._config.providers.items():
            if not provider.enabled:
                continue
            for known_model in provider.models:
                # Check if one is a prefix of the other
                if model.startswith(known_model) or known_model.startswith(model):
                    return provider_name
        return None

    def _match_by_strategy(self, ctx: RequestContext) -> Optional[RouteMatch]:
        """Select a provider using the configured routing strategy."""
        if not self._rr_providers:
            return None

        if self._strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(ctx)
        elif self._strategy == RoutingStrategy.PRIORITY:
            return self._priority_select(ctx)
        elif self._strategy == RoutingStrategy.SINGLE:
            return self._single_select(ctx)
        elif self._strategy == RoutingStrategy.MODEL_BASED:
            # Model-based routing already handled in Phase 2; no match here
            return None

        # Default: use the default provider
        return self._single_select(ctx)

    def _round_robin_select(self, ctx: RequestContext) -> Optional[RouteMatch]:
        """Select the next provider in round-robin order."""
        if not self._rr_providers:
            return None

        provider_name = self._rr_providers[self._rr_index % len(self._rr_providers)]
        self._rr_index += 1

        provider = self._config.get_provider(provider_name)
        if provider is None or not provider.enabled:
            return None

        return RouteMatch(
            provider_name=provider_name,
            provider_config=provider,
            model=ctx.model,
            strategy=RoutingStrategy.ROUND_ROBIN,
            metadata={"rr_index": self._rr_index - 1},
        )

    def _priority_select(self, ctx: RequestContext) -> Optional[RouteMatch]:
        """Select the first enabled provider (alphabetical order)."""
        for provider_name in self._rr_providers:
            provider = self._config.get_provider(provider_name)
            if provider and provider.enabled:
                return RouteMatch(
                    provider_name=provider_name,
                    provider_config=provider,
                    model=ctx.model,
                    strategy=RoutingStrategy.PRIORITY,
                )
        return None

    def _single_select(self, ctx: RequestContext) -> Optional[RouteMatch]:
        """Select the default provider."""
        default_name = self._config.default_provider
        if not default_name:
            return None

        provider = self._config.get_provider(default_name)
        if provider is None or not provider.enabled:
            return None

        return RouteMatch(
            provider_name=default_name,
            provider_config=provider,
            model=ctx.model,
            strategy=RoutingStrategy.SINGLE,
        )

    def _match_fallback(self, ctx: RequestContext) -> Optional[RouteMatch]:
        """Try the fallback provider."""
        if not self._fallback_provider:
            return None

        provider = self._config.get_provider(self._fallback_provider)
        if provider is None or not provider.enabled:
            return None

        return RouteMatch(
            provider_name=self._fallback_provider,
            provider_config=provider,
            model=ctx.model,
            strategy=self._strategy,
        )

    # -- Statistics --------------------------------------------------------- #

    def get_stats(self) -> dict[str, Any]:
        """Return routing statistics."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset routing statistics to zero."""
        self._stats = RouterStats()

    # -- Factory ------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        config: GatewayConfig,
        strategy: RoutingStrategy = RoutingStrategy.MODEL_BASED,
        rules: Optional[list[RouteRule]] = None,
    ) -> Router:
        """Create a Router from a gateway configuration.

        If the configuration contains routing rules in ``config.extra``,
        they are loaded automatically.

        Args:
            config: Gateway configuration.
            strategy: Default routing strategy.
            rules: Additional routing rules.

        Returns:
            A configured Router instance.
        """
        all_rules = list(rules) if rules else []

        # Load rules from config extra
        config_rules = config.extra.get("routing_rules", [])
        for rule_data in config_rules:
            if isinstance(rule_data, dict):
                all_rules.append(RouteRule.from_dict(rule_data))

        fallback = config.extra.get("fallback_provider", config.default_provider)

        strategy_str = config.extra.get("routing_strategy")
        if strategy_str:
            try:
                strategy = RoutingStrategy(strategy_str)
            except ValueError:
                logger.warning(
                    "Unknown routing strategy '%s' in config, using %s",
                    strategy_str,
                    strategy.value,
                )

        return cls(
            config=config,
            strategy=strategy,
            rules=all_rules,
            fallback_provider=fallback,
        )

    def __repr__(self) -> str:
        return (
            f"Router(strategy={self._strategy.value}, "
            f"rules={len(self._rules)}, "
            f"providers={len(self._rr_providers)}, "
            f"models={len(self._model_map)})"
        )


# --------------------------------------------------------------------------- #
# Router statistics
# --------------------------------------------------------------------------- #


@dataclass
class RouterStats:
    """Statistics gathered by the router.

    Attributes:
        total_requests: Total number of routing requests.
        rule_matches: Routes resolved by explicit rules.
        model_matches: Routes resolved by model-name lookup.
        strategy_matches: Routes resolved by routing strategy.
        fallback_matches: Routes resolved by fallback provider.
        no_route: Requests that could not be routed.
    """

    total_requests: int = 0
    rule_matches: int = 0
    model_matches: int = 0
    strategy_matches: int = 0
    fallback_matches: int = 0
    no_route: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_requests": self.total_requests,
            "rule_matches": self.rule_matches,
            "model_matches": self.model_matches,
            "strategy_matches": self.strategy_matches,
            "fallback_matches": self.fallback_matches,
            "no_route": self.no_route,
        }


# --------------------------------------------------------------------------- #
# Helper: extract model from request body
# --------------------------------------------------------------------------- #


def extract_model_from_body(body: Optional[dict[str, Any]]) -> Optional[str]:
    """Extract the model name from a parsed request body.

    Supports Anthropic (``model`` field) and OpenAI (``model`` field)
    request formats.

    Args:
        body: Parsed JSON request body.

    Returns:
        Model name if found, None otherwise.
    """
    if body is None:
        return None
    model = body.get("model")
    if isinstance(model, str) and model:
        return model
    return None


def extract_model_from_path(path: str) -> Optional[str]:
    """Extract a model name from a request path.

    Handles paths like ``/v1/models/claude-sonnet-4-20250514``.

    Args:
        path: The request path.

    Returns:
        Model name if found, None otherwise.
    """
    parts = path.rstrip("/").split("/")
    if len(parts) >= 3 and parts[-2] == "models":
        return parts[-1]
    return None
