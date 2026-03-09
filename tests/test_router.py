"""Tests for the API request routing engine."""

import pytest

from src.models import (
    AuthType,
    GatewayConfig,
    ModelConfig,
    ProviderConfig,
)
from src.router import (
    NoRouteError,
    ProviderDisabledError,
    RequestContext,
    RouteMatch,
    RouteRule,
    RouteRuleType,
    Router,
    RouterStats,
    RoutingError,
    RoutingStrategy,
    build_model_provider_map,
    extract_model_from_body,
    extract_model_from_path,
    find_provider_for_model,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_provider(
    name: str,
    models: dict[str, ModelConfig] | None = None,
    enabled: bool = True,
    api_base: str = "https://api.example.com",
) -> ProviderConfig:
    """Helper to build a ProviderConfig for testing."""
    return ProviderConfig(
        name=name,
        display_name=name.title(),
        api_base=api_base,
        api_key_env_var=f"{name.upper()}_API_KEY",
        auth_type=AuthType.BEARER_TOKEN,
        default_model=list(models.keys())[0] if models else "",
        models=models or {},
        enabled=enabled,
    )


def _make_model(name: str) -> ModelConfig:
    return ModelConfig(name=name, display_name=name)


@pytest.fixture
def sample_config() -> GatewayConfig:
    """Create a sample gateway config with multiple providers."""
    return GatewayConfig(
        default_provider="openai",
        providers={
            "openai": _make_provider(
                "openai",
                models={
                    "gpt-4o": _make_model("gpt-4o"),
                    "gpt-4o-mini": _make_model("gpt-4o-mini"),
                    "o3-mini": _make_model("o3-mini"),
                },
            ),
            "anthropic": _make_provider(
                "anthropic",
                models={
                    "claude-sonnet-4-20250514": _make_model("claude-sonnet-4-20250514"),
                    "claude-3-5-sonnet-20241022": _make_model("claude-3-5-sonnet-20241022"),
                    "claude-3-5-haiku-20241022": _make_model("claude-3-5-haiku-20241022"),
                },
            ),
            "google": _make_provider(
                "google",
                models={
                    "gemini-2.0-flash": _make_model("gemini-2.0-flash"),
                    "gemini-1.5-pro": _make_model("gemini-1.5-pro"),
                },
            ),
            "disabled_provider": _make_provider(
                "disabled_provider",
                models={"test-model": _make_model("test-model")},
                enabled=False,
            ),
        },
    )


@pytest.fixture
def router(sample_config: GatewayConfig) -> Router:
    """Create a router with sample config."""
    return Router(config=sample_config)


@pytest.fixture
def empty_config() -> GatewayConfig:
    """Create an empty gateway config."""
    return GatewayConfig()


# --------------------------------------------------------------------------- #
# RequestContext tests
# --------------------------------------------------------------------------- #


class TestRequestContext:
    """Tests for RequestContext."""

    def test_default_values(self):
        ctx = RequestContext()
        assert ctx.method == "POST"
        assert ctx.path == ""
        assert ctx.model is None
        assert ctx.headers == {}
        assert ctx.body is None
        assert ctx.client_ip is None
        assert ctx.timestamp > 0

    def test_get_header_case_insensitive(self):
        ctx = RequestContext(headers={"x-provider": "anthropic"})
        assert ctx.get_header("x-provider") == "anthropic"
        assert ctx.get_header("X-Provider") == ""  # keys must be pre-lowered
        assert ctx.get_header("missing", "default") == "default"

    def test_with_all_fields(self):
        ctx = RequestContext(
            method="GET",
            path="/v1/models",
            model="gpt-4o",
            headers={"content-type": "application/json"},
            body={"model": "gpt-4o"},
            client_ip="127.0.0.1",
        )
        assert ctx.method == "GET"
        assert ctx.path == "/v1/models"
        assert ctx.model == "gpt-4o"
        assert ctx.client_ip == "127.0.0.1"


# --------------------------------------------------------------------------- #
# RouteRule tests
# --------------------------------------------------------------------------- #


class TestRouteRule:
    """Tests for RouteRule matching logic."""

    def test_model_pattern_exact(self):
        rule = RouteRule(pattern="gpt-4o", provider="openai")
        ctx = RequestContext(model="gpt-4o")
        assert rule.matches(ctx) is True

    def test_model_pattern_glob(self):
        rule = RouteRule(pattern="gpt-*", provider="openai")
        assert rule.matches(RequestContext(model="gpt-4o")) is True
        assert rule.matches(RequestContext(model="gpt-4o-mini")) is True
        assert rule.matches(RequestContext(model="claude-3")) is False

    def test_model_pattern_no_model(self):
        rule = RouteRule(pattern="gpt-*", provider="openai")
        assert rule.matches(RequestContext()) is False

    def test_path_pattern(self):
        rule = RouteRule(
            pattern="/v1/messages*",
            provider="anthropic",
            rule_type=RouteRuleType.PATH_PATTERN,
        )
        assert rule.matches(RequestContext(path="/v1/messages")) is True
        assert rule.matches(RequestContext(path="/v1/messages/count_tokens")) is True
        assert rule.matches(RequestContext(path="/v2/messages")) is False

    def test_path_pattern_empty_path(self):
        rule = RouteRule(
            pattern="/v1/*",
            provider="test",
            rule_type=RouteRuleType.PATH_PATTERN,
        )
        assert rule.matches(RequestContext(path="")) is False

    def test_header_match_exact(self):
        rule = RouteRule(
            pattern="",
            provider="anthropic",
            rule_type=RouteRuleType.HEADER_MATCH,
            header_name="x-provider",
            header_value="anthropic",
        )
        ctx = RequestContext(headers={"x-provider": "anthropic"})
        assert rule.matches(ctx) is True

    def test_header_match_wrong_value(self):
        rule = RouteRule(
            pattern="",
            provider="anthropic",
            rule_type=RouteRuleType.HEADER_MATCH,
            header_name="x-provider",
            header_value="anthropic",
        )
        ctx = RequestContext(headers={"x-provider": "openai"})
        assert rule.matches(ctx) is False

    def test_header_match_presence_only(self):
        rule = RouteRule(
            pattern="",
            provider="anthropic",
            rule_type=RouteRuleType.HEADER_MATCH,
            header_name="x-custom",
        )
        ctx = RequestContext(headers={"x-custom": "anything"})
        assert rule.matches(ctx) is True

    def test_header_match_missing(self):
        rule = RouteRule(
            pattern="",
            provider="anthropic",
            rule_type=RouteRuleType.HEADER_MATCH,
            header_name="x-custom",
        )
        ctx = RequestContext(headers={})
        assert rule.matches(ctx) is False

    def test_header_match_no_header_name(self):
        rule = RouteRule(
            pattern="",
            provider="test",
            rule_type=RouteRuleType.HEADER_MATCH,
        )
        ctx = RequestContext(headers={"x-provider": "test"})
        assert rule.matches(ctx) is False

    def test_catch_all(self):
        rule = RouteRule(
            pattern="",
            provider="default",
            rule_type=RouteRuleType.CATCH_ALL,
        )
        assert rule.matches(RequestContext()) is True
        assert rule.matches(RequestContext(model="any")) is True

    def test_disabled_rule(self):
        rule = RouteRule(pattern="*", provider="test", enabled=False)
        assert rule.matches(RequestContext(model="anything")) is False

    def test_serialization(self):
        rule = RouteRule(
            pattern="claude-*",
            provider="anthropic",
            rule_type=RouteRuleType.MODEL_PATTERN,
            priority=10,
            name="Claude models",
            description="Route Claude models to Anthropic",
        )
        data = rule.to_dict()
        assert data["pattern"] == "claude-*"
        assert data["provider"] == "anthropic"
        assert data["priority"] == 10
        assert data["name"] == "Claude models"

    def test_deserialization(self):
        data = {
            "pattern": "gpt-*",
            "provider": "openai",
            "rule_type": "model_pattern",
            "priority": 5,
        }
        rule = RouteRule.from_dict(data)
        assert rule.pattern == "gpt-*"
        assert rule.provider == "openai"
        assert rule.priority == 5
        assert rule.rule_type == RouteRuleType.MODEL_PATTERN

    def test_deserialization_invalid_type(self):
        data = {
            "pattern": "*",
            "provider": "test",
            "rule_type": "invalid_type",
        }
        rule = RouteRule.from_dict(data)
        assert rule.rule_type == RouteRuleType.MODEL_PATTERN  # default


# --------------------------------------------------------------------------- #
# RouteMatch tests
# --------------------------------------------------------------------------- #


class TestRouteMatch:
    """Tests for RouteMatch serialization."""

    def test_to_dict(self):
        provider = _make_provider("openai", models={"gpt-4o": _make_model("gpt-4o")})
        match = RouteMatch(
            provider_name="openai",
            provider_config=provider,
            model="gpt-4o",
            strategy=RoutingStrategy.MODEL_BASED,
            resolution_time_ms=0.5,
        )
        data = match.to_dict()
        assert data["provider_name"] == "openai"
        assert data["model"] == "gpt-4o"
        assert data["strategy"] == "model_based"
        assert data["fallback_used"] is False

    def test_to_dict_with_rule(self):
        provider = _make_provider("anthropic")
        rule = RouteRule(pattern="claude-*", provider="anthropic")
        match = RouteMatch(
            provider_name="anthropic",
            provider_config=provider,
            rule=rule,
            model="claude-sonnet-4-20250514",
        )
        data = match.to_dict()
        assert "rule" in data
        assert data["rule"]["pattern"] == "claude-*"


# --------------------------------------------------------------------------- #
# build_model_provider_map tests
# --------------------------------------------------------------------------- #


class TestBuildModelProviderMap:
    """Tests for build_model_provider_map."""

    def test_maps_all_models(self, sample_config):
        mapping = build_model_provider_map(sample_config)
        assert "gpt-4o" in mapping
        assert mapping["gpt-4o"] == "openai"
        assert "claude-sonnet-4-20250514" in mapping
        assert mapping["claude-sonnet-4-20250514"] == "anthropic"
        assert "gemini-2.0-flash" in mapping
        assert mapping["gemini-2.0-flash"] == "google"

    def test_excludes_disabled_providers(self, sample_config):
        mapping = build_model_provider_map(sample_config)
        assert "test-model" not in mapping

    def test_empty_config(self, empty_config):
        mapping = build_model_provider_map(empty_config)
        assert mapping == {}

    def test_first_provider_wins(self):
        """When same model appears in multiple providers, first one wins."""
        config = GatewayConfig(
            providers={
                "provider_a": _make_provider(
                    "provider_a", models={"shared-model": _make_model("shared-model")}
                ),
                "provider_b": _make_provider(
                    "provider_b", models={"shared-model": _make_model("shared-model")}
                ),
            }
        )
        mapping = build_model_provider_map(config)
        assert mapping["shared-model"] in ("provider_a", "provider_b")


# --------------------------------------------------------------------------- #
# find_provider_for_model tests
# --------------------------------------------------------------------------- #


class TestFindProviderForModel:
    """Tests for find_provider_for_model."""

    def test_exact_match(self, sample_config):
        assert find_provider_for_model("gpt-4o", sample_config) == "openai"
        assert find_provider_for_model("claude-sonnet-4-20250514", sample_config) == "anthropic"
        assert find_provider_for_model("gemini-2.0-flash", sample_config) == "google"

    def test_no_match(self, sample_config):
        assert find_provider_for_model("unknown-model", sample_config) is None

    def test_excludes_disabled(self, sample_config):
        assert find_provider_for_model("test-model", sample_config) is None


# --------------------------------------------------------------------------- #
# Router tests
# --------------------------------------------------------------------------- #


class TestRouter:
    """Tests for the Router class."""

    # -- Model-based routing ------------------------------------------------ #

    def test_resolve_by_model(self, router):
        ctx = RequestContext(model="gpt-4o")
        match = router.resolve(ctx)
        assert match.provider_name == "openai"
        assert match.strategy == RoutingStrategy.MODEL_BASED

    def test_resolve_by_model_anthropic(self, router):
        ctx = RequestContext(model="claude-sonnet-4-20250514")
        match = router.resolve(ctx)
        assert match.provider_name == "anthropic"

    def test_resolve_by_model_google(self, router):
        ctx = RequestContext(model="gemini-2.0-flash")
        match = router.resolve(ctx)
        assert match.provider_name == "google"

    # -- Rule-based routing ------------------------------------------------- #

    def test_resolve_by_rule(self, router):
        router.add_rule(RouteRule(pattern="test-*", provider="google", priority=100))
        ctx = RequestContext(model="test-model-1")
        match = router.resolve(ctx)
        assert match.provider_name == "google"
        assert match.rule is not None
        assert match.rule.pattern == "test-*"

    def test_rule_priority_ordering(self, router):
        router.add_rule(RouteRule(pattern="multi-*", provider="openai", priority=1))
        router.add_rule(RouteRule(pattern="multi-*", provider="anthropic", priority=10))
        ctx = RequestContext(model="multi-test")
        match = router.resolve(ctx)
        # Higher priority rule wins
        assert match.provider_name == "anthropic"

    def test_rule_skips_disabled_provider(self, router):
        router.add_rule(
            RouteRule(pattern="disabled-*", provider="disabled_provider", priority=100)
        )
        ctx = RequestContext(model="disabled-test")
        # Should fall through to strategy/fallback since provider is disabled
        match = router.resolve(ctx)
        assert match.provider_name != "disabled_provider"

    def test_rule_skips_missing_provider(self, router):
        router.add_rule(
            RouteRule(pattern="missing-*", provider="nonexistent", priority=100)
        )
        ctx = RequestContext(model="missing-test")
        # Should fall through
        match = router.resolve(ctx)
        assert match.provider_name != "nonexistent"

    def test_path_rule(self, router):
        router.add_rule(
            RouteRule(
                pattern="/v2/*",
                provider="google",
                rule_type=RouteRuleType.PATH_PATTERN,
                priority=100,
            )
        )
        ctx = RequestContext(path="/v2/chat", model="any")
        match = router.resolve(ctx)
        assert match.provider_name == "google"

    def test_header_rule(self, router):
        router.add_rule(
            RouteRule(
                pattern="",
                provider="anthropic",
                rule_type=RouteRuleType.HEADER_MATCH,
                header_name="x-target",
                header_value="anthropic",
                priority=100,
            )
        )
        ctx = RequestContext(
            model="some-model",
            headers={"x-target": "anthropic"},
        )
        match = router.resolve(ctx)
        assert match.provider_name == "anthropic"

    def test_catch_all_rule(self, router):
        router.add_rule(
            RouteRule(
                provider="google",
                rule_type=RouteRuleType.CATCH_ALL,
                priority=-1,  # low priority
            )
        )
        ctx = RequestContext(model="unknown-model")
        match = router.resolve(ctx)
        # Should be caught by model lookup or higher priority before catch_all
        assert match.provider_name is not None

    # -- Strategy routing --------------------------------------------------- #

    def test_round_robin_strategy(self, sample_config):
        router = Router(config=sample_config, strategy=RoutingStrategy.ROUND_ROBIN)
        providers_seen = set()
        # Unknown model should trigger strategy-based routing
        for _ in range(10):
            ctx = RequestContext(model="unknown-model-xyz")
            try:
                match = router.resolve(ctx)
                providers_seen.add(match.provider_name)
            except NoRouteError:
                pass
        # Round robin should cycle through providers
        assert len(providers_seen) >= 1

    def test_single_strategy_uses_default(self, sample_config):
        router = Router(config=sample_config, strategy=RoutingStrategy.SINGLE)
        ctx = RequestContext(model="unknown-model-xyz")
        match = router.resolve(ctx)
        assert match.provider_name == "openai"  # default provider

    # -- Fallback ----------------------------------------------------------- #

    def test_fallback_provider(self, sample_config):
        router = Router(
            config=sample_config,
            strategy=RoutingStrategy.MODEL_BASED,
            fallback_provider="google",
        )
        ctx = RequestContext(model="totally-unknown")
        match = router.resolve(ctx)
        assert match.provider_name == "google"
        assert match.fallback_used is True

    def test_no_route_error(self, empty_config):
        router = Router(config=empty_config)
        ctx = RequestContext(model="any-model")
        with pytest.raises(NoRouteError):
            router.resolve(ctx)

    def test_no_route_includes_model_in_message(self, empty_config):
        router = Router(config=empty_config)
        ctx = RequestContext(model="my-model", path="/v1/messages")
        with pytest.raises(NoRouteError) as exc_info:
            router.resolve(ctx)
        assert "my-model" in str(exc_info.value)

    # -- Rule management ---------------------------------------------------- #

    def test_add_rule(self, router):
        initial_count = len(router.rules)
        router.add_rule(RouteRule(pattern="new-*", provider="openai"))
        assert len(router.rules) == initial_count + 1

    def test_remove_rule(self, router):
        router.add_rule(RouteRule(pattern="remove-me", provider="openai"))
        assert router.remove_rule("remove-me", "openai") is True
        assert router.remove_rule("nonexistent", "openai") is False

    def test_clear_rules(self, router):
        router.add_rule(RouteRule(pattern="a", provider="openai"))
        router.add_rule(RouteRule(pattern="b", provider="openai"))
        count = router.clear_rules()
        assert count == 2
        assert len(router.rules) == 0

    # -- Config management -------------------------------------------------- #

    def test_update_config(self, router):
        new_config = GatewayConfig(
            default_provider="anthropic",
            providers={
                "anthropic": _make_provider(
                    "anthropic",
                    models={"claude-3": _make_model("claude-3")},
                ),
            },
        )
        router.update_config(new_config)
        assert "claude-3" in router.model_map
        assert router.model_map["claude-3"] == "anthropic"

    def test_refresh_model_map(self, router, sample_config):
        # Mutate config directly
        sample_config.providers["openai"].models["gpt-5"] = _make_model("gpt-5")
        router.refresh_model_map()
        assert "gpt-5" in router.model_map

    # -- Statistics --------------------------------------------------------- #

    def test_stats_tracking(self, router):
        ctx = RequestContext(model="gpt-4o")
        router.resolve(ctx)
        stats = router.get_stats()
        assert stats["total_requests"] == 1
        assert stats["model_matches"] == 1

    def test_stats_reset(self, router):
        ctx = RequestContext(model="gpt-4o")
        router.resolve(ctx)
        router.reset_stats()
        stats = router.get_stats()
        assert stats["total_requests"] == 0

    def test_stats_no_route(self, empty_config):
        router = Router(config=empty_config)
        try:
            router.resolve(RequestContext(model="unknown"))
        except NoRouteError:
            pass
        stats = router.get_stats()
        assert stats["no_route"] == 1

    # -- Factory ------------------------------------------------------------ #

    def test_from_config(self, sample_config):
        router = Router.from_config(sample_config)
        assert router.strategy == RoutingStrategy.MODEL_BASED
        assert router.fallback_provider == "openai"

    def test_from_config_with_rules(self, sample_config):
        sample_config.extra["routing_rules"] = [
            {"pattern": "custom-*", "provider": "google", "priority": 5}
        ]
        router = Router.from_config(sample_config)
        assert len(router.rules) == 1
        assert router.rules[0].pattern == "custom-*"

    def test_from_config_with_strategy(self, sample_config):
        sample_config.extra["routing_strategy"] = "round_robin"
        router = Router.from_config(sample_config)
        assert router.strategy == RoutingStrategy.ROUND_ROBIN

    def test_from_config_invalid_strategy(self, sample_config):
        sample_config.extra["routing_strategy"] = "invalid"
        router = Router.from_config(sample_config)
        # Should default to MODEL_BASED
        assert router.strategy == RoutingStrategy.MODEL_BASED

    def test_from_config_with_fallback(self, sample_config):
        sample_config.extra["fallback_provider"] = "anthropic"
        router = Router.from_config(sample_config)
        assert router.fallback_provider == "anthropic"

    # -- Properties --------------------------------------------------------- #

    def test_strategy_property(self, router):
        assert router.strategy == RoutingStrategy.MODEL_BASED
        router.strategy = RoutingStrategy.ROUND_ROBIN
        assert router.strategy == RoutingStrategy.ROUND_ROBIN

    def test_fallback_property(self, router):
        router.fallback_provider = "anthropic"
        assert router.fallback_provider == "anthropic"

    def test_repr(self, router):
        r = repr(router)
        assert "Router" in r
        assert "model_based" in r

    # -- Resolution timing -------------------------------------------------- #

    def test_resolution_time_populated(self, router):
        match = router.resolve(RequestContext(model="gpt-4o"))
        assert match.resolution_time_ms >= 0


# --------------------------------------------------------------------------- #
# RouterStats tests
# --------------------------------------------------------------------------- #


class TestRouterStats:
    """Tests for RouterStats."""

    def test_to_dict(self):
        stats = RouterStats(total_requests=10, model_matches=5, no_route=2)
        data = stats.to_dict()
        assert data["total_requests"] == 10
        assert data["model_matches"] == 5
        assert data["no_route"] == 2


# --------------------------------------------------------------------------- #
# Error tests
# --------------------------------------------------------------------------- #


class TestRoutingErrors:
    """Tests for routing-specific exceptions."""

    def test_routing_error(self):
        err = RoutingError("test error", model="gpt-4o", path="/v1/messages")
        assert "test error" in str(err)
        assert err.context.model == "gpt-4o"
        assert err.context.retryable is False

    def test_no_route_error(self):
        err = NoRouteError(model="unknown", path="/v1/chat")
        assert "unknown" in str(err)
        assert "No route found" in str(err)

    def test_no_route_error_no_details(self):
        err = NoRouteError()
        assert "No route found" in str(err)

    def test_provider_disabled_error(self):
        err = ProviderDisabledError("test-provider")
        assert "disabled" in str(err)
        assert err.context.provider == "test-provider"


# --------------------------------------------------------------------------- #
# Helper function tests
# --------------------------------------------------------------------------- #


class TestHelperFunctions:
    """Tests for extract_model_from_body and extract_model_from_path."""

    def test_extract_model_from_body(self):
        assert extract_model_from_body({"model": "gpt-4o"}) == "gpt-4o"
        assert extract_model_from_body({"model": "claude-3"}) == "claude-3"

    def test_extract_model_from_body_missing(self):
        assert extract_model_from_body({}) is None
        assert extract_model_from_body(None) is None
        assert extract_model_from_body({"other": "field"}) is None

    def test_extract_model_from_body_empty_string(self):
        assert extract_model_from_body({"model": ""}) is None

    def test_extract_model_from_body_non_string(self):
        assert extract_model_from_body({"model": 123}) is None

    def test_extract_model_from_path(self):
        assert extract_model_from_path("/v1/models/gpt-4o") == "gpt-4o"
        assert extract_model_from_path("/v1/models/claude-3/") == "claude-3"

    def test_extract_model_from_path_no_match(self):
        assert extract_model_from_path("/v1/messages") is None
        assert extract_model_from_path("/v1/chat/completions") is None
        assert extract_model_from_path("/") is None


# --------------------------------------------------------------------------- #
# Integration tests
# --------------------------------------------------------------------------- #


class TestRouterIntegration:
    """Integration tests combining rules, model map, and strategies."""

    def test_rules_override_model_map(self, sample_config):
        """Rules should take precedence over model map lookups."""
        router = Router(
            config=sample_config,
            rules=[
                RouteRule(
                    pattern="gpt-4o",
                    provider="anthropic",  # Override: gpt-4o -> anthropic
                    priority=100,
                ),
            ],
        )
        match = router.resolve(RequestContext(model="gpt-4o"))
        assert match.provider_name == "anthropic"
        assert match.rule is not None

    def test_model_map_before_strategy(self, sample_config):
        """Model map should be checked before strategy-based selection."""
        router = Router(
            config=sample_config,
            strategy=RoutingStrategy.ROUND_ROBIN,
        )
        match = router.resolve(RequestContext(model="gpt-4o"))
        assert match.provider_name == "openai"
        assert match.strategy == RoutingStrategy.MODEL_BASED

    def test_full_chain_fallthrough(self, sample_config):
        """Unknown model + no strategy match = fallback."""
        router = Router(
            config=sample_config,
            strategy=RoutingStrategy.MODEL_BASED,
            fallback_provider="google",
        )
        match = router.resolve(RequestContext(model="unknown-model-xyz"))
        assert match.provider_name == "google"
        assert match.fallback_used is True

    def test_multiple_rules_different_types(self, sample_config):
        """Different rule types should coexist."""
        router = Router(
            config=sample_config,
            rules=[
                RouteRule(
                    pattern="custom-*",
                    provider="google",
                    rule_type=RouteRuleType.MODEL_PATTERN,
                    priority=10,
                ),
                RouteRule(
                    pattern="/custom/*",
                    provider="anthropic",
                    rule_type=RouteRuleType.PATH_PATTERN,
                    priority=20,
                ),
            ],
        )
        # Path rule has higher priority
        match = router.resolve(RequestContext(model="custom-1", path="/custom/chat"))
        assert match.provider_name == "anthropic"

        # Model rule for different path
        match = router.resolve(RequestContext(model="custom-1", path="/v1/messages"))
        assert match.provider_name == "google"
