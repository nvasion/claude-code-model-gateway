"""Tests for the request interception middleware."""

import time

import pytest

from src.interceptor import (
    AuthenticationInterceptor,
    ChainStats,
    HeaderRoutingInterceptor,
    InterceptAction,
    InterceptorChain,
    InterceptResult,
    ModelRoutingInterceptor,
    RateLimitInterceptor,
    RequestInterceptor,
    RequestTransformInterceptor,
    create_default_chain,
)
from src.models import (
    AuthType,
    GatewayConfig,
    ModelConfig,
    ProviderConfig,
)
from src.router import (
    RequestContext,
    Router,
    RoutingStrategy,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_provider(
    name: str,
    models: dict[str, ModelConfig] | None = None,
    enabled: bool = True,
    api_key_env_var: str = "",
    auth_type: AuthType = AuthType.BEARER_TOKEN,
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        display_name=name.title(),
        api_base=f"https://api.{name}.com/v1",
        api_key_env_var=api_key_env_var or f"{name.upper()}_API_KEY",
        auth_type=auth_type,
        default_model=list(models.keys())[0] if models else "",
        models=models or {},
        enabled=enabled,
    )


def _make_model(name: str) -> ModelConfig:
    return ModelConfig(name=name, display_name=name)


@pytest.fixture
def sample_config() -> GatewayConfig:
    return GatewayConfig(
        default_provider="openai",
        providers={
            "openai": _make_provider(
                "openai",
                models={
                    "gpt-4o": _make_model("gpt-4o"),
                    "gpt-4o-mini": _make_model("gpt-4o-mini"),
                },
                auth_type=AuthType.BEARER_TOKEN,
            ),
            "anthropic": _make_provider(
                "anthropic",
                models={
                    "claude-sonnet-4-20250514": _make_model("claude-sonnet-4-20250514"),
                    "claude-3-5-haiku-20241022": _make_model("claude-3-5-haiku-20241022"),
                },
                auth_type=AuthType.API_KEY,
            ),
            "google": _make_provider(
                "google",
                models={
                    "gemini-2.0-flash": _make_model("gemini-2.0-flash"),
                },
                enabled=True,
            ),
            "disabled": _make_provider(
                "disabled",
                models={"test-model": _make_model("test-model")},
                enabled=False,
            ),
        },
    )


@pytest.fixture
def router(sample_config: GatewayConfig) -> Router:
    return Router(config=sample_config)


# --------------------------------------------------------------------------- #
# InterceptResult tests
# --------------------------------------------------------------------------- #


class TestInterceptResult:
    """Tests for InterceptResult."""

    def test_default_is_skip(self):
        result = InterceptResult()
        assert result.action == InterceptAction.SKIP
        assert not result.is_terminal

    def test_forward_is_terminal(self):
        result = InterceptResult(action=InterceptAction.FORWARD)
        assert result.is_terminal

    def test_reject_is_terminal(self):
        result = InterceptResult(action=InterceptAction.REJECT)
        assert result.is_terminal

    def test_modify_is_not_terminal(self):
        result = InterceptResult(action=InterceptAction.MODIFY)
        assert not result.is_terminal

    def test_redirect_is_not_terminal(self):
        result = InterceptResult(action=InterceptAction.REDIRECT)
        assert not result.is_terminal

    def test_to_dict(self):
        result = InterceptResult(
            action=InterceptAction.FORWARD,
            provider_name="openai",
            interceptor_name="test",
        )
        data = result.to_dict()
        assert data["action"] == "forward"
        assert data["provider_name"] == "openai"
        assert data["interceptor_name"] == "test"

    def test_to_dict_reject(self):
        result = InterceptResult(
            action=InterceptAction.REJECT,
            status_code=429,
            error_message="Too many requests",
            interceptor_name="rate_limit",
        )
        data = result.to_dict()
        assert data["status_code"] == 429
        assert data["error_message"] == "Too many requests"


# --------------------------------------------------------------------------- #
# ModelRoutingInterceptor tests
# --------------------------------------------------------------------------- #


class TestModelRoutingInterceptor:
    """Tests for ModelRoutingInterceptor."""

    def test_routes_known_model(self, router):
        interceptor = ModelRoutingInterceptor(router=router)
        ctx = RequestContext(model="gpt-4o")
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.FORWARD
        assert result.provider_name == "openai"

    def test_routes_anthropic_model(self, router):
        interceptor = ModelRoutingInterceptor(router=router)
        ctx = RequestContext(model="claude-sonnet-4-20250514")
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.FORWARD
        assert result.provider_name == "anthropic"

    def test_extracts_model_from_body(self, router):
        interceptor = ModelRoutingInterceptor(router=router)
        ctx = RequestContext(
            body={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        )
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.FORWARD
        assert result.provider_name == "openai"
        assert ctx.model == "gpt-4o"  # Should be populated

    def test_no_model_skips(self, router):
        interceptor = ModelRoutingInterceptor(router=router)
        ctx = RequestContext()
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.SKIP

    def test_unknown_model_rejects(self, sample_config):
        # Use config without fallback
        config = GatewayConfig(providers={})
        router = Router(config=config)
        interceptor = ModelRoutingInterceptor(router=router)
        ctx = RequestContext(model="unknown-model")
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.REJECT
        assert result.status_code == 404

    def test_custom_name(self, router):
        interceptor = ModelRoutingInterceptor(router=router, name="my_router")
        assert interceptor.name == "my_router"

    def test_route_match_populated(self, router):
        interceptor = ModelRoutingInterceptor(router=router)
        ctx = RequestContext(model="gpt-4o")
        result = interceptor.intercept(ctx)
        assert result.route_match is not None
        assert result.route_match.provider_name == "openai"


# --------------------------------------------------------------------------- #
# HeaderRoutingInterceptor tests
# --------------------------------------------------------------------------- #


class TestHeaderRoutingInterceptor:
    """Tests for HeaderRoutingInterceptor."""

    def test_routes_by_header(self, sample_config):
        interceptor = HeaderRoutingInterceptor(
            header="x-provider",
            config=sample_config,
        )
        ctx = RequestContext(headers={"x-provider": "anthropic"})
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.FORWARD
        assert result.provider_name == "anthropic"

    def test_missing_header_skips(self, sample_config):
        interceptor = HeaderRoutingInterceptor(
            header="x-provider",
            config=sample_config,
        )
        ctx = RequestContext(headers={})
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.SKIP

    def test_unknown_provider_rejects(self, sample_config):
        interceptor = HeaderRoutingInterceptor(
            header="x-provider",
            config=sample_config,
        )
        ctx = RequestContext(headers={"x-provider": "nonexistent"})
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.REJECT
        assert result.status_code == 400

    def test_disabled_provider_rejects(self, sample_config):
        interceptor = HeaderRoutingInterceptor(
            header="x-provider",
            config=sample_config,
        )
        ctx = RequestContext(headers={"x-provider": "disabled"})
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.REJECT

    def test_without_config_redirects(self):
        interceptor = HeaderRoutingInterceptor(header="x-target")
        ctx = RequestContext(headers={"x-target": "some-provider"})
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.REDIRECT
        assert result.provider_name == "some-provider"

    def test_custom_header_name(self, sample_config):
        interceptor = HeaderRoutingInterceptor(
            header="x-custom-route",
            config=sample_config,
        )
        ctx = RequestContext(headers={"x-custom-route": "openai"})
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.FORWARD
        assert result.provider_name == "openai"

    def test_default_order(self):
        interceptor = HeaderRoutingInterceptor()
        assert interceptor.order == 10


# --------------------------------------------------------------------------- #
# RateLimitInterceptor tests
# --------------------------------------------------------------------------- #


class TestRateLimitInterceptor:
    """Tests for RateLimitInterceptor."""

    def test_allows_under_limit(self):
        interceptor = RateLimitInterceptor(max_requests_per_minute=10)
        ctx = RequestContext(client_ip="10.0.0.1")
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.SKIP

    def test_rejects_over_limit(self):
        interceptor = RateLimitInterceptor(max_requests_per_minute=3)
        ctx = RequestContext(client_ip="10.0.0.1")

        # First 3 should pass
        for _ in range(3):
            result = interceptor.intercept(ctx)
            assert result.action == InterceptAction.SKIP

        # 4th should be rejected
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.REJECT
        assert result.status_code == 429

    def test_separate_clients(self):
        interceptor = RateLimitInterceptor(max_requests_per_minute=2)

        # Client A: use up limit
        ctx_a = RequestContext(client_ip="10.0.0.1")
        interceptor.intercept(ctx_a)
        interceptor.intercept(ctx_a)
        result_a = interceptor.intercept(ctx_a)
        assert result_a.action == InterceptAction.REJECT

        # Client B: should still be allowed
        ctx_b = RequestContext(client_ip="10.0.0.2")
        result_b = interceptor.intercept(ctx_b)
        assert result_b.action == InterceptAction.SKIP

    def test_unknown_client(self):
        interceptor = RateLimitInterceptor(max_requests_per_minute=10)
        ctx = RequestContext()  # No client_ip
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.SKIP

    def test_get_client_usage(self):
        interceptor = RateLimitInterceptor(max_requests_per_minute=10)
        ctx = RequestContext(client_ip="10.0.0.1")
        interceptor.intercept(ctx)
        interceptor.intercept(ctx)

        usage = interceptor.get_client_usage("10.0.0.1")
        assert usage["current_count"] == 2
        assert usage["limit"] == 10
        assert usage["remaining"] == 8

    def test_get_client_usage_unknown(self):
        interceptor = RateLimitInterceptor(max_requests_per_minute=10)
        usage = interceptor.get_client_usage("unknown")
        assert usage["current_count"] == 0
        assert usage["remaining"] == 10

    def test_reset_specific_client(self):
        interceptor = RateLimitInterceptor(max_requests_per_minute=2)
        ctx = RequestContext(client_ip="10.0.0.1")
        interceptor.intercept(ctx)
        interceptor.intercept(ctx)

        interceptor.reset("10.0.0.1")
        usage = interceptor.get_client_usage("10.0.0.1")
        assert usage["current_count"] == 0

    def test_reset_all_clients(self):
        interceptor = RateLimitInterceptor(max_requests_per_minute=2)
        for ip in ("10.0.0.1", "10.0.0.2"):
            ctx = RequestContext(client_ip=ip)
            interceptor.intercept(ctx)

        interceptor.reset()
        assert interceptor.get_client_usage("10.0.0.1")["current_count"] == 0
        assert interceptor.get_client_usage("10.0.0.2")["current_count"] == 0

    def test_metadata_contains_count(self):
        interceptor = RateLimitInterceptor(max_requests_per_minute=10)
        ctx = RequestContext(client_ip="10.0.0.1")
        result = interceptor.intercept(ctx)
        assert "current_count" in result.metadata
        assert result.metadata["current_count"] == 1

    def test_error_body_on_reject(self):
        interceptor = RateLimitInterceptor(max_requests_per_minute=1)
        ctx = RequestContext(client_ip="10.0.0.1")
        interceptor.intercept(ctx)
        result = interceptor.intercept(ctx)
        assert result.error_body is not None
        assert result.error_body["error"]["type"] == "rate_limit_error"

    def test_retry_after_header(self):
        interceptor = RateLimitInterceptor(max_requests_per_minute=1)
        ctx = RequestContext(client_ip="10.0.0.1")
        interceptor.intercept(ctx)
        result = interceptor.intercept(ctx)
        assert "Retry-After" in result.modified_headers


# --------------------------------------------------------------------------- #
# AuthenticationInterceptor tests
# --------------------------------------------------------------------------- #


class TestAuthenticationInterceptor:
    """Tests for AuthenticationInterceptor."""

    def test_skips_without_provider(self):
        interceptor = AuthenticationInterceptor()
        ctx = RequestContext()
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.SKIP

    def test_injects_bearer_token(self, sample_config, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        provider = sample_config.providers["openai"]
        prev_result = InterceptResult(
            action=InterceptAction.FORWARD,
            provider_name="openai",
            provider_config=provider,
        )
        interceptor = AuthenticationInterceptor(config=sample_config)
        ctx = RequestContext()
        result = interceptor.intercept(ctx, current_result=prev_result)
        assert result.action == InterceptAction.MODIFY
        assert "Authorization" in result.modified_headers
        assert result.modified_headers["Authorization"] == "Bearer test-key-123"

    def test_injects_api_key_header(self, sample_config, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        provider = sample_config.providers["anthropic"]
        prev_result = InterceptResult(
            action=InterceptAction.FORWARD,
            provider_name="anthropic",
            provider_config=provider,
        )
        interceptor = AuthenticationInterceptor(config=sample_config)
        ctx = RequestContext()
        result = interceptor.intercept(ctx, current_result=prev_result)
        assert result.action == InterceptAction.MODIFY
        assert "x-api-key" in result.modified_headers
        assert result.modified_headers["x-api-key"] == "sk-ant-test"

    def test_rejects_missing_key(self, sample_config, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = sample_config.providers["openai"]
        prev_result = InterceptResult(
            action=InterceptAction.FORWARD,
            provider_name="openai",
            provider_config=provider,
        )
        interceptor = AuthenticationInterceptor(
            config=sample_config, require_api_key=True
        )
        ctx = RequestContext()
        result = interceptor.intercept(ctx, current_result=prev_result)
        assert result.action == InterceptAction.REJECT
        assert result.status_code == 401

    def test_allows_missing_key_when_not_required(self, sample_config, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = sample_config.providers["openai"]
        prev_result = InterceptResult(
            action=InterceptAction.FORWARD,
            provider_name="openai",
            provider_config=provider,
        )
        interceptor = AuthenticationInterceptor(
            config=sample_config, require_api_key=False
        )
        ctx = RequestContext()
        result = interceptor.intercept(ctx, current_result=prev_result)
        assert result.action == InterceptAction.MODIFY

    def test_uses_client_api_key_header(self, sample_config):
        provider = sample_config.providers["openai"]
        prev_result = InterceptResult(
            action=InterceptAction.FORWARD,
            provider_name="openai",
            provider_config=provider,
        )
        interceptor = AuthenticationInterceptor(config=sample_config)
        ctx = RequestContext(headers={"x-api-key": "client-key-from-header"})
        result = interceptor.intercept(ctx, current_result=prev_result)
        assert result.action == InterceptAction.MODIFY
        assert "Authorization" in result.modified_headers
        assert "client-key-from-header" in result.modified_headers["Authorization"]

    def test_uses_bearer_from_authorization_header(self, sample_config):
        provider = sample_config.providers["openai"]
        prev_result = InterceptResult(
            action=InterceptAction.FORWARD,
            provider_name="openai",
            provider_config=provider,
        )
        interceptor = AuthenticationInterceptor(config=sample_config)
        ctx = RequestContext(
            headers={"authorization": "Bearer my-bearer-token"}
        )
        result = interceptor.intercept(ctx, current_result=prev_result)
        assert result.action == InterceptAction.MODIFY


# --------------------------------------------------------------------------- #
# RequestTransformInterceptor tests
# --------------------------------------------------------------------------- #


class TestRequestTransformInterceptor:
    """Tests for RequestTransformInterceptor."""

    def test_model_alias_expansion(self):
        interceptor = RequestTransformInterceptor(
            model_aliases={"sonnet": "claude-sonnet-4-20250514"}
        )
        ctx = RequestContext(
            model="sonnet",
            body={"model": "sonnet", "messages": []},
        )
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.MODIFY
        assert ctx.model == "claude-sonnet-4-20250514"
        assert result.modified_body["model"] == "claude-sonnet-4-20250514"

    def test_no_alias_skips(self):
        interceptor = RequestTransformInterceptor(
            model_aliases={"sonnet": "claude-sonnet-4-20250514"}
        )
        ctx = RequestContext(model="gpt-4o")
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.SKIP

    def test_header_injection(self):
        interceptor = RequestTransformInterceptor(
            inject_headers={"X-Custom": "value", "X-Gateway": "true"}
        )
        ctx = RequestContext()
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.MODIFY
        assert result.modified_headers["X-Custom"] == "value"
        assert result.modified_headers["X-Gateway"] == "true"

    def test_custom_transform_fn(self):
        def add_metadata(ctx):
            if ctx.body is None:
                ctx.body = {}
            ctx.body["metadata"] = {"gateway": "true"}
            return ctx

        interceptor = RequestTransformInterceptor(transform_fn=add_metadata)
        ctx = RequestContext(body={"model": "test"})
        result = interceptor.intercept(ctx)
        assert ctx.body["metadata"] == {"gateway": "true"}

    def test_no_modifications_skips(self):
        interceptor = RequestTransformInterceptor()
        ctx = RequestContext()
        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.SKIP


# --------------------------------------------------------------------------- #
# InterceptorChain tests
# --------------------------------------------------------------------------- #


class TestInterceptorChain:
    """Tests for InterceptorChain."""

    def test_empty_chain_returns_skip(self):
        chain = InterceptorChain()
        ctx = RequestContext()
        result = chain.process(ctx)
        assert result.action == InterceptAction.SKIP

    def test_single_forward_interceptor(self, router):
        chain = InterceptorChain()
        chain.add(ModelRoutingInterceptor(router=router))
        ctx = RequestContext(model="gpt-4o")
        result = chain.process(ctx)
        assert result.action == InterceptAction.FORWARD
        assert result.provider_name == "openai"

    def test_chain_ordering(self, sample_config, router):
        chain = InterceptorChain()
        # Add in wrong order -- should be sorted by `order`
        chain.add(ModelRoutingInterceptor(router=router, order=50))
        chain.add(
            HeaderRoutingInterceptor(
                header="x-provider", config=sample_config, order=10
            )
        )

        interceptors = chain.interceptors
        assert interceptors[0].order < interceptors[1].order

    def test_header_routing_overrides_model(self, sample_config, router):
        chain = InterceptorChain()
        chain.add(
            HeaderRoutingInterceptor(
                header="x-provider", config=sample_config, order=10
            )
        )
        chain.add(ModelRoutingInterceptor(router=router, order=50))

        # Header says anthropic, model says openai
        ctx = RequestContext(
            model="gpt-4o",
            headers={"x-provider": "anthropic"},
        )
        result = chain.process(ctx)
        assert result.action == InterceptAction.FORWARD
        assert result.provider_name == "anthropic"

    def test_rate_limit_before_routing(self, router):
        chain = InterceptorChain()
        chain.add(RateLimitInterceptor(max_requests_per_minute=1, order=5))
        chain.add(ModelRoutingInterceptor(router=router, order=50))

        ctx = RequestContext(model="gpt-4o", client_ip="10.0.0.1")

        # First request: passes through
        result = chain.process(ctx)
        assert result.action == InterceptAction.FORWARD

        # Second request: rate limited
        result = chain.process(ctx)
        assert result.action == InterceptAction.REJECT
        assert result.status_code == 429

    def test_modifications_accumulate(self, router, sample_config, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        chain = InterceptorChain()
        chain.add(
            RequestTransformInterceptor(
                inject_headers={"X-Gateway": "true"},
                order=15,
            )
        )
        chain.add(ModelRoutingInterceptor(router=router, order=50))

        ctx = RequestContext(model="gpt-4o")
        result = chain.process(ctx)
        assert result.action == InterceptAction.FORWARD
        assert "X-Gateway" in result.modified_headers

    def test_reject_stops_chain(self, router):
        chain = InterceptorChain()
        chain.add(RateLimitInterceptor(max_requests_per_minute=0, order=5))
        chain.add(ModelRoutingInterceptor(router=router, order=50))

        ctx = RequestContext(model="gpt-4o", client_ip="10.0.0.1")
        result = chain.process(ctx)
        assert result.action == InterceptAction.REJECT

    def test_disabled_interceptor_skipped(self, router):
        chain = InterceptorChain()
        rate_limiter = RateLimitInterceptor(max_requests_per_minute=0, order=5)
        rate_limiter.enabled = False
        chain.add(rate_limiter)
        chain.add(ModelRoutingInterceptor(router=router, order=50))

        ctx = RequestContext(model="gpt-4o")
        result = chain.process(ctx)
        assert result.action == InterceptAction.FORWARD

    def test_exception_in_interceptor_doesnt_break_chain(self, router):
        """A broken interceptor should not crash the chain."""

        class BrokenInterceptor(RequestInterceptor):
            def intercept(self, ctx, current_result=None):
                raise RuntimeError("Something went wrong!")

        chain = InterceptorChain()
        chain.add(BrokenInterceptor(name="broken", order=5))
        chain.add(ModelRoutingInterceptor(router=router, order=50))

        ctx = RequestContext(model="gpt-4o")
        result = chain.process(ctx)
        assert result.action == InterceptAction.FORWARD

    def test_remove_interceptor(self, router):
        chain = InterceptorChain()
        chain.add(ModelRoutingInterceptor(router=router, name="model_routing"))
        assert len(chain) == 1
        assert chain.remove("model_routing") is True
        assert len(chain) == 0

    def test_remove_nonexistent(self):
        chain = InterceptorChain()
        assert chain.remove("nonexistent") is False

    def test_clear(self, router):
        chain = InterceptorChain()
        chain.add(ModelRoutingInterceptor(router=router))
        chain.add(RateLimitInterceptor())
        chain.clear()
        assert len(chain) == 0

    def test_stats_tracking(self, router):
        chain = InterceptorChain()
        chain.add(ModelRoutingInterceptor(router=router))

        ctx = RequestContext(model="gpt-4o")
        chain.process(ctx)
        chain.process(ctx)

        stats = chain.get_stats()
        assert stats["total"] == 2
        assert stats["forwarded"] == 2

    def test_stats_rejected(self, router):
        chain = InterceptorChain()
        chain.add(RateLimitInterceptor(max_requests_per_minute=1, order=5))
        chain.add(ModelRoutingInterceptor(router=router, order=50))

        ctx = RequestContext(model="gpt-4o", client_ip="10.0.0.1")
        chain.process(ctx)
        chain.process(ctx)

        stats = chain.get_stats()
        assert stats["forwarded"] == 1
        assert stats["rejected"] == 1

    def test_stats_errors(self, router):

        class BrokenInterceptor(RequestInterceptor):
            def intercept(self, ctx, current_result=None):
                raise RuntimeError("boom")

        chain = InterceptorChain()
        chain.add(BrokenInterceptor(name="broken", order=5))
        chain.add(ModelRoutingInterceptor(router=router, order=50))

        ctx = RequestContext(model="gpt-4o")
        chain.process(ctx)
        stats = chain.get_stats()
        assert stats["errors"] == 1

    def test_stats_reset(self, router):
        chain = InterceptorChain()
        chain.add(ModelRoutingInterceptor(router=router))
        chain.process(RequestContext(model="gpt-4o"))
        chain.reset_stats()
        stats = chain.get_stats()
        assert stats["total"] == 0

    def test_repr(self, router):
        chain = InterceptorChain()
        chain.add(ModelRoutingInterceptor(router=router, name="model"))
        chain.add(RateLimitInterceptor(name="rate"))
        r = repr(chain)
        assert "rate" in r
        assert "model" in r

    def test_chain_duration_in_metadata(self, router):
        chain = InterceptorChain()
        chain.add(ModelRoutingInterceptor(router=router))
        result = chain.process(RequestContext(model="gpt-4o"))
        assert "chain_duration_ms" in result.metadata

    def test_len(self, router):
        chain = InterceptorChain()
        assert len(chain) == 0
        chain.add(ModelRoutingInterceptor(router=router))
        assert len(chain) == 1


# --------------------------------------------------------------------------- #
# ChainStats tests
# --------------------------------------------------------------------------- #


class TestChainStats:
    """Tests for ChainStats."""

    def test_to_dict(self):
        stats = ChainStats(total=10, forwarded=7, rejected=2, passthrough=1)
        data = stats.to_dict()
        assert data["total"] == 10
        assert data["forwarded"] == 7
        assert data["rejected"] == 2
        assert data["passthrough"] == 1
        assert data["errors"] == 0


# --------------------------------------------------------------------------- #
# create_default_chain tests
# --------------------------------------------------------------------------- #


class TestCreateDefaultChain:
    """Tests for the default chain factory."""

    def test_creates_chain(self, sample_config):
        chain = create_default_chain(config=sample_config)
        assert len(chain) >= 2

    def test_with_rate_limit(self, sample_config):
        chain = create_default_chain(config=sample_config, max_rpm=100)
        names = [i.name for i in chain.interceptors]
        assert "rate_limit" in names

    def test_without_rate_limit(self, sample_config):
        chain = create_default_chain(config=sample_config, max_rpm=0)
        names = [i.name for i in chain.interceptors]
        assert "rate_limit" not in names

    def test_with_model_aliases(self, sample_config):
        chain = create_default_chain(
            config=sample_config,
            model_aliases={"sonnet": "claude-sonnet-4-20250514"},
        )
        names = [i.name for i in chain.interceptors]
        assert "request_transform" in names

    def test_with_custom_router(self, sample_config, router):
        chain = create_default_chain(config=sample_config, router=router)
        assert len(chain) >= 2

    def test_header_routing_present(self, sample_config):
        chain = create_default_chain(config=sample_config)
        names = [i.name for i in chain.interceptors]
        assert "header_routing" in names

    def test_model_routing_present(self, sample_config):
        chain = create_default_chain(config=sample_config)
        names = [i.name for i in chain.interceptors]
        assert "model_routing" in names

    def test_auth_present(self, sample_config):
        chain = create_default_chain(config=sample_config)
        names = [i.name for i in chain.interceptors]
        assert "authentication" in names


# --------------------------------------------------------------------------- #
# Integration: full chain processing
# --------------------------------------------------------------------------- #


class TestChainIntegration:
    """End-to-end tests for the full interceptor chain."""

    def test_full_chain_model_routing(self, sample_config, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        chain = create_default_chain(config=sample_config)
        ctx = RequestContext(
            method="POST",
            path="/v1/messages",
            body={"model": "gpt-4o", "messages": []},
            client_ip="127.0.0.1",
        )
        result = chain.process(ctx)
        assert result.action == InterceptAction.FORWARD
        assert result.provider_name == "openai"

    def test_full_chain_header_override(self, sample_config, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        chain = create_default_chain(config=sample_config)
        ctx = RequestContext(
            method="POST",
            path="/v1/messages",
            model="gpt-4o",
            headers={"x-provider": "anthropic"},
            client_ip="127.0.0.1",
        )
        result = chain.process(ctx)
        assert result.action == InterceptAction.FORWARD
        assert result.provider_name == "anthropic"

    def test_full_chain_with_aliases(self, sample_config, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        chain = create_default_chain(
            config=sample_config,
            model_aliases={"sonnet": "claude-sonnet-4-20250514"},
        )
        ctx = RequestContext(
            method="POST",
            path="/v1/messages",
            model="sonnet",
            body={"model": "sonnet"},
            client_ip="127.0.0.1",
        )
        result = chain.process(ctx)
        assert result.action == InterceptAction.FORWARD
        assert result.provider_name == "anthropic"

    def test_full_chain_rate_limited(self, sample_config):
        chain = create_default_chain(config=sample_config, max_rpm=1)
        ctx = RequestContext(
            model="gpt-4o",
            client_ip="10.0.0.1",
        )
        chain.process(ctx)
        result = chain.process(ctx)
        assert result.action == InterceptAction.REJECT
        assert result.status_code == 429
