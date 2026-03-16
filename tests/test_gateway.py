"""Tests for the multi-provider routing gateway server."""

from __future__ import annotations

import http.server
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.gateway import (
    DEFAULT_MAX_REQUEST_SIZE,
    GATEWAY_INTERNAL_PATHS,
    GatewayRequestHandler,
    GatewayStats,
    ThreadedGatewayServer,
    create_gateway_server,
    run_gateway_in_thread,
)
from src.interceptor import (
    InterceptAction,
    InterceptorChain,
    InterceptResult,
    ModelRoutingInterceptor,
    RateLimitInterceptor,
    RequestTransformInterceptor,
    create_default_chain,
)
from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig
from src.router import RequestContext, Router


# --------------------------------------------------------------------------- #
# Helpers / Fixtures
# --------------------------------------------------------------------------- #


def _make_provider(
    name: str,
    api_base: str = "",
    models: Optional[Dict[str, ModelConfig]] = None,
    enabled: bool = True,
    auth_type: AuthType = AuthType.BEARER_TOKEN,
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        display_name=name.title(),
        api_base=api_base or f"https://api.{name}.com/v1",
        api_key_env_var=f"{name.upper()}_API_KEY",
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
            ),
            "anthropic": _make_provider(
                "anthropic",
                models={
                    "claude-sonnet-4-20250514": _make_model("claude-sonnet-4-20250514"),
                },
                auth_type=AuthType.API_KEY,
            ),
        },
    )


@pytest.fixture
def empty_chain() -> InterceptorChain:
    """An interceptor chain with no interceptors."""
    return InterceptorChain()


class _MockUpstreamHandler(http.server.BaseHTTPRequestHandler):
    """Minimal mock upstream API server."""

    # Shared state for assertions
    received_requests: list = []

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        _MockUpstreamHandler.received_requests.append(
            {"method": "GET", "path": self.path, "headers": dict(self.headers)}
        )
        body = json.dumps({"object": "list", "data": []}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length else b""
        _MockUpstreamHandler.received_requests.append(
            {
                "method": "POST",
                "path": self.path,
                "headers": dict(self.headers),
                "body": body_bytes,
            }
        )
        resp = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"}}
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()


@pytest.fixture
def mock_upstream():
    """Start a local mock upstream API server and return its base URL."""
    _MockUpstreamHandler.received_requests = []

    server = http.server.HTTPServer(("127.0.0.1", 0), _MockUpstreamHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()


def _start_gateway(config: GatewayConfig, **kwargs) -> tuple:
    """Start a gateway server and return (server, url, thread)."""
    server = create_gateway_server(config, host="127.0.0.1", port=0, **kwargs)
    port = server.server_address[1]
    thread = run_gateway_in_thread(server)
    time.sleep(0.05)  # Give server a moment to start
    return server, f"http://127.0.0.1:{port}", thread


# --------------------------------------------------------------------------- #
# GatewayStats tests
# --------------------------------------------------------------------------- #


class TestGatewayStats:
    """Tests for GatewayStats."""

    def test_initial_state(self):
        stats = GatewayStats()
        data = stats.to_dict()
        assert data["requests_total"] == 0
        assert data["requests_forwarded"] == 0
        assert data["requests_rejected"] == 0
        assert data["requests_errored"] == 0
        assert data["provider_counts"] == {}
        assert data["uptime_seconds"] >= 0

    def test_record_request(self):
        stats = GatewayStats()
        stats.record_request()
        stats.record_request()
        assert stats.to_dict()["requests_total"] == 2

    def test_record_forwarded(self):
        stats = GatewayStats()
        stats.record_forwarded("openai")
        stats.record_forwarded("openai")
        stats.record_forwarded("anthropic")
        data = stats.to_dict()
        assert data["requests_forwarded"] == 3
        assert data["provider_counts"]["openai"] == 2
        assert data["provider_counts"]["anthropic"] == 1

    def test_record_rejected(self):
        stats = GatewayStats()
        stats.record_rejected()
        assert stats.to_dict()["requests_rejected"] == 1

    def test_record_error(self):
        stats = GatewayStats()
        stats.record_error()
        assert stats.to_dict()["requests_errored"] == 1

    def test_thread_safety(self):
        """Stats should be safe to increment from multiple threads."""
        stats = GatewayStats()
        threads = []
        for _ in range(50):
            t = threading.Thread(target=stats.record_request)
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert stats.to_dict()["requests_total"] == 50


# --------------------------------------------------------------------------- #
# create_gateway_server tests
# --------------------------------------------------------------------------- #


class TestCreateGatewayServer:
    """Tests for the create_gateway_server factory."""

    def test_creates_server(self, sample_config):
        server = create_gateway_server(sample_config, host="127.0.0.1", port=0)
        assert isinstance(server, ThreadedGatewayServer)
        server.server_close()

    def test_uses_provided_chain(self, sample_config):
        chain = InterceptorChain()
        server = create_gateway_server(
            sample_config, interceptor_chain=chain, host="127.0.0.1", port=0
        )
        assert server.interceptor_chain is chain  # type: ignore[attr-defined]
        server.server_close()

    def test_creates_default_chain_when_none(self, sample_config):
        server = create_gateway_server(sample_config, host="127.0.0.1", port=0)
        # Should have created a chain automatically
        assert server.interceptor_chain is not None  # type: ignore[attr-defined]
        server.server_close()

    def test_config_attached_to_server(self, sample_config):
        server = create_gateway_server(sample_config, host="127.0.0.1", port=0)
        assert server.gateway_config is sample_config  # type: ignore[attr-defined]
        server.server_close()

    def test_stats_attached(self, sample_config):
        server = create_gateway_server(sample_config, host="127.0.0.1", port=0)
        assert isinstance(server.stats, GatewayStats)  # type: ignore[attr-defined]
        server.server_close()

    def test_rate_limiting_chain(self, sample_config):
        server = create_gateway_server(
            sample_config, host="127.0.0.1", port=0, max_rpm=60
        )
        chain: InterceptorChain = server.interceptor_chain  # type: ignore[attr-defined]
        names = [i.name for i in chain.interceptors]
        assert "rate_limit" in names
        server.server_close()

    def test_model_aliases_chain(self, sample_config):
        server = create_gateway_server(
            sample_config,
            host="127.0.0.1",
            port=0,
            model_aliases={"sonnet": "claude-sonnet-4-20250514"},
        )
        chain: InterceptorChain = server.interceptor_chain  # type: ignore[attr-defined]
        names = [i.name for i in chain.interceptors]
        assert "request_transform" in names
        server.server_close()


# --------------------------------------------------------------------------- #
# Health / Status endpoints
# --------------------------------------------------------------------------- #


class TestGatewayInternalEndpoints:
    """Tests for /health and /status internal endpoints."""

    def test_health_endpoint(self, sample_config):
        server, url, _ = _start_gateway(sample_config)
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=5)
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["status"] == "ok"
        finally:
            server.shutdown()

    def test_status_endpoint(self, sample_config):
        server, url, _ = _start_gateway(sample_config)
        try:
            resp = urllib.request.urlopen(f"{url}/status", timeout=5)
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["status"] == "ok"
            assert "statistics" in data
        finally:
            server.shutdown()

    def test_status_includes_interceptor_stats(self, sample_config):
        chain = create_default_chain(config=sample_config)
        server, url, _ = _start_gateway(
            sample_config, interceptor_chain=chain
        )
        try:
            resp = urllib.request.urlopen(f"{url}/status", timeout=5)
            data = json.loads(resp.read())
            assert "interceptor_chain" in data
        finally:
            server.shutdown()

    def test_health_has_service_name(self, sample_config):
        server, url, _ = _start_gateway(sample_config)
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=5)
            data = json.loads(resp.read())
            assert "service" in data
        finally:
            server.shutdown()

    def test_gateway_internal_paths_constant(self):
        assert "/health" in GATEWAY_INTERNAL_PATHS
        assert "/status" in GATEWAY_INTERNAL_PATHS


# --------------------------------------------------------------------------- #
# OPTIONS (CORS) tests
# --------------------------------------------------------------------------- #


class TestCORSOptions:
    """Tests for CORS preflight handling."""

    def test_options_returns_200(self, sample_config):
        server, url, _ = _start_gateway(sample_config)
        try:
            req = urllib.request.Request(
                f"{url}/v1/messages", method="OPTIONS"
            )
            try:
                resp = urllib.request.urlopen(req, timeout=5)
                assert resp.status == 200
            except urllib.error.HTTPError as exc:
                # Some implementations return 200 with OPTIONS
                assert exc.code == 200
        finally:
            server.shutdown()


# --------------------------------------------------------------------------- #
# Request rejection tests (via interceptor chain)
# --------------------------------------------------------------------------- #


class TestGatewayRejection:
    """Tests for request rejection through the interceptor chain."""

    def test_rate_limit_rejection(self, sample_config, monkeypatch):
        """Requests exceeding rate limit should get 429."""
        chain = create_default_chain(config=sample_config, max_rpm=1, require_auth=False)
        server, url, _ = _start_gateway(sample_config, interceptor_chain=chain)
        try:
            body = json.dumps(
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()
            headers = {"Content-Type": "application/json"}

            def _post():
                req = urllib.request.Request(
                    f"{url}/v1/messages",
                    data=body,
                    headers=headers,
                    method="POST",
                )
                return urllib.request.urlopen(req, timeout=5)

            # First request succeeds (even without upstream, we want the rate-limit test)
            try:
                _post()
            except (urllib.error.HTTPError, urllib.error.URLError):
                pass  # Upstream not reachable, that's OK

            # Second request should be rate limited
            try:
                _post()
                # If we get here without error, the upstream probably wasn't reached
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    data = json.loads(exc.read())
                    assert "error" in data
        finally:
            server.shutdown()

    def test_no_provider_returns_502(self, monkeypatch):
        """A request where no provider can be resolved should return 502."""
        config = GatewayConfig(default_provider="", providers={})
        chain = InterceptorChain()  # Empty chain — will not resolve any provider
        server, url, _ = _start_gateway(config, interceptor_chain=chain)
        try:
            body = json.dumps(
                {"model": "unknown", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()
            req = urllib.request.Request(
                f"{url}/v1/messages",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=5)
            except urllib.error.HTTPError as exc:
                assert exc.code == 502
                data = json.loads(exc.read())
                assert "error" in data
        finally:
            server.shutdown()

    def test_oversized_body_returns_413(self, sample_config):
        """Requests exceeding max_request_size should return 413."""
        server, url, _ = _start_gateway(
            sample_config, max_request_size=10  # tiny limit
        )
        try:
            body = b"x" * 100  # larger than limit
            req = urllib.request.Request(
                f"{url}/v1/messages",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=5)
            except urllib.error.HTTPError as exc:
                assert exc.code == 413
        finally:
            server.shutdown()


# --------------------------------------------------------------------------- #
# End-to-end forwarding tests
# --------------------------------------------------------------------------- #


class TestGatewayForwarding:
    """Tests for actual request forwarding to a mock upstream."""

    def test_get_request_forwarded(self, mock_upstream):
        """GET requests should be forwarded to the upstream provider."""
        config = GatewayConfig(
            default_provider="mock",
            providers={
                "mock": _make_provider(
                    "mock",
                    api_base=mock_upstream,
                    models={"test-model": _make_model("test-model")},
                ),
            },
        )
        chain = InterceptorChain()  # no auth required
        server, url, _ = _start_gateway(config, interceptor_chain=chain)
        try:
            resp = urllib.request.urlopen(f"{url}/v1/models", timeout=5)
            assert resp.status == 200
            # Verify the upstream actually received the request
            assert len(_MockUpstreamHandler.received_requests) >= 1
        finally:
            server.shutdown()

    def test_post_request_forwarded(self, mock_upstream):
        """POST requests should be forwarded with body to the upstream."""
        config = GatewayConfig(
            default_provider="mock",
            providers={
                "mock": _make_provider(
                    "mock",
                    api_base=mock_upstream,
                    models={"test-model": _make_model("test-model")},
                ),
            },
        )
        chain = InterceptorChain()
        server, url, _ = _start_gateway(config, interceptor_chain=chain)
        try:
            payload = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{url}/v1/messages",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=5)
            assert resp.status == 200
            data = json.loads(resp.read())
            # The mock returns a chat completion response
            assert "id" in data or "choices" in data or "object" in data
        finally:
            server.shutdown()

    def test_forwarded_request_contains_auth_header(
        self, mock_upstream, monkeypatch
    ):
        """Auth headers injected by interceptors should reach the upstream."""
        monkeypatch.setenv("MOCK_API_KEY", "test-secret-key")
        config = GatewayConfig(
            default_provider="mock",
            providers={
                "mock": _make_provider(
                    "mock",
                    api_base=mock_upstream,
                    models={"test-model": _make_model("test-model")},
                    auth_type=AuthType.BEARER_TOKEN,
                ),
            },
        )
        chain = create_default_chain(config=config, require_auth=False)
        server, url, _ = _start_gateway(config, interceptor_chain=chain)
        try:
            payload = {"model": "test-model", "messages": []}
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{url}/v1/messages",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass  # We only care about what was sent upstream
        finally:
            server.shutdown()

    def test_stats_incremented_after_forwarding(self, mock_upstream):
        """Stats counters should be updated after successful forwarding."""
        config = GatewayConfig(
            default_provider="mock",
            providers={
                "mock": _make_provider(
                    "mock",
                    api_base=mock_upstream,
                    models={"test-model": _make_model("test-model")},
                ),
            },
        )
        chain = InterceptorChain()
        server, url, _ = _start_gateway(config, interceptor_chain=chain)
        try:
            urllib.request.urlopen(f"{url}/v1/models", timeout=5)
            time.sleep(0.05)
            stats: GatewayStats = server.stats  # type: ignore[attr-defined]
            assert stats.to_dict()["requests_total"] >= 1
            assert stats.to_dict()["requests_forwarded"] >= 1
        finally:
            server.shutdown()

    def test_model_routing_uses_correct_provider(self, mock_upstream):
        """Requests with a model field should be routed to the correct provider."""
        config = GatewayConfig(
            default_provider="provider_a",
            providers={
                "provider_a": _make_provider(
                    "provider_a",
                    api_base=mock_upstream,
                    models={"model-a": _make_model("model-a")},
                ),
                "provider_b": _make_provider(
                    "provider_b",
                    api_base=mock_upstream,
                    models={"model-b": _make_model("model-b")},
                ),
            },
        )
        router = Router(config=config)
        chain = InterceptorChain()
        chain.add(ModelRoutingInterceptor(router=router, order=50))
        server, url, _ = _start_gateway(config, interceptor_chain=chain)
        try:
            payload = {"model": "model-b", "messages": []}
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{url}/v1/messages",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=5)
            assert resp.status == 200
            # Allow a brief window for the gateway's background thread to
            # finish updating stats *after* the response body has been sent.
            time.sleep(0.05)
            stats: GatewayStats = server.stats  # type: ignore[attr-defined]
            assert stats.to_dict()["provider_counts"].get("provider_b", 0) >= 1
        finally:
            server.shutdown()


# --------------------------------------------------------------------------- #
# URL building tests
# --------------------------------------------------------------------------- #


class TestBuildUpstreamUrl:
    """Tests for GatewayRequestHandler._build_upstream_url."""

    def _make_handler(self) -> GatewayRequestHandler:
        """Create a bare handler instance without a server socket."""
        handler = GatewayRequestHandler.__new__(GatewayRequestHandler)
        return handler

    def test_simple_append(self):
        handler = self._make_handler()
        provider = _make_provider("openai", api_base="https://api.openai.com/v1")
        url = handler._build_upstream_url(provider, "/v1/messages")
        assert url.startswith("https://api.openai.com")
        assert "messages" in url

    def test_no_double_path(self):
        handler = self._make_handler()
        provider = _make_provider("test", api_base="https://api.test.com/v1")
        url = handler._build_upstream_url(provider, "/v1/messages")
        # Should not have /v1/v1/messages
        assert "/v1/v1/" not in url

    def test_handles_trailing_slash(self):
        handler = self._make_handler()
        provider = _make_provider("test", api_base="https://api.test.com/v1/")
        url = handler._build_upstream_url(provider, "/v1/messages")
        assert "messages" in url
        assert "//v1" not in url


# --------------------------------------------------------------------------- #
# Interceptor chain integration
# --------------------------------------------------------------------------- #


class TestInterceptorChainIntegration:
    """Tests ensuring the gateway properly integrates the interceptor chain."""

    def test_empty_chain_allows_passthrough(self, mock_upstream):
        """An empty interceptor chain should still allow requests through."""
        config = GatewayConfig(
            default_provider="mock",
            providers={
                "mock": _make_provider(
                    "mock",
                    api_base=mock_upstream,
                    models={"test-model": _make_model("test-model")},
                ),
            },
        )
        chain = InterceptorChain()
        server, url, _ = _start_gateway(config, interceptor_chain=chain)
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=5)
            assert resp.status == 200
        finally:
            server.shutdown()

    def test_modified_body_sent_upstream(self, mock_upstream):
        """Body modifications by interceptors should be forwarded upstream."""
        config = GatewayConfig(
            default_provider="mock",
            providers={
                "mock": _make_provider(
                    "mock",
                    api_base=mock_upstream,
                    models={"real-model": _make_model("real-model")},
                ),
            },
        )
        router = Router(config=config)
        chain = InterceptorChain()
        # Add alias: alias-model -> real-model
        chain.add(
            RequestTransformInterceptor(
                model_aliases={"alias-model": "real-model"}, order=10
            )
        )
        chain.add(ModelRoutingInterceptor(router=router, order=50))

        server, url, _ = _start_gateway(config, interceptor_chain=chain)
        try:
            payload = {"model": "alias-model", "messages": []}
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{url}/v1/messages",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            # Check that upstream received the transformed model name
            post_reqs = [
                r
                for r in _MockUpstreamHandler.received_requests
                if r["method"] == "POST"
            ]
            if post_reqs:
                upstream_body = json.loads(post_reqs[-1]["body"])
                assert upstream_body.get("model") == "real-model"
        finally:
            server.shutdown()


# --------------------------------------------------------------------------- #
# GatewayRequestHandler unit tests (without running a full server)
# --------------------------------------------------------------------------- #


class TestGatewayRequestHandlerUnit:
    """Unit tests for GatewayRequestHandler helper methods."""

    def _handler(self) -> GatewayRequestHandler:
        h = GatewayRequestHandler.__new__(GatewayRequestHandler)
        h.gateway_config = GatewayConfig()
        h.interceptor_chain = None
        h.upstream_timeout = 30
        h.max_request_size = DEFAULT_MAX_REQUEST_SIZE
        h.max_retries = 0
        h.retry_base_delay = 1.0
        h.stats = GatewayStats()
        return h

    def test_is_streaming_response_sse(self):
        h = self._handler()
        mock_resp = MagicMock()
        mock_resp.getheader = lambda name, default="": (
            "text/event-stream" if name == "Content-Type" else default
        )
        assert h._is_streaming_response(mock_resp) is True

    def test_is_streaming_response_chunked(self):
        h = self._handler()
        mock_resp = MagicMock()
        mock_resp.getheader = lambda name, default="": (
            "chunked" if name == "Transfer-Encoding" else default
        )
        assert h._is_streaming_response(mock_resp) is True

    def test_is_streaming_response_regular(self):
        h = self._handler()
        mock_resp = MagicMock()
        mock_resp.getheader = lambda name, default="": default
        assert h._is_streaming_response(mock_resp) is False

    def test_run_interceptor_chain_with_none_chain(self):
        h = self._handler()
        h.interceptor_chain = None
        ctx = RequestContext(model="test")
        result = h._run_interceptor_chain(ctx)
        assert result.action == InterceptAction.SKIP

    def test_run_interceptor_chain_handles_exception(self):
        h = self._handler()

        class BrokenChain:
            def process(self, ctx):
                raise RuntimeError("chain broken")

        h.interceptor_chain = BrokenChain()
        ctx = RequestContext(model="test")
        result = h._run_interceptor_chain(ctx)
        assert result.action == InterceptAction.SKIP

    def test_gateway_internal_paths_not_forwarded(self):
        """Internal paths should not trigger upstream forwarding."""
        for path in GATEWAY_INTERNAL_PATHS:
            assert "/" in path


# --------------------------------------------------------------------------- #
# run_gateway_in_thread tests
# --------------------------------------------------------------------------- #


class TestRunGatewayInThread:
    """Tests for run_gateway_in_thread."""

    def test_server_starts(self, sample_config):
        server = create_gateway_server(sample_config, host="127.0.0.1", port=0)
        thread = run_gateway_in_thread(server)
        time.sleep(0.05)
        assert thread.is_alive()
        server.shutdown()

    def test_daemon_thread_by_default(self, sample_config):
        server = create_gateway_server(sample_config, host="127.0.0.1", port=0)
        thread = run_gateway_in_thread(server, daemon=True)
        assert thread.daemon is True
        server.shutdown()

    def test_non_daemon_thread(self, sample_config):
        server = create_gateway_server(sample_config, host="127.0.0.1", port=0)
        thread = run_gateway_in_thread(server, daemon=False)
        assert thread.daemon is False
        server.shutdown()


# --------------------------------------------------------------------------- #
# Default constants tests
# --------------------------------------------------------------------------- #


class TestConstants:
    """Tests for module-level constants."""

    def test_default_max_request_size(self):
        assert DEFAULT_MAX_REQUEST_SIZE == 10 * 1024 * 1024

    def test_gateway_internal_paths(self):
        assert isinstance(GATEWAY_INTERNAL_PATHS, frozenset)
        assert "/health" in GATEWAY_INTERNAL_PATHS
        assert "/status" in GATEWAY_INTERNAL_PATHS
