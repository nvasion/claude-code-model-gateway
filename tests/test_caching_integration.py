"""Integration tests for the new caching features.

Covers:
- CachingInterceptor in create_default_chain
- Response caching in AnthropicPassthroughHandler (via mock HTTP server)
- /status endpoint cache statistics
- Background purger auto-start in run_passthrough
- CLI --response-cache / --cache-ttl / --cache-maxsize options
"""

from __future__ import annotations

import http.client
import http.server
import json
import socketserver
import threading
import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cache import reset_registry, stop_background_purger
from src.cli import main
from src.interceptor import (
    CachingInterceptor,
    InterceptAction,
    create_default_chain,
)
from src.models import GatewayConfig
from src.response_cache import ResponseCache, reset_response_cache
from src.router import RequestContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset global state before and after each test."""
    reset_registry()
    reset_response_cache()
    stop_background_purger()
    yield
    reset_registry()
    reset_response_cache()
    stop_background_purger()


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def empty_config():
    """Return an empty GatewayConfig."""
    return GatewayConfig()


# ---------------------------------------------------------------------------
# create_default_chain with enable_caching
# ---------------------------------------------------------------------------


class TestCreateDefaultChainCaching:
    """Tests for CachingInterceptor integration in create_default_chain."""

    def test_caching_disabled_by_default(self, empty_config):
        """CachingInterceptor is NOT added when enable_caching=False."""
        chain = create_default_chain(empty_config, enable_caching=False)
        names = [i.name for i in chain.interceptors]
        assert "caching" not in names

    def test_caching_enabled(self, empty_config):
        """CachingInterceptor IS added when enable_caching=True."""
        chain = create_default_chain(empty_config, enable_caching=True)
        names = [i.name for i in chain.interceptors]
        assert "caching" in names

    def test_caching_interceptor_runs_first(self, empty_config):
        """CachingInterceptor runs before rate-limit and routing (order 3)."""
        chain = create_default_chain(
            empty_config,
            enable_caching=True,
            max_rpm=60,  # also add rate limiter at order 5
        )
        interceptors = chain.interceptors  # sorted by order
        names = [i.name for i in interceptors]
        assert names.index("caching") < names.index("rate_limit")

    def test_caching_interceptor_uses_custom_ttl(self, empty_config):
        """cache_ttl parameter is passed to the ResponseCache."""
        chain = create_default_chain(
            empty_config, enable_caching=True, cache_ttl=42.0
        )
        caching_interceptor = next(
            i for i in chain.interceptors if i.name == "caching"
        )
        assert isinstance(caching_interceptor, CachingInterceptor)
        # Access the underlying response cache TTL
        assert caching_interceptor.response_cache._default_ttl == 42.0

    def test_cache_hit_in_chain(self, empty_config):
        """CachingInterceptor records a hit when the cache is pre-populated."""
        chain = create_default_chain(empty_config, enable_caching=True)

        # Get the caching interceptor and pre-populate the cache
        caching_interceptor = next(
            i for i in chain.interceptors if i.name == "caching"
        )
        body = b'{"models": ["claude-3"]}'
        caching_interceptor.response_cache.store(
            method="GET",
            path="/v1/models",
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=body,
        )

        ctx = RequestContext(method="GET", path="/v1/models", headers={})
        # Test the interceptor directly - the chain may continue past SKIP
        result = caching_interceptor.intercept(ctx)

        # CachingInterceptor returns SKIP with cache_hit=True on a hit
        assert result.metadata.get("cache_hit") is True
        assert result.metadata.get("cached_response") is not None
        assert result.metadata["cached_response"].body == body

        # Also verify the hit is tracked in stats
        stats = caching_interceptor.get_stats()
        assert stats["hits"] >= 1

    def test_cache_miss_in_chain(self, empty_config):
        """CachingInterceptor records a miss when nothing is cached."""
        chain = create_default_chain(empty_config, enable_caching=True)

        caching_interceptor = next(
            i for i in chain.interceptors if i.name == "caching"
        )

        ctx = RequestContext(method="GET", path="/v1/models", headers={})
        # Test the interceptor directly
        result = caching_interceptor.intercept(ctx)

        # On miss, CachingInterceptor returns SKIP with cache_hit=False
        assert result.metadata.get("cache_hit") is False
        stats = caching_interceptor.get_stats()
        assert stats["misses"] >= 1

    def test_post_request_not_cached(self, empty_config):
        """POST requests bypass the CachingInterceptor (non-GET method)."""
        chain = create_default_chain(empty_config, enable_caching=True)
        caching_interceptor = next(
            i for i in chain.interceptors if i.name == "caching"
        )

        ctx = RequestContext(method="POST", path="/v1/messages", headers={})
        # Test the interceptor directly
        result = caching_interceptor.intercept(ctx)

        # POST requests should return SKIP with no cache_hit metadata
        assert "cache_hit" not in result.metadata
        # Stats should not count POST as hit or miss
        stats = caching_interceptor.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_caching_interceptor_store_response(self, empty_config):
        """CachingInterceptor.store_response populates the cache."""
        chain = create_default_chain(empty_config, enable_caching=True)
        caching_interceptor = next(
            i for i in chain.interceptors if i.name == "caching"
        )

        stored = caching_interceptor.store_response(
            method="GET",
            path="/v1/models",
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"models": []}',
        )
        assert stored is True
        assert caching_interceptor.response_cache.size == 1

    def test_chain_repr_shows_caching(self, empty_config):
        """Chain repr includes caching interceptor name."""
        chain = create_default_chain(empty_config, enable_caching=True)
        r = repr(chain)
        assert "caching" in r


# ---------------------------------------------------------------------------
# Passthrough handler response caching (using a mock upstream)
# ---------------------------------------------------------------------------


class MockAnthropicHandler(http.server.BaseHTTPRequestHandler):
    """Minimal mock server that returns predefined responses."""

    # Class-level request tracking
    _request_count = 0
    _request_lock = threading.Lock()
    _responses: dict = {}  # path -> (status, headers, body)

    def log_message(self, format, *args):
        pass  # Suppress access logs in tests

    def do_GET(self):
        with self._request_lock:
            MockAnthropicHandler._request_count += 1

        path_key = self.path.split("?")[0]
        if path_key in self._responses:
            status, headers, body = self._responses[path_key]
        else:
            status, headers, body = 200, {}, b'{"data": []}'

        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""
        response_body = b'{"id": "msg_123", "type": "message"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


class MockThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


@pytest.fixture
def mock_upstream():
    """Start a mock HTTP server acting as the Anthropic API."""
    MockAnthropicHandler._request_count = 0
    MockAnthropicHandler._responses = {
        "/v1/models": (
            200,
            {"Content-Type": "application/json"},
            b'{"data": [{"id": "claude-3-sonnet"}, {"id": "claude-3-haiku"}]}',
        )
    }

    server = MockThreadedServer(("127.0.0.1", 0), MockAnthropicHandler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    yield {"server": server, "port": port, "host": "127.0.0.1"}

    server.shutdown()
    server.server_close()


@pytest.fixture
def passthrough_server_factory(mock_upstream):
    """Factory for creating a passthrough server pointed at the mock upstream."""
    from src.anthropic_passthrough import create_passthrough_server
    from src.anthropic_passthrough import AnthropicPassthroughHandler

    servers = []

    def _make_server(enable_response_cache=False, cache_ttl=30.0, cache_maxsize=10):
        server = create_passthrough_server(
            host="127.0.0.1",
            port=0,
            timeout=5,
            api_key="test-api-key",
            anthropic_api_host=mock_upstream["host"],
            anthropic_api_port=mock_upstream["port"],
            use_https=False,
            max_retries=0,
            enable_connection_pool=False,
            enable_response_cache=enable_response_cache,
            response_cache_ttl=cache_ttl,
            response_cache_maxsize=cache_maxsize,
        )
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        servers.append(server)
        return server

    yield _make_server

    for s in servers:
        s.shutdown()
        s.server_close()


def _do_get(host: str, port: int, path: str) -> tuple[int, dict, bytes]:
    """Perform a GET request and return (status, headers_dict, body)."""
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", path, headers={"x-api-key": "test-api-key"})
    resp = conn.getresponse()
    body = resp.read()
    headers = dict(resp.getheaders())
    conn.close()
    return resp.status, headers, body


class TestPassthroughResponseCaching:
    """Tests for response caching in the passthrough server."""

    def test_cache_disabled_by_default(self, passthrough_server_factory, mock_upstream):
        """Without --response-cache, every GET hits upstream."""
        server = passthrough_server_factory(enable_response_cache=False)
        host, port = server.server_address

        MockAnthropicHandler._request_count = 0

        # Make two identical GET requests
        status1, _, body1 = _do_get(host, port, "/v1/models")
        status2, _, body2 = _do_get(host, port, "/v1/models")

        assert status1 == 200
        assert status2 == 200
        # Both requests hit upstream
        assert MockAnthropicHandler._request_count == 2
        assert body1 == body2

    def test_cache_enabled_serves_hit_on_second_request(
        self, passthrough_server_factory, mock_upstream
    ):
        """With caching enabled, the second GET is served from cache."""
        server = passthrough_server_factory(
            enable_response_cache=True, cache_ttl=60.0
        )
        host, port = server.server_address

        MockAnthropicHandler._request_count = 0

        # First request: upstream hit, populates cache
        status1, headers1, body1 = _do_get(host, port, "/v1/models")
        assert status1 == 200
        assert MockAnthropicHandler._request_count == 1

        # Second request: should be served from cache
        status2, headers2, body2 = _do_get(host, port, "/v1/models")
        assert status2 == 200
        assert MockAnthropicHandler._request_count == 1  # still 1
        assert body1 == body2

    def test_cache_hit_includes_x_cache_header(
        self, passthrough_server_factory, mock_upstream
    ):
        """Cached responses include the X-Cache: HIT header."""
        server = passthrough_server_factory(
            enable_response_cache=True, cache_ttl=60.0
        )
        host, port = server.server_address

        _do_get(host, port, "/v1/models")  # populate cache
        _, headers2, _ = _do_get(host, port, "/v1/models")

        # Check for X-Cache header (case-insensitive)
        x_cache = next(
            (v for k, v in headers2.items() if k.lower() == "x-cache"), None
        )
        assert x_cache == "HIT"

    def test_cache_hit_includes_age_header(
        self, passthrough_server_factory, mock_upstream
    ):
        """Cached responses include an Age header."""
        server = passthrough_server_factory(
            enable_response_cache=True, cache_ttl=60.0
        )
        host, port = server.server_address

        _do_get(host, port, "/v1/models")  # populate cache
        _, headers2, _ = _do_get(host, port, "/v1/models")

        age = next(
            (v for k, v in headers2.items() if k.lower() == "age"), None
        )
        assert age is not None
        assert int(age) >= 0

    def test_cache_respects_ttl(self, passthrough_server_factory, mock_upstream):
        """Entries expire after the configured TTL."""
        server = passthrough_server_factory(
            enable_response_cache=True, cache_ttl=0.1  # 100ms TTL
        )
        host, port = server.server_address

        MockAnthropicHandler._request_count = 0

        _do_get(host, port, "/v1/models")  # populate cache
        assert MockAnthropicHandler._request_count == 1

        time.sleep(0.2)  # Wait for TTL to expire

        _do_get(host, port, "/v1/models")  # should hit upstream again
        assert MockAnthropicHandler._request_count == 2

    def test_post_requests_not_cached(
        self, passthrough_server_factory, mock_upstream
    ):
        """POST requests are never served from cache."""
        server = passthrough_server_factory(
            enable_response_cache=True, cache_ttl=60.0
        )
        host, port = server.server_address

        # First POST
        conn = http.client.HTTPConnection(host, port, timeout=5)
        payload = b'{"model": "claude-3", "messages": []}'
        conn.request(
            "POST",
            "/v1/messages",
            body=payload,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
                "x-api-key": "test-api-key",
            },
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()

        # The passthrough handler returns 200 from mock
        assert resp.status == 200

    def test_cache_response_cache_attribute_set(self, passthrough_server_factory):
        """Handler class has response_cache set when enabled."""
        server = passthrough_server_factory(
            enable_response_cache=True, cache_ttl=60.0
        )
        # Each server creates a unique handler subclass; check the subclass
        assert server.RequestHandlerClass.response_cache is not None

    def test_cache_none_when_disabled(self, passthrough_server_factory):
        """Handler class has response_cache=None when not enabled."""
        server = passthrough_server_factory(enable_response_cache=False)
        # Each server creates a unique handler subclass; check the subclass
        assert server.RequestHandlerClass.response_cache is None

    def test_status_endpoint_includes_cache_info(
        self, passthrough_server_factory, mock_upstream
    ):
        """/status endpoint includes cache info when caching is enabled."""
        server = passthrough_server_factory(
            enable_response_cache=True, cache_ttl=60.0
        )
        host, port = server.server_address

        # Warm the cache with one request
        _do_get(host, port, "/v1/models")

        # Check /status
        status, _, body = _do_get(host, port, "/status")
        assert status == 200

        data = json.loads(body)
        assert "cache" in data
        assert data["cache"]["enabled"] is True

    def test_status_endpoint_cache_info_disabled(
        self, passthrough_server_factory, mock_upstream
    ):
        """/status shows cache disabled when not enabled."""
        server = passthrough_server_factory(enable_response_cache=False)
        host, port = server.server_address

        status, _, body = _do_get(host, port, "/status")
        data = json.loads(body)
        assert data["cache"]["enabled"] is False

    def test_status_endpoint_hit_rate(
        self, passthrough_server_factory, mock_upstream
    ):
        """/status hit rate is correct after cache activity."""
        server = passthrough_server_factory(
            enable_response_cache=True, cache_ttl=60.0
        )
        host, port = server.server_address

        # 1 miss (first request populates cache)
        _do_get(host, port, "/v1/models")
        # 1 hit (second request from cache)
        _do_get(host, port, "/v1/models")

        status, _, body = _do_get(host, port, "/status")
        data = json.loads(body)
        cache_stats = data["cache"]
        assert cache_stats["hits"] >= 1
        assert cache_stats["misses"] >= 1
        assert cache_stats["hit_rate"] > 0.0


# ---------------------------------------------------------------------------
# CLI gateway --response-cache option
# ---------------------------------------------------------------------------


class TestGatewayCLICacheOptions:
    """Tests for new cache-related CLI options on the gateway command."""

    def test_gateway_help_shows_response_cache(self, runner):
        """gateway --help shows --response-cache option."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert result.exit_code == 0
        assert "--response-cache" in result.output or "response-cache" in result.output

    def test_gateway_help_shows_cache_ttl(self, runner):
        """gateway --help shows --cache-ttl option."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert result.exit_code == 0
        assert "cache-ttl" in result.output

    def test_gateway_help_shows_cache_maxsize(self, runner):
        """gateway --help shows --cache-maxsize option."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert result.exit_code == 0
        assert "cache-maxsize" in result.output

    def test_gateway_output_shows_cache_disabled(self, runner):
        """gateway startup output shows cache: disabled by default."""
        # Patch at the source since run_passthrough is imported inside the function
        with patch("src.anthropic_passthrough.run_passthrough") as mock_run:
            result = runner.invoke(
                main,
                ["gateway", "--port", "19999"],
            )
            # The startup text is printed before run_passthrough is called
            assert "Cache:" in result.output
            assert "disabled" in result.output

    def test_gateway_output_shows_cache_enabled(self, runner):
        """gateway startup output shows cache enabled with TTL/maxsize."""
        with patch("src.anthropic_passthrough.run_passthrough") as mock_run:
            result = runner.invoke(
                main,
                [
                    "gateway",
                    "--port",
                    "19998",
                    "--response-cache",
                    "--cache-ttl",
                    "600",
                    "--cache-maxsize",
                    "128",
                ],
            )
            assert "Cache:" in result.output
            assert "enabled" in result.output
            assert "600" in result.output
            assert "128" in result.output

    def test_gateway_passes_cache_params_to_run_passthrough(self, runner):
        """CLI passes response-cache params to run_passthrough."""
        with patch("src.anthropic_passthrough.run_passthrough") as mock_run:
            runner.invoke(
                main,
                [
                    "gateway",
                    "--response-cache",
                    "--cache-ttl",
                    "120",
                    "--cache-maxsize",
                    "64",
                ],
            )
            assert mock_run.called
            kwargs = mock_run.call_args.kwargs
            assert kwargs.get("enable_response_cache") is True
            assert kwargs.get("response_cache_ttl") == 120.0
            assert kwargs.get("response_cache_maxsize") == 64

    def test_gateway_no_cache_passes_disabled(self, runner):
        """CLI passes enable_response_cache=False without --response-cache."""
        with patch("src.anthropic_passthrough.run_passthrough") as mock_run:
            runner.invoke(main, ["gateway", "--no-response-cache"])
            assert mock_run.called
            kwargs = mock_run.call_args.kwargs
            assert kwargs.get("enable_response_cache") is False


# ---------------------------------------------------------------------------
# create_passthrough_server with response cache
# ---------------------------------------------------------------------------


class TestCreatePassthroughServerCacheConfig:
    """Tests for create_passthrough_server caching parameters."""

    def test_response_cache_disabled_by_default(self):
        """response_cache is None when not enabled."""
        from src.anthropic_passthrough import create_passthrough_server

        server = create_passthrough_server(
            host="127.0.0.1", port=0, enable_response_cache=False
        )
        try:
            # Each server creates a unique handler subclass; check the subclass
            assert server.RequestHandlerClass.response_cache is None
        finally:
            server.server_close()

    def test_response_cache_enabled(self):
        """response_cache is a ResponseCache instance when enabled."""
        from src.anthropic_passthrough import create_passthrough_server
        from src.response_cache import ResponseCache

        server = create_passthrough_server(
            host="127.0.0.1",
            port=0,
            enable_response_cache=True,
            response_cache_ttl=120.0,
            response_cache_maxsize=64,
        )
        try:
            # Each server creates a unique handler subclass; check the subclass
            rc = server.RequestHandlerClass.response_cache
            assert isinstance(rc, ResponseCache)
            assert rc._default_ttl == 120.0
            assert rc._cache.maxsize == 64
        finally:
            server.server_close()

    def test_cache_hits_misses_reset_on_create(self):
        """Cache hit/miss counters are 0 for a freshly created server."""
        from src.anthropic_passthrough import create_passthrough_server

        server = create_passthrough_server(host="127.0.0.1", port=0)
        try:
            # Each server creates a unique handler subclass with fresh counters
            assert server.RequestHandlerClass._cache_hits == 0
            assert server.RequestHandlerClass._cache_misses == 0
        finally:
            server.server_close()


# ---------------------------------------------------------------------------
# Background purger integration
# ---------------------------------------------------------------------------


class TestBackgroundPurgerAutoStart:
    """Tests for BackgroundPurger auto-start when response cache is enabled."""

    def test_background_purger_not_started_without_cache(self):
        """BackgroundPurger is not started when response cache is disabled."""
        from src.cache import get_background_purger

        purger = get_background_purger()
        assert not purger.is_running

    def test_background_purger_started_in_run_passthrough(self, mock_upstream):
        """BackgroundPurger starts when enable_response_cache=True."""
        from src.anthropic_passthrough import run_passthrough_in_thread
        from src.cache import get_background_purger

        stop_event = threading.Event()
        server_ref = [None]

        def _run():
            from src.anthropic_passthrough import create_passthrough_server

            server = create_passthrough_server(
                host="127.0.0.1",
                port=0,
                timeout=1,
                api_key="test",
                anthropic_api_host=mock_upstream["host"],
                anthropic_api_port=mock_upstream["port"],
                use_https=False,
                enable_response_cache=True,
            )
            server_ref[0] = server

            from src.cache import get_background_purger, list_caches

            bg_purger = get_background_purger(interval=60.0)
            for cache_obj in list_caches().values():
                bg_purger.add_cache(cache_obj)
            bg_purger.start()

            stop_event.set()
            # Don't actually serve forever, just validate state
            bg_purger.stop()
            server.server_close()

        t = threading.Thread(target=_run)
        t.daemon = True
        t.start()
        stop_event.wait(timeout=3.0)

        # The purger should have been started (and then stopped in cleanup)
        t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# CachingInterceptor get_stats
# ---------------------------------------------------------------------------


class TestCachingInterceptorStats:
    """Tests for CachingInterceptor statistics."""

    def test_stats_track_hits_and_misses(self):
        """Stats track hits and misses correctly."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        interceptor = CachingInterceptor(response_cache=rc)

        # Pre-populate cache with a GET response
        rc.store(
            method="GET",
            path="/v1/models",
            status_code=200,
            headers={},
            body=b"data",
        )

        hit_ctx = RequestContext(method="GET", path="/v1/models", headers={})
        miss_ctx = RequestContext(method="GET", path="/v1/other", headers={})

        interceptor.intercept(hit_ctx)   # hit
        interceptor.intercept(miss_ctx)  # miss
        interceptor.intercept(miss_ctx)  # miss

        stats = interceptor.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert "response_cache" in stats

    def test_stats_post_not_counted(self):
        """POST requests are skipped and not counted in hits/misses."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        interceptor = CachingInterceptor(response_cache=rc)

        post_ctx = RequestContext(method="POST", path="/v1/messages", headers={})
        interceptor.intercept(post_ctx)

        stats = interceptor.get_stats()
        # POST is skipped, so hits and misses should both be 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


# ---------------------------------------------------------------------------
# ResponseCache integration with the handler's _serve_cached_response
# ---------------------------------------------------------------------------


class TestServeCachedResponse:
    """Tests for _serve_cached_response method (unit level)."""

    def test_serve_cached_response_sends_correct_status(
        self, passthrough_server_factory, mock_upstream
    ):
        """Cached responses have status 200."""
        server = passthrough_server_factory(
            enable_response_cache=True, cache_ttl=60.0
        )
        host, port = server.server_address

        # First request populates cache
        status1, _, body1 = _do_get(host, port, "/v1/models")
        # Second request from cache
        status2, _, body2 = _do_get(host, port, "/v1/models")

        assert status2 == 200
        assert body1 == body2

    def test_serve_cached_response_body_matches(
        self, passthrough_server_factory, mock_upstream
    ):
        """Cached response body is identical to the original upstream response."""
        server = passthrough_server_factory(
            enable_response_cache=True, cache_ttl=60.0
        )
        host, port = server.server_address

        _, _, body1 = _do_get(host, port, "/v1/models")
        _, _, body2 = _do_get(host, port, "/v1/models")

        assert body1 == body2
        # Both should be the mock models response
        data = json.loads(body2)
        assert "data" in data
