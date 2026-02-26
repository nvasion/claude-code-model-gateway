"""Tests for the HTTP proxy server."""

import http.server
import json
import threading
import urllib.error
import urllib.request

import pytest
from click.testing import CliRunner

from src.cli import main
from src.proxy import (
    ProxyRequestHandler,
    ThreadedHTTPServer,
    create_proxy_server,
    run_proxy_in_thread,
)


# ------------------------------------------------------------------ #
# Helpers: a tiny upstream HTTP server for the proxy to talk to
# ------------------------------------------------------------------ #


class _UpstreamHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler used as the upstream target during tests."""

    def log_message(self, format, *args):
        """Silence logs during tests."""
        pass

    def do_GET(self):
        body = json.dumps({"method": "GET", "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Custom-Header", "upstream-value")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request_body = self.rfile.read(length).decode() if length else ""
        body = json.dumps(
            {"method": "POST", "path": self.path, "body": request_body}
        ).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):
        length = int(self.headers.get("Content-Length", 0))
        request_body = self.rfile.read(length).decode() if length else ""
        body = json.dumps(
            {"method": "PUT", "path": self.path, "body": request_body}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_DELETE(self):
        body = json.dumps({"method": "DELETE", "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Head-Test", "ok")
        self.end_headers()


@pytest.fixture(scope="module")
def upstream_server():
    """Start a tiny upstream HTTP server for tests, torn down after the module."""
    server = ThreadedHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def proxy_server():
    """Start the proxy server on a random port for tests."""
    server, thread = run_proxy_in_thread(host="127.0.0.1", port=0)
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


# ------------------------------------------------------------------ #
# Helper to build proxy-style URLs
# ------------------------------------------------------------------ #


def _proxy_url(proxy_port: int, upstream_port: int, path: str = "/") -> str:
    """Build a URL routed through the proxy to the upstream server."""
    return f"http://127.0.0.1:{upstream_port}{path}"


def _make_request(
    proxy_port: int,
    upstream_port: int,
    path: str = "/",
    method: str = "GET",
    body: bytes | None = None,
    headers: dict | None = None,
):
    """Make an HTTP request through the proxy."""
    url = _proxy_url(proxy_port, upstream_port, path)
    req = urllib.request.Request(url, method=method, data=body)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    # Configure proxy
    proxy_handler = urllib.request.ProxyHandler(
        {"http": f"http://127.0.0.1:{proxy_port}"}
    )
    opener = urllib.request.build_opener(proxy_handler)
    return opener.open(req, timeout=10)


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestCreateProxyServer:
    """Tests for the create_proxy_server factory function."""

    def test_creates_server_instance(self):
        """create_proxy_server returns a ThreadedHTTPServer."""
        server = create_proxy_server(host="127.0.0.1", port=0)
        assert isinstance(server, ThreadedHTTPServer)
        server.server_close()

    def test_server_binds_to_given_address(self):
        """Server binds to the requested host."""
        server = create_proxy_server(host="127.0.0.1", port=0)
        host, port = server.server_address
        assert host == "127.0.0.1"
        assert port > 0
        server.server_close()


class TestRunProxyInThread:
    """Tests for the run_proxy_in_thread helper."""

    def test_returns_server_and_thread(self):
        """run_proxy_in_thread returns a (server, thread) tuple."""
        server, thread = run_proxy_in_thread(host="127.0.0.1", port=0)
        assert isinstance(server, ThreadedHTTPServer)
        assert isinstance(thread, threading.Thread)
        assert thread.is_alive()
        server.shutdown()
        server.server_close()


class TestProxyGET:
    """Tests for proxying GET requests."""

    def test_proxy_get_success(self, proxy_server, upstream_server):
        """Proxy forwards GET and returns upstream response."""
        proxy_port = proxy_server.server_address[1]
        upstream_port = upstream_server.server_address[1]

        resp = _make_request(proxy_port, upstream_port, path="/hello")
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["method"] == "GET"
        assert data["path"] == "/hello"

    def test_proxy_get_preserves_query_string(self, proxy_server, upstream_server):
        """Proxy preserves query parameters."""
        proxy_port = proxy_server.server_address[1]
        upstream_port = upstream_server.server_address[1]

        resp = _make_request(proxy_port, upstream_port, path="/search?q=test&page=1")
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert "q=test" in data["path"]

    def test_proxy_get_relays_custom_headers(self, proxy_server, upstream_server):
        """Proxy relays custom response headers from upstream."""
        proxy_port = proxy_server.server_address[1]
        upstream_port = upstream_server.server_address[1]

        resp = _make_request(proxy_port, upstream_port, path="/")
        assert resp.headers.get("X-Custom-Header") == "upstream-value"


class TestProxyPOST:
    """Tests for proxying POST requests."""

    def test_proxy_post_with_body(self, proxy_server, upstream_server):
        """Proxy forwards POST body and returns upstream response."""
        proxy_port = proxy_server.server_address[1]
        upstream_port = upstream_server.server_address[1]

        payload = json.dumps({"key": "value"}).encode()
        resp = _make_request(
            proxy_port,
            upstream_port,
            path="/submit",
            method="POST",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 201
        data = json.loads(resp.read().decode())
        assert data["method"] == "POST"
        assert data["path"] == "/submit"
        assert json.loads(data["body"]) == {"key": "value"}


class TestProxyPUT:
    """Tests for proxying PUT requests."""

    def test_proxy_put(self, proxy_server, upstream_server):
        """Proxy forwards PUT requests."""
        proxy_port = proxy_server.server_address[1]
        upstream_port = upstream_server.server_address[1]

        payload = b"updated-data"
        resp = _make_request(
            proxy_port,
            upstream_port,
            path="/resource/1",
            method="PUT",
            body=payload,
        )
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["method"] == "PUT"
        assert data["body"] == "updated-data"


class TestProxyDELETE:
    """Tests for proxying DELETE requests."""

    def test_proxy_delete(self, proxy_server, upstream_server):
        """Proxy forwards DELETE requests."""
        proxy_port = proxy_server.server_address[1]
        upstream_port = upstream_server.server_address[1]

        resp = _make_request(
            proxy_port, upstream_port, path="/resource/1", method="DELETE"
        )
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["method"] == "DELETE"


class TestProxyHEAD:
    """Tests for proxying HEAD requests."""

    def test_proxy_head(self, proxy_server, upstream_server):
        """Proxy forwards HEAD requests (no body in response)."""
        proxy_port = proxy_server.server_address[1]
        upstream_port = upstream_server.server_address[1]

        resp = _make_request(
            proxy_port, upstream_port, path="/resource", method="HEAD"
        )
        assert resp.status == 200
        assert resp.headers.get("X-Head-Test") == "ok"


class TestProxyErrorHandling:
    """Tests for proxy error handling."""

    def test_proxy_bad_gateway_on_unreachable_host(self, proxy_server):
        """Proxy returns 502 when upstream is unreachable."""
        proxy_port = proxy_server.server_address[1]
        try:
            # Try to proxy to a port that is (almost certainly) not listening
            _make_request(proxy_port, 19999, path="/")
            pytest.fail("Expected an HTTP error")
        except urllib.error.HTTPError as exc:
            assert exc.code == 502


class TestProxyCLI:
    """Tests for the proxy CLI command."""

    def test_proxy_help(self, runner):
        """The proxy command help text is correct."""
        result = runner.invoke(main, ["proxy", "--help"])
        assert result.exit_code == 0
        assert "Start the HTTP proxy server" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--timeout" in result.output
        assert "--verbose" in result.output

    def test_proxy_appears_in_main_help(self, runner):
        """The proxy command is listed in the main help."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "proxy" in result.output
