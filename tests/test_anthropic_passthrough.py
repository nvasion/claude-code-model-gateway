"""Tests for the Anthropic API pass-through server."""

import http.server
import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest
from click.testing import CliRunner

from src.anthropic_passthrough import (
    ALLOWED_PATHS,
    ANTHROPIC_API_HOST,
    ANTHROPIC_API_VERSION,
    DEFAULT_MAX_REQUEST_SIZE,
    GATEWAY_INTERNAL_PATHS,
    AnthropicPassthroughHandler,
    ConnectionPool,
    ThreadedGatewayServer,
    create_passthrough_server,
    run_passthrough_in_thread,
)
from src.cli import main


# ------------------------------------------------------------------ #
# Helpers: a mock Anthropic API server
# ------------------------------------------------------------------ #


class _MockAnthropicHandler(http.server.BaseHTTPRequestHandler):
    """Mock handler simulating the Anthropic API for testing.

    Responds to the same paths as the real Anthropic API with
    predictable responses for assertion in tests.
    """

    def log_message(self, format, *args):
        """Silence logs during tests."""
        pass

    def do_GET(self):
        """Handle GET requests (e.g., /v1/models)."""
        if self.path == "/v1/models" or self.path.startswith("/v1/models/"):
            body = json.dumps(
                {
                    "data": [
                        {"id": "claude-sonnet-4-20250514", "type": "model"},
                        {"id": "claude-3-5-sonnet-20241022", "type": "model"},
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("request-id", "test-req-123")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/v1/messages/batches"):
            body = json.dumps(
                {
                    "data": [
                        {
                            "id": "batch_test_123",
                            "type": "message_batch",
                            "processing_status": "ended",
                        }
                    ],
                    "has_more": False,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/v1/"):
            # Generic /v1/* handler for passthrough mode testing
            body = json.dumps(
                {"path": self.path, "method": "GET", "passthrough": True}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = json.dumps(
                {
                    "type": "error",
                    "error": {"type": "not_found", "message": "Not found"},
                }
            ).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        """Handle POST requests (e.g., /v1/messages)."""
        length = int(self.headers.get("Content-Length", 0))
        request_body = self.rfile.read(length).decode() if length else ""

        # Verify required headers
        api_key = self.headers.get("x-api-key", "")
        anthropic_version = self.headers.get("anthropic-version", "")

        if not api_key:
            body = json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "Missing API key",
                    },
                }
            ).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        try:
            request_data = json.loads(request_body)
        except json.JSONDecodeError:
            request_data = {}

        if self.path == "/v1/messages":
            is_streaming = request_data.get("stream", False)

            if is_streaming:
                self._handle_streaming_messages(request_data, api_key)
            else:
                self._handle_messages(request_data, api_key)

        elif self.path == "/v1/messages/count_tokens":
            self._handle_count_tokens(request_data, api_key)

        elif self.path == "/v1/messages/batches":
            self._handle_create_batch(request_data, api_key)

        elif self.path.startswith("/v1/"):
            # Generic /v1/* handler for passthrough mode testing
            body = json.dumps(
                {"path": self.path, "method": "POST", "passthrough": True}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            body = json.dumps(
                {
                    "type": "error",
                    "error": {"type": "not_found", "message": "Not found"},
                }
            ).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_DELETE(self):
        """Handle DELETE requests (e.g., batch cancellation)."""
        api_key = self.headers.get("x-api-key", "")
        if not api_key:
            body = json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "Missing API key",
                    },
                }
            ).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/v1/messages/batches/") and self.path.endswith(
            "/cancel"
        ):
            body = json.dumps(
                {
                    "id": "batch_test_cancel",
                    "type": "message_batch",
                    "processing_status": "canceling",
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/v1/"):
            body = json.dumps(
                {"path": self.path, "method": "DELETE", "passthrough": True}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = json.dumps(
                {
                    "type": "error",
                    "error": {"type": "not_found", "message": "Not found"},
                }
            ).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_PUT(self):
        """Handle PUT requests."""
        api_key = self.headers.get("x-api-key", "")
        if self.path.startswith("/v1/"):
            body = json.dumps(
                {"path": self.path, "method": "PUT", "api_key_present": bool(api_key)}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_PATCH(self):
        """Handle PATCH requests."""
        api_key = self.headers.get("x-api-key", "")
        if self.path.startswith("/v1/"):
            body = json.dumps(
                {"path": self.path, "method": "PATCH", "api_key_present": bool(api_key)}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_HEAD(self):
        """Handle HEAD requests."""
        if self.path.startswith("/v1/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

    def _handle_messages(self, request_data: dict, api_key: str):
        """Handle a non-streaming /v1/messages request."""
        model = request_data.get("model", "claude-sonnet-4-20250514")
        body = json.dumps(
            {
                "id": "msg_test_123",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello! This is a test response.",
                    }
                ],
                "model": model,
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 25,
                    "output_tokens": 10,
                },
            }
        ).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("request-id", "test-req-456")
        self.send_header("x-ratelimit-limit-requests", "1000")
        self.end_headers()
        self.wfile.write(body)

    def _handle_streaming_messages(self, request_data: dict, api_key: str):
        """Handle a streaming /v1/messages request with SSE events."""
        model = request_data.get("model", "claude-sonnet-4-20250514")

        # Build SSE events
        events = [
            f'event: message_start\ndata: {{"type":"message_start","message":{{"id":"msg_test_stream","type":"message","role":"assistant","content":[],"model":"{model}","usage":{{"input_tokens":25,"output_tokens":0}}}}}}\n\n',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n',
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
            'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n',
            'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]

        body = "".join(events).encode()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("request-id", "test-req-stream-789")
        self.end_headers()

        # Write events
        for event in events:
            self.wfile.write(event.encode())
            self.wfile.flush()

    def _handle_count_tokens(self, request_data: dict, api_key: str):
        """Handle a /v1/messages/count_tokens request."""
        body = json.dumps(
            {
                "input_tokens": 42,
            }
        ).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_create_batch(self, request_data: dict, api_key: str):
        """Handle a /v1/messages/batches request."""
        body = json.dumps(
            {
                "id": "batch_test_new",
                "type": "message_batch",
                "processing_status": "in_progress",
                "request_counts": {
                    "processing": 2,
                    "succeeded": 0,
                    "errored": 0,
                    "canceled": 0,
                    "expired": 0,
                },
            }
        ).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def mock_anthropic_server():
    """Start a mock Anthropic API server for tests."""
    server = ThreadedGatewayServer(("127.0.0.1", 0), _MockAnthropicHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def passthrough_server(mock_anthropic_server):
    """Start the pass-through server pointed at the mock Anthropic server.

    Uses the mock server's host:port as the upstream instead of the real
    Anthropic API, allowing fully offline testing.
    """
    mock_host, mock_port = mock_anthropic_server.server_address

    server, thread = run_passthrough_in_thread(
        host="127.0.0.1",
        port=0,
        timeout=10,
        api_key="test-api-key-123",
        anthropic_api_host="127.0.0.1",
        anthropic_api_port=mock_port,
        use_https=False,  # Use HTTP for mock server
        enable_connection_pool=False,  # Simpler for tests
    )

    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def passthrough_mode_server(mock_anthropic_server):
    """Start a pass-through mode server pointed at the mock Anthropic server.

    This server allows ALL /v1/* paths, not just the whitelisted set.
    """
    mock_host, mock_port = mock_anthropic_server.server_address

    server, thread = run_passthrough_in_thread(
        host="127.0.0.1",
        port=0,
        timeout=10,
        api_key="test-api-key-123",
        anthropic_api_host="127.0.0.1",
        anthropic_api_port=mock_port,
        use_https=False,
        passthrough_mode=True,
        enable_connection_pool=False,
    )

    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def pooled_passthrough_server(mock_anthropic_server):
    """Start a pass-through server with connection pooling enabled."""
    mock_host, mock_port = mock_anthropic_server.server_address

    server, thread = run_passthrough_in_thread(
        host="127.0.0.1",
        port=0,
        timeout=10,
        api_key="test-api-key-pool",
        anthropic_api_host="127.0.0.1",
        anthropic_api_port=mock_port,
        use_https=False,
        enable_connection_pool=True,
        max_pool_connections=5,
    )

    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #


def _gateway_request(
    gateway_port: int,
    path: str = "/v1/messages",
    method: str = "POST",
    body: bytes | None = None,
    headers: dict | None = None,
    timeout: int = 10,
):
    """Make a request to the pass-through gateway.

    Args:
        gateway_port: Port the gateway is listening on.
        path: API path to request.
        method: HTTP method.
        body: Request body bytes.
        headers: Extra headers to include.
        timeout: Request timeout in seconds.

    Returns:
        The urllib response object.
    """
    url = f"http://127.0.0.1:{gateway_port}{path}"
    req = urllib.request.Request(url, method=method, data=body)

    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    return urllib.request.urlopen(req, timeout=timeout)


def _make_messages_request(
    gateway_port: int,
    model: str = "claude-sonnet-4-20250514",
    messages: list | None = None,
    stream: bool = False,
    max_tokens: int = 1024,
    extra_headers: dict | None = None,
):
    """Make a /v1/messages request to the gateway.

    Args:
        gateway_port: Port the gateway is listening on.
        model: Model to use.
        messages: Message list. Defaults to a simple user message.
        stream: Whether to request streaming.
        max_tokens: Max tokens in response.
        extra_headers: Additional headers to include.

    Returns:
        The urllib response object.
    """
    if messages is None:
        messages = [{"role": "user", "content": "Hello"}]

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
    ).encode()

    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    return _gateway_request(
        gateway_port,
        path="/v1/messages",
        method="POST",
        body=payload,
        headers=headers,
    )


# ------------------------------------------------------------------ #
# Tests: Server creation
# ------------------------------------------------------------------ #


class TestCreatePassthroughServer:
    """Tests for the create_passthrough_server factory function."""

    def test_creates_server_instance(self):
        """create_passthrough_server returns a ThreadedGatewayServer."""
        server = create_passthrough_server(host="127.0.0.1", port=0)
        assert isinstance(server, ThreadedGatewayServer)
        server.server_close()

    def test_server_binds_to_given_address(self):
        """Server binds to the requested host."""
        server = create_passthrough_server(host="127.0.0.1", port=0)
        host, port = server.server_address
        assert host == "127.0.0.1"
        assert port > 0
        server.server_close()

    def test_custom_timeout(self):
        """Server handler gets the configured timeout."""
        server = create_passthrough_server(
            host="127.0.0.1", port=0, timeout=600
        )
        assert server.RequestHandlerClass.upstream_timeout == 600
        server.server_close()

    def test_custom_api_key(self):
        """Server handler gets the configured API key."""
        server = create_passthrough_server(
            host="127.0.0.1", port=0, api_key="sk-test-key"
        )
        assert server.RequestHandlerClass.api_key == "sk-test-key"
        server.server_close()

    def test_custom_anthropic_version(self):
        """Server handler gets the configured API version."""
        server = create_passthrough_server(
            host="127.0.0.1", port=0, anthropic_version="2024-01-01"
        )
        assert server.RequestHandlerClass.anthropic_version == "2024-01-01"
        server.server_close()

    def test_passthrough_mode_disabled_by_default(self):
        """Passthrough mode is disabled by default."""
        server = create_passthrough_server(host="127.0.0.1", port=0)
        assert server.RequestHandlerClass.passthrough_mode is False
        server.server_close()

    def test_passthrough_mode_enabled(self):
        """Passthrough mode can be enabled."""
        server = create_passthrough_server(
            host="127.0.0.1", port=0, passthrough_mode=True
        )
        assert server.RequestHandlerClass.passthrough_mode is True
        server.server_close()

    def test_custom_max_request_size(self):
        """Server handler gets the configured max request size."""
        server = create_passthrough_server(
            host="127.0.0.1", port=0, max_request_size=5000
        )
        assert server.RequestHandlerClass.max_request_size == 5000
        server.server_close()

    def test_connection_pool_enabled_by_default(self):
        """Connection pooling is enabled by default."""
        server = create_passthrough_server(host="127.0.0.1", port=0)
        assert server.RequestHandlerClass.connection_pool is not None
        server.server_close()

    def test_connection_pool_can_be_disabled(self):
        """Connection pooling can be disabled."""
        server = create_passthrough_server(
            host="127.0.0.1", port=0, enable_connection_pool=False
        )
        assert server.RequestHandlerClass.connection_pool is None
        server.server_close()

    def test_connection_pool_max_connections(self):
        """Connection pool max connections is configurable."""
        server = create_passthrough_server(
            host="127.0.0.1", port=0, max_pool_connections=20
        )
        pool = server.RequestHandlerClass.connection_pool
        assert pool is not None
        assert pool.max_connections == 20
        server.server_close()

    def test_server_start_time_set(self):
        """Server start time is set on creation."""
        server = create_passthrough_server(host="127.0.0.1", port=0)
        assert server.RequestHandlerClass._server_start_time > 0
        server.server_close()


class TestRunPassthroughInThread:
    """Tests for the run_passthrough_in_thread helper."""

    def test_returns_server_and_thread(self):
        """run_passthrough_in_thread returns a (server, thread) tuple."""
        server, thread = run_passthrough_in_thread(
            host="127.0.0.1", port=0
        )
        assert isinstance(server, ThreadedGatewayServer)
        assert isinstance(thread, threading.Thread)
        assert thread.is_alive()
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------ #
# Tests: Connection pool
# ------------------------------------------------------------------ #


class TestConnectionPool:
    """Tests for the ConnectionPool class."""

    def test_create_pool(self):
        """ConnectionPool can be created."""
        pool = ConnectionPool(
            host="127.0.0.1",
            port=8080,
            use_https=False,
            timeout=10,
            max_connections=5,
        )
        assert pool.host == "127.0.0.1"
        assert pool.port == 8080
        assert pool.max_connections == 5
        assert pool.pool_size == 0
        assert pool.total_created == 0

    def test_get_connection_creates_new(self):
        """get_connection creates a new connection when pool is empty."""
        pool = ConnectionPool(
            host="127.0.0.1",
            port=8080,
            use_https=False,
            timeout=10,
        )
        conn = pool.get_connection()
        assert conn is not None
        assert pool.total_created == 1
        pool.close_connection(conn)

    def test_return_and_reuse_connection(self):
        """Returned connections can be reused."""
        pool = ConnectionPool(
            host="127.0.0.1",
            port=8080,
            use_https=False,
            timeout=10,
        )
        conn = pool.get_connection()
        pool.return_connection(conn)
        assert pool.pool_size == 1

    def test_close_all(self):
        """close_all empties the pool."""
        pool = ConnectionPool(
            host="127.0.0.1",
            port=8080,
            use_https=False,
            timeout=10,
        )
        conn = pool.get_connection()
        pool.return_connection(conn)
        assert pool.pool_size == 1
        pool.close_all()
        assert pool.pool_size == 0

    def test_pool_max_connections(self):
        """Pool respects max_connections limit."""
        pool = ConnectionPool(
            host="127.0.0.1",
            port=8080,
            use_https=False,
            timeout=10,
            max_connections=2,
        )
        conns = [pool.get_connection() for _ in range(5)]
        for conn in conns:
            pool.return_connection(conn)
        # Only 2 should be kept
        assert pool.pool_size == 2
        pool.close_all()


# ------------------------------------------------------------------ #
# Tests: Path validation
# ------------------------------------------------------------------ #


class TestPathValidation:
    """Tests for allowed path validation."""

    def test_allowed_paths_include_messages(self):
        """/v1/messages is an allowed path."""
        assert "/v1/messages" in ALLOWED_PATHS

    def test_allowed_paths_include_count_tokens(self):
        """/v1/messages/count_tokens is an allowed path."""
        assert "/v1/messages/count_tokens" in ALLOWED_PATHS

    def test_allowed_paths_include_models(self):
        """/v1/models is an allowed path."""
        assert "/v1/models" in ALLOWED_PATHS

    def test_allowed_paths_include_batches(self):
        """/v1/messages/batches is an allowed path."""
        assert "/v1/messages/batches" in ALLOWED_PATHS

    def test_disallowed_path_returns_404(self, passthrough_server):
        """Requesting a non-allowed path returns 404."""
        port = passthrough_server.server_address[1]
        try:
            _gateway_request(port, path="/v1/completions", method="POST")
            pytest.fail("Expected HTTP 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            error_data = json.loads(exc.read().decode())
            assert error_data["type"] == "error"
            assert error_data["error"]["type"] == "not_found"

    def test_root_path_returns_404(self, passthrough_server):
        """Requesting / returns 404."""
        port = passthrough_server.server_address[1]
        try:
            _gateway_request(port, path="/", method="GET")
            pytest.fail("Expected HTTP 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404

    def test_internal_paths_are_defined(self):
        """Internal gateway paths are defined."""
        assert "/health" in GATEWAY_INTERNAL_PATHS
        assert "/status" in GATEWAY_INTERNAL_PATHS


# ------------------------------------------------------------------ #
# Tests: Non-streaming message requests
# ------------------------------------------------------------------ #


class TestNonStreamingMessages:
    """Tests for non-streaming /v1/messages pass-through."""

    def test_basic_message_request(self, passthrough_server):
        """Basic message request returns a valid response."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(port)
        assert resp.status == 200

        data = json.loads(resp.read().decode())
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0
        assert data["content"][0]["type"] == "text"

    def test_message_request_preserves_model(self, passthrough_server):
        """Model parameter is forwarded to the upstream."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(
            port, model="claude-3-5-sonnet-20241022"
        )
        data = json.loads(resp.read().decode())
        assert data["model"] == "claude-3-5-sonnet-20241022"

    def test_message_response_content_type(self, passthrough_server):
        """Response has application/json content type."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(port)
        assert "application/json" in resp.headers.get("Content-Type", "")

    def test_message_response_has_usage(self, passthrough_server):
        """Response includes usage statistics."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(port)
        data = json.loads(resp.read().decode())
        assert "usage" in data
        assert "input_tokens" in data["usage"]
        assert "output_tokens" in data["usage"]

    def test_upstream_headers_forwarded(self, passthrough_server):
        """Response headers from the upstream are forwarded."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(port)
        # Mock server sends request-id header
        assert resp.headers.get("request-id") == "test-req-456"

    def test_cors_header_present(self, passthrough_server):
        """Response includes CORS header."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(port)
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_request_id_header_returned(self, passthrough_server):
        """Response includes x-request-id header."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(port)
        request_id = resp.headers.get("x-request-id")
        assert request_id is not None
        assert len(request_id) > 0

    def test_custom_request_id_propagated(self, passthrough_server):
        """Client-provided x-request-id is propagated."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(
            port,
            extra_headers={"x-request-id": "custom-req-123"},
        )
        assert resp.headers.get("x-request-id") == "custom-req-123"


# ------------------------------------------------------------------ #
# Tests: Streaming message requests
# ------------------------------------------------------------------ #


class TestStreamingMessages:
    """Tests for streaming /v1/messages pass-through."""

    def test_streaming_request_returns_sse(self, passthrough_server):
        """Streaming request returns text/event-stream content type."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(port, stream=True)
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")

    def test_streaming_response_contains_events(self, passthrough_server):
        """Streaming response contains SSE events."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(port, stream=True)
        body = resp.read().decode()

        # Should contain SSE event markers
        assert "event: message_start" in body
        assert "event: content_block_start" in body
        assert "event: content_block_delta" in body
        assert "event: message_stop" in body

    def test_streaming_response_has_text_deltas(self, passthrough_server):
        """Streaming response includes text delta events."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(port, stream=True)
        body = resp.read().decode()

        assert "Hello" in body
        assert "world" in body

    def test_streaming_response_has_request_id(self, passthrough_server):
        """Streaming response includes x-request-id header."""
        port = passthrough_server.server_address[1]
        resp = _make_messages_request(port, stream=True)
        assert resp.headers.get("x-request-id") is not None


# ------------------------------------------------------------------ #
# Tests: Token counting
# ------------------------------------------------------------------ #


class TestCountTokens:
    """Tests for /v1/messages/count_tokens pass-through."""

    def test_count_tokens_request(self, passthrough_server):
        """Token counting returns a valid response."""
        port = passthrough_server.server_address[1]

        payload = json.dumps(
            {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Hello, world!"}],
            }
        ).encode()

        resp = _gateway_request(
            port,
            path="/v1/messages/count_tokens",
            method="POST",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert "input_tokens" in data
        assert data["input_tokens"] == 42


# ------------------------------------------------------------------ #
# Tests: Model listing
# ------------------------------------------------------------------ #


class TestModelListing:
    """Tests for /v1/models pass-through."""

    def test_list_models(self, passthrough_server):
        """GET /v1/models returns model list."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(port, path="/v1/models", method="GET")
        assert resp.status == 200

        data = json.loads(resp.read().decode())
        assert "data" in data
        assert len(data["data"]) == 2

    def test_model_sub_paths(self, passthrough_server):
        """GET /v1/models/<id> returns a model."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(
            port, path="/v1/models/claude-sonnet-4-20250514", method="GET"
        )
        assert resp.status == 200


# ------------------------------------------------------------------ #
# Tests: Batch API
# ------------------------------------------------------------------ #


class TestBatchAPI:
    """Tests for /v1/messages/batches pass-through."""

    def test_list_batches(self, passthrough_server):
        """GET /v1/messages/batches returns batch list."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(
            port, path="/v1/messages/batches", method="GET"
        )
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert "data" in data

    def test_create_batch(self, passthrough_server):
        """POST /v1/messages/batches creates a batch."""
        port = passthrough_server.server_address[1]
        payload = json.dumps(
            {
                "requests": [
                    {
                        "custom_id": "req-1",
                        "params": {
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 1024,
                            "messages": [
                                {"role": "user", "content": "Hello"}
                            ],
                        },
                    }
                ]
            }
        ).encode()

        resp = _gateway_request(
            port,
            path="/v1/messages/batches",
            method="POST",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["type"] == "message_batch"

    def test_get_batch_by_id(self, passthrough_server):
        """GET /v1/messages/batches/{id} returns batch details."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(
            port, path="/v1/messages/batches/batch_123", method="GET"
        )
        assert resp.status == 200

    def test_get_batch_results(self, passthrough_server):
        """GET /v1/messages/batches/{id}/results works."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(
            port, path="/v1/messages/batches/batch_123/results", method="GET"
        )
        assert resp.status == 200


# ------------------------------------------------------------------ #
# Tests: HTTP methods
# ------------------------------------------------------------------ #


class TestHTTPMethods:
    """Tests for full HTTP method support."""

    def test_delete_method(self, passthrough_mode_server):
        """DELETE requests are forwarded in passthrough mode."""
        port = passthrough_mode_server.server_address[1]
        resp = _gateway_request(
            port,
            path="/v1/messages/batches/batch_123/cancel",
            method="DELETE",
        )
        assert resp.status == 200

    def test_put_method(self, passthrough_mode_server):
        """PUT requests are forwarded in passthrough mode."""
        port = passthrough_mode_server.server_address[1]
        resp = _gateway_request(
            port,
            path="/v1/some/resource",
            method="PUT",
            body=b'{"key": "value"}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["method"] == "PUT"

    def test_patch_method(self, passthrough_mode_server):
        """PATCH requests are forwarded in passthrough mode."""
        port = passthrough_mode_server.server_address[1]
        resp = _gateway_request(
            port,
            path="/v1/some/resource",
            method="PATCH",
            body=b'{"key": "value"}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["method"] == "PATCH"

    def test_head_method(self, passthrough_mode_server):
        """HEAD requests are forwarded in passthrough mode."""
        port = passthrough_mode_server.server_address[1]
        resp = _gateway_request(
            port,
            path="/v1/models",
            method="HEAD",
        )
        assert resp.status == 200

    def test_options_returns_cors(self, passthrough_server):
        """OPTIONS request returns CORS headers."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(
            port, path="/v1/messages", method="OPTIONS"
        )
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        methods = resp.headers.get("Access-Control-Allow-Methods", "")
        assert "POST" in methods
        assert "DELETE" in methods
        assert "PUT" in methods
        assert "PATCH" in methods


# ------------------------------------------------------------------ #
# Tests: Pass-through mode
# ------------------------------------------------------------------ #


class TestPassthroughMode:
    """Tests for pass-through mode (all /v1/* paths forwarded)."""

    def test_arbitrary_v1_path_allowed(self, passthrough_mode_server):
        """Any /v1/* path is forwarded in passthrough mode."""
        port = passthrough_mode_server.server_address[1]
        resp = _gateway_request(
            port, path="/v1/some/custom/endpoint", method="GET"
        )
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["passthrough"] is True

    def test_non_v1_path_rejected(self, passthrough_mode_server):
        """Non /v1/ paths are rejected even in passthrough mode."""
        port = passthrough_mode_server.server_address[1]
        try:
            _gateway_request(port, path="/api/custom", method="GET")
            pytest.fail("Expected HTTP 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            error_data = json.loads(exc.read().decode())
            assert "must start with /v1/" in error_data["error"]["message"]

    def test_messages_still_work(self, passthrough_mode_server):
        """Standard /v1/messages still works in passthrough mode."""
        port = passthrough_mode_server.server_address[1]
        resp = _make_messages_request(port)
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["type"] == "message"

    def test_strict_mode_rejects_unknown_paths(self, passthrough_server):
        """Strict mode rejects non-whitelisted /v1/ paths."""
        port = passthrough_server.server_address[1]
        try:
            _gateway_request(
                port, path="/v1/some/custom/endpoint", method="GET"
            )
            pytest.fail("Expected HTTP 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404


# ------------------------------------------------------------------ #
# Tests: Health endpoint
# ------------------------------------------------------------------ #


class TestHealthEndpoint:
    """Tests for the /health gateway endpoint."""

    def test_health_endpoint_returns_200(self, passthrough_server):
        """GET /health returns 200."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(port, path="/health", method="GET")
        assert resp.status == 200

    def test_health_response_format(self, passthrough_server):
        """Health response is valid JSON with expected fields."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(port, path="/health", method="GET")
        data = json.loads(resp.read().decode())

        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")
        assert "gateway" in data
        assert data["gateway"] == "anthropic-passthrough"
        assert "upstream" in data
        assert "version" in data

    def test_health_cache_control(self, passthrough_server):
        """Health response has no-cache header."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(port, path="/health", method="GET")
        assert resp.headers.get("Cache-Control") == "no-cache"

    def test_health_not_forwarded_to_upstream(self, passthrough_server):
        """Health endpoint is served locally, not forwarded."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(port, path="/health", method="GET")
        data = json.loads(resp.read().decode())
        # Should be a local response, not from mock server
        assert "data" not in data  # Mock server doesn't return this format


# ------------------------------------------------------------------ #
# Tests: Status endpoint
# ------------------------------------------------------------------ #


class TestStatusEndpoint:
    """Tests for the /status gateway endpoint."""

    def test_status_endpoint_returns_200(self, passthrough_server):
        """GET /status returns 200."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(port, path="/status", method="GET")
        assert resp.status == 200

    def test_status_response_format(self, passthrough_server):
        """Status response is valid JSON with expected fields."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(port, path="/status", method="GET")
        data = json.loads(resp.read().decode())

        assert "gateway" in data
        assert data["gateway"] == "anthropic-passthrough"
        assert "mode" in data
        assert data["mode"] in ("strict", "passthrough")
        assert "upstream" in data
        assert "host" in data["upstream"]
        assert "port" in data["upstream"]
        assert "config" in data
        assert "timeout" in data["config"]
        assert "max_retries" in data["config"]
        assert "stats" in data
        assert "total_requests" in data["stats"]

    def test_status_shows_connection_pool_info(self, pooled_passthrough_server):
        """Status response includes connection pool info when pooling enabled."""
        port = pooled_passthrough_server.server_address[1]
        resp = _gateway_request(port, path="/status", method="GET")
        data = json.loads(resp.read().decode())

        assert "connection_pool" in data
        pool_info = data["connection_pool"]
        assert "pool_size" in pool_info
        assert "total_created" in pool_info
        assert "max_connections" in pool_info

    def test_status_passthrough_mode_shown(self, passthrough_mode_server):
        """Status shows passthrough mode correctly."""
        port = passthrough_mode_server.server_address[1]
        resp = _gateway_request(port, path="/status", method="GET")
        data = json.loads(resp.read().decode())
        assert data["mode"] == "passthrough"


# ------------------------------------------------------------------ #
# Tests: Authentication
# ------------------------------------------------------------------ #


class TestAuthentication:
    """Tests for API key handling."""

    def test_no_api_key_returns_401(self):
        """Server without API key returns 401."""
        # Create a server with no API key configured
        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            api_key=None,
            enable_connection_pool=False,
        )
        port = server.server_address[1]

        # Clear env var for this test
        original_key = os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            # Make a request without any API key
            payload = json.dumps(
                {
                    "model": "claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1024,
                }
            ).encode()

            try:
                _gateway_request(
                    port,
                    path="/v1/messages",
                    method="POST",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )
                pytest.fail("Expected HTTP 401")
            except urllib.error.HTTPError as exc:
                assert exc.code == 401
                error_data = json.loads(exc.read().decode())
                assert error_data["error"]["type"] == "authentication_error"
        finally:
            if original_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_key
            server.shutdown()
            server.server_close()

    def test_client_header_api_key_used(self, mock_anthropic_server):
        """API key from client x-api-key header is forwarded."""
        mock_port = mock_anthropic_server.server_address[1]

        # Server with no API key, so it must come from the client
        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            api_key=None,
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=mock_port,
            use_https=False,
            enable_connection_pool=False,
        )
        port = server.server_address[1]

        # Clear env var for this test
        original_key = os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            resp = _make_messages_request(
                port,
                extra_headers={"x-api-key": "client-provided-key"},
            )
            assert resp.status == 200
        finally:
            if original_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_key
            server.shutdown()
            server.server_close()

    def test_401_includes_request_id(self):
        """401 error response includes x-request-id header."""
        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            api_key=None,
            enable_connection_pool=False,
        )
        port = server.server_address[1]

        original_key = os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            payload = json.dumps(
                {
                    "model": "claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1024,
                }
            ).encode()

            try:
                _gateway_request(
                    port,
                    path="/v1/messages",
                    method="POST",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )
                pytest.fail("Expected HTTP 401")
            except urllib.error.HTTPError as exc:
                assert exc.code == 401
                # Should have x-request-id
                assert exc.headers.get("x-request-id") is not None
        finally:
            if original_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_key
            server.shutdown()
            server.server_close()


# ------------------------------------------------------------------ #
# Tests: CORS / OPTIONS
# ------------------------------------------------------------------ #


class TestCORS:
    """Tests for CORS support."""

    def test_options_request(self, passthrough_server):
        """OPTIONS request returns CORS headers."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(
            port, path="/v1/messages", method="OPTIONS"
        )
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")

    def test_options_includes_extended_methods(self, passthrough_server):
        """OPTIONS response includes all HTTP methods."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(
            port, path="/v1/messages", method="OPTIONS"
        )
        methods = resp.headers.get("Access-Control-Allow-Methods", "")
        for method in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]:
            assert method in methods

    def test_options_includes_anthropic_beta_header(self, passthrough_server):
        """OPTIONS response allows anthropic-beta header."""
        port = passthrough_server.server_address[1]
        resp = _gateway_request(
            port, path="/v1/messages", method="OPTIONS"
        )
        allowed_headers = resp.headers.get("Access-Control-Allow-Headers", "")
        assert "anthropic-beta" in allowed_headers
        assert "x-api-key" in allowed_headers
        assert "x-request-id" in allowed_headers


# ------------------------------------------------------------------ #
# Tests: Error handling
# ------------------------------------------------------------------ #


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_bad_upstream_returns_502(self):
        """Connection failure to upstream returns 502."""
        # Point at a non-existent upstream
        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            timeout=2,
            api_key="test-key",
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=19998,  # Port nothing listens on
            use_https=False,
            enable_connection_pool=False,
        )
        port = server.server_address[1]

        try:
            payload = json.dumps(
                {
                    "model": "claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1024,
                }
            ).encode()

            try:
                _gateway_request(
                    port,
                    path="/v1/messages",
                    method="POST",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )
                pytest.fail("Expected HTTP error")
            except urllib.error.HTTPError as exc:
                assert exc.code == 502
                error_data = json.loads(exc.read().decode())
                assert error_data["type"] == "error"
                assert error_data["error"]["type"] == "api_error"
        finally:
            server.shutdown()
            server.server_close()

    def test_error_response_is_json(self, passthrough_server):
        """Error responses are in JSON format."""
        port = passthrough_server.server_address[1]
        try:
            _gateway_request(port, path="/v1/completions", method="POST")
            pytest.fail("Expected HTTP 404")
        except urllib.error.HTTPError as exc:
            content_type = exc.headers.get("Content-Type", "")
            assert "application/json" in content_type

    def test_error_includes_request_id(self, passthrough_server):
        """Error responses include x-request-id header."""
        port = passthrough_server.server_address[1]
        try:
            _gateway_request(port, path="/v1/completions", method="POST")
            pytest.fail("Expected HTTP 404")
        except urllib.error.HTTPError as exc:
            assert exc.headers.get("x-request-id") is not None


# ------------------------------------------------------------------ #
# Tests: Request size validation
# ------------------------------------------------------------------ #


class TestRequestSizeValidation:
    """Tests for request body size validation."""

    def test_oversized_request_returns_413(self, mock_anthropic_server):
        """Request larger than max_request_size returns 413."""
        mock_port = mock_anthropic_server.server_address[1]

        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            timeout=10,
            api_key="test-key",
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=mock_port,
            use_https=False,
            max_request_size=100,  # Very small limit for testing
            enable_connection_pool=False,
        )
        port = server.server_address[1]

        try:
            # Create a request body larger than 100 bytes
            large_payload = json.dumps(
                {
                    "model": "claude-sonnet-4-20250514",
                    "messages": [
                        {"role": "user", "content": "A" * 200}
                    ],
                    "max_tokens": 1024,
                }
            ).encode()

            assert len(large_payload) > 100

            try:
                _gateway_request(
                    port,
                    path="/v1/messages",
                    method="POST",
                    body=large_payload,
                    headers={"Content-Type": "application/json"},
                )
                pytest.fail("Expected HTTP 413")
            except urllib.error.HTTPError as exc:
                assert exc.code == 413
                error_data = json.loads(exc.read().decode())
                assert error_data["error"]["type"] == "invalid_request_error"
                assert "too large" in error_data["error"]["message"]
        finally:
            server.shutdown()
            server.server_close()

    def test_unlimited_request_size(self, mock_anthropic_server):
        """max_request_size=0 allows any size."""
        mock_port = mock_anthropic_server.server_address[1]

        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            timeout=10,
            api_key="test-key",
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=mock_port,
            use_https=False,
            max_request_size=0,  # Unlimited
            enable_connection_pool=False,
        )
        port = server.server_address[1]

        try:
            payload = json.dumps(
                {
                    "model": "claude-sonnet-4-20250514",
                    "messages": [
                        {"role": "user", "content": "A" * 10000}
                    ],
                    "max_tokens": 1024,
                }
            ).encode()

            resp = _gateway_request(
                port,
                path="/v1/messages",
                method="POST",
                body=payload,
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 200
        finally:
            server.shutdown()
            server.server_close()


# ------------------------------------------------------------------ #
# Tests: Connection pooling
# ------------------------------------------------------------------ #


class TestConnectionPooling:
    """Tests for connection pooling in the pass-through server."""

    def test_pooled_server_handles_requests(self, pooled_passthrough_server):
        """Pooled server handles requests correctly."""
        port = pooled_passthrough_server.server_address[1]
        resp = _make_messages_request(port)
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["type"] == "message"

    def test_pooled_server_multiple_requests(self, pooled_passthrough_server):
        """Pooled server handles multiple consecutive requests."""
        port = pooled_passthrough_server.server_address[1]
        for i in range(5):
            resp = _make_messages_request(port)
            assert resp.status == 200


# ------------------------------------------------------------------ #
# Tests: CLI integration
# ------------------------------------------------------------------ #


class TestGatewayCLI:
    """Tests for the gateway CLI command."""

    def test_gateway_help(self, runner):
        """The gateway command help text is correct."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert result.exit_code == 0
        assert "Anthropic API pass-through gateway" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--timeout" in result.output
        assert "--api-key" in result.output
        assert "--anthropic-version" in result.output
        assert "--verbose" in result.output

    def test_gateway_appears_in_main_help(self, runner):
        """The gateway command is listed in the main help."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "gateway" in result.output

    def test_gateway_help_shows_endpoints(self, runner):
        """The gateway help shows supported endpoints."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert "/v1/messages" in result.output
        assert "/v1/messages/count_tokens" in result.output
        assert "/v1/models" in result.output

    def test_gateway_help_mentions_streaming(self, runner):
        """The gateway help mentions streaming support."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert "streaming" in result.output.lower() or "SSE" in result.output

    def test_gateway_help_shows_passthrough_mode(self, runner):
        """The gateway help shows passthrough mode option."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert "--passthrough-mode" in result.output

    def test_gateway_help_shows_health_endpoint(self, runner):
        """The gateway help shows health and status endpoints."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert "/health" in result.output
        assert "/status" in result.output

    def test_gateway_help_shows_batches_endpoint(self, runner):
        """The gateway help shows batches endpoint."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert "/v1/messages/batches" in result.output

    def test_gateway_help_shows_max_request_size(self, runner):
        """The gateway help shows max-request-size option."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert "--max-request-size" in result.output

    def test_gateway_help_shows_pool_options(self, runner):
        """The gateway help shows connection pool options."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert "--no-connection-pool" in result.output
        assert "--pool-size" in result.output

    def test_gateway_help_shows_features(self, runner):
        """The gateway help mentions key features."""
        result = runner.invoke(main, ["gateway", "--help"])
        output = result.output.lower()
        assert "correlation" in output or "request-id" in output
        assert "connection pool" in output or "pool" in output

    def test_gateway_help_shows_response_cache_option(self, runner):
        """The gateway help shows the response cache toggle option."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert "--response-cache" in result.output

    def test_gateway_help_shows_cache_ttl_option(self, runner):
        """The gateway help shows the cache TTL option."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert "--cache-ttl" in result.output

    def test_gateway_help_shows_cache_maxsize_option(self, runner):
        """The gateway help shows the cache maxsize option."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert "--cache-maxsize" in result.output


# ------------------------------------------------------------------ #
# Tests: Server isolation — handler subclass per server
# ------------------------------------------------------------------ #


class TestServerIsolation:
    """Tests that each server instance uses its own independent configuration.

    Verifies that create_passthrough_server does not mutate the base
    AnthropicPassthroughHandler class, allowing multiple servers with
    different settings to coexist without interfering with each other.
    """

    def test_base_class_timeout_unchanged_after_server_creation(self):
        """Creating a server with custom timeout does not alter the base class."""
        original_timeout = AnthropicPassthroughHandler.upstream_timeout

        server = create_passthrough_server(
            host="127.0.0.1", port=0, timeout=999
        )
        try:
            # The subclass should have the custom timeout
            assert server.RequestHandlerClass.upstream_timeout == 999
            # The base class must be unchanged
            assert AnthropicPassthroughHandler.upstream_timeout == original_timeout
        finally:
            server.server_close()

    def test_base_class_passthrough_mode_unchanged_after_server_creation(self):
        """Creating a server with passthrough_mode=True does not alter the base class."""
        server = create_passthrough_server(
            host="127.0.0.1", port=0, passthrough_mode=True
        )
        try:
            assert server.RequestHandlerClass.passthrough_mode is True
            assert AnthropicPassthroughHandler.passthrough_mode is False
        finally:
            server.server_close()

    def test_two_servers_have_independent_timeouts(self):
        """Two servers with different timeouts do not interfere with each other."""
        server_a = create_passthrough_server(
            host="127.0.0.1", port=0, timeout=111
        )
        server_b = create_passthrough_server(
            host="127.0.0.1", port=0, timeout=222
        )
        try:
            assert server_a.RequestHandlerClass.upstream_timeout == 111
            assert server_b.RequestHandlerClass.upstream_timeout == 222
        finally:
            server_a.server_close()
            server_b.server_close()

    def test_two_servers_have_independent_request_counts(self, mock_anthropic_server):
        """Two servers track their own request counts independently."""
        mock_port = mock_anthropic_server.server_address[1]

        server_a, _ = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            api_key="test-key-a",
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=mock_port,
            use_https=False,
            enable_connection_pool=False,
        )
        server_b, _ = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            api_key="test-key-b",
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=mock_port,
            use_https=False,
            enable_connection_pool=False,
        )

        port_a = server_a.server_address[1]
        port_b = server_b.server_address[1]

        try:
            # Make one request to server A and two to server B
            _make_messages_request(port_a)
            _make_messages_request(port_b)
            _make_messages_request(port_b)

            # Allow handler threads to finish incrementing
            time.sleep(0.1)

            count_a = server_a.RequestHandlerClass._request_count
            count_b = server_b.RequestHandlerClass._request_count

            assert count_a == 1, f"Server A count should be 1, got {count_a}"
            assert count_b == 2, f"Server B count should be 2, got {count_b}"
        finally:
            server_a.shutdown()
            server_a.server_close()
            server_b.shutdown()
            server_b.server_close()

    def test_handler_subclass_name(self):
        """The handler subclass has the expected internal name."""
        server = create_passthrough_server(host="127.0.0.1", port=0)
        try:
            assert server.RequestHandlerClass.__name__ == "_AnthropicPassthroughHandlerInstance"
            # Verify it is indeed a subclass of the base handler
            assert issubclass(
                server.RequestHandlerClass, AnthropicPassthroughHandler
            )
        finally:
            server.server_close()


# ------------------------------------------------------------------ #
# Tests: Response cache integration
# ------------------------------------------------------------------ #


class TestResponseCache:
    """Integration tests for response caching in the pass-through server.

    Verifies that GET/HEAD responses are cached and served from cache on
    subsequent identical requests without hitting the upstream.
    """

    def test_response_cache_enabled_on_server(self, mock_anthropic_server):
        """Server with enable_response_cache=True has a non-None response_cache."""
        mock_port = mock_anthropic_server.server_address[1]

        server = create_passthrough_server(
            host="127.0.0.1",
            port=0,
            api_key="test-key",
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=mock_port,
            use_https=False,
            enable_connection_pool=False,
            enable_response_cache=True,
            response_cache_ttl=60.0,
            response_cache_maxsize=128,
        )
        try:
            assert server.RequestHandlerClass.response_cache is not None
        finally:
            server.server_close()

    def test_response_cache_disabled_by_default(self, mock_anthropic_server):
        """Server without enable_response_cache has response_cache=None."""
        mock_port = mock_anthropic_server.server_address[1]

        server = create_passthrough_server(
            host="127.0.0.1",
            port=0,
            api_key="test-key",
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=mock_port,
            use_https=False,
            enable_connection_pool=False,
        )
        try:
            assert server.RequestHandlerClass.response_cache is None
        finally:
            server.server_close()

    def test_cached_get_returns_x_cache_hit_header(self, mock_anthropic_server):
        """Second GET to a cached endpoint includes X-Cache: HIT header."""
        mock_port = mock_anthropic_server.server_address[1]

        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            timeout=10,
            api_key="test-key",
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=mock_port,
            use_https=False,
            enable_connection_pool=False,
            enable_response_cache=True,
            response_cache_ttl=60.0,
            response_cache_maxsize=128,
        )
        port = server.server_address[1]

        try:
            # First request — should be a cache miss (no X-Cache header or MISS)
            resp1 = _gateway_request(port, path="/v1/models", method="GET")
            assert resp1.status == 200
            x_cache_1 = resp1.headers.get("X-Cache", "")
            assert x_cache_1 != "HIT", "First request should not be a cache hit"

            # Second request — should be served from cache
            resp2 = _gateway_request(port, path="/v1/models", method="GET")
            assert resp2.status == 200
            x_cache_2 = resp2.headers.get("X-Cache", "")
            assert x_cache_2 == "HIT", (
                f"Second request should be a cache hit but got X-Cache={x_cache_2!r}"
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_cached_response_body_matches_original(self, mock_anthropic_server):
        """Response body served from cache matches the original response."""
        mock_port = mock_anthropic_server.server_address[1]

        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            timeout=10,
            api_key="test-key",
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=mock_port,
            use_https=False,
            enable_connection_pool=False,
            enable_response_cache=True,
            response_cache_ttl=60.0,
            response_cache_maxsize=128,
        )
        port = server.server_address[1]

        try:
            resp1 = _gateway_request(port, path="/v1/models", method="GET")
            body1 = json.loads(resp1.read().decode())

            resp2 = _gateway_request(port, path="/v1/models", method="GET")
            body2 = json.loads(resp2.read().decode())

            assert body1 == body2
            assert "data" in body2
        finally:
            server.shutdown()
            server.server_close()

    def test_post_requests_not_cached(self, mock_anthropic_server):
        """POST requests are never served from cache."""
        mock_port = mock_anthropic_server.server_address[1]

        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            timeout=10,
            api_key="test-key",
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=mock_port,
            use_https=False,
            enable_connection_pool=False,
            enable_response_cache=True,
            response_cache_ttl=60.0,
            response_cache_maxsize=128,
        )
        port = server.server_address[1]

        try:
            resp1 = _make_messages_request(port)
            resp2 = _make_messages_request(port)

            # Neither response should carry an X-Cache: HIT header
            assert resp1.headers.get("X-Cache") != "HIT"
            assert resp2.headers.get("X-Cache") != "HIT"
        finally:
            server.shutdown()
            server.server_close()

    def test_cache_stats_reflected_in_status_endpoint(self, mock_anthropic_server):
        """Status endpoint reflects cache hit/miss counts."""
        mock_port = mock_anthropic_server.server_address[1]

        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            timeout=10,
            api_key="test-key",
            anthropic_api_host="127.0.0.1",
            anthropic_api_port=mock_port,
            use_https=False,
            enable_connection_pool=False,
            enable_response_cache=True,
            response_cache_ttl=60.0,
            response_cache_maxsize=128,
        )
        port = server.server_address[1]

        try:
            # Trigger a miss then a hit
            _gateway_request(port, path="/v1/models", method="GET")
            _gateway_request(port, path="/v1/models", method="GET")

            resp = _gateway_request(port, path="/status", method="GET")
            data = json.loads(resp.read().decode())

            assert "cache" in data
            cache_info = data["cache"]
            assert cache_info.get("enabled") is True
            # At least one hit and one miss should have occurred
            assert cache_info.get("hits", 0) >= 1
            assert cache_info.get("misses", 0) >= 1
        finally:
            server.shutdown()
            server.server_close()


# ------------------------------------------------------------------ #
# Tests: Constants and module structure
# ------------------------------------------------------------------ #


class TestModuleConstants:
    """Tests for module-level constants and configuration."""

    def test_default_api_host(self):
        """Default API host is api.anthropic.com."""
        assert ANTHROPIC_API_HOST == "api.anthropic.com"

    def test_default_api_version(self):
        """Default API version is 2023-06-01."""
        assert ANTHROPIC_API_VERSION == "2023-06-01"

    def test_allowed_paths_not_empty(self):
        """ALLOWED_PATHS is a non-empty set."""
        assert len(ALLOWED_PATHS) >= 3

    def test_handler_default_timeout(self):
        """Handler default timeout is 300 seconds."""
        assert AnthropicPassthroughHandler.upstream_timeout == 300

    def test_default_max_request_size(self):
        """Default max request size is 10 MB."""
        assert DEFAULT_MAX_REQUEST_SIZE == 10 * 1024 * 1024

    def test_handler_default_passthrough_mode(self):
        """Handler default passthrough mode is False."""
        assert AnthropicPassthroughHandler.passthrough_mode is False

    def test_gateway_internal_paths(self):
        """Gateway internal paths are correctly defined."""
        assert "/health" in GATEWAY_INTERNAL_PATHS
        assert "/status" in GATEWAY_INTERNAL_PATHS
