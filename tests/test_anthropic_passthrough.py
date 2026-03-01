"""Tests for the Anthropic API pass-through server."""

import http.server
import json
import os
import threading
import urllib.error
import urllib.request

import pytest
from click.testing import CliRunner

from src.anthropic_passthrough import (
    ALLOWED_PATHS,
    ANTHROPIC_API_HOST,
    ANTHROPIC_API_VERSION,
    AnthropicPassthroughHandler,
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


# ------------------------------------------------------------------ #
# Tests: CLI integration
# ------------------------------------------------------------------ #


class TestGatewayCLI:
    """Tests for the gateway CLI command."""

    def test_gateway_help(self, runner):
        """The gateway command help text is correct."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert result.exit_code == 0
        assert "gateway" in result.output.lower()
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--timeout" in result.output
        assert "--api-key" in result.output
        assert "--anthropic-version" in result.output
        assert "--verbose" in result.output
        assert "--config" in result.output

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


# ------------------------------------------------------------------ #
# Tests: Config-driven mode (helper unit tests)
# ------------------------------------------------------------------ #


class TestParseApiBase:
    """Unit tests for _parse_api_base."""

    def _make_handler(self):
        """Create a bare AnthropicPassthroughHandler for unit testing."""
        handler = AnthropicPassthroughHandler.__new__(AnthropicPassthroughHandler)
        return handler

    def test_https_url(self):
        """Parses an HTTPS URL correctly."""
        h = self._make_handler()
        result = h._parse_api_base("https://openrouter.ai/api/v1")
        assert result is not None
        host, port, use_https, base_path = result
        assert host == "openrouter.ai"
        assert port == 443
        assert use_https is True
        assert base_path == "/api/v1"

    def test_http_url(self):
        """Parses an HTTP URL and sets use_https=False."""
        h = self._make_handler()
        result = h._parse_api_base("http://127.0.0.1:9000/v1")
        assert result is not None
        host, port, use_https, base_path = result
        assert host == "127.0.0.1"
        assert port == 9000
        assert use_https is False
        assert base_path == "/v1"

    def test_url_trailing_slash_stripped(self):
        """Trailing slash in path is stripped."""
        h = self._make_handler()
        result = h._parse_api_base("https://example.com/api/v1/")
        assert result is not None
        assert result[3] == "/api/v1"

    def test_invalid_url_returns_none(self):
        """Invalid URL returns None."""
        h = self._make_handler()
        assert h._parse_api_base("not-a-url") is None

    def test_url_without_path(self):
        """URL with no path component returns empty base_path."""
        h = self._make_handler()
        result = h._parse_api_base("https://api.example.com")
        assert result is not None
        assert result[3] == ""


class TestComputeForwardedPath:
    """Unit tests for _compute_forwarded_path."""

    def _make_handler(self):
        handler = AnthropicPassthroughHandler.__new__(AnthropicPassthroughHandler)
        return handler

    def test_messages_with_api_v1_base(self):
        """/v1/messages → /api/v1/messages when base_path=/api/v1."""
        h = self._make_handler()
        assert h._compute_forwarded_path("/v1/messages", "/api/v1") == "/api/v1/messages"

    def test_models_with_api_v1_base(self):
        """/v1/models → /api/v1/models when base_path=/api/v1."""
        h = self._make_handler()
        assert h._compute_forwarded_path("/v1/models", "/api/v1") == "/api/v1/models"

    def test_count_tokens_with_api_v1_base(self):
        """/v1/messages/count_tokens → /api/v1/messages/count_tokens."""
        h = self._make_handler()
        assert (
            h._compute_forwarded_path("/v1/messages/count_tokens", "/api/v1")
            == "/api/v1/messages/count_tokens"
        )

    def test_v1_base_path_is_idempotent(self):
        """/v1/messages with base_path=/v1 keeps the same logical path."""
        h = self._make_handler()
        # /v1 base_path + messages = /v1/messages
        assert h._compute_forwarded_path("/v1/messages", "/v1") == "/v1/messages"

    def test_query_string_preserved(self):
        """Query string parameters are preserved."""
        h = self._make_handler()
        result = h._compute_forwarded_path("/v1/models?limit=10", "/api/v1")
        assert result == "/api/v1/models?limit=10"


# ------------------------------------------------------------------ #
# Tests: Config-driven /v1/models endpoint
# ------------------------------------------------------------------ #


class _MockProviderHandler(http.server.BaseHTTPRequestHandler):
    """Mock OpenRouter-style provider for config-driven tests.

    Expects:
    - Authorization: Bearer <token>
    - Requests to /api/v1/messages (not /v1/messages)
    """

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode() if length else ""
        auth = self.headers.get("Authorization", "")

        if not auth.startswith("Bearer "):
            resp = json.dumps(
                {"type": "error", "error": {"type": "auth_error", "message": "Missing bearer token"}}
            ).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        # Record which path was called (tests can inspect via class attribute)
        _MockProviderHandler.last_path = self.path
        _MockProviderHandler.last_auth = auth

        resp_data = {
            "id": "msg_provider_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Provider response"}],
            "model": "test-model",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        resp = json.dumps(resp_data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


_MockProviderHandler.last_path = None
_MockProviderHandler.last_auth = None


def _build_openrouter_config(api_base: str) -> "GatewayConfig":
    """Build a minimal GatewayConfig that mimics an OpenRouter setup."""
    from src.models import AuthType, GatewayConfig, ModelConfig, ProviderConfig

    model_a = ModelConfig(
        name="arcee-ai/trinity-large-preview:free",
        display_name="Arcee AI: Trinity Large Preview (free)",
        max_tokens=8192,
        supports_streaming=True,
        supports_tools=True,
    )
    model_b = ModelConfig(
        name="google/gemma-3-27b-it:free",
        display_name="Google: Gemma 3 27B (free)",
        max_tokens=8192,
        supports_streaming=True,
        supports_tools=True,
    )
    provider = ProviderConfig(
        name="openrouter",
        display_name="OpenRouter",
        api_base=api_base,
        api_key_env_var="OPENROUTER_API_KEY",
        auth_type=AuthType.BEARER_TOKEN,
        models={"arcee-ai/trinity-large-preview:free": model_a, "google/gemma-3-27b-it:free": model_b},
    )
    return GatewayConfig(default_provider="openrouter", providers={"openrouter": provider})


# Import GatewayConfig for type hint usage in helpers above
from src.models import GatewayConfig  # noqa: E402


class TestConfigDrivenModels:
    """Tests for /v1/models served from gateway config."""

    @pytest.fixture(scope="class")
    def config_gateway(self):
        """Gateway with a config that has two OpenRouter models."""
        config = _build_openrouter_config("http://127.0.0.1:1")  # dummy URL, models are local
        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            gateway_config=config,
        )
        yield server
        server.shutdown()
        server.server_close()

    def test_models_returns_200(self, config_gateway):
        """GET /v1/models returns HTTP 200."""
        port = config_gateway.server_address[1]
        resp = _gateway_request(port, path="/v1/models", method="GET")
        assert resp.status == 200

    def test_models_returns_json(self, config_gateway):
        """GET /v1/models returns application/json."""
        port = config_gateway.server_address[1]
        resp = _gateway_request(port, path="/v1/models", method="GET")
        assert "application/json" in resp.headers.get("Content-Type", "")

    def test_models_contains_config_models(self, config_gateway):
        """GET /v1/models lists models defined in the gateway config."""
        port = config_gateway.server_address[1]
        resp = _gateway_request(port, path="/v1/models", method="GET")
        data = json.loads(resp.read().decode())

        assert "data" in data
        ids = {m["id"] for m in data["data"]}
        assert "arcee-ai/trinity-large-preview:free" in ids
        assert "google/gemma-3-27b-it:free" in ids

    def test_models_response_shape(self, config_gateway):
        """Each model entry has required fields."""
        port = config_gateway.server_address[1]
        resp = _gateway_request(port, path="/v1/models", method="GET")
        data = json.loads(resp.read().decode())

        for model in data["data"]:
            assert "id" in model
            assert "type" in model
            assert model["type"] == "model"
            assert "display_name" in model

    def test_models_has_correct_count(self, config_gateway):
        """Exactly two models are returned (matching the config)."""
        port = config_gateway.server_address[1]
        resp = _gateway_request(port, path="/v1/models", method="GET")
        data = json.loads(resp.read().decode())
        assert len(data["data"]) == 2

    def test_models_cors_header(self, config_gateway):
        """CORS header is present on /v1/models response."""
        port = config_gateway.server_address[1]
        resp = _gateway_request(port, path="/v1/models", method="GET")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_models_no_api_key_needed(self):
        """Config-driven /v1/models works without any API key configured."""
        config = _build_openrouter_config("http://127.0.0.1:1")
        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            api_key=None,
            gateway_config=config,
        )
        port = server.server_address[1]
        original_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            resp = _gateway_request(port, path="/v1/models", method="GET")
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert len(data["data"]) == 2
        finally:
            if original_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_key
            server.shutdown()
            server.server_close()


class TestConfigDrivenMessagesRouting:
    """Tests for /v1/messages routed to a configured provider."""

    @pytest.fixture(scope="class")
    def mock_provider_server(self):
        """Start a mock provider (OpenRouter-style) server."""
        server = ThreadedGatewayServer(("127.0.0.1", 0), _MockProviderHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield server
        server.shutdown()
        server.server_close()

    @pytest.fixture(scope="class")
    def config_gateway_with_provider(self, mock_provider_server):
        """Gateway configured to route messages to the mock provider."""
        mock_host, mock_port = mock_provider_server.server_address
        api_base = f"http://{mock_host}:{mock_port}/api/v1"
        config = _build_openrouter_config(api_base)
        os.environ["OPENROUTER_API_KEY"] = "test-openrouter-key"

        server, thread = run_passthrough_in_thread(
            host="127.0.0.1",
            port=0,
            gateway_config=config,
        )
        yield server
        server.shutdown()
        server.server_close()
        os.environ.pop("OPENROUTER_API_KEY", None)

    def test_messages_forwarded_to_provider(self, config_gateway_with_provider):
        """POST /v1/messages is forwarded to the configured provider."""
        port = config_gateway_with_provider.server_address[1]
        resp = _make_messages_request(port, model="arcee-ai/trinity-large-preview:free")
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["type"] == "message"

    def test_bearer_auth_sent_to_provider(self, config_gateway_with_provider):
        """Bearer token authentication is used for the provider."""
        _MockProviderHandler.last_auth = None
        port = config_gateway_with_provider.server_address[1]
        _make_messages_request(port, model="arcee-ai/trinity-large-preview:free")
        assert _MockProviderHandler.last_auth == "Bearer test-openrouter-key"

    def test_path_transformed_for_provider(self, config_gateway_with_provider):
        """Request path is transformed from /v1/messages to /api/v1/messages."""
        _MockProviderHandler.last_path = None
        port = config_gateway_with_provider.server_address[1]
        _make_messages_request(port, model="arcee-ai/trinity-large-preview:free")
        assert _MockProviderHandler.last_path == "/api/v1/messages"

    def test_create_passthrough_server_accepts_gateway_config(self):
        """create_passthrough_server correctly stores gateway_config on handler."""
        config = _build_openrouter_config("https://openrouter.ai/api/v1")
        server = create_passthrough_server(
            host="127.0.0.1", port=0, gateway_config=config
        )
        assert server.RequestHandlerClass.gateway_config is config
        server.server_close()
