"""Anthropic API direct pass-through server.

Provides a reverse proxy that accepts requests in the Anthropic Messages API
format on a local endpoint and forwards them directly to the Anthropic API.
Supports both streaming (SSE) and non-streaming responses.

Key features:
- Transparent pass-through of all Anthropic API endpoints
- Automatic API key injection from ANTHROPIC_API_KEY environment variable
- Full streaming (Server-Sent Events) support
- Configurable retry with exponential backoff for transient failures
- Configurable timeout and logging
- Request/response logging for debugging
"""

import http.client
import http.server
import json
import logging
import os
import socket
import socketserver
import ssl
import threading
import time
from typing import Any, Optional

from src.errors import (
    AuthenticationError,
    BadGatewayError,
    GatewayError,
    GatewayTimeoutError,
    NetworkError,
    ProviderError,
    RateLimitError,
    SSLError as GatewaySSLError,
    TimeoutError_,
    is_retryable_status,
)
from src.logging_config import get_logger, log_request
from src.retry import BackoffStrategy, RetryConfig, retry_call

logger = get_logger("anthropic_passthrough")

# Default Anthropic API settings
ANTHROPIC_API_HOST = "api.anthropic.com"
ANTHROPIC_API_PORT = 443
ANTHROPIC_API_VERSION = "2023-06-01"

# Paths that are proxied to the Anthropic API
ALLOWED_PATHS = {
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/models",
}

# Headers that should NOT be forwarded from client to upstream
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "proxy-connection",
}

# Headers managed by the gateway (not forwarded from client)
MANAGED_HEADERS = {
    "host",
    "x-api-key",
    "authorization",
    "content-length",
}


class AnthropicPassthroughHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler that forwards requests to the Anthropic API.

    Accepts requests on local Anthropic API endpoints and proxies them
    to api.anthropic.com with proper authentication and headers.
    Supports both streaming (SSE) and non-streaming responses.

    Non-streaming requests are retried on transient failures using
    exponential backoff.  Streaming requests are NOT retried because
    partial SSE data may have already been sent to the client.
    """

    # Configurable class-level settings
    upstream_timeout: int = 300
    api_key: Optional[str] = None
    anthropic_api_host: str = ANTHROPIC_API_HOST
    anthropic_api_port: int = ANTHROPIC_API_PORT
    anthropic_version: str = ANTHROPIC_API_VERSION
    use_https: bool = True  # Set to False for testing with mock HTTP servers

    # Retry settings
    max_retries: int = 2
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0

    def log_message(self, format: str, *args) -> None:
        """Route access logs through the logging module."""
        logger.info("anthropic-passthrough: %s", format % args)

    # ------------------------------------------------------------------ #
    # HTTP method handlers
    # ------------------------------------------------------------------ #

    def do_GET(self):
        """Handle GET requests (e.g., /v1/models)."""
        self._handle_request()

    def do_POST(self):
        """Handle POST requests (e.g., /v1/messages)."""
        self._handle_request()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, x-api-key, anthropic-version, anthropic-beta",
        )
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    # ------------------------------------------------------------------ #
    # Core pass-through logic
    # ------------------------------------------------------------------ #

    def _handle_request(self) -> None:
        """Forward the incoming request to the Anthropic API."""
        request_start = time.monotonic()
        # Validate path
        path = self.path.split("?")[0]  # Strip query string for validation
        if not self._is_allowed_path(path):
            self._send_error(
                404,
                {
                    "type": "error",
                    "error": {
                        "type": "not_found",
                        "message": (
                            f"Path '{self.path}' is not a supported Anthropic API "
                            f"endpoint. Supported: {', '.join(sorted(ALLOWED_PATHS))}"
                        ),
                    },
                },
            )
            return

        # Get API key
        api_key = self._get_api_key()
        if not api_key:
            self._send_error(
                401,
                {
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": (
                            "No API key configured. Set the ANTHROPIC_API_KEY "
                            "environment variable or pass --api-key to the gateway."
                        ),
                    },
                },
            )
            return

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Determine if this is a streaming request
        is_streaming = False
        if body:
            try:
                request_data = json.loads(body)
                is_streaming = request_data.get("stream", False)
            except (json.JSONDecodeError, AttributeError):
                pass

        # Build upstream headers
        upstream_headers = self._build_upstream_headers(api_key)

        # Add Content-Type and Content-Length for body requests
        if body is not None:
            upstream_headers["Content-Type"] = (
                self.headers.get("Content-Type", "application/json")
            )
            upstream_headers["Content-Length"] = str(len(body))

        logger.info(
            "Forwarding %s %s to %s (streaming=%s)",
            self.command,
            self.path,
            self.anthropic_api_host,
            is_streaming,
        )

        if is_streaming:
            # Streaming requests are NOT retried — partial data may
            # have been sent to the client already.
            self._forward_streaming(body, upstream_headers, request_start)
        else:
            # Non-streaming requests use retry logic
            self._forward_with_retry(body, upstream_headers, request_start)

    # ------------------------------------------------------------------ #
    # Non-streaming with retry
    # ------------------------------------------------------------------ #

    def _forward_with_retry(
        self,
        body: Optional[bytes],
        upstream_headers: dict[str, str],
        request_start: float,
    ) -> None:
        """Forward a non-streaming request with retry on transient failure."""
        total_attempts = self.max_retries + 1
        retry_config = RetryConfig(
            max_attempts=total_attempts,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
            jitter=True,
            retry_on_status={429, 500, 502, 503, 504, 529},
        )

        def _do_request() -> tuple[int, str, list, bytes]:
            """Perform a single upstream request.

            Returns:
                (status, reason, headers, body) from the upstream.

            Raises:
                GatewayError subclass on connection/protocol failures.
                ProviderError on retryable HTTP status codes.
            """
            try:
                if self.use_https:
                    context = ssl.create_default_context()
                    conn = http.client.HTTPSConnection(
                        self.anthropic_api_host,
                        self.anthropic_api_port,
                        timeout=self.upstream_timeout,
                        context=context,
                    )
                else:
                    conn = http.client.HTTPConnection(
                        self.anthropic_api_host,
                        self.anthropic_api_port,
                        timeout=self.upstream_timeout,
                    )

                conn.request(
                    self.command,
                    self.path,
                    body=body,
                    headers=upstream_headers,
                )
                upstream_resp = conn.getresponse()

                resp_status = upstream_resp.status
                resp_reason = upstream_resp.reason
                resp_headers = upstream_resp.getheaders()
                resp_body = upstream_resp.read()
                conn.close()

                # Raise on retryable status codes so retry_call retries
                if is_retryable_status(resp_status):
                    # Extract Retry-After if present
                    retry_after = None
                    for hdr_name, hdr_val in resp_headers:
                        if hdr_name.lower() == "retry-after":
                            try:
                                retry_after = float(hdr_val)
                            except ValueError:
                                pass
                            break

                    if resp_status == 429:
                        raise RateLimitError(
                            f"Anthropic API rate limit (HTTP {resp_status})",
                            retry_after=retry_after,
                            provider="anthropic",
                            status_code=resp_status,
                        )

                    raise ProviderError(
                        f"Anthropic API returned {resp_status} {resp_reason}",
                        status_code=resp_status,
                        provider="anthropic",
                        response_body=resp_body.decode("utf-8", errors="replace"),
                    )

                return resp_status, resp_reason, resp_headers, resp_body

            except (
                RateLimitError,
                ProviderError,
                NetworkError,
                TimeoutError_,
                GatewaySSLError,
                BadGatewayError,
            ):
                raise  # Already structured — propagate for retry
            except ssl.SSLError as exc:
                raise GatewaySSLError(
                    f"SSL error connecting to Anthropic API: {exc}",
                    host=self.anthropic_api_host,
                    port=self.anthropic_api_port,
                    cause=exc,
                )
            except (socket.timeout, TimeoutError) as exc:
                raise GatewayTimeoutError(
                    self.upstream_timeout,
                    host=self.anthropic_api_host,
                    port=self.anthropic_api_port,
                    cause=exc,
                )
            except ConnectionRefusedError as exc:
                raise NetworkError(
                    f"Connection refused by {self.anthropic_api_host}:{self.anthropic_api_port}",
                    host=self.anthropic_api_host,
                    port=self.anthropic_api_port,
                    cause=exc,
                )
            except (ConnectionError, OSError) as exc:
                raise NetworkError(
                    f"Connection error: {exc}",
                    host=self.anthropic_api_host,
                    port=self.anthropic_api_port,
                    cause=exc,
                )
            except Exception as exc:
                raise BadGatewayError(
                    f"Error connecting to Anthropic API: {exc}",
                    host=self.anthropic_api_host,
                    port=self.anthropic_api_port,
                    cause=exc,
                )

        try:
            status, reason, resp_headers, resp_body = retry_call(
                _do_request, config=retry_config
            )

            # Relay status
            self.send_response(status, reason)

            # Relay response headers
            for key, val in resp_headers:
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, val)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Relay body
            if resp_body:
                self.wfile.write(resp_body)

            duration_ms = (time.monotonic() - request_start) * 1000.0
            client_addr = self.client_address[0] if self.client_address else None
            log_request(
                method=self.command,
                path=self.path,
                status_code=status,
                duration_ms=duration_ms,
                client_ip=client_addr,
                extra={
                    "upstream": self.anthropic_api_host,
                    "streaming": False,
                },
                logger=logger,
            )

        except GatewayError as exc:
            self._handle_gateway_error(exc, request_start)
        except Exception as exc:
            logger.error(
                "Error forwarding %s %s: %s", self.command, self.path, exc
            )
            self._send_error(
                502,
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"Error connecting to Anthropic API: {exc}",
                    },
                },
            )

    # ------------------------------------------------------------------ #
    # Streaming (no retry — partial data prevents safe retry)
    # ------------------------------------------------------------------ #

    def _forward_streaming(
        self,
        body: Optional[bytes],
        upstream_headers: dict[str, str],
        request_start: float,
    ) -> None:
        """Forward a streaming request without retries."""
        try:
            if self.use_https:
                context = ssl.create_default_context()
                conn = http.client.HTTPSConnection(
                    self.anthropic_api_host,
                    self.anthropic_api_port,
                    timeout=self.upstream_timeout,
                    context=context,
                )
            else:
                conn = http.client.HTTPConnection(
                    self.anthropic_api_host,
                    self.anthropic_api_port,
                    timeout=self.upstream_timeout,
                )

            conn.request(
                self.command,
                self.path,
                body=body,
                headers=upstream_headers,
            )

            upstream_resp = conn.getresponse()
            self._relay_streaming_response(upstream_resp)
            conn.close()

            duration_ms = (time.monotonic() - request_start) * 1000.0
            client_addr = self.client_address[0] if self.client_address else None
            log_request(
                method=self.command,
                path=self.path,
                status_code=upstream_resp.status,
                duration_ms=duration_ms,
                client_ip=client_addr,
                extra={
                    "upstream": self.anthropic_api_host,
                    "streaming": True,
                },
                logger=logger,
            )

        except ssl.SSLError as exc:
            logger.error("SSL error connecting to Anthropic API: %s", exc)
            self._send_error(
                502,
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"SSL error connecting to Anthropic API: {exc}",
                    },
                },
            )
        except TimeoutError as exc:
            logger.error("Timeout connecting to Anthropic API: %s", exc)
            self._send_error(
                504,
                {
                    "type": "error",
                    "error": {
                        "type": "timeout_error",
                        "message": (
                            f"Request to Anthropic API timed out after "
                            f"{self.upstream_timeout}s"
                        ),
                    },
                },
            )
        except Exception as exc:
            logger.error(
                "Error forwarding %s %s: %s", self.command, self.path, exc
            )
            self._send_error(
                502,
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"Error connecting to Anthropic API: {exc}",
                    },
                },
            )

    # ------------------------------------------------------------------ #
    # Response relay helpers
    # ------------------------------------------------------------------ #

    def _relay_standard_response(
        self, upstream_resp: http.client.HTTPResponse
    ) -> None:
        """Relay a non-streaming response from the Anthropic API."""
        # Read the full response body
        response_body = upstream_resp.read()

        # Send status
        self.send_response(upstream_resp.status, upstream_resp.reason)

        # Forward response headers
        for key, val in upstream_resp.getheaders():
            if key.lower() not in HOP_BY_HOP_HEADERS:
                self.send_header(key, val)

        # Add CORS header
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Write body
        if response_body:
            self.wfile.write(response_body)

    def _relay_streaming_response(
        self, upstream_resp: http.client.HTTPResponse
    ) -> None:
        """Relay a streaming (SSE) response from the Anthropic API.

        Reads chunks from the upstream connection and forwards them
        to the client in real-time for Server-Sent Events streaming.
        """
        # Send status
        self.send_response(upstream_resp.status, upstream_resp.reason)

        # Forward response headers, adjusting for streaming
        for key, val in upstream_resp.getheaders():
            lower_key = key.lower()
            if lower_key not in HOP_BY_HOP_HEADERS:
                # Don't forward content-length for streaming (chunked)
                if lower_key == "content-length":
                    continue
                self.send_header(key, val)

        # Add CORS and streaming headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        # Stream the response body in chunks
        try:
            while True:
                chunk = upstream_resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Client disconnected during streaming response")

    # ------------------------------------------------------------------ #
    # Error handling helpers
    # ------------------------------------------------------------------ #

    def _handle_gateway_error(
        self, exc: GatewayError, request_start: float
    ) -> None:
        """Convert a GatewayError into an appropriate HTTP error response."""
        duration_ms = (time.monotonic() - request_start) * 1000.0

        # Determine HTTP status from the error
        status_code = exc.context.status_code or 502
        error_type = exc._error_type()

        # Build context-aware message
        message = str(exc)
        if exc.context.attempt and exc.context.max_attempts:
            message += (
                f" (after {exc.context.attempt}/{exc.context.max_attempts} "
                f"attempts, {exc.context.elapsed_seconds:.1f}s total)"
            )

        # For RetryExhaustedError, use the last error's status code
        from src.errors import RetryExhaustedError
        if isinstance(exc, RetryExhaustedError) and exc.last_error:
            if isinstance(exc.last_error, GatewayError):
                status_code = exc.last_error.context.status_code or 502
                error_type = exc.last_error._error_type()
            message = (
                f"All {exc.attempts} retry attempts exhausted "
                f"after {exc.total_elapsed:.1f}s. "
                f"Last error: {exc.last_error}"
            )

        logger.error(
            "Gateway error for %s %s: %s (%.1fms)",
            self.command, self.path, exc, duration_ms,
        )

        self._send_error(
            status_code,
            {
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": message,
                },
            },
        )

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _is_allowed_path(self, path: str) -> bool:
        """Check if the request path is an allowed Anthropic API endpoint.

        Args:
            path: The request path (without query string).

        Returns:
            True if the path is allowed.
        """
        # Exact match
        if path in ALLOWED_PATHS:
            return True
        # Allow paths under allowed prefixes (e.g., /v1/models/claude-3-opus)
        for allowed in ALLOWED_PATHS:
            if path.startswith(allowed + "/"):
                return True
        return False

    def _get_api_key(self) -> Optional[str]:
        """Get the Anthropic API key.

        Checks in order:
        1. Class-level api_key attribute (set via --api-key flag)
        2. x-api-key header from the client request
        3. ANTHROPIC_API_KEY environment variable

        Returns:
            The API key string, or None if not found.
        """
        # 1. Class-level override
        if self.api_key:
            return self.api_key

        # 2. Client-provided header
        client_key = self.headers.get("x-api-key")
        if client_key:
            return client_key

        # 3. Environment variable
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        if env_key:
            return env_key

        return None

    def _build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Build the headers to send to the Anthropic API.

        Args:
            api_key: The Anthropic API key.

        Returns:
            Dictionary of headers for the upstream request.
        """
        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.anthropic_version,
            "Host": self.anthropic_api_host,
            "Accept": "application/json",
        }

        # Forward select client headers
        for key, val in self.headers.items():
            lower_key = key.lower()
            # Skip hop-by-hop and managed headers
            if lower_key in HOP_BY_HOP_HEADERS:
                continue
            if lower_key in MANAGED_HEADERS:
                continue
            # Forward anthropic-specific headers
            if lower_key.startswith("anthropic-"):
                headers[key] = val
            # Forward user-agent
            elif lower_key == "user-agent":
                headers[key] = val
            # Forward accept headers
            elif lower_key.startswith("accept"):
                headers[key] = val

        return headers

    def _send_error(self, code: int, error_body: dict) -> None:
        """Send a JSON error response to the client.

        Args:
            code: HTTP status code.
            error_body: Error response body (Anthropic error format).
        """
        body = json.dumps(error_body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


class ThreadedGatewayServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """HTTP server that handles each request in a separate thread."""

    allow_reuse_address = True
    daemon_threads = True


def create_passthrough_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    timeout: int = 300,
    api_key: Optional[str] = None,
    anthropic_api_host: str = ANTHROPIC_API_HOST,
    anthropic_api_port: int = ANTHROPIC_API_PORT,
    anthropic_version: str = ANTHROPIC_API_VERSION,
    use_https: bool = True,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 30.0,
) -> ThreadedGatewayServer:
    """Create (but do not start) an Anthropic pass-through server.

    Args:
        host: Address to bind to (default: 127.0.0.1).
        port: Port to listen on (default: 8080).
        timeout: Upstream connection timeout in seconds (default: 300).
        api_key: Optional API key override. If not set, uses
            ANTHROPIC_API_KEY environment variable.
        anthropic_api_host: Anthropic API hostname to connect to.
        anthropic_api_port: Anthropic API port to connect to (default: 443).
        anthropic_version: Anthropic API version header value.
        use_https: Whether to use HTTPS for upstream connections
            (default: True). Set to False for testing with mock servers.
        max_retries: Number of retries for failed non-streaming requests
            (0 = no retries).
        retry_base_delay: Base delay in seconds for retry backoff.
        retry_max_delay: Maximum delay cap in seconds.

    Returns:
        A configured ThreadedGatewayServer instance.
    """
    handler = AnthropicPassthroughHandler
    handler.upstream_timeout = timeout
    handler.api_key = api_key
    handler.anthropic_api_host = anthropic_api_host
    handler.anthropic_api_port = anthropic_api_port
    handler.anthropic_version = anthropic_version
    handler.use_https = use_https
    handler.max_retries = max_retries
    handler.retry_base_delay = retry_base_delay
    handler.retry_max_delay = retry_max_delay

    server = ThreadedGatewayServer((host, port), handler)
    return server


def run_passthrough(
    host: str = "127.0.0.1",
    port: int = 8080,
    timeout: int = 300,
    api_key: Optional[str] = None,
    anthropic_api_host: str = ANTHROPIC_API_HOST,
    anthropic_api_port: int = ANTHROPIC_API_PORT,
    anthropic_version: str = ANTHROPIC_API_VERSION,
    use_https: bool = True,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 30.0,
) -> None:
    """Start the Anthropic pass-through server and serve requests forever.

    Args:
        host: Address to bind to (default: 127.0.0.1).
        port: Port to listen on (default: 8080).
        timeout: Upstream connection timeout in seconds (default: 300).
        api_key: Optional API key override.
        anthropic_api_host: Anthropic API hostname to connect to.
        anthropic_api_port: Anthropic API port to connect to.
        anthropic_version: Anthropic API version header value.
        use_https: Whether to use HTTPS for upstream connections.
        max_retries: Number of retries for non-streaming requests.
        retry_base_delay: Base delay in seconds for retry backoff.
        retry_max_delay: Maximum delay cap in seconds.
    """
    server = create_passthrough_server(
        host=host,
        port=port,
        timeout=timeout,
        api_key=api_key,
        anthropic_api_host=anthropic_api_host,
        anthropic_api_port=anthropic_api_port,
        anthropic_version=anthropic_version,
        use_https=use_https,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
    )
    logger.info(
        "Anthropic pass-through server listening on %s:%d "
        "(upstream: %s, version: %s, max_retries: %d)",
        host,
        port,
        anthropic_api_host,
        anthropic_version,
        max_retries,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Anthropic pass-through server")
    finally:
        server.server_close()


def run_passthrough_in_thread(
    host: str = "127.0.0.1",
    port: int = 8080,
    timeout: int = 300,
    api_key: Optional[str] = None,
    anthropic_api_host: str = ANTHROPIC_API_HOST,
    anthropic_api_port: int = ANTHROPIC_API_PORT,
    anthropic_version: str = ANTHROPIC_API_VERSION,
    use_https: bool = True,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 30.0,
) -> tuple[ThreadedGatewayServer, threading.Thread]:
    """Start the Anthropic pass-through server in a background thread.

    Useful for testing or embedding the gateway within another application.

    Args:
        host: Address to bind to.
        port: Port to listen on.
        timeout: Upstream connection timeout in seconds.
        api_key: Optional API key override.
        anthropic_api_host: Anthropic API hostname to connect to.
        anthropic_api_port: Anthropic API port to connect to.
        anthropic_version: Anthropic API version header value.
        use_https: Whether to use HTTPS for upstream connections.
        max_retries: Number of retries for non-streaming requests.
        retry_base_delay: Base delay in seconds for retry backoff.
        retry_max_delay: Maximum delay cap in seconds.

    Returns:
        A (server, thread) tuple. Call server.shutdown() to stop.
    """
    server = create_passthrough_server(
        host=host,
        port=port,
        timeout=timeout,
        api_key=api_key,
        anthropic_api_host=anthropic_api_host,
        anthropic_api_port=anthropic_api_port,
        anthropic_version=anthropic_version,
        use_https=use_https,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(
        "Anthropic pass-through server started in background on %s:%d",
        host,
        port,
    )
    return server, thread
