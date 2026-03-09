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
- Config-driven mode: serve /v1/models from config, route messages to any provider
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
from urllib.parse import urlparse

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

    # Config-driven mode: when set, /v1/models is served from config and
    # messages are routed to the configured provider instead of api.anthropic.com
    gateway_config: Optional[Any] = None  # GatewayConfig | None

    # Response caching (None = disabled)
    response_cache: Optional[Any] = None  # ResponseCache | None

    # Cache statistics counters (class-level, reset per server subclass)
    _cache_hits: int = 0
    _cache_misses: int = 0
    _cache_lock: threading.Lock = threading.Lock()

    def log_message(self, format: str, *args) -> None:
        """Route access logs through the logging module."""
        logger.info("anthropic-passthrough: %s", format % args)

    # ------------------------------------------------------------------ #
    # HTTP method handlers
    # ------------------------------------------------------------------ #

    def do_GET(self):
        """Handle GET requests (e.g., /v1/models, /status)."""
        path = self.path.split("?")[0]
        if path == "/status":
            self._serve_status()
            return
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
        """Forward the incoming request to the appropriate upstream."""
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

        # In config-driven mode, serve /v1/models locally from the config file.
        # This allows non-Anthropic providers (OpenRouter, etc.) to appear in
        # the Claude Code /model picker without forwarding to api.anthropic.com.
        if path == "/v1/models" and self.gateway_config is not None:
            self._serve_models_from_config()
            return

        # Read the request body early so we can inspect the ``model`` field
        # before deciding which upstream provider to route to.
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Extract model name and streaming flag from the body.
        is_streaming = False
        request_model: Optional[str] = None
        if body:
            try:
                request_data = json.loads(body)
                is_streaming = request_data.get("stream", False)
                request_model = request_data.get("model")
            except (json.JSONDecodeError, AttributeError):
                pass

        # Determine upstream connection params and auth headers.
        # When a gateway config is present, route by the model field so each
        # provider's models reach the correct upstream; fall back to the
        # default provider for unknown models or when no model is specified.
        upstream_host: str = self.anthropic_api_host
        upstream_port: int = self.anthropic_api_port
        upstream_https: bool = self.use_https
        upstream_path: str = self.path
        upstream_headers: dict[str, str]

        if self.gateway_config is not None:
            # Select provider by model name when available; fall back to default.
            if request_model:
                provider = self.gateway_config.find_provider_for_model(request_model)
            else:
                provider = self.gateway_config.get_provider()
            if provider and provider.api_base:
                parsed = self._parse_api_base(provider.api_base)
                if parsed is None:
                    self._send_error(
                        500,
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": (
                                    f"Invalid api_base '{provider.api_base}' "
                                    f"for provider '{provider.name}'"
                                ),
                            },
                        },
                    )
                    return
                upstream_host, upstream_port, upstream_https, base_path = parsed
                upstream_path = self._compute_forwarded_path(self.path, base_path)
                api_key = ""
                if provider.api_key_env_var:
                    api_key = os.environ.get(provider.api_key_env_var, "")
                upstream_headers = self._build_provider_upstream_headers(
                    provider, upstream_host, api_key
                )
            else:
                # Config present but no provider — fall back to Anthropic defaults
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
                upstream_headers = self._build_upstream_headers(api_key)
        else:
            # No config: default Anthropic pass-through
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
            upstream_headers = self._build_upstream_headers(api_key)

        # Add Content-Type and Content-Length for body requests
        if body is not None:
            upstream_headers["Content-Type"] = (
                self.headers.get("Content-Type", "application/json")
            )
            upstream_headers["Content-Length"] = str(len(body))

        logger.info(
            "Forwarding %s %s to %s%s (streaming=%s)",
            self.command,
            self.path,
            upstream_host,
            upstream_path,
            is_streaming,
        )

        if is_streaming:
            # Streaming requests are NOT retried — partial data may
            # have been sent to the client already.
            self._forward_streaming(
                body,
                upstream_headers,
                request_start,
                upstream_host,
                upstream_port,
                upstream_https,
                upstream_path,
            )
        else:
            # Non-streaming requests use retry logic
            self._forward_with_retry(
                body,
                upstream_headers,
                request_start,
                upstream_host,
                upstream_port,
                upstream_https,
                upstream_path,
            )

    # ------------------------------------------------------------------ #
    # Non-streaming with retry
    # ------------------------------------------------------------------ #

    def _forward_with_retry(
        self,
        body: Optional[bytes],
        upstream_headers: dict[str, str],
        request_start: float,
        upstream_host: Optional[str] = None,
        upstream_port: Optional[int] = None,
        upstream_https: Optional[bool] = None,
        upstream_path: Optional[str] = None,
    ) -> None:
        """Forward a non-streaming request with retry on transient failure."""
        # Resolve upstream connection params (use class defaults when not given)
        host = upstream_host if upstream_host is not None else self.anthropic_api_host
        port = upstream_port if upstream_port is not None else self.anthropic_api_port
        use_https = upstream_https if upstream_https is not None else self.use_https
        path = upstream_path if upstream_path is not None else self.path

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
                if use_https:
                    context = ssl.create_default_context()
                    conn = http.client.HTTPSConnection(
                        host,
                        port,
                        timeout=self.upstream_timeout,
                        context=context,
                    )
                else:
                    conn = http.client.HTTPConnection(
                        host,
                        port,
                        timeout=self.upstream_timeout,
                    )

                conn.request(
                    self.command,
                    path,
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
                            f"Upstream rate limit (HTTP {resp_status})",
                            retry_after=retry_after,
                            provider=host,
                            status_code=resp_status,
                        )

                    raise ProviderError(
                        f"Upstream returned {resp_status} {resp_reason}",
                        status_code=resp_status,
                        provider=host,
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
                    f"SSL error connecting to {host}: {exc}",
                    host=host,
                    port=port,
                    cause=exc,
                )
            except (socket.timeout, TimeoutError) as exc:
                raise GatewayTimeoutError(
                    self.upstream_timeout,
                    host=host,
                    port=port,
                    cause=exc,
                )
            except ConnectionRefusedError as exc:
                raise NetworkError(
                    f"Connection refused by {host}:{port}",
                    host=host,
                    port=port,
                    cause=exc,
                )
            except (ConnectionError, OSError) as exc:
                raise NetworkError(
                    f"Connection error: {exc}",
                    host=host,
                    port=port,
                    cause=exc,
                )
            except Exception as exc:
                raise BadGatewayError(
                    f"Error connecting to {host}: {exc}",
                    host=host,
                    port=port,
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
                    "upstream": host,
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
                        "message": f"Error connecting to upstream: {exc}",
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
        upstream_host: Optional[str] = None,
        upstream_port: Optional[int] = None,
        upstream_https: Optional[bool] = None,
        upstream_path: Optional[str] = None,
    ) -> None:
        """Forward a streaming request without retries."""
        # Resolve upstream connection params (use class defaults when not given)
        host = upstream_host if upstream_host is not None else self.anthropic_api_host
        port = upstream_port if upstream_port is not None else self.anthropic_api_port
        use_https = upstream_https if upstream_https is not None else self.use_https
        path = upstream_path if upstream_path is not None else self.path

        try:
            if use_https:
                context = ssl.create_default_context()
                conn = http.client.HTTPSConnection(
                    host,
                    port,
                    timeout=self.upstream_timeout,
                    context=context,
                )
            else:
                conn = http.client.HTTPConnection(
                    host,
                    port,
                    timeout=self.upstream_timeout,
                )

            conn.request(
                self.command,
                path,
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
                    "upstream": host,
                    "streaming": True,
                },
                logger=logger,
            )

        except ssl.SSLError as exc:
            logger.error("SSL error connecting to %s: %s", host, exc)
            self._send_error(
                502,
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"SSL error connecting to {host}: {exc}",
                    },
                },
            )
        except TimeoutError as exc:
            logger.error("Timeout connecting to %s: %s", host, exc)
            self._send_error(
                504,
                {
                    "type": "error",
                    "error": {
                        "type": "timeout_error",
                        "message": (
                            f"Request to {host} timed out after "
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
                        "message": f"Error connecting to upstream: {exc}",
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
    # Config-driven helpers
    # ------------------------------------------------------------------ #

    def _serve_models_from_config(self) -> None:
        """Serve GET /v1/models from the gateway config.

        Returns all models across all enabled providers in the Anthropic
        models list format so Claude Code can populate its /model picker.
        """
        models_data = []
        for provider in self.gateway_config.get_enabled_providers().values():
            for model in provider.models.values():
                models_data.append(
                    {
                        "type": "model",
                        "id": model.name,
                        "display_name": model.display_name,
                        "created_at": "2024-01-01T00:00:00Z",
                    }
                )

        response: dict[str, Any] = {
            "data": models_data,
            "has_more": False,
        }
        if models_data:
            response["first_id"] = models_data[0]["id"]
            response["last_id"] = models_data[-1]["id"]

        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
        logger.info("Served %d model(s) from gateway config", len(models_data))

    def _parse_api_base(
        self, api_base: str
    ) -> Optional[tuple[str, int, bool, str]]:
        """Parse a provider api_base URL into (host, port, use_https, base_path).

        Args:
            api_base: Full URL such as ``https://openrouter.ai/api/v1``.

        Returns:
            ``(host, port, use_https, base_path)`` tuple, or ``None`` if the
            URL cannot be parsed or has no hostname.
        """
        try:
            parsed = urlparse(api_base)
            host = parsed.hostname
            if not host:
                return None
            use_https = parsed.scheme == "https"
            port = parsed.port or (443 if use_https else 80)
            base_path = parsed.path.rstrip("/")
            return host, port, use_https, base_path
        except Exception:
            return None

    def _compute_forwarded_path(self, request_path: str, base_path: str) -> str:
        """Compute the upstream path from the incoming request path.

        Strips the ``/v1`` prefix from *request_path* and prepends
        *base_path* so that, for example::

            request_path="/v1/messages", base_path="/api/v1"
            → "/api/v1/messages"

        Args:
            request_path: The full incoming path including query string
                (e.g. ``/v1/messages?foo=bar``).
            base_path: The api_base path component (e.g. ``/api/v1``).

        Returns:
            The path string to send to the upstream server.
        """
        # Separate query string
        if "?" in request_path:
            path_part, query = request_path.split("?", 1)
            query = "?" + query
        else:
            path_part, query = request_path, ""

        # Strip the leading /v1 segment
        if path_part.startswith("/v1/"):
            endpoint = path_part[4:]  # e.g. "messages" or "models"
        elif path_part == "/v1":
            endpoint = ""
        else:
            endpoint = path_part.lstrip("/")

        if endpoint:
            return f"{base_path}/{endpoint}{query}"
        return f"{base_path}{query}"

    def _build_provider_upstream_headers(
        self, provider: Any, upstream_host: str, api_key: str
    ) -> dict[str, str]:
        """Build upstream headers for a configured provider.

        Applies the correct authentication scheme (bearer token or api-key)
        and includes any static headers defined in the provider config.

        Args:
            provider: A ``ProviderConfig`` instance.
            upstream_host: The upstream hostname (used for the Host header).
            api_key: The resolved API key string.

        Returns:
            Dictionary of HTTP headers to send upstream.
        """
        from src.models import AuthType

        headers: dict[str, str] = {
            "Host": upstream_host,
            "Accept": "application/json",
        }

        # Apply authentication based on provider auth_type
        if provider.auth_type == AuthType.BEARER_TOKEN:
            headers["Authorization"] = f"Bearer {api_key}"
        elif provider.auth_type == AuthType.API_KEY:
            headers["x-api-key"] = api_key
        # AuthType.NONE: no auth header added

        # Merge any static headers defined in the provider config
        headers.update(provider.headers)

        # Forward select client headers
        for key, val in self.headers.items():
            lower_key = key.lower()
            if lower_key in HOP_BY_HOP_HEADERS:
                continue
            if lower_key in MANAGED_HEADERS:
                continue
            # Forward user-agent and accept-* headers
            if lower_key == "user-agent" or lower_key.startswith("accept"):
                headers[key] = val

        return headers

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
    gateway_config: Optional[Any] = None,
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
        gateway_config: Optional GatewayConfig. When set, ``GET /v1/models``
            is served from the config and messages are routed to the
            default provider instead of api.anthropic.com.

    Returns:
        A configured ThreadedGatewayServer instance.
    """
    # Build a per-server subclass so that configuring one server does not
    # mutate the shared class-level defaults on AnthropicPassthroughHandler.
    handler: type[AnthropicPassthroughHandler] = type(
        "_GatewayHandler",
        (AnthropicPassthroughHandler,),
        {
            "upstream_timeout": timeout,
            "api_key": api_key,
            "anthropic_api_host": anthropic_api_host,
            "anthropic_api_port": anthropic_api_port,
            "anthropic_version": anthropic_version,
            "use_https": use_https,
            "max_retries": max_retries,
            "retry_base_delay": retry_base_delay,
            "retry_max_delay": retry_max_delay,
            "gateway_config": gateway_config,
        },
    )

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
    gateway_config: Optional[Any] = None,
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
        gateway_config: Optional GatewayConfig for config-driven mode.
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
        gateway_config=gateway_config,
    )
    upstream_label = (
        gateway_config.get_provider().api_base
        if gateway_config and gateway_config.get_provider()
        else anthropic_api_host
    )
    logger.info(
        "Gateway listening on %s:%d "
        "(upstream: %s, version: %s, max_retries: %d)",
        host,
        port,
        upstream_label,
        anthropic_version,
        max_retries,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down gateway server")
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
    gateway_config: Optional[Any] = None,
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
        gateway_config: Optional GatewayConfig for config-driven mode.

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
        gateway_config=gateway_config,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(
        "Gateway server started in background on %s:%d",
        host,
        port,
    )
    return server, thread
