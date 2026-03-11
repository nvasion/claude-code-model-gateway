"""Anthropic API direct pass-through server.

Provides a reverse proxy that accepts requests in the Anthropic Messages API
format on a local endpoint and forwards them directly to the Anthropic API.
Supports both streaming (SSE) and non-streaming responses.

Key features:
- Transparent pass-through of all Anthropic API endpoints
- Full pass-through mode: forwards any /v1/* request to the Anthropic API
- Automatic API key injection from ANTHROPIC_API_KEY environment variable
- Full streaming (Server-Sent Events) support
- Configurable retry with exponential backoff for transient failures
- Request correlation ID generation and propagation
- Health check endpoint (/health) for monitoring
- Status/metrics endpoint (/status) for observability
- Connection pooling for improved performance
- Request body size validation
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
import uuid
from typing import Any, Optional

from src.errors import (
    AuthenticationError,
    BadGatewayError,
    GatewayError,
    GatewayTimeoutError,
    NetworkError,
    ProviderError,
    RateLimitError,
    RetryExhaustedError,
    SSLError as GatewaySSLError,
    TimeoutError_,
    is_retryable_status,
)
from src.error_handling import get_error_tracker
from src.logging_config import get_logger, log_request, set_correlation_id
from src.retry import BackoffStrategy, RetryConfig, retry_call

# Cacheable HTTP methods for response caching
_CACHEABLE_METHODS = {"GET", "HEAD"}

# Default TTL for cached responses (seconds)
_RESPONSE_CACHE_TTL = 300  # 5 minutes for model listings etc.

logger = get_logger("anthropic_passthrough")

# Default Anthropic API settings
ANTHROPIC_API_HOST = "api.anthropic.com"
ANTHROPIC_API_PORT = 443
ANTHROPIC_API_VERSION = "2023-06-01"

# Maximum request body size (default 10 MB)
DEFAULT_MAX_REQUEST_SIZE = 10 * 1024 * 1024

# Paths that are proxied to the Anthropic API (strict mode)
ALLOWED_PATHS = {
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/messages/batches",
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

# Gateway internal paths (not forwarded to Anthropic)
GATEWAY_INTERNAL_PATHS = {
    "/health",
    "/status",
}


# --------------------------------------------------------------------------- #
# Connection pool
# --------------------------------------------------------------------------- #


class ConnectionPool:
    """Thread-safe pool of reusable HTTP(S) connections.

    Maintains a pool of persistent connections to the upstream Anthropic API
    to avoid the overhead of TLS handshake and TCP connection setup on
    every request.

    Args:
        host: Upstream hostname.
        port: Upstream port.
        use_https: Whether to use HTTPS.
        timeout: Connection timeout in seconds.
        max_connections: Maximum number of pooled connections.
    """

    def __init__(
        self,
        host: str,
        port: int,
        use_https: bool = True,
        timeout: int = 300,
        max_connections: int = 10,
    ) -> None:
        self.host = host
        self.port = port
        self.use_https = use_https
        self.timeout = timeout
        self.max_connections = max_connections
        self._pool: list[http.client.HTTPConnection] = []
        self._lock = threading.Lock()
        self._created_count = 0

    def get_connection(self) -> http.client.HTTPConnection:
        """Get a connection from the pool or create a new one.

        Returns:
            An HTTP(S) connection to the upstream server.
        """
        with self._lock:
            while self._pool:
                conn = self._pool.pop()
                # Test if the connection is still alive
                try:
                    # A quick check — if the socket is closed, this will fail
                    if conn.sock is not None:
                        return conn
                except Exception:
                    pass

        # Create a new connection
        return self._create_connection()

    def return_connection(self, conn: http.client.HTTPConnection) -> None:
        """Return a connection to the pool for reuse.

        Args:
            conn: The connection to return.
        """
        with self._lock:
            if len(self._pool) < self.max_connections:
                self._pool.append(conn)
            else:
                try:
                    conn.close()
                except Exception:
                    pass

    def close_connection(self, conn: http.client.HTTPConnection) -> None:
        """Close a connection without returning it to the pool.

        Args:
            conn: The connection to close.
        """
        try:
            conn.close()
        except Exception:
            pass

    def _create_connection(self) -> http.client.HTTPConnection:
        """Create a new connection to the upstream server.

        Returns:
            A new HTTP(S) connection.
        """
        if self.use_https:
            context = ssl.create_default_context()
            conn = http.client.HTTPSConnection(
                self.host,
                self.port,
                timeout=self.timeout,
                context=context,
            )
        else:
            conn = http.client.HTTPConnection(
                self.host,
                self.port,
                timeout=self.timeout,
            )
        self._created_count += 1
        return conn

    def close_all(self) -> None:
        """Close all pooled connections."""
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self._pool.clear()

    @property
    def pool_size(self) -> int:
        """Current number of connections in the pool."""
        with self._lock:
            return len(self._pool)

    @property
    def total_created(self) -> int:
        """Total number of connections created."""
        return self._created_count


# --------------------------------------------------------------------------- #
# Request handler
# --------------------------------------------------------------------------- #


class AnthropicPassthroughHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler that forwards requests to the Anthropic API.

    Accepts requests on local Anthropic API endpoints and proxies them
    to api.anthropic.com with proper authentication and headers.
    Supports both streaming (SSE) and non-streaming responses.

    Non-streaming requests are retried on transient failures using
    exponential backoff.  Streaming requests are NOT retried because
    partial SSE data may have already been sent to the client.

    Supports two operating modes:
    - **Strict mode** (default): Only whitelisted API paths are forwarded.
    - **Pass-through mode**: All /v1/* requests are forwarded transparently.

    The handler also serves internal gateway endpoints:
    - ``/health`` — Health check endpoint.
    - ``/status`` — Gateway metrics and status.
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

    # Pass-through mode: when True, all /v1/* paths are forwarded
    passthrough_mode: bool = False

    # Max request body size in bytes (0 = unlimited)
    max_request_size: int = DEFAULT_MAX_REQUEST_SIZE

    # Connection pool (shared across handler instances)
    connection_pool: Optional[ConnectionPool] = None

    # Response cache: caches GET/HEAD responses to avoid redundant upstream
    # calls for read-only endpoints such as /v1/models.
    # Set to a ResponseCache instance to enable; None to disable.
    response_cache: Optional[Any] = None

    # Token count cache: caches POST /v1/messages/count_tokens responses.
    # Token counting is deterministic — same input always yields the same
    # token count — so these responses are safe to cache indefinitely.
    # Set to a TokenCountCache instance to enable; None to disable.
    token_count_cache: Optional[Any] = None

    # Server start time (set by create_passthrough_server)
    _server_start_time: float = 0.0

    # Request counter (thread-safe via class-level lock)
    _request_count: int = 0
    _request_lock: threading.Lock = threading.Lock()

    # Cache hit/miss counters
    _cache_hits: int = 0
    _cache_misses: int = 0
    _cache_lock: threading.Lock = threading.Lock()

    # Token count cache hit/miss counters
    _token_count_cache_hits: int = 0
    _token_count_cache_misses: int = 0
    _token_count_cache_lock: threading.Lock = threading.Lock()

    def log_message(self, format: str, *args) -> None:
        """Route access logs through the logging module."""
        logger.info("anthropic-passthrough: %s", format % args)

    # ------------------------------------------------------------------ #
    # HTTP method handlers
    # ------------------------------------------------------------------ #

    def do_GET(self):
        """Handle GET requests (e.g., /v1/models, /health)."""
        self._handle_request()

    def do_POST(self):
        """Handle POST requests (e.g., /v1/messages)."""
        self._handle_request()

    def do_PUT(self):
        """Handle PUT requests."""
        self._handle_request()

    def do_DELETE(self):
        """Handle DELETE requests (e.g., batch cancellation)."""
        self._handle_request()

    def do_PATCH(self):
        """Handle PATCH requests."""
        self._handle_request()

    def do_HEAD(self):
        """Handle HEAD requests."""
        self._handle_request()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, x-api-key, anthropic-version, anthropic-beta, "
            "x-request-id, authorization",
        )
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    # ------------------------------------------------------------------ #
    # Internal gateway endpoints
    # ------------------------------------------------------------------ #

    def _handle_health(self) -> None:
        """Serve the /health endpoint.

        Returns a simple JSON response indicating the gateway is running
        and can reach the upstream (based on recent error tracking).
        """
        tracker = get_error_tracker()
        health = tracker.get_provider_health("anthropic")

        status = "healthy"
        http_code = 200

        if health.consecutive_failures >= 10:
            status = "unhealthy"
            http_code = 503
        elif health.consecutive_failures >= 3:
            status = "degraded"
            http_code = 200

        body = json.dumps(
            {
                "status": status,
                "gateway": "anthropic-passthrough",
                "upstream": f"{'https' if self.use_https else 'http'}://"
                f"{self.anthropic_api_host}:{self.anthropic_api_port}",
                "version": self.anthropic_version,
                "uptime_seconds": round(
                    time.monotonic() - self._server_start_time, 1
                )
                if self._server_start_time
                else 0,
            }
        ).encode("utf-8")

        self.send_response(http_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _handle_status(self) -> None:
        """Serve the /status endpoint.

        Returns detailed gateway metrics including request counts,
        error rates, and connection pool information.
        """
        tracker = get_error_tracker()
        system_health = tracker.get_system_health()

        with self._request_lock:
            total_requests = self._request_count

        pool_info = {}
        if self.connection_pool:
            pool_info = {
                "pool_size": self.connection_pool.pool_size,
                "total_created": self.connection_pool.total_created,
                "max_connections": self.connection_pool.max_connections,
            }

        # Gather cache statistics
        cache_info: dict = {"enabled": self.response_cache is not None}
        if self.response_cache is not None:
            try:
                rc_stats = self.response_cache.get_stats()
                with self._cache_lock:
                    cache_hits = self._cache_hits
                    cache_misses = self._cache_misses
                total_cache_req = cache_hits + cache_misses
                cache_info.update({
                    "hits": cache_hits,
                    "misses": cache_misses,
                    "hit_rate": round(
                        (cache_hits / total_cache_req * 100.0)
                        if total_cache_req > 0 else 0.0,
                        2,
                    ),
                    "size": self.response_cache.size,
                    "response_cache_stats": rc_stats.to_dict(),
                })
            except Exception:
                pass

        # Gather token count cache statistics
        token_count_cache_info: dict = {
            "enabled": self.token_count_cache is not None
        }
        if self.token_count_cache is not None:
            try:
                tc_stats = self.token_count_cache.get_stats()
                with self._token_count_cache_lock:
                    tc_hits = self._token_count_cache_hits
                    tc_misses = self._token_count_cache_misses
                total_tc_req = tc_hits + tc_misses
                token_count_cache_info.update({
                    "hits": tc_hits,
                    "misses": tc_misses,
                    "hit_rate": round(
                        (tc_hits / total_tc_req * 100.0)
                        if total_tc_req > 0 else 0.0,
                        2,
                    ),
                    "size": self.token_count_cache.size,
                    "stats": tc_stats.to_dict(),
                })
            except Exception:
                pass

        body = json.dumps(
            {
                "gateway": "anthropic-passthrough",
                "mode": "passthrough" if self.passthrough_mode else "strict",
                "upstream": {
                    "host": self.anthropic_api_host,
                    "port": self.anthropic_api_port,
                    "https": self.use_https,
                    "version": self.anthropic_version,
                },
                "config": {
                    "timeout": self.upstream_timeout,
                    "max_retries": self.max_retries,
                    "max_request_size": self.max_request_size,
                },
                "stats": {
                    "total_requests": total_requests,
                    "health": system_health.to_dict(),
                },
                "connection_pool": pool_info,
                "cache": cache_info,
                "token_count_cache": token_count_cache_info,
                "uptime_seconds": round(
                    time.monotonic() - self._server_start_time, 1
                )
                if self._server_start_time
                else 0,
            },
            indent=2,
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------ #
    # Core pass-through logic
    # ------------------------------------------------------------------ #

    def _handle_request(self) -> None:
        """Forward the incoming request to the Anthropic API."""
        request_start = time.monotonic()

        # Generate and set correlation ID
        request_id = self.headers.get("x-request-id") or str(uuid.uuid4())
        set_correlation_id(request_id)

        # Increment request counter
        with self._request_lock:
            type(self)._request_count += 1

        # Check for internal gateway endpoints first
        path = self.path.split("?")[0]  # Strip query string

        if path == "/health":
            self._handle_health()
            return

        if path == "/status":
            self._handle_status()
            return

        # Validate path
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
                        )
                        if not self.passthrough_mode
                        else (
                            f"Path '{self.path}' must start with /v1/ to be "
                            f"forwarded to the Anthropic API."
                        ),
                    },
                },
                request_id=request_id,
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
                request_id=request_id,
            )
            return

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))

        # Validate request body size
        if (
            self.max_request_size > 0
            and content_length > self.max_request_size
        ):
            self._send_error(
                413,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            f"Request body too large ({content_length} bytes). "
                            f"Maximum allowed: {self.max_request_size} bytes."
                        ),
                    },
                },
                request_id=request_id,
            )
            return

        body = self.rfile.read(content_length) if content_length > 0 else None

        # -----------------------------------------------------------------
        # Response cache lookup (GET/HEAD requests only)
        # -----------------------------------------------------------------
        if (
            self.response_cache is not None
            and self.command in _CACHEABLE_METHODS
        ):
            request_headers_dict = dict(self.headers)
            cached_resp = self.response_cache.lookup(
                method=self.command,
                path=self.path,
                headers=request_headers_dict,
                body=body,
            )
            if cached_resp is not None:
                with self._cache_lock:
                    type(self)._cache_hits += 1
                logger.info(
                    "Response cache HIT for %s %s (age=%.1fs, request_id=%s)",
                    self.command,
                    self.path,
                    cached_resp.age,
                    request_id,
                )
                self._serve_cached_response(cached_resp, request_id, request_start)
                return
            else:
                with self._cache_lock:
                    type(self)._cache_misses += 1

        # -----------------------------------------------------------------
        # Token count cache lookup (POST /v1/messages/count_tokens only)
        # -----------------------------------------------------------------
        if self.token_count_cache is not None and body:
            request_headers_dict = dict(self.headers)
            if self.token_count_cache.is_cacheable(
                path=self.path,
                method=self.command,
                request_body=body,
                request_headers=request_headers_dict,
            ):
                tc_entry = self.token_count_cache.lookup(
                    request_body=body,
                    request_headers=request_headers_dict,
                )
                if tc_entry is not None:
                    with self._token_count_cache_lock:
                        type(self)._token_count_cache_hits += 1
                    logger.info(
                        "Token count cache HIT for %s (tokens=%d, "
                        "age=%.1fs, request_id=%s)",
                        self.path,
                        tc_entry.input_tokens,
                        tc_entry.age,
                        request_id,
                    )
                    self._serve_token_count_cached_response(
                        tc_entry, request_id, request_start
                    )
                    return
                else:
                    with self._token_count_cache_lock:
                        type(self)._token_count_cache_misses += 1

        # Determine if this is a streaming request
        is_streaming = False
        if body:
            try:
                request_data = json.loads(body)
                is_streaming = request_data.get("stream", False)
            except (json.JSONDecodeError, AttributeError):
                pass

        # Build upstream headers
        upstream_headers = self._build_upstream_headers(api_key, request_id)

        # Add Content-Type and Content-Length for body requests
        if body is not None:
            upstream_headers["Content-Type"] = (
                self.headers.get("Content-Type", "application/json")
            )
            upstream_headers["Content-Length"] = str(len(body))

        logger.info(
            "Forwarding %s %s to %s (streaming=%s, request_id=%s)",
            self.command,
            self.path,
            self.anthropic_api_host,
            is_streaming,
            request_id,
        )

        if is_streaming:
            # Streaming requests are NOT retried — partial data may
            # have been sent to the client already.
            self._forward_streaming(body, upstream_headers, request_start, request_id)
        else:
            # Non-streaming requests use retry logic
            self._forward_with_retry(
                body, upstream_headers, request_start, request_id
            )

    # ------------------------------------------------------------------ #
    # Cached response serving
    # ------------------------------------------------------------------ #

    def _serve_cached_response(
        self,
        cached_resp: Any,
        request_id: str,
        request_start: float,
    ) -> None:
        """Serve a response directly from the response cache.

        Sends the cached status, headers, and body to the client without
        making an upstream request.

        Args:
            cached_resp: A :class:`~src.response_cache.CachedResponse` object.
            request_id: Request correlation ID for tracing.
            request_start: Monotonic timestamp when the request arrived.
        """
        try:
            self.send_response(cached_resp.status_code)

            # Forward cached response headers
            for key, val in cached_resp.headers.items():
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, val)

            # Add gateway-specific headers
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("x-request-id", request_id)
            self.send_header("X-Cache", "HIT")
            self.send_header("Age", str(int(cached_resp.age)))
            self.end_headers()

            if cached_resp.body:
                self.wfile.write(cached_resp.body)

        except (BrokenPipeError, ConnectionResetError):
            logger.warning(
                "Client disconnected while sending cached response for %s %s",
                self.command,
                self.path,
            )
            return

        duration_ms = (time.monotonic() - request_start) * 1000.0
        client_addr = self.client_address[0] if self.client_address else None

        log_request(
            method=self.command,
            path=self.path,
            status_code=cached_resp.status_code,
            duration_ms=duration_ms,
            client_ip=client_addr,
            extra={
                "upstream": "cache",
                "streaming": False,
                "request_id": request_id,
                "cache_hit": True,
                "cache_age_seconds": round(cached_resp.age, 2),
            },
            logger=logger,
        )

    def _serve_token_count_cached_response(
        self,
        tc_entry: Any,
        request_id: str,
        request_start: float,
    ) -> None:
        """Serve a token count response directly from the token count cache.

        Args:
            tc_entry: A :class:`~src.token_count_cache.TokenCountEntry`.
            request_id: Request correlation ID for tracing.
            request_start: Monotonic timestamp when the request arrived.
        """
        try:
            response_body = tc_entry.get_response_body()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("x-request-id", request_id)
            self.send_header("X-Token-Count-Cache", "HIT")
            self.send_header("Age", str(int(tc_entry.age)))
            self.end_headers()

            self.wfile.write(response_body)

        except (BrokenPipeError, ConnectionResetError):
            logger.warning(
                "Client disconnected while sending token count cached response "
                "for %s %s",
                self.command,
                self.path,
            )
            return

        duration_ms = (time.monotonic() - request_start) * 1000.0
        client_addr = self.client_address[0] if self.client_address else None

        log_request(
            method=self.command,
            path=self.path,
            status_code=200,
            duration_ms=duration_ms,
            client_ip=client_addr,
            extra={
                "upstream": "token_count_cache",
                "streaming": False,
                "request_id": request_id,
                "token_count_cache_hit": True,
                "cached_tokens": tc_entry.input_tokens,
                "cache_age_seconds": round(tc_entry.age, 2),
            },
            logger=logger,
        )

    # ------------------------------------------------------------------ #
    # Non-streaming with retry
    # ------------------------------------------------------------------ #

    def _forward_with_retry(
        self,
        body: Optional[bytes],
        upstream_headers: dict[str, str],
        request_start: float,
        request_id: str,
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
            conn = None
            use_pool = self.connection_pool is not None
            try:
                if use_pool:
                    conn = self.connection_pool.get_connection()
                elif self.use_https:
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

                # Return connection to pool on success (non-retryable status)
                if use_pool and not is_retryable_status(resp_status):
                    self.connection_pool.return_connection(conn)
                    conn = None  # Prevent double-close
                elif use_pool:
                    self.connection_pool.close_connection(conn)
                    conn = None
                else:
                    conn.close()
                    conn = None

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
                if conn and use_pool:
                    self.connection_pool.close_connection(conn)
                elif conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                raise  # Already structured — propagate for retry
            except ssl.SSLError as exc:
                if conn and use_pool:
                    self.connection_pool.close_connection(conn)
                elif conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                raise GatewaySSLError(
                    f"SSL error connecting to Anthropic API: {exc}",
                    host=self.anthropic_api_host,
                    port=self.anthropic_api_port,
                    cause=exc,
                )
            except (socket.timeout, TimeoutError) as exc:
                if conn and use_pool:
                    self.connection_pool.close_connection(conn)
                elif conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                raise GatewayTimeoutError(
                    self.upstream_timeout,
                    host=self.anthropic_api_host,
                    port=self.anthropic_api_port,
                    cause=exc,
                )
            except ConnectionRefusedError as exc:
                if conn and use_pool:
                    self.connection_pool.close_connection(conn)
                elif conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                raise NetworkError(
                    f"Connection refused by {self.anthropic_api_host}:{self.anthropic_api_port}",
                    host=self.anthropic_api_host,
                    port=self.anthropic_api_port,
                    cause=exc,
                )
            except (ConnectionError, OSError) as exc:
                if conn and use_pool:
                    self.connection_pool.close_connection(conn)
                elif conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                raise NetworkError(
                    f"Connection error: {exc}",
                    host=self.anthropic_api_host,
                    port=self.anthropic_api_port,
                    cause=exc,
                )
            except Exception as exc:
                if conn and use_pool:
                    self.connection_pool.close_connection(conn)
                elif conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
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

            # -----------------------------------------------------------------
            # Store successful GET/HEAD responses in the response cache
            # -----------------------------------------------------------------
            if (
                self.response_cache is not None
                and self.command in _CACHEABLE_METHODS
                and status == 200
                and resp_body
            ):
                resp_headers_dict = {k: v for k, v in resp_headers}
                try:
                    self.response_cache.store(
                        method=self.command,
                        path=self.path,
                        status_code=status,
                        headers=resp_headers_dict,
                        body=resp_body,
                        request_headers=dict(self.headers),
                        request_body=body,
                    )
                    logger.debug(
                        "Response cached for %s %s (size=%d bytes)",
                        self.command,
                        self.path,
                        len(resp_body),
                    )
                except Exception as cache_exc:
                    logger.warning(
                        "Failed to cache response for %s %s: %s",
                        self.command,
                        self.path,
                        cache_exc,
                    )

            # -----------------------------------------------------------------
            # Store successful token count responses in the token count cache
            # -----------------------------------------------------------------
            path_no_qs = self.path.split("?")[0]
            if (
                self.token_count_cache is not None
                and self.command == "POST"
                and path_no_qs == "/v1/messages/count_tokens"
                and status == 200
                and resp_body
                and body
            ):
                try:
                    # Extract input_tokens from the response JSON
                    resp_json = json.loads(resp_body)
                    input_tokens = int(resp_json.get("input_tokens", 0))
                    model = str(resp_json.get("model", ""))
                    self.token_count_cache.store(
                        request_body=body,
                        input_tokens=input_tokens,
                        response_body=resp_body,
                        model=model,
                    )
                    logger.debug(
                        "Token count cached for %s (tokens=%d, size=%d bytes)",
                        self.path,
                        input_tokens,
                        len(resp_body),
                    )
                except Exception as tc_exc:
                    logger.warning(
                        "Failed to cache token count response for %s: %s",
                        self.path,
                        tc_exc,
                    )

            # Relay status
            self.send_response(status, reason)

            # Relay response headers
            for key, val in resp_headers:
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, val)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("x-request-id", request_id)
            self.end_headers()

            # Relay body
            if resp_body:
                self.wfile.write(resp_body)

            duration_ms = (time.monotonic() - request_start) * 1000.0
            client_addr = self.client_address[0] if self.client_address else None

            # Record success in global tracker
            try:
                tracker = get_error_tracker()
                tracker.record_success(
                    "anthropic",
                    latency_ms=duration_ms,
                    status_code=status,
                )
            except Exception:
                pass

            log_request(
                method=self.command,
                path=self.path,
                status_code=status,
                duration_ms=duration_ms,
                client_ip=client_addr,
                extra={
                    "upstream": self.anthropic_api_host,
                    "streaming": False,
                    "request_id": request_id,
                    "cache_hit": False,
                },
                logger=logger,
            )

        except GatewayError as exc:
            # Record error in global tracker
            try:
                duration_ms = (time.monotonic() - request_start) * 1000.0
                tracker = get_error_tracker()
                tracker.record_error(
                    "anthropic", exc, latency_ms=duration_ms
                )
            except Exception:
                pass
            self._handle_gateway_error(exc, request_start, request_id)
        except Exception as exc:
            # Record error in global tracker
            try:
                duration_ms = (time.monotonic() - request_start) * 1000.0
                tracker = get_error_tracker()
                tracker.record_error(
                    "anthropic", exc, latency_ms=duration_ms
                )
            except Exception:
                pass
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
                request_id=request_id,
            )

    # ------------------------------------------------------------------ #
    # Streaming (no retry — partial data prevents safe retry)
    # ------------------------------------------------------------------ #

    def _forward_streaming(
        self,
        body: Optional[bytes],
        upstream_headers: dict[str, str],
        request_start: float,
        request_id: str,
    ) -> None:
        """Forward a streaming request without retries."""
        conn = None
        use_pool = self.connection_pool is not None
        try:
            if use_pool:
                conn = self.connection_pool.get_connection()
            elif self.use_https:
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
            self._relay_streaming_response(upstream_resp, request_id)

            # Don't return streaming connections to pool
            if use_pool:
                self.connection_pool.close_connection(conn)
            else:
                conn.close()
            conn = None

            duration_ms = (time.monotonic() - request_start) * 1000.0
            client_addr = self.client_address[0] if self.client_address else None

            # Record success
            try:
                tracker = get_error_tracker()
                tracker.record_success(
                    "anthropic",
                    latency_ms=duration_ms,
                    status_code=upstream_resp.status,
                )
            except Exception:
                pass

            log_request(
                method=self.command,
                path=self.path,
                status_code=upstream_resp.status,
                duration_ms=duration_ms,
                client_ip=client_addr,
                extra={
                    "upstream": self.anthropic_api_host,
                    "streaming": True,
                    "request_id": request_id,
                },
                logger=logger,
            )

        except ssl.SSLError as exc:
            if conn and use_pool:
                self.connection_pool.close_connection(conn)
            elif conn:
                try:
                    conn.close()
                except Exception:
                    pass
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
                request_id=request_id,
            )
        except TimeoutError as exc:
            if conn and use_pool:
                self.connection_pool.close_connection(conn)
            elif conn:
                try:
                    conn.close()
                except Exception:
                    pass
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
                request_id=request_id,
            )
        except Exception as exc:
            if conn and use_pool:
                self.connection_pool.close_connection(conn)
            elif conn:
                try:
                    conn.close()
                except Exception:
                    pass
            # Record error
            try:
                duration_ms = (time.monotonic() - request_start) * 1000.0
                tracker = get_error_tracker()
                tracker.record_error("anthropic", exc, latency_ms=duration_ms)
            except Exception:
                pass
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
                request_id=request_id,
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
        self,
        upstream_resp: http.client.HTTPResponse,
        request_id: str = "",
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
        if request_id:
            self.send_header("x-request-id", request_id)
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
        self,
        exc: GatewayError,
        request_start: float,
        request_id: str = "",
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
            "Gateway error for %s %s: %s (%.1fms, request_id=%s)",
            self.command, self.path, exc, duration_ms, request_id,
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
            request_id=request_id,
        )

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _is_allowed_path(self, path: str) -> bool:
        """Check if the request path is an allowed Anthropic API endpoint.

        In pass-through mode, all paths starting with /v1/ are allowed.
        In strict mode, only explicitly listed paths and their sub-paths
        are allowed.

        Args:
            path: The request path (without query string).

        Returns:
            True if the path is allowed.
        """
        # Internal paths are handled separately
        if path in GATEWAY_INTERNAL_PATHS:
            return False  # Will be handled by _handle_request

        # Pass-through mode: allow all /v1/ paths
        if self.passthrough_mode:
            return path.startswith("/v1/")

        # Strict mode: exact match or prefix match
        if path in ALLOWED_PATHS:
            return True
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

    def _build_upstream_headers(
        self,
        api_key: str,
        request_id: str = "",
    ) -> dict[str, str]:
        """Build the headers to send to the Anthropic API.

        Args:
            api_key: The Anthropic API key.
            request_id: Request correlation ID.

        Returns:
            Dictionary of headers for the upstream request.
        """
        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.anthropic_version,
            "Host": self.anthropic_api_host,
            "Accept": "application/json",
        }

        # Add request correlation ID
        if request_id:
            headers["x-request-id"] = request_id

        # Forward select client headers
        for key, val in self.headers.items():
            lower_key = key.lower()
            # Skip hop-by-hop and managed headers
            if lower_key in HOP_BY_HOP_HEADERS:
                continue
            if lower_key in MANAGED_HEADERS:
                continue
            # Forward anthropic-specific headers (anthropic-beta, etc.)
            if lower_key.startswith("anthropic-"):
                headers[key] = val
            # Forward user-agent
            elif lower_key == "user-agent":
                headers[key] = val
            # Forward accept headers
            elif lower_key.startswith("accept"):
                headers[key] = val

        return headers

    def _send_error(
        self,
        code: int,
        error_body: dict,
        request_id: str = "",
    ) -> None:
        """Send a JSON error response to the client.

        Args:
            code: HTTP status code.
            error_body: Error response body (Anthropic error format).
            request_id: Request correlation ID.
        """
        body = json.dumps(error_body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if request_id:
            self.send_header("x-request-id", request_id)
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
    passthrough_mode: bool = False,
    max_request_size: int = DEFAULT_MAX_REQUEST_SIZE,
    enable_connection_pool: bool = True,
    max_pool_connections: int = 10,
    enable_response_cache: bool = False,
    response_cache_ttl: float = _RESPONSE_CACHE_TTL,
    response_cache_maxsize: int = 256,
    enable_token_count_cache: bool = False,
    token_count_cache_ttl: float = 3600.0,
    token_count_cache_maxsize: int = 512,
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
        passthrough_mode: When True, forward all /v1/* paths instead of
            only the whitelisted set. Default: False (strict mode).
        max_request_size: Maximum request body size in bytes (0 = unlimited).
        enable_connection_pool: Whether to use connection pooling.
        max_pool_connections: Maximum connections in the pool.
        enable_response_cache: Whether to cache GET/HEAD responses to
            avoid redundant upstream calls. Ideal for endpoints like
            ``/v1/models`` that rarely change. Default: False.
        response_cache_ttl: TTL in seconds for cached responses
            (default: 300 seconds / 5 minutes).
        response_cache_maxsize: Maximum number of cached responses
            (default: 256).
        enable_token_count_cache: Whether to cache POST
            ``/v1/messages/count_tokens`` responses.  Token counting is
            deterministic, so caching eliminates redundant API calls for
            repeated identical inputs.  Default: False.
        token_count_cache_ttl: TTL in seconds for cached token count
            responses (default: 3600 seconds / 1 hour).
        token_count_cache_maxsize: Maximum number of cached token count
            responses (default: 512).

    Returns:
        A configured ThreadedGatewayServer instance.
    """
    # Build the connection pool before constructing the handler subclass so
    # the reference is captured in the class dict rather than on the mutable
    # base class.
    connection_pool: Optional[ConnectionPool] = None
    if enable_connection_pool:
        connection_pool = ConnectionPool(
            host=anthropic_api_host,
            port=anthropic_api_port,
            use_https=use_https,
            timeout=timeout,
            max_connections=max_pool_connections,
        )

    # Build the response cache (optional).
    response_cache_obj: Optional[Any] = None
    if enable_response_cache:
        from src.response_cache import ResponseCache

        response_cache_obj = ResponseCache(
            maxsize=response_cache_maxsize,
            default_ttl=response_cache_ttl,
            cacheable_methods=_CACHEABLE_METHODS,
            enable_compression=True,
            name="passthrough_response_cache",
        )
        logger.info(
            "Response cache enabled (maxsize=%d, ttl=%.0fs)",
            response_cache_maxsize,
            response_cache_ttl,
        )

    # Build the token count cache (optional).
    token_count_cache_obj: Optional[Any] = None
    if enable_token_count_cache:
        from src.token_count_cache import TokenCountCache

        token_count_cache_obj = TokenCountCache(
            maxsize=token_count_cache_maxsize,
            default_ttl=token_count_cache_ttl,
            enable_compression=True,
            name="passthrough_token_count_cache",
        )
        logger.info(
            "Token count cache enabled (maxsize=%d, ttl=%.0fs)",
            token_count_cache_maxsize,
            token_count_cache_ttl,
        )

    # Create a unique handler *subclass* for this server so that each server
    # instance carries its own configuration and counters without mutating
    # the base AnthropicPassthroughHandler class attributes.  This ensures
    # that default values on the base class remain unchanged and that
    # multiple concurrent server instances do not interfere with each other.
    handler = type(
        "_AnthropicPassthroughHandlerInstance",
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
            "passthrough_mode": passthrough_mode,
            "max_request_size": max_request_size,
            "connection_pool": connection_pool,
            "response_cache": response_cache_obj,
            "token_count_cache": token_count_cache_obj,
            "_server_start_time": time.monotonic(),
            "_request_count": 0,
            "_cache_hits": 0,
            "_cache_misses": 0,
            "_token_count_cache_hits": 0,
            "_token_count_cache_misses": 0,
            # Each server instance gets its own locks to avoid contention
            # between independently running servers during tests.
            "_request_lock": threading.Lock(),
            "_cache_lock": threading.Lock(),
            "_token_count_cache_lock": threading.Lock(),
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
    passthrough_mode: bool = False,
    max_request_size: int = DEFAULT_MAX_REQUEST_SIZE,
    enable_connection_pool: bool = True,
    max_pool_connections: int = 10,
    enable_response_cache: bool = False,
    response_cache_ttl: float = _RESPONSE_CACHE_TTL,
    response_cache_maxsize: int = 256,
    enable_token_count_cache: bool = False,
    token_count_cache_ttl: float = 3600.0,
    token_count_cache_maxsize: int = 512,
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
        passthrough_mode: When True, forward all /v1/* paths.
        max_request_size: Maximum request body size in bytes.
        enable_connection_pool: Whether to use connection pooling.
        max_pool_connections: Maximum connections in the pool.
        enable_response_cache: Whether to cache GET/HEAD responses.
        response_cache_ttl: TTL in seconds for cached responses.
        response_cache_maxsize: Maximum number of cached responses.
        enable_token_count_cache: Whether to cache token count responses.
        token_count_cache_ttl: TTL in seconds for token count cache entries.
        token_count_cache_maxsize: Maximum number of token count cache entries.
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
        passthrough_mode=passthrough_mode,
        max_request_size=max_request_size,
        enable_connection_pool=enable_connection_pool,
        max_pool_connections=max_pool_connections,
        enable_response_cache=enable_response_cache,
        response_cache_ttl=response_cache_ttl,
        response_cache_maxsize=response_cache_maxsize,
        enable_token_count_cache=enable_token_count_cache,
        token_count_cache_ttl=token_count_cache_ttl,
        token_count_cache_maxsize=token_count_cache_maxsize,
    )

    # Start background cache purger when any caching is enabled.
    # This cleans up expired entries periodically so memory stays bounded.
    bg_purger = None
    if enable_response_cache or enable_token_count_cache:
        from src.cache import get_background_purger, list_caches

        bg_purger = get_background_purger(interval=60.0)
        # Register all caches in the global registry with the purger so
        # expired entries are cleaned up automatically.
        for cache_obj in list_caches().values():
            bg_purger.add_cache(cache_obj)
        bg_purger.start()
        logger.info("Background cache purger started (interval=60s)")

    mode = "pass-through" if passthrough_mode else "strict"
    logger.info(
        "Anthropic pass-through server listening on %s:%d "
        "(upstream: %s, version: %s, max_retries: %d, mode: %s, "
        "response_cache: %s, token_count_cache: %s)",
        host,
        port,
        anthropic_api_host,
        anthropic_version,
        max_retries,
        mode,
        "enabled" if enable_response_cache else "disabled",
        "enabled" if enable_token_count_cache else "disabled",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Anthropic pass-through server")
    finally:
        # Stop background purger
        if bg_purger is not None:
            bg_purger.stop()
        # Clean up connection pool
        if server.RequestHandlerClass.connection_pool:
            server.RequestHandlerClass.connection_pool.close_all()
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
    passthrough_mode: bool = False,
    max_request_size: int = DEFAULT_MAX_REQUEST_SIZE,
    enable_connection_pool: bool = True,
    max_pool_connections: int = 10,
    enable_response_cache: bool = False,
    response_cache_ttl: float = _RESPONSE_CACHE_TTL,
    response_cache_maxsize: int = 256,
    enable_token_count_cache: bool = False,
    token_count_cache_ttl: float = 3600.0,
    token_count_cache_maxsize: int = 512,
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
        passthrough_mode: When True, forward all /v1/* paths.
        max_request_size: Maximum request body size in bytes.
        enable_connection_pool: Whether to use connection pooling.
        max_pool_connections: Maximum connections in the pool.
        enable_response_cache: Whether to cache GET/HEAD responses.
        response_cache_ttl: TTL in seconds for cached responses.
        response_cache_maxsize: Maximum number of cached responses.
        enable_token_count_cache: Whether to cache token count responses.
        token_count_cache_ttl: TTL in seconds for token count cache entries.
        token_count_cache_maxsize: Maximum number of token count cache entries.

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
        passthrough_mode=passthrough_mode,
        max_request_size=max_request_size,
        enable_connection_pool=enable_connection_pool,
        max_pool_connections=max_pool_connections,
        enable_response_cache=enable_response_cache,
        response_cache_ttl=response_cache_ttl,
        response_cache_maxsize=response_cache_maxsize,
        enable_token_count_cache=enable_token_count_cache,
        token_count_cache_ttl=token_count_cache_ttl,
        token_count_cache_maxsize=token_count_cache_maxsize,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(
        "Anthropic pass-through server started in background on %s:%d",
        host,
        port,
    )
    return server, thread
