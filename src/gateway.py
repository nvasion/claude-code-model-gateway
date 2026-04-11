"""Multi-provider routing gateway server.

Provides an HTTP server that:

1. Accepts requests in Anthropic Messages API format.
2. Runs them through the :class:`~src.interceptor.InterceptorChain` for
   routing decisions (authentication, rate limiting, model-based routing).
3. Translates requests to the target provider's API format via the
   :class:`~src.translators.registry.TranslatorRegistry`.
4. Forwards the translated request to the appropriate upstream provider.
5. Translates responses back to Anthropic format.
6. Returns the response to the client.

This module provides the glue layer that integrates the :mod:`src.router`,
:mod:`src.interceptor`, and :mod:`src.translators` subsystems into a fully
operational multi-provider gateway.

Typical usage::

    from src.gateway import create_gateway_server, run_gateway_in_thread
    from src.models import GatewayConfig, ProviderConfig, ModelConfig
    from src.interceptor import create_default_chain

    config = GatewayConfig(
        default_provider="openai",
        providers={
            "openai": ProviderConfig(
                name="openai",
                api_base="https://api.openai.com/v1",
                api_key_env_var="OPENAI_API_KEY",
            )
        },
    )

    chain = create_default_chain(config)
    server = create_gateway_server(config, interceptor_chain=chain)
    run_gateway_in_thread(server)

Internal endpoints (not forwarded upstream):

- ``GET /health`` – Health check, returns ``{"status": "ok"}``.
- ``GET /status`` – Gateway status and statistics.
"""

from __future__ import annotations

import http.client
import http.server
import json
import socket
import socketserver
import ssl
import threading
import time
import urllib.parse
import uuid
from typing import Any, Dict, Optional, Tuple

from src.errors import (
    BadGatewayError,
    GatewayError,
    NetworkError,
    SSLError as GatewaySSLError,
    TimeoutError_,
    is_retryable_status,
)
from src.interceptor import (
    InterceptAction,
    InterceptorChain,
    InterceptResult,
    create_default_chain,
)
from src.logging_config import get_logger, set_correlation_id
from src.models import GatewayConfig, ProviderConfig
from src.retry import BackoffStrategy, RetryConfig, retry_call
from src.router import (
    RequestContext,
    extract_model_from_body,
    extract_model_from_path,
)
from src.translators.registry import get_registry


# ---------------------------------------------------------------------------
# Format detection helpers
# ---------------------------------------------------------------------------

#: Maps provider names to their native API format.
_PROVIDER_FORMAT: Dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "openrouter": "openai",
    "local": "openai",
    "azure": "openai",
    "google": "google",
    "bedrock": "bedrock",
}

#: Maps incoming request paths to the API format they represent.
_PATH_FORMAT: Dict[str, str] = {
    "/v1/messages": "anthropic",
    "/v1/chat/completions": "openai",
}


def _detect_source_format(path: str) -> str:
    """Detect the API format of an incoming request from its path."""
    for prefix, fmt in _PATH_FORMAT.items():
        if path == prefix or path.startswith(prefix + "?"):
            return fmt
    return "unknown"


def _get_provider_format(provider_name: str) -> str:
    """Return the API format for a given provider."""
    return _PROVIDER_FORMAT.get(provider_name, "openai")

logger = get_logger("gateway")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Internal paths handled by the gateway itself (never forwarded upstream).
GATEWAY_INTERNAL_PATHS: frozenset[str] = frozenset({"/health", "/status"})

#: Default maximum request body size (10 MiB).
DEFAULT_MAX_REQUEST_SIZE: int = 10 * 1024 * 1024

#: Hop-by-hop headers that must NOT be forwarded to the upstream provider.
HOP_BY_HOP_HEADERS: frozenset[str] = frozenset(
    {
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
)

#: Provider-injected headers that should NOT be forwarded to clients.
INTERNAL_REQUEST_HEADERS: frozenset[str] = frozenset(
    {
        "x-provider",
        "x-api-key",
        "authorization",
    }
)


# ---------------------------------------------------------------------------
# Gateway statistics
# ---------------------------------------------------------------------------


class GatewayStats:
    """Thread-safe statistics container for the gateway.

    Attributes:
        requests_total: Total requests received.
        requests_forwarded: Requests successfully forwarded.
        requests_rejected: Requests rejected by the interceptor chain.
        requests_errored: Requests that errored during forwarding.
        provider_counts: Per-provider request counts.
        start_time: Gateway start timestamp.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total: int = 0
        self.requests_forwarded: int = 0
        self.requests_rejected: int = 0
        self.requests_errored: int = 0
        self.provider_counts: Dict[str, int] = {}
        self.start_time: float = time.time()

    def record_request(self) -> None:
        """Increment total request counter."""
        with self._lock:
            self.requests_total += 1

    def record_forwarded(self, provider: str) -> None:
        """Increment forwarded counter for a provider."""
        with self._lock:
            self.requests_forwarded += 1
            self.provider_counts[provider] = (
                self.provider_counts.get(provider, 0) + 1
            )

    def record_rejected(self) -> None:
        """Increment rejected counter."""
        with self._lock:
            self.requests_rejected += 1

    def record_error(self) -> None:
        """Increment error counter."""
        with self._lock:
            self.requests_errored += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize statistics to a dictionary."""
        with self._lock:
            uptime = time.time() - self.start_time
            return {
                "requests_total": self.requests_total,
                "requests_forwarded": self.requests_forwarded,
                "requests_rejected": self.requests_rejected,
                "requests_errored": self.requests_errored,
                "provider_counts": dict(self.provider_counts),
                "uptime_seconds": round(uptime, 1),
            }


# ---------------------------------------------------------------------------
# Gateway request handler
# ---------------------------------------------------------------------------


class GatewayRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the multi-provider routing gateway.

    Intercepts incoming HTTP requests, resolves the target provider through
    the :class:`~src.interceptor.InterceptorChain`, translates the request
    to the provider's API format, and proxies the response back.

    Class-level attributes are set by :func:`create_gateway_server`:

    Attributes:
        gateway_config: Gateway configuration containing provider definitions.
        interceptor_chain: Chain of interceptors for routing and auth.
        upstream_timeout: Upstream connection timeout in seconds.
        max_request_size: Maximum allowed request body size in bytes.
        max_retries: Number of upstream retry attempts.
        retry_base_delay: Initial retry back-off delay in seconds.
        stats: Shared :class:`GatewayStats` instance.
    """

    # Set by create_gateway_server
    gateway_config: GatewayConfig = GatewayConfig()
    interceptor_chain: Optional[InterceptorChain] = None
    upstream_timeout: int = 30
    max_request_size: int = DEFAULT_MAX_REQUEST_SIZE
    max_retries: int = 0
    retry_base_delay: float = 1.0
    stats: Optional[GatewayStats] = None

    def log_message(self, format: str, *args: Any) -> None:
        """Route access logs through the logging module."""
        logger.debug("gateway: %s", format % args)

    # ---------------------------------------------------------------------- #
    # HTTP method dispatchers
    # ---------------------------------------------------------------------- #

    def do_GET(self) -> None:
        """Handle GET requests."""
        self._handle_request("GET")

    def do_POST(self) -> None:
        """Handle POST requests."""
        self._handle_request("POST")

    def do_PUT(self) -> None:
        """Handle PUT requests."""
        self._handle_request("PUT")

    def do_DELETE(self) -> None:
        """Handle DELETE requests."""
        self._handle_request("DELETE")

    def do_PATCH(self) -> None:
        """Handle PATCH requests."""
        self._handle_request("PATCH")

    def do_HEAD(self) -> None:
        """Handle HEAD requests."""
        self._handle_request("HEAD")

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight and OPTIONS requests."""
        self._send_options_response()

    # ---------------------------------------------------------------------- #
    # Internal endpoint handlers
    # ---------------------------------------------------------------------- #

    def _handle_health(self) -> None:
        """Respond to GET /health with a simple health check."""
        body = json.dumps({"status": "ok", "service": "claude-code-model-gateway"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_status(self) -> None:
        """Respond to GET /status with gateway statistics."""
        data: Dict[str, Any] = {
            "status": "ok",
            "service": "claude-code-model-gateway",
        }
        if self.stats:
            data["statistics"] = self.stats.to_dict()
        if self.interceptor_chain is not None:
            data["interceptor_chain"] = self.interceptor_chain.get_stats()

        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_options_response(self) -> None:
        """Return CORS headers for preflight requests."""
        self.send_response(200)
        self.send_header("Allow", "GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, x-api-key, x-provider, anthropic-version",
        )
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ---------------------------------------------------------------------- #
    # Core request handling
    # ---------------------------------------------------------------------- #

    def _handle_request(self, method: str) -> None:
        """Central dispatcher for all incoming requests.

        Args:
            method: HTTP method string (e.g. ``"POST"``).
        """
        # Generate a correlation ID for request tracing
        correlation_id = str(uuid.uuid4())
        set_correlation_id(correlation_id)

        path = self.path.split("?", 1)[0]  # strip query string
        client_ip = self.client_address[0] if self.client_address else None

        if self.stats:
            self.stats.record_request()

        # ------------------------------------------------------------------ #
        # Internal endpoints
        # ------------------------------------------------------------------ #
        if path in GATEWAY_INTERNAL_PATHS:
            if path == "/health":
                self._handle_health()
            elif path == "/status":
                self._handle_status()
            return

        # ------------------------------------------------------------------ #
        # Read and validate request body
        # ------------------------------------------------------------------ #
        body_bytes, body_dict = self._read_body()
        if body_bytes is None:
            # Error already sent
            return

        # Parse request headers (lowercase keys)
        headers: Dict[str, str] = {
            k.lower(): v for k, v in self.headers.items()
        }

        # ------------------------------------------------------------------ #
        # Extract model (from body, then path)
        # ------------------------------------------------------------------ #
        model: Optional[str] = None
        if body_dict:
            model = extract_model_from_body(body_dict)
        if not model:
            model = extract_model_from_path(path)

        # ------------------------------------------------------------------ #
        # Build request context and run interceptor chain
        # ------------------------------------------------------------------ #
        ctx = RequestContext(
            method=method,
            path=path,
            model=model,
            headers=headers,
            body=body_dict,
            client_ip=client_ip,
        )

        intercept_result = self._run_interceptor_chain(ctx)

        if intercept_result.action == InterceptAction.REJECT:
            self._send_intercept_rejection(intercept_result)
            if self.stats:
                self.stats.record_rejected()
            return

        # ------------------------------------------------------------------ #
        # Resolve provider from interception result or config
        # ------------------------------------------------------------------ #
        provider_name = intercept_result.provider_name
        provider_config = intercept_result.provider_config

        # Fallback: use default provider
        if not provider_name:
            provider_name = self.gateway_config.default_provider
            provider_config = self.gateway_config.get_provider(provider_name)

        if not provider_name or not provider_config:
            self._send_error_response(
                502,
                "no_provider",
                "No upstream provider could be determined for this request.",
            )
            if self.stats:
                self.stats.record_error()
            return

        # ------------------------------------------------------------------ #
        # Merge accumulated headers from interceptors
        # ------------------------------------------------------------------ #
        upstream_headers: Dict[str, str] = {}

        # Start with provider-configured static headers
        if provider_config.headers:
            upstream_headers.update(provider_config.headers)

        # Overlay interceptor-injected headers (auth, transforms, etc.)
        if intercept_result.modified_headers:
            upstream_headers.update(intercept_result.modified_headers)

        # Safety net: ensure auth is always injected for the resolved provider.
        # The auth interceptor runs before model routing, so if routing sets
        # the provider late, auth headers may be missing.
        if (
            provider_config.api_key_env_var
            and "Authorization" not in upstream_headers
            and "x-api-key" not in upstream_headers
        ):
            import os as _os

            _api_key = _os.environ.get(provider_config.api_key_env_var)
            if _api_key:
                from src.models import AuthType

                if provider_config.auth_type == AuthType.BEARER_TOKEN:
                    upstream_headers["Authorization"] = f"Bearer {_api_key}"
                elif provider_config.auth_type == AuthType.API_KEY:
                    upstream_headers["x-api-key"] = _api_key
                logger.debug(
                    "gateway: injected auth for provider %s from %s",
                    provider_name,
                    provider_config.api_key_env_var,
                )

        # Use modified body if the interceptors updated it
        effective_body_dict = intercept_result.modified_body or body_dict
        effective_body_bytes: bytes = b""
        if effective_body_dict is not None:
            effective_body_bytes = json.dumps(effective_body_dict).encode("utf-8")
        elif body_bytes:
            effective_body_bytes = body_bytes

        # ------------------------------------------------------------------ #
        # Deduplicate model requests.
        # Claude Code fires parallel/retry requests for the same prompt
        # to multiple models.  When all route to the same backend, the
        # user sees duplicate output.  Dedup by hashing the last user
        # message — if the same content was forwarded within 30s, drop it.
        # Only applies to POST /v1/messages.
        # ------------------------------------------------------------------ #
        if method == "POST" and "/messages" in path and effective_body_dict:
            import hashlib as _hashlib

            _dedup_lock = getattr(self.__class__, "_dedup_lock", None)
            if _dedup_lock is None:
                import threading as _threading
                self.__class__._dedup_lock = _threading.Lock()
                self.__class__._dedup_cache: Dict[str, float] = {}
                _dedup_lock = self.__class__._dedup_lock

            # Hash the last user message for dedup key
            messages = effective_body_dict.get("messages", [])
            last_user_msg = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        last_user_msg = content
                    elif isinstance(content, list):
                        last_user_msg = str(content)
                    break
            dedup_key = _hashlib.md5(last_user_msg.encode()).hexdigest()

            with _dedup_lock:
                now = time.time()
                # Clean old entries
                self.__class__._dedup_cache = {
                    k: v for k, v in self.__class__._dedup_cache.items()
                    if now - v < 30.0
                }
                if dedup_key in self.__class__._dedup_cache:
                    logger.debug(
                        "gateway: dropping duplicate request (%s) — same message within 30s",
                        ctx.model or "",
                    )
                    self._send_error_response(529, "overloaded", "Model temporarily unavailable")
                    return
                self.__class__._dedup_cache[dedup_key] = now

        # ------------------------------------------------------------------ #
        # Format translation (Anthropic ↔ OpenAI bridging)
        # ------------------------------------------------------------------ #
        source_format = _detect_source_format(path)
        target_format = _get_provider_format(provider_name)
        needs_translation = (
            source_format != "unknown"
            and target_format != "unknown"
            and source_format != target_format
        )

        if needs_translation and effective_body_dict is not None:
            try:
                effective_body_dict, path = self._translate_request(
                    effective_body_dict,
                    source_format=source_format,
                    target_format=target_format,
                    target_model=provider_config.default_model,
                )
                effective_body_bytes = json.dumps(effective_body_dict).encode("utf-8")
                logger.info(
                    "gateway: translated request %s → %s (model=%s)",
                    source_format,
                    target_format,
                    provider_config.default_model,
                )
            except Exception as exc:
                logger.error("gateway: request translation failed: %s", exc)
                self._send_error_response(
                    500, "translation_error", f"Request format translation failed: {exc}"
                )
                if self.stats:
                    self.stats.record_error()
                return

        # Store translation context for response handling
        self._translation_ctx = {
            "needs_translation": needs_translation,
            "source_format": source_format,
            "target_format": target_format,
            "model": provider_config.default_model,
            "is_streaming": effective_body_dict.get("stream", False) if effective_body_dict else False,
        } if needs_translation else None

        # ------------------------------------------------------------------ #
        # Build upstream target URL
        # ------------------------------------------------------------------ #
        upstream_url = self._build_upstream_url(provider_config, path)
        parsed = urllib.parse.urlparse(upstream_url)
        upstream_host = parsed.hostname or ""
        upstream_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        use_ssl = parsed.scheme == "https"
        upstream_path = parsed.path or "/"
        if parsed.query:
            upstream_path = f"{upstream_path}?{parsed.query}"

        # Add required headers for the upstream
        upstream_headers.setdefault("Content-Type", "application/json")
        if effective_body_bytes:
            upstream_headers["Content-Length"] = str(len(effective_body_bytes))
        else:
            upstream_headers.pop("Content-Length", None)

        # Forward select client headers that are safe to pass through
        for header_name in ("accept", "user-agent"):
            if header_name in headers and header_name not in upstream_headers:
                upstream_headers[header_name] = headers[header_name]
        # Only forward anthropic-version if NOT translating to a different format
        if not needs_translation:
            if "anthropic-version" in headers and "anthropic-version" not in upstream_headers:
                upstream_headers["anthropic-version"] = headers["anthropic-version"]

        # ------------------------------------------------------------------ #
        # Forward to upstream (with retry)
        # ------------------------------------------------------------------ #
        logger.info(
            "gateway: %s %s -> %s/%s (provider=%s, model=%s, correlation=%s)",
            method,
            path,
            upstream_host,
            upstream_path,
            provider_name,
            ctx.model or "(none)",
            correlation_id,
        )

        try:
            self._forward_request(
                method=method,
                host=upstream_host,
                port=upstream_port,
                path=upstream_path,
                headers=upstream_headers,
                body=effective_body_bytes,
                use_ssl=use_ssl,
                provider_name=provider_name,
                correlation_id=correlation_id,
            )
            if self.stats:
                self.stats.record_forwarded(provider_name)

        except GatewayError as exc:
            logger.error(
                "gateway: upstream error for %s: %s",
                provider_name,
                exc,
            )
            status = exc.context.status_code if exc.context else 502
            try:
                self._send_error_response(
                    status or 502,
                    "gateway_error",
                    str(exc),
                )
            except (BrokenPipeError, ConnectionResetError, OSError):
                logger.debug("gateway: client disconnected before error response")
            if self.stats:
                self.stats.record_error()

        except (BrokenPipeError, ConnectionResetError):
            logger.debug("gateway: client disconnected during forwarding")
            if self.stats:
                self.stats.record_error()

        except Exception as exc:
            logger.error(
                "gateway: unexpected error forwarding to %s: %s",
                provider_name,
                exc,
                exc_info=True,
            )
            try:
                self._send_error_response(500, "internal_error", str(exc))
            except (BrokenPipeError, ConnectionResetError, OSError):
                logger.debug("gateway: client disconnected before error response")
            if self.stats:
                self.stats.record_error()

    def _read_body(self) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
        """Read and validate the request body.

        Returns:
            A tuple of ``(raw_bytes, parsed_dict)``.  Both will be ``None``
            on error (the error response is sent before returning).  ``parsed_dict``
            may be ``None`` for non-JSON requests.
        """
        content_length_str = self.headers.get("Content-Length", "0")
        try:
            content_length = int(content_length_str or 0)
        except ValueError:
            content_length = 0

        if content_length > self.max_request_size:
            self._send_error_response(
                413,
                "request_too_large",
                f"Request body exceeds maximum size of {self.max_request_size} bytes.",
            )
            return None, None

        body_bytes: bytes = b""
        if content_length > 0:
            try:
                body_bytes = self.rfile.read(content_length)
            except Exception as exc:
                logger.warning("gateway: failed to read request body: %s", exc)
                self._send_error_response(400, "bad_request", "Could not read request body.")
                return None, None

        body_dict: Optional[Dict[str, Any]] = None
        if body_bytes:
            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type or body_bytes.lstrip().startswith(b"{"):
                try:
                    body_dict = json.loads(body_bytes.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.debug("gateway: non-JSON body (ok for some endpoints): %s", exc)
                    # Not a fatal error — some endpoints may not require JSON

        return body_bytes, body_dict

    def _run_interceptor_chain(self, ctx: RequestContext) -> InterceptResult:
        """Execute the interceptor chain for the given request context.

        Returns:
            The final :class:`~src.interceptor.InterceptResult` from the chain.
        """
        if self.interceptor_chain is None:
            # No chain configured — pass through without any interception.
            from src.interceptor import InterceptResult, InterceptAction
            return InterceptResult(action=InterceptAction.SKIP)

        try:
            return self.interceptor_chain.process(ctx)
        except Exception as exc:
            logger.error(
                "gateway: interceptor chain raised unexpected exception: %s",
                exc,
                exc_info=True,
            )
            from src.interceptor import InterceptResult, InterceptAction
            return InterceptResult(action=InterceptAction.SKIP)

    # ------------------------------------------------------------------ #
    # Format translation helpers
    # ------------------------------------------------------------------ #

    def _translate_request(
        self,
        body: Dict[str, Any],
        *,
        source_format: str,
        target_format: str,
        target_model: str,
    ) -> Tuple[Dict[str, Any], str]:
        """Translate a request body and return (translated_body, new_path).

        Currently supports: anthropic → openai.

        Args:
            body: Parsed request body dict.
            source_format: Source API format (e.g. 'anthropic').
            target_format: Target API format (e.g. 'openai').
            target_model: Model name to use in the translated request.

        Returns:
            Tuple of (translated body dict, new URL path).
        """
        if source_format == "anthropic" and target_format == "openai":
            from src.translators.anthropic import AnthropicTranslator

            translator = AnthropicTranslator()
            canonical = translator.parse_request(body, model=target_model)
            # Force non-streaming from upstream — we'll wrap the JSON response
            # as Anthropic SSE ourselves.  This avoids streaming translation
            # issues and gives us full control over the SSE format.
            canonical["stream"] = False
            # Return /chat/completions (not /v1/chat/completions) because
            # provider api_base already includes the /v1 prefix
            # e.g. https://openrouter.ai/api/v1 + /chat/completions
            return canonical, "/chat/completions"

        # Add more format bridges here as needed
        raise ValueError(f"Unsupported translation: {source_format} → {target_format}")

    def _translate_response_body(
        self,
        resp_body: bytes,
        translation_ctx: Dict[str, Any],
    ) -> bytes:
        """Translate a non-streaming response body back to source format.

        Args:
            resp_body: Raw upstream response bytes.
            translation_ctx: Translation context from the request phase.

        Returns:
            Translated response bytes.
        """
        source_format = translation_ctx["source_format"]
        target_format = translation_ctx["target_format"]
        model = translation_ctx.get("model", "")

        if source_format == "anthropic" and target_format == "openai":
            from src.translators.anthropic import AnthropicTranslator

            try:
                openai_resp = json.loads(resp_body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return resp_body  # Can't translate, return as-is

            translator = AnthropicTranslator()
            anthropic_resp = translator.format_response(openai_resp, model=model)
            return json.dumps(anthropic_resp).encode("utf-8")

        return resp_body

    def _relay_streaming_translated(
        self,
        response: http.client.HTTPResponse,
        translation_ctx: Dict[str, Any],
    ) -> None:
        """Stream response with format translation (e.g. OpenAI SSE → Anthropic SSE).

        Buffers incoming SSE events, translates each to the source format,
        and re-emits to the client.

        Args:
            response: Upstream HTTP response.
            translation_ctx: Translation context from the request phase.
        """
        from src.translators.anthropic import AnthropicTranslator

        translator = AnthropicTranslator()
        model = translation_ctx.get("model", "")
        is_first = True
        sent_message_stop = False
        buffer = b""

        def _emit(event_text: str) -> bool:
            """Write an SSE event string; return False if client disconnected."""
            try:
                self.wfile.write(event_text.encode("utf-8"))
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                return False

        def _process_payload(payload: str) -> bool:
            """Process a single SSE data payload. Returns False to stop."""
            nonlocal is_first, sent_message_stop

            if payload.strip() == "[DONE]":
                # Ensure message_stop was sent
                if not sent_message_stop:
                    stop_event = 'event: message_stop\ndata: {"type": "message_stop"}\n\n'
                    sent_message_stop = True
                    return _emit(stop_event)
                return True

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                return True

            events = translator.format_sse_events(
                chunk, model=model, is_first=is_first
            )
            is_first = False

            for event_text in events:
                if "message_stop" in event_text:
                    sent_message_stop = True
                if not _emit(event_text):
                    return False
            return True

        try:
            while True:
                data = response.read(4096)
                if not data:
                    break
                buffer += data

                # Parse complete SSE events (delimited by double newline)
                while b"\n\n" in buffer:
                    event_raw, buffer = buffer.split(b"\n\n", 1)
                    event_str = event_raw.decode("utf-8", errors="replace").strip()
                    if not event_str:
                        continue

                    # Extract "data:" payload from SSE lines
                    for line in event_str.split("\n"):
                        line = line.strip()
                        if line.startswith("data: "):
                            if not _process_payload(line[6:]):
                                return
                        elif line.startswith("data:"):
                            if not _process_payload(line[5:]):
                                return

            # Process any remaining data in buffer
            remaining = buffer.decode("utf-8", errors="replace").strip()
            if remaining:
                for line in remaining.split("\n"):
                    line = line.strip()
                    if line.startswith("data: "):
                        _process_payload(line[6:])
                    elif line.startswith("data:"):
                        _process_payload(line[5:])

            # Always ensure stream is properly terminated
            if not sent_message_stop:
                _emit('event: message_stop\ndata: {"type": "message_stop"}\n\n')

        except (BrokenPipeError, ConnectionResetError, OSError):
            logger.debug("gateway: client disconnected during streaming")
        except Exception as exc:
            logger.debug("gateway: translated streaming relay ended: %s", exc)

    def _emit_json_as_sse(
        self,
        resp_body: bytes,
        translation_ctx: Dict[str, Any],
    ) -> None:
        """Convert a non-streaming OpenAI JSON response to Anthropic SSE events.

        Used when the client requested streaming but the upstream returned a
        plain JSON response.  Emits a complete set of Anthropic SSE events
        (message_start → content_block_start → content_block_delta →
        content_block_stop → message_delta → message_stop).
        """
        from src.translators.anthropic import AnthropicTranslator

        model = translation_ctx.get("model", "")

        try:
            openai_resp = json.loads(resp_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("gateway: could not parse upstream JSON for SSE wrapping")
            return

        translator = AnthropicTranslator()
        anthropic_resp = translator.format_response(openai_resp, model=model)

        # Extract the text content
        text = ""
        for block in anthropic_resp.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        msg_id = anthropic_resp.get("id", "msg_unknown")
        stop_reason = anthropic_resp.get("stop_reason", "end_turn")
        usage = anthropic_resp.get("usage", {})

        def _write(s: str) -> None:
            try:
                self.wfile.write(s.encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError, OSError):
                raise

        try:
            # message_start
            _write(f'event: message_start\ndata: {json.dumps({"type": "message_start", "message": {"id": msg_id, "type": "message", "role": "assistant", "content": [], "model": model, "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0}}})}\n\n')

            # content_block_start
            _write(f'event: content_block_start\ndata: {json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})}\n\n')

            # content_block_delta — send full text in one delta
            _write(f'event: content_block_delta\ndata: {json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})}\n\n')

            # content_block_stop
            _write(f'event: content_block_stop\ndata: {json.dumps({"type": "content_block_stop", "index": 0})}\n\n')

            # message_delta
            _write(f'event: message_delta\ndata: {json.dumps({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": usage.get("output_tokens", 0)}})}\n\n')

            # message_stop
            _write(f'event: message_stop\ndata: {json.dumps({"type": "message_stop"})}\n\n')

            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            logger.debug("gateway: client disconnected during JSON-to-SSE emit")

    def _build_upstream_url(self, provider: ProviderConfig, path: str) -> str:
        """Construct the full upstream URL for a given provider and request path.

        Args:
            provider: The target provider configuration.
            path: The original request path (e.g. ``/v1/messages``).

        Returns:
            Full URL string (e.g. ``https://api.openai.com/v1/messages``).
        """
        base = provider.api_base.rstrip("/")
        # If the provider base already ends with the path, don't double it
        if base.endswith(path.rstrip("/")):
            return base
        # If path starts with /v1/ and base ends with /v1, avoid /v1/v1/...
        base_path = urllib.parse.urlparse(base).path.rstrip("/")
        if path.startswith(base_path + "/") or path == base_path:
            parsed_base = urllib.parse.urlparse(base)
            return f"{parsed_base.scheme}://{parsed_base.netloc}{path}"
        return f"{base}{path}"

    def _forward_request(
        self,
        *,
        method: str,
        host: str,
        port: int,
        path: str,
        headers: Dict[str, str],
        body: bytes,
        use_ssl: bool,
        provider_name: str,
        correlation_id: str,
    ) -> None:
        """Forward the request to the upstream provider and relay the response.

        Handles both streaming (chunked transfer / SSE) and non-streaming
        responses.  Implements retry logic when :attr:`max_retries` > 0.

        Args:
            method: HTTP method.
            host: Upstream hostname.
            port: Upstream port.
            path: Upstream request path.
            headers: Request headers to send upstream.
            body: Request body bytes.
            use_ssl: Whether to use HTTPS.
            provider_name: Provider name (for logging).
            correlation_id: Request correlation ID (for tracing).
        """
        retry_config = RetryConfig(
            max_attempts=max(1, self.max_retries + 1),
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay=self.retry_base_delay,
            max_delay=30.0,
            retry_on_status={429, 500, 502, 503, 504},
        )

        def _attempt() -> None:
            self._do_forward(
                method=method,
                host=host,
                port=port,
                path=path,
                headers=headers,
                body=body,
                use_ssl=use_ssl,
                provider_name=provider_name,
            )

        if self.max_retries > 0:
            retry_call(_attempt, config=retry_config)
        else:
            _attempt()

    def _do_forward(
        self,
        *,
        method: str,
        host: str,
        port: int,
        path: str,
        headers: Dict[str, str],
        body: bytes,
        use_ssl: bool,
        provider_name: str,
    ) -> None:
        """Perform a single upstream HTTP request and relay the response.

        Args:
            method: HTTP method.
            host: Upstream hostname.
            port: Upstream port.
            path: Request path (may include query string).
            headers: Request headers.
            body: Request body bytes.
            use_ssl: Whether to use HTTPS.
            provider_name: Provider name for logging/errors.
        """
        try:
            if use_ssl:
                context = ssl.create_default_context()
                conn: http.client.HTTPConnection = http.client.HTTPSConnection(
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
        except Exception as exc:
            raise NetworkError(f"Failed to create connection to {host}:{port}: {exc}") from exc

        # Filter out hop-by-hop and internal gateway headers
        forwarded_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }

        try:
            conn.request(method, path, body=body or None, headers=forwarded_headers)
            response = conn.getresponse()
        except socket.timeout as exc:
            raise TimeoutError_(
                f"Upstream {host} timed out after {self.upstream_timeout}s"
            ) from exc
        except ssl.SSLError as exc:
            raise GatewaySSLError(f"SSL error connecting to {host}: {exc}") from exc
        except (ConnectionRefusedError, OSError) as exc:
            raise NetworkError(f"Cannot connect to upstream {host}:{port}: {exc}") from exc

        try:
            status = response.status
            reason = response.reason or ""
            is_streaming = self._is_streaming_response(response)
            translation_ctx = getattr(self, "_translation_ctx", None)

            if translation_ctx and translation_ctx["needs_translation"] and status < 400:
                # ---- Translated response path ---- #
                client_wants_stream = translation_ctx.get("is_streaming", False)
                # Only treat as SSE if Content-Type says so (not just chunked encoding)
                content_type = response.getheader("Content-Type", "")
                is_actual_sse = "text/event-stream" in content_type

                logger.debug(
                    "gateway: response path — is_actual_sse=%s, client_wants_stream=%s, "
                    "content_type=%s, transfer_encoding=%s",
                    is_actual_sse,
                    client_wants_stream,
                    content_type,
                    response.getheader("Transfer-Encoding", ""),
                )

                if is_actual_sse:
                    # Upstream is truly streaming SSE — translate events
                    self.send_response(status, reason)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    self._relay_streaming_translated(response, translation_ctx)
                elif client_wants_stream:
                    # Client requested streaming but upstream returned JSON.
                    # Wrap the JSON response as Anthropic SSE events so
                    # Claude Code doesn't get confused.
                    resp_body = response.read()
                    self.send_response(status, reason)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    self._emit_json_as_sse(resp_body, translation_ctx)
                else:
                    resp_body = response.read()
                    translated = self._translate_response_body(resp_body, translation_ctx)
                    self.send_response(status, reason)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(translated)))
                    self.end_headers()
                    self.wfile.write(translated)
            else:
                # ---- Raw passthrough (original behavior) ---- #
                self.send_response(status, reason)
                for header_name, header_val in response.getheaders():
                    name_lower = header_name.lower()
                    if name_lower in HOP_BY_HOP_HEADERS:
                        continue
                    self.send_header(header_name, header_val)

                self.end_headers()

                if is_streaming:
                    self._relay_streaming(response)
                else:
                    resp_body = response.read()
                    if resp_body:
                        self.wfile.write(resp_body)

            # Check for provider errors (non-retryable logged only)
            if status >= 400:
                logger.warning(
                    "gateway: upstream %s returned HTTP %d for provider=%s",
                    host,
                    status,
                    provider_name,
                )
                if is_retryable_status(status):
                    raise BadGatewayError(
                        f"Upstream {provider_name} returned {status}",
                        host=host,
                        port=port,
                    )

        finally:
            conn.close()

    def _is_streaming_response(self, response: http.client.HTTPResponse) -> bool:
        """Detect whether the response uses streaming / SSE.

        Args:
            response: The upstream HTTP response object.

        Returns:
            ``True`` if the response appears to be a streaming response.
        """
        content_type = response.getheader("Content-Type", "")
        transfer_encoding = response.getheader("Transfer-Encoding", "")
        return (
            "text/event-stream" in content_type
            or "chunked" in transfer_encoding.lower()
        )

    def _relay_streaming(self, response: http.client.HTTPResponse) -> None:
        """Stream response chunks to the client as they arrive.

        Args:
            response: The upstream HTTP response (already headers-sent to client).
        """
        try:
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                # Flush to ensure chunks are sent immediately
                try:
                    self.wfile.flush()
                except Exception:
                    break
        except Exception as exc:
            logger.debug("gateway: streaming relay ended: %s", exc)

    # ---------------------------------------------------------------------- #
    # Error response helpers
    # ---------------------------------------------------------------------- #

    def _send_error_response(
        self,
        status_code: int,
        error_type: str,
        message: str,
    ) -> None:
        """Send a JSON error response in Anthropic API format.

        Args:
            status_code: HTTP status code.
            error_type: Short error type string (e.g. ``"authentication_error"``).
            message: Human-readable error message.
        """
        body = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": message,
                },
            }
        ).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_intercept_rejection(self, result: InterceptResult) -> None:
        """Send the rejection response from an interceptor result.

        Args:
            result: The terminal REJECT result from the interceptor chain.
        """
        status = result.status_code or 400
        if result.error_body:
            body = json.dumps(result.error_body).encode("utf-8")
        else:
            body = json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "request_rejected",
                        "message": result.error_message or "Request rejected",
                    },
                }
            ).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))

        # Forward any headers the interceptor attached (e.g. Retry-After)
        for header_name, header_val in result.modified_headers.items():
            self.send_header(header_name, header_val)

        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Threaded server
# ---------------------------------------------------------------------------


class ThreadedGatewayServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Multi-threaded HTTP server for the routing gateway.

    Each request is handled in a separate daemon thread so that slow
    upstream responses do not block the server from accepting new connections.
    """

    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_gateway_server(
    config: GatewayConfig,
    *,
    interceptor_chain: Optional[InterceptorChain] = None,
    host: str = "127.0.0.1",
    port: int = 3001,
    upstream_timeout: int = 30,
    max_request_size: int = DEFAULT_MAX_REQUEST_SIZE,
    max_retries: int = 0,
    retry_base_delay: float = 1.0,
    max_rpm: int = 0,
    model_aliases: Optional[Dict[str, str]] = None,
    require_auth: bool = True,
) -> ThreadedGatewayServer:
    """Create a configured :class:`ThreadedGatewayServer`.

    If *interceptor_chain* is ``None``, a default chain is created from the
    provided *config* using :func:`~src.interceptor.create_default_chain`.

    Args:
        config: Gateway configuration.
        interceptor_chain: Pre-built interceptor chain.  Created from *config*
            if not provided.
        host: Host address to bind to.
        port: Port to listen on.
        upstream_timeout: Upstream connection timeout in seconds.
        max_request_size: Maximum request body size in bytes.
        max_retries: Number of upstream retry attempts on transient errors.
        retry_base_delay: Initial retry back-off delay in seconds.
        max_rpm: Per-client rate limit (requests per minute).  0 = unlimited.
        model_aliases: Optional model alias map for ``RequestTransformInterceptor``.
        require_auth: Whether to reject requests without API keys.

    Returns:
        A ready-to-serve :class:`ThreadedGatewayServer` instance.
    """
    if interceptor_chain is None:
        interceptor_chain = create_default_chain(
            config=config,
            max_rpm=max_rpm,
            model_aliases=model_aliases,
            require_auth=require_auth,
        )

    stats = GatewayStats()

    # Build a custom handler class with the configuration baked in via class
    # attributes (the standard library pattern for http.server).
    handler_class = type(
        "ConfiguredGatewayHandler",
        (GatewayRequestHandler,),
        {
            "gateway_config": config,
            "interceptor_chain": interceptor_chain,
            "upstream_timeout": upstream_timeout,
            "max_request_size": max_request_size,
            "max_retries": max_retries,
            "retry_base_delay": retry_base_delay,
            "stats": stats,
        },
    )

    server = ThreadedGatewayServer((host, port), handler_class)
    server.interceptor_chain = interceptor_chain  # type: ignore[attr-defined]
    server.gateway_config = config  # type: ignore[attr-defined]
    server.stats = stats  # type: ignore[attr-defined]

    logger.info(
        "Gateway server created on %s:%d (providers=%d, timeout=%ds)",
        host,
        port,
        len(config.get_enabled_providers()),
        upstream_timeout,
    )
    return server


def run_gateway(
    config: GatewayConfig,
    *,
    interceptor_chain: Optional[InterceptorChain] = None,
    host: str = "127.0.0.1",
    port: int = 3001,
    upstream_timeout: int = 30,
    max_request_size: int = DEFAULT_MAX_REQUEST_SIZE,
    max_retries: int = 0,
    retry_base_delay: float = 1.0,
    max_rpm: int = 0,
    model_aliases: Optional[Dict[str, str]] = None,
    require_auth: bool = True,
) -> None:
    """Start the gateway server and block until interrupted.

    Args:
        config: Gateway configuration.
        interceptor_chain: Pre-built interceptor chain.
        host: Bind host.
        port: Bind port.
        upstream_timeout: Upstream connection timeout in seconds.
        max_request_size: Maximum request body size in bytes.
        max_retries: Number of upstream retry attempts.
        retry_base_delay: Initial retry back-off delay.
        max_rpm: Per-client rate limit (requests per minute).
        model_aliases: Model alias map.
        require_auth: Whether to reject requests without API keys.
    """
    server = create_gateway_server(
        config,
        interceptor_chain=interceptor_chain,
        host=host,
        port=port,
        upstream_timeout=upstream_timeout,
        max_request_size=max_request_size,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        max_rpm=max_rpm,
        model_aliases=model_aliases,
        require_auth=require_auth,
    )
    logger.info("Starting gateway on %s:%d — press Ctrl+C to stop", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Gateway shutting down")
    finally:
        server.shutdown()


def run_gateway_in_thread(
    server: ThreadedGatewayServer,
    *,
    daemon: bool = True,
) -> threading.Thread:
    """Start the gateway server in a background thread.

    Args:
        server: A server instance created by :func:`create_gateway_server`.
        daemon: Whether the thread should be a daemon (exits with main thread).

    Returns:
        The started :class:`threading.Thread`.
    """
    thread = threading.Thread(target=server.serve_forever, daemon=daemon)
    thread.start()
    logger.info(
        "Gateway started in background thread (daemon=%s)",
        daemon,
    )
    return thread
