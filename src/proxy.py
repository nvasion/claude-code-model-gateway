"""HTTP proxy server for claude-code-model-gateway.

Provides a basic forward HTTP proxy that listens on a configurable host/port
and forwards incoming requests to their target destinations.  Includes
configurable retry logic with exponential backoff for transient upstream
failures.
"""

import http.server
import http.client
import logging
import socket
import socketserver
import ssl
import threading
import time
import urllib.parse
from typing import Optional

from src.errors import (
    BadGatewayError,
    GatewayError,
    GatewayTimeoutError,
    NetworkError,
    SSLError as GatewaySSLError,
    TimeoutError_,
)
from src.error_handling import get_error_tracker
from src.logging_config import get_logger, log_request
from src.retry import BackoffStrategy, RetryConfig, retry_call

logger = get_logger("proxy")


class ProxyRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler that forwards requests to the target server.

    Supports GET, POST, PUT, DELETE, PATCH, HEAD, and OPTIONS methods.
    Acts as a forward HTTP proxy: clients send full URLs and the proxy
    relays the request to the upstream server.

    Retry behaviour is controlled via the class-level ``max_retries``,
    ``retry_base_delay``, and ``retry_max_delay`` attributes, which can
    be set by :func:`create_proxy_server`.
    """

    # Timeout for upstream connections (seconds)
    upstream_timeout: int = 30

    # Optional set of allowed target hosts (None = allow all)
    allowed_hosts: Optional[set] = None

    # Retry configuration (set via create_proxy_server)
    max_retries: int = 0
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0

    # Suppress default stderr logging from BaseHTTPRequestHandler
    def log_message(self, format: str, *args) -> None:
        """Route access logs through the logging module."""
        logger.info("proxy: %s", format % args)

    # ------------------------------------------------------------------ #
    # HTTP method handlers
    # ------------------------------------------------------------------ #

    def do_GET(self):
        """Handle GET requests."""
        self._proxy_request()

    def do_POST(self):
        """Handle POST requests."""
        self._proxy_request()

    def do_PUT(self):
        """Handle PUT requests."""
        self._proxy_request()

    def do_DELETE(self):
        """Handle DELETE requests."""
        self._proxy_request()

    def do_PATCH(self):
        """Handle PATCH requests."""
        self._proxy_request()

    def do_HEAD(self):
        """Handle HEAD requests."""
        self._proxy_request()

    def do_OPTIONS(self):
        """Handle OPTIONS requests."""
        self._proxy_request()

    # ------------------------------------------------------------------ #
    # Core proxy logic
    # ------------------------------------------------------------------ #

    def _proxy_request(self) -> None:
        """Forward the incoming request to the upstream server and relay the response."""
        request_start = time.monotonic()
        parsed = urllib.parse.urlparse(self.path)

        # Determine target host and path
        if parsed.hostname:
            # Full URL provided (classic forward-proxy style)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"
            use_https = parsed.scheme == "https"
        else:
            # Relative path – use the Host header
            host_header = self.headers.get("Host", "")
            if ":" in host_header:
                host, port_str = host_header.rsplit(":", 1)
                port = int(port_str)
            else:
                host = host_header
                port = 80
            path = self.path
            use_https = False

        if not host:
            self._send_error(400, "Bad Request: unable to determine target host")
            return

        # Optional host filtering
        if self.allowed_hosts is not None and host not in self.allowed_hosts:
            self._send_error(403, f"Forbidden: host '{host}' is not allowed")
            return

        # Read request body (if any)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Build headers to forward (skip hop-by-hop headers)
        hop_by_hop = {
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
        forward_headers = {
            key: val
            for key, val in self.headers.items()
            if key.lower() not in hop_by_hop
        }

        # Build retry config — max_retries=0 means 1 attempt (no retries)
        total_attempts = self.max_retries + 1
        retry_config = RetryConfig(
            max_attempts=total_attempts,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
            jitter=True,
        )

        def _do_upstream_request() -> tuple[int, str, list, bytes]:
            """Make a single upstream request, returning response parts.

            Returns:
                (status, reason, headers_list, body) from the upstream.

            Raises:
                Appropriate GatewayError subclass on failure.
            """
            try:
                if use_https:
                    conn = http.client.HTTPSConnection(
                        host, port, timeout=self.upstream_timeout
                    )
                else:
                    conn = http.client.HTTPConnection(
                        host, port, timeout=self.upstream_timeout
                    )

                conn.request(self.command, path, body=body, headers=forward_headers)
                upstream_resp = conn.getresponse()

                # Read response
                resp_status = upstream_resp.status
                resp_reason = upstream_resp.reason
                resp_headers = [
                    (k, v)
                    for k, v in upstream_resp.getheaders()
                    if k.lower() not in hop_by_hop
                ]
                resp_body = upstream_resp.read()
                conn.close()

                # If upstream returned a server error, raise so retry can kick in
                if resp_status >= 500:
                    raise BadGatewayError(
                        f"Upstream returned {resp_status} {resp_reason}",
                        host=host,
                        port=port,
                    )

                return resp_status, resp_reason, resp_headers, resp_body

            except (BadGatewayError, NetworkError, TimeoutError_):
                raise  # Already a GatewayError — propagate for retry
            except ssl.SSLError as exc:
                raise GatewaySSLError(
                    f"SSL error: {exc}", host=host, port=port, cause=exc
                )
            except (socket.timeout, TimeoutError) as exc:
                raise GatewayTimeoutError(
                    self.upstream_timeout, host=host, port=port, cause=exc
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
                    f"Unexpected error: {exc}",
                    host=host,
                    port=port,
                    cause=exc,
                )

        try:
            status, reason, resp_headers, response_body = retry_call(
                _do_upstream_request, config=retry_config
            )

            # Relay status line
            self.send_response(status, reason)

            # Relay response headers
            for key, val in resp_headers:
                self.send_header(key, val)
            self.end_headers()

            # Relay response body
            if response_body:
                self.wfile.write(response_body)

            duration_ms = (time.monotonic() - request_start) * 1000.0
            client_addr = self.client_address[0] if self.client_address else None
            log_request(
                method=self.command,
                path=self.path,
                status_code=status,
                duration_ms=duration_ms,
                client_ip=client_addr,
                extra={"upstream_host": host, "upstream_port": port},
                logger=logger,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - request_start) * 1000.0
            logger.error(
                "Proxy error for %s %s: %s (%.1fms)",
                self.command, self.path, exc, duration_ms,
            )
            # Record error in global tracker
            try:
                tracker = get_error_tracker()
                tracker.record_error(
                    host or "unknown",
                    exc,
                    latency_ms=duration_ms,
                )
            except Exception:
                pass
            self._send_error(502, f"Bad Gateway: {exc}")

    def _send_error(self, code: int, message: str) -> None:
        """Send an error response to the client."""
        body = message.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """HTTP server that handles each request in a separate thread."""

    allow_reuse_address = True
    daemon_threads = True


def create_proxy_server(
    host: str = "127.0.0.1",
    port: int = 3000,
    timeout: int = 30,
    allowed_hosts: Optional[set] = None,
    max_retries: int = 0,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 30.0,
) -> ThreadedHTTPServer:
    """Create (but do not start) a proxy HTTP server.

    Args:
        host: Address to bind to (default: 127.0.0.1).
        port: Port to listen on (default: 3000).
        timeout: Upstream connection timeout in seconds.
        allowed_hosts: Optional set of allowed target hostnames.
        max_retries: Number of retries for failed upstream requests
            (0 = no retries, just one attempt).
        retry_base_delay: Base delay in seconds for retry backoff.
        retry_max_delay: Maximum delay cap in seconds.

    Returns:
        A configured ThreadedHTTPServer instance.
    """
    handler = ProxyRequestHandler
    handler.upstream_timeout = timeout
    handler.allowed_hosts = allowed_hosts
    handler.max_retries = max_retries
    handler.retry_base_delay = retry_base_delay
    handler.retry_max_delay = retry_max_delay

    server = ThreadedHTTPServer((host, port), handler)
    return server


def run_proxy(
    host: str = "127.0.0.1",
    port: int = 3000,
    timeout: int = 30,
    allowed_hosts: Optional[set] = None,
    max_retries: int = 0,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 30.0,
) -> None:
    """Start the proxy server and serve requests forever.

    Args:
        host: Address to bind to (default: 127.0.0.1).
        port: Port to listen on (default: 3000).
        timeout: Upstream connection timeout in seconds.
        allowed_hosts: Optional set of allowed target hostnames.
        max_retries: Number of retries for failed upstream requests.
        retry_base_delay: Base delay in seconds for retry backoff.
        retry_max_delay: Maximum delay cap in seconds.
    """
    server = create_proxy_server(
        host=host,
        port=port,
        timeout=timeout,
        allowed_hosts=allowed_hosts,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
    )
    logger.info("Proxy server listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down proxy server")
    finally:
        server.server_close()


def run_proxy_in_thread(
    host: str = "127.0.0.1",
    port: int = 3000,
    timeout: int = 30,
    allowed_hosts: Optional[set] = None,
    max_retries: int = 0,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 30.0,
) -> tuple[ThreadedHTTPServer, threading.Thread]:
    """Start the proxy server in a background daemon thread.

    Useful for testing or embedding the proxy within another application.

    Args:
        host: Address to bind to.
        port: Port to listen on.
        timeout: Upstream connection timeout in seconds.
        allowed_hosts: Optional set of allowed target hostnames.
        max_retries: Number of retries for failed upstream requests.
        retry_base_delay: Base delay in seconds for retry backoff.
        retry_max_delay: Maximum delay cap in seconds.

    Returns:
        A (server, thread) tuple.  Call server.shutdown() to stop.
    """
    server = create_proxy_server(
        host=host,
        port=port,
        timeout=timeout,
        allowed_hosts=allowed_hosts,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Proxy server started in background on %s:%d", host, port)
    return server, thread
