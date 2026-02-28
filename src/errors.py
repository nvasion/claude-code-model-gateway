"""Comprehensive error hierarchy for claude-code-model-gateway.

Provides a structured exception taxonomy covering all failure modes:
network errors, authentication errors, rate limiting, timeouts,
provider errors, and configuration errors. Each exception carries
structured metadata to support retry decisions, logging, and
user-facing error messages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ErrorCategory(str, Enum):
    """High-level error categories for classification and routing."""

    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    PROVIDER = "provider"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    PROXY = "proxy"
    INTERNAL = "internal"


class ErrorSeverity(str, Enum):
    """How severe is this error for operational monitoring."""

    LOW = "low"          # Transient / expected failures
    MEDIUM = "medium"    # Warrants attention but recoverable
    HIGH = "high"        # Requires investigation
    CRITICAL = "critical"  # Service-impacting


@dataclass
class ErrorContext:
    """Structured context attached to every gateway error.

    Captures details needed for debugging, retry decisions,
    and structured logging / metrics.
    """

    category: ErrorCategory = ErrorCategory.INTERNAL
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    retryable: bool = False
    status_code: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    request_id: Optional[str] = None
    upstream_host: Optional[str] = None
    upstream_port: Optional[int] = None
    retry_after: Optional[float] = None
    attempt: int = 0
    max_attempts: int = 0
    elapsed_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for logging / JSON responses."""
        data: dict[str, Any] = {
            "category": self.category.value,
            "severity": self.severity.value,
            "retryable": self.retryable,
            "timestamp": self.timestamp,
        }
        if self.status_code is not None:
            data["status_code"] = self.status_code
        if self.provider:
            data["provider"] = self.provider
        if self.model:
            data["model"] = self.model
        if self.request_id:
            data["request_id"] = self.request_id
        if self.upstream_host:
            data["upstream_host"] = self.upstream_host
        if self.upstream_port is not None:
            data["upstream_port"] = self.upstream_port
        if self.retry_after is not None:
            data["retry_after"] = self.retry_after
        if self.attempt:
            data["attempt"] = self.attempt
        if self.max_attempts:
            data["max_attempts"] = self.max_attempts
        if self.elapsed_seconds:
            data["elapsed_seconds"] = self.elapsed_seconds
        if self.details:
            data["details"] = self.details
        return data


# --------------------------------------------------------------------------- #
# Base exception
# --------------------------------------------------------------------------- #


class GatewayError(Exception):
    """Base exception for all gateway errors.

    Every gateway error carries an :class:`ErrorContext` for structured
    metadata and supports serialisation to dict / JSON-like responses.

    Args:
        message: Human-readable error description.
        context: Structured error context (auto-created if not provided).
        cause: The original exception that triggered this error.
    """

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.context = context or ErrorContext()
        self.cause = cause

    @property
    def is_retryable(self) -> bool:
        """Whether this error is safe to retry."""
        return self.context.retryable

    @property
    def category(self) -> ErrorCategory:
        """Shortcut to the error category."""
        return self.context.category

    @property
    def severity(self) -> ErrorSeverity:
        """Shortcut to the error severity."""
        return self.context.severity

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary matching Anthropic error format."""
        return {
            "type": "error",
            "error": {
                "type": self.context.category.value,
                "message": str(self),
            },
            "context": self.context.to_dict(),
        }

    def to_http_error(self) -> dict[str, Any]:
        """Format as an Anthropic-style JSON HTTP error body."""
        return {
            "type": "error",
            "error": {
                "type": self._error_type(),
                "message": str(self),
            },
        }

    def _error_type(self) -> str:
        """Map the error category to an Anthropic error type string."""
        mapping = {
            ErrorCategory.NETWORK: "api_error",
            ErrorCategory.AUTHENTICATION: "authentication_error",
            ErrorCategory.RATE_LIMIT: "rate_limit_error",
            ErrorCategory.TIMEOUT: "timeout_error",
            ErrorCategory.PROVIDER: "api_error",
            ErrorCategory.CONFIGURATION: "invalid_request_error",
            ErrorCategory.VALIDATION: "invalid_request_error",
            ErrorCategory.PROXY: "api_error",
            ErrorCategory.INTERNAL: "api_error",
        }
        return mapping.get(self.context.category, "api_error")


# --------------------------------------------------------------------------- #
# Network errors
# --------------------------------------------------------------------------- #


class NetworkError(GatewayError):
    """Error connecting to or communicating with an upstream host.

    These are generally retryable (DNS failures, connection refused,
    connection reset, etc.).
    """

    def __init__(
        self,
        message: str,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        cause: Optional[Exception] = None,
        context: Optional[ErrorContext] = None,
    ) -> None:
        ctx = context or ErrorContext(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            retryable=True,
            upstream_host=host,
            upstream_port=port,
        )
        super().__init__(message, context=ctx, cause=cause)


class ConnectionRefusedError_(NetworkError):
    """Upstream host actively refused the connection."""

    def __init__(
        self,
        host: str,
        port: int,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            f"Connection refused by {host}:{port}",
            host=host,
            port=port,
            cause=cause,
        )


class DNSResolutionError(NetworkError):
    """Failed to resolve the upstream hostname."""

    def __init__(
        self,
        host: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            f"DNS resolution failed for host '{host}'",
            host=host,
            cause=cause,
        )


class SSLError(NetworkError):
    """TLS / SSL handshake or certificate error."""

    def __init__(
        self,
        message: str,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        ctx = ErrorContext(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            retryable=False,  # SSL errors are typically not retryable
            upstream_host=host,
            upstream_port=port,
        )
        super().__init__(message, host=host, port=port, cause=cause, context=ctx)


# --------------------------------------------------------------------------- #
# Timeout errors
# --------------------------------------------------------------------------- #


class TimeoutError_(GatewayError):
    """An operation timed out.

    Timeout errors are retryable by default because they may be caused
    by transient network conditions or upstream load.
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: Optional[float] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        ctx = ErrorContext(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            retryable=True,
            upstream_host=host,
            upstream_port=port,
            details={"timeout_seconds": timeout_seconds} if timeout_seconds else {},
        )
        super().__init__(message, context=ctx, cause=cause)


class ConnectTimeoutError(TimeoutError_):
    """Timed out while establishing connection."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_seconds: float,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            f"Connection to {host}:{port} timed out after {timeout_seconds}s",
            timeout_seconds=timeout_seconds,
            host=host,
            port=port,
            cause=cause,
        )


class ReadTimeoutError(TimeoutError_):
    """Timed out while reading the upstream response."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_seconds: float,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            f"Read from {host}:{port} timed out after {timeout_seconds}s",
            timeout_seconds=timeout_seconds,
            host=host,
            port=port,
            cause=cause,
        )


# --------------------------------------------------------------------------- #
# Authentication errors
# --------------------------------------------------------------------------- #


class AuthenticationError(GatewayError):
    """Request failed due to invalid or missing credentials.

    Authentication errors are NOT retryable because the same credentials
    will fail again.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status_code: int = 401,
        cause: Optional[Exception] = None,
    ) -> None:
        ctx = ErrorContext(
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            retryable=False,
            status_code=status_code,
            provider=provider,
        )
        super().__init__(message, context=ctx, cause=cause)


class MissingAPIKeyError(AuthenticationError):
    """No API key was provided or could be discovered."""

    def __init__(self, provider: Optional[str] = None) -> None:
        sources = "environment variable, config file, or request header"
        super().__init__(
            f"No API key found. Provide one via {sources}.",
            provider=provider,
        )


class InvalidAPIKeyError(AuthenticationError):
    """The API key was rejected by the upstream provider."""

    def __init__(
        self,
        provider: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            "The API key was rejected by the provider.",
            provider=provider,
            cause=cause,
        )


# --------------------------------------------------------------------------- #
# Rate-limit errors
# --------------------------------------------------------------------------- #


class RateLimitError(GatewayError):
    """Request was rejected because the rate limit was exceeded.

    Rate-limit errors are retryable after a backoff period.  If the
    upstream returns a ``Retry-After`` header, it is stored in
    ``context.retry_after``.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: Optional[float] = None,
        provider: Optional[str] = None,
        status_code: int = 429,
        cause: Optional[Exception] = None,
    ) -> None:
        ctx = ErrorContext(
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.LOW,
            retryable=True,
            status_code=status_code,
            retry_after=retry_after,
            provider=provider,
        )
        super().__init__(message, context=ctx, cause=cause)


# --------------------------------------------------------------------------- #
# Provider / upstream errors
# --------------------------------------------------------------------------- #


class ProviderError(GatewayError):
    """The upstream provider returned an error response.

    Wraps errors returned by providers (4xx / 5xx). Whether the error
    is retryable depends on the status code:
    - 5xx: retryable (server-side issue)
    - 4xx: generally not retryable
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        response_body: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        retryable = status_code >= 500 or status_code == 429
        severity = (
            ErrorSeverity.LOW if status_code == 429
            else ErrorSeverity.HIGH if status_code >= 500
            else ErrorSeverity.MEDIUM
        )
        details: dict[str, Any] = {}
        if response_body:
            details["response_body"] = response_body

        ctx = ErrorContext(
            category=ErrorCategory.PROVIDER,
            severity=severity,
            retryable=retryable,
            status_code=status_code,
            provider=provider,
            model=model,
            details=details,
        )
        super().__init__(message, context=ctx, cause=cause)


class ProviderUnavailableError(ProviderError):
    """The upstream provider is temporarily unavailable (503)."""

    def __init__(
        self,
        provider: Optional[str] = None,
        retry_after: Optional[float] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            f"Provider{' ' + provider if provider else ''} is temporarily unavailable",
            status_code=503,
            provider=provider,
            cause=cause,
        )
        self.context.retry_after = retry_after


class OverloadedError(ProviderError):
    """The upstream provider is overloaded (529)."""

    def __init__(
        self,
        provider: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            f"Provider{' ' + provider if provider else ''} is overloaded",
            status_code=529,
            provider=provider,
            cause=cause,
        )


# --------------------------------------------------------------------------- #
# Proxy errors
# --------------------------------------------------------------------------- #


class ProxyError(GatewayError):
    """Error originating from the proxy layer itself."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        host: Optional[str] = None,
        port: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        ctx = ErrorContext(
            category=ErrorCategory.PROXY,
            severity=ErrorSeverity.MEDIUM,
            retryable=True,
            status_code=status_code,
            upstream_host=host,
            upstream_port=port,
        )
        super().__init__(message, context=ctx, cause=cause)


class BadGatewayError(ProxyError):
    """Invalid response from the upstream server (502)."""

    def __init__(
        self,
        message: str = "Bad gateway",
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message, status_code=502, host=host, port=port, cause=cause
        )


class GatewayTimeoutError(ProxyError):
    """Upstream did not respond within the configured timeout (504)."""

    def __init__(
        self,
        timeout_seconds: float,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            f"Gateway timed out after {timeout_seconds}s",
            status_code=504,
            host=host,
            port=port,
            cause=cause,
        )


class HostNotAllowedError(ProxyError):
    """The target host is not on the allow-list."""

    def __init__(self, host: str) -> None:
        super().__init__(
            f"Host '{host}' is not allowed",
            status_code=403,
        )
        self.context.retryable = False


# --------------------------------------------------------------------------- #
# Configuration errors (upgraded from config module)
# --------------------------------------------------------------------------- #


class ConfigurationError(GatewayError):
    """Error loading or processing configuration.

    Configuration errors are not retryable because they require
    human intervention to fix.
    """

    def __init__(
        self,
        message: str,
        *,
        config_path: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        details: dict[str, Any] = {}
        if config_path:
            details["config_path"] = config_path
        ctx = ErrorContext(
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            retryable=False,
            details=details,
        )
        super().__init__(message, context=ctx, cause=cause)


class ConfigValidationError_(GatewayError):
    """Configuration validation failed.

    Carries a list of individual validation error messages.
    """

    def __init__(
        self,
        errors: list[str],
        *,
        config_path: Optional[str] = None,
    ) -> None:
        self.errors = errors
        details: dict[str, Any] = {"validation_errors": errors}
        if config_path:
            details["config_path"] = config_path
        ctx = ErrorContext(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            retryable=False,
            details=details,
        )
        super().__init__(
            f"Configuration validation failed: {'; '.join(errors)}",
            context=ctx,
        )


# --------------------------------------------------------------------------- #
# Retry-specific errors
# --------------------------------------------------------------------------- #


class RetryExhaustedError(GatewayError):
    """All retry attempts have been exhausted.

    Wraps the *last* exception that caused the final retry failure,
    along with metadata about the retry history.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 0,
        total_elapsed: float = 0.0,
        last_error: Optional[Exception] = None,
        errors: Optional[list[Exception]] = None,
    ) -> None:
        self.attempts = attempts
        self.total_elapsed = total_elapsed
        self.last_error = last_error
        self.all_errors = errors or []
        ctx = ErrorContext(
            category=ErrorCategory.INTERNAL,
            severity=ErrorSeverity.HIGH,
            retryable=False,
            attempt=attempts,
            elapsed_seconds=total_elapsed,
        )
        super().__init__(message, context=ctx, cause=last_error)


class CircuitOpenError(GatewayError):
    """The circuit breaker is open; requests are being short-circuited.

    Not retryable until the circuit transitions to half-open.
    """

    def __init__(
        self,
        service: str,
        *,
        reset_timeout: Optional[float] = None,
    ) -> None:
        details: dict[str, Any] = {"service": service}
        if reset_timeout is not None:
            details["reset_timeout"] = reset_timeout
        ctx = ErrorContext(
            category=ErrorCategory.INTERNAL,
            severity=ErrorSeverity.HIGH,
            retryable=False,
            details=details,
        )
        super().__init__(
            f"Circuit breaker open for '{service}' — requests are being rejected",
            context=ctx,
        )


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def classify_http_status(status_code: int) -> ErrorCategory:
    """Map an HTTP status code to an error category.

    Args:
        status_code: The HTTP response status code.

    Returns:
        The corresponding ErrorCategory.
    """
    if status_code == 401 or status_code == 403:
        return ErrorCategory.AUTHENTICATION
    if status_code == 429:
        return ErrorCategory.RATE_LIMIT
    if status_code == 408 or status_code == 504:
        return ErrorCategory.TIMEOUT
    if 400 <= status_code < 500:
        return ErrorCategory.VALIDATION
    if status_code >= 500:
        return ErrorCategory.PROVIDER
    return ErrorCategory.INTERNAL


def is_retryable_status(status_code: int) -> bool:
    """Determine if an HTTP status code indicates a retryable error.

    Retryable status codes:
    - 408 Request Timeout
    - 429 Too Many Requests
    - 500 Internal Server Error
    - 502 Bad Gateway
    - 503 Service Unavailable
    - 504 Gateway Timeout
    - 529 Overloaded

    Args:
        status_code: The HTTP response status code.

    Returns:
        True if the error is retryable.
    """
    return status_code in {408, 429, 500, 502, 503, 504, 529}


def is_retryable_exception(exc: Exception) -> bool:
    """Determine if an exception is retryable.

    Checks for:
    - GatewayError subclasses with ``is_retryable`` flag
    - Standard library timeout / connection errors

    Args:
        exc: The exception to check.

    Returns:
        True if the exception is retryable.
    """
    if isinstance(exc, GatewayError):
        return exc.is_retryable

    # Standard library exceptions that are retryable
    import socket
    import ssl as _ssl

    retryable_types = (
        ConnectionError,
        ConnectionResetError,
        ConnectionAbortedError,
        ConnectionRefusedError,
        BrokenPipeError,
        TimeoutError,
        socket.timeout,
        OSError,
    )
    # SSL errors are NOT retryable by default
    if isinstance(exc, _ssl.SSLError):
        return False

    return isinstance(exc, retryable_types)


def exception_from_status(
    status_code: int,
    message: str = "",
    *,
    provider: Optional[str] = None,
    response_body: Optional[str] = None,
    retry_after: Optional[float] = None,
) -> GatewayError:
    """Create an appropriate GatewayError subclass from an HTTP status code.

    Args:
        status_code: The HTTP response status code.
        message: Error message (a default is used if empty).
        provider: Optional provider name for context.
        response_body: Optional response body text.
        retry_after: Optional seconds to wait before retrying.

    Returns:
        A GatewayError subclass appropriate for the status code.
    """
    if not message:
        message = f"HTTP {status_code} error"

    if status_code == 401:
        return AuthenticationError(message, provider=provider, status_code=401)
    if status_code == 403:
        return AuthenticationError(message, provider=provider, status_code=403)
    if status_code == 429:
        return RateLimitError(message, retry_after=retry_after, provider=provider)
    if status_code == 408:
        return TimeoutError_(message)
    if status_code == 502:
        return BadGatewayError(message)
    if status_code == 503:
        return ProviderUnavailableError(provider=provider, retry_after=retry_after)
    if status_code == 504:
        return GatewayTimeoutError(0, host=None, port=None)
    if status_code == 529:
        return OverloadedError(provider=provider)

    return ProviderError(
        message,
        status_code=status_code,
        provider=provider,
        response_body=response_body,
    )
