"""Tests for the error hierarchy module."""

import time

import pytest

from src.errors import (
    AuthenticationError,
    BadGatewayError,
    CircuitOpenError,
    ConfigurationError,
    ConfigValidationError_,
    ConnectionRefusedError_,
    ConnectTimeoutError,
    DNSResolutionError,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    GatewayError,
    GatewayTimeoutError,
    HostNotAllowedError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    NetworkError,
    OverloadedError,
    ProviderError,
    ProviderUnavailableError,
    ProxyError,
    RateLimitError,
    ReadTimeoutError,
    RetryExhaustedError,
    SSLError,
    TimeoutError_,
    classify_http_status,
    exception_from_status,
    is_retryable_exception,
    is_retryable_status,
)


# --------------------------------------------------------------------------- #
# ErrorContext tests
# --------------------------------------------------------------------------- #


class TestErrorContext:
    """Tests for ErrorContext dataclass."""

    def test_default_context(self):
        """Default context has sensible defaults."""
        ctx = ErrorContext()
        assert ctx.category == ErrorCategory.INTERNAL
        assert ctx.severity == ErrorSeverity.MEDIUM
        assert ctx.retryable is False
        assert ctx.status_code is None
        assert ctx.provider is None
        assert ctx.timestamp > 0

    def test_to_dict(self):
        """to_dict() produces a serializable dictionary."""
        ctx = ErrorContext(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            retryable=True,
            status_code=502,
            provider="anthropic",
            upstream_host="api.anthropic.com",
            upstream_port=443,
        )
        d = ctx.to_dict()
        assert d["category"] == "network"
        assert d["severity"] == "high"
        assert d["retryable"] is True
        assert d["status_code"] == 502
        assert d["provider"] == "anthropic"
        assert d["upstream_host"] == "api.anthropic.com"
        assert d["upstream_port"] == 443

    def test_to_dict_omits_none_fields(self):
        """to_dict() omits fields that are None or zero."""
        ctx = ErrorContext()
        d = ctx.to_dict()
        assert "status_code" not in d
        assert "provider" not in d
        assert "request_id" not in d
        assert "upstream_host" not in d
        assert "upstream_port" not in d
        assert "retry_after" not in d
        assert "attempt" not in d
        assert "max_attempts" not in d

    def test_to_dict_includes_details(self):
        """to_dict() includes custom details when present."""
        ctx = ErrorContext(details={"key": "value"})
        d = ctx.to_dict()
        assert d["details"] == {"key": "value"}


# --------------------------------------------------------------------------- #
# GatewayError tests
# --------------------------------------------------------------------------- #


class TestGatewayError:
    """Tests for the base GatewayError."""

    def test_basic_error(self):
        """GatewayError carries message and default context."""
        err = GatewayError("something failed")
        assert str(err) == "something failed"
        assert err.context is not None
        assert err.cause is None

    def test_is_retryable_default_false(self):
        """GatewayError is not retryable by default."""
        err = GatewayError("fail")
        assert err.is_retryable is False

    def test_with_custom_context(self):
        """GatewayError accepts a custom ErrorContext."""
        ctx = ErrorContext(category=ErrorCategory.NETWORK, retryable=True)
        err = GatewayError("net fail", context=ctx)
        assert err.is_retryable is True
        assert err.category == ErrorCategory.NETWORK

    def test_to_dict(self):
        """to_dict() returns Anthropic-format error."""
        err = GatewayError("test error")
        d = err.to_dict()
        assert d["type"] == "error"
        assert "error" in d
        assert d["error"]["message"] == "test error"
        assert "context" in d

    def test_to_http_error(self):
        """to_http_error() returns minimal Anthropic error body."""
        err = GatewayError("test error")
        d = err.to_http_error()
        assert d["type"] == "error"
        assert d["error"]["message"] == "test error"
        assert "context" not in d  # http_error is minimal

    def test_with_cause(self):
        """GatewayError wraps an original cause exception."""
        cause = ValueError("original")
        err = GatewayError("wrapped", cause=cause)
        assert err.cause is cause


# --------------------------------------------------------------------------- #
# Network error tests
# --------------------------------------------------------------------------- #


class TestNetworkErrors:
    """Tests for network error subclasses."""

    def test_network_error_is_retryable(self):
        """NetworkError is retryable by default."""
        err = NetworkError("net issue")
        assert err.is_retryable is True
        assert err.category == ErrorCategory.NETWORK

    def test_connection_refused(self):
        """ConnectionRefusedError_ captures host and port."""
        err = ConnectionRefusedError_("localhost", 8080)
        assert "localhost:8080" in str(err)
        assert err.is_retryable is True
        assert err.context.upstream_host == "localhost"
        assert err.context.upstream_port == 8080

    def test_dns_resolution_error(self):
        """DNSResolutionError captures hostname."""
        err = DNSResolutionError("bad.host.example")
        assert "bad.host.example" in str(err)
        assert err.is_retryable is True

    def test_ssl_error_not_retryable(self):
        """SSLError is NOT retryable by default."""
        err = SSLError("cert expired", host="api.example.com")
        assert err.is_retryable is False
        assert err.category == ErrorCategory.NETWORK
        assert err.severity == ErrorSeverity.HIGH


# --------------------------------------------------------------------------- #
# Timeout error tests
# --------------------------------------------------------------------------- #


class TestTimeoutErrors:
    """Tests for timeout error subclasses."""

    def test_timeout_is_retryable(self):
        """TimeoutError_ is retryable."""
        err = TimeoutError_("timed out")
        assert err.is_retryable is True
        assert err.category == ErrorCategory.TIMEOUT

    def test_connect_timeout(self):
        """ConnectTimeoutError captures details."""
        err = ConnectTimeoutError("api.example.com", 443, 30.0)
        assert "30" in str(err)
        assert err.is_retryable is True
        assert err.context.upstream_host == "api.example.com"

    def test_read_timeout(self):
        """ReadTimeoutError captures details."""
        err = ReadTimeoutError("api.example.com", 443, 60.0)
        assert "60" in str(err)
        assert err.is_retryable is True


# --------------------------------------------------------------------------- #
# Auth error tests
# --------------------------------------------------------------------------- #


class TestAuthenticationErrors:
    """Tests for authentication error subclasses."""

    def test_auth_error_not_retryable(self):
        """AuthenticationError is NOT retryable."""
        err = AuthenticationError("bad creds")
        assert err.is_retryable is False
        assert err.category == ErrorCategory.AUTHENTICATION
        assert err.context.status_code == 401

    def test_missing_api_key(self):
        """MissingAPIKeyError describes the issue."""
        err = MissingAPIKeyError(provider="openai")
        assert "API key" in str(err)
        assert err.is_retryable is False

    def test_invalid_api_key(self):
        """InvalidAPIKeyError describes the issue."""
        err = InvalidAPIKeyError(provider="anthropic")
        assert "rejected" in str(err)
        assert err.is_retryable is False


# --------------------------------------------------------------------------- #
# Rate limit error tests
# --------------------------------------------------------------------------- #


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_is_retryable(self):
        """RateLimitError is retryable."""
        err = RateLimitError()
        assert err.is_retryable is True
        assert err.category == ErrorCategory.RATE_LIMIT
        assert err.context.status_code == 429

    def test_retry_after(self):
        """RateLimitError carries retry_after."""
        err = RateLimitError(retry_after=30.0)
        assert err.context.retry_after == 30.0

    def test_severity_is_low(self):
        """Rate limits are expected and low severity."""
        err = RateLimitError()
        assert err.severity == ErrorSeverity.LOW


# --------------------------------------------------------------------------- #
# Provider error tests
# --------------------------------------------------------------------------- #


class TestProviderErrors:
    """Tests for provider error subclasses."""

    def test_5xx_is_retryable(self):
        """Provider errors with 5xx status are retryable."""
        err = ProviderError("server error", status_code=500)
        assert err.is_retryable is True
        assert err.severity == ErrorSeverity.HIGH

    def test_4xx_not_retryable(self):
        """Provider errors with 4xx status are NOT retryable."""
        err = ProviderError("bad request", status_code=400)
        assert err.is_retryable is False
        assert err.severity == ErrorSeverity.MEDIUM

    def test_429_is_retryable(self):
        """Provider error with 429 is retryable."""
        err = ProviderError("rate limited", status_code=429)
        assert err.is_retryable is True
        assert err.severity == ErrorSeverity.LOW

    def test_provider_unavailable(self):
        """ProviderUnavailableError is a 503."""
        err = ProviderUnavailableError(provider="anthropic", retry_after=10.0)
        assert err.context.status_code == 503
        assert err.is_retryable is True
        assert err.context.retry_after == 10.0

    def test_overloaded_error(self):
        """OverloadedError is a 529."""
        err = OverloadedError(provider="anthropic")
        assert err.context.status_code == 529
        assert err.is_retryable is True

    def test_response_body_in_details(self):
        """ProviderError captures response body in details."""
        err = ProviderError("err", status_code=500, response_body='{"error":"oops"}')
        assert err.context.details["response_body"] == '{"error":"oops"}'


# --------------------------------------------------------------------------- #
# Proxy error tests
# --------------------------------------------------------------------------- #


class TestProxyErrors:
    """Tests for proxy error subclasses."""

    def test_proxy_error_retryable(self):
        """ProxyError is retryable by default."""
        err = ProxyError("proxy fail")
        assert err.is_retryable is True
        assert err.category == ErrorCategory.PROXY
        assert err.context.status_code == 502

    def test_bad_gateway(self):
        """BadGatewayError is a 502."""
        err = BadGatewayError(host="upstream.com", port=443)
        assert err.context.status_code == 502
        assert err.is_retryable is True

    def test_gateway_timeout(self):
        """GatewayTimeoutError is a 504."""
        err = GatewayTimeoutError(30.0, host="upstream.com")
        assert err.context.status_code == 504
        assert "30" in str(err)

    def test_host_not_allowed_not_retryable(self):
        """HostNotAllowedError is NOT retryable."""
        err = HostNotAllowedError("evil.com")
        assert err.is_retryable is False
        assert err.context.status_code == 403


# --------------------------------------------------------------------------- #
# Config error tests
# --------------------------------------------------------------------------- #


class TestConfigurationErrors:
    """Tests for configuration error subclasses."""

    def test_config_error_not_retryable(self):
        """ConfigurationError is not retryable."""
        err = ConfigurationError("bad config")
        assert err.is_retryable is False
        assert err.category == ErrorCategory.CONFIGURATION

    def test_config_validation_error(self):
        """ConfigValidationError_ carries error list."""
        err = ConfigValidationError_(["field1 missing", "field2 invalid"])
        assert err.is_retryable is False
        assert len(err.errors) == 2
        assert "field1" in str(err)


# --------------------------------------------------------------------------- #
# Retry-specific error tests
# --------------------------------------------------------------------------- #


class TestRetryErrors:
    """Tests for retry-specific errors."""

    def test_retry_exhausted(self):
        """RetryExhaustedError carries attempt metadata."""
        last = ValueError("boom")
        err = RetryExhaustedError(
            "all failed",
            attempts=3,
            total_elapsed=5.5,
            last_error=last,
            errors=[ValueError("a"), ValueError("b"), last],
        )
        assert err.attempts == 3
        assert err.total_elapsed == 5.5
        assert err.last_error is last
        assert len(err.all_errors) == 3
        assert err.is_retryable is False

    def test_circuit_open_error(self):
        """CircuitOpenError is not retryable."""
        err = CircuitOpenError("my-service", reset_timeout=30.0)
        assert err.is_retryable is False
        assert "my-service" in str(err)
        assert err.context.details["reset_timeout"] == 30.0


# --------------------------------------------------------------------------- #
# Utility function tests
# --------------------------------------------------------------------------- #


class TestClassifyHTTPStatus:
    """Tests for classify_http_status()."""

    def test_401_is_auth(self):
        assert classify_http_status(401) == ErrorCategory.AUTHENTICATION

    def test_403_is_auth(self):
        assert classify_http_status(403) == ErrorCategory.AUTHENTICATION

    def test_429_is_rate_limit(self):
        assert classify_http_status(429) == ErrorCategory.RATE_LIMIT

    def test_408_is_timeout(self):
        assert classify_http_status(408) == ErrorCategory.TIMEOUT

    def test_504_is_timeout(self):
        assert classify_http_status(504) == ErrorCategory.TIMEOUT

    def test_400_is_validation(self):
        assert classify_http_status(400) == ErrorCategory.VALIDATION

    def test_422_is_validation(self):
        assert classify_http_status(422) == ErrorCategory.VALIDATION

    def test_500_is_provider(self):
        assert classify_http_status(500) == ErrorCategory.PROVIDER

    def test_503_is_provider(self):
        assert classify_http_status(503) == ErrorCategory.PROVIDER


class TestIsRetryableStatus:
    """Tests for is_retryable_status()."""

    def test_retryable_codes(self):
        """Known retryable status codes return True."""
        for code in [408, 429, 500, 502, 503, 504, 529]:
            assert is_retryable_status(code) is True, f"{code} should be retryable"

    def test_non_retryable_codes(self):
        """Non-retryable status codes return False."""
        for code in [200, 201, 400, 401, 403, 404, 422]:
            assert is_retryable_status(code) is False, f"{code} should not be retryable"


class TestIsRetryableException:
    """Tests for is_retryable_exception()."""

    def test_retryable_gateway_error(self):
        """GatewayError with retryable=True returns True."""
        err = NetworkError("net fail")
        assert is_retryable_exception(err) is True

    def test_non_retryable_gateway_error(self):
        """GatewayError with retryable=False returns False."""
        err = AuthenticationError("bad key")
        assert is_retryable_exception(err) is False

    def test_connection_error_retryable(self):
        """Standard ConnectionError is retryable."""
        assert is_retryable_exception(ConnectionError("reset")) is True

    def test_timeout_error_retryable(self):
        """Standard TimeoutError is retryable."""
        assert is_retryable_exception(TimeoutError("timeout")) is True

    def test_ssl_error_not_retryable(self):
        """Standard ssl.SSLError is NOT retryable."""
        import ssl
        assert is_retryable_exception(ssl.SSLError("cert fail")) is False

    def test_value_error_not_retryable(self):
        """ValueError is not retryable."""
        assert is_retryable_exception(ValueError("bad value")) is False


class TestExceptionFromStatus:
    """Tests for exception_from_status()."""

    def test_401_returns_auth_error(self):
        err = exception_from_status(401, "unauthorized")
        assert isinstance(err, AuthenticationError)

    def test_403_returns_auth_error(self):
        err = exception_from_status(403, "forbidden")
        assert isinstance(err, AuthenticationError)

    def test_429_returns_rate_limit(self):
        err = exception_from_status(429, "too many", retry_after=30.0)
        assert isinstance(err, RateLimitError)
        assert err.context.retry_after == 30.0

    def test_502_returns_bad_gateway(self):
        err = exception_from_status(502, "bad gw")
        assert isinstance(err, BadGatewayError)

    def test_503_returns_unavailable(self):
        err = exception_from_status(503, provider="anthropic")
        assert isinstance(err, ProviderUnavailableError)

    def test_529_returns_overloaded(self):
        err = exception_from_status(529, provider="anthropic")
        assert isinstance(err, OverloadedError)

    def test_500_returns_provider_error(self):
        err = exception_from_status(500, "server error")
        assert isinstance(err, ProviderError)
        assert err.context.status_code == 500

    def test_default_message(self):
        """Default message is generated if empty."""
        err = exception_from_status(418)
        assert "418" in str(err)


# --------------------------------------------------------------------------- #
# Error type mapping tests
# --------------------------------------------------------------------------- #


class TestErrorTypeMapping:
    """Tests for _error_type() mapping to Anthropic error types."""

    def test_network_maps_to_api_error(self):
        err = NetworkError("net fail")
        assert err._error_type() == "api_error"

    def test_auth_maps_to_authentication_error(self):
        err = AuthenticationError("bad key")
        assert err._error_type() == "authentication_error"

    def test_rate_limit_maps_correctly(self):
        err = RateLimitError()
        assert err._error_type() == "rate_limit_error"

    def test_timeout_maps_correctly(self):
        err = TimeoutError_("timeout")
        assert err._error_type() == "timeout_error"

    def test_validation_maps_to_invalid_request(self):
        err = ConfigValidationError_(["err"])
        assert err._error_type() == "invalid_request_error"
