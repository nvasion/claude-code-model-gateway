"""Request interception middleware for the model gateway.

Provides a composable interceptor chain that processes incoming API
requests before they reach the upstream provider. Each interceptor can
inspect, modify, reject, or re-route a request. Interceptors execute in
order and each can short-circuit the chain.

Built-in interceptors:
- **ModelRoutingInterceptor** -- Extracts the ``model`` field from the
  request body and resolves the target provider via the Router.
- **HeaderRoutingInterceptor** -- Routes based on a custom header
  (e.g., ``X-Provider``).
- **RateLimitInterceptor** -- Enforces per-client request rate limits.
- **AuthenticationInterceptor** -- Validates API keys and injects
  provider-specific authentication headers.
- **RequestTransformInterceptor** -- Applies request body
  transformations (e.g., model alias expansion).

Typical usage:

    from src.interceptor import (
        InterceptorChain,
        ModelRoutingInterceptor,
        HeaderRoutingInterceptor,
        RateLimitInterceptor,
    )

    chain = InterceptorChain()
    chain.add(HeaderRoutingInterceptor(header="X-Provider"))
    chain.add(ModelRoutingInterceptor(router=my_router))
    chain.add(RateLimitInterceptor(max_rpm=100))

    result = chain.process(request_context)
    if result.action == InterceptAction.FORWARD:
        # send to result.provider
        ...
"""

from __future__ import annotations

import abc
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from src.errors import (
    AuthenticationError,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    GatewayError,
    MissingAPIKeyError,
    RateLimitError,
)
from src.logging_config import get_logger
from src.models import GatewayConfig, ProviderConfig
from src.router import (
    NoRouteError,
    RequestContext,
    RouteMatch,
    Router,
    RoutingStrategy,
    extract_model_from_body,
)

logger = get_logger("interceptor")


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class InterceptAction(str, Enum):
    """Action to take on a request after interception."""

    FORWARD = "forward"      # Continue to the resolved provider
    REJECT = "reject"        # Reject the request with an error
    MODIFY = "modify"        # Modify the request and continue
    REDIRECT = "redirect"    # Redirect to a different provider
    SKIP = "skip"            # Skip this interceptor, continue chain


# --------------------------------------------------------------------------- #
# Intercept result
# --------------------------------------------------------------------------- #


@dataclass
class InterceptResult:
    """Result of processing a request through an interceptor.

    Attributes:
        action: What to do with the request.
        provider_name: Target provider name (for FORWARD / REDIRECT).
        provider_config: Full provider configuration (if resolved).
        route_match: Full routing match details (if resolved by router).
        status_code: HTTP status code for REJECT actions.
        error_message: Error message for REJECT actions.
        error_body: Full error response body for REJECT actions.
        modified_body: Modified request body (for MODIFY actions).
        modified_headers: Modified headers to merge into the request.
        interceptor_name: Name of the interceptor that produced this result.
        metadata: Additional metadata from the interceptor.
    """

    action: InterceptAction = InterceptAction.SKIP
    provider_name: Optional[str] = None
    provider_config: Optional[ProviderConfig] = None
    route_match: Optional[RouteMatch] = None
    status_code: int = 200
    error_message: str = ""
    error_body: Optional[dict[str, Any]] = None
    modified_body: Optional[dict[str, Any]] = None
    modified_headers: dict[str, str] = field(default_factory=dict)
    interceptor_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Whether this result stops further interceptor processing."""
        return self.action in (InterceptAction.REJECT, InterceptAction.FORWARD)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        data: dict[str, Any] = {
            "action": self.action.value,
            "interceptor_name": self.interceptor_name,
        }
        if self.provider_name:
            data["provider_name"] = self.provider_name
        if self.route_match:
            data["route_match"] = self.route_match.to_dict()
        if self.action == InterceptAction.REJECT:
            data["status_code"] = self.status_code
            data["error_message"] = self.error_message
        if self.modified_headers:
            data["modified_headers"] = list(self.modified_headers.keys())
        if self.metadata:
            data["metadata"] = self.metadata
        return data


# --------------------------------------------------------------------------- #
# Base interceptor
# --------------------------------------------------------------------------- #


class RequestInterceptor(abc.ABC):
    """Abstract base class for request interceptors.

    Subclasses must implement :meth:`intercept` which receives the
    current request context and returns an :class:`InterceptResult`.

    Attributes:
        name: Human-readable name for this interceptor.
        enabled: Whether this interceptor is active.
        order: Execution order (lower values execute first).
    """

    def __init__(
        self,
        name: str = "",
        enabled: bool = True,
        order: int = 0,
    ) -> None:
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self.order = order

    @abc.abstractmethod
    def intercept(
        self,
        ctx: RequestContext,
        current_result: Optional[InterceptResult] = None,
    ) -> InterceptResult:
        """Process an incoming request.

        Args:
            ctx: The incoming request context.
            current_result: Result from the previous interceptor in the
                chain, if any. Allows interceptors to build on prior
                decisions.

        Returns:
            An InterceptResult describing what to do with the request.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, order={self.order})"


# --------------------------------------------------------------------------- #
# Model routing interceptor
# --------------------------------------------------------------------------- #


class ModelRoutingInterceptor(RequestInterceptor):
    """Routes requests based on the model field in the request body.

    Extracts the ``model`` field from the JSON body and uses the
    :class:`Router` to resolve the target provider. If the context
    already has a model set it uses that directly.

    Args:
        router: The Router instance to use for resolution.
        name: Interceptor name.
        order: Execution order.
    """

    def __init__(
        self,
        router: Router,
        name: str = "model_routing",
        order: int = 50,
    ) -> None:
        super().__init__(name=name, order=order)
        self.router = router

    def intercept(
        self,
        ctx: RequestContext,
        current_result: Optional[InterceptResult] = None,
    ) -> InterceptResult:
        """Route based on model name.

        If the model is not present in the context, extracts it from
        the request body.
        """
        # Extract model if not already set
        if not ctx.model and ctx.body:
            ctx.model = extract_model_from_body(ctx.body)

        if not ctx.model:
            logger.debug("ModelRoutingInterceptor: no model in request, skipping")
            return InterceptResult(
                action=InterceptAction.SKIP,
                interceptor_name=self.name,
            )

        try:
            match = self.router.resolve(ctx)
            logger.info(
                "ModelRoutingInterceptor: %s -> %s (%.2fms)",
                ctx.model,
                match.provider_name,
                match.resolution_time_ms,
            )
            return InterceptResult(
                action=InterceptAction.FORWARD,
                provider_name=match.provider_name,
                provider_config=match.provider_config,
                route_match=match,
                interceptor_name=self.name,
                metadata={"model": ctx.model},
            )
        except NoRouteError as exc:
            logger.warning("ModelRoutingInterceptor: %s", exc)
            return InterceptResult(
                action=InterceptAction.REJECT,
                status_code=404,
                error_message=str(exc),
                error_body={
                    "type": "error",
                    "error": {
                        "type": "not_found",
                        "message": str(exc),
                    },
                },
                interceptor_name=self.name,
            )
        except GatewayError as exc:
            logger.error("ModelRoutingInterceptor: routing error: %s", exc)
            return InterceptResult(
                action=InterceptAction.REJECT,
                status_code=exc.context.status_code or 500,
                error_message=str(exc),
                interceptor_name=self.name,
            )


# --------------------------------------------------------------------------- #
# Header routing interceptor
# --------------------------------------------------------------------------- #


class HeaderRoutingInterceptor(RequestInterceptor):
    """Routes requests based on a custom HTTP header.

    If the configured header is present, its value is treated as the
    target provider name. This allows clients to explicitly choose
    which provider handles their request.

    Args:
        header: Header name to check (case-insensitive).
        config: Gateway config to look up provider definitions.
        name: Interceptor name.
        order: Execution order (default 10, runs before model routing).
    """

    def __init__(
        self,
        header: str = "x-provider",
        config: Optional[GatewayConfig] = None,
        name: str = "header_routing",
        order: int = 10,
    ) -> None:
        super().__init__(name=name, order=order)
        self.header = header.lower()
        self.config = config

    def intercept(
        self,
        ctx: RequestContext,
        current_result: Optional[InterceptResult] = None,
    ) -> InterceptResult:
        """Route based on header value."""
        provider_name = ctx.get_header(self.header)
        if not provider_name:
            return InterceptResult(
                action=InterceptAction.SKIP,
                interceptor_name=self.name,
            )

        logger.info(
            "HeaderRoutingInterceptor: header '%s' = '%s'",
            self.header,
            provider_name,
        )

        if self.config:
            provider = self.config.get_provider(provider_name)
            if provider is None:
                return InterceptResult(
                    action=InterceptAction.REJECT,
                    status_code=400,
                    error_message=f"Unknown provider '{provider_name}'",
                    error_body={
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": (
                                f"Provider '{provider_name}' specified in "
                                f"header '{self.header}' is not configured"
                            ),
                        },
                    },
                    interceptor_name=self.name,
                )
            if not provider.enabled:
                return InterceptResult(
                    action=InterceptAction.REJECT,
                    status_code=400,
                    error_message=f"Provider '{provider_name}' is disabled",
                    error_body={
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": f"Provider '{provider_name}' is disabled",
                        },
                    },
                    interceptor_name=self.name,
                )

            return InterceptResult(
                action=InterceptAction.FORWARD,
                provider_name=provider_name,
                provider_config=provider,
                interceptor_name=self.name,
                metadata={"header": self.header},
            )

        # No config to validate against -- trust the header
        return InterceptResult(
            action=InterceptAction.REDIRECT,
            provider_name=provider_name,
            interceptor_name=self.name,
            metadata={"header": self.header},
        )


# --------------------------------------------------------------------------- #
# Rate limit interceptor
# --------------------------------------------------------------------------- #


class RateLimitInterceptor(RequestInterceptor):
    """Enforces per-client request rate limits.

    Uses a simple sliding-window counter per client IP. Requests
    exceeding the limit receive a 429 response.

    Args:
        max_requests_per_minute: Maximum requests per minute per client.
        name: Interceptor name.
        order: Execution order (default 5, runs early).
    """

    def __init__(
        self,
        max_requests_per_minute: int = 60,
        name: str = "rate_limit",
        order: int = 5,
    ) -> None:
        super().__init__(name=name, order=order)
        self.max_rpm = max_requests_per_minute
        self._window_seconds = 60.0
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def intercept(
        self,
        ctx: RequestContext,
        current_result: Optional[InterceptResult] = None,
    ) -> InterceptResult:
        """Check rate limit for the client."""
        client_key = ctx.client_ip or "unknown"
        now = time.time()
        window_start = now - self._window_seconds

        with self._lock:
            # Prune expired entries
            timestamps = self._requests[client_key]
            timestamps[:] = [t for t in timestamps if t > window_start]
            current_count = len(timestamps)

            if current_count >= self.max_rpm:
                retry_after = timestamps[0] - window_start if timestamps else 1.0
                logger.warning(
                    "RateLimitInterceptor: client '%s' exceeded limit "
                    "(%d/%d rpm)",
                    client_key,
                    current_count,
                    self.max_rpm,
                )
                return InterceptResult(
                    action=InterceptAction.REJECT,
                    status_code=429,
                    error_message="Rate limit exceeded",
                    error_body={
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": (
                                f"Rate limit exceeded: {current_count}/{self.max_rpm} "
                                f"requests per minute"
                            ),
                        },
                    },
                    modified_headers={"Retry-After": str(int(retry_after) + 1)},
                    interceptor_name=self.name,
                    metadata={
                        "client": client_key,
                        "current_count": current_count,
                        "limit": self.max_rpm,
                    },
                )

            # Record this request
            timestamps.append(now)

        return InterceptResult(
            action=InterceptAction.SKIP,
            interceptor_name=self.name,
            metadata={
                "client": client_key,
                "current_count": current_count + 1,
                "limit": self.max_rpm,
            },
        )

    def get_client_usage(self, client_ip: str) -> dict[str, Any]:
        """Return current rate-limit status for a client.

        Args:
            client_ip: Client IP address.

        Returns:
            Dictionary with current count, limit, and remaining.
        """
        now = time.time()
        window_start = now - self._window_seconds

        with self._lock:
            timestamps = self._requests.get(client_ip, [])
            current = len([t for t in timestamps if t > window_start])

        return {
            "client": client_ip,
            "current_count": current,
            "limit": self.max_rpm,
            "remaining": max(0, self.max_rpm - current),
        }

    def reset(self, client_ip: Optional[str] = None) -> None:
        """Reset rate-limit counters.

        Args:
            client_ip: If provided, reset only this client. Otherwise
                reset all clients.
        """
        with self._lock:
            if client_ip:
                self._requests.pop(client_ip, None)
            else:
                self._requests.clear()


# --------------------------------------------------------------------------- #
# Authentication interceptor
# --------------------------------------------------------------------------- #


class AuthenticationInterceptor(RequestInterceptor):
    """Validates and injects provider-specific authentication headers.

    Checks that the target provider's API key is available (either from
    the environment or request headers) and prepares the authentication
    headers for the upstream request.

    Args:
        config: Gateway configuration.
        require_api_key: Whether to reject requests without an API key.
        name: Interceptor name.
        order: Execution order (default 20, after routing).
    """

    def __init__(
        self,
        config: Optional[GatewayConfig] = None,
        require_api_key: bool = True,
        name: str = "authentication",
        order: int = 20,
    ) -> None:
        super().__init__(name=name, order=order)
        self.config = config
        self.require_api_key = require_api_key

    def intercept(
        self,
        ctx: RequestContext,
        current_result: Optional[InterceptResult] = None,
    ) -> InterceptResult:
        """Validate authentication for the target provider."""
        # Need a provider to authenticate against
        if current_result is None or current_result.provider_config is None:
            return InterceptResult(
                action=InterceptAction.SKIP,
                interceptor_name=self.name,
            )

        provider = current_result.provider_config
        headers: dict[str, str] = {}

        # Check for API key
        api_key = self._resolve_api_key(provider, ctx)

        if not api_key and self.require_api_key and provider.api_key_env_var:
            logger.warning(
                "AuthenticationInterceptor: no API key for provider '%s' "
                "(env var: %s)",
                provider.name,
                provider.api_key_env_var,
            )
            return InterceptResult(
                action=InterceptAction.REJECT,
                status_code=401,
                error_message=f"No API key configured for provider '{provider.name}'",
                error_body={
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": (
                            f"No API key found for provider '{provider.name}'. "
                            f"Set the {provider.api_key_env_var} environment variable."
                        ),
                    },
                },
                interceptor_name=self.name,
            )

        # Build auth headers
        if api_key:
            from src.models import AuthType

            if provider.auth_type == AuthType.BEARER_TOKEN:
                headers["Authorization"] = f"Bearer {api_key}"
            elif provider.auth_type == AuthType.API_KEY:
                headers["x-api-key"] = api_key

        # Add any provider-specific headers
        if provider.headers:
            headers.update(provider.headers)

        return InterceptResult(
            action=InterceptAction.MODIFY,
            provider_name=current_result.provider_name,
            provider_config=provider,
            route_match=current_result.route_match,
            modified_headers=headers,
            interceptor_name=self.name,
            metadata={"auth_type": provider.auth_type.value if api_key else "none"},
        )

    def _resolve_api_key(
        self, provider: ProviderConfig, ctx: RequestContext
    ) -> Optional[str]:
        """Resolve the API key for a provider.

        Checks (in order):
        1. Client-supplied x-api-key header
        2. Client-supplied Authorization header
        3. Environment variable (provider.api_key_env_var)

        Args:
            provider: The provider configuration.
            ctx: The request context.

        Returns:
            API key string if found, None otherwise.
        """
        # Client-supplied key
        client_key = ctx.get_header("x-api-key")
        if client_key:
            return client_key

        auth_header = ctx.get_header("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            return auth_header[7:]

        # Environment variable
        if provider.api_key_env_var:
            env_key = os.environ.get(provider.api_key_env_var)
            if env_key:
                return env_key

        return None


# --------------------------------------------------------------------------- #
# Request transform interceptor
# --------------------------------------------------------------------------- #


class RequestTransformInterceptor(RequestInterceptor):
    """Applies transformations to the request before forwarding.

    Supports model alias expansion, header injection, and custom
    body transformations.

    Args:
        model_aliases: Mapping of alias -> canonical model name.
        inject_headers: Headers to add to every request.
        transform_fn: Optional custom transformation function.
        name: Interceptor name.
        order: Execution order (default 30).
    """

    def __init__(
        self,
        model_aliases: Optional[dict[str, str]] = None,
        inject_headers: Optional[dict[str, str]] = None,
        transform_fn: Optional[Callable[[RequestContext], RequestContext]] = None,
        name: str = "request_transform",
        order: int = 30,
    ) -> None:
        super().__init__(name=name, order=order)
        self.model_aliases = model_aliases or {}
        self.inject_headers = inject_headers or {}
        self.transform_fn = transform_fn

    def intercept(
        self,
        ctx: RequestContext,
        current_result: Optional[InterceptResult] = None,
    ) -> InterceptResult:
        """Apply transformations to the request."""
        modified_body = None
        modified_headers: dict[str, str] = {}

        # Model alias expansion
        if ctx.model and ctx.model in self.model_aliases:
            canonical = self.model_aliases[ctx.model]
            logger.info(
                "RequestTransformInterceptor: alias '%s' -> '%s'",
                ctx.model,
                canonical,
            )
            ctx.model = canonical
            if ctx.body:
                modified_body = dict(ctx.body)
                modified_body["model"] = canonical

        # Header injection
        if self.inject_headers:
            modified_headers.update(self.inject_headers)

        # Custom transform
        if self.transform_fn:
            ctx = self.transform_fn(ctx)

        has_modifications = modified_body is not None or bool(modified_headers)

        return InterceptResult(
            action=InterceptAction.MODIFY if has_modifications else InterceptAction.SKIP,
            modified_body=modified_body,
            modified_headers=modified_headers,
            interceptor_name=self.name,
            metadata={"aliases_applied": bool(modified_body)},
        )


# --------------------------------------------------------------------------- #
# Interceptor chain
# --------------------------------------------------------------------------- #


class InterceptorChain:
    """Ordered chain of request interceptors.

    Interceptors are executed in order (sorted by ``order`` attribute).
    The chain stops when an interceptor returns a terminal action
    (FORWARD or REJECT). Non-terminal actions (SKIP, MODIFY, REDIRECT)
    allow the chain to continue.

    Accumulated modifications (headers, body changes) are merged into
    the final result.

    Args:
        interceptors: Initial list of interceptors.
    """

    def __init__(
        self,
        interceptors: Optional[list[RequestInterceptor]] = None,
    ) -> None:
        self._interceptors: list[RequestInterceptor] = []
        self._lock = threading.Lock()
        self._stats = ChainStats()

        if interceptors:
            for i in interceptors:
                self.add(i)

    def add(self, interceptor: RequestInterceptor) -> None:
        """Add an interceptor to the chain.

        Interceptors are sorted by their ``order`` attribute.

        Args:
            interceptor: The interceptor to add.
        """
        with self._lock:
            self._interceptors.append(interceptor)
            self._interceptors.sort(key=lambda i: i.order)
        logger.info(
            "InterceptorChain: added %s (order=%d)",
            interceptor.name,
            interceptor.order,
        )

    def remove(self, name: str) -> bool:
        """Remove an interceptor by name.

        Args:
            name: Name of the interceptor to remove.

        Returns:
            True if an interceptor was removed.
        """
        with self._lock:
            before = len(self._interceptors)
            self._interceptors = [
                i for i in self._interceptors if i.name != name
            ]
            return len(self._interceptors) < before

    def clear(self) -> None:
        """Remove all interceptors."""
        with self._lock:
            self._interceptors.clear()

    @property
    def interceptors(self) -> list[RequestInterceptor]:
        """Return a copy of the interceptor list."""
        with self._lock:
            return list(self._interceptors)

    def process(self, ctx: RequestContext) -> InterceptResult:
        """Process a request through the interceptor chain.

        Each interceptor receives the request context and the result
        from the previous interceptor. The chain stops on a terminal
        action. Modifications from MODIFY actions are accumulated.

        Args:
            ctx: The incoming request context.

        Returns:
            The final InterceptResult after all interceptors have run
            (or the first terminal result).
        """
        start = time.monotonic()
        current_result: Optional[InterceptResult] = None
        accumulated_headers: dict[str, str] = {}
        accumulated_body: Optional[dict[str, Any]] = None

        with self._lock:
            active = [i for i in self._interceptors if i.enabled]

        for interceptor in active:
            try:
                result = interceptor.intercept(ctx, current_result)
                result.interceptor_name = interceptor.name

                logger.debug(
                    "Interceptor '%s': action=%s",
                    interceptor.name,
                    result.action.value,
                )

                # Accumulate modifications
                if result.modified_headers:
                    accumulated_headers.update(result.modified_headers)
                if result.modified_body is not None:
                    accumulated_body = result.modified_body

                # Handle actions
                if result.action == InterceptAction.REJECT:
                    duration_ms = (time.monotonic() - start) * 1000.0
                    self._stats.rejected += 1
                    self._stats.total += 1
                    result.metadata["chain_duration_ms"] = round(duration_ms, 3)
                    logger.info(
                        "Request rejected by '%s': %s (%.2fms)",
                        interceptor.name,
                        result.error_message,
                        duration_ms,
                    )
                    return result

                if result.action == InterceptAction.FORWARD:
                    # Merge accumulated modifications
                    result.modified_headers = {
                        **accumulated_headers,
                        **result.modified_headers,
                    }
                    if accumulated_body and result.modified_body is None:
                        result.modified_body = accumulated_body
                    duration_ms = (time.monotonic() - start) * 1000.0
                    self._stats.forwarded += 1
                    self._stats.total += 1
                    result.metadata["chain_duration_ms"] = round(duration_ms, 3)
                    return result

                if result.action == InterceptAction.REDIRECT:
                    # Update current result with redirect info and continue
                    current_result = result

                if result.action == InterceptAction.MODIFY:
                    # Update current result to carry forward routing info
                    if current_result is None:
                        current_result = result
                    else:
                        # Merge modification into existing result
                        if result.modified_headers:
                            current_result.modified_headers.update(
                                result.modified_headers
                            )
                        if result.modified_body is not None:
                            current_result.modified_body = result.modified_body

                if result.action == InterceptAction.SKIP:
                    # Continue with existing result
                    pass

            except Exception as exc:
                logger.error(
                    "Interceptor '%s' raised exception: %s",
                    interceptor.name,
                    exc,
                    exc_info=True,
                )
                # Don't let a broken interceptor kill the chain
                self._stats.errors += 1
                continue

        # Chain completed without a terminal action
        duration_ms = (time.monotonic() - start) * 1000.0
        self._stats.total += 1

        if current_result is not None:
            # Use the last non-skip result
            current_result.modified_headers = {
                **accumulated_headers,
                **current_result.modified_headers,
            }
            if accumulated_body and current_result.modified_body is None:
                current_result.modified_body = accumulated_body
            current_result.metadata["chain_duration_ms"] = round(duration_ms, 3)

            # Promote MODIFY/REDIRECT to FORWARD if we have a provider
            if current_result.provider_name:
                current_result.action = InterceptAction.FORWARD
                self._stats.forwarded += 1
            else:
                self._stats.passthrough += 1

            return current_result

        # No interceptor produced a meaningful result
        self._stats.passthrough += 1
        return InterceptResult(
            action=InterceptAction.SKIP,
            interceptor_name="chain",
            metadata={"chain_duration_ms": round(duration_ms, 3)},
        )

    def get_stats(self) -> dict[str, Any]:
        """Return chain processing statistics."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = ChainStats()

    def __repr__(self) -> str:
        names = [i.name for i in self._interceptors]
        return f"InterceptorChain({names})"

    def __len__(self) -> int:
        return len(self._interceptors)


# --------------------------------------------------------------------------- #
# Chain statistics
# --------------------------------------------------------------------------- #


@dataclass
class ChainStats:
    """Statistics for the interceptor chain.

    Attributes:
        total: Total requests processed.
        forwarded: Requests forwarded to a provider.
        rejected: Requests rejected by an interceptor.
        passthrough: Requests that passed through without routing.
        errors: Interceptor errors encountered.
    """

    total: int = 0
    forwarded: int = 0
    rejected: int = 0
    passthrough: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total": self.total,
            "forwarded": self.forwarded,
            "rejected": self.rejected,
            "passthrough": self.passthrough,
            "errors": self.errors,
        }


# --------------------------------------------------------------------------- #
# Caching interceptor
# --------------------------------------------------------------------------- #


class CachingInterceptor(RequestInterceptor):
    """Caches responses to avoid redundant upstream requests.

    Checks the :class:`ResponseCache` before the request is forwarded.
    If a cached response exists, the interceptor returns a REJECT result
    carrying the cached data (which the gateway can use to short-circuit
    the upstream call). After a successful upstream response, the caller
    should call :meth:`store_response` to populate the cache.

    This interceptor only caches requests whose method is in
    ``cacheable_methods`` (default: GET, HEAD).

    Args:
        response_cache: A :class:`ResponseCache` instance.
        cacheable_methods: Methods eligible for caching.
        name: Interceptor name.
        order: Execution order (default 3, runs very early).
    """

    def __init__(
        self,
        response_cache: Optional[Any] = None,
        cacheable_methods: Optional[set[str]] = None,
        name: str = "caching",
        order: int = 3,
    ) -> None:
        super().__init__(name=name, order=order)
        self._response_cache = response_cache
        self._cacheable_methods = cacheable_methods or {"GET", "HEAD"}
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    @property
    def response_cache(self) -> Any:
        """The underlying ResponseCache (lazily imported)."""
        if self._response_cache is None:
            from src.response_cache import get_response_cache

            self._response_cache = get_response_cache()
        return self._response_cache

    def intercept(
        self,
        ctx: RequestContext,
        current_result: Optional[InterceptResult] = None,
    ) -> InterceptResult:
        """Check cache before forwarding request.

        If a cached response is available, returns a REJECT result with
        the cached data in metadata (status 200). The gateway layer
        should check for ``metadata["cached_response"]`` and serve it
        directly.
        """
        method = (ctx.method or "").upper()
        if method not in self._cacheable_methods:
            return InterceptResult(
                action=InterceptAction.SKIP,
                interceptor_name=self.name,
            )

        headers_dict = dict(ctx.headers) if ctx.headers else None
        cached = self.response_cache.lookup(
            method=method,
            path=ctx.path or "",
            headers=headers_dict,
        )

        if cached is not None:
            with self._lock:
                self._hits += 1
            logger.info(
                "CachingInterceptor: cache hit for %s %s (age=%.1fs)",
                method,
                ctx.path,
                cached.age,
            )
            return InterceptResult(
                action=InterceptAction.SKIP,
                interceptor_name=self.name,
                metadata={
                    "cached_response": cached,
                    "cache_hit": True,
                    "cache_age": round(cached.age, 2),
                },
            )

        with self._lock:
            self._misses += 1

        return InterceptResult(
            action=InterceptAction.SKIP,
            interceptor_name=self.name,
            metadata={"cache_hit": False},
        )

    def store_response(
        self,
        method: str,
        path: str,
        status_code: int,
        headers: dict[str, str],
        body: bytes,
        request_headers: Optional[dict[str, str]] = None,
        ttl: Optional[float] = None,
    ) -> bool:
        """Store a response in the cache after a successful upstream call.

        Args:
            method: HTTP method.
            path: Request path.
            status_code: Response status code.
            headers: Response headers.
            body: Response body bytes.
            request_headers: Original request headers.
            ttl: Optional TTL override.

        Returns:
            True if the response was cached.
        """
        return self.response_cache.store(
            method=method,
            path=path,
            status_code=status_code,
            headers=headers,
            body=body,
            request_headers=request_headers,
            ttl=ttl,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return caching interceptor statistics."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "response_cache": self.response_cache.get_stats().to_dict(),
            }


# --------------------------------------------------------------------------- #
# Factory: build a standard interceptor chain
# --------------------------------------------------------------------------- #


def create_default_chain(
    config: GatewayConfig,
    router: Optional[Router] = None,
    max_rpm: int = 0,
    model_aliases: Optional[dict[str, str]] = None,
    require_auth: bool = True,
    enable_caching: bool = False,
    cache_ttl: float = 120.0,
) -> InterceptorChain:
    """Create a standard interceptor chain for the gateway.

    The default chain includes (in order):
    1. CachingInterceptor (if enable_caching=True, order 3)
    2. RateLimitInterceptor (if max_rpm > 0, order 5)
    3. HeaderRoutingInterceptor (checks X-Provider header, order 10)
    4. AuthenticationInterceptor (order 20)
    5. RequestTransformInterceptor (model aliases, if provided, order 30)
    6. ModelRoutingInterceptor (uses Router for model-based routing, order 50)

    Args:
        config: Gateway configuration.
        router: Router instance. Created from config if not provided.
        max_rpm: Maximum requests per minute per client (0 = no limit).
        model_aliases: Model alias mapping.
        require_auth: Whether to require API keys.
        enable_caching: Whether to add a CachingInterceptor for GET/HEAD
            responses. Defaults to False.
        cache_ttl: TTL in seconds for cached responses (default: 120s).
            Only used when enable_caching=True.

    Returns:
        A configured InterceptorChain.
    """
    if router is None:
        router = Router.from_config(config)

    chain = InterceptorChain()

    # Response caching (order 3, runs before rate-limiting)
    if enable_caching:
        from src.response_cache import ResponseCache

        response_cache = ResponseCache(
            maxsize=256,
            default_ttl=cache_ttl,
            name="interceptor_response_cache",
        )
        chain.add(CachingInterceptor(response_cache=response_cache, order=3))

    # Rate limiting (order 5)
    if max_rpm > 0:
        chain.add(RateLimitInterceptor(max_requests_per_minute=max_rpm, order=5))

    # Header-based routing (order 10)
    chain.add(HeaderRoutingInterceptor(header="x-provider", config=config, order=10))

    # Authentication (order 20)
    chain.add(
        AuthenticationInterceptor(
            config=config, require_api_key=require_auth, order=20
        )
    )

    # Request transforms (order 30)
    if model_aliases:
        chain.add(
            RequestTransformInterceptor(model_aliases=model_aliases, order=30)
        )

    # Model routing (order 50)
    chain.add(ModelRoutingInterceptor(router=router, order=50))

    logger.info(
        "Created default interceptor chain with %d interceptors "
        "(caching=%s, rate_limit=%s)",
        len(chain),
        enable_caching,
        max_rpm > 0,
    )
    return chain
