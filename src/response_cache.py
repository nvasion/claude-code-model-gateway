"""HTTP response caching for the model gateway.

Provides a specialized cache for HTTP responses that avoids redundant
upstream calls. Supports content-addressable keys, configurable cacheable
methods and status codes, and Cache-Control header awareness.

Key features:
- Content-addressable keys based on method, path, headers, and body hash.
- Configurable set of cacheable HTTP methods and status codes.
- Respects ``Cache-Control: no-cache`` and ``no-store`` directives.
- Automatic response body compression for large payloads.
- Thread-safe with full statistics tracking.

Typical usage::

    from src.response_cache import ResponseCache

    cache = ResponseCache(maxsize=256, default_ttl=120)

    # Check for cached response
    cached = cache.lookup(method="GET", path="/v1/models")
    if cached:
        return cached  # CachedResponse

    # After getting upstream response, store it
    cache.store(
        method="GET",
        path="/v1/models",
        status_code=200,
        headers={"Content-Type": "application/json"},
        body=response_bytes,
    )
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.cache import Cache, CacheStats, get_cache
from src.logging_config import get_logger

logger = get_logger("response_cache")

# ---------------------------------------------------------------------------
# Response cache entry
# ---------------------------------------------------------------------------

# Compress response bodies larger than this threshold (bytes)
_BODY_COMPRESS_THRESHOLD = 4096


@dataclass
class CachedResponse:
    """A cached HTTP response.

    Attributes:
        status_code: HTTP status code.
        headers: Response headers as a dict.
        body: Response body bytes.
        created_at: Timestamp when this response was cached.
        ttl: Time-to-live in seconds.
        request_method: The HTTP method of the original request.
        request_path: The path of the original request.
        body_size: Original body size before compression.
        compressed: Whether the body was stored compressed.
    """

    status_code: int
    headers: dict[str, str]
    body: bytes
    created_at: float = field(default_factory=time.time)
    ttl: float = 0.0
    request_method: str = ""
    request_path: str = ""
    body_size: int = 0
    compressed: bool = False

    @property
    def is_expired(self) -> bool:
        """Check whether this cached response has expired."""
        if self.ttl <= 0:
            return False
        return time.time() > (self.created_at + self.ttl)

    @property
    def age(self) -> float:
        """Seconds since this response was cached."""
        return time.time() - self.created_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata (excluding body) to a dictionary."""
        return {
            "status_code": self.status_code,
            "headers": self.headers,
            "request_method": self.request_method,
            "request_path": self.request_path,
            "body_size": self.body_size,
            "compressed": self.compressed,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "age": round(self.age, 2),
            "is_expired": self.is_expired,
        }


# ---------------------------------------------------------------------------
# Response cache statistics
# ---------------------------------------------------------------------------


@dataclass
class ResponseCacheStats:
    """Statistics for the response cache.

    Attributes:
        lookups: Total number of lookup attempts.
        hits: Number of cache hits.
        misses: Number of cache misses.
        stores: Number of responses stored.
        evictions: Number of evicted entries.
        bypasses: Number of requests that bypassed caching.
        bytes_saved: Estimated bytes saved by cache hits.
        compressed_stores: Number of responses stored compressed.
    """

    lookups: int = 0
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    bypasses: int = 0
    bytes_saved: int = 0
    compressed_stores: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        if self.lookups == 0:
            return 0.0
        return (self.hits / self.lookups) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats to a dictionary."""
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "bypasses": self.bypasses,
            "hit_rate": round(self.hit_rate, 2),
            "bytes_saved": self.bytes_saved,
            "compressed_stores": self.compressed_stores,
        }


# ---------------------------------------------------------------------------
# Response cache implementation
# ---------------------------------------------------------------------------


# Default cacheable HTTP methods (safe, idempotent methods)
DEFAULT_CACHEABLE_METHODS = {"GET", "HEAD"}

# Default cacheable status codes
DEFAULT_CACHEABLE_STATUS_CODES = {200, 203, 204, 206, 300, 301, 404, 405, 410}


class ResponseCache:
    """Thread-safe HTTP response cache.

    Caches HTTP responses keyed by a content-addressable hash of the
    request method, path, selected headers, and (optionally) request body.
    Respects ``Cache-Control`` directives and supports body compression.

    Args:
        maxsize: Maximum number of cached responses.
        default_ttl: Default time-to-live for cached responses in seconds.
        cacheable_methods: Set of HTTP methods eligible for caching.
        cacheable_status_codes: Set of HTTP status codes eligible for caching.
        vary_headers: Headers whose values vary the cache key.
        enable_compression: Whether to compress large response bodies.
        name: Human-readable name for this cache.
    """

    def __init__(
        self,
        maxsize: int = 256,
        default_ttl: float = 120.0,
        cacheable_methods: Optional[set[str]] = None,
        cacheable_status_codes: Optional[set[int]] = None,
        vary_headers: Optional[list[str]] = None,
        enable_compression: bool = True,
        name: str = "response_cache",
    ) -> None:
        self._name = name
        self._default_ttl = default_ttl
        self._cacheable_methods = cacheable_methods or DEFAULT_CACHEABLE_METHODS
        self._cacheable_status_codes = (
            cacheable_status_codes or DEFAULT_CACHEABLE_STATUS_CODES
        )
        self._vary_headers = [h.lower() for h in (vary_headers or [])]
        self._enable_compression = enable_compression
        self._cache = Cache(maxsize=maxsize, ttl=default_ttl, name=name)
        self._stats = ResponseCacheStats()
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Cache name."""
        return self._name

    @property
    def size(self) -> int:
        """Current number of cached responses."""
        return self._cache.size

    # -- Cache key generation ------------------------------------------------

    def _make_key(
        self,
        method: str,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[bytes] = None,
    ) -> str:
        """Build a content-addressable cache key.

        The key is a SHA-256 hash of the method, path, selected vary
        headers, and an optional body hash.

        Args:
            method: HTTP method.
            path: Request path (including query string).
            headers: Request headers.
            body: Request body bytes.

        Returns:
            A hex digest string suitable as a cache key.
        """
        parts = [method.upper(), path]

        # Include vary header values in the key
        if headers and self._vary_headers:
            for header_name in sorted(self._vary_headers):
                value = ""
                for k, v in headers.items():
                    if k.lower() == header_name:
                        value = v
                        break
                parts.append(f"{header_name}:{value}")

        # Include body hash for non-GET requests
        if body and method.upper() not in {"GET", "HEAD"}:
            body_hash = hashlib.sha256(body).hexdigest()[:16]
            parts.append(f"body:{body_hash}")

        key_input = "|".join(parts)
        return hashlib.sha256(key_input.encode()).hexdigest()

    # -- Cache-Control parsing -----------------------------------------------

    @staticmethod
    def _parse_cache_control(
        headers: Optional[dict[str, str]],
    ) -> dict[str, Any]:
        """Parse Cache-Control directives from headers.

        Args:
            headers: HTTP headers dict.

        Returns:
            Dictionary of parsed directives.
        """
        directives: dict[str, Any] = {}
        if not headers:
            return directives

        cc_value = ""
        for k, v in headers.items():
            if k.lower() == "cache-control":
                cc_value = v
                break

        if not cc_value:
            return directives

        for part in cc_value.split(","):
            part = part.strip().lower()
            if "=" in part:
                key, val = part.split("=", 1)
                try:
                    directives[key.strip()] = int(val.strip())
                except ValueError:
                    directives[key.strip()] = val.strip()
            else:
                directives[part] = True

        return directives

    # -- Cacheability checks -------------------------------------------------

    def is_cacheable_request(
        self,
        method: str,
        headers: Optional[dict[str, str]] = None,
    ) -> bool:
        """Check whether a request is eligible for cache lookup/storage.

        Args:
            method: HTTP method.
            headers: Request headers.

        Returns:
            True if the request can be served from / stored in cache.
        """
        if method.upper() not in self._cacheable_methods:
            return False

        directives = self._parse_cache_control(headers)
        if directives.get("no-cache") or directives.get("no-store"):
            return False

        return True

    def is_cacheable_response(
        self,
        status_code: int,
        headers: Optional[dict[str, str]] = None,
    ) -> bool:
        """Check whether a response is eligible for caching.

        Args:
            status_code: HTTP response status code.
            headers: Response headers.

        Returns:
            True if the response can be stored in cache.
        """
        if status_code not in self._cacheable_status_codes:
            return False

        directives = self._parse_cache_control(headers)
        if directives.get("no-store"):
            return False
        if directives.get("private"):
            return False

        return True

    # -- Core operations -----------------------------------------------------

    def lookup(
        self,
        method: str,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[bytes] = None,
    ) -> Optional[CachedResponse]:
        """Look up a cached response.

        Args:
            method: HTTP method.
            path: Request path.
            headers: Request headers.
            body: Request body.

        Returns:
            A CachedResponse if found, None otherwise.
        """
        with self._lock:
            self._stats.lookups += 1

        if not self.is_cacheable_request(method, headers):
            with self._lock:
                self._stats.bypasses += 1
            return None

        key = self._make_key(method, path, headers, body)
        entry = self._cache.get(key)

        if entry is None:
            with self._lock:
                self._stats.misses += 1
            return None

        # Reconstruct CachedResponse
        cached = self._deserialize_entry(entry)
        if cached is None:
            with self._lock:
                self._stats.misses += 1
            return None

        with self._lock:
            self._stats.hits += 1
            self._stats.bytes_saved += cached.body_size

        logger.debug(
            "ResponseCache '%s': hit for %s %s (age=%.1fs)",
            self._name,
            method,
            path,
            cached.age,
        )
        return cached

    def store(
        self,
        method: str,
        path: str,
        status_code: int,
        headers: dict[str, str],
        body: bytes,
        request_headers: Optional[dict[str, str]] = None,
        request_body: Optional[bytes] = None,
        ttl: Optional[float] = None,
    ) -> bool:
        """Store an HTTP response in the cache.

        Args:
            method: HTTP method of the request.
            path: Request path.
            status_code: Response status code.
            headers: Response headers.
            body: Response body bytes.
            request_headers: Original request headers (for key generation).
            request_body: Original request body (for key generation).
            ttl: Optional TTL override in seconds.

        Returns:
            True if the response was cached, False if it was not cacheable.
        """
        if not self.is_cacheable_response(status_code, headers):
            return False

        if not self.is_cacheable_request(method, request_headers):
            return False

        # Determine TTL
        effective_ttl = ttl if ttl is not None else self._default_ttl
        response_directives = self._parse_cache_control(headers)
        if "max-age" in response_directives:
            try:
                effective_ttl = float(response_directives["max-age"])
            except (ValueError, TypeError):
                pass

        key = self._make_key(method, path, request_headers, request_body)

        # Optionally compress body
        stored_body = body
        compressed = False
        if (
            self._enable_compression
            and len(body) > _BODY_COMPRESS_THRESHOLD
        ):
            compressed_body = gzip.compress(body, compresslevel=6)
            if len(compressed_body) < len(body):
                stored_body = compressed_body
                compressed = True

        entry = {
            "status_code": status_code,
            "headers": headers,
            "body": stored_body.hex(),
            "created_at": time.time(),
            "ttl": effective_ttl,
            "request_method": method,
            "request_path": path,
            "body_size": len(body),
            "compressed": compressed,
        }

        self._cache.set(key, entry, ttl=effective_ttl)

        with self._lock:
            self._stats.stores += 1
            if compressed:
                self._stats.compressed_stores += 1

        logger.debug(
            "ResponseCache '%s': stored %s %s (status=%d, size=%d, "
            "compressed=%s, ttl=%.0fs)",
            self._name,
            method,
            path,
            status_code,
            len(body),
            compressed,
            effective_ttl,
        )
        return True

    def invalidate(
        self,
        method: str,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[bytes] = None,
    ) -> bool:
        """Remove a cached response.

        Args:
            method: HTTP method.
            path: Request path.
            headers: Request headers.
            body: Request body.

        Returns:
            True if an entry was removed.
        """
        key = self._make_key(method, path, headers, body)
        return self._cache.delete(key)

    def invalidate_path(self, path: str) -> int:
        """Invalidate all cached responses for a given path.

        This is a brute-force scan — use sparingly.

        Args:
            path: The request path to invalidate.

        Returns:
            Number of entries removed.
        """
        removed = 0
        keys = self._cache.keys()
        for key in keys:
            entry = self._cache.get(key)
            if entry and isinstance(entry, dict):
                if entry.get("request_path") == path:
                    self._cache.delete(key)
                    removed += 1
        return removed

    def clear(self) -> int:
        """Clear all cached responses.

        Returns:
            Number of entries removed.
        """
        return self._cache.clear()

    # -- Serialization helpers -----------------------------------------------

    @staticmethod
    def _deserialize_entry(entry: Any) -> Optional[CachedResponse]:
        """Deserialize a cache entry back into a CachedResponse.

        Args:
            entry: The raw cached dict.

        Returns:
            A CachedResponse or None if deserialization fails.
        """
        if not isinstance(entry, dict):
            return None
        try:
            body_hex = entry["body"]
            body_bytes = bytes.fromhex(body_hex)

            # Decompress if needed
            if entry.get("compressed", False):
                body_bytes = gzip.decompress(body_bytes)

            return CachedResponse(
                status_code=entry["status_code"],
                headers=entry.get("headers", {}),
                body=body_bytes,
                created_at=entry.get("created_at", time.time()),
                ttl=entry.get("ttl", 0),
                request_method=entry.get("request_method", ""),
                request_path=entry.get("request_path", ""),
                body_size=entry.get("body_size", len(body_bytes)),
                compressed=entry.get("compressed", False),
            )
        except Exception:
            return None

    # -- Stats ---------------------------------------------------------------

    def get_stats(self) -> ResponseCacheStats:
        """Return a snapshot of response cache statistics."""
        with self._lock:
            return ResponseCacheStats(
                lookups=self._stats.lookups,
                hits=self._stats.hits,
                misses=self._stats.misses,
                stores=self._stats.stores,
                evictions=self._stats.evictions,
                bypasses=self._stats.bypasses,
                bytes_saved=self._stats.bytes_saved,
                compressed_stores=self._stats.compressed_stores,
            )

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._stats = ResponseCacheStats()

    def __repr__(self) -> str:
        return (
            f"ResponseCache(name={self._name!r}, maxsize={self._cache.maxsize}, "
            f"ttl={self._default_ttl}, size={self.size})"
        )

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Global response cache instance
# ---------------------------------------------------------------------------

_response_cache: Optional[ResponseCache] = None
_response_cache_lock = threading.Lock()


def get_response_cache(
    maxsize: int = 256,
    default_ttl: float = 120.0,
) -> ResponseCache:
    """Get or create the global response cache.

    Args:
        maxsize: Maximum cached responses (only used on creation).
        default_ttl: Default TTL in seconds (only used on creation).

    Returns:
        The global ResponseCache instance.
    """
    global _response_cache
    with _response_cache_lock:
        if _response_cache is None:
            _response_cache = ResponseCache(
                maxsize=maxsize,
                default_ttl=default_ttl,
                name="global_response_cache",
            )
        return _response_cache


def reset_response_cache() -> None:
    """Reset the global response cache. Primarily for testing."""
    global _response_cache
    with _response_cache_lock:
        if _response_cache is not None:
            _response_cache.clear()
        _response_cache = None
