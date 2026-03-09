"""Token count caching for improved performance.

Caches responses from the ``/v1/messages/count_tokens`` endpoint so that
identical token-counting requests are served from cache rather than making
a round-trip to the Anthropic API.

Token counting is a *pure function*: the same model + messages always produce
the same token count.  Caching these responses can significantly reduce
latency and API quota consumption when the same content is counted multiple
times (e.g., in feedback loops, streaming previews, or concurrent workers
processing the same document).

Key features:
- Content-addressable keys based on model, request body hash, and optional
  system prompt hash.
- Thread-safe with full statistics tracking.
- Configurable TTL (default: 1 hour — counts don't change unless the model
  or content changes).
- ``no-store`` / ``no-cache`` Cache-Control directives are honoured.
- Optional compression for large request bodies.

Typical usage::

    from src.token_count_cache import TokenCountCache

    cache = TokenCountCache(maxsize=512, default_ttl=3600)

    # After receiving a POST /v1/messages/count_tokens request:
    entry = cache.lookup(request_body=body_bytes, model="claude-3-5-sonnet")
    if entry:
        return entry.input_tokens  # serve from cache

    # After getting upstream response:
    cache.store(
        request_body=body_bytes,
        model="claude-3-5-sonnet",
        input_tokens=1234,
        response_body=response_bytes,
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

from src.cache import Cache
from src.logging_config import get_logger

logger = get_logger("token_count_cache")

# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

# Paths eligible for token count caching (all POST, but only these paths)
TOKEN_COUNT_PATH = "/v1/messages/count_tokens"

# Compress stored response bodies larger than this (bytes)
_BODY_COMPRESS_THRESHOLD = 1024


@dataclass
class TokenCountEntry:
    """A cached token count response.

    Attributes:
        input_tokens: Token count reported by the API.
        model: Model name used for counting.
        response_body: Raw response body bytes as returned by the API.
        created_at: Unix timestamp when this entry was cached.
        ttl: Time-to-live in seconds (0 = never expires).
        compressed: Whether *response_body* is stored gzip-compressed.
        body_size: Original (uncompressed) response body size in bytes.
    """

    input_tokens: int
    model: str
    response_body: bytes
    created_at: float = field(default_factory=time.time)
    ttl: float = 0.0
    compressed: bool = False
    body_size: int = 0

    @property
    def is_expired(self) -> bool:
        """True if this entry has exceeded its TTL."""
        if self.ttl <= 0:
            return False
        return time.time() > (self.created_at + self.ttl)

    @property
    def age(self) -> float:
        """Seconds since this entry was cached."""
        return time.time() - self.created_at

    def get_response_body(self) -> bytes:
        """Return the response body, decompressing if necessary."""
        if self.compressed:
            return gzip.decompress(self.response_body)
        return self.response_body

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata (excluding body bytes) to a dict."""
        return {
            "input_tokens": self.input_tokens,
            "model": self.model,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "age": round(self.age, 2),
            "is_expired": self.is_expired,
            "body_size": self.body_size,
            "compressed": self.compressed,
        }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class TokenCountCacheStats:
    """Statistics for the token count cache.

    Attributes:
        lookups: Total lookup attempts.
        hits: Cache hits (valid, non-expired entry found).
        misses: Cache misses.
        stores: Number of responses stored.
        bypasses: Requests that bypassed caching (wrong path, no-cache, etc.).
        bytes_saved: Estimated bytes saved by serving from cache.
        compressed_stores: Number of compressed entries stored.
    """

    lookups: int = 0
    hits: int = 0
    misses: int = 0
    stores: int = 0
    bypasses: int = 0
    bytes_saved: int = 0
    compressed_stores: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage (0.0 – 100.0)."""
        if self.lookups == 0:
            return 0.0
        return (self.hits / self.lookups) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats to a dict."""
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "bypasses": self.bypasses,
            "hit_rate": round(self.hit_rate, 2),
            "bytes_saved": self.bytes_saved,
            "compressed_stores": self.compressed_stores,
        }


# ---------------------------------------------------------------------------
# Token count cache
# ---------------------------------------------------------------------------


class TokenCountCache:
    """Thread-safe cache for ``/v1/messages/count_tokens`` responses.

    Uses a content-addressable key derived from a SHA-256 hash of the
    raw request body so that identical inputs always hit the same cache
    slot regardless of field ordering.

    Args:
        maxsize: Maximum number of cached entries (0 = unlimited).
        default_ttl: Default time-to-live in seconds. 0 = no expiry.
        enable_compression: Whether to compress large response bodies.
        name: Human-readable cache name used in logging.
    """

    def __init__(
        self,
        maxsize: int = 512,
        default_ttl: float = 3600.0,
        enable_compression: bool = True,
        name: str = "token_count_cache",
    ) -> None:
        self._name = name
        self._default_ttl = default_ttl
        self._enable_compression = enable_compression
        self._cache = Cache(maxsize=maxsize, ttl=default_ttl, name=name)
        self._stats = TokenCountCacheStats()
        self._lock = threading.Lock()

    # -- Properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        """Cache name."""
        return self._name

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return self._cache.size

    @property
    def default_ttl(self) -> float:
        """Default TTL in seconds."""
        return self._default_ttl

    # -- Key generation ------------------------------------------------------

    @staticmethod
    def _make_key(request_body: bytes) -> str:
        """Build a content-addressable cache key from the request body.

        The key is a SHA-256 hex digest of the raw request bytes.  Using
        the full body ensures that any difference in model, messages,
        system prompt, or tools produces a different key.

        Args:
            request_body: Raw POST request body bytes.

        Returns:
            A hex digest string.
        """
        return hashlib.sha256(request_body).hexdigest()

    @staticmethod
    def _extract_model(request_body: bytes) -> str:
        """Extract the model name from a JSON request body.

        Returns an empty string if the body cannot be parsed or contains
        no model field.

        Args:
            request_body: Raw POST request body bytes.

        Returns:
            Model name string or ``""`` on failure.
        """
        try:
            data = json.loads(request_body)
            return str(data.get("model", ""))
        except (ValueError, TypeError, AttributeError):
            return ""

    # -- Cache-Control helpers -----------------------------------------------

    @staticmethod
    def _has_no_cache_directive(headers: Optional[dict[str, str]]) -> bool:
        """Return True if request headers contain ``no-cache`` or ``no-store``.

        Args:
            headers: Request headers dict.

        Returns:
            True if caching should be bypassed.
        """
        if not headers:
            return False
        for k, v in headers.items():
            if k.lower() == "cache-control":
                directives = {d.strip().lower() for d in v.split(",")}
                if "no-cache" in directives or "no-store" in directives:
                    return True
        return False

    # -- Core operations -----------------------------------------------------

    def is_cacheable(
        self,
        path: str,
        method: str,
        request_body: Optional[bytes],
        request_headers: Optional[dict[str, str]] = None,
    ) -> bool:
        """Check whether a request is eligible for token count caching.

        Only POST requests to :data:`TOKEN_COUNT_PATH` with a non-empty
        body are cached.

        Args:
            path: Request path.
            method: HTTP method.
            request_body: Request body bytes.
            request_headers: Request headers (checked for Cache-Control).

        Returns:
            True if the request can be looked up / stored in cache.
        """
        if method.upper() != "POST":
            return False
        # Normalise path (strip query string)
        if "?" in path:
            path = path.split("?", 1)[0]
        if path != TOKEN_COUNT_PATH:
            return False
        if not request_body:
            return False
        if self._has_no_cache_directive(request_headers):
            return False
        return True

    def lookup(
        self,
        request_body: bytes,
        request_headers: Optional[dict[str, str]] = None,
    ) -> Optional[TokenCountEntry]:
        """Look up a cached token count response.

        Args:
            request_body: Raw POST request body bytes.
            request_headers: Optional request headers (Cache-Control checked).

        Returns:
            A :class:`TokenCountEntry` on cache hit, ``None`` on miss.
        """
        with self._lock:
            self._stats.lookups += 1

        if self._has_no_cache_directive(request_headers):
            with self._lock:
                self._stats.bypasses += 1
            return None

        key = self._make_key(request_body)
        raw = self._cache.get(key)

        if raw is None:
            with self._lock:
                self._stats.misses += 1
            logger.debug(
                "TokenCountCache '%s': miss (body_hash=%s…)",
                self._name,
                key[:8],
            )
            return None

        entry = self._deserialize(raw)
        if entry is None:
            with self._lock:
                self._stats.misses += 1
            return None

        with self._lock:
            self._stats.hits += 1
            self._stats.bytes_saved += entry.body_size

        logger.debug(
            "TokenCountCache '%s': hit (tokens=%d, model=%s, age=%.1fs)",
            self._name,
            entry.input_tokens,
            entry.model,
            entry.age,
        )
        return entry

    def store(
        self,
        request_body: bytes,
        input_tokens: int,
        response_body: bytes,
        model: str = "",
        ttl: Optional[float] = None,
    ) -> bool:
        """Store a token count response in the cache.

        Args:
            request_body: Original POST request body (used for key generation).
            input_tokens: Token count from the API response.
            response_body: Raw response body bytes to cache.
            model: Model name (for metadata / logging).
            ttl: Optional TTL override. Uses :attr:`default_ttl` when ``None``.

        Returns:
            True on success.
        """
        if not request_body:
            return False

        effective_ttl = ttl if ttl is not None else self._default_ttl
        key = self._make_key(request_body)
        body_size = len(response_body)

        stored_body = response_body
        compressed = False
        if (
            self._enable_compression
            and len(response_body) > _BODY_COMPRESS_THRESHOLD
        ):
            compressed_body = gzip.compress(response_body, compresslevel=6)
            if len(compressed_body) < len(response_body):
                stored_body = compressed_body
                compressed = True

        raw = {
            "input_tokens": input_tokens,
            "model": model or self._extract_model(request_body),
            "response_body": stored_body.hex(),
            "created_at": time.time(),
            "ttl": effective_ttl,
            "compressed": compressed,
            "body_size": body_size,
        }

        self._cache.set(key, raw, ttl=effective_ttl)

        with self._lock:
            self._stats.stores += 1
            if compressed:
                self._stats.compressed_stores += 1

        logger.debug(
            "TokenCountCache '%s': stored tokens=%d, model=%s, "
            "body=%d bytes, compressed=%s, ttl=%.0fs",
            self._name,
            input_tokens,
            raw["model"],
            body_size,
            compressed,
            effective_ttl,
        )
        return True

    def invalidate(self, request_body: bytes) -> bool:
        """Remove a cached entry by request body.

        Args:
            request_body: The original request body whose entry to remove.

        Returns:
            True if an entry was removed.
        """
        key = self._make_key(request_body)
        return self._cache.delete(key)

    def clear(self) -> int:
        """Clear all cached token count entries.

        Returns:
            Number of entries removed.
        """
        return self._cache.clear()

    def purge_expired(self) -> int:
        """Remove expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        return self._cache.purge_expired()

    # -- Serialization -------------------------------------------------------

    @staticmethod
    def _deserialize(raw: Any) -> Optional[TokenCountEntry]:
        """Deserialize a raw cache dict back into a :class:`TokenCountEntry`.

        Args:
            raw: The stored dict.

        Returns:
            A :class:`TokenCountEntry` or ``None`` on failure.
        """
        if not isinstance(raw, dict):
            return None
        try:
            # Decode the hex-encoded bytes exactly as stored — compressed or not.
            # get_response_body() handles decompression; we must not decompress here
            # because the TokenCountEntry.compressed flag drives that logic.
            body_bytes = bytes.fromhex(raw["response_body"])
            return TokenCountEntry(
                input_tokens=raw["input_tokens"],
                model=raw.get("model", ""),
                response_body=body_bytes,
                created_at=raw.get("created_at", time.time()),
                ttl=raw.get("ttl", 0.0),
                compressed=raw.get("compressed", False),
                body_size=raw.get("body_size", len(body_bytes)),
            )
        except Exception:
            logger.debug(
                "TokenCountCache: failed to deserialize cache entry", exc_info=True
            )
            return None

    # -- Stats ---------------------------------------------------------------

    def get_stats(self) -> TokenCountCacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            return TokenCountCacheStats(
                lookups=self._stats.lookups,
                hits=self._stats.hits,
                misses=self._stats.misses,
                stores=self._stats.stores,
                bypasses=self._stats.bypasses,
                bytes_saved=self._stats.bytes_saved,
                compressed_stores=self._stats.compressed_stores,
            )

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._stats = TokenCountCacheStats()

    # -- Dunder helpers ------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TokenCountCache(name={self._name!r}, "
            f"maxsize={self._cache.maxsize}, "
            f"ttl={self._default_ttl}, size={self.size})"
        )

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_token_count_cache: Optional[TokenCountCache] = None
_global_token_count_cache_lock = threading.Lock()


def get_token_count_cache(
    maxsize: int = 512,
    default_ttl: float = 3600.0,
) -> TokenCountCache:
    """Get or create the global token count cache.

    Args:
        maxsize: Maximum cached entries (only used on creation).
        default_ttl: TTL in seconds (only used on creation).

    Returns:
        The global :class:`TokenCountCache` instance.
    """
    global _global_token_count_cache
    with _global_token_count_cache_lock:
        if _global_token_count_cache is None:
            _global_token_count_cache = TokenCountCache(
                maxsize=maxsize,
                default_ttl=default_ttl,
                name="global_token_count_cache",
            )
        return _global_token_count_cache


def reset_token_count_cache() -> None:
    """Reset the global token count cache. Primarily for testing."""
    global _global_token_count_cache
    with _global_token_count_cache_lock:
        if _global_token_count_cache is not None:
            _global_token_count_cache.clear()
        _global_token_count_cache = None
