"""Caching mechanism for improved performance.

Provides an in-memory LRU cache with TTL (time-to-live) support,
a file-based persistent cache, and a decorator for transparent
function-level caching. All caches are thread-safe.

Typical usage:

    from src.cache import Cache, cached

    # Use directly
    cache = Cache(maxsize=256, ttl=300)
    cache.set("key", value)
    value = cache.get("key")

    # Use as a decorator
    @cached(ttl=60)
    def expensive_function(arg):
        ...

    # Use the global gateway cache
    from src.cache import get_default_cache
    default = get_default_cache()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from src.logging_config import get_logger

logger = get_logger("cache")

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Statistics for a cache instance.

    Attributes:
        hits: Number of cache hits (key found and not expired).
        misses: Number of cache misses (key not found or expired).
        evictions: Number of entries evicted due to size constraints.
        expirations: Number of entries removed due to TTL expiry.
        sets: Number of set operations.
        current_size: Current number of entries in the cache.
        max_size: Maximum allowed entries.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    sets: int = 0
    current_size: int = 0
    max_size: int = 0
    stale_hits: int = 0
    compressed_entries: int = 0

    @property
    def total_requests(self) -> int:
        """Total number of get requests."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage (0.0 - 100.0)."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats to a dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "sets": self.sets,
            "current_size": self.current_size,
            "max_size": self.max_size,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 2),
            "stale_hits": self.stale_hits,
            "compressed_entries": self.compressed_entries,
        }


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A single cache entry with expiration metadata.

    Attributes:
        key: The cache key.
        value: The cached value.
        created_at: Timestamp when the entry was created.
        ttl: Time-to-live in seconds (0 means no expiration).
        access_count: Number of times this entry has been accessed.
        last_accessed: Timestamp of last access.
    """

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: float = 0.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check whether this entry has expired."""
        if self.ttl <= 0:
            return False
        return time.time() > (self.created_at + self.ttl)

    @property
    def expires_at(self) -> Optional[float]:
        """Timestamp when this entry expires, or None if no TTL."""
        if self.ttl <= 0:
            return None
        return self.created_at + self.ttl

    @property
    def remaining_ttl(self) -> Optional[float]:
        """Seconds remaining until expiry, or None if no TTL."""
        if self.ttl <= 0:
            return None
        remaining = (self.created_at + self.ttl) - time.time()
        return max(0.0, remaining)

    def touch(self) -> None:
        """Record an access on this entry."""
        self.access_count += 1
        self.last_accessed = time.time()


# ---------------------------------------------------------------------------
# In-memory LRU cache with TTL
# ---------------------------------------------------------------------------


class Cache:
    """Thread-safe in-memory LRU cache with TTL support.

    Args:
        maxsize: Maximum number of entries. 0 means unlimited.
        ttl: Default time-to-live in seconds. 0 means no expiration.
        name: Human-readable name for this cache (used in logging/stats).
        stale_ttl: Extra seconds after TTL expiry during which an expired
            entry may still be served as stale (stale-while-revalidate).
            0 means disabled.
        refresh_callback: Optional callable ``(key) -> value`` invoked in a
            background thread when a stale hit is served, so the cache can
            be refreshed asynchronously.
    """

    def __init__(
        self,
        maxsize: int = 256,
        ttl: float = 0.0,
        name: str = "default",
        stale_ttl: float = 0.0,
        refresh_callback: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._maxsize = maxsize
        self._default_ttl = ttl
        self._name = name
        self._stale_ttl = stale_ttl
        self._refresh_callback = refresh_callback
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats(max_size=maxsize)
        self._refreshing_keys: set = set()

    # -- Properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        """Cache name."""
        return self._name

    @property
    def maxsize(self) -> int:
        """Maximum number of entries."""
        return self._maxsize

    @property
    def default_ttl(self) -> float:
        """Default TTL in seconds."""
        return self._default_ttl

    @property
    def stale_ttl(self) -> float:
        """Extra stale window in seconds (stale-while-revalidate)."""
        return self._stale_ttl

    @property
    def size(self) -> int:
        """Current number of entries (including expired ones not yet purged)."""
        with self._lock:
            return len(self._data)

    # -- Core operations -----------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key.

        If the entry exists but has expired it is removed and treated as a miss,
        unless stale_ttl > 0 and the entry is still within the stale window — in
        which case the stale value is returned and a background refresh may be
        triggered.

        Args:
            key: Cache key.
            default: Value returned on miss.

        Returns:
            The cached value or *default*.
        """
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._stats.misses += 1
                return default

            if not entry.is_expired:
                # Move to end (most recently used)
                self._data.move_to_end(key)
                entry.touch()
                self._stats.hits += 1
                logger.debug("Cache '%s': hit key '%s'", self._name, key)
                return entry.value

            # Entry is expired — check stale-while-revalidate
            if self._stale_ttl > 0:
                stale_expires_at = entry.created_at + entry.ttl + self._stale_ttl
                if time.time() <= stale_expires_at:
                    # Serve stale value
                    self._stats.hits += 1
                    self._stats.stale_hits += 1
                    logger.debug(
                        "Cache '%s': stale hit key '%s'", self._name, key
                    )
                    # Trigger background refresh if callback set and not already refreshing
                    if (
                        self._refresh_callback is not None
                        and key not in self._refreshing_keys
                    ):
                        self._refreshing_keys.add(key)
                        import threading as _t

                        cb = self._refresh_callback

                        def _do_refresh(k: str = key) -> None:
                            try:
                                new_val = cb(k)
                                self.set(k, new_val)
                            except Exception:
                                pass
                            finally:
                                with self._lock:
                                    self._refreshing_keys.discard(k)

                        _t.Thread(target=_do_refresh, daemon=True).start()
                    return entry.value

            # Truly expired
            del self._data[key]
            self._stats.expirations += 1
            self._stats.misses += 1
            self._stats.current_size = len(self._data)
            logger.debug("Cache '%s': expired key '%s'", self._name, key)
            return default

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Per-entry TTL override. ``None`` uses the cache default.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl

        with self._lock:
            if key in self._data:
                # Update existing
                del self._data[key]

            entry = CacheEntry(key=key, value=value, ttl=effective_ttl)
            self._data[key] = entry

            # Evict LRU entries if over capacity
            while self._maxsize > 0 and len(self._data) > self._maxsize:
                evicted_key, _ = self._data.popitem(last=False)
                self._stats.evictions += 1
                logger.debug(
                    "Cache '%s': evicted key '%s'", self._name, evicted_key
                )

            self._stats.sets += 1
            self._stats.current_size = len(self._data)

    def delete(self, key: str) -> bool:
        """Remove a single entry.

        Args:
            key: Cache key to remove.

        Returns:
            True if the key was present and removed.
        """
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._stats.current_size = len(self._data)
                return True
            return False

    def has(self, key: str) -> bool:
        """Check whether *key* is present and not expired (without counting as a hit)."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._data[key]
                self._stats.expirations += 1
                self._stats.current_size = len(self._data)
                return False
            return True

    def clear(self) -> int:
        """Remove all entries.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            count = len(self._data)
            self._data.clear()
            self._stats.current_size = 0
            logger.debug("Cache '%s': cleared %d entries", self._name, count)
            return count

    def keys(self) -> list[str]:
        """Return a list of all non-expired keys."""
        with self._lock:
            result = []
            expired_keys = []
            for key, entry in self._data.items():
                if entry.is_expired:
                    expired_keys.append(key)
                else:
                    result.append(key)
            for key in expired_keys:
                del self._data[key]
                self._stats.expirations += 1
            self._stats.current_size = len(self._data)
            return result

    def purge_expired(self) -> int:
        """Remove all expired entries.

        Entries within the stale-while-revalidate window (stale_ttl) are NOT
        purged, as they can still be served as stale values.

        Returns:
            Number of expired entries removed.
        """
        with self._lock:
            now = time.time()
            expired_keys = []
            for key, entry in self._data.items():
                if entry.is_expired:
                    stale_expires = entry.created_at + entry.ttl + self._stale_ttl
                    if now > stale_expires:
                        expired_keys.append(key)
            for key in expired_keys:
                del self._data[key]
                self._stats.expirations += 1
            self._stats.current_size = len(self._data)
            logger.debug(
                "Cache '%s': purged %d expired entries",
                self._name,
                len(expired_keys),
            )
            return len(expired_keys)

    # -- Stats ---------------------------------------------------------------

    def get_stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            self._stats.current_size = len(self._data)
            self._stats.max_size = self._maxsize
            # Return a copy
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                sets=self._stats.sets,
                current_size=self._stats.current_size,
                max_size=self._stats.max_size,
                stale_hits=self._stats.stale_hits,
                compressed_entries=self._stats.compressed_entries,
            )

    def reset_stats(self) -> None:
        """Reset statistics counters to zero."""
        with self._lock:
            self._stats = CacheStats(
                max_size=self._maxsize,
                current_size=len(self._data),
            )

    # -- Dunder helpers ------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __repr__(self) -> str:
        return (
            f"Cache(name={self._name!r}, maxsize={self._maxsize}, "
            f"ttl={self._default_ttl}, size={self.size})"
        )


# ---------------------------------------------------------------------------
# File-based persistent cache
# ---------------------------------------------------------------------------


class FileCache:
    """Simple file-based persistent cache backed by JSON files.

    Each entry is stored as a separate JSON file inside *directory*.
    Expired entries are cleaned up lazily on access or explicitly via
    :meth:`purge_expired`.

    Args:
        directory: Directory where cache files are stored.
        ttl: Default TTL in seconds for new entries. 0 means no expiration.
        name: Human-readable name for this cache.
    """

    _FILE_SUFFIX = ".cache.json"

    def __init__(
        self,
        directory: Path | str,
        ttl: float = 0.0,
        name: str = "file_cache",
    ) -> None:
        self._directory = Path(directory)
        self._default_ttl = ttl
        self._name = name
        self._lock = threading.Lock()
        self._stats = CacheStats()

    # -- Helpers -------------------------------------------------------------

    def _ensure_dir(self) -> None:
        """Create the cache directory if it doesn't exist."""
        self._directory.mkdir(parents=True, exist_ok=True)

    def _key_to_filename(self, key: str) -> Path:
        """Convert a cache key to a safe filename."""
        safe = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._directory / f"{safe}{self._FILE_SUFFIX}"

    def _read_entry(self, path: Path) -> Optional[dict[str, Any]]:
        """Read and parse a cache entry file."""
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _write_entry(self, path: Path, data: dict[str, Any]) -> None:
        """Write a cache entry to file."""
        self._ensure_dir()
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def _is_expired(entry_data: dict[str, Any]) -> bool:
        """Check whether a serialized entry has expired."""
        ttl = entry_data.get("ttl", 0)
        if ttl <= 0:
            return False
        created_at = entry_data.get("created_at", 0)
        return time.time() > (created_at + ttl)

    # -- Core operations -----------------------------------------------------

    @property
    def name(self) -> str:
        """Cache name."""
        return self._name

    @property
    def directory(self) -> Path:
        """Cache directory."""
        return self._directory

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a cached value.

        Args:
            key: Cache key.
            default: Returned on miss.

        Returns:
            The cached value or *default*.
        """
        with self._lock:
            path = self._key_to_filename(key)
            if not path.exists():
                self._stats.misses += 1
                return default

            entry_data = self._read_entry(path)
            if entry_data is None:
                self._stats.misses += 1
                return default

            if self._is_expired(entry_data):
                path.unlink(missing_ok=True)
                self._stats.expirations += 1
                self._stats.misses += 1
                return default

            self._stats.hits += 1
            return entry_data.get("value")

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value.

        The *value* must be JSON-serializable.

        Args:
            key: Cache key.
            value: JSON-serializable value.
            ttl: Optional per-entry TTL override.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl

        entry_data = {
            "key": key,
            "value": value,
            "created_at": time.time(),
            "ttl": effective_ttl,
        }

        with self._lock:
            path = self._key_to_filename(key)
            self._write_entry(path, entry_data)
            self._stats.sets += 1

    def delete(self, key: str) -> bool:
        """Remove a single entry.

        Args:
            key: Cache key to remove.

        Returns:
            True if the entry was removed.
        """
        with self._lock:
            path = self._key_to_filename(key)
            if path.exists():
                path.unlink()
                return True
            return False

    def has(self, key: str) -> bool:
        """Check whether *key* is cached and not expired."""
        with self._lock:
            path = self._key_to_filename(key)
            if not path.exists():
                return False
            entry_data = self._read_entry(path)
            if entry_data is None:
                return False
            if self._is_expired(entry_data):
                path.unlink(missing_ok=True)
                self._stats.expirations += 1
                return False
            return True

    def clear(self) -> int:
        """Remove all entries.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            count = 0
            if self._directory.exists():
                for path in self._directory.glob(f"*{self._FILE_SUFFIX}"):
                    path.unlink(missing_ok=True)
                    count += 1
            logger.debug("FileCache '%s': cleared %d entries", self._name, count)
            return count

    def purge_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of expired entries removed.
        """
        with self._lock:
            removed = 0
            if not self._directory.exists():
                return 0
            for path in self._directory.glob(f"*{self._FILE_SUFFIX}"):
                entry_data = self._read_entry(path)
                if entry_data is not None and self._is_expired(entry_data):
                    path.unlink(missing_ok=True)
                    self._stats.expirations += 1
                    removed += 1
            return removed

    def get_stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        with self._lock:
            size = 0
            if self._directory.exists():
                size = sum(
                    1 for _ in self._directory.glob(f"*{self._FILE_SUFFIX}")
                )
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                sets=self._stats.sets,
                current_size=size,
                max_size=0,
            )

    def __repr__(self) -> str:
        return (
            f"FileCache(name={self._name!r}, "
            f"directory={self._directory!r}, ttl={self._default_ttl})"
        )


# ---------------------------------------------------------------------------
# Decorator for function-level caching
# ---------------------------------------------------------------------------


def _make_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Build a deterministic cache key from function signature and arguments."""
    parts = [func.__module__, func.__qualname__]
    for arg in args:
        parts.append(repr(arg))
    for k, v in sorted(kwargs.items()):
        parts.append(f"{k}={v!r}")
    return ":".join(parts)


def cached(
    ttl: float = 0.0,
    maxsize: int = 128,
    cache: Optional[Cache] = None,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable:
    """Decorator that caches function return values.

    Args:
        ttl: Time-to-live for cached results in seconds. 0 = no expiry.
        maxsize: Maximum cache entries (ignored if *cache* is provided).
        cache: An existing :class:`Cache` to use. A new one is created
            if not provided.
        key_func: Optional function ``(func, args, kwargs) -> str`` that
            produces the cache key. Defaults to :func:`_make_cache_key`.

    Returns:
        Decorated function with caching behavior.

    Example::

        @cached(ttl=60)
        def load_heavy_data(path):
            ...

        # Manual cache control:
        load_heavy_data.cache_clear()
        stats = load_heavy_data.cache_stats()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        _cache = cache if cache is not None else Cache(
            maxsize=maxsize, ttl=ttl, name=f"cached:{func.__qualname__}"
        )
        _key_func = key_func or _make_cache_key

        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = _key_func(func, args, kwargs)
            result = _cache.get(key, _SENTINEL)
            if result is not _SENTINEL:
                return result
            result = func(*args, **kwargs)
            _cache.set(key, result)
            return result

        # Attach cache management helpers
        wrapper.cache = _cache  # type: ignore[attr-defined]
        wrapper.cache_clear = _cache.clear  # type: ignore[attr-defined]
        wrapper.cache_stats = _cache.get_stats  # type: ignore[attr-defined]
        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__

        return wrapper  # type: ignore[return-value]

    return decorator


# Sentinel for distinguishing "not in cache" from cached ``None``
_SENTINEL = object()


# ---------------------------------------------------------------------------
# Global / default cache registry
# ---------------------------------------------------------------------------

_registry_lock = threading.Lock()
_cache_registry: dict[str, Cache] = {}


def get_cache(name: str, maxsize: int = 256, ttl: float = 0.0) -> Cache:
    """Get or create a named cache from the global registry.

    If a cache with *name* already exists it is returned as-is; otherwise a
    new :class:`Cache` is created with the given parameters and registered.

    Args:
        name: Cache name.
        maxsize: Maximum entries (only used on creation).
        ttl: Default TTL in seconds (only used on creation).

    Returns:
        The named cache instance.
    """
    with _registry_lock:
        if name not in _cache_registry:
            _cache_registry[name] = Cache(maxsize=maxsize, ttl=ttl, name=name)
        return _cache_registry[name]


def get_default_cache() -> Cache:
    """Get the default application-wide cache.

    Returns:
        The default ``Cache`` instance (maxsize=512, ttl=300s).
    """
    return get_cache("gateway_default", maxsize=512, ttl=300)


def list_caches() -> dict[str, Cache]:
    """Return all registered caches.

    Returns:
        Dictionary mapping cache name to Cache instance.
    """
    with _registry_lock:
        return dict(_cache_registry)


def clear_all_caches() -> dict[str, int]:
    """Clear every registered cache.

    Returns:
        Dictionary mapping cache name to number of entries removed.
    """
    with _registry_lock:
        results = {}
        for name, c in _cache_registry.items():
            results[name] = c.clear()
        return results


def remove_cache(name: str) -> bool:
    """Remove a cache from the registry.

    Args:
        name: Cache name to remove.

    Returns:
        True if the cache was found and removed.
    """
    with _registry_lock:
        if name in _cache_registry:
            _cache_registry[name].clear()
            del _cache_registry[name]
            return True
        return False


def reset_registry() -> None:
    """Clear the entire cache registry. Primarily for testing."""
    with _registry_lock:
        for c in _cache_registry.values():
            c.clear()
        _cache_registry.clear()


# ---------------------------------------------------------------------------
# Pre-built caches for gateway subsystems
# ---------------------------------------------------------------------------


def get_config_cache() -> Cache:
    """Cache for parsed configuration objects.

    Uses a short TTL so file changes are picked up quickly.

    Returns:
        A Cache tuned for config loading (maxsize=32, ttl=30s).
    """
    return get_cache("config", maxsize=32, ttl=30)


def get_provider_cache() -> Cache:
    """Cache for built-in provider lookups.

    Built-in providers are static so TTL is long.

    Returns:
        A Cache tuned for provider lookups (maxsize=64, ttl=600s).
    """
    return get_cache("providers", maxsize=64, ttl=600)


# ---------------------------------------------------------------------------
# Compression utilities
# ---------------------------------------------------------------------------

import gzip as _gzip
import json as _json

_COMPRESSION_THRESHOLD = 1024  # bytes — compress values larger than this


def compress_value(value: Any) -> tuple[Any, bool]:
    """Compress a value if it's large enough to benefit from compression.

    Args:
        value: The value to potentially compress.

    Returns:
        Tuple of (possibly_compressed_value, was_compressed).
    """
    try:
        serialized = _json.dumps(value).encode("utf-8")
    except (TypeError, ValueError):
        return value, False

    if len(serialized) < _COMPRESSION_THRESHOLD:
        return value, False

    import base64 as _base64

    compressed = _gzip.compress(serialized)
    return {
        "__compressed__": True,
        "data": _base64.b64encode(compressed).decode("ascii"),
    }, True


def decompress_value(value: Any) -> Any:
    """Decompress a potentially compressed value.

    Args:
        value: The value to decompress.

    Returns:
        The decompressed value, or the original value if not compressed.
    """
    if not isinstance(value, dict) or "__compressed__" not in value:
        return value

    import base64 as _base64

    compressed_data = _base64.b64decode(value["data"])
    decompressed = _gzip.decompress(compressed_data)
    return _json.loads(decompressed.decode("utf-8"))


# ---------------------------------------------------------------------------
# Cache warmup
# ---------------------------------------------------------------------------


@dataclass
class WarmupEntry:
    """A single cache warmup entry.

    Attributes:
        key: The cache key.
        loader: Callable that returns the value for this key.
        ttl: Optional TTL override for this entry.
    """

    key: str
    loader: Callable[[], Any]
    ttl: Optional[float] = None


class CacheWarmer:
    """Pre-populates a cache with known values.

    Args:
        name: Optional name for this warmer.
    """

    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._entries: list[WarmupEntry] = []
        self._lock = threading.Lock()

    def add(
        self,
        key: str,
        loader: Callable[[], Any],
        ttl: Optional[float] = None,
    ) -> None:
        """Add an entry to warm up.

        Args:
            key: Cache key.
            loader: Callable that returns the value for this key.
            ttl: Optional TTL override.
        """
        with self._lock:
            self._entries.append(WarmupEntry(key=key, loader=loader, ttl=ttl))

    def remove(self, key: str) -> bool:
        """Remove an entry by key.

        Args:
            key: The key to remove.

        Returns:
            True if the entry was found and removed.
        """
        with self._lock:
            for i, entry in enumerate(self._entries):
                if entry.key == key:
                    del self._entries[i]
                    return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()

    @property
    def entries(self) -> list:
        """Return a copy of the entries list."""
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:
        return f"CacheWarmer(name={self._name!r}, entries={len(self)})"

    def warmup(
        self,
        cache: Cache,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> dict[str, bool]:
        """Populate the cache with all registered entries.

        Args:
            cache: The cache to warm.
            parallel: Whether to load entries in parallel using a thread pool.
            max_workers: Number of worker threads for parallel warmup.

        Returns:
            Dict mapping key -> True (success) or False (failed).
        """
        with self._lock:
            entries = list(self._entries)

        if not entries:
            return {}

        def _load_one(entry: WarmupEntry) -> tuple[str, bool]:
            try:
                value = entry.loader()
                cache.set(entry.key, value, ttl=entry.ttl)
                return entry.key, True
            except Exception:
                return entry.key, False

        if parallel:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_load_one, entries))
        else:
            results = [_load_one(e) for e in entries]

        return dict(results)


# ---------------------------------------------------------------------------
# Background purger
# ---------------------------------------------------------------------------


class BackgroundPurger:
    """Periodically purges expired entries from registered caches.

    Args:
        interval: Seconds between purge cycles.
        name: Optional name for this purger.
    """

    def __init__(self, interval: float = 60.0, name: str = "default") -> None:
        self._interval = interval
        self._name = name
        self._caches: list[Cache] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def interval(self) -> float:
        """Purge interval in seconds."""
        return self._interval

    @property
    def is_running(self) -> bool:
        """Whether the background thread is running."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def add_cache(self, cache: Cache) -> None:
        """Register a cache to be purged.

        Args:
            cache: The cache to register.
        """
        with self._lock:
            if cache not in self._caches:
                self._caches.append(cache)

    def remove_cache(self, cache: Cache) -> bool:
        """Unregister a cache.

        Args:
            cache: The cache to unregister.

        Returns:
            True if it was registered and has now been removed.
        """
        with self._lock:
            if cache in self._caches:
                self._caches.remove(cache)
                return True
            return False

    def purge_now(self) -> dict[str, int]:
        """Immediately purge all registered caches.

        Returns:
            Dict mapping cache name to number of expired entries removed.
        """
        with self._lock:
            caches = list(self._caches)
        results = {}
        for cache in caches:
            results[cache.name] = cache.purge_expired()
        return results

    def start(self) -> None:
        """Start the background purge thread (no-op if already running)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name=f"cache-purger-{self._name}",
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the background purge thread (no-op if not running)."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=self._interval + 1)
            with self._lock:
                self._thread = None

    def _run(self) -> None:
        """Background thread main loop."""
        while not self._stop_event.wait(timeout=self._interval):
            self.purge_now()

    def __repr__(self) -> str:
        return (
            f"BackgroundPurger(name={self._name!r}, "
            f"interval={self._interval}, running={self.is_running})"
        )


# ---------------------------------------------------------------------------
# Tiered cache (L1 in-memory + L2 file-based)
# ---------------------------------------------------------------------------


class TieredCache:
    """Two-tier cache: fast in-memory L1 + persistent file-based L2.

    On a miss in L1, the value is fetched from L2 and promoted to L1.

    Args:
        l1_maxsize: Maximum entries for the L1 (in-memory) cache.
        l1_ttl: TTL in seconds for the L1 cache.
        l2_directory: Directory for the L2 (file) cache.
        l2_ttl: TTL in seconds for the L2 cache.
        name: Optional name for this tiered cache.
        enable_compression: Whether to compress large values in L2.
    """

    def __init__(
        self,
        l1_maxsize: int = 256,
        l1_ttl: float = 0.0,
        l2_directory: Optional[Path] = None,
        l2_ttl: float = 0.0,
        name: str = "tiered",
        enable_compression: bool = False,
    ) -> None:
        self._name = name
        self._enable_compression = enable_compression
        self.l1 = Cache(maxsize=l1_maxsize, ttl=l1_ttl, name=f"{name}.l1")
        if l2_directory is None:
            import tempfile

            l2_directory = Path(tempfile.mkdtemp())
        self.l2 = FileCache(
            directory=l2_directory, ttl=l2_ttl, name=f"{name}.l2"
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value, checking L1 then L2 (and promoting on L2 hit).

        Args:
            key: Cache key.
            default: Value returned if key is absent in both tiers.

        Returns:
            The cached value or *default*.
        """
        value = self.l1.get(key)
        if value is not None:
            return value

        # Try L2
        value = self.l2.get(key)
        if value is not None:
            # Decompress if needed
            if self._enable_compression:
                value = decompress_value(value)
            # Promote to L1
            self.l1.set(key, value)
            return value

        return default

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value in both tiers.

        Args:
            key: Cache key.
            value: Value to store.
            ttl: Optional per-entry TTL override.
        """
        self.l1.set(key, value, ttl=ttl)
        # Compress for L2 if enabled
        l2_value = value
        if self._enable_compression:
            compressed_val, was_compressed = compress_value(value)
            if was_compressed:
                l2_value = compressed_val
        self.l2.set(key, l2_value, ttl=ttl)

    def delete(self, key: str) -> bool:
        """Remove a key from both tiers.

        Args:
            key: Cache key to remove.

        Returns:
            True if the key was present in at least one tier.
        """
        r1 = self.l1.delete(key)
        r2 = self.l2.delete(key)
        return r1 or r2

    def has(self, key: str) -> bool:
        """Check if key exists in either tier.

        Args:
            key: Cache key to check.

        Returns:
            True if the key is present (and not expired) in either tier.
        """
        return self.l1.has(key) or self.l2.has(key)

    def clear(self) -> int:
        """Clear both tiers.

        Returns:
            Total number of entries removed across both tiers.
        """
        return self.l1.clear() + self.l2.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return stats for both tiers.

        Returns:
            Dict with 'name', 'l1', and 'l2' keys.
        """
        return {
            "name": self._name,
            "l1": self.l1.get_stats().to_dict(),
            "l2": self.l2.get_stats().to_dict(),
        }

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __repr__(self) -> str:
        return f"TieredCache(name={self._name!r}, l1_size={self.l1.size})"


# ---------------------------------------------------------------------------
# Global background purger
# ---------------------------------------------------------------------------

_global_purger: Optional[BackgroundPurger] = None
_global_purger_lock = threading.Lock()


def get_background_purger(interval: float = 60.0) -> BackgroundPurger:
    """Get or create the global background purger.

    Args:
        interval: Purge interval in seconds (only used on first call).

    Returns:
        The global BackgroundPurger instance.
    """
    global _global_purger
    with _global_purger_lock:
        if _global_purger is None:
            _global_purger = BackgroundPurger(interval=interval, name="global")
        return _global_purger


def stop_background_purger() -> None:
    """Stop the global background purger if running."""
    global _global_purger
    with _global_purger_lock:
        purger = _global_purger
        _global_purger = None
    if purger is not None:
        purger.stop()


def get_response_cache() -> Any:
    """Get the global response cache.

    Returns:
        The global ResponseCache instance.
    """
    from src.response_cache import get_response_cache as _get_rc

    return _get_rc()
