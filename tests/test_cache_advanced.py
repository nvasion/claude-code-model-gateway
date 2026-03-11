"""Tests for advanced caching features.

Covers TieredCache, stale-while-revalidate, CacheWarmer,
BackgroundPurger, compression utilities, ResponseCache,
CachingInterceptor, and new CLI commands.
"""

import gzip
import json
import threading
import time

import pytest
from click.testing import CliRunner

from src.cache import (
    BackgroundPurger,
    Cache,
    CacheStats,
    CacheWarmer,
    TieredCache,
    WarmupEntry,
    compress_value,
    decompress_value,
    get_background_purger,
    get_cache,
    get_response_cache,
    list_caches,
    reset_registry,
    stop_background_purger,
)
from src.cli import main
from src.response_cache import (
    CachedResponse,
    ResponseCache,
    ResponseCacheStats,
    get_response_cache as get_global_response_cache,
    reset_response_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset global state before and after each test."""
    reset_registry()
    reset_response_cache()
    stop_background_purger()
    yield
    reset_registry()
    reset_response_cache()
    stop_background_purger()


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


# ---------------------------------------------------------------------------
# Stale-while-revalidate tests
# ---------------------------------------------------------------------------


class TestStaleWhileRevalidate:
    """Tests for the stale-while-revalidate feature."""

    def test_stale_hit_after_ttl(self):
        """Entry is served stale after TTL but within stale_ttl."""
        c = Cache(maxsize=10, ttl=0.1, stale_ttl=5.0, name="swr_test")
        c.set("k", "v")
        assert c.get("k") == "v"

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should still get stale value
        result = c.get("k")
        assert result == "v"

        stats = c.get_stats()
        assert stats.stale_hits >= 1

    def test_stale_miss_after_stale_ttl(self):
        """Entry is not served after both TTL and stale_ttl expire."""
        c = Cache(maxsize=10, ttl=0.05, stale_ttl=0.1, name="swr_expired")
        c.set("k", "v")

        # Wait for both TTL and stale_ttl to expire
        time.sleep(0.2)

        result = c.get("k")
        assert result is None

    def test_stale_triggers_refresh_callback(self):
        """Background refresh is triggered when serving stale data."""
        refreshed = threading.Event()
        new_value = "refreshed_value"

        def refresh_fn(key):
            refreshed.set()
            return new_value

        c = Cache(
            maxsize=10,
            ttl=0.1,
            stale_ttl=5.0,
            name="swr_refresh",
            refresh_callback=refresh_fn,
        )
        c.set("k", "original")
        time.sleep(0.15)

        # Should get stale value
        result = c.get("k")
        assert result == "original"

        # Wait for refresh to complete
        assert refreshed.wait(timeout=2.0)
        time.sleep(0.05)  # Let set() complete

        # Now should get the refreshed value
        result = c.get("k")
        assert result == new_value

    def test_no_stale_without_stale_ttl(self):
        """Without stale_ttl, expired entries are misses."""
        c = Cache(maxsize=10, ttl=0.1, stale_ttl=0, name="no_swr")
        c.set("k", "v")
        time.sleep(0.15)

        result = c.get("k")
        assert result is None

    def test_stale_stats_tracking(self):
        """Stale hits are tracked separately in stats."""
        c = Cache(maxsize=10, ttl=0.1, stale_ttl=5.0, name="swr_stats")
        c.set("k", "v")
        c.get("k")  # regular hit
        time.sleep(0.15)
        c.get("k")  # stale hit

        stats = c.get_stats()
        assert stats.hits == 2  # both count as hits
        assert stats.stale_hits == 1

    def test_stale_ttl_property(self):
        """stale_ttl property is exposed."""
        c = Cache(maxsize=10, ttl=60, stale_ttl=300)
        assert c.stale_ttl == 300

    def test_refresh_only_once_per_key(self):
        """Multiple stale hits for the same key only trigger one refresh."""
        call_count = 0

        def refresh_fn(key):
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate slow refresh
            return "refreshed"

        c = Cache(
            maxsize=10,
            ttl=0.05,
            stale_ttl=5.0,
            name="swr_dedup",
            refresh_callback=refresh_fn,
        )
        c.set("k", "original")
        time.sleep(0.1)

        # Multiple rapid accesses
        for _ in range(5):
            c.get("k")

        time.sleep(0.3)  # Let refresh complete
        assert call_count == 1

    def test_refresh_failure_doesnt_crash(self):
        """Failed refresh callback doesn't crash the cache."""

        def failing_refresh(key):
            raise ValueError("Refresh failed!")

        c = Cache(
            maxsize=10,
            ttl=0.1,
            stale_ttl=5.0,
            name="swr_fail",
            refresh_callback=failing_refresh,
        )
        c.set("k", "v")
        time.sleep(0.15)

        # Should still serve stale
        result = c.get("k")
        assert result == "v"
        time.sleep(0.1)  # Let failed refresh complete


# ---------------------------------------------------------------------------
# Compression tests
# ---------------------------------------------------------------------------


class TestCompression:
    """Tests for cache value compression."""

    def test_compress_small_value_unchanged(self):
        """Small values are not compressed."""
        value = {"key": "small"}
        result, compressed = compress_value(value)
        assert not compressed
        assert result == value

    def test_compress_large_value(self):
        """Large JSON-serializable values are compressed."""
        # Create a value larger than the threshold
        value = {"data": "x" * 2000}
        result, compressed = compress_value(value)
        assert compressed
        assert "__compressed__" in result

    def test_decompress_roundtrip(self):
        """Compress then decompress returns original value."""
        value = {"data": "x" * 2000, "list": [1, 2, 3], "nested": {"a": "b"}}
        compressed_val, was_compressed = compress_value(value)
        assert was_compressed

        decompressed = decompress_value(compressed_val)
        assert decompressed == value

    def test_decompress_uncompressed_value(self):
        """Decompressing a non-compressed value returns it as-is."""
        value = {"normal": "dict"}
        result = decompress_value(value)
        assert result == value

    def test_decompress_non_dict(self):
        """Decompressing a non-dict returns it as-is."""
        assert decompress_value("string") == "string"
        assert decompress_value(42) == 42
        assert decompress_value(None) is None

    def test_compress_non_serializable(self):
        """Non-JSON-serializable values are returned unchanged."""
        value = {"set": {1, 2, 3}}  # sets are not JSON-serializable
        result, compressed = compress_value(value)
        assert not compressed
        assert result == value


# ---------------------------------------------------------------------------
# TieredCache tests
# ---------------------------------------------------------------------------


class TestTieredCache:
    """Tests for the two-tier cache."""

    def test_set_and_get(self, tmp_path):
        """Basic set/get round-trip through both tiers."""
        tc = TieredCache(
            l1_maxsize=10,
            l1_ttl=60,
            l2_directory=tmp_path / "l2",
            l2_ttl=600,
            name="test_tiered",
        )
        tc.set("k", "v")
        assert tc.get("k") == "v"

    def test_l2_promotion(self, tmp_path):
        """L2 hit promotes value to L1."""
        tc = TieredCache(
            l1_maxsize=10,
            l1_ttl=60,
            l2_directory=tmp_path / "l2",
            l2_ttl=600,
            name="promo_test",
        )
        tc.set("k", "v")

        # Clear L1 to force L2 lookup
        tc.l1.clear()
        assert tc.l1.get("k") is None

        # Get should find in L2 and promote to L1
        result = tc.get("k")
        assert result == "v"
        assert tc.l1.get("k") == "v"

    def test_miss_returns_default(self, tmp_path):
        """Cache miss returns default value."""
        tc = TieredCache(
            l1_maxsize=10,
            l2_directory=tmp_path / "l2",
            name="miss_test",
        )
        assert tc.get("missing") is None
        assert tc.get("missing", "fallback") == "fallback"

    def test_delete_both_tiers(self, tmp_path):
        """Delete removes from both L1 and L2."""
        tc = TieredCache(
            l1_maxsize=10,
            l2_directory=tmp_path / "l2",
            name="del_test",
        )
        tc.set("k", "v")
        assert tc.delete("k") is True
        assert tc.get("k") is None

    def test_has(self, tmp_path):
        """has() checks both tiers."""
        tc = TieredCache(
            l1_maxsize=10,
            l2_directory=tmp_path / "l2",
            name="has_test",
        )
        tc.set("k", "v")
        assert tc.has("k") is True
        assert tc.has("missing") is False

        # Clear L1, should still find in L2
        tc.l1.clear()
        assert tc.has("k") is True

    def test_clear_both_tiers(self, tmp_path):
        """clear() removes entries from both tiers."""
        tc = TieredCache(
            l1_maxsize=10,
            l2_directory=tmp_path / "l2",
            name="clear_test",
        )
        tc.set("a", 1)
        tc.set("b", 2)
        count = tc.clear()
        assert count >= 2
        assert tc.get("a") is None

    def test_get_stats(self, tmp_path):
        """get_stats returns stats for both tiers."""
        tc = TieredCache(
            l1_maxsize=10,
            l2_directory=tmp_path / "l2",
            name="stats_test",
        )
        tc.set("k", "v")
        tc.get("k")

        stats = tc.get_stats()
        assert "l1" in stats
        assert "l2" in stats
        assert stats["name"] == "stats_test"

    def test_contains_operator(self, tmp_path):
        """'in' operator works."""
        tc = TieredCache(
            l1_maxsize=10,
            l2_directory=tmp_path / "l2",
            name="in_test",
        )
        tc.set("k", "v")
        assert "k" in tc
        assert "missing" not in tc

    def test_repr(self, tmp_path):
        """repr is informative."""
        tc = TieredCache(
            l1_maxsize=10,
            l2_directory=tmp_path / "l2",
            name="repr_test",
        )
        r = repr(tc)
        assert "repr_test" in r
        assert "TieredCache" in r

    def test_compression_in_l2(self, tmp_path):
        """Large values are compressed in L2."""
        tc = TieredCache(
            l1_maxsize=10,
            l2_directory=tmp_path / "l2",
            name="compress_test",
            enable_compression=True,
        )
        large_value = {"data": "x" * 5000}
        tc.set("big", large_value)

        # Clear L1, get from L2 (which should be compressed)
        tc.l1.clear()
        result = tc.get("big")
        assert result == large_value

    def test_complex_values(self, tmp_path):
        """Complex nested values survive tier round-trips."""
        tc = TieredCache(
            l1_maxsize=10,
            l2_directory=tmp_path / "l2",
            name="complex_test",
        )
        value = {
            "list": [1, 2, [3, 4]],
            "nested": {"a": {"b": "c"}},
            "number": 42.5,
            "null": None,
            "bool": True,
        }
        tc.set("complex", value)

        # Clear L1, get from L2
        tc.l1.clear()
        result = tc.get("complex")
        assert result == value


# ---------------------------------------------------------------------------
# CacheWarmer tests
# ---------------------------------------------------------------------------


class TestCacheWarmer:
    """Tests for the cache warmer."""

    def test_basic_warmup(self):
        """Warmer populates cache with loader results."""
        cache = Cache(maxsize=10, ttl=60, name="warm_test")
        warmer = CacheWarmer(name="test")
        warmer.add("k1", lambda: "v1")
        warmer.add("k2", lambda: 42)

        results = warmer.warmup(cache)
        assert results == {"k1": True, "k2": True}
        assert cache.get("k1") == "v1"
        assert cache.get("k2") == 42

    def test_parallel_warmup(self):
        """Parallel warmup works correctly."""
        cache = Cache(maxsize=100, ttl=60, name="parallel_warm")
        warmer = CacheWarmer(name="parallel")

        for i in range(10):
            warmer.add(f"key_{i}", lambda i=i: f"value_{i}")

        results = warmer.warmup(cache, parallel=True, max_workers=4)
        assert all(results.values())
        assert len(results) == 10
        for i in range(10):
            assert cache.get(f"key_{i}") == f"value_{i}"

    def test_warmup_with_failure(self):
        """Failed loaders are tracked without crashing."""
        cache = Cache(maxsize=10, ttl=60, name="fail_warm")
        warmer = CacheWarmer(name="fail_test")
        warmer.add("good", lambda: "ok")
        warmer.add("bad", lambda: 1 / 0)  # will raise

        results = warmer.warmup(cache)
        assert results["good"] is True
        assert results["bad"] is False
        assert cache.get("good") == "ok"

    def test_warmup_with_custom_ttl(self):
        """Per-entry TTL is respected."""
        cache = Cache(maxsize=10, ttl=600, name="ttl_warm")
        warmer = CacheWarmer()
        warmer.add("short", lambda: "v", ttl=0.1)
        warmer.add("long", lambda: "v", ttl=600)

        warmer.warmup(cache)
        time.sleep(0.2)
        assert cache.get("short") is None
        assert cache.get("long") == "v"

    def test_remove_entry(self):
        """Entries can be removed before warmup."""
        warmer = CacheWarmer()
        warmer.add("k1", lambda: 1)
        warmer.add("k2", lambda: 2)
        assert warmer.remove("k1") is True
        assert warmer.remove("nonexistent") is False
        assert len(warmer) == 1

    def test_clear_entries(self):
        """clear removes all entries."""
        warmer = CacheWarmer()
        warmer.add("k1", lambda: 1)
        warmer.clear()
        assert len(warmer) == 0

    def test_empty_warmup(self):
        """Warming with no entries returns empty dict."""
        cache = Cache(maxsize=10, name="empty_warm")
        warmer = CacheWarmer()
        results = warmer.warmup(cache)
        assert results == {}

    def test_entries_property(self):
        """entries property returns a copy."""
        warmer = CacheWarmer()
        warmer.add("k1", lambda: 1)
        entries = warmer.entries
        assert len(entries) == 1
        assert entries[0].key == "k1"

    def test_repr(self):
        """repr is informative."""
        warmer = CacheWarmer(name="my_warmer")
        warmer.add("k", lambda: 1)
        r = repr(warmer)
        assert "my_warmer" in r
        assert "1" in r


# ---------------------------------------------------------------------------
# BackgroundPurger tests
# ---------------------------------------------------------------------------


class TestBackgroundPurger:
    """Tests for the background purger."""

    def test_purge_now(self):
        """purge_now immediately purges expired entries."""
        cache = Cache(maxsize=10, ttl=0.1, name="purge_now_test")
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        time.sleep(0.15)

        purger = BackgroundPurger(interval=60)
        purger.add_cache(cache)
        results = purger.purge_now()
        assert results["purge_now_test"] == 2

    def test_start_stop(self):
        """Purger can start and stop cleanly."""
        purger = BackgroundPurger(interval=0.1)
        cache = Cache(maxsize=10, ttl=0.05, name="start_stop")
        purger.add_cache(cache)

        cache.set("k", "v")
        purger.start()
        assert purger.is_running

        time.sleep(0.2)  # Let at least one purge cycle run

        purger.stop()
        assert not purger.is_running

    def test_periodic_purge(self):
        """Background thread periodically purges expired entries."""
        cache = Cache(maxsize=100, ttl=0.05, name="periodic_purge")
        purger = BackgroundPurger(interval=0.1, name="periodic")
        purger.add_cache(cache)

        cache.set("k1", "v1")
        purger.start()
        time.sleep(0.3)  # Wait for purge cycle
        purger.stop()

        # Expired entries should have been purged
        stats = cache.get_stats()
        assert stats.expirations >= 1

    def test_add_remove_cache(self):
        """Caches can be added and removed."""
        purger = BackgroundPurger()
        c1 = Cache(maxsize=10, name="c1")
        c2 = Cache(maxsize=10, name="c2")

        purger.add_cache(c1)
        purger.add_cache(c2)
        assert purger.remove_cache(c1) is True
        assert purger.remove_cache(c1) is False  # already removed

    def test_double_start(self):
        """Starting an already-started purger is a no-op."""
        purger = BackgroundPurger(interval=1)
        purger.start()
        purger.start()  # should not create a second thread
        assert purger.is_running
        purger.stop()

    def test_double_stop(self):
        """Stopping an already-stopped purger is safe."""
        purger = BackgroundPurger(interval=1)
        purger.stop()  # no-op

    def test_interval_property(self):
        """interval property returns the configured interval."""
        purger = BackgroundPurger(interval=42.0)
        assert purger.interval == 42.0

    def test_repr(self):
        """repr is informative."""
        purger = BackgroundPurger(interval=30, name="my_purger")
        r = repr(purger)
        assert "my_purger" in r
        assert "30" in r

    def test_global_purger(self):
        """Global purger is accessible."""
        purger = get_background_purger()
        assert purger is not None
        assert not purger.is_running


# ---------------------------------------------------------------------------
# CacheStats extended fields tests
# ---------------------------------------------------------------------------


class TestCacheStatsExtended:
    """Tests for extended CacheStats fields."""

    def test_stale_hits_in_stats(self):
        """stale_hits field is present in stats."""
        stats = CacheStats(stale_hits=5)
        assert stats.stale_hits == 5
        d = stats.to_dict()
        assert d["stale_hits"] == 5

    def test_compressed_entries_in_stats(self):
        """compressed_entries field is present in stats."""
        stats = CacheStats(compressed_entries=3)
        assert stats.compressed_entries == 3
        d = stats.to_dict()
        assert d["compressed_entries"] == 3


# ---------------------------------------------------------------------------
# ResponseCache tests
# ---------------------------------------------------------------------------


class TestResponseCache:
    """Tests for the HTTP response cache."""

    def test_store_and_lookup(self):
        """Basic store/lookup round-trip."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        body = b'{"models": ["claude-3"]}'

        stored = rc.store(
            method="GET",
            path="/v1/models",
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=body,
        )
        assert stored is True

        cached = rc.lookup(method="GET", path="/v1/models")
        assert cached is not None
        assert cached.status_code == 200
        assert cached.body == body

    def test_miss_returns_none(self):
        """Lookup miss returns None."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        assert rc.lookup(method="GET", path="/missing") is None

    def test_non_cacheable_method(self):
        """POST requests are not cached by default."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        rc.store(
            method="POST",
            path="/v1/messages",
            status_code=200,
            headers={},
            body=b"{}",
        )
        assert rc.lookup(method="POST", path="/v1/messages") is None

    def test_custom_cacheable_methods(self):
        """Custom cacheable methods are respected."""
        rc = ResponseCache(
            maxsize=10,
            default_ttl=60,
            cacheable_methods={"GET", "POST"},
        )
        rc.store(
            method="POST",
            path="/v1/test",
            status_code=200,
            headers={},
            body=b"result",
        )
        cached = rc.lookup(method="POST", path="/v1/test")
        assert cached is not None

    def test_non_cacheable_status(self):
        """500 responses are not cached by default."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        stored = rc.store(
            method="GET",
            path="/error",
            status_code=500,
            headers={},
            body=b"error",
        )
        assert stored is False

    def test_cache_control_no_store(self):
        """Responses with Cache-Control: no-store are not cached."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        stored = rc.store(
            method="GET",
            path="/private",
            status_code=200,
            headers={"Cache-Control": "no-store"},
            body=b"secret",
        )
        assert stored is False

    def test_cache_control_no_cache_request(self):
        """Requests with Cache-Control: no-cache bypass cache."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        rc.store(
            method="GET",
            path="/v1/models",
            status_code=200,
            headers={},
            body=b"data",
        )

        # Request with no-cache should bypass
        cached = rc.lookup(
            method="GET",
            path="/v1/models",
            headers={"Cache-Control": "no-cache"},
        )
        assert cached is None

    def test_cache_control_max_age(self):
        """Response max-age overrides default TTL."""
        rc = ResponseCache(maxsize=10, default_ttl=600)
        rc.store(
            method="GET",
            path="/short",
            status_code=200,
            headers={"Cache-Control": "max-age=1"},
            body=b"data",
        )
        # Should be available immediately
        assert rc.lookup(method="GET", path="/short") is not None

    def test_invalidate(self):
        """invalidate removes a specific cached response."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        rc.store(
            method="GET",
            path="/v1/models",
            status_code=200,
            headers={},
            body=b"data",
        )
        assert rc.invalidate(method="GET", path="/v1/models") is True
        assert rc.lookup(method="GET", path="/v1/models") is None

    def test_invalidate_path(self):
        """invalidate_path removes all responses for a path."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        rc.store(
            method="GET",
            path="/v1/models",
            status_code=200,
            headers={},
            body=b"data",
        )
        removed = rc.invalidate_path("/v1/models")
        assert removed >= 1

    def test_clear(self):
        """clear removes all cached responses."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        rc.store(method="GET", path="/a", status_code=200, headers={}, body=b"a")
        rc.store(method="GET", path="/b", status_code=200, headers={}, body=b"b")
        count = rc.clear()
        assert count == 2

    def test_body_compression(self):
        """Large response bodies are compressed."""
        rc = ResponseCache(maxsize=10, default_ttl=60, enable_compression=True)
        large_body = b"x" * 10000
        rc.store(
            method="GET",
            path="/big",
            status_code=200,
            headers={},
            body=large_body,
        )

        cached = rc.lookup(method="GET", path="/big")
        assert cached is not None
        assert cached.body == large_body

        stats = rc.get_stats()
        assert stats.compressed_stores >= 1

    def test_stats_tracking(self):
        """Stats are properly tracked."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        rc.store(method="GET", path="/a", status_code=200, headers={}, body=b"a")
        rc.lookup(method="GET", path="/a")  # hit
        rc.lookup(method="GET", path="/b")  # miss
        rc.lookup(method="POST", path="/c")  # bypass

        stats = rc.get_stats()
        assert stats.stores == 1
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.bypasses == 1
        assert stats.lookups == 3

    def test_vary_headers(self):
        """Different vary header values produce different cache keys."""
        rc = ResponseCache(
            maxsize=10,
            default_ttl=60,
            vary_headers=["Accept-Language"],
        )
        rc.store(
            method="GET",
            path="/v1/models",
            status_code=200,
            headers={},
            body=b"english",
            request_headers={"Accept-Language": "en"},
        )
        rc.store(
            method="GET",
            path="/v1/models",
            status_code=200,
            headers={},
            body=b"french",
            request_headers={"Accept-Language": "fr"},
        )

        en = rc.lookup(
            method="GET",
            path="/v1/models",
            headers={"Accept-Language": "en"},
        )
        fr = rc.lookup(
            method="GET",
            path="/v1/models",
            headers={"Accept-Language": "fr"},
        )
        assert en is not None
        assert fr is not None
        assert en.body == b"english"
        assert fr.body == b"french"

    def test_size(self):
        """size and __len__ work."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        assert len(rc) == 0
        rc.store(method="GET", path="/a", status_code=200, headers={}, body=b"a")
        assert rc.size == 1
        assert len(rc) == 1

    def test_repr(self):
        """repr is informative."""
        rc = ResponseCache(maxsize=100, default_ttl=120, name="my_rc")
        r = repr(rc)
        assert "my_rc" in r
        assert "100" in r

    def test_reset_stats(self):
        """reset_stats clears counters."""
        rc = ResponseCache(maxsize=10, default_ttl=60)
        rc.store(method="GET", path="/a", status_code=200, headers={}, body=b"a")
        rc.lookup(method="GET", path="/a")
        rc.reset_stats()
        stats = rc.get_stats()
        assert stats.hits == 0
        assert stats.stores == 0


class TestCachedResponse:
    """Tests for the CachedResponse dataclass."""

    def test_age(self):
        """age returns seconds since creation."""
        cr = CachedResponse(
            status_code=200,
            headers={},
            body=b"test",
            created_at=time.time() - 5.0,
        )
        assert cr.age >= 5.0

    def test_is_expired(self):
        """is_expired checks TTL."""
        cr = CachedResponse(
            status_code=200,
            headers={},
            body=b"test",
            created_at=time.time() - 10.0,
            ttl=5.0,
        )
        assert cr.is_expired is True

    def test_not_expired(self):
        """Not expired within TTL."""
        cr = CachedResponse(
            status_code=200,
            headers={},
            body=b"test",
            ttl=600.0,
        )
        assert cr.is_expired is False

    def test_to_dict(self):
        """Serialization works."""
        cr = CachedResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b"test",
            request_method="GET",
            request_path="/v1/models",
            body_size=4,
        )
        d = cr.to_dict()
        assert d["status_code"] == 200
        assert d["request_method"] == "GET"


class TestResponseCacheStats:
    """Tests for ResponseCacheStats."""

    def test_hit_rate_zero(self):
        """Hit rate is 0 with no lookups."""
        stats = ResponseCacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Hit rate is calculated correctly."""
        stats = ResponseCacheStats(lookups=10, hits=7)
        assert stats.hit_rate == 70.0

    def test_to_dict(self):
        """Serialization includes all fields."""
        stats = ResponseCacheStats(lookups=5, hits=3, stores=2)
        d = stats.to_dict()
        assert d["lookups"] == 5
        assert d["hit_rate"] == 60.0


class TestGlobalResponseCache:
    """Tests for the global response cache singleton."""

    def test_get_global_response_cache(self):
        """get_response_cache returns a singleton."""
        rc1 = get_global_response_cache()
        rc2 = get_global_response_cache()
        assert rc1 is rc2

    def test_reset_response_cache(self):
        """reset_response_cache clears and resets."""
        rc = get_global_response_cache()
        rc.store(method="GET", path="/a", status_code=200, headers={}, body=b"a")
        reset_response_cache()
        rc2 = get_global_response_cache()
        assert rc2 is not rc
        assert rc2.size == 0


# ---------------------------------------------------------------------------
# CachingInterceptor tests
# ---------------------------------------------------------------------------


class TestCachingInterceptor:
    """Tests for the CachingInterceptor."""

    def test_cache_hit(self):
        """Interceptor returns cached response on hit."""
        from src.interceptor import CachingInterceptor, InterceptAction
        from src.response_cache import ResponseCache
        from src.router import RequestContext

        rc = ResponseCache(maxsize=10, default_ttl=60)
        rc.store(
            method="GET",
            path="/v1/models",
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"models": []}',
        )

        interceptor = CachingInterceptor(response_cache=rc)
        ctx = RequestContext(
            method="GET",
            path="/v1/models",
            headers={},
        )

        result = interceptor.intercept(ctx)
        assert result.metadata.get("cache_hit") is True
        assert result.metadata.get("cached_response") is not None

    def test_cache_miss(self):
        """Interceptor returns SKIP on miss."""
        from src.interceptor import CachingInterceptor, InterceptAction
        from src.response_cache import ResponseCache
        from src.router import RequestContext

        rc = ResponseCache(maxsize=10, default_ttl=60)
        interceptor = CachingInterceptor(response_cache=rc)
        ctx = RequestContext(
            method="GET",
            path="/v1/models",
            headers={},
        )

        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.SKIP
        assert result.metadata.get("cache_hit") is False

    def test_non_cacheable_method_skips(self):
        """POST requests skip caching."""
        from src.interceptor import CachingInterceptor, InterceptAction
        from src.response_cache import ResponseCache
        from src.router import RequestContext

        rc = ResponseCache(maxsize=10, default_ttl=60)
        interceptor = CachingInterceptor(response_cache=rc)
        ctx = RequestContext(
            method="POST",
            path="/v1/messages",
            headers={},
        )

        result = interceptor.intercept(ctx)
        assert result.action == InterceptAction.SKIP
        assert "cache_hit" not in result.metadata

    def test_store_response(self):
        """store_response stores a response in the cache."""
        from src.interceptor import CachingInterceptor
        from src.response_cache import ResponseCache

        rc = ResponseCache(maxsize=10, default_ttl=60)
        interceptor = CachingInterceptor(response_cache=rc)

        stored = interceptor.store_response(
            method="GET",
            path="/v1/models",
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b"data",
        )
        assert stored is True
        assert rc.size == 1

    def test_stats(self):
        """get_stats returns interceptor statistics."""
        from src.interceptor import CachingInterceptor
        from src.response_cache import ResponseCache
        from src.router import RequestContext

        rc = ResponseCache(maxsize=10, default_ttl=60)
        interceptor = CachingInterceptor(response_cache=rc)

        ctx = RequestContext(method="GET", path="/a", headers={})
        interceptor.intercept(ctx)  # miss

        stats = interceptor.get_stats()
        assert stats["misses"] == 1
        assert "response_cache" in stats


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------


class TestCacheCLIAdvanced:
    """Tests for advanced cache CLI commands."""

    def test_cache_info_no_caches(self, runner):
        """cache info with no active caches."""
        result = runner.invoke(main, ["cache", "info"])
        assert result.exit_code == 0
        assert "No active caches" in result.output

    def test_cache_info_with_data(self, runner):
        """cache info shows detailed information."""
        c = get_cache("test_info", maxsize=100, ttl=60)
        c.set("a", 1)

        result = runner.invoke(main, ["cache", "info"])
        assert result.exit_code == 0
        assert "test_info" in result.output
        assert "Max size:" in result.output
        assert "Default TTL:" in result.output
        assert "Stale TTL:" in result.output

    def test_cache_info_json(self, runner):
        """cache info --format json outputs valid JSON."""
        get_cache("json_info", maxsize=50, ttl=30)
        result = runner.invoke(main, ["cache", "info", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "caches" in data
        assert "json_info" in data["caches"]
        assert "stale_ttl" in data["caches"]["json_info"]

    def test_cache_warmup_providers(self, runner):
        """cache warmup --providers warms provider cache."""
        result = runner.invoke(main, ["cache", "warmup", "--providers"])
        assert result.exit_code == 0
        assert "Warming caches" in result.output
        assert "Warmup complete" in result.output

    def test_cache_warmup_no_targets(self, runner):
        """cache warmup with no targets shows message."""
        result = runner.invoke(main, ["cache", "warmup", "--no-providers"])
        assert result.exit_code == 0
        assert "Nothing to warm" in result.output

    def test_cache_response_stats(self, runner):
        """cache response-stats shows response cache statistics."""
        result = runner.invoke(main, ["cache", "response-stats"])
        assert result.exit_code == 0
        assert "Response Cache Statistics" in result.output
        assert "Lookups:" in result.output
        assert "Hit rate:" in result.output

    def test_cache_response_stats_json(self, runner):
        """cache response-stats --format json outputs valid JSON."""
        result = runner.invoke(main, ["cache", "response-stats", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "lookups" in data
        assert "hit_rate" in data

    def test_cache_warmup_help(self, runner):
        """cache warmup --help shows help."""
        result = runner.invoke(main, ["cache", "warmup", "--help"])
        assert result.exit_code == 0
        assert "Pre-populate caches" in result.output

    def test_cache_info_help(self, runner):
        """cache info --help shows help."""
        result = runner.invoke(main, ["cache", "info", "--help"])
        assert result.exit_code == 0
        assert "detailed cache information" in result.output


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestCachingIntegration:
    """Integration tests combining multiple caching features."""

    def test_warmer_with_tiered_cache(self, tmp_path):
        """CacheWarmer can populate a TieredCache's L1."""
        tc = TieredCache(
            l1_maxsize=10,
            l2_directory=tmp_path / "l2",
            name="warm_tiered",
        )
        warmer = CacheWarmer(name="tiered_warmer")
        warmer.add("k1", lambda: "v1")

        # Warm the L1 cache
        results = warmer.warmup(tc.l1)
        assert results["k1"] is True
        assert tc.get("k1") == "v1"

    def test_purger_with_stale_cache(self):
        """BackgroundPurger works with stale-while-revalidate cache."""
        cache = Cache(
            maxsize=10,
            ttl=0.05,
            stale_ttl=0.1,
            name="purge_stale",
        )
        cache.set("k", "v")

        purger = BackgroundPurger(interval=0.05)
        purger.add_cache(cache)

        time.sleep(0.2)  # Wait for stale to fully expire
        results = purger.purge_now()
        assert "purge_stale" in results

    def test_response_cache_with_registry(self):
        """Response cache's internal cache appears in stats."""
        rc = get_global_response_cache()
        rc.store(
            method="GET",
            path="/test",
            status_code=200,
            headers={},
            body=b"data",
        )

        # The response cache uses the global cache internally
        stats = rc.get_stats()
        assert stats.stores == 1

    def test_full_cache_lifecycle(self, tmp_path):
        """Full lifecycle: warm -> use -> stale -> purge."""
        # Set up
        cache = Cache(maxsize=100, ttl=0.2, stale_ttl=0.3, name="lifecycle")

        # 1. Warm
        warmer = CacheWarmer(name="lifecycle_warmer")
        warmer.add("k1", lambda: "warmed_v1")
        warmer.add("k2", lambda: "warmed_v2")
        warmer.warmup(cache)

        assert cache.get("k1") == "warmed_v1"
        assert cache.get("k2") == "warmed_v2"

        # 2. Use normally (hits)
        assert cache.get("k1") == "warmed_v1"
        stats = cache.get_stats()
        assert stats.hits >= 2

        # 3. TTL expires -> stale serving
        time.sleep(0.25)
        result = cache.get("k1")
        assert result == "warmed_v1"  # stale hit
        stats = cache.get_stats()
        assert stats.stale_hits >= 1

        # 4. Stale expires -> miss
        time.sleep(0.35)
        result = cache.get("k1")
        assert result is None

        # 5. Purge remaining
        purger = BackgroundPurger(interval=60)
        purger.add_cache(cache)
        purger.purge_now()
