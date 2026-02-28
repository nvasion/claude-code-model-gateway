"""Comprehensive tests for the caching mechanism."""

import time
import threading

import pytest
from click.testing import CliRunner

from src.cache import (
    Cache,
    CacheEntry,
    CacheStats,
    FileCache,
    cached,
    clear_all_caches,
    get_cache,
    get_config_cache,
    get_default_cache,
    get_provider_cache,
    list_caches,
    remove_cache,
    reset_registry,
)
from src.cli import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the global cache registry before and after each test."""
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def cache():
    """Create a fresh Cache instance."""
    return Cache(maxsize=10, ttl=0, name="test")


@pytest.fixture
def ttl_cache():
    """Create a Cache with a short TTL."""
    return Cache(maxsize=10, ttl=0.2, name="ttl_test")


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def file_cache(tmp_path):
    """Create a FileCache using a temporary directory."""
    return FileCache(directory=tmp_path / "cache", ttl=0, name="file_test")


# ---------------------------------------------------------------------------
# CacheStats tests
# ---------------------------------------------------------------------------


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_defaults(self):
        """Stats start at zero."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0
        assert stats.sets == 0
        assert stats.current_size == 0
        assert stats.max_size == 0

    def test_total_requests(self):
        """total_requests is hits + misses."""
        stats = CacheStats(hits=5, misses=3)
        assert stats.total_requests == 8

    def test_hit_rate_zero_requests(self):
        """Hit rate is 0.0 when there are no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Hit rate is computed correctly."""
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == 70.0

    def test_to_dict(self):
        """Serialization produces all expected fields."""
        stats = CacheStats(hits=10, misses=2, sets=12, current_size=5, max_size=100)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 2
        assert d["total_requests"] == 12
        assert d["hit_rate"] == pytest.approx(83.33, abs=0.01)
        assert d["current_size"] == 5
        assert d["max_size"] == 100


# ---------------------------------------------------------------------------
# CacheEntry tests
# ---------------------------------------------------------------------------


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_not_expired_without_ttl(self):
        """Entry with ttl=0 never expires."""
        entry = CacheEntry(key="k", value="v", ttl=0)
        assert not entry.is_expired
        assert entry.expires_at is None
        assert entry.remaining_ttl is None

    def test_expired_with_short_ttl(self):
        """Entry expires after TTL elapses."""
        entry = CacheEntry(key="k", value="v", created_at=time.time() - 1, ttl=0.5)
        assert entry.is_expired

    def test_not_yet_expired(self):
        """Entry is not expired when TTL hasn't elapsed."""
        entry = CacheEntry(key="k", value="v", ttl=60)
        assert not entry.is_expired
        assert entry.remaining_ttl > 0
        assert entry.expires_at is not None

    def test_touch(self):
        """Touch increments access_count and updates last_accessed."""
        entry = CacheEntry(key="k", value="v")
        assert entry.access_count == 0
        before = entry.last_accessed
        time.sleep(0.01)
        entry.touch()
        assert entry.access_count == 1
        assert entry.last_accessed >= before


# ---------------------------------------------------------------------------
# Cache (in-memory LRU) tests
# ---------------------------------------------------------------------------


class TestCache:
    """Tests for the in-memory Cache."""

    def test_set_and_get(self, cache):
        """Basic set/get round-trip."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing(self, cache):
        """Get returns default on miss."""
        assert cache.get("missing") is None
        assert cache.get("missing", "fallback") == "fallback"

    def test_overwrite(self, cache):
        """Set overwrites existing entries."""
        cache.set("k", 1)
        cache.set("k", 2)
        assert cache.get("k") == 2

    def test_delete(self, cache):
        """Delete removes an entry."""
        cache.set("k", "v")
        assert cache.delete("k") is True
        assert cache.get("k") is None

    def test_delete_missing(self, cache):
        """Delete returns False for missing keys."""
        assert cache.delete("missing") is False

    def test_has(self, cache):
        """has() returns True for existing keys."""
        cache.set("k", "v")
        assert cache.has("k") is True
        assert cache.has("missing") is False

    def test_clear(self, cache):
        """clear() removes all entries."""
        cache.set("a", 1)
        cache.set("b", 2)
        count = cache.clear()
        assert count == 2
        assert cache.size == 0

    def test_keys(self, cache):
        """keys() returns non-expired keys."""
        cache.set("a", 1)
        cache.set("b", 2)
        assert sorted(cache.keys()) == ["a", "b"]

    def test_len(self, cache):
        """__len__ reports size."""
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2

    def test_contains(self, cache):
        """__contains__ (in) operator works."""
        cache.set("a", 1)
        assert "a" in cache
        assert "b" not in cache

    def test_repr(self, cache):
        """repr is informative."""
        r = repr(cache)
        assert "test" in r
        assert "maxsize=10" in r

    # -- LRU eviction --------------------------------------------------------

    def test_lru_eviction(self):
        """Oldest entry is evicted when maxsize is exceeded."""
        c = Cache(maxsize=3, name="lru_test")
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.set("d", 4)  # should evict "a"
        assert c.get("a") is None
        assert c.get("d") == 4
        assert c.size == 3

    def test_lru_access_refreshes(self):
        """Accessing an entry moves it to most-recently-used."""
        c = Cache(maxsize=3, name="lru_refresh")
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        # Access "a" so it becomes most recent
        c.get("a")
        c.set("d", 4)  # should evict "b" (least recently used)
        assert c.get("a") == 1
        assert c.get("b") is None

    # -- TTL -----------------------------------------------------------------

    def test_ttl_expiration(self, ttl_cache):
        """Entries expire after TTL."""
        ttl_cache.set("k", "v")
        assert ttl_cache.get("k") == "v"
        time.sleep(0.3)
        assert ttl_cache.get("k") is None

    def test_per_entry_ttl_override(self, cache):
        """Per-entry TTL overrides the cache default."""
        cache.set("short", "v", ttl=0.1)
        cache.set("long", "v", ttl=60)
        time.sleep(0.2)
        assert cache.get("short") is None
        assert cache.get("long") == "v"

    def test_purge_expired(self):
        """purge_expired removes only expired entries."""
        c = Cache(maxsize=10, ttl=0, name="purge_test")
        c.set("expire", "v", ttl=0.1)
        c.set("keep", "v", ttl=60)
        time.sleep(0.2)
        purged = c.purge_expired()
        assert purged == 1
        assert c.get("keep") == "v"
        assert c.size == 1

    def test_has_removes_expired(self):
        """has() cleans up expired entries."""
        c = Cache(maxsize=10, ttl=0.1, name="has_test")
        c.set("k", "v")
        time.sleep(0.2)
        assert not c.has("k")

    # -- Stats ---------------------------------------------------------------

    def test_stats_tracking(self, cache):
        """Stats are properly tracked."""
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # hit
        cache.get("c")  # miss
        cache.get("a")  # hit

        stats = cache.get_stats()
        assert stats.sets == 2
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.current_size == 2
        assert stats.hit_rate == pytest.approx(66.67, abs=0.01)

    def test_stats_eviction_tracking(self):
        """Evictions are counted in stats."""
        c = Cache(maxsize=2, name="evict_stats")
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)  # evicts "a"
        stats = c.get_stats()
        assert stats.evictions == 1

    def test_reset_stats(self, cache):
        """reset_stats clears counters but preserves entries."""
        cache.set("a", 1)
        cache.get("a")
        cache.reset_stats()
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.current_size == 1  # entries preserved

    # -- Thread safety -------------------------------------------------------

    def test_concurrent_access(self):
        """Cache handles concurrent reads and writes safely."""
        c = Cache(maxsize=100, name="thread_test")
        errors = []

        def writer(offset):
            try:
                for i in range(100):
                    c.set(f"key_{offset}_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader(offset):
            try:
                for i in range(100):
                    c.get(f"key_{offset}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for t_id in range(4):
            threads.append(threading.Thread(target=writer, args=(t_id,)))
            threads.append(threading.Thread(target=reader, args=(t_id,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    # -- Cache None values ---------------------------------------------------

    def test_cache_none_value(self, cache):
        """None can be stored and retrieved (distinguished from miss)."""
        cache.set("null_key", None)
        # Using a sentinel-based get to verify:
        assert cache.has("null_key") is True
        # get returns None (the stored value), not the default
        result = cache.get("null_key", "not_found")
        # Since stored value is None, it's returned as None
        assert result is None

    # -- Unlimited cache (maxsize=0) -----------------------------------------

    def test_unlimited_size(self):
        """Cache with maxsize=0 has no evictions."""
        c = Cache(maxsize=0, name="unlimited")
        for i in range(1000):
            c.set(f"k{i}", i)
        assert c.size == 1000
        stats = c.get_stats()
        assert stats.evictions == 0


# ---------------------------------------------------------------------------
# FileCache tests
# ---------------------------------------------------------------------------


class TestFileCache:
    """Tests for the file-based persistent cache."""

    def test_set_and_get(self, file_cache):
        """Basic set/get round-trip."""
        file_cache.set("key1", "value1")
        assert file_cache.get("key1") == "value1"

    def test_get_missing(self, file_cache):
        """Get returns default for missing keys."""
        assert file_cache.get("missing") is None
        assert file_cache.get("missing", "fallback") == "fallback"

    def test_set_complex_value(self, file_cache):
        """Complex JSON-serializable values are stored correctly."""
        data = {"list": [1, 2, 3], "nested": {"key": "value"}, "count": 42}
        file_cache.set("complex", data)
        assert file_cache.get("complex") == data

    def test_overwrite(self, file_cache):
        """Set overwrites existing entries."""
        file_cache.set("k", 1)
        file_cache.set("k", 2)
        assert file_cache.get("k") == 2

    def test_delete(self, file_cache):
        """Delete removes an entry."""
        file_cache.set("k", "v")
        assert file_cache.delete("k") is True
        assert file_cache.get("k") is None

    def test_delete_missing(self, file_cache):
        """Delete returns False for missing keys."""
        assert file_cache.delete("missing") is False

    def test_has(self, file_cache):
        """has() returns True for existing keys."""
        file_cache.set("k", "v")
        assert file_cache.has("k") is True
        assert file_cache.has("missing") is False

    def test_clear(self, file_cache):
        """clear() removes all entries."""
        file_cache.set("a", 1)
        file_cache.set("b", 2)
        count = file_cache.clear()
        assert count == 2
        assert file_cache.get("a") is None

    def test_ttl_expiration(self, tmp_path):
        """Entries with TTL expire."""
        fc = FileCache(directory=tmp_path / "ttl_cache", ttl=0.1, name="ttl_file")
        fc.set("k", "v")
        assert fc.get("k") == "v"
        time.sleep(0.2)
        assert fc.get("k") is None

    def test_per_entry_ttl(self, file_cache):
        """Per-entry TTL overrides the default."""
        file_cache.set("short", "v", ttl=0.1)
        file_cache.set("long", "v", ttl=60)
        time.sleep(0.2)
        assert file_cache.get("short") is None
        assert file_cache.get("long") == "v"

    def test_purge_expired(self, tmp_path):
        """purge_expired removes only expired entries."""
        fc = FileCache(directory=tmp_path / "purge", ttl=0, name="purge_file")
        fc.set("expire", "v", ttl=0.1)
        fc.set("keep", "v", ttl=60)
        time.sleep(0.2)
        purged = fc.purge_expired()
        assert purged == 1
        assert fc.get("keep") == "v"

    def test_stats(self, file_cache):
        """Stats are tracked correctly."""
        file_cache.set("a", 1)
        file_cache.get("a")  # hit
        file_cache.get("b")  # miss

        stats = file_cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.sets == 1
        assert stats.current_size == 1

    def test_repr(self, file_cache):
        """repr is informative."""
        r = repr(file_cache)
        assert "file_test" in r

    def test_creates_directory(self, tmp_path):
        """Cache directory is created automatically."""
        path = tmp_path / "nested" / "deep" / "cache"
        fc = FileCache(directory=path, name="auto_dir")
        fc.set("k", "v")
        assert path.exists()
        assert fc.get("k") == "v"


# ---------------------------------------------------------------------------
# @cached decorator tests
# ---------------------------------------------------------------------------


class TestCachedDecorator:
    """Tests for the @cached decorator."""

    def test_basic_caching(self):
        """Decorated function caches its result."""
        call_count = 0

        @cached(ttl=60)
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive(5) == 10
        assert expensive(5) == 10
        assert call_count == 1  # only called once

    def test_different_args_different_keys(self):
        """Different arguments produce different cache keys."""
        call_count = 0

        @cached(ttl=60)
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x + 1

        assert compute(1) == 2
        assert compute(2) == 3
        assert call_count == 2

    def test_ttl_expiration(self):
        """Cached results expire after TTL."""
        call_count = 0

        @cached(ttl=0.1)
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x

        compute(1)
        assert call_count == 1
        time.sleep(0.2)
        compute(1)
        assert call_count == 2

    def test_cache_clear(self):
        """cache_clear() removes all cached results."""
        call_count = 0

        @cached(ttl=60)
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x

        compute(1)
        assert call_count == 1
        compute.cache_clear()
        compute(1)
        assert call_count == 2

    def test_cache_stats(self):
        """cache_stats() returns statistics."""

        @cached(ttl=60)
        def compute(x):
            return x

        compute(1)
        compute(1)
        compute(2)

        stats = compute.cache_stats()
        assert stats.hits == 1
        assert stats.sets == 2

    def test_kwargs_in_key(self):
        """Keyword arguments are included in cache key."""
        call_count = 0

        @cached(ttl=60)
        def compute(x, multiplier=1):
            nonlocal call_count
            call_count += 1
            return x * multiplier

        assert compute(5, multiplier=2) == 10
        assert compute(5, multiplier=3) == 15
        assert call_count == 2

    def test_custom_cache_instance(self):
        """Decorator can use a provided Cache instance."""
        shared_cache = Cache(maxsize=50, ttl=60, name="shared")

        @cached(cache=shared_cache)
        def compute(x):
            return x * 2

        compute(3)
        assert shared_cache.get_stats().sets == 1

    def test_preserves_function_name(self):
        """Decorator preserves function name and docstring."""

        @cached(ttl=60)
        def my_function():
            """My docstring."""
            return 42

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


# ---------------------------------------------------------------------------
# Global registry tests
# ---------------------------------------------------------------------------


class TestCacheRegistry:
    """Tests for the global cache registry."""

    def test_get_cache_creates_new(self):
        """get_cache creates a new cache when none exists."""
        c = get_cache("test_new", maxsize=100, ttl=60)
        assert c.name == "test_new"
        assert c.maxsize == 100

    def test_get_cache_returns_existing(self):
        """get_cache returns an existing cache by name."""
        c1 = get_cache("test_existing", maxsize=100)
        c2 = get_cache("test_existing", maxsize=999)  # params ignored
        assert c1 is c2
        assert c1.maxsize == 100

    def test_list_caches(self):
        """list_caches returns all registered caches."""
        get_cache("alpha")
        get_cache("beta")
        caches = list_caches()
        assert "alpha" in caches
        assert "beta" in caches

    def test_clear_all_caches(self):
        """clear_all_caches clears every registered cache."""
        c1 = get_cache("one")
        c2 = get_cache("two")
        c1.set("a", 1)
        c2.set("b", 2)

        results = clear_all_caches()
        assert results["one"] == 1
        assert results["two"] == 1
        assert c1.size == 0
        assert c2.size == 0

    def test_remove_cache(self):
        """remove_cache removes and clears a cache from the registry."""
        get_cache("removable")
        assert remove_cache("removable") is True
        assert "removable" not in list_caches()
        assert remove_cache("removable") is False

    def test_get_default_cache(self):
        """get_default_cache returns the default gateway cache."""
        c = get_default_cache()
        assert c.name == "gateway_default"
        assert c.maxsize == 512

    def test_get_config_cache(self):
        """get_config_cache returns a cache tuned for config loading."""
        c = get_config_cache()
        assert c.name == "config"
        assert c.maxsize == 32

    def test_get_provider_cache(self):
        """get_provider_cache returns a cache tuned for providers."""
        c = get_provider_cache()
        assert c.name == "providers"
        assert c.maxsize == 64

    def test_reset_registry(self):
        """reset_registry clears all caches and the registry itself."""
        get_cache("temp")
        reset_registry()
        assert list_caches() == {}


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------


class TestCacheCLI:
    """Tests for cache management CLI commands."""

    def test_cache_group_help(self, runner):
        """cache --help displays help."""
        result = runner.invoke(main, ["cache", "--help"])
        assert result.exit_code == 0
        assert "Manage the application cache" in result.output

    def test_cache_stats_no_caches(self, runner):
        """cache stats with no active caches."""
        result = runner.invoke(main, ["cache", "stats"])
        assert result.exit_code == 0
        assert "No active caches" in result.output

    def test_cache_stats_with_data(self, runner):
        """cache stats shows statistics for active caches."""
        c = get_cache("test_cache", maxsize=100)
        c.set("a", 1)
        c.get("a")
        c.get("missing")

        result = runner.invoke(main, ["cache", "stats"])
        assert result.exit_code == 0
        assert "test_cache" in result.output
        assert "Hits:" in result.output
        assert "Misses:" in result.output

    def test_cache_stats_json(self, runner):
        """cache stats --format json outputs JSON."""
        import json

        get_cache("json_test")
        result = runner.invoke(main, ["cache", "stats", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "caches" in data
        assert "json_test" in data["caches"]

    def test_cache_clear_all(self, runner):
        """cache clear --all clears all caches."""
        c = get_cache("clearable")
        c.set("key", "val")

        result = runner.invoke(main, ["cache", "clear", "--all"])
        assert result.exit_code == 0
        assert "Cleared" in result.output
        assert c.size == 0

    def test_cache_clear_named(self, runner):
        """cache clear <name> clears a specific cache."""
        c = get_cache("specific")
        c.set("k", "v")

        result = runner.invoke(main, ["cache", "clear", "specific"])
        assert result.exit_code == 0
        assert "specific" in result.output
        assert c.size == 0

    def test_cache_clear_unknown(self, runner):
        """cache clear <unknown> gives error."""
        result = runner.invoke(main, ["cache", "clear", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_cache_clear_no_args(self, runner):
        """cache clear with no args lists available caches."""
        get_cache("listed")
        result = runner.invoke(main, ["cache", "clear"])
        assert result.exit_code == 0
        assert "listed" in result.output

    def test_cache_purge(self, runner):
        """cache purge removes expired entries."""
        c = get_cache("purgeable", ttl=0.1)
        c.set("old", "val")
        time.sleep(0.2)

        result = runner.invoke(main, ["cache", "purge"])
        assert result.exit_code == 0

    def test_cache_purge_no_expired(self, runner):
        """cache purge when nothing is expired."""
        c = get_cache("fresh", ttl=600)
        c.set("k", "v")

        result = runner.invoke(main, ["cache", "purge"])
        assert result.exit_code == 0
        assert "No expired entries" in result.output


# ---------------------------------------------------------------------------
# Integration: config caching
# ---------------------------------------------------------------------------


class TestConfigCaching:
    """Tests for caching integration in config loading."""

    def test_config_load_uses_cache(self, tmp_path):
        """Loading the same config twice uses the cache."""
        import yaml

        from src.config import load_config

        config_file = tmp_path / "gateway.yaml"
        config_data = {
            "default_provider": "",
            "log_level": "info",
            "timeout": 30,
            "max_retries": 3,
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # First load populates cache
        config1 = load_config(path=config_file, validate=False, use_cache=True)
        # Second load should hit cache
        config2 = load_config(path=config_file, validate=False, use_cache=True)

        assert config1.timeout == config2.timeout
        assert config1.log_level == config2.log_level

        # Verify cache was used
        cache = get_config_cache()
        stats = cache.get_stats()
        assert stats.hits >= 1

    def test_config_load_bypass_cache(self, tmp_path):
        """use_cache=False bypasses the cache."""
        import yaml

        from src.config import load_config

        config_file = tmp_path / "gateway.yaml"
        config_data = {"log_level": "debug", "timeout": 60}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        load_config(path=config_file, validate=False, use_cache=False)
        load_config(path=config_file, validate=False, use_cache=False)

        # Config cache should have no hits (may not even exist)
        caches = list_caches()
        if "config" in caches:
            stats = caches["config"].get_stats()
            assert stats.hits == 0


# ---------------------------------------------------------------------------
# Integration: provider caching
# ---------------------------------------------------------------------------


class TestProviderCaching:
    """Tests for caching integration in provider lookups."""

    def test_get_builtin_providers_caches(self):
        """get_builtin_providers caches its result."""
        from src.providers import get_builtin_providers

        providers1 = get_builtin_providers(use_cache=True)
        providers2 = get_builtin_providers(use_cache=True)

        assert set(providers1.keys()) == set(providers2.keys())

        cache = get_provider_cache()
        stats = cache.get_stats()
        assert stats.hits >= 1

    def test_get_builtin_provider_caches(self):
        """get_builtin_provider caches individual provider lookups."""
        from src.providers import get_builtin_provider

        p1 = get_builtin_provider("openai", use_cache=True)
        p2 = get_builtin_provider("openai", use_cache=True)

        assert p1 is not None
        assert p2 is not None
        assert p1.name == p2.name

        cache = get_provider_cache()
        stats = cache.get_stats()
        assert stats.hits >= 1

    def test_get_builtin_provider_bypass_cache(self):
        """use_cache=False bypasses the provider cache."""
        from src.providers import get_builtin_provider

        get_builtin_provider("anthropic", use_cache=False)
        get_builtin_provider("anthropic", use_cache=False)

        caches = list_caches()
        if "providers" in caches:
            stats = caches["providers"].get_stats()
            # Should have no hits from the bypass calls
            assert stats.hits == 0

    def test_get_builtin_provider_not_found(self):
        """get_builtin_provider returns None for unknown providers."""
        from src.providers import get_builtin_provider

        result = get_builtin_provider("nonexistent", use_cache=True)
        assert result is None
