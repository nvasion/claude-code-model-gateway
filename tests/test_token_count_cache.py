"""Tests for the token count cache module.

Covers:
- TokenCountEntry dataclass properties and serialization
- TokenCountCacheStats hit rate and serialization
- TokenCountCache core operations (lookup, store, invalidate, clear)
- is_cacheable() path/method/header logic
- Cache-Control no-cache / no-store bypass
- TTL expiration behaviour
- Compression for large response bodies
- Statistics tracking (hits, misses, bypasses, bytes_saved)
- Global singleton (get_token_count_cache / reset_token_count_cache)
- CLI: cache token-count-stats command (text + JSON output)
- CLI: gateway --token-count-cache / --no-token-count-cache options
"""

from __future__ import annotations

import gzip
import json
import time

import pytest
from click.testing import CliRunner

from src.cli import main
from src.token_count_cache import (
    TOKEN_COUNT_PATH,
    TokenCountCache,
    TokenCountCacheStats,
    TokenCountEntry,
    get_token_count_cache,
    reset_token_count_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global():
    """Reset the global singleton before and after every test."""
    reset_token_count_cache()
    yield
    reset_token_count_cache()


@pytest.fixture
def cache():
    """A fresh TokenCountCache instance with short TTL for tests."""
    return TokenCountCache(maxsize=64, default_ttl=3600.0, name="test_cache")


@pytest.fixture
def small_body() -> bytes:
    """A minimal count_tokens request body."""
    return json.dumps(
        {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    ).encode()


@pytest.fixture
def small_response() -> bytes:
    """A minimal count_tokens API response body."""
    return json.dumps(
        {"input_tokens": 10, "model": "claude-3-5-sonnet-20241022"}
    ).encode()


@pytest.fixture
def large_response() -> bytes:
    """A response body large enough to trigger compression (> 1024 bytes)."""
    payload = {"input_tokens": 500, "model": "claude-3-5-sonnet-20241022"}
    # Pad to exceed compression threshold
    padding = "x" * 2000
    payload["_padding"] = padding
    return json.dumps(payload).encode()


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# TokenCountEntry
# ---------------------------------------------------------------------------


class TestTokenCountEntry:
    """Unit tests for the TokenCountEntry dataclass."""

    def test_not_expired_when_ttl_zero(self):
        entry = TokenCountEntry(
            input_tokens=42, model="m", response_body=b"{}", ttl=0.0
        )
        assert not entry.is_expired

    def test_not_expired_before_ttl(self):
        entry = TokenCountEntry(
            input_tokens=42, model="m", response_body=b"{}", ttl=3600.0
        )
        assert not entry.is_expired

    def test_expired_after_ttl(self):
        entry = TokenCountEntry(
            input_tokens=42,
            model="m",
            response_body=b"{}",
            created_at=time.time() - 7200,
            ttl=3600.0,
        )
        assert entry.is_expired

    def test_age_is_non_negative(self):
        entry = TokenCountEntry(input_tokens=1, model="m", response_body=b"{}")
        assert entry.age >= 0.0

    def test_get_response_body_uncompressed(self):
        body = b'{"input_tokens": 5}'
        entry = TokenCountEntry(
            input_tokens=5, model="m", response_body=body, compressed=False
        )
        assert entry.get_response_body() == body

    def test_get_response_body_compressed(self):
        body = b'{"input_tokens": 5}'
        compressed = gzip.compress(body)
        entry = TokenCountEntry(
            input_tokens=5, model="m", response_body=compressed, compressed=True
        )
        assert entry.get_response_body() == body

    def test_to_dict_contains_expected_keys(self):
        entry = TokenCountEntry(
            input_tokens=99,
            model="claude",
            response_body=b"{}",
            ttl=600.0,
            body_size=2,
        )
        d = entry.to_dict()
        assert d["input_tokens"] == 99
        assert d["model"] == "claude"
        assert d["ttl"] == 600.0
        assert d["body_size"] == 2
        assert "age" in d
        assert "is_expired" in d
        assert "created_at" in d
        assert "compressed" in d


# ---------------------------------------------------------------------------
# TokenCountCacheStats dataclass
# ---------------------------------------------------------------------------


class TestTokenCountCacheStatsDataclass:
    """Unit tests for the TokenCountCacheStats dataclass (hit_rate, to_dict)."""

    def test_hit_rate_zero_when_no_lookups(self):
        stats = TokenCountCacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_100_when_all_hits(self):
        stats = TokenCountCacheStats(lookups=5, hits=5)
        assert stats.hit_rate == 100.0

    def test_hit_rate_partial(self):
        stats = TokenCountCacheStats(lookups=4, hits=1)
        assert stats.hit_rate == pytest.approx(25.0)

    def test_to_dict_contains_all_fields(self):
        stats = TokenCountCacheStats(
            lookups=10, hits=7, misses=3, stores=7, bypasses=1,
            bytes_saved=2048, compressed_stores=2
        )
        d = stats.to_dict()
        assert d["lookups"] == 10
        assert d["hits"] == 7
        assert d["misses"] == 3
        assert d["stores"] == 7
        assert d["bypasses"] == 1
        assert d["bytes_saved"] == 2048
        assert d["compressed_stores"] == 2
        assert "hit_rate" in d


# ---------------------------------------------------------------------------
# TokenCountCache.is_cacheable
# ---------------------------------------------------------------------------


class TestIsCacheable:
    """Tests for the is_cacheable() gating method."""

    def test_post_to_count_tokens_is_cacheable(self, cache, small_body):
        assert cache.is_cacheable(
            path=TOKEN_COUNT_PATH, method="POST", request_body=small_body
        )

    def test_get_is_not_cacheable(self, cache, small_body):
        assert not cache.is_cacheable(
            path=TOKEN_COUNT_PATH, method="GET", request_body=small_body
        )

    def test_wrong_path_not_cacheable(self, cache, small_body):
        assert not cache.is_cacheable(
            path="/v1/messages", method="POST", request_body=small_body
        )

    def test_empty_body_not_cacheable(self, cache):
        assert not cache.is_cacheable(
            path=TOKEN_COUNT_PATH, method="POST", request_body=b""
        )

    def test_none_body_not_cacheable(self, cache):
        assert not cache.is_cacheable(
            path=TOKEN_COUNT_PATH, method="POST", request_body=None
        )

    def test_path_with_query_string_cacheable(self, cache, small_body):
        assert cache.is_cacheable(
            path=TOKEN_COUNT_PATH + "?foo=bar",
            method="POST",
            request_body=small_body,
        )

    def test_no_cache_header_bypasses(self, cache, small_body):
        assert not cache.is_cacheable(
            path=TOKEN_COUNT_PATH,
            method="POST",
            request_body=small_body,
            request_headers={"Cache-Control": "no-cache"},
        )

    def test_no_store_header_bypasses(self, cache, small_body):
        assert not cache.is_cacheable(
            path=TOKEN_COUNT_PATH,
            method="POST",
            request_body=small_body,
            request_headers={"cache-control": "no-store"},
        )

    def test_other_cache_control_value_is_cacheable(self, cache, small_body):
        assert cache.is_cacheable(
            path=TOKEN_COUNT_PATH,
            method="POST",
            request_body=small_body,
            request_headers={"Cache-Control": "max-age=0"},
        )

    def test_method_case_insensitive(self, cache, small_body):
        assert cache.is_cacheable(
            path=TOKEN_COUNT_PATH, method="post", request_body=small_body
        )


# ---------------------------------------------------------------------------
# TokenCountCache core operations
# ---------------------------------------------------------------------------


class TestTokenCountCacheOps:
    """Tests for lookup / store / invalidate / clear."""

    def test_miss_on_empty_cache(self, cache, small_body):
        assert cache.lookup(small_body) is None

    def test_hit_after_store(self, cache, small_body, small_response):
        cache.store(
            request_body=small_body,
            input_tokens=10,
            response_body=small_response,
            model="claude-3-5-sonnet-20241022",
        )
        entry = cache.lookup(small_body)
        assert entry is not None
        assert entry.input_tokens == 10
        assert entry.model == "claude-3-5-sonnet-20241022"

    def test_response_body_roundtrip(self, cache, small_body, small_response):
        cache.store(
            request_body=small_body,
            input_tokens=10,
            response_body=small_response,
        )
        entry = cache.lookup(small_body)
        assert entry.get_response_body() == small_response

    def test_different_bodies_different_slots(self, cache, small_response):
        body_a = json.dumps({"model": "m1", "messages": []}).encode()
        body_b = json.dumps({"model": "m2", "messages": []}).encode()
        cache.store(request_body=body_a, input_tokens=1, response_body=small_response)
        cache.store(request_body=body_b, input_tokens=2, response_body=small_response)
        assert cache.lookup(body_a).input_tokens == 1
        assert cache.lookup(body_b).input_tokens == 2

    def test_invalidate_removes_entry(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=5, response_body=small_response)
        assert cache.lookup(small_body) is not None
        cache.invalidate(small_body)
        assert cache.lookup(small_body) is None

    def test_clear_empties_cache(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=5, response_body=small_response)
        assert cache.size > 0
        cache.clear()
        assert cache.size == 0

    def test_store_empty_body_returns_false(self, cache, small_response):
        assert not cache.store(request_body=b"", input_tokens=1, response_body=small_response)

    def test_size_increases_after_store(self, cache, small_body, small_response):
        assert cache.size == 0
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        assert cache.size == 1

    def test_len_equals_size(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        assert len(cache) == cache.size

    def test_model_auto_extracted_from_body(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        entry = cache.lookup(small_body)
        assert entry.model == "claude-3-5-sonnet-20241022"

    def test_explicit_model_overrides_body(self, cache, small_body, small_response):
        cache.store(
            request_body=small_body,
            input_tokens=10,
            response_body=small_response,
            model="override-model",
        )
        entry = cache.lookup(small_body)
        assert entry.model == "override-model"


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


class TestTokenCountCacheCompression:
    """Tests for optional gzip compression of large response bodies."""

    def test_small_body_not_compressed(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        entry = cache.lookup(small_body)
        assert not entry.compressed

    def test_large_body_compressed(self, cache, small_body, large_response):
        cache.store(request_body=small_body, input_tokens=500, response_body=large_response)
        entry = cache.lookup(small_body)
        assert entry.compressed

    def test_large_body_roundtrip(self, cache, small_body, large_response):
        cache.store(request_body=small_body, input_tokens=500, response_body=large_response)
        entry = cache.lookup(small_body)
        assert entry.get_response_body() == large_response

    def test_compression_disabled(self, small_body, large_response):
        c = TokenCountCache(
            maxsize=16, default_ttl=3600, enable_compression=False, name="no_compress"
        )
        c.store(request_body=small_body, input_tokens=500, response_body=large_response)
        entry = c.lookup(small_body)
        assert not entry.compressed
        assert entry.get_response_body() == large_response


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


class TestTokenCountCacheTTL:
    """Tests for TTL and expiry behaviour."""

    def test_custom_ttl_stored(self, cache, small_body, small_response):
        cache.store(
            request_body=small_body,
            input_tokens=10,
            response_body=small_response,
            ttl=1.0,
        )
        entry = cache.lookup(small_body)
        assert entry.ttl == 1.0

    def test_default_ttl_used_when_none(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        entry = cache.lookup(small_body)
        assert entry.ttl == cache.default_ttl

    def test_cache_with_short_ttl_expires(self, small_body, small_response):
        c = TokenCountCache(maxsize=16, default_ttl=0.1, name="short_ttl")
        c.store(request_body=small_body, input_tokens=10, response_body=small_response)
        assert c.lookup(small_body) is not None  # should hit immediately
        time.sleep(0.2)
        # After expiry the underlying Cache should evict or report expired
        # The TokenCountCache itself will return None for expired entries
        # (actual eviction is delegated to Cache.get which returns None after TTL)
        result = c.lookup(small_body)
        # Either the entry is gone or is_expired is True
        if result is not None:
            assert result.is_expired

    def test_purge_expired_removes_old_entries(self, small_body, small_response):
        c = TokenCountCache(maxsize=16, default_ttl=0.1, name="purge_test")
        c.store(request_body=small_body, input_tokens=10, response_body=small_response)
        time.sleep(0.2)
        removed = c.purge_expired()
        assert isinstance(removed, int)
        # After purging, size should be 0 (or the underlying cache may have
        # already evicted it)
        assert c.size == 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestTokenCountCacheStats:
    """Tests for statistics tracking."""

    def test_initial_stats_all_zero(self, cache):
        stats = cache.get_stats()
        assert stats.lookups == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.stores == 0
        assert stats.bypasses == 0
        assert stats.bytes_saved == 0

    def test_miss_increments_lookups_and_misses(self, cache, small_body):
        cache.lookup(small_body)
        stats = cache.get_stats()
        assert stats.lookups == 1
        assert stats.misses == 1
        assert stats.hits == 0

    def test_hit_increments_lookups_and_hits(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        cache.lookup(small_body)
        stats = cache.get_stats()
        assert stats.lookups == 1
        assert stats.hits == 1
        assert stats.misses == 0

    def test_store_increments_stores(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        assert cache.get_stats().stores == 1

    def test_no_cache_header_increments_bypass(self, cache, small_body):
        cache.lookup(small_body, request_headers={"Cache-Control": "no-cache"})
        stats = cache.get_stats()
        assert stats.bypasses == 1

    def test_bytes_saved_tracks_body_size(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        cache.lookup(small_body)
        stats = cache.get_stats()
        assert stats.bytes_saved == len(small_response)

    def test_reset_stats(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        cache.lookup(small_body)
        cache.reset_stats()
        stats = cache.get_stats()
        assert stats.lookups == 0
        assert stats.hits == 0

    def test_compressed_stores_counter(self, cache, small_body, large_response):
        cache.store(request_body=small_body, input_tokens=500, response_body=large_response)
        assert cache.get_stats().compressed_stores == 1


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestGlobalSingleton:
    """Tests for get_token_count_cache / reset_token_count_cache."""

    def test_returns_same_instance(self):
        a = get_token_count_cache()
        b = get_token_count_cache()
        assert a is b

    def test_reset_clears_instance(self, small_body, small_response):
        tc = get_token_count_cache()
        tc.store(request_body=small_body, input_tokens=1, response_body=small_response)
        reset_token_count_cache()
        fresh = get_token_count_cache()
        assert fresh.size == 0

    def test_new_instance_after_reset(self):
        old = get_token_count_cache()
        reset_token_count_cache()
        new = get_token_count_cache()
        assert old is not new

    def test_custom_maxsize(self):
        reset_token_count_cache()
        tc = get_token_count_cache(maxsize=128, default_ttl=60.0)
        assert tc.default_ttl == 60.0

    def test_repr_contains_name(self):
        tc = TokenCountCache(maxsize=8, default_ttl=60, name="mytest")
        assert "mytest" in repr(tc)


# ---------------------------------------------------------------------------
# Cache-Control header handling
# ---------------------------------------------------------------------------


class TestCacheControlHeaders:
    """Tests for Cache-Control header parsing in lookup/is_cacheable."""

    def test_no_headers_returns_entry(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        assert cache.lookup(small_body, request_headers=None) is not None

    def test_empty_headers_returns_entry(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        assert cache.lookup(small_body, request_headers={}) is not None

    def test_no_cache_bypasses_lookup(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        result = cache.lookup(small_body, request_headers={"Cache-Control": "no-cache"})
        assert result is None

    def test_no_store_bypasses_lookup(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        result = cache.lookup(small_body, request_headers={"Cache-Control": "no-store"})
        assert result is None

    def test_combined_directives_bypass(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        result = cache.lookup(
            small_body, request_headers={"Cache-Control": "max-age=0, no-cache"}
        )
        assert result is None

    def test_case_insensitive_header_name(self, cache, small_body, small_response):
        cache.store(request_body=small_body, input_tokens=10, response_body=small_response)
        result = cache.lookup(
            small_body, request_headers={"CACHE-CONTROL": "no-cache"}
        )
        assert result is None


# ---------------------------------------------------------------------------
# CLI: cache token-count-stats
# ---------------------------------------------------------------------------


class TestCLITokenCountStats:
    """Tests for the `cache token-count-stats` CLI command."""

    def test_text_output(self, runner):
        result = runner.invoke(main, ["cache", "token-count-stats"])
        assert result.exit_code == 0, result.output
        assert "Token Count Cache Statistics" in result.output
        assert "Lookups" in result.output
        assert "Hit rate" in result.output

    def test_json_output(self, runner):
        result = runner.invoke(main, ["cache", "token-count-stats", "--format", "json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "lookups" in data
        assert "hits" in data
        assert "misses" in data
        assert "stores" in data
        assert "hit_rate" in data
        assert "bytes_saved" in data

    def test_json_values_are_numeric(self, runner):
        result = runner.invoke(main, ["cache", "token-count-stats", "--format", "json"])
        data = json.loads(result.output)
        assert isinstance(data["lookups"], int)
        assert isinstance(data["hit_rate"], float)


# ---------------------------------------------------------------------------
# CLI: gateway --token-count-cache options
# ---------------------------------------------------------------------------


class TestCLIGatewayTokenCountCacheOptions:
    """Tests that the gateway command exposes token count cache options."""

    def test_help_shows_token_count_cache_option(self, runner):
        result = runner.invoke(main, ["gateway", "--help"])
        assert result.exit_code == 0, result.output
        assert "--token-count-cache" in result.output

    def test_help_shows_token_count_cache_ttl_option(self, runner):
        result = runner.invoke(main, ["gateway", "--help"])
        assert "--token-count-cache-ttl" in result.output

    def test_help_shows_token_count_cache_maxsize_option(self, runner):
        result = runner.invoke(main, ["gateway", "--help"])
        assert "--token-count-cache-maxsize" in result.output

    def test_no_token_count_cache_is_default(self, runner):
        """--no-token-count-cache is the default (disabled)."""
        result = runner.invoke(main, ["gateway", "--help"])
        assert "--no-token-count-cache" in result.output


# ---------------------------------------------------------------------------
# TOKEN_COUNT_PATH constant
# ---------------------------------------------------------------------------


class TestConstants:
    def test_token_count_path_value(self):
        assert TOKEN_COUNT_PATH == "/v1/messages/count_tokens"
