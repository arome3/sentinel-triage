"""Tests for the in-memory content hash cache."""

from app.content_hash_cache import DEFAULT_MAX_SIZE, HashCache


class TestHashCache:
    """Unit tests for HashCache."""

    def test_put_and_get(self):
        """Stored values are retrievable."""
        cache = HashCache()
        cache.put("abc123", {"result": "spam"})
        assert cache.get("abc123") == {"result": "spam"}

    def test_get_missing_returns_none(self):
        """Missing keys return None."""
        cache = HashCache()
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """Oldest entry is evicted when max_size is exceeded."""
        cache = HashCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_access_promotes_entry(self):
        """Accessing an entry prevents its eviction."""
        cache = HashCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # promote "a"
        cache.put("c", 3)  # should evict "b", not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_put_updates_existing(self):
        """Putting an existing key updates the value."""
        cache = HashCache()
        cache.put("key", "old")
        cache.put("key", "new")
        assert cache.get("key") == "new"
        assert cache.size == 1

    def test_invalidate_existing(self):
        """Invalidate removes the entry and returns True."""
        cache = HashCache()
        cache.put("key", "value")
        assert cache.invalidate("key") is True
        assert cache.get("key") is None

    def test_invalidate_missing(self):
        """Invalidate returns False for missing keys."""
        cache = HashCache()
        assert cache.invalidate("nope") is False

    def test_clear(self):
        """Clear removes all entries."""
        cache = HashCache()
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size == 0

    def test_size_property(self):
        """Size reflects current entry count."""
        cache = HashCache()
        assert cache.size == 0
        cache.put("a", 1)
        assert cache.size == 1

    def test_default_max_size(self):
        """Default max_size matches the module constant."""
        cache = HashCache()
        assert cache.max_size == DEFAULT_MAX_SIZE
