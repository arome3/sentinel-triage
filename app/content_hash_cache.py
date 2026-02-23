"""In-memory content hash cache for fast deduplication.

Provides an LRU-bounded cache that maps content fingerprints to their
processing results, avoiding redundant embedding and routing calls for
recently-seen identical content.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

#: Default maximum cache entries.
DEFAULT_MAX_SIZE: int = 1024


@dataclass
class HashCache:
    """LRU cache keyed by content fingerprint.

    Thread-safety note: this cache is designed for single-threaded async
    usage within one event loop.  It is NOT safe for concurrent access
    from multiple threads.
    """

    max_size: int = DEFAULT_MAX_SIZE
    _store: OrderedDict[str, Any] = field(default_factory=OrderedDict, repr=False)

    def get(self, fingerprint: str) -> Any | None:
        """Retrieve a cached value, promoting it to most-recent."""
        if fingerprint in self._store:
            self._store.move_to_end(fingerprint)
            return self._store[fingerprint]
        return None

    def put(self, fingerprint: str, value: Any) -> None:
        """Store a value, evicting the oldest entry if at capacity."""
        if fingerprint in self._store:
            self._store.move_to_end(fingerprint)
        self._store[fingerprint] = value
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def invalidate(self, fingerprint: str) -> bool:
        """Remove a specific entry.  Returns True if it existed."""
        try:
            del self._store[fingerprint]
            return True
        except KeyError:
            return False

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._store)
