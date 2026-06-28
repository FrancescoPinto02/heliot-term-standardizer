from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True)
class CacheStats:
    """Basic cache statistics useful for debugging and profiling."""

    hits: int
    misses: int
    size: int
    max_size: int
    enabled: bool


class LruCache(Generic[K, V]):
    """A minimal least-recently-used cache."""

    def __init__(self, max_size: int = 10_000, enabled: bool = True) -> None:
        if max_size < 0:
            raise ValueError("max_size must be >= 0.")

        self.max_size = max_size
        self.enabled = enabled and max_size > 0
        self._items: OrderedDict[K, V] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: K) -> V | None:
        """Return cached value, or None when key is missing."""
        if not self.enabled:
            self._misses += 1
            return None

        if key not in self._items:
            self._misses += 1
            return None

        self._hits += 1
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def set(self, key: K, value: V) -> None:
        """Store value in cache."""
        if not self.enabled:
            return

        if key in self._items:
            self._items.pop(key)

        self._items[key] = value

        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def clear(self) -> None:
        """Clear cached values and reset statistics."""
        self._items.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            size=len(self._items),
            max_size=self.max_size,
            enabled=self.enabled,
        )