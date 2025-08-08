"""
Memory-aware TTL cache implementation with sophisticated eviction strategies.

This module extracts patterns from the my-fullstack-agent Rust cache implementation,
providing high-performance in-memory caching with:
- TTL-based expiration
- LRU eviction
- Memory pressure awareness
- System resource monitoring
- Comprehensive statistics
"""

from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
import hashlib
import threading
import time
from typing import Any

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class CacheItem:
    """Cache item with TTL and access tracking."""

    value: bytes
    created_at: float
    ttl_seconds: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    content_hash: str | None = None
    memory_size: int = 0
    value_type: str = "str"  # Track original type: 'str', 'bytes', 'json'

    def __post_init__(self):
        if self.content_hash is None:
            self.content_hash = hashlib.sha256(self.value).hexdigest()[:16]

        if self.memory_size == 0:
            # Estimate memory usage: value size + object overhead
            self.memory_size = len(self.value) + 200  # Approximate object overhead

    @property
    def is_expired(self) -> bool:
        """Check if item has expired."""
        return time.time() > (self.created_at + self.ttl_seconds)

    @property
    def remaining_ttl(self) -> int:
        """Get remaining TTL in seconds."""
        remaining = (self.created_at + self.ttl_seconds) - time.time()
        return max(0, int(remaining))

    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class MemoryCache:
    """
    High-performance in-memory cache with TTL and LRU eviction.

    Features:
    - TTL-based expiration
    - LRU eviction when at capacity
    - Thread-safe operations
    - Comprehensive statistics
    - Memory usage tracking
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheItem] = {}
        self._access_order = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.current_memory = 0

    def get(self, key: str) -> str | bytes | dict | list | Any | None:
        """Get value from cache."""
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                self.misses += 1
                return None

            if item.is_expired:
                # Remove expired item
                self._remove_item(key)
                self.misses += 1
                return None

            # Update access statistics
            item.update_access()
            self._access_order.move_to_end(key)
            self.hits += 1

            # Restore value based on original type
            if item.value_type == "bytes":
                return item.value
            elif item.value_type == "json":
                try:
                    import json

                    return json.loads(item.value.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return item.value
            else:  # 'str' type
                try:
                    return item.value.decode("utf-8")
                except UnicodeDecodeError:
                    return item.value

    def set(self, key: str, value: str | bytes | dict | list | Any, ttl: int | None = None) -> bool:
        """Set value in cache."""
        # Convert to bytes before creating CacheItem and track original type
        if isinstance(value, str):
            value_bytes = value.encode("utf-8")
            value_type = "str"
        elif isinstance(value, bytes):
            value_bytes = value
            value_type = "bytes"
        else:
            # Serialize complex objects to JSON
            import json

            value_bytes = json.dumps(value, default=str).encode("utf-8")
            value_type = "json"

        ttl = ttl or self.default_ttl
        item = CacheItem(
            value=value_bytes,
            created_at=time.time(),
            ttl_seconds=ttl,
            value_type=value_type,
        )

        with self._lock:
            # Remove existing item if present
            if key in self._cache:
                self._remove_item(key)

            # Check capacity and evict if necessary
            if len(self._cache) >= self.max_size:
                self._evict_expired()
                if len(self._cache) >= self.max_size:
                    self._evict_lru()

            # Add new item
            self._cache[key] = item
            self._access_order[key] = True
            self.current_memory += item.memory_size
            self.sets += 1

        return True

    def delete(self, key: str) -> bool:
        """Remove key from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_item(key)
                return True
            return False

    def clear(self, pattern: str | None = None) -> int:
        """Clear cache entries."""
        with self._lock:
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
                self._access_order.clear()
                self.current_memory = 0
                return count
            else:
                keys_to_remove = [k for k in self._cache if pattern in k]
                count = 0
                for key in keys_to_remove:
                    self._remove_item(key)
                    count += 1
                return count

    def contains(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return False
            if item.is_expired:
                # Schedule for removal
                with self._lock:
                    self._remove_item(key)
                return False
            return True

    def keys(self, pattern: str | None = None, limit: int | None = None) -> list[str]:
        """Get keys matching pattern."""
        with self._lock:
            result = []
            count = 0

            for key, item in self._cache.items():
                if item.is_expired:
                    continue
                if pattern is None or pattern in key:
                    result.append(key)
                    count += 1
                    if limit and count >= limit:
                        break

            return result

    def stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_bytes": self.current_memory,
            "memory_mb": round(self.current_memory / (1024 * 1024), 2),
        }

    def _remove_item(self, key: str):
        """Remove item from cache (internal method)."""
        if key in self._cache:
            item = self._cache.pop(key)
            self._access_order.pop(key, None)
            self.current_memory -= item.memory_size

    def _evict_expired(self):
        """Remove expired entries."""
        expired_keys = [key for key, item in self._cache.items() if item.is_expired]
        for key in expired_keys:
            self._remove_item(key)
            self.evictions += 1

    def _evict_lru(self):
        """Remove least recently used entry."""
        if self._access_order:
            lru_key = next(iter(self._access_order))
            self._remove_item(lru_key)
            self.evictions += 1


class MemoryAwareCache(MemoryCache):
    """
    Memory-aware cache that monitors system resources and evicts based on memory pressure.

    Additional features:
    - System memory monitoring
    - Memory pressure-based eviction
    - Configurable memory thresholds
    - Advanced eviction strategies
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        max_memory_mb: int = 100,
        memory_threshold: float = 0.85,
    ):
        super().__init__(max_size, default_ttl)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.memory_threshold = max(0.1, min(0.95, memory_threshold))
        self.memory_evictions = 0

    def set(self, key: str, value: str | bytes | dict | list | Any, ttl: int | None = None) -> bool:
        """Set value with memory pressure checking."""
        # Calculate estimated size based on original value
        if isinstance(value, str):
            estimated_size = len(value.encode("utf-8")) + 200
        elif isinstance(value, bytes):
            estimated_size = len(value) + 200
        else:
            # For complex objects, estimate JSON size
            import json

            estimated_size = len(json.dumps(value, default=str).encode("utf-8")) + 200

        # Check memory pressure before adding
        if self._should_evict_for_memory(estimated_size):
            self._evict_for_memory_pressure(estimated_size)

        return super().set(key, value, ttl)

    def _should_evict_for_memory(self, new_item_size: int) -> bool:
        """Check if memory eviction is needed."""
        # Check cache memory limit
        if self.current_memory + new_item_size > self.max_memory_bytes:
            return True

        # Check system memory pressure
        if PSUTIL_AVAILABLE:
            try:
                memory_percent = psutil.virtual_memory().percent / 100.0
                return memory_percent > self.memory_threshold
            except:
                return False
        return False

    def _evict_for_memory_pressure(self, required_memory: int):
        """Evict items to free memory."""
        # Calculate memory to free (at least 10% of max or required amount)
        memory_to_free = max(self.max_memory_bytes // 10, required_memory)

        with self._lock:
            # First, remove expired entries
            self._evict_expired()

            # Then remove LRU items until enough memory is freed
            freed = 0
            lru_keys = list(self._access_order.keys())

            for key in lru_keys:
                if freed >= memory_to_free:
                    break

                item = self._cache.get(key)
                if item:
                    freed += item.memory_size
                    self._remove_item(key)
                    self.memory_evictions += 1

    def memory_info(self) -> dict[str, int | float]:
        """Get memory-specific statistics."""
        base_info = {
            "current_bytes": self.current_memory,
            "current_mb": round(self.current_memory / (1024 * 1024), 2),
            "max_bytes": self.max_memory_bytes,
            "max_mb": round(self.max_memory_bytes / (1024 * 1024), 2),
            "memory_threshold_percent": round(self.memory_threshold * 100, 1),
            "memory_evictions": self.memory_evictions,
        }

        if PSUTIL_AVAILABLE:
            try:
                system_memory = psutil.virtual_memory()
                base_info.update(
                    {
                        "system_total_mb": round(system_memory.total / (1024 * 1024), 2),
                        "system_used_mb": round(system_memory.used / (1024 * 1024), 2),
                        "system_percent": system_memory.percent,
                    }
                )
            except:
                pass  # Use base_info only

        return base_info

    def stats(self) -> dict[str, int | float]:
        """Get comprehensive statistics including memory info."""
        base_stats = super().stats()
        base_stats.update(self.memory_info())
        return base_stats
