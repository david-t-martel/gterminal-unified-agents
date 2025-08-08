#!/usr/bin/env python3
"""Cache Management for Performance Optimization.

This module provides caching functionality merged from GAPP including:
- In-memory caching with TTL support
- Persistent caching to disk
- Cache invalidation strategies
- Performance metrics tracking
"""

import json
import logging
from pathlib import Path
import time
from typing import Any

logger = logging.getLogger(__name__)


class CacheManager:
    """High-performance cache manager with multiple backend support."""

    def __init__(
        self, max_size: int = 1000, default_ttl: int = 3600, cache_dir: Path | None = None
    ):
        """Initialize cache manager.

        Args:
            max_size: Maximum number of items in memory cache
            default_ttl: Default time-to-live in seconds
            cache_dir: Directory for persistent cache storage
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache_dir = cache_dir or Path.cwd() / ".cache"
        self.cache_dir.mkdir(exist_ok=True)

        # In-memory cache storage
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._access_times: dict[str, float] = {}

        # Performance metrics
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "writes": 0}

        logger.info(f"✅ CacheManager initialized with max_size={max_size}, ttl={default_ttl}s")

    async def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]

            # Check if expired
            if time.time() > entry["expires_at"]:
                await self._remove_key(key)
                self.stats["misses"] += 1
                return None

            # Update access time and return value
            self._access_times[key] = time.time()
            self.stats["hits"] += 1
            return entry["value"]

        # Check persistent cache
        persistent_value = await self._get_persistent(key)
        if persistent_value is not None:
            # Load into memory cache
            await self.set(key, persistent_value, ttl=self.default_ttl)
            self.stats["hits"] += 1
            return persistent_value

        self.stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl

        # Ensure we don't exceed max size
        await self._ensure_capacity()

        # Store in memory cache
        self._memory_cache[key] = {
            "value": value,
            "created_at": time.time(),
            "expires_at": expires_at,
            "ttl": ttl,
        }
        self._access_times[key] = time.time()
        self.stats["writes"] += 1

        # Also store persistently for valuable data
        await self._set_persistent(key, value, expires_at)

        logger.debug(f"📦 Cached key '{key}' with TTL {ttl}s")

    async def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key existed and was deleted
        """
        existed = key in self._memory_cache
        await self._remove_key(key)
        await self._delete_persistent(key)
        return existed

    async def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        self._access_times.clear()

        # Clear persistent cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info("🧹 Cache cleared")

    async def _ensure_capacity(self) -> None:
        """Ensure cache doesn't exceed maximum size."""
        while len(self._memory_cache) >= self.max_size:
            # Find least recently accessed key
            lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            await self._remove_key(lru_key)
            self.stats["evictions"] += 1
            logger.debug(f"🗑️ Evicted LRU key: {lru_key}")

    async def _remove_key(self, key: str) -> None:
        """Remove key from memory cache."""
        self._memory_cache.pop(key, None)
        self._access_times.pop(key, None)

    async def _get_persistent(self, key: str) -> Any | None:
        """Get value from persistent cache."""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            if not cache_file.exists():
                return None

            with open(cache_file) as f:
                data = json.load(f)

            # Check expiration
            if time.time() > data["expires_at"]:
                cache_file.unlink()
                return None

            return data["value"]
        except Exception as e:
            logger.warning(f"Failed to read persistent cache for '{key}': {e}")
            return None

    async def _set_persistent(self, key: str, value: Any, expires_at: float) -> None:
        """Store value in persistent cache."""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            data = {"value": value, "expires_at": expires_at, "created_at": time.time()}

            with open(cache_file, "w") as f:
                json.dump(data, f, default=str)
        except Exception as e:
            logger.warning(f"Failed to write persistent cache for '{key}': {e}")

    async def _delete_persistent(self, key: str) -> None:
        """Delete key from persistent cache."""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete persistent cache for '{key}': {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            "memory_cache_size": len(self._memory_cache),
            "max_size": self.max_size,
            "hit_rate_percent": round(hit_rate, 2),
            "total_hits": self.stats["hits"],
            "total_misses": self.stats["misses"],
            "total_writes": self.stats["writes"],
            "total_evictions": self.stats["evictions"],
            "cache_directory": str(self.cache_dir),
        }

    async def cleanup_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of expired entries removed
        """
        current_time = time.time()
        expired_keys = []

        # Check memory cache
        for key, entry in self._memory_cache.items():
            if current_time > entry["expires_at"]:
                expired_keys.append(key)

        # Remove expired keys
        for key in expired_keys:
            await self._remove_key(key)

        # Cleanup persistent cache
        persistent_expired = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    data = json.load(f)

                if current_time > data["expires_at"]:
                    cache_file.unlink()
                    persistent_expired += 1
            except Exception as e:
                logger.warning(f"Failed to check expiration for {cache_file}: {e}")

        total_expired = len(expired_keys) + persistent_expired
        if total_expired > 0:
            logger.info(f"🧹 Cleaned up {total_expired} expired cache entries")

        return total_expired

    def __repr__(self) -> str:
        """String representation of cache manager."""
        return (
            f"CacheManager(size={len(self._memory_cache)}/{self.max_size}, "
            f"hit_rate={self.get_stats()['hit_rate_percent']:.1f}%)"
        )
