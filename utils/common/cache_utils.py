"""Shared caching utilities using Rust bindings for performance.

Provides consistent caching across all agents with high-performance
Rust backend and intelligent cache management.
"""

import hashlib
import json
import logging
from typing import Any

from gterminal.utils.rust_extensions.rust_bindings import rust_utils

logger = logging.getLogger(__name__)


class CacheUtils:
    """High-performance caching utilities with Rust backend."""

    def __init__(self) -> None:
        """Initialize cache utilities."""
        self.rust = rust_utils
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}

        logger.info("Initialized CacheUtils with Rust bindings")

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a consistent cache key from arguments."""
        key_data = {"args": args, "kwargs": sorted(kwargs.items())}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        try:
            value = self.rust.cache_get(key)
            if value is not None:
                self.cache_stats["hits"] += 1
                logger.debug(f"Cache hit for key: {key[:16]}...")
            else:
                self.cache_stats["misses"] += 1
                logger.debug(f"Cache miss for key: {key[:16]}...")
            return value
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            self.cache_stats["misses"] += 1
            return None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Set value in cache."""
        try:
            success = self.rust.cache_set(key, value, ttl_seconds)
            if success:
                self.cache_stats["sets"] += 1
                logger.debug(f"Cache set for key: {key[:16]}...")
            return success
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            success = self.rust.cache_delete(key)
            if success:
                self.cache_stats["deletes"] += 1
                logger.debug(f"Cache delete for key: {key[:16]}...")
            return success
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear entire cache."""
        try:
            success = self.rust.cache_clear()
            if success:
                logger.info("Cache cleared successfully")
                # Reset stats
                self.cache_stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}
            return success
        except Exception as e:
            logger.exception(f"Cache clear failed: {e}")
            return False

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0.0

        rust_stats = self.rust.cache_stats()

        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            **rust_stats,
        }

    # Decorator for caching function results

    def cached(
        self, key_prefix: str = "", ttl_seconds: int | None = None, use_args: bool = True
    ) -> None:
        """Decorator to cache function results."""

        def decorator(func) -> None:
            def wrapper(*args, **kwargs) -> None:
                # Generate cache key
                if use_args:
                    cache_key = f"{key_prefix}{func.__name__}:{self._generate_key(*args, **kwargs)}"
                else:
                    cache_key = f"{key_prefix}{func.__name__}"

                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl_seconds)

                return result

            return wrapper

        return decorator

    def cached_async(
        self, key_prefix: str = "", ttl_seconds: int | None = None, use_args: bool = True
    ):
        """Decorator to cache async function results."""

        def decorator(func) -> None:
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if use_args:
                    cache_key = f"{key_prefix}{func.__name__}:{self._generate_key(*args, **kwargs)}"
                else:
                    cache_key = f"{key_prefix}{func.__name__}"

                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = await func(*args, **kwargs)
                self.set(cache_key, result, ttl_seconds)

                return result

            return wrapper

        return decorator

    # Utility methods for common caching patterns

    def cache_file_content(self, file_path: str, content: str) -> bool:
        """Cache file content with path-based key."""
        key = f"file_content:{file_path}"
        return self.set(key, content, ttl_seconds=1800)  # 30 minutes

    def get_cached_file_content(self, file_path: str) -> str | None:
        """Get cached file content."""
        key = f"file_content:{file_path}"
        return self.get(key)

    def cache_file_hash(self, file_path: str, hash_value: str) -> bool:
        """Cache file hash with path-based key."""
        key = f"file_hash:{file_path}"
        return self.set(key, hash_value, ttl_seconds=3600)  # 1 hour

    def get_cached_file_hash(self, file_path: str) -> str | None:
        """Get cached file hash."""
        key = f"file_hash:{file_path}"
        return self.get(key)

    def cache_search_results(self, query: str, results: list) -> bool:
        """Cache search results."""
        key = f"search:{self._generate_key(query)}"
        return self.set(key, results, ttl_seconds=600)  # 10 minutes

    def get_cached_search_results(self, query: str) -> list | None:
        """Get cached search results."""
        key = f"search:{self._generate_key(query)}"
        return self.get(key)

    def invalidate_file_cache(self, file_path: str) -> None:
        """Invalidate all cache entries for a file."""
        content_key = f"file_content:{file_path}"
        hash_key = f"file_hash:{file_path}"

        self.delete(content_key)
        self.delete(hash_key)

        logger.debug(f"Invalidated cache for file: {file_path}")

    def warm_cache(self, data_dict: dict[str, Any]) -> int:
        """Warm the cache with a dictionary of key-value pairs."""
        success_count = 0

        for key, value in data_dict.items():
            if self.set(key, value):
                success_count += 1

        logger.info(f"Warmed cache with {success_count}/{len(data_dict)} entries")
        return success_count


# Global instance for easy access
cache_utils = CacheUtils()


# Convenience functions for direct access
def get_cached(key: str) -> Any | None:
    """Get value from global cache."""
    return cache_utils.get(key)


def set_cached(key: str, value: Any, ttl_seconds: int | None = None) -> bool:
    """Set value in global cache."""
    return cache_utils.set(key, value, ttl_seconds)


def delete_cached(key: str) -> bool:
    """Delete key from global cache."""
    return cache_utils.delete(key)


def cached(key_prefix: str = "", ttl_seconds: int | None = None) -> None:
    """Decorator for caching function results."""
    return cache_utils.cached(key_prefix, ttl_seconds)


def cached_async(key_prefix: str = "", ttl_seconds: int | None = None) -> None:
    """Decorator for caching async function results."""
    return cache_utils.cached_async(key_prefix, ttl_seconds)
