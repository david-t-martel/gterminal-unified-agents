"""
Multi-layer cache management with sophisticated eviction and monitoring.

This module provides advanced cache management extracted from my-fullstack-agent,
combining multiple cache layers for optimal performance:
- L1: Memory cache (fastest, limited size)
- L2: Redis cache (shared, persistent)
- Statistics and monitoring
- Health checks and failover
"""

from dataclasses import dataclass
import time
from typing import Any

from .memory_cache import MemoryAwareCache
from .redis_cache import REDIS_AVAILABLE
from .redis_cache import RedisCache
from .redis_cache import RedisConfig


@dataclass
class CacheConfig:
    """Comprehensive cache configuration."""

    # Memory cache settings
    memory_max_size: int = 5000
    memory_ttl: int = 1800  # 30 minutes
    memory_max_mb: int = 100
    memory_threshold: float = 0.85

    # Redis cache settings
    redis_enabled: bool = False
    redis_config: RedisConfig | None = None
    redis_ttl: int = 3600  # 1 hour

    # Multi-layer behavior
    write_through: bool = True  # Write to both layers
    read_through: bool = True  # Populate L1 from L2 on miss

    # Health check settings
    health_check_interval: int = 60  # seconds


class CacheManager:
    """
    Multi-layer cache manager with sophisticated eviction and monitoring.

    Features:
    - L1 memory cache for ultra-fast access
    - L2 Redis cache for persistence and sharing
    - Automatic failover and health monitoring
    - Write-through and read-through patterns
    - Comprehensive statistics and metrics
    """

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()

        # Initialize L1 cache (memory)
        self.l1_cache = MemoryAwareCache(
            max_size=self.config.memory_max_size,
            default_ttl=self.config.memory_ttl,
            max_memory_mb=self.config.memory_max_mb,
            memory_threshold=self.config.memory_threshold,
        )

        # Initialize L2 cache (Redis) if enabled
        self.l2_cache: RedisCache | None = None
        if self.config.redis_enabled and REDIS_AVAILABLE and self.config.redis_config:
            try:
                self.l2_cache = RedisCache(
                    config=self.config.redis_config, default_ttl=self.config.redis_ttl
                )
                # Test connection
                if not self.l2_cache.is_connected():
                    print("Warning: Redis connection failed, disabling L2 cache")
                    self.l2_cache = None
            except Exception as e:
                print(f"Warning: Redis initialization failed: {e}")
                self.l2_cache = None

        # Statistics
        self.total_hits = 0
        self.total_misses = 0
        self.l1_hits = 0
        self.l2_hits = 0
        self.l1_misses = 0
        self.l2_misses = 0
        self.errors = 0

        # Health monitoring
        self._last_health_check = 0
        self._redis_healthy = self.l2_cache is not None

    def get(self, key: str) -> str | dict | list | None:
        """Get value from multi-layer cache with read-through."""
        # Try L1 cache first
        try:
            value = self.l1_cache.get(key)
            if value is not None:
                self.l1_hits += 1
                self.total_hits += 1
                return value
            self.l1_misses += 1
        except Exception as e:
            self.errors += 1
            print(f"L1 cache error: {e}")

        # Try L2 cache if available and read-through enabled
        if self.l2_cache and self.config.read_through:
            try:
                value = self.l2_cache.get(key)
                if value is not None:
                    self.l2_hits += 1
                    self.total_hits += 1

                    # Populate L1 cache for future access
                    try:
                        self.l1_cache.set(key, str(value) if not isinstance(value, str) else value)
                    except Exception as e:
                        print(f"L1 backfill error: {e}")

                    return value
                self.l2_misses += 1
            except Exception as e:
                self.errors += 1
                print(f"L2 cache error: {e}")

        # Cache miss
        self.total_misses += 1
        return None

    def set(self, key: str, value: str | dict | list | int | float, ttl: int | None = None) -> bool:
        """Set value in multi-layer cache with write-through."""
        l1_success = False
        l2_success = True  # Assume success if L2 not available

        # Set in L1 cache
        try:
            l1_success = self.l1_cache.set(
                key,
                str(value) if not isinstance(value, str) else value,
                ttl or self.config.memory_ttl,
            )
        except Exception as e:
            self.errors += 1
            print(f"L1 cache set error: {e}")

        # Set in L2 cache if available and write-through enabled
        if self.l2_cache and self.config.write_through:
            try:
                l2_success = self.l2_cache.set(key, value, ttl or self.config.redis_ttl)
            except Exception as e:
                self.errors += 1
                print(f"L2 cache set error: {e}")
                l2_success = False

        return l1_success or l2_success

    def delete(self, key: str) -> bool:
        """Delete key from all cache layers."""
        l1_deleted = False
        l2_deleted = False

        # Delete from L1
        try:
            l1_deleted = self.l1_cache.delete(key)
        except Exception as e:
            self.errors += 1
            print(f"L1 cache delete error: {e}")

        # Delete from L2
        if self.l2_cache:
            try:
                l2_deleted = self.l2_cache.delete(key)
            except Exception as e:
                self.errors += 1
                print(f"L2 cache delete error: {e}")

        return l1_deleted or l2_deleted

    def exists(self, key: str) -> bool:
        """Check if key exists in any cache layer."""
        # Check L1 first (fastest)
        try:
            if self.l1_cache.contains(key):
                return True
        except Exception:
            pass

        # Check L2 if available
        if self.l2_cache:
            try:
                return self.l2_cache.exists(key)
            except Exception:
                pass

        return False

    def clear(self, pattern: str | None = None) -> int:
        """Clear keys from all cache layers."""
        total_cleared = 0

        # Clear L1
        try:
            total_cleared += self.l1_cache.clear(pattern)
        except Exception as e:
            self.errors += 1
            print(f"L1 cache clear error: {e}")

        # Clear L2
        if self.l2_cache:
            try:
                total_cleared += self.l2_cache.clear(pattern)
            except Exception as e:
                self.errors += 1
                print(f"L2 cache clear error: {e}")

        return total_cleared

    def keys(self, pattern: str | None = None, limit: int | None = None) -> list[str]:
        """Get keys from all cache layers."""
        all_keys = set()

        # Get keys from L1
        try:
            l1_keys = self.l1_cache.keys(pattern, limit)
            all_keys.update(l1_keys)
        except Exception as e:
            self.errors += 1
            print(f"L1 cache keys error: {e}")

        # Get keys from L2
        if self.l2_cache:
            try:
                l2_keys = self.l2_cache.keys(pattern, limit)
                all_keys.update(l2_keys)
            except Exception as e:
                self.errors += 1
                print(f"L2 cache keys error: {e}")

        result = list(all_keys)
        if limit:
            result = result[:limit]

        return result

    def stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.total_hits + self.total_misses
        hit_rate = (self.total_hits / total_requests * 100) if total_requests > 0 else 0

        stats = {
            # Overall statistics
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "errors": self.errors,
            # L1 statistics
            "l1_hits": self.l1_hits,
            "l1_misses": self.l1_misses,
            "l1_hit_rate": round(
                (self.l1_hits / total_requests * 100) if total_requests > 0 else 0, 2
            ),
            # L2 statistics
            "l2_enabled": self.l2_cache is not None,
            "l2_healthy": self._redis_healthy,
            "l2_hits": self.l2_hits,
            "l2_misses": self.l2_misses,
            "l2_hit_rate": round(
                (self.l2_hits / total_requests * 100) if total_requests > 0 else 0, 2
            ),
        }

        # Add L1 detailed stats
        try:
            l1_stats = self.l1_cache.stats()
            for key, value in l1_stats.items():
                stats[f"l1_{key}"] = value
        except Exception as e:
            stats["l1_error"] = str(e)

        # Add L2 detailed stats
        if self.l2_cache:
            try:
                l2_stats = self.l2_cache.stats()
                for key, value in l2_stats.items():
                    stats[f"l2_{key}"] = value
            except Exception as e:
                stats["l2_error"] = str(e)

        return stats

    def health_check(self) -> dict[str, Any]:
        """Comprehensive health check of all cache layers."""
        current_time = time.time()

        # Skip if recently checked
        if current_time - self._last_health_check < self.config.health_check_interval:
            return {
                "checked": False,
                "reason": "Recently checked",
                "l1_healthy": True,  # Memory cache is always healthy
                "l2_healthy": self._redis_healthy,
            }

        health = {
            "checked": True,
            "timestamp": current_time,
            "l1_healthy": True,  # Memory cache is assumed healthy
            "l2_healthy": False,
            "errors": [],
        }

        # Check L2 (Redis) health
        if self.l2_cache:
            try:
                l2_health = self.l2_cache.health_check()
                health["l2_healthy"] = l2_health.get("connected", False)
                if not health["l2_healthy"]:
                    health["errors"].append(f"Redis: {l2_health.get('error', 'Connection failed')}")
            except Exception as e:
                health["l2_healthy"] = False
                health["errors"].append(f"Redis health check failed: {e}")

        self._redis_healthy = health["l2_healthy"]
        self._last_health_check = current_time

        # Overall health
        health["overall_healthy"] = health["l1_healthy"] and (
            not self.config.redis_enabled or health["l2_healthy"]
        )

        return health

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern across all layers."""
        return self.clear(pattern)

    def get_memory_info(self) -> dict[str, Any]:
        """Get memory usage information."""
        memory_info = {}

        try:
            memory_info.update(self.l1_cache.memory_info())
        except Exception as e:
            memory_info["l1_memory_error"] = str(e)

        return memory_info


class MultiLayerCache:
    """
    Simplified multi-layer cache interface for easy integration.

    This class provides a simple interface that automatically handles
    the complexity of multi-layer caching with sensible defaults.
    """

    def __init__(
        self,
        memory_size: int = 1000,
        memory_ttl: int = 1800,
        redis_url: str | None = None,
        redis_ttl: int = 3600,
    ):
        # Configure cache manager
        config = CacheConfig(
            memory_max_size=memory_size,
            memory_ttl=memory_ttl,
            redis_ttl=redis_ttl,
            redis_enabled=redis_url is not None,
        )

        if redis_url and REDIS_AVAILABLE:
            # Parse Redis URL for configuration
            config.redis_config = RedisConfig()
            if redis_url.startswith("redis://"):
                # Simple URL parsing
                parts = redis_url.replace("redis://", "").split(":")
                if len(parts) >= 2:
                    config.redis_config.host = parts[0]
                    config.redis_config.port = int(parts[1].split("/")[0])

        self.manager = CacheManager(config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with optional default."""
        result = self.manager.get(key)
        return result if result is not None else default

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value with optional TTL."""
        return self.manager.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete key."""
        return self.manager.delete(key)

    def clear(self) -> int:
        """Clear all cache entries."""
        return self.manager.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.manager.stats()

    def health(self) -> dict[str, Any]:
        """Get health status."""
        return self.manager.health_check()
