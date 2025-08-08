"""
Redis-based cache implementation with connection pooling and async support.

This module provides Redis caching patterns extracted from my-fullstack-agent,
including:
- Connection pooling and management
- Async Redis operations
- Error handling and resilience
- Statistics and monitoring
- TTL and expiration management
"""

from dataclasses import dataclass
import json
import time
from typing import Any

try:
    import redis
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class RedisConfig:
    """Redis connection configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    key_prefix: str = ""
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30


class RedisCache:
    """
    Redis-backed cache with connection pooling and resilience.

    Features:
    - Connection pooling
    - Automatic reconnection
    - TTL management
    - JSON serialization support
    - Comprehensive error handling
    - Statistics tracking
    """

    def __init__(self, config: RedisConfig | None = None, default_ttl: int = 3600):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for RedisCache")

        self.config = config or RedisConfig()
        self.default_ttl = default_ttl
        self._pool = None
        self._client = None

        # Statistics
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.errors = 0

        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize Redis connection pool."""
        try:
            self._pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
                decode_responses=True,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            # Test connection
            self._client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self._client = None
            self._pool = None

    def _get_key(self, key: str) -> str:
        """Get prefixed key."""
        return f"{self.config.key_prefix}{key}" if self.config.key_prefix else key

    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self._client:
            return False
        try:
            self._client.ping()
            return True
        except:
            return False

    def get(self, key: str) -> str | dict | list | None:
        """Get value from Redis cache."""
        if not self._client:
            self.errors += 1
            return None

        try:
            redis_key = self._get_key(key)
            value = self._client.get(redis_key)

            if value is None:
                self.misses += 1
                return None

            self.hits += 1

            # Try to parse as JSON, return string if parsing fails
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except Exception as e:
            self.errors += 1
            print(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: str | dict | list | int | float, ttl: int | None = None) -> bool:
        """Set value in Redis cache."""
        if not self._client:
            self.errors += 1
            return False

        try:
            redis_key = self._get_key(key)
            ttl = ttl or self.default_ttl

            # Serialize complex objects as JSON
            if isinstance(value, dict | list):
                value = json.dumps(value, ensure_ascii=False)
            elif not isinstance(value, str):
                value = str(value)

            result = self._client.setex(redis_key, ttl, value)
            if result:
                self.sets += 1
                return True
            return False

        except Exception as e:
            self.errors += 1
            print(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self._client:
            return False

        try:
            redis_key = self._get_key(key)
            result = self._client.delete(redis_key)
            return result > 0
        except Exception as e:
            self.errors += 1
            print(f"Redis delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self._client:
            return False

        try:
            redis_key = self._get_key(key)
            return self._client.exists(redis_key) > 0
        except Exception as e:
            self.errors += 1
            print(f"Redis exists error: {e}")
            return False

    def clear(self, pattern: str | None = None) -> int:
        """Clear keys matching pattern."""
        if not self._client:
            return 0

        try:
            if pattern:
                search_pattern = self._get_key(f"*{pattern}*")
                keys = self._client.keys(search_pattern)
            # Clear all keys with prefix
            elif self.config.key_prefix:
                keys = self._client.keys(f"{self.config.key_prefix}*")
            else:
                print("Warning: Clearing all keys in database")
                keys = self._client.keys("*")

            if keys:
                return self._client.delete(*keys)
            return 0

        except Exception as e:
            self.errors += 1
            print(f"Redis clear error: {e}")
            return 0

    def keys(self, pattern: str | None = None, limit: int | None = None) -> list[str]:
        """Get keys matching pattern."""
        if not self._client:
            return []

        try:
            if pattern:
                search_pattern = self._get_key(f"*{pattern}*")
            else:
                search_pattern = f"{self.config.key_prefix}*" if self.config.key_prefix else "*"

            keys = self._client.keys(search_pattern)

            # Remove prefix from keys
            if self.config.key_prefix:
                keys = [k[len(self.config.key_prefix) :] for k in keys]

            if limit:
                keys = keys[:limit]

            return keys

        except Exception as e:
            self.errors += 1
            print(f"Redis keys error: {e}")
            return []

    def ttl(self, key: str) -> int:
        """Get TTL for key in seconds."""
        if not self._client:
            return -1

        try:
            redis_key = self._get_key(key)
            return self._client.ttl(redis_key)
        except Exception as e:
            self.errors += 1
            print(f"Redis TTL error: {e}")
            return -1

    def stats(self) -> dict[str, int | float | str]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        base_stats = {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "errors": self.errors,
            "hit_rate_percent": round(hit_rate, 2),
            "connected": self.is_connected(),
        }

        # Add Redis server info if connected
        if self._client and self.is_connected():
            try:
                info = self._client.info()
                base_stats.update(
                    {
                        "redis_version": info.get("redis_version", "unknown"),
                        "used_memory_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                        "connected_clients": info.get("connected_clients", 0),
                        "total_commands_processed": info.get("total_commands_processed", 0),
                    }
                )
            except Exception as e:
                base_stats["redis_info_error"] = str(e)

        return base_stats

    def health_check(self) -> dict[str, Any]:
        """Comprehensive health check."""
        health = {"connected": False, "latency_ms": None, "error": None}

        if not self._client:
            health["error"] = "No Redis client"
            return health

        try:
            start_time = time.time()
            self._client.ping()
            latency = (time.time() - start_time) * 1000

            health.update({"connected": True, "latency_ms": round(latency, 2)})

        except Exception as e:
            health["error"] = str(e)

        return health


class AsyncRedisCache:
    """
    Async Redis cache implementation for high-performance applications.

    Features:
    - Full async/await support
    - Connection pooling
    - Batch operations
    - Pipeline support
    """

    def __init__(self, config: RedisConfig | None = None, default_ttl: int = 3600):
        if not REDIS_AVAILABLE:
            raise ImportError("redis[hiredis] package is required for AsyncRedisCache")

        self.config = config or RedisConfig()
        self.default_ttl = default_ttl
        self._pool = None
        self._client = None

        # Statistics
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.errors = 0

    async def initialize(self):
        """Initialize async Redis connection."""
        try:
            self._pool = aioredis.ConnectionPool.from_url(
                f"redis://{self.config.host}:{self.config.port}/{self.config.db}",
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
                decode_responses=True,
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
            await self._client.ping()
        except Exception as e:
            print(f"Async Redis connection failed: {e}")
            self._client = None

    def _get_key(self, key: str) -> str:
        """Get prefixed key."""
        return f"{self.config.key_prefix}{key}" if self.config.key_prefix else key

    async def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self._client:
            return False
        try:
            await self._client.ping()
            return True
        except:
            return False

    async def get(self, key: str) -> str | dict | list | None:
        """Get value from Redis cache."""
        if not self._client:
            self.errors += 1
            return None

        try:
            redis_key = self._get_key(key)
            value = await self._client.get(redis_key)

            if value is None:
                self.misses += 1
                return None

            self.hits += 1

            # Try to parse as JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except Exception as e:
            self.errors += 1
            print(f"Async Redis get error: {e}")
            return None

    async def set(
        self, key: str, value: str | dict | list | int | float, ttl: int | None = None
    ) -> bool:
        """Set value in Redis cache."""
        if not self._client:
            self.errors += 1
            return False

        try:
            redis_key = self._get_key(key)
            ttl = ttl or self.default_ttl

            # Serialize complex objects as JSON
            if isinstance(value, dict | list):
                value = json.dumps(value, ensure_ascii=False)
            elif not isinstance(value, str):
                value = str(value)

            result = await self._client.setex(redis_key, ttl, value)
            if result:
                self.sets += 1
                return True
            return False

        except Exception as e:
            self.errors += 1
            print(f"Async Redis set error: {e}")
            return False

    async def batch_get(self, keys: list[str]) -> dict[str, str | dict | list]:
        """Get multiple values efficiently."""
        if not self._client or not keys:
            return {}

        try:
            redis_keys = [self._get_key(key) for key in keys]
            values = await self._client.mget(redis_keys)

            results = {}
            for _i, (original_key, value) in enumerate(zip(keys, values, strict=False)):
                if value is not None:
                    self.hits += 1
                    try:
                        results[original_key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        results[original_key] = value
                else:
                    self.misses += 1

            return results

        except Exception as e:
            self.errors += 1
            print(f"Async Redis batch_get error: {e}")
            return {}

    async def batch_set(
        self, items: dict[str, str | dict | list | int | float], ttl: int | None = None
    ) -> int:
        """Set multiple values efficiently."""
        if not self._client or not items:
            return 0

        try:
            ttl = ttl or self.default_ttl
            pipe = self._client.pipeline()

            for key, value in items.items():
                redis_key = self._get_key(key)

                # Serialize complex objects as JSON
                if isinstance(value, dict | list):
                    value = json.dumps(value, ensure_ascii=False)
                elif not isinstance(value, str):
                    value = str(value)

                pipe.setex(redis_key, ttl, value)

            results = await pipe.execute()
            success_count = sum(1 for result in results if result)
            self.sets += success_count
            return success_count

        except Exception as e:
            self.errors += 1
            print(f"Async Redis batch_set error: {e}")
            return 0

    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()

    async def stats(self) -> dict[str, int | float | str]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "errors": self.errors,
            "hit_rate_percent": round(hit_rate, 2),
            "connected": await self.is_connected(),
        }
