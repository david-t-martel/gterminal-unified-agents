#!/usr/bin/env python3
"""
Shared Redis Utilities for All AI Agents and Tools

This module provides consistent Redis integration patterns for Claude Code,
ruff-claude.sh, and all agentic tools in the development environment.

Features:
- Connection management with failover
- Session and memory management for agents
- Caching layer with automatic TTL
- Task queue for inter-agent communication
- Performance monitoring and health checks
- Error handling with fallback strategies
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from functools import wraps
import hashlib
import json
import logging
import os
import time
from typing import Any, Optional
import uuid

try:
    import redis
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Configuration
@lru_cache(maxsize=1)
def get_redis_url() -> str:
    """Get Redis URL from environment with default."""
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


DB_ALLOCATION = {
    "claude_sessions": 0,  # Claude conversation contexts
    "claude_cache": 1,  # Claude tool results cache
    "ruff_cache": 2,  # Ruff linting results cache
    "agent_memory": 3,  # Agent long-term memory
    "agent_sessions": 4,  # Agent session data
    "task_queue": 5,  # Inter-agent task queue
    "api_cache": 6,  # API response cache
    "file_analysis": 7,  # File analysis results
    "performance_metrics": 8,  # Performance tracking
    "error_logs": 9,  # Error tracking
}


# Data Classes
@dataclass
class RedisConfig:
    """Redis connection configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    max_connections: int = 20


class TaskStatus(Enum):
    """Task status for agent coordination."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentTask:
    """Task for inter-agent communication."""

    id: str
    agent_id: str
    task_type: str
    payload: dict[str, Any]
    status: TaskStatus
    created_at: float
    completed_at: float | None = None
    result: dict[str, Any] | None = None


# Decorators
def redis_retry(retries: int = 3, delay: float = 1.0):
    """Decorator for Redis operations with retry logic."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not REDIS_AVAILABLE:
                logger.warning("Redis not available, skipping operation")
                return None

            last_exception = None

            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except (redis.ConnectionError, redis.TimeoutError) as e:
                    last_exception = e
                    if attempt < retries - 1:
                        await asyncio.sleep(delay * (2**attempt))
                        logger.warning(
                            f"Redis operation {func.__name__} failed, retrying... ({attempt + 1}/{retries})"
                        )
                    continue
                except Exception as e:
                    logger.exception(
                        f"Redis operation {func.__name__} failed with non-recoverable error: {e}"
                    )
                    raise

            logger.error(f"Redis operation {func.__name__} failed after {retries} attempts")
            raise last_exception

        return wrapper

    return decorator


# Connection Management
class RedisManager:
    """Singleton Redis connection manager with failover support."""

    _instance: Optional["RedisManager"] = None
    _clients: dict[str, redis.Redis | aioredis.Redis] = {}

    def __new__(cls) -> "RedisManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def get_client(
        cls,
        purpose: str = "default",
        db_override: int | None = None,
        async_client: bool = True,
        config: RedisConfig | None = None,
    ) -> redis.Redis | aioredis.Redis:
        """Get or create Redis client for specific purpose."""
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required")

        cls()
        config = config or RedisConfig()

        # Determine database from purpose
        db = db_override if db_override is not None else DB_ALLOCATION.get(purpose, 0)
        client_key = f"{purpose}_{db}_{async_client}"

        if client_key not in cls._clients:
            try:
                if async_client:
                    cls._clients[client_key] = await aioredis.from_url(
                        f"{get_redis_url()}/{db}",
                        password=config.password,
                        socket_timeout=config.socket_timeout,
                        socket_connect_timeout=config.socket_connect_timeout,
                        retry_on_timeout=config.retry_on_timeout,
                        health_check_interval=config.health_check_interval,
                        max_connections=config.max_connections,
                        decode_responses=True,
                    )
                    await cls._clients[client_key].ping()
                else:
                    cls._clients[client_key] = redis.Redis(
                        host=config.host,
                        port=config.port,
                        db=db,
                        password=config.password,
                        socket_timeout=config.socket_timeout,
                        socket_connect_timeout=config.socket_connect_timeout,
                        retry_on_timeout=config.retry_on_timeout,
                        health_check_interval=config.health_check_interval,
                        decode_responses=True,
                    )
                    cls._clients[client_key].ping()

                logger.info(f"Created Redis client for {purpose} (DB {db})")

            except Exception as e:
                logger.exception(f"Failed to create Redis client for {purpose}: {e}")
                raise

        return cls._clients[client_key]

    @classmethod
    async def close_all(cls):
        """Close all Redis connections."""
        for client_key, client in list(cls._clients.items()):
            try:
                if hasattr(client, "close"):
                    await client.close()
                else:
                    client.close()
                logger.info(f"Closed Redis client: {client_key}")
            except Exception as e:
                logger.exception(f"Error closing Redis client {client_key}: {e}")
        cls._clients.clear()


# Context Managers
@asynccontextmanager
async def redis_client(purpose: str = "default", **kwargs):
    """Context manager for Redis client."""
    client = await RedisManager.get_client(purpose, **kwargs)
    try:
        yield client
    except Exception as e:
        logger.exception(f"Redis operation failed: {e}")
        raise
    # Client cleanup handled by RedisManager


# Session Management
class AgentSessionManager:
    """Session management for AI agents."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.purpose = "agent_sessions"

    async def save_session(self, session_data: dict[str, Any], ttl: int = 3600):
        """Save agent session data."""
        async with redis_client(self.purpose) as redis:
            key = f"session:{self.agent_id}"
            session_data["last_updated"] = time.time()
            await redis.setex(key, ttl, json.dumps(session_data, default=str))

    async def load_session(self) -> dict[str, Any]:
        """Load agent session data."""
        async with redis_client(self.purpose) as redis:
            key = f"session:{self.agent_id}"
            data = await redis.get(key)
            return json.loads(data) if data else {}

    async def clear_session(self):
        """Clear agent session."""
        async with redis_client(self.purpose) as redis:
            key = f"session:{self.agent_id}"
            await redis.delete(key)


class AgentMemoryManager:
    """Long-term memory management for AI agents."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.purpose = "agent_memory"

    async def store_memory(self, key: str, value: Any, ttl: int | None = None):
        """Store long-term memory."""
        async with redis_client(self.purpose) as redis:
            memory_key = f"memory:{self.agent_id}"
            if ttl:
                # Store with TTL in separate key
                await redis.setex(f"{memory_key}:{key}", ttl, json.dumps(value, default=str))
            else:
                # Store in hash for permanent memory
                await redis.hset(memory_key, key, json.dumps(value, default=str))

    async def recall_memory(self, key: str) -> Any:
        """Recall from long-term memory."""
        async with redis_client(self.purpose) as redis:
            memory_key = f"memory:{self.agent_id}"

            # Try TTL key first
            ttl_data = await redis.get(f"{memory_key}:{key}")
            if ttl_data:
                return json.loads(ttl_data)

            # Try permanent memory
            data = await redis.hget(memory_key, key)
            return json.loads(data) if data else None

    async def get_all_memories(self) -> dict[str, Any]:
        """Get all memories for agent."""
        async with redis_client(self.purpose) as redis:
            memory_key = f"memory:{self.agent_id}"
            data = await redis.hgetall(memory_key)
            return {k: json.loads(v) for k, v in data.items()}


# Caching Layer
class SmartCache:
    """Intelligent caching layer with automatic TTL and fallback."""

    def __init__(self, purpose: str = "claude_cache", default_ttl: int = 300):
        self.purpose = purpose
        self.default_ttl = default_ttl
        self.fallback_storage = {}  # In-memory fallback

    def _make_key(self, namespace: str, key: str, params: dict | None = None) -> str:
        """Generate cache key with optional parameters."""
        if params:
            param_string = json.dumps(params, sort_keys=True)
            param_hash = hashlib.sha256(param_string.encode()).hexdigest()[:16]
            return f"{namespace}:{key}:{param_hash}"
        return f"{namespace}:{key}"

    @redis_retry(retries=2, delay=0.5)
    async def get(self, namespace: str, key: str, params: dict | None = None) -> Any:
        """Get cached value with fallback."""
        cache_key = self._make_key(namespace, key, params)

        try:
            async with redis_client(self.purpose) as redis:
                data = await redis.get(cache_key)
                if data:
                    return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis cache get failed for {cache_key}: {e}")

        # Fallback to in-memory storage
        return self.fallback_storage.get(cache_key)

    @redis_retry(retries=2, delay=0.5)
    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: int | None = None,
        params: dict | None = None,
    ):
        """Set cached value with fallback."""
        cache_key = self._make_key(namespace, key, params)
        ttl = ttl or self.default_ttl

        # Always store in fallback for immediate access
        self.fallback_storage[cache_key] = value

        try:
            async with redis_client(self.purpose) as redis:
                await redis.setex(cache_key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning(f"Redis cache set failed for {cache_key}: {e}")

    async def invalidate(self, namespace: str, pattern: str = "*"):
        """Invalidate cache entries matching pattern."""
        try:
            async with redis_client(self.purpose) as redis:
                search_pattern = f"{namespace}:{pattern}"
                keys = []
                async for key in redis.scan_iter(match=search_pattern):
                    keys.append(key)
                if keys:
                    await redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis cache invalidation failed: {e}")

        # Clear matching fallback entries
        keys_to_remove = [k for k in self.fallback_storage if k.startswith(f"{namespace}:")]
        for key in keys_to_remove:
            del self.fallback_storage[key]


# Task Queue for Agent Coordination
class AgentTaskQueue:
    """Task queue for inter-agent communication and coordination."""

    def __init__(self):
        self.purpose = "task_queue"

    async def enqueue_task(
        self, agent_id: str, task_type: str, payload: dict[str, Any], priority: int = 0
    ) -> str:
        """Add task to queue with priority."""
        task_id = str(uuid.uuid4())
        task = AgentTask(
            id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            payload=payload,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )

        async with redis_client(self.purpose) as redis:
            # Add to priority queue
            queue_key = f"queue:{task_type}"
            await redis.zadd(queue_key, {task_id: priority})

            # Store task details
            task_dict = asdict(task)
            task_dict["status"] = task_dict["status"].value  # Convert enum to string
            await redis.hset(
                f"task:{task_id}",
                mapping={"data": json.dumps(task_dict), "status": task.status.value},
            )

            # Set TTL for task cleanup (24 hours)
            await redis.expire(f"task:{task_id}", 86400)

        return task_id

    async def dequeue_task(self, task_type: str, timeout: int = 30) -> AgentTask | None:
        """Get highest priority task from queue."""
        async with redis_client(self.purpose) as redis:
            queue_key = f"queue:{task_type}"

            # Get highest priority task (highest score)
            result = await redis.bzpopmax(queue_key, timeout)
            if result:
                _, task_id, _ = result

                # Get task data
                task_data = await redis.hget(f"task:{task_id}", "data")
                if task_data:
                    task_dict = json.loads(task_data)
                    task = AgentTask(**task_dict)

                    # Mark as running
                    await redis.hset(f"task:{task_id}", "status", TaskStatus.RUNNING.value)

                    return task

        return None

    async def complete_task(self, task_id: str, result: dict[str, Any]):
        """Mark task as completed with result."""
        async with redis_client(self.purpose) as redis:
            await redis.hset(
                f"task:{task_id}",
                mapping={
                    "status": TaskStatus.COMPLETED.value,
                    "result": json.dumps(result, default=str),
                    "completed_at": str(time.time()),
                },
            )

    async def fail_task(self, task_id: str, error: str):
        """Mark task as failed with error."""
        async with redis_client(self.purpose) as redis:
            await redis.hset(
                f"task:{task_id}",
                mapping={
                    "status": TaskStatus.FAILED.value,
                    "error": error,
                    "completed_at": str(time.time()),
                },
            )

    async def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get task status and details."""
        async with redis_client(self.purpose) as redis:
            data = await redis.hgetall(f"task:{task_id}")
            if data:
                task_info = json.loads(data.get("data", "{}"))
                task_info["status"] = data.get("status")
                task_info["result"] = json.loads(data.get("result", "null"))
                task_info["error"] = data.get("error")
                return task_info
        return None


# Performance Monitoring
class RedisPerformanceMonitor:
    """Performance monitoring for Redis operations."""

    def __init__(self):
        self.purpose = "performance_metrics"

    @asynccontextmanager
    async def measure_operation(self, operation_name: str, agent_id: str = "unknown"):
        """Context manager to measure operation performance."""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            # Record error
            await self._record_error(operation_name, agent_id, str(e))
            raise
        finally:
            duration = time.time() - start_time
            await self._record_metric(operation_name, agent_id, duration)

    async def _record_metric(self, operation: str, agent_id: str, duration: float):
        """Record performance metric."""
        try:
            async with redis_client(self.purpose) as redis:
                timestamp = int(time.time())
                metric_key = f"metrics:{operation}:{agent_id}"

                # Store with timestamp as score
                await redis.zadd(metric_key, {timestamp: duration})

                # Keep only last 1000 measurements
                await redis.zremrangebyrank(metric_key, 0, -1001)

                # Set TTL (7 days)
                await redis.expire(metric_key, 604800)
        except Exception as e:
            logger.warning(f"Failed to record metric: {e}")

    async def _record_error(self, operation: str, agent_id: str, error: str):
        """Record error for monitoring."""
        try:
            async with redis_client("error_logs") as redis:
                error_key = f"errors:{operation}:{agent_id}"
                timestamp = int(time.time())

                await redis.zadd(error_key, {f"{timestamp}:{error}": timestamp})

                # Keep only last 100 errors
                await redis.zremrangebyrank(error_key, 0, -101)

                # Set TTL (7 days)
                await redis.expire(error_key, 604800)
        except Exception as e:
            logger.warning(f"Failed to record error: {e}")

    async def get_operation_stats(self, operation: str, agent_id: str = "*") -> dict[str, Any]:
        """Get performance statistics for operation."""
        try:
            async with redis_client(self.purpose) as redis:
                if agent_id == "*":
                    # Aggregate across all agents
                    pattern = f"metrics:{operation}:*"
                    keys = []
                    async for key in redis.scan_iter(match=pattern):
                        keys.append(key)
                else:
                    keys = [f"metrics:{operation}:{agent_id}"]

                all_durations = []
                for key in keys:
                    durations = await redis.zrange(key, 0, -1, withscores=True)
                    all_durations.extend([float(score) for _, score in durations])

                if not all_durations:
                    return {"operation": operation, "agent_id": agent_id, "count": 0}

                return {
                    "operation": operation,
                    "agent_id": agent_id,
                    "count": len(all_durations),
                    "avg_ms": round(sum(all_durations) * 1000 / len(all_durations), 2),
                    "min_ms": round(min(all_durations) * 1000, 2),
                    "max_ms": round(max(all_durations) * 1000, 2),
                    "p95_ms": round(
                        sorted(all_durations)[int(len(all_durations) * 0.95)] * 1000, 2
                    ),
                }
        except Exception as e:
            logger.exception(f"Failed to get operation stats: {e}")
            return {"error": str(e)}


# Health Monitoring
class RedisHealthChecker:
    """Health monitoring for Redis infrastructure."""

    async def comprehensive_health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "databases": {},
            "performance": {},
            "errors": [],
        }

        # Check each database
        for purpose, db_num in DB_ALLOCATION.items():
            try:
                async with redis_client(purpose) as redis:
                    start_time = time.time()
                    await redis.ping()
                    latency = (time.time() - start_time) * 1000

                    info = await redis.info("memory")

                    health["databases"][purpose] = {
                        "db": db_num,
                        "status": "healthy",
                        "latency_ms": round(latency, 2),
                        "memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                    }
            except Exception as e:
                health["databases"][purpose] = {
                    "db": db_num,
                    "status": "unhealthy",
                    "error": str(e),
                }
                health["errors"].append(f"{purpose}: {e!s}")

        # Overall status
        unhealthy_dbs = [
            db for db, status in health["databases"].items() if status.get("status") != "healthy"
        ]

        if unhealthy_dbs:
            health["overall_status"] = (
                "degraded" if len(unhealthy_dbs) < len(DB_ALLOCATION) else "unhealthy"
            )

        return health

    async def get_memory_usage(self) -> dict[str, Any]:
        """Get Redis memory usage across all databases."""
        try:
            async with redis_client("claude_sessions") as redis:
                info = await redis.info("memory")

                return {
                    "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                    "used_memory_peak_mb": round(info.get("used_memory_peak", 0) / 1024 / 1024, 2),
                    "maxmemory_mb": round(info.get("maxmemory", 0) / 1024 / 1024, 2),
                    "mem_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0),
                    "maxmemory_policy": info.get("maxmemory_policy", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                }
        except Exception as e:
            logger.exception(f"Failed to get memory usage: {e}")
            return {"error": str(e)}


# Convenience Functions
async def quick_cache_get(namespace: str, key: str, purpose: str = "claude_cache") -> Any:
    """Quick cache get operation."""
    cache = SmartCache(purpose)
    return await cache.get(namespace, key)


async def quick_cache_set(
    namespace: str, key: str, value: Any, ttl: int = 300, purpose: str = "claude_cache"
):
    """Quick cache set operation."""
    cache = SmartCache(purpose)
    await cache.set(namespace, key, value, ttl)


async def create_agent_session(agent_id: str) -> AgentSessionManager:
    """Create agent session manager."""
    return AgentSessionManager(agent_id)


async def create_agent_memory(agent_id: str) -> AgentMemoryManager:
    """Create agent memory manager."""
    return AgentMemoryManager(agent_id)


async def create_task_queue() -> AgentTaskQueue:
    """Create agent task queue."""
    return AgentTaskQueue()


async def get_performance_monitor() -> RedisPerformanceMonitor:
    """Get performance monitor."""
    return RedisPerformanceMonitor()


async def health_check() -> dict[str, Any]:
    """Perform Redis health check."""
    checker = RedisHealthChecker()
    return await checker.comprehensive_health_check()


# Cleanup
async def cleanup_redis_connections():
    """Cleanup all Redis connections."""
    await RedisManager.close_all()


# For backward compatibility and ease of import
__all__ = [
    "AgentMemoryManager",
    "AgentSessionManager",
    "AgentTaskQueue",
    "RedisHealthChecker",
    "RedisManager",
    "RedisPerformanceMonitor",
    "SmartCache",
    "cleanup_redis_connections",
    "create_agent_memory",
    "create_agent_session",
    "create_task_queue",
    "get_performance_monitor",
    "health_check",
    "quick_cache_get",
    "quick_cache_set",
    "redis_client",
]
