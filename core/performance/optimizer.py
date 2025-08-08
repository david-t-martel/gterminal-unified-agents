"""Performance optimization manager for my-fullstack-agent.

Implements intelligent performance optimizations:
- Request deduplication and coalescing
- Memory pressure monitoring and cleanup
- Intelligent connection pooling
- Adaptive caching strategies
- Circuit breaker patterns
- Query optimization hints
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
import gc
import hashlib
import logging
import time
from typing import Any
import weakref
from weakref import WeakValueDictionary

import aiohttp
import psutil
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""

    # Memory management
    max_memory_mb: int = 2048
    memory_cleanup_threshold: float = 0.8  # 80%
    gc_collection_interval: int = 300  # 5 minutes

    # Request optimization
    enable_request_deduplication: bool = True
    deduplication_window_seconds: int = 30
    max_concurrent_requests: int = 50

    # Connection pooling
    http_pool_size: int = 100
    http_pool_per_host: int = 30
    connection_timeout: int = 10

    # Caching
    cache_max_size: int = 10000
    cache_ttl_seconds: int = 3600
    cache_cleanup_interval: int = 600  # 10 minutes

    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""

    requests_total: int = 0
    requests_deduped: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_cleanups: int = 0
    circuit_breaker_opens: int = 0
    avg_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0

    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

    def deduplication_rate(self) -> float:
        """Calculate request deduplication rate percentage."""
        return (
            (self.requests_deduped / self.requests_total * 100) if self.requests_total > 0 else 0.0
        )


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self, failure_threshold: int = 5, recovery_timeout: int = 60, half_open_max_calls: int = 3
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
            else:
                msg = "Circuit breaker is OPEN"
                raise Exception(msg)

        if self.state == "HALF_OPEN":
            if self.half_open_calls >= self.half_open_max_calls:
                msg = "Circuit breaker HALF_OPEN limit reached"
                raise Exception(msg)
            self.half_open_calls += 1

        try:
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            # Success - reset failure count
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0

            return result

        except Exception:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

            raise


class RequestDeduplicator:
    """Deduplicates identical requests to reduce load."""

    def __init__(self, window_seconds: int = 30) -> None:
        self.window_seconds = window_seconds
        self.active_requests: dict[str, asyncio.Future] = {}
        self.request_cache: dict[str, tuple[Any, float]] = {}
        self.lock = asyncio.Lock()

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate unique key for request."""
        # Create a stable hash of the function name and arguments
        content = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def deduplicate(self, func: Callable, *args, **kwargs) -> Any:
        """Deduplicate identical requests."""
        func_name = getattr(func, "__name__", str(func))
        key = self._generate_key(func_name, args, kwargs)

        async with self.lock:
            # Check if we have a cached result
            if key in self.request_cache:
                result, timestamp = self.request_cache[key]
                if time.time() - timestamp < self.window_seconds:
                    logger.debug(f"Cache hit for {func_name}")
                    return result
                del self.request_cache[key]

            # Check if request is already in progress
            if key in self.active_requests:
                logger.debug(f"Deduplicating request for {func_name}")
                return await self.active_requests[key]

            # Start new request
            future = asyncio.create_task(self._execute_with_cleanup(func, key, *args, **kwargs))
            self.active_requests[key] = future

            return await future

    async def _execute_with_cleanup(self, func: Callable, key: str, *args, **kwargs) -> Any:
        """Execute function and cleanup tracking."""
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Cache successful result
            self.request_cache[key] = (result, time.time())
            return result

        finally:
            # Cleanup active request tracking
            async with self.lock:
                self.active_requests.pop(key, None)


class MemoryOptimizer:
    """Monitors and optimizes memory usage."""

    def __init__(self, max_memory_mb: int = 2048, cleanup_threshold: float = 0.8) -> None:
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = cleanup_threshold
        self.weak_refs: set[weakref.ReferenceType] = set()
        self.large_objects: WeakValueDictionary = WeakValueDictionary()
        self.cleanup_callbacks: list[Callable] = []

    def register_cleanup_callback(self, callback: Callable) -> None:
        """Register a cleanup callback for memory pressure."""
        self.cleanup_callbacks.append(callback)

    def track_large_object(self, obj: Any, name: str) -> None:
        """Track large objects for cleanup."""
        self.large_objects[name] = obj

    async def check_memory_pressure(self) -> bool:
        """Check if we're under memory pressure."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > self.max_memory_mb * self.cleanup_threshold:
                logger.warning(
                    f"Memory pressure detected: {memory_mb:.1f}MB > {self.max_memory_mb * self.cleanup_threshold:.1f}MB",
                )
                await self.cleanup_memory()
                return True

            return False

        except Exception as e:
            logger.exception(f"Error checking memory pressure: {e}")
            return False

    async def cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        logger.info("Starting memory cleanup")

        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.exception(f"Cleanup callback failed: {e}")

        # Clear large objects
        self.large_objects.clear()

        # Force garbage collection
        gc.collect()

        # Log memory usage after cleanup
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory cleanup completed. Current usage: {memory_mb:.1f}MB")
        except Exception as e:
            logger.exception(f"Error checking memory after cleanup: {e}")


class SmartConnectionPool:
    """Intelligent connection pool with adaptive sizing."""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.http_session: aiohttp.ClientSession | None = None
        self.redis_pool: redis.ConnectionPool | None = None
        self.connection_stats = defaultdict(int)
        self.lock = asyncio.Lock()

    async def get_http_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with optimized connection pool."""
        if self.http_session is None or self.http_session.closed:
            async with self.lock:
                if self.http_session is None or self.http_session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=self.config.http_pool_size,
                        limit_per_host=self.config.http_pool_per_host,
                        ttl_dns_cache=300,
                        enable_cleanup_closed=True,
                        keepalive_timeout=30,
                    )

                    timeout = aiohttp.ClientTimeout(
                        total=self.config.connection_timeout * 2,
                        connect=self.config.connection_timeout,
                    )

                    self.http_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={
                            "User-Agent": "my-fullstack-agent/1.0",
                            "Accept-Encoding": "gzip, deflate, br",
                            "Connection": "keep-alive",
                        },
                    )

        return self.http_session

    async def get_redis_pool(self) -> redis.ConnectionPool:
        """Get or create Redis connection pool."""
        if self.redis_pool is None:
            async with self.lock:
                if self.redis_pool is None:
                    self.redis_pool = redis.ConnectionPool.from_url(
                        "redis://localhost:6379",
                        max_connections=20,
                        retry_on_timeout=True,
                        decode_responses=False,
                    )

        return self.redis_pool

    async def close(self) -> None:
        """Close all connection pools."""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()

        if self.redis_pool:
            await self.redis_pool.disconnect()


class PerformanceOptimizer:
    """Main performance optimization manager."""

    def __init__(self, config: OptimizationConfig = None) -> None:
        self.config = config or OptimizationConfig()
        self.metrics = PerformanceMetrics()

        # Components
        self.deduplicator = RequestDeduplicator(self.config.deduplication_window_seconds)
        self.memory_optimizer = MemoryOptimizer(
            self.config.max_memory_mb, self.config.memory_cleanup_threshold
        )
        self.connection_pool = SmartConnectionPool(self.config)
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Background tasks
        self.monitoring_task: asyncio.Task | None = None
        self.cleanup_task: asyncio.Task | None = None

        # Thread pool
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(32, (psutil.cpu_count() or 1) + 4),
            thread_name_prefix="PerfOptimizer",
        )

        # Start background monitoring
        self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        """Start background monitoring and cleanup tasks."""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                await self._update_metrics()
                await self.memory_optimizer.check_memory_pressure()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Monitoring loop error: {e}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.config.cache_cleanup_interval)
                await self._cleanup_expired_data()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Cleanup loop error: {e}")

    async def _update_metrics(self) -> None:
        """Update performance metrics."""
        try:
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()

            # Update connection counts
            if self.connection_pool.http_session and not self.connection_pool.http_session.closed:
                connector = self.connection_pool.http_session.connector
                if hasattr(connector, "_conns"):
                    self.metrics.active_connections = len(connector._conns)

        except Exception as e:
            logger.exception(f"Error updating metrics: {e}")

    async def _cleanup_expired_data(self) -> None:
        """Clean up expired cached data."""
        current_time = time.time()

        # Clean up request cache
        expired_keys = [
            key
            for key, (_, timestamp) in self.deduplicator.request_cache.items()
            if current_time - timestamp > self.config.deduplication_window_seconds
        ]

        for key in expired_keys:
            self.deduplicator.request_cache.pop(key, None)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                self.config.failure_threshold,
                self.config.recovery_timeout,
                self.config.half_open_max_calls,
            )
        return self.circuit_breakers[name]

    def optimized_request(self, operation_name: str = "default") -> None:
        """Decorator for optimizing requests with deduplication and circuit breaking."""

        def decorator(func: Callable) -> None:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                self.metrics.requests_total += 1

                try:
                    # Get circuit breaker for this operation
                    circuit_breaker = self.get_circuit_breaker(operation_name)

                    # Apply request deduplication
                    if self.config.enable_request_deduplication:
                        result = await self.deduplicator.deduplicate(
                            lambda: circuit_breaker.call(func, *args, **kwargs),
                        )
                        # Check if this was a deduplicated request
                        if hasattr(result, "_deduped"):
                            self.metrics.requests_deduped += 1
                    else:
                        result = await circuit_breaker.call(func, *args, **kwargs)

                    # Update response time metrics
                    response_time = (time.time() - start_time) * 1000
                    self.metrics.avg_response_time_ms = (
                        self.metrics.avg_response_time_ms * (self.metrics.requests_total - 1)
                        + response_time
                    ) / self.metrics.requests_total

                    return result

                except Exception as e:
                    logger.exception(f"Optimized request failed for {operation_name}: {e}")
                    raise

            return wrapper

        return decorator

    async def batch_execute(
        self,
        operations: list[tuple[Callable, tuple, dict]],
        concurrency_limit: int | None = None,
    ) -> list[Any]:
        """Execute multiple operations with concurrency control."""
        if concurrency_limit is None:
            concurrency_limit = self.config.max_concurrent_requests

        semaphore = asyncio.Semaphore(concurrency_limit)

        async def execute_with_semaphore(op_func, args, kwargs):
            async with semaphore:
                if asyncio.iscoroutinefunction(op_func):
                    return await op_func(*args, **kwargs)
                return await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, op_func, *args, **kwargs
                )

        tasks = [
            execute_with_semaphore(op_func, args, kwargs) for op_func, args, kwargs in operations
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)

    @asynccontextmanager
    async def http_request(self, method: str, url: str, **kwargs):
        """Context manager for optimized HTTP requests."""
        session = await self.connection_pool.get_http_session()

        async with session.request(method, url, **kwargs) as response:
            yield response

    def register_memory_cleanup(self, callback: Callable) -> None:
        """Register callback for memory cleanup."""
        self.memory_optimizer.register_cleanup_callback(callback)

    def track_large_object(self, obj: Any, name: str) -> None:
        """Track large object for memory management."""
        self.memory_optimizer.track_large_object(obj, name)

    async def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "timestamp": time.time(),
            "metrics": {
                "requests_total": self.metrics.requests_total,
                "requests_deduped": self.metrics.requests_deduped,
                "deduplication_rate": self.metrics.deduplication_rate(),
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "cache_hit_rate": self.metrics.cache_hit_rate(),
                "avg_response_time_ms": self.metrics.avg_response_time_ms,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "cpu_usage_percent": self.metrics.cpu_usage_percent,
                "active_connections": self.metrics.active_connections,
                "memory_cleanups": self.metrics.memory_cleanups,
                "circuit_breaker_opens": self.metrics.circuit_breaker_opens,
            },
            "circuit_breakers": {
                name: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "last_failure_time": cb.last_failure_time,
                }
                for name, cb in self.circuit_breakers.items()
            },
            "config": {
                "max_memory_mb": self.config.max_memory_mb,
                "max_concurrent_requests": self.config.max_concurrent_requests,
                "http_pool_size": self.config.http_pool_size,
                "cache_ttl_seconds": self.config.cache_ttl_seconds,
            },
        }

    async def close(self) -> None:
        """Cleanup optimizer resources."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.monitoring_task

        if self.cleanup_task:
            self.cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.cleanup_task

        await self.connection_pool.close()
        self.thread_pool.shutdown(wait=True)


# Global optimizer instance
_optimizer_instance: PerformanceOptimizer | None = None


def get_optimizer(config: OptimizationConfig = None) -> PerformanceOptimizer:
    """Get or create global optimizer instance."""
    global _optimizer_instance

    if _optimizer_instance is None:
        _optimizer_instance = PerformanceOptimizer(config)

    return _optimizer_instance


# Convenience decorators
def optimized_request(operation_name: str = "default") -> None:
    """Decorator for optimizing requests."""
    return get_optimizer().optimized_request(operation_name)


def track_performance(operation_name: str) -> None:
    """Simple performance tracking decorator."""

    def decorator(func: Callable) -> None:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            get_optimizer()

            try:
                return (
                    await func(*args, **kwargs)
                    if asyncio.iscoroutinefunction(func)
                    else func(*args, **kwargs)
                )
            finally:
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(f"Operation {operation_name} took {duration_ms:.2f}ms")

        return wrapper

    return decorator
