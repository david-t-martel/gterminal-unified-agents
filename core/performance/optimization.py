"""Performance optimization orchestrator.

Integrates caching, connection pooling, and metrics to provide a unified
performance optimization layer for the fullstack agent.
"""

import asyncio
from collections.abc import Callable
import logging
import os
import time
from typing import Any
import uuid

from .cache import CacheManager
from .cache import MemoryCache
from .cache import RedisCache
from .connection_pool import ConnectionPoolManager
from .metrics import PerformanceMetrics
from .metrics import PerformanceMonitor

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""

    def __init__(
        self,
        redis_url: str | None = None,
        enable_redis: bool = True,
        cache_config: dict[str, Any] | None = None,
        connection_config: dict[str, Any] | None = None,
        alert_callback: Callable | None = None,
    ) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.enable_redis = enable_redis and self._check_redis_available()

        # Initialize components
        self.cache_manager = self._setup_cache_manager(cache_config)
        self.connection_manager = self._setup_connection_manager(connection_config)
        self.metrics = PerformanceMetrics(PerformanceMonitor(alert_callback))

        # Performance state
        self._request_cache: dict[str, Any] = {}
        self._batch_operations: dict[str, list[Any]] = {}
        self._batch_timers: dict[str, asyncio.Task] = {}

        logger.info(f"PerformanceOptimizer initialized - Redis: {self.enable_redis}")

    def _check_redis_available(self) -> bool:
        """Check if Redis is available."""
        try:
            import redis

            client = redis.Redis.from_url(self.redis_url, socket_timeout=1)
            client.ping()
            return True
        except Exception:
            logger.warning("Redis not available, using memory-only caching")
            return False

    def _setup_cache_manager(self, config: dict[str, Any] | None) -> CacheManager:
        """Setup cache manager with appropriate backends."""
        config = config or {}

        # Memory cache configuration
        memory_config = config.get("memory", {})
        memory_cache = MemoryCache(
            max_size=memory_config.get("max_size", 500),
            default_ttl=memory_config.get("default_ttl", 1800),  # 30 minutes
        )

        # Redis cache configuration
        redis_cache = None
        if self.enable_redis:
            redis_config = config.get("redis", {})
            redis_cache = RedisCache(
                redis_url=self.redis_url,
                key_prefix=redis_config.get("key_prefix", "fullstack_agent:"),
                default_ttl=redis_config.get("default_ttl", 3600),  # 1 hour
                max_connections=redis_config.get("max_connections", 20),
            )

        return CacheManager(
            memory_cache=memory_cache,
            redis_cache=redis_cache,
            enable_l2_cache=self.enable_redis,
        )

    def _setup_connection_manager(self, config: dict[str, Any] | None) -> ConnectionPoolManager:
        """Setup connection pool manager."""
        config = config or {}

        return ConnectionPoolManager(
            http_pool_config=config.get("http", {}),
            vertex_ai_config=config.get("vertex_ai", {}),
            google_search_config=config.get("google_search", {}),
            postgresql_config=config.get("postgresql", {}),
        )

    async def cache_ai_response(
        self,
        model: str,
        prompt: str,
        response: Any,
        ttl: float | None = None,
    ) -> None:
        """Cache AI model response."""
        cache_key = f"ai_response:{model}:{hash(prompt)}"
        await self.cache_manager.set(cache_key, response, ttl or 3600)  # 1 hour default
        await self.metrics.cache_hit("ai_response")
        logger.debug(f"Cached AI response for model {model}")

    async def get_cached_ai_response(self, model: str, prompt: str) -> Any | None:
        """Get cached AI model response."""
        cache_key = f"ai_response:{model}:{hash(prompt)}"
        response = await self.cache_manager.get(cache_key)

        if response is not None:
            await self.metrics.cache_hit("ai_response")
            logger.debug(f"Cache hit for AI response - model: {model}")
            return response

        await self.metrics.cache_miss("ai_response")
        return None

    async def cache_search_results(
        self,
        query: str,
        results: Any,
        ttl: float | None = None,
    ) -> None:
        """Cache search results."""
        cache_key = f"search:{hash(query)}"
        await self.cache_manager.set(cache_key, results, ttl or 1800)  # 30 minutes default
        logger.debug(f"Cached search results for query: {query[:50]}...")

    async def get_cached_search_results(self, query: str) -> Any | None:
        """Get cached search results."""
        cache_key = f"search:{hash(query)}"
        results = await self.cache_manager.get(cache_key)

        if results is not None:
            await self.metrics.cache_hit("search")
            logger.debug(f"Cache hit for search: {query[:50]}...")
            return results

        await self.metrics.cache_miss("search")
        return None

    async def cache_file_content(
        self,
        file_path: str,
        content: Any,
        ttl: float | None = None,
    ) -> None:
        """Cache file content."""
        cache_key = f"file:{file_path}"
        await self.cache_manager.set(cache_key, content, ttl or 600)  # 10 minutes default
        logger.debug(f"Cached file content: {file_path}")

    async def get_cached_file_content(self, file_path: str) -> Any | None:
        """Get cached file content."""
        cache_key = f"file:{file_path}"
        content = await self.cache_manager.get(cache_key)

        if content is not None:
            await self.metrics.cache_hit("file")
            logger.debug(f"Cache hit for file: {file_path}")
            return content

        await self.metrics.cache_miss("file")
        return None

    async def optimized_ai_call(self, model: str, prompt: str, **kwargs) -> Any:
        """Make optimized AI call with caching and connection pooling."""
        request_id = str(uuid.uuid4())
        start_time = await self.metrics.start_request(request_id, f"ai_call_{model}")

        try:
            # Check cache first
            cached_response = await self.get_cached_ai_response(model, prompt)
            if cached_response is not None:
                await self.metrics.end_request(request_id, f"ai_call_{model}", start_time, True)
                return cached_response

            # Use request coalescing for identical requests
            coalesce_key = f"ai:{model}:{hash(prompt)}"

            async def make_ai_request():
                response = await self.connection_manager.vertex_ai_pool.generate_content(
                    model, prompt, **kwargs
                )
                # Cache the response
                await self.cache_ai_response(model, prompt, response)
                return response

            response = await self.metrics.coalesce_request(coalesce_key, make_ai_request)
            await self.metrics.end_request(request_id, f"ai_call_{model}", start_time, True)
            return response

        except Exception as e:
            await self.metrics.end_request(request_id, f"ai_call_{model}", start_time, False)
            logger.exception(f"Optimized AI call failed: {e}")
            raise

    async def optimized_search(self, query: str, num_results: int = 10, **kwargs) -> Any:
        """Make optimized search with caching and rate limiting."""
        request_id = str(uuid.uuid4())
        start_time = await self.metrics.start_request(request_id, "search")

        try:
            # Check cache first
            cached_results = await self.get_cached_search_results(query)
            if cached_results is not None:
                await self.metrics.end_request(request_id, "search", start_time, True)
                return cached_results

            # Use request coalescing for identical searches
            coalesce_key = f"search:{hash(query)}"

            async def make_search_request():
                results = await self.connection_manager.google_search_pool.search(
                    query=query,
                    num_results=num_results,
                    http_pool=self.connection_manager.http_pool,
                    **kwargs,
                )
                # Cache the results
                await self.cache_search_results(query, results)
                return results

            results = await self.metrics.coalesce_request(coalesce_key, make_search_request)
            await self.metrics.end_request(request_id, "search", start_time, True)
            return results

        except Exception as e:
            await self.metrics.end_request(request_id, "search", start_time, False)
            logger.exception(f"Optimized search failed: {e}")
            raise

    async def batch_ai_calls(
        self,
        calls: list[dict[str, Any]],
        batch_delay: float = 0.1,
    ) -> list[Any]:
        """Batch multiple AI calls for better throughput."""
        str(uuid.uuid4())
        logger.debug(f"Batching {len(calls)} AI calls with {batch_delay}s delay")

        # Group calls by model for better efficiency
        calls_by_model: dict[str, Any] = {}
        for i, call in enumerate(calls):
            model = call.get("model", "default")
            if model not in calls_by_model:
                calls_by_model[model] = []
            calls_by_model[model].append((i, call))

        # Execute batches concurrently
        results = [None] * len(calls)
        tasks: list[Any] = []

        for model, model_calls in calls_by_model.items():
            task = asyncio.create_task(self._execute_model_batch(model, model_calls, results))
            tasks.append(task)

        # Wait for all batches to complete
        await asyncio.gather(*tasks)

        return results

    async def optimized_database_query(
        self,
        query: str,
        *args,
        cache_key: str | None = None,
        cache_ttl: float = 300.0,
        **kwargs,
    ) -> Any:
        """Execute optimized database query with caching and connection pooling."""
        if not self.connection_manager.postgresql_pool:
            msg = "PostgreSQL connection pool not configured"
            raise ValueError(msg)

        # Check cache first if cache_key provided
        if cache_key:
            cached_result = await self.cache_manager.get(f"db_query:{cache_key}")
            if cached_result is not None:
                await self.metrics.cache_hit("database")
                return cached_result

        # Execute query through connection pool
        request_id = str(uuid.uuid4())
        start_time = await self.metrics.start_request(request_id, "database_query")

        try:
            result = await self.connection_manager.postgresql_pool.execute_query(
                query, *args, **kwargs
            )

            # Cache result if cache_key provided
            if cache_key:
                await self.cache_manager.set(f"db_query:{cache_key}", result, cache_ttl)
                await self.metrics.cache_miss("database")

            await self.metrics.end_request(request_id, "database_query", start_time, True)
            return result

        except Exception as e:
            await self.metrics.end_request(request_id, "database_query", start_time, False)
            logger.exception(f"Optimized database query failed: {e}")
            raise

    async def optimized_database_transaction(
        self, transaction_func: Callable, *args, **kwargs
    ) -> Any:
        """Execute optimized database transaction with connection pooling."""
        if not self.connection_manager.postgresql_pool:
            msg = "PostgreSQL connection pool not configured"
            raise ValueError(msg)

        request_id = str(uuid.uuid4())
        start_time = await self.metrics.start_request(request_id, "database_transaction")

        try:
            result = await self.connection_manager.postgresql_pool.execute_transaction(
                transaction_func,
                *args,
                **kwargs,
            )
            await self.metrics.end_request(request_id, "database_transaction", start_time, True)
            return result

        except Exception as e:
            await self.metrics.end_request(request_id, "database_transaction", start_time, False)
            logger.exception(f"Optimized database transaction failed: {e}")
            raise

    async def _execute_model_batch(
        self, model: str, model_calls: list[tuple], results: list[Any]
    ) -> None:
        """Execute a batch of calls for a specific model."""
        batch_tasks: list[Any] = []

        for original_index, call in model_calls:
            task = asyncio.create_task(self._execute_single_call(call, original_index, results))
            batch_tasks.append(task)

        await asyncio.gather(*batch_tasks, return_exceptions=True)

    async def _execute_single_call(
        self, call: dict[str, Any], result_index: int, results: list[Any]
    ) -> None:
        """Execute a single call and store result."""
        try:
            result = await self.optimized_ai_call(**call)
            results[result_index] = result
        except Exception as e:
            logger.exception(f"Batch call {result_index} failed: {e}")
            results[result_index] = {"error": str(e)}

    async def parallel_operations(
        self,
        operations: list[Callable],
        max_concurrency: int = 10,
    ) -> list[Any]:
        """Execute operations in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def limited_operation(op):
            async with semaphore:
                return await op()

        tasks = [asyncio.create_task(limited_operation(op)) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        # Get metrics report
        metrics_report = await self.metrics.get_report()

        # Get cache statistics
        cache_stats = await self.cache_manager.get_stats()

        # Get connection pool statistics
        connection_stats = await self.connection_manager.get_comprehensive_stats()

        return {
            "timestamp": time.time(),
            "metrics": metrics_report,
            "cache": cache_stats,
            "connections": connection_stats,
            "redis_enabled": self.enable_redis,
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all components."""
        health = {
            "timestamp": time.time(),
            "status": "healthy",
            "components": {},
        }

        # Check cache manager
        try:
            await self.cache_manager.set("health_check", "ok", 10)
            cache_result = await self.cache_manager.get("health_check")
            health["components"]["cache"] = {
                "status": "healthy" if cache_result == "ok" else "degraded",
                "redis_enabled": self.enable_redis,
            }
        except Exception as e:
            health["components"]["cache"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "degraded"

        # Check connection pools
        try:
            conn_stats = await self.connection_manager.get_comprehensive_stats()
            health["components"]["connections"] = {
                "status": "healthy",
                "http_error_rate": conn_stats["http_pool"]["overall_stats"]["error_rate"],
                "vertex_ai_requests": conn_stats["vertex_ai_pool"]["stats"]["total_requests"],
            }
        except Exception as e:
            health["components"]["connections"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "degraded"

        return health

    async def warm_up_cache(self, warm_up_data: dict[str, Any] | None = None) -> None:
        """Warm up cache with common data."""
        if not warm_up_data:
            warm_up_data = {
                "ai_models": ["gemini-2.5-pro", "gemini-2.5-flash"],
                "common_queries": [
                    "latest technology trends",
                    "artificial intelligence research",
                    "software development best practices",
                ],
            }

        logger.info("Starting cache warm-up...")

        # Pre-cache common model metadata
        for model in warm_up_data.get("ai_models", []):
            cache_key = f"model_metadata:{model}"
            metadata = {
                "model": model,
                "cached_at": time.time(),
                "warm_up": True,
            }
            await self.cache_manager.set(cache_key, metadata, 3600)

        logger.info(f"Cache warm-up completed for {len(warm_up_data.get('ai_models', []))} models")

    async def close(self) -> None:
        """Close all performance optimization components."""
        logger.info("Closing performance optimizer...")

        # Close components
        await self.cache_manager.close()
        await self.connection_manager.close_all()
        await self.metrics.close()

        # Cancel any running batch timers
        for timer in self._batch_timers.values():
            timer.cancel()

        logger.info("Performance optimizer closed")
