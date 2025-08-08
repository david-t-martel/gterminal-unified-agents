"""Performance integration layer for the fullstack agent.

This module integrates the performance optimization components with the existing
agent system, providing optimized versions of key operations.
"""

import asyncio
from functools import wraps
import logging
import os
import time
from typing import Any, Optional
import uuid

from .config import config
from .performance import PerformanceOptimizer
from .utils.rust_extensions import RUST_CORE_AVAILABLE
from .utils.rust_extensions import EnhancedTtlCache

# Import Rust extensions for high-performance caching
from .utils.rust_extensions import RustCore

logger = logging.getLogger(__name__)


class OptimizedAgentComponents:
    """Optimized versions of agent components with caching and connection pooling."""

    def __init__(self, performance_optimizer: PerformanceOptimizer) -> None:
        self.optimizer = performance_optimizer
        self._session_contexts: dict[str, dict[str, Any]] = {}
        self._rust_cache: EnhancedTtlCache | None = None
        self._rust_core: RustCore | None = None
        self._initialize_rust_components()

    def _initialize_rust_components(self) -> None:
        """Initialize Rust components for high-performance operations."""
        if RUST_CORE_AVAILABLE:
            try:
                self._rust_core = RustCore()
                # Initialize with 30 minute TTL for component-level caching
                self._rust_cache = EnhancedTtlCache(1800)
                logger.info("Rust components initialized in OptimizedAgentComponents")
            except Exception as e:
                logger.warning(f"Failed to initialize Rust components: {e}")
                self._rust_core = None
                self._rust_cache = None
        else:
            logger.debug("Rust components not available, using Python fallbacks")

    def _generate_cache_key(self, operation: str, *args, **kwargs) -> str:
        """Generate cache key using Rust string operations for performance."""
        if self._rust_core:
            # Use Rust for efficient string operations
            key_parts = (
                [operation]
                + [str(arg) for arg in args]
                + [f"{k}={v}" for k, v in sorted(kwargs.items())]
            )
            raw_key = ":".join(key_parts)
            # Use Rust for string processing (more efficient than Python for large strings)
            return f"cache:{raw_key}"
        # Fallback to Python
        import hashlib

        key_parts = (
            [operation]
            + [str(arg) for arg in args]
            + [f"{k}={v}" for k, v in sorted(kwargs.items())]
        )
        raw_key = ":".join(key_parts)
        return f"cache:{hashlib.md5(raw_key.encode()).hexdigest()}"

    async def _rust_cache_get(self, key: str) -> dict[str, Any] | None:
        """Get from Rust cache with fallback to standard cache."""
        if self._rust_cache:
            try:
                import json

                cached_data = self._rust_cache.get(key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Rust cache get failed, falling back: {e}")

        # Fallback to standard cache
        return await self.optimizer.cache_manager.get(key)

    async def _rust_cache_set(self, key: str, value: dict[str, Any], ttl: int = 1800) -> None:
        """Set in Rust cache with fallback to standard cache."""
        if self._rust_cache:
            try:
                import json

                self._rust_cache.set_with_ttl(key, json.dumps(value), ttl)
                return
            except Exception as e:
                logger.warning(f"Rust cache set failed, falling back: {e}")

        # Fallback to standard cache
        await self.optimizer.cache_manager.set(key, value, ttl)

    async def optimized_google_search(
        self, query: str, num_results: int = 10, **kwargs
    ) -> dict[str, Any]:
        """Optimized Google search with Rust-powered caching and rate limiting."""
        # Generate cache key using Rust if available
        cache_key = self._generate_cache_key("search", query, num_results, **kwargs)

        # Try Rust cache first for maximum performance
        cached_result = await self._rust_cache_get(cache_key)
        if cached_result:
            logger.debug(f"Search cache hit (Rust): {query[:50]}...")
            return cached_result

        # Execute search with standard optimization
        result = await self.optimizer.optimized_search(
            query=query, num_results=num_results, **kwargs
        )

        # Cache result in Rust cache for future use
        await self._rust_cache_set(cache_key, result, ttl=1800)  # 30 minutes

        return result

    async def optimized_gemini_call(self, model: str, prompt: str, **kwargs) -> Any:
        """Optimized Gemini model call with caching and connection pooling."""
        return await self.optimizer.optimized_ai_call(model=model, prompt=prompt, **kwargs)

    async def batch_gemini_calls(
        self,
        calls: list[dict[str, Any]],
        batch_delay: float = 0.1,
    ) -> list[Any]:
        """Batch multiple Gemini calls for better performance."""
        return await self.optimizer.batch_ai_calls(calls, batch_delay)

    async def get_session_context(
        self, session_id: str, context_key: str = "default"
    ) -> dict[str, Any] | None:
        """Get cached session context."""
        cache_key = f"session:{session_id}:{context_key}"
        return await self.optimizer.cache_manager.get(cache_key)

    async def cache_session_context(
        self,
        session_id: str,
        context: dict[str, Any],
        context_key: str = "default",
        ttl: float = 3600,
    ) -> None:
        """Cache session context."""
        cache_key = f"session:{session_id}:{context_key}"
        await self.optimizer.cache_manager.set(cache_key, context, ttl)

    async def parallel_research_tasks(
        self,
        tasks: list[dict[str, Any]],
        max_concurrency: int = 5,
    ) -> list[Any]:
        """Execute research tasks in parallel."""

        async def execute_task(task):
            task_type = task.get("type", "search")
            if task_type == "search":
                return await self.optimized_google_search(**task.get("params", {}))
            if task_type == "gemini":
                return await self.optimized_gemini_call(**task.get("params", {}))
            msg = f"Unknown task type: {task_type}"
            raise ValueError(msg)

        operations = [lambda t=task: execute_task(t) for task in tasks]
        return await self.optimizer.parallel_operations(operations, max_concurrency)


class PerformanceIntegration:
    """Main integration class for performance optimization."""

    _instance: Optional["PerformanceIntegration"] = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        # Performance configuration
        perf_config = {
            "cache": {
                "memory": {
                    "max_size": int(os.getenv("CACHE_MEMORY_SIZE", "1000")),
                    "default_ttl": int(os.getenv("CACHE_MEMORY_TTL", "1800")),
                },
                "redis": {
                    "key_prefix": os.getenv("CACHE_KEY_PREFIX", "fullstack_agent:"),
                    "default_ttl": int(os.getenv("CACHE_REDIS_TTL", "3600")),
                    "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
                },
            },
            "http": {
                "total_connections": int(os.getenv("HTTP_POOL_SIZE", "100")),
                "connections_per_host": int(os.getenv("HTTP_CONNECTIONS_PER_HOST", "30")),
                "keepalive_timeout": int(os.getenv("HTTP_KEEPALIVE_TIMEOUT", "30")),
                "connect_timeout": int(os.getenv("HTTP_CONNECT_TIMEOUT", "10")),
                "read_timeout": int(os.getenv("HTTP_READ_TIMEOUT", "30")),
            },
            "vertex_ai": {
                "max_clients": int(os.getenv("VERTEX_AI_MAX_CLIENTS", "10")),
                "client_timeout": float(os.getenv("VERTEX_AI_TIMEOUT", "30.0")),
            },
            "google_search": {
                "api_key": os.getenv("GOOGLE_SEARCH_API_KEY"),
                "search_engine_id": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
                "max_requests_per_second": float(os.getenv("SEARCH_RATE_LIMIT", "10.0")),
                "max_requests_per_day": int(os.getenv("SEARCH_DAILY_LIMIT", "10000")),
            },
        }

        # Initialize performance optimizer
        self.optimizer = PerformanceOptimizer(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            enable_redis=os.getenv("ENABLE_REDIS", "true").lower() == "true",
            cache_config=perf_config["cache"],
            connection_config={
                "http": perf_config["http"],
                "vertex_ai": perf_config["vertex_ai"],
                "google_search": perf_config["google_search"],
            },
            alert_callback=self._performance_alert_callback,
        )

        # Initialize optimized components
        self.components = OptimizedAgentComponents(self.optimizer)

        # Performance state
        self._initialized = False

        logger.info("PerformanceIntegration initialized")

    @classmethod
    async def get_instance(cls) -> "PerformanceIntegration":
        """Get singleton instance of PerformanceIntegration."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.initialize()
        return cls._instance

    async def initialize(self) -> None:
        """Initialize the performance integration system."""
        if self._initialized:
            return

        try:
            # Warm up cache with common data
            await self.optimizer.warm_up_cache(
                {
                    "ai_models": [config.worker_model, config.critic_model],
                    "common_queries": [
                        "artificial intelligence research",
                        "technology trends",
                        "software development",
                    ],
                },
            )

            # Perform health check
            health = await self.optimizer.health_check()
            if health["status"] != "healthy":
                logger.warning(f"Performance system health: {health['status']}")

            self._initialized = True
            logger.info("PerformanceIntegration initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize PerformanceIntegration: {e}")
            raise

    async def _performance_alert_callback(
        self,
        budget_name: str,
        current_value: float,
        threshold_value: float,
        budget: Any,
    ) -> None:
        """Handle performance budget alerts."""
        logger.warning(
            f"Performance budget exceeded - {budget_name}: "
            f"{current_value:.2f}{budget.unit} > {threshold_value:.2f}{budget.unit}",
        )

        # Could integrate with external alerting system here
        # For now, just log the alert

    async def get_performance_dashboard(self) -> dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        report = await self.optimizer.get_performance_report()
        health = await self.optimizer.health_check()

        # Calculate some additional derived metrics
        cache_stats = report.get("cache", {})
        overall_cache = cache_stats.get("overall", {})

        return {
            "timestamp": time.time(),
            "health": health,
            "performance_summary": {
                "cache_hit_rate": overall_cache.get("hit_rate", 0) * 100,
                "total_requests": report["metrics"]["raw_metrics"]["counters"].get(
                    "total_requests", 0
                ),
                "error_rate": report["metrics"]["derived_metrics"].get("error_rate", 0),
                "redis_enabled": report.get("redis_enabled", False),
            },
            "budgets": report["metrics"].get("performance_budgets", {}),
            "detailed_stats": {
                "cache": cache_stats,
                "connections": report.get("connections", {}),
                "coalesced_requests": report["metrics"].get("coalesced_requests", 0),
            },
        }

    async def close(self) -> None:
        """Close the performance integration system."""
        if self.optimizer:
            await self.optimizer.close()
        self._initialized = False
        logger.info("PerformanceIntegration closed")


def performance_optimized(func) -> None:
    """Decorator to add performance optimization to functions."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        integration = await PerformanceIntegration.get_instance()
        request_id = str(uuid.uuid4())

        # Start performance tracking
        start_time = await integration.optimizer.metrics.start_request(request_id, func.__name__)

        try:
            result = await func(*args, **kwargs)
            await integration.optimizer.metrics.end_request(
                request_id, func.__name__, start_time, True
            )
            return result
        except Exception:
            await integration.optimizer.metrics.end_request(
                request_id, func.__name__, start_time, False
            )
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> None:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(async_wrapper(*args, **kwargs))

    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# Global performance integration instance
_performance_integration: PerformanceIntegration | None = None


async def get_performance_integration() -> PerformanceIntegration:
    """Get the global performance integration instance."""
    global _performance_integration
    if _performance_integration is None:
        _performance_integration = await PerformanceIntegration.get_instance()
    return _performance_integration


async def optimize_agent_operation(operation_name: str, operation_func, *args, **kwargs):
    """Optimize any agent operation with caching and performance tracking."""
    integration = await get_performance_integration()

    # Generate cache key based on operation and arguments
    cache_key = f"operation:{operation_name}:{hash(str(args) + str(sorted(kwargs.items())))}"

    # Try to get from cache first
    cached_result = await integration.optimizer.cache_manager.get(cache_key)
    if cached_result is not None:
        await integration.optimizer.metrics.cache_hit("operation")
        return cached_result

    # Use request coalescing for identical operations
    coalesce_key = f"op:{operation_name}:{hash(str(args) + str(sorted(kwargs.items())))}"

    async def execute_operation():
        result = await operation_func(*args, **kwargs)
        # Cache the result with appropriate TTL based on operation type
        ttl = 3600 if "search" in operation_name else 1800  # 1 hour for search, 30 min for others
        await integration.optimizer.cache_manager.set(cache_key, result, ttl)
        return result

    await integration.optimizer.metrics.cache_miss("operation")
    return await integration.optimizer.metrics.coalesce_request(coalesce_key, execute_operation)
