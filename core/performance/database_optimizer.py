"""Database query optimization and connection pooling for my-fullstack-agent.

Features:
- Slow query detection and optimization
- Connection pool optimization
- Query caching and result memoization
- Database performance monitoring
- Index recommendation system
- Query plan analysis
"""

from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from functools import wraps
import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

import asyncpg
from asyncpg import Connection
from asyncpg import Pool
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a database query."""

    query_hash: str
    query_text: str
    execution_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    last_executed: float = 0.0
    error_count: int = 0
    rows_affected: int = 0

    def update(self, execution_time_ms: float, rows: int = 0, error: bool = False) -> None:
        """Update metrics with new execution data."""
        self.execution_count += 1
        self.total_time_ms += execution_time_ms
        self.min_time_ms = min(self.min_time_ms, execution_time_ms)
        self.max_time_ms = max(self.max_time_ms, execution_time_ms)
        self.avg_time_ms = self.total_time_ms / self.execution_count
        self.last_executed = time.time()
        self.rows_affected += rows

        if error:
            self.error_count += 1


@dataclass
class QueryPlan:
    """Database query execution plan."""

    query_hash: str
    plan_json: dict[str, Any]
    estimated_cost: float
    estimated_rows: int
    scan_types: list[str] = field(default_factory=list)
    indexes_used: list[str] = field(default_factory=list)

    @property
    def has_sequential_scan(self) -> bool:
        """Check if plan includes sequential scans."""
        return any("Seq Scan" in scan for scan in self.scan_types)

    @property
    def optimization_score(self) -> float:
        """Calculate optimization score (0-100, higher is better)."""
        score = 100.0

        # Penalty for sequential scans
        if self.has_sequential_scan:
            score -= 30

        # Penalty for high cost
        if self.estimated_cost > 1000:
            score -= min(40, (self.estimated_cost / 1000) * 10)

        # Bonus for index usage
        if self.indexes_used:
            score += min(20, len(self.indexes_used) * 5)

        return max(0, score)


class QueryCache:
    """Intelligent query result caching."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: dict[str, tuple[Any, float, int]] = {}  # key: (result, expiry, access_count)
        self.access_times: dict[str, float] = {}
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _generate_cache_key(self, query: str, params: tuple = ()) -> str:
        """Generate cache key for query and parameters."""
        import hashlib

        content = f"{query}:{params}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, query: str, params: tuple = ()) -> Any | None:
        """Get cached query result."""
        key = self._generate_cache_key(query, params)

        if key in self.cache:
            result, expiry, access_count = self.cache[key]

            if time.time() < expiry:
                # Update access stats
                self.cache[key] = (result, expiry, access_count + 1)
                self.access_times[key] = time.time()
                self.stats["hits"] += 1
                return result
            # Expired, remove from cache
            del self.cache[key]
            self.access_times.pop(key, None)

        self.stats["misses"] += 1
        return None

    def set(self, query: str, params: tuple, result: Any, ttl: int | None = None) -> None:
        """Cache query result."""
        key = self._generate_cache_key(query, params)
        expiry = time.time() + (ttl or self.default_ttl)

        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = (result, expiry, 1)
        self.access_times[key] = time.time()

    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self.access_times:
            return

        # Remove oldest 10% of entries
        items_to_remove = max(1, len(self.access_times) // 10)
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])

        for key, _ in sorted_items[:items_to_remove]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
            self.stats["evictions"] += 1

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        count = 0
        keys_to_remove: list[Any] = []

        for key in self.cache:
            if pattern in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
            count += 1

        return count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests) * 100 if total_requests > 0 else 0

        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size,
        }


class ConnectionPoolOptimizer:
    """Optimizes database connection pool settings."""

    def __init__(self, pool: Pool) -> None:
        self.pool = pool
        self.connection_metrics = defaultdict(list)
        self.pool_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "wait_times": deque(maxlen=1000),
            "connection_errors": 0,
        }

    async def get_optimized_connection(self) -> Connection:
        """Get connection with performance tracking."""
        wait_start = time.time()

        try:
            connection = await self.pool.acquire()
            wait_time = (time.time() - wait_start) * 1000

            self.pool_stats["wait_times"].append(wait_time)
            self.pool_stats["active_connections"] += 1

            return connection

        except Exception as e:
            self.pool_stats["connection_errors"] += 1
            logger.exception(f"Failed to acquire connection: {e}")
            raise

    async def release_connection(self, connection: Connection) -> None:
        """Release connection with tracking."""
        try:
            await self.pool.release(connection)
            self.pool_stats["active_connections"] = max(
                0, self.pool_stats["active_connections"] - 1
            )
        except Exception as e:
            logger.exception(f"Error releasing connection: {e}")

    def get_pool_recommendations(self) -> list[dict[str, Any]]:
        """Get connection pool optimization recommendations."""
        recommendations: list[Any] = []

        # Check wait times
        if self.pool_stats["wait_times"]:
            avg_wait_time = sum(self.pool_stats["wait_times"]) / len(self.pool_stats["wait_times"])

            if avg_wait_time > 100:  # 100ms threshold
                recommendations.append(
                    {
                        "type": "pool_size",
                        "priority": "high",
                        "current_value": self.pool.get_size(),
                        "recommended_value": min(
                            self.pool.get_max_size(), self.pool.get_size() + 5
                        ),
                        "reason": f"Average connection wait time is {avg_wait_time:.1f}ms",
                        "action": "Increase pool size to reduce connection wait times",
                    },
                )

        # Check error rate
        total_requests = sum(len(metrics) for metrics in self.connection_metrics.values())
        if total_requests > 0:
            error_rate = (self.pool_stats["connection_errors"] / total_requests) * 100

            if error_rate > 5:  # 5% error threshold
                recommendations.append(
                    {
                        "type": "pool_health",
                        "priority": "critical",
                        "current_value": f"{error_rate:.1f}%",
                        "reason": "High connection error rate detected",
                        "action": "Review database connectivity and pool configuration",
                    },
                )

        return recommendations


class DatabaseOptimizer:
    """Main database optimization manager."""

    def __init__(self, database_url: str, redis_url: str = "redis://localhost:6379") -> None:
        self.database_url = database_url
        self.redis_url = redis_url

        # Components
        self.pool: Pool | None = None
        self.pool_optimizer: ConnectionPoolOptimizer | None = None
        self.query_cache = QueryCache()
        self.redis_client: redis.Redis | None = None

        # Query tracking
        self.query_metrics: dict[str, QueryMetrics] = {}
        self.slow_query_threshold_ms = 1000  # 1 second
        self.query_plans: dict[str, QueryPlan] = {}

        # Monitoring
        self.monitoring_enabled = True
        self.stats = {
            "total_queries": 0,
            "slow_queries": 0,
            "cached_queries": 0,
            "optimization_recommendations": 0,
        }

    async def initialize(self) -> None:
        """Initialize database connections and caching."""
        # Initialize PostgreSQL connection pool
        parsed_url = urlparse(self.database_url)

        self.pool = await asyncpg.create_pool(
            host=parsed_url.hostname,
            port=parsed_url.port or 5432,
            database=parsed_url.path.lstrip("/"),
            user=parsed_url.username,
            password=parsed_url.password,
            min_size=5,
            max_size=20,
            command_timeout=60,
            server_settings={
                "application_name": "my-fullstack-agent-optimizer",
                "search_path": "public",
            },
        )

        self.pool_optimizer = ConnectionPoolOptimizer(self.pool)

        # Initialize Redis for distributed caching
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"Redis not available, using local cache only: {e}")
            self.redis_client = None

    def _generate_query_hash(self, query: str) -> str:
        """Generate hash for query normalization."""
        import hashlib

        # Normalize query for consistent hashing
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        normalized = re.sub(r"\$\d+", "?", normalized)  # Replace parameters
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    async def execute_query(
        self,
        query: str,
        *args,
        cache_ttl: int | None = None,
        force_cache: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute query with optimization and caching."""
        start_time = time.time()
        query_hash = self._generate_query_hash(query)

        self.stats["total_queries"] += 1

        # Try cache first
        if cache_ttl is not None or force_cache:
            cached_result = self.query_cache.get(query, args)
            if cached_result is not None:
                self.stats["cached_queries"] += 1
                return cached_result

        # Initialize metrics if new query
        if query_hash not in self.query_metrics:
            self.query_metrics[query_hash] = QueryMetrics(query_hash=query_hash, query_text=query)

        connection = await self.pool_optimizer.get_optimized_connection()

        try:
            # Execute query
            if args:
                result = await connection.fetch(query, *args)
            else:
                result = await connection.fetch(query)

            # Convert to list of dicts
            result_list = [dict(row) for row in result]

            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self.query_metrics[query_hash].update(execution_time, len(result_list), False)

            # Check if slow query
            if execution_time > self.slow_query_threshold_ms:
                self.stats["slow_queries"] += 1
                await self._handle_slow_query(query_hash, query, execution_time)

            # Cache result if requested
            if cache_ttl is not None:
                self.query_cache.set(query, args, result_list, cache_ttl)

            return result_list

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.query_metrics[query_hash].update(execution_time, 0, True)
            logger.exception(f"Query execution failed: {e}")
            raise

        finally:
            await self.pool_optimizer.release_connection(connection)

    async def execute_command(self, command: str, *args) -> str:
        """Execute command (INSERT, UPDATE, DELETE) with optimization."""
        start_time = time.time()
        query_hash = self._generate_query_hash(command)

        self.stats["total_queries"] += 1

        # Initialize metrics if new command
        if query_hash not in self.query_metrics:
            self.query_metrics[query_hash] = QueryMetrics(query_hash=query_hash, query_text=command)

        connection = await self.pool_optimizer.get_optimized_connection()

        try:
            if args:
                result = await connection.execute(command, *args)
            else:
                result = await connection.execute(command)

            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            rows_affected = int(result.split()[-1]) if result.split() else 0
            self.query_metrics[query_hash].update(execution_time, rows_affected, False)

            # Check if slow query
            if execution_time > self.slow_query_threshold_ms:
                self.stats["slow_queries"] += 1
                await self._handle_slow_query(query_hash, command, execution_time)

            # Invalidate related cache entries for data-modifying commands
            if any(
                cmd in command.upper()
                for cmd in ["INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]
            ):
                await self._invalidate_related_cache(command)

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.query_metrics[query_hash].update(execution_time, 0, True)
            logger.exception(f"Command execution failed: {e}")
            raise

        finally:
            await self.pool_optimizer.release_connection(connection)

    async def _handle_slow_query(self, query_hash: str, query: str, execution_time: float) -> None:
        """Handle slow query detection and analysis."""
        logger.warning(f"Slow query detected ({execution_time:.1f}ms): {query[:100]}...")

        # Get query plan for analysis
        if query_hash not in self.query_plans:
            try:
                plan = await self._get_query_plan(query)
                if plan:
                    self.query_plans[query_hash] = plan
            except Exception as e:
                logger.exception(f"Failed to get query plan: {e}")

    async def _get_query_plan(self, query: str) -> QueryPlan | None:
        """Get query execution plan for analysis."""
        if not query.strip().upper().startswith("SELECT"):
            return None

        connection = await self.pool_optimizer.get_optimized_connection()

        try:
            # Get execution plan
            explain_query = f"EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {query}"
            result = await connection.fetch(explain_query)

            if result:
                plan_data = result[0]["QUERY PLAN"][0]

                # Extract plan information
                scan_types = self._extract_scan_types(plan_data)
                indexes_used = self._extract_indexes_used(plan_data)

                return QueryPlan(
                    query_hash=self._generate_query_hash(query),
                    plan_json=plan_data,
                    estimated_cost=plan_data.get("Total Cost", 0),
                    estimated_rows=plan_data.get("Plan Rows", 0),
                    scan_types=scan_types,
                    indexes_used=indexes_used,
                )

        except Exception as e:
            logger.exception(f"Failed to analyze query plan: {e}")

        finally:
            await self.pool_optimizer.release_connection(connection)

        return None

    def _extract_scan_types(self, plan_node: dict[str, Any]) -> list[str]:
        """Extract scan types from query plan."""
        scan_types: list[Any] = []

        def traverse_plan(node) -> None:
            if "Node Type" in node:
                scan_types.append(node["Node Type"])

            for child in node.get("Plans", []):
                traverse_plan(child)

        traverse_plan(plan_node)
        return scan_types

    def _extract_indexes_used(self, plan_node: dict[str, Any]) -> list[str]:
        """Extract indexes used from query plan."""
        indexes: list[Any] = []

        def traverse_plan(node) -> None:
            if "Index Name" in node:
                indexes.append(node["Index Name"])

            for child in node.get("Plans", []):
                traverse_plan(child)

        traverse_plan(plan_node)
        return indexes

    async def _invalidate_related_cache(self, command: str) -> None:
        """Invalidate cache entries related to modified tables."""
        # Extract table names from command
        table_pattern = r"(?:FROM|INTO|UPDATE|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        tables = re.findall(table_pattern, command.upper())

        for table in tables:
            self.query_cache.invalidate_pattern(table.lower())

    async def get_optimization_recommendations(self) -> list[dict[str, Any]]:
        """Get database optimization recommendations."""
        recommendations: list[Any] = []

        # Slow query recommendations
        slow_queries = [
            (query_hash, metrics)
            for query_hash, metrics in self.query_metrics.items()
            if metrics.avg_time_ms > self.slow_query_threshold_ms
        ]

        for query_hash, metrics in slow_queries:
            rec = {
                "type": "slow_query",
                "priority": "high" if metrics.avg_time_ms > 2000 else "medium",
                "query_hash": query_hash,
                "avg_time_ms": metrics.avg_time_ms,
                "execution_count": metrics.execution_count,
                "recommendation": "Optimize this frequently executed slow query",
            }

            # Add specific recommendations based on query plan
            if query_hash in self.query_plans:
                plan = self.query_plans[query_hash]
                if plan.has_sequential_scan:
                    rec["specific_actions"] = [
                        "Add appropriate indexes to avoid sequential scans",
                        "Consider query rewriting to use existing indexes",
                    ]
                elif plan.estimated_cost > 10000:
                    rec["specific_actions"] = [
                        "Review query structure for optimization opportunities",
                        "Consider breaking complex query into smaller parts",
                    ]

            recommendations.append(rec)

        # Connection pool recommendations
        pool_recs = self.pool_optimizer.get_pool_recommendations()
        recommendations.extend(pool_recs)

        # Cache recommendations
        cache_stats = self.query_cache.get_stats()
        if cache_stats["hit_rate"] < 50:  # Less than 50% hit rate
            recommendations.append(
                {
                    "type": "caching",
                    "priority": "medium",
                    "current_hit_rate": cache_stats["hit_rate"],
                    "recommendation": "Low cache hit rate - consider increasing cache TTL or identifying cacheable queries",
                    "actions": [
                        "Review query patterns for caching opportunities",
                        "Increase cache size if memory allows",
                        "Implement cache warming for predictable queries",
                    ],
                },
            )

        self.stats["optimization_recommendations"] = len(recommendations)
        return recommendations

    async def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive database performance report."""
        # Calculate summary statistics
        total_queries = sum(m.execution_count for m in self.query_metrics.values())
        total_time = sum(m.total_time_ms for m in self.query_metrics.values())
        avg_query_time = total_time / total_queries if total_queries > 0 else 0

        # Find top slow queries
        slow_queries = sorted(
            [(hash, m) for hash, m in self.query_metrics.items()],
            key=lambda x: x[1].avg_time_ms,
            reverse=True,
        )[:10]

        # Pool statistics
        pool_stats = {
            "size": self.pool.get_size() if self.pool else 0,
            "min_size": self.pool.get_min_size() if self.pool else 0,
            "max_size": self.pool.get_max_size() if self.pool else 0,
            "idle_size": self.pool.get_idle_size() if self.pool else 0,
        }

        if self.pool_optimizer and self.pool_optimizer.pool_stats["wait_times"]:
            pool_stats["avg_wait_time_ms"] = sum(
                self.pool_optimizer.pool_stats["wait_times"]
            ) / len(
                self.pool_optimizer.pool_stats["wait_times"],
            )

        return {
            "timestamp": time.time(),
            "summary": {
                "total_queries_executed": total_queries,
                "total_execution_time_ms": total_time,
                "average_query_time_ms": avg_query_time,
                "slow_queries_count": len(
                    [
                        m
                        for m in self.query_metrics.values()
                        if m.avg_time_ms > self.slow_query_threshold_ms
                    ],
                ),
                "error_rate": (
                    sum(m.error_count for m in self.query_metrics.values()) / total_queries * 100
                    if total_queries > 0
                    else 0
                ),
            },
            "cache_performance": self.query_cache.get_stats(),
            "connection_pool": pool_stats,
            "slow_queries": [
                {
                    "query_hash": query_hash,
                    "query_text": (
                        metrics.query_text[:200] + "..."
                        if len(metrics.query_text) > 200
                        else metrics.query_text
                    ),
                    "avg_time_ms": metrics.avg_time_ms,
                    "execution_count": metrics.execution_count,
                    "error_count": metrics.error_count,
                }
                for query_hash, metrics in slow_queries
            ],
            "optimization_score": self._calculate_optimization_score(),
            "recommendations": await self.get_optimization_recommendations(),
        }

    def _calculate_optimization_score(self) -> float:
        """Calculate overall database optimization score (0-100)."""
        score = 100.0

        # Penalize slow queries
        total_queries = sum(m.execution_count for m in self.query_metrics.values())
        slow_query_ratio = self.stats["slow_queries"] / total_queries if total_queries > 0 else 0
        score -= min(40, slow_query_ratio * 100)

        # Penalize low cache hit rate
        cache_stats = self.query_cache.get_stats()
        if cache_stats["hit_rate"] < 70:
            score -= min(20, (70 - cache_stats["hit_rate"]) / 2)

        # Penalize connection issues
        if self.pool_optimizer and self.pool_optimizer.pool_stats["connection_errors"] > 0:
            error_rate = (
                self.pool_optimizer.pool_stats["connection_errors"] / total_queries
                if total_queries > 0
                else 0
            )
            score -= min(30, error_rate * 1000)  # High penalty for connection errors

        return max(0, score)

    async def close(self) -> None:
        """Close database connections."""
        if self.pool:
            await self.pool.close()

        if self.redis_client:
            await self.redis_client.close()


# Decorator for automatic query optimization
def optimize_database_query(cache_ttl: int | None = None, force_cache: bool = False) -> None:
    """Decorator to automatically optimize database queries."""

    def decorator(func) -> None:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract database optimizer from context or create one
            optimizer = kwargs.pop("db_optimizer", None)

            if optimizer and hasattr(func, "_query_template"):
                # Use optimizer if available and function has query template
                query = func._query_template
                return await optimizer.execute_query(
                    query, *args, cache_ttl=cache_ttl, force_cache=force_cache
                )
            # Fall back to original function
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Example usage functions
@optimize_database_query(cache_ttl=300)
async def get_user_by_id(user_id: int, db_optimizer: DatabaseOptimizer) -> dict[str, Any] | None:
    """Example optimized user lookup query."""
    results = await db_optimizer.execute_query(
        "SELECT * FROM users WHERE id = $1", user_id, cache_ttl=300
    )
    return results[0] if results else None


get_user_by_id._query_template = "SELECT * FROM users WHERE id = $1"


@optimize_database_query(cache_ttl=600)
async def get_recent_posts(
    limit: int = 10, db_optimizer: DatabaseOptimizer = None
) -> list[dict[str, Any]]:
    """Example optimized recent posts query."""
    return await db_optimizer.execute_query(
        "SELECT * FROM posts ORDER BY created_at DESC LIMIT $1",
        limit,
        cache_ttl=600,
    )


get_recent_posts._query_template = "SELECT * FROM posts ORDER BY created_at DESC LIMIT $1"
