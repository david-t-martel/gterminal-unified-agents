#!/usr/bin/env python3
"""
Claude Code Redis Integration Module

This module provides Redis integration for Claude Code framework,
enabling session management, caching, and cross-session state persistence.

Features:
- Session context preservation across conversations
- Tool result caching for performance
- Analysis result storage and retrieval
- Memory management for long-running tasks
- Health monitoring and error tracking
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from typing import Any

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from shared_redis_utils import AgentSessionManager
    from shared_redis_utils import RedisPerformanceMonitor
    from shared_redis_utils import SmartCache
    from shared_redis_utils import health_check
    from shared_redis_utils import quick_cache_get
    from shared_redis_utils import quick_cache_set
    from shared_redis_utils import redis_client

    REDIS_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Redis utils not available: {e}")
    REDIS_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClaudeRedisIntegration:
    """Main Redis integration class for Claude Code."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or self._generate_session_id()
        self.cache = SmartCache("claude_cache", default_ttl=1800)  # 30 minutes
        self.session_manager = AgentSessionManager(f"claude_{self.session_id}")
        self.performance_monitor = RedisPerformanceMonitor()

        # Initialize session
        asyncio.create_task(self._initialize_session())

    def _generate_session_id(self) -> str:
        """Generate unique session ID based on environment."""
        # Use combination of PID, timestamp, and working directory
        unique_string = f"{os.getpid()}_{time.time()}_{os.getcwd()}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]

    async def _initialize_session(self):
        """Initialize Claude session in Redis."""
        try:
            session_data = {
                "session_id": self.session_id,
                "created_at": time.time(),
                "last_active": time.time(),
                "working_directory": os.getcwd(),
                "python_version": sys.version,
                "environment": dict(os.environ),
                "tool_usage_count": 0,
                "cached_operations": 0,
            }

            await self.session_manager.save_session(session_data, ttl=7200)  # 2 hours
            logger.info(f"Claude session initialized: {self.session_id}")

        except Exception as e:
            logger.warning(f"Failed to initialize Redis session: {e}")

    async def update_session_activity(self, activity: str, metadata: dict | None = None):
        """Update session activity in Redis."""
        try:
            session_data = await self.session_manager.load_session()
            session_data.update(
                {
                    "last_active": time.time(),
                    "last_activity": activity,
                    "activity_metadata": metadata or {},
                    "tool_usage_count": session_data.get("tool_usage_count", 0) + 1,
                }
            )

            await self.session_manager.save_session(session_data)

        except Exception as e:
            logger.warning(f"Failed to update session activity: {e}")

    async def cache_tool_result(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        result: Any,
        ttl: int | None = None,
    ):
        """Cache tool execution result."""
        try:
            async with self.performance_monitor.measure_operation(
                "cache_tool_result", self.session_id
            ):
                await self.cache.set(
                    namespace="tool_results",
                    key=tool_name,
                    value={
                        "parameters": parameters,
                        "result": result,
                        "cached_at": time.time(),
                        "session_id": self.session_id,
                    },
                    ttl=ttl,
                    params=parameters,
                )

            # Update session stats
            await self._increment_cache_stats("tool_cache_set")

        except Exception as e:
            logger.warning(f"Failed to cache tool result for {tool_name}: {e}")

    async def get_cached_tool_result(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> Any | None:
        """Get cached tool result if available."""
        try:
            async with self.performance_monitor.measure_operation(
                "get_cached_tool_result", self.session_id
            ):
                cached_data = await self.cache.get(
                    namespace="tool_results", key=tool_name, params=parameters
                )

            if cached_data:
                await self._increment_cache_stats("tool_cache_hit")
                logger.info(f"Cache hit for {tool_name}")
                return cached_data.get("result")
            else:
                await self._increment_cache_stats("tool_cache_miss")
                return None

        except Exception as e:
            logger.warning(f"Failed to get cached tool result for {tool_name}: {e}")
            return None

    async def store_analysis_result(
        self,
        file_path: str,
        analysis_type: str,
        result: dict[str, Any],
        ttl: int = 3600,
    ):
        """Store code analysis results."""
        try:
            # Generate cache key based on file path and modification time
            file_stat = os.stat(file_path) if os.path.exists(file_path) else None
            cache_key = f"{file_path}_{analysis_type}"

            if file_stat:
                cache_key += f"_{int(file_stat.st_mtime)}"

            await quick_cache_set(
                namespace="file_analysis",
                key=cache_key,
                value={
                    "file_path": file_path,
                    "analysis_type": analysis_type,
                    "result": result,
                    "analyzed_at": time.time(),
                    "file_size": file_stat.st_size if file_stat else 0,
                    "file_mtime": file_stat.st_mtime if file_stat else 0,
                },
                ttl=ttl,
                purpose="file_analysis",
            )

            logger.info(f"Stored {analysis_type} analysis for {file_path}")

        except Exception as e:
            logger.warning(f"Failed to store analysis result: {e}")

    async def get_analysis_result(
        self, file_path: str, analysis_type: str
    ) -> dict[str, Any] | None:
        """Get cached analysis result."""
        try:
            # Check if file exists and get modification time
            if not os.path.exists(file_path):
                return None

            file_stat = os.stat(file_path)
            cache_key = f"{file_path}_{analysis_type}_{int(file_stat.st_mtime)}"

            cached_data = await quick_cache_get(
                namespace="file_analysis", key=cache_key, purpose="file_analysis"
            )

            if cached_data:
                # Verify file hasn't changed
                if cached_data.get("file_mtime") == file_stat.st_mtime:
                    logger.info(f"Cache hit for {analysis_type} analysis of {file_path}")
                    return cached_data.get("result")

            return None

        except Exception as e:
            logger.warning(f"Failed to get cached analysis: {e}")
            return None

    async def store_conversation_context(
        self, context: dict[str, Any], context_type: str = "general"
    ):
        """Store conversation context for session continuity."""
        try:
            context_key = f"context_{context_type}_{self.session_id}"

            await quick_cache_set(
                namespace="conversation_context",
                key=context_key,
                value={
                    "context": context,
                    "context_type": context_type,
                    "session_id": self.session_id,
                    "stored_at": time.time(),
                    "working_directory": os.getcwd(),
                },
                ttl=7200,  # 2 hours
                purpose="claude_sessions",
            )

        except Exception as e:
            logger.warning(f"Failed to store conversation context: {e}")

    async def get_conversation_context(
        self, context_type: str = "general"
    ) -> dict[str, Any] | None:
        """Get stored conversation context."""
        try:
            context_key = f"context_{context_type}_{self.session_id}"

            cached_data = await quick_cache_get(
                namespace="conversation_context",
                key=context_key,
                purpose="claude_sessions",
            )

            if cached_data:
                return cached_data.get("context")

            return None

        except Exception as e:
            logger.warning(f"Failed to get conversation context: {e}")
            return None

    async def track_error(self, error: Exception, context: dict | None = None):
        """Track errors for monitoring and debugging."""
        try:
            error_data = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "session_id": self.session_id,
                "timestamp": time.time(),
                "working_directory": os.getcwd(),
                "context": context or {},
            }

            await quick_cache_set(
                namespace="errors",
                key=f"error_{int(time.time())}_{self.session_id}",
                value=error_data,
                ttl=86400,  # 1 day
                purpose="error_logs",
            )

        except Exception as e:
            logger.warning(f"Failed to track error: {e}")

    async def get_session_statistics(self) -> dict[str, Any]:
        """Get session statistics and performance metrics."""
        try:
            session_data = await self.session_manager.load_session()

            # Get performance stats
            performance_stats = await self.performance_monitor.get_operation_stats(
                operation="*", agent_id=self.session_id
            )

            return {
                "session_info": {
                    "session_id": self.session_id,
                    "created_at": session_data.get("created_at"),
                    "last_active": session_data.get("last_active"),
                    "tool_usage_count": session_data.get("tool_usage_count", 0),
                    "cached_operations": session_data.get("cached_operations", 0),
                },
                "performance": performance_stats,
                "cache_stats": {
                    "tool_cache_hits": await self._get_cache_stat("tool_cache_hit"),
                    "tool_cache_misses": await self._get_cache_stat("tool_cache_miss"),
                    "tool_cache_sets": await self._get_cache_stat("tool_cache_set"),
                },
            }

        except Exception as e:
            logger.warning(f"Failed to get session statistics: {e}")
            return {}

    async def cleanup_session(self):
        """Cleanup session data."""
        try:
            await self.session_manager.clear_session()
            logger.info(f"Cleaned up session: {self.session_id}")

        except Exception as e:
            logger.warning(f"Failed to cleanup session: {e}")

    async def _increment_cache_stats(self, stat_name: str):
        """Increment cache statistics."""
        try:
            async with redis_client("performance_metrics") as redis:
                await redis.incr(f"claude_cache_stats:{self.session_id}:{stat_name}")

        except Exception as e:
            logger.debug(f"Failed to increment cache stat {stat_name}: {e}")

    async def _get_cache_stat(self, stat_name: str) -> int:
        """Get cache statistic value."""
        try:
            async with redis_client("performance_metrics") as redis:
                value = await redis.get(f"claude_cache_stats:{self.session_id}:{stat_name}")
                return int(value) if value else 0

        except Exception as e:
            logger.debug(f"Failed to get cache stat {stat_name}: {e}")
            return 0


# Global instance for easy access
_claude_redis_instance: ClaudeRedisIntegration | None = None


def get_claude_redis(session_id: str | None = None) -> ClaudeRedisIntegration:
    """Get global Claude Redis integration instance."""
    global _claude_redis_instance

    if not REDIS_UTILS_AVAILABLE:
        # Return a mock object if Redis is not available
        class MockClaudeRedis:
            async def cache_tool_result(self, *args, **kwargs):
                pass

            async def get_cached_tool_result(self, *args, **kwargs):
                return None

            async def store_analysis_result(self, *args, **kwargs):
                pass

            async def get_analysis_result(self, *args, **kwargs):
                return None

            async def store_conversation_context(self, *args, **kwargs):
                pass

            async def get_conversation_context(self, *args, **kwargs):
                return None

            async def track_error(self, *args, **kwargs):
                pass

            async def get_session_statistics(self, *args, **kwargs):
                return {}

            async def cleanup_session(self, *args, **kwargs):
                pass

            async def update_session_activity(self, *args, **kwargs):
                pass

        return MockClaudeRedis()

    if _claude_redis_instance is None:
        _claude_redis_instance = ClaudeRedisIntegration(session_id)

    return _claude_redis_instance


# Convenience functions for common operations
async def cache_file_analysis(file_path: str, analysis_type: str, result: dict[str, Any]):
    """Cache file analysis result."""
    claude_redis = get_claude_redis()
    await claude_redis.store_analysis_result(file_path, analysis_type, result)


async def get_cached_file_analysis(file_path: str, analysis_type: str) -> dict[str, Any] | None:
    """Get cached file analysis result."""
    claude_redis = get_claude_redis()
    return await claude_redis.get_analysis_result(file_path, analysis_type)


async def cache_tool_execution(tool_name: str, parameters: dict, result: Any, ttl: int = 300):
    """Cache tool execution result."""
    claude_redis = get_claude_redis()
    await claude_redis.cache_tool_result(tool_name, parameters, result, ttl)


async def get_cached_tool_execution(tool_name: str, parameters: dict) -> Any | None:
    """Get cached tool execution result."""
    claude_redis = get_claude_redis()
    return await claude_redis.get_cached_tool_result(tool_name, parameters)


async def update_activity(activity: str, metadata: dict | None = None):
    """Update Claude activity in session."""
    claude_redis = get_claude_redis()
    await claude_redis.update_session_activity(activity, metadata)


async def store_context(context: dict, context_type: str = "general"):
    """Store conversation context."""
    claude_redis = get_claude_redis()
    await claude_redis.store_conversation_context(context, context_type)


async def get_context(context_type: str = "general") -> dict | None:
    """Get conversation context."""
    claude_redis = get_claude_redis()
    return await claude_redis.get_conversation_context(context_type)


async def track_claude_error(error: Exception, context: dict | None = None):
    """Track error for monitoring."""
    claude_redis = get_claude_redis()
    await claude_redis.track_error(error, context)


async def get_claude_stats() -> dict[str, Any]:
    """Get Claude session statistics."""
    claude_redis = get_claude_redis()
    return await claude_redis.get_session_statistics()


async def cleanup_claude_session():
    """Cleanup Claude session."""
    claude_redis = get_claude_redis()
    await claude_redis.cleanup_session()


# Health check function
async def redis_health_check() -> dict[str, Any]:
    """Perform Redis health check for Claude integration."""
    try:
        if not REDIS_UTILS_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "Redis utils not available",
                "timestamp": time.time(),
            }

        health_result = await health_check()

        # Add Claude-specific checks
        claude_redis = get_claude_redis()
        stats = await claude_redis.get_session_statistics()

        health_result["claude_integration"] = {
            "session_active": bool(stats.get("session_info")),
            "tool_usage_count": stats.get("session_info", {}).get("tool_usage_count", 0),
            "cache_hit_rate": _calculate_cache_hit_rate(stats.get("cache_stats", {})),
        }

        return health_result

    except Exception as e:
        return {"status": "error", "message": str(e), "timestamp": time.time()}


def _calculate_cache_hit_rate(cache_stats: dict[str, int]) -> float:
    """Calculate cache hit rate from statistics."""
    hits = cache_stats.get("tool_cache_hits", 0)
    misses = cache_stats.get("tool_cache_misses", 0)
    total = hits + misses

    if total == 0:
        return 0.0

    return round((hits / total) * 100, 2)


# CLI interface for testing and management
async def main():
    """CLI interface for Redis integration testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Claude Redis Integration CLI")
    parser.add_argument(
        "action",
        choices=["health", "stats", "cleanup", "test"],
        help="Action to perform",
    )
    parser.add_argument("--session-id", help="Session ID to use")

    args = parser.parse_args()

    if args.action == "health":
        result = await redis_health_check()
        print(json.dumps(result, indent=2, default=str))

    elif args.action == "stats":
        stats = await get_claude_stats()
        print(json.dumps(stats, indent=2, default=str))

    elif args.action == "cleanup":
        await cleanup_claude_session()
        print("Session cleaned up")

    elif args.action == "test":
        # Test basic functionality
        print("Testing Claude Redis integration...")

        # Test caching
        await cache_tool_execution("test_tool", {"param": "value"}, {"result": "success"})
        cached_result = await get_cached_tool_execution("test_tool", {"param": "value"})
        print(f"Cache test: {'PASS' if cached_result else 'FAIL'}")

        # Test context storage
        await store_context({"test": "context"}, "test")
        stored_context = await get_context("test")
        print(f"Context test: {'PASS' if stored_context else 'FAIL'}")

        # Test activity tracking
        await update_activity("test_activity", {"test": True})
        print("Activity tracking: PASS")

        print("All tests completed")


if __name__ == "__main__":
    asyncio.run(main())
