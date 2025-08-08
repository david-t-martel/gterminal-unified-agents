"""Performance optimization components for the fullstack agent.

This package provides comprehensive caching, connection pooling, and performance
monitoring capabilities designed to reduce response times from 8+ seconds to under 2 seconds.
"""

from .cache import CacheManager
from .cache import MemoryCache
from .cache import RedisCache
from .connection_pool import ConnectionPoolManager
from .metrics import PerformanceMetrics
from .optimization import PerformanceOptimizer

__all__ = [
    "CacheManager",
    "ConnectionPoolManager",
    "MemoryCache",
    "PerformanceMetrics",
    "PerformanceOptimizer",
    "RedisCache",
]
