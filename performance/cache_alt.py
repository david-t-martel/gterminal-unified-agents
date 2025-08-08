"""Cache module providing SmartCacheManager and related functionality.

This module re-exports cache functionality to maintain compatibility
with existing imports that expect app.performance.cache.
"""

from gterminal.utils.database.cache import CacheBackend
from gterminal.utils.database.cache import CacheItem
from gterminal.utils.database.cache import CacheManager
from gterminal.utils.database.cache import MemoryCache
from gterminal.utils.database.cache import RedisCache
from gterminal.utils.database.cache import SmartCacheManager
from gterminal.utils.database.cache import cache_key
from gterminal.utils.database.cache import cache_with_invalidation
from gterminal.utils.database.cache import cached
from gterminal.utils.database.cache import conditional_cache

# For backward compatibility, make SmartCacheManager the default export
__all__ = [
    "CacheBackend",
    "CacheItem",
    "CacheManager",
    "MemoryCache",
    "RedisCache",
    "SmartCacheManager",
    "cache_key",
    "cache_with_invalidation",
    "cached",
    "conditional_cache",
]
