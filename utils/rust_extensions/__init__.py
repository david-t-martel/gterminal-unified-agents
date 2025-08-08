"""High-Performance Rust Extensions for Fullstack Agent.

This package provides enhanced Python interfaces to Rust implementations with
automatic fallbacks and comprehensive error handling. Core working components
include RustCore for basic operations and EnhancedTtlCache for high-performance caching.
"""

import logging

logger = logging.getLogger(__name__)

# Core Rust module import
try:
    import fullstack_agent_rust

    RUST_CORE_AVAILABLE = True
    logger.info("Core Rust extensions loaded successfully")

    # Core functionality - WORKING
    RustCore = fullstack_agent_rust.RustCore

    # TTL Caching - WORKING
    EnhancedTtlCache = fullstack_agent_rust.EnhancedTtlCache
    CacheStats = fullstack_agent_rust.CacheStats

    # Verify module information
    _rust_version = getattr(fullstack_agent_rust, "__version__", "unknown")
    _rust_description = getattr(fullstack_agent_rust, "__description__", "Rust extensions")
    logger.info(f"Rust extensions version: {_rust_version}")
    logger.info(f"Description: {_rust_description}")

except ImportError as e:
    RUST_CORE_AVAILABLE = False
    logger.warning(f"Core Rust extensions not available: {e}")

    # Fallback implementations when Rust not available
    class RustCore:
        def __init__(self) -> None:
            self.version = "fallback"

        def test_rust_integration(self) -> str:
            return "Python fallback - Rust not available"

        def reverse_string(self, s):
            return s[::-1]

        def add_numbers(self, a, b):
            return a + b

        def process_dict(self, d):
            return {k: v * 2 for k, v in d.items()}

    class EnhancedTtlCache:
        def __init__(self, ttl_seconds) -> None:
            import time

            self._cache = {}
            self._ttl = ttl_seconds
            self._stats = {"hits": 0, "misses": 0, "total_entries": 0, "expired_entries": 0}

        def set(self, key, value) -> None:
            import time

            self._cache[key] = {"value": value, "expires": time.time() + self._ttl}
            self._stats["total_entries"] += 1

        def get(self, key):
            import time

            if key in self._cache:
                if time.time() < self._cache[key]["expires"]:
                    self._stats["hits"] += 1
                    return self._cache[key]["value"]
                del self._cache[key]
                self._stats["misses"] += 1
                self._stats["expired_entries"] += 1
                return None
            self._stats["misses"] += 1
            return None

        @property
        def size(self):
            return len(self._cache)

    class CacheStats:
        def __init__(self) -> None:
            self.hits = 0
            self.misses = 0
            self.total_entries = 0
            self.expired_entries = 0

        @property
        def hit_ratio(self):
            total = self.hits + self.misses
            return (self.hits / total * 100.0) if total > 0 else 0.0


# Placeholder classes for future Rust components
RustJsonProcessor = None
RustMessagePack = None
RustCache = None
RustCacheManager = None
RustFileOps = None
RustSearchEngine = None
RustStringOps = None
RustAsyncOps = None
TaskResult = None
TaskStatus = None
RustAuthValidator = None
RustTokenManager = None
RustPerformanceMetrics = None
RustResourceMonitor = None
RustCommandExecutor = None
RustBufferPool = None
RustAdvancedSearch = None
RustPathUtils = None

# Legacy wrapper imports - conditionally import if available
try:
    from . import rust_bindings
    from .wrapper import RUST_EXTENSIONS_AVAILABLE
    from .wrapper import get_rust_status

    legacy_wrappers_available = True
    logger.info("Legacy wrapper modules available")
except ImportError as e:
    legacy_wrappers_available = False
    RUST_EXTENSIONS_AVAILABLE = RUST_CORE_AVAILABLE
    logger.info(f"Legacy wrappers not available: {e}")

    def get_rust_status():
        """Fallback rust status function."""
        return {
            "rust_core_available": RUST_CORE_AVAILABLE,
            "version": _rust_version if RUST_CORE_AVAILABLE else "unavailable",
            "working_components": ["RustCore", "EnhancedTtlCache", "CacheStats"]
            if RUST_CORE_AVAILABLE
            else [],
            "legacy_wrappers": False,
        }


# Convenience functions
def test_rust_integration():
    """Test if Rust extensions are working properly."""
    if RUST_CORE_AVAILABLE:
        try:
            core = RustCore()
            result = core.test_rust_integration()
            cache = EnhancedTtlCache(60)
            cache.set("test", "value")
            cache_test = cache.get("test") == "value"
            return {
                "status": "working",
                "core_test": result,
                "cache_test": cache_test,
                "version": core.version,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    else:
        return {"status": "fallback", "message": "Using Python fallbacks"}


__all__ = [
    # Core availability flags
    "RUST_CORE_AVAILABLE",
    "RUST_EXTENSIONS_AVAILABLE",
    "CacheStats",
    "EnhancedTtlCache",
    "RustAdvancedSearch",
    "RustAsyncOps",
    "RustAuthValidator",
    "RustBufferPool",
    "RustCache",
    "RustCacheManager",
    "RustCommandExecutor",
    # Core high-performance Rust components (currently working)
    "RustCore",
    "RustFileOps",
    # Future components (placeholders for development)
    "RustJsonProcessor",
    "RustMessagePack",
    "RustPathUtils",
    "RustPerformanceMetrics",
    "RustResourceMonitor",
    "RustSearchEngine",
    "RustStringOps",
    "RustTokenManager",
    "TaskResult",
    "TaskStatus",
    # Core functions
    "get_rust_status",
    # Legacy compatibility (conditional)
    "legacy_wrappers_available",
    "test_rust_integration",
]
