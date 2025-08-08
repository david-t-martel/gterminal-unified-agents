"""Reorganized utilities package providing comprehensive functionality across multiple domains.

This package provides a clean, organized structure for all utilities:

- database/: Redis, caching, and connection pooling
- rust_extensions/: High-performance Rust PyO3 bindings with Python fallbacks
- monitoring/: Performance metrics, tracing, and system monitoring
- common/: File system operations, base classes, and shared utilities

All utilities provide complete functional coverage of rust-fs MCP server capabilities.
"""

# Import organized subpackages
from . import common
from . import database
from . import monitoring
from . import rust_extensions
from .common import FileSystemUtils
from .common import copy
from .common import create
from .common import delete
from .common import execute
from .common import find
from .common import fs
from .common import move
from .common import read
from .common import replace
from .common import search
from .common import stat
from .common import write

# Import key utilities for convenient access
from .database import CacheManager
from .database import RedisManager
from .database import get_redis_client
from .monitoring import PerformanceMetrics
from .monitoring import PerformanceMonitor
from .rust_extensions import get_system_info
from .rust_extensions import read_file_fast
from .rust_extensions import rust_cache
from .rust_extensions import rust_file_ops
from .rust_extensions import write_file_fast

__all__ = [
    "CacheManager",
    "FileSystemUtils",
    "PerformanceMetrics",
    "PerformanceMonitor",
    # Key utilities
    "RedisManager",
    "common",
    "copy",
    # rust-fs compatible functions
    "create",
    # Subpackages
    "database",
    "delete",
    "execute",
    "find",
    "fs",
    "get_redis_client",
    "get_system_info",
    "monitoring",
    "move",
    "read",
    "read_file_fast",
    "replace",
    "rust_cache",
    "rust_extensions",
    "rust_file_ops",
    "search",
    "stat",
    "write",
    "write_file_fast",
]
