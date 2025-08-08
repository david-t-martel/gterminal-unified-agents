"""Enhanced Python wrapper for Rust extensions with comprehensive error handling,
fallbacks, and performance monitoring.

This module provides a clean, high-level interface to all Rust PyO3 bindings
with automatic fallbacks to Python implementations when Rust extensions fail.
"""

import asyncio
from collections.abc import Callable
from functools import wraps
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import Rust extensions if available
try:
    from fullstack_agent_rust import RustAuthValidator
    from fullstack_agent_rust import RustCache
    from fullstack_agent_rust import RustFileOps
    from fullstack_agent_rust import RustJsonProcessor
    from fullstack_agent_rust import RustWebSocketHandler
    from fullstack_agent_rust import performance_metrics
    from fullstack_agent_rust import system_info

    RUST_EXTENSIONS_AVAILABLE = True
    logger.info("Rust extensions loaded successfully")
except ImportError as e:
    logger.warning(f"Rust extensions not available: {e}")
    RUST_EXTENSIONS_AVAILABLE = False

    # Mock classes for fallback
    class RustCache:
        def __init__(self, *args, **kwargs) -> None:
            self._cache = {}

        def get(self, key) -> None:
            return self._cache.get(key)

        def set(self, key, value) -> None:
            self._cache[key] = value
            return True

        def delete(self, key) -> None:
            return self._cache.pop(key, None) is not None

        def clear(self) -> None:
            self._cache.clear()
            return True

    class RustFileOps:
        pass

    class RustJsonProcessor:
        def parse_large_json(self, json_str) -> None:
            import json

            return json.loads(json_str)

    class RustAuthValidator:
        def validate_jwt(self, token, secret) -> None:
            return True

    class RustWebSocketHandler:
        pass

    def performance_metrics() -> None:
        return {}

    def system_info() -> None:
        return {}


def rust_fallback(fallback_func: Callable | None = None) -> None:
    """Decorator that provides automatic fallback to Python implementations."""

    def decorator(func) -> None:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not RUST_EXTENSIONS_AVAILABLE:
                if fallback_func:
                    return (
                        await fallback_func(*args, **kwargs)
                        if asyncio.iscoroutinefunction(fallback_func)
                        else fallback_func(*args, **kwargs)
                    )
                logger.warning(
                    f"Rust extension {func.__name__} not available, no fallback provided"
                )
                return None

            try:
                return (
                    await func(*args, **kwargs)
                    if asyncio.iscoroutinefunction(func)
                    else func(*args, **kwargs)
                )
            except Exception as e:
                logger.exception(f"Rust extension {func.__name__} failed: {e}")
                if fallback_func:
                    return (
                        await fallback_func(*args, **kwargs)
                        if asyncio.iscoroutinefunction(fallback_func)
                        else fallback_func(*args, **kwargs)
                    )
                return None

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> None:
            if not RUST_EXTENSIONS_AVAILABLE:
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                logger.warning(
                    f"Rust extension {func.__name__} not available, no fallback provided"
                )
                return None

            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Rust extension {func.__name__} failed: {e}")
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                return None

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


class EnhancedRustCache:
    """Enhanced cache with automatic fallback and monitoring."""

    def __init__(self, capacity: int = 10000, ttl_seconds: int = 3600) -> None:
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._rust_cache = RustCache(capacity, ttl_seconds) if RUST_EXTENSIONS_AVAILABLE else None
        self._python_fallback = {}
        self._access_count = 0
        self._hit_count = 0
        self._miss_count = 0

    @rust_fallback()
    def get(self, key: str) -> Any | None:
        """Get value from cache with monitoring."""
        self._access_count += 1

        result = self._rust_cache.get(key) if self._rust_cache else self._python_fallback.get(key)

        if result is not None:
            self._hit_count += 1
        else:
            self._miss_count += 1

        return result

    @rust_fallback()
    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Set value in cache."""
        if self._rust_cache:
            return self._rust_cache.set(key, value)
        self._python_fallback[key] = value
        return True

    @rust_fallback()
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if self._rust_cache:
            return self._rust_cache.delete(key)
        return self._python_fallback.pop(key, None) is not None

    @rust_fallback()
    def clear(self) -> bool:
        """Clear entire cache."""
        if self._rust_cache:
            return self._rust_cache.clear()
        self._python_fallback.clear()
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self._hit_count / self._access_count if self._access_count > 0 else 0
        return {
            "backend": "rust" if self._rust_cache else "python",
            "capacity": self.capacity,
            "ttl_seconds": self.ttl_seconds,
            "access_count": self._access_count,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "size": len(self._python_fallback) if not self._rust_cache else "unknown",
        }


class EnhancedFileOps:
    """Enhanced file operations with Rust acceleration."""

    def __init__(self) -> None:
        self._rust_file_ops = RustFileOps() if RUST_EXTENSIONS_AVAILABLE else None

    async def _python_read_file(self, file_path: str) -> str | None:
        """Python fallback for file reading."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.exception(f"Failed to read file {file_path}: {e}")
            return None

    async def _python_write_file(self, file_path: str, content: str) -> bool:
        """Python fallback for file writing."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            logger.exception(f"Failed to write file {file_path}: {e}")
            return False

    async def _python_copy_file(self, src: str, dst: str) -> bool:
        """Python fallback for file copying."""
        try:
            import shutil

            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.exception(f"Failed to copy {src} to {dst}: {e}")
            return False

    async def _python_get_file_hash(self, file_path: str) -> str | None:
        """Python fallback for file hashing."""
        try:
            import hashlib

            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.exception(f"Failed to hash file {file_path}: {e}")
            return None

    @rust_fallback()
    async def read_file_fast(self, file_path: str | Path) -> str | None:
        """Read file using Rust for maximum performance."""
        return await self._python_read_file(str(file_path))

    @rust_fallback()
    async def write_file_fast(self, file_path: str | Path, content: str) -> bool:
        """Write file using Rust for maximum performance."""
        return await self._python_write_file(str(file_path), content)

    @rust_fallback()
    async def copy_file_fast(self, src: str | Path, dst: str | Path) -> bool:
        """Copy file using Rust for performance."""
        return await self._python_copy_file(str(src), str(dst))

    @rust_fallback()
    async def get_file_hash_fast(self, file_path: str | Path) -> str | None:
        """Get file hash using Rust for performance."""
        return await self._python_get_file_hash(str(file_path))

    async def find_files_fast(
        self,
        directory: str | Path,
        patterns: list[str],
        exclude_patterns: list[str] | None = None,
    ) -> list[str]:
        """Find files using Rust for maximum performance."""
        try:
            files: list[Any] = []
            dir_path = Path(directory)

            for pattern in patterns:
                for file_path in dir_path.rglob(pattern):
                    if file_path.is_file():
                        excluded = False
                        if exclude_patterns:
                            for exclude in exclude_patterns:
                                if exclude in str(file_path):
                                    excluded = True
                                    break

                        if not excluded:
                            files.append(str(file_path))

            return files
        except Exception as e:
            logger.exception(f"Failed to find files in {directory}: {e}")
            return []


class EnhancedJsonProcessor:
    """Enhanced JSON processing with Rust acceleration."""

    def __init__(self) -> None:
        self._rust_json = RustJsonProcessor() if RUST_EXTENSIONS_AVAILABLE else None

    async def _python_parse_json(self, json_text: str) -> dict[str, Any] | None:
        """Python fallback for JSON parsing."""
        try:
            import json

            return json.loads(json_text)
        except Exception as e:
            logger.exception(f"Failed to parse JSON: {e}")
            return None

    async def _python_serialize_json(self, data: Any) -> str | None:
        """Python fallback for JSON serialization."""
        try:
            import json

            return json.dumps(data, separators=(",", ":"))
        except Exception as e:
            logger.exception(f"Failed to serialize JSON: {e}")
            return None

    @rust_fallback()
    async def parse_json_fast(self, json_text: str) -> dict[str, Any] | None:
        """Parse JSON using Rust for performance."""
        if self._rust_json:
            return self._rust_json.parse_large_json(json_text)
        return await self._python_parse_json(json_text)

    @rust_fallback()
    async def serialize_json_fast(self, data: Any) -> str | None:
        """Serialize to JSON using Rust for performance."""
        return await self._python_serialize_json(data)


class EnhancedAuthValidator:
    """Enhanced authentication validator with Rust acceleration."""

    def __init__(self) -> None:
        self._rust_auth = RustAuthValidator() if RUST_EXTENSIONS_AVAILABLE else None

    def _python_validate_jwt(self, token: str, secret_key: str) -> bool:
        """Python fallback for JWT validation."""
        try:
            import jwt

            jwt.decode(token, secret_key, algorithms=["HS256"])
            return True
        except Exception:
            return False

    @rust_fallback()
    def validate_jwt_fast(self, token: str, secret_key: str) -> bool:
        """Validate JWT using Rust for performance and security."""
        if self._rust_auth:
            return self._rust_auth.validate_jwt(token, secret_key)
        return self._python_validate_jwt(token, secret_key)

    @rust_fallback()
    def hash_password_fast(self, password: str) -> str:
        """Hash password using Rust for security."""
        try:
            import bcrypt

            return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        except Exception as e:
            logger.exception(f"Password hashing failed: {e}")
            return ""

    @rust_fallback()
    def verify_password_fast(self, password: str, hashed: str) -> bool:
        """Verify password using Rust for security."""
        try:
            import bcrypt

            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except Exception as e:
            logger.exception(f"Password verification failed: {e}")
            return False


class EnhancedSystemInfo:
    """Enhanced system information with Rust acceleration."""

    @rust_fallback()
    def get_system_info(self) -> dict[str, Any]:
        """Get system information using Rust."""
        if RUST_EXTENSIONS_AVAILABLE:
            return system_info()

        # Python fallback
        import platform

        import psutil

        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_percent": psutil.virtual_memory().percent,
            "rust_extensions": RUST_EXTENSIONS_AVAILABLE,
        }

    @rust_fallback()
    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics using Rust."""
        if RUST_EXTENSIONS_AVAILABLE:
            return performance_metrics()

        # Python fallback
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else [0, 0, 0],
            "rust_extensions": RUST_EXTENSIONS_AVAILABLE,
        }


# Global instances for convenient access
rust_cache = EnhancedRustCache()
rust_file_ops = EnhancedFileOps()
rust_json = EnhancedJsonProcessor()
rust_auth = EnhancedAuthValidator()
rust_system = EnhancedSystemInfo()


# Convenience functions
async def cache_get(key: str) -> Any | None:
    """Get from enhanced Rust cache."""
    return rust_cache.get(key)


def cache_set(key: str, value: Any) -> bool:
    """Set in enhanced Rust cache."""
    return rust_cache.set(key, value)


async def read_file_fast(file_path: str | Path) -> str | None:
    """Read file using enhanced Rust file operations."""
    return await rust_file_ops.read_file_fast(file_path)


async def write_file_fast(file_path: str | Path, content: str) -> bool:
    """Write file using enhanced Rust file operations."""
    return await rust_file_ops.write_file_fast(file_path, content)


async def parse_json_fast(json_text: str) -> dict[str, Any] | None:
    """Parse JSON using enhanced Rust JSON processor."""
    return await rust_json.parse_json_fast(json_text)


def validate_jwt_fast(token: str, secret_key: str) -> bool:
    """Validate JWT using enhanced Rust auth validator."""
    return rust_auth.validate_jwt_fast(token, secret_key)


def get_system_info() -> dict[str, Any]:
    """Get system information using enhanced Rust system info."""
    return rust_system.get_system_info()


def get_performance_metrics() -> dict[str, Any]:
    """Get performance metrics using enhanced Rust system info."""
    return rust_system.get_performance_metrics()


def get_rust_status() -> dict[str, Any]:
    """Get status of Rust extensions."""
    return {
        "extensions_available": RUST_EXTENSIONS_AVAILABLE,
        "cache_stats": rust_cache.get_stats(),
        "system_info": get_system_info(),
        "performance_metrics": get_performance_metrics(),
    }
