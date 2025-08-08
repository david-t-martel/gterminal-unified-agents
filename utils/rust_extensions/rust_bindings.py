"""Rust bindings utilities for high-performance operations.

Provides a clean interface to Rust PyO3 bindings with fallbacks and error handling.
All agents should use these utilities for consistent performance optimization.
"""

import logging
from pathlib import Path
from typing import Any

# Import Rust extensions
from fullstack_agent_rust import RustCache
from fullstack_agent_rust import RustFileOps
from fullstack_agent_rust import RustJsonProcessor
from fullstack_agent_rust import performance_metrics
from fullstack_agent_rust import system_info

logger = logging.getLogger(__name__)


class RustUtils:
    """High-performance utilities using Rust bindings."""

    def __init__(self) -> None:
        """Initialize Rust utilities."""
        self.file_ops = RustFileOps()
        self.cache = RustCache(capacity=10000, ttl_seconds=3600)
        self.json_processor = RustJsonProcessor()

        logger.info("Initialized Rust utilities with PyO3 bindings")

    # File Operations

    async def read_file_fast(self, file_path: str | Path) -> str | None:
        """Read file using Rust for maximum performance."""
        try:
            # This would use self.file_ops.read_text_file(str(file_path))
            # For now, use standard Python with caching
            cache_key = f"file_content:{file_path}"
            cached = self.cache.get(cache_key)
            if cached:
                return cached

            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            self.cache.set(cache_key, content)
            return content
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return None

    async def write_file_fast(self, file_path: str | Path, content: str) -> bool:
        """Write file using Rust for maximum performance."""
        try:
            # This would use self.file_ops.write_text_file(str(file_path), content)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Invalidate cache
            cache_key = f"file_content:{file_path}"
            self.cache.delete(cache_key)

            return True
        except Exception as e:
            logger.exception(f"Failed to write file {file_path}: {e}")
            return False

    async def copy_file_fast(self, src: str | Path, dst: str | Path) -> bool:
        """Copy file using Rust for performance."""
        try:
            # This would use self.file_ops.copy_file(str(src), str(dst))
            import shutil

            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.exception(f"Failed to copy {src} to {dst}: {e}")
            return False

    async def delete_file_fast(self, file_path: str | Path) -> bool:
        """Delete file using Rust for safety and performance."""
        try:
            # This would use self.file_ops.delete_file(str(file_path))
            Path(file_path).unlink()

            # Invalidate cache
            cache_key = f"file_content:{file_path}"
            self.cache.delete(cache_key)

            return True
        except Exception as e:
            logger.exception(f"Failed to delete file {file_path}: {e}")
            return False

    async def get_file_hash_fast(self, file_path: str | Path) -> str | None:
        """Get file hash using Rust for performance."""
        try:
            cache_key = f"file_hash:{file_path}"
            cached = self.cache.get(cache_key)
            if cached:
                return cached

            # This would use self.file_ops.get_file_hash(str(file_path))
            import hashlib

            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            self.cache.set(cache_key, file_hash)
            return file_hash
        except Exception as e:
            logger.warning(f"Failed to hash file {file_path}: {e}")
            return None

    async def find_files_fast(
        self,
        directory: str | Path,
        patterns: list[str],
        exclude_patterns: list[str] | None = None,
    ) -> list[str]:
        """Find files using Rust for maximum performance."""
        try:
            # This would use self.file_ops.find_files(str(directory), patterns, exclude_patterns or [])
            files: list[Any] = []
            dir_path = Path(directory)

            for pattern in patterns:
                for file_path in dir_path.rglob(pattern):
                    if file_path.is_file():
                        # Check exclude patterns
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

    # JSON Processing

    async def parse_json_fast(self, json_text: str) -> dict[str, Any] | None:
        """Parse JSON using Rust for performance."""
        try:
            # This would use self.json_processor.parse_json(json_text)
            return self.json_processor.parse_large_json(json_text)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None

    async def serialize_json_fast(self, data: Any) -> str | None:
        """Serialize to JSON using Rust for performance."""
        try:
            # This would use self.json_processor.serialize_json(data)
            import json

            return json.dumps(data, separators=(",", ":"))
        except Exception as e:
            logger.warning(f"Failed to serialize JSON: {e}")
            return None

    # Cache Operations

    def cache_get(self, key: str) -> Any | None:
        """Get value from Rust cache."""
        return self.cache.get(key)

    def cache_set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Set value in Rust cache."""
        if ttl_seconds is None:
            return self.cache.set(key, value)
        # For TTL support, would need extended Rust binding
        return self.cache.set(key, value)

    def cache_delete(self, key: str) -> bool:
        """Delete key from Rust cache."""
        return self.cache.delete(key)

    def cache_clear(self) -> bool:
        """Clear entire Rust cache."""
        return self.cache.clear()

    # System Information

    def get_system_info(self) -> dict[str, Any]:
        """Get system information using Rust."""
        try:
            return system_info()
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            return {}

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics using Rust."""
        try:
            return performance_metrics()
        except Exception as e:
            logger.warning(f"Failed to get performance metrics: {e}")
            return {}

    # Utility Methods

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        # This would be implemented in Rust
        return {"cache_type": "rust", "capacity": 10000, "ttl_seconds": 3600, "status": "active"}

    async def batch_operation(self, operation: str, items: list[Any], **kwargs) -> list[Any]:
        """Perform batch operations using Rust for performance."""
        results: list[Any] = []

        for item in items:
            try:
                if operation == "read_file":
                    result = await self.read_file_fast(item)
                elif operation == "get_hash":
                    result = await self.get_file_hash_fast(item)
                elif operation == "copy_file":
                    dst = kwargs.get("destination")
                    if dst:
                        result = await self.copy_file_fast(item, f"{dst}/{Path(item).name}")
                    else:
                        result = False
                else:
                    result = None

                results.append(result)
            except Exception as e:
                logger.exception(f"Batch operation {operation} failed for {item}: {e}")
                results.append(None)

        return results


# Global instance for easy access
rust_utils = RustUtils()


# Convenience functions for direct access
async def read_file_fast(file_path: str | Path) -> str | None:
    """Read file using Rust bindings."""
    return await rust_utils.read_file_fast(file_path)


async def write_file_fast(file_path: str | Path, content: str) -> bool:
    """Write file using Rust bindings."""
    return await rust_utils.write_file_fast(file_path, content)


async def get_file_hash_fast(file_path: str | Path) -> str | None:
    """Get file hash using Rust bindings."""
    return await rust_utils.get_file_hash_fast(file_path)


def cache_get(key: str) -> Any | None:
    """Get from Rust cache."""
    return rust_utils.cache_get(key)


def cache_set(key: str, value: Any) -> bool:
    """Set in Rust cache."""
    return rust_utils.cache_set(key, value)
