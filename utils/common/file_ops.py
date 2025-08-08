"""Shared file operations utilities for all agents.

Provides consistent file handling with performance optimization,
safety checks, and error handling across all agents.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

from .rust_bindings import rust_utils

logger = logging.getLogger(__name__)


class FileOperations:
    """Shared file operations with Rust performance optimization."""

    def __init__(self) -> None:
        """Initialize file operations."""
        self.rust = rust_utils

        # Common file patterns
        self.code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".php",
            ".rb",
            ".swift",
            ".kt",
            ".scala",
            ".m",
            ".mm",
        }

        self.config_extensions = {
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".conf",
            ".cfg",
            ".ini",
            ".toml",
        }

        self.exclude_patterns = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "env",
            "target",
            "build",
            "dist",
            ".cache",
            ".tox",
            ".pytest_cache",
            ".mypy_cache",
            ".coverage",
            "*.egg-info",
            ".DS_Store",
        }

        logger.info("Initialized FileOperations with Rust bindings")

    async def read_file_safe(self, file_path: str | Path) -> str | None:
        """Safely read file with Rust performance and error handling."""
        try:
            return await self.rust.read_file_fast(file_path)
        except Exception as e:
            logger.warning(f"Safe file read failed for {file_path}: {e}")
            return None

    async def write_file_safe(
        self, file_path: str | Path, content: str, create_backup: bool = True
    ) -> bool:
        """Safely write file with optional backup."""
        try:
            file_path = Path(file_path)

            # Create backup if requested and file exists
            if create_backup and file_path.exists():
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                await self.rust.copy_file_fast(file_path, backup_path)

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            return await self.rust.write_file_fast(file_path, content)
        except Exception as e:
            logger.exception(f"Safe file write failed for {file_path}: {e}")
            return False

    async def find_files_by_pattern(
        self,
        directory: str | Path,
        patterns: list[str],
        exclude_dirs: list[str] | None = None,
    ) -> list[Path]:
        """Find files matching patterns with exclusions."""
        try:
            exclude_dirs = exclude_dirs or list(self.exclude_patterns)

            found_files = await self.rust.find_files_fast(directory, patterns, exclude_dirs)

            return [Path(f) for f in found_files]
        except Exception as e:
            logger.exception(f"Pattern search failed in {directory}: {e}")
            return []

    async def find_code_files(self, directory: str | Path) -> list[Path]:
        """Find all code files in directory."""
        patterns = [f"*{ext}" for ext in self.code_extensions]
        return await self.find_files_by_pattern(directory, patterns)

    async def find_config_files(self, directory: str | Path) -> list[Path]:
        """Find all configuration files in directory."""
        patterns = [f"*{ext}" for ext in self.config_extensions]
        return await self.find_files_by_pattern(directory, patterns)

    async def get_file_metadata(self, file_path: str | Path) -> dict[str, Any]:
        """Get comprehensive file metadata."""
        try:
            path = Path(file_path)
            stat = path.stat()

            # Get hash using Rust
            file_hash = await self.rust.get_file_hash_fast(file_path)

            return {
                "path": str(path),
                "name": path.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "ctime": stat.st_ctime,
                "hash": file_hash,
                "extension": path.suffix,
                "is_code": path.suffix in self.code_extensions,
                "is_config": path.suffix in self.config_extensions,
            }
        except Exception as e:
            logger.exception(f"Failed to get metadata for {file_path}: {e}")
            return {}

    async def compare_files(self, file1: str | Path, file2: str | Path) -> dict[str, Any]:
        """Compare two files for similarity."""
        try:
            # Get hashes first (fast comparison)
            hash1 = await self.rust.get_file_hash_fast(file1)
            hash2 = await self.rust.get_file_hash_fast(file2)

            if not hash1 or not hash2:
                return {"error": "Failed to hash files"}

            # Exact match
            if hash1 == hash2:
                return {"identical": True, "similarity": 1.0, "hash1": hash1, "hash2": hash2}

            # Size comparison for similarity estimate
            try:
                size1 = Path(file1).stat().st_size
                size2 = Path(file2).stat().st_size

                size_similarity = (
                    0.0
                    if size1 == 0 or size2 == 0
                    else 1.0 - abs(size1 - size2) / max(size1, size2)
                )

                return {
                    "identical": False,
                    "similarity": size_similarity,
                    "hash1": hash1,
                    "hash2": hash2,
                    "size1": size1,
                    "size2": size2,
                }
            except OSError:
                return {"error": "Failed to get file sizes"}

        except Exception as e:
            logger.exception(f"File comparison failed: {e}")
            return {"error": str(e)}

    async def create_backup(
        self, file_path: str | Path, backup_dir: str | Path | None = None
    ) -> Path | None:
        """Create a backup of a file."""
        try:
            source_path = Path(file_path)

            if backup_dir:
                backup_path = Path(backup_dir) / f"{source_path.name}.backup"
                # Ensure backup directory exists
                Path(backup_dir).mkdir(parents=True, exist_ok=True)
            else:
                backup_path = source_path.with_suffix(f"{source_path.suffix}.backup")

            success = await self.rust.copy_file_fast(source_path, backup_path)

            if success:
                logger.info(f"Created backup: {backup_path}")
                return backup_path
            logger.error(f"Failed to create backup for {file_path}")
            return None

        except Exception as e:
            logger.exception(f"Backup creation failed for {file_path}: {e}")
            return None

    async def batch_process_files(
        self, files: list[str | Path], operation: str, **kwargs
    ) -> list[dict[str, Any]]:
        """Process multiple files in batch with Rust performance."""
        results: list[Any] = []

        for file_path in files:
            try:
                if operation == "metadata":
                    result = await self.get_file_metadata(file_path)
                elif operation == "hash":
                    hash_val = await self.rust.get_file_hash_fast(file_path)
                    result = {"path": str(file_path), "hash": hash_val}
                elif operation == "backup":
                    backup_path = await self.create_backup(file_path, kwargs.get("backup_dir"))
                    result = {
                        "path": str(file_path),
                        "backup": str(backup_path) if backup_path else None,
                        "success": backup_path is not None,
                    }
                elif operation == "read":
                    content = await self.read_file_safe(file_path)
                    result = {
                        "path": str(file_path),
                        "content": content,
                        "success": content is not None,
                    }
                else:
                    result = {"error": f"Unknown operation: {operation}"}

                results.append(result)

            except Exception as e:
                logger.exception(f"Batch operation {operation} failed for {file_path}: {e}")
                results.append({"path": str(file_path), "error": str(e), "success": False})

        return results

    async def cleanup_backups(self, directory: str | Path, older_than_days: int = 7) -> int:
        """Clean up old backup files."""
        try:
            backup_files = await self.find_files_by_pattern(directory, ["*.backup", "*.bak"])

            cleaned_count = 0
            cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 3600)

            for backup_file in backup_files:
                try:
                    if backup_file.stat().st_mtime < cutoff_time:
                        backup_file.unlink()
                        cleaned_count += 1
                        logger.debug(f"Cleaned old backup: {backup_file}")
                except OSError:
                    continue

            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} old backup files")

            return cleaned_count

        except Exception as e:
            logger.exception(f"Backup cleanup failed: {e}")
            return 0

    def is_excluded_path(self, path: str | Path) -> bool:
        """Check if path should be excluded based on common patterns."""
        path_str = str(path)
        return any(pattern in path_str for pattern in self.exclude_patterns)

    def get_file_type(self, file_path: str | Path) -> str:
        """Get file type classification."""
        ext = Path(file_path).suffix.lower()

        if ext in self.code_extensions:
            return "code"
        if ext in self.config_extensions:
            return "config"
        if ext in {".md", ".txt", ".rst"}:
            return "documentation"
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
            return "image"
        if ext in {".mp4", ".avi", ".mkv", ".mov"}:
            return "video"
        if ext in {".mp3", ".wav", ".flac", ".ogg"}:
            return "audio"
        return "other"


# Global instance for easy access
file_ops = FileOperations()
