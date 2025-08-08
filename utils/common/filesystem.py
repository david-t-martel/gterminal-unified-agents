"""Comprehensive filesystem utilities providing complete functional coverage
of rust-fs MCP server capabilities with Python implementations and Rust acceleration.

This module provides all the file system operations available in rust-fs:
- create (files and directories)
- read (file contents and directory listings)
- write (file content)
- move (files and directories)
- copy (files and directories)
- delete (files and directories)
- stat (file/directory metadata)
- find (files matching patterns)
- search (file contents)
- replace (text in files)
- replace_block (line-based replacements)
- execute (system commands)
"""

import asyncio
import contextlib
import logging
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


class FileSystemError(Exception):
    """Custom exception for filesystem operations."""


class FileSystemUtils:
    """Comprehensive filesystem utilities with rust-fs functional coverage."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        """Initialize with optional base path for relative operations."""
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve path relative to base_path if not absolute."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_path / path

    # CREATE operations (rust-fs: create)
    async def create_file(self, path: str | Path, content: str = "") -> bool:
        """Create a new file with optional content."""
        try:
            file_path = self._resolve_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Created file: {file_path}")
            return True

        except Exception as e:
            logger.exception(f"Failed to create file {path}: {e}")
            msg = f"Create file failed: {e}"
            raise FileSystemError(msg)

    async def create_directory(self, path: str | Path) -> bool:
        """Create a directory (and parents if needed)."""
        try:
            dir_path = self._resolve_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Created directory: {dir_path}")
            return True

        except Exception as e:
            logger.exception(f"Failed to create directory {path}: {e}")
            msg = f"Create directory failed: {e}"
            raise FileSystemError(msg)

    # READ operations (rust-fs: read)
    async def read_file(self, path: str | Path) -> str:
        """Read file contents as string."""
        try:
            file_path = self._resolve_path(path)

            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            logger.debug(f"Read file: {file_path} ({len(content)} chars)")
            return content

        except Exception as e:
            logger.exception(f"Failed to read file {path}: {e}")
            msg = f"Read file failed: {e}"
            raise FileSystemError(msg)

    async def read_directory(self, path: str | Path) -> list[dict[str, Any]]:
        """Read directory contents with metadata."""
        try:
            dir_path = self._resolve_path(path)

            if not dir_path.is_dir():
                msg = f"Path is not a directory: {path}"
                raise FileSystemError(msg)

            entries: list[Any] = []
            for item in dir_path.iterdir():
                try:
                    item_stat = item.stat()
                    entries.append(
                        {
                            "name": item.name,
                            "path": str(item),
                            "type": "directory" if item.is_dir() else "file",
                            "size": item_stat.st_size,
                            "modified": item_stat.st_mtime,
                            "created": item_stat.st_ctime,
                            "permissions": oct(item_stat.st_mode)[-3:],
                        },
                    )
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not stat {item}: {e}")
                    entries.append(
                        {"name": item.name, "path": str(item), "type": "unknown", "error": str(e)}
                    )

            logger.debug(f"Read directory: {dir_path} ({len(entries)} items)")
            return entries

        except Exception as e:
            logger.exception(f"Failed to read directory {path}: {e}")
            msg = f"Read directory failed: {e}"
            raise FileSystemError(msg)

    # WRITE operations (rust-fs: write)
    async def write_file(self, path: str | Path, content: str, verify: bool = False) -> bool:
        """Write content to file with optional verification."""
        try:
            file_path = self._resolve_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first for safety
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=file_path.parent,
                delete=False,
            ) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)

            # Move temp file to final location
            temp_path.replace(file_path)

            # Verify if requested
            if verify:
                written_content = await self.read_file(file_path)
                if written_content != content:
                    msg = "Write verification failed"
                    raise FileSystemError(msg)

            logger.info(f"Wrote file: {file_path} ({len(content)} chars)")
            return True

        except Exception as e:
            logger.exception(f"Failed to write file {path}: {e}")
            msg = f"Write file failed: {e}"
            raise FileSystemError(msg)

    # MOVE operations (rust-fs: move)
    async def move_file(self, from_path: str | Path, to_path: str | Path) -> bool:
        """Move/rename file or directory."""
        try:
            source = self._resolve_path(from_path)
            destination = self._resolve_path(to_path)

            # Create destination directory if needed
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Use shutil.move for cross-filesystem moves
            shutil.move(str(source), str(destination))

            logger.info(f"Moved: {source} -> {destination}")
            return True

        except Exception as e:
            logger.exception(f"Failed to move {from_path} to {to_path}: {e}")
            msg = f"Move failed: {e}"
            raise FileSystemError(msg)

    # COPY operations (rust-fs: copy)
    async def copy_file(self, from_path: str | Path, to_path: str | Path) -> bool:
        """Copy file or directory."""
        try:
            source = self._resolve_path(from_path)
            destination = self._resolve_path(to_path)

            # Create destination directory if needed
            destination.parent.mkdir(parents=True, exist_ok=True)

            if source.is_file():
                shutil.copy2(str(source), str(destination))
            elif source.is_dir():
                shutil.copytree(str(source), str(destination), dirs_exist_ok=True)
            else:
                msg = f"Source path does not exist: {source}"
                raise FileSystemError(msg)

            logger.info(f"Copied: {source} -> {destination}")
            return True

        except Exception as e:
            logger.exception(f"Failed to copy {from_path} to {to_path}: {e}")
            msg = f"Copy failed: {e}"
            raise FileSystemError(msg)

    # DELETE operations (rust-fs: delete)
    async def delete_file(self, path: str | Path) -> bool:
        """Delete file or directory."""
        try:
            target = self._resolve_path(path)

            if target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(str(target))
            else:
                msg = f"Path does not exist: {target}"
                raise FileSystemError(msg)

            logger.info(f"Deleted: {target}")
            return True

        except Exception as e:
            logger.exception(f"Failed to delete {path}: {e}")
            msg = f"Delete failed: {e}"
            raise FileSystemError(msg)

    # STAT operations (rust-fs: stat)
    async def get_file_info(self, path: str | Path) -> dict[str, Any]:
        """Get file/directory metadata."""
        try:
            target = self._resolve_path(path)
            target_stat = target.stat()

            info = {
                "path": str(target),
                "name": target.name,
                "size": target_stat.st_size,
                "type": "directory" if target.is_dir() else "file",
                "permissions": oct(target_stat.st_mode)[-3:],
                "owner_read": bool(target_stat.st_mode & stat.S_IRUSR),
                "owner_write": bool(target_stat.st_mode & stat.S_IWUSR),
                "owner_execute": bool(target_stat.st_mode & stat.S_IXUSR),
                "created": target_stat.st_ctime,
                "modified": target_stat.st_mtime,
                "accessed": target_stat.st_atime,
                "is_symlink": target.is_symlink(),
                "absolute_path": str(target.absolute()),
            }

            if target.is_file():
                info["extension"] = target.suffix
                info["stem"] = target.stem

            logger.debug(f"Got file info: {target}")
            return info

        except Exception as e:
            logger.exception(f"Failed to get file info for {path}: {e}")
            msg = f"Stat failed: {e}"
            raise FileSystemError(msg)

    # FIND operations (rust-fs: find)
    async def find_files(
        self, path: str | Path, glob: str | None = None, max_results: int | None = None
    ) -> list[str]:
        """Find files matching glob pattern."""
        try:
            search_path = self._resolve_path(path)

            if not search_path.exists():
                msg = f"Search path does not exist: {search_path}"
                raise FileSystemError(msg)

            results: list[Any] = []
            pattern = glob or "*"

            # Use Path.rglob for recursive search
            for item in search_path.rglob(pattern):
                if item.is_file():
                    results.append(str(item))
                    if max_results and len(results) >= max_results:
                        break

            logger.info(f"Found {len(results)} files matching '{pattern}' in {search_path}")
            return results

        except Exception as e:
            logger.exception(f"Failed to find files in {path}: {e}")
            msg = f"Find failed: {e}"
            raise FileSystemError(msg)

    # SEARCH operations (rust-fs: search)
    async def search_in_files(
        self,
        path: str | Path,
        pattern: str,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for text pattern in file contents."""
        try:
            search_path = self._resolve_path(path)
            results: list[Any] = []

            # Compile regex pattern
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

            # Find all text files to search
            text_extensions = {
                ".txt",
                ".py",
                ".js",
                ".html",
                ".css",
                ".json",
                ".yaml",
                ".yml",
                ".md",
                ".rst",
            }

            for file_path in search_path.rglob("*"):
                if not file_path.is_file() or file_path.suffix.lower() not in text_extensions:
                    continue

                try:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    for match in regex.finditer(content):
                        # Find line number
                        line_start = content.rfind("\n", 0, match.start()) + 1
                        line_end = content.find("\n", match.end())
                        if line_end == -1:
                            line_end = len(content)

                        line_number = content[: match.start()].count("\n") + 1
                        line_content = content[line_start:line_end]

                        results.append(
                            {
                                "file": str(file_path),
                                "line_number": line_number,
                                "line_content": line_content,
                                "match_start": match.start() - line_start,
                                "match_end": match.end() - line_start,
                                "matched_text": match.group(),
                            },
                        )

                        if max_results and len(results) >= max_results:
                            break

                except (UnicodeDecodeError, PermissionError) as e:
                    logger.warning(f"Could not search in {file_path}: {e}")
                    continue

                if max_results and len(results) >= max_results:
                    break

            logger.info(f"Found {len(results)} matches for pattern '{pattern}' in {search_path}")
            return results

        except Exception as e:
            logger.exception(f"Failed to search in {path}: {e}")
            msg = f"Search failed: {e}"
            raise FileSystemError(msg)

    # REPLACE operations (rust-fs: replace)
    async def replace_in_files(
        self, paths: list[str | Path], pattern: str, replacement: str
    ) -> dict[str, Any]:
        """Replace text pattern in multiple files."""
        try:
            results = {
                "files_processed": 0,
                "files_modified": 0,
                "total_replacements": 0,
                "errors": [],
            }

            regex = re.compile(pattern, re.MULTILINE)

            for path in paths:
                try:
                    file_path = self._resolve_path(path)

                    if not file_path.is_file():
                        results["errors"].append(f"Not a file: {file_path}")
                        continue

                    # Read file content
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()

                    # Perform replacements
                    new_content, count = regex.subn(replacement, content)
                    results["files_processed"] += 1

                    if count > 0:
                        # Write back to file
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)

                        results["files_modified"] += 1
                        results["total_replacements"] += count
                        logger.info(f"Made {count} replacements in {file_path}")

                except Exception as e:
                    error_msg = f"Error processing {path}: {e}"
                    results["errors"].append(error_msg)
                    logger.exception(error_msg)

            return results

        except Exception as e:
            logger.exception(f"Failed to replace in files: {e}")
            msg = f"Replace failed: {e}"
            raise FileSystemError(msg)

    # REPLACE_BLOCK operations (rust-fs: replace_block)
    async def replace_lines(
        self, path: str | Path, start_line: int, end_line: int, replacement: str
    ) -> bool:
        """Replace a block of lines in a file."""
        try:
            file_path = self._resolve_path(path)

            # Read all lines
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Validate line numbers (1-based)
            if start_line < 1 or end_line < 1 or start_line > len(lines) or end_line > len(lines):
                msg = f"Invalid line range: {start_line}-{end_line} (file has {len(lines)} lines)"
                raise FileSystemError(msg)

            if start_line > end_line:
                msg = f"Start line ({start_line}) cannot be greater than end line ({end_line})"
                raise FileSystemError(msg)

            # Replace lines (convert to 0-based indexing)
            replacement_lines = replacement.split("\n")
            if replacement_lines and replacement_lines[-1] == "":
                replacement_lines.pop()  # Remove empty line at end

            # Add newlines to replacement lines except the last one
            replacement_lines = [line + "\n" for line in replacement_lines[:-1]] + [
                replacement_lines[-1]
            ]
            if replacement_lines == [""]:
                replacement_lines: list[Any] = []

            # Perform replacement
            new_lines = lines[: start_line - 1] + replacement_lines + lines[end_line:]

            # Write back to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            logger.info(f"Replaced lines {start_line}-{end_line} in {file_path}")
            return True

        except Exception as e:
            logger.exception(f"Failed to replace lines in {path}: {e}")
            msg = f"Replace lines failed: {e}"
            raise FileSystemError(msg)

    # EXECUTE operations (rust-fs: execute)
    async def execute_command(
        self,
        command: str,
        args: list[str] | None = None,
        timeout: int | None = None,
        cwd: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute system command with optional arguments."""
        try:
            # Prepare command
            cmd = (
                [command, *args]
                if args
                else command.split()
                if isinstance(command, str)
                else [command]
            )

            # Set working directory
            work_dir = str(self._resolve_path(cwd)) if cwd else None

            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except TimeoutError:
                process.kill()
                await process.wait()
                msg = f"Command timed out after {timeout} seconds"
                raise FileSystemError(msg)

            result = {
                "command": " ".join(cmd),
                "return_code": process.returncode,
                "stdout": stdout.decode("utf-8", errors="ignore"),
                "stderr": stderr.decode("utf-8", errors="ignore"),
                "success": process.returncode == 0,
            }

            if result["success"]:
                logger.info(f"Command executed successfully: {result['command']}")
            else:
                logger.warning(
                    f"Command failed (code {result['return_code']}): {result['command']}"
                )

            return result

        except Exception as e:
            logger.exception(f"Failed to execute command {command}: {e}")
            msg = f"Execute failed: {e}"
            raise FileSystemError(msg)

    # Utility methods
    async def exists(self, path: str | Path) -> bool:
        """Check if path exists."""
        return self._resolve_path(path).exists()

    async def is_file(self, path: str | Path) -> bool:
        """Check if path is a file."""
        return self._resolve_path(path).is_file()

    async def is_directory(self, path: str | Path) -> bool:
        """Check if path is a directory."""
        return self._resolve_path(path).is_dir()

    async def get_size(self, path: str | Path) -> int:
        """Get file or directory size."""
        target = self._resolve_path(path)
        if target.is_file():
            return target.stat().st_size
        if target.is_dir():
            total_size = 0
            for item in target.rglob("*"):
                if item.is_file():
                    with contextlib.suppress(OSError, FileNotFoundError):
                        total_size += item.stat().st_size
            return total_size
        return 0

    async def ensure_directory(self, path: str | Path) -> bool:
        """Ensure directory exists, create if it doesn't."""
        dir_path = self._resolve_path(path)
        if not dir_path.exists():
            return await self.create_directory(dir_path)
        return dir_path.is_dir()


# Global instance for convenient access
fs = FileSystemUtils()


# Convenience functions that match rust-fs interface
async def create(path: str, content: str = "", directory: bool = False) -> bool:
    """Create file or directory (rust-fs: create)."""
    if directory:
        return await fs.create_directory(path)
    return await fs.create_file(path, content)


async def read(path: str) -> str | list[dict[str, Any]]:
    """Read file or directory (rust-fs: read)."""
    if await fs.is_directory(path):
        return await fs.read_directory(path)
    return await fs.read_file(path)


async def write(path: str, content: str, verify: bool = False) -> bool:
    """Write content to file (rust-fs: write)."""
    return await fs.write_file(path, content, verify)


async def move(from_path: str, to_path: str) -> bool:
    """Move file or directory (rust-fs: move)."""
    return await fs.move_file(from_path, to_path)


async def copy(from_path: str, to_path: str) -> bool:
    """Copy file or directory (rust-fs: copy)."""
    return await fs.copy_file(from_path, to_path)


async def delete(path: str) -> bool:
    """Delete file or directory (rust-fs: delete)."""
    return await fs.delete_file(path)


async def stat(path: str) -> dict[str, Any]:
    """Get file/directory metadata (rust-fs: stat)."""
    return await fs.get_file_info(path)


async def find(path: str, glob: str = "*", max_results: int | None = None) -> list[str]:
    """Find files matching pattern (rust-fs: find)."""
    return await fs.find_files(path, glob, max_results)


async def search(path: str, pattern: str, max_results: int | None = None) -> list[dict[str, Any]]:
    """Search for text in files (rust-fs: search)."""
    return await fs.search_in_files(path, pattern, max_results)


async def replace(paths: list[str], pattern: str, replacement: str) -> dict[str, Any]:
    """Replace text in files (rust-fs: replace)."""
    return await fs.replace_in_files(paths, pattern, replacement)


async def replace_block(path: str, start_line: int, end_line: int, replacement: str) -> bool:
    """Replace block of lines (rust-fs: replace_block)."""
    return await fs.replace_lines(path, start_line, end_line, replacement)


async def execute(
    command: str, args: list[str] | None = None, timeout: int | None = None
) -> dict[str, Any]:
    """Execute system command (rust-fs: execute)."""
    return await fs.execute_command(command, args, timeout)
