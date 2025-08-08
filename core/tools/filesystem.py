#!/usr/bin/env python3
"""File System Tools - High-performance file operations using Rust extensions."""

import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any

from gterminal.core.tools.registry import BaseTool
from gterminal.core.tools.registry import ToolParameter
from gterminal.core.tools.registry import ToolResult

# Try to import Rust extensions for performance
try:
    from fullstack_agent_rust import RustFileOps

    RUST_AVAILABLE = True
    rust_file_ops = RustFileOps()
except ImportError:
    RUST_AVAILABLE = False
    rust_file_ops = None

logger = logging.getLogger(__name__)


class ReadFileTool(BaseTool):
    """Tool for reading file contents."""

    def __init__(self) -> None:
        super().__init__(
            name="read_file",
            description="Read the contents of a file",
            category="filesystem",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file to read",
                required=True,
            ),
            ToolParameter(
                name="encoding",
                type="string",
                description="File encoding",
                required=False,
                default="utf-8",
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            path = Path(params["path"])
            encoding = params.get("encoding", "utf-8")

            if not path.exists():
                return ToolResult(success=False, data=None, error=f"File not found: {path}")

            # Use Rust extension if available for better performance
            if RUST_AVAILABLE and rust_file_ops:
                content = await rust_file_ops.read_file(str(path))
            else:
                with path.open(encoding=encoding) as f:
                    content = f.read()

            return ToolResult(
                success=True,
                data={"path": str(path), "content": content, "size": len(content)},
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class WriteFileTool(BaseTool):
    """Tool for writing content to files."""

    def __init__(self) -> None:
        super().__init__(
            name="write_file",
            description="Write content to a file",
            category="filesystem",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file to write",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Content to write to the file",
                required=True,
            ),
            ToolParameter(
                name="create_dirs",
                type="boolean",
                description="Create parent directories if they don't exist",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="encoding",
                type="string",
                description="File encoding",
                required=False,
                default="utf-8",
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            path = Path(params["path"])
            content = params["content"]
            create_dirs = params.get("create_dirs", True)
            encoding = params.get("encoding", "utf-8")

            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            # Use Rust extension if available for better performance
            if RUST_AVAILABLE and rust_file_ops:
                await rust_file_ops.write_file(str(path), content)
            else:
                with path.open("w", encoding=encoding) as f:
                    f.write(content)

            return ToolResult(
                success=True,
                data={"path": str(path), "size": len(content), "created": True},
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""

    def __init__(self) -> None:
        super().__init__(
            name="list_directory",
            description="List contents of a directory",
            category="filesystem",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the directory",
                required=True,
            ),
            ToolParameter(
                name="recursive",
                type="boolean",
                description="List recursively",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="File pattern to filter (e.g., '*.py')",
                required=False,
                default=None,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            path = Path(params["path"])
            recursive = params.get("recursive", False)
            pattern = params.get("pattern")

            if not path.exists():
                return ToolResult(success=False, data=None, error=f"Directory not found: {path}")

            if not path.is_dir():
                return ToolResult(success=False, data=None, error=f"Not a directory: {path}")

            files: list[Any] = []
            directories: list[Any] = []

            if recursive:
                items = path.rglob(pattern) if pattern else path.rglob("*")
            elif pattern:
                items = path.glob(pattern)
            else:
                items = path.iterdir()

            for item in items:
                relative_path = item.relative_to(path)
                if item.is_file():
                    files.append(
                        {
                            "name": str(relative_path),
                            "size": item.stat().st_size,
                            "modified": item.stat().st_mtime,
                        },
                    )
                elif item.is_dir():
                    directories.append({"name": str(relative_path)})

            return ToolResult(
                success=True,
                data={
                    "path": str(path),
                    "files": files,
                    "directories": directories,
                    "total_files": len(files),
                    "total_directories": len(directories),
                },
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class SearchFilesTool(BaseTool):
    """Tool for searching files using ripgrep or fallback methods."""

    def __init__(self) -> None:
        super().__init__(
            name="search_files",
            description="Search for patterns in files",
            category="filesystem",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Directory to search in",
                required=True,
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="Pattern to search for",
                required=True,
            ),
            ToolParameter(
                name="file_pattern",
                type="string",
                description="File pattern to search in (e.g., '*.py')",
                required=False,
                default="*",
            ),
            ToolParameter(
                name="case_sensitive",
                type="boolean",
                description="Case sensitive search",
                required=False,
                default=True,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            path = Path(params["path"])
            pattern = params["pattern"]
            file_pattern = params.get("file_pattern", "*")
            case_sensitive = params.get("case_sensitive", True)

            if not path.exists():
                return ToolResult(success=False, data=None, error=f"Directory not found: {path}")

            # Try to use ripgrep if available
            try:
                rg_args = ["rg", pattern, str(path)]
                if not case_sensitive:
                    rg_args.append("-i")
                if file_pattern != "*":
                    rg_args.extend(["-g", file_pattern])

                result = subprocess.run(
                    rg_args, capture_output=True, text=True, timeout=30, check=False
                )

                matches: list[Any] = []
                for line in result.stdout.splitlines():
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        matches.append(
                            {
                                "file": parts[0],
                                "line_number": parts[1],
                                "content": parts[2],
                            },
                        )

                return ToolResult(
                    success=True,
                    data={"pattern": pattern, "matches": matches, "count": len(matches)},
                )

            except (subprocess.SubprocessError, FileNotFoundError):
                # Fallback to Python search
                matches: list[Any] = []
                for file_path in path.rglob(file_pattern):
                    if file_path.is_file():
                        try:
                            with file_path.open(encoding="utf-8") as f:
                                for line_num, line in enumerate(f, 1):
                                    search_line = line if case_sensitive else line.lower()
                                    search_pattern = pattern if case_sensitive else pattern.lower()
                                    if search_pattern in search_line:
                                        matches.append(
                                            {
                                                "file": str(file_path),
                                                "line_number": line_num,
                                                "content": line.strip(),
                                            },
                                        )
                        except (UnicodeDecodeError, PermissionError):
                            continue

                return ToolResult(
                    success=True,
                    data={"pattern": pattern, "matches": matches, "count": len(matches)},
                )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class DeleteFileTool(BaseTool):
    """Tool for deleting files or directories."""

    def __init__(self) -> None:
        super().__init__(
            name="delete_file",
            description="Delete a file or directory",
            category="filesystem",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to delete",
                required=True,
            ),
            ToolParameter(
                name="recursive",
                type="boolean",
                description="Delete directories recursively",
                required=False,
                default=False,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            path = Path(params["path"])
            recursive = params.get("recursive", False)

            if not path.exists():
                return ToolResult(success=False, data=None, error=f"Path not found: {path}")

            if path.is_file():
                path.unlink()
            elif path.is_dir():
                if recursive:
                    shutil.rmtree(path)
                else:
                    path.rmdir()
            else:
                return ToolResult(success=False, data=None, error=f"Unknown path type: {path}")

            return ToolResult(success=True, data={"deleted": str(path)})

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class MoveFileTool(BaseTool):
    """Tool for moving or renaming files."""

    def __init__(self) -> None:
        super().__init__(
            name="move_file",
            description="Move or rename a file or directory",
            category="filesystem",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="source",
                type="string",
                description="Source path",
                required=True,
            ),
            ToolParameter(
                name="destination",
                type="string",
                description="Destination path",
                required=True,
            ),
            ToolParameter(
                name="overwrite",
                type="boolean",
                description="Overwrite if destination exists",
                required=False,
                default=False,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            source = Path(params["source"])
            destination = Path(params["destination"])
            overwrite = params.get("overwrite", False)

            if not source.exists():
                return ToolResult(success=False, data=None, error=f"Source not found: {source}")

            if destination.exists() and not overwrite:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Destination already exists: {destination}",
                )

            # Create parent directory if needed
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Move the file/directory
            shutil.move(str(source), str(destination))

            return ToolResult(
                success=True,
                data={"source": str(source), "destination": str(destination), "moved": True},
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
