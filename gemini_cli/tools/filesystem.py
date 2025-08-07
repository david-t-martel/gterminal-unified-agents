"""Filesystem operations using fd and rg for performance."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .base import Tool

logger = logging.getLogger(__name__)


class FilesystemTool(Tool):
    """High-performance filesystem operations using fd and rg."""

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return "File and directory operations using fd and rg"

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute filesystem operation.

        Args:
            params: Operation parameters
                - action: read_file, write_file, list_files, search_files, search_content
                - path: File or directory path
                - content: Content for write operations
                - pattern: Pattern for search operations

        Returns:
            Operation results
        """
        action = params.get("action")
        path = params.get("path")

        if not action:
            return {"error": "Missing 'action' parameter"}

        if not path:
            return {"error": "Missing 'path' parameter"}

        try:
            if action == "read_file":
                return await self._read_file(path)
            elif action == "write_file":
                content = params.get("content", "")
                return await self._write_file(path, content)
            elif action == "list_files":
                return await self._list_files(path, params.get("pattern"))
            elif action == "search_files":
                pattern = params.get("pattern")
                return await self._search_files(path, pattern)
            elif action == "search_content":
                pattern = params.get("pattern")
                return await self._search_content(path, pattern)
            elif action == "analyze_workspace":
                return await self.analyze_workspace(path)
            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"Filesystem operation failed: {e}")
            return {"error": str(e)}

    async def _read_file(self, path: str) -> dict[str, Any]:
        """Read file contents.

        Args:
            path: File path

        Returns:
            File contents and metadata
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                return {"error": f"File not found: {path}"}

            if not file_path.is_file():
                return {"error": f"Not a file: {path}"}

            content = file_path.read_text(encoding="utf-8")

            return {
                "path": path,
                "content": content,
                "size": file_path.stat().st_size,
                "lines": len(content.splitlines()),
            }

        except Exception as e:
            return {"error": f"Failed to read file: {e}"}

    async def _write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write content to file.

        Args:
            path: File path
            content: Content to write

        Returns:
            Write operation results
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            return {
                "path": path,
                "size": len(content),
                "lines": len(content.splitlines()),
                "success": True,
            }

        except Exception as e:
            return {"error": f"Failed to write file: {e}"}

    async def _list_files(
        self, path: str, pattern: str | None = None
    ) -> dict[str, Any]:
        """List files using fd.

        Args:
            path: Directory path
            pattern: Optional file pattern

        Returns:
            List of files
        """
        try:
            cmd = ["fd", "--type", "f"]
            if pattern:
                cmd.extend(["--glob", pattern])
            cmd.append(path)

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return {"error": f"fd failed: {stderr.decode()}"}

            files = [
                line.strip() for line in stdout.decode().splitlines() if line.strip()
            ]

            return {
                "path": path,
                "pattern": pattern,
                "files": files,
                "count": len(files),
            }

        except Exception as e:
            return {"error": f"Failed to list files: {e}"}

    async def _search_files(self, path: str, pattern: str) -> dict[str, Any]:
        """Search for files by name using fd.

        Args:
            path: Directory path
            pattern: File name pattern

        Returns:
            Matching files
        """
        return await self._list_files(path, pattern)

    async def _search_content(self, path: str, pattern: str) -> dict[str, Any]:
        """Search file contents using rg.

        Args:
            path: Directory path
            pattern: Search pattern

        Returns:
            Search results
        """
        try:
            cmd = ["rg", "--json", "--max-count", "20", pattern, path]  # Limit results

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode not in (0, 1):  # 1 is "no matches found"
                return {"error": f"rg failed: {stderr.decode()}"}

            results = []
            for line in stdout.decode().splitlines():
                if line.strip():
                    try:
                        match_data = json.loads(line)
                        if match_data.get("type") == "match":
                            results.append(
                                {
                                    "file": match_data["data"]["path"]["text"],
                                    "line": match_data["data"]["line_number"],
                                    "content": match_data["data"]["lines"][
                                        "text"
                                    ].strip(),
                                    "match": match_data["data"]["submatches"][0][
                                        "match"
                                    ]["text"],
                                }
                            )
                    except (json.JSONDecodeError, KeyError):
                        continue

            return {
                "pattern": pattern,
                "path": path,
                "matches": results,
                "count": len(results),
            }

        except Exception as e:
            return {"error": f"Failed to search content: {e}"}

    async def analyze_workspace(self, path: str | Path) -> dict[str, Any]:
        """Analyze workspace structure.

        Args:
            path: Workspace path

        Returns:
            Workspace analysis
        """
        workspace_path = Path(path)

        if not workspace_path.exists():
            return {"error": f"Path does not exist: {path}"}

        # Get overall file statistics
        file_stats = await self._list_files(str(workspace_path))
        if "error" in file_stats:
            return file_stats

        # Analyze file types
        file_types = {}
        code_files = []
        config_files = []

        for file_path in file_stats["files"]:
            suffix = Path(file_path).suffix.lower()
            file_types[suffix] = file_types.get(suffix, 0) + 1

            # Categorize files
            if suffix in [
                ".py",
                ".js",
                ".ts",
                ".rs",
                ".go",
                ".java",
                ".cpp",
                ".c",
                ".h",
            ]:
                code_files.append(file_path)
            elif suffix in [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"]:
                config_files.append(file_path)

        # Look for common project markers
        project_markers = {}
        marker_files = [
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
            "Makefile",
            "CMakeLists.txt",
        ]

        for marker in marker_files:
            marker_path = workspace_path / marker
            if marker_path.exists():
                project_markers[marker] = str(marker_path)

        return {
            "path": str(workspace_path),
            "total_files": file_stats["count"],
            "file_types": file_types,
            "code_files": len(code_files),
            "config_files": len(config_files),
            "project_markers": project_markers,
            "analysis": self._generate_workspace_summary(file_types, project_markers),
        }

    def _generate_workspace_summary(
        self, file_types: dict[str, int], project_markers: dict[str, str]
    ) -> str:
        """Generate a human-readable workspace summary.

        Args:
            file_types: File type counts
            project_markers: Detected project markers

        Returns:
            Workspace summary text
        """
        summary_parts = []

        # Project type detection
        if "pyproject.toml" in project_markers or ".py" in file_types:
            summary_parts.append("Python project")
        if (
            "package.json" in project_markers
            or ".js" in file_types
            or ".ts" in file_types
        ):
            summary_parts.append("JavaScript/TypeScript project")
        if "Cargo.toml" in project_markers or ".rs" in file_types:
            summary_parts.append("Rust project")
        if "go.mod" in project_markers or ".go" in file_types:
            summary_parts.append("Go project")

        if not summary_parts:
            summary_parts.append("Generic project")

        # File type breakdown
        top_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_types:
            type_summary = ", ".join(f"{count} {ext} files" for ext, count in top_types)
            summary_parts.append(f"Contains: {type_summary}")

        return ". ".join(summary_parts) + "."
