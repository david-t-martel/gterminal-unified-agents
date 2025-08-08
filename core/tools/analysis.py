#!/usr/bin/env python3
"""Code Analysis Tools - Tools for analyzing and understanding code."""

import ast
import asyncio
import json
import logging
from pathlib import Path
import time
from typing import Any

try:
    import tomllib
except ImportError:
    # Fallback for Python < 3.11
    import toml as tomllib

from gterminal.core.tools.registry import BaseTool
from gterminal.core.tools.registry import ToolParameter
from gterminal.core.tools.registry import ToolResult

logger = logging.getLogger(__name__)


class AnalyzeCodeTool(BaseTool):
    """Tool for analyzing code structure and complexity."""

    def __init__(self) -> None:
        super().__init__(
            name="analyze_code",
            description="Analyze code structure, complexity, and quality",
            category="analysis",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to file or directory to analyze",
                required=True,
            ),
            ToolParameter(
                name="metrics",
                type="list",
                description="Metrics to calculate (complexity, lines, functions)",
                required=False,
                default=["complexity", "lines", "functions"],
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            path = Path(params["path"])
            metrics = params.get("metrics", ["complexity", "lines", "functions"])

            if not path.exists():
                return ToolResult(success=False, data=None, error=f"Path not found: {path}")

            results: dict[str, Any] = {}

            if path.is_file():
                results[str(path)] = await self._analyze_file(path, metrics)
            else:
                # Analyze all Python files in directory
                for py_file in path.rglob("*.py"):
                    results[str(py_file)] = await self._analyze_file(py_file, metrics)

            # Calculate summary
            summary = {
                "total_files": len(results),
                "total_lines": sum(r.get("lines", 0) for r in results.values()),
                "total_functions": sum(r.get("functions", 0) for r in results.values()),
                "average_complexity": (
                    sum(r.get("complexity", 0) for r in results.values()) / len(results)
                    if results
                    else 0
                ),
            }

            return ToolResult(
                success=True,
                data={"files": results, "summary": summary},
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    async def _analyze_file(self, file_path: Path, metrics: list[str]) -> dict[str, Any]:
        """Analyze a single Python file."""
        result: dict[str, Any] = {}

        try:
            with file_path.open(encoding="utf-8") as f:
                content = f.read()

            # Line count
            if "lines" in metrics:
                lines = content.splitlines()
                result["lines"] = len(lines)
                result["non_empty_lines"] = len([line for line in lines if line.strip()])

            # Parse AST for detailed analysis
            try:
                tree = ast.parse(content)

                if "functions" in metrics:
                    functions = [
                        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    ]
                    result["functions"] = len(functions)
                    result["function_names"] = [f.name for f in functions]

                if "classes" in metrics:
                    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                    result["classes"] = len(classes)
                    result["class_names"] = [c.name for c in classes]

                if "complexity" in metrics:
                    # Simple cyclomatic complexity approximation
                    complexity = 1  # Base complexity
                    for node in ast.walk(tree):
                        if isinstance(node, ast.If | ast.While | ast.For | ast.ExceptHandler):
                            complexity += 1
                    result["complexity"] = complexity

            except SyntaxError as e:
                result["parse_error"] = str(e)

        except Exception as e:
            result["error"] = str(e)

        return result


class FindDependenciesTool(BaseTool):
    """Tool for finding project dependencies."""

    def __init__(self) -> None:
        super().__init__(
            name="find_dependencies",
            description="Find and analyze project dependencies",
            category="analysis",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Project root path",
                required=True,
            ),
            ToolParameter(
                name="include_dev",
                type="boolean",
                description="Include dev dependencies",
                required=False,
                default=True,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            path = Path(params["path"])
            include_dev = params.get("include_dev", True)

            if not path.exists():
                return ToolResult(success=False, data=None, error=f"Path not found: {path}")

            dependencies = {
                "python": {},
                "javascript": {},
                "rust": {},
            }

            # Check for Python dependencies
            pyproject_path = path / "pyproject.toml"
            requirements_path = path / "requirements.txt"

            if pyproject_path.exists():
                dependencies["python"] = await self._parse_pyproject(pyproject_path, include_dev)
            elif requirements_path.exists():
                dependencies["python"] = await self._parse_requirements(requirements_path)

            # Check for JavaScript dependencies
            package_json_path = path / "package.json"
            if package_json_path.exists():
                dependencies["javascript"] = await self._parse_package_json(
                    package_json_path, include_dev
                )

            # Check for Rust dependencies
            cargo_toml_path = path / "Cargo.toml"
            if cargo_toml_path.exists():
                dependencies["rust"] = await self._parse_cargo_toml(cargo_toml_path, include_dev)

            # Count total dependencies
            total = sum(
                len(deps.get("dependencies", {})) + len(deps.get("dev_dependencies", {}))
                for deps in dependencies.values()
            )

            return ToolResult(
                success=True,
                data={"dependencies": dependencies, "total_count": total},
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    async def _parse_pyproject(self, path: Path, include_dev: bool) -> dict[str, Any]:
        """Parse pyproject.toml for dependencies."""
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)

            result: dict[str, Any] = {}
            if "project" in data and "dependencies" in data["project"]:
                result["dependencies"] = data["project"]["dependencies"]

            if include_dev and "project" in data and "optional-dependencies" in data["project"]:
                result["dev_dependencies"] = data["project"]["optional-dependencies"]

            return result
        except Exception:
            return {}

    async def _parse_requirements(self, path: Path) -> dict[str, Any]:
        """Parse requirements.txt for dependencies."""
        try:
            with path.open() as f:
                lines = f.readlines()

            deps: list[Any] = []
            for orig_line in lines:
                line = orig_line.strip()
                if line and not line.startswith("#"):
                    deps.append(line)

            return {"dependencies": deps}
        except Exception:
            return {}

    async def _parse_package_json(self, path: Path, include_dev: bool) -> dict[str, Any]:
        """Parse package.json for dependencies."""
        try:
            with path.open() as f:
                data = json.load(f)

            result: dict[str, Any] = {}
            if "dependencies" in data:
                result["dependencies"] = data["dependencies"]

            if include_dev and "devDependencies" in data:
                result["dev_dependencies"] = data["devDependencies"]

            return result
        except Exception:
            return {}

    async def _parse_cargo_toml(self, path: Path, include_dev: bool) -> dict[str, Any]:
        """Parse Cargo.toml for dependencies."""
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)

            result: dict[str, Any] = {}
            if "dependencies" in data:
                result["dependencies"] = data["dependencies"]

            if include_dev and "dev-dependencies" in data:
                result["dev_dependencies"] = data["dev-dependencies"]

            return result
        except Exception:
            return {}


class ProfilePerformanceTool(BaseTool):
    """Tool for profiling code performance."""

    def __init__(self) -> None:
        super().__init__(
            name="profile_performance",
            description="Profile code performance and identify bottlenecks",
            category="analysis",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="Command to profile",
                required=True,
            ),
            ToolParameter(
                name="duration",
                type="integer",
                description="Duration in seconds",
                required=False,
                default=10,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            command = params["command"]
            duration = params.get("duration", 10)

            # Use py-spy for profiling if available
            profile_command = f"py-spy record -d {duration} -o profile.svg -- {command}"

            process = await asyncio.create_subprocess_shell(
                profile_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return ToolResult(
                    success=True,
                    data={
                        "profile_output": "profile.svg",
                        "command": command,
                        "duration": duration,
                    },
                )
            # Fallback to basic timing
            start = time.time()
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            elapsed = time.time() - start

            return ToolResult(
                success=True,
                data={
                    "execution_time": elapsed,
                    "command": command,
                    "stdout": stdout.decode("utf-8")[:1000] if stdout else "",
                },
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class LintCodeTool(BaseTool):
    """Tool for linting code."""

    def __init__(self) -> None:
        super().__init__(
            name="lint_code",
            description="Lint code using ruff and other tools",
            category="analysis",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to lint",
                required=True,
            ),
            ToolParameter(
                name="fix",
                type="boolean",
                description="Auto-fix issues",
                required=False,
                default=False,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            path = params["path"]
            fix = params.get("fix", False)

            # Run ruff
            command = f"uv run ruff check {path}"
            if fix:
                command += " --fix"

            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Parse output for issues
            issues: list[Any] = []
            if stdout:
                for line in stdout.decode("utf-8").splitlines():
                    if ":" in line and any(level in line for level in ["ERROR", "WARNING", "INFO"]):
                        issues.append(line.strip())

            return ToolResult(
                success=process.returncode == 0,
                data={
                    "path": path,
                    "issues_found": len(issues),
                    "issues": issues[:20],  # Limit to first 20 issues
                    "fixed": fix and process.returncode == 0,
                },
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
