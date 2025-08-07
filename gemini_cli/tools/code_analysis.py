"""Code analysis tool for Gemini CLI."""

import asyncio
import logging
from pathlib import Path
from typing import Any

from .base import Tool

logger = logging.getLogger(__name__)


class CodeAnalysisTool(Tool):
    """Code analysis and quality assessment tool."""

    @property
    def name(self) -> str:
        return "code_analysis"

    @property
    def description(self) -> str:
        return "Analyze code quality, structure, and patterns"

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute code analysis operation.

        Args:
            params: Analysis parameters
                - action: analyze_file, analyze_directory, check_syntax, get_metrics
                - path: File or directory path
                - language: Programming language (auto-detected if not specified)

        Returns:
            Analysis results
        """
        action = params.get("action", "analyze_file")
        path = params.get("path")

        if not path:
            return {"error": "Missing 'path' parameter"}

        try:
            if action == "analyze_file":
                return await self._analyze_file(path)
            elif action == "analyze_directory":
                return await self._analyze_directory(path)
            elif action == "check_syntax":
                return await self._check_syntax(path, params.get("language"))
            elif action == "get_metrics":
                return await self._get_metrics(path)
            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_file(self, path: str) -> dict[str, Any]:
        """Analyze a single file.

        Args:
            path: File path

        Returns:
            File analysis results
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                return {"error": f"File not found: {path}"}

            if not file_path.is_file():
                return {"error": f"Not a file: {path}"}

            content = file_path.read_text(encoding="utf-8")
            language = self._detect_language(file_path)

            # Basic metrics
            lines = content.splitlines()
            total_lines = len(lines)
            non_empty_lines = len([line for line in lines if line.strip()])
            comment_lines = self._count_comment_lines(lines, language)

            # Analyze complexity indicators
            complexity = self._analyze_complexity(content, language)

            return {
                "path": path,
                "language": language,
                "metrics": {
                    "total_lines": total_lines,
                    "code_lines": non_empty_lines - comment_lines,
                    "comment_lines": comment_lines,
                    "blank_lines": total_lines - non_empty_lines,
                    "complexity_score": complexity,
                },
                "analysis": self._generate_file_analysis(content, language, complexity),
            }

        except Exception as e:
            return {"error": f"Failed to analyze file: {e}"}

    async def _analyze_directory(self, path: str) -> dict[str, Any]:
        """Analyze code in a directory.

        Args:
            path: Directory path

        Returns:
            Directory analysis results
        """
        try:
            # Find code files
            cmd = [
                "fd",
                "--type",
                "f",
                "--extension",
                "py",
                "--extension",
                "js",
                "--extension",
                "ts",
                "--extension",
                "rs",
                "--extension",
                "go",
                "--extension",
                "java",
                "--extension",
                "cpp",
                "--extension",
                "c",
                "--extension",
                "h",
                path,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return {"error": f"Failed to find code files: {stderr.decode()}"}

            files = [
                line.strip() for line in stdout.decode().splitlines() if line.strip()
            ]

            if not files:
                return {"error": "No code files found"}

            # Analyze each file
            file_analyses = []
            total_metrics = {
                "total_lines": 0,
                "code_lines": 0,
                "comment_lines": 0,
                "blank_lines": 0,
                "files": 0,
            }

            language_stats = {}

            for file_path in files[:20]:  # Limit to prevent overwhelming results
                analysis = await self._analyze_file(file_path)
                if "error" not in analysis:
                    file_analyses.append(
                        {
                            "path": file_path,
                            "language": analysis["language"],
                            "lines": analysis["metrics"]["total_lines"],
                            "complexity": analysis["metrics"]["complexity_score"],
                        }
                    )

                    # Aggregate metrics
                    metrics = analysis["metrics"]
                    total_metrics["total_lines"] += metrics["total_lines"]
                    total_metrics["code_lines"] += metrics["code_lines"]
                    total_metrics["comment_lines"] += metrics["comment_lines"]
                    total_metrics["blank_lines"] += metrics["blank_lines"]
                    total_metrics["files"] += 1

                    # Language statistics
                    lang = analysis["language"]
                    if lang not in language_stats:
                        language_stats[lang] = {"files": 0, "lines": 0}
                    language_stats[lang]["files"] += 1
                    language_stats[lang]["lines"] += metrics["total_lines"]

            return {
                "path": path,
                "total_files": len(files),
                "analyzed_files": len(file_analyses),
                "metrics": total_metrics,
                "language_stats": language_stats,
                "file_details": file_analyses,
                "summary": self._generate_directory_summary(
                    total_metrics, language_stats
                ),
            }

        except Exception as e:
            return {"error": f"Failed to analyze directory: {e}"}

    async def _check_syntax(
        self, path: str, language: str | None = None
    ) -> dict[str, Any]:
        """Check syntax of a file.

        Args:
            path: File path
            language: Programming language (auto-detected if None)

        Returns:
            Syntax check results
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                return {"error": f"File not found: {path}"}

            if not language:
                language = self._detect_language(file_path)

            # Simple syntax checks based on language
            if language == "python":
                return await self._check_python_syntax(path)
            elif language in ["javascript", "typescript"]:
                return await self._check_js_syntax(path)
            else:
                return {"message": f"Syntax checking not implemented for {language}"}

        except Exception as e:
            return {"error": f"Syntax check failed: {e}"}

    async def _check_python_syntax(self, path: str) -> dict[str, Any]:
        """Check Python syntax using python -m py_compile.

        Args:
            path: Python file path

        Returns:
            Syntax check results
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3",
                "-m",
                "py_compile",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            return {
                "path": path,
                "language": "python",
                "valid": proc.returncode == 0,
                "errors": stderr.decode() if stderr else None,
            }

        except Exception as e:
            return {"error": f"Python syntax check failed: {e}"}

    async def _check_js_syntax(self, path: str) -> dict[str, Any]:
        """Check JavaScript/TypeScript syntax using node.

        Args:
            path: JS/TS file path

        Returns:
            Syntax check results
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "node",
                "--check",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            return {
                "path": path,
                "language": "javascript",
                "valid": proc.returncode == 0,
                "errors": stderr.decode() if stderr else None,
            }

        except Exception as e:
            return {"error": f"JavaScript syntax check failed: {e}"}

    async def _get_metrics(self, path: str) -> dict[str, Any]:
        """Get detailed code metrics.

        Args:
            path: File or directory path

        Returns:
            Code metrics
        """
        path_obj = Path(path)

        if path_obj.is_file():
            return await self._analyze_file(path)
        elif path_obj.is_dir():
            return await self._analyze_directory(path)
        else:
            return {"error": f"Invalid path: {path}"}

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Detected language name
        """
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".sh": "shell",
            ".bash": "shell",
            ".zsh": "shell",
        }

        return extension_map.get(file_path.suffix.lower(), "unknown")

    def _count_comment_lines(self, lines: list[str], language: str) -> int:
        """Count comment lines based on language.

        Args:
            lines: File lines
            language: Programming language

        Returns:
            Number of comment lines
        """
        comment_prefixes = {
            "python": ["#"],
            "javascript": ["//", "/*", "*", "*/"],
            "typescript": ["//", "/*", "*", "*/"],
            "rust": ["//", "/*", "*", "*/"],
            "go": ["//", "/*", "*", "*/"],
            "java": ["//", "/*", "*", "*/"],
            "cpp": ["//", "/*", "*", "*/"],
            "c": ["//", "/*", "*", "*/"],
            "shell": ["#"],
        }

        prefixes = comment_prefixes.get(language, [])
        if not prefixes:
            return 0

        count = 0
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(prefix) for prefix in prefixes):
                count += 1

        return count

    def _analyze_complexity(self, content: str, language: str) -> int:
        """Analyze code complexity (simplified).

        Args:
            content: File content
            language: Programming language

        Returns:
            Complexity score (higher = more complex)
        """
        complexity = 0

        # Count control flow keywords
        control_keywords = {
            "python": [
                "if",
                "elif",
                "else",
                "for",
                "while",
                "try",
                "except",
                "finally",
                "with",
            ],
            "javascript": [
                "if",
                "else",
                "for",
                "while",
                "switch",
                "case",
                "try",
                "catch",
                "finally",
            ],
            "typescript": [
                "if",
                "else",
                "for",
                "while",
                "switch",
                "case",
                "try",
                "catch",
                "finally",
            ],
            "rust": [
                "if",
                "else",
                "for",
                "while",
                "loop",
                "match",
                "if let",
                "while let",
            ],
            "go": ["if", "else", "for", "switch", "case", "select"],
            "java": [
                "if",
                "else",
                "for",
                "while",
                "switch",
                "case",
                "try",
                "catch",
                "finally",
            ],
        }

        keywords = control_keywords.get(language, [])
        for keyword in keywords:
            complexity += content.count(f" {keyword} ")
            complexity += content.count(f"{keyword} ")

        # Count function definitions (simplified)
        if language == "python":
            complexity += content.count("def ")
            complexity += content.count("class ")
        elif language in ["javascript", "typescript"]:
            complexity += content.count("function ")
            complexity += content.count("class ")

        return min(complexity, 100)  # Cap at 100

    def _generate_file_analysis(
        self, content: str, language: str, complexity: int
    ) -> str:
        """Generate human-readable file analysis.

        Args:
            content: File content
            language: Programming language
            complexity: Complexity score

        Returns:
            Analysis summary
        """
        lines = len(content.splitlines())

        size_category = "small" if lines < 50 else "medium" if lines < 200 else "large"
        complexity_level = (
            "low" if complexity < 10 else "medium" if complexity < 25 else "high"
        )

        return (
            f"{language.title()} file with {lines} lines. "
            f"Size: {size_category}, Complexity: {complexity_level}."
        )

    def _generate_directory_summary(
        self, metrics: dict[str, int], language_stats: dict[str, dict[str, int]]
    ) -> str:
        """Generate directory analysis summary.

        Args:
            metrics: Aggregated metrics
            language_stats: Per-language statistics

        Returns:
            Directory summary
        """
        total_files = metrics["files"]
        total_lines = metrics["total_lines"]

        # Find dominant language
        dominant_lang = (
            max(language_stats.items(), key=lambda x: x[1]["lines"])[0]
            if language_stats
            else "unknown"
        )

        return (
            f"Analyzed {total_files} files with {total_lines} total lines. "
            f"Primary language: {dominant_lang.title()}. "
            f"Code density: {metrics['code_lines'] / max(total_lines, 1) * 100:.1f}%."
        )
