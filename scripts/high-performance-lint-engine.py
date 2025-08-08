#!/usr/bin/env python3
"""High-Performance Lint Engine with Rust PyO3 Optimizations.

Advanced linting system using chunked caches, optimized file operations,
and ast-grep for maximum performance across all auto-linting tools.

Performance Features:
- Chunked file caching with Rust backend
- Parallel lint processing with async workers
- ast-grep integration for fast pattern detection
- Smart error categorization and prioritization
- Memory-efficient streaming for large codebases
"""

import asyncio
from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

# Color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
PURPLE = "\033[95m"
RESET = "\033[0m"

# Try to import Rust extensions for maximum performance
RUST_AVAILABLE = False
try:
    from gterminal_rust_extensions import RustCache
    from gterminal_rust_extensions import RustCommandExecutor
    from gterminal_rust_extensions import RustFileOps
    from gterminal_rust_extensions import RustJsonProcessor
    from gterminal_rust_extensions import init_tracing
    from gterminal_rust_extensions import version as rust_version

    RUST_AVAILABLE = True
    print(f"{GREEN}🚀 Rust extensions v{rust_version()} loaded - Maximum performance mode{RESET}")
    init_tracing("info")
except ImportError as e:
    print(f"{YELLOW}⚠️  Rust extensions not available: {e}{RESET}")

logger = logging.getLogger(__name__)


@dataclass
class LintChunk:
    """Represents a chunk of files for processing."""

    files: list[Path]
    chunk_id: str
    total_size: int
    priority: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            # Generate deterministic chunk ID based on file paths and sizes
            content = "|".join(str(f) for f in sorted(self.files))
            self.chunk_id = hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class LintResult:
    """Result from linting a file or chunk."""

    file_path: Path
    tool: str
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    cache_hit: bool = False
    chunk_id: str | None = None


class ChunkedCache:
    """High-performance chunked cache using Rust backend when available."""

    def __init__(self, capacity: int = 10000, chunk_size: int = 1000):
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.chunks: dict[str, dict[str, Any]] = {}

        if RUST_AVAILABLE:
            try:
                self.rust_cache = RustCache(capacity=capacity, ttl_seconds=3600)
                self.cache_type = "rust"
                logger.info("Initialized Rust-powered chunked cache")
            except Exception as e:
                logger.warning(f"Rust cache initialization failed: {e}")
                self.rust_cache = None
                self.cache_type = "python"
        else:
            self.rust_cache = None
            self.cache_type = "python"

    def _get_chunk_key(self, key: str) -> tuple[str, str]:
        """Get chunk ID and item key for a cache key."""
        chunk_id = hashlib.sha256(key.encode()).hexdigest()[:8]
        return chunk_id, key

    def get(self, key: str) -> Any | None:
        """Get item from cache."""
        if self.rust_cache:
            try:
                return self.rust_cache.get(key)
            except Exception:
                pass

        chunk_id, _ = self._get_chunk_key(key)
        chunk = self.chunks.get(chunk_id, {})
        return chunk.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set item in cache."""
        if self.rust_cache:
            try:
                self.rust_cache.insert(key, value, ttl_seconds=ttl or 3600)
                return True
            except Exception:
                pass

        chunk_id, _ = self._get_chunk_key(key)
        if chunk_id not in self.chunks:
            self.chunks[chunk_id] = {}

        # Implement simple LRU at chunk level
        if len(self.chunks[chunk_id]) >= self.chunk_size:
            # Remove oldest items
            items = list(self.chunks[chunk_id].items())
            self.chunks[chunk_id] = dict(items[self.chunk_size // 2 :])

        self.chunks[chunk_id][key] = value
        return True

    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        if self.rust_cache:
            try:
                return self.rust_cache.contains(key)
            except Exception:
                pass

        chunk_id, _ = self._get_chunk_key(key)
        return key in self.chunks.get(chunk_id, {})

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self.rust_cache:
            try:
                return self.rust_cache.stats()
            except Exception:
                pass

        total_items = sum(len(chunk) for chunk in self.chunks.values())
        return {
            "type": self.cache_type,
            "chunks": len(self.chunks),
            "items": total_items,
            "capacity": self.capacity,
        }


class OptimizedFileProcessor:
    """High-performance file processor with Rust backend."""

    def __init__(self):
        if RUST_AVAILABLE:
            try:
                self.file_ops = RustFileOps()
                self.processor_type = "rust"
                logger.info("Initialized Rust file processor")
            except Exception as e:
                logger.warning(f"Rust file processor failed: {e}")
                self.file_ops = None
                self.processor_type = "python"
        else:
            self.file_ops = None
            self.processor_type = "python"

    async def read_file_chunked(self, file_path: Path, chunk_size: int = 8192) -> list[str]:
        """Read file in chunks for memory efficiency."""
        if self.file_ops:
            try:
                content = await self.file_ops.read_file_async(str(file_path))
                # Split into chunks
                return [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
            except Exception:
                pass

        # Fallback to Python
        chunks = []
        try:
            with file_path.open("r", encoding="utf-8") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
        except Exception as e:
            logger.exception(f"Failed to read {file_path}: {e}")

        return chunks

    async def write_file_atomic(self, file_path: Path, content: str) -> bool:
        """Write file atomically using Rust backend."""
        if self.file_ops:
            try:
                return await self.file_ops.write_file_async(str(file_path), content)
            except Exception:
                pass

        # Fallback to Python atomic write
        try:
            temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            temp_path.write_text(content, encoding="utf-8")
            temp_path.replace(file_path)
            return True
        except Exception as e:
            logger.exception(f"Failed to write {file_path}: {e}")
            return False

    def find_files_fast(self, directory: Path, pattern: str, max_depth: int = 10) -> list[Path]:
        """Fast file finding using Rust backend."""
        if self.file_ops:
            try:
                file_paths = self.file_ops.find_files_by_pattern(
                    str(directory), pattern, max_depth=max_depth
                )
                return [Path(p) for p in file_paths]
            except Exception:
                pass

        # Fallback to Python
        if pattern.startswith("**"):
            pattern = pattern[3:] if pattern.startswith("**/") else pattern[2:]
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))


class ASTGrepAnalyzer:
    """High-performance AST analysis using ast-grep."""

    def __init__(self):
        self.patterns = {
            # Performance patterns
            "inefficient_loops": {
                "pattern": "for $i in range(len($list)): $list[$i]",
                "suggestion": "Use enumerate() or iterate directly",
                "category": "performance",
            },
            "string_concatenation": {
                "pattern": "$str + $other",
                "suggestion": "Use f-strings or join() for multiple concatenations",
                "category": "performance",
            },
            # Security patterns
            "subprocess_shell": {
                "pattern": "subprocess.$func($$$, shell=True)",
                "suggestion": "Avoid shell=True for security",
                "category": "security",
            },
            "eval_usage": {
                "pattern": "eval($expr)",
                "suggestion": "Avoid eval() - use ast.literal_eval() if needed",
                "category": "security",
            },
            # Code quality patterns
            "missing_type_hints": {
                "pattern": "def $func($$$):",
                "suggestion": "Add return type annotation",
                "category": "quality",
            },
            "bare_except": {
                "pattern": "except:",
                "suggestion": "Catch specific exceptions",
                "category": "quality",
            },
            # Modernization patterns
            "dict_keys_iteration": {
                "pattern": "for $key in $dict.keys():",
                "suggestion": "Iterate directly over dict",
                "category": "modernization",
            },
            "format_string": {
                "pattern": '"%s" % $var',
                "suggestion": "Use f-strings",
                "category": "modernization",
            },
        }

    async def analyze_with_pattern(
        self, file_path: Path, pattern_name: str
    ) -> list[dict[str, Any]]:
        """Analyze file with specific ast-grep pattern."""
        if pattern_name not in self.patterns:
            return []

        pattern_config = self.patterns[pattern_name]

        try:
            cmd = [
                "ast-grep",
                "--lang",
                "python",
                "--pattern",
                pattern_config["pattern"],
                "--json",
                str(file_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0 and result.stdout.strip():
                matches = json.loads(result.stdout)
                return [
                    {
                        "pattern": pattern_name,
                        "category": pattern_config["category"],
                        "suggestion": pattern_config["suggestion"],
                        "match": match,
                    }
                    for match in matches
                ]

        except Exception as e:
            logger.warning(f"ast-grep analysis failed for {pattern_name}: {e}")

        return []

    async def analyze_all_patterns(self, file_path: Path) -> dict[str, list[dict[str, Any]]]:
        """Analyze file with all patterns concurrently."""
        tasks = [
            self.analyze_with_pattern(file_path, pattern_name) for pattern_name in self.patterns
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        analysis_results = {}
        for pattern_name, result in zip(self.patterns.keys(), results, strict=False):
            if isinstance(result, Exception):
                logger.warning(f"Pattern {pattern_name} failed: {result}")
                analysis_results[pattern_name] = []
            else:
                analysis_results[pattern_name] = result

        return analysis_results


class HighPerformanceLintEngine:
    """Advanced lint engine with all optimizations."""

    def __init__(self, max_workers: int = 8, cache_size: int = 10000):
        self.max_workers = max_workers
        self.cache = ChunkedCache(capacity=cache_size)
        self.file_processor = OptimizedFileProcessor()
        self.ast_analyzer = ASTGrepAnalyzer()

        # Initialize Rust command executor for tools
        if RUST_AVAILABLE:
            try:
                self.command_executor = RustCommandExecutor(
                    max_processes=max_workers, default_timeout_secs=120
                )
                self.executor_type = "rust"
            except Exception:
                self.command_executor = None
                self.executor_type = "python"
        else:
            self.command_executor = None
            self.executor_type = "python"

        # Tool configurations
        self.tools = {
            "ruff": {
                "cmd": ["uv", "run", "ruff", "check", "--output-format", "json"],
                "parser": self._parse_ruff_output,
                "priority": 1,
            },
            "mypy": {
                "cmd": [
                    "uv",
                    "run",
                    "mypy",
                    "--no-error-summary",
                    "--show-error-codes",
                ],
                "parser": self._parse_mypy_output,
                "priority": 2,
            },
            "bandit": {
                "cmd": ["uv", "run", "bandit", "-f", "json"],
                "parser": self._parse_bandit_output,
                "priority": 3,
            },
            "ast-grep": {
                "cmd": None,  # Special handling
                "parser": self._parse_ast_grep_output,
                "priority": 0,  # Highest priority - fastest
            },
        }

    def create_file_chunks(self, files: list[Path], chunk_size: int = 50) -> list[LintChunk]:
        """Create optimized file chunks for processing."""
        chunks = []
        current_chunk = []
        current_size = 0

        # Sort files by size for better load balancing
        sorted_files = sorted(files, key=lambda f: f.stat().st_size if f.exists() else 0)

        for file_path in sorted_files:
            file_size = file_path.stat().st_size if file_path.exists() else 0

            if (
                len(current_chunk) >= chunk_size or current_size + file_size > 100_000
            ):  # 100KB chunk limit
                if current_chunk:
                    chunk = LintChunk(
                        files=current_chunk.copy(), chunk_id="", total_size=current_size
                    )
                    chunks.append(chunk)
                    current_chunk = []
                    current_size = 0

            current_chunk.append(file_path)
            current_size += file_size

        # Add remaining files
        if current_chunk:
            chunk = LintChunk(files=current_chunk, chunk_id="", total_size=current_size)
            chunks.append(chunk)

        # Assign priorities based on chunk characteristics
        for _i, chunk in enumerate(chunks):
            if any(f.suffix == ".py" and "test" in f.name for f in chunk.files):
                chunk.priority = 1  # Test files higher priority
            elif chunk.total_size < 10_000:  # Small files
                chunk.priority = 2
            else:
                chunk.priority = 3

        return sorted(chunks, key=lambda c: c.priority)

    async def _execute_tool_rust(
        self, cmd: list[str], file_path: Path | None = None
    ) -> dict[str, Any]:
        """Execute tool using Rust command executor."""
        if not self.command_executor:
            raise RuntimeError("Rust executor not available")

        command = cmd[0]
        args = cmd[1:] if len(cmd) > 1 else []

        if file_path:
            args.append(str(file_path))

        return self.command_executor.execute(
            command=command,
            args=args,
            timeout_secs=120,
            capture_output=True,
            check_allowed=True,
        )

    async def _execute_tool_python(
        self, cmd: list[str], file_path: Path | None = None
    ) -> dict[str, Any]:
        """Execute tool using Python subprocess."""
        full_cmd = cmd.copy()
        if file_path:
            full_cmd.append(str(file_path))

        try:
            result = subprocess.run(
                full_cmd, capture_output=True, text=True, check=False, timeout=120
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Command timed out", "returncode": 124}

    def _parse_ruff_output(self, output: str) -> tuple[list[dict], list[dict], list[dict]]:
        """Parse ruff JSON output."""
        errors, warnings, suggestions = [], [], []

        try:
            if not output.strip():
                return errors, warnings, suggestions

            ruff_results = json.loads(output)
            for result in ruff_results:
                severity = result.get("fix", {}).get("applicability", "error")
                item = {
                    "line": result.get("location", {}).get("row", 0),
                    "column": result.get("location", {}).get("column", 0),
                    "code": result.get("code", ""),
                    "message": result.get("message", ""),
                    "rule": result.get("rule", ""),
                    "raw": result,
                }

                if severity == "automatic":
                    suggestions.append(item)
                elif result.get("code", "").startswith(("E", "W")):
                    warnings.append(item)
                else:
                    errors.append(item)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse ruff output: {e}")

        return errors, warnings, suggestions

    def _parse_mypy_output(self, output: str) -> tuple[list[dict], list[dict], list[dict]]:
        """Parse mypy output."""
        errors, warnings, suggestions = [], [], []

        for line in output.splitlines():
            if ": error:" in line or ": note:" in line:
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    item = {
                        "line": int(parts[1]) if parts[1].isdigit() else 0,
                        "column": 0,
                        "code": "mypy",
                        "message": parts[3].strip(),
                        "raw": line,
                    }

                    if ": note:" in line:
                        suggestions.append(item)
                    else:
                        errors.append(item)

        return errors, warnings, suggestions

    def _parse_bandit_output(self, output: str) -> tuple[list[dict], list[dict], list[dict]]:
        """Parse bandit JSON output."""
        errors, warnings, suggestions = [], [], []

        try:
            if not output.strip():
                return errors, warnings, suggestions

            bandit_results = json.loads(output)
            for result in bandit_results.get("results", []):
                severity = result.get("issue_severity", "").lower()
                item = {
                    "line": result.get("line_number", 0),
                    "column": 0,
                    "code": result.get("test_id", ""),
                    "message": result.get("issue_text", ""),
                    "confidence": result.get("issue_confidence", ""),
                    "raw": result,
                }

                if severity in ("high", "medium"):
                    errors.append(item)
                else:
                    warnings.append(item)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse bandit output: {e}")

        return errors, warnings, suggestions

    def _parse_ast_grep_output(
        self, output: dict[str, list[dict]]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Parse ast-grep analysis output."""
        errors, warnings, suggestions = [], [], []

        for pattern_name, matches in output.items():
            for match_data in matches:
                category = match_data.get("category", "quality")
                item = {
                    "line": match_data.get("match", {})
                    .get("range", {})
                    .get("start", {})
                    .get("line", 0),
                    "column": match_data.get("match", {})
                    .get("range", {})
                    .get("start", {})
                    .get("column", 0),
                    "code": f"ast-grep:{pattern_name}",
                    "message": match_data.get("suggestion", ""),
                    "category": category,
                    "raw": match_data,
                }

                if category == "security":
                    errors.append(item)
                elif category == "performance":
                    warnings.append(item)
                else:
                    suggestions.append(item)

        return errors, warnings, suggestions

    async def lint_file(self, file_path: Path, tools: list[str] | None = None) -> LintResult:
        """Lint a single file with specified tools."""
        start_time = time.time()
        tools_to_run = tools or list(self.tools.keys())

        # Check cache first
        cache_key = f"lint:{file_path}:{file_path.stat().st_mtime if file_path.exists() else 0}"
        cached_result = self.cache.get(cache_key)

        if cached_result:
            result = LintResult(**cached_result)
            result.cache_hit = True
            return result

        all_errors = []
        all_warnings = []
        all_suggestions = []

        # Run ast-grep analysis first (fastest)
        if "ast-grep" in tools_to_run:
            try:
                ast_analysis = await self.ast_analyzer.analyze_all_patterns(file_path)
                errors, warnings, suggestions = self._parse_ast_grep_output(ast_analysis)
                all_errors.extend(errors)
                all_warnings.extend(warnings)
                all_suggestions.extend(suggestions)
            except Exception as e:
                logger.warning(f"ast-grep analysis failed for {file_path}: {e}")

        # Run other tools
        for tool_name in tools_to_run:
            if tool_name == "ast-grep":
                continue  # Already handled

            tool_config = self.tools.get(tool_name)
            if not tool_config:
                continue

            try:
                if self.command_executor:
                    result = await self._execute_tool_rust(tool_config["cmd"], file_path)
                    output = result.get("stdout", "")
                else:
                    result = await self._execute_tool_python(tool_config["cmd"], file_path)
                    output = result.get("stdout", "")

                errors, warnings, suggestions = tool_config["parser"](output)
                all_errors.extend(errors)
                all_warnings.extend(warnings)
                all_suggestions.extend(suggestions)

            except Exception as e:
                logger.warning(f"Tool {tool_name} failed for {file_path}: {e}")

        execution_time = time.time() - start_time

        result = LintResult(
            file_path=file_path,
            tool=",".join(tools_to_run),
            errors=all_errors,
            warnings=all_warnings,
            suggestions=all_suggestions,
            execution_time=execution_time,
            cache_hit=False,
        )

        # Cache the result
        self.cache.set(
            cache_key,
            {
                "file_path": str(file_path),
                "tool": result.tool,
                "errors": result.errors,
                "warnings": result.warnings,
                "suggestions": result.suggestions,
                "execution_time": result.execution_time,
            },
        )

        return result

    async def lint_chunk(
        self, chunk: LintChunk, tools: list[str] | None = None
    ) -> list[LintResult]:
        """Lint a chunk of files concurrently."""
        print(f"{CYAN}Processing chunk {chunk.chunk_id} with {len(chunk.files)} files{RESET}")

        # Process files in chunk concurrently
        semaphore = asyncio.Semaphore(self.max_workers)

        async def lint_file_with_semaphore(file_path: Path) -> LintResult:
            async with semaphore:
                result = await self.lint_file(file_path, tools)
                result.chunk_id = chunk.chunk_id
                return result

        tasks = [lint_file_with_semaphore(file_path) for file_path in chunk.files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"File processing failed: {result}")
            else:
                valid_results.append(result)

        return valid_results

    async def lint_directory(
        self,
        directory: Path,
        pattern: str = "**/*.py",
        tools: list[str] | None = None,
        chunk_size: int = 50,
    ) -> list[LintResult]:
        """Lint entire directory with chunked processing."""
        print(f"{GREEN}🚀 Starting high-performance lint analysis{RESET}")
        print(f"{BLUE}Directory: {directory}{RESET}")
        print(f"{BLUE}Pattern: {pattern}{RESET}")
        print(f"{BLUE}Tools: {tools or list(self.tools.keys())}{RESET}")

        # Find files using optimized file processor
        files = self.file_processor.find_files_fast(directory, pattern)

        # Filter out excluded patterns
        excluded = {
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            ".pytest_cache",
            ".mypy_cache",
        }
        files = [f for f in files if not any(ex in str(f) for ex in excluded)]

        print(f"{GREEN}Found {len(files)} files to analyze{RESET}")

        if not files:
            return []

        # Create optimized chunks
        chunks = self.create_file_chunks(files, chunk_size)
        print(f"{CYAN}Created {len(chunks)} optimized chunks{RESET}")

        # Process chunks concurrently
        start_time = time.time()
        chunk_tasks = [self.lint_chunk(chunk, tools) for chunk in chunks]
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        # Flatten results
        all_results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, Exception):
                logger.error(f"Chunk processing failed: {chunk_result}")
            else:
                all_results.extend(chunk_result)

        total_time = time.time() - start_time

        # Print performance summary
        self._print_performance_summary(all_results, total_time)

        return all_results

    def _print_performance_summary(self, results: list[LintResult], total_time: float):
        """Print comprehensive performance summary."""
        if not results:
            return

        total_files = len(results)
        cache_hits = sum(1 for r in results if r.cache_hit)
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        total_suggestions = sum(len(r.suggestions) for r in results)

        avg_time_per_file = total_time / total_files if total_files > 0 else 0
        cache_hit_rate = (cache_hits / total_files * 100) if total_files > 0 else 0

        cache_stats = self.cache.get_stats()

        print(f"\n{GREEN}=== PERFORMANCE SUMMARY ==={RESET}")
        print(f"📁 Files processed: {total_files}")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"🚀 Avg time per file: {avg_time_per_file:.3f}s")
        print(f"💾 Cache hit rate: {cache_hit_rate:.1f}%")
        print(f"🔧 Executor type: {self.executor_type}")
        print(f"📊 File processor: {self.file_processor.processor_type}")

        print(f"\n{BLUE}=== LINT RESULTS ==={RESET}")
        print(f"❌ Errors: {total_errors}")
        print(f"⚠️  Warnings: {total_warnings}")
        print(f"💡 Suggestions: {total_suggestions}")

        print(f"\n{CYAN}=== CACHE STATISTICS ==={RESET}")
        for key, value in cache_stats.items():
            print(f"  {key}: {value}")


async def main():
    """Main entry point for high-performance linting."""
    import argparse

    parser = argparse.ArgumentParser(description="High-Performance Lint Engine")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to lint")
    parser.add_argument("--pattern", default="**/*.py", help="File pattern")
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=["ruff", "mypy", "bandit", "ast-grep"],
        help="Tools to run",
    )
    parser.add_argument("--chunk-size", type=int, default=50, help="Files per chunk")
    parser.add_argument("--workers", type=int, default=8, help="Max concurrent workers")
    parser.add_argument("--cache-size", type=int, default=10000, help="Cache capacity")

    args = parser.parse_args()

    engine = HighPerformanceLintEngine(max_workers=args.workers, cache_size=args.cache_size)

    directory = Path(args.directory)
    if not directory.exists():
        print(f"{RED}Error: Directory {directory} does not exist{RESET}")
        return 1

    results = await engine.lint_directory(
        directory=directory,
        pattern=args.pattern,
        tools=args.tools,
        chunk_size=args.chunk_size,
    )

    if any(r.errors for r in results):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
