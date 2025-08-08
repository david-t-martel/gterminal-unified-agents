#!/usr/bin/env python3
"""Rust Cache Performance Test Framework.

This module provides comprehensive testing for PyO3 compiled Rust functions
with cache functionality for maximum efficiency.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

# Color codes
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Rust extensions with chunked cache
RUST_CACHE_AVAILABLE = False
try:
    from gterminal_rust_extensions import RustCache
    from gterminal_rust_extensions import RustCommandExecutor
    from gterminal_rust_extensions import RustFileOps
    from gterminal_rust_extensions import RustJsonProcessor
    from gterminal_rust_extensions import init_tracing
    from gterminal_rust_extensions import version as rust_version

    RUST_CACHE_AVAILABLE = True
    print(f"{GREEN}🚀 Rust PyO3 cache extensions loaded (v{rust_version()}){RESET}")
    init_tracing("info")
except ImportError as e:
    print(f"{YELLOW}⚠️  Rust cache extensions not available: {e}{RESET}")
    print(f"{YELLOW}   Using fallback implementations{RESET}")


class ChunkedCacheManager:
    """High-performance chunked cache manager using Rust PyO3."""

    def __init__(self, cache_capacity: int = 50000, chunk_size: int = 8192):
        self.cache_capacity = cache_capacity
        self.chunk_size = chunk_size

        if RUST_CACHE_AVAILABLE:
            # Use Rust chunked cache for maximum performance
            self.file_cache = RustCache(
                capacity=cache_capacity,
                default_ttl_secs=3600,  # 1 hour TTL
                max_memory_bytes=512 * 1024 * 1024,  # 512MB max
                cleanup_interval_secs=300,  # 5 minutes
            )
            self.analysis_cache = RustCache(
                capacity=cache_capacity // 2,
                default_ttl_secs=1800,  # 30 minutes TTL
                max_memory_bytes=256 * 1024 * 1024,  # 256MB max
                cleanup_interval_secs=600,  # 10 minutes
            )
            self.file_ops = RustFileOps(
                max_file_size=100 * 1024 * 1024,  # 100MB max file size
                parallel_threshold=10,
            )
            print(f"{CYAN}✅ Initialized Rust chunked cache system{RESET}")
        else:
            # Fallback to basic dictionary cache
            self.file_cache = {}
            self.analysis_cache = {}
            self.file_ops = None
            print(f"{YELLOW}⚠️  Using fallback cache system{RESET}")

    async def get_file_content_chunked(self, file_path: str) -> str | None:
        """Get file content using chunked cache for large files."""
        if not Path(file_path).exists():
            return None

        # Create cache key with file modification time
        try:
            stat = Path(file_path).stat()
            cache_key = f"file:{file_path}:{stat.st_mtime}:{stat.st_size}"
        except OSError:
            return None

        if RUST_CACHE_AVAILABLE:
            # Try to get from Rust cache first
            cached_content = self.file_cache.get(cache_key)
            if cached_content is not None:
                return cached_content

            # Read file using Rust file ops with chunking
            try:
                content = await self.file_ops.read_file_async(
                    file_path, encoding="utf-8", chunk_size=self.chunk_size
                )
                if content:
                    # Store in cache with TTL
                    self.file_cache.set(cache_key, content, ttl_secs=3600)
                return content
            except Exception as e:
                logger.warning(f"Rust file read failed: {e}")

        # Fallback to standard file operations
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            if isinstance(self.file_cache, dict):
                self.file_cache[cache_key] = content
            return content
        except Exception as e:
            logger.exception(f"File read failed: {e}")
            return None

    async def cache_analysis_result(self, key: str, result: Any, ttl: int = 1800) -> bool:
        """Cache analysis results with chunked storage."""
        if RUST_CACHE_AVAILABLE:
            try:
                return self.analysis_cache.set(key, result, ttl_secs=ttl)
            except Exception as e:
                logger.warning(f"Rust cache storage failed: {e}")

        # Fallback storage
        if isinstance(self.analysis_cache, dict):
            self.analysis_cache[key] = result
        return True

    def get_cached_analysis(self, key: str) -> Any | None:
        """Get cached analysis result."""
        if RUST_CACHE_AVAILABLE:
            try:
                return self.analysis_cache.get(key)
            except Exception as e:
                logger.warning(f"Rust cache retrieval failed: {e}")

        # Fallback retrieval
        if isinstance(self.analysis_cache, dict):
            return self.analysis_cache.get(key)
        return None

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        if RUST_CACHE_AVAILABLE:
            try:
                file_stats = self.file_cache.stats()
                analysis_stats = self.analysis_cache.stats()
                return {
                    "rust_enabled": True,
                    "file_cache": file_stats,
                    "analysis_cache": analysis_stats,
                    "chunk_size": self.chunk_size,
                    "cache_capacity": self.cache_capacity,
                }
            except Exception as e:
                logger.warning(f"Cache stats retrieval failed: {e}")

        return {
            "rust_enabled": False,
            "file_cache_entries": (
                len(self.file_cache) if isinstance(self.file_cache, dict) else 0
            ),
            "analysis_cache_entries": (
                len(self.analysis_cache) if isinstance(self.analysis_cache, dict) else 0
            ),
            "fallback_mode": True,
        }


class OptimizedToolRunner:
    """Ultra-high performance tool runner using Rust PyO3 and chunked cache."""

    def __init__(self, cache_manager: ChunkedCacheManager):
        self.cache = cache_manager

        if RUST_CACHE_AVAILABLE:
            self.command_executor = RustCommandExecutor(
                max_processes=20, default_timeout_secs=300, rate_limit_per_minute=500
            )
            self.json_processor = RustJsonProcessor()
            print(f"{CYAN}✅ Initialized Rust command executor and JSON processor{RESET}")
        else:
            self.command_executor = None
            self.json_processor = None

    async def run_tool_with_cache(
        self, tool_name: str, args: list[str], file_path: str
    ) -> dict[str, Any]:
        """Run tool with intelligent caching."""
        # Create cache key based on file content hash
        content = await self.cache.get_file_content_chunked(file_path)
        if not content:
            return {"error": f"Could not read file: {file_path}"}

        # Simple content hash for cache key
        content_hash = hash(content) & 0x7FFFFFFFFFFFFFFF  # Ensure positive
        cache_key = f"tool:{tool_name}:{file_path}:{content_hash}"

        # Check cache first
        cached_result = self.cache.get_cached_analysis(cache_key)
        if cached_result is not None:
            return {
                "cached": True,
                "result": cached_result,
                "tool": tool_name,
                "file": file_path,
            }

        # Run tool with Rust executor if available
        cmd = [tool_name, *args, file_path]

        if RUST_CACHE_AVAILABLE and self.command_executor:
            try:
                result = await self._run_with_rust_executor(cmd)
            except Exception as e:
                logger.warning(f"Rust executor failed: {e}, falling back to subprocess")
                result = await self._run_with_subprocess(cmd)
        else:
            result = await self._run_with_subprocess(cmd)

        # Cache the result
        await self.cache.cache_analysis_result(cache_key, result, ttl=1800)

        return {"cached": False, "result": result, "tool": tool_name, "file": file_path}

    async def _run_with_rust_executor(self, cmd: list[str]) -> dict[str, Any]:
        """Run command using Rust executor."""
        command = cmd[0]
        args = cmd[1:] if len(cmd) > 1 else None

        result = self.command_executor.execute(
            command=command,
            args=args,
            timeout_secs=300,
            capture_output=True,
            check_allowed=True,
        )

        return {
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "returncode": result.get("exit_code", 0),
            "executor": "rust",
        }

    async def _run_with_subprocess(self, cmd: list[str]) -> dict[str, Any]:
        """Run command using subprocess as fallback."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "returncode": process.returncode,
                "executor": "subprocess",
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": 1,
                "executor": "subprocess",
            }

    async def run_ast_grep_analysis(self, pattern: str, file_path: str) -> dict[str, Any]:
        """Run ast-grep analysis with caching."""
        cache_key = f"astgrep:{pattern}:{file_path}"

        # Check cache first
        cached_result = self.cache.get_cached_analysis(cache_key)
        if cached_result is not None:
            return {"cached": True, "result": cached_result}

        # Run ast-grep
        cmd = [
            "ast-grep",
            "--lang",
            "python",
            "--pattern",
            pattern,
            "--json",
            file_path,
        ]
        result = await self.run_tool_with_cache("ast-grep", cmd[1:], "")

        # Parse JSON result if available
        if RUST_CACHE_AVAILABLE and self.json_processor and result.get("result", {}).get("stdout"):
            try:
                parsed_result = self.json_processor.parse_json(result["result"]["stdout"])
                result["parsed"] = parsed_result
            except Exception as e:
                logger.warning(f"JSON parsing failed: {e}")

        # Cache the result
        await self.cache.cache_analysis_result(cache_key, result, ttl=3600)

        return result


class HighPerformanceLinter:
    """High-performance linting system using PyO3 chunked cache."""

    def __init__(self):
        self.cache_manager = ChunkedCacheManager(
            cache_capacity=100000,  # Large cache for better hit rates
            chunk_size=16384,  # 16KB chunks for optimal performance
        )
        self.tool_runner = OptimizedToolRunner(self.cache_manager)

        # Linting tool configurations
        self.tools = {
            "ruff": {
                "command": "ruff",
                "args": ["check", "--output-format", "json"],
                "output_format": "json",
            },
            "mypy": {
                "command": "mypy",
                "args": ["--no-error-summary", "--show-error-codes"],
                "output_format": "text",
            },
            "flake8": {
                "command": "flake8",
                "args": ["--format=json"],
                "output_format": "json",
            },
            "pylint": {
                "command": "pylint",
                "args": ["--output-format=json"],
                "output_format": "json",
            },
        }

        # AST-grep patterns for code analysis
        self.ast_patterns = {
            "missing_return_types": "def $func($$$):",
            "bare_except": "except:",
            "old_string_format": '"%s" % $var',
            "missing_docstrings": "def $func($$$): $body",
            "untyped_variables": "$var = $value",
        }

    async def analyze_file(self, file_path: str, tools: list[str] | None = None) -> dict[str, Any]:
        """Analyze file with selected tools using chunked cache."""
        if not Path(file_path).exists():
            return {"error": f"File not found: {file_path}"}

        selected_tools = tools or ["ruff", "mypy"]
        results = {}

        print(f"{BLUE}🔍 Analyzing {file_path} with {len(selected_tools)} tools...{RESET}")

        # Run tools concurrently
        tasks = []
        for tool_name in selected_tools:
            if tool_name in self.tools:
                tool_config = self.tools[tool_name]
                task = self.tool_runner.run_tool_with_cache(
                    tool_config["command"], tool_config["args"], file_path
                )
                tasks.append((tool_name, task))

        # Run AST-grep patterns concurrently
        ast_tasks = []
        for pattern_name, pattern in self.ast_patterns.items():
            task = self.tool_runner.run_ast_grep_analysis(pattern, file_path)
            ast_tasks.append((pattern_name, task))

        # Execute all tasks concurrently
        all_tasks = [task for _, task in tasks] + [task for _, task in ast_tasks]
        tool_results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Process tool results
        for i, (tool_name, _) in enumerate(tasks):
            result = tool_results[i]
            if isinstance(result, Exception):
                results[tool_name] = {"error": str(result)}
            else:
                results[tool_name] = result

        # Process AST-grep results
        ast_results = {}
        for i, (pattern_name, _) in enumerate(ast_tasks):
            result = tool_results[len(tasks) + i]
            if isinstance(result, Exception):
                ast_results[pattern_name] = {"error": str(result)}
            else:
                ast_results[pattern_name] = result

        results["ast_analysis"] = ast_results

        # Add cache statistics
        results["cache_stats"] = self.cache_manager.get_cache_stats()

        return results

    async def analyze_directory(
        self, directory: str, pattern: str = "**/*.py", tools: list[str] | None = None
    ) -> dict[str, Any]:
        """Analyze directory with chunked cache optimization."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory}"}

        # Find Python files using Rust file ops if available
        if RUST_CACHE_AVAILABLE and self.cache_manager.file_ops:
            try:
                python_files = self.cache_manager.file_ops.find_files_by_pattern(
                    str(dir_path), pattern, max_depth=10
                )
            except Exception as e:
                logger.warning(f"Rust file discovery failed: {e}")
                python_files = [str(p) for p in dir_path.rglob(pattern.replace("**/", ""))]
        else:
            python_files = [str(p) for p in dir_path.rglob(pattern.replace("**/", ""))]

        print(f"{CYAN}📁 Analyzing {len(python_files)} files in {directory}...{RESET}")

        # Analyze files concurrently with chunked processing
        chunk_size = 10  # Process files in chunks to manage memory
        all_results = {}

        for i in range(0, len(python_files), chunk_size):
            chunk = python_files[i : i + chunk_size]
            chunk_tasks = [self.analyze_file(file_path, tools) for file_path in chunk]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

            for j, file_path in enumerate(chunk):
                result = chunk_results[j]
                if isinstance(result, Exception):
                    all_results[file_path] = {"error": str(result)}
                else:
                    all_results[file_path] = result

            # Show progress
            processed = min(i + chunk_size, len(python_files))
            print(f"  📊 Processed {processed}/{len(python_files)} files")

        # Aggregate statistics
        total_issues = 0
        cache_hits = 0
        cache_misses = 0

        for file_result in all_results.values():
            if "error" not in file_result:
                for tool_result in file_result.values():
                    if isinstance(tool_result, dict):
                        if tool_result.get("cached"):
                            cache_hits += 1
                        else:
                            cache_misses += 1

                        # Count issues (simplified)
                        if "result" in tool_result and "stdout" in tool_result["result"]:
                            output = tool_result["result"]["stdout"]
                            if output and output.strip():
                                total_issues += len(output.splitlines())

        return {
            "files_analyzed": len(python_files),
            "total_issues": total_issues,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_ratio": (
                cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
            ),
            "results": all_results,
            "final_cache_stats": self.cache_manager.get_cache_stats(),
        }


async def main():
    """Main demonstration of PyO3 chunked cache performance."""
    print(f"{GREEN}🚀 ADVANCED PyO3 CHUNKED CACHE PERFORMANCE DEMO{RESET}")

    linter = HighPerformanceLinter()

    # Test single file analysis
    test_file = "scripts/claude-auto-fix.py"
    if Path(test_file).exists():
        print(f"\n{CYAN}=== SINGLE FILE ANALYSIS ==={RESET}")
        result = await linter.analyze_file(test_file, ["ruff", "mypy"])

        # Show cache performance
        cache_stats = result.get("cache_stats", {})
        print("📊 Cache Performance:")
        if cache_stats.get("rust_enabled"):
            print("  ✅ Rust PyO3 cache active")
            file_cache = cache_stats.get("file_cache", {})
            print(
                f"  📁 File cache: {file_cache.get('hits', 0)} hits, {file_cache.get('misses', 0)} misses"
            )
            analysis_cache = cache_stats.get("analysis_cache", {})
            print(
                f"  🔍 Analysis cache: {analysis_cache.get('hits', 0)} hits, {analysis_cache.get('misses', 0)} misses"
            )
        else:
            print(f"  ⚠️  Fallback mode: {cache_stats.get('file_cache_entries', 0)} file entries")

    # Test directory analysis with chunked processing
    print(f"\n{CYAN}=== DIRECTORY ANALYSIS WITH CHUNKED CACHE ==={RESET}")
    dir_result = await linter.analyze_directory("scripts/", tools=["ruff"])

    print("📈 Analysis Summary:")
    print(f"  Files: {dir_result.get('files_analyzed', 0)}")
    print(f"  Issues: {dir_result.get('total_issues', 0)}")
    print(f"  Cache hits: {dir_result.get('cache_hits', 0)}")
    print(f"  Cache misses: {dir_result.get('cache_misses', 0)}")
    print(f"  Cache hit ratio: {dir_result.get('cache_hit_ratio', 0):.2%}")

    final_stats = dir_result.get("final_cache_stats", {})
    print(f"\n{GREEN}🎯 Final Cache Statistics:{RESET}")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
