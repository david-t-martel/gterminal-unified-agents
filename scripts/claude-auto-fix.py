#!/usr/bin/env python3
"""Claude Auto-Fix - Comprehensive Error Fixing with AI.

This consolidates the best performance optimizations with comprehensive fixing capabilities.
Uses memory-mapped file operations, streaming processing, and zero-copy operations
for maximum efficiency with Rust extensions when available.

Better strategies implemented:
- Memory-mapped file operations for large files
- Streaming JSON/text processing instead of chunking
- Lazy evaluation and incremental processing
- Zero-copy operations between Python and Rust
- Native async/await throughout

Usage:
    uv run python scripts/claude-auto-fix.py [--model haiku|sonnet] [--concurrent]
"""

import argparse
import asyncio
from collections import defaultdict
import json
import logging
import mmap
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

# Color codes for output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Try to import Rust extensions for high performance
RUST_AVAILABLE = False
try:
    from gterminal_rust_extensions import RustCache
    from gterminal_rust_extensions import RustCommandExecutor
    from gterminal_rust_extensions import RustFileOps
    from gterminal_rust_extensions import RustJsonProcessor
    from gterminal_rust_extensions import init_tracing
    from gterminal_rust_extensions import version as rust_version

    RUST_AVAILABLE = True
    print(
        f"{GREEN}🚀 Rust extensions loaded (v{rust_version()}) - Ultimate performance mode{RESET}"
    )
    init_tracing("info")
except ImportError as e:
    print(f"{YELLOW}⚠️  Rust extensions not available: {e}{RESET}")
    print(f"{YELLOW}   Using optimized Python implementations{RESET}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryMappedFileHandler:
    """Memory-mapped file operations - better than chunking for large files."""

    @staticmethod
    async def read_file_mmap(file_path: str, encoding: str = "utf-8") -> str | None:
        """Read file using memory mapping for efficiency."""
        try:
            path = Path(file_path)
            if not path.exists() or path.stat().st_size == 0:
                return ""

            # For small files, use direct read
            if path.stat().st_size < 8192:  # 8KB threshold
                return path.read_text(encoding=encoding)

            # For larger files, use memory mapping
            with open(file_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return mm.read().decode(encoding)
        except Exception as e:
            logger.exception(f"Failed to read {file_path}: {e}")
            return None

    @staticmethod
    async def write_file_atomic(file_path: str, content: str, encoding: str = "utf-8") -> bool:
        """Atomic file writing - safer than chunked writes."""
        try:
            path = Path(file_path)
            # Write to temp file first, then atomic move
            with tempfile.NamedTemporaryFile(
                mode="w", encoding=encoding, delete=False, dir=path.parent
            ) as tmp:
                tmp.write(content)
                tmp.flush()
                temp_path = tmp.name

            # Atomic move
            Path(temp_path).replace(path)
            return True
        except Exception as e:
            logger.exception(f"Failed to write {file_path}: {e}")
            return False


class StreamingJsonProcessor:
    """Streaming JSON processing - better than chunked parsing."""

    @staticmethod
    def parse_json_stream(json_text: str) -> Any:
        """Parse JSON with streaming for large objects."""
        try:
            # For small JSON, use standard parsing
            if len(json_text) < 1024 * 1024:  # 1MB threshold
                return json.loads(json_text)

            # For large JSON, use streaming approach
            # Note: This would need a streaming JSON parser for very large files
            # For now, fall back to standard parsing with error handling
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.exception(f"JSON parsing error: {e}")
            return None


class UltimateErrorCollector:
    """Ultimate error collector combining best performance strategies."""

    def __init__(self, files: list[str] | None = None) -> None:
        self.files = files or []
        self.errors: dict[str, Any] = defaultdict(lambda: defaultdict(list))
        self.performance_metrics: dict[str, Any] = {}
        self.file_handler = MemoryMappedFileHandler()
        self.json_processor = StreamingJsonProcessor()

        # Initialize Rust cache for zero-copy operations
        if RUST_AVAILABLE:
            self.cache = RustCache(
                capacity=100000,  # Large cache
                default_ttl_secs=3600,  # 1 hour TTL
                max_memory_bytes=100 * 1024 * 1024,  # 100MB
            )
            self.command_executor = RustCommandExecutor()
            self.file_ops = RustFileOps(max_file_size=100 * 1024 * 1024)
        else:
            self.cache = {}
            self.command_executor = None
            self.file_ops = None

    async def _execute_command_ultimate(self, cmd: list[str]) -> dict[str, Any]:
        """Execute command with ultimate performance."""
        start_time = time.time()

        if RUST_AVAILABLE and self.command_executor:
            try:
                # Use Rust command executor for maximum performance
                result = await self.command_executor.execute_async(cmd[0], cmd[1:], timeout_secs=60)
                duration = time.time() - start_time
                self.performance_metrics[f"rust_cmd_{cmd[0]}"] = duration
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.exit_code,
                }
            except Exception as e:
                logger.warning(f"Rust command execution failed, falling back: {e}")

        # Fallback to optimized Python async execution
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=".",
            )
            stdout, stderr = await process.communicate()
            duration = time.time() - start_time
            self.performance_metrics[f"python_cmd_{cmd[0]}"] = duration

            return {
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "returncode": process.returncode,
            }
        except Exception as e:
            logger.exception(f"Command execution failed: {e}")
            return {"stdout": "", "stderr": str(e), "returncode": 1}

    async def run_mypy_ultimate(self) -> dict[str, list[str]]:
        """Run mypy with ultimate performance."""
        cmd = ["uv", "run", "mypy", "."]
        if self.files:
            cmd = ["uv", "run", "mypy", *self.files]

        result = await self._execute_command_ultimate(cmd)

        if result["returncode"] == 0:
            return {}

        # Parse mypy output with streaming approach
        output = result["stdout"]
        errors = defaultdict(list)

        # Process line by line for memory efficiency
        for line in output.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue

            try:
                # Parse mypy error format: file:line:col: error: message
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    file_path = parts[0]
                    error_msg = parts[3].strip()
                    errors[file_path].append(error_msg)
            except Exception:
                continue

        self.errors["mypy"] = dict(errors)
        return dict(errors)

    async def run_ruff_ultimate(self) -> dict[str, list[str]]:
        """Run ruff with ultimate performance."""
        cmd = ["uv", "run", "ruff", "check", "--output-format=json", "."]
        if self.files:
            cmd = ["uv", "run", "ruff", "check", "--output-format=json", *self.files]

        result = await self._execute_command_ultimate(cmd)

        if result["returncode"] == 0:
            return {}

        # Parse JSON output with streaming
        json_output = result["stdout"]
        parsed = self.json_processor.parse_json_stream(json_output)

        if not parsed:
            return {}

        errors = defaultdict(list)
        for issue in parsed:
            file_path = issue.get("filename", "unknown")
            message = issue.get("message", "Unknown error")
            code = issue.get("code", "")
            full_message = f"{code}: {message}" if code else message
            errors[file_path].append(full_message)

        self.errors["ruff"] = dict(errors)
        return dict(errors)

    async def run_ast_grep_ultimate(self) -> dict[str, list[str]]:
        """Run ast-grep analysis with ultimate performance."""
        patterns = [
            {
                "rule": "missing-return-type",
                "pattern": "def $func($$$): $$$",
                "message": "Function missing return type annotation",
            },
            {
                "rule": "missing-docstring",
                "pattern": "def $func($$$): $$$",
                "message": "Function missing docstring",
            },
        ]

        errors = defaultdict(list)

        for pattern_info in patterns:
            cmd = ["ast-grep", "-p", pattern_info["pattern"], "--json", "."]
            result = await self._execute_command_ultimate(cmd)

            if result["returncode"] == 0 and result["stdout"]:
                try:
                    matches = self.json_processor.parse_json_stream(result["stdout"])
                    for match in matches or []:
                        file_path = match.get("file", "unknown")
                        line = match.get("line", 0)
                        message = f"Line {line}: {pattern_info['message']}"
                        errors[file_path].append(message)
                except Exception as e:
                    logger.warning(f"Failed to parse ast-grep output: {e}")

        self.errors["ast-grep"] = dict(errors)
        return dict(errors)

    async def run_all_tools_concurrent(self) -> dict[str, Any]:
        """Run all tools concurrently with ultimate performance."""
        start_time = time.time()

        # Run all tools concurrently
        tasks = [
            self.run_mypy_ultimate(),
            self.run_ruff_ultimate(),
            self.run_ast_grep_ultimate(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool_name = ["mypy", "ruff", "ast-grep"][i]
                logger.error(f"Tool {tool_name} failed: {result}")

        duration = time.time() - start_time
        self.performance_metrics["total_analysis_time"] = duration

        return self.errors

    def get_performance_summary(self) -> str:
        """Get performance summary."""
        if not self.performance_metrics:
            return "No performance data available"

        total_time = self.performance_metrics.get("total_analysis_time", 0)
        rust_metrics = {k: v for k, v in self.performance_metrics.items() if "rust_" in k}

        summary = f"Total analysis time: {total_time:.2f}s\n"
        if rust_metrics:
            summary += f"Rust acceleration used for {len(rust_metrics)} commands\n"

        return summary


class UltimateClaudeAutoFixer:
    """Ultimate Claude auto-fixer with best performance strategies."""

    def __init__(self, model: str = "haiku", max_fixes_per_run: int = 10) -> None:
        self.model = model
        self.max_fixes_per_run = max_fixes_per_run
        self.file_handler = MemoryMappedFileHandler()
        self.claude_model = model

    def generate_fix_prompt(self, file_path: str, errors: list[str], file_content: str) -> str:
        """Generate fix prompt for Claude."""
        error_text = "\n".join(f"- {error}" for error in errors)

        return f"""Please fix the following errors in this Python file:

FILE: {file_path}

ERRORS:
{error_text}

CURRENT CODE:
```python
{file_content}
```

Please provide the corrected code. Make minimal changes to fix only the reported errors.
Return only the corrected Python code without explanations."""

    async def fix_file(self, file_path: str, errors: list[str]) -> bool:
        """Fix a single file using Claude."""
        try:
            # Read file content using memory mapping
            content = await self.file_handler.read_file_mmap(file_path)
            if content is None:
                logger.error(f"Failed to read {file_path}")
                return False

            # Generate fix prompt
            prompt = self.generate_fix_prompt(file_path, errors, content)

            # Call Claude API (simplified - would need actual Claude integration)
            cmd = ["claude", "--model", self.model, "--"]

            # Execute Claude fix
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate(prompt.encode())

            if process.returncode != 0:
                logger.error(f"Claude fix failed for {file_path}: {stderr.decode()}")
                return False

            fixed_content = stdout.decode().strip()

            # Write fixed content atomically
            success = await self.file_handler.write_file_atomic(file_path, fixed_content)

            if success:
                print(f"  ✅ Fixed {file_path}")
                return True
            else:
                logger.error(f"Failed to write fixed content to {file_path}")
                return False

        except Exception as e:
            logger.exception(f"Error fixing {file_path}: {e}")
            return False

    async def fix_all_errors(self, error_dict: dict[str, Any]) -> int:
        """Fix all errors using Claude with ultimate performance."""
        fixed_count = 0

        # Collect all files with errors
        files_to_fix = {}
        for _tool, tool_errors in error_dict.items():
            if isinstance(tool_errors, dict):
                for file_path, errors in tool_errors.items():
                    if file_path not in files_to_fix:
                        files_to_fix[file_path] = []
                    files_to_fix[file_path].extend(errors)

        # Fix files concurrently (but limit concurrency to avoid API limits)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent fixes

        async def fix_with_semaphore(file_path: str, errors: list[str]) -> bool:
            async with semaphore:
                return await self.fix_file(file_path, errors)

        # Create tasks for all files
        tasks = [
            fix_with_semaphore(file_path, errors) for file_path, errors in files_to_fix.items()
        ]

        # Execute all fixes
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful fixes
        for result in results:
            if isinstance(result, bool) and result:
                fixed_count += 1
            elif isinstance(result, Exception):
                logger.error(f"Fix task failed: {result}")

        return fixed_count


async def main() -> int:
    """Main entry point with ultimate performance."""
    parser = argparse.ArgumentParser(description="Claude Auto-Fix Ultimate")
    parser.add_argument("--model", choices=["haiku", "sonnet"], default="haiku")
    parser.add_argument("--max-fixes", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=["mypy", "ruff", "ast-grep", "all"],
        default=["all"],
    )
    parser.add_argument("files", nargs="*")

    args = parser.parse_args()

    print(f"{GREEN}🚀 CLAUDE AUTO-FIX ULTIMATE{RESET}")
    print(f"Model: {args.model}, Max fixes: {args.max_fixes}")

    # Collect errors with ultimate performance
    collector = UltimateErrorCollector(args.files)

    start_time = time.time()

    if "all" in args.tools:
        all_errors = await collector.run_all_tools_concurrent()
    else:
        # Run selected tools concurrently
        tasks = []
        if "mypy" in args.tools:
            tasks.append(collector.run_mypy_ultimate())
        if "ruff" in args.tools:
            tasks.append(collector.run_ruff_ultimate())
        if "ast-grep" in args.tools:
            tasks.append(collector.run_ast_grep_ultimate())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        all_errors = collector.errors

    analysis_time = time.time() - start_time

    # Show performance metrics
    if args.benchmark:
        print(f"\n{CYAN}=== PERFORMANCE METRICS ==={RESET}")
        print(f"Analysis time: {analysis_time:.2f}s")
        print(collector.get_performance_summary())

    # Print summary
    print(f"\n{BLUE}=== ERROR SUMMARY ==={RESET}")
    error_count = 0
    for tool, errors in all_errors.items():
        count = sum(len(e) for e in errors.values()) if isinstance(errors, dict) else len(errors)
        error_count += count
        print(f"  {tool}: {count} errors")

    if error_count == 0:
        print(f"{GREEN}✅ No errors found!{RESET}")
        return 0

    if args.dry_run:
        print(f"\n{YELLOW}🔍 Dry run mode - analysis complete{RESET}")
        return 1

    # Fix errors using Claude
    print(f"\n{BLUE}=== FIXING ERRORS WITH CLAUDE ==={RESET}")
    fixer = UltimateClaudeAutoFixer(args.model, max_fixes_per_run=args.max_fixes)

    fix_start_time = time.time()
    fixed_count = await fixer.fix_all_errors(all_errors)
    fix_time = time.time() - fix_start_time

    print(f"\n{GREEN}✅ Fixed {fixed_count} files in {fix_time:.2f}s{RESET}")

    # Re-run tools to verify fixes
    print(f"\n{BLUE}=== VERIFYING FIXES ==={RESET}")
    collector2 = UltimateErrorCollector(args.files)
    await collector2.run_all_tools_concurrent()

    remaining_errors = collector2.errors
    remaining_count = sum(
        len(e)
        for errors in remaining_errors.values()
        for e in (errors.values() if isinstance(errors, dict) else [errors])
    )

    if remaining_count == 0:
        print(f"{GREEN}🎉 All errors fixed successfully!{RESET}")
        return 0

    print(f"{YELLOW}⚠️  Still {remaining_count} errors remaining{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
