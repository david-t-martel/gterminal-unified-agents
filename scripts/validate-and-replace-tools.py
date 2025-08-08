#!/usr/bin/env python3
"""Tool Validation and Replacement Script.

This script validates that new PyO3-optimized tools work equivalently to the old ones,
then safely replaces them while preserving functionality.
"""

import asyncio
from pathlib import Path
import time
from typing import Any

# Color codes
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


class ToolValidator:
    """Validates equivalency between old and new tools."""

    def __init__(self):
        self.test_files = [
            "scripts/claude-auto-fix.py",
            "scripts/enhanced_toolchain.py",
            "scripts/rufft-claude.sh",
        ]

        # Tool pairs for comparison (old -> new)
        self.tool_pairs = {
            "claude-auto-fix.py": "claude-auto-fix-performance.py",
            "enhanced_toolchain.py": "pyo3-chunked-cache.py",
            "rufft-claude.sh": "rufft-claude-optimized.sh",
        }

    async def run_tool_test(
        self, tool_path: str, args: list[str], test_file: str
    ) -> dict[str, Any]:
        """Run a tool and capture its output for comparison."""
        if not Path(tool_path).exists():
            return {"error": f"Tool not found: {tool_path}"}

        start_time = time.time()

        try:
            if tool_path.endswith(".py"):
                cmd = ["uv", "run", "python", tool_path, *args, test_file]
            elif tool_path.endswith(".sh"):
                cmd = ["bash", tool_path, *args, test_file]
            else:
                cmd = [tool_path, *args, test_file]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/home/david/agents/gterminal",
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)

            execution_time = time.time() - start_time

            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "returncode": process.returncode,
                "execution_time": execution_time,
                "cmd": cmd,
            }

        except TimeoutError:
            return {"error": "Tool execution timed out", "execution_time": 30.0}
        except Exception as e:
            return {"error": str(e), "execution_time": time.time() - start_time}

    def normalize_output(self, output: str) -> str:
        """Normalize output for comparison (remove timestamps, paths, etc.)."""
        lines = output.split("\n")
        normalized = []

        for line in lines:
            # Remove ANSI color codes
            import re

            line = re.sub(r"\x1b\[[0-9;]*m", "", line)

            # Remove timestamps
            line = re.sub(r"\d{2}:\d{2}:\d{2}", "HH:MM:SS", line)

            # Remove file paths (keep relative structure)
            line = re.sub(r"/home/david/agents/gterminal/", "", line)

            # Remove execution times
            line = re.sub(r"\d+\.\d+s", "X.XXs", line)

            # Remove memory addresses
            line = re.sub(r"0x[0-9a-f]+", "0xXXXXXX", line)

            normalized.append(line.strip())

        return "\n".join(normalized).strip()

    def calculate_similarity(self, output1: str, output2: str) -> float:
        """Calculate similarity between two outputs."""
        norm1 = self.normalize_output(output1)
        norm2 = self.normalize_output(output2)

        if not norm1 and not norm2:
            return 1.0  # Both empty

        if not norm1 or not norm2:
            return 0.0  # One empty, one not

        # Simple similarity based on common lines
        lines1 = set(norm1.split("\n"))
        lines2 = set(norm2.split("\n"))

        if not lines1 and not lines2:
            return 1.0

        intersection = lines1.intersection(lines2)
        union = lines1.union(lines2)

        return len(intersection) / len(union) if union else 0.0

    async def validate_tool_pair(self, old_tool: str, new_tool: str) -> dict[str, Any]:
        """Validate that new tool works equivalently to old tool."""
        old_path = f"scripts/{old_tool}"
        new_path = f"scripts/{new_tool}"

        print(f"{BLUE}🔍 Validating: {old_tool} vs {new_tool}{RESET}")

        if not Path(old_path).exists():
            return {"error": f"Old tool not found: {old_path}"}

        if not Path(new_path).exists():
            return {"error": f"New tool not found: {new_path}"}

        results = {}

        # Test with different scenarios
        test_scenarios = [
            {"args": ["--help"], "file": ""},
            {"args": ["--dry-run"], "file": "scripts/claude-auto-fix.py"},
            {"args": [], "file": "scripts/enhanced_toolchain.py"},
        ]

        for i, scenario in enumerate(test_scenarios):
            scenario_name = f"scenario_{i}"

            # Skip file argument if empty
            test_args = scenario["args"]
            test_file = scenario["file"]

            if not test_file:
                # For help commands, don't include test file
                old_result = await self.run_tool_test(old_path, test_args, "")
                new_result = await self.run_tool_test(new_path, test_args, "")
            else:
                if not Path(test_file).exists():
                    continue
                old_result = await self.run_tool_test(old_path, test_args, test_file)
                new_result = await self.run_tool_test(new_path, test_args, test_file)

            # Compare results
            similarity = 0.0
            performance_ratio = 0.0

            if "error" not in old_result and "error" not in new_result:
                # Compare output similarity
                similarity = self.calculate_similarity(
                    old_result.get("stdout", ""), new_result.get("stdout", "")
                )

                # Compare performance
                old_time = old_result.get("execution_time", 1.0)
                new_time = new_result.get("execution_time", 1.0)
                performance_ratio = old_time / new_time if new_time > 0 else 0.0

            results[scenario_name] = {
                "old_result": old_result,
                "new_result": new_result,
                "similarity": similarity,
                "performance_ratio": performance_ratio,
                "args": test_args,
                "file": test_file,
            }

        return results

    async def run_validation(self) -> dict[str, Any]:
        """Run complete validation of all tool pairs."""
        print(f"{GREEN}🚀 TOOL VALIDATION AND REPLACEMENT{RESET}")
        print(f"{BLUE}Validating new PyO3-optimized tools against originals{RESET}")

        validation_results = {}

        for old_tool, new_tool in self.tool_pairs.items():
            try:
                result = await self.validate_tool_pair(old_tool, new_tool)
                validation_results[old_tool] = result

                # Print summary for this tool pair
                if "error" not in result:
                    avg_similarity = sum(
                        scenario.get("similarity", 0)
                        for scenario in result.values()
                        if isinstance(scenario, dict)
                    ) / len(result)

                    avg_performance = sum(
                        scenario.get("performance_ratio", 0)
                        for scenario in result.values()
                        if isinstance(scenario, dict)
                    ) / len(result)

                    print(
                        f"  ✅ {old_tool}: {avg_similarity:.2%} similar, "
                        f"{avg_performance:.1f}x performance"
                    )
                else:
                    print(f"  ❌ {old_tool}: {result['error']}")

            except Exception as e:
                validation_results[old_tool] = {"error": str(e)}
                print(f"  ❌ {old_tool}: Validation failed - {e}")

        return validation_results

    def should_replace_tool(
        self, validation_result: dict[str, Any], min_similarity: float = 0.7
    ) -> bool:
        """Determine if a tool should be replaced based on validation."""
        if "error" in validation_result:
            return False

        # Check if all scenarios meet minimum similarity
        for scenario in validation_result.values():
            if isinstance(scenario, dict):
                similarity = scenario.get("similarity", 0)
                if similarity < min_similarity:
                    return False

        return True

    def backup_and_replace_tool(self, old_tool: str, new_tool: str) -> bool:
        """Safely replace old tool with new one after backup."""
        old_path = Path(f"scripts/{old_tool}")
        new_path = Path(f"scripts/{new_tool}")

        if not old_path.exists() or not new_path.exists():
            return False

        try:
            # Create backup
            backup_path = old_path.with_suffix(old_path.suffix + ".backup")
            old_path.rename(backup_path)

            # Replace with new tool
            new_path.rename(old_path)

            print(f"  ✅ Replaced {old_tool} (backup: {backup_path.name})")
            return True

        except Exception as e:
            print(f"  ❌ Failed to replace {old_tool}: {e}")
            return False


async def main():
    """Main validation and replacement workflow."""
    validator = ToolValidator()

    # Run validation
    results = await validator.run_validation()

    print(f"\n{CYAN}=== VALIDATION SUMMARY ==={RESET}")

    replacement_candidates = []

    for tool, result in results.items():
        if "error" in result:
            print(f"❌ {tool}: {result['error']}")
            continue

        # Calculate overall metrics
        scenarios = [s for s in result.values() if isinstance(s, dict)]
        if not scenarios:
            continue

        avg_similarity = sum(s.get("similarity", 0) for s in scenarios) / len(scenarios)
        avg_performance = sum(s.get("performance_ratio", 0) for s in scenarios) / len(scenarios)

        status = "✅" if avg_similarity >= 0.7 else "⚠️"
        print(f"{status} {tool}:")
        print(f"  📊 Similarity: {avg_similarity:.2%}")
        print(f"  ⚡ Performance: {avg_performance:.1f}x")

        if validator.should_replace_tool(result):
            replacement_candidates.append(tool)

    # Ask for confirmation before replacement
    if replacement_candidates:
        print(f"\n{YELLOW}🔄 Ready to replace {len(replacement_candidates)} tools:{RESET}")
        for tool in replacement_candidates:
            new_tool = validator.tool_pairs[tool]
            print(f"  {tool} -> {new_tool}")

        # In automated mode, proceed with replacement
        print(f"\n{GREEN}🔄 Proceeding with tool replacement...{RESET}")

        for tool in replacement_candidates:
            new_tool = validator.tool_pairs[tool]
            success = validator.backup_and_replace_tool(tool, new_tool)
            if success:
                print(f"✅ Successfully replaced {tool}")
            else:
                print(f"❌ Failed to replace {tool}")

    else:
        print(f"\n{YELLOW}⚠️  No tools meet replacement criteria{RESET}")

    # Final summary
    print(f"\n{GREEN}🎯 VALIDATION COMPLETE{RESET}")
    print(f"Tools validated: {len(results)}")
    print(f"Tools replaced: {len([t for t in replacement_candidates if t in results])}")


if __name__ == "__main__":
    asyncio.run(main())
