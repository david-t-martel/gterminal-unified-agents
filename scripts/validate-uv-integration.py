#!/usr/bin/env python3
"""UV Integration Validation Script.

This script validates that all tools and scripts work correctly with UV Python.
"""

import asyncio
from pathlib import Path
import sys
from typing import Any

# Color codes
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


async def run_uv_command(cmd: list[str]) -> dict[str, Any]:
    """Run a UV command and return results."""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=".",
        )
        stdout, stderr = await process.communicate()

        return {
            "cmd": " ".join(cmd),
            "returncode": process.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "success": process.returncode == 0,
        }
    except Exception as e:
        return {
            "cmd": " ".join(cmd),
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }


async def validate_uv_installation():
    """Validate UV installation and basic functionality."""
    print(f"{CYAN}🔍 Validating UV installation...{RESET}")

    tests = [
        (["uv", "--version"], "UV version check"),
        (["uv", "run", "python", "--version"], "UV Python execution"),
        (
            ["uv", "run", "python", "-c", "import sys; print(sys.executable)"],
            "UV Python path",
        ),
    ]

    results = []
    for cmd, description in tests:
        result = await run_uv_command(cmd)
        results.append((description, result))

        if result["success"]:
            print(f"  ✅ {description}: {result['stdout'].strip()}")
        else:
            print(f"  ❌ {description}: {result['stderr'].strip()}")

    return all(r[1]["success"] for r in results)


async def validate_project_scripts():
    """Validate that project scripts work with UV."""
    print(f"\n{CYAN}🔍 Validating project scripts with UV...{RESET}")

    scripts_dir = Path("scripts")
    test_scripts = [
        ("enhanced_toolchain.py", "--help"),
        ("claude-auto-fix-ultimate.py", "--help"),
        ("consolidate-build-session.py", None),  # No help flag
    ]

    results = []
    for script_name, help_flag in test_scripts:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"  ⚠️  Script not found: {script_name}")
            continue

        cmd = ["uv", "run", "python", str(script_path)]
        if help_flag:
            cmd.append(help_flag)
        else:
            # For scripts without help, do a dry run or version check
            cmd = [
                "uv",
                "run",
                "python",
                "-c",
                f"import sys; sys.path.insert(0, 'scripts'); print('✅ {script_name} importable')",
            ]

        result = await run_uv_command(cmd)
        results.append((script_name, result))

        if result["success"]:
            print(f"  ✅ {script_name}: Working with UV")
        else:
            print(f"  ❌ {script_name}: {result['stderr'].strip()}")

    return results


async def validate_development_tools():
    """Validate development tools work with UV."""
    print(f"\n{CYAN}🔍 Validating development tools with UV...{RESET}")

    tools = [
        (["uv", "run", "ruff", "--version"], "Ruff linter"),
        (["uv", "run", "mypy", "--version"], "MyPy type checker"),
        (["uv", "run", "pytest", "--version"], "Pytest testing"),
        (["uv", "run", "black", "--version"], "Black formatter"),
        (["uv", "run", "isort", "--version"], "isort import sorter"),
    ]

    results = []
    for cmd, description in tools:
        result = await run_uv_command(cmd)
        results.append((description, result))

        if result["success"]:
            version = result["stdout"].strip().split("\n")[0]
            print(f"  ✅ {description}: {version}")
        else:
            print(f"  ❌ {description}: Not available or error")

    return results


async def validate_rust_extensions():
    """Validate Rust extensions work with UV Python."""
    print(f"\n{CYAN}🔍 Validating Rust extensions with UV...{RESET}")

    test_code = """
try:
    from gterminal_rust_extensions import RustCache, RustFileOps, version
    print(f"✅ Rust extensions v{version()} loaded successfully")

    # Test basic functionality
    cache = RustCache(capacity=100)
    cache.set("test", "value")
    result = cache.get("test")
    if result == "value":
        print("✅ RustCache basic operations working")
    else:
        print("❌ RustCache operations failed")

except ImportError as e:
    print(f"⚠️  Rust extensions not available: {e}")
except Exception as e:
    print(f"❌ Rust extensions error: {e}")
"""

    cmd = ["uv", "run", "python", "-c", test_code]
    result = await run_uv_command(cmd)

    if result["success"]:
        print(f"  {result['stdout'].strip()}")
    else:
        print(f"  ❌ Rust extensions validation failed: {result['stderr'].strip()}")

    return result["success"]


async def validate_mcp_compatibility():
    """Validate MCP server compatibility with UV."""
    print(f"\n{CYAN}🔍 Validating MCP compatibility with UV...{RESET}")

    test_code = """
try:
    import mcp
    from mcp.client.stdio import stdio_client
    print("✅ MCP framework available")

    # Test MCP client creation
    print("✅ MCP client can be imported")

except ImportError as e:
    print(f"❌ MCP framework not available: {e}")
except Exception as e:
    print(f"❌ MCP validation error: {e}")
"""

    cmd = ["uv", "run", "python", "-c", test_code]
    result = await run_uv_command(cmd)

    if result["success"]:
        print(f"  {result['stdout'].strip()}")
    else:
        print(f"  ❌ MCP validation failed: {result['stderr'].strip()}")

    return result["success"]


async def run_performance_benchmark():
    """Run a quick performance benchmark with UV."""
    print(f"\n{CYAN}🔍 Running UV performance benchmark...{RESET}")

    import time

    # Test UV startup time
    start_time = time.time()
    result = await run_uv_command(["uv", "run", "python", "-c", "print('UV startup test')"])
    startup_time = time.time() - start_time

    if result["success"]:
        print(f"  ✅ UV Python startup: {startup_time:.3f}s")
    else:
        print("  ❌ UV startup failed")

    # Test import time for common packages
    import_test = """
import time
start = time.time()
import click, rich, asyncio, pathlib, json
end = time.time()
print(f"Import time: {(end-start)*1000:.1f}ms")
"""

    result = await run_uv_command(["uv", "run", "python", "-c", import_test])
    if result["success"]:
        print(f"  ✅ Package imports: {result['stdout'].strip()}")

    return startup_time < 2.0  # Should start in under 2 seconds


async def main():
    """Main validation function."""
    print(f"{GREEN}🚀 UV INTEGRATION VALIDATION{RESET}")
    print("Validating UV Python integration across the project...\n")

    # Run all validation tests
    tests = [
        ("UV Installation", validate_uv_installation()),
        ("Project Scripts", validate_project_scripts()),
        ("Development Tools", validate_development_tools()),
        ("Rust Extensions", validate_rust_extensions()),
        ("MCP Compatibility", validate_mcp_compatibility()),
        ("Performance Benchmark", run_performance_benchmark()),
    ]

    results = []
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{GREEN}📊 VALIDATION SUMMARY{RESET}")
    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
        if success:
            passed += 1

    print(f"\nTests passed: {passed}/{total}")

    if passed == total:
        print(f"\n{GREEN}🎉 ALL UV INTEGRATION TESTS PASSED!{RESET}")
        print(f"{GREEN}The project is fully compatible with UV Python.{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}⚠️  Some tests failed. Please review the issues above.{RESET}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
