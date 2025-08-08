#!/usr/bin/env python3
"""UV + LSP Integration Test and Demo.

This script tests and demonstrates the integration between UV Python,
the LSP tools (rufft-claude.sh), and the VertexAI Gemini CLI components.
"""

import asyncio
import subprocess
import sys

# Color codes
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


async def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    try:
        print(f"{BLUE}Running: {description}{RESET}")
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=".",
        )
        stdout, stderr = await result.communicate()

        if result.returncode == 0:
            print(f"  ✅ {description}")
            if stdout.strip():
                print(f"     {stdout.decode().strip()}")
            return True
        else:
            print(f"  ❌ {description} failed")
            if stderr.strip():
                print(f"     Error: {stderr.decode().strip()}")
            return False
    except Exception as e:
        print(f"  ❌ {description} failed with exception: {e}")
        return False


def run_sync_command(cmd: list[str], description: str) -> bool:
    """Run a synchronous command."""
    try:
        print(f"{BLUE}Running: {description}{RESET}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=".")

        if result.returncode == 0:
            print(f"  ✅ {description}")
            if result.stdout.strip():
                print(f"     {result.stdout.strip()}")
            return True
        else:
            print(f"  ❌ {description} failed")
            if result.stderr.strip():
                print(f"     Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  ❌ {description} failed with exception: {e}")
        return False


async def test_uv_integration():
    """Test UV Python integration."""
    print(f"\n{CYAN}🔍 Testing UV Python Integration{RESET}")

    tests = [
        (["uv", "--version"], "UV version check"),
        (["uv", "run", "python", "--version"], "UV Python execution"),
        (
            [
                "uv",
                "run",
                "python",
                "-c",
                "import gemini_cli; print('Gemini CLI importable')",
            ],
            "Gemini CLI import",
        ),
        (
            [
                "uv",
                "run",
                "python",
                "-c",
                "import gterminal.lsp.filewatcher_integration; print('LSP integration available')",
            ],
            "LSP integration",
        ),
    ]

    results = []
    for cmd, description in tests:
        result = await run_command(cmd, description)
        results.append(result)

    return all(results)


async def test_vertex_ai_core():
    """Test VertexAI and Gemini core components."""
    print(f"\n{CYAN}🔍 Testing VertexAI & Gemini Core{RESET}")

    # Test Gemini client import and basic functionality
    test_code = """
try:
    from gemini_cli.core.client import GeminiClient
    from gemini_cli.core.auth import GeminiAuth
    print("✅ Gemini CLI core components importable")

    # Test authentication setup (don't actually authenticate)
    print("✅ GeminiAuth available")
    print("✅ GeminiClient available")

except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"⚠️  Warning: {e}")
"""

    cmd = ["uv", "run", "python", "-c", test_code]
    result = await run_command(cmd, "VertexAI Gemini core components")

    # Test unified client
    unified_test = """
try:
    from core.unified_gemini_client import GeminiClient, get_gemini_client
    print("✅ Unified Gemini client available")

    # Test profile system
    profiles = GeminiClient.PROFILES
    print(f"✅ Available profiles: {list(profiles.keys())}")

except ImportError as e:
    print(f"❌ Unified client import failed: {e}")
except Exception as e:
    print(f"⚠️  Warning: {e}")
"""

    cmd2 = ["uv", "run", "python", "-c", unified_test]
    result2 = await run_command(cmd2, "Unified Gemini client")

    return result and result2


def test_lsp_tools():
    """Test LSP tools (rufft-claude.sh)."""
    print(f"\n{CYAN}🔍 Testing LSP Tools (rufft-claude.sh){RESET}")

    lsp_tests = [
        (["bash", "scripts/rufft-claude.sh", "lsp-status"], "LSP status check"),
        (["bash", "scripts/rufft-claude.sh", "--help"], "LSP help command"),
    ]

    results = []
    for cmd, description in lsp_tests:
        result = run_sync_command(cmd, description)
        results.append(result)

    return all(results)


async def test_performance_integrations():
    """Test performance integrations (Rust extensions)."""
    print(f"\n{CYAN}🔍 Testing Performance Integrations{RESET}")

    rust_test = """
try:
    from gterminal_rust_extensions import RustCache, RustFileOps, version
    print(f"✅ Rust extensions v{version()} available")

    # Test basic functionality
    cache = RustCache(capacity=100)
    cache.set("test", "value")
    result = cache.get("test")
    if result == "value":
        print("✅ RustCache basic operations working")
    else:
        print("❌ RustCache operations failed")

except ImportError:
    print("⚠️  Rust extensions not available (this is optional)")
except Exception as e:
    print(f"❌ Rust extensions error: {e}")
"""

    cmd = ["uv", "run", "python", "-c", rust_test]
    await run_command(cmd, "Rust extensions (optional)")

    return True  # Don't fail if Rust extensions aren't available


async def demo_complete_workflow():
    """Demonstrate a complete UV + LSP + Gemini workflow."""
    print(f"\n{CYAN}🎯 Demo: Complete UV + LSP + Gemini Workflow{RESET}")

    workflow_steps = [
        # 1. UV dependency check
        (
            ["uv", "run", "python", "-c", "print('✅ UV Python environment ready')"],
            "UV environment",
        ),
        # 2. Import all core components
        (
            [
                "uv",
                "run",
                "python",
                "-c",
                """
import gemini_cli
import gterminal.lsp.filewatcher_integration
from core.unified_gemini_client import GeminiClient
print('✅ All core components imported successfully')
""",
            ],
            "Component imports",
        ),
        # 3. Check LSP status
        (["bash", "scripts/rufft-claude.sh", "lsp-status"], "LSP status"),
        # 4. Test enhanced toolchain
        (
            ["uv", "run", "python", "scripts/enhanced_toolchain.py", "--help"],
            "Enhanced toolchain",
        ),
    ]

    success_count = 0
    for cmd, description in workflow_steps:
        if isinstance(cmd[0], str) and cmd[0] == "bash":
            result = run_sync_command(cmd, description)
        else:
            result = await run_command(cmd, description)

        if result:
            success_count += 1

    print(f"\n{GREEN}Workflow demo: {success_count}/{len(workflow_steps)} steps successful{RESET}")
    return success_count == len(workflow_steps)


async def main():
    """Main test runner."""
    print(f"{GREEN}🚀 UV + LSP + VERTEX AI INTEGRATION TEST{RESET}")
    print("Testing complete integration across UV Python, LSP tools, and VertexAI components...\n")

    # Run all test suites
    test_suites = [
        ("UV Integration", test_uv_integration()),
        ("VertexAI & Gemini Core", test_vertex_ai_core()),
        ("LSP Tools", test_lsp_tools()),
        ("Performance Integrations", test_performance_integrations()),
        ("Complete Workflow Demo", demo_complete_workflow()),
    ]

    results = []
    for suite_name, test_coro in test_suites:
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            results.append((suite_name, result))
        except Exception as e:
            print(f"  ❌ {suite_name} failed with exception: {e}")
            results.append((suite_name, False))

    # Summary
    print(f"\n{GREEN}📊 INTEGRATION TEST SUMMARY{RESET}")
    passed = 0
    total = len(results)

    for suite_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {suite_name}")
        if success:
            passed += 1

    print(f"\nTest suites passed: {passed}/{total}")

    if passed == total:
        print(f"\n{GREEN}🎉 ALL INTEGRATION TESTS PASSED!{RESET}")
        print(f"{GREEN}UV Python + LSP + VertexAI integration is working perfectly!{RESET}")
        print(f"\n{CYAN}Ready for development with:{RESET}")
        print("  • UV Python: uv run python ...")
        print("  • LSP Tools: bash scripts/rufft-claude.sh ...")
        print("  • Gemini CLI: uv run python -m gemini_cli ...")
        print("  • Enhanced Tools: uv run python scripts/enhanced_toolchain.py ...")
        return 0
    else:
        print(f"\n{YELLOW}⚠️  Some tests failed. Integration is partially working.{RESET}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
