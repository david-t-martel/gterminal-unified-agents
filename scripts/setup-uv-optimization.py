#!/usr/bin/env python3
"""UV Setup and Optimization Script.

This script sets up UV Python environment and optimizes the project for UV usage.
"""

from pathlib import Path
import subprocess
import sys

# Color codes
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    try:
        print(f"{BLUE}Running: {description}{RESET}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            print(f"  ✅ {description} completed")
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


def setup_uv_environment():
    """Set up UV Python environment."""
    print(f"{GREEN}🚀 SETTING UP UV PYTHON ENVIRONMENT{RESET}")

    steps = [
        (["uv", "sync", "--dev"], "Syncing development dependencies"),
        (["uv", "sync", "--extra", "dev"], "Installing dev extras"),
        (["uv", "sync", "--extra", "test"], "Installing test extras"),
        (["uv", "sync", "--extra", "quality"], "Installing quality tools"),
        (["uv", "sync", "--extra", "performance"], "Installing performance tools"),
        (["uv", "sync", "--extra", "mcp"], "Installing MCP extras"),
    ]

    success_count = 0
    for cmd, description in steps:
        if run_command(cmd, description):
            success_count += 1

    print(f"\n{CYAN}Environment setup: {success_count}/{len(steps)} successful{RESET}")
    return success_count == len(steps)


def validate_uv_installation():
    """Validate UV installation."""
    print(f"\n{GREEN}🔍 VALIDATING UV INSTALLATION{RESET}")

    checks = [
        (["uv", "--version"], "UV version"),
        (["uv", "run", "python", "--version"], "UV Python execution"),
        (
            [
                "uv",
                "run",
                "python",
                "-c",
                "import sys; print(f'Python: {sys.executable}')",
            ],
            "Python path",
        ),
    ]

    success_count = 0
    for cmd, description in checks:
        if run_command(cmd, description):
            success_count += 1

    return success_count == len(checks)


def optimize_for_uv():
    """Optimize project scripts for UV usage."""
    print(f"\n{GREEN}⚡ OPTIMIZING PROJECT FOR UV{RESET}")

    # Check that key scripts work with UV
    scripts_to_test = [
        "scripts/enhanced_toolchain.py",
        "scripts/claude-auto-fix-ultimate.py",
        "gemini_cli/main.py",
    ]

    success_count = 0
    total_scripts = 0

    for script_path in scripts_to_test:
        if Path(script_path).exists():
            total_scripts += 1
            cmd = ["uv", "run", "python", script_path, "--help"]

            # Special case for main.py
            if "main.py" in script_path:
                cmd = ["uv", "run", "python", "-m", "gemini_cli", "--help"]

            if run_command(cmd, f"Testing {script_path}"):
                success_count += 1

    print(f"\n{CYAN}Script compatibility: {success_count}/{total_scripts} working{RESET}")
    return success_count == total_scripts


def setup_development_tools():
    """Set up development tools with UV."""
    print(f"\n{GREEN}🛠️  SETTING UP DEVELOPMENT TOOLS{RESET}")

    tools = [
        (["uv", "run", "pre-commit", "install"], "Installing pre-commit hooks"),
        (
            ["uv", "run", "pre-commit", "install", "--hook-type", "commit-msg"],
            "Installing commit-msg hooks",
        ),
    ]

    success_count = 0
    for cmd, description in tools:
        if run_command(cmd, description):
            success_count += 1

    return success_count == len(tools)


def create_uv_aliases():
    """Create helpful UV aliases and shortcuts."""
    print(f"\n{GREEN}📝 CREATING UV SHORTCUTS{RESET}")

    # Create a .uvrc file with helpful aliases
    uvrc_content = """# UV Python Shortcuts for gterminal project
# Add these to your .bashrc or .zshrc:

alias uvpy="uv run python"
alias uvtest="uv run pytest"
alias uvlint="uv run ruff check"
alias uvformat="uv run ruff format"
alias uvtype="uv run mypy"
alias uvfix="uv run python scripts/claude-auto-fix-ultimate.py"
alias uvtool="uv run python scripts/enhanced_toolchain.py"

# Project-specific commands
alias gt-cli="uv run python -m gemini_cli"
alias gt-test="uv run pytest tests/"
alias gt-lint="uv run ruff check . && uv run mypy ."
alias gt-fix="uv run python scripts/claude-auto-fix-ultimate.py --tools all"
"""

    uvrc_path = Path(".uvrc")
    try:
        uvrc_path.write_text(uvrc_content)
        print(f"  ✅ Created {uvrc_path} with UV shortcuts")
        print("     Add 'source .uvrc' to your shell profile to use aliases")
        return True
    except Exception as e:
        print(f"  ❌ Failed to create .uvrc: {e}")
        return False


def run_comprehensive_test():
    """Run a comprehensive test of UV integration."""
    print(f"\n{GREEN}🧪 RUNNING COMPREHENSIVE UV TEST{RESET}")

    test_commands = [
        (
            [
                "uv",
                "run",
                "python",
                "-c",
                "import gemini_cli; print('✅ Main package importable')",
            ],
            "Package import test",
        ),
        (
            ["uv", "run", "ruff", "check", "--select", "F", "gemini_cli/"],
            "Quick lint check",
        ),
        (
            [
                "uv",
                "run",
                "python",
                "-c",
                "import click, rich, asyncio; print('✅ Core dependencies available')",
            ],
            "Dependency test",
        ),
    ]

    success_count = 0
    for cmd, description in test_commands:
        if run_command(cmd, description):
            success_count += 1

    return success_count == len(test_commands)


def main():
    """Main setup function."""
    print(f"{GREEN}🚀 UV PYTHON SETUP AND OPTIMIZATION{RESET}")
    print("Setting up UV Python environment for gterminal project...\n")

    # Run all setup steps
    steps = [
        ("UV Installation Validation", validate_uv_installation),
        ("Environment Setup", setup_uv_environment),
        ("Project Optimization", optimize_for_uv),
        ("Development Tools", setup_development_tools),
        ("UV Shortcuts", create_uv_aliases),
        ("Comprehensive Test", run_comprehensive_test),
    ]

    results = []
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"  ❌ {step_name} failed with exception: {e}")
            results.append((step_name, False))

    # Summary
    print(f"\n{GREEN}📊 SETUP SUMMARY{RESET}")
    passed = 0
    total = len(results)

    for step_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status} {step_name}")
        if success:
            passed += 1

    print(f"\nSetup steps completed: {passed}/{total}")

    if passed == total:
        print(f"\n{GREEN}🎉 UV SETUP COMPLETE!{RESET}")
        print(f"{GREEN}The project is now fully optimized for UV Python.{RESET}")
        print(f"\n{CYAN}Next steps:{RESET}")
        print("  1. Add 'source .uvrc' to your shell profile for UV aliases")
        print("  2. Use 'uv run python' instead of 'python' for all commands")
        print("  3. Use 'uv add package' instead of 'pip install package'")
        print("  4. Use 'uv sync' to update dependencies")
        return 0
    else:
        print(f"\n{YELLOW}⚠️  Some setup steps failed. Please review the issues above.{RESET}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
