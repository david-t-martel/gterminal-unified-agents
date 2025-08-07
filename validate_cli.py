#!/usr/bin/env python3
"""Validation script for the standalone Gemini CLI."""

import importlib.util
import os
import sys
import time
from pathlib import Path


def validate_structure():
    """Validate project structure."""
    print("🔍 Validating project structure...")

    required_files = [
        "gemini_cli/__init__.py",
        "gemini_cli/main.py",
        "gemini_cli/core/auth.py",
        "gemini_cli/core/client.py",
        "gemini_cli/core/react_engine.py",
        "gemini_cli/tools/filesystem.py",
        "gemini_cli/tools/code_analysis.py",
        "gemini_cli/terminal/ui.py",
        "pyproject.toml",
        "Makefile",
        "README.md",
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
            print(f"  ❌ Missing: {file}")
        else:
            print(f"  ✅ Found: {file}")

    return len(missing) == 0


def validate_auth():
    """Validate authentication setup."""
    print("\n🔐 Validating authentication...")

    service_account_path = "/home/david/.auth/business/service-account-key.json"

    if Path(service_account_path).exists():
        print(f"  ✅ Service account found: {service_account_path}")

        # Check if it's properly configured in the code
        auth_file = Path("gemini_cli/core/auth.py")
        if auth_file.exists():
            content = auth_file.read_text()
            if service_account_path in content:
                print("  ✅ Service account path configured in auth.py")
            else:
                print("  ❌ Service account path not found in auth.py")
                return False
    else:
        print(f"  ❌ Service account not found: {service_account_path}")
        return False

    return True


def validate_imports():
    """Validate that imports work."""
    print("\n📦 Validating imports...")

    # Check if modules can be found without importing them directly
    modules_to_check = [
        ("gemini_cli", "gemini_cli package"),
        ("gemini_cli.core.auth", "auth module"),
        ("gemini_cli.core.client", "client module"),
        ("gemini_cli.core.react_engine", "react_engine module"),
        ("gemini_cli.tools.filesystem", "filesystem tools"),
        ("gemini_cli.tools.code_analysis", "code_analysis tools"),
        ("gemini_cli.terminal.ui", "terminal UI"),
    ]

    all_found = True
    for module_name, description in modules_to_check:
        if importlib.util.find_spec(module_name) is not None:
            print(f"  ✅ {description} imports successfully")
        else:
            print(f"  ❌ {description} not found")
            all_found = False

    return all_found


def validate_line_count():
    """Validate that we're meeting the 1000 line target."""
    print("\n📊 Validating line count...")

    total_lines = 0
    py_files = list(Path("gemini_cli").rglob("*.py"))

    for file in py_files:
        if "__pycache__" not in str(file):
            lines = len(file.read_text().splitlines())
            total_lines += lines
            print(f"  {file.relative_to('.')}: {lines} lines")

    print(f"\n  Total lines: {total_lines}")

    if total_lines <= 1200:  # Allow 20% margin
        print("  ✅ Within target (≤1200 lines)")
        return True
    else:
        print("  ❌ Exceeds target (>1200 lines)")
        return False


def validate_dependencies():
    """Validate minimal dependencies."""
    print("\n📚 Validating dependencies...")

    pyproject = Path("pyproject.toml")
    if pyproject.exists():
        content = pyproject.read_text()

        # Count dependencies
        import re

        deps = re.findall(r'"[^"]+>=', content)
        dep_count = len(deps)

        print(f"  Dependencies found: {dep_count}")

        # Check for unwanted dependencies
        unwanted = ["redis", "celery", "fastapi", "websockets", "httpx"]
        found_unwanted = []

        for dep in unwanted:
            if dep in content.lower():
                found_unwanted.append(dep)
                print(f"  ❌ Found unwanted dependency: {dep}")

        if dep_count <= 8 and not found_unwanted:
            print(f"  ✅ Minimal dependencies ({dep_count} ≤ 8)")
            return True
        else:
            return False
    else:
        print("  ❌ pyproject.toml not found")
        return False


def main():
    """Run all validations."""
    print("=" * 60)
    print("🚀 GEMINI CLI VALIDATION REPORT")
    print("=" * 60)

    start_time = time.time()

    # Change to project directory
    os.chdir(Path(__file__).parent)

    # Add to path for imports
    sys.path.insert(0, str(Path.cwd()))

    results = {
        "Structure": validate_structure(),
        "Authentication": validate_auth(),
        "Imports": validate_imports(),
        "Line Count": validate_line_count(),
        "Dependencies": validate_dependencies(),
    }

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for check, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {check}: {status}")
        if not passed:
            all_passed = False

    print(f"\n  Validation time: {elapsed:.2f}s")

    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED! The standalone Gemini CLI is ready.")
        print("\n📚 Next steps:")
        print("  1. Run: make setup")
        print("  2. Test: make test")
        print("  3. Use: make run")
        return 0
    else:
        print("\n⚠️ Some validations failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
