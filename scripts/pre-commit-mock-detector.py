#!/usr/bin/env python3
"""Pre-commit hook for detecting mock tests and stub implementations.
Fails the commit if mock tests or stub implementations are found.
"""

from pathlib import Path
import re
import sys

# Patterns that indicate mock/stub/placeholder code
MOCK_PATTERNS = [
    r"@patch\s*\(",
    r"@mock\.",
    r"Mock\s*\(",
    r"MagicMock\s*\(",
    r"PropertyMock\s*\(",
    r"create_autospec\s*\(",
    r"mock\.patch",
    r"unittest\.mock",
    r"from\s+unittest\.mock\s+import",
    r"from\s+mock\s+import",
]

STUB_PATTERNS = [
    r"pass\s*#\s*TODO",
    r"pass\s*#\s*FIXME",
    r"raise\s+NotImplementedError",
    r"return\s+None\s*#\s*TODO",
    r'return\s+["\']stub',
    r'return\s+["\']placeholder',
    r"^\s*\.\.\.\s*$",
    r"# STUB:",
    r"# MOCK:",
    r"# PLACEHOLDER:",
]

# Allow mocks in conftest.py and fixture files
ALLOWED_MOCK_FILES = [
    "conftest.py",
    "fixtures.py",
    "test_fixtures.py",
]


def check_file_for_mocks(filepath: Path) -> list[str]:
    """Check a file for mock patterns."""
    issues = []

    # Allow mocks in conftest and fixture files
    if filepath.name in ALLOWED_MOCK_FILES:
        return issues

    try:
        content = filepath.read_text()
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            for pattern in MOCK_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(f"{filepath}:{i}: Mock pattern found: {pattern}")
                    break
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)

    return issues


def check_file_for_stubs(filepath: Path) -> list[str]:
    """Check a file for stub implementations."""
    issues = []

    # Skip test files for stub checking (only check actual implementations)
    if "test" in filepath.name.lower():
        return issues

    try:
        content = filepath.read_text()
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            for pattern in STUB_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(f"{filepath}:{i}: Stub pattern found: {pattern}")
                    break
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)

    return issues


def main() -> int:
    """Main pre-commit hook entry point."""
    files = sys.argv[1:]

    if not files:
        print("No files to check")
        return 0

    mock_issues = []
    stub_issues = []

    for filepath_str in files:
        filepath = Path(filepath_str)

        # Skip non-Python files
        if filepath.suffix != ".py":
            continue

        # Skip virtual environment and cache directories
        if any(part in filepath.parts for part in [".venv", "__pycache__", ".pytest_cache"]):
            continue

        # Check for mocks in test files (except allowed files)
        if "test" in filepath.name.lower():
            mock_issues.extend(check_file_for_mocks(filepath))

        # Check for stubs in all Python files
        stub_issues.extend(check_file_for_stubs(filepath))

    # Report findings
    if mock_issues or stub_issues:
        print("\n❌ Mock/Stub Detection Failed!\n", file=sys.stderr)

        if mock_issues:
            print("🎭 Mock Tests Detected (NOT ALLOWED):", file=sys.stderr)
            for issue in mock_issues[:10]:  # Show first 10
                print(f"  • {issue}", file=sys.stderr)
            if len(mock_issues) > 10:
                print(f"  ... and {len(mock_issues) - 10} more", file=sys.stderr)
            print("\n✅ Fix: Replace mock tests with real integration tests", file=sys.stderr)
            print(
                "✅ Exception: Mocks are allowed in conftest.py and fixture files", file=sys.stderr
            )

        if stub_issues:
            print("\n📝 Stub Implementations Detected:", file=sys.stderr)
            for issue in stub_issues[:10]:  # Show first 10
                print(f"  • {issue}", file=sys.stderr)
            if len(stub_issues) > 10:
                print(f"  ... and {len(stub_issues) - 10} more", file=sys.stderr)
            print("\n✅ Fix: Implement all stub functions with real logic", file=sys.stderr)

        print("\n" + "=" * 60, file=sys.stderr)
        print("🚫 Commit blocked: Fix mock tests and stub implementations", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)

        return 1

    print("✅ No problematic mock tests or stub implementations detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
