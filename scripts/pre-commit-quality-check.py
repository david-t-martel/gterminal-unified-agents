#!/usr/bin/env python3
"""Pre-commit hook for checking code quality patterns.
Detects low-quality tests and code patterns.
"""

import ast
from pathlib import Path
import re
import sys

# Patterns that indicate poor test quality
POOR_QUALITY_PATTERNS = [
    (r"assert\s+True\s*$", "Meaningless assertion: assert True"),
    (r"assert\s+True\s*#", "Meaningless assertion with comment"),
    (r"assert\s+False\s*#\s*TODO", "Incomplete test with TODO"),
    (r"assert\s+1\s*==\s*1", "Trivial assertion: 1 == 1"),
    (r"self\.assertTrue\(True\)", "Meaningless assertTrue"),
    (r"self\.assertFalse\(False\)", "Meaningless assertFalse"),
    (r"def\s+test_\w+\(.*\):\s*pass", "Empty test function"),
]

# Code smell patterns
CODE_SMELL_PATTERNS = [
    (r"except\s*:\s*pass", "Bare except with pass (swallows all errors)"),
    (r"except\s+Exception\s*:\s*pass", "Generic exception with pass"),
    (r"# TODO:.*critical|urgent|important", "Unresolved critical TODO"),
    (r"# FIXME:.*critical|urgent|important", "Unresolved critical FIXME"),
    (r"print\s*\(.*\)\s*#\s*DEBUG", "Debug print statement left in code"),
    (r"console\.log\s*\(", "Console.log found (wrong language context)"),
]


def check_file_quality(filepath: Path) -> list[str]:
    """Check a file for quality issues."""
    issues = []

    try:
        content = filepath.read_text()
        lines = content.split("\n")

        # Check test files for poor quality patterns
        if "test" in filepath.name.lower():
            for i, line in enumerate(lines, 1):
                for pattern, description in POOR_QUALITY_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(f"{filepath}:{i}: {description}")

        # Check all files for code smells
        for i, line in enumerate(lines, 1):
            for pattern, description in CODE_SMELL_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(f"{filepath}:{i}: {description}")

        # AST-based checks
        try:
            tree = ast.parse(content)
            issues.extend(check_ast_quality(tree, filepath))
        except SyntaxError:
            pass  # Syntax errors will be caught by other tools

    except Exception as e:
        print(f"Error checking {filepath}: {e}", file=sys.stderr)

    return issues


def check_ast_quality(tree: ast.AST, filepath: Path) -> list[str]:
    """Check AST for quality issues."""
    issues = []

    for node in ast.walk(tree):
        # Check for functions that are too long
        if isinstance(node, ast.FunctionDef):
            if len(node.body) > 50:
                issues.append(
                    f"{filepath}:{node.lineno}: Function '{node.name}' is too long "
                    f"({len(node.body)} statements, max 50)",
                )

            # Check for too many arguments
            total_args = len(node.args.args) + len(node.args.kwonlyargs)
            if total_args > 8:  # Slightly more permissive for Gemini CLI
                issues.append(
                    f"{filepath}:{node.lineno}: Function '{node.name}' has too many arguments ({total_args}, max 8)",
                )

            # Check for empty test functions
            if node.name.startswith("test_") and len(node.body) == 1:
                if isinstance(node.body[0], ast.Pass):
                    issues.append(f"{filepath}:{node.lineno}: Empty test function '{node.name}'")
                elif (
                    isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and node.body[0].value.value == ...
                ):
                    issues.append(
                        f"{filepath}:{node.lineno}: Test function '{node.name}' contains only ellipsis"
                    )

        # Check for deeply nested code
        if isinstance(node, ast.If | ast.For | ast.While | ast.With):
            depth = get_nesting_depth(node)
            if depth > 4:
                issues.append(
                    f"{filepath}:{node.lineno}: Code is too deeply nested (depth {depth}, max 4)"
                )

    return issues


def get_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Get the maximum nesting depth of a node."""
    max_depth = current_depth

    for child in ast.walk(node):
        if isinstance(child, ast.If | ast.For | ast.While | ast.With | ast.Try):
            child_depth = get_nesting_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)

    return max_depth


def main() -> int:
    """Main pre-commit hook entry point."""
    files = sys.argv[1:]

    if not files:
        print("No files to check")
        return 0

    all_issues = []

    for filepath_str in files:
        filepath = Path(filepath_str)

        # Skip non-Python files
        if filepath.suffix != ".py":
            continue

        # Skip virtual environment and cache directories
        if any(part in filepath.parts for part in [".venv", "__pycache__", ".pytest_cache"]):
            continue

        issues = check_file_quality(filepath)
        all_issues.extend(issues)

    # Report findings
    if all_issues:
        print("\n⚠️  Code Quality Check Found Issues!\n", file=sys.stderr)

        for issue in all_issues[:20]:  # Show first 20
            print(f"  • {issue}", file=sys.stderr)

        if len(all_issues) > 20:
            print(f"  ... and {len(all_issues) - 20} more issues", file=sys.stderr)

        print("\n" + "=" * 60, file=sys.stderr)
        print("💡 Recommendations:", file=sys.stderr)
        print("  1. Replace trivial assertions with meaningful tests", file=sys.stderr)
        print("  2. Implement empty test functions", file=sys.stderr)
        print("  3. Handle exceptions properly", file=sys.stderr)
        print("  4. Reduce function complexity", file=sys.stderr)
        print("  5. Resolve critical TODOs", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)

        # Quality checks are warnings, not failures (return 0)
        # Change to return 1 if you want to block commits
        return 0

    print("✅ Code quality checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
