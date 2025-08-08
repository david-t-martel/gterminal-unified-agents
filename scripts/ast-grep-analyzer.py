#!/usr/bin/env python3
"""High-Performance Code Analysis using ast-grep for Claude Auto-Fix.

This utility demonstrates using ast-grep for ultra-fast code analysis and transformation
as a replacement for slower Python AST operations.

Usage:
    uv run python scripts/ast-grep-analyzer.py [file_or_directory]
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

# Color codes
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


class HighPerformanceASTAnalyzer:
    """Ultra-fast AST analysis using ast-grep."""

    def __init__(self):
        self.patterns = {
            "missing_return_types": {
                "pattern": "def $FUNC($$$):",
                "description": "Functions missing return type annotations",
            },
            "untyped_assignments": {
                "pattern": "$VAR = $VALUE",
                "description": "Variable assignments without type hints",
            },
            "missing_docstrings": {
                "pattern": "def $FUNC($$$): $BODY",
                "description": "Functions without docstrings",
            },
            "except_bare": {"pattern": "except:", "description": "Bare except clauses"},
            "string_formatting": {
                "pattern": '"%s" % $VAR',
                "description": "Old-style string formatting",
            },
            "class_definitions": {
                "pattern": "class $CLASS($$$): $BODY",
                "description": "Class definitions",
            },
            "import_statements": {
                "pattern": "import $MODULE",
                "description": "Import statements",
            },
            "function_calls": {
                "pattern": "$FUNC($$$)",
                "description": "Function calls",
            },
        }

    def analyze_file(self, file_path: str) -> dict[str, list[dict[str, Any]]]:
        """Analyze a single file using ast-grep."""
        results = {}

        for rule_name, rule_config in self.patterns.items():
            try:
                cmd = [
                    "ast-grep",
                    "--lang",
                    "python",
                    "--pattern",
                    rule_config["pattern"],
                    "--json",
                    file_path,
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, check=False)

                if result.returncode == 0 and result.stdout.strip():
                    matches = json.loads(result.stdout)
                    if matches:
                        results[rule_name] = {
                            "description": rule_config["description"],
                            "matches": matches,
                            "count": len(matches),
                        }

            except (json.JSONDecodeError, subprocess.SubprocessError) as e:
                print(f"{YELLOW}Warning: ast-grep failed for {rule_name}: {e}{RESET}")

        return results

    def analyze_directory(self, directory: str) -> dict[str, dict[str, Any]]:
        """Analyze all Python files in a directory."""
        python_files = list(Path(directory).rglob("*.py"))
        all_results = {}

        print(f"{BLUE}Analyzing {len(python_files)} Python files with ast-grep...{RESET}")

        for python_file in python_files:
            if any(skip in str(python_file) for skip in ["__pycache__", ".git", ".venv", "venv"]):
                continue

            file_results = self.analyze_file(str(python_file))
            if file_results:
                all_results[str(python_file)] = file_results

        return all_results

    def generate_fix_suggestions(self, results: dict[str, Any]) -> list[str]:
        """Generate fix suggestions based on analysis results."""
        suggestions = []

        for file_path, file_results in results.items():
            suggestions.append(f"\n{CYAN}File: {file_path}{RESET}")

            for rule_name, rule_data in file_results.items():
                count = rule_data["count"]
                description = rule_data["description"]

                suggestions.append(f"  {YELLOW}⚠️  {description}: {count} issues{RESET}")

                # Specific suggestions based on rule type
                if rule_name == "missing_return_types":
                    suggestions.append("    💡 Add return type annotations to functions")
                elif rule_name == "except_bare":
                    suggestions.append("    💡 Replace bare except with specific exceptions")
                elif rule_name == "string_formatting":
                    suggestions.append("    💡 Use f-strings or .format() instead of % formatting")
                elif rule_name == "missing_docstrings":
                    suggestions.append("    💡 Add docstrings to functions and classes")

        return suggestions

    def export_for_claude(self, results: dict[str, Any]) -> str:
        """Export results in a format suitable for Claude auto-fix."""
        claude_report = []
        claude_report.append("CODE ANALYSIS REPORT (Generated by ast-grep)")
        claude_report.append("=" * 50)

        total_issues = 0
        for file_path, file_results in results.items():
            claude_report.append(f"\nFile: {file_path}")

            for _rule_name, rule_data in file_results.items():
                count = rule_data["count"]
                total_issues += count
                description = rule_data["description"]

                claude_report.append(f"  - {description}: {count} issues")

                # Include specific line numbers from matches
                for match in rule_data["matches"][:5]:  # Limit to first 5 matches
                    line_info = match.get("range", {}).get("start", {})
                    line_no = line_info.get("line", "unknown")
                    claude_report.append(f"    Line {line_no}")

        claude_report.append(f"\nTotal issues found: {total_issues}")
        claude_report.append("\nRecommendations:")
        claude_report.append("- Add type annotations where missing")
        claude_report.append("- Replace bare except clauses with specific exceptions")
        claude_report.append("- Add docstrings to functions and classes")
        claude_report.append("- Use modern string formatting (f-strings)")

        return "\n".join(claude_report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="High-performance AST analysis using ast-grep")
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="File or directory to analyze (default: current directory)",
    )
    parser.add_argument(
        "--export-claude",
        action="store_true",
        help="Export results for Claude auto-fix",
    )
    parser.add_argument("--pattern", help="Custom ast-grep pattern to search for")

    args = parser.parse_args()

    # Check if ast-grep is available
    try:
        subprocess.run(["ast-grep", "--version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print(f"{RED}Error: ast-grep not found. Please install ast-grep first.{RESET}")
        print("Installation: cargo install ast-grep")
        return 1

    analyzer = HighPerformanceASTAnalyzer()

    print(f"{GREEN}🚀 HIGH-PERFORMANCE AST ANALYSIS{RESET}")
    print(f"{BLUE}Using ast-grep for ultra-fast code analysis{RESET}")

    # Custom pattern search
    if args.pattern:
        print(f"\n{CYAN}Searching for custom pattern: {args.pattern}{RESET}")
        try:
            cmd = [
                "ast-grep",
                "--lang",
                "python",
                "--pattern",
                args.pattern,
                args.target,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.stdout:
                print(result.stdout)
            else:
                print("No matches found")
        except Exception as e:
            print(f"{RED}Error running custom pattern: {e}{RESET}")
        return 0

    # Standard analysis
    target_path = Path(args.target)

    if target_path.is_file():
        results = {str(target_path): analyzer.analyze_file(str(target_path))}
    elif target_path.is_dir():
        results = analyzer.analyze_directory(str(target_path))
    else:
        print(f"{RED}Error: {args.target} is not a valid file or directory{RESET}")
        return 1

    if not results:
        print(f"{GREEN}✅ No issues found!{RESET}")
        return 0

    # Generate and display suggestions
    suggestions = analyzer.generate_fix_suggestions(results)
    print(f"\n{GREEN}=== ANALYSIS RESULTS ==={RESET}")
    for suggestion in suggestions:
        print(suggestion)

    # Export for Claude if requested
    if args.export_claude:
        claude_report = analyzer.export_for_claude(results)
        output_file = "ast-analysis-for-claude.txt"
        Path(output_file).write_text(claude_report, encoding="utf-8")
        print(f"\n{CYAN}📄 Claude report exported to: {output_file}{RESET}")

    total_files = len(results)
    total_issues = sum(
        sum(rule_data["count"] for rule_data in file_results.values())
        for file_results in results.values()
    )

    print(f"\n{GREEN}⚡ Analysis complete:{RESET}")
    print(f"  Files analyzed: {total_files}")
    print(f"  Issues found: {total_issues}")
    print("  Powered by ast-grep for maximum performance!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
