"""Comprehensive test runner for gterminal consolidation validation.

This script provides a comprehensive test runner that validates the successful
consolidation from gapp to gterminal, with detailed reporting and validation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


class ConsolidationTestRunner:
    """Comprehensive test runner for consolidation validation."""

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialize the test runner."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_results = {}
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the test runner."""
        logger = logging.getLogger("consolidation_test_runner")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def run_import_tests(self) -> dict[str, Any]:
        """Run import validation tests."""
        self.logger.info("Running import validation tests...")

        start_time = time.time()

        # Run pytest for import tests specifically
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.project_root / "tests" / "test_imports.py"),
            "-v",
            "--tb=short",
            "--no-cov",
        ]

        try:
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, cwd=self.project_root
            )

            end_time = time.time()
            duration = end_time - start_time

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as e:
            return {"status": "error", "duration": time.time() - start_time, "error": str(e)}

    def run_consolidation_tests(self) -> dict[str, Any]:
        """Run consolidation validation tests."""
        self.logger.info("Running consolidation validation tests...")

        start_time = time.time()

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.project_root / "tests" / "test_consolidation.py"),
            "-v",
            "--tb=short",
            "--no-cov",
            "-m",
            "not slow",
        ]

        try:
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, cwd=self.project_root
            )

            end_time = time.time()
            duration = end_time - start_time

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as e:
            return {"status": "error", "duration": time.time() - start_time, "error": str(e)}

    def run_integration_tests(self) -> dict[str, Any]:
        """Run integration tests."""
        self.logger.info("Running integration tests...")

        start_time = time.time()

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.project_root / "tests" / "test_integration.py"),
            "-v",
            "--tb=short",
            "--no-cov",
            "-m",
            "integration and not slow",
        ]

        try:
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, cwd=self.project_root
            )

            end_time = time.time()
            duration = end_time - start_time

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as e:
            return {"status": "error", "duration": time.time() - start_time, "error": str(e)}

    def run_quick_validation(self) -> dict[str, Any]:
        """Run quick validation tests."""
        self.logger.info("Running quick validation...")

        validation_results = {
            "structure_check": self._check_project_structure(),
            "import_syntax_check": self._check_import_syntax(),
            "legacy_references_check": self._check_legacy_references(),
        }

        return validation_results

    def _check_project_structure(self) -> dict[str, Any]:
        """Check that expected project structure exists."""
        expected_dirs = [
            "gterminal/agents",
            "gterminal/auth",
            "gterminal/cache",
            "gterminal/core",
            "gterminal/terminal",
            "gterminal/gemini_cli",
            "gterminal/utils",
            "gterminal/mcp_servers",
        ]

        missing_dirs = []
        existing_dirs = []

        for dir_path in expected_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                existing_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)

        return {
            "status": "passed"
            if len(missing_dirs) == 0
            else "warning"
            if len(missing_dirs) <= 2
            else "failed",
            "existing_dirs": existing_dirs,
            "missing_dirs": missing_dirs,
            "total_expected": len(expected_dirs),
            "total_existing": len(existing_dirs),
        }

    def _check_import_syntax(self) -> dict[str, Any]:
        """Check for basic import syntax issues."""
        gterminal_path = self.project_root / "gterminal"
        python_files = list(gterminal_path.rglob("*.py"))

        syntax_errors = []
        import_errors = []

        for py_file in python_files:
            if py_file.name.startswith("test_"):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")

                # Check for basic syntax issues
                try:
                    compile(content, str(py_file), "exec")
                except SyntaxError as e:
                    syntax_errors.append(
                        {"file": str(py_file.relative_to(self.project_root)), "error": str(e)}
                    )

                # Check for obvious import issues
                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line.startswith("from ") or line.startswith("import "):
                        # Check for incomplete imports
                        if line.endswith(" from") or line.endswith(" import"):
                            import_errors.append(
                                {
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": line_num,
                                    "content": line,
                                }
                            )

            except UnicodeDecodeError:
                continue
            except Exception as e:
                syntax_errors.append(
                    {
                        "file": str(py_file.relative_to(self.project_root)),
                        "error": f"Could not process file: {e}",
                    }
                )

        return {
            "status": "passed" if len(syntax_errors) == 0 and len(import_errors) == 0 else "failed",
            "syntax_errors": syntax_errors,
            "import_errors": import_errors,
            "files_checked": len(python_files),
        }

    def _check_legacy_references(self) -> dict[str, Any]:
        """Check for legacy gapp/app references."""
        gterminal_path = self.project_root / "gterminal"
        python_files = list(gterminal_path.rglob("*.py"))

        legacy_references = []

        for py_file in python_files:
            if py_file.name.startswith("test_"):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    if line.strip().startswith("#"):
                        continue

                    # Check for legacy imports
                    if (
                        "from gapp." in line
                        or "import gapp" in line
                        or ("from app." in line and "gapp" not in line)
                    ):
                        legacy_references.append(
                            {
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": line_num,
                                "content": line.strip(),
                            }
                        )

            except UnicodeDecodeError:
                continue

        return {
            "status": "passed"
            if len(legacy_references) == 0
            else "warning"
            if len(legacy_references) <= 3
            else "failed",
            "legacy_references": legacy_references,
            "total_references": len(legacy_references),
        }

    def run_coverage_analysis(self) -> dict[str, Any]:
        """Run test coverage analysis."""
        self.logger.info("Running coverage analysis...")

        start_time = time.time()

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--cov=gterminal",
            "--cov-report=json",
            "--cov-report=term-missing",
            "--cov-fail-under=85",
            str(self.project_root / "tests"),
            "-m",
            "not slow and not requires_api_key",
        ]

        try:
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, cwd=self.project_root
            )

            # Try to read coverage report
            coverage_data = None
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                try:
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                except Exception:
                    pass

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "duration": time.time() - start_time,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "coverage_data": coverage_data,
            }
        except Exception as e:
            return {"status": "error", "duration": time.time() - start_time, "error": str(e)}

    def run_mcp_compliance_tests(self) -> dict[str, Any]:
        """Run MCP protocol compliance tests."""
        self.logger.info("Running MCP compliance tests...")

        start_time = time.time()

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.project_root / "tests"),
            "-v",
            "--tb=short",
            "--no-cov",
            "-m",
            "mcp",
        ]

        try:
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, cwd=self.project_root
            )

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "duration": time.time() - start_time,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as e:
            return {"status": "error", "duration": time.time() - start_time, "error": str(e)}

    def generate_report(self, results: dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report_lines = [
            "=" * 80,
            "GTERMINAL CONSOLIDATION TEST REPORT",
            "=" * 80,
            f"Test run completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Project root: {self.project_root}",
            "",
        ]

        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get("status") == "passed")
        failed_tests = sum(1 for r in results.values() if r.get("status") == "failed")
        warning_tests = sum(1 for r in results.values() if r.get("status") == "warning")
        error_tests = sum(1 for r in results.values() if r.get("status") == "error")

        report_lines.extend(
            [
                "SUMMARY",
                "-" * 40,
                f"Total test suites: {total_tests}",
                f"Passed: {passed_tests}",
                f"Failed: {failed_tests}",
                f"Warnings: {warning_tests}",
                f"Errors: {error_tests}",
                "",
            ]
        )

        # Detailed results
        for test_name, test_result in results.items():
            status = test_result.get("status", "unknown")
            duration = test_result.get("duration", 0)

            status_symbol = {
                "passed": "✅",
                "failed": "❌",
                "warning": "⚠️",
                "error": "💥",
                "unknown": "❓",
            }.get(status, "❓")

            report_lines.extend(
                [
                    f"{status_symbol} {test_name.upper().replace('_', ' ')}",
                    f"   Status: {status}",
                    f"   Duration: {duration:.2f}s",
                ]
            )

            # Add specific details
            if test_name == "quick_validation":
                for validation_name, validation_result in test_result.items():
                    if isinstance(validation_result, dict) and "status" in validation_result:
                        val_status = validation_result["status"]
                        val_symbol = {"passed": "✅", "failed": "❌", "warning": "⚠️"}.get(
                            val_status, "❓"
                        )
                        report_lines.append(f"     {val_symbol} {validation_name}: {val_status}")

            if test_result.get("stderr"):
                report_lines.extend(
                    [
                        "   Errors:",
                        "   " + "\n   ".join(test_result["stderr"].split("\n")[:5]),
                    ]
                )

            report_lines.append("")

        # Recommendations
        report_lines.extend(
            [
                "RECOMMENDATIONS",
                "-" * 40,
            ]
        )

        if failed_tests > 0:
            report_lines.append(
                "❌ CRITICAL: Some tests failed. Review errors above and fix issues."
            )
        elif warning_tests > 0:
            report_lines.append("⚠️  WARNING: Some tests have warnings. Consider addressing them.")
        else:
            report_lines.append("✅ SUCCESS: All tests passed! Consolidation appears successful.")

        if "coverage_analysis" in results:
            coverage_result = results["coverage_analysis"]
            if coverage_result.get("status") == "failed":
                report_lines.append(
                    "📊 Consider improving test coverage to meet the 85% requirement."
                )

        report_lines.extend(
            [
                "",
                "=" * 80,
            ]
        )

        return "\n".join(report_lines)

    def run_all_tests(self) -> dict[str, Any]:
        """Run all consolidation tests."""
        self.logger.info("Starting comprehensive consolidation test run...")

        all_results = {}

        # Run tests in order of importance
        test_suites = [
            ("quick_validation", self.run_quick_validation),
            ("import_tests", self.run_import_tests),
            ("consolidation_tests", self.run_consolidation_tests),
            ("integration_tests", self.run_integration_tests),
            ("coverage_analysis", self.run_coverage_analysis),
            ("mcp_compliance_tests", self.run_mcp_compliance_tests),
        ]

        for test_name, test_func in test_suites:
            self.logger.info(f"Running {test_name}...")
            try:
                result = test_func()
                all_results[test_name] = result

                status = result.get("status", "unknown")
                self.logger.info(f"{test_name} completed with status: {status}")

                # If critical tests fail, we might want to continue but note it
                if status == "failed" and test_name in ["import_tests", "consolidation_tests"]:
                    self.logger.warning(f"Critical test {test_name} failed, but continuing...")

            except Exception as e:
                self.logger.exception(f"Error running {test_name}: {e}")
                all_results[test_name] = {"status": "error", "error": str(e)}

        return all_results


def main() -> None:
    """Main entry point for the test runner."""
    project_root = Path(__file__).parent.parent

    runner = ConsolidationTestRunner(project_root)

    # Run all tests
    results = runner.run_all_tests()

    # Generate and display report
    report = runner.generate_report(results)
    print(report)

    # Save report to file
    report_file = project_root / "consolidation_test_report.md"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\nDetailed report saved to: {report_file}")

    # Exit with appropriate code
    failed_tests = sum(1 for r in results.values() if r.get("status") == "failed")
    error_tests = sum(1 for r in results.values() if r.get("status") == "error")

    if failed_tests > 0 or error_tests > 0:
        print("\n❌ Some tests failed. Please review the issues above.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
