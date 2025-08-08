#!/usr/bin/env python3
"""MCP Server Implementation Validator for Gterminal.

Validates MCP server implementations for compliance, performance, and best practices.
Adapted from my-fullstack-agent for gterminal project structure.
"""

import ast
import importlib.util
from pathlib import Path
import sys
from typing import Any


class MCPServerValidator:
    """Comprehensive MCP server implementation validator for gterminal."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.metrics: dict[str, Any] = {}

    def validate_server_file(self, server_file: Path) -> bool:
        """Validate a single MCP server implementation file."""
        print(f"🔍 Validating MCP server: {server_file}")

        if not server_file.exists():
            self.errors.append(f"Server file not found: {server_file}")
            return False

        success = True

        # Static code analysis
        success &= self._validate_syntax(server_file)
        success &= self._validate_imports(server_file)
        success &= self._validate_mcp_compliance(server_file)
        success &= self._validate_error_handling(server_file)
        success &= self._validate_async_patterns(server_file)
        success &= self._validate_security(server_file)

        # Dynamic validation (if possible)
        if success:
            success &= self._validate_runtime(server_file)

        return success

    def _validate_syntax(self, server_file: Path) -> bool:
        """Validate Python syntax and basic structure."""
        try:
            with open(server_file, encoding="utf-8") as f:
                source = f.read()

            # Parse AST
            tree = ast.parse(source, filename=str(server_file))
            print("  ✅ Syntax validation passed")

            # Store AST for further analysis
            self._ast_tree = tree
            return True

        except SyntaxError as e:
            self.errors.append(f"Syntax error in {server_file}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error parsing {server_file}: {e}")
            return False

    def _validate_imports(self, server_file: Path) -> bool:
        """Validate import statements and dependencies."""
        success = True
        required_imports = {"mcp", "asyncio"}  # Core MCP requirements
        recommended_imports = {"pydantic", "logging"}
        found_imports = set()

        # Extract imports from AST
        for node in ast.walk(self._ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found_imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                found_imports.add(node.module.split(".")[0])

        # Check required imports
        missing_imports = required_imports - found_imports
        if missing_imports:
            self.warnings.append(f"Missing recommended imports: {missing_imports}")

        # Check for recommended imports
        missing_recommended = recommended_imports - found_imports
        if missing_recommended:
            self.warnings.append(f"Missing recommended imports: {missing_recommended}")

        # Check for problematic imports
        problematic_imports = {"eval", "exec", "compile", "__import__"}
        dangerous_imports = found_imports & problematic_imports
        if dangerous_imports:
            self.warnings.append(f"Potentially dangerous imports: {dangerous_imports}")

        print(f"  📦 Found {len(found_imports)} imports")
        return success

    def _validate_mcp_compliance(self, server_file: Path) -> bool:
        """Validate MCP protocol compliance for gterminal."""
        success = True

        # Check for MCP server patterns
        has_mcp_server = False
        has_tools = False
        has_error_handling = False
        tool_count = 0

        for node in ast.walk(self._ast_tree):
            # Check for MCP server initialization
            if isinstance(node, ast.Call):
                if (hasattr(node.func, "attr") and "MCP" in str(node.func.attr)) or (
                    hasattr(node.func, "id") and "mcp" in str(getattr(node.func, "id", "")).lower()
                ):
                    has_mcp_server = True

            # Check for tool decorators or registrations
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if (hasattr(decorator, "attr") and decorator.attr in ["tool", "resource"]) or (
                        hasattr(decorator, "id") and decorator.id in ["tool", "resource"]
                    ):
                        has_tools = True
                        tool_count += 1

            # Check for error handling
            if isinstance(node, ast.ExceptHandler):
                has_error_handling = True

        if not has_mcp_server:
            self.errors.append(f"No MCP server initialization found in {server_file}")
            success = False
        else:
            print("  ✅ MCP server initialization found")

        if not has_tools:
            self.warnings.append(f"No MCP tools found in {server_file}")
        else:
            print(f"  ✅ Found {tool_count} MCP tools")

        if not has_error_handling:
            self.warnings.append(f"Limited error handling in {server_file}")
        else:
            print("  ✅ Error handling found")

        return success

    def _validate_error_handling(self, server_file: Path) -> bool:
        """Validate error handling patterns."""
        success = True
        error_patterns = []

        for node in ast.walk(self._ast_tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:  # bare except
                    self.warnings.append(
                        f"Bare except clause found in {server_file} (line {node.lineno})"
                    )
                else:
                    error_patterns.append(node.type)

            # Check for proper async error handling
            if isinstance(node, ast.AsyncFunctionDef):
                has_try_except = any(isinstance(child, ast.Try) for child in ast.walk(node))
                if not has_try_except and node.name.startswith("handle_"):
                    self.warnings.append(f"Async handler {node.name} lacks error handling")

        print(f"  🛡️  Found {len(error_patterns)} error handling patterns")
        return success

    def _validate_async_patterns(self, server_file: Path) -> bool:
        """Validate async/await patterns and concurrency."""
        success = True
        async_functions = []
        blocking_calls = []

        for node in ast.walk(self._ast_tree):
            if isinstance(node, ast.AsyncFunctionDef):
                async_functions.append(node.name)

            # Check for potentially blocking calls in async functions
            if isinstance(node, ast.Call) and hasattr(node.func, "attr"):
                func_name = node.func.attr
                if func_name in ["sleep", "time", "input", "open"] and not hasattr(node.func, "id"):
                    blocking_calls.append(func_name)

        if async_functions:
            print(f"  ⚡ Found {len(async_functions)} async functions")
        else:
            self.warnings.append("No async functions found - may impact performance")

        if blocking_calls:
            self.warnings.append(f"Potentially blocking calls in async code: {set(blocking_calls)}")

        return success

    def _validate_security(self, server_file: Path) -> bool:
        """Validate security patterns and potential vulnerabilities."""
        success = True
        security_issues = []

        source_code = server_file.read_text(encoding="utf-8")

        # Check for common security issues
        dangerous_patterns = [
            ("eval(", "Use of eval() function"),
            ("exec(", "Use of exec() function"),
            ("__import__(", "Dynamic imports"),
            ("shell=True", "Shell command execution"),
            ("os.system(", "System command execution"),
            ("subprocess.call(", "Subprocess without shell=False"),
        ]

        for pattern, description in dangerous_patterns:
            if pattern in source_code:
                security_issues.append(description)

        # Check for hardcoded secrets
        secret_patterns = ["password", "secret", "key", "token", "api_key"]

        lines = source_code.split("\n")
        for i, line in enumerate(lines, 1):
            if "=" in line and any(pattern in line.lower() for pattern in secret_patterns):
                # Check if it's a hardcoded value (not a variable reference)
                if '"' in line or "'" in line:
                    parts = line.split("=", 1)
                    if len(parts) > 1 and ('"' in parts[1] or "'" in parts[1]):
                        if not parts[1].strip().startswith(("os.environ", "${", "config.")):
                            security_issues.append(f"Potential hardcoded secret on line {i}")

        if security_issues:
            for issue in security_issues:
                self.warnings.append(f"Security concern in {server_file}: {issue}")
        else:
            print("  🔒 Security validation passed")

        return success

    def _validate_runtime(self, server_file: Path) -> bool:
        """Attempt runtime validation of the MCP server."""
        success = True

        try:
            # Try to import the module
            spec = importlib.util.spec_from_file_location("mcp_server", server_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)

                # Add to sys.modules temporarily
                sys.modules["mcp_server"] = module
                spec.loader.exec_module(module)

                # Basic runtime checks
                if hasattr(module, "__all__"):
                    print(f"  📋 Module exports: {len(module.__all__)} items")

                # Check for main function or server instance
                if hasattr(module, "main") or hasattr(module, "server") or hasattr(module, "app"):
                    print("  ✅ Runtime validation passed")
                else:
                    self.warnings.append(f"No main entry point found in {server_file}")

                # Cleanup
                if "mcp_server" in sys.modules:
                    del sys.modules["mcp_server"]

        except ImportError as e:
            self.warnings.append(f"Import error during runtime validation: {e}")
        except Exception as e:
            self.warnings.append(f"Runtime validation error: {e}")

        return success

    def validate_multiple_servers(self, server_files: list[Path]) -> bool:
        """Validate multiple MCP server files."""
        print(f"🚀 Validating {len(server_files)} MCP servers")
        print("=" * 60)

        overall_success = True

        for server_file in server_files:
            success = self.validate_server_file(server_file)
            overall_success &= success
            print()  # Spacing between servers

        return overall_success

    def print_results(self) -> bool:
        """Print validation results."""
        print("=" * 60)

        if self.errors:
            print("❌ Validation Errors:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("✅ All MCP servers are valid!")
        elif not self.errors:
            print("✅ MCP servers are valid (with warnings)")
        else:
            print("❌ MCP server validation failed")

        # Print metrics if available
        if self.metrics:
            print("\n📊 Metrics:")
            for key, value in self.metrics.items():
                print(f"  • {key}: {value}")

        return len(self.errors) == 0


def main() -> None:
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: python validate-mcp-servers.py <server_file> [server_file2] ...")
        print("Example: python validate-mcp-servers.py mcp/gemini_server.py")
        sys.exit(1)

    validator = MCPServerValidator()
    server_files = [Path(f) for f in sys.argv[1:]]

    validator.validate_multiple_servers(server_files)
    overall_success = validator.print_results()

    if overall_success:
        print("\n✅ Pre-commit server validation passed")
        sys.exit(0)
    else:
        print("\n❌ Pre-commit server validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
