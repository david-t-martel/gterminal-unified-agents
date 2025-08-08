#!/usr/bin/env python3
"""MCP Configuration Validation Script.

Validates MCP configuration files for syntax, structure, and best practices.
Adapted for gterminal project structure.
"""

import json
import os
from pathlib import Path
import sys
from typing import Any

try:
    import jsonschema
    from jsonschema import validate

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    print("Warning: jsonschema not installed, using basic validation only")


class MCPConfigValidator:
    """Comprehensive MCP configuration validator for gterminal."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.schema = self._get_mcp_schema()

    def _get_mcp_schema(self) -> dict[str, Any]:
        """Get the MCP configuration JSON schema."""
        return {
            "type": "object",
            "required": ["mcpServers"],
            "properties": {
                "mcpServers": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z0-9-_]+$": {
                            "type": "object",
                            "required": ["command", "args"],
                            "properties": {
                                "command": {"type": "string", "minLength": 1},
                                "args": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                                "cwd": {"type": "string"},
                                "env": {
                                    "type": "object",
                                    "patternProperties": {".*": {"type": "string"}},
                                },
                                "disabled": {"type": "boolean"},
                                "timeout": {"type": "number", "minimum": 1},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "additionalProperties": False,
                },
                "version": {"type": "string"},
                "description": {"type": "string"},
            },
            "additionalProperties": False,
        }

    def validate_file(self, config_file: Path) -> bool:
        """Validate a single MCP configuration file."""
        print(f"🔍 Validating {config_file}")

        if not config_file.exists():
            self.errors.append(f"Configuration file not found: {config_file}")
            return False

        try:
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in {config_file}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading {config_file}: {e}")
            return False

        # Schema validation
        if HAS_JSONSCHEMA:
            try:
                validate(instance=config, schema=self.schema)
                print("  ✅ Schema validation passed")
            except jsonschema.ValidationError as e:
                self.errors.append(f"Schema validation failed for {config_file}: {e.message}")
                return False
        # Basic validation without jsonschema
        elif not self._basic_validation(config, config_file):
            return False

        # Detailed validation
        return self._validate_detailed(config, config_file)

    def _basic_validation(self, config: dict[str, Any], config_file: Path) -> bool:
        """Basic validation when jsonschema is not available."""
        if "mcpServers" not in config:
            self.errors.append(f"Missing 'mcpServers' key in {config_file}")
            return False

        if not isinstance(config["mcpServers"], dict):
            self.errors.append(f"'mcpServers' must be an object in {config_file}")
            return False

        print("  ✅ Basic structure validation passed")
        return True

    def _validate_detailed(self, config: dict[str, Any], config_file: Path) -> bool:
        """Perform detailed validation of MCP configuration."""
        success = True
        servers = config.get("mcpServers", {})

        if not servers:
            self.errors.append(f"No MCP servers defined in {config_file}")
            return False

        print(f"  📦 Validating {len(servers)} MCP servers")

        for server_name, server_config in servers.items():
            success &= self._validate_server(server_name, server_config, config_file)

        # Check for security issues
        self._check_security_issues(servers, config_file)

        return success

    def _validate_server(self, name: str, config: dict[str, Any], config_file: Path) -> bool:
        """Validate individual MCP server configuration."""
        success = True

        # Server name validation
        if not name.replace("-", "").replace("_", "").isalnum():
            self.errors.append(f"Invalid server name '{name}' in {config_file}")
            success = False

        # Command validation
        command = config.get("command", "")
        if not command:
            self.errors.append(f"Missing command for server '{name}' in {config_file}")
            success = False
        elif not os.path.exists(command) and not self._is_system_command(command):
            # Only warn for gterminal, as commands might be in virtual env
            self.warnings.append(f"Command not found: {command} for server '{name}'")

        # Args validation
        args = config.get("args", [])
        if not args:
            self.errors.append(f"Missing args for server '{name}' in {config_file}")
            success = False

        # Python module validation for gterminal
        if "-m" in args:
            try:
                module_idx = args.index("-m") + 1
                if module_idx < len(args):
                    module_name = args[module_idx]
                    success &= self._validate_python_module(name, module_name, config)
            except ValueError:
                pass

        # Environment validation
        env = config.get("env", {})
        success &= self._validate_environment(name, env, config_file)

        # Working directory validation
        cwd = config.get("cwd")
        if cwd and not os.path.exists(cwd):
            self.warnings.append(f"Working directory not found: {cwd} for server '{name}'")

        return success

    def _validate_python_module(
        self, server_name: str, module_name: str, config: dict[str, Any]
    ) -> bool:
        """Validate Python module exists for gterminal structure."""

        # Check common gterminal module paths
        gterminal_paths = [
            f"mcp/{module_name.replace('.', '/')}.py",
            f"gemini_cli/{module_name.replace('.', '/')}.py",
            f"{module_name.replace('.', '/')}.py",
        ]

        found = False
        for path in gterminal_paths:
            if os.path.exists(path):
                print(f"    ✅ Module found: {path}")
                found = True
                break

        if not found:
            # Try to import (basic check)
            try:
                import importlib.util

                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    self.warnings.append(
                        f"Python module not found: {module_name} for server '{server_name}'"
                    )
                else:
                    print(f"    ✅ Module importable: {module_name}")
                    found = True
            except Exception:
                self.warnings.append(f"Could not validate module: {module_name}")

        return found

    def _validate_environment(
        self, server_name: str, env: dict[str, str], config_file: Path
    ) -> bool:
        """Validate environment variables."""
        success = True
        required_vars = set()
        sensitive_vars = {
            "GOOGLE_APPLICATION_CREDENTIALS",
            "API_KEY",
            "SECRET",
            "TOKEN",
        }

        # Check for required variables based on server type
        if "gemini" in server_name.lower():
            required_vars.update(
                {
                    "GOOGLE_CLOUD_PROJECT",
                    "VERTEX_AI_REGION",
                }
            )

        # Check required variables
        for var in required_vars:
            if var not in env:
                self.warnings.append(
                    f"Missing recommended environment variable '{var}' for server '{server_name}'"
                )

        # Check sensitive variables
        for var, value in env.items():
            if any(sensitive in var.upper() for sensitive in sensitive_vars):
                if not value or value.startswith("${"):
                    # Good - using environment variable expansion
                    continue
                if os.path.exists(value):
                    # File path - check permissions
                    try:
                        stat = os.stat(value)
                        mode = stat.st_mode
                        if mode & 0o077:  # World or group readable
                            self.warnings.append(f"Sensitive file {value} has loose permissions")
                    except OSError:
                        pass
                else:
                    self.warnings.append(
                        f"Hardcoded sensitive value in {var} for server '{server_name}'"
                    )

        return success

    def _check_security_issues(self, servers: dict[str, dict[str, Any]], config_file: Path) -> None:
        """Check for security issues in configuration."""
        for server_name, config in servers.items():
            command = config.get("command", "")
            args = config.get("args", [])

            # Check for dangerous commands
            dangerous_commands = {"sudo", "su", "chmod", "chown"}
            if any(cmd in command for cmd in dangerous_commands):
                self.warnings.append(
                    f"Potentially dangerous command in server '{server_name}': {command}"
                )

            # Check for shell injection possibilities
            for arg in args:
                if any(char in arg for char in [";", "|", "&", "`", "$("]):
                    self.warnings.append(
                        f"Potential shell injection risk in server '{server_name}': {arg}"
                    )

    def _is_system_command(self, command: str) -> bool:
        """Check if command is available in system PATH."""
        import shutil

        return shutil.which(command) is not None

    def print_results(self) -> bool:
        """Print validation results."""
        print("\n" + "=" * 60)

        if self.errors:
            print("❌ Validation Errors:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("✅ All MCP configurations are valid!")
        elif not self.errors:
            print("✅ MCP configurations are valid (with warnings)")
        else:
            print("❌ MCP configuration validation failed")

        return len(self.errors) == 0


def main() -> None:
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: python validate-mcp-config.py <config_file> [config_file2] ...")
        print("Example: python validate-mcp-config.py mcp/.mcp.json")
        sys.exit(1)

    validator = MCPConfigValidator()
    success = True

    print("🚀 MCP Configuration Validator (Gterminal)")
    print("=" * 60)

    for config_file in sys.argv[1:]:
        config_path = Path(config_file)
        success &= validator.validate_file(config_path)

    # Print results and exit
    overall_success = validator.print_results()

    if overall_success:
        print("\n✅ Pre-commit validation passed")
        sys.exit(0)
    else:
        print("\n❌ Pre-commit validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
