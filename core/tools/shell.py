#!/usr/bin/env python3
"""Shell Execution Tools - Secure command execution using rust-fs MCP server.

This module provides secure command execution through the rust-fs MCP server
or falls back to rust-exec/claude-exec for enhanced security features.
"""

import asyncio
import logging
import os
from pathlib import Path
import shlex
from typing import Any

from gterminal.core.tools.registry import BaseTool
from gterminal.core.tools.registry import ToolParameter
from gterminal.core.tools.registry import ToolResult
from gterminal.gemini_agents.utils.file_ops.rust_fs_integration import RustFsClient

logger = logging.getLogger(__name__)

# Path to rust-exec binary as secondary option
RUST_EXEC_PATH = Path("/home/david/.claude/rust-exec/target/release/claude-exec")


class SecureCommandExecutor:
    """Secure command execution using rust-fs MCP server with rust-exec fallback."""

    def __init__(self, use_rust_fs: bool = True) -> None:
        self.use_rust_fs = use_rust_fs
        self.rust_fs_client = RustFsClient(fallback_to_python=True) if use_rust_fs else None
        self.rust_exec = str(RUST_EXEC_PATH) if RUST_EXEC_PATH.exists() else None

    async def execute_secure(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        timeout: int = 30,
        use_ai_mode: bool = False,
    ) -> dict[str, Any]:
        """Execute command securely using rust-fs or rust-exec.

        Args:
            command: Base command to execute
            args: Command arguments (safer than shell mode)
            cwd: Working directory
            timeout: Timeout in seconds
            use_ai_mode: Use AI-optimized context management

        Returns:
            Dictionary with execution results

        """
        # Try rust-fs first if available
        if self.use_rust_fs and self.rust_fs_client:
            try:
                result = await self.rust_fs_client.execute_command(
                    command=command,
                    args=args,
                    timeout=timeout,
                    cwd=cwd,
                )

                # Normalize result format
                if "success" not in result:
                    result["success"] = result.get("return_code", 1) == 0

                return result
            except Exception as e:
                logger.warning(f"rust-fs execution failed, trying rust-exec: {e}")

        # Fallback to rust-exec if available
        if self.rust_exec and os.path.exists(self.rust_exec):
            return await self._execute_with_rust_exec(command, args, cwd, timeout, use_ai_mode)

        # Final fallback to Python subprocess
        return await self._execute_with_python(command, args, cwd, timeout)

    async def _execute_with_rust_exec(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        timeout: int = 30,
        use_ai_mode: bool = False,
    ) -> dict[str, Any]:
        """Execute command using rust-exec/claude-exec."""
        # Build rust-exec command
        rust_command = [self.rust_exec]

        if use_ai_mode:
            rust_command.extend(["ai", "execute"])
        else:
            rust_command.append("execute")

        # Add timeout
        rust_command.extend(["--timeout", str(timeout)])

        # Add working directory if specified
        if cwd:
            rust_command.extend(["--dir", cwd])

        # Add the actual command
        rust_command.append("--")
        rust_command.append(command)

        # Add arguments if provided
        if args:
            rust_command.extend(args)

        logger.debug(f"Executing with rust-exec: {rust_command}")

        # Execute through rust-exec
        process = await asyncio.create_subprocess_exec(
            *rust_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._get_secure_env(),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout + 5,  # Give rust-exec time to enforce its own timeout
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "return_code": -1,
            }

        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "return_code": process.returncode,
        }

    async def _execute_with_python(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Python subprocess fallback."""
        cmd_list = [command]
        if args:
            cmd_list.extend(args)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=self._get_secure_env(),
            )

            if timeout:
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    return {
                        "success": False,
                        "stdout": "",
                        "stderr": f"Command timed out after {timeout} seconds",
                        "return_code": -1,
                    }
            else:
                stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "return_code": process.returncode,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
            }

    def _get_secure_env(self) -> dict[str, str]:
        """Get secure environment variables."""
        # Start with minimal environment
        secure_env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "TERM": os.environ.get("TERM", "xterm"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "USER": os.environ.get("USER", "nobody"),
        }

        # Add Python/UV specific paths if needed
        if "VIRTUAL_ENV" in os.environ:
            secure_env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
            # Add venv bin to PATH
            venv_bin = os.path.join(os.environ["VIRTUAL_ENV"], "bin")
            secure_env["PATH"] = f"{venv_bin}:{secure_env['PATH']}"

        return secure_env


class ExecuteCommandTool(BaseTool):
    """Tool for executing shell commands securely using rust-fs MCP server."""

    def __init__(self) -> None:
        super().__init__(
            name="execute_command",
            description="Execute a shell command securely using rust-fs or rust-exec",
            category="shell",
        )
        self.executor = SecureCommandExecutor(use_rust_fs=True)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="Command to execute (without arguments for better security)",
                required=True,
            ),
            ToolParameter(
                name="args",
                type="array",
                description="Command arguments as array (safer than shell parsing)",
                required=False,
                default=[],
            ),
            ToolParameter(
                name="cwd",
                type="string",
                description="Working directory",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Timeout in seconds",
                required=False,
                default=30,
            ),
            ToolParameter(
                name="use_ai_mode",
                type="boolean",
                description="Use AI-optimized context management for large outputs",
                required=False,
                default=False,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            command = params["command"]
            args = params.get("args", [])
            cwd = params.get("cwd")
            timeout = params.get("timeout", 30)
            use_ai_mode = params.get("use_ai_mode", False)

            # For backward compatibility, parse command if args not provided
            if not args and " " in command:
                parts = shlex.split(command)
                command = parts[0]
                args = parts[1:]

            # Execute securely through rust-exec
            result = await self.executor.execute_secure(
                command=command,
                args=args,
                cwd=cwd,
                timeout=timeout,
                use_ai_mode=use_ai_mode,
            )

            return ToolResult(
                success=result["success"],
                data={
                    "command": f"{command} {' '.join(args)}".strip(),
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                    "return_code": result["return_code"],
                },
                error=result["stderr"] if not result["success"] else None,
            )

        except Exception as e:
            logger.exception(f"Command execution failed: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
            )


class InstallDependenciesTool(BaseTool):
    """Tool for installing project dependencies using UV through rust-exec."""

    def __init__(self) -> None:
        super().__init__(
            name="install_dependencies",
            description="Install project dependencies using UV securely",
            category="shell",
        )
        self.executor = SecureCommandExecutor()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="package",
                type="string",
                description="Package to install (optional, installs all if not specified)",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="dev",
                type="boolean",
                description="Install dev dependencies",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="extras",
                type="string",
                description="Extras to install (comma-separated)",
                required=False,
                default=None,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            package = params.get("package")
            dev = params.get("dev", False)
            extras = params.get("extras")

            # Build UV command and args
            if package:
                # Install specific package
                command = "uv"
                args = ["pip", "install", package]
            else:
                # Install from pyproject.toml
                command = "uv"
                args = ["sync"]
                if dev:
                    args.append("--all-extras")
                elif extras:
                    for extra in extras.split(","):
                        args.extend(["--extra", extra.strip()])

            # Execute installation securely
            result = await self.executor.execute_secure(
                command=command,
                args=args,
                timeout=120,  # Longer timeout for installations
            )

            return ToolResult(
                success=result["success"],
                data={
                    "command": f"{command} {' '.join(args)}",
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                },
                error=result["stderr"] if not result["success"] else None,
            )

        except Exception as e:
            logger.exception(f"Dependency installation failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))


class RunTestsTool(BaseTool):
    """Tool for running tests with pytest through rust-exec."""

    def __init__(self) -> None:
        super().__init__(
            name="run_tests",
            description="Run tests using pytest securely",
            category="shell",
        )
        self.executor = SecureCommandExecutor()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to test file or directory",
                required=False,
                default="tests/",
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="Test pattern to match",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="coverage",
                type="boolean",
                description="Run with coverage",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="verbose",
                type="boolean",
                description="Verbose output",
                required=False,
                default=False,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            path = params.get("path", "tests/")
            pattern = params.get("pattern")
            coverage = params.get("coverage", True)
            verbose = params.get("verbose", False)

            # Build pytest command and args
            command = "uv"
            args = ["run", "pytest"]

            if pattern:
                args.extend(["-k", pattern])

            if coverage:
                args.extend(["--cov=app", "--cov-report=term-missing"])

            if verbose:
                args.append("-vv")

            args.append(path)

            # Execute tests securely
            result = await self.executor.execute_secure(
                command=command,
                args=args,
                timeout=300,  # 5 minutes for test runs
                use_ai_mode=True,  # Handle large test outputs
            )

            # Parse test results
            output = result["stdout"]
            passed = failed = skipped = 0

            for line in output.splitlines():
                if "passed" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "passed" in part and i > 0:
                                passed = int(parts[i - 1])
                            elif "failed" in part and i > 0:
                                failed = int(parts[i - 1])
                            elif "skipped" in part and i > 0:
                                skipped = int(parts[i - 1])
                    except Exception:
                        pass

            return ToolResult(
                success=result["success"],
                data={
                    "command": f"{command} {' '.join(args)}",
                    "output": output,
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "return_code": result["return_code"],
                },
                error=result["stderr"] if not result["success"] else None,
            )

        except Exception as e:
            logger.exception(f"Test execution failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))


class BuildProjectTool(BaseTool):
    """Tool for building the project securely using rust-exec."""

    def __init__(self) -> None:
        super().__init__(
            name="build_project",
            description="Build the project using make or other build tools securely",
            category="shell",
        )
        self.executor = SecureCommandExecutor()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="target",
                type="string",
                description="Build target (e.g., 'build', 'dev', 'release')",
                required=False,
                default="build",
            ),
            ToolParameter(
                name="clean",
                type="boolean",
                description="Clean before building",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="parallel",
                type="boolean",
                description="Use parallel build if available",
                required=False,
                default=True,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            target = params.get("target", "build")
            clean = params.get("clean", False)
            parallel = params.get("parallel", True)

            build_commands: list[Any] = []

            # Clean if requested
            if clean:
                build_commands.append(("make", ["clean"]))

            # Determine build command based on target
            if target == "dev":
                build_commands.append(("make", ["dev"]))
            elif target == "release":
                build_commands.append(("make", ["release"]))
            elif target == "rust":
                build_commands.append(("make", ["rust-build"]))
            elif target == "parallel" and parallel:
                build_commands.append(("make", ["build-parallel"]))
            else:
                build_commands.append(("make", [target]))

            results: list[Any] = []
            for cmd, args in build_commands:
                result = await self.executor.execute_secure(
                    command=cmd,
                    args=args,
                    timeout=600,  # 10 minutes for builds
                    use_ai_mode=True,  # Handle large build outputs
                )

                if not result["success"]:
                    return ToolResult(
                        success=False,
                        data={
                            "command": f"{cmd} {' '.join(args)}",
                            "error": result["stderr"],
                        },
                        error=f"Build failed at: {cmd} {' '.join(args)}",
                    )

                results.append(
                    {
                        "command": f"{cmd} {' '.join(args)}",
                        "output": result["stdout"],
                    },
                )

            return ToolResult(
                success=True,
                data={
                    "target": target,
                    "commands_executed": [
                        f"{cmd} {' '.join(args)}" for cmd, args in build_commands
                    ],
                    "results": results,
                },
            )

        except Exception as e:
            logger.exception(f"Build failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))


class SecureSystemInfoTool(BaseTool):
    """Tool for gathering system information securely using rust-exec."""

    def __init__(self) -> None:
        super().__init__(
            name="system_info",
            description="Get system information and resource usage securely",
            category="shell",
        )
        self.executor = SecureCommandExecutor()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="info_type",
                type="string",
                description="Type of info: all, cpu, memory, disk, network, process",
                required=False,
                default="all",
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            info_type = params.get("info_type", "all")

            commands = {
                "cpu": [
                    ("uname", ["-a"]),
                    ("lscpu", ["--parse=CPU,CORE,SOCKET,MAXMHZ"]),
                ],
                "memory": [
                    ("free", ["-h"]),
                    ("cat", ["/proc/meminfo"]),
                ],
                "disk": [
                    ("df", ["-h"]),
                    ("du", ["-sh", "."]),
                ],
                "network": [
                    ("hostname", ["-I"]),
                    ("ip", ["addr", "show"]),
                ],
                "process": [
                    ("ps", ["aux", "--sort=-pcpu"]),
                    ("top", ["-b", "-n", "1"]),
                ],
            }

            # Select commands based on info_type
            if info_type == "all":
                selected_commands: list[Any] = []
                for cmd_list in commands.values():
                    selected_commands.extend(cmd_list)
            else:
                selected_commands = commands.get(info_type, [])

            results: dict[str, Any] = {}
            for cmd, args in selected_commands:
                try:
                    result = await self.executor.execute_secure(
                        command=cmd,
                        args=args,
                        timeout=10,
                        use_ai_mode=True,  # Smart truncation for large outputs
                    )

                    if result["success"]:
                        key = f"{cmd}_{' '.join(args).replace(' ', '_')}"
                        results[key] = result["stdout"]
                except Exception as e:
                    logger.warning(f"Failed to execute {cmd}: {e}")
                    continue

            return ToolResult(
                success=True,
                data={
                    "info_type": info_type,
                    "results": results,
                },
            )

        except Exception as e:
            logger.exception(f"System info gathering failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))


class SecureCodeAnalysisTool(BaseTool):
    """Tool for secure code analysis using rust-exec with proper sandboxing."""

    def __init__(self) -> None:
        super().__init__(
            name="code_analysis",
            description="Analyze code files securely for quality, security, and performance",
            category="shell",
        )
        self.executor = SecureCommandExecutor()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the code file to analyze",
                required=True,
            ),
            ToolParameter(
                name="analysis_type",
                type="string",
                description="Type of analysis: lint, security, complexity, all",
                required=False,
                default="all",
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            file_path = params["file_path"]
            analysis_type = params.get("analysis_type", "all")

            # Determine file type
            if file_path.endswith(".py"):
                language = "python"
            elif file_path.endswith((".js", ".ts", ".jsx", ".tsx")):
                language = "javascript"
            elif file_path.endswith(".rs"):
                language = "rust"
            else:
                language = "unknown"

            analyses: dict[str, Any] = {}

            # Python analysis
            if language == "python" and analysis_type in ["lint", "all"]:
                # Ruff linting
                result = await self.executor.execute_secure(
                    command="ruff",
                    args=["check", file_path, "--output-format", "json"],
                    timeout=30,
                )
                if (
                    result["success"] or result["return_code"] == 1
                ):  # Ruff returns 1 if issues found
                    analyses["lint"] = result["stdout"]

                # MyPy type checking
                result = await self.executor.execute_secure(
                    command="mypy",
                    args=[file_path, "--no-error-summary"],
                    timeout=30,
                )
                analyses["type_check"] = result["stdout"]

            if language == "python" and analysis_type in ["security", "all"]:
                # Bandit security analysis
                result = await self.executor.execute_secure(
                    command="bandit",
                    args=["-r", file_path, "-f", "json"],
                    timeout=30,
                )
                if result["success"]:
                    analyses["security"] = result["stdout"]

            if language == "python" and analysis_type in ["complexity", "all"]:
                # Radon complexity analysis
                result = await self.executor.execute_secure(
                    command="radon",
                    args=["cc", file_path, "-j"],
                    timeout=30,
                )
                if result["success"]:
                    analyses["complexity"] = result["stdout"]

            # JavaScript/TypeScript analysis
            if language == "javascript" and analysis_type in ["lint", "all"]:
                # ESLint
                result = await self.executor.execute_secure(
                    command="npx",
                    args=["eslint", file_path, "--format", "json"],
                    timeout=30,
                )
                if result["success"] or result["return_code"] == 1:
                    analyses["lint"] = result["stdout"]

            # Rust analysis
            if language == "rust" and analysis_type in ["lint", "all"]:
                # Clippy
                result = await self.executor.execute_secure(
                    command="cargo",
                    args=["clippy", "--message-format=json"],
                    cwd=os.path.dirname(file_path),
                    timeout=60,
                )
                if result["success"]:
                    analyses["lint"] = result["stdout"]

            return ToolResult(
                success=True,
                data={
                    "file_path": file_path,
                    "language": language,
                    "analysis_type": analysis_type,
                    "analyses": analyses,
                },
            )

        except Exception as e:
            logger.exception(f"Code analysis failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))
