"""
Unit tests for Rust-based tooling and test frameworks.
Tests MCP servers implemented in Rust and related infrastructure.
"""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestRustMCPServers:
    """Test Rust-based MCP servers (rust-fs, rust-fetch, rust-link)."""

    @pytest.fixture
    def rust_fs_config(self):
        """Configuration for rust-fs MCP server."""
        return {
            "name": "rust-fs",
            "command": "rust-fs",
            "args": [],
            "env": {},
            "transport": "stdio",
        }

    @pytest.fixture
    def rust_fetch_config(self):
        """Configuration for rust-fetch MCP server."""
        return {
            "name": "rust-fetch",
            "command": "rust-fetch",
            "args": [],
            "env": {},
            "transport": "stdio",
        }

    @pytest.fixture
    def rust_link_config(self):
        """Configuration for rust-link MCP server."""
        return {
            "name": "rust-link",
            "command": "rust-link",
            "args": [],
            "env": {},
            "transport": "stdio",
        }

    def test_rust_fs_executable_exists(self):
        """Test that rust-fs executable exists in PATH."""
        result = subprocess.run(["which", "rust-fs"], check=False, capture_output=True, text=True)
        assert result.returncode == 0, "rust-fs not found in PATH"
        assert "/rust-fs" in result.stdout

    def test_rust_fetch_executable_exists(self):
        """Test that rust-fetch executable exists in PATH."""
        result = subprocess.run(["which", "rust-fetch"], check=False, capture_output=True, text=True)
        assert result.returncode == 0, "rust-fetch not found in PATH"
        assert "/rust-fetch" in result.stdout

    @pytest.mark.asyncio
    async def test_rust_fs_stdio_protocol(self, rust_fs_config):
        """Test rust-fs MCP server stdio communication."""
        # Create subprocess for rust-fs
        process = await asyncio.create_subprocess_exec(
            rust_fs_config["command"],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Send initialize request
            request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {"protocolVersion": "1.0.0", "capabilities": {}},
                "id": 1,
            }

            process.stdin.write((json.dumps(request) + "\n").encode())
            await process.stdin.drain()

            # Read response with timeout
            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)

            response = json.loads(response_line.decode())
            assert response.get("jsonrpc") == "2.0"
            assert "result" in response or "error" in response

        finally:
            process.terminate()
            await process.wait()

    @pytest.mark.asyncio
    async def test_rust_fs_tools_list(self):
        """Test rust-fs tools/list method."""
        process = await asyncio.create_subprocess_exec(
            "rust-fs",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Initialize first
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {"protocolVersion": "1.0.0"},
                "id": 1,
            }
            process.stdin.write((json.dumps(init_request) + "\n").encode())
            await process.stdin.drain()
            await asyncio.wait_for(process.stdout.readline(), timeout=5.0)

            # Request tools list
            tools_request = {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2}
            process.stdin.write((json.dumps(tools_request) + "\n").encode())
            await process.stdin.drain()

            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)

            response = json.loads(response_line.decode())
            assert "result" in response
            assert "tools" in response["result"]

            # Verify expected tools
            tools = response["result"]["tools"]
            tool_names = [tool["name"] for tool in tools]

            # rust-fs should have file operation tools
            expected_tools = [
                "read",
                "write",
                "create",
                "delete",
                "move",
                "copy",
                "list",
                "stat",
                "find",
                "search",
                "replace",
                "execute",
            ]

            for expected in expected_tools:
                assert any(expected in name for name in tool_names), (
                    f"Expected tool '{expected}' not found in rust-fs"
                )

        finally:
            process.terminate()
            await process.wait()

    @pytest.mark.asyncio
    async def test_rust_fetch_tools(self):
        """Test rust-fetch HTTP fetching capabilities."""
        process = await asyncio.create_subprocess_exec(
            "rust-fetch",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {"protocolVersion": "1.0.0"},
                "id": 1,
            }
            process.stdin.write((json.dumps(init_request) + "\n").encode())
            await process.stdin.drain()
            await asyncio.wait_for(process.stdout.readline(), timeout=5.0)

            # Get tools
            tools_request = {"jsonrpc": "2.0", "method": "tools/list", "id": 2}
            process.stdin.write((json.dumps(tools_request) + "\n").encode())
            await process.stdin.drain()

            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)

            response = json.loads(response_line.decode())
            tools = response["result"]["tools"]
            tool_names = [tool["name"] for tool in tools]

            # rust-fetch should have HTTP tools
            expected_tools = ["fetch", "batch_fetch", "cache", "head", "validate"]
            for expected in expected_tools:
                assert any(expected in name for name in tool_names), (
                    f"Expected tool '{expected}' not found in rust-fetch"
                )

        finally:
            process.terminate()
            await process.wait()


class TestRustTestFrameworks:
    """Test Rust testing frameworks and infrastructure."""

    def test_cargo_test_available(self):
        """Test that cargo test is available."""
        result = subprocess.run(["cargo", "--version"], check=False, capture_output=True, text=True)
        assert result.returncode == 0, "cargo not found"
        assert "cargo" in result.stdout.lower()

    def test_rust_project_structure(self):
        """Test expected Rust project structure."""
        rust_projects = [
            project_root / "rust-fs",
            project_root / "rust-fetch",
            project_root / "rust-link",
        ]

        for project in rust_projects:
            if project.exists():
                # Check for Cargo.toml
                cargo_toml = project / "Cargo.toml"
                assert cargo_toml.exists(), f"Missing Cargo.toml in {project}"

                # Check for src directory
                src_dir = project / "src"
                assert src_dir.exists(), f"Missing src directory in {project}"

                # Check for main.rs or lib.rs
                main_rs = src_dir / "main.rs"
                lib_rs = src_dir / "lib.rs"
                assert main_rs.exists() or lib_rs.exists(), (
                    f"Missing main.rs or lib.rs in {project}"
                )

    @pytest.mark.skipif(not (project_root / "rust-fs").exists(), reason="rust-fs project not found")
    def test_rust_fs_tests(self):
        """Test that rust-fs has test suite."""
        rust_fs_dir = project_root / "rust-fs"

        # Check for tests directory or test modules
        tests_dir = rust_fs_dir / "tests"
        src_tests = list((rust_fs_dir / "src").glob("**/test*.rs"))

        assert tests_dir.exists() or len(src_tests) > 0, "No tests found in rust-fs project"

        # Run cargo test in check mode
        result = subprocess.run(
            ["cargo", "test", "--no-run"], check=False, cwd=rust_fs_dir, capture_output=True, text=True
        )

        assert result.returncode == 0, f"Cargo test failed: {result.stderr}"

    def test_rust_toolchain_version(self):
        """Test Rust toolchain version."""
        result = subprocess.run(["rustc", "--version"], check=False, capture_output=True, text=True)

        assert result.returncode == 0, "rustc not found"
        version_output = result.stdout

        # Extract version number
        import re

        version_match = re.search(r"rustc (\d+\.\d+\.\d+)", version_output)
        assert version_match, "Could not parse rustc version"

        version_parts = version_match.group(1).split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1])

        # Ensure we have a recent enough version (1.70+)
        assert major >= 1 and minor >= 70, f"Rust version too old: {version_match.group(1)}"


class TestRustMCPProtocol:
    """Test Rust MCP protocol implementation."""

    @pytest.mark.asyncio
    async def test_mcp_jsonrpc_format(self):
        """Test that Rust servers follow JSON-RPC format."""
        servers = ["rust-fs", "rust-fetch"]

        for server in servers:
            # Check if server exists
            which_result = subprocess.run(["which", server], check=False, capture_output=True)

            if which_result.returncode != 0:
                pytest.skip(f"{server} not found")
                continue

            process = await asyncio.create_subprocess_exec(
                server,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                # Send malformed request to test error handling
                bad_request = {"jsonrpc": "2.0", "method": "invalid_method", "id": 1}

                process.stdin.write((json.dumps(bad_request) + "\n").encode())
                await process.stdin.drain()

                response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)

                response = json.loads(response_line.decode())

                # Should return error in JSON-RPC format
                assert response.get("jsonrpc") == "2.0"
                assert "error" in response
                assert "code" in response["error"]
                assert "message" in response["error"]
                assert response["error"]["code"] < 0  # Error codes are negative

            finally:
                process.terminate()
                await process.wait()

    @pytest.mark.asyncio
    async def test_mcp_tool_schema_validation(self):
        """Test that Rust MCP servers provide valid tool schemas."""
        process = await asyncio.create_subprocess_exec(
            "rust-fs",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {"protocolVersion": "1.0.0"},
                "id": 1,
            }
            process.stdin.write((json.dumps(init_request) + "\n").encode())
            await process.stdin.drain()
            await asyncio.wait_for(process.stdout.readline(), timeout=5.0)

            # Get tools
            tools_request = {"jsonrpc": "2.0", "method": "tools/list", "id": 2}
            process.stdin.write((json.dumps(tools_request) + "\n").encode())
            await process.stdin.drain()

            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)

            response = json.loads(response_line.decode())
            tools = response["result"]["tools"]

            # Validate each tool schema
            for tool in tools:
                assert "name" in tool, "Tool missing 'name' field"
                assert "description" in tool, f"Tool {tool.get('name')} missing description"
                assert "inputSchema" in tool, f"Tool {tool.get('name')} missing inputSchema"

                # Validate input schema structure
                schema = tool["inputSchema"]
                assert "type" in schema, f"Tool {tool['name']} schema missing 'type'"
                assert schema["type"] == "object", (
                    f"Tool {tool['name']} schema type should be 'object'"
                )

                if "properties" in schema:
                    assert isinstance(schema["properties"], dict), (
                        f"Tool {tool['name']} properties should be dict"
                    )

                if "required" in schema:
                    assert isinstance(schema["required"], list), (
                        f"Tool {tool['name']} required should be list"
                    )

        finally:
            process.terminate()
            await process.wait()


class TestRustIntegrationWithPython:
    """Test integration between Rust tools and Python code."""

    def test_subprocess_communication(self):
        """Test Python can communicate with Rust processes."""
        # Test simple echo through rust-fs execute
        result = subprocess.run(
            ["rust-fs"],
            check=False, input=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": "execute", "arguments": {"command": "echo 'test'"}},
                    "id": 1,
                }
            )
            + "\n",
            capture_output=True,
            text=True,
            timeout=5,
        )

        # Should get valid JSON response
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                try:
                    response = json.loads(line)
                    if response.get("id") == 1:
                        assert "result" in response or "error" in response
                        break
                except json.JSONDecodeError:
                    continue

    @pytest.mark.asyncio
    async def test_async_rust_integration(self):
        """Test async integration with Rust MCP servers."""

        class RustMCPClient:
            """Simple client for Rust MCP servers."""

            def __init__(self, command: str):
                self.command = command
                self.process = None
                self.initialized = False

            async def connect(self):
                """Connect to Rust MCP server."""
                self.process = await asyncio.create_subprocess_exec(
                    self.command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Initialize
                await self._send_request(
                    {"method": "initialize", "params": {"protocolVersion": "1.0.0"}}
                )
                self.initialized = True

            async def _send_request(self, request: dict) -> dict:
                """Send request and get response."""
                if "id" not in request:
                    request["id"] = 1
                if "jsonrpc" not in request:
                    request["jsonrpc"] = "2.0"

                self.process.stdin.write((json.dumps(request) + "\n").encode())
                await self.process.stdin.drain()

                response_line = await asyncio.wait_for(self.process.stdout.readline(), timeout=5.0)

                return json.loads(response_line.decode())

            async def list_tools(self) -> list:
                """Get list of available tools."""
                response = await self._send_request({"method": "tools/list", "params": {}})
                return response.get("result", {}).get("tools", [])

            async def call_tool(self, name: str, arguments: dict) -> any:
                """Call a tool."""
                response = await self._send_request(
                    {"method": "tools/call", "params": {"name": name, "arguments": arguments}}
                )
                return response.get("result")

            async def close(self):
                """Close connection."""
                if self.process:
                    self.process.terminate()
                    await self.process.wait()

        # Test the client
        client = RustMCPClient("rust-fs")

        try:
            await client.connect()
            assert client.initialized

            # List tools
            tools = await client.list_tools()
            assert len(tools) > 0

            # Test a simple tool call
            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = os.path.join(tmpdir, "test.txt")

                # Write file
                await client.call_tool(
                    "write", {"path": test_file, "content": "Test content"}
                )

                # Verify file was created
                assert os.path.exists(test_file)
                with open(test_file) as f:
                    assert f.read() == "Test content"

        finally:
            await client.close()


class TestRustPerformance:
    """Test performance characteristics of Rust tools."""

    @pytest.mark.performance
    def test_rust_fs_performance(self):
        """Test rust-fs performance for file operations."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = []
            for i in range(100):
                test_file = os.path.join(tmpdir, f"test_{i}.txt")
                with open(test_file, "w") as f:
                    f.write(f"Content {i}")
                test_files.append(test_file)

            # Time rust-fs list operation
            start_time = time.time()

            subprocess.run(
                ["rust-fs"],
                check=False, input=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {"name": "list", "arguments": {"path": tmpdir}},
                        "id": 1,
                    }
                )
                + "\n",
                capture_output=True,
                text=True,
                timeout=10,
            )

            elapsed = time.time() - start_time

            # Should complete quickly (< 1 second for 100 files)
            assert elapsed < 1.0, f"rust-fs list took {elapsed:.2f}s for 100 files"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_rust_concurrent_operations(self):
        """Test concurrent operations with Rust MCP servers."""

        async def run_operation(server: str, operation_id: int):
            """Run a single operation."""
            process = await asyncio.create_subprocess_exec(
                server,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                # Initialize
                init_req = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {"protocolVersion": "1.0.0"},
                    "id": operation_id * 100,
                }
                process.stdin.write((json.dumps(init_req) + "\n").encode())
                await process.stdin.drain()
                await asyncio.wait_for(process.stdout.readline(), timeout=5.0)

                # Get tools
                tools_req = {"jsonrpc": "2.0", "method": "tools/list", "id": operation_id * 100 + 1}
                process.stdin.write((json.dumps(tools_req) + "\n").encode())
                await process.stdin.drain()

                response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)

                response = json.loads(response_line.decode())
                return response.get("result", {}).get("tools", [])

            finally:
                process.terminate()
                await process.wait()

        # Run multiple operations concurrently
        import time

        start_time = time.time()

        tasks = [run_operation("rust-fs", i) for i in range(5)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time

        # All should complete
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) >= 3, "Less than 3 concurrent operations succeeded"

        # Should complete reasonably quickly
        assert elapsed < 10.0, f"Concurrent operations took {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
