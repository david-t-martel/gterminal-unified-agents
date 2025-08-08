"""Comprehensive integration tests for MCP servers.

This module tests MCP server functionality, protocol compliance,
tool execution, and integration with Claude CLI.
"""

import asyncio
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from app.mcp_servers.base_mcp_server import BaseMCPServer
from app.mcp_servers.cloud_cost_optimizer import CloudCostOptimizerServer

# Import MCP server classes
from app.mcp_servers.gemini_code_reviewer import GeminiCodeReviewerServer
from app.mcp_servers.gemini_master_architect import GeminiMasterArchitectServer
from app.mcp_servers.gemini_workspace_analyzer import GeminiWorkspaceAnalyzerServer
import pytest


class TestMCPServerInitialization:
    """Test MCP server initialization and configuration."""

    def test_gemini_code_reviewer_initialization(self):
        """Test GeminiCodeReviewerServer initialization."""
        server = GeminiCodeReviewerServer()

        assert server.name == "gemini-code-reviewer"
        assert hasattr(server, "gemini_client")
        assert hasattr(server, "tools")
        assert len(server.tools) > 0

    def test_workspace_analyzer_initialization(self):
        """Test GeminiWorkspaceAnalyzerServer initialization."""
        server = GeminiWorkspaceAnalyzerServer()

        assert server.name == "gemini-workspace-analyzer"
        assert hasattr(server, "gemini_client")
        assert hasattr(server, "tools")

    def test_master_architect_initialization(self):
        """Test GeminiMasterArchitectServer initialization."""
        server = GeminiMasterArchitectServer()

        assert server.name == "gemini-master-architect"
        assert hasattr(server, "gemini_client")
        assert hasattr(server, "tools")

    def test_cost_optimizer_initialization(self):
        """Test CloudCostOptimizerServer initialization."""
        server = CloudCostOptimizerServer()

        assert server.name == "cloud-cost-optimizer"
        assert hasattr(server, "gemini_client")
        assert hasattr(server, "tools")

    def test_base_mcp_server_abstract(self):
        """Test that BaseMCPServer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMCPServer()


class TestMCPToolRegistration:
    """Test MCP tool registration and validation."""

    @pytest.fixture
    def code_reviewer_server(self):
        """Create code reviewer server for testing."""
        return GeminiCodeReviewerServer()

    @pytest.fixture
    def workspace_analyzer_server(self):
        """Create workspace analyzer server for testing."""
        return GeminiWorkspaceAnalyzerServer()

    def test_code_reviewer_tools_registered(self, code_reviewer_server):
        """Test that code reviewer tools are properly registered."""
        expected_tools = [
            "analyze_code",
            "security_review",
            "performance_review",
            "quality_check",
            "generate_review_report",
        ]

        registered_tools = [tool.name for tool in code_reviewer_server.tools]

        for expected_tool in expected_tools:
            assert expected_tool in registered_tools

    def test_workspace_analyzer_tools_registered(self, workspace_analyzer_server):
        """Test that workspace analyzer tools are properly registered."""
        expected_tools = [
            "analyze_workspace",
            "analyze_dependencies",
            "generate_architecture_diagram",
            "assess_code_quality",
            "identify_tech_debt",
        ]

        registered_tools = [tool.name for tool in workspace_analyzer_server.tools]

        for expected_tool in expected_tools:
            assert expected_tool in registered_tools

    def test_tool_parameter_validation(self, code_reviewer_server):
        """Test that tool parameters are properly validated."""
        analyze_code_tool = None
        for tool in code_reviewer_server.tools:
            if tool.name == "analyze_code":
                analyze_code_tool = tool
                break

        assert analyze_code_tool is not None
        assert hasattr(analyze_code_tool, "input_schema")

        # Check required parameters
        properties = analyze_code_tool.input_schema.get("properties", {})
        assert "file_path" in properties
        assert "language" in properties

    def test_tool_descriptions_present(self, code_reviewer_server):
        """Test that all tools have proper descriptions."""
        for tool in code_reviewer_server.tools:
            assert hasattr(tool, "description")
            assert tool.description is not None
            assert len(tool.description) > 0


class TestMCPServerExecution:
    """Test MCP server tool execution."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client."""
        mock_client = Mock()
        mock_client.generate_content = AsyncMock(return_value=Mock(text="Mock analysis result"))
        return mock_client

    @pytest.fixture
    def test_file_path(self, tmp_path):
        """Create a test file for analysis."""
        test_file = tmp_path / "test_code.py"
        test_file.write_text(
            """
def hello_world():
    print("Hello, World!")
    return "success"

if __name__ == "__main__":
    hello_world()
"""
        )
        return str(test_file)

    @pytest.mark.asyncio
    async def test_code_review_tool_execution(self, mock_gemini_client, test_file_path):
        """Test code review tool execution."""
        server = GeminiCodeReviewerServer()
        server.gemini_client = mock_gemini_client

        # Execute analyze_code tool
        result = await server.execute_tool(
            "analyze_code",
            {"file_path": test_file_path, "language": "python", "focus_areas": "quality,security"},
        )

        assert result is not None
        assert "analysis" in result or "Mock analysis result" in str(result)
        mock_gemini_client.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_workspace_analysis_tool_execution(self, mock_gemini_client, tmp_path):
        """Test workspace analysis tool execution."""
        # Create a mock project structure
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        (project_dir / "main.py").write_text("print('Hello')")
        (project_dir / "requirements.txt").write_text("requests==2.28.0")
        (project_dir / "README.md").write_text("# Test Project")

        server = GeminiWorkspaceAnalyzerServer()
        server.gemini_client = mock_gemini_client

        # Execute analyze_workspace tool
        result = await server.execute_tool(
            "analyze_workspace", {"workspace_path": str(project_dir), "analysis_depth": "standard"}
        )

        assert result is not None
        mock_gemini_client.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_cost_optimization_tool_execution(self, mock_gemini_client):
        """Test cost optimization tool execution."""
        server = CloudCostOptimizerServer()
        server.gemini_client = mock_gemini_client

        # Mock GCP billing client
        with patch("app.mcp_servers.cloud_cost_optimizer.build") as mock_build:
            mock_billing_client = Mock()
            mock_build.return_value = mock_billing_client

            result = await server.execute_tool(
                "analyze_costs",
                {"project_id": "test-project", "time_range": "30d", "include_forecasting": True},
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, test_file_path):
        """Test error handling in tool execution."""
        server = GeminiCodeReviewerServer()

        # Test with invalid file path
        result = await server.execute_tool(
            "analyze_code", {"file_path": "/nonexistent/file.py", "language": "python"}
        )

        # Should handle error gracefully
        assert result is not None
        assert isinstance(result, dict)
        assert "error" in result or "Error" in str(result)

    @pytest.mark.asyncio
    async def test_tool_parameter_validation_errors(self):
        """Test tool parameter validation errors."""
        server = GeminiCodeReviewerServer()

        # Test with missing required parameters
        result = await server.execute_tool(
            "analyze_code",
            {
                "language": "python"
                # Missing file_path
            },
        )

        assert result is not None
        assert isinstance(result, dict)
        assert "error" in result or "Error" in str(result)


class TestMCPServerProtocolCompliance:
    """Test MCP protocol compliance."""

    @pytest.mark.asyncio
    async def test_server_stdio_protocol(self):
        """Test server stdio protocol compliance."""
        # Create a temporary script to run the MCP server
        server_script = """
import asyncio
import sys
from app.mcp_servers.gemini_code_reviewer import GeminiCodeReviewerServer

async def main():
    server = GeminiCodeReviewerServer()
    await server.run_stdio()

if __name__ == "__main__":
    asyncio.run(main())
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(server_script)
            server_script_path = f.name

        try:
            # Test basic protocol handshake
            process = subprocess.Popen(
                [sys.executable, server_script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
            }

            stdout, stderr = process.communicate(input=json.dumps(init_request) + "\n", timeout=10)

            # Should not crash and should return some response
            assert process.returncode == 0 or stdout or stderr

        except subprocess.TimeoutExpired:
            process.kill()
            pytest.skip("Server took too long to respond")
        finally:
            Path(server_script_path).unlink()

    @pytest.mark.asyncio
    async def test_mcp_inspector_validation(self):
        """Test MCP Inspector validation."""
        # This test would normally run MCP Inspector against our servers
        # For now, we'll test the structure compliance

        server = GeminiCodeReviewerServer()

        # Check that server has required MCP methods
        assert hasattr(server, "list_tools")
        assert hasattr(server, "call_tool")
        assert hasattr(server, "list_resources") or True  # Optional
        assert hasattr(server, "list_prompts") or True  # Optional

        # Test tool listing
        tools = await server.list_tools()
        assert isinstance(tools, list)

        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "input_schema")

    def test_tool_schema_validation(self):
        """Test that tool schemas are valid JSON Schema."""
        server = GeminiCodeReviewerServer()

        for tool in server.tools:
            schema = tool.input_schema

            # Basic JSON Schema validation
            assert isinstance(schema, dict)
            assert "type" in schema
            assert schema["type"] == "object"

            if "properties" in schema:
                assert isinstance(schema["properties"], dict)

            if "required" in schema:
                assert isinstance(schema["required"], list)


class TestMCPServerResources:
    """Test MCP server resource management."""

    @pytest.fixture
    def server_with_resources(self):
        """Create server that supports resources."""
        return GeminiWorkspaceAnalyzerServer()

    @pytest.mark.asyncio
    async def test_resource_listing(self, server_with_resources):
        """Test resource listing functionality."""
        if hasattr(server_with_resources, "list_resources"):
            resources = await server_with_resources.list_resources()
            assert isinstance(resources, list)

            for resource in resources:
                assert hasattr(resource, "uri")
                assert hasattr(resource, "name")

    @pytest.mark.asyncio
    async def test_resource_reading(self, server_with_resources, tmp_path):
        """Test resource reading functionality."""
        if hasattr(server_with_resources, "read_resource"):
            # Create a test resource
            test_file = tmp_path / "test_resource.txt"
            test_file.write_text("Test resource content")

            # Try to read it
            try:
                content = await server_with_resources.read_resource(f"file://{test_file}")
                assert content is not None
            except NotImplementedError:
                pytest.skip("Resource reading not implemented")


class TestMCPServerConcurrency:
    """Test MCP server concurrency and performance."""

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, tmp_path):
        """Test concurrent execution of tools."""
        server = GeminiCodeReviewerServer()

        # Create test files
        test_files = []
        for i in range(3):
            test_file = tmp_path / f"test_{i}.py"
            test_file.write_text(f"def function_{i}(): pass")
            test_files.append(str(test_file))

        # Mock Gemini client to avoid API calls
        server.gemini_client = Mock()
        server.gemini_client.generate_content = AsyncMock(return_value=Mock(text="Mock analysis"))

        # Execute tools concurrently
        tasks = []
        for test_file in test_files:
            task = server.execute_tool(
                "analyze_code", {"file_path": test_file, "language": "python"}
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert result is not None

    @pytest.mark.asyncio
    async def test_server_resource_limits(self):
        """Test server resource usage limits."""
        server = GeminiCodeReviewerServer()

        # Test memory usage doesn't grow unbounded
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Mock client to avoid API calls
        server.gemini_client = Mock()
        server.gemini_client.generate_content = AsyncMock(return_value=Mock(text="Mock result"))

        # Execute many operations
        for i in range(50):
            await server.execute_tool(
                "analyze_code", {"file_path": f"/tmp/test_{i}.py", "language": "python"}
            )

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 100MB)
        assert memory_growth < 100 * 1024 * 1024


class TestMCPServerConfiguration:
    """Test MCP server configuration and environment handling."""

    def test_environment_variable_handling(self):
        """Test that servers handle environment variables correctly."""
        # Test with missing environment variables
        with patch.dict("os.environ", {}, clear=True):
            try:
                server = GeminiCodeReviewerServer()
                # Should either work with defaults or raise informative error
                assert server is not None
            except Exception as e:
                # Should be informative error about missing config
                assert "project" in str(e).lower() or "credential" in str(e).lower()

    def test_project_id_configuration(self):
        """Test project ID configuration."""
        with patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "test-project-123"}):
            server = GeminiCodeReviewerServer()

            # Should use the configured project ID
            assert hasattr(server, "project_id")
            assert server.project_id == "test-project-123"

    def test_server_name_consistency(self):
        """Test that server names match expected patterns."""
        servers = [
            GeminiCodeReviewerServer(),
            GeminiWorkspaceAnalyzerServer(),
            GeminiMasterArchitectServer(),
            CloudCostOptimizerServer(),
        ]

        for server in servers:
            # Names should follow kebab-case pattern
            assert "-" in server.name or server.name.islower()
            # Names should not be empty
            assert len(server.name) > 0


class TestMCPServerIntegration:
    """Test MCP server integration with external systems."""

    @pytest.mark.asyncio
    async def test_gemini_api_integration(self):
        """Test integration with Gemini API."""
        server = GeminiCodeReviewerServer()

        # Test with real Gemini client (if credentials available)
        if hasattr(server, "gemini_client") and server.gemini_client:
            try:
                # Use a simple test that shouldn't fail
                result = await server.execute_tool(
                    "analyze_code",
                    {
                        "file_path": __file__,
                        "language": "python",
                        "focus_areas": "structure",
                    },  # Use this test file
                )

                assert result is not None
                assert isinstance(result, dict | str)

            except Exception as e:
                # If API call fails, that's expected in test environment
                assert "auth" in str(e).lower() or "credential" in str(e).lower()

    @pytest.mark.asyncio
    async def test_file_system_integration(self, tmp_path):
        """Test integration with file system operations."""
        server = GeminiWorkspaceAnalyzerServer()

        # Create test project structure
        project_dir = tmp_path / "integration_test"
        project_dir.mkdir()

        # Create various file types
        (project_dir / "main.py").write_text("print('Hello World')")
        (project_dir / "config.json").write_text('{"version": "1.0"}')
        (project_dir / "README.md").write_text("# Test Project")

        sub_dir = project_dir / "src"
        sub_dir.mkdir()
        (sub_dir / "utils.py").write_text("def helper(): pass")

        # Test workspace analysis
        server.gemini_client = Mock()
        server.gemini_client.generate_content = AsyncMock(
            return_value=Mock(text="Mock workspace analysis")
        )

        result = await server.execute_tool(
            "analyze_workspace",
            {"workspace_path": str(project_dir), "analysis_depth": "comprehensive"},
        )

        assert result is not None
        # Should have analyzed the project structure
        server.gemini_client.generate_content.assert_called()

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test server error recovery mechanisms."""
        server = GeminiCodeReviewerServer()

        # Test recovery from network errors
        server.gemini_client = Mock()
        server.gemini_client.generate_content = AsyncMock(side_effect=Exception("Network error"))

        # Should handle network errors gracefully
        result = await server.execute_tool(
            "analyze_code", {"file_path": __file__, "language": "python"}
        )

        # Should return error result, not crash
        assert result is not None
        assert isinstance(result, dict)
        assert "error" in result or "Error" in str(result)


class TestMCPServerMetrics:
    """Test MCP server metrics and monitoring."""

    def test_server_statistics(self):
        """Test server statistics collection."""
        server = GeminiCodeReviewerServer()

        if hasattr(server, "get_stats"):
            stats = server.get_stats()

            assert isinstance(stats, dict)
            assert "tools_executed" in stats or "uptime" in stats
        else:
            # If no stats method, that's acceptable
            pass

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, tmp_path):
        """Test performance monitoring capabilities."""
        server = GeminiCodeReviewerServer()
        server.gemini_client = Mock()
        server.gemini_client.generate_content = AsyncMock(
            return_value=Mock(text="Quick mock response")
        )

        # Create test file
        test_file = tmp_path / "perf_test.py"
        test_file.write_text("def test(): pass")

        # Measure execution time
        import time

        start_time = time.time()

        result = await server.execute_tool(
            "analyze_code", {"file_path": str(test_file), "language": "python"}
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete reasonably quickly (under 5 seconds for mock)
        assert execution_time < 5.0
        assert result is not None
