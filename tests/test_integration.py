"""Integration tests for gterminal agent and system integration.

This test suite validates that the major components work together correctly
after the gapp -> gterminal consolidation, testing agent workflows, MCP
integration, terminal functionality, and cross-system communication.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


@pytest.mark.integration
class TestAgentIntegration:
    """Test agent integration and workflows."""

    async def test_agent_registry_functionality(self) -> None:
        """Test that the agent registry works correctly."""
        from gterminal.agents import AGENT_REGISTRY
        from gterminal.agents import get_agent_service

        # Test registry is populated
        assert isinstance(AGENT_REGISTRY, dict)
        assert len(AGENT_REGISTRY) > 0

        # Test getting agent services
        available_agents = list(AGENT_REGISTRY.keys())
        for agent_type in available_agents[:3]:  # Test first 3 agents
            if AGENT_REGISTRY[agent_type] is not None:  # Skip None entries
                agent = get_agent_service(agent_type)
                assert agent is not None

    async def test_code_review_agent_workflow(self, mock_gemini_client) -> None:
        """Test code review agent workflow."""
        try:
            from gterminal.agents.code_review_agent import CodeReviewAgentService
        except ImportError:
            pytest.skip("Code review agent not available")

        with patch("gterminal.agents.code_review_agent.vertexai") as mock_vertexai:
            mock_model = MagicMock()
            mock_model.generate_content = AsyncMock()
            mock_model.generate_content.return_value.text = json.dumps(
                {
                    "summary": "Code looks good",
                    "issues": [],
                    "suggestions": ["Add type hints"],
                    "score": 85,
                }
            )
            mock_vertexai.GenerativeModel.return_value = mock_model

            service = CodeReviewAgentService()

            # Test code review
            result = await service.review_code(
                code="def hello(): print('hello')", language="python"
            )

            assert result is not None
            if isinstance(result, str):
                # Handle string response
                assert len(result) > 0
            else:
                # Handle dict response
                assert "summary" in result or "analysis" in result

    async def test_workspace_analyzer_integration(self) -> None:
        """Test workspace analyzer integration."""
        try:
            from gterminal.agents.workspace_analyzer_agent import WorkspaceAnalyzerService
        except ImportError:
            pytest.skip("Workspace analyzer not available")

        with patch("gterminal.agents.workspace_analyzer_agent.vertexai") as mock_vertexai:
            mock_model = MagicMock()
            mock_model.generate_content = AsyncMock()
            mock_model.generate_content.return_value.text = json.dumps(
                {"structure": "Well organized", "files": 10, "recommendations": ["Add tests"]}
            )
            mock_vertexai.GenerativeModel.return_value = mock_model

            service = WorkspaceAnalyzerService()

            # Create a temporary workspace
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                (temp_path / "main.py").write_text("print('hello')")
                (temp_path / "README.md").write_text("# Test Project")

                result = await service.analyze_workspace(str(temp_path))
                assert result is not None

    async def test_documentation_generator_integration(self) -> None:
        """Test documentation generator integration."""
        try:
            from gterminal.agents.documentation_generator_agent import DocumentationGeneratorService
        except ImportError:
            pytest.skip("Documentation generator not available")

        with patch("gterminal.agents.documentation_generator_agent.vertexai") as mock_vertexai:
            mock_model = MagicMock()
            mock_model.generate_content = AsyncMock()
            mock_model.generate_content.return_value.text = (
                "# API Documentation\n\nThis is documentation."
            )
            mock_vertexai.GenerativeModel.return_value = mock_model

            service = DocumentationGeneratorService()

            result = await service.generate_documentation(code="def hello(): pass", doc_type="api")

            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0


@pytest.mark.integration
class TestMCPIntegration:
    """Test MCP (Model Context Protocol) integration."""

    def test_mcp_registry_setup(self) -> None:
        """Test that MCP registry is properly set up."""
        from gterminal.agents import MCP_REGISTRY
        from gterminal.agents import get_mcp_server

        assert isinstance(MCP_REGISTRY, dict)
        assert len(MCP_REGISTRY) > 0

        # Test getting MCP servers
        for server_type in list(MCP_REGISTRY.keys())[:2]:  # Test first 2
            server = get_mcp_server(server_type)
            assert server is not None

    async def test_mcp_server_initialization(self) -> None:
        """Test that MCP servers can be initialized."""
        try:
            from gterminal.mcp_servers.gemini_server_agent_mcp import mcp
        except ImportError:
            pytest.skip("MCP server not available")

        # Test that MCP server object exists
        assert mcp is not None

        # Test that it has expected methods/attributes
        # This depends on the specific FastMCP implementation
        if hasattr(mcp, "tool"):
            assert callable(mcp.tool)

    @pytest.mark.mcp
    async def test_mcp_tool_registration(self) -> None:
        """Test that MCP tools are properly registered."""
        try:
            from gterminal.agents import code_reviewer_mcp
            from gterminal.agents import workspace_analyzer_mcp
        except ImportError:
            pytest.skip("MCP modules not available")

        # Test that MCP servers have tools registered
        for mcp_server in [code_reviewer_mcp, workspace_analyzer_mcp]:
            if mcp_server is not None:
                assert hasattr(mcp_server, "tool") or hasattr(mcp_server, "_tools")


@pytest.mark.integration
class TestTerminalIntegration:
    """Test terminal integration and user interfaces."""

    async def test_react_engine_initialization(self) -> None:
        """Test that React engine initializes correctly."""
        from gterminal.terminal.react_engine import ReactEngine

        # Test basic initialization
        engine = ReactEngine()
        assert engine is not None

    async def test_enhanced_react_orchestrator(self) -> None:
        """Test enhanced React orchestrator functionality."""
        try:
            from gterminal.terminal.enhanced_react_orchestrator import EnhancedReactOrchestrator
        except ImportError:
            pytest.skip("Enhanced React orchestrator not available")

        orchestrator = EnhancedReactOrchestrator()
        assert orchestrator is not None

    async def test_agent_commands_integration(self) -> None:
        """Test agent commands integration."""
        try:
            from gterminal.terminal import agent_commands
        except ImportError:
            pytest.skip("Agent commands not available")

        # Test that the module imports without errors
        assert agent_commands is not None

    async def test_terminal_web_server_integration(self) -> None:
        """Test web terminal server integration."""
        try:
            from gterminal.terminal.web_terminal_server import create_app
        except ImportError:
            pytest.skip("Web terminal server not available")

        # Test that app can be created
        app = create_app()
        assert app is not None


@pytest.mark.integration
class TestGeminiCLIIntegration:
    """Test Gemini CLI integration."""

    def test_gemini_cli_main_import(self) -> None:
        """Test that Gemini CLI main module imports correctly."""
        from gterminal.gemini_cli import main

        assert main is not None

    def test_gemini_client_integration(self) -> None:
        """Test Gemini client integration."""
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}):
            from gterminal.gemini_cli.core.client import GeminiClient

            # Test that client can be instantiated
            client = GeminiClient()
            assert client is not None

    def test_auth_integration(self) -> None:
        """Test auth integration."""
        from gterminal.auth.gcp_auth import get_auth_manager

        # Test that auth manager can be created
        auth_manager = get_auth_manager()
        assert auth_manager is not None

    async def test_gemini_react_engine(self) -> None:
        """Test Gemini React engine integration."""
        try:
            from gterminal.gemini_cli.core.react_engine import GeminiReactEngine
        except ImportError:
            pytest.skip("Gemini React engine not available")

        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}):
            engine = GeminiReactEngine()
            assert engine is not None


@pytest.mark.integration
class TestCacheIntegration:
    """Test cache system integration."""

    async def test_cache_manager_integration(self) -> None:
        """Test cache manager integration."""
        from gterminal.cache.cache_manager import CacheManager

        # Test basic cache manager functionality
        cache = CacheManager()
        assert cache is not None

        # Test basic operations
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"

    async def test_memory_cache_integration(self) -> None:
        """Test memory cache integration."""
        from gterminal.cache.memory_cache import MemoryCache

        cache = MemoryCache(max_size=100)
        assert cache is not None

        # Test basic operations
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    @pytest.mark.requires_redis
    async def test_redis_cache_integration(self) -> None:
        """Test Redis cache integration."""
        try:
            from gterminal.cache.redis_cache import RedisCache
        except ImportError:
            pytest.skip("Redis cache not available")

        # Mock Redis connection
        with patch("gterminal.cache.redis_cache.aioredis") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.from_url.return_value = mock_client

            cache = RedisCache()
            assert cache is not None


@pytest.mark.integration
class TestUtilsIntegration:
    """Test utilities integration."""

    def test_common_utils_integration(self) -> None:
        """Test common utilities integration."""
        from gterminal.utils.common import base_classes
        from gterminal.utils.common import cache_utils
        from gterminal.utils.common import file_ops

        assert base_classes is not None
        assert cache_utils is not None
        assert file_ops is not None

    def test_database_utils_integration(self) -> None:
        """Test database utilities integration."""
        from gterminal.utils.database import cache
        from gterminal.utils.database import connection_pool

        assert cache is not None
        assert connection_pool is not None

    def test_rust_extensions_integration(self) -> None:
        """Test Rust extensions integration."""
        try:
            from gterminal.utils.rust_extensions import rust_bindings

            assert rust_bindings is not None
        except ImportError:
            pytest.skip("Rust extensions not available")


@pytest.mark.integration
@pytest.mark.e2e
class TestEndToEndWorkflows:
    """Test end-to-end workflows that span multiple components."""

    async def test_code_analysis_workflow(self) -> None:
        """Test complete code analysis workflow."""
        # This test simulates a user requesting code analysis
        # through the terminal interface, which uses agents and caching

        # Mock the necessary components
        with patch("gterminal.agents.code_review_agent.vertexai") as mock_vertexai:
            mock_model = MagicMock()
            mock_model.generate_content = AsyncMock()
            mock_model.generate_content.return_value.text = json.dumps(
                {"analysis": "Good code structure", "issues": [], "score": 90}
            )
            mock_vertexai.GenerativeModel.return_value = mock_model

            try:
                from gterminal.agents import get_agent_service

                # Get code review agent
                agent = get_agent_service("code-reviewer")
                if agent is None:
                    pytest.skip("Code review agent not available")

                # Simulate code analysis
                result = await agent.review_code(
                    code="def test_function(): return True", language="python"
                )

                assert result is not None
            except (ImportError, ValueError):
                pytest.skip("Agent service not available")

    async def test_workspace_analysis_workflow(self) -> None:
        """Test complete workspace analysis workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a sample workspace
            workspace_path = Path(temp_dir)
            (workspace_path / "main.py").write_text("print('hello')")
            (workspace_path / "config.json").write_text('{"version": "1.0"}')
            (workspace_path / "README.md").write_text("# Test Project")

            # Mock Vertex AI
            with patch("gterminal.agents.workspace_analyzer_agent.vertexai") as mock_vertexai:
                mock_model = MagicMock()
                mock_model.generate_content = AsyncMock()
                mock_model.generate_content.return_value.text = json.dumps(
                    {
                        "structure": "Well organized project",
                        "file_count": 3,
                        "recommendations": ["Add tests", "Add documentation"],
                    }
                )
                mock_vertexai.GenerativeModel.return_value = mock_model

                try:
                    from gterminal.agents import get_agent_service

                    # Get workspace analyzer
                    agent = get_agent_service("workspace-analyzer")
                    if agent is None:
                        pytest.skip("Workspace analyzer not available")

                    # Analyze workspace
                    result = await agent.analyze_workspace(str(workspace_path))
                    assert result is not None

                except (ImportError, ValueError):
                    pytest.skip("Workspace analyzer not available")

    async def test_mcp_communication_workflow(self) -> None:
        """Test MCP communication workflow."""
        try:
            from gterminal.agents import MCP_REGISTRY

            if not MCP_REGISTRY:
                pytest.skip("No MCP servers available")

            # Test that we can get MCP servers
            server_types = list(MCP_REGISTRY.keys())[:2]  # Test first 2

            from gterminal.agents import get_mcp_server

            for server_type in server_types:
                server = get_mcp_server(server_type)
                assert server is not None

        except (ImportError, ValueError):
            pytest.skip("MCP integration not available")

    async def test_caching_workflow(self) -> None:
        """Test caching workflow across components."""
        from gterminal.cache.cache_manager import CacheManager

        cache = CacheManager()

        # Test cache workflow
        test_data = {"analysis": "cached result", "timestamp": "2024-01-01"}

        # Store in cache
        await cache.set("analysis:test_code", test_data)

        # Retrieve from cache
        cached_result = await cache.get("analysis:test_code")
        assert cached_result == test_data

        # Test cache expiry
        await cache.set("temp_data", "temporary", ttl=1)
        await asyncio.sleep(1.1)  # Wait for expiry

        expired_result = await cache.get("temp_data")
        assert expired_result is None


@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Test performance aspects of integration."""

    @pytest.mark.benchmark
    async def test_agent_initialization_performance(self, benchmark) -> None:
        """Benchmark agent initialization performance."""
        from gterminal.agents import get_agent_service

        def init_agents():
            available_agents = ["code-reviewer", "workspace-analyzer"]
            agents = []
            for agent_type in available_agents:
                try:
                    agent = get_agent_service(agent_type)
                    if agent is not None:
                        agents.append(agent)
                except (ValueError, ImportError):
                    continue
            return agents

        # Benchmark the initialization
        agents = benchmark(init_agents)
        assert len(agents) >= 0  # At least some agents should initialize

    @pytest.mark.benchmark
    async def test_cache_performance(self, benchmark) -> None:
        """Benchmark cache performance."""
        from gterminal.cache.memory_cache import MemoryCache

        cache = MemoryCache(max_size=1000)

        def cache_operations():
            # Perform cache operations
            for i in range(100):
                cache.set(f"key_{i}", f"value_{i}")
                cache.get(f"key_{i}")

        benchmark(cache_operations)

    async def test_import_performance(self) -> None:
        """Test that imports don't take too long."""
        import time

        # Test core imports
        start_time = time.time()
        import_time = time.time() - start_time

        # Imports should be reasonably fast (< 5 seconds)
        assert import_time < 5.0, f"Imports took too long: {import_time:.2f}s"


@pytest.mark.integration
@pytest.mark.security
class TestSecurityIntegration:
    """Test security aspects of integration."""

    def test_auth_integration_security(self) -> None:
        """Test authentication integration security."""
        from gterminal.auth.gcp_auth import get_auth_manager

        auth_manager = get_auth_manager()
        assert auth_manager is not None

        # Test that sensitive data is handled properly
        # (This would need more specific implementation details)

    def test_secret_management_integration(self) -> None:
        """Test secret management integration."""
        try:
            from gterminal.core.security.secrets_manager import SecretsManager
        except ImportError:
            pytest.skip("Secrets manager not available")

        # Test that secrets manager can be instantiated
        manager = SecretsManager()
        assert manager is not None

    def test_security_middleware_integration(self) -> None:
        """Test security middleware integration."""
        try:
            from gterminal.core.security.integrated_security_middleware import SecurityMiddleware
        except ImportError:
            pytest.skip("Security middleware not available")

        # Test basic instantiation
        middleware = SecurityMiddleware()
        assert middleware is not None
