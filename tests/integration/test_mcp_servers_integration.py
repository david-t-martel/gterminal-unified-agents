"""
Integration tests for MCP servers to ensure proper functionality and compatibility.
These tests validate MCP server behavior, tool registration, and external integrations.
"""

import asyncio
import os
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from app.mcp_servers.base_mcp_server import BaseMCPServer
from app.mcp_servers.gemini_mcp_base import ContextCache
from app.mcp_servers.gemini_mcp_base import GeminiMCPBase
from app.mcp_servers.gemini_mcp_base import GeminiMCPConfig
import pytest


class TestMCPServer(BaseMCPServer):
    """Test implementation of BaseMCPServer for testing."""

    def __init__(self, name="test_server"):
        super().__init__(name)
        self.test_data = {}

    async def _initialize_components(self):
        """Test implementation of component initialization."""
        self.test_data["initialized"] = True


class TestGeminiMCPServer(GeminiMCPBase):
    """Test implementation of GeminiMCPBase for testing."""

    def __init__(self, name="test_gemini_server", model_type="pro"):
        # Mock the model initialization to avoid real API calls
        with patch.object(GeminiMCPBase, "_initialize_model"):
            super().__init__(name, model_type)
        self.mock_model = Mock()
        self.model = self.mock_model


@pytest.fixture
def test_mcp_server():
    """Create a test MCP server instance."""
    return TestMCPServer()


@pytest.fixture
def test_gemini_server():
    """Create a test Gemini MCP server instance."""
    return TestGeminiMCPServer()


@pytest.fixture
def gemini_config():
    """Create a test Gemini MCP configuration."""
    with patch.dict(
        os.environ,
        {
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
        },
    ):
        return GeminiMCPConfig()


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory with sample files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)

        # Create sample Python files
        (project_path / "main.py").write_text(
            """
def main():
    print("Hello World")

if __name__ == "__main__":
    main()
"""
        )

        (project_path / "utils.py").write_text(
            """
def helper_function(x, y):
    return x + y

class UtilityClass:
    def __init__(self, value):
        self.value = value

    def process(self):
        return self.value * 2
"""
        )

        # Create a subdirectory with more files
        sub_dir = project_path / "submodule"
        sub_dir.mkdir()

        (sub_dir / "module.py").write_text(
            """
import os
from typing import List

def process_data(data: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in data}
"""
        )

        # Create non-Python files
        (project_path / "README.md").write_text("# Test Project")
        (project_path / "requirements.txt").write_text("fastapi>=0.100.0\npytest>=7.0.0")

        yield str(project_path)


class TestBaseMCPServer:
    """Test base MCP server functionality."""

    def test_server_initialization(self, test_mcp_server):
        """Test MCP server initialization."""
        assert test_mcp_server.name == "test_server"
        assert test_mcp_server.logger is not None
        assert test_mcp_server.initialized is False

    @pytest.mark.asyncio
    async def test_server_initialization_lifecycle(self, test_mcp_server):
        """Test complete server initialization lifecycle."""
        # Initial state
        assert not test_mcp_server.initialized

        # Initialize
        await test_mcp_server.initialize()
        assert test_mcp_server.initialized
        assert test_mcp_server.test_data.get("initialized") is True

        # Should not re-initialize
        await test_mcp_server.initialize()
        assert test_mcp_server.initialized

    @pytest.mark.asyncio
    async def test_server_shutdown(self, test_mcp_server):
        """Test server shutdown."""
        await test_mcp_server.initialize()
        assert test_mcp_server.initialized

        await test_mcp_server.shutdown()
        assert not test_mcp_server.initialized

    def test_error_handling(self, test_mcp_server):
        """Test error handling functionality."""
        test_error = Exception("Test error")
        result = test_mcp_server.handle_error(test_error, "test context")

        assert "error" in result
        assert result["error"] == "Test error"
        assert result["context"] == "test context"

    @pytest.mark.asyncio
    async def test_redis_integration(self, test_mcp_server, mock_redis_client):
        """Test Redis integration through RedisClientMixin."""
        # Mock Redis operations
        test_mcp_server.redis_client = mock_redis_client

        # Test Redis operations
        await test_mcp_server.redis_client.set("test_key", "test_value")
        mock_redis_client.set.assert_called_with("test_key", "test_value")

        await test_mcp_server.redis_client.get("test_key")
        mock_redis_client.get.assert_called_with("test_key")


class TestGeminiMCPConfig:
    """Test Gemini MCP configuration."""

    def test_default_configuration(self, gemini_config):
        """Test default configuration values."""
        assert gemini_config.project == "test-project"
        assert gemini_config.location == "us-central1"
        assert gemini_config.use_vertex is True

        # Model configurations
        assert "master" in gemini_config.models
        assert "flash" in gemini_config.models
        assert "pro" in gemini_config.models

        # Performance settings
        assert gemini_config.max_retries >= 1
        assert gemini_config.timeout > 0
        assert gemini_config.max_concurrent_requests > 0

    def test_environment_variable_override(self):
        """Test configuration override via environment variables."""
        with patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "custom-project",
                "GOOGLE_CLOUD_LOCATION": "europe-west1",
                "GOOGLE_GENAI_USE_VERTEXAI": "false",
            },
        ):
            config = GeminiMCPConfig()
            assert config.project == "custom-project"
            assert config.location == "europe-west1"
            assert config.use_vertex is False

    def test_model_configurations(self, gemini_config):
        """Test model configuration settings."""
        # Check that all model types are defined
        assert "master" in gemini_config.models
        assert "flash" in gemini_config.models
        assert "pro" in gemini_config.models

        # Check context limits
        assert gemini_config.max_context_size["gemini-2.5-pro"] > 0
        assert gemini_config.max_context_size["gemini-2.0-flash"] > 0


class TestContextCache:
    """Test context caching functionality."""

    def test_cache_initialization(self):
        """Test cache initialization with custom parameters."""
        cache = ContextCache(max_size=50, ttl=1800)
        assert cache._max_size == 50
        assert cache._ttl == 1800
        assert len(cache._cache) == 0

    def test_cache_set_get(self):
        """Test basic cache set and get operations."""
        cache = ContextCache()

        # Set value
        cache.set("test_key", "test_value")
        assert len(cache._cache) == 1

        # Get value
        value = cache.get("test_key")
        assert value == "test_value"

    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        cache = ContextCache(ttl=0.1)  # Very short TTL

        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"

        # Wait for expiration
        import time

        time.sleep(0.2)

        # Should be expired
        assert cache.get("test_key") is None

    def test_cache_size_limit(self):
        """Test cache size limit and eviction."""
        cache = ContextCache(max_size=2)

        # Fill cache to limit
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache._cache) == 2

        # Add third item (should evict oldest)
        cache.set("key3", "value3")
        assert len(cache._cache) == 2
        assert cache.get("key1") is None  # Should be evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_clear(self):
        """Test cache clear functionality."""
        cache = ContextCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache._cache) == 2

        cache.clear()
        assert len(cache._cache) == 0
        assert len(cache._access_times) == 0


class TestGeminiMCPBase:
    """Test Gemini MCP base server functionality."""

    def test_server_initialization(self, test_gemini_server):
        """Test Gemini server initialization."""
        assert test_gemini_server.server_name == "test_gemini_server"
        assert test_gemini_server.model_type == "pro"
        assert test_gemini_server.config is not None
        assert test_gemini_server.context_cache is not None

    @pytest.mark.asyncio
    async def test_aiohttp_session_creation(self, test_gemini_server):
        """Test aiohttp session creation and configuration."""
        session = await test_gemini_server.get_aiohttp_session()

        assert session is not None
        assert isinstance(session, type(AsyncMock()))  # Mock session

        # Should reuse same session
        session2 = await test_gemini_server.get_aiohttp_session()
        assert session is session2

    @pytest.mark.asyncio
    async def test_content_generation_with_cache(self, test_gemini_server):
        """Test content generation with caching."""
        # Mock model response
        mock_response = Mock()
        mock_response.text = "Generated content"
        test_gemini_server.mock_model.generate_content.return_value = mock_response

        # First call - should hit model
        result1 = await test_gemini_server.generate_content("test prompt")
        assert result1 == "Generated content"
        test_gemini_server.mock_model.generate_content.assert_called_once()

        # Second call with same prompt - should hit cache
        test_gemini_server.mock_model.generate_content.reset_mock()
        result2 = await test_gemini_server.generate_content("test prompt")
        assert result2 == "Generated content"
        test_gemini_server.mock_model.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_content_generation_without_cache(self, test_gemini_server):
        """Test content generation without caching."""
        mock_response = Mock()
        mock_response.text = "Generated content"
        test_gemini_server.mock_model.generate_content.return_value = mock_response

        result = await test_gemini_server.generate_content("test prompt", use_cache=False)
        assert result == "Generated content"
        test_gemini_server.mock_model.generate_content.assert_called_once()

        # Second call should also hit model
        result2 = await test_gemini_server.generate_content("test prompt", use_cache=False)
        assert result2 == "Generated content"
        assert test_gemini_server.mock_model.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_streaming_generation(self, test_gemini_server):
        """Test streaming content generation."""
        # Mock streaming response
        mock_chunks = [Mock(text="chunk1"), Mock(text="chunk2"), Mock(text="chunk3")]
        test_gemini_server.mock_model.generate_content.return_value = mock_chunks

        chunks = []
        async for chunk in test_gemini_server._generate_streaming("test prompt"):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2", "chunk3"]

    @pytest.mark.asyncio
    async def test_context_collection_python_fallback(self, test_gemini_server, temp_project_dir):
        """Test context collection using Python fallback."""
        # Force Python fallback
        test_gemini_server.rust_file_ops = None

        context = await test_gemini_server.collect_context_fast(
            temp_project_dir, patterns=["*.py"], max_files=10
        )

        assert context["rust_accelerated"] is False
        assert context["file_count"] > 0
        assert len(context["files"]) > 0

        # Check that Python files were found
        python_files = [f for f in context["files"] if f["path"].endswith(".py")]
        assert len(python_files) > 0

        # Verify file content is included
        main_file = next((f for f in context["files"] if "main.py" in f["path"]), None)
        assert main_file is not None
        assert "def main" in main_file["content"]

    @pytest.mark.asyncio
    async def test_context_collection_with_patterns(self, test_gemini_server, temp_project_dir):
        """Test context collection with specific file patterns."""
        context = await test_gemini_server.collect_context_fast(
            temp_project_dir, patterns=["*.md", "*.txt"], max_files=10
        )

        assert context["file_count"] >= 2  # README.md and requirements.txt

        # Check that only specified file types were collected
        for file_info in context["files"]:
            assert file_info["path"].endswith((".md", ".txt"))

    @pytest.mark.asyncio
    async def test_context_collection_file_limit(self, test_gemini_server, temp_project_dir):
        """Test context collection with file limit."""
        context = await test_gemini_server.collect_context_fast(temp_project_dir, max_files=2)

        assert context["file_count"] <= 2
        assert len(context["files"]) <= 2

    @pytest.mark.asyncio
    async def test_analyze_with_rust_unavailable(self, test_gemini_server, temp_project_dir):
        """Test Rust analysis when Rust is not available."""
        # Force Rust unavailable
        test_gemini_server.rust_analyzer = None

        result = await test_gemini_server.analyze_with_rust(temp_project_dir)
        assert result == {}

    @pytest.mark.asyncio
    async def test_server_cleanup(self, test_gemini_server):
        """Test server resource cleanup."""
        # Create mock session
        mock_session = AsyncMock()
        test_gemini_server.session = mock_session

        await test_gemini_server.close()

        mock_session.close.assert_called_once()
        assert len(test_gemini_server.context_cache._cache) == 0


class TestMCPServerIntegration:
    """Test MCP server integration scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_servers_lifecycle(self):
        """Test managing multiple MCP servers."""
        servers = []

        # Create multiple servers
        for i in range(3):
            server = TestMCPServer(f"server_{i}")
            servers.append(server)

        # Initialize all servers
        for server in servers:
            await server.initialize()
            assert server.initialized

        # Shutdown all servers
        for server in servers:
            await server.shutdown()
            assert not server.initialized

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, test_gemini_server):
        """Test concurrent operations on MCP server."""
        # Mock model response
        mock_response = Mock()
        mock_response.text = "Concurrent response"
        test_gemini_server.mock_model.generate_content.return_value = mock_response

        # Run multiple concurrent operations
        tasks = []
        for i in range(5):
            task = test_gemini_server.generate_content(f"prompt_{i}", use_cache=False)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r == "Concurrent response" for r in results)
        assert test_gemini_server.mock_model.generate_content.call_count == 5

    @pytest.mark.asyncio
    async def test_error_handling_in_generation(self, test_gemini_server):
        """Test error handling during content generation."""
        # Mock model to raise exception
        test_gemini_server.mock_model.generate_content.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await test_gemini_server.generate_content("test prompt")

    @pytest.mark.asyncio
    async def test_cache_behavior_across_operations(self, test_gemini_server):
        """Test cache behavior across different operations."""
        mock_response = Mock()
        mock_response.text = "Cached response"
        test_gemini_server.mock_model.generate_content.return_value = mock_response

        # First generation - should cache
        result1 = await test_gemini_server.generate_content("cache test")
        assert result1 == "Cached response"

        # Clear model mock calls
        test_gemini_server.mock_model.generate_content.reset_mock()

        # Second generation with same prompt - should use cache
        result2 = await test_gemini_server.generate_content("cache test")
        assert result2 == "Cached response"
        test_gemini_server.mock_model.generate_content.assert_not_called()

        # Different prompt - should call model
        result3 = await test_gemini_server.generate_content("different prompt")
        assert result3 == "Cached response"
        test_gemini_server.mock_model.generate_content.assert_called_once()


class TestMCPServerConfiguration:
    """Test MCP server configuration and environment handling."""

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with invalid environment
        with patch.dict(
            os.environ, {"GOOGLE_CLOUD_PROJECT": "", "GOOGLE_GENAI_USE_VERTEXAI": "invalid"}
        ):
            config = GeminiMCPConfig()
            assert config.project == ""  # Should handle empty values
            assert config.use_vertex is False  # Should default to False for invalid values

    def test_model_selection(self, gemini_config):
        """Test model selection logic."""
        # Test different model types
        model_types = ["master", "flash", "pro"]

        for model_type in model_types:
            assert model_type in gemini_config.models
            model_name = gemini_config.models[model_type]
            assert model_name.startswith("gemini-")

    def test_performance_settings(self, gemini_config):
        """Test performance-related settings."""
        assert gemini_config.max_retries > 0
        assert gemini_config.timeout > 0
        assert gemini_config.max_concurrent_requests > 0
        assert gemini_config.cache_ttl > 0

        # Test context limits
        for _model, limit in gemini_config.max_context_size.items():
            assert limit > 0
            assert isinstance(limit, int)


class TestMCPServerFactories:
    """Test MCP server factory functions."""

    def test_create_mcp_server(self):
        """Test MCP server factory function."""
        from app.mcp_servers.gemini_mcp_base import create_mcp_server

        with patch.object(GeminiMCPBase, "_initialize_model"):
            server = create_mcp_server("test_server", "flash")

            assert server.server_name == "test_server"
            assert server.model_type == "flash"
            assert isinstance(server, GeminiMCPBase)

    def test_factory_with_defaults(self):
        """Test factory function with default parameters."""
        from app.mcp_servers.gemini_mcp_base import create_mcp_server

        with patch.object(GeminiMCPBase, "_initialize_model"):
            server = create_mcp_server("default_server")

            assert server.server_name == "default_server"
            assert server.model_type == "pro"  # Default model type


@pytest.mark.integration
class TestMCPServerEndToEnd:
    """End-to-end integration tests for MCP servers."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, temp_project_dir):
        """Test complete MCP server workflow."""
        with patch.object(GeminiMCPBase, "_initialize_model"):
            server = TestGeminiMCPServer("workflow_server")

            try:
                # Initialize server
                await server.initialize()

                # Collect project context
                context = await server.collect_context_fast(temp_project_dir)
                assert context["file_count"] > 0

                # Generate content based on context
                mock_response = Mock()
                mock_response.text = "Analysis complete"
                server.mock_model.generate_content.return_value = mock_response

                analysis_prompt = f"Analyze this project with {context['file_count']} files"
                result = await server.generate_content(analysis_prompt)
                assert result == "Analysis complete"

                # Test caching behavior
                result2 = await server.generate_content(analysis_prompt)
                assert result2 == "Analysis complete"

            finally:
                await server.close()

    @pytest.mark.asyncio
    async def test_error_recovery(self, temp_project_dir):
        """Test error recovery in MCP servers."""
        with patch.object(GeminiMCPBase, "_initialize_model"):
            server = TestGeminiMCPServer("recovery_server")

            try:
                # Test recovery from context collection error
                with patch.object(
                    server, "_collect_context_python", side_effect=Exception("Collection error")
                ):
                    context = await server.collect_context_fast(temp_project_dir)
                    # Should return empty context instead of crashing
                    assert context["file_count"] == 0

                # Test recovery from generation error
                server.mock_model.generate_content.side_effect = [
                    Exception("First error"),
                    Mock(text="Recovery successful"),
                ]

                # First call should fail
                with pytest.raises(Exception, match="First error"):
                    await server.generate_content("test prompt", use_cache=False)

                # Second call should succeed (simulating retry)
                server.mock_model.generate_content.side_effect = Mock(text="Recovery successful")
                result = await server.generate_content("test prompt", use_cache=False)
                assert result == "Recovery successful"

            finally:
                await server.close()


@pytest.mark.performance
class TestMCPServerPerformance:
    """Performance tests for MCP servers."""

    @pytest.mark.asyncio
    async def test_cache_performance(self, test_gemini_server):
        """Test cache performance with many operations."""
        import time

        mock_response = Mock()
        mock_response.text = "Cached content"
        test_gemini_server.mock_model.generate_content.return_value = mock_response

        # Generate content to populate cache
        await test_gemini_server.generate_content("performance test")

        # Measure cache hit performance
        start_time = time.time()

        for _ in range(100):
            result = await test_gemini_server.generate_content("performance test")
            assert result == "Cached content"

        end_time = time.time()
        cache_time = end_time - start_time

        # Cache hits should be very fast
        assert cache_time < 1.0  # Should complete in less than 1 second

        # Verify only one model call was made (initial population)
        test_gemini_server.mock_model.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_context_collection(self, temp_project_dir):
        """Test concurrent context collection performance."""
        with patch.object(GeminiMCPBase, "_initialize_model"):
            server = TestGeminiMCPServer("perf_server")

            try:
                # Run multiple concurrent context collections
                tasks = []
                for _i in range(5):
                    task = server.collect_context_fast(
                        temp_project_dir, patterns=["*.py"], max_files=5
                    )
                    tasks.append(task)

                import time

                start_time = time.time()
                results = await asyncio.gather(*tasks)
                end_time = time.time()

                # Should complete reasonably quickly
                assert end_time - start_time < 10.0

                # All results should be successful
                for result in results:
                    assert result["file_count"] > 0

            finally:
                await server.close()

    def test_cache_memory_usage(self):
        """Test cache memory usage with large datasets."""
        cache = ContextCache(max_size=1000)

        # Add many items to cache
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}" * 100)  # ~600 bytes per item

        assert len(cache._cache) == 1000

        # Add more items (should evict old ones)
        for i in range(100):
            cache.set(f"new_key_{i}", f"new_value_{i}" * 100)

        # Cache should maintain size limit
        assert len(cache._cache) == 1000
