"""
Comprehensive unit tests for GeminiMCPBase class and related components.

Tests initialization, configuration, model management, caching, context collection,
Rust integration, and all core MCP functionality with extensive mocking.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import mock_open
from unittest.mock import patch

from app.mcp_servers.gemini_mcp_base import ContextCache
from app.mcp_servers.gemini_mcp_base import GeminiMCPBase
from app.mcp_servers.gemini_mcp_base import GeminiMCPConfig
from app.mcp_servers.gemini_mcp_base import create_mcp_server
from app.mcp_servers.gemini_mcp_base import get_secure_credentials_path
from app.mcp_servers.gemini_mcp_base import sanitize_input
from app.mcp_servers.gemini_mcp_base import validate_project_id
import pytest


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_secure_credentials_path_env_var(self):
        """Test getting credentials path from environment variable."""
        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/test/creds.json"}):
            with patch("pathlib.Path.exists", return_value=True):
                path = get_secure_credentials_path()

        assert path == "/test/creds.json"

    def test_get_secure_credentials_path_standard_locations(self):
        """Test getting credentials from standard locations."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.exists") as mock_exists:
                # First two locations don't exist, third one does
                mock_exists.side_effect = [False, False, True]

                path = get_secure_credentials_path()

                assert path is not None
                assert "personal" in path

    def test_get_secure_credentials_path_none_found(self):
        """Test when no credentials are found."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.exists", return_value=False):
                path = get_secure_credentials_path()

        assert path is None

    def test_validate_project_id_valid(self):
        """Test valid project ID validation."""
        assert validate_project_id("test-project-123") is True
        assert validate_project_id("my_project") is True
        assert validate_project_id("simple123") is True

    def test_validate_project_id_invalid(self):
        """Test invalid project ID validation."""
        assert validate_project_id("") is False
        assert validate_project_id(None) is False
        assert validate_project_id("a") is False  # Too short
        assert validate_project_id("a" * 31) is False  # Too long
        assert validate_project_id("project@invalid") is False  # Invalid character
        assert validate_project_id("Project-Name") is False  # Uppercase

    def test_sanitize_input_basic(self):
        """Test basic input sanitization."""
        result = sanitize_input("  test input  ", 100)
        assert result == "test input"

    def test_sanitize_input_length_limit(self):
        """Test input length limiting."""
        long_input = "a" * 200
        result = sanitize_input(long_input, 50)
        assert len(result) == 50

    def test_sanitize_input_control_characters(self):
        """Test removal of control characters."""
        input_with_controls = "test\x00\x01\x02valid\ttab\nline"
        result = sanitize_input(input_with_controls)

        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x02" not in result
        assert "test" in result
        assert "valid" in result
        assert "\t" in result  # Tab should be preserved
        assert "\n" in result  # Newline should be preserved

    def test_sanitize_input_invalid_type(self):
        """Test sanitization with invalid input type."""
        with pytest.raises(ValueError, match="Input must be a string"):
            sanitize_input(123)


class TestGeminiMCPConfig:
    """Test GeminiMCPConfig class."""

    def test_initialization_with_env_vars(self):
        """Test config initialization with environment variables."""
        env_vars = {
            "GOOGLE_CLOUD_PROJECT": "test-project-123",
            "GOOGLE_CLOUD_LOCATION": "us-west1",
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GEMINI_MASTER_MODEL": "gemini-2.5-pro-test",
            "GEMINI_MAX_RETRIES": "5",
            "GEMINI_TIMEOUT_SECONDS": "120",
            "GEMINI_MAX_CONCURRENT": "20",
            "GEMINI_CACHE_TTL": "7200",
            "GEMINI_STREAMING_ENABLED": "false",
        }

        with patch.dict(os.environ, env_vars), patch(
            "app.mcp_servers.gemini_mcp_base.get_secure_credentials_path",
            return_value="/test/creds.json",
        ):
            config = GeminiMCPConfig()

        assert config.project == "test-project-123"
        assert config.location == "us-west1"
        assert config.use_vertex is True
        assert config.models["master"] == "gemini-2.5-pro-test"
        assert config.max_retries == 5
        assert config.timeout == 120
        assert config.max_concurrent_requests == 20
        assert config.cache_ttl == 7200
        assert config.streaming_enabled is False

    def test_initialization_defaults(self):
        """Test config initialization with defaults."""
        with patch.dict(os.environ, {}, clear=True), patch(
            "app.mcp_servers.gemini_mcp_base.get_secure_credentials_path", return_value=None
        ):
            config = GeminiMCPConfig()

        assert config.project == ""
        assert config.location == "us-central1"
        assert config.use_vertex is True
        assert config.models["master"] == "gemini-2.5-pro"
        assert config.models["flash"] == "gemini-2.0-flash"
        assert config.models["pro"] == "gemini-2.5-pro"
        assert config.max_retries == 3
        assert config.timeout == 60
        assert config.max_concurrent_requests == 10
        assert config.cache_ttl == 3600
        assert config.streaming_enabled is True

    def test_invalid_project_id_handling(self):
        """Test handling of invalid project ID."""
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "invalid@project"}), patch(
            "app.mcp_servers.gemini_mcp_base.get_secure_credentials_path", return_value=None
        ):
            config = GeminiMCPConfig()

        assert config.project == ""

    def test_get_model_context_limit(self):
        """Test getting model context limits."""
        config = GeminiMCPConfig()

        assert config.get_model_context_limit("gemini-2.5-pro") == 1000000
        assert config.get_model_context_limit("gemini-2.0-flash") == 32000
        assert config.get_model_context_limit("unknown-model") == 32000  # Default

    def test_validation_warnings(self, caplog):
        """Test configuration validation warnings."""
        env_vars = {
            "GEMINI_MAX_RETRIES": "15",  # Too high
            "GEMINI_TIMEOUT_SECONDS": "5",  # Too low
            "GEMINI_MAX_CONCURRENT": "200",  # Too high
        }

        with patch.dict(os.environ, env_vars), patch(
            "app.mcp_servers.gemini_mcp_base.get_secure_credentials_path", return_value=None
        ):
            GeminiMCPConfig()

        # Should have generated warnings
        assert "Unusual max_retries value" in caplog.text
        assert "Unusual timeout value" in caplog.text
        assert "Unusual max_concurrent_requests" in caplog.text


class TestContextCache:
    """Test ContextCache class."""

    @pytest.fixture
    def cache(self):
        """Create a test cache."""
        return ContextCache(max_size=3, ttl=1)

    def test_initialization(self, cache):
        """Test cache initialization."""
        assert cache._max_size == 3
        assert cache._ttl == 1
        assert isinstance(cache._cache, dict)
        assert isinstance(cache._access_times, dict)

    def test_set_and_get_valid(self, cache):
        """Test setting and getting valid cache items."""
        cache.set("key1", "value1")

        result = cache.get("key1")
        assert result == "value1"

    def test_get_nonexistent(self, cache):
        """Test getting nonexistent key."""
        result = cache.get("nonexistent")
        assert result is None

    def test_get_with_invalid_key(self, cache):
        """Test getting with invalid key."""
        assert cache.get("") is None
        assert cache.get(None) is None

    def test_set_with_invalid_key(self, cache):
        """Test setting with invalid key."""
        cache.set("", "value")  # Should be ignored
        cache.set(None, "value")  # Should be ignored

        assert len(cache._cache) == 0

    def test_ttl_expiration(self, cache):
        """Test TTL expiration."""
        cache.set("key1", "value1")

        # Wait for expiration
        time.sleep(1.1)

        result = cache.get("key1")
        assert result is None
        assert "key1" not in cache._cache

    def test_eviction_by_size(self, cache):
        """Test eviction when max size is reached."""
        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Add one more - should evict oldest
        cache.set("key4", "value4")

        assert len(cache._cache) == 3
        assert cache.get("key1") is None  # Should be evicted
        assert cache.get("key4") == "value4"

    def test_access_time_update(self, cache):
        """Test that access updates access time."""
        cache.set("key1", "value1")
        original_time = cache._access_times["key1"]

        time.sleep(0.1)
        cache.get("key1")  # Should update access time

        assert cache._access_times["key1"] > original_time

    def test_clear(self, cache):
        """Test cache clearing."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert len(cache._cache) == 0
        assert len(cache._access_times) == 0

    def test_get_stats(self, cache):
        """Test getting cache statistics."""
        cache.set("key1", "value1")

        stats = cache.get_stats()

        assert stats["size"] == 1
        assert stats["max_size"] == 3
        assert stats["ttl_seconds"] == 1
        assert "hit_rate" in stats


class TestGeminiMCPBase:
    """Test GeminiMCPBase class."""

    @pytest.fixture
    def mock_vertexai(self):
        """Mock vertexai module."""
        with patch("app.mcp_servers.gemini_mcp_base.vertexai") as mock:
            yield mock

    @pytest.fixture
    def mock_generative_model(self):
        """Mock GenerativeModel class."""
        with patch("app.mcp_servers.gemini_mcp_base.GenerativeModel") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_config(self):
        """Mock GeminiMCPConfig."""
        config = MagicMock()
        config.use_vertex = True
        config.project = "test-project"
        config.location = "us-central1"
        config.credentials_path = "/test/creds.json"
        config.models = {"pro": "gemini-2.5-pro"}
        config.timeout = 60
        config.streaming_enabled = True
        return config

    @pytest.fixture
    def mock_mcp_base(self, mock_vertexai, mock_generative_model, mock_config):
        """Create a mock GeminiMCPBase instance."""
        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPConfig", return_value=mock_config):
            with patch("app.mcp_servers.gemini_mcp_base.RUST_AVAILABLE", False):
                return GeminiMCPBase("test-server", "pro")

    def test_initialization_valid(self, mock_mcp_base):
        """Test valid MCP base initialization."""
        assert mock_mcp_base.server_name == "test-server"
        assert mock_mcp_base.model_type == "pro"
        assert isinstance(mock_mcp_base.context_cache, ContextCache)
        assert isinstance(mock_mcp_base.executor, ThreadPoolExecutor)

    def test_initialization_invalid_name(self, mock_vertexai, mock_generative_model, mock_config):
        """Test initialization with invalid server name."""
        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPConfig", return_value=mock_config):
            with pytest.raises(ValueError, match="Server name must be a non-empty string"):
                GeminiMCPBase("", "pro")

    def test_initialization_with_rust(self, mock_vertexai, mock_generative_model, mock_config):
        """Test initialization with Rust components available."""
        mock_rust_module = MagicMock()
        mock_rust_module.PyFileOps.return_value = MagicMock()
        mock_rust_module.PyCodeAnalyzer.return_value = MagicMock()

        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPConfig", return_value=mock_config):
            with patch("app.mcp_servers.gemini_mcp_base.RUST_AVAILABLE", True):
                with patch("app.mcp_servers.gemini_mcp_base.rust_llm_py", mock_rust_module):
                    mcp_base = GeminiMCPBase("test-server", "pro")

        assert mcp_base.rust_file_ops is not None
        assert mcp_base.rust_analyzer is not None

    def test_initialization_rust_failure(
        self, mock_vertexai, mock_generative_model, mock_config, caplog
    ):
        """Test initialization with Rust components failing."""
        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPConfig", return_value=mock_config):
            with patch("app.mcp_servers.gemini_mcp_base.RUST_AVAILABLE", True):
                with patch("app.mcp_servers.gemini_mcp_base.rust_llm_py") as mock_rust:
                    mock_rust.PyFileOps.side_effect = Exception("Rust init failed")

                    mcp_base = GeminiMCPBase("test-server", "pro")

        assert "Failed to initialize Rust components" in caplog.text
        assert mcp_base.rust_file_ops is None

    def test_model_initialization_vertex_disabled(self, mock_config):
        """Test model initialization with Vertex AI disabled."""
        mock_config.use_vertex = False

        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPConfig", return_value=mock_config):
            with patch("app.mcp_servers.gemini_mcp_base.RUST_AVAILABLE", False):
                mcp_base = GeminiMCPBase("test-server", "pro")

        assert mcp_base.model is None

    def test_model_initialization_no_project(self, mock_vertexai, mock_config):
        """Test model initialization without project ID."""
        mock_config.project = ""

        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPConfig", return_value=mock_config):
            with pytest.raises(ValueError, match="Google Cloud project ID is required"):
                GeminiMCPBase("test-server", "pro")

    def test_model_initialization_no_credentials(
        self, mock_vertexai, mock_generative_model, mock_config, caplog
    ):
        """Test model initialization without credentials."""
        mock_config.credentials_path = None

        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPConfig", return_value=mock_config):
            with caplog.at_level("WARNING"):
                GeminiMCPBase("test-server", "pro")

        assert "No credentials found" in caplog.text

    def test_model_initialization_vertex_failure(self, mock_config):
        """Test model initialization with Vertex AI failure."""
        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPConfig", return_value=mock_config):
            with patch(
                "app.mcp_servers.gemini_mcp_base.vertexai.init",
                side_effect=Exception("Vertex init failed"),
            ):
                with pytest.raises(ValueError, match="Model initialization failed"):
                    GeminiMCPBase("test-server", "pro")

    @pytest.mark.asyncio
    async def test_get_aiohttp_session_new(self, mock_mcp_base):
        """Test getting new aiohttp session."""
        session = await mock_mcp_base.get_aiohttp_session()

        assert session is not None
        assert mock_mcp_base.session is session

    @pytest.mark.asyncio
    async def test_get_aiohttp_session_reuse(self, mock_mcp_base):
        """Test reusing existing aiohttp session."""
        session1 = await mock_mcp_base.get_aiohttp_session()
        session2 = await mock_mcp_base.get_aiohttp_session()

        assert session1 is session2

    @pytest.mark.asyncio
    async def test_get_aiohttp_session_closed(self, mock_mcp_base):
        """Test getting session when existing one is closed."""
        session1 = await mock_mcp_base.get_aiohttp_session()

        # Mock the closed property to return True
        type(session1).closed = PropertyMock(return_value=True)

        session2 = await mock_mcp_base.get_aiohttp_session()

        assert session1 is not session2

    def test_create_cache_key(self, mock_mcp_base):
        """Test cache key creation."""
        key1 = mock_mcp_base._create_cache_key("test prompt", param1="value1")
        key2 = mock_mcp_base._create_cache_key("test prompt", param1="value1")
        key3 = mock_mcp_base._create_cache_key("different prompt", param1="value1")

        assert key1 == key2  # Same inputs should produce same key
        assert key1 != key3  # Different inputs should produce different keys
        assert len(key1) == 32  # MD5 hash length

    @pytest.mark.asyncio
    async def test_generate_content_success(self, mock_mcp_base):
        """Test successful content generation."""
        mock_response = MagicMock()
        mock_response.text = "Generated content"
        mock_mcp_base.model.generate_content.return_value = mock_response

        result = await mock_mcp_base.generate_content("test prompt")

        assert result == "Generated content"
        mock_mcp_base.model.generate_content.assert_called_once_with("test prompt")

    @pytest.mark.asyncio
    async def test_generate_content_invalid_prompt(self, mock_mcp_base):
        """Test content generation with invalid prompt."""
        with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
            await mock_mcp_base.generate_content("")

    @pytest.mark.asyncio
    async def test_generate_content_no_model(self, mock_mcp_base):
        """Test content generation without model."""
        mock_mcp_base.model = None

        with pytest.raises(ValueError, match="Model not initialized"):
            await mock_mcp_base.generate_content("test prompt")

    @pytest.mark.asyncio
    async def test_generate_content_with_cache(self, mock_mcp_base):
        """Test content generation with caching."""
        mock_response = MagicMock()
        mock_response.text = "Generated content"
        mock_mcp_base.model.generate_content.return_value = mock_response

        # First call
        result1 = await mock_mcp_base.generate_content("test prompt", use_cache=True)

        # Second call should use cache
        result2 = await mock_mcp_base.generate_content("test prompt", use_cache=True)

        assert result1 == result2
        # Model should only be called once
        assert mock_mcp_base.model.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_content_cache_disabled(self, mock_mcp_base):
        """Test content generation with cache disabled."""
        mock_response = MagicMock()
        mock_response.text = "Generated content"
        mock_mcp_base.model.generate_content.return_value = mock_response

        # Two calls with cache disabled
        result1 = await mock_mcp_base.generate_content("test prompt", use_cache=False)
        result2 = await mock_mcp_base.generate_content("test prompt", use_cache=False)

        assert result1 == result2
        # Model should be called twice
        assert mock_mcp_base.model.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_content_empty_response(self, mock_mcp_base):
        """Test content generation with empty response."""
        mock_response = MagicMock()
        mock_response.text = ""
        mock_mcp_base.model.generate_content.return_value = mock_response

        with pytest.raises(ValueError, match="Empty response from model"):
            await mock_mcp_base.generate_content("test prompt")

    @pytest.mark.asyncio
    async def test_generate_content_streaming(self, mock_mcp_base):
        """Test streaming content generation."""
        mock_mcp_base.config.streaming_enabled = True

        with patch.object(mock_mcp_base, "_generate_streaming") as mock_stream:
            mock_stream.return_value = AsyncMock()

            await mock_mcp_base.generate_content("test prompt", stream=True)

            mock_stream.assert_called_once_with("test prompt")

    @pytest.mark.asyncio
    async def test_generate_streaming_success(self, mock_mcp_base):
        """Test successful streaming generation."""
        # Create mock chunks
        mock_chunks = []
        for text in ["chunk1", "chunk2", "chunk3"]:
            chunk = MagicMock()
            chunk.text = text
            mock_chunks.append(chunk)

        mock_mcp_base.model.generate_content.return_value = mock_chunks

        result = []
        async for chunk in mock_mcp_base._generate_streaming("test prompt"):
            result.append(chunk)

        assert result == ["chunk1", "chunk2", "chunk3"]

    @pytest.mark.asyncio
    async def test_generate_streaming_no_model(self, mock_mcp_base):
        """Test streaming generation without model."""
        mock_mcp_base.model = None

        with pytest.raises(ValueError, match="Model not initialized"):
            async for _chunk in mock_mcp_base._generate_streaming("test prompt"):
                pass

    @pytest.mark.asyncio
    async def test_collect_context_fast_invalid_path(self, mock_mcp_base):
        """Test context collection with invalid path."""
        with pytest.raises(ValueError, match="Project path must be a non-empty string"):
            await mock_mcp_base.collect_context_fast("")

    @pytest.mark.asyncio
    async def test_collect_context_fast_nonexistent_path(self, mock_mcp_base):
        """Test context collection with nonexistent path."""
        with pytest.raises(ValueError, match="Project path does not exist"):
            await mock_mcp_base.collect_context_fast("/nonexistent/path")

    @pytest.mark.asyncio
    async def test_collect_context_fast_not_directory(self, mock_mcp_base, tmp_path):
        """Test context collection with file instead of directory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError, match="Project path is not a directory"):
            await mock_mcp_base.collect_context_fast(str(test_file))

    @pytest.mark.asyncio
    async def test_collect_context_fast_python_fallback(self, mock_mcp_base, tmp_path):
        """Test context collection using Python fallback."""
        # Create test project structure
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (tmp_path / "README.md").write_text("# Test Project")

        context = await mock_mcp_base.collect_context_fast(str(tmp_path))

        assert context["rust_accelerated"] is False
        assert context["file_count"] > 0
        assert any("main.py" in file["path"] for file in context["files"])

    @pytest.mark.asyncio
    async def test_collect_context_fast_with_patterns(self, mock_mcp_base, tmp_path):
        """Test context collection with specific patterns."""
        # Create test files
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "test.js").write_text("console.log('test')")
        (tmp_path / "README.md").write_text("# Test")

        context = await mock_mcp_base.collect_context_fast(str(tmp_path), patterns=["*.py"])

        # Should only include Python files
        python_files = [f for f in context["files"] if f["path"].endswith(".py")]
        js_files = [f for f in context["files"] if f["path"].endswith(".js")]

        assert len(python_files) > 0
        assert len(js_files) == 0

    @pytest.mark.asyncio
    async def test_collect_context_fast_max_files_limit(self, mock_mcp_base, tmp_path):
        """Test context collection with file limit."""
        # Create more files than the limit
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"# File {i}")

        context = await mock_mcp_base.collect_context_fast(str(tmp_path), max_files=5)

        assert context["file_count"] <= 5

    @pytest.mark.asyncio
    async def test_collect_context_python_large_files(self, mock_mcp_base, tmp_path):
        """Test Python context collection skips large files."""
        # Create a large file
        large_content = "x" * 200000  # 200KB
        (tmp_path / "large.py").write_text(large_content)
        (tmp_path / "small.py").write_text("print('small')")

        context = await mock_mcp_base._collect_context_python(str(tmp_path), ["*.py"], 100)

        # Should skip large file
        file_paths = [f["path"] for f in context["files"]]
        assert any("small.py" in path for path in file_paths)
        assert not any("large.py" in path for path in file_paths)

    @pytest.mark.asyncio
    async def test_collect_context_python_hidden_directories(self, mock_mcp_base, tmp_path):
        """Test Python context collection skips hidden directories."""
        # Create files in hidden directory
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "secret.py").write_text("secret code")
        (tmp_path / "visible.py").write_text("visible code")

        context = await mock_mcp_base._collect_context_python(str(tmp_path), ["*.py"], 100)

        # Should skip hidden files
        file_paths = [f["path"] for f in context["files"]]
        assert any("visible.py" in path for path in file_paths)
        assert not any("secret.py" in path for path in file_paths)

    def test_read_file_safe_success(self, mock_mcp_base):
        """Test successful file reading."""
        content = "test file content"

        with patch("builtins.open", mock_open(read_data=content)):
            result = mock_mcp_base._read_file_safe("test.txt")

        assert result == content

    def test_read_file_safe_failure(self, mock_mcp_base):
        """Test file reading failure."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            result = mock_mcp_base._read_file_safe("nonexistent.txt")

        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_with_rust_not_available(self, mock_mcp_base):
        """Test Rust analysis when not available."""
        mock_mcp_base.rust_analyzer = None

        result = await mock_mcp_base.analyze_with_rust("/test/path")

        assert result["error"] == "Rust analyzer not available"

    @pytest.mark.asyncio
    async def test_analyze_with_rust_invalid_path(self, mock_mcp_base):
        """Test Rust analysis with invalid path."""
        mock_mcp_base.rust_analyzer = MagicMock()

        with pytest.raises(ValueError, match="Project path must be a non-empty string"):
            await mock_mcp_base.analyze_with_rust("")

    @pytest.mark.asyncio
    async def test_analyze_with_rust_success(self, mock_mcp_base):
        """Test successful Rust analysis."""
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_project_structure.return_value = '{"files": 10}'
        mock_analyzer.find_security_issues.return_value = '{"issues": []}'
        mock_analyzer.analyze_dependencies.return_value = '{"deps": []}'
        mock_analyzer.performance_analysis.return_value = '{"score": 85}'

        mock_mcp_base.rust_analyzer = mock_analyzer

        result = await mock_mcp_base.analyze_with_rust("/test/path")

        assert "structure" in result
        assert "security" in result
        assert "dependencies" in result
        assert "performance" in result

    @pytest.mark.asyncio
    async def test_analyze_with_rust_failure(self, mock_mcp_base):
        """Test Rust analysis failure."""
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze_project_structure.side_effect = Exception("Analysis failed")

        mock_mcp_base.rust_analyzer = mock_analyzer

        result = await mock_mcp_base.analyze_with_rust("/test/path")

        assert "error" in result
        assert "Analysis failed" in result["error"]

    @pytest.mark.asyncio
    async def test_close_cleanup(self, mock_mcp_base):
        """Test resource cleanup on close."""
        # Setup resources
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_mcp_base.session = mock_session

        await mock_mcp_base.close()

        mock_session.close.assert_called_once()
        # Cache should be cleared
        assert len(mock_mcp_base.context_cache._cache) == 0

    @pytest.mark.asyncio
    async def test_close_with_exceptions(self, mock_mcp_base):
        """Test close with exceptions during cleanup."""
        # Setup failing session
        mock_session = AsyncMock()
        mock_session.close.side_effect = Exception("Close failed")
        mock_mcp_base.session = mock_session

        # Should not raise exception
        await mock_mcp_base.close()

    def test_get_server_stats(self, mock_mcp_base):
        """Test getting server statistics."""
        stats = mock_mcp_base.get_server_stats()

        assert stats["server_name"] == "test-server"
        assert stats["model_type"] == "pro"
        assert "rust_available" in stats
        assert "cache_stats" in stats
        assert "config" in stats
        assert stats["config"]["project"] == "test-project"


class TestCreateMCPServer:
    """Test create_mcp_server factory function."""

    def test_create_mcp_server_default(self):
        """Test creating MCP server with default model type."""
        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPBase") as mock_class:
            create_mcp_server("test-server")

            mock_class.assert_called_once_with("test-server", "pro")

    def test_create_mcp_server_custom_model(self):
        """Test creating MCP server with custom model type."""
        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPBase") as mock_class:
            create_mcp_server("test-server", "flash")

            mock_class.assert_called_once_with("test-server", "flash")
