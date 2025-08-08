"""Comprehensive test configuration for the Gemini CLI Terminal.
Configures pytest with fixtures, mocks, and test utilities.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure asyncio for testing
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def project_root():
    """Provide path to project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def gemini_cli_directory(project_root):
    """Provide path to gemini_cli directory."""
    return project_root / "gemini_cli"


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary test data directory."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()

    # Create sample files
    (data_dir / "sample.py").write_text("print('test')")
    (data_dir / "data.json").write_text('{"test": "data"}')
    (data_dir / "README.md").write_text("# Test Project")
    (data_dir / "config.yaml").write_text("key: value")

    yield data_dir

    # Cleanup
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def mock_gemini_client():
    """Create a mock Gemini client with all necessary methods."""
    mock = MagicMock()

    # Configure basic response
    mock.generate_content = AsyncMock()
    mock.generate_content.return_value.text = json.dumps(
        {"status": "success", "data": "Mock response", "timestamp": datetime.now().isoformat()},
    )

    # Configure streaming response
    mock.generate_content_stream = AsyncMock()

    async def mock_stream():
        for chunk in ["Mock ", "streaming ", "response"]:
            yield MagicMock(text=chunk)

    mock.generate_content_stream.return_value = mock_stream()

    # Configure chat session
    mock.start_chat = MagicMock()
    chat_session = MagicMock()
    chat_session.send_message = AsyncMock(return_value=MagicMock(text="Chat response"))
    mock.start_chat.return_value = chat_session

    # Configure count tokens
    mock.count_tokens = AsyncMock(return_value=MagicMock(total_tokens=100))

    return mock


@pytest.fixture
def mock_vertex_ai_client():
    """Create mock Vertex AI client for testing."""
    mock = MagicMock()

    # Configure generative model
    mock.GenerativeModel = MagicMock()
    model_instance = MagicMock()
    model_instance.generate_content = AsyncMock(return_value=MagicMock(text="Vertex AI response"))
    model_instance.start_chat = MagicMock()

    # Configure chat session
    chat_session = MagicMock()
    chat_session.send_message = AsyncMock(return_value=MagicMock(text="Chat response"))
    model_instance.start_chat.return_value = chat_session

    mock.GenerativeModel.return_value = model_instance

    return mock


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing external API calls."""
    mock = AsyncMock()

    # Configure response
    response = MagicMock()
    response.status = 200
    response.json = AsyncMock(return_value={"status": "success"})
    response.text = AsyncMock(return_value="Success")

    # Configure client methods
    mock.get = AsyncMock(return_value=response)
    mock.post = AsyncMock(return_value=response)
    mock.put = AsyncMock(return_value=response)
    mock.delete = AsyncMock(return_value=response)

    return mock


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for testing."""
    server = MagicMock()
    server.tool = MagicMock(return_value=lambda func: func)
    server.resource = MagicMock(return_value=lambda func: func)
    server.start = AsyncMock()
    server.stop = AsyncMock()
    server.handle_request = AsyncMock(return_value={"result": "success"})
    return server


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "api_base_url": "http://localhost:8000",
        "test_timeout": 30,
        "mock_external_apis": True,
        "log_level": "DEBUG",
        "gemini": {
            "project_id": "test-project",
            "location": "us-central1",
            "model": "gemini-2.5-flash",
        },
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch) -> None:
    """Automatically set up test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    monkeypatch.setenv("VERTEX_AI_REGION", "us-central1")


@pytest.fixture
def capture_logs():
    """Capture and return logs for testing."""
    from io import StringIO
    import logging

    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)

    # Get root logger
    logger = logging.getLogger()
    logger.addHandler(ch)

    yield log_capture_string

    # Clean up
    logger.removeHandler(ch)


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for testing external command execution."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
        yield mock_run


@pytest.fixture
def mock_async_subprocess():
    """Mock async subprocess for testing."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"Output", b""))
        process.returncode = 0
        mock_exec.return_value = process
        yield mock_exec


# Markers for test categorization
def pytest_configure(config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "mcp: MCP compliance tests")
    config.addinivalue_line("markers", "slow: Tests that take a long time")
    config.addinivalue_line("markers", "requires_api_key: Tests requiring API keys")
    config.addinivalue_line("markers", "cli: CLI interface tests")
    config.addinivalue_line("markers", "gemini: Gemini API tests")


# Test data factories
@pytest.fixture
def create_test_project():
    """Factory for creating test project data."""

    def _create(name="Test Project", **kwargs):
        return {
            "id": kwargs.get("id", "proj_123"),
            "name": name,
            "path": kwargs.get("path", f"/test/{name.lower().replace(' ', '_')}"),
            "language": kwargs.get("language", "python"),
            "framework": kwargs.get("framework", "fastapi"),
            "created_at": kwargs.get("created_at", datetime.now().isoformat()),
            "updated_at": kwargs.get("updated_at", datetime.now().isoformat()),
        }

    return _create


@pytest.fixture
def gemini_test_prompts():
    """Collection of test prompts for different scenarios."""
    return {
        "simple_question": "What is the capital of France?",
        "code_generation": "Generate a Python function to calculate factorial",
        "code_analysis": "Analyze this Python code for potential improvements",
        "technical_explanation": "Explain how async/await works in Python",
        "problem_solving": "How would you implement a rate limiter in Python?",
        "review_request": "Review this code and suggest improvements",
        "documentation": "Generate documentation for this function",
        "testing": "Write unit tests for this code",
        "debugging": "Help debug this error in my Python code",
        "architecture": "Design a scalable web application architecture",
    }


@pytest.fixture
def mock_gemini_responses():
    """Mock responses for different Gemini scenarios."""
    return {
        "success": {
            "text": "This is a successful response from Gemini",
            "candidates": [{"content": {"parts": [{"text": "Success response"}]}}],
        },
        "code_generation": {
            "text": '''def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)''',
        },
        "analysis": {
            "text": json.dumps(
                {
                    "analysis": {
                        "summary": "Code looks good overall",
                        "issues": ["Minor style issue"],
                        "suggestions": ["Use type hints"],
                        "score": 85,
                    }
                }
            ),
        },
        "empty": {
            "text": "",
            "candidates": [],
        },
        "error": {
            "text": "",
            "error": {
                "code": 400,
                "message": "Invalid request",
                "status": "INVALID_ARGUMENT",
            },
        },
    }


@pytest.fixture
def cli_runner():
    """Click CLI test runner."""
    from click.testing import CliRunner

    return CliRunner()


@pytest.fixture
def mock_terminal_session():
    """Mock terminal session for testing interactive features."""

    class MockTerminalSession:
        def __init__(self):
            self.history = []
            self.current_prompt = "gemini> "

        def add_input(self, text: str):
            self.history.append({"type": "input", "content": text})

        def add_output(self, text: str):
            self.history.append({"type": "output", "content": text})

        def get_history(self):
            return self.history

        def clear(self):
            self.history = []

    return MockTerminalSession()


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing."""

    class MockFileOps:
        def __init__(self):
            self.files = {}

        def write_file(self, path: str, content: str):
            self.files[path] = content

        def read_file(self, path: str) -> str:
            return self.files.get(path, "")

        def exists(self, path: str) -> bool:
            return path in self.files

        def list_files(self) -> list[str]:
            return list(self.files.keys())

    return MockFileOps()


@pytest.fixture
def sample_code_files():
    """Sample code files for testing code analysis features."""
    return {
        "simple.py": """
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
""",
        "with_issues.py": """
import os, sys
def badFunction():
    x=1+1
    print(x)
    return
""",
        "good_example.py": '''
"""Well-documented Python module."""

from typing import List, Optional


def calculate_sum(numbers: List[int]) -> int:
    """Calculate the sum of a list of numbers.

    Args:
        numbers: List of integers to sum

    Returns:
        The sum of all numbers
    """
    return sum(numbers)


def find_max(numbers: List[int]) -> Optional[int]:
    """Find the maximum number in a list.

    Args:
        numbers: List of integers

    Returns:
        Maximum number or None if list is empty
    """
    if not numbers:
        return None
    return max(numbers)
''',
    }


@pytest.fixture
def mcp_test_config():
    """Test MCP configuration."""
    return {
        "mcpServers": {
            "test-server": {
                "command": "python",
                "args": ["-m", "test_server"],
                "env": {"TEST_MODE": "true"},
                "timeout": 30,
            },
            "gemini-server": {
                "command": "uv",
                "args": ["run", "python", "-m", "mcp.gemini_server"],
                "env": {"GOOGLE_CLOUD_PROJECT": "test-project"},
            },
        }
    }


@pytest.fixture
def performance_test_data():
    """Test data for performance testing."""
    return {
        "small_text": "This is a small test text.",
        "medium_text": "This is a medium test text. " * 100,
        "large_text": "This is a large test text. " * 1000,
        "code_snippet": """
def example_function():
    for i in range(100):
        print(f"Item {i}")
"""
        * 10,
    }


# Consolidation-specific fixtures
@pytest.fixture
def consolidation_test_data():
    """Test data for consolidation validation."""
    return {
        "expected_gterminal_modules": [
            "agents",
            "auth",
            "cache",
            "core",
            "terminal",
            "gemini_cli",
            "utils",
            "mcp_servers",
        ],
        "legacy_patterns": [
            r"\bfrom\s+gapp\.",
            r"\bimport\s+gapp\b",
            r"\bfrom\s+app\.",  # But not gapp
        ],
        "gterminal_patterns": [
            r"\bfrom\s+gterminal\.",
            r"\bimport\s+gterminal\b",
        ],
    }


@pytest.fixture
def mock_gterminal_agent():
    """Mock gterminal agent for testing."""
    mock = MagicMock()

    # Configure agent methods
    mock.analyze_code = AsyncMock()
    mock.analyze_code.return_value = {
        "analysis": "Code analysis complete",
        "issues": [],
        "suggestions": ["Add type hints"],
        "score": 85,
    }

    mock.review_code = AsyncMock()
    mock.review_code.return_value = {
        "summary": "Code review complete",
        "issues": [],
        "recommendations": ["Consider refactoring"],
    }

    mock.generate_documentation = AsyncMock()
    mock.generate_documentation.return_value = (
        "# Generated Documentation\n\nThis is generated documentation."
    )

    return mock


@pytest.fixture
def mock_consolidation_cache():
    """Mock cache for consolidation testing."""

    class MockConsolidationCache:
        def __init__(self):
            self.data = {}

        async def get(self, key: str):
            return self.data.get(key)

        async def set(self, key: str, value, ttl=None):
            self.data[key] = value

        async def delete(self, key: str):
            if key in self.data:
                del self.data[key]

        async def clear(self):
            self.data.clear()

        def keys(self):
            return list(self.data.keys())

    return MockConsolidationCache()


@pytest.fixture
def gterminal_project_structure(project_root):
    """Provide gterminal project structure information."""
    gterminal_path = project_root / "gterminal"

    structure = {
        "root": gterminal_path,
        "agents": gterminal_path / "agents",
        "core": gterminal_path / "core",
        "terminal": gterminal_path / "terminal",
        "auth": gterminal_path / "auth",
        "cache": gterminal_path / "cache",
        "utils": gterminal_path / "utils",
        "mcp_servers": gterminal_path / "mcp_servers",
        "gemini_cli": gterminal_path / "gemini_cli",
    }

    return structure


@pytest.fixture
def consolidation_validation_config():
    """Configuration for consolidation validation tests."""
    return {
        "max_legacy_references": 5,  # Maximum allowed legacy references
        "min_gterminal_references": 10,  # Minimum expected gterminal references
        "required_modules": [
            "gterminal.agents",
            "gterminal.core.agents",
            "gterminal.terminal.react_engine",
            "gterminal.auth.gcp_auth",
            "gterminal.cache.cache_manager",
        ],
        "deprecated_modules": [
            "gapp.agents",
            "gapp.core",
            "app.agents",  # But not gapp
        ],
    }


@pytest.fixture
def mock_mcp_consolidation_server():
    """Mock MCP server for consolidation testing."""

    class MockMCPConsolidationServer:
        def __init__(self):
            self.tools = {}
            self.resources = {}

        def tool(self, func=None):
            """Mock tool decorator."""
            if func is None:
                return self.tool

            self.tools[func.__name__] = func
            return func

        def resource(self, func=None):
            """Mock resource decorator."""
            if func is None:
                return self.resource

            self.resources[func.__name__] = func
            return func

        async def handle_request(self, request):
            """Mock request handler."""
            return {"result": "success", "data": "mock response"}

        def get_tools(self):
            """Get registered tools."""
            return list(self.tools.keys())

        def get_resources(self):
            """Get registered resources."""
            return list(self.resources.keys())

    return MockMCPConsolidationServer()


@pytest.fixture
def performance_benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        "max_import_time": 5.0,  # Maximum time for imports in seconds
        "max_agent_init_time": 2.0,  # Maximum time for agent initialization
        "cache_operations_per_second": 1000,  # Expected cache performance
        "max_memory_usage_mb": 500,  # Maximum memory usage for tests
    }


@pytest.fixture
def integration_test_timeout():
    """Timeout configuration for integration tests."""
    return {
        "short": 5,  # 5 seconds for quick operations
        "medium": 30,  # 30 seconds for normal operations
        "long": 120,  # 2 minutes for complex operations
    }
