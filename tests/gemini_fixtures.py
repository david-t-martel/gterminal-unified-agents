"""Gemini client fixtures for testing.
Provides mock Gemini clients and responses.
"""

from datetime import datetime
import json
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest


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
def mock_gemini_response():
    """Create mock Gemini responses for different scenarios."""

    def _create_response(response_type: str = "default", **kwargs):
        responses = {
            "default": {
                "text": "Default mock response",
                "candidates": [{"content": {"parts": [{"text": "Default mock response"}]}}],
            },
            "analysis": {
                "text": json.dumps(
                    {
                        "analysis": {
                            "summary": "Code analysis complete",
                            "issues": ["Issue 1", "Issue 2"],
                            "recommendations": ["Recommendation 1", "Recommendation 2"],
                            "metrics": {
                                "complexity": 5,
                                "maintainability": 85,
                                "test_coverage": 75,
                            },
                        },
                    },
                ),
            },
            "code_generation": {
                "text": """def example_function(param1: str, param2: int) -> Dict[str, Any]:
    \"\"\"Example generated function.\"\"\"
    result = {
        "input": param1,
        "processed": param2 * 2,
        "timestamp": datetime.now().isoformat()
    }
    return result""",
            },
            "error": {
                "text": "",
                "error": {
                    "code": 400,
                    "message": "Mock error response",
                    "status": "INVALID_ARGUMENT",
                },
            },
            "rate_limit": {
                "text": "",
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded",
                    "status": "RESOURCE_EXHAUSTED",
                },
            },
        }

        response = responses.get(response_type, responses["default"])
        response.update(kwargs)

        mock_response = MagicMock()
        mock_response.text = response.get("text", "")
        mock_response.candidates = response.get("candidates", [])
        mock_response.error = response.get("error", None)

        return mock_response

    return _create_response


@pytest.fixture
def mock_vertex_ai_client():
    """Create mock Vertex AI client for testing."""
    mock = MagicMock()

    # Configure generative model
    mock.GenerativeModel = MagicMock()
    model_instance = MagicMock()
    model_instance.generate_content = AsyncMock(return_value=MagicMock(text="Vertex AI response"))
    mock.GenerativeModel.return_value = model_instance

    return mock


@pytest.fixture
def gemini_test_prompts():
    """Collection of test prompts for different agent types."""
    return {
        "context_analysis": "Analyze the project structure and provide insights",
        "code_generation": "Generate a Python function to calculate factorial",
        "memory_retrieval": "Retrieve information about previous conversations",
        "architecture_review": "Review the system architecture and suggest improvements",
        "test_generation": "Generate unit tests for the provided code",
        "refactoring": "Refactor the code to improve readability and performance",
        "documentation": "Generate comprehensive documentation for the module",
        "security_review": "Perform security analysis on the codebase",
    }


@pytest.fixture
def gemini_mock_tools():
    """Mock tools that Gemini agents might use."""

    class MockFilesystemTool:
        async def read_file(self, path: str) -> str:
            return f"Mock content of {path}"

        async def write_file(self, path: str, content: str) -> bool:
            return True

        async def list_directory(self, path: str) -> list[str]:
            return ["file1.py", "file2.py", "README.md"]

    class MockSearchTool:
        async def search_code(self, query: str) -> list[dict[str, Any]]:
            return [
                {"file": "main.py", "line": 10, "match": query},
                {"file": "utils.py", "line": 25, "match": query},
            ]

    class MockGitTool:
        async def get_diff(self) -> str:
            return "Mock git diff output"

        async def get_status(self) -> str:
            return "Mock git status output"

    return {"filesystem": MockFilesystemTool(), "search": MockSearchTool(), "git": MockGitTool()}


@pytest.fixture
def gemini_config():
    """Test configuration for Gemini agents."""
    return {
        "api_key": "test-api-key",
        "project_id": "test-project",
        "location": "us-central1",
        "models": {
            "default": "gemini-2.5-flash",
            "advanced": "gemini-2.5-pro",
            "vision": "gemini-2.0-flash-vision",
        },
        "safety_settings": {
            "harassment": "BLOCK_NONE",
            "hate_speech": "BLOCK_NONE",
            "sexually_explicit": "BLOCK_NONE",
            "dangerous_content": "BLOCK_NONE",
        },
        "generation_config": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 8192,
        },
        "retry_config": {"max_retries": 3, "initial_delay": 1, "max_delay": 60, "multiplier": 2},
    }


@pytest.fixture
def mock_gemini_auth():
    """Mock authentication for Gemini."""
    mock_auth = MagicMock()

    # Mock service account
    mock_auth.service_account = MagicMock()
    mock_auth.service_account.from_service_account_file = MagicMock(
        return_value=MagicMock(project_id="test-project")
    )

    # Mock API key authentication
    mock_auth.api_key = "test-api-key-12345"

    # Mock OAuth
    mock_auth.oauth = MagicMock()
    mock_auth.oauth.get_credentials = AsyncMock(return_value={"access_token": "mock-oauth-token"})

    return mock_auth


@pytest.fixture
def gemini_test_contexts():
    """Test contexts for different scenarios."""
    return {
        "simple_project": {
            "path": "/test/simple_project",
            "files": ["main.py", "utils.py", "README.md"],
            "language": "python",
            "framework": None,
            "dependencies": ["requests", "pytest"],
        },
        "complex_project": {
            "path": "/test/complex_project",
            "files": [
                "src/main.py",
                "src/api/routes.py",
                "src/models/user.py",
                "tests/test_api.py",
                "docker-compose.yml",
                "Makefile",
            ],
            "language": "python",
            "framework": "fastapi",
            "dependencies": ["fastapi", "sqlalchemy", "redis", "celery", "pytest"],
            "services": ["postgres", "redis", "rabbitmq"],
        },
        "frontend_project": {
            "path": "/test/frontend",
            "files": [
                "src/App.tsx",
                "src/components/Dashboard.tsx",
                "package.json",
                "tsconfig.json",
            ],
            "language": "typescript",
            "framework": "react",
            "dependencies": ["react", "react-dom", "typescript", "vite"],
        },
    }


@pytest.fixture
async def gemini_mock_session():
    """Mock session for testing stateful interactions."""

    class MockSession:
        def __init__(self) -> None:
            self.history = []
            self.context = {}
            self.model = "gemini-2.5-flash"

        async def send_message(self, message: str) -> str:
            self.history.append({"role": "user", "content": message})
            response = f"Response to: {message}"
            self.history.append({"role": "assistant", "content": response})
            return response

        def get_history(self) -> list[dict[str, str]]:
            return self.history

        def clear_history(self) -> None:
            self.history = []

        def update_context(self, **kwargs) -> None:
            self.context.update(kwargs)

        def get_context(self) -> dict[str, Any]:
            return self.context

    return MockSession()


@pytest.fixture
def gemini_error_scenarios():
    """Collection of error scenarios for testing error handling."""

    class GeminiError(Exception):
        def __init__(self, code: int, message: str, status: str) -> None:
            self.code = code
            self.message = message
            self.status = status
            super().__init__(message)

    return {
        "rate_limit": GeminiError(429, "Rate limit exceeded", "RESOURCE_EXHAUSTED"),
        "invalid_api_key": GeminiError(401, "Invalid API key", "UNAUTHENTICATED"),
        "quota_exceeded": GeminiError(429, "Quota exceeded", "RESOURCE_EXHAUSTED"),
        "model_not_found": GeminiError(404, "Model not found", "NOT_FOUND"),
        "invalid_request": GeminiError(400, "Invalid request", "INVALID_ARGUMENT"),
        "server_error": GeminiError(500, "Internal server error", "INTERNAL"),
        "timeout": GeminiError(504, "Request timeout", "DEADLINE_EXCEEDED"),
        "content_filtered": GeminiError(400, "Content filtered", "FAILED_PRECONDITION"),
    }
