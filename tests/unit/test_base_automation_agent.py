"""
Comprehensive unit tests for BaseAutomationAgent class.

Tests initialization, model management, file operations, command execution,
Gemini integration, and all abstract base functionality with extensive mocking.
"""

import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

from app.automation.base_automation_agent import AsyncMixin
from app.automation.base_automation_agent import BaseAutomationAgent
from app.automation.base_automation_agent import FileProcessingMixin
from app.automation.base_automation_agent import GitMixin
import pytest


class MockAutomationAgent(BaseAutomationAgent):
    """Mock implementation of BaseAutomationAgent for testing."""

    def __init__(self, agent_name: str = "test-agent", description: str = "Test automation agent"):
        super().__init__(agent_name, description)
        self.register_tools_called = False
        self.run_called = False

    def register_tools(self) -> None:
        """Mock tool registration."""
        self.register_tools_called = True


class TestBaseAutomationAgent:
    """Test BaseAutomationAgent functionality."""

    @pytest.fixture
    def mock_fastmcp(self):
        """Mock FastMCP for testing."""
        with patch("app.automation.base_automation_agent.FastMCP") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_get_model_for_task(self):
        """Mock get_model_for_task function."""
        with patch("app.automation.base_automation_agent.get_model_for_task") as mock:
            mock_model = MagicMock()
            mock.return_value = mock_model
            yield mock

    @pytest.fixture
    def mock_agent(self, mock_fastmcp, mock_get_model_for_task):
        """Create a mock automation agent for testing."""
        return MockAutomationAgent()

    def test_initialization_valid(self, mock_agent):
        """Test valid agent initialization."""
        assert mock_agent.agent_name == "test-agent"
        assert mock_agent.description == "Test automation agent"
        assert hasattr(mock_agent, "logger")
        assert hasattr(mock_agent, "mcp")
        assert hasattr(mock_agent, "_model_cache")
        assert isinstance(mock_agent._model_cache, dict)

    def test_initialization_empty_description(self, mock_fastmcp, mock_get_model_for_task):
        """Test initialization with empty description."""
        agent = MockAutomationAgent("test-agent", "")
        assert agent.description == ""

    def test_setup_logging(self, mock_agent):
        """Test logging setup."""
        logger = mock_agent._setup_logging()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "MockAutomationAgent"

    def test_get_model_caching(self, mock_agent, mock_get_model_for_task):
        """Test model caching functionality."""
        task_type = "analysis"

        # First call should create and cache model
        model1 = mock_agent.get_model(task_type)
        assert task_type in mock_agent._model_cache
        mock_get_model_for_task.assert_called_once_with(task_type)

        # Second call should use cached model
        mock_get_model_for_task.reset_mock()
        model2 = mock_agent.get_model(task_type)
        assert model1 is model2
        mock_get_model_for_task.assert_not_called()

    def test_get_model_different_tasks(self, mock_agent, mock_get_model_for_task):
        """Test getting models for different task types."""
        mock_get_model_for_task.side_effect = lambda task: f"model_{task}"

        model1 = mock_agent.get_model("analysis")
        model2 = mock_agent.get_model("generation")

        assert model1 != model2
        assert len(mock_agent._model_cache) == 2
        assert mock_get_model_for_task.call_count == 2

    def test_create_success_response_basic(self, mock_agent):
        """Test creating basic success response."""
        data = {"result": "test", "count": 5}
        message = "Operation successful"

        response = mock_agent.create_success_response(data, message)

        assert response["status"] == "success"
        assert response["message"] == message
        assert response["agent"] == "test-agent"
        assert response["result"] == "test"
        assert response["count"] == 5
        assert "timestamp" in response
        # Check timestamp format
        assert "T" in response["timestamp"]

    def test_create_success_response_default_message(self, mock_agent):
        """Test creating success response with default message."""
        data = {"test": "data"}

        response = mock_agent.create_success_response(data)

        assert response["message"] == "Operation completed successfully"

    def test_create_error_response_string(self, mock_agent, caplog):
        """Test creating error response from string."""
        error_msg = "Test error occurred"
        context = {"operation": "test_op", "attempt": 1}

        with caplog.at_level(logging.ERROR):
            response = mock_agent.create_error_response(error_msg, context)

        assert response["status"] == "error"
        assert response["error"] == error_msg
        assert response["agent"] == "test-agent"
        assert response["context"] == context
        assert "timestamp" in response

        # Check logging
        assert "Error in test-agent: Test error occurred" in caplog.text

    def test_create_error_response_exception(self, mock_agent, caplog):
        """Test creating error response from exception."""
        exception = ValueError("Test exception")

        with caplog.at_level(logging.ERROR):
            response = mock_agent.create_error_response(exception)

        assert response["status"] == "error"
        assert response["error"] == "Test exception"
        assert "context" not in response

    def test_safe_file_read_success(self, mock_agent):
        """Test successful file reading."""
        test_content = "Test file content\nLine 2"

        with patch("builtins.open", mock_open(read_data=test_content)):
            content = mock_agent.safe_file_read("test.txt")

        assert content == test_content

    def test_safe_file_read_path_object(self, mock_agent):
        """Test file reading with Path object."""
        test_content = "Path object content"

        with patch("builtins.open", mock_open(read_data=test_content)):
            content = mock_agent.safe_file_read(Path("test.txt"))

        assert content == test_content

    def test_safe_file_read_failure(self, mock_agent, caplog):
        """Test file reading failure."""
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with caplog.at_level(logging.ERROR):
                content = mock_agent.safe_file_read("nonexistent.txt")

        assert content is None
        assert "Failed to read file" in caplog.text

    def test_safe_file_read_encoding_error(self, mock_agent, caplog):
        """Test file reading with encoding error."""
        with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")):
            with caplog.at_level(logging.ERROR):
                content = mock_agent.safe_file_read("binary.txt")

        assert content is None

    def test_safe_file_write_success(self, mock_agent):
        """Test successful file writing."""
        content = "Test content to write"

        with patch("builtins.open", mock_open()) as mock_file, patch("pathlib.Path.mkdir"):
            result = mock_agent.safe_file_write("test.txt", content)

        assert result is True
        mock_file.assert_called_once_with(Path("test.txt"), "w", encoding="utf-8")
        mock_file().write.assert_called_once_with(content)

    def test_safe_file_write_create_dirs(self, mock_agent):
        """Test file writing with directory creation."""
        content = "Test content"

        with patch("builtins.open", mock_open()), patch("pathlib.Path.mkdir") as mock_mkdir:
            result = mock_agent.safe_file_write(
                "dir/subdir/test.txt", content, create_dirs=True
            )

        assert result is True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_safe_file_write_no_create_dirs(self, mock_agent):
        """Test file writing without directory creation."""
        content = "Test content"

        with patch("builtins.open", mock_open()), patch("pathlib.Path.mkdir") as mock_mkdir:
            result = mock_agent.safe_file_write("test.txt", content, create_dirs=False)

        assert result is True
        mock_mkdir.assert_not_called()

    def test_safe_file_write_failure(self, mock_agent, caplog):
        """Test file writing failure."""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with caplog.at_level(logging.ERROR):
                result = mock_agent.safe_file_write("test.txt", "content")

        assert result is False
        assert "Failed to write file" in caplog.text

    def test_find_files_basic(self, mock_agent):
        """Test basic file finding."""
        mock_files = [Path("file1.py"), Path("file2.py"), Path("file3.txt")]

        with patch("pathlib.Path.rglob", return_value=mock_files) as mock_rglob:
            files = mock_agent.find_files("/test/dir", "*.py")

        mock_rglob.assert_called_once_with("*.py")
        assert len(files) == 3
        assert all(isinstance(f, Path) for f in files)

    def test_find_files_with_exclusions(self, mock_agent):
        """Test file finding with exclusions."""
        mock_files = [
            Path("src/main.py"),
            Path("src/__pycache__/cache.py"),
            Path("tests/test_main.py"),
            Path(".git/config"),
        ]

        with patch("pathlib.Path.rglob", return_value=mock_files):
            files = mock_agent.find_files(
                "/test/dir", "*.py", exclude_patterns=["__pycache__", ".git"]
            )

        # Should exclude files containing __pycache__ or .git
        assert len(files) == 2
        file_paths = [str(f) for f in files]
        assert any("main.py" in p for p in file_paths)
        assert any("test_main.py" in p for p in file_paths)
        assert not any("__pycache__" in p for p in file_paths)
        assert not any(".git" in p for p in file_paths)

    def test_find_files_failure(self, mock_agent, caplog):
        """Test file finding failure."""
        with patch("pathlib.Path.rglob", side_effect=OSError("Access denied")):
            with caplog.at_level(logging.ERROR):
                files = mock_agent.find_files("/test/dir", "*.py")

        assert files == []
        assert "Failed to find files" in caplog.text

    @pytest.mark.asyncio
    async def test_generate_with_gemini_success(self, mock_agent):
        """Test successful Gemini generation."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated response"
        mock_model.generate_content.return_value = mock_response
        mock_agent.get_model = MagicMock(return_value=mock_model)

        result = await mock_agent.generate_with_gemini("Test prompt", "analysis")

        assert result == "Generated response"
        mock_agent.get_model.assert_called_once_with("analysis")
        mock_model.generate_content.assert_called_once_with("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_with_gemini_parse_json(self, mock_agent):
        """Test Gemini generation with JSON parsing."""
        json_response = '{"result": "success", "data": [1, 2, 3]}'

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json_response
        mock_model.generate_content.return_value = mock_response
        mock_agent.get_model = MagicMock(return_value=mock_model)

        with patch("app.automation.base_automation_agent.safe_json_parse") as mock_parse:
            mock_parse.return_value = {"result": "success", "data": [1, 2, 3]}
            result = await mock_agent.generate_with_gemini("Test prompt", parse_json=True)

        assert isinstance(result, dict)
        mock_parse.assert_called_once_with(json_response)

    @pytest.mark.asyncio
    async def test_generate_with_gemini_failure(self, mock_agent, caplog):
        """Test Gemini generation failure."""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_agent.get_model = MagicMock(return_value=mock_model)

        with caplog.at_level(logging.ERROR):
            result = await mock_agent.generate_with_gemini("Test prompt")

        assert result is None
        assert "Gemini generation failed" in caplog.text

    def test_run_command_success(self, mock_agent):
        """Test successful command execution."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Command output"

        with patch(
            "app.automation.base_automation_agent.safe_subprocess_run", return_value=mock_result
        ) as mock_run:
            result = mock_agent.run_command(["echo", "test"], timeout=30)

        assert result == mock_result
        mock_run.assert_called_once_with(["echo", "test"], timeout=30)

    def test_run_command_failure(self, mock_agent, caplog):
        """Test command execution failure."""
        with patch(
            "app.automation.base_automation_agent.safe_subprocess_run",
            side_effect=Exception("Command failed"),
        ), caplog.at_level(logging.ERROR):
            result = mock_agent.run_command(["invalid", "command"])

        assert result is None
        assert "Command failed" in caplog.text

    def test_extract_code_blocks_with_language(self, mock_agent):
        """Test extracting code blocks with specific language."""
        text = """
        Here's some Python code:
        ```python
        def hello():
            print("Hello, World!")
        ```

        And some JavaScript:
        ```javascript
        console.log("Hello, World!");
        ```
        """

        python_blocks = mock_agent.extract_code_blocks(text, "python")

        assert len(python_blocks) == 1
        assert "def hello():" in python_blocks[0]
        assert "print(" in python_blocks[0]

    def test_extract_code_blocks_any_language(self, mock_agent):
        """Test extracting all code blocks regardless of language."""
        text = """
        ```python
        def hello():
            return "Hello"
        ```

        ```bash
        echo "test"
        ```

        ```
        plain code block
        ```
        """

        blocks = mock_agent.extract_code_blocks(text)

        assert len(blocks) == 3
        assert any("def hello" in block for block in blocks)
        assert any("echo" in block for block in blocks)
        assert any("plain code" in block for block in blocks)

    def test_extract_code_blocks_no_matches(self, mock_agent):
        """Test extracting code blocks when none exist."""
        text = "This is just plain text with no code blocks."

        blocks = mock_agent.extract_code_blocks(text, "python")

        assert blocks == []

    def test_create_backup_success(self, mock_agent):
        """Test successful backup creation."""
        original_content = "Original file content"

        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(mock_agent, "safe_file_read", return_value=original_content):
                with patch.object(mock_agent, "safe_file_write", return_value=True):
                    backup_path = mock_agent.create_backup("test.py")

        assert backup_path is not None
        assert str(backup_path).endswith(".py.backup")

    def test_create_backup_file_not_exists(self, mock_agent):
        """Test backup creation when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            backup_path = mock_agent.create_backup("nonexistent.py")

        assert backup_path is None

    def test_create_backup_read_failure(self, mock_agent):
        """Test backup creation with read failure."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(mock_agent, "safe_file_read", return_value=None):
                backup_path = mock_agent.create_backup("test.py")

        assert backup_path is None

    def test_create_backup_write_failure(self, mock_agent):
        """Test backup creation with write failure."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch.object(mock_agent, "safe_file_read", return_value="content"):
                with patch.object(mock_agent, "safe_file_write", return_value=False):
                    backup_path = mock_agent.create_backup("test.py")

        assert backup_path is None

    def test_create_backup_exception(self, mock_agent, caplog):
        """Test backup creation with exception."""
        with patch("pathlib.Path.exists", side_effect=Exception("Access error")):
            with caplog.at_level(logging.ERROR):
                backup_path = mock_agent.create_backup("test.py")

        assert backup_path is None
        assert "Failed to create backup" in caplog.text

    def test_run_mcp_server(self, mock_agent):
        """Test running the MCP server."""
        mock_agent.run()

        assert mock_agent.register_tools_called is True
        mock_agent.mcp.run.assert_called_once()

    def test_abstract_register_tools(self):
        """Test that register_tools is abstract."""
        # Should not be able to instantiate without implementing register_tools
        with pytest.raises(TypeError):
            BaseAutomationAgent("test", "test")


class TestFileProcessingMixin:
    """Test FileProcessingMixin functionality."""

    class MockAgentWithFileProcessing(BaseAutomationAgent, FileProcessingMixin):
        """Mock agent with file processing mixin."""

        def __init__(self):
            self.logger = MagicMock()

        def register_tools(self):
            pass

    @pytest.fixture
    def mock_agent_with_files(self):
        """Create mock agent with file processing."""
        return self.MockAgentWithFileProcessing()

    def test_process_files_batch_success(self, mock_agent_with_files):
        """Test successful batch file processing."""
        files = [Path(f"file{i}.py") for i in range(5)]

        def mock_processor(file_path):
            return {"file": str(file_path), "status": "processed"}

        results = mock_agent_with_files.process_files_batch(files, mock_processor, batch_size=2)

        assert len(results) == 5
        assert all(result["status"] == "processed" for result in results)

    def test_process_files_batch_with_errors(self, mock_agent_with_files):
        """Test batch processing with some errors."""
        files = [Path(f"file{i}.py") for i in range(3)]

        def mock_processor(file_path):
            if "file1" in str(file_path):
                raise ValueError("Processing error")
            return {"file": str(file_path), "status": "processed"}

        results = mock_agent_with_files.process_files_batch(files, mock_processor)

        assert len(results) == 3
        assert results[1]["status"] == "error"
        assert "Processing error" in results[1]["error"]

    def test_process_files_batch_logging(self, mock_agent_with_files):
        """Test batch processing logs progress."""
        files = [Path(f"file{i}.py") for i in range(5)]

        def mock_processor(file_path):
            return {"status": "processed"}

        mock_agent_with_files.process_files_batch(files, mock_processor, batch_size=2)

        # Should have logged batch progress
        assert mock_agent_with_files.logger.info.call_count >= 1


class TestAsyncMixin:
    """Test AsyncMixin functionality."""

    class MockAgentWithAsync(BaseAutomationAgent, AsyncMixin):
        """Mock agent with async mixin."""

        def register_tools(self):
            pass

    @pytest.fixture
    def mock_agent_with_async(self):
        """Create mock agent with async functionality."""
        with patch("app.automation.base_automation_agent.FastMCP"):
            with patch("app.automation.base_automation_agent.get_model_for_task"):
                return self.MockAgentWithAsync("test", "test")

    @pytest.mark.asyncio
    async def test_run_async_tasks_success(self, mock_agent_with_async):
        """Test running async tasks with concurrency limit."""

        async def mock_task(value):
            await asyncio.sleep(0.01)  # Simulate async work
            return value * 2

        tasks = [mock_task(i) for i in range(5)]
        results = await mock_agent_with_async.run_async_tasks(tasks, max_concurrent=2)

        assert len(results) == 5
        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_run_async_tasks_with_errors(self, mock_agent_with_async):
        """Test async tasks with some failures."""

        async def mock_task(value):
            if value == 2:
                raise ValueError("Task error")
            return value * 2

        tasks = [mock_task(i) for i in range(4)]

        with pytest.raises(ValueError):
            await mock_agent_with_async.run_async_tasks(tasks)


class TestGitMixin:
    """Test GitMixin functionality."""

    class MockAgentWithGit(BaseAutomationAgent, GitMixin):
        """Mock agent with Git mixin."""

        def __init__(self):
            self.logger = MagicMock()

        def register_tools(self):
            pass

        def run_command(self, command, timeout=60, capture_output=True):
            """Mock run_command method."""
            if command[1:3] == ["status", "--porcelain"]:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "M file1.py\nA file2.py\n"
                return mock_result
            elif command[1:3] == ["diff", "HEAD"]:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "diff --git a/file.py b/file.py\n+added line\n"
                return mock_result
            elif command[1:3] == ["add", "-A"] or command[1:3] == ["commit", "-m"]:
                mock_result = MagicMock()
                mock_result.returncode = 0
                return mock_result
            return None

    @pytest.fixture
    def mock_agent_with_git(self):
        """Create mock agent with Git functionality."""
        return self.MockAgentWithGit()

    def test_get_git_status_success(self, mock_agent_with_git):
        """Test getting Git status."""
        status = mock_agent_with_git.get_git_status()

        assert status == "M file1.py\nA file2.py\n"

    def test_get_git_status_failure(self, mock_agent_with_git):
        """Test Git status when command fails."""

        # Override run_command to return failure
        def failing_run_command(command, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 1
            return mock_result

        mock_agent_with_git.run_command = failing_run_command
        status = mock_agent_with_git.get_git_status()

        assert status is None

    def test_get_git_diff_all_changes(self, mock_agent_with_git):
        """Test getting Git diff for all changes."""
        diff = mock_agent_with_git.get_git_diff()

        assert "diff --git" in diff
        assert "+added line" in diff

    def test_get_git_diff_specific_file(self, mock_agent_with_git):
        """Test getting Git diff for specific file."""
        diff = mock_agent_with_git.get_git_diff("specific_file.py")

        assert diff is not None
        # The mock should have been called with the file path

    def test_commit_changes_success(self, mock_agent_with_git):
        """Test successful commit with add all."""
        result = mock_agent_with_git.commit_changes("Test commit message", add_all=True)

        assert result is True

    def test_commit_changes_no_add(self, mock_agent_with_git):
        """Test commit without adding all files."""
        result = mock_agent_with_git.commit_changes("Test commit", add_all=False)

        assert result is True

    def test_commit_changes_failure(self, mock_agent_with_git, caplog):
        """Test commit failure."""

        def failing_run_command(command, **kwargs):
            if "commit" in command:
                mock_result = MagicMock()
                mock_result.returncode = 1
                return mock_result
            # Return success for add command
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result

        mock_agent_with_git.run_command = failing_run_command

        with caplog.at_level(logging.ERROR):
            result = mock_agent_with_git.commit_changes("Test commit")

        assert result is False

    def test_commit_changes_exception(self, mock_agent_with_git, caplog):
        """Test commit with exception."""

        def exception_run_command(command, **kwargs):
            raise Exception("Git command failed")

        mock_agent_with_git.run_command = exception_run_command

        result = mock_agent_with_git.commit_changes("Test commit")

        assert result is False
        # Check that logger.error was called with the expected message
        mock_agent_with_git.logger.error.assert_called_once()
        error_call_args = mock_agent_with_git.logger.error.call_args[0][0]
        assert "Git commit failed" in error_call_args
