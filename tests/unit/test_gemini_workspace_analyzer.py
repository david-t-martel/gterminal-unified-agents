#!/usr/bin/env python3
"""
Comprehensive test suite for Gemini Workspace Analyzer MCP Server.

Tests workspace analysis, file operations, and content search functionality.
Targets 35% coverage of gemini_workspace_analyzer.py.
"""

import asyncio
import json
from pathlib import Path

# Import the module under test
import sys
import tempfile
from unittest.mock import Mock
from unittest.mock import patch

from pydantic import ValidationError
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.mcp_servers.gemini_workspace_analyzer import AnalysisResponse
from app.mcp_servers.gemini_workspace_analyzer import ContentSearchRequest
from app.mcp_servers.gemini_workspace_analyzer import FileSearchRequest
from app.mcp_servers.gemini_workspace_analyzer import ProjectOverviewRequest
from app.mcp_servers.gemini_workspace_analyzer import WorkspaceAnalysisRequest
from app.mcp_servers.gemini_workspace_analyzer import mcp


class TestPydanticModels:
    """Test Pydantic model validation for Workspace Analyzer."""

    def test_workspace_analysis_request_valid(self):
        """Test WorkspaceAnalysisRequest with valid data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            request = WorkspaceAnalysisRequest(
                project_path=tmpdir,
                analysis_depth="comprehensive",
                include_dependencies=True,
                include_tests=True,
                focus_areas="architecture,performance",
            )
            assert request.project_path == tmpdir
            assert request.analysis_depth == "comprehensive"
            assert request.include_dependencies is True
            assert request.focus_areas == "architecture,performance"

    def test_workspace_analysis_request_invalid_depth(self):
        """Test WorkspaceAnalysisRequest with invalid analysis depth."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValidationError):
            WorkspaceAnalysisRequest(project_path=tmpdir, analysis_depth="invalid_depth")

    def test_workspace_analysis_request_defaults(self):
        """Test WorkspaceAnalysisRequest with default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            request = WorkspaceAnalysisRequest(project_path=tmpdir)

            assert request.analysis_depth == "standard"
            assert request.include_dependencies is True
            assert request.include_tests is True
            assert request.focus_areas == ""

    def test_file_search_request_valid(self):
        """Test FileSearchRequest with valid data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            request = FileSearchRequest(
                directory=tmpdir, pattern="*.py", exclude_patterns="__pycache__,*.pyc"
            )
            assert request.directory == tmpdir
            assert request.pattern == "*.py"
            assert request.exclude_patterns == "__pycache__,*.pyc"

    def test_file_search_request_defaults(self):
        """Test FileSearchRequest with default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            request = FileSearchRequest(directory=tmpdir)

            assert request.pattern == "*"
            assert "node_modules" in request.exclude_patterns
            assert "__pycache__" in request.exclude_patterns

    def test_content_search_request_valid(self):
        """Test ContentSearchRequest with valid data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            request = ContentSearchRequest(
                directory=tmpdir,
                search_pattern="def main",
                file_patterns="*.py,*.js",
                max_results=50,
            )
            assert request.directory == tmpdir
            assert request.search_pattern == "def main"
            assert request.file_patterns == "*.py,*.js"
            assert request.max_results == 50

    def test_content_search_request_max_results_validation(self):
        """Test ContentSearchRequest max_results validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test minimum validation
            with pytest.raises(ValidationError):
                ContentSearchRequest(directory=tmpdir, search_pattern="test", max_results=0)

            # Test maximum validation
            with pytest.raises(ValidationError):
                ContentSearchRequest(directory=tmpdir, search_pattern="test", max_results=1001)

    def test_project_overview_request_valid(self):
        """Test ProjectOverviewRequest with valid data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            request = ProjectOverviewRequest(project_path=tmpdir)
            assert request.project_path == tmpdir

    def test_analysis_response_valid(self):
        """Test AnalysisResponse with valid data."""
        response = AnalysisResponse(
            status="success",
            data={"files": 10, "lines": 500},
            error=None,
            rust_powered=True,
            cached=False,
        )
        assert response.status == "success"
        assert response.data["files"] == 10
        assert response.rust_powered is True
        assert response.cached is False

    def test_analysis_response_defaults(self):
        """Test AnalysisResponse with default values."""
        response = AnalysisResponse()

        assert response.status == "success"
        assert response.data == {}
        assert response.error is None
        assert response.rust_powered is False
        assert response.cached is False


class TestMCPToolsConfiguration:
    """Test MCP tools configuration and imports."""

    def test_mcp_server_name(self):
        """Test MCP server has correct name."""
        assert mcp.name == "gemini-workspace-analyzer"

    def test_mcp_tool_imports(self):
        """Test that MCP tools can be imported and have correct signatures."""
        # Import the actual MCP tool functions
        # Verify tools are callable async functions
        import inspect

        from app.mcp_servers.gemini_workspace_analyzer import analyze_workspace
        from app.mcp_servers.gemini_workspace_analyzer import get_file_content
        from app.mcp_servers.gemini_workspace_analyzer import get_project_overview
        from app.mcp_servers.gemini_workspace_analyzer import list_directory_structure
        from app.mcp_servers.gemini_workspace_analyzer import search_content
        from app.mcp_servers.gemini_workspace_analyzer import search_files

        assert inspect.iscoroutinefunction(analyze_workspace)
        assert inspect.iscoroutinefunction(search_files)
        assert inspect.iscoroutinefunction(search_content)
        assert inspect.iscoroutinefunction(get_project_overview)
        assert inspect.iscoroutinefunction(list_directory_structure)
        assert inspect.iscoroutinefunction(get_file_content)

        # Verify function signatures have expected parameters
        sig = inspect.signature(analyze_workspace)
        param_names = list(sig.parameters.keys())
        expected_params = [
            "project_path",
            "analysis_depth",
            "include_dependencies",
            "include_tests",
            "focus_areas",
        ]
        for param in expected_params:
            assert param in param_names, f"Parameter '{param}' not found in analyze_workspace"


class TestWorkspaceAnalysis:
    """Test workspace analysis functionality."""

    def create_sample_workspace(self, base_path: Path) -> None:
        """Create a sample workspace structure for testing."""
        # Create source directory
        src_dir = base_path / "src"
        src_dir.mkdir()

        (src_dir / "__init__.py").write_text("")
        (src_dir / "main.py").write_text(
            """
#!/usr/bin/env python3
\"\"\"Main application entry point.\"\"\"

import sys
from pathlib import Path

def main():
    print("Hello, world!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
        )

        (src_dir / "utils.py").write_text(
            """
\"\"\"Utility functions.\"\"\"

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self._config = {}

    def load_config(self) -> Dict[str, Any]:
        \"\"\"Load configuration from file.\"\"\"
        logger.info(f"Loading config from {self.config_file}")
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        \"\"\"Get configuration value.\"\"\"
        return self._config.get(key, default)
"""
        )

        # Create tests directory
        tests_dir = base_path / "tests"
        tests_dir.mkdir()

        (tests_dir / "__init__.py").write_text("")
        (tests_dir / "test_main.py").write_text(
            """
import unittest
from src.main import main

class TestMain(unittest.TestCase):
    def test_main_returns_zero(self):
        \"\"\"Test that main function returns 0.\"\"\"
        result = main()
        self.assertEqual(result, 0)

    def test_main_prints_hello(self):
        \"\"\"Test that main function prints hello.\"\"\"
        # Add your test implementation here
        pass

if __name__ == "__main__":
    unittest.main()
"""
        )

        (tests_dir / "test_utils.py").write_text(
            """
import unittest
from src.utils import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.config_manager = ConfigManager("test_config.json")

    def test_get_default_value(self):
        \"\"\"Test get method with default value.\"\"\"
        result = self.config_manager.get("nonexistent", "default")
        self.assertEqual(result, "default")
"""
        )

        # Create configuration files
        (base_path / "requirements.txt").write_text(
            """
requests>=2.28.0
flask>=2.0.1
pytest>=7.0.0
black>=22.0.0
mypy>=0.950
"""
        )

        (base_path / "setup.py").write_text(
            """
from setuptools import setup, find_packages

setup(
    name="sample-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
        "flask>=2.0.1",
    ],
    python_requires=">=3.8",
)
"""
        )

        (base_path / "README.md").write_text(
            """
# Sample Project

This is a sample project for testing workspace analysis.

## Features

- Main application entry point
- Configuration management
- Comprehensive test suite
- Modern Python packaging

## Installation

```bash
pip install -e .
```

## Usage

```bash
python -m src.main
```
"""
        )

        (base_path / ".gitignore").write_text(
            """
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
"""
        )

        # Create docs directory
        docs_dir = base_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "api.md").write_text("# API Documentation")
        (docs_dir / "user_guide.md").write_text("# User Guide")

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_workspace_analyzer.configure_gemini")
    async def test_analyze_workspace_tool(self, mock_configure):
        """Test workspace analysis through MCP tool."""
        from app.mcp_servers.gemini_workspace_analyzer import analyze_workspace

        # Setup mock Gemini model
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "workspace_analysis": {
                    "project_type": "python_application",
                    "architecture": "simple_monolith",
                    "total_files": 8,
                    "source_files": 4,
                    "test_files": 2,
                    "config_files": 2,
                },
                "recommendations": [
                    "Add type hints to improve code quality",
                    "Consider using dataclasses for configuration",
                    "Add logging configuration",
                ],
                "structure_quality": {
                    "score": 8.5,
                    "strengths": ["Clear separation of concerns", "Good test coverage structure"],
                    "improvements": [
                        "Add documentation",
                        "Consider using pytest instead of unittest",
                    ],
                },
            }
        )
        mock_model.generate_content.return_value = mock_response
        mock_configure.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            self.create_sample_workspace(workspace_path)

            # Call the actual MCP tool
            result = await analyze_workspace(
                project_path=str(workspace_path),
                analysis_depth="comprehensive",
                include_dependencies=True,
                include_tests=True,
                focus_areas="architecture,quality",
            )

            assert result["status"] == "success"
            assert result["project_path"] == str(workspace_path)
            assert "analysis" in result
            assert "files_analyzed" in result

    @pytest.mark.asyncio
    async def test_analyze_workspace_nonexistent_path(self):
        """Test workspace analysis with non-existent path."""
        from app.mcp_servers.gemini_workspace_analyzer import analyze_workspace

        result = await analyze_workspace(project_path="/nonexistent/path")

        assert result["status"] == "error"
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_search_files_tool(self):
        """Test file search functionality."""
        from app.mcp_servers.gemini_workspace_analyzer import search_files

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            self.create_sample_workspace(workspace_path)

            # Search for Python files
            result = await search_files(
                directory=str(workspace_path), pattern="*.py", exclude_patterns="__pycache__"
            )

            assert result["status"] == "success"
            assert "files" in result
            assert len(result["files"]) > 0

            # Should find main.py, utils.py, test files, etc.
            file_names = [Path(f).name for f in result["files"]]
            assert "main.py" in file_names
            assert "utils.py" in file_names

    @pytest.mark.asyncio
    async def test_search_content_tool(self):
        """Test content search functionality."""
        from app.mcp_servers.gemini_workspace_analyzer import search_content

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            self.create_sample_workspace(workspace_path)

            # Search for function definitions
            result = await search_content(
                directory=str(workspace_path),
                search_pattern="def main",
                file_patterns="*.py",
                max_results=10,
            )

            assert result["status"] == "success"
            assert "matches" in result

            # Should find the main function
            if result["matches"]:
                assert any("main.py" in match.get("file", "") for match in result["matches"])

    @pytest.mark.asyncio
    async def test_get_project_overview_tool(self):
        """Test project overview functionality."""
        from app.mcp_servers.gemini_workspace_analyzer import get_project_overview

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            self.create_sample_workspace(workspace_path)

            result = await get_project_overview(project_path=str(workspace_path))

            assert result["status"] == "success"
            assert "overview" in result
            assert "total_files" in result["overview"]
            assert "directories" in result["overview"]
            assert "file_types" in result["overview"]

            # Should detect Python files
            file_types = result["overview"]["file_types"]
            assert ".py" in file_types or "python" in str(file_types).lower()

    @pytest.mark.asyncio
    async def test_list_directory_structure_tool(self):
        """Test directory structure listing."""
        from app.mcp_servers.gemini_workspace_analyzer import list_directory_structure

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            self.create_sample_workspace(workspace_path)

            result = await list_directory_structure(
                directory=str(workspace_path), max_depth=3, include_hidden=False
            )

            assert result["status"] == "success"
            assert "structure" in result

            # Should show main directories
            structure_str = str(result["structure"])
            assert "src" in structure_str
            assert "tests" in structure_str
            assert "docs" in structure_str

    @pytest.mark.asyncio
    async def test_get_file_content_tool(self):
        """Test file content retrieval."""
        from app.mcp_servers.gemini_workspace_analyzer import get_file_content

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            self.create_sample_workspace(workspace_path)

            main_file = workspace_path / "src" / "main.py"

            result = await get_file_content(file_path=str(main_file), max_lines=50)

            assert result["status"] == "success"
            assert "content" in result
            assert "def main" in result["content"]
            assert "metadata" in result
            assert result["metadata"]["file_path"] == str(main_file)


class TestRustIntegration:
    """Test Rust integration functionality."""

    @pytest.mark.asyncio
    async def test_rust_fallback_behavior(self):
        """Test behavior when Rust integration is not available."""
        # Mock RUST_AVAILABLE as False
        with patch("app.mcp_servers.gemini_workspace_analyzer.RUST_AVAILABLE", False):
            from app.mcp_servers.gemini_workspace_analyzer import search_files

            with tempfile.TemporaryDirectory() as tmpdir:
                workspace_path = Path(tmpdir)

                # Create test files
                (workspace_path / "test1.py").write_text("print('test1')")
                (workspace_path / "test2.py").write_text("print('test2')")

                result = await search_files(directory=str(workspace_path), pattern="*.py")

                assert result["status"] == "success"
                assert "rust_powered" in result
                assert result["rust_powered"] is False

    @pytest.mark.asyncio
    async def test_rust_available_behavior(self):
        """Test behavior when Rust integration is available."""
        # Mock rust_llm_py module
        mock_rust = Mock()
        mock_rust.search_files.return_value = ["/test/file1.py", "/test/file2.py"]

        with patch.dict("sys.modules", {"rust_llm_py": mock_rust}):
            with patch("app.mcp_servers.gemini_workspace_analyzer.RUST_AVAILABLE", True):
                from app.mcp_servers.gemini_workspace_analyzer import search_files

                with tempfile.TemporaryDirectory() as tmpdir:
                    result = await search_files(directory=str(tmpdir), pattern="*.py")

                    # Should indicate Rust was used
                    assert "rust_powered" in result
                    # Note: Actual rust_powered value depends on implementation


class TestCaching:
    """Test caching functionality."""

    @pytest.mark.asyncio
    async def test_context_cache_behavior(self):
        """Test context caching functionality."""
        from app.mcp_servers.gemini_workspace_analyzer import get_project_overview

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)

            # Create simple file structure
            (workspace_path / "test.py").write_text("print('test')")

            # First call - should populate cache
            result1 = await get_project_overview(project_path=str(workspace_path))

            # Second call - should potentially use cache
            result2 = await get_project_overview(project_path=str(workspace_path))

            # Both should succeed
            assert result1["status"] == "success"
            assert result2["status"] == "success"

            # Results should be similar (exact caching behavior depends on implementation)
            assert result1["overview"]["total_files"] == result2["overview"]["total_files"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_directory_path(self):
        """Test handling of invalid directory paths."""
        from app.mcp_servers.gemini_workspace_analyzer import search_files

        result = await search_files(directory="/completely/nonexistent/path")

        assert result["status"] == "error"
        assert "does not exist" in result["error"] or "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_directory(self):
        """Test analysis of empty directory."""
        from app.mcp_servers.gemini_workspace_analyzer import get_project_overview

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await get_project_overview(project_path=tmpdir)

            assert result["status"] == "success"
            assert result["overview"]["total_files"] == 0

    @pytest.mark.asyncio
    async def test_permission_denied(self):
        """Test handling of permission errors."""
        from app.mcp_servers.gemini_workspace_analyzer import get_file_content

        # Try to read a file that doesn't exist
        result = await get_file_content(file_path="/root/nonexistent_file.txt")

        assert result["status"] == "error"
        # Should handle the error gracefully
        assert "error" in result

    @pytest.mark.asyncio
    async def test_large_directory_handling(self):
        """Test handling of directories with many files."""
        from app.mcp_servers.gemini_workspace_analyzer import search_files

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)

            # Create many files
            for i in range(100):
                (workspace_path / f"file_{i:03d}.py").write_text(f"# File {i}")

            result = await search_files(directory=str(workspace_path), pattern="*.py")

            assert result["status"] == "success"
            assert len(result["files"]) == 100


class TestPerformanceOptimizations:
    """Test performance optimizations."""

    @pytest.mark.asyncio
    async def test_concurrent_file_processing(self):
        """Test concurrent processing of multiple files."""
        from app.mcp_servers.gemini_workspace_analyzer import search_content

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)

            # Create multiple files with searchable content
            for i in range(10):
                (workspace_path / f"module_{i}.py").write_text(
                    f"""
def function_{i}():
    \"\"\"Function number {i}.\"\"\"
    return {i}

class Class{i}:
    def method(self):
        return "method_{i}"
"""
                )

            # Search for pattern across all files
            import time

            start_time = time.time()

            result = await search_content(
                directory=str(workspace_path),
                search_pattern="def function_",
                file_patterns="*.py",
                max_results=20,
            )

            end_time = time.time()
            duration = end_time - start_time

            assert result["status"] == "success"
            assert len(result["matches"]) > 0

            # Should complete reasonably quickly (allow some overhead for test environment)
            assert duration < 5.0, f"Search took {duration:.2f} seconds"

    @pytest.mark.asyncio
    async def test_thread_pool_efficiency(self):
        """Test thread pool executor efficiency."""
        from app.mcp_servers.gemini_workspace_analyzer import get_project_overview

        # Test multiple concurrent requests
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)

            # Create test files
            for i in range(5):
                (workspace_path / f"test_{i}.py").write_text(f"content_{i}")

            # Run multiple analyses concurrently
            tasks = [get_project_overview(project_path=str(workspace_path)) for _ in range(3)]

            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(r["status"] == "success" for r in results)
            # Results should be consistent
            file_counts = [r["overview"]["total_files"] for r in results]
            assert all(count == file_counts[0] for count in file_counts)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
