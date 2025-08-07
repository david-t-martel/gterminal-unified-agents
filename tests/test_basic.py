"""Basic tests for Gemini CLI components."""

from pathlib import Path

import pytest

from gemini_cli.tools.code_analysis import CodeAnalysisTool
from gemini_cli.tools.filesystem import FilesystemTool
from gemini_cli.tools.registry import ToolRegistry


def test_tool_registry():
    """Test tool registry functionality."""
    registry = ToolRegistry()

    # Test empty registry
    assert len(registry.list_tools()) == 0
    assert registry.get_tool("nonexistent") is None
    assert not registry.has_tool("nonexistent")

    # Test tool registration
    tool = FilesystemTool()
    registry.register("test_tool", tool)

    assert registry.has_tool("test_tool")
    assert registry.get_tool("test_tool") == tool
    assert "test_tool" in registry.list_tools()

    # Test tool unregistration
    registry.unregister("test_tool")
    assert not registry.has_tool("test_tool")


def test_filesystem_tool():
    """Test filesystem tool properties."""
    tool = FilesystemTool()

    assert tool.name == "filesystem"
    assert isinstance(tool.description, str)
    assert len(tool.description) > 0


def test_code_analysis_tool():
    """Test code analysis tool properties."""
    tool = CodeAnalysisTool()

    assert tool.name == "code_analysis"
    assert isinstance(tool.description, str)
    assert len(tool.description) > 0


@pytest.mark.asyncio
async def test_filesystem_tool_read_file():
    """Test filesystem tool read operation."""
    tool = FilesystemTool()

    # Test missing parameters
    result = await tool.execute({})
    assert "error" in result
    assert "action" in result["error"]

    # Test missing path
    result = await tool.execute({"action": "read_file"})
    assert "error" in result
    assert "path" in result["error"]


@pytest.mark.asyncio
async def test_code_analysis_tool_basic():
    """Test code analysis tool basic functionality."""
    tool = CodeAnalysisTool()

    # Test missing parameters
    result = await tool.execute({})
    assert "error" in result
    assert "path" in result["error"]


def test_language_detection():
    """Test language detection functionality."""
    tool = CodeAnalysisTool()

    assert tool._detect_language(Path("test.py")) == "python"
    assert tool._detect_language(Path("test.js")) == "javascript"
    assert tool._detect_language(Path("test.ts")) == "typescript"
    assert tool._detect_language(Path("test.rs")) == "rust"
    assert tool._detect_language(Path("test.unknown")) == "unknown"


def test_comment_counting():
    """Test comment line counting."""
    tool = CodeAnalysisTool()

    python_lines = [
        "# This is a comment",
        "def function():",
        "    # Another comment",
        "    return True",
    ]

    assert tool._count_comment_lines(python_lines, "python") == 2

    js_lines = [
        "// This is a comment",
        "function test() {",
        "    // Another comment",
        "    return true;",
        "}",
    ]

    assert tool._count_comment_lines(js_lines, "javascript") == 2


def test_complexity_analysis():
    """Test basic complexity analysis."""
    tool = CodeAnalysisTool()

    simple_code = "def simple(): return True"
    complex_code = """
def complex_function():
    if condition1:
        for item in items:
            if condition2:
                while loop_condition:
                    try:
                        process_item(item)
                    except Exception:
                        handle_error()
    else:
        return None
"""

    simple_complexity = tool._analyze_complexity(simple_code, "python")
    complex_complexity = tool._analyze_complexity(complex_code, "python")

    assert complex_complexity > simple_complexity
    assert simple_complexity >= 0
    assert complex_complexity <= 100  # Capped at 100
