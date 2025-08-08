"""
Unit tests for Gemini-based MCP server agents.
Tests the working Gemini agents to increase test coverage.
"""

import asyncio
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestGeminiMCPBase:
    """Test the Gemini MCP base class functionality."""

    @patch("google.genai.GenerativeModel")
    def test_gemini_mcp_base_import(self, mock_model):
        """Test that Gemini MCP base can be imported."""
        from app.mcp_servers.gemini_mcp_base import GeminiMCPBase

        # Create instance with mock
        base = GeminiMCPBase(name="test-server", model_name="gemini-2.5-flash")

        assert base.name == "test-server"
        assert base.model_name == "gemini-2.5-flash"

    @patch("google.genai.GenerativeModel")
    def test_gemini_base_tools(self, mock_model):
        """Test Gemini base tool registration."""
        from app.mcp_servers.gemini_mcp_base import GeminiMCPBase

        base = GeminiMCPBase(name="test-server", model_name="gemini-2.5-flash")

        # Mock tool registration
        @base.tool()
        def test_tool(param: str) -> str:
            """Test tool function."""
            return f"Result: {param}"

        # Tool should be registered
        assert hasattr(base, "test_tool")


class TestGeminiCodeReviewer:
    """Test the Gemini Code Reviewer MCP server."""

    @patch("google.genai.GenerativeModel")
    def test_code_reviewer_import(self, mock_model):
        """Test that code reviewer can be imported."""
        from app.mcp_servers.gemini_code_reviewer import format_code_review_prompt

        # Test format_code_review_prompt
        prompt = format_code_review_prompt(
            code="def test(): pass", language="python", context="Unit test"
        )
        assert "def test():" in prompt
        assert "python" in prompt.lower()

    def test_review_response_parsing(self):
        """Test parsing of code review responses."""
        from app.mcp_servers.gemini_code_reviewer import parse_review_response

        response = """
        {
            "issues": [
                {"severity": "high", "message": "Security issue"},
                {"severity": "low", "message": "Style issue"}
            ],
            "suggestions": ["Add docstring"],
            "score": 75
        }
        """

        parsed = parse_review_response(response)
        assert "issues" in parsed
        assert len(parsed["issues"]) == 2
        assert parsed["score"] == 75

    def test_review_score_calculation(self):
        """Test review score calculation."""
        from app.mcp_servers.gemini_code_reviewer import calculate_review_score

        issues = [
            {"severity": "critical", "count": 1},
            {"severity": "high", "count": 2},
            {"severity": "medium", "count": 3},
            {"severity": "low", "count": 5},
        ]

        score = calculate_review_score(issues)
        assert isinstance(score, int | float)
        assert 0 <= score <= 100


class TestGeminiWorkspaceAnalyzer:
    """Test the Gemini Workspace Analyzer MCP server."""

    @patch("google.genai.GenerativeModel")
    def test_workspace_analyzer_import(self, mock_model):
        """Test that workspace analyzer can be imported."""
        from app.mcp_servers.gemini_workspace_analyzer import analyze_project_structure

        # Test analyze_project_structure
        structure = analyze_project_structure(
            {"files": ["main.py", "test.py"], "directories": ["src/", "tests/"]}
        )
        assert "file_count" in structure
        assert "directory_count" in structure

    def test_framework_detection(self):
        """Test framework detection."""
        from app.mcp_servers.gemini_workspace_analyzer import detect_frameworks

        files = ["package.json", "requirements.txt", "Cargo.toml", "go.mod"]

        frameworks = detect_frameworks(files)
        assert "javascript" in frameworks or "node" in frameworks
        assert "python" in frameworks
        assert "rust" in frameworks
        assert "go" in frameworks

    def test_complexity_calculation(self):
        """Test project complexity calculation."""
        from app.mcp_servers.gemini_workspace_analyzer import calculate_complexity

        metrics = {
            "total_files": 100,
            "total_lines": 5000,
            "languages": ["python", "javascript", "rust"],
            "dependencies": 25,
        }

        complexity = calculate_complexity(metrics)
        assert complexity in ["low", "medium", "high", "very_high"]


class TestGeminiMasterArchitect:
    """Test the Gemini Master Architect MCP server."""

    @patch("google.genai.GenerativeModel")
    def test_master_architect_import(self, mock_model):
        """Test that master architect can be imported."""
        from app.mcp_servers.gemini_master_architect import analyze_architecture

        # Test analyze_architecture
        analysis = analyze_architecture(
            {
                "components": ["api", "database", "cache"],
                "connections": [["api", "database"], ["api", "cache"]],
            }
        )
        assert "component_count" in analysis
        assert "connection_count" in analysis

    def test_architecture_recommendations(self):
        """Test architecture recommendation generation."""
        from app.mcp_servers.gemini_master_architect import generate_recommendations

        issues = [
            {"type": "coupling", "severity": "high"},
            {"type": "scalability", "severity": "medium"},
            {"type": "security", "severity": "critical"},
        ]

        recommendations = generate_recommendations(issues)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should prioritize critical issues
        if recommendations:
            assert any("security" in r.lower() for r in recommendations)

    def test_pattern_identification(self):
        """Test architectural pattern identification."""
        from app.mcp_servers.gemini_master_architect import identify_patterns

        structure = {
            "directories": ["models/", "views/", "controllers/"],
            "files": ["app.py", "routes.py", "database.py"],
        }

        patterns = identify_patterns(structure)
        assert isinstance(patterns, list)
        # Should identify MVC pattern
        assert any("mvc" in p.lower() for p in patterns)


class TestGeminiCostGovernor:
    """Test the Gemini Cost Governor MCP server."""

    @patch("google.genai.GenerativeModel")
    def test_cost_governor_import(self, mock_model):
        """Test that cost governor can be imported."""
        from app.mcp_servers.gemini_cost_governor import calculate_token_cost

        # Test calculate_token_cost
        cost = calculate_token_cost(input_tokens=1000, output_tokens=500, model="gemini-2.5-flash")
        assert isinstance(cost, int | float)
        assert cost >= 0

    def test_cost_estimation(self):
        """Test operation cost estimation."""
        from app.mcp_servers.gemini_cost_governor import estimate_operation_cost

        operation = {
            "type": "code_review",
            "file_count": 10,
            "avg_file_size": 500,
            "model": "gemini-2.5-pro",
        }

        estimate = estimate_operation_cost(operation)
        assert "estimated_cost" in estimate
        assert "estimated_tokens" in estimate
        assert estimate["estimated_cost"] > 0

    def test_prompt_optimization(self):
        """Test prompt optimization for cost reduction."""
        from app.mcp_servers.gemini_cost_governor import optimize_prompt_for_cost

        original_prompt = (
            """
        Please analyze this code in great detail and provide comprehensive feedback
        including all possible improvements, suggestions, best practices, and
        potential issues that might arise in any scenario.
        """
            * 10
        )  # Make it long

        optimized = optimize_prompt_for_cost(original_prompt)
        assert len(optimized) < len(original_prompt)
        assert "analyze" in optimized.lower()


class TestGeminiSchemas:
    """Test the Gemini schema definitions."""

    def test_schema_imports(self):
        """Test that schemas can be imported."""
        from app.mcp_servers.gemini_schemas import ArchitectureRequest
        from app.mcp_servers.gemini_schemas import CodeReviewRequest
        from app.mcp_servers.gemini_schemas import WorkspaceAnalysisRequest

        # Test CodeReviewRequest
        review_req = CodeReviewRequest(
            code="def test(): pass", language="python", context="unit test"
        )
        assert review_req.code == "def test(): pass"
        assert review_req.language == "python"

        # Test WorkspaceAnalysisRequest
        workspace_req = WorkspaceAnalysisRequest(
            path="/project", patterns=["*.py", "*.js"], max_depth=3
        )
        assert workspace_req.path == "/project"
        assert len(workspace_req.patterns) == 2

        # Test ArchitectureRequest
        arch_req = ArchitectureRequest(
            project_path="/app", analysis_type="microservices", include_diagrams=True
        )
        assert arch_req.project_path == "/app"
        assert arch_req.include_diagrams is True


class TestGeminiIntegration:
    """Test integration between different Gemini components."""

    @patch("google.genai.GenerativeModel")
    async def test_async_gemini_workflow(self, mock_model):
        """Test async workflow with Gemini agents."""
        from app.mcp_servers.gemini_mcp_base import GeminiMCPBase

        # Create mock async response
        mock_response = AsyncMock()
        mock_response.text = json.dumps({"result": "success", "data": {"analyzed": True}})
        mock_model.return_value.generate_content_async = mock_response

        base = GeminiMCPBase(name="test-async", model_name="gemini-2.5-flash")

        # Define async tool
        @base.tool()
        async def async_analyze(content: str) -> dict:
            """Async analysis tool."""
            response = await mock_model.return_value.generate_content_async(content)
            return json.loads(response.text)

        # Test async execution
        result = await async_analyze("test content")
        assert result["result"] == "success"
        assert result["data"]["analyzed"] is True

    @patch("google.genai.GenerativeModel")
    def test_gemini_tool_chaining(self, mock_model):
        """Test chaining multiple Gemini tools."""
        from app.mcp_servers.gemini_mcp_base import GeminiMCPBase

        base = GeminiMCPBase(name="test-chain", model_name="gemini-2.5-flash")

        # Define tools that can be chained
        @base.tool()
        def analyze_code(code: str) -> dict:
            """Analyze code."""
            return {"issues": ["issue1"], "score": 80}

        @base.tool()
        def generate_report(analysis: dict) -> str:
            """Generate report from analysis."""
            issues = analysis.get("issues", [])
            score = analysis.get("score", 0)
            return f"Report: {len(issues)} issues found, score: {score}"

        # Chain the tools
        code_analysis = analyze_code("def test(): pass")
        report = generate_report(code_analysis)

        assert "1 issues found" in report
        assert "score: 80" in report

    def test_gemini_error_handling(self):
        """Test error handling in Gemini agents."""
        from app.mcp_servers.gemini_mcp_base import GeminiError
        from app.mcp_servers.gemini_mcp_base import handle_gemini_error
        from app.mcp_servers.gemini_mcp_base import retry_on_error

        # Test GeminiError
        error = GeminiError("API rate limit exceeded")
        assert str(error) == "API rate limit exceeded"

        # Test error handler
        @handle_gemini_error
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()
        assert result is None or "error" in str(result).lower()

        # Test retry decorator
        call_count = 0

        @retry_on_error(max_retries=3)
        def retry_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = retry_function()
        assert result == "success"
        assert call_count == 3


class TestGeminiPerformance:
    """Test performance aspects of Gemini agents."""

    @pytest.mark.performance
    def test_gemini_response_caching(self):
        """Test response caching for Gemini agents."""
        from app.mcp_servers.gemini_mcp_base import ResponseCache

        cache = ResponseCache(max_size=100, ttl=60)

        # Add items to cache
        cache.set("key1", {"data": "value1"})
        cache.set("key2", {"data": "value2"})

        # Retrieve from cache
        assert cache.get("key1") == {"data": "value1"}
        assert cache.get("key2") == {"data": "value2"}
        assert cache.get("key3") is None

        # Test cache size limit
        for i in range(150):
            cache.set(f"key_{i}", {"data": f"value_{i}"})

        # Cache should not exceed max_size
        assert len(cache) <= 100

    @pytest.mark.performance
    async def test_gemini_concurrent_requests(self):
        """Test concurrent request handling."""
        from app.mcp_servers.gemini_mcp_base import GeminiMCPBase

        with patch("google.genai.GenerativeModel") as mock_model:
            mock_response = Mock()
            mock_response.text = "response"
            mock_model.return_value.generate_content = Mock(return_value=mock_response)

            base = GeminiMCPBase(name="test-concurrent", model_name="gemini-2.5-flash")

            # Define tool for concurrent testing
            @base.tool()
            async def concurrent_tool(id: int) -> str:
                """Tool for concurrent testing."""
                await asyncio.sleep(0.01)  # Simulate work
                return f"Result {id}"

            # Run multiple concurrent requests
            tasks = [concurrent_tool(i) for i in range(10)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(f"Result {i}" == results[i] for i in range(10))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
