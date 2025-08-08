#!/usr/bin/env python3
"""
Comprehensive test suite for Gemini Code Reviewer MCP Server.

Tests core functionality, Pydantic models, caching, and API integration.
Focuses on achieving 40% coverage of gemini_code_reviewer.py.
"""

import asyncio
import json
from pathlib import Path

# Import the module under test
import sys
import tempfile
import time
from unittest.mock import Mock
from unittest.mock import patch

from pydantic import ValidationError
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.mcp_servers.gemini_code_reviewer import CACHE_TTL
from app.mcp_servers.gemini_code_reviewer import CacheEntry
from app.mcp_servers.gemini_code_reviewer import CodeContext
from app.mcp_servers.gemini_code_reviewer import CodeIssue
from app.mcp_servers.gemini_code_reviewer import CodeMetadata
from app.mcp_servers.gemini_code_reviewer import CodeMetrics
from app.mcp_servers.gemini_code_reviewer import RelatedFile
from app.mcp_servers.gemini_code_reviewer import ReviewResult
from app.mcp_servers.gemini_code_reviewer import SecurityIssue
from app.mcp_servers.gemini_code_reviewer import collect_code_context_fast
from app.mcp_servers.gemini_code_reviewer import enhanced_cache
from app.mcp_servers.gemini_code_reviewer import ensure_model_ready
from app.mcp_servers.gemini_code_reviewer import generate_review_with_gemini
from app.mcp_servers.gemini_code_reviewer import get_file_hash
from app.mcp_servers.gemini_code_reviewer import get_file_stats
from app.mcp_servers.gemini_code_reviewer import read_file_async
from app.mcp_servers.gemini_code_reviewer import review_code
from app.mcp_servers.gemini_code_reviewer import review_performance
from app.mcp_servers.gemini_code_reviewer import review_security
from app.mcp_servers.gemini_code_reviewer import scan_file_for_security


class TestPydanticModels:
    """Test Pydantic model validation and serialization."""

    def test_code_metadata_valid(self):
        """Test CodeMetadata model with valid data."""
        metadata = CodeMetadata(lines=100, size=5000, extension=".py", encoding="utf-8")
        assert metadata.lines == 100
        assert metadata.size == 5000
        assert metadata.extension == ".py"
        assert metadata.encoding == "utf-8"

    def test_code_metadata_invalid_lines(self):
        """Test CodeMetadata model with invalid negative lines."""
        with pytest.raises(ValidationError):
            CodeMetadata(lines=-1, size=5000, extension=".py")

    def test_code_metadata_invalid_size(self):
        """Test CodeMetadata model with invalid negative size."""
        with pytest.raises(ValidationError):
            CodeMetadata(lines=100, size=-1, extension=".py")

    def test_related_file_valid(self):
        """Test RelatedFile model with valid data."""
        related = RelatedFile(path="/test/file.py", content="print('hello')", relationship="test")
        assert related.path == "/test/file.py"
        assert related.content == "print('hello')"
        assert related.relationship == "test"

    def test_related_file_content_truncation(self):
        """Test RelatedFile model content length validation."""
        # Content exactly at limit should pass
        content = "x" * 10000
        related = RelatedFile(path="/test.py", content=content)
        assert len(related.content) == 10000

        # Content over limit should fail
        with pytest.raises(ValidationError):
            long_content = "x" * 10001
            RelatedFile(path="/test.py", content=long_content)

    def test_code_issue_valid(self):
        """Test CodeIssue model with valid data."""
        issue = CodeIssue(
            type="security",
            severity="high",
            line="42",
            description="Potential vulnerability",
            suggestion="Use parameterized queries",
            confidence=0.9,
        )
        assert issue.type == "security"
        assert issue.severity == "high"
        assert issue.confidence == 0.9

    def test_code_issue_invalid_confidence(self):
        """Test CodeIssue model with invalid confidence values."""
        # Test confidence > 1.0
        with pytest.raises(ValidationError):
            CodeIssue(
                type="security",
                severity="high",
                line="42",
                description="Test",
                suggestion="Test",
                confidence=1.5,
            )

        # Test confidence < 0.0
        with pytest.raises(ValidationError):
            CodeIssue(
                type="security",
                severity="high",
                line="42",
                description="Test",
                suggestion="Test",
                confidence=-0.1,
            )

    def test_security_issue_valid(self):
        """Test SecurityIssue model with valid data."""
        issue = SecurityIssue(
            file="/test.py",
            type="sql_injection",
            severity="critical",
            line_number=42,
            description="SQL injection vulnerability",
            cwe_id="CWE-89",
        )
        assert issue.file == "/test.py"
        assert issue.type == "sql_injection"
        assert issue.cwe_id == "CWE-89"

    def test_security_issue_invalid_line_number(self):
        """Test SecurityIssue model with invalid line number."""
        with pytest.raises(ValidationError):
            SecurityIssue(
                file="/test.py", type="sql_injection", severity="critical", line_number=0
            )  # Must be >= 1

    def test_cache_entry_expiration(self):
        """Test CacheEntry TTL and expiration logic."""
        # Fresh entry should not be expired
        entry = CacheEntry(data="test", timestamp=time.time())
        assert not entry.is_expired()

        # Old entry should be expired
        old_entry = CacheEntry(data="test", timestamp=time.time() - CACHE_TTL - 1, ttl=CACHE_TTL)
        assert old_entry.is_expired()

    def test_cache_entry_touch(self):
        """Test CacheEntry access counting."""
        entry = CacheEntry(data="test", timestamp=time.time())
        assert entry.access_count == 0

        entry.touch()
        assert entry.access_count == 1

        entry.touch()
        entry.touch()
        assert entry.access_count == 3


class TestFileOperations:
    """Test file reading and hashing operations."""

    @pytest.mark.asyncio
    async def test_read_file_async_success(self):
        """Test successful file reading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            test_content = "print('Hello, world!')\n"
            f.write(test_content)
            f.flush()

            content = await read_file_async(f.name)
            assert content == test_content

            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_read_file_async_nonexistent(self):
        """Test reading non-existent file."""
        content = await read_file_async("/nonexistent/file.py")
        assert content is None

    @pytest.mark.asyncio
    async def test_get_file_hash_success(self):
        """Test successful file hashing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("test content")
            f.flush()

            file_hash = await get_file_hash(f.name)
            assert file_hash != ""
            assert len(file_hash) == 64  # SHA256 hex length

            # Same content should produce same hash
            file_hash2 = await get_file_hash(f.name)
            assert file_hash == file_hash2

            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_get_file_hash_nonexistent(self):
        """Test hashing non-existent file."""
        file_hash = await get_file_hash("/nonexistent/file.py")
        assert file_hash == ""

    @pytest.mark.asyncio
    async def test_get_file_stats_success(self):
        """Test successful file stats retrieval."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("test content")
            f.flush()

            stats = await get_file_stats(f.name)
            assert stats is not None
            assert "size" in stats
            assert "modified" in stats
            assert stats["size"] > 0
            assert stats["is_file"] is True

            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_get_file_stats_nonexistent(self):
        """Test stats for non-existent file."""
        stats = await get_file_stats("/nonexistent/file.py")
        assert stats is None


class TestCacheSystem:
    """Test the enhanced caching system."""

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        key = "test_key"
        value = {"data": "test_value"}

        # Set value in cache
        await enhanced_cache.set(key, value)

        # Get value from cache
        cached_value = await enhanced_cache.get(key)
        assert cached_value == value

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss for non-existent key."""
        result = await enhanced_cache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache TTL expiration."""
        key = "expire_test"
        value = "test_data"

        # Set with very short TTL
        await enhanced_cache.set(key, value, ttl=1)

        # Should be available immediately
        result = await enhanced_cache.get(key)
        assert result == value

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired now
        result = await enhanced_cache.get(key)
        assert result is None

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        stats = enhanced_cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "cache_size" in stats
        assert isinstance(stats["hit_rate"], float)
        assert 0 <= stats["hit_rate"] <= 1


@pytest.mark.asyncio
class TestContextCollection:
    """Test code context collection functionality."""

    async def test_collect_code_context_fast_success(self):
        """Test successful context collection."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            test_code = '''
def example_function():
    """Example function for testing."""
    return "Hello, world!"

if __name__ == "__main__":
    print(example_function())
'''
            f.write(test_code)
            f.flush()

            context = await collect_code_context_fast(f.name, include_related=False)

            assert isinstance(context, CodeContext)
            assert context.target_file == f.name
            assert context.content == test_code
            assert context.metadata.extension == ".py"
            assert context.metadata.lines > 0
            assert context.metadata.size > 0

            # Cleanup
            Path(f.name).unlink()

    async def test_collect_code_context_with_related_files(self):
        """Test context collection with related files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create main file
            main_file = tmpdir_path / "main.py"
            main_file.write_text("print('main')")

            # Create test file
            test_file = tmpdir_path / "test_main.py"
            test_file.write_text("def test_main(): pass")

            # Create __init__ file
            init_file = tmpdir_path / "__init__.py"
            init_file.write_text("")

            context = await collect_code_context_fast(str(main_file), include_related=True)

            assert isinstance(context, CodeContext)
            assert len(context.related_files) > 0

            # Should find test file and __init__.py
            related_paths = [rf.path for rf in context.related_files]
            assert any("test_main.py" in path for path in related_paths)

    async def test_collect_code_context_nonexistent_file(self):
        """Test context collection for non-existent file."""
        context = await collect_code_context_fast("/nonexistent/file.py")

        assert isinstance(context, CodeContext)
        assert context.content == ""
        assert context.metadata.lines == 0
        assert context.metadata.size == 0


class TestSecurityScanning:
    """Test security pattern detection functionality."""

    @pytest.mark.asyncio
    async def test_scan_file_for_security_sql_injection(self):
        """Test detection of SQL injection patterns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            vulnerable_code = """
def bad_query(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_query(query)
"""
            f.write(vulnerable_code)
            f.flush()

            semaphore = asyncio.Semaphore(1)
            issues = await scan_file_for_security(Path(f.name), semaphore)

            # Should detect SQL injection
            sql_issues = [i for i in issues if i.type == "sql_injection"]
            assert len(sql_issues) > 0
            assert sql_issues[0].severity == "critical"
            assert sql_issues[0].cwe_id == "CWE-89"

            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_scan_file_for_security_hardcoded_credentials(self):
        """Test detection of hardcoded credentials."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            vulnerable_code = """
api_key = "sk-1234567890abcdef"
password = "super_secret_password"
"""
            f.write(vulnerable_code)
            f.flush()

            semaphore = asyncio.Semaphore(1)
            issues = await scan_file_for_security(Path(f.name), semaphore)

            # Should detect hardcoded credentials
            cred_issues = [i for i in issues if i.type == "hardcoded_credentials"]
            assert len(cred_issues) > 0
            assert cred_issues[0].severity == "high"
            assert cred_issues[0].cwe_id == "CWE-798"

            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_scan_file_for_security_clean_code(self):
        """Test scanning clean code with no issues."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            clean_code = '''
def safe_function(data):
    """A safe function with no security issues."""
    return len(data)
'''
            f.write(clean_code)
            f.flush()

            semaphore = asyncio.Semaphore(1)
            issues = await scan_file_for_security(Path(f.name), semaphore)

            # Should find no issues
            assert len(issues) == 0

            # Cleanup
            Path(f.name).unlink()


class TestMCPToolIntegration:
    """Test MCP tool integration through the actual tool interface."""

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    @patch("app.mcp_servers.gemini_code_reviewer.collect_code_context_fast")
    @patch("app.mcp_servers.gemini_code_reviewer.generate_review_with_gemini")
    async def test_review_code_tool_success(self, mock_generate, mock_context, mock_model):
        """Test successful code review through MCP tool interface."""
        # Import the actual MCP tool function
        from app.mcp_servers.gemini_code_reviewer import review_code

        # Setup mocks
        mock_model.return_value = Mock()

        mock_context.return_value = CodeContext(
            target_file="/test.py",
            content="def test(): pass",
            metadata=CodeMetadata(lines=1, size=20, extension=".py"),
            related_files=[],
        )

        mock_generate.return_value = ReviewResult(
            summary="Good code quality",
            quality_score=8.5,
            issues=[],
            positive_aspects=["Clean code structure"],
            recommendations=["Add docstrings"],
            metrics=CodeMetrics(
                complexity="Low", maintainability="High", test_coverage="Needs improvement"
            ),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test(): pass")
            f.flush()

            # Test the actual MCP tool
            result = await review_code(
                file_path=f.name,
                focus_areas="security,performance,quality",
                include_suggestions="true",
                severity_threshold="medium",
            )

            assert result["status"] == "success"
            assert result["file_path"] == f.name
            assert "review" in result
            assert result["review"]["quality_score"] == 8.5
            assert "metadata" in result
            assert result["metadata"]["focus_areas"] == ["security", "performance", "quality"]

            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_review_code_tool_file_not_found(self):
        """Test review_code MCP tool with non-existent file."""
        from app.mcp_servers.gemini_code_reviewer import review_code

        result = await review_code(file_path="/nonexistent/file.py")

        assert result["status"] == "error"
        assert "File not found" in result["error"]

    @pytest.mark.asyncio
    async def test_review_code_tool_parameter_validation(self):
        """Test MCP tool parameter validation and defaults."""
        from app.mcp_servers.gemini_code_reviewer import review_code

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test(): pass")
            f.flush()

            # Test with invalid severity threshold - should use default
            with patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready") as mock_model:
                mock_model.return_value = None  # Will cause early return

                result = await review_code(file_path=f.name, severity_threshold="invalid_threshold")

                # Should handle invalid input gracefully
                assert "severity_threshold" in result.get("metadata", {})

            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    async def test_review_security_tool_success(self, mock_model):
        """Test security review MCP tool functionality."""

        mock_model.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file with security issue
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text('password = "hardcoded123"')

            result = await review_security(
                directory=tmpdir, file_patterns="*.py", scan_depth="comprehensive"
            )

            assert result["status"] == "success"
            assert "scan_summary" in result
            assert result["scan_summary"]["files_scanned"] > 0
            assert "metadata" in result
            assert result["metadata"]["patterns_scanned"] == ["*.py"]

    @pytest.mark.asyncio
    async def test_review_security_tool_directory_not_found(self):
        """Test security review MCP tool with non-existent directory."""

        result = await review_security(directory="/nonexistent/directory")

        assert result["status"] == "error"
        assert "Directory not found" in result["error"]

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    async def test_review_performance_tool_success(self, mock_model):
        """Test performance review MCP tool functionality."""
        from app.mcp_servers.gemini_code_reviewer import review_performance

        # Setup mock model with performance analysis response
        mock_model.return_value = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "overall_assessment": "Good performance",
                "performance_score": 8.0,
                "bottlenecks": [],
                "optimization_opportunities": [],
                "best_practices": [],
                "metrics": {
                    "estimated_complexity": "O(n)",
                    "memory_efficiency": "Good",
                    "io_efficiency": "Excellent",
                    "concurrency_safety": "Thread safe",
                },
            }
        )
        mock_model.return_value.generate_content.return_value = mock_response

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def performance_test(): pass")
            f.flush()

            result = await review_performance(file_path=f.name, include_profiling="true")

            assert result["status"] == "success"
            assert "performance_analysis" in result
            assert result["performance"]["include_profiling"] is True

            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_get_cache_stats_tool(self):
        """Test cache statistics MCP tool."""
        from app.mcp_servers.gemini_code_reviewer import get_cache_stats

        result = await get_cache_stats()

        assert result["status"] == "success"
        assert "cache_statistics" in result
        assert "thread_pool" in result
        assert "configuration" in result
        assert "metadata" in result

        # Verify cache stats structure
        cache_stats = result["cache_statistics"]
        assert "hits" in cache_stats
        assert "misses" in cache_stats
        assert "hit_rate" in cache_stats
        assert "cache_size" in cache_stats

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    async def test_comprehensive_analysis_tool_single_file(self, mock_model):
        """Test comprehensive analysis MCP tool with single file."""
        from app.mcp_servers.gemini_code_reviewer import comprehensive_analysis

        mock_model.return_value = Mock()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test_function(): pass")
            f.flush()

            result = await comprehensive_analysis(
                target_path=f.name,
                analysis_types="code_quality,security",
                file_patterns="*.py",
                max_files="10",
            )

            assert result["status"] == "success"
            assert result["target_path"] == f.name
            assert (
                "code_quality" in result["analyses_performed"]
                or "security" in result["analyses_performed"]
            )
            assert "results" in result
            assert "metadata" in result

            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    async def test_comprehensive_analysis_tool_directory(self, mock_model):
        """Test comprehensive analysis MCP tool with directory."""
        from app.mcp_servers.gemini_code_reviewer import comprehensive_analysis

        mock_model.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test files
            test_files = [
                (Path(tmpdir) / "file1.py", "def function1(): pass"),
                (Path(tmpdir) / "file2.py", "def function2(): pass"),
            ]

            for file_path, content in test_files:
                file_path.write_text(content)

            result = await comprehensive_analysis(
                target_path=tmpdir, analysis_types="security", file_patterns="*.py", max_files="5"
            )

            assert result["status"] == "success"
            assert result["target_path"] == tmpdir
            assert "security" in result["analyses_performed"]
            assert "results" in result


class TestModelIntegration:
    """Test Gemini model integration and mocking."""

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.configure_gemini")
    async def test_ensure_model_ready_success(self, mock_configure):
        """Test successful model initialization."""
        mock_model = Mock()
        mock_configure.return_value = mock_model

        # Reset global model
        import app.mcp_servers.gemini_code_reviewer as gcr

        gcr.gemini_model = None

        model = await ensure_model_ready()
        assert model == mock_model
        mock_configure.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.configure_gemini")
    async def test_ensure_model_ready_cached(self, mock_configure):
        """Test model initialization with cached model."""
        mock_model = Mock()

        # Set cached model
        import app.mcp_servers.gemini_code_reviewer as gcr

        gcr.gemini_model = mock_model

        model = await ensure_model_ready()
        assert model == mock_model
        mock_configure.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    async def test_generate_review_with_gemini_success(self, mock_ensure):
        """Test successful review generation with Gemini."""
        # Setup mock model
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "summary": "Code review completed",
                "quality_score": 7.5,
                "issues": [
                    {
                        "type": "quality",
                        "severity": "medium",
                        "line": "10",
                        "description": "Missing docstring",
                        "suggestion": "Add function documentation",
                        "confidence": 0.8,
                    }
                ],
                "positive_aspects": ["Good variable naming"],
                "recommendations": ["Add type hints"],
                "metrics": {
                    "complexity": "Medium",
                    "maintainability": "Good",
                    "test_coverage": "Needs improvement",
                },
            }
        )
        mock_model.generate_content.return_value = mock_response
        mock_ensure.return_value = mock_model

        # Create test context
        context = CodeContext(
            target_file="/test.py",
            content="def hello(): print('hello')",
            metadata=CodeMetadata(lines=1, size=30, extension=".py"),
            related_files=[],
        )

        result = await generate_review_with_gemini(
            context, focus_areas=["quality"], include_suggestions=True, severity_threshold="medium"
        )

        assert isinstance(result, ReviewResult)
        assert result.quality_score == 7.5
        assert len(result.issues) == 1
        assert result.issues[0].type == "quality"
        assert result.summary == "Code review completed"

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    async def test_generate_review_with_gemini_json_parse_error(self, mock_ensure):
        """Test review generation with JSON parsing error."""
        # Setup mock model with invalid JSON response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Invalid JSON response from model"
        mock_model.generate_content.return_value = mock_response
        mock_ensure.return_value = mock_model

        context = CodeContext(
            target_file="/test.py",
            content="def test(): pass",
            metadata=CodeMetadata(lines=1, size=20, extension=".py"),
            related_files=[],
        )

        result = await generate_review_with_gemini(
            context, focus_areas=["quality"], include_suggestions=True, severity_threshold="medium"
        )

        # Should return fallback result
        assert isinstance(result, ReviewResult)
        assert "parsing failed" in result.summary
        assert result.quality_score == 5.0
        assert len(result.issues) == 0


class TestPerformanceAnalysis:
    """Test performance analysis functionality."""

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    @patch("app.mcp_servers.gemini_code_reviewer.collect_code_context_fast")
    async def test_review_performance_success(self, mock_context, mock_model):
        """Test successful performance analysis."""
        # Setup mocks
        mock_model.return_value = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "overall_assessment": "Good performance",
                "performance_score": 8.0,
                "bottlenecks": [
                    {
                        "type": "algorithmic",
                        "severity": "medium",
                        "line_range": "15-20",
                        "description": "Nested loop could be optimized",
                        "impact": "O(n²) complexity",
                        "solution": "Use hash map for lookup",
                        "estimated_improvement": "50% faster",
                    }
                ],
                "optimization_opportunities": ["Use list comprehension"],
                "best_practices": ["Avoid premature optimization"],
                "metrics": {
                    "estimated_complexity": "O(n)",
                    "memory_efficiency": "Good",
                    "io_efficiency": "Excellent",
                    "concurrency_safety": "Thread safe",
                },
            }
        )
        mock_model.return_value.generate_content.return_value = mock_response

        mock_context.return_value = CodeContext(
            target_file="/test.py",
            content="def performance_test(): pass",
            metadata=CodeMetadata(lines=1, size=30, extension=".py"),
            related_files=[],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test(): pass")
            f.flush()

            result = await review_performance(f.name)

            assert result["status"] == "success"
            assert "performance_analysis" in result
            analysis = result["performance_analysis"]
            assert analysis["performance_score"] == 8.0
            assert len(analysis["bottlenecks"]) == 1

            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_review_performance_file_not_found(self):
        """Test performance review with non-existent file."""
        result = await review_performance("/nonexistent/file.py")

        assert result["status"] == "error"
        assert "File not found" in result["error"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_review_code_with_exception(self):
        """Test review_code with exception during processing."""
        with patch(
            "app.mcp_servers.gemini_code_reviewer.collect_code_context_fast"
        ) as mock_context:
            mock_context.side_effect = Exception("Test exception")

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write("def test(): pass")
                f.flush()

                result = await review_code(f.name)

                assert result["status"] == "error"
                assert "Test exception" in result["error"]

                # Cleanup
                Path(f.name).unlink()

    def test_invalid_input_parameters(self):
        """Test handling of invalid input parameters."""

        # Test with invalid severity threshold
        async def run_test():
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write("def test(): pass")
                f.flush()

                # Should handle invalid severity and use default
                result = await review_code(f.name, severity_threshold="invalid")

                # Should not fail but use default
                assert "severity_threshold" in result.get("metadata", {})

                # Cleanup
                Path(f.name).unlink()

        asyncio.run(run_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
