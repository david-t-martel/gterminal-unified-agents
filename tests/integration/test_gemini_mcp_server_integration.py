#!/usr/bin/env python3
"""
Integration tests for Gemini MCP Server interactions.

Tests end-to-end workflows combining multiple MCP servers and agent functionality.
Focuses on realistic usage scenarios and cross-server communication.
"""

import asyncio
import json
from pathlib import Path

# Import the modules under test
import sys
import tempfile
from unittest.mock import Mock
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))


class TestCodeReviewWorkflow:
    """Test end-to-end code review workflow combining multiple servers."""

    def create_complex_project(self, base_path: Path) -> None:
        """Create a complex project structure for integration testing."""
        # Main application
        src_dir = base_path / "src"
        src_dir.mkdir()

        (src_dir / "__init__.py").write_text("")
        (src_dir / "app.py").write_text(
            """
#!/usr/bin/env python3
\"\"\"Main application with potential issues for testing.\"\"\"

import os
import sqlite3
import hashlib

# Security issue: hardcoded secret
SECRET_KEY = "my_super_secret_key_123"

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = None

    def connect(self):
        \"\"\"Connect to database.\"\"\"
        self.connection = sqlite3.connect(self.db_path)

    def get_user(self, user_id):
        \"\"\"Get user by ID - potential SQL injection.\"\"\"
        cursor = self.connection.cursor()
        # Security issue: SQL injection vulnerability
        query = f"SELECT * FROM users WHERE id = {user_id}"
        cursor.execute(query)
        return cursor.fetchone()

    def hash_password(self, password):
        \"\"\"Hash password - using weak hash function.\"\"\"
        # Security issue: weak hashing algorithm
        return hashlib.md5(password.encode()).hexdigest()

    def inefficient_search(self, items, target):
        \"\"\"Inefficient search algorithm.\"\"\"
        # Performance issue: O(n²) complexity
        for i in range(len(items)):
            for j in range(len(items)):
                if items[i] == target and items[j] == target:
                    return i
        return -1

class UserManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def authenticate(self, username, password):
        \"\"\"Authenticate user.\"\"\"
        # Missing input validation
        hashed = self.db_manager.hash_password(password)
        # More SQL injection potential
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{hashed}'"
        # ... rest of implementation
        pass

if __name__ == "__main__":
    db = DatabaseManager("app.db")
    db.connect()
    user_mgr = UserManager(db)
"""
        )

        # Configuration module
        (src_dir / "config.py").write_text(
            """
\"\"\"Configuration management.\"\"\"

import os
from typing import Any, Dict, Optional, Union

class Config:
    \"\"\"Application configuration.\"\"\"

    def __init__(self):
        self.settings = {}

    def load_from_env(self):
        \"\"\"Load configuration from environment variables.\"\"\"
        self.settings = {
            'database_url': os.getenv('DATABASE_URL', 'sqlite:///app.db'),
            'secret_key': os.getenv('SECRET_KEY', 'default-secret'),
            'debug': os.getenv('DEBUG', 'False').lower() == 'true',
            'max_connections': int(os.getenv('MAX_CONNECTIONS', '100')),
        }

    def get(self, key: str, default: Any = None) -> Any:
        \"\"\"Get configuration value.\"\"\"
        return self.settings.get(key, default)

    def validate(self) -> bool:
        \"\"\"Validate configuration.\"\"\"
        required_keys = ['database_url', 'secret_key']
        for key in required_keys:
            if not self.get(key):
                return False
        return True

# Global configuration instance
config = Config()
"""
        )

        # Tests with some issues
        tests_dir = base_path / "tests"
        tests_dir.mkdir()

        (tests_dir / "__init__.py").write_text("")
        (tests_dir / "test_app.py").write_text(
            """
import unittest
from src.app import DatabaseManager, UserManager

class TestDatabaseManager(unittest.TestCase):
    \"\"\"Test database manager functionality.\"\"\"

    def setUp(self):
        self.db_manager = DatabaseManager(":memory:")
        self.db_manager.connect()

    def test_hash_password(self):
        \"\"\"Test password hashing.\"\"\"
        password = "test123"
        hashed = self.db_manager.hash_password(password)
        self.assertIsNotNone(hashed)
        self.assertNotEqual(password, hashed)

    def test_inefficient_search(self):
        \"\"\"Test search functionality.\"\"\"
        items = [1, 2, 3, 4, 5]
        result = self.db_manager.inefficient_search(items, 3)
        self.assertEqual(result, 2)

    # Missing tearDown method
    # Missing test for SQL injection vulnerability
    # Missing test for authentication

class TestUserManager(unittest.TestCase):
    \"\"\"Test user manager functionality.\"\"\"

    def test_authenticate(self):
        \"\"\"Test user authentication.\"\"\"
        # Incomplete test
        pass

if __name__ == "__main__":
    unittest.main()
"""
        )

        # Configuration files
        (base_path / "requirements.txt").write_text(
            """
sqlite3
hashlib
requests==2.28.0
flask==2.0.1
"""
        )

        (base_path / "setup.py").write_text(
            """
from setuptools import setup, find_packages

setup(
    name="complex-app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
        "flask>=2.0.1",
    ],
)
"""
        )

        (base_path / "README.md").write_text(
            """
# Complex Application

A complex application with intentional issues for testing.

## Security Issues
- Hardcoded secrets
- SQL injection vulnerabilities
- Weak hashing algorithms

## Performance Issues
- Inefficient algorithms
- Missing optimizations

## Code Quality Issues
- Missing error handling
- Incomplete tests
- Poor documentation
"""
        )

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    @patch("app.mcp_servers.gemini_master_architect.configure_gemini")
    @patch("app.mcp_servers.gemini_workspace_analyzer.configure_gemini")
    async def test_comprehensive_project_analysis(
        self, mock_ws_gemini, mock_arch_gemini, mock_review_gemini
    ):
        """Test comprehensive analysis workflow using all three MCP servers."""
        # Setup all mock models
        for mock_gemini in [mock_ws_gemini, mock_arch_gemini, mock_review_gemini]:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps(
                {
                    "analysis": "Comprehensive analysis completed",
                    "findings": [
                        "Multiple security issues found",
                        "Performance optimizations needed",
                    ],
                    "recommendations": [
                        "Fix SQL injection vulnerabilities",
                        "Improve algorithm efficiency",
                    ],
                }
            )
            mock_model.generate_content.return_value = mock_response
            mock_gemini.return_value = mock_model

        # Import the MCP tools
        from app.mcp_servers.gemini_code_reviewer import comprehensive_analysis
        from app.mcp_servers.gemini_code_reviewer import review_security
        from app.mcp_servers.gemini_master_architect import analyze_system_architecture
        from app.mcp_servers.gemini_workspace_analyzer import analyze_workspace
        from app.mcp_servers.gemini_workspace_analyzer import search_files

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            self.create_complex_project(project_path)

            # Step 1: Workspace analysis - get project overview
            workspace_result = await analyze_workspace(
                project_path=str(project_path),
                analysis_depth="comprehensive",
                include_dependencies=True,
                include_tests=True,
                focus_areas="structure,quality",
            )

            assert workspace_result["status"] == "success"
            assert "analysis" in workspace_result

            # Step 2: Architecture analysis - understand system design
            architecture_result = await analyze_system_architecture(
                project_path=str(project_path),
                analysis_depth="comprehensive",
                focus_areas="security,scalability",
                include_dependencies="true",
                include_tests="true",
            )

            assert architecture_result["status"] == "success"

            # Step 3: Security review - find vulnerabilities
            security_result = await review_security(
                directory=str(project_path), file_patterns="*.py", scan_depth="comprehensive"
            )

            assert security_result["status"] == "success"
            assert "scan_summary" in security_result

            # Step 4: File search - find specific files
            search_result = await search_files(
                directory=str(project_path), pattern="*.py", exclude_patterns="__pycache__"
            )

            assert search_result["status"] == "success"
            assert len(search_result["files"]) > 0

            # Step 5: Comprehensive code analysis
            comprehensive_result = await comprehensive_analysis(
                target_path=str(project_path),
                analysis_types="code_quality,security,performance",
                file_patterns="*.py",
                max_files="10",
            )

            assert comprehensive_result["status"] == "success"
            assert "results" in comprehensive_result

            # Verify workflow completed successfully
            assert all(
                result["status"] == "success"
                for result in [
                    workspace_result,
                    architecture_result,
                    security_result,
                    search_result,
                    comprehensive_result,
                ]
            )

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    async def test_security_focused_workflow(self, mock_gemini):
        """Test security-focused analysis workflow."""
        # Setup mock for security analysis
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "security_analysis": {"critical_issues": 3, "high_issues": 5, "medium_issues": 2},
                "vulnerabilities": [
                    {"type": "sql_injection", "severity": "critical", "file": "app.py", "line": 25},
                    {
                        "type": "hardcoded_credentials",
                        "severity": "high",
                        "file": "app.py",
                        "line": 9,
                    },
                    {"type": "weak_crypto", "severity": "high", "file": "app.py", "line": 35},
                ],
                "recommendations": [
                    "Use parameterized queries to prevent SQL injection",
                    "Store secrets in environment variables",
                    "Use strong hashing algorithms like bcrypt",
                ],
            }
        )
        mock_model.generate_content.return_value = mock_response
        mock_gemini.return_value = mock_model

        from app.mcp_servers.gemini_code_reviewer import review_code
        from app.mcp_servers.gemini_code_reviewer import review_security

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            self.create_complex_project(project_path)

            # Security scan of entire project
            security_result = await review_security(
                directory=str(project_path), file_patterns="*.py", scan_depth="comprehensive"
            )

            assert security_result["status"] == "success"

            # Detailed review of main application file
            app_file = project_path / "src" / "app.py"
            code_review_result = await review_code(
                file_path=str(app_file),
                focus_areas="security",
                include_suggestions="true",
                severity_threshold="high",
            )

            assert code_review_result["status"] == "success"
            assert "review" in code_review_result

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_master_architect.configure_gemini")
    async def test_architecture_analysis_workflow(self, mock_gemini):
        """Test architecture analysis workflow."""
        # Setup mock for architecture analysis
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "architecture_analysis": {
                    "pattern": "layered_architecture",
                    "complexity": "medium",
                    "maintainability": 7.5,
                    "scalability": 6.0,
                },
                "components": [
                    {
                        "name": "DatabaseManager",
                        "type": "data_access",
                        "responsibilities": ["database_operations"],
                    },
                    {
                        "name": "UserManager",
                        "type": "business_logic",
                        "responsibilities": ["user_management"],
                    },
                    {
                        "name": "Config",
                        "type": "configuration",
                        "responsibilities": ["settings_management"],
                    },
                ],
                "improvements": [
                    "Separate concerns between data access and business logic",
                    "Add proper error handling and logging",
                    "Implement dependency injection for better testability",
                ],
            }
        )
        mock_model.generate_content.return_value = mock_response
        mock_gemini.return_value = mock_model

        from app.mcp_servers.gemini_master_architect import analyze_code_relationships
        from app.mcp_servers.gemini_master_architect import analyze_system_architecture
        from app.mcp_servers.gemini_master_architect import generate_refactoring_plan

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            self.create_complex_project(project_path)

            # System architecture analysis
            arch_result = await analyze_system_architecture(
                project_path=str(project_path),
                analysis_depth="comprehensive",
                focus_areas="maintainability,scalability",
            )

            assert arch_result["status"] == "success"

            # Code relationships analysis
            relationships_result = await analyze_code_relationships(
                project_path=str(project_path), entry_points="src/app.py", max_depth="5"
            )

            assert relationships_result["status"] == "success"

            # Refactoring plan generation
            refactoring_result = await generate_refactoring_plan(
                project_path=str(project_path),
                target_improvements="security,performance",
                risk_tolerance="medium",
            )

            assert refactoring_result["status"] == "success"


class TestMCPServerCommunication:
    """Test communication patterns between MCP servers."""

    @pytest.mark.asyncio
    async def test_server_isolation(self):
        """Test that MCP servers operate independently."""
        from app.mcp_servers.gemini_code_reviewer import mcp as review_mcp
        from app.mcp_servers.gemini_master_architect import mcp as arch_mcp
        from app.mcp_servers.gemini_workspace_analyzer import mcp as workspace_mcp

        # Verify servers have different names
        assert review_mcp.name == "gemini-code-reviewer"
        assert arch_mcp.name == "gemini-master-architect"
        assert workspace_mcp.name == "gemini-workspace-analyzer"

        # Verify servers are independent instances
        assert review_mcp is not arch_mcp
        assert arch_mcp is not workspace_mcp
        assert workspace_mcp is not review_mcp

    @pytest.mark.asyncio
    async def test_shared_configuration(self):
        """Test that servers can share configuration appropriately."""
        # Test that all servers can handle the same project path
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "test.py").write_text("def test(): pass")

            # Test each server can handle the same project
            from app.mcp_servers.gemini_code_reviewer import get_cache_stats
            from app.mcp_servers.gemini_workspace_analyzer import get_project_overview

            # These should not interfere with each other
            cache_result = await get_cache_stats()
            overview_result = await get_project_overview(project_path=str(project_path))

            assert cache_result["status"] == "success"
            assert overview_result["status"] == "success"


class TestPerformanceIntegration:
    """Test performance characteristics of integrated workflows."""

    @pytest.mark.asyncio
    async def test_concurrent_server_operations(self):
        """Test running multiple MCP server operations concurrently."""
        from app.mcp_servers.gemini_code_reviewer import get_cache_stats
        from app.mcp_servers.gemini_workspace_analyzer import get_project_overview

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create test files
            for i in range(5):
                (project_path / f"file_{i}.py").write_text(f"def function_{i}(): pass")

            # Run operations concurrently
            import time

            start_time = time.time()

            tasks = [
                get_cache_stats(),
                get_project_overview(project_path=str(project_path)),
                get_cache_stats(),
                get_project_overview(project_path=str(project_path)),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            duration = end_time - start_time

            # All operations should succeed
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 2

            # Should complete in reasonable time
            assert duration < 10.0, f"Concurrent operations took {duration:.2f} seconds"

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    async def test_large_project_handling(self, mock_gemini):
        """Test handling of larger projects across multiple servers."""
        # Setup mock
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "analysis": "Large project analysis completed",
                "performance": {"files_processed": 50, "time_taken": 5.2},
            }
        )
        mock_model.generate_content.return_value = mock_response
        mock_gemini.return_value = mock_model

        from app.mcp_servers.gemini_code_reviewer import comprehensive_analysis
        from app.mcp_servers.gemini_workspace_analyzer import search_files

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create larger project structure
            for i in range(50):
                module_dir = project_path / f"module_{i}"
                module_dir.mkdir()

                (module_dir / "__init__.py").write_text("")
                (module_dir / "main.py").write_text(
                    f"""
def function_{i}():
    return {i}

class Class{i}:
    def method(self):
        return "result_{i}"
"""
                )

            # Test file search performance
            search_result = await search_files(directory=str(project_path), pattern="*.py")

            assert search_result["status"] == "success"
            assert len(search_result["files"]) == 100  # 50 __init__.py + 50 main.py

            # Test analysis performance with file limit
            analysis_result = await comprehensive_analysis(
                target_path=str(project_path),
                analysis_types="code_quality",
                max_files="20",  # Limit for performance
            )

            assert analysis_result["status"] == "success"


class TestErrorHandlingIntegration:
    """Test error handling across integrated MCP server operations."""

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self):
        """Test handling when some operations fail."""
        from app.mcp_servers.gemini_code_reviewer import review_code
        from app.mcp_servers.gemini_workspace_analyzer import get_project_overview

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "test.py").write_text("def test(): pass")

            # One operation that should succeed
            overview_result = await get_project_overview(project_path=str(project_path))

            # One operation that should fail (nonexistent file)
            review_result = await review_code(file_path="/nonexistent/file.py")

            # Verify mixed results
            assert overview_result["status"] == "success"
            assert review_result["status"] == "error"

            # The successful operation should not be affected by the failed one
            assert "overview" in overview_result

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of operation timeouts."""
        from app.mcp_servers.gemini_workspace_analyzer import search_content

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create files for search
            for i in range(10):
                (project_path / f"file_{i}.py").write_text(f"def function_{i}(): pass")

            # Test with reasonable parameters
            result = await search_content(
                directory=str(project_path), search_pattern="def function_", max_results=5
            )

            assert result["status"] == "success"
            # Should handle the search operation within reasonable time

    @pytest.mark.asyncio
    async def test_resource_cleanup(self):
        """Test that resources are properly cleaned up after operations."""
        from app.mcp_servers.gemini_code_reviewer import get_cache_stats

        # Get initial cache stats
        initial_stats = await get_cache_stats()
        initial_cache_size = initial_stats["cache_statistics"]["cache_size"]

        # Perform some operations that might affect cache
        for _i in range(3):
            stats = await get_cache_stats()
            assert stats["status"] == "success"

        # Get final cache stats
        final_stats = await get_cache_stats()
        final_cache_size = final_stats["cache_statistics"]["cache_size"]

        # Cache size should be reasonable (not growing unbounded)
        assert final_cache_size >= initial_cache_size
        # Should not grow excessively
        assert final_cache_size < initial_cache_size + 100


class TestRealWorldScenarios:
    """Test realistic development scenarios."""

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    @patch("app.mcp_servers.gemini_workspace_analyzer.configure_gemini")
    async def test_new_developer_onboarding(self, mock_ws_gemini, mock_review_gemini):
        """Test scenario: new developer exploring unfamiliar codebase."""
        # Setup mocks
        for mock_gemini in [mock_ws_gemini, mock_review_gemini]:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = json.dumps(
                {
                    "analysis": "Codebase overview for new developer",
                    "entry_points": ["src/app.py"],
                    "key_modules": ["src/config.py", "src/app.py"],
                    "recommendations": ["Start with app.py", "Review config.py for settings"],
                }
            )
            mock_model.generate_content.return_value = mock_response
            mock_gemini.return_value = mock_model

        from app.mcp_servers.gemini_code_reviewer import review_code
        from app.mcp_servers.gemini_workspace_analyzer import analyze_workspace
        from app.mcp_servers.gemini_workspace_analyzer import get_project_overview
        from app.mcp_servers.gemini_workspace_analyzer import search_files

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            self.create_complex_project(project_path)

            # Step 1: Get project overview
            overview = await get_project_overview(project_path=str(project_path))
            assert overview["status"] == "success"

            # Step 2: Find main entry points
            python_files = await search_files(directory=str(project_path), pattern="*.py")
            assert python_files["status"] == "success"

            # Step 3: Analyze workspace structure
            workspace_analysis = await analyze_workspace(
                project_path=str(project_path),
                analysis_depth="standard",
                focus_areas="structure,entry_points",
            )
            assert workspace_analysis["status"] == "success"

            # Step 4: Review key files
            app_file = project_path / "src" / "app.py"
            if app_file.exists():
                code_review = await review_code(
                    file_path=str(app_file),
                    focus_areas="quality,security",
                    severity_threshold="medium",
                )
                assert code_review["status"] == "success"

    @pytest.mark.asyncio
    @patch("app.mcp_servers.gemini_code_reviewer.ensure_model_ready")
    async def test_security_audit_scenario(self, mock_gemini):
        """Test scenario: security team conducting code audit."""
        # Setup mock for security findings
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "security_audit": {
                    "overall_score": 4.5,
                    "critical_findings": 2,
                    "high_findings": 3,
                    "vulnerabilities": [
                        {"type": "sql_injection", "file": "app.py", "severity": "critical"},
                        {"type": "hardcoded_secrets", "file": "app.py", "severity": "high"},
                    ],
                },
                "remediation_plan": [
                    "Immediate: Fix SQL injection in app.py",
                    "Priority: Remove hardcoded secrets",
                    "Follow-up: Implement security scanning in CI/CD",
                ],
            }
        )
        mock_model.generate_content.return_value = mock_response
        mock_gemini.return_value = mock_model

        from app.mcp_servers.gemini_code_reviewer import comprehensive_analysis
        from app.mcp_servers.gemini_code_reviewer import review_security

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            self.create_complex_project(project_path)

            # Security-focused analysis
            security_scan = await review_security(
                directory=str(project_path), file_patterns="*.py", scan_depth="comprehensive"
            )

            assert security_scan["status"] == "success"
            assert "scan_summary" in security_scan

            # Comprehensive analysis with security focus
            comprehensive_scan = await comprehensive_analysis(
                target_path=str(project_path), analysis_types="security", file_patterns="*.py"
            )

            assert comprehensive_scan["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
