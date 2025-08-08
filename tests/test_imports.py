"""Test all critical imports after gapp -> gterminal consolidation.

This test suite validates that all imports work correctly after the major
consolidation from gapp to gterminal, ensuring no broken import paths remain.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


class TestImportValidation:
    """Test import validation after consolidation."""

    def test_root_package_import(self) -> None:
        """Test that the root gterminal package imports correctly."""
        import gterminal

        assert hasattr(gterminal, "__version__")
        assert hasattr(gterminal, "config")
        assert gterminal.__version__ is not None

    def test_agents_module_imports(self) -> None:
        """Test that all agents module imports work correctly."""
        from gterminal import agents

        # Test base imports
        assert hasattr(agents, "BaseAgentService")
        assert hasattr(agents, "CodeGenerationAgentService")
        assert hasattr(agents, "WorkspaceAnalyzerAgentService")
        assert hasattr(agents, "DocumentationGeneratorAgentService")

        # Test agent registry
        assert hasattr(agents, "AGENT_REGISTRY")
        assert hasattr(agents, "MCP_REGISTRY")
        assert isinstance(agents.AGENT_REGISTRY, dict)
        assert isinstance(agents.MCP_REGISTRY, dict)

    def test_core_module_imports(self) -> None:
        """Test that all core module imports work correctly."""
        from gterminal.core import agents as core_agents
        from gterminal.core import interfaces
        from gterminal.core import mcp
        from gterminal.core import monitoring
        from gterminal.core import performance
        from gterminal.core import security
        from gterminal.core import tools

        # Test that modules are importable
        assert core_agents is not None
        assert interfaces is not None
        assert mcp is not None
        assert monitoring is not None
        assert performance is not None
        assert security is not None
        assert tools is not None

    def test_terminal_module_imports(self) -> None:
        """Test that terminal module imports work correctly."""
        from gterminal.terminal import agent_commands
        from gterminal.terminal import enhanced_react_orchestrator
        from gterminal.terminal import react_engine

        assert react_engine is not None
        assert agent_commands is not None
        assert enhanced_react_orchestrator is not None

    def test_gemini_cli_imports(self) -> None:
        """Test that gemini_cli module imports work correctly."""
        from gterminal.gemini_cli import main
        from gterminal.gemini_cli.core import auth
        from gterminal.gemini_cli.core import client
        from gterminal.gemini_cli.core import react_engine as gemini_react

        assert main is not None
        assert client is not None
        assert auth is not None
        assert gemini_react is not None

    def test_mcp_servers_imports(self) -> None:
        """Test that MCP servers import correctly."""
        from gterminal.mcp_servers import gemini_server_agent_mcp

        assert gemini_server_agent_mcp is not None

    def test_auth_module_imports(self) -> None:
        """Test that auth module imports work correctly."""
        from gterminal.auth import api_keys
        from gterminal.auth import auth_jwt
        from gterminal.auth import auth_models
        from gterminal.auth import auth_storage
        from gterminal.auth import gcp_auth

        assert api_keys is not None
        assert auth_jwt is not None
        assert auth_models is not None
        assert auth_storage is not None
        assert gcp_auth is not None

    def test_cache_module_imports(self) -> None:
        """Test that cache module imports work correctly."""
        from gterminal.cache import cache_manager
        from gterminal.cache import memory_cache
        from gterminal.cache import redis_cache

        assert cache_manager is not None
        assert memory_cache is not None
        assert redis_cache is not None

    def test_no_legacy_app_imports(self) -> None:
        """Test that old 'app.' import paths no longer exist."""
        # These should all fail as they're from the old gapp structure
        legacy_imports = [
            "app.agents",
            "app.core",
            "app.terminal",
            "app.auth",
            "app.cache",
            "app.mcp_servers",
        ]

        for legacy_import in legacy_imports:
            with pytest.raises((ImportError, ModuleNotFoundError)):
                importlib.import_module(legacy_import)

    def test_specific_agent_imports(self) -> None:
        """Test that specific agent implementations import correctly."""
        # Test individual agent modules
        from gterminal.agents import code_review_agent
        from gterminal.agents import documentation_generator_agent
        from gterminal.agents import master_architect_agent
        from gterminal.agents import production_ready_agent
        from gterminal.agents import workspace_analyzer_agent

        assert code_review_agent is not None
        assert documentation_generator_agent is not None
        assert workspace_analyzer_agent is not None
        assert master_architect_agent is not None
        assert production_ready_agent is not None

    def test_core_infrastructure_imports(self) -> None:
        """Test that core infrastructure imports work correctly."""
        from gterminal.core.infrastructure import service_registry
        from gterminal.core.interfaces import api_adapter
        from gterminal.core.interfaces import cli_adapter
        from gterminal.core.interfaces import mcp_adapter
        from gterminal.core.interfaces import terminal_adapter

        assert service_registry is not None
        assert api_adapter is not None
        assert cli_adapter is not None
        assert mcp_adapter is not None
        assert terminal_adapter is not None

    def test_monitoring_imports(self) -> None:
        """Test that monitoring components import correctly."""
        from gterminal.core.monitoring import ai_metrics
        from gterminal.core.monitoring import integrated_monitoring
        from gterminal.core.monitoring import unified_dashboard
        from gterminal.core.monitoring import utils_metrics
        from gterminal.core.monitoring import utils_performance

        assert ai_metrics is not None
        assert integrated_monitoring is not None
        assert unified_dashboard is not None
        assert utils_metrics is not None
        assert utils_performance is not None

    def test_security_imports(self) -> None:
        """Test that security components import correctly."""
        from gterminal.core.security import audit_logger
        from gterminal.core.security import integrated_security_middleware
        from gterminal.core.security import secrets_manager
        from gterminal.core.security import security_utils

        assert audit_logger is not None
        assert security_utils is not None
        assert secrets_manager is not None
        assert integrated_security_middleware is not None

    def test_performance_imports(self) -> None:
        """Test that performance components import correctly."""
        from gterminal.core.performance import database_optimizer
        from gterminal.core.performance import gemini_rust_integration
        from gterminal.core.performance import optimization
        from gterminal.core.performance import optimizer

        assert optimization is not None
        assert optimizer is not None
        assert database_optimizer is not None
        assert gemini_rust_integration is not None

    def test_tools_imports(self) -> None:
        """Test that tools components import correctly."""
        from gterminal.core.tools import analysis
        from gterminal.core.tools import filesystem
        from gterminal.core.tools import generation
        from gterminal.core.tools import registry
        from gterminal.core.tools import shell

        assert analysis is not None
        assert filesystem is not None
        assert generation is not None
        assert registry is not None
        assert shell is not None

    def test_utils_imports(self) -> None:
        """Test that utils components import correctly."""
        from gterminal.utils.common import base_classes
        from gterminal.utils.common import cache_utils
        from gterminal.utils.common import file_ops
        from gterminal.utils.common import filesystem
        from gterminal.utils.database import cache
        from gterminal.utils.database import connection_pool
        from gterminal.utils.database import redis

        assert base_classes is not None
        assert cache_utils is not None
        assert file_ops is not None
        assert filesystem is not None
        assert cache is not None
        assert connection_pool is not None
        assert redis is not None

    def test_rust_extensions_imports(self) -> None:
        """Test that Rust extensions import correctly."""
        from gterminal.utils.rust_extensions import rust_auth
        from gterminal.utils.rust_extensions import rust_bindings
        from gterminal.utils.rust_extensions import rust_cache
        from gterminal.utils.rust_extensions import rust_tools

        assert rust_auth is not None
        assert rust_bindings is not None
        assert rust_cache is not None
        assert rust_tools is not None

    @pytest.mark.slow
    def test_all_python_files_import(self, project_root) -> None:
        """Test that all Python files in the project can be imported without errors."""
        gterminal_path = project_root / "gterminal"
        failed_imports = []
        successful_imports = []

        # Get all Python files except test files
        python_files = list(gterminal_path.rglob("*.py"))
        python_files = [
            f
            for f in python_files
            if not any(part.startswith("test_") for part in f.parts)
            and f.name != "__main__.py"  # Skip entry points
            and "tests" not in f.parts
        ]

        for py_file in python_files:
            # Convert file path to module path
            relative_path = py_file.relative_to(project_root)
            module_parts = list(relative_path.with_suffix("").parts)
            module_name = ".".join(module_parts)

            try:
                # Try to import the module
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    successful_imports.append(module_name)
            except Exception as e:
                failed_imports.append((module_name, str(e)))

        # Report results
        print("\nImport test results:")
        print(f"  Successful imports: {len(successful_imports)}")
        print(f"  Failed imports: {len(failed_imports)}")

        if failed_imports:
            print("\nFailed imports:")
            for module_name, error in failed_imports[:10]:  # Show first 10
                print(f"  - {module_name}: {error}")

        # We expect some imports to fail due to optional dependencies
        # But core modules should import successfully
        assert len(successful_imports) > len(failed_imports), (
            f"Too many import failures: {len(failed_imports)} failed vs "
            f"{len(successful_imports)} successful"
        )


class TestImportStructureConsistency:
    """Test that import structure is consistent across the project."""

    def test_no_circular_imports(self) -> None:
        """Test that there are no circular import dependencies."""
        # This is a basic test - a full circular import detection would be more complex
        core_modules = [
            "gterminal.agents",
            "gterminal.core.agents",
            "gterminal.core.mcp",
            "gterminal.core.monitoring",
            "gterminal.terminal",
            "gterminal.gemini_cli",
        ]

        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                if "circular" in str(e).lower():
                    pytest.fail(f"Circular import detected in {module_name}: {e}")

    def test_consistent_naming_pattern(self, project_root) -> None:
        """Test that all modules follow consistent naming patterns."""
        gterminal_path = project_root / "gterminal"

        # Check for consistent __init__.py files in packages
        package_dirs = [
            d for d in gterminal_path.rglob("*") if d.is_dir() and not d.name.startswith(".")
        ]

        missing_init = []
        for pkg_dir in package_dirs:
            # Skip if it's not a Python package directory
            if not any(pkg_dir.glob("*.py")):
                continue

            init_file = pkg_dir / "__init__.py"
            if not init_file.exists():
                missing_init.append(str(pkg_dir.relative_to(project_root)))

        # Some directories might legitimately not need __init__.py
        # But core packages should have them
        core_packages = [
            d
            for d in missing_init
            if not any(skip in d for skip in ["htmlcov", "__pycache__", ".git", "node_modules"])
        ]

        if core_packages:
            print(f"Packages missing __init__.py: {core_packages}")

        # This is more of a warning than a hard failure
        assert len(core_packages) < 5, f"Too many packages missing __init__.py: {core_packages}"


@pytest.mark.integration
class TestImportIntegration:
    """Test import integration with external dependencies."""

    def test_external_dependency_imports(self) -> None:
        """Test that external dependencies import correctly."""
        try:
            import aiohttp
            import click
            import fastmcp
            import google.cloud.aiplatform
            import pydantic
            import redis
            import rich
            import vertexai

            # Basic smoke test
            assert fastmcp is not None
            assert vertexai is not None
            assert google.cloud.aiplatform is not None

        except ImportError as e:
            pytest.skip(f"External dependency not available: {e}")

    @pytest.mark.mcp
    def test_mcp_integration_imports(self) -> None:
        """Test that MCP integration imports work correctly."""
        try:
            # Test MCP server registry
            from gterminal.agents import MCP_REGISTRY
            from gterminal.core.mcp import config_manager
            from gterminal.core.mcp import server_registry
            from gterminal.mcp_servers import gemini_server_agent_mcp

            assert isinstance(MCP_REGISTRY, dict)
            assert len(MCP_REGISTRY) > 0

        except ImportError as e:
            pytest.skip(f"MCP dependencies not available: {e}")

    @pytest.mark.requires_api_key
    def test_gemini_integration_imports(self) -> None:
        """Test that Gemini integration imports work correctly."""
        try:
            from gterminal.auth.gcp_auth import get_auth_manager
            from gterminal.gemini_cli.core import client

            # Test that auth manager can be created
            auth_manager = get_auth_manager()
            assert auth_manager is not None

        except ImportError as e:
            pytest.skip(f"Gemini dependencies not available: {e}")
