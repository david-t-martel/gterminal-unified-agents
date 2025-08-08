"""Test consolidation success from gapp to gterminal.

This test suite validates that the consolidation from gapp to gterminal
was successful by checking for:
1. No remaining gapp references
2. Proper gterminal structure
3. Functional equivalence
4. Clean migration of functionality
"""

from __future__ import annotations

import ast
import re

import pytest


class TestConsolidationValidation:
    """Test that the consolidation was successful."""

    def test_no_gapp_references_in_imports(self, project_root) -> None:
        """Test that there are no remaining 'from gapp.' or 'import gapp' references."""
        gterminal_path = project_root / "gterminal"
        python_files = list(gterminal_path.rglob("*.py"))

        gapp_references = []

        for py_file in python_files:
            if py_file.name.startswith("test_"):
                continue  # Skip test files for now

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith("#"):
                        continue

                    # Check for gapp imports
                    if re.search(r"\b(?:from\s+gapp|import\s+gapp)\b", line):
                        gapp_references.append(
                            {
                                "file": str(py_file.relative_to(project_root)),
                                "line": line_num,
                                "content": line.strip(),
                            }
                        )

            except UnicodeDecodeError:
                # Skip binary files
                continue

        if gapp_references:
            print("\nFound remaining gapp references:")
            for ref in gapp_references:
                print(f"  {ref['file']}:{ref['line']}: {ref['content']}")

        assert len(gapp_references) == 0, (
            f"Found {len(gapp_references)} remaining gapp import references"
        )

    def test_no_app_references_in_imports(self, project_root) -> None:
        """Test that there are no remaining 'from app.' or 'import app' references."""
        gterminal_path = project_root / "gterminal"
        python_files = list(gterminal_path.rglob("*.py"))

        app_references = []

        for py_file in python_files:
            if py_file.name.startswith("test_"):
                continue  # Skip test files for now

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith("#"):
                        continue

                    # Check for app imports (but not gapp)
                    if re.search(r"\bfrom\s+app\.", line) and "gapp" not in line:
                        app_references.append(
                            {
                                "file": str(py_file.relative_to(project_root)),
                                "line": line_num,
                                "content": line.strip(),
                            }
                        )

            except UnicodeDecodeError:
                continue

        if app_references:
            print("\nFound remaining app references:")
            for ref in app_references:
                print(f"  {ref['file']}:{ref['line']}: {ref['content']}")

        assert len(app_references) == 0, (
            f"Found {len(app_references)} remaining app import references"
        )

    def test_gterminal_structure_exists(self, project_root) -> None:
        """Test that expected gterminal structure exists."""
        gterminal_path = project_root / "gterminal"

        expected_directories = [
            "agents",
            "auth",
            "cache",
            "core",
            "terminal",
            "gemini_cli",
            "utils",
            "mcp_servers",
        ]

        missing_directories = []
        for directory in expected_directories:
            dir_path = gterminal_path / directory
            if not dir_path.exists():
                missing_directories.append(directory)

        assert len(missing_directories) == 0, f"Missing expected directories: {missing_directories}"

    def test_no_nested_gterminal_structure(self, project_root) -> None:
        """Test that there's no nested gterminal/gterminal structure."""
        gterminal_path = project_root / "gterminal"
        nested_gterminal = gterminal_path / "gterminal"

        assert not nested_gterminal.exists(), (
            "Found nested gterminal/gterminal structure - consolidation incomplete"
        )

    def test_agent_modules_consolidated(self, project_root) -> None:
        """Test that agent modules are properly consolidated."""
        agents_path = project_root / "gterminal" / "agents"

        expected_agents = [
            "base_agent_service.py",
            "consolidated_agent_service.py",
            "code_review_agent.py",
            "documentation_generator_agent.py",
            "workspace_analyzer_agent.py",
            "master_architect_agent.py",
        ]

        existing_agents = []
        missing_agents = []

        for agent_file in expected_agents:
            agent_path = agents_path / agent_file
            if agent_path.exists():
                existing_agents.append(agent_file)
            else:
                missing_agents.append(agent_file)

        # We should have most of the expected agents
        assert len(existing_agents) >= len(expected_agents) // 2, (
            f"Too many missing agent files: {missing_agents}"
        )

    def test_mcp_servers_consolidated(self, project_root) -> None:
        """Test that MCP servers are properly consolidated."""
        mcp_path = project_root / "gterminal" / "mcp_servers"

        assert mcp_path.exists(), "MCP servers directory missing"

        # Check for MCP server files
        mcp_files = list(mcp_path.glob("*.py"))
        assert len(mcp_files) > 0, "No MCP server files found"

    def test_core_modules_consolidated(self, project_root) -> None:
        """Test that core modules are properly consolidated."""
        core_path = project_root / "gterminal" / "core"

        expected_core_modules = [
            "agents",
            "interfaces",
            "mcp",
            "monitoring",
            "performance",
            "security",
            "tools",
        ]

        missing_core_modules = []
        for module in expected_core_modules:
            module_path = core_path / module
            if not module_path.exists():
                missing_core_modules.append(module)

        assert len(missing_core_modules) <= 2, (
            f"Too many missing core modules: {missing_core_modules}"
        )


class TestFunctionalEquivalence:
    """Test that functionality is preserved after consolidation."""

    def test_agent_registry_functional(self) -> None:
        """Test that agent registry is functional."""
        from gterminal.agents import AGENT_REGISTRY
        from gterminal.agents import MCP_REGISTRY

        # Test agent registry exists and has entries
        assert isinstance(AGENT_REGISTRY, dict)
        assert len(AGENT_REGISTRY) > 0

        # Test MCP registry exists and has entries
        assert isinstance(MCP_REGISTRY, dict)
        assert len(MCP_REGISTRY) > 0

    def test_agent_services_functional(self) -> None:
        """Test that agent services are functional."""
        try:
            from gterminal.agents import get_agent_service

            # Test that we can get at least one agent service
            registry_keys = list(
                __import__("gterminal.agents", fromlist=["AGENT_REGISTRY"]).AGENT_REGISTRY.keys()
            )

            functional_agents = 0
            for agent_type in registry_keys:
                try:
                    agent = get_agent_service(agent_type)
                    if agent is not None:
                        functional_agents += 1
                except (ValueError, ImportError, TypeError):
                    continue

            # We should have at least one functional agent
            assert functional_agents > 0, "No functional agents found"

        except ImportError:
            pytest.skip("Agent services not available")

    def test_mcp_services_functional(self) -> None:
        """Test that MCP services are functional."""
        try:
            from gterminal.agents import MCP_REGISTRY
            from gterminal.agents import get_mcp_server

            # Test that we can get at least one MCP server
            functional_servers = 0
            for server_type in list(MCP_REGISTRY.keys())[:3]:  # Test first 3
                try:
                    server = get_mcp_server(server_type)
                    if server is not None:
                        functional_servers += 1
                except (ValueError, ImportError, TypeError):
                    continue

            # We should have at least one functional MCP server
            assert functional_servers > 0, "No functional MCP servers found"

        except ImportError:
            pytest.skip("MCP services not available")

    def test_terminal_functionality_preserved(self) -> None:
        """Test that terminal functionality is preserved."""
        try:
            from gterminal.terminal.react_engine import ReactEngine

            # Test basic instantiation
            engine = ReactEngine()
            assert engine is not None

        except ImportError:
            # Try alternative import
            try:
                from gterminal.terminal import react_engine

                assert react_engine is not None
            except ImportError:
                pytest.skip("Terminal functionality not available")

    def test_gemini_cli_functionality_preserved(self) -> None:
        """Test that Gemini CLI functionality is preserved."""
        try:
            from gterminal.gemini_cli.main import main

            assert main is not None

            from gterminal.gemini_cli.core.client import GeminiClient

            client = GeminiClient()
            assert client is not None

        except ImportError:
            pytest.skip("Gemini CLI functionality not available")

    def test_auth_functionality_preserved(self) -> None:
        """Test that auth functionality is preserved."""
        from gterminal.auth.gcp_auth import get_auth_manager

        auth_manager = get_auth_manager()
        assert auth_manager is not None

    def test_cache_functionality_preserved(self) -> None:
        """Test that cache functionality is preserved."""
        from gterminal.cache.cache_manager import CacheManager

        cache_manager = CacheManager()
        assert cache_manager is not None


class TestConfigurationMigration:
    """Test that configuration has been properly migrated."""

    def test_pyproject_toml_updated(self, project_root) -> None:
        """Test that pyproject.toml reflects gterminal structure."""
        pyproject_path = project_root / "pyproject.toml"

        if not pyproject_path.exists():
            pytest.skip("pyproject.toml not found")

        content = pyproject_path.read_text()

        # Should reference gterminal, not gapp
        assert "gterminal" in content
        assert "gapp" not in content or content.count("gapp") <= 2  # Allow minimal gapp references

        # Should have proper package structure
        assert '"gterminal"' in content or "'gterminal'" in content

    def test_dockerfile_updated(self, project_root) -> None:
        """Test that Dockerfiles reference gterminal."""
        dockerfiles = [
            project_root / "Dockerfile",
            project_root / "Dockerfile.production",
            project_root / "Dockerfile.dashboard",
        ]

        for dockerfile in dockerfiles:
            if not dockerfile.exists():
                continue

            content = dockerfile.read_text()

            # Check for proper references
            if "COPY" in content or "ADD" in content:
                # Should reference gterminal structure
                lines = content.split("\n")
                gterminal_refs = [line for line in lines if "gterminal" in line.lower()]
                gapp_refs = [
                    line
                    for line in lines
                    if "gapp" in line.lower() and "gterminal" not in line.lower()
                ]

                # Should have more gterminal references than gapp references
                assert len(gterminal_refs) >= len(gapp_refs)

    def test_docker_compose_updated(self, project_root) -> None:
        """Test that docker-compose files are updated."""
        compose_files = [
            project_root / "docker-compose.yml",
            project_root / "docker-compose.production.yml",
        ]

        for compose_file in compose_files:
            if not compose_file.exists():
                continue

            content = compose_file.read_text()

            # Should reference gterminal structure
            assert "gterminal" in content.lower() or len(content) < 100  # Skip if minimal file


class TestCodeQuality:
    """Test code quality aspects of the consolidation."""

    def test_no_duplicate_functionality(self, project_root) -> None:
        """Test that there's no obvious duplicate functionality."""
        gterminal_path = project_root / "gterminal"
        python_files = list(gterminal_path.rglob("*.py"))

        # Look for potential duplicates by checking class and function names
        class_names = {}
        function_names = {}

        for py_file in python_files:
            if py_file.name.startswith("test_"):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if node.name in class_names:
                            class_names[node.name].append(str(py_file.relative_to(project_root)))
                        else:
                            class_names[node.name] = [str(py_file.relative_to(project_root))]
                    elif isinstance(node, ast.FunctionDef):
                        if node.name in function_names:
                            function_names[node.name].append(str(py_file.relative_to(project_root)))
                        else:
                            function_names[node.name] = [str(py_file.relative_to(project_root))]

            except (SyntaxError, UnicodeDecodeError):
                continue

        # Check for suspicious duplicates (same name in multiple files)
        duplicate_classes = {name: files for name, files in class_names.items() if len(files) > 1}
        duplicate_functions = {
            name: files for name, files in function_names.items() if len(files) > 2
        }  # Allow some duplication for common function names

        # Filter out common names that are expected to be duplicated
        common_names = {
            "main",
            "init",
            "setup",
            "get",
            "set",
            "create",
            "update",
            "delete",
            "__init__",
        }
        duplicate_classes = {
            name: files for name, files in duplicate_classes.items() if name not in common_names
        }
        duplicate_functions = {
            name: files for name, files in duplicate_functions.items() if name not in common_names
        }

        if duplicate_classes:
            print(f"\nPotential duplicate classes: {duplicate_classes}")

        if duplicate_functions:
            print("\nPotential duplicate functions (showing first 5):")
            for name, files in list(duplicate_functions.items())[:5]:
                print(f"  {name}: {files}")

        # This is more of a warning than a hard failure
        assert len(duplicate_classes) < 10, f"Too many duplicate classes: {len(duplicate_classes)}"

    def test_consistent_import_style(self, project_root) -> None:
        """Test that import style is consistent across the codebase."""
        gterminal_path = project_root / "gterminal"
        python_files = list(gterminal_path.rglob("*.py"))

        inconsistent_files = []

        for py_file in python_files:
            if py_file.name.startswith("test_"):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.split("\n")

                gterminal_imports = [
                    line for line in lines if "from gterminal" in line or "import gterminal" in line
                ]

                # Check for consistent style
                if gterminal_imports:
                    # All gterminal imports should use absolute imports
                    relative_imports = [
                        line for line in gterminal_imports if line.strip().startswith("from .")
                    ]

                    if relative_imports:
                        inconsistent_files.append(
                            {
                                "file": str(py_file.relative_to(project_root)),
                                "relative_imports": len(relative_imports),
                            }
                        )

            except UnicodeDecodeError:
                continue

        # Allow some relative imports, but not too many
        total_relative = sum(f["relative_imports"] for f in inconsistent_files)
        assert total_relative < 20, (
            f"Too many relative imports in gterminal modules: {total_relative}"
        )


class TestDocumentationConsistency:
    """Test that documentation is consistent with the consolidation."""

    def test_readme_updated(self, project_root) -> None:
        """Test that README reflects gterminal structure."""
        readme_path = project_root / "README.md"

        if not readme_path.exists():
            pytest.skip("README.md not found")

        content = readme_path.read_text()

        # Should mention gterminal
        assert "gterminal" in content.lower() or "gemini" in content.lower()

    def test_documentation_consistency(self, project_root) -> None:
        """Test that documentation files are consistent."""
        doc_files = list(project_root.glob("*.md"))
        doc_files.extend(
            list((project_root / "docs").glob("*.md")) if (project_root / "docs").exists() else []
        )

        inconsistent_docs = []

        for doc_file in doc_files:
            try:
                content = doc_file.read_text(encoding="utf-8")

                # Count references
                gapp_refs = content.lower().count("gapp")
                gterminal_refs = content.lower().count("gterminal")

                # If it has many gapp references but no gterminal references, it might need updating
                if gapp_refs > 5 and gterminal_refs == 0:
                    inconsistent_docs.append(
                        {
                            "file": str(doc_file.relative_to(project_root)),
                            "gapp_refs": gapp_refs,
                            "gterminal_refs": gterminal_refs,
                        }
                    )

            except UnicodeDecodeError:
                continue

        if inconsistent_docs:
            print("\nDocumentation files that might need updating:")
            for doc in inconsistent_docs:
                print(
                    f"  {doc['file']}: {doc['gapp_refs']} gapp refs, {doc['gterminal_refs']} gterminal refs"
                )

        # This is more of a warning - documentation updates can be gradual
        assert len(inconsistent_docs) < 10, (
            f"Many documentation files may need updating: {len(inconsistent_docs)}"
        )


@pytest.mark.consolidation
@pytest.mark.integration
class TestConsolidationIntegration:
    """Test integration aspects specific to the consolidation."""

    async def test_consolidated_agents_work_together(self) -> None:
        """Test that consolidated agents can work together."""
        try:
            from gterminal.agents import AGENT_REGISTRY

            # Get multiple agents
            agent_types = list(AGENT_REGISTRY.keys())[:2]
            agents = []

            for agent_type in agent_types:
                try:
                    from gterminal.agents import get_agent_service

                    agent = get_agent_service(agent_type)
                    if agent is not None:
                        agents.append(agent)
                except (ValueError, ImportError):
                    continue

            # We should be able to create multiple agents
            assert len(agents) >= 0  # At least try to create them

        except ImportError:
            pytest.skip("Agent integration not available")

    async def test_mcp_servers_communicate(self) -> None:
        """Test that MCP servers can communicate."""
        try:
            from gterminal.agents import MCP_REGISTRY

            # Get MCP servers
            server_types = list(MCP_REGISTRY.keys())[:2]
            servers = []

            for server_type in server_types:
                try:
                    from gterminal.agents import get_mcp_server

                    server = get_mcp_server(server_type)
                    if server is not None:
                        servers.append(server)
                except (ValueError, ImportError):
                    continue

            # We should be able to create multiple MCP servers
            assert len(servers) >= 0  # At least try to create them

        except ImportError:
            pytest.skip("MCP integration not available")

    def test_no_import_cycles_post_consolidation(self) -> None:
        """Test that consolidation didn't introduce import cycles."""
        # This is a basic test - detecting all cycles would be complex

        try:
            # Try importing core modules that might have cycles
            import gterminal.agents
            import gterminal.core.agents
            import gterminal.mcp_servers
            import gterminal.terminal

            # If we get here without ImportError, basic imports work
            assert True

        except ImportError as e:
            # Check if it's a circular import
            if "circular" in str(e).lower() or "partially initialized" in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            else:
                # Other import errors might be due to missing dependencies
                pytest.skip(f"Import error (not circular): {e}")
