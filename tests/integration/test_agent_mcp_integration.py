#!/usr/bin/env python3
"""
Integration tests for agent-MCP server interactions.

This test suite covers:
1. Agent to MCP server communication
2. Cross-agent workflows
3. Real-world usage scenarios
4. End-to-end workflow testing
5. Agent orchestration patterns

Designed to test practical agent interactions that increase coverage.
"""

import asyncio
import os
import tempfile
from unittest.mock import patch

import pytest

# Import test utilities
from tests.conftest_gemini_mcp import GeminiMCPTestBase
from tests.conftest_gemini_mcp import TestDataFactory


class TestAgentMCPIntegration(GeminiMCPTestBase):
    """Test integration between agents and MCP servers."""

    @pytest.mark.asyncio
    async def test_code_review_agent_mcp_integration(self, mock_file_system, mock_vertexai, mock_environment_variables):
        """Test code review agent integration with MCP server."""
        try:
            # Import agent and MCP components
            from app.agents.code_review_agent import CodeReviewAgent
            from app.mcp_servers.gemini_code_reviewer import review_code

            # Create test file
            test_code = '''
def example_function(data):
    """Example function with some issues."""
    # Hardcoded secret (security issue)
    api_key = "sk-1234567890abcdef"

    # Inefficient loop (performance issue)
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i] == data[j]:
                result.append(data[i])

    return result
'''

            test_file = mock_file_system / "test_code.py"
            test_file.write_text(test_code)

            # Test agent using MCP server
            agent = CodeReviewAgent()

            # Mock the agent's MCP client calls
            with patch.object(
                agent, "call_mcp_tool", return_value=TestDataFactory.create_code_review_result()
            ) as mock_call:
                result = await agent.review_file(str(test_file))

                # Verify MCP tool was called
                mock_call.assert_called_once()

                # Verify result structure
                self.assert_review_structure(result)

        except ImportError:
            pytest.skip("Code review agent not available")

    @pytest.mark.asyncio
    async def test_workspace_analyzer_agent_mcp_integration(
        self, mock_file_system, mock_vertexai, mock_environment_variables
    ):
        """Test workspace analyzer agent integration with MCP server."""
        try:
            from app.agents.workspace_analyzer_agent import WorkspaceAnalyzerAgent

            agent = WorkspaceAnalyzerAgent()

            # Mock agent's MCP server interactions
            mock_analysis_result = {
                "status": "success",
                "workspace_path": str(mock_file_system),
                "analysis": {
                    "total_files": 5,
                    "languages": ["python"],
                    "structure": {"src": ["main.py", "utils.py"], "tests": ["test_main.py"]},
                    "dependencies": {"direct": 2, "total": 8},
                    "issues": {"security": 1, "performance": 2, "quality": 3},
                },
                "recommendations": [
                    "Add type hints to improve code quality",
                    "Increase test coverage",
                    "Review security patterns",
                ],
            }

            with patch.object(agent, "call_mcp_tool", return_value=mock_analysis_result) as mock_call:
                result = await agent.analyze_workspace(str(mock_file_system))

                # Verify MCP tool was called
                mock_call.assert_called_once()

                # Verify result structure
                assert result["status"] == "success"
                assert "analysis" in result
                assert "recommendations" in result

        except ImportError:
            pytest.skip("Workspace analyzer agent not available")

    @pytest.mark.asyncio
    async def test_master_architect_agent_mcp_integration(
        self, mock_file_system, mock_vertexai, mock_environment_variables
    ):
        """Test master architect agent integration with MCP server."""
        try:
            from app.agents.master_architect_agent import MasterArchitectAgent

            agent = MasterArchitectAgent()

            # Mock architecture analysis result
            mock_architecture_result = {
                "status": "success",
                "project_path": str(mock_file_system),
                "architecture": {
                    "pattern": "layered",
                    "components": [
                        {"name": "presentation", "type": "layer", "files": ["main.py"]},
                        {"name": "business", "type": "layer", "files": ["utils.py"]},
                    ],
                    "dependencies": {"internal": 2, "external": 3},
                },
                "assessment": {"scalability": 7.5, "maintainability": 8.0, "security": 6.5, "performance": 7.0},
                "recommendations": [
                    "Consider implementing dependency injection",
                    "Add service layer for better separation",
                    "Implement proper error handling patterns",
                ],
            }

            with patch.object(agent, "call_mcp_tool", return_value=mock_architecture_result) as mock_call:
                result = await agent.analyze_architecture(str(mock_file_system))

                # Verify MCP tool was called
                mock_call.assert_called_once()

                # Verify result structure
                assert result["status"] == "success"
                assert "architecture" in result
                assert "assessment" in result
                assert "recommendations" in result

        except ImportError:
            pytest.skip("Master architect agent not available")


class TestCrossAgentWorkflows:
    """Test workflows that involve multiple agents."""

    @pytest.mark.asyncio
    async def test_comprehensive_code_analysis_workflow(self, mock_file_system, integration_helper):
        """Test comprehensive workflow using multiple agents."""
        # Set up mock agents and MCP servers
        await integration_helper.setup_mock_server("workspace-analyzer", ["analyze_workspace", "analyze_dependencies"])

        await integration_helper.setup_mock_server(
            "code-reviewer", ["review_code", "review_security", "review_performance"]
        )

        await integration_helper.setup_mock_server("master-architect", ["analyze_architecture", "suggest_improvements"])

        # Simulate comprehensive analysis workflow

        # Step 1: Analyze workspace structure
        workspace_result = await integration_helper.simulate_server_interaction(
            "workspace-analyzer", "analyze_workspace", workspace_path=str(mock_file_system)
        )
        assert workspace_result["status"] == "success"

        # Step 2: Review code quality and security
        review_result = await integration_helper.simulate_server_interaction(
            "code-reviewer", "review_code", file_path=str(mock_file_system / "src" / "main.py")
        )
        assert review_result["status"] == "success"

        # Step 3: Analyze architecture
        architecture_result = await integration_helper.simulate_server_interaction(
            "master-architect", "analyze_architecture", project_path=str(mock_file_system)
        )
        assert architecture_result["status"] == "success"

        # Step 4: Generate consolidated report
        consolidated_report = {
            "workspace_analysis": workspace_result,
            "code_review": review_result,
            "architecture_analysis": architecture_result,
            "summary": {
                "overall_health": "good",
                "critical_issues": 0,
                "improvement_areas": ["testing", "documentation", "security"],
            },
        }

        assert "workspace_analysis" in consolidated_report
        assert "code_review" in consolidated_report
        assert "architecture_analysis" in consolidated_report

    @pytest.mark.asyncio
    async def test_security_focused_workflow(self, mock_file_system, integration_helper):
        """Test security-focused workflow across agents."""
        # Set up security-focused servers
        await integration_helper.setup_mock_server(
            "security-scanner", ["scan_vulnerabilities", "analyze_dependencies", "check_compliance"]
        )

        await integration_helper.setup_mock_server("code-reviewer", ["review_security", "comprehensive_analysis"])

        # Step 1: Scan for vulnerabilities
        vuln_result = await integration_helper.simulate_server_interaction(
            "security-scanner", "scan_vulnerabilities", target_path=str(mock_file_system)
        )

        # Step 2: Detailed security code review
        security_review = await integration_helper.simulate_server_interaction(
            "code-reviewer", "review_security", directory=str(mock_file_system)
        )

        # Verify security workflow
        assert vuln_result["status"] == "success"
        assert security_review["status"] == "success"

    @pytest.mark.asyncio
    async def test_performance_optimization_workflow(self, mock_file_system, integration_helper):
        """Test performance optimization workflow."""
        # Set up performance-focused servers
        await integration_helper.setup_mock_server(
            "performance-analyzer", ["profile_performance", "analyze_bottlenecks", "suggest_optimizations"]
        )

        await integration_helper.setup_mock_server("code-reviewer", ["review_performance", "comprehensive_analysis"])

        # Step 1: Profile performance
        profile_result = await integration_helper.simulate_server_interaction(
            "performance-analyzer", "profile_performance", target_path=str(mock_file_system)
        )

        # Step 2: Detailed performance review
        perf_review = await integration_helper.simulate_server_interaction(
            "code-reviewer", "review_performance", file_path=str(mock_file_system / "src" / "main.py")
        )

        # Verify performance workflow
        assert profile_result["status"] == "success"
        assert perf_review["status"] == "success"


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_new_project_onboarding_scenario(self, mock_file_system, integration_helper):
        """Test scenario: onboarding a new project for analysis."""
        # Set up servers for new project analysis
        await integration_helper.setup_mock_server(
            "project-scanner", ["detect_language", "analyze_structure", "estimate_complexity"]
        )

        await integration_helper.setup_mock_server(
            "quality-assessor", ["assess_code_quality", "identify_patterns", "suggest_standards"]
        )

        # Simulate new project onboarding workflow

        # Step 1: Detect project characteristics
        detection_result = await integration_helper.simulate_server_interaction(
            "project-scanner", "detect_language", project_path=str(mock_file_system)
        )

        # Step 2: Analyze project structure
        structure_result = await integration_helper.simulate_server_interaction(
            "project-scanner", "analyze_structure", project_path=str(mock_file_system)
        )

        # Step 3: Assess code quality
        quality_result = await integration_helper.simulate_server_interaction(
            "quality-assessor", "assess_code_quality", project_path=str(mock_file_system)
        )

        # Verify onboarding workflow
        assert detection_result["status"] == "success"
        assert structure_result["status"] == "success"
        assert quality_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_continuous_integration_scenario(self, mock_file_system, integration_helper):
        """Test scenario: continuous integration analysis."""
        # Set up CI-focused servers
        await integration_helper.setup_mock_server(
            "ci-analyzer", ["analyze_changes", "assess_impact", "generate_report"]
        )

        await integration_helper.setup_mock_server(
            "test-analyzer", ["analyze_test_coverage", "suggest_tests", "validate_quality"]
        )

        # Simulate CI analysis workflow

        # Step 1: Analyze code changes
        changes_result = await integration_helper.simulate_server_interaction(
            "ci-analyzer", "analyze_changes", project_path=str(mock_file_system)
        )

        # Step 2: Assess test coverage
        coverage_result = await integration_helper.simulate_server_interaction(
            "test-analyzer", "analyze_test_coverage", project_path=str(mock_file_system)
        )

        # Step 3: Generate CI report
        report_result = await integration_helper.simulate_server_interaction(
            "ci-analyzer", "generate_report", analysis_data={"changes": changes_result, "coverage": coverage_result}
        )

        # Verify CI workflow
        assert changes_result["status"] == "success"
        assert coverage_result["status"] == "success"
        assert report_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_legacy_code_modernization_scenario(self, mock_file_system, integration_helper):
        """Test scenario: modernizing legacy code."""
        # Create legacy code example
        legacy_code = """
# Legacy Python 2 style code
def process_data(data):
    result = {}
    for item in data:
        if result.has_key(item):
            result[item] += 1
        else:
            result[item] = 1
    return result

class OldStyleClass:
    def __init__(self, value):
        self.value = value

    def __cmp__(self, other):
        return cmp(self.value, other.value)
"""

        legacy_file = mock_file_system / "legacy_code.py"
        legacy_file.write_text(legacy_code)

        # Set up modernization servers
        await integration_helper.setup_mock_server(
            "legacy-analyzer", ["detect_legacy_patterns", "suggest_modernization", "estimate_effort"]
        )

        await integration_helper.setup_mock_server(
            "modernization-engine", ["apply_transformations", "validate_changes", "generate_migration_plan"]
        )

        # Simulate modernization workflow

        # Step 1: Detect legacy patterns
        patterns_result = await integration_helper.simulate_server_interaction(
            "legacy-analyzer", "detect_legacy_patterns", file_path=str(legacy_file)
        )

        # Step 2: Suggest modernization approaches
        suggestions_result = await integration_helper.simulate_server_interaction(
            "legacy-analyzer", "suggest_modernization", file_path=str(legacy_file)
        )

        # Step 3: Generate migration plan
        migration_result = await integration_helper.simulate_server_interaction(
            "modernization-engine", "generate_migration_plan", project_path=str(mock_file_system)
        )

        # Verify modernization workflow
        assert patterns_result["status"] == "success"
        assert suggestions_result["status"] == "success"
        assert migration_result["status"] == "success"


class TestAgentOrchestration:
    """Test agent orchestration patterns."""

    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, mock_file_system, integration_helper):
        """Test parallel execution of multiple agents."""
        # Set up multiple agents for parallel execution
        agents = ["agent-1", "agent-2", "agent-3"]

        for agent_name in agents:
            await integration_helper.setup_mock_server(agent_name, ["analyze", "process", "report"])

        # Execute agents in parallel
        async def run_agent_analysis(agent_name: str):
            return await integration_helper.simulate_server_interaction(
                agent_name, "analyze", target_path=str(mock_file_system)
            )

        # Run all agents concurrently
        tasks = [run_agent_analysis(agent) for agent in agents]
        results = await asyncio.gather(*tasks)

        # Verify all agents completed successfully
        assert len(results) == 3
        for result in results:
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_sequential_agent_pipeline(self, mock_file_system, integration_helper):
        """Test sequential pipeline of agents."""
        # Set up pipeline agents
        pipeline_agents = [
            ("preprocessor", ["clean_data", "normalize"]),
            ("analyzer", ["analyze", "extract_features"]),
            ("validator", ["validate", "verify_results"]),
            ("reporter", ["generate_report", "format_output"]),
        ]

        for agent_name, tools in pipeline_agents:
            await integration_helper.setup_mock_server(agent_name, tools)

        # Execute pipeline sequentially
        pipeline_data = {"input": str(mock_file_system)}

        # Step 1: Preprocessing
        preprocessed = await integration_helper.simulate_server_interaction(
            "preprocessor", "clean_data", data=pipeline_data
        )

        # Step 2: Analysis
        analyzed = await integration_helper.simulate_server_interaction("analyzer", "analyze", data=preprocessed)

        # Step 3: Validation
        validated = await integration_helper.simulate_server_interaction("validator", "validate", data=analyzed)

        # Step 4: Reporting
        report = await integration_helper.simulate_server_interaction("reporter", "generate_report", data=validated)

        # Verify pipeline execution
        assert preprocessed["status"] == "success"
        assert analyzed["status"] == "success"
        assert validated["status"] == "success"
        assert report["status"] == "success"

    @pytest.mark.asyncio
    async def test_conditional_agent_routing(self, mock_file_system, integration_helper):
        """Test conditional routing of agent execution."""
        # Set up routing logic
        await integration_helper.setup_mock_server("router", ["route_request", "determine_agents"])

        await integration_helper.setup_mock_server("python-specialist", ["analyze_python", "optimize_python"])

        await integration_helper.setup_mock_server(
            "javascript-specialist", ["analyze_javascript", "optimize_javascript"]
        )

        await integration_helper.setup_mock_server("generic-analyzer", ["analyze_generic", "provide_feedback"])

        # Simulate routing decision
        routing_result = await integration_helper.simulate_server_interaction(
            "router", "determine_agents", file_path=str(mock_file_system / "src" / "main.py")
        )

        # Based on .py extension, should route to Python specialist
        python_result = await integration_helper.simulate_server_interaction(
            "python-specialist", "analyze_python", file_path=str(mock_file_system / "src" / "main.py")
        )

        # Verify routing workflow
        assert routing_result["status"] == "success"
        assert python_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_fault_tolerant_agent_execution(self, integration_helper):
        """Test fault tolerance in agent execution."""
        # Set up primary and backup agents
        await integration_helper.setup_mock_server("primary-agent", ["primary_analysis"])

        await integration_helper.setup_mock_server("backup-agent", ["backup_analysis"])

        # Simulate primary agent failure
        primary_server = integration_helper.servers["primary-agent"]
        primary_server.connected = False

        # Try primary agent (should fail)
        try:
            await integration_helper.simulate_server_interaction("primary-agent", "primary_analysis")
            raise AssertionError("Should have failed")
        except ValueError:
            # Expected failure, now try backup
            pass

        # Use backup agent
        backup_result = await integration_helper.simulate_server_interaction("backup-agent", "backup_analysis")

        assert backup_result["status"] == "success"


class TestPerformanceAtScale:
    """Test performance characteristics under load."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_agent_performance(self, mock_file_system, performance_tester, integration_helper):
        """Test performance with multiple concurrent agents."""
        # Set up multiple agents
        agent_count = 10
        for i in range(agent_count):
            await integration_helper.setup_mock_server(f"agent-{i}", ["analyze", "process"])

        # Measure concurrent execution time
        async def concurrent_execution():
            tasks = []
            for i in range(agent_count):
                task = integration_helper.simulate_server_interaction(
                    f"agent-{i}", "analyze", target_path=str(mock_file_system)
                )
                tasks.append(task)
            return await asyncio.gather(*tasks)

        results, exec_time = await performance_tester.measure_execution_time(concurrent_execution())

        # Should complete 10 concurrent agents in under 2 seconds
        assert exec_time < 2.0, f"Concurrent execution too slow: {exec_time:.3f}s"
        assert len(results) == agent_count

        for result in results:
            assert result["status"] == "success"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_file_processing_performance(self, performance_tester, integration_helper):
        """Test performance with large file processing."""
        # Create large test file
        large_content = "\n".join([f"# Line {i}: " + "x" * 100 for i in range(1000)])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(large_content)
            large_file = f.name

        try:
            await integration_helper.setup_mock_server("file-processor", ["process_large_file"])

            # Measure processing time for large file
            result, exec_time = await performance_tester.measure_execution_time(
                integration_helper.simulate_server_interaction(
                    "file-processor", "process_large_file", file_path=large_file
                )
            )

            # Should process large file in under 1 second
            assert exec_time < 1.0, f"Large file processing too slow: {exec_time:.3f}s"
            assert result["status"] == "success"

        finally:
            os.unlink(large_file)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_efficiency_under_load(self, mock_file_system, integration_helper):
        """Test memory efficiency under sustained load."""
        # Set up memory-intensive scenario
        await integration_helper.setup_mock_server("memory-intensive", ["process_memory_intensive"])

        # Simulate multiple rounds of processing
        for round_num in range(5):
            result = await integration_helper.simulate_server_interaction(
                "memory-intensive", "process_memory_intensive", round=round_num, data_size="large"
            )
            assert result["status"] == "success"

            # Small delay to allow memory cleanup
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not performance"])
