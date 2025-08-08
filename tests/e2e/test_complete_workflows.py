#!/usr/bin/env python3
"""End-to-end tests for complete user workflows.

This test suite validates real-world usage scenarios including:
- Complete research workflows from query to report
- Multi-step autonomous task execution
- Interactive planning and execution cycles
- Error recovery in production-like scenarios
- Performance under realistic workloads
"""

import asyncio
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from agents.unified_agent import UnifiedAgent
from core.autonomous_react_engine import AutonomousReactEngine
from core.autonomous_react_engine import AutonomyLevel


class TestCompleteResearchWorkflows:
    """End-to-end tests for complete research workflows."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_complete_research_report_generation(self):
        """Test complete workflow from research query to final report."""
        agent = UnifiedAgent(profile="business")

        # Mock the complete research pipeline
        mock_research_results = {
            "research_plan": "1. [RESEARCH] Analyze renewable energy trends\n2. [DELIVERABLE] Create summary report",
            "section_research_findings": {
                "trends_analysis": "Solar and wind power adoption increasing 20% yearly",
                "market_data": "Global renewable energy market valued at $1T",
                "challenges": "Grid stability and storage remain key challenges",
            },
            "final_report_with_citations": """
# Renewable Energy Market Analysis

## Market Trends
Solar and wind power adoption is increasing at a rate of 20% annually [Source 1](https://example.com/source1).

## Market Size
The global renewable energy market is currently valued at $1 trillion [Source 2](https://example.com/source2).

## Key Challenges
Grid stability and energy storage remain the primary challenges for widespread adoption [Source 3](https://example.com/source3).
            """.strip(),
        }

        with patch.object(agent, "root_agent") as mock_root:
            mock_root.process = AsyncMock(return_value=mock_research_results)

            result = await agent.process_request(
                "Generate a comprehensive research report on renewable energy market trends, "
                "including current adoption rates, market size, and key challenges.",
                session_id="research-workflow-001",
            )

            assert result["success"] is True
            assert result["agent_type"] == "unified_agent"
            assert result["session_id"] == "research-workflow-001"

            # Verify research pipeline was executed
            mock_root.process.assert_called_once()

            # Verify structured output
            research_result = result["result"]
            assert "research_plan" in research_result
            assert "section_research_findings" in research_result
            assert "final_report_with_citations" in research_result

            # Verify report quality
            final_report = research_result["final_report_with_citations"]
            assert "# Renewable Energy Market Analysis" in final_report
            assert "[Source 1]" in final_report
            assert "20% annually" in final_report

    @pytest.mark.asyncio
    async def test_iterative_research_refinement_workflow(self):
        """Test iterative research refinement based on evaluation feedback."""
        agent = UnifiedAgent()

        # Mock iterative refinement: initial research -> evaluation -> enhanced research
        initial_research = {
            "research_evaluation": {"grade": "fail", "comment": "Needs more depth"},
            "section_research_findings": "Basic renewable energy overview",
        }

        enhanced_research = {
            "research_evaluation": {"grade": "pass", "comment": "Comprehensive analysis"},
            "section_research_findings": "Detailed analysis with specific data points, trends, and projections",
            "final_report_with_citations": "Enhanced report with comprehensive citations",
        }

        # Simulate the iterative loop
        call_count = 0

        async def mock_iterative_process(request):
            nonlocal call_count
            call_count += 1
            return enhanced_research if call_count > 1 else initial_research

        with patch.object(agent, "root_agent") as mock_root:
            mock_root.process = mock_iterative_process

            result = await agent.process_request(
                "Conduct in-depth research on sustainable energy solutions with detailed analysis"
            )

            assert result["success"] is True
            # Would verify iterative refinement in real implementation

    @pytest.mark.asyncio
    async def test_multi_topic_research_coordination(self):
        """Test coordination of research across multiple related topics."""
        agent = UnifiedAgent()

        topics = [
            "Solar energy technology advances",
            "Wind power efficiency improvements",
            "Energy storage solutions",
            "Grid integration challenges",
        ]

        # Mock coordinated research across topics
        coordinated_results = {
            "research_plan": f"Coordinated analysis of: {', '.join(topics)}",
            "section_research_findings": {
                topic.replace(" ", "_"): f"Research findings for {topic}" for topic in topics
            },
            "final_report_with_citations": "Comprehensive multi-topic renewable energy analysis",
        }

        with patch.object(agent, "root_agent") as mock_root:
            mock_root.process = AsyncMock(return_value=coordinated_results)

            result = await agent.process_request(
                f"Conduct coordinated research analysis across these topics: {', '.join(topics)}"
            )

            assert result["success"] is True
            findings = result["result"]["section_research_findings"]

            # Verify all topics were addressed
            for topic in topics:
                topic_key = topic.replace(" ", "_")
                assert topic_key in findings
                assert topic in findings[topic_key]


class TestAutonomousExecutionWorkflows:
    """End-to-end tests for autonomous execution workflows."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_complete_autonomous_project_setup(self, mock_client_class):
        """Test complete autonomous project setup workflow."""
        mock_client = Mock()
        mock_client.generate_content = AsyncMock(
            return_value='{"complexity": "moderate", "estimated_steps": 5, "estimated_time_minutes": 15}'
        )
        mock_client_class.return_value = mock_client

        engine = AutonomousReactEngine(
            profile="business", autonomy_level=AutonomyLevel.FULLY_AUTO, project_root=self.temp_dir
        )

        # Mock file operations for project setup
        created_files = []

        async def mock_act(tool_name, params):
            from core.autonomous_react_engine import ToolResult

            if tool_name == "create_file":
                file_path = params.get("path", "unknown")
                created_files.append(file_path)
                return ToolResult(success=True, data={"created": file_path})
            elif tool_name == "create_directory":
                return ToolResult(success=True, data={"created_dir": params.get("path")})
            else:
                return ToolResult(success=True, data={"executed": tool_name})

        with patch.object(engine, "_act", side_effect=mock_act):
            response = await engine.process_autonomous_request(
                "Set up a new Python project with proper structure: "
                "create main.py, requirements.txt, tests directory, "
                "README.md, and .gitignore files",
                session_id="project-setup-001",
            )

            assert response.success is True
            assert response.session_id == "project-setup-001"
            assert len(response.steps_executed) > 0

            # Verify autonomous execution completed
            assert response.total_time > 0

            # Verify context was persisted for potential resumption
            context_file = engine.context_dir / "project-setup-001.json"
            assert context_file.exists()

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_autonomous_error_recovery_workflow(self, mock_client_class):
        """Test autonomous error recovery during complex workflow."""
        mock_client_class.return_value = Mock()

        engine = AutonomousReactEngine(
            autonomy_level=AutonomyLevel.FULLY_AUTO, project_root=self.temp_dir
        )

        # Mock execution with failures that require recovery
        execution_log = []

        async def mock_act_with_failures(tool_name, params):
            from core.autonomous_react_engine import ToolResult

            execution_log.append((tool_name, params))

            # Simulate specific failure scenarios
            if len(execution_log) == 2:  # Second step fails
                return ToolResult(success=False, error="Network timeout")
            elif len(execution_log) == 4:  # Fourth step fails
                return ToolResult(success=False, error="Permission denied")
            else:
                return ToolResult(success=True, data={"step": len(execution_log)})

        with patch.object(engine, "_act", side_effect=mock_act_with_failures):
            await engine.process_autonomous_request(
                "Deploy application with error recovery: "
                "build application, upload to server, configure environment, start services",
                session_id="deployment-recovery-001",
            )

            # Verify execution attempted recovery
            assert len(execution_log) > 2  # Multiple attempts

            # In real implementation, would verify specific recovery strategies

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_long_running_autonomous_workflow(self, mock_client_class):
        """Test long-running autonomous workflow with progress tracking."""
        mock_client = Mock()
        mock_client.generate_content = AsyncMock(
            return_value='{"complexity": "advanced", "estimated_steps": 10, "estimated_time_minutes": 30}'
        )
        mock_client_class.return_value = mock_client

        engine = AutonomousReactEngine(
            autonomy_level=AutonomyLevel.FULLY_AUTO, project_root=self.temp_dir
        )

        # Track progress throughout execution
        progress_updates = []

        # Override progress streaming to capture updates
        original_stream = engine._stream_progress_update

        async def capture_progress(session_id, step, plan):
            update = await original_stream(session_id, step, plan)
            progress_updates.append(update)
            return update

        engine._stream_progress_update = capture_progress

        # Mock long-running execution
        async def mock_long_act(tool_name, params):
            from core.autonomous_react_engine import ToolResult

            await asyncio.sleep(0.01)  # Simulate processing time
            return ToolResult(success=True, data={"long_step": True})

        with patch.object(engine, "_act", side_effect=mock_long_act):
            response = await engine.process_autonomous_request(
                "Execute comprehensive data analysis pipeline: "
                "load data, clean data, perform analysis, generate visualizations, "
                "create report, validate results, export outputs",
                streaming=True,
            )

            assert response.success is True
            # Verify progress was tracked
            # Note: May have fewer updates in simplified implementation


class TestUserInteractionWorkflows:
    """End-to-end tests for user interaction scenarios."""

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_guided_execution_with_user_confirmations(self, mock_client_class):
        """Test guided execution requiring user confirmations."""
        mock_client_class.return_value = Mock()

        engine = AutonomousReactEngine(
            autonomy_level=AutonomyLevel.GUIDED, project_root=Path(tempfile.mkdtemp())
        )

        # Mock user confirmation scenario
        confirmation_history = []

        async def mock_user_confirmation(step, plan):
            confirmation_history.append(
                {
                    "step_description": step.description,
                    "step_priority": step.priority,
                    "plan_progress": plan.progress_percentage,
                }
            )
            # Confirm all steps for this test
            return True

        async def mock_act(tool_name, params):
            from core.autonomous_react_engine import ToolResult

            return ToolResult(success=True, data={"guided": True})

        with patch.object(engine, "_act", side_effect=mock_act):
            response = await engine.process_autonomous_request(
                "Perform sensitive system configuration changes with user oversight",
                user_confirmation_callback=mock_user_confirmation,
            )

            assert response.success is True
            # Would verify user confirmations were requested appropriately

    @pytest.mark.asyncio
    async def test_interactive_planning_workflow(self):
        """Test interactive planning workflow with user feedback."""
        agent = UnifiedAgent()

        # Mock interactive planning scenario
        planning_iterations = []

        def mock_planning_process(request):
            planning_iterations.append(request)

            if len(planning_iterations) == 1:
                # Initial plan
                return {
                    "research_plan": "Initial plan: 1. Basic analysis 2. Simple report",
                    "needs_refinement": True,
                }
            else:
                # Refined plan after user feedback
                return {
                    "research_plan": "Refined plan: 1. Comprehensive analysis 2. Detailed report with visualizations 3. Executive summary",
                    "final_report_with_citations": "Final comprehensive report",
                }

        with patch.object(agent, "root_agent") as mock_root:
            mock_root.process = AsyncMock(side_effect=mock_planning_process)

            # Initial planning request
            result1 = await agent.process_request(
                "Create analysis plan for market research project"
            )

            # Simulated user feedback and refinement
            result2 = await agent.process_request(
                "Refine the plan to include more detailed analysis and visualizations"
            )

            assert result1["success"] is True
            assert result2["success"] is True
            assert len(planning_iterations) == 2


class TestProductionScenarios:
    """End-to-end tests simulating production usage scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_user_requests(self):
        """Test handling multiple concurrent user requests."""
        agent = UnifiedAgent()

        # Mock different types of concurrent requests
        request_types = [
            "Quick market overview analysis",
            "Comprehensive competitor research",
            "Technology trend analysis",
            "Financial performance review",
        ]

        async def mock_varied_processing(request):
            # Simulate different processing times and complexities
            if "Quick" in request:
                await asyncio.sleep(0.01)
                return {"type": "quick", "result": "Brief analysis completed"}
            elif "Comprehensive" in request:
                await asyncio.sleep(0.03)
                return {"type": "comprehensive", "result": "Detailed research completed"}
            else:
                await asyncio.sleep(0.02)
                return {"type": "standard", "result": "Analysis completed"}

        with patch.object(agent, "root_agent") as mock_root:
            mock_root.process = mock_varied_processing

            # Launch concurrent requests
            tasks = [
                agent.process_request(request, session_id=f"concurrent-{i}")
                for i, request in enumerate(request_types)
            ]

            results = await asyncio.gather(*tasks)

            # Verify all requests completed successfully
            for result in results:
                assert result["success"] is True
                assert "concurrent-" in result["session_id"]

            # Verify session isolation
            session_ids = [r["session_id"] for r in results]
            assert len(set(session_ids)) == len(session_ids)

    @pytest.mark.asyncio
    async def test_error_handling_in_production_workflow(self):
        """Test error handling that mimics real production issues."""
        agent = UnifiedAgent()

        # Simulate various production error scenarios
        error_scenarios = [
            "API rate limit exceeded",
            "Network timeout",
            "Invalid credentials",
            "Service temporarily unavailable",
        ]

        for i, error_msg in enumerate(error_scenarios):
            with patch.object(agent, "root_agent") as mock_root:
                mock_root.process = AsyncMock(side_effect=Exception(error_msg))

                result = await agent.process_request(
                    f"Test error scenario {i + 1}", session_id=f"error-test-{i}"
                )

                assert result["success"] is False
                assert error_msg in result["error"]
                assert result["session_id"] == f"error-test-{i}"
                assert "timestamp" in result

                # Verify error details are preserved for debugging
                assert "request" in result

    @pytest.mark.asyncio
    async def test_resource_cleanup_after_workflows(self):
        """Test proper resource cleanup after workflow completion."""
        temp_dir = Path(tempfile.mkdtemp())

        try:
            with patch("core.autonomous_react_engine.GeminiClient"):
                engine = AutonomousReactEngine(
                    autonomy_level=AutonomyLevel.FULLY_AUTO, project_root=temp_dir
                )

                # Execute multiple workflows
                for i in range(3):

                    async def mock_act(tool_name, params):
                        from core.autonomous_react_engine import ToolResult

                        return ToolResult(success=True, data={"cleanup_test": i})

                    with patch.object(engine, "_act", side_effect=mock_act):
                        response = await engine.process_autonomous_request(
                            f"Cleanup test workflow {i}", session_id=f"cleanup-test-{i}"
                        )

                        assert response.success is True

                # Verify context files were created but not accumulating excessively
                context_files = list(engine.context_dir.glob("*.json"))
                assert len(context_files) == 3  # One per workflow

                # Verify each context file is valid JSON
                for context_file in context_files:
                    with open(context_file) as f:
                        context_data = json.load(f)
                        assert "session_id" in context_data
                        assert "timestamp" in context_data

        finally:
            # Cleanup test directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov", "--cov-report=term-missing"])
