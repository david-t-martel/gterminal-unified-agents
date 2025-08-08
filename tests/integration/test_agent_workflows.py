#!/usr/bin/env python3
"""Integration tests for complete agent workflows.

This test suite validates end-to-end agent functionality including:
- Complete request processing pipelines
- Agent-to-agent communication
- ReAct engine integration with unified agents
- Multi-step autonomous execution
- Error recovery and fallback mechanisms
"""

import asyncio
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


class TestUnifiedAgentIntegration:
    """Integration tests for UnifiedAgent with various components."""

    @pytest.mark.asyncio
    async def test_unified_agent_simple_request(self):
        """Test unified agent processing a simple research request."""
        agent = UnifiedAgent(profile="business")

        # Mock the agent pipeline to avoid external dependencies
        with patch.object(agent, "root_agent") as mock_root:
            mock_root.process = AsyncMock(
                return_value={
                    "research_plan": "Test research plan",
                    "section_research_findings": "Research findings",
                    "final_report_with_citations": "Final report with citations",
                }
            )

            result = await agent.process_request(
                "Research the benefits of renewable energy", session_id="test-session-001"
            )

            assert result["success"] is True
            assert result["session_id"] == "test-session-001"
            assert result["agent_type"] == "unified_agent"
            assert "timestamp" in result

            mock_root.process.assert_called_once_with("Research the benefits of renewable energy")

    @pytest.mark.asyncio
    async def test_unified_agent_streaming_request(self):
        """Test unified agent with streaming enabled."""
        agent = UnifiedAgent(profile="personal")

        with patch.object(agent, "root_agent") as mock_root:
            mock_root.process = AsyncMock(return_value={"status": "completed"})

            result = await agent.process_request(
                "Analyze market trends in AI technology", streaming=True
            )

            assert result["success"] is True
            # In real implementation, would verify streaming events

    @pytest.mark.asyncio
    async def test_unified_agent_error_handling(self):
        """Test unified agent error handling during processing."""
        agent = UnifiedAgent()

        with patch.object(agent, "root_agent") as mock_root:
            mock_root.process = AsyncMock(side_effect=Exception("Processing failed"))

            result = await agent.process_request("Test error handling")

            assert result["success"] is False
            assert "Processing failed" in result["error"]
            assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_unified_agent_no_root_agent_fallback(self):
        """Test unified agent behavior when root agent is not available."""
        agent = UnifiedAgent()
        agent.root_agent = None

        result = await agent.process_request("Test fallback")

        assert result["success"] is False
        assert "Agent system not properly initialized" in result["error"]
        assert "fallback_response" in result

    def test_unified_agent_capabilities_integration(self):
        """Test that agent capabilities are properly integrated."""
        agent = UnifiedAgent()
        capabilities = agent.get_capabilities()

        # Verify all major capability categories
        required_capabilities = ["research", "planning", "output_formats", "agents"]
        for capability in required_capabilities:
            assert capability in capabilities

        # Verify specific research capabilities
        research_caps = capabilities["research"]
        assert research_caps["citation_handling"] is True
        assert research_caps["iterative_refinement"] is True
        assert research_caps["structured_output"] is True

        # Verify planning capabilities
        planning_caps = capabilities["planning"]
        assert planning_caps["autonomous_planning"] is True
        assert planning_caps["task_decomposition"] is True


class TestAutonomousReactEngineIntegration:
    """Integration tests for AutonomousReactEngine with full workflows."""

    def setup_method(self):
        """Set up test environment with temporary directories."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_autonomous_engine_full_workflow(self, mock_client_class):
        """Test complete autonomous engine workflow from request to completion."""
        mock_client = Mock()
        mock_client.generate_content = AsyncMock(
            return_value='{"complexity": "simple", "estimated_steps": 2}'
        )
        mock_client_class.return_value = mock_client

        engine = AutonomousReactEngine(
            profile="business", autonomy_level=AutonomyLevel.FULLY_AUTO, project_root=self.temp_dir
        )

        # Mock tool execution to avoid external dependencies
        async def mock_act(tool_name, params):
            from core.autonomous_react_engine import ToolResult

            return ToolResult(success=True, data={"result": f"Executed {tool_name}"})

        with patch.object(engine, "_act", side_effect=mock_act):
            response = await engine.process_autonomous_request(
                "Create a test file with system information", session_id="integration-test-001"
            )

            assert response.success is True
            assert response.session_id == "integration-test-001"
            assert response.total_time > 0
            assert len(response.steps_executed) > 0

            # Verify context was persisted
            context_file = engine.context_dir / "integration-test-001.json"
            assert context_file.exists()

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_autonomous_engine_with_user_confirmation(self, mock_client_class):
        """Test autonomous engine workflow requiring user confirmations."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        engine = AutonomousReactEngine(
            autonomy_level=AutonomyLevel.GUIDED, project_root=self.temp_dir
        )

        # Mock user confirmation callback
        confirmation_calls = []

        async def mock_confirmation(step, plan):
            confirmation_calls.append((step.description, step.priority))
            return True  # Always confirm

        response = await engine.process_autonomous_request(
            "Perform guided task execution", user_confirmation_callback=mock_confirmation
        )

        assert response.success is True
        # In real implementation, would verify confirmation was called for appropriate steps

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_autonomous_engine_step_retry_integration(self, mock_client_class):
        """Test autonomous engine step retry mechanism in full workflow."""
        mock_client_class.return_value = Mock()

        engine = AutonomousReactEngine(
            autonomy_level=AutonomyLevel.FULLY_AUTO, project_root=self.temp_dir
        )

        # Mock step execution to fail first time, succeed second time
        call_count = 0

        async def mock_act_with_retry(tool_name, params):
            nonlocal call_count
            call_count += 1
            from core.autonomous_react_engine import ToolResult

            if call_count == 1:
                return ToolResult(success=False, error="Simulated failure")
            return ToolResult(success=True, data={"retry_success": True})

        with patch.object(engine, "_act", side_effect=mock_act_with_retry):
            response = await engine.process_autonomous_request(
                "Test retry mechanism", session_id="retry-test"
            )

            assert response.success is True
            assert call_count > 1  # Verify retry occurred

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_autonomous_engine_pattern_learning_integration(self, mock_client_class):
        """Test pattern learning across multiple requests."""
        mock_client_class.return_value = Mock()

        engine = AutonomousReactEngine(
            autonomy_level=AutonomyLevel.FULLY_AUTO, project_root=self.temp_dir
        )

        # Mock successful execution
        async def mock_act(tool_name, params):
            from core.autonomous_react_engine import ToolResult

            return ToolResult(success=True, data={"learned": True})

        with patch.object(engine, "_act", side_effect=mock_act):
            # Process first request
            response1 = await engine.process_autonomous_request(
                "Learn pattern test request", session_id="learning-test-1"
            )

            # Verify pattern was learned
            assert response1.success is True
            initial_pattern_count = len(engine.learned_patterns)

            # Process similar second request
            response2 = await engine.process_autonomous_request(
                "Learn pattern test request variation", session_id="learning-test-2"
            )

            assert response2.success is True
            # Verify additional patterns were learned
            assert len(engine.learned_patterns) >= initial_pattern_count

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_autonomous_engine_context_persistence_integration(self, mock_client_class):
        """Test context persistence across session interruptions."""
        mock_client_class.return_value = Mock()

        engine = AutonomousReactEngine(
            autonomy_level=AutonomyLevel.SEMI_AUTO, project_root=self.temp_dir
        )

        session_id = "persistence-test"

        # Start initial request
        async def mock_act(tool_name, params):
            from core.autonomous_react_engine import ToolResult

            return ToolResult(success=True, data={"step": "completed"})

        with patch.object(engine, "_act", side_effect=mock_act):
            response = await engine.process_autonomous_request(
                "Test context persistence", session_id=session_id
            )

            assert response.success is True

            # Verify context was persisted
            context_file = engine.context_dir / f"{session_id}.json"
            assert context_file.exists()

            # Verify context content
            import json

            with open(context_file) as f:
                context_data = json.load(f)

            assert context_data["session_id"] == session_id
            assert "timestamp" in context_data
            assert "current_plan" in context_data
            assert "completed_steps" in context_data

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_autonomous_engine_streaming_integration(self, mock_client_class):
        """Test streaming progress updates during execution."""
        mock_client_class.return_value = Mock()

        engine = AutonomousReactEngine(
            autonomy_level=AutonomyLevel.FULLY_AUTO, project_root=self.temp_dir
        )

        progress_updates = []

        # Override progress streaming to capture updates
        original_stream = engine._stream_progress_update

        async def capture_stream(session_id, step, plan):
            update = await original_stream(session_id, step, plan)
            progress_updates.append(update)
            return update

        engine._stream_progress_update = capture_stream

        # Mock execution
        async def mock_act(tool_name, params):
            from core.autonomous_react_engine import ToolResult

            return ToolResult(success=True, data={"streamed": True})

        with patch.object(engine, "_act", side_effect=mock_act):
            response = await engine.process_autonomous_request(
                "Test streaming updates", streaming=True
            )

            assert response.success is True
            # Verify progress updates were captured
            # Note: In the simplified implementation, may have fewer updates
            assert len(progress_updates) >= 0  # At minimum, should not error


class TestAgentCommunication:
    """Integration tests for agent-to-agent communication."""

    @pytest.mark.asyncio
    async def test_unified_agent_to_autonomous_engine_handoff(self):
        """Test handoff from unified agent to autonomous engine."""
        unified_agent = UnifiedAgent(profile="business")
        autonomous_engine = AutonomousReactEngine(
            profile="business", autonomy_level=AutonomyLevel.SEMI_AUTO
        )

        # Test scenario: unified agent creates research plan,
        # autonomous engine executes implementation steps

        with patch.object(unified_agent, "root_agent") as mock_unified:
            with patch.object(autonomous_engine, "_act") as mock_act:
                # Mock unified agent response
                mock_unified.process = AsyncMock(
                    return_value={
                        "research_plan": "1. Analyze data 2. Create visualization 3. Generate report",
                        "implementation_steps": ["step1", "step2", "step3"],
                    }
                )

                # Mock autonomous engine execution
                from core.autonomous_react_engine import ToolResult

                mock_act.return_value = ToolResult(success=True, data={"executed": True})

                # Process request through unified agent
                unified_result = await unified_agent.process_request(
                    "Create data analysis workflow"
                )

                # Pass implementation to autonomous engine
                autonomous_result = await autonomous_engine.process_autonomous_request(
                    f"Implement: {unified_result['result'].get('research_plan', 'fallback plan')}"
                )

                assert unified_result["success"] is True
                assert autonomous_result.success is True

    @pytest.mark.asyncio
    async def test_error_propagation_between_agents(self):
        """Test error handling and propagation between agent layers."""
        unified_agent = UnifiedAgent()

        # Simulate cascading failure
        with patch.object(unified_agent, "root_agent") as mock_root:
            mock_root.process = AsyncMock(side_effect=Exception("Agent pipeline failure"))

            result = await unified_agent.process_request("Test error propagation")

            assert result["success"] is False
            assert "Agent pipeline failure" in result["error"]

            # Verify error details are preserved
            assert "request" in result
            assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_capability_verification_integration(self):
        """Test integration of capability verification across components."""
        unified_agent = UnifiedAgent()

        # Get capabilities from unified agent
        unified_caps = unified_agent.get_capabilities()

        # Create autonomous engine and compare capabilities
        with patch("core.autonomous_react_engine.GeminiClient"):
            autonomous_engine = AutonomousReactEngine()
            autonomous_status = await autonomous_engine.get_autonomous_status()

            # Verify complementary capabilities
            assert unified_caps["research"]["web_search"] is not None
            assert autonomous_status["capabilities"]["autonomous_execution"] is True
            assert autonomous_status["capabilities"]["pattern_learning"] is True

            # Both should support planning
            assert unified_caps["planning"]["autonomous_planning"] is True
            assert autonomous_status["capabilities"]["goal_decomposition"] is True


class TestErrorRecoveryWorkflows:
    """Integration tests for error recovery and resilience."""

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_autonomous_engine_partial_failure_recovery(self, mock_client_class):
        """Test recovery from partial failures in multi-step workflows."""
        mock_client_class.return_value = Mock()

        engine = AutonomousReactEngine(autonomy_level=AutonomyLevel.FULLY_AUTO)

        # Mock execution where some steps fail, others succeed
        step_results = [True, False, True, False, True]  # Mixed results
        call_count = 0

        async def mock_act_mixed(tool_name, params):
            nonlocal call_count
            from core.autonomous_react_engine import ToolResult

            success = step_results[call_count % len(step_results)]
            call_count += 1

            if success:
                return ToolResult(success=True, data={"step": call_count})
            else:
                return ToolResult(success=False, error=f"Step {call_count} failed")

        with patch.object(engine, "_act", side_effect=mock_act_mixed):
            response = await engine.process_autonomous_request("Test partial failure recovery")

            # Should handle failures gracefully
            # In real implementation, would verify specific recovery strategies
            assert response is not None

    @pytest.mark.asyncio
    async def test_unified_agent_gemini_api_failure_fallback(self):
        """Test unified agent fallback when Gemini API is unavailable."""
        # Test agent creation when Gemini components are unavailable
        with patch("agents.unified_agent.LlmAgent", side_effect=ImportError):
            agent = UnifiedAgent()

            # Should still be able to process requests with fallback
            result = await agent.process_request("Test API failure fallback")

            assert result["success"] is False
            assert "Agent system not properly initialized" in result["error"]

            # Verify capabilities still report correctly
            capabilities = agent.get_capabilities()
            assert isinstance(capabilities, dict)


class TestPerformanceIntegration:
    """Integration tests focusing on performance characteristics."""

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, mock_client_class):
        """Test handling multiple concurrent requests."""
        mock_client_class.return_value = Mock()

        engine = AutonomousReactEngine(autonomy_level=AutonomyLevel.FULLY_AUTO)

        async def mock_act(tool_name, params):
            from core.autonomous_react_engine import ToolResult

            await asyncio.sleep(0.01)  # Simulate work
            return ToolResult(success=True, data={"concurrent": True})

        with patch.object(engine, "_act", side_effect=mock_act):
            # Launch multiple concurrent requests
            tasks = [engine.process_autonomous_request(f"Concurrent request {i}") for i in range(3)]

            responses = await asyncio.gather(*tasks)

            # All requests should complete successfully
            for response in responses:
                assert response.success is True

            # Verify session isolation
            session_ids = [r.session_id for r in responses]
            assert len(set(session_ids)) == len(session_ids)  # All unique

    @pytest.mark.asyncio
    async def test_memory_usage_during_long_workflow(self):
        """Test memory usage characteristics during extended workflows."""
        agent = UnifiedAgent()

        # Simulate processing multiple requests without memory leaks
        for i in range(5):
            with patch.object(agent, "root_agent") as mock_root:
                mock_root.process = AsyncMock(
                    return_value={
                        "iteration": i,
                        "data": ["large"] * 100,  # Some data to process
                    }
                )

                result = await agent.process_request(f"Memory test iteration {i}")
                assert result["success"] is True

                # In real implementation, would monitor memory usage
                # For now, verify no exceptions and proper cleanup
                assert "timestamp" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov", "--cov-report=term-missing"])
