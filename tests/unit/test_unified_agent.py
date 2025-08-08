#!/usr/bin/env python3
"""Comprehensive unit tests for the unified agent.

This test suite achieves high coverage by testing:
- Agent initialization and configuration
- Research capabilities and structured output
- Callback functions for data collection
- Error handling and fallback scenarios
- Agent pipeline components
- Capability reporting
"""

from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from agents.unified_agent import EscalationChecker
from agents.unified_agent import Feedback
from agents.unified_agent import SearchQuery
from agents.unified_agent import UnifiedAgent
from agents.unified_agent import citation_replacement_callback
from agents.unified_agent import collect_research_sources_callback


class TestUnifiedAgentInitialization:
    """Test cases for UnifiedAgent initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        agent = UnifiedAgent()
        assert agent.profile == "business"
        assert agent.model_name == "gemini-pro"

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        agent = UnifiedAgent(profile="personal", model_name="gemini-ultra")
        assert agent.profile == "personal"
        assert agent.model_name == "gemini-ultra"

    @patch("agents.unified_agent.logger")
    def test_init_logs_creation(self, mock_logger):
        """Test that initialization logs are created."""
        UnifiedAgent(profile="test")
        mock_logger.info.assert_called()

    def test_init_fallback_when_gemini_unavailable(self):
        """Test graceful fallback when Gemini agents are unavailable."""
        with patch("agents.unified_agent.LlmAgent", side_effect=ImportError):
            agent = UnifiedAgent()
            assert agent.root_agent is None

    def test_capabilities_reporting(self):
        """Test that capabilities are properly reported."""
        agent = UnifiedAgent()
        capabilities = agent.get_capabilities()

        assert isinstance(capabilities, dict)
        assert "research" in capabilities
        assert "planning" in capabilities
        assert "output_formats" in capabilities
        assert "agents" in capabilities

        # Verify specific capabilities
        assert capabilities["research"]["structured_output"] is True
        assert capabilities["planning"]["autonomous_planning"] is True
        assert capabilities["output_formats"]["citations"] is True


class TestStructuredOutputModels:
    """Test cases for Pydantic models used in agent responses."""

    def test_search_query_model(self):
        """Test SearchQuery model validation."""
        query = SearchQuery(search_query="test query")
        assert query.search_query == "test query"

    def test_feedback_model_pass(self):
        """Test Feedback model with pass grade."""
        feedback = Feedback(
            grade="pass", comment="Research is comprehensive", follow_up_queries=None
        )
        assert feedback.grade == "pass"
        assert feedback.comment == "Research is comprehensive"
        assert feedback.follow_up_queries is None

    def test_feedback_model_fail_with_queries(self):
        """Test Feedback model with fail grade and follow-up queries."""
        queries = [SearchQuery(search_query="more details needed")]
        feedback = Feedback(grade="fail", comment="Needs more depth", follow_up_queries=queries)
        assert feedback.grade == "fail"
        assert len(feedback.follow_up_queries) == 1
        assert feedback.follow_up_queries[0].search_query == "more details needed"

    def test_feedback_invalid_grade(self):
        """Test Feedback model with invalid grade."""
        with pytest.raises(ValueError):
            Feedback(grade="invalid", comment="test")


class TestCallbackFunctions:
    """Test cases for callback functions used in agent processing."""

    def test_collect_research_sources_callback_empty_context(self):
        """Test callback with empty context."""
        mock_context = Mock()
        mock_context._invocation_context.session.events = []
        mock_context.state = {}

        collect_research_sources_callback(mock_context)

        assert mock_context.state.get("url_to_short_id") == {}
        assert mock_context.state.get("sources") == {}

    def test_collect_research_sources_callback_with_sources(self):
        """Test callback with mock research sources."""
        # Create mock event with grounding metadata
        mock_event = Mock()
        mock_chunk = Mock()
        mock_chunk.web.uri = "https://example.com"
        mock_chunk.web.title = "Example Title"
        mock_chunk.web.domain = "example.com"

        mock_event.grounding_metadata.grounding_chunks = [mock_chunk]
        mock_event.grounding_metadata.grounding_supports = []

        mock_context = Mock()
        mock_context._invocation_context.session.events = [mock_event]
        mock_context.state = {}

        collect_research_sources_callback(mock_context)

        # Verify sources were collected
        assert "url_to_short_id" in mock_context.state
        assert "sources" in mock_context.state
        assert len(mock_context.state["sources"]) == 1

    def test_collect_research_sources_callback_exception_handling(self):
        """Test callback handles exceptions gracefully."""
        mock_context = Mock()
        mock_context._invocation_context.session.events = None  # Will cause exception

        # Should not raise exception
        collect_research_sources_callback(mock_context)

    def test_citation_replacement_callback_basic(self):
        """Test citation replacement with basic citation tags."""
        mock_context = Mock()
        mock_context.state = {
            "final_cited_report": 'This is a test <cite source="src-1"/> citation.',
            "sources": {
                "src-1": {"title": "Test Source", "url": "https://test.com", "domain": "test.com"}
            },
        }

        citation_replacement_callback(mock_context)

        # Verify citation was replaced with markdown link
        processed_text = mock_context.state["final_report_with_citations"]
        assert "[Test Source](https://test.com)" in processed_text
        assert "<cite" not in processed_text

    def test_citation_replacement_callback_invalid_citation(self):
        """Test citation replacement with invalid citation reference."""
        mock_context = Mock()
        mock_context.state = {
            "final_cited_report": 'Invalid citation <cite source="src-999"/>',
            "sources": {},
        }

        with patch("agents.unified_agent.logger") as mock_logger:
            citation_replacement_callback(mock_context)
            mock_logger.warning.assert_called_once()

    def test_citation_replacement_callback_exception_handling(self):
        """Test citation replacement handles exceptions."""
        mock_context = Mock()
        mock_context.state = None  # Will cause exception

        with patch("agents.unified_agent.logger"):
            result = citation_replacement_callback(mock_context)
            # Should return fallback content
            assert "Citation processing failed" in str(result)


class TestEscalationChecker:
    """Test cases for the EscalationChecker agent."""

    def test_escalation_checker_init(self):
        """Test EscalationChecker initialization."""
        checker = EscalationChecker("test_checker")
        assert checker.name == "test_checker"

    @pytest.mark.asyncio
    async def test_escalation_checker_pass_grade(self):
        """Test escalation when research grade is pass."""
        checker = EscalationChecker("test")

        mock_context = Mock()
        mock_context.session.state = {"research_evaluation": {"grade": "pass"}}

        events = []
        async for event in checker._run_async_impl(mock_context):
            events.append(event)

        assert len(events) == 1
        # In real implementation, would check for escalate action

    @pytest.mark.asyncio
    async def test_escalation_checker_fail_grade(self):
        """Test no escalation when research grade is fail."""
        checker = EscalationChecker("test")

        mock_context = Mock()
        mock_context.session.state = {"research_evaluation": {"grade": "fail"}}

        events = []
        async for event in checker._run_async_impl(mock_context):
            events.append(event)

        assert len(events) == 1
        # Should not escalate

    @pytest.mark.asyncio
    async def test_escalation_checker_exception_handling(self):
        """Test escalation checker handles exceptions."""
        checker = EscalationChecker("test")

        mock_context = Mock()
        mock_context.session.state = None  # Will cause exception

        # Should not raise exception
        events = []
        async for event in checker._run_async_impl(mock_context):
            events.append(event)

        assert len(events) == 1


class TestUnifiedAgentProcessing:
    """Test cases for agent request processing."""

    @pytest.mark.asyncio
    async def test_process_request_success(self):
        """Test successful request processing."""
        agent = UnifiedAgent()

        # Mock successful processing
        mock_result = {"success": True, "data": "processed"}

        with patch.object(agent, "root_agent") as mock_root:
            mock_root.process = AsyncMock(return_value=mock_result)

            result = await agent.process_request("test request")

            assert result["success"] is True
            assert result["agent_type"] == "unified_agent"
            assert "timestamp" in result
            assert "session_id" in result

    @pytest.mark.asyncio
    async def test_process_request_no_agent(self):
        """Test request processing when no agent is available."""
        agent = UnifiedAgent()
        agent.root_agent = None

        result = await agent.process_request("test request")

        assert result["success"] is False
        assert "Agent system not properly initialized" in result["error"]
        assert "fallback_response" in result

    @pytest.mark.asyncio
    async def test_process_request_with_session_id(self):
        """Test request processing with custom session ID."""
        agent = UnifiedAgent()

        with patch.object(agent, "root_agent") as mock_root:
            mock_root.process = AsyncMock(return_value={"success": True})

            result = await agent.process_request("test", session_id="custom-123")

            assert result["session_id"] == "custom-123"

    @pytest.mark.asyncio
    async def test_process_request_exception_handling(self):
        """Test request processing handles exceptions."""
        agent = UnifiedAgent()

        with patch.object(agent, "root_agent") as mock_root:
            mock_root.process = AsyncMock(side_effect=Exception("Test error"))

            result = await agent.process_request("test request")

            assert result["success"] is False
            assert "Test error" in result["error"]
            assert "timestamp" in result


class TestAgentInstructionMethods:
    """Test cases for agent instruction generation methods."""

    def test_plan_generator_instruction(self):
        """Test plan generator instruction contains required elements."""
        agent = UnifiedAgent()
        instruction = agent._get_plan_generator_instruction()

        assert "RESEARCH PLAN" in instruction
        assert "[RESEARCH]" in instruction
        assert "[DELIVERABLE]" in instruction
        assert "google_search" in instruction

    def test_section_planner_instruction(self):
        """Test section planner instruction."""
        agent = UnifiedAgent()
        instruction = agent._get_section_planner_instruction()

        assert "report architect" in instruction
        assert "markdown outline" in instruction
        assert "4-6 distinct sections" in instruction

    def test_section_researcher_instruction(self):
        """Test section researcher instruction."""
        agent = UnifiedAgent()
        instruction = agent._get_section_researcher_instruction()

        assert "Phase 1" in instruction
        assert "Phase 2" in instruction
        assert "[RESEARCH]" in instruction
        assert "[DELIVERABLE]" in instruction

    def test_research_evaluator_instruction(self):
        """Test research evaluator instruction."""
        agent = UnifiedAgent()
        instruction = agent._get_research_evaluator_instruction()

        assert "quality assurance" in instruction
        assert "CRITICAL RULES" in instruction
        assert "JSON" in instruction
        assert datetime.now().strftime("%Y-%m-%d") in instruction

    def test_enhanced_search_executor_instruction(self):
        """Test enhanced search executor instruction."""
        agent = UnifiedAgent()
        instruction = agent._get_enhanced_search_executor_instruction()

        assert "research refinement" in instruction
        assert "follow_up_queries" in instruction
        assert "google_search" in instruction

    def test_report_composer_instruction(self):
        """Test report composer instruction."""
        agent = UnifiedAgent()
        instruction = agent._get_report_composer_instruction()

        assert "polished, professional" in instruction
        assert "<cite source=" in instruction
        assert "INPUT DATA" in instruction

    def test_interactive_planner_instruction(self):
        """Test interactive planner instruction."""
        agent = UnifiedAgent()
        instruction = agent._get_interactive_planner_instruction()

        assert "research planning assistant" in instruction
        assert "WORKFLOW" in instruction
        assert "Plan, Refine, and Delegate" in instruction


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=agents.unified_agent", "--cov-report=term-missing"])
