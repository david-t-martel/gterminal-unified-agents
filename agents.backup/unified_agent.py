#!/usr/bin/env python3
"""Unified Agent Implementation - Merged from GAPP.

This module consolidates agent functionality from GAPP into gterminal,
providing comprehensive research capabilities, autonomous operation,
and advanced agent coordination.

Key Features:
- Advanced research with web search and citation
- Autonomous planning and execution
- Structured output with Pydantic models
- Multiple agent types and coordination
- Callback systems for data collection
"""

from collections.abc import AsyncGenerator
from datetime import datetime
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel
from pydantic import Field

# Gemini Agent imports with fallback handling
try:
    from google.generativeai import types as genai_types
    from google.generativeai.agents import AgentTool
    from google.generativeai.agents import BaseAgent
    from google.generativeai.agents import BuiltInPlanner
    from google.generativeai.agents import Event
    from google.generativeai.agents import EventActions
    from google.generativeai.agents import InvocationContext
    from google.generativeai.agents import LlmAgent
    from google.generativeai.agents import LoopAgent
    from google.generativeai.agents import SequentialAgent
    from google.generativeai.agents.config import config
    from google.generativeai.agents.utils import google_search
    from google.generativeai.types import CallbackContext
except ImportError:
    # Fallback for missing dependencies - define stub classes
    class CallbackContext:
        """Fallback CallbackContext when Gemini agents not available."""

        pass

    class genai_types:
        """Fallback genai_types namespace."""

        class Content:
            """Fallback Content class."""

            def __init__(self, parts=None):
                self.parts = parts or []

        class Part:
            """Fallback Part class."""

            def __init__(self, text=""):
                self.text = text

        class ThinkingConfig:
            """Fallback ThinkingConfig class."""

            def __init__(self, include_thoughts=False):
                self.include_thoughts = include_thoughts

    class BaseAgent:
        """Fallback BaseAgent when Gemini agents not available."""

        def __init__(self, name: str) -> None:
            self.name = name

    class LlmAgent:
        """Fallback LlmAgent when Gemini agents not available."""

        pass

    class SequentialAgent:
        """Fallback SequentialAgent when Gemini agents not available."""

        pass

    class LoopAgent:
        """Fallback LoopAgent when Gemini agents not available."""

        pass

    class InvocationContext:
        """Fallback InvocationContext when Gemini agents not available."""

        pass

    class Event:
        """Fallback Event when Gemini agents not available."""

        pass

    class EventActions:
        """Fallback EventActions when Gemini agents not available."""

        pass

    class BuiltInPlanner:
        """Fallback BuiltInPlanner when Gemini agents not available."""

        pass

    class AgentTool:
        """Fallback AgentTool when Gemini agents not available."""

        pass

    def google_search(*_args, **_kwargs) -> None:
        """Stub function for google_search when library is not available."""
        return None

    class ConfigStub:
        """Stub configuration when Gemini agents not available."""

        worker_model = "gemini-pro"
        critic_model = "gemini-pro"
        max_search_iterations = 3

    config = ConfigStub()

logger = logging.getLogger(__name__)


# --- Structured Output Models ---
class SearchQuery(BaseModel):
    """Model representing a specific search query for web search."""

    search_query: str = Field(description="A highly specific and targeted query for web search.")


class Feedback(BaseModel):
    """Model for providing evaluation feedback on research quality."""

    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result. 'pass' if the research is sufficient, 'fail' if it needs revision.",
    )
    comment: str = Field(
        description="Detailed explanation of the evaluation, highlighting strengths and/or weaknesses of the research.",
    )
    follow_up_queries: list[SearchQuery] | None = Field(
        default=None,
        description="A list of specific, targeted follow-up search queries needed to fix research gaps. This should be null or empty if the grade is 'pass'.",
    )


# --- Callbacks ---
def collect_research_sources_callback(callback_context: CallbackContext) -> None:
    """Collects and organizes web-based research sources and their supported claims from agent events.

    This function processes the agent's `session.events` to extract web source details (URLs,
    titles, domains from `grounding_chunks`) and associated text segments with confidence scores
    (from `grounding_supports`). The aggregated source information and a mapping of URLs to short
    IDs are cumulatively stored in `callback_context.state`.

    Args:
        callback_context: The context object providing access to the agent's
            session events and persistent state.
    """
    try:
        session = callback_context._invocation_context.session
        url_to_short_id = callback_context.state.get("url_to_short_id", {})
        sources = callback_context.state.get("sources", {})
        id_counter = len(url_to_short_id) + 1

        for event in session.events:
            if not (event.grounding_metadata and event.grounding_metadata.grounding_chunks):
                continue
            chunks_info: dict[str, Any] = {}
            for idx, chunk in enumerate(event.grounding_metadata.grounding_chunks):
                if not chunk.web:
                    continue
                url = chunk.web.uri
                title = chunk.web.title if chunk.web.title != chunk.web.domain else chunk.web.domain
                if url not in url_to_short_id:
                    short_id = f"src-{id_counter}"
                    url_to_short_id[url] = short_id
                    sources[short_id] = {
                        "short_id": short_id,
                        "title": title,
                        "url": url,
                        "domain": chunk.web.domain,
                        "supported_claims": [],
                    }
                    id_counter += 1
                chunks_info[idx] = url_to_short_id[url]
            if event.grounding_metadata.grounding_supports:
                for support in event.grounding_metadata.grounding_supports:
                    confidence_scores = support.confidence_scores or []
                    chunk_indices = support.grounding_chunk_indices or []
                    for i, chunk_idx in enumerate(chunk_indices):
                        if chunk_idx in chunks_info:
                            short_id = chunks_info[chunk_idx]
                            confidence = confidence_scores[i] if i < len(confidence_scores) else 0.5
                            text_segment = support.segment.text if support.segment else ""
                            sources[short_id]["supported_claims"].append(
                                {
                                    "text_segment": text_segment,
                                    "confidence": confidence,
                                },
                            )
        callback_context.state["url_to_short_id"] = url_to_short_id
        callback_context.state["sources"] = sources
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"Failed to collect research sources: {e}")


def citation_replacement_callback(
    callback_context: CallbackContext,
) -> genai_types.Content:
    """Replaces citation tags in a report with Markdown-formatted links.

    Processes 'final_cited_report' from context state, converting tags like
    `<cite source="src-N"/>` into hyperlinks using source information from
    `callback_context.state["sources"]`. Also fixes spacing around punctuation.

    Args:
        callback_context: Contains the report and source information.

    Returns:
        The processed report with Markdown citation links.
    """
    try:
        final_report = callback_context.state.get("final_cited_report", "")
        sources = callback_context.state.get("sources", {})

        def tag_replacer(match: re.Match) -> str:
            short_id = match.group(1)
            if not (source_info := sources.get(short_id)):
                logger.warning(f"Invalid citation tag found and removed: {match.group(0)}")
                return ""
            display_text = source_info.get("title", source_info.get("domain", short_id))
            return f" [{display_text}]({source_info['url']})"

        processed_report = re.sub(
            r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/>',
            tag_replacer,
            final_report,
        )
        processed_report = re.sub(r"\s+([.,;:])", r"\1", processed_report)
        callback_context.state["final_report_with_citations"] = processed_report
        return genai_types.Content(parts=[genai_types.Part(text=processed_report)])
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"Citation replacement failed: {e}")
        return genai_types.Content(parts=[genai_types.Part(text="Citation processing failed")])


# --- Custom Agent for Loop Control ---
class EscalationChecker(BaseAgent):
    """Checks research evaluation and escalates to stop the loop if grade is 'pass'."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Run the escalation checker logic."""
        try:
            evaluation_result = ctx.session.state.get("research_evaluation")
            if evaluation_result and evaluation_result.get("grade") == "pass":
                logger.info(f"[{self.name}] Research evaluation passed. Escalating to stop loop.")
                yield Event(author=self.name, actions=EventActions(escalate=True))
            else:
                logger.info(
                    f"[{self.name}] Research evaluation failed or not found. Loop will continue."
                )
                yield Event(author=self.name)
        except (KeyError, AttributeError, RuntimeError) as e:
            logger.warning(f"EscalationChecker failed: {e}")
            yield Event(author=self.name)


# --- Main Unified Agent Class ---
class UnifiedAgent:
    """Unified Agent providing comprehensive research and analysis capabilities.

    This class consolidates functionality from GAPP's agent system into gterminal,
    providing autonomous research, planning, and execution capabilities.
    """

    def __init__(self, profile: str = "business", model_name: str | None = None):
        """Initialize the unified agent.

        Args:
            profile: GCP profile to use ('business' or 'personal')
            model_name: Optional model name override
        """
        self.profile = profile
        self.model_name = model_name or "gemini-pro"

        # Initialize agent components
        self._initialize_agents()

        logger.info("✅ Unified Agent initialized")
        logger.info(f"   Profile: {profile}")
        logger.info(f"   Model: {self.model_name}")

    def _initialize_agents(self) -> None:
        """Initialize the agent pipeline components."""
        try:
            # Plan generator agent
            self.plan_generator = LlmAgent(
                model=config.worker_model,
                name="plan_generator",
                description="Generates or refine research plans using minimal search for clarification.",
                instruction=self._get_plan_generator_instruction(),
                tools=[google_search] if google_search else [],
            )

            # Section planner agent
            self.section_planner = LlmAgent(
                model=config.worker_model,
                name="section_planner",
                description="Breaks down the research plan into a structured markdown outline.",
                instruction=self._get_section_planner_instruction(),
                output_key="report_sections",
            )

            # Section researcher agent
            self.section_researcher = LlmAgent(
                model=config.worker_model,
                name="section_researcher",
                description="Performs comprehensive web research and synthesis.",
                planner=BuiltInPlanner(
                    thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
                ),
                instruction=self._get_section_researcher_instruction(),
                tools=[google_search] if google_search else [],
                output_key="section_research_findings",
                after_agent_callback=collect_research_sources_callback,
            )

            # Research evaluator agent
            self.research_evaluator = LlmAgent(
                model=config.critic_model,
                name="research_evaluator",
                description="Critically evaluates research and generates follow-up queries.",
                instruction=self._get_research_evaluator_instruction(),
                output_schema=Feedback,
                disallow_transfer_to_parent=True,
                disallow_transfer_to_peers=True,
                output_key="research_evaluation",
            )

            # Enhanced search executor agent
            self.enhanced_search_executor = LlmAgent(
                model=config.worker_model,
                name="enhanced_search_executor",
                description="Executes follow-up searches and integrates new findings.",
                planner=BuiltInPlanner(
                    thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
                ),
                instruction=self._get_enhanced_search_executor_instruction(),
                tools=[google_search] if google_search else [],
                output_key="section_research_findings",
                after_agent_callback=collect_research_sources_callback,
            )

            # Report composer agent
            self.report_composer = LlmAgent(
                model=config.critic_model,
                name="report_composer_with_citations",
                include_contents="none",
                description="Transforms research data into a final, cited report.",
                instruction=self._get_report_composer_instruction(),
                output_key="final_cited_report",
                after_agent_callback=citation_replacement_callback,
            )

            # Build the research pipeline
            self.research_pipeline = SequentialAgent(
                name="research_pipeline",
                description="Executes comprehensive research with iterative refinement.",
                sub_agents=[
                    self.section_planner,
                    self.section_researcher,
                    LoopAgent(
                        name="iterative_refinement_loop",
                        max_iterations=config.max_search_iterations,
                        sub_agents=[
                            self.research_evaluator,
                            EscalationChecker(name="escalation_checker"),
                            self.enhanced_search_executor,
                        ],
                    ),
                    self.report_composer,
                ],
            )

            # Interactive planner agent
            self.interactive_planner_agent = LlmAgent(
                name="interactive_planner_agent",
                model=config.worker_model,
                description="Primary research assistant for collaborative planning and execution.",
                instruction=self._get_interactive_planner_instruction(),
                sub_agents=[self.research_pipeline],
                tools=[AgentTool(self.plan_generator)],
                output_key="research_plan",
            )

            # Root agent
            self.root_agent = self.interactive_planner_agent

        except (ImportError, AttributeError, ValueError) as e:
            logger.warning(f"Failed to initialize agents with full functionality: {e}")
            # Fallback to basic functionality
            self.root_agent = None

    def _get_plan_generator_instruction(self) -> str:
        """Get instruction for plan generator agent."""
        return f"""
        You are a research strategist. Create a high-level RESEARCH PLAN, not a summary.

        RESEARCH PLAN(SO FAR):
        {{ research_plan? }}

        **TASK CLASSIFICATION:**
        Each bullet point should start with a task type prefix:
        - **[RESEARCH]**: Information gathering, investigation, analysis, data collection
        - **[DELIVERABLE]**: Synthesizing information, creating structured outputs, final artifacts

        **INITIAL RULE:** Start with 5 action-oriented research goals classified as [RESEARCH].
        **REFINEMENT RULE:** Integrate feedback and mark changes with [MODIFIED], [NEW], or [IMPLIED].

        **TOOL USE LIMITATION:** Only use google_search for ambiguous or time-sensitive topics.
        Current date: {datetime.now().strftime("%Y-%m-%d")}
        """

    def _get_section_planner_instruction(self) -> str:
        """Get instruction for section planner agent."""
        return """
        You are an expert report architect. Using the research topic and plan from 'research_plan',
        design a logical structure for the final report.

        Create a markdown outline with 4-6 distinct sections covering the topic comprehensively.
        Do not include a "References" or "Sources" section - citations will be handled in-line.
        """

    def _get_section_researcher_instruction(self) -> str:
        """Get instruction for section researcher agent."""
        return """
        You are a research and synthesis agent. Execute the research plan with absolute fidelity.

        **Phase 1: Information Gathering ([RESEARCH] Tasks)**
        - Process every [RESEARCH] goal systematically
        - Generate 4-5 targeted search queries per goal
        - Execute all queries using google_search
        - Synthesize results into detailed summaries

        **Phase 2: Synthesis and Output Creation ([DELIVERABLE] Tasks)**
        - Process every [DELIVERABLE] goal after Phase 1 completion
        - Use ONLY summaries from Phase 1 (no new searches)
        - Generate specific artifacts as described in goals

        **Final Output:** Complete set of research summaries and deliverable artifacts.
        """

    def _get_research_evaluator_instruction(self) -> str:
        """Get instruction for research evaluator agent."""
        return f"""
        You are a quality assurance analyst evaluating research findings in 'section_research_findings'.

        **CRITICAL RULES:**
        1. Assume the research topic is correct - don't question the subject
        2. Assess ONLY the quality, depth, and completeness of research
        3. Evaluate: comprehensiveness, logical flow, credible sources, depth, clarity
        4. If suggesting follow-up queries, dive deeper into existing topic

        Be critical about research QUALITY. Grade "fail" for significant gaps.
        Current date: {datetime.now().strftime("%Y-%m-%d")}
        Response must be valid JSON matching the Feedback schema.
        """

    def _get_enhanced_search_executor_instruction(self) -> str:
        """Get instruction for enhanced search executor agent."""
        return """
        You are executing research refinement after a "fail" grade.

        1. Review 'research_evaluation' for feedback and required fixes
        2. Execute EVERY query in 'follow_up_queries' using google_search
        3. Synthesize new findings with existing 'section_research_findings'
        4. Output: complete, improved research findings
        """

    def _get_report_composer_instruction(self) -> str:
        """Get instruction for report composer agent."""
        return """
        Transform research data into a polished, professional, cited report.

        **INPUT DATA:**
        - Research Plan: {research_plan}
        - Research Findings: {section_research_findings}
        - Citation Sources: {sources}
        - Report Structure: {report_sections}

        **CITATION SYSTEM:**
        Use format: <cite source="src-ID_NUMBER" />

        Generate comprehensive report following the Report Structure outline.
        All citations must be in-line - no separate references section.
        """

    def _get_interactive_planner_instruction(self) -> str:
        """Get instruction for interactive planner agent."""
        return f"""
        You are a research planning assistant. Convert ANY user request into a research plan.

        **CRITICAL RULE:** Never answer directly or refuse requests. Always create a research plan first.

        **WORKFLOW:**
        1. **Plan:** Use plan_generator to create draft plan and present to user
        2. **Refine:** Incorporate user feedback until plan is approved
        3. **Execute:** On explicit approval, delegate to research_pipeline

        Current date: {datetime.now().strftime("%Y-%m-%d")}
        Your job: Plan, Refine, and Delegate. No direct research.
        """

    async def process_request(
        self, request: str, session_id: str | None = None, _streaming: bool = False
    ) -> dict[str, Any]:
        """Process a research request using the unified agent system.

        Args:
            request: Natural language research request
            session_id: Optional session ID for context
            _streaming: Whether to stream progress updates (currently unused)

        Returns:
            Dict containing the research results and metadata
        """
        try:
            if not self.root_agent:
                return {
                    "success": False,
                    "error": "Agent system not properly initialized",
                    "fallback_response": f"Unable to process request: {request}",
                }

            # Process request through the agent pipeline
            result = await self.root_agent.process(request)

            return {
                "success": True,
                "result": result,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "agent_type": "unified_agent",
            }

        except Exception:
            logger.exception("Failed to process request")
            return {
                "success": False,
                "error": "Request processing failed",
                "request": request,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
            }

    def get_capabilities(self) -> dict[str, Any]:
        """Get information about agent capabilities.

        Returns:
            Dict describing available capabilities
        """
        return {
            "research": {
                "web_search": google_search is not None,
                "citation_handling": True,
                "iterative_refinement": True,
                "structured_output": True,
            },
            "planning": {
                "autonomous_planning": True,
                "task_decomposition": True,
                "progress_tracking": True,
            },
            "output_formats": {
                "markdown_reports": True,
                "structured_data": True,
                "citations": True,
            },
            "agents": {
                "plan_generator": True,
                "section_planner": True,
                "researcher": True,
                "evaluator": True,
                "composer": True,
            },
        }
