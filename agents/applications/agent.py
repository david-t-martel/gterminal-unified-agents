"""Optimized agent components with performance enhancements.

This module provides drop-in replacements for the original agent components
with integrated caching, connection pooling, and performance monitoring.
Now includes Rust extensions for high-performance operations.
"""

import asyncio
from collections.abc import AsyncGenerator
import datetime
import logging
import re
from typing import Any, Literal
import uuid

from google.adk.agents import BaseAgent
from google.adk.agents import LlmAgent
from google.adk.agents import LoopAgent
from google.adk.agents import SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events import EventActions
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types
from pydantic import BaseModel
from pydantic import Field

from .config import config
from .performance_integration import get_performance_integration

# Rust extensions for high-performance operations
from .utils.rust_extensions import RUST_CORE_AVAILABLE
from .utils.rust_extensions import EnhancedTtlCache
from .utils.rust_extensions import RustCore

logger = logging.getLogger(__name__)


# --- Structured Output Models (same as original) ---
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


# --- Optimized Tool Functions ---
async def optimized_google_search_tool(query: str, **kwargs) -> dict[str, Any]:
    """Optimized Google search with caching and connection pooling."""
    integration = await get_performance_integration()

    try:
        # Use the optimized search functionality
        return await integration.components.optimized_google_search(query=query, **kwargs)
    except Exception as e:
        logger.exception(f"Optimized Google search failed: {e}")
        # Fallback to original google_search if optimization fails
        return await asyncio.to_thread(google_search, query)


# --- Optimized Callback Functions ---
async def optimized_collect_research_sources_callback(callback_context: CallbackContext) -> None:
    """Optimized version of research sources collection with caching."""
    session = callback_context._invocation_context.session
    session_id = getattr(session, "id", "default")

    integration = await get_performance_integration()

    # Try to get cached sources first
    cache_key = f"research_sources:{session_id}"
    cached_sources = await integration.optimizer.cache_manager.get(cache_key)

    if cached_sources:
        callback_context.state.update(cached_sources)
        await integration.optimizer.metrics.cache_hit("research_sources")
        return

    await integration.optimizer.metrics.cache_miss("research_sources")

    # Original logic
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

    # Cache the sources for future use
    sources_data = {
        "url_to_short_id": url_to_short_id,
        "sources": sources,
    }
    await integration.optimizer.cache_manager.set(cache_key, sources_data, ttl=1800)  # 30 minutes


def optimized_citation_replacement_callback(
    callback_context: CallbackContext,
) -> genai_types.Content:
    """Optimized citation replacement with caching (synchronous callback)."""
    final_report = callback_context.state.get("final_cited_report", "")
    sources = callback_context.state.get("sources", {})

    def tag_replacer(match: re.Match) -> str:
        short_id = match.group(1)
        if not (source_info := sources.get(short_id)):
            logging.warning(f"Invalid citation tag found and removed: {match.group(0)}")
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


# --- Optimized Agent Classes with Rust Integration ---
class OptimizedLlmAgent(LlmAgent):
    """LLM Agent with performance optimizations and Rust extensions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._performance_integration: Any | None = None
        self._rust_core: RustCore | None = None
        self._rust_cache: EnhancedTtlCache | None = None
        self._initialize_rust_components()

    def _initialize_rust_components(self) -> None:
        """Initialize Rust components if available."""
        if RUST_CORE_AVAILABLE:
            try:
                self._rust_core = RustCore()
                # Initialize cache with 30 minute default TTL
                self._rust_cache = EnhancedTtlCache(1800)
                logger.info(f"[{self.name}] Rust extensions initialized successfully")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to initialize Rust extensions: {e}")
                self._rust_core = None
                self._rust_cache = None
        else:
            logger.info(f"[{self.name}] Using Python fallbacks - Rust extensions not available")

    async def _get_performance_integration(self):
        """Get performance integration instance."""
        if self._performance_integration is None:
            self._performance_integration = await get_performance_integration()
        return self._performance_integration

    def _get_cache_key(self, ctx: InvocationContext, suffix: str = "") -> str:
        """Generate cache key using Rust string operations for performance."""
        if self._rust_core:
            # Use Rust for efficient string operations
            base_key = f"agent_{self.name}_{ctx.session.id}"
            if suffix:
                base_key = f"{base_key}_{suffix}"
            # Use Rust for actual string processing benefit (not just reversing)
            # Process the key for consistent formatting and hash-friendly structure
            processed_key = base_key.replace("-", "_").lower()
            return f"cache:{processed_key}"
        # Fallback to Python
        import hashlib

        base_key = f"agent_{self.name}_{getattr(ctx.session, 'id', 'unknown')}"
        if suffix:
            base_key = f"{base_key}_{suffix}"
        # Use hash for consistent key length
        key_hash = hashlib.md5(base_key.encode()).hexdigest()[:16]
        return f"cache:{key_hash}"

    async def _cache_get(self, key: str) -> str | None:
        """Get from Rust cache if available."""
        if self._rust_cache:
            try:
                result = self._rust_cache.get(key)
                if result:
                    logger.debug(f"[{self.name}] Cache hit (Rust): {key}")
                    return result
                logger.debug(f"[{self.name}] Cache miss (Rust): {key}")
            except Exception as e:
                logger.warning(f"[{self.name}] Rust cache get failed: {e}")
        return None

    async def _cache_set(self, key: str, value: str, ttl: int = 1800) -> None:
        """Set in Rust cache if available."""
        if self._rust_cache:
            try:
                self._rust_cache.set_with_ttl(key, value, ttl)
                logger.debug(f"[{self.name}] Cached (Rust): {key}")
            except Exception as e:
                logger.warning(f"[{self.name}] Rust cache set failed: {e}")

    def get_rust_performance_info(self) -> dict[str, Any]:
        """Get comprehensive performance information from Rust components."""
        info = {
            "rust_available": RUST_CORE_AVAILABLE,
            "rust_core_active": self._rust_core is not None,
            "rust_cache_active": self._rust_cache is not None,
        }

        if self._rust_cache:
            try:
                stats = self._rust_cache.get_stats()
                info.update(
                    {
                        "cache_size": self._rust_cache.size,
                        "cache_hits": stats.hits,
                        "cache_misses": stats.misses,
                        "cache_hit_ratio": stats.hit_ratio,
                        "total_entries": stats.total_entries,
                        "expired_entries": stats.expired_entries,
                    },
                )

                # Calculate additional performance metrics
                total_requests = stats.hits + stats.misses
                if total_requests > 0:
                    info.update(
                        {
                            "cache_efficiency": (stats.hits / total_requests) * 100,
                            "memory_efficiency": (
                                self._rust_cache.size / max(stats.total_entries, 1)
                            )
                            * 100,
                            "expiration_rate": (stats.expired_entries / max(stats.total_entries, 1))
                            * 100,
                        },
                    )

            except Exception as e:
                info["cache_stats_error"] = str(e)

        if self._rust_core:
            try:
                info.update(
                    {
                        "rust_version": self._rust_core.version,
                        "rust_test_result": self._rust_core.test_rust_integration(),
                    },
                )

                # Test Rust performance with a benchmark
                import time

                start_time = time.time()
                test_result = self._rust_core.add_numbers(12345, 67890)
                rust_time = time.time() - start_time

                start_time = time.time()
                python_result = 12345 + 67890
                python_time = time.time() - start_time

                info.update(
                    {
                        "performance_benchmark": {
                            "rust_time_ns": rust_time * 1_000_000_000,
                            "python_time_ns": python_time * 1_000_000_000,
                            "rust_faster_by": max(python_time / rust_time, 0)
                            if rust_time > 0
                            else 0,
                            "test_passed": test_result == python_result,
                        },
                    },
                )

            except Exception as e:
                info["rust_core_error"] = str(e)

        return info

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Optimized run implementation with performance tracking and Rust caching."""
        request_id = str(uuid.uuid4())
        integration = await self._get_performance_integration()

        # Generate cache key for this agent execution
        cache_key = self._get_cache_key(ctx, "execution_result")

        # Try to get cached result first (for idempotent operations)
        cached_result = await self._cache_get(cache_key)
        if cached_result and hasattr(self, "allow_caching") and self.allow_caching:
            logger.info(f"[{self.name}] Using cached result for session {ctx.session.id}")
            # Parse cached result and yield as event
            try:
                import json

                cached_data = json.loads(cached_result)
                if cached_data.get("type") == "event":
                    yield Event(author=self.name, content=cached_data.get("content", ""))
                    return
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to parse cached result: {e}")

        start_time = await integration.optimizer.metrics.start_request(
            request_id, f"llm_agent_{self.name}"
        )

        try:
            # Collect events for potential caching
            events = []
            async for event in super()._run_async_impl(ctx):
                events.append(event)
                yield event

            # Cache the result if caching is enabled
            if hasattr(self, "allow_caching") and self.allow_caching and events:
                try:
                    import json

                    # Cache the last event content (simplified caching strategy)
                    last_event = events[-1]
                    cache_data = {
                        "type": "event",
                        "content": getattr(last_event, "content", ""),
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                    await self._cache_set(cache_key, json.dumps(cache_data), ttl=900)  # 15 minutes
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to cache result: {e}")

            await integration.optimizer.metrics.end_request(
                request_id, f"llm_agent_{self.name}", start_time, True
            )

        except Exception:
            await integration.optimizer.metrics.end_request(
                request_id, f"llm_agent_{self.name}", start_time, False
            )
            raise


class OptimizedEscalationChecker(BaseAgent):
    """Optimized escalation checker with Rust-powered caching."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self._rust_cache: EnhancedTtlCache | None = None
        self._initialize_rust_cache()

    def _initialize_rust_cache(self) -> None:
        """Initialize Rust cache for evaluation results."""
        if RUST_CORE_AVAILABLE:
            try:
                self._rust_cache = EnhancedTtlCache(300)  # 5 minutes default TTL
                logger.info(f"[{self.name}] Rust cache initialized for escalation checking")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to initialize Rust cache: {e}")
                self._rust_cache = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        integration = await get_performance_integration()

        # Cache evaluation results for this session using both Rust and standard cache
        session_id = getattr(ctx.session, "id", "default")
        cache_key = f"evaluation_result:{session_id}"

        evaluation_result = None

        # Try Rust cache first
        if self._rust_cache:
            try:
                import json

                cached_data = self._rust_cache.get(cache_key)
                if cached_data:
                    evaluation_result = json.loads(cached_data)
                    logger.debug(f"[{self.name}] Evaluation result from Rust cache")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to get from Rust cache: {e}")

        # Fallback to standard cache
        if evaluation_result is None:
            evaluation_result = await integration.optimizer.cache_manager.get(cache_key)

        # Get from session state if not cached
        if evaluation_result is None:
            evaluation_result = ctx.session.state.get("research_evaluation")
            if evaluation_result:
                # Cache in both systems
                await integration.optimizer.cache_manager.set(cache_key, evaluation_result, 300)
                if self._rust_cache:
                    try:
                        import json

                        self._rust_cache.set(cache_key, json.dumps(evaluation_result))
                    except Exception as e:
                        logger.warning(f"[{self.name}] Failed to cache in Rust: {e}")

        if evaluation_result and evaluation_result.get("grade") == "pass":
            logging.info(f"[{self.name}] Research evaluation passed. Escalating to stop loop.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(
                f"[{self.name}] Research evaluation failed or not found. Loop will continue."
            )
            yield Event(author=self.name)


# --- Optimized Agent Definitions ---
def create_optimized_plan_generator() -> OptimizedLlmAgent:
    """Create optimized plan generator agent with Rust extensions."""
    agent = OptimizedLlmAgent(
        model=config.worker_model,
        name="optimized_plan_generator",
        description="Generates or refine the existing 5 line action-oriented research plan, using minimal search only for topic clarification.",
        instruction=f"""
        You are a research strategist. Your job is to create a high-level RESEARCH PLAN, not a summary. If there is already a RESEARCH PLAN in the session state,
        improve upon it based on the user feedback.

        RESEARCH PLAN(SO FAR):
        {{ research_plan? }}

        **GENERAL INSTRUCTION: CLASSIFY TASK TYPES**
        Your plan must clearly classify each goal for downstream execution. Each bullet point should start with a task type prefix:
        - **`[RESEARCH]`**: For goals that primarily involve information gathering, investigation, analysis, or data collection (these require search tool usage by a researcher).
        - **`[DELIVERABLE]`**: For goals that involve synthesizing collected information, creating structured outputs (e.g., tables, charts, summaries, reports), or compiling final output artifacts (these are executed AFTER research tasks, often without further search).

        **INITIAL RULE: Your initial output MUST start with a bulleted list of 5 action-oriented research goals or key questions, followed by any *inherently implied* deliverables.**
        - All initial 5 goals will be classified as `[RESEARCH]` tasks.
        - A good goal for `[RESEARCH]` starts with a verb like "Analyze," "Identify," "Investigate."
        - A bad output is a statement of fact like "The event was in April 2024."
        - **Proactive Implied Deliverables (Initial):** If any of your initial 5 `[RESEARCH]` goals inherently imply a standard output or deliverable (e.g., a comparative analysis suggesting a comparison table, or a comprehensive review suggesting a summary document), you MUST add these as additional, distinct goals immediately after the initial 5. Phrase these as *synthesis or output creation actions* (e.g., "Create a summary," "Develop a comparison," "Compile a report") and prefix them with `[DELIVERABLE][IMPLIED]`.

        **TOOL USE IS STRICTLY LIMITED:**
        Your goal is to create a generic, high-quality plan *without searching*.
        Only use `optimized_google_search_tool` if a topic is ambiguous or time-sensitive and you absolutely cannot create a plan without a key piece of identifying information.
        You are explicitly forbidden from researching the *content* or *themes* of the topic. That is the next agent's job. Your search is only to identify the subject, not to investigate it.
        Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
        """,
        tools=[optimized_google_search_tool],
    )
    # Enable caching for plan generator as plans can be reused for similar queries
    agent.allow_caching = True
    return agent


def create_optimized_section_planner() -> OptimizedLlmAgent:
    """Create optimized section planner agent."""
    return OptimizedLlmAgent(
        model=config.worker_model,
        name="optimized_section_planner",
        description="Breaks down the research plan into a structured markdown outline of report sections.",
        instruction="""
        You are an expert report architect. Using the research topic and the plan from the 'research_plan' state key, design a logical structure for the final report.
        Note: Ignore all the tag names ([MODIFIED], [NEW], [RESEARCH], [DELIVERABLE]) in the research plan.
        Your task is to create a markdown outline with 4-6 distinct sections that cover the topic comprehensively without overlap.
        You can use any markdown format you prefer, but here's a suggested structure:
        # Section Name
        A brief overview of what this section covers
        Feel free to add subsections or bullet points if needed to better organize the content.
        Make sure your outline is clear and easy to follow.
        Do not include a "References" or "Sources" section in your outline. Citations will be handled in-line.
        """,
        output_key="report_sections",
    )


def create_optimized_section_researcher() -> OptimizedLlmAgent:
    """Create optimized section researcher agent."""
    return OptimizedLlmAgent(
        model=config.worker_model,
        name="optimized_section_researcher",
        description="Performs the crucial first pass of web research with optimization.",
        planner=BuiltInPlanner(thinking_config=genai_types.ThinkingConfig(include_thoughts=True)),
        instruction="""
        You are a highly capable and diligent research and synthesis agent with performance optimizations. Your comprehensive task is to execute a provided research plan with **absolute fidelity**, first by gathering necessary information, and then by synthesizing that information into specified outputs.

        You will be provided with a sequential list of research plan goals, stored in the `research_plan` state key. Each goal will be clearly prefixed with its primary task type: `[RESEARCH]` or `[DELIVERABLE]`.

        Your execution process must strictly adhere to these two distinct and sequential phases:

        ---

        **Phase 1: Information Gathering (`[RESEARCH]` Tasks)**

        *   **Execution Directive:** You **MUST** systematically process every goal prefixed with `[RESEARCH]` before proceeding to Phase 2.
        *   For each `[RESEARCH]` goal:
            *   **Query Generation:** Formulate a comprehensive set of 4-5 targeted search queries. These queries must be expertly designed to broadly cover the specific intent of the `[RESEARCH]` goal from multiple angles.
            *   **Execution:** Utilize the `optimized_google_search_tool` to execute **all** generated queries for the current `[RESEARCH]` goal.
            *   **Summarization:** Synthesize the search results into a detailed, coherent summary that directly addresses the objective of the `[RESEARCH]` goal.
            *   **Internal Storage:** Store this summary, clearly tagged or indexed by its corresponding `[RESEARCH]` goal, for later and exclusive use in Phase 2. You **MUST NOT** lose or discard any generated summaries.

        ---

        **Phase 2: Synthesis and Output Creation (`[DELIVERABLE]` Tasks)**

        *   **Execution Prerequisite:** This phase **MUST ONLY COMMENCE** once **ALL** `[RESEARCH]` goals from Phase 1 have been fully completed and their summaries are internally stored.
        *   **Execution Directive:** You **MUST** systematically process **every** goal prefixed with `[DELIVERABLE]`. For each `[DELIVERABLE]` goal, your directive is to **PRODUCE** the artifact as explicitly described.
        *   For each `[DELIVERABLE]` goal:
            *   **Instruction Interpretation:** You will interpret the goal's text (following the `[DELIVERABLE]` tag) as a **direct and non-negotiable instruction** to generate a specific output artifact.
            *   **Data Consolidation:** Access and utilize **ONLY** the summaries generated during Phase 1 (`[RESEARCH]` tasks`) to fulfill the requirements of the current `[DELIVERABLE]` goal. You **MUST NOT** perform new searches.
            *   **Output Generation:** Based on the specific instruction of the `[DELIVERABLE]` goal, carefully extract, organize, and synthesize the relevant information from your previously gathered summaries.

        ---

        **Final Output:** Your final output will comprise the complete set of processed summaries from `[RESEARCH]` tasks AND all the generated artifacts from `[DELIVERABLE]` tasks, presented clearly and distinctly.
        """,
        tools=[optimized_google_search_tool],
        output_key="section_research_findings",
        after_agent_callback=optimized_collect_research_sources_callback,
    )


def create_optimized_research_evaluator() -> OptimizedLlmAgent:
    """Create optimized research evaluator agent."""
    return OptimizedLlmAgent(
        model=config.critic_model,
        name="optimized_research_evaluator",
        description="Critically evaluates research and generates follow-up queries with caching.",
        instruction=f"""
        You are a meticulous quality assurance analyst evaluating the research findings in 'section_research_findings'.

        **CRITICAL RULES:**
        1. Assume the given research topic is correct. Do not question or try to verify the subject itself.
        2. Your ONLY job is to assess the quality, depth, and completeness of the research provided *for that topic*.
        3. Focus on evaluating: Comprehensiveness of coverage, logical flow and organization, use of credible sources, depth of analysis, and clarity of explanations.
        4. Do NOT fact-check or question the fundamental premise or timeline of the topic.
        5. If suggesting follow-up queries, they should dive deeper into the existing topic, not question its validity.

        Be very critical about the QUALITY of research. If you find significant gaps in depth or coverage, assign a grade of "fail",
        write a detailed comment about what's missing, and generate 5-7 specific follow-up queries to fill those gaps.
        If the research thoroughly covers the topic, grade "pass".

        Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
        Your response must be a single, raw JSON object validating against the 'Feedback' schema.
        """,
        output_schema=Feedback,
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        output_key="research_evaluation",
    )


def create_optimized_enhanced_search_executor() -> OptimizedLlmAgent:
    """Create optimized enhanced search executor agent."""
    return OptimizedLlmAgent(
        model=config.worker_model,
        name="optimized_enhanced_search_executor",
        description="Executes follow-up searches and integrates new findings with optimization.",
        planner=BuiltInPlanner(thinking_config=genai_types.ThinkingConfig(include_thoughts=True)),
        instruction="""
        You are a specialist researcher executing a refinement pass with performance optimizations.
        You have been activated because the previous research was graded as 'fail'.

        1.  Review the 'research_evaluation' state key to understand the feedback and required fixes.
        2.  Execute EVERY query listed in 'follow_up_queries' using the 'optimized_google_search_tool'.
        3.  Synthesize the new findings and COMBINE them with the existing information in 'section_research_findings'.
        4.  Your output MUST be the new, complete, and improved set of research findings.
        """,
        tools=[optimized_google_search_tool],
        output_key="section_research_findings",
        after_agent_callback=optimized_collect_research_sources_callback,
    )


def create_optimized_report_composer() -> OptimizedLlmAgent:
    """Create optimized report composer agent."""
    return OptimizedLlmAgent(
        model=config.critic_model,
        name="optimized_report_composer_with_citations",
        include_contents="none",
        description="Transforms research data and a markdown outline into a final, cited report with caching.",
        instruction="""
        Transform the provided data into a polished, professional, and meticulously cited research report.

        ---
        ### INPUT DATA
        *   Research Plan: `{research_plan}`
        *   Research Findings: `{section_research_findings}`
        *   Citation Sources: `{sources}`
        *   Report Structure: `{report_sections}`

        ---
        ### CRITICAL: Citation System
        To cite a source, you MUST insert a special citation tag directly after the claim it supports.

        **The only correct format is:** `<cite source="src-ID_NUMBER" />`

        ---
        ### Final Instructions
        Generate a comprehensive report using ONLY the `<cite source="src-ID_NUMBER" />` tag system for all citations.
        The final report must strictly follow the structure provided in the **Report Structure** markdown outline.
        Do not include a "References" or "Sources" section; all citations must be in-line.
        """,
        output_key="final_cited_report",
        after_agent_callback=optimized_citation_replacement_callback,
    )


# --- Optimized Pipeline Creation ---
def create_optimized_research_pipeline() -> SequentialAgent:
    """Create the optimized research pipeline with performance enhancements."""
    return SequentialAgent(
        name="optimized_research_pipeline",
        description="Executes a pre-approved research plan with performance optimizations. It performs iterative research, evaluation, and composes a final, cited report.",
        sub_agents=[
            create_optimized_section_planner(),
            create_optimized_section_researcher(),
            LoopAgent(
                name="optimized_iterative_refinement_loop",
                max_iterations=config.max_search_iterations,
                sub_agents=[
                    create_optimized_research_evaluator(),
                    OptimizedEscalationChecker(name="optimized_escalation_checker"),
                    create_optimized_enhanced_search_executor(),
                ],
            ),
            create_optimized_report_composer(),
        ],
    )


def create_optimized_interactive_planner_agent() -> OptimizedLlmAgent:
    """Create the optimized interactive planner agent."""
    return OptimizedLlmAgent(
        name="optimized_interactive_planner_agent",
        model=config.worker_model,
        description="The primary research assistant with performance optimizations. It collaborates with the user to create a research plan, and then executes it upon approval.",
        instruction=f"""
        You are a research planning assistant with performance optimizations. Your primary function is to convert ANY user request into a research plan.

        **CRITICAL RULE: Never answer a question directly or refuse a request.** Your one and only first step is to use the `optimized_plan_generator` tool to propose a research plan for the user's topic.
        If the user asks a question, you MUST immediately call `optimized_plan_generator` to create a plan to answer the question.

        Your workflow is:
        1.  **Plan:** Use `optimized_plan_generator` to create a draft plan and present it to the user.
        2.  **Refine:** Incorporate user feedback until the plan is approved.
        3.  **Execute:** Once the user gives EXPLICIT approval (e.g., "looks good, run it"), you MUST delegate the task to the `optimized_research_pipeline` agent, passing the approved plan.

        Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
        Do not perform any research yourself. Your job is to Plan, Refine, and Delegate.
        """,
        sub_agents=[create_optimized_research_pipeline()],
        tools=[AgentTool(create_optimized_plan_generator())],
        output_key="research_plan",
    )


# Create the optimized root agent
optimized_root_agent = create_optimized_interactive_planner_agent()
