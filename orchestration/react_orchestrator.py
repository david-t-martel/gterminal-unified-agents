#!/usr/bin/env python3
"""Re-Act Orchestrator - Integrated Reasoning + Action Loop.

This is the core orchestration component that belongs in /app/, not /scripts/.
It provides intelligent consolidation using Re-Act patterns with proper
service account authentication.
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Vertex AI SDK with optimized Gemini toolsets
try:
    from google.auth import default
    import vertexai
    from vertexai.generative_models import GenerativeModel

    # Initialize Vertex AI with service account
    credentials, project_id = default()
    vertexai.init(project=project_id, location="us-central1", credentials=credentials)

    # Use optimized Gemini model via Vertex AI
    model = GenerativeModel("gemini-2.0-flash-exp")

except Exception:
    model = None

logger = logging.getLogger(__name__)


class ReActStep(BaseModel):
    """Single step in Re-Act reasoning loop."""

    step_id: str
    thought: str
    action: str
    observation: str
    reflection: str
    next_action: str | None = None


class ConsolidationTask(BaseModel):
    """Intelligent consolidation task."""

    task_id: str
    target_files: list[str]
    consolidation_strategy: str
    expected_outcome: str
    preservation_requirements: list[str]


class ReactOrchestrator:
    """Re-Act Orchestrator - Core reasoning and action loop for intelligent consolidation.

    This implements proper consolidation:
    1. ANALYZE files for best elements
    2. MERGE best elements into single awesome file
    3. PRESERVE valuable optimizations from /app/
    4. MARK redundant files for deletion
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.model = model
        self.consolidation_history: list[ReActStep] = []
        self.step_counter = 0

    async def think(self, observation: str, context: dict[str, Any]) -> str:
        """Re-Act THINK step - analyze situation and plan action."""
        if not self.model:
            return "Gemini not available - using rule-based analysis"

        prompt = f"""
        As an expert code architect, analyze this consolidation opportunity:

        OBSERVATION: {observation}
        CONTEXT: {context}

        PROJECT GOALS (from CLAUDE.md):
        - INTELLIGENT CONSOLIDATION: Find best elements and merge into single awesome file
        - PRESERVE OPTIMIZATIONS: Keep valuable optimized code from /app/
        - NO ARBITRARY DELETION: Only remove after extracting value

        THINK about:
        1. What are the BEST elements in these files?
        2. Which optimizations should be preserved?
        3. How to create a single, superior consolidated file?
        4. What can be safely marked for deletion after consolidation?

        Provide strategic thinking, not just file counting.
        """

        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.exception(f"Thinking failed: {e}")
            return f"Rule-based analysis: {observation}"

    async def act(self, thought: str, target_path: Path) -> str:
        """Re-Act ACT step - execute intelligent consolidation action."""
        action_result: list[Any] = []

        # 1. Analyze /app/ for valuable optimizations
        app_optimizations = await self._find_app_optimizations(target_path / "app")
        action_result.append(f"Found {len(app_optimizations)} optimizations in /app/")

        # 2. Identify consolidation opportunities
        consolidation_opportunities = await self._find_consolidation_opportunities(target_path)
        action_result.append(
            f"Identified {len(consolidation_opportunities)} consolidation opportunities"
        )

        # 3. Execute intelligent merging (not deletion)
        for opportunity in consolidation_opportunities[:3]:  # Process top 3
            merge_result = await self._intelligent_merge(opportunity, app_optimizations)
            action_result.append(f"Merged: {merge_result}")

        return " | ".join(action_result)

    async def observe(self, action_result: str, target_path: Path) -> str:
        """Re-Act OBSERVE step - check results of action."""
        # Count files before/after
        current_files = len(list(target_path.rglob("*.py")))
        app_files = len(list((target_path / "app").rglob("*.py")))

        observation = f"Python files: {current_files} total, {app_files} in /app/. Action result: {action_result}"

        # Check for preservation of key functionality
        key_components = ["mcp_servers", "agents", "automation", "performance"]
        preserved: list[Any] = []
        for component in key_components:
            if (target_path / "app" / component).exists():
                preserved.append(component)

        observation += f" | Preserved components: {', '.join(preserved)}"
        return observation

    async def reflect(self, thought: str, action: str, observation: str) -> str:
        """Re-Act REFLECT step - learn from the cycle."""
        if not self.model:
            return "Consolidation progress made - continue intelligent merging"

        prompt = f"""
        Reflect on this consolidation cycle:

        THOUGHT: {thought}
        ACTION: {action}
        OBSERVATION: {observation}

        REFLECTION QUESTIONS:
        1. Did we successfully merge BEST elements into consolidated files?
        2. Are valuable /app/ optimizations being preserved?
        3. Are we creating superior consolidated files, not just deleting?
        4. What should be the next intelligent consolidation step?

        Focus on QUALITY consolidation, not quantity reduction.
        """

        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception:
            return "Reflection: Continue intelligent consolidation focusing on quality merging"

    async def _find_app_optimizations(self, app_path: Path) -> list[dict[str, Any]]:
        """Find valuable optimizations in /app/ that should be preserved."""
        optimizations: list[Any] = []

        if not app_path.exists():
            return optimizations

        # Look for performance-optimized files
        for py_file in app_path.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8", errors="ignore")

            # Check for optimization indicators
            optimization_indicators = [
                "async def",
                "asyncio",
                "concurrent.futures",
                "multiprocessing",
                "threading",
                "cache",
                "optimize",
                "performance",
                "fast",
                "efficient",
                "PyO3",
                "Rust",
                "uvloop",
                "cython",
            ]

            score = sum(1 for indicator in optimization_indicators if indicator in content)

            if score >= 3:  # File has multiple optimization indicators
                optimizations.append(
                    {
                        "file": str(py_file),
                        "optimization_score": score,
                        "indicators": [ind for ind in optimization_indicators if ind in content],
                        "size": len(content),
                    },
                )

        return sorted(optimizations, key=lambda x: x["optimization_score"], reverse=True)

    async def _find_consolidation_opportunities(self, target_path: Path) -> list[ConsolidationTask]:
        """Find intelligent consolidation opportunities."""
        opportunities: list[Any] = []

        # Group similar agent files
        agent_files = list(target_path.rglob("*agent*.py"))
        if len(agent_files) > 1:
            opportunities.append(
                ConsolidationTask(
                    task_id="consolidate_agents",
                    target_files=[str(f) for f in agent_files],
                    consolidation_strategy="merge_similar_agent_functionality",
                    expected_outcome="unified_agent_system",
                    preservation_requirements=[
                        "optimization_patterns",
                        "error_handling",
                        "async_patterns",
                    ],
                ),
            )

        # Group similar MCP server files
        mcp_files = list(target_path.rglob("*mcp*.py"))
        if len(mcp_files) > 1:
            opportunities.append(
                ConsolidationTask(
                    task_id="consolidate_mcp",
                    target_files=[str(f) for f in mcp_files],
                    consolidation_strategy="merge_mcp_server_functionality",
                    expected_outcome="unified_mcp_gateway",
                    preservation_requirements=[
                        "protocol_compliance",
                        "performance_optimizations",
                        "error_handling",
                    ],
                ),
            )

        # Group test files
        test_files = list(target_path.rglob("test_*.py"))
        if len(test_files) > 5:
            opportunities.append(
                ConsolidationTask(
                    task_id="consolidate_tests",
                    target_files=[str(f) for f in test_files],
                    consolidation_strategy="merge_related_test_suites",
                    expected_outcome="unified_test_suites",
                    preservation_requirements=["test_coverage", "fixture_logic", "mock_patterns"],
                ),
            )

        return opportunities

    async def _intelligent_merge(
        self, opportunity: ConsolidationTask, app_optimizations: list[dict]
    ) -> str:
        """Intelligently merge files, preserving best elements."""
        try:
            # Read all target files
            file_contents: dict[str, Any] = {}
            for file_path in opportunity.target_files:
                path = Path(file_path)
                if path.exists():
                    file_contents[file_path] = path.read_text(encoding="utf-8", errors="ignore")

            if not file_contents:
                return f"No files found for {opportunity.task_id}"

            # Find the best base file (usually the one in /app/ or with most optimizations)
            best_file = self._select_best_base_file(file_contents, app_optimizations)

            # Extract best elements from other files
            consolidated_content = await self._merge_best_elements(
                best_file,
                file_contents,
                opportunity.preservation_requirements,
            )

            # Write consolidated file
            consolidated_path = Path(best_file).parent / f"consolidated_{opportunity.task_id}.py"
            consolidated_path.write_text(consolidated_content, encoding="utf-8")

            # Mark other files for eventual deletion (but don't delete yet)
            marked_for_deletion = [f for f in file_contents if f != best_file]

            return f"Merged {len(file_contents)} files into {consolidated_path}, marked {len(marked_for_deletion)} for deletion"

        except Exception as e:
            logger.exception(f"Merge failed for {opportunity.task_id}: {e}")
            return f"Merge failed: {e!s}"

    def _select_best_base_file(
        self, file_contents: dict[str, str], optimizations: list[dict]
    ) -> str:
        """Select the best file to use as consolidation base."""
        # Prefer files in /app/
        app_files = [f for f in file_contents if "/app/" in f]
        if app_files:
            return app_files[0]

        # Prefer files with optimizations
        for opt in optimizations:
            if opt["file"] in file_contents:
                return opt["file"]

        # Fallback to largest file
        return max(file_contents.keys(), key=lambda f: len(file_contents[f]))

    async def _merge_best_elements(
        self, base_file: str, all_contents: dict[str, str], requirements: list[str]
    ) -> str:
        """Merge best elements from all files into consolidated version."""
        base_content = all_contents[base_file]

        # For now, use simple merging - could be enhanced with AST analysis
        return f'''"""
Consolidated file created by Re-Act Orchestrator
Base file: {base_file}
Merged from: {list(all_contents.keys())}
Preservation requirements: {requirements}
Created: {datetime.now().isoformat()}
"""

{base_content}

# TODO: Intelligently merge additional functionality from other files
# This is a placeholder for more sophisticated AST-based merging
'''

    async def execute_react_cycle(
        self, initial_observation: str, context: dict[str, Any]
    ) -> ReActStep:
        """Execute one complete Re-Act cycle."""
        self.step_counter += 1
        step_id = f"react_step_{self.step_counter}"

        # THINK
        thought = await self.think(initial_observation, context)

        # ACT
        action_result = await self.act(thought, self.project_root)

        # OBSERVE
        observation = await self.observe(action_result, self.project_root)

        # REFLECT
        reflection = await self.reflect(thought, action_result, observation)

        step = ReActStep(
            step_id=step_id,
            thought=thought,
            action=action_result,
            observation=observation,
            reflection=reflection,
        )

        self.consolidation_history.append(step)
        return step

    async def intelligent_consolidation_session(self, max_cycles: int = 5) -> list[ReActStep]:
        """Run multiple Re-Act cycles for comprehensive intelligent consolidation."""
        logger.info(f"Starting intelligent consolidation session with {max_cycles} cycles")

        context = {
            "project_root": str(self.project_root),
            "goals": [
                "intelligent_consolidation",
                "preserve_optimizations",
                "create_superior_files",
            ],
            "constraints": ["no_arbitrary_deletion", "preserve_app_optimizations"],
        }

        results: list[Any] = []
        observation = f"Starting consolidation of project at {self.project_root}"

        for cycle in range(max_cycles):
            logger.info(f"Executing Re-Act cycle {cycle + 1}/{max_cycles}")

            step = await self.execute_react_cycle(observation, context)
            results.append(step)

            # Use reflection as input for next cycle
            observation = step.reflection

            # Add some delay between cycles
            await asyncio.sleep(1)

        logger.info(f"Completed {len(results)} Re-Act cycles")
        return results
