"""Core ReAct (Reasoning and Acting) engine for the terminal interface.

This module implements the ReAct paradigm for AI agent reasoning, where the agent
alternates between thinking about the problem, taking actions, and observing results
until it reaches a satisfactory conclusion.

The engine coordinates with the Gemini unified server for reasoning and
executes actions through various agent services.
"""

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from .react_types import Action
from .react_types import ActionType
from .react_types import Observation
from .react_types import ReActContext
from .react_types import ReActResult
from .react_types import ReActStatus
from .react_types import ReActStep
from .react_types import StepType
from .rust_terminal_ops import TerminalRustOps


class ReActEngine:
    def __init__(
        self,
        gemini_server_url: str = "http://localhost:8100",
        max_iterations: int = 10,
        timeout_per_step: int = 120,
        cache_enabled: bool = True,
    ) -> None:
        """Initialize the ReAct engine.

        Args:
            gemini_server_url: URL of the Gemini unified server
            max_iterations: Maximum number of ReAct iterations
            timeout_per_step: Timeout in seconds for each step
            cache_enabled: Whether to enable caching for performance

        """
        self.logger = logging.getLogger(__name__)
        self.gemini_server_url = gemini_server_url.rstrip("/")
        self.max_iterations = max_iterations
        self.timeout_per_step = timeout_per_step
        self.cache_enabled = cache_enabled

        # Initialize Rust operations for high performance
        self.rust_ops = TerminalRustOps() if cache_enabled else None

        # HTTP client for Gemini server communication
        self.http_client = httpx.AsyncClient(timeout=timeout_per_step)

        # Agent service mapping
        self.agent_services = {
            ActionType.ANALYZE_CODE: "code-reviewer",
            ActionType.GENERATE_CODE: "code-generation",
            ActionType.REVIEW_CODE: "code-reviewer",
            ActionType.WORKSPACE_ANALYZE: "workspace-analyzer",
            ActionType.DOCUMENTATION: "documentation-generator",
            ActionType.ARCHITECT: "master-architect",
            ActionType.FILE_OPERATION: "file-operations",
            ActionType.SYSTEM_COMMAND: "system-command",
        }

        # ReAct prompt templates
        self.reasoning_template = """
You are an advanced AI agent using the ReAct (Reasoning and Acting) paradigm to solve complex tasks.

Task: {task}

Current Context:
{context_history}

Available Actions:
- analyze_code: Analyze code for issues, patterns, or improvements
- generate_code: Generate new code based on requirements
- review_code: Perform code review for security and quality
- workspace_analyze: Analyze project structure and architecture
- file_operation: Read, write, or modify files
- system_command: Execute system commands
- search: Search for information or files
- documentation: Generate or update documentation
- architect: Provide architectural recommendations

Based on the task and current context, what should be your next step?

Respond in this format:
Thought: [Your reasoning about what to do next]
Action: [The action type and parameters in JSON format]

OR if you have a final answer:
Thought: [Your final reasoning]
Final Answer: [Your complete answer to the task]
"""

    async def execute_react_loop(
        self, task: str, context: ReActContext | None = None
    ) -> ReActResult:
        """Execute the main ReAct reasoning and acting loop.

        Args:
            task: The task to be solved
            context: Optional existing context to continue from

        Returns:
            ReActResult containing the final result and execution details

        """
        start_time = time.time()

        # Initialize or reuse context
        if context is None:
            context = ReActContext(
                task=task,
                max_iterations=self.max_iterations,
                status=ReActStatus.INITIALIZED,
            )

        self.logger.info(f"Starting ReAct loop for task: {task[:100]}...")

        try:
            # Cache initial context
            if self.rust_ops:
                await self.rust_ops.cache_context(context)

            context.status = ReActStatus.THINKING
            final_answer = None

            while context.current_iteration < context.max_iterations:
                context.current_iteration += 1
                self.logger.info(
                    f"ReAct iteration {context.current_iteration}/{context.max_iterations}"
                )

                # Generate reasoning step
                thought_step = await self._generate_thought(context)
                context.add_step(thought_step)

                # Check if we have a final answer
                if thought_step.step_type == StepType.FINAL_ANSWER:
                    final_answer = thought_step.content
                    context.status = ReActStatus.COMPLETED
                    break

                # Execute action if specified
                if thought_step.action:
                    context.status = ReActStatus.ACTING
                    action_result = await self._execute_action(thought_step.action, context)

                    # Create observation step
                    observation_step = ReActStep(
                        step_type=StepType.OBSERVATION,
                        content=(
                            str(action_result.result)
                            if action_result.success
                            else f"Error: {action_result.error}"
                        ),
                        observation=action_result,
                    )
                    context.add_step(observation_step)
                    context.status = ReActStatus.OBSERVING

                    # Update context with new information
                    if action_result.success and action_result.metadata:
                        context.context_data.update(action_result.metadata)

                # Cache updated context
                if self.rust_ops:
                    await self.rust_ops.cache_context(context)

                # Brief pause between iterations
                await asyncio.sleep(0.1)

            # Handle completion or timeout
            if context.current_iteration >= context.max_iterations and final_answer is None:
                context.status = ReActStatus.TERMINATED
                final_answer = "Maximum iterations reached without finding a final answer."

            execution_time = time.time() - start_time

            # Create result
            result = ReActResult(
                context=context,
                final_answer=final_answer,
                success=context.status == ReActStatus.COMPLETED,
                total_steps=len(context.steps),
                execution_time=execution_time,
                performance_metrics={
                    "iterations_used": context.current_iteration,
                    "avg_time_per_iteration": (
                        execution_time / context.current_iteration
                        if context.current_iteration > 0
                        else 0
                    ),
                    "cache_enabled": self.cache_enabled,
                    "rust_extensions": self.rust_ops is not None,
                },
            )

            # Cache final result
            if self.rust_ops:
                await self.rust_ops.cache_result(result)

            self.logger.info(
                f"ReAct loop completed in {execution_time:.2f}s with {context.current_iteration} iterations",
            )
            return result

        except Exception as e:
            self.logger.error(f"ReAct loop failed: {e}", exc_info=True)
            context.status = ReActStatus.FAILED

            return ReActResult(
                context=context,
                success=False,
                error=str(e),
                total_steps=len(context.steps),
                execution_time=time.time() - start_time,
            )

    async def _generate_thought(self, context: ReActContext) -> ReActStep:
        """Generate the next reasoning step using the Gemini server.

        Args:
            context: Current ReAct context

        Returns:
            ReActStep containing the thought and potentially an action

        """
        # Prepare context history
        context_history = context.get_step_history()

        # Create prompt
        prompt = self.reasoning_template.format(
            task=context.task,
            context_history=context_history if context_history else "No previous steps",
        )

        try:
            # Send reasoning request to Gemini server
            response = await self.http_client.post(
                f"{self.gemini_server_url}/task",
                json={
                    "task_type": "reasoning",
                    "instruction": prompt,
                    "context": {
                        "session_id": context.session_id,
                        "iteration": context.current_iteration,
                        "previous_steps": len(context.steps),
                    },
                },
            )
            response.raise_for_status()

            result = response.json()
            reasoning_text = result.get("result", "")

            # Parse the response to extract thought and action
            thought, action = await self._parse_reasoning_response(reasoning_text)

            # Create step
            return ReActStep(
                step_type=(
                    StepType.FINAL_ANSWER if "Final Answer:" in reasoning_text else StepType.THOUGHT
                ),
                content=thought,
                action=action,
            )

        except Exception as e:
            self.logger.exception(f"Failed to generate thought: {e}")
            # Fallback thought
            return ReActStep(
                step_type=StepType.THOUGHT,
                content=f"Error generating reasoning: {e}. Will attempt to continue.",
            )

    async def _parse_reasoning_response(self, response: str) -> tuple[str, Action | None]:
        """Parse the reasoning response to extract thought and action.

        Args:
            response: Raw response text from Gemini

        Returns:
            Tuple of (thought_text, action_or_none)

        """
        lines = response.strip().split("\n")
        thought = ""
        action = None

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("Thought:"):
                thought = line[8:].strip()
                # Look for multi-line thoughts
                i += 1
                while i < len(lines) and not lines[i].strip().startswith(
                    ("Action:", "Final Answer:")
                ):
                    thought += "\n" + lines[i].strip()
                    i += 1
                continue

            if line.startswith("Action:"):
                action_text = line[7:].strip()
                # Look for multi-line actions
                i += 1
                while i < len(lines) and not lines[i].strip().startswith(
                    ("Thought:", "Final Answer:")
                ):
                    action_text += "\n" + lines[i].strip()
                    i += 1

                # Parse action JSON
                try:
                    if self.rust_ops:
                        action_data = await self.rust_ops.parse_json(action_text)
                    else:
                        action_data = json.loads(action_text)

                    action = Action(
                        action_type=ActionType(action_data.get("type", "custom")),
                        parameters=action_data.get("parameters", {}),
                        description=action_data.get("description", ""),
                        timeout=action_data.get("timeout"),
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to parse action JSON: {e}")
                continue

            if line.startswith("Final Answer:"):
                # This is a final answer, not a thought
                final_answer = line[13:].strip()
                i += 1
                while i < len(lines):
                    final_answer += "\n" + lines[i].strip()
                    i += 1
                return final_answer, None

            i += 1

        return thought, action

    async def _execute_action(self, action: Action, context: ReActContext) -> Observation:
        """Execute an action through the appropriate agent service.

        Args:
            action: Action to execute
            context: Current ReAct context

        Returns:
            Observation containing the execution result

        """
        start_time = time.time()
        action_id = f"action_{len(context.steps)}"

        try:
            # Determine the appropriate service
            service_name = self.agent_services.get(action.action_type, "general")

            # Prepare request
            request_data = {
                "task_type": service_name,
                "instruction": action.description,
                "parameters": action.parameters,
                "context": {
                    "session_id": context.session_id,
                    "action_id": action_id,
                    "timeout": action.timeout or self.timeout_per_step,
                },
            }

            # Execute through Gemini server
            response = await self.http_client.post(
                f"{self.gemini_server_url}/task", json=request_data
            )
            response.raise_for_status()

            result = response.json()
            execution_time = time.time() - start_time

            return Observation(
                action_id=action_id,
                result=result.get("result"),
                success=result.get("success", True),
                error=result.get("error"),
                execution_time=execution_time,
                metadata=result.get("metadata", {}),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.exception(f"Action execution failed: {e}")

            return Observation(
                action_id=action_id,
                result=None,
                success=False,
                error=str(e),
                execution_time=execution_time,
            )

    async def load_session(self, session_id: str) -> ReActContext | None:
        """Load a ReAct session from cache.

        Args:
            session_id: Session ID to load

        Returns:
            ReActContext if found, None otherwise

        """
        if not self.rust_ops:
            return None

        return await self.rust_ops.load_context(session_id)

    async def get_session_history(self, session_id: str) -> list[ReActStep] | None:
        """Get the step history for a session.

        Args:
            session_id: Session ID

        Returns:
            List of steps if session exists, None otherwise

        """
        context = await self.load_session(session_id)
        return context.steps if context else None

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.http_client:
            await self.http_client.aclose()

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            "engine": {
                "max_iterations": self.max_iterations,
                "timeout_per_step": self.timeout_per_step,
                "cache_enabled": self.cache_enabled,
                "gemini_server_url": self.gemini_server_url,
            },
        }

        if self.rust_ops:
            rust_metrics = await self.rust_ops.get_performance_metrics()
            metrics["rust_operations"] = rust_metrics

        return metrics
