"""Simplified ReAct Engine for Gemini CLI.

Implements the ReAct (Reason, Act, Observe) pattern using Google Gemini
for reasoning and a simple tool system for actions.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import logging
from typing import Any

from ..tools.registry import ToolRegistry
from .client import GeminiClient

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Types of steps in the ReAct process."""

    REASON = "reason"
    ACT = "act"
    OBSERVE = "observe"
    COMPLETE = "complete"


@dataclass
class Step:
    """Represents a single step in the ReAct process."""

    type: StepType
    description: str
    tool_name: str | None = None
    tool_params: dict[str, Any] | None = None
    result: Any | None = None
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SimpleReactEngine:
    """Lightweight ReAct engine for CLI usage."""

    def __init__(self, model_name: str = "gemini-2.0-flash-exp") -> None:
        """Initialize the ReAct engine.

        Args:
            model_name: The Gemini model to use
        """
        self.client = GeminiClient(model_name)
        self.tool_registry = ToolRegistry()
        self.steps: list[Step] = []

        # Register essential tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default tools."""
        from ..tools.code_analysis import CodeAnalysisTool
        from ..tools.filesystem import FilesystemTool

        self.tool_registry.register("filesystem", FilesystemTool())
        self.tool_registry.register("code_analysis", CodeAnalysisTool())

    async def process(self, request: str) -> str:
        """Process a request using the ReAct pattern.

        Args:
            request: The user request

        Returns:
            The final response
        """
        self.steps = []  # Reset steps

        # Step 1: Reason about the request
        plan = await self._reason(request)

        # Step 2: Execute actions
        results = []
        for action in plan.get("actions", []):
            result = await self._act(action)
            results.append(result)

        # Step 3: Complete with observation
        return await self._complete(request, results)

    async def _reason(self, request: str) -> dict[str, Any]:
        """Reason about the request and create a plan.

        Args:
            request: The user request

        Returns:
            A plan with actions to take
        """
        step = Step(type=StepType.REASON, description="Planning actions")
        self.steps.append(step)

        # Get available tools
        available_tools = list(self.tool_registry.list_tools().keys())
        tools_list = [
            self.tool_registry.get_tool(name)
            for name in available_tools
            if self.tool_registry.get_tool(name)
        ]

        reasoning_prompt = f"""
Analyze this request and create a plan using available tools: {available_tools}

Request: {request}

You have access to the following tools with function calling:
- filesystem: File and directory operations (read_file, write_file, list_files, search_files, search_content)
- code_analysis: Code analysis operations (analyze_complexity, check_style, find_issues)

You can call these tools directly if needed. Otherwise, respond with a JSON plan in this format:
{{
    "reasoning": "Your analysis of what needs to be done",
    "actions": [
        {{
            "tool": "tool_name",
            "description": "what this action does",
            "params": {{"param1": "value1"}}
        }}
    ]
}}

Keep it simple and focused. Use only the tools that are necessary.
"""

        try:
            # Use process_with_tools to enable function calling
            response = await self.client.process_with_tools(reasoning_prompt, tools_list)

            # Check if function calls were made
            if "Function calls made:" in response:
                logger.info("Function calling was triggered during reasoning")
                # Extract the plan from the response
                plan = {
                    "reasoning": "Function calling was used directly",
                    "actions": [],  # No manual actions needed if functions were called
                }
            else:
                plan = self._extract_json_from_response(response)

            step.result = plan
            logger.info(f"Plan created with {len(plan.get('actions', []))} actions")
            return plan
        except Exception as e:
            logger.exception("Reasoning failed")
            step.result = {"error": str(e)}
            return {"actions": []}

    async def _act(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute an action using the specified tool.

        Args:
            action: The action to execute

        Returns:
            The action result
        """
        tool_name = action.get("tool")
        params = action.get("params", {})

        step = Step(
            type=StepType.ACT,
            description=action.get("description", "Executing action"),
            tool_name=tool_name,
            tool_params=params,
        )
        self.steps.append(step)

        try:
            if not tool_name or tool_name not in self.tool_registry.list_tools():
                error = f"Tool '{tool_name}' not available"
                step.result = {"error": error}
                return {"error": error}

            tool = self.tool_registry.get_tool(tool_name)
            if tool is None:
                error = f"Tool '{tool_name}' not found"
                step.result = {"error": error}
                return {"error": error}
            result = await tool.execute(params)
            step.result = result
            return result

        except Exception as e:
            error = f"Action failed: {e}"
            step.result = {"error": error}
            logger.exception(error)
            return {"error": error}

    async def _complete(self, request: str, results: list[dict[str, Any]]) -> str:
        """Complete the process by synthesizing results.

        Args:
            request: The original request
            results: Results from all actions

        Returns:
            The final response
        """
        step = Step(type=StepType.COMPLETE, description="Synthesizing results")
        self.steps.append(step)

        completion_prompt = f"""
Original request: {request}

Action results: {json.dumps(results, indent=2)}

Provide a clear, helpful response based on these results.
If there were errors, explain them and suggest solutions.
Be concise but complete.
"""

        try:
            response = await self.client.process(completion_prompt)
            step.result = response
            return response
        except Exception as e:
            error = f"Completion failed: {e}"
            step.result = error
            logger.exception(error)
            return error

    def _extract_json_from_response(self, response: str) -> dict[str, Any]:
        """Extract JSON from LLM response.

        Args:
            response: The LLM response text

        Returns:
            Extracted JSON data
        """
        try:
            # Try to find JSON in the response
            start = response.find("{")
            end = response.rfind("}") + 1

            if start != -1 and end != 0:
                json_str = response[start:end]
                parsed_json: dict[str, Any] = json.loads(json_str)
                return parsed_json
            else:
                # Fallback: return simple structure
                return {"reasoning": response, "actions": []}
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from response")
            return {"reasoning": response, "actions": []}

    def get_execution_summary(self) -> dict[str, Any]:
        """Get a summary of the execution steps.

        Returns:
            Summary of steps and their results
        """
        return {
            "total_steps": len(self.steps),
            "steps": [
                {
                    "type": step.type.value,
                    "description": step.description,
                    "tool": step.tool_name,
                    "success": step.result and "error" not in str(step.result),
                    "timestamp": step.timestamp.isoformat() if step.timestamp else "",
                }
                for step in self.steps
            ],
        }
