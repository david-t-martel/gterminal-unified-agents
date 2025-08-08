#!/usr/bin/env python3
"""Tool Registry System - Manages all available tools for the ReAct engine."""

from abc import ABC
from abc import abstractmethod
import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel

# Tools will be imported lazily to avoid circular imports

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool
    data: Any
    error: str | None = None
    execution_time: float = 0.0


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class ToolDescription(BaseModel):
    """Description of a tool's capabilities."""

    name: str
    description: str
    parameters: list[ToolParameter]
    category: str


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, name: str, description: str, category: str = "general") -> None:
        self.name = name
        self.description = description
        self.category = category

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            params: Parameters for tool execution

        Returns:
            ToolResult with execution outcome

        """

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """Get parameter definitions for this tool.

        Returns:
            List of parameter definitions

        """

    def get_description(self) -> ToolDescription:
        """Get complete tool description.

        Returns:
            ToolDescription with all metadata

        """
        return ToolDescription(
            name=self.name,
            description=self.description,
            parameters=self.get_parameters(),
            category=self.category,
        )

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validate parameters before execution.

        Args:
            params: Parameters to validate

        Returns:
            True if valid, raises exception otherwise

        """
        required_params = [p for p in self.get_parameters() if p.required]

        for param in required_params:
            if param.name not in params:
                msg = f"Required parameter '{param.name}' not provided"
                raise ValueError(msg)

        return True


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self) -> None:
        self.tools: dict[str, BaseTool] = {}
        self._register_core_tools()
        logger.info("Tool Registry initialized")

    def _register_core_tools(self) -> None:
        """Register core development tools using lazy imports."""
        # Lazy imports to avoid circular dependencies
        from gterminal.core.tools.analysis import AnalyzeCodeTool
        from gterminal.core.tools.analysis import FindDependenciesTool
        from gterminal.core.tools.analysis import LintCodeTool
        from gterminal.core.tools.analysis import ProfilePerformanceTool
        from gterminal.core.tools.filesystem import DeleteFileTool
        from gterminal.core.tools.filesystem import ListDirectoryTool
        from gterminal.core.tools.filesystem import MoveFileTool
        from gterminal.core.tools.filesystem import ReadFileTool
        from gterminal.core.tools.filesystem import SearchFilesTool
        from gterminal.core.tools.filesystem import WriteFileTool
        from gterminal.core.tools.generation import GenerateBoilerplateTool
        from gterminal.core.tools.generation import GenerateCodeTool
        from gterminal.core.tools.generation import GenerateTestsTool
        from gterminal.core.tools.generation import RefactorCodeTool
        from gterminal.core.tools.shell import BuildProjectTool
        from gterminal.core.tools.shell import ExecuteCommandTool
        from gterminal.core.tools.shell import InstallDependenciesTool
        from gterminal.core.tools.shell import RunTestsTool
        from gterminal.core.tools.shell import SecureCodeAnalysisTool
        from gterminal.core.tools.shell import SecureSystemInfoTool

        # File system tools
        self.register(ReadFileTool())
        self.register(WriteFileTool())
        self.register(ListDirectoryTool())
        self.register(SearchFilesTool())
        self.register(DeleteFileTool())
        self.register(MoveFileTool())

        # Analysis tools
        self.register(AnalyzeCodeTool())
        self.register(FindDependenciesTool())
        self.register(ProfilePerformanceTool())
        self.register(LintCodeTool())

        # Generation tools
        self.register(GenerateCodeTool())
        self.register(RefactorCodeTool())
        self.register(GenerateTestsTool())
        self.register(GenerateBoilerplateTool())

        # Shell execution tools
        self.register(ExecuteCommandTool())
        self.register(InstallDependenciesTool())
        self.register(RunTestsTool())
        self.register(BuildProjectTool())
        self.register(SecureSystemInfoTool())
        self.register(SecureCodeAnalysisTool())

        logger.info(f"Registered {len(self.tools)} core tools")

    def register(self, tool: BaseTool) -> None:
        """Register a new tool.

        Args:
            tool: Tool instance to register

        """
        if tool.name in self.tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")

        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, tool_name: str) -> None:
        """Remove a tool from the registry.

        Args:
            tool_name: Name of tool to remove

        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.debug(f"Unregistered tool: {tool_name}")

    async def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            params: Parameters for tool execution

        Returns:
            ToolResult from execution

        """
        if tool_name not in self.tools:
            return ToolResult(success=False, data=None, error=f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]

        try:
            # Validate parameters
            tool.validate_params(params)

            # Execute tool
            start_time = time.time()
            result = await tool.execute(params)
            result.execution_time = time.time() - start_time

            logger.info(f"Tool '{tool_name}' executed successfully in {result.execution_time:.2f}s")
            return result

        except Exception as e:
            logger.exception(f"Tool '{tool_name}' execution failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))

    def get_tool(self, tool_name: str) -> BaseTool | None:
        """Get a tool instance by name.

        Args:
            tool_name: Name of tool to retrieve

        Returns:
            Tool instance or None if not found

        """
        return self.tools.get(tool_name)

    def get_tool_descriptions(self) -> dict[str, dict[str, Any]]:
        """Get descriptions of all available tools.

        Returns:
            Dictionary of tool descriptions

        """
        descriptions: dict[str, Any] = {}
        for name, tool in self.tools.items():
            desc = tool.get_description()
            descriptions[name] = {
                "description": desc.description,
                "category": desc.category,
                "parameters": [p.model_dump() for p in desc.parameters],
            }
        return descriptions

    def get_tools_by_category(self, category: str) -> list[BaseTool]:
        """Get all tools in a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of tools in the category

        """
        return [tool for tool in self.tools.values() if tool.category == category]

    def list_categories(self) -> list[str]:
        """Get all available tool categories.

        Returns:
            List of unique categories

        """
        categories = {tool.category for tool in self.tools.values()}
        return sorted(categories)

    def list_tools(self) -> list[str]:
        """Get names of all registered tools.

        Returns:
            List of tool names

        """
        return sorted(self.tools.keys())

    async def execute_parallel(
        self, executions: list[tuple[str, dict[str, Any]]]
    ) -> list[ToolResult]:
        """Execute multiple tools in parallel.

        Args:
            executions: List of (tool_name, params) tuples

        Returns:
            List of ToolResults in the same order as input

        """
        tasks: list[Any] = []
        for tool_name, params in executions:
            tasks.append(self.execute(tool_name, params))

        return await asyncio.gather(*tasks)
