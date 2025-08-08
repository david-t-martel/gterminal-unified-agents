"""Enhanced MCP Integration for Terminal Interface.

This module provides comprehensive MCP (Model Context Protocol) integration for the terminal,
including MCP tool discovery, execution, real-time progress tracking, and ReAct reasoning.

Features:
- Discover and execute MCP tools dynamically
- Real-time tool execution with progress tracking
- Integration with ReAct context for tool selection reasoning
- Error recovery and fallback mechanisms
- Tool result caching and optimization
- Security validation for tool execution
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
import json
import logging
import time
from typing import Any

from rich.console import Console

from gterminal.terminal.react_types import ReActContext
from gterminal.terminal.react_types import ReActStep
from gterminal.terminal.react_types import StepType
from gterminal.terminal.rust_terminal_ops import TerminalRustOps


class MCPToolDescriptor:
    """Descriptor for an MCP tool with metadata and validation."""

    def __init__(
        self, name: str, description: str, parameters: dict[str, Any], server: str
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.server = server
        self.usage_count = 0
        self.average_execution_time = 0.0
        self.success_rate = 1.0
        self.last_used = None

    def to_dict(self) -> dict[str, Any]:
        """Convert tool descriptor to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "server": self.server,
            "usage_count": self.usage_count,
            "average_execution_time": self.average_execution_time,
            "success_rate": self.success_rate,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


class MCPTerminalIntegration:
    """Enhanced MCP integration for terminal interface.

    This class provides:
    - Dynamic MCP tool discovery and management
    - Real-time tool execution with progress tracking
    - ReAct context integration for intelligent tool selection
    - Error recovery and fallback mechanisms
    - Tool result caching and performance optimization
    - Security validation for tool execution
    """

    def __init__(self, rust_ops: TerminalRustOps | None = None) -> None:
        """Initialize MCP terminal integration."""
        self.logger = logging.getLogger(__name__)
        self.console = Console()

        # Core components
        self.rust_ops = rust_ops or TerminalRustOps()

        # MCP tool management
        self.available_tools: dict[str, MCPToolDescriptor] = {}
        self.active_servers: dict[str, dict[str, Any]] = {}
        self.tool_categories: dict[str, list[str]] = {}

        # Execution tracking
        self.active_executions: dict[str, dict[str, Any]] = {}
        self.execution_history: list[dict[str, Any]] = []

        # Security and validation
        self.allowed_tool_patterns: list[str] = [
            "read_*",
            "list_*",
            "search_*",
            "analyze_*",
            "get_*",
            "find_*",
            "stat_*",
            "directory_tree",
            "validate_*",
        ]
        self.blocked_tool_patterns: list[str] = [
            "delete_*",
            "remove_*",
            "destroy_*",
            "format_*",
            "kill_*",
        ]

        # Performance metrics
        self.performance_metrics: dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "cache_hits": 0,
            "tool_popularity": {},
        }

        # Tool selection reasoning
        self.tool_selection_context: dict[str, Any] = {}

        self.logger.info("MCPTerminalIntegration initialized")

    async def initialize(self) -> bool:
        """Initialize MCP integration and discover available tools."""
        try:
            self.logger.info("🔍 Initializing MCP Terminal Integration...")

            # Discover available MCP servers and tools
            await self._discover_mcp_servers()
            await self._discover_mcp_tools()

            # Load cached tool metadata
            await self._load_cached_tool_metadata()

            # Initialize tool categories
            await self._categorize_tools()

            # Setup tool execution monitoring
            await self._setup_execution_monitoring()

            self.logger.info(
                f"✅ MCP Integration initialized with {len(self.available_tools)} tools from {len(self.active_servers)} servers",
            )
            return True

        except Exception as e:
            self.logger.exception(f"❌ MCP Integration initialization failed: {e}")
            return False

    async def _discover_mcp_servers(self) -> None:
        """Discover available MCP servers."""
        try:
            # This would typically use the MCP client to discover servers
            # For now, we'll use a simplified approach that checks common MCP servers

            potential_servers = [
                "rust-fs",
                "desktop-commander-wsl",
                "github-wsl",
                "sequential-thinking-wsl",
                "memory-wsl",
                "wsl-filesystem-wsl",
            ]

            for server_name in potential_servers:
                try:
                    # Simulate server discovery - in real implementation this would
                    # use MCP client to connect and validate servers
                    server_info = {
                        "name": server_name,
                        "status": "available",
                        "capabilities": ["tools", "resources"],
                        "discovered_at": datetime.now().isoformat(),
                    }

                    self.active_servers[server_name] = server_info
                    self.logger.info(f"✅ Discovered MCP server: {server_name}")

                except Exception as e:
                    self.logger.warning(f"⚠️ Could not connect to MCP server {server_name}: {e}")

        except Exception as e:
            self.logger.exception(f"MCP server discovery failed: {e}")

    async def _discover_mcp_tools(self) -> None:
        """Discover available tools from MCP servers."""
        try:
            # Tool definitions for known MCP servers
            # In a real implementation, this would query the servers directly

            tool_definitions = {
                "rust-fs": [
                    {
                        "name": "read_file",
                        "description": "Read file contents with high performance",
                        "parameters": {"path": {"type": "string", "required": True}},
                    },
                    {
                        "name": "write_file",
                        "description": "Write content to file with verification",
                        "parameters": {
                            "path": {"type": "string", "required": True},
                            "content": {"type": "string", "required": True},
                        },
                    },
                    {
                        "name": "list_files",
                        "description": "List files in directory with patterns",
                        "parameters": {
                            "path": {"type": "string", "required": True},
                            "pattern": {"type": "string", "required": False},
                        },
                    },
                    {
                        "name": "find_files",
                        "description": "Find files matching patterns",
                        "parameters": {
                            "path": {"type": "string", "required": True},
                            "glob": {"type": "string", "required": False},
                        },
                    },
                    {
                        "name": "search_content",
                        "description": "Search file contents using ripgrep",
                        "parameters": {
                            "path": {"type": "string", "required": True},
                            "pattern": {"type": "string", "required": True},
                        },
                    },
                ],
                "desktop-commander-wsl": [
                    {
                        "name": "execute_command",
                        "description": "Execute system commands securely",
                        "parameters": {
                            "command": {"type": "string", "required": True},
                            "timeout": {"type": "number", "required": False},
                        },
                    },
                    {
                        "name": "get_system_info",
                        "description": "Get system information and metrics",
                        "parameters": {},
                    },
                ],
                "wsl-filesystem-wsl": [
                    {
                        "name": "read_text_file",
                        "description": "Read text file with encoding detection",
                        "parameters": {
                            "path": {"type": "string", "required": True},
                            "head": {"type": "number", "required": False},
                            "tail": {"type": "number", "required": False},
                        },
                    },
                    {
                        "name": "list_directory",
                        "description": "List directory contents with details",
                        "parameters": {"path": {"type": "string", "required": True}},
                    },
                    {
                        "name": "search_files",
                        "description": "Search for files by name pattern",
                        "parameters": {
                            "path": {"type": "string", "required": True},
                            "pattern": {"type": "string", "required": True},
                        },
                    },
                ],
            }

            # Register tools from active servers
            for server_name in self.active_servers:
                if server_name in tool_definitions:
                    for tool_def in tool_definitions[server_name]:
                        tool_descriptor = MCPToolDescriptor(
                            name=tool_def["name"],
                            description=tool_def["description"],
                            parameters=tool_def["parameters"],
                            server=server_name,
                        )

                        tool_key = f"{server_name}::{tool_def['name']}"
                        self.available_tools[tool_key] = tool_descriptor

                        self.logger.info(f"✅ Registered tool: {tool_key}")

        except Exception as e:
            self.logger.exception(f"MCP tool discovery failed: {e}")

    async def _load_cached_tool_metadata(self) -> None:
        """Load cached tool usage statistics and metadata."""
        try:
            cached_metadata = await self.rust_ops.cache_get("mcp_tool_metadata")
            if cached_metadata:
                if isinstance(cached_metadata, str):
                    metadata = await self.rust_ops.parse_json(cached_metadata)
                else:
                    metadata = cached_metadata

                # Update tool descriptors with cached data
                for tool_key, tool_data in metadata.items():
                    if tool_key in self.available_tools:
                        tool = self.available_tools[tool_key]
                        tool.usage_count = tool_data.get("usage_count", 0)
                        tool.average_execution_time = tool_data.get("average_execution_time", 0.0)
                        tool.success_rate = tool_data.get("success_rate", 1.0)
                        if tool_data.get("last_used"):
                            tool.last_used = datetime.fromisoformat(tool_data["last_used"])

                self.logger.info("📊 Loaded cached tool metadata")

        except Exception as e:
            self.logger.warning(f"Could not load cached tool metadata: {e}")

    async def _categorize_tools(self) -> None:
        """Categorize tools by functionality for better organization."""
        self.tool_categories = {
            "file_operations": [],
            "search_and_discovery": [],
            "system_commands": [],
            "analysis": [],
            "utilities": [],
        }

        for tool_key, tool in self.available_tools.items():
            name = tool.name.lower()

            if any(pattern in name for pattern in ["read", "write", "create", "move", "copy"]):
                self.tool_categories["file_operations"].append(tool_key)
            elif any(pattern in name for pattern in ["search", "find", "list", "discover"]):
                self.tool_categories["search_and_discovery"].append(tool_key)
            elif any(pattern in name for pattern in ["execute", "command", "system", "process"]):
                self.tool_categories["system_commands"].append(tool_key)
            elif any(pattern in name for pattern in ["analyze", "validate", "check", "inspect"]):
                self.tool_categories["analysis"].append(tool_key)
            else:
                self.tool_categories["utilities"].append(tool_key)

    async def _setup_execution_monitoring(self) -> None:
        """Setup monitoring for tool execution performance."""

        async def cleanup_old_executions() -> None:
            while True:
                try:
                    await asyncio.sleep(300)  # Clean up every 5 minutes
                    cutoff = datetime.now().timestamp() - 3600  # 1 hour ago

                    # Clean up old execution records
                    self.execution_history = [
                        record
                        for record in self.execution_history
                        if datetime.fromisoformat(record["timestamp"]).timestamp() > cutoff
                    ]

                except Exception as e:
                    self.logger.exception(f"Execution cleanup error: {e}")

        asyncio.create_task(cleanup_old_executions())

    async def discover_tools(self, category: str | None = None) -> dict[str, list[dict[str, Any]]]:
        """Discover available MCP tools, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            Dictionary of tools organized by category

        """
        try:
            if category and category in self.tool_categories:
                # Return tools for specific category
                tools_in_category: dict[str, Any] = {}
                for tool_key in self.tool_categories[category]:
                    if tool_key in self.available_tools:
                        if category not in tools_in_category:
                            tools_in_category[category] = []
                        tools_in_category[category].append(self.available_tools[tool_key].to_dict())
                return tools_in_category

            # Return all tools organized by category
            organized_tools: dict[str, Any] = {}
            for cat, tool_keys in self.tool_categories.items():
                organized_tools[cat] = []
                for tool_key in tool_keys:
                    if tool_key in self.available_tools:
                        organized_tools[cat].append(self.available_tools[tool_key].to_dict())

            return organized_tools

        except Exception as e:
            self.logger.exception(f"Tool discovery failed: {e}")
            return {}

    async def execute_tool_with_reasoning(
        self,
        tool_selection_query: str,
        parameters: dict[str, Any],
        react_context: ReActContext | None = None,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Execute MCP tool with ReAct reasoning for tool selection.

        Args:
            tool_selection_query: Natural language query for tool selection
            parameters: Parameters for tool execution
            react_context: ReAct context for reasoning
            progress_callback: Optional progress callback

        Returns:
            Tool execution result with reasoning trace

        """
        execution_id = f"exec_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            # Initialize execution tracking
            self.active_executions[execution_id] = {
                "query": tool_selection_query,
                "parameters": parameters,
                "start_time": start_time,
                "status": "reasoning",
                "progress": 0,
            }

            # Step 1: Reasoning for tool selection
            if progress_callback:
                step = ReActStep(
                    step_type=StepType.THOUGHT,
                    content=f"I need to select the best MCP tool for: {tool_selection_query}. Let me analyze available tools and their capabilities.",
                )
                await progress_callback(step)

            # Select best tool using reasoning
            selected_tool, reasoning = await self._select_tool_with_reasoning(
                tool_selection_query, parameters
            )

            if not selected_tool:
                return {
                    "execution_id": execution_id,
                    "error": "No suitable tool found for the query",
                    "reasoning": reasoning,
                    "available_tools": len(self.available_tools),
                    "execution_time": time.time() - start_time,
                }

            # Step 2: Action - Execute selected tool
            if progress_callback:
                step = ReActStep(
                    step_type=StepType.ACTION,
                    content=f"Selected tool '{selected_tool}' based on reasoning. Executing with parameters: {parameters}",
                )
                await progress_callback(step)

            # Update execution status
            self.active_executions[execution_id].update(
                {
                    "status": "executing",
                    "progress": 50,
                    "selected_tool": selected_tool,
                    "reasoning": reasoning,
                },
            )

            # Execute the tool
            execution_result = await self._execute_mcp_tool(
                selected_tool, parameters, progress_callback
            )

            # Step 3: Observation - Analyze results
            if progress_callback:
                observation_content = (
                    f"Tool execution completed. Result: {str(execution_result)[:100]}..."
                )
                step = ReActStep(
                    step_type=StepType.OBSERVATION,
                    content=observation_content,
                    tool_result=execution_result,
                )
                await progress_callback(step)

            # Update performance metrics
            execution_time = time.time() - start_time
            await self._update_tool_performance_metrics(selected_tool, execution_time, True)

            # Final result
            enhanced_result = {
                "execution_id": execution_id,
                "selected_tool": selected_tool,
                "reasoning": reasoning,
                "parameters": parameters,
                "result": execution_result,
                "execution_time": execution_time,
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

            # Cache result
            await self._cache_execution_result(execution_id, enhanced_result)

            # Clean up
            del self.active_executions[execution_id]

            return enhanced_result

        except Exception as e:
            self.logger.exception(f"Tool execution with reasoning failed: {e}")

            # Update metrics for failure
            if "selected_tool" in self.active_executions.get(execution_id, {}):
                tool_name = self.active_executions[execution_id]["selected_tool"]
                await self._update_tool_performance_metrics(
                    tool_name, time.time() - start_time, False
                )

            error_result = {
                "execution_id": execution_id,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "success": False,
                "timestamp": datetime.now().isoformat(),
            }

            # Clean up
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]

            return error_result

    async def _select_tool_with_reasoning(
        self,
        query: str,
        parameters: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any]]:
        """Select the best tool for a query using reasoning.

        Args:
            query: Natural language query
            parameters: Available parameters

        Returns:
            Tuple of (selected_tool_key, reasoning_dict)

        """
        reasoning = {
            "query": query,
            "analysis": [],
            "candidates": [],
            "selection_criteria": [],
            "final_decision": None,
        }

        try:
            # Analyze query for keywords and intent
            query_lower = query.lower()
            intent_keywords = {
                "read": ["read", "view", "show", "display", "content"],
                "write": ["write", "save", "create", "store"],
                "search": ["search", "find", "locate", "discover"],
                "list": ["list", "show all", "enumerate", "directory"],
                "execute": ["run", "execute", "command", "process"],
                "analyze": ["analyze", "inspect", "check", "validate"],
            }

            # Determine primary intent
            detected_intents: list[Any] = []
            for intent, keywords in intent_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    detected_intents.append(intent)

            reasoning["analysis"].append(f"Detected intents: {detected_intents}")

            # Find candidate tools based on intent and parameters
            candidates: list[Any] = []
            for tool_key, tool in self.available_tools.items():
                tool_name = tool.name.lower()
                score = 0
                reasons: list[Any] = []

                # Score based on intent match
                for intent in detected_intents:
                    if intent in tool_name:
                        score += 10
                        reasons.append(f"Tool name matches intent '{intent}'")

                # Score based on parameter compatibility
                required_params = [
                    p for p, details in tool.parameters.items() if details.get("required", False)
                ]
                available_params = set(parameters.keys())

                if all(param in available_params for param in required_params):
                    score += 5
                    reasons.append("All required parameters available")
                elif any(param in available_params for param in required_params):
                    score += 2
                    reasons.append("Some required parameters available")

                # Score based on tool success rate and usage
                score += tool.success_rate * 3
                score += min(tool.usage_count / 10, 2)  # Cap at 2 points

                # Security check
                if self._is_tool_safe(tool_name):
                    score += 1
                    reasons.append("Tool passes security validation")
                else:
                    score -= 10
                    reasons.append("Tool blocked by security policy")

                if score > 0:
                    candidates.append(
                        {"tool_key": tool_key, "tool": tool, "score": score, "reasons": reasons}
                    )

            # Sort candidates by score
            candidates.sort(key=lambda x: x["score"], reverse=True)
            reasoning["candidates"] = [
                {"tool": c["tool_key"], "score": c["score"], "reasons": c["reasons"]}
                for c in candidates[:5]  # Top 5 candidates
            ]

            # Select best candidate
            if candidates:
                best_candidate = candidates[0]
                selected_tool = best_candidate["tool_key"]

                reasoning["selection_criteria"] = [
                    f"Highest score: {best_candidate['score']}",
                    f"Reasons: {', '.join(best_candidate['reasons'])}",
                ]
                reasoning["final_decision"] = selected_tool

                return selected_tool, reasoning
            reasoning["final_decision"] = "No suitable tool found"
            return None, reasoning

        except Exception as e:
            self.logger.exception(f"Tool selection reasoning failed: {e}")
            reasoning["error"] = str(e)
            return None, reasoning

    async def _execute_mcp_tool(
        self,
        tool_key: str,
        parameters: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> Any:
        """Execute an MCP tool with the given parameters.

        Args:
            tool_key: Tool identifier (server::tool_name)
            parameters: Tool parameters
            progress_callback: Optional progress callback

        Returns:
            Tool execution result

        """
        if tool_key not in self.available_tools:
            msg = f"Tool not found: {tool_key}"
            raise ValueError(msg)

        tool = self.available_tools[tool_key]

        try:
            # Update tool usage statistics
            tool.usage_count += 1
            tool.last_used = datetime.now()

            # Validate parameters
            validation_result = self._validate_tool_parameters(tool, parameters)
            if not validation_result["valid"]:
                msg = f"Parameter validation failed: {validation_result['errors']}"
                raise ValueError(msg)

            # Check cache for recent similar executions
            cache_key = f"mcp_tool:{tool_key}:{hash(json.dumps(parameters, sort_keys=True))}"
            cached_result = await self.rust_ops.cache_get(cache_key)

            if cached_result:
                self.performance_metrics["cache_hits"] += 1
                if isinstance(cached_result, str):
                    return await self.rust_ops.parse_json(cached_result)
                return cached_result

            # Execute tool based on server type
            result = await self._dispatch_tool_execution(tool, parameters)

            # Cache result
            await self.rust_ops.cache_set(cache_key, result, ttl=300)  # 5 minutes

            # Update performance metrics
            self.performance_metrics["total_executions"] += 1
            self.performance_metrics["successful_executions"] += 1

            return result

        except Exception as e:
            self.performance_metrics["failed_executions"] += 1
            self.logger.exception(f"Tool execution failed for {tool_key}: {e}")
            raise

    async def _dispatch_tool_execution(
        self, tool: MCPToolDescriptor, parameters: dict[str, Any]
    ) -> Any:
        """Dispatch tool execution to appropriate handler based on server."""
        server_name = tool.server
        tool_name = tool.name

        # This is a simplified implementation
        # In a real MCP integration, this would use the MCP client to call the actual servers

        if server_name == "rust-fs":
            return await self._execute_rust_fs_tool(tool_name, parameters)
        if server_name == "desktop-commander-wsl":
            return await self._execute_desktop_commander_tool(tool_name, parameters)
        if server_name == "wsl-filesystem-wsl":
            return await self._execute_wsl_filesystem_tool(tool_name, parameters)
        # Generic execution for unknown servers
        return await self._execute_generic_mcp_tool(tool, parameters)

    async def _execute_rust_fs_tool(self, tool_name: str, parameters: dict[str, Any]) -> Any:
        """Execute rust-fs MCP tool."""
        if tool_name == "read_file":
            return await self.rust_ops.read_file(parameters["path"])
        if tool_name == "write_file":
            return await self.rust_ops.write_file(parameters["path"], parameters["content"])
        if tool_name == "list_files":
            pattern = parameters.get("pattern", "*")
            return await self.rust_ops.list_files(parameters["path"], pattern)
        msg = f"Unknown rust-fs tool: {tool_name}"
        raise ValueError(msg)

    async def _execute_desktop_commander_tool(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> Any:
        """Execute desktop-commander-wsl MCP tool."""
        # This would integrate with the actual desktop-commander MCP server
        # For now, return a simulated result
        return {
            "tool": tool_name,
            "parameters": parameters,
            "result": "Simulated execution result",
            "timestamp": datetime.now().isoformat(),
        }

    async def _execute_wsl_filesystem_tool(self, tool_name: str, parameters: dict[str, Any]) -> Any:
        """Execute wsl-filesystem-wsl MCP tool."""
        # This would integrate with the actual wsl-filesystem MCP server
        # For now, return a simulated result
        return {
            "tool": tool_name,
            "parameters": parameters,
            "result": "Simulated WSL filesystem result",
            "timestamp": datetime.now().isoformat(),
        }

    async def _execute_generic_mcp_tool(
        self, tool: MCPToolDescriptor, parameters: dict[str, Any]
    ) -> Any:
        """Execute generic MCP tool via standard MCP protocol."""
        # This would use the MCP client to execute the tool
        # For now, return a generic result
        return {
            "server": tool.server,
            "tool": tool.name,
            "parameters": parameters,
            "result": "Generic MCP execution result",
            "timestamp": datetime.now().isoformat(),
        }

    def _validate_tool_parameters(
        self, tool: MCPToolDescriptor, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate tool parameters against schema."""
        validation_result = {"valid": True, "errors": []}

        try:
            # Check required parameters
            for param_name, param_info in tool.parameters.items():
                if param_info.get("required", False) and param_name not in parameters:
                    validation_result["errors"].append(f"Required parameter missing: {param_name}")
                    validation_result["valid"] = False

            # Check parameter types (basic validation)
            for param_name, param_value in parameters.items():
                if param_name in tool.parameters:
                    expected_type = tool.parameters[param_name].get("type", "string")
                    if expected_type == "string" and not isinstance(param_value, str):
                        validation_result["errors"].append(
                            f"Parameter {param_name} should be string"
                        )
                        validation_result["valid"] = False
                    elif expected_type == "number" and not isinstance(param_value, int | float):
                        validation_result["errors"].append(
                            f"Parameter {param_name} should be number"
                        )
                        validation_result["valid"] = False

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {e}")

        return validation_result

    def _is_tool_safe(self, tool_name: str) -> bool:
        """Check if tool is safe to execute based on security policies."""
        tool_lower = tool_name.lower()

        # Check blocked patterns
        for pattern in self.blocked_tool_patterns:
            if pattern.replace("*", "") in tool_lower:
                return False

        # Check allowed patterns
        for pattern in self.allowed_tool_patterns:
            if pattern.replace("*", "") in tool_lower:
                return True

        # Default to safe for unknown patterns
        return True

    async def _update_tool_performance_metrics(
        self, tool_key: str, execution_time: float, success: bool
    ) -> None:
        """Update performance metrics for a tool."""
        if tool_key in self.available_tools:
            tool = self.available_tools[tool_key]

            # Update average execution time
            total_time = tool.average_execution_time * (tool.usage_count - 1)
            tool.average_execution_time = (total_time + execution_time) / tool.usage_count

            # Update success rate
            if success:
                tool.success_rate = (
                    (tool.success_rate * (tool.usage_count - 1)) + 1.0
                ) / tool.usage_count
            else:
                tool.success_rate = (tool.success_rate * (tool.usage_count - 1)) / tool.usage_count

            # Update global metrics
            self.performance_metrics["tool_popularity"][tool_key] = tool.usage_count

    async def _cache_execution_result(self, execution_id: str, result: dict[str, Any]) -> None:
        """Cache execution result for analysis and debugging."""
        try:
            cache_key = f"mcp_execution:{execution_id}"
            await self.rust_ops.cache_set(cache_key, result, ttl=3600)

            # Add to execution history
            self.execution_history.append(
                {
                    "execution_id": execution_id,
                    "timestamp": result["timestamp"],
                    "tool": result.get("selected_tool"),
                    "success": result.get("success", False),
                    "execution_time": result.get("execution_time", 0),
                },
            )

        except Exception as e:
            self.logger.warning(f"Could not cache execution result: {e}")

    async def get_available_tools(self, category: str | None = None) -> dict[str, Any]:
        """Get available tools with optional category filter."""
        return {
            "total_tools": len(self.available_tools),
            "active_servers": len(self.active_servers),
            "categories": list(self.tool_categories.keys()),
            "tools": await self.discover_tools(category),
        }

    async def get_tool_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive tool performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "overview": {
                "total_tools": len(self.available_tools),
                "total_executions": self.performance_metrics["total_executions"],
                "success_rate": (
                    self.performance_metrics["successful_executions"]
                    / max(self.performance_metrics["total_executions"], 1)
                ),
                "cache_hit_rate": self.performance_metrics["cache_hits"]
                / max(self.performance_metrics["total_executions"], 1),
                "active_executions": len(self.active_executions),
            },
            "tool_performance": {},
            "popular_tools": [],
            "recommendations": [],
        }

        # Add per-tool performance data
        for tool_key, tool in self.available_tools.items():
            if tool.usage_count > 0:
                report["tool_performance"][tool_key] = {
                    "usage_count": tool.usage_count,
                    "success_rate": tool.success_rate,
                    "average_execution_time": tool.average_execution_time,
                    "last_used": tool.last_used.isoformat() if tool.last_used else None,
                }

        # Popular tools
        popular_tools = sorted(
            [(k, v.usage_count) for k, v in self.available_tools.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        report["popular_tools"] = [{"tool": tool, "usage": count} for tool, count in popular_tools]

        # Generate recommendations
        if report["overview"]["success_rate"] < 0.8:
            report["recommendations"].append("Low success rate - review tool selection logic")

        if report["overview"]["cache_hit_rate"] < 0.3:
            report["recommendations"].append("Low cache hit rate - consider longer cache TTL")

        return report

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active tool execution."""
        if execution_id in self.active_executions:
            try:
                self.active_executions[execution_id]["status"] = "cancelled"
                self.active_executions[execution_id]["cancelled_at"] = datetime.now().isoformat()
                return True
            except Exception as e:
                self.logger.exception(f"Failed to cancel execution {execution_id}: {e}")
                return False
        return False

    async def cleanup(self) -> None:
        """Clean up resources and save tool metadata."""
        try:
            # Save tool metadata to cache
            tool_metadata: dict[str, Any] = {}
            for tool_key, tool in self.available_tools.items():
                tool_metadata[tool_key] = {
                    "usage_count": tool.usage_count,
                    "average_execution_time": tool.average_execution_time,
                    "success_rate": tool.success_rate,
                    "last_used": tool.last_used.isoformat() if tool.last_used else None,
                }

            serialized_metadata = await self.rust_ops.serialize_json(tool_metadata)
            await self.rust_ops.cache_set("mcp_tool_metadata", serialized_metadata, ttl=86400)

            self.logger.info("🧹 MCP Terminal Integration cleanup completed")

        except Exception as e:
            self.logger.exception(f"MCP cleanup failed: {e}")
