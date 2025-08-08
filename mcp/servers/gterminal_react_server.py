#!/usr/bin/env python3
"""GTerminal ReAct Agent MCP Server.

Advanced MCP server providing ReAct (Reasoning + Acting) capabilities
with integrated authentication, monitoring, and caching.

CORE FEATURES:
- ReAct reasoning with thought chains
- Tool execution and coordination
- Session management and persistence
- Google Cloud authentication integration
- Comprehensive error handling and recovery

MCP TOOLS:
- react_reason: Execute ReAct reasoning cycle
- execute_tool: Execute tools within ReAct context
- manage_session: Session lifecycle management
- analyze_reasoning: Reasoning pattern analysis
"""

import asyncio
from datetime import UTC
from datetime import datetime
import json
import logging
import os
import sys
from typing import Any

# Set up Google Cloud environment
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", "gterminal-dev"))
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel
    from pydantic import Field
    import vertexai
    from vertexai.generative_models import GenerativeModel

    from ...auth import Permissions

    # Import our authentication and base server
    from ..base_server import BaseMCPServer
    from ..base_server import MCPServerConfig

except ImportError as e:
    logger.exception(f"Failed to import required packages: {e}")
    sys.exit(1)

# Initialize Vertex AI
try:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "gterminal-dev")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    vertexai.init(project=project_id, location=location)
    model = GenerativeModel("gemini-2.0-flash-exp")
    logger.info(f"✅ Vertex AI initialized successfully - Project: {project_id}")
except Exception as e:
    logger.exception(f"Failed to initialize Vertex AI: {e}")
    # Continue without Vertex AI for testing
    model = None


# Request/Response Models
class ReActRequest(BaseModel):
    """Request model for ReAct reasoning."""

    task: str = Field(..., description="Task to accomplish")
    context: str = Field(default="", description="Additional context")
    max_steps: int = Field(default=10, description="Maximum reasoning steps")
    session_id: str | None = Field(None, description="Session ID for persistence")
    tools_available: list[str] = Field(default_factory=list, description="Available tools")


class ToolExecutionRequest(BaseModel):
    """Request model for tool execution."""

    tool_name: str = Field(..., description="Name of tool to execute")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    reasoning_context: str = Field(default="", description="Reasoning context")
    session_id: str | None = Field(None, description="Session ID")


class SessionRequest(BaseModel):
    """Request model for session management."""

    action: str = Field(..., description="Action: create, get, update, delete, list")
    session_id: str | None = Field(None, description="Session ID")
    session_data: dict[str, Any] | None = Field(None, description="Session data")


class ReasoningAnalysisRequest(BaseModel):
    """Request model for reasoning analysis."""

    reasoning_steps: list[dict[str, Any]] = Field(..., description="Reasoning steps to analyze")
    analysis_type: str = Field(default="patterns", description="Type of analysis")


class GTerminalReActServer(BaseMCPServer):
    """GTerminal ReAct Agent MCP Server implementation."""

    def __init__(self):
        config = MCPServerConfig(
            name="gterminal-react",
            description="GTerminal ReAct Agent with reasoning and tool execution",
            version="1.0.0",
            require_auth=True,
            allowed_permissions={
                Permissions.REACT_AGENT_ACCESS,
                Permissions.TERMINAL_ACCESS,
                Permissions.WRITE_SESSIONS,
            },
            rate_limit_requests_per_minute=60,
            enable_caching=True,
            cache_ttl_seconds=300,
            tool_timeout_seconds=60,
        )

        super().__init__(config)

        # Session storage (in production, use database)
        self.sessions: dict[str, dict[str, Any]] = {}

        # Available tools registry
        self.available_tools = {
            "file_read": "Read file contents",
            "file_write": "Write to file",
            "terminal_execute": "Execute terminal commands",
            "web_search": "Search the web",
            "code_analyze": "Analyze code",
        }

    def _setup_custom_tools(self) -> None:
        """Setup ReAct-specific tools."""

        # Register ReAct reasoning tool
        self.register_tool(
            "react_reason",
            self._react_reason,
            required_permission=Permissions.REACT_AGENT_ACCESS,
            enable_caching=False,  # Reasoning should not be cached
            timeout_seconds=120,
        )

        # Register tool execution
        self.register_tool(
            "execute_tool",
            self._execute_tool,
            required_permission=Permissions.REACT_AGENT_ACCESS,
            enable_caching=True,
            timeout_seconds=60,
        )

        # Register session management
        self.register_tool(
            "manage_session",
            self._manage_session,
            required_permission=Permissions.WRITE_SESSIONS,
            enable_caching=False,
        )

        # Register reasoning analysis
        self.register_tool(
            "analyze_reasoning",
            self._analyze_reasoning,
            required_permission=Permissions.REACT_AGENT_ACCESS,
            enable_caching=True,
        )

    async def _react_reason(self, request: ReActRequest, **kwargs) -> dict[str, Any]:
        """Execute ReAct reasoning cycle."""
        if not model:
            return {"error": "Vertex AI model not available"}

        try:
            # Get or create session
            session_id = request.session_id or f"react_{int(asyncio.get_event_loop().time())}"
            session = self.sessions.get(
                session_id,
                {
                    "id": session_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "steps": [],
                    "context": {},
                },
            )

            # Prepare reasoning prompt
            prompt = f"""
            You are a ReAct (Reasoning + Acting) agent. For the given task, follow this pattern:

            Thought: [Your reasoning about what to do next]
            Action: [The action you want to take]
            Action Input: [The input to the action]
            Observation: [The result of the action]

            Continue this cycle until the task is complete, then provide:
            Final Answer: [Your final response]

            Task: {request.task}
            Context: {request.context}

            Available Tools: {", ".join(request.tools_available or self.available_tools.keys())}

            Previous Steps: {json.dumps(session.get("steps", [])[-3:] if session.get("steps") else [], indent=2)}

            Start reasoning:
            """

            # Generate reasoning
            response = await model.generate_content_async(prompt)
            reasoning_text = response.text

            # Parse reasoning steps
            steps = self._parse_reasoning_steps(reasoning_text)

            # Update session
            session["steps"].extend(steps)
            session["last_updated"] = datetime.now(UTC).isoformat()
            session["context"].update(
                {
                    "task": request.task,
                    "context": request.context,
                    "max_steps": request.max_steps,
                }
            )
            self.sessions[session_id] = session

            # Determine if reasoning is complete
            is_complete = any("final answer" in step.get("type", "").lower() for step in steps)

            result = {
                "session_id": session_id,
                "reasoning_steps": steps,
                "is_complete": is_complete,
                "next_action": (self._extract_next_action(steps) if not is_complete else None),
                "total_steps": len(session["steps"]),
                "raw_reasoning": reasoning_text,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            logger.info(f"ReAct reasoning completed: {len(steps)} steps generated")
            return result

        except Exception as e:
            logger.exception(f"ReAct reasoning failed: {e}")
            return {"error": str(e), "session_id": session_id}

    async def _execute_tool(self, request: ToolExecutionRequest, **kwargs) -> dict[str, Any]:
        """Execute a tool within ReAct context."""
        try:
            tool_name = request.tool_name
            tool_args = request.tool_args

            # Simulate tool execution (in production, integrate with actual tools)
            if tool_name == "file_read":
                result = await self._simulate_file_read(tool_args.get("path", ""))
            elif tool_name == "file_write":
                result = await self._simulate_file_write(
                    tool_args.get("path", ""), tool_args.get("content", "")
                )
            elif tool_name == "terminal_execute":
                result = await self._simulate_terminal_execute(tool_args.get("command", ""))
            elif tool_name == "web_search":
                result = await self._simulate_web_search(tool_args.get("query", ""))
            elif tool_name == "code_analyze":
                result = await self._simulate_code_analyze(tool_args.get("code", ""))
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            # Update session if provided
            if request.session_id and request.session_id in self.sessions:
                session = self.sessions[request.session_id]
                session.setdefault("tool_executions", []).append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "reasoning_context": request.reasoning_context,
                    }
                )
                session["last_updated"] = datetime.now(UTC).isoformat()

            return {
                "tool_name": tool_name,
                "result": result,
                "execution_time": "simulated",
                "session_id": request.session_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.exception(f"Tool execution failed: {e}")
            return {"error": str(e), "tool_name": request.tool_name}

    async def _manage_session(self, request: SessionRequest, **kwargs) -> dict[str, Any]:
        """Manage ReAct sessions."""
        try:
            action = request.action.lower()

            if action == "create":
                session_id = f"react_{int(asyncio.get_event_loop().time())}"
                session = {
                    "id": session_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "steps": [],
                    "context": request.session_data or {},
                    "tool_executions": [],
                }
                self.sessions[session_id] = session
                return {
                    "action": "create",
                    "session_id": session_id,
                    "session": session,
                }

            elif action == "get":
                if not request.session_id or request.session_id not in self.sessions:
                    return {
                        "error": "Session not found",
                        "session_id": request.session_id,
                    }
                return {"action": "get", "session": self.sessions[request.session_id]}

            elif action == "update":
                if not request.session_id or request.session_id not in self.sessions:
                    return {
                        "error": "Session not found",
                        "session_id": request.session_id,
                    }

                session = self.sessions[request.session_id]
                if request.session_data:
                    session["context"].update(request.session_data)
                    session["last_updated"] = datetime.now(UTC).isoformat()

                return {"action": "update", "session": session}

            elif action == "delete":
                if not request.session_id or request.session_id not in self.sessions:
                    return {
                        "error": "Session not found",
                        "session_id": request.session_id,
                    }

                del self.sessions[request.session_id]
                return {"action": "delete", "session_id": request.session_id}

            elif action == "list":
                sessions_info = [
                    {
                        "id": session["id"],
                        "created_at": session["created_at"],
                        "last_updated": session.get("last_updated"),
                        "steps_count": len(session.get("steps", [])),
                        "tools_used": len(session.get("tool_executions", [])),
                    }
                    for session in self.sessions.values()
                ]
                return {"action": "list", "sessions": sessions_info}

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception(f"Session management failed: {e}")
            return {"error": str(e), "action": request.action}

    async def _analyze_reasoning(
        self, request: ReasoningAnalysisRequest, **kwargs
    ) -> dict[str, Any]:
        """Analyze reasoning patterns and quality."""
        try:
            steps = request.reasoning_steps
            analysis_type = request.analysis_type

            if analysis_type == "patterns":
                analysis = self._analyze_reasoning_patterns(steps)
            elif analysis_type == "quality":
                analysis = self._analyze_reasoning_quality(steps)
            elif analysis_type == "efficiency":
                analysis = self._analyze_reasoning_efficiency(steps)
            else:
                analysis = {"error": f"Unknown analysis type: {analysis_type}"}

            return {
                "analysis_type": analysis_type,
                "steps_analyzed": len(steps),
                "analysis": analysis,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.exception(f"Reasoning analysis failed: {e}")
            return {"error": str(e)}

    # Helper methods
    def _parse_reasoning_steps(self, reasoning_text: str) -> list[dict[str, Any]]:
        """Parse reasoning text into structured steps."""
        steps = []
        lines = reasoning_text.split("\n")
        current_step = {}

        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                if current_step:
                    steps.append(current_step)
                current_step = {"type": "thought", "content": line[8:].strip()}
            elif line.startswith("Action:"):
                current_step["action"] = line[7:].strip()
            elif line.startswith("Action Input:"):
                current_step["action_input"] = line[13:].strip()
            elif line.startswith("Observation:"):
                current_step["observation"] = line[12:].strip()
            elif line.startswith("Final Answer:"):
                if current_step:
                    steps.append(current_step)
                steps.append({"type": "final_answer", "content": line[13:].strip()})
                current_step = {}

        if current_step:
            steps.append(current_step)

        return steps

    def _extract_next_action(self, steps: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Extract the next action from reasoning steps."""
        for step in reversed(steps):
            if step.get("action") and "action_input" in step:
                return {
                    "action": step["action"],
                    "input": step["action_input"],
                }
        return None

    def _analyze_reasoning_patterns(self, steps: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze reasoning patterns."""
        thought_count = sum(1 for step in steps if step.get("type") == "thought")
        action_count = sum(1 for step in steps if step.get("action"))
        final_answer_count = sum(1 for step in steps if step.get("type") == "final_answer")

        return {
            "total_steps": len(steps),
            "thought_steps": thought_count,
            "action_steps": action_count,
            "final_answers": final_answer_count,
            "avg_thought_length": sum(
                len(step.get("content", "")) for step in steps if step.get("type") == "thought"
            )
            / max(1, thought_count),
            "reasoning_complete": final_answer_count > 0,
        }

    def _analyze_reasoning_quality(self, steps: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze reasoning quality."""
        return {
            "coherence_score": 85,  # Placeholder
            "logical_flow": "good",
            "completeness": (
                "partial"
                if not any(step.get("type") == "final_answer" for step in steps)
                else "complete"
            ),
            "clarity": "high",
        }

    def _analyze_reasoning_efficiency(self, steps: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze reasoning efficiency."""
        return {
            "step_efficiency": min(
                100, max(0, 100 - len(steps) * 10)
            ),  # Fewer steps = more efficient
            "redundancy_score": 10,  # Placeholder
            "goal_directedness": "high",
        }

    # Simulated tool implementations (in production, these would be real tools)
    async def _simulate_file_read(self, path: str) -> dict[str, Any]:
        """Simulate file reading."""
        return {"content": f"[Simulated file content for {path}]", "size": 1024}

    async def _simulate_file_write(self, path: str, content: str) -> dict[str, Any]:
        """Simulate file writing."""
        return {"bytes_written": len(content), "path": path}

    async def _simulate_terminal_execute(self, command: str) -> dict[str, Any]:
        """Simulate terminal command execution."""
        return {
            "stdout": f"[Simulated output for: {command}]",
            "stderr": "",
            "exit_code": 0,
        }

    async def _simulate_web_search(self, query: str) -> dict[str, Any]:
        """Simulate web search."""
        return {"results": [f"[Simulated search result for: {query}]"], "count": 1}

    async def _simulate_code_analyze(self, code: str) -> dict[str, Any]:
        """Simulate code analysis."""
        return {"issues": [], "score": 95, "language": "python"}


def main():
    """Main entry point for the MCP server."""
    server = GTerminalReActServer()

    # Get configuration from environment
    host = os.getenv("MCP_HOST", "localhost")
    port = int(os.getenv("MCP_PORT", "3000"))

    logger.info("🧠 Starting GTerminal ReAct MCP Server")
    logger.info(f"🎯 Available tools: {list(server.tools.keys())}")
    logger.info(f"🔐 Authentication required: {server.config.require_auth}")

    try:
        server.run(host=host, port=port)
    except Exception as e:
        logger.exception(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
