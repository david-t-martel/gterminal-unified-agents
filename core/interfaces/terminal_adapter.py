#!/usr/bin/env python3
"""Terminal Interface Adapter for Core Agent Framework.

This adapter integrates the sophisticated terminal framework as the 4th interface type
alongside CLI, API, and MCP interfaces. It provides unified access to core agents
through a rich terminal UI with ReAct reasoning display.

Key Features:
- Cross-platform PTY support (Windows/WSL/Linux)
- Rich terminal UI with prompt-toolkit
- Real-time ReAct reasoning display
- WebSocket streaming integration
- Command completion and history
- Export/import session functionality
- Integration with all core agents

Usage:
    from gterminal.core.interfaces.terminal_adapter import TerminalAdapter

    adapter = TerminalAdapter()
    await adapter.start_terminal_session()
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
from typing import Any

from gterminal.core.agents.unified_code_generation_agent import UnifiedCodeGenerationAgent
from gterminal.core.agents.unified_code_review_agent import UnifiedCodeReviewAgent
from gterminal.core.agents.unified_documentation_generator import UnifiedDocumentationGenerator
from gterminal.core.agents.unified_workspace_analyzer import UnifiedWorkspaceAnalyzer
from gterminal.core.monitoring.integrated_monitoring import IntegratedMonitoring
from gterminal.core.performance.hybrid_gemini_client import HybridGeminiClient
from gterminal.core.security.integrated_security_middleware import IntegratedSecurityMiddleware

logger = logging.getLogger(__name__)


@dataclass
class TerminalSessionConfig:
    """Configuration for terminal interface sessions."""

    session_id: str
    agent_type: str = "unified_code_generation"
    enable_react_display: bool = True
    enable_websocket_streaming: bool = True
    export_sessions: bool = True
    history_file: str | None = None
    style_theme: str = "default"
    cross_platform_pty: bool = True


class TerminalAdapter:
    """Terminal Interface Adapter.

    Provides unified access to core agents through the sophisticated terminal framework.
    Integrates with all core performance, security, and monitoring components.
    """

    def __init__(self, config: TerminalSessionConfig | None = None) -> None:
        self.config = config or TerminalSessionConfig(session_id="terminal_default")
        self.terminal_interface = None
        self.active_sessions: dict[str, Any] = {}

        # Initialize core components
        self.hybrid_client = HybridGeminiClient()
        self.security_middleware = IntegratedSecurityMiddleware()
        self.monitoring = IntegratedMonitoring()

        # Initialize unified agents
        self.agents = {
            "code_generation": UnifiedCodeGenerationAgent(
                client=self.hybrid_client,
                security=self.security_middleware,
                monitoring=self.monitoring,
            ),
            "code_review": UnifiedCodeReviewAgent(
                client=self.hybrid_client,
                security=self.security_middleware,
                monitoring=self.monitoring,
            ),
            "documentation": UnifiedDocumentationGenerator(
                client=self.hybrid_client,
                security=self.security_middleware,
                monitoring=self.monitoring,
            ),
            "workspace_analysis": UnifiedWorkspaceAnalyzer(
                client=self.hybrid_client,
                security=self.security_middleware,
                monitoring=self.monitoring,
            ),
        }

        logger.info(f"TerminalAdapter initialized with session: {self.config.session_id}")

    async def start_terminal_session(self) -> None:
        """Start a terminal interface session with integrated agent access."""
        try:
            # Import and initialize terminal interface
            from gterminal.terminal.main import EnhancedTerminalUI as TerminalInterface

            # Create terminal interface (EnhancedTerminalUI takes no parameters)
            self.terminal_interface = TerminalInterface()

            # Configure session settings manually if needed
            self.terminal_interface.server_session_id = self.config.session_id

            # Start terminal session (method is called 'run', not 'run_terminal_session')
            await self.terminal_interface.run()

        except Exception as e:
            logger.exception(f"Failed to start terminal session: {e}")
            raise

    async def execute_agent_command(
        self,
        agent_type: str,
        command: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a command through a unified agent via terminal interface.

        Args:
            agent_type: Type of agent to use
            command: Command to execute
            parameters: Optional command parameters

        Returns:
            Agent execution result

        """
        if agent_type not in self.agents:
            msg = f"Unknown agent type: {agent_type}"
            raise ValueError(msg)

        agent = self.agents[agent_type]

        try:
            # Execute through the agent with monitoring
            result = await agent.execute_task(
                {
                    "command": command,
                    "parameters": parameters or {},
                    "session_id": self.config.session_id,
                    "interface": "terminal",
                },
            )

            # Log to monitoring
            await self.monitoring.log_agent_execution(
                agent_type=agent_type,
                command=command,
                result=result,
                interface="terminal",
            )

            return result

        except Exception as e:
            logger.exception(f"Agent command execution failed: {e}")
            await self.monitoring.log_error(
                error=str(e),
                context={"agent_type": agent_type, "command": command, "interface": "terminal"},
            )
            raise

    async def get_session_history(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Get terminal session history.

        Args:
            session_id: Optional specific session ID, defaults to current

        Returns:
            List of session history entries

        """
        target_session = session_id or self.config.session_id

        if target_session in self.active_sessions:
            return self.active_sessions[target_session].get("history", [])

        return []

    async def export_session(
        self, session_id: str | None = None, export_path: str | None = None
    ) -> str:
        """Export terminal session data.

        Args:
            session_id: Optional specific session ID, defaults to current
            export_path: Optional export file path

        Returns:
            Export file path

        """
        target_session = session_id or self.config.session_id

        if not export_path:
            export_path = (
                f"terminal_session_{target_session}_{int(asyncio.get_event_loop().time())}.json"
            )

        history = await self.get_session_history(target_session)

        export_data = {
            "session_id": target_session,
            "config": self.config.__dict__,
            "history": history,
            "agents_used": list(self.agents.keys()),
            "exported_at": asyncio.get_event_loop().time(),
        }

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Terminal session exported to: {export_path}")
        return export_path

    @asynccontextmanager
    async def terminal_context(self):
        """Async context manager for terminal sessions."""
        try:
            logger.info(f"Starting terminal context: {self.config.session_id}")
            yield self
        except Exception as e:
            logger.exception(f"Terminal context error: {e}")
            raise
        finally:
            await self.cleanup_session()

    async def cleanup_session(self) -> None:
        """Clean up terminal session resources."""
        try:
            if self.terminal_interface:
                await self.terminal_interface.cleanup()

            # Clean up agent resources
            for agent in self.agents.values():
                if hasattr(agent, "cleanup"):
                    await agent.cleanup()

            logger.info(f"Terminal session cleaned up: {self.config.session_id}")

        except Exception as e:
            logger.exception(f"Session cleanup error: {e}")

    def get_available_agents(self) -> dict[str, str]:
        """Get list of available agents for terminal interface.

        Returns:
            Dictionary mapping agent names to descriptions

        """
        return {
            "code_generation": "Generate and modify code with AI assistance",
            "code_review": "Review code for security, performance, and quality",
            "documentation": "Generate comprehensive documentation",
            "workspace_analysis": "Analyze project structure and dependencies",
        }

    def get_interface_info(self) -> dict[str, Any]:
        """Get terminal interface information.

        Returns:
            Interface capability information

        """
        return {
            "interface_type": "terminal",
            "features": [
                "Cross-platform PTY support",
                "Rich terminal UI with prompt-toolkit",
                "Real-time ReAct reasoning display",
                "WebSocket streaming integration",
                "Command completion and history",
                "Export/import session functionality",
            ],
            "supported_platforms": ["Windows", "WSL", "Linux", "macOS"],
            "agents": self.get_available_agents(),
            "session_config": self.config.__dict__,
        }


# Factory functions for easy access
async def create_terminal_session(
    session_id: str = "default",
    agent_type: str = "code_generation",
    **kwargs,
) -> TerminalAdapter:
    """Factory function to create and start a terminal session.

    Args:
        session_id: Unique session identifier
        agent_type: Default agent type to use
        **kwargs: Additional configuration options

    Returns:
        TerminalAdapter instance

    """
    config = TerminalSessionConfig(session_id=session_id, agent_type=agent_type, **kwargs)

    return TerminalAdapter(config)


async def terminal_interface_demo() -> None:
    """Demo function showing terminal interface capabilities."""
    async with terminal_context() as terminal:
        # Demo agent command
        await terminal.execute_agent_command(
            agent_type="code_generation",
            command="analyze_project",
            parameters={"project_path": "./", "depth": 2},
        )

        # Export session
        await terminal.export_session()


@asynccontextmanager
async def terminal_context(session_id: str = "demo", **kwargs):
    """Convenient context manager for terminal sessions."""
    adapter = await create_terminal_session(session_id=session_id, **kwargs)
    async with adapter.terminal_context():
        yield adapter


if __name__ == "__main__":
    # Run demo
    asyncio.run(terminal_interface_demo())
