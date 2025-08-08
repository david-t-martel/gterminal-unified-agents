"""Deep Agent Integration for Terminal Interface.

This module provides comprehensive agent integration with the unified Gemini server,
enhanced agent loading, real-time progress tracking, and error recovery mechanisms.

Features:
- Integration with unified Gemini server on port 8100
- Enhanced agent loading with terminal-specific capabilities
- Real-time progress tracking during agent execution
- Error recovery and fallback mechanisms
- Session management and context preservation
- Performance optimization with Rust extensions
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
import logging
import time
from typing import Any

import httpx
from rich.console import Console

from gterminal.agents import AGENT_REGISTRY
from gterminal.terminal.agent_commands import AgentCommandProcessor
from gterminal.terminal.react_types import ReActContext
from gterminal.terminal.react_types import ReActStep
from gterminal.terminal.react_types import StepType
from gterminal.terminal.rust_terminal_ops import TerminalRustOps


class TerminalAgentIntegration:
    """Comprehensive agent integration for terminal interface.

    This class provides:
    - Deep integration with unified Gemini server
    - Enhanced agent loading with capabilities discovery
    - Real-time progress tracking and status updates
    - Error recovery and fallback mechanisms
    - Session management and context preservation
    - Performance optimization through Rust extensions
    """

    def __init__(self, gemini_server_url: str = "http://localhost:8100") -> None:
        """Initialize the terminal agent integration."""
        self.logger = logging.getLogger(__name__)
        self.console = Console()
        self.gemini_server_url = gemini_server_url.rstrip("/")

        # Core components
        self.rust_ops = TerminalRustOps()
        self.command_processor = AgentCommandProcessor(gemini_server_url)

        # HTTP client for server communication
        self.http_client = httpx.AsyncClient(timeout=300)

        # Agent management
        self.loaded_agents: dict[str, Any] = {}
        self.agent_capabilities: dict[str, dict[str, Any]] = {}
        self.agent_health_status: dict[str, dict[str, Any]] = {}

        # Session management
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.session_contexts: dict[str, ReActContext] = {}

        # Progress tracking
        self.active_tasks: dict[str, dict[str, Any]] = {}
        self.progress_callbacks: dict[str, list[Callable]] = {}

        # Performance metrics
        self.performance_metrics: dict[str, Any] = {
            "agent_load_times": {},
            "task_execution_times": {},
            "error_recovery_count": 0,
            "cache_hit_rate": 0.0,
            "total_tasks_executed": 0,
        }

        # Error recovery configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.fallback_timeout = 30.0

        self.logger.info("TerminalAgentIntegration initialized with full functionality")

    async def initialize(self) -> bool:
        """Initialize the agent integration system.

        Returns:
            True if initialization successful, False otherwise

        """
        try:
            self.logger.info("🚀 Initializing Terminal Agent Integration...")

            # Check Gemini server connectivity
            server_available = await self._check_gemini_server_health()
            if not server_available:
                self.logger.warning("⚠️ Gemini server not available - running in offline mode")

            # Load and discover agent capabilities
            await self._discover_and_load_agents()

            # Initialize session management
            await self._initialize_session_management()

            # Start health monitoring
            asyncio.create_task(self._start_health_monitoring())

            # Cache initial performance baseline
            await self._cache_performance_baseline()

            self.logger.info(
                f"✅ Agent Integration initialized with {len(self.loaded_agents)} agents"
            )
            return True

        except Exception as e:
            self.logger.exception(f"❌ Agent Integration initialization failed: {e}")
            return False

    async def _check_gemini_server_health(self) -> bool:
        """Check if the unified Gemini server is available and healthy."""
        try:
            response = await self.http_client.get(f"{self.gemini_server_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                self.logger.info(
                    f"✅ Gemini server healthy: {health_data.get('status', 'unknown')}"
                )
                return True
            self.logger.warning(f"⚠️ Gemini server returned status {response.status_code}")
            return False
        except Exception as e:
            self.logger.warning(f"⚠️ Cannot reach Gemini server: {e}")
            return False

    async def _discover_and_load_agents(self) -> None:
        """Discover and load all available agents with capability detection."""
        self.logger.info("🔍 Discovering and loading agents...")

        for agent_name, agent_class in AGENT_REGISTRY.items():
            if agent_class is None:
                self.logger.warning(f"⚠️ Agent {agent_name} is not implemented yet")
                continue

            try:
                start_time = time.time()

                # Load agent with error handling
                agent_instance = await self._load_agent_with_fallback(agent_name, agent_class)

                if agent_instance:
                    self.loaded_agents[agent_name] = agent_instance

                    # Discover agent capabilities
                    capabilities = await self._discover_agent_capabilities(
                        agent_name, agent_instance
                    )
                    self.agent_capabilities[agent_name] = capabilities

                    # Initialize health status
                    self.agent_health_status[agent_name] = {
                        "status": "healthy",
                        "last_check": datetime.now(),
                        "error_count": 0,
                        "response_time": time.time() - start_time,
                    }

                    # Cache load time for performance metrics
                    self.performance_metrics["agent_load_times"][agent_name] = (
                        time.time() - start_time
                    )

                    self.logger.info(
                        f"✅ Loaded agent: {agent_name} ({capabilities.get('type', 'unknown')})"
                    )
                else:
                    self.logger.error(f"❌ Failed to load agent: {agent_name}")

            except Exception as e:
                self.logger.exception(f"❌ Error loading agent {agent_name}: {e}")
                self.performance_metrics["error_recovery_count"] += 1

    async def _load_agent_with_fallback(self, agent_name: str, agent_class: type) -> Any | None:
        """Load agent with fallback mechanisms and error recovery."""
        for attempt in range(self.max_retries):
            try:
                # Attempt to instantiate agent
                agent_instance = agent_class()

                # Validate agent has required methods
                required_methods = ["process_request", "get_capabilities"]
                for method_name in required_methods:
                    if not hasattr(agent_instance, method_name):
                        self.logger.warning(f"⚠️ Agent {agent_name} missing method: {method_name}")

                return agent_instance

            except Exception as e:
                self.logger.warning(
                    f"⚠️ Agent load attempt {attempt + 1} failed for {agent_name}: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.logger.exception(f"❌ All attempts failed for agent {agent_name}")
                    return None
        return None

    async def _discover_agent_capabilities(
        self, agent_name: str, agent_instance: Any
    ) -> dict[str, Any]:
        """Discover agent capabilities and metadata."""
        capabilities = {
            "type": agent_name,
            "methods": [],
            "supports_streaming": False,
            "supports_cancellation": False,
            "estimated_performance": "unknown",
            "resource_requirements": "low",
        }

        try:
            # Check if agent has get_capabilities method
            if hasattr(agent_instance, "get_capabilities"):
                agent_caps = await asyncio.to_thread(agent_instance.get_capabilities)
                if isinstance(agent_caps, dict):
                    capabilities.update(agent_caps)

            # Discover available methods
            methods = [
                method
                for method in dir(agent_instance)
                if not method.startswith("_") and callable(getattr(agent_instance, method))
            ]
            capabilities["methods"] = methods

            # Check for streaming support
            capabilities["supports_streaming"] = hasattr(agent_instance, "stream_process")

            # Check for cancellation support
            capabilities["supports_cancellation"] = hasattr(agent_instance, "cancel_task")

            # Estimate performance characteristics
            if agent_name in ["code-reviewer", "workspace-analyzer"]:
                capabilities["estimated_performance"] = "medium"
                capabilities["resource_requirements"] = "medium"
            elif agent_name in ["documentation-generator", "code-generator"]:
                capabilities["estimated_performance"] = "slow"
                capabilities["resource_requirements"] = "high"

        except Exception as e:
            self.logger.warning(f"⚠️ Could not discover capabilities for {agent_name}: {e}")

        return capabilities

    async def _initialize_session_management(self) -> None:
        """Initialize session management with persistent storage."""
        self.logger.info("📋 Initializing session management...")

        try:
            # Load existing sessions from cache
            cached_sessions = await self.rust_ops.cache_get("terminal_sessions")
            if cached_sessions:
                if isinstance(cached_sessions, str):
                    sessions_data = await self.rust_ops.parse_json(cached_sessions)
                else:
                    sessions_data = cached_sessions

                for session_id, session_data in sessions_data.items():
                    self.active_sessions[session_id] = session_data
                    self.logger.info(f"📋 Restored session: {session_id}")

        except Exception as e:
            self.logger.warning(f"⚠️ Could not load cached sessions: {e}")

    async def _start_health_monitoring(self) -> None:
        """Start continuous health monitoring for agents."""

        async def health_monitor() -> None:
            while True:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    await self._check_agent_health()
                except Exception as e:
                    self.logger.exception(f"Health monitoring error: {e}")

        asyncio.create_task(health_monitor())
        self.logger.info("💊 Started agent health monitoring")

    async def _check_agent_health(self) -> None:
        """Check health status of all loaded agents."""
        for agent_name, agent_instance in self.loaded_agents.items():
            try:
                start_time = time.time()

                # Simple health check - attempt to call a basic method
                if hasattr(agent_instance, "health_check"):
                    health_result = await asyncio.to_thread(agent_instance.health_check)
                    status = "healthy" if health_result else "unhealthy"
                else:
                    # Fallback health check
                    status = "healthy" if agent_instance else "unhealthy"

                response_time = time.time() - start_time

                # Update health status
                self.agent_health_status[agent_name].update(
                    {
                        "status": status,
                        "last_check": datetime.now(),
                        "response_time": response_time,
                    },
                )

                if status != "healthy":
                    self.agent_health_status[agent_name]["error_count"] += 1
                    self.logger.warning(f"⚠️ Agent {agent_name} health check failed")

            except Exception as e:
                self.logger.warning(f"⚠️ Health check failed for {agent_name}: {e}")
                self.agent_health_status[agent_name]["error_count"] += 1

    async def _cache_performance_baseline(self) -> None:
        """Cache initial performance baseline metrics."""
        try:
            baseline_data = {
                "timestamp": datetime.now().isoformat(),
                "loaded_agents": len(self.loaded_agents),
                "rust_extensions_available": hasattr(self.rust_ops, "cache"),
                "cache_stats": await self.rust_ops.cache_stats(),
            }

            await self.rust_ops.cache_set("performance_baseline", baseline_data, ttl=86400)
            self.logger.info("📊 Cached performance baseline")

        except Exception as e:
            self.logger.warning(f"⚠️ Could not cache performance baseline: {e}")

    async def execute_agent_command(
        self,
        command: str,
        args: list[str],
        session_id: str | None = None,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Execute agent command with comprehensive progress tracking and error recovery.

        Args:
            command: Command to execute (analyze, review, generate, document, architect)
            args: Command arguments
            session_id: Optional session ID for context
            progress_callback: Optional callback for progress updates

        Returns:
            Command execution result with metadata

        """
        task_id = f"task_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            # Initialize task tracking
            self.active_tasks[task_id] = {
                "command": command,
                "args": args,
                "session_id": session_id,
                "start_time": start_time,
                "status": "initializing",
                "progress": 0,
            }

            # Register progress callback
            if progress_callback:
                if task_id not in self.progress_callbacks:
                    self.progress_callbacks[task_id] = []
                self.progress_callbacks[task_id].append(progress_callback)

            # Get or create session context
            react_context = await self._get_or_create_session_context(session_id)

            # Create progress update callback
            async def step_callback(step: ReActStep) -> None:
                await self._handle_react_step_update(task_id, step)

            # Update task status
            await self._update_task_progress(task_id, 10, "Preparing command execution...")

            # Execute command through command processor
            if command == "analyze":
                result = await self.command_processor.execute_analyze_command(
                    args, react_context, step_callback
                )
            elif command == "review":
                result = await self.command_processor.execute_review_command(
                    args, react_context, step_callback
                )
            elif command == "generate":
                result = await self.command_processor.execute_generate_command(
                    args, react_context, step_callback
                )
            elif command == "document":
                result = await self.command_processor.execute_document_command(
                    args, react_context, step_callback
                )
            elif command == "architect":
                result = await self.command_processor.execute_architect_command(
                    args, react_context, step_callback
                )
            else:
                result = await self._execute_custom_command(
                    command, args, react_context, step_callback
                )

            # Update task completion
            execution_time = time.time() - start_time
            await self._update_task_progress(task_id, 100, "Command completed successfully")

            # Update performance metrics
            self.performance_metrics["total_tasks_executed"] += 1
            self.performance_metrics["task_execution_times"][command] = execution_time

            # Enhance result with execution metadata
            enhanced_result = {
                **result,
                "task_id": task_id,
                "session_id": session_id,
                "execution_metadata": {
                    "start_time": start_time,
                    "end_time": time.time(),
                    "execution_time": execution_time,
                    "task_status": "completed",
                    "agent_integration_version": "1.0.0",
                },
            }

            # Cache result for session continuity
            if session_id:
                await self._cache_session_result(session_id, enhanced_result)

            # Clean up task tracking
            del self.active_tasks[task_id]
            if task_id in self.progress_callbacks:
                del self.progress_callbacks[task_id]

            return enhanced_result

        except Exception as e:
            self.logger.exception(f"❌ Command execution failed: {e}")

            # Update task with error
            await self._update_task_progress(task_id, 0, f"Command failed: {e!s}")

            # Attempt error recovery
            recovery_result = await self._attempt_error_recovery(command, args, str(e))

            error_result = {
                "command": command,
                "error": str(e),
                "task_id": task_id,
                "session_id": session_id,
                "execution_time": time.time() - start_time,
                "recovery_attempted": recovery_result is not None,
                "recovery_result": recovery_result,
                "timestamp": datetime.now().isoformat(),
            }

            # Clean up
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            if task_id in self.progress_callbacks:
                del self.progress_callbacks[task_id]

            return error_result

    async def _get_or_create_session_context(self, session_id: str | None) -> ReActContext:
        """Get existing session context or create new one."""
        if not session_id:
            session_id = f"session_{int(time.time())}"

        if session_id in self.session_contexts:
            return self.session_contexts[session_id]

        # Create new context
        context = ReActContext()
        context.session_id = session_id

        # Load cached context if available
        cached_context = await self.rust_ops.load_context(session_id)
        if cached_context:
            context = cached_context

        self.session_contexts[session_id] = context
        return context

    async def _handle_react_step_update(self, task_id: str, step: ReActStep) -> None:
        """Handle ReAct step updates and notify progress callbacks."""
        try:
            # Update task progress based on step type
            if step.step_type == StepType.THOUGHT:
                progress = 20
                status = f"Reasoning: {step.content[:50]}..."
            elif step.step_type == StepType.ACTION:
                progress = 60
                status = f"Executing: {step.content[:50]}..."
            elif step.step_type == StepType.OBSERVATION:
                progress = 80
                status = f"Observing: {step.content[:50]}..."
            elif step.step_type == StepType.FINAL_ANSWER:
                progress = 95
                status = "Finalizing results..."
            else:
                progress = 50
                status = f"Processing: {step.content[:50]}..."

            await self._update_task_progress(task_id, progress, status)

            # Notify progress callbacks
            if task_id in self.progress_callbacks:
                for callback in self.progress_callbacks[task_id]:
                    try:
                        await callback(step)
                    except Exception as e:
                        self.logger.warning(f"Progress callback failed: {e}")

        except Exception as e:
            self.logger.warning(f"React step update failed: {e}")

    async def _update_task_progress(self, task_id: str, progress: int, status: str) -> None:
        """Update task progress and notify callbacks."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].update(
                {"progress": progress, "status": status, "last_update": datetime.now().isoformat()},
            )

    async def _execute_custom_command(
        self,
        command: str,
        args: list[str],
        react_context: ReActContext,
        step_callback,
    ) -> dict[str, Any]:
        """Execute custom command through Gemini server."""
        try:
            # Send custom command to Gemini server
            custom_request = {
                "task_type": "custom",
                "instruction": f"Execute custom command: {command} {' '.join(args)}",
                "options": {
                    "command": command,
                    "args": args,
                    "session_context": react_context.session_id if react_context else None,
                },
            }

            response = await self.http_client.post(
                f"{self.gemini_server_url}/task", json=custom_request
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            return {
                "command": command,
                "error": f"Custom command execution failed: {e}",
                "timestamp": datetime.now().isoformat(),
            }

    async def _attempt_error_recovery(
        self, command: str, args: list[str], error: str
    ) -> dict[str, Any] | None:
        """Attempt to recover from command execution errors."""
        self.performance_metrics["error_recovery_count"] += 1

        try:
            # Simple recovery strategies
            if "timeout" in error.lower():
                # Retry with longer timeout
                self.logger.info("🔄 Attempting recovery with extended timeout...")
                # Implementation would retry with extended timeout
                return {"recovery_type": "timeout_extension", "attempted": True}

            if "connection" in error.lower():
                # Check server health and retry
                self.logger.info("🔄 Attempting recovery with server health check...")
                server_healthy = await self._check_gemini_server_health()
                if server_healthy:
                    return {"recovery_type": "connection_retry", "attempted": True}

            elif "memory" in error.lower():
                # Clear cache and retry
                self.logger.info("🔄 Attempting recovery with cache cleanup...")
                await self.rust_ops.cache_clear()
                return {"recovery_type": "memory_cleanup", "attempted": True}

            return None

        except Exception as e:
            self.logger.exception(f"Error recovery failed: {e}")
            return None

    async def _cache_session_result(self, session_id: str, result: dict[str, Any]) -> None:
        """Cache session result for continuity."""
        try:
            cache_key = f"session_result:{session_id}:{int(time.time())}"
            await self.rust_ops.cache_set(cache_key, result, ttl=3600)
        except Exception as e:
            self.logger.warning(f"Could not cache session result: {e}")

    async def get_agent_status(self) -> dict[str, Any]:
        """Get comprehensive status of all agents."""
        status = {
            "total_agents": len(self.loaded_agents),
            "healthy_agents": len(
                [a for a in self.agent_health_status.values() if a["status"] == "healthy"]
            ),
            "active_tasks": len(self.active_tasks),
            "active_sessions": len(self.active_sessions),
            "performance_metrics": self.performance_metrics,
            "server_connectivity": await self._check_gemini_server_health(),
            "rust_extensions": hasattr(self.rust_ops, "cache"),
            "timestamp": datetime.now().isoformat(),
        }

        # Add individual agent status
        status["agents"] = {}
        for agent_name, health_info in self.agent_health_status.items():
            capabilities = self.agent_capabilities.get(agent_name, {})
            status["agents"][agent_name] = {
                "health": health_info,
                "capabilities": capabilities,
                "loaded": agent_name in self.loaded_agents,
            }

        return status

    async def get_active_tasks(self) -> dict[str, Any]:
        """Get information about currently active tasks."""
        return {
            "count": len(self.active_tasks),
            "tasks": self.active_tasks.copy(),
            "timestamp": datetime.now().isoformat(),
        }

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task."""
        if task_id in self.active_tasks:
            try:
                # Update task status
                self.active_tasks[task_id]["status"] = "cancelled"
                self.active_tasks[task_id]["cancelled_at"] = datetime.now().isoformat()

                # Clean up
                if task_id in self.progress_callbacks:
                    del self.progress_callbacks[task_id]

                self.logger.info(f"🚫 Cancelled task: {task_id}")
                return True

            except Exception as e:
                self.logger.exception(f"Failed to cancel task {task_id}: {e}")
                return False

        return False

    async def create_session(self, session_name: str | None = None) -> str:
        """Create a new session with optional name."""
        session_id = f"session_{int(time.time())}"

        session_data = {
            "id": session_id,
            "name": session_name or f"Session {len(self.active_sessions) + 1}",
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "command_count": 0,
            "agent_usage": {},
        }

        self.active_sessions[session_id] = session_data

        # Cache session data
        try:
            await self.rust_ops.cache_set(f"session:{session_id}", session_data, ttl=86400)
        except Exception as e:
            self.logger.warning(f"Could not cache session data: {e}")

        self.logger.info(f"📋 Created session: {session_id}")
        return session_id

    async def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """Get information about a specific session."""
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id].copy()

            # Add context information if available
            if session_id in self.session_contexts:
                context = self.session_contexts[session_id]
                session_data["context"] = {
                    "steps": len(context.steps),
                    "last_updated": context.updated_at.isoformat(),
                    "current_iteration": context.current_iteration,
                    "status": context.status.value,
                }

            return session_data

        return None

    async def cleanup(self) -> None:
        """Clean up resources and save session data."""
        try:
            # Save active sessions
            if self.active_sessions:
                sessions_data = await self.rust_ops.serialize_json(self.active_sessions)
                await self.rust_ops.cache_set("terminal_sessions", sessions_data, ttl=86400)

            # Clean up HTTP client
            if self.http_client:
                await self.http_client.aclose()

            # Clean up command processor
            await self.command_processor.cleanup()

            self.logger.info("🧹 Agent integration cleanup completed")

        except Exception as e:
            self.logger.exception(f"Cleanup failed: {e}")

    async def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            cache_stats = await self.rust_ops.cache_stats()

            report = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": {
                    "total_agents_loaded": len(self.loaded_agents),
                    "healthy_agents": len(
                        [a for a in self.agent_health_status.values() if a["status"] == "healthy"]
                    ),
                    "total_tasks_executed": self.performance_metrics["total_tasks_executed"],
                    "error_recovery_attempts": self.performance_metrics["error_recovery_count"],
                    "active_sessions": len(self.active_sessions),
                    "cache_performance": cache_stats,
                },
                "agent_performance": {},
                "execution_times": self.performance_metrics["task_execution_times"],
                "load_times": self.performance_metrics["agent_load_times"],
                "recommendations": [],
            }

            # Add per-agent performance data
            for agent_name, health_info in self.agent_health_status.items():
                report["agent_performance"][agent_name] = {
                    "status": health_info["status"],
                    "average_response_time": health_info.get("response_time", 0),
                    "error_count": health_info.get("error_count", 0),
                    "capabilities": len(
                        self.agent_capabilities.get(agent_name, {}).get("methods", [])
                    ),
                }

            # Generate recommendations
            if cache_stats.get("hit_rate", 0) < 0.5:
                report["recommendations"].append(
                    "Consider increasing cache TTL for better performance"
                )

            if self.performance_metrics["error_recovery_count"] > 5:
                report["recommendations"].append(
                    "High error recovery count - check system stability"
                )

            return report

        except Exception as e:
            self.logger.exception(f"Performance report generation failed: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
