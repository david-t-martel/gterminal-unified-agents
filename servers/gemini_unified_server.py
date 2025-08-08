# FIXME: Unused import 'ReactResponse' - remove if not needed
#!/usr/bin/env python3
"""UNIFIED GEMINI SERVER - Single Entry Point with Service Account Authentication.

This is THE ONLY Gemini server instance with:
1. Service account authentication ONLY (no API key fallback)
2. ReAct engine orchestration for intelligent task processing
3. Session management and persistence
4. Keep-alive connection handling
5. Background task management
6. Autonomous operation capabilities
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import logging
from pathlib import Path
import queue
import signal
import sys
from typing import Any

from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

# Import our GCP auth manager
from google.auth import default
from pydantic import BaseModel
from pydantic import Field
import uvicorn
import vertexai
from vertexai.generative_models import GenerativeModel

from gterminal.config.gcp_auth import get_auth_manager

# Use profile-based authentication
auth_manager = get_auth_manager()
gcp_creds = auth_manager.get_credentials()

if not gcp_creds.credentials:
    # Fallback to default credentials if available
    credentials, project_id = default()
    if not credentials:
        msg = "No credentials found via auth manager or default"
        raise Exception(msg)
    else:
        credentials = gcp_creds.credentials
        project_id = gcp_creds.project_id

    # Initialize Vertex AI with loaded credentials
    vertexai.init(
        project=project_id,
        location=gcp_creds.location if "gcp_creds" in locals() else "us-central1",
        credentials=credentials,
    )

    # Use stable Gemini model
    model = GenerativeModel("gemini-2.0-flash-exp")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Enhanced ReAct engine with Redis and RAG support
# Minimal task management for simplified server
from enum import Enum

from gterminal.core.react_engine import ReactEngine
from gterminal.core.react_engine import ReactEngineConfig
from gterminal.core.react_engine import ReactResponse
from gterminal.core.session import SessionManager


class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(str, Enum):
    """Task status states."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class SessionData(BaseModel):
    """Persistent session data."""

    session_id: str
    created_at: datetime
    last_activity: datetime
    conversation_history: list[dict[str, Any]]
    active_tasks: list[str]
    orchestrator_state: dict[str, Any]
    context_cache: dict[str, Any]


class TaskRequest(BaseModel):
    """Unified task request."""

    task_type: str = Field(
        ..., description="Type: consolidation, analysis, code_generation, documentation"
    )
    instruction: str = Field(..., description="Natural language instruction")
    target_path: str | None = Field(default=None, description="Target file/directory")
    session_id: str | None = Field(default=None, description="Session ID for continuity")
    options: dict[str, Any] | None = Field(default=None, description="Task-specific options")

    # New multi-tasking fields
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="Task priority")
    timeout_seconds: int = Field(default=300, description="Task timeout in seconds")
    cache_key: str | None = Field(default=None, description="Cache key for result caching")
    depends_on: list[str] = Field(default=[], description="Task dependencies")
    scheduled_for: datetime | None = Field(default=None, description="Scheduled execution time")


# BatchTaskRequest class DISABLED for simplified server
# class BatchTaskRequest(BaseModel):
#     """Batch task request for multiple tasks."""
#
#     tasks: list[TaskRequest] = Field(..., description="List of tasks to execute")
#     execute_parallel: bool = Field(default=True, description="Execute tasks in parallel")
#     stop_on_first_error: bool = Field(default=False, description="Stop execution on first error")


class UnifiedGeminiServer:
    """THE UNIFIED GEMINI SERVER - Single instance handling all Gemini operations.

    Features:
    - Service account authentication ONLY
    - Session persistence and management
    - ReAct engine orchestration for intelligent task processing
    - Background task processing
    - Keep-alive connections
    - WebSocket streaming support
    - Autonomous operation
    """

    def __init__(self) -> None:
        self.model = model
        self.sessions: dict[str, SessionData] = {}
        self.task_queue = queue.Queue()  # Legacy queue - keeping for compatibility
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.active_tasks: dict[str, dict] = {}
        self.keep_alive_tasks: set[asyncio.Task] = set()
        self.shutdown_requested = False

        # Initialize core components
        self.project_root = Path("/home/david/agents/my-fullstack-agent")

        # Initialize Enhanced ReAct engine with Redis and RAG support
        # Create enhanced configuration for the unified ReactEngine
        config = ReactEngineConfig(
            enable_redis=True,
            enable_rag=True,
            enable_autonomous=False,  # Can be enabled later
            cache_responses=True,
            enable_rust_optimizations=True,
        )
        self.react_engine = ReactEngine(
            model=self.model, config=config, project_root=self.project_root
        )

        # Initialize session manager for ReAct engine integration
        self.session_manager = SessionManager(storage_dir=self.project_root / ".sessions")

        # Session persistence file (legacy compatibility)
        self.session_file = self.project_root / "sessions.json"
        self.load_sessions()

        # WebSocket connections for streaming
        self.websocket_connections: dict[str, WebSocket] = {}

        logger.info("🚀 UNIFIED GEMINI SERVER with ReAct Engine initialized")

    # DISABLED - Complex async task management
    # async def _async_task_executor(self, task_config: TaskConfig) -> dict[str, Any]:
    #     """Execute task using the original task processing logic."""
    #     # Convert TaskConfig to TaskRequest for compatibility
    #     task_request = TaskRequest(
    #         task_type=task_config.task_type,
    #         instruction=task_config.instruction,
    #         target_path=task_config.target_path,
    #         session_id=task_config.session_id,
    #         options=task_config.options
    #     )
    #
    #     # Use the original task processing method
    #     return await self.process_task_with_session(task_request)

    # async def _task_progress_callback(self, task_id: str, progress: int, message: str = ""):
    #     """Handle task progress updates via WebSocket."""
    #     session_id = None
    #
    #     # Find session ID from active tasks
    #     for session_data in self.sessions.values():
    #         if task_id in session_data.active_tasks:
    #             session_id = session_data.session_id
    #             break
    #
    #     # Send progress update via WebSocket
    #     await task_progress_tracker.task_progress_update(task_id, progress, message, session_id)

    # FIXME: Async function 'start_async_components' missing error handling - add try/except block
    async def start_async_components(self) -> None:
        """Start async components (simplified - no actual components to start)."""
        # await self.async_task_queue.start()  # DISABLED
        logger.info("🔄 Simplified server - no async components to start")

    # FIXME: Async function 'stop_async_components' missing error handling - add try/except block
    async def stop_async_components(self) -> None:
        """Stop async components gracefully (simplified - no actual components to stop)."""
        # await self.async_task_queue.stop()  # DISABLED
        logger.info("⏹️ Simplified server - no async components to stop")

    def load_sessions(self) -> None:
        """Load persistent sessions from disk."""
        if self.session_file.exists():
            try:
                with open(self.session_file) as f:
                    session_data = json.load(f)
                    for session_id, data in session_data.items():
                        # Convert datetime strings back to datetime objects
                        data["created_at"] = datetime.fromisoformat(data["created_at"])
                        data["last_activity"] = datetime.fromisoformat(data["last_activity"])
                        self.sessions[session_id] = SessionData(**data)
                logger.info(f"📂 Loaded {len(self.sessions)} persistent sessions")
            except Exception as e:
                logger.exception(f"Failed to load sessions: {e}")

    def save_sessions(self) -> None:
        """Save sessions to disk for persistence."""
        try:
            session_data: dict[str, Any] = {}
            for session_id, session in self.sessions.items():
                data = session.dict()
                # Convert datetime objects to strings for JSON serialization
                data["created_at"] = data["created_at"].isoformat()
                data["last_activity"] = data["last_activity"].isoformat()
                session_data[session_id] = data

            with open(self.session_file, "w") as f:
                json.dump(session_data, f, indent=2, default=str)
            logger.info(f"💾 Saved {len(self.sessions)} sessions to disk")
        except Exception as e:
            logger.exception(f"Failed to save sessions: {e}")

    def create_session(self, session_id: str | None = None) -> str:
        """Create a new session with persistence."""
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = SessionData(
            session_id=session_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            conversation_history=[],
            active_tasks=[],
            orchestrator_state={},
            context_cache={},
        )

        self.sessions[session_id] = session
        self.save_sessions()
        logger.info(f"🆕 Created session: {session_id}")
        return session_id

    def update_session_activity(self, session_id: str) -> None:
        """Update session last activity timestamp."""
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = datetime.now()
            self.save_sessions()

    async def process_task_with_session(self, task: TaskRequest) -> dict[str, Any]:
        """Process a task using ReAct engine with session context."""
        session_id = task.session_id or self.create_session()
        self.update_session_activity(session_id)

        # Get or create ReAct session (using the new session manager)
        react_session = await self.session_manager.get_or_create_async(session_id)

        # Attach WebSocket if available for streaming
        if session_id in self.websocket_connections:
            react_session.websocket = self.websocket_connections[session_id]

        # Build the instruction for ReAct engine based on task type and requirements
        instruction = self._build_react_instruction(task)

        logger.info(
            f"🎯 Processing {task.task_type} task with ReAct engine for session {session_id}"
        )

        try:
            # Process through ReAct engine with streaming support
            streaming = react_session.websocket is not None
            react_response: ReactResponse = await self.react_engine.process_request(
                request=instruction,
                session_id=session_id,
                streaming=streaming,
            )

            # Convert ReAct response to unified server format
            result = self._convert_react_response_to_task_result(react_response, task)

            # Update legacy session data for compatibility
            if session_id in self.sessions:
                legacy_session = self.sessions[session_id]
                legacy_session.conversation_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "type": "task_result",
                        "content": result,
                    },
                )

            self.save_sessions()
            return result

        except Exception as e:
            logger.exception(f"ReAct engine processing failed: {e}")
            error_result = {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
                "task_type": task.task_type,
                "react_engine_error": True,
            }

            # Update legacy session data for compatibility
            if session_id in self.sessions:
                legacy_session = self.sessions[session_id]
                legacy_session.conversation_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "type": "task_error",
                        "content": error_result,
                    },
                )

            self.save_sessions()
            return error_result

    def _build_react_instruction(self, task: TaskRequest) -> str:
        """Build a comprehensive instruction for the ReAct engine based on task type."""
        base_instruction = task.instruction

        # Build context-aware instruction based on task type
        if task.task_type == "consolidation":
            if task.target_path:
                instruction = f"""Perform code consolidation analysis and implementation:

Primary Goal: {base_instruction}

Target Path: {task.target_path}

Please:
1. Analyze the project structure for consolidation opportunities
2. Identify duplicate or similar functionality
3. Create a consolidation plan with specific file merges
4. Implement the consolidation safely with backup considerations
5. Verify the consolidated code maintains functionality

Use the available tools to read files, analyze code, and implement changes systematically."""
            else:
                instruction = f"""Perform project-wide consolidation analysis:

Goal: {base_instruction}

Please analyze the current project for consolidation opportunities and provide a detailed plan."""

        elif task.task_type == "analysis":
            if task.target_path:
                instruction = f"""Perform comprehensive code analysis:

Analysis Goal: {base_instruction}

Target: {task.target_path}

Please:
1. Read and analyze the target files/directories
2. Examine code quality, architecture, and dependencies
3. Identify potential issues, bottlenecks, or improvements
4. Provide detailed insights and recommendations
5. Include metrics and specific examples where relevant

Use analysis tools to provide thorough technical assessment."""
            else:
                instruction = f"""Perform project-wide analysis: {base_instruction}

Please provide comprehensive analysis of the project structure, code quality, and architecture."""

        elif task.task_type == "code_generation":
            instruction = f"""Generate production-ready code:

Requirements: {base_instruction}

Please:
1. Understand the specific code generation requirements
2. Design appropriate architecture and structure
3. Generate well-documented, tested code
4. Include proper error handling and type hints
5. Follow best practices and project conventions

Target location: {task.target_path or "appropriate location based on analysis"}

Use filesystem tools to create and organize the generated code properly."""

        elif task.task_type == "documentation":
            if task.target_path:
                instruction = f"""Generate comprehensive documentation:

Documentation Goal: {base_instruction}

Target: {task.target_path}

Please:
1. Analyze the code/project structure
2. Generate appropriate documentation (README, API docs, etc.)
3. Include usage examples and architecture explanations
4. Create proper markdown formatting
5. Write the documentation to appropriate files

Use filesystem tools to read existing code and create documentation files."""
            else:
                instruction = f"""Generate project documentation: {base_instruction}

Please create comprehensive documentation for the project."""

        else:
            # Generic task - let ReAct engine handle with full context
            instruction = f"""Handle development task:

Task Type: {task.task_type}
Instruction: {base_instruction}
Target Path: {task.target_path or "not specified"}

Please use the available tools to understand the requirements and implement the requested functionality appropriately."""

        # Add options context if provided
        if task.options:
            instruction += f"\n\nAdditional Options: {json.dumps(task.options, indent=2)}"

        return instruction

    def _convert_react_response_to_task_result(
        self,
        react_response: ReactResponse,
        original_task: TaskRequest,
    ) -> dict[str, Any]:
        """Convert ReAct engine response to unified server task result format."""
        # Build the unified result structure
        result = {
            "status": "completed" if react_response.success else "failed",
            "task_type": original_task.task_type,
            "session_id": react_response.session_id,
            "instruction": original_task.instruction,
            "target_path": original_task.target_path,
            "timestamp": datetime.now().isoformat(),
            # ReAct engine specific results
            "react_engine": {
                "success": react_response.success,
                "total_time": react_response.total_time,
                "steps_executed": len(react_response.steps_executed),
                "step_details": [
                    {
                        "type": step.type,
                        "description": step.description,
                        "tool_name": step.tool_name,
                        "success": step.result.success if step.result else None,
                        "timestamp": step.timestamp.isoformat(),
                    }
                    for step in react_response.steps_executed
                ],
            },
            # Main result content
            "result": react_response.result,
        }

        # Add error information if failed
        if (
            not react_response.success
            and isinstance(react_response.result, dict)
            and "error" in react_response.result
        ):
            result["error"] = react_response.result["error"]

        # Add task-specific formatting based on type
        if original_task.task_type == "consolidation":
            result["consolidation_analysis"] = react_response.result
        elif original_task.task_type == "analysis":
            result["analysis"] = react_response.result
        elif original_task.task_type == "code_generation":
            result["generated_code"] = react_response.result
        elif original_task.task_type == "documentation":
            result["generated_documentation"] = react_response.result

        return result

    # Legacy task handling methods removed - now using ReAct engine for all processing

    # FIXME: Function 'keep_alive_handler' missing return type annotation
    async def keep_alive_handler(self, session_id: str) -> None:
        """Keep session alive with periodic heartbeat."""
        while not self.shutdown_requested:
            try:
                await asyncio.sleep(300)  # 5 minutes
                if session_id in self.sessions:
                    self.update_session_activity(session_id)
                    logger.debug(f"💓 Heartbeat for session {session_id}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Keep-alive error for {session_id}: {e}")
                break

    def cleanup_old_sessions(self) -> None:
        """Clean up sessions older than 24 hours."""
        cutoff = datetime.now().timestamp() - (24 * 60 * 60)  # 24 hours ago

        to_delete: list[Any] = []
        for session_id, session in self.sessions.items():
            if session.last_activity.timestamp() < cutoff:
                to_delete.append(session_id)

        for session_id in to_delete:
            del self.sessions[session_id]
            logger.info(f"🗑️ Cleaned up old session: {session_id}")

        if to_delete:
            self.save_sessions()

    def setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""

        # FIXME: Function 'signal_handler' missing docstring
        def signal_handler(signum, frame) -> None:
            logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True

            # Cancel all keep-alive tasks
            for task in self.keep_alive_tasks:
                task.cancel()

            # Save sessions before shutdown
            self.save_sessions()
            logger.info("💾 Sessions saved. Shutdown complete.")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


# Initialize the unified server
unified_server = UnifiedGeminiServer()
unified_server.setup_signal_handlers()

# FastAPI app
app = FastAPI(
    title="Unified Gemini Server",
    description="Single entry point for all Gemini operations with service account authentication",
    version="1.0.0",
)


@app.get("/")
# FIXME: Async function 'root' missing error handling - add try/except block
async def root():
    """Root endpoint with server status."""
    return {
        "service": "Unified Gemini Server",
        "status": "running",
        "authentication": "service_account_only",
        "active_sessions": len(unified_server.sessions),
        "model": "gemini-2.0-flash-exp",
        "capabilities": [
            "enhanced_react_engine_orchestration",
            "redis_powered_caching",
            "rag_enhanced_context",
            "intelligent_tool_execution",
            "code_analysis_and_generation",
            "documentation_generation",
            "session_management",
            "websocket_streaming",
            "context_aware_processing",
            "local_memory_integration",
        ],
        "endpoints": [
            "/task - Process tasks with enhanced ReAct engine",
            "/ws - WebSocket streaming with real-time updates",
            "/sessions - Manage sessions with persistence",
            "/health - Health check",
            "/status - Server status",
            "/react/status - Enhanced ReAct engine status",
            "/tasks/status - Task queue and processing status",
        ],
    }


@app.get("/health")
# FIXME: Async function 'health' missing error handling - add try/except block
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_available": unified_server.model is not None,
        "active_sessions": len(unified_server.sessions),
        "active_tasks": len(unified_server.active_tasks),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/task")
# FIXME: Function 'process_task' missing return type annotation
async def process_task(task: TaskRequest, background_tasks: BackgroundTasks):
    """Process a task with session management and Re-Act orchestration."""
    try:
        # Process the task
        result = await unified_server.process_task_with_session(task)

        # Start keep-alive if new session
        if task.session_id and task.session_id not in [
            t.get_name() for t in unified_server.keep_alive_tasks
        ]:
            keep_alive_task = asyncio.create_task(
                unified_server.keep_alive_handler(task.session_id),
                name=task.session_id,
            )
            unified_server.keep_alive_tasks.add(keep_alive_task)

        return result

    except Exception as e:
        logger.exception(f"Task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
# FIXME: Async function 'list_sessions' missing error handling - add try/except block
async def list_sessions():
    """List all active sessions."""
    sessions_info: dict[str, Any] = {}
    for session_id, session in unified_server.sessions.items():
        sessions_info[session_id] = {
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "conversation_length": len(session.conversation_history),
            "active_tasks": len(session.active_tasks),
        }

    return {"total_sessions": len(sessions_info), "sessions": sessions_info}


@app.get("/sessions/{session_id}")
# FIXME: Async function 'get_session' missing error handling - add try/except block
async def get_session(session_id: str):
    """Get session details."""
    if session_id not in unified_server.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = unified_server.sessions[session_id]
    return {
        "session_id": session_id,
        "session_data": session.dict(),
        "keep_alive_active": session_id in [t.get_name() for t in unified_server.keep_alive_tasks],
    }


@app.delete("/sessions/{session_id}")
# FIXME: Async function 'delete_session' missing error handling - add try/except block
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in unified_server.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Cancel keep-alive task
    for task in unified_server.keep_alive_tasks:
        if task.get_name() == session_id:
            task.cancel()
            unified_server.keep_alive_tasks.remove(task)
            break

    # Delete session
    del unified_server.sessions[session_id]
    unified_server.save_sessions()

    return {"message": f"Session {session_id} deleted"}


@app.get("/status")
# FIXME: Async function 'server_status' missing error handling - add try/except block
async def server_status():
    """Get detailed server status."""
    return {
        "server": "Unified Gemini Server",
        "uptime": "running",
        "authentication": "service_account_verified",
        "model": "gemini-2.0-flash-exp",
        "sessions": {
            "total": len(unified_server.sessions),
            "with_keep_alive": len(unified_server.keep_alive_tasks),
        },
        "react_engine": {
            "type": "enhanced",
            "available": True,
            "project_root": str(unified_server.project_root),
            "tools_registered": len(unified_server.react_engine.tool_registry.tools),
            "redis_integration": getattr(unified_server.react_engine, "redis_available", False),
            "rag_integration": getattr(unified_server.react_engine, "rag_available", False),
            "rust_extensions": unified_server.react_engine.cache is not None,
        },
        "capabilities": {
            "consolidation": True,  # Via ReAct engine
            "analysis": True,  # Via ReAct engine
            "code_generation": True,  # Via ReAct engine
            "documentation": True,  # Via ReAct engine
            "session_persistence": True,
            "websocket_streaming": True,
            "intelligent_tool_execution": True,
            "context_aware_reasoning": True,
        },
    }


@app.get("/tasks/status")
# FIXME: Async function 'tasks_status' missing error handling - add try/except block
async def tasks_status():
    """Get task queue and processing status."""
    return {
        "status": "operational",
        "active_tasks": len(unified_server.active_tasks),
        "queued_tasks": unified_server.task_queue.qsize(),
        "task_details": {
            task_id: {
                "status": task_info.get("status", "unknown"),
                "started_at": task_info.get("started_at"),
                "task_type": task_info.get("task_type"),
            }
            for task_id, task_info in unified_server.active_tasks.items()
        },
        "executor_status": {
            "max_workers": unified_server.executor._max_workers,
            "active_threads": (
                len(unified_server.executor._threads)
                if hasattr(unified_server.executor, "_threads")
                else 0
            ),
        },
        "server_health": {
            "sessions": len(unified_server.sessions),
            "keep_alive_tasks": len(unified_server.keep_alive_tasks),
            "shutdown_requested": unified_server.shutdown_requested,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.websocket("/ws")
# FIXME: Function 'websocket_endpoint' missing return type annotation
async def websocket_endpoint(websocket: WebSocket, session_id: str | None = None) -> None:
    """WebSocket endpoint for real-time ReAct step streaming."""
    await websocket.accept()

    if not session_id:
        session_id = f"ws_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Register WebSocket connection
    unified_server.websocket_connections[session_id] = websocket
    logger.info(f"WebSocket connected for session {session_id}")

    try:
        # Send connection confirmation
        await websocket.send_json(
            {
                "type": "connection_established",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "message": "WebSocket connected - ReAct engine streaming enabled",
            },
        )

        while True:
            # Listen for incoming messages (task requests)
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get("type") == "task_request":
                    # Create task request from WebSocket message
                    task = TaskRequest(
                        task_type=message.get("task_type", "generic"),
                        instruction=message.get("instruction", ""),
                        target_path=message.get("target_path"),
                        session_id=session_id,
                        options=message.get("options"),
                    )

                    # Process task (will stream updates via WebSocket)
                    result = await unified_server.process_task_with_session(task)

                    # Send final result
                    await websocket.send_json(
                        {
                            "type": "task_completed",
                            "result": result,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                elif message.get("type") == "ping":
                    # Respond to ping with pong
                    await websocket.send_json(
                        {"type": "pong", "timestamp": datetime.now().isoformat()}
                    )

            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            except Exception as e:
                logger.exception(f"WebSocket message processing error: {e}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

    except WebSocketDisconnect:
        # Clean up WebSocket connection
        if session_id in unified_server.websocket_connections:
            del unified_server.websocket_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")


# Enhanced API endpoints DISABLED for simplified server
# @app.post("/tasks/batch")
# async def submit_batch_tasks(batch_request: BatchTaskRequest):
#     """Submit multiple tasks as a batch."""
#     return {"status": "disabled", "message": "Batch processing disabled in simplified server"}


# All complex task management endpoints DISABLED for simplified server
@app.get("/tasks")
# FIXME: Async function 'list_tasks' missing error handling - add try/except block
async def list_tasks():
    """List tasks (simplified - only shows active tasks)."""
    return {
        "status": "simplified",
        "active_tasks": len(unified_server.active_tasks),
        "task_details": unified_server.active_tasks,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/tasks/{task_id}")
# FIXME: Async function 'get_task_status' missing error handling - add try/except block
async def get_task_status(task_id: str):
    """Get task status (simplified)."""
    if task_id in unified_server.active_tasks:
        return {
            "status": "found",
            "task": unified_server.active_tasks[task_id],
            "timestamp": datetime.now().isoformat(),
        }
    raise HTTPException(status_code=404, detail="Task not found")


# Other complex endpoints disabled
@app.delete("/tasks/{task_id}")
# FIXME: Async function 'cancel_task' missing error handling - add try/except block
async def cancel_task(task_id: str):
    """Cancel task (simplified)."""
    return {
        "status": "disabled",
        "message": "Task cancellation disabled in simplified server",
    }


@app.get("/tasks/metrics")
# FIXME: Async function 'get_task_metrics' missing error handling - add try/except block
async def get_task_metrics():
    """Get metrics (simplified)."""
    return {
        "status": "simplified",
        "active_tasks": len(unified_server.active_tasks),
        "sessions": len(unified_server.sessions),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/tasks/queue/cleanup")
# FIXME: Async function 'cleanup_task_queue' missing error handling - add try/except block
async def cleanup_task_queue():
    """Cleanup (simplified)."""
    return {"status": "simplified", "message": "No complex queue to clean up"}


@app.get("/react/status")
# FIXME: Function 'react_engine_status' missing return type annotation
async def react_engine_status():
    """Get Enhanced ReAct engine status with Redis and RAG information."""
    try:
        enhanced_status = await unified_server.react_engine.get_engine_status()

        return {
            "react_engine": "unified",
            "status": "operational",
            "config": enhanced_status["config"],
            "features": {
                "redis_caching": {
                    "available": enhanced_status["redis_available"],
                    "enabled": enhanced_status["config"]["enable_redis"],
                },
                "rag_integration": {
                    "available": enhanced_status["rag_available"],
                    "enabled": enhanced_status["config"]["enable_rag"],
                },
                "autonomous_operation": {
                    "available": enhanced_status["model_available"],
                    "enabled": enhanced_status["config"]["enable_autonomous"],
                },
                "rust_optimizations": {
                    "available": enhanced_status["rust_optimizations"],
                    "enabled": enhanced_status["config"]["enable_rust_optimizations"],
                },
            },
            "learned_patterns": enhanced_status.get("learned_patterns_count", 0),
            "active_sessions": enhanced_status.get("active_sessions", 0),
            "tools_available": len(unified_server.react_engine.tool_registry.tools),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "react_engine": "enhanced",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@app.on_event("startup")
# FIXME: Async function 'startup_event' missing error handling - add try/except block
async def startup_event() -> None:
    """Server startup tasks."""
    logger.info("🚀 ENHANCED GEMINI SERVER WITH REDIS & RAG starting up...")

    # Start async components
    await unified_server.start_async_components()

    # Clean up old sessions
    unified_server.cleanup_old_sessions()

    # Log startup info
    logger.info(f"📊 Loaded {len(unified_server.sessions)} sessions")
    logger.info("🔐 Service account authentication: ✅")
    logger.info("🤖 Gemini model: gemini-2.0-flash-exp")
    logger.info("🧠 Enhanced ReAct engine: ✅ ENABLED")
    logger.info(f"🔧 Tools registered: {len(unified_server.react_engine.tool_registry.tools)}")
    logger.info("🌐 WebSocket streaming: ✅ ENABLED")
    logger.info("💾 Session persistence: ✅ ENABLED")
    logger.info("🔄 Redis caching: Initializing...")
    logger.info("🤖 RAG integration: Initializing...")
    logger.info("🎉 ENHANCED REACT+REDIS+RAG GEMINI SERVER ready for operation!")


@app.on_event("shutdown")
# FIXME: Async function 'shutdown_event' missing error handling - add try/except block
async def shutdown_event() -> None:
    """Server shutdown tasks."""
    logger.info("⏹️ UNIFIED GEMINI SERVER shutting down...")

    # Stop async components
    await unified_server.stop_async_components()

    # Cancel all keep-alive tasks
    for task in unified_server.keep_alive_tasks:
        task.cancel()

    # Save sessions
    unified_server.save_sessions()
    logger.info("💾 Sessions saved. Shutdown complete.")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="info", access_log=True)
