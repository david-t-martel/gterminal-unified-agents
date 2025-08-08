#!/usr/bin/env python3
"""FastAPI Server for My Fullstack Agent.

Production-ready API server with ReAct engine integration and WebSocket support.
"""

from contextlib import asynccontextmanager
from contextlib import suppress
from datetime import datetime
import logging
import os
from typing import Annotated, Any

from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from pydantic import Field
import uvicorn
from vertexai.generative_models import GenerativeModel

from gterminal.agent_endpoints import router as agent_router
from gterminal.agents import get_agent_service
from gterminal.automation.auth_middleware import AuthMiddleware
from gterminal.automation.auth_middleware import get_current_user
from gterminal.core.react_engine import ReactEngine
from gterminal.health_app import health_router

# Import monitoring system
from gterminal.monitoring.integrated_monitoring import IntegratedMonitoringSystem
from gterminal.monitoring.integrated_monitoring import MonitoringConfig
from gterminal.monitoring.unified_dashboard import UnifiedMonitoringDashboard
from gterminal.monitoring_middleware import MonitoringMiddleware
from gterminal.security import CORSSecurityMiddleware
from gterminal.security import SecurityHeadersMiddleware

logger = logging.getLogger(__name__)

# Global monitoring system instance
monitoring_system = None
monitoring_dashboard = None

# Global ReAct engine instance
react_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global monitoring_system, monitoring_dashboard, react_engine

    # Startup

    # Initialize monitoring system
    try:
        monitoring_config = MonitoringConfig(
            apm_service_name="fullstack-agent-api",
            enable_cross_system_correlation=True,
            performance_budget_enforcement=True,
        )
        monitoring_system = IntegratedMonitoringSystem(monitoring_config)
        monitoring_dashboard = UnifiedMonitoringDashboard(monitoring_system)
    except Exception:
        pass

    # Initialize ReAct engine
    try:
        # Try to initialize with Gemini model
        model = None
        try:
            from pathlib import Path

            import google.auth

            # Check if credentials are available
            credentials, project = google.auth.default()
            if credentials:
                model = GenerativeModel("gemini-2.0-flash-exp")
        except Exception:
            pass

        react_engine = ReactEngine(
            model=model, project_root=Path("/home/david/agents/my-fullstack-agent")
        )
    except Exception:
        pass

    # Initialize services
    try:
        # Load agents
        agent_types = [
            "code-reviewer",
            "workspace-analyzer",
            "documentation-generator",
            "master-architect",
            "code-generator",
        ]

        for agent_type in agent_types:
            with suppress(Exception):
                get_agent_service(agent_type)

    except Exception:
        pass

    yield

    # Shutdown
    if monitoring_system:
        await monitoring_system.close()
    if monitoring_dashboard:
        await monitoring_dashboard.close()
    if react_engine:
        # Save all active sessions
        with suppress(Exception):
            await react_engine.session_manager.save_all()


# Create FastAPI app
app = FastAPI(
    title="My Fullstack Agent API",
    description="Comprehensive AI Agent Framework with MCP Integration",
    version="0.2.0",
    lifespan=lifespan,
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add secure CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
# SECURITY: Never use "*" in production
if "*" in cors_origins:
    import warnings

    warnings.warn(
        "Using wildcard CORS origins is insecure. Set specific origins in CORS_ORIGINS environment variable.",
        stacklevel=2,
    )

app.add_middleware(
    CORSSecurityMiddleware,
    allowed_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allowed_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowed_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-API-Key",
    ],
    allow_credentials=True,
    max_age=600,  # 10 minutes cache for preflight requests
)

# Add authentication middleware
app.add_middleware(AuthMiddleware)

# Add monitoring middleware (after auth but before instrumentation)
app.add_middleware(MonitoringMiddleware, monitoring_system=monitoring_system)

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Include routers
app.include_router(agent_router, prefix="/api/v1")
app.include_router(health_router, prefix="/health")

# Add monitoring endpoints
from gterminal.monitoring_endpoints import router as monitoring_router

app.include_router(monitoring_router, prefix="/api/v1/monitoring")

# Add enhanced auto-claude endpoints
from gterminal.auto_claude_endpoints import router as auto_claude_router

app.include_router(auto_claude_router, prefix="/api/v1/auto-claude")


# ========================
# ReAct Engine Models
# ========================


class TaskRequest(BaseModel):
    """Request model for ReAct task execution."""

    task: str = Field(..., description="Natural language task description")
    session_id: str | None = Field(None, description="Session ID for context")
    streaming: bool = Field(False, description="Enable real-time streaming")


class TaskResponse(BaseModel):
    """Response model for ReAct task execution."""

    success: bool
    result: Any
    session_id: str
    steps_count: int
    execution_time: float
    streaming_available: bool = False


class SessionInfo(BaseModel):
    """Session information model."""

    id: str
    created_at: str
    last_activity: str
    interaction_count: int
    has_websocket: bool


class ToolInfo(BaseModel):
    """Tool information model."""

    name: str
    description: str
    category: str
    parameters: list[dict[str, Any]]


# ========================
# WebSocket Connection Manager
# ========================


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket

        # Attach WebSocket to session if ReAct engine is available
        if react_engine:
            session = await react_engine.session_manager.get_or_create_async(session_id)
            session.websocket = websocket
            logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, session_id: str) -> None:
        """Remove WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

            # Remove WebSocket from session
            if react_engine and session_id in react_engine.session_manager.sessions:
                session = react_engine.session_manager.sessions[session_id]
                session.websocket = None
                logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_personal_message(self, message: dict[str, Any], session_id: str) -> None:
        """Send message to specific session."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.exception(f"Failed to send message to session {session_id}: {e}")
                self.disconnect(session_id)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        disconnected: list[Any] = []
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.exception(f"Failed to broadcast to session {session_id}: {e}")
                disconnected.append(session_id)

        # Clean up disconnected clients
        for session_id in disconnected:
            self.disconnect(session_id)


manager = ConnectionManager()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "My Fullstack Agent",
        "version": "0.2.0",
        "status": "operational",
        "features": {
            "react_engine": react_engine is not None,
            "gemini_model": react_engine.model is not None if react_engine else False,
            "websocket_support": True,
            "tool_count": len(react_engine.tool_registry.tools) if react_engine else 0,
        },
        "endpoints": {
            "api": "/api/v1",
            "task": "/api/task",
            "tools": "/api/tools",
            "sessions": "/api/sessions",
            "websocket": "/ws/{session_id}",
            "health": "/health",
            "monitoring": "/api/v1/monitoring",
            "metrics": "/metrics",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get("/api/v1")
async def api_root():
    """API root endpoint."""
    return {
        "agents": {
            "code-reviewer": "/api/v1/agents/code-reviewer",
            "workspace-analyzer": "/api/v1/agents/workspace-analyzer",
            "documentation-generator": "/api/v1/agents/documentation-generator",
            "master-architect": "/api/v1/agents/master-architect",
            "code-generator": "/api/v1/agents/code-generator",
        },
        "react": {
            "task": "/api/task",
            "tools": "/api/tools",
            "sessions": "/api/sessions",
        },
        "jobs": "/api/v1/agents/jobs",
        "status": "/api/v1/agents/status",
    }


# ========================
# ReAct Engine Endpoints
# ========================


@app.post("/api/task", response_model=TaskResponse)
async def execute_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Execute a task using the ReAct engine."""
    if not react_engine:
        raise HTTPException(
            status_code=503, detail="ReAct engine not available. Please check server configuration."
        )

    try:
        logger.info(f"Executing task: {request.task[:100]}...")

        # Execute task through ReAct engine
        response = await react_engine.process_request(
            request=request.task,
            session_id=request.session_id,
            streaming=request.streaming,
        )

        return TaskResponse(
            success=response.success,
            result=response.result,
            session_id=response.session_id,
            steps_count=len(response.steps_executed),
            execution_time=response.total_time,
            streaming_available=request.streaming,
        )

    except Exception as e:
        logger.exception(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task execution failed: {e!s}")


@app.get("/api/tools", response_model=list[ToolInfo])
async def list_tools():
    """List all available tools in the tool registry."""
    if not react_engine:
        raise HTTPException(
            status_code=503, detail="ReAct engine not available. Please check server configuration."
        )

    try:
        tool_descriptions = react_engine.tool_registry.get_tool_descriptions()
        tools: list[Any] = []

        for name, desc in tool_descriptions.items():
            tools.append(
                ToolInfo(
                    name=name,
                    description=desc["description"],
                    category=desc["category"],
                    parameters=desc["parameters"],
                ),
            )

        return sorted(tools, key=lambda x: (x.category, x.name))

    except Exception as e:
        logger.exception(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {e!s}")


@app.get("/api/tools/categories")
async def list_tool_categories():
    """List all tool categories."""
    if not react_engine:
        raise HTTPException(
            status_code=503, detail="ReAct engine not available. Please check server configuration."
        )

    try:
        categories = react_engine.tool_registry.list_categories()
        return {"categories": categories}

    except Exception as e:
        logger.exception(f"Failed to list tool categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tool categories: {e!s}")


@app.get("/api/tools/{category}")
async def list_tools_by_category(category: str):
    """List tools in a specific category."""
    if not react_engine:
        raise HTTPException(
            status_code=503, detail="ReAct engine not available. Please check server configuration."
        )

    try:
        tools_in_category = react_engine.tool_registry.get_tools_by_category(category)

        tool_info: list[Any] = []
        for tool in tools_in_category:
            desc = tool.get_description()
            tool_info.append(
                ToolInfo(
                    name=desc.name,
                    description=desc.description,
                    category=desc.category,
                    parameters=[p.model_dump() for p in desc.parameters],
                ),
            )

        return {"category": category, "tools": tool_info}

    except Exception as e:
        logger.exception(f"Failed to list tools for category {category}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list tools for category {category}: {e!s}"
        )


# ========================
# Session Management Endpoints
# ========================


@app.get("/api/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """List all active sessions."""
    if not react_engine:
        raise HTTPException(
            status_code=503, detail="ReAct engine not available. Please check server configuration."
        )

    try:
        sessions = react_engine.session_manager.get_active_sessions()
        return [SessionInfo(**session) for session in sessions]

    except Exception as e:
        logger.exception(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {e!s}")


@app.get("/api/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get information about a specific session."""
    if not react_engine:
        raise HTTPException(
            status_code=503, detail="ReAct engine not available. Please check server configuration."
        )

    try:
        if session_id not in react_engine.session_manager.sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session = react_engine.session_manager.sessions[session_id]
        return SessionInfo(
            id=session.id,
            created_at=session.created_at.isoformat(),
            last_activity=session.last_activity.isoformat(),
            interaction_count=len(session.interactions),
            has_websocket=session.websocket is not None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {e!s}")


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if not react_engine:
        raise HTTPException(
            status_code=503, detail="ReAct engine not available. Please check server configuration."
        )

    try:
        if session_id not in react_engine.session_manager.sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Disconnect WebSocket if connected
        if session_id in manager.active_connections:
            manager.disconnect(session_id)

        # Save session before removing
        await react_engine.session_manager.save(session_id)
        react_engine.session_manager.remove(session_id)

        return {"message": f"Session {session_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {e!s}")


@app.post("/api/sessions/{session_id}/clear")
async def clear_session_history(session_id: str):
    """Clear the interaction history for a session."""
    if not react_engine:
        raise HTTPException(
            status_code=503, detail="ReAct engine not available. Please check server configuration."
        )

    try:
        session = await react_engine.session_manager.get_or_create_async(session_id)
        session.clear_interactions()
        await react_engine.session_manager.save(session_id)

        return {"message": f"Session {session_id} history cleared"}

    except Exception as e:
        logger.exception(f"Failed to clear session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {e!s}")


# ========================
# WebSocket Endpoint
# ========================


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time ReAct step streaming."""
    try:
        await manager.connect(websocket, session_id)

        # Send connection confirmation
        await manager.send_personal_message(
            {
                "type": "connection",
                "message": f"Connected to session {session_id}",
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
            },
            session_id,
        )

        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_json()

                # Handle different message types
                if data.get("type") == "ping":
                    await manager.send_personal_message(
                        {"type": "pong", "timestamp": datetime.now().isoformat()},
                        session_id,
                    )

                elif data.get("type") == "task" and react_engine:
                    # Execute task with streaming
                    task = data.get("task", "")
                    if task:
                        logger.info(f"WebSocket task execution: {task[:50]}...")

                        response = await react_engine.process_request(
                            request=task,
                            session_id=session_id,
                            streaming=True,
                        )

                        # Send final result
                        await manager.send_personal_message(
                            {
                                "type": "task_complete",
                                "response": {
                                    "success": response.success,
                                    "result": response.result,
                                    "session_id": response.session_id,
                                    "steps_count": len(response.steps_executed),
                                    "execution_time": response.total_time,
                                },
                                "timestamp": datetime.now().isoformat(),
                            },
                            session_id,
                        )

                else:
                    await manager.send_personal_message(
                        {
                            "type": "error",
                            "message": "Unknown message type or ReAct engine not available",
                            "timestamp": datetime.now().isoformat(),
                        },
                        session_id,
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.exception(f"WebSocket error for session {session_id}: {e}")
                await manager.send_personal_message(
                    {"type": "error", "message": str(e), "timestamp": datetime.now().isoformat()},
                    session_id,
                )

    except Exception as e:
        logger.exception(f"WebSocket connection error: {e}")
    finally:
        manager.disconnect(session_id)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code, "path": str(request.url)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    import traceback

    # Log the error
    traceback.print_exc()

    # Return error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if os.getenv("DEBUG") else "An error occurred",
            "path": str(request.url),
        },
    )


def main() -> None:
    """Run the server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("ENV", "production") == "development"

    uvicorn.run("app.server:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    main()
