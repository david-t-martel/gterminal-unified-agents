#!/usr/bin/env python3
"""API Adapter for Unified Agents.

Single FastAPI server exposing ALL unified agents via REST endpoints.
Consolidates multiple API servers into a unified interface with WebSocket support.
"""

from datetime import datetime
import json
import logging
import os
from typing import Any

from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import Field
import uvicorn

# Import all unified agents
from gterminal.core.agents.unified_code_reviewer import UnifiedCodeReviewer
from gterminal.core.agents.unified_documentation_generator import UnifiedDocumentationGenerator
from gterminal.core.agents.unified_gemini_orchestrator import UnifiedGeminiOrchestrator
from gterminal.core.agents.unified_workspace_analyzer import UnifiedWorkspaceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Unified Fullstack Agent API",
    description="Comprehensive AI Agent Framework with unified interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global agent instances
unified_code_reviewer = None
unified_workspace_analyzer = None
unified_documentation_generator = None
unified_orchestrator = None

# Global state for jobs and sessions
active_jobs: dict[str, Any] = {}
active_sessions: dict[str, Any] = {}
websocket_connections: dict[str, WebSocket] = {}


# ===========================
# Pydantic Models
# ===========================


class CodeReviewRequest(BaseModel):
    """Request for code review."""

    file_path: str = Field(..., description="Path to file to review")
    focus_areas: list[str] = Field(
        default=["security", "performance", "quality"], description="Areas to focus on"
    )
    include_suggestions: bool = Field(default=True, description="Include fix suggestions")
    severity_threshold: str = Field(default="medium", description="Minimum severity level")


class DirectoryReviewRequest(BaseModel):
    """Request for directory review."""

    directory_path: str = Field(..., description="Path to directory")
    file_patterns: list[str] = Field(
        default=["*.py", "*.js", "*.ts"], description="File patterns to review"
    )
    max_files: int = Field(default=100, description="Maximum files to review")
    focus_areas: list[str] = Field(default=["security", "performance"], description="Focus areas")


class WorkspaceAnalysisRequest(BaseModel):
    """Request for workspace analysis."""

    project_path: str = Field(..., description="Path to project directory")
    analysis_depth: str = Field(default="comprehensive", description="Analysis depth level")
    include_dependencies: bool = Field(default=True, description="Include dependency analysis")
    include_tests: bool = Field(default=True, description="Include test analysis")
    max_files: int = Field(default=1000, description="Maximum files to analyze")


class ProjectStructureRequest(BaseModel):
    """Request for project structure."""

    project_path: str = Field(..., description="Path to project directory")
    max_depth: int = Field(default=3, description="Maximum directory depth")
    include_hidden: bool = Field(default=False, description="Include hidden files/directories")


class DocumentationRequest(BaseModel):
    """Request for documentation generation."""

    source_path: str = Field(..., description="Path to source code or project")
    doc_type: str = Field(default="api", description="Type of documentation")
    output_format: str = Field(default="markdown", description="Output format")
    include_examples: bool = Field(default=True, description="Include examples")


class ReadmeRequest(BaseModel):
    """Request for README generation."""

    project_path: str = Field(..., description="Path to project directory")
    template_style: str = Field(default="comprehensive", description="README style")
    include_badges: bool = Field(default=True, description="Include status badges")
    include_installation: bool = Field(
        default=True, description="Include installation instructions"
    )


class OrchestrationRequest(BaseModel):
    """Request for multi-agent orchestration."""

    task: str = Field(..., description="Natural language task description")
    specific_agents: list[str] = Field(default=[], description="Specific agents to use")
    session_id: str | None = Field(None, description="Session ID for context")
    streaming: bool = Field(default=False, description="Enable streaming")


class JobResponse(BaseModel):
    """Response for job operations."""

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    progress: float = Field(..., description="Progress percentage")
    created_at: str = Field(..., description="Job creation time")
    result: Any = Field(None, description="Job result if completed")


class SessionResponse(BaseModel):
    """Response for session operations."""

    session_id: str = Field(..., description="Session identifier")
    created_at: str = Field(..., description="Session creation time")
    last_activity: str = Field(..., description="Last activity time")
    interaction_count: int = Field(..., description="Number of interactions")


# ===========================
# Startup/Shutdown Events
# ===========================


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize all unified agents on startup."""
    global unified_code_reviewer, unified_workspace_analyzer
    global unified_documentation_generator, unified_orchestrator

    logger.info("Initializing unified agents...")

    try:
        # Initialize agent instances
        unified_code_reviewer = UnifiedCodeReviewer()
        unified_workspace_analyzer = UnifiedWorkspaceAnalyzer()
        unified_documentation_generator = UnifiedDocumentationGenerator()
        unified_orchestrator = UnifiedGeminiOrchestrator()

        # Initialize each agent
        await unified_code_reviewer.initialize()
        await unified_workspace_analyzer.initialize()
        await unified_documentation_generator.initialize()
        await unified_orchestrator.initialize()

        logger.info("✅ All unified agents initialized successfully")

    except Exception as e:
        logger.exception(f"❌ Failed to initialize agents: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Shutdown all unified agents."""
    logger.info("Shutting down unified agents...")

    try:
        if unified_code_reviewer:
            await unified_code_reviewer.shutdown()
        if unified_workspace_analyzer:
            await unified_workspace_analyzer.shutdown()
        if unified_documentation_generator:
            await unified_documentation_generator.shutdown()
        if unified_orchestrator:
            await unified_orchestrator.shutdown()

        logger.info("✅ All unified agents shut down successfully")

    except Exception as e:
        logger.exception(f"❌ Error during agent shutdown: {e}")


# ===========================
# Root Endpoints
# ===========================


@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Unified Fullstack Agent API",
        "version": "1.0.0",
        "status": "operational",
        "agents": {
            "code_reviewer": "Available",
            "workspace_analyzer": "Available",
            "documentation_generator": "Available",
            "orchestrator": "Available",
        },
        "endpoints": {
            "code_review": "/api/v1/review",
            "workspace_analysis": "/api/v1/workspace",
            "documentation": "/api/v1/docs",
            "orchestration": "/api/v1/orchestrate",
            "jobs": "/api/v1/jobs",
            "sessions": "/api/v1/sessions",
            "websocket": "/ws/{session_id}",
            "docs": "/docs",
            "redoc": "/redoc",
        },
        "active_jobs": len(active_jobs),
        "active_sessions": len(active_sessions),
        "websocket_connections": len(websocket_connections),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents_initialized": all(
            [
                unified_code_reviewer is not None,
                unified_workspace_analyzer is not None,
                unified_documentation_generator is not None,
                unified_orchestrator is not None,
            ],
        ),
    }


# ===========================
# Code Review Endpoints
# ===========================


@app.post("/api/v1/review/file")
async def review_file(request: CodeReviewRequest, background_tasks: BackgroundTasks):
    """Review a single file for issues."""
    if not unified_code_reviewer:
        raise HTTPException(status_code=503, detail="Code reviewer not available")

    try:
        logger.info(f"Starting file review: {request.file_path}")

        # Execute review
        result = await unified_code_reviewer.review_file(
            file_path=request.file_path,
            focus_areas=request.focus_areas,
            include_suggestions=request.include_suggestions,
            severity_threshold=request.severity_threshold,
        )

        return {"success": True, "file_path": request.file_path, "review_results": result}

    except Exception as e:
        logger.exception(f"File review failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/review/directory")
async def review_directory(request: DirectoryReviewRequest, background_tasks: BackgroundTasks):
    """Review all files in a directory."""
    if not unified_code_reviewer:
        raise HTTPException(status_code=503, detail="Code reviewer not available")

    try:
        logger.info(f"Starting directory review: {request.directory_path}")

        # Execute directory review
        result = await unified_code_reviewer.review_directory(
            directory_path=request.directory_path,
            patterns=request.file_patterns,
            max_files=request.max_files,
            focus_areas=request.focus_areas,
        )

        return {"success": True, "directory_path": request.directory_path, "results": result}

    except Exception as e:
        logger.exception(f"Directory review failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# Workspace Analysis Endpoints
# ===========================


@app.post("/api/v1/workspace/analyze")
async def analyze_workspace(request: WorkspaceAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze project workspace."""
    if not unified_workspace_analyzer:
        raise HTTPException(status_code=503, detail="Workspace analyzer not available")

    try:
        logger.info(f"Starting workspace analysis: {request.project_path}")

        # Execute analysis
        result = await unified_workspace_analyzer.analyze_project(
            project_path=request.project_path,
            analysis_depth=request.analysis_depth,
            include_dependencies=request.include_dependencies,
            include_tests=request.include_tests,
            max_files=request.max_files,
        )

        return {"success": True, "project_path": request.project_path, "analysis_results": result}

    except Exception as e:
        logger.exception(f"Workspace analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/workspace/structure")
async def get_project_structure(request: ProjectStructureRequest):
    """Get project directory structure."""
    if not unified_workspace_analyzer:
        raise HTTPException(status_code=503, detail="Workspace analyzer not available")

    try:
        result = await unified_workspace_analyzer.get_project_structure(
            project_path=request.project_path,
            max_depth=request.max_depth,
            include_hidden=request.include_hidden,
        )

        return {"success": True, "project_path": request.project_path, "structure": result}

    except Exception as e:
        logger.exception(f"Project structure analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# Documentation Endpoints
# ===========================


@app.post("/api/v1/docs/generate")
async def generate_documentation(request: DocumentationRequest, background_tasks: BackgroundTasks):
    """Generate documentation from source code."""
    if not unified_documentation_generator:
        raise HTTPException(status_code=503, detail="Documentation generator not available")

    try:
        logger.info(f"Starting documentation generation: {request.source_path}")

        # Execute documentation generation
        result = await unified_documentation_generator.generate_documentation(
            source_path=request.source_path,
            doc_type=request.doc_type,
            output_format=request.output_format,
            include_examples=request.include_examples,
        )

        return {"success": True, "source_path": request.source_path, "documentation": result}

    except Exception as e:
        logger.exception(f"Documentation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/docs/readme")
async def generate_readme(request: ReadmeRequest, background_tasks: BackgroundTasks):
    """Generate README file for a project."""
    if not unified_documentation_generator:
        raise HTTPException(status_code=503, detail="Documentation generator not available")

    try:
        logger.info(f"Starting README generation: {request.project_path}")

        result = await unified_documentation_generator.generate_readme(
            project_path=request.project_path,
            template_style=request.template_style,
            include_badges=request.include_badges,
            include_installation=request.include_installation,
        )

        return {"success": True, "project_path": request.project_path, "readme_content": result}

    except Exception as e:
        logger.exception(f"README generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# Orchestration Endpoints
# ===========================


@app.post("/api/v1/orchestrate/task")
async def orchestrate_task(request: OrchestrationRequest, background_tasks: BackgroundTasks):
    """Execute a complex task using multiple agents."""
    if not unified_orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")

    try:
        logger.info(f"Starting task orchestration: {request.task[:100]}...")

        # Execute orchestration
        result = await unified_orchestrator.execute_task(
            task=request.task,
            specific_agents=request.specific_agents,
            session_id=request.session_id,
            streaming=request.streaming,
        )

        return {"success": True, "task": request.task, "orchestration_result": result}

    except Exception as e:
        logger.exception(f"Task orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/orchestrate/session", response_model=SessionResponse)
async def create_orchestration_session(session_id: str | None = None):
    """Create a new orchestration session."""
    if not unified_orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")

    try:
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        session = await unified_orchestrator.create_session(session_id)
        active_sessions[session_id] = session

        return SessionResponse(
            session_id=session_id,
            created_at=session.get("created_at", ""),
            last_activity=session.get("last_activity", ""),
            interaction_count=session.get("interaction_count", 0),
        )

    except Exception as e:
        logger.exception(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================
# Job Management Endpoints
# ===========================


@app.get("/api/v1/jobs", response_model=list[JobResponse])
async def list_jobs():
    """List all active jobs."""
    jobs_info = []
    for job_id, job in active_jobs.items():
        jobs_info.append(
            JobResponse(
                job_id=job_id,
                status=job.status,
                progress=job.progress_percentage,
                created_at=job.created_at.isoformat(),
                result=job.result if job.status == "completed" else None,
            ),
        )

    return jobs_info


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get status of a specific job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = active_jobs[job_id]
    return JobResponse(
        job_id=job_id,
        status=job.status,
        progress=job.progress_percentage,
        created_at=job.created_at.isoformat(),
        result=job.result if job.status == "completed" else None,
    )


@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # TODO: Implement job cancellation logic
    job = active_jobs[job_id]
    job.status = "cancelled"

    return {"message": f"Job {job_id} cancelled"}


# ===========================
# Session Management Endpoints
# ===========================


@app.get("/api/v1/sessions", response_model=list[SessionResponse])
async def list_sessions():
    """List all active sessions."""
    sessions_info = []
    for session_id, session in active_sessions.items():
        sessions_info.append(
            SessionResponse(
                session_id=session_id,
                created_at=session.get("created_at", ""),
                last_activity=session.get("last_activity", ""),
                interaction_count=session.get("interaction_count", 0),
            ),
        )

    return sessions_info


@app.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get information about a specific session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    session = active_sessions[session_id]
    return SessionResponse(
        session_id=session_id,
        created_at=session.get("created_at", ""),
        last_activity=session.get("last_activity", ""),
        interaction_count=session.get("interaction_count", 0),
    )


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Disconnect WebSocket if connected
    if session_id in websocket_connections:
        websocket = websocket_connections[session_id]
        await websocket.close()
        del websocket_connections[session_id]

    del active_sessions[session_id]

    return {"message": f"Session {session_id} deleted"}


# ===========================
# WebSocket Support
# ===========================


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    websocket_connections[session_id] = websocket

    try:
        # Send connection confirmation
        await websocket.send_json(
            {
                "type": "connection",
                "message": f"Connected to session {session_id}",
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
            },
        )

        # Handle messages
        while True:
            try:
                data = await websocket.receive_json()

                # Handle different message types
                if data.get("type") == "ping":
                    await websocket.send_json(
                        {"type": "pong", "timestamp": datetime.now().isoformat()}
                    )

                elif data.get("type") == "task" and unified_orchestrator:
                    task = data.get("task", "")
                    if task:
                        logger.info(f"WebSocket task execution: {task[:50]}...")

                        # Execute task with streaming
                        result = await unified_orchestrator.execute_task(
                            task=task,
                            session_id=session_id,
                            streaming=True,
                        )

                        # Send result
                        await websocket.send_json(
                            {
                                "type": "task_complete",
                                "result": result,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )

                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Unknown message type",
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.exception(f"WebSocket error: {e}")
                await websocket.send_json(
                    {"type": "error", "message": str(e), "timestamp": datetime.now().isoformat()}
                )

    except Exception as e:
        logger.exception(f"WebSocket connection error: {e}")
    finally:
        websocket_connections.pop(session_id, None)


# ===========================
# Streaming Endpoints
# ===========================


@app.post("/api/v1/orchestrate/stream")
async def stream_task(request: OrchestrationRequest):
    """Execute task with streaming response."""
    if not unified_orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")

    async def generate_stream():
        try:
            # This would be implemented with actual streaming from the orchestrator
            yield f"data: {json.dumps({'type': 'start', 'task': request.task})}\n\n"

            # Execute task (simplified - real implementation would stream)
            result = await unified_orchestrator.execute_task(
                task=request.task,
                specific_agents=request.specific_agents,
                session_id=request.session_id,
                streaming=True,
            )

            yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/plain")


# ===========================
# Error Handlers
# ===========================


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
    logger.error(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc), "path": str(request.url)},
    )


def main() -> None:
    """Main entry point for API server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8200"))
    reload = os.getenv("ENV", "production") == "development"

    logger.info(f"Starting Unified Agent API server on {host}:{port}")

    uvicorn.run(
        "app.core.interfaces.api_adapter:app", host=host, port=port, reload=reload, log_level="info"
    )


if __name__ == "__main__":
    main()
