"""API Adapter - REST API interface adapter for unified agents.

This adapter provides RESTful HTTP endpoints for all unified agents while maintaining
the consolidated architecture. Eliminates the need for multiple API implementations.

PROVIDES REST API FOR:
- unified_code_reviewer
- unified_workspace_analyzer
- unified_documentation_generator
- unified_gemini_orchestrator

ELIMINATES NEED FOR:
- Multiple separate API implementations
- Duplicate endpoint definitions
- Scattered API configuration
"""

from datetime import UTC
from datetime import datetime
import logging
from typing import Any

from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import Field
import uvicorn

from gterminal.core.agents.unified_code_reviewer import UnifiedCodeReviewer
from gterminal.core.agents.unified_documentation_generator import UnifiedDocumentationGenerator
from gterminal.core.agents.unified_gemini_orchestrator import UnifiedGeminiOrchestrator
from gterminal.core.agents.unified_workspace_analyzer import UnifiedWorkspaceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Unified Agents API",
    description="Comprehensive AI-powered code analysis and automation API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize unified agents
code_reviewer = UnifiedCodeReviewer()
workspace_analyzer = UnifiedWorkspaceAnalyzer()
documentation_generator = UnifiedDocumentationGenerator()
orchestrator = UnifiedGeminiOrchestrator()

# Background task tracking
background_jobs = {}


# Request/Response Models
class CodeReviewRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to review")
    focus_areas: list[str] = Field(
        default=["security", "performance", "quality"],
        description="Focus areas for review",
    )
    include_suggestions: bool = Field(
        default=True, description="Whether to include fix suggestions"
    )
    severity_threshold: str = Field(default="medium", description="Minimum severity threshold")


class SecurityScanRequest(BaseModel):
    directory: str = Field(..., description="Directory to scan")
    file_patterns: list[str] = Field(
        default=["*.py", "*.js", "*.ts", "*.java", "*.rs", "*.go"],
        description="File patterns to scan",
    )
    scan_depth: str = Field(default="comprehensive", description="Scan depth level")


class ComprehensiveAnalysisRequest(BaseModel):
    target_path: str = Field(..., description="File or directory path to analyze")
    analysis_types: list[str] = Field(
        default=["code_quality", "security", "performance"],
        description="Types of analysis to perform",
    )
    file_patterns: list[str] = Field(
        default=["*.py", "*.js", "*.ts", "*.rs", "*.go"],
        description="File patterns to include",
    )
    max_files: int = Field(default=50, description="Maximum number of files to analyze")


class WorkspaceAnalysisRequest(BaseModel):
    project_path: str = Field(..., description="Path to project directory")
    analysis_depth: str = Field(default="comprehensive", description="Analysis depth level")
    include_dependencies: bool = Field(default=True, description="Whether to analyze dependencies")
    include_security: bool = Field(default=True, description="Whether to include security analysis")


class DependencyScanRequest(BaseModel):
    project_path: str = Field(..., description="Path to project directory")
    check_vulnerabilities: bool = Field(
        default=True, description="Whether to check for vulnerabilities"
    )
    check_licenses: bool = Field(
        default=False, description="Whether to check license compatibility"
    )


class DocumentationRequest(BaseModel):
    source_path: str = Field(..., description="Path to source code or project")
    doc_type: str = Field(default="readme", description="Type of documentation to generate")
    target_audience: str = Field(default="developers", description="Target audience")
    include_examples: bool = Field(default=True, description="Whether to include usage examples")


class APIDocsRequest(BaseModel):
    source_path: str = Field(..., description="Path to API source code")
    framework: str = Field(default="auto", description="API framework")
    include_schemas: bool = Field(
        default=True, description="Whether to include request/response schemas"
    )
    include_examples: bool = Field(default=True, description="Whether to include usage examples")


class WorkflowRequest(BaseModel):
    workflow_definition: dict[str, Any] = Field(..., description="Workflow definition with tasks")
    workflow_name: str = Field(default="Custom Workflow", description="Name for the workflow")
    stop_on_failure: bool = Field(default=True, description="Whether to stop on first failure")


class JobResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(default="", description="Status message")
    timestamp: str = Field(..., description="Response timestamp")


class ResultResponse(BaseModel):
    status: str = Field(..., description="Response status")
    result: Any = Field(None, description="Result data")
    job_id: str = Field(None, description="Associated job ID")
    timestamp: str = Field(..., description="Response timestamp")
    error: str = Field(None, description="Error message if failed")


# Utility functions
def create_response(
    result: Any = None, error: str | None = None, job_id: str | None = None
) -> ResultResponse:
    """Create standardized API response."""
    return ResultResponse(
        status="success" if error is None else "error",
        result=result,
        job_id=job_id,
        timestamp=datetime.now(UTC).isoformat(),
        error=error,
    )


async def execute_job_with_tracking(
    agent,
    job_type: str,
    parameters: dict[str, Any],
    background_tasks: BackgroundTasks = None,
) -> tuple[str, Any]:
    """Execute job with optional background tracking."""
    job_id = await agent.create_job(job_type, parameters)

    if background_tasks:
        # Execute in background
        background_tasks.add_task(agent.execute_job_async, job_id)
        return job_id, None
    # Execute synchronously
    result = await agent.execute_job_async(job_id)
    return job_id, result


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check agent status
        status_job_id = await orchestrator.create_job("get_agent_status", {})
        agent_status = await orchestrator.execute_job_async(status_job_id)

        return {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "agents": agent_status["agents"],
            "version": "1.0.0",
        }
    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e!s}")


# Code Review Endpoints
@app.post("/api/v1/code-review/file", response_model=ResultResponse)
async def review_code_file(request: CodeReviewRequest, background_tasks: BackgroundTasks = None):
    """Review a specific file for code quality, security, and performance issues."""
    try:
        job_id, result = await execute_job_with_tracking(
            code_reviewer,
            "review_file",
            {
                "file_path": request.file_path,
                "focus_areas": request.focus_areas,
                "include_suggestions": request.include_suggestions,
                "severity_threshold": request.severity_threshold,
            },
            background_tasks,
        )

        return create_response(result=result, job_id=job_id)

    except Exception as e:
        logger.exception(f"Code review failed: {e}")
        return create_response(error=str(e))


@app.post("/api/v1/code-review/security", response_model=ResultResponse)
async def security_scan(request: SecurityScanRequest, background_tasks: BackgroundTasks = None):
    """Perform security-focused scan of directory."""
    try:
        job_id, result = await execute_job_with_tracking(
            code_reviewer,
            "security_scan",
            {
                "target_path": request.directory,
                "file_patterns": request.file_patterns,
                "scan_depth": request.scan_depth,
            },
            background_tasks,
        )

        return create_response(result=result, job_id=job_id)

    except Exception as e:
        logger.exception(f"Security scan failed: {e}")
        return create_response(error=str(e))


@app.post("/api/v1/code-review/comprehensive", response_model=ResultResponse)
async def comprehensive_analysis(
    request: ComprehensiveAnalysisRequest, background_tasks: BackgroundTasks = None
):
    """Perform comprehensive code analysis combining all review types."""
    try:
        job_id, result = await execute_job_with_tracking(
            code_reviewer,
            "comprehensive_analysis",
            {
                "target_path": request.target_path,
                "analysis_types": request.analysis_types,
                "file_patterns": request.file_patterns,
                "max_files": request.max_files,
            },
            background_tasks,
        )

        return create_response(result=result, job_id=job_id)

    except Exception as e:
        logger.exception(f"Comprehensive analysis failed: {e}")
        return create_response(error=str(e))


# Workspace Analysis Endpoints
@app.post("/api/v1/workspace/analyze", response_model=ResultResponse)
async def analyze_workspace(
    request: WorkspaceAnalysisRequest, background_tasks: BackgroundTasks = None
):
    """Analyze workspace structure and provide insights."""
    try:
        job_type = (
            "comprehensive_report"
            if request.analysis_depth == "comprehensive"
            else "analyze_project"
        )

        job_id, result = await execute_job_with_tracking(
            workspace_analyzer,
            job_type,
            {
                "project_path": request.project_path,
                "analysis_depth": request.analysis_depth,
                "include_dependencies": request.include_dependencies,
                "include_security": request.include_security,
            },
            background_tasks,
        )

        return create_response(result=result, job_id=job_id)

    except Exception as e:
        logger.exception(f"Workspace analysis failed: {e}")
        return create_response(error=str(e))


@app.post("/api/v1/workspace/dependencies", response_model=ResultResponse)
async def scan_dependencies(
    request: DependencyScanRequest, background_tasks: BackgroundTasks = None
):
    """Scan project dependencies for security and licensing issues."""
    try:
        job_id, result = await execute_job_with_tracking(
            workspace_analyzer,
            "scan_dependencies",
            {
                "project_path": request.project_path,
                "check_vulnerabilities": request.check_vulnerabilities,
                "check_licenses": request.check_licenses,
            },
            background_tasks,
        )

        return create_response(result=result, job_id=job_id)

    except Exception as e:
        logger.exception(f"Dependency scan failed: {e}")
        return create_response(error=str(e))


@app.get("/api/v1/workspace/technologies/{project_path:path}", response_model=ResultResponse)
async def detect_technologies(project_path: str, background_tasks: BackgroundTasks = None):
    """Detect technologies used in the project."""
    try:
        job_id, result = await execute_job_with_tracking(
            workspace_analyzer,
            "detect_technologies",
            {"project_path": project_path},
            background_tasks,
        )

        return create_response(result=result, job_id=job_id)

    except Exception as e:
        logger.exception(f"Technology detection failed: {e}")
        return create_response(error=str(e))


# Documentation Generation Endpoints
@app.post("/api/v1/docs/generate", response_model=ResultResponse)
async def generate_documentation(
    request: DocumentationRequest, background_tasks: BackgroundTasks = None
):
    """Generate comprehensive documentation."""
    try:
        # Map doc_type to job_type
        job_type_map = {
            "readme": "generate_readme",
            "api": "generate_api_docs",
            "architecture": "generate_architecture_docs",
            "user_guide": "generate_user_guide",
            "changelog": "generate_changelog",
        }

        job_type = job_type_map.get(request.doc_type, "generate_readme")

        job_id, result = await execute_job_with_tracking(
            documentation_generator,
            job_type,
            {
                "project_path": request.source_path,
                "doc_type": request.doc_type,
                "target_audience": request.target_audience,
                "include_examples": request.include_examples,
            },
            background_tasks,
        )

        return create_response(result=result, job_id=job_id)

    except Exception as e:
        logger.exception(f"Documentation generation failed: {e}")
        return create_response(error=str(e))


@app.post("/api/v1/docs/api", response_model=ResultResponse)
async def generate_api_documentation(
    request: APIDocsRequest, background_tasks: BackgroundTasks = None
):
    """Generate API documentation from source code."""
    try:
        job_id, result = await execute_job_with_tracking(
            documentation_generator,
            "generate_api_docs",
            {
                "source_path": request.source_path,
                "framework": request.framework,
                "include_schemas": request.include_schemas,
                "include_examples": request.include_examples,
            },
            background_tasks,
        )

        return create_response(result=result, job_id=job_id)

    except Exception as e:
        logger.exception(f"API documentation generation failed: {e}")
        return create_response(error=str(e))


@app.post("/api/v1/docs/comprehensive/{project_path:path}", response_model=ResultResponse)
async def generate_comprehensive_docs(project_path: str, background_tasks: BackgroundTasks = None):
    """Generate comprehensive documentation suite."""
    try:
        job_id, result = await execute_job_with_tracking(
            documentation_generator,
            "comprehensive_docs",
            {"project_path": project_path},
            background_tasks,
        )

        return create_response(result=result, job_id=job_id)

    except Exception as e:
        logger.exception(f"Comprehensive documentation generation failed: {e}")
        return create_response(error=str(e))


# Orchestration Endpoints
@app.post("/api/v1/orchestrate/workflow", response_model=ResultResponse)
async def execute_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks = None):
    """Execute a custom workflow with multiple tasks."""
    try:
        # Create workflow
        create_job_id = await orchestrator.create_job(
            "create_workflow",
            {
                "name": request.workflow_name,
                "description": f"Custom workflow: {request.workflow_name}",
                "tasks": request.workflow_definition.get("tasks", []),
                "metadata": {"stop_on_failure": request.stop_on_failure},
            },
        )

        create_result = await orchestrator.execute_job_async(create_job_id)
        workflow_id = create_result["workflow_id"]

        # Execute workflow
        exec_job_id, result = await execute_job_with_tracking(
            orchestrator,
            "execute_workflow",
            {"workflow_id": workflow_id},
            background_tasks,
        )

        return create_response(result=result, job_id=exec_job_id)

    except Exception as e:
        logger.exception(f"Workflow execution failed: {e}")
        return create_response(error=str(e))


@app.get("/api/v1/orchestrate/status", response_model=ResultResponse)
async def get_agent_status():
    """Get comprehensive status of all unified agents."""
    try:
        job_id = await orchestrator.create_job("get_agent_status", {})
        result = await orchestrator.execute_job_async(job_id)

        return create_response(result=result, job_id=job_id)

    except Exception as e:
        logger.exception(f"Failed to get agent status: {e}")
        return create_response(error=str(e))


@app.get("/api/v1/orchestrate/performance", response_model=ResultResponse)
async def analyze_performance():
    """Analyze performance of all agents and orchestration."""
    try:
        job_id = await orchestrator.create_job("analyze_performance", {})
        result = await orchestrator.execute_job_async(job_id)

        return create_response(result=result, job_id=job_id)

    except Exception as e:
        logger.exception(f"Performance analysis failed: {e}")
        return create_response(error=str(e))


# Job Management Endpoints
@app.get("/api/v1/jobs/{agent_type}")
async def list_jobs(agent_type: str, status: str | None = None):
    """List jobs for a specific agent type."""
    try:
        agent_map = {
            "code_reviewer": code_reviewer,
            "workspace_analyzer": workspace_analyzer,
            "documentation_generator": documentation_generator,
            "orchestrator": orchestrator,
        }

        agent = agent_map.get(agent_type)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

        from gterminal.core.agents.base_unified_agent import JobStatus

        status_filter = JobStatus(status) if status else None

        jobs = await agent.list_jobs(status_filter)

        return create_response(result={"jobs": jobs, "agent_type": agent_type})

    except Exception as e:
        logger.exception(f"Failed to list jobs: {e}")
        return create_response(error=str(e))


@app.get("/api/v1/jobs/{agent_type}/{job_id}")
async def get_job_status(agent_type: str, job_id: str):
    """Get status of a specific job."""
    try:
        agent_map = {
            "code_reviewer": code_reviewer,
            "workspace_analyzer": workspace_analyzer,
            "documentation_generator": documentation_generator,
            "orchestrator": orchestrator,
        }

        agent = agent_map.get(agent_type)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

        job_status = await agent.get_job_status(job_id)

        return create_response(result=job_status)

    except Exception as e:
        logger.exception(f"Failed to get job status: {e}")
        return create_response(error=str(e))


# Server lifecycle
@app.on_event("startup")
async def startup() -> None:
    """Initialize the API server."""
    logger.info("Starting Unified Agents API Server")
    await orchestrator.start_task_processor()
    logger.info("Unified Agents API Server started")


@app.on_event("shutdown")
async def shutdown() -> None:
    """Cleanup resources on server shutdown."""
    logger.info("Shutting down Unified Agents API Server")

    # Cleanup all agents
    await code_reviewer.cleanup()
    await workspace_analyzer.cleanup()
    await documentation_generator.cleanup()
    await orchestrator.cleanup()

    logger.info("Unified Agents API Server shutdown complete")


# Main entry point
if __name__ == "__main__":
    uvicorn.run("api_adapter:app", host="0.0.0.0", port=8100, reload=False, access_log=True)
