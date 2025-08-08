#!/usr/bin/env python3
"""MCP Adapter for Unified Agents.

Single MCP server that exposes ALL unified agents to Claude CLI.
This consolidates multiple MCP servers into a single interface.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel
from pydantic import Field

# Import all unified agents
from gterminal.core.agents.unified_code_reviewer import UnifiedCodeReviewer
from gterminal.core.agents.unified_documentation_generator import UnifiedDocumentationGenerator
from gterminal.core.agents.unified_gemini_orchestrator import UnifiedGeminiOrchestrator
from gterminal.core.agents.unified_workspace_analyzer import UnifiedWorkspaceAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("Unified Fullstack Agent")

# Initialize unified agents
unified_code_reviewer = UnifiedCodeReviewer()
unified_workspace_analyzer = UnifiedWorkspaceAnalyzer()
unified_documentation_generator = UnifiedDocumentationGenerator()
unified_orchestrator = UnifiedGeminiOrchestrator()

# Global state for jobs and sessions
active_jobs: dict[str, Any] = {}
active_sessions: dict[str, Any] = {}


# ===========================
# Data Models
# ===========================


class CodeReviewRequest(BaseModel):
    """Request for code review."""

    file_path: str = Field(..., description="Path to file to review")
    focus_areas: str = Field(
        default="security,performance,quality", description="Areas to focus on (comma-separated)"
    )
    include_suggestions: bool = Field(default=True, description="Include fix suggestions")
    severity_threshold: str = Field(default="medium", description="Minimum severity level")


class WorkspaceAnalysisRequest(BaseModel):
    """Request for workspace analysis."""

    project_path: str = Field(..., description="Path to project directory")
    analysis_depth: str = Field(default="comprehensive", description="Analysis depth level")
    include_dependencies: bool = Field(default=True, description="Include dependency analysis")
    include_tests: bool = Field(default=True, description="Include test analysis")
    max_files: int = Field(default=1000, description="Maximum files to analyze")


class DocumentationRequest(BaseModel):
    """Request for documentation generation."""

    source_path: str = Field(..., description="Path to source code or project")
    doc_type: str = Field(default="api", description="Type of documentation")
    output_format: str = Field(default="markdown", description="Output format")
    include_examples: bool = Field(default=True, description="Include examples")


class OrchestrationRequest(BaseModel):
    """Request for multi-agent orchestration."""

    task: str = Field(..., description="Natural language task description")
    agents: str = Field(default="", description="Specific agents to use (comma-separated)")
    session_id: str | None = Field(None, description="Session ID for context")
    streaming: bool = Field(default=False, description="Enable streaming")


# ===========================
# Code Review Tools
# ===========================


@mcp.tool()
async def review_code(request: CodeReviewRequest) -> dict[str, Any]:
    """Review code file for security, performance, and quality issues.

    This tool performs comprehensive code analysis including:
    - Security vulnerability detection
    - Performance optimization opportunities
    - Code quality improvements
    - Best practice recommendations
    """
    try:
        logger.info(f"Starting code review for: {request.file_path}")

        # Validate file exists
        if not Path(request.file_path).exists():
            return {"error": f"File not found: {request.file_path}"}

        # Parse focus areas
        focus_list = [area.strip() for area in request.focus_areas.split(",") if area.strip()]

        # Execute review
        result = await unified_code_reviewer.review_file(
            file_path=request.file_path,
            focus_areas=focus_list,
            include_suggestions=request.include_suggestions,
            severity_threshold=request.severity_threshold,
        )

        return {"success": True, "file_path": request.file_path, "review_results": result}

    except Exception as e:
        logger.exception(f"Code review failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def review_directory(
    directory_path: str,
    file_patterns: str = "*.py,*.js,*.ts,*.rs,*.go",
    max_files: int = 100,
    focus_areas: str = "security,performance",
) -> dict[str, Any]:
    """Review multiple files in a directory.

    Performs batch code review on files matching patterns.
    """
    try:
        patterns = [p.strip() for p in file_patterns.split(",") if p.strip()]
        focus_list = [area.strip() for area in focus_areas.split(",") if area.strip()]

        result = await unified_code_reviewer.review_directory(
            directory_path=directory_path,
            patterns=patterns,
            max_files=max_files,
            focus_areas=focus_list,
        )

        return {"success": True, "directory_path": directory_path, "results": result}

    except Exception as e:
        logger.exception(f"Directory review failed: {e}")
        return {"error": str(e)}


# ===========================
# Workspace Analysis Tools
# ===========================


@mcp.tool()
async def analyze_workspace(request: WorkspaceAnalysisRequest) -> dict[str, Any]:
    """Analyze project workspace for architecture, dependencies, and code quality.

    This tool provides comprehensive project analysis including:
    - Architecture overview and patterns
    - Dependency analysis and vulnerability scanning
    - Code quality metrics
    - Test coverage analysis
    - Performance bottleneck identification
    """
    try:
        logger.info(f"Starting workspace analysis for: {request.project_path}")

        # Validate project path exists
        if not Path(request.project_path).exists():
            return {"error": f"Project path not found: {request.project_path}"}

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
        return {"error": str(e)}


@mcp.tool()
async def get_project_structure(
    project_path: str, max_depth: int = 3, include_hidden: bool = False
) -> dict[str, Any]:
    """Get project directory structure."""
    try:
        result = await unified_workspace_analyzer.get_project_structure(
            project_path=project_path,
            max_depth=max_depth,
            include_hidden=include_hidden,
        )

        return {"success": True, "project_path": project_path, "structure": result}

    except Exception as e:
        logger.exception(f"Project structure analysis failed: {e}")
        return {"error": str(e)}


# ===========================
# Documentation Tools
# ===========================


@mcp.tool()
async def generate_documentation(request: DocumentationRequest) -> dict[str, Any]:
    """Generate documentation from source code.

    This tool creates comprehensive documentation including:
    - API documentation with examples
    - README files with setup instructions
    - Code documentation with type hints
    - User guides with tutorials
    """
    try:
        logger.info(f"Starting documentation generation for: {request.source_path}")

        # Validate source path exists
        if not Path(request.source_path).exists():
            return {"error": f"Source path not found: {request.source_path}"}

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
        return {"error": str(e)}


@mcp.tool()
async def generate_readme(
    project_path: str,
    template_style: str = "comprehensive",
    include_badges: bool = True,
    include_installation: bool = True,
) -> dict[str, Any]:
    """Generate or update README file for a project."""
    try:
        result = await unified_documentation_generator.generate_readme(
            project_path=project_path,
            template_style=template_style,
            include_badges=include_badges,
            include_installation=include_installation,
        )

        return {"success": True, "project_path": project_path, "readme_content": result}

    except Exception as e:
        logger.exception(f"README generation failed: {e}")
        return {"error": str(e)}


# ===========================
# Orchestration Tools
# ===========================


@mcp.tool()
async def orchestrate_task(request: OrchestrationRequest) -> dict[str, Any]:
    """Orchestrate multi-agent task execution.

    This tool coordinates multiple agents to handle complex tasks:
    - Automatic agent selection based on task requirements
    - Parallel execution of independent subtasks
    - Context sharing between agents
    - Streaming progress updates
    """
    try:
        logger.info(f"Starting orchestration for task: {request.task[:100]}...")

        # Parse agents if specified
        agent_list = []
        if request.agents:
            agent_list = [agent.strip() for agent in request.agents.split(",") if agent.strip()]

        # Execute orchestration
        result = await unified_orchestrator.execute_task(
            task=request.task,
            specific_agents=agent_list,
            session_id=request.session_id,
            streaming=request.streaming,
        )

        return {"success": True, "task": request.task, "orchestration_result": result}

    except Exception as e:
        logger.exception(f"Task orchestration failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def create_session(session_id: str | None = None) -> dict[str, Any]:
    """Create a new agent session for maintaining context."""
    try:
        if not session_id:
            import uuid

            session_id = str(uuid.uuid4())

        session = await unified_orchestrator.create_session(session_id)
        active_sessions[session_id] = session

        return {
            "success": True,
            "session_id": session_id,
            "created_at": session.get("created_at", ""),
        }

    except Exception as e:
        logger.exception(f"Session creation failed: {e}")
        return {"error": str(e)}


# ===========================
# Status and Management Tools
# ===========================


@mcp.tool()
async def get_job_status(job_id: str) -> dict[str, Any]:
    """Get status of a running job."""
    try:
        if job_id not in active_jobs:
            return {"error": f"Job {job_id} not found"}

        job = active_jobs[job_id]
        return {
            "success": True,
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress_percentage,
            "started_at": job.created_at.isoformat(),
            "result": job.result if job.status == "completed" else None,
        }

    except Exception as e:
        logger.exception(f"Failed to get job status: {e}")
        return {"error": str(e)}


@mcp.tool()
async def list_active_jobs() -> dict[str, Any]:
    """List all active jobs."""
    try:
        jobs_info = []
        for job_id, job in active_jobs.items():
            jobs_info.append(
                {
                    "job_id": job_id,
                    "status": job.status,
                    "progress": job.progress_percentage,
                    "started_at": job.created_at.isoformat(),
                    "agent": getattr(job, "agent_name", "unknown"),
                },
            )

        return {"success": True, "active_jobs": jobs_info, "total_count": len(jobs_info)}

    except Exception as e:
        logger.exception(f"Failed to list jobs: {e}")
        return {"error": str(e)}


@mcp.tool()
async def list_sessions() -> dict[str, Any]:
    """List all active sessions."""
    try:
        sessions_info = []
        for session_id, session in active_sessions.items():
            sessions_info.append(
                {
                    "session_id": session_id,
                    "created_at": session.get("created_at", ""),
                    "last_activity": session.get("last_activity", ""),
                    "interaction_count": session.get("interaction_count", 0),
                },
            )

        return {
            "success": True,
            "active_sessions": sessions_info,
            "total_count": len(sessions_info),
        }

    except Exception as e:
        logger.exception(f"Failed to list sessions: {e}")
        return {"error": str(e)}


@mcp.tool()
async def get_server_info() -> dict[str, Any]:
    """Get information about the unified agent server."""
    return {
        "success": True,
        "server_name": "Unified Fullstack Agent MCP Server",
        "version": "1.0.0",
        "agents": {
            "code_reviewer": "Available",
            "workspace_analyzer": "Available",
            "documentation_generator": "Available",
            "orchestrator": "Available",
        },
        "capabilities": {
            "code_review": True,
            "workspace_analysis": True,
            "documentation_generation": True,
            "multi_agent_orchestration": True,
            "session_management": True,
            "job_tracking": True,
            "streaming_support": True,
        },
        "active_jobs": len(active_jobs),
        "active_sessions": len(active_sessions),
    }


# ===========================
# Utility Functions
# ===========================


async def initialize_agents() -> None:
    """Initialize all unified agents."""
    try:
        logger.info("Initializing unified agents...")

        # Initialize each agent
        await unified_code_reviewer.initialize()
        await unified_workspace_analyzer.initialize()
        await unified_documentation_generator.initialize()
        await unified_orchestrator.initialize()

        logger.info("All unified agents initialized successfully")

    except Exception as e:
        logger.exception(f"Failed to initialize agents: {e}")
        raise


async def shutdown_agents() -> None:
    """Shutdown all unified agents."""
    try:
        logger.info("Shutting down unified agents...")

        # Shutdown each agent
        await unified_code_reviewer.shutdown()
        await unified_workspace_analyzer.shutdown()
        await unified_documentation_generator.shutdown()
        await unified_orchestrator.shutdown()

        logger.info("All unified agents shut down successfully")

    except Exception as e:
        logger.exception(f"Error during agent shutdown: {e}")


def main() -> None:
    """Main entry point for MCP server."""
    import signal
    import sys

    async def cleanup() -> None:
        """Cleanup function for graceful shutdown."""
        logger.info("Received shutdown signal, cleaning up...")
        await shutdown_agents()
        sys.exit(0)

    # Setup signal handlers
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda s, f: asyncio.create_task(cleanup()))

    # Initialize and run
    async def startup() -> None:
        await initialize_agents()
        logger.info("MCP Adapter server started successfully")
        logger.info("Available tools:")
        for tool_name in mcp._tools:
            logger.info(f"  - {tool_name}")

    # Run startup in event loop
    asyncio.run(startup())

    # Run MCP server
    mcp.run()


if __name__ == "__main__":
    main()
