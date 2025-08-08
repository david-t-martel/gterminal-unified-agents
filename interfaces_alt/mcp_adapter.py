"""MCP Adapter - Interface adapter for Claude CLI integration.

This adapter exposes unified agents as MCP servers for seamless integration
with Claude CLI while maintaining the consolidated architecture.

PROVIDES MCP COMPATIBILITY FOR:
- unified_code_reviewer
- unified_workspace_analyzer
- unified_documentation_generator
- unified_gemini_orchestrator

ELIMINATES NEED FOR:
- Multiple separate MCP server implementations
- Duplicate MCP tool definitions
- Scattered MCP configuration files
"""

import asyncio
from datetime import UTC
from datetime import datetime
import logging
from typing import Any

from fastmcp import FastMCP

from gterminal.core.agents.unified_code_reviewer import UnifiedCodeReviewer
from gterminal.core.agents.unified_documentation_generator import UnifiedDocumentationGenerator
from gterminal.core.agents.unified_gemini_orchestrator import UnifiedGeminiOrchestrator
from gterminal.core.agents.unified_workspace_analyzer import UnifiedWorkspaceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Unified Agents MCP Server")

# Initialize unified agents
code_reviewer = UnifiedCodeReviewer()
workspace_analyzer = UnifiedWorkspaceAnalyzer()
documentation_generator = UnifiedDocumentationGenerator()
orchestrator = UnifiedGeminiOrchestrator()


# Code Review MCP Tools
@mcp.tool()
async def review_code(
    file_path: str,
    focus_areas: str = "security,performance,quality",
    include_suggestions: str = "true",
    severity_threshold: str = "medium",
) -> dict[str, Any]:
    """Review code for security, performance, and quality issues.

    Args:
        file_path: Path to the file to review
        focus_areas: Comma-separated focus areas
        include_suggestions: Whether to include fix suggestions
        severity_threshold: Minimum severity to report

    Returns:
        Comprehensive code review results

    """
    try:
        job_id = await code_reviewer.create_job(
            "review_file",
            {
                "file_path": file_path,
                "focus_areas": focus_areas.split(","),
                "include_suggestions": include_suggestions.lower() == "true",
                "severity_threshold": severity_threshold,
            },
        )

        result = await code_reviewer.execute_job_async(job_id)
        return {
            "status": "success",
            "result": result,
            "job_id": job_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Code review failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@mcp.tool()
async def review_security(
    directory: str,
    file_patterns: str = "*.py,*.js,*.ts,*.java,*.rs,*.go",
    scan_depth: str = "comprehensive",
) -> dict[str, Any]:
    """Security-focused review of multiple files.

    Args:
        directory: Directory to scan
        file_patterns: Comma-separated file patterns
        scan_depth: Scan depth (quick, standard, comprehensive)

    Returns:
        Security analysis results with CWE mappings

    """
    try:
        job_id = await code_reviewer.create_job(
            "security_scan",
            {
                "target_path": directory,
                "file_patterns": file_patterns.split(","),
                "scan_depth": scan_depth,
            },
        )

        result = await code_reviewer.execute_job_async(job_id)
        return {
            "status": "success",
            "result": result,
            "job_id": job_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Security review failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@mcp.tool()
async def comprehensive_analysis(
    target_path: str,
    analysis_types: str = "code_quality,security,performance",
    file_patterns: str = "*.py,*.js,*.ts,*.rs,*.go",
    max_files: str = "50",
) -> dict[str, Any]:
    """Comprehensive code analysis combining quality, security, and performance.

    Args:
        target_path: File or directory path to analyze
        analysis_types: Comma-separated analysis types
        file_patterns: File patterns to include
        max_files: Maximum number of files to analyze

    Returns:
        Comprehensive analysis results with consolidated findings

    """
    try:
        job_id = await code_reviewer.create_job(
            "comprehensive_analysis",
            {
                "target_path": target_path,
                "analysis_types": analysis_types.split(","),
                "file_patterns": file_patterns.split(","),
                "max_files": int(max_files),
            },
        )

        result = await code_reviewer.execute_job_async(job_id)
        return {
            "status": "success",
            "result": result,
            "job_id": job_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Comprehensive analysis failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


# Workspace Analysis MCP Tools
@mcp.tool()
async def analyze_workspace(
    project_path: str,
    analysis_depth: str = "comprehensive",
    include_dependencies: str = "true",
    include_security: str = "true",
) -> dict[str, Any]:
    """Analyze workspace structure and provide insights.

    Args:
        project_path: Path to project directory
        analysis_depth: Analysis depth (quick, standard, comprehensive)
        include_dependencies: Whether to analyze dependencies
        include_security: Whether to include security analysis

    Returns:
        Comprehensive workspace analysis results

    """
    try:
        job_type = (
            "comprehensive_report" if analysis_depth == "comprehensive" else "analyze_project"
        )

        job_id = await workspace_analyzer.create_job(
            job_type,
            {
                "project_path": project_path,
                "analysis_depth": analysis_depth,
                "include_dependencies": include_dependencies.lower() == "true",
                "include_security": include_security.lower() == "true",
            },
        )

        result = await workspace_analyzer.execute_job_async(job_id)
        return {
            "status": "success",
            "result": result,
            "job_id": job_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Workspace analysis failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@mcp.tool()
async def scan_dependencies(
    project_path: str,
    check_vulnerabilities: str = "true",
    check_licenses: str = "false",
) -> dict[str, Any]:
    """Scan project dependencies for security and licensing issues.

    Args:
        project_path: Path to project directory
        check_vulnerabilities: Whether to check for vulnerabilities
        check_licenses: Whether to check license compatibility

    Returns:
        Dependency analysis results

    """
    try:
        job_id = await workspace_analyzer.create_job(
            "scan_dependencies",
            {
                "project_path": project_path,
                "check_vulnerabilities": check_vulnerabilities.lower() == "true",
                "check_licenses": check_licenses.lower() == "true",
            },
        )

        result = await workspace_analyzer.execute_job_async(job_id)
        return {
            "status": "success",
            "result": result,
            "job_id": job_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Dependency scan failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


# Documentation Generation MCP Tools
@mcp.tool()
async def generate_documentation(
    source_path: str,
    doc_type: str = "readme",
    target_audience: str = "developers",
    include_examples: str = "true",
) -> dict[str, Any]:
    """Generate comprehensive documentation.

    Args:
        source_path: Path to source code or project
        doc_type: Type of documentation (readme, api, architecture, user_guide)
        target_audience: Target audience (developers, end_users, architects)
        include_examples: Whether to include usage examples

    Returns:
        Generated documentation content and metadata

    """
    try:
        # Map doc_type to job_type
        job_type_map = {
            "readme": "generate_readme",
            "api": "generate_api_docs",
            "architecture": "generate_architecture_docs",
            "user_guide": "generate_user_guide",
            "changelog": "generate_changelog",
        }

        job_type = job_type_map.get(doc_type, "generate_readme")

        job_id = await documentation_generator.create_job(
            job_type,
            {
                "project_path": source_path,
                "doc_type": doc_type,
                "target_audience": target_audience,
                "include_examples": include_examples.lower() == "true",
            },
        )

        result = await documentation_generator.execute_job_async(job_id)
        return {
            "status": "success",
            "result": result,
            "job_id": job_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Documentation generation failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@mcp.tool()
async def generate_api_docs(
    source_path: str,
    framework: str = "auto",
    include_schemas: str = "true",
    include_examples: str = "true",
) -> dict[str, Any]:
    """Generate API documentation from source code.

    Args:
        source_path: Path to API source code
        framework: API framework (auto, fastapi, flask, django, express)
        include_schemas: Whether to include request/response schemas
        include_examples: Whether to include usage examples

    Returns:
        API documentation content

    """
    try:
        job_id = await documentation_generator.create_job(
            "generate_api_docs",
            {
                "source_path": source_path,
                "framework": framework,
                "include_schemas": include_schemas.lower() == "true",
                "include_examples": include_examples.lower() == "true",
            },
        )

        result = await documentation_generator.execute_job_async(job_id)
        return {
            "status": "success",
            "result": result,
            "job_id": job_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"API documentation generation failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


# Orchestration MCP Tools
@mcp.tool()
async def execute_workflow(
    workflow_definition: str,
    workflow_name: str = "Custom Workflow",
    stop_on_failure: str = "true",
) -> dict[str, Any]:
    """Execute a custom workflow with multiple tasks.

    Args:
        workflow_definition: JSON string defining workflow tasks
        workflow_name: Name for the workflow
        stop_on_failure: Whether to stop on first failure

    Returns:
        Workflow execution results

    """
    try:
        import json

        workflow_spec = json.loads(workflow_definition)

        # Create workflow
        create_job_id = await orchestrator.create_job(
            "create_workflow",
            {
                "name": workflow_name,
                "description": f"Custom workflow: {workflow_name}",
                "tasks": workflow_spec.get("tasks", []),
                "metadata": {"stop_on_failure": stop_on_failure.lower() == "true"},
            },
        )

        create_result = await orchestrator.execute_job_async(create_job_id)
        workflow_id = create_result["workflow_id"]

        # Execute workflow
        exec_job_id = await orchestrator.create_job(
            "execute_workflow", {"workflow_id": workflow_id}
        )

        result = await orchestrator.execute_job_async(exec_job_id)
        return {
            "status": "success",
            "result": result,
            "workflow_id": workflow_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Workflow execution failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@mcp.tool()
async def get_agent_status() -> dict[str, Any]:
    """Get comprehensive status of all unified agents.

    Returns:
        Status information for all agents and orchestrator

    """
    try:
        job_id = await orchestrator.create_job("get_agent_status", {})
        result = await orchestrator.execute_job_async(job_id)

        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Failed to get agent status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@mcp.tool()
async def analyze_performance() -> dict[str, Any]:
    """Analyze performance of all agents and orchestration.

    Returns:
        Performance analysis with insights and recommendations

    """
    try:
        job_id = await orchestrator.create_job("analyze_performance", {})
        result = await orchestrator.execute_job_async(job_id)

        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Performance analysis failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


# Server lifecycle management
async def startup() -> None:
    """Initialize the MCP server and start orchestrator."""
    logger.info("Starting Unified Agents MCP Server")
    await orchestrator.start_task_processor()
    logger.info("Unified Agents MCP Server started")


async def shutdown() -> None:
    """Cleanup resources on server shutdown."""
    logger.info("Shutting down Unified Agents MCP Server")

    # Cleanup all agents
    await code_reviewer.cleanup()
    await workspace_analyzer.cleanup()
    await documentation_generator.cleanup()
    await orchestrator.cleanup()

    logger.info("Unified Agents MCP Server shutdown complete")


# Main entry point for MCP server
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        await startup()
        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await shutdown()

    asyncio.run(main())
