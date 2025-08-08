"""CLI Adapter - Command-line interface adapter for unified agents.

This adapter provides a unified CLI interface for all agents while maintaining
the consolidated architecture. Eliminates the need for multiple CLI implementations.

PROVIDES CLI ACCESS FOR:
- unified_code_reviewer
- unified_workspace_analyzer
- unified_documentation_generator
- unified_gemini_orchestrator

ELIMINATES NEED FOR:
- Multiple separate CLI implementations
- Duplicate command definitions
- Scattered CLI configuration
"""

import asyncio
import json
import logging
from pathlib import Path
import sys
from typing import Any

import click

from gterminal.core.agents.unified_code_reviewer import UnifiedCodeReviewer
from gterminal.core.agents.unified_documentation_generator import UnifiedDocumentationGenerator
from gterminal.core.agents.unified_gemini_orchestrator import UnifiedGeminiOrchestrator
from gterminal.core.agents.unified_workspace_analyzer import UnifiedWorkspaceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Global agent instances
code_reviewer = None
workspace_analyzer = None
documentation_generator = None
orchestrator = None


def init_agents() -> None:
    """Initialize all unified agents."""
    global code_reviewer, workspace_analyzer, documentation_generator, orchestrator

    if code_reviewer is None:
        code_reviewer = UnifiedCodeReviewer()
        workspace_analyzer = UnifiedWorkspaceAnalyzer()
        documentation_generator = UnifiedDocumentationGenerator()
        orchestrator = UnifiedGeminiOrchestrator()


def print_result(result: dict[str, Any], format: str = "json") -> None:
    """Print result in specified format."""
    if format == "json":
        click.echo(json.dumps(result, indent=2, default=str))
    elif format == "summary":
        if "error" in result:
            click.echo(f"❌ Error: {result['error']}")
        else:
            click.echo(f"✅ Success: {result.get('status', 'completed')}")
            if "result" in result:
                # Print key metrics
                res = result["result"]
                if isinstance(res, dict):
                    for key, value in res.items():
                        if key in [
                            "score",
                            "quality_score",
                            "total_files",
                            "total_issues",
                        ]:
                            click.echo(f"   {key}: {value}")


# Main CLI group
@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """Unified Agents CLI - Comprehensive AI-powered code analysis and automation."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    init_agents()


# Code Review Commands
@cli.group()
def review() -> None:
    """Code review and analysis commands."""


@review.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--focus",
    "-f",
    default="security,performance,quality",
    help="Focus areas (comma-separated)",
)
@click.option("--suggestions", "-s", is_flag=True, help="Include fix suggestions")
@click.option("--threshold", "-t", default="medium", help="Severity threshold (low/medium/high)")
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def file(file_path: str, focus: str, suggestions: bool, threshold: str, format: str) -> None:
    """Review a specific file for issues."""

    async def run_review() -> None:
        try:
            job_id = await code_reviewer.create_job(
                "review_file",
                {
                    "file_path": file_path,
                    "focus_areas": focus.split(","),
                    "include_suggestions": suggestions,
                    "severity_threshold": threshold,
                },
            )

            result = await code_reviewer.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(run_review())


@review.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--patterns", "-p", default="*.py,*.js,*.ts,*.java,*.rs,*.go", help="File patterns")
@click.option(
    "--depth",
    "-d",
    default="comprehensive",
    help="Scan depth (quick/standard/comprehensive)",
)
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def security(directory: str, patterns: str, depth: str, format: str) -> None:
    """Perform security-focused review."""

    async def run_security() -> None:
        try:
            job_id = await code_reviewer.create_job(
                "security_scan",
                {
                    "target_path": directory,
                    "file_patterns": patterns.split(","),
                    "scan_depth": depth,
                },
            )

            result = await code_reviewer.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(run_security())


@review.command()
@click.argument("target_path", type=click.Path(exists=True))
@click.option("--types", "-t", default="code_quality,security,performance", help="Analysis types")
@click.option("--patterns", "-p", default="*.py,*.js,*.ts,*.rs,*.go", help="File patterns")
@click.option("--max-files", "-m", default=50, help="Maximum files to analyze")
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def comprehensive(target_path: str, types: str, patterns: str, max_files: int, format: str) -> None:
    """Comprehensive analysis combining all review types."""

    async def run_comprehensive() -> None:
        try:
            job_id = await code_reviewer.create_job(
                "comprehensive_analysis",
                {
                    "target_path": target_path,
                    "analysis_types": types.split(","),
                    "file_patterns": patterns.split(","),
                    "max_files": max_files,
                },
            )

            result = await code_reviewer.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(run_comprehensive())


# Workspace Analysis Commands
@cli.group()
def workspace() -> None:
    """Workspace analysis and insights commands."""


@workspace.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--depth",
    "-d",
    default="comprehensive",
    help="Analysis depth (quick/standard/comprehensive)",
)
@click.option("--dependencies", is_flag=True, help="Include dependency analysis")
@click.option("--security", is_flag=True, help="Include security analysis")
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def analyze(project_path: str, depth: str, dependencies: bool, security: bool, format: str) -> None:
    """Analyze workspace structure and provide insights."""

    async def run_analysis() -> None:
        try:
            job_type = "comprehensive_report" if depth == "comprehensive" else "analyze_project"

            job_id = await workspace_analyzer.create_job(
                job_type,
                {
                    "project_path": project_path,
                    "analysis_depth": depth,
                    "include_dependencies": dependencies,
                    "include_security": security,
                },
            )

            result = await workspace_analyzer.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(run_analysis())


@workspace.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.option("--vulnerabilities", is_flag=True, help="Check for vulnerabilities")
@click.option("--licenses", is_flag=True, help="Check license compatibility")
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def dependencies(project_path: str, vulnerabilities: bool, licenses: bool, format: str) -> None:
    """Scan project dependencies."""

    async def run_deps() -> None:
        try:
            job_id = await workspace_analyzer.create_job(
                "scan_dependencies",
                {
                    "project_path": project_path,
                    "check_vulnerabilities": vulnerabilities,
                    "check_licenses": licenses,
                },
            )

            result = await workspace_analyzer.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(run_deps())


@workspace.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def technologies(project_path: str, format: str) -> None:
    """Detect technologies used in the project."""

    async def run_tech() -> None:
        try:
            job_id = await workspace_analyzer.create_job(
                "detect_technologies", {"project_path": project_path}
            )

            result = await workspace_analyzer.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(run_tech())


# Documentation Commands
@cli.group()
def docs() -> None:
    """Documentation generation commands."""


@docs.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--audience",
    "-a",
    default="developers",
    help="Target audience (developers/end_users/architects)",
)
@click.option("--examples", is_flag=True, help="Include usage examples")
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def readme(project_path: str, audience: str, examples: bool, format: str) -> None:
    """Generate README.md file."""

    async def run_readme() -> None:
        try:
            job_id = await documentation_generator.create_job(
                "generate_readme",
                {
                    "project_path": project_path,
                    "target_audience": audience,
                    "include_examples": examples,
                },
            )

            result = await documentation_generator.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(run_readme())


@docs.command()
@click.argument("source_path", type=click.Path(exists=True))
@click.option(
    "--framework",
    "-f",
    default="auto",
    help="API framework (auto/fastapi/flask/django/express)",
)
@click.option("--schemas", is_flag=True, help="Include request/response schemas")
@click.option("--examples", is_flag=True, help="Include usage examples")
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def api(source_path: str, framework: str, schemas: bool, examples: bool, format: str) -> None:
    """Generate API documentation."""

    async def run_api() -> None:
        try:
            job_id = await documentation_generator.create_job(
                "generate_api_docs",
                {
                    "source_path": source_path,
                    "framework": framework,
                    "include_schemas": schemas,
                    "include_examples": examples,
                },
            )

            result = await documentation_generator.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(run_api())


@docs.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def architecture(project_path: str, format: str) -> None:
    """Generate architecture documentation."""

    async def run_arch() -> None:
        try:
            job_id = await documentation_generator.create_job(
                "generate_architecture_docs",
                {"project_path": project_path},
            )

            result = await documentation_generator.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(run_arch())


@docs.command()
@click.argument("project_path", type=click.Path(exists=True, file_okay=False))
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def comprehensive(project_path: str, format: str) -> None:
    """Generate comprehensive documentation suite."""

    async def run_comprehensive_docs() -> None:
        try:
            job_id = await documentation_generator.create_job(
                "comprehensive_docs", {"project_path": project_path}
            )

            result = await documentation_generator.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(run_comprehensive_docs())


# Orchestration Commands
@cli.group()
def orchestrate() -> None:
    """Orchestration and workflow commands."""


@orchestrate.command()
@click.argument("workflow_file", type=click.Path(exists=True))
@click.option("--name", "-n", help="Workflow name")
@click.option("--stop-on-failure", is_flag=True, help="Stop on first failure")
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def workflow(workflow_file: str, name: str, stop_on_failure: bool, format: str) -> None:
    """Execute a workflow from JSON definition file."""

    async def run_workflow() -> None:
        try:
            with open(workflow_file) as f:
                workflow_def = f.read()

            if not name:
                name = Path(workflow_file).stem

            # Create workflow
            create_job_id = await orchestrator.create_job(
                "create_workflow",
                {
                    "name": name,
                    "description": f"Workflow from {workflow_file}",
                    "tasks": json.loads(workflow_def).get("tasks", []),
                    "metadata": {"stop_on_failure": stop_on_failure},
                },
            )

            create_result = await orchestrator.execute_job_async(create_job_id)
            workflow_id = create_result["workflow_id"]

            # Execute workflow
            exec_job_id = await orchestrator.create_job(
                "execute_workflow", {"workflow_id": workflow_id}
            )

            result = await orchestrator.execute_job_async(exec_job_id)
            print_result(
                {"status": "success", "result": result, "workflow_id": workflow_id},
                format,
            )

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(run_workflow())


@orchestrate.command()
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def status(format: str) -> None:
    """Get status of all agents."""

    async def get_status() -> None:
        try:
            job_id = await orchestrator.create_job("get_agent_status", {})
            result = await orchestrator.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(get_status())


@orchestrate.command()
@click.option("--format", "-o", default="summary", help="Output format (json/summary)")
def performance(format: str) -> None:
    """Analyze performance of all agents."""

    async def analyze_perf() -> None:
        try:
            job_id = await orchestrator.create_job("analyze_performance", {})
            result = await orchestrator.execute_job_async(job_id)
            print_result({"status": "success", "result": result}, format)

        except Exception as e:
            print_result({"error": str(e)}, format)
            sys.exit(1)

    asyncio.run(analyze_perf())


# Utility Commands
@cli.command()
@click.option("--format", "-o", default="json", help="Output format (json/summary)")
def version(format: str) -> None:
    """Show version and agent information."""
    info = {
        "version": "1.0.0",
        "unified_agents": {
            "code_reviewer": "Comprehensive code review with security and performance analysis",
            "workspace_analyzer": "Project structure analysis and technology detection",
            "documentation_generator": "AI-powered documentation generation",
            "gemini_orchestrator": "Multi-agent workflow coordination",
        },
        "features": [
            "PyO3 Rust integration for 5-10x performance",
            "Gemini 2.0 Flash AI analysis",
            "Advanced caching and parallel processing",
            "MCP server compliance",
            "Real-time progress tracking",
        ],
    }

    print_result(info, format)


if __name__ == "__main__":
    cli()
