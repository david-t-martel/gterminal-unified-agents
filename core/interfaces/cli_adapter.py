#!/usr/bin/env python3
"""CLI Adapter for Unified Agents - Phase 2 Enhanced.

Unified command-line interface for all agent operations.
Consolidates all agent functionality into a single CLI tool.

Phase 2 Features:
- Interactive mode with conversation history using prompt_toolkit
- Batch processing with YAML/JSON script support
- Job control (pause/resume/cancel) integration
- Rich Terminal UI enhancements with live dashboards
- Session export/import functionality
- Rust tool integration with proper fallback handling
- Advanced configuration management
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import sys
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.syntax import Syntax
from rich.table import Table

# Import all unified agents
from gterminal.core.agents.unified_code_reviewer import UnifiedCodeReviewer
from gterminal.core.agents.unified_documentation_generator import UnifiedDocumentationGenerator
from gterminal.core.agents.unified_gemini_orchestrator import UnifiedGeminiOrchestrator
from gterminal.core.agents.unified_workspace_analyzer import UnifiedWorkspaceAnalyzer
from gterminal.core.interfaces.batch_processor import BatchProcessor

# Import Phase 2 features
from gterminal.core.interfaces.enhanced_interactive_mode import start_interactive_mode
from gterminal.core.interfaces.job_control import JobManager
from gterminal.core.interfaces.job_control import JobType
from gterminal.core.interfaces.job_control import controllable_job
from gterminal.core.interfaces.job_control import get_job_manager
from gterminal.core.react_engine import ReactEngine
from gterminal.core.react_engine import ReactEngineConfig

# Import core components for Phase 2
from gterminal.core.session import SessionManager

# Import Rust extensions with fallback handling
from gterminal.utils.rust_extensions.wrapper import RUST_EXTENSIONS_AVAILABLE
from gterminal.utils.rust_extensions.wrapper import get_rust_status
from gterminal.utils.rust_extensions.wrapper import rust_system

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global agent instances
unified_code_reviewer = None
unified_workspace_analyzer = None
unified_documentation_generator = None
unified_orchestrator = None
react_engine = None

# Phase 2 components
session_manager = None
current_session = None
job_manager: JobManager | None = None
batch_processor: BatchProcessor | None = None
config_profiles: dict[str, dict[str, Any]] = {}


class JobState(str, Enum):
    """Job execution states."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobControl:
    """Job control and state management for Phase 2."""

    job_id: str
    state: JobState = JobState.PENDING
    can_pause: bool = True
    can_resume: bool = False
    can_cancel: bool = True
    can_update: bool = True
    current_step: int = 0
    total_steps: int = 0
    pause_requested: bool = False
    cancel_requested: bool = False
    update_requested: bool = False
    update_instruction: str = ""
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


async def initialize_agents(profile: str = "default", enhanced: bool = True) -> None:
    """Initialize all unified agents with Phase 2 enhancements."""
    global unified_code_reviewer, unified_workspace_analyzer
    global unified_documentation_generator, unified_orchestrator
    global session_manager, current_session, react_engine
    global job_manager, batch_processor

    # Display initialization banner with Rust status
    rust_status = "🦀 Rust Extensions" if RUST_EXTENSIONS_AVAILABLE else "🐍 Python Fallback"

    console.print(
        Panel(
            f"[bold cyan]🚀 Initializing Unified Agents Framework[/bold cyan]\n"
            f"Profile: [bold yellow]{profile}[/bold yellow]\n"
            f"Enhanced Mode: [bold green]{'✅ Enabled' if enhanced else '❌ Basic'}[/bold green]\n"
            f"Acceleration: [bold magenta]{rust_status}[/bold magenta]",
            title="Phase 2 CLI Adapter",
            border_style="cyan",
        ),
    )

    try:
        # Load configuration profile
        config = load_profile(profile)

        # Initialize core components
        session_manager = SessionManager()

        # Initialize React Engine with enhanced configuration
        if enhanced:
            react_config = ReactEngineConfig(
                enable_redis=config.get("enable_redis", True),
                enable_rag=config.get("enable_rag", True),
                enable_autonomous=config.get("enable_autonomous", False),
                enable_streaming=config.get("enable_streaming", True),
                enable_rust_optimizations=RUST_EXTENSIONS_AVAILABLE,
                cache_responses=config.get("cache_responses", True),
                parallel_tool_execution=config.get("parallel_execution", True),
            )
            react_engine = ReactEngine(config=react_config, profile=profile)

        # Initialize Phase 2 components
        if enhanced:
            job_manager = await get_job_manager()
            batch_processor = BatchProcessor(profile=profile)

            # Warm up Rust extensions if available
            if RUST_EXTENSIONS_AVAILABLE:
                console.print("[cyan]🦀 Warming up Rust extensions...[/cyan]")
                rust_status_info = get_rust_status()
                console.print(
                    f"[dim]Cache initialized with {rust_status_info['cache_stats']['capacity']} capacity[/dim]",
                )

        # Initialize unified agents
        with console.status("[cyan]Initializing agents...", spinner="dots"):
            unified_code_reviewer = UnifiedCodeReviewer()
            unified_workspace_analyzer = UnifiedWorkspaceAnalyzer()
            unified_documentation_generator = UnifiedDocumentationGenerator()
            unified_orchestrator = UnifiedGeminiOrchestrator()

            # Initialize each agent
            await unified_code_reviewer.initialize()
            await unified_workspace_analyzer.initialize()
            await unified_documentation_generator.initialize()
            await unified_orchestrator.initialize()

        # Show initialization summary
        init_table = Table(title="Initialization Summary")
        init_table.add_column("Component", style="cyan")
        init_table.add_column("Status", style="green")
        init_table.add_column("Details", style="dim")

        init_table.add_row("Code Reviewer", "✅ Ready", "Security & Performance Analysis")
        init_table.add_row("Workspace Analyzer", "✅ Ready", "Project Structure Analysis")
        init_table.add_row("Documentation Generator", "✅ Ready", "API & Code Documentation")
        init_table.add_row("Gemini Orchestrator", "✅ Ready", "Multi-Agent Coordination")

        if enhanced:
            init_table.add_row("React Engine", "✅ Ready", "Intelligent Task Processing")
            init_table.add_row("Job Manager", "✅ Ready", "Background Task Control")
            init_table.add_row("Batch Processor", "✅ Ready", "Script & Automation Support")

        if RUST_EXTENSIONS_AVAILABLE:
            init_table.add_row("Rust Acceleration", "✅ Active", "High-Performance Extensions")
        else:
            init_table.add_row("Python Fallback", "✅ Active", "Standard Performance Mode")

        console.print(init_table)
        console.print("[bold green]🎉 All components initialized successfully![/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Failed to initialize agents: {e}[/red]")
        logger.exception(f"Initialization error: {e}")
        sys.exit(1)


def load_profile(profile_name: str) -> dict[str, Any]:
    """Load configuration profile."""
    config_path = Path.home() / ".fullstack-agent" / "profiles" / f"{profile_name}.json"

    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Default profile
    default_config = {
        "timeout": 300,
        "max_concurrent_jobs": 5,
        "default_focus_areas": ["security", "performance", "quality"],
        "output_format": "rich",
        "auto_save_sessions": True,
        "ui_theme": "default",
    }

    # Create config directory and save default
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(default_config, f, indent=2)

    return default_config


async def shutdown_agents() -> None:
    """Shutdown all unified agents and Phase 2 components."""
    global job_manager, batch_processor, react_engine

    console.print("[yellow]🔄 Shutting down components...[/yellow]")

    try:
        # Shutdown Phase 2 components first
        if job_manager:
            await job_manager.shutdown()
            job_manager = None
            console.print("[dim]✅ Job manager shut down[/dim]")

        if react_engine:
            # React engine doesn't have explicit shutdown, but clear references
            react_engine = None
            console.print("[dim]✅ React engine shut down[/dim]")

        if batch_processor:
            batch_processor = None
            console.print("[dim]✅ Batch processor shut down[/dim]")

        # Shutdown unified agents
        if unified_code_reviewer:
            await unified_code_reviewer.shutdown()
            console.print("[dim]✅ Code reviewer shut down[/dim]")
        if unified_workspace_analyzer:
            await unified_workspace_analyzer.shutdown()
            console.print("[dim]✅ Workspace analyzer shut down[/dim]")
        if unified_documentation_generator:
            await unified_documentation_generator.shutdown()
            console.print("[dim]✅ Documentation generator shut down[/dim]")
        if unified_orchestrator:
            await unified_orchestrator.shutdown()
            console.print("[dim]✅ Gemini orchestrator shut down[/dim]")

        # Save any pending sessions
        if session_manager:
            await session_manager.save_all()
            console.print("[dim]✅ Sessions saved[/dim]")

        console.print("[green]🎉 All components shut down successfully[/green]")

    except Exception as e:
        console.print(f"[red]❌ Shutdown error: {e}[/red]")
        logger.exception(f"Shutdown error: {e}")


# ===========================
# Main CLI Group
# ===========================


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--json", "json_output", is_flag=True, help="Output results as JSON")
@click.option("--profile", default="default", help="Configuration profile to use")
@click.option("--session", help="Session ID to continue")
@click.option("--enhanced", is_flag=True, default=True, help="Enable Phase 2 enhanced features")
@click.pass_context
def cli(ctx, debug: bool, json_output: bool, profile: str, session: str, enhanced: bool) -> None:
    """Unified Fullstack Agent CLI - Phase 2 Enhanced.

    Comprehensive AI agent framework for code review, workspace analysis,
    documentation generation, and multi-agent orchestration.

    New Phase 2 Features:
    - Interactive mode with conversation history
    - Configuration profiles for different environments
    - Batch processing and scripting support
    - Pause/resume/cancel job controls
    - Session export/import functionality

    Examples:
      agent review /path/to/file.py
      agent workspace analyze /path/to/project
      agent docs generate /path/to/source
      agent orchestrate "Optimize this codebase for performance"
      agent interactive --profile dev
      agent batch commands.yaml

    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["json"] = json_output
    ctx.obj["profile"] = profile
    ctx.obj["session"] = session
    ctx.obj["enhanced"] = enhanced

    if debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)


# ===========================
# Code Review Commands
# ===========================


@cli.group()
def review() -> None:
    """Code review operations."""


@review.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--focus", default="security,performance,quality", help="Focus areas (comma-separated)"
)
@click.option(
    "--severity",
    default="medium",
    type=click.Choice(["low", "medium", "high", "critical"]),
    help="Minimum severity threshold",
)
@click.option("--no-suggestions", is_flag=True, help="Skip fix suggestions")
@click.option("--job-control", is_flag=True, help="Enable job control for this operation")
@click.pass_context
def file(
    ctx, file_path: str, focus: str, severity: str, no_suggestions: bool, job_control: bool
) -> None:
    """Review a single file for issues."""

    async def _review() -> None:
        profile = ctx.obj.get("profile", "default")
        enhanced = ctx.obj.get("enhanced", True)
        await initialize_agents(profile=profile, enhanced=enhanced)

        try:
            # Parse focus areas
            focus_areas = [area.strip() for area in focus.split(",") if area.strip()]

            if job_control and job_manager:
                # Execute with job control
                async with controllable_job(
                    job_manager=job_manager,
                    job_name=f"Review {Path(file_path).name}",
                    job_type=JobType.REVIEW,
                    total_steps=3,
                ) as job_ctx:
                    await job_ctx.update_progress(0, "Starting review...")

                    await job_ctx.update_progress(1, "Analyzing code...")
                    result = await unified_code_reviewer.review_file(
                        file_path=file_path,
                        focus_areas=focus_areas,
                        include_suggestions=not no_suggestions,
                        severity_threshold=severity,
                    )

                    await job_ctx.update_progress(2, "Processing results...")
                    await asyncio.sleep(0.5)  # Brief pause for UI

                    await job_ctx.update_progress(3, "Review completed")
                    console.print(f"[green]✅ Job {job_ctx.job_id[:8]}... completed[/green]")
            else:
                # Execute without job control (original behavior)
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Reviewing {Path(file_path).name}...", total=None)

                    # Execute review
                    result = await unified_code_reviewer.review_file(
                        file_path=file_path,
                        focus_areas=focus_areas,
                        include_suggestions=not no_suggestions,
                        severity_threshold=severity,
                    )

                    progress.update(task, completed=True)

            # Display results
            if ctx.obj["json"]:
                console.print_json(data=result)
            else:
                _display_review_results(file_path, result)

        except Exception as e:
            console.print(f"[red]❌ Review failed: {e}[/red]")
        finally:
            await shutdown_agents()

    asyncio.run(_review())


@review.command()
@click.argument("directory_path", type=click.Path(exists=True))
@click.option(
    "--patterns",
    default="*.py,*.js,*.ts,*.rs,*.go",
    help="File patterns to review (comma-separated)",
)
@click.option("--max-files", default=100, type=int, help="Maximum files to review")
@click.option("--focus", default="security,performance", help="Focus areas")
@click.pass_context
def directory(ctx, directory_path: str, patterns: str, max_files: int, focus: str) -> None:
    """Review all files in a directory."""

    async def _review_dir() -> None:
        await initialize_agents()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Reviewing directory...", total=None)

                # Parse patterns and focus areas
                pattern_list = [p.strip() for p in patterns.split(",") if p.strip()]
                focus_areas = [area.strip() for area in focus.split(",") if area.strip()]

                # Execute directory review
                result = await unified_code_reviewer.review_directory(
                    directory_path=directory_path,
                    patterns=pattern_list,
                    max_files=max_files,
                    focus_areas=focus_areas,
                )

                progress.update(task, completed=True)

            # Display results
            if ctx.obj["json"]:
                console.print_json(data=result)
            else:
                _display_directory_review_results(directory_path, result)

        except Exception as e:
            console.print(f"[red]❌ Directory review failed: {e}[/red]")
        finally:
            await shutdown_agents()

    asyncio.run(_review_dir())


# ===========================
# Workspace Analysis Commands
# ===========================


@cli.group()
def workspace() -> None:
    """Workspace analysis operations."""


@workspace.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option(
    "--depth",
    default="comprehensive",
    type=click.Choice(["quick", "standard", "comprehensive"]),
    help="Analysis depth level",
)
@click.option("--no-deps", is_flag=True, help="Skip dependency analysis")
@click.option("--no-tests", is_flag=True, help="Skip test analysis")
@click.option("--max-files", default=1000, type=int, help="Maximum files to analyze")
@click.pass_context
def analyze(
    ctx, project_path: str, depth: str, no_deps: bool, no_tests: bool, max_files: int
) -> None:
    """Analyze project workspace."""

    async def _analyze() -> None:
        profile = ctx.obj.get("profile", "default")
        enhanced = ctx.obj.get("enhanced", True)
        await initialize_agents(profile=profile, enhanced=enhanced)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Analyzing workspace...", total=None)

                # Execute analysis
                result = await unified_workspace_analyzer.analyze_project(
                    project_path=project_path,
                    analysis_depth=depth,
                    include_dependencies=not no_deps,
                    include_tests=not no_tests,
                    max_files=max_files,
                )

                progress.update(task, completed=True)

            # Display results
            if ctx.obj["json"]:
                console.print_json(data=result)
            else:
                _display_workspace_results(project_path, result)

        except Exception as e:
            console.print(f"[red]❌ Workspace analysis failed: {e}[/red]")
        finally:
            await shutdown_agents()

    asyncio.run(_analyze())


@workspace.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option("--max-depth", default=3, type=int, help="Maximum directory depth")
@click.option("--include-hidden", is_flag=True, help="Include hidden files/directories")
@click.pass_context
def structure(ctx, project_path: str, max_depth: int, include_hidden: bool) -> None:
    """Get project directory structure."""

    async def _structure() -> None:
        await initialize_agents()

        try:
            result = await unified_workspace_analyzer.get_project_structure(
                project_path=project_path,
                max_depth=max_depth,
                include_hidden=include_hidden,
            )

            if ctx.obj["json"]:
                console.print_json(data=result)
            else:
                _display_project_structure(project_path, result)

        except Exception as e:
            console.print(f"[red]❌ Structure analysis failed: {e}[/red]")
        finally:
            await shutdown_agents()

    asyncio.run(_structure())


# ===========================
# Documentation Commands
# ===========================


@cli.group()
def docs() -> None:
    """Documentation generation operations."""


@docs.command()
@click.argument("source_path", type=click.Path(exists=True))
@click.option(
    "--type",
    "doc_type",
    default="api",
    type=click.Choice(["api", "code", "readme", "user-guide"]),
    help="Documentation type",
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "html", "rst"]),
    help="Output format",
)
@click.option("--no-examples", is_flag=True, help="Skip examples")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def generate(
    ctx, source_path: str, doc_type: str, output_format: str, no_examples: bool, output: str
) -> None:
    """Generate documentation from source code."""

    async def _generate() -> None:
        profile = ctx.obj.get("profile", "default")
        enhanced = ctx.obj.get("enhanced", True)
        await initialize_agents(profile=profile, enhanced=enhanced)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating documentation...", total=None)

                # Execute documentation generation
                result = await unified_documentation_generator.generate_documentation(
                    source_path=source_path,
                    doc_type=doc_type,
                    output_format=output_format,
                    include_examples=not no_examples,
                )

                progress.update(task, completed=True)

            # Save to file if specified
            if output:
                with open(output, "w") as f:
                    if isinstance(result, dict) and "content" in result:
                        f.write(result["content"])
                    else:
                        f.write(str(result))
                console.print(f"[green]✅ Documentation saved to {output}[/green]")

            # Display results
            if ctx.obj["json"]:
                console.print_json(data=result)
            else:
                _display_documentation_results(source_path, result)

        except Exception as e:
            console.print(f"[red]❌ Documentation generation failed: {e}[/red]")
        finally:
            await shutdown_agents()

    asyncio.run(_generate())


@docs.command()
@click.argument("project_path", type=click.Path(exists=True))
@click.option(
    "--style",
    default="comprehensive",
    type=click.Choice(["minimal", "standard", "comprehensive"]),
    help="README style",
)
@click.option("--no-badges", is_flag=True, help="Skip status badges")
@click.option("--no-install", is_flag=True, help="Skip installation section")
@click.pass_context
def readme(ctx, project_path: str, style: str, no_badges: bool, no_install: bool) -> None:
    """Generate README file for project."""

    async def _readme() -> None:
        await initialize_agents()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating README...", total=None)

                result = await unified_documentation_generator.generate_readme(
                    project_path=project_path,
                    template_style=style,
                    include_badges=not no_badges,
                    include_installation=not no_install,
                )

                progress.update(task, completed=True)

            if ctx.obj["json"]:
                console.print_json(data=result)
            else:
                console.print("[green]✅ README generated successfully[/green]")
                console.print(result)

        except Exception as e:
            console.print(f"[red]❌ README generation failed: {e}[/red]")
        finally:
            await shutdown_agents()

    asyncio.run(_readme())


# ===========================
# Orchestration Commands
# ===========================


@cli.group()
def orchestrate() -> None:
    """Multi-agent orchestration operations."""


@orchestrate.command()
@click.argument("task")
@click.option("--agents", help="Specific agents to use (comma-separated)")
@click.option("--session-id", help="Session ID for context")
@click.option("--streaming", is_flag=True, help="Enable streaming output")
@click.pass_context
def task(ctx, task: str, agents: str, session_id: str, streaming: bool) -> None:
    """Execute a complex task using multiple agents."""

    async def _orchestrate() -> None:
        profile = ctx.obj.get("profile", "default")
        enhanced = ctx.obj.get("enhanced", True)
        await initialize_agents(profile=profile, enhanced=enhanced)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress_task = progress.add_task("Orchestrating task...", total=None)

                # Parse agents if specified
                agent_list = []
                if agents:
                    agent_list = [a.strip() for a in agents.split(",") if a.strip()]

                # Execute orchestration
                result = await unified_orchestrator.execute_task(
                    task=task,
                    specific_agents=agent_list,
                    session_id=session_id,
                    streaming=streaming,
                )

                progress.update(progress_task, completed=True)

            if ctx.obj["json"]:
                console.print_json(data=result)
            else:
                _display_orchestration_results(task, result)

        except Exception as e:
            console.print(f"[red]❌ Task orchestration failed: {e}[/red]")
        finally:
            await shutdown_agents()

    asyncio.run(_orchestrate())


@orchestrate.command()
@click.option("--session-id", help="Specific session ID")
@click.pass_context
def session_create(ctx, session_id: str) -> None:
    """Create a new orchestration session."""

    async def _create_session() -> None:
        await initialize_agents()

        try:
            session = await unified_orchestrator.create_session(session_id)

            console.print(f"[green]✅ Session created: {session['session_id']}[/green]")
            console.print(f"Created at: {session.get('created_at', '')}")

        except Exception as e:
            console.print(f"[red]❌ Session creation failed: {e}[/red]")
        finally:
            await shutdown_agents()

    asyncio.run(_create_session())


# ===========================
# Info and Status Commands
# ===========================


@cli.command()
@click.pass_context
def status(ctx) -> None:
    """Show agent system status."""

    async def _status() -> None:
        profile = ctx.obj.get("profile", "default")
        enhanced = ctx.obj.get("enhanced", True)
        await initialize_agents(profile=profile, enhanced=enhanced)

        try:
            # Get status from all agents
            status_info = {
                "code_reviewer": "Available",
                "workspace_analyzer": "Available",
                "documentation_generator": "Available",
                "orchestrator": "Available",
            }

            if ctx.obj["json"]:
                console.print_json(data=status_info)
            else:
                table = Table(title="Unified Agent Status")
                table.add_column("Agent", style="cyan")
                table.add_column("Status", style="green")

                for agent, status in status_info.items():
                    table.add_row(agent, status)

                console.print(table)

        except Exception as e:
            console.print(f"[red]❌ Status check failed: {e}[/red]")
        finally:
            await shutdown_agents()

    asyncio.run(_status())


# ===========================
# Phase 2 CLI Commands
# ===========================


@cli.command()
@click.option("--profile", default="default", help="Configuration profile to use")
@click.option("--session-id", help="Existing session ID to resume")
@click.option("--enhanced", is_flag=True, default=True, help="Enable Phase 2 features")
@click.pass_context
def interactive(ctx, profile: str, session_id: str, enhanced: bool) -> None:
    """Start interactive mode with conversation history and advanced features."""

    async def _interactive() -> None:
        # Initialize with enhanced features
        await initialize_agents(profile=profile, enhanced=enhanced)

        try:
            # Start interactive session
            config = None
            if enhanced and react_engine:
                config = react_engine.config

            await start_interactive_mode(profile=profile, session_id=session_id, config=config)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interactive session interrupted[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Interactive session failed: {e}[/red]")
        finally:
            await shutdown_agents()

    asyncio.run(_interactive())


@cli.command()
@click.argument("script_path", type=click.Path(exists=True))
@click.option("--profile", default="default", help="Configuration profile")
@click.option("--live-output", is_flag=True, default=True, help="Show live execution dashboard")
@click.option("--results-file", help="Save results to specific file")
@click.pass_context
def batch(ctx, script_path: str, profile: str, live_output: bool, results_file: str) -> None:
    """Execute batch script with job control and monitoring."""

    async def _batch() -> None:
        await initialize_agents(profile=profile, enhanced=True)

        try:
            if not batch_processor:
                console.print("[red]❌ Batch processor not available[/red]")
                return

            # Load and execute script
            script = await batch_processor.load_script(script_path)

            # Override results file if specified
            if results_file:
                script.results_file = results_file

            console.print(f"[cyan]📋 Executing batch script: {script.name}[/cyan]")
            console.print(f"Tasks: {len(script.tasks)}, Parallel limit: {script.parallel_limit}")

            result = await batch_processor.execute_script(script=script, live_output=live_output)

            # Display final results
            if result.success:
                console.print(
                    Panel(
                        f"[green]✅ Batch execution completed successfully![/green]\n"
                        f"Tasks: {result.tasks_completed}/{result.tasks_total} completed\n"
                        f"Duration: {result.total_duration_seconds:.1f}s\n"
                        f"Parallel efficiency: {result.parallel_efficiency:.1%}",
                        title="Batch Results",
                        border_style="green",
                    ),
                )
            else:
                console.print(
                    Panel(
                        f"[red]❌ Batch execution failed[/red]\n"
                        f"Tasks: {result.tasks_completed}/{result.tasks_total} completed\n"
                        f"Failed: {result.tasks_failed}\n"
                        f"Duration: {result.total_duration_seconds:.1f}s",
                        title="Batch Results",
                        border_style="red",
                    ),
                )

            if ctx.obj["json"]:
                console.print_json(data=result.model_dump())

        except Exception as e:
            console.print(f"[red]❌ Batch execution failed: {e}[/red]")
        finally:
            await shutdown_agents()

    asyncio.run(_batch())


@cli.group()
def jobs() -> None:
    """Job control and monitoring commands."""


@jobs.command()
@click.option("--state", help="Filter by job state")
@click.option("--user-id", help="Filter by user ID")
@click.pass_context
def list(ctx, state: str, user_id: str) -> None:
    """List all jobs with optional filtering."""

    async def _list() -> None:
        job_mgr = await get_job_manager()

        from gterminal.core.interfaces.job_control import JobState

        state_filter = JobState(state) if state else None

        jobs_list = await job_mgr.list_jobs(state_filter=state_filter, user_id_filter=user_id)

        if ctx.obj["json"]:
            console.print_json(data=jobs_list)
            return

        if not jobs_list:
            console.print("[dim]No jobs found[/dim]")
            return

        # Create jobs table
        jobs_table = Table(title="Active Jobs")
        jobs_table.add_column("ID", style="cyan")
        jobs_table.add_column("Name", style="white")
        jobs_table.add_column("State", style="yellow")
        jobs_table.add_column("Progress", style="green")
        jobs_table.add_column("Runtime", style="dim")

        for job_info in jobs_list:
            job_id_short = job_info["job_id"][:8] + "..."
            progress = f"{job_info['progress_percentage']:.1f}%"
            runtime = f"{job_info.get('runtime_seconds', 0):.1f}s"

            jobs_table.add_row(job_id_short, job_info["name"], job_info["state"], progress, runtime)

        console.print(jobs_table)

    asyncio.run(_list())


@jobs.command()
@click.argument("job_id")
@click.pass_context
def status(ctx, job_id: str) -> None:
    """Get detailed status of a specific job."""

    async def _status() -> None:
        job_mgr = await get_job_manager()
        job_status = await job_mgr.get_job_status(job_id)

        if not job_status:
            console.print(f"[red]❌ Job {job_id} not found[/red]")
            return

        if ctx.obj["json"]:
            console.print_json(data=job_status)
            return

        # Display detailed status
        status_panel = Panel(
            f"State: [bold]{job_status['state']}[/bold]\n"
            f"Progress: {job_status['progress_percentage']:.1f}%\n"
            f"Current Step: {job_status.get('current_step', 'N/A')}\n"
            f"Steps: {job_status['completed_steps']}/{job_status['total_steps']}\n"
            f"Runtime: {job_status.get('runtime_seconds', 0):.1f}s\n"
            f"Errors: {job_status['error_count']}",
            title=f"Job {job_id[:8]}... - {job_status['name']}",
            border_style="blue",
        )

        console.print(status_panel)

    asyncio.run(_status())


@jobs.command()
@click.argument("job_id")
def pause(job_id: str) -> None:
    """Pause a running job."""

    async def _pause() -> None:
        job_mgr = await get_job_manager()
        success = await job_mgr.pause_job(job_id)

        if success:
            console.print(f"[green]✅ Job {job_id[:8]}... paused[/green]")
        else:
            console.print(f"[red]❌ Failed to pause job {job_id[:8]}...[/red]")

    asyncio.run(_pause())


@jobs.command()
@click.argument("job_id")
def resume(job_id: str) -> None:
    """Resume a paused job."""

    async def _resume() -> None:
        job_mgr = await get_job_manager()
        success = await job_mgr.resume_job(job_id)

        if success:
            console.print(f"[green]✅ Job {job_id[:8]}... resumed[/green]")
        else:
            console.print(f"[red]❌ Failed to resume job {job_id[:8]}...[/red]")

    asyncio.run(_resume())


@jobs.command()
@click.argument("job_id")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def cancel(job_id: str, confirm: bool) -> None:
    """Cancel a running job."""

    async def _cancel() -> None:
        if not confirm and not click.confirm(f"Cancel job {job_id[:8]}...?"):
            console.print("[yellow]Operation cancelled[/yellow]")
            return

        job_mgr = await get_job_manager()
        success = await job_mgr.cancel_job(job_id)

        if success:
            console.print(f"[green]✅ Job {job_id[:8]}... cancelled[/green]")
        else:
            console.print(f"[red]❌ Failed to cancel job {job_id[:8]}...[/red]")

    asyncio.run(_cancel())


@jobs.command()
@click.argument("job_id")
@click.option("--follow", "-f", is_flag=True, help="Follow job progress in real-time")
@click.option("--interval", default=2, type=int, help="Update interval in seconds")
def monitor(job_id: str, follow: bool, interval: int) -> None:
    """Monitor job execution in real-time."""

    async def _monitor() -> None:
        job_mgr = await get_job_manager()

        if follow:
            # Real-time monitoring with live updates
            try:
                from rich.live import Live
                from rich.panel import Panel
                from rich.progress import BarColumn
                from rich.progress import Progress
                from rich.progress import TextColumn
                from rich.progress import TimeElapsedColumn

                with Live(console=console, refresh_per_second=1 / interval) as live:
                    while True:
                        job_status = await job_mgr.get_job_status(job_id)
                        if not job_status:
                            live.update(
                                Panel(f"[red]❌ Job {job_id} not found[/red]", title="Job Monitor")
                            )
                            break

                        # Create progress display
                        progress = Progress(
                            TextColumn("[bold blue]{task.fields[step]}"),
                            BarColumn(bar_width=None),
                            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                            TimeElapsedColumn(),
                            expand=True,
                        )

                        progress.add_task(
                            "processing",
                            total=job_status["total_steps"],
                            completed=job_status["completed_steps"],
                            step=job_status.get("current_step", "N/A"),
                        )

                        status_panel = Panel(
                            progress,
                            title=f"Job {job_id[:8]}... - {job_status['name']}",
                            subtitle=f"State: {job_status['state']} | Runtime: {job_status.get('runtime_seconds', 0):.1f}s",
                            border_style="blue" if job_status["state"] == "running" else "yellow",
                        )

                        live.update(status_panel)

                        # Exit conditions
                        if job_status["state"] in ["completed", "failed", "cancelled"]:
                            break

                        await asyncio.sleep(interval)

            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped[/yellow]")
        else:
            # Single status check
            job_status = await job_mgr.get_job_status(job_id)
            if job_status:
                console.print_json(data=job_status)
            else:
                console.print(f"[red]❌ Job {job_id} not found[/red]")

    asyncio.run(_monitor())


@cli.group()
def session() -> None:
    """Session management commands."""


@session.command()
@click.argument("session_id")
@click.argument("output_file")
@click.pass_context
def export(ctx, session_id: str, output_file: str) -> None:
    """Export session to file."""

    async def _export() -> None:
        await initialize_agents(enhanced=False)  # Basic initialization

        try:
            session_obj = await session_manager.get_or_create_async(session_id)

            if not session_obj.interactions:
                console.print(f"[yellow]⚠️  Session {session_id} has no interactions[/yellow]")

            export_data = {
                "session_id": session_obj.id,
                "created_at": session_obj.created_at.isoformat(),
                "last_activity": session_obj.last_activity.isoformat(),
                "interactions": [
                    {
                        "role": i.role,
                        "content": i.content,
                        "timestamp": i.timestamp.isoformat(),
                        "metadata": i.metadata,
                    }
                    for i in session_obj.interactions
                ],
                "context": session_obj.context,
                "export_timestamp": datetime.now().isoformat(),
            }

            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)

            console.print(f"[green]✅ Session exported to: {output_file}[/green]")
            console.print(f"Interactions: {len(session_obj.interactions)}")

        except Exception as e:
            console.print(f"[red]❌ Export failed: {e}[/red]")

    asyncio.run(_export())


@session.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--new-session-id", help="Create new session with different ID")
@click.pass_context
def import_session(ctx, input_file: str, new_session_id: str) -> None:
    """Import session from file."""

    async def _import() -> None:
        await initialize_agents(enhanced=False)

        try:
            with open(input_file) as f:
                import_data = json.load(f)

            session_id = new_session_id or import_data["session_id"]
            session_obj = await session_manager.get_or_create_async(session_id)

            # Clear existing interactions if any
            session_obj.clear_interactions()

            # Restore interactions
            from gterminal.core.session import Interaction

            for interaction_data in import_data.get("interactions", []):
                interaction = Interaction(
                    role=interaction_data["role"],
                    content=interaction_data["content"],
                    timestamp=datetime.fromisoformat(interaction_data["timestamp"]),
                    metadata=interaction_data.get("metadata", {}),
                )
                session_obj.interactions.append(interaction)

            # Restore context
            session_obj.context = import_data.get("context", {})

            # Save imported session
            await session_manager.save(session_obj.id)

            console.print(f"[green]✅ Session imported as: {session_id}[/green]")
            console.print(f"Interactions: {len(session_obj.interactions)}")

        except Exception as e:
            console.print(f"[red]❌ Import failed: {e}[/red]")

    asyncio.run(_import())


@session.command()
def list_sessions() -> None:
    """List all available sessions."""

    async def _list() -> None:
        await initialize_agents(enhanced=False)

        try:
            sessions = await session_manager.list_sessions()

            if not sessions:
                console.print("[dim]No sessions found[/dim]")
                return

            sessions_table = Table(title="Available Sessions")
            sessions_table.add_column("Session ID", style="cyan")
            sessions_table.add_column("Created", style="yellow")
            sessions_table.add_column("Last Active", style="green")
            sessions_table.add_column("Interactions", justify="right", style="blue")

            for session_info in sessions:
                sessions_table.add_row(
                    session_info["id"][:16] + "..."
                    if len(session_info["id"]) > 16
                    else session_info["id"],
                    session_info["created_at"][:10],  # Date only
                    session_info["last_activity"][:10],  # Date only
                    str(session_info.get("interaction_count", 0)),
                )

            console.print(sessions_table)

        except Exception as e:
            console.print(f"[red]❌ Failed to list sessions: {e}[/red]")

    asyncio.run(_list())


@session.command()
@click.argument("session_id")
def show(session_id: str) -> None:
    """Show detailed session history."""

    async def _show() -> None:
        await initialize_agents(enhanced=False)

        try:
            session_obj = await session_manager.get_or_create_async(session_id)

            if not session_obj.interactions:
                console.print(f"[yellow]⚠️  Session {session_id} has no interactions[/yellow]")
                return

            # Display session summary
            summary_panel = Panel(
                f"Session ID: {session_obj.id}\n"
                f"Created: {session_obj.created_at}\n"
                f"Last Activity: {session_obj.last_activity}\n"
                f"Interactions: {len(session_obj.interactions)}",
                title="Session Summary",
                border_style="blue",
            )
            console.print(summary_panel)

            # Display recent interactions (last 5)
            console.print("\n[bold]Recent Interactions:[/bold]")
            for _i, interaction in enumerate(session_obj.interactions[-5:], 1):
                role_color = "blue" if interaction.role == "user" else "green"
                console.print(
                    f"\n[{role_color}]{interaction.role.upper()}[/{role_color}]: {interaction.content[:200]}{'...' if len(interaction.content) > 200 else ''}",
                )
                console.print(f"[dim]  {interaction.timestamp}[/dim]")

        except Exception as e:
            console.print(f"[red]❌ Failed to show session: {e}[/red]")

    asyncio.run(_show())


@cli.command()
def system() -> None:
    """Show system status and diagnostics."""

    async def _system() -> None:
        try:
            # Initialize minimal components
            await initialize_agents(enhanced=True)

            # System information table
            system_table = Table(title="System Status")
            system_table.add_column("Component", style="cyan")
            system_table.add_column("Status", style="green")
            system_table.add_column("Details", style="dim")

            # Rust extensions status
            if RUST_EXTENSIONS_AVAILABLE:
                rust_status_info = get_rust_status()
                system_table.add_row("Rust Extensions", "🦀 Active", "Performance mode enabled")

                cache_stats = rust_status_info["cache_stats"]
                system_table.add_row(
                    "Rust Cache",
                    f"✅ {cache_stats['hit_rate']:.1%} hit rate",
                    f"{cache_stats['access_count']} accesses",
                )
            else:
                system_table.add_row("Python Fallback", "🐍 Active", "Standard performance mode")

            # Job manager status
            if job_manager:
                mgr_status = await job_manager.get_manager_status()
                system_table.add_row(
                    "Job Manager", "✅ Active", f"{mgr_status['total_jobs']} total jobs"
                )

            # React engine status
            if react_engine:
                engine_status = await react_engine.get_engine_status()
                system_table.add_row(
                    "React Engine",
                    "✅ Active",
                    f"Config: {engine_status['config']['autonomy_level']}",
                )

            # Session manager status
            if session_manager:
                active_sessions = session_manager.get_active_sessions()
                system_table.add_row(
                    "Session Manager", "✅ Active", f"{len(active_sessions)} active sessions"
                )

            console.print(system_table)

            # Performance metrics
            if RUST_EXTENSIONS_AVAILABLE:
                try:
                    perf_metrics = rust_system.get_performance_metrics()
                    console.print(
                        f"\n[dim]CPU: {perf_metrics['cpu_percent']:.1f}% | "
                        f"Memory: {perf_metrics['memory_percent']:.1f}% | "
                        f"Rust Acceleration: Active[/dim]",
                    )
                except Exception:
                    console.print("\n[dim]Performance metrics unavailable[/dim]")

        except Exception as e:
            console.print(f"[red]❌ System status failed: {e}[/red]")

    asyncio.run(_system())


@cli.group()
def profile() -> None:
    """Configuration profile management commands."""


@profile.command()
def list_profiles() -> None:
    """List available configuration profiles."""
    profiles_dir = Path.home() / ".fullstack-agent" / "profiles"

    if not profiles_dir.exists():
        console.print("[dim]No profiles directory found[/dim]")
        return

    profile_files = list(profiles_dir.glob("*.json"))

    if not profile_files:
        console.print("[dim]No profiles found[/dim]")
        return

    profiles_table = Table(title="Available Profiles")
    profiles_table.add_column("Profile Name", style="cyan")
    profiles_table.add_column("File", style="yellow")
    profiles_table.add_column("Modified", style="green")

    for profile_file in profile_files:
        name = profile_file.stem
        modified = datetime.fromtimestamp(profile_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        profiles_table.add_row(name, profile_file.name, modified)

    console.print(profiles_table)


@profile.command()
@click.argument("profile_name")
def show(profile_name: str) -> None:
    """Show profile configuration details."""
    config = load_profile(profile_name)

    console.print(f"\n[bold]Profile: {profile_name}[/bold]")
    console.print_json(data=config)


@profile.command()
@click.argument("profile_name")
@click.option("--timeout", type=int, help="Default timeout")
@click.option("--max-jobs", type=int, help="Maximum concurrent jobs")
@click.option("--focus-areas", help="Default focus areas (comma-separated)")
def create(profile_name: str, timeout: int, max_jobs: int, focus_areas: str) -> None:
    """Create a new configuration profile."""
    config = {
        "timeout": timeout or 300,
        "max_concurrent_jobs": max_jobs or 5,
        "default_focus_areas": focus_areas.split(",")
        if focus_areas
        else ["security", "performance", "quality"],
        "output_format": "rich",
        "auto_save_sessions": True,
        "ui_theme": "default",
        "enable_redis": True,
        "enable_rag": True,
        "enable_autonomous": False,
        "enable_streaming": True,
        "cache_responses": True,
        "parallel_execution": True,
    }

    config_path = Path.home() / ".fullstack-agent" / "profiles" / f"{profile_name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    console.print(f"[green]✅ Profile '{profile_name}' created at {config_path}[/green]")


@profile.command()
@click.argument("profile_name")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def delete(profile_name: str, confirm: bool) -> None:
    """Delete a configuration profile."""
    if profile_name == "default":
        console.print("[red]❌ Cannot delete default profile[/red]")
        return

    config_path = Path.home() / ".fullstack-agent" / "profiles" / f"{profile_name}.json"

    if not config_path.exists():
        console.print(f"[red]❌ Profile '{profile_name}' not found[/red]")
        return

    if not confirm and not click.confirm(f"Delete profile '{profile_name}'?"):
        console.print("[yellow]Operation cancelled[/yellow]")
        return

    config_path.unlink()
    console.print(f"[green]✅ Profile '{profile_name}' deleted[/green]")


@cli.command()
def examples() -> None:
    """Show example commands and usage patterns."""
    examples = [
        {
            "title": "Phase 2 Interactive & Batch",
            "commands": [
                "agent interactive --profile dev --enhanced",
                "agent batch scripts/analysis.yaml --live-output",
                "agent jobs list --state running",
                "agent jobs pause <job_id>",
                "agent jobs monitor <job_id> --follow",
            ],
        },
        {
            "title": "Session Management",
            "commands": [
                "agent session list-sessions",
                "agent session show <session_id>",
                "agent session export my_session output.json",
                "agent session import-session saved.json --new-session-id new_id",
                "agent system  # Show system status",
            ],
        },
        {
            "title": "Profile Management & Validation",
            "commands": [
                "agent profile list-profiles",
                "agent profile show dev",
                "agent profile create dev --max-jobs 10 --focus-areas security,performance",
                "agent --profile dev review file app.py",
                "agent validate --quick  # Test Phase 2 components",
            ],
        },
        {
            "title": "Code Review",
            "commands": [
                "agent review file /path/to/file.py --focus security,performance",
                "agent review directory /path/to/project --patterns '*.py,*.js'",
                "agent review file app.py --severity high --no-suggestions",
            ],
        },
        {
            "title": "Workspace Analysis",
            "commands": [
                "agent workspace analyze /path/to/project",
                "agent workspace structure /path/to/project --max-depth 2",
                "agent workspace analyze . --depth quick --no-deps",
            ],
        },
        {
            "title": "Documentation Generation",
            "commands": [
                "agent docs generate /path/to/source --type api",
                "agent docs readme /path/to/project --style comprehensive",
                "agent docs generate . --format html --output docs.html",
            ],
        },
        {
            "title": "Multi-Agent Orchestration",
            "commands": [
                "agent orchestrate task 'Review and document this codebase'",
                "agent orchestrate task 'Optimize performance' --agents code-reviewer,workspace-analyzer",
                "agent orchestrate session-create --session-id my-session",
            ],
        },
    ]

    console.print(
        Panel("[bold cyan]🚀 Unified Agent CLI - Phase 2 Examples[/bold cyan]", border_style="cyan")
    )

    for example_group in examples:
        console.print(f"\n[bold yellow]{example_group['title']}:[/bold yellow]")
        for cmd in example_group["commands"]:
            console.print(f"  [green]$[/green] {cmd}")
        console.print()


@cli.command()
@click.option("--quick", is_flag=True, help="Run quick validation only")
def validate() -> None:
    """Validate Phase 2 CLI features and components."""

    async def _validate() -> None:
        console.print(
            Panel("[bold cyan]🔍 CLI Phase 2 Validation[/bold cyan]", border_style="cyan")
        )

        validation_results = []

        # Test 1: Component imports
        console.print("[cyan]Testing component imports...[/cyan]")
        try:
            from gterminal.core.interfaces.job_control import get_job_manager

            validation_results.append(("Component Imports", "✅", "All imports successful"))
        except Exception as e:
            validation_results.append(("Component Imports", "❌", f"Import error: {e}"))

        # Test 2: Profile system
        console.print("[cyan]Testing profile system...[/cyan]")
        try:
            test_config = load_profile("default")
            if test_config and isinstance(test_config, dict):
                validation_results.append(
                    ("Profile System", "✅", f"Default profile loaded ({len(test_config)} keys)")
                )
            else:
                validation_results.append(
                    ("Profile System", "⚠️", "Profile loaded but may be empty")
                )
        except Exception as e:
            validation_results.append(("Profile System", "❌", f"Profile error: {e}"))

        # Test 3: Rust extensions
        console.print("[cyan]Testing Rust extensions...[/cyan]")
        if RUST_EXTENSIONS_AVAILABLE:
            try:
                status = get_rust_status()
                validation_results.append(
                    (
                        "Rust Extensions",
                        "✅",
                        f"Active with {status['cache_stats']['capacity']} cache capacity",
                    ),
                )
            except Exception as e:
                validation_results.append(
                    ("Rust Extensions", "⚠️", f"Available but status error: {e}")
                )
        else:
            validation_results.append(("Rust Extensions", "📋", "Using Python fallback"))

        # Test 4: Job manager (if not quick)
        if not quick:
            console.print("[cyan]Testing job manager...[/cyan]")
            try:
                job_mgr = await get_job_manager()
                status = await job_mgr.get_manager_status()
                validation_results.append(
                    ("Job Manager", "✅", f"Active with {status.get('total_jobs', 0)} total jobs"),
                )
            except Exception as e:
                validation_results.append(("Job Manager", "❌", f"Job manager error: {e}"))

        # Test 5: Session manager
        console.print("[cyan]Testing session manager...[/cyan]")
        try:
            global session_manager
            if session_manager is None:
                from gterminal.core.session import SessionManager

                session_manager = SessionManager()
            validation_results.append(("Session Manager", "✅", "Session manager available"))
        except Exception as e:
            validation_results.append(("Session Manager", "❌", f"Session manager error: {e}"))

        # Display results
        results_table = Table(title="Validation Results")
        results_table.add_column("Component", style="cyan", width=20)
        results_table.add_column("Status", width=8)
        results_table.add_column("Details", style="dim")

        for component, status, details in validation_results:
            results_table.add_row(component, status, details)

        console.print(results_table)

        # Summary
        passed = sum(1 for _, status, _ in validation_results if status == "✅")
        total = len(validation_results)

        if passed == total:
            console.print(f"\n[bold green]🎉 All {total} validations passed![/bold green]")
        else:
            console.print(f"\n[yellow]⚠️  {passed}/{total} validations passed[/yellow]")

    asyncio.run(_validate())


# ===========================
# Display Helper Functions
# ===========================


def _display_review_results(file_path: str, result: dict[str, Any]) -> None:
    """Display code review results in a formatted way."""
    console.print(f"\n[bold]Code Review Results for {Path(file_path).name}[/bold]")

    if "issues" in result:
        issues = result["issues"]
        if issues:
            table = Table(title="Issues Found")
            table.add_column("Severity", style="red")
            table.add_column("Line", justify="right", style="yellow")
            table.add_column("Issue", style="white")
            table.add_column("Category", style="cyan")

            for issue in issues[:10]:  # Show first 10
                table.add_row(
                    issue.get("severity", "unknown"),
                    str(issue.get("line", "")),
                    issue.get("description", "")[:80] + "..."
                    if len(issue.get("description", "")) > 80
                    else issue.get("description", ""),
                    issue.get("category", ""),
                )

            console.print(table)

            if len(issues) > 10:
                console.print(f"\n[dim]... and {len(issues) - 10} more issues[/dim]")
        else:
            console.print("[green]✅ No issues found![/green]")

    # Show summary stats if available
    if "summary" in result:
        summary = result["summary"]
        stats_panel = Panel(
            f"Total Issues: {summary.get('total_issues', 0)}\n"
            f"Critical: {summary.get('critical', 0)}\n"
            f"High: {summary.get('high', 0)}\n"
            f"Medium: {summary.get('medium', 0)}\n"
            f"Low: {summary.get('low', 0)}",
            title="Summary",
            border_style="blue",
        )
        console.print(stats_panel)


def _display_directory_review_results(directory_path: str, result: dict[str, Any]) -> None:
    """Display directory review results."""
    console.print(f"\n[bold]Directory Review Results for {directory_path}[/bold]")

    if "files_reviewed" in result:
        console.print(f"Files reviewed: {result['files_reviewed']}")

    if "summary" in result:
        summary = result["summary"]
        console.print(f"Total issues found: {summary.get('total_issues', 0)}")


def _display_workspace_results(project_path: str, result: dict[str, Any]) -> None:
    """Display workspace analysis results."""
    console.print(f"\n[bold]Workspace Analysis Results for {project_path}[/bold]")

    if "summary" in result:
        summary = result["summary"]
        stats_text = f"""
Files analyzed: {summary.get("files_analyzed", 0)}
Lines of code: {summary.get("lines_of_code", 0)}
Architecture score: {summary.get("architecture_score", "N/A")}
Quality score: {summary.get("quality_score", "N/A")}
"""
        console.print(Panel(stats_text, title="Project Summary", border_style="green"))


def _display_project_structure(project_path: str, result: dict[str, Any]) -> None:
    """Display project structure."""
    console.print(f"\n[bold]Project Structure for {project_path}[/bold]")
    console.print_json(data=result)


def _display_documentation_results(source_path: str, result: dict[str, Any]) -> None:
    """Display documentation generation results."""
    console.print(f"\n[bold]Documentation Generated for {source_path}[/bold]")

    if isinstance(result, dict) and "content" in result:
        syntax = Syntax(result["content"][:1000], "markdown", theme="monokai")
        console.print(syntax)
        if len(result["content"]) > 1000:
            console.print(f"\n[dim]... and {len(result['content']) - 1000} more characters[/dim]")
    else:
        console.print(str(result)[:1000])


def _display_orchestration_results(task: str, result: dict[str, Any]) -> None:
    """Display orchestration results."""
    console.print("\n[bold]Orchestration Results[/bold]")
    console.print(f"Task: {task}")

    if "agents_used" in result:
        console.print(f"Agents used: {', '.join(result['agents_used'])}")

    if "result" in result:
        console.print("\n[bold]Result:[/bold]")
        console.print(result["result"])


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
