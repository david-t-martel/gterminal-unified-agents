#!/usr/bin/env python3
"""Gemini Server v0.4.0 - Fullstack Agent with PyO3 Integration.

High-performance server for code analysis and generation powered by Google Gemini
and optimized with Rust extensions via PyO3.
"""

import asyncio
import json
from pathlib import Path
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import Rust extensions for performance
try:
    from fullstack_agent_rust import RustAuthValidator
    from fullstack_agent_rust import RustFileOps
    from fullstack_agent_rust import RustPerformanceMetrics
    from fullstack_agent_rust import RustResourceMonitor
    from fullstack_agent_rust import RustSearchEngine
    from fullstack_agent_rust import RustTokenManager

    RUST_ENABLED = True
except ImportError:
    RUST_ENABLED = False

# Import agent implementations
try:
    from gterminal.agents.code_review_agent import CodeReviewAgentService
    from gterminal.agents.documentation_generator_agent import DocumentationGeneratorService
    from gterminal.agents.workspace_analyzer_agent import WorkspaceAnalyzerService
    from gterminal.gemini_unified_server import GeminiUnifiedServer
except ImportError:
    # Try relative imports if running as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gterminal.agents.code_review_agent import CodeReviewAgentService
    from gterminal.agents.workspace_analyzer_agent import WorkspaceAnalyzerService
    from gterminal.gemini_unified_server import GeminiUnifiedServer

console = Console()

__version__ = "0.4.0"


class GeminiServerCLI:
    """Enhanced CLI for Gemini Server with Rust optimizations."""

    def __init__(self) -> None:
        self.server = None
        self.rust_monitor = RustResourceMonitor() if RUST_ENABLED else None
        self.rust_metrics = RustPerformanceMetrics() if RUST_ENABLED else None

    async def initialize(self) -> None:
        """Initialize the unified server."""
        self.server = GeminiUnifiedServer()
        await self.server.initialize()

    def show_status(self) -> None:
        """Display server status with Rust performance metrics."""
        table = Table(title="Gemini Server v0.4.0 Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Performance", style="yellow")

        # Rust extensions status
        rust_status = "✅ Enabled" if RUST_ENABLED else "❌ Disabled"
        table.add_row("Rust Extensions", rust_status, "PyO3 Integration")

        # Performance metrics
        if self.rust_metrics:
            metrics = self.rust_metrics.get_metrics()
            cpu_usage = metrics.get("cpu_usage_percent", 0)
            memory_usage = metrics.get("memory_usage_mb", 0)
            table.add_row("CPU Usage", f"{cpu_usage:.1f}%", "Real-time")
            table.add_row("Memory", f"{memory_usage:.1f} MB", "Current")

        # Agent status
        table.add_row("Code Review Agent", "✅ Ready", "Rust-accelerated")
        table.add_row("Documentation Agent", "✅ Ready", "High-performance")
        table.add_row("Workspace Analyzer", "✅ Ready", "Parallel processing")

        console.print(table)

    async def analyze_code(self, file_path: str) -> None:
        """Analyze code using Rust-accelerated file operations."""
        if RUST_ENABLED:
            file_ops = RustFileOps()
            content = file_ops.read_file(file_path)
        else:
            with open(file_path) as f:
                content = f.read()

        agent = CodeReviewAgentService()
        result = await agent.analyze(content)

        console.print(Panel(result, title=f"Code Analysis: {file_path}", border_style="green"))

    async def analyze_workspace(self, directory: str) -> None:
        """Analyze workspace with Rust search engine."""
        agent = WorkspaceAnalyzerService()

        if RUST_ENABLED:
            search_engine = RustSearchEngine()
            # Use Rust's high-performance file search
            py_files = search_engine.search_files(directory, "*.py", False)
            console.print(f"Found {len(py_files)} Python files using Rust engine")

        result = await agent.analyze(directory)
        console.print(
            Panel(
                json.dumps(result, indent=2),
                title=f"Workspace Analysis: {directory}",
                border_style="blue",
            ),
        )


@click.group()
@click.version_option(__version__, prog_name="Gemini Server")
def cli() -> None:
    """Gemini Server v0.4.0 - AI-powered development assistant with Rust extensions."""


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, help="Server port")
def serve(host: str, port: int) -> None:
    """Start the Gemini server."""
    console.print(f"[bold green]Starting Gemini Server v{__version__}[/bold green]")
    console.print(f"Rust Extensions: {'Enabled ✅' if RUST_ENABLED else 'Disabled ❌'}")

    async def run_server() -> None:
        server = GeminiServerCLI()
        await server.initialize()
        server.show_status()

        # Start the unified server
        import uvicorn

        config = uvicorn.Config(
            "app.gemini_unified_server:app", host=host, port=port, reload=False, log_level="info"
        )
        server_instance = uvicorn.Server(config)
        await server_instance.serve()

    asyncio.run(run_server())


@cli.command()
def status() -> None:
    """Show server status and performance metrics."""

    async def show() -> None:
        server = GeminiServerCLI()
        await server.initialize()
        server.show_status()

    asyncio.run(show())


@cli.command()
@click.argument("file_path")
def analyze(file_path: str) -> None:
    """Analyze a code file."""

    async def run_analysis() -> None:
        server = GeminiServerCLI()
        await server.initialize()
        await server.analyze_code(file_path)

    asyncio.run(run_analysis())


@cli.command()
@click.argument("directory")
def workspace(directory: str) -> None:
    """Analyze a workspace directory."""

    async def run_workspace_analysis() -> None:
        server = GeminiServerCLI()
        await server.initialize()
        await server.analyze_workspace(directory)

    asyncio.run(run_workspace_analysis())


@cli.command()
def benchmark() -> None:
    """Run performance benchmarks."""
    console.print("[bold cyan]Running Performance Benchmarks...[/bold cyan]")

    if not RUST_ENABLED:
        console.print("[red]Rust extensions not available. Install for benchmarks.[/red]")
        return

    import time

    # File operations benchmark
    file_ops = RustFileOps()
    start = time.time()
    for _ in range(1000):
        file_ops.list_directory(".")
    file_time = time.time() - start

    # Auth operations benchmark
    auth = RustAuthValidator()
    start = time.time()
    for i in range(1000):
        auth.hash_password(f"password{i}")
    auth_time = time.time() - start

    table = Table(title="Performance Benchmarks")
    table.add_column("Operation", style="cyan")
    table.add_column("Time (ms)", style="green")
    table.add_column("Ops/sec", style="yellow")

    table.add_row("File Operations (1000x)", f"{file_time * 1000:.2f}", f"{1000 / file_time:.0f}")
    table.add_row("Auth Hashing (1000x)", f"{auth_time * 1000:.2f}", f"{1000 / auth_time:.0f}")

    console.print(table)


if __name__ == "__main__":
    cli()
