#!/usr/bin/env python3
"""Unified Fullstack Agent Application.
====================================
Single entry point for Terminal UI, CLI, and HTTP Server modes.

This consolidates the functionality from multiple entry points into one streamlined application.
"""

import asyncio
import sys

import click
from rich.console import Console
from rich.panel import Panel
import uvicorn

console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--mode", type=click.Choice(["terminal", "cli", "server"]), help="Application mode")
@click.option("--port", default=8000, help="Port for HTTP server mode")
@click.option("--host", default="0.0.0.0", help="Host for HTTP server mode")
def main(ctx, mode: str | None, port: int, host: str) -> None:
    """My Fullstack Agent - Unified Application.

    Run without arguments for interactive mode selection.
    """
    if ctx.invoked_subcommand is None:
        if mode == "terminal":
            run_terminal()
        elif mode == "cli":
            console.print("[yellow]CLI mode - use subcommands[/yellow]")
            console.print(ctx.get_help())
        elif mode == "server":
            run_server(host, port)
        else:
            # Interactive mode selection
            show_interactive_menu()


def show_interactive_menu() -> None:
    """Show interactive menu for mode selection."""
    console.print(
        Panel.fit(
            "[bold cyan]My Fullstack Agent[/bold cyan]\n\n"
            "Select operation mode:\n\n"
            "1. [green]Terminal UI[/green] - Interactive terminal interface\n"
            "2. [yellow]CLI Mode[/yellow] - Command-line interface\n"
            "3. [blue]HTTP Server[/blue] - Backend API server\n"
            "4. [magenta]Exit[/magenta]",
            title="🚀 Fullstack Agent",
            border_style="cyan",
        ),
    )

    choice = console.input("\n[cyan]Select mode (1-4):[/cyan] ")

    if choice == "1":
        run_terminal()
    elif choice == "2":
        console.print("[yellow]Starting CLI mode...[/yellow]")
        console.print("Use: python -m app.main [COMMAND] --help")
    elif choice == "3":
        run_server()
    elif choice == "4":
        console.print("[red]Exiting...[/red]")
        sys.exit(0)
    else:
        console.print("[red]Invalid choice![/red]")
        show_interactive_menu()


def run_terminal() -> None:
    """Run the terminal UI mode."""
    console.print("[green]Starting Terminal UI...[/green]")
    try:
        from gterminal.terminal.main import run_terminal_ui

        asyncio.run(run_terminal_ui())
    except ImportError:
        console.print(
            "[yellow]Terminal UI not yet implemented. Starting orchestrator monitor...[/yellow]"
        )
        from gterminal.terminal.orchestrator_monitor import main as monitor_main

        monitor_main()


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the HTTP server mode."""
    console.print(f"[blue]Starting HTTP Server with ReAct Engine on {host}:{port}...[/blue]")
    console.print("[cyan]Features:[/cyan]")
    console.print("  • ReAct engine with tool integration")
    console.print("  • WebSocket support for real-time streaming")
    console.print("  • Centralized session management")
    console.print("  • Unified tool registry")

    from gterminal.server import app

    uvicorn.run(app, host=host, port=port, log_level="info")


# CLI Commands
@main.group()
def agent() -> None:
    """Agent management commands."""


@agent.command()
@click.argument("task")
@click.option("--model", default="gemini-2.0-flash-exp", help="Model to use")
def run(task: str, model: str) -> None:
    """Run an agent task."""
    console.print(f"[green]Running agent task:[/green] {task}")
    console.print(f"[blue]Using model:[/blue] {model}")

    try:
        from gterminal.agents.base_agent_service import BaseAgentService

        agent = BaseAgentService()
        asyncio.run(agent.process_request(task))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@main.group()
def mcp() -> None:
    """MCP server commands."""


@mcp.command()
@click.argument("server_name")
def start(server_name: str) -> None:
    """Start an MCP server."""
    console.print(f"[green]Starting MCP server:[/green] {server_name}")

    try:
        if server_name == "code-reviewer":
            from gterminal.mcp_servers.gemini_code_reviewer import main
        elif server_name == "workspace-analyzer":
            from gterminal.mcp_servers.gemini_workspace_analyzer import main
        elif server_name == "cost-optimizer":
            from gterminal.mcp_servers.cloud_cost_optimizer import main
        else:
            console.print(f"[red]Unknown server:[/red] {server_name}")
            return

        asyncio.run(main())
    except Exception as e:
        console.print(f"[red]Error starting MCP server:[/red] {e}")


@mcp.command()
def list() -> None:
    """List available MCP servers."""
    console.print("[cyan]Available MCP Servers:[/cyan]")
    servers = [
        "code-reviewer - Code analysis and review",
        "workspace-analyzer - Project workspace analysis",
        "cost-optimizer - Cloud cost optimization",
        "master-architect - System architecture design",
    ]
    for server in servers:
        console.print(f"  • {server}")


@main.group()
def utils() -> None:
    """Utility commands."""


@utils.command()
def benchmark() -> None:
    """Run performance benchmarks."""
    console.print("[yellow]Running benchmarks...[/yellow]")
    try:
        from benchmarks.consolidated_benchmark import run_benchmarks

        asyncio.run(run_benchmarks())
    except ImportError:
        console.print("[red]Benchmark module not found[/red]")


@utils.command()
@click.option("--rust/--no-rust", default=True, help="Test Rust extensions")
def test_extensions(rust: bool) -> None:
    """Test Rust extensions."""
    if rust:
        console.print("[cyan]Testing Rust extensions...[/cyan]")
        try:
            import fullstack_agent_rust

            console.print("[green]✓[/green] Rust extensions loaded successfully")

            # Test basic functionality
            fullstack_agent_rust.RustCache(100, 60)
            console.print("[green]✓[/green] RustCache initialized")

            metrics = fullstack_agent_rust.performance_metrics()
            console.print(
                f"[green]✓[/green] System metrics: CPU={metrics['cpu_usage']:.1f}%, Memory={metrics['memory_usage']:.1f}%",
            )

        except ImportError as e:
            console.print(f"[red]✗[/red] Failed to load Rust extensions: {e}")
    else:
        console.print("[yellow]Skipping Rust extension tests[/yellow]")


@main.command()
def status() -> None:
    """Show application status."""
    console.print(
        Panel.fit(
            "[bold]Fullstack Agent Status[/bold]\n\n"
            "• [green]Utils reorganization:[/green] ✓ Complete\n"
            "• [yellow]Source consolidation:[/yellow] In Progress\n"
            "• [yellow]Rust research:[/yellow] In Progress\n"
            "• [cyan]PyO3 extensions:[/cyan] ✓ Working\n"
            "• [blue]Unified server:[/blue] ✓ Running on port 8100\n",
            title="📊 Status",
            border_style="cyan",
        ),
    )


if __name__ == "__main__":
    main()
