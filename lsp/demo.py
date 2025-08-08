#!/usr/bin/env python3
"""
Ruff LSP Integration Demo

This script demonstrates the complete ruff LSP integration system with:
- LSP server connection and management
- Real-time diagnostic streaming
- AI-powered fix suggestions
- Filewatcher integration
- Dynamic configuration management
- Performance monitoring

Usage:
    python demo.py [--workspace PATH] [--test-file PATH] [--mode MODE]
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

from ai_suggestion_engine import AISuggestionEngine
from ai_suggestion_engine import SuggestionRequest
from config_manager import RuffConfigManager
from filewatcher_integration import FilewatcherConfig
from filewatcher_integration import FilewatcherIntegration
from performance_monitor import LSPPerformanceMonitor
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.table import Table

# Import our LSP integration modules
from ruff_lsp_client import RuffLSPClient
from ruff_lsp_client import RuffLSPConfig

if TYPE_CHECKING:
    from diagnostic_streamer import DiagnosticStreamer


class RuffLSPDemo:
    """Comprehensive demonstration of the ruff LSP integration system"""

    def __init__(self, workspace: Path, console: Console):
        self.workspace = workspace
        self.console = console
        self.logger = logging.getLogger("ruff-lsp-demo")

        # Initialize all components
        self.lsp_client: RuffLSPClient | None = None
        self.diagnostic_streamer: DiagnosticStreamer | None = None
        self.ai_engine: AISuggestionEngine | None = None
        self.performance_monitor: LSPPerformanceMonitor | None = None
        self.filewatcher: FilewatcherIntegration | None = None
        self.config_manager: RuffConfigManager | None = None

        # Demo state
        self.demo_stats = {
            "files_processed": 0,
            "diagnostics_found": 0,
            "ai_suggestions_generated": 0,
            "performance_issues_detected": 0,
            "config_optimizations": 0,
        }

    async def run_complete_demo(self) -> None:
        """Run the complete LSP integration demo"""
        self.console.print(Panel.fit("🚀 Ruff LSP Integration System Demo", style="bold blue"))

        try:
            # Step 1: Project analysis and configuration
            await self._demo_project_analysis()

            # Step 2: LSP server setup
            await self._demo_lsp_setup()

            # Step 3: Diagnostic streaming
            await self._demo_diagnostic_streaming()

            # Step 4: AI suggestions
            await self._demo_ai_suggestions()

            # Step 5: Filewatcher integration
            await self._demo_filewatcher_integration()

            # Step 6: Performance monitoring
            await self._demo_performance_monitoring()

            # Step 7: Live dashboard
            await self._demo_live_dashboard()

        except Exception as e:
            self.console.print(f"❌ Demo failed: {e}", style="bold red")
            raise
        finally:
            await self._cleanup_demo()

    async def _demo_project_analysis(self) -> None:
        """Demonstrate project analysis and dynamic configuration"""
        self.console.print(
            "\n📊 [bold cyan]Step 1: Project Analysis & Dynamic Configuration[/bold cyan]"
        )

        self.config_manager = RuffConfigManager(self.workspace)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing project structure...", total=None)

            # Analyze the project
            await self.config_manager.analyze_project()
            progress.update(task, description="Generating optimized configuration...")

            # Generate configuration
            await self.config_manager.generate_config()

            # Save configurations
            await self.config_manager.save_config()
            await self.config_manager.create_lsp_server_config()

            progress.update(task, description="✅ Project analysis complete")

        # Display analysis results
        table = Table(title="Project Analysis Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        stats = self.config_manager.get_project_stats()
        table.add_row("Project Type", stats.get("project_type", "unknown"))
        table.add_row("Frameworks", ", ".join(stats.get("frameworks", [])))
        table.add_row("Python Files", str(stats.get("file_count", 0)))
        table.add_row("Total Lines", str(stats.get("total_lines", 0)))
        table.add_row("Suggested Rules", str(stats.get("suggested_rules_count", 0)))
        table.add_row("Performance Critical", str(stats.get("performance_critical", False)))

        self.console.print(table)
        self.demo_stats["config_optimizations"] += 1

    async def _demo_lsp_setup(self) -> None:
        """Demonstrate LSP server setup and connection"""
        self.console.print("\n🔧 [bold cyan]Step 2: LSP Server Setup[/bold cyan]")

        # Create LSP configuration
        lsp_config = RuffLSPConfig(
            workspace_root=self.workspace,
            ruff_config_path=self.workspace / ".ruff-server.json",
            enable_ai_suggestions=True,
            enable_performance_monitoring=True,
            metrics_file=self.workspace / "lsp-metrics.json",
        )

        # Initialize and start LSP client
        self.lsp_client = RuffLSPClient(lsp_config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Starting LSP server...", total=None)

            await self.lsp_client.start()

            progress.update(task, description="✅ LSP server started successfully")

        # Health check
        health = await self.lsp_client.health_check()
        status_color = "green" if health["status"] == "healthy" else "yellow"
        self.console.print(
            f"LSP Health Status: [{status_color}]{health['status']}[/{status_color}]"
        )

    async def _demo_diagnostic_streaming(self) -> None:
        """Demonstrate real-time diagnostic streaming"""
        self.console.print("\n📡 [bold cyan]Step 3: Real-time Diagnostic Streaming[/bold cyan]")

        # Find Python files to test with
        test_files = list(self.workspace.glob("**/*.py"))[:5]  # Test with first 5 files

        if not test_files:
            self.console.print("⚠️ No Python files found for testing", style="yellow")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing files for diagnostics...", total=len(test_files))

            for file_path in test_files:
                # Open file in LSP
                await self.lsp_client.open_document(file_path)

                # Wait for diagnostics
                await asyncio.sleep(0.5)

                # Get diagnostics
                diagnostics = self.lsp_client.get_diagnostics(file_path)

                if diagnostics:
                    self.console.print(f"📝 {file_path.name}: {len(diagnostics)} issues found")
                    self.demo_stats["diagnostics_found"] += len(diagnostics)
                else:
                    self.console.print(f"✅ {file_path.name}: No issues found")

                self.demo_stats["files_processed"] += 1
                progress.advance(task)

    async def _demo_ai_suggestions(self) -> None:
        """Demonstrate AI-powered fix suggestions"""
        self.console.print("\n🤖 [bold cyan]Step 4: AI-Powered Fix Suggestions[/bold cyan]")

        # Initialize AI engine
        self.ai_engine = AISuggestionEngine(claude_cli="claude", model="haiku")

        # Find files with diagnostics
        files_with_issues = []
        for file_path in self.workspace.glob("**/*.py"):
            if file_path.name.startswith("test_"):
                continue  # Skip test files for this demo

            diagnostics = self.lsp_client.get_diagnostics(file_path) if self.lsp_client else []
            if diagnostics:
                files_with_issues.append((file_path, diagnostics))

            if len(files_with_issues) >= 3:  # Limit for demo
                break

        if not files_with_issues:
            self.console.print("ℹ️ No files with issues found for AI suggestions", style="blue")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Generating AI suggestions...", total=len(files_with_issues))

            for file_path, diagnostics in files_with_issues:
                # Convert diagnostics to format expected by AI engine
                diag_data = []
                for diag in diagnostics[:3]:  # Limit for demo
                    diag_data.append(
                        {
                            "code": diag.code or "unknown",
                            "message": diag.message,
                            "location": {
                                "row": diag.range.start.line + 1,
                                "column": diag.range.start.character + 1,
                            },
                        }
                    )

                # Generate AI suggestions
                request = SuggestionRequest(
                    file_path=file_path, diagnostics=diag_data, max_suggestions=2
                )

                try:
                    response = await self.ai_engine.generate_suggestions(request)

                    if response.suggestions:
                        self.console.print(
                            f"🤖 {file_path.name}: {len(response.suggestions)} AI suggestions (confidence: {response.confidence:.2f})"
                        )
                        self.demo_stats["ai_suggestions_generated"] += len(response.suggestions)
                    else:
                        self.console.print(f"🤖 {file_path.name}: No AI suggestions generated")

                except Exception as e:
                    self.console.print(
                        f"⚠️ AI suggestion failed for {file_path.name}: {e}",
                        style="yellow",
                    )

                progress.advance(task)

    async def _demo_filewatcher_integration(self) -> None:
        """Demonstrate filewatcher integration"""
        self.console.print("\n🔗 [bold cyan]Step 5: Filewatcher Integration[/bold cyan]")

        # Initialize filewatcher integration
        filewatcher_config = FilewatcherConfig(
            host="localhost", port=8768, auto_refresh_diagnostics=True
        )

        self.filewatcher = FilewatcherIntegration(filewatcher_config, self.lsp_client)

        # Try to connect (might fail if filewatcher isn't running)
        try:
            # Set up a brief connection test
            connection_test = asyncio.create_task(self.filewatcher.start())

            # Wait a short time for connection
            await asyncio.wait_for(asyncio.sleep(2), timeout=3)

            if self.filewatcher.is_connected():
                self.console.print("✅ Connected to filewatcher successfully")
            else:
                self.console.print(
                    "⚠️ Could not connect to filewatcher (not running?)", style="yellow"
                )
                self.console.print("   Start filewatcher with: cargo run -- watch", style="dim")

            connection_test.cancel()

        except Exception as e:
            self.console.print(f"⚠️ Filewatcher connection test failed: {e}", style="yellow")

    async def _demo_performance_monitoring(self) -> None:
        """Demonstrate performance monitoring"""
        self.console.print("\n📊 [bold cyan]Step 6: Performance Monitoring[/bold cyan]")

        # Initialize performance monitor
        metrics_file = self.workspace / "lsp-performance-metrics.json"
        self.performance_monitor = LSPPerformanceMonitor(metrics_file=metrics_file)

        await self.performance_monitor.start_monitoring()

        # Simulate some activity for metrics
        for i in range(5):
            await self.performance_monitor.record_request()
            await self.performance_monitor.record_response_time(50 + i * 10)
            await self.performance_monitor.record_response()
            await asyncio.sleep(0.1)

        # Get performance metrics
        metrics = self.performance_monitor.current_metrics

        # Display metrics table
        table = Table(title="LSP Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Avg Response Time", f"{metrics.avg_response_time_ms:.1f} ms")
        table.add_row("Requests Sent", str(metrics.requests_sent))
        table.add_row("Responses Received", str(metrics.responses_received))
        table.add_row("Diagnostics Received", str(metrics.diagnostics_received))
        table.add_row("Uptime", f"{metrics.uptime_seconds:.1f} seconds")

        self.console.print(table)

        await self.performance_monitor.stop_monitoring()

    async def _demo_live_dashboard(self) -> None:
        """Demonstrate live dashboard functionality"""
        self.console.print("\n📱 [bold cyan]Step 7: Live Dashboard (5 second preview)[/bold cyan]")

        # Create dashboard layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(Layout(name="stats"), Layout(name="files"))

        # Dashboard update function
        def update_dashboard():
            # Header
            layout["header"].update(Panel("🚀 Ruff LSP Integration Dashboard", style="bold blue"))

            # Stats table
            stats_table = Table(title="Demo Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Count", style="green")

            for key, value in self.demo_stats.items():
                formatted_key = key.replace("_", " ").title()
                stats_table.add_row(formatted_key, str(value))

            layout["stats"].update(stats_table)

            # Files table
            files_table = Table(title="Recent Activity")
            files_table.add_column("File", style="yellow")
            files_table.add_column("Status", style="green")

            python_files = list(self.workspace.glob("**/*.py"))[:5]
            for file_path in python_files:
                files_table.add_row(file_path.name, "✅ Processed")

            layout["files"].update(files_table)

            # Footer
            layout["footer"].update(
                Panel(f"Demo completed at {time.strftime('%H:%M:%S')}", style="dim")
            )

        # Show live dashboard for 5 seconds
        with Live(layout, console=self.console, refresh_per_second=2):
            for _ in range(10):  # 5 seconds at 2 fps
                update_dashboard()
                await asyncio.sleep(0.5)

    async def _cleanup_demo(self) -> None:
        """Clean up demo resources"""
        self.console.print("\n🧹 [bold cyan]Cleaning up demo resources[/bold cyan]")

        # Shutdown components
        if self.lsp_client:
            await self.lsp_client.shutdown()

        if self.filewatcher:
            await self.filewatcher.stop()

        if self.performance_monitor:
            await self.performance_monitor.stop_monitoring()

        self.console.print("✅ Demo cleanup complete")


async def main():
    """Main entry point for the demo"""
    parser = argparse.ArgumentParser(description="Ruff LSP Integration Demo")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace directory to analyze",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "analysis-only"],
        default="quick",
        help="Demo mode",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    console = Console()

    # Check workspace
    if not args.workspace.exists():
        console.print(f"❌ Workspace not found: {args.workspace}", style="bold red")
        sys.exit(1)

    # Initialize and run demo
    demo = RuffLSPDemo(args.workspace, console)

    try:
        if args.mode == "analysis-only":
            await demo._demo_project_analysis()
        elif args.mode == "quick":
            await demo._demo_project_analysis()
            await demo._demo_lsp_setup()
            await demo._demo_diagnostic_streaming()
        else:  # full
            await demo.run_complete_demo()

        console.print("\n🎉 [bold green]Demo completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.print("\n⚠️ Demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n❌ Demo failed: {e}", style="bold red")
        if args.verbose:
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
