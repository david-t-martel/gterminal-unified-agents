#!/usr/bin/env python3
"""Real-time Monitoring Interface for CLI Adapter - Phase 2
Provides real-time monitoring capabilities for system status and job progress.
"""

import asyncio
from datetime import datetime
import logging
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.table import Table

from gterminal.utils.rust_extensions.wrapper import RUST_EXTENSIONS_AVAILABLE
from gterminal.utils.rust_extensions.wrapper import get_rust_status

logger = logging.getLogger(__name__)
console = Console()


class SystemMetrics(BaseModel):
    """System performance metrics."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    rust_active: bool = False
    cache_hit_rate: float = 0.0
    active_jobs: int = 0
    timestamp: datetime


class RealTimeMonitor:
    """Real-time system and job monitoring."""

    def __init__(self) -> None:
        self.running = False
        self.update_interval = 2.0
        self.metrics_history: list[SystemMetrics] = []
        self.max_history = 100

    async def start_monitoring(
        self,
        job_manager=None,
        update_interval: float = 2.0,
        show_jobs: bool = True,
        show_system: bool = True,
    ) -> None:
        """Start real-time monitoring dashboard."""
        self.running = True
        self.update_interval = update_interval

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=3)
        )

        if show_jobs and show_system:
            layout["body"].split_row(Layout(name="jobs"), Layout(name="system"))
        elif show_jobs:
            layout["body"] = Layout(name="jobs")
        else:
            layout["body"] = Layout(name="system")

        try:
            with Live(layout, refresh_per_second=1 / update_interval, screen=True):
                while self.running:
                    # Update header
                    layout["header"].update(
                        Panel(
                            f"[bold cyan]🔍 Real-Time Monitor[/bold cyan] | "
                            f"Updated: {datetime.now().strftime('%H:%M:%S')} | "
                            f"Interval: {update_interval}s",
                            style="cyan",
                        ),
                    )

                    # Update system metrics
                    if show_system:
                        system_panel = await self._create_system_panel()
                        layout["system" if show_jobs else "body"].update(system_panel)

                    # Update jobs panel
                    if show_jobs and job_manager:
                        jobs_panel = await self._create_jobs_panel(job_manager)
                        layout["jobs" if show_system else "body"].update(jobs_panel)

                    # Update footer
                    layout["footer"].update(
                        Panel("[dim]Press Ctrl+C to exit monitoring[/dim]", style="dim")
                    )

                    await asyncio.sleep(update_interval)

        except KeyboardInterrupt:
            self.running = False
            console.print("\n[yellow]Monitoring stopped[/yellow]")

    async def _create_system_panel(self) -> Panel:
        """Create system metrics panel."""
        # Collect system metrics
        metrics = SystemMetrics(timestamp=datetime.now(), rust_active=RUST_EXTENSIONS_AVAILABLE)

        if RUST_EXTENSIONS_AVAILABLE:
            try:
                rust_status = get_rust_status()
                metrics.cache_hit_rate = rust_status["cache_stats"]["hit_rate"]
            except Exception:
                pass

        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)

        # Create system table
        system_table = Table(title="System Status", show_header=True, header_style="bold blue")
        system_table.add_column("Metric", style="cyan", width=20)
        system_table.add_column("Value", style="green")
        system_table.add_column("Trend", style="yellow", width=10)

        # Add rows
        system_table.add_row(
            "Rust Extensions",
            "🦀 Active" if metrics.rust_active else "🐍 Python",
            "✅" if metrics.rust_active else "➖",
        )

        if metrics.rust_active and metrics.cache_hit_rate > 0:
            system_table.add_row(
                "Cache Hit Rate",
                f"{metrics.cache_hit_rate:.1%}",
                "📈" if metrics.cache_hit_rate > 0.8 else "📊",
            )

        system_table.add_row(
            "Uptime", str(datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)), "⏱️"
        )

        system_table.add_row("History Entries", str(len(self.metrics_history)), "📊")

        return Panel(
            system_table, title="[bold green]System Monitoring[/bold green]", border_style="green"
        )

    async def _create_jobs_panel(self, job_manager) -> Panel:
        """Create jobs monitoring panel."""
        try:
            jobs_list = await job_manager.list_jobs()

            if not jobs_list:
                return Panel(
                    "[dim]No active jobs[/dim]",
                    title="[bold blue]Job Monitoring[/bold blue]",
                    border_style="blue",
                )

            # Create jobs table
            jobs_table = Table(title="Active Jobs", show_header=True, header_style="bold blue")
            jobs_table.add_column("ID", style="cyan", width=12)
            jobs_table.add_column("Name", style="white", width=25)
            jobs_table.add_column("State", style="yellow", width=12)
            jobs_table.add_column("Progress", style="green", width=15)
            jobs_table.add_column("Runtime", style="dim", width=10)

            # Add job rows
            for job_info in jobs_list[:10]:  # Show first 10 jobs
                job_id_short = job_info["job_id"][:8] + "..."
                progress_bar = f"{job_info['progress_percentage']:.1f}%"
                runtime = f"{job_info.get('runtime_seconds', 0):.1f}s"

                # State emoji mapping
                state_emoji = {
                    "running": "🔄",
                    "paused": "⏸️",
                    "completed": "✅",
                    "failed": "❌",
                    "cancelled": "🚫",
                }.get(job_info["state"], "❓")

                jobs_table.add_row(
                    job_id_short,
                    job_info["name"][:23] + "..."
                    if len(job_info["name"]) > 23
                    else job_info["name"],
                    f"{state_emoji} {job_info['state']}",
                    progress_bar,
                    runtime,
                )

            if len(jobs_list) > 10:
                jobs_table.add_row("...", f"+ {len(jobs_list) - 10} more jobs", "...", "...", "...")

            return Panel(
                jobs_table, title="[bold blue]Job Monitoring[/bold blue]", border_style="blue"
            )

        except Exception as e:
            return Panel(
                f"[red]Error loading jobs: {e}[/red]",
                title="[bold blue]Job Monitoring[/bold blue]",
                border_style="red",
            )

    def stop_monitoring(self) -> None:
        """Stop the real-time monitoring."""
        self.running = False

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-10:]

        return {
            "total_samples": len(self.metrics_history),
            "recent_samples": len(recent_metrics),
            "rust_availability": all(m.rust_active for m in recent_metrics),
            "avg_cache_hit_rate": sum(m.cache_hit_rate for m in recent_metrics)
            / len(recent_metrics)
            if recent_metrics
            else 0,
            "monitoring_duration": (
                recent_metrics[-1].timestamp - self.metrics_history[0].timestamp
            ).total_seconds()
            if len(self.metrics_history) > 1
            else 0,
        }


# Global monitor instance
_global_monitor: RealTimeMonitor | None = None


async def get_real_time_monitor() -> RealTimeMonitor:
    """Get or create global real-time monitor instance."""
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = RealTimeMonitor()

    return _global_monitor


def create_progress_display() -> Progress:
    """Create a standardized progress display for Phase 2 operations."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        expand=True,
    )


def create_job_status_panel(job_status: dict[str, Any]) -> Panel:
    """Create a standardized job status panel."""
    state_colors = {
        "running": "blue",
        "paused": "yellow",
        "completed": "green",
        "failed": "red",
        "cancelled": "dim",
    }

    state_color = state_colors.get(job_status["state"], "white")

    content = f"""[{state_color}]State: {job_status["state"].upper()}[/{state_color}]
Progress: {job_status["progress_percentage"]:.1f}%
Step: {job_status.get("current_step", "N/A")} of {job_status["total_steps"]}
Runtime: {job_status.get("runtime_seconds", 0):.1f}s
Errors: {job_status.get("error_count", 0)}"""

    return Panel(
        content,
        title=f"Job {job_status['job_id'][:8]}... - {job_status['name']}",
        border_style=state_color,
    )
