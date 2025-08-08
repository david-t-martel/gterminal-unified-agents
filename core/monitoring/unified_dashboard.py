#!/usr/bin/env python3
"""Unified Monitoring Dashboard.

Provides a comprehensive monitoring dashboard that integrates:
- Real-time performance metrics
- AI operation monitoring
- Incident response tracking
- SLO compliance monitoring
- Cross-system alert correlation
- Performance insights and recommendations

Can be used standalone or through MCP integration for terminal output.
"""

import asyncio
from datetime import UTC
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Rich for beautiful terminal output
try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress
    from rich.progress import SpinnerColumn
    from rich.progress import TextColumn
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from monitoring.integrated_monitoring import IntegratedMonitoringSystem
from monitoring.integrated_monitoring import MonitoringConfig

logger = logging.getLogger(__name__)


class UnifiedMonitoringDashboard:
    """Unified dashboard for comprehensive monitoring display."""

    def __init__(self, monitoring_system: IntegratedMonitoringSystem | None = None) -> None:
        self.monitoring_system = monitoring_system or self._create_monitoring_system()
        self.console = Console() if RICH_AVAILABLE else None
        self.last_update = datetime.now(UTC)
        self.update_interval = 5  # seconds

        # Dashboard state
        self.is_streaming = False
        self.stream_data = []

    def _create_monitoring_system(self) -> IntegratedMonitoringSystem:
        """Create monitoring system with default configuration."""
        config = MonitoringConfig(
            apm_service_name="unified-dashboard",
            enable_cross_system_correlation=True,
            performance_budget_enforcement=True,
        )
        return IntegratedMonitoringSystem(config)

    async def get_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            # Get unified dashboard data
            unified_data = self.monitoring_system.get_unified_dashboard_data()

            # Get real-time metrics
            rum_dashboard = self.monitoring_system.rum.get_real_time_dashboard()
            apm_status = self.monitoring_system.apm.get_health_status()
            ai_health = self.monitoring_system.ai_metrics.get_system_health()
            slo_summary = self.monitoring_system.slo_manager.get_compliance_summary()
            incident_summary = self.monitoring_system.incident_response.get_incident_summary()

            # Compile comprehensive dashboard data
            dashboard_data = {
                "timestamp": datetime.now(UTC).isoformat(),
                "last_update": self.last_update.isoformat(),
                "system_overview": {
                    "overall_status": self._determine_overall_status(unified_data),
                    "components": {
                        "apm": {
                            "status": apm_status.get("status", "unknown"),
                            "operations": apm_status.get("total_operations", 0),
                            "error_rate": apm_status.get("error_rate_percent", 0),
                            "response_time_p95": apm_status.get("p95_response_time_ms", 0),
                        },
                        "rum": {
                            "active_users": rum_dashboard.get("active_users", 0),
                            "bounce_rate": rum_dashboard.get("bounce_rate", 0),
                            "avg_response_time": rum_dashboard.get("avg_response_time_ms", 0),
                            "errors_last_hour": rum_dashboard.get("errors_last_hour", 0),
                        },
                        "ai_metrics": {
                            "status": ai_health["status"],
                            "total_operations": ai_health["system_metrics"]["total_operations"],
                            "error_rate": ai_health["system_metrics"]["error_rate"],
                            "cost_per_hour": ai_health["system_metrics"]["cost_per_hour"],
                        },
                        "slo": {
                            "compliant_slos": slo_summary["compliant_slos"],
                            "violated_slos": slo_summary["violated_slos"],
                            "total_slos": slo_summary["slo_count"],
                            "compliance_rate": (
                                slo_summary["compliant_slos"] / max(slo_summary["slo_count"], 1)
                            )
                            * 100,
                        },
                        "incidents": {
                            "active_count": incident_summary["summary"]["active_incidents"],
                            "resolved_today": incident_summary["summary"]["resolved_today"],
                            "mttr_minutes": incident_summary["summary"]["mttr_minutes"],
                        },
                    },
                },
                "performance_metrics": {
                    "response_times": {
                        "avg_ms": rum_dashboard.get("avg_response_time_ms", 0),
                        "p95_ms": apm_status.get("p95_response_time_ms", 0),
                        "p99_ms": apm_status.get("p99_response_time_ms", 0),
                    },
                    "error_rates": {
                        "total_rate": apm_status.get("error_rate_percent", 0),
                        "ai_error_rate": ai_health["system_metrics"]["error_rate"],
                        "user_errors": rum_dashboard.get("errors_last_hour", 0),
                    },
                    "throughput": {
                        "requests_per_second": apm_status.get("requests_per_second", 0),
                        "ai_operations_per_hour": ai_health["system_metrics"]["total_operations"],
                        "active_users": rum_dashboard.get("active_users", 0),
                    },
                    "resources": {
                        "memory_usage_mb": apm_status.get("memory_usage_mb", 0),
                        "cpu_utilization": apm_status.get("cpu_percent", 0),
                        "cache_hit_rate": apm_status.get("cache_hit_rate", 0),
                    },
                },
                "alerts_and_incidents": {
                    "active_correlations": len(self.monitoring_system.alert_correlation),
                    "recent_violations": len(self.monitoring_system.performance_violations),
                    "active_incidents": incident_summary["summary"]["active_incidents"],
                    "recent_activity": incident_summary.get("recent_activity", [])[
                        :5
                    ],  # Last 5 activities
                },
                "insights": await self._get_dashboard_insights(),
            }

            self.last_update = datetime.now(UTC)
            return dashboard_data

        except Exception as e:
            logger.exception(f"Error getting dashboard data: {e}")
            return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}

    def _determine_overall_status(self, unified_data: dict[str, Any]) -> str:
        """Determine overall system status from unified data."""
        try:
            system_overview = unified_data.get("system_overview", {})

            critical_count = 0
            warning_count = 0

            for status in system_overview.values():
                if isinstance(status, dict):
                    component_status = status.get("status", "unknown")
                    if component_status in ["critical", "unhealthy"]:
                        critical_count += 1
                    elif component_status in ["warning", "degraded"]:
                        warning_count += 1

            if critical_count > 0:
                return "critical"
            if warning_count > 1:
                return "degraded"
            if warning_count > 0:
                return "warning"
            return "healthy"

        except Exception:
            return "unknown"

    async def _get_dashboard_insights(self) -> dict[str, Any]:
        """Get performance insights for the dashboard."""
        try:
            insights = {
                "apm_insights": self.monitoring_system._analyze_apm_patterns(),
                "rum_insights": self.monitoring_system._analyze_user_behavior(),
                "ai_insights": self.monitoring_system._analyze_ai_performance(),
                "slo_insights": self.monitoring_system._analyze_slo_trends(),
                "incident_insights": self.monitoring_system._analyze_incident_patterns(),
            }

            # Extract top recommendations
            all_recommendations: list[Any] = []
            for category, category_insights in insights.items():
                if isinstance(category_insights, dict) and "recommendations" in category_insights:
                    for rec in category_insights["recommendations"]:
                        all_recommendations.append(
                            {
                                "category": category.replace("_insights", ""),
                                "recommendation": rec,
                                "priority": (
                                    "high"
                                    if any(
                                        word in rec.lower()
                                        for word in ["critical", "urgent", "immediate"]
                                    )
                                    else "medium"
                                ),
                            },
                        )

            return {
                "total_recommendations": len(all_recommendations),
                "high_priority": len([r for r in all_recommendations if r["priority"] == "high"]),
                "top_recommendations": all_recommendations[:5],  # Top 5 recommendations
                "category_breakdown": insights,
            }

        except Exception as e:
            logger.exception(f"Error getting insights: {e}")
            return {"error": str(e)}

    def render_dashboard_text(self, dashboard_data: dict[str, Any]) -> str:
        """Render dashboard as formatted text for terminal output."""
        if dashboard_data.get("error"):
            return f"❌ Dashboard Error: {dashboard_data['error']}"

        timestamp = datetime.fromisoformat(dashboard_data["timestamp"])

        output: list[Any] = []
        output.append("=" * 80)
        output.append(
            f"🖥️  UNIFIED MONITORING DASHBOARD - {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        output.append("=" * 80)

        # System Overview
        overview = dashboard_data["system_overview"]
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "degraded": "🔶",
            "critical": "🚨",
            "unknown": "❓",
        }

        output.append(
            f"\n🏥 SYSTEM HEALTH: {status_emoji.get(overview['overall_status'], '❓')} {overview['overall_status'].upper()}",
        )
        output.append("-" * 40)

        components = overview["components"]
        output.append(
            f"   APM Status: {components['apm']['status']} | Operations: {components['apm']['operations']:,}"
        )
        output.append(
            f"   RUM Users: {components['rum']['active_users']:,} | Bounce Rate: {components['rum']['bounce_rate']:.1f}%",
        )
        output.append(
            f"   AI Status: {components['ai_metrics']['status']} | Operations: {components['ai_metrics']['total_operations']:,}",
        )
        output.append(
            f"   SLO Compliance: {components['slo']['compliant_slos']}/{components['slo']['total_slos']} ({components['slo']['compliance_rate']:.1f}%)",
        )
        output.append(f"   Active Incidents: {components['incidents']['active_count']}")

        # Performance Metrics
        output.append("\n📊 PERFORMANCE METRICS")
        output.append("-" * 40)
        perf = dashboard_data["performance_metrics"]

        output.append(
            f"   Response Times: Avg {perf['response_times']['avg_ms']:.0f}ms | P95 {perf['response_times']['p95_ms']:.0f}ms | P99 {perf['response_times']['p99_ms']:.0f}ms",
        )
        output.append(
            f"   Error Rates: Total {perf['error_rates']['total_rate']:.2f}% | AI {perf['error_rates']['ai_error_rate']:.2f}% | User {perf['error_rates']['user_errors']}",
        )
        output.append(
            f"   Throughput: {perf['throughput']['requests_per_second']:.1f} req/s | {perf['throughput']['ai_operations_per_hour']:,} AI ops/hr",
        )
        output.append(
            f"   Resources: {perf['resources']['memory_usage_mb']:.0f}MB RAM | {perf['resources']['cpu_utilization']:.1f}% CPU | {perf['resources']['cache_hit_rate']:.1f}% Cache",
        )

        # Alerts & Incidents
        output.append("\n🚨 ALERTS & INCIDENTS")
        output.append("-" * 40)
        alerts = dashboard_data["alerts_and_incidents"]

        output.append(f"   Alert Correlations: {alerts['active_correlations']}")
        output.append(f"   Performance Violations: {alerts['recent_violations']}")
        output.append(f"   Active Incidents: {alerts['active_incidents']}")

        if alerts.get("recent_activity"):
            output.append("   Recent Activity:")
            for activity in alerts["recent_activity"][:3]:  # Show top 3
                output.append(f"     • {activity}")

        # Insights
        output.append("\n💡 PERFORMANCE INSIGHTS")
        output.append("-" * 40)
        insights = dashboard_data["insights"]

        if not insights.get("error"):
            output.append(
                f"   Total Recommendations: {insights['total_recommendations']} ({insights['high_priority']} high priority)",
            )

            if insights.get("top_recommendations"):
                output.append("   Top Recommendations:")
                for rec in insights["top_recommendations"][:3]:  # Show top 3
                    priority_emoji = "🔴" if rec["priority"] == "high" else "🟡"
                    output.append(
                        f"     {priority_emoji} [{rec['category'].upper()}] {rec['recommendation'][:80]}..."
                    )
        else:
            output.append(f"   ❌ Insights Error: {insights['error']}")

        output.append("\n" + "=" * 80)

        return "\n".join(output)

    def render_dashboard_rich(self, dashboard_data: dict[str, Any]) -> Layout:
        """Render dashboard using Rich for beautiful terminal output."""
        if not RICH_AVAILABLE:
            return None

        if dashboard_data.get("error"):
            return Panel(f"❌ Dashboard Error: {dashboard_data['error']}", style="red")

        # Create main layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3), Layout(name="main"), Layout(name="footer", size=3)
        )

        # Header
        timestamp = datetime.fromisoformat(dashboard_data["timestamp"])
        header_text = Text(
            f"🖥️  UNIFIED MONITORING DASHBOARD - {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            style="bold cyan",
        )
        layout["header"].update(Panel(header_text, style="blue"))

        # Main content
        layout["main"].split_row(Layout(name="left"), Layout(name="right"))

        # Left panel - System Status
        layout["left"].split_column(
            Layout(name="system", size=12), Layout(name="performance", size=10)
        )

        # System Overview Table
        system_table = Table(title="🏥 System Health", show_header=True, header_style="bold blue")
        system_table.add_column("Component", style="cyan")
        system_table.add_column("Status", style="green")
        system_table.add_column("Metrics", style="yellow")

        overview = dashboard_data["system_overview"]
        components = overview["components"]

        # Status styling
        def get_status_style(status) -> None:
            if status in ["healthy", "good"]:
                return "green"
            if status in ["warning", "degraded"]:
                return "yellow"
            if status in ["critical", "unhealthy"]:
                return "red"
            return "white"

        system_table.add_row(
            "APM",
            f"[{get_status_style(components['apm']['status'])}]{components['apm']['status']}[/]",
            f"{components['apm']['operations']:,} ops, {components['apm']['error_rate']:.2f}% errors",
        )

        system_table.add_row(
            "RUM",
            "Active",
            f"{components['rum']['active_users']:,} users, {components['rum']['bounce_rate']:.1f}% bounce",
        )

        system_table.add_row(
            "AI Metrics",
            f"[{get_status_style(components['ai_metrics']['status'])}]{components['ai_metrics']['status']}[/]",
            f"{components['ai_metrics']['total_operations']:,} ops, ${components['ai_metrics']['cost_per_hour']:.2f}/hr",
        )

        system_table.add_row(
            "SLO",
            (
                f"[green]{components['slo']['compliance_rate']:.1f}%[/]"
                if components["slo"]["compliance_rate"] > 80
                else f"[red]{components['slo']['compliance_rate']:.1f}%[/]"
            ),
            f"{components['slo']['compliant_slos']}/{components['slo']['total_slos']} compliant",
        )

        system_table.add_row(
            "Incidents",
            (
                f"[red]{components['incidents']['active_count']}[/]"
                if components["incidents"]["active_count"] > 0
                else "[green]0[/]"
            ),
            f"{components['incidents']['resolved_today']} resolved today, {components['incidents']['mttr_minutes']:.0f}min MTTR",
        )

        layout["system"].update(Panel(system_table, style="blue"))

        # Performance Metrics Table
        perf_table = Table(
            title="📊 Performance Metrics", show_header=True, header_style="bold magenta"
        )
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Current", style="yellow")
        perf_table.add_column("Details", style="white")

        perf = dashboard_data["performance_metrics"]

        perf_table.add_row(
            "Response Time",
            f"{perf['response_times']['avg_ms']:.0f}ms avg",
            f"P95: {perf['response_times']['p95_ms']:.0f}ms, P99: {perf['response_times']['p99_ms']:.0f}ms",
        )

        perf_table.add_row(
            "Error Rate",
            f"{perf['error_rates']['total_rate']:.2f}%",
            f"AI: {perf['error_rates']['ai_error_rate']:.2f}%, User: {perf['error_rates']['user_errors']}",
        )

        perf_table.add_row(
            "Throughput",
            f"{perf['throughput']['requests_per_second']:.1f} req/s",
            f"{perf['throughput']['ai_operations_per_hour']:,} AI ops/hr",
        )

        perf_table.add_row(
            "Resources",
            f"{perf['resources']['memory_usage_mb']:.0f}MB RAM",
            f"{perf['resources']['cpu_utilization']:.1f}% CPU, {perf['resources']['cache_hit_rate']:.1f}% cache",
        )

        layout["performance"].update(Panel(perf_table, style="magenta"))

        # Right panel - Alerts and Insights
        layout["right"].split_column(
            Layout(name="alerts", size=10), Layout(name="insights", size=12)
        )

        # Alerts Table
        alerts_table = Table(
            title="🚨 Alerts & Incidents", show_header=True, header_style="bold red"
        )
        alerts_table.add_column("Type", style="cyan")
        alerts_table.add_column("Count", style="yellow")
        alerts_table.add_column("Status", style="white")

        alerts = dashboard_data["alerts_and_incidents"]

        alerts_table.add_row(
            "Alert Correlations",
            str(alerts["active_correlations"]),
            "Active" if alerts["active_correlations"] > 0 else "None",
        )
        alerts_table.add_row(
            "Performance Violations",
            str(alerts["recent_violations"]),
            "Recent" if alerts["recent_violations"] > 0 else "None",
        )
        alerts_table.add_row(
            "Active Incidents",
            str(alerts["active_incidents"]),
            "[red]Critical[/]" if alerts["active_incidents"] > 0 else "[green]All Clear[/]",
        )

        layout["alerts"].update(Panel(alerts_table, style="red"))

        # Insights Panel
        insights = dashboard_data["insights"]
        if not insights.get("error"):
            insights_text: list[Any] = []
            insights_text.append(
                f"💡 {insights['total_recommendations']} recommendations ({insights['high_priority']} high priority)",
            )
            insights_text.append("")

            if insights.get("top_recommendations"):
                insights_text.append("🔥 Top Recommendations:")
                for i, rec in enumerate(insights["top_recommendations"][:4], 1):
                    priority_color = "red" if rec["priority"] == "high" else "yellow"
                    insights_text.append(
                        f"  {i}. [{priority_color}][{rec['category'].upper()}][/] {rec['recommendation'][:60]}...",
                    )
        else:
            insights_text = [f"❌ Insights Error: {insights['error']}"]

        layout["insights"].update(
            Panel("\n".join(insights_text), title="💡 Performance Insights", style="green")
        )

        # Footer
        footer_text = Text(
            f"Last Update: {dashboard_data['last_update']} | Update Interval: {self.update_interval}s | Status: {overview['overall_status'].upper()}",
            style="dim",
        )
        layout["footer"].update(Panel(footer_text, style="dim"))

        return layout

    async def display_dashboard_once(self, use_rich: bool = True) -> str:
        """Display dashboard once and return the output."""
        dashboard_data = await self.get_dashboard_data()

        if use_rich and RICH_AVAILABLE and self.console:
            layout = self.render_dashboard_rich(dashboard_data)
            if layout:
                self.console.print(layout)
                return "Rich dashboard displayed"

        # Fallback to text output
        return self.render_dashboard_text(dashboard_data)

    async def stream_dashboard(self, duration_seconds: int = 300, use_rich: bool = True) -> None:
        """Stream dashboard updates for the specified duration."""
        self.is_streaming = True
        start_time = datetime.now(UTC)

        try:
            if use_rich and RICH_AVAILABLE and self.console:
                # Use Rich Live for updating display
                with Live(auto_refresh=False) as live:
                    while (
                        self.is_streaming
                        and (datetime.now(UTC) - start_time).total_seconds() < duration_seconds
                    ):
                        dashboard_data = await self.get_dashboard_data()
                        layout = self.render_dashboard_rich(dashboard_data)
                        if layout:
                            live.update(layout, refresh=True)

                        await asyncio.sleep(self.update_interval)
            else:
                # Fallback to periodic text updates
                while (
                    self.is_streaming
                    and (datetime.now(UTC) - start_time).total_seconds() < duration_seconds
                ):
                    os.system("clear" if os.name == "posix" else "cls")  # Clear screen
                    await self.display_dashboard_once(use_rich=False)
                    await asyncio.sleep(self.update_interval)

        except KeyboardInterrupt:
            self.is_streaming = False
            logger.info("Dashboard streaming stopped by user")
        except Exception as e:
            logger.exception(f"Error streaming dashboard: {e}")
        finally:
            self.is_streaming = False

    def stop_streaming(self) -> None:
        """Stop dashboard streaming."""
        self.is_streaming = False

    async def get_terminal_summary(self) -> dict[str, Any]:
        """Get a summary suitable for terminal display."""
        dashboard_data = await self.get_dashboard_data()

        if dashboard_data.get("error"):
            return {
                "status": "error",
                "message": f"Dashboard Error: {dashboard_data['error']}",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        overview = dashboard_data["system_overview"]
        perf = dashboard_data["performance_metrics"]
        alerts = dashboard_data["alerts_and_incidents"]

        return {
            "status": overview["overall_status"],
            "summary": {
                "health": f"{overview['overall_status'].upper()}",
                "active_users": overview["components"]["rum"]["active_users"],
                "response_time_ms": perf["response_times"]["avg_ms"],
                "error_rate": perf["error_rates"]["total_rate"],
                "active_incidents": alerts["active_incidents"],
                "slo_compliance": overview["components"]["slo"]["compliance_rate"],
            },
            "details": {
                "apm_operations": overview["components"]["apm"]["operations"],
                "ai_operations_per_hour": perf["throughput"]["ai_operations_per_hour"],
                "memory_usage_mb": perf["resources"]["memory_usage_mb"],
                "cache_hit_rate": perf["resources"]["cache_hit_rate"],
                "alert_correlations": alerts["active_correlations"],
            },
            "timestamp": dashboard_data["timestamp"],
        }

    async def close(self) -> None:
        """Cleanup dashboard resources."""
        self.is_streaming = False
        if self.monitoring_system:
            await self.monitoring_system.close()


async def main() -> None:
    """Main entry point for standalone dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Unified Monitoring Dashboard")
    parser.add_argument("--mode", choices=["once", "stream"], default="once", help="Display mode")
    parser.add_argument("--duration", type=int, default=300, help="Stream duration in seconds")
    parser.add_argument("--no-rich", action="store_true", help="Disable Rich formatting")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    dashboard = UnifiedMonitoringDashboard()
    dashboard.update_interval = args.interval

    try:
        if args.mode == "once":
            await dashboard.display_dashboard_once(use_rich=not args.no_rich)
        else:
            await dashboard.stream_dashboard(
                duration_seconds=args.duration, use_rich=not args.no_rich
            )
    except KeyboardInterrupt:
        pass
    finally:
        await dashboard.close()


if __name__ == "__main__":
    asyncio.run(main())
