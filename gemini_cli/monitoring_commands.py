from typing import Any

#!/usr/bin/env python3
"""
CLI Commands for Monitoring System

Provides command-line interface for accessing monitoring capabilities:
- Real-time dashboard display
- System health checks
- Performance metrics queries
- Incident management
- Alert correlation analysis
- Streaming monitoring data
"""

import asyncio
from datetime import UTC
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.integrated_monitoring import IntegratedMonitoringSystem
from monitoring.integrated_monitoring import MonitoringConfig
from monitoring.unified_dashboard import UnifiedMonitoringDashboard

logger = logging.getLogger(__name__)


@click.group(name="monitoring")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def monitoring_cli(verbose: bool) -> None:
    """Monitoring system commands for real-time performance tracking."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@monitoring_cli.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json", "rich"]),
    default="rich",
    help="Output format",
)
@click.option("--no-color", is_flag=True, help="Disable colored output")
def status(format: str, no_color: bool) -> None:
    """Get current system health status."""

    async def _get_status() -> None:
        dashboard = UnifiedMonitoringDashboard()
        try:
            if format == "json":
                summary = await dashboard.get_terminal_summary()
                click.echo(json.dumps(summary, indent=2))
            elif format == "text" or no_color:
                await dashboard.display_dashboard_once(use_rich=False)
            else:  # rich format
                await dashboard.display_dashboard_once(use_rich=True)
        finally:
            await dashboard.close()

    asyncio.run(_get_status())


@monitoring_cli.command()
@click.option("--duration", "-d", type=int, default=300, help="Streaming duration in seconds")
@click.option("--interval", "-i", type=int, default=5, help="Update interval in seconds")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "rich"]),
    default="rich",
    help="Output format",
)
def stream(duration: int, interval: int, format: str) -> None:
    """Stream real-time monitoring data."""

    async def _stream() -> None:
        dashboard = UnifiedMonitoringDashboard()
        dashboard.update_interval = interval
        try:
            click.echo(
                f"Starting monitoring stream for {duration} seconds (update every {interval}s)"
            )
            click.echo("Press Ctrl+C to stop...")
            await dashboard.stream_dashboard(duration_seconds=duration, use_rich=(format == "rich"))
        except KeyboardInterrupt:
            click.echo("\nStreaming stopped by user")
        finally:
            await dashboard.close()

    asyncio.run(_stream())


@monitoring_cli.command()
@click.option("--output", "-o", type=click.Path(), help="Save metrics to file")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "csv"]),
    default="json",
    help="Output format",
)
def metrics(output: str, format: str) -> None:
    """Get current performance metrics."""

    async def _get_metrics() -> None:
        config = MonitoringConfig()
        monitoring_system = IntegratedMonitoringSystem(config)

        try:
            # Get comprehensive metrics
            dashboard_data = monitoring_system.get_unified_dashboard_data()

            if format == "json":
                metrics_data = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "system_overview": dashboard_data["system_overview"],
                    "performance_insights": dashboard_data["performance_insights"],
                    "alert_correlation": dashboard_data["alert_correlation"],
                }
            else:  # CSV format
                # Flatten data for CSV
                csv_data: list[Any] = []
                timestamp = datetime.now(UTC).isoformat()

                # Add system overview metrics
                for component, data in dashboard_data["system_overview"].items():
                    if isinstance(data, dict):
                        for key, value in data.items():
                            csv_data.append(
                                {
                                    "timestamp": timestamp,
                                    "category": "system",
                                    "component": component,
                                    "metric": key,
                                    "value": value,
                                },
                            )

                metrics_data = csv_data

            if output:
                if format == "json":
                    with open(output, "w") as f:
                        json.dump(metrics_data, f, indent=2)
                else:  # CSV
                    import csv

                    with open(output, "w", newline="") as f:
                        if metrics_data:
                            writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
                            writer.writeheader()
                            writer.writerows(metrics_data)

                click.echo(f"Metrics saved to {output}")
            elif format == "json":
                click.echo(json.dumps(metrics_data, indent=2))
            else:
                # Print CSV to stdout
                import csv
                import io

                output_buffer = io.StringIO()
                if metrics_data:
                    writer = csv.DictWriter(output_buffer, fieldnames=metrics_data[0].keys())
                    writer.writeheader()
                    writer.writerows(metrics_data)
                    click.echo(output_buffer.getvalue())

        finally:
            await monitoring_system.close()

    asyncio.run(_get_metrics())


@monitoring_cli.command()
@click.option("--active-only", is_flag=True, help="Show only active incidents")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def incidents(active_only: bool, format: str) -> None:
    """Get incident status and management."""

    async def _get_incidents() -> None:
        config = MonitoringConfig()
        monitoring_system = IntegratedMonitoringSystem(config)

        try:
            incident_summary = monitoring_system.incident_response.get_incident_summary()

            if format == "json":
                click.echo(json.dumps(incident_summary, indent=2, default=str))
            else:
                # Text format
                summary = incident_summary["summary"]
                click.echo("🚨 Incident Status")
                click.echo("=" * 40)
                click.echo(f"Active Incidents: {summary['active_incidents']}")
                click.echo(f"Resolved Today: {summary['resolved_today']}")
                click.echo(f"Mean Time to Resolution: {summary['mttr_minutes']:.0f} minutes")

                if not active_only or summary["active_incidents"] > 0:
                    active_incidents = incident_summary.get("active_incidents", [])
                    if active_incidents:
                        click.echo("\n📋 Active Incidents:")
                        for incident in active_incidents:
                            click.echo(f"  • {incident['id']}: {incident['title']}")
                            click.echo(
                                f"    Severity: {incident['severity']} | Status: {incident['status']}"
                            )
                            click.echo(f"    Created: {incident['created_at']}")

                    recent_activity = incident_summary.get("recent_activity", [])
                    if recent_activity:
                        click.echo("\n📈 Recent Activity:")
                        for activity in recent_activity[:5]:
                            click.echo(f"  • {activity}")

        finally:
            await monitoring_system.close()

    asyncio.run(_get_incidents())


@monitoring_cli.command()
@click.argument("title")
@click.argument("description")
@click.option(
    "--severity",
    type=click.Choice(["low", "medium", "high", "critical"]),
    default="medium",
    help="Incident severity",
)
@click.option("--source", default="cli", help="Incident source")
def create_incident(title: str, description: str, severity: str, source: str) -> None:
    """Create a new incident."""

    async def _create_incident() -> None:
        config = MonitoringConfig()
        monitoring_system = IntegratedMonitoringSystem(config)

        try:
            from monitoring.incident_response import IncidentSeverity

            incident_severity = IncidentSeverity(severity.upper())

            incident_id = await monitoring_system.incident_response.create_incident(
                title=title,
                description=description,
                severity=incident_severity,
                source=source,
            )

            click.echo(f"✅ Created incident {incident_id}: {title}")
            click.echo(f"   Severity: {severity}")
            click.echo(f"   Source: {source}")

        finally:
            await monitoring_system.close()

    asyncio.run(_create_incident())


@monitoring_cli.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def slo(format: str) -> None:
    """Get Service Level Objective status."""

    async def _get_slo() -> None:
        config = MonitoringConfig()
        monitoring_system = IntegratedMonitoringSystem(config)

        try:
            slo_status = monitoring_system.slo_manager.get_slo_status()
            compliance_summary = monitoring_system.slo_manager.get_compliance_summary()
            alert_summary = monitoring_system.slo_manager.get_alert_summary()

            slo_data = {
                "compliance_summary": compliance_summary,
                "slo_details": slo_status,
                "alerts": alert_summary,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            if format == "json":
                click.echo(json.dumps(slo_data, indent=2, default=str))
            else:
                # Text format
                click.echo("📊 SLO Status")
                click.echo("=" * 40)
                click.echo(f"Compliant SLOs: {compliance_summary['compliant_slos']}")
                click.echo(f"Violated SLOs: {compliance_summary['violated_slos']}")
                click.echo(f"Total SLOs: {compliance_summary['slo_count']}")

                if compliance_summary["slo_count"] > 0:
                    compliance_rate = (
                        compliance_summary["compliant_slos"] / compliance_summary["slo_count"]
                    ) * 100
                    click.echo(f"Compliance Rate: {compliance_rate:.1f}%")

                # Show alert summary
                total_alerts = alert_summary["active_alerts"]["total"]
                if total_alerts > 0:
                    click.echo(f"\n🚨 Active Alerts: {total_alerts}")
                    for severity, count in alert_summary["active_alerts"].items():
                        if severity != "total" and count > 0:
                            click.echo(f"  {severity.capitalize()}: {count}")

        finally:
            await monitoring_system.close()

    asyncio.run(_get_slo())


@monitoring_cli.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def insights(format: str) -> None:
    """Get AI-powered performance insights."""

    async def _get_insights() -> None:
        config = MonitoringConfig()
        monitoring_system = IntegratedMonitoringSystem(config)

        try:
            # Get insights from integrated analysis
            insights_data = {
                "timestamp": datetime.now(UTC).isoformat(),
                "apm_insights": monitoring_system._analyze_apm_patterns(),
                "rum_insights": monitoring_system._analyze_user_behavior(),
                "ai_insights": monitoring_system._analyze_ai_performance(),
                "slo_insights": monitoring_system._analyze_slo_trends(),
                "incident_insights": monitoring_system._analyze_incident_patterns(),
            }

            if format == "json":
                click.echo(json.dumps(insights_data, indent=2, default=str))
            else:
                # Text format
                click.echo("💡 Performance Insights")
                click.echo("=" * 40)

                # Aggregate recommendations
                all_recommendations: list[Any] = []
                for category, category_insights in insights_data.items():
                    if category == "timestamp":
                        continue

                    category_name = category.replace("_insights", "").upper()
                    click.echo(f"\n{category_name}:")

                    if isinstance(category_insights, dict):
                        if "error" in category_insights:
                            click.echo(f"  ❌ Error: {category_insights['error']}")
                        else:
                            # Show key metrics
                            for key, value in category_insights.items():
                                if key != "recommendations" and not key.startswith("_"):
                                    click.echo(f"  {key}: {value}")

                            # Show recommendations
                            recommendations = category_insights.get("recommendations", [])
                            if recommendations:
                                click.echo("  Recommendations:")
                                for rec in recommendations:
                                    all_recommendations.append((category_name, rec))
                                    click.echo(f"    • {rec}")

                # Summary
                if all_recommendations:
                    click.echo(f"\n📋 Summary: {len(all_recommendations)} total recommendations")

        finally:
            await monitoring_system.close()

    asyncio.run(_get_insights())


@monitoring_cli.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def alerts(format: str) -> None:
    """Get alert correlation analysis."""

    async def _get_alerts() -> None:
        config = MonitoringConfig()
        monitoring_system = IntegratedMonitoringSystem(config)

        try:
            correlation_data = {
                "timestamp": datetime.now(UTC).isoformat(),
                "active_correlations": dict(monitoring_system.alert_correlation),
                "recent_violations": monitoring_system.performance_violations[-10:],  # Last 10
                "correlation_patterns": {},
            }

            # Analyze patterns
            if monitoring_system.alert_correlation:
                patterns: dict[str, Any] = {}
                for alerts in monitoring_system.alert_correlation.values():
                    for alert_type, alert_list in alerts.items():
                        if alert_list:
                            patterns[alert_type] = patterns.get(alert_type, 0) + len(alert_list)
                correlation_data["correlation_patterns"] = patterns

            if format == "json":
                click.echo(json.dumps(correlation_data, indent=2, default=str))
            else:
                # Text format
                click.echo("🔗 Alert Correlation")
                click.echo("=" * 40)
                click.echo(f"Active Correlations: {len(correlation_data['active_correlations'])}")
                click.echo(f"Recent Violations: {len(correlation_data['recent_violations'])}")

                if correlation_data["correlation_patterns"]:
                    click.echo("\n📊 Alert Patterns:")
                    for pattern, count in correlation_data["correlation_patterns"].items():
                        click.echo(f"  {pattern}: {count} alerts")

                if correlation_data["recent_violations"]:
                    click.echo("\n⚠️ Recent Violations:")
                    for violation in correlation_data["recent_violations"][-5:]:  # Last 5
                        if isinstance(violation, dict):
                            system = violation.get("system", "unknown")
                            issue = violation.get("issue", "unknown issue")
                            click.echo(f"  • {system}: {issue}")

        finally:
            await monitoring_system.close()

    asyncio.run(_get_alerts())


@monitoring_cli.command()
@click.option("--duration", "-d", type=int, default=60, help="Collection duration in seconds")
@click.option("--output", "-o", type=click.Path(), help="Save stream data to file")
def collect_stream(duration: int, output: str) -> None:
    """Collect streaming monitoring data for analysis."""

    async def _collect_stream() -> None:
        config = MonitoringConfig()
        monitoring_system = IntegratedMonitoringSystem(config)
        dashboard = UnifiedMonitoringDashboard(monitoring_system)

        try:
            click.echo(f"Collecting monitoring data for {duration} seconds...")

            monitoring_data: list[Any] = []
            start_time = datetime.now(UTC)

            # Collect data points every 5 seconds
            for i in range(0, duration, 5):
                current_time = datetime.now(UTC)

                # Get current dashboard data
                dashboard_data = await dashboard.get_dashboard_data()

                data_point = {
                    "timestamp": current_time.isoformat(),
                    "elapsed_seconds": i,
                    "dashboard_data": dashboard_data,
                }

                monitoring_data.append(data_point)
                click.echo(
                    f"  Collected data point {len(monitoring_data)} at {current_time.strftime('%H:%M:%S')}"
                )

                # Wait 5 seconds before next collection
                if i < duration - 5:
                    await asyncio.sleep(5)

            stream_summary = {
                "collection_info": {
                    "duration_seconds": duration,
                    "data_points": len(monitoring_data),
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now(UTC).isoformat(),
                },
                "monitoring_stream": monitoring_data,
            }

            if output:
                with open(output, "w") as f:
                    json.dump(stream_summary, f, indent=2, default=str)
                click.echo(f"✅ Stream data saved to {output}")
            else:
                click.echo(json.dumps(stream_summary, indent=2, default=str))

        finally:
            await dashboard.close()
            await monitoring_system.close()

    asyncio.run(_collect_stream())


if __name__ == "__main__":
    monitoring_cli()
