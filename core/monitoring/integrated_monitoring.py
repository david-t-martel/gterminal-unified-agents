"""Integrated Monitoring System.

This module provides a unified interface for all monitoring components,
orchestrating APM, RUM, AI metrics, SLOs, and incident response.
"""

import asyncio
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
import logging
from typing import Any

from .ai_metrics import AIPerformanceMonitor
from .ai_metrics import OperationType
from .ai_metrics import QualityMetricType
from .apm import EnhancedAPMSystem
from .incident_response import IncidentResponseSystem
from .incident_response import IncidentSeverity
from .rum import RealUserMonitoring
from .slo_manager import AlertSeverity
from .slo_manager import SLOManager

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for the integrated monitoring system."""

    # APM Configuration
    apm_service_name: str = "fullstack-agent"
    apm_jaeger_endpoint: str | None = None
    apm_prometheus_port: int = 8000

    # RUM Configuration
    rum_retention_days: int = 30

    # AI Metrics Configuration
    ai_metrics_retention_days: int = 7

    # SLO Configuration
    slo_prometheus_client: Any | None = None

    # Incident Response Configuration
    incident_auto_remediation: bool = True
    incident_max_concurrent_actions: int = 5

    # Integration settings
    enable_cross_system_correlation: bool = True
    alert_correlation_window_minutes: int = 5
    performance_budget_enforcement: bool = True


class IntegratedMonitoringSystem:
    """Unified monitoring system that orchestrates all monitoring components."""

    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self.config = config or MonitoringConfig()
        self.start_time = datetime.now(UTC)

        # Initialize monitoring components
        self.apm = EnhancedAPMSystem(
            service_name=self.config.apm_service_name,
            jaeger_endpoint=self.config.apm_jaeger_endpoint,
            prometheus_port=self.config.apm_prometheus_port,
        )

        self.rum = RealUserMonitoring(retention_days=self.config.rum_retention_days)

        self.ai_metrics = AIPerformanceMonitor(retention_days=self.config.ai_metrics_retention_days)

        self.slo_manager = SLOManager(prometheus_client=self.config.slo_prometheus_client)

        self.incident_response = IncidentResponseSystem()

        # Cross-system correlation
        self.alert_correlation: dict[str, list[dict[str, Any]]] = {}
        self.performance_violations: list[dict[str, Any]] = []

        # Callbacks and integrations
        self._setup_integrations()

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []
        self._start_integrated_monitoring()

    def _setup_integrations(self) -> None:
        """Set up integrations between monitoring components."""
        # SLO alerts trigger incident creation
        self.slo_manager.add_alert_callback(self._handle_slo_alert)

        # Incident callbacks
        self.incident_response.add_incident_callback(self._handle_incident_event)
        self.incident_response.add_remediation_callback(self._handle_remediation_event)

    def _start_integrated_monitoring(self) -> None:
        """Start integrated monitoring tasks."""
        self._background_tasks = [
            asyncio.create_task(self._correlate_alerts()),
            asyncio.create_task(self._monitor_performance_budgets()),
            asyncio.create_task(self._generate_insights()),
        ]

    async def _handle_slo_alert(self, alert) -> None:
        """Handle SLO burn rate alerts by creating incidents."""
        try:
            # Map SLO severity to incident severity
            severity_mapping = {
                AlertSeverity.INFO: IncidentSeverity.LOW,
                AlertSeverity.WARNING: IncidentSeverity.MEDIUM,
                AlertSeverity.CRITICAL: IncidentSeverity.HIGH,
                AlertSeverity.EMERGENCY: IncidentSeverity.CRITICAL,
            }

            incident_severity = severity_mapping.get(alert.severity, IncidentSeverity.MEDIUM)

            # Create incident
            incident_id = await self.incident_response.create_incident(
                title=f"SLO Alert: {alert.slo_name}",
                description=alert.message,
                severity=incident_severity,
                source="slo_monitoring",
                alerts=[asdict(alert)],
            )

            logger.info(f"Created incident {incident_id} from SLO alert {alert.slo_name}")

        except Exception as e:
            logger.exception(f"Error handling SLO alert: {e}")

    async def _handle_incident_event(self, event_type: str, incident) -> None:
        """Handle incident events."""
        try:
            logger.info(f"Incident event: {event_type} - {incident.incident_id}")

            # Record custom metrics
            if hasattr(self.apm, "record_custom_metric"):
                if event_type == "incident_created":
                    self.apm.record_custom_metric("incidents_created_total", 1.0)
                elif event_type == "incident_resolved":
                    resolution_time = incident.time_to_resolve_minutes or 0
                    self.apm.record_custom_metric(
                        "incident_resolution_time_minutes", resolution_time
                    )

        except Exception as e:
            logger.exception(f"Error handling incident event: {e}")

    async def _handle_remediation_event(self, event_type: str, execution) -> None:
        """Handle remediation events."""
        try:
            logger.info(f"Remediation event: {event_type} - {execution.action_name}")

            # Record remediation metrics
            if hasattr(self.apm, "record_custom_metric") and event_type == "remediation_completed":
                success = 1.0 if execution.status.value == "success" else 0.0
                self.apm.record_custom_metric("remediation_success_rate", success)
                self.apm.record_custom_metric(
                    "remediation_duration_seconds", execution.duration_seconds
                )

        except Exception as e:
            logger.exception(f"Error handling remediation event: {e}")

    async def _correlate_alerts(self) -> None:
        """Correlate alerts across different monitoring systems."""
        while True:
            try:
                if self.config.enable_cross_system_correlation:
                    await self._perform_alert_correlation()

                await asyncio.sleep(60)  # Correlate every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error correlating alerts: {e}")
                await asyncio.sleep(60)

    async def _perform_alert_correlation(self) -> None:
        """Perform cross-system alert correlation."""
        try:
            # Get recent alerts from different systems
            datetime.now(UTC) - timedelta(minutes=self.config.alert_correlation_window_minutes)

            # Collect alerts from different sources
            correlated_alerts = {
                "apm_anomalies": [],
                "slo_violations": [],
                "performance_degradations": [],
                "user_experience_issues": [],
            }

            # Check for APM anomalies
            apm_status = self.apm.get_health_status()
            if apm_status.get("active_anomalies", 0) > 0:
                correlated_alerts["apm_anomalies"].append(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "source": "apm",
                        "anomaly_count": apm_status["active_anomalies"],
                    },
                )

            # Check for SLO violations
            slo_alert_summary = self.slo_manager.get_alert_summary()
            if slo_alert_summary["active_alerts"]["total"] > 0:
                correlated_alerts["slo_violations"].append(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "source": "slo",
                        "active_alerts": slo_alert_summary["active_alerts"],
                    },
                )

            # Check for RUM performance issues
            rum_dashboard = self.rum.get_real_time_dashboard()
            if rum_dashboard["errors_last_hour"] > 10:  # Threshold
                correlated_alerts["user_experience_issues"].append(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "source": "rum",
                        "errors_last_hour": rum_dashboard["errors_last_hour"],
                        "bounce_rate": rum_dashboard["bounce_rate"],
                    },
                )

            # Check for AI performance issues
            ai_health = self.ai_metrics.get_system_health()
            if ai_health["status"] != "healthy":
                correlated_alerts["performance_degradations"].append(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "source": "ai_metrics",
                        "status": ai_health["status"],
                        "issues": ai_health["issues"],
                    },
                )

            # Store correlated alerts
            correlation_id = f"correlation_{int(datetime.now(UTC).timestamp())}"
            if any(alerts for alerts in correlated_alerts.values()):
                self.alert_correlation[correlation_id] = correlated_alerts

                # Check if we should create a meta-incident
                if self._should_create_meta_incident(correlated_alerts):
                    await self._create_meta_incident(correlation_id, correlated_alerts)

        except Exception as e:
            logger.exception(f"Error performing alert correlation: {e}")

    def _should_create_meta_incident(self, correlated_alerts: dict[str, list]) -> bool:
        """Determine if correlated alerts warrant a meta-incident."""
        # Count different types of issues
        issue_types = sum(1 for alerts in correlated_alerts.values() if alerts)

        # Create meta-incident if multiple systems are affected
        return issue_types >= 2

    async def _create_meta_incident(
        self, correlation_id: str, correlated_alerts: dict[str, list]
    ) -> None:
        """Create a meta-incident from correlated alerts."""
        try:
            # Determine severity based on alert types
            severity = IncidentSeverity.MEDIUM
            if correlated_alerts["slo_violations"] and correlated_alerts["user_experience_issues"]:
                severity = IncidentSeverity.HIGH

            # Create description
            issue_summary: list[Any] = []
            for alert_type, alerts in correlated_alerts.items():
                if alerts:
                    issue_summary.append(f"{alert_type}: {len(alerts)} issues")

            description = f"Multiple monitoring systems detected issues: {', '.join(issue_summary)}"

            # Create incident
            incident_id = await self.incident_response.create_incident(
                title="Cross-System Performance Degradation",
                description=description,
                severity=severity,
                source="alert_correlation",
                alerts=correlated_alerts,
            )

            logger.warning(f"Created meta-incident {incident_id} from correlation {correlation_id}")

        except Exception as e:
            logger.exception(f"Error creating meta-incident: {e}")

    async def _monitor_performance_budgets(self) -> None:
        """Monitor performance budgets across all systems."""
        while True:
            try:
                if self.config.performance_budget_enforcement:
                    await self._check_performance_budgets()

                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error monitoring performance budgets: {e}")
                await asyncio.sleep(300)

    async def _check_performance_budgets(self) -> None:
        """Check performance budgets and take action if violated."""
        try:
            violations: list[Any] = []

            # Check APM performance budgets
            apm_status = self.apm.get_health_status()
            if apm_status["status"] == "unhealthy":
                violations.append(
                    {
                        "system": "apm",
                        "issue": "System unhealthy",
                        "details": apm_status,
                    },
                )

            # Check AI performance budgets
            ai_health = self.ai_metrics.get_system_health()
            if ai_health["status"] in ["degraded", "unhealthy"]:
                violations.append(
                    {
                        "system": "ai_metrics",
                        "issue": f"System {ai_health['status']}",
                        "details": ai_health,
                    },
                )

            # Check SLO compliance
            slo_status = self.slo_manager.get_slo_status()
            for slo_name, slo_data in slo_status.items():
                if isinstance(slo_data, dict) and slo_data.get("status") == "critical":
                    violations.append(
                        {
                            "system": "slo",
                            "slo_name": slo_name,
                            "issue": "SLO in critical state",
                            "details": slo_data,
                        },
                    )

            # Store violations
            if violations:
                self.performance_violations.extend(violations)

                # Keep only recent violations (last 24 hours)
                cutoff = datetime.now(UTC) - timedelta(hours=24)
                self.performance_violations = [
                    v
                    for v in self.performance_violations
                    if datetime.fromisoformat(v.get("timestamp", datetime.now(UTC).isoformat()))
                    >= cutoff
                ]

                logger.warning(f"Performance budget violations detected: {len(violations)} issues")

        except Exception as e:
            logger.exception(f"Error checking performance budgets: {e}")

    async def _generate_insights(self) -> None:
        """Generate performance insights and recommendations."""
        while True:
            try:
                await self._perform_insight_generation()
                await asyncio.sleep(1800)  # Generate every 30 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error generating insights: {e}")
                await asyncio.sleep(1800)

    async def _perform_insight_generation(self) -> None:
        """Analyze data across systems to generate insights."""
        try:
            insights = {
                "timestamp": datetime.now(UTC).isoformat(),
                "apm_insights": self._analyze_apm_patterns(),
                "rum_insights": self._analyze_user_behavior(),
                "ai_insights": self._analyze_ai_performance(),
                "slo_insights": self._analyze_slo_trends(),
                "incident_insights": self._analyze_incident_patterns(),
            }

            # Log significant insights
            for category, category_insights in insights.items():
                if isinstance(category_insights, dict) and category_insights.get("recommendations"):
                    logger.info(
                        f"Performance insights ({category}): {len(category_insights['recommendations'])} recommendations",
                    )

        except Exception as e:
            logger.exception(f"Error performing insight generation: {e}")

    def _analyze_apm_patterns(self) -> dict[str, Any]:
        """Analyze APM data for patterns and insights."""
        try:
            # Get recent APM data
            apm_status = self.apm.get_health_status()

            insights = {
                "system_health": apm_status["status"],
                "total_operations": apm_status.get("total_operations", 0),
                "error_rate": apm_status.get("error_rate_percent", 0),
                "recommendations": [],
            }

            # Generate recommendations based on patterns
            if apm_status.get("error_rate_percent", 0) > 5:
                insights["recommendations"].append(
                    "High error rate detected - investigate error patterns and implement error handling improvements",
                )

            if apm_status.get("active_operations", 0) > 100:
                insights["recommendations"].append(
                    "High number of active operations - consider implementing request queuing or rate limiting",
                )

            return insights

        except Exception as e:
            logger.exception(f"Error analyzing APM patterns: {e}")
            return {"error": str(e)}

    def _analyze_user_behavior(self) -> dict[str, Any]:
        """Analyze RUM data for user behavior insights."""
        try:
            # Get RUM dashboard data
            rum_data = self.rum.get_real_time_dashboard()

            insights = {
                "active_users": rum_data.get("active_users", 0),
                "bounce_rate": rum_data.get("bounce_rate", 0),
                "avg_response_time": rum_data.get("avg_response_time_ms", 0),
                "recommendations": [],
            }

            # Generate recommendations
            if rum_data.get("bounce_rate", 0) > 60:
                insights["recommendations"].append(
                    "High bounce rate - improve page load performance and user experience",
                )

            if rum_data.get("avg_response_time_ms", 0) > 3000:
                insights["recommendations"].append(
                    "Slow response times affecting user experience - optimize AI inference pipeline",
                )

            return insights

        except Exception as e:
            logger.exception(f"Error analyzing user behavior: {e}")
            return {"error": str(e)}

    def _analyze_ai_performance(self) -> dict[str, Any]:
        """Analyze AI metrics for performance insights."""
        try:
            # Get AI system health
            ai_health = self.ai_metrics.get_system_health()

            insights = {
                "system_status": ai_health["status"],
                "total_operations": ai_health["system_metrics"]["total_operations"],
                "error_rate": ai_health["system_metrics"]["error_rate"],
                "cost_per_hour": ai_health["system_metrics"]["cost_per_hour"],
                "recommendations": [],
            }

            # Generate recommendations
            if ai_health["system_metrics"]["error_rate"] > 3:
                insights["recommendations"].append(
                    "AI error rate is high - review model configurations and implement fallback mechanisms",
                )

            if ai_health["system_metrics"]["cost_per_hour"] > 20:
                insights["recommendations"].append(
                    "AI costs are high - consider model optimization, caching, or usage optimization",
                )

            return insights

        except Exception as e:
            logger.exception(f"Error analyzing AI performance: {e}")
            return {"error": str(e)}

    def _analyze_slo_trends(self) -> dict[str, Any]:
        """Analyze SLO compliance trends."""
        try:
            # Get SLO compliance summary
            slo_summary = self.slo_manager.get_compliance_summary()

            insights = {
                "compliant_slos": slo_summary["compliant_slos"],
                "violated_slos": slo_summary["violated_slos"],
                "total_slos": slo_summary["slo_count"],
                "recommendations": [],
            }

            # Generate recommendations
            compliance_rate = (
                slo_summary["compliant_slos"] / slo_summary["slo_count"]
                if slo_summary["slo_count"] > 0
                else 0
            )

            if compliance_rate < 0.8:
                insights["recommendations"].append(
                    "SLO compliance is low - review error budgets and implement performance improvements",
                )

            return insights

        except Exception as e:
            logger.exception(f"Error analyzing SLO trends: {e}")
            return {"error": str(e)}

    def _analyze_incident_patterns(self) -> dict[str, Any]:
        """Analyze incident response patterns."""
        try:
            # Get incident summary
            incident_summary = self.incident_response.get_incident_summary()

            insights = {
                "active_incidents": incident_summary["summary"]["active_incidents"],
                "resolved_today": incident_summary["summary"]["resolved_today"],
                "mttr_minutes": incident_summary["summary"]["mttr_minutes"],
                "recommendations": [],
            }

            # Generate recommendations
            if incident_summary["summary"]["mttr_minutes"] > 60:
                insights["recommendations"].append(
                    "Mean Time To Resolution is high - review incident response procedures and automation",
                )

            if incident_summary["summary"]["active_incidents"] > 5:
                insights["recommendations"].append(
                    "High number of active incidents - consider implementing proactive monitoring and prevention",
                )

            return insights

        except Exception as e:
            logger.exception(f"Error analyzing incident patterns: {e}")
            return {"error": str(e)}

    # Public API methods

    async def start_user_session(
        self,
        session_id: str,
        user_id: str | None = None,
        user_agent: str = "",
        device_info: dict[str, Any] | None = None,
        location_info: dict[str, str] | None = None,
    ):
        """Start tracking a user session."""
        return self.rum.start_session(session_id, user_id, user_agent, device_info, location_info)

    async def record_ai_operation(
        self,
        session_id: str,
        model_name: str,
        operation_type: OperationType,
        input_tokens: int = 0,
        context_size_tokens: int = 0,
    ) -> str:
        """Start tracking an AI operation."""
        # Record in both APM and AI metrics
        async with self.apm.trace_ai_operation(model_name, operation_type.value, input_tokens) as (
            ai_metrics,
            span,
        ):
            span.get_span_context().span_id

        return self.ai_metrics.start_operation(
            session_id,
            model_name,
            operation_type,
            input_tokens,
            context_size_tokens,
        )

    async def complete_ai_operation(
        self,
        operation_id: str,
        output_tokens: int = 0,
        quality_scores: dict[QualityMetricType, float] | None = None,
        success: bool = True,
        error_message: str = "",
    ) -> None:
        """Complete an AI operation."""
        self.ai_metrics.end_operation(
            operation_id,
            output_tokens=output_tokens,
            quality_scores=quality_scores,
            success=success,
            error_message=error_message,
        )

    def get_unified_dashboard_data(self) -> dict[str, Any]:
        """Get unified dashboard data across all monitoring systems."""
        try:
            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "system_overview": {
                    "apm_status": self.apm.get_health_status(),
                    "rum_status": self.rum.get_real_time_dashboard(),
                    "ai_status": self.ai_metrics.get_system_health(),
                    "slo_status": self.slo_manager.get_compliance_summary(),
                    "incident_status": self.incident_response.get_incident_summary(),
                },
                "alert_correlation": {
                    "active_correlations": len(self.alert_correlation),
                    "recent_violations": len(self.performance_violations),
                },
                "performance_insights": {
                    "total_insights": 0,  # Would be calculated from stored insights
                    "recommendations_pending": 0,
                },
            }
        except Exception as e:
            logger.exception(f"Error getting unified dashboard data: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Clean shutdown of integrated monitoring system."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Close individual systems
        await asyncio.gather(
            self.apm.close(),
            self.rum.close(),
            self.ai_metrics.close(),
            self.slo_manager.close(),
            self.incident_response.close(),
            return_exceptions=True,
        )

        logger.info("Integrated monitoring system shutdown complete")


# Export main classes
__all__ = [
    "IntegratedMonitoringSystem",
    "MonitoringConfig",
]
