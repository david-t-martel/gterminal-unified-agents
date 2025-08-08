"""Service Level Objectives (SLO) Manager.

This module provides comprehensive SLO management for AI operations including:
- SLO definition and tracking
- Error budget calculations
- Alert threshold management
- SLO compliance reporting
- Burn rate detection
- Incident impact assessment
"""

import asyncio
from collections import defaultdict
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
import logging
from statistics import mean
from typing import Any

logger = logging.getLogger(__name__)


class SLOType(Enum):
    """Types of SLOs."""

    AVAILABILITY = "availability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUALITY = "quality"
    COST = "cost"


class SLOStatus(Enum):
    """SLO compliance status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"  # Error budget exhausted


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SLODefinition:
    """Definition of a Service Level Objective."""

    name: str
    description: str
    slo_type: SLOType
    target_percentage: float  # e.g., 99.9% availability
    measurement_window_hours: int  # e.g., 24 hours, 720 hours (30 days)
    service: str  # e.g., "ai-inference", "chat-service"

    # Query definitions for calculating SLI
    good_events_query: str  # Prometheus query for good events
    total_events_query: str  # Prometheus query for total events

    # Alert thresholds (as percentage of error budget consumed)
    warning_threshold: float = 50.0  # Alert when 50% of error budget consumed
    critical_threshold: float = 80.0  # Alert when 80% of error budget consumed

    # Burn rate thresholds for different time windows
    fast_burn_rate_threshold: float = 14.4  # 2% of error budget in 1 hour
    slow_burn_rate_threshold: float = 6.0  # 10% of error budget in 6 hours

    # Additional metadata
    owner_team: str = ""
    documentation_url: str = ""
    runbook_url: str = ""
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class SLIDataPoint:
    """Single SLI (Service Level Indicator) measurement."""

    timestamp: datetime
    good_events: float
    total_events: float
    sli_value: float  # good_events / total_events

    @property
    def error_rate(self) -> float:
        """Calculate error rate (1 - SLI)."""
        return 1.0 - self.sli_value


@dataclass
class ErrorBudget:
    """Error budget calculation for an SLO."""

    total_budget: float  # Total error budget for the period
    consumed_budget: float  # Amount of error budget consumed
    remaining_budget: float  # Remaining error budget
    consumption_rate: float  # Rate of budget consumption (per hour)
    estimated_exhaustion_time: datetime | None = None  # When budget will be exhausted

    @property
    def consumption_percentage(self) -> float:
        """Percentage of error budget consumed."""
        if self.total_budget == 0:
            return 0.0
        return (self.consumed_budget / self.total_budget) * 100

    @property
    def status(self) -> SLOStatus:
        """Determine status based on consumption."""
        consumption_pct = self.consumption_percentage

        if consumption_pct >= 100:
            return SLOStatus.EXHAUSTED
        if consumption_pct >= 80:
            return SLOStatus.CRITICAL
        if consumption_pct >= 50:
            return SLOStatus.WARNING
        return SLOStatus.HEALTHY


@dataclass
class BurnRateAlert:
    """Burn rate alert information."""

    slo_name: str
    time_window_hours: int
    burn_rate: float
    threshold: float
    severity: AlertSeverity
    message: str
    detected_at: datetime
    projected_exhaustion: datetime | None = None


class SLOManager:
    """Comprehensive SLO management system."""

    def __init__(self, prometheus_client: Any | None = None) -> None:
        self.prometheus_client = prometheus_client
        self.start_time = datetime.now(UTC)

        # SLO definitions
        self.slo_definitions: dict[str, SLODefinition] = {}

        # SLI data storage
        self.sli_data: dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Error budget tracking
        self.error_budgets: dict[str, ErrorBudget] = {}

        # Alert history
        self.active_alerts: dict[str, list[BurnRateAlert]] = defaultdict(list)
        self.alert_history: deque = deque(maxlen=1000)

        # Compliance tracking
        self.compliance_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Callbacks for alerts
        self.alert_callbacks: list[Callable] = []

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []

        # Initialize default SLOs
        self._initialize_default_slos()

        # Start monitoring
        self._start_background_monitoring()

    def _initialize_default_slos(self) -> None:
        """Initialize default SLOs for AI operations."""
        default_slos = [
            # AI Inference Availability
            SLODefinition(
                name="ai_inference_availability",
                description="AI inference service availability",
                slo_type=SLOType.AVAILABILITY,
                target_percentage=99.9,
                measurement_window_hours=24,
                service="ai-inference",
                good_events_query='sum(rate(ai_inference_total{status="success"}[5m]))',
                total_events_query="sum(rate(ai_inference_total[5m]))",
                owner_team="ai-platform",
                runbook_url="https://runbooks.example.com/ai-inference-availability",
            ),
            # AI Response Time
            SLODefinition(
                name="ai_response_latency_p95",
                description="95% of AI responses within 2 seconds",
                slo_type=SLOType.LATENCY,
                target_percentage=95.0,
                measurement_window_hours=24,
                service="ai-inference",
                good_events_query='sum(rate(ai_inference_duration_seconds_bucket{le="2.0"}[5m]))',
                total_events_query="sum(rate(ai_inference_duration_seconds_count[5m]))",
                owner_team="ai-platform",
                runbook_url="https://runbooks.example.com/ai-latency",
            ),
            # Chat Service Availability
            SLODefinition(
                name="chat_service_availability",
                description="Chat service availability for users",
                slo_type=SLOType.AVAILABILITY,
                target_percentage=99.5,
                measurement_window_hours=168,  # 7 days
                service="chat-service",
                good_events_query='sum(rate(http_requests_total{service="chat",status!~"5.."}[5m]))',
                total_events_query='sum(rate(http_requests_total{service="chat"}[5m]))',
                owner_team="frontend",
                runbook_url="https://runbooks.example.com/chat-availability",
            ),
            # AI Quality SLO
            SLODefinition(
                name="ai_quality_score",
                description="AI response quality above threshold",
                slo_type=SLOType.QUALITY,
                target_percentage=90.0,
                measurement_window_hours=168,  # 7 days
                service="ai-inference",
                good_events_query='sum(rate(ai_quality_score_total{score="good"}[5m]))',
                total_events_query="sum(rate(ai_quality_score_total[5m]))",
                owner_team="ai-platform",
                runbook_url="https://runbooks.example.com/ai-quality",
            ),
            # Frontend Performance
            SLODefinition(
                name="frontend_core_web_vitals",
                description="Core Web Vitals compliance (LCP < 2.5s)",
                slo_type=SLOType.LATENCY,
                target_percentage=75.0,  # 75% of page loads
                measurement_window_hours=24,
                service="frontend",
                good_events_query='sum(rate(rum_core_web_vitals_bucket{metric="lcp",le="2500"}[5m]))',
                total_events_query='sum(rate(rum_core_web_vitals_count{metric="lcp"}[5m]))',
                owner_team="frontend",
                runbook_url="https://runbooks.example.com/frontend-performance",
            ),
        ]

        for slo in default_slos:
            self.register_slo(slo)

    def register_slo(self, slo_definition: SLODefinition) -> None:
        """Register a new SLO definition."""
        self.slo_definitions[slo_definition.name] = slo_definition
        logger.info(f"Registered SLO: {slo_definition.name} ({slo_definition.target_percentage}%)")

    def add_alert_callback(self, callback: Callable) -> None:
        """Add a callback function for SLO alerts."""
        self.alert_callbacks.append(callback)

    def _start_background_monitoring(self) -> None:
        """Start background SLO monitoring tasks."""
        self._background_tasks = [
            asyncio.create_task(self._collect_sli_data()),
            asyncio.create_task(self._calculate_error_budgets()),
            asyncio.create_task(self._monitor_burn_rates()),
            asyncio.create_task(self._generate_compliance_reports()),
        ]

    async def _collect_sli_data(self) -> None:
        """Collect SLI data for all registered SLOs."""
        while True:
            try:
                for slo_name, slo_def in self.slo_definitions.items():
                    await self._collect_slo_data(slo_name, slo_def)

                await asyncio.sleep(60)  # Collect every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error collecting SLI data: {e}")
                await asyncio.sleep(60)

    async def _collect_slo_data(self, slo_name: str, slo_def: SLODefinition) -> None:
        """Collect SLI data for a specific SLO."""
        try:
            # In a real implementation, this would query Prometheus
            # For now, simulate data collection
            if self.prometheus_client:
                # Query Prometheus for actual data
                good_events = await self._query_prometheus(slo_def.good_events_query)
                total_events = await self._query_prometheus(slo_def.total_events_query)
            else:
                # Simulate data for demonstration
                good_events = self._simulate_metric_value(slo_def.slo_type, is_good=True)
                total_events = self._simulate_metric_value(slo_def.slo_type, is_good=False)

            if total_events > 0:
                sli_value = min(1.0, good_events / total_events)
            else:
                sli_value = 1.0  # No events means perfect SLI

            data_point = SLIDataPoint(
                timestamp=datetime.now(UTC),
                good_events=good_events,
                total_events=total_events,
                sli_value=sli_value,
            )

            self.sli_data[slo_name].append(data_point)

        except Exception as e:
            logger.exception(f"Error collecting data for SLO {slo_name}: {e}")

    def _simulate_metric_value(self, slo_type: SLOType, is_good: bool) -> float:
        """Simulate metric values for demonstration."""
        import random

        if slo_type == SLOType.AVAILABILITY:
            if is_good:
                return random.uniform(95, 100)  # 95-100 successful operations
            return 100  # Total operations
        if slo_type == SLOType.LATENCY:
            if is_good:
                return random.uniform(85, 98)  # 85-98% within latency threshold
            return 100  # Total requests
        if slo_type == SLOType.QUALITY:
            if is_good:
                return random.uniform(80, 95)  # 80-95% good quality
            return 100  # Total responses
        return random.uniform(90, 99) if is_good else 100

    async def _query_prometheus(self, query: str) -> float:
        """Query Prometheus for metric data."""
        # Placeholder for actual Prometheus integration
        # In a real implementation, this would use the prometheus_client
        return 0.0

    async def _calculate_error_budgets(self) -> None:
        """Calculate error budgets for all SLOs."""
        while True:
            try:
                for slo_name, slo_def in self.slo_definitions.items():
                    error_budget = self._calculate_error_budget(slo_name, slo_def)
                    self.error_budgets[slo_name] = error_budget

                await asyncio.sleep(300)  # Calculate every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error calculating error budgets: {e}")
                await asyncio.sleep(300)

    def _calculate_error_budget(self, slo_name: str, slo_def: SLODefinition) -> ErrorBudget:
        """Calculate error budget for a specific SLO."""
        if slo_name not in self.sli_data or not self.sli_data[slo_name]:
            return ErrorBudget(
                total_budget=0.0,
                consumed_budget=0.0,
                remaining_budget=0.0,
                consumption_rate=0.0,
            )

        # Get data for the measurement window
        window_start = datetime.now(UTC) - timedelta(hours=slo_def.measurement_window_hours)
        window_data = [dp for dp in self.sli_data[slo_name] if dp.timestamp >= window_start]

        if not window_data:
            return ErrorBudget(
                total_budget=0.0,
                consumed_budget=0.0,
                remaining_budget=0.0,
                consumption_rate=0.0,
            )

        # Calculate total events and error budget
        total_events = sum(dp.total_events for dp in window_data)
        if total_events == 0:
            return ErrorBudget(
                total_budget=0.0,
                consumed_budget=0.0,
                remaining_budget=0.0,
                consumption_rate=0.0,
            )

        # Calculate allowed errors (error budget)
        allowed_error_rate = (100.0 - slo_def.target_percentage) / 100.0
        total_error_budget = total_events * allowed_error_rate

        # Calculate actual errors
        actual_errors = sum(dp.total_events - dp.good_events for dp in window_data)
        consumed_budget = min(actual_errors, total_error_budget)
        remaining_budget = max(0.0, total_error_budget - consumed_budget)

        # Calculate consumption rate (errors per hour)
        window_hours = len(window_data) / 60.0  # Assuming 1-minute intervals
        consumption_rate = actual_errors / max(window_hours, 1.0)

        # Estimate exhaustion time
        estimated_exhaustion_time = None
        if consumption_rate > 0 and remaining_budget > 0:
            hours_to_exhaustion = remaining_budget / consumption_rate
            estimated_exhaustion_time = datetime.now(UTC) + timedelta(hours=hours_to_exhaustion)

        return ErrorBudget(
            total_budget=total_error_budget,
            consumed_budget=consumed_budget,
            remaining_budget=remaining_budget,
            consumption_rate=consumption_rate,
            estimated_exhaustion_time=estimated_exhaustion_time,
        )

    async def _monitor_burn_rates(self) -> None:
        """Monitor burn rates and generate alerts."""
        while True:
            try:
                for slo_name, slo_def in self.slo_definitions.items():
                    await self._check_burn_rate(slo_name, slo_def)

                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error monitoring burn rates: {e}")
                await asyncio.sleep(60)

    async def _check_burn_rate(self, slo_name: str, slo_def: SLODefinition) -> None:
        """Check burn rate for a specific SLO and generate alerts."""
        if slo_name not in self.error_budgets:
            return

        error_budget = self.error_budgets[slo_name]

        # Check different time windows for burn rate
        time_windows = [
            (1, slo_def.fast_burn_rate_threshold),  # 1 hour window
            (6, slo_def.slow_burn_rate_threshold),  # 6 hour window
        ]

        for window_hours, threshold in time_windows:
            burn_rate = self._calculate_burn_rate(slo_name, window_hours)

            if burn_rate > threshold:
                # Generate burn rate alert
                severity = AlertSeverity.CRITICAL if window_hours == 1 else AlertSeverity.WARNING

                alert = BurnRateAlert(
                    slo_name=slo_name,
                    time_window_hours=window_hours,
                    burn_rate=burn_rate,
                    threshold=threshold,
                    severity=severity,
                    message=f"High burn rate detected for {slo_name}: {burn_rate:.2f}x threshold in {window_hours}h window",
                    detected_at=datetime.now(UTC),
                    projected_exhaustion=error_budget.estimated_exhaustion_time,
                )

                await self._handle_burn_rate_alert(alert)

    def _calculate_burn_rate(self, slo_name: str, window_hours: int) -> float:
        """Calculate burn rate for a specific time window."""
        if slo_name not in self.sli_data or not self.sli_data[slo_name]:
            return 0.0

        window_start = datetime.now(UTC) - timedelta(hours=window_hours)
        window_data = [dp for dp in self.sli_data[slo_name] if dp.timestamp >= window_start]

        if not window_data:
            return 0.0

        # Calculate error rate in this window
        total_events = sum(dp.total_events for dp in window_data)
        if total_events == 0:
            return 0.0

        actual_errors = sum(dp.total_events - dp.good_events for dp in window_data)
        error_rate = actual_errors / total_events

        # Get SLO definition
        slo_def = self.slo_definitions[slo_name]
        allowed_error_rate = (100.0 - slo_def.target_percentage) / 100.0

        # Calculate burn rate (how many times faster than allowed)
        if allowed_error_rate == 0:
            return float("inf") if error_rate > 0 else 0.0

        return error_rate / allowed_error_rate

    async def _handle_burn_rate_alert(self, alert: BurnRateAlert) -> None:
        """Handle a burn rate alert."""
        # Check if this is a duplicate alert
        existing_alerts = self.active_alerts[alert.slo_name]
        duplicate = any(
            existing.time_window_hours == alert.time_window_hours
            and existing.severity == alert.severity
            for existing in existing_alerts
        )

        if not duplicate:
            self.active_alerts[alert.slo_name].append(alert)
            self.alert_history.append(alert)

            logger.warning(f"SLO Alert: {alert.message}")

            # Call registered alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.exception(f"Error in alert callback: {e}")

    async def _generate_compliance_reports(self) -> None:
        """Generate compliance reports for all SLOs."""
        while True:
            try:
                for slo_name in self.slo_definitions:
                    report = self._generate_compliance_report(slo_name)
                    self.compliance_history[slo_name].append(report)

                    # Keep only last 30 days of reports
                    cutoff_date = datetime.now(UTC) - timedelta(days=30)
                    self.compliance_history[slo_name] = [
                        r
                        for r in self.compliance_history[slo_name]
                        if datetime.fromisoformat(r["timestamp"]) >= cutoff_date
                    ]

                await asyncio.sleep(3600)  # Generate every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error generating compliance reports: {e}")
                await asyncio.sleep(3600)

    def _generate_compliance_report(self, slo_name: str) -> dict[str, Any]:
        """Generate compliance report for a specific SLO."""
        if slo_name not in self.slo_definitions:
            return {}

        slo_def = self.slo_definitions[slo_name]
        error_budget = self.error_budgets.get(slo_name)

        # Calculate current SLI
        recent_data = list(self.sli_data[slo_name])[-60:]  # Last hour
        current_sli = mean([dp.sli_value for dp in recent_data]) if recent_data else 0.0

        return {
            "slo_name": slo_name,
            "timestamp": datetime.now(UTC).isoformat(),
            "target_percentage": slo_def.target_percentage,
            "current_sli_percentage": current_sli * 100,
            "compliance": current_sli * 100 >= slo_def.target_percentage,
            "error_budget": asdict(error_budget) if error_budget else None,
            "status": error_budget.status.value if error_budget else "unknown",
            "measurement_window_hours": slo_def.measurement_window_hours,
        }

    def get_slo_status(self, slo_name: str | None = None) -> dict[str, Any]:
        """Get current status of SLOs."""
        if slo_name and slo_name in self.slo_definitions:
            # Return status for specific SLO
            slo_def = self.slo_definitions[slo_name]
            error_budget = self.error_budgets.get(slo_name)
            recent_data = list(self.sli_data[slo_name])[-60:]

            return {
                "slo_name": slo_name,
                "description": slo_def.description,
                "target_percentage": slo_def.target_percentage,
                "current_sli": (
                    mean([dp.sli_value for dp in recent_data]) * 100 if recent_data else 0.0
                ),
                "error_budget": asdict(error_budget) if error_budget else None,
                "status": error_budget.status.value if error_budget else "unknown",
                "active_alerts": len(self.active_alerts[slo_name]),
                "last_updated": recent_data[-1].timestamp.isoformat() if recent_data else None,
            }

        # Return status for all SLOs
        status_summary: dict[str, Any] = {}
        for slo_name in self.slo_definitions:
            status_summary[slo_name] = self.get_slo_status(slo_name)

        return status_summary

    def get_compliance_summary(self, days: int = 7) -> dict[str, Any]:
        """Get compliance summary for the last N days."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        summary = {
            "period_days": days,
            "slo_count": len(self.slo_definitions),
            "compliant_slos": 0,
            "violated_slos": 0,
            "slo_details": {},
        }

        for slo_name, slo_def in self.slo_definitions.items():
            # Get recent compliance data
            recent_reports = [
                r
                for r in self.compliance_history[slo_name]
                if datetime.fromisoformat(r["timestamp"]) >= cutoff_date
            ]

            if recent_reports:
                compliance_rate = (
                    sum(1 for r in recent_reports if r["compliance"]) / len(recent_reports) * 100
                )
                avg_sli = mean([r["current_sli_percentage"] for r in recent_reports])

                is_compliant = compliance_rate >= 95  # 95% of time periods must be compliant

                if is_compliant:
                    summary["compliant_slos"] += 1
                else:
                    summary["violated_slos"] += 1

                summary["slo_details"][slo_name] = {
                    "target": slo_def.target_percentage,
                    "avg_sli": avg_sli,
                    "compliance_rate": compliance_rate,
                    "is_compliant": is_compliant,
                    "error_budget_status": (
                        self.error_budgets[slo_name].status.value
                        if slo_name in self.error_budgets
                        else "unknown"
                    ),
                }

        return summary

    def get_alert_summary(self) -> dict[str, Any]:
        """Get summary of active and recent alerts."""
        now = datetime.now(UTC)

        # Count active alerts by severity
        active_by_severity = defaultdict(int)
        for alerts in self.active_alerts.values():
            for alert in alerts:
                active_by_severity[alert.severity.value] += 1

        # Count recent alerts (last 24 hours)
        recent_alerts = [
            alert for alert in self.alert_history if now - alert.detected_at <= timedelta(hours=24)
        ]

        return {
            "active_alerts": {
                "total": sum(len(alerts) for alerts in self.active_alerts.values()),
                "by_severity": dict(active_by_severity),
                "by_slo": {
                    slo: len(alerts) for slo, alerts in self.active_alerts.items() if alerts
                },
            },
            "recent_alerts_24h": len(recent_alerts),
            "most_alerting_slos": [
                {"slo": slo, "count": len(alerts)}
                for slo, alerts in sorted(
                    self.active_alerts.items(), key=lambda x: len(x[1]), reverse=True
                )[:5]
                if alerts
            ],
        }

    async def close(self) -> None:
        """Clean shutdown of SLO manager."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("SLO Manager shutdown complete")


# Export main classes
__all__ = [
    "AlertSeverity",
    "BurnRateAlert",
    "ErrorBudget",
    "SLIDataPoint",
    "SLODefinition",
    "SLOManager",
    "SLOStatus",
    "SLOType",
]
