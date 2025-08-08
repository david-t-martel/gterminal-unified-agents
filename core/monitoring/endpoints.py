#!/usr/bin/env python3
"""Consolidated Monitoring API Endpoints.

Integrates existing performance monitoring with new monitoring infrastructure.
Consolidates functionality from performance_endpoints.py and adds new monitoring capabilities.
"""

import asyncio
from datetime import UTC
from datetime import datetime
import logging
from typing import Any

from fastapi import APIRouter
from fastapi import BackgroundTasks
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from .monitoring.ai_metrics import OperationType
from .monitoring.ai_metrics import QualityMetricType
from .monitoring.incident_response import IncidentSeverity
from .performance_integration import get_performance_integration

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["monitoring"])


# Dependency to get monitoring system from global state
def get_monitoring_system() -> None:
    """Get the global monitoring system instance."""
    from .server import monitoring_system

    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not initialized")
    return monitoring_system


def get_monitoring_dashboard() -> None:
    """Get the global monitoring dashboard instance."""
    from .server import monitoring_dashboard

    if not monitoring_dashboard:
        raise HTTPException(status_code=503, detail="Monitoring dashboard not initialized")
    return monitoring_dashboard


# Pydantic models for API responses
class SystemHealthResponse(BaseModel):
    """System health status response."""

    timestamp: str
    overall_status: str
    components: dict[str, Any]
    alerts: dict[str, Any]
    insights: dict[str, Any]


class RealTimeMetricsResponse(BaseModel):
    """Real-time metrics response."""

    timestamp: str
    performance: dict[str, Any]
    resources: dict[str, Any]
    incidents: dict[str, Any]
    slo_compliance: dict[str, Any]


class UserSessionRequest(BaseModel):
    """User session creation request."""

    session_id: str
    user_id: str | None = None
    user_agent: str | None = ""
    device_type: str | None = "desktop"
    location_country: str | None = ""
    location_city: str | None = ""


class AIOperationRequest(BaseModel):
    """AI operation tracking request."""

    session_id: str
    model_name: str
    operation_type: str
    input_tokens: int = 0
    context_size_tokens: int = 0


class AIOperationCompleteRequest(BaseModel):
    """AI operation completion request."""

    operation_id: str
    output_tokens: int = 0
    quality_score: float = 0.0
    quality_metric: str = "overall"
    success: bool = True
    error_message: str = ""


class IncidentRequest(BaseModel):
    """Incident creation request."""

    title: str
    description: str
    severity: str = "medium"
    source: str = "api"


class StreamRequest(BaseModel):
    """Monitoring stream request."""

    duration_seconds: int = Field(default=60, ge=10, le=3600)


# API Endpoints


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get comprehensive system health status."""
    try:
        get_monitoring_system()
        dashboard = get_monitoring_dashboard()

        dashboard_data = await dashboard.get_dashboard_data()

        if dashboard_data.get("error"):
            raise HTTPException(status_code=500, detail=dashboard_data["error"])

        # Extract system health data
        overview = dashboard_data["system_overview"]

        return SystemHealthResponse(
            timestamp=dashboard_data["timestamp"],
            overall_status=overview["overall_status"],
            components=overview["components"],
            alerts=dashboard_data["alerts_and_incidents"],
            insights=dashboard_data["insights"],
        )

    except Exception as e:
        logger.exception(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/realtime", response_model=RealTimeMetricsResponse)
async def get_real_time_metrics():
    """Get real-time performance metrics."""
    try:
        monitoring_system = get_monitoring_system()

        # Get real-time data from all systems
        rum_dashboard = monitoring_system.rum.get_real_time_dashboard()
        apm_status = monitoring_system.apm.get_health_status()
        ai_health = monitoring_system.ai_metrics.get_system_health()
        slo_summary = monitoring_system.slo_manager.get_compliance_summary()
        incident_summary = monitoring_system.incident_response.get_incident_summary()

        return RealTimeMetricsResponse(
            timestamp=datetime.now(UTC).isoformat(),
            performance={
                "response_times": {
                    "avg_ms": rum_dashboard.get("avg_response_time_ms", 0),
                    "p95_ms": apm_status.get("p95_response_time_ms", 0),
                    "p99_ms": apm_status.get("p99_response_time_ms", 0),
                },
                "error_rates": {
                    "total_rate": apm_status.get("error_rate_percent", 0),
                    "ai_error_rate": ai_health["system_metrics"]["error_rate"],
                    "user_errors_last_hour": rum_dashboard.get("errors_last_hour", 0),
                },
                "throughput": {
                    "requests_per_second": apm_status.get("requests_per_second", 0),
                    "ai_operations_per_hour": ai_health["system_metrics"]["total_operations"],
                    "active_users": rum_dashboard.get("active_users", 0),
                },
            },
            resources={
                "memory_usage_mb": apm_status.get("memory_usage_mb", 0),
                "cpu_utilization": apm_status.get("cpu_percent", 0),
                "ai_cost_per_hour": ai_health["system_metrics"]["cost_per_hour"],
                "cache_hit_rate": apm_status.get("cache_hit_rate", 0),
            },
            incidents={
                "active_count": incident_summary["summary"]["active_incidents"],
                "resolved_today": incident_summary["summary"]["resolved_today"],
                "mttr_minutes": incident_summary["summary"]["mttr_minutes"],
            },
            slo_compliance={
                "compliant_count": slo_summary["compliant_slos"],
                "violated_count": slo_summary["violated_slos"],
                "compliance_rate": (
                    slo_summary["compliant_slos"] / max(slo_summary["slo_count"], 1)
                )
                * 100,
            },
        )

    except Exception as e:
        logger.exception(f"Error getting real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/start")
async def start_user_session(request: UserSessionRequest):
    """Start tracking a new user session."""
    try:
        monitoring_system = get_monitoring_system()

        device_info = {"type": request.device_type} if request.device_type else None
        location_info: dict[str, Any] = {}
        if request.location_country:
            location_info["country"] = request.location_country
        if request.location_city:
            location_info["city"] = request.location_city

        await monitoring_system.start_user_session(
            session_id=request.session_id,
            user_id=request.user_id if request.user_id else None,
            user_agent=request.user_agent,
            device_info=device_info,
            location_info=location_info if location_info else None,
        )

        return {
            "success": True,
            "session_id": request.session_id,
            "started_at": datetime.now(UTC).isoformat(),
            "message": f"Started tracking session {request.session_id}",
        }

    except Exception as e:
        logger.exception(f"Error starting user session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/operations/start")
async def start_ai_operation(request: AIOperationRequest):
    """Start tracking an AI operation."""
    try:
        monitoring_system = get_monitoring_system()

        # Validate operation type
        try:
            op_type = OperationType(request.operation_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid operation type: {request.operation_type}. Valid types: {[op.value for op in OperationType]}",
            )

        operation_id = await monitoring_system.record_ai_operation(
            session_id=request.session_id,
            model_name=request.model_name,
            operation_type=op_type,
            input_tokens=request.input_tokens,
            context_size_tokens=request.context_size_tokens,
        )

        return {
            "success": True,
            "operation_id": operation_id,
            "started_at": datetime.now(UTC).isoformat(),
            "message": f"Started tracking AI operation {operation_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error starting AI operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/operations/complete")
async def complete_ai_operation(request: AIOperationCompleteRequest):
    """Complete an AI operation and record metrics."""
    try:
        monitoring_system = get_monitoring_system()

        quality_scores: dict[str, Any] = {}
        if request.quality_score > 0:
            try:
                quality_metric_type = QualityMetricType(request.quality_metric.lower())
                quality_scores[quality_metric_type] = request.quality_score
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid quality metric: {request.quality_metric}. Valid metrics: {[qm.value for qm in QualityMetricType]}",
                )

        await monitoring_system.complete_ai_operation(
            operation_id=request.operation_id,
            output_tokens=request.output_tokens,
            quality_scores=quality_scores if quality_scores else None,
            success=request.success,
            error_message=request.error_message,
        )

        return {
            "success": True,
            "operation_id": request.operation_id,
            "completed_at": datetime.now(UTC).isoformat(),
            "message": f"Completed AI operation {request.operation_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error completing AI operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/incidents")
async def get_incident_status():
    """Get current incident status and active incidents."""
    try:
        monitoring_system = get_monitoring_system()
        incident_summary = monitoring_system.incident_response.get_incident_summary()

        # Get detailed information about active incidents
        active_incidents: list[Any] = []
        for incident_id in incident_summary.get("active_incident_ids", []):
            try:
                incident = monitoring_system.incident_response.get_incident(incident_id)
                if incident:
                    active_incidents.append(
                        {
                            "id": incident.incident_id,
                            "title": incident.title,
                            "severity": incident.severity.value,
                            "status": incident.status.value,
                            "created_at": incident.created_at.isoformat(),
                            "updated_at": incident.updated_at.isoformat(),
                            "source": incident.source,
                            "alert_count": len(incident.alerts),
                        },
                    )
            except Exception as e:
                logger.debug(f"Error getting incident {incident_id}: {e}")

        return {
            "summary": incident_summary["summary"],
            "active_incidents": active_incidents,
            "recent_activity": incident_summary.get("recent_activity", []),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Error getting incident status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/incidents")
async def create_incident(request: IncidentRequest):
    """Create a new incident."""
    try:
        monitoring_system = get_monitoring_system()

        try:
            incident_severity = IncidentSeverity(request.severity.upper())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid severity: {request.severity}. Valid severities: {[s.value.lower() for s in IncidentSeverity]}",
            )

        incident_id = await monitoring_system.incident_response.create_incident(
            title=request.title,
            description=request.description,
            severity=incident_severity,
            source=request.source,
        )

        return {
            "success": True,
            "incident_id": incident_id,
            "created_at": datetime.now(UTC).isoformat(),
            "message": f"Created incident {incident_id}: {request.title}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/slo")
async def get_slo_status():
    """Get Service Level Objective status and compliance."""
    try:
        monitoring_system = get_monitoring_system()

        slo_status = monitoring_system.slo_manager.get_slo_status()
        compliance_summary = monitoring_system.slo_manager.get_compliance_summary()
        alert_summary = monitoring_system.slo_manager.get_alert_summary()

        return {
            "compliance_summary": compliance_summary,
            "slo_details": slo_status,
            "alerts": alert_summary,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Error getting SLO status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights")
async def get_performance_insights():
    """Get AI-powered performance insights and recommendations."""
    try:
        monitoring_system = get_monitoring_system()

        insights = {
            "timestamp": datetime.now(UTC).isoformat(),
            "apm_insights": monitoring_system._analyze_apm_patterns(),
            "rum_insights": monitoring_system._analyze_user_behavior(),
            "ai_insights": monitoring_system._analyze_ai_performance(),
            "slo_insights": monitoring_system._analyze_slo_trends(),
            "incident_insights": monitoring_system._analyze_incident_patterns(),
        }

        # Aggregate recommendations
        all_recommendations: list[Any] = []
        for category, category_insights in insights.items():
            if category == "timestamp":
                continue
            if isinstance(category_insights, dict) and "recommendations" in category_insights:
                for rec in category_insights["recommendations"]:
                    all_recommendations.append(
                        {
                            "category": category.replace("_insights", ""),
                            "recommendation": rec,
                            "priority": "high" if "critical" in rec.lower() else "medium",
                        },
                    )

        insights["aggregated_recommendations"] = all_recommendations
        insights["recommendation_count"] = len(all_recommendations)

        return insights

    except Exception as e:
        logger.exception(f"Error getting performance insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alert_correlation():
    """Get cross-system alert correlation analysis."""
    try:
        monitoring_system = get_monitoring_system()

        correlation_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "active_correlations": dict(monitoring_system.alert_correlation),
            "recent_violations": monitoring_system.performance_violations[-10:],  # Last 10
            "correlation_patterns": {},
        }

        # Analyze correlation patterns
        if monitoring_system.alert_correlation:
            patterns: dict[str, Any] = {}
            for alerts in monitoring_system.alert_correlation.values():
                for alert_type, alert_list in alerts.items():
                    if alert_list:
                        patterns[alert_type] = patterns.get(alert_type, 0) + len(alert_list)
            correlation_data["correlation_patterns"] = patterns

        return correlation_data

    except Exception as e:
        logger.exception(f"Error getting alert correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_monitoring_dashboard():
    """Get unified monitoring dashboard data."""
    try:
        dashboard = get_monitoring_dashboard()

        dashboard_data = await dashboard.get_dashboard_data()

        if dashboard_data.get("error"):
            raise HTTPException(status_code=500, detail=dashboard_data["error"])

        return dashboard_data

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get dashboard summary for quick status checks."""
    try:
        dashboard = get_monitoring_dashboard()

        summary = await dashboard.get_terminal_summary()

        if summary.get("status") == "error":
            raise HTTPException(status_code=500, detail=summary["message"])

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_monitoring_data(request: StreamRequest, background_tasks: BackgroundTasks):
    """Start a monitoring data collection stream."""
    try:
        get_monitoring_system()
        dashboard = get_monitoring_dashboard()

        # Start background task for data collection
        collection_id = f"stream_{int(datetime.now(UTC).timestamp())}"

        async def collect_stream_data() -> None:
            """Background task to collect streaming data."""
            try:
                monitoring_data: list[Any] = []
                datetime.now(UTC)

                # Collect data points every 5 seconds
                for i in range(0, request.duration_seconds, 5):
                    current_time = datetime.now(UTC)

                    # Get current dashboard data
                    dashboard_data = await dashboard.get_dashboard_data()

                    data_point = {
                        "timestamp": current_time.isoformat(),
                        "elapsed_seconds": i,
                        "dashboard_data": dashboard_data,
                    }

                    monitoring_data.append(data_point)

                    # Wait 5 seconds before next collection
                    if i < request.duration_seconds - 5:
                        await asyncio.sleep(5)

                # Store results (in production, you'd store this in a database or cache)
                logger.info(
                    f"Completed stream collection {collection_id}: {len(monitoring_data)} data points"
                )

            except Exception as e:
                logger.exception(f"Error in stream collection {collection_id}: {e}")

        background_tasks.add_task(collect_stream_data)

        return {
            "success": True,
            "collection_id": collection_id,
            "duration_seconds": request.duration_seconds,
            "started_at": datetime.now(UTC).isoformat(),
            "message": f"Started monitoring stream collection for {request.duration_seconds} seconds",
        }

    except Exception as e:
        logger.exception(f"Error starting monitoring stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === CONSOLIDATED PERFORMANCE ENDPOINTS (from performance_endpoints.py) ===


@router.get("/performance/dashboard")
async def get_performance_dashboard():
    """Get complete performance dashboard (consolidated from existing performance_endpoints)."""
    try:
        integration = await get_performance_integration()
        return await integration.get_performance_dashboard()
    except Exception as e:
        logger.exception(f"Failed to get performance dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/summary")
async def get_performance_summary():
    """Get performance summary for quick overview (consolidated from existing performance_endpoints)."""
    try:
        integration = await get_performance_integration()
        dashboard = await integration.get_performance_dashboard()

        return {
            "timestamp": dashboard["timestamp"],
            "overall_health": dashboard["health"]["status"],
            "cache_hit_rate": dashboard["performance_summary"]["cache_hit_rate"],
            "total_requests": dashboard["performance_summary"]["total_requests"],
            "error_rate": dashboard["performance_summary"]["error_rate"],
            "redis_enabled": dashboard["performance_summary"]["redis_enabled"],
            "budget_violations": [
                name
                for name, budget in dashboard["budgets"].items()
                if budget.get("status") in ["POOR", "CRITICAL"]
            ],
        }
    except Exception as e:
        logger.exception(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics (consolidated from existing performance_endpoints)."""
    try:
        integration = await get_performance_integration()
        return await integration.optimizer.cache_manager.get_stats()
    except Exception as e:
        logger.exception(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections/stats")
async def get_connection_stats():
    """Get connection pool statistics (consolidated from existing performance_endpoints)."""
    try:
        integration = await get_performance_integration()
        return await integration.optimizer.connection_manager.get_comprehensive_stats()
    except Exception as e:
        logger.exception(f"Failed to get connection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache(pattern: str | None = None):
    """Clear cache entries matching pattern (consolidated from existing performance_endpoints)."""
    try:
        integration = await get_performance_integration()
        count = await integration.optimizer.cache_manager.clear(pattern)
        return {
            "cleared_items": count,
            "pattern": pattern or "all",
            "timestamp": asyncio.get_event_loop().time(),
        }
    except Exception as e:
        logger.exception(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/warm-up")
async def warm_up_cache():
    """Warm up cache with common data (consolidated from existing performance_endpoints)."""
    try:
        integration = await get_performance_integration()
        await integration.optimizer.warm_up_cache()
        return {
            "status": "success",
            "message": "Cache warm-up completed",
            "timestamp": asyncio.get_event_loop().time(),
        }
    except Exception as e:
        logger.exception(f"Failed to warm up cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check for monitoring system specifically
@router.get("/status")
async def monitoring_system_status():
    """Get comprehensive monitoring system component status."""
    try:
        monitoring_system = get_monitoring_system()
        dashboard = get_monitoring_dashboard()

        # Get performance integration status too
        performance_status = "unavailable"
        try:
            integration = await get_performance_integration()
            health = await integration.health_check()
            performance_status = health.get("status", "unknown")
        except Exception as e:
            logger.debug(f"Performance integration unavailable: {e}")

        return {
            "monitoring_system": "operational" if monitoring_system else "unavailable",
            "dashboard": "operational" if dashboard else "unavailable",
            "performance_integration": performance_status,
            "components": {
                "apm": "operational",
                "rum": "operational",
                "ai_metrics": "operational",
                "slo_manager": "operational",
                "incident_response": "operational",
                "cache_manager": performance_status,
                "connection_pool": performance_status,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Error getting monitoring system status: {e}")
        return {
            "monitoring_system": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }
