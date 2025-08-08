"""Real User Monitoring (RUM) System.

This module provides comprehensive real user monitoring capabilities including:
- Frontend performance tracking
- User experience metrics (Core Web Vitals)
- Real user journey tracking
- Client-side error monitoring
- Network performance analysis
- Device and browser performance insights
"""

import asyncio
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
import logging
from statistics import mean
from statistics import median
from typing import Any
import uuid

logger = logging.getLogger(__name__)


class UserActionType(Enum):
    """Types of user actions to track."""

    PAGE_LOAD = "page_load"
    NAVIGATION = "navigation"
    INTERACTION = "interaction"
    FORM_SUBMISSION = "form_submission"
    API_CALL = "api_call"
    ERROR = "error"
    CUSTOM = "custom"


class PerformanceMetricType(Enum):
    """Types of performance metrics."""

    # Core Web Vitals
    FIRST_CONTENTFUL_PAINT = "fcp"
    LARGEST_CONTENTFUL_PAINT = "lcp"
    CUMULATIVE_LAYOUT_SHIFT = "cls"
    FIRST_INPUT_DELAY = "fid"

    # Additional metrics
    TIME_TO_FIRST_BYTE = "ttfb"
    TOTAL_BLOCKING_TIME = "tbt"
    SPEED_INDEX = "si"

    # Custom metrics
    AI_RESPONSE_TIME = "ai_response_time"
    CHAT_LOAD_TIME = "chat_load_time"
    SEARCH_TIME = "search_time"


@dataclass
class UserSession:
    """Represents a user session."""

    session_id: str
    user_id: str | None
    start_time: datetime
    end_time: datetime | None = None
    page_views: int = 0
    interactions: int = 0
    errors: int = 0
    user_agent: str = ""
    device_type: str = ""  # mobile, tablet, desktop
    browser: str = ""
    os: str = ""
    country: str = ""
    referrer: str = ""
    entry_page: str = ""
    exit_page: str = ""
    total_duration_ms: float = 0
    bounce: bool = False


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""

    metric_type: PerformanceMetricType
    value: float
    timestamp: datetime
    session_id: str
    page_url: str
    user_agent: str = ""
    connection_type: str = ""  # 4g, wifi, etc.
    device_memory: float | None = None  # GB
    cpu_cores: int | None = None
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class UserAction:
    """Represents a user action/event."""

    action_id: str
    session_id: str
    action_type: UserActionType
    timestamp: datetime
    page_url: str
    element_selector: str = ""
    action_name: str = ""
    duration_ms: float = 0
    success: bool = True
    error_message: str = ""
    custom_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreWebVitals:
    """Core Web Vitals metrics for a page."""

    url: str
    fcp: float | None = None  # First Contentful Paint (ms)
    lcp: float | None = None  # Largest Contentful Paint (ms)
    cls: float | None = None  # Cumulative Layout Shift (score)
    fid: float | None = None  # First Input Delay (ms)
    ttfb: float | None = None  # Time to First Byte (ms)

    def get_score(self, metric: PerformanceMetricType) -> str:
        """Get performance score (good/needs-improvement/poor) for a metric."""
        value = getattr(self, metric.value, None)
        if value is None:
            return "unknown"

        # Define thresholds based on Core Web Vitals standards
        thresholds = {
            PerformanceMetricType.FIRST_CONTENTFUL_PAINT: (1800, 3000),
            PerformanceMetricType.LARGEST_CONTENTFUL_PAINT: (2500, 4000),
            PerformanceMetricType.CUMULATIVE_LAYOUT_SHIFT: (0.1, 0.25),
            PerformanceMetricType.FIRST_INPUT_DELAY: (100, 300),
            PerformanceMetricType.TIME_TO_FIRST_BYTE: (800, 1800),
        }

        if metric not in thresholds:
            return "unknown"

        good_threshold, poor_threshold = thresholds[metric]

        if value <= good_threshold:
            return "good"
        if value <= poor_threshold:
            return "needs-improvement"
        return "poor"


class RealUserMonitoring:
    """Real User Monitoring system for frontend performance tracking."""

    def __init__(self, retention_days: int = 30) -> None:
        self.retention_days = retention_days
        self.start_time = datetime.now(UTC)

        # Data storage
        self.sessions: dict[str, UserSession] = {}
        self.performance_metrics: dict[str, list[PerformanceMetric]] = defaultdict(list)
        self.user_actions: dict[str, list[UserAction]] = defaultdict(list)
        self.core_web_vitals: dict[str, list[CoreWebVitals]] = defaultdict(list)

        # Aggregated metrics
        self.page_performance: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.user_journey_paths: dict[str, int] = defaultdict(int)
        self.error_patterns: dict[str, int] = defaultdict(int)

        # Real-time metrics
        self.active_sessions: set[str] = set()
        self.real_time_metrics = {
            "active_users": 0,
            "page_views_per_minute": deque(maxlen=60),
            "errors_per_minute": deque(maxlen=60),
            "avg_response_time": deque(maxlen=100),
        }

        # Start background tasks
        self._background_tasks: list[asyncio.Task] = []
        self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        self._background_tasks = [
            asyncio.create_task(self._process_real_time_metrics()),
            asyncio.create_task(self._cleanup_old_data()),
            asyncio.create_task(self._analyze_user_journeys()),
        ]

    async def _process_real_time_metrics(self) -> None:
        """Process real-time metrics every minute."""
        while True:
            try:
                await self._calculate_real_time_metrics()
                await asyncio.sleep(60)  # Update every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error processing real-time metrics: {e}")
                await asyncio.sleep(60)

    async def _cleanup_old_data(self) -> None:
        """Clean up old data based on retention policy."""
        while True:
            try:
                cutoff_date = datetime.now(UTC) - timedelta(days=self.retention_days)
                await self._cleanup_data_before_date(cutoff_date)
                await asyncio.sleep(3600)  # Clean up every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error cleaning up old data: {e}")
                await asyncio.sleep(3600)

    async def _analyze_user_journeys(self) -> None:
        """Analyze user journey patterns."""
        while True:
            try:
                await self._analyze_journey_patterns()
                await asyncio.sleep(300)  # Analyze every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error analyzing user journeys: {e}")
                await asyncio.sleep(300)

    def start_session(
        self,
        session_id: str,
        user_id: str | None = None,
        user_agent: str = "",
        device_info: dict[str, Any] | None = None,
        location_info: dict[str, str] | None = None,
    ) -> UserSession:
        """Start tracking a new user session."""
        device_info = device_info or {}
        location_info = location_info or {}

        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(UTC),
            user_agent=user_agent,
            device_type=device_info.get("type", "unknown"),
            browser=device_info.get("browser", "unknown"),
            os=device_info.get("os", "unknown"),
            country=location_info.get("country", "unknown"),
            referrer=device_info.get("referrer", ""),
        )

        self.sessions[session_id] = session
        self.active_sessions.add(session_id)

        logger.debug(f"Started session tracking: {session_id}")
        return session

    def end_session(self, session_id: str, exit_page: str = "") -> None:
        """End a user session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.end_time = datetime.now(UTC)
            session.exit_page = exit_page

            if session.start_time:
                session.total_duration_ms = (
                    session.end_time - session.start_time
                ).total_seconds() * 1000

            # Determine if it was a bounce (single page, short duration)
            session.bounce = (
                session.page_views <= 1 and session.total_duration_ms < 30000
            )  # Less than 30 seconds

            self.active_sessions.discard(session_id)
            logger.debug(f"Ended session tracking: {session_id}")

    def record_performance_metric(
        self,
        session_id: str,
        metric_type: PerformanceMetricType,
        value: float,
        page_url: str,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """Record a performance metric."""
        additional_data = additional_data or {}

        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(UTC),
            session_id=session_id,
            page_url=page_url,
            user_agent=additional_data.get("user_agent", ""),
            connection_type=additional_data.get("connection_type", ""),
            device_memory=additional_data.get("device_memory"),
            cpu_cores=additional_data.get("cpu_cores"),
            labels=additional_data.get("labels", {}),
        )

        self.performance_metrics[session_id].append(metric)

        # Update aggregated page performance
        self.page_performance[page_url][metric_type.value].append(value)

        # Update real-time metrics
        if metric_type == PerformanceMetricType.AI_RESPONSE_TIME:
            self.real_time_metrics["avg_response_time"].append(value)

    def record_core_web_vitals(
        self,
        session_id: str,
        page_url: str,
        fcp: float | None = None,
        lcp: float | None = None,
        cls: float | None = None,
        fid: float | None = None,
        ttfb: float | None = None,
    ) -> None:
        """Record Core Web Vitals for a page."""
        vitals = CoreWebVitals(
            url=page_url,
            fcp=fcp,
            lcp=lcp,
            cls=cls,
            fid=fid,
            ttfb=ttfb,
        )

        self.core_web_vitals[session_id].append(vitals)

        # Record individual metrics
        metrics_to_record = [
            (PerformanceMetricType.FIRST_CONTENTFUL_PAINT, fcp),
            (PerformanceMetricType.LARGEST_CONTENTFUL_PAINT, lcp),
            (PerformanceMetricType.CUMULATIVE_LAYOUT_SHIFT, cls),
            (PerformanceMetricType.FIRST_INPUT_DELAY, fid),
            (PerformanceMetricType.TIME_TO_FIRST_BYTE, ttfb),
        ]

        for metric_type, value in metrics_to_record:
            if value is not None:
                self.record_performance_metric(session_id, metric_type, value, page_url)

    def record_user_action(
        self,
        session_id: str,
        action_type: UserActionType,
        page_url: str,
        action_name: str = "",
        element_selector: str = "",
        duration_ms: float = 0,
        success: bool = True,
        error_message: str = "",
        custom_data: dict[str, Any] | None = None,
    ) -> None:
        """Record a user action/event."""
        action = UserAction(
            action_id=str(uuid.uuid4()),
            session_id=session_id,
            action_type=action_type,
            timestamp=datetime.now(UTC),
            page_url=page_url,
            element_selector=element_selector,
            action_name=action_name,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            custom_data=custom_data or {},
        )

        self.user_actions[session_id].append(action)

        # Update session counters
        if session_id in self.sessions:
            session = self.sessions[session_id]

            if action_type == UserActionType.PAGE_LOAD:
                session.page_views += 1
                if not session.entry_page:
                    session.entry_page = page_url
            elif action_type == UserActionType.INTERACTION:
                session.interactions += 1
            elif action_type == UserActionType.ERROR:
                session.errors += 1
                self.error_patterns[f"{page_url}:{error_message}"] += 1

    def get_page_performance_summary(self, page_url: str) -> dict[str, Any]:
        """Get performance summary for a specific page."""
        if page_url not in self.page_performance:
            return {"error": f"No performance data for {page_url}"}

        page_data = self.page_performance[page_url]
        summary = {"page_url": page_url, "metrics": {}}

        for metric_name, values in page_data.items():
            if not values:
                continue

            sorted_values = sorted(values)
            summary["metrics"][metric_name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": mean(values),
                "median": median(values),
                "p75": (
                    sorted_values[int(len(sorted_values) * 0.75)]
                    if len(sorted_values) > 4
                    else max(values)
                ),
                "p95": (
                    sorted_values[int(len(sorted_values) * 0.95)]
                    if len(sorted_values) > 20
                    else max(values)
                ),
                "p99": (
                    sorted_values[int(len(sorted_values) * 0.99)]
                    if len(sorted_values) > 100
                    else max(values)
                ),
            }

        return summary

    def get_core_web_vitals_summary(self, page_url: str | None = None) -> dict[str, Any]:
        """Get Core Web Vitals summary."""
        # Collect all Core Web Vitals data
        all_vitals: list[Any] = []
        for session_vitals in self.core_web_vitals.values():
            if page_url:
                all_vitals.extend([v for v in session_vitals if v.url == page_url])
            else:
                all_vitals.extend(session_vitals)

        if not all_vitals:
            return {"error": "No Core Web Vitals data available"}

        # Calculate aggregated metrics
        metrics: dict[str, Any] = {}
        metric_fields = ["fcp", "lcp", "cls", "fid", "ttfb"]

        for field in metric_fields:
            values = [
                getattr(vital, field) for vital in all_vitals if getattr(vital, field) is not None
            ]
            if values:
                sorted_values = sorted(values)
                metrics[field] = {
                    "count": len(values),
                    "p75": (
                        sorted_values[int(len(sorted_values) * 0.75)]
                        if len(sorted_values) > 4
                        else max(values)
                    ),
                    "median": median(values),
                    "mean": mean(values),
                }

                # Add performance scores
                metric_type = getattr(PerformanceMetricType, field.upper(), None)
                if metric_type:
                    sample_vital = CoreWebVitals(url="")
                    setattr(sample_vital, field, metrics[field]["p75"])
                    metrics[field]["score"] = sample_vital.get_score(metric_type)

        return {
            "page_url": page_url or "all_pages",
            "total_measurements": len(all_vitals),
            "metrics": metrics,
        }

    def get_user_journey_analysis(self, session_id: str | None = None) -> dict[str, Any]:
        """Analyze user journey patterns."""
        if session_id and session_id in self.user_actions:
            # Single session journey
            actions = self.user_actions[session_id]
            journey = [
                {
                    "timestamp": action.timestamp.isoformat(),
                    "action_type": action.action_type.value,
                    "page_url": action.page_url,
                    "action_name": action.action_name,
                    "duration_ms": action.duration_ms,
                    "success": action.success,
                }
                for action in sorted(actions, key=lambda a: a.timestamp)
            ]

            return {
                "session_id": session_id,
                "journey": journey,
                "total_actions": len(actions),
            }

        # Overall journey analysis
        page_sequences = defaultdict(int)

        for session_actions in self.user_actions.values():
            page_loads = [
                action
                for action in session_actions
                if action.action_type == UserActionType.PAGE_LOAD
            ]

            if len(page_loads) > 1:
                # Track page sequences
                for i in range(len(page_loads) - 1):
                    from_page = page_loads[i].page_url
                    to_page = page_loads[i + 1].page_url
                    page_sequences[f"{from_page} -> {to_page}"] += 1

        # Find most common paths
        top_paths = sorted(page_sequences.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_sessions_analyzed": len(self.user_actions),
            "most_common_paths": [{"path": path, "count": count} for path, count in top_paths],
            "total_unique_paths": len(page_sequences),
        }

    def get_error_analysis(self) -> dict[str, Any]:
        """Get error analysis and patterns."""
        # Collect all errors
        all_errors: list[Any] = []
        for session_actions in self.user_actions.values():
            errors = [
                action for action in session_actions if action.action_type == UserActionType.ERROR
            ]
            all_errors.extend(errors)

        if not all_errors:
            return {"total_errors": 0, "message": "No errors recorded"}

        # Analyze error patterns
        error_by_page = defaultdict(int)
        error_by_type = defaultdict(int)
        error_by_time = defaultdict(int)

        for error in all_errors:
            error_by_page[error.page_url] += 1
            error_by_type[error.error_message] += 1
            hour = error.timestamp.hour
            error_by_time[hour] += 1

        return {
            "total_errors": len(all_errors),
            "errors_by_page": dict(
                sorted(error_by_page.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "errors_by_type": dict(
                sorted(error_by_type.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "errors_by_hour": dict(error_by_time),
            "error_rate": len(all_errors) / len(self.sessions) * 100 if self.sessions else 0,
        }

    def get_real_time_dashboard(self) -> dict[str, Any]:
        """Get real-time dashboard data."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "active_users": len(self.active_sessions),
            "total_sessions": len(self.sessions),
            "page_views_last_hour": sum(self.real_time_metrics["page_views_per_minute"]),
            "errors_last_hour": sum(self.real_time_metrics["errors_per_minute"]),
            "avg_response_time_ms": (
                mean(self.real_time_metrics["avg_response_time"])
                if self.real_time_metrics["avg_response_time"]
                else 0
            ),
            "bounce_rate": self._calculate_bounce_rate(),
            "top_pages": self._get_top_pages(),
            "device_breakdown": self._get_device_breakdown(),
        }

    async def _calculate_real_time_metrics(self) -> None:
        """Calculate real-time metrics."""
        try:
            now = datetime.now(UTC)
            one_minute_ago = now - timedelta(minutes=1)

            # Count page views in last minute
            page_views = 0
            errors = 0

            for session_actions in self.user_actions.values():
                recent_actions = [
                    action for action in session_actions if action.timestamp >= one_minute_ago
                ]

                page_views += len(
                    [
                        action
                        for action in recent_actions
                        if action.action_type == UserActionType.PAGE_LOAD
                    ],
                )

                errors += len(
                    [
                        action
                        for action in recent_actions
                        if action.action_type == UserActionType.ERROR
                    ]
                )

            self.real_time_metrics["page_views_per_minute"].append(page_views)
            self.real_time_metrics["errors_per_minute"].append(errors)
            self.real_time_metrics["active_users"] = len(self.active_sessions)

        except Exception as e:
            logger.exception(f"Error calculating real-time metrics: {e}")

    async def _cleanup_data_before_date(self, cutoff_date: datetime) -> None:
        """Clean up data older than cutoff date."""
        try:
            # Clean up sessions
            sessions_to_remove = [
                session_id
                for session_id, session in self.sessions.items()
                if session.start_time and session.start_time < cutoff_date
            ]

            for session_id in sessions_to_remove:
                self.sessions.pop(session_id, None)
                self.performance_metrics.pop(session_id, None)
                self.user_actions.pop(session_id, None)
                self.core_web_vitals.pop(session_id, None)

            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")

        except Exception as e:
            logger.exception(f"Error cleaning up old data: {e}")

    async def _analyze_journey_patterns(self) -> None:
        """Analyze user journey patterns."""
        try:
            # This would implement more sophisticated journey analysis
            # For now, just update the journey paths counter
            self.user_journey_paths.clear()

            # Analyze recent journeys
            for session_actions in self.user_actions.values():
                page_loads = [
                    action.page_url
                    for action in session_actions
                    if action.action_type == UserActionType.PAGE_LOAD
                ]

                if len(page_loads) > 1:
                    journey_path = " -> ".join(page_loads)
                    self.user_journey_paths[journey_path] += 1

        except Exception as e:
            logger.exception(f"Error analyzing journey patterns: {e}")

    def _calculate_bounce_rate(self) -> float:
        """Calculate overall bounce rate."""
        if not self.sessions:
            return 0.0

        completed_sessions = [
            session for session in self.sessions.values() if session.end_time is not None
        ]

        if not completed_sessions:
            return 0.0

        bounced_sessions = len([session for session in completed_sessions if session.bounce])
        return (bounced_sessions / len(completed_sessions)) * 100

    def _get_top_pages(self) -> list[dict[str, Any]]:
        """Get top pages by page views."""
        page_views = defaultdict(int)

        for session_actions in self.user_actions.values():
            for action in session_actions:
                if action.action_type == UserActionType.PAGE_LOAD:
                    page_views[action.page_url] += 1

        return [
            {"page": page, "views": views}
            for page, views in sorted(page_views.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

    def _get_device_breakdown(self) -> dict[str, int]:
        """Get breakdown of sessions by device type."""
        device_counts = defaultdict(int)

        for session in self.sessions.values():
            device_counts[session.device_type] += 1

        return dict(device_counts)

    async def close(self) -> None:
        """Clean shutdown of RUM system."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("RUM system shutdown complete")


# Export main classes
__all__ = [
    "CoreWebVitals",
    "PerformanceMetric",
    "PerformanceMetricType",
    "RealUserMonitoring",
    "UserAction",
    "UserActionType",
    "UserSession",
]
