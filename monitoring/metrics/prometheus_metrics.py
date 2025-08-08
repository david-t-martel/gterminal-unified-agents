"""Prometheus metrics collection for GTerminal.

Provides comprehensive metrics collection using the Prometheus client library
with custom metrics specific to GTerminal functionality.
"""

from contextlib import contextmanager
from functools import wraps
import logging
import time

try:
    from prometheus_client import CONTENT_TYPE_LATEST
    from prometheus_client import CollectorRegistry
    from prometheus_client import Counter
    from prometheus_client import Gauge
    from prometheus_client import Histogram
    from prometheus_client import Info
    from prometheus_client import generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus metrics collector for GTerminal."""

    def __init__(self, registry: CollectorRegistry | None = None):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics will be disabled")
            self._enabled = False
            return

        self._enabled = True
        self.registry = registry or CollectorRegistry()

        # HTTP metrics
        self.http_requests = Counter(
            "gterminal_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.http_request_duration = Histogram(
            "gterminal_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        # ReAct agent metrics
        self.react_reasoning_total = Counter(
            "gterminal_react_reasoning_total",
            "Total ReAct reasoning cycles",
            ["session_type", "completion_status"],
            registry=self.registry,
        )

        self.react_reasoning_duration = Histogram(
            "gterminal_react_reasoning_duration_seconds",
            "ReAct reasoning cycle duration",
            ["session_type"],
            buckets=[1, 5, 10, 30, 60, 120, 300],
            registry=self.registry,
        )

        self.react_steps = Histogram(
            "gterminal_react_steps_count",
            "Number of steps in ReAct reasoning",
            ["session_type"],
            buckets=[1, 2, 5, 10, 20, 50, 100],
            registry=self.registry,
        )

        self.react_errors = Counter(
            "gterminal_react_reasoning_errors_total",
            "ReAct reasoning errors",
            ["error_type", "session_type"],
            registry=self.registry,
        )

        # MCP server metrics
        self.mcp_requests = Counter(
            "gterminal_mcp_requests_total",
            "Total MCP requests",
            ["server", "tool", "status"],
            registry=self.registry,
        )

        self.mcp_request_duration = Histogram(
            "gterminal_mcp_request_duration_seconds",
            "MCP request duration",
            ["server", "tool"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

        self.mcp_rate_limited = Counter(
            "gterminal_mcp_rate_limited_total",
            "MCP requests that were rate limited",
            ["server", "client"],
            registry=self.registry,
        )

        # Authentication metrics
        self.auth_events = Counter(
            "gterminal_auth_events_total",
            "Authentication events",
            ["event_type", "provider", "status"],
            registry=self.registry,
        )

        self.auth_failed_logins = Counter(
            "gterminal_auth_failed_logins_total",
            "Failed login attempts",
            ["provider", "reason"],
            registry=self.registry,
        )

        self.auth_suspicious_activity = Counter(
            "gterminal_auth_suspicious_activity_total",
            "Suspicious authentication activity",
            ["activity_type", "source_ip"],
            registry=self.registry,
        )

        # Session management
        self.active_sessions = Gauge(
            "gterminal_active_sessions",
            "Number of active sessions",
            ["session_type"],
            registry=self.registry,
        )

        self.sessions_created = Counter(
            "gterminal_sessions_created_total",
            "Total sessions created",
            ["session_type"],
            registry=self.registry,
        )

        self.session_duration = Histogram(
            "gterminal_session_duration_seconds",
            "Session duration",
            ["session_type"],
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800],
            registry=self.registry,
        )

        # Cache metrics
        self.cache_hits = Counter(
            "gterminal_cache_hits_total",
            "Cache hits",
            ["cache_type"],
            registry=self.registry,
        )

        self.cache_misses = Counter(
            "gterminal_cache_misses_total",
            "Cache misses",
            ["cache_type"],
            registry=self.registry,
        )

        self.cache_size = Gauge(
            "gterminal_cache_size_bytes",
            "Cache size in bytes",
            ["cache_type"],
            registry=self.registry,
        )

        # API Key usage
        self.api_key_usage = Counter(
            "gterminal_api_key_usage_total",
            "API key usage",
            ["key_type", "scope"],
            registry=self.registry,
        )

        self.api_key_errors = Counter(
            "gterminal_api_key_errors_total",
            "API key authentication errors",
            ["error_type"],
            registry=self.registry,
        )

        # System metrics
        self.system_info = Info(
            "gterminal_system", "System information", registry=self.registry
        )

        self.application_info = Info(
            "gterminal_application", "Application information", registry=self.registry
        )

        # Performance metrics
        self.memory_usage = Gauge(
            "gterminal_memory_usage_bytes",
            "Memory usage by component",
            ["component"],
            registry=self.registry,
        )

        self.cpu_usage = Gauge(
            "gterminal_cpu_usage_percent",
            "CPU usage by component",
            ["component"],
            registry=self.registry,
        )

        # Tool execution metrics
        self.tool_executions = Counter(
            "gterminal_tool_executions_total",
            "Tool executions",
            ["tool_name", "status"],
            registry=self.registry,
        )

        self.tool_duration = Histogram(
            "gterminal_tool_duration_seconds",
            "Tool execution duration",
            ["tool_name"],
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        logger.info("Prometheus metrics initialized successfully")

    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._enabled

    def record_http_request(
        self, method: str, endpoint: str, status: int, duration: float
    ) -> None:
        """Record HTTP request metrics."""
        if not self._enabled:
            return

        self.http_requests.labels(
            method=method, endpoint=endpoint, status=str(status)
        ).inc()
        self.http_request_duration.labels(method=method, endpoint=endpoint).observe(
            duration
        )

    def record_react_reasoning(
        self,
        session_type: str,
        duration: float,
        steps_count: int,
        completion_status: str,
    ) -> None:
        """Record ReAct reasoning cycle metrics."""
        if not self._enabled:
            return

        self.react_reasoning_total.labels(
            session_type=session_type, completion_status=completion_status
        ).inc()

        self.react_reasoning_duration.labels(session_type=session_type).observe(
            duration
        )
        self.react_steps.labels(session_type=session_type).observe(steps_count)

    def record_react_error(self, error_type: str, session_type: str) -> None:
        """Record ReAct reasoning error."""
        if not self._enabled:
            return

        self.react_errors.labels(error_type=error_type, session_type=session_type).inc()

    def record_mcp_request(
        self, server: str, tool: str, status: str, duration: float
    ) -> None:
        """Record MCP request metrics."""
        if not self._enabled:
            return

        self.mcp_requests.labels(server=server, tool=tool, status=status).inc()
        self.mcp_request_duration.labels(server=server, tool=tool).observe(duration)

    def record_mcp_rate_limit(self, server: str, client: str) -> None:
        """Record MCP rate limiting event."""
        if not self._enabled:
            return

        self.mcp_rate_limited.labels(server=server, client=client).inc()

    def record_auth_event(self, event_type: str, provider: str, status: str) -> None:
        """Record authentication event."""
        if not self._enabled:
            return

        self.auth_events.labels(
            event_type=event_type, provider=provider, status=status
        ).inc()

    def record_failed_login(self, provider: str, reason: str) -> None:
        """Record failed login attempt."""
        if not self._enabled:
            return

        self.auth_failed_logins.labels(provider=provider, reason=reason).inc()

    def record_suspicious_activity(self, activity_type: str, source_ip: str) -> None:
        """Record suspicious authentication activity."""
        if not self._enabled:
            return

        self.auth_suspicious_activity.labels(
            activity_type=activity_type, source_ip=source_ip
        ).inc()

    def set_active_sessions(self, session_type: str, count: int) -> None:
        """Set number of active sessions."""
        if not self._enabled:
            return

        self.active_sessions.labels(session_type=session_type).set(count)

    def record_session_created(self, session_type: str) -> None:
        """Record session creation."""
        if not self._enabled:
            return

        self.sessions_created.labels(session_type=session_type).inc()

    def record_session_duration(self, session_type: str, duration: float) -> None:
        """Record session duration."""
        if not self._enabled:
            return

        self.session_duration.labels(session_type=session_type).observe(duration)

    def record_cache_hit(self, cache_type: str) -> None:
        """Record cache hit."""
        if not self._enabled:
            return

        self.cache_hits.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str) -> None:
        """Record cache miss."""
        if not self._enabled:
            return

        self.cache_misses.labels(cache_type=cache_type).inc()

    def set_cache_size(self, cache_type: str, size_bytes: int) -> None:
        """Set cache size."""
        if not self._enabled:
            return

        self.cache_size.labels(cache_type=cache_type).set(size_bytes)

    def record_api_key_usage(self, key_type: str, scope: str) -> None:
        """Record API key usage."""
        if not self._enabled:
            return

        self.api_key_usage.labels(key_type=key_type, scope=scope).inc()

    def record_api_key_error(self, error_type: str) -> None:
        """Record API key error."""
        if not self._enabled:
            return

        self.api_key_errors.labels(error_type=error_type).inc()

    def record_tool_execution(
        self, tool_name: str, status: str, duration: float
    ) -> None:
        """Record tool execution metrics."""
        if not self._enabled:
            return

        self.tool_executions.labels(tool_name=tool_name, status=status).inc()
        self.tool_duration.labels(tool_name=tool_name).observe(duration)

    def set_system_info(self, info: dict[str, str]) -> None:
        """Set system information."""
        if not self._enabled:
            return

        self.system_info.info(info)

    def set_application_info(self, info: dict[str, str]) -> None:
        """Set application information."""
        if not self._enabled:
            return

        self.application_info.info(info)

    def set_memory_usage(self, component: str, bytes_used: int) -> None:
        """Set memory usage for component."""
        if not self._enabled:
            return

        self.memory_usage.labels(component=component).set(bytes_used)

    def set_cpu_usage(self, component: str, percentage: float) -> None:
        """Set CPU usage for component."""
        if not self._enabled:
            return

        self.cpu_usage.labels(component=component).set(percentage)

    @contextmanager
    def time_mcp_request(self, server: str, tool: str):
        """Context manager to time MCP requests."""
        if not self._enabled:
            yield
            return

        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            self.record_mcp_request(server, tool, status, duration)

    @contextmanager
    def time_http_request(self, method: str, endpoint: str):
        """Context manager to time HTTP requests."""
        if not self._enabled:
            yield None
            return

        start_time = time.time()
        status = 200

        try:
            yield lambda s: setattr(locals(), "status", s)
        except Exception:
            status = 500
            raise
        finally:
            duration = time.time() - start_time
            self.record_http_request(method, endpoint, status, duration)

    def generate_metrics(self) -> str:
        """Generate Prometheus metrics output."""
        if not self._enabled:
            return "# Metrics collection disabled\n"

        return generate_latest(self.registry).decode("utf-8")

    def get_content_type(self) -> str:
        """Get content type for metrics response."""
        return CONTENT_TYPE_LATEST


# Global metrics registry
_metrics_registry: PrometheusMetrics | None = None


def get_metrics_registry() -> PrometheusMetrics:
    """Get the global metrics registry."""
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = PrometheusMetrics()
    return _metrics_registry


# Decorator for timing functions
def timed_operation(operation_name: str, labels: dict[str, str] | None = None):
    """Decorator to time operations and record metrics."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics_registry()
            if not metrics.is_enabled():
                return func(*args, **kwargs)

            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                # Record as tool execution
                metrics.record_tool_execution(operation_name, status, duration)

        return wrapper

    return decorator
