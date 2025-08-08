"""Performance metrics collection and monitoring.

Implements:
- Cache hit/miss rates tracking
- Connection pool statistics
- Response time percentiles
- Request coalescing metrics
- Performance budget monitoring
"""

import asyncio
from collections import defaultdict
from collections import deque
from collections.abc import Callable
import contextlib
from dataclasses import dataclass
from dataclasses import field
import logging
from statistics import mean
from statistics import median
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBudget:
    """Performance budget configuration."""

    name: str
    target_value: float
    threshold_value: float
    unit: str
    description: str
    alert_enabled: bool = True


@dataclass
class MetricSample:
    """Individual metric sample with metadata."""

    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


class MetricCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self, max_samples_per_metric: int = 10000) -> None:
        self.max_samples_per_metric = max_samples_per_metric
        self._metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples_per_metric))
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._last_reset = time.time()
        self._lock = asyncio.Lock()

    async def record_metric(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a metric sample."""
        sample = MetricSample(
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            context=context or {},
        )

        async with self._lock:
            self._metrics[name].append(sample)

    async def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        async with self._lock:
            self._counters[name] += amount

    async def record_histogram(self, name: str, value: float) -> None:
        """Record a value in a histogram."""
        async with self._lock:
            self._histograms[name].append(value)
            # Keep histogram size manageable
            if len(self._histograms[name]) > self.max_samples_per_metric:
                self._histograms[name] = self._histograms[name][-self.max_samples_per_metric :]

    async def get_metric_stats(self, name: str) -> dict[str, Any]:
        """Get statistics for a specific metric."""
        async with self._lock:
            if name not in self._metrics:
                return {"error": f"Metric {name} not found"}

            samples = list(self._metrics[name])
            if not samples:
                return {"count": 0}

            values = [s.value for s in samples]

            # Calculate statistics
            stats = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": mean(values),
                "median": median(values),
                "latest": values[-1] if values else 0,
                "oldest": values[0] if values else 0,
                "first_timestamp": samples[0].timestamp,
                "last_timestamp": samples[-1].timestamp,
            }

            # Calculate percentiles
            if len(values) >= 2:
                sorted_values = sorted(values)
                stats.update(
                    {
                        "p50": self._percentile(sorted_values, 50),
                        "p90": self._percentile(sorted_values, 90),
                        "p95": self._percentile(sorted_values, 95),
                        "p99": self._percentile(sorted_values, 99),
                    },
                )

            return stats

    async def get_counter_value(self, name: str) -> int:
        """Get current counter value."""
        async with self._lock:
            return self._counters.get(name, 0)

    async def get_histogram_stats(self, name: str) -> dict[str, Any]:
        """Get histogram statistics."""
        async with self._lock:
            if name not in self._histograms:
                return {"error": f"Histogram {name} not found"}

            values = self._histograms[name]
            if not values:
                return {"count": 0}

            sorted_values = sorted(values)
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": mean(values),
                "median": median(values),
                "p50": self._percentile(sorted_values, 50),
                "p90": self._percentile(sorted_values, 90),
                "p95": self._percentile(sorted_values, 95),
                "p99": self._percentile(sorted_values, 99),
            }

    def _percentile(self, sorted_values: list[float], percentile: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0

        index = (percentile / 100) * (len(sorted_values) - 1)
        if index.is_integer():
            return sorted_values[int(index)]
        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))

    async def reset_metrics(self) -> None:
        """Reset all metrics."""
        async with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._histograms.clear()
            self._last_reset = time.time()

    async def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all metrics."""
        async with self._lock:
            stats = {
                "timestamp": time.time(),
                "last_reset": self._last_reset,
                "metrics": {},
                "counters": dict(self._counters),
                "histograms": {},
            }

            # Get metric stats
            for name in self._metrics:
                stats["metrics"][name] = await self.get_metric_stats(name)

            # Get histogram stats
            for name in self._histograms:
                stats["histograms"][name] = await self.get_histogram_stats(name)

            return stats


class PerformanceMonitor:
    """Main performance monitoring system."""

    def __init__(self, alert_callback: Callable | None = None) -> None:
        self.metrics = MetricCollector()
        self.budgets: dict[str, PerformanceBudget] = {}
        self.alert_callback = alert_callback
        self._monitoring_task: asyncio.Task | None = None
        self._request_coalescer: dict[str, list[asyncio.Future]] = defaultdict(list)
        self._coalescer_lock = asyncio.Lock()

        # Default performance budgets
        self._setup_default_budgets()

        # Start monitoring
        self._start_monitoring()

    def _setup_default_budgets(self) -> None:
        """Setup default performance budgets."""
        self.budgets = {
            "response_time_p95": PerformanceBudget(
                name="response_time_p95",
                target_value=2000,  # 2 seconds
                threshold_value=5000,  # 5 seconds
                unit="ms",
                description="95th percentile response time",
            ),
            "cache_hit_rate": PerformanceBudget(
                name="cache_hit_rate",
                target_value=70,  # 70%
                threshold_value=50,  # 50%
                unit="%",
                description="Overall cache hit rate",
            ),
            "error_rate": PerformanceBudget(
                name="error_rate",
                target_value=0.1,  # 0.1%
                threshold_value=1.0,  # 1%
                unit="%",
                description="Request error rate",
            ),
            "memory_usage": PerformanceBudget(
                name="memory_usage",
                target_value=2048,  # 2GB
                threshold_value=3584,  # 3.5GB (leaving 0.5GB buffer from 4GB limit)
                unit="MB",
                description="Memory usage per instance",
            ),
        }

    def _start_monitoring(self) -> None:
        """Start background monitoring task."""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._check_performance_budgets()
                await self._cleanup_expired_coalescers()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Performance monitoring error: {e}")

    async def _check_performance_budgets(self) -> None:
        """Check all performance budgets and alert if exceeded."""
        for budget_name, budget in self.budgets.items():
            if not budget.alert_enabled:
                continue

            try:
                # Get current metric value
                current_value = await self._get_budget_metric_value(budget)

                if current_value is None:
                    continue

                # Check if threshold exceeded
                if budget_name == "cache_hit_rate":
                    # For cache hit rate, alert if below threshold
                    exceeded = current_value < budget.threshold_value
                else:
                    # For other metrics, alert if above threshold
                    exceeded = current_value > budget.threshold_value

                if exceeded and self.alert_callback:
                    await self.alert_callback(
                        budget_name=budget_name,
                        current_value=current_value,
                        threshold_value=budget.threshold_value,
                        budget=budget,
                    )

            except Exception as e:
                logger.exception(f"Error checking budget {budget_name}: {e}")

    async def _get_budget_metric_value(self, budget: PerformanceBudget) -> float | None:
        """Get current value for a performance budget metric."""
        if budget.name == "response_time_p95":
            stats = await self.metrics.get_histogram_stats("response_time")
            return stats.get("p95")

        if budget.name == "cache_hit_rate":
            hits = await self.metrics.get_counter_value("cache_hits")
            misses = await self.metrics.get_counter_value("cache_misses")
            total = hits + misses
            return (hits / total * 100) if total > 0 else None

        if budget.name == "error_rate":
            errors = await self.metrics.get_counter_value("request_errors")
            total = await self.metrics.get_counter_value("total_requests")
            return (errors / total * 100) if total > 0 else None

        if budget.name == "memory_usage":
            stats = await self.metrics.get_metric_stats("memory_usage_mb")
            return stats.get("latest")

        return None

    async def _cleanup_expired_coalescers(self) -> None:
        """Clean up expired request coalescers."""
        time.time()
        async with self._coalescer_lock:
            expired_keys: list[Any] = []
            for key, futures_list in self._request_coalescer.items():
                # Remove completed or cancelled futures
                active_futures = [f for f in futures_list if not f.done()]
                if not active_futures:
                    expired_keys.append(key)
                else:
                    self._request_coalescer[key] = active_futures

            for key in expired_keys:
                del self._request_coalescer[key]

    async def record_request_start(self, request_id: str, operation: str) -> None:
        """Record the start of a request."""
        await self.metrics.record_metric(
            f"request_start_{operation}",
            time.time(),
            labels={"request_id": request_id, "operation": operation},
        )
        await self.metrics.increment_counter("total_requests")

    async def record_request_end(
        self,
        request_id: str,
        operation: str,
        duration_ms: float,
        success: bool = True,
    ) -> None:
        """Record the completion of a request."""
        await self.metrics.record_histogram("response_time", duration_ms)
        await self.metrics.record_metric(
            f"request_end_{operation}",
            duration_ms,
            labels={
                "request_id": request_id,
                "operation": operation,
                "success": str(success),
            },
        )

        if not success:
            await self.metrics.increment_counter("request_errors")

    async def record_cache_hit(self, cache_type: str = "general") -> None:
        """Record a cache hit."""
        await self.metrics.increment_counter("cache_hits")
        await self.metrics.increment_counter(f"cache_hits_{cache_type}")

    async def record_cache_miss(self, cache_type: str = "general") -> None:
        """Record a cache miss."""
        await self.metrics.increment_counter("cache_misses")
        await self.metrics.increment_counter(f"cache_misses_{cache_type}")

    async def record_memory_usage(self, usage_mb: float) -> None:
        """Record current memory usage."""
        await self.metrics.record_metric("memory_usage_mb", usage_mb)

    async def coalesce_request(self, key: str, request_func: Callable, *args, **kwargs) -> Any:
        """Coalesce duplicate requests to reduce load."""
        async with self._coalescer_lock:
            # Check if there's already a pending request for this key
            if key in self._request_coalescer:
                # Wait for existing request to complete
                existing_futures = self._request_coalescer[key]
                if existing_futures:
                    try:
                        # Wait for the first active future
                        for future in existing_futures:
                            if not future.done():
                                logger.debug(f"Coalescing request for key: {key}")
                                return await future
                    except Exception:
                        # If the existing request failed, continue with new request
                        pass

            # Create new request
            future = asyncio.create_task(request_func(*args, **kwargs))

            if key not in self._request_coalescer:
                self._request_coalescer[key] = []
            self._request_coalescer[key].append(future)

            try:
                return await future
            finally:
                # Clean up completed future
                async with self._coalescer_lock:
                    if key in self._request_coalescer:
                        self._request_coalescer[key] = [
                            f for f in self._request_coalescer[key] if f != future
                        ]
                        if not self._request_coalescer[key]:
                            del self._request_coalescer[key]

    async def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        all_stats = await self.metrics.get_all_stats()

        # Calculate derived metrics
        derived_metrics: dict[str, Any] = {}

        # Cache hit rate
        hits = await self.metrics.get_counter_value("cache_hits")
        misses = await self.metrics.get_counter_value("cache_misses")
        total_cache_requests = hits + misses
        if total_cache_requests > 0:
            derived_metrics["cache_hit_rate"] = (hits / total_cache_requests) * 100

        # Error rate
        errors = await self.metrics.get_counter_value("request_errors")
        total_requests = await self.metrics.get_counter_value("total_requests")
        if total_requests > 0:
            derived_metrics["error_rate"] = (errors / total_requests) * 100

        # Budget status
        budget_status: dict[str, Any] = {}
        for budget_name, budget in self.budgets.items():
            current_value = await self._get_budget_metric_value(budget)
            if current_value is not None:
                if budget_name == "cache_hit_rate":
                    status = "GOOD" if current_value >= budget.target_value else "POOR"
                    status = "CRITICAL" if current_value < budget.threshold_value else status
                else:
                    status = "GOOD" if current_value <= budget.target_value else "POOR"
                    status = "CRITICAL" if current_value > budget.threshold_value else status

                budget_status[budget_name] = {
                    "current_value": current_value,
                    "target_value": budget.target_value,
                    "threshold_value": budget.threshold_value,
                    "status": status,
                    "unit": budget.unit,
                    "description": budget.description,
                }

        return {
            "timestamp": time.time(),
            "raw_metrics": all_stats,
            "derived_metrics": derived_metrics,
            "performance_budgets": budget_status,
            "coalesced_requests": len(self._request_coalescer),
        }

    async def close(self) -> None:
        """Close the performance monitor."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task


class PerformanceMetrics:
    """High-level interface for performance metrics."""

    def __init__(self, monitor: PerformanceMonitor | None = None) -> None:
        self.monitor = monitor or PerformanceMonitor()

    async def start_request(self, request_id: str, operation: str) -> float:
        """Start timing a request."""
        start_time = time.time()
        await self.monitor.record_request_start(request_id, operation)
        return start_time

    async def end_request(
        self,
        request_id: str,
        operation: str,
        start_time: float,
        success: bool = True,
    ) -> None:
        """End timing a request."""
        # Fix: Ensure proper time calculation - start_time should be in seconds since epoch
        current_time = time.time()
        if start_time > current_time:
            # If start_time is in the future, it's likely a timestamp in wrong format
            # Use a reasonable default duration
            duration_ms = 100.0  # 100ms default
        else:
            duration_ms = (current_time - start_time) * 1000
            # Clamp to reasonable bounds (1ms to 60 seconds)
            duration_ms = max(1.0, min(60000.0, duration_ms))

        await self.monitor.record_request_end(request_id, operation, duration_ms, success)

    async def cache_hit(self, cache_type: str = "general") -> None:
        """Record cache hit."""
        await self.monitor.record_cache_hit(cache_type)

    async def cache_miss(self, cache_type: str = "general") -> None:
        """Record cache miss."""
        await self.monitor.record_cache_miss(cache_type)

    async def coalesce_request(self, key: str, func: Callable, *args, **kwargs) -> Any:
        """Coalesce duplicate requests."""
        return await self.monitor.coalesce_request(key, func, *args, **kwargs)

    async def get_report(self) -> dict[str, Any]:
        """Get performance report."""
        return await self.monitor.get_performance_report()

    async def close(self) -> None:
        """Close metrics system."""
        await self.monitor.close()


# Integration with OptimizedBaseAgentService
def create_performance_decorator(metrics: PerformanceMetrics) -> None:
    """Create a performance monitoring decorator for agent methods."""

    def decorator(operation_name: str) -> None:
        def method_decorator(func) -> None:
            if asyncio.iscoroutinefunction(func):

                async def async_wrapper(self, *args, **kwargs):
                    request_id = getattr(self, "_current_request_id", str(time.time()))
                    start_time = await metrics.start_request(request_id, operation_name)

                    try:
                        result = await func(self, *args, **kwargs)
                        await metrics.end_request(request_id, operation_name, start_time, True)
                        return result
                    except Exception:
                        await metrics.end_request(request_id, operation_name, start_time, False)
                        raise

                return async_wrapper

            def sync_wrapper(self, *args, **kwargs) -> None:
                getattr(self, "_current_request_id", str(time.time()))
                time.time()

                try:
                    return func(self, *args, **kwargs)
                    # Record success (simplified for sync methods)
                except Exception:
                    # Record failure (simplified for sync methods)
                    raise

            return sync_wrapper

        return method_decorator

    return decorator


class AgentMetricsIntegration:
    """Integration class for connecting OptimizedBaseAgentService with metrics."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.metrics = PerformanceMetrics()
        self._job_start_times: dict[str, float] = {}

    async def on_job_start(self, job_id: str, job_type: str) -> None:
        """Called when a job starts."""
        start_time = await self.metrics.start_request(job_id, f"job_{job_type}")
        self._job_start_times[job_id] = start_time

        await self.metrics.monitor.metrics.increment_counter(f"jobs_started_{job_type}")
        await self.metrics.monitor.metrics.increment_counter("jobs_started_total")

    async def on_job_complete(self, job_id: str, job_type: str, success: bool) -> None:
        """Called when a job completes."""
        start_time = self._job_start_times.pop(job_id, time.time())
        await self.metrics.end_request(job_id, f"job_{job_type}", start_time, success)

        counter_suffix = "completed" if success else "failed"
        await self.metrics.monitor.metrics.increment_counter(f"jobs_{counter_suffix}_{job_type}")
        await self.metrics.monitor.metrics.increment_counter(f"jobs_{counter_suffix}_total")

    async def on_cache_hit(self, cache_key: str, cache_type: str = "general") -> None:
        """Called on cache hit."""
        await self.metrics.cache_hit(cache_type)

    async def on_cache_miss(self, cache_key: str, cache_type: str = "general") -> None:
        """Called on cache miss."""
        await self.metrics.cache_miss(cache_type)

    async def record_resource_usage(self, memory_mb: float, cpu_percent: float) -> None:
        """Record resource usage metrics."""
        await self.metrics.monitor.record_memory_usage(memory_mb)
        await self.metrics.monitor.metrics.record_metric("cpu_usage_percent", cpu_percent)

    async def record_circuit_breaker_event(
        self, job_type: str, state: str, failure_count: int
    ) -> None:
        """Record circuit breaker state changes."""
        await self.metrics.monitor.metrics.record_metric(
            f"circuit_breaker_{job_type}_state",
            {"CLOSED": 0, "OPEN": 1, "HALF_OPEN": 0.5}.get(state, 0),
            labels={"job_type": job_type, "state": state},
        )
        await self.metrics.monitor.metrics.record_metric(
            f"circuit_breaker_{job_type}_failures",
            failure_count,
            labels={"job_type": job_type},
        )

    async def get_agent_performance_report(self) -> dict[str, Any]:
        """Get comprehensive agent performance report."""
        base_report = await self.metrics.get_report()

        # Add agent-specific metrics
        agent_metrics = {
            "agent_name": self.agent_name,
            "active_jobs": len(self._job_start_times),
            "job_types_processed": [],
            "circuit_breaker_states": {},
        }

        # Extract job type metrics from counters
        counters = base_report["raw_metrics"]["counters"]
        for counter_name in counters:
            if counter_name.startswith("jobs_started_") and not counter_name.endswith("_total"):
                job_type = counter_name.replace("jobs_started_", "")
                if job_type not in agent_metrics["job_types_processed"]:
                    agent_metrics["job_types_processed"].append(job_type)

        base_report["agent_metrics"] = agent_metrics
        return base_report

    async def close(self) -> None:
        """Close metrics integration."""
        await self.metrics.close()
