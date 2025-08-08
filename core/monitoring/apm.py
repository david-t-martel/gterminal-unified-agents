"""Enhanced Application Performance Monitoring (APM) System.

This module provides comprehensive APM capabilities including:
- Distributed tracing with OpenTelemetry
- Custom AI operation metrics
- Performance anomaly detection
- Resource utilization tracking
- Database query performance monitoring
"""

import asyncio
from collections import defaultdict
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
import logging
from statistics import mean
import time
from typing import Any
import uuid

# Optional OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import metrics
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # OpenTelemetry not available - APM will work in limited mode
    OPENTELEMETRY_AVAILABLE = False
    metrics = None
    trace = None
    JaegerExporter = None
    PrometheusMetricReader = None
    AioHttpClientInstrumentor = None
    AsyncioInstrumentor = None
    MeterProvider = None
    TracerProvider = None
    BatchSpanProcessor = None

logger = logging.getLogger(__name__)


class NullMetricInstrument:
    """Null object pattern for metric instruments when OpenTelemetry is not available."""

    def add(self, value: float, attributes: dict[str, str] | None = None) -> None:
        """No-op add method."""

    def record(self, value: float, attributes: dict[str, str] | None = None) -> None:
        """No-op record method."""


class NullSpan:
    """Null object pattern for spans when OpenTelemetry is not available."""

    def set_status(self, status) -> None:
        """No-op set_status method."""

    def record_exception(self, exception) -> None:
        """No-op record_exception method."""

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class NullTracer:
    """Null object pattern for tracer when OpenTelemetry is not available."""

    def start_as_current_span(self, name: str, attributes: dict[str, str] | None = None) -> None:
        """Return a null span."""
        return NullSpan()


@dataclass
class OperationMetrics:
    """Metrics for a specific operation type."""

    operation_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_ms: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_types: dict[str, int] = field(default_factory=dict)
    resource_usage: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def average_response_time_ms(self) -> float:
        """Calculate average response time in milliseconds."""
        if not self.response_times:
            return 0.0
        return mean(self.response_times)

    @property
    def p95_response_time_ms(self) -> float:
        """Calculate 95th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]


@dataclass
class AIOperationMetrics:
    """Metrics specific to AI operations."""

    model_name: str
    operation_type: str  # inference, training, embedding, etc.
    inference_count: int = 0
    total_inference_time_ms: float = 0.0
    input_token_count: int = 0
    output_token_count: int = 0
    model_accuracy_scores: list[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    queue_wait_times: list[float] = field(default_factory=list)
    memory_usage_mb: list[float] = field(default_factory=list)
    error_count: int = 0
    throttling_events: int = 0

    @property
    def average_inference_time_ms(self) -> float:
        """Calculate average inference time."""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time_ms / self.inference_count

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return (self.cache_hits / total_requests) * 100

    @property
    def average_accuracy(self) -> float:
        """Calculate average model accuracy."""
        if not self.model_accuracy_scores:
            return 0.0
        return mean(self.model_accuracy_scores)

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens processed per second."""
        if self.total_inference_time_ms == 0:
            return 0.0
        total_tokens = self.input_token_count + self.output_token_count
        total_seconds = self.total_inference_time_ms / 1000
        return total_tokens / total_seconds


class EnhancedAPMSystem:
    """Enhanced Application Performance Monitoring system."""

    def __init__(
        self,
        service_name: str = "fullstack-agent",
        jaeger_endpoint: str | None = None,
        prometheus_port: int = 8000,
        enable_auto_instrumentation: bool = True,
    ) -> None:
        self.service_name = service_name
        self.start_time = datetime.now(UTC)

        # Metrics storage
        self.operation_metrics: dict[str, OperationMetrics] = {}
        self.ai_metrics: dict[str, AIOperationMetrics] = {}
        self.custom_metrics: dict[str, list[float]] = defaultdict(list)

        # Active operations tracking
        self.active_operations: dict[str, dict[str, Any]] = {}
        self.anomaly_detectors: dict[str, AnomalyDetector] = {}

        # Performance baselines
        self.performance_baselines: dict[str, dict[str, float]] = {}

        # Initialize OpenTelemetry
        self._initialize_telemetry(jaeger_endpoint, prometheus_port)

        if enable_auto_instrumentation:
            self._enable_auto_instrumentation()

        # Create metric instruments after telemetry is initialized
        self._create_metric_instruments()

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []
        self._start_background_tasks()

    def _initialize_telemetry(self, jaeger_endpoint: str | None, prometheus_port: int) -> None:
        """Initialize OpenTelemetry tracing and metrics."""
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available - APM running in limited mode")
            self.tracer = NullTracer()
            self.meter = None
            return

        # Configure tracing
        trace_provider = TracerProvider()
        trace.set_tracer_provider(trace_provider)

        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace_provider.add_span_processor(span_processor)

        self.tracer = trace.get_tracer(__name__)

        # Configure metrics
        prometheus_reader = PrometheusMetricReader(port=prometheus_port)
        metric_provider = MeterProvider(metric_readers=[prometheus_reader])
        metrics.set_meter_provider(metric_provider)
        self.meter = metrics.get_meter(__name__)

    def _create_metric_instruments(self) -> None:
        """Create OpenTelemetry metric instruments."""
        if not OPENTELEMETRY_AVAILABLE or self.meter is None:
            # Initialize null instruments for graceful degradation
            self.request_counter = NullMetricInstrument()
            self.request_duration = NullMetricInstrument()
            self.ai_inference_counter = NullMetricInstrument()
            self.ai_inference_duration = NullMetricInstrument()
            self.ai_tokens_processed = NullMetricInstrument()
            self.ai_cache_hits = NullMetricInstrument()
            self.ai_queue_length = NullMetricInstrument()
            self.memory_usage = NullMetricInstrument()
            self.active_connections = NullMetricInstrument()
            return

        # Standard HTTP metrics
        self.request_counter = self.meter.create_counter(
            "http_requests_total",
            description="Total number of HTTP requests",
        )

        self.request_duration = self.meter.create_histogram(
            "http_request_duration_seconds",
            description="HTTP request duration in seconds",
        )

        # AI-specific metrics
        self.ai_inference_counter = self.meter.create_counter(
            "ai_inference_total",
            description="Total number of AI inferences",
        )

        self.ai_inference_duration = self.meter.create_histogram(
            "ai_inference_duration_seconds",
            description="AI inference duration in seconds",
        )

        self.ai_tokens_processed = self.meter.create_counter(
            "ai_tokens_processed_total",
            description="Total number of tokens processed",
        )

        self.ai_cache_hits = self.meter.create_counter(
            "ai_cache_hits_total",
            description="Total number of AI cache hits",
        )

        self.ai_queue_length = self.meter.create_up_down_counter(
            "ai_queue_length",
            description="Current AI operation queue length",
        )

        # Resource metrics
        self.memory_usage = self.meter.create_up_down_counter(
            "memory_usage_bytes",
            description="Current memory usage in bytes",
        )

        self.active_connections = self.meter.create_up_down_counter(
            "active_connections",
            description="Number of active connections",
        )

    def _enable_auto_instrumentation(self) -> None:
        """Enable automatic instrumentation for common libraries."""
        if not OPENTELEMETRY_AVAILABLE:
            logger.info("OpenTelemetry not available - skipping auto-instrumentation")
            return

        try:
            # Instrument aiohttp
            AioHttpClientInstrumentor().instrument()

            # Instrument asyncio
            AsyncioInstrumentor().instrument()

            logger.info("Auto-instrumentation enabled for aiohttp and asyncio")
        except Exception as e:
            logger.warning(f"Failed to enable auto-instrumentation: {e}")

    def _start_background_tasks(self) -> None:
        """Start background monitoring and cleanup tasks."""
        self._background_tasks = [
            asyncio.create_task(self._metrics_calculation_loop()),
            asyncio.create_task(self._anomaly_detection_loop()),
            asyncio.create_task(self._cleanup_old_data_loop()),
        ]

    async def _metrics_calculation_loop(self) -> None:
        """Background task to calculate aggregate metrics."""
        while True:
            try:
                await self._calculate_aggregate_metrics()
                await asyncio.sleep(30)  # Calculate every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in metrics calculation loop: {e}")
                await asyncio.sleep(30)

    async def _anomaly_detection_loop(self) -> None:
        """Background task for anomaly detection."""
        while True:
            try:
                await self._detect_performance_anomalies()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(60)

    async def _cleanup_old_data_loop(self) -> None:
        """Background task to clean up old data."""
        while True:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)

    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        operation_type: str = "generic",
        labels: dict[str, str] | None = None,
    ):
        """Context manager for tracing operations with comprehensive metrics."""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        labels = labels or {}

        # Create span
        with self.tracer.start_as_current_span(
            operation_name,
            attributes={
                "operation.type": operation_type,
                "operation.id": operation_id,
                **labels,
            },
        ) as span:
            # Track active operation
            self.active_operations[operation_id] = {
                "name": operation_name,
                "type": operation_type,
                "start_time": start_time,
                "labels": labels,
            }

            # Initialize operation metrics if needed
            if operation_name not in self.operation_metrics:
                self.operation_metrics[operation_name] = OperationMetrics(operation_name)

            operation_metrics = self.operation_metrics[operation_name]
            operation_metrics.total_calls += 1

            try:
                yield operation_id, span

                # Operation succeeded
                operation_metrics.successful_calls += 1
                if OPENTELEMETRY_AVAILABLE:
                    span.set_status(trace.Status(trace.StatusCode.OK))
                else:
                    span.set_status("OK")

            except Exception as e:
                # Operation failed
                operation_metrics.failed_calls += 1
                error_type = type(e).__name__
                operation_metrics.error_types[error_type] = (
                    operation_metrics.error_types.get(error_type, 0) + 1
                )

                # Record error in span
                span.record_exception(e)
                if OPENTELEMETRY_AVAILABLE:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                else:
                    span.set_status(f"ERROR: {e!s}")

                # Update error counter
                self.request_counter.add(
                    1, {"operation": operation_name, "status": "error", "error_type": error_type}
                )

                raise

            finally:
                # Calculate duration and update metrics
                duration_ms = (time.time() - start_time) * 1000
                operation_metrics.total_duration_ms += duration_ms
                operation_metrics.response_times.append(duration_ms)
                operation_metrics.last_updated = datetime.now(UTC)

                # Update OpenTelemetry metrics
                self.request_counter.add(1, {"operation": operation_name, "status": "success"})

                self.request_duration.record(
                    duration_ms / 1000,
                    {"operation": operation_name},  # Convert to seconds
                )

                # Clean up active operation
                self.active_operations.pop(operation_id, None)

    @asynccontextmanager
    async def trace_ai_operation(
        self,
        model_name: str,
        operation_type: str,
        input_tokens: int = 0,
        labels: dict[str, str] | None = None,
    ):
        """Context manager specifically for AI operations."""
        operation_key = f"{model_name}_{operation_type}"
        start_time = time.time()
        labels = labels or {}

        # Initialize AI metrics if needed
        if operation_key not in self.ai_metrics:
            self.ai_metrics[operation_key] = AIOperationMetrics(model_name, operation_type)

        ai_metrics = self.ai_metrics[operation_key]
        ai_metrics.inference_count += 1
        ai_metrics.input_token_count += input_tokens

        # Update queue length
        self.ai_queue_length.add(1, {"model": model_name, "operation": operation_type})

        with self.tracer.start_as_current_span(
            f"ai_{operation_type}",
            attributes={
                "ai.model.name": model_name,
                "ai.operation.type": operation_type,
                "ai.input.tokens": input_tokens,
                **labels,
            },
        ) as span:
            try:
                yield ai_metrics, span

            except Exception as e:
                ai_metrics.error_count += 1
                span.record_exception(e)
                if OPENTELEMETRY_AVAILABLE:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                else:
                    span.set_status(f"ERROR: {e!s}")
                raise

            finally:
                # Calculate duration and update metrics
                duration_ms = (time.time() - start_time) * 1000
                ai_metrics.total_inference_time_ms += duration_ms

                # Update OpenTelemetry metrics
                self.ai_inference_counter.add(1, {"model": model_name, "operation": operation_type})

                self.ai_inference_duration.record(
                    duration_ms / 1000,
                    {"model": model_name, "operation": operation_type},
                )

                # Update queue length
                self.ai_queue_length.add(-1, {"model": model_name, "operation": operation_type})

    def record_ai_result(
        self,
        model_name: str,
        operation_type: str,
        output_tokens: int = 0,
        accuracy_score: float | None = None,
        was_cached: bool = False,
        queue_wait_time_ms: float = 0,
        memory_usage_mb: float = 0,
    ) -> None:
        """Record AI operation results."""
        operation_key = f"{model_name}_{operation_type}"

        if operation_key in self.ai_metrics:
            ai_metrics = self.ai_metrics[operation_key]
            ai_metrics.output_token_count += output_tokens

            if accuracy_score is not None:
                ai_metrics.model_accuracy_scores.append(accuracy_score)

            if was_cached:
                ai_metrics.cache_hits += 1
                self.ai_cache_hits.add(1, {"model": model_name, "hit": "true"})
            else:
                ai_metrics.cache_misses += 1
                self.ai_cache_hits.add(1, {"model": model_name, "hit": "false"})

            if queue_wait_time_ms > 0:
                ai_metrics.queue_wait_times.append(queue_wait_time_ms)

            if memory_usage_mb > 0:
                ai_metrics.memory_usage_mb.append(memory_usage_mb)

            # Update token counter
            self.ai_tokens_processed.add(
                output_tokens, {"model": model_name, "token_type": "output"}
            )

    def record_custom_metric(
        self, metric_name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a custom metric value."""
        self.custom_metrics[metric_name].append(value)

        # Also record in OpenTelemetry if available
        if not OPENTELEMETRY_AVAILABLE or self.meter is None:
            return

        try:
            if not hasattr(self, f"custom_{metric_name}"):
                # Create histogram dynamically
                histogram = self.meter.create_histogram(
                    f"custom_{metric_name}",
                    description=f"Custom metric: {metric_name}",
                )
                setattr(self, f"custom_{metric_name}", histogram)

            histogram = getattr(self, f"custom_{metric_name}")
            histogram.record(value, labels or {})
        except Exception as e:
            logger.warning(f"Failed to record custom metric {metric_name}: {e}")

    async def _calculate_aggregate_metrics(self) -> None:
        """Calculate aggregate metrics across all operations."""
        try:
            # Calculate overall system metrics
            total_operations = sum(op.total_calls for op in self.operation_metrics.values())
            total_errors = sum(op.failed_calls for op in self.operation_metrics.values())

            if total_operations > 0:
                overall_error_rate = (total_errors / total_operations) * 100
                self.record_custom_metric("overall_error_rate", overall_error_rate)

            # Calculate AI-specific aggregates
            total_ai_operations = sum(ai.inference_count for ai in self.ai_metrics.values())
            total_ai_errors = sum(ai.error_count for ai in self.ai_metrics.values())

            if total_ai_operations > 0:
                ai_error_rate = (total_ai_errors / total_ai_operations) * 100
                self.record_custom_metric("ai_error_rate", ai_error_rate)

                # Calculate average AI metrics
                avg_inference_time = mean(
                    [
                        ai.average_inference_time_ms
                        for ai in self.ai_metrics.values()
                        if ai.inference_count > 0
                    ],
                )
                self.record_custom_metric("avg_ai_inference_time_ms", avg_inference_time)

                avg_cache_hit_rate = mean(
                    [
                        ai.cache_hit_rate
                        for ai in self.ai_metrics.values()
                        if ai.cache_hits + ai.cache_misses > 0
                    ],
                )
                self.record_custom_metric("avg_ai_cache_hit_rate", avg_cache_hit_rate)

        except Exception as e:
            logger.exception(f"Error calculating aggregate metrics: {e}")

    async def _detect_performance_anomalies(self) -> None:
        """Detect performance anomalies using statistical analysis."""
        try:
            for operation_name, metrics in self.operation_metrics.items():
                if len(metrics.response_times) < 10:  # Need minimum data points
                    continue

                # Get anomaly detector for this operation
                if operation_name not in self.anomaly_detectors:
                    self.anomaly_detectors[operation_name] = AnomalyDetector(operation_name)

                detector = self.anomaly_detectors[operation_name]

                # Check for response time anomalies
                recent_times = list(metrics.response_times)[-50:]  # Last 50 operations
                anomalies = detector.detect_anomalies(recent_times)

                if anomalies:
                    logger.warning(
                        f"Performance anomaly detected in {operation_name}: {len(anomalies)} anomalous response times",
                    )

                    # Record anomaly metric
                    self.record_custom_metric(
                        f"anomalies_{operation_name}",
                        len(anomalies),
                        {"operation": operation_name},
                    )

        except Exception as e:
            logger.exception(f"Error in anomaly detection: {e}")

    async def _cleanup_old_data(self) -> None:
        """Clean up old performance data to prevent memory leaks."""
        try:
            cutoff_time = datetime.now(UTC) - timedelta(hours=24)

            # Clean up old operation metrics
            for metrics in self.operation_metrics.values():
                if metrics.last_updated < cutoff_time:
                    # Keep only recent response times
                    metrics.response_times = deque(
                        list(metrics.response_times)[-100:],
                        maxlen=1000,  # Keep last 100
                    )

            # Clean up custom metrics (keep last 1000 values)
            for metric_name in self.custom_metrics:
                if len(self.custom_metrics[metric_name]) > 1000:
                    self.custom_metrics[metric_name] = self.custom_metrics[metric_name][-1000:]

            logger.debug("Completed cleanup of old performance data")

        except Exception as e:
            logger.exception(f"Error cleaning up old data: {e}")

    def get_operation_summary(self, operation_name: str | None = None) -> dict[str, Any]:
        """Get summary of operation metrics."""
        if operation_name and operation_name in self.operation_metrics:
            metrics = self.operation_metrics[operation_name]
            return {
                "operation_name": metrics.operation_name,
                "total_calls": metrics.total_calls,
                "success_rate": metrics.success_rate,
                "average_response_time_ms": metrics.average_response_time_ms,
                "p95_response_time_ms": metrics.p95_response_time_ms,
                "error_types": dict(metrics.error_types),
                "last_updated": metrics.last_updated.isoformat(),
            }

        # Return summary of all operations
        return {
            operation_name: {
                "total_calls": metrics.total_calls,
                "success_rate": metrics.success_rate,
                "average_response_time_ms": metrics.average_response_time_ms,
                "p95_response_time_ms": metrics.p95_response_time_ms,
            }
            for operation_name, metrics in self.operation_metrics.items()
        }

    def get_ai_metrics_summary(self, model_name: str | None = None) -> dict[str, Any]:
        """Get summary of AI operation metrics."""
        if model_name:
            # Filter metrics for specific model
            relevant_metrics = {
                key: metrics
                for key, metrics in self.ai_metrics.items()
                if metrics.model_name == model_name
            }
        else:
            relevant_metrics = self.ai_metrics

        return {
            key: {
                "model_name": metrics.model_name,
                "operation_type": metrics.operation_type,
                "inference_count": metrics.inference_count,
                "average_inference_time_ms": metrics.average_inference_time_ms,
                "cache_hit_rate": metrics.cache_hit_rate,
                "average_accuracy": metrics.average_accuracy,
                "tokens_per_second": metrics.tokens_per_second,
                "error_count": metrics.error_count,
                "throttling_events": metrics.throttling_events,
            }
            for key, metrics in relevant_metrics.items()
        }

    def get_health_status(self) -> dict[str, Any]:
        """Get overall system health status."""
        now = datetime.now(UTC)
        uptime = now - self.start_time

        # Calculate overall metrics
        total_operations = sum(op.total_calls for op in self.operation_metrics.values())
        total_errors = sum(op.failed_calls for op in self.operation_metrics.values())
        error_rate = (total_errors / total_operations * 100) if total_operations > 0 else 0

        # Determine health status
        if error_rate > 5:  # >5% error rate
            health_status = "unhealthy"
        elif error_rate > 1:  # >1% error rate
            health_status = "degraded"
        else:
            health_status = "healthy"

        # Check for active anomalies
        active_anomalies = sum(
            len(detector.recent_anomalies) for detector in self.anomaly_detectors.values()
        )

        return {
            "status": health_status,
            "uptime_seconds": uptime.total_seconds(),
            "total_operations": total_operations,
            "error_rate_percent": error_rate,
            "active_operations": len(self.active_operations),
            "active_anomalies": active_anomalies,
            "ai_models_monitored": len(
                {metrics.model_name for metrics in self.ai_metrics.values()}
            ),
            "timestamp": now.isoformat(),
        }

    async def close(self) -> None:
        """Clean shutdown of APM system."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("APM system shutdown complete")


class AnomalyDetector:
    """Statistical anomaly detector for performance metrics."""

    def __init__(self, metric_name: str, window_size: int = 100) -> None:
        self.metric_name = metric_name
        self.window_size = window_size
        self.baseline_values = deque(maxlen=window_size)
        self.recent_anomalies = deque(maxlen=50)  # Keep last 50 anomalies

    def detect_anomalies(self, values: list[float], threshold: float = 2.0) -> list[int]:
        """Detect anomalies using statistical analysis."""
        if len(values) < 10:
            return []

        # Update baseline
        self.baseline_values.extend(values[:-10])  # Use all but last 10 for baseline

        if len(self.baseline_values) < 20:
            return []

        # Calculate baseline statistics
        baseline_mean = mean(self.baseline_values)
        baseline_std = self._calculate_std(self.baseline_values, baseline_mean)

        if baseline_std == 0:
            return []

        # Check recent values for anomalies
        anomalies: list[Any] = []
        recent_values = values[-10:]  # Check last 10 values

        for i, value in enumerate(recent_values):
            z_score = abs(value - baseline_mean) / baseline_std
            if z_score > threshold:
                anomalies.append(len(values) - 10 + i)
                self.recent_anomalies.append(
                    {
                        "timestamp": datetime.now(UTC),
                        "value": value,
                        "z_score": z_score,
                        "baseline_mean": baseline_mean,
                        "baseline_std": baseline_std,
                    },
                )

        return anomalies

    def _calculate_std(self, values: list[float], mean_val: float) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5


# Export main APM class
__all__ = ["AIOperationMetrics", "AnomalyDetector", "EnhancedAPMSystem", "OperationMetrics"]
