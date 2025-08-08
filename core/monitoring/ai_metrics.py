"""AI-Specific Performance Metrics System.

This module provides comprehensive monitoring for AI operations including:
- Model inference performance tracking
- Token usage and cost monitoring
- Model accuracy and quality metrics
- AI pipeline bottleneck detection
- Model switching and fallback tracking
- Context management performance
"""

import asyncio
from collections import defaultdict
from collections import deque
from dataclasses import asdict
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


class ModelProvider(Enum):
    """AI model providers."""

    GOOGLE_VERTEX = "google_vertex"
    GOOGLE_AI = "google_ai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HUGGING_FACE = "hugging_face"


class OperationType(Enum):
    """Types of AI operations."""

    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"
    VIDEO_GENERATION = "video_generation"
    AUDIO_GENERATION = "audio_generation"
    CODE_GENERATION = "code_generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"


class QualityMetricType(Enum):
    """Types of quality metrics."""

    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"
    FACTUALITY = "factuality"
    CREATIVITY = "creativity"


@dataclass
class ModelConfiguration:
    """Configuration for an AI model."""

    model_name: str
    provider: ModelProvider
    version: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    max_tokens: int = 0
    context_window: int = 0
    supports_streaming: bool = False
    supports_function_calling: bool = False


@dataclass
class AIOperationMetrics:
    """Comprehensive metrics for an AI operation."""

    operation_id: str
    session_id: str
    model_config: ModelConfiguration
    operation_type: OperationType
    start_time: datetime
    end_time: datetime | None = None

    # Performance metrics
    total_duration_ms: float = 0
    queue_wait_time_ms: float = 0
    inference_time_ms: float = 0
    network_latency_ms: float = 0

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    # Cost metrics
    estimated_cost: float = 0.0

    # Quality metrics
    quality_scores: dict[QualityMetricType, float] = field(default_factory=dict)

    # Context metrics
    context_size_tokens: int = 0
    context_utilization: float = 0.0  # Percentage of context window used

    # Pipeline metrics
    preprocessing_time_ms: float = 0
    postprocessing_time_ms: float = 0

    # Error and retry metrics
    success: bool = True
    error_type: str = ""
    error_message: str = ""
    retry_count: int = 0
    fallback_used: bool = False
    fallback_model: str = ""

    # Streaming metrics (if applicable)
    first_token_latency_ms: float = 0
    streaming_chunks: int = 0

    # Custom metrics
    custom_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.inference_time_ms == 0:
            return 0.0
        total_tokens = self.input_tokens + self.output_tokens
        return total_tokens / (self.inference_time_ms / 1000)

    @property
    def cost_per_token(self) -> float:
        """Calculate cost per token."""
        total_tokens = self.input_tokens + self.output_tokens
        if total_tokens == 0:
            return 0.0
        return self.estimated_cost / total_tokens


@dataclass
class ModelPerformanceProfile:
    """Performance profile for a specific model."""

    model_name: str
    provider: ModelProvider
    total_operations: int = 0
    successful_operations: int = 0

    # Performance statistics
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    token_throughput: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Quality statistics
    quality_scores: dict[QualityMetricType, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Cost statistics
    total_cost: float = 0.0
    total_tokens: int = 0

    # Error statistics
    error_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Context utilization
    context_utilizations: list[float] = field(default_factory=list)

    # Last updated
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100

    @property
    def average_response_time_ms(self) -> float:
        """Calculate average response time."""
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

    @property
    def average_cost_per_operation(self) -> float:
        """Calculate average cost per operation."""
        if self.successful_operations == 0:
            return 0.0
        return self.total_cost / self.successful_operations

    @property
    def average_quality_score(self) -> float:
        """Calculate average quality score across all metrics."""
        all_scores: list[Any] = []
        for scores_list in self.quality_scores.values():
            all_scores.extend(scores_list)
        return mean(all_scores) if all_scores else 0.0

    @property
    def average_context_utilization(self) -> float:
        """Calculate average context utilization."""
        if not self.context_utilizations:
            return 0.0
        return mean(self.context_utilizations)


@dataclass
class AISystemMetrics:
    """System-wide AI metrics."""

    total_operations: int = 0
    operations_per_minute: float = 0.0
    active_models: int = 0
    queue_depth: int = 0

    # Performance metrics
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    total_tokens_processed: int = 0
    tokens_per_second: float = 0.0

    # Cost metrics
    total_cost_today: float = 0.0
    cost_per_hour: float = 0.0

    # Quality metrics
    overall_quality_score: float = 0.0

    # Error metrics
    error_rate: float = 0.0
    fallback_rate: float = 0.0

    # Efficiency metrics
    cache_hit_rate: float = 0.0
    context_efficiency: float = 0.0


class AIPerformanceMonitor:
    """Comprehensive AI performance monitoring system."""

    def __init__(self, retention_days: int = 7) -> None:
        self.retention_days = retention_days
        self.start_time = datetime.now(UTC)

        # Model configurations
        self.model_configs: dict[str, ModelConfiguration] = {}

        # Active operations
        self.active_operations: dict[str, AIOperationMetrics] = {}

        # Completed operations (for analysis)
        self.completed_operations: deque = deque(maxlen=10000)

        # Model performance profiles
        self.model_profiles: dict[str, ModelPerformanceProfile] = {}

        # Queue monitoring
        self.operation_queue: dict[str, list[str]] = defaultdict(list)  # model -> operation_ids

        # System metrics
        self.system_metrics = AISystemMetrics()

        # Real-time metrics
        self.real_time_metrics = {
            "operations_per_minute": deque(maxlen=60),
            "errors_per_minute": deque(maxlen=60),
            "cost_per_minute": deque(maxlen=60),
            "quality_scores": deque(maxlen=100),
        }

        # Alerting thresholds
        self.alert_thresholds = {
            "high_latency_ms": 10000,  # 10 seconds
            "high_error_rate": 5.0,  # 5%
            "high_cost_per_hour": 50.0,  # $50/hour
            "low_quality_score": 0.6,  # 60%
            "high_queue_depth": 50,  # 50 operations
        }

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []
        self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        self._background_tasks = [
            asyncio.create_task(self._calculate_system_metrics()),
            asyncio.create_task(self._monitor_queues()),
            asyncio.create_task(self._detect_anomalies()),
            asyncio.create_task(self._cleanup_old_data()),
        ]

    def register_model(self, config: ModelConfiguration) -> None:
        """Register a model configuration."""
        self.model_configs[config.model_name] = config

        if config.model_name not in self.model_profiles:
            self.model_profiles[config.model_name] = ModelPerformanceProfile(
                model_name=config.model_name,
                provider=config.provider,
            )

        logger.info(f"Registered model: {config.model_name} ({config.provider.value})")

    def start_operation(
        self,
        session_id: str,
        model_name: str,
        operation_type: OperationType,
        input_tokens: int = 0,
        context_size_tokens: int = 0,
        custom_metrics: dict[str, float] | None = None,
    ) -> str:
        """Start tracking an AI operation."""
        operation_id = str(uuid.uuid4())

        if model_name not in self.model_configs:
            logger.warning(f"Model {model_name} not registered, using default config")
            config = ModelConfiguration(model_name=model_name, provider=ModelProvider.GOOGLE_VERTEX)
        else:
            config = self.model_configs[model_name]

        operation = AIOperationMetrics(
            operation_id=operation_id,
            session_id=session_id,
            model_config=config,
            operation_type=operation_type,
            start_time=datetime.now(UTC),
            input_tokens=input_tokens,
            context_size_tokens=context_size_tokens,
            custom_metrics=custom_metrics or {},
        )

        # Calculate context utilization
        if config.context_window > 0:
            operation.context_utilization = context_size_tokens / config.context_window

        self.active_operations[operation_id] = operation

        # Add to queue for monitoring
        self.operation_queue[model_name].append(operation_id)

        logger.debug(f"Started AI operation: {operation_id} ({model_name}, {operation_type.value})")
        return operation_id

    def end_operation(
        self,
        operation_id: str,
        output_tokens: int = 0,
        quality_scores: dict[QualityMetricType, float] | None = None,
        success: bool = True,
        error_type: str = "",
        error_message: str = "",
        fallback_used: bool = False,
        fallback_model: str = "",
        custom_metrics: dict[str, float] | None = None,
    ) -> None:
        """End an AI operation and record metrics."""
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found in active operations")
            return

        operation = self.active_operations[operation_id]
        operation.end_time = datetime.now(UTC)
        operation.output_tokens = output_tokens
        operation.quality_scores = quality_scores or {}
        operation.success = success
        operation.error_type = error_type
        operation.error_message = error_message
        operation.fallback_used = fallback_used
        operation.fallback_model = fallback_model

        if custom_metrics:
            operation.custom_metrics.update(custom_metrics)

        # Calculate durations
        if operation.start_time and operation.end_time:
            operation.total_duration_ms = (
                operation.end_time - operation.start_time
            ).total_seconds() * 1000

        # Calculate cost
        config = operation.model_config
        operation.estimated_cost = (
            operation.input_tokens * config.cost_per_input_token
            + operation.output_tokens * config.cost_per_output_token
        ) / 1000  # Assuming cost is per 1K tokens

        # Update model profile
        self._update_model_profile(operation)

        # Move to completed operations
        self.completed_operations.append(operation)
        del self.active_operations[operation_id]

        # Remove from queue
        model_name = operation.model_config.model_name
        if operation_id in self.operation_queue[model_name]:
            self.operation_queue[model_name].remove(operation_id)

        logger.debug(
            f"Completed AI operation: {operation_id} ({operation.total_duration_ms:.1f}ms)"
        )

    def record_queue_wait_time(self, operation_id: str, wait_time_ms: float) -> None:
        """Record queue wait time for an operation."""
        if operation_id in self.active_operations:
            self.active_operations[operation_id].queue_wait_time_ms = wait_time_ms

    def record_streaming_metrics(
        self,
        operation_id: str,
        first_token_latency_ms: float,
        chunks_count: int,
    ) -> None:
        """Record streaming-specific metrics."""
        if operation_id in self.active_operations:
            operation = self.active_operations[operation_id]
            operation.first_token_latency_ms = first_token_latency_ms
            operation.streaming_chunks = chunks_count

    def record_pipeline_metrics(
        self,
        operation_id: str,
        preprocessing_time_ms: float = 0,
        postprocessing_time_ms: float = 0,
        network_latency_ms: float = 0,
    ) -> None:
        """Record pipeline-specific metrics."""
        if operation_id in self.active_operations:
            operation = self.active_operations[operation_id]
            operation.preprocessing_time_ms = preprocessing_time_ms
            operation.postprocessing_time_ms = postprocessing_time_ms
            operation.network_latency_ms = network_latency_ms

            # Calculate inference time (excluding pre/post processing)
            if operation.total_duration_ms > 0:
                operation.inference_time_ms = max(
                    0,
                    operation.total_duration_ms - preprocessing_time_ms - postprocessing_time_ms,
                )

    def _update_model_profile(self, operation: AIOperationMetrics) -> None:
        """Update model performance profile with operation data."""
        model_name = operation.model_config.model_name

        if model_name not in self.model_profiles:
            self.model_profiles[model_name] = ModelPerformanceProfile(
                model_name=model_name,
                provider=operation.model_config.provider,
            )

        profile = self.model_profiles[model_name]
        profile.total_operations += 1
        profile.last_updated = datetime.now(UTC)

        if operation.success:
            profile.successful_operations += 1

            # Update performance metrics
            if operation.total_duration_ms > 0:
                profile.response_times.append(operation.total_duration_ms)

            if operation.tokens_per_second > 0:
                profile.token_throughput.append(operation.tokens_per_second)

            # Update cost metrics
            profile.total_cost += operation.estimated_cost
            profile.total_tokens += operation.input_tokens + operation.output_tokens

            # Update quality scores
            for quality_type, score in operation.quality_scores.items():
                profile.quality_scores[quality_type].append(score)

            # Update context utilization
            if operation.context_utilization > 0:
                profile.context_utilizations.append(operation.context_utilization)
        else:
            # Update error statistics
            error_key = operation.error_type or "unknown"
            profile.error_counts[error_key] += 1

    async def _calculate_system_metrics(self) -> None:
        """Calculate system-wide metrics."""
        while True:
            try:
                await self._update_system_metrics()
                await asyncio.sleep(60)  # Update every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error calculating system metrics: {e}")
                await asyncio.sleep(60)

    async def _update_system_metrics(self) -> None:
        """Update system-wide metrics."""
        try:
            # Calculate operations per minute
            now = datetime.now(UTC)
            one_minute_ago = now - timedelta(minutes=1)

            recent_operations = [
                op
                for op in self.completed_operations
                if op.end_time and op.end_time >= one_minute_ago
            ]

            ops_per_minute = len(recent_operations)
            self.real_time_metrics["operations_per_minute"].append(ops_per_minute)
            self.system_metrics.operations_per_minute = ops_per_minute

            # Calculate error rate
            if recent_operations:
                errors = len([op for op in recent_operations if not op.success])
                error_rate = (errors / len(recent_operations)) * 100
                self.real_time_metrics["errors_per_minute"].append(errors)
                self.system_metrics.error_rate = error_rate

            # Calculate cost metrics
            recent_cost = sum(op.estimated_cost for op in recent_operations)
            self.real_time_metrics["cost_per_minute"].append(recent_cost)
            self.system_metrics.cost_per_hour = recent_cost * 60

            # Calculate today's total cost
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_operations = [
                op for op in self.completed_operations if op.end_time and op.end_time >= today_start
            ]
            self.system_metrics.total_cost_today = sum(op.estimated_cost for op in today_operations)

            # Calculate average quality score
            recent_quality_scores: list[Any] = []
            for op in recent_operations:
                if op.quality_scores:
                    recent_quality_scores.extend(op.quality_scores.values())

            if recent_quality_scores:
                avg_quality = mean(recent_quality_scores)
                self.real_time_metrics["quality_scores"].append(avg_quality)
                self.system_metrics.overall_quality_score = avg_quality

            # Update other system metrics
            self.system_metrics.total_operations = len(self.completed_operations)
            self.system_metrics.active_models = len(self.model_profiles)
            self.system_metrics.queue_depth = sum(
                len(queue) for queue in self.operation_queue.values()
            )

            # Calculate performance metrics
            if recent_operations:
                response_times = [
                    op.total_duration_ms for op in recent_operations if op.total_duration_ms > 0
                ]
                if response_times:
                    self.system_metrics.average_response_time_ms = mean(response_times)
                    sorted_times = sorted(response_times)
                    p95_index = int(len(sorted_times) * 0.95)
                    self.system_metrics.p95_response_time_ms = (
                        sorted_times[p95_index]
                        if p95_index < len(sorted_times)
                        else sorted_times[-1]
                    )

                # Calculate tokens per second
                total_tokens = sum(op.input_tokens + op.output_tokens for op in recent_operations)
                total_time_s = sum(op.total_duration_ms for op in recent_operations) / 1000
                if total_time_s > 0:
                    self.system_metrics.tokens_per_second = total_tokens / total_time_s

                self.system_metrics.total_tokens_processed = total_tokens

            # Calculate fallback rate
            fallback_operations = len([op for op in recent_operations if op.fallback_used])
            if recent_operations:
                self.system_metrics.fallback_rate = (
                    fallback_operations / len(recent_operations)
                ) * 100

        except Exception as e:
            logger.exception(f"Error updating system metrics: {e}")

    async def _monitor_queues(self) -> None:
        """Monitor operation queues."""
        while True:
            try:
                for model_name, queue in self.operation_queue.items():
                    if len(queue) > self.alert_thresholds["high_queue_depth"]:
                        logger.warning(
                            f"High queue depth for {model_name}: {len(queue)} operations"
                        )

                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error monitoring queues: {e}")
                await asyncio.sleep(30)

    async def _detect_anomalies(self) -> None:
        """Detect performance anomalies."""
        while True:
            try:
                # Check for high latency
                if (
                    self.system_metrics.p95_response_time_ms
                    > self.alert_thresholds["high_latency_ms"]
                ):
                    logger.warning(
                        f"High latency detected: {self.system_metrics.p95_response_time_ms:.1f}ms"
                    )

                # Check for high error rate
                if self.system_metrics.error_rate > self.alert_thresholds["high_error_rate"]:
                    logger.warning(
                        f"High error rate detected: {self.system_metrics.error_rate:.1f}%"
                    )

                # Check for high cost
                if self.system_metrics.cost_per_hour > self.alert_thresholds["high_cost_per_hour"]:
                    logger.warning(
                        f"High cost rate detected: ${self.system_metrics.cost_per_hour:.2f}/hour"
                    )

                # Check for low quality
                if (
                    self.system_metrics.overall_quality_score
                    < self.alert_thresholds["low_quality_score"]
                ):
                    logger.warning(
                        f"Low quality score detected: {self.system_metrics.overall_quality_score:.2f}"
                    )

                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error detecting anomalies: {e}")
                await asyncio.sleep(300)

    async def _cleanup_old_data(self) -> None:
        """Clean up old performance data."""
        while True:
            try:
                cutoff_date = datetime.now(UTC) - timedelta(days=self.retention_days)

                # Clean up completed operations
                operations_to_keep: list[Any] = []
                for op in self.completed_operations:
                    if op.end_time and op.end_time >= cutoff_date:
                        operations_to_keep.append(op)

                removed_count = len(self.completed_operations) - len(operations_to_keep)
                if removed_count > 0:
                    self.completed_operations.clear()
                    self.completed_operations.extend(operations_to_keep)
                    logger.info(f"Cleaned up {removed_count} old AI operations")

                await asyncio.sleep(3600)  # Clean up every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error cleaning up old data: {e}")
                await asyncio.sleep(3600)

    def get_model_comparison(self) -> dict[str, Any]:
        """Get comparison of model performance."""
        comparison: dict[str, Any] = {}

        for model_name, profile in self.model_profiles.items():
            comparison[model_name] = {
                "provider": profile.provider.value,
                "total_operations": profile.total_operations,
                "success_rate": profile.success_rate,
                "avg_response_time_ms": profile.average_response_time_ms,
                "p95_response_time_ms": profile.p95_response_time_ms,
                "avg_cost_per_operation": profile.average_cost_per_operation,
                "avg_quality_score": profile.average_quality_score,
                "avg_context_utilization": profile.average_context_utilization,
                "total_cost": profile.total_cost,
                "total_tokens": profile.total_tokens,
            }

        return comparison

    def get_operation_analytics(
        self, operation_type: OperationType | None = None
    ) -> dict[str, Any]:
        """Get analytics for specific operation types."""
        if operation_type:
            operations = [
                op for op in self.completed_operations if op.operation_type == operation_type
            ]
        else:
            operations = list(self.completed_operations)

        if not operations:
            return {"message": "No operations found"}

        # Calculate analytics
        successful_ops = [op for op in operations if op.success]

        response_times = [op.total_duration_ms for op in successful_ops if op.total_duration_ms > 0]
        costs = [op.estimated_cost for op in successful_ops]
        quality_scores: list[Any] = []
        for op in successful_ops:
            quality_scores.extend(op.quality_scores.values())

        analytics = {
            "operation_type": operation_type.value if operation_type else "all",
            "total_operations": len(operations),
            "successful_operations": len(successful_ops),
            "success_rate": (len(successful_ops) / len(operations)) * 100,
        }

        if response_times:
            sorted_times = sorted(response_times)
            analytics["response_time_ms"] = {
                "min": min(response_times),
                "max": max(response_times),
                "mean": mean(response_times),
                "median": median(response_times),
                "p95": sorted_times[int(len(sorted_times) * 0.95)],
                "p99": sorted_times[int(len(sorted_times) * 0.99)],
            }

        if costs:
            analytics["cost"] = {
                "total": sum(costs),
                "mean": mean(costs),
                "min": min(costs),
                "max": max(costs),
            }

        if quality_scores:
            analytics["quality_score"] = {
                "mean": mean(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores),
            }

        return analytics

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health status."""
        # Determine health status
        health_issues: list[Any] = []

        if self.system_metrics.error_rate > self.alert_thresholds["high_error_rate"]:
            health_issues.append(f"High error rate: {self.system_metrics.error_rate:.1f}%")

        if self.system_metrics.p95_response_time_ms > self.alert_thresholds["high_latency_ms"]:
            health_issues.append(f"High latency: {self.system_metrics.p95_response_time_ms:.1f}ms")

        if self.system_metrics.cost_per_hour > self.alert_thresholds["high_cost_per_hour"]:
            health_issues.append(f"High cost: ${self.system_metrics.cost_per_hour:.2f}/hour")

        if self.system_metrics.overall_quality_score < self.alert_thresholds["low_quality_score"]:
            health_issues.append(f"Low quality: {self.system_metrics.overall_quality_score:.2f}")

        if self.system_metrics.queue_depth > self.alert_thresholds["high_queue_depth"]:
            health_issues.append(f"High queue depth: {self.system_metrics.queue_depth}")

        health_status = (
            "healthy"
            if not health_issues
            else "degraded"
            if len(health_issues) <= 2
            else "unhealthy"
        )

        return {
            "status": health_status,
            "issues": health_issues,
            "system_metrics": asdict(self.system_metrics),
            "active_operations": len(self.active_operations),
            "uptime_hours": (datetime.now(UTC) - self.start_time).total_seconds() / 3600,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def close(self) -> None:
        """Clean shutdown of AI monitoring system."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("AI performance monitoring shutdown complete")


# Export main classes
__all__ = [
    "AIOperationMetrics",
    "AIPerformanceMonitor",
    "AISystemMetrics",
    "ModelConfiguration",
    "ModelPerformanceProfile",
    "ModelProvider",
    "OperationType",
    "QualityMetricType",
]
