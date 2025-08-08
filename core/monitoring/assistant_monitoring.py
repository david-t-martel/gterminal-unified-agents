# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monitoring and analytics system for AI assistant performance tracking."""

import asyncio
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
import json
import logging
import time
from typing import Any

from google.cloud import logging as cloud_logging
from opentelemetry import metrics
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


class MetricType(Enum):
    """Types of metrics to track."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class ConversationMetrics:
    """Metrics for a single conversation."""

    session_id: str
    start_time: datetime
    end_time: datetime | None = None
    message_count: int = 0
    turn_count: int = 0
    intents_detected: list[str] = field(default_factory=list)
    entities_extracted: list[str] = field(default_factory=list)
    sentiment_scores: list[float] = field(default_factory=list)
    response_times: list[float] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    user_satisfaction: float | None = None
    task_completed: bool = False
    escalated: bool = False


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""

    active_sessions: int = 0
    total_messages: int = 0
    messages_per_second: float = 0.0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    intent_accuracy: float = 0.0
    entity_precision: float = 0.0
    fallback_rate: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    model_inference_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


class AssistantAnalytics:
    """Analytics system for AI assistant monitoring."""

    def __init__(self, project_id: str | None = None) -> None:
        self.project_id = project_id
        self.logger = logging.getLogger(__name__)

        # Initialize metrics storage
        self.conversation_metrics: dict[str, ConversationMetrics] = {}
        self.system_metrics = SystemMetrics()
        self.metric_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1000))

        # Initialize OpenTelemetry
        self._initialize_telemetry()

        # Initialize Cloud Logging
        if project_id:
            self.cloud_logger = cloud_logging.Client(project=project_id).logger("ai-assistant")
        else:
            self.cloud_logger = None

        # Start background tasks
        self._start_background_tasks()

    def _initialize_telemetry(self) -> None:
        """Initialize OpenTelemetry instrumentation."""
        # Set up tracing
        trace.set_tracer_provider(TracerProvider())
        if self.project_id:
            trace.get_tracer_provider().add_span_processor(
                BatchSpanProcessor(CloudTraceSpanExporter())
            )
        self.tracer = trace.get_tracer(__name__)

        # Set up metrics
        reader = PeriodicExportingMetricReader(
            exporter=self._create_metrics_exporter(),
            export_interval_millis=60000,  # Export every minute
        )
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        self.meter = metrics.get_meter(__name__)

        # Create metric instruments
        self._create_metric_instruments()

    def _create_metrics_exporter(self) -> None:
        """Create metrics exporter based on environment."""
        # In production, use Google Cloud Monitoring
        # For now, use console exporter
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

        return ConsoleMetricExporter()

    def _create_metric_instruments(self) -> None:
        """Create OpenTelemetry metric instruments."""
        # Counters
        self.message_counter = self.meter.create_counter(
            "assistant.messages.total",
            description="Total number of messages processed",
        )

        self.error_counter = self.meter.create_counter(
            "assistant.errors.total", description="Total number of errors"
        )

        # Gauges
        self.active_sessions_gauge = self.meter.create_observable_gauge(
            "assistant.sessions.active",
            callbacks=[lambda: self.system_metrics.active_sessions],
            description="Number of active sessions",
        )

        # Histograms
        self.response_time_histogram = self.meter.create_histogram(
            "assistant.response.time",
            description="Response time in milliseconds",
            unit="ms",
        )

        self.intent_confidence_histogram = self.meter.create_histogram(
            "assistant.intent.confidence",
            description="Intent detection confidence scores",
        )

    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        self._metrics_task = asyncio.create_task(self._calculate_metrics_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_old_sessions())

    async def _calculate_metrics_loop(self) -> None:
        """Background loop to calculate aggregate metrics."""
        while True:
            try:
                await self._calculate_system_metrics()
                await asyncio.sleep(10)  # Calculate every 10 seconds
            except Exception as e:
                self.logger.exception(f"Error calculating metrics: {e}")

    async def _cleanup_old_sessions(self) -> None:
        """Clean up old session data."""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)
                old_sessions = [
                    sid
                    for sid, metrics in self.conversation_metrics.items()
                    if metrics.start_time < cutoff_time
                ]

                for session_id in old_sessions:
                    del self.conversation_metrics[session_id]

                await asyncio.sleep(3600)  # Clean up every hour
            except Exception as e:
                self.logger.exception(f"Error cleaning up sessions: {e}")

    async def _calculate_system_metrics(self) -> None:
        """Calculate system-wide metrics."""
        if not self.conversation_metrics:
            return

        # Active sessions
        self.system_metrics.active_sessions = sum(
            1 for m in self.conversation_metrics.values() if m.end_time is None
        )

        # Response times
        all_response_times: list[float] = []
        for conv_metrics in self.conversation_metrics.values():
            all_response_times.extend(conv_metrics.response_times)

        if all_response_times:
            sorted_times = sorted(all_response_times)
            self.system_metrics.avg_response_time = sum(sorted_times) / len(sorted_times)
            self.system_metrics.p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
            self.system_metrics.p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]

        # Error rate
        total_messages = sum(m.message_count for m in self.conversation_metrics.values())
        total_errors = sum(len(m.errors) for m in self.conversation_metrics.values())
        self.system_metrics.error_rate = total_errors / total_messages if total_messages > 0 else 0

        # Fallback rate
        fallback_intents = sum(
            1
            for m in self.conversation_metrics.values()
            for intent in m.intents_detected
            if intent == "unknown"
        )
        total_intents = sum(len(m.intents_detected) for m in self.conversation_metrics.values())
        self.system_metrics.fallback_rate = (
            fallback_intents / total_intents if total_intents > 0 else 0
        )

    def start_conversation(self, session_id: str) -> ConversationMetrics:
        """Start tracking a new conversation."""
        metrics = ConversationMetrics(session_id=session_id, start_time=datetime.now())
        self.conversation_metrics[session_id] = metrics

        # Log event
        self._log_event("conversation_started", {"session_id": session_id})

        return metrics

    def end_conversation(self, session_id: str, user_satisfaction: float | None = None) -> None:
        """End conversation tracking."""
        if session_id in self.conversation_metrics:
            metrics = self.conversation_metrics[session_id]
            metrics.end_time = datetime.now()
            metrics.user_satisfaction = user_satisfaction

            # Log conversation summary
            duration = (metrics.end_time - metrics.start_time).total_seconds()
            self._log_event(
                "conversation_ended",
                {
                    "session_id": session_id,
                    "duration": duration,
                    "message_count": metrics.message_count,
                    "task_completed": metrics.task_completed,
                    "user_satisfaction": user_satisfaction,
                },
            )

    def track_message(self, session_id: str, message_data: dict[str, Any]) -> None:
        """Track a message in the conversation."""
        if session_id not in self.conversation_metrics:
            self.start_conversation(session_id)

        metrics = self.conversation_metrics[session_id]
        metrics.message_count += 1

        # Track intent
        if "intent" in message_data:
            metrics.intents_detected.append(message_data["intent"])

            # Track confidence
            if "confidence" in message_data:
                self.intent_confidence_histogram.record(
                    message_data["confidence"], {"intent": message_data["intent"]}
                )

        # Track entities
        if "entities" in message_data:
            for entity in message_data["entities"]:
                metrics.entities_extracted.append(entity.get("type", "unknown"))

        # Track sentiment
        if "sentiment" in message_data and "score" in message_data["sentiment"]:
            metrics.sentiment_scores.append(message_data["sentiment"]["score"])

        # Track response time
        if "response_time" in message_data:
            response_time_ms = message_data["response_time"] * 1000
            metrics.response_times.append(response_time_ms)
            self.response_time_histogram.record(response_time_ms, {"session_id": session_id})

        # Increment counter
        self.message_counter.add(1, {"session_id": session_id})

    def track_error(self, session_id: str, error_data: dict[str, Any]) -> None:
        """Track an error in the conversation."""
        if session_id in self.conversation_metrics:
            self.conversation_metrics[session_id].errors.append(
                {"timestamp": datetime.now(), "error": error_data}
            )

        # Increment error counter
        self.error_counter.add(
            1,
            {"session_id": session_id, "error_type": error_data.get("type", "unknown")},
        )

        # Log error
        self._log_event(
            "error_occurred",
            {"session_id": session_id, "error": error_data},
            severity="ERROR",
        )

    def track_task_completion(self, session_id: str, task_type: str, success: bool) -> None:
        """Track task completion."""
        if session_id in self.conversation_metrics:
            self.conversation_metrics[session_id].task_completed = success

        self._log_event(
            "task_completed",
            {"session_id": session_id, "task_type": task_type, "success": success},
        )

    def track_escalation(self, session_id: str, reason: str) -> None:
        """Track conversation escalation."""
        if session_id in self.conversation_metrics:
            self.conversation_metrics[session_id].escalated = True

        self._log_event("conversation_escalated", {"session_id": session_id, "reason": reason})

    @trace.instrumentation_wrapper
    async def track_model_inference(self, model_name: str, operation: str) -> None:
        """Track model inference performance."""
        start_time = time.time()

        try:
            # The actual operation would be yielded here
            yield
        finally:
            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Update metrics
            self.metric_history["model_inference_time"].append(inference_time)

            # Record in histogram
            if hasattr(self, "model_inference_histogram"):
                self.model_inference_histogram.record(
                    inference_time, {"model": model_name, "operation": operation}
                )

    def get_conversation_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get summary of a conversation."""
        if session_id not in self.conversation_metrics:
            return None

        metrics = self.conversation_metrics[session_id]
        duration: Any | None = None
        if metrics.end_time:
            duration = (metrics.end_time - metrics.start_time).total_seconds()

        # Calculate average sentiment
        avg_sentiment: Any | None = None
        if metrics.sentiment_scores:
            avg_sentiment = sum(metrics.sentiment_scores) / len(metrics.sentiment_scores)

        # Calculate average response time
        avg_response_time: Any | None = None
        if metrics.response_times:
            avg_response_time = sum(metrics.response_times) / len(metrics.response_times)

        return {
            "session_id": session_id,
            "start_time": metrics.start_time.isoformat(),
            "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
            "duration": duration,
            "message_count": metrics.message_count,
            "turn_count": metrics.turn_count,
            "unique_intents": list(set(metrics.intents_detected)),
            "unique_entities": list(set(metrics.entities_extracted)),
            "avg_sentiment": avg_sentiment,
            "avg_response_time": avg_response_time,
            "error_count": len(metrics.errors),
            "task_completed": metrics.task_completed,
            "escalated": metrics.escalated,
            "user_satisfaction": metrics.user_satisfaction,
        }

    def get_system_dashboard(self) -> dict[str, Any]:
        """Get system-wide dashboard data."""
        return {
            "real_time_metrics": {
                "active_sessions": self.system_metrics.active_sessions,
                "messages_per_second": self.system_metrics.messages_per_second,
                "avg_response_time": self.system_metrics.avg_response_time,
                "error_rate": self.system_metrics.error_rate,
                "fallback_rate": self.system_metrics.fallback_rate,
            },
            "performance_metrics": {
                "p95_response_time": self.system_metrics.p95_response_time,
                "p99_response_time": self.system_metrics.p99_response_time,
                "cache_hit_rate": self.system_metrics.cache_hit_rate,
                "model_inference_time": self.system_metrics.model_inference_time,
            },
            "quality_metrics": {
                "intent_accuracy": self.system_metrics.intent_accuracy,
                "entity_precision": self.system_metrics.entity_precision,
                "avg_user_satisfaction": self._calculate_avg_satisfaction(),
            },
            "resource_metrics": {
                "cpu_usage": self.system_metrics.cpu_usage,
                "memory_usage": self.system_metrics.memory_usage,
            },
        }

    def _calculate_avg_satisfaction(self) -> float | None:
        """Calculate average user satisfaction."""
        satisfactions = [
            m.user_satisfaction
            for m in self.conversation_metrics.values()
            if m.user_satisfaction is not None
        ]

        if satisfactions:
            return sum(satisfactions) / len(satisfactions)
        return None

    def create_monitoring_dashboard(self) -> dict[str, Any]:
        """Create monitoring dashboard configuration."""
        return {
            "panels": [
                {
                    "title": "Active Sessions",
                    "type": "gauge",
                    "metric": "assistant.sessions.active",
                    "thresholds": [
                        {"value": 100, "color": "green"},
                        {"value": 500, "color": "yellow"},
                        {"value": 1000, "color": "red"},
                    ],
                },
                {
                    "title": "Response Time",
                    "type": "graph",
                    "metrics": [
                        "assistant.response.time.p50",
                        "assistant.response.time.p95",
                        "assistant.response.time.p99",
                    ],
                    "unit": "ms",
                },
                {
                    "title": "Intent Detection",
                    "type": "heatmap",
                    "metric": "assistant.intent.distribution",
                    "groupBy": "intent",
                },
                {
                    "title": "Error Rate",
                    "type": "graph",
                    "metric": "assistant.errors.rate",
                    "alert": {"condition": "value > 0.05", "severity": "warning"},
                },
                {
                    "title": "User Satisfaction",
                    "type": "gauge",
                    "metric": "assistant.satisfaction.average",
                    "min": 1,
                    "max": 5,
                    "thresholds": [
                        {"value": 4, "color": "green"},
                        {"value": 3, "color": "yellow"},
                        {"value": 2, "color": "red"},
                    ],
                },
            ],
            "alerts": [
                {
                    "name": "high_error_rate",
                    "condition": "assistant.errors.rate > 0.1",
                    "severity": "critical",
                    "notification": "email,slack",
                },
                {
                    "name": "slow_response_time",
                    "condition": "assistant.response.time.p95 > 2000",
                    "severity": "warning",
                    "notification": "slack",
                },
                {
                    "name": "high_fallback_rate",
                    "condition": "assistant.fallback.rate > 0.2",
                    "severity": "warning",
                    "notification": "email",
                },
            ],
            "refresh_interval": 30,  # seconds
        }

    def _log_event(self, event_type: str, data: dict[str, Any], severity: str = "INFO") -> None:
        """Log event to cloud logging."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
            "severity": severity,
        }

        if self.cloud_logger:
            self.cloud_logger.log_struct(log_entry, severity=severity)
        else:
            self.logger.log(
                getattr(logging, severity, logging.INFO),
                f"{event_type}: {json.dumps(data)}",
            )

    def analyze_conversation_quality(self, time_range: timedelta) -> dict[str, Any]:
        """Analyze conversation quality over time range."""
        cutoff_time = datetime.now() - time_range
        recent_conversations = [
            m for m in self.conversation_metrics.values() if m.start_time >= cutoff_time
        ]

        if not recent_conversations:
            return {"error": "No conversations in time range"}

        # Calculate quality metrics
        total_conversations = len(recent_conversations)
        completed_tasks = sum(1 for m in recent_conversations if m.task_completed)
        escalated = sum(1 for m in recent_conversations if m.escalated)

        # Intent accuracy (based on fallback rate)
        total_intents = sum(len(m.intents_detected) for m in recent_conversations)
        unknown_intents = sum(
            1 for m in recent_conversations for intent in m.intents_detected if intent == "unknown"
        )
        intent_accuracy = 1 - (unknown_intents / total_intents) if total_intents > 0 else 0

        # Response time analysis
        all_response_times: list[float] = []
        for m in recent_conversations:
            all_response_times.extend(m.response_times)

        response_time_stats: dict[str, Any] = {}
        if all_response_times:
            sorted_times = sorted(all_response_times)
            response_time_stats = {
                "avg": sum(sorted_times) / len(sorted_times),
                "p50": sorted_times[len(sorted_times) // 2],
                "p95": sorted_times[int(len(sorted_times) * 0.95)],
                "p99": sorted_times[int(len(sorted_times) * 0.99)],
            }

        # Sentiment analysis
        all_sentiments: list[float] = []
        for m in recent_conversations:
            all_sentiments.extend(m.sentiment_scores)

        avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0

        # Error analysis
        total_errors = sum(len(m.errors) for m in recent_conversations)
        error_types: defaultdict[str, int] = defaultdict(int)
        for m in recent_conversations:
            for error in m.errors:
                error_types[error.get("error", {}).get("type", "unknown")] += 1

        return {
            "time_range": str(time_range),
            "total_conversations": total_conversations,
            "quality_metrics": {
                "task_completion_rate": completed_tasks / total_conversations,
                "escalation_rate": escalated / total_conversations,
                "intent_accuracy": intent_accuracy,
                "avg_sentiment": avg_sentiment,
            },
            "performance_metrics": response_time_stats,
            "error_analysis": {
                "total_errors": total_errors,
                "error_rate": total_errors / sum(m.message_count for m in recent_conversations),
                "error_types": dict(error_types),
            },
            "recommendations": self._generate_recommendations(
                intent_accuracy,
                response_time_stats.get("p95", 0),
                total_errors / total_conversations if total_conversations > 0 else 0,
            ),
        }

    def _generate_recommendations(
        self,
        intent_accuracy: float,
        p95_response_time: float,
        error_rate: float,
    ) -> list[dict[str, Any]]:
        """Generate improvement recommendations."""
        recommendations: list[dict[str, Any]] = []

        if intent_accuracy < 0.85:
            recommendations.append(
                {
                    "area": "Intent Recognition",
                    "issue": f"Low accuracy: {intent_accuracy:.2%}",
                    "recommendation": "Retrain intent classifier with more examples",
                    "priority": "high",
                },
            )

        if p95_response_time > 2000:  # 2 seconds
            recommendations.append(
                {
                    "area": "Performance",
                    "issue": f"Slow response time: {p95_response_time:.0f}ms",
                    "recommendation": "Optimize model inference or enable caching",
                    "priority": "medium",
                },
            )

        if error_rate > 0.05:  # 5% error rate
            recommendations.append(
                {
                    "area": "Reliability",
                    "issue": f"High error rate: {error_rate:.2%}",
                    "recommendation": "Investigate error patterns and add error handling",
                    "priority": "high",
                },
            )

        return recommendations


class ConversationQualityAnalyzer:
    """Analyze conversation quality and identify improvement areas."""

    def __init__(self, analytics: AssistantAnalytics) -> None:
        self.analytics = analytics
        self.logger = logging.getLogger(__name__)

    def analyze_conversations(self, time_range: timedelta) -> dict[str, Any]:
        """Analyze conversation quality metrics."""
        analysis = self.analytics.analyze_conversation_quality(time_range)

        # Add detailed analysis
        analysis["detailed_analysis"] = {
            "intent_patterns": self._analyze_intent_patterns(time_range),
            "conversation_flow": self._analyze_conversation_flow(time_range),
            "user_satisfaction": self._analyze_satisfaction(time_range),
            "error_patterns": self._analyze_error_patterns(time_range),
        }

        return analysis

    def _analyze_intent_patterns(self, time_range: timedelta) -> dict[str, Any]:
        """Analyze intent detection patterns."""
        cutoff_time = datetime.now() - time_range
        conversations = [
            m for m in self.analytics.conversation_metrics.values() if m.start_time >= cutoff_time
        ]

        # Count intent frequencies
        intent_counts: defaultdict[str, int] = defaultdict(int)
        intent_transitions: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for conv in conversations:
            intents = conv.intents_detected
            for intent in intents:
                intent_counts[intent] += 1

            # Track transitions
            for i in range(len(intents) - 1):
                intent_transitions[intents[i]][intents[i + 1]] += 1

        # Find common patterns
        total_intents = sum(intent_counts.values())
        intent_distribution = {
            intent: count / total_intents for intent, count in intent_counts.items()
        }

        return {
            "intent_distribution": intent_distribution,
            "most_common_intents": sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[
                :10
            ],
            "intent_transitions": dict(intent_transitions),
        }

    def _analyze_conversation_flow(self, time_range: timedelta) -> dict[str, Any]:
        """Analyze conversation flow patterns."""
        cutoff_time = datetime.now() - time_range
        conversations = [
            m for m in self.analytics.conversation_metrics.values() if m.start_time >= cutoff_time
        ]

        # Analyze conversation lengths
        conversation_lengths = [m.message_count for m in conversations]
        avg_length = (
            sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
        )

        # Analyze completion patterns
        completed_conversations = [m for m in conversations if m.task_completed]
        completion_rate = len(completed_conversations) / len(conversations) if conversations else 0

        # Analyze conversation duration
        durations: list[float] = []
        for m in conversations:
            if m.end_time:
                duration = (m.end_time - m.start_time).total_seconds()
                durations.append(duration)

        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "avg_conversation_length": avg_length,
            "completion_rate": completion_rate,
            "avg_duration_seconds": avg_duration,
            "length_distribution": self._calculate_distribution(
                [float(x) for x in conversation_lengths]
            ),
        }

    def _analyze_satisfaction(self, time_range: timedelta) -> dict[str, Any]:
        """Analyze user satisfaction patterns."""
        cutoff_time = datetime.now() - time_range
        conversations = [
            m
            for m in self.analytics.conversation_metrics.values()
            if m.start_time >= cutoff_time and m.user_satisfaction is not None
        ]

        if not conversations:
            return {"message": "No satisfaction data available"}

        satisfactions = [m.user_satisfaction for m in conversations]
        avg_satisfaction = sum(satisfactions) / len(satisfactions)

        # Correlate with other metrics
        high_satisfaction = [m for m in conversations if m.user_satisfaction >= 4]
        low_satisfaction = [m for m in conversations if m.user_satisfaction <= 2]

        high_sat_metrics = self._calculate_conversation_metrics(high_satisfaction)
        low_sat_metrics = self._calculate_conversation_metrics(low_satisfaction)

        return {
            "avg_satisfaction": avg_satisfaction,
            "satisfaction_distribution": self._calculate_distribution(satisfactions),
            "high_satisfaction_characteristics": high_sat_metrics,
            "low_satisfaction_characteristics": low_sat_metrics,
            "correlation_insights": self._generate_correlation_insights(
                high_sat_metrics, low_sat_metrics
            ),
        }

    def _analyze_error_patterns(self, time_range: timedelta) -> dict[str, Any]:
        """Analyze error patterns."""
        cutoff_time = datetime.now() - time_range
        conversations = [
            m for m in self.analytics.conversation_metrics.values() if m.start_time >= cutoff_time
        ]

        # Collect all errors
        all_errors: list[dict[str, Any]] = []
        for conv in conversations:
            for error in conv.errors:
                all_errors.append(
                    {
                        "session_id": conv.session_id,
                        "timestamp": error["timestamp"],
                        "error": error["error"],
                    },
                )

        # Group by error type
        error_types = defaultdict(list)
        for error in all_errors:
            error_type = error["error"].get("type", "unknown")
            error_types[error_type].append(error)

        # Analyze error timing
        error_times = [e["timestamp"].hour for e in all_errors]
        error_time_distribution: defaultdict[int, int] = defaultdict(int)
        for hour in error_times:
            error_time_distribution[hour] += 1

        return {
            "total_errors": len(all_errors),
            "error_types": {k: len(v) for k, v in error_types.items()},
            "error_time_distribution": dict(error_time_distribution),
            "most_common_errors": sorted(
                error_types.items(), key=lambda x: len(x[1]), reverse=True
            )[:5],
        }

    def _calculate_distribution(self, values: list[float]) -> dict[str, float]:
        """Calculate distribution statistics."""
        if not values:
            return {}

        sorted_values = sorted(values)
        return {
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(values) / len(values),
            "median": sorted_values[len(sorted_values) // 2],
            "p25": sorted_values[int(len(sorted_values) * 0.25)],
            "p75": sorted_values[int(len(sorted_values) * 0.75)],
        }

    def _calculate_conversation_metrics(
        self, conversations: list[ConversationMetrics]
    ) -> dict[str, Any]:
        """Calculate aggregate metrics for a set of conversations."""
        if not conversations:
            return {}

        return {
            "avg_message_count": sum(c.message_count for c in conversations) / len(conversations),
            "avg_response_time": sum(
                sum(c.response_times) / len(c.response_times)
                for c in conversations
                if c.response_times
            )
            / len([c for c in conversations if c.response_times]),
            "error_rate": sum(len(c.errors) for c in conversations)
            / sum(c.message_count for c in conversations),
            "escalation_rate": sum(1 for c in conversations if c.escalated) / len(conversations),
        }

    def _generate_correlation_insights(
        self, high_sat: dict[str, Any], low_sat: dict[str, Any]
    ) -> list[str]:
        """Generate insights from satisfaction correlation."""
        insights: list[str] = []

        if high_sat.get("avg_message_count", 0) < low_sat.get("avg_message_count", 0):
            insights.append("Shorter conversations correlate with higher satisfaction")

        if high_sat.get("error_rate", 0) < low_sat.get("error_rate", 0):
            insights.append("Lower error rates correlate with higher satisfaction")

        if high_sat.get("avg_response_time", 0) < low_sat.get("avg_response_time", 0):
            insights.append("Faster response times correlate with higher satisfaction")

        return insights


# Export monitoring utilities
__all__ = [
    "AssistantAnalytics",
    "ConversationMetrics",
    "ConversationQualityAnalyzer",
    "MetricType",
    "SystemMetrics",
]
