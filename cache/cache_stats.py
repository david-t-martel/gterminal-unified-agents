"""
Cache statistics and metrics collection.

This module provides comprehensive statistics and metrics for cache performance
monitoring, extracted from my-fullstack-agent patterns.
"""

from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
import threading
import time
from typing import Any


@dataclass
class CacheMetric:
    """Individual cache metric with timestamp."""

    timestamp: float
    operation: str  # 'get', 'set', 'delete', 'clear'
    hit: bool
    latency_ms: float
    cache_layer: str  # 'l1', 'l2', 'both'
    key_size: int = 0
    value_size: int = 0


@dataclass
class CacheStats:
    """Comprehensive cache statistics."""

    # Basic counters
    total_operations: int = 0
    total_hits: int = 0
    total_misses: int = 0
    total_errors: int = 0

    # Operation counters
    gets: int = 0
    sets: int = 0
    deletes: int = 0
    clears: int = 0

    # Layer-specific counters
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0

    # Performance metrics
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    # Memory usage
    estimated_memory_bytes: int = 0
    key_size_bytes: int = 0
    value_size_bytes: int = 0

    # Time-based metrics
    start_time: float = field(default_factory=time.time)
    last_reset: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate as percentage."""
        total = self.total_hits + self.total_misses
        return (self.total_hits / total * 100) if total > 0 else 0.0

    @property
    def operations_per_second(self) -> float:
        """Calculate operations per second."""
        duration = time.time() - self.start_time
        return self.total_operations / duration if duration > 0 else 0.0

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time


class CacheMetrics:
    """
    Advanced cache metrics collector with time-series data.

    Features:
    - Real-time metric collection
    - Time-series data with configurable retention
    - Performance percentiles
    - Trend analysis
    - Memory usage tracking
    """

    def __init__(self, max_metrics: int = 10000, retention_seconds: int = 3600):
        self.max_metrics = max_metrics
        self.retention_seconds = retention_seconds

        # Thread-safe metrics storage
        self._lock = threading.RLock()
        self._metrics: deque = deque(maxlen=max_metrics)

        # Aggregated statistics
        self.stats = CacheStats()

        # Time-series data for trend analysis
        self._operation_counts = defaultdict(int)
        self._latency_samples = deque(maxlen=1000)  # Keep last 1000 samples

        # Performance percentiles
        self._percentile_cache = {}
        self._percentile_cache_time = 0

    def record_metric(
        self,
        operation: str,
        hit: bool,
        latency_ms: float,
        cache_layer: str,
        key_size: int = 0,
        value_size: int = 0,
    ):
        """Record a cache operation metric."""
        metric = CacheMetric(
            timestamp=time.time(),
            operation=operation,
            hit=hit,
            latency_ms=latency_ms,
            cache_layer=cache_layer,
            key_size=key_size,
            value_size=value_size,
        )

        with self._lock:
            # Add to metrics queue
            self._metrics.append(metric)

            # Update aggregated stats
            self.stats.total_operations += 1

            if hit:
                self.stats.total_hits += 1
                if cache_layer == "l1":
                    self.stats.l1_hits += 1
                elif cache_layer == "l2":
                    self.stats.l2_hits += 1
            else:
                self.stats.total_misses += 1
                if cache_layer == "l1":
                    self.stats.l1_misses += 1
                elif cache_layer == "l2":
                    self.stats.l2_misses += 1

            # Update operation counters
            if operation == "get":
                self.stats.gets += 1
            elif operation == "set":
                self.stats.sets += 1
            elif operation == "delete":
                self.stats.deletes += 1
            elif operation == "clear":
                self.stats.clears += 1

            # Update latency metrics
            self.stats.min_latency_ms = min(self.stats.min_latency_ms, latency_ms)
            self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency_ms)

            # Update running average (exponential moving average)
            alpha = 0.1  # Smoothing factor
            self.stats.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.stats.avg_latency_ms

            # Update memory estimates
            self.stats.key_size_bytes += key_size
            self.stats.value_size_bytes += value_size
            self.stats.estimated_memory_bytes = (
                self.stats.key_size_bytes + self.stats.value_size_bytes
            )

            # Add to latency samples for percentile calculation
            self._latency_samples.append(latency_ms)

            # Invalidate percentile cache
            self._percentile_cache = {}

            # Clean up old metrics
            self._cleanup_old_metrics()

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics as dictionary."""
        with self._lock:
            return {
                "basic": {
                    "total_operations": self.stats.total_operations,
                    "total_hits": self.stats.total_hits,
                    "total_misses": self.stats.total_misses,
                    "total_errors": self.stats.total_errors,
                    "hit_rate_percent": round(self.stats.hit_rate, 2),
                },
                "operations": {
                    "gets": self.stats.gets,
                    "sets": self.stats.sets,
                    "deletes": self.stats.deletes,
                    "clears": self.stats.clears,
                },
                "layers": {
                    "l1_hits": self.stats.l1_hits,
                    "l1_misses": self.stats.l1_misses,
                    "l2_hits": self.stats.l2_hits,
                    "l2_misses": self.stats.l2_misses,
                    "l1_hit_rate": self._calculate_layer_hit_rate("l1"),
                    "l2_hit_rate": self._calculate_layer_hit_rate("l2"),
                },
                "performance": {
                    "avg_latency_ms": round(self.stats.avg_latency_ms, 3),
                    "min_latency_ms": round(self.stats.min_latency_ms, 3),
                    "max_latency_ms": round(self.stats.max_latency_ms, 3),
                    "operations_per_second": round(self.stats.operations_per_second, 2),
                    "uptime_seconds": round(self.stats.uptime_seconds, 2),
                },
                "memory": {
                    "estimated_memory_bytes": self.stats.estimated_memory_bytes,
                    "estimated_memory_mb": round(
                        self.stats.estimated_memory_bytes / (1024 * 1024), 2
                    ),
                    "key_size_bytes": self.stats.key_size_bytes,
                    "value_size_bytes": self.stats.value_size_bytes,
                    "avg_key_size": self._calculate_avg_key_size(),
                    "avg_value_size": self._calculate_avg_value_size(),
                },
            }

    def get_percentiles(self, percentiles: list[float] | None = None) -> dict[str, float]:
        """Get latency percentiles."""
        if percentiles is None:
            percentiles = [50.0, 90.0, 95.0, 99.0, 99.9]

        current_time = time.time()

        # Use cached percentiles if recent
        if current_time - self._percentile_cache_time < 10:  # Cache for 10 seconds
            return {str(p): self._percentile_cache.get(p, 0.0) for p in percentiles}

        with self._lock:
            if not self._latency_samples:
                return {str(p): 0.0 for p in percentiles}

            sorted_samples = sorted(self._latency_samples)
            result = {}

            for percentile in percentiles:
                if percentile <= 0 or percentile >= 100:
                    continue

                index = int((percentile / 100) * len(sorted_samples))
                index = min(index, len(sorted_samples) - 1)
                result[str(percentile)] = sorted_samples[index]
                self._percentile_cache[percentile] = sorted_samples[index]

            self._percentile_cache_time = current_time
            return result

    def get_time_series(
        self, operation: str | None = None, window_seconds: int = 300
    ) -> list[tuple[float, int]]:
        """Get time-series data for operations."""
        current_time = time.time()
        start_time = current_time - window_seconds

        # Group metrics by time buckets (1-second buckets)
        buckets = defaultdict(int)

        with self._lock:
            for metric in self._metrics:
                if metric.timestamp < start_time:
                    continue

                if operation and metric.operation != operation:
                    continue

                # Round timestamp to 1-second bucket
                bucket_time = int(metric.timestamp)
                buckets[bucket_time] += 1

        # Convert to sorted list
        return sorted(buckets.items())

    def get_trend_analysis(self, window_seconds: int = 300) -> dict[str, Any]:
        """Get trend analysis for cache performance."""
        current_time = time.time()
        start_time = current_time - window_seconds

        # Collect recent metrics
        recent_metrics = []
        with self._lock:
            for metric in self._metrics:
                if metric.timestamp >= start_time:
                    recent_metrics.append(metric)

        if not recent_metrics:
            return {
                "trend": "stable",
                "hit_rate_trend": 0.0,
                "latency_trend": 0.0,
                "operation_trend": 0.0,
            }

        # Split metrics into first and second half for trend calculation
        mid_point = len(recent_metrics) // 2
        first_half = recent_metrics[:mid_point]
        second_half = recent_metrics[mid_point:]

        # Calculate trends
        first_hit_rate = self._calculate_hit_rate(first_half)
        second_hit_rate = self._calculate_hit_rate(second_half)
        hit_rate_trend = second_hit_rate - first_hit_rate

        first_avg_latency = self._calculate_avg_latency(first_half)
        second_avg_latency = self._calculate_avg_latency(second_half)
        latency_trend = second_avg_latency - first_avg_latency

        operation_trend = len(second_half) - len(first_half)

        # Determine overall trend
        trend = "stable"
        if hit_rate_trend > 1 and latency_trend < 0:
            trend = "improving"
        elif hit_rate_trend < -1 or latency_trend > 10:
            trend = "degrading"

        return {
            "trend": trend,
            "hit_rate_trend": round(hit_rate_trend, 2),
            "latency_trend": round(latency_trend, 3),
            "operation_trend": operation_trend,
            "sample_size": len(recent_metrics),
        }

    def reset_stats(self):
        """Reset all statistics."""
        with self._lock:
            self._metrics.clear()
            self.stats = CacheStats()
            self._latency_samples.clear()
            self._percentile_cache = {}
            self._percentile_cache_time = 0

    def export_metrics(self, format: str = "dict") -> Any:
        """Export metrics in various formats."""
        if format == "dict":
            return {
                "stats": self.get_stats(),
                "percentiles": self.get_percentiles(),
                "trend_analysis": self.get_trend_analysis(),
            }
        elif format == "prometheus":
            # Basic Prometheus format (could be extended)
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        if not self._metrics:
            return

        current_time = time.time()
        cutoff_time = current_time - self.retention_seconds

        # Remove old metrics from the left side
        while self._metrics and self._metrics[0].timestamp < cutoff_time:
            self._metrics.popleft()

    def _calculate_layer_hit_rate(self, layer: str) -> float:
        """Calculate hit rate for specific layer."""
        if layer == "l1":
            total = self.stats.l1_hits + self.stats.l1_misses
            return (self.stats.l1_hits / total * 100) if total > 0 else 0.0
        elif layer == "l2":
            total = self.stats.l2_hits + self.stats.l2_misses
            return (self.stats.l2_hits / total * 100) if total > 0 else 0.0
        return 0.0

    def _calculate_avg_key_size(self) -> float:
        """Calculate average key size."""
        return (
            self.stats.key_size_bytes / self.stats.total_operations
            if self.stats.total_operations > 0
            else 0.0
        )

    def _calculate_avg_value_size(self) -> float:
        """Calculate average value size."""
        return self.stats.value_size_bytes / self.stats.sets if self.stats.sets > 0 else 0.0

    def _calculate_hit_rate(self, metrics: list[CacheMetric]) -> float:
        """Calculate hit rate for a list of metrics."""
        if not metrics:
            return 0.0
        hits = sum(1 for m in metrics if m.hit)
        return (hits / len(metrics)) * 100

    def _calculate_avg_latency(self, metrics: list[CacheMetric]) -> float:
        """Calculate average latency for a list of metrics."""
        if not metrics:
            return 0.0
        return sum(m.latency_ms for m in metrics) / len(metrics)

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        stats = self.get_stats()
        lines = [
            "# HELP cache_operations_total Total number of cache operations",
            "# TYPE cache_operations_total counter",
            f"cache_operations_total {stats['basic']['total_operations']}",
            "",
            "# HELP cache_hits_total Total number of cache hits",
            "# TYPE cache_hits_total counter",
            f"cache_hits_total {stats['basic']['total_hits']}",
            "",
            "# HELP cache_hit_rate_percent Cache hit rate percentage",
            "# TYPE cache_hit_rate_percent gauge",
            f"cache_hit_rate_percent {stats['basic']['hit_rate_percent']}",
            "",
            "# HELP cache_latency_ms_avg Average cache latency in milliseconds",
            "# TYPE cache_latency_ms_avg gauge",
            f"cache_latency_ms_avg {stats['performance']['avg_latency_ms']}",
        ]
        return "\n".join(lines)
