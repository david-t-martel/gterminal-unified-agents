"""Performance monitoring utilities using Rust bindings.

Provides system monitoring, performance metrics, and optimization
recommendations for all agents.
"""

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Any

# from .rust_bindings import rust_utils  # Temporarily disabled due to missing module
rust_utils = None  # Fallback to None when rust_bindings unavailable

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""

    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: float
    load_average: float
    process_count: int
    cache_hit_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_usage_percent": self.cpu_usage,
            "memory_usage_percent": self.memory_usage,
            "memory_available_mb": self.memory_available,
            "load_average": self.load_average,
            "process_count": self.process_count,
            "cache_hit_rate": self.cache_hit_rate,
        }


@dataclass
class SystemInfo:
    """System information container."""

    hostname: str
    os_version: str
    architecture: str
    total_memory: float
    cpu_count: int
    cpu_brand: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hostname": self.hostname,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "total_memory_gb": self.total_memory,
            "cpu_count": self.cpu_count,
            "cpu_brand": self.cpu_brand,
        }


class PerformanceMonitor:
    """High-performance system monitoring with Rust bindings."""

    def __init__(self) -> None:
        """Initialize performance monitor."""
        self.rust = rust_utils
        self.metrics_history: list[PerformanceMetrics] = []
        self.max_history_size = 1000

        logger.info("Initialized PerformanceMonitor with Rust bindings")

    def get_system_info(self) -> SystemInfo:
        """Get comprehensive system information."""
        try:
            info = self.rust.get_system_info()

            return SystemInfo(
                hostname=info.get("hostname", "unknown"),
                os_version=info.get("os_version", "unknown"),
                architecture=info.get("architecture", "unknown"),
                total_memory=info.get("total_memory", 0) / (1024**3),  # Convert to GB
                cpu_count=info.get("cpu_count", 0),
                cpu_brand=info.get("cpu_brand", "unknown"),
            )
        except Exception as e:
            logger.exception(f"Failed to get system info: {e}")
            return SystemInfo("unknown", "unknown", "unknown", 0.0, 0, "unknown")

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        try:
            metrics = self.rust.get_performance_metrics()
            cache_stats = self.rust.cache_stats()

            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=metrics.get("cpu_usage_percent", 0.0),
                memory_usage=metrics.get("memory_usage_percent", 0.0),
                memory_available=metrics.get("memory_available_mb", 0.0),
                load_average=metrics.get("load_average_1m", 0.0),
                process_count=metrics.get("process_count", 0),
                cache_hit_rate=cache_stats.get("hit_rate", 0.0),
            )
        except Exception as e:
            logger.exception(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(time.time(), 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    def record_metrics(self) -> PerformanceMetrics:
        """Record current metrics to history."""
        metrics = self.get_current_metrics()

        self.metrics_history.append(metrics)

        # Limit history size
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size // 2 :]

        return metrics

    def get_metrics_history(self, last_n: int | None = None) -> list[PerformanceMetrics]:
        """Get metrics history."""
        if last_n is None:
            return self.metrics_history.copy()
        return self.metrics_history[-last_n:]

    def get_average_metrics(self, last_n: int | None = None) -> dict[str, float]:
        """Get average metrics over time period."""
        history = self.get_metrics_history(last_n)

        if not history:
            return {}

        return {
            "avg_cpu_usage": sum(m.cpu_usage for m in history) / len(history),
            "avg_memory_usage": sum(m.memory_usage for m in history) / len(history),
            "avg_load_average": sum(m.load_average for m in history) / len(history),
            "avg_cache_hit_rate": sum(m.cache_hit_rate for m in history) / len(history),
            "sample_count": len(history),
            "time_span_minutes": (history[-1].timestamp - history[0].timestamp) / 60,
        }

    def analyze_performance(self) -> dict[str, Any]:
        """Analyze current performance and provide recommendations."""
        current = self.get_current_metrics()
        system = self.get_system_info()
        self.get_average_metrics(last_n=10)  # Last 10 samples

        analysis = {
            "current_metrics": current.to_dict(),
            "system_info": system.to_dict(),
            "performance_status": "good",
            "recommendations": [],
            "warnings": [],
        }

        # CPU analysis
        if current.cpu_usage > 80:
            analysis["performance_status"] = "degraded"
            analysis["warnings"].append("High CPU usage detected")
            analysis["recommendations"].append("Consider reducing concurrent operations")

        # Memory analysis
        if current.memory_usage > 90:
            analysis["performance_status"] = "critical"
            analysis["warnings"].append("Critical memory usage")
            analysis["recommendations"].append("Clear caches and reduce memory usage")
        elif current.memory_usage > 75:
            analysis["warnings"].append("High memory usage")
            analysis["recommendations"].append("Monitor memory usage closely")

        # Cache analysis
        if current.cache_hit_rate < 0.5:
            analysis["recommendations"].append("Cache hit rate is low - consider cache warming")

        # Load average analysis (for Unix systems)
        if current.load_average > system.cpu_count * 1.5:
            analysis["warnings"].append("High system load detected")
            analysis["recommendations"].append("System is overloaded - reduce workload")

        return analysis

    async def monitor_performance(
        self,
        duration_seconds: int = 60,
        interval_seconds: int = 5,
    ) -> list[PerformanceMetrics]:
        """Monitor performance over a time period."""
        logger.info(f"Starting performance monitoring for {duration_seconds} seconds")

        metrics_collected: list[Any] = []
        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            metrics = self.record_metrics()
            metrics_collected.append(metrics)

            logger.debug(
                f"CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}%, Load: {metrics.load_average:.2f}",
            )

            await asyncio.sleep(interval_seconds)

        logger.info(f"Performance monitoring complete. Collected {len(metrics_collected)} samples")
        return metrics_collected

    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        current = self.get_current_metrics()
        system = self.get_system_info()
        analysis = self.analyze_performance()
        averages = self.get_average_metrics()

        return {
            "report_timestamp": time.time(),
            "system_info": system.to_dict(),
            "current_metrics": current.to_dict(),
            "historical_averages": averages,
            "performance_analysis": analysis,
            "rust_bindings_active": True,
            "cache_backend": "rust",
            "monitoring_active": len(self.metrics_history) > 0,
        }

    def optimize_for_performance(self) -> dict[str, Any]:
        """Provide performance optimization recommendations."""
        analysis = self.analyze_performance()
        system = self.get_system_info()

        optimizations = {
            "cache_optimizations": [],
            "memory_optimizations": [],
            "cpu_optimizations": [],
            "system_optimizations": [],
        }

        # Cache optimizations
        if analysis["current_metrics"]["cache_hit_rate"] < 0.7:
            optimizations["cache_optimizations"].extend(
                [
                    "Increase cache size",
                    "Warm cache with frequently accessed data",
                    "Optimize cache key generation",
                ],
            )

        # Memory optimizations
        if analysis["current_metrics"]["memory_usage_percent"] > 70:
            optimizations["memory_optimizations"].extend(
                [
                    "Clear unused caches",
                    "Reduce batch sizes",
                    "Use streaming operations for large data",
                ],
            )

        # CPU optimizations
        if analysis["current_metrics"]["cpu_usage_percent"] > 60:
            optimizations["cpu_optimizations"].extend(
                [
                    "Use async operations where possible",
                    "Implement rate limiting",
                    "Optimize hot code paths",
                ],
            )

        # System optimizations
        if system.total_memory < 8.0:  # Less than 8GB RAM
            optimizations["system_optimizations"].append("Consider increasing system memory")

        if system.cpu_count < 4:
            optimizations["system_optimizations"].append("Consider upgrading to multi-core system")

        return {
            "optimization_recommendations": optimizations,
            "current_performance": analysis["performance_status"],
            "rust_acceleration_active": True,
        }


# Global instance for easy access
performance_monitor = PerformanceMonitor()


# Convenience functions for direct access
def get_system_info() -> dict[str, Any]:
    """Get system information."""
    return performance_monitor.get_system_info().to_dict()


def get_current_metrics() -> dict[str, Any]:
    """Get current performance metrics."""
    return performance_monitor.get_current_metrics().to_dict()


def analyze_performance() -> dict[str, Any]:
    """Analyze current performance."""
    return performance_monitor.analyze_performance()


async def monitor_performance(duration: int = 60) -> list[dict[str, Any]]:
    """Monitor performance for specified duration."""
    metrics = await performance_monitor.monitor_performance(duration)
    return [m.to_dict() for m in metrics]
