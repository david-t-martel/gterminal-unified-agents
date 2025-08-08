#!/usr/bin/env python3
"""Performance Monitoring for gterminal.

This module provides comprehensive performance monitoring including:
- Execution time tracking
- Memory usage monitoring
- Resource utilization metrics
- Performance bottleneck detection
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
import logging
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionProfile:
    """Execution profile for a specific operation."""

    operation_name: str
    start_time: float
    end_time: float | None = None
    duration: float | None = None
    memory_start: float = 0.0
    memory_peak: float = 0.0
    memory_end: float = 0.0
    cpu_percent: float = 0.0
    success: bool = True
    error: str | None = None
    metrics: list[PerformanceMetric] = field(default_factory=list)


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(self, max_profiles: int = 1000):
        """Initialize performance monitor.

        Args:
            max_profiles: Maximum number of execution profiles to retain
        """
        self.max_profiles = max_profiles
        self.execution_profiles: list[ExecutionProfile] = []
        self.global_metrics: list[PerformanceMetric] = []
        self.process = psutil.Process()

        # Performance thresholds
        self.thresholds = {
            "execution_time_warning": 5.0,  # seconds
            "execution_time_critical": 30.0,  # seconds
            "memory_usage_warning": 500,  # MB
            "memory_usage_critical": 1000,  # MB
            "cpu_usage_warning": 80.0,  # percent
            "cpu_usage_critical": 95.0,  # percent
        }

        logger.info("✅ PerformanceMonitor initialized")

    @asynccontextmanager
    async def profile_execution(self, operation_name: str):
        """Context manager for profiling operation execution.

        Args:
            operation_name: Name of the operation being profiled
        """
        profile = ExecutionProfile(
            operation_name=operation_name,
            start_time=time.time(),
            memory_start=self._get_memory_usage(),
        )

        try:
            logger.debug(f"📊 Starting performance profile: {operation_name}")
            yield profile

            profile.success = True

        except Exception as e:
            profile.success = False
            profile.error = str(e)
            logger.warning(f"❌ Operation failed during profiling: {operation_name} - {e}")
            raise

        finally:
            # Complete the profile
            profile.end_time = time.time()
            profile.duration = profile.end_time - profile.start_time
            profile.memory_end = self._get_memory_usage()
            profile.memory_peak = max(profile.memory_start, profile.memory_end)
            profile.cpu_percent = self._get_cpu_usage()

            # Add to profiles list
            self._add_profile(profile)

            # Check for performance issues
            await self._check_performance_thresholds(profile)

            logger.debug(
                f"📊 Completed performance profile: {operation_name} "
                f"({profile.duration:.3f}s, {profile.memory_peak:.1f}MB)"
            )

    def record_metric(
        self, name: str, value: float, unit: str = "count", context: dict[str, Any] | None = None
    ) -> None:
        """Record a custom performance metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            context: Additional context information
        """
        metric = PerformanceMetric(
            name=name, value=value, unit=unit, timestamp=datetime.now(), context=context or {}
        )

        self.global_metrics.append(metric)

        # Keep only recent metrics
        if len(self.global_metrics) > self.max_profiles:
            self.global_metrics = self.global_metrics[-self.max_profiles :]

        logger.debug(f"📈 Recorded metric: {name}={value}{unit}")

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics summary."""
        if not self.execution_profiles:
            return {"message": "No execution profiles available"}

        successful_profiles = [p for p in self.execution_profiles if p.success]
        failed_profiles = [p for p in self.execution_profiles if not p.success]

        durations = [p.duration for p in successful_profiles if p.duration]
        memory_peaks = [p.memory_peak for p in self.execution_profiles]
        cpu_usages = [p.cpu_percent for p in self.execution_profiles]

        return {
            "total_operations": len(self.execution_profiles),
            "successful_operations": len(successful_profiles),
            "failed_operations": len(failed_profiles),
            "success_rate_percent": (
                len(successful_profiles) / len(self.execution_profiles) * 100
                if self.execution_profiles
                else 0
            ),
            "execution_times": {
                "average_seconds": sum(durations) / len(durations) if durations else 0,
                "min_seconds": min(durations) if durations else 0,
                "max_seconds": max(durations) if durations else 0,
            },
            "memory_usage": {
                "average_mb": sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0,
                "peak_mb": max(memory_peaks) if memory_peaks else 0,
            },
            "cpu_usage": {
                "average_percent": sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0,
                "peak_percent": max(cpu_usages) if cpu_usages else 0,
            },
        }

    def get_recent_profiles(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent execution profiles.

        Args:
            limit: Maximum number of profiles to return

        Returns:
            List of profile dictionaries
        """
        recent_profiles = self.execution_profiles[-limit:] if self.execution_profiles else []

        return [
            {
                "operation_name": profile.operation_name,
                "duration_seconds": profile.duration,
                "memory_peak_mb": profile.memory_peak,
                "cpu_percent": profile.cpu_percent,
                "success": profile.success,
                "error": profile.error,
                "timestamp": datetime.fromtimestamp(profile.start_time).isoformat(),
            }
            for profile in reversed(recent_profiles)
        ]

    def get_performance_warnings(self) -> list[dict[str, Any]]:
        """Get list of recent performance warnings."""
        warnings = []

        for profile in self.execution_profiles[-50:]:  # Check last 50 operations
            if profile.duration and profile.duration > self.thresholds["execution_time_warning"]:
                severity = (
                    "critical"
                    if profile.duration > self.thresholds["execution_time_critical"]
                    else "warning"
                )
                warnings.append(
                    {
                        "type": "slow_execution",
                        "severity": severity,
                        "operation": profile.operation_name,
                        "duration_seconds": profile.duration,
                        "threshold_seconds": self.thresholds["execution_time_warning"],
                        "timestamp": datetime.fromtimestamp(profile.start_time).isoformat(),
                    }
                )

            if profile.memory_peak > self.thresholds["memory_usage_warning"]:
                severity = (
                    "critical"
                    if profile.memory_peak > self.thresholds["memory_usage_critical"]
                    else "warning"
                )
                warnings.append(
                    {
                        "type": "high_memory_usage",
                        "severity": severity,
                        "operation": profile.operation_name,
                        "memory_mb": profile.memory_peak,
                        "threshold_mb": self.thresholds["memory_usage_warning"],
                        "timestamp": datetime.fromtimestamp(profile.start_time).isoformat(),
                    }
                )

        return warnings

    def get_system_info(self) -> dict[str, Any]:
        """Get current system performance information."""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()

            return {
                "process_id": self.process.pid,
                "memory": {
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                    "percent": self.process.memory_percent(),
                },
                "cpu": {
                    "percent": cpu_percent,
                    "num_threads": self.process.num_threads(),
                },
                "system": {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                    "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
                    "disk_usage_percent": psutil.disk_usage("/").percent,
                },
            }
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            return {"error": str(e)}

    def clear_metrics(self) -> None:
        """Clear all stored metrics and profiles."""
        self.execution_profiles.clear()
        self.global_metrics.clear()
        logger.info("🧹 Performance metrics cleared")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent()
        except Exception:
            return 0.0

    def _add_profile(self, profile: ExecutionProfile) -> None:
        """Add profile to the list, maintaining size limit."""
        self.execution_profiles.append(profile)

        # Keep only recent profiles
        if len(self.execution_profiles) > self.max_profiles:
            self.execution_profiles = self.execution_profiles[-self.max_profiles :]

    async def _check_performance_thresholds(self, profile: ExecutionProfile) -> None:
        """Check if performance thresholds were exceeded."""
        if profile.duration and profile.duration > self.thresholds["execution_time_warning"]:
            level = (
                "CRITICAL"
                if profile.duration > self.thresholds["execution_time_critical"]
                else "WARNING"
            )
            logger.warning(
                f"🐌 {level}: Slow execution detected - {profile.operation_name} "
                f"took {profile.duration:.2f}s (threshold: {self.thresholds['execution_time_warning']}s)"
            )

        if profile.memory_peak > self.thresholds["memory_usage_warning"]:
            level = (
                "CRITICAL"
                if profile.memory_peak > self.thresholds["memory_usage_critical"]
                else "WARNING"
            )
            logger.warning(
                f"🐏 {level}: High memory usage detected - {profile.operation_name} "
                f"used {profile.memory_peak:.1f}MB (threshold: {self.thresholds['memory_usage_warning']}MB)"
            )

    def __repr__(self) -> str:
        """String representation of performance monitor."""
        return (
            f"PerformanceMonitor(profiles={len(self.execution_profiles)}, "
            f"metrics={len(self.global_metrics)})"
        )
