#!/usr/bin/env python3
"""
LSP Performance Monitor - Comprehensive monitoring and health checking

This module provides real-time performance monitoring for the ruff LSP system,
including health checks, metrics collection, and alerting capabilities.

Features:
- Real-time LSP performance monitoring
- Health status tracking and alerting
- Response time analysis and trending
- Resource usage monitoring
- Dashboard integration for visual feedback
"""

import asyncio
import contextlib
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
import logging
from pathlib import Path
import statistics
import time
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable


class HealthStatus(Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class PerformanceMetrics:
    """LSP performance metrics"""

    # Timing metrics
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0

    # Request metrics
    requests_sent: int = 0
    responses_received: int = 0
    errors: int = 0
    timeouts: int = 0

    # Diagnostic metrics
    diagnostics_received: int = 0
    code_actions_generated: int = 0
    ai_suggestions_made: int = 0

    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # Connection metrics
    uptime_seconds: float = 0.0
    reconnections: int = 0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_size: int = 0

    timestamp: datetime = field(default_factory=datetime.now)


class LSPPerformanceMonitor:
    """
    Comprehensive LSP performance monitoring system

    Tracks all aspects of LSP performance including response times,
    resource usage, health status, and provides alerts for issues.
    """

    def __init__(
        self,
        metrics_file: Path | None = None,
        alert_thresholds: dict[str, float] | None = None,
    ):
        self.metrics_file = metrics_file or Path("/tmp/ruff-lsp-metrics.json")
        self.console = Console()
        self.logger = logging.getLogger("lsp-monitor")

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "response_time_ms": 1000.0,
            "error_rate_percent": 5.0,
            "memory_usage_mb": 500.0,
            "cpu_usage_percent": 80.0,
            "timeout_rate_percent": 10.0,
        }

        # Metrics storage
        self.current_metrics = PerformanceMetrics()
        self.metrics_history: list[PerformanceMetrics] = []
        self.response_times: list[float] = []
        self.max_history_size = 1000

        # Alert state
        self.alert_callbacks: list[Callable[[str, dict[str, Any]], None]] = []
        self.last_alerts: dict[str, datetime] = {}
        self.alert_cooldown = timedelta(minutes=5)

        # Monitoring state
        self.start_time = time.time()
        self.is_monitoring = False
        self.monitor_task: asyncio.Task | None = None

    async def start_monitoring(self) -> None:
        """Start performance monitoring"""
        self.logger.info("🚀 Starting LSP performance monitoring...")
        self.is_monitoring = True
        self.start_time = time.time()

        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("✅ Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.logger.info("Stopping LSP performance monitoring...")
        self.is_monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor_task

        # Save final metrics
        await self._save_metrics()
        self.logger.info("✅ Performance monitoring stopped")

    async def record_response_time(self, response_time_ms: float) -> None:
        """Record a response time measurement"""
        self.response_times.append(response_time_ms)

        # Limit memory usage
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-500:]  # Keep last 500

        # Update current metrics
        if self.response_times:
            self.current_metrics.avg_response_time_ms = statistics.mean(self.response_times)
            if len(self.response_times) >= 20:  # Need enough data for percentiles
                self.current_metrics.p95_response_time_ms = statistics.quantiles(
                    self.response_times, n=20
                )[18]  # 95th percentile
                self.current_metrics.p99_response_time_ms = statistics.quantiles(
                    self.response_times, n=100
                )[98]  # 99th percentile

    async def record_request(self) -> None:
        """Record a request sent"""
        self.current_metrics.requests_sent += 1

    async def record_response(self) -> None:
        """Record a response received"""
        self.current_metrics.responses_received += 1

    async def record_error(self) -> None:
        """Record an error"""
        self.current_metrics.errors += 1

    async def record_timeout(self) -> None:
        """Record a timeout"""
        self.current_metrics.timeouts += 1

    async def record_diagnostic(self) -> None:
        """Record diagnostic received"""
        self.current_metrics.diagnostics_received += 1

    async def record_code_action(self) -> None:
        """Record code action generated"""
        self.current_metrics.code_actions_generated += 1

    async def record_ai_suggestion(self) -> None:
        """Record AI suggestion made"""
        self.current_metrics.ai_suggestions_made += 1

    async def record_cache_hit(self) -> None:
        """Record cache hit"""
        self.current_metrics.cache_hits += 1

    async def record_cache_miss(self) -> None:
        """Record cache miss"""
        self.current_metrics.cache_misses += 1

    async def update_cache_size(self, size: int) -> None:
        """Update cache size"""
        self.current_metrics.cache_size = size
