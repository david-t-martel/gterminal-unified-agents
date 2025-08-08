"""Shared Performance Metrics class used across the codebase."""

from dataclasses import dataclass
from dataclasses import field
import time
from typing import Any


@dataclass
class PerformanceMetrics:
    """Unified performance metrics tracking."""

    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration: float | None = None

    # Resource metrics
    cpu_usage: float | None = None
    memory_usage: float | None = None

    # Request metrics
    request_count: int = 0
    error_count: int = 0
    success_rate: float | None = None

    # Latency metrics
    avg_latency: float | None = None
    p50_latency: float | None = None
    p95_latency: float | None = None
    p99_latency: float | None = None

    # Custom metrics
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Start timing."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop timing and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if self.request_count > 0 and self.error_count >= 0:
            self.success_rate = (self.request_count - self.error_count) / self.request_count * 100

    def add_request(self, success: bool = True) -> None:
        """Track a request."""
        self.request_count += 1
        if not success:
            self.error_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "duration": self.duration,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "avg_latency": self.avg_latency,
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "p99_latency": self.p99_latency,
            **self.custom_metrics,
        }
