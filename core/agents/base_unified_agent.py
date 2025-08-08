"""Base Unified Agent - Consolidated foundation for all agent implementations.

This module consolidates functionality from:
- BaseAgentService (job management, streaming, progress tracking)
- BaseAutomationAgent (MCP server, Gemini integration, common utilities)
- OptimizedBaseAgent (performance optimizations, advanced patterns)

Key Features:
- Unified job management with async execution and streaming
- Enhanced MCP server integration with FastMCP
- Google Gemini integration with Vertex AI SDK
- PyO3 Rust extensions integration for 5-10x performance
- Advanced caching, connection pooling, and circuit breakers
- Comprehensive error handling and logging
- Resource monitoring and performance metrics
"""

from abc import ABC
from abc import abstractmethod
import asyncio
from collections import defaultdict
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
import contextlib
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from enum import Enum
from enum import auto
import logging
from pathlib import Path
import time
import traceback
from typing import Any, Protocol, TypeVar, runtime_checkable
import uuid
from weakref import WeakValueDictionary

# Core dependencies
from fastmcp import FastMCP
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from vertexai.generative_models import GenerativeModel

# Project imports
from gterminal.automation.gemini_config import get_model_for_task
from gterminal.core.security.security_utils import safe_json_parse

# Optional PyO3 Rust extensions for performance
try:
    from fullstack_agent_rust import RustCache
    from fullstack_agent_rust import RustFileOps
    from fullstack_agent_rust import RustJsonProcessor

    RUST_EXTENSIONS_AVAILABLE = True
except ImportError:
    RUST_EXTENSIONS_AVAILABLE = False

# Optional performance monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Type variables
T = TypeVar("T")
P = TypeVar("P")
R = TypeVar("R")


class JobStatus(Enum):
    """Enhanced job execution status with additional states."""

    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()
    RETRYING = auto()


class Priority(Enum):
    """Job priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@pydantic_dataclass
class JobConfiguration:
    """Comprehensive job configuration with validation."""

    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: float = Field(default=300.0, gt=0, le=3600)
    priority: Priority = Field(default=Priority.NORMAL)
    retry_backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    max_retry_delay: float = Field(default=60.0, gt=0, le=300)
    circuit_breaker_threshold: int = Field(default=5, ge=1, le=20)
    circuit_breaker_timeout: float = Field(default=60.0, gt=0, le=600)
    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: float = Field(default=3600.0, gt=0)
    enable_streaming: bool = Field(default=False)
    batch_size: int = Field(default=10, ge=1, le=1000)
    memory_limit_mb: int = Field(default=512, ge=64, le=8192)


@runtime_checkable
class JobExecutor(Protocol):
    """Protocol for job execution implementations."""

    async def execute(
        self,
        job_id: str,
        parameters: dict[str, Any],
        config: JobConfiguration,
        progress_callback: Callable[[float, str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Execute a job with the given parameters."""
        ...


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    consecutive_failures: int = 0
    next_attempt_time: float | None = None


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""


class CircuitBreaker:
    """Circuit breaker implementation with exponential backoff."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._stats.state

    async def __aenter__(self):
        """Async context manager entry."""
        await self._check_state()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with state management."""
        if exc_type is None:
            await self._on_success()
        elif exc_type and issubclass(exc_type, self.expected_exception):
            await self._on_failure()
        return False  # Re-raise the exception

    async def _check_state(self) -> None:
        """Check and update circuit breaker state."""
        async with self._lock:
            current_time = time.time()
            if self._stats.state == CircuitBreakerState.OPEN:
                if self._stats.next_attempt_time and current_time >= self._stats.next_attempt_time:
                    self._stats.state = CircuitBreakerState.HALF_OPEN
                else:
                    msg = f"Circuit breaker is open. Next attempt in {self._stats.next_attempt_time - current_time:.1f}s"
                    raise CircuitBreakerOpenError(
                        msg,
                    )

    async def _on_success(self) -> None:
        """Handle successful execution."""
        async with self._lock:
            self._stats.success_count += 1
            self._stats.consecutive_failures = 0
            if self._stats.state == CircuitBreakerState.HALF_OPEN:
                self._stats.state = CircuitBreakerState.CLOSED

    async def _on_failure(self) -> None:
        """Handle failed execution."""
        async with self._lock:
            self._stats.failure_count += 1
            self._stats.consecutive_failures += 1
            self._stats.last_failure_time = time.time()
            if self._stats.consecutive_failures >= self.failure_threshold:
                self._stats.state = CircuitBreakerState.OPEN
                self._stats.next_attempt_time = time.time() + self.recovery_timeout


@dataclass
class JobMetrics:
    """Comprehensive job metrics tracking."""

    job_id: str
    correlation_id: str
    start_time: float
    end_time: float | None = None
    duration: float | None = None
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    retry_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_message: str | None = None
    stack_trace: str | None = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_completed(self) -> bool:
        """Check if job is in a completed state."""
        return self.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        }

    def complete(self, status: JobStatus, error: Exception | None = None) -> None:
        """Mark job as completed with final status."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        if error:
            self.error_message = str(error)
            self.stack_trace = traceback.format_exc()


class Job:
    """Enhanced job implementation with comprehensive tracking."""

    def __init__(self, job_id: str, job_type: str, parameters: dict[str, Any]) -> None:
        # Validate inputs
        if not job_id or not isinstance(job_id, str):
            msg = "Job ID must be a non-empty string"
            raise ValueError(msg)
        if not job_type or not isinstance(job_type, str):
            msg = "Job type must be a non-empty string"
            raise ValueError(msg)
        if not isinstance(parameters, dict):
            msg = "Parameters must be a dictionary"
            raise ValueError(msg)

        self.job_id = job_id.strip()
        self.job_type = job_type.strip()
        self.parameters = parameters.copy()
        self.status = JobStatus.PENDING
        self.created_at = datetime.now(UTC)
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.progress = 0.0
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.logs: list[str] = []

        # Progress tracking
        self._progress_callbacks: list[Callable[[float, str], Awaitable[None]]] = []
        self._cancel_event = asyncio.Event()
        self._pause_event = asyncio.Event()

    def start(self) -> None:
        """Mark job as started with validation."""
        if self.status != JobStatus.PENDING:
            msg = f"Cannot start job in {self.status.value} status"
            raise ValueError(msg)
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def complete(self, result: dict[str, Any]) -> None:
        """Mark job as completed with result validation."""
        if self.status != JobStatus.RUNNING:
            msg = f"Cannot complete job in {self.status.value} status"
            raise ValueError(msg)
        if not isinstance(result, dict):
            msg = "Result must be a dictionary"
            raise ValueError(msg)
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now(UTC)
        self.progress = 100.0
        self.result = result.copy()

    def fail(self, error: str) -> None:
        """Mark job as failed with error validation."""
        if not error or not isinstance(error, str):
            msg = "Error message must be a non-empty string"
            raise ValueError(msg)
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now(UTC)
        self.error = error.strip()

    def cancel(self) -> None:
        """Cancel the job."""
        if self.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            msg = f"Cannot cancel job in {self.status.value} status"
            raise ValueError(msg)
        self._cancel_event.set()
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.now(UTC)

    async def pause(self) -> None:
        """Pause job execution."""
        self._pause_event.set()
        self.status = JobStatus.PAUSED

    async def resume(self) -> None:
        """Resume job execution."""
        self._pause_event.clear()
        self.status = JobStatus.RUNNING

    @property
    def is_cancelled(self) -> bool:
        """Check if job is cancelled."""
        return self._cancel_event.is_set()

    @property
    def is_paused(self) -> bool:
        """Check if job is paused."""
        return self._pause_event.is_set()

    async def wait_if_paused(self) -> None:
        """Wait if job is paused."""
        if self.is_paused:
            await self._pause_event.wait()

    def add_log(self, message: str) -> None:
        """Add log message with validation."""
        if not message or not isinstance(message, str):
            return
        timestamp = datetime.now(UTC).isoformat()
        self.logs.append(f"[{timestamp}] {message.strip()}")
        # Limit log size to prevent memory issues
        if len(self.logs) > 1000:
            self.logs = self.logs[-500:]

    def update_progress(self, progress: float, message: str | None = None) -> None:
        """Update job progress with validation."""
        if not isinstance(progress, int | float):
            msg = "Progress must be a number"
            raise ValueError(msg)
        self.progress = max(0.0, min(100.0, float(progress)))
        if message:
            self.add_log(message)

    async def add_progress_callback(
        self, callback: Callable[[float, str], Awaitable[None]]
    ) -> None:
        """Add a progress callback."""
        self._progress_callbacks.append(callback)

    async def update_progress_async(self, progress: float, message: str = "") -> None:
        """Update job progress and notify callbacks."""
        self.progress = max(0.0, min(100.0, progress))
        if self._progress_callbacks:
            await asyncio.gather(
                *[callback(progress, message) for callback in self._progress_callbacks],
                return_exceptions=True,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary with enhanced data."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "parameters": self.parameters,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "logs": self.logs[-10:],
            "duration_seconds": self.get_duration(),
        }

    def get_duration(self) -> float | None:
        """Get job duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now(UTC)
        return (end_time - self.started_at).total_seconds()


class ResourceMonitor:
    """Monitor resource usage for memory and CPU optimization."""

    def __init__(self, check_interval: float = 1.0) -> None:
        self.check_interval = check_interval
        self._monitoring = False
        self._task: asyncio.Task | None = None
        self._metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    async def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return
        self._monitoring = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        if not PSUTIL_AVAILABLE:
            return

        try:
            import psutil

            process = psutil.Process()

            while self._monitoring:
                try:
                    # Memory usage
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    self._metrics["memory_mb"].append(memory_mb)

                    # CPU usage (averaged over interval)
                    cpu_percent = process.cpu_percent()
                    self._metrics["cpu_percent"].append(cpu_percent)

                    # Check for memory pressure
                    if memory_mb > 1024:  # 1GB threshold
                        logging.warning(f"High memory usage: {memory_mb:.1f} MB")

                    await asyncio.sleep(self.check_interval)

                except Exception as e:
                    logging.exception(f"Resource monitoring error: {e}")
                    await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            pass

    def get_current_usage(self) -> dict[str, float]:
        """Get current resource usage."""
        return {
            "memory_mb": (self._metrics["memory_mb"][-1] if self._metrics["memory_mb"] else 0.0),
            "cpu_percent": (
                self._metrics["cpu_percent"][-1] if self._metrics["cpu_percent"] else 0.0
            ),
        }

    def get_average_usage(self, window: int = 10) -> dict[str, float]:
        """Get average resource usage over a window."""
        result: dict[str, Any] = {}
        for metric, values in self._metrics.items():
            if values:
                recent_values = list(values)[-window:]
                result[metric] = sum(recent_values) / len(recent_values)
            else:
                result[metric] = 0.0
        return result


class BaseUnifiedAgent(ABC):
    """Unified base class for all agent implementations.

    Consolidates functionality from BaseAgentService, BaseAutomationAgent, and OptimizedBaseAgent:
    - Job management with async execution and streaming
    - MCP server integration with FastMCP
    - Google Gemini integration with Vertex AI SDK
    - PyO3 Rust extensions for performance
    - Advanced caching and resource management
    - Circuit breaker patterns and error handling
    - Comprehensive monitoring and metrics
    """

    def __init__(
        self,
        agent_name: str,
        description: str = "",
        max_concurrent_jobs: int = 10,
        enable_resource_monitoring: bool = True,
        enable_rust_extensions: bool = RUST_EXTENSIONS_AVAILABLE,
    ) -> None:
        if not agent_name or not isinstance(agent_name, str):
            msg = "Agent name must be a non-empty string"
            raise ValueError(msg)

        self.agent_name = agent_name.strip()
        self.description = description.strip()
        self.max_concurrent_jobs = max_concurrent_jobs
        self.enable_resource_monitoring = enable_resource_monitoring
        self.enable_rust_extensions = enable_rust_extensions and RUST_EXTENSIONS_AVAILABLE

        # Setup logging with enhanced format
        self.logger = self._setup_logging()

        # Initialize MCP server
        self.mcp = FastMCP(agent_name)

        # Job management
        self.jobs: dict[str, Job] = {}
        self.max_job_history = 1000
        self.running_jobs = 0
        self._job_queue: asyncio.Queue = asyncio.Queue()
        self._running_jobs: set[str] = set()
        self._job_semaphore = asyncio.Semaphore(max_concurrent_jobs)

        # Caching and performance
        self._model_cache: dict[str, GenerativeModel] = {}
        self._context_cache: WeakValueDictionary = WeakValueDictionary()
        self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}

        # PyO3 Rust extensions
        if self.enable_rust_extensions:
            try:
                self.rust_cache = RustCache(capacity=10000, ttl_seconds=3600)
                self.rust_file_ops = RustFileOps()
                self.rust_json_processor = RustJsonProcessor()
                self.logger.info("PyO3 Rust extensions initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Rust extensions: {e}")
                self.enable_rust_extensions = False

        # Circuit breakers per job type
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # Resource monitoring
        self._resource_monitor: ResourceMonitor | None = None
        if enable_resource_monitoring:
            self._resource_monitor = ResourceMonitor()

        # Background tasks
        self._background_tasks: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        # Performance metrics
        self._performance_metrics = {
            "total_jobs": 0,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "avg_execution_time": 0.0,
            "cache_hit_rate": 0.0,
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0,
        }

        self.logger.info(f"Initialized unified agent '{agent_name}': {description}")

    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging configuration."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        # Create formatter with more detail
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Add console handler if not already present
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    async def startup(self) -> None:
        """Initialize async components and start background tasks."""
        try:
            # Start resource monitoring
            if self._resource_monitor:
                await self._resource_monitor.start_monitoring()

            # Start background job processor
            task = asyncio.create_task(self._job_processor())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            # Start metrics collection
            task = asyncio.create_task(self._metrics_collector_loop())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            self.logger.info(f"Unified agent '{self.agent_name}' started successfully")

        except Exception as e:
            self.logger.exception(f"Failed to initialize agent: {e}")
            raise

    async def shutdown(self) -> None:
        """Graceful shutdown of the agent."""
        self.logger.info("Shutting down unified agent...")
        self._shutdown_event.set()

        # Cancel all background tasks
        for task in list(self._background_tasks):
            task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Cancel running jobs
        for job_id in list(self._running_jobs):
            if job_id in self.jobs:
                await self.jobs[job_id].cancel()

        # Stop resource monitoring
        if self._resource_monitor:
            await self._resource_monitor.stop_monitoring()

        self.logger.info("Agent shutdown complete")

    async def _job_processor(self) -> None:
        """Background task to process queued jobs."""
        while not self._shutdown_event.is_set():
            try:
                # Get next job from queue with timeout
                job = await asyncio.wait_for(self._job_queue.get(), timeout=1.0)

                # Process job with concurrency limit
                async with self._job_semaphore:
                    await self._execute_job_internal(job)

                self._job_queue.task_done()

            except TimeoutError:
                continue  # Check shutdown event
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Job processor error: {e}")

    async def _execute_job_internal(self, job: Job) -> None:
        """Internal job execution with full monitoring."""
        job_id = job.job_id
        self._running_jobs.add(job_id)

        try:
            self.logger.info(f"Starting job {job_id} of type {job.job_type}")
            start_time = time.time()

            job.start()
            job.add_log(f"Starting {job.job_type} execution")

            # Validate job parameters
            if not self.validate_job_parameters(job.job_type, job.parameters):
                error_msg = "Job parameter validation failed"
                job.fail(error_msg)
                return

            # Execute with circuit breaker
            circuit_breaker = self.get_circuit_breaker(job.job_type)

            async with circuit_breaker:
                result = await self._execute_job_implementation(job)
                job.complete(result)
                job.add_log("Job completed successfully")

            execution_time = time.time() - start_time

            # Update performance metrics
            self._performance_metrics["total_jobs"] += 1
            self._performance_metrics["successful_jobs"] += 1

            # Update average execution time
            current_avg = self._performance_metrics["avg_execution_time"]
            total_jobs = self._performance_metrics["total_jobs"]
            self._performance_metrics["avg_execution_time"] = (
                current_avg * (total_jobs - 1) + execution_time
            ) / total_jobs

            self.logger.info(f"Job {job_id} completed successfully in {execution_time:.2f}s")

        except asyncio.CancelledError:
            job.cancel()
            self.logger.info(f"Job {job_id} was cancelled")
        except Exception as e:
            self._performance_metrics["total_jobs"] += 1
            self._performance_metrics["failed_jobs"] += 1
            job.fail(str(e))
            self.logger.exception(f"Job {job_id} failed: {e}")
        finally:
            self._running_jobs.discard(job_id)

    async def _metrics_collector_loop(self) -> None:
        """Background task to collect performance metrics."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds

                # Get cache statistics
                total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
                if total_requests > 0:
                    self._performance_metrics["cache_hit_rate"] = (
                        self._cache_stats["hits"] / total_requests
                    )

                # Get resource usage
                if self._resource_monitor:
                    resource_usage = self._resource_monitor.get_current_usage()
                    self._performance_metrics["memory_usage_mb"] = resource_usage["memory_mb"]
                    self._performance_metrics["cpu_usage_percent"] = resource_usage["cpu_percent"]

                self.logger.debug(f"Performance metrics: {self._performance_metrics}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Metrics collection error: {e}")

    def get_model(self, task_type: str = "analysis") -> GenerativeModel:
        """Get Gemini model for specific task type with caching."""
        if task_type not in self._model_cache:
            self._model_cache[task_type] = get_model_for_task(task_type)
        return self._model_cache[task_type]

    def get_circuit_breaker(self, job_type: str) -> CircuitBreaker:
        """Get or create circuit breaker for job type."""
        if job_type not in self._circuit_breakers:
            self._circuit_breakers[job_type] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0,
            )
        return self._circuit_breakers[job_type]

    def create_job(
        self,
        job_type: str,
        parameters: dict[str, Any],
        config: JobConfiguration | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Create a new job and return job ID."""
        if not job_type or not isinstance(job_type, str):
            msg = "Job type must be a non-empty string"
            raise ValueError(msg)
        if not isinstance(parameters, dict):
            msg = "Parameters must be a dictionary"
            raise ValueError(msg)

        job_id = str(uuid.uuid4())
        job = Job(job_id, job_type, parameters)

        # Cleanup old jobs if needed
        if len(self.jobs) >= self.max_job_history:
            self._cleanup_old_jobs(keep_recent=self.max_job_history // 2)

        self.jobs[job_id] = job
        self.logger.info(f"Created job {job_id} of type {job_type}")
        return job_id

    async def execute_job_async(
        self, job_id: str, wait_for_completion: bool = False
    ) -> dict[str, Any]:
        """Execute job asynchronously with optional waiting."""
        if job_id not in self.jobs:
            return self.create_error_response(f"Job {job_id} not found")

        job = self.jobs[job_id]

        # Queue job for execution
        await self._job_queue.put(job)

        if wait_for_completion:
            # Wait for job to complete
            while job.status not in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ]:
                await asyncio.sleep(0.1)

            if job.status == JobStatus.COMPLETED:
                return self.create_success_response(
                    {"job": job.to_dict(), "result": job.result},
                    "Job completed successfully",
                )
            return self.create_error_response(
                f"Job failed with status: {job.status.name}",
                {"job": job.to_dict()},
            )

        return self.create_success_response(
            {"job_id": job_id, "status": "queued"}, "Job queued for execution"
        )

    async def stream_job_progress(self, job_id: str) -> AsyncGenerator[dict[str, Any], None]:
        """Stream job progress updates."""
        if job_id not in self.jobs:
            yield {"error": f"Job {job_id} not found"}
            return

        job = self.jobs[job_id]
        progress_updates: asyncio.Queue = asyncio.Queue()

        # Subscribe to progress updates
        async def progress_callback(progress: float, message: str) -> None:
            await progress_updates.put(
                {
                    "job_id": job_id,
                    "progress": progress,
                    "message": message,
                    "timestamp": time.time(),
                    "status": job.status.name,
                },
            )

        await job.add_progress_callback(progress_callback)

        try:
            # Stream real-time updates
            while job.status not in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ]:
                try:
                    update = await asyncio.wait_for(progress_updates.get(), timeout=1.0)
                    yield update
                except TimeoutError:
                    # Send heartbeat
                    yield {
                        "job_id": job_id,
                        "type": "heartbeat",
                        "timestamp": time.time(),
                    }

            # Send final status
            yield {
                "job_id": job_id,
                "type": "completion",
                "status": job.status.name,
                "timestamp": time.time(),
                "duration": job.get_duration(),
            }

        except asyncio.CancelledError:
            self.logger.info(f"Streaming cancelled for job {job_id}")

    async def generate_with_progress(
        self,
        prompt: str,
        task_type: str = "analysis",
        job: Job | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str | dict[str, Any] | None:
        """Generate content with Gemini while tracking progress."""
        if not prompt or not isinstance(prompt, str):
            self.logger.error("Invalid prompt provided")
            return None

        try:
            if job:
                job.update_progress(10.0, "Initializing model...")
            if progress_callback:
                progress_callback(10.0, "Initializing model...")

            model = self.get_model(task_type)

            if job:
                job.update_progress(30.0, "Sending request to Gemini...")
            if progress_callback:
                progress_callback(30.0, "Sending request to Gemini...")

            # Generate content
            response = model.generate_content(prompt.strip(), stream=False)

            if job:
                job.update_progress(80.0, "Processing response...")
            if progress_callback:
                progress_callback(80.0, "Processing response...")

            content = response.text

            if job:
                job.update_progress(100.0, "Generation completed")
            if progress_callback:
                progress_callback(100.0, "Generation completed")

            return content

        except Exception as e:
            error_msg = f"Gemini generation failed: {e}"
            if job:
                job.add_log(error_msg)
            self.logger.exception(error_msg)
            return None

    def create_success_response(
        self,
        data: dict[str, Any],
        message: str = "Operation completed successfully",
    ) -> dict[str, Any]:
        """Create standardized success response."""
        return {
            "status": "success",
            "message": message,
            "timestamp": datetime.now(UTC).isoformat(),
            "agent": self.agent_name,
            **data,
        }

    def create_error_response(
        self, error: str | Exception, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create standardized error response."""
        error_msg = str(error)
        self.logger.error(f"Error in {self.agent_name}: {error_msg}")

        response = {
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now(UTC).isoformat(),
            "agent": self.agent_name,
        }

        if context:
            response["context"] = context

        return response

    def safe_file_read(self, file_path: str | Path) -> str | None:
        """Safely read file content with Rust optimization."""
        try:
            # Use Rust extensions if available for better performance
            if self.enable_rust_extensions:
                try:
                    return self.rust_file_ops.read_file(str(file_path))
                except Exception as e:
                    self.logger.debug(f"Rust file read failed, using Python fallback: {e}")

            # Python fallback
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.exception(f"Failed to read file {file_path}: {e}")
            return None

    def safe_file_write(
        self, file_path: str | Path, content: str, create_dirs: bool = True
    ) -> bool:
        """Safely write content to file with Rust optimization."""
        try:
            # Use Rust extensions if available
            if self.enable_rust_extensions:
                try:
                    self.rust_file_ops.write_file(str(file_path), content, create_dirs)
                    return True
                except Exception as e:
                    self.logger.debug(f"Rust file write failed, using Python fallback: {e}")

            # Python fallback
            path = Path(file_path)
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            self.logger.exception(f"Failed to write file {file_path}: {e}")
            return False

    async def generate_with_gemini(
        self,
        prompt: str,
        task_type: str = "analysis",
        parse_json: bool = False,
    ) -> str | dict[str, Any] | None:
        """Generate content using Gemini with error handling."""
        try:
            model = self.get_model(task_type)
            response = model.generate_content(prompt)
            content = response.text

            if parse_json:
                if self.enable_rust_extensions:
                    try:
                        return self.rust_json_processor.parse(content)
                    except Exception as e:
                        self.logger.debug(f"Rust JSON parsing failed, using Python: {e}")
                return safe_json_parse(content)
            return content

        except Exception as e:
            self.logger.exception(f"Gemini generation failed: {e}")
            return None

    def validate_job_parameters(self, job_type: str, parameters: dict[str, Any]) -> bool:
        """Validate job parameters for specific job type."""
        if not isinstance(parameters, dict):
            self.logger.error("Parameters must be a dictionary")
            return False

        # Base validation - override in subclasses
        required_params = self.get_required_parameters(job_type)

        for param in required_params:
            if param not in parameters:
                self.logger.error(f"Missing required parameter: {param}")
                return False
            if parameters[param] is None:
                self.logger.error(f"Parameter cannot be None: {param}")
                return False

        return True

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Get required parameters for job type. Override in subclasses."""
        return []

    def _cleanup_old_jobs(self, keep_recent: int = 500) -> None:
        """Clean up old completed jobs."""
        if len(self.jobs) <= keep_recent:
            return

        # Sort jobs by creation time and keep the most recent
        sorted_jobs = sorted(self.jobs.items(), key=lambda x: x[1].created_at, reverse=True)
        jobs_to_keep = dict(sorted_jobs[:keep_recent])
        removed_count = len(self.jobs) - len(jobs_to_keep)
        self.jobs = jobs_to_keep

        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old jobs")

    def get_agent_stats(self) -> dict[str, Any]:
        """Get comprehensive agent statistics."""
        total_jobs = len(self.jobs)
        completed_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.COMPLETED)
        failed_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.FAILED)
        running_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.RUNNING)

        # Calculate average duration for completed jobs
        completed_durations = [
            job.get_duration()
            for job in self.jobs.values()
            if job.status == JobStatus.COMPLETED and job.get_duration() is not None
        ]
        avg_duration = (
            sum(completed_durations) / len(completed_durations) if completed_durations else 0
        )

        stats = {
            "agent_name": self.agent_name,
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "running_jobs": running_jobs,
            "success_rate": completed_jobs / max(total_jobs, 1) * 100,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "average_duration_seconds": round(avg_duration, 2),
            "current_running_jobs": len(self._running_jobs),
            "rust_extensions_enabled": self.enable_rust_extensions,
            "resource_monitoring_enabled": bool(self._resource_monitor),
        }

        # Add performance metrics
        stats.update(self._performance_metrics)

        # Add resource usage if available
        if self._resource_monitor:
            resource_usage = self._resource_monitor.get_current_usage()
            stats["current_memory_mb"] = resource_usage["memory_mb"]
            stats["current_cpu_percent"] = resource_usage["cpu_percent"]

        return stats

    @abstractmethod
    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Execute the specific job implementation. Must be implemented by subclasses."""

    @abstractmethod
    def register_tools(self) -> None:
        """Register MCP tools for this agent. Must be implemented by subclasses."""

    def run(self) -> None:
        """Run the MCP server after registering tools."""
        self.register_tools()
        self.mcp.run()
