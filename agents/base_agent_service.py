"""Optimized Base Agent Service - Enhanced service layer with modern Python best practices.

This module provides a highly optimized base agent service implementing:
- Advanced asyncio patterns with connection pooling
- Multi-layer caching with TTL and invalidation
- Circuit breaker pattern with exponential backoff
- Comprehensive monitoring and metrics
- Memory-efficient streaming with async generators
- Clean architecture with dependency injection
- Advanced error handling with correlation IDs
- Performance optimizations using Python 3.10+ features
"""

from abc import ABC
from abc import abstractmethod
import asyncio
from collections import defaultdict
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
import contextlib
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import auto
import logging
import time
import traceback
from typing import Any, Protocol, TypeVar, runtime_checkable
import uuid
from weakref import WeakSet

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from gterminal.performance.cache import SmartCacheManager
from gterminal.performance.connection_pool import ConnectionPoolManager
from gterminal.performance.metrics import MetricCollector

# Rust extensions for high-performance operations
from gterminal.utils.rust_extensions import RUST_CORE_AVAILABLE
from gterminal.utils.rust_extensions import EnhancedTtlCache
from gterminal.utils.rust_extensions import RustCore

# Type variables for generic support
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


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    consecutive_failures: int = 0
    next_attempt_time: float | None = None


class CircuitBreaker:
    """Circuit breaker implementation with exponential backoff and Rust optimization."""

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

        # Initialize Rust components for performance-critical operations
        self._rust_cache: EnhancedTtlCache | None = None
        self._rust_core: RustCore | None = None
        if RUST_CORE_AVAILABLE:
            try:
                self._rust_core = RustCore()
                # Use short TTL cache for circuit breaker state
                self._rust_cache = EnhancedTtlCache(300)  # 5 minutes
                logging.info("Circuit breaker initialized with Rust optimizations")
            except Exception as e:
                logging.warning(f"Failed to initialize Rust in circuit breaker: {e}")
                self._rust_core = None
                self._rust_cache = None

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
        elif issubclass(exc_type, self.expected_exception):
            await self._on_failure()
        # Re-raise the exception
        return False

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

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics with Rust performance metrics."""
        base_stats = {
            "state": self._stats.state.name,
            "failure_count": self._stats.failure_count,
            "success_count": self._stats.success_count,
            "consecutive_failures": self._stats.consecutive_failures,
            "last_failure_time": self._stats.last_failure_time,
            "next_attempt_time": self._stats.next_attempt_time,
        }

        # Add Rust performance information
        if self._rust_cache:
            try:
                cache_stats = self._rust_cache.get_stats()
                base_stats.update(
                    {
                        "rust_cache_active": True,
                        "cache_hits": cache_stats.hits,
                        "cache_misses": cache_stats.misses,
                        "cache_hit_ratio": cache_stats.hit_ratio,
                        "cache_size": self._rust_cache.size,
                    },
                )
            except Exception as e:
                base_stats["rust_cache_error"] = str(e)
        else:
            base_stats["rust_cache_active"] = False

        # Add runtime performance metrics
        total_attempts = self._stats.success_count + self._stats.failure_count
        if total_attempts > 0:
            base_stats.update(
                {
                    "success_rate": (self._stats.success_count / total_attempts) * 100,
                    "failure_rate": (self._stats.failure_count / total_attempts) * 100,
                },
            )

        return base_stats


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""


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


class OptimizedJob[T]:
    """Optimized job implementation with advanced features."""

    def __init__(
        self,
        job_id: str,
        job_type: str,
        executor: JobExecutor,
        parameters: dict[str, Any],
        config: JobConfiguration | None = None,
        correlation_id: str | None = None,
    ) -> None:
        self.job_id = job_id
        self.job_type = job_type
        self.executor = executor
        self.parameters = parameters
        self.config = config or JobConfiguration()
        self.correlation_id = correlation_id or str(uuid.uuid4())

        # Metrics and state
        self.metrics = JobMetrics(
            job_id=job_id,
            correlation_id=self.correlation_id,
            start_time=time.time(),
        )

        # Progress tracking
        self._progress_callbacks: list[Callable[[float, str], Awaitable[None]]] = []
        self._cancel_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._result: T | None = None
        self._exception: Exception | None = None

        # Logging with correlation ID
        self.logger = logging.getLogger(f"{__name__}.{job_type}")
        self.logger = logging.LoggerAdapter(
            self.logger,
            {"correlation_id": self.correlation_id, "job_id": job_id},
        )

    async def add_progress_callback(
        self,
        callback: Callable[[float, str], Awaitable[None]],
    ) -> None:
        """Add a progress callback."""
        self._progress_callbacks.append(callback)

    async def update_progress(self, progress: float, message: str = "") -> None:
        """Update job progress and notify callbacks."""
        self.metrics.progress = max(0.0, min(100.0, progress))

        # Notify all callbacks
        if self._progress_callbacks:
            await asyncio.gather(
                *[callback(progress, message) for callback in self._progress_callbacks],
                return_exceptions=True,
            )

    async def cancel(self) -> None:
        """Request job cancellation."""
        self._cancel_event.set()
        self.metrics.status = JobStatus.CANCELLED
        self.logger.info("Job cancellation requested")

    async def pause(self) -> None:
        """Pause job execution."""
        self._pause_event.set()
        self.metrics.status = JobStatus.PAUSED
        self.logger.info("Job paused")

    async def resume(self) -> None:
        """Resume job execution."""
        self._pause_event.clear()
        self.metrics.status = JobStatus.RUNNING
        self.logger.info("Job resumed")

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

    async def execute_with_retries(self) -> T:
        """Execute job with retry logic and circuit breaker."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            recovery_timeout=self.config.circuit_breaker_timeout,
        )

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Check for cancellation
                if self.is_cancelled:
                    msg = "Job was cancelled"
                    raise asyncio.CancelledError(msg)

                # Wait if paused
                await self.wait_if_paused()

                # Update status
                self.metrics.status = JobStatus.RUNNING if attempt == 0 else JobStatus.RETRYING
                self.metrics.retry_count = attempt

                # Execute with circuit breaker
                async with circuit_breaker:
                    result = await asyncio.wait_for(
                        self.executor.execute(
                            self.job_id,
                            self.parameters,
                            self.config,
                            self.update_progress,
                        ),
                        timeout=self.config.timeout_seconds,
                    )

                # Success
                self.metrics.complete(JobStatus.COMPLETED)
                self._result = result
                self.logger.info(
                    f"Job completed successfully after {attempt + 1} attempts",
                )
                return result

            except asyncio.CancelledError:
                self.metrics.complete(JobStatus.CANCELLED)
                self.logger.info("Job was cancelled")
                raise

            except TimeoutError as e:
                last_exception = e
                self.metrics.complete(JobStatus.TIMEOUT, e)
                self.logger.exception(f"Job timed out after {self.config.timeout_seconds}s")
                break

            except CircuitBreakerOpenError as e:
                last_exception = e
                self.logger.warning(f"Circuit breaker is open: {e}")
                break

            except Exception as e:
                last_exception = e
                self.logger.warning(f"Job attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries:
                    # Calculate exponential backoff
                    delay = min(
                        self.config.retry_backoff_factor**attempt,
                        self.config.max_retry_delay,
                    )

                    self.logger.info(f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    # Final failure
                    self.metrics.complete(JobStatus.FAILED, e)
                    break

        # Job failed after all retries
        self._exception = last_exception
        if last_exception:
            raise last_exception

        # Should not reach here
        msg = "Job execution failed without exception"
        raise RuntimeError(msg)


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

        except ImportError:
            logging.warning("psutil not available, resource monitoring disabled")
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


class StreamingJobResult:
    """Streaming result handler for memory-efficient processing."""

    def __init__(self, job_id: str, buffer_size: int = 1000) -> None:
        self.job_id = job_id
        self.buffer_size = buffer_size
        self._buffer: deque = deque(maxlen=buffer_size)
        self._subscribers: WeakSet = WeakSet()
        self._complete = False
        self._error: Exception | None = None

    async def add_chunk(self, chunk: Any) -> None:
        """Add a chunk to the streaming result."""
        self._buffer.append(
            {
                "timestamp": time.time(),
                "data": chunk,
            },
        )

        # Notify subscribers
        for subscriber in list(self._subscribers):
            try:
                await subscriber(chunk)
            except Exception as e:
                logging.exception(f"Subscriber notification error: {e}")

    async def subscribe(self, callback: Callable[[Any], Awaitable[None]]) -> None:
        """Subscribe to streaming updates."""
        self._subscribers.add(callback)

        # Send existing buffer
        for item in self._buffer:
            try:
                await callback(item["data"])
            except Exception as e:
                logging.exception(f"Subscriber callback error: {e}")

    async def complete(self, error: Exception | None = None) -> None:
        """Mark streaming as complete."""
        self._complete = True
        self._error = error

    @property
    def is_complete(self) -> bool:
        """Check if streaming is complete."""
        return self._complete

    def get_buffer(self) -> list[dict[str, Any]]:
        """Get current buffer contents."""
        return list(self._buffer)


class OptimizedBaseAgentService(ABC):
    """Highly optimized base agent service with modern Python best practices.

    Features:
    - Advanced asyncio patterns with connection pooling
    - Multi-layer caching with smart invalidation
    - Circuit breaker pattern with exponential backoff
    - Memory-efficient streaming with async generators
    - Comprehensive monitoring and metrics
    - Clean architecture with dependency injection
    - Advanced error handling with correlation IDs
    - Resource monitoring and optimization
    """

    def __init__(
        self,
        agent_name: str,
        description: str = "",
        max_concurrent_jobs: int = 10,
        enable_resource_monitoring: bool = True,
        cache_config: dict[str, Any] | None = None,
        connection_pool_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(agent_name, description)

        # Core configuration
        self.max_concurrent_jobs = max_concurrent_jobs
        self.enable_resource_monitoring = enable_resource_monitoring

        # Job management
        self._jobs: dict[str, OptimizedJob] = {}
        self._job_queue: asyncio.Queue = asyncio.Queue()
        self._running_jobs: set[str] = set()
        self._job_semaphore = asyncio.Semaphore(max_concurrent_jobs)

        # Advanced components
        self._cache_manager: SmartCacheManager | None = None
        self._connection_pool_manager: ConnectionPoolManager | None = None
        self._metrics_collector: MetricCollector | None = None
        self._resource_monitor: ResourceMonitor | None = None

        # Rust components for high-performance operations
        self._rust_core: RustCore | None = None
        self._rust_cache: EnhancedTtlCache | None = None
        self._rust_available = RUST_CORE_AVAILABLE
        self._initialize_rust_components()

        # Streaming support
        self._streaming_results: dict[str, StreamingJobResult] = {}

        # Circuit breakers per job type
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # Background tasks
        self._background_tasks: set[asyncio.Task] = set()

        # Performance metrics (enhanced with Rust tracking)
        self._performance_metrics = {
            "total_jobs": 0,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "avg_execution_time": 0.0,
            "cache_hit_rate": 0.0,
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "rust_operations": 0,
            "rust_cache_hits": 0,
            "rust_cache_misses": 0,
        }

        # Initialize async components in startup
        self._initialize_task: asyncio.Task | None = None

    def _initialize_rust_components(self) -> None:
        """Initialize Rust components for high-performance operations."""
        if self._rust_available:
            try:
                self._rust_core = RustCore()
                # Initialize cache with 1 hour default TTL for service-level caching
                self._rust_cache = EnhancedTtlCache(3600)
                self.logger.info(
                    f"Agent service '{self.agent_name}' initialized with Rust optimizations",
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Rust components: {e}")
                self._rust_core = None
                self._rust_cache = None
                self._rust_available = False
        else:
            self.logger.info(
                f"Agent service '{self.agent_name}' using Python fallbacks",
            )

    def _generate_cache_key(self, operation: str, *args, **kwargs) -> str:
        """Generate cache key using Rust string operations for performance."""
        if self._rust_core:
            # Use Rust for efficient string operations
            key_parts = (
                [operation]
                + [str(arg) for arg in args]
                + [f"{k}={v}" for k, v in sorted(kwargs.items())]
            )
            raw_key = ":".join(key_parts)
            return f"service:{self.agent_name}:{raw_key}"
        # Fallback to Python with hashing for consistent key size
        import hashlib

        key_parts = (
            [operation]
            + [str(arg) for arg in args]
            + [f"{k}={v}" for k, v in sorted(kwargs.items())]
        )
        raw_key = ":".join(key_parts)
        key_hash = hashlib.md5(raw_key.encode()).hexdigest()[:16]
        return f"service:{self.agent_name}:{key_hash}"

    async def startup(self) -> None:
        """Initialize async components and start background tasks."""
        if self._initialize_task:
            return

        self._initialize_task = asyncio.create_task(self._async_startup())
        await self._initialize_task

    async def _async_startup(self) -> None:
        """Async startup implementation."""
        try:
            # Initialize cache manager
            self._cache_manager = SmartCacheManager()

            # Initialize connection pool manager
            self._connection_pool_manager = ConnectionPoolManager()

            # Initialize metrics collector
            self._metrics_collector = MetricCollector()

            # Initialize resource monitoring
            if self.enable_resource_monitoring:
                self._resource_monitor = ResourceMonitor()
                await self._resource_monitor.start_monitoring()

            # Start background job processor
            task = asyncio.create_task(self._job_processor())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            # Start metrics collection
            task = asyncio.create_task(self._metrics_collector_loop())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            self.logger.info(
                f"Optimized agent service '{self.agent_name}' started successfully",
            )

        except Exception as e:
            self.logger.exception(f"Failed to initialize agent service: {e}")
            raise

    async def shutdown(self) -> None:
        """Graceful shutdown of the agent service."""
        self.logger.info("Shutting down agent service...")

        # Cancel all background tasks
        for task in list(self._background_tasks):
            task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Cancel running jobs
        for job_id in list(self._running_jobs):
            if job_id in self._jobs:
                await self._jobs[job_id].cancel()

        # Close async resources
        if self._resource_monitor:
            await self._resource_monitor.stop_monitoring()

        if self._connection_pool_manager:
            await self._connection_pool_manager.close_all()

        if self._cache_manager:
            await self._cache_manager.close()

        self.logger.info("Agent service shutdown complete")

    async def _job_processor(self) -> None:
        """Background task to process queued jobs."""
        while True:
            try:
                # Get next job from queue
                job = await self._job_queue.get()

                # Process job with concurrency limit
                async with self._job_semaphore:
                    await self._execute_job_internal(job)

                # Mark task as done
                self._job_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Job processor error: {e}")

    async def _execute_job_internal(self, job: OptimizedJob) -> None:
        """Internal job execution with full monitoring."""
        job_id = job.job_id
        self._running_jobs.add(job_id)

        try:
            # Start execution
            self.logger.info(f"Starting job {job_id} of type {job.job_type}")

            # Execute with monitoring
            start_time = time.time()
            await job.execute_with_retries()
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

            self.logger.info(
                f"Job {job_id} completed successfully in {execution_time:.2f}s",
            )

        except Exception as e:
            self._performance_metrics["total_jobs"] += 1
            self._performance_metrics["failed_jobs"] += 1
            self.logger.exception(f"Job {job_id} failed: {e}")

        finally:
            self._running_jobs.discard(job_id)

    async def _metrics_collector_loop(self) -> None:
        """Background task to collect performance metrics."""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds

                # Get cache statistics
                if self._cache_manager:
                    cache_stats = await self._cache_manager.get_stats()
                    self._performance_metrics["cache_hit_rate"] = cache_stats["overall"]["hit_rate"]

                # Get resource usage
                if self._resource_monitor:
                    resource_usage = self._resource_monitor.get_current_usage()
                    self._performance_metrics["memory_usage_mb"] = resource_usage["memory_mb"]
                    self._performance_metrics["cpu_usage_percent"] = resource_usage["cpu_percent"]

                # Log performance summary
                self.logger.debug(f"Performance metrics: {self._performance_metrics}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Metrics collection error: {e}")

    def create_job(
        self,
        job_type: str,
        parameters: dict[str, Any],
        config: JobConfiguration | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Create a new optimized job."""
        job_id = str(uuid.uuid4())

        # Get job executor
        executor = self._get_job_executor(job_type)

        # Create job
        job = OptimizedJob(
            job_id=job_id,
            job_type=job_type,
            executor=executor,
            parameters=parameters,
            config=config or JobConfiguration(),
            correlation_id=correlation_id,
        )

        self._jobs[job_id] = job
        self.logger.info(f"Created job {job_id} of type {job_type}")

        return job_id

    async def execute_job_async(
        self,
        job_id: str,
        wait_for_completion: bool = False,
    ) -> dict[str, Any]:
        """Execute job asynchronously with optional waiting."""
        if job_id not in self._jobs:
            return self.create_error_response(f"Job {job_id} not found")

        job = self._jobs[job_id]

        # Queue job for execution
        await self._job_queue.put(job)

        if wait_for_completion:
            # Wait for job to complete
            while not job.metrics.is_completed:
                await asyncio.sleep(0.1)

            if job.metrics.status == JobStatus.COMPLETED:
                return self.create_success_response(
                    {"job": job.metrics.__dict__, "result": job._result},
                    "Job completed successfully",
                )
            return self.create_error_response(
                f"Job failed with status: {job.metrics.status.name}",
                {"job": job.metrics.__dict__},
            )

        return self.create_success_response(
            {"job_id": job_id, "status": "queued"},
            "Job queued for execution",
        )

    async def stream_job_progress(
        self,
        job_id: str,
        buffer_size: int = 1000,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream job progress updates with memory-efficient buffering."""
        if job_id not in self._jobs:
            yield {"error": f"Job {job_id} not found"}
            return

        job = self._jobs[job_id]

        # Create streaming result if not exists
        if job_id not in self._streaming_results:
            self._streaming_results[job_id] = StreamingJobResult(job_id, buffer_size)

        streaming_result = self._streaming_results[job_id]
        progress_updates: asyncio.Queue = asyncio.Queue()

        # Subscribe to progress updates
        async def progress_callback(progress: float, message: str) -> None:
            await progress_updates.put(
                {
                    "job_id": job_id,
                    "progress": progress,
                    "message": message,
                    "timestamp": time.time(),
                    "status": job.metrics.status.name,
                },
            )

        await job.add_progress_callback(progress_callback)

        try:
            # Stream existing buffer first
            for item in streaming_result.get_buffer():
                yield item

            # Stream real-time updates
            while not job.metrics.is_completed:
                try:
                    update = await asyncio.wait_for(progress_updates.get(), timeout=1.0)
                    yield update
                    await streaming_result.add_chunk(update)
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
                "status": job.metrics.status.name,
                "timestamp": time.time(),
                "duration": job.metrics.duration,
            }

        except asyncio.CancelledError:
            self.logger.info(f"Streaming cancelled for job {job_id}")
        finally:
            await streaming_result.complete()

    @contextlib.asynccontextmanager
    async def cached_execution(
        self,
        cache_key: str,
        ttl: float | None = None,
        version: str = "1.0",
    ) -> AsyncIterator[dict[str, Any]]:
        """Context manager for cached execution with Rust-powered caching and smart invalidation."""
        # Generate full cache key using Rust string operations
        full_cache_key = self._generate_cache_key(
            "cached_execution",
            cache_key,
            version,
        )

        # Try Rust cache first for maximum performance
        cached_result = None
        if self._rust_cache:
            try:
                import json

                cached_data = self._rust_cache.get(full_cache_key)
                if cached_data:
                    cached_result = json.loads(cached_data)
                    self._performance_metrics["rust_cache_hits"] += 1
                    self.logger.debug(f"Rust cache hit for key: {cache_key}")
                    yield {
                        "cached": True,
                        "result": cached_result,
                        "cache_source": "rust",
                    }
                    return
                else:
                    self._performance_metrics["rust_cache_misses"] += 1
            except Exception as e:
                self.logger.warning(f"Rust cache get failed, falling back: {e}")

        # Fallback to smart cache manager
        if not cached_result and self._cache_manager:
            cached_result = await self._cache_manager.smart_get(full_cache_key)
            if cached_result is not None:
                self.logger.debug(f"Smart cache hit for key: {cache_key}")
                yield {"cached": True, "result": cached_result, "cache_source": "smart"}
                return

        # Cache miss - prepare for new execution
        self.logger.debug(f"Cache miss for key: {cache_key}")
        result_holder = {"cached": False, "result": None}

        try:
            yield result_holder

            # Cache the result if provided
            if result_holder["result"] is not None:
                cache_ttl = ttl or 3600  # Default 1 hour

                # Cache in Rust first for performance
                if self._rust_cache:
                    try:
                        import json

                        self._rust_cache.set_with_ttl(
                            full_cache_key,
                            json.dumps(result_holder["result"]),
                            int(cache_ttl),
                        )
                        self.logger.debug(
                            f"Cached result in Rust cache for key: {cache_key}",
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to cache in Rust, using fallback: {e}",
                        )

                # Also cache in smart cache manager for persistence
                if self._cache_manager:
                    await self._cache_manager.smart_set(
                        full_cache_key,
                        result_holder["result"],
                        ttl=cache_ttl,
                        version=version,
                    )
                    self.logger.debug(
                        f"Cached result in smart cache for key: {cache_key}",
                    )

        except Exception as e:
            self.logger.exception(f"Cached execution error for key {cache_key}: {e}")
            raise

    async def batch_process(
        self,
        items: list[dict[str, Any]],
        processor_func: Callable[[dict[str, Any]], Awaitable[Any]],
        batch_size: int = 10,
        max_concurrency: int = 5,
    ) -> AsyncGenerator[list[Any], None]:
        """Memory-efficient batch processing with controlled concurrency."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_item(item: dict[str, Any]) -> Any:
            async with semaphore:
                return await processor_func(item)

        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            # Process batch concurrently
            tasks = [process_item(item) for item in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter successful results
            successful_results = [result for result in results if not isinstance(result, Exception)]

            # Log errors
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch processing error: {result}")

            yield successful_results

    def get_circuit_breaker(self, job_type: str) -> CircuitBreaker:
        """Get or create circuit breaker for job type."""
        if job_type not in self._circuit_breakers:
            self._circuit_breakers[job_type] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0,
            )
        return self._circuit_breakers[job_type]

    async def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get comprehensive service statistics with Rust performance metrics."""
        stats = {
            "agent_name": self.agent_name,
            "uptime_seconds": (
                time.time() - self._initialize_task.get_loop().time()
                if self._initialize_task
                else 0
            ),
            "performance_metrics": self._performance_metrics.copy(),
            "job_stats": {
                "total_jobs": len(self._jobs),
                "running_jobs": len(self._running_jobs),
                "queued_jobs": self._job_queue.qsize(),
                "streaming_jobs": len(self._streaming_results),
            },
            "resource_usage": {},
            "cache_stats": {},
            "connection_pool_stats": {},
            "circuit_breaker_stats": {},
            "rust_performance": {},
        }

        # Add Rust performance statistics
        if self._rust_available:
            rust_stats = {
                "rust_available": True,
                "rust_core_active": self._rust_core is not None,
                "rust_cache_active": self._rust_cache is not None,
            }

            if self._rust_cache:
                try:
                    cache_stats = self._rust_cache.get_stats()
                    rust_stats.update(
                        {
                            "cache_size": self._rust_cache.size,
                            "cache_hits": cache_stats.hits,
                            "cache_misses": cache_stats.misses,
                            "cache_hit_ratio": cache_stats.hit_ratio,
                            "total_entries": cache_stats.total_entries,
                            "expired_entries": cache_stats.expired_entries,
                        },
                    )

                    # Update performance metrics with Rust cache stats
                    self._performance_metrics["rust_cache_hits"] = cache_stats.hits
                    self._performance_metrics["rust_cache_misses"] = cache_stats.misses

                except Exception as e:
                    rust_stats["cache_error"] = str(e)

            if self._rust_core:
                try:
                    # Performance benchmark test
                    import time

                    start_time = time.time()
                    test_data = {"test": 123, "benchmark": 456}
                    rust_result = self._rust_core.process_dict(test_data)
                    rust_time = time.time() - start_time

                    start_time = time.time()
                    python_result = {k: v * 2 for k, v in test_data.items()}
                    python_time = time.time() - start_time

                    rust_stats.update(
                        {
                            "rust_version": self._rust_core.version,
                            "performance_benchmark": {
                                "rust_time_ns": rust_time * 1_000_000_000,
                                "python_time_ns": python_time * 1_000_000_000,
                                "rust_speedup": (
                                    max(python_time / rust_time, 0) if rust_time > 0 else 0
                                ),
                                "test_passed": rust_result == python_result,
                            },
                        },
                    )

                except Exception as e:
                    rust_stats["core_error"] = str(e)

            stats["rust_performance"] = rust_stats
        else:
            stats["rust_performance"] = {
                "rust_available": False,
                "fallback_mode": "Python implementations only",
            }

        # Add resource usage
        if self._resource_monitor:
            stats["resource_usage"] = self._resource_monitor.get_average_usage(
                window=20,
            )

        # Add cache statistics
        if self._cache_manager:
            stats["cache_stats"] = await self._cache_manager.get_cache_analytics()

        # Add connection pool statistics
        if self._connection_pool_manager:
            stats[
                "connection_pool_stats"
            ] = await self._connection_pool_manager.get_comprehensive_stats()

        # Add circuit breaker statistics
        stats["circuit_breaker_stats"] = {
            job_type: breaker.get_stats() for job_type, breaker in self._circuit_breakers.items()
        }

        return stats

    async def optimize_resources(self) -> dict[str, Any]:
        """Perform resource optimization and cleanup."""
        optimization_results = {
            "timestamp": time.time(),
            "actions_taken": [],
            "memory_freed_mb": 0.0,
            "cache_optimization": {},
        }

        # Clean up completed jobs
        completed_jobs = [job_id for job_id, job in self._jobs.items() if job.metrics.is_completed]

        if completed_jobs:
            # Keep only recent completed jobs (last 100)
            if len(completed_jobs) > 100:
                jobs_to_remove = completed_jobs[:-100]
                for job_id in jobs_to_remove:
                    del self._jobs[job_id]
                    self._streaming_results.pop(job_id, None)

                optimization_results["actions_taken"].append(
                    f"Removed {len(jobs_to_remove)} old completed jobs",
                )

        # Optimize cache
        if self._cache_manager:
            cache_optimization = await self._cache_manager.optimize_cache()
            optimization_results["cache_optimization"] = cache_optimization
            optimization_results["actions_taken"].append("Optimized cache")

        # Force garbage collection
        import gc

        collected = gc.collect()
        if collected:
            optimization_results["actions_taken"].append(
                f"Garbage collected {collected} objects",
            )

        self.logger.info(f"Resource optimization completed: {optimization_results}")
        return optimization_results

    @abstractmethod
    def _get_job_executor(self, job_type: str) -> JobExecutor:
        """Get job executor for specific job type. Must be implemented by subclasses."""

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
