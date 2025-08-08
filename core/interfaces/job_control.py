#!/usr/bin/env python3
"""Job Control System for CLI Adapter - Phase 2
Provides pause/resume/cancel/update capabilities for long-running operations.
"""

import asyncio
from contextlib import asynccontextmanager
from contextlib import suppress
from datetime import datetime
from datetime import timedelta
from enum import Enum
import logging
import signal
import threading
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field
from rich.console import Console

from gterminal.utils.rust_extensions.wrapper import RUST_EXTENSIONS_AVAILABLE

logger = logging.getLogger(__name__)
console = Console()


class JobState(str, Enum):
    """Job execution states."""

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UPDATING = "updating"


class JobPriority(str, Enum):
    """Job priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class JobType(str, Enum):
    """Types of jobs that can be managed."""

    BATCH_SCRIPT = "batch_script"
    INTERACTIVE_TASK = "interactive_task"
    BACKGROUND_PROCESS = "background_process"
    FILE_OPERATION = "file_operation"
    AGENT_ORCHESTRATION = "agent_orchestration"


class JobControlSignal(str, Enum):
    """Signals that can be sent to jobs."""

    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    UPDATE = "update"
    STATUS = "status"


class JobMetadata(BaseModel):
    """Metadata for a managed job."""

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    job_type: JobType
    description: str | None = None

    # State tracking
    state: JobState = JobState.CREATED
    priority: JobPriority = JobPriority.NORMAL

    # Timing information
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    paused_at: datetime | None = None
    resumed_at: datetime | None = None
    completed_at: datetime | None = None

    # Progress tracking
    total_steps: int = 0
    completed_steps: int = 0
    current_step_name: str | None = None
    progress_percentage: float = 0.0

    # Resource information
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    estimated_completion: datetime | None = None

    # Control capabilities
    can_pause: bool = True
    can_resume: bool = False
    can_cancel: bool = True
    can_update: bool = False

    # Error information
    last_error: str | None = None
    error_count: int = 0

    # User context
    user_id: str | None = None
    session_id: str | None = None

    # Custom data
    context: dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class JobUpdate(BaseModel):
    """Update instruction for a running job."""

    job_id: str
    update_type: str  # "parameters", "instruction", "priority", etc.
    update_data: dict[str, Any]
    message: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class JobControlEvent(BaseModel):
    """Event logged during job control operations."""

    job_id: str
    event_type: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: dict[str, Any] = Field(default_factory=dict)


class ManagedJob:
    """A job that can be controlled (paused, resumed, cancelled, updated)."""

    def __init__(self, metadata: JobMetadata, coroutine_func, *args, **kwargs) -> None:
        self.metadata = metadata
        self.coroutine_func = coroutine_func
        self.args = args
        self.kwargs = kwargs

        # Control mechanisms
        self._task: asyncio.Task | None = None
        self._pause_event = asyncio.Event()
        self._cancel_event = asyncio.Event()
        self._update_queue: asyncio.Queue[JobUpdate] = asyncio.Queue()

        # Set initial pause state (not paused)
        self._pause_event.set()

        # Progress tracking
        self._progress_lock = asyncio.Lock()
        self._last_heartbeat = datetime.now()

        # Event log
        self.events: list[JobControlEvent] = []

    async def start(self) -> None:
        """Start the job execution."""
        if self.metadata.state != JobState.CREATED:
            msg = f"Job {self.metadata.job_id} cannot be started from state {self.metadata.state}"
            raise ValueError(msg)

        self.metadata.state = JobState.RUNNING
        self.metadata.started_at = datetime.now()
        self.metadata.can_resume = False

        # Log event
        await self._log_event("job_started", "Job execution started")

        # Create and start the async task
        self._task = asyncio.create_task(self._run_with_control())

    async def pause(self) -> bool:
        """Pause the job execution."""
        if not self.metadata.can_pause or self.metadata.state != JobState.RUNNING:
            return False

        self._pause_event.clear()
        self.metadata.state = JobState.PAUSED
        self.metadata.paused_at = datetime.now()
        self.metadata.can_resume = True

        await self._log_event("job_paused", "Job execution paused")
        return True

    async def resume(self) -> bool:
        """Resume the job execution."""
        if not self.metadata.can_resume or self.metadata.state != JobState.PAUSED:
            return False

        self._pause_event.set()
        self.metadata.state = JobState.RUNNING
        self.metadata.resumed_at = datetime.now()
        self.metadata.can_resume = False

        await self._log_event("job_resumed", "Job execution resumed")
        return True

    async def cancel(self) -> bool:
        """Cancel the job execution."""
        if not self.metadata.can_cancel:
            return False

        self._cancel_event.set()

        if self._task and not self._task.done():
            self._task.cancel()

        self.metadata.state = JobState.CANCELLED
        self.metadata.completed_at = datetime.now()

        await self._log_event("job_cancelled", "Job execution cancelled")
        return True

    async def update(self, update: JobUpdate) -> bool:
        """Send an update instruction to the job."""
        if not self.metadata.can_update:
            return False

        await self._update_queue.put(update)
        await self._log_event(
            "job_updated", f"Job update: {update.update_type}", update.update_data
        )
        return True

    async def wait_for_completion(self) -> Any:
        """Wait for the job to complete and return the result."""
        if not self._task:
            msg = "Job has not been started"
            raise ValueError(msg)

        try:
            return await self._task
        except asyncio.CancelledError:
            self.metadata.state = JobState.CANCELLED
            self.metadata.completed_at = datetime.now()
            raise
        except Exception as e:
            self.metadata.state = JobState.FAILED
            self.metadata.last_error = str(e)
            self.metadata.error_count += 1
            self.metadata.completed_at = datetime.now()
            raise

    async def get_status(self) -> dict[str, Any]:
        """Get current job status."""
        # Update timing calculations
        now = datetime.now()
        runtime = None

        if self.metadata.started_at:
            if self.metadata.completed_at:
                runtime = (self.metadata.completed_at - self.metadata.started_at).total_seconds()
            else:
                runtime = (now - self.metadata.started_at).total_seconds()

        return {
            "job_id": self.metadata.job_id,
            "name": self.metadata.name,
            "state": self.metadata.state.value,
            "priority": self.metadata.priority.value,
            "progress_percentage": self.metadata.progress_percentage,
            "current_step": self.metadata.current_step_name,
            "completed_steps": self.metadata.completed_steps,
            "total_steps": self.metadata.total_steps,
            "runtime_seconds": runtime,
            "can_pause": self.metadata.can_pause,
            "can_resume": self.metadata.can_resume,
            "can_cancel": self.metadata.can_cancel,
            "can_update": self.metadata.can_update,
            "last_error": self.metadata.last_error,
            "error_count": self.metadata.error_count,
            "cpu_usage": self.metadata.cpu_usage,
            "memory_usage": self.metadata.memory_usage,
        }

    async def update_progress(
        self,
        completed_steps: int | None = None,
        total_steps: int | None = None,
        current_step_name: str | None = None,
        percentage: float | None = None,
    ) -> None:
        """Update job progress information."""
        async with self._progress_lock:
            if completed_steps is not None:
                self.metadata.completed_steps = completed_steps
            if total_steps is not None:
                self.metadata.total_steps = total_steps
            if current_step_name is not None:
                self.metadata.current_step_name = current_step_name

            # Calculate percentage if not provided
            if percentage is not None:
                self.metadata.progress_percentage = percentage
            elif self.metadata.total_steps > 0:
                self.metadata.progress_percentage = (
                    self.metadata.completed_steps / self.metadata.total_steps * 100
                )

            # Update estimated completion
            if self.metadata.progress_percentage > 0 and self.metadata.started_at:
                elapsed = datetime.now() - self.metadata.started_at
                estimated_total = elapsed / (self.metadata.progress_percentage / 100)
                self.metadata.estimated_completion = self.metadata.started_at + estimated_total

            self._last_heartbeat = datetime.now()

    async def _run_with_control(self) -> Any:
        """Run the job with control mechanisms."""
        try:
            # Wrap the coroutine to inject control points
            if asyncio.iscoroutinefunction(self.coroutine_func):
                result = await self._run_controlled_coroutine()
            else:
                # Handle sync functions by running in thread
                result = await asyncio.to_thread(self.coroutine_func, *self.args, **self.kwargs)

            # Job completed successfully
            self.metadata.state = JobState.COMPLETED
            self.metadata.completed_at = datetime.now()
            await self._log_event("job_completed", "Job completed successfully")

            return result

        except asyncio.CancelledError:
            self.metadata.state = JobState.CANCELLED
            self.metadata.completed_at = datetime.now()
            await self._log_event("job_cancelled", "Job was cancelled")
            raise

        except Exception as e:
            self.metadata.state = JobState.FAILED
            self.metadata.last_error = str(e)
            self.metadata.error_count += 1
            self.metadata.completed_at = datetime.now()
            await self._log_event("job_failed", f"Job failed: {e!s}")
            raise

    async def _run_controlled_coroutine(self) -> Any:
        """Run coroutine with control checkpoints."""
        # Create a controlled execution context
        context = JobExecutionContext(self)

        # Inject the context into the coroutine kwargs
        enhanced_kwargs = {**self.kwargs, "_job_context": context}

        # Execute the coroutine
        return await self.coroutine_func(*self.args, **enhanced_kwargs)

    async def _log_event(
        self, event_type: str, message: str, data: dict[str, Any] | None = None
    ) -> None:
        """Log a job control event."""
        event = JobControlEvent(
            job_id=self.metadata.job_id, event_type=event_type, message=message, data=data or {}
        )
        self.events.append(event)
        logger.info(f"Job {self.metadata.job_id}: {message}")


class JobExecutionContext:
    """Context object passed to controlled job functions."""

    def __init__(self, job: ManagedJob) -> None:
        self.job = job
        self._checkpoint_count = 0

    async def checkpoint(self, step_name: str | None = None) -> None:
        """Control checkpoint - handles pause/resume/cancel/update."""
        self._checkpoint_count += 1

        # Check for cancellation
        if self.job._cancel_event.is_set():
            msg = "Job was cancelled"
            raise asyncio.CancelledError(msg)

        # Handle pause
        await self.job._pause_event.wait()

        # Process any pending updates
        await self._process_updates()

        # Update progress if step name provided
        if step_name:
            await self.job.update_progress(
                completed_steps=self._checkpoint_count, current_step_name=step_name
            )

    async def update_progress(
        self,
        completed: int | None = None,
        total: int | None = None,
        current_step: str | None = None,
        percentage: float | None = None,
    ) -> None:
        """Update job progress."""
        await self.job.update_progress(completed, total, current_step, percentage)

    async def _process_updates(self) -> None:
        """Process any pending job updates."""
        try:
            while True:
                update = self.job._update_queue.get_nowait()
                await self._apply_update(update)
        except asyncio.QueueEmpty:
            pass

    async def _apply_update(self, update: JobUpdate) -> None:
        """Apply a job update instruction."""
        if update.update_type == "priority":
            new_priority = JobPriority(update.update_data.get("priority", "normal"))
            self.job.metadata.priority = new_priority

        elif update.update_type == "parameters":
            # Update job parameters (implementation depends on job type)
            self.job.kwargs.update(update.update_data)

        elif update.update_type == "instruction":
            # Handle instruction updates (implementation specific)
            pass

        logger.info(f"Applied update {update.update_type} to job {self.job.metadata.job_id}")


class JobManager:
    """Central manager for all controlled jobs."""

    def __init__(self, max_concurrent_jobs: int = 10) -> None:
        self.max_concurrent_jobs = max_concurrent_jobs
        self.jobs: dict[str, ManagedJob] = {}
        self.job_queue: asyncio.Queue[str] = asyncio.Queue()

        # Background task for job monitoring
        self._monitor_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        # Statistics
        self.stats = {
            "jobs_created": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "jobs_cancelled": 0,
        }

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

    async def start(self) -> None:
        """Start the job manager."""
        self._monitor_task = asyncio.create_task(self._monitor_jobs())
        logger.info("Job Manager started")

    async def shutdown(self) -> None:
        """Shutdown the job manager gracefully."""
        self._shutdown_event.set()

        # Cancel all running jobs
        for job in self.jobs.values():
            if job.metadata.state == JobState.RUNNING:
                await job.cancel()

        # Wait for monitor task to finish
        if self._monitor_task:
            self._monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor_task

        logger.info("Job Manager shutdown complete")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown())

    async def create_job(
        self,
        name: str,
        job_type: JobType,
        coroutine_func,
        *args,
        description: str | None = None,
        priority: JobPriority = JobPriority.NORMAL,
        can_pause: bool = True,
        can_update: bool = False,
        user_id: str | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> str:
        """Create a new managed job."""
        metadata = JobMetadata(
            name=name,
            job_type=job_type,
            description=description,
            priority=priority,
            can_pause=can_pause,
            can_update=can_update,
            user_id=user_id,
            session_id=session_id,
        )

        job = ManagedJob(metadata, coroutine_func, *args, **kwargs)
        self.jobs[metadata.job_id] = job
        self.stats["jobs_created"] += 1

        # Queue job for execution
        await self.job_queue.put(metadata.job_id)

        logger.info(f"Created job {metadata.job_id}: {name}")
        return metadata.job_id

    async def get_job(self, job_id: str) -> ManagedJob | None:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    async def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        job = self.jobs.get(job_id)
        if job:
            return await job.pause()
        return False

    async def resume_job(self, job_id: str) -> bool:
        """Resume a job."""
        job = self.jobs.get(job_id)
        if job:
            return await job.resume()
        return False

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.jobs.get(job_id)
        if job:
            success = await job.cancel()
            if success:
                self.stats["jobs_cancelled"] += 1
            return success
        return False

    async def update_job(self, job_id: str, update: JobUpdate) -> bool:
        """Send an update to a job."""
        job = self.jobs.get(job_id)
        if job:
            return await job.update(update)
        return False

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of a specific job."""
        job = self.jobs.get(job_id)
        if job:
            return await job.get_status()
        return None

    async def list_jobs(
        self,
        state_filter: JobState | None = None,
        user_id_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all jobs with optional filtering."""
        jobs_status = []

        for job in self.jobs.values():
            if state_filter and job.metadata.state != state_filter:
                continue
            if user_id_filter and job.metadata.user_id != user_id_filter:
                continue

            status = await job.get_status()
            jobs_status.append(status)

        # Sort by created time (most recent first)
        jobs_status.sort(key=lambda x: self.jobs[x["job_id"]].metadata.created_at, reverse=True)

        return jobs_status

    async def get_manager_status(self) -> dict[str, Any]:
        """Get overall job manager status."""
        job_counts = {}
        for state in JobState:
            job_counts[state.value] = sum(
                1 for job in self.jobs.values() if job.metadata.state == state
            )

        return {
            "total_jobs": len(self.jobs),
            "job_counts_by_state": job_counts,
            "max_concurrent": self.max_concurrent_jobs,
            "queue_size": self.job_queue.qsize(),
            "statistics": self.stats.copy(),
            "rust_acceleration": RUST_EXTENSIONS_AVAILABLE,
        }

    async def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed/failed/cancelled jobs."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []

        for job_id, job in self.jobs.items():
            if (
                job.metadata.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]
                and job.metadata.completed_at
                and job.metadata.completed_at < cutoff_time
            ):
                to_remove.append(job_id)

        for job_id in to_remove:
            del self.jobs[job_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")

        return len(to_remove)

    async def _monitor_jobs(self) -> None:
        """Background task to monitor and manage jobs."""
        while not self._shutdown_event.is_set():
            try:
                # Process job queue
                running_count = sum(
                    1 for job in self.jobs.values() if job.metadata.state == JobState.RUNNING
                )

                # Start jobs from queue if we have capacity
                while running_count < self.max_concurrent_jobs:
                    try:
                        job_id = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)

                        job = self.jobs.get(job_id)
                        if job and job.metadata.state == JobState.CREATED:
                            await job.start()
                            running_count += 1

                    except TimeoutError:
                        break

                # Update job statistics
                for job in self.jobs.values():
                    if job.metadata.state == JobState.COMPLETED:
                        self.stats["jobs_completed"] += 1
                    elif job.metadata.state == JobState.FAILED:
                        self.stats["jobs_failed"] += 1

                # Cleanup old jobs periodically
                if len(self.jobs) > 100:  # Cleanup when we have many jobs
                    await self.cleanup_completed_jobs()

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.exception(f"Job monitor error: {e}")
                await asyncio.sleep(5)  # Wait longer on error


# Global job manager instance
_job_manager: JobManager | None = None


async def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
        await _job_manager.start()
    return _job_manager


# Decorator for making functions job-controllable
def controllable_job(
    name: str | None = None,
    job_type: JobType = JobType.AGENT_ORCHESTRATION,
    can_pause: bool = True,
    can_update: bool = False,
    priority: JobPriority = JobPriority.NORMAL,
):
    """Decorator to make a function controllable as a job."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            job_manager = await get_job_manager()

            job_name = name or func.__name__
            job_id = await job_manager.create_job(
                name=job_name,
                job_type=job_type,
                coroutine_func=func,
                can_pause=can_pause,
                can_update=can_update,
                priority=priority,
                *args,
                **kwargs,
            )

            job = await job_manager.get_job(job_id)
            if job:
                return await job.wait_for_completion()
            msg = f"Failed to create job {job_name}"
            raise RuntimeError(msg)

        return wrapper

    return decorator


# Context manager for job control in CLI
@asynccontextmanager
async def job_controlled_execution(
    name: str,
    job_type: JobType = JobType.INTERACTIVE_TASK,
    can_pause: bool = True,
    can_update: bool = False,
):
    """Context manager for job-controlled execution."""
    await get_job_manager()

    # This is a placeholder - in practice, this would wrap the execution
    # with job control capabilities
    yield


if __name__ == "__main__":
    # Test the job control system
    @controllable_job("test_job", JobType.BACKGROUND_PROCESS)
    async def test_long_running_job(_job_context: JobExecutionContext) -> str:
        """Test job with control checkpoints."""
        for i in range(10):
            await _job_context.checkpoint(f"Step {i + 1}")
            await _job_context.update_progress(completed=i + 1, total=10)
            await asyncio.sleep(1)  # Simulate work
        return "Job completed successfully"

    async def test_job_control() -> None:
        """Test the job control system."""
        console.print("Testing job control system...")

        # This would run the test job
        result = await test_long_running_job()
        console.print(f"Job result: {result}")

    asyncio.run(test_job_control())
