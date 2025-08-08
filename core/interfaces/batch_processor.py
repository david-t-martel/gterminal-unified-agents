#!/usr/bin/env python3
"""Batch Processing System for CLI Adapter - Phase 2
Supports YAML/JSON script execution with job control and monitoring.
"""

import asyncio
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import validator
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskID
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.table import Table
import yaml

from gterminal.core.react_engine import ReactEngine
from gterminal.core.react_engine import ReactEngineConfig
from gterminal.core.session import SessionManager
from gterminal.utils.rust_extensions.wrapper import RUST_EXTENSIONS_AVAILABLE
from gterminal.utils.rust_extensions.wrapper import parse_json_fast
from gterminal.utils.rust_extensions.wrapper import rust_cache

logger = logging.getLogger(__name__)
console = Console()


class TaskType(str, Enum):
    """Types of batch tasks."""

    REVIEW = "review"
    ANALYZE = "analyze"
    GENERATE = "generate"
    DOCUMENT = "document"
    ORCHESTRATE = "orchestrate"
    CUSTOM = "custom"


class TaskPriority(str, Enum):
    """Task execution priorities."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class BatchTask(BaseModel):
    """Individual batch task definition."""

    id: str
    name: str
    type: TaskType
    description: str | None = None
    priority: TaskPriority = TaskPriority.NORMAL

    # Task execution parameters
    command: str | None = None  # Natural language command
    agent: str | None = None  # Specific agent to use
    parameters: dict[str, Any] = Field(default_factory=dict)

    # Dependencies and conditions
    depends_on: list[str] = Field(default_factory=list)  # Task IDs this depends on
    conditions: dict[str, Any] = Field(default_factory=dict)  # Conditional execution

    # Execution control
    timeout_minutes: int | None = 30
    retry_count: int = 0
    max_retries: int = 2
    continue_on_failure: bool = False

    # Runtime state (not serialized in config)
    status: TaskStatus = TaskStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    result: Any | None = None
    error: str | None = None

    @validator("id")
    def validate_id(self, v):
        if not v or not isinstance(v, str):
            msg = "Task ID must be a non-empty string"
            raise ValueError(msg)
        return v

    @validator("depends_on")
    def validate_dependencies(self, v):
        return list(set(v))  # Remove duplicates

    class Config:
        use_enum_values = True


class BatchScript(BaseModel):
    """Complete batch script definition."""

    name: str
    description: str | None = None
    version: str = "1.0"

    # Execution settings
    profile: str = "default"
    parallel_limit: int = 3  # Max parallel tasks
    fail_fast: bool = False  # Stop on first failure

    # Session management
    session_id: str | None = None
    preserve_session: bool = True

    # Output settings
    output_format: str = "rich"  # rich, json, yaml
    save_results: bool = True
    results_file: str | None = None

    # Task definitions
    tasks: list[BatchTask]

    # Global variables and context
    variables: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)

    @validator("tasks")
    def validate_tasks(self, v):
        if not v:
            msg = "At least one task is required"
            raise ValueError(msg)

        # Validate unique task IDs
        task_ids = [task.id for task in v]
        if len(task_ids) != len(set(task_ids)):
            msg = "Task IDs must be unique"
            raise ValueError(msg)

        # Validate dependencies exist
        for task in v:
            for dep_id in task.depends_on:
                if dep_id not in task_ids:
                    msg = f"Task {task.id} depends on non-existent task {dep_id}"
                    raise ValueError(msg)

        return v

    class Config:
        use_enum_values = True


class BatchExecutionResult(BaseModel):
    """Results from batch execution."""

    script_name: str
    success: bool
    start_time: datetime
    end_time: datetime
    total_duration_seconds: float

    # Task results
    tasks_total: int
    tasks_completed: int
    tasks_failed: int
    tasks_skipped: int

    # Detailed results
    task_results: dict[str, Any]
    execution_log: list[dict[str, Any]]

    # Performance metrics
    parallel_efficiency: float  # How well parallel execution worked
    cache_hit_rate: float | None = None
    rust_acceleration: bool = False


class BatchProcessor:
    """Advanced batch processor with job control and monitoring."""

    def __init__(
        self,
        profile: str = "default",
        config: ReactEngineConfig | None = None,
    ) -> None:
        self.profile = profile
        self.config = config or ReactEngineConfig(
            enable_redis=True,
            enable_rag=True,
            enable_autonomous=True,  # Better for batch processing
            enable_streaming=False,  # We handle our own progress
            cache_responses=True,
            parallel_tool_execution=True,
        )

        # Core components
        self.react_engine = ReactEngine(config=self.config, profile=profile)
        self.session_manager = SessionManager()

        # Runtime state
        self.current_script: BatchScript | None = None
        self.execution_state: dict[str, Any] = {}
        self.running_tasks: dict[str, asyncio.Task] = {}
        self.completed_tasks: dict[str, BatchTask] = {}
        self.failed_tasks: dict[str, BatchTask] = {}

        # Progress tracking
        self.progress: Progress | None = None
        self.task_progress: dict[str, TaskID] = {}

        # Cancellation support
        self.cancelled = False
        self.pause_requested = False

    async def load_script(self, script_path: str | Path) -> BatchScript:
        """Load batch script from file."""
        script_path = Path(script_path)

        if not script_path.exists():
            msg = f"Script file not found: {script_path}"
            raise FileNotFoundError(msg)

        try:
            with open(script_path) as f:
                content = f.read()

            # Determine format and parse
            if script_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(content)
            elif script_path.suffix.lower() == ".json":
                if RUST_EXTENSIONS_AVAILABLE:
                    data = await parse_json_fast(content)
                else:
                    data = json.loads(content)
            else:
                msg = f"Unsupported script format: {script_path.suffix}"
                raise ValueError(msg)

            # Create and validate script
            script = BatchScript(**data)

            # Set default results file if not specified
            if script.save_results and not script.results_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                script.results_file = f"batch_results_{script.name}_{timestamp}.json"

            self.current_script = script
            return script

        except Exception as e:
            msg = f"Failed to load script {script_path}: {e}"
            raise ValueError(msg)

    async def execute_script(
        self, script: BatchScript | None = None, live_output: bool = True
    ) -> BatchExecutionResult:
        """Execute a batch script with live monitoring."""
        if script:
            self.current_script = script
        elif not self.current_script:
            msg = "No script loaded for execution"
            raise ValueError(msg)

        script = self.current_script
        start_time = datetime.now()

        # Initialize components
        await self._initialize_execution()

        # Setup progress tracking
        if live_output:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            )

        execution_log: list[dict[str, Any]] = []

        try:
            with (
                Live(self._create_dashboard(), console=console, refresh_per_second=2)
                if live_output
                else nullcontext()
            ):
                # Log script start
                execution_log.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "event": "script_started",
                        "script": script.name,
                        "tasks_count": len(script.tasks),
                    },
                )

                # Execute tasks
                await self._execute_tasks_with_dependencies(script, execution_log)

        except KeyboardInterrupt:
            self.cancelled = True
            console.print("[red]❌ Execution cancelled by user[/red]")

        except Exception as e:
            logger.exception(f"Batch execution error: {e}")
            execution_log.append(
                {"timestamp": datetime.now().isoformat(), "event": "error", "message": str(e)}
            )

        finally:
            # Cleanup running tasks
            await self._cleanup_tasks()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Calculate results
        result = await self._create_execution_result(
            script, start_time, end_time, duration, execution_log
        )

        # Save results if requested
        if script.save_results and script.results_file:
            await self._save_results(result, script.results_file)

        return result

    async def _initialize_execution(self) -> None:
        """Initialize execution environment."""
        # Initialize React engine components
        if self.config.enable_redis:
            await self.react_engine.initialize_redis()
        if self.config.enable_rag:
            await self.react_engine.initialize_rag()

        # Reset state
        self.cancelled = False
        self.pause_requested = False
        self.running_tasks.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        self.task_progress.clear()

    async def _execute_tasks_with_dependencies(
        self, script: BatchScript, execution_log: list[dict[str, Any]]
    ) -> None:
        """Execute tasks respecting dependencies and parallel limits."""
        # Build dependency graph
        self._build_dependency_graph(script.tasks)

        # Get tasks that can start immediately (no dependencies)
        ready_tasks = [task for task in script.tasks if not task.depends_on]
        completed_task_ids = set()

        while ready_tasks or self.running_tasks:
            # Start new tasks up to parallel limit
            while (
                ready_tasks
                and len(self.running_tasks) < script.parallel_limit
                and not self.cancelled
            ):
                task = ready_tasks.pop(0)
                await self._start_task(task, script, execution_log)

            # Wait for at least one task to complete
            if self.running_tasks:
                done_tasks = await self._wait_for_task_completion()

                for task_id in done_tasks:
                    completed_task_ids.add(task_id)

                    # Find newly ready tasks
                    newly_ready = self._find_newly_ready_tasks(
                        script.tasks, completed_task_ids, ready_tasks
                    )
                    ready_tasks.extend(newly_ready)

                    # Check fail_fast condition
                    if script.fail_fast and task_id in self.failed_tasks:
                        console.print(
                            "[red]❌ Stopping execution due to task failure (fail_fast=true)[/red]"
                        )
                        self.cancelled = True
                        break

            # Handle pause requests
            if self.pause_requested:
                await self._handle_pause()

    def _build_dependency_graph(self, tasks: list[BatchTask]) -> dict[str, list[str]]:
        """Build task dependency graph."""
        graph = {}
        for task in tasks:
            graph[task.id] = task.depends_on.copy()
        return graph

    def _find_newly_ready_tasks(
        self,
        all_tasks: list[BatchTask],
        completed_task_ids: set,
        current_ready: list[BatchTask],
    ) -> list[BatchTask]:
        """Find tasks that are now ready to execute."""
        current_ready_ids = {task.id for task in current_ready}
        newly_ready = []

        for task in all_tasks:
            if (
                task.id not in completed_task_ids
                and task.id not in self.running_tasks
                and task.id not in current_ready_ids
                and task.id not in self.failed_tasks
                and all(dep_id in completed_task_ids for dep_id in task.depends_on)
            ):
                newly_ready.append(task)

        return newly_ready

    async def _start_task(
        self, task: BatchTask, script: BatchScript, execution_log: list[dict[str, Any]]
    ) -> None:
        """Start executing a single task."""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()

        # Create progress tracking
        if self.progress:
            progress_id = self.progress.add_task(f"[cyan]{task.name}[/cyan]", total=100, start=True)
            self.task_progress[task.id] = progress_id

        # Log task start
        execution_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": "task_started",
                "task_id": task.id,
                "task_name": task.name,
                "priority": task.priority.value,
            },
        )

        # Create async task for execution
        async_task = asyncio.create_task(self._execute_single_task(task, script, execution_log))
        self.running_tasks[task.id] = async_task

    async def _execute_single_task(
        self, task: BatchTask, script: BatchScript, execution_log: list[dict[str, Any]]
    ):
        """Execute a single task with retry logic."""
        for attempt in range(task.max_retries + 1):
            try:
                if attempt > 0:
                    execution_log.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "event": "task_retry",
                            "task_id": task.id,
                            "attempt": attempt + 1,
                        },
                    )

                # Update progress
                if task.id in self.task_progress and self.progress:
                    self.progress.update(
                        self.task_progress[task.id],
                        advance=20,
                        description=f"[cyan]{task.name}[/cyan] (Attempt {attempt + 1})",
                    )

                # Execute the actual task
                result = await self._execute_task_command(task, script)

                # Task completed successfully
                task.status = TaskStatus.COMPLETED
                task.end_time = datetime.now()
                task.result = result

                self.completed_tasks[task.id] = task

                # Update progress to completion
                if task.id in self.task_progress and self.progress:
                    self.progress.update(
                        self.task_progress[task.id],
                        completed=100,
                        description=f"[green]✅ {task.name}[/green]",
                    )

                execution_log.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "event": "task_completed",
                        "task_id": task.id,
                        "duration_seconds": (task.end_time - task.start_time).total_seconds(),
                    },
                )

                return result

            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                raise

            except Exception as e:
                task.retry_count += 1
                error_msg = str(e)

                execution_log.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "event": "task_error",
                        "task_id": task.id,
                        "attempt": attempt + 1,
                        "error": error_msg,
                    },
                )

                if attempt >= task.max_retries:
                    # Task failed after all retries
                    task.status = TaskStatus.FAILED
                    task.end_time = datetime.now()
                    task.error = error_msg

                    self.failed_tasks[task.id] = task

                    # Update progress to show failure
                    if task.id in self.task_progress and self.progress:
                        self.progress.update(
                            self.task_progress[task.id], description=f"[red]❌ {task.name}[/red]"
                        )

                    execution_log.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "event": "task_failed",
                            "task_id": task.id,
                            "final_error": error_msg,
                        },
                    )

                    if not task.continue_on_failure:
                        raise

                else:
                    # Wait before retry
                    await asyncio.sleep(min(2**attempt, 30))  # Exponential backoff
        return None

    async def _execute_task_command(self, task: BatchTask, script: BatchScript) -> Any:
        """Execute the actual task command."""
        # Substitute variables in command
        command = self._substitute_variables(task.command or task.name, script.variables)

        # Get or create session
        session = await self.session_manager.get_or_create_async(
            script.session_id or "batch_session"
        )

        # Add task context to session
        session.update_context(
            f"task_{task.id}",
            {
                "task_id": task.id,
                "task_name": task.name,
                "task_type": task.type.value,
                "parameters": task.parameters,
            },
        )

        # Execute through React engine
        response = await self.react_engine.process_request(
            request=command, session_id=session.id, streaming=False
        )

        if not response.success:
            msg = f"Task execution failed: {response.result.get('error', 'Unknown error')}"
            raise Exception(msg)

        return response.result

    def _substitute_variables(self, text: str, variables: dict[str, Any]) -> str:
        """Substitute variables in text using ${variable} syntax."""
        import re

        def replacer(match):
            var_name = match.group(1)
            return str(variables.get(var_name, match.group(0)))

        return re.sub(r"\$\{(\w+)\}", replacer, text)

    async def _wait_for_task_completion(self) -> list[str]:
        """Wait for at least one running task to complete."""
        if not self.running_tasks:
            return []

        done, pending = await asyncio.wait(
            self.running_tasks.values(), return_when=asyncio.FIRST_COMPLETED
        )

        completed_task_ids = []

        for task in done:
            # Find which task ID this corresponds to
            for task_id, async_task in self.running_tasks.items():
                if async_task == task:
                    completed_task_ids.append(task_id)
                    del self.running_tasks[task_id]
                    break

        return completed_task_ids

    async def _cleanup_tasks(self) -> None:
        """Cancel all running tasks."""
        for async_task in self.running_tasks.values():
            if not async_task.done():
                async_task.cancel()

        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)

        self.running_tasks.clear()

    async def _handle_pause(self) -> None:
        """Handle pause request."""
        console.print("[yellow]⏸️  Execution paused. Press Enter to continue...[/yellow]")
        input()  # Simple pause - in production, this would be more sophisticated
        self.pause_requested = False

    def _create_dashboard(self) -> Panel:
        """Create live dashboard for batch execution."""
        if not self.current_script:
            return Panel("No script loaded")

        # Create status table
        status_table = Table(title="Batch Execution Status")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="white")

        total_tasks = len(self.current_script.tasks)
        completed = len(self.completed_tasks)
        running = len(self.running_tasks)
        failed = len(self.failed_tasks)

        status_table.add_row("Script", self.current_script.name)
        status_table.add_row("Total Tasks", str(total_tasks))
        status_table.add_row("Completed", f"[green]{completed}[/green]")
        status_table.add_row("Running", f"[yellow]{running}[/yellow]")
        status_table.add_row("Failed", f"[red]{failed}[/red]")
        status_table.add_row("Pending", str(total_tasks - completed - running - failed))

        # Add rust status
        if RUST_EXTENSIONS_AVAILABLE:
            cache_stats = rust_cache.get_stats()
            status_table.add_row("Cache Hit Rate", f"{cache_stats.get('hit_rate', 0):.1%}")

        # Combine with progress if available
        if self.progress:
            return Panel.fit(
                status_table, title="📊 Batch Execution Dashboard", border_style="blue"
            )
        return Panel.fit(status_table, title="📊 Batch Execution Dashboard", border_style="blue")

    async def _create_execution_result(
        self,
        script: BatchScript,
        start_time: datetime,
        end_time: datetime,
        duration: float,
        execution_log: list[dict[str, Any]],
    ) -> BatchExecutionResult:
        """Create final execution result."""
        total_tasks = len(script.tasks)
        completed_count = len(self.completed_tasks)
        failed_count = len(self.failed_tasks)
        skipped_count = total_tasks - completed_count - failed_count

        # Calculate parallel efficiency
        if duration > 0:
            serial_time = sum(
                (task.end_time - task.start_time).total_seconds()
                for task in self.completed_tasks.values()
                if task.end_time and task.start_time
            )
            parallel_efficiency = min(serial_time / duration, 1.0) if duration > 0 else 0
        else:
            parallel_efficiency = 0

        # Collect task results
        task_results = {}
        for task_id, task in {**self.completed_tasks, **self.failed_tasks}.items():
            task_results[task_id] = {
                "name": task.name,
                "status": task.status.value,
                "start_time": task.start_time.isoformat() if task.start_time else None,
                "end_time": task.end_time.isoformat() if task.end_time else None,
                "duration_seconds": (
                    (task.end_time - task.start_time).total_seconds()
                    if task.end_time and task.start_time
                    else 0
                ),
                "result": task.result,
                "error": task.error,
                "retry_count": task.retry_count,
            }

        # Get cache hit rate if available
        cache_hit_rate = None
        if RUST_EXTENSIONS_AVAILABLE:
            cache_stats = rust_cache.get_stats()
            cache_hit_rate = cache_stats.get("hit_rate")

        return BatchExecutionResult(
            script_name=script.name,
            success=failed_count == 0 and not self.cancelled,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=duration,
            tasks_total=total_tasks,
            tasks_completed=completed_count,
            tasks_failed=failed_count,
            tasks_skipped=skipped_count,
            task_results=task_results,
            execution_log=execution_log,
            parallel_efficiency=parallel_efficiency,
            cache_hit_rate=cache_hit_rate,
            rust_acceleration=RUST_EXTENSIONS_AVAILABLE,
        )

    async def _save_results(self, result: BatchExecutionResult, filename: str) -> None:
        """Save execution results to file."""
        try:
            results_data = result.model_dump()

            with open(filename, "w") as f:
                json.dump(results_data, f, indent=2, default=str)

            console.print(f"[success]✅ Results saved to: {filename}[/success]")

        except Exception as e:
            console.print(f"[error]❌ Failed to save results: {e}[/error]")

    # Job control methods
    def pause_execution(self) -> None:
        """Request pause of current execution."""
        self.pause_requested = True

    def cancel_execution(self) -> None:
        """Cancel current execution."""
        self.cancelled = True

    def get_execution_status(self) -> dict[str, Any]:
        """Get current execution status."""
        if not self.current_script:
            return {"status": "no_script_loaded"}

        return {
            "script_name": self.current_script.name,
            "total_tasks": len(self.current_script.tasks),
            "completed_tasks": len(self.completed_tasks),
            "running_tasks": len(self.running_tasks),
            "failed_tasks": len(self.failed_tasks),
            "cancelled": self.cancelled,
            "paused": self.pause_requested,
        }


# Utility class for null context manager
class nullcontext:
    """Null context manager for Python < 3.7 compatibility."""

    def __enter__(self):
        return None

    def __exit__(self, *excinfo):
        return None


# Example script creation utilities
def create_sample_script() -> BatchScript:
    """Create a sample batch script for demonstration."""
    return BatchScript(
        name="sample_analysis",
        description="Sample batch script for code analysis",
        profile="default",
        parallel_limit=2,
        tasks=[
            BatchTask(
                id="analyze_main",
                name="Analyze Main Module",
                type=TaskType.ANALYZE,
                command="Analyze the main.py file for code quality and performance issues",
                parameters={"file": "main.py", "depth": "comprehensive"},
            ),
            BatchTask(
                id="review_tests",
                name="Review Test Coverage",
                type=TaskType.REVIEW,
                command="Review test coverage and identify missing tests",
                depends_on=["analyze_main"],
                parameters={"test_dir": "tests/"},
            ),
            BatchTask(
                id="generate_docs",
                name="Generate Documentation",
                type=TaskType.GENERATE,
                command="Generate API documentation for the analyzed modules",
                depends_on=["analyze_main"],
                parameters={"output": "docs/api.md"},
            ),
        ],
        variables={"project_name": "my_project", "version": "1.0.0"},
    )


if __name__ == "__main__":
    # Test the batch processor
    async def test_batch() -> None:
        processor = BatchProcessor()
        script = create_sample_script()
        result = await processor.execute_script(script)
        console.print(f"Execution completed: {result.success}")

    asyncio.run(test_batch())
