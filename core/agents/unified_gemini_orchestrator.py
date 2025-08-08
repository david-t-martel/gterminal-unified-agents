"""Unified Gemini Orchestrator - Central orchestration and coordination agent.

CONSOLIDATES AND ELIMINATES:
1. app/agents/gemini_orchestrator.py (Multiple orchestrator implementations)
2. app/agents/gemini_server_agent.py (Server-specific orchestration)
3. app/agents/gemini_consolidator_agent.py (Consolidation functionality)
4. app/agents/master_architect_agent.py (Architecture orchestration)
5. app/automation/workflow_orchestrator.py (Workflow management)

UNIFIED FEATURES:
- Central orchestration of all agent activities
- Intelligent task routing and load balancing
- Multi-agent workflow coordination
- Resource management and optimization
- Error handling and recovery strategies
- Performance monitoring and optimization
- Gemini 2.0 Flash integration for decision making
- Real-time status tracking and reporting
"""

import asyncio
import contextlib
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from enum import Enum
import logging
from typing import Any
import uuid

import vertexai
from vertexai.generative_models import GenerativeModel

from gterminal.core.agents.base_unified_agent import BaseUnifiedAgent
from gterminal.core.agents.base_unified_agent import JobStatus
from gterminal.core.agents.base_unified_agent import UnifiedJob
from gterminal.core.agents.unified_code_reviewer import UnifiedCodeReviewer
from gterminal.core.agents.unified_documentation_generator import UnifiedDocumentationGenerator
from gterminal.core.agents.unified_workspace_analyzer import UnifiedWorkspaceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class WorkflowStatus(Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Individual task in a workflow."""

    task_id: str
    agent_type: str
    job_type: str
    parameters: dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: list[str] = field(default_factory=list)
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class Workflow:
    """Multi-task workflow definition."""

    workflow_id: str
    name: str
    description: str
    tasks: list[Task] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None


class UnifiedGeminiOrchestrator(BaseUnifiedAgent):
    """Unified Gemini Orchestrator for coordinating all agent activities.

    Provides centralized orchestration including:
    - Task routing and agent selection
    - Multi-agent workflow coordination
    - Resource management and load balancing
    - Error handling and recovery
    - Performance optimization
    - Real-time monitoring and reporting
    - Intelligent decision making with Gemini AI
    """

    def __init__(self) -> None:
        super().__init__(
            "unified_gemini_orchestrator",
            "Central orchestration and coordination of all agent activities",
        )

        # Initialize Vertex AI
        vertexai.init()
        self.model = GenerativeModel("gemini-2.0-flash-exp")

        # Initialize specialized agents
        self.agents = {
            "code_reviewer": UnifiedCodeReviewer(),
            "workspace_analyzer": UnifiedWorkspaceAnalyzer(),
            "documentation_generator": UnifiedDocumentationGenerator(),
        }

        # Workflow management
        self._workflows: dict[str, Workflow] = {}
        self._workflow_lock = asyncio.Lock()

        # Task queues by priority
        self._task_queues = {
            TaskPriority.CRITICAL: asyncio.Queue(),
            TaskPriority.HIGH: asyncio.Queue(),
            TaskPriority.MEDIUM: asyncio.Queue(),
            TaskPriority.LOW: asyncio.Queue(),
        }

        # Resource management
        self._agent_load = dict.fromkeys(self.agents.keys(), 0)
        self._max_concurrent_tasks = 10
        self._running_tasks = set()

        # Performance tracking
        self._orchestration_stats = {
            "workflows_executed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_workflow_time": 0.0,
            "agent_utilization": {},
            "error_rate": 0.0,
        }

        # Start task processor
        self._task_processor = None

        logger.info("Initialized Unified Gemini Orchestrator with all specialized agents")

    def get_supported_job_types(self) -> list[str]:
        """Return supported orchestration job types."""
        return [
            "execute_workflow",
            "create_workflow",
            "route_task",
            "analyze_performance",
            "optimize_resources",
            "get_agent_status",
            "emergency_recovery",
        ]

    async def execute_job(self, job: UnifiedJob) -> Any:
        """Execute orchestration job with progress tracking."""
        job_type = job.job_type
        parameters = job.parameters

        job.update_progress(5.0, f"Starting {job_type}")

        try:
            if job_type == "execute_workflow":
                return await self._execute_workflow(job, parameters["workflow_id"])
            if job_type == "create_workflow":
                return await self._create_workflow(job, parameters)
            if job_type == "route_task":
                return await self._route_task(job, parameters)
            if job_type == "analyze_performance":
                return await self._analyze_performance(job)
            if job_type == "optimize_resources":
                return await self._optimize_resources(job)
            if job_type == "get_agent_status":
                return await self._get_agent_status(job)
            if job_type == "emergency_recovery":
                return await self._emergency_recovery(job, parameters)
            msg = f"Unsupported job type: {job_type}"
            raise ValueError(msg)

        except Exception as e:
            logger.exception(f"Orchestration job {job.job_id} failed: {e!s}")
            raise

    async def start_task_processor(self) -> None:
        """Start the background task processor."""
        if self._task_processor is None:
            self._task_processor = asyncio.create_task(self._process_tasks())
            logger.info("Started task processor")

    async def stop_task_processor(self) -> None:
        """Stop the background task processor."""
        if self._task_processor:
            self._task_processor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task_processor
            self._task_processor = None
            logger.info("Stopped task processor")

    async def _create_workflow(self, job: UnifiedJob, parameters: dict[str, Any]) -> dict[str, Any]:
        """Create a new workflow definition."""
        job.update_progress(20.0, "Creating workflow definition")

        workflow_id = parameters.get("workflow_id", f"workflow_{uuid.uuid4().hex[:8]}")
        name = parameters.get("name", "Unnamed Workflow")
        description = parameters.get("description", "")

        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            metadata=parameters.get("metadata", {}),
        )

        # Create tasks from specification
        task_specs = parameters.get("tasks", [])

        for i, task_spec in enumerate(task_specs):
            task_id = task_spec.get("task_id", f"task_{i}")

            task = Task(
                task_id=task_id,
                agent_type=task_spec["agent_type"],
                job_type=task_spec["job_type"],
                parameters=task_spec.get("parameters", {}),
                priority=TaskPriority(task_spec.get("priority", 2)),
                dependencies=task_spec.get("dependencies", []),
            )

            workflow.tasks.append(task)

        job.update_progress(60.0, f"Created workflow with {len(workflow.tasks)} tasks")

        # Validate workflow dependencies
        await self._validate_workflow(workflow)

        job.update_progress(80.0, "Validating workflow")

        # Store workflow
        async with self._workflow_lock:
            self._workflows[workflow_id] = workflow

        job.update_progress(100.0, "Workflow created successfully")

        return {
            "workflow_id": workflow_id,
            "name": name,
            "task_count": len(workflow.tasks),
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
        }

    async def _execute_workflow(self, job: UnifiedJob, workflow_id: str) -> dict[str, Any]:
        """Execute a workflow with all its tasks."""
        job.update_progress(10.0, f"Starting workflow execution: {workflow_id}")

        async with self._workflow_lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                msg = f"Workflow {workflow_id} not found"
                raise ValueError(msg)

        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now(UTC)

        try:
            # Build dependency graph
            job.update_progress(20.0, "Building task dependency graph")
            execution_order = await self._build_execution_order(workflow.tasks)

            # Execute tasks in dependency order
            completed_tasks = {}
            failed_tasks = {}

            total_tasks = len(workflow.tasks)
            completed_count = 0

            for task_batch in execution_order:
                # Execute tasks in this batch concurrently
                batch_tasks = []
                for task_id in task_batch:
                    task = next(t for t in workflow.tasks if t.task_id == task_id)
                    batch_tasks.append(self._execute_task(task, completed_tasks))

                # Wait for batch completion
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for task_id, result in zip(task_batch, batch_results, strict=False):
                    if isinstance(result, Exception):
                        failed_tasks[task_id] = str(result)
                        logger.error(f"Task {task_id} failed: {result}")
                    else:
                        completed_tasks[task_id] = result
                        completed_count += 1

                    # Update progress
                    progress = 20.0 + (70.0 * completed_count / total_tasks)
                    job.update_progress(
                        progress, f"Completed {completed_count}/{total_tasks} tasks"
                    )

                # Stop on critical failures
                if failed_tasks and workflow.metadata.get("stop_on_failure", True):
                    break

            # Update workflow status
            if failed_tasks:
                workflow.status = WorkflowStatus.FAILED
            else:
                workflow.status = WorkflowStatus.COMPLETED

            workflow.completed_at = datetime.now(UTC)

            # Update orchestration stats
            self._orchestration_stats["workflows_executed"] += 1
            self._orchestration_stats["tasks_completed"] += len(completed_tasks)
            self._orchestration_stats["tasks_failed"] += len(failed_tasks)

            execution_time = (workflow.completed_at - workflow.started_at).total_seconds()

            job.update_progress(100.0, f"Workflow completed: {workflow.status.value}")

            return {
                "workflow_id": workflow_id,
                "status": workflow.status.value,
                "execution_time": execution_time,
                "completed_tasks": len(completed_tasks),
                "failed_tasks": len(failed_tasks),
                "results": completed_tasks,
                "errors": failed_tasks,
                "started_at": workflow.started_at.isoformat(),
                "completed_at": workflow.completed_at.isoformat(),
            }

        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.now(UTC)
            logger.exception(f"Workflow {workflow_id} failed: {e!s}")
            raise

    async def _route_task(self, job: UnifiedJob, parameters: dict[str, Any]) -> dict[str, Any]:
        """Route a single task to the appropriate agent."""
        job.update_progress(15.0, "Analyzing task for routing")

        task_type = parameters["task_type"]
        task_parameters = parameters.get("parameters", {})

        # Determine best agent using AI
        routing_decision = await self._determine_optimal_agent(task_type, task_parameters)

        agent_type = routing_decision["agent_type"]
        job_type = routing_decision["job_type"]

        if agent_type not in self.agents:
            return {"error": f"Agent type {agent_type} not available"}

        job.update_progress(40.0, f"Routing to {agent_type} agent")

        # Execute task on selected agent
        agent = self.agents[agent_type]

        # Create job on agent
        agent_job_id = await agent.create_job(job_type, task_parameters)

        job.update_progress(60.0, f"Executing job {agent_job_id} on {agent_type}")

        # Execute and get result
        result = await agent.execute_job_async(agent_job_id)

        # Update agent load tracking
        self._agent_load[agent_type] = self._agent_load.get(agent_type, 0) + 1

        job.update_progress(100.0, "Task routing completed")

        return {
            "task_type": task_type,
            "routed_to": agent_type,
            "job_type": job_type,
            "agent_job_id": agent_job_id,
            "result": result,
            "routing_decision": routing_decision,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _analyze_performance(self, job: UnifiedJob) -> dict[str, Any]:
        """Analyze orchestration and agent performance."""
        job.update_progress(20.0, "Collecting performance metrics")

        # Collect agent statistics
        agent_stats = {}
        for agent_type, agent in self.agents.items():
            agent_info = agent.get_agent_info()
            agent_stats[agent_type] = {
                "statistics": agent_info["statistics"],
                "job_count": agent_info["job_count"],
                "current_load": self._agent_load.get(agent_type, 0),
            }

        job.update_progress(50.0, "Analyzing performance patterns")

        # Calculate performance metrics
        total_tasks = sum(
            stats["statistics"]["jobs_completed"] + stats["statistics"]["jobs_failed"]
            for stats in agent_stats.values()
        )
        total_failed = sum(stats["statistics"]["jobs_failed"] for stats in agent_stats.values())

        error_rate = (total_failed / total_tasks) if total_tasks > 0 else 0.0

        # AI-powered performance analysis
        job.update_progress(75.0, "Generating performance insights")

        performance_insights = await self._generate_performance_insights(
            agent_stats, self._orchestration_stats
        )

        job.update_progress(100.0, "Performance analysis completed")

        return {
            "orchestration_stats": self._orchestration_stats.copy(),
            "agent_stats": agent_stats,
            "system_metrics": {
                "total_tasks": total_tasks,
                "error_rate": error_rate,
                "active_workflows": len(
                    [w for w in self._workflows.values() if w.status == WorkflowStatus.RUNNING]
                ),
            },
            "performance_insights": performance_insights,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _optimize_resources(self, job: UnifiedJob) -> dict[str, Any]:
        """Optimize resource allocation and agent utilization."""
        job.update_progress(25.0, "Analyzing resource utilization")

        # Analyze current load distribution
        load_analysis = await self._analyze_load_distribution()

        job.update_progress(50.0, "Generating optimization recommendations")

        # Generate optimization recommendations with AI
        optimization_recommendations = await self._generate_optimization_recommendations(
            load_analysis
        )

        job.update_progress(75.0, "Applying optimizations")

        # Apply automatic optimizations
        applied_optimizations = await self._apply_optimizations(optimization_recommendations)

        job.update_progress(100.0, "Resource optimization completed")

        return {
            "load_analysis": load_analysis,
            "recommendations": optimization_recommendations,
            "applied_optimizations": applied_optimizations,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _get_agent_status(self, job: UnifiedJob) -> dict[str, Any]:
        """Get comprehensive status of all agents."""
        job.update_progress(30.0, "Collecting agent status")

        status = {
            "orchestrator": {
                "agent_id": self.agent_id,
                "status": "running",
                "workflows": len(self._workflows),
                "running_workflows": len(
                    [w for w in self._workflows.values() if w.status == WorkflowStatus.RUNNING]
                ),
                "task_processor": self._task_processor is not None,
            },
            "agents": {},
        }

        for agent_type, agent in self.agents.items():
            agent_info = agent.get_agent_info()
            status["agents"][agent_type] = {
                "agent_id": agent_info["agent_id"],
                "description": agent_info["description"],
                "supported_job_types": agent_info["supported_job_types"],
                "statistics": agent_info["statistics"],
                "job_count": agent_info["job_count"],
                "current_load": self._agent_load.get(agent_type, 0),
            }

        job.update_progress(100.0, "Agent status collected")

        return status

    async def _emergency_recovery(
        self, job: UnifiedJob, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform emergency recovery operations."""
        job.update_progress(20.0, "Initiating emergency recovery")

        recovery_type = parameters.get("recovery_type", "full")

        recovery_actions = []

        if recovery_type in ["full", "workflows"]:
            # Cancel all running workflows
            cancelled_workflows = []
            async with self._workflow_lock:
                for workflow_id, workflow in self._workflows.items():
                    if workflow.status == WorkflowStatus.RUNNING:
                        workflow.status = WorkflowStatus.CANCELLED
                        workflow.completed_at = datetime.now(UTC)
                        cancelled_workflows.append(workflow_id)

            recovery_actions.append(f"Cancelled {len(cancelled_workflows)} running workflows")

        job.update_progress(50.0, "Resetting agent states")

        if recovery_type in ["full", "agents"]:
            # Reset agent states
            for agent_type, agent in self.agents.items():
                try:
                    await agent.cleanup()
                    self._agent_load[agent_type] = 0
                    recovery_actions.append(f"Reset {agent_type} agent")
                except Exception as e:
                    recovery_actions.append(f"Failed to reset {agent_type}: {e!s}")

        job.update_progress(75.0, "Clearing task queues")

        if recovery_type in ["full", "queues"]:
            # Clear task queues
            queue_counts = {}
            for priority, queue in self._task_queues.items():
                count = queue.qsize()
                queue_counts[priority.name] = count
                # Clear queue
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            recovery_actions.append(f"Cleared task queues: {queue_counts}")

        job.update_progress(100.0, "Emergency recovery completed")

        return {
            "recovery_type": recovery_type,
            "recovery_actions": recovery_actions,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    # Helper methods

    async def _process_tasks(self) -> None:
        """Background task processor."""
        logger.info("Task processor started")

        while True:
            try:
                # Process tasks by priority
                for priority in [
                    TaskPriority.CRITICAL,
                    TaskPriority.HIGH,
                    TaskPriority.MEDIUM,
                    TaskPriority.LOW,
                ]:
                    queue = self._task_queues[priority]

                    if not queue.empty() and len(self._running_tasks) < self._max_concurrent_tasks:
                        try:
                            task = queue.get_nowait()
                            asyncio.create_task(self._execute_queued_task(task))
                        except asyncio.QueueEmpty:
                            continue

                # Wait before next iteration
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Task processor error: {e}")
                await asyncio.sleep(1)

        logger.info("Task processor stopped")

    async def _execute_queued_task(self, task: Task) -> None:
        """Execute a queued task."""
        self._running_tasks.add(task.task_id)

        try:
            agent = self.agents.get(task.agent_type)
            if not agent:
                msg = f"Agent {task.agent_type} not available"
                raise ValueError(msg)

            task.status = JobStatus.RUNNING
            task.started_at = datetime.now(UTC)

            # Execute task
            agent_job_id = await agent.create_job(task.job_type, task.parameters)
            result = await agent.execute_job_async(agent_job_id)

            task.status = JobStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now(UTC)

        except Exception as e:
            task.status = JobStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now(UTC)

        finally:
            self._running_tasks.discard(task.task_id)

    async def _execute_task(self, task: Task, context: dict[str, Any]) -> Any:
        """Execute a single task with context."""
        agent = self.agents.get(task.agent_type)
        if not agent:
            msg = f"Agent {task.agent_type} not available"
            raise ValueError(msg)

        task.status = JobStatus.RUNNING
        task.started_at = datetime.now(UTC)

        try:
            # Add context data to parameters
            enhanced_params = task.parameters.copy()
            for dep_task_id in task.dependencies:
                if dep_task_id in context:
                    enhanced_params[f"dependency_{dep_task_id}"] = context[dep_task_id]

            # Execute task
            agent_job_id = await agent.create_job(task.job_type, enhanced_params)
            result = await agent.execute_job_async(agent_job_id)

            task.status = JobStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now(UTC)

            return result

        except Exception as e:
            task.status = JobStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now(UTC)
            raise

    async def cleanup(self) -> None:
        """Cleanup orchestrator resources."""
        await self.stop_task_processor()

        # Cleanup all agents
        for agent in self.agents.values():
            await agent.cleanup()

        await super().cleanup()
