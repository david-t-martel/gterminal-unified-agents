"""Gemini Server Agent - Comprehensive AI super-agent with advanced capabilities.

This agent provides:
- Rust-exec integration for local command execution
- Cost optimization with intelligent model selection and caching
- GCP/GCS integrations with service account auth
- Session management with Redis persistence
- MCP server tool integration
- Real-time visibility with WebSocket updates
- Multi-tasking with priority queue system
- Full visibility and control for Claude and human users
"""

import asyncio
from collections import defaultdict
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
import uuid

import aioredis
from google.cloud import bigquery
from google.cloud import storage
from google.oauth2 import service_account
from pydantic import BaseModel
from pydantic import Field
import websockets

from gterminal.agents.base_agent_service import BaseAgentService
from gterminal.agents.base_agent_service import Job
from gterminal.automation.gemini_config import get_model_for_task
from gterminal.performance.cache import RedisCacheManager


class TaskPriority(Enum):
    """Task priority levels for queue management."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class ModelTier(Enum):
    """Model tier for cost optimization."""

    FLASH_THINKING = "gemini-2.0-flash-thinking-exp"  # Most expensive, highest quality
    FLASH_EXP = "gemini-2.0-flash-exp"  # Balanced performance/cost
    FLASH_SPEED = "gemini-2.5-flash"  # Fastest, most cost-effective


class TaskMetrics(BaseModel):
    """Task execution metrics for monitoring."""

    task_id: str
    start_time: datetime
    end_time: datetime | None = None
    duration_ms: int | None = None
    tokens_used: int = 0
    model_used: str = ""
    cost_estimate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    success: bool = True
    error_message: str | None = None


class SessionInfo(BaseModel):
    """Session information with persistence."""

    session_id: str
    user_id: str = "default"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context_window: int = 200000
    model_preference: ModelTier = ModelTier.FLASH_EXP
    cost_limit_daily: float = 100.0
    cost_used_today: float = 0.0
    tasks_completed: int = 0
    active: bool = True


class PriorityTask(BaseModel):
    """Task with priority and dependencies."""

    task_id: str
    priority: TaskPriority
    job_id: str
    dependencies: list[str] = Field(default_factory=list)
    max_retries: int = 3
    retry_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    scheduled_at: datetime | None = None
    resource_requirements: dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class GeminiServerAgent(BaseAgentService):
    """Advanced Gemini Server Agent with comprehensive capabilities.

    Features:
    - Multi-tasking with priority queue
    - Cost optimization and budget management
    - Real-time monitoring and WebSocket updates
    - MCP server integration
    - Session persistence with Redis
    - GCP/GCS integration
    - Rust-exec command execution
    """

    def __init__(self) -> None:
        super().__init__(
            agent_name="gemini-server-agent",
            description="Comprehensive AI super-agent with advanced multi-tasking capabilities",
        )

        # Initialize components
        self._setup_advanced_logging()
        self._initialize_redis()
        self._initialize_gcp()
        self._initialize_priority_queue()
        self._initialize_monitoring()
        self._initialize_mcp_integrations()

        # Configuration
        self.max_concurrent_tasks = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))
        self.context_window_limit = int(os.getenv("CONTEXT_WINDOW_LIMIT", "200000"))
        self.websocket_port = int(os.getenv("WEBSOCKET_PORT", "8765"))

        # State tracking
        self.active_sessions: dict[str, SessionInfo] = {}
        self.task_metrics: dict[str, TaskMetrics] = {}
        self.connected_clients: set = set()
        self.resource_usage = defaultdict(float)

        self.logger.info("GeminiServerAgent initialized with advanced capabilities")

    def _setup_advanced_logging(self) -> None:
        """Setup advanced logging with structured output."""
        # Create logs directory
        log_dir = Path("/tmp/gemini-server-agent")
        log_dir.mkdir(exist_ok=True)

        # Setup file handler with JSON format
        log_file = log_dir / f"agent-{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.info("Advanced logging initialized")

    def _initialize_redis(self) -> None:
        """Initialize Redis connection for caching and session management."""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_cache = RedisCacheManager(redis_url)
            self.redis_client = aioredis.from_url(redis_url)
            self.logger.info("Redis connection initialized")
        except Exception as e:
            self.logger.exception(f"Redis initialization failed: {e}")
            self.redis_cache = None
            self.redis_client = None

    def _initialize_gcp(self) -> None:
        """Initialize GCP services with service account authentication."""
        try:
            # Service account authentication
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path and Path(credentials_path).exists():
                self.gcp_credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=[
                        "https://www.googleapis.com/auth/cloud-platform",
                        "https://www.googleapis.com/auth/bigquery",
                        "https://www.googleapis.com/auth/devstorage.full_control",
                    ],
                )

                # Initialize GCP clients
                self.storage_client = storage.Client(credentials=self.gcp_credentials)
                self.bigquery_client = bigquery.Client(credentials=self.gcp_credentials)

                # GCS bucket for context persistence
                self.context_bucket_name = os.getenv("GCS_CONTEXT_BUCKET", "gemini-agent-contexts")

                self.logger.info("GCP services initialized with service account")
            else:
                self.logger.warning("GCP credentials not found, using default authentication")
                self.gcp_credentials = None
                self.storage_client = None
                self.bigquery_client = None

        except Exception as e:
            self.logger.exception(f"GCP initialization failed: {e}")
            self.gcp_credentials = None

    def _initialize_priority_queue(self) -> None:
        """Initialize priority queue system for multi-tasking."""
        self.task_queue = asyncio.PriorityQueue()
        self.running_tasks: dict[str, asyncio.Task] = {}
        self.task_dependencies: dict[str, set] = defaultdict(set)
        self.completed_tasks: set = set()

        # Start queue processor
        asyncio.create_task(self._process_task_queue())
        self.logger.info("Priority queue system initialized")

    def _initialize_monitoring(self) -> None:
        """Initialize real-time monitoring system."""
        self.monitoring_active = True
        self.performance_metrics = {
            "tasks_per_minute": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "memory_usage": 0.0,
            "cpu_usage": 0.0,
            "active_connections": 0,
        }

        # Start monitoring tasks
        asyncio.create_task(self._collect_metrics())
        asyncio.create_task(self._start_websocket_server())

        self.logger.info("Real-time monitoring initialized")

    def _initialize_mcp_integrations(self) -> None:
        """Initialize MCP server tool integrations."""
        self.mcp_tools = {
            "rust-fs": self._get_rust_fs_client,
            "rust-fetch": self._get_rust_fetch_client,
            "desktop-commander-wsl": self._get_desktop_commander_client,
            "rust-exec": self._get_rust_exec_client,
        }

        # Test MCP connections
        asyncio.create_task(self._test_mcp_connections())
        self.logger.info("MCP integrations initialized")

    async def _get_rust_exec_client(self):
        """Get rust-exec client for command execution."""
        try:
            # Use the rust-exec binary directly
            rust_exec_path = "/home/david/.claude/rust-exec/target/release/claude-exec"
            if not Path(rust_exec_path).exists():
                # Build if not exists
                build_cmd = ["cargo", "build", "--release"]
                process = await asyncio.create_subprocess_exec(
                    *build_cmd,
                    cwd="/home/david/.claude/rust-exec",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await process.communicate()

            return rust_exec_path
        except Exception as e:
            self.logger.exception(f"Rust-exec client initialization failed: {e}")
            return None

    async def execute_rust_command(
        self, command: str, args: list[str] | None = None
    ) -> dict[str, Any]:
        """Execute command using rust-exec for high performance."""
        try:
            rust_exec = await self._get_rust_exec_client()
            if not rust_exec:
                return {"error": "Rust-exec not available"}

            # Prepare command
            full_command = [rust_exec, "exec", command]
            if args:
                full_command.extend(args)

            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)

            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "return_code": process.returncode,
            }

        except TimeoutError:
            return {"error": "Command execution timeout"}
        except Exception as e:
            return {"error": f"Command execution failed: {e}"}

    async def _test_mcp_connections(self):
        """Test all MCP server connections."""
        results: dict[str, Any] = {}
        for tool_name, client_func in self.mcp_tools.items():
            try:
                client = await client_func()
                results[tool_name] = client is not None
            except Exception as e:
                self.logger.exception(f"MCP connection test failed for {tool_name}: {e}")
                results[tool_name] = False

        self.logger.info(f"MCP connection test results: {results}")
        return results

    def optimize_model_selection(
        self, task_type: str, context_length: int, budget_remaining: float
    ) -> ModelTier:
        """Intelligent model selection based on task type, context, and budget."""
        # Cost per 1K tokens (approximate)
        model_costs = {
            ModelTier.FLASH_THINKING: 0.0015,
            ModelTier.FLASH_EXP: 0.001,
            ModelTier.FLASH_SPEED: 0.0005,
        }

        # Estimate tokens (rough approximation)
        estimated_tokens = context_length / 4

        # Calculate costs
        costs = {tier: estimated_tokens * cost / 1000 for tier, cost in model_costs.items()}

        # Select based on task type and budget
        if task_type in ["code_review", "architecture", "complex_analysis"]:
            # High-quality tasks
            if budget_remaining > costs[ModelTier.FLASH_THINKING]:
                return ModelTier.FLASH_THINKING
            if budget_remaining > costs[ModelTier.FLASH_EXP]:
                return ModelTier.FLASH_EXP

        elif task_type in ["documentation", "simple_analysis", "formatting"]:
            # Speed-focused tasks
            if budget_remaining > costs[ModelTier.FLASH_SPEED]:
                return ModelTier.FLASH_SPEED
            if budget_remaining > costs[ModelTier.FLASH_EXP]:
                return ModelTier.FLASH_EXP

        # Default to most economical option
        return ModelTier.FLASH_SPEED

    async def create_session(
        self, user_id: str = "default", preferences: dict[str, Any] | None = None
    ) -> str:
        """Create new session with persistence."""
        session_id = str(uuid.uuid4())

        session_info = SessionInfo(session_id=session_id, user_id=user_id)

        # Apply preferences
        if preferences:
            if "model_preference" in preferences:
                session_info.model_preference = ModelTier(preferences["model_preference"])
            if "cost_limit_daily" in preferences:
                session_info.cost_limit_daily = preferences["cost_limit_daily"]
            if "context_window" in preferences:
                session_info.context_window = preferences["context_window"]

        self.active_sessions[session_id] = session_info

        # Persist to Redis
        if self.redis_client:
            await self.redis_client.setex(
                f"session:{session_id}",
                3600 * 24,  # 24 hours
                session_info.model_dump_json(),
            )

        self.logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    async def get_session(self, session_id: str) -> SessionInfo | None:
        """Retrieve session information."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # Try Redis
        if self.redis_client:
            try:
                session_data = await self.redis_client.get(f"session:{session_id}")
                if session_data:
                    session_info = SessionInfo.model_validate_json(session_data)
                    self.active_sessions[session_id] = session_info
                    return session_info
            except Exception as e:
                self.logger.exception(f"Session retrieval failed: {e}")

        return None

    async def add_priority_task(
        self,
        job_id: str,
        priority: TaskPriority,
        dependencies: list[str] | None = None,
        resource_requirements: dict[str, Any] | None = None,
    ) -> str:
        """Add task to priority queue."""
        task_id = str(uuid.uuid4())

        task = PriorityTask(
            task_id=task_id,
            priority=priority,
            job_id=job_id,
            dependencies=dependencies or [],
            resource_requirements=resource_requirements or {},
        )

        # Add to queue with priority (lower number = higher priority)
        await self.task_queue.put((priority.value, time.time(), task))

        # Track dependencies
        if dependencies:
            self.task_dependencies[task_id] = set(dependencies)

        self.logger.info(f"Added priority task {task_id} with priority {priority.name}")
        return task_id

    async def _process_task_queue(self) -> None:
        """Process tasks from priority queue."""
        while True:
            try:
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                    continue

                # Get next task
                priority, timestamp, task = await self.task_queue.get()

                # Check dependencies
                if task.dependencies and not all(
                    dep in self.completed_tasks for dep in task.dependencies
                ):
                    # Re-queue if dependencies not met
                    await self.task_queue.put((priority, timestamp, task))
                    await asyncio.sleep(0.1)
                    continue

                # Check resource requirements
                if not self._check_resource_availability(task.resource_requirements):
                    # Re-queue if resources not available
                    await self.task_queue.put((priority, timestamp, task))
                    await asyncio.sleep(0.5)
                    continue

                # Execute task
                asyncio_task = asyncio.create_task(self._execute_priority_task(task))
                self.running_tasks[task.task_id] = asyncio_task

            except Exception as e:
                self.logger.exception(f"Task queue processing error: {e}")
                await asyncio.sleep(1)

    def _check_resource_availability(self, requirements: dict[str, Any]) -> bool:
        """Check if required resources are available."""
        if not requirements:
            return True

        # Check memory requirement
        if "memory_mb" in requirements:
            current_memory = self.resource_usage.get("memory_mb", 0)
            if current_memory + requirements["memory_mb"] > 8000:  # 8GB limit
                return False

        # Check CPU requirement
        if "cpu_percent" in requirements:
            current_cpu = self.resource_usage.get("cpu_percent", 0)
            if current_cpu + requirements["cpu_percent"] > 80:  # 80% limit
                return False

        return True

    async def _execute_priority_task(self, task: PriorityTask) -> None:
        """Execute a priority task."""
        start_time = datetime.now(UTC)

        try:
            # Get job
            job = self.get_job(task.job_id)
            if not job:
                self.logger.error(f"Job {task.job_id} not found for task {task.task_id}")
                return

            # Track resource usage
            self._allocate_resources(task.resource_requirements)

            # Create metrics
            metrics = TaskMetrics(task_id=task.task_id, start_time=start_time)
            self.task_metrics[task.task_id] = metrics

            # Execute job
            result = await self.execute_job_async(task.job_id)

            # Update metrics
            end_time = datetime.now(UTC)
            metrics.end_time = end_time
            metrics.duration_ms = int((end_time - start_time).total_seconds() * 1000)
            metrics.success = result.get("status") == "success"

            if not metrics.success:
                metrics.error_message = result.get("error", "Unknown error")

            # Broadcast update
            await self._broadcast_task_update(task.task_id, "completed", result)

            # Mark as completed
            self.completed_tasks.add(task.task_id)

        except Exception as e:
            self.logger.exception(f"Task execution failed: {e}")
            await self._broadcast_task_update(task.task_id, "failed", {"error": str(e)})

        finally:
            # Clean up
            self._deallocate_resources(task.resource_requirements)
            self.running_tasks.pop(task.task_id, None)

    def _allocate_resources(self, requirements: dict[str, Any]) -> None:
        """Allocate resources for task execution."""
        for resource, amount in requirements.items():
            self.resource_usage[resource] += amount

    def _deallocate_resources(self, requirements: dict[str, Any]) -> None:
        """Deallocate resources after task completion."""
        for resource, amount in requirements.items():
            self.resource_usage[resource] -= amount
            self.resource_usage[resource] = max(0, self.resource_usage[resource])

    async def _collect_metrics(self) -> None:
        """Collect performance metrics continuously."""
        while self.monitoring_active:
            try:
                import psutil

                # System metrics
                self.performance_metrics["memory_usage"] = psutil.virtual_memory().percent
                self.performance_metrics["cpu_usage"] = psutil.cpu_percent()

                # Task metrics
                completed_last_minute = sum(
                    1
                    for metrics in self.task_metrics.values()
                    if metrics.end_time
                    and (datetime.now(UTC) - metrics.end_time).total_seconds() < 60
                )
                self.performance_metrics["tasks_per_minute"] = completed_last_minute

                # Error rate
                total_tasks = len(self.task_metrics)
                failed_tasks = sum(1 for m in self.task_metrics.values() if not m.success)
                self.performance_metrics["error_rate"] = failed_tasks / max(total_tasks, 1) * 100

                # Average response time
                response_times = [
                    m.duration_ms for m in self.task_metrics.values() if m.duration_ms is not None
                ]
                self.performance_metrics["average_response_time"] = (
                    (sum(response_times) / max(len(response_times), 1)) if response_times else 0
                )

                # Active connections
                self.performance_metrics["active_connections"] = len(self.connected_clients)

                # Store in BigQuery if available
                if self.bigquery_client:
                    await self._store_metrics_bigquery()

                await asyncio.sleep(10)  # Collect every 10 seconds

            except Exception as e:
                self.logger.exception(f"Metrics collection failed: {e}")
                await asyncio.sleep(30)

    async def _store_metrics_bigquery(self) -> None:
        """Store metrics in BigQuery for analytics."""
        try:
            table_id = "gemini_agent_metrics"
            rows_to_insert = [
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "agent_name": self.agent_name,
                    **self.performance_metrics,
                    "active_tasks": len(self.running_tasks),
                    "queued_tasks": self.task_queue.qsize(),
                    "total_sessions": len(self.active_sessions),
                },
            ]

            # Insert in background to avoid blocking
            asyncio.create_task(self._insert_bigquery_rows(table_id, rows_to_insert))

        except Exception as e:
            self.logger.exception(f"BigQuery metrics storage failed: {e}")

    async def _insert_bigquery_rows(self, table_id: str, rows: list[dict]) -> None:
        """Insert rows to BigQuery asynchronously."""
        try:
            os.getenv("GOOGLE_CLOUD_PROJECT")
            dataset_id = "gemini_agent_analytics"

            table_ref = self.bigquery_client.dataset(dataset_id).table(table_id)
            errors = self.bigquery_client.insert_rows_json(table_ref, rows)

            if errors:
                self.logger.error(f"BigQuery insert errors: {errors}")

        except Exception as e:
            self.logger.exception(f"BigQuery insert failed: {e}")

    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for real-time updates."""
        try:

            async def handle_client(websocket, path) -> None:
                self.connected_clients.add(websocket)
                self.logger.info(f"New WebSocket client connected: {path}")

                try:
                    # Send initial status
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "status",
                                "data": {
                                    "agent_name": self.agent_name,
                                    "active_tasks": len(self.running_tasks),
                                    "queued_tasks": self.task_queue.qsize(),
                                    "metrics": self.performance_metrics,
                                },
                            },
                        ),
                    )

                    # Keep connection alive
                    async for message in websocket:
                        # Handle client messages
                        try:
                            data = json.loads(message)
                            await self._handle_websocket_message(websocket, data)
                        except json.JSONDecodeError:
                            await websocket.send(
                                json.dumps({"type": "error", "message": "Invalid JSON message"})
                            )

                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.connected_clients.discard(websocket)
                    self.logger.info("WebSocket client disconnected")

            # Start server
            server = await websockets.serve(handle_client, "localhost", self.websocket_port)

            self.logger.info(f"WebSocket server started on port {self.websocket_port}")
            await server.wait_closed()

        except Exception as e:
            self.logger.exception(f"WebSocket server failed: {e}")

    async def _handle_websocket_message(self, websocket, data: dict[str, Any]) -> None:
        """Handle incoming WebSocket messages."""
        message_type = data.get("type")

        if message_type == "get_status":
            await websocket.send(
                json.dumps(
                    {
                        "type": "status",
                        "data": {
                            "agent_name": self.agent_name,
                            "active_tasks": len(self.running_tasks),
                            "queued_tasks": self.task_queue.qsize(),
                            "metrics": self.performance_metrics,
                            "resource_usage": dict(self.resource_usage),
                        },
                    },
                ),
            )

        elif message_type == "get_tasks":
            tasks_info: dict[str, Any] = {}
            for task_id, metrics in self.task_metrics.items():
                tasks_info[task_id] = {
                    "start_time": metrics.start_time.isoformat(),
                    "duration_ms": metrics.duration_ms,
                    "success": metrics.success,
                    "model_used": metrics.model_used,
                    "tokens_used": metrics.tokens_used,
                }

            await websocket.send(json.dumps({"type": "tasks", "data": tasks_info}))

        elif message_type == "cancel_task":
            task_id = data.get("task_id")
            if task_id in self.running_tasks:
                self.running_tasks[task_id].cancel()
                await websocket.send(json.dumps({"type": "task_cancelled", "task_id": task_id}))

    async def _broadcast_task_update(self, task_id: str, status: str, data: dict[str, Any]) -> None:
        """Broadcast task updates to all connected clients."""
        if not self.connected_clients:
            return

        message = json.dumps(
            {
                "type": "task_update",
                "task_id": task_id,
                "status": status,
                "timestamp": datetime.now(UTC).isoformat(),
                "data": data,
            },
        )

        # Send to all connected clients
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)

        # Clean up disconnected clients
        self.connected_clients -= disconnected

    async def save_context_to_gcs(self, session_id: str, context_data: str) -> bool:
        """Save session context to Google Cloud Storage."""
        if not self.storage_client:
            return False

        try:
            bucket = self.storage_client.bucket(self.context_bucket_name)
            blob_name = f"contexts/{session_id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            blob = bucket.blob(blob_name)

            # Upload context with metadata
            blob.upload_from_string(context_data, content_type="application/json")

            # Set metadata
            blob.metadata = {
                "session_id": session_id,
                "created_at": datetime.now(UTC).isoformat(),
                "agent": self.agent_name,
            }
            blob.patch()

            self.logger.info(f"Context saved to GCS: {blob_name}")
            return True

        except Exception as e:
            self.logger.exception(f"GCS context save failed: {e}")
            return False

    async def load_context_from_gcs(self, session_id: str) -> str | None:
        """Load latest session context from Google Cloud Storage."""
        if not self.storage_client:
            return None

        try:
            bucket = self.storage_client.bucket(self.context_bucket_name)
            prefix = f"contexts/{session_id}/"

            # List blobs and get the latest
            blobs = list(bucket.list_blobs(prefix=prefix))
            if not blobs:
                return None

            # Sort by name (timestamp) and get latest
            latest_blob = sorted(blobs, key=lambda b: b.name)[-1]
            context_data = latest_blob.download_as_text()

            self.logger.info(f"Context loaded from GCS: {latest_blob.name}")
            return context_data

        except Exception as e:
            self.logger.exception(f"GCS context load failed: {e}")
            return None

    async def cleanup_sessions(self, max_age_hours: int = 24) -> None:
        """Clean up expired sessions."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=max_age_hours)

        expired_sessions: list[Any] = []
        for session_id, session_info in self.active_sessions.items():
            if session_info.last_accessed < cutoff_time:
                expired_sessions.append(session_id)

        # Remove expired sessions
        for session_id in expired_sessions:
            del self.active_sessions[session_id]

            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(f"session:{session_id}")

        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def register_tools(self) -> None:
        """Register MCP tools for the super-agent."""

        @self.mcp.tool()
        def create_priority_task(
            job_type: str,
            parameters: str,
            priority: str = "normal",
            dependencies: str = "",
            resource_requirements: str = "{}",
        ) -> dict:
            """Create a new priority task in the queue.

            Args:
                job_type: Type of job to create
                parameters: JSON string of job parameters
                priority: Task priority (critical, high, normal, low, background)
                dependencies: Comma-separated list of task IDs this task depends on
                resource_requirements: JSON string of resource requirements

            """
            try:
                # Parse parameters
                params = json.loads(parameters)
                deps = [d.strip() for d in dependencies.split(",") if d.strip()]
                resources = json.loads(resource_requirements)

                # Create job
                job_id = self.create_job(job_type, params)

                # Add to priority queue
                task_priority = TaskPriority[priority.upper()]
                task_id = asyncio.create_task(
                    self.add_priority_task(job_id, task_priority, deps, resources)
                )

                return self.create_success_response(
                    {
                        "job_id": job_id,
                        "task_id": str(task_id),
                        "priority": priority,
                        "queue_position": self.task_queue.qsize(),
                    },
                    "Priority task created successfully",
                )

            except Exception as e:
                return self.create_error_response(f"Failed to create priority task: {e}")

        @self.mcp.tool()
        def get_agent_status(include_metrics: bool = True) -> dict:
            """Get comprehensive agent status and metrics.

            Args:
                include_metrics: Whether to include performance metrics

            """
            try:
                status = {
                    "agent_name": self.agent_name,
                    "active_tasks": len(self.running_tasks),
                    "queued_tasks": self.task_queue.qsize(),
                    "total_jobs": len(self.jobs),
                    "active_sessions": len(self.active_sessions),
                    "connected_clients": len(self.connected_clients),
                    "resource_usage": dict(self.resource_usage),
                }

                if include_metrics:
                    status["performance_metrics"] = self.performance_metrics
                    status["recent_tasks"] = [
                        {
                            "task_id": task_id,
                            "duration_ms": metrics.duration_ms,
                            "success": metrics.success,
                            "model_used": metrics.model_used,
                        }
                        for task_id, metrics in list(self.task_metrics.items())[-10:]
                    ]

                return self.create_success_response(status, "Agent status retrieved")

            except Exception as e:
                return self.create_error_response(f"Failed to get agent status: {e}")

        @self.mcp.tool()
        def execute_rust_command_tool(command: str, args: str = "") -> dict:
            """Execute system command using high-performance rust-exec.

            Args:
                command: Command to execute
                args: Space-separated command arguments

            """
            try:
                arg_list = args.split() if args else []
                result = asyncio.create_task(self.execute_rust_command(command, arg_list))

                return self.create_success_response(
                    {"command": command, "args": arg_list, "execution_result": result},
                    "Command executed via rust-exec",
                )

            except Exception as e:
                return self.create_error_response(f"Rust command execution failed: {e}")

        @self.mcp.tool()
        def manage_session(
            action: str,
            session_id: str = "",
            user_id: str = "default",
            preferences: str = "{}",
        ) -> dict:
            """Manage user sessions with persistence.

            Args:
                action: Action to perform (create, get, update, delete)
                session_id: Session ID (for get, update, delete actions)
                user_id: User ID (for create action)
                preferences: JSON string of session preferences

            """
            try:
                if action == "create":
                    prefs = json.loads(preferences) if preferences != "{}" else {}
                    session_id = asyncio.create_task(self.create_session(user_id, prefs))
                    return self.create_success_response(
                        {"session_id": session_id, "user_id": user_id},
                        "Session created successfully",
                    )

                if action == "get":
                    session = asyncio.create_task(self.get_session(session_id))
                    if session:
                        return self.create_success_response(
                            {"session": session.model_dump()},
                            "Session retrieved successfully",
                        )
                    return self.create_error_response("Session not found")

                # Add other session management actions as needed
                return self.create_error_response(f"Unknown action: {action}")

            except Exception as e:
                return self.create_error_response(f"Session management failed: {e}")

        @self.mcp.tool()
        def optimize_cost_strategy(current_usage: str, budget_limit: str, task_types: str) -> dict:
            """Get cost optimization recommendations.

            Args:
                current_usage: JSON string of current cost usage
                budget_limit: Daily budget limit as string
                task_types: Comma-separated list of expected task types

            """
            try:
                usage = json.loads(current_usage)
                budget = float(budget_limit)
                types = [t.strip() for t in task_types.split(",")]

                recommendations: list[Any] = []
                total_usage = usage.get("total_today", 0)
                remaining_budget = budget - total_usage

                # Model recommendations
                for task_type in types:
                    recommended_model = self.optimize_model_selection(
                        task_type, 50000, remaining_budget
                    )
                    recommendations.append(
                        {
                            "task_type": task_type,
                            "recommended_model": recommended_model.value,
                            "estimated_cost": remaining_budget * 0.1,  # Rough estimate
                        },
                    )

                return self.create_success_response(
                    {
                        "budget_remaining": remaining_budget,
                        "usage_percentage": (total_usage / budget) * 100,
                        "recommendations": recommendations,
                        "cost_saving_tips": [
                            "Use gemini-2.5-flash for simple tasks",
                            "Enable aggressive caching for repeated contexts",
                            "Batch similar tasks to reduce overhead",
                        ],
                    },
                    "Cost optimization analysis completed",
                )

            except Exception as e:
                return self.create_error_response(f"Cost optimization failed: {e}")

    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Execute specific job implementation with enhanced capabilities."""
        job_type = job.job_type
        params = job.parameters

        try:
            if job_type == "code_analysis":
                return await self._execute_code_analysis(job, params)
            if job_type == "system_command":
                return await self._execute_system_command(job, params)
            if job_type == "content_generation":
                return await self._execute_content_generation(job, params)
            if job_type == "file_processing":
                return await self._execute_file_processing(job, params)
            return {"error": f"Unknown job type: {job_type}"}

        except Exception as e:
            self.logger.exception(f"Job execution failed: {e}")
            return {"error": str(e)}

    async def _execute_code_analysis(self, job: Job, params: dict) -> dict[str, Any]:
        """Execute code analysis with intelligent model selection."""
        try:
            code_content = params.get("code", "")
            analysis_type = params.get("type", "general")

            # Select optimal model
            model_tier = self.optimize_model_selection(
                "code_analysis",
                len(code_content),
                100.0,  # Default budget
            )

            # Update job progress
            job.update_progress(20, f"Using model: {model_tier.value}")

            # Get model
            model = get_model_for_task("code_review")

            # Create analysis prompt
            prompt = f"""
            Analyze the following code for {analysis_type}:

            ```
            {code_content}
            ```

            Provide analysis in JSON format with:
            - issues: List of potential issues
            - suggestions: Improvement recommendations
            - metrics: Code quality metrics
            - security: Security considerations
            """

            job.update_progress(50, "Generating analysis...")

            # Generate analysis
            response = model.generate_content(prompt)
            analysis_text = response.text

            job.update_progress(80, "Processing results...")

            # Try to parse as JSON, fallback to text
            try:
                analysis_result = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis_result = {"analysis": analysis_text}

            job.update_progress(100, "Analysis completed")

            return {
                "analysis": analysis_result,
                "model_used": model_tier.value,
                "code_length": len(code_content),
                "analysis_type": analysis_type,
            }

        except Exception as e:
            return {"error": f"Code analysis failed: {e}"}

    async def _execute_system_command(self, job: Job, params: dict) -> dict[str, Any]:
        """Execute system command using rust-exec."""
        try:
            command = params.get("command", "")
            args = params.get("args", [])

            job.update_progress(30, "Executing command via rust-exec...")

            result = await self.execute_rust_command(command, args)

            job.update_progress(100, "Command execution completed")

            return {"command_result": result, "command": command, "args": args}

        except Exception as e:
            return {"error": f"System command failed: {e}"}

    async def _execute_content_generation(self, job: Job, params: dict) -> dict[str, Any]:
        """Execute content generation with context management."""
        try:
            prompt = params.get("prompt", "")
            content_type = params.get("type", "general")
            session_id = params.get("session_id")

            # Load context if session provided
            context = ""
            if session_id:
                job.update_progress(10, "Loading session context...")
                context = await self.load_context_from_gcs(session_id) or ""

            # Select model
            total_context_length = len(prompt) + len(context)
            model_tier = self.optimize_model_selection(content_type, total_context_length, 100.0)

            job.update_progress(30, f"Generating content with {model_tier.value}...")

            # Prepare full prompt with context
            full_prompt = f"{context}\n\n{prompt}" if context else prompt

            # Generate content
            model = get_model_for_task(content_type)
            response = model.generate_content(full_prompt)
            generated_content = response.text

            job.update_progress(80, "Saving context...")

            # Save updated context if session provided
            if session_id:
                updated_context = f"{context}\n\nUser: {prompt}\nAssistant: {generated_content}"
                await self.save_context_to_gcs(session_id, updated_context)

            job.update_progress(100, "Content generation completed")

            return {
                "content": generated_content,
                "model_used": model_tier.value,
                "context_length": total_context_length,
                "content_type": content_type,
            }

        except Exception as e:
            return {"error": f"Content generation failed: {e}"}

    async def _execute_file_processing(self, job: Job, params: dict) -> dict[str, Any]:
        """Execute file processing with MCP tool integration."""
        try:
            file_path = params.get("file_path", "")
            operation = params.get("operation", "read")

            job.update_progress(20, f"Processing file: {file_path}")

            # Use rust-fs for file operations
            if operation == "read":
                result = await self.execute_rust_command("cat", [file_path])
            elif operation == "list":
                result = await self.execute_rust_command("ls", ["-la", file_path])
            elif operation == "analyze":
                # Read file and analyze
                read_result = await self.execute_rust_command("cat", [file_path])
                if read_result.get("success"):
                    content = read_result.get("stdout", "")
                    # Perform analysis
                    analysis_job = Job(
                        str(uuid.uuid4()),
                        "code_analysis",
                        {"code": content, "type": "file_analysis"},
                    )
                    result = await self._execute_code_analysis(
                        analysis_job, {"code": content, "type": "file_analysis"}
                    )
                else:
                    result = read_result
            else:
                result = {"error": f"Unknown file operation: {operation}"}

            job.update_progress(100, "File processing completed")

            return {"file_operation_result": result, "file_path": file_path, "operation": operation}

        except Exception as e:
            return {"error": f"File processing failed: {e}"}


# Main execution
if __name__ == "__main__":
    agent = GeminiServerAgent()
    agent.run()
