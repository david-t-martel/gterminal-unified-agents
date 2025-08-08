"""Incident Response and Automated Remediation System.

This module provides comprehensive incident response capabilities including:
- Incident detection and classification
- Automated remediation actions
- Escalation management
- Post-incident analysis
- Runbook automation
- Recovery validation
"""

import asyncio
from collections import defaultdict
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
import logging
from typing import Any
import uuid

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class IncidentStatus(Enum):
    """Incident status."""

    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class RemediationStatus(Enum):
    """Remediation action status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class RemediationType(Enum):
    """Types of remediation actions."""

    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CIRCUIT_BREAKER = "circuit_breaker"
    TRAFFIC_REDIRECT = "traffic_redirect"
    CACHE_CLEAR = "cache_clear"
    CONNECTION_POOL_RESET = "connection_pool_reset"
    FALLBACK_ENABLE = "fallback_enable"
    RATE_LIMIT_ADJUST = "rate_limit_adjust"
    CUSTOM_SCRIPT = "custom_script"


@dataclass
class RemediationAction:
    """Definition of a remediation action."""

    name: str
    description: str
    remediation_type: RemediationType
    command: str  # Command to execute or action to take
    timeout_seconds: int = 300  # 5 minutes default
    requires_confirmation: bool = False  # Some actions need human approval
    rollback_command: str | None = None  # Command to rollback if needed
    validation_command: str | None = None  # Command to validate success
    prerequisites: list[str] = field(default_factory=list)  # Required conditions
    side_effects: list[str] = field(default_factory=list)  # Potential side effects
    estimated_impact: str = "low"  # low, medium, high

    # Execution constraints
    max_executions_per_hour: int = 3
    cooldown_minutes: int = 15

    # Metadata
    owner: str = ""
    documentation_url: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class RemediationExecution:
    """Record of a remediation action execution."""

    execution_id: str
    action_name: str
    incident_id: str
    status: RemediationStatus
    started_at: datetime
    completed_at: datetime | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    rollback_performed: bool = False
    validation_passed: bool = False

    @property
    def duration_seconds(self) -> float:
        """Calculate execution duration."""
        if not self.completed_at:
            return (datetime.now(UTC) - self.started_at).total_seconds()
        return (self.completed_at - self.started_at).total_seconds()


@dataclass
class Incident:
    """Incident record."""

    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    detected_at: datetime
    source: str  # alert, monitoring, user report
    affected_services: list[str] = field(default_factory=list)

    # Timeline
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    closed_at: datetime | None = None

    # Assignments
    assigned_to: str = ""
    escalated_to: str = ""

    # Remediation
    remediation_actions: list[str] = field(default_factory=list)  # Action IDs
    automatic_remediation_attempted: bool = False
    manual_intervention_required: bool = False

    # Context
    alerts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    # Communication
    notifications_sent: list[str] = field(default_factory=list)
    status_page_updated: bool = False

    # Post-incident
    root_cause: str = ""
    lessons_learned: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float:
        """Calculate incident duration in minutes."""
        end_time = self.resolved_at or datetime.now(UTC)
        return (end_time - self.detected_at).total_seconds() / 60

    @property
    def time_to_acknowledge_minutes(self) -> float | None:
        """Calculate time to acknowledgment."""
        if not self.acknowledged_at:
            return None
        return (self.acknowledged_at - self.detected_at).total_seconds() / 60

    @property
    def time_to_resolve_minutes(self) -> float | None:
        """Calculate time to resolution."""
        if not self.resolved_at:
            return None
        return (self.resolved_at - self.detected_at).total_seconds() / 60


@dataclass
class Playbook:
    """Incident response playbook."""

    name: str
    description: str
    triggers: list[str]  # Conditions that trigger this playbook
    applicable_severities: list[IncidentSeverity]
    remediation_steps: list[str]  # Remediation action names in order
    escalation_rules: dict[str, Any]

    # Automation settings
    auto_execute: bool = False  # Execute without human approval
    max_auto_actions: int = 3  # Max automated actions per incident
    require_confirmation_after: int = 1  # Require confirmation after N actions

    # Metadata
    owner_team: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    success_rate: float = 0.0  # Historical success rate
    avg_resolution_time_minutes: float = 0.0


class IncidentResponseSystem:
    """Comprehensive incident response and remediation system."""

    def __init__(self) -> None:
        self.start_time = datetime.now(UTC)

        # Core data storage
        self.incidents: dict[str, Incident] = {}
        self.remediation_actions: dict[str, RemediationAction] = {}
        self.playbooks: dict[str, Playbook] = {}
        self.executions: dict[str, RemediationExecution] = {}

        # Execution tracking
        self.active_executions: set[str] = set()
        self.execution_history: deque = deque(maxlen=1000)
        self.rate_limits: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

        # Event callbacks
        self.incident_callbacks: list[Callable] = []
        self.remediation_callbacks: list[Callable] = []

        # Configuration
        self.config = {
            "auto_remediation_enabled": True,
            "max_concurrent_actions": 5,
            "default_timeout_seconds": 300,
            "notification_channels": ["slack", "email", "pagerduty"],
            "require_approval_for_critical": True,
        }

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []

        # Initialize default remediation actions and playbooks
        self._initialize_default_actions()
        self._initialize_default_playbooks()

        # Start background monitoring
        self._start_background_tasks()

    def _initialize_default_actions(self) -> None:
        """Initialize default remediation actions."""
        default_actions = [
            # Service restart actions
            RemediationAction(
                name="restart_ai_service",
                description="Restart AI inference service",
                remediation_type=RemediationType.RESTART_SERVICE,
                command="kubectl rollout restart deployment/ai-inference",
                timeout_seconds=180,
                validation_command="kubectl get pods -l app=ai-inference | grep Running | wc -l",
                prerequisites=["kubernetes_accessible"],
                side_effects=["temporary_service_disruption"],
                estimated_impact="medium",
                max_executions_per_hour=2,
                cooldown_minutes=30,
            ),
            # Scaling actions
            RemediationAction(
                name="scale_up_ai_service",
                description="Scale up AI service replicas",
                remediation_type=RemediationType.SCALE_UP,
                command="kubectl scale deployment/ai-inference --replicas=5",
                timeout_seconds=300,
                rollback_command="kubectl scale deployment/ai-inference --replicas=3",
                validation_command="kubectl get deployment ai-inference -o jsonpath='{.status.readyReplicas}'",
                prerequisites=["resource_availability"],
                side_effects=["increased_resource_usage", "higher_costs"],
                estimated_impact="low",
            ),
            # Cache management
            RemediationAction(
                name="clear_redis_cache",
                description="Clear Redis cache to resolve stale data issues",
                remediation_type=RemediationType.CACHE_CLEAR,
                command="redis-cli FLUSHALL",
                timeout_seconds=30,
                validation_command="redis-cli INFO keyspace",
                prerequisites=["redis_accessible"],
                side_effects=["cache_miss_spike", "temporary_performance_degradation"],
                estimated_impact="medium",
                max_executions_per_hour=3,
            ),
            # Circuit breaker
            RemediationAction(
                name="enable_circuit_breaker",
                description="Enable circuit breaker for AI service",
                remediation_type=RemediationType.CIRCUIT_BREAKER,
                command="curl -X POST http://ai-service:8080/admin/circuit-breaker/enable",
                timeout_seconds=10,
                rollback_command="curl -X POST http://ai-service:8080/admin/circuit-breaker/disable",
                validation_command="curl http://ai-service:8080/admin/circuit-breaker/status",
                prerequisites=["service_reachable"],
                side_effects=["reduced_functionality", "fallback_responses"],
                estimated_impact="medium",
            ),
            # Connection pool reset
            RemediationAction(
                name="reset_connection_pool",
                description="Reset database connection pool",
                remediation_type=RemediationType.CONNECTION_POOL_RESET,
                command="curl -X POST http://ai-service:8080/admin/db/reset-pool",
                timeout_seconds=30,
                validation_command="curl http://ai-service:8080/health/db",
                prerequisites=["service_reachable"],
                side_effects=["temporary_connection_drops"],
                estimated_impact="low",
            ),
            # Traffic redirection
            RemediationAction(
                name="redirect_to_fallback",
                description="Redirect traffic to fallback service",
                remediation_type=RemediationType.TRAFFIC_REDIRECT,
                command='kubectl patch service ai-service -p \'{"spec":{"selector":{"app":"ai-fallback"}}}\'',
                timeout_seconds=60,
                rollback_command='kubectl patch service ai-service -p \'{"spec":{"selector":{"app":"ai-inference"}}}\'',
                validation_command="kubectl get service ai-service -o jsonpath='{.spec.selector.app}'",
                requires_confirmation=True,
                prerequisites=["fallback_service_healthy"],
                side_effects=["reduced_performance", "limited_functionality"],
                estimated_impact="high",
            ),
            # Rate limiting adjustment
            RemediationAction(
                name="increase_rate_limits",
                description="Temporarily increase rate limits",
                remediation_type=RemediationType.RATE_LIMIT_ADJUST,
                command='kubectl patch configmap rate-limits -p \'{"data":{"max_requests_per_minute":"200"}}\'',
                timeout_seconds=30,
                rollback_command='kubectl patch configmap rate-limits -p \'{"data":{"max_requests_per_minute":"100"}}\'',
                validation_command="kubectl get configmap rate-limits -o jsonpath='{.data.max_requests_per_minute}'",
                prerequisites=["configmap_exists"],
                side_effects=["increased_load", "potential_downstream_impact"],
                estimated_impact="medium",
                cooldown_minutes=60,
            ),
        ]

        for action in default_actions:
            self.register_remediation_action(action)

    def _initialize_default_playbooks(self) -> None:
        """Initialize default incident response playbooks."""
        default_playbooks = [
            # High latency playbook
            Playbook(
                name="high_latency_response",
                description="Response to high AI inference latency",
                triggers=["ai_response_time_high", "ai_latency_p95_exceeded"],
                applicable_severities=[IncidentSeverity.MEDIUM, IncidentSeverity.HIGH],
                remediation_steps=[
                    "clear_redis_cache",
                    "reset_connection_pool",
                    "scale_up_ai_service",
                ],
                escalation_rules={
                    "escalate_after_minutes": 15,
                    "escalate_to": "ai-platform-oncall",
                    "auto_escalate": True,
                },
                auto_execute=True,
                max_auto_actions=2,
                owner_team="ai-platform",
            ),
            # Service unavailable playbook
            Playbook(
                name="service_unavailable_response",
                description="Response to AI service unavailability",
                triggers=["ai_service_down", "high_error_rate"],
                applicable_severities=[IncidentSeverity.HIGH, IncidentSeverity.CRITICAL],
                remediation_steps=[
                    "restart_ai_service",
                    "enable_circuit_breaker",
                    "redirect_to_fallback",
                ],
                escalation_rules={
                    "escalate_after_minutes": 5,
                    "escalate_to": "ai-platform-lead",
                    "auto_escalate": True,
                    "notify_stakeholders": True,
                },
                auto_execute=True,
                max_auto_actions=1,
                require_confirmation_after=1,
                owner_team="ai-platform",
            ),
            # Resource exhaustion playbook
            Playbook(
                name="resource_exhaustion_response",
                description="Response to resource exhaustion issues",
                triggers=["high_memory_usage", "high_cpu_usage", "disk_space_low"],
                applicable_severities=[IncidentSeverity.MEDIUM, IncidentSeverity.HIGH],
                remediation_steps=[
                    "clear_redis_cache",
                    "scale_up_ai_service",
                    "enable_circuit_breaker",
                ],
                escalation_rules={
                    "escalate_after_minutes": 20,
                    "escalate_to": "infrastructure-team",
                    "auto_escalate": False,
                },
                auto_execute=True,
                max_auto_actions=3,
                owner_team="infrastructure",
            ),
        ]

        for playbook in default_playbooks:
            self.register_playbook(playbook)

    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        self._background_tasks = [
            asyncio.create_task(self._monitor_incidents()),
            asyncio.create_task(self._cleanup_old_data()),
            asyncio.create_task(self._update_playbook_metrics()),
        ]

    def register_remediation_action(self, action: RemediationAction) -> None:
        """Register a remediation action."""
        self.remediation_actions[action.name] = action
        logger.info(f"Registered remediation action: {action.name}")

    def register_playbook(self, playbook: Playbook) -> None:
        """Register an incident response playbook."""
        self.playbooks[playbook.name] = playbook
        logger.info(f"Registered playbook: {playbook.name}")

    def add_incident_callback(self, callback: Callable) -> None:
        """Add callback for incident events."""
        self.incident_callbacks.append(callback)

    def add_remediation_callback(self, callback: Callable) -> None:
        """Add callback for remediation events."""
        self.remediation_callbacks.append(callback)

    async def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        source: str = "monitoring",
        affected_services: list[str] | None = None,
        alerts: list[dict[str, Any]] | None = None,
        auto_respond: bool = True,
    ) -> str:
        """Create a new incident."""
        incident_id = str(uuid.uuid4())

        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.DETECTED,
            detected_at=datetime.now(UTC),
            source=source,
            affected_services=affected_services or [],
            alerts=alerts or [],
        )

        self.incidents[incident_id] = incident

        logger.warning(f"Incident created: {incident_id} - {title} ({severity.value})")

        # Notify callbacks
        for callback in self.incident_callbacks:
            try:
                await callback("incident_created", incident)
            except Exception as e:
                logger.exception(f"Error in incident callback: {e}")

        # Auto-respond if enabled
        if auto_respond and self.config["auto_remediation_enabled"]:
            await self._auto_respond_to_incident(incident_id)

        return incident_id

    async def _auto_respond_to_incident(self, incident_id: str) -> None:
        """Automatically respond to an incident using playbooks."""
        if incident_id not in self.incidents:
            return

        incident = self.incidents[incident_id]

        # Find applicable playbooks
        applicable_playbooks = self._find_applicable_playbooks(incident)

        if not applicable_playbooks:
            logger.info(f"No applicable playbooks found for incident {incident_id}")
            return

        # Use the first applicable playbook
        playbook = applicable_playbooks[0]
        logger.info(f"Applying playbook '{playbook.name}' to incident {incident_id}")

        # Execute remediation steps
        if playbook.auto_execute:
            await self._execute_playbook(incident_id, playbook.name)

    def _find_applicable_playbooks(self, incident: Incident) -> list[Playbook]:
        """Find playbooks applicable to an incident."""
        applicable: list[Any] = []

        for playbook in self.playbooks.values():
            # Check severity match
            if incident.severity not in playbook.applicable_severities:
                continue

            # Check triggers (simplified matching)
            trigger_match = any(
                trigger.lower() in incident.description.lower()
                or trigger.lower() in incident.title.lower()
                for trigger in playbook.triggers
            )

            if trigger_match:
                applicable.append(playbook)

        # Sort by success rate (higher first)
        applicable.sort(key=lambda p: p.success_rate, reverse=True)
        return applicable

    async def _execute_playbook(self, incident_id: str, playbook_name: str) -> None:
        """Execute a playbook for an incident."""
        if playbook_name not in self.playbooks:
            logger.error(f"Playbook {playbook_name} not found")
            return

        playbook = self.playbooks[playbook_name]
        incident = self.incidents[incident_id]

        # Update incident status
        incident.status = IncidentStatus.MITIGATING
        incident.automatic_remediation_attempted = True

        executed_actions = 0
        for action_name in playbook.remediation_steps:
            if executed_actions >= playbook.max_auto_actions:
                logger.info(
                    f"Reached max auto actions ({playbook.max_auto_actions}) for playbook {playbook_name}"
                )
                break

            # Check if confirmation required
            if (
                executed_actions >= playbook.require_confirmation_after
                and not self._has_human_approval(
                    incident_id,
                    action_name,
                )
            ):
                logger.info(f"Human approval required for action {action_name}")
                incident.manual_intervention_required = True
                break

            # Execute remediation action
            success = await self.execute_remediation_action(incident_id, action_name)

            if success:
                executed_actions += 1
                # Check if incident is resolved
                if await self._check_incident_resolved(incident_id):
                    await self.resolve_incident(
                        incident_id, "Automatically resolved by remediation"
                    )
                    break
            else:
                logger.warning(
                    f"Remediation action {action_name} failed for incident {incident_id}"
                )
                # Continue with next action unless it's critical
                if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.EMERGENCY]:
                    incident.manual_intervention_required = True
                    break

    async def execute_remediation_action(
        self,
        incident_id: str,
        action_name: str,
        force: bool = False,
    ) -> bool:
        """Execute a specific remediation action."""
        if action_name not in self.remediation_actions:
            logger.error(f"Remediation action {action_name} not found")
            return False

        action = self.remediation_actions[action_name]

        # Check rate limits
        if not force and not self._check_rate_limits(action):
            logger.warning(f"Rate limit exceeded for action {action_name}")
            return False

        # Check prerequisites
        if not await self._check_prerequisites(action):
            logger.warning(f"Prerequisites not met for action {action_name}")
            return False

        # Create execution record
        execution_id = str(uuid.uuid4())
        execution = RemediationExecution(
            execution_id=execution_id,
            action_name=action_name,
            incident_id=incident_id,
            status=RemediationStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

        self.executions[execution_id] = execution
        self.active_executions.add(execution_id)

        # Update incident
        if incident_id in self.incidents:
            self.incidents[incident_id].remediation_actions.append(execution_id)

        logger.info(f"Executing remediation action: {action_name} for incident {incident_id}")

        try:
            # Execute the command
            success = await self._execute_command(execution, action)

            if success:
                execution.status = RemediationStatus.SUCCESS

                # Run validation if specified
                if action.validation_command:
                    validation_success = await self._validate_remediation(execution, action)
                    execution.validation_passed = validation_success

                    if not validation_success:
                        logger.warning(f"Validation failed for action {action_name}")
                        # Consider rollback
                        if action.rollback_command:
                            await self._rollback_remediation(execution, action)
                        success = False
            else:
                execution.status = RemediationStatus.FAILED

            execution.completed_at = datetime.now(UTC)

            # Update rate limits
            self.rate_limits[action_name].append(datetime.now(UTC))

            # Notify callbacks
            for callback in self.remediation_callbacks:
                try:
                    await callback("remediation_completed", execution)
                except Exception as e:
                    logger.exception(f"Error in remediation callback: {e}")

            return success

        except Exception as e:
            execution.status = RemediationStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now(UTC)
            logger.exception(f"Error executing remediation action {action_name}: {e}")
            return False

        finally:
            self.active_executions.discard(execution_id)
            self.execution_history.append(execution)

    async def _execute_command(
        self, execution: RemediationExecution, action: RemediationAction
    ) -> bool:
        """Execute a remediation command."""
        try:
            # Use asyncio subprocess for non-blocking execution
            process = await asyncio.create_subprocess_shell(
                action.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=action.timeout_seconds
                )

                execution.exit_code = process.returncode
                execution.stdout = stdout.decode() if stdout else ""
                execution.stderr = stderr.decode() if stderr else ""

                return process.returncode == 0

            except TimeoutError:
                execution.status = RemediationStatus.TIMEOUT
                execution.error_message = (
                    f"Command timed out after {action.timeout_seconds} seconds"
                )

                # Kill the process
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass

                return False

        except Exception as e:
            execution.error_message = str(e)
            return False

    async def _validate_remediation(
        self, execution: RemediationExecution, action: RemediationAction
    ) -> bool:
        """Validate that a remediation action was successful."""
        if not action.validation_command:
            return True

        try:
            process = await asyncio.create_subprocess_shell(
                action.validation_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30,  # Short timeout for validation
            )

            return process.returncode == 0

        except Exception as e:
            logger.exception(f"Validation failed for {action.name}: {e}")
            return False

    async def _rollback_remediation(
        self, execution: RemediationExecution, action: RemediationAction
    ) -> None:
        """Rollback a remediation action."""
        if not action.rollback_command:
            return

        logger.info(f"Rolling back remediation action: {action.name}")

        try:
            process = await asyncio.create_subprocess_shell(
                action.rollback_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await asyncio.wait_for(process.communicate(), timeout=60)
            execution.rollback_performed = True

        except Exception as e:
            logger.exception(f"Rollback failed for {action.name}: {e}")

    def _check_rate_limits(self, action: RemediationAction) -> bool:
        """Check if action is within rate limits."""
        now = datetime.now(UTC)
        recent_executions = [
            exec_time
            for exec_time in self.rate_limits[action.name]
            if now - exec_time <= timedelta(hours=1)
        ]

        return len(recent_executions) < action.max_executions_per_hour

    async def _check_prerequisites(self, action: RemediationAction) -> bool:
        """Check if action prerequisites are met."""
        # Simplified prerequisite checking
        # In a real implementation, this would check actual system state
        return True

    def _has_human_approval(self, incident_id: str, action_name: str) -> bool:
        """Check if human has approved this action."""
        # Placeholder for approval system
        # In a real implementation, this would check approval status
        return False

    async def _check_incident_resolved(self, incident_id: str) -> bool:
        """Check if incident appears to be resolved."""
        # Simplified resolution checking
        # In a real implementation, this would check metrics and alerts
        incident = self.incidents[incident_id]

        # Check if recent remediation actions were successful
        recent_executions = [
            exec_id
            for exec_id in incident.remediation_actions[-3:]  # Last 3 actions
            if exec_id in self.executions
            and self.executions[exec_id].status == RemediationStatus.SUCCESS
        ]

        # Consider resolved if we had successful recent actions
        return len(recent_executions) >= 1

    async def resolve_incident(self, incident_id: str, resolution_note: str = "") -> None:
        """Mark an incident as resolved."""
        if incident_id not in self.incidents:
            return

        incident = self.incidents[incident_id]
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now(UTC)
        incident.root_cause = resolution_note

        logger.info(f"Incident resolved: {incident_id} - {incident.title}")

        # Notify callbacks
        for callback in self.incident_callbacks:
            try:
                await callback("incident_resolved", incident)
            except Exception as e:
                logger.exception(f"Error in incident callback: {e}")

    async def _monitor_incidents(self) -> None:
        """Monitor active incidents for escalation and timeout."""
        while True:
            try:
                datetime.now(UTC)

                for incident in self.incidents.values():
                    if incident.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                        continue

                    # Check for escalation
                    if (
                        incident.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]
                        and not incident.escalated_to
                        and incident.duration_minutes > 15
                    ):
                        await self._escalate_incident(incident.incident_id)

                    # Check for stale incidents
                    if (
                        incident.status == IncidentStatus.DETECTED
                        and incident.duration_minutes > 30
                    ):
                        logger.warning(f"Stale incident detected: {incident.incident_id}")
                        incident.manual_intervention_required = True

                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error monitoring incidents: {e}")
                await asyncio.sleep(60)

    async def _escalate_incident(self, incident_id: str) -> None:
        """Escalate an incident."""
        if incident_id not in self.incidents:
            return

        incident = self.incidents[incident_id]
        incident.escalated_to = "oncall-engineer"

        logger.warning(f"Incident escalated: {incident_id}")

        # In a real implementation, this would send notifications

    async def _cleanup_old_data(self) -> None:
        """Clean up old incident and execution data."""
        while True:
            try:
                cutoff_date = datetime.now(UTC) - timedelta(days=30)

                # Clean up old closed incidents
                old_incidents = [
                    incident_id
                    for incident_id, incident in self.incidents.items()
                    if incident.status == IncidentStatus.CLOSED
                    and incident.closed_at
                    and incident.closed_at < cutoff_date
                ]

                for incident_id in old_incidents:
                    del self.incidents[incident_id]

                # Clean up old executions
                old_executions = [
                    exec_id
                    for exec_id, execution in self.executions.items()
                    if execution.completed_at and execution.completed_at < cutoff_date
                ]

                for exec_id in old_executions:
                    del self.executions[exec_id]

                if old_incidents or old_executions:
                    logger.info(
                        f"Cleaned up {len(old_incidents)} incidents and {len(old_executions)} executions"
                    )

                await asyncio.sleep(3600)  # Clean up every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error cleaning up old data: {e}")
                await asyncio.sleep(3600)

    async def _update_playbook_metrics(self) -> None:
        """Update playbook success metrics."""
        while True:
            try:
                for playbook_name, playbook in self.playbooks.items():
                    # Calculate success rate based on recent incidents
                    recent_incidents = [
                        incident
                        for incident in self.incidents.values()
                        if incident.detected_at >= datetime.now(UTC) - timedelta(days=7)
                    ]

                    # Find incidents where this playbook was used
                    playbook_incidents = [
                        incident
                        for incident in recent_incidents
                        if any(
                            playbook_name.lower() in action.lower()
                            for action in incident.remediation_actions
                        )
                    ]

                    if playbook_incidents:
                        resolved_count = len(
                            [
                                incident
                                for incident in playbook_incidents
                                if incident.status == IncidentStatus.RESOLVED
                            ],
                        )

                        playbook.success_rate = resolved_count / len(playbook_incidents)

                        # Calculate average resolution time
                        resolved_incidents = [
                            incident
                            for incident in playbook_incidents
                            if incident.time_to_resolve_minutes is not None
                        ]

                        if resolved_incidents:
                            avg_time = sum(
                                incident.time_to_resolve_minutes for incident in resolved_incidents
                            ) / len(
                                resolved_incidents,
                            )
                            playbook.avg_resolution_time_minutes = avg_time

                await asyncio.sleep(3600)  # Update every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error updating playbook metrics: {e}")
                await asyncio.sleep(3600)

    def get_incident_summary(self) -> dict[str, Any]:
        """Get summary of incident statistics."""
        now = datetime.now(UTC)

        # Count incidents by status
        by_status = defaultdict(int)
        by_severity = defaultdict(int)

        active_incidents: list[Any] = []
        resolved_today: list[Any] = []

        for incident in self.incidents.values():
            by_status[incident.status.value] += 1
            by_severity[incident.severity.value] += 1

            if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                active_incidents.append(incident)

            if incident.resolved_at and incident.resolved_at >= now - timedelta(days=1):
                resolved_today.append(incident)

        # Calculate MTTR (Mean Time To Resolution)
        mttr_minutes = 0.0
        if resolved_today:
            total_resolution_time = sum(
                incident.time_to_resolve_minutes or 0 for incident in resolved_today
            )
            mttr_minutes = total_resolution_time / len(resolved_today)

        return {
            "summary": {
                "total_incidents": len(self.incidents),
                "active_incidents": len(active_incidents),
                "resolved_today": len(resolved_today),
                "mttr_minutes": mttr_minutes,
            },
            "by_status": dict(by_status),
            "by_severity": dict(by_severity),
            "active_incidents": [
                {
                    "incident_id": incident.incident_id,
                    "title": incident.title,
                    "severity": incident.severity.value,
                    "duration_minutes": incident.duration_minutes,
                    "status": incident.status.value,
                }
                for incident in active_incidents[:10]  # Latest 10
            ],
            "top_affected_services": self._get_top_affected_services(),
        }

    def _get_top_affected_services(self) -> list[dict[str, Any]]:
        """Get services most affected by incidents."""
        service_incidents = defaultdict(int)

        for incident in self.incidents.values():
            for service in incident.affected_services:
                service_incidents[service] += 1

        return [
            {"service": service, "incident_count": count}
            for service, count in sorted(
                service_incidents.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]

    def get_remediation_summary(self) -> dict[str, Any]:
        """Get summary of remediation action statistics."""
        now = datetime.now(UTC)

        # Recent executions (last 24 hours)
        recent_executions = [
            execution
            for execution in self.executions.values()
            if execution.started_at >= now - timedelta(days=1)
        ]

        # Count by status
        by_status = defaultdict(int)
        by_action = defaultdict(int)

        total_duration = 0.0
        success_count = 0

        for execution in recent_executions:
            by_status[execution.status.value] += 1
            by_action[execution.action_name] += 1

            if execution.completed_at:
                total_duration += execution.duration_seconds

                if execution.status == RemediationStatus.SUCCESS:
                    success_count += 1

        success_rate = (success_count / len(recent_executions) * 100) if recent_executions else 0.0
        avg_duration = total_duration / len(recent_executions) if recent_executions else 0.0

        return {
            "summary": {
                "total_executions_24h": len(recent_executions),
                "success_rate": success_rate,
                "avg_duration_seconds": avg_duration,
                "active_executions": len(self.active_executions),
            },
            "by_status": dict(by_status),
            "most_used_actions": [
                {"action": action, "count": count}
                for action, count in sorted(by_action.items(), key=lambda x: x[1], reverse=True)[:5]
            ],
            "playbook_performance": {
                playbook.name: {
                    "success_rate": playbook.success_rate,
                    "avg_resolution_time_minutes": playbook.avg_resolution_time_minutes,
                }
                for playbook in self.playbooks.values()
            },
        }

    async def execute_playbook(
        self,
        playbook_name: str,
        context: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        """Execute a playbook directly with optional context.

        Args:
            playbook_name: Name of the playbook to execute
            context: Optional context for the playbook execution
            force: Whether to force execution bypassing rate limits

        Returns:
            bool: True if playbook execution was successful, False otherwise

        """
        if playbook_name not in self.playbooks:
            logger.error(f"Playbook {playbook_name} not found")
            return False

        self.playbooks[playbook_name]

        # Create a temporary incident for standalone playbook execution
        temp_incident_id = f"playbook-execution-{int(datetime.now(UTC).timestamp())}"
        temp_incident = Incident(
            incident_id=temp_incident_id,
            title=f"Standalone Playbook Execution: {playbook_name}",
            description=f"Direct execution of playbook {playbook_name}",
            severity=IncidentSeverity.MEDIUM,
            status=IncidentStatus.INVESTIGATING,
            source="manual_playbook_execution",
            detected_at=datetime.now(UTC),
            context=context or {},
        )

        # Store temporary incident
        self.incidents[temp_incident_id] = temp_incident

        try:
            # Execute the playbook
            await self._execute_playbook(temp_incident_id, playbook_name)

            # Check if any actions were successful
            successful_actions = 0
            for execution_id in temp_incident.remediation_actions:
                if execution_id in self.executions:
                    execution = self.executions[execution_id]
                    if execution.status == RemediationStatus.SUCCESS:
                        successful_actions += 1

            success = successful_actions > 0

            if success:
                logger.info(
                    f"Playbook {playbook_name} executed successfully ({successful_actions} successful actions)"
                )
            else:
                logger.warning(
                    f"Playbook {playbook_name} execution completed with no successful actions"
                )

            return success

        except Exception as e:
            logger.exception(f"Error executing playbook {playbook_name}: {e}")
            return False

        finally:
            # Clean up temporary incident
            if temp_incident_id in self.incidents:
                del self.incidents[temp_incident_id]

    async def close(self) -> None:
        """Clean shutdown of incident response system."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("Incident response system shutdown complete")


# Export main classes
__all__ = [
    "Incident",
    "IncidentResponseSystem",
    "IncidentSeverity",
    "IncidentStatus",
    "Playbook",
    "RemediationAction",
    "RemediationExecution",
    "RemediationStatus",
    "RemediationType",
]
