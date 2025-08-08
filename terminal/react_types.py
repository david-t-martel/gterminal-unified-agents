from typing import Union

# FIXME: Unused import 'Union' - remove if not needed
"""
Type definitions for the ReAct engine.

This module provides the core type definitions used throughout the ReAct engine
for reasoning, acting, and observing in AI agent workflows.
"""

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class StepType(Enum):
    """Enumeration of ReAct step types."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class ActionType(Enum):
    """Enumeration of available action types."""

    ANALYZE_CODE = "analyze_code"
    GENERATE_CODE = "generate_code"
    REVIEW_CODE = "review_code"
    WORKSPACE_ANALYZE = "workspace_analyze"
    FILE_OPERATION = "file_operation"
    SYSTEM_COMMAND = "system_command"
    SEARCH = "search"
    DOCUMENTATION = "documentation"
    ARCHITECT = "architect"
    CUSTOM = "custom"


class ReActStatus(Enum):
    """Enumeration of ReAct execution statuses."""

    INITIALIZED = "initialized"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class Action:
    """Represents an action to be executed in the ReAct loop."""

    action_type: ActionType
    parameters: dict[str, Any]
    description: str
    agent_name: str | None = None
    timeout: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Represents an observation from an executed action."""

    action_id: str
    result: Any
    success: bool
    error: str | None = None
    execution_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReActStep:
    """Represents a single step in the ReAct process."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: StepType = StepType.THOUGHT
    content: str = ""
    action: Action | None = None
    observation: Observation | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReActContext:
    """Context for the ReAct execution session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: str = ""
    max_iterations: int = 10
    current_iteration: int = 0
    status: ReActStatus = ReActStatus.INITIALIZED
    steps: list[ReActStep] = field(default_factory=list)
    context_data: dict[str, Any] = field(default_factory=dict)
    agent_preferences: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_step(self, step: ReActStep) -> None:
        """Add a step to the context and update timestamp."""
        self.steps.append(step)
        self.updated_at = datetime.now()

    def get_recent_steps(self, count: int = 5) -> list[ReActStep]:
        """Get the most recent steps for context."""
        return self.steps[-count:] if len(self.steps) > count else self.steps

    def get_step_history(self) -> str:
        """Get a formatted string of the step history."""
        history = []
        for step in self.steps:
            if step.step_type == StepType.THOUGHT:
                history.append(f"Thought: {step.content}")
            elif step.step_type == StepType.ACTION:
                action_desc = step.action.description if step.action else "Unknown action"
                history.append(f"Action: {action_desc}")
            elif step.step_type == StepType.OBSERVATION:
                obs_result = step.observation.result if step.observation else "No result"
                history.append(f"Observation: {obs_result}")
            elif step.step_type == StepType.FINAL_ANSWER:
                history.append(f"Final Answer: {step.content}")
        return "\n".join(history)


@dataclass
class ReActResult:
    """Result of a ReAct execution session."""

    context: ReActContext
    final_answer: str | None = None
    success: bool = False
    error: str | None = None
    total_steps: int = 0
    execution_time: float | None = None
    agent_stats: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def completed_successfully(self) -> bool:
        """Check if the ReAct execution completed successfully."""
        return self.success and self.context.status == ReActStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "session_id": self.context.session_id,
            "task": self.context.task,
            "final_answer": self.final_answer,
            "success": self.success,
            "error": self.error,
            "total_steps": self.total_steps,
            "execution_time": self.execution_time,
            "status": self.context.status.value,
            "step_count": len(self.context.steps),
            "agent_stats": self.agent_stats,
            "performance_metrics": self.performance_metrics,
            "created_at": self.context.created_at.isoformat(),
            "completed_at": self.context.updated_at.isoformat(),
        }


@dataclass
class AgentCall:
    """Represents a call to a specific agent with parameters."""

    agent_name: str
    method: str
    parameters: dict[str, Any]
    expected_output: str | None = None
    timeout: int | None = None

    def to_action(self) -> Action:
        """Convert agent call to an Action."""
        return Action(
            action_type=ActionType.CUSTOM,
            parameters={
                "agent_name": self.agent_name,
                "method": self.method,
                "parameters": self.parameters,
                "timeout": self.timeout,
            },
            description=f"Call {self.agent_name}.{self.method}",
            agent_name=self.agent_name,
            timeout=self.timeout,
        )


# Type aliases for better readability
ReActStepList = list[ReActStep]
ActionResult = Union[str, dict[str, Any], list[Any]]
ContextData = dict[str, Any]
AgentStats = dict[str, Any]
PerformanceMetrics = dict[str, Any]
