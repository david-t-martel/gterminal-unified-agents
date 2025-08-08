#!/usr/bin/env python3
"""Autonomous ReAct Engine - Phase 1 Implementation.

This module extends the base ReAct engine with autonomous capabilities:
1. Goal decomposition and multi-step planning
2. Persistent context across sessions
3. Learning from successful patterns
4. Enhanced error recovery and plan adjustment
5. Integration with the unified Gemini client

Key Features:
- Autonomous goal analysis and task breakdown
- Multi-step execution with dependency tracking
- Context persistence and session restoration
- Pattern learning and similarity matching
- Streaming progress reporting
- Comprehensive error handling and recovery
"""

import asyncio
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import Field

# Import enhanced utilities
from gterminal.core.json_utils import extract_json_from_llm_response
from gterminal.core.json_utils import validate_analysis_data
from gterminal.core.json_utils import validate_plan_data
from gterminal.core.parameter_validator import ParameterValidationError
from gterminal.core.parameter_validator import validate_tool_parameters

# Import base ReAct components
from gterminal.core.react_engine import Plan
from gterminal.core.react_engine import ReactEngine
from gterminal.core.react_engine import ReactResponse
from gterminal.core.react_engine import Step
from gterminal.core.react_engine import StepType
from gterminal.core.react_engine import ToolResult
from gterminal.core.unified_gemini_client import get_gemini_client

logger = logging.getLogger(__name__)


class AutonomyLevel(str, Enum):
    """Levels of autonomous operation."""

    MANUAL = "manual"  # User confirms each step
    GUIDED = "guided"  # User confirms major steps
    SEMI_AUTO = "semi_auto"  # User confirms critical steps
    FULLY_AUTO = "fully_auto"  # Full autonomous operation


class TaskComplexity(str, Enum):
    """Task complexity levels."""

    SIMPLE = "simple"  # Single-step tasks
    MODERATE = "moderate"  # 2-5 steps
    COMPLEX = "complex"  # 6-15 steps
    ADVANCED = "advanced"  # 16+ steps or dependencies


class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Dependency:
    """Represents a dependency between tasks/steps."""

    source_step_id: str
    target_step_id: str
    dependency_type: str = "completion"  # completion, data, resource
    required_data: dict[str, Any] | None = None


@dataclass
class ContextSnapshot:
    """Snapshot of execution context for persistence."""

    session_id: str
    timestamp: datetime
    current_plan: dict[str, Any] | None = None
    completed_steps: list[dict[str, Any]] = field(default_factory=list)
    active_goals: list[str] = field(default_factory=list)
    learned_patterns: list[dict[str, Any]] = field(default_factory=list)
    success_metrics: dict[str, float] = field(default_factory=dict)


class AutonomousStep(Step):
    """Enhanced step with autonomous capabilities."""

    step_id: str = Field(
        default_factory=lambda: f"step_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    dependencies: list[str] = Field(
        default_factory=list,
    )  # Changed from List[Dependency] to List[str] for JSON compatibility
    estimated_duration: float | None = Field(default=None)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    success_probability: float | None = Field(default=None)
    learned_from: str | None = Field(default=None)  # Pattern ID this step was learned from

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs) -> None:
        super().__pydantic_init_subclass__(**kwargs)

    def __init__(self, **data) -> None:
        # Handle string dependencies conversion
        if "dependencies" in data and isinstance(data["dependencies"], str):
            # Split comma-separated dependencies
            if data["dependencies"].strip():
                data["dependencies"] = [dep.strip() for dep in data["dependencies"].split(",")]
            else:
                data["dependencies"] = []

        # Handle string priority conversion
        if "priority" in data and isinstance(data["priority"], str):
            priority_map = {
                "low": TaskPriority.LOW,
                "medium": TaskPriority.MEDIUM,
                "high": TaskPriority.HIGH,
                "critical": TaskPriority.CRITICAL,
            }
            data["priority"] = priority_map.get(data["priority"].lower(), TaskPriority.MEDIUM)

        # Handle string type conversion
        if "type" in data and isinstance(data["type"], str):
            try:
                data["type"] = StepType(data["type"])
            except ValueError:
                data["type"] = StepType.ACT  # Default fallback

        super().__init__(**data)


class AutonomousPlan(Plan):
    """Enhanced plan with autonomous capabilities."""

    plan_id: str = Field(
        default_factory=lambda: f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    complexity: TaskComplexity = Field(default=TaskComplexity.SIMPLE)
    autonomy_level: AutonomyLevel = Field(default=AutonomyLevel.SEMI_AUTO)
    estimated_total_time: float | None = Field(default=None)
    success_probability: float = Field(default=0.8)
    fallback_plans: list[dict[str, Any]] = Field(default_factory=list)
    dependencies: list[str] = Field(
        default_factory=list,
    )  # Changed from List[Dependency] to List[str] for JSON compatibility
    checkpoints: list[str] = Field(default_factory=list)  # Step IDs for checkpoints

    # Progress tracking
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    progress_percentage: float = Field(default=0.0)
    current_step_index: int = Field(default=0)

    def __init__(self, **data) -> None:
        # Handle checkpoints conversion from integers to strings
        if "checkpoints" in data and isinstance(data["checkpoints"], list):
            data["checkpoints"] = [str(checkpoint) for checkpoint in data["checkpoints"]]

        # Handle dependencies conversion from integers to strings
        if "dependencies" in data and isinstance(data["dependencies"], list):
            data["dependencies"] = [str(dep) for dep in data["dependencies"]]
        elif "dependencies" in data and isinstance(data["dependencies"], str):
            # Split comma-separated dependencies
            if data["dependencies"].strip():
                data["dependencies"] = [dep.strip() for dep in data["dependencies"].split(",")]
            else:
                data["dependencies"] = []

        # Handle enum conversions
        if "complexity" in data and isinstance(data["complexity"], str):
            try:
                data["complexity"] = TaskComplexity(data["complexity"])
            except ValueError:
                data["complexity"] = TaskComplexity.SIMPLE

        if "autonomy_level" in data and isinstance(data["autonomy_level"], str):
            try:
                data["autonomy_level"] = AutonomyLevel(data["autonomy_level"])
            except ValueError:
                data["autonomy_level"] = AutonomyLevel.SEMI_AUTO

        super().__init__(**data)


class AutonomousReactEngine(ReactEngine):
    """Enhanced ReAct engine with autonomous capabilities.

    This engine extends the base ReAct functionality with:
    - Autonomous goal decomposition
    - Multi-step planning with dependencies
    - Context persistence and learning
    - Enhanced error recovery
    - Progress tracking and reporting
    """

    def __init__(
        self,
        profile: str = "business",
        model_name: str | None = None,
        autonomy_level: AutonomyLevel = AutonomyLevel.SEMI_AUTO,
        project_root: Path | None = None,
    ) -> None:
        """Initialize autonomous ReAct engine.

        Args:
            profile: GCP profile to use ('business' or 'personal')
            model_name: Optional model name override
            autonomy_level: Level of autonomous operation
            project_root: Root directory for project operations

        """
        # Initialize unified Gemini client
        self.gemini_client = get_gemini_client(profile=profile, model_name=model_name)

        # Get the model for parent initialization
        try:
            model = self.gemini_client.get_model()
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini model: {e}")
            model = None

        # Initialize parent with the model
        super().__init__(model, project_root)

        # Autonomous engine configuration
        self.autonomy_level = autonomy_level
        self.profile = profile

        # Context persistence
        self.context_dir = self.project_root / ".react_context"
        self.context_dir.mkdir(exist_ok=True)

        # Pattern learning storage
        self.patterns_file = self.context_dir / "learned_patterns.json"
        self.learned_patterns: dict[str, dict[str, Any]] = {}
        self._load_learned_patterns()

        # Session management
        self.active_sessions: dict[str, ContextSnapshot] = {}

        # Performance tracking
        self.success_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_completion_time": 0.0,
            "pattern_matches": 0,
            "autonomous_completions": 0,
        }

        logger.info("✅ Autonomous ReAct Engine initialized")
        logger.info(f"   Profile: {profile}")
        logger.info(f"   Autonomy Level: {autonomy_level.value}")
        logger.info(f"   Context Directory: {self.context_dir}")

    async def process_autonomous_request(
        self,
        request: str,
        session_id: str | None = None,
        autonomy_level: AutonomyLevel | None = None,
        streaming: bool = False,
        user_confirmation_callback: callable | None = None,
    ) -> ReactResponse:
        """Process a request with autonomous capabilities.

        Args:
            request: Natural language request
            session_id: Session ID for context persistence
            autonomy_level: Override autonomy level for this request
            streaming: Whether to stream progress updates
            user_confirmation_callback: Callback for user confirmations

        Returns:
            ReactResponse with autonomous execution results

        """
        start_time = datetime.now()
        session_id = session_id or f"auto_{start_time.strftime('%Y%m%d_%H%M%S')}"
        autonomy = autonomy_level or self.autonomy_level

        self.success_metrics["total_requests"] += 1

        logger.info("🚀 Starting autonomous request processing")
        logger.info(f"   Request: {request[:100]}...")
        logger.info(f"   Session ID: {session_id}")
        logger.info(f"   Autonomy Level: {autonomy.value}")

        try:
            # 1. ANALYZE: Understand the request and determine complexity
            analysis = await self._analyze_request(request)
            logger.info(f"📊 Request Analysis: {analysis['complexity'].value} complexity")

            # 2. DECOMPOSE: Break down into manageable goals and steps
            autonomous_plan = await self._create_autonomous_plan(request, analysis, autonomy)
            logger.info(f"📋 Generated plan with {len(autonomous_plan.steps)} steps")

            # 3. PERSIST: Save context snapshot
            context = self._create_context_snapshot(session_id, autonomous_plan)
            self.active_sessions[session_id] = context
            await self._persist_context(context)

            # 4. EXECUTE: Run the plan autonomously
            steps_executed = []

            for i, step in enumerate(autonomous_plan.steps):
                autonomous_plan.current_step_index = i
                autonomous_plan.progress_percentage = (i / len(autonomous_plan.steps)) * 100

                # Check if user confirmation is needed
                if await self._needs_user_confirmation(step, autonomy):
                    if user_confirmation_callback:
                        confirmed = await user_confirmation_callback(step, autonomous_plan)
                        if not confirmed:
                            logger.info(f"⏸️ User cancelled execution at step: {step.description}")
                            break
                    elif autonomy in [AutonomyLevel.MANUAL, AutonomyLevel.GUIDED]:
                        logger.warning("User confirmation required but no callback provided")
                        break

                # Execute the step with retry logic
                step_result = await self._execute_step_with_retry(step, autonomous_plan)
                steps_executed.append(step)

                # Stream progress if requested
                if streaming:
                    await self._stream_progress_update(session_id, step, autonomous_plan)

                # Update context
                try:
                    # Use model_dump with proper serialization
                    step_dict = step.model_dump()
                    # Convert datetime objects to ISO format strings for JSON serialization
                    if "timestamp" in step_dict and hasattr(step_dict["timestamp"], "isoformat"):
                        step_dict["timestamp"] = step_dict["timestamp"].isoformat()
                    context.completed_steps.append(step_dict)
                    await self._persist_context(context)
                except Exception as e:
                    logger.warning(f"Failed to serialize step for context: {e}")
                    # Simple fallback serialization
                    simple_step = {
                        "type": (
                            str(step.type.value) if hasattr(step.type, "value") else str(step.type)
                        ),
                        "description": step.description,
                        "tool_name": step.tool_name,
                        "success": step.result.success if step.result else False,
                    }
                    context.completed_steps.append(simple_step)

                # Check if plan needs adjustment
                if step_result and not step_result.success:
                    adjustment_needed = await self._assess_plan_adjustment(
                        step, step_result, autonomous_plan
                    )
                    if adjustment_needed:
                        logger.info("🔄 Plan adjustment needed, re-planning...")
                        # Create a mock observation for the parent _adjust_plan method
                        from gterminal.core.react_engine import Observation

                        observation = Observation(
                            step_successful=step_result.success,
                            needs_adjustment=True,
                            adjustment_reason=f"Step failed: {step_result.error}",
                            next_action="adjust",
                        )
                        # Get the session object from the session manager
                        session = self.session_manager.get_or_create(session_id)
                        autonomous_plan = await self._adjust_plan(
                            autonomous_plan, observation, session
                        )

            # 5. COMPLETE: Generate final response and learn from execution
            final_result = await self._complete_autonomous_execution(
                steps_executed, autonomous_plan, context
            )

            # 6. LEARN: Extract patterns for future use
            await self._learn_from_execution(request, autonomous_plan, steps_executed, final_result)

            # Create response
            response = ReactResponse(
                success=True,
                result=final_result,
                steps_executed=steps_executed,
                total_time=(datetime.now() - start_time).total_seconds(),
                session_id=session_id,
            )

            # Update success metrics
            self.success_metrics["successful_requests"] += 1
            self.success_metrics["average_completion_time"] = (
                self.success_metrics["average_completion_time"]
                * (self.success_metrics["successful_requests"] - 1)
                + response.total_time
            ) / self.success_metrics["successful_requests"]

            if autonomy == AutonomyLevel.FULLY_AUTO:
                self.success_metrics["autonomous_completions"] += 1

            logger.info(
                f"✅ Autonomous request completed successfully in {response.total_time:.2f}s"
            )
            return response

        except Exception as e:
            logger.exception(f"❌ Autonomous request processing failed: {e}")
            return ReactResponse(
                success=False,
                result={
                    "error": str(e),
                    "autonomous_engine": True,
                    "session_id": session_id,
                },
                steps_executed=[],
                total_time=(datetime.now() - start_time).total_seconds(),
                session_id=session_id,
            )

    async def _analyze_request(self, request: str) -> dict[str, Any]:
        """Analyze the request to determine complexity and approach."""
        prompt = f"""Analyze this request to determine the best autonomous approach:

Request: {request}

Please analyze and respond in JSON format:
{{
    "complexity": "simple|moderate|complex|advanced",
    "estimated_steps": <number>,
    "required_capabilities": ["list", "of", "capabilities"],
    "potential_challenges": ["list", "of", "challenges"],
    "recommended_autonomy": "manual|guided|semi_auto|fully_auto",
    "success_probability": <0.0-1.0>,
    "estimated_time_minutes": <number>,
    "requires_external_resources": true/false,
    "can_be_parallelized": true/false
}}"""

        try:
            response = await self.gemini_client.generate_content(prompt)

            # Use enhanced JSON extraction
            analysis = extract_json_from_llm_response(response, expected_type="dict")

            # Validate the analysis data
            if not validate_analysis_data(analysis):
                logger.warning("Analysis data validation failed, using defaults")
                msg = "Invalid analysis structure"
                raise ValueError(msg)

            # Map complexity string to enum
            complexity_map = {
                "simple": TaskComplexity.SIMPLE,
                "moderate": TaskComplexity.MODERATE,
                "complex": TaskComplexity.COMPLEX,
                "advanced": TaskComplexity.ADVANCED,
            }
            analysis["complexity"] = complexity_map.get(
                analysis.get("complexity", "moderate"), TaskComplexity.MODERATE
            )

            logger.info("✅ Request analysis completed successfully")
            logger.info(f"   Complexity: {analysis['complexity'].value}")
            logger.info(f"   Estimated Steps: {analysis.get('estimated_steps', 'unknown')}")
            logger.info(f"   Success Probability: {analysis.get('success_probability', 0.7):.1%}")

            return analysis
        except Exception as e:
            logger.warning(f"Request analysis failed, using defaults: {e}")
            return {
                "complexity": TaskComplexity.MODERATE,
                "estimated_steps": 3,
                "required_capabilities": ["general"],
                "potential_challenges": ["unknown"],
                "recommended_autonomy": "semi_auto",
                "success_probability": 0.7,
                "estimated_time_minutes": 10,
                "requires_external_resources": False,
                "can_be_parallelized": False,
            }

    async def _create_autonomous_plan(
        self,
        request: str,
        analysis: dict[str, Any],
        autonomy_level: AutonomyLevel,
    ) -> AutonomousPlan:
        """Create an autonomous execution plan."""
        # Check for similar learned patterns
        similar_patterns = await self._find_similar_patterns(request)
        pattern_context = ""
        if similar_patterns:
            self.success_metrics["pattern_matches"] += 1
            pattern_context = f"\nSimilar successful patterns found:\n{json.dumps(similar_patterns[:3], indent=2)}"

        # Get available tools
        available_tools = self.tool_registry.get_tool_descriptions()

        prompt = f"""Create an autonomous execution plan for this request:

Request: {request}

Analysis Results:
{json.dumps(analysis, indent=2)}

Available Tools:
{json.dumps(available_tools, indent=2)}

Autonomy Level: {autonomy_level.value}
{pattern_context}

IMPORTANT: Use ONLY tools from the Available Tools list above. If no suitable tool exists, use "text_response" for text-based responses.

Create a detailed autonomous plan in JSON format:
{{
    "goal": "Clear main objective",
    "steps": [
        {{
            "type": "act",
            "description": "Detailed step description",
            "tool_name": "text_response",
            "tool_params": {{"message": "Response content"}},
            "priority": "medium",
            "estimated_duration": 5,
            "dependencies": [],
            "success_probability": 0.9,
            "retry_strategy": "immediate",
            "checkpoint": false
        }}
    ],
    "complexity": "{analysis["complexity"].value}",
    "estimated_total_time": 10,
    "success_probability": 0.8,
    "fallback_strategies": ["text_response"],
    "critical_checkpoints": [],
    "parallel_groups": []
}}"""

        try:
            response = await self.gemini_client.generate_content(prompt)

            # Use enhanced JSON extraction
            plan_data = extract_json_from_llm_response(response, expected_type="dict")

            # Validate the plan data
            if not validate_plan_data(plan_data):
                logger.warning("Plan data validation failed, attempting repair")
                # Try to fix common issues
                if "goal" not in plan_data:
                    plan_data["goal"] = request
                if "steps" not in plan_data:
                    plan_data["steps"] = []

            # Convert to AutonomousStep objects with parameter validation
            autonomous_steps = []
            for i, step_data in enumerate(plan_data.get("steps", [])):
                try:
                    # Ensure step_data has required fields with defaults
                    step_data = {
                        "type": step_data.get("type", "act"),
                        "description": step_data.get("description", f"Step {i + 1}"),
                        "tool_name": step_data.get("tool_name"),
                        "tool_params": step_data.get("tool_params", {}),
                        "priority": step_data.get("priority", "medium"),
                        "estimated_duration": step_data.get("estimated_duration"),
                        "dependencies": step_data.get("dependencies", []),
                        "success_probability": step_data.get("success_probability"),
                        "checkpoint": step_data.get("checkpoint", False),
                    }

                    # Validate and sanitize tool parameters if present
                    if step_data.get("tool_params") and step_data.get("tool_name"):
                        tool_name = step_data["tool_name"]
                        try:
                            validated_params = validate_tool_parameters(
                                tool_name, step_data["tool_params"]
                            )
                            step_data["tool_params"] = validated_params
                        except (ParameterValidationError, Exception) as e:
                            logger.warning(f"Parameter validation failed for step {i}: {e}")
                            # Keep original params as fallback

                    # Convert string enums to proper enums
                    if isinstance(step_data.get("type"), str):
                        step_data["type"] = StepType(step_data["type"])

                    if isinstance(step_data.get("priority"), str):
                        priority_map = {
                            "low": TaskPriority.LOW,
                            "medium": TaskPriority.MEDIUM,
                            "high": TaskPriority.HIGH,
                            "critical": TaskPriority.CRITICAL,
                        }
                        step_data["priority"] = priority_map.get(
                            step_data["priority"].lower(), TaskPriority.MEDIUM
                        )

                    # Ensure dependencies is a list of strings
                    if step_data.get("dependencies"):
                        if isinstance(step_data["dependencies"], str):
                            step_data["dependencies"] = [step_data["dependencies"]]
                        elif isinstance(step_data["dependencies"], list):
                            step_data["dependencies"] = [
                                str(dep) for dep in step_data["dependencies"]
                            ]
                        else:
                            step_data["dependencies"] = []

                    step = AutonomousStep(**step_data)
                    autonomous_steps.append(step)
                except Exception as e:
                    logger.warning(f"Failed to create step {i}: {e}, creating simple fallback step")
                    # Create a simple fallback step
                    fallback_step = AutonomousStep(
                        type=StepType.ACT,
                        description=step_data.get("description", f"Fallback step {i + 1}"),
                        tool_name=step_data.get("tool_name", "text_response"),
                        tool_params=step_data.get("tool_params", {"message": "Processing step"}),
                    )
                    autonomous_steps.append(fallback_step)
                    continue

            # Create autonomous plan
            autonomous_plan = AutonomousPlan(
                goal=plan_data.get("goal", request),
                steps=autonomous_steps,
                complexity=TaskComplexity(plan_data.get("complexity", "moderate")),
                autonomy_level=autonomy_level,
                estimated_total_time=plan_data.get("estimated_total_time", 300.0),
                success_probability=plan_data.get("success_probability", 0.8),
                checkpoints=plan_data.get("critical_checkpoints", []),
                started_at=datetime.now(),
            )

            logger.info("📋 Autonomous plan created:")
            logger.info(f"   Goal: {autonomous_plan.goal}")
            logger.info(f"   Steps: {len(autonomous_plan.steps)}")
            logger.info(f"   Complexity: {autonomous_plan.complexity.value}")
            logger.info(f"   Estimated Time: {autonomous_plan.estimated_total_time:.1f}s")

            return autonomous_plan

        except Exception as e:
            logger.exception(f"Plan creation failed: {e}")
            # Fallback to simple plan with available tools
            available_tool_names = list(available_tools.keys()) if available_tools else []

            # Use text_response if available, otherwise use the first available tool
            tool_name = "text_response"
            tool_params = {"message": f"Processing request: {request}"}

            if tool_name not in available_tool_names and available_tool_names:
                tool_name = available_tool_names[0]
                tool_params = {"request": request}

            simple_step = AutonomousStep(
                type=StepType.ACT,
                description=f"Process request: {request}",
                tool_name=tool_name,
                tool_params=tool_params,
                priority=TaskPriority.MEDIUM,
            )
            return AutonomousPlan(
                goal=request,
                steps=[simple_step],
                complexity=TaskComplexity.SIMPLE,
                autonomy_level=autonomy_level,
            )

    async def _execute_step_with_retry(
        self, step: AutonomousStep, plan: AutonomousPlan
    ) -> ToolResult | None:
        """Execute a step with automatic retry logic."""
        logger.info(f"🔧 Executing step: {step.description}")

        for attempt in range(step.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"   🔄 Retry attempt {attempt}/{step.max_retries}")

                result = await self._act(step.tool_name, step.tool_params or {})
                step.result = result

                if result.success:
                    logger.info("   ✅ Step completed successfully")
                    return result
                logger.warning(f"   ⚠️ Step failed: {result.error}")
                step.retry_count = attempt + 1

                if attempt < step.max_retries:
                    # Apply retry delay (exponential backoff)
                    delay = 2**attempt
                    logger.info(f"   ⏳ Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)

            except Exception as e:
                logger.exception(f"   ❌ Step execution error: {e}")
                step.retry_count = attempt + 1

                if attempt < step.max_retries:
                    await asyncio.sleep(2**attempt)

        # All retries failed
        logger.error(f"❌ Step failed after {step.max_retries + 1} attempts")
        return ToolResult(
            success=False,
            data=None,
            error=f"Max retries exceeded for step: {step.description}",
        )

    async def _needs_user_confirmation(
        self, step: AutonomousStep, autonomy_level: AutonomyLevel
    ) -> bool:
        """Determine if user confirmation is needed for a step."""
        if autonomy_level == AutonomyLevel.FULLY_AUTO:
            return False
        if autonomy_level == AutonomyLevel.MANUAL:
            return True
        if autonomy_level == AutonomyLevel.GUIDED:
            return step.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]
        if autonomy_level == AutonomyLevel.SEMI_AUTO:
            return step.priority == TaskPriority.CRITICAL

        return False

    async def _assess_plan_adjustment(
        self, step: AutonomousStep, result: ToolResult, plan: AutonomousPlan
    ) -> bool:
        """Assess whether plan adjustment is needed based on step failure."""
        if result.success:
            return False

        # Simple rules-based assessment
        if step.retry_count >= step.max_retries:
            return True  # Multiple failures suggest plan issue

        if "not found" in (result.error or "").lower():
            return True  # Tool/resource not found

        # For now, we'll use simple rules. In the future, this could use Gemini for analysis
        return False

    async def _complete_autonomous_execution(
        self,
        steps: list[Step],
        plan: AutonomousPlan,
        context: ContextSnapshot,
    ) -> dict[str, Any]:
        """Complete autonomous execution and generate final results."""
        # Mark plan as completed
        plan.completed_at = datetime.now()

        # Use the parent's completion method
        return await self._complete(steps, context.session_id)

    def _create_context_snapshot(self, session_id: str, plan: AutonomousPlan) -> ContextSnapshot:
        """Create a context snapshot for persistence."""
        return ContextSnapshot(
            session_id=session_id,
            timestamp=datetime.now(),
            current_plan=self._safe_model_dump(plan),
            active_goals=[plan.goal],
            success_metrics=self.success_metrics.copy(),
        )

    async def _persist_context(self, context: ContextSnapshot) -> None:
        """Persist context snapshot to disk."""
        try:
            context_file = self.context_dir / f"{context.session_id}.json"
            context_data = {
                "session_id": context.session_id,
                "timestamp": context.timestamp.isoformat(),
                "current_plan": context.current_plan,
                "completed_steps": context.completed_steps,
                "active_goals": context.active_goals,
                "learned_patterns": context.learned_patterns,
                "success_metrics": context.success_metrics,
            }

            with open(context_file, "w") as f:
                json.dump(context_data, f, indent=2)

            logger.debug(f"💾 Context persisted: {context_file}")

        except Exception as e:
            logger.warning(f"Failed to persist context: {e}")

    async def _find_similar_patterns(self, request: str) -> list[dict[str, Any]]:
        """Find similar learned patterns for the request."""
        if not self.learned_patterns:
            return []

        try:
            # Simple similarity matching based on keywords
            request_words = set(request.lower().split())
            similar_patterns = []

            for pattern_id, pattern in self.learned_patterns.items():
                pattern_words = set(pattern.get("original_request", "").lower().split())

                # Calculate Jaccard similarity
                intersection = request_words.intersection(pattern_words)
                union = request_words.union(pattern_words)

                if union:
                    similarity = len(intersection) / len(union)
                    if similarity > 0.3:  # 30% similarity threshold
                        pattern["similarity"] = similarity
                        pattern["pattern_id"] = pattern_id
                        similar_patterns.append(pattern)

            # Sort by similarity and return top matches
            similar_patterns.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_patterns[:5]

        except Exception as e:
            logger.warning(f"Pattern matching failed: {e}")
            return []

    async def _learn_from_execution(
        self,
        request: str,
        plan: AutonomousPlan,
        steps: list[Step],
        result: dict[str, Any],
    ) -> None:
        """Learn from successful execution to improve future performance."""
        if not result or not result.get("success", True):
            return

        try:
            # Create learning pattern
            pattern_id = f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            pattern = {
                "original_request": request,
                "complexity": plan.complexity.value,
                "autonomy_level": plan.autonomy_level.value,
                "successful_steps": [
                    {
                        "description": step.description,
                        "tool_name": step.tool_name,
                        "tool_params": step.tool_params,
                        "success": step.result.success if step.result else False,
                    }
                    for step in steps
                    if step.tool_name
                ],
                "execution_time": (
                    plan.completed_at.timestamp() - plan.started_at.timestamp()
                    if plan.completed_at and plan.started_at
                    else 0
                ),
                "success_probability": plan.success_probability,
                "learned_at": datetime.now().isoformat(),
                "usage_count": 1,
            }

            # Store the pattern
            self.learned_patterns[pattern_id] = pattern

            # Persist patterns
            await self._save_learned_patterns()

            logger.info(f"🧠 Learned new pattern: {pattern_id}")

        except Exception as e:
            logger.warning(f"Failed to learn from execution: {e}")

    def _load_learned_patterns(self) -> None:
        """Load learned patterns from disk."""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file) as f:
                    self.learned_patterns = json.load(f)
                logger.info(f"📚 Loaded {len(self.learned_patterns)} learned patterns")
            else:
                self.learned_patterns = {}
        except Exception as e:
            logger.warning(f"Failed to load learned patterns: {e}")
            self.learned_patterns = {}

    async def _save_learned_patterns(self) -> None:
        """Save learned patterns to disk."""
        try:
            with open(self.patterns_file, "w") as f:
                json.dump(self.learned_patterns, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save learned patterns: {e}")

    async def _stream_progress_update(
        self, session_id: str, step: AutonomousStep, plan: AutonomousPlan
    ):
        """Stream progress update for the current step."""
        update = {
            "type": "autonomous_progress",
            "session_id": session_id,
            "plan_id": plan.plan_id,
            "current_step": {
                "id": step.step_id,
                "description": step.description,
                "progress": plan.progress_percentage,
                "status": ("completed" if step.result and step.result.success else "failed"),
            },
            "overall_progress": {
                "steps_completed": plan.current_step_index + 1,
                "total_steps": len(plan.steps),
                "percentage": plan.progress_percentage,
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"📡 Progress Update: {update['overall_progress']['percentage']:.1f}% - {step.description}"
        )

        # In a real implementation, this would stream to WebSocket or similar
        # For now, we just log the update
        return update

    def _safe_model_dump(self, model) -> dict[str, Any]:
        """Safely serialize a Pydantic model to dict with proper datetime handling."""
        try:
            # Use model_dump with proper serialization
            data = model.model_dump()

            # Handle datetime serialization recursively
            return self._serialize_datetimes(data)
        except Exception as e:
            logger.warning(f"Model dump failed: {e}, using fallback serialization")
            # Fallback to basic dict representation
            if hasattr(model, "__dict__"):
                return self._serialize_datetimes(dict(model.__dict__))
            return {"error": f"Failed to serialize model: {e!s}"}

    def _serialize_datetimes(self, data) -> Any:
        """Recursively serialize datetime objects to ISO format strings."""
        if isinstance(data, dict):
            return {key: self._serialize_datetimes(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self._serialize_datetimes(item) for item in data]
        if isinstance(data, datetime):
            return data.isoformat()
        if hasattr(data, "value"):  # Handle enum values
            return data.value
        return data

    async def get_autonomous_status(self) -> dict[str, Any]:
        """Get comprehensive status of the autonomous engine."""
        return {
            "engine_info": {
                "profile": self.profile,
                "autonomy_level": self.autonomy_level.value,
                "gemini_status": self.gemini_client.get_status(),
                "context_directory": str(self.context_dir),
            },
            "learning_data": {
                "learned_patterns": len(self.learned_patterns),
                "active_sessions": len(self.active_sessions),
                "patterns_file": str(self.patterns_file),
                "patterns_exist": self.patterns_file.exists(),
            },
            "performance_metrics": self.success_metrics,
            "capabilities": {
                "goal_decomposition": True,
                "multi_step_planning": True,
                "context_persistence": True,
                "pattern_learning": True,
                "error_recovery": True,
                "streaming_progress": True,
                "autonomous_execution": True,
            },
        }


# Convenience functions
def create_autonomous_engine(
    profile: str = "business",
    autonomy_level: AutonomyLevel = AutonomyLevel.SEMI_AUTO,
) -> AutonomousReactEngine:
    """Create a configured autonomous ReAct engine."""
    return AutonomousReactEngine(profile=profile, autonomy_level=autonomy_level)


async def test_autonomous_capabilities(profile: str = "business") -> dict[str, Any]:
    """Test autonomous capabilities with a simple request."""
    logger.info("🧪 Testing autonomous capabilities...")

    engine = create_autonomous_engine(profile=profile, autonomy_level=AutonomyLevel.FULLY_AUTO)

    test_request = "Create a simple test file with current timestamp and system info"

    try:
        response = await engine.process_autonomous_request(test_request, streaming=True)

        return {
            "test_successful": response.success,
            "execution_time": response.total_time,
            "steps_executed": len(response.steps_executed),
            "autonomous_status": await engine.get_autonomous_status(),
            "error": response.result.get("error") if not response.success else None,
        }

    except Exception as e:
        return {
            "test_successful": False,
            "error": str(e),
            "autonomous_status": await engine.get_autonomous_status(),
        }


if __name__ == "__main__":
    # Quick test when run directly
    import asyncio

    async def main() -> None:
        results = await test_autonomous_capabilities()

        for value in results.values():
            if isinstance(value, dict):
                for _sub_key, _sub_value in value.items():
                    pass
            else:
                pass

    asyncio.run(main())
