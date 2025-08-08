#!/usr/bin/env python3
"""Comprehensive unit tests for the autonomous ReAct engine.

This test suite achieves high coverage by testing:
- Engine initialization and configuration
- Autonomous request processing pipeline
- Plan creation and execution
- Context persistence and learning
- Error handling and retry logic
- Progress tracking and streaming
- Pattern learning and similarity matching
"""

import asyncio
from datetime import datetime
import json
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import mock_open
from unittest.mock import patch

import pytest

from core.autonomous_react_engine import AutonomousPlan
from core.autonomous_react_engine import AutonomousReactEngine
from core.autonomous_react_engine import AutonomousStep
from core.autonomous_react_engine import AutonomyLevel
from core.autonomous_react_engine import ContextSnapshot
from core.autonomous_react_engine import Dependency
from core.autonomous_react_engine import TaskComplexity
from core.autonomous_react_engine import TaskPriority
from core.autonomous_react_engine import create_autonomous_engine


class TestEnums:
    """Test cases for enum classes."""

    def test_autonomy_level_values(self):
        """Test AutonomyLevel enum values."""
        assert AutonomyLevel.MANUAL == "manual"
        assert AutonomyLevel.GUIDED == "guided"
        assert AutonomyLevel.SEMI_AUTO == "semi_auto"
        assert AutonomyLevel.FULLY_AUTO == "fully_auto"

    def test_task_complexity_values(self):
        """Test TaskComplexity enum values."""
        assert TaskComplexity.SIMPLE == "simple"
        assert TaskComplexity.MODERATE == "moderate"
        assert TaskComplexity.COMPLEX == "complex"
        assert TaskComplexity.ADVANCED == "advanced"

    def test_task_priority_values(self):
        """Test TaskPriority enum values."""
        assert TaskPriority.LOW == "low"
        assert TaskPriority.MEDIUM == "medium"
        assert TaskPriority.HIGH == "high"
        assert TaskPriority.CRITICAL == "critical"


class TestDataClasses:
    """Test cases for dataclass models."""

    def test_dependency_creation(self):
        """Test Dependency dataclass creation."""
        dep = Dependency(
            source_step_id="step1",
            target_step_id="step2",
            dependency_type="data",
            required_data={"key": "value"},
        )
        assert dep.source_step_id == "step1"
        assert dep.target_step_id == "step2"
        assert dep.dependency_type == "data"
        assert dep.required_data == {"key": "value"}

    def test_dependency_defaults(self):
        """Test Dependency with default values."""
        dep = Dependency(source_step_id="step1", target_step_id="step2")
        assert dep.dependency_type == "completion"
        assert dep.required_data is None

    def test_context_snapshot_creation(self):
        """Test ContextSnapshot dataclass creation."""
        snapshot = ContextSnapshot(
            session_id="test-session",
            timestamp=datetime.now(),
            current_plan={"goal": "test"},
            completed_steps=[{"step": "done"}],
            active_goals=["goal1"],
            learned_patterns=[{"pattern": "test"}],
            success_metrics={"success": 0.9},
        )
        assert snapshot.session_id == "test-session"
        assert isinstance(snapshot.timestamp, datetime)
        assert snapshot.current_plan["goal"] == "test"

    def test_context_snapshot_defaults(self):
        """Test ContextSnapshot with default values."""
        snapshot = ContextSnapshot(session_id="test", timestamp=datetime.now())
        assert snapshot.current_plan is None
        assert snapshot.completed_steps == []
        assert snapshot.active_goals == []
        assert snapshot.learned_patterns == []
        assert snapshot.success_metrics == {}


class TestAutonomousStep:
    """Test cases for AutonomousStep class."""

    def test_autonomous_step_creation(self):
        """Test basic AutonomousStep creation."""
        step = AutonomousStep(
            type="act", description="Test step", tool_name="test_tool", priority=TaskPriority.HIGH
        )
        assert step.description == "Test step"
        assert step.tool_name == "test_tool"
        assert step.priority == TaskPriority.HIGH
        assert step.retry_count == 0
        assert step.max_retries == 3

    def test_autonomous_step_string_priority_conversion(self):
        """Test string to TaskPriority enum conversion."""
        step = AutonomousStep(type="act", description="Test", priority="high")
        assert step.priority == TaskPriority.HIGH

    def test_autonomous_step_invalid_priority_fallback(self):
        """Test fallback for invalid priority string."""
        step = AutonomousStep(type="act", description="Test", priority="invalid")
        assert step.priority == TaskPriority.MEDIUM

    def test_autonomous_step_dependencies_string_conversion(self):
        """Test string dependencies conversion to list."""
        step = AutonomousStep(type="act", description="Test", dependencies="step1, step2, step3")
        assert step.dependencies == ["step1", "step2", "step3"]

    def test_autonomous_step_empty_dependencies_string(self):
        """Test empty dependencies string conversion."""
        step = AutonomousStep(type="act", description="Test", dependencies="")
        assert step.dependencies == []

    def test_autonomous_step_generates_id(self):
        """Test that step ID is automatically generated."""
        step = AutonomousStep(type="act", description="Test")
        assert step.step_id.startswith("step_")
        assert len(step.step_id) > 10


class TestAutonomousPlan:
    """Test cases for AutonomousPlan class."""

    def test_autonomous_plan_creation(self):
        """Test basic AutonomousPlan creation."""
        steps = [AutonomousStep(type="act", description="Step 1")]
        plan = AutonomousPlan(goal="Test goal", steps=steps, complexity=TaskComplexity.MODERATE)
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 1
        assert plan.complexity == TaskComplexity.MODERATE
        assert plan.progress_percentage == 0.0

    def test_autonomous_plan_string_complexity_conversion(self):
        """Test string to TaskComplexity enum conversion."""
        plan = AutonomousPlan(goal="Test", steps=[], complexity="complex")
        assert plan.complexity == TaskComplexity.COMPLEX

    def test_autonomous_plan_invalid_complexity_fallback(self):
        """Test fallback for invalid complexity string."""
        plan = AutonomousPlan(goal="Test", steps=[], complexity="invalid")
        assert plan.complexity == TaskComplexity.SIMPLE

    def test_autonomous_plan_string_autonomy_conversion(self):
        """Test string to AutonomyLevel enum conversion."""
        plan = AutonomousPlan(goal="Test", steps=[], autonomy_level="fully_auto")
        assert plan.autonomy_level == AutonomyLevel.FULLY_AUTO

    def test_autonomous_plan_dependencies_conversion(self):
        """Test dependencies string conversion."""
        plan = AutonomousPlan(goal="Test", steps=[], dependencies="dep1,dep2,dep3")
        assert plan.dependencies == ["dep1", "dep2", "dep3"]

    def test_autonomous_plan_generates_id(self):
        """Test that plan ID is automatically generated."""
        plan = AutonomousPlan(goal="Test", steps=[])
        assert plan.plan_id.startswith("plan_")
        assert len(plan.plan_id) > 10


class TestAutonomousReactEngineInitialization:
    """Test cases for AutonomousReactEngine initialization."""

    @patch("core.autonomous_react_engine.GeminiClient")
    def test_engine_init_success(self, mock_client_class):
        """Test successful engine initialization."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        engine = AutonomousReactEngine(profile="business", autonomy_level=AutonomyLevel.SEMI_AUTO)

        assert engine.profile == "business"
        assert engine.autonomy_level == AutonomyLevel.SEMI_AUTO
        assert engine.gemini_client == mock_client
        assert engine.context_dir.name == ".react_context"

    @patch("core.autonomous_react_engine.GeminiClient", side_effect=Exception("Client failed"))
    def test_engine_init_client_failure(self, mock_client_class):
        """Test engine initialization with client failure."""
        with patch("core.autonomous_react_engine.logger"):
            AutonomousReactEngine()
            # Should still initialize successfully with fallback

    def test_engine_context_directory_creation(self):
        """Test that context directory is created."""
        with patch("core.autonomous_react_engine.GeminiClient"):
            with patch("core.autonomous_react_engine.Path"):
                mock_project_root = Mock()
                mock_context_dir = Mock()
                mock_project_root.__truediv__.return_value = mock_context_dir

                AutonomousReactEngine(project_root=mock_project_root)
                mock_context_dir.mkdir.assert_called_once_with(exist_ok=True)

    @patch("core.autonomous_react_engine.GeminiClient")
    def test_engine_loads_learned_patterns(self, mock_client_class):
        """Test that learned patterns are loaded during initialization."""
        with patch.object(AutonomousReactEngine, "_load_learned_patterns") as mock_load:
            AutonomousReactEngine()
            mock_load.assert_called_once()

    @patch("core.autonomous_react_engine.GeminiClient")
    def test_engine_success_metrics_initialization(self, mock_client_class):
        """Test success metrics are initialized."""
        engine = AutonomousReactEngine()

        expected_metrics = {
            "total_requests",
            "successful_requests",
            "average_completion_time",
            "pattern_matches",
            "autonomous_completions",
        }
        assert set(engine.success_metrics.keys()) == expected_metrics


class TestRequestAnalysis:
    """Test cases for request analysis functionality."""

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_analyze_request_success(self, mock_client_class):
        """Test successful request analysis."""
        mock_client = Mock()
        mock_client.generate_content = AsyncMock(
            return_value='{"complexity": "moderate", "estimated_steps": 3}'
        )
        mock_client_class.return_value = mock_client

        engine = AutonomousReactEngine()
        engine.gemini_client = mock_client

        result = await engine._analyze_request("test request")

        assert result["complexity"] == TaskComplexity.MODERATE
        assert result["estimated_steps"] == 3
        mock_client.generate_content.assert_called_once()

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_analyze_request_client_failure(self, mock_client_class):
        """Test request analysis with client failure."""
        engine = AutonomousReactEngine()
        engine.gemini_client = Mock(spec=[])  # No generate_content method

        result = await engine._analyze_request("test request")

        # Should return defaults
        assert result["complexity"] == TaskComplexity.MODERATE
        assert result["estimated_steps"] == 3
        assert result["success_probability"] == 0.7

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_analyze_request_json_parse_error(self, mock_client_class):
        """Test request analysis with invalid JSON response."""
        mock_client = Mock()
        mock_client.generate_content = AsyncMock(return_value="invalid json")
        mock_client_class.return_value = mock_client

        engine = AutonomousReactEngine()
        engine.gemini_client = mock_client

        result = await engine._analyze_request("test request")

        # Should fallback to defaults
        assert result["complexity"] == TaskComplexity.MODERATE


class TestPlanCreation:
    """Test cases for autonomous plan creation."""

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_create_autonomous_plan_success(self, mock_client_class):
        """Test successful plan creation."""
        engine = AutonomousReactEngine()

        analysis = {
            "complexity": TaskComplexity.MODERATE,
            "estimated_steps": 2,
            "estimated_time_minutes": 5,
        }

        plan = await engine._create_autonomous_plan(
            "test request", analysis, AutonomyLevel.SEMI_AUTO
        )

        assert isinstance(plan, AutonomousPlan)
        assert plan.goal == "test request"
        assert len(plan.steps) == 1  # Fallback to simple step
        assert plan.complexity == TaskComplexity.MODERATE
        assert plan.autonomy_level == AutonomyLevel.SEMI_AUTO

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_create_autonomous_plan_exception(self, mock_client_class):
        """Test plan creation with exception."""
        engine = AutonomousReactEngine()

        # Simulate exception during plan creation
        with patch.object(engine, "_get_available_tools", side_effect=Exception("Test error")):
            analysis = {"complexity": TaskComplexity.SIMPLE}

            plan = await engine._create_autonomous_plan(
                "test request", analysis, AutonomyLevel.SEMI_AUTO
            )

            # Should still create fallback plan
            assert isinstance(plan, AutonomousPlan)
            assert plan.goal == "test request"
            assert len(plan.steps) == 1


class TestStepExecution:
    """Test cases for step execution with retry logic."""

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_execute_step_success(self, mock_client_class):
        """Test successful step execution."""
        engine = AutonomousReactEngine()

        step = AutonomousStep(type="act", description="Test step", tool_name="test_tool")
        plan = AutonomousPlan(goal="Test", steps=[step])

        result = await engine._execute_step_with_retry(step, plan)

        assert result is not None
        assert result.success is True
        assert step.result == result

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_execute_step_with_retries(self, mock_client_class):
        """Test step execution with retries."""
        engine = AutonomousReactEngine()

        step = AutonomousStep(type="act", description="Test step", max_retries=2)
        plan = AutonomousPlan(goal="Test", steps=[step])

        # Mock to fail first two attempts, succeed on third
        call_count = 0

        async def mock_act(tool_name, params):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                from core.autonomous_react_engine import ToolResult

                return ToolResult(success=False, error="Simulated failure")
            return ToolResult(success=True, data={"success": True})

        with patch.object(engine, "_act", side_effect=mock_act):
            result = await engine._execute_step_with_retry(step, plan)

            assert result.success is True
            assert step.retry_count == 2

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_execute_step_max_retries_exceeded(self, mock_client_class):
        """Test step execution when max retries are exceeded."""
        engine = AutonomousReactEngine()

        step = AutonomousStep(type="act", description="Test step", max_retries=1)
        plan = AutonomousPlan(goal="Test", steps=[step])

        # Mock to always fail
        async def mock_act(tool_name, params):
            from core.autonomous_react_engine import ToolResult

            return ToolResult(success=False, error="Always fails")

        with patch.object(engine, "_act", side_effect=mock_act):
            result = await engine._execute_step_with_retry(step, plan)

            assert result.success is False
            assert "Max retries exceeded" in result.error
            assert step.retry_count == 2  # 1 original + 1 retry


class TestUserConfirmation:
    """Test cases for user confirmation logic."""

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_needs_user_confirmation_fully_auto(self, mock_client_class):
        """Test no confirmation needed for fully autonomous mode."""
        engine = AutonomousReactEngine()

        step = AutonomousStep(type="act", description="Test", priority=TaskPriority.CRITICAL)

        needs_confirmation = await engine._needs_user_confirmation(step, AutonomyLevel.FULLY_AUTO)

        assert needs_confirmation is False

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_needs_user_confirmation_manual(self, mock_client_class):
        """Test confirmation needed for manual mode."""
        engine = AutonomousReactEngine()

        step = AutonomousStep(type="act", description="Test", priority=TaskPriority.LOW)

        needs_confirmation = await engine._needs_user_confirmation(step, AutonomyLevel.MANUAL)

        assert needs_confirmation is True

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_needs_user_confirmation_guided_high_priority(self, mock_client_class):
        """Test confirmation needed for high priority in guided mode."""
        engine = AutonomousReactEngine()

        step = AutonomousStep(type="act", description="Test", priority=TaskPriority.HIGH)

        needs_confirmation = await engine._needs_user_confirmation(step, AutonomyLevel.GUIDED)

        assert needs_confirmation is True

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_needs_user_confirmation_semi_auto_critical_only(self, mock_client_class):
        """Test confirmation only for critical priority in semi-auto mode."""
        engine = AutonomousReactEngine()

        high_step = AutonomousStep(type="act", description="Test", priority=TaskPriority.HIGH)
        critical_step = AutonomousStep(
            type="act", description="Test", priority=TaskPriority.CRITICAL
        )

        high_confirmation = await engine._needs_user_confirmation(
            high_step, AutonomyLevel.SEMI_AUTO
        )
        critical_confirmation = await engine._needs_user_confirmation(
            critical_step, AutonomyLevel.SEMI_AUTO
        )

        assert high_confirmation is False
        assert critical_confirmation is True


class TestContextPersistence:
    """Test cases for context persistence functionality."""

    @patch("core.autonomous_react_engine.GeminiClient")
    def test_create_context_snapshot(self, mock_client_class):
        """Test context snapshot creation."""
        engine = AutonomousReactEngine()

        plan = AutonomousPlan(goal="Test goal", steps=[])

        snapshot = engine._create_context_snapshot("test-session", plan)

        assert snapshot.session_id == "test-session"
        assert isinstance(snapshot.timestamp, datetime)
        assert snapshot.active_goals == ["Test goal"]
        assert "total_requests" in snapshot.success_metrics

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_persist_context_success(self, mock_client_class):
        """Test successful context persistence."""
        engine = AutonomousReactEngine()

        snapshot = ContextSnapshot(
            session_id="test", timestamp=datetime.now(), active_goals=["goal"]
        )

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            await engine._persist_context(snapshot)

            mock_file.assert_called_once()
            # Verify JSON was written
            written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)
            assert "test" in written_content

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_persist_context_failure(self, mock_client_class):
        """Test context persistence handles file errors."""
        engine = AutonomousReactEngine()

        snapshot = ContextSnapshot(session_id="test", timestamp=datetime.now())

        with patch("builtins.open", side_effect=OSError("File error")):
            with patch("core.autonomous_react_engine.logger") as mock_logger:
                await engine._persist_context(snapshot)
                mock_logger.warning.assert_called_once()


class TestPatternLearning:
    """Test cases for pattern learning functionality."""

    @patch("core.autonomous_react_engine.GeminiClient")
    def test_load_learned_patterns_success(self, mock_client_class):
        """Test successful loading of learned patterns."""
        engine = AutonomousReactEngine()

        patterns = {"pattern1": {"request": "test", "complexity": "simple"}}

        mock_file = mock_open(read_data=json.dumps(patterns))
        with patch("builtins.open", mock_file):
            with patch.object(engine.patterns_file, "exists", return_value=True):
                engine._load_learned_patterns()

                assert engine.learned_patterns == patterns

    @patch("core.autonomous_react_engine.GeminiClient")
    def test_load_learned_patterns_file_not_exists(self, mock_client_class):
        """Test loading patterns when file doesn't exist."""
        engine = AutonomousReactEngine()

        with patch.object(engine.patterns_file, "exists", return_value=False):
            engine._load_learned_patterns()

            assert engine.learned_patterns == {}

    @patch("core.autonomous_react_engine.GeminiClient")
    def test_load_learned_patterns_json_error(self, mock_client_class):
        """Test loading patterns with invalid JSON."""
        engine = AutonomousReactEngine()

        mock_file = mock_open(read_data="invalid json")
        with patch("builtins.open", mock_file):
            with patch.object(engine.patterns_file, "exists", return_value=True):
                with patch("core.autonomous_react_engine.logger"):
                    engine._load_learned_patterns()

                    assert engine.learned_patterns == {}

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_save_learned_patterns_success(self, mock_client_class):
        """Test successful saving of learned patterns."""
        engine = AutonomousReactEngine()
        engine.learned_patterns = {"pattern1": {"data": "test"}}

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            await engine._save_learned_patterns()

            mock_file.assert_called_once()

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_learn_from_execution_success(self, mock_client_class):
        """Test learning from successful execution."""
        engine = AutonomousReactEngine()

        plan = AutonomousPlan(
            goal="test",
            steps=[],
            complexity=TaskComplexity.SIMPLE,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )

        result = {"success": True}

        with patch.object(
            engine, "_save_learned_patterns", return_value=asyncio.Future()
        ) as mock_save:
            mock_save.return_value.set_result(None)

            await engine._learn_from_execution("test request", plan, [], result)

            assert len(engine.learned_patterns) > 0
            mock_save.assert_called_once()

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_learn_from_execution_failure(self, mock_client_class):
        """Test no learning from failed execution."""
        engine = AutonomousReactEngine()

        plan = AutonomousPlan(goal="test", steps=[])
        result = {"success": False}

        await engine._learn_from_execution("test request", plan, [], result)

        assert len(engine.learned_patterns) == 0


class TestProgressStreaming:
    """Test cases for progress streaming functionality."""

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_stream_progress_update(self, mock_client_class):
        """Test progress update streaming."""
        engine = AutonomousReactEngine()

        step = AutonomousStep(type="act", description="Test step", step_id="step-123")
        step.result = Mock(success=True)

        plan = AutonomousPlan(
            goal="Test",
            steps=[step],
            plan_id="plan-456",
            progress_percentage=50.0,
            current_step_index=0,
        )

        update = await engine._stream_progress_update("session-789", step, plan)

        assert update["type"] == "autonomous_progress"
        assert update["session_id"] == "session-789"
        assert update["plan_id"] == "plan-456"
        assert update["current_step"]["id"] == "step-123"
        assert update["overall_progress"]["percentage"] == 50.0
        assert "timestamp" in update


class TestStatusAndUtilities:
    """Test cases for status reporting and utility functions."""

    @patch("core.autonomous_react_engine.GeminiClient")
    @pytest.mark.asyncio
    async def test_get_autonomous_status(self, mock_client_class):
        """Test autonomous status reporting."""
        engine = AutonomousReactEngine(profile="test", autonomy_level=AutonomyLevel.GUIDED)

        status = await engine.get_autonomous_status()

        assert status["engine_info"]["profile"] == "test"
        assert status["engine_info"]["autonomy_level"] == "guided"
        assert "learning_data" in status
        assert "performance_metrics" in status
        assert "capabilities" in status

        # Verify capabilities
        capabilities = status["capabilities"]
        assert capabilities["goal_decomposition"] is True
        assert capabilities["autonomous_execution"] is True

    @patch("core.autonomous_react_engine.GeminiClient")
    def test_safe_model_dump_success(self, mock_client_class):
        """Test successful model serialization."""
        engine = AutonomousReactEngine()

        plan = AutonomousPlan(goal="test", steps=[])

        result = engine._safe_model_dump(plan)

        assert isinstance(result, dict)
        assert result["goal"] == "test"

    @patch("core.autonomous_react_engine.GeminiClient")
    def test_safe_model_dump_fallback(self, mock_client_class):
        """Test model serialization fallback."""
        engine = AutonomousReactEngine()

        # Create mock object that will fail model_dump
        mock_obj = Mock()
        mock_obj.model_dump.side_effect = Exception("Serialization failed")
        mock_obj.__dict__ = {"key": "value"}

        result = engine._safe_model_dump(mock_obj)

        assert isinstance(result, dict)
        assert result["key"] == "value"

    @patch("core.autonomous_react_engine.GeminiClient")
    def test_serialize_datetimes(self, mock_client_class):
        """Test datetime serialization utility."""
        engine = AutonomousReactEngine()

        now = datetime.now()
        data = {
            "timestamp": now,
            "nested": {"date": now},
            "list": [now, "string"],
            "string": "test",
        }

        result = engine._serialize_datetimes(data)

        assert isinstance(result["timestamp"], str)
        assert isinstance(result["nested"]["date"], str)
        assert isinstance(result["list"][0], str)
        assert result["string"] == "test"


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @patch("core.autonomous_react_engine.GeminiClient")
    def test_create_autonomous_engine(self, mock_client_class):
        """Test convenience function for creating engine."""
        engine = create_autonomous_engine(
            profile="personal", autonomy_level=AutonomyLevel.FULLY_AUTO
        )

        assert isinstance(engine, AutonomousReactEngine)
        assert engine.profile == "personal"
        assert engine.autonomy_level == AutonomyLevel.FULLY_AUTO


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=core.autonomous_react_engine", "--cov-report=term-missing"])
