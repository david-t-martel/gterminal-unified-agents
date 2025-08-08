"""
Tests for automation and orchestration modules.
Tests workflow orchestrator, migration agent, and related automation features.
"""

import asyncio
from pathlib import Path
import tempfile
from unittest.mock import Mock
from unittest.mock import patch

from app.automation.migration_agent import analyze_project_differences
from app.automation.migration_agent import migrate_auto_claude_updates
from app.automation.migration_agent import migrate_file
from app.automation.workflow_orchestrator import ErrorAction
from app.automation.workflow_orchestrator import StepResult
from app.automation.workflow_orchestrator import StepStatus
from app.automation.workflow_orchestrator import StepType
from app.automation.workflow_orchestrator import WorkflowDefinition
from app.automation.workflow_orchestrator import WorkflowOrchestrator
from app.automation.workflow_orchestrator import WorkflowResult
from app.automation.workflow_orchestrator import WorkflowStep
import pytest


@pytest.fixture
def sample_workflow_definition():
    """Create sample workflow definition for testing."""
    return {
        "id": "test_workflow",
        "name": "Test Workflow",
        "description": "A test workflow for validation",
        "version": "1.0.0",
        "steps": [
            {
                "id": "step1",
                "name": "Test Step 1",
                "type": "sequential",
                "action": "shell_command",
                "parameters": {"command": "echo 'Hello World'"},
                "timeout": 30,
            },
            {
                "id": "step2",
                "name": "Test Step 2",
                "type": "sequential",
                "action": "file_operation",
                "parameters": {
                    "operation": "write",
                    "path": "/tmp/test_file.txt",
                    "content": "Test content",
                },
                "depends_on": ["step1"],
            },
        ],
        "variables": {"env": "test"},
        "max_parallel": 3,
        "timeout": 300,
    }


@pytest.fixture
def complex_workflow_definition():
    """Create complex workflow definition with parallel and conditional steps."""
    return {
        "id": "complex_workflow",
        "name": "Complex Test Workflow",
        "description": "Complex workflow with parallel and conditional execution",
        "version": "2.0.0",
        "steps": [
            {
                "id": "init",
                "name": "Initialize",
                "type": "sequential",
                "action": "shell_command",
                "parameters": {"command": "echo 'Starting workflow'"},
            },
            {
                "id": "parallel_group",
                "name": "Parallel Tasks",
                "type": "parallel",
                "steps": [
                    {
                        "id": "task1",
                        "name": "Task 1",
                        "type": "sequential",
                        "action": "shell_command",
                        "parameters": {"command": "echo 'Task 1'"},
                    },
                    {
                        "id": "task2",
                        "name": "Task 2",
                        "type": "sequential",
                        "action": "shell_command",
                        "parameters": {"command": "echo 'Task 2'"},
                    },
                ],
                "depends_on": ["init"],
            },
            {
                "id": "conditional_step",
                "name": "Conditional Logic",
                "type": "conditional",
                "steps": [
                    {
                        "id": "branch1",
                        "name": "Branch 1",
                        "type": "sequential",
                        "action": "shell_command",
                        "parameters": {"command": "echo 'Branch 1'"},
                        "condition": "env == 'test'",
                    },
                    {
                        "id": "branch2",
                        "name": "Branch 2",
                        "type": "sequential",
                        "action": "shell_command",
                        "parameters": {"command": "echo 'Branch 2'"},
                        "condition": "env == 'prod'",
                    },
                ],
                "depends_on": ["parallel_group"],
            },
        ],
        "variables": {"env": "test", "debug": True},
    }


@pytest.fixture
def workflow_orchestrator():
    """Create workflow orchestrator instance."""
    return WorkflowOrchestrator()


@pytest.fixture
def temp_project_dirs():
    """Create temporary directories for migration testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = Path(temp_dir) / "source"
        target_dir = Path(temp_dir) / "target"

        source_dir.mkdir()
        target_dir.mkdir()

        # Create some test files
        (source_dir / "test.py").write_text("print('Hello from source')")
        (source_dir / "config.yaml").write_text("version: 1.0\nfeatures: [a, b, c]")
        (source_dir / "new_file.py").write_text("# New functionality")

        (target_dir / "test.py").write_text("print('Hello from target')")
        (target_dir / "config.yaml").write_text("version: 0.9\nfeatures: [a, b]")
        (target_dir / "old_file.py").write_text("# Old functionality")

        yield source_dir, target_dir


class TestWorkflowDefinition:
    """Test workflow definition models and validation."""

    def test_workflow_step_creation(self):
        """Test WorkflowStep model creation and validation."""
        step = WorkflowStep(
            id="test_step",
            name="Test Step",
            action="shell_command",
            parameters={"command": "echo test"},
        )

        assert step.id == "test_step"
        assert step.name == "Test Step"
        assert step.type == StepType.SEQUENTIAL
        assert step.action == "shell_command"
        assert step.parameters["command"] == "echo test"
        assert step.on_error == ErrorAction.FAIL
        assert step.retries == 0
        assert step.timeout == 60

    def test_workflow_step_with_substeps(self):
        """Test WorkflowStep with sub-steps."""
        parent_step = WorkflowStep(
            id="parent",
            name="Parent Step",
            type=StepType.PARALLEL,
            steps=[
                WorkflowStep(
                    id="child1",
                    name="Child 1",
                    action="shell_command",
                    parameters={"command": "echo child1"},
                ),
                WorkflowStep(
                    id="child2",
                    name="Child 2",
                    action="shell_command",
                    parameters={"command": "echo child2"},
                ),
            ],
        )

        assert parent_step.type == StepType.PARALLEL
        assert len(parent_step.steps) == 2
        assert parent_step.steps[0].id == "child1"
        assert parent_step.steps[1].id == "child2"

    def test_workflow_definition_creation(self, sample_workflow_definition):
        """Test WorkflowDefinition model creation."""
        workflow = WorkflowDefinition(**sample_workflow_definition)

        assert workflow.id == "test_workflow"
        assert workflow.name == "Test Workflow"
        assert workflow.version == "1.0.0"
        assert len(workflow.steps) == 2
        assert workflow.variables["env"] == "test"
        assert workflow.max_parallel == 3
        assert workflow.timeout == 300

    def test_workflow_definition_step_dependencies(self, sample_workflow_definition):
        """Test workflow step dependencies."""
        workflow = WorkflowDefinition(**sample_workflow_definition)

        step1 = workflow.steps[0]
        step2 = workflow.steps[1]

        assert len(step1.depends_on) == 0
        assert len(step2.depends_on) == 1
        assert step2.depends_on[0] == "step1"


class TestWorkflowResult:
    """Test workflow result models."""

    def test_step_result_creation(self):
        """Test StepResult creation and properties."""
        import time

        start_time = time.time()

        result = StepResult(
            step_id="test_step",
            status=StepStatus.SUCCESS,
            start_time=start_time,
            end_time=start_time + 5.0,
            output={"message": "success"},
        )

        assert result.step_id == "test_step"
        assert result.status == StepStatus.SUCCESS
        assert result.duration == 5.0
        assert result.output["message"] == "success"
        assert result.retries == 0

    def test_step_result_without_end_time(self):
        """Test StepResult without end time."""
        import time

        result = StepResult(
            step_id="running_step", status=StepStatus.RUNNING, start_time=time.time()
        )

        assert result.duration is None

    def test_workflow_result_creation(self):
        """Test WorkflowResult creation and properties."""
        import time

        start_time = time.time()

        result = WorkflowResult(
            workflow_id="test_workflow",
            status=StepStatus.SUCCESS,
            start_time=start_time,
            end_time=start_time + 10.0,
        )

        assert result.workflow_id == "test_workflow"
        assert result.status == StepStatus.SUCCESS
        assert result.duration == 10.0
        assert len(result.steps) == 0
        assert result.success_rate == 0.0

    def test_workflow_result_success_rate(self):
        """Test workflow result success rate calculation."""
        import time

        result = WorkflowResult(workflow_id="test_workflow", status=StepStatus.SUCCESS)

        # Add successful steps
        result.steps.append(
            StepResult(step_id="step1", status=StepStatus.SUCCESS, start_time=time.time())
        )
        result.steps.append(
            StepResult(step_id="step2", status=StepStatus.SUCCESS, start_time=time.time())
        )
        result.steps.append(
            StepResult(step_id="step3", status=StepStatus.FAILED, start_time=time.time())
        )

        # 2 out of 3 successful = 66.67%
        assert abs(result.success_rate - 66.67) < 0.01


class TestWorkflowOrchestrator:
    """Test WorkflowOrchestrator functionality."""

    def test_orchestrator_initialization(self, workflow_orchestrator):
        """Test workflow orchestrator initialization."""
        assert workflow_orchestrator.agent_name == "workflow-orchestrator"
        assert len(workflow_orchestrator.workflows) == 0
        assert len(workflow_orchestrator.running_workflows) == 0
        assert len(workflow_orchestrator.workflow_history) == 0
        assert len(workflow_orchestrator.actions) > 0

    def test_register_actions(self, workflow_orchestrator):
        """Test action registration."""
        actions = workflow_orchestrator.actions

        required_actions = [
            "shell_command",
            "http_request",
            "file_operation",
            "git_operation",
            "docker_operation",
            "test_runner",
            "deploy",
            "notify",
            "validate",
            "transform",
        ]

        for action in required_actions:
            assert action in actions
            assert callable(actions[action])

    @pytest.mark.asyncio
    async def test_create_workflow_tool(self, workflow_orchestrator, sample_workflow_definition):
        """Test create_workflow MCP tool."""
        # Mock MCP tool registration
        with patch.object(workflow_orchestrator, "mcp") as mock_mcp:
            workflow_orchestrator.register_tools()

            # Get the create_workflow function
            create_workflow_calls = list(mock_mcp.tool.call_args_list)
            assert len(create_workflow_calls) > 0

        # Test workflow creation directly
        workflow = WorkflowDefinition(**sample_workflow_definition)
        workflow_orchestrator.workflows[workflow.id] = workflow

        assert "test_workflow" in workflow_orchestrator.workflows
        assert workflow_orchestrator.workflows["test_workflow"].name == "Test Workflow"

    def test_condition_evaluation(self, workflow_orchestrator):
        """Test condition evaluation."""
        context = {"env": "test", "debug": True, "count": 5}

        # Test simple conditions
        assert workflow_orchestrator._evaluate_condition("env == 'test'", context) is True
        assert workflow_orchestrator._evaluate_condition("env == 'prod'", context) is False
        assert workflow_orchestrator._evaluate_condition("debug", context) is True
        assert workflow_orchestrator._evaluate_condition("count > 3", context) is True
        assert workflow_orchestrator._evaluate_condition("count < 3", context) is False

        # Test invalid condition
        assert workflow_orchestrator._evaluate_condition("invalid_var", context) is False

    def test_check_dependencies(self, workflow_orchestrator):
        """Test dependency checking."""
        # Create workflow result with completed steps
        import time

        workflow_result = WorkflowResult(
            workflow_id="test", status=StepStatus.RUNNING, start_time=time.time()
        )

        # Add completed steps
        workflow_result.steps.append(
            StepResult(step_id="step1", status=StepStatus.SUCCESS, start_time=time.time())
        )
        workflow_result.steps.append(
            StepResult(step_id="step2", status=StepStatus.SUCCESS, start_time=time.time())
        )

        # Test step with no dependencies
        step_no_deps = WorkflowStep(id="test", name="Test", action="shell_command")
        assert workflow_orchestrator._check_dependencies(step_no_deps, workflow_result) is True

        # Test step with met dependencies
        step_met_deps = WorkflowStep(
            id="test", name="Test", action="shell_command", depends_on=["step1", "step2"]
        )
        assert workflow_orchestrator._check_dependencies(step_met_deps, workflow_result) is True

        # Test step with unmet dependencies
        step_unmet_deps = WorkflowStep(
            id="test",
            name="Test",
            action="shell_command",
            depends_on=["step1", "step3"],  # step3 doesn't exist
        )
        assert workflow_orchestrator._check_dependencies(step_unmet_deps, workflow_result) is False

    def test_calculate_progress(self, workflow_orchestrator):
        """Test progress calculation."""
        import time

        # Empty workflow
        empty_result = WorkflowResult(
            workflow_id="empty", status=StepStatus.RUNNING, start_time=time.time()
        )
        assert workflow_orchestrator._calculate_progress(empty_result) == 0.0

        # Workflow with steps
        result = WorkflowResult(
            workflow_id="test", status=StepStatus.RUNNING, start_time=time.time()
        )

        # Add steps in various states
        result.steps.append(
            StepResult(step_id="step1", status=StepStatus.SUCCESS, start_time=time.time())
        )
        result.steps.append(
            StepResult(step_id="step2", status=StepStatus.FAILED, start_time=time.time())
        )
        result.steps.append(
            StepResult(step_id="step3", status=StepStatus.RUNNING, start_time=time.time())
        )
        result.steps.append(
            StepResult(step_id="step4", status=StepStatus.PENDING, start_time=time.time())
        )

        # 2 out of 4 completed (SUCCESS + FAILED) = 50%
        progress = workflow_orchestrator._calculate_progress(result)
        assert progress == 50.0


class TestWorkflowExecution:
    """Test workflow execution functionality."""

    @pytest.mark.asyncio
    async def test_simple_workflow_execution(
        self, workflow_orchestrator, sample_workflow_definition
    ):
        """Test execution of simple workflow."""
        workflow = WorkflowDefinition(**sample_workflow_definition)

        # Mock action execution
        with patch.object(workflow_orchestrator, "_action_shell_command") as mock_shell:
            with patch.object(workflow_orchestrator, "_action_file_operation") as mock_file:
                mock_shell.return_value = {"stdout": "Hello World", "returncode": 0}
                mock_file.return_value = {"created": "/tmp/test_file.txt"}

                result = await workflow_orchestrator._execute_workflow(workflow)

                assert result.workflow_id == "test_workflow"
                assert result.status == StepStatus.SUCCESS
                assert len(result.steps) == 2
                assert result.steps[0].status == StepStatus.SUCCESS
                assert result.steps[1].status == StepStatus.SUCCESS
                assert result.duration is not None

    @pytest.mark.asyncio
    async def test_parallel_workflow_execution(
        self, workflow_orchestrator, complex_workflow_definition
    ):
        """Test execution of workflow with parallel steps."""
        workflow = WorkflowDefinition(**complex_workflow_definition)

        # Mock action execution
        with patch.object(workflow_orchestrator, "_action_shell_command") as mock_shell:
            mock_shell.return_value = {"stdout": "output", "returncode": 0}

            result = await workflow_orchestrator._execute_workflow(workflow)

            assert result.workflow_id == "complex_workflow"
            assert result.status == StepStatus.SUCCESS

            # Should have executed init, parallel group (with 2 sub-steps), and conditional
            # The parallel group creates sub-steps
            assert len(result.steps) >= 3

    @pytest.mark.asyncio
    async def test_workflow_with_error_handling(self, workflow_orchestrator):
        """Test workflow execution with error handling."""
        workflow_def = {
            "id": "error_workflow",
            "name": "Error Handling Workflow",
            "steps": [
                {
                    "id": "failing_step",
                    "name": "Failing Step",
                    "action": "shell_command",
                    "parameters": {"command": "exit 1"},
                    "on_error": "continue",
                },
                {
                    "id": "success_step",
                    "name": "Success Step",
                    "action": "shell_command",
                    "parameters": {"command": "echo success"},
                    "depends_on": ["failing_step"],
                },
            ],
        }

        workflow = WorkflowDefinition(**workflow_def)

        with patch.object(workflow_orchestrator, "_action_shell_command") as mock_shell:
            # First call fails, second succeeds
            mock_shell.side_effect = [
                Exception("Command failed"),
                {"stdout": "success", "returncode": 0},
            ]

            result = await workflow_orchestrator._execute_workflow(workflow)

            assert result.workflow_id == "error_workflow"
            assert result.status == StepStatus.SUCCESS  # Overall success despite one failure
            assert len(result.steps) == 2
            assert result.steps[0].status == StepStatus.FAILED
            assert result.steps[1].status == StepStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_workflow_with_retry(self, workflow_orchestrator):
        """Test workflow execution with retry logic."""
        workflow_def = {
            "id": "retry_workflow",
            "name": "Retry Workflow",
            "steps": [
                {
                    "id": "retry_step",
                    "name": "Retry Step",
                    "action": "shell_command",
                    "parameters": {"command": "echo test"},
                    "on_error": "retry",
                    "retries": 2,
                }
            ],
        }

        workflow = WorkflowDefinition(**workflow_def)

        with patch.object(workflow_orchestrator, "_action_shell_command") as mock_shell:
            # Fail twice, then succeed
            mock_shell.side_effect = [
                Exception("First failure"),
                Exception("Second failure"),
                {"stdout": "success", "returncode": 0},
            ]

            result = await workflow_orchestrator._execute_workflow(workflow)

            assert result.workflow_id == "retry_workflow"
            assert result.status == StepStatus.SUCCESS
            assert len(result.steps) == 1
            assert result.steps[0].status == StepStatus.SUCCESS
            assert result.steps[0].retries == 2


class TestWorkflowActions:
    """Test individual workflow actions."""

    @pytest.mark.asyncio
    async def test_shell_command_action(self, workflow_orchestrator):
        """Test shell command action."""
        with patch.object(workflow_orchestrator, "run_command") as mock_run:
            mock_result = Mock()
            mock_result.stdout = "Hello World"
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = await workflow_orchestrator._action_shell_command(
                {"command": "echo 'Hello World'", "timeout": 30}, {}
            )

            assert result["stdout"] == "Hello World"
            assert result["returncode"] == 0
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_file_operation_action(self, workflow_orchestrator):
        """Test file operation action."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"

            # Test write operation
            result = await workflow_orchestrator._action_file_operation(
                {"operation": "write", "path": str(test_file), "content": "Test content"}, {}
            )

            assert test_file.exists()
            assert test_file.read_text() == "Test content"

            # Test read operation
            with patch.object(workflow_orchestrator, "safe_file_read", return_value="Test content"):
                result = await workflow_orchestrator._action_file_operation(
                    {"operation": "read", "path": str(test_file)}, {}
                )

                assert result == "Test content"

            # Test create directory operation
            test_dir = Path(temp_dir) / "new_dir"
            result = await workflow_orchestrator._action_file_operation(
                {"operation": "create_dir", "path": str(test_dir)}, {}
            )

            assert test_dir.exists()
            assert test_dir.is_dir()

    @pytest.mark.asyncio
    async def test_git_operation_action(self, workflow_orchestrator):
        """Test git operation action."""
        with patch.object(workflow_orchestrator, "run_command") as mock_run:
            mock_result = Mock()
            mock_result.stdout = "Already up to date."
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            # Test git pull
            result = await workflow_orchestrator._action_git_operation({"operation": "pull"}, {})

            assert result["stdout"] == "Already up to date."
            assert result["returncode"] == 0
            mock_run.assert_called_with(["git", "pull"])

    @pytest.mark.asyncio
    async def test_test_runner_action(self, workflow_orchestrator):
        """Test test runner action."""
        with patch.object(workflow_orchestrator, "run_command") as mock_run:
            mock_result = Mock()
            mock_result.stdout = "2 passed, 0 failed"
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = await workflow_orchestrator._action_test_runner(
                {"type": "unit", "framework": "pytest"}, {}
            )

            assert result["passed"] is True
            assert result["output"] == "2 passed, 0 failed"

    @pytest.mark.asyncio
    async def test_notify_action(self, workflow_orchestrator):
        """Test notification action."""
        result = await workflow_orchestrator._action_notify(
            {"channel": "console", "message": "Test notification", "details": {"key": "value"}}, {}
        )

        assert result["sent"] is True
        assert result["channel"] == "console"


class TestMigrationAgent:
    """Test migration agent functionality."""

    @pytest.mark.asyncio
    async def test_analyze_project_differences(self, temp_project_dirs):
        """Test project difference analysis."""
        source_dir, target_dir = temp_project_dirs

        with patch("app.automation.migration_agent.get_model_for_task") as mock_get_model:
            mock_model = Mock()
            mock_model.generate_content.return_value.text = (
                "Analysis: Found differences in configuration and new files."
            )
            mock_get_model.return_value = mock_model

            result = await analyze_project_differences(
                str(source_dir), str(target_dir), ["*.py", "*.yaml"]
            )

            assert "new_files" in result
            assert "modified_files" in result
            assert "deleted_files" in result
            assert "analysis" in result

            # Should find new_file.py as new
            assert "new_file.py" in result["new_files"]

            # Should find old_file.py as deleted
            assert "old_file.py" in result["deleted_files"]

            # Should find test.py and config.yaml as modified
            modified_files = result["modified_files"]
            assert any("test.py" in str(f) for f in modified_files)
            assert any("config.yaml" in str(f) for f in modified_files)

    @pytest.mark.asyncio
    async def test_migrate_file_new_file(self, temp_project_dirs):
        """Test migrating a new file."""
        source_dir, target_dir = temp_project_dirs

        source_file = source_dir / "new_file.py"
        target_file = target_dir / "new_file.py"

        result = await migrate_file(str(source_file), str(target_file))

        assert result["status"] == "created"
        assert target_file.exists()
        assert target_file.read_text() == "# New functionality"

    @pytest.mark.asyncio
    async def test_migrate_file_identical(self, temp_project_dirs):
        """Test migrating identical files."""
        source_dir, target_dir = temp_project_dirs

        # Create identical files
        identical_content = "# Identical content"
        (source_dir / "identical.py").write_text(identical_content)
        (target_dir / "identical.py").write_text(identical_content)

        result = await migrate_file(
            str(source_dir / "identical.py"), str(target_dir / "identical.py")
        )

        assert result["status"] == "unchanged"

    @pytest.mark.asyncio
    async def test_migrate_file_overwrite(self, temp_project_dirs):
        """Test migrating with overwrite strategy."""
        source_dir, target_dir = temp_project_dirs

        source_file = source_dir / "test.py"
        target_file = target_dir / "test.py"

        original_source_content = source_file.read_text()

        result = await migrate_file(str(source_file), str(target_file), "overwrite")

        assert result["status"] == "overwritten"
        assert target_file.read_text() == original_source_content

    @pytest.mark.asyncio
    async def test_migrate_file_intelligent_merge(self, temp_project_dirs):
        """Test migrating with intelligent merge."""
        source_dir, target_dir = temp_project_dirs

        source_file = source_dir / "test.py"
        target_file = target_dir / "test.py"

        with patch("app.automation.migration_agent.get_model_for_task") as mock_get_model:
            mock_model = Mock()
            # Mock response with merged content
            mock_model.generate_content.return_value.text = """
            The files have different print statements. Here's the merged version:

            ```python
            print('Hello from merged version')
            ```
            """
            mock_get_model.return_value = mock_model

            result = await migrate_file(str(source_file), str(target_file), "intelligent")

            assert result["status"] == "merged"
            assert "backup" in result
            assert target_file.read_text() == "print('Hello from merged version')"

            # Check backup was created
            backup_path = Path(result["backup"])
            assert backup_path.exists()

    @pytest.mark.asyncio
    async def test_migrate_file_nonexistent_source(self, temp_project_dirs):
        """Test migrating non-existent source file."""
        source_dir, target_dir = temp_project_dirs

        result = await migrate_file(
            str(source_dir / "nonexistent.py"), str(target_dir / "target.py")
        )

        assert "error" in result
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_migrate_auto_claude_updates(self, temp_project_dirs):
        """Test auto-claude specific migration."""
        source_dir, target_dir = temp_project_dirs

        # Create auto-claude related files
        (source_dir / "auto-claude-config.yaml").write_text("version: 2.0")
        (source_dir / "scripts" / "auto-claude-setup.py").touch()
        (source_dir / "scripts").mkdir(exist_ok=True)
        (source_dir / "scripts" / "auto-claude-setup.py").write_text("# Setup script")

        with patch("app.automation.migration_agent.get_model_for_task") as mock_get_model:
            mock_model = Mock()

            # Mock project analysis
            mock_model.generate_content.side_effect = [
                # First call: identify files
                Mock(text='["auto-claude-config.yaml", "scripts/auto-claude-setup.py"]'),
                # Second call: migration summary
                Mock(text="Successfully migrated auto-claude configuration and setup script."),
            ]
            mock_get_model.return_value = mock_model

            # Mock analyze_project_differences
            with patch(
                "app.automation.migration_agent.analyze_project_differences"
            ) as mock_analyze:
                mock_analyze.return_value = {
                    "new_files": ["auto-claude-config.yaml", "scripts/auto-claude-setup.py"],
                    "modified_files": [],
                    "deleted_files": [],
                }

                result = await migrate_auto_claude_updates(str(source_dir), str(target_dir))

                assert "files_migrated" in result
                assert "migration_results" in result
                assert "summary" in result
                assert result["files_migrated"] >= 0


class TestWorkflowIntegration:
    """Test workflow and migration integration."""

    @pytest.mark.asyncio
    async def test_migration_workflow(self, workflow_orchestrator, temp_project_dirs):
        """Test migration as part of workflow."""
        source_dir, target_dir = temp_project_dirs

        # Create migration workflow
        migration_workflow = {
            "id": "migration_workflow",
            "name": "Project Migration Workflow",
            "steps": [
                {
                    "id": "analyze",
                    "name": "Analyze Differences",
                    "action": "shell_command",
                    "parameters": {"command": f"echo 'Analyzing {source_dir} vs {target_dir}'"},
                },
                {
                    "id": "backup",
                    "name": "Create Backup",
                    "action": "file_operation",
                    "parameters": {"operation": "create_dir", "path": str(target_dir / "backup")},
                    "depends_on": ["analyze"],
                },
                {
                    "id": "migrate",
                    "name": "Migrate Files",
                    "action": "shell_command",
                    "parameters": {"command": "echo 'Migration completed'"},
                    "depends_on": ["backup"],
                },
            ],
        }

        workflow = WorkflowDefinition(**migration_workflow)

        with patch.object(workflow_orchestrator, "_action_shell_command") as mock_shell:
            with patch.object(workflow_orchestrator, "_action_file_operation") as mock_file:
                mock_shell.return_value = {"stdout": "success", "returncode": 0}
                mock_file.return_value = {"created": str(target_dir / "backup")}

                result = await workflow_orchestrator._execute_workflow(workflow)

                assert result.status == StepStatus.SUCCESS
                assert len(result.steps) == 3
                assert all(step.status == StepStatus.SUCCESS for step in result.steps)

    @pytest.mark.asyncio
    async def test_ci_cd_workflow(self, workflow_orchestrator):
        """Test CI/CD workflow with testing and deployment."""
        cicd_workflow = {
            "id": "cicd_workflow",
            "name": "CI/CD Pipeline",
            "steps": [
                {
                    "id": "checkout",
                    "name": "Checkout Code",
                    "action": "git_operation",
                    "parameters": {"operation": "pull"},
                },
                {
                    "id": "test",
                    "name": "Run Tests",
                    "action": "test_runner",
                    "parameters": {"type": "unit", "framework": "pytest"},
                    "depends_on": ["checkout"],
                },
                {
                    "id": "deploy",
                    "name": "Deploy Application",
                    "action": "deploy",
                    "parameters": {"environment": "staging", "strategy": "rolling"},
                    "depends_on": ["test"],
                },
                {
                    "id": "notify",
                    "name": "Send Notification",
                    "action": "notify",
                    "parameters": {
                        "channel": "console",
                        "message": "Deployment completed successfully",
                    },
                    "depends_on": ["deploy"],
                },
            ],
        }

        workflow = WorkflowDefinition(**cicd_workflow)

        with patch.object(workflow_orchestrator, "_action_git_operation") as mock_git:
            with patch.object(workflow_orchestrator, "_action_test_runner") as mock_test:
                with patch.object(workflow_orchestrator, "_action_deploy") as mock_deploy:
                    with patch.object(workflow_orchestrator, "_action_notify") as mock_notify:
                        mock_git.return_value = {"stdout": "Already up to date.", "returncode": 0}
                        mock_test.return_value = {"passed": True, "output": "All tests passed"}
                        mock_deploy.return_value = {"status": "deployed", "environment": "staging"}
                        mock_notify.return_value = {"sent": True, "channel": "console"}

                        result = await workflow_orchestrator._execute_workflow(workflow)

                        assert result.status == StepStatus.SUCCESS
                        assert len(result.steps) == 4
                        assert result.success_rate == 100.0


class TestWorkflowErrorScenarios:
    """Test workflow error scenarios and recovery."""

    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, workflow_orchestrator):
        """Test workflow timeout handling."""
        timeout_workflow = {
            "id": "timeout_workflow",
            "name": "Timeout Test Workflow",
            "timeout": 1,  # 1 second timeout
            "steps": [
                {
                    "id": "long_running",
                    "name": "Long Running Task",
                    "action": "shell_command",
                    "parameters": {"command": "sleep 10"},
                    "timeout": 1,
                }
            ],
        }

        workflow = WorkflowDefinition(**timeout_workflow)

        with patch.object(workflow_orchestrator, "_action_shell_command") as mock_shell:
            # Simulate long-running command
            async def slow_command(*args, **kwargs):
                await asyncio.sleep(2)
                return {"stdout": "completed", "returncode": 0}

            mock_shell.side_effect = slow_command

            # This should timeout or fail gracefully
            result = await workflow_orchestrator._execute_workflow(workflow)

            # Workflow should handle the timeout
            assert result.workflow_id == "timeout_workflow"

    @pytest.mark.asyncio
    async def test_workflow_with_invalid_action(self, workflow_orchestrator):
        """Test workflow with invalid action."""
        invalid_workflow = {
            "id": "invalid_workflow",
            "name": "Invalid Action Workflow",
            "steps": [
                {
                    "id": "invalid_action",
                    "name": "Invalid Action",
                    "action": "nonexistent_action",
                    "parameters": {},
                }
            ],
        }

        workflow = WorkflowDefinition(**invalid_workflow)

        result = await workflow_orchestrator._execute_workflow(workflow)

        assert result.status == StepStatus.FAILED
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.FAILED
        assert "unknown action" in result.steps[0].error.lower()

    @pytest.mark.asyncio
    async def test_workflow_partial_failure_recovery(self, workflow_orchestrator):
        """Test workflow recovery from partial failures."""
        recovery_workflow = {
            "id": "recovery_workflow",
            "name": "Recovery Test Workflow",
            "steps": [
                {
                    "id": "step1",
                    "name": "First Step",
                    "action": "shell_command",
                    "parameters": {"command": "echo step1"},
                },
                {
                    "id": "step2",
                    "name": "Failing Step",
                    "action": "shell_command",
                    "parameters": {"command": "exit 1"},
                    "on_error": "continue",
                    "depends_on": ["step1"],
                },
                {
                    "id": "step3",
                    "name": "Recovery Step",
                    "action": "shell_command",
                    "parameters": {"command": "echo recovery"},
                    "depends_on": ["step1"],  # Depends on step1, not step2
                },
            ],
        }

        workflow = WorkflowDefinition(**recovery_workflow)

        with patch.object(workflow_orchestrator, "_action_shell_command") as mock_shell:
            mock_shell.side_effect = [
                {"stdout": "step1", "returncode": 0},  # step1 succeeds
                Exception("Command failed"),  # step2 fails
                {"stdout": "recovery", "returncode": 0},  # step3 succeeds
            ]

            result = await workflow_orchestrator._execute_workflow(workflow)

            assert result.status == StepStatus.SUCCESS  # Overall success despite one failure
            assert len(result.steps) == 3
            assert result.steps[0].status == StepStatus.SUCCESS
            assert result.steps[1].status == StepStatus.FAILED
            assert result.steps[2].status == StepStatus.SUCCESS
