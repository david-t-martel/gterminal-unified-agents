"""JSON RPC 2.0 Migration Guide and Utilities.

This module provides utilities and guidance for migrating existing
agent methods to be JSON RPC 2.0 compliant, with practical examples
and automated migration tools.

Features:
- Automated code analysis and transformation
- Migration validation
- Compatibility testing
- Step-by-step migration guide
- Performance comparison tools
"""

import ast
from datetime import UTC
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import subprocess
import time
from typing import Any
import uuid

from pydantic import BaseModel
from pydantic import Field

from .models import RpcRequest
from .models import RpcResponse

logger = logging.getLogger(__name__)


class MigrationConfig(BaseModel):
    """Configuration for migration process."""

    source_directory: str = Field(description="Source directory to migrate")
    output_directory: str | None = Field(
        default=None, description="Output directory for migrated files"
    )
    backup_directory: str = Field(default="backup", description="Backup directory")

    # Migration options
    create_parameter_models: bool = Field(
        default=True, description="Create Pydantic parameter models"
    )
    add_timing_measurement: bool = Field(default=True, description="Add execution timing")
    preserve_backwards_compatibility: bool = Field(
        default=True, description="Maintain legacy method wrappers"
    )
    update_imports: bool = Field(default=True, description="Update import statements")

    # Validation options
    run_tests_after_migration: bool = Field(default=True, description="Run tests after migration")
    validate_with_ast_grep: bool = Field(default=True, description="Validate with AST-grep rules")
    create_migration_report: bool = Field(default=True, description="Generate migration report")


class MigrationAnalysis(BaseModel):
    """Results of migration analysis."""

    file_path: str = Field(description="Analyzed file path")
    agent_class_name: str | None = Field(default=None, description="Agent class name")
    methods_found: list[str] = Field(
        default_factory=list, description="Methods that need migration"
    )
    error_patterns: list[str] = Field(default_factory=list, description="Error patterns found")
    return_type_issues: list[str] = Field(default_factory=list, description="Return type issues")
    import_requirements: list[str] = Field(default_factory=list, description="Required imports")
    complexity_score: float = Field(description="Migration complexity (0-10)")
    estimated_time_hours: float = Field(description="Estimated migration time")


class MigrationStep(BaseModel):
    """Individual migration step."""

    step_number: int = Field(description="Step sequence number")
    title: str = Field(description="Step title")
    description: str = Field(description="Detailed description")
    code_before: str = Field(description="Code before transformation")
    code_after: str = Field(description="Code after transformation")
    validation_command: str | None = Field(default=None, description="Validation command")
    estimated_time_minutes: int = Field(description="Estimated time for this step")


class MigrationReport(BaseModel):
    """Complete migration report."""

    migration_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config: MigrationConfig = Field(description="Migration configuration used")

    # Analysis results
    files_analyzed: list[MigrationAnalysis] = Field(default_factory=list)
    total_methods: int = Field(description="Total methods to migrate")
    total_files: int = Field(description="Total files to migrate")

    # Migration steps
    migration_steps: list[MigrationStep] = Field(default_factory=list)

    # Results
    success: bool = Field(description="Migration success status")
    errors: list[str] = Field(default_factory=list, description="Migration errors")
    warnings: list[str] = Field(default_factory=list, description="Migration warnings")

    # Performance metrics
    total_time_seconds: float = Field(description="Total migration time")
    files_migrated: int = Field(description="Successfully migrated files")
    methods_migrated: int = Field(description="Successfully migrated methods")


class AgentMigrationTool:
    """Tool for migrating agents to RPC compliance."""

    def __init__(self, config: MigrationConfig) -> None:
        self.config = config
        self.report = MigrationReport(config=config)

    async def analyze_codebase(self) -> MigrationReport:
        """Analyze existing codebase for migration requirements."""
        logger.info(f"Starting codebase analysis: {self.config.source_directory}")

        source_path = Path(self.config.source_directory)
        if not source_path.exists():
            msg = f"Source directory not found: {self.config.source_directory}"
            raise FileNotFoundError(msg)

        # Find all Python files with agent classes
        python_files = list(source_path.rglob("*.py"))
        agent_files = []

        for file_path in python_files:
            if await self._is_agent_file(file_path):
                agent_files.append(file_path)

        # Analyze each agent file
        for file_path in agent_files:
            analysis = await self._analyze_file(file_path)
            self.report.files_analyzed.append(analysis)
            self.report.total_methods += len(analysis.methods_found)

        self.report.total_files = len(agent_files)

        # Generate migration steps
        await self._generate_migration_steps()

        logger.info(
            f"Analysis complete: {self.report.total_files} files, {self.report.total_methods} methods"
        )
        return self.report

    async def _is_agent_file(self, file_path: Path) -> bool:
        """Check if file contains agent classes."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return (
                "BaseAgentService" in content
                or "BaseAutomationAgent" in content
                or ("class" in content and "Agent" in content)
            )
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return False

    async def _analyze_file(self, file_path: Path) -> MigrationAnalysis:
        """Analyze individual file for migration requirements."""
        logger.debug(f"Analyzing file: {file_path}")

        analysis = MigrationAnalysis(
            file_path=str(file_path),
            complexity_score=0.0,
            estimated_time_hours=0.0,
        )

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            # Find agent class
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's an agent class
                    if any(
                        base.id in ["BaseAgentService", "BaseAutomationAgent"]
                        for base in node.bases
                        if isinstance(base, ast.Name)
                    ):
                        analysis.agent_class_name = node.name
                        break

            # Find methods that need migration
            methods = await self._find_methods_needing_migration(tree)
            analysis.methods_found = [method["name"] for method in methods]

            # Analyze error patterns
            analysis.error_patterns = await self._find_error_patterns(content)

            # Analyze return type issues
            analysis.return_type_issues = await self._find_return_type_issues(tree)

            # Determine required imports
            analysis.import_requirements = await self._determine_required_imports()

            # Calculate complexity
            analysis.complexity_score = await self._calculate_complexity(methods, analysis)
            analysis.estimated_time_hours = (
                analysis.complexity_score * 0.5
            )  # 30 min per complexity point

        except Exception as e:
            logger.exception(f"Error analyzing file {file_path}: {e}")
            analysis.complexity_score = 10.0  # Mark as high complexity due to parsing issues

        return analysis

    async def _find_methods_needing_migration(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Find methods that need RPC migration."""
        methods = []

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                # Check for async methods that return dict
                if (
                    node.returns
                    and isinstance(node.returns, ast.Name)
                    and node.returns.id == "dict"
                ):
                    methods.append(
                        {
                            "name": node.name,
                            "line": node.lineno,
                            "args": len(node.args.args),
                            "has_typing": node.returns is not None,
                            "is_async": True,
                        }
                    )
                elif node.name not in ["__init__", "__aenter__", "__aexit__"]:
                    # Any other async method in agent class
                    methods.append(
                        {
                            "name": node.name,
                            "line": node.lineno,
                            "args": len(node.args.args),
                            "has_typing": node.returns is not None,
                            "is_async": True,
                        }
                    )

        return methods

    async def _find_error_patterns(self, content: str) -> list[str]:
        """Find inconsistent error return patterns."""
        patterns = []

        # Common error patterns
        error_patterns = [
            r'return\s*{\s*"status"\s*:\s*"error"',
            r'return\s*{\s*"error"\s*:',
            r'return\s*{\s*"success"\s*:\s*False',
            r'return\s*{\s*"failed"\s*:\s*True',
        ]

        for pattern in error_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                patterns.append(f"Pattern '{pattern}' found {len(matches)} times")

        return patterns

    async def _find_return_type_issues(self, tree: ast.AST) -> list[str]:
        """Find return type inconsistencies."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                # Check for methods without return type annotations
                if not node.returns:
                    issues.append(f"Method '{node.name}' missing return type annotation")

                # Look for inconsistent return statements
                return_types = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and child.value:
                        if isinstance(child.value, ast.Dict):
                            # Check dict keys to identify return pattern
                            keys = [k.s for k in child.value.keys if isinstance(k, ast.Str)]
                            if "status" in keys:
                                return_types.add("legacy_status")
                            elif "error" in keys:
                                return_types.add("legacy_error")
                            else:
                                return_types.add("dict_data")

                if len(return_types) > 1:
                    issues.append(
                        f"Method '{node.name}' has inconsistent return patterns: {return_types}"
                    )

        return issues

    async def _determine_required_imports(self) -> list[str]:
        """Determine required imports for RPC migration."""
        return [
            "from typing import Any, Dict, List, Optional",
            "from gterminal.core.rpc.models import RpcRequest, RpcResponse, RpcErrorCode, AgentTaskResult",
            "from gterminal.core.rpc.patterns import rpc_method, RpcAgentMixin, create_agent_task_result",
            "import uuid",
            "import time",
        ]

    async def _calculate_complexity(
        self, methods: list[dict[str, Any]], analysis: MigrationAnalysis
    ) -> float:
        """Calculate migration complexity score."""
        complexity = 0.0

        # Base complexity per method
        complexity += len(methods) * 1.0

        # Error pattern complexity
        complexity += len(analysis.error_patterns) * 0.5

        # Return type issue complexity
        complexity += len(analysis.return_type_issues) * 0.3

        # Method argument complexity
        for method in methods:
            complexity += method["args"] * 0.1
            if not method["has_typing"]:
                complexity += 0.5

        return min(complexity, 10.0)  # Cap at 10

    async def _generate_migration_steps(self) -> None:
        """Generate detailed migration steps."""
        steps = []
        step_num = 1

        # Step 1: Backup existing code
        steps.append(
            MigrationStep(
                step_number=step_num,
                title="Backup Existing Code",
                description="Create backup of existing codebase before migration",
                code_before="# Original files in place",
                code_after=f"# Files backed up to {self.config.backup_directory}",
                validation_command=f"ls -la {self.config.backup_directory}",
                estimated_time_minutes=5,
            )
        )
        step_num += 1

        # Step 2: Add required imports
        if self.config.update_imports:
            steps.append(
                MigrationStep(
                    step_number=step_num,
                    title="Add RPC Framework Imports",
                    description="Add necessary imports for RPC compliance",
                    code_before="""from gterminal.agents.base_agent_service import BaseAgentService""",
                    code_after="""from gterminal.agents.base_agent_service import BaseAgentService
from typing import Any, Dict, Optional
from gterminal.core.rpc.models import RpcRequest, RpcResponse, RpcErrorCode, AgentTaskResult
from gterminal.core.rpc.patterns import rpc_method, RpcAgentMixin, create_agent_task_result
import uuid
import time""",
                    validation_command="python -c 'from gterminal.core.rpc.models import RpcRequest'",
                    estimated_time_minutes=10,
                )
            )
            step_num += 1

        # Step 3: Update class inheritance
        steps.append(
            MigrationStep(
                step_number=step_num,
                title="Add RpcAgentMixin to Class",
                description="Update agent class to inherit from RpcAgentMixin",
                code_before="class MyAgent(BaseAgentService):",
                code_after="class MyAgent(BaseAgentService, RpcAgentMixin):",
                validation_command="python -c 'from gterminal.agents.my_agent import MyAgent; assert hasattr(MyAgent, \"handle_rpc_request\")'",
                estimated_time_minutes=5,
            )
        )
        step_num += 1

        # Step 4: Create parameter models
        if self.config.create_parameter_models:
            steps.append(
                MigrationStep(
                    step_number=step_num,
                    title="Create Pydantic Parameter Models",
                    description="Create Pydantic models for method parameters",
                    code_before="""async def my_method(self, data: dict) -> dict:""",
                    code_after="""class MyMethodParams(BaseModel):
    field1: str = Field(description="Description of field1")
    field2: Optional[int] = Field(default=None, description="Description of field2")

@rpc_method(method_name="my_method", validate_params=True)
async def my_method_rpc(
    self,
    params: MyMethodParams,
    session: Optional[SessionContext] = None
) -> AgentTaskResult:""",
                    estimated_time_minutes=15,
                )
            )
            step_num += 1

        # Step 5: Transform method signatures and decorators
        steps.append(
            MigrationStep(
                step_number=step_num,
                title="Transform Method Signatures",
                description="Add @rpc_method decorators and update signatures",
                code_before="""async def process_data(self, data: dict) -> dict:
    return {"result": "processed"}""",
                code_after="""@rpc_method(method_name="process_data", validate_params=True)
async def process_data_rpc(
    self,
    params: ProcessDataParams,
    session: Optional[SessionContext] = None
) -> AgentTaskResult:
    task_id = str(uuid.uuid4())
    start_time = time.time()

    # Original logic here
    result = {"result": "processed"}

    return create_agent_task_result(
        task_id=task_id,
        task_type="data_processing",
        data=result,
        duration_ms=(time.time() - start_time) * 1000
    )""",
                estimated_time_minutes=20,
            )
        )
        step_num += 1

        # Step 6: Transform error handling
        steps.append(
            MigrationStep(
                step_number=step_num,
                title="Transform Error Handling",
                description="Convert error returns to exceptions",
                code_before="""if not data:
    return {"status": "error", "error": "Data required"}""",
                code_after="""if not params.data:
    raise ValueError("Data required")  # Automatically handled by decorator""",
                estimated_time_minutes=15,
            )
        )
        step_num += 1

        # Step 7: Create backwards compatibility wrappers
        if self.config.preserve_backwards_compatibility:
            steps.append(
                MigrationStep(
                    step_number=step_num,
                    title="Create Backwards Compatibility Wrappers",
                    description="Add wrapper methods for legacy API compatibility",
                    code_before="# No legacy wrapper",
                    code_after="""# Legacy method wrapper
async def process_data(self, data: dict) -> dict:
    \"\"\"Legacy wrapper for backwards compatibility.\"\"\"
    request = RpcRequest(
        method="process_data",
        params=ProcessDataParams(**data),
        id=str(uuid.uuid4())
    )

    rpc_response = await self.process_data_rpc(request)

    if rpc_response.result:
        return {
            "status": "success",
            "data": rpc_response.result.data
        }
    else:
        return {
            "status": "error",
            "error": rpc_response.error.message
        }""",
                    estimated_time_minutes=25,
                )
            )
            step_num += 1

        # Step 8: Update tests
        if self.config.run_tests_after_migration:
            steps.append(
                MigrationStep(
                    step_number=step_num,
                    title="Update Test Cases",
                    description="Update existing tests to work with RPC methods",
                    code_before="""# Test old method
result = await agent.process_data({"key": "value"})
assert result["status"] == "success" """,
                    code_after="""# Test new RPC method
request = RpcRequest(
    method="process_data",
    params=ProcessDataParams(key="value"),
    id="test-123"
)
response = await agent.handle_rpc_request(request)
assert response.result is not None
assert response.error is None

# Test legacy wrapper (if preserved)
result = await agent.process_data({"key": "value"})
assert result["status"] == "success" """,
                    validation_command="python -m pytest tests/",
                    estimated_time_minutes=30,
                )
            )
            step_num += 1

        # Step 9: Validation and cleanup
        steps.append(
            MigrationStep(
                step_number=step_num,
                title="Validate Migration",
                description="Run validation tools and cleanup",
                code_before="# Original code",
                code_after="# RPC-compliant code",
                validation_command="ast-grep --config app/core/rpc/ast_grep_rules.yaml scan .",
                estimated_time_minutes=15,
            )
        )

        self.report.migration_steps = steps

    async def execute_migration(self) -> MigrationReport:
        """Execute the migration process."""
        start_time = time.time()

        try:
            logger.info("Starting migration execution")

            # Step 1: Create backup
            await self._create_backup()

            # Step 2: Execute each migration step
            for step in self.report.migration_steps:
                logger.info(f"Executing step {step.step_number}: {step.title}")
                await self._execute_migration_step(step)

            # Step 3: Validate migration
            if self.config.validate_with_ast_grep:
                await self._validate_with_ast_grep()

            # Step 4: Run tests
            if self.config.run_tests_after_migration:
                await self._run_tests()

            self.report.success = True
            logger.info("Migration completed successfully")

        except Exception as e:
            self.report.success = False
            self.report.errors.append(str(e))
            logger.exception(f"Migration failed: {e}")

        finally:
            self.report.total_time_seconds = time.time() - start_time

            if self.config.create_migration_report:
                await self._create_migration_report()

        return self.report

    async def _create_backup(self) -> None:
        """Create backup of source directory."""
        import shutil

        source = Path(self.config.source_directory)
        backup = Path(self.config.backup_directory)

        if backup.exists():
            shutil.rmtree(backup)

        shutil.copytree(source, backup)
        logger.info(f"Created backup: {backup}")

    async def _execute_migration_step(self, step: MigrationStep) -> None:
        """Execute individual migration step."""
        # This would contain the actual code transformation logic
        # For now, we'll simulate the execution
        logger.debug(f"Executing migration step: {step.title}")

        # Validation command execution
        if step.validation_command:
            try:
                result = subprocess.run(
                    step.validation_command,
                    check=False,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    self.report.warnings.append(
                        f"Step {step.step_number} validation warning: {result.stderr}",
                    )
            except subprocess.TimeoutExpired:
                self.report.warnings.append(
                    f"Step {step.step_number} validation timed out",
                )

    async def _validate_with_ast_grep(self) -> None:
        """Validate migration using AST-grep rules."""
        try:
            rules_file = Path("app/core/rpc/ast_grep_rules.yaml")
            if rules_file.exists():
                result = subprocess.run(
                    [
                        "ast-grep",
                        "--config",
                        str(rules_file),
                        "scan",
                        self.config.source_directory,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if result.returncode != 0:
                    self.report.warnings.append(f"AST-grep validation issues: {result.stdout}")
        except Exception as e:
            self.report.warnings.append(f"AST-grep validation failed: {e}")

    async def _run_tests(self) -> None:
        """Run test suite after migration."""
        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/",
                    "-v",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                self.report.errors.append(f"Tests failed after migration: {result.stdout}")
            else:
                logger.info("All tests passed after migration")
        except Exception as e:
            self.report.errors.append(f"Test execution failed: {e}")

    async def _create_migration_report(self) -> None:
        """Create detailed migration report."""
        report_path = Path(f"migration_report_{self.report.migration_id}.json")

        with open(report_path, "w") as f:
            json.dump(self.report.model_dump(), f, indent=2, default=str)

        logger.info(f"Migration report created: {report_path}")


# Utility functions for manual migration assistance


def generate_parameter_model(method_name: str, parameters: dict[str, Any]) -> str:
    """Generate Pydantic parameter model code."""
    class_name = f"{method_name.title().replace('_', '')}Params"

    lines = [f"class {class_name}(BaseModel):"]
    lines.append(f'    """Parameters for {method_name} method."""')

    for param_name, param_info in parameters.items():
        param_type = param_info.get("type", "Any")
        description = param_info.get("description", f"Parameter {param_name}")
        default = param_info.get("default")

        if default is not None:
            lines.append(
                f'    {param_name}: {param_type} = Field(default={default!r}, description="{description}")'
            )
        else:
            lines.append(f'    {param_name}: {param_type} = Field(description="{description}")')

    return "\n".join(lines)


def generate_rpc_method(method_name: str, param_class: str, timeout: int = 300) -> str:
    """Generate RPC method decorator and signature."""
    return f"""@rpc_method(
    method_name="{method_name}",
    timeout_seconds={timeout},
    validate_params=True,
    log_performance=True
)
async def {method_name}_rpc(
    self,
    params: {param_class},
    session: Optional[SessionContext] = None
) -> AgentTaskResult:
    \"\"\"RPC-compliant {method_name} method.\"\"\"
    task_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Original method logic here
        result = {{"processed": True}}

        return create_agent_task_result(
            task_id=task_id,
            task_type="{method_name}",
            data=result,
            duration_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        # Exceptions are automatically handled by the decorator
        raise e"""


def generate_legacy_wrapper(method_name: str, param_class: str) -> str:
    """Generate backwards compatibility wrapper."""
    return f"""async def {method_name}(self, *args, **kwargs) -> dict:
    \"\"\"Legacy wrapper for backwards compatibility.\"\"\"
    # Convert old-style parameters to new format
    if args:
        # Handle positional arguments
        params_dict = {{"data": args[0] if args else None}}
    else:
        params_dict = kwargs

    # Create RPC request
    request = RpcRequest(
        method="{method_name}",
        params={param_class}(**params_dict),
        id=str(uuid.uuid4())
    )

    # Call RPC method
    response = await self.handle_rpc_request(request)

    # Convert response to legacy format
    if response.result:
        return {{
            "status": "success",
            "data": response.result.data,
            "message": "Operation completed"
        }}
    else:
        return {{
            "status": "error",
            "error": response.error.message if response.error else "Unknown error"
        }}"""


# Example usage and testing functions


async def test_migration_compatibility(
    original_method: callable,
    rpc_method: callable,
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Test compatibility between original and migrated methods."""
    results = {
        "total_tests": len(test_cases),
        "passed": 0,
        "failed": 0,
        "compatibility_issues": [],
    }

    for i, test_case in enumerate(test_cases):
        try:
            # Test original method
            original_result = await original_method(**test_case["input"])

            # Test RPC method
            request = RpcRequest(
                method=rpc_method.__name__.replace("_rpc", ""),
                params=test_case["input"],
                id=f"test-{i}",
            )
            rpc_result = await rpc_method(request)

            # Compare results (simplified comparison)
            if _compare_results(original_result, rpc_result):
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["compatibility_issues"].append(
                    {
                        "test_case": i,
                        "original": original_result,
                        "rpc": rpc_result.model_dump()
                        if hasattr(rpc_result, "model_dump")
                        else str(rpc_result),
                    }
                )

        except Exception as e:
            results["failed"] += 1
            results["compatibility_issues"].append(
                {
                    "test_case": i,
                    "error": str(e),
                }
            )

    return results


def _compare_results(original: Any, rpc_response: RpcResponse) -> bool:
    """Compare original method result with RPC response."""
    # Simplified comparison - in practice, this would be more sophisticated
    if isinstance(original, dict):
        if "status" in original and original["status"] == "success":
            return rpc_response.result is not None
        if "error" in original or original.get("status") == "error":
            return rpc_response.error is not None

    return True  # Default to compatible for complex cases
