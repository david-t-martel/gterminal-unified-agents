"""JSON RPC 2.0 Transformation Examples.

This module shows practical examples of how current agent methods
are transformed to be RPC-compliant, solving typing issues and
providing consistent error handling.

Examples include:
- Before/After method implementations
- Error handling transformations
- Type safety improvements
- Backwards compatibility patterns
"""

from datetime import UTC
from datetime import datetime
import time
from typing import Any
import uuid

from pydantic import BaseModel
from pydantic import Field

from gterminal.agents.base_agent_service import BaseAgentService

from .models import AgentTaskResult
from .models import RpcRequest
from .patterns import RpcAgentMixin
from .patterns import create_agent_task_result
from .patterns import rpc_method

# =================== EXAMPLE 1: CODE GENERATION AGENT ===================


class CodeGenerationParams(BaseModel):
    """Parameters for code generation requests."""

    specification: dict[str, Any] = Field(description="Code generation specification")
    output_path: str | None = Field(default=None, description="Output file path")
    template_name: str | None = Field(default="default", description="Template to use")
    language: str | None = Field(default="python", description="Target language")


class CodeGenerationResult(BaseModel):
    """Result of code generation operation."""

    generated_files: list[str] = Field(description="List of generated files")
    total_lines: int = Field(description="Total lines of code generated")
    warnings: list[str] = Field(default_factory=list, description="Generation warnings")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")


# BEFORE: Original implementation with typing issues
class OriginalCodeGenerationService(BaseAgentService):
    """Original implementation with typing and error handling issues."""

    async def generate_code(self, specification: dict) -> dict:
        """Original method with unclear return type and error handling."""
        try:
            # Simulation of code generation
            if not specification:
                return {
                    "status": "error",
                    "error": "Specification required",
                }

            return {
                "status": "success",
                "data": {
                    "generated_files": ["example.py", "test_example.py"],
                    "total_lines": 150,
                    "warnings": [],
                },
            }

        except Exception as e:
            # Inconsistent error format
            return {"error": str(e), "status": "failed"}


# AFTER: RPC-compliant implementation
class RpcCodeGenerationService(BaseAgentService, RpcAgentMixin):
    """RPC-compliant code generation service with type safety."""

    def __init__(self) -> None:
        super().__init__("code_generator", "RPC-compliant code generation service")

    @rpc_method(
        method_name="generate_code",
        timeout_seconds=300,
        validate_params=True,
        log_performance=True,
    )
    async def generate_code_rpc(
        self,
        params: CodeGenerationParams,
        session: Any | None = None,
    ) -> AgentTaskResult:
        """Generate code from specification with full RPC compliance.

        Args:
            params: Validated code generation parameters
            session: Optional session context

        Returns:
            AgentTaskResult with generation details

        """
        task_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Validate specification
            if not params.specification:
                msg = "Specification cannot be empty"
                raise ValueError(msg)

            # Simulate code generation
            generated_files = []
            total_lines = 0
            warnings = []
            suggestions = []

            # Process specification
            spec = params.specification
            if "endpoints" in spec:
                for endpoint in spec["endpoints"]:
                    filename = f"{endpoint['name']}.py"
                    generated_files.append(filename)
                    total_lines += 50  # Estimated lines per endpoint

            if "models" in spec:
                for model in spec["models"]:
                    filename = f"models/{model['name']}.py"
                    generated_files.append(filename)
                    total_lines += 30  # Estimated lines per model

            # Add test files
            test_files = [f"tests/test_{f}" for f in generated_files if f.endswith(".py")]
            generated_files.extend(test_files)
            total_lines += len(test_files) * 40

            # Generate warnings and suggestions
            if total_lines > 500:
                warnings.append("Large codebase generated - consider splitting into modules")

            if params.language != "python":
                suggestions.append("Consider using Python for better framework integration")

            # Calculate execution time
            duration_ms = (time.time() - start_time) * 1000

            # Create result
            result_data = CodeGenerationResult(
                generated_files=generated_files,
                total_lines=total_lines,
                warnings=warnings,
                suggestions=suggestions,
            )

            return create_agent_task_result(
                task_id=task_id,
                task_type="code_generation",
                data=result_data.model_dump(),
                files_created=generated_files,
                warnings=warnings,
                suggestions=suggestions,
                duration_ms=duration_ms,
            )

        except ValueError:
            # Validation errors are automatically handled by decorator
            raise
        except Exception as e:
            # Other errors are automatically wrapped by decorator
            msg = f"Code generation failed: {e!s}"
            raise RuntimeError(msg)


# =================== EXAMPLE 2: WORKSPACE ANALYZER AGENT ===================


class WorkspaceAnalysisParams(BaseModel):
    """Parameters for workspace analysis."""

    project_path: str = Field(description="Path to project directory")
    depth: int = Field(default=3, ge=1, le=10, description="Analysis depth")
    include_patterns: list[str] = Field(
        default_factory=lambda: ["*.py", "*.js"], description="File patterns to include"
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["node_modules/*", "*.pyc"], description="File patterns to exclude"
    )


class WorkspaceAnalysisResult(BaseModel):
    """Result of workspace analysis."""

    total_files: int = Field(description="Total files analyzed")
    lines_of_code: int = Field(description="Total lines of code")
    languages: dict[str, int] = Field(description="Language breakdown")
    complexity_score: float = Field(description="Overall complexity score")
    suggestions: list[str] = Field(description="Optimization suggestions")
    dependencies: dict[str, list[str]] = Field(description="Project dependencies")


# BEFORE: Original implementation
class OriginalWorkspaceAnalyzer(BaseAgentService):
    """Original analyzer with inconsistent returns."""

    async def analyze_workspace(self, project_path: str, options: dict | None = None) -> dict:
        """Original method with unclear parameters and return types."""
        try:
            if not project_path:
                return {"error": "Project path required"}

            # Simulate analysis
            return {
                "success": True,
                "result": {
                    "files": 150,
                    "loc": 12000,
                    "languages": {"python": 8000, "javascript": 4000},
                },
            }
        except Exception:
            return {"success": False, "error": "Analysis failed"}


# AFTER: RPC-compliant implementation
class RpcWorkspaceAnalyzer(BaseAgentService, RpcAgentMixin):
    """RPC-compliant workspace analyzer."""

    def __init__(self) -> None:
        super().__init__("workspace_analyzer", "RPC-compliant workspace analysis")

    @rpc_method(
        method_name="analyze_workspace",
        timeout_seconds=600,  # Longer timeout for large projects
        validate_params=True,
        enable_caching=True,
        cache_ttl_seconds=1800,  # 30 minutes
    )
    async def analyze_workspace_rpc(
        self,
        params: WorkspaceAnalysisParams,
        session: Any | None = None,
    ) -> AgentTaskResult:
        """Analyze workspace structure and provide insights.

        Args:
            params: Validated analysis parameters
            session: Optional session context

        Returns:
            AgentTaskResult with analysis details

        """
        task_id = str(uuid.uuid4())
        start_time = time.time()

        # Validate project path exists
        from pathlib import Path

        project_path = Path(params.project_path)
        if not project_path.exists():
            msg = f"Project path does not exist: {params.project_path}"
            raise FileNotFoundError(msg)

        if not project_path.is_dir():
            msg = f"Project path is not a directory: {params.project_path}"
            raise ValueError(msg)

        # Perform analysis
        analysis_result = await self._analyze_directory(project_path, params)

        duration_ms = (time.time() - start_time) * 1000

        return create_agent_task_result(
            task_id=task_id,
            task_type="workspace_analysis",
            data=analysis_result.model_dump(),
            files_modified=[],  # Analysis doesn't modify files
            files_created=[],  # No files created during analysis
            warnings=analysis_result.suggestions[:3],  # First 3 as warnings
            suggestions=analysis_result.suggestions,
            duration_ms=duration_ms,
        )

    async def _analyze_directory(
        self,
        project_path: Path,
        params: WorkspaceAnalysisParams,
    ) -> WorkspaceAnalysisResult:
        """Internal method to perform directory analysis."""
        # Simulate file analysis
        total_files = 0
        lines_of_code = 0
        languages = {}
        dependencies = {"python": [], "javascript": []}

        # Mock analysis results
        for pattern in params.include_patterns:
            if "*.py" in pattern:
                py_files = 50
                py_lines = 8000
                total_files += py_files
                lines_of_code += py_lines
                languages["python"] = py_lines
                dependencies["python"] = ["fastapi", "pydantic", "uvicorn"]

            if "*.js" in pattern:
                js_files = 30
                js_lines = 4000
                total_files += js_files
                lines_of_code += js_lines
                languages["javascript"] = js_lines
                dependencies["javascript"] = ["react", "axios", "lodash"]

        # Calculate complexity score (0-10)
        complexity_score = min(10.0, (lines_of_code / 1000) * 0.5 + (total_files / 10) * 0.3)

        # Generate suggestions
        suggestions = []
        if lines_of_code > 10000:
            suggestions.append("Consider breaking down large modules into smaller components")
        if complexity_score > 7.0:
            suggestions.append("High complexity detected - review code structure")
        if len(languages) > 3:
            suggestions.append("Multiple languages detected - ensure consistent tooling")

        return WorkspaceAnalysisResult(
            total_files=total_files,
            lines_of_code=lines_of_code,
            languages=languages,
            complexity_score=complexity_score,
            suggestions=suggestions,
            dependencies=dependencies,
        )


# =================== EXAMPLE 3: ERROR HANDLING TRANSFORMATION ===================


class FileOperationParams(BaseModel):
    """Parameters for file operations."""

    file_path: str = Field(description="Target file path")
    operation: str = Field(description="Operation type (read, write, delete)")
    content: str | None = Field(default=None, description="Content for write operations")


# BEFORE: Inconsistent error handling
class OriginalFileService(BaseAgentService):
    """Original file service with inconsistent error patterns."""

    async def handle_file(self, path: str, operation: str, content: str | None = None):
        """Original method with various error return patterns."""
        try:
            if operation == "read":
                try:
                    with open(path) as f:
                        return {"data": f.read(), "success": True}
                except FileNotFoundError:
                    return {"error": "File not found", "code": 404}
                except PermissionError:
                    return {"error": "Permission denied", "success": False}

            elif operation == "write":
                if not content:
                    return {"error": "Content required for write"}
                with open(path, "w") as f:
                    f.write(content)
                return {"success": True, "message": "File written"}

            else:
                return {"error": f"Unknown operation: {operation}"}

        except Exception as e:
            return {"failed": True, "message": str(e)}


# AFTER: Standardized RPC error handling
class RpcFileService(BaseAgentService, RpcAgentMixin):
    """RPC-compliant file service with standardized error handling."""

    def __init__(self) -> None:
        super().__init__("file_service", "RPC-compliant file operations")

    @rpc_method(
        method_name="handle_file",
        timeout_seconds=30,
        validate_params=True,
    )
    async def handle_file_rpc(
        self,
        params: FileOperationParams,
        session: Any | None = None,
    ) -> AgentTaskResult:
        """Handle file operations with standardized error handling.

        All errors are automatically converted to proper RPC error responses
        by the decorator, with appropriate error codes and context.
        """
        task_id = str(uuid.uuid4())
        start_time = time.time()

        from pathlib import Path

        file_path = Path(params.file_path)
        result_data = {"operation": params.operation, "file_path": params.file_path}
        files_modified = []
        files_created = []

        if params.operation == "read":
            if not file_path.exists():
                # This will be automatically converted to RpcErrorCode.FILE_NOT_FOUND
                msg = f"File not found: {params.file_path}"
                raise FileNotFoundError(msg)

            if not file_path.is_file():
                msg = f"Path is not a file: {params.file_path}"
                raise ValueError(msg)

            try:
                content = file_path.read_text()
                result_data["content"] = content
                result_data["size_bytes"] = len(content.encode("utf-8"))
            except PermissionError:
                # This will be automatically converted to RpcErrorCode.PERMISSION_DENIED
                msg = f"Permission denied reading file: {params.file_path}"
                raise PermissionError(msg)

        elif params.operation == "write":
            if not params.content:
                msg = "Content is required for write operations"
                raise ValueError(msg)

            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(params.content)

                result_data["bytes_written"] = len(params.content.encode("utf-8"))
                files_modified.append(
                    str(file_path)
                ) if file_path.exists() else files_created.append(str(file_path))

            except PermissionError:
                msg = f"Permission denied writing file: {params.file_path}"
                raise PermissionError(msg)

        elif params.operation == "delete":
            if not file_path.exists():
                msg = f"File not found: {params.file_path}"
                raise FileNotFoundError(msg)

            try:
                file_path.unlink()
                result_data["deleted"] = True
                files_modified.append(str(file_path))
            except PermissionError:
                msg = f"Permission denied deleting file: {params.file_path}"
                raise PermissionError(msg)

        else:
            msg = f"Unknown operation: {params.operation}"
            raise ValueError(msg)

        duration_ms = (time.time() - start_time) * 1000

        return create_agent_task_result(
            task_id=task_id,
            task_type="file_operation",
            data=result_data,
            files_modified=files_modified,
            files_created=files_created,
            duration_ms=duration_ms,
        )


# =================== EXAMPLE 4: BACKWARDS COMPATIBILITY ===================


class BackwardsCompatibleAgent(BaseAgentService, RpcAgentMixin):
    """Example showing how to maintain backwards compatibility
    while adding RPC compliance.
    """

    def __init__(self) -> None:
        super().__init__("compatible_agent", "Backwards compatible RPC agent")

    # Legacy method - maintains old interface
    async def legacy_method(self, data: dict) -> dict:
        """Legacy method that returns old format."""
        # Convert to RPC request internally
        request = RpcRequest(
            method="process_data",
            params=data,
            id=str(uuid.uuid4()),
        )

        # Use RPC method internally
        rpc_response = await self.process_data_rpc(request)

        # Convert back to legacy format
        if rpc_response.result:
            return {
                "status": "success",
                "data": rpc_response.result.data,
                "message": "Operation completed",
            }
        return {
            "status": "error",
            "error": rpc_response.error.message if rpc_response.error else "Unknown error",
        }

    # New RPC-compliant method
    @rpc_method(method_name="process_data")
    async def process_data_rpc(
        self,
        params: dict[str, Any],
        session: Any | None = None,
    ) -> AgentTaskResult:
        """RPC-compliant method with full type safety."""
        task_id = str(uuid.uuid4())
        start_time = time.time()

        # Process data
        processed_data = {
            "input_keys": list(params.keys()),
            "processed_at": datetime.now(UTC).isoformat(),
            "result": "Data processed successfully",
        }

        duration_ms = (time.time() - start_time) * 1000

        return create_agent_task_result(
            task_id=task_id,
            task_type="data_processing",
            data=processed_data,
            duration_ms=duration_ms,
        )


# =================== SUMMARY OF TRANSFORMATIONS ===================

"""
Key Transformations Applied:

1. TYPE SAFETY:
   - Before: def method(self, data: dict) -> dict
   - After: async def method_rpc(self, params: ValidationModel, session) -> AgentTaskResult

2. ERROR HANDLING:
   - Before: Multiple return formats for errors {"error": ..., "success": False}
   - After: Exceptions automatically converted to standardized RpcError objects

3. PARAMETER VALIDATION:
   - Before: Manual validation with inconsistent error formats
   - After: Pydantic models with automatic validation and standard error responses

4. RESPONSE STANDARDIZATION:
   - Before: Various response formats (success/error, data/result, etc.)
   - After: Single RpcResponse[T] format with result or error fields

5. METADATA ENHANCEMENT:
   - Before: No timing, correlation IDs, or performance metrics
   - After: Automatic correlation IDs, execution timing, performance metrics

6. SESSION MANAGEMENT:
   - Before: No session support
   - After: Optional session context with automatic lifecycle management

7. BACKWARDS COMPATIBILITY:
   - Wrapper methods that convert between old and new formats
   - Gradual migration path for existing clients
   - Internal use of RPC methods with format conversion
"""
