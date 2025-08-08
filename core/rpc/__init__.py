"""JSON RPC 2.0 Compliance Framework for My-Fullstack-Agent.

This package provides a comprehensive JSON RPC 2.0 implementation for
standardizing agent method interfaces, error handling, and response formats.

Key Components:
- models: Pydantic models for RPC requests, responses, and errors
- patterns: Implementation patterns and decorators for RPC compliance
- examples: Practical examples showing before/after transformations
- migration_guide: Tools and utilities for migrating existing code
- ast_grep_rules: Automated transformation rules

Features:
- Type-safe request/response handling
- Standardized error classification and reporting
- Batch request processing
- Session and context management
- Performance monitoring and metrics
- Backwards compatibility patterns
- Automated migration tools

Usage:
    from gterminal.core.rpc import rpc_method, RpcAgentMixin, RpcResponse
    from gterminal.core.rpc.models import AgentTaskResult, RpcErrorCode

    class MyAgent(BaseAgentService, RpcAgentMixin):
        @rpc_method(method_name="process_data", validate_params=True)
        async def process_data_rpc(self, params: MyParams, session=None) -> AgentTaskResult:
            # Implementation here
            pass
"""

from .models import AgentResponse  # Type aliases
from .models import AgentTaskResult  # Agent-specific models
from .models import BatchAgentResponse
from .models import BatchRpcRequest  # Batch processing
from .models import BatchRpcResponse
from .models import ErrorSeverity
from .models import HealthCheckResult
from .models import HealthResponse
from .models import RpcError
from .models import RpcErrorCode
from .models import RpcErrorData
from .models import RpcRequest  # Core RPC models
from .models import RpcResponse
from .models import SessionContext
from .models import create_error_response
from .models import create_success_response  # Utility functions
from .patterns import AgentMethodRegistry
from .patterns import RpcAgentMixin
from .patterns import RpcMethodConfig
from .patterns import RpcTaskManager
from .patterns import create_agent_task_result  # Utilities
from .patterns import measure_execution_time
from .patterns import method_registry  # Global registry
from .patterns import rpc_method  # Core patterns
from .patterns import validate_agent_parameters

# Version information
__version__ = "1.0.0"
__author__ = "My-Fullstack-Agent Team"
__description__ = "JSON RPC 2.0 Compliance Framework for AI Agents"

# Export commonly used items at package level
__all__ = [
    # Type aliases
    "AgentResponse",
    "AgentTaskResult",
    "HealthResponse",
    "RpcAgentMixin",
    "RpcError",
    "RpcErrorCode",
    # Core classes
    "RpcRequest",
    "RpcResponse",
    "SessionContext",
    # Constants
    "__version__",
    "create_agent_task_result",
    "create_error_response",
    # Utilities
    "create_success_response",
    # Decorators and mixins
    "rpc_method",
]

# Configuration defaults
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_CACHE_TTL_SECONDS = 3600
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 100

# Error code mappings for common exceptions
EXCEPTION_TO_ERROR_CODE = {
    ValueError: RpcErrorCode.VALIDATION_ERROR,
    TypeError: RpcErrorCode.TYPE_ERROR,
    FileNotFoundError: RpcErrorCode.FILE_NOT_FOUND,
    PermissionError: RpcErrorCode.PERMISSION_DENIED,
    TimeoutError: RpcErrorCode.AGENT_TIMEOUT,
    ConnectionError: RpcErrorCode.EXTERNAL_SERVICE_ERROR,
    KeyError: RpcErrorCode.RESOURCE_NOT_FOUND,
    NotImplementedError: RpcErrorCode.METHOD_NOT_FOUND,
}


def get_error_code_for_exception(exception: Exception) -> RpcErrorCode:
    """Get appropriate RPC error code for Python exception."""
    return EXCEPTION_TO_ERROR_CODE.get(type(exception), RpcErrorCode.INTERNAL_ERROR)


# Framework initialization and validation
def validate_framework_setup() -> dict[str, Any]:
    """Validate that the RPC framework is properly set up."""
    validation_results = {
        "status": "healthy",
        "components": {},
        "warnings": [],
        "errors": [],
    }

    try:
        # Check if required imports work
        validation_results["components"]["pydantic"] = True

        # Check method registry
        validation_results["components"]["method_registry"] = len(method_registry.list_methods())

        # Check if AST-grep is available for transformations
        try:
            import subprocess

            result = subprocess.run(
                ["ast-grep", "--version"], check=False, capture_output=True, timeout=5
            )
            validation_results["components"]["ast_grep"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            validation_results["components"]["ast_grep"] = False
            validation_results["warnings"].append(
                "AST-grep not available - automated transformations disabled"
            )

        # Validate example models can be instantiated
        try:
            RpcRequest(method="test", params={"test": True})
            RpcResponse.success({"result": "ok"})
            validation_results["components"]["model_validation"] = True
        except Exception as e:
            validation_results["components"]["model_validation"] = False
            validation_results["errors"].append(f"Model validation failed: {e}")

    except Exception as e:
        validation_results["status"] = "unhealthy"
        validation_results["errors"].append(f"Framework validation failed: {e}")

    return validation_results


# Quick start utilities
def create_basic_agent_template(agent_name: str, methods: list[str]) -> str:
    """Generate basic RPC-compliant agent template code."""
    template = f'''"""
{agent_name} - RPC-compliant agent implementation

Generated using My-Fullstack-Agent RPC Framework
"""

from typing import Any, Dict, Optional
from gterminal.agents.base_agent_service import BaseAgentService
from gterminal.core.rpc import rpc_method, RpcAgentMixin, AgentTaskResult, create_agent_task_result
from pydantic import BaseModel, Field
import uuid
import time


class {agent_name}(BaseAgentService, RpcAgentMixin):
    """RPC-compliant {agent_name.lower()} agent."""

    def __init__(self):
        super().__init__("{agent_name.lower()}", "RPC-compliant agent")
'''

    for method in methods:
        # Create parameter model
        param_class = f"{method.title().replace('_', '')}Params"
        template += f'''

class {param_class}(BaseModel):
    """Parameters for {method} method."""
    # Add your parameter definitions here
    data: Dict[str, Any] = Field(description="Method input data")
'''

        # Create RPC method
        template += f'''
    @rpc_method(method_name="{method}", validate_params=True)
    async def {method}_rpc(
        self,
        params: {param_class},
        session: Optional[Any] = None
    ) -> AgentTaskResult:
        """RPC-compliant {method} method."""
        task_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # TODO: Implement your method logic here
            result = {{"processed": True, "method": "{method}"}}

            return create_agent_task_result(
                task_id=task_id,
                task_type="{method}",
                data=result,
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            # Exceptions are automatically handled by the decorator
            raise e
'''

    return template


# Migration shortcuts
def quick_migrate_method(method_code: str, method_name: str) -> dict[str, str]:
    """Quickly migrate a single method to RPC compliance."""
    # This is a simplified version - full migration should use the migration_guide

    param_class = f"{method_name.title().replace('_', '')}Params"

    # Extract parameters (very basic parsing)
    import re

    param_match = re.search(r"async def \w+\(self,\s*([^)]+)\)", method_code)
    params = []
    if param_match:
        param_str = param_match.group(1)
        # Basic parameter extraction
        for param in param_str.split(","):
            param = param.strip()
            if ":" in param:
                param_name = param.split(":")[0].strip()
                if param_name not in ["self", "session"]:
                    params.append(param_name)

    # Generate parameter model
    param_model = f'''class {param_class}(BaseModel):
    """Parameters for {method_name} method."""'''

    for param in params:
        param_model += f"""
    {param}: Any = Field(description="Parameter {param}")"""

    # Generate RPC method
    rpc_method_code = f'''@rpc_method(method_name="{method_name}", validate_params=True)
async def {method_name}_rpc(
    self,
    params: {param_class},
    session: Optional[Any] = None
) -> AgentTaskResult:
    """RPC-compliant {method_name} method."""
    task_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Original logic (needs manual integration)
        result = {{"migrated": True}}

        return create_agent_task_result(
            task_id=task_id,
            task_type="{method_name}",
            data=result,
            duration_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        raise e'''

    return {
        "parameter_model": param_model,
        "rpc_method": rpc_method_code,
        "original_method": method_code,
    }
