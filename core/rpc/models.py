"""JSON RPC 2.0 Compliance Models for My-Fullstack-Agent.

This module provides comprehensive JSON RPC 2.0 standard implementation
with Pydantic v2 validation, type safety, and full compatibility with
MCP and Claude protocol standards.

Features:
- Generic type support for flexible response data
- Error classification system with standard codes
- Batch request handling
- Request/response correlation tracking
- Session context management
- Comprehensive error handling patterns
- Type-safe response models for agent operations
"""

from datetime import UTC
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar
import uuid

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

# Type variables for generic models
T = TypeVar("T")
P = TypeVar("P")  # Parameters type
R = TypeVar("R")  # Result type
E = TypeVar("E")  # Error data type


class RpcErrorCode(Enum):
    """Standard JSON RPC 2.0 error codes with extensions for agent operations."""

    # Standard JSON RPC 2.0 errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Server error range (-32099 to -32000)
    SERVER_ERROR = -32000

    # Agent-specific error codes (-31999 to -31000)
    AGENT_NOT_FOUND = -31999
    AGENT_UNAVAILABLE = -31998
    AGENT_TIMEOUT = -31997
    AGENT_OVERLOADED = -31996
    AGENT_CONFIG_ERROR = -31995

    # Authentication and authorization errors (-30999 to -30900)
    UNAUTHORIZED = -30999
    FORBIDDEN = -30998
    TOKEN_EXPIRED = -30997
    INVALID_CREDENTIALS = -30996

    # Validation errors (-30899 to -30800)
    VALIDATION_ERROR = -30899
    SCHEMA_ERROR = -30898
    TYPE_ERROR = -30897
    CONSTRAINT_ERROR = -30896

    # Resource errors (-30799 to -30700)
    RESOURCE_NOT_FOUND = -30799
    RESOURCE_LOCKED = -30798
    RESOURCE_EXHAUSTED = -30797
    QUOTA_EXCEEDED = -30796

    # External service errors (-30699 to -30600)
    EXTERNAL_SERVICE_ERROR = -30699
    GEMINI_API_ERROR = -30698
    DATABASE_ERROR = -30697
    CACHE_ERROR = -30696

    # File system and I/O errors (-30599 to -30500)
    FILE_NOT_FOUND = -30599
    FILE_ACCESS_ERROR = -30598
    DIRECTORY_ERROR = -30597
    PERMISSION_DENIED = -30596


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RpcErrorData[E](BaseModel):
    """Extended error data for detailed error reporting."""

    model_config = ConfigDict(extra="allow")

    code: int = Field(description="Specific error code for programmatic handling")
    category: str = Field(description="Error category for classification")
    severity: ErrorSeverity = Field(
        default=ErrorSeverity.MEDIUM, description="Error severity level"
    )
    details: E | None = Field(default=None, description="Detailed error information")
    context: dict[str, Any] = Field(default_factory=dict, description="Error context information")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str | None = Field(default=None, description="Request correlation ID")
    retry_after: int | None = Field(default=None, description="Retry delay in seconds")
    suggestions: list[str] = Field(default_factory=list, description="Suggested error resolutions")


class RpcError(BaseModel):
    """Standard JSON RPC 2.0 error object with enhanced data."""

    code: int = Field(description="JSON RPC 2.0 error code")
    message: str = Field(description="Human-readable error message")
    data: RpcErrorData | None = Field(default=None, description="Additional error information")

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        code: RpcErrorCode = RpcErrorCode.INTERNAL_ERROR,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "RpcError":
        """Create RpcError from Python exception."""
        error_data = RpcErrorData(
            code=code.value,
            category=type(exception).__name__,
            severity=ErrorSeverity.HIGH,
            details=str(exception),
            context=context or {},
            correlation_id=correlation_id,
            suggestions=[
                "Check agent configuration",
                "Verify input parameters",
                "Review system resources",
            ],
        )

        return cls(
            code=code.value,
            message=str(exception) or f"{type(exception).__name__} occurred",
            data=error_data,
        )

    @classmethod
    def validation_error(
        cls,
        message: str,
        field: str | None = None,
        value: Any = None,
        correlation_id: str | None = None,
    ) -> "RpcError":
        """Create validation error."""
        context = {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)

        error_data = RpcErrorData(
            code=RpcErrorCode.VALIDATION_ERROR.value,
            category="ValidationError",
            severity=ErrorSeverity.MEDIUM,
            context=context,
            correlation_id=correlation_id,
            suggestions=[
                "Check parameter format and types",
                "Verify required fields are provided",
                "Review API documentation for valid values",
            ],
        )

        return cls(
            code=RpcErrorCode.VALIDATION_ERROR.value,
            message=message,
            data=error_data,
        )


class RpcRequest[P](BaseModel):
    """JSON RPC 2.0 request with enhanced metadata."""

    jsonrpc: str = Field(default="2.0", description="JSON RPC version")
    method: str = Field(description="Method name to call")
    params: P | None = Field(default=None, description="Method parameters")
    id: str | int | None = Field(default=None, description="Request identifier")

    # Enhanced metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = Field(default=None, description="Session identifier")
    agent_context: dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific context"
    )
    priority: int = Field(
        default=5, ge=1, le=10, description="Request priority (1=lowest, 10=highest)"
    )
    timeout: int | None = Field(default=None, description="Request timeout in seconds")

    @field_validator("jsonrpc")
    @classmethod
    def validate_jsonrpc(cls, v: str) -> str:
        if v != "2.0":
            msg = "JSON RPC version must be '2.0'"
            raise ValueError(msg)
        return v


class RpcResponse[R](BaseModel):
    """JSON RPC 2.0 response with enhanced metadata and type safety."""

    jsonrpc: str = Field(default="2.0", description="JSON RPC version")
    result: R | None = Field(default=None, description="Operation result")
    error: RpcError | None = Field(default=None, description="Error information")
    id: str | int | None = Field(default=None, description="Request identifier")

    # Enhanced metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str | None = Field(default=None, description="Request correlation ID")
    execution_time_ms: float | None = Field(
        default=None, description="Execution time in milliseconds"
    )
    agent_name: str | None = Field(default=None, description="Responding agent name")
    session_id: str | None = Field(default=None, description="Session identifier")

    # Performance and debugging information
    performance_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Performance and resource usage metrics",
    )
    debug_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Debug information (only included in debug mode)",
    )

    @field_validator("jsonrpc")
    @classmethod
    def validate_jsonrpc(cls, v: str) -> str:
        if v != "2.0":
            msg = "JSON RPC version must be '2.0'"
            raise ValueError(msg)
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that either result or error is present."""
        if self.result is None and self.error is None:
            msg = "Either result or error must be present"
            raise ValueError(msg)
        if self.result is not None and self.error is not None:
            msg = "Result and error cannot both be present"
            raise ValueError(msg)

    @classmethod
    def success(
        cls,
        result: R,
        request_id: str | int | None = None,
        correlation_id: str | None = None,
        agent_name: str | None = None,
        execution_time_ms: float | None = None,
        **kwargs,
    ) -> "RpcResponse[R]":
        """Create successful response."""
        return cls(
            result=result,
            id=request_id,
            correlation_id=correlation_id,
            agent_name=agent_name,
            execution_time_ms=execution_time_ms,
            **kwargs,
        )

    @classmethod
    def error(
        cls,
        error: RpcError,
        request_id: str | int | None = None,
        correlation_id: str | None = None,
        agent_name: str | None = None,
        **kwargs,
    ) -> "RpcResponse[Any]":
        """Create error response."""
        return cls(
            error=error,
            id=request_id,
            correlation_id=correlation_id,
            agent_name=agent_name,
            **kwargs,
        )


class BatchRpcRequest(BaseModel):
    """Batch JSON RPC 2.0 request for multiple operations."""

    requests: list[RpcRequest] = Field(description="List of RPC requests")
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str | None = Field(default=None, description="Session identifier")
    max_parallel: int = Field(default=5, ge=1, le=20, description="Maximum parallel execution")
    fail_fast: bool = Field(default=False, description="Stop on first error")

    @field_validator("requests")
    @classmethod
    def validate_requests(cls, v: list[RpcRequest]) -> list[RpcRequest]:
        if not v:
            msg = "Batch request must contain at least one request"
            raise ValueError(msg)
        if len(v) > 100:
            msg = "Batch request cannot contain more than 100 requests"
            raise ValueError(msg)
        return v


class BatchRpcResponse(BaseModel):
    """Batch JSON RPC 2.0 response for multiple operations."""

    responses: list[RpcResponse] = Field(description="List of RPC responses")
    batch_id: str = Field(description="Batch identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_execution_time_ms: float = Field(description="Total batch execution time")

    # Batch statistics
    total_requests: int = Field(description="Total number of requests")
    successful_requests: int = Field(description="Number of successful requests")
    failed_requests: int = Field(description="Number of failed requests")

    performance_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate performance metrics",
    )


# Agent-specific response models
class AgentTaskResult(BaseModel):
    """Standard result format for agent task execution."""

    task_id: str = Field(description="Unique task identifier")
    task_type: str = Field(description="Type of task executed")
    status: str = Field(description="Task execution status")

    # Result data
    data: dict[str, Any] = Field(default_factory=dict, description="Task result data")
    files_modified: list[str] = Field(
        default_factory=list, description="Files modified during task"
    )
    files_created: list[str] = Field(default_factory=list, description="Files created during task")

    # Execution metadata
    started_at: datetime = Field(description="Task start time")
    completed_at: datetime = Field(description="Task completion time")
    duration_ms: float = Field(description="Task duration in milliseconds")

    # Resource usage
    memory_used_mb: float | None = Field(default=None, description="Peak memory usage")
    cpu_time_ms: float | None = Field(default=None, description="CPU time used")

    # Warnings and suggestions
    warnings: list[str] = Field(default_factory=list, description="Task warnings")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")


class SessionContext(BaseModel):
    """Session context for maintaining state across requests."""

    session_id: str = Field(description="Unique session identifier")
    agent_name: str = Field(description="Associated agent name")
    created_at: datetime = Field(description="Session creation time")
    last_activity: datetime = Field(description="Last activity timestamp")

    # Session state
    context_data: dict[str, Any] = Field(default_factory=dict, description="Session context data")
    active_tasks: list[str] = Field(default_factory=list, description="Currently active task IDs")
    completed_tasks: list[str] = Field(default_factory=list, description="Completed task IDs")

    # Session configuration
    timeout_minutes: int = Field(default=30, description="Session timeout in minutes")
    max_concurrent_tasks: int = Field(default=5, description="Maximum concurrent tasks")

    def is_expired(self) -> bool:
        """Check if session has expired."""
        from datetime import timedelta

        expiry_time = self.last_activity + timedelta(minutes=self.timeout_minutes)
        return datetime.now(UTC) > expiry_time

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(UTC)


class HealthCheckResult(BaseModel):
    """Health check result for agent status monitoring."""

    agent_name: str = Field(description="Agent name")
    status: str = Field(description="Health status (healthy, degraded, unhealthy)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Component health
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Individual component health status",
    )

    # Performance metrics
    response_time_ms: float = Field(description="Health check response time")
    memory_usage_mb: float = Field(description="Current memory usage")
    cpu_usage_percent: float = Field(description="Current CPU usage")

    # Resource availability
    available_capacity: float = Field(description="Available processing capacity (0-1)")
    active_connections: int = Field(description="Number of active connections")
    queued_requests: int = Field(description="Number of queued requests")

    # Error information
    errors: list[str] = Field(default_factory=list, description="Recent errors")
    warnings: list[str] = Field(default_factory=list, description="Current warnings")


# Type aliases for common response patterns
AgentResponse = RpcResponse[AgentTaskResult]
HealthResponse = RpcResponse[HealthCheckResult]
BatchAgentResponse = BatchRpcResponse


# Utility functions for response creation
def create_success_response[T](
    result: T,
    request_id: str | int | None = None,
    correlation_id: str | None = None,
    agent_name: str | None = None,
    execution_time_ms: float | None = None,
) -> RpcResponse[T]:
    """Utility function to create successful RPC response."""
    return RpcResponse.success(
        result=result,
        request_id=request_id,
        correlation_id=correlation_id,
        agent_name=agent_name,
        execution_time_ms=execution_time_ms,
    )


def create_error_response(
    code: RpcErrorCode,
    message: str,
    request_id: str | int | None = None,
    correlation_id: str | None = None,
    agent_name: str | None = None,
    context: dict[str, Any] | None = None,
    suggestions: list[str] | None = None,
) -> RpcResponse[Any]:
    """Utility function to create error RPC response."""
    error_data = RpcErrorData(
        code=code.value,
        category=code.name,
        severity=ErrorSeverity.HIGH,
        context=context or {},
        correlation_id=correlation_id,
        suggestions=suggestions or [],
    )

    error = RpcError(
        code=code.value,
        message=message,
        data=error_data,
    )

    return RpcResponse.error(
        error=error,
        request_id=request_id,
        correlation_id=correlation_id,
        agent_name=agent_name,
    )
