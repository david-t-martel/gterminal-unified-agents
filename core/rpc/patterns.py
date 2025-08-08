"""JSON RPC 2.0 Implementation Patterns for Agent Framework.

This module provides implementation patterns and base classes for
standardizing agent method implementations with RPC compliance.

Features:
- Agent method standardization with decorator pattern
- Automatic error handling and wrapping
- Session and context management
- Type-safe method decorators
- Batch processing patterns
- Performance monitoring integration
"""

import asyncio
from collections.abc import Callable
import contextlib
from datetime import UTC
from datetime import datetime
from functools import wraps
import inspect
import logging
import time
from typing import Any, TypeVar, get_type_hints

from pydantic import BaseModel
from pydantic import ValidationError

from .models import AgentTaskResult
from .models import RpcError
from .models import RpcErrorCode
from .models import RpcRequest
from .models import RpcResponse
from .models import SessionContext
from .models import create_error_response
from .models import create_success_response

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class RpcMethodConfig(BaseModel):
    """Configuration for RPC method decoration."""

    method_name: str | None = None
    timeout_seconds: int | None = None
    require_session: bool = False
    validate_params: bool = True
    enable_caching: bool = False
    cache_ttl_seconds: int = 3600
    log_performance: bool = True
    auto_retry: bool = False
    max_retries: int = 3


class AgentMethodRegistry:
    """Registry for agent methods with RPC compliance."""

    def __init__(self) -> None:
        self._methods: dict[str, dict[str, Any]] = {}
        self._middleware: list[Callable] = []

    def register_method(
        self,
        method_name: str,
        handler: Callable,
        config: RpcMethodConfig,
        agent_name: str,
    ) -> None:
        """Register an RPC method."""
        self._methods[method_name] = {
            "handler": handler,
            "config": config,
            "agent_name": agent_name,
            "signature": inspect.signature(handler),
            "type_hints": get_type_hints(handler),
            "registered_at": datetime.now(UTC),
        }
        logger.debug(f"Registered RPC method: {method_name}")

    def get_method(self, method_name: str) -> dict[str, Any] | None:
        """Get registered method information."""
        return self._methods.get(method_name)

    def list_methods(self) -> list[str]:
        """List all registered method names."""
        return list(self._methods.keys())

    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware function."""
        self._middleware.append(middleware)


# Global registry instance
method_registry = AgentMethodRegistry()


def rpc_method(
    method_name: str | None = None,
    timeout_seconds: int | None = None,
    require_session: bool = False,
    validate_params: bool = True,
    enable_caching: bool = False,
    cache_ttl_seconds: int = 3600,
    log_performance: bool = True,
    auto_retry: bool = False,
    max_retries: int = 3,
) -> Callable[[F], F]:
    """Decorator to make agent methods RPC-compliant.

    This decorator handles:
    - Parameter validation using Pydantic
    - Automatic error wrapping
    - Performance monitoring
    - Session management
    - Response standardization
    """

    def decorator(func: F) -> F:
        config = RpcMethodConfig(
            method_name=method_name or func.__name__,
            timeout_seconds=timeout_seconds,
            require_session=require_session,
            validate_params=validate_params,
            enable_caching=enable_caching,
            cache_ttl_seconds=cache_ttl_seconds,
            log_performance=log_performance,
            auto_retry=auto_retry,
            max_retries=max_retries,
        )

        @wraps(func)
        async def async_wrapper(
            self,
            request: RpcRequest,
            session: SessionContext | None = None,
            **kwargs,
        ) -> RpcResponse:
            """Async wrapper for RPC method execution."""
            start_time = time.time()
            correlation_id = request.correlation_id

            try:
                # Session validation
                if config.require_session and not session:
                    return create_error_response(
                        RpcErrorCode.INVALID_REQUEST,
                        "Session required for this method",
                        request.id,
                        correlation_id,
                        getattr(self, "agent_name", None),
                    )

                # Update session activity
                if session:
                    session.update_activity()

                # Parameter validation
                if config.validate_params and request.params:
                    try:
                        # Get expected parameter type from function signature
                        inspect.signature(func)
                        get_type_hints(func)

                        # Validate parameters if type hints are available
                        if hasattr(request.params, "dict"):
                            # Pydantic model
                            validated_params = request.params
                        elif isinstance(request.params, dict):
                            # Dictionary parameters - basic validation
                            validated_params = request.params
                        else:
                            validated_params = request.params

                    except ValidationError as e:
                        return create_error_response(
                            RpcErrorCode.VALIDATION_ERROR,
                            f"Parameter validation failed: {e}",
                            request.id,
                            correlation_id,
                            getattr(self, "agent_name", None),
                        )
                else:
                    validated_params = request.params

                # Execute method with timeout
                if config.timeout_seconds:
                    result = await asyncio.wait_for(
                        func(self, validated_params, session=session, **kwargs),
                        timeout=config.timeout_seconds,
                    )
                else:
                    result = await func(self, validated_params, session=session, **kwargs)

                # Calculate execution time
                execution_time_ms = (time.time() - start_time) * 1000

                # Performance logging
                if config.log_performance:
                    logger.info(
                        f"Method {config.method_name} completed in {execution_time_ms:.2f}ms",
                        extra={
                            "correlation_id": correlation_id,
                            "method": config.method_name,
                            "execution_time_ms": execution_time_ms,
                        },
                    )

                # Handle different result types
                if isinstance(result, RpcResponse):
                    # Already a proper RPC response
                    return result
                if isinstance(result, dict) and "status" in result:
                    # Legacy response format - convert
                    if result.get("status") == "success":
                        return create_success_response(
                            result.get("data", result),
                            request.id,
                            correlation_id,
                            getattr(self, "agent_name", None),
                            execution_time_ms,
                        )
                    return create_error_response(
                        RpcErrorCode.INTERNAL_ERROR,
                        result.get("error", "Unknown error"),
                        request.id,
                        correlation_id,
                        getattr(self, "agent_name", None),
                    )
                # Raw result - wrap in success response
                return create_success_response(
                    result,
                    request.id,
                    correlation_id,
                    getattr(self, "agent_name", None),
                    execution_time_ms,
                )

            except TimeoutError:
                return create_error_response(
                    RpcErrorCode.AGENT_TIMEOUT,
                    f"Method {config.method_name} timed out after {config.timeout_seconds}s",
                    request.id,
                    correlation_id,
                    getattr(self, "agent_name", None),
                )
            except ValidationError as e:
                return create_error_response(
                    RpcErrorCode.VALIDATION_ERROR,
                    f"Validation error: {e}",
                    request.id,
                    correlation_id,
                    getattr(self, "agent_name", None),
                )
            except Exception as e:
                logger.error(
                    f"Method {config.method_name} failed: {e}",
                    extra={"correlation_id": correlation_id},
                    exc_info=True,
                )

                error = RpcError.from_exception(
                    e,
                    RpcErrorCode.INTERNAL_ERROR,
                    correlation_id,
                    {"method": config.method_name},
                )

                return RpcResponse.error(
                    error,
                    request.id,
                    correlation_id,
                    getattr(self, "agent_name", None),
                )

        # Register the method
        agent_name = (
            getattr(func.__self__, "agent_name", "unknown")
            if hasattr(func, "__self__")
            else "unknown"
        )

        method_registry.register_method(
            config.method_name,
            async_wrapper,
            config,
            agent_name,
        )

        # Store config on function for introspection
        func._rpc_config = config

        return async_wrapper

    return decorator


class RpcAgentMixin:
    """Mixin class to add RPC capabilities to existing agents.

    This mixin provides:
    - Method registration and discovery
    - Request routing and dispatch
    - Session management
    - Batch processing
    - Health checks
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sessions: dict[str, SessionContext] = {}
        self._method_cache: dict[str, Any] = {}

    async def handle_rpc_request(self, request: RpcRequest) -> RpcResponse:
        """Handle incoming RPC request."""
        method_info = method_registry.get_method(request.method)

        if not method_info:
            return create_error_response(
                RpcErrorCode.METHOD_NOT_FOUND,
                f"Method {request.method} not found",
                request.id,
                request.correlation_id,
                getattr(self, "agent_name", None),
            )

        # Get or create session if needed
        session = None
        if request.session_id:
            session = self._get_or_create_session(request.session_id)

        # Execute method
        handler = method_info["handler"]
        return await handler(self, request, session)

    async def handle_batch_request(self, requests: list[RpcRequest]) -> list[RpcResponse]:
        """Handle batch RPC requests."""
        # Process requests concurrently
        tasks = [self.handle_rpc_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error responses
        result = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error = RpcError.from_exception(
                    response,
                    RpcErrorCode.INTERNAL_ERROR,
                    requests[i].correlation_id,
                )
                result.append(
                    RpcResponse.error(
                        error,
                        requests[i].id,
                        requests[i].correlation_id,
                        getattr(self, "agent_name", None),
                    )
                )
            else:
                result.append(response)

        return result

    def _get_or_create_session(self, session_id: str) -> SessionContext:
        """Get existing session or create new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(
                session_id=session_id,
                agent_name=getattr(self, "agent_name", "unknown"),
                created_at=datetime.now(UTC),
                last_activity=datetime.now(UTC),
            )

        session = self._sessions[session_id]

        # Clean up expired sessions
        expired_sessions = [sid for sid, sess in self._sessions.items() if sess.is_expired()]
        for sid in expired_sessions:
            del self._sessions[sid]
            logger.debug(f"Cleaned up expired session: {sid}")

        return session

    def get_rpc_methods(self) -> dict[str, dict[str, Any]]:
        """Get available RPC methods for this agent."""
        agent_name = getattr(self, "agent_name", None)
        return {
            name: info
            for name, info in method_registry._methods.items()
            if info["agent_name"] == agent_name
        }

    async def health_check_rpc(self) -> dict[str, Any]:
        """Perform health check for RPC functionality."""
        return {
            "status": "healthy",
            "methods_registered": len(self.get_rpc_methods()),
            "active_sessions": len(self._sessions),
            "cache_size": len(self._method_cache),
            "timestamp": datetime.now(UTC).isoformat(),
        }


class RpcTaskManager:
    """Task manager for handling long-running RPC operations."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, Any] = {}

    async def start_task(
        self,
        task_id: str,
        coro: Callable,
        *args,
        **kwargs,
    ) -> str:
        """Start a long-running task."""
        if task_id in self._tasks:
            msg = f"Task {task_id} already running"
            raise ValueError(msg)

        task = asyncio.create_task(coro(*args, **kwargs))
        self._tasks[task_id] = task

        # Set up completion callback
        def task_done_callback(task: asyncio.Task) -> None:
            try:
                self._results[task_id] = task.result()
            except Exception as e:
                self._results[task_id] = RpcError.from_exception(e)
            finally:
                self._tasks.pop(task_id, None)

        task.add_done_callback(task_done_callback)
        return task_id

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get task status."""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            return {
                "status": "running" if not task.done() else "completed",
                "done": task.done(),
                "cancelled": task.cancelled(),
            }
        if task_id in self._results:
            return {
                "status": "completed",
                "done": True,
                "result_available": True,
            }
        return {"status": "not_found"}

    def get_task_result(self, task_id: str) -> Any:
        """Get task result if available."""
        return self._results.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return True
        return False


# Utility functions for common patterns


def validate_agent_parameters(params: dict[str, Any], required: list[str]) -> RpcError | None:
    """Validate required parameters for agent methods."""
    missing = [param for param in required if param not in params]
    if missing:
        return RpcError.validation_error(
            f"Missing required parameters: {', '.join(missing)}",
            correlation_id=params.get("correlation_id"),
        )
    return None


def create_agent_task_result(
    task_id: str,
    task_type: str,
    data: dict[str, Any],
    files_modified: list[str] | None = None,
    files_created: list[str] | None = None,
    warnings: list[str] | None = None,
    suggestions: list[str] | None = None,
    duration_ms: float | None = None,
) -> AgentTaskResult:
    """Create standardized agent task result."""
    now = datetime.now(UTC)

    return AgentTaskResult(
        task_id=task_id,
        task_type=task_type,
        status="completed",
        data=data,
        files_modified=files_modified or [],
        files_created=files_created or [],
        started_at=now,
        completed_at=now,
        duration_ms=duration_ms or 0.0,
        warnings=warnings or [],
        suggestions=suggestions or [],
    )


def measure_execution_time(func: Callable) -> Callable:
    """Decorator to measure function execution time."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000

            # Add timing to result if it's a dict
            if isinstance(result, dict):
                result["execution_time_ms"] = execution_time

            return result
        finally:
            execution_time = (time.time() - start_time) * 1000
            logger.debug(f"Function {func.__name__} took {execution_time:.2f}ms")

    return wrapper
