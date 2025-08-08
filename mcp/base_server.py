"""Base MCP Server implementation for GTerminal.

Provides a robust foundation for MCP servers with authentication,
monitoring, error handling, and protocol compliance.
"""

from abc import ABC
from abc import abstractmethod
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
import hashlib
import json
import logging
import time
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel
from pydantic import Field

from ..auth import security_middleware

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for MCP servers."""

    name: str
    description: str
    version: str = "1.0.0"

    # Authentication settings
    require_auth: bool = True
    allowed_permissions: set[str] = field(default_factory=set)

    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int | None = None

    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300

    # Tool configuration
    tool_timeout_seconds: int = 30
    max_concurrent_tools: int = 10


class MCPRequest(BaseModel):
    """Base MCP request model."""

    request_id: str = Field(..., description="Unique request identifier")
    timestamp: float = Field(default_factory=time.time, description="Request timestamp")
    user_context: dict[str, Any] | None = Field(None, description="User context information")


class MCPResponse(BaseModel):
    """Base MCP response model."""

    request_id: str = Field(..., description="Matching request identifier")
    success: bool = Field(..., description="Whether the operation was successful")
    data: dict[str, Any] | None = Field(None, description="Response data")
    error: str | None = Field(None, description="Error message if unsuccessful")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    execution_time_ms: float | None = Field(None, description="Execution time in milliseconds")
    cached: bool = Field(False, description="Whether response was served from cache")


class BaseMCPServer(ABC):
    """Abstract base class for MCP servers with comprehensive features."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.mcp = FastMCP(config.name)
        self.start_time = time.time()

        # Performance tracking
        self.request_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0

        # Tool registry
        self.tools: dict[str, Callable] = {}
        self.tool_metrics: dict[str, dict[str, Any]] = {}

        # Simple in-memory cache (in production, use Redis)
        self.cache: dict[str, tuple[Any, float]] = {}

        # Semaphore for controlling concurrent tool execution
        self.tool_semaphore = asyncio.Semaphore(config.max_concurrent_tools)

        # Configure logging
        logging.getLogger(f"mcp.{config.name}").setLevel(config.log_level)

        # Initialize server
        self._setup_base_tools()
        self._setup_custom_tools()

    @abstractmethod
    def _setup_custom_tools(self) -> None:
        """Setup server-specific tools. Must be implemented by subclasses."""
        pass

    def _setup_base_tools(self) -> None:
        """Setup base tools available to all MCP servers."""

        @self.mcp.tool()
        async def server_health() -> dict[str, Any]:
            """Get server health and status information."""
            uptime = time.time() - self.start_time
            avg_execution_time = self.total_execution_time / max(1, self.request_count)

            return {
                "server_name": self.config.name,
                "version": self.config.version,
                "status": "healthy",
                "uptime_seconds": uptime,
                "total_requests": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(1, self.request_count),
                "average_execution_time_ms": avg_execution_time * 1000,
                "cache_entries": len(self.cache),
                "active_tools": len(self.tools),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        @self.mcp.tool()
        async def server_metrics() -> dict[str, Any]:
            """Get detailed server metrics and tool performance data."""
            if not self.config.enable_metrics:
                return {"error": "Metrics collection is disabled"}

            return {
                "server_metrics": {
                    "uptime_seconds": time.time() - self.start_time,
                    "total_requests": self.request_count,
                    "error_count": self.error_count,
                    "cache_hit_rate": self._calculate_cache_hit_rate(),
                },
                "tool_metrics": self.tool_metrics,
                "system_info": {
                    "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                    "fastmcp_version": "2.11.1",  # This should be dynamically determined
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }

        @self.mcp.tool()
        async def clear_cache() -> dict[str, Any]:
            """Clear the server cache (admin only)."""
            # This would typically require admin permissions
            cache_size = len(self.cache)
            self.cache.clear()

            return {
                "message": "Cache cleared successfully",
                "entries_removed": cache_size,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def register_tool(
        self,
        name: str,
        func: Callable,
        required_permission: str | None = None,
        enable_caching: bool | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        """Register a tool with the MCP server."""

        if enable_caching is None:
            enable_caching = self.config.enable_caching

        if timeout_seconds is None:
            timeout_seconds = self.config.tool_timeout_seconds

        # Wrap the function with our enhanced functionality
        async def wrapped_tool(*args, **kwargs) -> dict[str, Any]:
            return await self._execute_tool(
                name,
                func,
                args,
                kwargs,
                required_permission,
                enable_caching,
                timeout_seconds,
            )

        # Register with FastMCP
        self.mcp.tool()(wrapped_tool)

        # Store in our registry
        self.tools[name] = func
        self.tool_metrics[name] = {
            "call_count": 0,
            "total_execution_time": 0.0,
            "error_count": 0,
            "cache_hits": 0,
            "last_called": None,
        }

        logger.info(f"Registered tool: {name}")

    async def _execute_tool(
        self,
        tool_name: str,
        func: Callable,
        args: tuple,
        kwargs: dict[str, Any],
        required_permission: str | None,
        enable_caching: bool,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Execute a tool with full feature support."""
        start_time = time.time()
        request_id = kwargs.get("request_id", f"{tool_name}_{int(start_time)}")

        try:
            # Update counters
            self.request_count += 1
            self.tool_metrics[tool_name]["call_count"] += 1
            self.tool_metrics[tool_name]["last_called"] = datetime.now(UTC).isoformat()

            # Check authentication and permissions
            if self.config.require_auth:
                auth_info = kwargs.get("auth_info")
                if not auth_info:
                    raise PermissionError("Authentication required")

                if required_permission and not security_middleware.check_permissions(
                    auth_info, required_permission
                ):
                    raise PermissionError(f"Permission required: {required_permission}")

            # Check cache
            cache_key = None
            if enable_caching:
                cache_key = self._generate_cache_key(tool_name, args, kwargs)
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.tool_metrics[tool_name]["cache_hits"] += 1
                    execution_time = time.time() - start_time
                    self.total_execution_time += execution_time

                    return MCPResponse(
                        request_id=request_id,
                        success=True,
                        data=cached_result,
                        execution_time_ms=execution_time * 1000,
                        cached=True,
                    ).dict()

            # Execute with semaphore and timeout
            async with self.tool_semaphore:
                try:
                    result = await asyncio.wait_for(
                        self._call_function(func, args, kwargs), timeout=timeout_seconds
                    )

                    # Cache successful results
                    if enable_caching and cache_key:
                        self._store_in_cache(cache_key, result)

                    execution_time = time.time() - start_time
                    self.total_execution_time += execution_time
                    self.tool_metrics[tool_name]["total_execution_time"] += execution_time

                    return MCPResponse(
                        request_id=request_id,
                        success=True,
                        data=result,
                        execution_time_ms=execution_time * 1000,
                        cached=False,
                    ).dict()

                except TimeoutError:
                    raise Exception(f"Tool execution timeout ({timeout_seconds}s)")

        except Exception as e:
            self.error_count += 1
            self.tool_metrics[tool_name]["error_count"] += 1
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time

            logger.exception(f"Tool {tool_name} failed: {e}")

            return MCPResponse(
                request_id=request_id,
                success=False,
                error=str(e),
                execution_time_ms=execution_time * 1000,
                cached=False,
            ).dict()

    async def _call_function(self, func: Callable, args: tuple, kwargs: dict[str, Any]) -> Any:
        """Call function with proper async/sync handling."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _generate_cache_key(self, tool_name: str, args: tuple, kwargs: dict[str, Any]) -> str:
        """Generate a cache key for the tool call."""
        # Remove non-cacheable items like auth_info, request_id, etc.
        cacheable_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["auth_info", "request_id", "timestamp", "user_context"]
        }

        cache_data = {
            "tool": tool_name,
            "args": args,
            "kwargs": cacheable_kwargs,
        }

        cache_string = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.sha256(cache_string.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Any | None:
        """Get item from cache if not expired."""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl_seconds:
                return data
            else:
                # Remove expired item
                del self.cache[cache_key]
        return None

    def _store_in_cache(self, cache_key: str, data: Any) -> None:
        """Store item in cache with timestamp."""
        self.cache[cache_key] = (data, time.time())

        # Simple cache cleanup - remove oldest 10% when cache gets too large
        if len(self.cache) > 1000:  # Arbitrary limit
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
            items_to_remove = int(len(sorted_items) * 0.1)
            for i in range(items_to_remove):
                del self.cache[sorted_items[i][0]]

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate across all tools."""
        total_hits = sum(metrics["cache_hits"] for metrics in self.tool_metrics.values())
        total_calls = sum(metrics["call_count"] for metrics in self.tool_metrics.values())
        return total_hits / max(1, total_calls)

    def get_server_info(self) -> dict[str, Any]:
        """Get comprehensive server information."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "uptime": time.time() - self.start_time,
            "tools": list(self.tools.keys()),
            "config": {
                "require_auth": self.config.require_auth,
                "rate_limit_rpm": self.config.rate_limit_requests_per_minute,
                "caching_enabled": self.config.enable_caching,
                "cache_ttl": self.config.cache_ttl_seconds,
                "tool_timeout": self.config.tool_timeout_seconds,
            },
            "stats": {
                "total_requests": self.request_count,
                "total_errors": self.error_count,
                "cache_entries": len(self.cache),
                "avg_execution_time": self.total_execution_time / max(1, self.request_count),
            },
        }

    def run(self, host: str = "localhost", port: int = 3000) -> None:
        """Run the MCP server."""
        logger.info(f"Starting {self.config.name} MCP server on {host}:{port}")
        logger.info(f"Available tools: {list(self.tools.keys())}")

        try:
            self.mcp.run(host=host, port=port)
        except KeyboardInterrupt:
            logger.info(f"Shutting down {self.config.name} MCP server")
        except Exception as e:
            logger.exception(f"Server error: {e}")
            raise
