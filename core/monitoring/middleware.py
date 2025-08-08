#!/usr/bin/env python3
"""Monitoring Middleware.

FastAPI middleware for automatic monitoring data collection.
Integrates seamlessly with existing request/response cycle.
"""

from collections.abc import Callable
import logging
import time
import uuid

from fastapi import Request
from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware

from .monitoring.ai_metrics import OperationType

logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic monitoring data collection."""

    def __init__(self, app, monitoring_system=None) -> None:
        super().__init__(app)
        self.monitoring_system = monitoring_system
        self.request_id_header = "X-Request-ID"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect monitoring data."""
        # Generate or extract request ID
        request_id = request.headers.get(self.request_id_header, str(uuid.uuid4()))

        # Record request start
        start_time = time.time()

        # Extract session information
        session_id = self._extract_session_id(request)
        user_id = self._extract_user_id(request)

        try:
            # Start user session tracking if needed
            if session_id and self.monitoring_system:
                await self._start_user_session_if_needed(request, session_id, user_id)

            # Process request
            response = await call_next(request)

            # Record successful request completion
            await self._record_request_metrics(request, response, request_id, start_time, True)

            # Add request ID to response headers
            response.headers[self.request_id_header] = request_id

            return response

        except Exception as e:
            # Record failed request
            await self._record_request_metrics(request, None, request_id, start_time, False, str(e))
            raise

    def _extract_session_id(self, request: Request) -> str:
        """Extract session ID from request."""
        # Try multiple sources for session ID
        session_id = (
            request.headers.get("X-Session-ID")
            or request.cookies.get("session_id")
            or request.query_params.get("session_id")
        )

        if not session_id:
            # Generate session ID based on client info
            client_ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "")
            session_id = f"auto_{hash(f'{client_ip}_{user_agent}')}"

        return session_id

    def _extract_user_id(self, request: Request) -> str | None:
        """Extract user ID from request."""
        # Try multiple sources for user ID
        user_id = (
            request.headers.get("X-User-ID")
            or request.cookies.get("user_id")
            or request.query_params.get("user_id")
        )

        # Try to extract from JWT token if present
        if not user_id:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                try:
                    # In a real implementation, you'd decode the JWT
                    # For now, just extract a mock user ID
                    token = auth_header.split(" ")[1]
                    user_id = f"jwt_user_{hash(token) % 10000}"
                except Exception:
                    pass

        return user_id

    async def _start_user_session_if_needed(
        self, request: Request, session_id: str, user_id: str | None
    ) -> None:
        """Start user session tracking if monitoring system is available."""
        if not self.monitoring_system:
            return

        try:
            # Extract device and location info
            user_agent = request.headers.get("user-agent", "")
            device_type = self._detect_device_type(user_agent)

            # Extract location from headers (if available)
            location_info: dict[str, Any] = {}
            if "CF-IPCountry" in request.headers:  # Cloudflare country header
                location_info["country"] = request.headers["CF-IPCountry"]
            if "X-Forwarded-For" in request.headers:
                # Could use for geolocation lookup
                pass

            # Start session (will be ignored if already exists)
            await self.monitoring_system.start_user_session(
                session_id=session_id,
                user_id=user_id,
                user_agent=user_agent,
                device_info={"type": device_type},
                location_info=location_info if location_info else None,
            )

        except Exception as e:
            logger.debug(f"Error starting user session: {e}")

    def _detect_device_type(self, user_agent: str) -> str:
        """Detect device type from user agent."""
        user_agent_lower = user_agent.lower()

        if any(mobile in user_agent_lower for mobile in ["mobile", "android", "iphone"]):
            return "mobile"
        if any(tablet in user_agent_lower for tablet in ["tablet", "ipad"]):
            return "tablet"
        return "desktop"

    async def _record_request_metrics(
        self,
        request: Request,
        response: Response | None,
        request_id: str,
        start_time: float,
        success: bool,
        error_message: str = "",
    ) -> None:
        """Record request metrics in monitoring system."""
        if not self.monitoring_system:
            return

        try:
            # Calculate response time
            duration_ms = (time.time() - start_time) * 1000

            # Get request details
            method = request.method
            path = request.url.path
            status_code = response.status_code if response else 500

            # Determine operation type based on path
            self._determine_operation_type(path)

            # Record in APM system
            await self.monitoring_system.apm.record_operation(
                operation_name=f"{method} {path}",
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
                metadata={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "user_agent": request.headers.get("user-agent", ""),
                },
            )

            # If this is an AI-related endpoint, record AI metrics
            if self._is_ai_endpoint(path):
                await self._record_ai_operation_metrics(
                    request, response, request_id, duration_ms, success
                )

        except Exception as e:
            logger.debug(f"Error recording request metrics: {e}")

    def _determine_operation_type(self, path: str) -> str:
        """Determine operation type from request path."""
        if "/agents/" in path:
            return "agent_operation"
        if "/monitoring/" in path:
            return "monitoring_query"
        if "/health" in path:
            return "health_check"
        return "api_request"

    def _is_ai_endpoint(self, path: str) -> bool:
        """Check if endpoint is AI-related."""
        ai_paths = [
            "/agents/code-reviewer",
            "/agents/workspace-analyzer",
            "/agents/documentation-generator",
            "/agents/master-architect",
            "/agents/code-generator",
        ]
        return any(ai_path in path for ai_path in ai_paths)

    async def _record_ai_operation_metrics(
        self,
        request: Request,
        response: Response | None,
        request_id: str,
        duration_ms: float,
        success: bool,
    ) -> None:
        """Record AI-specific metrics for AI endpoints."""
        if not self.monitoring_system:
            return

        try:
            # Extract session ID
            session_id = self._extract_session_id(request)

            # Determine model and operation type from path
            path = request.url.path
            model_name = "gemini-2.5-pro"  # Default model
            operation_type = OperationType.INFERENCE

            if "code-reviewer" in path or "workspace-analyzer" in path:
                operation_type = OperationType.ANALYSIS
            elif "documentation-generator" in path:
                operation_type = OperationType.COMPLETION
            elif "master-architect" in path:
                operation_type = OperationType.ANALYSIS
            elif "code-generator" in path:
                operation_type = OperationType.COMPLETION

            # Start and immediately complete AI operation
            # (since we don't have separate start/complete for middleware)
            operation_id = await self.monitoring_system.record_ai_operation(
                session_id=session_id,
                model_name=model_name,
                operation_type=operation_type,
                input_tokens=self._estimate_input_tokens(request),
                context_size_tokens=0,
            )

            await self.monitoring_system.complete_ai_operation(
                operation_id=operation_id,
                output_tokens=self._estimate_output_tokens(response),
                quality_scores=None,  # Would need actual quality metrics
                success=success,
                error_message="" if success else "Request failed",
            )

        except Exception as e:
            logger.debug(f"Error recording AI operation metrics: {e}")

    def _estimate_input_tokens(self, request: Request) -> int:
        """Estimate input tokens from request content."""
        try:
            # Simple estimation based on content length
            content_length = int(request.headers.get("content-length", 0))
            # Rough estimation: 1 token per 4 characters
            return max(1, content_length // 4)
        except Exception:
            return 100  # Default estimate

    def _estimate_output_tokens(self, response: Response | None) -> int:
        """Estimate output tokens from response content."""
        if not response:
            return 0

        try:
            # Simple estimation based on content
            content_length = len(getattr(response, "body", b""))
            # Rough estimation: 1 token per 4 characters
            return max(1, content_length // 4)
        except Exception:
            return 50  # Default estimate


def get_monitoring_middleware(monitoring_system=None) -> None:
    """Factory function to create monitoring middleware with system dependency."""

    def create_middleware(app) -> None:
        return MonitoringMiddleware(app, monitoring_system)

    return create_middleware
