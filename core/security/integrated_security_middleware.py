from typing import Any

"""
Integrated security middleware that combines all security components.

This middleware integrates:
- Input validation and sanitization
- Audit logging
- Security monitoring
- Rate limiting
- IP blocking
- Request/response security
"""

from collections.abc import Callable
import logging
import time

from fastapi import Request
from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .audit_logger import SecurityEventType
from .audit_logger import SecurityLevel
from .audit_logger import audit_logger
from .input_validator import input_validator
from .security_monitor import initialize_security_monitor
from .security_monitor import security_monitor

logger = logging.getLogger(__name__)


class IntegratedSecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware integrating all security components."""

    def __init__(self, app) -> None:
        """Initialize integrated security middleware."""
        super().__init__(app)

        # Initialize security monitor if not already done
        if security_monitor is None:
            initialize_security_monitor()

        # Rate limiting configuration
        self.rate_limits = {
            "/api/auth/": {"requests": 10, "window": 60},  # 10 requests per minute for auth
            "/api/": {"requests": 100, "window": 60},  # 100 requests per minute for API
            "default": {"requests": 200, "window": 60},  # 200 requests per minute default
        }

        # Request tracking
        self.request_counts = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive security controls."""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        request_id = self._generate_request_id()

        # Add request ID to request state
        request.state.request_id = request_id

        try:
            # 1. Check if IP is blocked
            if security_monitor and security_monitor.is_ip_blocked(client_ip):
                await self._log_blocked_request(client_ip, request, request_id)
                return JSONResponse(
                    status_code=403,
                    content={"error": "Access denied", "message": "IP address is blocked"},
                )

            # 2. Rate limiting
            if not await self._check_rate_limit(request, client_ip):
                await self._handle_rate_limit_exceeded(client_ip, request, request_id)
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded", "message": "Too many requests"},
                )

            # 3. Input validation for query parameters and path
            try:
                await self._validate_request_inputs(request)
            except ValueError as e:
                await self._handle_input_validation_failure(str(e), client_ip, request, request_id)
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid input", "message": "Request contains invalid data"},
                )

            # 4. Process request
            response = await call_next(request)

            # 5. Post-process response
            await self._post_process_response(request, response, client_ip, request_id, start_time)

            return response

        except Exception as e:
            # Handle unexpected errors securely
            await self._handle_unexpected_error(e, client_ip, request, request_id)
            logger.exception(f"Unexpected error in security middleware: {e}")

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An error occurred processing your request",
                },
            )

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid

        return f"req_{uuid.uuid4().hex[:12]}"

    async def _check_rate_limit(self, request: Request, client_ip: str) -> bool:
        """Check if request is within rate limits."""
        # Determine rate limit based on path
        path = request.url.path
        rate_config = self.rate_limits["default"]

        for path_prefix, config in self.rate_limits.items():
            if path_prefix != "default" and path.startswith(path_prefix):
                rate_config = config
                break

        # Get current time window
        current_time = int(time.time())
        window_start = current_time - (current_time % rate_config["window"])

        # Create key for this IP and window
        rate_key = f"{client_ip}:{window_start}"

        # Get current count
        current_count = self.request_counts.get(rate_key, 0)

        # Check limit
        if current_count >= rate_config["requests"]:
            return False

        # Increment counter
        self.request_counts[rate_key] = current_count + 1

        # Clean up old entries (simple cleanup)
        if len(self.request_counts) > 10000:
            await self._cleanup_rate_limit_data()

        return True

    async def _cleanup_rate_limit_data(self) -> None:
        """Clean up old rate limiting data."""
        current_time = int(time.time())
        cutoff_time = current_time - 3600  # Keep last hour

        keys_to_remove: list[Any] = []
        for key in self.request_counts:
            try:
                window_time = int(key.split(":")[-1])
                if window_time < cutoff_time:
                    keys_to_remove.append(key)
            except (ValueError, IndexError):
                # Invalid key format, remove it
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.request_counts[key]

    async def _validate_request_inputs(self, request: Request) -> None:
        """Validate request inputs for security."""
        # Validate query parameters
        for key, value in request.query_params.items():
            try:
                # Basic validation for all query params
                input_validator.validate_string_input(
                    key, max_length=100, check_sql_injection=True, check_xss=True
                )
                input_validator.validate_string_input(
                    value, max_length=1000, check_sql_injection=True, check_xss=True
                )
            except ValueError as e:
                msg = f"Invalid query parameter '{key}': {e}"
                raise ValueError(msg)

        # Validate path parameters
        path = request.url.path
        try:
            input_validator.validate_string_input(
                path,
                max_length=500,
                check_sql_injection=True,
                check_command_injection=True,
            )
        except ValueError as e:
            msg = f"Invalid request path: {e}"
            raise ValueError(msg)

        # Additional validation for file paths in query params
        for param_name in ["file_path", "path", "filename", "directory"]:
            if param_name in request.query_params:
                try:
                    input_validator.validate_file_path(request.query_params[param_name])
                except ValueError as e:
                    msg = f"Invalid file path parameter '{param_name}': {e}"
                    raise ValueError(msg)

    async def _log_blocked_request(self, client_ip: str, request: Request, request_id: str) -> None:
        """Log blocked request attempt."""
        audit_logger.log_security_event(
            SecurityEventType.SECURITY_VIOLATION,
            SecurityLevel.HIGH,
            "Request from blocked IP address",
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent"),
            request_id=request_id,
            resource=request.url.path,
            action="blocked_ip_access",
            metadata={"method": request.method, "url": str(request.url), "blocked": True},
        )

    async def _handle_rate_limit_exceeded(
        self, client_ip: str, request: Request, request_id: str
    ) -> None:
        """Handle rate limit exceeded."""
        # Log the event
        audit_logger.log_security_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            SecurityLevel.MEDIUM,
            "Rate limit exceeded",
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent"),
            request_id=request_id,
            resource=request.url.path,
            action="rate_limit_exceeded",
            metadata={"method": request.method, "url": str(request.url)},
        )

        # Track in security monitor
        if security_monitor:
            security_monitor.track_security_event(
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                SecurityLevel.MEDIUM,
                client_ip=client_ip,
                resource=request.url.path,
                metadata={"method": request.method},
            )

    async def _handle_input_validation_failure(
        self,
        error_message: str,
        client_ip: str,
        request: Request,
        request_id: str,
    ) -> None:
        """Handle input validation failure."""
        # Log the event
        audit_logger.log_security_event(
            SecurityEventType.INPUT_VALIDATION_FAILURE,
            SecurityLevel.HIGH,
            f"Input validation failed: {error_message}",
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent"),
            request_id=request_id,
            resource=request.url.path,
            action="input_validation_failure",
            metadata={
                "method": request.method,
                "url": str(request.url),
                "error": error_message,
                "query_params": dict(request.query_params),
            },
        )

        # Track in security monitor
        if security_monitor:
            security_monitor.track_security_event(
                SecurityEventType.INPUT_VALIDATION_FAILURE,
                SecurityLevel.HIGH,
                client_ip=client_ip,
                resource=request.url.path,
                metadata={"method": request.method, "error": error_message},
            )

    async def _post_process_response(
        self,
        request: Request,
        response: Response,
        client_ip: str,
        request_id: str,
        start_time: float,
    ) -> None:
        """Post-process response for security logging."""
        processing_time = time.time() - start_time

        # Log successful requests (INFO level for audit trail)
        if response.status_code < 400:
            audit_logger.log_security_event(
                (
                    SecurityEventType.FILE_ACCESS_ATTEMPT
                    if "file" in request.url.path
                    else SecurityEventType.AUTHORIZATION_SUCCESS
                ),
                SecurityLevel.LOW,
                f"Request processed successfully: {request.method} {request.url.path}",
                client_ip=client_ip,
                user_agent=request.headers.get("user-agent"),
                request_id=request_id,
                resource=request.url.path,
                action=request.method.lower(),
                metadata={
                    "status_code": response.status_code,
                    "processing_time": processing_time,
                    "response_size": len(response.body) if hasattr(response, "body") else 0,
                },
            )
        else:
            # Log failed requests
            severity = (
                SecurityLevel.HIGH if response.status_code in [401, 403] else SecurityLevel.MEDIUM
            )

            audit_logger.log_security_event(
                (
                    SecurityEventType.AUTHORIZATION_FAILURE
                    if response.status_code in [401, 403]
                    else SecurityEventType.SECURITY_VIOLATION
                ),
                severity,
                f"Request failed: {request.method} {request.url.path} - Status {response.status_code}",
                client_ip=client_ip,
                user_agent=request.headers.get("user-agent"),
                request_id=request_id,
                resource=request.url.path,
                action=request.method.lower(),
                metadata={"status_code": response.status_code, "processing_time": processing_time},
            )

            # Track in security monitor for 4xx/5xx responses
            if security_monitor and response.status_code >= 400:
                event_type = (
                    SecurityEventType.AUTHORIZATION_FAILURE
                    if response.status_code in [401, 403]
                    else SecurityEventType.SECURITY_VIOLATION
                )
                security_monitor.track_security_event(
                    event_type,
                    severity,
                    client_ip=client_ip,
                    resource=request.url.path,
                    metadata={"status_code": response.status_code, "method": request.method},
                )

    async def _handle_unexpected_error(
        self,
        error: Exception,
        client_ip: str,
        request: Request,
        request_id: str,
    ) -> None:
        """Handle unexpected errors."""
        audit_logger.log_security_event(
            SecurityEventType.SECURITY_VIOLATION,
            SecurityLevel.HIGH,
            f"Unexpected error in security middleware: {type(error).__name__}",
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent"),
            request_id=request_id,
            resource=request.url.path,
            action="middleware_error",
            metadata={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "method": request.method,
                "url": str(request.url),
            },
        )

        # Track in security monitor
        if security_monitor:
            security_monitor.track_security_event(
                SecurityEventType.SECURITY_VIOLATION,
                SecurityLevel.HIGH,
                client_ip=client_ip,
                resource=request.url.path,
                metadata={"error_type": type(error).__name__, "method": request.method},
            )
