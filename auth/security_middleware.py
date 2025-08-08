"""Security middleware for GTerminal authentication and authorization.

Provides comprehensive security features including rate limiting, CORS,
security headers, and authentication/authorization middleware.
"""

import asyncio
from collections import defaultdict
from collections import deque
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from functools import wraps
import logging
import time
from typing import Any

import jwt

from .auth_jwt import JWTManager
from .auth_models import AuthEvent
from .auth_models import Permissions
from .auth_storage import auth_storage

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter with multiple strategies."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int | None = None,
        cleanup_interval: int = 300,  # 5 minutes
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        self.cleanup_interval = cleanup_interval

        # Client buckets: client_id -> (tokens, last_refill)
        self.buckets: dict[str, tuple[float, float]] = {}

        # Request history for analytics
        self.request_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Last cleanup time
        self.last_cleanup = time.time()

    def is_allowed(self, client_id: str, tokens_requested: int = 1) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()

        # Cleanup old buckets periodically
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_buckets(now)

        # Get or create bucket
        current_tokens, last_refill = self.buckets.get(client_id, (self.burst_size, now))

        # Calculate token refill
        time_passed = now - last_refill
        tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
        current_tokens = min(self.burst_size, current_tokens + tokens_to_add)

        # Check if request can be fulfilled
        if current_tokens >= tokens_requested:
            # Deduct tokens
            current_tokens -= tokens_requested
            self.buckets[client_id] = (current_tokens, now)

            # Record request
            self.request_history[client_id].append(now)

            return True

        # Update bucket even if request is denied
        self.buckets[client_id] = (current_tokens, now)
        return False

    def get_rate_limit_status(self, client_id: str) -> dict[str, Any]:
        """Get current rate limit status for a client."""
        now = time.time()
        current_tokens, last_refill = self.buckets.get(client_id, (self.burst_size, now))

        # Calculate current tokens
        time_passed = now - last_refill
        tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
        current_tokens = min(self.burst_size, current_tokens + tokens_to_add)

        # Calculate reset time
        if current_tokens < self.burst_size:
            seconds_to_full = (self.burst_size - current_tokens) / (self.requests_per_minute / 60.0)
            reset_time = now + seconds_to_full
        else:
            reset_time = now

        return {
            "limit": self.requests_per_minute,
            "remaining": int(current_tokens),
            "reset": int(reset_time),
            "retry_after": max(0, int((1 - current_tokens) / (self.requests_per_minute / 60.0)))
            if current_tokens < 1
            else 0,
        }

    def get_client_stats(self, client_id: str, hours: int = 24) -> dict[str, Any]:
        """Get request statistics for a client."""
        now = time.time()
        cutoff = now - (hours * 3600)

        history = self.request_history.get(client_id, deque())
        recent_requests = [t for t in history if t > cutoff]

        if not recent_requests:
            return {
                "total_requests": 0,
                "requests_per_hour": 0,
                "peak_hour_requests": 0,
                "first_request": None,
                "last_request": None,
            }

        # Calculate hourly distribution
        hourly_requests = defaultdict(int)
        for timestamp in recent_requests:
            hour_bucket = int(timestamp) // 3600
            hourly_requests[hour_bucket] += 1

        return {
            "total_requests": len(recent_requests),
            "requests_per_hour": len(recent_requests) / hours,
            "peak_hour_requests": max(hourly_requests.values()) if hourly_requests else 0,
            "first_request": datetime.fromtimestamp(min(recent_requests), UTC).isoformat(),
            "last_request": datetime.fromtimestamp(max(recent_requests), UTC).isoformat(),
        }

    def _cleanup_old_buckets(self, now: float) -> None:
        """Remove old, inactive buckets to prevent memory leaks."""
        cutoff = now - self.cleanup_interval

        old_buckets = [
            client_id
            for client_id, (_, last_refill) in self.buckets.items()
            if last_refill < cutoff
        ]

        for client_id in old_buckets:
            del self.buckets[client_id]
            if client_id in self.request_history:
                del self.request_history[client_id]

        self.last_cleanup = now

        if old_buckets:
            logger.info(f"Cleaned up {len(old_buckets)} inactive rate limit buckets")


class SecurityMiddleware:
    """Comprehensive security middleware for GTerminal applications."""

    def __init__(
        self,
        jwt_manager: JWTManager | None = None,
        rate_limiter: RateLimiter | None = None,
        enable_cors: bool = True,
        allowed_origins: list[str] | None = None,
    ):
        self.jwt_manager = jwt_manager
        self.rate_limiter = rate_limiter or RateLimiter()
        self.enable_cors = enable_cors
        self.allowed_origins = allowed_origins or ["http://localhost:3000", "http://localhost:8080"]

        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        }

    def get_client_id(self, request_info: dict[str, Any]) -> str:
        """Generate client ID for rate limiting."""
        # Try to use authenticated user ID first
        user_id = request_info.get("user_id")
        if user_id:
            return f"user:{user_id}"

        # Fall back to IP address
        ip_address = request_info.get("ip_address", "unknown")
        return f"ip:{ip_address}"

    async def authenticate_request(self, request_info: dict[str, Any]) -> dict[str, Any] | None:
        """Authenticate request using JWT token or API key."""
        auth_header = request_info.get("authorization", "")

        # Try JWT token authentication
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix

            if self.jwt_manager:
                try:
                    user_info = self.jwt_manager.get_user_from_token(token)

                    # Log successful authentication
                    await self._log_auth_event(
                        event_type="token_auth_success",
                        user_id=user_info["id"],
                        ip_address=request_info.get("ip_address"),
                        user_agent=request_info.get("user_agent"),
                        success=True,
                        metadata={"auth_method": "jwt"},
                    )

                    return {
                        "auth_type": "jwt",
                        "user": user_info,
                        "authenticated": True,
                    }

                except jwt.InvalidTokenError as e:
                    logger.warning(f"JWT authentication failed: {e}")

                    await self._log_auth_event(
                        event_type="token_auth_failed",
                        ip_address=request_info.get("ip_address"),
                        user_agent=request_info.get("user_agent"),
                        success=False,
                        metadata={"auth_method": "jwt", "error": str(e)},
                    )

        # Try API key authentication
        api_key = request_info.get("x_api_key") or request_info.get("api_key")
        if api_key:
            result = auth_storage.verify_api_key(api_key)
            if result:
                api_key_obj, user = result

                # Log successful authentication
                await self._log_auth_event(
                    event_type="api_key_auth_success",
                    user_id=user.id,
                    ip_address=request_info.get("ip_address"),
                    user_agent=request_info.get("user_agent"),
                    success=True,
                    metadata={"auth_method": "api_key", "key_name": api_key_obj.name},
                )

                return {
                    "auth_type": "api_key",
                    "api_key": api_key_obj,
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                        "role": user.role.value,
                        "permissions": list(user.permissions),
                        "is_active": user.is_active,
                        "is_verified": user.is_verified,
                    },
                    "authenticated": True,
                }
            else:
                await self._log_auth_event(
                    event_type="api_key_auth_failed",
                    ip_address=request_info.get("ip_address"),
                    user_agent=request_info.get("user_agent"),
                    success=False,
                    metadata={"auth_method": "api_key"},
                )

        return None

    def check_rate_limit(self, request_info: dict[str, Any]) -> dict[str, Any]:
        """Check rate limits for the request."""
        client_id = self.get_client_id(request_info)

        # Different limits for authenticated vs unauthenticated requests
        tokens_requested = 1
        if not request_info.get("authenticated"):
            tokens_requested = 2  # Higher cost for unauthenticated requests

        is_allowed = self.rate_limiter.is_allowed(client_id, tokens_requested)
        status = self.rate_limiter.get_rate_limit_status(client_id)

        return {
            "allowed": is_allowed,
            "status": status,
            "client_id": client_id,
        }

    def check_permissions(
        self, auth_info: dict[str, Any], required_permission: str, resource_id: str | None = None
    ) -> bool:
        """Check if authenticated user/key has required permissions."""
        if not auth_info.get("authenticated"):
            return False

        auth_type = auth_info.get("auth_type")

        if auth_type == "jwt":
            user_info = auth_info.get("user", {})
            permissions = set(user_info.get("permissions", []))
            role = user_info.get("role", "")

            # Admin role has all permissions
            if role == "admin" or Permissions.ADMIN in permissions:
                return True

            return required_permission in permissions

        elif auth_type == "api_key":
            api_key = auth_info.get("api_key")
            if not api_key:
                return False

            # Check API key scopes
            if Permissions.ADMIN in api_key.scopes:
                return True

            return required_permission in api_key.scopes

        return False

    def get_security_headers(self, request_info: dict[str, Any]) -> dict[str, str]:
        """Get security headers for response."""
        headers = self.security_headers.copy()

        # Add CORS headers if enabled
        if self.enable_cors:
            origin = request_info.get("origin")
            if origin in self.allowed_origins:
                headers.update(
                    {
                        "Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
                        "Access-Control-Max-Age": "86400",
                    }
                )

        # Add rate limit headers
        if "auth_info" in request_info:
            client_id = self.get_client_id(request_info)
            rate_status = self.rate_limiter.get_rate_limit_status(client_id)
            headers.update(
                {
                    "X-RateLimit-Limit": str(rate_status["limit"]),
                    "X-RateLimit-Remaining": str(rate_status["remaining"]),
                    "X-RateLimit-Reset": str(rate_status["reset"]),
                }
            )

            if rate_status["retry_after"] > 0:
                headers["Retry-After"] = str(rate_status["retry_after"])

        return headers

    async def process_request(self, request_info: dict[str, Any]) -> dict[str, Any]:
        """Process incoming request through security middleware."""
        result = {
            "allowed": True,
            "auth_info": None,
            "rate_limit_info": None,
            "security_headers": {},
            "errors": [],
        }

        try:
            # Check rate limits first
            rate_limit_info = self.check_rate_limit(request_info)
            result["rate_limit_info"] = rate_limit_info

            if not rate_limit_info["allowed"]:
                result["allowed"] = False
                result["errors"].append("Rate limit exceeded")
                result["security_headers"] = self.get_security_headers(request_info)
                return result

            # Attempt authentication
            auth_info = await self.authenticate_request(request_info)
            if auth_info:
                result["auth_info"] = auth_info
                request_info.update(auth_info)  # Add auth info for other checks

            # Get security headers
            result["security_headers"] = self.get_security_headers(request_info)

        except Exception as e:
            logger.exception(f"Security middleware error: {e}")
            result["allowed"] = False
            result["errors"].append("Security processing failed")

        return result

    async def _log_auth_event(
        self,
        event_type: str,
        success: bool,
        user_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Log authentication event."""
        try:
            from .auth_models import AuthProvider

            event = AuthEvent(
                user_id=user_id,
                event_type=event_type,
                success=success,
                provider=AuthProvider.LOCAL,  # Default provider
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=metadata or {},
            )

            auth_storage.log_auth_event(event)

        except Exception as e:
            logger.exception(f"Failed to log auth event: {e}")

    def get_security_report(self) -> dict[str, Any]:
        """Generate security monitoring report."""
        now = datetime.now(UTC)

        # Get recent auth events
        recent_events = auth_storage.get_auth_events(limit=1000)

        # Analyze events
        failed_logins = [e for e in recent_events if not e.success and "auth" in e.event_type]
        successful_logins = [e for e in recent_events if e.success and "auth" in e.event_type]

        # Rate limiting stats
        rate_limit_stats = {
            "active_clients": len(self.rate_limiter.buckets),
            "total_requests_tracked": sum(
                len(hist) for hist in self.rate_limiter.request_history.values()
            ),
        }

        # Security alerts
        alerts = []

        # Check for suspicious activity
        if len(failed_logins) > 10:  # More than 10 failed logins recently
            alerts.append(
                {
                    "type": "high_failed_logins",
                    "severity": "medium",
                    "count": len(failed_logins),
                    "message": f"{len(failed_logins)} failed login attempts detected",
                }
            )

        # Check for rate limit violations (high request counts)
        for client_id, history in self.rate_limiter.request_history.items():
            if len(history) > 500:  # More than 500 requests tracked
                alerts.append(
                    {
                        "type": "high_request_volume",
                        "severity": "low",
                        "client_id": client_id,
                        "count": len(history),
                        "message": f"High request volume from {client_id}: {len(history)} requests",
                    }
                )

        return {
            "report_generated": now.isoformat(),
            "authentication": {
                "successful_logins": len(successful_logins),
                "failed_logins": len(failed_logins),
                "success_rate": len(successful_logins) / max(1, len(recent_events)) * 100,
            },
            "rate_limiting": rate_limit_stats,
            "security_alerts": alerts,
            "alert_level": "high"
            if any(a["severity"] == "high" for a in alerts)
            else "medium"
            if alerts
            else "low",
        }


# Authentication decorator for functions
def require_auth(permission: str | None = None):
    """Decorator to require authentication for functions."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract request info from arguments
            request_info = kwargs.get("request_info") or {}

            middleware = SecurityMiddleware()
            result = await middleware.process_request(request_info)

            if not result["allowed"]:
                raise PermissionError(f"Access denied: {', '.join(result['errors'])}")

            auth_info = result.get("auth_info")
            if not auth_info:
                raise PermissionError("Authentication required")

            # Check specific permission if required
            if permission and not middleware.check_permissions(auth_info, permission):
                raise PermissionError(f"Insufficient permissions: {permission} required")

            # Add auth info to kwargs
            kwargs["auth_info"] = auth_info
            kwargs["security_headers"] = result["security_headers"]

            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run the async authentication in an event loop
            return asyncio.run(async_wrapper(*args, **kwargs))

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Global security middleware instance
security_middleware = SecurityMiddleware()
