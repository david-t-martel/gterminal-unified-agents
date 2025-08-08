"""Security headers middleware for FastAPI applications.

Implements comprehensive security headers following OWASP recommendations
and security best practices.
"""

from collections.abc import Callable
import logging

from fastapi import Request
from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    def __init__(
        self,
        app,
        strict_transport_security: bool = True,
        content_security_policy: str | None = None,
        x_frame_options: str = "DENY",
        x_content_type_options: bool = True,
        referrer_policy: str = "strict-origin-when-cross-origin",
        permissions_policy: str | None = None,
        cross_origin_embedder_policy: str = "require-corp",
        cross_origin_opener_policy: str = "same-origin",
        cross_origin_resource_policy: str = "same-origin",
    ) -> None:
        """Initialize security headers middleware.

        Args:
            app: FastAPI application instance
            strict_transport_security: Enable HSTS header
            content_security_policy: CSP header value
            x_frame_options: X-Frame-Options header value
            x_content_type_options: Enable X-Content-Type-Options header
            referrer_policy: Referrer-Policy header value
            permissions_policy: Permissions-Policy header value
            cross_origin_embedder_policy: COEP header value
            cross_origin_opener_policy: COOP header value
            cross_origin_resource_policy: CORP header value

        """
        super().__init__(app)
        self.strict_transport_security = strict_transport_security
        self.content_security_policy = content_security_policy or self._default_csp()
        self.x_frame_options = x_frame_options
        self.x_content_type_options = x_content_type_options
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy or self._default_permissions_policy()
        self.cross_origin_embedder_policy = cross_origin_embedder_policy
        self.cross_origin_opener_policy = cross_origin_opener_policy
        self.cross_origin_resource_policy = cross_origin_resource_policy

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add security headers to response."""
        response = await call_next(request)

        # Add security headers
        self._add_security_headers(response)

        return response

    def _add_security_headers(self, response: StarletteResponse) -> None:
        """Add security headers to response."""
        # Strict Transport Security (HSTS)
        if self.strict_transport_security:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Content Security Policy
        if self.content_security_policy:
            response.headers["Content-Security-Policy"] = self.content_security_policy

        # X-Frame-Options
        if self.x_frame_options:
            response.headers["X-Frame-Options"] = self.x_frame_options

        # X-Content-Type-Options
        if self.x_content_type_options:
            response.headers["X-Content-Type-Options"] = "nosniff"

        # Referrer Policy
        if self.referrer_policy:
            response.headers["Referrer-Policy"] = self.referrer_policy

        # Permissions Policy
        if self.permissions_policy:
            response.headers["Permissions-Policy"] = self.permissions_policy

        # Cross-Origin Embedder Policy
        if self.cross_origin_embedder_policy:
            response.headers["Cross-Origin-Embedder-Policy"] = self.cross_origin_embedder_policy

        # Cross-Origin Opener Policy
        if self.cross_origin_opener_policy:
            response.headers["Cross-Origin-Opener-Policy"] = self.cross_origin_opener_policy

        # Cross-Origin Resource Policy
        if self.cross_origin_resource_policy:
            response.headers["Cross-Origin-Resource-Policy"] = self.cross_origin_resource_policy

        # Additional security headers
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"

        # Remove server header for security through obscurity
        if "server" in response.headers:
            del response.headers["server"]
        if "Server" in response.headers:
            del response.headers["Server"]

        # Remove X-Powered-By if present
        if "x-powered-by" in response.headers:
            del response.headers["x-powered-by"]
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]

    def _default_csp(self) -> str:
        """Generate default Content Security Policy."""
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "media-src 'self'; "
            "object-src 'none'; "
            "frame-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'; "
            "upgrade-insecure-requests"
        )

    def _default_permissions_policy(self) -> str:
        """Generate default Permissions Policy."""
        return (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "speaker=(), "
            "vibrate=(), "
            "fullscreen=(), "
            "payment=(), "
            "sync-xhr=()"
        )


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced CORS middleware with security controls."""

    def __init__(
        self,
        app,
        allowed_origins: list[str] | None = None,
        allowed_methods: list[str] | None = None,
        allowed_headers: list[str] | None = None,
        allow_credentials: bool = False,
        max_age: int = 600,
    ) -> None:
        """Initialize CORS security middleware.

        Args:
            app: FastAPI application instance
            allowed_origins: List of allowed origins (no wildcards in production)
            allowed_methods: List of allowed HTTP methods
            allowed_headers: List of allowed headers
            allow_credentials: Whether to allow credentials
            max_age: Preflight cache duration in seconds

        """
        super().__init__(app)
        self.allowed_origins = allowed_origins or []
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-API-Key",
        ]
        self.allow_credentials = allow_credentials
        self.max_age = max_age

        # Validate configuration
        if "*" in self.allowed_origins and allow_credentials:
            msg = "Cannot use wildcard origin with credentials"
            raise ValueError(msg)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process CORS for requests."""
        origin = request.headers.get("origin")

        # Handle preflight requests
        if request.method == "OPTIONS":
            return self._handle_preflight(request, origin)

        # Process actual request
        response = await call_next(request)

        # Add CORS headers to response
        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"

        return response

    def _handle_preflight(self, request: Request, origin: str) -> Response:
        """Handle CORS preflight requests."""
        if not origin or not self._is_origin_allowed(origin):
            return Response(status_code=403)

        # Check requested method
        requested_method = request.headers.get("access-control-request-method")
        if requested_method and requested_method not in self.allowed_methods:
            return Response(status_code=403)

        # Check requested headers
        requested_headers = request.headers.get("access-control-request-headers")
        if requested_headers:
            headers = [h.strip().lower() for h in requested_headers.split(",")]
            allowed_headers_lower = [h.lower() for h in self.allowed_headers]
            if not all(h in allowed_headers_lower for h in headers):
                return Response(status_code=403)

        # Create preflight response
        response = Response(status_code=200)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        response.headers["Access-Control-Max-Age"] = str(self.max_age)

        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"

        return response

    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if "*" in self.allowed_origins:
            return True
        return origin in self.allowed_origins
