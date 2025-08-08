"""Authentication data models for GTerminal.

Defines core authentication entities including Users, API Keys, and Auth Providers.
Based on proven patterns from production systems with enhanced security.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
import hashlib
import secrets
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field
from pydantic import validator


class AuthProvider(str, Enum):
    """Supported authentication providers."""

    LOCAL = "local"
    GOOGLE = "google"
    GITHUB = "github"
    API_KEY = "api_key"


class UserRole(str, Enum):
    """User roles with hierarchical permissions."""

    ADMIN = "admin"
    USER = "user"
    AGENT = "agent"  # For service accounts
    READONLY = "readonly"


class User(BaseModel):
    """User account model with comprehensive security features."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique user identifier")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="User email address")
    full_name: str | None = Field(None, max_length=100, description="Full display name")

    # Authentication
    password_hash: str | None = Field(None, description="Hashed password")
    provider: AuthProvider = Field(AuthProvider.LOCAL, description="Primary auth provider")
    provider_id: str | None = Field(None, description="External provider user ID")

    # Status and security
    is_active: bool = Field(True, description="Account is active")
    is_verified: bool = Field(False, description="Email verified")
    role: UserRole = Field(UserRole.USER, description="User role")
    permissions: set[str] = Field(default_factory=set, description="Specific permissions")

    # Security tracking
    failed_login_attempts: int = Field(0, description="Failed login count")
    locked_until: datetime | None = Field(None, description="Account locked until")
    last_login: datetime | None = Field(None, description="Last successful login")
    last_password_change: datetime | None = Field(None, description="Last password change")

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional user data")

    @validator("email")
    def validate_email(cls, v):
        """Basic email validation."""
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email format")
        return v.lower()

    @validator("username")
    def validate_username(cls, v):
        """Username validation."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username can only contain letters, numbers, hyphens, and underscores")
        return v.lower()

    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if not self.locked_until:
            return False
        return datetime.now(UTC) < self.locked_until

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        if self.role == UserRole.ADMIN:
            return True
        return permission in self.permissions

    def add_permission(self, permission: str) -> None:
        """Add permission to user."""
        self.permissions.add(permission)
        self.updated_at = datetime.now(UTC)

    def remove_permission(self, permission: str) -> None:
        """Remove permission from user."""
        self.permissions.discard(permission)
        self.updated_at = datetime.now(UTC)


class APIKey(BaseModel):
    """API Key model with scoping and security features."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique key identifier")
    user_id: str = Field(..., description="Owner user ID")
    name: str = Field(..., min_length=1, max_length=100, description="Key name/description")

    # Key data
    key_hash: str = Field(..., description="Hashed API key")
    key_prefix: str = Field(..., description="Key prefix for identification")

    # Permissions and scoping
    scopes: set[str] = Field(default_factory=set, description="API scopes")
    is_active: bool = Field(True, description="Key is active")

    # Usage tracking
    usage_count: int = Field(0, description="Number of times used")
    last_used: datetime | None = Field(None, description="Last usage timestamp")
    last_used_ip: str | None = Field(None, description="Last used IP address")

    # Lifecycle
    expires_at: datetime | None = Field(None, description="Key expiration")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    revoked_at: datetime | None = Field(None, description="When key was revoked")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional key data")

    @classmethod
    def generate_key(
        cls,
        user_id: str,
        name: str,
        scopes: set[str] | None = None,
        expires_days: int | None = None,
    ) -> tuple["APIKey", str]:
        """Generate a new API key and return both the model and raw key."""
        # Generate a secure random key
        raw_key = secrets.token_urlsafe(32)
        key_prefix = raw_key[:8]

        # Hash the key for storage
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        # Set expiration if specified
        expires_at = None
        if expires_days:
            expires_at = datetime.now(UTC) + timedelta(days=expires_days)

        api_key = cls(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            scopes=scopes or set(),
            expires_at=expires_at,
        )

        return api_key, raw_key

    def verify_key(self, raw_key: str) -> bool:
        """Verify a raw key against this API key."""
        if not self.is_active or self.is_expired() or self.is_revoked():
            return False

        # Verify the key hash
        provided_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        return secrets.compare_digest(self.key_hash, provided_hash)

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if not self.expires_at:
            return False
        return datetime.now(UTC) >= self.expires_at

    def is_revoked(self) -> bool:
        """Check if key is revoked."""
        return self.revoked_at is not None

    def revoke(self) -> None:
        """Revoke the API key."""
        self.revoked_at = datetime.now(UTC)
        self.is_active = False

    def has_scope(self, scope: str) -> bool:
        """Check if key has specific scope."""
        return scope in self.scopes or "admin" in self.scopes


class Session(BaseModel):
    """User session model for web-based authentication."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Session identifier")
    user_id: str = Field(..., description="User ID")

    # Session data
    jwt_token: str = Field(..., description="JWT token")
    refresh_token: str | None = Field(None, description="Refresh token")

    # Session metadata
    ip_address: str | None = Field(None, description="Client IP")
    user_agent: str | None = Field(None, description="Client user agent")

    # Lifecycle
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = Field(..., description="Session expiration")
    last_activity: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Status
    is_active: bool = Field(True, description="Session is active")
    revoked_at: datetime | None = Field(None, description="When session was revoked")

    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(UTC) >= self.expires_at

    def is_valid(self) -> bool:
        """Check if session is valid (active and not expired)."""
        return self.is_active and not self.is_expired() and self.revoked_at is None

    def revoke(self) -> None:
        """Revoke the session."""
        self.revoked_at = datetime.now(UTC)
        self.is_active = False

    def extend(self, additional_seconds: int = 3600) -> None:
        """Extend session expiration."""
        self.expires_at = max(
            self.expires_at, datetime.now(UTC) + timedelta(seconds=additional_seconds)
        )
        self.last_activity = datetime.now(UTC)


class AuthEvent(BaseModel):
    """Authentication event for audit logging."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str | None = Field(None, description="User ID (if applicable)")

    # Event details
    event_type: str = Field(..., description="Event type (login, logout, failed_login, etc.)")
    success: bool = Field(..., description="Whether the event was successful")
    provider: AuthProvider = Field(..., description="Auth provider used")

    # Context
    ip_address: str | None = Field(None, description="Client IP")
    user_agent: str | None = Field(None, description="Client user agent")

    # Additional data
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional event data")

    # Timestamp
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Permissions constants
class Permissions:
    """Standard permission constants."""

    # System permissions
    ADMIN = "admin"
    USER_MANAGEMENT = "user_management"
    API_KEY_MANAGEMENT = "api_key_management"

    # Application permissions
    TERMINAL_ACCESS = "terminal_access"
    MCP_ACCESS = "mcp_access"
    REACT_AGENT_ACCESS = "react_agent_access"

    # Resource permissions
    READ_SESSIONS = "read_sessions"
    WRITE_SESSIONS = "write_sessions"
    DELETE_SESSIONS = "delete_sessions"

    # Integration permissions
    GOOGLE_CLOUD_ACCESS = "google_cloud_access"
    GITHUB_ACCESS = "github_access"
