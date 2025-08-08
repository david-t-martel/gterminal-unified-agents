"""JWT token management and password utilities for GTerminal authentication.

Provides secure JWT token generation/validation and password hashing/verification
with industry best practices and configurable security parameters.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
import logging
import secrets
from typing import Any

import jwt
from passlib.context import CryptContext

from .auth_models import AuthProvider
from .auth_models import User
from .auth_models import UserRole

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """Token type enumeration to avoid hardcoded strings."""

    ACCESS = "access"
    REFRESH = "refresh"


class PasswordManager:
    """Secure password hashing and verification using Argon2."""

    def __init__(self):
        # Use Argon2 with secure parameters
        self.pwd_context = CryptContext(
            schemes=["argon2"],
            deprecated="auto",
            argon2__memory_cost=65536,  # 64 MB
            argon2__time_cost=3,  # 3 iterations
            argon2__parallelism=2,  # 2 threads
        )

    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        if not password or len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")

        return self.pwd_context.hash(password)

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        try:
            return self.pwd_context.verify(password, hashed)
        except (ValueError, TypeError) as e:
            logger.warning(f"Password verification failed: {e}")
            return False

    def needs_update(self, hashed: str) -> bool:
        """Check if password hash needs updating."""
        return self.pwd_context.needs_update(hashed)

    def generate_password(self, length: int = 16) -> str:
        """Generate a secure random password."""
        if length < 8:
            raise ValueError("Password length must be at least 8")

        # Mix of letters, numbers, and symbols for strong passwords
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return "".join(secrets.choice(alphabet) for _ in range(length))


class JWTManager:
    """JWT token management with comprehensive security features."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 60,
        refresh_token_expire_days: int = 7,
        issuer: str = "gterminal",
    ):
        if not secret_key or len(secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")

        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.issuer = issuer

        # Token blacklist for revoked tokens (in production, use Redis)
        self._blacklisted_tokens = set()

    def create_access_token(
        self,
        user: User,
        expires_delta: timedelta | None = None,
        additional_claims: dict[str, Any] | None = None,
    ) -> str:
        """Create a JWT access token for a user."""
        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(minutes=self.access_token_expire_minutes)

        # Standard JWT claims
        payload = {
            "sub": user.id,  # Subject (user ID)
            "iss": self.issuer,  # Issuer
            "iat": datetime.now(UTC),  # Issued at
            "exp": expire,  # Expiration
            "jti": secrets.token_hex(16),  # JWT ID (for revocation)
            # Custom claims
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "permissions": list(user.permissions),
            "provider": user.provider.value,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
        }

        # Add any additional claims
        if additional_claims:
            payload.update(additional_claims)

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created access token for user {user.username}")
            return token
        except Exception:
            logger.exception("Failed to create access token")
            raise

    def create_refresh_token(self, user: User) -> str:
        """Create a refresh token for long-term authentication."""
        expire = datetime.now(UTC) + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "sub": user.id,
            "iss": self.issuer,
            "iat": datetime.now(UTC),
            "exp": expire,
            "jti": secrets.token_hex(16),
            "type": TokenType.REFRESH.value,
            "username": user.username,
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created refresh token for user {user.username}")
            return token
        except Exception:
            logger.exception("Failed to create refresh token")
            raise

    def verify_token(self, token: str, token_type: str = TokenType.ACCESS.value) -> dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            # Check if token is blacklisted
            if token in self._blacklisted_tokens:
                raise jwt.InvalidTokenError("Token has been revoked")

            # Decode and verify token
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm], issuer=self.issuer
            )

            # Verify token type for refresh tokens
            if (
                token_type == TokenType.REFRESH.value
                and payload.get("type") != TokenType.REFRESH.value
            ):
                raise jwt.InvalidTokenError("Invalid token type")

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise
        except Exception:
            logger.exception("Token verification failed")
            raise jwt.InvalidTokenError("Token verification failed")

    def get_user_from_token(self, token: str) -> dict[str, Any]:
        """Extract user information from a valid token."""
        payload = self.verify_token(token)

        return {
            "id": payload["sub"],
            "username": payload.get("username"),
            "email": payload.get("email"),
            "role": payload.get("role"),
            "permissions": payload.get("permissions", []),
            "provider": payload.get("provider"),
            "is_active": payload.get("is_active", True),
            "is_verified": payload.get("is_verified", False),
        }

    def revoke_token(self, token: str) -> None:
        """Revoke a token by adding it to blacklist."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False},  # Don't verify expiration for revocation
            )
            jti = payload.get("jti")
            if jti:
                self._blacklisted_tokens.add(token)
                logger.info(f"Revoked token with JTI: {jti}")
        except (jwt.InvalidTokenError, jwt.DecodeError) as e:
            logger.warning(f"Failed to revoke token: {e}")

    def refresh_access_token(self, refresh_token: str, user: User) -> str:
        """Create a new access token using a valid refresh token."""
        # Verify refresh token
        payload = self.verify_token(refresh_token, token_type=TokenType.REFRESH.value)

        # Ensure refresh token belongs to the user
        if payload["sub"] != user.id:
            raise jwt.InvalidTokenError("Refresh token does not match user")

        # Create new access token
        return self.create_access_token(user)

    def cleanup_blacklist(self) -> None:
        """Clean up expired tokens from blacklist (call periodically)."""
        # In production, implement proper cleanup based on token expiration
        # For now, this is a placeholder for the in-memory blacklist
        logger.info("Token blacklist cleanup completed")


def create_default_admin_user() -> User:
    """Create default admin user for initial setup."""
    password_manager = PasswordManager()

    # Generate secure default password
    default_password = password_manager.generate_password(16)
    password_hash = password_manager.hash_password(default_password)

    admin_user = User(
        username="admin",
        email="admin@gterminal.local",
        full_name="System Administrator",
        password_hash=password_hash,
        role=UserRole.ADMIN,
        is_active=True,
        is_verified=True,
        provider=AuthProvider.LOCAL,
        permissions={
            "admin",
            "user_management",
            "api_key_management",
            "terminal_access",
            "mcp_access",
            "react_agent_access",
        },
    )

    logger.info(f"Created default admin user with password: {default_password}")
    logger.warning("Please change the default admin password immediately!")

    return admin_user


# Global instances (configure with proper secrets in production)
password_manager = PasswordManager()
