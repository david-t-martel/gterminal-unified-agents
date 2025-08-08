"""API Key management for GTerminal authentication.

Provides comprehensive API key management with scoping, rate limiting,
and secure key generation patterns.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
import logging

from .auth_models import APIKey
from .auth_models import Permissions
from .auth_models import User
from .auth_storage import auth_storage

logger = logging.getLogger(__name__)


class APIKeyManager:
    """API Key management with advanced features."""

    def __init__(self):
        self.storage = auth_storage

        # Default scopes for different key types
        self.default_scopes = {
            "admin": {
                Permissions.ADMIN,
                Permissions.USER_MANAGEMENT,
                Permissions.API_KEY_MANAGEMENT,
                Permissions.TERMINAL_ACCESS,
                Permissions.MCP_ACCESS,
                Permissions.REACT_AGENT_ACCESS,
            },
            "user": {
                Permissions.TERMINAL_ACCESS,
                Permissions.MCP_ACCESS,
                Permissions.REACT_AGENT_ACCESS,
                Permissions.READ_SESSIONS,
            },
            "agent": {
                Permissions.TERMINAL_ACCESS,
                Permissions.MCP_ACCESS,
                Permissions.READ_SESSIONS,
                Permissions.WRITE_SESSIONS,
            },
            "readonly": {
                Permissions.READ_SESSIONS,
            },
        }

    def create_api_key(
        self,
        user: User,
        name: str,
        scopes: set[str] | None = None,
        expires_days: int | None = None,
        key_type: str = "user",
        metadata: dict | None = None,
    ) -> tuple[APIKey, str]:
        """Create a new API key for a user."""

        # Validate inputs
        if not name or len(name.strip()) < 1:
            raise ValueError("API key name is required")

        if not user or not user.is_active:
            raise ValueError("User must be active to create API keys")

        # Set default scopes based on key type
        if scopes is None:
            scopes = self.default_scopes.get(key_type, self.default_scopes["user"]).copy()

        # Ensure user has permissions for requested scopes
        if not user.has_permission(Permissions.API_KEY_MANAGEMENT):
            # Non-admin users can only create keys with their own permissions
            allowed_scopes = user.permissions.intersection(scopes)
            if allowed_scopes != scopes:
                logger.warning(
                    f"User {user.username} attempted to create key with unauthorized scopes"
                )
                scopes = allowed_scopes

        # Generate the API key
        api_key, raw_key = APIKey.generate_key(
            user_id=user.id, name=name.strip(), scopes=scopes, expires_days=expires_days
        )

        # Add metadata
        if metadata:
            api_key.metadata.update(metadata)

        api_key.metadata.update(
            {
                "key_type": key_type,
                "created_by": user.username,
                "user_role": user.role.value,
            }
        )

        # Save to storage
        if not self.storage.create_api_key(api_key):
            raise RuntimeError("Failed to save API key")

        logger.info(f"Created API key '{name}' for user {user.username} with scopes: {scopes}")
        return api_key, raw_key

    def get_api_key(self, key_id: str) -> APIKey | None:
        """Get API key by ID."""
        return self.storage.get_api_key_by_id(key_id)

    def list_user_keys(self, user: User) -> list[APIKey]:
        """List all API keys for a user."""
        return self.storage.list_api_keys(user_id=user.id)

    def list_all_keys(self, user: User) -> list[APIKey]:
        """List all API keys (admin only)."""
        if not user.has_permission(Permissions.API_KEY_MANAGEMENT):
            raise PermissionError("Insufficient permissions to list all API keys")

        return self.storage.list_api_keys()

    def verify_api_key(self, raw_key: str) -> tuple[APIKey, User] | None:
        """Verify an API key and return the key and associated user."""
        api_key = self.storage.verify_api_key(raw_key)
        if not api_key:
            return None

        user = self.storage.get_user_by_id(api_key.user_id)
        if not user or not user.is_active:
            logger.warning(f"API key {api_key.key_prefix} belongs to inactive user")
            return None

        return api_key, user

    def update_api_key(
        self,
        user: User,
        key_id: str,
        name: str | None = None,
        scopes: set[str] | None = None,
        is_active: bool | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """Update an existing API key."""
        api_key = self.storage.get_api_key_by_id(key_id)
        if not api_key:
            return False

        # Check permissions
        if not user.has_permission(Permissions.API_KEY_MANAGEMENT) and api_key.user_id != user.id:
            raise PermissionError("Cannot modify other users' API keys")

        # Update fields
        if name is not None:
            api_key.name = name.strip()

        if scopes is not None:
            # Validate scopes
            if not user.has_permission(Permissions.API_KEY_MANAGEMENT):
                allowed_scopes = user.permissions.intersection(scopes)
                if allowed_scopes != scopes:
                    raise PermissionError("Cannot assign unauthorized scopes")
            api_key.scopes = scopes

        if is_active is not None:
            api_key.is_active = is_active

        if metadata is not None:
            api_key.metadata.update(metadata)

        # Save changes
        return self.storage.update_api_key(api_key)

    def revoke_api_key(self, user: User, key_id: str, reason: str = "") -> bool:
        """Revoke an API key."""
        api_key = self.storage.get_api_key_by_id(key_id)
        if not api_key:
            return False

        # Check permissions
        if not user.has_permission(Permissions.API_KEY_MANAGEMENT) and api_key.user_id != user.id:
            raise PermissionError("Cannot revoke other users' API keys")

        # Revoke the key
        api_key.revoke()
        api_key.metadata.update(
            {
                "revoked_by": user.username,
                "revoke_reason": reason,
                "revoked_at": datetime.now(UTC).isoformat(),
            }
        )

        success = self.storage.update_api_key(api_key)
        if success:
            logger.info(f"Revoked API key '{api_key.name}' (ID: {key_id}) - Reason: {reason}")

        return success

    def delete_api_key(self, user: User, key_id: str) -> bool:
        """Permanently delete an API key."""
        api_key = self.storage.get_api_key_by_id(key_id)
        if not api_key:
            return False

        # Check permissions
        if not user.has_permission(Permissions.API_KEY_MANAGEMENT) and api_key.user_id != user.id:
            raise PermissionError("Cannot delete other users' API keys")

        success = self.storage.delete_api_key(key_id)
        if success:
            logger.info(f"Deleted API key '{api_key.name}' (ID: {key_id})")

        return success

    def rotate_api_key(self, user: User, key_id: str) -> tuple[APIKey, str] | None:
        """Rotate an API key by creating a new one with the same settings."""
        old_key = self.storage.get_api_key_by_id(key_id)
        if not old_key:
            return None

        # Check permissions
        if not user.has_permission(Permissions.API_KEY_MANAGEMENT) and old_key.user_id != user.id:
            raise PermissionError("Cannot rotate other users' API keys")

        # Create new key with same settings
        try:
            new_name = f"{old_key.name} (rotated)"
            expires_days = None
            if old_key.expires_at:
                expires_days = (old_key.expires_at - datetime.now(UTC)).days
                expires_days = max(1, expires_days)  # At least 1 day

            new_key, raw_key = self.create_api_key(
                user=user,
                name=new_name,
                scopes=old_key.scopes.copy(),
                expires_days=expires_days,
                metadata={
                    **old_key.metadata,
                    "rotated_from": old_key.id,
                    "rotation_date": datetime.now(UTC).isoformat(),
                },
            )

            # Revoke old key
            self.revoke_api_key(user, key_id, "Key rotated")

            logger.info(f"Rotated API key '{old_key.name}' (ID: {key_id})")
            return new_key, raw_key

        except Exception:
            logger.exception(f"Failed to rotate API key {key_id}")
            return None

    def cleanup_expired_keys(self) -> int:
        """Clean up expired API keys and return count of cleaned keys."""
        all_keys = self.storage.list_api_keys()
        expired_keys = [key for key in all_keys if key.is_expired()]

        cleaned_count = 0
        for key in expired_keys:
            if self.storage.delete_api_key(key.id):
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired API keys")

        return cleaned_count

    def get_key_usage_stats(self, user: User, days: int = 30) -> dict:
        """Get API key usage statistics."""
        if not user.has_permission(Permissions.API_KEY_MANAGEMENT):
            # Non-admin users can only see their own stats
            keys = self.list_user_keys(user)
        else:
            keys = self.list_all_keys(user)

        now = datetime.now(UTC)
        cutoff_date = now - timedelta(days=days)

        stats = {
            "total_keys": len(keys),
            "active_keys": len([k for k in keys if k.is_active]),
            "expired_keys": len([k for k in keys if k.is_expired()]),
            "revoked_keys": len([k for k in keys if k.is_revoked()]),
            "recently_used": len([k for k in keys if k.last_used and k.last_used > cutoff_date]),
            "usage_by_scope": {},
            "usage_by_user": {},
        }

        # Usage by scope
        for key in keys:
            for scope in key.scopes:
                stats["usage_by_scope"][scope] = stats["usage_by_scope"].get(scope, 0) + 1

        # Usage by user (admin only)
        if user.has_permission(Permissions.API_KEY_MANAGEMENT):
            for key in keys:
                user_id = key.user_id
                stats["usage_by_user"][user_id] = stats["usage_by_user"].get(user_id, 0) + 1

        return stats

    def check_key_permissions(self, api_key: APIKey, required_permission: str) -> bool:
        """Check if an API key has the required permission."""
        if not api_key.is_active or api_key.is_expired() or api_key.is_revoked():
            return False

        # Admin scope grants all permissions
        if "admin" in api_key.scopes:
            return True

        return required_permission in api_key.scopes

    def generate_key_report(self, user: User) -> dict:
        """Generate a comprehensive API key report."""
        if not user.has_permission(Permissions.API_KEY_MANAGEMENT):
            raise PermissionError("Insufficient permissions for key reports")

        keys = self.list_all_keys(user)
        now = datetime.now(UTC)

        # Security analysis
        security_issues = []

        # Check for keys without expiration
        no_expiry_keys = [k for k in keys if k.expires_at is None and k.is_active]
        if no_expiry_keys:
            security_issues.append(
                {
                    "type": "no_expiration",
                    "count": len(no_expiry_keys),
                    "severity": "medium",
                    "message": f"{len(no_expiry_keys)} active keys have no expiration date",
                }
            )

        # Check for unused keys (not used in 30+ days)
        unused_cutoff = now - timedelta(days=30)
        unused_keys = [
            k for k in keys if k.is_active and (not k.last_used or k.last_used < unused_cutoff)
        ]
        if unused_keys:
            security_issues.append(
                {
                    "type": "unused_keys",
                    "count": len(unused_keys),
                    "severity": "low",
                    "message": f"{len(unused_keys)} active keys haven't been used in 30+ days",
                }
            )

        # Check for overprivileged keys
        admin_keys = [k for k in keys if "admin" in k.scopes and k.is_active]
        if admin_keys:
            security_issues.append(
                {
                    "type": "admin_keys",
                    "count": len(admin_keys),
                    "severity": "high",
                    "message": f"{len(admin_keys)} active keys have admin privileges",
                }
            )

        return {
            "summary": self.get_key_usage_stats(user),
            "security_analysis": {
                "issues": security_issues,
                "risk_level": "high"
                if any(issue["severity"] == "high" for issue in security_issues)
                else "medium"
                if security_issues
                else "low",
            },
            "recommendations": self._generate_security_recommendations(keys),
            "generated_at": now.isoformat(),
            "generated_by": user.username,
        }

    def _generate_security_recommendations(self, keys: list[APIKey]) -> list[str]:
        """Generate security recommendations based on key analysis."""
        recommendations = []

        # Check for keys without expiration
        no_expiry_count = len([k for k in keys if k.expires_at is None and k.is_active])
        if no_expiry_count > 0:
            recommendations.append(
                f"Set expiration dates for {no_expiry_count} active keys without expiration"
            )

        # Check for unused keys
        now = datetime.now(UTC)
        unused_cutoff = now - timedelta(days=30)
        unused_count = len(
            [k for k in keys if k.is_active and (not k.last_used or k.last_used < unused_cutoff)]
        )
        if unused_count > 0:
            recommendations.append(f"Review and consider revoking {unused_count} unused keys")

        # Check for admin keys
        admin_count = len([k for k in keys if "admin" in k.scopes and k.is_active])
        if admin_count > 1:
            recommendations.append(f"Review necessity of {admin_count} admin-level keys")

        # Check for expired but not cleaned keys
        expired_count = len([k for k in keys if k.is_expired()])
        if expired_count > 0:
            recommendations.append(f"Clean up {expired_count} expired keys")

        if not recommendations:
            recommendations.append("API key security posture is good")

        return recommendations


# Global API key manager instance
api_key_manager = APIKeyManager()
