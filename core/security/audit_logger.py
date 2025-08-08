"""Comprehensive audit logging system for security events.

Provides structured logging of security events, authentication attempts,
authorization decisions, and other security-relevant activities.
"""

from datetime import UTC
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import threading
from typing import Any

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events."""

    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_SUCCESS = "authorization_success"
    AUTHORIZATION_FAILURE = "authorization_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_VIOLATION = "security_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    FILE_ACCESS_ATTEMPT = "file_access_attempt"
    ADMIN_ACTION = "admin_action"
    PASSWORD_CHANGE = "password_change"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    SESSION_CREATED = "session_created"
    SESSION_TERMINATED = "session_terminated"


class SecurityLevel(Enum):
    """Security event severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLogger:
    """Comprehensive audit logging system."""

    def __init__(self, log_file_path: str | None = None) -> None:
        """Initialize audit logger.

        Args:
            log_file_path: Path to audit log file. If None, uses default location.

        """
        self.log_file_path = log_file_path or self._get_default_log_path()
        self.lock = threading.Lock()

        # Setup structured logging
        self.audit_logger = logging.getLogger("security_audit")
        self.audit_logger.setLevel(logging.INFO)

        # Create file handler if it doesn't exist
        if not self.audit_logger.handlers:
            self._setup_file_handler()

    def _get_default_log_path(self) -> str:
        """Get default audit log file path."""
        # Try project logs directory first
        project_root = Path(__file__).parent.parent.parent
        logs_dir = project_root / "logs"
        if logs_dir.exists() or logs_dir.parent.exists():
            logs_dir.mkdir(exist_ok=True)
            return str(logs_dir / "security_audit.jsonl")

        # Fall back to user config directory
        config_dir = Path.home() / ".config" / "fullstack-agent" / "logs"
        config_dir.mkdir(parents=True, exist_ok=True)
        return str(config_dir / "security_audit.jsonl")

    def _setup_file_handler(self) -> None:
        """Setup file handler for audit logging."""
        log_path = Path(self.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file handler with JSON formatter
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # JSON formatter
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)

        self.audit_logger.addHandler(file_handler)

        # Set secure permissions on log file
        try:
            log_path.chmod(0o640)  # Read/write for owner, read for group
        except OSError:
            logger.warning(f"Could not set secure permissions on audit log: {log_path}")

    def log_security_event(
        self,
        event_type: SecurityEventType,
        severity: SecurityLevel,
        message: str,
        user_id: str | None = None,
        client_ip: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
        resource: str | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a security event.

        Args:
            event_type: Type of security event
            severity: Severity level
            message: Human-readable message
            user_id: ID of user involved (if applicable)
            client_ip: Client IP address
            user_agent: Client user agent string
            request_id: Request ID for correlation
            resource: Resource being accessed
            action: Action being performed
            metadata: Additional event metadata

        """
        with self.lock:
            event = {
                "timestamp": datetime.now(UTC).isoformat(),
                "event_type": event_type.value,
                "severity": severity.value,
                "message": message,
                "user_id": self._sanitize_user_data(user_id),
                "client_ip": self._anonymize_ip(client_ip),
                "user_agent": self._sanitize_user_agent(user_agent),
                "request_id": request_id,
                "resource": resource,
                "action": action,
                "metadata": self._sanitize_metadata(metadata or {}),
            }

            # Remove None values
            event = {k: v for k, v in event.items() if v is not None}

            # Log as JSON
            self.audit_logger.info(json.dumps(event))

            # Also log to standard logger for critical events
            if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                getattr(logger, severity.value.lower())(
                    f"SECURITY EVENT: {event_type.value} - {message}"
                )

    def log_authentication_attempt(
        self,
        success: bool,
        user_id: str | None = None,
        username: str | None = None,
        client_ip: str | None = None,
        user_agent: str | None = None,
        failure_reason: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Log authentication attempt.

        Args:
            success: Whether authentication succeeded
            user_id: User ID (if successful)
            username: Username attempted
            client_ip: Client IP address
            user_agent: Client user agent
            failure_reason: Reason for failure (if unsuccessful)
            request_id: Request ID for correlation

        """
        event_type = (
            SecurityEventType.AUTHENTICATION_SUCCESS
            if success
            else SecurityEventType.AUTHENTICATION_FAILURE
        )
        severity = SecurityLevel.LOW if success else SecurityLevel.MEDIUM

        message = f"Authentication {'succeeded' if success else 'failed'}"
        if username:
            message += f" for user: {self._sanitize_username(username)}"
        if failure_reason:
            message += f" - {failure_reason}"

        metadata = {}
        if username:
            metadata["username"] = self._sanitize_username(username)
        if failure_reason:
            metadata["failure_reason"] = failure_reason

        self.log_security_event(
            event_type=event_type,
            severity=severity,
            message=message,
            user_id=user_id,
            client_ip=client_ip,
            user_agent=user_agent,
            request_id=request_id,
            action="authenticate",
            metadata=metadata,
        )

    def log_authorization_attempt(
        self,
        success: bool,
        user_id: str,
        resource: str,
        action: str,
        client_ip: str | None = None,
        required_permission: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Log authorization attempt.

        Args:
            success: Whether authorization succeeded
            user_id: User ID
            resource: Resource being accessed
            action: Action being performed
            client_ip: Client IP address
            required_permission: Required permission (if unsuccessful)
            request_id: Request ID for correlation

        """
        event_type = (
            SecurityEventType.AUTHORIZATION_SUCCESS
            if success
            else SecurityEventType.AUTHORIZATION_FAILURE
        )
        severity = SecurityLevel.LOW if success else SecurityLevel.HIGH

        message = f"Authorization {'granted' if success else 'denied'} for {action} on {resource}"

        metadata = {"resource": resource, "action": action}
        if required_permission:
            metadata["required_permission"] = required_permission

        self.log_security_event(
            event_type=event_type,
            severity=severity,
            message=message,
            user_id=user_id,
            client_ip=client_ip,
            resource=resource,
            action=action,
            request_id=request_id,
            metadata=metadata,
        )

    def log_suspicious_activity(
        self,
        activity_type: str,
        message: str,
        user_id: str | None = None,
        client_ip: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log suspicious activity.

        Args:
            activity_type: Type of suspicious activity
            message: Description of the activity
            user_id: User ID (if known)
            client_ip: Client IP address
            user_agent: Client user agent
            request_id: Request ID for correlation
            metadata: Additional metadata

        """
        self.log_security_event(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            severity=SecurityLevel.HIGH,
            message=f"Suspicious activity detected: {activity_type} - {message}",
            user_id=user_id,
            client_ip=client_ip,
            user_agent=user_agent,
            request_id=request_id,
            metadata={**(metadata or {}), "activity_type": activity_type},
        )

    def log_admin_action(
        self,
        admin_user_id: str,
        action: str,
        target_resource: str | None = None,
        target_user_id: str | None = None,
        client_ip: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log administrative action.

        Args:
            admin_user_id: ID of admin user performing action
            action: Administrative action performed
            target_resource: Resource being modified (if applicable)
            target_user_id: Target user ID (if applicable)
            client_ip: Client IP address
            request_id: Request ID for correlation
            metadata: Additional metadata

        """
        message = f"Admin action: {action}"
        if target_resource:
            message += f" on {target_resource}"
        if target_user_id:
            message += f" for user {target_user_id}"

        self.log_security_event(
            event_type=SecurityEventType.ADMIN_ACTION,
            severity=SecurityLevel.MEDIUM,
            message=message,
            user_id=admin_user_id,
            client_ip=client_ip,
            resource=target_resource,
            action=action,
            request_id=request_id,
            metadata={
                **(metadata or {}),
                "target_user_id": target_user_id,
                "is_admin_action": True,
            },
        )

    def _sanitize_user_data(self, user_id: str | None) -> str | None:
        """Sanitize user data for logging."""
        if not user_id:
            return None
        # Hash or truncate sensitive user data if needed
        return str(user_id)[:50]  # Limit length

    def _sanitize_username(self, username: str) -> str:
        """Sanitize username for logging."""
        # Remove potentially sensitive information
        return username[:50]  # Limit length

    def _anonymize_ip(self, ip: str | None) -> str | None:
        """Anonymize IP address for privacy compliance."""
        if not ip:
            return None

        # For IPv4, zero out the last octet
        if "." in ip and ip.count(".") == 3:
            parts = ip.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0"

        # For IPv6, zero out the last 64 bits
        if ":" in ip:
            parts = ip.split(":")
            if len(parts) >= 4:
                return ":".join(parts[:4]) + "::0"

        return ip  # Return as-is if can't parse

    def _sanitize_user_agent(self, user_agent: str | None) -> str | None:
        """Sanitize user agent string."""
        if not user_agent:
            return None
        # Limit length and remove potentially sensitive info
        return user_agent[:200]

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize metadata for logging."""
        sanitized = {}
        sensitive_keys = ["password", "token", "key", "secret", "credential"]

        for key, value in metadata.items():
            # Redact sensitive keys
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 1000:
                # Truncate long strings
                sanitized[key] = value[:500] + "... [TRUNCATED]"
            else:
                sanitized[key] = value

        return sanitized

    def get_recent_events(
        self,
        hours: int = 24,
        event_types: list[SecurityEventType] | None = None,
        severity_levels: list[SecurityLevel] | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent security events.

        Args:
            hours: Number of hours to look back
            event_types: Filter by event types
            severity_levels: Filter by severity levels

        Returns:
            List of recent security events

        """
        events = []
        cutoff_time = datetime.now(UTC).timestamp() - (hours * 3600)

        try:
            with open(self.log_file_path) as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        event_time = datetime.fromisoformat(event["timestamp"]).timestamp()

                        if event_time < cutoff_time:
                            continue

                        # Filter by event types
                        if event_types:
                            if event["event_type"] not in [et.value for et in event_types]:
                                continue

                        # Filter by severity
                        if severity_levels:
                            if event["severity"] not in [sl.value for sl in severity_levels]:
                                continue

                        events.append(event)

                    except (json.JSONDecodeError, KeyError):
                        continue

        except FileNotFoundError:
            logger.warning(f"Audit log file not found: {self.log_file_path}")

        return sorted(events, key=lambda x: x["timestamp"], reverse=True)


# Global audit logger instance
audit_logger = AuditLogger()
