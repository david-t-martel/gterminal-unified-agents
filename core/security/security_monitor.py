"""Security monitoring and alerting system.

Provides real-time security monitoring, threat detection, and alerting
for security events and suspicious activities.
"""

import asyncio
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
import logging
import smtplib
import threading
from typing import Any

from .audit_logger import AuditLogger
from .audit_logger import SecurityEventType
from .audit_logger import SecurityLevel

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""

    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class SecurityAlert:
    """Security alert data structure."""

    id: str
    threat_level: ThreatLevel
    title: str
    description: str
    timestamp: datetime
    source_events: list[str]
    affected_resources: list[str]
    recommended_actions: list[str]
    metadata: dict[str, Any]


@dataclass
class ThreatPattern:
    """Threat detection pattern."""

    name: str
    description: str
    event_types: list[SecurityEventType]
    time_window_minutes: int
    threshold_count: int
    threat_level: ThreatLevel
    conditions: dict[str, Any]


class SecurityMonitor:
    """Real-time security monitoring and threat detection."""

    def __init__(self, audit_logger: AuditLogger) -> None:
        """Initialize security monitor.

        Args:
            audit_logger: Audit logger instance for event tracking

        """
        self.audit_logger = audit_logger
        self.lock = threading.Lock()

        # Event tracking
        self.recent_events: deque = deque(maxsize=10000)
        self.ip_activity: defaultdict = defaultdict(list)
        self.user_activity: defaultdict = defaultdict(list)
        self.failed_attempts: defaultdict = defaultdict(int)
        self.blocked_ips: set[str] = set()

        # Alert configuration
        self.alert_channels: list[AlertChannel] = [AlertChannel.LOG, AlertChannel.CONSOLE]
        self.email_config: dict[str, str] = {}
        self.webhook_urls: list[str] = []

        # Threat detection patterns
        self.threat_patterns = self._initialize_threat_patterns()

        # Background monitoring
        self.monitoring_active = False
        self.monitoring_task: asyncio.Task | None = None

    def _initialize_threat_patterns(self) -> list[ThreatPattern]:
        """Initialize threat detection patterns."""
        return [
            # Brute force attack detection
            ThreatPattern(
                name="brute_force_attack",
                description="Multiple failed authentication attempts from same IP",
                event_types=[SecurityEventType.AUTHENTICATION_FAILURE],
                time_window_minutes=5,
                threshold_count=5,
                threat_level=ThreatLevel.HIGH,
                conditions={"same_ip": True},
            ),
            # Account enumeration
            ThreatPattern(
                name="account_enumeration",
                description="Multiple authentication attempts for different users from same IP",
                event_types=[SecurityEventType.AUTHENTICATION_FAILURE],
                time_window_minutes=10,
                threshold_count=10,
                threat_level=ThreatLevel.MEDIUM,
                conditions={"same_ip": True, "different_users": True},
            ),
            # Privilege escalation attempts
            ThreatPattern(
                name="privilege_escalation",
                description="Multiple authorization failures for admin resources",
                event_types=[SecurityEventType.AUTHORIZATION_FAILURE],
                time_window_minutes=15,
                threshold_count=3,
                threat_level=ThreatLevel.CRITICAL,
                conditions={"admin_resources": True},
            ),
            # Suspicious file access
            ThreatPattern(
                name="suspicious_file_access",
                description="Attempts to access sensitive files or directories",
                event_types=[SecurityEventType.SECURITY_VIOLATION],
                time_window_minutes=5,
                threshold_count=3,
                threat_level=ThreatLevel.HIGH,
                conditions={"file_access": True},
            ),
            # Rate limiting violations
            ThreatPattern(
                name="rate_limit_abuse",
                description="Repeated rate limit violations",
                event_types=[SecurityEventType.RATE_LIMIT_EXCEEDED],
                time_window_minutes=5,
                threshold_count=10,
                threat_level=ThreatLevel.MEDIUM,
                conditions={"same_ip": True},
            ),
            # Input validation attacks
            ThreatPattern(
                name="injection_attempts",
                description="Multiple input validation failures indicating injection attempts",
                event_types=[SecurityEventType.INPUT_VALIDATION_FAILURE],
                time_window_minutes=10,
                threshold_count=5,
                threat_level=ThreatLevel.HIGH,
                conditions={"same_ip": True},
            ),
        ]

    def configure_email_alerts(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: list[str],
    ) -> None:
        """Configure email alerting.

        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: From email address
            to_emails: List of recipient email addresses

        """
        self.email_config = {
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "username": username,
            "password": password,
            "from_email": from_email,
            "to_emails": to_emails,
        }

        if AlertChannel.EMAIL not in self.alert_channels:
            self.alert_channels.append(AlertChannel.EMAIL)

    def configure_webhook_alerts(self, webhook_urls: list[str]) -> None:
        """Configure webhook alerting.

        Args:
            webhook_urls: List of webhook URLs for alerts

        """
        self.webhook_urls = webhook_urls

        if AlertChannel.WEBHOOK not in self.alert_channels:
            self.alert_channels.append(AlertChannel.WEBHOOK)

    def track_security_event(
        self,
        event_type: SecurityEventType,
        severity: SecurityLevel,
        user_id: str | None = None,
        client_ip: str | None = None,
        resource: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track a security event for monitoring.

        Args:
            event_type: Type of security event
            severity: Event severity
            user_id: User ID (if applicable)
            client_ip: Client IP address
            resource: Resource being accessed
            metadata: Additional event metadata

        """
        event = {
            "timestamp": datetime.now(UTC),
            "event_type": event_type,
            "severity": severity,
            "user_id": user_id,
            "client_ip": client_ip,
            "resource": resource,
            "metadata": metadata or {},
        }

        with self.lock:
            self.recent_events.append(event)

            # Track IP activity
            if client_ip:
                self.ip_activity[client_ip].append(event)
                # Keep only recent activity (last hour)
                cutoff = datetime.now(UTC) - timedelta(hours=1)
                self.ip_activity[client_ip] = [
                    e for e in self.ip_activity[client_ip] if e["timestamp"] > cutoff
                ]

            # Track user activity
            if user_id:
                self.user_activity[user_id].append(event)
                # Keep only recent activity (last hour)
                cutoff = datetime.now(UTC) - timedelta(hours=1)
                self.user_activity[user_id] = [
                    e for e in self.user_activity[user_id] if e["timestamp"] > cutoff
                ]

            # Track failed attempts
            if event_type == SecurityEventType.AUTHENTICATION_FAILURE and client_ip:
                self.failed_attempts[client_ip] += 1

        # Check for threats
        asyncio.create_task(self._check_threat_patterns(event))

    async def _check_threat_patterns(self, event: dict[str, Any]) -> None:
        """Check if event matches any threat patterns."""
        for pattern in self.threat_patterns:
            if await self._matches_pattern(event, pattern):
                alert = self._create_alert_from_pattern(event, pattern)
                await self._send_alert(alert)

    async def _matches_pattern(self, event: dict[str, Any], pattern: ThreatPattern) -> bool:
        """Check if event matches a threat pattern."""
        # Check event type
        if event["event_type"] not in pattern.event_types:
            return False

        # Get recent events in time window
        cutoff = datetime.now(UTC) - timedelta(minutes=pattern.time_window_minutes)
        recent_events = [
            e
            for e in self.recent_events
            if e["timestamp"] > cutoff and e["event_type"] in pattern.event_types
        ]

        # Apply pattern conditions
        filtered_events = self._apply_pattern_conditions(recent_events, pattern.conditions, event)

        # Check threshold
        return len(filtered_events) >= pattern.threshold_count

    def _apply_pattern_conditions(
        self,
        events: list[dict[str, Any]],
        conditions: dict[str, Any],
        current_event: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply pattern conditions to filter events."""
        filtered = events

        # Same IP condition
        if conditions.get("same_ip") and current_event.get("client_ip"):
            filtered = [e for e in filtered if e.get("client_ip") == current_event["client_ip"]]

        # Different users condition
        if conditions.get("different_users"):
            user_ids = {e.get("user_id") for e in filtered if e.get("user_id")}
            if len(user_ids) < 3:  # Need at least 3 different users
                return []

        # Admin resources condition
        if conditions.get("admin_resources"):
            filtered = [
                e
                for e in filtered
                if e.get("resource") and ("admin" in e["resource"] or "user" in e["resource"])
            ]

        # File access condition
        if conditions.get("file_access"):
            filtered = [
                e
                for e in filtered
                if e.get("metadata", {}).get("file_path")
                or e.get("resource", "").startswith("file:")
            ]

        return filtered

    def _create_alert_from_pattern(
        self, event: dict[str, Any], pattern: ThreatPattern
    ) -> SecurityAlert:
        """Create alert from threat pattern match."""
        alert_id = f"{pattern.name}_{int(datetime.now().timestamp())}"

        # Collect affected resources
        affected_resources: list[Any] = []
        if event.get("resource"):
            affected_resources.append(event["resource"])

        # Generate recommended actions
        recommended_actions = self._get_recommended_actions(pattern)

        return SecurityAlert(
            id=alert_id,
            threat_level=pattern.threat_level,
            title=f"Security Threat Detected: {pattern.name.replace('_', ' ').title()}",
            description=pattern.description,
            timestamp=datetime.now(UTC),
            source_events=[event.get("event_id", "unknown")],
            affected_resources=affected_resources,
            recommended_actions=recommended_actions,
            metadata={
                "pattern_name": pattern.name,
                "client_ip": event.get("client_ip"),
                "user_id": event.get("user_id"),
                "detection_confidence": "high",
            },
        )

    def _get_recommended_actions(self, pattern: ThreatPattern) -> list[str]:
        """Get recommended actions for a threat pattern."""
        actions = {
            "brute_force_attack": [
                "Block the attacking IP address",
                "Review authentication logs for the affected accounts",
                "Consider implementing account lockout policies",
                "Notify affected users to change passwords",
            ],
            "account_enumeration": [
                "Block the attacking IP address",
                "Review user account security",
                "Consider implementing CAPTCHA for authentication",
                "Monitor for further enumeration attempts",
            ],
            "privilege_escalation": [
                "Immediately review the user's permissions",
                "Audit recent administrative actions",
                "Consider temporarily suspending the user account",
                "Review access control policies",
            ],
            "suspicious_file_access": [
                "Review file access permissions",
                "Audit the user's recent file access history",
                "Consider restricting access to sensitive files",
                "Monitor for data exfiltration attempts",
            ],
            "rate_limit_abuse": [
                "Block the abusing IP address",
                "Review rate limiting policies",
                "Consider implementing progressive delays",
                "Monitor for distributed attacks",
            ],
            "injection_attempts": [
                "Block the attacking IP address",
                "Review input validation mechanisms",
                "Audit recent database/system activity",
                "Consider implementing WAF rules",
            ],
        }

        return actions.get(pattern.name, ["Review security logs", "Monitor for further activity"])

    async def _send_alert(self, alert: SecurityAlert) -> None:
        """Send alert through configured channels."""
        for channel in self.alert_channels:
            try:
                if channel == AlertChannel.LOG:
                    self._send_log_alert(alert)
                elif channel == AlertChannel.CONSOLE:
                    self._send_console_alert(alert)
                elif channel == AlertChannel.EMAIL and self.email_config:
                    await self._send_email_alert(alert)
                elif channel == AlertChannel.WEBHOOK and self.webhook_urls:
                    await self._send_webhook_alert(alert)
            except Exception as e:
                logger.exception(f"Failed to send alert via {channel.value}: {e}")

    def _send_log_alert(self, alert: SecurityAlert) -> None:
        """Send alert to log."""
        logger.critical(f"SECURITY ALERT: {alert.title} - {alert.description}")

    def _send_console_alert(self, alert: SecurityAlert) -> None:
        """Send alert to console."""
        for _i, _action in enumerate(alert.recommended_actions, 1):
            pass

    async def _send_email_alert(self, alert: SecurityAlert) -> None:
        """Send alert via email."""
        if not self.email_config:
            return

        try:
            # Create email message
            msg = MimeMultipart()
            msg["From"] = self.email_config["from_email"]
            msg["To"] = ", ".join(self.email_config["to_emails"])
            msg["Subject"] = f"🚨 Security Alert: {alert.title}"

            # Create HTML body
            body = f"""
            <html>
            <body>
                <h2 style="color: {"red" if alert.threat_level == ThreatLevel.CRITICAL else "orange"}">
                    Security Alert - {alert.threat_level.value.upper()}
                </h2>
                <p><strong>Title:</strong> {alert.title}</p>
                <p><strong>Description:</strong> {alert.description}</p>
                <p><strong>Time:</strong> {alert.timestamp}</p>
                <p><strong>Alert ID:</strong> {alert.id}</p>

                <h3>Affected Resources:</h3>
                <ul>
                    {"".join(f"<li>{resource}</li>" for resource in alert.affected_resources)}
                </ul>

                <h3>Recommended Actions:</h3>
                <ol>
                    {"".join(f"<li>{action}</li>" for action in alert.recommended_actions)}
                </ol>

                <h3>Metadata:</h3>
                <ul>
                    {"".join(f"<li><strong>{k}:</strong> {v}</li>" for k, v in alert.metadata.items())}
                </ul>
            </body>
            </html>
            """

            msg.attach(MimeText(body, "html"))

            # Send email
            with smtplib.SMTP(
                self.email_config["smtp_server"], self.email_config["smtp_port"]
            ) as server:
                server.starttls()
                server.login(self.email_config["username"], self.email_config["password"])
                server.send_message(msg)

        except Exception as e:
            logger.exception(f"Failed to send email alert: {e}")

    async def _send_webhook_alert(self, alert: SecurityAlert) -> None:
        """Send alert via webhook."""
        import aiohttp

        alert_data = {
            "id": alert.id,
            "threat_level": alert.threat_level.value,
            "title": alert.title,
            "description": alert.description,
            "timestamp": alert.timestamp.isoformat(),
            "affected_resources": alert.affected_resources,
            "recommended_actions": alert.recommended_actions,
            "metadata": alert.metadata,
        }

        async with aiohttp.ClientSession() as session:
            for webhook_url in self.webhook_urls:
                try:
                    async with session.post(
                        webhook_url,
                        json=alert_data,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        if response.status >= 400:
                            logger.error(f"Webhook alert failed: {response.status}")
                except Exception as e:
                    logger.exception(f"Failed to send webhook alert to {webhook_url}: {e}")

    def block_ip(self, ip_address: str, duration_minutes: int = 60) -> None:
        """Block an IP address temporarily.

        Args:
            ip_address: IP address to block
            duration_minutes: Duration to block for

        """
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP address: {ip_address} for {duration_minutes} minutes")

        # Schedule unblock
        async def unblock_later() -> None:
            await asyncio.sleep(duration_minutes * 60)
            self.blocked_ips.discard(ip_address)
            logger.info(f"Unblocked IP address: {ip_address}")

        asyncio.create_task(unblock_later())

    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if an IP address is blocked.

        Args:
            ip_address: IP address to check

        Returns:
            True if IP is blocked

        """
        return ip_address in self.blocked_ips

    def get_security_metrics(self) -> dict[str, Any]:
        """Get current security metrics.

        Returns:
            Dictionary of security metrics

        """
        with self.lock:
            # Calculate metrics from recent events
            hour_ago = datetime.now(UTC) - timedelta(hours=1)
            recent_events = [e for e in self.recent_events if e["timestamp"] > hour_ago]

            event_counts = defaultdict(int)
            severity_counts = defaultdict(int)

            for event in recent_events:
                event_counts[event["event_type"].value] += 1
                severity_counts[event["severity"].value] += 1

            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "total_events_last_hour": len(recent_events),
                "events_by_type": dict(event_counts),
                "events_by_severity": dict(severity_counts),
                "blocked_ips": len(self.blocked_ips),
                "unique_active_ips": len(self.ip_activity),
                "unique_active_users": len(self.user_activity),
                "failed_attempts_by_ip": dict(self.failed_attempts),
                "monitoring_active": self.monitoring_active,
            }


# Global security monitor instance will be initialized after audit_logger
security_monitor = None


def initialize_security_monitor() -> None:
    """Initialize the global security monitor with audit logger."""
    global security_monitor
    from .audit_logger import audit_logger

    security_monitor = SecurityMonitor(audit_logger)
