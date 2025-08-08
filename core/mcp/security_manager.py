"""
Security Management System for MCP Servers

Provides comprehensive security policies, validation, and enforcement
for MCP server operations including authentication, authorization,
and secure execution environments.
"""

from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import time
from typing import Any

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for MCP servers"""

    MINIMAL = "minimal"
    STANDARD = "standard"
    SECURE = "secure"
    GATEWAY = "gateway"
    MONITORING = "monitoring"


class ThreatLevel(Enum):
    """Threat assessment levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPolicy:
    """Security policy configuration"""

    level: SecurityLevel
    allowed_commands: set[str]
    blocked_commands: set[str]
    allowed_paths: set[str]
    blocked_paths: set[str]
    require_wrapper: bool
    rate_limit_requests: int
    rate_limit_window: int
    max_execution_time: int
    require_auth: bool
    audit_all_operations: bool


@dataclass
class SecurityViolation:
    """Security violation record"""

    timestamp: float
    server_name: str
    violation_type: str
    severity: ThreatLevel
    details: str
    command: str | None = None
    path: str | None = None


class SecurityManager:
    """
    Comprehensive security management for MCP servers.

    Provides security policy enforcement, threat detection,
    audit logging, and secure execution environments.
    """

    # Default security policies by level
    DEFAULT_POLICIES = {
        SecurityLevel.MINIMAL: SecurityPolicy(
            level=SecurityLevel.MINIMAL,
            allowed_commands={"echo", "cat", "ls", "pwd"},
            blocked_commands={"rm", "sudo", "chmod", "chown", "dd"},
            allowed_paths={"/tmp", "/home/david/agents"},
            blocked_paths={"/etc", "/root", "/sys", "/proc"},
            require_wrapper=False,
            rate_limit_requests=1000,
            rate_limit_window=60,
            max_execution_time=30,
            require_auth=False,
            audit_all_operations=False,
        ),
        SecurityLevel.STANDARD: SecurityPolicy(
            level=SecurityLevel.STANDARD,
            allowed_commands={"echo", "cat", "ls", "pwd", "grep", "find", "python", "uv"},
            blocked_commands={"rm", "sudo", "chmod", "chown", "dd", "curl", "wget"},
            allowed_paths={"/tmp", "/home/david/agents", "/home/david/projects"},
            blocked_paths={"/etc", "/root", "/sys", "/proc", "/boot"},
            require_wrapper=True,
            rate_limit_requests=500,
            rate_limit_window=60,
            max_execution_time=60,
            require_auth=True,
            audit_all_operations=True,
        ),
        SecurityLevel.SECURE: SecurityPolicy(
            level=SecurityLevel.SECURE,
            allowed_commands={"uv", "python", "echo"},
            blocked_commands={"rm", "sudo", "chmod", "chown", "dd", "curl", "wget", "sh", "bash"},
            allowed_paths={"/home/david/agents/my-fullstack-agent"},
            blocked_paths={"/etc", "/root", "/sys", "/proc", "/boot", "/home/david/.ssh"},
            require_wrapper=True,
            rate_limit_requests=100,
            rate_limit_window=60,
            max_execution_time=30,
            require_auth=True,
            audit_all_operations=True,
        ),
        SecurityLevel.GATEWAY: SecurityPolicy(
            level=SecurityLevel.GATEWAY,
            allowed_commands={"uv", "python"},
            blocked_commands={
                "rm",
                "sudo",
                "chmod",
                "chown",
                "dd",
                "curl",
                "wget",
                "sh",
                "bash",
                "exec",
            },
            allowed_paths={"/home/david/agents/my-fullstack-agent"},
            blocked_paths={
                "/etc",
                "/root",
                "/sys",
                "/proc",
                "/boot",
                "/home/david/.ssh",
                "/home/david/.auth",
            },
            require_wrapper=True,
            rate_limit_requests=50,
            rate_limit_window=60,
            max_execution_time=15,
            require_auth=True,
            audit_all_operations=True,
        ),
        SecurityLevel.MONITORING: SecurityPolicy(
            level=SecurityLevel.MONITORING,
            allowed_commands={"uv", "python", "echo", "ps", "df", "free"},
            blocked_commands={"rm", "sudo", "chmod", "chown", "dd", "curl", "wget"},
            allowed_paths={"/home/david/agents", "/tmp", "/proc"},
            blocked_paths={"/etc", "/root", "/boot", "/home/david/.ssh", "/home/david/.auth"},
            require_wrapper=True,
            rate_limit_requests=200,
            rate_limit_window=60,
            max_execution_time=45,
            require_auth=True,
            audit_all_operations=True,
        ),
    }

    def __init__(self, audit_log_path: Path | None = None):
        """
        Initialize security manager.

        Args:
            audit_log_path: Path to audit log file
        """
        self.policies: dict[str, SecurityPolicy] = {}
        self.violations: list[SecurityViolation] = []
        self.rate_limit_tracking: dict[str, list[float]] = {}

        # Set up audit logging
        self.audit_log_path = audit_log_path or Path.home() / ".cache" / "mcp-security-audit.log"
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize default policies
        for level, policy in self.DEFAULT_POLICIES.items():
            self.policies[level.value] = policy

    def set_policy(self, server_name: str, security_level: SecurityLevel) -> None:
        """
        Set security policy for a server.

        Args:
            server_name: Name of the MCP server
            security_level: Security level to apply
        """
        if security_level.value not in self.policies:
            logger.warning(f"Unknown security level: {security_level.value}")
            security_level = SecurityLevel.STANDARD

        self.policies[server_name] = self.policies[security_level.value]
        logger.info(f"Set {security_level.value} security policy for {server_name}")

    def validate_command(
        self, server_name: str, command: str, args: list[str]
    ) -> tuple[bool, str | None]:
        """
        Validate a command against security policy.

        Args:
            server_name: Name of the MCP server
            command: Command to validate
            args: Command arguments

        Returns:
            Tuple of (is_allowed, reason_if_blocked)
        """
        policy = self.policies.get(server_name)
        if not policy:
            # No policy set, use standard
            policy = self.policies[SecurityLevel.STANDARD.value]

        # Check if command is explicitly blocked
        if command in policy.blocked_commands:
            violation = SecurityViolation(
                timestamp=time.time(),
                server_name=server_name,
                violation_type="blocked_command",
                severity=ThreatLevel.HIGH,
                details=f"Attempted to execute blocked command: {command}",
                command=command,
            )
            self._record_violation(violation)
            return False, f"Command '{command}' is blocked by security policy"

        # Check if command is in allowed list (if allowlist is defined)
        if policy.allowed_commands and command not in policy.allowed_commands:
            violation = SecurityViolation(
                timestamp=time.time(),
                server_name=server_name,
                violation_type="unauthorized_command",
                severity=ThreatLevel.MEDIUM,
                details=f"Attempted to execute unauthorized command: {command}",
                command=command,
            )
            self._record_violation(violation)
            return False, f"Command '{command}' is not in allowed command list"

        # Check for dangerous argument patterns
        dangerous_patterns = ["../", "sudo", "rm -rf", "&", "&&", "||", "|", ">", ">>"]
        full_command = " ".join([command, *args])

        for pattern in dangerous_patterns:
            if pattern in full_command:
                violation = SecurityViolation(
                    timestamp=time.time(),
                    server_name=server_name,
                    violation_type="dangerous_pattern",
                    severity=ThreatLevel.HIGH,
                    details=f"Command contains dangerous pattern '{pattern}': {full_command}",
                    command=command,
                )
                self._record_violation(violation)
                return False, f"Command contains dangerous pattern: {pattern}"

        return True, None

    def validate_path_access(
        self, server_name: str, path: str, operation: str = "read"
    ) -> tuple[bool, str | None]:
        """
        Validate path access against security policy.

        Args:
            server_name: Name of the MCP server
            path: File/directory path to validate
            operation: Type of operation (read, write, execute)

        Returns:
            Tuple of (is_allowed, reason_if_blocked)
        """
        policy = self.policies.get(server_name, self.policies[SecurityLevel.STANDARD.value])

        # Resolve absolute path
        try:
            abs_path = str(Path(path).resolve())
        except Exception:
            return False, f"Invalid path: {path}"

        # Check blocked paths
        for blocked_path in policy.blocked_paths:
            if abs_path.startswith(blocked_path):
                violation = SecurityViolation(
                    timestamp=time.time(),
                    server_name=server_name,
                    violation_type="blocked_path_access",
                    severity=ThreatLevel.HIGH,
                    details=f"Attempted {operation} access to blocked path: {abs_path}",
                    path=abs_path,
                )
                self._record_violation(violation)
                return False, f"Access to path '{path}' is blocked by security policy"

        # Check allowed paths (if allowlist is defined)
        if policy.allowed_paths:
            allowed = False
            for allowed_path in policy.allowed_paths:
                if abs_path.startswith(allowed_path):
                    allowed = True
                    break

            if not allowed:
                violation = SecurityViolation(
                    timestamp=time.time(),
                    server_name=server_name,
                    violation_type="unauthorized_path_access",
                    severity=ThreatLevel.MEDIUM,
                    details=f"Attempted {operation} access to unauthorized path: {abs_path}",
                    path=abs_path,
                )
                self._record_violation(violation)
                return False, f"Access to path '{path}' is not authorized"

        return True, None

    def check_rate_limit(self, server_name: str) -> tuple[bool, str | None]:
        """
        Check rate limiting for a server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Tuple of (is_allowed, reason_if_blocked)
        """
        policy = self.policies.get(server_name, self.policies[SecurityLevel.STANDARD.value])

        if not policy.rate_limit_requests:
            return True, None  # No rate limiting

        current_time = time.time()

        # Initialize tracking for server if needed
        if server_name not in self.rate_limit_tracking:
            self.rate_limit_tracking[server_name] = []

        # Clean old entries outside the time window
        window_start = current_time - policy.rate_limit_window
        self.rate_limit_tracking[server_name] = [
            timestamp
            for timestamp in self.rate_limit_tracking[server_name]
            if timestamp > window_start
        ]

        # Check if limit exceeded
        if len(self.rate_limit_tracking[server_name]) >= policy.rate_limit_requests:
            violation = SecurityViolation(
                timestamp=current_time,
                server_name=server_name,
                violation_type="rate_limit_exceeded",
                severity=ThreatLevel.MEDIUM,
                details=f"Rate limit exceeded: {len(self.rate_limit_tracking[server_name])} requests in {policy.rate_limit_window}s",
            )
            self._record_violation(violation)
            return (
                False,
                f"Rate limit exceeded: {policy.rate_limit_requests} requests per {policy.rate_limit_window}s",
            )

        # Record the request
        self.rate_limit_tracking[server_name].append(current_time)
        return True, None

    def create_secure_wrapper(
        self, server_name: str, base_command: list[str], wrapper_dir: Path
    ) -> Path:
        """
        Create a secure wrapper script for a server.

        Args:
            server_name: Name of the MCP server
            base_command: Base command to wrap
            wrapper_dir: Directory to create wrapper in

        Returns:
            Path to the created wrapper script
        """
        policy = self.policies.get(server_name, self.policies[SecurityLevel.STANDARD.value])
        wrapper_dir.mkdir(parents=True, exist_ok=True)

        wrapper_script = wrapper_dir / f"{server_name}_secure_wrapper.sh"

        # Generate wrapper script content
        script_content = f"""#!/bin/bash
# Secure wrapper for {server_name} MCP server
# Generated by SecurityManager

set -euo pipefail

# Set security environment
export PATH="/usr/local/bin:/usr/bin:/bin"
export RUST_EXTENSIONS_ENABLED=true
export LOG_LEVEL=INFO

# Authentication check
if [[ -z "${{GOOGLE_APPLICATION_CREDENTIALS:-}}" ]]; then
    echo "Error: GOOGLE_APPLICATION_CREDENTIALS not set" >&2
    exit 1
fi

# Verify credentials file exists
if [[ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
    echo "Error: Credentials file not found: $GOOGLE_APPLICATION_CREDENTIALS" >&2
    exit 1
fi

# Change to safe working directory
cd "/home/david/agents/my-fullstack-agent"

# Set resource limits
ulimit -t {policy.max_execution_time}  # CPU time
ulimit -v $((1024 * 1024 * 512))       # Virtual memory: 512MB
ulimit -n 64                           # File descriptors

# Audit log entry
echo "$(date -Iseconds) [SECURITY] $USER started {server_name} with command: {" ".join(base_command)}" >> ~/.cache/mcp-security-audit.log

# Execute with timeout
timeout {policy.max_execution_time} {" ".join(base_command)} "$@"
"""

        # Write wrapper script
        with open(wrapper_script, "w") as f:
            f.write(script_content)

        # Make executable
        wrapper_script.chmod(0o755)

        logger.info(f"Created secure wrapper for {server_name}: {wrapper_script}")
        return wrapper_script

    def _record_violation(self, violation: SecurityViolation) -> None:
        """Record a security violation"""
        self.violations.append(violation)

        # Write to audit log
        audit_entry = {
            "timestamp": violation.timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(violation.timestamp)),
            "server": violation.server_name,
            "type": violation.violation_type,
            "severity": violation.severity.value,
            "details": violation.details,
            "command": violation.command,
            "path": violation.path,
        }

        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

        logger.warning(
            f"Security violation recorded: {violation.violation_type} for {violation.server_name}"
        )

    def get_policy(self, server_name: str) -> SecurityPolicy:
        """
        Get security policy for a server.

        Args:
            server_name: Name of the MCP server

        Returns:
            SecurityPolicy object
        """
        return self.policies.get(server_name, self.policies[SecurityLevel.STANDARD.value])

    def get_violations(
        self, server_name: str | None = None, since: float | None = None
    ) -> list[SecurityViolation]:
        """
        Get security violations.

        Args:
            server_name: Filter by server name
            since: Filter by timestamp (Unix time)

        Returns:
            List of SecurityViolation objects
        """
        violations = self.violations

        if server_name:
            violations = [v for v in violations if v.server_name == server_name]

        if since:
            violations = [v for v in violations if v.timestamp >= since]

        return violations

    def get_security_report(self) -> dict[str, Any]:
        """
        Generate security status report.

        Returns:
            Dictionary with security status information
        """
        recent_violations = self.get_violations(since=time.time() - 3600)  # Last hour

        violation_counts = {}
        for violation in recent_violations:
            server = violation.server_name
            if server not in violation_counts:
                violation_counts[server] = {"total": 0, "by_type": {}}
            violation_counts[server]["total"] += 1
            violation_counts[server]["by_type"][violation.violation_type] = (
                violation_counts[server]["by_type"].get(violation.violation_type, 0) + 1
            )

        return {
            "total_servers_managed": len(self.policies),
            "total_violations": len(self.violations),
            "recent_violations": len(recent_violations),
            "violation_breakdown": violation_counts,
            "active_rate_limits": len(self.rate_limit_tracking),
            "audit_log_path": str(self.audit_log_path),
            "policies_configured": list(self.policies.keys()),
        }
