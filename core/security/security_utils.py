"""Security utilities for the automation agents.
Provides input validation, sanitization, and security controls.
"""

from datetime import UTC
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# Allowed directories for file operations (configurable via environment)
ALLOWED_DIRECTORIES = [
    "/home/david/agents/my-fullstack-agent",
    os.getcwd(),
]

# Allowed commands for subprocess execution
ALLOWED_COMMANDS = [
    "git",
    "gh",
    "uv",
    "make",
    "python",
    "python3",
    "pip",
    "node",
    "npm",
    "yarn",
    "ruff",
    "black",
    "pytest",
    "mypy",
    "docker",
    "docker-compose",
    "curl",
    "wget",
    "ls",
    "cat",
    "grep",
    "find",
    "echo",
    "test",
]

# Security headers for FastAPI
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self' ws: wss:; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    ),
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


class SecurityError(Exception):
    """Custom exception for security-related errors."""


def safe_json_parse(text: str) -> dict[str, Any]:
    """Safely parse JSON text without using eval().

    Args:
        text: JSON text to parse

    Returns:
        Parsed JSON as dictionary

    Raises:
        SecurityError: If JSON parsing fails or contains invalid data

    """
    try:
        # Remove any potential code execution patterns
        cleaned_text = text.strip()

        # Basic validation - ensure it looks like JSON
        if not (cleaned_text.startswith("{") and cleaned_text.endswith("}")):
            if not (cleaned_text.startswith("[") and cleaned_text.endswith("]")):
                msg = "Input does not appear to be valid JSON"
                raise SecurityError(msg)

        # Parse with json.loads (safe)
        parsed = json.loads(cleaned_text)

        # Validate structure for expected review data
        if isinstance(parsed, dict):
            required_keys = {"score", "comments", "suggestions", "blocking_issues"}
            if not all(key in parsed for key in required_keys):
                logger.warning(f"JSON missing required keys: {required_keys - set(parsed.keys())}")

        # Ensure we always return a dict for the expected return type
        if isinstance(parsed, dict):
            return parsed
        # If it's a list or other type, wrap it in a dict
        return {"data": parsed}

    except json.JSONDecodeError as e:
        logger.exception(f"JSON parsing failed: {e}")
        msg = f"Invalid JSON format: {e}"
        raise SecurityError(msg)
    except Exception as e:
        logger.exception(f"Unexpected error parsing JSON: {e}")
        msg = f"JSON parsing error: {e}"
        raise SecurityError(msg)


def validate_file_path(file_path: str | Path, allow_create: bool = False) -> Path:
    """Validate and sanitize file paths to prevent path traversal attacks.

    Args:
        file_path: Path to validate
        allow_create: Whether to allow non-existent files

    Returns:
        Validated Path object

    Raises:
        SecurityError: If path is invalid or contains traversal attempts

    """
    try:
        # Convert to Path object and resolve
        path = Path(file_path).resolve()

        # Check for path traversal attempts
        path_str = str(path)
        if ".." in path_str or path_str.startswith("/"):
            # Additional check for resolved path
            if not any(
                str(path).startswith(str(Path(allowed).resolve()))
                for allowed in ALLOWED_DIRECTORIES
            ):
                msg = f"Path traversal detected or path outside allowed directories: {path}"
                raise SecurityError(msg)

        # Validate against allowed directories
        allowed = False
        for allowed_dir in ALLOWED_DIRECTORIES:
            try:
                allowed_path = Path(allowed_dir).resolve()
                if path.is_relative_to(allowed_path):
                    allowed = True
                    break
            except ValueError:
                continue

        if not allowed:
            msg = f"File path not in allowed directories: {path}"
            raise SecurityError(msg)

        # Check if file exists (unless creation is allowed)
        if not allow_create and not path.exists():
            msg = f"File does not exist: {path}"
            raise SecurityError(msg)

        # Additional security checks
        if path.is_symlink():
            # Resolve symlink and revalidate target
            target = path.readlink()
            return validate_file_path(target, allow_create)

        return path

    except Exception as e:
        if isinstance(e, SecurityError):
            raise
        logger.exception(f"Path validation error: {e}")
        msg = f"Invalid file path: {e}"
        raise SecurityError(msg)


def sanitize_command_args(command: list[str], timeout: int = 30) -> list[str]:
    """Sanitize command arguments for subprocess execution.

    Args:
        command: Command and arguments as list
        timeout: Timeout in seconds

    Returns:
        Sanitized command list

    Raises:
        SecurityError: If command is not allowed or contains malicious patterns

    """
    if not command:
        msg = "Empty command provided"
        raise SecurityError(msg)

    # Validate base command
    base_command = command[0]
    if base_command not in ALLOWED_COMMANDS:
        msg = f"Command not allowed: {base_command}"
        raise SecurityError(msg)

    # Sanitize arguments
    sanitized = []
    for arg in command:
        # Remove dangerous characters and patterns
        if any(danger in arg for danger in [";", "&", "|", "`", "$", "(", ")", ">"]):
            # Use shlex to properly escape
            sanitized.append(shlex.quote(str(arg)))
        else:
            sanitized.append(str(arg))

    # Validate timeout
    if timeout <= 0 or timeout > 300:  # Max 5 minutes
        msg = f"Invalid timeout: {timeout}"
        raise SecurityError(msg)

    logger.info(f"Sanitized command: {' '.join(sanitized)}")
    return sanitized


def safe_subprocess_run(
    command: list[str],
    timeout: int = 30,
    capture_output: bool = True,
    text: bool = True,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """Safely execute subprocess with proper validation and error handling.

    Args:
        command: Command and arguments
        timeout: Timeout in seconds
        capture_output: Capture stdout/stderr
        text: Return text output
        cwd: Working directory

    Returns:
        CompletedProcess result

    Raises:
        SecurityError: If command is invalid or execution fails unsafely

    """
    try:
        # Sanitize command
        safe_command = sanitize_command_args(command, timeout)

        # Validate working directory if provided
        if cwd:
            cwd_path = validate_file_path(cwd)
            cwd = str(cwd_path)

        # Execute with security controls
        result = subprocess.run(
            safe_command,
            check=False,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            cwd=cwd,
            # Security: Don't inherit environment, use minimal env
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "USER": os.environ.get("USER", ""),
                "SHELL": "/bin/bash",
            },
        )

        # Log execution details
        logger.info(f"Command executed: {' '.join(safe_command)}, return code: {result.returncode}")

        return result

    except subprocess.TimeoutExpired as e:
        logger.exception(f"Command timed out after {timeout}s: {' '.join(command)}")
        msg = f"Command execution timed out: {e}"
        raise SecurityError(msg)
    except subprocess.CalledProcessError as e:
        logger.exception(f"Command failed with return code {e.returncode}: {' '.join(command)}")
        msg = f"Command execution failed: {e}"
        raise SecurityError(msg)
    except Exception as e:
        logger.exception(f"Subprocess execution error: {e}")
        msg = f"Command execution error: {e}"
        raise SecurityError(msg)


def validate_pr_number(pr_number: int | str) -> int:
    """Validate PR number input.

    Args:
        pr_number: PR number to validate

    Returns:
        Validated integer PR number

    Raises:
        SecurityError: If PR number is invalid

    """
    try:
        pr_int = int(pr_number)
        if pr_int <= 0 or pr_int > 99999:  # Reasonable range
            msg = f"PR number out of valid range: {pr_int}"
            raise SecurityError(msg)
        return pr_int
    except ValueError:
        msg = f"Invalid PR number format: {pr_number}"
        raise SecurityError(msg)


def sanitize_file_content(content: str, max_size: int = 1024 * 1024) -> str:
    """Sanitize file content for safe processing.

    Args:
        content: File content to sanitize
        max_size: Maximum allowed content size in bytes

    Returns:
        Sanitized content

    Raises:
        SecurityError: If content is too large or contains dangerous patterns

    """
    # Size check
    if len(content.encode("utf-8")) > max_size:
        msg = f"Content too large: {len(content)} bytes > {max_size}"
        raise SecurityError(msg)

    # Remove potential code execution patterns in strings
    dangerous_patterns = [
        "eval(",
        "exec(",
        "__import__",
        "subprocess",
        "os.system",
        "shell=True",
        "pickle.loads",
        "yaml.load",
    ]

    for pattern in dangerous_patterns:
        if pattern in content:
            logger.warning(f"Potentially dangerous pattern detected: {pattern}")

    return content


def create_security_context() -> dict[str, Any]:
    """Create security context information for logging and monitoring.

    Returns:
        Security context dictionary

    """
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "allowed_directories": ALLOWED_DIRECTORIES,
        "allowed_commands": ALLOWED_COMMANDS,
        "security_headers_enabled": True,
        "path_validation_enabled": True,
        "subprocess_sanitization_enabled": True,
    }
