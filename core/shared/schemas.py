"""Shared schema definitions to prevent duplication."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class Severity(str, Enum):
    """Severity levels for issues and alerts."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class CodeIssue:
    """Represents a code issue or finding."""

    type: str
    severity: Severity
    file_path: str
    line_number: int
    description: str
    suggested_fix: str
    code_snippet: str = ""
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "severity": self.severity.value
            if isinstance(self.severity, Severity)
            else self.severity,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "description": self.description,
            "suggested_fix": self.suggested_fix,
            "code_snippet": self.code_snippet,
            "category": self.category,
        }


class SearchQuery(BaseModel):
    """Search query parameters."""

    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=10, description="Maximum results to return")
    include_patterns: list[str] = Field(
        default_factory=list, description="File patterns to include"
    )
    exclude_patterns: list[str] = Field(
        default_factory=list, description="File patterns to exclude"
    )
    search_type: str = Field(
        default="content", description="Type of search: content, filename, or both"
    )


class MockMCPServer:
    """Mock MCP server for testing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.running = False
        self.health = 100
        self.requests_handled = 0

    def start(self) -> None:
        """Start the mock server."""
        self.running = True

    def stop(self) -> None:
        """Stop the mock server."""
        self.running = False

    def handle_request(self) -> None:
        """Simulate handling a request."""
        if self.running:
            self.requests_handled += 1
            return {"status": "success", "server": self.name}
        return {"status": "error", "message": "Server not running"}
