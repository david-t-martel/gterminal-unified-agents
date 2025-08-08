"""Automation package for AI-powered development workflows.

This package provides a comprehensive base class architecture for building
automation agents with reduced code duplication and consistent patterns.

Base Classes:
- BaseAutomationAgent: Abstract base for all agents
- CodeAnalysisAgent: For code analysis and AST parsing
- DocumentationAgent: For documentation generation
- TestGenerationAgent: For test generation
- CodeReviewAgent: For code review and security analysis
- MigrationAgent: For migration and diff analysis
- OrchestrationAgent: For coordinating multiple agents

Mixins:
- FileProcessingMixin: Batch file processing utilities
- AsyncMixin: Async operation helpers
- GitMixin: Git operation utilities

V2 Agents (demonstrating new architecture):
- AutoDocumentationAgent: Enhanced documentation generation
- AutoTestWriterAgent: Enhanced test generation
- AutoCodeReviewAgent: Enhanced code review with architecture analysis
- AutoOrchestrationAgent: Enhanced workflow orchestration
"""

__version__ = "0.3.0"

# Import base classes
# Legacy agent modules (original FastMCP servers) - moved to app/agents/
from .base_automation_agent import AsyncMixin
from .base_automation_agent import BaseAutomationAgent
from .base_automation_agent import FileProcessingMixin
from .base_automation_agent import GitMixin

# Import V2 agents (examples of new architecture)
# Note: V2 agents will be implemented after consolidation
try:
    from .specialized_bases import CodeAnalysisAgent
    from .specialized_bases import CodeReviewAgent
    from .specialized_bases import DocumentationAgent
    from .specialized_bases import MigrationAgent
    from .specialized_bases import OrchestrationAgent
    from .specialized_bases import TestGenerationAgent
except ImportError:
    pass  # Specialized bases not yet implemented

__all__ = [
    "AsyncMixin",
    # Base classes
    "BaseAutomationAgent",
    # Mixins
    "FileProcessingMixin",
    "GitMixin",
]
