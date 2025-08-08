"""Gemini Agents Integration Module.

This module provides integration of gemini-agents capabilities into the
my-fullstack-agent framework. It includes specialized agents for code review,
workspace analysis, documentation generation, architecture design, and code generation.
"""

from typing import Any

__version__ = "0.3.0"

# Import from consolidated agent service (only available classes)
# Import actual agent services from consolidated files
# Temporarily disabled due to MCP FastMCP import issues
# from .code_review_agent import CodeReviewAgentService
from .consolidated_agent_service import BaseAgentService
from .consolidated_agent_service import CodeGenerationService as CodeGenerationAgentService
from .consolidated_agent_service import Job
from .consolidated_agent_service import JobStatus
from .documentation_generator_agent import (
    DocumentationGeneratorService as DocumentationGeneratorAgentService,
)
from .workspace_analyzer_agent import WorkspaceAnalyzerService as WorkspaceAnalyzerAgentService

MasterArchitectAgentService = None  # Still to be implemented

# MCP server modules (now from consolidated agent files)
from .code_review_agent import mcp as code_reviewer_mcp
from .documentation_generator_agent import mcp as documentation_mcp
from .mcp_code_generator import mcp as code_generator_mcp
from .mcp_master_architect import mcp as master_architect_mcp
from .workspace_analyzer_agent import mcp as workspace_analyzer_mcp

# WorkspaceAnalyzerService imported above from consolidated service

__all__ = [
    # Base classes
    "BaseAgentService",
    "CodeGenerationAgentService",
    # Agent services
    "CodeReviewAgentService",
    "DocumentationGeneratorAgentService",
    "Job",
    "JobStatus",
    "MasterArchitectAgentService",
    "WorkspaceAnalyzerAgentService",
    "code_generator_mcp",
    # MCP servers
    "code_reviewer_mcp",
    "documentation_mcp",
    "master_architect_mcp",
    "workspace_analyzer_mcp",
]

# Agent registry for dynamic loading
AGENT_REGISTRY = {
    "code-reviewer": CodeReviewAgentService,
    "workspace-analyzer": WorkspaceAnalyzerAgentService,
    "documentation-generator": DocumentationGeneratorAgentService,
    "master-architect": MasterArchitectAgentService,
    "code-generator": CodeGenerationAgentService,
}

# MCP server registry
MCP_REGISTRY = {
    "code-reviewer": code_reviewer_mcp,
    "workspace-analyzer": workspace_analyzer_mcp,
    "documentation-generator": documentation_mcp,
    "master-architect": master_architect_mcp,
    "code-generator": code_generator_mcp,
}


def get_agent_service(agent_type: str) -> BaseAgentService:
    """Get an agent service instance by type.

    Args:
        agent_type: The type of agent to get

    Returns:
        An instance of the requested agent service

    Raises:
        ValueError: If the agent type is not recognized

    """
    if agent_type not in AGENT_REGISTRY:
        msg = f"Unknown agent type: {agent_type}. Available: {list(AGENT_REGISTRY.keys())}"
        raise ValueError(msg)

    return AGENT_REGISTRY[agent_type]()


def get_mcp_server(agent_type: str) -> Any:
    """Get an MCP server instance by type.

    Args:
        agent_type: The type of MCP server to get

    Returns:
        The MCP server instance

    Raises:
        ValueError: If the agent type is not recognized

    """
    if agent_type not in MCP_REGISTRY:
        msg = f"Unknown MCP server type: {agent_type}. Available: {list(MCP_REGISTRY.keys())}"
        raise ValueError(msg)

    return MCP_REGISTRY[agent_type]
