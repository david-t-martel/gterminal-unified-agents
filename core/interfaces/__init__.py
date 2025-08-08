"""Interface Adapters for Unified Agents.

This module provides interface adapters that expose the unified agents
through different access patterns:

- MCP Adapter: Single MCP server for Claude CLI integration
- CLI Adapter: Unified command-line interface
- API Adapter: Single REST API with WebSocket support
- Terminal Adapter: Rich terminal interface with ReAct reasoning display

All adapters provide access to the same unified agents:
- UnifiedCodeReviewer
- UnifiedWorkspaceAnalyzer
- UnifiedDocumentationGenerator
- UnifiedGeminiOrchestrator
"""

from .api_adapter import app as unified_api
from .cli_adapter import cli as unified_cli
from .mcp_adapter import mcp as unified_mcp
from .terminal_adapter import TerminalAdapter
from .terminal_adapter import create_terminal_session

__all__ = [
    "TerminalAdapter",
    "create_terminal_session",
    "unified_api",
    "unified_cli",
    "unified_mcp",
]
