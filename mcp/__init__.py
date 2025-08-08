"""MCP Servers for Multi-Agent Framework.

This module contains MCP server implementations combining:
- Google Gemini AI capabilities
- PyO3-wrapped Rust performance (rust-llm)
- Direct tool access for Claude Code

Available MCP Servers:
- gemini_workspace_analyzer: High-performance workspace analysis
- Additional servers to be added
"""

__all__ = [
    "gemini_workspace_analyzer",
]
