"""
MCP (Model Context Protocol) Consolidation Framework

This module provides unified infrastructure for managing MCP servers,
authentication, configuration, and security across the fullstack agent system.

Core Components:
- ConsolidatedAuth: Unified authentication system
- MCPConfigManager: Configuration management and validation
- SecurityManager: Security policies and validation
- ServerRegistry: MCP server lifecycle management
"""

from .config_manager import MCPConfigManager
from .consolidated_auth import ConsolidatedAuth
from .security_manager import SecurityManager
from .server_registry import ServerRegistry

__all__ = ["ConsolidatedAuth", "MCPConfigManager", "SecurityManager", "ServerRegistry"]
