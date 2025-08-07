"""Tool registry for managing available tools."""

import logging
from typing import Any

from .base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self) -> None:
        """Initialize the tool registry."""
        self._tools: dict[str, Tool] = {}
    
    def register(self, name: str, tool: Tool) -> None:
        """Register a tool.
        
        Args:
            name: Tool name
            tool: Tool instance
        """
        self._tools[name] = tool
        logger.debug(f"Registered tool: {name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a tool.
        
        Args:
            name: Tool name to remove
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
    
    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self) -> dict[str, str]:
        """List all available tools.
        
        Returns:
            Dictionary mapping tool names to descriptions
        """
        return {
            name: tool.description
            for name, tool in self._tools.items()
        }
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool is registered
        """
        return name in self._tools