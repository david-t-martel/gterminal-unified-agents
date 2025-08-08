#!/usr/bin/env python3
"""
Simple test MCP server to verify configuration.
"""

from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("Test MCP Server")


@mcp.tool()
async def hello_world(name: str = "World") -> str:
    """
    Simple hello world function.

    Args:
        name: Name to greet

    Returns:
        Greeting message
    """
    return f"Hello, {name}!"


@mcp.tool()
async def echo(message: str) -> str:
    """
    Echo back the message.

    Args:
        message: Message to echo

    Returns:
        The same message
    """
    return message


if __name__ == "__main__":
    mcp.run()
