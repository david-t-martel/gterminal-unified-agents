#!/usr/bin/env python3
"""MCP Client implementation for gterminal."""

import asyncio
import logging
from typing import Any

from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class McpClient:
    """MCP Client for communicating with MCP servers."""

    def __init__(self, server_path: str, server_args: list[str] | None = None) -> None:
        """Initialize MCP client.

        Args:
            server_path: Path to the MCP server executable
            server_args: Optional arguments for the server
        """
        self.server_path = server_path
        self.server_args = server_args or []
        self.session: ClientSession | None = None

    async def connect(self) -> None:
        """Connect to the MCP server."""
        try:
            server_params = StdioServerParameters(
                command=self.server_path, args=self.server_args, env=None
            )

            self.stdio_client = stdio_client(server_params)
            self.read_stream, self.write_stream = await self.stdio_client.__aenter__()

            # Initialize session
            from mcp.client.session import ClientSession

            self.session = ClientSession(self.read_stream, self.write_stream)

            # Initialize the connection
            await self.session.initialize()
            logger.info("Successfully connected to MCP server")

        except Exception as e:
            logger.exception(f"Failed to connect to MCP server: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        try:
            if self.session:
                await self.session.close()
            if hasattr(self, "stdio_client"):
                await self.stdio_client.__aexit__(None, None, None)
            logger.info("Disconnected from MCP server")
        except Exception as e:
            logger.exception(f"Error during disconnect: {e}")

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the MCP server."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            response = await self.session.list_tools()
            return [tool.model_dump() for tool in response.tools]
        except Exception as e:
            logger.exception(f"Failed to list tools: {e}")
            raise

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server.

        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            response = await self.session.call_tool(name, arguments)
            return {
                "content": [content.model_dump() for content in response.content],
                "isError": response.isError,
            }
        except Exception as e:
            logger.exception(f"Failed to call tool {name}: {e}")
            raise

    async def analyze_code(
        self, code: str, analysis_type: str = "comprehensive", options: str = ""
    ) -> dict[str, Any]:
        """Analyze code using the MCP server.

        Args:
            code: Code to analyze
            analysis_type: Type of analysis to perform
            options: Comma-separated analysis options

        Returns:
            Analysis results
        """
        return await self.call_tool(
            "analyze_code", {"code": code, "analysis_type": analysis_type, "options": options}
        )

    async def execute_command(
        self, command: str, args: str = "", timeout: int = 30
    ) -> dict[str, Any]:
        """Execute a command using the MCP server.

        Args:
            command: Command to execute
            args: Comma-separated arguments
            timeout: Command timeout in seconds

        Returns:
            Command execution results
        """
        return await self.call_tool(
            "execute_command", {"command": command, "args": args, "timeout": timeout}
        )

    async def file_operation(
        self, path: str, operation: str = "read", content: str = ""
    ) -> dict[str, Any]:
        """Perform file operations using the MCP server.

        Args:
            path: File path to operate on
            operation: Operation type (read, write, delete)
            content: File content for write operations

        Returns:
            File operation results
        """
        return await self.call_tool(
            "file_operations", {"path": path, "operation": operation, "content": content}
        )


async def create_mcp_client(server_path: str, server_args: list[str] | None = None) -> McpClient:
    """Create and connect to an MCP client.

    Args:
        server_path: Path to the MCP server executable
        server_args: Optional arguments for the server

    Returns:
        Connected MCP client
    """
    client = McpClient(server_path, server_args)
    await client.connect()
    return client


# Usage example
async def main() -> None:
    """Example usage of MCP client."""
    try:
        # Connect to local MCP server
        client = await create_mcp_client("python", ["-m", "mcp.server"])

        # List available tools
        tools = await client.list_tools()
        logger.info(f"Available tools: {[tool['name'] for tool in tools]}")

        # Analyze some code
        result = await client.analyze_code(
            code="print('Hello, World!')", analysis_type="security", options="security,performance"
        )
        logger.info(f"Analysis result: {result}")

        # Disconnect
        await client.disconnect()

    except Exception as e:
        logger.exception(f"Example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
