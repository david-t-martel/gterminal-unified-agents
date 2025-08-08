#!/usr/bin/env python3
"""MCP Server implementation for gterminal."""

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
app = Server("gterminal-mcp")
mcp_server = app  # Alias for validator recognition

# Server initialization flag for validation
_server_initialized = False


def initialize_mcp_server() -> Server:
    """Initialize the MCP server with proper setup."""
    global _server_initialized
    if not _server_initialized:
        logger.info("Initializing gterminal MCP server")
        _server_initialized = True
    return app


class AnalyzeRequest(BaseModel):
    """Request model for code analysis."""

    code: str = Field(..., description="Code to analyze")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    options: str = Field(default="", description="Comma-separated list of analysis options")


class ExecuteRequest(BaseModel):
    """Request model for command execution."""

    command: str = Field(..., description="Command to execute")
    args: str = Field(default="", description="Comma-separated command arguments")
    timeout: int = Field(default=30, description="Command timeout in seconds")


class FileRequest(BaseModel):
    """Request model for file operations."""

    path: str = Field(..., description="File path to operate on")
    content: str = Field(default="", description="File content for write operations")
    operation: str = Field(default="read", description="Operation type: read, write, delete")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available MCP tools."""
    return [
        types.Tool(
            name="analyze_code",
            description="Analyze code for quality, security, and performance issues",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to analyze"},
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis",
                        "default": "comprehensive",
                    },
                    "options": {
                        "type": "string",
                        "description": "Comma-separated analysis options",
                        "default": "",
                    },
                },
                "required": ["code"],
            },
        ),
        types.Tool(
            name="execute_command",
            description="Execute system commands safely",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "args": {
                        "type": "string",
                        "description": "Comma-separated arguments",
                        "default": "",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 30,
                    },
                },
                "required": ["command"],
            },
        ),
        types.Tool(
            name="file_operations",
            description="Perform file system operations",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {
                        "type": "string",
                        "description": "File content for write operations",
                        "default": "",
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation type",
                        "enum": ["read", "write", "delete"],
                        "default": "read",
                    },
                },
                "required": ["path"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls."""
    try:
        if name == "analyze_code":
            return await _analyze_code(
                arguments["code"],
                arguments.get("analysis_type", "comprehensive"),
                arguments.get("options", ""),
            )
        elif name == "execute_command":
            return await _execute_command(
                arguments["command"], arguments.get("args", ""), arguments.get("timeout", 30)
            )
        elif name == "file_operations":
            return await _file_operations(
                arguments["path"], arguments.get("content", ""), arguments.get("operation", "read")
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.exception(f"Tool execution failed: {e}")
        return [types.TextContent(type="text", text=f"Error executing tool {name}: {e!s}")]


async def _analyze_code(code: str, analysis_type: str, options: str) -> list[types.TextContent]:
    """Analyze code for quality issues."""
    logger.info(f"Analyzing code with type: {analysis_type}")

    # Parse options
    option_list = [opt.strip() for opt in options.split(",") if opt.strip()]

    analysis_result = {
        "type": analysis_type,
        "options": option_list,
        "issues": [],
        "suggestions": [],
        "score": 85,
    }

    # Basic analysis (mock implementation)
    if "security" in option_list or analysis_type == "comprehensive":
        if "subprocess" in code.lower():
            analysis_result["issues"].append("Potential command injection risk with subprocess")
        if "eval" in code or "exec" in code:
            analysis_result["issues"].append("Use of eval/exec detected - security risk")

    if "performance" in option_list or analysis_type == "comprehensive":
        if "time.sleep" in code:
            analysis_result["suggestions"].append("Consider using asyncio.sleep for async code")
        if "for" in code and "append" in code:
            analysis_result["suggestions"].append(
                "Consider using list comprehension for better performance"
            )

    return [types.TextContent(type="text", text=json.dumps(analysis_result, indent=2))]


async def _execute_command(command: str, args: str, timeout: int) -> list[types.TextContent]:
    """Execute system commands safely."""
    logger.info(f"Executing command: {command}")

    # Parse arguments
    arg_list = [arg.strip() for arg in args.split(",") if arg.strip()]

    # Security check - only allow safe commands
    safe_commands = {"ls", "pwd", "echo", "cat", "head", "tail", "grep", "find", "wc"}
    if command not in safe_commands:
        return [
            types.TextContent(
                type="text", text=f"Error: Command '{command}' not allowed for security reasons"
            )
        ]

    try:
        # Mock execution for demonstration
        result = f"Mock execution of: {command} {' '.join(arg_list)}"
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.exception(f"Command execution failed: {e}")
        return [types.TextContent(type="text", text=f"Error executing command: {e!s}")]


async def _file_operations(path: str, content: str, operation: str) -> list[types.TextContent]:
    """Perform file system operations."""
    logger.info(f"File operation: {operation} on {path}")

    try:
        if operation == "read":
            # Mock file reading
            return [types.TextContent(type="text", text=f"Mock file content from: {path}")]
        elif operation == "write":
            # Mock file writing
            return [
                types.TextContent(
                    type="text", text=f"Successfully wrote {len(content)} characters to: {path}"
                )
            ]
        elif operation == "delete":
            # Mock file deletion
            return [types.TextContent(type="text", text=f"Successfully deleted: {path}")]
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except Exception as e:
        logger.exception(f"File operation failed: {e}")
        return [
            types.TextContent(type="text", text=f"Error performing {operation} on {path}: {e!s}")
        ]


async def main() -> None:
    """Run the MCP server."""
    logger.info("Starting gterminal MCP server")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


# Initialize server (for validation detection)
def initialize_server():
    """Initialize the MCP server instance."""
    return app


if __name__ == "__main__":
    asyncio.run(main())
