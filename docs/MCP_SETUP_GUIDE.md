# MCP Server Setup Guide for VS Code WSL

## Overview

This guide explains how to set up and use Model Context Protocol (MCP) servers in VS Code on WSL. Our workspace is configured with 12 different MCP servers, each with unique identifiers for proper operation.

## 🚀 Available MCP Servers

### Rust-based High-Performance Servers

1. **rust-fs-server-75bfda66** - File system operations (primary)

   - Command: `/home/david/.local/bin/rust-fs-server`
   - Tools: read, write, create, move, copy, delete, stat, find, search, replace, execute

2. **rust-fs-optimized-8a2c4e91** - Optimized file operations (backup)

   - Command: `/home/david/.local/bin/rust-fs-optimized`
   - Verified working with MCP mode

3. **rust-fetch-3d4c8f29** - HTTP requests and API interactions

   - Command: `/home/david/.local/bin/rust-fetch`

4. **rust-memory-b5e9a7c1** - Memory management and workspace state

   - Command: `/home/david/.local/bin/rust-memory`
   - Memory file: `/home/david/agents/workspace-memory.json`

5. **rust-bridge-f8d6c4e2** - Cross-system integration bridge

   - Command: `/home/david/.local/bin/rust-bridge`

6. **rust-link-a7b3d9f5** - Resource linking and references

   - Command: `/home/david/.local/bin/rust-link`

7. **rust-sequential-thinking-c2e8f6a4** - Sequential reasoning and logic
   - Command: `/home/david/.local/bin/rust-sequential-thinking`

### Node.js-based WSL-Optimized Servers

8. **sequential-thinking-wsl-9b4e7c3a** - NPX sequential thinking server

   - Command: `npx -y @modelcontextprotocol/server-sequential-thinking`

9. **wsl-filesystem-d6f8a2b9** - WSL-optimized filesystem operations
   - Command: `npx -y @modelcontextprotocol/server-filesystem`
   - Allowed directories: `/home/david/agents`

### Python AI-Enhanced Development Servers

10. **gterminal-mcp-1f2e3d4c** - GTerminal AI terminal integration

    - Working directory: `/home/david/agents/gterminal`
    - Module: `mcp.server`

11. **py-gemini-mcp-5a6b7c8d** - Python Gemini AI services

    - Working directory: `/home/david/agents/py-gemini`
    - Module: `shared.core.mcp.server`

12. **gapp-mcp-9e0f1a2b** - Google Apps integration

    - Working directory: `/home/david/agents/gapp`
    - Module: `core.mcp.server`

13. **unified-mcp-3c4d5e6f** - Unified MCP management service
    - Working directory: `/home/david/agents/unified-gapp-gterminal`
    - Module: `mcp.servers.unified_mcp_manager`

## 🔧 Key Configuration Requirements

### Unique Identifiers

Each MCP server **must** have a unique identifier in the format `server-name-xxxxxxxx` where `xxxxxxxx` is a unique 8-character hex string. This prevents conflicts and ensures proper server registration.

### File Paths and Working Directories

- **Rust servers**: Use absolute paths to `/home/david/.local/bin/`
- **Python servers**: Use `uv run python -m module.path` with proper `cwd` and `PYTHONPATH`
- **Node.js servers**: Use `npx -y package-name` for auto-installation

### Environment Variables

All Python servers require Google Cloud authentication:

```json
{
  "GOOGLE_APPLICATION_CREDENTIALS": "/home/david/.auth/business/service-account-key.json",
  "GOOGLE_CLOUD_PROJECT": "auricleinc-gemini",
  "GOOGLE_CLOUD_LOCATION": "us-central1",
  "GOOGLE_GENAI_USE_VERTEXAI": "true"
}
```

## 🚀 Testing MCP Server Functionality

### Example: Testing rust-fs-server

```javascript
// Use the MCP server with its unique identifier
mcp_rust-fs-server-75bfda66_read("/home/david/agents")
mcp_rust-fs-server-75bfda66_execute(["ls", "-la"])
```

### Verified Working Tools

- **File Operations**: read, write, create, delete, move, copy
- **Search Functions**: find, search, replace
- **Command Execution**: execute with proper argument handling
- **Development Tools**: Integration with rufft-claude.sh for automated fixes

## 📋 Autonomous Execution Settings

The workspace is configured for autonomous execution without permission prompts:

```json
{
  "mcp.executeCommandsWithoutConfirmation": true,
  "mcp.allowExecutableRuns": true,
  "security.workspace.trust.enabled": false
}
```

## 🔄 Maintenance and Updates

### Adding New Servers

1. Generate unique 8-character hex identifier
2. Add to workspace `mcp.servers` section
3. Test functionality with basic operations
4. Update this documentation

### Debugging Server Issues

1. Check server binary exists and is executable
2. Verify environment variables are set correctly
3. Test with simple operations first
4. Check VS Code developer console for errors

This setup provides a comprehensive MCP-enabled development environment optimized for VS Code WSL workflows.
