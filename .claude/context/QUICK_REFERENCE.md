# GTerminal Quick Reference

## Project Summary

**GTerminal** - A complete AI-powered terminal with ReAct engine, MCP dual-mode support, and multi-agent capabilities.

## Key Commands

```bash
# Setup and build
make setup-full         # Complete setup with Rust extensions
make rust-build         # Build Rust extensions

# Run the terminal
make run               # Start GTerminal
make run-debug         # Start with debug logging

# Development
make test              # Run tests
make benchmark         # Performance benchmarks
make format            # Format code
```

## Architecture Overview

```
gterminal/
├── core/              # ReAct engine implementation
├── mcp/              # MCP client/server modules
├── terminal/         # PTY/TTY terminal UI
├── gterminal/        # Main application entry
└── tests/            # Test suites
```

## Key Technologies

- **ReAct Engine**: Think-Act-Observe reasoning loops
- **MCP Dual-Mode**: Both client and server capabilities
- **Rust Extensions**: 5-100x performance improvements
- **Service Auth**: No API keys, service accounts only

## Current Status (2025-08-07)

✅ **Implemented**

- Complete ReAct engine with persistence
- MCP server with 20+ tools
- MCP client for Gemini servers
- True terminal emulation
- Multi-agent communication

🚧 **In Progress**

- Rust extension build/install
- HTTP/WebSocket transports
- Real Gemini API integration

## Performance Targets

- ReAct steps: <100ms
- File operations: 5-15x faster
- Terminal rendering: 60+ FPS
- Memory usage: <500MB

## Agent Team Contributions

- **Backend Architect**: System design and architecture
- **Python Pro**: ReAct engine and MCP implementation
- **Rust Pro**: Performance extensions
- **Security Auditor**: Authentication and security
- **Deployment Engineer**: Build and deployment

## Quick Start

1. Clone the repository
2. Run `make setup-full`
3. Configure service account credentials
4. Run `make run` to start GTerminal

## Key Files

- `core/react_engine.py` - ReAct implementation
- `mcp/server.py` - MCP server with tools
- `mcp/client.py` - MCP client connections
- `terminal/ui.py` - Terminal interface
- `Makefile` - All build commands
