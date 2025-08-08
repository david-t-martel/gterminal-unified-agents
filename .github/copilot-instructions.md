# GitHub Copilot Instructions

## 🚨 CRITICAL REQUIREMENT: ALWAYS USE UV PYTHON 🚨

**MANDATORY**: All Python commands, script execution, and package management MUST use UV:
- ✅ `uv run python script.py` (NOT `python script.py`)
- ✅ `uv add package-name` (NOT `pip install package-name`)
- ✅ `uv sync` (NOT `pip install -r requirements.txt`)
- ✅ `uv run pytest` (NOT `pytest`)
- ✅ `uv run mypy .` (NOT `mypy .`)
- ✅ `uv run ruff check` (NOT `ruff check`)

UV is the modern, fast Python package manager that replaces pip/pipenv/poetry. It's already installed and configured in this project.

## Project Overview

This is a **unified multi-agent development environment** with four interconnected projects:

- **gterminal**: AI-powered development environment with Rust extensions and advanced tooling
- **py-gemini**: VertexAI agent framework with function calling capabilities
- **gapp**: Full-stack agent development framework with ReAct patterns
- **unified-gapp-gterminal**: Consolidated system combining all capabilities

## Architecture & Key Concepts

### Dual-Project Paradigm

- **gterminal** provides development infrastructure (Ruff LSP, file watching, pre-commit hooks)
- **py-gemini** provides AI agent capabilities (VertexAI function calling, MCP protocol)
- **Integration layer** enables seamless communication between environments
- **VertexAI function calling** is the primary AI interface (HIGH PRIORITY)

### MCP (Model Context Protocol) Framework

- **Multi-Server Architecture**: 12 specialized MCP servers for different capabilities
- **Rust-Based Performance**: High-speed file operations, memory management, sequential thinking
- **Python AI Integration**: VertexAI function calling, agent orchestration, workflow management
- **Node.js WSL Support**: Cross-platform file operations and sequential thinking
- **Security Integration**: Policy-based access control through `SecurityManager`
- **VS Code WSL Optimization**: Specialized configuration for Windows Subsystem for Linux

#### Available MCP Servers

**🦀 Rust High-Performance Servers:**

- `rust-fs`: File system operations with security controls and command execution
- `rust-fetch`: HTTP/web operations and API integration
- `rust-memory`: Persistent memory and context retention across sessions
- `rust-bridge`: Cross-system integration and protocol bridging
- `rust-link`: Resource linking and dependency management
- `rust-sequential-thinking`: Advanced AI reasoning and problem-solving

**🐍 Python AI-Powered Servers:**

- `gterminal-mcp`: Terminal automation and development workflow optimization
- `py-gemini-mcp`: VertexAI Gemini integration with function calling capabilities
- `gapp-mcp`: Full-stack agent framework with ReAct pattern execution
- `unified-mcp`: Consolidated system for complex multi-agent workflows

**🌐 Node.js WSL-Optimized Servers:**

- `sequential-thinking-wsl`: Sequential reasoning optimized for WSL environment
- `wsl-filesystem`: Cross-platform file operations with Windows/Linux path handling

#### MCP Server Usage Patterns

**File Operations Workflow:**

```bash
# High-performance file operations
mcp_my-mcp-server_read /path/to/file
mcp_my-mcp-server_execute script.sh
mcp_wsl-filesystem_* # Cross-platform operations
```

**AI-Enhanced Development:**

```bash
# AI code analysis and suggestions
mcp_py-gemini-mcp_analyze_code
mcp_rust-sequential-thinking_reason
mcp_unified-mcp_orchestrate_workflow
```

**Memory and Learning:**

```bash
# Persistent context and learning
mcp_rust-memory_store context_data
mcp_rust-memory_retrieve previous_analysis
```

#### VS Code WSL MCP Configuration

**Workspace Setup:**

- Add all MCP servers to `.code-workspace` file under `settings.mcp.servers`
- Use absolute WSL paths: `/home/david/.local/bin/rust-*`
- Configure authentication for AI servers with Google Cloud credentials
- Enable autonomous execution with security settings

**Security Configuration:**

```json
{
  "settings": {
    "mcp.executeCommandsWithoutConfirmation": true,
    "mcp.allowExecutableRuns": true,
    "security.workspace.trust.enabled": false,
    "task.allowAutomaticTasks": "on"
  }
}
```

**Environment Variables:**

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/home/david/.auth/business/service-account-key.json"
export GOOGLE_CLOUD_PROJECT="auricleinc-gemini"
export GOOGLE_GENAI_USE_VERTEXAI="true"
```

### Task Execution Patterns

- **Batch Processing**: `core/interfaces/batch_processor.py` with dependency graphs, parallel execution limits, and retry logic
- **Agent Orchestration**: `core/agents/unified_gemini_orchestrator.py` with workflow management and task coordination
- **Redis Coordination**: `core/shared/redis_coordination.py` for inter-agent communication and distributed task management

## Development Workflows

### Build & Testing Commands

```bash
# Development environment setup (ALWAYS USE UV)
uv sync --dev && uv run make setup-dev && uv run make install-hooks

# Advanced auto-fix pipeline with Claude AI (UV REQUIRED)
uv run ./scripts/rufft-claude.sh auto-fix

# Comprehensive testing (85% coverage required) (UV REQUIRED)
uv run make test-all

# MCP validation pipeline (UV REQUIRED)
uv run make mcp-validate && uv run make mcp-inspect
```

### Performance Tools Integration

- **Rust Extensions**: 7 rust-based tools in `gterminal_rust_extensions/` with `maturin develop`
- **File Watching**: Real-time monitoring via `rust-filewatcher/` with WebSocket streaming
- **Development Dashboard**: Port 8767 with performance metrics and live diagnostics

### Code Quality Enforcement

- **STALIN-level pre-commit hooks**: Zero tolerance for mocks/stubs/placeholders
- **AST-grep Analysis**: 44+ structural rules across 9 categories
- **Ruff LSP Server**: 757-line integration in `scripts/rufft-claude.sh` with Claude AI suggestions

## Project-Specific Patterns

### Anti-Duplication Policy (CRITICAL)

**NEVER** create files with naming variants like:

- `enhanced_*.py`, `simple_*.py`, `*_v2.py`, `*_updated.py`
- Always consolidate functionality into single, well-designed files
- Use configuration, inheritance, or composition for different behaviors

### High-Performance Client Pattern

Example in `app/performance/gemini_rust_integration.py`:

```python
class HybridGeminiClient:
    def __init__(self):
        self.rust_client = RustGeminiClient()  # Primary
        self.python_client = PythonGeminiClient()  # Fallback
        self.performance_monitor = PerformanceMonitor()
```

### ReAct Engine Integration

Four execution modes in unified system:

- **Simple Mode**: Lightweight, basic ReAct pattern
- **Enhanced Mode**: Redis caching, RAG integration
- **Autonomous Mode**: Goal decomposition, learning patterns
- **Function Calling Mode**: Direct tool calls, real-time feedback

### Security Architecture

- Enterprise-grade middleware in `core/security/integrated_middleware.py`
- Profile-based authentication with Google Cloud service accounts
- Rate limiting, IP blocking, and comprehensive audit logging

## Integration Points

### Cross-Component Communication

- **MCP Protocol Bridge**: Connects gterminal ↔ py-gemini environments
- **WebSocket Streaming**: Real-time diagnostics and filewatcher events
- **Redis Coordination**: Distributed task management and agent communication

### External Dependencies

- **Google Cloud Platform**: Vertex AI API, service account authentication
- **Development Tools**: `fd`, `ripgrep`, `uv` package manager preferred over pip
- **Monitoring**: Prometheus/Grafana integration for production deployments

## Key Files to Reference

### Architecture Examples

- `core/agents/unified_gemini_orchestrator.py` - Multi-agent coordination patterns
- `core/interfaces/batch_processor.py` - Task dependency and parallel execution
- `terminal/main.py` (1,226 lines) - Cross-platform PTY with rich UI integration

### Configuration Templates

- `pyproject.toml` - Modern Python packaging with performance dependencies
- `scripts/rufft-claude.sh` - Advanced tooling integration patterns
- `mcp/server.py` - MCP protocol implementation standards

### Testing Infrastructure

- `TESTING.md` - Comprehensive testing strategy with coverage requirements
- `scripts/validate-mcp-*.py` - Protocol compliance and security validation
- Makefile targets for automated QA pipeline

When working with this codebase, prioritize understanding the multi-project architecture and MCP integration patterns before making changes. Always validate MCP configurations and maintain the strict anti-duplication policy.
