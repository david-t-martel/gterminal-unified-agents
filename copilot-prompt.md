# GitHub Copilot MCP Integration Prompt

## Multi-Agent Development Environment Setup

You are working in a **unified multi-agent development environment** with 12 specialized MCP (Model Context Protocol) servers providing advanced AI capabilities. This workspace combines gterminal, py-gemini, gapp, and unified systems for comprehensive development workflows.

## Available MCP Servers

### 🦀 **Rust High-Performance Servers** (6 servers)

#### 1. **rust-fs-optimized** (`rust-fs-optimized-server-a1b2c3d4`)

- **Purpose**: High-speed file operations with security controls
- **Capabilities**: File CRUD, directory operations, command execution, file watching
- **Usage**: `mcp_rust-fs-optimized_read`, `mcp_rust-fs-optimized_execute`
- **Performance**: ~10x faster than Python equivalents

#### 2. **rust-fetch** (`rust-fetch-server-e5f6g7h8`)

- **Purpose**: HTTP/web operations and API integration
- **Capabilities**: Parallel requests, caching, rate limiting, batch operations
- **Usage**: `mcp_rust-fetch_fetch`, `mcp_rust-fetch_batch_fetch`
- **Features**: Up to 10 concurrent requests with intelligent caching

#### 3. **rust-memory** (`rust-memory-server-i9j0k1l2`)

- **Purpose**: Persistent memory and context retention
- **Capabilities**: Cross-session memory, context graphs, semantic storage
- **Usage**: `mcp_rust-memory_store`, `mcp_rust-memory_retrieve`
- **Backend**: SQLite with 1GB capacity

#### 4. **rust-bridge** (`rust-bridge-server-m3n4o5p6`)

- **Purpose**: Cross-system integration and protocol bridging
- **Capabilities**: Protocol translation, async communication, system bridging
- **Usage**: `mcp_rust-bridge_connect`, `mcp_rust-bridge_translate`

#### 5. **rust-link** (`rust-link-server-q7r8s9t0`)

- **Purpose**: Resource linking and dependency management
- **Capabilities**: Resource discovery, dependency tracking, link validation
- **Usage**: `mcp_rust-link_link`, `mcp_rust-link_discover`

#### 6. **rust-sequential-thinking** (`rust-sequential-thinking-server-u1v2w3x4`)

- **Purpose**: Advanced AI reasoning and problem-solving
- **Capabilities**: Multi-step reasoning, hypothesis generation, solution verification
- **Usage**: `mcp_rust-sequential-thinking_think`, max 50 thoughts
- **Advanced**: Chain-of-thought reasoning with revision capabilities

### 🐍 **Python AI-Powered Servers** (4 servers)

#### 7. **gterminal-mcp** (`gterminal-mcp-server-y5z6a7b8`)

- **Purpose**: Terminal automation and development workflow optimization
- **Capabilities**: Build automation, testing pipelines, development task orchestration
- **Usage**: Integration with rufft-claude.sh, automated testing, CI/CD workflows
- **Features**: Advanced auto-fix pipeline with Claude AI integration

#### 8. **py-gemini-mcp** (`py-gemini-mcp-server-c9d0e1f2`)

- **Purpose**: VertexAI Gemini integration with function calling capabilities
- **Capabilities**: Function calling validation, AI agent orchestration, VertexAI integration
- **Usage**: Primary AI interface for complex reasoning tasks
- **Authentication**: Google Cloud service account with VertexAI enabled

#### 9. **gapp-mcp** (`gapp-mcp-server-g3h4i5j6`)

- **Purpose**: Full-stack agent framework with ReAct pattern execution
- **Capabilities**: 4 execution modes (simple, enhanced, autonomous, function calling)
- **Usage**: Complex multi-step workflows, terminal interfaces, enterprise features
- **Features**: Redis caching, performance monitoring, security middleware

#### 10. **unified-mcp** (`unified-mcp-server-k7l8m9n0`)

- **Purpose**: Consolidated system for complex multi-agent workflows
- **Capabilities**: Cross-component communication, unified authentication, deployment
- **Usage**: Enterprise-grade workflows combining all system capabilities
- **Features**: Zero-downtime deployment, unified monitoring

### 🧠 **Memory and Coordination Servers** (2 servers)

#### 11. **redis-memory** (`redis-memory-mcp-server-o1p2q3r4`)

- **Purpose**: Distributed memory and inter-agent communication
- **Capabilities**: Redis-based caching, session management, agent coordination
- **Usage**: `mcp_redis-memory_store`, `mcp_redis-memory_retrieve`
- **Backend**: Redis localhost:6379 DB 0

#### 12. **redis-memory-gpu** (`redis-memory-gpu-server-s5t6u7v8`)

- **Purpose**: GPU-accelerated collaborative memory with semantic search
- **Capabilities**: 20+ MCP tools for advanced agent coordination
- **Key Tools**:
  - `store_memory` - GPU-accelerated embedding generation
  - `semantic_search` - GPU similarity search across memories
  - `distribute_task` - Intelligent task distribution to agents
  - `register_agent` - Agent management and coordination
  - `gpu_status` - Real-time GPU performance monitoring
- **Features**: FAISS vector search, CUDA/ROCm acceleration, collaborative workflows
- **Backend**: Redis localhost:6379 DB 1 with GPU acceleration

## Workspace Configuration

### Autonomous Execution Enabled

```json
{
  "mcp.executeCommandsWithoutConfirmation": true,
  "mcp.allowExecutableRuns": true,
  "security.workspace.trust.enabled": false,
  "task.allowAutomaticTasks": "on"
}
```

### Environment Setup

- **Python**: `/home/david/.local/share/uv/python/cpython-3.12.7-linux-x86_64-gnu/bin/python`
- **Package Manager**: `uv` (preferred over pip)
- **Authentication**: Google Cloud service account at `/home/david/.auth/business/service-account-key.json`
- **Project**: `auricleinc-gemini`

## Development Workflows

### High-Performance File Operations

```bash
# Use rust-fs-optimized for all file operations
mcp_rust-fs-optimized_read /path/to/file
mcp_rust-fs-optimized_execute "complex_command.sh"
mcp_rust-fs-optimized_watch /directory
```

### AI-Enhanced Development

```bash
# Use py-gemini-mcp for AI reasoning
mcp_py-gemini-mcp_analyze_code
mcp_py-gemini-mcp_function_call

# Use rust-sequential-thinking for complex problems
mcp_rust-sequential-thinking_reason
```

### Collaborative Memory Management

```bash
# Use redis-memory-gpu for advanced coordination
mcp_redis-memory-gpu_store_memory
mcp_redis-memory-gpu_semantic_search
mcp_redis-memory-gpu_distribute_task
mcp_redis-memory-gpu_gpu_status
```

### Batch Processing and Performance

```bash
# Use rust-fetch for web operations
mcp_rust-fetch_batch_fetch
mcp_rust-fetch_cache_status

# Use unified-mcp for enterprise workflows
mcp_unified-mcp_orchestrate_workflow
```

## Best Practices

### Performance Optimization

1. **Prefer Rust servers** for file and network operations (10x performance boost)
2. **Use GPU acceleration** via redis-memory-gpu for memory-intensive tasks
3. **Leverage caching** through rust-fetch and redis-memory systems
4. **Batch operations** when processing multiple items

### AI Integration

1. **Function calling** through py-gemini-mcp for VertexAI capabilities
2. **Sequential thinking** via rust-sequential-thinking for complex reasoning
3. **Multi-agent coordination** using redis-memory-gpu task distribution
4. **Context sharing** between agents via collaborative memory systems

### Development Automation

1. **rufft-claude.sh integration** via gterminal-mcp for auto-fix workflows
2. **CI/CD automation** through gapp-mcp ReAct patterns
3. **Unified deployment** via unified-mcp zero-downtime strategies
4. **Real-time monitoring** through GPU status and performance metrics

### Security and Reliability

1. **Autonomous execution** enabled for development efficiency
2. **Service account authentication** for all AI servers
3. **Redis backend security** with separate databases for different purposes
4. **Error handling** with graceful fallbacks in all Rust servers

## Tool Usage Patterns

### File Operations Workflow

```
1. rust-fs-optimized → High-speed file operations
2. rust-memory → Persistent context storage
3. redis-memory → Session management
4. redis-memory-gpu → Semantic file search
```

### AI Reasoning Workflow

```
1. py-gemini-mcp → VertexAI function calling
2. rust-sequential-thinking → Complex reasoning
3. redis-memory-gpu → Collaborative context
4. unified-mcp → Workflow orchestration
```

### Performance Monitoring Workflow

```
1. redis-memory-gpu_gpu_status → GPU performance
2. rust-fetch_cache_status → Network cache status
3. gapp-mcp → Application performance metrics
4. gterminal-mcp → Development pipeline health
```

This comprehensive MCP setup provides enterprise-grade AI development capabilities with GPU acceleration, collaborative memory, and high-performance Rust tooling. All servers are configured for autonomous execution with proper authentication and security controls.
