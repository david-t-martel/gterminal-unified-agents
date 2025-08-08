# CLAUDE.md - GTerminal Development Environment

This file provides guidance to Claude Code when working with the GTerminal AI-powered development environment.

## 🚀 Project Overview

GTerminal is a comprehensive AI-powered development environment designed for professional AI agent development with **VertexAI function calling capabilities as HIGH PRIORITY**. This project works in tandem with **py-gemini** to create a production-ready dual-project architecture.

### Dual-Project Architecture

1. **gterminal** (This Repository)
   - Advanced development environment with rust-based performance tools
   - Ruff LSP server integration with Claude AI suggestions
   - Real-time filewatcher with WebSocket streaming
   - Comprehensive pre-commit hooks with STALIN-level enforcement
   - Development dashboard and monitoring

2. **py-gemini** (../py-gemini/)
   - **HIGH PRIORITY**: VertexAI agent framework with function calling capabilities
   - MCP protocol servers for agent orchestration
   - Production-ready AI agent implementations
   - Google Cloud Platform integration

### Core Mission

Create a production-ready AI development ecosystem where:
- **gterminal** provides the development tooling infrastructure
- **py-gemini** provides the AI agent capabilities
- **Integration layer** enables seamless communication between environments
- **VertexAI function calling** is the primary interface for AI interactions

## 🔧 Current Infrastructure Status

### Comprehensive Tooling Implemented ✅
- **Ruff LSP Server**: 757-line advanced integration with Claude AI (`scripts/rufft-claude.sh`)
- **Rust Tools Integration**: 522-line performance monitoring system (`scripts/rust-tools-integration.sh`)
- **Real-time Filewatcher**: High-performance rust-based file monitoring
- **Development Dashboard**: Real-time metrics on port 8767
- **AST-grep Analysis**: 44+ structural code analysis rules across 9 categories
- **Pre-commit Hooks**: STALIN-level enforcement (zero tolerance for mocks/stubs/placeholders)

### Performance Metrics
- **7 rust-based tools** integrated and functional
- **416 Python issues automatically resolved** (853→437)
- **Dashboard server** running with WebSocket streaming
- **85% test coverage requirement** enforced

### Current Work Status
- **In Progress**: Fine-tuning remaining 437 ruff issues and JSON validation
- **HIGH PRIORITY**: VertexAI function calling implementation
- **Next**: gterminal ↔ py-gemini integration layer

## 🎯 HIGH PRIORITY: VertexAI Function Calling

### Immediate Objectives
1. **Function Calling Validation Pipeline**
   - Ensure py-gemini agents have proper function calling capabilities
   - Validate VertexAI API integration and authentication
   - Test agent-to-agent communication through function calls

2. **Integration Layer Development**
   - MCP protocol bridge between gterminal and py-gemini
   - Seamless authentication flow
   - Performance monitoring across both environments

3. **Production Readiness**
   - Comprehensive testing of VertexAI function calling
   - Error handling and timeout management
   - Monitoring and observability

## 🏗️ Build Commands and Workflows

### Development Environment
```bash
# Setup comprehensive development environment
./scripts/rust-tools-integration.sh setup

# Start development dashboard
./scripts/rust-tools-integration.sh dashboard

# Run comprehensive auto-fix pipeline
./scripts/rust-tools-integration.sh auto-fix

# Start ruff LSP server with Claude AI integration
./scripts/rufft-claude.sh lsp-start

# Start real-time diagnostic streaming
./scripts/rufft-claude.sh stream
```

### VertexAI Function Calling (HIGH PRIORITY)
```bash
# Test VertexAI function calling capabilities (py-gemini)
cd ../py-gemini
python test_function_calling.py

# Validate MCP protocol compliance
python validate_integration.py

# Run comprehensive agent testing
python test_mcp_servers_comprehensive.py
```

### Quality Assurance (STALIN-level enforcement)
```bash
# Run comprehensive quality checks
./scripts/rufft-claude.sh auto-fix      # Apply all automated fixes
./scripts/rust-tools-integration.sh benchmark  # Performance benchmarking
ruff check .                           # Ensure zero issues
mypy .                                # Type checking
pre-commit run --all-files             # STALIN-level validation
```

### Integration Testing
```bash
# Test gterminal ↔ py-gemini integration
./scripts/test-integration.sh

# Validate MCP server configurations
./scripts/validate-mcp-servers.py

# Run end-to-end agent workflows
./scripts/integration-test-runner.sh
```

## 📁 Architecture and File Structure

### Key Components
```
/home/david/agents/gterminal/
├── scripts/
│   ├── rufft-claude.sh              # 757-line Ruff LSP + Claude AI integration
│   ├── rust-tools-integration.sh    # 522-line performance monitoring
│   ├── start-filewatcher.sh        # Rust filewatcher startup
│   └── unified-rust-tools.sh       # Combined tooling wrapper
├── rust-filewatcher/               # High-performance file monitoring
├── .ast-grep/                      # Structural analysis rules (44+ rules)
├── pyproject.toml                  # Comprehensive dependencies
├── dashboard_status.json           # Real-time development metrics
└── deployment/                     # Production deployment configs
```

### Integration Points with py-gemini
```
/home/david/agents/py-gemini/
├── servers/                        # MCP protocol servers
│   ├── gemini_master_architect.py  # Core AI orchestration
│   ├── unified_mcp_gateway.py      # 32-tool unified gateway
│   └── function_calling_utils.py   # VertexAI function calling
├── shared/                         # Common utilities
└── test_function_calling.py        # Function calling validation
```

## 🤖 AI Agent Integration

### Current Agent Capabilities
- **Code Review Agent**: Security and performance analysis
- **Workspace Analyzer**: Project architecture analysis  
- **Documentation Generator**: Comprehensive documentation
- **Master Architect**: Cross-project orchestration

### VertexAI Function Calling Architecture
```python
# Example function calling pattern (HIGH PRIORITY)
from vertexai import generative_models as genai

def create_function_calling_agent():
    tools = [
        genai.Tool.from_function(analyze_code),
        genai.Tool.from_function(generate_documentation),
        genai.Tool.from_function(review_security),
    ]
    
    model = genai.GenerativeModel(
        'gemini-1.5-pro-002',  # Latest model with function calling
        tools=tools
    )
    
    return model
```

## 🔒 Quality Standards (STALIN-level Enforcement)

### Zero Tolerance Policy
- **NEVER** create enhanced/simple file variants
- **NEVER** use mocks, placeholders, or stubs to satisfy tests
- **NEVER** commit code that doesn't meet quality standards
- **NEVER** use `--no-verify` to bypass pre-commit hooks

### Quality Gates
- **85% minimum test coverage** (enforced by pytest-cov)
- **Zero ruff issues** (currently 437 remaining, target: 0)
- **MyPy type checking** with strict configuration
- **AST-grep structural analysis** (44+ rules across 9 categories)
- **Security scanning** with bandit and safety

### Pre-commit Hook Enforcement
```bash
# Pre-commit hooks will automatically block:
- Code with linting issues
- Tests that don't pass
- Coverage below 85%
- Type checking errors
- Security vulnerabilities
- JSON validation failures
```

## 🚧 Current Development Status

### Completed ✅
- [x] Comprehensive rust-tools integration (7 tools)
- [x] Advanced ruff LSP server with Claude AI integration
- [x] Real-time development dashboard with WebSocket streaming
- [x] Pre-commit hooks with STALIN-level enforcement
- [x] AST-grep structural analysis framework
- [x] Production-ready DevOps infrastructure (Docker, Kubernetes, Terraform)

### In Progress 🔄
- [ ] **Fine-tuning 437 remaining ruff issues** (down from 853 - 48% improvement)
- [ ] JSON validation fixes for dashboard and configuration files
- [ ] **CLAUDE.md documentation updates** (this file)

### HIGH PRIORITY Next Steps 🎯
- [ ] **VertexAI function calling validation pipeline**
- [ ] **gterminal ↔ py-gemini integration layer with MCP protocol**
- [ ] **Automated testing for VertexAI agent capabilities**
- [ ] Complete elimination of remaining ruff issues

## 🌐 Development Workflow

### Daily Development Cycle
1. **Start Environment**
   ```bash
   ./scripts/rust-tools-integration.sh setup
   ./scripts/rufft-claude.sh lsp-start
   ```

2. **Real-time Development**
   - Dashboard at http://localhost:8767
   - WebSocket diagnostics on port 8768
   - Auto-fix pipeline continuously running

3. **Quality Validation**
   ```bash
   ./scripts/rufft-claude.sh auto-fix
   pre-commit run --all-files
   ```

4. **VertexAI Testing** (HIGH PRIORITY)
   ```bash
   cd ../py-gemini
   python test_function_calling.py
   ```

### Integration Testing Workflow
```bash
# Test cross-project integration
./scripts/integration-test-runner.sh

# Validate MCP protocol compliance
./scripts/validate-mcp-config.py

# Performance benchmarking
./scripts/rust-tools-integration.sh benchmark
```

## 🔮 Future Roadmap

### Phase 1: VertexAI Function Calling (HIGH PRIORITY)
- Complete function calling implementation in py-gemini
- Comprehensive testing and validation
- Error handling and timeout management
- Performance optimization

### Phase 2: Integration Layer
- MCP protocol bridge between gterminal and py-gemini  
- Seamless authentication flow
- Cross-project monitoring and observability
- Unified development experience

### Phase 3: Production Optimization
- Advanced caching strategies
- WebSocket streaming optimization
- Monitoring dashboard enhancements
- Performance profiling and optimization

### Phase 4: Ecosystem Expansion
- Additional VertexAI model support
- Multi-modal capabilities (text, images, code)
- Advanced agent orchestration
- Enterprise deployment features

## 📋 Development Guidelines

### Code Quality
1. **ALWAYS** read files before editing (use Read tool first)
2. **NEVER** create duplicate files with variants like "enhanced_", "simple_", etc.
3. **ALWAYS** use comprehensive testing (85% coverage minimum)
4. **ALWAYS** ensure pre-commit hooks pass (STALIN-level enforcement)

### Performance
1. **Use rust-based tools** for maximum performance (rg, fd, ast-grep)
2. **Enable caching** through Redis integration
3. **Monitor performance** through dashboard metrics
4. **Profile bottlenecks** using integrated profiling tools

### AI Integration
1. **Prioritize VertexAI function calling** in all AI-related work
2. **Use MCP protocol** for agent-to-agent communication
3. **Ensure proper authentication** for Google Cloud services
4. **Test function calling thoroughly** before production deployment

## 🎭 Agent Coordination

### Successful Agent Patterns
- **deployment-engineer**: DevOps infrastructure setup
- **rust-pro**: Performance tooling integration
- **python-pro**: LSP server enhancement
- **code-reviewer**: Quality enforcement

### Cross-Agent Dependencies
- Context preserved through Memory MCP server
- Project state saved in `.claude/context/` directories
- Agent coordination through structured todo management
- Performance metrics shared across agent sessions

## 🔗 Integration with py-gemini

### MCP Protocol Communication
- **Unified MCP Gateway**: 32-tool integration point
- **Function Calling Utils**: VertexAI-specific utilities
- **Shared Authentication**: Google Cloud profile management
- **Cross-Project Monitoring**: Unified observability

### Development Sync Points
1. **Authentication**: Shared GCP profile management
2. **Quality Standards**: Same pre-commit hook enforcement
3. **Testing**: Integrated test suites
4. **Deployment**: Coordinated production deployment

This dual-project architecture ensures that gterminal provides the development environment while py-gemini delivers the AI capabilities, with VertexAI function calling as the primary integration interface.