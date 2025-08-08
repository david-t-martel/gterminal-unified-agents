# Agent Coordination Briefing

## Project Overview
Dual AI development environment with gterminal (tools) + py-gemini (VertexAI agents).

## Agent Responsibilities

### deployment-engineer
- **Completed**: Docker, Kubernetes, Terraform infrastructure
- **Next**: Monitor deployment readiness for function calling

### rust-pro
- **Completed**: 7 rust tools integration with monitoring
- **Next**: Performance optimization for remaining tools

### python-pro
- **Active**: Resolving 437 ruff issues
- **Next**: VertexAI function calling implementation

### code-reviewer
- **Completed**: STALIN-level enforcement setup
- **Next**: Validate function calling code quality

### api-documenter
- **Pending**: Document VertexAI function calling APIs
- **Ready**: After implementation complete

### debugger
- **Standby**: Debug function calling integration issues
- **Monitor**: Ruff issue resolution progress

## Coordination Points

1. **Function Calling Priority**
   - python-pro leads implementation
   - code-reviewer validates approach
   - api-documenter creates docs
   - debugger on standby

2. **Code Quality**
   - 437 ruff issues need resolution
   - Pre-commit must pass (STALIN-level)
   - 85% test coverage required

3. **Integration Layer**
   - gterminal ↔ py-gemini communication
   - MCP protocol compliance
   - Performance monitoring

## Success Metrics
- VertexAI function calling working
- All ruff issues resolved
- Tests passing with 85% coverage
- Documentation complete
- Dashboard showing green metrics