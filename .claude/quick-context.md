# gterminal + py-gemini Quick Context

## Current Focus
- **HIGH PRIORITY**: Implement VertexAI function calling in py-gemini
- **Active**: Resolving 437 remaining ruff issues (down from 853)
- **Active**: JSON validation fixes for MCP compliance

## Key Commands
```bash
# Check ruff issues
cd /home/david/agents/gterminal
./rufft-claude.sh check

# Run dashboard
./rufft-claude.sh serve

# Test pre-commit hooks
pre-commit run --all-files

# Rust tools status
./rust-tools-integration.sh status
```

## Recent Achievements
- ✅ 7 rust-based tools integrated
- ✅ Ruff LSP with AI suggestions (757 lines)
- ✅ Dashboard on port 8767
- ✅ 416 ruff issues auto-resolved (48.8%)
- ✅ STALIN-level pre-commit enforcement

## Active Issues
- 437 ruff formatting/linting issues remaining
- JSON validation for MCP schemas
- VertexAI function calling not yet implemented

## Key Files
- `/home/david/agents/gterminal/rufft-claude.sh` - Enhanced ruff LSP
- `/home/david/agents/gterminal/rust-tools-integration.sh` - Rust tools
- `/home/david/agents/gterminal/pyproject.toml` - Dependencies
- `/home/david/agents/py-gemini/` - VertexAI agent framework

## Quality Standards
- 85% test coverage minimum
- Zero mocks/stubs in production
- 100 char line length
- Pre-commit hooks must pass