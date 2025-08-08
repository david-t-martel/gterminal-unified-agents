# AST-grep Rules Summary - Gemini CLI Terminal

This document summarizes the comprehensive AST-grep rules extracted from my-fullstack-agent project to enhance the gterminal project's code fixing capabilities.

## Extracted Rules Overview (44+ Rules Across 9 Categories)

### 1. Python Performance Patterns (`python-performance-patterns.yml`)

**25 rules covering**:

- List comprehension optimizations
- Dictionary performance patterns
- String concatenation efficiency
- Function call optimizations
- Iterator usage best practices
- File I/O performance
- Memory management
- Async/await patterns
- JSON processing optimization

**Key rules**:

- `slow-list-append-loop`: Suggests list comprehensions over append loops
- `inefficient-string-format`: Promotes f-strings over .format()
- `blocking-io-in-async`: Detects blocking calls in async functions
- `cpu-intensive-python`: Suggests Rust extensions for heavy computation

### 2. MCP Patterns (`mcp-patterns.yml`)

**28 rules covering**:

- MCP tool schema validation
- FastMCP usage patterns
- Parameter validation for MCP tools
- Claude CLI compatibility (avoiding complex union types)
- Structured error responses
- Authentication integration

**Key rules**:

- `mcp-complex-union-types`: Prevents Claude CLI incompatible parameter types
- `mcp-comma-separated-params`: Promotes Claude CLI compatible parameter parsing
- `mcp-tool-missing-schema`: Ensures proper tool documentation

### 3. Security Patterns (`security-patterns.yml`)

**25 rules covering**:

- Hardcoded secrets detection
- Command injection prevention
- Path traversal prevention
- SQL injection prevention
- HTTPS validation
- Authentication patterns
- Input validation
- File upload security

**Key rules**:

- `hardcoded-api-key`: Detects hardcoded API keys
- `shell-injection-risk`: Prevents command injection vulnerabilities
- `path-traversal-risk`: Validates file path operations
- `disabled-ssl-verification`: Catches disabled SSL verification

### 4. Async Patterns (`async-patterns.yml`)

**21 rules covering**:

- Async function patterns
- Await usage validation
- Asyncio best practices
- Error handling in async code
- Resource management
- Concurrency patterns
- Timeout handling
- Event loop patterns

**Key rules**:

- `blocking-calls-in-async`: Detects blocking I/O in async functions
- `missing-await`: Catches missing await keywords
- `sequential-async-calls`: Suggests concurrent execution
- `missing-timeout`: Ensures async operations have timeouts

### 5. Rust Performance Rules (For Future Extensions)

- **`avoid-clone-in-loop.yml`**: Detects unnecessary cloning inside loops
- **`box-default.yml`**: Suggests using Box::default() instead of Box::new(Default::default())
- **`inefficient-hashmap-lookup.yml`**: Identifies inefficient HashMap lookup patterns
- **`string-concatenation-loop.yml`**: Catches inefficient string concatenation in loops
- **`unnecessary-collect.yml`**: Finds unnecessary .collect() calls followed by .into_iter()
- **`vec-push-in-loop.yml`**: Suggests Vec::with_capacity() for loops with push operations

### 6. Rust Security Rules

- **`unsafe-code.yml`**: Requires documentation for unsafe blocks
- **`raw-pointers.yml`**: Validates raw pointer usage
- **`transmute.yml`**: Flags dangerous transmute operations

### 7. Rust Error Handling Rules

- **`handling.yml`**: The main error handling rule that catches .unwrap() calls and suggests better alternatives

### 8. MCP-Specific Rust Rules

- **`rust-mcp-command-validation.yml`**: Ensures MCP commands validate parameters
- **`rust-mcp-permission-check.yml`**: Requires permission checks for file operations in MCP handlers
- **`server-patterns.yml`**: TypeScript MCP server patterns

### 9. Code Quality Rules

- **`type-patterns.yml`**: Type annotation and validation patterns
- **`refactoring-patterns.yml`**: Common refactoring opportunities
- **`typing-fixes.yml`**: TypeScript/Python typing improvements
- **`react-patterns.yml`**: React/UI component patterns

## Configuration

### Main Configuration (`sgconfig.yml`)

The configuration imports all rule files and sets up:

- File patterns for Python (`.py`, `.pyi`) and Rust (`.rs`)
- Severity levels (error, warning, info)
- Project-specific patterns for Gemini CLI, MCP servers, and Rust extensions
- Auto-fix configuration with safe and unsafe fix categories

### Usage Examples

```bash
# Scan all files with all rules
ast-grep scan --config .ast-grep/sgconfig.yml

# Test specific rule on specific file
ast-grep scan --rule .ast-grep/rules/mcp-patterns.yml mcp/

# Run only Python performance rules
ast-grep scan --rule .ast-grep/rules/python-performance-patterns.yml gemini_cli/

# Run security patterns
ast-grep scan --rule .ast-grep/rules/security-patterns.yml .
```

## Benefits for Gemini CLI Terminal Project

### 1. Code Quality Improvements

- **Python Code**: MCP patterns ensure Claude CLI compatibility
- **Security**: Comprehensive security rule coverage prevents common vulnerabilities
- **Performance**: Identifies bottlenecks and suggests optimizations

### 2. MCP Compliance

- **Parameter Types**: Ensures MCP tools use Claude CLI compatible parameter types
- **Error Handling**: Promotes structured error responses
- **Authentication**: Validates security patterns in MCP servers

### 3. Developer Experience

- **Auto-fixes**: Many rules provide automatic fixes
- **Clear Messages**: Each rule includes explanatory notes
- **IDE Integration**: Can be integrated with VS Code for real-time feedback

## Integration with Build Process

The rules are integrated into our build pipeline through:

1. **Make targets**: Enhanced Makefile with AST-grep integration
2. **Pre-commit hooks**: Automatic validation before commits
3. **CI/CD**: GitHub Actions integration for pull request validation

## Next Steps for gterminal

1. **VS Code Integration**: Add AST-grep extension configuration
2. **Custom Rules**: Create gterminal-specific rules for CLI patterns
3. **Auto-fix Pipeline**: Implement automated fixing for safe patterns
4. **Metrics**: Track code quality improvements over time
5. **Documentation**: Integrate rule violations into documentation generation

This comprehensive rule set significantly enhances gterminal's code fixing capabilities and ensures consistent, secure, and performant code across the entire project.
