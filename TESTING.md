# Testing Infrastructure Documentation

This document describes the comprehensive testing infrastructure migrated from my-fullstack-agent and adapted for the gterminal project.

## 🎯 Overview

The gterminal project now includes a world-class testing infrastructure with:

- **85% minimum code coverage** with comprehensive reporting
- **Zero-tolerance for mock tests** (real integration tests only)
- **Comprehensive pre-commit hooks** with quality gates
- **MCP server validation** pipeline
- **Security scanning** with bandit and safety
- **Claude AI integration** for automatic error fixing

## 📦 Dependencies

### Core Testing Framework

- `pytest>=8.0.0` - Primary testing framework
- `pytest-asyncio>=0.23.0` - Async test support
- `pytest-cov>=4.0.0` - Code coverage
- `pytest-xdist>=3.6.0` - Parallel test execution
- `pytest-mock>=3.14.0` - Mocking (allowed only in fixtures)
- `pytest-benchmark>=4.0.0` - Performance benchmarking
- `pytest-timeout>=2.3.1` - Test timeouts

### Code Quality Tools

- `ruff>=0.6.0` - Fast Python linter and formatter
- `black>=24.0.0` - Code formatter
- `mypy>=1.8.0` - Static type checking
- `bandit[toml]>=1.7.9` - Security scanning
- `safety>=3.2.0` - Dependency vulnerability scanning
- `pre-commit>=3.8.0` - Git hooks management

### Optional Dependencies

- `jsonschema>=4.23.0` - MCP configuration validation
- `fastmcp>=2.0.0` - MCP server development

## 🧪 Testing Strategy

### Test Categories

Tests are organized by markers for selective execution:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests with real services
- `@pytest.mark.e2e` - End-to-end workflow tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.security` - Security validation tests
- `@pytest.mark.mcp` - MCP compliance tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.cli` - CLI interface tests
- `@pytest.mark.gemini` - Gemini API integration tests

### Anti-Mock Policy

The testing infrastructure enforces a strict no-mock policy:

- ✅ **Real fixtures** in `conftest.py` are allowed
- ✅ **Integration tests** with actual services
- ❌ **Mock objects** in test implementations
- ❌ **Stub methods** with placeholder responses

This ensures tests validate actual behavior rather than test implementation assumptions.

## 🔧 Configuration

### pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
minversion = "8.0"
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--cov=gemini_cli",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under=85",  # 85% minimum coverage
    "--durations=10",       # Show slowest 10 tests
]
```

### Coverage Configuration

```toml
[tool.coverage.run]
source = ["gemini_cli"]
omit = [
    "tests/*",
    "gemini_cli/__main__.py",
    "**/migrations/*",
    "**/venv/*",
    "**/.venv/*",
]
branch = true  # Branch coverage enabled
```

## 🎮 Usage

### Quick Start

```bash
# Full setup with all testing dependencies
make setup-full

# Run comprehensive quality checks
make qa

# Run tests with coverage
make test-all

# Run in parallel for speed
make test-parallel
```

### Individual Commands

```bash
# Code quality
make lint              # Full linting suite
make lint-security     # Security-focused linting
make fix              # Auto-fix code issues
make fix-claude       # AI-assisted fixing

# Testing
make test             # Standard test run (85% coverage)
make test-all         # Comprehensive tests with reporting
make test-parallel    # Parallel execution with xdist
make test-security    # Security validation tests

# MCP Validation
make mcp-validate     # Validate MCP configurations
make mcp-test         # Test MCP server implementations
make mcp-inspect      # MCP Inspector compliance checks
```

## 🔍 Pre-commit Hooks

The pre-commit configuration includes comprehensive validation:

### Code Quality Hooks

- **Ruff**: Fast linting with auto-fix
- **Black**: Code formatting
- **isort**: Import sorting
- **MyPy**: Static type checking

### Security Hooks

- **Bandit**: Security vulnerability scanning
- **detect-private-key**: Prevents committing secrets

### Custom Validation Hooks

- **Mock Detection**: Blocks commits with mock tests
- **Quality Check**: Detects poor code patterns
- **MCP Validation**: Validates MCP server configurations

### Installation

```bash
# Install pre-commit hooks
make pre-commit-setup

# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run mock-detection --all-files
```

## 📊 MCP Validation Pipeline

### MCP Configuration Validation

The `scripts/validate-mcp-config.py` script validates:

- **JSON Schema**: Structure and required fields
- **Security**: Hardcoded secrets and dangerous commands
- **Module Resolution**: Python module availability
- **Environment Variables**: Required settings

### MCP Server Implementation Validation

The `scripts/validate-mcp-servers.py` script checks:

- **Protocol Compliance**: MCP standard adherence
- **Code Quality**: AST analysis and patterns
- **Security**: Vulnerability scanning
- **Performance**: Async patterns and blocking calls

### MCP Inspector Integration

The `scripts/mcp-inspector-check.sh` script provides:

- **Protocol Testing**: Tools/resources/prompts endpoints
- **Compliance Reporting**: JSON formatted results
- **Timeout Handling**: Graceful failure management

## 🤖 Claude AI Integration

### Auto-Fix Capabilities

The `scripts/claude-auto-fix.py` script provides:

- **Error Collection**: MyPy, Ruff, and pytest failures
- **AI Analysis**: Claude-powered code fixes
- **Safe Application**: Backup creation and validation
- **Batch Processing**: Multiple file handling

### Usage Examples

```bash
# Quick fixes with Haiku model (fast)
make fix-claude

# Comprehensive fixes with Sonnet (accurate)
uv run python scripts/claude-auto-fix.py --model sonnet --max-fixes 15

# Dry run to see potential fixes
uv run python scripts/claude-auto-fix.py --dry-run
```

## 📈 Performance Considerations

### Parallel Testing

Tests can run in parallel using pytest-xdist:

```bash
# Auto-detect CPU cores
make test-parallel

# Manual core specification
uv run pytest -n 4 --dist worksteal
```

### Test Optimization

- **Fast unit tests** run first
- **Integration tests** run with appropriate timeouts
- **Slow tests** are marked for selective execution
- **Benchmark tests** track performance regressions

## 🔒 Security Integration

### Security Scanning

Multiple layers of security validation:

1. **Bandit**: Python-specific security patterns
2. **Safety**: Dependency vulnerability database
3. **Ruff Security Rules**: S-prefixed security linters
4. **Secret Detection**: Pre-commit hook for credentials

### Security Test Execution

```bash
# Comprehensive security scan
make test-security

# Security-focused linting only
make lint-security
```

## 📋 Test Fixtures

### Core Fixtures (`tests/conftest.py`)

The conftest.py provides comprehensive fixtures:

- **Mock Clients**: Gemini and Vertex AI clients
- **Test Data**: Sample projects, prompts, and responses
- **File Operations**: Temporary directories and file management
- **Authentication**: Mock auth providers
- **Terminal Sessions**: Interactive session simulation

### Usage in Tests

```python
def test_gemini_analysis(mock_gemini_client, sample_code_files):
    """Test Gemini code analysis with real-like fixtures."""
    # Use fixtures that provide realistic test data
    # without mocking the actual implementation
    result = analyze_code(sample_code_files["good_example.py"])
    assert result.score > 80
```

## 🎯 Quality Gates

### Coverage Requirements

- **Minimum 85%** overall coverage
- **Branch coverage** enabled
- **Missing lines** reported
- **HTML reports** generated in `htmlcov/`

### Code Quality Standards

- **Zero linting errors** (Ruff)
- **Consistent formatting** (Black)
- **Type safety** (MyPy)
- **Security compliance** (Bandit)
- **No mock tests** (Custom hook)

### MCP Compliance

- **Configuration validation** against JSON schema
- **Server implementation** standards
- **Protocol compliance** via MCP Inspector
- **Performance benchmarks** for response times

## 🚀 CI/CD Integration

The testing infrastructure is designed for CI/CD integration:

### GitHub Actions Integration

```yaml
- name: Run Quality Assurance
  run: make qa

- name: Validate MCP Servers
  run: make mcp-validate

- name: Security Scan
  run: make test-security
```

### Parallel Execution Support

```yaml
- name: Run Tests in Parallel
  run: make test-parallel
```

## 🎉 Migration Summary

Successfully migrated and adapted from my-fullstack-agent:

### ✅ Completed Components

1. **Pre-commit Configuration** - Comprehensive hooks with MCP validation
2. **Test Infrastructure** - pytest with 85% coverage requirement
3. **Quality Assurance Scripts** - Mock detection and quality checks
4. **MCP Validation Pipeline** - Config, server, and inspector validation
5. **Makefile Targets** - 25+ targets for all development workflows
6. **Comprehensive Fixtures** - Real test data without mocks
7. **Claude AI Integration** - Automatic error fixing capabilities
8. **Security Integration** - Multi-layer security scanning
9. **Performance Testing** - Parallel execution and benchmarking
10. **Documentation** - Complete usage and configuration guide

### 🎯 Key Benefits

- **Zero technical debt** from mock tests
- **High confidence** through real integration testing
- **Automated quality gates** prevent regressions
- **AI-powered fixes** reduce manual debugging time
- **MCP compliance** ensures protocol adherence
- **Security-first** approach with multiple scanning layers
- **Developer productivity** through comprehensive automation

The gterminal project now has enterprise-grade testing infrastructure that ensures code quality, security, and maintainability while supporting rapid development cycles.
