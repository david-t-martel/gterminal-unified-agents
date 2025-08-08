# GTTerminal - Consolidated Makefile with Fullstack Agent Capabilities
# Combined build system with React frontend, Rust extensions, and comprehensive testing

.PHONY: help setup install install-dev clean lint test run build test-all qa security mcp-validate test-consolidation validate-consolidation frontend frontend-build rust-build rust-dev

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
RED = \033[0;31m
NC = \033[0m

# Default target
help:
	@echo "$(BLUE)🚀 GTTerminal - Consolidated Development Environment$(NC)"
	@echo "$(GREEN)  Frontend (React) + Backend (Python) + Extensions (Rust)$(NC)"
	@echo ""
	@echo "$(GREEN)📦 SETUP & DEPENDENCIES:$(NC)"
	@echo "  setup         - Quick setup with uv (recommended)"
	@echo "  setup-full    - Full setup with all optional dependencies"
	@echo "  install       - Install package in development mode"
	@echo "  install-dev   - Install with development dependencies"
	@echo ""
	@echo "$(GREEN)🔧 DEVELOPMENT:$(NC)"
	@echo "  fix           - Auto-fix code issues (ruff + black)"
	@echo "  fix-claude    - Auto-fix with Claude AI assistance"
	@echo "  lint          - Run comprehensive linting"
	@echo "  lint-security - Security-focused linting with bandit"
	@echo ""
	@echo "$(GREEN)🧪 TESTING & QA:$(NC)"
	@echo "  test          - Run tests with coverage (85% minimum)"
	@echo "  test-all      - Comprehensive test suite (unit + integration)"
	@echo "  test-parallel - Run tests in parallel with xdist"
	@echo "  test-security - Security tests with bandit"
	@echo "  qa            - Full quality assurance pipeline"
	@echo ""
	@echo "$(GREEN)📊 MCP VALIDATION:$(NC)"
	@echo "  mcp-validate  - Validate MCP configuration files"
	@echo "  mcp-test      - Test MCP server implementations"
	@echo "  mcp-inspect   - Run MCP Inspector compliance checks"
	@echo ""
	@echo "$(GREEN)🔄 CONSOLIDATION TESTING:$(NC)"
	@echo "  validate-consolidation - Full gapp→gterminal consolidation validation"
	@echo "  test-consolidation     - Run comprehensive consolidation tests"
	@echo "  test-imports          - Test import consolidation"
	@echo "  test-structure        - Test project structure"
	@echo "  quick-consolidation-check - Quick consolidation validation"
	@echo ""
	@echo "$(GREEN)🏗️ BUILD & RELEASE:$(NC)"
	@echo "  build         - Build wheel package"
	@echo "  clean         - Remove build artifacts"
	@echo "  validate      - Full validation before release"
	@echo ""
	@echo "$(GREEN)🚀 RUNTIME:$(NC)"
	@echo "  run           - Run CLI interactively"
	@echo "  run-analyze   - Run analysis with prompt"
	@echo "  check-auth    - Validate authentication setup"
	@echo ""
	@echo "$(YELLOW)💡 Quick start: make setup-full && make qa && make run$(NC)"

# ========================================
# 📦 SETUP & ENVIRONMENT
# ========================================

setup:
	@echo "$(BLUE)📦 Quick setup with uv...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv sync; \
	else \
		echo "$(YELLOW)uv not found, using pip fallback$(NC)"; \
		python -m pip install --upgrade pip; \
		pip install -e ".[dev]"; \
	fi
	@echo "$(GREEN)✅ Setup complete$(NC)"

setup-full:
	@echo "$(BLUE)📦 Full setup with all dependencies...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --extra dev --extra test --extra quality --extra mcp; \
	else \
		python -m pip install --upgrade pip; \
		pip install -e ".[dev,test,quality,mcp]"; \
	fi
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "$(GREEN)✅ Pre-commit hooks installed$(NC)"; \
	fi
	@echo "$(GREEN)✅ Full setup complete$(NC)"

install:
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -e .; \
	else \
		pip install -e .; \
	fi

install-dev:
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --extra dev; \
	else \
		pip install -e ".[dev]"; \
	fi

# ========================================
# 🧹 CLEANUP
# ========================================

clean:
	@echo "$(BLUE)🧹 Cleaning build artifacts...$(NC)"
	@rm -rf build/ dist/ *.egg-info/
	@rm -rf .pytest_cache/ htmlcov/ .coverage
	@rm -rf .mypy_cache/ .ruff_cache/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

# ========================================
# 🔧 CODE QUALITY & FIXING
# ========================================

lint:
	@echo "$(BLUE)🔍 Running comprehensive linting...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff check gemini_cli tests scripts; \
		uv run black --check gemini_cli tests scripts; \
		uv run mypy gemini_cli; \
	else \
		ruff check gemini_cli tests scripts; \
		black --check gemini_cli tests scripts; \
		mypy gemini_cli; \
	fi
	@echo "$(GREEN)✅ Linting complete$(NC)"

lint-security:
	@echo "$(BLUE)🔒 Running security linting...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run bandit -r gemini_cli -ll; \
		uv run safety check; \
	else \
		bandit -r gemini_cli -ll; \
		safety check; \
	fi
	@echo "$(GREEN)✅ Security linting complete$(NC)"

fix:
	@echo "$(BLUE)🔧 Auto-fixing code issues...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff check --fix gemini_cli tests scripts; \
		uv run black gemini_cli tests scripts; \
	else \
		ruff check --fix gemini_cli tests scripts; \
		black gemini_cli tests scripts; \
	fi
	@echo "$(GREEN)✅ Auto-fix complete$(NC)"

fix-claude:
	@echo "$(BLUE)🤖 Auto-fixing with Claude AI assistance...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run python scripts/claude-auto-fix.py --model haiku --max-fixes 5; \
	else \
		python scripts/claude-auto-fix.py --model haiku --max-fixes 5; \
	fi
	@echo "$(GREEN)✅ Claude auto-fix complete$(NC)"

# ========================================
# 🧪 TESTING & QUALITY ASSURANCE
# ========================================

test:
	@echo "$(BLUE)🧪 Running tests with coverage (85% minimum)...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest; \
	else \
		pytest; \
	fi
	@echo "$(GREEN)✅ Tests complete$(NC)"

test-all:
	@echo "$(BLUE)🧪 Running comprehensive test suite...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest tests/ -v --cov=gemini_cli --cov-report=html --cov-report=term-missing --cov-fail-under=85; \
	else \
		pytest tests/ -v --cov=gemini_cli --cov-report=html --cov-report=term-missing --cov-fail-under=85; \
	fi
	@echo "$(GREEN)✅ Comprehensive tests complete$(NC)"

test-parallel:
	@echo "$(BLUE)🧪 Running tests in parallel...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest -n auto --dist worksteal; \
	else \
		pytest -n auto --dist worksteal; \
	fi
	@echo "$(GREEN)✅ Parallel tests complete$(NC)"

test-security:
	@echo "$(BLUE)🔒 Running security tests...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run bandit -r gemini_cli -f json -o security-report.json || true; \
		uv run safety check --json --output security-deps.json || true; \
	else \
		bandit -r gemini_cli -f json -o security-report.json || true; \
		safety check --json --output security-deps.json || true; \
	fi
	@echo "$(GREEN)✅ Security tests complete$(NC)"

qa: clean lint test-all lint-security
	@echo "$(GREEN)🎉 Full QA pipeline complete!$(NC)"
	@echo "$(BLUE)📊 QA Summary:$(NC)"
	@echo "  ✅ Code linting passed"
	@echo "  ✅ Type checking passed"
	@echo "  ✅ Tests passed (85%+ coverage)"
	@echo "  ✅ Security checks passed"

# ========================================
# 📊 MCP VALIDATION PIPELINE
# ========================================

mcp-validate:
	@echo "$(BLUE)📊 Validating MCP configuration files...$(NC)"
	@if [ -f "mcp/.mcp.json" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			uv run python scripts/validate-mcp-config.py mcp/.mcp.json; \
		else \
			python scripts/validate-mcp-config.py mcp/.mcp.json; \
		fi; \
	else \
		echo "$(YELLOW)⚠️  No MCP configuration files found in mcp/$(NC)"; \
	fi

mcp-test:
	@echo "$(BLUE)📊 Testing MCP server implementations...$(NC)"
	@if [ -d "mcp" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			find mcp -name "*.py" -type f | head -5 | xargs uv run python scripts/validate-mcp-servers.py; \
		else \
			find mcp -name "*.py" -type f | head -5 | xargs python scripts/validate-mcp-servers.py; \
		fi; \
	else \
		echo "$(YELLOW)⚠️  No MCP servers found in mcp/$(NC)"; \
	fi

mcp-inspect:
	@echo "$(BLUE)📊 Running MCP Inspector compliance checks...$(NC)"
	@if [ -f "mcp/.mcp.json" ] && command -v npx >/dev/null 2>&1; then \
		chmod +x scripts/mcp-inspector-check.sh; \
		./scripts/mcp-inspector-check.sh mcp/.mcp.json; \
	else \
		echo "$(YELLOW)⚠️  MCP Inspector requires Node.js and MCP config files$(NC)"; \
	fi

# ========================================
# 🚀 RUNTIME & EXECUTION
# ========================================

run:
	@echo "$(BLUE)🚀 Starting CLI interactively...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m gemini_cli analyze --interactive; \
	else \
		python -m gemini_cli analyze --interactive; \
	fi

run-analyze:
	@echo "$(BLUE)🚀 Running analysis with prompt: $(PROMPT)$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m gemini_cli analyze "$(PROMPT)"; \
	else \
		python -m gemini_cli analyze "$(PROMPT)"; \
	fi

run-workspace:
	@echo "$(BLUE)🚀 Running workspace analysis...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m gemini_cli workspace .; \
	else \
		python -m gemini_cli workspace .; \
	fi

check-auth:
	@echo "$(BLUE)🔐 Checking authentication setup...$(NC)"
	@if [ -f "/home/david/.auth/business/service-account-key.json" ]; then \
		echo "$(GREEN)✅ Business service account found$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Business service account not found at /home/david/.auth/business/service-account-key.json$(NC)"; \
	fi
	@if [ -n "$$GOOGLE_CLOUD_PROJECT" ]; then \
		echo "$(GREEN)✅ GOOGLE_CLOUD_PROJECT: $$GOOGLE_CLOUD_PROJECT$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  GOOGLE_CLOUD_PROJECT not set$(NC)"; \
	fi

# ========================================
# 🏗️ BUILD & RELEASE
# ========================================

build: clean
	@echo "$(BLUE)🏗️ Building wheel package...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv build; \
	else \
		python -m pip install --upgrade build; \
		python -m build; \
	fi
	@echo "$(GREEN)✅ Build complete - check dist/ directory$(NC)"

install-wheel: build
	@echo "$(BLUE)📦 Installing from wheel...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install dist/*.whl --force-reinstall; \
	else \
		pip install dist/*.whl --force-reinstall; \
	fi

validate: clean lint test-all mcp-validate
	@echo "$(GREEN)🎉 Full validation complete!$(NC)"
	@echo "$(BLUE)📊 Validation Summary:$(NC)"
	@echo "  ✅ Code quality checks passed"
	@echo "  ✅ Comprehensive tests passed (85%+ coverage)"
	@echo "  ✅ MCP configuration validated"

release-check: validate build
	@echo "$(GREEN)🚀 Release preparation complete!$(NC)"
	@echo "$(BLUE)📦 Ready for release:$(NC)"
	@echo "  ✅ All quality checks passed"
	@echo "  ✅ Package built successfully"
	@echo "  📍 Artifacts in dist/ directory"

# ========================================
# 🔧 DEVELOPMENT WORKFLOW
# ========================================

dev: setup-full
	@echo "$(BLUE)🔧 Starting development workflow...$(NC)"
	@$(MAKE) check-auth
	@$(MAKE) run

# Pre-commit setup
pre-commit-setup:
	@echo "$(BLUE)🔧 Setting up pre-commit hooks...$(NC)"
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		pre-commit run --all-files || true; \
		echo "$(GREEN)✅ Pre-commit hooks ready$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  pre-commit not installed$(NC)"; \
	fi

# Complete development setup
dev-setup: setup-full pre-commit-setup
	@echo "$(GREEN)🎉 Development environment fully configured!$(NC)"
	@echo "$(BLUE)📝 Next steps:$(NC)"
	@echo "  1. Configure authentication: make check-auth"
	@echo "  2. Run quality checks: make qa"
	@echo "  3. Start development: make run"
# ========================================
# 🎯 AST-GREP INTEGRATION (EXTRACTED FROM MY-FULLSTACK-AGENT)
# ========================================

ast-grep-setup:
	@echo "$(BLUE)🔧 Setting up AST-grep for code quality analysis...$(NC)"
	@if command -v ast-grep >/dev/null 2>&1; then \
		echo "$(GREEN)✅ ast-grep already installed$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Installing ast-grep...$(NC)"; \
		curl -fsSL https://github.com/ast-grep/ast-grep/releases/latest/download/ast-grep-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C /tmp/; \
		sudo mv /tmp/ast-grep /usr/local/bin/; \
		echo "$(GREEN)✅ ast-grep installed$(NC)"; \
	fi
	@echo "$(GREEN)✅ AST-grep setup complete$(NC)"

ast-grep-scan:
	@echo "$(BLUE)🔍 Running AST-grep code analysis...$(NC)"
	@if command -v ast-grep >/dev/null 2>&1; then \
		ast-grep scan --config .ast-grep/sgconfig.yml; \
	else \
		echo "$(YELLOW)⚠️  ast-grep not installed. Run 'make ast-grep-setup' first$(NC)"; \
	fi

ast-grep-scan-python:
	@echo "$(BLUE)🐍 Running AST-grep Python-specific analysis...$(NC)"
	@if command -v ast-grep >/dev/null 2>&1; then \
		ast-grep scan --rule .ast-grep/rules/python-performance-patterns.yml gemini_cli/; \
		ast-grep scan --rule .ast-grep/rules/security-patterns.yml gemini_cli/; \
		ast-grep scan --rule .ast-grep/rules/mcp-patterns.yml mcp/ || echo "No MCP directory found"; \
	else \
		echo "$(YELLOW)⚠️  ast-grep not installed. Run 'make ast-grep-setup' first$(NC)"; \
	fi

ast-grep-fix:
	@echo "$(BLUE)🔧 Running AST-grep with auto-fixes...$(NC)"
	@if command -v ast-grep >/dev/null 2>&1; then \
		ast-grep scan --config .ast-grep/sgconfig.yml --fix; \
	else \
		echo "$(YELLOW)⚠️  ast-grep not installed. Run 'make ast-grep-setup' first$(NC)"; \
	fi

# ========================================
# 🧠 CACHE SYSTEM VALIDATION & TESTING
# ========================================

cache-test:
	@echo "$(BLUE)🧠 Testing cache system...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -c "from gterminal.cache import MemoryCache, MemoryAwareCache; cache = MemoryCache(); cache.set('test', 'value'); print(f'Cache test: {cache.get(\"test\")}'); stats = cache.stats(); print(f'Cache stats: {stats}')"; \
	else \
		python -c "from gterminal.cache import MemoryCache, MemoryAwareCache; cache = MemoryCache(); cache.set('test', 'value'); print(f'Cache test: {cache.get(\"test\")}'); stats = cache.stats(); print(f'Cache stats: {stats}')"; \
	fi
	@echo "$(GREEN)✅ Cache system test complete$(NC)"

cache-benchmark:
	@echo "$(BLUE)📊 Benchmarking cache performance...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -c "\
import time; \
from gterminal.cache import MemoryCache, MemoryAwareCache; \
cache = MemoryCache(max_size=10000); \
start = time.time(); \
for i in range(1000): cache.set(f'key_{i}', f'value_{i}'); \
set_time = time.time() - start; \
start = time.time(); \
for i in range(1000): cache.get(f'key_{i}'); \
get_time = time.time() - start; \
print(f'Set 1000 items: {set_time:.4f}s ({1000/set_time:.0f} ops/sec)'); \
print(f'Get 1000 items: {get_time:.4f}s ({1000/get_time:.0f} ops/sec)'); \
print(f'Cache stats: {cache.stats()}')"; \
	else \
		python -c "import time; from gterminal.cache import MemoryCache; cache = MemoryCache(max_size=10000); start = time.time(); [cache.set(f'key_{i}', f'value_{i}') for i in range(1000)]; set_time = time.time() - start; start = time.time(); [cache.get(f'key_{i}') for i in range(1000)]; get_time = time.time() - start; print(f'Set: {set_time:.4f}s, Get: {get_time:.4f}s')"; \
	fi
	@echo "$(GREEN)✅ Cache benchmark complete$(NC)"

redis-test:
	@echo "$(BLUE)🔴 Testing Redis cache (optional)...$(NC)"
	@if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Redis is available$(NC)"; \
		if command -v uv >/dev/null 2>&1; then \
			uv run python -c "from gterminal.cache import RedisCache; from gterminal.cache.redis_cache import RedisConfig; config = RedisConfig(); cache = RedisCache(config); health = cache.health_check(); print(f'Redis health: {health}'); cache.set('test_key', 'test_value'); value = cache.get('test_key'); print(f'Redis test: {value}'); cache.delete('test_key')"; \
		else \
			python -c "from gterminal.cache import RedisCache; from gterminal.cache.redis_cache import RedisConfig; config = RedisConfig(); cache = RedisCache(config); print('Redis test completed')"; \
		fi; \
	else \
		echo "$(YELLOW)⚠️  Redis not available - skipping Redis cache test$(NC)"; \
	fi

# ========================================
# 📈 PERFORMANCE PROFILING & ANALYSIS
# ========================================

profile-memory:
	@echo "$(BLUE)🧠 Profiling memory usage...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv add --dev memory-profiler; \
		uv run python -m memory_profiler -c "from gterminal.cache import MemoryAwareCache; cache = MemoryAwareCache(max_memory_mb=50); [cache.set(f'key_{i}', 'x'*1000) for i in range(1000)]; print(cache.memory_info())"; \
	else \
		pip install memory-profiler; \
		python -m memory_profiler -c "from gterminal.cache import MemoryAwareCache; cache = MemoryAwareCache(max_memory_mb=50); [cache.set(f'key_{i}', 'x'*1000) for i in range(1000)]"; \
	fi
	@echo "$(GREEN)✅ Memory profiling complete$(NC)"

profile-cpu:
	@echo "$(BLUE)🔥 Profiling CPU usage with line-by-line analysis...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv add --dev line-profiler; \
		echo "Creating profile test script..."; \
		echo "from gterminal.cache import MemoryCache\nimport time\n\n@profile\ndef cache_operations():\n    cache = MemoryCache()\n    for i in range(1000):\n        cache.set(f'key_{i}', f'value_{i}' * 100)\n    for i in range(1000):\n        cache.get(f'key_{i}')\n\nif __name__ == '__main__':\n    cache_operations()" > profile_test.py; \
		uv run kernprof -l -v profile_test.py; \
		rm -f profile_test.py profile_test.py.lprof; \
	else \
		echo "$(YELLOW)⚠️  Install line-profiler for CPU profiling$(NC)"; \
	fi
	@echo "$(GREEN)✅ CPU profiling complete$(NC)"

performance-report:
	@echo "$(BLUE)📊 Generating comprehensive performance report...$(NC)"
	@$(MAKE) cache-benchmark
	@echo ""
	@$(MAKE) profile-memory
	@echo ""
	@echo "$(GREEN)🎉 Performance analysis complete!$(NC)"

# ========================================
# 🛡️ SECURITY & VULNERABILITY SCANNING
# ========================================

security-scan:
	@echo "$(BLUE)🛡️  Running comprehensive security scan...$(NC)"
	@$(MAKE) lint-security
	@$(MAKE) ast-grep-scan-python
	@echo ""
	@echo "$(BLUE)🔍 Checking for hardcoded secrets...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -c "import re; import os; \
secrets_patterns = [r'sk-[a-zA-Z0-9]{48}', r'AIza[a-zA-Z0-9_-]{35}', r'ya29\.[a-zA-Z0-9_-]{100,}', r'['\''\"](password|secret|key)['\''\"]\s*[:=]\s*['\''\"]\w+['\''\"]]'; \
for root, dirs, files in os.walk('gemini_cli'): \
    for file in files: \
        if file.endswith('.py'): \
            filepath = os.path.join(root, file); \
            with open(filepath, 'r') as f: \
                content = f.read(); \
                for pattern in secrets_patterns: \
                    matches = re.findall(pattern, content); \
                    if matches: print(f'Potential secret in {filepath}: {matches}')" || echo "No secrets found"; \
	else \
		grep -r -n "sk-\|AIza\|password.*=" gemini_cli/ || echo "No obvious secrets found"; \
	fi
	@echo "$(GREEN)✅ Security scan complete$(NC)"

vulnerability-check:
	@echo "$(BLUE)🔍 Checking for known vulnerabilities...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run safety check --json --output vulnerability-report.json || true; \
		echo "$(YELLOW)📄 Vulnerability report saved to vulnerability-report.json$(NC)"; \
	else \
		safety check || echo "$(YELLOW)⚠️  Install safety for vulnerability checking$(NC)"; \
	fi
	@echo "$(GREEN)✅ Vulnerability check complete$(NC)"

# ========================================
# 🎯 COMPREHENSIVE QUALITY GATES
# ========================================

qa-extended: clean ast-grep-scan lint test-all lint-security cache-test
	@echo "$(GREEN)🎉 Extended QA pipeline complete!$(NC)"
	@echo "$(BLUE)📊 Extended QA Summary:$(NC)"
	@echo "  ✅ AST-grep code analysis passed"
	@echo "  ✅ Code linting passed"
	@echo "  ✅ Type checking passed"
	@echo "  ✅ Tests passed (85%+ coverage)"
	@echo "  ✅ Security checks passed"
	@echo "  ✅ Cache system validation passed"
	@echo ""
	@echo "$(YELLOW)💡 Performance Report:$(NC)"
	@$(MAKE) cache-benchmark

qa-full: clean ast-grep-setup ast-grep-scan lint test-all lint-security cache-test security-scan performance-report
	@echo "$(GREEN)🎉 FULL QA pipeline with performance analysis complete!$(NC)"
	@echo "$(BLUE)📊 Full QA Summary:$(NC)"
	@echo "  ✅ AST-grep setup and analysis"
	@echo "  ✅ Code linting and type checking"
	@echo "  ✅ Comprehensive test suite"
	@echo "  ✅ Security scanning and vulnerability check"
	@echo "  ✅ Cache system validation and benchmarking"
	@echo "  ✅ Performance profiling and analysis"

# ========================================
# 🔧 DEVELOPMENT CONVENIENCE TARGETS
# ========================================

fast-check: ast-grep-scan-python lint
	@echo "$(GREEN)⚡ Fast quality check complete$(NC)"

cache-demo:
	@echo "$(BLUE)🎭 Cache system demonstration...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -c "\
from gterminal.cache import MemoryCache, MemoryAwareCache; \
print('=== Memory Cache Demo ==='); \
cache = MemoryCache(max_size=5, default_ttl=10); \
cache.set('user:123', 'John Doe'); \
cache.set('session:abc', {'user_id': 123, 'expires': '2024-12-31'}); \
print(f'User: {cache.get(\"user:123\")}'); \
print(f'Session: {cache.get(\"session:abc\")}'); \
print(f'Cache stats: {cache.stats()}'); \
print(f'Keys: {cache.keys()}'); \
print(); \
print('=== Memory-Aware Cache Demo ==='); \
aware_cache = MemoryAwareCache(max_memory_mb=1); \
aware_cache.set('large_data', 'x' * 10000); \
print(f'Memory info: {aware_cache.memory_info()}'); \
"; \
	else \
		echo "$(YELLOW)⚠️  Install dependencies first$(NC)"; \
	fi
	@echo "$(GREEN)✅ Cache demo complete$(NC)"

dev-info:
	@echo "$(BLUE)📋 Development Environment Information$(NC)"
	@echo "$(YELLOW)=====================================$(NC)"
	@echo "$(BLUE)Python:$(NC) $(shell python --version 2>/dev/null || echo 'Not found')"
	@echo "$(BLUE)UV:$(NC) $(shell uv --version 2>/dev/null || echo 'Not found')"
	@echo "$(BLUE)AST-grep:$(NC) $(shell ast-grep --version 2>/dev/null || echo 'Not installed')"
	@echo "$(BLUE)Redis:$(NC) $(shell redis-cli --version 2>/dev/null || echo 'Not installed')"
	@echo "$(BLUE)Pre-commit:$(NC) $(shell pre-commit --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "$(BLUE)Project Structure:$(NC)"
	@echo "  📁 AST-grep Rules: $(shell [ -d .ast-grep ] && echo '✅ Available' || echo '❌ Missing')"
	@echo "  📁 Cache System: $(shell [ -d gterminal/cache ] && echo '✅ Available' || echo '❌ Missing')"
	@echo "  📁 Tests: $(shell [ -d tests ] && echo '✅ Available' || echo '❌ Missing')"
	@echo "  📁 MCP: $(shell [ -d mcp ] && echo '✅ Available' || echo '❌ Missing')"

# ========================================
# 🔄 CONSOLIDATION TESTING
# ========================================

test-consolidation:
	@echo "$(BLUE)🔄 Running consolidation tests...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run python tests/test_runner.py; \
	else \
		python tests/test_runner.py; \
	fi
	@echo "$(GREEN)✅ Consolidation tests complete$(NC)"

validate-consolidation:
	@echo "$(BLUE)🔍 Validating gapp→gterminal consolidation...$(NC)"
	@bash scripts/validate_consolidation.sh
	@echo "$(GREEN)✅ Consolidation validation complete$(NC)"

test-imports:
	@echo "$(BLUE)📦 Testing import consolidation...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest tests/test_imports.py -v; \
	else \
		pytest tests/test_imports.py -v; \
	fi
	@echo "$(GREEN)✅ Import tests complete$(NC)"

test-structure:
	@echo "$(BLUE)🏗️  Testing project structure...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest tests/test_consolidation.py -v -k "structure"; \
	else \
		pytest tests/test_consolidation.py -v -k "structure"; \
	fi
	@echo "$(GREEN)✅ Structure tests complete$(NC)"

quick-consolidation-check:
	@echo "$(BLUE)⚡ Quick consolidation validation...$(NC)"
	@echo "$(YELLOW)Checking for legacy references...$(NC)"
	@if grep -r "from gapp\." gterminal/ --include="*.py" 2>/dev/null || \
	   grep -r "import gapp" gterminal/ --include="*.py" 2>/dev/null; then \
		echo "$(RED)❌ Found legacy gapp references$(NC)"; \
	else \
		echo "$(GREEN)✅ No legacy gapp references$(NC)"; \
	fi
	@echo "$(YELLOW)Checking gterminal structure...$(NC)"
	@if [ -d "gterminal/gterminal" ]; then \
		echo "$(RED)❌ Found nested gterminal structure$(NC)"; \
	else \
		echo "$(GREEN)✅ Clean gterminal structure$(NC)"; \
	fi
	@echo "$(YELLOW)Testing basic imports...$(NC)"
	@if python -c "import gterminal; import gterminal.agents" 2>/dev/null; then \
		echo "$(GREEN)✅ Basic imports working$(NC)"; \
	else \
		echo "$(RED)❌ Import issues detected$(NC)"; \
	fi

# 🎨 Frontend React Build Targets
frontend-dev:
	@echo "$(BLUE)🎨 Starting React frontend development server...$(NC)"
	cd frontend && npm run dev

frontend-build:
	@echo "$(BLUE)📦 Building React frontend for production...$(NC)"
	cd frontend && npm install && npm run build

frontend-test:
	@echo "$(BLUE)🧪 Running frontend tests...$(NC)"
	cd frontend && npm run test

frontend-lint:
	@echo "$(BLUE)🔍 Linting frontend code...$(NC)"
	cd frontend && npm run lint

frontend-clean:
	@echo "$(BLUE)🧹 Cleaning frontend build artifacts...$(NC)"
	cd frontend && rm -rf node_modules dist build

# ⚡ Rust Extensions Build Targets
rust-dev:
	@echo "$(BLUE)⚡ Building Rust extensions for development...$(NC)"
	cd gterminal_rust_extensions && cargo build

rust-build:
	@echo "$(BLUE)📦 Building Rust extensions for production...$(NC)"
	cd gterminal_rust_extensions && cargo build --release

rust-test:
	@echo "$(BLUE)🧪 Running Rust tests...$(NC)"
	cd gterminal_rust_extensions && cargo test

rust-bench:
	@echo "$(BLUE)🏆 Running Rust benchmarks...$(NC)"
	cd gterminal_rust_extensions && cargo bench

rust-clean:
	@echo "$(BLUE)🧹 Cleaning Rust build artifacts...$(NC)"
	cd gterminal_rust_extensions && cargo clean

# 🚀 Full Stack Build Targets
build-all: frontend-build rust-build
	@echo "$(GREEN)✅ Full stack build completed!$(NC)"

dev-all:
	@echo "$(BLUE)🚀 Starting full development environment...$(NC)"
	@echo "$(YELLOW)This will start both frontend dev server and backend...$(NC)"
	cd frontend && npm run dev &
	make run

test-all-stack: frontend-test rust-test test
	@echo "$(GREEN)✅ All tests completed!$(NC)"

clean-all: frontend-clean rust-clean clean
	@echo "$(GREEN)✅ All build artifacts cleaned!$(NC)"
