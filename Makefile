# Gemini CLI - Makefile for build automation
# Keep it simple and focused

.PHONY: help setup install install-dev clean lint test run build

# Default target
help:
	@echo "Gemini CLI - Standalone Gemini CLI tool"
	@echo ""
	@echo "Available targets:"
	@echo "  setup      - Setup development environment"  
	@echo "  install    - Install package in development mode"
	@echo "  install-dev - Install with development dependencies"
	@echo "  clean      - Remove build artifacts"
	@echo "  lint       - Run linting (ruff, black, mypy)"
	@echo "  test       - Run tests"
	@echo "  run        - Run the CLI interactively"
	@echo "  build      - Build wheel package"
	@echo ""
	@echo "Quick start:"
	@echo "  make setup && make run"

# Setup development environment
setup:
	@echo "Setting up Gemini CLI development environment..."
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	@echo "✅ Setup complete"

# Install package in development mode
install:
	pip install -e .

# Install with development dependencies  
install-dev:
	pip install -e ".[dev]"

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run linting
lint:
	@echo "Running code quality checks..."
	ruff check gemini_cli/
	black --check gemini_cli/  
	mypy gemini_cli/
	@echo "✅ Linting complete"

# Auto-fix code issues
fix:
	@echo "Auto-fixing code issues..."
	ruff check --fix gemini_cli/
	black gemini_cli/
	@echo "✅ Auto-fix complete"

# Run tests
test:
	pytest

# Run tests with coverage
test-cov:
	pytest --cov=gemini_cli --cov-report=html --cov-report=term

# Run the CLI interactively
run:
	python -m gemini_cli analyze --interactive

# Run the CLI with a specific command
run-analyze:
	python -m gemini_cli analyze "$(PROMPT)"

# Run workspace analysis
run-workspace:
	python -m gemini_cli workspace .

# Build wheel package
build: clean
	python -m pip install --upgrade build
	python -m build
	@echo "✅ Build complete - check dist/ directory"

# Install from wheel
install-wheel: build
	pip install dist/*.whl

# Development workflow - setup and run
dev: setup
	python -m gemini_cli analyze --interactive

# Check authentication
check-auth:
	@echo "Checking authentication setup..."
	@if [ -f "/home/david/.auth/business/service-account-key.json" ]; then \
		echo "✅ Business service account found"; \
	else \
		echo "❌ Business service account not found at /home/david/.auth/business/service-account-key.json"; \
	fi

# Quick validation
validate: lint test
	@echo "✅ Validation complete"

# Release preparation
release-check: clean lint test build
	@echo "✅ Release checks complete"
	@echo "Package ready in dist/ directory"