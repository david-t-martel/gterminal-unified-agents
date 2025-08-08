# Contributing to gterminal-unified-agents

Thank you for your interest in contributing to gterminal-unified-agents! This document provides guidelines and information about contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/gterminal-unified-agents.git`
3. Add the upstream remote: `git remote add upstream https://github.com/david-t-martel/gterminal-unified-agents.git`

## Development Setup

### Prerequisites

- Python 3.11 or 3.12
- Rust (latest stable)
- uv (Python package manager)
- Node.js (for frontend development)

### Environment Setup

```bash
# Install dependencies
uv sync --dev

# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Setup pre-commit hooks
uv run pre-commit install
```

### Building Rust Extensions

```bash
cd gterminal_rust_extensions
maturin develop
```

## Making Changes

### Branch Naming

- Feature: `feature/description`
- Bug fix: `fix/description`
- Documentation: `docs/description`
- Security: `security/description`

### Code Style

- Python: Follow PEP 8, use `black` for formatting
- Rust: Use `rustfmt` for formatting
- Run linters before committing: `./scripts/lint.sh`

### Commit Messages

Follow conventional commits format:

```
type(scope): description

body (optional)

footer (optional)
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Testing

### Running Tests

```bash
# Python tests
uv run pytest

# Rust tests
cd gterminal_rust_extensions && cargo test

# Integration tests
./scripts/test-integration.sh

# Security tests
./scripts/security-scan.sh
```

### Test Coverage

- Maintain minimum 80% test coverage
- Add tests for new features
- Update tests when modifying existing code

## Submitting Changes

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add or update tests
4. Ensure all tests pass
5. Update documentation
6. Submit a pull request

### Pull Request Requirements

- [ ] Tests pass
- [ ] Code is linted
- [ ] Documentation updated
- [ ] Security reviewed
- [ ] Performance impact assessed

## Security

### Reporting Security Issues

Please don't report security vulnerabilities through GitHub issues. Instead:

- Email us at security@example.com
- Use GitHub's private security reporting

### Security Guidelines

- Never commit secrets or credentials
- Validate all inputs
- Use secure coding practices
- Follow authentication/authorization patterns

## Performance

### Performance Guidelines

- Profile performance-critical code
- Benchmark changes
- Consider memory usage
- Optimize hot paths

## Release Process

### Versioning

We use Semantic Versioning (SemVer):

- MAJOR.MINOR.PATCH
- Major: Breaking changes
- Minor: New features (backwards compatible)
- Patch: Bug fixes

### Release Steps

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release PR
4. Merge to main
5. Tag release
6. GitHub Actions handles deployment

## Questions?

- Join our discussions
- Open an issue for bugs
- Email us for security issues

Thank you for contributing!
