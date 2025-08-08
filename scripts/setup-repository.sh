#!/bin/bash
# Repository setup and configuration script for gterminal-unified-agents
set -euo pipefail

# Configuration
REPO_OWNER="david-t-martel"
REPO_NAME="gterminal-unified-agents"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}"
}

info() { log "${BLUE}INFO${NC}" "$@"; }
warn() { log "${YELLOW}WARN${NC}" "$@"; }
error() { log "${RED}ERROR${NC}" "$@"; }
success() { log "${GREEN}SUCCESS${NC}" "$@"; }

# Check if gh CLI is available
check_gh_cli() {
    if ! command -v gh >/dev/null 2>&1; then
        error "GitHub CLI (gh) is required but not installed"
        error "Install with: curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg"
        error "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main' | sudo tee /etc/apt/sources.list.d/github-cli.list"
        error "sudo apt update && sudo apt install gh"
        return 1
    fi

    # Check authentication
    if ! gh auth status >/dev/null 2>&1; then
        error "GitHub CLI not authenticated. Run: gh auth login"
        return 1
    fi

    success "✅ GitHub CLI ready"
    return 0
}

# Setup branch protection rules
setup_branch_protection() {
    info "🔒 Setting up branch protection rules..."

    # Main branch protection
    gh api repos/${REPO_OWNER}/${REPO_NAME}/branches/main/protection \
        --method PUT \
        --field required_status_checks='{"strict":true,"contexts":["ci"]}' \
        --field enforce_admins=true \
        --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true,"require_code_owner_reviews":false}' \
        --field restrictions=null \
        --field allow_force_pushes=false \
        --field allow_deletions=false \
        2>/dev/null || warn "Branch protection may already exist or require admin privileges"

    success "✅ Branch protection configured"
}

# Configure repository settings
configure_repository() {
    info "⚙️ Configuring repository settings..."

    # General repository settings
    gh api repos/${REPO_OWNER}/${REPO_NAME} \
        --method PATCH \
        --field allow_merge_commit=false \
        --field allow_squash_merge=true \
        --field allow_rebase_merge=true \
        --field delete_branch_on_merge=true \
        --field allow_auto_merge=true \
        2>/dev/null || warn "Some repository settings may require admin privileges"

    # Enable security features
    gh api repos/${REPO_OWNER}/${REPO_NAME}/vulnerability-alerts \
        --method PUT \
        2>/dev/null || warn "Vulnerability alerts may already be enabled"

    gh api repos/${REPO_OWNER}/${REPO_NAME}/automated-security-fixes \
        --method PUT \
        2>/dev/null || warn "Automated security fixes may already be enabled"

    success "✅ Repository settings configured"
}

# Setup repository secrets
setup_secrets() {
    info "🔐 Setting up repository secrets..."

    # List of secrets that need to be configured
    local secrets=(
        "GOOGLE_APPLICATION_CREDENTIALS_JSON"
        "CODECOV_TOKEN"
        "SEMGREP_APP_TOKEN"
        "GITGUARDIAN_API_KEY"
    )

    info "The following secrets need to be configured manually in GitHub:"
    for secret in "${secrets[@]}"; do
        info "  - ${secret}"
    done

    info "Configure secrets at: https://github.com/${REPO_OWNER}/${REPO_NAME}/settings/secrets/actions"

    # Create example environment files for local development
    cat > "${PROJECT_ROOT}/.env.example" << EOF
# Example environment configuration
# Copy this to .env and fill in your values

# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
GOOGLE_CLOUD_PROJECT=your-project-id

# Development Settings
DEBUG=true
LOG_LEVEL=INFO

# Optional: Codecov token for coverage reporting
CODECOV_TOKEN=your-codecov-token

# Optional: Security scanning tokens
SEMGREP_APP_TOKEN=your-semgrep-token
GITGUARDIAN_API_KEY=your-gitguardian-key
EOF

    success "✅ Environment template created: .env.example"
}

# Setup repository labels
setup_labels() {
    info "🏷️ Setting up repository labels..."

    # Define labels with colors
    declare -A labels=(
        ["bug"]="d73a4a"
        ["enhancement"]="a2eeef"
        ["documentation"]="0075ca"
        ["security"]="b60205"
        ["performance"]="fbca04"
        ["testing"]="0052cc"
        ["ci/cd"]="1d76db"
        ["dependencies"]="0366d6"
        ["rust"]="dea584"
        ["python"]="306998"
        ["good first issue"]="7057ff"
        ["help wanted"]="008672"
        ["priority:high"]="b60205"
        ["priority:medium"]="fbca04"
        ["priority:low"]="0e8a16"
    )

    for label in "${!labels[@]}"; do
        gh api repos/${REPO_OWNER}/${REPO_NAME}/labels \
            --method POST \
            --field name="${label}" \
            --field color="${labels[$label]}" \
            2>/dev/null || echo "Label '${label}' may already exist"
    done

    success "✅ Repository labels configured"
}

# Setup issue templates
setup_issue_templates() {
    info "📋 Setting up issue templates..."

    mkdir -p "${PROJECT_ROOT}/.github/ISSUE_TEMPLATE"

    # Bug report template
    cat > "${PROJECT_ROOT}/.github/ISSUE_TEMPLATE/bug_report.md" << 'EOF'
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: ['bug']
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Screenshots
If applicable, add screenshots to help explain your problem.

## Environment
- OS: [e.g. Ubuntu 22.04]
- Python Version: [e.g. 3.11]
- Package Version: [e.g. 1.0.0]

## Additional Context
Add any other context about the problem here.

## Logs
If applicable, include relevant log output:
```
paste logs here
```
EOF

    # Feature request template
    cat > "${PROJECT_ROOT}/.github/ISSUE_TEMPLATE/feature_request.md" << 'EOF'
---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: ['enhancement']
assignees: ''
---

## Is your feature request related to a problem?
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

## Describe the solution you'd like
A clear and concise description of what you want to happen.

## Describe alternatives you've considered
A clear and concise description of any alternative solutions or features you've considered.

## Additional context
Add any other context or screenshots about the feature request here.

## Implementation Notes
If you have ideas about how this could be implemented, please describe them here.
EOF

    # Security issue template
    cat > "${PROJECT_ROOT}/.github/ISSUE_TEMPLATE/security_issue.md" << 'EOF'
---
name: Security Issue
about: Report a security vulnerability
title: '[SECURITY] '
labels: ['security']
assignees: ''
---

## ⚠️ Security Issue

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security issues by:
1. Emailing us directly at security@example.com
2. Using GitHub's private security reporting feature

## What happens next?
1. We'll acknowledge receipt within 48 hours
2. We'll investigate and provide updates
3. We'll coordinate disclosure if confirmed

Thank you for helping keep our project secure!
EOF

    success "✅ Issue templates created"
}

# Setup pull request template
setup_pr_template() {
    info "📝 Setting up pull request template..."

    mkdir -p "${PROJECT_ROOT}/.github"

    cat > "${PROJECT_ROOT}/.github/pull_request_template.md" << 'EOF'
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] All tests pass locally

## Security
- [ ] Security implications considered
- [ ] No sensitive data exposed
- [ ] Authentication/authorization reviewed
- [ ] Input validation implemented

## Performance
- [ ] Performance impact assessed
- [ ] No significant performance regression
- [ ] Performance improvements documented

## Documentation
- [ ] Code comments updated
- [ ] README updated (if needed)
- [ ] API documentation updated (if applicable)
- [ ] Deployment notes updated (if needed)

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Related Issues
Closes #(issue number)

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Additional Notes
Any additional information that reviewers should know.
EOF

    success "✅ Pull request template created"
}

# Setup contributing guidelines
setup_contributing() {
    info "📖 Setting up contributing guidelines..."

    cat > "${PROJECT_ROOT}/CONTRIBUTING.md" << 'EOF'
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
EOF

    success "✅ Contributing guidelines created"
}

# Main setup function
main() {
    info "🚀 Setting up gterminal-unified-agents repository configuration..."

    # Check prerequisites
    if ! check_gh_cli; then
        error "GitHub CLI setup required"
        exit 1
    fi

    # Run setup functions
    setup_branch_protection
    configure_repository
    setup_secrets
    setup_labels
    setup_issue_templates
    setup_pr_template
    setup_contributing

    # Commit new files
    cd "$PROJECT_ROOT"
    if [[ -n "$(git status --porcelain)" ]]; then
        info "📝 Committing repository configuration files..."
        git add .
        git commit -m "feat: add comprehensive repository configuration

- Add GitHub Actions workflows for CI/CD and security
- Configure branch protection and repository settings
- Add issue and PR templates
- Create contributing guidelines
- Setup deployment scripts and automation

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
        git push origin main
        success "✅ Repository configuration committed and pushed"
    fi

    success "🎉 Repository setup completed!"
    info "Next steps:"
    info "1. Configure secrets at: https://github.com/${REPO_OWNER}/${REPO_NAME}/settings/secrets/actions"
    info "2. Review branch protection at: https://github.com/${REPO_OWNER}/${REPO_NAME}/settings/branches"
    info "3. Test workflows by creating a pull request"
    info "4. Set up monitoring and alerting"
}

# Show usage if help requested
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    cat << EOF
Usage: $0 [OPTIONS]

Setup gterminal-unified-agents repository for production deployment

Options:
    -h, --help     Show this help message

This script will:
- Configure branch protection rules
- Setup repository settings and security
- Create issue and PR templates
- Add contributing guidelines
- Commit and push changes

Prerequisites:
- GitHub CLI (gh) installed and authenticated
- Git repository with origin remote configured

EOF
    exit 0
fi

# Run main setup
main "$@"
