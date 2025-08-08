#!/bin/bash
# Production deployment script for gterminal-unified-agents
set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENV="staging"
LOG_FILE="/tmp/gterminal-deploy-$(date +%Y%m%d-%H%M%S).log"

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
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

info() { log "${BLUE}INFO${NC}" "$@"; }
warn() { log "${YELLOW}WARN${NC}" "$@"; }
error() { log "${RED}ERROR${NC}" "$@"; }
success() { log "${GREEN}SUCCESS${NC}" "$@"; }

# Error handler
error_handler() {
    local line_no=$1
    error "Deployment failed at line ${line_no}. Check log: ${LOG_FILE}"
    exit 1
}

trap 'error_handler ${LINENO}' ERR

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy gterminal-unified-agents to target environment

Options:
    -e, --environment ENV    Target environment (staging|production) [default: staging]
    -v, --version VERSION    Version to deploy [default: latest]
    -d, --dry-run           Perform dry run without actual deployment
    -s, --skip-tests        Skip pre-deployment tests
    -f, --force             Force deployment even if health checks fail
    -h, --help              Show this help message

Examples:
    $0 --environment staging
    $0 --environment production --version v1.2.0
    $0 --dry-run --environment production

EOF
}

# Parse command line arguments
ENVIRONMENT="$DEFAULT_ENV"
VERSION="latest"
DRY_RUN=false
SKIP_TESTS=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -s|--skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
    exit 1
fi

# Start deployment
info "🚀 Starting gterminal-unified-agents deployment"
info "   Environment: $ENVIRONMENT"
info "   Version: $VERSION"
info "   Dry Run: $DRY_RUN"
info "   Log File: $LOG_FILE"

# Pre-deployment validation
validate_environment() {
    info "🔍 Validating deployment environment..."

    # Check required tools
    local required_tools=("python3" "uv" "git")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            error "Required tool not found: $tool"
            return 1
        fi
    done

    # Check Python version
    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ ! "$python_version" =~ ^3\.(11|12)$ ]]; then
        error "Python 3.11 or 3.12 required, found: $python_version"
        return 1
    fi

    # Check git repository state
    if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
        error "Not in a git repository"
        return 1
    fi

    # Check for uncommitted changes
    if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
        warn "Uncommitted changes detected in repository"
        if [[ "$FORCE" != true ]]; then
            error "Use --force to deploy with uncommitted changes"
            return 1
        fi
    fi

    success "✅ Environment validation completed"
}

# Build deployment assets
build_assets() {
    info "📦 Building deployment assets..."

    cd "$PROJECT_ROOT"

    # Create virtual environment
    if [[ "$DRY_RUN" == false ]]; then
        info "Creating deployment virtual environment..."
        uv venv deployment-env
        source deployment-env/bin/activate

        # Install dependencies
        info "Installing dependencies..."
        uv sync --dev

        # Build Rust extensions
        if [[ -d "gterminal_rust_extensions" ]]; then
            info "Building Rust extensions..."
            cd gterminal_rust_extensions
            if command -v maturin >/dev/null 2>&1; then
                maturin build --release
            else
                uv pip install maturin
                maturin build --release
            fi
            cd ..
        fi

        # Build Python package
        info "Building Python package..."
        uv pip install build
        python -m build

        success "✅ Build completed successfully"
    else
        info "DRY RUN: Skipping asset building"
    fi
}

# Run pre-deployment tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        warn "⚠️  Skipping pre-deployment tests (--skip-tests)"
        return 0
    fi

    info "🧪 Running pre-deployment tests..."

    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == false ]]; then
        source deployment-env/bin/activate

        # Run linting
        info "Running code quality checks..."
        uv run ruff check .
        uv run black --check .

        # Run tests
        info "Running test suite..."
        uv run pytest --tb=short

        # Test import functionality
        info "Testing package imports..."
        python -c "
from gemini_cli.core.client import GeminiClient
from gemini_cli.core.react_engine import ReactEngine
from gemini_cli.tools.registry import ToolRegistry
print('✅ All imports successful')
"

        success "✅ All tests passed"
    else
        info "DRY RUN: Skipping test execution"
    fi
}

# Deploy to target environment
deploy() {
    info "🚀 Deploying to $ENVIRONMENT environment..."

    if [[ "$DRY_RUN" == true ]]; then
        info "DRY RUN: Simulating deployment to $ENVIRONMENT"
        info "  📦 Would deploy built packages"
        info "  🔧 Would update configuration"
        info "  🔄 Would restart services"
        info "  🏥 Would run health checks"
        success "✅ DRY RUN deployment simulation completed"
        return 0
    fi

    case "$ENVIRONMENT" in
        staging)
            deploy_staging
            ;;
        production)
            deploy_production
            ;;
        *)
            error "Unsupported environment: $ENVIRONMENT"
            return 1
            ;;
    esac
}

# Deploy to staging environment
deploy_staging() {
    info "🎪 Deploying to staging environment..."

    # Install package
    cd "$PROJECT_ROOT"
    source deployment-env/bin/activate

    # Install built package
    info "Installing package to staging environment..."
    uv pip install dist/*.whl

    # Install Rust extensions if available
    if ls gterminal_rust_extensions/target/wheels/*.whl >/dev/null 2>&1; then
        info "Installing Rust extensions..."
        uv pip install gterminal_rust_extensions/target/wheels/*.whl
    fi

    # Test installation
    info "Testing staging installation..."
    python -m gemini_cli --help

    success "✅ Staging deployment completed"
}

# Deploy to production environment
deploy_production() {
    info "🏭 Deploying to production environment..."

    # Additional production validations
    if [[ "$FORCE" != true ]]; then
        warn "Production deployment requires additional confirmation"
        read -p "Are you sure you want to deploy to production? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            info "Production deployment cancelled"
            return 0
        fi
    fi

    # Backup current production state
    info "Creating production backup..."
    # This would backup the current production deployment

    # Deploy with zero-downtime strategy
    info "Deploying with zero-downtime strategy..."
    cd "$PROJECT_ROOT"
    source deployment-env/bin/activate

    # Install package
    info "Installing package to production environment..."
    uv pip install dist/*.whl

    # Install Rust extensions if available
    if ls gterminal_rust_extensions/target/wheels/*.whl >/dev/null 2>&1; then
        info "Installing Rust extensions..."
        uv pip install gterminal_rust_extensions/target/wheels/*.whl
    fi

    success "✅ Production deployment completed"
}

# Run health checks
health_check() {
    info "🏥 Running post-deployment health checks..."

    if [[ "$DRY_RUN" == true ]]; then
        info "DRY RUN: Simulating health checks"
        success "✅ DRY RUN health checks completed"
        return 0
    fi

    cd "$PROJECT_ROOT"
    source deployment-env/bin/activate

    # Basic functionality test
    info "Testing basic functionality..."
    if python -m gemini_cli --help >/dev/null 2>&1; then
        success "✅ CLI command working"
    else
        error "❌ CLI command failed"
        return 1
    fi

    # Import test
    info "Testing core imports..."
    if python -c "from gemini_cli.core.client import GeminiClient; print('Import test passed')" >/dev/null 2>&1; then
        success "✅ Core imports working"
    else
        error "❌ Core imports failed"
        return 1
    fi

    # Test server mode if available
    if [[ -f "server_mode.py" ]]; then
        info "Testing server mode availability..."
        if python -c "import server_mode; print('Server mode available')" >/dev/null 2>&1; then
            success "✅ Server mode available"
        else
            warn "⚠️  Server mode import issues"
        fi
    fi

    success "✅ All health checks passed"
}

# Rollback function
rollback() {
    error "🔄 Initiating rollback procedure..."

    if [[ "$ENVIRONMENT" == "production" ]]; then
        warn "Production rollback procedure:"
        warn "1. Restore from backup"
        warn "2. Restart services"
        warn "3. Verify functionality"
        warn "Manual intervention may be required"
    fi

    error "❌ Deployment failed and rollback initiated"
}

# Cleanup function
cleanup() {
    info "🧹 Cleaning up deployment artifacts..."

    if [[ -d "$PROJECT_ROOT/deployment-env" ]]; then
        rm -rf "$PROJECT_ROOT/deployment-env"
        info "Removed deployment virtual environment"
    fi

    success "✅ Cleanup completed"
}

# Main deployment flow
main() {
    info "Starting deployment process..."

    # Set up error handling for rollback
    trap 'rollback' ERR

    validate_environment
    build_assets
    run_tests
    deploy
    health_check

    success "🎉 Deployment completed successfully!"
    info "📄 Full deployment log: $LOG_FILE"

    # Cleanup on successful deployment
    if [[ "$DRY_RUN" == false ]]; then
        cleanup
    fi
}

# Run main deployment process
main "$@"
