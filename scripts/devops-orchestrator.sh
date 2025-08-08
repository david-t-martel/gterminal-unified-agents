#!/bin/bash
# DevOps Orchestrator - Master control script for GTerminal development workflow
# Integrates all Rust and Python tools into seamless development experience

set -euo pipefail

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="docker-compose.production.yml"
ORCHESTRATOR_SERVICE="orchestration/service-orchestrator.py"

# Default values
COMMAND=""
SERVICE=""
ENVIRONMENT="development"
VERBOSE=false
DRY_RUN=false
FORCE=false

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $*${NC}" >&2
}

error() {
    echo -e "${RED}[ERROR] $*${NC}" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS] $*${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[WARN] $*${NC}" >&2
}

info() {
    echo -e "${CYAN}[INFO] $*${NC}" >&2
}

highlight() {
    echo -e "${BOLD}${PURPLE}$*${NC}"
}

# Utility functions
check_dependencies() {
    local deps=(docker docker-compose python3 uv cargo)
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        error "Missing dependencies: ${missing[*]}"
        error "Please install all required dependencies before proceeding."
        exit 1
    fi
}

check_project_structure() {
    local required_dirs=("rust-filewatcher" "orchestration" "monitoring" "scripts")
    local required_files=("$COMPOSE_FILE" "$ORCHESTRATOR_SERVICE" "pyproject.toml")

    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$PROJECT_ROOT/$dir" ]; then
            error "Required directory not found: $dir"
            exit 1
        fi
    done

    for file in "${required_files[@]}"; do
        if [ ! -f "$PROJECT_ROOT/$file" ]; then
            error "Required file not found: $file"
            exit 1
        fi
    done
}

# Environment management
setup_environment() {
    local env_file="$PROJECT_ROOT/.env.${ENVIRONMENT}"

    if [ -f "$env_file" ]; then
        log "Loading environment configuration from $env_file"
        set -a
        # shellcheck source=/dev/null
        source "$env_file"
        set +a
    else
        warn "Environment file not found: $env_file, using defaults"
        create_default_env_file "$env_file"
    fi
}

create_default_env_file() {
    local env_file="$1"

    cat > "$env_file" << 'EOF'
# GTerminal DevOps Environment Configuration

# Core Configuration
LOG_LEVEL=INFO
ENVIRONMENT=development
DEBUG=true

# Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_GENAI_USE_VERTEXAI=True
GCP_PROFILE=business

# Database
POSTGRES_DB=gterminal
POSTGRES_USER=gterminal
POSTGRES_PASSWORD=secure_password

# Redis
REDIS_PASSWORD=

# Monitoring
GRAFANA_PASSWORD=admin
PROMETHEUS_RETENTION=30d

# Performance
RUST_LOG=info
FILEWATCHER_BATCH_SIZE=100
FILEWATCHER_DEBOUNCE_MS=50
LSP_MAX_CONCURRENT=10
DASHBOARD_UPDATE_INTERVAL=5

# Security (set these in production)
JWT_SECRET_KEY=change-me-in-production
CLAUDE_API_KEY=
SLACK_WEBHOOK_URL=
SMTP_PASSWORD=

# Development
HOT_RELOAD=true
DEVELOPMENT_MODE=true
EOF

    info "Created default environment file: $env_file"
    warn "Please review and update the configuration, especially secrets!"
}

# Service management functions
build_services() {
    local services=("$@")

    log "Building services: ${services[*]:-all}"

    if [ ${#services[@]} -eq 0 ]; then
        # Build all services
        if $DRY_RUN; then
            info "[DRY RUN] Would run: docker-compose -f $COMPOSE_FILE build"
        else
            docker-compose -f "$COMPOSE_FILE" build
        fi
    else
        # Build specific services
        if $DRY_RUN; then
            info "[DRY RUN] Would run: docker-compose -f $COMPOSE_FILE build ${services[*]}"
        else
            docker-compose -f "$COMPOSE_FILE" build "${services[@]}"
        fi
    fi

    success "Build completed"
}

start_services() {
    local services=("$@")

    log "Starting services: ${services[*]:-all}"

    # Ensure environment is set up
    setup_environment

    if [ ${#services[@]} -eq 0 ]; then
        if $DRY_RUN; then
            info "[DRY RUN] Would run: docker-compose -f $COMPOSE_FILE up -d"
        else
            docker-compose -f "$COMPOSE_FILE" up -d
            wait_for_services_health
        fi
    else
        if $DRY_RUN; then
            info "[DRY RUN] Would run: docker-compose -f $COMPOSE_FILE up -d ${services[*]}"
        else
            docker-compose -f "$COMPOSE_FILE" up -d "${services[@]}"
            wait_for_services_health "${services[@]}"
        fi
    fi

    success "Services started successfully"
    show_service_urls
}

stop_services() {
    local services=("$@")

    log "Stopping services: ${services[*]:-all}"

    if [ ${#services[@]} -eq 0 ]; then
        if $DRY_RUN; then
            info "[DRY RUN] Would run: docker-compose -f $COMPOSE_FILE down"
        else
            if $FORCE; then
                docker-compose -f "$COMPOSE_FILE" down --remove-orphans -v
            else
                docker-compose -f "$COMPOSE_FILE" down
            fi
        fi
    else
        if $DRY_RUN; then
            info "[DRY RUN] Would run: docker-compose -f $COMPOSE_FILE stop ${services[*]}"
        else
            docker-compose -f "$COMPOSE_FILE" stop "${services[@]}"
        fi
    fi

    success "Services stopped"
}

restart_services() {
    local services=("$@")

    log "Restarting services: ${services[*]:-all}"

    stop_services "${services[@]}"
    sleep 2
    start_services "${services[@]}"
}

wait_for_services_health() {
    local services=("$@")
    local timeout=300  # 5 minutes
    local elapsed=0
    local interval=5

    log "Waiting for services to become healthy (timeout: ${timeout}s)..."

    while [ $elapsed -lt $timeout ]; do
        local all_healthy=true
        local service_status=()

        # Check specific services or all services
        if [ ${#services[@]} -eq 0 ]; then
            # Get all running services
            mapfile -t services < <(docker-compose -f "$COMPOSE_FILE" ps --services)
        fi

        for service in "${services[@]}"; do
            local health_status
            health_status=$(docker-compose -f "$COMPOSE_FILE" ps -q "$service" | xargs -I {} docker inspect {} --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")

            if [ "$health_status" != "healthy" ] && [ "$health_status" != "unknown" ]; then
                all_healthy=false
                service_status+=("$service: $health_status")
            else
                service_status+=("$service: ✓")
            fi
        done

        if $all_healthy; then
            success "All services are healthy!"
            return 0
        fi

        if $VERBOSE; then
            info "Service status: ${service_status[*]}"
        else
            echo -n "."
        fi

        sleep $interval
        elapsed=$((elapsed + interval))
    done

    warn "Timeout waiting for services to become healthy"
    show_service_status
    return 1
}

show_service_status() {
    log "Current service status:"
    docker-compose -f "$COMPOSE_FILE" ps
}

show_service_urls() {
    cat << 'EOF'

🚀 Service URLs:
┌─────────────────────────────────────────────────────────┐
│  Service               │  URL                           │
├─────────────────────────────────────────────────────────┤
│  GTerminal App         │  http://localhost:8080         │
│  Development Dashboard │  http://localhost:8769         │
│  Ruff LSP Server       │  http://localhost:8768         │
│  Filewatcher Metrics   │  http://localhost:8766         │
│  Prometheus            │  http://localhost:9090         │
│  Grafana               │  http://localhost:3003         │
│  Jaeger Tracing        │  http://localhost:16686        │
└─────────────────────────────────────────────────────────┘

EOF
}

# Development workflow functions
dev_setup() {
    log "Setting up development environment..."

    # Check dependencies
    check_dependencies
    check_project_structure

    # Setup Python environment
    log "Setting up Python environment..."
    if command -v uv &> /dev/null; then
        uv sync
    else
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -e .
    fi

    # Build Rust components
    log "Building Rust filewatcher..."
    (cd rust-filewatcher && cargo build --release)

    # Setup monitoring
    log "Preparing monitoring configuration..."
    if [ ! -d "monitoring/provisioning" ]; then
        mkdir -p monitoring/provisioning/{dashboards,datasources}
    fi

    # Copy dashboard configurations
    if [ ! -f "monitoring/provisioning/dashboards/dashboard.yml" ]; then
        cat > "monitoring/provisioning/dashboards/dashboard.yml" << 'EOF'
apiVersion: 1
providers:
  - name: 'default'
    orgId: 1
    folder: 'GTerminal'
    type: file
    disableDeletion: false
    options:
      path: /var/lib/grafana/dashboards
EOF
    fi

    # Setup datasources
    if [ ! -f "monitoring/provisioning/datasources/prometheus.yml" ]; then
        cat > "monitoring/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
EOF
    fi

    success "Development environment setup completed!"
    info "Run 'devops-orchestrator.sh start' to launch all services"
}

run_tests() {
    local test_type="${1:-all}"

    log "Running tests: $test_type"

    case "$test_type" in
        "rust")
            (cd rust-filewatcher && cargo test --verbose)
            ;;
        "python")
            if command -v uv &> /dev/null; then
                uv run pytest tests/ -v
            else
                python -m pytest tests/ -v
            fi
            ;;
        "integration")
            # Start services for integration tests
            start_services
            sleep 10  # Wait for startup

            if command -v uv &> /dev/null; then
                uv run pytest tests/integration/ --integration -v
            else
                python -m pytest tests/integration/ --integration -v
            fi
            ;;
        "all"|*)
            run_tests rust
            run_tests python
            run_tests integration
            ;;
    esac

    success "Tests completed: $test_type"
}

show_logs() {
    local service="$1"
    local follow="${2:-false}"

    if [ -z "$service" ]; then
        log "Showing logs for all services"
        docker-compose -f "$COMPOSE_FILE" logs --tail=100
    else
        log "Showing logs for service: $service"
        if [ "$follow" = "true" ]; then
            docker-compose -f "$COMPOSE_FILE" logs -f "$service"
        else
            docker-compose -f "$COMPOSE_FILE" logs --tail=100 "$service"
        fi
    fi
}

run_benchmarks() {
    log "Running performance benchmarks..."

    # Ensure services are running
    if ! docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        warn "Services not running, starting them first..."
        start_services
    fi

    # Run Rust benchmarks
    log "Running Rust filewatcher benchmarks..."
    (cd rust-filewatcher && cargo bench --features benchmarks)

    # Run load tests
    log "Running load tests..."
    if command -v wrk &> /dev/null; then
        # Test filewatcher metrics endpoint
        wrk -t4 -c100 -d30s --timeout 30s http://localhost:8766/health

        # Test LSP server
        wrk -t4 -c50 -d30s --timeout 30s http://localhost:8768/health

        # Test main application
        wrk -t4 -c20 -d30s --timeout 30s http://localhost:8080/health
    else
        warn "wrk not installed, skipping load tests"
    fi

    success "Benchmarks completed"
}

cleanup() {
    log "Cleaning up development environment..."

    # Stop services
    stop_services

    # Clean Docker resources
    if $FORCE; then
        docker system prune -f
        docker volume prune -f
    fi

    # Clean Rust build artifacts
    (cd rust-filewatcher && cargo clean)

    # Clean Python cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true

    success "Cleanup completed"
}

# Advanced operations
debug_service() {
    local service="$1"

    if [ -z "$service" ]; then
        error "Service name required for debugging"
        return 1
    fi

    log "Debugging service: $service"

    # Show service details
    docker-compose -f "$COMPOSE_FILE" ps "$service"

    # Show recent logs
    echo -e "\n${BOLD}Recent logs:${NC}"
    docker-compose -f "$COMPOSE_FILE" logs --tail=50 "$service"

    # Show resource usage
    echo -e "\n${BOLD}Resource usage:${NC}"
    docker stats --no-stream "$(docker-compose -f "$COMPOSE_FILE" ps -q "$service")"

    # Show health check status
    echo -e "\n${BOLD}Health check status:${NC}"
    local container_id
    container_id=$(docker-compose -f "$COMPOSE_FILE" ps -q "$service")
    docker inspect "$container_id" --format='{{.State.Health.Status}}' 2>/dev/null || echo "No health check"
}

performance_profile() {
    local service="$1"
    local duration="${2:-60}"

    log "Profiling service '$service' for ${duration} seconds..."

    local container_id
    container_id=$(docker-compose -f "$COMPOSE_FILE" ps -q "$service")

    if [ -z "$container_id" ]; then
        error "Service '$service' not running"
        return 1
    fi

    # Collect performance metrics
    docker stats --no-stream "$container_id" > "profile_${service}_$(date +%s).log"

    success "Performance profile saved for $service"
}

# Usage information
usage() {
    cat << 'EOF'
🚀 GTerminal DevOps Orchestrator

USAGE:
    devops-orchestrator.sh <command> [options] [arguments]

COMMANDS:
    setup                   Setup development environment
    start [services...]     Start services (all or specific)
    stop [services...]      Stop services (all or specific)
    restart [services...]   Restart services
    build [services...]     Build services
    status                  Show service status
    logs [service] [--follow] Show service logs

    test [type]            Run tests (rust|python|integration|all)
    benchmark              Run performance benchmarks
    debug <service>        Debug a specific service
    profile <service> [duration] Profile service performance

    cleanup [--force]      Clean up environment
    urls                   Show service URLs

OPTIONS:
    --environment, -e ENV   Environment (development|staging|production)
    --verbose, -v          Enable verbose output
    --dry-run, -n          Show what would be done without executing
    --force, -f            Force operations (use with caution)
    --help, -h             Show this help message

EXAMPLES:
    devops-orchestrator.sh setup
    devops-orchestrator.sh start rust-filewatcher ruff-lsp
    devops-orchestrator.sh test integration
    devops-orchestrator.sh logs gterminal-app --follow
    devops-orchestrator.sh debug rust-filewatcher
    devops-orchestrator.sh benchmark
    devops-orchestrator.sh cleanup --force

ENVIRONMENT MANAGEMENT:
    Configuration is loaded from .env.<environment> files.
    Default environment is 'development'.

    Available environments: development, staging, production

SERVICE GROUPS:
    core:        rust-filewatcher, ruff-lsp, gterminal-app
    monitoring:  prometheus, grafana, jaeger
    infra:       redis, postgres, nginx
    dev:         development-dashboard

For more information, see: https://wiki.gterminal.dev/devops
EOF
}

# Argument parsing
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
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
            -*)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                if [ -z "$COMMAND" ]; then
                    COMMAND="$1"
                else
                    break
                fi
                shift
                ;;
        esac
    done

    # Remaining arguments are passed to the command
    ARGS=("$@")
}

# Main execution
main() {
    # Parse command line arguments
    parse_args "$@"

    # Change to project root
    cd "$PROJECT_ROOT"

    # Enable verbose logging if requested
    if $VERBOSE; then
        set -x
    fi

    # Execute command
    case "$COMMAND" in
        "setup")
            dev_setup
            ;;
        "start")
            start_services "${ARGS[@]}"
            ;;
        "stop")
            stop_services "${ARGS[@]}"
            ;;
        "restart")
            restart_services "${ARGS[@]}"
            ;;
        "build")
            build_services "${ARGS[@]}"
            ;;
        "status")
            show_service_status
            ;;
        "logs")
            show_logs "${ARGS[0]:-}" "${ARGS[1]:-false}"
            ;;
        "test")
            run_tests "${ARGS[0]:-all}"
            ;;
        "benchmark")
            run_benchmarks
            ;;
        "debug")
            debug_service "${ARGS[0]:-}"
            ;;
        "profile")
            performance_profile "${ARGS[0]:-}" "${ARGS[1]:-60}"
            ;;
        "cleanup")
            cleanup
            ;;
        "urls")
            show_service_urls
            ;;
        "")
            error "No command specified"
            usage
            exit 1
            ;;
        *)
            error "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac
}

# Error handling
trap 'error "Script failed at line $LINENO"' ERR

# Run main function
main "$@"
