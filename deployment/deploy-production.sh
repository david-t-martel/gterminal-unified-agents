#!/bin/bash
# Production Deployment Script for GTerminal DevOps Stack
# Comprehensive deployment orchestration for AWS/Kubernetes

set -euo pipefail

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOYMENT_DIR="$SCRIPT_DIR"

# Default values
ENVIRONMENT="production"
AWS_REGION="us-west-2"
CLUSTER_NAME="gterminal-production"
NAMESPACE="gterminal"
DOCKER_REGISTRY="ghcr.io/gterminal"
VERSION_TAG="${GITHUB_SHA:-latest}"
DRY_RUN=false
SKIP_TERRAFORM=false
SKIP_BUILD=false
FORCE_REDEPLOY=false
ROLLBACK=false

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
    local deps=(aws kubectl helm terraform docker)
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        error "Missing dependencies: ${missing[*]}"
        error "Please install all required dependencies before proceeding."
        cat << 'EOF'

Required dependencies:
- aws: AWS CLI v2
- kubectl: Kubernetes CLI
- helm: Helm package manager
- terraform: Infrastructure as code
- docker: Container runtime

Installation links:
- AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
- kubectl: https://kubernetes.io/docs/tasks/tools/
- helm: https://helm.sh/docs/intro/install/
- terraform: https://learn.hashicorp.com/tutorials/terraform/install-cli
- docker: https://docs.docker.com/get-docker/

EOF
        exit 1
    fi
}

check_aws_authentication() {
    log "Checking AWS authentication..."

    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS authentication failed"
        error "Please configure AWS credentials using 'aws configure' or environment variables"
        exit 1
    fi

    local account_id
    account_id=$(aws sts get-caller-identity --query Account --output text)
    local current_region
    current_region=$(aws configure get region || echo "us-west-2")

    success "AWS authenticated - Account: $account_id, Region: $current_region"
}

# Infrastructure deployment
deploy_infrastructure() {
    if $SKIP_TERRAFORM; then
        warn "Skipping Terraform infrastructure deployment"
        return 0
    fi

    log "Deploying infrastructure with Terraform..."

    cd "$DEPLOYMENT_DIR"

    # Initialize Terraform
    if [ ! -d ".terraform" ]; then
        log "Initializing Terraform..."
        terraform init
    fi

    # Plan infrastructure changes
    log "Planning infrastructure changes..."
    terraform plan \
        -var="aws_region=$AWS_REGION" \
        -var="cluster_name=gterminal" \
        -var="environment=$ENVIRONMENT" \
        -out=tfplan

    if $DRY_RUN; then
        info "[DRY RUN] Would apply Terraform plan"
        return 0
    fi

    # Apply infrastructure changes
    log "Applying infrastructure changes..."
    terraform apply tfplan

    # Get outputs
    local cluster_endpoint
    cluster_endpoint=$(terraform output -raw cluster_endpoint)
    local rds_endpoint
    rds_endpoint=$(terraform output -raw rds_endpoint)
    local redis_endpoint
    redis_endpoint=$(terraform output -raw redis_endpoint)

    success "Infrastructure deployment completed"
    info "Cluster endpoint: $cluster_endpoint"
    info "Database endpoint: $rds_endpoint"
    info "Redis endpoint: $redis_endpoint"
}

configure_kubectl() {
    log "Configuring kubectl for EKS cluster..."

    # Update kubeconfig
    aws eks update-kubeconfig \
        --region "$AWS_REGION" \
        --name "$CLUSTER_NAME-$ENVIRONMENT"

    # Verify cluster access
    if kubectl cluster-info &> /dev/null; then
        success "kubectl configured successfully"
        kubectl get nodes
    else
        error "Failed to configure kubectl"
        exit 1
    fi
}

# Container build and push
build_and_push_images() {
    if $SKIP_BUILD; then
        warn "Skipping container image build"
        return 0
    fi

    log "Building and pushing container images..."

    cd "$PROJECT_ROOT"

    # Login to GitHub Container Registry
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        echo "$GITHUB_TOKEN" | docker login ghcr.io -u "${GITHUB_ACTOR:-$(whoami)}" --password-stdin
    else
        warn "GITHUB_TOKEN not set, assuming already logged in to registry"
    fi

    # Build and push main application image
    log "Building main application image..."
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --file Dockerfile.production \
        --target production \
        --tag "$DOCKER_REGISTRY/gterminal:$VERSION_TAG" \
        --tag "$DOCKER_REGISTRY/gterminal:latest" \
        --push .

    # Build and push Rust filewatcher image
    log "Building Rust filewatcher image..."
    (cd rust-filewatcher &&
     docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --tag "$DOCKER_REGISTRY/rust-filewatcher:$VERSION_TAG" \
        --tag "$DOCKER_REGISTRY/rust-filewatcher:latest" \
        --push .)

    # Build and push Ruff LSP image
    log "Building Ruff LSP image..."
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --file Dockerfile.ruff-lsp \
        --tag "$DOCKER_REGISTRY/ruff-lsp:$VERSION_TAG" \
        --tag "$DOCKER_REGISTRY/ruff-lsp:latest" \
        --push .

    # Build and push development dashboard image
    log "Building development dashboard image..."
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --file Dockerfile.dashboard \
        --tag "$DOCKER_REGISTRY/dashboard:$VERSION_TAG" \
        --tag "$DOCKER_REGISTRY/dashboard:latest" \
        --push .

    success "Container images built and pushed successfully"
}

# Kubernetes deployment
deploy_kubernetes_resources() {
    log "Deploying Kubernetes resources..."

    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi

    # Label namespace for monitoring
    kubectl label namespace "$NAMESPACE" \
        monitoring.coreos.com/enabled=true \
        --overwrite

    # Apply Kubernetes manifests
    log "Applying Kubernetes manifests..."
    if $DRY_RUN; then
        info "[DRY RUN] Would apply Kubernetes manifests"
        kubectl apply --dry-run=client -f "$DEPLOYMENT_DIR/kubernetes-production.yml"
    else
        # Update image tags in manifests
        sed -i.bak "s|:latest|:$VERSION_TAG|g" "$DEPLOYMENT_DIR/kubernetes-production.yml"

        kubectl apply -f "$DEPLOYMENT_DIR/kubernetes-production.yml"

        # Restore original manifest
        mv "$DEPLOYMENT_DIR/kubernetes-production.yml.bak" "$DEPLOYMENT_DIR/kubernetes-production.yml"
    fi

    success "Kubernetes resources deployed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    local deployments=(
        "postgres"
        "redis"
        "rust-filewatcher"
        "ruff-lsp"
        "development-dashboard"
        "gterminal-app"
    )

    log "Waiting for deployments to be ready..."

    for deployment in "${deployments[@]}"; do
        info "Waiting for deployment: $deployment"
        kubectl rollout status deployment/"$deployment" \
            --namespace="$NAMESPACE" \
            --timeout=300s
    done

    success "All deployments are ready"
}

# Health checks
run_health_checks() {
    log "Running health checks..."

    local services=(
        "rust-filewatcher:8766:/health"
        "ruff-lsp:8768:/health"
        "development-dashboard:8080:/health"
        "gterminal-app:8000:/health"
    )

    for service in "${services[@]}"; do
        IFS=':' read -r service_name port path <<< "$service"

        info "Checking health of $service_name"

        # Port forward to check health
        kubectl port-forward \
            --namespace="$NAMESPACE" \
            "service/$service_name" \
            "$port:$port" &
        local port_forward_pid=$!

        sleep 5  # Wait for port forward to establish

        local health_status=false
        for i in {1..10}; do
            if curl -sf "http://localhost:$port$path" > /dev/null 2>&1; then
                health_status=true
                break
            fi
            sleep 2
        done

        # Clean up port forward
        kill $port_forward_pid 2>/dev/null || true

        if $health_status; then
            success "$service_name is healthy"
        else
            error "$service_name health check failed"
            return 1
        fi
    done

    success "All health checks passed"
}

# Monitoring setup
setup_monitoring() {
    log "Setting up monitoring stack..."

    # Add Prometheus Helm repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update

    # Install Prometheus
    log "Installing Prometheus..."
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
        --set grafana.adminPassword=admin123 \
        --values - << 'EOF'
prometheus:
  prometheusSpec:
    additionalScrapeConfigs:
      - job_name: 'rust-filewatcher'
        static_configs:
          - targets: ['rust-filewatcher.gterminal:8766']
      - job_name: 'ruff-lsp'
        static_configs:
          - targets: ['ruff-lsp.gterminal:8768']
      - job_name: 'gterminal-app'
        static_configs:
          - targets: ['gterminal-app.gterminal:8000']
      - job_name: 'development-dashboard'
        static_configs:
          - targets: ['development-dashboard.gterminal:8080']

grafana:
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
      - name: 'gterminal'
        orgId: 1
        folder: 'GTerminal'
        type: file
        disableDeletion: false
        options:
          path: /var/lib/grafana/dashboards/gterminal

  dashboards:
    gterminal:
      gterminal-overview:
        json: |
EOF

    # Install AlertManager
    log "Installing AlertManager..."
    kubectl apply -f "$PROJECT_ROOT/monitoring/alertmanager.yml"

    success "Monitoring stack installed"
}

# Database migration
run_database_migration() {
    log "Running database migrations..."

    # Create a migration job
    kubectl apply -f - << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: gterminal-migration-$(date +%s)
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: migration
        image: $DOCKER_REGISTRY/gterminal:$VERSION_TAG
        command: ["python", "-m", "gterminal.db", "migrate"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: gterminal-secrets
              key: DATABASE_URL
      restartPolicy: Never
  backoffLimit: 3
EOF

    # Wait for migration to complete
    local job_name
    job_name=$(kubectl get jobs -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp -o name | tail -n1)

    kubectl wait --for=condition=complete --timeout=300s "$job_name" -n "$NAMESPACE"

    success "Database migration completed"
}

# Rollback functionality
perform_rollback() {
    log "Performing rollback to previous version..."

    local deployments=(
        "rust-filewatcher"
        "ruff-lsp"
        "development-dashboard"
        "gterminal-app"
    )

    for deployment in "${deployments[@]}"; do
        info "Rolling back deployment: $deployment"
        kubectl rollout undo deployment/"$deployment" \
            --namespace="$NAMESPACE"
    done

    # Wait for rollback to complete
    wait_for_deployment

    success "Rollback completed successfully"
}

# Smoke tests
run_smoke_tests() {
    log "Running smoke tests..."

    # Get external IPs or LoadBalancer endpoints
    local app_endpoint
    app_endpoint=$(kubectl get service gterminal-app \
        --namespace="$NAMESPACE" \
        -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

    if [ -n "$app_endpoint" ]; then
        # Test main application
        if curl -sf "http://$app_endpoint/health" > /dev/null; then
            success "Main application smoke test passed"
        else
            error "Main application smoke test failed"
            return 1
        fi
    else
        warn "External endpoint not available, skipping external smoke tests"
    fi

    # Internal smoke tests
    kubectl run smoke-test \
        --namespace="$NAMESPACE" \
        --image=curlimages/curl:latest \
        --rm -i --tty \
        --restart=Never \
        -- sh -c "
            curl -sf http://gterminal-app:8000/health &&
            curl -sf http://rust-filewatcher:8766/health &&
            curl -sf http://ruff-lsp:8768/health &&
            curl -sf http://development-dashboard:8080/health
        "

    success "Smoke tests completed successfully"
}

# Cleanup
cleanup_old_resources() {
    log "Cleaning up old resources..."

    # Remove completed jobs older than 1 day
    kubectl delete jobs \
        --namespace="$NAMESPACE" \
        --field-selector=status.successful=1 \
        --ignore-not-found=true \
        $(kubectl get jobs \
            --namespace="$NAMESPACE" \
            --field-selector=status.successful=1 \
            -o jsonpath='{range .items[?(@.status.completionTime < "'$(date -d '1 day ago' -Iseconds)'")]}{.metadata.name}{" "}{end}')

    # Clean up old ReplicaSets
    kubectl delete replicasets \
        --namespace="$NAMESPACE" \
        --field-selector='status.replicas=0' \
        --ignore-not-found=true

    success "Cleanup completed"
}

# Status reporting
show_deployment_status() {
    highlight "🚀 Deployment Status Report"

    echo ""
    echo "📊 Cluster Information:"
    kubectl cluster-info

    echo ""
    echo "📦 Deployed Services:"
    kubectl get deployments,services,ingress \
        --namespace="$NAMESPACE" \
        -o wide

    echo ""
    echo "🔍 Pod Status:"
    kubectl get pods \
        --namespace="$NAMESPACE" \
        -o wide

    echo ""
    echo "📈 Resource Usage:"
    kubectl top nodes || true
    kubectl top pods --namespace="$NAMESPACE" || true

    echo ""
    echo "🌐 External Access:"
    kubectl get ingress \
        --namespace="$NAMESPACE" \
        -o custom-columns="NAME:.metadata.name,HOSTS:.spec.rules[*].host,PATHS:.spec.rules[*].http.paths[*].path"
}

# Usage
usage() {
    cat << 'EOF'
🚀 GTerminal Production Deployment Script

USAGE:
    deploy-production.sh [options]

OPTIONS:
    --environment ENV        Environment (default: production)
    --region REGION         AWS region (default: us-west-2)
    --cluster-name NAME     EKS cluster name (default: gterminal-production)
    --namespace NS          Kubernetes namespace (default: gterminal)
    --version TAG           Container image tag (default: latest)

    --dry-run              Show what would be done without executing
    --skip-terraform       Skip Terraform infrastructure deployment
    --skip-build          Skip container image build and push
    --force-redeploy      Force redeployment even if no changes
    --rollback            Rollback to previous version

    --help                Show this help message

EXAMPLES:
    # Full production deployment
    deploy-production.sh --environment production --version v1.2.3

    # Deploy to staging
    deploy-production.sh --environment staging --cluster-name gterminal-staging

    # Dry run
    deploy-production.sh --dry-run

    # Rollback
    deploy-production.sh --rollback

    # Skip infrastructure changes
    deploy-production.sh --skip-terraform --version v1.2.4

PREREQUISITES:
    - AWS CLI configured with appropriate permissions
    - Docker logged in to container registry
    - kubectl installed
    - helm installed
    - terraform installed (if not using --skip-terraform)

ENVIRONMENT VARIABLES:
    GITHUB_TOKEN           GitHub token for container registry
    GITHUB_ACTOR          GitHub username
    GITHUB_SHA            Git commit SHA (used as image tag)

For more information, see: https://wiki.gterminal.dev/deployment
EOF
}

# Argument parsing
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --region)
                AWS_REGION="$2"
                shift 2
                ;;
            --cluster-name)
                CLUSTER_NAME="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --version)
                VERSION_TAG="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-terraform)
                SKIP_TERRAFORM=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --force-redeploy)
                FORCE_REDEPLOY=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --help|-h)
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
}

# Main deployment orchestration
main() {
    parse_args "$@"

    highlight "🚀 GTerminal Production Deployment"
    info "Environment: $ENVIRONMENT"
    info "AWS Region: $AWS_REGION"
    info "Cluster: $CLUSTER_NAME"
    info "Namespace: $NAMESPACE"
    info "Version: $VERSION_TAG"

    if $DRY_RUN; then
        warn "DRY RUN MODE - No changes will be applied"
    fi

    # Prerequisites
    check_dependencies
    check_aws_authentication

    if $ROLLBACK; then
        configure_kubectl
        perform_rollback
        run_health_checks
        show_deployment_status
        success "Rollback completed successfully! 🎉"
        return 0
    fi

    # Main deployment flow
    deploy_infrastructure
    configure_kubectl
    build_and_push_images
    deploy_kubernetes_resources
    run_database_migration
    wait_for_deployment
    setup_monitoring
    run_health_checks
    run_smoke_tests
    cleanup_old_resources

    # Final status
    show_deployment_status

    success "Production deployment completed successfully! 🎉"

    # Show access URLs
    cat << 'EOF'

🌐 Access URLs:
┌─────────────────────────────────────────────────────┐
│  Service          │  URL                            │
├─────────────────────────────────────────────────────┤
│  Main App         │  https://gterminal.dev          │
│  Dashboard        │  https://dashboard.gterminal.dev│
│  Grafana          │  https://grafana.gterminal.dev  │
└─────────────────────────────────────────────────────┘

EOF
}

# Error handling
trap 'error "Deployment failed at line $LINENO"' ERR

# Run main function
main "$@"
