# GTerminal DevOps Integration Stack

A comprehensive production-ready development environment that orchestrates all Rust and Python components into a seamless, scalable system.

## 🚀 Overview

This DevOps integration creates a unified development and production environment that maximizes the power of:

- **Rust Filewatcher** - High-performance file monitoring with WebSocket streaming
- **Ruff LSP Integration** - AI-powered Python code analysis and fixing
- **Development Dashboard** - Real-time monitoring and metrics visualization
- **rufft-claude.sh** - AI-powered automated code improvement

## 📋 Complete System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                    │
├─────────────────────────────────────────────────────────────┤
│  gterminal.dev  │  dashboard.gterminal.dev  │  grafana.*   │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                  Service Mesh Layer                         │
├─────────────────┬─────────────────┬─────────────────────────┤
│ Rust Filewatcher│   Ruff LSP      │  Development Dashboard  │
│ Port: 8765/8766 │  Port: 8767/8768│  Port: 8080/8081       │
└─────────────────┴─────────────────┴─────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 Core Application Layer                      │
├─────────────────┬─────────────────┬─────────────────────────┤
│ GTerminal App   │  GTerminal MCP  │  GTerminal ReAct        │
│ Port: 8000      │  Port: 3000     │  Port: 8001            │
└─────────────────┴─────────────────┴─────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│              Infrastructure Layer                           │
├─────────────────┬─────────────────┬─────────────────────────┤
│ PostgreSQL      │     Redis       │   Monitoring Stack     │
│ Port: 5432      │   Port: 6379    │ Prometheus, Grafana     │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## 🛠️ Key Components

### 1. Enhanced Docker/Compose Setup

**Files:**

- `docker-compose.production.yml` - Complete production stack
- `Dockerfile.production` - Multi-stage optimized builds
- `rust-filewatcher/Dockerfile` - High-performance Rust container
- `Dockerfile.ruff-lsp` - AI-integrated LSP server
- `Dockerfile.dashboard` - Real-time development dashboard

**Features:**

- Multi-stage builds for optimization
- Health checks for all services
- Resource limits and reservations
- Service dependencies and startup ordering
- Volume management for persistent data
- Network isolation and security

### 2. Service Orchestration

**Files:**

- `orchestration/service-orchestrator.py` - Process lifecycle management
- `orchestration/services.yml` - Service configuration

**Capabilities:**

- Dependency-aware service startup
- Health monitoring with automatic restarts
- Resource usage tracking
- Graceful shutdown handling
- Service discovery and communication
- Load balancing and failover

### 3. CI/CD Integration

**Files:**

- `.github/workflows/devops-integration.yml` - Complete CI/CD pipeline

**Pipeline Stages:**

1. **Security Scanning** - Trivy, Bandit, cargo audit
2. **Multi-Platform Builds** - Rust and Python across OS/architectures
3. **Integration Testing** - End-to-end service communication
4. **Container Publishing** - Multi-arch images to GHCR
5. **Performance Testing** - Load testing and benchmarking
6. **Deployment** - Automated staging and production deployment

### 4. Comprehensive Monitoring

**Files:**

- `monitoring/prometheus.production.yml` - Metrics collection
- `monitoring/rules/gterminal-alerts.yml` - Advanced alerting
- `monitoring/alertmanager.yml` - Notification routing
- `monitoring/dashboards/gterminal-overview.json` - Grafana dashboards

**Monitoring Capabilities:**

- **Rust Filewatcher**: Event rates, WebSocket connections, performance metrics
- **Ruff LSP**: Response times, AI integration success rates, error tracking
- **Dashboard**: Real-time updates, client connections, system health
- **Infrastructure**: CPU, memory, disk, network metrics
- **Application**: HTTP metrics, database connections, business metrics

### 5. Development Workflow Automation

**Files:**

- `scripts/devops-orchestrator.sh` - Master development control script
- `scripts/integration-test-runner.sh` - Comprehensive integration testing

**Automation Features:**

- One-command environment setup
- Intelligent service management
- Automated testing across all components
- Performance profiling and benchmarking
- Development and production parity
- Hot reloading and debugging support

### 6. Production Infrastructure

**Files:**

- `deployment/kubernetes-production.yml` - Kubernetes manifests
- `deployment/terraform-aws.tf` - AWS infrastructure as code
- `deployment/deploy-production.sh` - Production deployment orchestration

**Infrastructure Features:**

- **Kubernetes**: Auto-scaling, rolling updates, health checks
- **AWS EKS**: Managed Kubernetes with optimized node groups
- **Database**: RDS PostgreSQL with read replicas
- **Cache**: ElastiCache Redis with clustering
- **Storage**: S3 for application data, EBS for persistent volumes
- **Networking**: VPC, Load Balancers, Route53, ACM certificates

## 🚀 Quick Start

### Development Environment

```bash
# 1. Setup complete development environment
./scripts/devops-orchestrator.sh setup

# 2. Start all services
./scripts/devops-orchestrator.sh start

# 3. View service status
./scripts/devops-orchestrator.sh status

# 4. Run integration tests
./scripts/integration-test-runner.sh run

# 5. View logs
./scripts/devops-orchestrator.sh logs rust-filewatcher --follow
```

### Production Deployment

```bash
# 1. Deploy infrastructure
./deployment/deploy-production.sh --environment production

# 2. Deploy specific version
./deployment/deploy-production.sh --version v1.2.3

# 3. Rollback if needed
./deployment/deploy-production.sh --rollback
```

## 📊 Service URLs

### Development

- **GTerminal App**: http://localhost:8080
- **Development Dashboard**: http://localhost:8769
- **Ruff LSP Server**: http://localhost:8768
- **Filewatcher Metrics**: http://localhost:8766
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3003

### Production

- **Main Application**: https://gterminal.dev
- **Development Dashboard**: https://dashboard.gterminal.dev
- **Monitoring (Grafana)**: https://grafana.gterminal.dev

## 🔧 Configuration

### Environment Variables

```bash
# Core Configuration
LOG_LEVEL=INFO
ENVIRONMENT=production
DEBUG=false

# Google Cloud Integration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_GENAI_USE_VERTEXAI=True
GCP_PROFILE=business

# Database Configuration
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379

# Service Integration
FILEWATCHER_WS_URL=ws://rust-filewatcher:8765
LSP_SERVER_URL=http://ruff-lsp:8767
DASHBOARD_URL=http://development-dashboard:8080

# Performance Tuning
RUST_LOG=info
FILEWATCHER_BATCH_SIZE=100
FILEWATCHER_DEBOUNCE_MS=50
LSP_MAX_CONCURRENT=10
DASHBOARD_UPDATE_INTERVAL=5

# Security (Production)
JWT_SECRET_KEY=secure-secret-key
CLAUDE_API_KEY=your-claude-api-key
SLACK_WEBHOOK_URL=your-slack-webhook
```

### Service Configuration

**Rust Filewatcher** (`rust-filewatcher/config.toml`):

```toml
[server]
websocket_port = 8765
metrics_port = 8766

[watching]
paths = ["/workspace"]
recursive = true

[performance]
batch_size = 100
debounce_ms = 50
max_events_per_second = 1000
```

**Service Orchestrator** (`orchestration/services.yml`):

```yaml
services:
  rust-filewatcher:
    command: ["rust-filewatcher", "--config", "/config/filewatcher.toml"]
    health_check_url: "http://localhost:8766/health"
    dependencies: []

  ruff-lsp:
    command: ["python", "/app/lsp-server.py"]
    health_check_url: "http://localhost:8768/health"
    dependencies: ["rust-filewatcher"]
```

## 🧪 Testing

### Integration Tests

The integration test suite validates complete system functionality:

```bash
# Run all integration tests
./scripts/integration-test-runner.sh run

# Setup test environment only
./scripts/integration-test-runner.sh setup

# Clean up test resources
./scripts/integration-test-runner.sh cleanup
```

**Test Coverage:**

- ✅ Filewatcher WebSocket connectivity
- ✅ File change event detection and streaming
- ✅ Ruff LSP diagnostics and AI suggestions
- ✅ Service-to-service communication
- ✅ Performance metrics collection
- ✅ Health check endpoints
- ✅ Error handling and recovery

### Performance Testing

```bash
# Run performance benchmarks
./scripts/devops-orchestrator.sh benchmark

# Profile specific service
./scripts/devops-orchestrator.sh profile rust-filewatcher 60

# Load testing with wrk
wrk -t4 -c100 -d30s http://localhost:8766/health
```

## 📈 Monitoring and Alerting

### Key Metrics

**Rust Filewatcher:**

- `rust_filewatcher_events_processed_total` - Events processed
- `rust_filewatcher_websocket_connections` - Active WebSocket connections
- `rust_filewatcher_memory_usage_bytes` - Memory consumption
- `rust_filewatcher_cpu_usage_seconds_total` - CPU usage

**Ruff LSP:**

- `ruff_lsp_request_duration_seconds` - Response time histogram
- `ruff_lsp_errors_total` - Error count
- `ruff_lsp_ai_requests_total` - AI integration usage
- `ruff_lsp_diagnostics_processed_total` - Code issues analyzed

**System:**

- `up` - Service availability
- `node_memory_MemAvailable_bytes` - Available memory
- `node_cpu_seconds_total` - CPU utilization
- `container_memory_usage_bytes` - Container resource usage

### Alerting Rules

**Critical Alerts:**

- Service down (immediate notification)
- High error rates (>10% for 2 minutes)
- Memory exhaustion (>90% for 5 minutes)
- Database connection failures

**Warning Alerts:**

- High response times (>5s for LSP, >1s for filewatcher)
- WebSocket connection drops
- AI integration failures
- Resource utilization trends

### Notification Channels

- **Slack**: Real-time alerts to development channels
- **Email**: Critical alerts to on-call team
- **PagerDuty**: Production incidents (via webhook)
- **Grafana**: Dashboard-based visual monitoring

## 🔒 Security

### Container Security

- Non-root user execution
- Read-only root filesystems where possible
- Minimal base images (Alpine/Distroless)
- Regular security scanning with Trivy
- Secrets management via Kubernetes secrets

### Network Security

- Network policies for pod-to-pod communication
- TLS encryption for all external traffic
- VPC isolation in AWS
- Security groups with minimal access
- WAF protection for web applications

### Access Control

- RBAC for Kubernetes resources
- IAM roles with least privilege
- Service accounts with limited permissions
- API authentication and authorization
- Audit logging for all access

## 🚢 Deployment Strategies

### Development

- Docker Compose for local development
- Hot reloading for fast iteration
- Integrated debugging support
- Real-time metrics and logs

### Staging

- Kubernetes deployment with reduced resources
- Blue-green deployments for testing
- Automated integration test execution
- Performance regression detection

### Production

- Multi-zone Kubernetes cluster
- Rolling updates with health checks
- Canary deployments for risk mitigation
- Automated rollback on failure
- Zero-downtime deployment guarantee

## 🔧 Troubleshooting

### Common Issues

**Service Won't Start:**

```bash
# Check service logs
./scripts/devops-orchestrator.sh logs <service-name>

# Debug specific service
./scripts/devops-orchestrator.sh debug <service-name>

# Check dependencies
kubectl get pods -n gterminal
```

**Performance Issues:**

```bash
# Check resource usage
./scripts/devops-orchestrator.sh profile <service-name>

# View metrics
curl http://localhost:8766/metrics  # Filewatcher
curl http://localhost:8768/metrics  # Ruff LSP
```

**Integration Problems:**

```bash
# Test service connectivity
./scripts/integration-test-runner.sh run

# Check WebSocket connections
curl -I http://localhost:8765

# Verify LSP functionality
curl -X POST http://localhost:8767/diagnostics \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/tmp/test.py"}'
```

### Performance Tuning

**Rust Filewatcher:**

- Adjust `FILEWATCHER_BATCH_SIZE` for event processing
- Tune `DEBOUNCE_MS` for file change sensitivity
- Scale WebSocket connections based on client count

**Ruff LSP:**

- Increase `LSP_MAX_CONCURRENT` for parallel processing
- Enable AI caching to reduce API calls
- Optimize workspace size for faster analysis

**Infrastructure:**

- Scale Kubernetes nodes based on resource utilization
- Adjust PostgreSQL connection pooling
- Tune Redis memory policies for caching efficiency

## 📚 Additional Resources

### Documentation

- [Architecture Guide](./docs/ARCHITECTURE.md)
- [API Documentation](./docs/API.md)
- [Deployment Guide](./docs/DEPLOYMENT.md)
- [Monitoring Runbook](./docs/MONITORING.md)

### Development

- [Contributing Guide](./CONTRIBUTING.md)
- [Development Setup](./docs/DEVELOPMENT.md)
- [Testing Guide](./TESTING.md)
- [Security Audit](./SECURITY_AUDIT_REPORT.md)

### Operations

- [Infrastructure as Code](./deployment/terraform-aws.tf)
- [Kubernetes Manifests](./deployment/kubernetes-production.yml)
- [Monitoring Configuration](./monitoring/)
- [CI/CD Pipeline](./.github/workflows/devops-integration.yml)

---

## 🏆 Key Benefits

✅ **Unified Development Experience** - Single command setup and management
✅ **Production-Ready Infrastructure** - Kubernetes, monitoring, and security built-in
✅ **AI-Powered Code Quality** - Automated code analysis and improvement
✅ **Real-Time Performance Monitoring** - Comprehensive metrics and alerting
✅ **Zero-Downtime Deployments** - Rolling updates with automatic rollback
✅ **Multi-Platform Support** - Consistent experience across development and production
✅ **Comprehensive Testing** - Unit, integration, and performance testing
✅ **Security First** - Container security, network policies, and access control

This DevOps integration stack provides a complete, production-ready development environment that maximizes the power of all integrated Rust and Python tools working together seamlessly.
