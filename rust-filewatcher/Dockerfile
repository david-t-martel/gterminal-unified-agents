# High-Performance Rust Filewatcher Container
# Optimized for production workloads

ARG RUST_VERSION=1.75

FROM rust:${RUST_VERSION}-slim as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Cargo files first (for better caching)
COPY Cargo.toml Cargo.lock ./

# Create dummy main to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    echo 'name = "rust-filewatcher"' >> Cargo.toml && \
    echo '[[bin]]' >> Cargo.toml && \
    echo 'name = "rust-filewatcher"' >> Cargo.toml && \
    echo 'path = "src/main.rs"' >> Cargo.toml

# Build dependencies
RUN cargo build --release && rm -rf src

# Copy actual source code
COPY src/ ./src/
COPY benches/ ./benches/

# Build the application with optimizations
RUN cargo build --release --features production

# ========================
# Runtime Stage
# ========================
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r filewatcher && useradd -r -g filewatcher -d /app -s /bin/bash filewatcher

# Copy binary from builder
COPY --from=builder /app/target/release/rust-filewatcher /usr/local/bin/rust-filewatcher

# Create required directories
RUN mkdir -p /data /logs /config \
    && chown -R filewatcher:filewatcher /data /logs /config

# Create default configuration
RUN cat > /config/filewatcher.toml << 'EOF'
# Rust Filewatcher Configuration
[server]
websocket_port = 8765
metrics_port = 8766
host = "0.0.0.0"

[watching]
paths = ["/workspace"]
recursive = true
ignore_patterns = [
    ".git/**",
    "node_modules/**",
    ".venv/**",
    "__pycache__/**",
    "*.pyc",
    "target/**",
    ".cache/**"
]

[performance]
batch_size = 100
debounce_ms = 50
max_events_per_second = 1000
worker_threads = 4

[features]
metrics_enabled = true
health_check = true
compression = true
rate_limiting = true

[logging]
level = "info"
file = "/logs/filewatcher.log"
max_size = "100MB"
max_files = 5
EOF

# Set ownership
RUN chown -R filewatcher:filewatcher /config

# Create healthcheck script
RUN cat > /usr/local/bin/health-check.sh << 'EOF'
#!/bin/bash
set -e

# Check if WebSocket server is responding
if curl -f "http://localhost:${METRICS_PORT:-8766}/health" > /dev/null 2>&1; then
    echo "✅ Filewatcher is healthy"
    exit 0
else
    echo "❌ Filewatcher health check failed"
    exit 1
fi
EOF

RUN chmod +x /usr/local/bin/health-check.sh

# Switch to non-root user
USER filewatcher

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 8765 8766

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD /usr/local/bin/health-check.sh

# Run the filewatcher
CMD ["rust-filewatcher", "--config", "/config/filewatcher.toml"]
