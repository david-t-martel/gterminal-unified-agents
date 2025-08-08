# Multi-stage Dockerfile for GTerminal ReAct Agent
# Optimized for production deployment with minimal image size

# ==================== Build Stage ====================
FROM python:3.12-slim as builder

# Build arguments
ARG VERSION=1.0.0
ARG BUILD_DATE
ARG VCS_REF
ARG PYTHON_VERSION=3.12

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    curl \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Rust for extensions
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /build

# Copy project files
COPY pyproject.toml uv.lock ./
COPY gemini_cli ./gemini_cli
COPY gterminal ./gterminal
COPY terminal ./terminal
COPY core ./core
COPY mcp ./mcp
COPY scripts ./scripts
COPY gterminal_rust_extensions ./gterminal_rust_extensions

# Build Rust extensions first
RUN cd gterminal_rust_extensions && \
    cargo build --release

# Create virtual environment and install dependencies
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv sync --no-dev

# Build wheel
RUN uv build --wheel && \
    cp dist/*.whl /tmp/

# ==================== Runtime Stage ====================
FROM python:3.12-slim

# Labels for container metadata
LABEL org.opencontainers.image.title="GTerminal ReAct Agent" \
      org.opencontainers.image.description="Advanced ReAct Agent with Gemini Integration and MCP Support" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/user/gterminal" \
      org.opencontainers.image.licenses="MIT"

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tini \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with specific UID/GID for security
RUN groupadd -g 1000 gterminal && \
    useradd -m -u 1000 -g 1000 -s /bin/bash gterminal && \
    mkdir -p /app /data /logs /config && \
    chown -R gterminal:gterminal /app /data /logs /config

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /tmp/*.whl /tmp/
COPY --from=builder /build/gterminal_rust_extensions/target/release/ /opt/rust_extensions/

# Install the application
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -f /tmp/*.whl

# Copy application files with proper ownership
WORKDIR /app
COPY --chown=gterminal:gterminal gemini_cli ./gemini_cli
COPY --chown=gterminal:gterminal gterminal ./gterminal
COPY --chown=gterminal:gterminal terminal ./terminal
COPY --chown=gterminal:gterminal core ./core
COPY --chown=gterminal:gterminal mcp ./mcp
COPY --chown=gterminal:gterminal scripts ./scripts

# Environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    APP_HOME=/app \
    DATA_DIR=/data \
    LOG_DIR=/logs \
    CONFIG_DIR=/config \
    PORT=8000 \
    MCP_PORT=3000 \
    WORKER_TIMEOUT=300 \
    MAX_WORKERS=4 \
    LOG_LEVEL=INFO \
    RUST_EXTENSIONS_PATH=/opt/rust_extensions

# Switch to non-root user
USER gterminal

# Health check script
COPY --chown=gterminal:gterminal <<'EOF' /app/healthcheck.py
#!/usr/bin/env python3
import sys
import requests
import os

def main():
    try:
        port = os.environ.get('PORT', '8000')
        response = requests.get(f'http://localhost:{port}/health', timeout=5)
        if response.status_code == 200:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception:
        sys.exit(1)

if __name__ == '__main__':
    main()
EOF

RUN chmod +x /app/healthcheck.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/healthcheck.py

# Create startup script
COPY --chown=gterminal:gterminal <<'EOF' /app/start.sh
#!/bin/bash
set -e

# Initialize directories
mkdir -p "$DATA_DIR" "$LOG_DIR" "$CONFIG_DIR"

# Start the application based on mode
case "${START_MODE:-server}" in
    "server")
        echo "Starting GTerminal server on port $PORT"
        exec python -m gterminal.server --host 0.0.0.0 --port "$PORT"
        ;;
    "mcp")
        echo "Starting MCP server on port $MCP_PORT"
        exec python -m mcp.server --host 0.0.0.0 --port "$MCP_PORT"
        ;;
    "react")
        echo "Starting ReAct agent server"
        exec python -m terminal.react_server --host 0.0.0.0 --port "$PORT"
        ;;
    *)
        echo "Unknown start mode: $START_MODE"
        echo "Available modes: server, mcp, react"
        exit 1
        ;;
esac
EOF

RUN chmod +x /app/start.sh

# Volumes for persistent data
VOLUME ["/data", "/logs", "/config"]

# Expose ports
EXPOSE ${PORT} ${MCP_PORT}

# Use tini as entrypoint to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["/app/start.sh"]
