#!/bin/bash
# Start gterminal-filewatcher for development

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILEWATCHER_DIR="$PROJECT_ROOT/rust-filewatcher"
FILEWATCHER_BIN="$FILEWATCHER_DIR/target/release/gterminal-filewatcher"

main() {
    log "Starting gterminal-filewatcher..."

    # Check if binary exists
    if [ ! -f "$FILEWATCHER_BIN" ]; then
        warn "Binary not found, building..."
        build_filewatcher
    fi

    # Check for configuration
    check_configuration

    # Start the filewatcher
    start_filewatcher
}

build_filewatcher() {
    log "Building gterminal-filewatcher..."

    cd "$FILEWATCHER_DIR"

    # Build release version
    if ! cargo build --release; then
        error "Build failed"
        exit 1
    fi

    success "Build completed"
}

check_configuration() {
    local config_file="$PROJECT_ROOT/.filewatcher.toml"

    if [ ! -f "$config_file" ]; then
        warn "No configuration file found at $config_file"
        warn "Using default configuration"
    else
        log "Using configuration: $config_file"
    fi
}

start_filewatcher() {
    local args=(
        "watch"
        "--path" "$PROJECT_ROOT"
        "--port" "8767"
        "--ws-port" "8768"
    )

    # Add verbose logging if requested
    if [ "${VERBOSE:-}" = "1" ]; then
        export RUST_LOG="debug,gterminal_filewatcher=trace"
    else
        export RUST_LOG="info,gterminal_filewatcher=debug"
    fi

    log "Starting filewatcher with:"
    log "  Project path: $PROJECT_ROOT"
    log "  HTTP API: http://localhost:8767"
    log "  WebSocket: ws://localhost:8768/ws"
    log "  Log level: $RUST_LOG"

    # Start the filewatcher
    exec "$FILEWATCHER_BIN" "${args[@]}"
}

# Handle command line arguments
case "${1:-}" in
    build)
        build_filewatcher
        exit 0
        ;;
    config)
        if [ -f "$PROJECT_ROOT/.filewatcher.toml" ]; then
            cat "$PROJECT_ROOT/.filewatcher.toml"
        else
            warn "No configuration file found"
        fi
        exit 0
        ;;
    install)
        log "Running installation script..."
        exec "$FILEWATCHER_DIR/scripts/install.sh"
        ;;
    help|--help|-h)
        cat << 'EOF'
Usage: start-filewatcher.sh [command]

Commands:
  (none)    Start the filewatcher
  build     Build the filewatcher binary
  config    Show current configuration
  install   Run the installation script
  help      Show this help

Environment Variables:
  VERBOSE=1    Enable verbose logging

Examples:
  ./scripts/start-filewatcher.sh              # Start with default settings
  VERBOSE=1 ./scripts/start-filewatcher.sh    # Start with debug logging
  ./scripts/start-filewatcher.sh build        # Build the binary
  ./scripts/start-filewatcher.sh config       # Show configuration
EOF
        exit 0
        ;;
esac

# Run main function
main "$@"
