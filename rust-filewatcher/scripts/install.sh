#!/bin/bash
# Installation script for gterminal-filewatcher

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

# Project paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GTERMINAL_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
INSTALL_DIR="/usr/local/bin"

main() {
    log "Installing gterminal-filewatcher..."

    # Check dependencies
    check_dependencies

    # Build the project
    build_project

    # Install binary
    install_binary

    # Create configuration
    setup_configuration

    # Create systemd service (optional)
    setup_systemd_service

    # Integration with existing tools
    setup_gterminal_integration

    success "gterminal-filewatcher installed successfully!"
    log "Run 'gterminal-filewatcher --help' for usage information"
}

check_dependencies() {
    log "Checking dependencies..."

    # Check Rust
    if ! command -v cargo &> /dev/null; then
        error "Rust/Cargo not found. Please install Rust first."
        exit 1
    fi

    # Check optional tools
    local tools=("ruff" "mypy" "ast-grep" "biome" "tsc")
    local missing_tools=()

    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -gt 0 ]; then
        warn "Optional tools not found: ${missing_tools[*]}"
        warn "Some analysis features may not be available"
    fi

    success "Dependencies check completed"
}

build_project() {
    log "Building gterminal-filewatcher..."

    cd "$PROJECT_ROOT"

    # Clean previous builds
    cargo clean

    # Build optimized release
    cargo build --release

    success "Build completed"
}

install_binary() {
    log "Installing binary to $INSTALL_DIR..."

    local binary="$PROJECT_ROOT/target/release/gterminal-filewatcher"

    if [ ! -f "$binary" ]; then
        error "Binary not found: $binary"
        exit 1
    fi

    # Copy binary
    sudo cp "$binary" "$INSTALL_DIR/"
    sudo chmod +x "$INSTALL_DIR/gterminal-filewatcher"

    success "Binary installed to $INSTALL_DIR/gterminal-filewatcher"
}

setup_configuration() {
    log "Setting up configuration..."

    local config_dir="$HOME/.config/gterminal-filewatcher"
    local config_file="$config_dir/config.toml"

    # Create config directory
    mkdir -p "$config_dir"

    # Create default configuration if it doesn't exist
    if [ ! -f "$config_file" ]; then
        cat > "$config_file" << 'EOF'
# gterminal-filewatcher configuration

[watch]
extensions = ["py", "ts", "tsx", "js", "jsx", "rs", "json", "yaml", "yml", "toml"]
ignore_dirs = ["node_modules", "target", "__pycache__", ".git", ".venv", "venv", "dist", "build"]
ignore_patterns = ["*.pyc", "*.pyo", "*.log", "*.tmp"]
debounce_ms = 100
recursive = true

[server]
host = "127.0.0.1"
http_port = 8767
websocket_port = 8768
cors_enabled = true
request_timeout = 30

[performance]
max_parallel_jobs = 8
batch_size = 10
process_interval_ms = 50
cache_size = 1000
memory_optimization = true

[integration]
rufft_claude_script = "scripts/rufft-claude.sh"
dashboard_status_file = "dashboard_status.json"
mcp_enabled = false
notifications_enabled = false

[tools.ruff]
executable = "ruff"
args = ["check", "--output-format", "json"]
extensions = ["py"]
auto_fix = true
priority = 1
timeout = 30

[tools.ruff-format]
executable = "ruff"
args = ["format"]
extensions = ["py"]
auto_fix = true
priority = 2
timeout = 30

[tools.mypy]
executable = "mypy"
args = ["--show-error-codes", "--no-error-summary"]
extensions = ["py"]
auto_fix = false
priority = 3
timeout = 60

[tools.ast-grep]
executable = "ast-grep"
args = ["scan", "--json"]
extensions = ["py", "js", "ts", "rs"]
auto_fix = true
priority = 0
timeout = 30
EOF
        success "Created default configuration: $config_file"
    else
        log "Configuration file already exists: $config_file"
    fi
}

setup_systemd_service() {
    log "Setting up systemd service..."

    local service_file="/etc/systemd/system/gterminal-filewatcher.service"
    local user="$(whoami)"

    # Ask if user wants systemd service
    read -p "Create systemd service for automatic startup? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Skipping systemd service setup"
        return
    fi

    sudo tee "$service_file" > /dev/null << EOF
[Unit]
Description=gterminal-filewatcher - High-performance file watcher
After=network.target
Wants=network.target

[Service]
Type=simple
User=$user
Group=$user
WorkingDirectory=$GTERMINAL_ROOT
ExecStart=$INSTALL_DIR/gterminal-filewatcher watch --path $GTERMINAL_ROOT
Restart=always
RestartSec=10
Environment=RUST_LOG=info
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=30

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=$GTERMINAL_ROOT

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload

    success "Systemd service created: $service_file"
    log "To enable: sudo systemctl enable gterminal-filewatcher"
    log "To start: sudo systemctl start gterminal-filewatcher"
}

setup_gterminal_integration() {
    log "Setting up gterminal project integration..."

    # Create integration script in gterminal scripts directory
    local integration_script="$GTERMINAL_ROOT/scripts/start-filewatcher.sh"

    cat > "$integration_script" << 'EOF'
#!/bin/bash
# gterminal-filewatcher integration script

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILEWATCHER_CONFIG="$PROJECT_ROOT/.filewatcher.toml"

# Create project-specific configuration if it doesn't exist
if [ ! -f "$FILEWATCHER_CONFIG" ]; then
    cat > "$FILEWATCHER_CONFIG" << 'INNER_EOF'
# gterminal project filewatcher configuration

[watch]
extensions = ["py", "ts", "tsx", "js", "jsx", "rs", "json", "yaml", "yml", "toml"]
ignore_dirs = [
    "node_modules", "target", "__pycache__", ".git", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", "htmlcov"
]
ignore_patterns = ["*.pyc", "*.pyo", "*.log", "*.tmp", "*~", ".DS_Store"]
debounce_ms = 50  # Faster response for development
recursive = true

[server]
host = "127.0.0.1"
http_port = 8767
websocket_port = 8768
cors_enabled = true

[integration]
rufft_claude_script = "scripts/rufft-claude.sh"
dashboard_status_file = "dashboard_status.json"
mcp_enabled = true

[tools.ruff]
executable = "ruff"
extensions = ["py"]
auto_fix = true
priority = 1
timeout = 30

[tools.mypy]
executable = "mypy"
extensions = ["py"]
auto_fix = false
priority = 3
timeout = 60

[tools.ast-grep]
executable = "ast-grep"
extensions = ["py", "js", "ts", "rs"]
auto_fix = true
priority = 0
timeout = 30
INNER_EOF
fi

# Start the filewatcher
echo "🚀 Starting gterminal-filewatcher..."
exec gterminal-filewatcher watch --path "$PROJECT_ROOT" --port 8767 --ws-port 8768
EOF

    chmod +x "$integration_script"
    success "Created integration script: $integration_script"

    # Update rufft-claude.sh to work with filewatcher
    local rufft_claude="$GTERMINAL_ROOT/scripts/rufft-claude.sh"
    if [ -f "$rufft_claude" ]; then
        # Add filewatcher integration to rufft-claude.sh
        if ! grep -q "gterminal-filewatcher" "$rufft_claude"; then
            log "Adding filewatcher integration to rufft-claude.sh..."

            # Backup original
            cp "$rufft_claude" "$rufft_claude.bak"

            # Add integration function
            cat >> "$rufft_claude" << 'EOF'

# gterminal-filewatcher integration
notify_filewatcher() {
    local action="$1"
    local file="${2:-}"
    local port="${FILEWATCHER_PORT:-8767}"

    if command -v curl &> /dev/null; then
        case "$action" in
            fix_started)
                curl -s -X POST "http://localhost:$port/dashboard/update" \
                    -H "Content-Type: application/json" \
                    -d "{\"type\": \"fix_started\", \"file\": \"$file\"}" || true
                ;;
            fix_completed)
                curl -s -X POST "http://localhost:$port/dashboard/update" \
                    -H "Content-Type: application/json" \
                    -d "{\"type\": \"fix_completed\", \"file\": \"$file\"}" || true
                ;;
        esac
    fi
}

# Hook into existing functions if they exist
if declare -f auto_fix_all > /dev/null; then
    # Wrap auto_fix_all with notifications
    _original_auto_fix_all="$(declare -f auto_fix_all)"
    eval "${_original_auto_fix_all/auto_fix_all/_original_auto_fix_all}"

    auto_fix_all() {
        notify_filewatcher fix_started
        _original_auto_fix_all "$@"
        notify_filewatcher fix_completed
    }
fi
EOF
            success "Added filewatcher integration to rufft-claude.sh"
        fi
    fi

    # Create desktop entry (optional)
    create_desktop_entry
}

create_desktop_entry() {
    local desktop_dir="$HOME/.local/share/applications"
    local desktop_file="$desktop_dir/gterminal-filewatcher.desktop"

    mkdir -p "$desktop_dir"

    cat > "$desktop_file" << EOF
[Desktop Entry]
Name=gterminal-filewatcher
Comment=High-performance file watcher for gterminal project
Exec=$INSTALL_DIR/gterminal-filewatcher watch --path $GTERMINAL_ROOT
Icon=folder-watching
Terminal=true
Type=Application
Categories=Development;
StartupNotify=false
EOF

    success "Created desktop entry: $desktop_file"
}

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error "Installation failed with exit code $exit_code"
    fi
    exit $exit_code
}

trap cleanup EXIT

# Run main installation
main "$@"
