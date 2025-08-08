#!/bin/bash
# Canonical Integration Script - Comprehensive toolchain integration for gterminal
# Combines enhanced integration wrapper with rust tools integration and performance tracking

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENHANCED_TOOLCHAIN="$SCRIPT_DIR/enhanced_toolchain.py"
CONFIG_FILE="$PROJECT_ROOT/config/enhanced_toolchain.json"
RUFFT_CLAUDE="$SCRIPT_DIR/rufft-claude.sh"
DASHBOARD_PORT=${DASHBOARD_PORT:-8767}
DASHBOARD_FILE="$PROJECT_ROOT/.development-dashboard.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Tool performance tracking
declare -A TOOL_PERFORMANCE

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $*${NC}" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS] $*${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[WARN] $*${NC}" >&2
}

error() {
    echo -e "${RED}[ERROR] $*${NC}" >&2
}

info() {
    echo -e "${PURPLE}[INFO] $*${NC}" >&2
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    local missing_deps=()

    # Check for uv
    if ! command -v uv &> /dev/null; then
        missing_deps+=("uv")
    fi

    # Check for Python 3.12+
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ $(echo "$python_version < 3.12" | bc -l 2>/dev/null || echo "1") -eq 1 ]]; then
            missing_deps+=("python3.12+")
        fi
    else
        missing_deps+=("python3")
    fi

    # Check if enhanced toolchain exists
    if [[ ! -f "$ENHANCED_TOOLCHAIN" ]]; then
        log_error "Enhanced toolchain script not found: $ENHANCED_TOOLCHAIN"
        exit 1
    fi

    # Check if rufft-claude exists
    if [[ ! -f "$RUFFT_CLAUDE" ]]; then
        log_error "Rufft-claude script not found: $RUFFT_CLAUDE"
        exit 1
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install missing dependencies and try again"
        exit 1
    fi

    log_success "All dependencies satisfied"
}

# Install optional dependencies
install_optional_deps() {
    log_info "Installing optional dependencies for enhanced features..."

    # Try to install watchdog for file watching
    if uv add watchdog --quiet 2>/dev/null; then
        log_success "File watching support enabled (watchdog installed)"
    else
        log_warning "File watching support disabled (watchdog not available)"
    fi

    # Try to install websockets for real-time communication
    if uv add websockets --quiet 2>/dev/null; then
        log_success "WebSocket support enabled"
    else
        log_warning "WebSocket support disabled"
    fi
}

# Tool detection with performance tracking
detect_rust_tools() {
    log "Detecting rust-based development tools..."

    local tools=(
        "ruff:Python linter/formatter"
        "ast-grep:Structural code analysis"
        "biome:JavaScript/TypeScript formatter"
        "taplo:TOML formatter"
        "stylua:Lua formatter"
        "fd:Fast file finder"
        "rg:Fast grep replacement"
        "cargo:Rust package manager"
        "rust-analyzer:Rust LSP"
    )

    local available_tools=()
    for tool_info in "${tools[@]}"; do
        local tool="${tool_info%%:*}"
        local desc="${tool_info#*:}"

        local start_time=$(date +%s%3N)
        if command -v "$tool" >/dev/null 2>&1; then
            local end_time=$(date +%s%3N)
            local duration=$((end_time - start_time))
            TOOL_PERFORMANCE["$tool"]=$duration
            available_tools+=("$tool:$desc")
            success "$tool ($desc) - Available (${duration}ms detection time)"
        else
            warn "$tool ($desc) - Not available"
        fi
    done

    echo "${#available_tools[@]} rust-based tools available"
    return 0
}

# Performance benchmarking
benchmark_tools() {
    log "Benchmarking tool performance..."

    local test_file="$PROJECT_ROOT/gterminal/main.py"
    if [[ ! -f "$test_file" ]]; then
        warn "Test file not found, creating sample..."
        test_file="/tmp/benchmark_test.py"
        cat > "$test_file" << 'EOF'
import os
import sys
from typing import Optional

def example_function(name: str, age: int = 25) -> Optional[str]:
    """Example function for benchmarking"""
    if name:
        return f"Hello {name}, you are {age} years old"
    return None

if __name__ == "__main__":
    print("Benchmark test file")
EOF
    fi

    # Benchmark each tool
    declare -A benchmarks

    # Ruff benchmark
    if command -v ruff >/dev/null 2>&1; then
        local start=$(date +%s%3N)
        ruff check "$test_file" >/dev/null 2>&1 || true
        local end=$(date +%s%3N)
        benchmarks["ruff_check"]=$((end - start))

        start=$(date +%s%3N)
        ruff format --diff "$test_file" >/dev/null 2>&1 || true
        end=$(date +%s%3N)
        benchmarks["ruff_format"]=$((end - start))
    fi

    # ast-grep benchmark
    if command -v ast-grep >/dev/null 2>&1; then
        local start=$(date +%s%3N)
        ast-grep scan "$test_file" >/dev/null 2>&1 || true
        local end=$(date +%s%3N)
        benchmarks["ast_grep_scan"]=$((end - start))
    fi

    # Display results
    info "Tool Performance Benchmarks:"
    for tool in "${!benchmarks[@]}"; do
        echo "  $tool: ${benchmarks[$tool]}ms"
    done

    # Update dashboard
    update_dashboard_metrics "benchmarks" "${benchmarks[@]}"
}

# AST-grep integration with project rules
setup_ast_grep_rules() {
    log "Setting up AST-grep rules for gterminal project..."

    local ast_grep_dir="$PROJECT_ROOT/.ast-grep"

    # Copy rules from extracted infrastructure
    if [[ -d "$ast_grep_dir/rules" ]]; then
        success "AST-grep rules already configured ($(find "$ast_grep_dir/rules" -name "*.yml" | wc -l) rules)"
    else
        warn "AST-grep rules not found - they should have been extracted from my-fullstack-agent"
        return 1
    fi

    # Test rule application
    log "Testing AST-grep rule application..."
    local test_output="/tmp/ast-grep-test.json"

    if ast-grep scan --json "$PROJECT_ROOT" > "$test_output" 2>/dev/null; then
        local issue_count=$(jq length "$test_output" 2>/dev/null || echo "0")
        info "AST-grep found $issue_count issues to review"

        # Show sample issues (first 3)
        if [[ "$issue_count" != "0" ]]; then
            log "Sample issues found:"
            jq -r '.[0:3][] | "  - \(.rule.id): \(.message) in \(.file)"' "$test_output" 2>/dev/null || true
        fi
    else
        warn "AST-grep scan failed"
    fi

    rm -f "$test_output"
}

# Ruff server integration (if available)
setup_ruff_server() {
    log "Configuring Ruff server integration..."

    if ! command -v ruff >/dev/null 2>&1; then
        warn "Ruff not available"
        return 1
    fi

    # Check if ruff supports server mode
    if ruff server --help >/dev/null 2>&1; then
        success "Ruff server mode available"

        # Create ruff server config
        cat > "$PROJECT_ROOT/.ruff-server.json" << EOF
{
    "settings": {
        "lint": {
            "select": ["E", "F", "I", "B", "C4", "UP", "ARG", "SIM", "TCH", "PTH", "ERA", "PL", "RUF"],
            "ignore": ["E501", "PLR0913", "PLR0915", "RUF012"]
        },
        "format": {
            "quote-style": "double",
            "line-length": 100
        }
    }
}
EOF
        success "Ruff server configuration created"
    else
        info "Ruff server mode not available in this version"
    fi
}

# Dashboard data collection
collect_dashboard_data() {
    local timestamp=$(date -Iseconds)

    # Collect metrics
    local metrics="{
        \"timestamp\": \"$timestamp\",
        \"project\": \"gterminal\",
        \"tools\": {
            \"available\": $(detect_rust_tools | tail -1 | cut -d' ' -f1),
            \"performance\": $(printf '%s\n' "${!TOOL_PERFORMANCE[@]}" | jq -R . | jq -s 'map(split(":")) | from_entries' <<< "$(for k in "${!TOOL_PERFORMANCE[@]}"; do echo "$k:${TOOL_PERFORMANCE[$k]}"; done)" 2>/dev/null || echo "{}")
        },
        \"status\": {
            \"ruff_issues\": $(ruff check . --output-format json 2>/dev/null | jq length || echo 0),
            \"ast_grep_issues\": $(ast-grep scan --json . 2>/dev/null | jq length || echo 0),
            \"mypy_errors\": $(mypy . --ignore-missing-imports 2>&1 | grep -c "error:" || echo 0),
            \"file_count\": $(find . -name "*.py" | wc -l)
        }
    }"

    echo "$metrics" > "$DASHBOARD_FILE"
}

# Update dashboard with specific metrics
update_dashboard_metrics() {
    local category="$1"
    shift
    local values="$*"

    if [[ ! -f "$DASHBOARD_FILE" ]]; then
        collect_dashboard_data
    fi

    # Update specific section using jq
    local temp_file="/tmp/dashboard_update.json"
    if command -v jq >/dev/null 2>&1; then
        jq --argjson values "{$(printf '"%s"' "$values")}" \
           ".status.$category = \$values | .timestamp = \"$(date -Iseconds)\"" \
           "$DASHBOARD_FILE" > "$temp_file" && mv "$temp_file" "$DASHBOARD_FILE"
    fi
}

# Fast development server for dashboard
start_dashboard_server() {
    log "Starting fast development dashboard on port $DASHBOARD_PORT..."

    # Create dashboard HTML
    cat > "$PROJECT_ROOT/.dashboard.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>GTerminal Development Dashboard</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: monospace; background: #1e1e1e; color: #d4d4d4; margin: 20px; }
        .header { border-bottom: 2px solid #007acc; padding: 10px 0; margin-bottom: 20px; }
        .metric { background: #252526; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .metric h3 { margin: 0 0 10px 0; color: #4ec9b0; }
        .status { display: inline-block; padding: 3px 8px; border-radius: 3px; margin: 2px; }
        .success { background: #4caf50; color: white; }
        .warning { background: #ff9800; color: white; }
        .error { background: #f44336; color: white; }
        .info { background: #2196f3; color: white; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .json { background: #2d2d30; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 GTerminal Development Dashboard</h1>
        <p>Real-time development metrics and tool status</p>
    </div>

    <div class="grid">
        <div class="metric">
            <h3>📊 Quick Status</h3>
            <div id="quickStatus">Loading...</div>
        </div>

        <div class="metric">
            <h3>🛠️ Tool Performance</h3>
            <div id="toolPerformance">Loading...</div>
        </div>
    </div>

    <div class="metric">
        <h3>📋 Raw Data</h3>
        <pre id="rawData" class="json">Loading...</pre>
    </div>

    <script>
        async function updateDashboard() {
            try {
                const response = await fetch('/data');
                const data = await response.json();

                // Quick status
                const quickStatus = document.getElementById('quickStatus');
                quickStatus.innerHTML = `
                    <span class="status ${data.status.ruff_issues > 0 ? 'warning' : 'success'}">
                        Ruff: ${data.status.ruff_issues} issues
                    </span>
                    <span class="status ${data.status.ast_grep_issues > 0 ? 'warning' : 'success'}">
                        AST-grep: ${data.status.ast_grep_issues} issues
                    </span>
                    <span class="status ${data.status.mypy_errors > 0 ? 'error' : 'success'}">
                        MyPy: ${data.status.mypy_errors} errors
                    </span>
                    <span class="status info">Files: ${data.status.file_count}</span>
                `;

                // Tool performance
                const toolPerf = document.getElementById('toolPerformance');
                if (data.tools.performance) {
                    let perfHTML = '';
                    for (const [tool, time] of Object.entries(data.tools.performance)) {
                        const status = time < 100 ? 'success' : time < 500 ? 'warning' : 'error';
                        perfHTML += `<span class="status ${status}">${tool}: ${time}ms</span>`;
                    }
                    toolPerf.innerHTML = perfHTML;
                } else {
                    toolPerf.innerHTML = '<span class="status info">No performance data</span>';
                }

                // Raw data
                document.getElementById('rawData').textContent = JSON.stringify(data, null, 2);

            } catch (error) {
                console.error('Dashboard update failed:', error);
            }
        }

        // Update every 5 seconds
        setInterval(updateDashboard, 5000);
        updateDashboard();
    </script>
</body>
</html>
EOF

    # Start simple Python HTTP server with custom handler
    python3 -c "
import http.server
import socketserver
import json
import os
from urllib.parse import urlparse

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('$PROJECT_ROOT/.dashboard.html', 'rb') as f:
                self.wfile.write(f.read())
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                with open('$DASHBOARD_FILE', 'r') as f:
                    self.wfile.write(f.read().encode())
            except FileNotFoundError:
                self.wfile.write(b'{\"error\": \"Dashboard data not available\"}')
        else:
            super().do_GET()

os.chdir('$PROJECT_ROOT')
with socketserver.TCPServer(('', $DASHBOARD_PORT), DashboardHandler) as httpd:
    print(f'Dashboard server running at http://localhost:$DASHBOARD_PORT')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\\nDashboard server stopped')
" &

    local server_pid=$!
    echo "$server_pid" > "$PROJECT_ROOT/.dashboard.pid"
    success "Dashboard server started (PID: $server_pid)"
    success "Access dashboard at: http://localhost:$DASHBOARD_PORT"
}

# Auto-fix pipeline using rust tools
run_auto_fix() {
    log "Running comprehensive auto-fix using rust-based tools..."

    # Collect initial metrics
    collect_dashboard_data

    # Apply fixes in parallel
    local fix_commands=()

    # Ruff fixes
    if command -v ruff >/dev/null 2>&1; then
        fix_commands+=("ruff check --fix --unsafe-fixes . || true")
        fix_commands+=("ruff format . || true")
    fi

    # AST-grep fixes
    if command -v ast-grep >/dev/null 2>&1 && [[ -d "$PROJECT_ROOT/.ast-grep" ]]; then
        fix_commands+=("ast-grep scan --update-all . || true")
    fi

    # Run fixes in parallel with limited concurrency
    local max_jobs=3
    local pids=()

    for cmd in "${fix_commands[@]}"; do
        while (( ${#pids[@]} >= max_jobs )); do
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[i]}" 2>/dev/null; then
                    wait "${pids[i]}" 2>/dev/null || true
                    unset "pids[i]"
                fi
            done
            pids=("${pids[@]}")  # Reindex
            sleep 0.1
        done

        eval "$cmd" &
        pids+=($!)
    done

    # Wait for all fixes to complete
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    # Collect final metrics
    collect_dashboard_data

    success "Auto-fix pipeline completed"
}

# Run deployment checks
run_deploy_check() {
    log_info "Running pre-deployment validation suite..."

    local overall_success=true

    # Critical tools that must pass for deployment
    local critical_tools=("rufft_claude_auto" "ruff" "mypy")

    log_info "Running critical deployment checks..."
    for tool in "${critical_tools[@]}"; do
        log_info "Running $tool..."
        if uv run python "$ENHANCED_TOOLCHAIN" --run-suite . --tool-filter "$tool" > /dev/null 2>&1; then
            log_success "$tool passed"
        else
            log_error "$tool failed - deployment blocked"
            overall_success=false
        fi
    done

    if [[ "$overall_success" == "true" ]]; then
        log_success "🚀 All deployment checks passed - ready to deploy!"
        return 0
    else
        log_error "❌ Deployment checks failed - fix issues before deploying"
        return 1
    fi
}

# Quick fix using rufft-claude
quick_fix() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        log_error "File not found: $file"
        exit 1
    fi

    log_info "Running quick fix on $file..."

    if uv run "$RUFFT_CLAUDE" auto-fix "$file"; then
        log_success "Quick fix completed for $file"
    else
        log_error "Quick fix failed for $file"
        exit 1
    fi
}

# AI suggestions using rufft-claude
ai_suggest() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        log_error "File not found: $file"
        exit 1
    fi

    log_info "Getting AI suggestions for $file..."

    if uv run "$RUFFT_CLAUDE" ai-suggest "$file"; then
        log_success "AI suggestions completed for $file"
    else
        log_error "AI suggestions failed for $file"
        exit 1
    fi
}

# Show configuration
show_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        log_info "Current configuration:"
        cat "$CONFIG_FILE"
    else
        log_warning "Configuration file not found: $CONFIG_FILE"
        log_info "Using default configuration"
    fi
}

# Show usage
show_usage() {
    cat << EOF
Canonical Integration Script - Comprehensive toolchain integration for gterminal

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    monitor                 Start monitoring mode with real-time dashboard
    run-suite FILE          Run complete tool suite on file/directory
    watch DIR               Watch directory for changes and auto-run tools
    quick-fix FILE          Quick rufft-claude auto-fix for single file
    ai-suggest FILE         Get AI suggestions for single file
    deploy-check            Run pre-deployment validation suite
    setup                   Install optional dependencies and setup tools
    config                  Show current configuration
    dashboard               Start development dashboard server
    benchmark               Benchmark tool performance
    auto-fix                Run comprehensive auto-fix pipeline
    status                  Show current status as JSON

OPTIONS:
    --tool-filter STR       Filter tools by name pattern
    --config FILE           Use custom configuration file
    --verbose               Enable verbose logging
    --help                  Show this help message

EXAMPLES:
    # Start monitoring dashboard
    $0 monitor

    # Run complete suite on Python file
    $0 run-suite src/main.py

    # Watch current directory for changes
    $0 watch .

    # Quick fix with rufft-claude
    $0 quick-fix problematic_file.py

    # Get AI suggestions
    $0 ai-suggest complex_code.py

    # Run deployment validation
    $0 deploy-check

    # Setup with enhanced features
    $0 setup

    # Start performance dashboard
    $0 dashboard

    # Run comprehensive auto-fix
    $0 auto-fix

INTEGRATION FEATURES:
    • Automatic file watching with smart debouncing
    • Integrated rufft-claude.sh with priority handling
    • Real-time monitoring dashboard with performance metrics
    • Deployment pipeline integration
    • AI-powered code suggestions
    • Rust-based tool integration with benchmarking
    • Multi-tool orchestration with dependency management

EOF
}

# Main execution
main() {
    local command="${1:-help}"
    shift || true

    case "$command" in
        "monitor")
            check_dependencies
            log_info "Starting monitoring mode..."
            uv run python "$ENHANCED_TOOLCHAIN" --monitor
            ;;
        "run-suite")
            if [[ $# -eq 0 ]]; then
                log_error "File/directory required for run-suite command"
                exit 1
            fi
            check_dependencies
            uv run python "$ENHANCED_TOOLCHAIN" --run-suite "$1" "${@:2}"
            ;;
        "watch")
            if [[ $# -eq 0 ]]; then
                log_error "Directory required for watch command"
                exit 1
            fi
            check_dependencies
            log_info "Starting file watcher on $1..."
            uv run python "$ENHANCED_TOOLCHAIN" --watch "$1" "${@:2}"
            ;;
        "quick-fix")
            if [[ $# -eq 0 ]]; then
                log_error "File required for quick-fix command"
                exit 1
            fi
            check_dependencies
            quick_fix "$1"
            ;;
        "ai-suggest")
            if [[ $# -eq 0 ]]; then
                log_error "File required for ai-suggest command"
                exit 1
            fi
            check_dependencies
            ai_suggest "$1"
            ;;
        "deploy-check")
            check_dependencies
            run_deploy_check
            ;;
        "setup")
            check_dependencies
            detect_rust_tools
            setup_ast_grep_rules
            setup_ruff_server
            install_optional_deps
            collect_dashboard_data
            log_success "Setup completed"
            ;;
        "config")
            show_config
            ;;
        "dashboard")
            collect_dashboard_data
            start_dashboard_server
            ;;
        "benchmark")
            benchmark_tools
            ;;
        "auto-fix")
            run_auto_fix
            ;;
        "status")
            detect_rust_tools
            collect_dashboard_data
            cat "$DASHBOARD_FILE"
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Change to project directory
cd "$PROJECT_ROOT"

# Run main function
main "$@"
