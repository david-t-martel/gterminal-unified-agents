#!/bin/bash
# Integration Test Runner for GTerminal DevOps Stack
# Comprehensive testing across Rust and Python components

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_RESULTS_DIR="$PROJECT_ROOT/test-results"
TIMEOUT=300  # 5 minutes

# Test configuration
FILEWATCHER_PORT=8765
LSP_PORT=8767
DASHBOARD_PORT=8080
APP_PORT=8000
METRICS_PORT=8766

# State tracking
declare -A SERVICE_PIDS
declare -A TEST_RESULTS
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

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

# Utility functions
cleanup_services() {
    log "Cleaning up test services..."

    for service in "${!SERVICE_PIDS[@]}"; do
        local pid="${SERVICE_PIDS[$service]}"
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    # Clean up any remaining processes
    pkill -f "rust-filewatcher" || true
    pkill -f "lsp-server.py" || true
    pkill -f "dashboard_server.py" || true

    # Clean up test files
    rm -rf /tmp/test-filewatcher-* /tmp/test-workspace /tmp/integration-test-* || true
}

setup_test_environment() {
    log "Setting up integration test environment..."

    # Create test results directory
    mkdir -p "$TEST_RESULTS_DIR"

    # Create test workspace
    local test_workspace="/tmp/test-workspace"
    rm -rf "$test_workspace"
    mkdir -p "$test_workspace"/{src,tests,docs}

    # Create sample Python files for testing
    cat > "$test_workspace/src/example.py" << 'EOF'
import os
import sys
from typing import List, Dict, Any

def process_data(items: List[str]) -> Dict[str, Any]:
    """Process a list of items and return statistics."""
    if not items:
        return {}

    return {
        'count': len(items),
        'longest': max(items, key=len),
        'shortest': min(items, key=len)
    }

# Intentional issues for testing
def bad_function():
    unused_var = "test"  # F841: Local variable is assigned but never used
    print("Hello world")

class BadClass:
    def __init__(self):
        pass

    def method_without_self(self):  # Missing self usage
        return "test"
EOF

    # Create test configuration
    cat > "$test_workspace/pyproject.toml" << 'EOF'
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "D", "UP", "ANN", "BLE", "B", "A", "C4", "DTZ", "T10", "ISC", "ICN", "G", "PIE", "T20", "PYI", "PT", "Q", "RSE", "RET", "SLF", "SIM", "TID", "TCH", "ARG", "PTH", "ERA", "PGH", "PL", "TRY", "FLY", "NPY", "RUF"]
ignore = ["D100", "D101", "D102", "D103", "D104", "D105"]

[tool.ruff.lint.pydocstyle]
convention = "google"
EOF

    # Create filewatcher test config
    cat > "/tmp/test-filewatcher-config.toml" << EOF
[server]
websocket_port = $FILEWATCHER_PORT
metrics_port = $METRICS_PORT
host = "127.0.0.1"

[watching]
paths = ["$test_workspace"]
recursive = true
ignore_patterns = [
    ".git/**",
    "__pycache__/**",
    "*.pyc",
    ".cache/**"
]

[performance]
batch_size = 50
debounce_ms = 100
max_events_per_second = 500
worker_threads = 2

[features]
metrics_enabled = true
health_check = true
compression = false

[logging]
level = "debug"
file = "/tmp/test-filewatcher.log"
EOF

    info "Test environment setup completed"
    echo "Test workspace: $test_workspace"
}

start_test_services() {
    log "Starting test services..."

    # Start Rust filewatcher
    info "Starting Rust filewatcher..."
    if [ -f "$PROJECT_ROOT/rust-filewatcher/target/release/rust-filewatcher" ]; then
        "$PROJECT_ROOT/rust-filewatcher/target/release/rust-filewatcher" \
            --config /tmp/test-filewatcher-config.toml &
        SERVICE_PIDS["filewatcher"]=$!
    else
        # Fallback to debug build
        (cd "$PROJECT_ROOT/rust-filewatcher" &&
         cargo run -- --config /tmp/test-filewatcher-config.toml) &
        SERVICE_PIDS["filewatcher"]=$!
    fi

    # Wait for filewatcher to start
    wait_for_service "Filewatcher" "localhost:$METRICS_PORT/health" 30

    # Start Ruff LSP server
    info "Starting Ruff LSP server..."
    export LSP_PORT METRICS_PORT
    export WORKSPACE_PATH="/tmp/test-workspace"
    python3 -c "
import sys
sys.path.append('$PROJECT_ROOT')
import asyncio
from gterminal.lsp.ruff_lsp_client import start_lsp_server

async def main():
    await start_lsp_server('$LSP_PORT', '$METRICS_PORT', '$WORKSPACE_PATH')

if __name__ == '__main__':
    asyncio.run(main())
" &
    SERVICE_PIDS["ruff-lsp"]=$!

    # Wait for LSP server to start
    wait_for_service "Ruff LSP" "localhost:$((METRICS_PORT + 2))/health" 30

    success "All test services started successfully"
}

wait_for_service() {
    local service_name="$1"
    local health_url="$2"
    local timeout="$3"

    local elapsed=0
    while [ $elapsed -lt "$timeout" ]; do
        if curl -s -f "http://$health_url" > /dev/null 2>&1; then
            success "$service_name is healthy"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
        echo -n "."
    done

    error "$service_name failed to start within ${timeout}s"
    return 1
}

# Test functions
test_filewatcher_basic() {
    local test_name="Filewatcher Basic Functionality"
    log "Running test: $test_name"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # Test health endpoint
    if ! curl -s -f "http://localhost:$METRICS_PORT/health" | grep -q "healthy"; then
        error "Filewatcher health check failed"
        TEST_RESULTS["$test_name"]="FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi

    # Test metrics endpoint
    if ! curl -s -f "http://localhost:$METRICS_PORT/metrics" | grep -q "rust_filewatcher"; then
        error "Filewatcher metrics not available"
        TEST_RESULTS["$test_name"]="FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi

    success "$test_name passed"
    TEST_RESULTS["$test_name"]="PASSED"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    return 0
}

test_filewatcher_websocket() {
    local test_name="Filewatcher WebSocket Connection"
    log "Running test: $test_name"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # Test WebSocket connection using Python
    python3 -c "
import asyncio
import websockets
import json
import sys

async def test_websocket():
    try:
        uri = f'ws://localhost:$FILEWATCHER_PORT'
        async with websockets.connect(uri, timeout=10) as websocket:
            print('WebSocket connected successfully')
            # Send a test message
            await websocket.send(json.dumps({'type': 'ping'}))

            # Wait for response (with timeout)
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            print(f'Received: {response}')
            return True
    except Exception as e:
        print(f'WebSocket test failed: {e}')
        return False

result = asyncio.run(test_websocket())
sys.exit(0 if result else 1)
"

    if [ $? -eq 0 ]; then
        success "$test_name passed"
        TEST_RESULTS["$test_name"]="PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        error "$test_name failed"
        TEST_RESULTS["$test_name"]="FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

test_filewatcher_events() {
    local test_name="Filewatcher File Change Detection"
    log "Running test: $test_name"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # Create a test file and monitor for events
    local test_file="/tmp/test-workspace/test_file.py"

    # Start WebSocket listener in background
    python3 -c "
import asyncio
import websockets
import json
import sys
import time

events_received = []

async def listen_for_events():
    try:
        uri = f'ws://localhost:$FILEWATCHER_PORT'
        async with websockets.connect(uri, timeout=10) as websocket:
            print('Listening for file change events...', flush=True)

            # Set timeout for receiving events
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=15)
                    event = json.loads(response)
                    events_received.append(event)
                    print(f'Event received: {event}', flush=True)

                    # Check if we got our test file event
                    if 'path' in event and '$test_file' in event['path']:
                        print('Target file event detected!', flush=True)
                        return True
            except asyncio.TimeoutError:
                print('Timeout waiting for events', flush=True)
                return len(events_received) > 0

    except Exception as e:
        print(f'WebSocket error: {e}', flush=True)
        return False

result = asyncio.run(listen_for_events())
print(f'Events received: {len(events_received)}', flush=True)
sys.exit(0 if result else 1)
" &
    local listener_pid=$!

    # Give listener time to connect
    sleep 2

    # Create/modify test file
    echo "print('Hello, World!')" > "$test_file"
    sleep 1
    echo "print('Modified file')" >> "$test_file"

    # Wait for listener to finish
    wait $listener_pid
    local result=$?

    if [ $result -eq 0 ]; then
        success "$test_name passed"
        TEST_RESULTS["$test_name"]="PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        error "$test_name failed"
        TEST_RESULTS["$test_name"]="FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi

    rm -f "$test_file"
}

test_ruff_lsp_integration() {
    local test_name="Ruff LSP Integration"
    log "Running test: $test_name"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # Test LSP health
    if ! curl -s -f "http://localhost:$((METRICS_PORT + 2))/health" | grep -q "healthy"; then
        error "Ruff LSP health check failed"
        TEST_RESULTS["$test_name"]="FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi

    # Test diagnostics endpoint
    local test_response
    test_response=$(curl -s -X POST "http://localhost:$((METRICS_PORT + 2))/diagnostics" \
        -H "Content-Type: application/json" \
        -d '{"file_path": "/tmp/test-workspace/src/example.py", "ai_suggestions": true}')

    if echo "$test_response" | jq -e '.issues | length > 0' > /dev/null 2>&1; then
        success "Ruff LSP detected code issues as expected"

        # Check if AI suggestions are present
        if echo "$test_response" | jq -e '.ai_suggestions | length > 0' > /dev/null 2>&1; then
            success "AI suggestions generated successfully"
        else
            warn "AI suggestions not generated (may be expected if no API key)"
        fi

        TEST_RESULTS["$test_name"]="PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        error "Ruff LSP did not detect expected code issues"
        TEST_RESULTS["$test_name"]="FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

test_service_integration() {
    local test_name="Service Integration Test"
    log "Running test: $test_name"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # Test that services can communicate
    # Create a file and verify that both filewatcher and LSP respond

    local test_file="/tmp/test-workspace/integration_test.py"
    cat > "$test_file" << 'EOF'
# Integration test file
def test_function():
    unused_variable = "test"  # This should trigger a warning
    print("Integration test")
EOF

    # Wait for file system events to propagate
    sleep 2

    # Check filewatcher detected the file
    local filewatcher_events
    filewatcher_events=$(curl -s "http://localhost:$METRICS_PORT/metrics" | grep -c "events_processed" || echo "0")

    # Check LSP can analyze the file
    local lsp_response
    lsp_response=$(curl -s -X POST "http://localhost:$((METRICS_PORT + 2))/diagnostics" \
        -H "Content-Type: application/json" \
        -d "{\"file_path\": \"$test_file\"}")

    local has_issues=false
    if echo "$lsp_response" | jq -e '.issues | length > 0' > /dev/null 2>&1; then
        has_issues=true
    fi

    if [ "$filewatcher_events" -gt 0 ] && [ "$has_issues" = true ]; then
        success "$test_name passed - Services are integrated properly"
        TEST_RESULTS["$test_name"]="PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        error "$test_name failed - Service integration issues detected"
        error "Filewatcher events: $filewatcher_events, LSP issues detected: $has_issues"
        TEST_RESULTS["$test_name"]="FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi

    rm -f "$test_file"
}

test_performance_metrics() {
    local test_name="Performance Metrics Collection"
    log "Running test: $test_name"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # Generate some load and check metrics
    log "Generating test load..."

    # Create multiple files quickly
    for i in {1..10}; do
        echo "print('Test file $i')" > "/tmp/test-workspace/perf_test_$i.py"
    done

    # Wait for processing
    sleep 3

    # Check filewatcher metrics
    local filewatcher_metrics
    filewatcher_metrics=$(curl -s "http://localhost:$METRICS_PORT/metrics")

    # Check LSP metrics
    local lsp_metrics
    lsp_metrics=$(curl -s "http://localhost:$((METRICS_PORT + 2))/metrics")

    local has_filewatcher_metrics=false
    local has_lsp_metrics=false

    if echo "$filewatcher_metrics" | grep -q "rust_filewatcher"; then
        has_filewatcher_metrics=true
    fi

    if echo "$lsp_metrics" | grep -q "processing_time_ms"; then
        has_lsp_metrics=true
    fi

    if [ "$has_filewatcher_metrics" = true ] && [ "$has_lsp_metrics" = true ]; then
        success "$test_name passed"
        TEST_RESULTS["$test_name"]="PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        error "$test_name failed"
        TEST_RESULTS["$test_name"]="FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi

    # Cleanup test files
    rm -f /tmp/test-workspace/perf_test_*.py
}

run_integration_tests() {
    log "Starting comprehensive integration tests..."

    # Setup test environment
    setup_test_environment

    # Start test services
    start_test_services

    # Run individual tests
    test_filewatcher_basic
    test_filewatcher_websocket
    test_filewatcher_events
    test_ruff_lsp_integration
    test_service_integration
    test_performance_metrics

    # Generate test report
    generate_test_report

    # Cleanup
    cleanup_services

    # Exit with appropriate code
    if [ $FAILED_TESTS -eq 0 ]; then
        success "All integration tests passed! 🎉"
        exit 0
    else
        error "$FAILED_TESTS out of $TOTAL_TESTS tests failed"
        exit 1
    fi
}

generate_test_report() {
    local report_file="$TEST_RESULTS_DIR/integration-test-report.json"
    local html_report="$TEST_RESULTS_DIR/integration-test-report.html"

    log "Generating test report..."

    # JSON report
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "summary": {
        "total_tests": $TOTAL_TESTS,
        "passed_tests": $PASSED_TESTS,
        "failed_tests": $FAILED_TESTS,
        "success_rate": $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)
    },
    "results": {
EOF

    local first=true
    for test_name in "${!TEST_RESULTS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$report_file"
        fi
        echo "        \"$test_name\": \"${TEST_RESULTS[$test_name]}\"" >> "$report_file"
    done

    cat >> "$report_file" << EOF
    },
    "environment": {
        "filewatcher_port": $FILEWATCHER_PORT,
        "lsp_port": $LSP_PORT,
        "metrics_port": $METRICS_PORT,
        "test_workspace": "/tmp/test-workspace"
    }
}
EOF

    # HTML report
    cat > "$html_report" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>GTerminal Integration Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 30px; }
        .metric { background: #f8f9fa; padding: 15px; border-radius: 4px; text-align: center; }
        .metric h3 { margin: 0; color: #666; }
        .metric .value { font-size: 2em; margin: 10px 0; }
        .passed { color: #28a745; }
        .failed { color: #dc3545; }
        .results table { width: 100%; border-collapse: collapse; }
        .results th, .results td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        .results th { background: #f8f9fa; }
        .status-passed { color: #28a745; font-weight: bold; }
        .status-failed { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 GTerminal Integration Test Report</h1>
            <p>Generated on: <span id="timestamp"></span></p>
        </div>

        <div class="summary">
            <div class="metric">
                <h3>Total Tests</h3>
                <div class="value" id="total-tests"></div>
            </div>
            <div class="metric">
                <h3>Passed</h3>
                <div class="value passed" id="passed-tests"></div>
            </div>
            <div class="metric">
                <h3>Failed</h3>
                <div class="value failed" id="failed-tests"></div>
            </div>
        </div>

        <div class="results">
            <h2>Test Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="results-body">
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Load and display results from JSON
        fetch('./integration-test-report.json')
            .then(response => response.json())
            .then(data => {
                document.getElementById('timestamp').textContent = data.timestamp;
                document.getElementById('total-tests').textContent = data.summary.total_tests;
                document.getElementById('passed-tests').textContent = data.summary.passed_tests;
                document.getElementById('failed-tests').textContent = data.summary.failed_tests;

                const tbody = document.getElementById('results-body');
                Object.entries(data.results).forEach(([testName, status]) => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${testName}</td>
                        <td><span class="status-${status.toLowerCase()}">${status}</span></td>
                    `;
                    tbody.appendChild(row);
                });
            });
    </script>
</body>
</html>
EOF

    success "Test report generated: $report_file"
    success "HTML report generated: $html_report"

    # Show summary
    cat << EOF

📊 Test Summary:
┌─────────────────────────────────────────┐
│  Total Tests: $TOTAL_TESTS                        │
│  Passed:      $PASSED_TESTS                        │
│  Failed:      $FAILED_TESTS                        │
│  Success Rate: $(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)%                   │
└─────────────────────────────────────────┘

EOF
}

# Handle script signals
trap cleanup_services EXIT INT TERM

# Main execution
case "${1:-run}" in
    "run")
        run_integration_tests
        ;;
    "setup")
        setup_test_environment
        ;;
    "cleanup")
        cleanup_services
        ;;
    "help"|"-h"|"--help")
        cat << 'EOF'
Integration Test Runner for GTerminal DevOps Stack

USAGE:
    integration-test-runner.sh [command]

COMMANDS:
    run      Run all integration tests (default)
    setup    Setup test environment only
    cleanup  Clean up test services and files
    help     Show this help message

DESCRIPTION:
    This script runs comprehensive integration tests across all
    GTerminal components including:

    - Rust filewatcher functionality
    - WebSocket communication
    - File change event detection
    - Ruff LSP server integration
    - AI-powered code analysis
    - Service integration
    - Performance metrics collection

    Test results are saved in ./test-results/ directory.

EOF
        ;;
    *)
        error "Unknown command: $1"
        echo "Use 'integration-test-runner.sh help' for usage information"
        exit 1
        ;;
esac
