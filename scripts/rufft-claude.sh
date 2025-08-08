#!/bin/bash
# rufft-claude.sh - Advanced Ruff LSP Integration with Claude AI
# Enhanced version with full LSP server capabilities and real-time diagnostics

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
CLAUDE_CLI="${CLAUDE_CLI:-claude}"
CLAUDE_MODEL="${CLAUDE_MODEL:-haiku}" # Fast model for development
RUST_EXEC="${RUST_EXEC:-/home/david/.local/bin/rust-exec}"
RUST_FS_MCP="${RUST_FS_MCP:-/home/david/.local/bin/rust-fs}"
RUST_MEMORY_MCP="${RUST_MEMORY_MCP:-/home/david/.local/bin/rust-memory}"
AST_GREP_BIN="${AST_GREP_BIN:-ast-grep}"

# MCP Integration
MCP_RUST_FS_AVAILABLE=false
MCP_RUST_MEMORY_AVAILABLE=false

# Check MCP availability
# if [[ -x "$RUST_FS_MCP" ]]; then
#     MCP_RUST_FS_AVAILABLE=true
#     log "✅ Rust-FS MCP server available for fast file operations"
# fi

# if [[ -x "$RUST_MEMORY_MCP" ]]; then
#     MCP_RUST_MEMORY_AVAILABLE=true
#     log "✅ Rust-Memory MCP server available for context persistence"
# fi

# LSP Integration
LSP_CLIENT_SCRIPT="${LSP_CLIENT_SCRIPT:-$(dirname "$0")/../gterminal/lsp/ruff_lsp_client.py}"
RUFF_LSP_PORT="${RUFF_LSP_PORT:-8767}"
FILEWATCHER_WS_PORT="${FILEWATCHER_WS_PORT:-8765}"
LSP_METRICS_FILE="/tmp/ruff-lsp-metrics.json"
LSP_PID_FILE="/tmp/rufft-claude-lsp.pid"

# Logging
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

lsp_log() {
    echo -e "${PURPLE}[LSP $(date +'%H:%M:%S')] $*${NC}" >&2
}

stream_log() {
    echo -e "${CYAN}[STREAM $(date +'%H:%M:%S')] $*${NC}" >&2
}

# Fast execute using rust-exec or rust-fs
fast_exec() {
    local cmd="$1"
    shift

    if [[ -x "$RUST_EXEC" ]]; then
        "$RUST_EXEC" "$cmd" "$@"
    else
        # Fallback to regular command
        "$cmd" "$@"
    fi
}

# MCP-enhanced file operations
mcp_read_file() {
    local file_path="$1"
    if [[ "$MCP_RUST_FS_AVAILABLE" == "true" ]]; then
        # Use rust-fs MCP for faster file reading
        "$RUST_FS_MCP" read "$file_path" 2>/dev/null || cat "$file_path"
    else
        cat "$file_path"
    fi
}

mcp_write_file() {
    local file_path="$1"
    local content="$2"
    if [[ "$MCP_RUST_FS_AVAILABLE" == "true" ]]; then
        # Use rust-fs MCP for atomic file writing
        echo "$content" | "$RUST_FS_MCP" write "$file_path" || echo "$content" > "$file_path"
    else
        echo "$content" > "$file_path"
    fi
}

mcp_find_files() {
    local pattern="$1"
    local directory="${2:-.}"
    if [[ "$MCP_RUST_FS_AVAILABLE" == "true" ]]; then
        # Use rust-fs MCP for faster file finding
        "$RUST_FS_MCP" find "$directory" --pattern "$pattern" 2>/dev/null || find "$directory" -name "$pattern" -type f
    else
        find "$directory" -name "$pattern" -type f
    fi
}

# Enhanced error categorization
categorize_ruff_errors() {
    local ruff_output="$1"
    local file_path="$2"

    # Extract error codes and categorize by severity and type
    local critical_errors=()
    local fixable_errors=()
    local style_errors=()
    local security_errors=()

    while IFS= read -r line; do
        if [[ "$line" =~ F[0-9]+ ]]; then
            critical_errors+=("$line")  # Syntax/import errors
        elif [[ "$line" =~ (E[0-9]+|W[0-9]+) ]]; then
            style_errors+=("$line")     # Style issues
        elif [[ "$line" =~ S[0-9]+ ]]; then
            security_errors+=("$line")  # Security issues
        elif [[ "$line" =~ (UP[0-9]+|SIM[0-9]+|PTH[0-9]+) ]]; then
            fixable_errors+=("$line")   # Auto-fixable modernization
        fi
    done <<< "$ruff_output"

    # Store categorized errors for intelligent processing
    if [[ "$MCP_RUST_MEMORY_AVAILABLE" == "true" ]]; then
        local context_key="ruff_errors_$(basename "$file_path" .py)"
        echo "{\"critical\": $(printf '%s\n' "${critical_errors[@]}" | jq -R . | jq -s .), \"fixable\": $(printf '%s\n' "${fixable_errors[@]}" | jq -R . | jq -s .), \"style\": $(printf '%s\n' "${style_errors[@]}" | jq -R . | jq -s .), \"security\": $(printf '%s\n' "${security_errors[@]}" | jq -R . | jq -s .)}" | "$RUST_MEMORY_MCP" store "$context_key" || true
    fi

    echo "CRITICAL: ${#critical_errors[@]}, FIXABLE: ${#fixable_errors[@]}, STYLE: ${#style_errors[@]}, SECURITY: ${#security_errors[@]}"
}

# LSP Server Management Functions
start_lsp_server() {
    local workspace="${1:-$(pwd)}"
    local config_file="${2:-}"

    lsp_log "Starting Ruff LSP server for workspace: $workspace"

    # Check if already running
    if is_lsp_running; then
        lsp_log "LSP server already running (PID: $(cat "$LSP_PID_FILE"))"
        return 0
    fi

    # Prepare arguments
    local args=(
        "--workspace" "$workspace"
        "--log-level" "INFO"
        "--metrics-file" "$LSP_METRICS_FILE"
    )

    if [[ -n "$config_file" && -f "$config_file" ]]; then
        args+=("--config" "$config_file")
    fi

    # Start LSP client in background
    python3 "$LSP_CLIENT_SCRIPT" "${args[@]}" > "/tmp/rufft-lsp.log" 2>&1 &
    local lsp_pid=$!
    echo "$lsp_pid" > "$LSP_PID_FILE"

    # Wait for startup
    sleep 2

    if is_lsp_running; then
        lsp_log "✅ LSP server started successfully (PID: $lsp_pid)"
        return 0
    else
        error "❌ Failed to start LSP server"
        return 1
    fi
}

stop_lsp_server() {
    if [[ -f "$LSP_PID_FILE" ]]; then
        local pid=$(cat "$LSP_PID_FILE")
        lsp_log "Stopping LSP server (PID: $pid)..."

        if kill -TERM "$pid" 2>/dev/null; then
            # Wait for graceful shutdown
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                # Force kill if still running
                kill -KILL "$pid" 2>/dev/null
            fi
        fi

        rm -f "$LSP_PID_FILE"
        lsp_log "✅ LSP server stopped"
    else
        lsp_log "No LSP server running"
    fi
}

is_lsp_running() {
    if [[ -f "$LSP_PID_FILE" ]]; then
        local pid=$(cat "$LSP_PID_FILE")
        kill -0 "$pid" 2>/dev/null
    else
        return 1
    fi
}

restart_lsp_server() {
    lsp_log "Restarting LSP server..."
    stop_lsp_server
    sleep 1
    start_lsp_server "$@"
}

# Real-time Diagnostic Streaming Functions
stream_diagnostics() {
    local target_file="${1:-}"
    local watch_mode="${2:-auto}"  # auto, manual, continuous

    stream_log "Starting diagnostic streaming for ${target_file:-all files}"

    # Ensure LSP server is running
    if ! is_lsp_running; then
        start_lsp_server
    fi

    case "$watch_mode" in
        "auto")
            stream_auto_diagnostics "$target_file"
            ;;
        "manual")
            stream_manual_diagnostics "$target_file"
            ;;
        "continuous")
            stream_continuous_diagnostics "$target_file"
            ;;
        *)
            error "Unknown watch mode: $watch_mode"
            return 1
            ;;
    esac
}

stream_auto_diagnostics() {
    local target_file="$1"
    stream_log "Auto-streaming diagnostics with filewatcher integration"

    # Connect to filewatcher WebSocket if available
    if nc -z localhost "$FILEWATCHER_WS_PORT" 2>/dev/null; then
        stream_log "Connecting to filewatcher on port $FILEWATCHER_WS_PORT"

        # Use WebSocket to get file change notifications
        python3 -c "
import asyncio
import websockets
import json
import sys
import os

async def handle_filewatcher():
    try:
        uri = f'ws://localhost:$FILEWATCHER_WS_PORT'
        async with websockets.connect(uri) as websocket:
            print('🔗 Connected to filewatcher', file=sys.stderr)

            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'file_changed':
                        file_path = data.get('path', '')
                        if file_path.endswith('.py'):
                            print(f'📝 File changed: {file_path}', file=sys.stderr)
                            # Trigger diagnostic refresh
                            os.system(f'echo \"{file_path}\" >> /tmp/rufft-changed-files.txt')
                except Exception as e:
                    print(f'Error processing message: {e}', file=sys.stderr)

    except Exception as e:
        print(f'Failed to connect to filewatcher: {e}', file=sys.stderr)
        return 1

asyncio.run(handle_filewatcher())
" &
        local ws_pid=$!
        echo "$ws_pid" > "/tmp/rufft-ws.pid"

        stream_log "✅ Auto-diagnostic streaming started (WebSocket PID: $ws_pid)"
    else
        # Fallback to file system watching
        stream_log "Filewatcher not available, using inotify fallback"
        stream_inotify_diagnostics "$target_file"
    fi
}

stream_inotify_diagnostics() {
    local target_file="$1"
    local watch_dir="${target_file:-.}"

    stream_log "Setting up inotify watch on $watch_dir"

    # Use inotifywait to watch for file changes
    inotifywait -m -r -e modify,create,delete --format '%w%f %e' "$watch_dir" 2>/dev/null | while read file event; do
        if [[ "$file" =~ \.py$ ]]; then
            stream_log "📝 File $event: $file"
            # Trigger diagnostic check
            echo "$file" >> /tmp/rufft-changed-files.txt
        fi
    done &

    echo $! > "/tmp/rufft-inotify.pid"
    stream_log "✅ inotify diagnostic streaming started"
}

stream_manual_diagnostics() {
    local target_file="$1"
    stream_log "Manual diagnostic streaming mode"

    if [[ -n "$target_file" ]]; then
        get_file_diagnostics "$target_file"
    else
        # Get diagnostics for all Python files
        find . -name "*.py" -type f | while read -r pyfile; do
            get_file_diagnostics "$pyfile"
        done
    fi
}

stream_continuous_diagnostics() {
    local target_file="$1"
    stream_log "Continuous diagnostic streaming every 5 seconds"

    while true; do
        if [[ -n "$target_file" ]]; then
            get_file_diagnostics "$target_file"
        else
            # Process any changed files
            if [[ -f /tmp/rufft-changed-files.txt ]]; then
                while read -r changed_file; do
                    get_file_diagnostics "$changed_file"
                done < /tmp/rufft-changed-files.txt
                > /tmp/rufft-changed-files.txt  # Clear the file
            fi
        fi
        sleep 5
    done
}

get_file_diagnostics() {
    local file_path="$1"

    if [[ ! -f "$file_path" ]]; then
        return 1
    fi

    stream_log "Getting diagnostics for: $file_path"

    # Run ruff check and capture JSON output
    local ruff_output
    ruff_output=$(ruff check --output-format=json "$file_path" 2>/dev/null | jq -c '.')

    if [[ -n "$ruff_output" && "$ruff_output" != "[]" ]]; then
        local issue_count
        issue_count=$(echo "$ruff_output" | jq length)
        stream_log "📊 Found $issue_count issues in $(basename "$file_path")"

        # Pretty print diagnostics
        echo "$ruff_output" | jq -r '.[] | "\(.filename):\(.location.row):\(.location.column) [\(.code)] \(.message)"' | while read -r diagnostic; do
            echo -e "  ${YELLOW}⚠️  $diagnostic${NC}"
        done

        # Generate AI suggestions if enabled
        if command -v "$CLAUDE_CLI" &> /dev/null; then
            generate_ai_suggestions "$file_path" "$ruff_output"
        fi
    else
        stream_log "✅ No issues found in $(basename "$file_path")"
    fi
}

# AI-Powered Fix Suggestion Functions
generate_ai_suggestions() {
    local file_path="$1"
    local ruff_json="$2"

    lsp_log "🤖 Generating enhanced AI suggestions for $file_path"

    if ! command -v "$CLAUDE_CLI" &> /dev/null; then
        warn "Claude CLI not available, skipping AI suggestions"
        return 1
    fi

    # Get file content using MCP if available
    local file_content
    file_content=$(mcp_read_file "$file_path")

    # Categorize errors for intelligent processing
    local error_summary
    error_summary=$(categorize_ruff_errors "$ruff_json" "$file_path")

    # Extract specific error details with better categorization
    local critical_issues
    critical_issues=$(echo "$ruff_json" | jq -r '.[] | select(.code | startswith("F")) | "Line \(.location.row): \(.code) - \(.message)"' | head -5)

    local fixable_issues
    fixable_issues=$(echo "$ruff_json" | jq -r '.[] | select(.code | test("UP|SIM|PTH|ERA|ARG")) | "Line \(.location.row): \(.code) - \(.message)"' | head -5)

    local security_issues
    security_issues=$(echo "$ruff_json" | jq -r '.[] | select(.code | startswith("S")) | "Line \(.location.row): \(.code) - \(.message)"' | head -3)

    if [[ -z "$critical_issues$fixable_issues$security_issues" ]]; then
        return 0
    fi

    # Build intelligent, context-aware prompt
    local prompt="🔧 PYTHON CODE ANALYSIS & MODERNIZATION

FILE: $file_path
ERROR SUMMARY: $error_summary

CODEBASE CONTEXT: gterminal unified agents project
- Target: Python 3.12+ with modern syntax
- Framework: Pydantic V2 (use Field(), ConfigDict, model_validate)
- Style: Async/await preferred, comprehensive type hints
- Performance: Critical paths need optimization

🚨 CRITICAL FIXES NEEDED:
$critical_issues

⚡ MODERNIZATION OPPORTUNITIES:
$fixable_issues

🔒 SECURITY IMPROVEMENTS:
$security_issues

TASK: Provide EXACT before/after code fixes that:
1. Fix syntax/import errors immediately
2. Modernize to Python 3.12+ features (match/case, improved typing)
3. Update Pydantic V1 → V2 patterns
4. Convert sync→async where beneficial
5. Add missing type hints and error handling
6. Use pathlib instead of os.path

FORMAT for each fix:
``$(python
# Line X: BEFORE
old_code_line_here

# Line X: AFTER
new_code_line_here
# REASON: Explanation of improvement
)``

Focus on ready-to-apply fixes that reduce error count immediately."

    # Call Claude with enhanced prompt and longer timeout
    local claude_response
    claude_response=$(timeout 60s "$CLAUDE_CLI" --model "$CLAUDE_MODEL" "$prompt" 2>/dev/null)

    if [[ $? -eq 0 && -n "$claude_response" ]]; then
        echo -e "\n${PURPLE}🤖 Enhanced AI Analysis for $(basename "$file_path"):${NC}"
        echo -e "${CYAN}$claude_response${NC}\n"

        # Save enhanced suggestions with metadata
        local suggestions_file
        suggestions_file="/tmp/rufft-ai-suggestions-$(basename "$file_path" .py).txt"
        cat > "$suggestions_file" <<EOF
=== ENHANCED AI CODE ANALYSIS ===
File: $file_path
Generated: $(date)
Error Summary: $error_summary
Claude Model: $CLAUDE_MODEL
Context: gterminal unified agents (Python 3.12+, Pydantic V2)

$claude_response
EOF

        # Store in MCP memory for cross-session persistence
        if [[ "$MCP_RUST_MEMORY_AVAILABLE" == "true" ]]; then
            local memory_key
            memory_key="ai_suggestions_$(basename "$file_path" .py)"
            echo "$claude_response" | "$RUST_MEMORY_MCP" store "$memory_key" 2>/dev/null || true
        fi

        lsp_log "💾 Enhanced AI suggestions saved to: $suggestions_file"

        # Extract and display actionable fixes count
        local fix_count
        fix_count=$(echo "$claude_response" | grep -c "# Line.*: AFTER" || echo "0")
        if [[ "$fix_count" -gt 0 ]]; then
            success "✨ Generated $fix_count actionable code fixes"
        fi
    else
        warn "Claude AI suggestion generation failed or timed out"
    fi
}

# LSP Performance and Health Monitoring
lsp_health_check() {
    lsp_log "Performing LSP health check..."

    local health_status="healthy"
    local issues=()

    # Check if LSP server is running
    if ! is_lsp_running; then
        health_status="unhealthy"
        issues+=("LSP server not running")
    fi

    # Check LSP metrics file
    if [[ -f "$LSP_METRICS_FILE" ]]; then
        local last_update
        last_update=$(stat -c %Y "$LSP_METRICS_FILE" 2>/dev/null)
        local current_time
        current_time=$(date +%s)

        if [[ $((current_time - last_update)) -gt 300 ]]; then  # 5 minutes
            health_status="degraded"
            issues+=("Metrics not updated in 5+ minutes")
        fi

        # Check response times
        local avg_response_time
        avg_response_time=$(jq -r '.avg_response_time_ms // 0' "$LSP_METRICS_FILE" 2>/dev/null)

        if (( $(echo "$avg_response_time > 1000" | bc -l) )); then
            health_status="degraded"
            issues+=("High response time: ${avg_response_time}ms")
        fi
    else
        issues+=("No metrics file found")
    fi

    # Display health status
    case "$health_status" in
        "healthy")
            lsp_log "✅ LSP server is healthy"
            ;;
        "degraded")
            warn "⚠️ LSP server is degraded: ${issues[*]}"
            ;;
        "unhealthy")
            error "❌ LSP server is unhealthy: ${issues[*]}"
            ;;
    esac

    # Show performance metrics if available
    if [[ -f "$LSP_METRICS_FILE" ]]; then
        lsp_log "📊 Performance metrics:"
        jq -r '"  Requests: \(.requests_sent // 0), Diagnostics: \(.diagnostics_received // 0), Avg Response: \(.avg_response_time_ms // 0)ms"' "$LSP_METRICS_FILE"
    fi

    return $([ "$health_status" = "healthy" ] && echo 0 || echo 1)
}

stop_all_streaming() {
    lsp_log "Stopping all diagnostic streaming processes..."

    # Stop WebSocket connection
    if [[ -f "/tmp/rufft-ws.pid" ]]; then
        local ws_pid=$(cat "/tmp/rufft-ws.pid")
        kill "$ws_pid" 2>/dev/null
        rm -f "/tmp/rufft-ws.pid"
    fi

    # Stop inotify watcher
    if [[ -f "/tmp/rufft-inotify.pid" ]]; then
        local inotify_pid=$(cat "/tmp/rufft-inotify.pid")
        kill "$inotify_pid" 2>/dev/null
        rm -f "/tmp/rufft-inotify.pid"
    fi

    # Clean up temp files
    rm -f /tmp/rufft-changed-files.txt

    lsp_log "✅ All streaming processes stopped"
}

# Auto-fix functions
fix_ruff() {
    log "Running ruff auto-fix..."
    if command -v ruff &> /dev/null; then
        ruff check --fix --unsafe-fixes .
        ruff format .
        success "Ruff fixes applied"
    else
        warn "ruff not found"
    fi
}

fix_mypy() {
    log "Running mypy checks..."
    if command -v mypy &> /dev/null; then
        mypy . || warn "MyPy issues found - may need manual fixes"
    else
        warn "mypy not found"
    fi
}

fix_ast_grep() {
    log "Running AST-grep auto-fixes..."
    if [[ -f .ast-grep/sgconfig.yml && -x "$AST_GREP_BIN" ]]; then
        # Apply AST-grep rules with auto-fix
        "$AST_GREP_BIN" scan --update-all . || warn "AST-grep fixes may need manual review"
        success "AST-grep rules applied"
    else
        warn "AST-grep not configured or binary not found"
    fi
}

# Claude operations
claude_analyze() {
    local prompt="$1"
    log "Analyzing with Claude ($CLAUDE_MODEL)..."

    # Use fast model for quick analysis
    "$CLAUDE_CLI" --model "$CLAUDE_MODEL" "$prompt"
}

claude_fix() {
    local file="$1"
    local issue="$2"
    log "Asking Claude to fix: $issue in $file"

    local prompt="Fix this issue in $file: $issue. Provide only the corrected code."
    claude_analyze "$prompt"
}

# Dashboard update
update_dashboard() {
    log "Updating development dashboard..."

    # Generate status report
    local status_file="dashboard_status.json"
    cat > "$status_file" <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "project": "gterminal",
    "status": {
        "ruff_issues": $(ruff check --output-format json . 2>/dev/null | jq length || echo 0),
        "mypy_issues": $(mypy --show-error-codes . 2>/dev/null | wc -l || echo 0),
        "ast_grep_issues": $(ast-grep scan --reporter json . 2>/dev/null | jq length || echo 0),
        "test_coverage": $(coverage report --format json 2>/dev/null | jq -r '.totals.percent_covered' || echo 0)
    }
}
EOF

    success "Dashboard updated: $status_file"
}

# Main auto-fix pipeline
# Enhanced auto-fix pipeline with intelligent batching
auto_fix_all() {
    log "🚀 Starting enhanced auto-fix pipeline..."

    # Pre-fix status
    update_dashboard

    # Get list of Python files to process
    local python_files
    readarray -t python_files < <(mcp_find_files "*.py" ".")

    local total_files=${#python_files[@]}
    log "Found $total_files Python files to analyze"

    # Phase 1: Quick ruff auto-fixes (parallel processing)
    log "Phase 1: Running ruff auto-fixes..."
    if command -v ruff &> /dev/null; then
        ruff check --fix --unsafe-fixes . || warn "Some ruff auto-fixes failed"
        success "✅ Ruff auto-fixes completed"
    fi

    # Phase 2: Categorize remaining errors by severity
    log "Phase 2: Categorizing remaining errors..."
    local critical_files=()
    local fixable_files=()
    local style_files=()

    for file in "${python_files[@]}"; do
        if [[ ! -f "$file" ]]; then continue; fi

        local ruff_output
        ruff_output=$(ruff check --output-format=json "$file" 2>/dev/null)

        if [[ -n "$ruff_output" && "$ruff_output" != "[]" ]]; then
            # Check error types to prioritize
            local has_critical
            has_critical=$(echo "$ruff_output" | jq -r 'any(.code | startswith("F"))')
            local has_fixable
            has_fixable=$(echo "$ruff_output" | jq -r 'any(.code | test("UP|SIM|PTH|ERA"))')

            if [[ "$has_critical" == "true" ]]; then
                critical_files+=("$file")
            elif [[ "$has_fixable" == "true" ]]; then
                fixable_files+=("$file")
            else
                style_files+=("$file")
            fi
        fi
    done

    # Phase 3: Process critical files first (sequential for accuracy)
    if [[ ${#critical_files[@]} -gt 0 ]]; then
        log "Phase 3: Processing ${#critical_files[@]} files with critical errors..."
        for file in "${critical_files[@]}"; do
            local ruff_json
            ruff_json=$(ruff check --output-format=json "$file" 2>/dev/null)
            if [[ -n "$ruff_json" && "$ruff_json" != "[]" ]]; then
                log "🔧 Analyzing critical issues in $(basename "$file")"
                generate_ai_suggestions "$file" "$ruff_json"
            fi
        done
        success "✅ Critical files analysis completed"
    fi

    # Phase 4: Batch process fixable files (parallel)
    if [[ ${#fixable_files[@]} -gt 0 ]]; then
        log "Phase 4: Batch processing ${#fixable_files[@]} files with modernization opportunities..."

        # Process in parallel batches of 5
        local batch_size=5
        for ((i=0; i<${#fixable_files[@]}; i+=batch_size)); do
            local batch=("${fixable_files[@]:i:batch_size}")
            log "Processing batch: ${batch[*]}"

            # Start background processes for batch
            for file in "${batch[@]}"; do
                (
                    local ruff_json
                    ruff_json=$(ruff check --output-format=json "$file" 2>/dev/null)
                    if [[ -n "$ruff_json" && "$ruff_json" != "[]" ]]; then
                        generate_ai_suggestions "$file" "$ruff_json"
                    fi
                ) &
            done
            wait  # Wait for batch to complete
        done
        success "✅ Fixable files batch processing completed"
    fi    # Phase 5: AST-grep structural improvements
    log "Phase 5: Running structural analysis..."
    fix_ast_grep

    # Phase 6: Final validation and metrics
    log "Phase 6: Final validation..."
    local final_errors
    final_errors=$(ruff check --output-format=json . 2>/dev/null | jq length)

    # Update dashboard with results
    update_dashboard

    log "🎯 Auto-fix pipeline completed"
    success "📊 Remaining errors: $final_errors"

    # Store session summary in MCP memory
    if [[ "$MCP_RUST_MEMORY_AVAILABLE" == "true" ]]; then
        local session_summary="{\"timestamp\": \"$(date -Iseconds)\", \"total_files\": $total_files, \"critical_files\": ${#critical_files[@]}, \"fixable_files\": ${#fixable_files[@]}, \"final_errors\": $final_errors}"
        echo "$session_summary" | "$RUST_MEMORY_MCP" store "autofix_session_$(date +%s)" 2>/dev/null || true
    fi

    success "Auto-fix pipeline completed"
}

# Python 3.12+ modernization with Pydantic V2 migration
modernize_python312() {
    local file_path="$1"
    log "🚀 Modernizing $file_path for Python 3.12+ with Pydantic V2..."

    if [[ ! -f "$file_path" ]]; then
        warn "File not found: $file_path"
        return 1
    fi

    local file_content
    file_content=$(mcp_read_file "$file_path")

    # Detect modernization opportunities
    local needs_pydantic_v2=false
    local needs_py312_features=false
    local needs_async_improvements=false

    if echo "$file_content" | grep -q "from pydantic import.*BaseModel"; then
        needs_pydantic_v2=true
    fi

    if echo "$file_content" | grep -q "Union\|Optional"; then
        needs_py312_features=true
    fi

    if echo "$file_content" | grep -q "async def\|await"; then
        needs_async_improvements=true
    fi

    # Generate targeted modernization prompt
    local modernization_prompt="🚀 PYTHON 3.12+ MODERNIZATION TASK

TARGET FILE: $file_path

MODERNIZATION OBJECTIVES:
1. 🔄 Pydantic V2 Migration (if applicable): $([ "$needs_pydantic_v2" = true ] && echo "REQUIRED" || echo "N/A")
2. 🐍 Python 3.12+ Features: $([ "$needs_py312_features" = true ] && echo "REQUIRED" || echo "N/A")
3. ⚡ Async/Await Improvements: $([ "$needs_async_improvements" = true ] && echo "REQUIRED" || echo "N/A")

SPECIFIC MODERNIZATION PATTERNS:
- Replace Union[X, Y] with X | Y syntax
- Replace Optional[X] with X | None
- Use match/case statements for complex conditionals
- Update Pydantic V1 patterns to V2:
  * BaseSettings → BaseSettings from pydantic_settings
  * Config class → model_config = ConfigDict()
  * validator → field_validator
  * root_validator → model_validator
- Use new typing features (Self, TypeVar defaults)
- Optimize async/await patterns with exception groups
- Add comprehensive type hints using modern syntax

CODEBASE CONTEXT:
- This is part of the gterminal AI development environment
- Focus on performance, type safety, and maintainability
- Ensure compatibility with VertexAI function calling
- Maintain MCP protocol compatibility

Please analyze and modernize this file with the above patterns. Provide the complete modernized file content.

CURRENT FILE CONTENT:
$file_content"

    # Use enhanced Claude analysis
    log "Generating Python 3.12+ modernization suggestions..."
    echo "$modernization_prompt" | claude > "/tmp/modernization_$(basename "$file_path").md"

    success "✅ Modernization analysis saved to /tmp/modernization_$(basename "$file_path").md"

    # Store modernization context in MCP memory
    if [[ "$MCP_RUST_MEMORY_AVAILABLE" == "true" ]]; then
        local context_key="py312_modernization_$(basename "$file_path" .py)"
        echo "$modernization_prompt" | "$RUST_MEMORY_MCP" store "$context_key" 2>/dev/null || true
        log "💾 Modernization context stored in MCP memory: $context_key"
    fi
}
server_mode() {
    log "Starting rufft-claude server mode on port 8766..."

    # Simple HTTP server for triggering fixes
    python3 -c "
import http.server
import socketserver
import subprocess
import json
from urllib.parse import parse_qs, urlparse

class AutoFixHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            with open('dashboard_status.json', 'r') as f:
                self.wfile.write(f.read().encode())
        elif self.path == '/fix':
            subprocess.run(['$0', 'auto-fix'], check=False)
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Auto-fix triggered')
        else:
            self.send_response(404)
            self.end_headers()

with socketserver.TCPServer(('', 8766), AutoFixHandler) as httpd:
    print('Serving at port 8766')
    httpd.serve_forever()
" &

    local server_pid=$!
    echo "$server_pid" > rufft-claude.pid

    log "Server running at http://localhost:8766 (PID: $server_pid)"
    log "Endpoints: /status, /fix"
}

# Usage
usage() {
    cat << EOF
Usage: $0 <command> [args...]

Traditional Commands:
  auto-fix          Run comprehensive auto-fix pipeline
  modernize <file>  Modernize single file to Python 3.12+ with Pydantic V2
  batch-modernize   Modernize all Python files to 3.12+ with Pydantic V2
  accelerate        Full auto-fix + modernization pipeline
  ruff              Apply ruff fixes only
  mypy              Run mypy checks
  ast-grep          Apply AST-grep fixes
  analyze <prompt>  Analyze with Claude
  fix <file> <issue> Fix specific issue with Claude
  server            Start server mode on port 8766
  dashboard         Update dashboard status

🚀 LSP Integration Commands:
  lsp-start [workspace] [config]    Start Ruff LSP server
  lsp-stop                          Stop Ruff LSP server
  lsp-restart [workspace] [config]  Restart Ruff LSP server
  lsp-status                        Check LSP server status
  lsp-health                        Perform LSP health check

📊 Real-time Diagnostic Streaming:
  stream [file] [mode]              Start diagnostic streaming
    Modes: auto (default), manual, continuous
  stream-stop                       Stop all streaming processes
  diagnostics <file>                Get diagnostics for specific file

🤖 AI-Powered Suggestions:
  ai-suggest <file>                 Generate AI suggestions for file
  ai-batch [directory]              Batch AI analysis for directory

🔧 Advanced Features:
  metrics                           Show LSP performance metrics
  cleanup                           Clean up all temporary files

Environment Variables:
  CLAUDE_CLI              Claude CLI command (default: claude)
  CLAUDE_MODEL            Claude model to use (default: haiku)
  RUST_EXEC               Rust executor path
  AST_GREP_BIN            AST-grep binary path
  LSP_CLIENT_SCRIPT       Path to LSP client script
  RUFF_LSP_PORT           Ruff LSP server port (default: 8767)
  FILEWATCHER_WS_PORT     Filewatcher WebSocket port (default: 8765)

Examples:
  $0 lsp-start                              # Start LSP server
  $0 stream myfile.py auto                  # Auto-stream diagnostics
  $0 ai-suggest src/main.py                 # Get AI suggestions
  $0 lsp-health                             # Check LSP health
  $0 auto-fix                               # Traditional auto-fix
EOF
}

# Main command dispatcher
main() {
    case "${1:-help}" in
        # Traditional commands
        auto-fix)
            auto_fix_all
            ;;
        modernize)
            if [[ -n "$2" ]]; then
                modernize_python312 "$2"
            else
                log "Usage: $0 modernize <file_path>"
                exit 1
            fi
            ;;
        batch-modernize)
            log "🚀 Starting batch Python 3.12+ modernization..."
            readarray -t python_files < <(mcp_find_files "*.py" ".")
            for file in "${python_files[@]}"; do
                if [[ -f "$file" ]]; then
                    modernize_python312 "$file"
                fi
            done
            success "✅ Batch modernization completed"
            ;;
        accelerate)
            log "⚡ Starting accelerated modernization pipeline..."
            # Combined auto-fix and modernization
            auto_fix_all
            log "Starting Python 3.12+ modernization..."
            readarray -t python_files < <(mcp_find_files "*.py" ".")
            for file in "${python_files[@]}"; do
                if [[ -f "$file" ]]; then
                    modernize_python312 "$file"
                fi
            done
            success "🎯 Accelerated pipeline completed"
            ;;
        ruff)
            fix_ruff
            ;;
        mypy)
            fix_mypy
            ;;
        ast-grep)
            fix_ast_grep
            ;;
        analyze)
            shift
            claude_analyze "$*"
            ;;
        fix)
            claude_fix "$2" "$3"
            ;;
        server)
            server_mode
            ;;
        dashboard)
            update_dashboard
            ;;

        # LSP Integration commands
        lsp-start)
            start_lsp_server "$2" "$3"
            ;;
        lsp-stop)
            stop_lsp_server
            ;;
        lsp-restart)
            restart_lsp_server "$2" "$3"
            ;;
        lsp-status)
            if is_lsp_running; then
                lsp_log "✅ LSP server is running (PID: $(cat "$LSP_PID_FILE"))"
            else
                lsp_log "❌ LSP server is not running"
                exit 1
            fi
            ;;
        lsp-health)
            lsp_health_check
            ;;

        # Real-time diagnostic streaming
        stream)
            stream_diagnostics "$2" "${3:-auto}"
            ;;
        stream-stop)
            stop_all_streaming
            ;;
        diagnostics)
            if [[ -z "$2" ]]; then
                error "File path required for diagnostics command"
                exit 1
            fi
            get_file_diagnostics "$2"
            ;;

        # AI-powered suggestions
        ai-suggest)
            if [[ -z "$2" ]]; then
                error "File path required for ai-suggest command"
                exit 1
            fi
            # Run ruff first to get issues
            local ruff_json
            ruff_json=$(ruff check --output-format=json "$2" 2>/dev/null | jq -c '.')
            if [[ -n "$ruff_json" && "$ruff_json" != "[]" ]]; then
                generate_ai_suggestions "$2" "$ruff_json"
            else
                lsp_log "No issues found in $2, no AI suggestions needed"
            fi
            ;;
        ai-batch)
            local target_dir="${2:-.}"
            lsp_log "Running batch AI analysis on $target_dir"
            find "$target_dir" -name "*.py" -type f | while read -r pyfile; do
                local ruff_json
                ruff_json=$(ruff check --output-format=json "$pyfile" 2>/dev/null | jq -c '.')
                if [[ -n "$ruff_json" && "$ruff_json" != "[]" ]]; then
                    generate_ai_suggestions "$pyfile" "$ruff_json"
                fi
            done
            ;;

        # Advanced features
        metrics)
            if [[ -f "$LSP_METRICS_FILE" ]]; then
                lsp_log "📊 LSP Performance Metrics:"
                jq '.' "$LSP_METRICS_FILE"
            else
                warn "No metrics file found at $LSP_METRICS_FILE"
            fi
            ;;
        cleanup)
            lsp_log "🧹 Cleaning up temporary files..."
            rm -f /tmp/rufft-*.{pid,txt,json,log}
            rm -f /tmp/rufft-ai-suggestions-*.txt
            success "Cleanup complete"
            ;;

        help|--help|-h)
            usage
            ;;
        *)
            error "Unknown command: $1"
            echo ""
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
