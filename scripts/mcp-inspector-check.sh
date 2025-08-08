#!/bin/bash
"""
MCP Inspector Compliance Check Script for Gterminal

Runs MCP Inspector validation on configuration files to ensure compliance.
Adapted from my-fullstack-agent for gterminal project structure.
"""

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSPECTOR_TIMEOUT=30
TEMP_DIR="/tmp/mcp-inspector-$$"
LOG_FILE="$TEMP_DIR/inspector.log"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if MCP Inspector is available
check_inspector() {
    print_status "$BLUE" "🔍 Checking MCP Inspector availability..."

    if command -v npx >/dev/null 2>&1; then
        print_status "$GREEN" "  ✅ npx is available"
    else
        print_status "$RED" "  ❌ npx not found - please install Node.js"
        return 1
    fi

    # Test MCP Inspector
    if timeout 10 npx @modelcontextprotocol/inspector --version >/dev/null 2>&1; then
        print_status "$GREEN" "  ✅ MCP Inspector is available"
        return 0
    else
        print_status "$YELLOW" "  ⚠️  MCP Inspector not installed - installing..."
        if npm install -g @modelcontextprotocol/inspector 2>/dev/null; then
            print_status "$GREEN" "  ✅ MCP Inspector installed successfully"
            return 0
        else
            print_status "$RED" "  ❌ Failed to install MCP Inspector"
            return 1
        fi
    fi
}

# Function to validate a single MCP configuration file
validate_config() {
    local config_file=$1
    local config_name=$(basename "$config_file" .json)

    print_status "$BLUE" "📋 Validating configuration: $config_name"

    if [[ ! -f "$config_file" ]]; then
        print_status "$RED" "  ❌ Configuration file not found: $config_file"
        return 1
    fi

    # Check JSON syntax first
    if ! jq empty "$config_file" 2>/dev/null; then
        print_status "$RED" "  ❌ Invalid JSON syntax in $config_file"
        return 1
    fi

    print_status "$GREEN" "  ✅ JSON syntax is valid"

    # Extract server names from config
    local servers
    servers=$(jq -r '.mcpServers | keys[]' "$config_file" 2>/dev/null || echo "")

    if [[ -z "$servers" ]]; then
        print_status "$YELLOW" "  ⚠️  No MCP servers found in configuration"
        return 0
    fi

    local server_count
    server_count=$(echo "$servers" | wc -l)
    print_status "$BLUE" "  📦 Found $server_count MCP servers"

    # Validate each server with MCP Inspector
    local overall_success=0

    while IFS= read -r server_name; do
        print_status "$BLUE" "    🔧 Testing server: $server_name"

        # Create temporary config with just this server
        local temp_config="$TEMP_DIR/temp_${server_name}.json"
        jq ".mcpServers = {\"$server_name\": .mcpServers.\"$server_name\"}" "$config_file" > "$temp_config"

        # Run MCP Inspector validation with shorter timeout for gterminal
        local inspector_output="$TEMP_DIR/inspector_${server_name}.log"

        if timeout $INSPECTOR_TIMEOUT npx @modelcontextprotocol/inspector \
            --config "$temp_config" \
            --server "$server_name" \
            --method tools/list \
            --timeout 10000 > "$inspector_output" 2>&1; then

            print_status "$GREEN" "      ✅ Inspector validation passed"

            # Check for specific compliance indicators
            if grep -q "tools/list" "$inspector_output"; then
                print_status "$GREEN" "      ✅ Tools endpoint responsive"
            fi

            if grep -q "error" "$inspector_output"; then
                print_status "$YELLOW" "      ⚠️  Warnings in inspector output"
                grep "error" "$inspector_output" | head -3 | sed 's/^/        /'
            fi

        else
            print_status "$RED" "      ❌ Inspector validation failed"
            overall_success=1

            # Show error details
            if [[ -s "$inspector_output" ]]; then
                print_status "$RED" "      Error details:"
                tail -n 5 "$inspector_output" | sed 's/^/        /'
            fi
        fi

        # Test additional MCP methods if available
        test_additional_methods "$temp_config" "$server_name"

    done <<< "$servers"

    return $overall_success
}

# Function to test additional MCP methods
test_additional_methods() {
    local config_file=$1
    local server_name=$2

    local methods=("resources/list" "prompts/list")

    for method in "${methods[@]}"; do
        local output_file="$TEMP_DIR/method_${server_name}_$(echo $method | tr '/' '_').log"

        if timeout 10 npx @modelcontextprotocol/inspector \
            --config "$config_file" \
            --server "$server_name" \
            --method "$method" \
            --timeout 5000 > "$output_file" 2>&1; then

            print_status "$GREEN" "      ✅ Method $method available"
        else
            # Not an error - just not implemented
            print_status "$BLUE" "      ℹ️  Method $method not implemented"
        fi
    done
}

# Function to generate compliance report
generate_report() {
    local config_files=("$@")
    local report_file="mcp-inspector-report.json"

    print_status "$BLUE" "📊 Generating compliance report..."

    {
        echo "{"
        echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
        echo "  \"project\": \"gterminal\","
        echo "  \"configurations\": ["

        local first=true
        for config_file in "${config_files[@]}"; do
            if [[ ! $first ]]; then echo ","; fi
            first=false

            local config_name=$(basename "$config_file" .json)
            echo "    {"
            echo "      \"name\": \"$config_name\","
            echo "      \"file\": \"$config_file\","
            echo "      \"servers\": ["

            local servers
            servers=$(jq -r '.mcpServers | keys[]' "$config_file" 2>/dev/null || echo "")

            local server_first=true
            while IFS= read -r server_name; do
                if [[ -z "$server_name" ]]; then continue; fi

                if [[ ! $server_first ]]; then echo ","; fi
                server_first=false

                echo "        {"
                echo "          \"name\": \"$server_name\","

                # Check if validation logs exist
                local log_file="$TEMP_DIR/inspector_${server_name}.log"
                if [[ -f "$log_file" ]]; then
                    local success=$(grep -q "error" "$log_file" && echo "false" || echo "true")
                    echo "          \"compliance\": $success,"
                    echo "          \"log_file\": \"$log_file\""
                else
                    echo "          \"compliance\": false,"
                    echo "          \"log_file\": null"
                fi

                echo "        }"
            done <<< "$servers"

            echo "      ]"
            echo "    }"
        done

        echo "  ]"
        echo "}"
    } > "$report_file"

    print_status "$GREEN" "  ✅ Report generated: $report_file"
}

# Function to clean up temporary files
cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] <config_file> [config_file2] ...

Options:
    -h, --help          Show this help message
    -v, --verbose       Verbose output
    -t, --timeout SEC   Set inspector timeout (default: $INSPECTOR_TIMEOUT)
    --report-only       Generate report only (skip validation)

Examples:
    $0 mcp/.mcp.json
    $0 --verbose mcp/.mcp-gemini.json
    $0 -t 60 mcp/.mcp*.json

EOF
}

# Main function
main() {
    local config_files=()
    local report_only=false
    local verbose=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -t|--timeout)
                INSPECTOR_TIMEOUT="$2"
                shift 2
                ;;
            --report-only)
                report_only=true
                shift
                ;;
            -*)
                print_status "$RED" "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                config_files+=("$1")
                shift
                ;;
        esac
    done

    # Check if config files provided
    if [[ ${#config_files[@]} -eq 0 ]]; then
        print_status "$RED" "❌ No configuration files provided"
        usage
        exit 1
    fi

    # Set up signal handling
    trap cleanup EXIT

    # Create temporary directory
    mkdir -p "$TEMP_DIR"

    print_status "$BLUE" "🚀 MCP Inspector Compliance Check (Gterminal)"
    print_status "$BLUE" "$(printf '=%.0s' {1..60})"

    # Check inspector availability
    if ! check_inspector; then
        print_status "$RED" "❌ MCP Inspector not available"
        exit 1
    fi

    local overall_success=0

    # Skip validation if report-only mode
    if [[ "$report_only" != true ]]; then
        # Validate each configuration file
        for config_file in "${config_files[@]}"; do
            if validate_config "$config_file"; then
                print_status "$GREEN" "✅ $config_file validation passed"
            else
                print_status "$RED" "❌ $config_file validation failed"
                overall_success=1
            fi

            echo  # Spacing between configs
        done
    fi

    # Generate report
    generate_report "${config_files[@]}"

    # Print summary
    print_status "$BLUE" "$(printf '=%.0s' {1..60})"

    if [[ $overall_success -eq 0 ]]; then
        print_status "$GREEN" "✅ All MCP Inspector compliance checks passed"
        print_status "$GREEN" "✅ Pre-commit validation successful"
    else
        print_status "$RED" "❌ Some MCP Inspector compliance checks failed"
        print_status "$RED" "❌ Pre-commit validation failed"
    fi

    exit $overall_success
}

# Run main function with all arguments
main "$@"
