#!/bin/bash

# Comprehensive consolidation validation script
# This script validates the successful consolidation from gapp to gterminal

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔍 GTERMINAL CONSOLIDATION VALIDATION"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "Timestamp: $(date)"
echo ""

# Check if required tools are available
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ Error: $1 is not installed or not in PATH"
        exit 1
    fi
}

echo "📋 Checking prerequisites..."
check_tool python3
check_tool pytest
echo "✅ Prerequisites check passed"
echo ""

# Install test dependencies if needed
echo "📦 Installing test dependencies..."
if command -v uv &> /dev/null; then
    uv pip install -e ".[test]" --quiet
else
    python3 -m pip install -e ".[test]" --quiet
fi
echo "✅ Dependencies installed"
echo ""

# Run the comprehensive test runner
echo "🧪 Running comprehensive consolidation tests..."
python3 tests/test_runner.py

# Additional quick checks
echo ""
echo "🔬 Running additional validation checks..."

# Check for remaining gapp references
echo "   Checking for legacy gapp references..."
if grep -r "from gapp\." gterminal/ --include="*.py" 2>/dev/null || \
   grep -r "import gapp" gterminal/ --include="*.py" 2>/dev/null; then
    echo "   ⚠️  Found legacy gapp references (see above)"
else
    echo "   ✅ No legacy gapp references found"
fi

# Check for remaining app references (but not gapp)
echo "   Checking for legacy app references..."
if grep -r "from app\." gterminal/ --include="*.py" 2>/dev/null | grep -v gapp; then
    echo "   ⚠️  Found legacy app references (see above)"
else
    echo "   ✅ No legacy app references found"
fi

# Check structure
echo "   Checking gterminal structure..."
required_dirs=(
    "gterminal/agents"
    "gterminal/auth" 
    "gterminal/cache"
    "gterminal/core"
    "gterminal/terminal"
    "gterminal/gemini_cli"
    "gterminal/utils"
    "gterminal/mcp_servers"
)

missing_dirs=()
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        missing_dirs+=("$dir")
    fi
done

if [ ${#missing_dirs[@]} -eq 0 ]; then
    echo "   ✅ All required directories present"
else
    echo "   ⚠️  Missing directories: ${missing_dirs[*]}"
fi

# Check for nested gterminal structure
if [ -d "gterminal/gterminal" ]; then
    echo "   ❌ Found nested gterminal/gterminal structure"
    exit 1
else
    echo "   ✅ No nested gterminal structure found"
fi

# Try basic imports
echo "   Testing basic imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    import gterminal
    import gterminal.agents
    import gterminal.core.agents
    import gterminal.terminal
    print('   ✅ Basic imports successful')
except ImportError as e:
    print(f'   ⚠️  Import issue: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 CONSOLIDATION VALIDATION COMPLETE"
echo "===================================="

# Check if test report exists
if [ -f "consolidation_test_report.md" ]; then
    echo "📊 Detailed test report available: consolidation_test_report.md"
    
    # Show summary from report
    if grep -q "SUCCESS: All tests passed" consolidation_test_report.md; then
        echo "✅ Overall result: CONSOLIDATION SUCCESSFUL"
        exit 0
    elif grep -q "Some tests failed" consolidation_test_report.md; then
        echo "❌ Overall result: CONSOLIDATION HAS ISSUES"
        echo "📋 Please review the detailed report for specific issues to address."
        exit 1
    else
        echo "❓ Overall result: UNKNOWN - please review the detailed report"
        exit 1
    fi
else
    echo "⚠️  No detailed test report generated"
    echo "❓ Consolidation status unclear - manual verification recommended"
    exit 1
fi