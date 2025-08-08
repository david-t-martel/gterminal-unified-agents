#!/bin/bash
# Setup script for Enhanced Gemini CLI

set -e

echo "🚀 Setting up Enhanced Gemini CLI..."

# Check if we're in the right directory
if [ ! -f "enhanced_gemini_cli.py" ]; then
    echo "❌ Error: Please run this script from the gterminal directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install/upgrade dependencies
echo "📦 Installing dependencies..."
pip install --upgrade \
    click \
    rich \
    google-auth \
    google-auth-oauthlib \
    google-auth-httplib2 \
    pydantic \
    fastmcp

# Build PyO3 extensions if Rust is available
if command -v cargo &> /dev/null; then
    echo "🦀 Building PyO3 Rust extensions..."
    cd gterminal_rust_extensions
    if [ -f "Cargo.toml" ]; then
        maturin develop --release || echo "⚠️  Rust extensions build failed - will run in pure Python mode"
    fi
    cd ..
else
    echo "ℹ️  Rust not found - will run in pure Python mode"
fi

# Create symlink for easy access
if [ -d "$HOME/.local/bin" ]; then
    ln -sf "$PWD/gemini-enhanced" "$HOME/.local/bin/gemini-enhanced" 2>/dev/null || true
    echo "✓ Created symlink in ~/.local/bin/gemini-enhanced"
fi

# Test the installation
echo "🧪 Testing installation..."
if python3 enhanced_gemini_cli.py --version | grep -q "2.0.0-enhanced"; then
    echo "✅ Enhanced Gemini CLI successfully installed!"
    echo ""
    echo "Usage:"
    echo "  ./gemini-enhanced --help              # Show all commands"
    echo "  ./gemini-enhanced analyze -i          # Interactive mode"
    echo "  ./gemini-enhanced super-analyze .     # Analyze current directory with 1M+ context"
    echo "  ./gemini-enhanced profiles            # Show GCP profiles"
    echo "  ./gemini-enhanced test-integration    # Test all integrations"
    echo ""
    echo "For global access, add to PATH:"
    echo "  export PATH=\"$HOME/.local/bin:\$PATH\""
else
    echo "❌ Installation test failed"
    exit 1
fi