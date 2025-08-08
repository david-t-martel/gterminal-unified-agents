#!/usr/bin/env python3
"""Use Gemini documentation agent to create comprehensive docs for auto-claude.

#TODO: Add debugging hints for Rust and PyO3 frameworks
#TODO: Include maturin build troubleshooting
#TODO: Document UV package manager integration
#FIXME: Update with PyO3 compilation error solutions
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, Path(os.path.abspath(__file__).parent))

from pathlib import Path

from .documentation_agent import generate_documentation
from .documentation_agent import generate_readme


async def document_auto_claude() -> None:
    """Generate comprehensive documentation for auto-claude."""
    target_project = "/home/david/projects/auto-claude"

    # Document key Python files
    python_files = [
        "auto-claude",
        "install-auto-claude.sh",
        "scripts/migrate-auto-claude.py",
    ]

    for file in python_files:
        file_path = Path(target_project) / file
        if Path(file_path).exists():
            await generate_documentation(
                file_path=file_path,
                output_dir=Path(target_project) / "docs/api",
                doc_style="comprehensive",
            )

    # Generate comprehensive README
    await generate_readme(project_dir=target_project, include_examples=True, include_api_docs=True)

    # Create migration guide
    migration_guide = """# Auto-Claude Migration Guide

## Overview

This guide documents the migration of auto-claude enhancements from the my-fullstack-agent project.

## What Was Migrated

### Configuration Files
- `.auto-claude.yaml` - Enhanced configuration with auto-fix settings
- `.auto-claude/config.json` - Core configuration
- `.auto-claude/context.json` - Project context
- `.gitignore.auto-claude` - Git ignore patterns

### Scripts and Tools
- `auto-claude` - Main executable with intelligent merging
- `install-auto-claude.sh` - Installation script
- `scripts/migrate-auto-claude.py` - Migration orchestration

## New Features

### 1. Enhanced Auto-Fix Capabilities
- Type hint fixing
- Docstring generation
- Import sorting
- Best practices application

### 2. Integration Features
- Test generation after fixes
- Documentation updates
- Pre-commit hook integration

### 3. Commit Automation
- Auto-commit with Claude co-authorship
- Configurable commit messages
- Auto-staging support

## Configuration

The `.auto-claude.yaml` file provides rich configuration:

```yaml
auto_add_new_files: true
track_patterns:
  - "tests/test_*.py"
  - "docs/**/*.md"
auto_fix:
  fix_type_hints: true
  add_docstrings: true
```

## Usage

### Basic Commands
```bash
# Initialize auto-claude in a project
./auto-claude init

# Fix issues in current directory
./auto-claude fix

# Fix specific files
./auto-claude fix src/main.py src/utils.py

# Analyze without fixing
./auto-claude analyze

# View history
./auto-claude history
```

### Advanced Usage
```bash
# Configure settings
./auto-claude config set model sonnet
./auto-claude config set auto_commit true

# Install git hooks
./auto-claude install-hooks

# Rollback changes
./auto-claude rollback
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - Ensure auto-claude is executable: `chmod +x auto-claude`

2. **Import Errors**
   - Install dependencies: `./install-auto-claude.sh`

3. **Configuration Issues**
   - Reset config: `rm -rf .auto-claude && ./auto-claude init`

## Contributing

See CONTRIBUTING.md for guidelines on contributing to auto-claude.
"""

    # Save migration guide
    guide_path = Path(target_project) / "docs" / "MIGRATION_GUIDE.md"
    guide_path.parent.mkdir(exist_ok=True)
    guide_path.write_text(migration_guide)


if __name__ == "__main__":
    asyncio.run(document_auto_claude())
