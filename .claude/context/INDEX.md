# GTerminal Context Index

This directory contains comprehensive project context for the GTerminal project, maintained for AI agent coordination and knowledge persistence.

## Context Files

### Primary Context Documents
- **[project_context_20250807.yaml](project_context_20250807.yaml)** - Complete project context in YAML format (human-readable)
- **[project_context_20250807.json](project_context_20250807.json)** - Complete project context in JSON format (machine-readable)

### Reference Documents  
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference guide for common tasks and architecture
- **[VERSION.md](VERSION.md)** - Version history and change tracking

## Usage

### For AI Agents
1. Load the JSON file for structured data access
2. Reference QUICK_REFERENCE.md for command examples
3. Check VERSION.md for latest updates

### For Developers
1. Read YAML file for comprehensive understanding
2. Use QUICK_REFERENCE.md for daily tasks
3. Update VERSION.md when making significant changes

## Update Policy

Context should be updated when:
- Major features are completed
- Architecture changes significantly
- Performance milestones are reached
- New agents join the project
- Technology stack evolves

## Access Patterns

```python
# Load context programmatically
import json
with open('.claude/context/project_context_20250807.json') as f:
    context = json.load(f)

# Access specific information
tech_stack = context['architecture']['technology_stack']
roadmap = context['roadmap']
```

## Maintenance

- Keep contexts timestamped (YYYYMMDD format)
- Maintain both YAML and JSON versions
- Update VERSION.md with each change
- Archive old contexts after major updates