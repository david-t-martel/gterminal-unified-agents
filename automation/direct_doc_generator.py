#!/usr/bin/env python3
"""Direct documentation generation using Gemini model."""

from pathlib import Path

from gemini_config import get_model_for_task


def generate_auto_claude_docs() -> None:
    """Generate documentation for auto-claude using Gemini."""
    model = get_model_for_task("documentation")
    target_project = "/home/david/projects/auto-claude"

    # Read the main auto-claude file
    auto_claude_path = Path(target_project) / "auto-claude"
    if auto_claude_path.exists():
        content = auto_claude_path.read_text()

        prompt = f"""
        Generate comprehensive documentation for this auto-claude tool.

        File content:
        ```python
        {content[:2000]}  # First 2000 chars
        ```

        Please create:
        1. A detailed feature list
        2. Usage examples for each command
        3. Configuration options
        4. Best practices
        5. Troubleshooting guide

        Format as a complete README.md file.
        """

        response = model.generate_content(prompt)

        # Save the documentation
        readme_path = Path(target_project) / "README_ENHANCED.md"
        readme_path.write_text(response.text)

    # Document the configuration
    config_path = Path(target_project) / ".auto-claude.yaml"
    if config_path.exists():
        config_content = config_path.read_text()

        config_prompt = f"""
        Document this auto-claude configuration file in detail.

        Configuration:
        ```yaml
        {config_content}
        ```

        Create a configuration guide that explains:
        1. Each configuration option
        2. How to customize settings
        3. Integration options
        4. Examples of different configurations

        Format as a CONFIG_GUIDE.md file.
        """

        config_response = model.generate_content(config_prompt)

        config_guide_path = Path(target_project) / "docs" / "CONFIG_GUIDE.md"
        config_guide_path.parent.mkdir(exist_ok=True)
        config_guide_path.write_text(config_response.text)

    # Create migration summary
    migration_prompt = """
    Create a summary of the auto-claude migration that was performed.

    The Gemini agents migrated auto-claude enhancements from my-fullstack-agent to the main auto-claude project.

    Key items migrated:
    - Enhanced configuration files (.auto-claude.yaml, config.json)
    - Installation scripts
    - Git ignore patterns
    - Auto-fix capabilities
    - Integration features

    Create a MIGRATION_SUMMARY.md that documents:
    1. What was migrated
    2. New features added
    3. How to use the new features
    4. Benefits of the migration
    """

    migration_response = model.generate_content(migration_prompt)

    migration_path = Path(target_project) / "MIGRATION_SUMMARY.md"
    migration_path.write_text(migration_response.text)


if __name__ == "__main__":
    generate_auto_claude_docs()
