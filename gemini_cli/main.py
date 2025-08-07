#!/usr/bin/env python3
"""Main CLI interface for Gemini CLI tool."""

import asyncio
from pathlib import Path

import click

from .core.client import GeminiClient
from .terminal.ui import GeminiTerminal


@click.group()
@click.version_option(version="1.0.0")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def cli(debug: bool) -> None:
    """Standalone Gemini CLI tool with rich terminal interface."""
    if debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)


@cli.command()
@click.argument("prompt", required=False)
@click.option("--interactive", "-i", is_flag=True, help="Start interactive terminal mode")
def analyze(prompt: str | None, interactive: bool) -> None:
    """Analyze code or start interactive session."""
    if interactive or not prompt:
        # Start interactive terminal
        terminal = GeminiTerminal()
        asyncio.run(terminal.run())
    else:
        # Single prompt execution
        client = GeminiClient()
        response = asyncio.run(client.process(prompt))
        click.echo(response)


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def workspace(path: Path) -> None:
    """Analyze workspace/project structure."""
    from .tools.filesystem import FilesystemTool

    fs_tool = FilesystemTool()
    analysis = asyncio.run(fs_tool.analyze_workspace(path))
    click.echo(analysis)


def main() -> int:
    """Main entry point."""
    try:
        cli()
        return 0
    except KeyboardInterrupt:
        click.echo("Interrupted by user", err=True)
        return 130
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    main()
