#!/usr/bin/env python3
"""Main entry point for GTerminal package."""

import sys
from typing import NoReturn

try:
    from gterminal.gemini_cli.main import main as cli_main
    from gterminal.terminal.main import main as terminal_main
except ImportError:
    # Fallback imports if package structure is not properly installed
    try:
        from gemini_cli.main import main as cli_main
        from terminal.main import main as terminal_main
    except ImportError:
        print(
            "Error: Could not import required modules. Please ensure gterminal is properly installed.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> NoReturn:
    """Main entry point with command routing."""

    # Check if we should route to a specific subcommand
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "terminal":
            # Remove 'terminal' from args and run terminal interface
            sys.argv = [sys.argv[0], *sys.argv[2:]]
            terminal_main()
        elif command == "cli":
            # Remove 'cli' from args and run CLI interface
            sys.argv = [sys.argv[0], *sys.argv[2:]]
            sys.exit(cli_main())
        elif command in ["-h", "--help"]:
            print_help()
            sys.exit(0)
        else:
            # Default to CLI for backward compatibility
            sys.exit(cli_main())
    else:
        # No arguments, show help
        print_help()
        sys.exit(0)


def print_help() -> None:
    """Print main help message."""
    help_text = """
GTerminal - AI-Powered Development Environment

Usage:
    gterminal [COMMAND] [OPTIONS]

Commands:
    terminal    Start the enhanced terminal interface with file editing
    cli         Start the Gemini CLI tool (default)
    -h, --help  Show this help message

Examples:
    gterminal terminal          # Start enhanced terminal UI
    gterminal cli analyze "..."  # Run Gemini CLI analysis
    gterminal --help            # Show this help

For command-specific help:
    gterminal terminal --help
    gterminal cli --help
"""
    print(help_text)


if __name__ == "__main__":
    main()
