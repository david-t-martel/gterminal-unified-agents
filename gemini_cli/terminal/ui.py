"""Rich terminal UI for Gemini CLI."""

import asyncio
import logging
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import PathCompleter, WordCompleter, merge_completers
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from ..core.react_engine import SimpleReactEngine

logger = logging.getLogger(__name__)


class GeminiTerminal:
    """Rich terminal interface for Gemini CLI."""

    def __init__(self) -> None:
        """Initialize the terminal UI."""
        self.console = Console()
        self.engine = SimpleReactEngine()

        # Initialize prompt session
        self.session = PromptSession(
            history=FileHistory(".gemini_cli_history"),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self._create_completer(),
            style=self._create_style(),
        )

        self._running = False

    def _create_completer(self):
        """Create command completer with file paths."""
        commands = [
            # Analysis commands
            "analyze",
            "review",
            "explain",
            "summarize",
            # File operations
            "read",
            "write",
            "edit",
            "create",
            "delete",
            # Search operations
            "search",
            "find",
            "grep",
            # Code operations
            "generate",
            "refactor",
            "fix",
            "optimize",
            # System commands
            "help",
            "status",
            "tools",
            "clear",
            "exit",
            "quit",
            # Workspace operations
            "workspace",
            "project",
            "structure",
        ]

        return merge_completers(
            [WordCompleter(commands, ignore_case=True), PathCompleter()]
        )

    def _create_style(self) -> Style:
        """Create terminal styling."""
        return Style.from_dict(
            {
                "completion-menu.completion": "bg:#008888 #ffffff",
                "completion-menu.completion.current": "bg:#00aaaa #000000",
                "prompt": "bold cyan",
                "success": "green",
                "error": "red",
                "warning": "yellow",
                "info": "blue",
            }
        )

    async def run(self) -> None:
        """Run the interactive terminal session."""
        self._running = True

        # Show welcome message
        self._show_welcome()

        try:
            while self._running:
                try:
                    # Get user input
                    prompt_text = self._get_prompt_text()
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, self.session.prompt, prompt_text
                    )

                    if not user_input.strip():
                        continue

                    # Handle built-in commands
                    if await self._handle_builtin_command(user_input.strip()):
                        continue

                    # Process with ReAct engine
                    await self._process_request(user_input.strip())

                except KeyboardInterrupt:
                    self.console.print(
                        "\\n[yellow]Use 'exit' or 'quit' to leave[/yellow]"
                    )
                    continue
                except EOFError:
                    break

        finally:
            self._running = False
            self.console.print("\\n[cyan]Goodbye! 👋[/cyan]")

    def _show_welcome(self) -> None:
        """Show welcome message and help."""
        welcome_text = """
# Welcome to Gemini CLI! 🚀

A standalone command-line interface for Google Gemini with rich terminal UI.

## Quick Commands:
- `analyze <file/directory>` - Analyze code or workspace
- `help` - Show detailed help
- `tools` - List available tools
- `clear` - Clear screen
- `exit` or `quit` - Exit the CLI

## Examples:
- `analyze ./src` - Analyze source directory
- `read myfile.py` - Read and analyze a Python file
- `workspace .` - Analyze current workspace structure

Type your command or question below:
        """

        panel = Panel(
            Markdown(welcome_text),
            title="[bold cyan]Gemini CLI v1.0.0[/bold cyan]",
            border_style="cyan",
        )

        self.console.print(panel)
        self.console.print()

    def _get_prompt_text(self) -> str:
        """Get the prompt text with current directory."""
        cwd = Path.cwd().name
        return f"[bold cyan]gemini-cli[/bold cyan] [dim]{cwd}[/dim] > "

    async def _handle_builtin_command(self, command: str) -> bool:
        """Handle built-in commands.

        Args:
            command: The user command

        Returns:
            True if command was handled, False otherwise
        """
        parts = command.lower().split()
        if not parts:
            return False

        cmd = parts[0]

        if cmd in ["exit", "quit"]:
            self._running = False
            return True

        elif cmd == "clear":
            self.console.clear()
            self._show_welcome()
            return True

        elif cmd == "help":
            self._show_help()
            return True

        elif cmd == "status":
            self._show_status()
            return True

        elif cmd == "tools":
            self._show_tools()
            return True

        return False

    def _show_help(self) -> None:
        """Show detailed help information."""
        help_text = """
# Gemini CLI Help

## Built-in Commands:
- `help` - Show this help message
- `status` - Show system status
- `tools` - List available tools
- `clear` - Clear the screen
- `exit`, `quit` - Exit the CLI

## Analysis Commands:
- `analyze <path>` - Analyze file or directory
- `workspace <path>` - Analyze workspace structure
- `review <file>` - Review code quality
- `explain <file>` - Explain code functionality

## File Operations:
- `read <file>` - Read file contents
- `search <pattern> in <path>` - Search for text in files
- `find <pattern> in <path>` - Find files by name

## General Usage:
You can ask questions in natural language, for example:
- "What's the structure of this project?"
- "Analyze the Python files in ./src"
- "Find all TODO comments in the codebase"
- "Explain what this function does"

The CLI uses a ReAct (Reason-Act-Observe) engine to break down
complex requests into actions using available tools.
        """

        panel = Panel(
            Markdown(help_text),
            title="[bold cyan]Help[/bold cyan]",
            border_style="cyan",
        )

        self.console.print(panel)

    def _show_status(self) -> None:
        """Show system status."""
        from ..core.auth import GeminiAuth

        # Check authentication
        auth_status = "❌ Not configured"
        try:
            if Path(GeminiAuth.BUSINESS_ACCOUNT_PATH).exists():
                auth_status = "✅ Business account configured"
        except Exception:
            pass

        # Check tools
        tools = self.engine.tool_registry.list_tools()
        tool_count = len(tools)

        status_text = f"""
# System Status

**Authentication:** {auth_status}
**Available Tools:** {tool_count}
**Model:** {self.engine.client.model_name}
**Session:** Active

**Tool List:**
{chr(10).join(f'- {name}: {desc}' for name, desc in tools.items())}
        """

        panel = Panel(
            Markdown(status_text),
            title="[bold cyan]Status[/bold cyan]",
            border_style="cyan",
        )

        self.console.print(panel)

    def _show_tools(self) -> None:
        """Show available tools."""
        tools = self.engine.tool_registry.list_tools()

        if not tools:
            self.console.print("[yellow]No tools available[/yellow]")
            return

        tools_text = "# Available Tools\\n\\n"
        for name, description in tools.items():
            tools_text += f"**{name}:** {description}\\n"

        panel = Panel(
            Markdown(tools_text),
            title="[bold cyan]Tools[/bold cyan]",
            border_style="cyan",
        )

        self.console.print(panel)

    async def _process_request(self, request: str) -> None:
        """Process user request with ReAct engine.

        Args:
            request: The user request
        """
        # Show processing indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Processing request...", total=None)

            try:
                # Process with ReAct engine
                response = await self.engine.process(request)

                # Show response
                self._show_response(response)

                # Show execution summary if verbose
                summary = self.engine.get_execution_summary()
                if summary["total_steps"] > 1:
                    self._show_execution_summary(summary)

            except Exception as e:
                logger.error(f"Request processing failed: {e}")
                self.console.print(f"[red]Error: {e}[/red]")

    def _show_response(self, response: str) -> None:
        """Show the response from the engine.

        Args:
            response: The response text
        """
        # Try to detect if response contains code
        if any(
            marker in response
            for marker in ["```", "def ", "class ", "function ", "import "]
        ):
            # Show as syntax-highlighted panel
            try:
                panel = Panel(
                    Syntax(response, "markdown", theme="monokai"),
                    title="[bold green]Response[/bold green]",
                    border_style="green",
                )
            except Exception:
                # Fallback to markdown
                panel = Panel(
                    Markdown(response),
                    title="[bold green]Response[/bold green]",
                    border_style="green",
                )
        else:
            # Show as markdown
            panel = Panel(
                Markdown(response),
                title="[bold green]Response[/bold green]",
                border_style="green",
            )

        self.console.print(panel)

    def _show_execution_summary(self, summary: dict[str, Any]) -> None:
        """Show execution summary.

        Args:
            summary: Execution summary from ReAct engine
        """
        steps_text = "\\n".join(
            [
                f"**{i+1}.** {step['type'].title()}: {step['description']} "
                f"({'✅' if step['success'] else '❌'})"
                for i, step in enumerate(summary["steps"])
            ]
        )

        summary_panel = Panel(
            Markdown(f"# Execution Summary\\n\\n{steps_text}"),
            title="[dim]Execution Details[/dim]",
            border_style="dim",
        )

        self.console.print(summary_panel)
