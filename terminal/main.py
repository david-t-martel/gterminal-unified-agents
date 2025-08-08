# FIXME: Unused import 'Any' - remove if not needed
#!/usr/bin/env python3
"""Enhanced Terminal UI with Full File Editing Capabilities."""

import asyncio
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any

import aiohttp
import click
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Initialize console
console = Console()

# Configuration
# FIXME: Hardcoded URL - move to configuration
GEMINI_SERVER_URL = os.getenv("GEMINI_SERVER_URL", "http://127.0.0.1:8100")


class EnhancedTerminalUI:
    """Enhanced terminal with full file editing and code manipulation capabilities."""

    def __init__(self) -> None:
        self.console = console
        self.server_available = False
        self.server_session_id = None
        self.http_session = None

        # Tool registry for tracking available tools
        self.available_tools = {}

        # Command history with file path completion
        self.session = PromptSession(
            history=FileHistory(".agent_terminal_history"),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self._create_completer(),
        )

        # Rich styling
        self.style = Style.from_dict(
            {
                "completion-menu.completion": "bg:#008888 #ffffff",
                "completion-menu.completion.current": "bg:#00aaaa #000000",
                "prompt": "bold cyan",
                "success": "green",
                "error": "red",
                "warning": "yellow",
                "info": "blue",
            },
        )

    def _create_completer(self) -> None:
        """Create advanced command completer with file paths."""
        commands = [
            # File operations
            "read",
            "write",
            "edit",
            "create",
            "delete",
            "copy",
            "move",
            # Code operations
            "analyze",
            "review",
            "generate",
            "refactor",
            "fix",
            "optimize",
            # Documentation
            "document",
            "explain",
            "summarize",
            # System
            "help",
            "status",
            "tools",
            "session",
            "clear",
            "exit",
            # Advanced
            "search",
            "replace",
            "diff",
            "commit",
        ]

        # Combine word completer with path completer
        from prompt_toolkit.completion import merge_completers

        return merge_completers([WordCompleter(commands, ignore_case=True), PathCompleter()])

    async def _initialize(self) -> None:
        """Initialize server connection and tools."""
        self.http_session = aiohttp.ClientSession()
        await self._check_server()
        await self._load_tools()

    async def _check_server(self) -> bool:
        """Check Gemini server availability."""
        try:
            async with self.http_session.get(f"{GEMINI_SERVER_URL}/status") as resp:
                if resp.status == 200:
                    await resp.json()
                    self.server_available = True
                    self.console.print("[green]✓ Connected to Gemini Server[/green]")

                    # Create session
                    self.server_session_id = f"terminal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    return True
        except Exception as e:
            self.console.print(f"[yellow]⚠ Gemini Server not available: {e}[/yellow]")
            self.console.print("[info]Running in local mode with limited capabilities[/info]")
        return False

    async def _load_tools(self) -> None:
        """Load available tools from server."""
        if not self.server_available:
            # Define local tools
            self.available_tools = {
                "read_file": "Read file contents",
                "write_file": "Write content to file",
                "list_directory": "List directory contents",
                "search_files": "Search for files by pattern",
                "execute_command": "Execute shell commands",
            }
            return

        try:
            async with self.http_session.get(f"{GEMINI_SERVER_URL}/tools") as resp:
                if resp.status == 200:
                    tools = await resp.json()
                    self.available_tools = {t["name"]: t["description"] for t in tools}
                    self.console.print(f"[green]✓ Loaded {len(self.available_tools)} tools[/green]")
        except Exception as e:
            self.console.print(f"[warning]Could not load tools: {e}[/warning]")

    def show_help(self) -> None:
        """Show comprehensive help."""
        help_text = """
# My Fullstack Agent Terminal

## File Operations
- **read <file>** - Read and display file contents
- **write <file> <content>** - Write content to file
- **edit <file>** - Interactive file editing
- **create <file>** - Create new file
- **delete <file>** - Delete file (with confirmation)
- **copy <src> <dst>** - Copy file
- **move <src> <dst>** - Move/rename file

## Code Operations
- **analyze <file/dir>** - Analyze code structure and quality
- **review <file>** - Comprehensive code review
- **generate <spec>** - Generate code from specification
- **refactor <file>** - Suggest refactoring improvements
- **fix <file>** - Auto-fix common issues
- **optimize <file>** - Performance optimization suggestions

## Documentation
- **document <file/dir>** - Generate documentation
- **explain <file>** - Explain code functionality
- **summarize <dir>** - Project summary

## Search & Replace
- **search <pattern> [path]** - Search for pattern in files
- **replace <pattern> <replacement> [path]** - Replace text in files

## System Commands
- **status** - Show system and server status
- **tools** - List available tools
- **session** - Show session information
- **clear** - Clear screen
- **exit** - Exit terminal

## Advanced Usage
- **<any natural language>** - Direct AI task execution
- Use Tab for file path completion
- Use ↑/↓ for command history
"""
        self.console.print(Markdown(help_text))

    # FIXME: Function 'read_file' missing return type annotation
    async def read_file(self, file_path: str) -> None:
        """Read and display file contents with syntax highlighting."""
        path = Path(file_path).resolve()

        if not path.exists():
            self.console.print(f"[error]File not found: {path}[/error]")
            return

        if not path.is_file():
            self.console.print(f"[error]Not a file: {path}[/error]")
            return

        try:
            content = path.read_text()

            # Determine language for syntax highlighting
            suffix = path.suffix.lower()
            language_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".rs": "rust",
                ".go": "go",
                ".java": "java",
                ".cpp": "cpp",
                ".c": "c",
                ".html": "html",
                ".css": "css",
                ".json": "json",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".md": "markdown",
                ".sh": "bash",
            }
            language = language_map.get(suffix, "text")

            # Display with syntax highlighting
            self.console.print(f"\n[bold]{path}[/bold]")
            self.console.print("─" * 80)

            if language == "markdown":
                self.console.print(Markdown(content))
            else:
                syntax = Syntax(content, language, theme="monokai", line_numbers=True)
                self.console.print(syntax)

        except Exception as e:
            self.console.print(f"[error]Failed to read file: {e}[/error]")

    # FIXME: Function 'write_file' missing return type annotation
    async def write_file(self, file_path: str, content: str | None = None) -> None:
        """Write content to file."""
        path = Path(file_path).resolve()

        if content is None:
            # Interactive mode
            self.console.print("[info]Enter content (Ctrl+D when done):[/info]")
            lines: list[Any] = []
            try:
                while True:
                    line = await asyncio.get_event_loop().run_in_executor(None, input)
                    lines.append(line)
            except EOFError:
                content = "\n".join(lines)

        # Confirm overwrite if exists
        if path.exists():
            confirm = await self._confirm(f"Overwrite existing file {path}?")
            if not confirm:
                self.console.print("[warning]Write cancelled[/warning]")
                return

        try:
            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            path.write_text(content)
            self.console.print(f"[success]✓ Written {len(content)} bytes to {path}[/success]")

        except Exception as e:
            self.console.print(f"[error]Failed to write file: {e}[/error]")

    # FIXME: Async function 'edit_file' missing error handling - add try/except block
    async def edit_file(self, file_path: str) -> None:
        """Interactive file editing with AI assistance."""
        path = Path(file_path).resolve()

        if not path.exists():
            self.console.print(f"[warning]File does not exist. Creating new file: {path}[/warning]")
            content = ""
        else:
            content = path.read_text()

        self.console.print(f"\n[bold]Editing: {path}[/bold]")
        self.console.print("Commands: save, cancel, help, ai <instruction>")
        self.console.print("─" * 80)

        # Show current content
        if content:
            syntax = Syntax(content, self._get_language(path), line_numbers=True)
            self.console.print(syntax)

        while True:
            cmd = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.prompt("edit> ", style=self.style),
            )

            if cmd == "save":
                path.write_text(content)
                self.console.print(f"[success]✓ Saved {path}[/success]")
                break
            if cmd == "cancel":
                self.console.print("[warning]Edit cancelled[/warning]")
                break
            if cmd == "help":
                self.console.print(
                    """
Edit Commands:
- save - Save changes and exit
- cancel - Discard changes and exit
- ai <instruction> - Use AI to modify code
- show - Display current content
- clear - Clear content
""",
                )
            elif cmd.startswith("ai "):
                instruction = cmd[3:]
                new_content = await self._ai_edit(content, instruction, path)
                if new_content:
                    content = new_content
                    syntax = Syntax(content, self._get_language(path), line_numbers=True)
                    self.console.print("\n[bold]Updated content:[/bold]")
                    self.console.print(syntax)
            elif cmd == "show":
                syntax = Syntax(content, self._get_language(path), line_numbers=True)
                self.console.print(syntax)
            elif cmd == "clear":
                content = ""
                self.console.print("[info]Content cleared[/info]")
            else:
                # Treat as content addition
                content += "\n" + cmd

    async def _ai_edit(self, content: str, instruction: str, path: Path) -> str | None:
        """Use AI to edit code based on instruction."""
        if not self.server_available:
            self.console.print("[error]AI editing requires Gemini server connection[/error]")
            return None

        prompt = f"""
Edit the following code according to this instruction: {instruction}

File: {path}
Current content:
```
{content}
```

Return ONLY the modified code without any explanation.
"""

        try:
            async with self.http_session.post(
                f"{GEMINI_SERVER_URL}/task",
                json={
                    "instruction": prompt,
                    "type": "code_generation",
                    "session_id": self.server_session_id,
                },
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("result", "")
                error = await resp.text()
                self.console.print(f"[error]AI edit failed: {error}[/error]")
        except Exception as e:
            self.console.print(f"[error]AI edit error: {e}[/error]")

        return None

    def _get_language(self, path: Path) -> str:
        """Get language from file extension."""
        suffix = path.suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".html": "html",
            ".css": "css",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
            ".sh": "bash",
        }
        return language_map.get(suffix, "text")

    async def _confirm(self, message: str) -> bool:
        """Get user confirmation."""
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.session.prompt(f"{message} (y/N): ", style=self.style),
        )
        return response.lower() in ["y", "yes"]

    # FIXME: Function 'execute_task' missing return type annotation
    async def execute_task(self, instruction: str, task_type: str = "general") -> None:
        """Execute task via Gemini server."""
        if not self.server_available:
            self.console.print("[error]This command requires Gemini server connection[/error]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task_id = progress.add_task("Processing...", total=None)

            try:
                async with self.http_session.post(
                    f"{GEMINI_SERVER_URL}/task",
                    json={
                        "instruction": instruction,
                        "type": task_type,
                        "session_id": self.server_session_id,
                        "streaming": False,
                    },
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        progress.update(task_id, completed=True)

                        # Display result
                        self.console.print(
                            Panel(
                                result.get("result", "No result"),
                                title=f"[bold green]{task_type.title()} Result[/bold green]",
                                border_style="green",
                            ),
                        )

                        if "total_time" in result:
                            self.console.print(f"[dim]Time: {result['total_time']:.2f}s[/dim]")
                    else:
                        error = await resp.text()
                        progress.update(task_id, completed=True)
                        self.console.print(f"[error]Task failed: {error}[/error]")

            except Exception as e:
                progress.update(task_id, completed=True)
                self.console.print(f"[error]Error: {e}[/error]")

    # FIXME: Async function 'process_command' missing error handling - add try/except block
    async def process_command(self, command: str) -> bool:
        """Process user command."""
        parts = command.strip().split(maxsplit=2)
        if not parts:
            return True

        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        # System commands
        if cmd in ["exit", "quit"]:
            return False
        if cmd == "clear":
            self.console.clear()
        elif cmd == "help":
            self.show_help()
        elif cmd == "status":
            await self.show_status()
        elif cmd == "tools":
            self.show_tools()

        # File operations
        elif cmd == "read" and args:
            await self.read_file(args[0])
        elif cmd == "write" and args:
            content = args[1] if len(args) > 1 else None
            await self.write_file(args[0], content)
        elif cmd == "edit" and args:
            await self.edit_file(args[0])
        elif cmd == "create" and args:
            await self.write_file(args[0], "")

        # Code operations
        elif cmd in ["analyze", "review"] and args:
            instruction = f"{cmd.title()} the code at {args[0]}"
            await self.execute_task(instruction, "code_analysis")
        elif cmd == "generate" and args:
            instruction = " ".join(args)
            await self.execute_task(f"Generate code: {instruction}", "code_generation")
        elif cmd == "document" and args:
            instruction = f"Generate comprehensive documentation for {args[0]}"
            await self.execute_task(instruction, "documentation")

        # Search operations
        elif cmd == "search" and args:
            pattern = args[0]
            path = args[1] if len(args) > 1 else "."
            await self.search_files(pattern, path)

        # Direct AI execution
        else:
            # Treat as natural language task
            await self.execute_task(command, "general")

        return True

    # FIXME: Async function 'search_files' missing error handling - add try/except block
    async def search_files(self, pattern: str, path: str = ".") -> None:
        """Search for pattern in files."""
        self.console.print(f"[info]Searching for '{pattern}' in {path}...[/info]")

        instruction = (
            f"Search for '{pattern}' in all files under {path} and show matches with context"
        )
        await self.execute_task(instruction, "workspace_analysis")

    # FIXME: Async function 'show_status' missing error handling - add try/except block
    async def show_status(self) -> None:
        """Show comprehensive status."""
        table = Table(title="System Status", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Gemini Server", "Connected" if self.server_available else "Not Connected")
        if self.server_available:
            table.add_row("Session ID", self.server_session_id or "None")
        table.add_row("Available Tools", str(len(self.available_tools)))
        table.add_row("Working Directory", os.getcwd())
        table.add_row("Python Version", sys.version.split()[0])

        self.console.print(table)

    def show_tools(self) -> None:
        """Show available tools."""
        if not self.available_tools:
            self.console.print("[warning]No tools available[/warning]")
            return

        table = Table(title="Available Tools")
        table.add_column("Tool", style="cyan")
        table.add_column("Description", style="white")

        for name, desc in sorted(self.available_tools.items()):
            table.add_row(name, desc)

        self.console.print(table)

    # FIXME: Function 'run' missing return type annotation
    async def run(self) -> None:
        """Run the terminal UI."""
        # Initialize connections
        await self._initialize()

        # Welcome banner
        self.console.print(
            Panel(
                Text("My Fullstack Agent Terminal", style="bold cyan", justify="center"),
                subtitle="Enhanced File Editing & Code Manipulation",
                border_style="blue",
            ),
        )

        self.console.print("\nType 'help' for commands or use natural language")
        if self.server_available:
            self.console.print("[success]AI-powered editing and code generation enabled[/success]")
        self.console.print()

        # Main loop
        while True:
            try:
                # Get input
                command = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.prompt("▶ ", style=self.style),
                )

                # Process command
                if not await self.process_command(command):
                    break

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[error]Error: {e}[/error]")

        # Cleanup
        if self.http_session:
            await self.http_session.close()

        self.console.print("\n[yellow]Thank you for using My Fullstack Agent![/yellow]")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--server-url",
    "-s",
    default=None,
    help="Gemini server URL (default: http://127.0.0.1:8100)",
)
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode with verbose logging")
@click.option("--no-server", "-n", is_flag=True, help="Run in local mode without Gemini server")
@click.option("--session-id", default=None, help="Resume a specific session ID")
@click.option("--config", "-c", type=click.Path(exists=True), help="Load configuration from file")
@click.version_option(version="0.4.0", prog_name="My Fullstack Agent Terminal")
def main(server_url: str, debug: bool, no_server: bool, session_id: str, config: str) -> None:
    r"""My Fullstack Agent Terminal - Enhanced UI with AI-Powered File Editing.

    \b
    This terminal provides a powerful interface for:
    • Reading and writing files with syntax highlighting
    • Interactive file editing with AI assistance
    • Code generation, analysis, and refactoring
    • Natural language task execution
    • Full integration with Gemini AI for intelligent operations

    \b
    QUICK START:
    1. Start with Gemini server: make terminal-server
    2. Start in local mode: make terminal-enhanced --no-server
    3. Get help in terminal: type 'help'

    \b
    EXAMPLES:
    # Read a file with syntax highlighting
    $ read app/main.py

    # Edit a file with AI assistance
    $ edit config.yaml
    edit> ai add a database connection section
    edit> save

    # Generate code from specification
    $ generate Create a REST API for user management

    # Search for patterns
    $ search "TODO" src/

    # Natural language commands
    $ Fix all linting errors in the project

    \b
    FILE OPERATIONS:
    • read <file>        - Display file with syntax highlighting
    • write <file>       - Write content to file
    • edit <file>        - Interactive editing with AI
    • create <file>      - Create new file
    • delete <file>      - Delete file (with confirmation)
    • copy <src> <dst>   - Copy file
    • move <src> <dst>   - Move/rename file

    \b
    CODE OPERATIONS:
    • analyze <file>     - Analyze code structure
    • review <file>      - Comprehensive code review
    • generate <spec>    - Generate code from description
    • refactor <file>    - Suggest improvements
    • fix <file>         - Auto-fix common issues
    • optimize <file>    - Performance optimization

    \b
    AI EDITING COMMANDS (in edit mode):
    • ai <instruction>   - Use AI to modify code
    • save              - Save changes and exit
    • cancel            - Discard changes
    • show              - Display current content
    • clear             - Clear all content

    \b
    CONFIGURATION:
    Set GEMINI_SERVER_URL environment variable or use --server-url option.
    Default server URL: http://127.0.0.1:8100

    \b
    For more information, visit: https://github.com/david-t-martel/agents/my-fullstack-agent
    """
    if server_url:
        global GEMINI_SERVER_URL
        GEMINI_SERVER_URL = server_url

    # Set up logging
    import logging

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress noisy logs
    if not debug:
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("aiohttp").setLevel(logging.WARNING)

    # Run terminal
    terminal = EnhancedTerminalUI()

    try:
        asyncio.run(terminal.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        if debug:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
