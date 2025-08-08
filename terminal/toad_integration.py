#!/usr/bin/env python3
"""Toad-Style Terminal Integration for Enhanced ReAct Engine.

Inspired by Will McGugan's Toad (https://willmcgugan.github.io/announcing-toad/),
this module provides a native terminal interface using Textual that can work
alongside our web-based terminal.

Architecture Comparison:
- Our Web Terminal: Browser-based UI with WebSocket communication
- Toad-Style: Native terminal UI with JSON over stdin/stdout
- This Integration: Native terminal UI with our Enhanced ReAct Engine

Key Benefits:
- No "jank" - smooth terminal updates
- Text selection and interaction
- Rich formatting with Textual
- Local execution with no browser dependency
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any

from rich.panel import Panel
from rich.table import Table
from textual import events
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.containers import Vertical
from textual.widgets import Button
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Log
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane

# Import our enhanced orchestrator
from .enhanced_react_orchestrator import EnhancedReActOrchestrator
from .enhanced_react_orchestrator import EnhancedReActTask
from .enhanced_react_orchestrator import TaskPriority
from .enhanced_react_orchestrator import create_enhanced_orchestrator

logger = logging.getLogger(__name__)


class ReActStatusWidget(Static):
    """Widget showing current ReAct engine status."""

    def __init__(self) -> None:
        super().__init__()
        self.orchestrator: EnhancedReActOrchestrator | None = None
        self.current_task: EnhancedReActTask | None = None
        self.task_progress: dict[str, Any] = {}

    def set_orchestrator(self, orchestrator: EnhancedReActOrchestrator) -> None:
        """Set the orchestrator instance."""
        self.orchestrator = orchestrator

    def update_task_progress(self, progress: dict[str, Any]) -> None:
        """Update task progress display."""
        self.task_progress = progress
        self.refresh()

    def set_current_task(self, task: EnhancedReActTask | None) -> None:
        """Set the current task being processed."""
        self.current_task = task
        self.refresh()

    def render(self) -> Panel:
        """Render the status panel."""
        if not self.orchestrator:
            return Panel("ReAct Engine: Not initialized", style="red")

        # Build status content
        status_lines = []
        status_lines.append("🧠 Enhanced ReAct Engine Status")
        status_lines.append("")

        if self.current_task:
            status_lines.append(f"📋 Current Task: {self.current_task.description[:50]}...")
            status_lines.append(f"   Priority: {self.current_task.priority.value}")
            status_lines.append(f"   Privacy: {self.current_task.privacy_level}")

            if self.task_progress:
                iteration = self.task_progress.get("iteration", 0)
                status = self.task_progress.get("status", "unknown")
                actions = self.task_progress.get("actions_taken", 0)
                status_lines.append(
                    f"   Progress: Iteration {iteration}, Status: {status}, Actions: {actions}"
                )
        else:
            status_lines.append("📋 Current Task: None")

        status_lines.append("")
        status_lines.append("🎯 Ready for commands")

        return Panel("\n".join(status_lines), title="ReAct Status", style="green")


class TaskHistoryWidget(Static):
    """Widget showing task execution history."""

    def __init__(self) -> None:
        super().__init__()
        self.task_history: list[dict[str, Any]] = []

    def add_task_result(self, task: EnhancedReActTask, result: dict[str, Any]) -> None:
        """Add a completed task to history."""
        self.task_history.append(
            {
                "task": task,
                "result": result,
                "timestamp": datetime.now(),
            }
        )

        # Keep only last 10 tasks
        if len(self.task_history) > 10:
            self.task_history = self.task_history[-10:]

        self.refresh()

    def render(self) -> Panel:
        """Render the task history panel."""
        if not self.task_history:
            return Panel("No completed tasks", title="Task History", style="blue")

        # Create table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Time", style="dim", width=8)
        table.add_column("Task", style="cyan", min_width=30)
        table.add_column("Status", style="green", width=8)
        table.add_column("Duration", style="yellow", width=8)

        for entry in self.task_history[-5:]:  # Show last 5
            task = entry["task"]
            result = entry["result"]
            timestamp = entry["timestamp"]

            status_icon = "✅" if result.get("success") else "❌"
            duration = f"{result.get('execution_time', 0):.1f}s"
            task_desc = (
                task.description[:40] + "..." if len(task.description) > 40 else task.description
            )

            table.add_row(timestamp.strftime("%H:%M:%S"), task_desc, status_icon, duration)

        return Panel(table, title="Recent Tasks", style="blue")


class CommandInput(Input):
    """Enhanced command input widget."""

    def __init__(self) -> None:
        super().__init__(placeholder="Enter ReAct command (e.g., 'react Analyze the codebase')")
        self.command_history: list[str] = []
        self.history_index = -1

    def add_to_history(self, command: str) -> None:
        """Add command to history."""
        if command and command != self.command_history[-1:]:
            self.command_history.append(command)
            if len(self.command_history) > 100:
                self.command_history = self.command_history[-100:]
        self.history_index = -1

    def on_key(self, event: events.Key) -> None:
        """Handle key events for history navigation."""
        if event.key == "up":
            if self.command_history and self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                self.value = self.command_history[-(self.history_index + 1)]
                event.prevent_default()
        elif event.key == "down":
            if self.history_index > 0:
                self.history_index -= 1
                self.value = self.command_history[-(self.history_index + 1)]
                event.prevent_default()
            elif self.history_index == 0:
                self.history_index = -1
                self.value = ""
                event.prevent_default()


class ToadStyleReActTerminal(App):
    """Toad-style terminal interface for Enhanced ReAct Engine."""

    CSS_PATH = "toad_terminal.css"
    TITLE = "Enhanced ReAct Terminal (Toad Style)"

    def __init__(self) -> None:
        super().__init__()
        self.orchestrator: EnhancedReActOrchestrator | None = None
        self.current_task: EnhancedReActTask | None = None

        # Widgets
        self.status_widget = ReActStatusWidget()
        self.history_widget = TaskHistoryWidget()
        self.command_input = CommandInput()
        self.output_log = Log(auto_scroll=True)

    async def on_mount(self) -> None:
        """Initialize the application."""
        # Initialize orchestrator
        self.output_log.write("🚀 Initializing Enhanced ReAct Engine...")
        try:
            self.orchestrator = await create_enhanced_orchestrator(
                enable_local_llm=True,
                enable_web_fetch=True,
            )
            self.status_widget.set_orchestrator(self.orchestrator)
            self.output_log.write("✅ ReAct Engine initialized successfully!")
            self.output_log.write("💡 Type commands like: 'react Analyze the codebase'")
            self.output_log.write("")
        except Exception as e:
            self.output_log.write(f"❌ Failed to initialize ReAct Engine: {e}")
            logger.exception("Orchestrator initialization failed")

    def compose(self) -> ComposeResult:
        """Create the terminal layout."""
        with TabbedContent(initial="main"):
            with TabPane("Main", id="main"):
                yield Header()

                with Horizontal():
                    # Left panel - Status and history
                    with Vertical(classes="left-panel"):
                        yield self.status_widget
                        yield self.history_widget

                    # Right panel - Output and input
                    with Vertical(classes="right-panel"):
                        yield self.output_log
                        with Horizontal(classes="input-panel"):
                            yield self.command_input
                            yield Button("Execute", id="execute", variant="primary")
                            yield Button("Clear", id="clear", variant="default")

            with TabPane("Metrics", id="metrics"):
                yield Static("📊 System Metrics", id="metrics-display")

            with TabPane("Sessions", id="sessions"):
                yield Static("🖥️ Active Sessions", id="sessions-display")

        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "execute":
            await self.execute_command()
        elif event.button.id == "clear":
            self.output_log.clear()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command input submission."""
        if event.input == self.command_input:
            await self.execute_command()

    async def execute_command(self) -> None:
        """Execute the current command."""
        command = self.command_input.value.strip()
        if not command:
            return

        # Add to history and clear input
        self.command_input.add_to_history(command)
        self.command_input.value = ""

        # Display command
        self.output_log.write(f"$ {command}")

        try:
            # Route command
            if command.startswith("react "):
                await self.handle_react_command(command[6:])
            elif command == "help":
                self.show_help()
            elif command == "status":
                await self.show_status()
            elif command == "metrics":
                await self.show_metrics()
            elif command == "clear":
                self.output_log.clear()
            elif command == "sessions":
                self.show_sessions()
            else:
                self.output_log.write(f"❌ Unknown command: {command}")
                self.output_log.write("💡 Type 'help' for available commands")

        except Exception as e:
            self.output_log.write(f"❌ Error executing command: {e}")
            logger.exception(f"Command execution failed: {command}")

    async def handle_react_command(self, task_description: str) -> None:
        """Handle ReAct engine commands."""
        if not self.orchestrator:
            self.output_log.write("❌ ReAct Engine not initialized")
            return

        if not task_description:
            self.output_log.write("❌ Please provide a task description")
            return

        self.output_log.write(f"🧠 Processing ReAct task: {task_description}")
        self.output_log.write("")

        # Create task
        task = EnhancedReActTask(
            description=task_description,
            priority=TaskPriority.MEDIUM,
            privacy_level="standard",
        )

        self.current_task = task
        self.status_widget.set_current_task(task)

        # Progress callback for real-time updates
        def progress_callback(progress: dict[str, Any]):
            # Update status widget
            self.status_widget.update_task_progress(progress)

            # Log progress
            iteration = progress.get("iteration", 0)
            status = progress.get("status", "unknown")
            actions = progress.get("actions_taken", 0)

            self.output_log.write(f"🔄 Iteration {iteration}: {status} ({actions} actions)")

            if progress.get("latest_step"):
                step_preview = (
                    progress["latest_step"][:100] + "..."
                    if len(progress["latest_step"]) > 100
                    else progress["latest_step"]
                )
                self.output_log.write(f"   💭 {step_preview}")

        try:
            # Process the task
            start_time = time.time()
            result = await self.orchestrator.process_enhanced_task(task, progress_callback)
            duration = time.time() - start_time

            # Display results
            success_icon = "✅" if result["success"] else "❌"
            self.output_log.write("")
            self.output_log.write(f"{success_icon} Task completed!")
            self.output_log.write(f"   Duration: {duration:.2f}s")
            self.output_log.write(f"   Iterations: {result['iterations']}")
            self.output_log.write(f"   Used local LLM: {result.get('used_local_llm', False)}")

            if result.get("result"):
                self.output_log.write("")
                self.output_log.write("📝 Result:")
                # Format result nicely
                result_text = str(result["result"])
                for line in result_text.split("\n"):
                    self.output_log.write(f"   {line}")

            self.output_log.write("")

            # Add to history
            self.history_widget.add_task_result(task, result)

        except Exception as e:
            self.output_log.write(f"❌ Task processing failed: {e}")
            logger.exception("ReAct task processing failed")

        finally:
            # Clear current task
            self.current_task = None
            self.status_widget.set_current_task(None)

    def show_help(self) -> None:
        """Display help information."""
        help_text = """
🎯 Enhanced ReAct Terminal (Toad Style) - Available Commands:

📋 BASIC COMMANDS:
  help                    - Show this help message
  status                  - Show engine status and metrics
  metrics                 - Show comprehensive system metrics
  sessions                - Show session information
  clear                   - Clear the output log

🧠 REACT ENGINE COMMANDS:
  react <description>     - Execute ReAct task with given description

💡 EXAMPLES:
  react Analyze the codebase structure and suggest improvements
  react Review security vulnerabilities in the authentication system
  react Generate comprehensive API documentation

🔧 FEATURES:
  • Smooth, flicker-free terminal updates (Toad-style)
  • Multi-agent coordination
  • Privacy-sensitive processing with local LLM
  • Web content integration
  • Real-time progress tracking
  • Command history (use ↑/↓ arrows)
  • Rich formatting and interactive elements

💻 NAVIGATION:
  • Use Tab to switch between Main/Metrics/Sessions tabs
  • Use ↑/↓ arrows for command history
  • Press Enter to execute commands
  • Use Ctrl+C to exit
"""

        for line in help_text.split("\n"):
            self.output_log.write(line)

    async def show_status(self) -> None:
        """Show detailed engine status."""
        if not self.orchestrator:
            self.output_log.write("❌ ReAct Engine not initialized")
            return

        try:
            metrics = await self.orchestrator.get_comprehensive_metrics()

            self.output_log.write("🎯 Enhanced ReAct Engine Status:")
            self.output_log.write("")

            # Orchestrator metrics
            orch_metrics = metrics["orchestrator_metrics"]
            self.output_log.write("📊 Performance Metrics:")
            self.output_log.write(f"   Tasks completed: {orch_metrics['tasks_completed']}")
            self.output_log.write(f"   Tasks failed: {orch_metrics['tasks_failed']}")
            self.output_log.write(f"   Average task time: {orch_metrics['average_task_time']:.2f}s")
            self.output_log.write(f"   LLM calls: {orch_metrics['llm_calls']}")
            self.output_log.write(f"   Local LLM calls: {orch_metrics['local_llm_calls']}")
            self.output_log.write("")

            # System capabilities
            self.output_log.write("🔧 System Capabilities:")
            self.output_log.write(
                f"   Rust extensions: {'✅' if metrics['rust_extensions'] else '❌'}"
            )
            self.output_log.write(
                f"   Local LLM: {'✅' if metrics['local_llm_available'] else '❌'}"
            )
            self.output_log.write(
                f"   Web fetch: {'✅' if metrics['web_fetch_available'] else '❌'}"
            )
            self.output_log.write("")

            # Message queue
            queue_stats = metrics["message_queue_stats"]
            self.output_log.write("📡 Message Queue:")
            self.output_log.write(f"   Messages sent: {queue_stats['sent']}")
            self.output_log.write(f"   Messages processed: {queue_stats['processed']}")
            self.output_log.write(f"   Active queues: {queue_stats['active_queues']}")
            self.output_log.write("")

        except Exception as e:
            self.output_log.write(f"❌ Failed to get status: {e}")

    async def show_metrics(self) -> None:
        """Show comprehensive system metrics."""
        # This could be enhanced to update the Metrics tab
        await self.show_status()

    def show_sessions(self) -> None:
        """Show session information."""
        self.output_log.write("🖥️ Terminal Session Information:")
        self.output_log.write("")
        self.output_log.write("   Session type: Native Terminal (Toad-style)")
        self.output_log.write("   UI Framework: Python Textual")
        self.output_log.write(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.output_log.write(f"   Commands executed: {len(self.command_input.command_history)}")
        self.output_log.write(f"   Tasks completed: {len(self.history_widget.task_history)}")
        self.output_log.write("")

        if self.current_task:
            self.output_log.write("🔄 Current Task:")
            self.output_log.write(f"   Description: {self.current_task.description}")
            self.output_log.write(f"   Priority: {self.current_task.priority.value}")
            self.output_log.write(f"   Privacy level: {self.current_task.privacy_level}")
        else:
            self.output_log.write("📋 No active task")
        self.output_log.write("")

    async def on_unmount(self) -> None:
        """Cleanup when application exits."""
        if self.orchestrator:
            try:
                await self.orchestrator.cleanup()
                self.output_log.write("🧹 ReAct Engine cleanup completed")
            except Exception as e:
                logger.exception(f"Cleanup failed: {e}")


def create_toad_style_css():
    """Create CSS for the Toad-style terminal."""
    css_content = """
/* Toad-style Enhanced ReAct Terminal CSS */

Screen {
    background: #0a0a0a;
    color: #ffffff;
}

.left-panel {
    width: 40%;
    background: #111111;
    border-right: solid #333333;
}

.right-panel {
    width: 60%;
    background: #0a0a0a;
}

.input-panel {
    height: 3;
    background: #1a1a1a;
    border-top: solid #333333;
}

Header {
    background: #1a1a1a;
    color: #00ff00;
    text-style: bold;
}

Footer {
    background: #1a1a1a;
    color: #888888;
}

Input {
    background: #222222;
    color: #ffffff;
    border: solid #555555;
}

Input:focus {
    border: solid #00ff00;
}

Button {
    background: #333333;
    color: #ffffff;
    border: solid #555555;
}

Button:hover {
    background: #555555;
}

Button.-primary {
    background: #00aa00;
    color: #ffffff;
}

Button.-primary:hover {
    background: #00ff00;
}

Log {
    background: #0a0a0a;
    border: solid #333333;
    scrollbar-background: #222222;
    scrollbar-color: #555555;
}

TabbedContent {
    border: solid #333333;
}

TabPane {
    background: #111111;
}

Static {
    background: transparent;
}

#metrics-display, #sessions-display {
    padding: 1;
    background: #111111;
    border: solid #333333;
}
"""

    css_file = Path(__file__).parent / "toad_terminal.css"
    with open(css_file, "w") as f:
        f.write(css_content)


async def main():
    """Main entry point for the Toad-style terminal."""
    # Create CSS file
    create_toad_style_css()

    print("🎯 Starting Toad-Style Enhanced ReAct Terminal...")
    print("🔧 Features:")
    print("   • Native terminal UI with Textual")
    print("   • Smooth, flicker-free updates")
    print("   • Rich formatting and interactive elements")
    print("   • Multi-agent ReAct engine integration")
    print("   • Command history and session management")
    print("")
    print("💡 Use Ctrl+C to exit")
    print("")

    try:
        app = ToadStyleReActTerminal()
        await app.run_async()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")
        logger.exception("Application error")


if __name__ == "__main__":
    asyncio.run(main())
