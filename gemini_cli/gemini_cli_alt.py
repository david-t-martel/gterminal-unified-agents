#!/usr/bin/env python3
"""Gemini CLI Client - Interactive client for the enhanced Gemini unified server.

Provides real-time WebSocket integration with the multi-tasking Gemini server,
rich terminal UI, task management, and live progress updates.
"""

import asyncio
import json
import signal
import sys
from typing import Any

import aiohttp
import click
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
import websockets
from websockets.exceptions import ConnectionClosed
from websockets.exceptions import WebSocketException

console = Console()


class GeminiCLIClient:
    """Interactive CLI client for the enhanced Gemini unified server.

    Features:
    - WebSocket connection for real-time updates
    - Task submission and monitoring
    - Batch task operations
    - Live progress tracking
    - Session management
    - Rich terminal UI
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8100",
        websocket_url: str = "ws://localhost:8100/ws",
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.websocket_url = websocket_url
        self.session = None
        self.websocket = None
        self.current_session_id = None
        self.active_tasks: dict[str, dict[str, Any]] = {}
        self.task_progress: dict[str, int] = {}
        self.shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        self.shutdown_requested = True
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        sys.exit(0)

    async def connect(self, session_id: str | None = None) -> None:
        """Connect to server and establish WebSocket connection."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession()

            # Test server connection
            async with self.session.get(f"{self.server_url}/health") as resp:
                if resp.status != 200:
                    msg = f"Server health check failed: {resp.status}"
                    raise Exception(msg)
                health_data = await resp.json()
                console.print(
                    f"[green]✅ Connected to server (status: {health_data['status']})[/green]"
                )

            # Establish WebSocket connection
            ws_url = self.websocket_url
            if session_id:
                ws_url += f"?session_id={session_id}"
                self.current_session_id = session_id

            self.websocket = await websockets.connect(ws_url)
            console.print("[green]✅ WebSocket connected for real-time updates[/green]")

            # Start WebSocket message handler
            asyncio.create_task(self._websocket_handler())

        except Exception as e:
            console.print(f"[red]❌ Connection failed: {e}[/red]")
            raise

    async def disconnect(self) -> None:
        """Disconnect from server."""
        if self.websocket:
            await self.websocket.close()
        if self.session:
            await self.session.close()
        console.print("[yellow]Disconnected from server[/yellow]")

    async def _websocket_handler(self) -> None:
        """Handle incoming WebSocket messages."""
        try:
            while not self.shutdown_requested:
                message = await self.websocket.recv()
                data = json.loads(message)
                await self._handle_websocket_message(data)
        except ConnectionClosed:
            console.print("[yellow]WebSocket connection closed[/yellow]")
        except WebSocketException as e:
            console.print(f"[red]WebSocket error: {e}[/red]")
        except Exception as e:
            console.print(f"[red]WebSocket handler error: {e}[/red]")

    async def _handle_websocket_message(self, data: dict[str, Any]) -> None:
        """Handle incoming WebSocket messages."""
        message_type = data.get("type")
        message_data = data.get("data", {})

        if message_type == "task_started":
            task_id = message_data.get("task_id")
            task_type = message_data.get("task_type")
            console.print(f"[cyan]🚀 Task started: {task_id} ({task_type})[/cyan]")
            self.active_tasks[task_id] = message_data
            self.task_progress[task_id] = 0

        elif message_type == "task_progress":
            task_id = message_data.get("task_id")
            progress = message_data.get("progress", 0)
            message = message_data.get("message", "")
            self.task_progress[task_id] = progress
            console.print(f"[blue]📊 {task_id}: {progress}% - {message}[/blue]")

        elif message_type == "task_completed":
            task_id = message_data.get("task_id")
            console.print(f"[green]✅ Task completed: {task_id}[/green]")
            self.active_tasks.pop(task_id, None)
            self.task_progress.pop(task_id, None)

        elif message_type == "task_failed":
            task_id = message_data.get("task_id")
            error = message_data.get("error", "Unknown error")
            console.print(f"[red]❌ Task failed: {task_id} - {error}[/red]")
            self.active_tasks.pop(task_id, None)
            self.task_progress.pop(task_id, None)

        elif message_type == "task_cancelled":
            task_id = message_data.get("task_id")
            reason = message_data.get("reason", "")
            console.print(f"[yellow]⚠️ Task cancelled: {task_id} - {reason}[/yellow]")
            self.active_tasks.pop(task_id, None)
            self.task_progress.pop(task_id, None)

        elif message_type == "connection_established":
            connection_id = message_data.get("connection_id")
            session_id = message_data.get("session_id")
            console.print(f"[green]🔗 Connection established: {connection_id}[/green]")
            if session_id:
                self.current_session_id = session_id
                console.print(f"[blue]📋 Session: {session_id}[/blue]")

    async def submit_task(self, task_type: str, instruction: str, **kwargs) -> str:
        """Submit a single task to the server."""
        task_data = {
            "task_type": task_type,
            "instruction": instruction,
            "session_id": self.current_session_id,
            **kwargs,
        }

        try:
            async with self.session.post(f"{self.server_url}/task", json=task_data) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    msg = f"Task submission failed: {resp.status} - {error_text}"
                    raise Exception(msg)

                result = await resp.json()
                task_id = result.get("session_id")  # Server returns session_id, but we want task_id
                console.print("[green]✅ Task submitted successfully[/green]")
                return task_id

        except Exception as e:
            console.print(f"[red]❌ Task submission failed: {e}[/red]")
            raise

    async def submit_batch_tasks(
        self, tasks: list[dict[str, Any]], execute_parallel: bool = True
    ) -> list[str]:
        """Submit multiple tasks as a batch."""
        batch_data = {
            "tasks": tasks,
            "execute_parallel": execute_parallel,
            "stop_on_first_error": False,
        }

        try:
            async with self.session.post(f"{self.server_url}/tasks/batch", json=batch_data) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    msg = f"Batch submission failed: {resp.status} - {error_text}"
                    raise Exception(msg)

                result = await resp.json()
                task_ids = result.get("task_ids", [])
                console.print(f"[green]✅ Batch submitted: {len(task_ids)} tasks[/green]")
                return task_ids

        except Exception as e:
            console.print(f"[red]❌ Batch submission failed: {e}[/red]")
            raise

    async def list_tasks(
        self, status_filter: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List tasks with optional filtering."""
        params = {"limit": limit}
        if status_filter:
            params["status"] = status_filter
        if self.current_session_id:
            params["session_id"] = self.current_session_id

        try:
            async with self.session.get(f"{self.server_url}/tasks", params=params) as resp:
                if resp.status != 200:
                    msg = f"Task listing failed: {resp.status}"
                    raise Exception(msg)

                result = await resp.json()
                return result.get("tasks", [])

        except Exception as e:
            console.print(f"[red]❌ Task listing failed: {e}[/red]")
            return []

    async def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get detailed status of a specific task."""
        try:
            async with self.session.get(f"{self.server_url}/tasks/{task_id}") as resp:
                if resp.status == 404:
                    return None
                if resp.status != 200:
                    msg = f"Status retrieval failed: {resp.status}"
                    raise Exception(msg)

                result = await resp.json()
                return result.get("task")

        except Exception as e:
            console.print(f"[red]❌ Status retrieval failed: {e}[/red]")
            return None

    async def cancel_task(self, task_id: str, reason: str = "User requested") -> bool:
        """Cancel a task."""
        try:
            async with self.session.delete(
                f"{self.server_url}/tasks/{task_id}", params={"reason": reason}
            ) as resp:
                if resp.status != 200:
                    return False

                result = await resp.json()
                return result.get("status") == "cancelled"

        except Exception as e:
            console.print(f"[red]❌ Task cancellation failed: {e}[/red]")
            return False

    async def get_metrics(self) -> dict[str, Any]:
        """Get server and queue metrics."""
        try:
            async with self.session.get(f"{self.server_url}/tasks/metrics") as resp:
                if resp.status != 200:
                    msg = f"Metrics retrieval failed: {resp.status}"
                    raise Exception(msg)

                return await resp.json()

        except Exception as e:
            console.print(f"[red]❌ Metrics retrieval failed: {e}[/red]")
            return {}

    def display_task_table(self, tasks: list[dict[str, Any]]) -> None:
        """Display tasks in a formatted table."""
        if not tasks:
            console.print("[yellow]No tasks found[/yellow]")
            return

        table = Table(title="Tasks")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Progress", style="blue")
        table.add_column("Created", style="white")

        for task in tasks:
            task_id = (
                task.get("task_id", "")[:12] + "..."
                if len(task.get("task_id", "")) > 12
                else task.get("task_id", "")
            )
            task_type = task.get("task_type", "")
            status = task.get("status", "")
            progress = f"{self.task_progress.get(task.get('task_id', ''), 0)}%"
            created = task.get("created_at", "")[:19] if task.get("created_at") else ""

            table.add_row(task_id, task_type, status, progress, created)

        console.print(table)

    def display_metrics(self, metrics: dict[str, Any]) -> None:
        """Display server metrics."""
        queue_metrics = metrics.get("queue_metrics", {})
        websocket_stats = metrics.get("websocket_stats", {})

        # Queue metrics table
        queue_table = Table(title="Queue Metrics")
        queue_table.add_column("Metric", style="cyan")
        queue_table.add_column("Value", style="white")

        for key, value in queue_metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    queue_table.add_row(f"{key}.{sub_key}", str(sub_value))
            else:
                queue_table.add_row(key, str(value))

        # WebSocket stats table
        ws_table = Table(title="WebSocket Statistics")
        ws_table.add_column("Metric", style="cyan")
        ws_table.add_column("Value", style="white")

        for key, value in websocket_stats.items():
            ws_table.add_row(key, str(value))

        console.print(queue_table)
        console.print(ws_table)


@click.group()
@click.option("--server", default="http://localhost:8100", help="Server URL")
@click.option("--websocket", default="ws://localhost:8100/ws", help="WebSocket URL")
@click.pass_context
def cli(ctx, server: str, websocket: str) -> None:
    """Gemini CLI Client - Interactive client for enhanced Gemini server."""
    ctx.ensure_object(dict)
    ctx.obj["server_url"] = server
    ctx.obj["websocket_url"] = websocket


@cli.command()
@click.option("--session-id", help="Session ID to connect to")
@click.pass_context
def interactive(ctx, session_id: str) -> None:
    """Start interactive mode with real-time updates."""

    async def _interactive() -> None:
        client = GeminiCLIClient(ctx.obj["server_url"], ctx.obj["websocket_url"])

        try:
            await client.connect(session_id)

            console.print("\n[bold cyan]🚀 Gemini CLI Interactive Mode[/bold cyan]")
            console.print("Type 'help' for commands, 'quit' to exit\n")

            while not client.shutdown_requested:
                try:
                    command = await asyncio.to_thread(
                        Prompt.ask, "[bold green]gemini>[/bold green]", default=""
                    )

                    if not command:
                        continue

                    if command.lower() in ["quit", "exit", "q"]:
                        break
                    if command.lower() == "help":
                        console.print(
                            """
[bold]Available Commands:[/bold]
  submit <type> <instruction>  - Submit a task
  list [status]               - List tasks
  status <task_id>            - Get task status
  cancel <task_id>            - Cancel a task
  metrics                     - Show server metrics
  batch                       - Submit batch tasks
  clear                       - Clear screen
  help                        - Show this help
  quit/exit                   - Exit
                        """,
                        )
                    elif command.lower() == "clear":
                        console.clear()
                    elif command.lower() == "metrics":
                        metrics = await client.get_metrics()
                        client.display_metrics(metrics)
                    elif command.lower().startswith("list"):
                        parts = command.split()
                        status_filter = parts[1] if len(parts) > 1 else None
                        tasks = await client.list_tasks(status_filter)
                        client.display_task_table(tasks)
                    elif command.lower().startswith("submit"):
                        parts = command.split(maxsplit=2)
                        if len(parts) >= 3:
                            task_type = parts[1]
                            instruction = parts[2]
                            await client.submit_task(task_type, instruction)
                        else:
                            console.print("[red]Usage: submit <type> <instruction>[/red]")
                    elif command.lower().startswith("status"):
                        parts = command.split()
                        if len(parts) >= 2:
                            task_id = parts[1]
                            status = await client.get_task_status(task_id)
                            if status:
                                console.print_json(data=status)
                            else:
                                console.print("[yellow]Task not found[/yellow]")
                        else:
                            console.print("[red]Usage: status <task_id>[/red]")
                    elif command.lower().startswith("cancel"):
                        parts = command.split()
                        if len(parts) >= 2:
                            task_id = parts[1]
                            success = await client.cancel_task(task_id)
                            if success:
                                console.print("[green]✅ Task cancelled[/green]")
                            else:
                                console.print("[red]❌ Failed to cancel task[/red]")
                        else:
                            console.print("[red]Usage: cancel <task_id>[/red]")
                    elif command.lower() == "batch":
                        console.print(
                            "[cyan]Batch mode - enter tasks (empty line to finish):[/cyan]"
                        )
                        tasks: list[Any] = []
                        while True:
                            task_input = await asyncio.to_thread(
                                Prompt.ask,
                                f"[blue]Task {len(tasks) + 1}>[/blue]",
                                default="",
                            )
                            if not task_input:
                                break
                            parts = task_input.split(maxsplit=1)
                            if len(parts) >= 2:
                                tasks.append(
                                    {
                                        "task_type": parts[0],
                                        "instruction": parts[1],
                                        "session_id": client.current_session_id,
                                    },
                                )

                        if tasks:
                            await client.submit_batch_tasks(tasks)
                        else:
                            console.print("[yellow]No tasks submitted[/yellow]")
                    else:
                        console.print(f"[red]Unknown command: {command}[/red]")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Command error: {e}[/red]")

        finally:
            await client.disconnect()

    asyncio.run(_interactive())


@cli.command()
@click.argument("task_type")
@click.argument("instruction")
@click.option("--session-id", help="Session ID")
@click.option(
    "--priority",
    type=click.Choice(["low", "normal", "high", "urgent"]),
    default="normal",
)
@click.option("--timeout", type=int, default=300, help="Timeout in seconds")
@click.option("--wait", is_flag=True, help="Wait for completion")
@click.pass_context
def submit(
    ctx,
    task_type: str,
    instruction: str,
    session_id: str,
    priority: str,
    timeout: int,
    wait: bool,
) -> None:
    """Submit a single task."""

    async def _submit() -> None:
        client = GeminiCLIClient(ctx.obj["server_url"], ctx.obj["websocket_url"])

        try:
            await client.connect(session_id)

            task_id = await client.submit_task(
                task_type=task_type,
                instruction=instruction,
                priority=priority,
                timeout_seconds=timeout,
            )

            console.print(f"[green]✅ Task submitted: {task_id}[/green]")

            if wait:
                console.print("[cyan]Waiting for completion...[/cyan]")
                # Wait for task completion via WebSocket updates
                while task_id in client.active_tasks:
                    await asyncio.sleep(1)

                # Get final status
                status = await client.get_task_status(task_id)
                if status:
                    console.print_json(data=status)

        finally:
            await client.disconnect()

    asyncio.run(_submit())


@cli.command()
@click.option("--status", help="Filter by status")
@click.option("--limit", type=int, default=50, help="Maximum number of tasks")
@click.pass_context
def list_tasks(ctx, status: str, limit: int) -> None:
    """List tasks."""

    async def _list() -> None:
        client = GeminiCLIClient(ctx.obj["server_url"], ctx.obj["websocket_url"])

        try:
            await client.connect()
            tasks = await client.list_tasks(status, limit)
            client.display_task_table(tasks)

        finally:
            await client.disconnect()

    asyncio.run(_list())


@cli.command()
@click.pass_context
def metrics(ctx) -> None:
    """Show server metrics."""

    async def _metrics() -> None:
        client = GeminiCLIClient(ctx.obj["server_url"], ctx.obj["websocket_url"])

        try:
            await client.connect()
            metrics_data = await client.get_metrics()
            client.display_metrics(metrics_data)

        finally:
            await client.disconnect()

    asyncio.run(_metrics())


if __name__ == "__main__":
    cli()
