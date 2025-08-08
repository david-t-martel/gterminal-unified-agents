#!/usr/bin/env python3
"""Web Terminal Server - Browser-based ReAct Engine Interface.

This server provides a web-based terminal interface for the Enhanced ReAct Engine
using WebSockets for real-time communication and xterm.js for terminal emulation.

Features:
- Real-time terminal emulation in browser
- WebSocket-based communication
- Streaming ReAct engine output
- Interactive command interface
- Session persistence and restoration
- Multi-tab support with session management
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any
import uuid

from fastapi import FastAPI
from fastapi import Request
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import our enhanced orchestrator
from .enhanced_react_orchestrator import EnhancedReActOrchestrator
from .enhanced_react_orchestrator import EnhancedReActTask
from .enhanced_react_orchestrator import TaskPriority
from .enhanced_react_orchestrator import create_enhanced_orchestrator

logger = logging.getLogger(__name__)


class TerminalSession:
    """Represents a terminal session with history and state."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.history: list[dict[str, Any]] = []
        self.current_task: EnhancedReActTask | None = None
        self.websocket: WebSocket | None = None
        self.orchestrator: EnhancedReActOrchestrator | None = None

    async def send_output(self, output: str, output_type: str = "stdout") -> None:
        """Send output to the connected WebSocket."""
        if self.websocket:
            try:
                message = {
                    "type": "output",
                    "output_type": output_type,
                    "content": output,
                    "timestamp": datetime.now().isoformat(),
                }
                await self.websocket.send_json(message)
            except Exception as e:
                logger.exception(f"Failed to send output to WebSocket: {e}")

    async def send_progress(self, progress: dict[str, Any]) -> None:
        """Send progress update to the WebSocket."""
        if self.websocket:
            try:
                message = {
                    "type": "progress",
                    "progress": progress,
                    "timestamp": datetime.now().isoformat(),
                }
                await self.websocket.send_json(message)
            except Exception as e:
                logger.exception(f"Failed to send progress to WebSocket: {e}")

    def add_to_history(self, command: str, result: dict[str, Any]) -> None:
        """Add command and result to session history."""
        self.history.append(
            {
                "command": command,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.last_activity = datetime.now()

    async def cleanup(self) -> None:
        """Cleanup session resources."""
        if self.orchestrator:
            await self.orchestrator.cleanup()


class WebTerminalManager:
    """Manages web terminal sessions and orchestrator instances."""

    def __init__(self):
        self.sessions: dict[str, TerminalSession] = {}
        self.active_connections: dict[str, WebSocket] = {}

    async def create_session(self, session_id: str | None = None) -> TerminalSession:
        """Create a new terminal session."""
        if session_id is None:
            session_id = f"term_{uuid.uuid4().hex[:8]}"

        session = TerminalSession(session_id)

        # Initialize orchestrator for the session
        session.orchestrator = await create_enhanced_orchestrator(
            enable_local_llm=True,
            enable_web_fetch=True,
        )

        self.sessions[session_id] = session
        logger.info(f"Created terminal session: {session_id}")

        return session

    async def get_session(self, session_id: str) -> TerminalSession | None:
        """Get an existing session or create a new one."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        return await self.create_session(session_id)

    async def connect_websocket(self, session_id: str, websocket: WebSocket) -> TerminalSession:
        """Connect a WebSocket to a session."""
        session = await self.get_session(session_id)
        session.websocket = websocket
        self.active_connections[session_id] = websocket

        # Send welcome message
        await session.send_output("🎉 Enhanced ReAct Terminal Connected!", "info")
        await session.send_output("Type 'help' for available commands.\n", "info")

        return session

    async def disconnect_websocket(self, session_id: str) -> None:
        """Disconnect WebSocket from session."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.websocket = None

    async def cleanup_session(self, session_id: str) -> None:
        """Cleanup and remove a session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            await session.cleanup()
            del self.sessions[session_id]

        if session_id in self.active_connections:
            del self.active_connections[session_id]

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all active sessions."""
        return [
            {
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "connected": session.websocket is not None,
                "history_count": len(session.history),
                "current_task": session.current_task.task_id if session.current_task else None,
            }
            for session_id, session in self.sessions.items()
        ]


# Global terminal manager
terminal_manager = WebTerminalManager()

# FastAPI application
app = FastAPI(title="Enhanced ReAct Web Terminal", version="1.0.0")

# Set up templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

# Create directories if they don't exist
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class CommandProcessor:
    """Processes terminal commands and integrates with ReAct engine."""

    def __init__(self, session: TerminalSession):
        self.session = session

    async def process_command(self, command: str) -> dict[str, Any]:
        """Process a terminal command and return the result."""
        command = command.strip()

        if not command:
            return {"output": "", "success": True}

        # Add command to history
        await self.session.send_output(f"$ {command}\n", "command")

        try:
            # Route command to appropriate handler
            if command.startswith("react "):
                return await self._handle_react_command(command[6:])
            elif command == "help":
                return await self._handle_help_command()
            elif command == "status":
                return await self._handle_status_command()
            elif command == "sessions":
                return await self._handle_sessions_command()
            elif command == "clear":
                return await self._handle_clear_command()
            elif command.startswith("task "):
                return await self._handle_task_command(command[5:])
            elif command == "metrics":
                return await self._handle_metrics_command()
            elif command.startswith("history"):
                return await self._handle_history_command(command)
            else:
                return await self._handle_unknown_command(command)

        except Exception as e:
            logger.exception(f"Command processing failed: {e}")
            return {"output": f"❌ Error processing command: {e}\n", "success": False}

    async def _handle_react_command(self, task_description: str) -> dict[str, Any]:
        """Handle ReAct engine commands."""
        if not task_description:
            return {
                "output": "❌ Please provide a task description. Usage: react <task description>\n",
                "success": False,
            }

        await self.session.send_output(
            f"🧠 Starting ReAct processing for: {task_description}\n", "info"
        )

        # Create task
        task = EnhancedReActTask(
            description=task_description,
            priority=TaskPriority.MEDIUM,
            privacy_level="standard",
        )

        self.session.current_task = task

        # Progress callback for real-time updates
        async def progress_callback(progress: dict[str, Any]):
            await self.session.send_progress(progress)
            await self.session.send_output(
                f"🔄 Iteration {progress['iteration']}: {progress['status']}\n", "progress"
            )

        try:
            # Process the task
            result = await self.session.orchestrator.process_enhanced_task(task, progress_callback)

            # Send results
            success_indicator = "✅" if result["success"] else "❌"
            await self.session.send_output(
                f"\n{success_indicator} ReAct task completed!\n", "result"
            )
            await self.session.send_output(
                f"   Execution time: {result['execution_time']:.2f}s\n", "result"
            )
            await self.session.send_output(f"   Iterations: {result['iterations']}\n", "result")

            if result.get("result"):
                await self.session.send_output(f"\n📝 Result:\n{result['result']}\n\n", "result")

            self.session.current_task = None

            return {
                "output": "ReAct processing complete.",
                "success": result["success"],
                "result": result,
            }

        except Exception as e:
            self.session.current_task = None
            return {"output": f"❌ ReAct processing failed: {e}\n", "success": False}

    async def _handle_help_command(self) -> dict[str, Any]:
        """Display help information."""
        help_text = """
🎯 Enhanced ReAct Web Terminal - Available Commands:

📋 BASIC COMMANDS:
  help                    - Show this help message
  status                  - Show terminal and orchestrator status
  sessions                - List all terminal sessions
  clear                   - Clear the terminal screen
  metrics                 - Show comprehensive system metrics
  history [count]         - Show command history (default: 10)

🧠 REACT ENGINE COMMANDS:
  react <description>     - Execute ReAct task with given description
  task <subcommand>       - Task management commands:
    task list             - List recent tasks
    task status           - Show current task status
    task cancel           - Cancel current task

💡 EXAMPLES:
  react Analyze the codebase and suggest improvements
  react Review security vulnerabilities in the project
  react Generate comprehensive documentation for the API

🔧 FEATURES:
  • Multi-agent coordination
  • Privacy-sensitive processing with local LLM
  • Web content integration
  • Real-time progress updates
  • Session persistence and restoration

Type any command and press Enter to execute.
"""

        await self.session.send_output(help_text, "info")
        return {"output": "Help displayed", "success": True}

    async def _handle_status_command(self) -> dict[str, Any]:
        """Show system status."""
        if not self.session.orchestrator:
            return {"output": "❌ Orchestrator not initialized\n", "success": False}

        try:
            metrics = await self.session.orchestrator.get_comprehensive_metrics()

            status_text = f"""
🎯 Enhanced ReAct Terminal Status:

📊 SESSION INFO:
  Session ID: {self.session.session_id}
  Created: {self.session.created_at.strftime("%Y-%m-%d %H:%M:%S")}
  Commands run: {len(self.session.history)}
  Current task: {"Active" if self.session.current_task else "None"}

🧠 ORCHESTRATOR STATUS:
  Tasks completed: {metrics["orchestrator_metrics"]["tasks_completed"]}
  Tasks failed: {metrics["orchestrator_metrics"]["tasks_failed"]}
  Average task time: {metrics["orchestrator_metrics"]["average_task_time"]:.2f}s
  LLM calls: {metrics["orchestrator_metrics"]["llm_calls"]}
  Local LLM calls: {metrics["orchestrator_metrics"]["local_llm_calls"]}

🔧 SYSTEM CAPABILITIES:
  Rust extensions: {"✅" if metrics["rust_extensions"] else "❌"}
  Local LLM: {"✅" if metrics["local_llm_available"] else "❌"}
  Web fetch: {"✅" if metrics["web_fetch_available"] else "❌"}
  Message queue: {metrics["message_queue_stats"]["processed"]} processed

📡 MESSAGE QUEUE:
  Active queues: {metrics["message_queue_stats"]["active_queues"]}
  Messages sent: {metrics["message_queue_stats"]["sent"]}
  Messages delivered: {metrics["message_queue_stats"]["delivered"]}
"""

            await self.session.send_output(status_text, "info")
            return {"output": "Status displayed", "success": True}

        except Exception as e:
            return {"output": f"❌ Failed to get status: {e}\n", "success": False}

    async def _handle_sessions_command(self) -> dict[str, Any]:
        """List all terminal sessions."""
        sessions = terminal_manager.list_sessions()

        if not sessions:
            await self.session.send_output("No active sessions.\n", "info")
            return {"output": "No sessions", "success": True}

        sessions_text = "🖥️  Active Terminal Sessions:\n\n"
        for session_info in sessions:
            status_icon = "🟢" if session_info["connected"] else "🔴"
            sessions_text += f"{status_icon} {session_info['session_id']}\n"
            sessions_text += f"   Created: {session_info['created_at']}\n"
            sessions_text += f"   History: {session_info['history_count']} commands\n"
            if session_info["current_task"]:
                sessions_text += f"   Current task: {session_info['current_task']}\n"
            sessions_text += "\n"

        await self.session.send_output(sessions_text, "info")
        return {"output": "Sessions listed", "success": True}

    async def _handle_clear_command(self) -> dict[str, Any]:
        """Clear the terminal screen."""
        clear_message = {
            "type": "clear",
            "timestamp": datetime.now().isoformat(),
        }
        await self.session.websocket.send_json(clear_message)
        return {"output": "Terminal cleared", "success": True}

    async def _handle_task_command(self, subcommand: str) -> dict[str, Any]:
        """Handle task management commands."""
        parts = subcommand.split()
        if not parts:
            return {
                "output": "❌ Task subcommand required. Use: task list|status|cancel\n",
                "success": False,
            }

        action = parts[0]

        if action == "list":
            # Show recent tasks from history
            recent_tasks = [h for h in self.session.history if h.get("result", {}).get("result")][
                -5:
            ]
            if not recent_tasks:
                await self.session.send_output("No recent tasks found.\n", "info")
            else:
                tasks_text = "📋 Recent Tasks:\n\n"
                for i, task_history in enumerate(recent_tasks, 1):
                    result = task_history.get("result", {})
                    success_icon = "✅" if result.get("success") else "❌"
                    tasks_text += f"{i}. {success_icon} {task_history['command']}\n"
                    tasks_text += f"   Time: {result.get('execution_time', 0):.2f}s\n"
                    tasks_text += f"   Timestamp: {task_history['timestamp']}\n\n"
                await self.session.send_output(tasks_text, "info")
            return {"output": "Task list displayed", "success": True}

        elif action == "status":
            if self.session.current_task:
                await self.session.send_output(
                    f"🔄 Current task: {self.session.current_task.description}\n", "info"
                )
                await self.session.send_output(
                    f"   Priority: {self.session.current_task.priority.value}\n", "info"
                )
                await self.session.send_output(
                    f"   Privacy level: {self.session.current_task.privacy_level}\n", "info"
                )
            else:
                await self.session.send_output("No active task.\n", "info")
            return {"output": "Task status displayed", "success": True}

        elif action == "cancel":
            if self.session.current_task:
                self.session.current_task = None
                await self.session.send_output("⏹️  Current task cancelled.\n", "info")
                return {"output": "Task cancelled", "success": True}
            else:
                await self.session.send_output("No active task to cancel.\n", "info")
                return {"output": "No task to cancel", "success": True}

        else:
            return {"output": f"❌ Unknown task subcommand: {action}\n", "success": False}

    async def _handle_metrics_command(self) -> dict[str, Any]:
        """Show comprehensive system metrics."""
        try:
            metrics = await self.session.orchestrator.get_comprehensive_metrics()

            metrics_text = f"""
📊 Comprehensive System Metrics:

🎯 ORCHESTRATOR PERFORMANCE:
  Tasks completed: {metrics["orchestrator_metrics"]["tasks_completed"]}
  Tasks failed: {metrics["orchestrator_metrics"]["tasks_failed"]}
  Success rate: {(metrics["orchestrator_metrics"]["tasks_completed"] / max(1, metrics["orchestrator_metrics"]["tasks_completed"] + metrics["orchestrator_metrics"]["tasks_failed"]) * 100):.1f}%
  Average task time: {metrics["orchestrator_metrics"]["average_task_time"]:.2f}s
  Cache hits: {metrics["orchestrator_metrics"]["cache_hits"]}
  LLM calls (cloud): {metrics["orchestrator_metrics"]["llm_calls"]}
  LLM calls (local): {metrics["orchestrator_metrics"]["local_llm_calls"]}

📡 MESSAGE QUEUE METRICS:
  Messages sent: {metrics["message_queue_stats"]["sent"]}
  Messages delivered: {metrics["message_queue_stats"]["delivered"]}
  Messages processed: {metrics["message_queue_stats"]["processed"]}
  Failed deliveries: {metrics["message_queue_stats"]["failed"]}
  Active queues: {metrics["message_queue_stats"]["active_queues"]}

🔧 COMPONENT STATUS:
  Rust extensions: {"🟢 Active" if metrics["rust_extensions"] else "🔴 Unavailable"}
  Local LLM: {"🟢 Available" if metrics["local_llm_available"] else "🔴 Unavailable"}
  Web fetch service: {"🟢 Available" if metrics["web_fetch_available"] else "🔴 Unavailable"}

🖥️  TERMINAL SESSION:
  Session ID: {self.session.session_id}
  Uptime: {datetime.now() - self.session.created_at}
  Commands executed: {len(self.session.history)}
  Current task active: {"Yes" if self.session.current_task else "No"}
"""

            await self.session.send_output(metrics_text, "info")
            return {"output": "Metrics displayed", "success": True}

        except Exception as e:
            return {"output": f"❌ Failed to get metrics: {e}\n", "success": False}

    async def _handle_history_command(self, command: str) -> dict[str, Any]:
        """Show command history."""
        parts = command.split()
        count = 10  # default

        if len(parts) > 1:
            try:
                count = int(parts[1])
            except ValueError:
                return {"output": "❌ Invalid count for history command\n", "success": False}

        if not self.session.history:
            await self.session.send_output("No command history available.\n", "info")
            return {"output": "No history", "success": True}

        recent_history = self.session.history[-count:]

        history_text = f"📜 Command History (last {len(recent_history)} commands):\n\n"
        for i, entry in enumerate(recent_history, 1):
            timestamp = entry["timestamp"][:19]  # Remove microseconds
            success_icon = "✅" if entry.get("result", {}).get("success", True) else "❌"
            history_text += f"{i:2d}. {success_icon} {entry['command']}\n"
            history_text += f"     {timestamp}\n\n"

        await self.session.send_output(history_text, "info")
        return {"output": "History displayed", "success": True}

    async def _handle_unknown_command(self, command: str) -> dict[str, Any]:
        """Handle unknown commands."""
        suggestions = []

        if "help" in command.lower():
            suggestions.append("help")
        if "status" in command.lower():
            suggestions.append("status")
        if any(word in command.lower() for word in ["run", "execute", "do", "perform"]):
            suggestions.append("react <description>")

        output = f"❌ Unknown command: '{command}'\n"
        if suggestions:
            output += f"Did you mean: {', '.join(suggestions)}\n"
        output += "Type 'help' for available commands.\n"

        await self.session.send_output(output, "error")
        return {"output": output, "success": False}


@app.get("/", response_class=HTMLResponse)
async def get_terminal(request: Request):
    """Serve the main terminal interface."""
    return templates.TemplateResponse("terminal.html", {"request": request})


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for terminal communication."""
    await websocket.accept()
    logger.info(f"WebSocket connected for session: {session_id}")

    try:
        # Connect session to WebSocket
        session = await terminal_manager.connect_websocket(session_id, websocket)
        processor = CommandProcessor(session)

        while True:
            # Receive command from client
            data = await websocket.receive_json()
            command = data.get("command", "").strip()

            if command:
                # Process command
                result = await processor.process_command(command)
                session.add_to_history(command, result)

                # Send completion message
                completion_message = {
                    "type": "complete",
                    "success": result.get("success", True),
                    "timestamp": datetime.now().isoformat(),
                }
                await websocket.send_json(completion_message)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.exception(f"WebSocket error for session {session_id}: {e}")
    finally:
        await terminal_manager.disconnect_websocket(session_id)


@app.get("/api/sessions")
async def list_sessions():
    """API endpoint to list all sessions."""
    return {"sessions": terminal_manager.list_sessions()}


@app.post("/api/sessions/{session_id}/cleanup")
async def cleanup_session(session_id: str):
    """API endpoint to cleanup a session."""
    await terminal_manager.cleanup_session(session_id)
    return {"message": f"Session {session_id} cleaned up"}


def create_terminal_html():
    """Create the terminal HTML template."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced ReAct Web Terminal</title>
    <link rel="stylesheet" href="https://unpkg.com/xterm@5.3.0/css/xterm.css" />
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #000;
            color: #fff;
            font-family: 'Courier New', monospace;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 20px;
            padding: 10px;
            border-bottom: 2px solid #333;
        }

        .header h1 {
            color: #00ff00;
            margin: 0;
        }

        .header p {
            color: #888;
            margin: 5px 0 0 0;
        }

        .terminal-container {
            border: 2px solid #333;
            border-radius: 8px;
            padding: 10px;
            background-color: #111;
            height: 70vh;
            overflow: hidden;
        }

        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background-color: #222;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 12px;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .status-connected {
            color: #00ff00;
        }

        .status-disconnected {
            color: #ff0000;
        }

        .controls {
            margin-top: 10px;
            text-align: center;
        }

        .controls button {
            background-color: #333;
            color: #fff;
            border: 1px solid #555;
            padding: 8px 16px;
            margin: 0 5px;
            cursor: pointer;
            border-radius: 4px;
        }

        .controls button:hover {
            background-color: #555;
        }

        .progress-bar {
            width: 100%;
            height: 4px;
            background-color: #333;
            border-radius: 2px;
            overflow: hidden;
            margin: 5px 0;
        }

        .progress-fill {
            height: 100%;
            background-color: #00ff00;
            width: 0%;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Enhanced ReAct Web Terminal</h1>
            <p>Interactive AI Agent System with Multi-Agent Coordination</p>
        </div>

        <div class="terminal-container" id="terminal"></div>

        <div class="status-bar">
            <div class="status-item">
                <span>Connection:</span>
                <span id="connection-status" class="status-disconnected">Disconnected</span>
            </div>
            <div class="status-item">
                <span>Session:</span>
                <span id="session-id">-</span>
            </div>
            <div class="status-item">
                <span>Commands:</span>
                <span id="command-count">0</span>
            </div>
            <div class="status-item">
                <span>Uptime:</span>
                <span id="uptime">0s</span>
            </div>
        </div>

        <div id="progress-container" style="display: none;">
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
            <div id="progress-text" style="text-align: center; color: #888; font-size: 12px; margin-top: 5px;"></div>
        </div>

        <div class="controls">
            <button onclick="clearTerminal()">Clear Terminal</button>
            <button onclick="showHelp()">Help</button>
            <button onclick="showStatus()">Status</button>
            <button onclick="reconnect()">Reconnect</button>
        </div>
    </div>

    <script src="https://unpkg.com/xterm@5.3.0/lib/xterm.js"></script>
    <script src="https://unpkg.com/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.js"></script>
    <script>
        // Initialize xterm.js
        const terminal = new Terminal({
            theme: {
                background: '#111111',
                foreground: '#ffffff',
                cursor: '#00ff00',
                cursorAccent: '#00ff00',
                selection: '#333333',
                black: '#000000',
                red: '#ff0000',
                green: '#00ff00',
                yellow: '#ffff00',
                blue: '#0000ff',
                magenta: '#ff00ff',
                cyan: '#00ffff',
                white: '#ffffff',
                brightBlack: '#333333',
                brightRed: '#ff5555',
                brightGreen: '#55ff55',
                brightYellow: '#ffff55',
                brightBlue: '#5555ff',
                brightMagenta: '#ff55ff',
                brightCyan: '#55ffff',
                brightWhite: '#ffffff'
            },
            fontSize: 14,
            fontFamily: '"Courier New", monospace',
            cursorBlink: true,
            cursorStyle: 'block',
        });

        const fitAddon = new FitAddon.FitAddon();
        terminal.loadAddon(fitAddon);

        terminal.open(document.getElementById('terminal'));
        fitAddon.fit();

        // WebSocket connection
        let socket = null;
        let sessionId = generateSessionId();
        let commandCount = 0;
        let startTime = Date.now();
        let currentCommand = '';

        function generateSessionId() {
            return 'web_' + Math.random().toString(36).substr(2, 8);
        }

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;

            socket = new WebSocket(wsUrl);

            socket.onopen = function() {
                updateConnectionStatus(true);
                terminal.writeln('\\r\\n🎉 Connected to Enhanced ReAct Terminal!');
                terminal.writeln('Type "help" for available commands.\\r\\n');
                showPrompt();
            };

            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            socket.onclose = function() {
                updateConnectionStatus(false);
                terminal.writeln('\\r\\n❌ Connection closed. Click Reconnect to restore connection.\\r\\n');
            };

            socket.onerror = function(error) {
                updateConnectionStatus(false);
                terminal.writeln(`\\r\\n❌ Connection error: ${error}\\r\\n`);
            };
        }

        function handleMessage(data) {
            switch (data.type) {
                case 'output':
                    if (data.output_type === 'command') {
                        terminal.write('\\r\\n' + data.content);
                    } else if (data.output_type === 'progress') {
                        terminal.write('\\r' + data.content);
                    } else {
                        terminal.write(data.content);
                    }
                    break;

                case 'progress':
                    handleProgress(data.progress);
                    break;

                case 'clear':
                    terminal.clear();
                    showPrompt();
                    break;

                case 'complete':
                    hideProgress();
                    showPrompt();
                    break;
            }
        }

        function handleProgress(progress) {
            showProgress();
            const progressPercent = (progress.iteration / 20) * 100; // Assume max 20 iterations
            document.getElementById('progress-fill').style.width = progressPercent + '%';
            document.getElementById('progress-text').textContent =
                `Iteration ${progress.iteration} - ${progress.status} - ${progress.actions_taken || 0} actions`;
        }

        function showProgress() {
            document.getElementById('progress-container').style.display = 'block';
        }

        function hideProgress() {
            document.getElementById('progress-container').style.display = 'none';
        }

        function showPrompt() {
            terminal.write('\\r\\n$ ');
        }

        function updateConnectionStatus(connected) {
            const statusEl = document.getElementById('connection-status');
            const sessionEl = document.getElementById('session-id');

            if (connected) {
                statusEl.textContent = 'Connected';
                statusEl.className = 'status-connected';
                sessionEl.textContent = sessionId;
            } else {
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'status-disconnected';
                sessionEl.textContent = '-';
            }
        }

        function updateCommandCount() {
            document.getElementById('command-count').textContent = commandCount;
        }

        function updateUptime() {
            const uptimeSeconds = Math.floor((Date.now() - startTime) / 1000);
            const hours = Math.floor(uptimeSeconds / 3600);
            const minutes = Math.floor((uptimeSeconds % 3600) / 60);
            const seconds = uptimeSeconds % 60;

            let uptimeStr = '';
            if (hours > 0) uptimeStr += hours + 'h ';
            if (minutes > 0) uptimeStr += minutes + 'm ';
            uptimeStr += seconds + 's';

            document.getElementById('uptime').textContent = uptimeStr;
        }

        // Terminal input handling
        terminal.onData(function(data) {
            if (data.charCodeAt(0) === 13) { // Enter key
                if (currentCommand.trim()) {
                    sendCommand(currentCommand.trim());
                    commandCount++;
                    updateCommandCount();
                    currentCommand = '';
                } else {
                    showPrompt();
                }
            } else if (data.charCodeAt(0) === 127) { // Backspace
                if (currentCommand.length > 0) {
                    currentCommand = currentCommand.slice(0, -1);
                    terminal.write('\\b \\b');
                }
            } else if (data.charCodeAt(0) >= 32) { // Printable characters
                currentCommand += data;
                terminal.write(data);
            }
        });

        function sendCommand(command) {
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({
                    command: command
                }));
            } else {
                terminal.writeln('\\r\\n❌ Not connected. Please reconnect.\\r\\n');
                showPrompt();
            }
        }

        // Control functions
        function clearTerminal() {
            terminal.clear();
            showPrompt();
        }

        function showHelp() {
            if (socket && socket.readyState === WebSocket.OPEN) {
                sendCommand('help');
            } else {
                terminal.writeln('\\r\\n❌ Not connected. Please reconnect to use commands.\\r\\n');
            }
        }

        function showStatus() {
            if (socket && socket.readyState === WebSocket.OPEN) {
                sendCommand('status');
            } else {
                terminal.writeln('\\r\\n❌ Not connected. Please reconnect to use commands.\\r\\n');
            }
        }

        function reconnect() {
            if (socket) {
                socket.close();
            }
            sessionId = generateSessionId();
            startTime = Date.now();
            commandCount = 0;
            updateCommandCount();
            connect();
        }

        // Auto-resize terminal
        window.addEventListener('resize', () => {
            fitAddon.fit();
        });

        // Update uptime every second
        setInterval(updateUptime, 1000);

        // Initial connection
        connect();
    </script>
</body>
</html>"""

    # Ensure templates directory exists and write the template
    template_file = Path(__file__).parent / "templates" / "terminal.html"
    template_file.parent.mkdir(exist_ok=True)

    with open(template_file, "w") as f:
        f.write(html_content)


def main():
    """Run the web terminal server."""
    # Create the HTML template
    create_terminal_html()

    print("🌐 Starting Enhanced ReAct Web Terminal Server...")
    print("📱 Features:")
    print("   • Real-time terminal emulation in browser")
    print("   • WebSocket-based communication")
    print("   • Multi-agent ReAct engine integration")
    print("   • Session persistence and restoration")
    print("   • Progress tracking and status updates")
    print()
    print("🚀 Server will be available at: http://localhost:8080")
    print("🎯 Open your browser and navigate to the URL to use the terminal")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
