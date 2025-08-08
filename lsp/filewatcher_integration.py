#!/usr/bin/env python3
"""
Filewatcher Integration - Seamless integration with Rust filewatcher

This module provides seamless integration between the Python ruff LSP system
and the Rust filewatcher, enabling real-time file change notifications and
automatic diagnostic refreshes.

Features:
- WebSocket connection to Rust filewatcher
- Real-time file change notifications
- Automatic LSP diagnostic refresh on changes
- Bidirectional communication for status updates
- Performance monitoring and reconnection logic
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import time
from typing import Any

import aiofiles
from pydantic import BaseModel
from pydantic import Field
from rich.console import Console
import websockets
from websockets.exceptions import ConnectionClosed
from websockets.exceptions import WebSocketException


class FileEventType(Enum):
    """File event types from filewatcher"""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class FileEvent:
    """File event from filewatcher"""

    event_type: FileEventType
    file_path: str
    timestamp: datetime = field(default_factory=datetime.now)
    old_path: str | None = None  # For rename events
    metadata: dict[str, Any] | None = None


class FilewatcherConfig(BaseModel):
    """Configuration for filewatcher integration"""

    # Connection settings
    host: str = "localhost"
    port: int = 8768  # Default WebSocket port for filewatcher
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    connection_timeout: float = 10.0

    # File filtering
    watch_extensions: list[str] = Field(default_factory=lambda: [".py", ".pyi", ".pyx"])
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            ".git",
            ".venv",
            "node_modules",
            ".mypy_cache",
            ".pytest_cache",
        ]
    )

    # Event processing
    debounce_delay: float = 0.5  # Seconds to wait before processing events
    batch_events: bool = True
    max_batch_size: int = 50

    # LSP integration
    auto_refresh_diagnostics: bool = True
    refresh_delay: float = 1.0  # Delay before refreshing diagnostics

    # Performance
    max_events_per_second: int = 100
    enable_event_logging: bool = True


class FilewatcherIntegration:
    """
    Integration layer between ruff LSP and Rust filewatcher

    Provides real-time file change notifications and automatic diagnostic
    refresh capabilities through WebSocket communication.
    """

    def __init__(self, config: FilewatcherConfig, lsp_client=None):
        self.config = config
        self.lsp_client = lsp_client
        self.console = Console()
        self.logger = logging.getLogger("filewatcher-integration")

        # Connection state
        self.websocket: Any | None = None
        self.connected = False
        self.reconnect_count = 0
        self.last_connect_attempt = 0

        # Event processing
        self.event_queue: asyncio.Queue[FileEvent] = asyncio.Queue()
        self.pending_events: dict[str, FileEvent] = {}  # For debouncing
        self.event_callbacks: list[Callable[[FileEvent], None]] = []

        # Performance tracking
        self.stats = {
            "events_received": 0,
            "events_processed": 0,
            "reconnections": 0,
            "connection_time": 0.0,
            "last_event_time": 0.0,
        }

        # Background tasks
        self.connection_task: asyncio.Task | None = None
        self.event_processor_task: asyncio.Task | None = None
        self.debounce_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the filewatcher integration"""
        self.logger.info("🔗 Starting filewatcher integration...")

        # Start background tasks
        self.connection_task = asyncio.create_task(self._connection_manager())
        self.event_processor_task = asyncio.create_task(self._event_processor())

        if self.config.batch_events:
            self.debounce_task = asyncio.create_task(self._debounce_processor())

        self.logger.info("✅ Filewatcher integration started")

    async def stop(self) -> None:
        """Stop the filewatcher integration"""
        self.logger.info("Stopping filewatcher integration...")

        # Cancel tasks
        if self.connection_task:
            self.connection_task.cancel()
        if self.event_processor_task:
            self.event_processor_task.cancel()
        if self.debounce_task:
            self.debounce_task.cancel()

        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()

        self.connected = False
        self.logger.info("✅ Filewatcher integration stopped")

    async def _connection_manager(self) -> None:
        """Manage WebSocket connection with automatic reconnection"""
        while True:
            try:
                if not self.connected:
                    await self._connect()

                if self.connected:
                    await self._listen_for_events()

            except Exception as e:
                self.logger.exception(f"Connection error: {e}")
                self.connected = False
                self.reconnect_count += 1
                self.stats["reconnections"] += 1

                if self.reconnect_count >= self.config.max_reconnect_attempts:
                    self.logger.exception("Max reconnection attempts reached")
                    break

                self.logger.info(f"Reconnecting in {self.config.reconnect_delay} seconds...")
                await asyncio.sleep(self.config.reconnect_delay)

    async def _connect(self) -> None:
        """Establish WebSocket connection to filewatcher"""
        self.last_connect_attempt = time.time()
        uri = f"ws://{self.config.host}:{self.config.port}"

        self.logger.info(f"Connecting to filewatcher at {uri}")

        try:
            self.websocket = await asyncio.wait_for(
                websockets.connect(uri), timeout=self.config.connection_timeout
            )

            # Send initial handshake
            handshake = {
                "type": "handshake",
                "client": "ruff-lsp-integration",
                "version": "1.0.0",
                "capabilities": {
                    "file_events": True,
                    "diagnostics": True,
                    "auto_refresh": self.config.auto_refresh_diagnostics,
                },
            }

            await self.websocket.send(json.dumps(handshake))
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)

            response_data = json.loads(response)
            if response_data.get("type") == "handshake_ack":
                self.connected = True
                self.reconnect_count = 0
                self.stats["connection_time"] = time.time() - self.last_connect_attempt

                self.logger.info("✅ Connected to filewatcher successfully")
            else:
                raise RuntimeError(f"Invalid handshake response: {response_data}")

        except (TimeoutError, ConnectionClosed, WebSocketException) as e:
            raise RuntimeError(f"Failed to connect to filewatcher: {e}")

    async def _listen_for_events(self) -> None:
        """Listen for file events from filewatcher"""
        try:
            async for message in self.websocket:
                try:
                    event_data = json.loads(message)
                    await self._handle_filewatcher_message(event_data)

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON from filewatcher: {e}")
                except Exception as e:
                    self.logger.exception(f"Error processing filewatcher message: {e}")

        except ConnectionClosed:
            self.logger.warning("Filewatcher connection closed")
            self.connected = False
        except Exception as e:
            self.logger.exception(f"Error listening to filewatcher: {e}")
            self.connected = False

    async def _handle_filewatcher_message(self, data: dict[str, Any]) -> None:
        """Handle incoming message from filewatcher"""
        message_type = data.get("type", "")

        if message_type == "file_event":
            await self._handle_file_event(data)
        elif message_type == "status":
            await self._handle_status_message(data)
        elif message_type == "error":
            await self._handle_error_message(data)
        else:
            self.logger.debug(f"Unknown message type: {message_type}")

    async def _handle_file_event(self, data: dict[str, Any]) -> None:
        """Handle file event from filewatcher"""
        try:
            file_path = data.get("path", "")
            event_type_str = data.get("event", "")

            # Filter by extension
            if not any(file_path.endswith(ext) for ext in self.config.watch_extensions):
                return

            # Filter by ignore patterns
            if any(pattern in file_path for pattern in self.config.ignore_patterns):
                return

            # Create file event
            event_type = FileEventType(event_type_str.lower())
            event = FileEvent(
                event_type=event_type,
                file_path=file_path,
                old_path=data.get("old_path"),
                metadata=data.get("metadata", {}),
            )

            self.stats["events_received"] += 1
            self.stats["last_event_time"] = time.time()

            if self.config.enable_event_logging:
                self.logger.info(f"📝 File {event_type.value}: {Path(file_path).name}")

            # Queue event for processing
            await self.event_queue.put(event)

            # Add to pending events for debouncing if enabled
            if self.config.batch_events:
                self.pending_events[file_path] = event

        except (ValueError, KeyError) as e:
            self.logger.warning(f"Invalid file event data: {e}")

    async def _handle_status_message(self, data: dict[str, Any]) -> None:
        """Handle status message from filewatcher"""
        status = data.get("status", "unknown")
        self.logger.info(f"Filewatcher status: {status}")

    async def _handle_error_message(self, data: dict[str, Any]) -> None:
        """Handle error message from filewatcher"""
        error = data.get("error", "Unknown error")
        self.logger.error(f"Filewatcher error: {error}")

    async def _event_processor(self) -> None:
        """Process file events from the queue"""
        while True:
            try:
                event = await self.event_queue.get()
                await self._process_file_event(event)
                self.stats["events_processed"] += 1

            except Exception as e:
                self.logger.exception(f"Error processing file event: {e}")

    async def _process_file_event(self, event: FileEvent) -> None:
        """Process a single file event"""
        # Notify callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.exception(f"Error in event callback: {e}")

        # Auto-refresh diagnostics if LSP client is available
        if (
            self.lsp_client
            and self.config.auto_refresh_diagnostics
            and event.event_type in [FileEventType.MODIFIED, FileEventType.CREATED]
        ):
            await asyncio.sleep(self.config.refresh_delay)
            await self._refresh_diagnostics(event.file_path)

    async def _refresh_diagnostics(self, file_path: str) -> None:
        """Refresh diagnostics for a file"""
        if not self.lsp_client:
            return

        try:
            file_path_obj = Path(file_path)

            # Check if file is already open in LSP
            uri = file_path_obj.as_uri()
            if uri in self.lsp_client.open_documents:
                # File is already open, update content
                if file_path_obj.exists():
                    async with aiofiles.open(file_path_obj, encoding="utf-8") as f:
                        content = await f.read()
                    await self.lsp_client.update_document(file_path_obj, content)
            # Open the file in LSP
            elif file_path_obj.exists():
                await self.lsp_client.open_document(file_path_obj)

            self.logger.debug(f"Refreshed diagnostics for: {file_path_obj.name}")

        except Exception as e:
            self.logger.exception(f"Error refreshing diagnostics for {file_path}: {e}")

    async def _debounce_processor(self) -> None:
        """Process debounced events in batches"""
        while True:
            await asyncio.sleep(self.config.debounce_delay)

            if self.pending_events:
                # Process batch of events
                events_to_process = list(self.pending_events.values())
                self.pending_events.clear()

                self.logger.debug(f"Processing batch of {len(events_to_process)} events")

                for event in events_to_process:
                    await self._process_file_event(event)

    def add_event_callback(self, callback: Callable[[FileEvent], None]) -> None:
        """Add callback for file events"""
        self.event_callbacks.append(callback)

    def remove_event_callback(self, callback: Callable[[FileEvent], None]) -> None:
        """Remove event callback"""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)

    async def send_status_update(self, status: dict[str, Any]) -> None:
        """Send status update to filewatcher"""
        if self.connected and self.websocket:
            message = {
                "type": "status_update",
                "timestamp": datetime.now().isoformat(),
                "status": status,
            }

            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                self.logger.exception(f"Error sending status update: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get integration statistics"""
        return {
            **self.stats,
            "connected": self.connected,
            "reconnect_count": self.reconnect_count,
            "pending_events": len(self.pending_events),
            "queue_size": self.event_queue.qsize(),
        }

    def is_connected(self) -> bool:
        """Check if connected to filewatcher"""
        return self.connected
