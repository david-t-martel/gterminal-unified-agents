#!/usr/bin/env python3
"""
Diagnostic Streamer - Real-time diagnostic streaming and event management

This module provides real-time diagnostic streaming capabilities that integrate
with the Rust filewatcher and ruff LSP client to provide seamless developer feedback.

Features:
- Real-time diagnostic event streaming
- WebSocket integration with Rust filewatcher
- File change detection and auto-refresh
- Event filtering and batching
- Dashboard integration for visual feedback
"""

import asyncio
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
import logging
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic import Field
from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable


class StreamEventType(str, Enum):
    """Types of diagnostic stream events."""

    FILE_CHANGED = "file_changed"
    DIAGNOSTICS_RECEIVED = "diagnostics_received"
    DIAGNOSTICS_CLEARED = "diagnostics_cleared"
    LSP_CONNECTED = "lsp_connected"
    LSP_DISCONNECTED = "lsp_disconnected"
    ERROR_OCCURRED = "error_occurred"
    BATCH_COMPLETE = "batch_complete"


@dataclass
class StreamEvent:
    """Diagnostic stream event"""

    type: StreamEventType
    timestamp: datetime = field(default_factory=datetime.now)
    file_path: str | None = None
    data: dict[str, Any] | None = None
    message: str | None = None


class DiagnosticStreamConfig(BaseModel):
    """Configuration for diagnostic streaming"""

    # WebSocket connection
    filewatcher_host: str = "localhost"
    filewatcher_port: int = 8765
    websocket_timeout: float = 30.0

    # File filtering
    file_extensions: list[str] = Field(default_factory=lambda: [".py"])
    ignore_patterns: list[str] = Field(default_factory=lambda: ["__pycache__", ".git", ".venv"])

    # Event processing
    batch_size: int = 10
    batch_timeout: float = 2.0
    event_debounce_ms: int = 250

    # Dashboard integration
    enable_dashboard_updates: bool = True
    dashboard_update_interval: float = 5.0

    # Logging
    log_level: str = "INFO"
    log_events: bool = True

    # Performance
    max_events_in_memory: int = 1000
    cleanup_interval: float = 60.0


class DiagnosticStreamer:
    """
    Real-time diagnostic streaming system with filewatcher integration

    This streamer connects to the Rust filewatcher via WebSocket and provides
    real-time diagnostic updates, event filtering, and dashboard integration.
    """

    def __init__(self, config: DiagnosticStreamConfig, lsp_client=None):
        self.config = config
        self.lsp_client = lsp_client
        self.console = Console()
        self.logger = self._setup_logging()

        # Event management
        self.event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self.event_history: list[StreamEvent] = []
        self.event_callbacks: list[Callable[[StreamEvent], None]] = []

        # WebSocket connection
        self.websocket: Any | None = None
        self.connected = False

        # File tracking
        self.tracked_files: set[str] = set()
        self.file_diagnostics: dict[str, list[Any]] = {}
        self.last_file_changes: dict[str, float] = {}

        # Performance tracking
        self.stats = {
            "events_processed": 0,
            "files_tracked": 0,
            "diagnostics_received": 0,
            "websocket_reconnects": 0,
            "start_time": time.time(),
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the streamer"""
        logger = logging.getLogger("diagnostic-streamer")
        logger.setLevel(getattr(logging, self.config.log_level))
        return logger

    async def start(self) -> None:
        """Start the diagnostic streaming system"""
        self.logger.info("🚀 Starting diagnostic streamer...")

        # Start background tasks
        tasks = [
            asyncio.create_task(self._connect_websocket()),
            asyncio.create_task(self._process_events()),
            asyncio.create_task(self._cleanup_task()),
        ]

        if self.config.enable_dashboard_updates:
            tasks.append(asyncio.create_task(self._dashboard_updater()))

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.exception(f"Error in streaming tasks: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shutdown the streamer"""
        self.logger.info("Shutting down diagnostic streamer...")

        if self.websocket:
            await self.websocket.close()

        self.connected = False
        self.logger.info("✅ Diagnostic streamer shutdown complete")
