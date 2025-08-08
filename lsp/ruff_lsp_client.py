#!/usr/bin/env python3
"""
Ruff LSP Client - Comprehensive LSP integration for ruff server

This module provides a full-featured LSP client that interfaces with the ruff LSP server,
providing real-time diagnostics, code actions, and AI-powered fix suggestions.

Features:
- Persistent LSP connection with ruff server
- Real-time diagnostic streaming
- Advanced code actions (auto-import, refactoring, optimization)
- AI-powered fix suggestions using Claude analysis
- Performance monitoring and health checks
- Integration with Rust filewatcher
"""

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic import Field
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from collections.abc import Callable
    import subprocess

"""
Ruff LSP Client - Comprehensive LSP integration for ruff server

This module provides a full-featured LSP client that interfaces with the ruff LSP server,
providing real-time diagnostics, code actions, and AI-powered fix suggestions.

Features:
- Persistent LSP connection with ruff server
- Real-time diagnostic streaming
- Advanced code actions (auto-import, refactoring, optimization)
- AI-powered fix suggestions using Claude analysis
- Performance monitoring and health checks
- Integration with Rust filewatcher
"""


# LSP Protocol Types
@dataclass
class Position:
    """LSP Position type"""

    line: int
    character: int


@dataclass
class Range:
    """LSP Range type"""

    start: Position
    end: Position


@dataclass
class Diagnostic:
    """LSP Diagnostic type with ruff-specific extensions"""

    range: Range
    message: str
    severity: int  # 1=Error, 2=Warning, 3=Information, 4=Hint
    code: str | None = None
    source: str = "ruff"
    data: dict[str, Any] | None = None
    # Ruff-specific fields
    fix: dict[str, Any] | None = None
    noqa: str | None = None
    url: str | None = None


@dataclass
class CodeAction:
    """LSP CodeAction with ruff enhancements"""

    title: str
    kind: str
    diagnostics: list[Diagnostic]
    edit: dict[str, Any] | None = None
    command: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    # AI enhancement data
    confidence: float = 1.0
    explanation: str | None = None


@dataclass
class LSPMessage:
    """LSP JSON-RPC message"""

    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str | None = None
    params: dict[str, Any] | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None


class RuffLSPConfig(BaseModel):
    """Configuration for Ruff LSP client"""

    # Server configuration
    server_cmd: list[str] = Field(default_factory=lambda: ["ruff", "server", "--preview"])
    workspace_root: Path = Field(default_factory=lambda: Path.cwd())

    # LSP client settings
    client_name: str = "gterminal-ruff-client"
    client_version: str = "1.0.0"
    trace: str = "messages"  # off, messages, verbose

    # Ruff-specific settings
    ruff_config_path: Path | None = None
    lint_rules: list[str] = Field(default_factory=list)
    ignore_rules: list[str] = Field(default_factory=list)
    format_options: dict[str, Any] = Field(default_factory=dict)

    # Performance settings
    diagnostic_debounce_ms: int = 250
    max_diagnostics_per_file: int = 1000
    enable_auto_import: bool = True
    enable_code_actions: bool = True

    # AI integration
    claude_model: str = "haiku"
    enable_ai_suggestions: bool = True
    ai_confidence_threshold: float = 0.7

    # Monitoring
    enable_performance_monitoring: bool = True
    log_level: str = "INFO"
    metrics_file: Path | None = None


class RuffLSPClient:
    """
    High-performance Ruff LSP client with AI-powered enhancements

    This client provides:
    - Persistent connection to ruff LSP server
    - Real-time diagnostic streaming
    - Advanced code actions and fixes
    - AI-powered suggestion system
    - Performance monitoring
    """

    def __init__(self, config: RuffLSPConfig):
        self.config = config
        self.console = Console()
        self.logger = self._setup_logging()

        # LSP connection state
        self.server_process: subprocess.Popen | None = None
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.request_id = 0
        self.pending_requests: dict[int, asyncio.Future] = {}

        # Document state tracking
        self.open_documents: set[str] = set()
        self.document_versions: dict[str, int] = {}
        self.diagnostics_cache: dict[str, list[Diagnostic]] = {}

        # Performance monitoring
        self.performance_metrics = {
            "requests_sent": 0,
            "responses_received": 0,
            "diagnostics_received": 0,
            "code_actions_generated": 0,
            "ai_suggestions_made": 0,
            "avg_response_time_ms": 0.0,
            "uptime_start": time.time(),
        }  # AI integration
        self.claude_client: Any | None = None
        self.ai_suggestion_cache: dict[str, list[CodeAction]] = {}

        # Event callbacks
        self.diagnostic_callbacks: list[Callable[[str, list[Diagnostic]], None]] = []
        self.code_action_callbacks: list[Callable[[str, list[CodeAction]], None]] = []

    def _setup_logging(self) -> logging.Logger:
        """Setup rich logging with appropriate level"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)],
        )
        return logging.getLogger("ruff-lsp")

    async def start(self) -> None:
        """Start the ruff LSP server and initialize connection"""
        self.logger.info(f"Starting Ruff LSP server: {' '.join(self.config.server_cmd)}")

        try:
            # Start ruff server process
            self.server_process = await asyncio.create_subprocess_exec(
                *self.config.server_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.workspace_root,
            )

            if not self.server_process.stdin or not self.server_process.stdout:
                raise RuntimeError("Failed to create subprocess pipes")

            self.reader = self.server_process.stdout
            self.writer = self.server_process.stdin  # Initialize LSP connection
            await self._initialize_lsp()

            # Start background tasks
            asyncio.create_task(self._message_reader())
            asyncio.create_task(self._performance_monitor())

            self.logger.info("✅ Ruff LSP client started successfully")

        except Exception as e:
            self.logger.exception(f"❌ Failed to start Ruff LSP server: {e}")
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown the LSP client"""
        self.logger.info("Shutting down Ruff LSP client...")

        try:
            # Send shutdown request
            if self.writer and not self.writer.is_closing():
                await self._send_request("shutdown", {})
                await self._send_notification("exit", {})

            # Close streams
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()

            # Terminate server process
            if self.server_process:
                self.server_process.terminate()
                await self.server_process.wait()

        except Exception as e:
            self.logger.exception(f"Error during shutdown: {e}")

        self.logger.info("✅ Ruff LSP client shutdown complete")


# CLI interface for testing
async def main():
    """Main CLI interface for testing the Ruff LSP client"""
    import argparse

    parser = argparse.ArgumentParser(description="Ruff LSP Client")
    parser.add_argument(
        "--workspace", type=Path, default=Path.cwd(), help="Workspace root directory"
    )
    parser.add_argument("--config", type=Path, help="Ruff configuration file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument("--metrics-file", type=Path, help="File to write performance metrics")
    parser.add_argument("--test-file", type=Path, help="Python file to test with")

    args = parser.parse_args()

    # Create configuration
    config = RuffLSPConfig(
        workspace_root=args.workspace,
        ruff_config_path=args.config,
        log_level=args.log_level,
        metrics_file=args.metrics_file,
    )

    # Create and start client
    client = RuffLSPClient(config)

    try:
        await client.start()

        # Test with file if provided
        if args.test_file and args.test_file.exists():
            await client.open_document(args.test_file)

            # Wait for diagnostics
            await asyncio.sleep(2)

            # Get diagnostics
            diagnostics = client.get_diagnostics(args.test_file)
            print(f"Found {len(diagnostics)} diagnostics")

        # Run health check
        health = await client.health_check()
        print(f"Health check: {health['status']}")

        # Keep running for testing
        print("LSP client running. Press Ctrl+C to exit.")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await client.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
