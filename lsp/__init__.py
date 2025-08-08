"""
Ruff LSP Integration Module

This module provides comprehensive Language Server Protocol (LSP) integration
for ruff, transforming it from a simple linter into a full AI-powered Python
development assistant.

Features:
- Persistent LSP connection with ruff server
- Real-time diagnostic streaming
- Advanced code actions and refactoring
- AI-powered fix suggestions using Claude
- Performance monitoring and health checks
- Integration with Rust filewatcher
"""

from .ai_suggestion_engine import AISuggestionEngine
from .ai_suggestion_engine import FixConfidence
from .ai_suggestion_engine import SuggestionRequest
from .ai_suggestion_engine import SuggestionResponse
from .diagnostic_streamer import DiagnosticStreamConfig
from .diagnostic_streamer import DiagnosticStreamer
from .diagnostic_streamer import StreamEvent
from .diagnostic_streamer import StreamEventType
from .performance_monitor import HealthStatus
from .performance_monitor import LSPPerformanceMonitor
from .performance_monitor import PerformanceMetrics
from .ruff_lsp_client import CodeAction
from .ruff_lsp_client import Diagnostic
from .ruff_lsp_client import LSPMessage
from .ruff_lsp_client import Position
from .ruff_lsp_client import Range
from .ruff_lsp_client import RuffLSPClient
from .ruff_lsp_client import RuffLSPConfig

__all__ = [
    # AI suggestions
    "AISuggestionEngine",
    "CodeAction",
    # LSP types
    "Diagnostic",
    "DiagnosticStreamConfig",
    # Diagnostic streaming
    "DiagnosticStreamer",
    "FixConfidence",
    "HealthStatus",
    "LSPMessage",
    # Performance monitoring
    "LSPPerformanceMonitor",
    "PerformanceMetrics",
    "Position",
    "Range",
    # Core LSP client
    "RuffLSPClient",
    "RuffLSPConfig",
    "StreamEvent",
    "StreamEventType",
    "SuggestionRequest",
    "SuggestionResponse",
]
