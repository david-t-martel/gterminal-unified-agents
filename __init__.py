"""GTerminal - AI-Powered Development Environment.

A comprehensive development environment designed for professional AI agent development
with VertexAI function calling capabilities, Rust-based performance tools, and
real-time development monitoring.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Python 3.12+ requirement check

# Package metadata
try:
    __version__ = version("gterminal")
except PackageNotFoundError:
    # Package is not installed, likely in development mode
    __version__ = "0.1.0-dev"

__author__ = "GTerminal Development Team"
__email__ = "dev@gterminal.ai"
__license__ = "MIT"

# Public API exports
__all__ = [
    "AutonomousReactEngine",
    "ReactEngine",
    "__author__",
    "__email__",
    "__license__",
    "__version__",
    "cache",
    # Core modules
    "core",
    "get_gemini_client",
    "lsp",
    "monitoring",
]

# Import key classes for easy access
try:
    from gterminal.core.autonomous_react_engine import AutonomousReactEngine
    from gterminal.core.react_engine import ReactEngine
    from gterminal.core.unified_gemini_client import get_gemini_client
except ImportError as e:
    # Handle import errors gracefully during development
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Some imports failed during development: {e}")


# Module-level configuration using Python 3.12+ features
class Config:
    """Global configuration for GTerminal."""

    # Use match/case for environment detection (Python 3.12+)
    @staticmethod
    def get_environment() -> str:
        """Detect current environment."""
        import os

        env = os.getenv("GTERMINAL_ENV", "development").lower()

        match env:
            case "production" | "prod":
                return "production"
            case "staging" | "stage":
                return "staging"
            case "testing" | "test":
                return "testing"
            case "development" | "dev" | _:
                return "development"

    # Python 3.12+ type union syntax
    @staticmethod
    def get_config_value(key: str, default: Any | None = None) -> Any | None:
        """Get configuration value with optional default."""
        import os

        # Use walrus operator for cleaner code
        if value := os.getenv(f"GTERMINAL_{key.upper()}"):
            return value
        return default


# Initialize configuration
config = Config()


# Lazy imports for better performance
def __getattr__(name: str) -> Any:
    """Lazy load heavy modules only when needed."""
    match name:
        case "cache":
            from . import cache

            return cache
        case "lsp":
            from . import lsp

            return lsp
        case "monitoring":
            from . import monitoring

            return monitoring
        case _:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Development mode detection
if config.get_environment() == "development":
    # Enable additional debugging features in development
    import logging

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.debug(f"GTerminal {__version__} initialized in development mode")
