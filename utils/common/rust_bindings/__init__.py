# /home/david/agents/my-fullstack-agent/app/utils/common/rust_bindings/__init__.py
import importlib
import logging

logger = logging.getLogger(__name__)

_rust_extension = None


def try_import_rust_extension() -> object | None:
    """Attempts to import the Rust extension module.

    Returns:
        The Rust extension module if import is successful, otherwise None.

    """
    try:
        global _rust_extension
        if _rust_extension is None:
            _rust_extension = importlib.import_module("fullstack_agent_rust")
        return _rust_extension
    except ImportError as e:
        logger.warning(
            f"Failed to import Rust extension: {e}. Falling back to Python implementation."
        )
        return None
    except Exception as e:
        logger.exception(f"Unexpected error while importing Rust extension: {e}")
        return None


# Attempt to import the Rust extension on module load
rust_extension = try_import_rust_extension()


# Create utility module for accessing rust_utils safely
def get_rust_utils() -> object | None:
    """Get rust utilities with fallback to basic implementation."""
    if rust_extension:
        try:
            return rust_extension
        except AttributeError:
            logger.warning(
                "Rust extension does not have expected utilities. Using python fallback."
            )
            return None
    return None


# Make rust_utils available for import
rust_utils = get_rust_utils()
