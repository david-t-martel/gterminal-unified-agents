"""Example module to test automation capabilities.
This module demonstrates various Python features for testing automation.
"""

from dataclasses import dataclass
import logging
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExampleConfig:
    """Configuration for example operations."""

    name: str
    value: int = 42
    enabled: bool = True
    options: dict[str, Any] | None = None


class ExampleProcessor:
    """Processes examples with various features."""

    def __init__(self, config: ExampleConfig) -> None:
        """Initialize the processor with configuration."""
        self.config = config
        self.processed_count = 0
        logger.info(f"Initialized ExampleProcessor with {config.name}")

    def process_items(self, items: list[str]) -> list[str]:
        """Process a list of items.

        Args:
            items: List of items to process

        Returns:
            Processed items with prefix

        """
        if not self.config.enabled:
            logger.warning("Processing disabled, returning empty list")
            return []

        processed = []
        for item in items:
            processed_item = f"{self.config.name}:{item}"
            processed.append(processed_item)
            self.processed_count += 1

        logger.info(f"Processed {len(items)} items")
        return processed

    def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_processed": self.processed_count,
            "config_name": self.config.name,
            "enabled": self.config.enabled,
        }


def calculate_something(x: int, y: int) -> int:
    """Calculate something important.

    Args:
        x: First number
        y: Second number

    Returns:
        The sum of x and y

    """
    # This could have a bug - what if someone passes None?
    return x + y


# Example of code that might need fixing
def problematic_function(data: dict[str, int]) -> int:
    # Missing type hints
    # No docstring
    # Potential bug: no validation
    return data["key"] * 2
