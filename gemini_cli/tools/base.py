"""Base tool interface for Gemini CLI tools."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for all CLI tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool description."""
        pass

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with given parameters.

        Args:
            params: Tool parameters

        Returns:
            Tool execution results
        """
        pass
