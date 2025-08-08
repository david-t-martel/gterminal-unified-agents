"""Common base classes to reduce duplication."""

from abc import ABC
from abc import abstractmethod
import logging
from typing import Any


class BaseService(ABC):
    """Base class for all services."""

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service."""


class ConfigurableMixin:
    """Mixin for configurable components."""

    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration."""
        if hasattr(self, "config"):
            self.config.update(config)
        else:
            self.config = config

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if hasattr(self, "config"):
            return self.config.get(key, default)
        return default
