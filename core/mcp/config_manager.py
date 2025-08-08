"""
MCP Configuration Management System

Provides centralized configuration management for MCP servers,
including validation, templating, and security policy enforcement.
"""

from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import validate

logger = logging.getLogger(__name__)


class ServerType(Enum):
    """Types of MCP servers"""

    CODE_REVIEWER = "gemini-code-reviewer"
    WORKSPACE_ANALYZER = "gemini-workspace-analyzer"
    MASTER_ARCHITECT = "gemini-master-architect"
    COST_OPTIMIZER = "cloud-cost-optimizer"
    UNIFIED_GATEWAY = "unified-mcp-gateway"
    MONITORING_SERVER = "unified-monitoring-server"


@dataclass
class ServerConfig:
    """Configuration for a single MCP server"""

    name: str
    command: str
    args: list[str]
    env: dict[str, str]
    working_dir: str | None = None
    timeout: int = 30
    auto_restart: bool = True
    security_profile: str = "standard"


@dataclass
class SecurityConfig:
    """Security configuration for MCP servers"""

    authentication_mode: str = "service_account_only"
    api_keys_disabled: bool = True
    wrapper_scripts_dir: str | None = None
    allowed_commands: list[str] = None
    rate_limits: dict[str, int] = None


class MCPConfigManager:
    """
    Centralized configuration management for MCP servers.

    Handles configuration validation, templating, security policies,
    and integration with the authentication system.
    """

    # JSON Schema for MCP configuration validation
    MCP_CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "mcpServers": {
                "type": "object",
                "patternProperties": {
                    "^[a-zA-Z0-9-_]+$": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "args": {"type": "array", "items": {"type": "string"}},
                            "env": {
                                "type": "object",
                                "patternProperties": {"^[A-Z_][A-Z0-9_]*$": {"type": "string"}},
                            },
                            "working_dir": {"type": "string"},
                            "timeout": {"type": "integer", "minimum": 1},
                            "auto_restart": {"type": "boolean"},
                        },
                        "required": ["command", "args"],
                        "additionalProperties": False,
                    }
                },
            },
            "security": {
                "type": "object",
                "properties": {
                    "authentication_mode": {
                        "type": "string",
                        "enum": ["service_account_only", "api_key", "mixed"],
                    },
                    "api_keys_disabled": {"type": "boolean"},
                    "wrapper_scripts_dir": {"type": "string"},
                    "allowed_commands": {"type": "array", "items": {"type": "string"}},
                    "rate_limits": {
                        "type": "object",
                        "patternProperties": {"^[a-zA-Z0-9-_]+$": {"type": "integer"}},
                    },
                },
                "additionalProperties": False,
            },
        },
        "required": ["mcpServers"],
        "additionalProperties": False,
    }

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to MCP configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config_data: dict[str, Any] | None = None
        self._servers: dict[str, ServerConfig] = {}
        self._security: SecurityConfig | None = None

    def load_config(self, config_path: str | Path | None = None) -> dict[str, Any]:
        """
        Load MCP configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Loaded configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if config_path:
            self.config_path = Path(config_path)

        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path) as f:
                self._config_data = json.load(f)

            # Validate configuration
            self.validate_config(self._config_data)

            # Parse server configurations
            self._parse_servers()
            self._parse_security()

            logger.info(f"Loaded MCP configuration from {self.config_path}")
            return self._config_data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")

    def validate_config(self, config_data: dict[str, Any]) -> None:
        """
        Validate MCP configuration against schema.

        Args:
            config_data: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            validate(instance=config_data, schema=self.MCP_CONFIG_SCHEMA)
            logger.info("Configuration validation passed")

        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")
        except Exception as e:
            raise ValueError(f"Configuration validation error: {e}")

    def _parse_servers(self) -> None:
        """Parse server configurations from loaded data"""
        if not self._config_data:
            return

        servers_data = self._config_data.get("mcpServers", {})
        self._servers = {}

        for name, server_data in servers_data.items():
            self._servers[name] = ServerConfig(
                name=name,
                command=server_data["command"],
                args=server_data["args"],
                env=server_data.get("env", {}),
                working_dir=server_data.get("working_dir"),
                timeout=server_data.get("timeout", 30),
                auto_restart=server_data.get("auto_restart", True),
                security_profile=server_data.get("security_profile", "standard"),
            )

    def _parse_security(self) -> None:
        """Parse security configuration from loaded data"""
        if not self._config_data:
            return

        security_data = self._config_data.get("security", {})
        self._security = SecurityConfig(
            authentication_mode=security_data.get("authentication_mode", "service_account_only"),
            api_keys_disabled=security_data.get("api_keys_disabled", True),
            wrapper_scripts_dir=security_data.get("wrapper_scripts_dir"),
            allowed_commands=security_data.get("allowed_commands", []),
            rate_limits=security_data.get("rate_limits", {}),
        )

    def get_server_config(self, server_name: str) -> ServerConfig | None:
        """
        Get configuration for a specific server.

        Args:
            server_name: Name of the MCP server

        Returns:
            ServerConfig object or None if not found
        """
        return self._servers.get(server_name)

    def get_all_servers(self) -> dict[str, ServerConfig]:
        """
        Get all server configurations.

        Returns:
            Dictionary of server name to ServerConfig
        """
        return self._servers.copy()

    def get_security_config(self) -> SecurityConfig | None:
        """
        Get security configuration.

        Returns:
            SecurityConfig object or None if not loaded
        """
        return self._security

    def add_server(self, server_config: ServerConfig) -> None:
        """
        Add a new server configuration.

        Args:
            server_config: ServerConfig object to add
        """
        self._servers[server_config.name] = server_config
        logger.info(f"Added server configuration: {server_config.name}")

    def remove_server(self, server_name: str) -> bool:
        """
        Remove a server configuration.

        Args:
            server_name: Name of server to remove

        Returns:
            True if removed, False if not found
        """
        if server_name in self._servers:
            del self._servers[server_name]
            logger.info(f"Removed server configuration: {server_name}")
            return True
        return False

    def update_server_env(self, server_name: str, env_vars: dict[str, str]) -> bool:
        """
        Update environment variables for a server.

        Args:
            server_name: Name of the server
            env_vars: Environment variables to add/update

        Returns:
            True if updated, False if server not found
        """
        if server_name not in self._servers:
            return False

        self._servers[server_name].env.update(env_vars)
        logger.info(f"Updated environment variables for {server_name}")
        return True

    def generate_claude_config(self, output_path: str | Path | None = None) -> dict[str, Any]:
        """
        Generate Claude CLI compatible configuration.

        Args:
            output_path: Path to write the configuration file

        Returns:
            Claude CLI configuration dictionary
        """
        if not self._servers:
            raise ValueError("No server configurations loaded")

        claude_config = {"mcpServers": {}}

        for name, server in self._servers.items():
            claude_config["mcpServers"][name] = {
                "command": server.command,
                "args": server.args,
                "env": server.env,
            }

            if server.working_dir:
                claude_config["mcpServers"][name]["working_dir"] = server.working_dir

        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w") as f:
                json.dump(claude_config, f, indent=2)
            logger.info(f"Generated Claude CLI config at {output_path}")

        return claude_config

    def create_server_template(self, server_type: ServerType) -> ServerConfig:
        """
        Create a server configuration template.

        Args:
            server_type: Type of server to create template for

        Returns:
            ServerConfig template
        """
        templates = {
            ServerType.CODE_REVIEWER: ServerConfig(
                name=server_type.value,
                command="uv",
                args=["run", "python", "-m", "app.mcp_servers.gemini_code_reviewer"],
                env={"LOG_LEVEL": "INFO", "RUST_EXTENSIONS_ENABLED": "true"},
                security_profile="secure",
            ),
            ServerType.WORKSPACE_ANALYZER: ServerConfig(
                name=server_type.value,
                command="uv",
                args=["run", "python", "-m", "app.mcp_servers.gemini_workspace_analyzer"],
                env={"LOG_LEVEL": "INFO", "RUST_EXTENSIONS_ENABLED": "true"},
                security_profile="secure",
            ),
            ServerType.MASTER_ARCHITECT: ServerConfig(
                name=server_type.value,
                command="uv",
                args=["run", "python", "-m", "app.mcp_servers.gemini_master_architect"],
                env={"LOG_LEVEL": "INFO", "RUST_EXTENSIONS_ENABLED": "true"},
                security_profile="secure",
            ),
            ServerType.COST_OPTIMIZER: ServerConfig(
                name=server_type.value,
                command="uv",
                args=["run", "python", "-m", "app.mcp_servers.cloud_cost_optimizer"],
                env={"LOG_LEVEL": "INFO"},
                security_profile="standard",
            ),
            ServerType.UNIFIED_GATEWAY: ServerConfig(
                name=server_type.value,
                command="uv",
                args=["run", "python", "-m", "app.mcp_servers.unified_mcp_gateway"],
                env={"LOG_LEVEL": "INFO", "RUST_EXTENSIONS_ENABLED": "true"},
                security_profile="gateway",
            ),
            ServerType.MONITORING_SERVER: ServerConfig(
                name=server_type.value,
                command="uv",
                args=["run", "python", "-m", "app.mcp_servers.unified_monitoring_server"],
                env={"LOG_LEVEL": "INFO"},
                security_profile="monitoring",
            ),
        }

        return templates.get(
            server_type,
            ServerConfig(
                name=server_type.value, command="echo", args=["Server not implemented"], env={}
            ),
        )

    def save_config(self, output_path: str | Path | None = None) -> None:
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration file
        """
        save_path = Path(output_path) if output_path else self.config_path

        if not save_path:
            raise ValueError("No output path specified")

        config_data = {
            "mcpServers": {
                name: {
                    "command": server.command,
                    "args": server.args,
                    "env": server.env,
                    "timeout": server.timeout,
                    "auto_restart": server.auto_restart,
                }
                for name, server in self._servers.items()
            }
        }

        if self._security:
            config_data["security"] = asdict(self._security)

        with open(save_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved configuration to {save_path}")
