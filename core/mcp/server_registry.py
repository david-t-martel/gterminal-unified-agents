"""
MCP Server Registry and Lifecycle Management

Provides centralized registration, discovery, and lifecycle management
for MCP servers including startup, health monitoring, and graceful shutdown.
"""

import asyncio
import contextlib
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import logging
from pathlib import Path
import subprocess
import time
from typing import TYPE_CHECKING, Any

import psutil

from .config_manager import MCPConfigManager
from .config_manager import ServerConfig
from .consolidated_auth import ConsolidatedAuth
from .security_manager import SecurityLevel
from .security_manager import SecurityManager

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    """MCP server status states"""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ServerProcess:
    """Information about a running MCP server process"""

    config: ServerConfig
    process: subprocess.Popen | None = None
    pid: int | None = None
    status: ServerStatus = ServerStatus.STOPPED
    start_time: float | None = None
    restart_count: int = 0
    last_health_check: float | None = None
    health_status: bool = False
    error_log: list[str] = field(default_factory=list)


class ServerRegistry:
    """
    Centralized registry and lifecycle manager for MCP servers.

    Handles server registration, startup, monitoring, health checks,
    and graceful shutdown with integration to security and auth systems.
    """

    def __init__(
        self,
        config_manager: MCPConfigManager,
        security_manager: SecurityManager,
        auth_manager: ConsolidatedAuth,
    ):
        """
        Initialize server registry.

        Args:
            config_manager: Configuration management instance
            security_manager: Security management instance
            auth_manager: Authentication management instance
        """
        self.config_manager = config_manager
        self.security_manager = security_manager
        self.auth_manager = auth_manager

        self.servers: dict[str, ServerProcess] = {}
        self.health_check_interval = 30  # seconds
        self.max_restart_attempts = 3
        self._monitoring_task: asyncio.Task | None = None
        self._shutdown_requested = False

        # Callbacks for server events
        self.on_server_started: list[Callable[[str], Awaitable[None]]] = []
        self.on_server_stopped: list[Callable[[str], Awaitable[None]]] = []
        self.on_server_failed: list[Callable[[str, str], Awaitable[None]]] = []

    def register_server(self, server_config: ServerConfig) -> None:
        """
        Register a new MCP server.

        Args:
            server_config: Server configuration
        """
        if server_config.name in self.servers:
            logger.warning(f"Server {server_config.name} already registered, updating config")

        # Set security policy based on config
        if hasattr(server_config, "security_profile") and server_config.security_profile:
            try:
                security_level = SecurityLevel(server_config.security_profile)
                self.security_manager.set_policy(server_config.name, security_level)
            except ValueError:
                logger.warning(
                    f"Invalid security profile '{server_config.security_profile}', using standard"
                )
                self.security_manager.set_policy(server_config.name, SecurityLevel.STANDARD)
        else:
            self.security_manager.set_policy(server_config.name, SecurityLevel.STANDARD)

        self.servers[server_config.name] = ServerProcess(config=server_config)
        logger.info(f"Registered MCP server: {server_config.name}")

    def unregister_server(self, server_name: str) -> bool:
        """
        Unregister an MCP server.

        Args:
            server_name: Name of server to unregister

        Returns:
            True if unregistered, False if not found
        """
        if server_name not in self.servers:
            return False

        # Stop server if running
        if self.servers[server_name].status == ServerStatus.RUNNING:
            asyncio.create_task(self.stop_server(server_name))

        del self.servers[server_name]
        logger.info(f"Unregistered MCP server: {server_name}")
        return True

    async def start_server(self, server_name: str) -> bool:
        """
        Start an MCP server.

        Args:
            server_name: Name of server to start

        Returns:
            True if started successfully, False otherwise
        """
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not registered")
            return False

        server = self.servers[server_name]

        if server.status == ServerStatus.RUNNING:
            logger.info(f"Server {server_name} is already running")
            return True

        server.status = ServerStatus.STARTING
        logger.info(f"Starting MCP server: {server_name}")

        try:
            # Prepare environment variables
            env = os.environ.copy()

            # Add authentication environment
            auth_env = self.auth_manager.get_environment_vars()
            env.update(auth_env)

            # Add server-specific environment
            env.update(server.config.env)

            # Create secure working directory
            working_dir = Path(server.config.working_dir) if server.config.working_dir else None

            # Check if we need to use a secure wrapper
            policy = self.security_manager.get_policy(server_name)
            command_args = [server.config.command, *server.config.args]

            if policy.require_wrapper:
                # Create secure wrapper if needed
                wrapper_dir = Path.home() / ".cache" / "mcp-wrappers"
                wrapper_script = self.security_manager.create_secure_wrapper(
                    server_name, command_args, wrapper_dir
                )
                command_args = ["bash", str(wrapper_script)]

            # Start the process
            server.process = subprocess.Popen(
                command_args,
                env=env,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            server.pid = server.process.pid
            server.start_time = time.time()
            server.status = ServerStatus.RUNNING
            server.restart_count += 1

            logger.info(f"Started MCP server {server_name} with PID {server.pid}")

            # Trigger started callbacks
            for callback in self.on_server_started:
                try:
                    await callback(server_name)
                except Exception as e:
                    logger.exception(f"Error in server started callback: {e}")

            return True

        except Exception as e:
            server.status = ServerStatus.FAILED
            server.error_log.append(f"Failed to start: {e}")
            logger.exception(f"Failed to start server {server_name}: {e}")

            # Trigger failed callbacks
            for callback in self.on_server_failed:
                try:
                    await callback(server_name, str(e))
                except Exception as callback_error:
                    logger.exception(f"Error in server failed callback: {callback_error}")

            return False

    async def stop_server(self, server_name: str, timeout: int = 10) -> bool:
        """
        Stop an MCP server gracefully.

        Args:
            server_name: Name of server to stop
            timeout: Timeout in seconds for graceful shutdown

        Returns:
            True if stopped successfully, False otherwise
        """
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not registered")
            return False

        server = self.servers[server_name]

        if server.status != ServerStatus.RUNNING or not server.process:
            logger.info(f"Server {server_name} is not running")
            server.status = ServerStatus.STOPPED
            return True

        server.status = ServerStatus.STOPPING
        logger.info(f"Stopping MCP server: {server_name}")

        try:
            # Try graceful shutdown first
            server.process.terminate()

            try:
                server.process.wait(timeout=timeout)
                logger.info(f"Server {server_name} stopped gracefully")

            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning(f"Server {server_name} did not stop gracefully, forcing shutdown")
                server.process.kill()
                server.process.wait()

            server.status = ServerStatus.STOPPED
            server.process = None
            server.pid = None

            # Trigger stopped callbacks
            for callback in self.on_server_stopped:
                try:
                    await callback(server_name)
                except Exception as e:
                    logger.exception(f"Error in server stopped callback: {e}")

            return True

        except Exception as e:
            logger.exception(f"Error stopping server {server_name}: {e}")
            server.error_log.append(f"Error stopping: {e}")
            return False

    async def restart_server(self, server_name: str) -> bool:
        """
        Restart an MCP server.

        Args:
            server_name: Name of server to restart

        Returns:
            True if restarted successfully, False otherwise
        """
        logger.info(f"Restarting MCP server: {server_name}")

        await self.stop_server(server_name)
        await asyncio.sleep(1)  # Brief pause
        return await self.start_server(server_name)

    async def health_check_server(self, server_name: str) -> bool:
        """
        Perform health check on an MCP server.

        Args:
            server_name: Name of server to check

        Returns:
            True if healthy, False otherwise
        """
        if server_name not in self.servers:
            return False

        server = self.servers[server_name]

        if server.status != ServerStatus.RUNNING or not server.process or not server.pid:
            server.health_status = False
            return False

        try:
            # Check if process is still alive
            if server.process.poll() is not None:
                # Process has terminated
                server.status = ServerStatus.FAILED
                server.health_status = False
                logger.warning(f"Server {server_name} process terminated unexpectedly")
                return False

            # Check process using psutil for more detailed health info
            try:
                proc = psutil.Process(server.pid)

                # Check if process is running and responsive
                if proc.status() == psutil.STATUS_ZOMBIE:
                    server.health_status = False
                    return False

                # Additional health checks can be added here
                # For example: memory usage, CPU usage, file handles

            except psutil.NoSuchProcess:
                server.status = ServerStatus.FAILED
                server.health_status = False
                return False

            server.health_status = True
            server.last_health_check = time.time()
            return True

        except Exception as e:
            logger.exception(f"Error during health check for {server_name}: {e}")
            server.health_status = False
            return False

    async def start_all_servers(self) -> dict[str, bool]:
        """
        Start all registered servers.

        Returns:
            Dictionary mapping server names to start success status
        """
        results = {}

        for server_name in self.servers:
            results[server_name] = await self.start_server(server_name)

        return results

    async def stop_all_servers(self) -> dict[str, bool]:
        """
        Stop all running servers.

        Returns:
            Dictionary mapping server names to stop success status
        """
        results = {}

        for server_name, server in self.servers.items():
            if server.status == ServerStatus.RUNNING:
                results[server_name] = await self.stop_server(server_name)
            else:
                results[server_name] = True

        return results

    async def start_monitoring(self) -> None:
        """Start background health monitoring for all servers"""
        if self._monitoring_task:
            logger.warning("Monitoring already started")
            return

        self._monitoring_task = asyncio.create_task(self._monitor_servers())
        logger.info("Started server monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task
            self._monitoring_task = None

        logger.info("Stopped server monitoring")

    async def _monitor_servers(self) -> None:
        """Background task for monitoring server health"""
        while not self._shutdown_requested:
            try:
                for server_name in list(self.servers.keys()):
                    if self._shutdown_requested:
                        break

                    server = self.servers[server_name]

                    if server.status == ServerStatus.RUNNING:
                        is_healthy = await self.health_check_server(server_name)

                        if not is_healthy and server.config.auto_restart:
                            if server.restart_count < self.max_restart_attempts:
                                logger.info(f"Auto-restarting unhealthy server: {server_name}")
                                await self.restart_server(server_name)
                            else:
                                logger.error(
                                    f"Server {server_name} failed too many times, marking as failed"
                                )
                                server.status = ServerStatus.FAILED

                                # Trigger failed callbacks
                                for callback in self.on_server_failed:
                                    try:
                                        await callback(server_name, "Too many restart attempts")
                                    except Exception as e:
                                        logger.exception(f"Error in server failed callback: {e}")

                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in server monitoring: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    def get_server_status(self, server_name: str) -> dict[str, Any] | None:
        """
        Get detailed status information for a server.

        Args:
            server_name: Name of server

        Returns:
            Dictionary with server status information
        """
        if server_name not in self.servers:
            return None

        server = self.servers[server_name]

        status_info = {
            "name": server_name,
            "status": server.status.value,
            "pid": server.pid,
            "start_time": server.start_time,
            "restart_count": server.restart_count,
            "last_health_check": server.last_health_check,
            "health_status": server.health_status,
            "auto_restart": server.config.auto_restart,
            "security_profile": getattr(server.config, "security_profile", "standard"),
            "command": server.config.command,
            "args": server.config.args,
            "working_dir": server.config.working_dir,
            "recent_errors": server.error_log[-5:],  # Last 5 errors
        }

        # Add uptime if running
        if server.start_time and server.status == ServerStatus.RUNNING:
            status_info["uptime"] = time.time() - server.start_time

        return status_info

    def get_all_server_status(self) -> dict[str, Any]:
        """
        Get status information for all registered servers.

        Returns:
            Dictionary with all server status information
        """
        return {
            "servers": {name: self.get_server_status(name) for name in self.servers},
            "total_servers": len(self.servers),
            "running_servers": len(
                [s for s in self.servers.values() if s.status == ServerStatus.RUNNING]
            ),
            "failed_servers": len(
                [s for s in self.servers.values() if s.status == ServerStatus.FAILED]
            ),
            "monitoring_active": self._monitoring_task is not None,
            "health_check_interval": self.health_check_interval,
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the registry and all servers"""
        self._shutdown_requested = True
        logger.info("Shutting down server registry...")

        # Stop monitoring
        await self.stop_monitoring()

        # Stop all servers
        await self.stop_all_servers()

        logger.info("Server registry shutdown complete")
