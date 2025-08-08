#!/usr/bin/env python3
"""
Service Orchestrator for GTerminal DevOps Stack
Manages lifecycle of all integrated Rust and Python services
"""

import asyncio
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import aiohttp
import psutil
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/var/log/gterminal/orchestrator.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ServiceState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    DEGRADED = "degraded"


@dataclass
class ServiceConfig:
    name: str
    command: list[str]
    working_dir: str
    env: dict[str, str]
    port: int | None = None
    health_check_url: str | None = None
    health_check_interval: int = 30
    restart_policy: str = "always"  # always, on_failure, never
    dependencies: list[str] = None
    timeout: int = 60
    cpu_limit: float | None = None
    memory_limit: int | None = None


@dataclass
class ServiceStatus:
    name: str
    state: ServiceState
    pid: int | None = None
    start_time: float | None = None
    last_health_check: float | None = None
    health_status: str = "unknown"
    restart_count: int = 0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    error_message: str | None = None


class ServiceOrchestrator:
    def __init__(self, config_file: str = "orchestration/services.yml"):
        self.config_file = config_file
        self.services: dict[str, ServiceConfig] = {}
        self.statuses: dict[str, ServiceStatus] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self.health_tasks: dict[str, asyncio.Task] = {}
        self.shutdown_event = asyncio.Event()

        # Load configuration
        self.load_config()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown_all())

    def load_config(self):
        """Load service configuration from YAML file"""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                self.create_default_config()

            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            for service_name, service_config in config_data.get("services", {}).items():
                self.services[service_name] = ServiceConfig(name=service_name, **service_config)
                self.statuses[service_name] = ServiceStatus(
                    name=service_name, state=ServiceState.STOPPED
                )

            logger.info(f"Loaded {len(self.services)} service configurations")

        except Exception as e:
            logger.exception(f"Failed to load configuration: {e}")
            sys.exit(1)

    def create_default_config(self):
        """Create default service configuration"""
        default_config = {
            "services": {
                "rust-filewatcher": {
                    "command": ["rust-filewatcher", "--config", "/config/filewatcher.toml"],
                    "working_dir": "/app",
                    "env": {"RUST_LOG": "info", "WEBSOCKET_PORT": "8765", "METRICS_PORT": "8766"},
                    "port": 8765,
                    "health_check_url": "http://localhost:8766/health",
                    "dependencies": [],
                },
                "ruff-lsp": {
                    "command": ["python", "/app/lsp-server.py"],
                    "working_dir": "/app",
                    "env": {"LSP_PORT": "8767", "METRICS_PORT": "8768"},
                    "port": 8767,
                    "health_check_url": "http://localhost:8768/health",
                    "dependencies": [],
                },
                "development-dashboard": {
                    "command": ["python", "/app/dashboard_server.py"],
                    "working_dir": "/app",
                    "env": {"DASHBOARD_PORT": "8080"},
                    "port": 8080,
                    "health_check_url": "http://localhost:8080/health",
                    "dependencies": ["rust-filewatcher", "ruff-lsp"],
                },
                "gterminal-app": {
                    "command": ["python", "-m", "gterminal.server"],
                    "working_dir": "/app",
                    "env": {
                        "PORT": "8000",
                        "FILEWATCHER_WS_URL": "ws://localhost:8765",
                        "LSP_SERVER_URL": "http://localhost:8767",
                    },
                    "port": 8000,
                    "health_check_url": "http://localhost:8000/health",
                    "dependencies": ["rust-filewatcher", "ruff-lsp"],
                },
            }
        }

        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

        logger.info(f"Created default configuration: {self.config_file}")

    async def start_service(self, service_name: str) -> bool:
        """Start a specific service"""
        if service_name not in self.services:
            logger.error(f"Service '{service_name}' not found")
            return False

        service = self.services[service_name]
        status = self.statuses[service_name]

        # Check if already running
        if status.state in [ServiceState.RUNNING, ServiceState.STARTING]:
            logger.warning(f"Service '{service_name}' is already {status.state.value}")
            return True

        # Check dependencies
        if service.dependencies:
            for dep in service.dependencies:
                dep_status = self.statuses.get(dep)
                if not dep_status or dep_status.state != ServiceState.RUNNING:
                    logger.error(f"Dependency '{dep}' is not running for service '{service_name}'")
                    return False

        logger.info(f"Starting service '{service_name}'...")
        status.state = ServiceState.STARTING

        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(service.env)

            # Start the process
            process = subprocess.Popen(
                service.command,
                cwd=service.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            self.processes[service_name] = process
            status.pid = process.pid
            status.start_time = time.time()
            status.restart_count += 1

            # Wait a bit for startup
            await asyncio.sleep(2)

            # Check if process is still running
            if process.poll() is None:
                status.state = ServiceState.RUNNING
                logger.info(f"Service '{service_name}' started successfully (PID: {process.pid})")

                # Start health monitoring
                if service.health_check_url:
                    self.health_tasks[service_name] = asyncio.create_task(
                        self.monitor_health(service_name)
                    )

                # Start log monitoring
                asyncio.create_task(self.monitor_logs(service_name))

                return True
            else:
                output, _ = process.communicate()
                status.state = ServiceState.ERROR
                status.error_message = f"Process exited with code {process.returncode}: {output}"
                logger.error(f"Service '{service_name}' failed to start: {status.error_message}")
                return False

        except Exception as e:
            status.state = ServiceState.ERROR
            status.error_message = str(e)
            logger.exception(f"Failed to start service '{service_name}': {e}")
            return False

    async def stop_service(self, service_name: str, force: bool = False) -> bool:
        """Stop a specific service"""
        if service_name not in self.services:
            logger.error(f"Service '{service_name}' not found")
            return False

        status = self.statuses[service_name]
        process = self.processes.get(service_name)

        if status.state == ServiceState.STOPPED:
            logger.warning(f"Service '{service_name}' is already stopped")
            return True

        logger.info(f"Stopping service '{service_name}'...")
        status.state = ServiceState.STOPPING

        # Cancel health monitoring
        if service_name in self.health_tasks:
            self.health_tasks[service_name].cancel()
            del self.health_tasks[service_name]

        # Stop the process
        if process and process.poll() is None:
            try:
                if force:
                    process.kill()
                else:
                    process.terminate()

                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_process(process)), timeout=10
                    )
                except TimeoutError:
                    logger.warning(f"Service '{service_name}' did not stop gracefully, killing...")
                    process.kill()

                del self.processes[service_name]

            except Exception as e:
                logger.exception(f"Error stopping service '{service_name}': {e}")

        status.state = ServiceState.STOPPED
        status.pid = None
        status.start_time = None
        logger.info(f"Service '{service_name}' stopped")
        return True

    async def _wait_for_process(self, process):
        """Wait for process to terminate"""
        while process.poll() is None:
            await asyncio.sleep(0.1)

    async def restart_service(self, service_name: str) -> bool:
        """Restart a service"""
        logger.info(f"Restarting service '{service_name}'...")
        await self.stop_service(service_name)
        await asyncio.sleep(1)
        return await self.start_service(service_name)

    async def monitor_health(self, service_name: str):
        """Monitor service health"""
        service = self.services[service_name]
        status = self.statuses[service_name]

        if not service.health_check_url:
            return

        while status.state == ServiceState.RUNNING:
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(service.health_check_url) as response:
                        if response.status == 200:
                            status.health_status = "healthy"
                            status.last_health_check = time.time()
                        else:
                            status.health_status = "unhealthy"
                            logger.warning(
                                f"Health check failed for '{service_name}': HTTP {response.status}"
                            )

            except Exception as e:
                status.health_status = "error"
                logger.warning(f"Health check error for '{service_name}': {e}")

            await asyncio.sleep(service.health_check_interval)

    async def monitor_logs(self, service_name: str):
        """Monitor service logs"""
        process = self.processes.get(service_name)
        if not process:
            return

        try:
            while process.poll() is None:
                line = await asyncio.create_task(self._read_line_async(process))
                if line:
                    logger.info(f"[{service_name}] {line.strip()}")
                else:
                    break
        except Exception as e:
            logger.exception(f"Log monitoring error for '{service_name}': {e}")

    async def _read_line_async(self, process):
        """Read line from process stdout asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, process.stdout.readline)

    def update_resource_usage(self):
        """Update CPU and memory usage for running services"""
        for _service_name, status in self.statuses.items():
            if status.pid:
                try:
                    process = psutil.Process(status.pid)
                    status.cpu_percent = process.cpu_percent()
                    status.memory_mb = process.memory_info().rss / 1024 / 1024
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

    async def start_all(self, order: list[str] | None = None):
        """Start all services in dependency order"""
        logger.info("Starting all services...")

        if order is None:
            order = self._resolve_start_order()

        for service_name in order:
            success = await self.start_service(service_name)
            if not success and self.services[service_name].restart_policy != "never":
                logger.warning(f"Failed to start '{service_name}', but continuing...")
            await asyncio.sleep(1)  # Brief pause between starts

        logger.info("All services started")

    async def stop_all(self, force: bool = False):
        """Stop all services in reverse dependency order"""
        logger.info("Stopping all services...")

        order = self._resolve_stop_order()

        for service_name in order:
            await self.stop_service(service_name, force=force)
            await asyncio.sleep(0.5)  # Brief pause between stops

        logger.info("All services stopped")

    def _resolve_start_order(self) -> list[str]:
        """Resolve service start order based on dependencies"""
        ordered = []
        remaining = set(self.services.keys())

        while remaining:
            # Find services with no unmet dependencies
            ready = []
            for service_name in remaining:
                service = self.services[service_name]
                if not service.dependencies or all(dep in ordered for dep in service.dependencies):
                    ready.append(service_name)

            if not ready:
                # Circular dependency or missing dependency
                logger.warning(
                    "Circular dependency detected, starting remaining services in arbitrary order"
                )
                ready = list(remaining)

            for service_name in ready:
                ordered.append(service_name)
                remaining.remove(service_name)

        return ordered

    def _resolve_stop_order(self) -> list[str]:
        """Resolve service stop order (reverse of start order)"""
        return list(reversed(self._resolve_start_order()))

    async def shutdown_all(self):
        """Graceful shutdown of orchestrator and all services"""
        logger.info("Initiating graceful shutdown...")
        self.shutdown_event.set()
        await self.stop_all(force=False)

    def get_status(self) -> dict[str, Any]:
        """Get status of all services"""
        self.update_resource_usage()

        return {
            "services": {name: asdict(status) for name, status in self.statuses.items()},
            "orchestrator": {
                "uptime": time.time() - self.start_time if hasattr(self, "start_time") else 0,
                "total_services": len(self.services),
                "running_services": len(
                    [s for s in self.statuses.values() if s.state == ServiceState.RUNNING]
                ),
                "healthy_services": len(
                    [s for s in self.statuses.values() if s.health_status == "healthy"]
                ),
            },
        }

    async def run(self):
        """Main orchestrator loop"""
        self.start_time = time.time()
        logger.info("Service orchestrator starting...")

        try:
            # Start all services
            await self.start_all()

            # Main monitoring loop
            while not self.shutdown_event.is_set():
                # Update resource usage
                self.update_resource_usage()

                # Check for failed services and restart if needed
                for service_name, status in self.statuses.items():
                    service = self.services[service_name]

                    if status.state == ServiceState.ERROR and service.restart_policy in [
                        "always",
                        "on_failure",
                    ]:
                        logger.info(f"Restarting failed service '{service_name}'...")
                        await self.restart_service(service_name)

                await asyncio.sleep(5)  # Check every 5 seconds

        except Exception as e:
            logger.exception(f"Orchestrator error: {e}")
        finally:
            await self.stop_all(force=True)
            logger.info("Service orchestrator stopped")


# CLI interface
async def main():
    import argparse

    parser = argparse.ArgumentParser(description="GTerminal Service Orchestrator")
    parser.add_argument("--config", default="orchestration/services.yml", help="Configuration file")
    parser.add_argument(
        "--action",
        choices=["start", "stop", "restart", "status"],
        default="run",
        help="Action to perform",
    )
    parser.add_argument("--service", help="Specific service name (for start/stop/restart)")
    parser.add_argument("--force", action="store_true", help="Force stop services")

    args = parser.parse_args()

    orchestrator = ServiceOrchestrator(args.config)

    if args.action == "start":
        if args.service:
            await orchestrator.start_service(args.service)
        else:
            await orchestrator.start_all()
    elif args.action == "stop":
        if args.service:
            await orchestrator.stop_service(args.service, args.force)
        else:
            await orchestrator.stop_all(args.force)
    elif args.action == "restart":
        if args.service:
            await orchestrator.restart_service(args.service)
        else:
            await orchestrator.stop_all()
            await asyncio.sleep(2)
            await orchestrator.start_all()
    elif args.action == "status":
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2))
    else:
        # Run orchestrator
        await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
