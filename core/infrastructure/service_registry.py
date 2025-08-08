#!/usr/bin/env python3
"""Service Discovery and Registry System.

This module provides dynamic service discovery, health monitoring, and
automatic failover capabilities for the MCP server architecture.
"""

import asyncio
import contextlib
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
import logging
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import aiohttp

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class ServiceInfo:
    """Information about a registered service."""

    name: str
    host: str
    port: int
    protocol: str = "https"
    health_endpoint: str = "/health"
    capabilities: list[str] = field(default_factory=list)
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_health_check: datetime | None = None
    status: str = "starting"  # starting, healthy, unhealthy, unknown
    consecutive_failures: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        """Get the base URL for this service."""
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        """Get the health check URL for this service."""
        return urljoin(self.url, self.health_endpoint)

    @property
    def is_healthy(self) -> bool:
        """Check if the service is currently healthy."""
        return self.status == "healthy"

    @property
    def uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return (datetime.now(UTC) - self.registered_at).total_seconds()


class ServiceRegistry:
    """Dynamic service discovery and health monitoring registry.

    Features:
    - Automatic service registration and discovery
    - Health monitoring with configurable intervals
    - Automatic failover and load balancing
    - Service capability matching
    - Event-based notifications
    """

    def __init__(
        self,
        health_check_interval: int = 30,
        max_consecutive_failures: int = 3,
        request_timeout: int = 5,
    ) -> None:
        self.services: dict[str, ServiceInfo] = {}
        self.health_check_interval = health_check_interval
        self.max_consecutive_failures = max_consecutive_failures
        self.request_timeout = request_timeout

        # Event callbacks
        self.on_service_healthy: Callable[[ServiceInfo], None] | None = None
        self.on_service_unhealthy: Callable[[ServiceInfo], None] | None = None
        self.on_service_registered: Callable[[ServiceInfo], None] | None = None
        self.on_service_deregistered: Callable[[ServiceInfo], None] | None = None

        # Background tasks
        self._health_monitor_task: asyncio.Task | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._running = False

    async def start(self) -> None:
        """Start the service registry and health monitoring."""
        if self._running:
            return

        self._running = True

        # Create HTTP session for health checks
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=10,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        self._http_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        )

        # Start health monitoring
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

        logger.info("Service registry started")

    async def stop(self) -> None:
        """Stop the service registry and cleanup resources."""
        if not self._running:
            return

        self._running = False

        # Cancel health monitoring
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_monitor_task

        # Close HTTP session
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

        logger.info("Service registry stopped")

    async def register_service(
        self,
        name: str,
        host: str,
        port: int,
        protocol: str = "https",
        health_endpoint: str = "/health",
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ServiceInfo:
        """Register a service with the registry.

        Args:
            name: Unique service name
            host: Service host/IP address
            port: Service port number
            protocol: Service protocol (http/https)
            health_endpoint: Health check endpoint path
            capabilities: List of service capabilities
            metadata: Additional service metadata

        Returns:
            ServiceInfo object for the registered service

        """
        service_info = ServiceInfo(
            name=name,
            host=host,
            port=port,
            protocol=protocol,
            health_endpoint=health_endpoint,
            capabilities=capabilities or [],
            metadata=metadata or {},
        )

        self.services[name] = service_info

        # Trigger registration callback
        if self.on_service_registered:
            try:
                self.on_service_registered(service_info)
            except Exception as e:
                logger.exception(f"Error in service registration callback: {e}")

        logger.info(f"Registered service: {name} at {service_info.url}")

        # Perform initial health check
        if self._http_session:
            asyncio.create_task(self._health_check_service(service_info))

        return service_info

    async def deregister_service(self, name: str) -> bool:
        """Deregister a service from the registry.

        Args:
            name: Service name to deregister

        Returns:
            True if service was deregistered, False if not found

        """
        if name not in self.services:
            return False

        service_info = self.services.pop(name)

        # Trigger deregistration callback
        if self.on_service_deregistered:
            try:
                self.on_service_deregistered(service_info)
            except Exception as e:
                logger.exception(f"Error in service deregistration callback: {e}")

        logger.info(f"Deregistered service: {name}")
        return True

    async def discover_service(self, name: str) -> ServiceInfo | None:
        """Discover a service by name.

        Args:
            name: Service name to discover

        Returns:
            ServiceInfo if service is found and healthy, None otherwise

        """
        if name not in self.services:
            return None

        service = self.services[name]

        # Only return healthy services
        if service.is_healthy:
            return service

        return None

    async def discover_services_by_capability(self, capability: str) -> list[ServiceInfo]:
        """Discover services that have a specific capability.

        Args:
            capability: Required capability

        Returns:
            List of healthy services with the specified capability

        """
        matching_services: list[Any] = []

        for service in self.services.values():
            if service.is_healthy and capability in service.capabilities:
                matching_services.append(service)

        # Sort by uptime (prefer more stable services)
        matching_services.sort(key=lambda s: s.uptime_seconds, reverse=True)

        return matching_services

    async def get_healthy_services(self) -> list[ServiceInfo]:
        """Get all currently healthy services."""
        return [service for service in self.services.values() if service.is_healthy]

    async def get_service_status(self) -> dict[str, Any]:
        """Get comprehensive status of all services."""
        total_services = len(self.services)
        healthy_services = len(await self.get_healthy_services())

        status_summary = {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": total_services - healthy_services,
            "health_percentage": (healthy_services / total_services * 100)
            if total_services > 0
            else 0,
            "services": {},
        }

        for name, service in self.services.items():
            status_summary["services"][name] = {
                "status": service.status,
                "url": service.url,
                "uptime_seconds": service.uptime_seconds,
                "last_health_check": service.last_health_check.isoformat()
                if service.last_health_check
                else None,
                "consecutive_failures": service.consecutive_failures,
                "capabilities": service.capabilities,
            }

        return status_summary

    async def _health_monitor_loop(self) -> None:
        """Background task for monitoring service health."""
        while self._running:
            try:
                # Check all services in parallel
                health_check_tasks = [
                    self._health_check_service(service) for service in self.services.values()
                ]

                if health_check_tasks:
                    await asyncio.gather(*health_check_tasks, return_exceptions=True)

                # Wait for next health check interval
                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in health monitor loop: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _health_check_service(self, service: ServiceInfo) -> None:
        """Perform health check on a single service."""
        if not self._http_session:
            return

        try:
            start_time = time.time()

            async with self._http_session.get(service.health_url) as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    # Service is healthy
                    previous_status = service.status
                    service.status = "healthy"
                    service.last_health_check = datetime.now(UTC)
                    service.consecutive_failures = 0
                    service.metadata["response_time_ms"] = response_time_ms

                    # Trigger healthy callback if status changed
                    if previous_status != "healthy" and self.on_service_healthy:
                        try:
                            self.on_service_healthy(service)
                        except Exception as e:
                            logger.exception(f"Error in service healthy callback: {e}")

                else:
                    # Service returned non-200 status
                    await self._mark_service_unhealthy(service, f"HTTP {response.status}")

        except TimeoutError:
            await self._mark_service_unhealthy(service, "Timeout")
        except aiohttp.ClientError as e:
            await self._mark_service_unhealthy(service, f"Connection error: {e}")
        except Exception as e:
            await self._mark_service_unhealthy(service, f"Unexpected error: {e}")

    async def _mark_service_unhealthy(self, service: ServiceInfo, reason: str) -> None:
        """Mark a service as unhealthy and handle failure threshold."""
        service.consecutive_failures += 1
        service.last_health_check = datetime.now(UTC)
        service.metadata["last_failure_reason"] = reason

        if service.consecutive_failures >= self.max_consecutive_failures:
            if service.status != "unhealthy":
                service.status = "unhealthy"

                # Trigger unhealthy callback
                if self.on_service_unhealthy:
                    try:
                        self.on_service_unhealthy(service)
                    except Exception as e:
                        logger.exception(f"Error in service unhealthy callback: {e}")

                logger.warning(
                    f"Service {service.name} marked unhealthy after "
                    f"{service.consecutive_failures} consecutive failures. "
                    f"Last failure: {reason}",
                )
        else:
            logger.debug(
                f"Health check failed for {service.name} "
                f"({service.consecutive_failures}/{self.max_consecutive_failures}): {reason}",
            )


# Global service registry instance
_global_registry: ServiceRegistry | None = None


async def get_service_registry() -> ServiceRegistry:
    """Get or create the global service registry instance."""
    global _global_registry

    if _global_registry is None:
        _global_registry = ServiceRegistry()
        await _global_registry.start()

    return _global_registry


async def shutdown_service_registry() -> None:
    """Shutdown the global service registry."""
    global _global_registry

    if _global_registry:
        await _global_registry.stop()
        _global_registry = None
