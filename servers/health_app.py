#!/usr/bin/env python3
"""Health Check FastAPI Application.

Comprehensive health monitoring and status endpoints for the AI Assistant
and all agent services including MCP servers.
"""

import asyncio
import builtins
import contextlib
from datetime import datetime
from datetime import timedelta
import logging
import os
import platform
import time
from typing import Any

import aiohttp
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psutil
from pydantic import BaseModel
from pydantic import Field
import redis.asyncio as redis

from gterminal.agent_endpoints import router as agent_router
from gterminal.agents.code_generation_agent import code_generation_service

# Import agent services
from gterminal.agents.code_review_agent import code_review_service
from gterminal.agents.documentation_generator_agent import documentation_service
from gterminal.agents.master_architect_agent import architect_service
from gterminal.agents.workspace_analyzer_agent import workspace_analyzer_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
# TODO: Add comprehensive OpenAPI documentation with examples and schemas
# TODO: Add API versioning strategy and deprecation policies
app = FastAPI(
    title="AI Assistant Health Check API",
    description="Comprehensive health monitoring for AI Assistant and agent services",
    version="1.0.0",
    docs_url="/health/docs",
    redoc_url="/health/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include agent endpoints
app.include_router(agent_router)

# Global variables for caching health data
_health_cache: dict[str, Any] = {}
_cache_timestamp = None
_cache_duration = 30  # Cache health data for 30 seconds


# ========================
# Response Models
# ========================


class ServiceHealth(BaseModel):
    """Model for individual service health status."""

    name: str
    status: str = Field(..., description="healthy, unhealthy, or degraded")
    uptime_seconds: float
    version: str = "1.0.0"
    last_check: str
    details: dict[str, Any] = Field(default_factory=dict)
    response_time_ms: float | None = None
    error: str | None = None


class SystemHealth(BaseModel):
    """Model for overall system health."""

    status: str = Field(..., description="healthy, unhealthy, or degraded")
    timestamp: str
    uptime_seconds: float
    services: list[ServiceHealth]
    system_metrics: dict[str, Any]
    summary: dict[str, Any]


class DatabaseHealth(BaseModel):
    """Model for database health status."""

    connected: bool
    connection_pool_size: int
    active_connections: int
    response_time_ms: float
    last_query_time: str | None = None
    error: str | None = None


class RedisHealth(BaseModel):
    """Model for Redis health status."""

    connected: bool
    memory_usage_mb: float
    connected_clients: int
    keys_count: int
    response_time_ms: float
    error: str | None = None


# ========================
# Health Check Functions
# ========================


async def check_agent_service_health(service_name: str, service_instance) -> ServiceHealth:
    """Check health of an individual agent service."""
    start_time = time.time()

    try:
        # Get service statistics
        stats = service_instance.get_agent_stats()

        response_time = (time.time() - start_time) * 1000

        # Determine status based on service health
        status = "healthy"
        if stats.get("active_jobs", 0) > 10:  # Too many active jobs
            status = "degraded"
        if stats.get("failed_jobs", 0) > stats.get("total_jobs", 1) * 0.1:  # >10% failure rate
            status = "degraded"

        return ServiceHealth(
            name=service_name,
            status=status,
            uptime_seconds=stats.get("uptime_seconds", 0),
            last_check=datetime.utcnow().isoformat(),
            response_time_ms=response_time,
            details={
                "active_jobs": stats.get("active_jobs", 0),
                "total_jobs": stats.get("total_jobs", 0),
                "failed_jobs": stats.get("failed_jobs", 0),
                "success_rate": stats.get("success_rate", 0.0),
                "avg_job_duration": stats.get("avg_job_duration", 0.0),
                "memory_usage_mb": stats.get("memory_usage_mb", 0),
            },
        )

    except Exception as e:
        logger.exception(f"Health check failed for {service_name}: {e}")
        return ServiceHealth(
            name=service_name,
            status="unhealthy",
            uptime_seconds=0,
            last_check=datetime.utcnow().isoformat(),
            error=str(e),
        )


async def check_mcp_server_health(server_name: str, port: int) -> ServiceHealth:
    """Check health of an MCP server."""
    start_time = time.time()

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"http://localhost:{port}/health") as response:
                response_time = (time.time() - start_time) * 1000

                if response.status == 200:
                    data = await response.json()
                    return ServiceHealth(
                        name=server_name,
                        status="healthy",
                        uptime_seconds=data.get("uptime", 0),
                        last_check=datetime.utcnow().isoformat(),
                        response_time_ms=response_time,
                        details=data,
                    )
                return ServiceHealth(
                    name=server_name,
                    status="unhealthy",
                    uptime_seconds=0,
                    last_check=datetime.utcnow().isoformat(),
                    response_time_ms=response_time,
                    error=f"HTTP {response.status}",
                )

    except Exception as e:
        logger.warning(f"MCP server {server_name} health check failed: {e}")
        return ServiceHealth(
            name=server_name,
            status="unhealthy",
            uptime_seconds=0,
            last_check=datetime.utcnow().isoformat(),
            error=str(e),
        )


async def check_redis_health() -> RedisHealth:
    """Check Redis connection and health."""
    start_time = time.time()

    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url)

        # Test connection and get info
        await redis_client.ping()
        info = await redis_client.info()

        response_time = (time.time() - start_time) * 1000

        return RedisHealth(
            connected=True,
            memory_usage_mb=info.get("used_memory", 0) / 1024 / 1024,
            connected_clients=info.get("connected_clients", 0),
            keys_count=await redis_client.dbsize(),
            response_time_ms=response_time,
        )

    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return RedisHealth(
            connected=False,
            memory_usage_mb=0,
            connected_clients=0,
            keys_count=0,
            response_time_ms=0,
            error=str(e),
        )
    finally:
        with contextlib.suppress(builtins.BaseException):
            await redis_client.close()


async def check_database_health() -> DatabaseHealth:
    """Check database connection and health.

    # TODO: Implement actual database health checks with real connection pooling
    # TODO: Add database-specific metrics (query performance, connection counts)
    # TODO: Add support for multiple database types (PostgreSQL, MySQL, etc.)
    """
    start_time = time.time()

    try:
        # For now, we'll simulate database health since we don't have a direct DB connection
        # In a real implementation, you'd connect to your actual database

        # Simulate database check
        await asyncio.sleep(0.01)  # Simulate DB query time

        response_time = (time.time() - start_time) * 1000

        return DatabaseHealth(
            connected=True,
            connection_pool_size=10,
            active_connections=2,
            response_time_ms=response_time,
            last_query_time=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return DatabaseHealth(
            connected=False,
            connection_pool_size=0,
            active_connections=0,
            response_time_ms=0,
            error=str(e),
        )


def get_system_metrics() -> dict[str, Any]:
    """Get system-level metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / 1024 / 1024 / 1024,
            "memory_total_gb": memory.total / 1024 / 1024 / 1024,
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / 1024 / 1024 / 1024,
            "disk_total_gb": disk.total / 1024 / 1024 / 1024,
            "load_average": os.getloadavg() if hasattr(os, "getloadavg") else [0, 0, 0],
            "uptime_seconds": time.time() - psutil.boot_time(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }
    except Exception as e:
        logger.exception(f"Failed to get system metrics: {e}")
        return {"error": str(e)}


async def get_comprehensive_health() -> SystemHealth:
    """Get comprehensive health status of all services."""
    global _health_cache, _cache_timestamp

    # Check cache first
    if (
        _cache_timestamp
        and datetime.utcnow() - _cache_timestamp < timedelta(seconds=_cache_duration)
        and _health_cache
    ):
        return SystemHealth(**_health_cache)

    start_time = time.time()

    # Check all services concurrently
    health_checks: list[Any] = []

    # Agent services
    agent_services = [
        ("code_review_agent", code_review_service),
        ("workspace_analyzer_agent", workspace_analyzer_service),
        ("documentation_generator_agent", documentation_service),
        ("master_architect_agent", architect_service),
        ("code_generation_agent", code_generation_service),
    ]

    for service_name, service_instance in agent_services:
        health_checks.append(check_agent_service_health(service_name, service_instance))

    # MCP servers
    mcp_servers = [
        ("mcp_code_reviewer", 3001),
        ("mcp_workspace_analyzer", 3002),
        ("mcp_documentation", 3003),
        ("mcp_master_architect", 3004),
        ("mcp_code_generator", 3005),
    ]

    for server_name, port in mcp_servers:
        health_checks.append(check_mcp_server_health(server_name, port))

    # External dependencies
    health_checks.extend(
        [
            check_redis_health(),
            check_database_health(),
        ],
    )

    # Execute all health checks concurrently
    results = await asyncio.gather(*health_checks, return_exceptions=True)

    # Process results
    services: list[Any] = []
    redis_health = None
    db_health = None

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Health check {i} failed: {result}")
            continue

        if isinstance(result, ServiceHealth):
            services.append(result)
        elif isinstance(result, RedisHealth):
            redis_health = result
        elif isinstance(result, DatabaseHealth):
            db_health = result

    # Get system metrics
    system_metrics = get_system_metrics()

    # Add external service health to system metrics
    if redis_health:
        system_metrics["redis"] = redis_health.dict()
    if db_health:
        system_metrics["database"] = db_health.dict()

    # Determine overall status
    healthy_services = sum(1 for s in services if s.status == "healthy")
    degraded_services = sum(1 for s in services if s.status == "degraded")
    unhealthy_services = sum(1 for s in services if s.status == "unhealthy")

    if unhealthy_services == 0 and degraded_services == 0:
        overall_status = "healthy"
    elif unhealthy_services == 0:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    # Create summary
    summary = {
        "total_services": len(services),
        "healthy_services": healthy_services,
        "degraded_services": degraded_services,
        "unhealthy_services": unhealthy_services,
        "overall_response_time_ms": (time.time() - start_time) * 1000,
        "redis_connected": redis_health.connected if redis_health else False,
        "database_connected": db_health.connected if db_health else False,
    }

    health_data = SystemHealth(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=system_metrics.get("uptime_seconds", 0),
        services=services,
        system_metrics=system_metrics,
        summary=summary,
    )

    # Update cache
    _health_cache = health_data.dict()
    _cache_timestamp = datetime.utcnow()

    return health_data


# ========================
# Health Check Endpoints
# ========================


@app.get("/health", response_model=dict)
async def basic_health_check():
    """Basic health check endpoint for load balancers."""
    try:
        # Quick health check - just verify the application is responding
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "ai-assistant",
            "version": "1.0.0",
        }
    except Exception as e:
        logger.exception(f"Basic health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/health/detailed", response_model=SystemHealth)
async def detailed_health_check():
    """Comprehensive health check with all service details."""
    try:
        return await get_comprehensive_health()
    except Exception as e:
        logger.exception(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {e!s}")


@app.get("/health/services", response_model=list[ServiceHealth])
async def services_health_check():
    """Health check for all agent services only."""
    try:
        health_data = await get_comprehensive_health()
        return health_data.services
    except Exception as e:
        logger.exception(f"Services health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Services health check failed: {e!s}")


@app.get("/health/mcp", response_model=list[ServiceHealth])
async def mcp_servers_health_check():
    """Health check for MCP servers only."""
    try:
        health_data = await get_comprehensive_health()
        return [s for s in health_data.services if s.name.startswith("mcp_")]
    except Exception as e:
        logger.exception(f"MCP servers health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"MCP health check failed: {e!s}")


@app.get("/health/system")
async def system_health_check():
    """System metrics and resource usage."""
    try:
        return get_system_metrics()
    except Exception as e:
        logger.exception(f"System health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"System health check failed: {e!s}")


@app.get("/health/redis", response_model=RedisHealth)
async def redis_health_check():
    """Redis-specific health check."""
    try:
        return await check_redis_health()
    except Exception as e:
        logger.exception(f"Redis health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Redis health check failed: {e!s}")


@app.get("/health/database", response_model=DatabaseHealth)
async def database_health_check():
    """Database-specific health check."""
    try:
        return await check_database_health()
    except Exception as e:
        logger.exception(f"Database health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Database health check failed: {e!s}")


@app.get("/readiness")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    try:
        health_data = await get_comprehensive_health()

        # Service is ready if at least 80% of services are healthy or degraded
        total_services = health_data.summary["total_services"]
        healthy_or_degraded = (
            health_data.summary["healthy_services"] + health_data.summary["degraded_services"]
        )

        if total_services == 0 or (healthy_or_degraded / total_services) >= 0.8:
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat(),
                "healthy_services": healthy_or_degraded,
                "total_services": total_services,
            }
        raise HTTPException(status_code=503, detail="Service not ready")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/liveness")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    try:
        # Simple liveness check - just verify the application is running
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - psutil.Process().create_time(),
        }
    except Exception as e:
        logger.exception(f"Liveness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not alive")


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus-compatible metrics endpoint."""
    try:
        health_data = await get_comprehensive_health()

        # Generate Prometheus-style metrics
        metrics: list[Any] = []

        # Service metrics
        for service in health_data.services:
            status_value = 1 if service.status == "healthy" else 0
            metrics.append(
                f'ai_assistant_service_health{{service="{service.name}"}} {status_value}'
            )

            if service.response_time_ms:
                metrics.append(
                    f'ai_assistant_service_response_time_ms{{service="{service.name}"}} {service.response_time_ms}',
                )

            # Service-specific metrics
            for key, value in service.details.items():
                if isinstance(value, int | float):
                    metrics.append(
                        f'ai_assistant_service_{key}{{service="{service.name}"}} {value}'
                    )

        # System metrics
        sys_metrics = health_data.system_metrics
        for key, value in sys_metrics.items():
            if isinstance(value, int | float):
                metrics.append(f"ai_assistant_system_{key} {value}")

        # Summary metrics
        for key, value in health_data.summary.items():
            if isinstance(value, int | float):
                metrics.append(f"ai_assistant_summary_{key} {value}")

        return JSONResponse(content="\n".join(metrics), media_type="text/plain")

    except Exception as e:
        logger.exception(f"Metrics endpoint failed: {e}")
        raise HTTPException(status_code=503, detail=f"Metrics unavailable: {e!s}")


# ========================
# Startup and Shutdown
# ========================


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the health check service."""
    logger.info("Health check service starting up...")

    # Warm up the health cache
    try:
        await get_comprehensive_health()
        logger.info("Health check service started successfully")
    except Exception as e:
        logger.exception(f"Failed to initialize health check service: {e}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources on shutdown."""
    logger.info("Health check service shutting down...")


if __name__ == "__main__":
    import uvicorn

    # Configure for production
    uvicorn.run(
        app, host="0.0.0.0", port=8080, workers=1, loop="uvloop", access_log=True, log_level="info"
    )
