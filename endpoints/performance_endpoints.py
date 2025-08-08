"""Performance monitoring endpoints for the fullstack agent.

Provides HTTP endpoints for monitoring cache performance, connection pool
statistics, and overall system health.
"""

import asyncio
import logging

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from .performance_integration import get_performance_integration

logger = logging.getLogger(__name__)


class PerformanceAPI:
    """FastAPI application for performance monitoring."""

    def __init__(self) -> None:
        self.app = FastAPI(
            title="Fullstack Agent Performance Monitor",
            description="Performance monitoring and optimization dashboard",
            version="1.0.0",
        )
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                integration = await get_performance_integration()
                health = await integration.health_check()
                return JSONResponse(content=health)
            except Exception as e:
                logger.exception(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail="Service Unavailable")

        @self.app.get("/metrics")
        async def get_metrics():
            """Get comprehensive performance metrics."""
            try:
                integration = await get_performance_integration()
                report = await integration.get_performance_dashboard()
                return JSONResponse(content=report)
            except Exception as e:
                logger.exception(f"Failed to get metrics: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.get("/cache/stats")
        async def get_cache_stats():
            """Get cache statistics."""
            try:
                integration = await get_performance_integration()
                cache_stats = await integration.optimizer.cache_manager.get_stats()
                return JSONResponse(content=cache_stats)
            except Exception as e:
                logger.exception(f"Failed to get cache stats: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.get("/connections/stats")
        async def get_connection_stats():
            """Get connection pool statistics."""
            try:
                integration = await get_performance_integration()
                conn_stats = (
                    await integration.optimizer.connection_manager.get_comprehensive_stats()
                )
                return JSONResponse(content=conn_stats)
            except Exception as e:
                logger.exception(f"Failed to get connection stats: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.post("/cache/clear")
        async def clear_cache(pattern: str | None = None):
            """Clear cache entries matching pattern."""
            try:
                integration = await get_performance_integration()
                count = await integration.optimizer.cache_manager.clear(pattern)
                return JSONResponse(
                    content={
                        "cleared_items": count,
                        "pattern": pattern or "all",
                        "timestamp": asyncio.get_event_loop().time(),
                    },
                )
            except Exception as e:
                logger.exception(f"Failed to clear cache: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.post("/cache/warm-up")
        async def warm_up_cache():
            """Warm up cache with common data."""
            try:
                integration = await get_performance_integration()
                await integration.optimizer.warm_up_cache()
                return JSONResponse(
                    content={
                        "status": "success",
                        "message": "Cache warm-up completed",
                        "timestamp": asyncio.get_event_loop().time(),
                    },
                )
            except Exception as e:
                logger.exception(f"Failed to warm up cache: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.get("/performance/budgets")
        async def get_performance_budgets():
            """Get performance budget status."""
            try:
                integration = await get_performance_integration()
                report = await integration.get_performance_dashboard()
                return JSONResponse(content=report.get("budgets", {}))
            except Exception as e:
                logger.exception(f"Failed to get performance budgets: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.get("/performance/dashboard")
        async def get_performance_dashboard():
            """Get complete performance dashboard."""
            try:
                integration = await get_performance_integration()
                dashboard = await integration.get_performance_dashboard()
                return JSONResponse(content=dashboard)
            except Exception as e:
                logger.exception(f"Failed to get performance dashboard: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.get("/performance/summary")
        async def get_performance_summary():
            """Get performance summary for quick overview."""
            try:
                integration = await get_performance_integration()
                dashboard = await integration.get_performance_dashboard()

                summary = {
                    "timestamp": dashboard["timestamp"],
                    "overall_health": dashboard["health"]["status"],
                    "cache_hit_rate": dashboard["performance_summary"]["cache_hit_rate"],
                    "total_requests": dashboard["performance_summary"]["total_requests"],
                    "error_rate": dashboard["performance_summary"]["error_rate"],
                    "redis_enabled": dashboard["performance_summary"]["redis_enabled"],
                    "budget_violations": [
                        name
                        for name, budget in dashboard["budgets"].items()
                        if budget.get("status") in ["POOR", "CRITICAL"]
                    ],
                }

                return JSONResponse(content=summary)
            except Exception as e:
                logger.exception(f"Failed to get performance summary: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")


# Global performance API instance
performance_api = PerformanceAPI()


def create_performance_monitoring_server(
    host: str = "0.0.0.0",
    port: int = 8081,
    log_level: str = "info",
) -> None:
    """Create and run the performance monitoring server."""
    logger.info(f"Starting performance monitoring server on {host}:{port}")

    uvicorn.run(
        performance_api.app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run performance monitoring server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    create_performance_monitoring_server(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
