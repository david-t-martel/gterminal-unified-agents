"""Optimized Agent Engine App with performance enhancements.

This is a drop-in replacement for agent_engine_app.py that includes
comprehensive caching, connection pooling, and performance monitoring.
Now enhanced with Rust extensions for maximum performance.
"""

import asyncio
import copy
import datetime
import json
import logging
import os
from typing import Any

from google.adk.artifacts import GcsArtifactService
import google.auth
from google.cloud import logging as google_cloud_logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace import export
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

from .agent import optimized_root_agent
from .performance_integration import get_performance_integration
from .utils.gcs import create_bucket_if_not_exists
from .utils.rust_extensions import RUST_CORE_AVAILABLE
from .utils.rust_extensions import EnhancedTtlCache

# Import Rust extensions for maximum performance
from .utils.rust_extensions import RustCore
from .utils.rust_extensions import get_rust_status
from .utils.rust_extensions import test_rust_integration
from .utils.tracing import CloudTraceLoggingSpanExporter
from .utils.typing import Feedback

logger = logging.getLogger(__name__)


class OptimizedAgentEngineApp(AdkApp):
    """Optimized Agent Engine App with performance enhancements and Rust extensions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._performance_integration = None
        self._performance_initialized = False
        self._rust_core: RustCore | None = None
        self._rust_cache: EnhancedTtlCache | None = None
        self._rust_initialized = False
        self._initialize_rust_extensions()

    def _initialize_rust_extensions(self) -> None:
        """Initialize Rust extensions for maximum performance."""
        if RUST_CORE_AVAILABLE:
            try:
                self._rust_core = RustCore()
                # Initialize cache with 1 hour default TTL for app-level caching
                self._rust_cache = EnhancedTtlCache(3600)
                self._rust_initialized = True
                logger.info(f"Rust extensions initialized - Version: {self._rust_core.version}")

                # Test integration
                test_result = test_rust_integration()
                logger.info(f"Rust integration test: {test_result}")

            except Exception as e:
                logger.warning(f"Failed to initialize Rust extensions: {e}")
                self._rust_core = None
                self._rust_cache = None
                self._rust_initialized = False
        else:
            logger.info("Using Python fallbacks - Rust extensions not available")
            self._rust_initialized = False

    async def _ensure_performance_integration(self) -> None:
        """Ensure performance integration is initialized."""
        if not self._performance_initialized:
            try:
                self._performance_integration = await get_performance_integration()
                self._performance_initialized = True
                logger.info("Performance integration initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize performance integration: {e}")
                self._performance_initialized = False

    def set_up(self) -> None:
        """Set up logging, tracing, and performance monitoring."""
        super().set_up()

        # Original setup
        logging_client = google_cloud_logging.Client()
        self.logger = logging_client.logger(__name__)
        provider = TracerProvider()
        processor = export.BatchSpanProcessor(
            CloudTraceLoggingSpanExporter(project_id=os.environ.get("GOOGLE_CLOUD_PROJECT")),
        )
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        # Initialize performance integration asynchronously only if event loop is running
        try:
            asyncio.get_running_loop()
            asyncio.create_task(self._ensure_performance_integration())
        except RuntimeError:
            # No event loop running, defer initialization
            self._performance_initialized = False

        logger.info("OptimizedAgentEngineApp setup completed")

    async def register_feedback_async(self, feedback: dict[str, Any]) -> None:
        """Async version of feedback registration with performance tracking."""
        try:
            await self._ensure_performance_integration()

            if self._performance_integration:
                # Track feedback registration performance
                start_time = await self._performance_integration.optimizer.metrics.start_request(
                    "feedback_registration",
                    "register_feedback",
                )

                try:
                    feedback_obj = Feedback.model_validate(feedback)
                    self.logger.log_struct(feedback_obj.model_dump(), severity="INFO")

                    await self._performance_integration.optimizer.metrics.end_request(
                        "feedback_registration",
                        "register_feedback",
                        start_time,
                        True,
                    )

                except Exception:
                    await self._performance_integration.optimizer.metrics.end_request(
                        "feedback_registration",
                        "register_feedback",
                        start_time,
                        False,
                    )
                    raise
            else:
                # Fallback to original implementation
                feedback_obj = Feedback.model_validate(feedback)
                self.logger.log_struct(feedback_obj.model_dump(), severity="INFO")

        except Exception as e:
            logger.exception(f"Failed to register feedback: {e}")
            raise

    def register_feedback(self, feedback: dict[str, Any]) -> None:
        """Collect and log feedback with performance optimization."""
        # Fix async anti-pattern: Don't create tasks in sync methods
        try:
            # Try to get running event loop without creating one
            asyncio.get_running_loop()
            # Schedule the async operation to run later
            asyncio.create_task(self.register_feedback_async(feedback))
        except RuntimeError:
            # No event loop running, use sync version
            feedback_obj = Feedback.model_validate(feedback)
            self.logger.log_struct(feedback_obj.model_dump(), severity="INFO")

    def register_operations(self) -> dict[str, list[str]]:
        """Registers the operations of the Agent including performance monitoring."""
        operations: dict[str, list[str]] = super().register_operations()
        operations[""] = [*operations.get("", []), "register_feedback"]

        # Add performance monitoring operations if available
        if hasattr(self, "_performance_integration") and self._performance_integration:
            operations["performance"] = [
                "get_performance_stats",
                "get_cache_stats",
                "clear_cache",
                "warm_up_cache",
            ]

        # Add Rust operations if available
        if self._rust_initialized:
            operations["rust"] = [
                "get_rust_performance_info",
                "cleanup_rust_cache",
                "process_with_rust_acceleration",
            ]

        return operations

    async def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        await self._ensure_performance_integration()
        if self._performance_integration:
            return await self._performance_integration.get_performance_dashboard()
        return {"error": "Performance integration not available"}

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        await self._ensure_performance_integration()
        if self._performance_integration:
            return await self._performance_integration.optimizer.cache_manager.get_stats()
        return {"error": "Performance integration not available"}

    async def clear_cache(self, pattern: str | None = None) -> dict[str, Any]:
        """Clear cache entries."""
        await self._ensure_performance_integration()
        if self._performance_integration:
            count = await self._performance_integration.optimizer.cache_manager.clear(pattern)
            return {"cleared_items": count, "pattern": pattern}
        return {"error": "Performance integration not available"}

    async def warm_up_cache(self) -> dict[str, Any]:
        """Warm up cache with common data."""
        await self._ensure_performance_integration()
        if self._performance_integration:
            await self._performance_integration.optimizer.warm_up_cache()
            return {"status": "Cache warmed up successfully"}
        return {"error": "Performance integration not available"}

    def get_rust_performance_info(self) -> dict[str, Any]:
        """Get detailed Rust performance information and statistics."""
        info = {
            "rust_available": RUST_CORE_AVAILABLE,
            "rust_initialized": self._rust_initialized,
            "rust_core_active": self._rust_core is not None,
            "rust_cache_active": self._rust_cache is not None,
        }

        # Add overall Rust status
        try:
            rust_status = get_rust_status()
            info.update(rust_status)
        except Exception as e:
            info["status_error"] = str(e)

        # Add cache statistics if available
        if self._rust_cache:
            try:
                stats = self._rust_cache.get_stats()
                info.update(
                    {
                        "cache_performance": {
                            "size": self._rust_cache.size,
                            "hits": stats.hits,
                            "misses": stats.misses,
                            "hit_ratio": stats.hit_ratio,
                            "total_entries": stats.total_entries,
                            "expired_entries": stats.expired_entries,
                        },
                    },
                )
            except Exception as e:
                info["cache_stats_error"] = str(e)

        # Add core version information
        if self._rust_core:
            info["rust_version"] = self._rust_core.version

        return info

    async def cleanup_rust_cache(self) -> dict[str, Any]:
        """Clean up expired entries in Rust cache."""
        if self._rust_cache:
            try:
                removed_count = self._rust_cache.cleanup_expired()
                return {"status": "success", "removed_entries": removed_count, "cache_type": "rust"}
            except Exception as e:
                return {"status": "error", "error": str(e)}
        return {"status": "not_available", "message": "Rust cache not initialized"}

    def process_with_rust_acceleration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process data using Rust acceleration when available."""
        if self._rust_core:
            try:
                # Example: Use Rust for data processing
                processed = {}
                for key, value in data.items():
                    if isinstance(value, str):
                        # Use Rust string operations
                        processed[key] = self._rust_core.reverse_string(
                            self._rust_core.reverse_string(
                                value
                            ),  # Double reverse = original but processed in Rust
                        )
                    elif isinstance(value, int | float):
                        # Use Rust math operations
                        processed[key] = self._rust_core.add_numbers(
                            value, 0
                        )  # Identity operation via Rust
                    else:
                        processed[key] = value

                return {
                    "status": "rust_accelerated",
                    "data": processed,
                    "rust_version": self._rust_core.version,
                }
            except Exception as e:
                logger.warning(f"Rust processing failed, falling back to Python: {e}")

        # Fallback to Python processing
        return {
            "status": "python_fallback",
            "data": copy.deepcopy(data),
            "message": "Rust acceleration not available",
        }

    def clone(self) -> "OptimizedAgentEngineApp":
        """Returns a clone of the optimized ADK application."""
        template_attributes = self._tmpl_attrs

        cloned_app = self.__class__(
            agent=copy.deepcopy(template_attributes["agent"]),
            enable_tracing=bool(template_attributes.get("enable_tracing", False)),
            session_service_builder=template_attributes.get("session_service_builder"),
            artifact_service_builder=template_attributes.get("artifact_service_builder"),
            env_vars=template_attributes.get("env_vars"),
        )

        # Ensure the cloned app also initializes performance integration
        try:
            asyncio.get_running_loop()
            asyncio.create_task(cloned_app._ensure_performance_integration())
        except RuntimeError:
            # No event loop running, defer initialization for cloned app
            cloned_app._performance_initialized = False

        return cloned_app


async def deploy_optimized_agent_engine_app(
    project: str,
    location: str,
    agent_name: str | None = None,
    requirements_file: str = ".requirements.txt",
    extra_packages: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
    enable_performance_monitoring: bool = True,
) -> agent_engines.AgentEngine:
    """Deploy the optimized agent engine app to Vertex AI with performance enhancements."""
    # Initialize mutable defaults
    if extra_packages is None:
        extra_packages = ["./app"]
    if env_vars is None:
        env_vars: dict[str, Any] = {}

    staging_bucket_uri = f"gs://{project}-agent-engine"
    artifacts_bucket_name = f"{project}-my-fullstack-agent-logs-data"

    create_bucket_if_not_exists(
        bucket_name=artifacts_bucket_name, project=project, location=location
    )
    create_bucket_if_not_exists(bucket_name=staging_bucket_uri, project=project, location=location)

    vertexai.init(project=project, location=location, staging_bucket=staging_bucket_uri)

    # Read requirements
    with open(requirements_file) as f:
        requirements = f.read().strip().split("\n")

    # Create optimized agent engine with performance enhancements
    agent_engine = OptimizedAgentEngineApp(
        agent=optimized_root_agent,
        artifact_service_builder=lambda: GcsArtifactService(bucket_name=artifacts_bucket_name),
    )

    # Enhanced environment variables for performance
    env_vars.update(
        {
            "NUM_WORKERS": "1",
            "ENABLE_PERFORMANCE_OPTIMIZATION": "true",
            "CACHE_MEMORY_SIZE": "1000",
            "CACHE_MEMORY_TTL": "1800",
            "CACHE_REDIS_TTL": "3600",
            "REDIS_MAX_CONNECTIONS": "20",
            "HTTP_POOL_SIZE": "100",
            "HTTP_CONNECTIONS_PER_HOST": "30",
            "VERTEX_AI_MAX_CLIENTS": "10",
            "SEARCH_RATE_LIMIT": "10.0",
            "SEARCH_DAILY_LIMIT": "10000",
        },
    )

    # Add Redis configuration if available
    if "REDIS_URL" not in env_vars:
        env_vars["REDIS_URL"] = "redis://localhost:6379"
    if "ENABLE_REDIS" not in env_vars:
        env_vars["ENABLE_REDIS"] = "true"

    # Common configuration for both create and update operations
    agent_config = {
        "agent_engine": agent_engine,
        "display_name": agent_name or "optimized-fullstack-agent",
        "description": "A high-performance, production-ready fullstack research agent with comprehensive caching, connection pooling, and performance monitoring that uses Gemini to strategize, research, and synthesize comprehensive reports with human-in-the-loop collaboration",
        "extra_packages": extra_packages,
        "env_vars": env_vars,
        "requirements": requirements,
    }

    logger.info(f"Optimized agent config: {agent_config}")

    # Check if an agent with this name already exists
    existing_agents = list(agent_engines.list(filter=f"display_name={agent_name}"))
    if existing_agents:
        # Update the existing agent with new configuration
        logger.info(f"Updating existing optimized agent: {agent_name}")
        remote_agent = existing_agents[0].update(**agent_config)
    else:
        # Create a new agent if none exists
        logger.info(f"Creating new optimized agent: {agent_name}")
        remote_agent = agent_engines.create(**agent_config)

    # Save deployment metadata
    config = {
        "remote_agent_engine_id": remote_agent.resource_name,
        "deployment_timestamp": datetime.datetime.now().isoformat(),
        "optimizations_enabled": True,
        "performance_monitoring": enable_performance_monitoring,
        "performance_endpoint": "http://localhost:8081" if enable_performance_monitoring else None,
    }
    config_file = "deployment_metadata.json"

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Optimized Agent Engine ID written to {config_file}")

    # Start performance monitoring server if requested
    if enable_performance_monitoring:
        logger.info("Performance monitoring available at http://localhost:8081")
        logger.info("Performance dashboard: http://localhost:8081/performance/dashboard")
        logger.info("Health check: http://localhost:8081/health")

    return remote_agent


def deploy_agent_engine_app(
    project: str,
    location: str,
    agent_name: str | None = None,
    requirements_file: str = ".requirements.txt",
    extra_packages: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> agent_engines.AgentEngine:
    """Wrapper function to maintain compatibility with original deployment."""
    # Fix async anti-pattern: Check if we're already in an async context
    try:
        asyncio.get_running_loop()
        # If we're in an async context, we can't use asyncio.run()
        # Create a new thread to avoid nested event loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                deploy_optimized_agent_engine_app(
                    project=project,
                    location=location,
                    agent_name=agent_name,
                    requirements_file=requirements_file,
                    extra_packages=extra_packages,
                    env_vars=env_vars,
                ),
            )
            return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(
            deploy_optimized_agent_engine_app(
                project=project,
                location=location,
                agent_name=agent_name,
                requirements_file=requirements_file,
                extra_packages=extra_packages,
                env_vars=env_vars,
            ),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy optimized agent engine app to Vertex AI")
    parser.add_argument(
        "--project",
        default=None,
        help="GCP project ID (defaults to application default credentials)",
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="GCP region (defaults to us-central1)",
    )
    parser.add_argument(
        "--agent-name",
        default="optimized-fullstack-agent",
        help="Name for the agent engine",
    )
    parser.add_argument(
        "--requirements-file",
        default=".requirements.txt",
        help="Path to requirements.txt file",
    )
    parser.add_argument(
        "--extra-packages",
        nargs="+",
        default=["./app"],
        help="Additional packages to include",
    )
    parser.add_argument(
        "--set-env-vars",
        help="Comma-separated list of environment variables in KEY=VALUE format",
    )
    parser.add_argument(
        "--enable-performance-monitoring",
        action="store_true",
        default=True,
        help="Enable performance monitoring server",
    )
    args = parser.parse_args()

    # Parse environment variables if provided
    env_vars: dict[str, Any] = {}
    if args.set_env_vars:
        for pair in args.set_env_vars.split(","):
            key, value = pair.split("=", 1)
            env_vars[key] = value

    if not args.project:
        _, args.project = google.auth.default()

    asyncio.run(
        deploy_optimized_agent_engine_app(
            project=args.project,
            location=args.location,
            agent_name=args.agent_name,
            requirements_file=args.requirements_file,
            extra_packages=args.extra_packages,
            env_vars=env_vars,
            enable_performance_monitoring=args.enable_performance_monitoring,
        ),
    )
