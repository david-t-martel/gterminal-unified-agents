#!/usr/bin/env python3
"""Gemini Rust Integration Module.

This module provides integration between the existing Python Gemini clients
and the new high-performance RustGeminiClient, allowing seamless fallback
and performance optimization.

Key Features:
- Automatic fallback from Rust to Python clients
- Performance monitoring and comparison
- Configuration-driven optimization selection
- Backward compatibility with existing code
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import time
from typing import Any

try:
    from fullstack_agent_rust import RustGeminiClient

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustGeminiClient = None

# Import existing Gemini client
from gterminal.gemini_agents.client import GeminiClient

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    use_rust_client: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_requests: int = 50
    connection_pool_size: int = 100
    enable_compression: bool = True
    retry_attempts: int = 3
    timeout_seconds: int = 30
    enable_metrics: bool = True


@dataclass
class RequestMetrics:
    """Metrics for tracking request performance."""

    request_id: str
    client_type: str  # 'rust' or 'python'
    start_time: float
    end_time: float
    success: bool
    cached: bool
    response_size: int
    error: str | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class HybridGeminiClient:
    """Hybrid Gemini client that intelligently selects between Rust and Python implementations
    based on performance characteristics and availability.
    """

    def __init__(
        self,
        auth_config: dict[str, Any],
        performance_config: PerformanceConfig | None = None,
    ) -> None:
        self.auth_config = auth_config
        self.performance_config = performance_config or PerformanceConfig()
        self.metrics: list[RequestMetrics] = []

        # Initialize clients
        self.rust_client: RustGeminiClient | None = None
        self.python_client: GeminiClient | None = None

        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize available clients based on configuration."""
        # Initialize Rust client if available and enabled
        if RUST_AVAILABLE and self.performance_config.use_rust_client:
            try:
                rust_auth_config = {
                    "project_id": self.auth_config.get("project_id", ""),
                    "location": self.auth_config.get("location", "us-central1"),
                    "api_key": self.auth_config.get("api_key"),
                    "access_token": self.auth_config.get("access_token"),
                    "use_vertex_ai": self.auth_config.get("use_vertex_ai", False),
                }

                self.rust_client = RustGeminiClient(
                    auth_config=json.dumps(rust_auth_config),
                    max_concurrent=self.performance_config.max_concurrent_requests,
                )
                logger.info("Rust Gemini client initialized successfully")

            except Exception as e:
                logger.warning(f"Failed to initialize Rust client: {e}")
                self.rust_client = None

        # Initialize Python client as fallback
        try:
            profile = "business" if self.auth_config.get("use_vertex_ai") else "personal"
            self.python_client = GeminiClient(profile=profile)
            logger.info("Python Gemini client initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize Python client: {e}")
            self.python_client = None

        if not self.rust_client and not self.python_client:
            msg = "No Gemini clients could be initialized"
            raise RuntimeError(msg)

    async def generate_text(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        use_cache: bool = True,
        **kwargs,
    ) -> str:
        """Generate text using the most appropriate client.

        Args:
            prompt: Input prompt
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use caching
            **kwargs: Additional parameters

        Returns:
            Generated text response

        """
        request_id = f"req_{int(time.time() * 1000)}"
        start_time = time.perf_counter()

        # Try Rust client first if available
        if self.rust_client:
            try:
                config = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "cache_ttl": (self.performance_config.cache_ttl_seconds if use_cache else None),
                    "retry_attempts": self.performance_config.retry_attempts,
                    "timeout": self.performance_config.timeout_seconds,
                }

                response = await self.rust_client.generate_text(
                    prompt=prompt, config=json.dumps(config)
                )

                end_time = time.perf_counter()

                # Record metrics
                if self.performance_config.enable_metrics:
                    metrics = RequestMetrics(
                        request_id=request_id,
                        client_type="rust",
                        start_time=start_time,
                        end_time=end_time,
                        success=True,
                        cached=False,  # Would need to check cache stats
                        response_size=len(response.encode("utf-8")),
                    )
                    self.metrics.append(metrics)

                logger.debug(f"Rust client generated response in {end_time - start_time:.3f}s")
                return response

            except Exception as e:
                logger.warning(f"Rust client failed: {e}, falling back to Python client")

        # Fallback to Python client
        if not self.python_client:
            msg = "No available Gemini clients"
            raise RuntimeError(msg)

        try:
            await self.python_client.initialize()

            response = await self.python_client.generate_text(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            end_time = time.perf_counter()

            # Record metrics
            if self.performance_config.enable_metrics:
                metrics = RequestMetrics(
                    request_id=request_id,
                    client_type="python",
                    start_time=start_time,
                    end_time=end_time,
                    success=True,
                    cached=False,
                    response_size=len(response.encode("utf-8")),
                )
                self.metrics.append(metrics)

            logger.debug(f"Python client generated response in {end_time - start_time:.3f}s")
            return response

        except Exception as e:
            end_time = time.perf_counter()

            # Record failed metrics
            if self.performance_config.enable_metrics:
                metrics = RequestMetrics(
                    request_id=request_id,
                    client_type="python",
                    start_time=start_time,
                    end_time=end_time,
                    success=False,
                    cached=False,
                    response_size=0,
                    error=str(e),
                )
                self.metrics.append(metrics)

            raise

    async def generate_batch(
        self,
        prompts: list[str],
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        **kwargs,
    ) -> list[str]:
        """Generate responses for multiple prompts efficiently.

        Args:
            prompts: List of input prompts
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            List of generated responses

        """
        # Use Rust batch processing if available
        if self.rust_client:
            try:
                config = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "cache_ttl": self.performance_config.cache_ttl_seconds,
                    "retry_attempts": self.performance_config.retry_attempts,
                }

                responses = await self.rust_client.batch_request(
                    requests=prompts, config=json.dumps(config)
                )

                logger.info(f"Rust client processed {len(prompts)} prompts in batch")
                return responses

            except Exception as e:
                logger.warning(f"Rust batch processing failed: {e}, falling back to sequential")

        # Fallback to sequential processing with Python client
        responses: list[Any] = []
        for prompt in prompts:
            response = await self.generate_text(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            responses.append(response)

        return responses

    async def generate_stream(
        self, prompt: str, model: str = "gemini-2.5-flash", **kwargs
    ) -> list[str]:
        """Generate streaming response.

        Args:
            prompt: Input prompt
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            List of response chunks

        """
        # Use Rust streaming if available
        if self.rust_client:
            try:
                config = {
                    "model": model,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 8192),
                }

                return await self.rust_client.generate_stream(
                    prompt=prompt, config=json.dumps(config)
                )

            except Exception as e:
                logger.warning(f"Rust streaming failed: {e}, falling back to Python")

        # Fallback to Python client (simulate streaming)
        if self.python_client:
            response = await self.generate_text(prompt=prompt, model=model, **kwargs)

            # Split into chunks for streaming simulation
            chunk_size = 50
            return [response[i : i + chunk_size] for i in range(0, len(response), chunk_size)]

        msg = "No streaming client available"
        raise RuntimeError(msg)

    async def warmup_connections(self) -> None:
        """Warm up client connections for optimal performance."""
        if self.rust_client:
            try:
                await self.rust_client.warmup_connections()
                logger.info("Rust client connections warmed up")
            except Exception as e:
                logger.warning(f"Rust client warmup failed: {e}")

        if self.python_client:
            try:
                await self.python_client.initialize()
                logger.info("Python client initialized and ready")
            except Exception as e:
                logger.warning(f"Python client initialization failed: {e}")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.performance_config.enable_metrics or not self.metrics:
            return {"error": "Metrics not enabled or no data available"}

        rust_metrics = [m for m in self.metrics if m.client_type == "rust"]
        python_metrics = [m for m in self.metrics if m.client_type == "python"]

        def calculate_stats(metrics_list: list[RequestMetrics]) -> dict[str, Any]:
            if not metrics_list:
                return {}

            durations = [m.duration for m in metrics_list if m.success]
            success_rate = len([m for m in metrics_list if m.success]) / len(metrics_list)

            return {
                "total_requests": len(metrics_list),
                "successful_requests": len([m for m in metrics_list if m.success]),
                "success_rate": success_rate,
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "total_response_size": sum(m.response_size for m in metrics_list),
            }

        rust_stats = calculate_stats(rust_metrics)
        python_stats = calculate_stats(python_metrics)

        # Calculate performance improvements
        improvement_ratio = 0
        if rust_stats.get("avg_duration", 0) > 0 and python_stats.get("avg_duration", 0) > 0:
            improvement_ratio = python_stats["avg_duration"] / rust_stats["avg_duration"]

        return {
            "rust_client": rust_stats,
            "python_client": python_stats,
            "performance_improvement": (
                f"{improvement_ratio:.2f}x" if improvement_ratio > 0 else "N/A"
            ),
            "rust_available": self.rust_client is not None,
            "total_requests": len(self.metrics),
        }

    def clear_metrics(self) -> None:
        """Clear performance metrics."""
        self.metrics.clear()

    async def close(self) -> None:
        """Close clients and cleanup resources."""
        if self.python_client:
            try:
                await self.python_client.close()
            except Exception as e:
                logger.warning(f"Error closing Python client: {e}")

        # Rust client cleanup (if needed)
        if self.rust_client:
            try:
                # Clear cache and reset connections
                self.rust_client.clear_cache()
            except Exception as e:
                logger.warning(f"Error cleaning up Rust client: {e}")

    @asynccontextmanager
    async def performance_context(self):
        """Context manager for performance monitoring."""
        await self.warmup_connections()
        try:
            yield self
        finally:
            metrics = self.get_performance_metrics()
            logger.info(f"Performance metrics: {metrics}")
            await self.close()


# Convenience factory function
def create_optimized_gemini_client(
    auth_config: dict[str, Any],
    performance_config: PerformanceConfig | None = None,
) -> HybridGeminiClient:
    """Create an optimized Gemini client with intelligent Rust/Python selection.

    Args:
        auth_config: Authentication configuration
        performance_config: Performance optimization settings

    Returns:
        Configured HybridGeminiClient instance

    """
    return HybridGeminiClient(auth_config, performance_config)


# Example usage
async def example_usage() -> None:
    """Example of using the hybrid Gemini client."""
    # Configuration
    auth_config = {
        "project_id": "your-project-id",
        "location": "us-central1",
        "api_key": "your-api-key",
        "use_vertex_ai": False,
    }

    performance_config = PerformanceConfig(
        use_rust_client=True,
        enable_caching=True,
        cache_ttl_seconds=1800,
        max_concurrent_requests=25,
        enable_metrics=True,
    )

    # Create optimized client
    async with create_optimized_gemini_client(
        auth_config, performance_config
    ).performance_context() as client:
        # Single request
        await client.generate_text(
            prompt="Explain the benefits of Rust for AI applications",
            model="gemini-2.5-flash",
            temperature=0.7,
        )

        # Batch processing
        prompts = [
            "What is machine learning?",
            "How does HTTP/2 work?",
            "Explain connection pooling",
        ]

        await client.generate_batch(prompts)

        # Streaming
        await client.generate_stream(
            prompt="Write a short story about AI", model="gemini-2.5-flash"
        )

        # Performance metrics
        client.get_performance_metrics()


if __name__ == "__main__":
    asyncio.run(example_usage())
