"""Production-Ready Optimized Agent Implementation.

This module demonstrates a production-ready agent that incorporates all
the optimizations and best practices from the OptimizedBaseAgentService.
"""

import asyncio
from collections.abc import Callable
import contextlib
import logging
import os
from typing import Any

from gterminal.agents.example_optimized_agent import CodeAnalysisExecutor
from gterminal.agents.example_optimized_agent import DocumentationExecutor
from gterminal.agents.optimized_base_agent import JobConfiguration
from gterminal.agents.optimized_base_agent import JobExecutor
from gterminal.agents.optimized_base_agent import OptimizedBaseAgentService
from gterminal.agents.optimized_base_agent import Priority
from gterminal.performance.metrics import AgentMetricsIntegration


class ProductionCodeAnalysisExecutor(CodeAnalysisExecutor):
    """Production-ready code analysis executor with enhanced features."""

    async def execute(
        self,
        job_id: str,
        parameters: dict[str, Any],
        config: JobConfiguration,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Execute with production-grade error handling and monitoring."""
        try:
            # Enhanced parameter validation
            code_content = parameters.get("code", "").strip()
            if not code_content:
                msg = "Non-empty code content is required for analysis"
                raise ValueError(msg)

            if len(code_content) > 1_000_000:  # 1MB limit
                msg = "Code content too large (max 1MB)"
                raise ValueError(msg)

            analysis_type = parameters.get("analysis_type", "basic")
            if analysis_type not in ["basic", "comprehensive", "security", "performance"]:
                msg = f"Invalid analysis type: {analysis_type}"
                raise ValueError(msg)

            # Call parent implementation with enhanced error context
            result = await super().execute(job_id, parameters, config, progress_callback)

            # Add production metadata
            result.update(
                {
                    "execution_metadata": {
                        "executor_version": "1.0.0",
                        "max_retries_used": getattr(config, "max_retries", 0),
                        "caching_enabled": getattr(config, "enable_caching", False),
                        "analysis_depth": analysis_type,
                    },
                    "quality_metrics": {
                        "code_size_bytes": len(code_content.encode("utf-8")),
                        "processing_time_ms": result.get("duration_ms", 0),
                        "issues_found": (
                            len(result.get("syntax_issues", []))
                            + len(result.get("style_issues", []))
                        ),
                    },
                },
            )

            return result

        except Exception as e:
            # Enhanced error reporting for production
            error_context = {
                "job_id": job_id,
                "analysis_type": parameters.get("analysis_type", "unknown"),
                "code_length": len(parameters.get("code", "")),
                "config": config.__dict__ if hasattr(config, "__dict__") else str(config),
            }

            logging.exception(f"Code analysis execution failed: {e}", extra=error_context)
            raise


class ProductionOptimizedAgent(OptimizedBaseAgentService):
    """Production-ready optimized agent with comprehensive monitoring,
    security, and reliability features.
    """

    def __init__(
        self,
        agent_name: str = "production-optimized-agent",
        environment: str = "production",
        **kwargs,
    ) -> None:
        # Production-optimized defaults
        production_defaults = {
            "max_concurrent_jobs": int(os.getenv("MAX_CONCURRENT_JOBS", "20")),
            "enable_resource_monitoring": True,
            "cache_config": {
                "enable_l2_cache": True,
                "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
                "default_ttl": 3600,  # 1 hour
            },
            "connection_pool_config": {
                "total_connections": 100,
                "connections_per_host": 30,
                "keepalive_timeout": 30,
            },
        }

        # Merge with provided kwargs
        for key, value in production_defaults.items():
            kwargs.setdefault(key, value)

        super().__init__(agent_name, **kwargs)

        self.environment = environment

        # Production components
        self._metrics_integration: AgentMetricsIntegration | None = None
        self._health_check_task: asyncio.Task | None = None

        # Enhanced job executors
        self._executors = {
            "code_analysis": ProductionCodeAnalysisExecutor(),
            "documentation": DocumentationExecutor(),
        }

        # Security and compliance
        self._request_rate_limiter = asyncio.Semaphore(50)  # 50 requests per second
        self._sensitive_data_patterns = [
            r'password\s*=\s*[\'"][^\'"]+[\'"]',
            r'api_key\s*=\s*[\'"][^\'"]+[\'"]',
            r'secret\s*=\s*[\'"][^\'"]+[\'"]',
        ]

    async def startup(self) -> None:
        """Enhanced startup with production monitoring."""
        await super().startup()

        # Initialize metrics integration
        self._metrics_integration = AgentMetricsIntegration(self.agent_name)

        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        # Log startup
        self.logger.info(
            f"Production agent '{self.agent_name}' started in {self.environment} environment",
            extra={
                "environment": self.environment,
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "resource_monitoring": self.enable_resource_monitoring,
            },
        )

    async def shutdown(self) -> None:
        """Enhanced shutdown with proper cleanup."""
        self.logger.info(f"Shutting down production agent '{self.agent_name}'...")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        # Close metrics
        if self._metrics_integration:
            await self._metrics_integration.close()

        # Call parent shutdown
        await super().shutdown()

        self.logger.info("Production agent shutdown complete")

    async def _health_check_loop(self) -> None:
        """Background health monitoring for production."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Get comprehensive stats
                stats = await self.get_comprehensive_stats()

                # Health checks
                memory_usage = stats.get("resource_usage", {}).get("memory_mb", 0)
                cpu_usage = stats.get("resource_usage", {}).get("cpu_percent", 0)
                cache_hit_rate = stats.get("cache_stats", {}).get("overall", {}).get("hit_rate", 0)

                # Alert thresholds
                if memory_usage > 2048:  # 2GB
                    self.logger.warning(f"High memory usage: {memory_usage:.1f}MB")

                if cpu_usage > 80:
                    self.logger.warning(f"High CPU usage: {cpu_usage:.1f}%")

                if cache_hit_rate < 0.5:  # 50%
                    self.logger.warning(f"Low cache hit rate: {cache_hit_rate:.1%}")

                # Record metrics
                if self._metrics_integration:
                    await self._metrics_integration.record_resource_usage(memory_usage, cpu_usage)

                # Log health summary
                self.logger.debug(
                    f"Health check: Memory={memory_usage:.1f}MB, CPU={cpu_usage:.1f}%, Cache={cache_hit_rate:.1%}",
                    extra={"health_check": True},
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Health check error: {e}")

    def _get_job_executor(self, job_type: str) -> JobExecutor:
        """Get job executor with enhanced error handling."""
        if job_type not in self._executors:
            self.logger.error(f"Unknown job type requested: {job_type}")
            msg = f"Unknown job type: {job_type}. Available: {list(self._executors.keys())}"
            raise ValueError(msg)

        return self._executors[job_type]

    async def create_job_with_validation(
        self,
        job_type: str,
        parameters: dict[str, Any],
        config: JobConfiguration | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Create job with production-grade validation."""
        # Rate limiting
        async with self._request_rate_limiter:
            # Security validation
            await self._validate_security(parameters)

            # Parameter validation
            validated_params = await self._validate_parameters(job_type, parameters)

            # Create job with metrics
            job_id = self.create_job(job_type, validated_params, config, correlation_id)

            # Record metrics
            if self._metrics_integration:
                await self._metrics_integration.on_job_start(job_id, job_type)

            return job_id

    async def _validate_security(self, parameters: dict[str, Any]) -> None:
        """Validate parameters for security issues."""
        import re

        # Check for sensitive data patterns
        for key, value in parameters.items():
            if isinstance(value, str):
                for pattern in self._sensitive_data_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        self.logger.warning(
                            f"Potential sensitive data detected in parameter '{key}'",
                            extra={"security_alert": True},
                        )
                        # In production, you might want to reject the request
                        # raise ValueError(f"Sensitive data detected in parameter '{key}'")

    async def _validate_parameters(
        self, job_type: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and sanitize job parameters."""
        validated = parameters.copy()

        if job_type == "code_analysis":
            # Validate code analysis parameters
            if "code" not in validated:
                msg = "Missing required parameter: code"
                raise ValueError(msg)

            code = validated["code"]
            if not isinstance(code, str):
                msg = "Parameter 'code' must be a string"
                raise ValueError(msg)

            if len(code) > 1_000_000:  # 1MB limit
                msg = "Code content too large (max 1MB)"
                raise ValueError(msg)

            # Sanitize analysis type
            analysis_type = validated.get("analysis_type", "basic")
            if analysis_type not in ["basic", "comprehensive", "security", "performance"]:
                validated["analysis_type"] = "basic"
                self.logger.warning(f"Invalid analysis_type '{analysis_type}', using 'basic'")

        elif job_type == "documentation":
            # Validate documentation parameters
            if "code" not in validated:
                msg = "Missing required parameter: code"
                raise ValueError(msg)

            # Sanitize format
            doc_format = validated.get("format", "markdown")
            if doc_format not in ["markdown", "html", "plain"]:
                validated["format"] = "markdown"
                self.logger.warning(f"Invalid format '{doc_format}', using 'markdown'")

        return validated

    async def execute_job_with_monitoring(
        self,
        job_id: str,
        wait_for_completion: bool = False,
    ) -> dict[str, Any]:
        """Execute job with comprehensive monitoring."""
        job = self.get_job(job_id)
        if not job:
            return self.create_error_response(f"Job {job_id} not found")

        try:
            # Execute with parent method
            result = await self.execute_job_async(job_id, wait_for_completion)

            # Record success metrics
            if self._metrics_integration:
                await self._metrics_integration.on_job_complete(
                    job_id, job.job_type, result.get("status") == "success"
                )

            return result

        except Exception as e:
            # Record failure metrics
            if self._metrics_integration:
                await self._metrics_integration.on_job_complete(job_id, job.job_type, False)

            self.logger.exception(
                f"Job execution failed: {e}",
                extra={
                    "job_id": job_id,
                    "job_type": job.job_type,
                    "error_type": type(e).__name__,
                },
            )
            raise

    async def get_production_health_report(self) -> dict[str, Any]:
        """Get comprehensive production health report."""
        base_stats = await self.get_comprehensive_stats()

        # Add production-specific health indicators
        health_indicators = {
            "overall_health": "HEALTHY",
            "alerts": [],
            "recommendations": [],
        }

        # Analyze metrics for health indicators
        memory_usage = base_stats.get("resource_usage", {}).get("memory_mb", 0)
        cpu_usage = base_stats.get("resource_usage", {}).get("cpu_percent", 0)
        cache_hit_rate = base_stats.get("cache_stats", {}).get("overall", {}).get("hit_rate", 0)
        error_rate = base_stats.get("performance_metrics", {}).get("failed_jobs", 0) / max(
            base_stats.get("performance_metrics", {}).get("total_jobs", 1),
            1,
        )

        # Health assessment
        if memory_usage > 2048:
            health_indicators["alerts"].append(f"High memory usage: {memory_usage:.1f}MB")
            health_indicators["overall_health"] = "WARNING"

        if cpu_usage > 80:
            health_indicators["alerts"].append(f"High CPU usage: {cpu_usage:.1f}%")
            health_indicators["overall_health"] = "WARNING"

        if cache_hit_rate < 0.3:
            health_indicators["alerts"].append(f"Low cache hit rate: {cache_hit_rate:.1%}")
            health_indicators["recommendations"].append("Consider optimizing cache strategy")

        if error_rate > 0.05:  # 5% error rate
            health_indicators["alerts"].append(f"High error rate: {error_rate:.1%}")
            health_indicators["overall_health"] = "CRITICAL"

        # Add recommendations
        if len(health_indicators["alerts"]) == 0:
            health_indicators["recommendations"].append("System is operating normally")

        if cache_hit_rate > 0.8:
            health_indicators["recommendations"].append("Excellent cache performance")

        return {
            **base_stats,
            "health_indicators": health_indicators,
            "environment": self.environment,
            "agent_version": "1.0.0",
        }

    async def bulk_analyze_repository(
        self,
        repository_files: list[dict[str, str]],
        analysis_type: str = "comprehensive",
    ) -> dict[str, Any]:
        """Analyze an entire repository with optimized batch processing."""
        start_time = asyncio.get_running_loop().time()

        self.logger.info(
            f"Starting bulk repository analysis of {len(repository_files)} files",
            extra={
                "file_count": len(repository_files),
                "analysis_type": analysis_type,
            },
        )

        # Process files in optimized batches
        results: list[Any] = []
        failed_files: list[Any] = []

        async def process_file(file_info: dict[str, str]) -> dict[str, Any]:
            """Process a single file."""
            try:
                job_id = await self.create_job_with_validation(
                    "code_analysis",
                    {
                        "code": file_info["content"],
                        "analysis_type": analysis_type,
                        "filename": file_info.get("filename", "unknown"),
                    },
                    config=JobConfiguration(
                        priority=Priority.NORMAL,
                        enable_caching=True,
                        cache_ttl_seconds=3600,
                    ),
                )

                result = await self.execute_job_with_monitoring(job_id, wait_for_completion=True)

                return {
                    "filename": file_info.get("filename", "unknown"),
                    "status": "success",
                    "analysis": result.get("result", {}),
                }

            except Exception as e:
                self.logger.exception(
                    f"Failed to analyze file {file_info.get('filename', 'unknown')}: {e}"
                )
                return {
                    "filename": file_info.get("filename", "unknown"),
                    "status": "error",
                    "error": str(e),
                }

        # Use optimized batch processing
        async for batch_results in self.batch_process(
            repository_files,
            process_file,
            batch_size=10,
            max_concurrency=5,
        ):
            for result in batch_results:
                if result["status"] == "success":
                    results.append(result)
                else:
                    failed_files.append(result)

        # Generate repository summary
        total_issues = 0
        complexity_total = 0
        files_analyzed = len(results)

        for result in results:
            analysis = result.get("analysis", {})
            if isinstance(analysis, dict) and "syntax_issues" in analysis:
                total_issues += len(analysis.get("syntax_issues", []))
                total_issues += len(analysis.get("style_issues", []))

                complexity_metrics = analysis.get("complexity_metrics", {})
                if isinstance(complexity_metrics, dict):
                    complexity_total += complexity_metrics.get("cyclomatic_complexity", 0)

        execution_time = asyncio.get_running_loop().time() - start_time

        summary = {
            "repository_analysis": {
                "total_files": len(repository_files),
                "files_analyzed": files_analyzed,
                "files_failed": len(failed_files),
                "total_issues_found": total_issues,
                "average_complexity": complexity_total / max(files_analyzed, 1),
                "execution_time_seconds": execution_time,
                "analysis_type": analysis_type,
            },
            "file_results": results,
            "failed_files": failed_files,
            "performance_metrics": {
                "files_per_second": len(repository_files) / max(execution_time, 0.001),
                "success_rate": files_analyzed / len(repository_files) * 100,
            },
        }

        self.logger.info(
            f"Repository analysis completed: {files_analyzed}/{len(repository_files)} files, "
            f"{total_issues} issues found in {execution_time:.2f}s",
            extra={"repository_analysis_summary": True},
        )

        return summary


# Example production usage
async def production_example() -> None:
    """Example of using the production-ready optimized agent."""
    # Configure production logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(correlation_id)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("agent.log"),
        ],
    )

    # Example repository files
    sample_repository = [
        {
            "filename": "main.py",
            "content": '''
def main() -> None:
    """Main application entry point."""
    print("Hello, World!")
    config = load_config()
    app = create_app(config)
    app.run()

def load_config() -> None:
    return {"debug": True, "port": 8000}

def create_app(config) -> None:
    class App:
        def __init__(self, config) -> None:
            self.config = config

        def run(self) -> None:
            print(f"Running on port {self.config['port']}")

    return App(config)

if __name__ == "__main__":
    main()
''',
        },
        {
            "filename": "utils.py",
            "content": '''
import json
import os

def read_json_file(filepath) -> None:
    """Read and parse JSON file."""
    if not Path(filepath).exists():
        return None

    with open(filepath, 'r') as f:
        return json.load(f)

def write_json_file(filepath, data) -> None:
    """Write data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

class ConfigManager:
    def __init__(self, config_path) -> None:
        self.config_path = config_path
        self._config = None

    def load(self) -> None:
        self._config = read_json_file(self.config_path)
        return self._config

    def save(self) -> None:
        if self._config:
            write_json_file(self.config_path, self._config)
''',
        },
    ]

    # Create and use production agent
    async with ProductionOptimizedAgent(
        environment="production",
        max_concurrent_jobs=10,
    ) as agent:
        # Single file analysis
        job_id = await agent.create_job_with_validation(
            "code_analysis",
            {
                "code": sample_repository[0]["content"],
                "analysis_type": "comprehensive",
            },
        )

        result = await agent.execute_job_with_monitoring(job_id, wait_for_completion=True)
        "Success" if result["status"] == "success" else "Failed"

        # Bulk repository analysis
        await agent.bulk_analyze_repository(sample_repository, analysis_type="comprehensive")

        # Health report
        health_report = await agent.get_production_health_report()

        if health_report["health_indicators"]["alerts"]:
            for _alert in health_report["health_indicators"]["alerts"]:
                pass

        if health_report["health_indicators"]["recommendations"]:
            for _rec in health_report["health_indicators"]["recommendations"]:
                pass


if __name__ == "__main__":
    asyncio.run(production_example())
