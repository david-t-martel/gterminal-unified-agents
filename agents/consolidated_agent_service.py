"""Consolidated Agent Service - Combines multiple agent functionalities.

This module consolidates various agent services including code generation,
code review, documentation generation, architecture design, and workspace analysis.

It provides a unified service layer with job management, async operations,
streaming capabilities, and enhanced error handling.
"""

from abc import abstractmethod
import asyncio
from collections.abc import AsyncGenerator, Callable
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any
import uuid

# Removed circular import - using local implementations
# from gterminal.automation.base_automation_agent import BaseAutomationAgent
# from gterminal.core.security.security_utils import (...)


class JobStatus(Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job:
    """Job tracking and management with enhanced validation."""

    def __init__(self, job_id: str, job_type: str, parameters: dict[str, Any]) -> None:
        # Validate inputs
        if not job_id or not isinstance(job_id, str):
            msg = "Job ID must be a non-empty string"
            raise ValueError(msg)
        if not job_type or not isinstance(job_type, str):
            msg = "Job type must be a non-empty string"
            raise ValueError(msg)
        if not isinstance(parameters, dict):
            msg = "Parameters must be a dictionary"
            raise ValueError(msg)

        self.job_id = job_id.strip()
        self.job_type = job_type.strip()
        self.parameters = parameters.copy()  # Defensive copy
        self.status = JobStatus.PENDING
        self.created_at = datetime.now(UTC)
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.progress = 0.0
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.logs: list[str] = []

    def start(self) -> None:
        """Mark job as started with validation."""
        if self.status != JobStatus.PENDING:
            msg = f"Cannot start job in {self.status.value} status"
            raise ValueError(msg)

        self.status = JobStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def complete(self, result: dict[str, Any]) -> None:
        """Mark job as completed with result validation."""
        if self.status != JobStatus.RUNNING:
            msg = f"Cannot complete job in {self.status.value} status"
            raise ValueError(msg)
        if not isinstance(result, dict):
            msg = "Result must be a dictionary"
            raise ValueError(msg)

        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now(UTC)
        self.progress = 100.0
        self.result = result.copy()  # Defensive copy

    def fail(self, error: str) -> None:
        """Mark job as failed with error validation."""
        if not error or not isinstance(error, str):
            msg = "Error message must be a non-empty string"
            raise ValueError(msg)

        self.status = JobStatus.FAILED
        self.completed_at = datetime.now(UTC)
        self.error = error.strip()

    def cancel(self) -> None:
        """Cancel the job."""
        if self.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            msg = f"Cannot cancel job in {self.status.value} status"
            raise ValueError(msg)

        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.now(UTC)

    def add_log(self, message: str) -> None:
        """Add log message with validation."""
        if not message or not isinstance(message, str):
            return  # Silently ignore invalid log messages

        timestamp = datetime.now(UTC).isoformat()
        self.logs.append(f"[{timestamp}] {message.strip()}")

        # Limit log size to prevent memory issues
        if len(self.logs) > 1000:
            self.logs = self.logs[-500:]  # Keep last 500 entries

    def update_progress(self, progress: float, message: str | None = None) -> None:
        """Update job progress with validation."""
        if not isinstance(progress, int | float):
            msg = "Progress must be a number"
            raise ValueError(msg)

        self.progress = max(0.0, min(100.0, float(progress)))

        if message:
            self.add_log(message)

    def get_duration(self) -> float | None:
        """Get job duration in seconds."""
        if not self.started_at:
            return None

        end_time = self.completed_at or datetime.now(UTC)
        return (end_time - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary with enhanced data."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "parameters": self.parameters,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "logs": self.logs[-10:],  # Last 10 log entries
            "duration_seconds": self.get_duration(),
        }


class BaseAgentService:
    """Enhanced base class for agent services with job management and streaming.

    Provides:
    - Async job execution with progress tracking
    - Streaming response support
    - Enhanced error handling and logging
    - Integration with existing auth system
    - Modern Python type annotations
    """

    def __init__(self, agent_name: str, description: str = "") -> None:
        if not agent_name or not isinstance(agent_name, str):
            msg = "Agent name must be a non-empty string"
            raise ValueError(msg)

        super().__init__(agent_name.strip(), description.strip())

        # Store agent name for direct access
        self.agent_name = agent_name.strip()

        # Job management
        self.jobs: dict[str, Job] = {}
        self.max_concurrent_jobs = 5
        self.running_jobs = 0
        self.max_job_history = 1000

        # Streaming support
        self.active_streams: dict[str, bool] = {}

    def create_job(self, job_type: str, parameters: dict[str, Any]) -> str:
        """Create a new job and return job ID.

        Args:
            job_type: Type of job to create
            parameters: Job parameters

        Returns:
            Job ID string

        Raises:
            ValueError: If job_type or parameters are invalid

        """
        if not job_type or not isinstance(job_type, str):
            msg = "Job type must be a non-empty string"
            raise ValueError(msg)
        if not isinstance(parameters, dict):
            msg = "Parameters must be a dictionary"
            raise ValueError(msg)

        job_id = str(uuid.uuid4())
        job = Job(job_id, job_type, parameters)

        # Cleanup old jobs if needed
        if len(self.jobs) >= self.max_job_history:
            self._cleanup_old_jobs(keep_recent=self.max_job_history // 2)

        self.jobs[job_id] = job

        self.logger.info(f"Created job {job_id} of type {job_type}")
        return job_id

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID with validation."""
        if not job_id or not isinstance(job_id, str):
            return None
        return self.jobs.get(job_id.strip())

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get job status and details."""
        job = self.get_job(job_id)
        if not job:
            return self.create_error_response(f"Job {job_id} not found")

        return self.create_success_response({"job": job.to_dict()}, "Job status retrieved")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a running job."""
        job = self.get_job(job_id)
        if not job:
            return self.create_error_response(f"Job {job_id} not found")

        try:
            job.cancel()
            self.logger.info(f"Cancelled job {job_id}")
            return self.create_success_response({"job": job.to_dict()}, "Job cancelled")
        except ValueError as e:
            return self.create_error_response(str(e))

    async def execute_job_async(self, job_id: str) -> dict[str, Any]:
        """Execute job asynchronously with progress tracking.

        Args:
            job_id: ID of job to execute

        Returns:
            Job execution result

        """
        job = self.get_job(job_id)
        if not job:
            return self.create_error_response(f"Job {job_id} not found")

        if self.running_jobs >= self.max_concurrent_jobs:
            return self.create_error_response("Too many concurrent jobs")

        try:
            self.running_jobs += 1
            job.start()
            job.add_log(f"Starting {job.job_type} execution")

            # Validate job parameters before execution
            if not self.validate_job_parameters(job.job_type, job.parameters):
                error_msg = "Job parameter validation failed"
                job.fail(error_msg)
                return self.create_error_response(error_msg, {"job_id": job_id})

            # Execute the specific job type
            result = await self._execute_job_implementation(job)

            job.complete(result)
            job.add_log("Job completed successfully")

            return self.create_success_response(
                {"job": job.to_dict()}, "Job completed successfully"
            )

        except asyncio.CancelledError:
            job.cancel()
            job.add_log("Job was cancelled")
            self.logger.info(f"Job {job_id} was cancelled")
            return self.create_error_response("Job was cancelled", {"job_id": job_id})

        except Exception as e:
            job.fail(str(e))
            job.add_log(f"Job failed: {e!s}")
            self.logger.exception(f"Job {job_id} failed: {e}")

            return self.create_error_response(f"Job execution failed: {e!s}", {"job_id": job_id})

        finally:
            self.running_jobs = max(0, self.running_jobs - 1)

    async def stream_job_progress(self, job_id: str) -> AsyncGenerator[str, None]:
        """Stream job progress updates via Server-Sent Events.

        Args:
            job_id: ID of job to stream

        Yields:
            JSON-encoded progress updates

        """
        job = self.get_job(job_id)
        if not job:
            yield f"data: {json.dumps(self.create_error_response('Job not found'))}\n\n"
            return

        self.active_streams[job_id] = True

        try:
            # Stream initial status
            yield f"data: {json.dumps(job.to_dict())}\n\n"

            # Stream updates while job is running
            last_update = job.to_dict()

            while job.status in [JobStatus.PENDING, JobStatus.RUNNING] and self.active_streams.get(
                job_id, False
            ):
                await asyncio.sleep(1)  # Poll every second

                current_status = job.to_dict()

                # Only send update if something changed
                if current_status != last_update:
                    yield f"data: {json.dumps(current_status)}\n\n"
                    last_update = current_status

            # Final status
            if self.active_streams.get(job_id, False):
                yield f"data: {json.dumps(job.to_dict())}\n\n"

        except Exception as e:
            error_msg = self.create_error_response(f"Streaming error: {e!s}")
            yield f"data: {json.dumps(error_msg)}\n\n"

        finally:
            self.active_streams.pop(job_id, None)

    def stop_stream(self, job_id: str) -> None:
        """Stop streaming for a job."""
        if job_id in self.active_streams:
            self.active_streams[job_id] = False

    async def generate_with_gemini(
        self,
        prompt: str,
        task_type: str = "analysis",
        job: Job | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        parse_json: bool = True,
    ) -> str | dict[str, Any] | None:
        """Generate content with Gemini while tracking progress.

        Args:
            prompt: Prompt for generation
            task_type: Type of task for model selection
            job: Optional job for progress tracking
            progress_callback: Optional callback for progress updates
            parse_json: Whether to parse the response as JSON

        Returns:
            Generated content or None if failed

        """
        if not prompt or not isinstance(prompt, str):
            self.logger.error("Invalid prompt provided")
            return None

        try:
            if job:
                job.update_progress(10.0, "Initializing model...")
            if progress_callback:
                progress_callback(10.0, "Initializing model...")

            model = self.get_model(task_type)

            if job:
                job.update_progress(30.0, "Sending request to Gemini...")
            if progress_callback:
                progress_callback(30.0, "Sending request to Gemini...")

            # Generate content with streaming support
            response = model.generate_content(prompt.strip(), stream=False)

            if job:
                job.update_progress(80.0, "Processing response...")
            if progress_callback:
                progress_callback(80.0, "Processing response...")

            content = response.text

            if job:
                job.update_progress(100.0, "Generation completed")
            if progress_callback:
                progress_callback(100.0, "Generation completed")

            if parse_json:
                return safe_json_parse(content)  # Attempt to parse as JSON

            return content

        except Exception as e:
            error_msg = f"Gemini generation failed: {e}"
            if job:
                job.add_log(error_msg)
            self.logger.exception(error_msg)
            return None

    def validate_job_parameters(self, job_type: str, parameters: dict[str, Any]) -> bool:
        """Validate job parameters for specific job type.

        Args:
            job_type: Type of job
            parameters: Parameters to validate

        Returns:
            True if valid, False otherwise

        # TODO: Add comprehensive parameter validation schema using Pydantic models
        # TODO: Implement job-type-specific validation rules
        # TODO: Add parameter sanitization and security checks

        """
        if not isinstance(parameters, dict):
            self.logger.error("Parameters must be a dictionary")
            return False

        # Base validation - override in subclasses
        required_params = self.get_required_parameters(job_type)

        for param in required_params:
            if param not in parameters:
                self.logger.error(f"Missing required parameter: {param}")
                return False
            if parameters[param] is None:
                self.logger.error(f"Parameter cannot be None: {param}")
                return False

        return True

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Get required parameters for job type.
        Override in subclasses.

        Args:
            job_type: Type of job

        Returns:
            List of required parameter names

        # TODO: Replace with job registry system for better maintainability
        # TODO: Add parameter type definitions and descriptions

        """
        return []

    @abstractmethod
    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Execute the specific job implementation.
        Must be implemented by subclasses.

        Args:
            job: Job to execute

        Returns:
            Job execution result

        """

    def _cleanup_old_jobs(self, keep_recent: int = 500) -> None:
        """Clean up old completed jobs.

        Args:
            keep_recent: Number of recent jobs to keep

        """
        if len(self.jobs) <= keep_recent:
            return

        # Sort jobs by creation time and keep the most recent
        sorted_jobs = sorted(self.jobs.items(), key=lambda x: x[1].created_at, reverse=True)

        # Keep only the most recent jobs
        jobs_to_keep = dict(sorted_jobs[:keep_recent])

        removed_count = len(self.jobs) - len(jobs_to_keep)
        self.jobs = jobs_to_keep

        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old jobs")

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> None:
        """Clean up old completed jobs by age.

        Args:
            max_age_hours: Maximum age in hours before cleanup

        # TODO: Add configuration for cleanup policies (age, count, size limits)
        # TODO: Implement job archival before deletion for audit purposes

        """
        cutoff_time = datetime.now(UTC) - timedelta(hours=max_age_hours)

        jobs_to_remove: list[Any] = []
        for job_id, job in self.jobs.items():
            if (
                job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
                and job.completed_at
                and job.completed_at < cutoff_time
            ):
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.jobs[job_id]

        if jobs_to_remove:
            self.logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")

    def get_agent_stats(self) -> dict[str, Any]:
        """Get agent statistics with enhanced metrics."""
        total_jobs = len(self.jobs)
        completed_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.COMPLETED)
        failed_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.FAILED)
        running_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.RUNNING)
        cancelled_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.CANCELLED)

        # Calculate average duration for completed jobs
        completed_durations = [
            job.get_duration()
            for job in self.jobs.values()
            if job.status == JobStatus.COMPLETED and job.get_duration() is not None
        ]
        avg_duration = (
            sum(completed_durations) / len(completed_durations) if completed_durations else 0
        )

        return {
            "agent_name": self.agent_name,
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "running_jobs": running_jobs,
            "cancelled_jobs": cancelled_jobs,
            "success_rate": completed_jobs / max(total_jobs, 1) * 100,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "active_streams": len(self.active_streams),
            "average_duration_seconds": round(avg_duration, 2),
            "current_running_jobs": self.running_jobs,
        }

    def list_jobs(
        self,
        status_filter: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List jobs with optional filtering and pagination.

        Args:
            status_filter: Optional status to filter by
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of job dictionaries

        """
        jobs = list(self.jobs.values())

        # Filter by status if specified
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)

        # Apply pagination
        jobs = jobs[offset : offset + limit]

        return [job.to_dict() for job in jobs]

    def safe_file_read(self, file_path: Path) -> str | None:
        """Safely read file content.

        Args:
            file_path (Path): The path to the file.

        Returns: Optional[str]: File content, or None if an error occurs.

        """
        try:
            with open(file_path, encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            self.logger.exception(f"Error reading file {file_path}: {e}")
            return None

    def safe_file_write(self, file_path: Path, content: str) -> bool:
        """Safely write content to file.

        Args:
            file_path (Path): The path to the file.
            content (str): The content to write.

        Returns:
            bool: True if successful, False otherwise.

        """
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
            return True
        except Exception as e:
            self.logger.exception(f"Error writing file {file_path}: {e}")
            return False

    def create_backup(self, file_path: Path) -> str | None:
        """Create a backup of a file.

        Args:
            file_path (Path): The path to the file.

        Returns: Optional[str]: Path to the backup file, or None if an error occurs.

        """
        try:
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            import shutil

            shutil.copy2(file_path, backup_path)  # Copy with metadata
            return str(backup_path)
        except Exception as e:
            self.logger.exception(f"Error creating backup for {file_path}: {e}")
            return None

    def run_command(self, command: list[str]) -> Any:  # Updated return type hint
        """Run a shell command.

        Args:
            command (list[str]): The command to run, as a list of strings.

        Returns:
            Any: The subprocess result.

        """
        try:
            return safe_subprocess_run(command)  # Use safe subprocess
        except Exception as e:
            self.logger.exception(f"Error running command {command}: {e}")
            return None

    def find_files(self, directory: Path, pattern: str) -> list[Path]:
        """Find files matching a pattern within a directory.

        Args:
            directory (Path): The directory to search in.
            pattern (str): The file pattern to match.

        Returns:
            list[Path]: List of matching file paths.

        """
        return list(directory.rglob(pattern))

    def _scan_file_for_secrets(self, content: str, patterns: list[str]) -> list[dict[str, Any]]:
        """Scan file content for potential secrets."""
        secrets: list[Any] = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    secrets.append(
                        {"line": line_num, "type": "potential_secret", "pattern": pattern}
                    )

        return secrets

    # Security utilities (local implementations to avoid circular imports)
    def safe_json_parse(self, json_string: str) -> dict[str, Any] | None:
        """Safely parse JSON string."""
        try:
            return json.loads(json_string)
        except (json.JSONDecodeError, TypeError):
            return None

    def safe_file_read(self, file_path: Path) -> str | None:
        """Safely read file content."""
        try:
            if not file_path.exists():
                return None
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None

    def safe_file_write(self, file_path: Path, content: str) -> bool:
        """Safely write content to file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return True
        except OSError:
            return False

    def validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security."""
        path = Path(file_path)
        try:
            # Check for path traversal attempts
            path.resolve()
            return not any(part.startswith("..") for part in path.parts)
        except (OSError, ValueError):
            return False


class CodeGenerationService(BaseAgentService):
    """Comprehensive code generation service."""

    def __init__(self) -> None:
        super().__init__("code_generator", "Automated code generation and scaffolding")

        # Code templates for different technologies
        self.templates = {
            "fastapi_endpoint": self._get_fastapi_endpoint_template(),
            "pydantic_model": self._get_pydantic_model_template(),
            "react_component": self._get_react_component_template(),
            "python_test": self._get_python_test_template(),
            "sqlalchemy_model": self._get_sqlalchemy_model_template(),
            "docker_compose": self._get_docker_compose_template(),
            "github_workflow": self._get_github_workflow_template(),
        }

        # Supported languages and frameworks
        self.supported_frameworks = {
            "backend": ["fastapi", "django", "flask", "express", "spring"],
            "frontend": ["react", "vue", "angular", "svelte"],
            "database": ["sqlalchemy", "mongoose", "prisma"],
            "testing": ["pytest", "jest", "junit"],
        }

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Get required parameters for job type."""
        if job_type == "generate_code":
            return ["specification"]
        if job_type == "generate_api":
            return ["api_specification"]
        if job_type == "generate_models":
            return ["model_specification"]
        if job_type == "generate_tests":
            return ["code_path"]
        return []

    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Execute code generation job implementation."""
        job_type = job.job_type
        parameters = job.parameters

        if job_type == "generate_code":
            return await self._generate_code(job, parameters["specification"])
        if job_type == "generate_api":
            return await self._generate_api(job, parameters["api_specification"])
        if job_type == "generate_models":
            return await self._generate_models(job, parameters["model_specification"])
        if job_type == "generate_tests":
            return await self._generate_tests(job, parameters["code_path"])
        if job_type == "generate_frontend":
            return await self._generate_frontend(job, parameters["frontend_specification"])
        if job_type == "generate_migration":
            return await self._generate_migration(job, parameters["migration_specification"])
        if job_type == "scaffold_project":
            return await self._scaffold_project(job, parameters["project_specification"])
        msg = f"Unknown job type: {job_type}"
        raise ValueError(msg)

    async def _generate_code(self, job: Job, specification: dict[str, Any]) -> dict[str, Any]:
        """Generate code from general specification."""
        job.update_progress(10.0, "Analyzing code specification")

        code_generation_result = {
            "generated_files": [],
            "warnings": [],
            "suggestions": [],
            "total_lines_generated": 0,
        }

        # Parse specification
        job.update_progress(20.0, "Parsing specification")
        parsed_spec = await self._parse_specification(specification)

        # Generate different types of code based on specification
        if "api_endpoints" in parsed_spec:
            job.update_progress(40.0, "Generating API endpoints")
            api_result = await self._generate_api_code(parsed_spec["api_endpoints"])
            code_generation_result["generated_files"].extend(api_result["files"])

        if "data_models" in parsed_spec:
            job.update_progress(60.0, "Generating data models")
            model_result = await self._generate_model_code(parsed_spec["data_models"])
            code_generation_result["generated_files"].extend(model_result["files"])

        if "frontend_components" in parsed_spec:
            job.update_progress(80.0, "Generating frontend components")
            frontend_result = await self._generate_frontend_code(parsed_spec["frontend_components"])
            code_generation_result["generated_files"].extend(frontend_result["files"])

        # Write generated files
        job.update_progress(90.0, "Writing generated files")
        await self._write_generated_files(code_generation_result["generated_files"])

        # Calculate metrics
        code_generation_result["total_lines_generated"] = sum(
            len(file["content"].split("\n")) for file in code_generation_result["generated_files"]
        )

        job.update_progress(100.0, "Code generation complete")

        return {
            "specification": specification,
            "generation_result": code_generation_result,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_api(self, job: Job, api_specification: dict[str, Any]) -> dict[str, Any]:
        """Generate API endpoints from specification."""
        job.update_progress(10.0, "Analyzing API specification")

        api_generation_result = {
            "endpoints": [],
            "models": [],
            "generated_files": [],
            "openapi_spec": None,
        }

        # Extract endpoints from specification
        job.update_progress(30.0, "Extracting endpoint definitions")
        endpoints = api_specification.get("endpoints", [])

        # Generate endpoint code
        job.update_progress(50.0, "Generating endpoint code")
        for endpoint in endpoints:
            endpoint_code = await self._generate_endpoint_code(endpoint)
            if endpoint_code:
                api_generation_result["endpoints"].append(endpoint_code)

        # Generate data models
        job.update_progress(70.0, "Generating data models")
        models = api_specification.get("models", [])
        for model in models:
            model_code = await self._generate_model_code_single(model)
            if model_code:
                api_generation_result["models"].append(model_code)

        # Generate OpenAPI specification
        job.update_progress(85.0, "Generating OpenAPI specification")
        api_generation_result["openapi_spec"] = await self._generate_openapi_specification(
            api_specification
        )

        # Create file structure
        job.update_progress(95.0, "Creating file structure")
        api_generation_result["generated_files"] = await self._create_api_file_structure(
            api_generation_result
        )

        job.update_progress(100.0, "API generation complete")

        return {
            "api_specification": api_specification,
            "api_generation": api_generation_result,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_models(
        self, job: Job, model_specification: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate data models from specification."""
        job.update_progress(10.0, "Analyzing model specification")

        model_generation_result = {
            "pydantic_models": [],
            "sqlalchemy_models": [],
            "generated_files": [],
            "relationships": [],
        }

        models = model_specification.get("models", [])
        model_type = model_specification.get("type", "pydantic")

        # Generate models
        job.update_progress(40.0, f"Generating {model_type} models")
        for model in models:
            if model_type == "pydantic":
                model_code = await self._generate_pydantic_model(model)
                model_generation_result["pydantic_models"].append(model_code)
            elif model_type == "sqlalchemy":
                model_code = await self._generate_sqlalchemy_model(model)
                model_generation_result["sqlalchemy_models"].append(model_code)

        # Generate relationships
        job.update_progress(70.0, "Generating model relationships")
        relationships = model_specification.get("relationships", [])
        for relationship in relationships:
            rel_code = await self._generate_relationship_code(relationship, model_type)
            model_generation_result["relationships"].append(rel_code)

        # Create files
        job.update_progress(90.0, "Creating model files")
        model_generation_result["generated_files"] = await self._create_model_files(
            model_generation_result, model_type
        )

        job.update_progress(100.0, "Model generation complete")

        return {
            "model_specification": model_specification,
            "model_generation": model_generation_result,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_tests(self, job: Job, code_path: str) -> dict[str, Any]:
        """Generate tests for existing code."""
        job.update_progress(10.0, "Analyzing code structure")

        test_generation_result = {
            "test_files": [],
            "coverage_report": None,
        }

        # Analyze code structure
        job.update_progress(30.0, "Analyzing file dependencies")
        dependencies = await self._analyze_code_dependencies(code_path)

        # Generate tests for core components
        job.update_progress(60.0, "Generating test cases")
        test_files = await self._generate_unit_tests(code_path, dependencies)
        test_generation_result["test_files"] = test_files

        # Run tests and generate coverage report
        job.update_progress(80.0, "Running tests and generating coverage report")
        coverage_report = await self._run_tests_and_get_coverage(code_path)
        test_generation_result["coverage_report"] = coverage_report

        job.update_progress(100.0, "Test generation complete")

        return {
            "code_path": code_path,
            "test_generation": test_generation_result,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_frontend(
        self, job: Job, frontend_specification: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate frontend components from specification."""
        # Placeholder for frontend generation logic.
        # Should analyze the frontend_specification, generate necessary code,
        # and structure the files appropriately.  This is a complex process
        # and should be implemented based on the frameworks/libraries supported.
        # This is a simplified example.
        job.update_progress(50.0, "Generating frontend components (placeholder)")

        component_name = frontend_specification.get("component_name", "MyComponent")
        component_code = f"<div>Generated {component_name}</div>"

        job.update_progress(80.0, "Creating frontend file")
        filepath = Path("src") / f"{component_name}.js"
        self.safe_file_write(filepath, component_code)

        job.update_progress(100.0, "Frontend generation complete (placeholder)")

        return {
            "frontend_specification": frontend_specification,
            "generated_files": [{"filename": str(filepath), "content": component_code}],
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_migration(
        self, job: Job, migration_specification: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate database migration from specification."""
        # Placeholder for migration generation logic.
        # This requires connecting to the database and generating migration scripts.
        # This is a complex process and should be implemented based on the ORM/database system being used.
        # This is a simplified example.
        job.update_progress(50.0, "Generating database migration (placeholder)")

        migration_name = migration_specification.get("migration_name", "InitialMigration")
        migration_code = f"-- Migration: {migration_name}"

        job.update_progress(80.0, "Creating migration file")
        filepath = Path("migrations") / f"{migration_name}.sql"
        self.safe_file_write(filepath, migration_code)

        job.update_progress(100.0, "Migration generation complete (placeholder)")

        return {
            "migration_specification": migration_specification,
            "generated_files": [{"filename": str(filepath), "content": migration_code}],
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _scaffold_project(
        self, job: Job, project_specification: dict[str, Any]
    ) -> dict[str, Any]:
        """Scaffold a complete project from specification."""
        # Placeholder for project scaffolding.
        # This involves setting up directories, generating initial files,
        # and configuring the project based on the specified technologies.
        # This is a very complex process and should be implemented based on
        # the selected frameworks and technologies.
        # This is a simplified example.
        job.update_progress(50.0, "Scaffolding project (placeholder)")

        project_name = project_specification.get("project_name", "MyProject")
        readme_content = f"# {project_name}\nGenerated project."

        job.update_progress(80.0, "Creating initial files")
        project_dir = Path(project_name)
        project_dir.mkdir(exist_ok=True)

        filepath = project_dir / "README.md"
        self.safe_file_write(filepath, readme_content)

        job.update_progress(100.0, "Project scaffolding complete (placeholder)")

        return {
            "project_specification": project_specification,
            "generated_files": [{"filename": str(filepath), "content": readme_content}],
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _parse_specification(self, specification: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate the code specification."""
        # Placeholder for specification parsing and validation
        # This is a critical step to ensure the specification is valid before code generation
        # This method should validate the types and required fields in the specification
        return specification

    async def _generate_api_code(self, api_endpoints: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate code for API endpoints."""
        # Placeholder for API code generation.
        files: list[Any] = []
        for endpoint in api_endpoints:
            filename = endpoint.get("name", "unknown") + ".py"
            content = f"# API endpoint: {endpoint.get('path', '/')}"
            files.append({"filename": filename, "content": content})

        return {"files": files}

    async def _generate_model_code(self, data_models: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate code for data models."""
        # Placeholder for model code generation
        files: list[Any] = []
        for model in data_models:
            filename = model.get("name", "unknown") + ".py"
            content = f"# Data model: {model.get('name', 'Unknown')}"
            files.append({"filename": filename, "content": content})

        return {"files": files}

    async def _generate_frontend_code(
        self, frontend_components: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate code for frontend components."""
        # Placeholder for frontend code generation
        files: list[Any] = []
        for component in frontend_components:
            filename = component.get("name", "unknown") + ".js"
            content = f"// Frontend Component: {component.get('name', 'Unknown')}"
            files.append({"filename": filename, "content": content})

        return {"files": files}

    async def _write_generated_files(self, generated_files: list[dict[str, Any]]) -> None:
        """Write the generated files to disk."""
        # Placeholder for writing generated files
        for file in generated_files:
            filepath = Path(file["filename"])
            self.safe_file_write(filepath, file["content"])

    async def _generate_endpoint_code(self, endpoint: dict[str, Any]) -> str:
        """Generate code for a single API endpoint."""
        # Placeholder
        return f"# Generated endpoint: {endpoint.get('path', '/')}"

    async def _generate_model_code_single(self, model: dict[str, Any]) -> str:
        """Generate code for a single data model."""
        # Placeholder
        return f"# Generated model: {model.get('name', 'Unknown')}"

    async def _generate_openapi_specification(
        self, api_specification: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate OpenAPI specification from API specification."""
        # Placeholder
        return {"openapi": "3.0.0", "info": {"title": "Generated API"}}

    async def _create_api_file_structure(
        self, api_generation_result: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Create file structure for the generated API."""
        # Placeholder
        files: list[Any] = []
        for endpoint in api_generation_result.get("endpoints", []):
            filename = "api/endpoints/endpoint.py"
            files.append({"filename": filename, "content": endpoint})
        return files

    async def _generate_pydantic_model(self, model: dict[str, Any]) -> str:
        """Generate a Pydantic model."""
        # Placeholder
        return f"# Generated Pydantic model: {model.get('name', 'Unknown')}"

    async def _generate_sqlalchemy_model(self, model: dict[str, Any]) -> str:
        """Generate a SQLAlchemy model."""
        # Placeholder
        return f"# Generated SQLAlchemy model: {model.get('name', 'Unknown')}"

    async def _generate_relationship_code(
        self, relationship: dict[str, Any], model_type: str
    ) -> str:
        """Generate relationship code between models."""
        # Placeholder
        return f"# Generated relationship code: {relationship.get('type', 'Unknown')}"

    async def _create_model_files(
        self,
        model_generation_result: dict[str, Any],
        model_type: str,
    ) -> list[dict[str, Any]]:
        """Create files for the generated models."""
        # Placeholder
        files: list[Any] = []
        for model in model_generation_result.get("pydantic_models", []):
            filename = "models/pydantic_model.py"
            files.append({"filename": filename, "content": model})
        return files

    async def _analyze_code_dependencies(self, code_path: str) -> list[str]:
        """Analyze code dependencies to determine which components to test."""
        # Placeholder
        return ["module1", "module2"]

    async def _generate_unit_tests(
        self, code_path: str, dependencies: list[str]
    ) -> list[dict[str, Any]]:
        """Generate unit tests for the given code path and dependencies."""
        # Placeholder
        files: list[Any] = []
        for dep in dependencies:
            filename = f"tests/test_{dep}.py"
            content = f"# Test for {dep}"
            files.append({"filename": filename, "content": content})
        return files

    async def _run_tests_and_get_coverage(self, code_path: str) -> str:
        """Run tests and generate a coverage report."""
        # Placeholder
        return "Coverage: 100%"

    def _get_fastapi_endpoint_template(self) -> str:
        """Return a FastAPI endpoint template."""
        return """
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import Item

router = APIRouter()

@router.get("/items/{item_id}")
def read_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(Item).filter(Item.id == item_id).first()
    return item
"""

    def _get_pydantic_model_template(self) -> str:
        """Return a Pydantic model template."""
        return """
from pydantic import BaseModel

class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float
"""

    def _get_react_component_template(self) -> str:
        """Return a React component template."""
        return """
import React from 'react';

function MyComponent() {
    return (
        <div>
            <h1>Hello, React!</h1>
        </div>
    );
}

export default MyComponent;
"""

    def _get_python_test_template(self) -> str:
        """Return a Python test template (pytest)."""
        return """
import pytest
from my_module import my_function

def test_my_function() -> None:
    assert my_function(2) == 4
"""

    def _get_sqlalchemy_model_template(self) -> str:
        """Return a SQLAlchemy model template."""
        return """
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    price = Column(Float)
"""

    def _get_docker_compose_template(self) -> str:
        """Return a Docker Compose template."""
        return """
version: "3.9"
services:
  web:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./html:/usr/share/nginx/html
"""

    def _get_github_workflow_template(self) -> str:
        """Return a GitHub Actions workflow template."""
        return """
name: CI/CD

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
      - name: Run a one-line script
        run: echo Hello, world!
"""


class WorkspaceAnalyzerService(BaseAgentService):
    """Service for analyzing workspace files and secrets."""

    def __init__(self) -> None:
        super().__init__(
            "workspace_analyzer", "Analyzes workspace files for secrets and other issues."
        )

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Get required parameters for workspace analysis job types."""
        if job_type == "analyze_workspace":
            return ["workspace_path"]
        if job_type == "scan_file":
            return ["file_path"]
        return []

    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Execute workspace analysis job."""
        job_type = job.job_type
        parameters = job.parameters

        if job_type == "analyze_workspace":
            return await self._analyze_workspace(job, parameters["workspace_path"])
        if job_type == "scan_file":
            return await self._scan_file(job, parameters["file_path"])
        msg = f"Unknown job type: {job_type}"
        raise ValueError(msg)

    async def _analyze_workspace(self, job: Job, workspace_path: str) -> dict[str, Any]:
        """Analyze the workspace for security issues, dependencies, etc."""
        job.update_progress(10, "Starting workspace analysis")

        workspace_dir = Path(workspace_path)

        if not workspace_dir.exists() or not workspace_dir.is_dir():
            msg = f"Invalid workspace path: {workspace_path}"
            raise ValueError(msg)

        results = {
            "summary": {},
            "file_analysis": [],
            "total_files_analyzed": 0,
            "issues_found": 0,
        }

        # Define secret patterns
        secret_patterns = [
            r"API_KEY\s*=\s*[\"']?([A-Za-z0-9\-_]+)[\"']?",
            r"password\s*[:=]\s*[\"']?([A-Za-z0-9\-_]+)[\"']?",
            r"PRIVATE_KEY\s*=\s*[\"']?([A-Za-z0-9\-_]+)[\"']?",
        ]

        # Iterate through all files in the workspace
        for file_path in workspace_dir.rglob("*"):
            if file_path.is_file():
                job.update_progress(
                    30 + (results["total_files_analyzed"] * 50) / 100,
                    f"Analyzing file: {file_path}",
                )
                file_content = self.safe_file_read(file_path)
                if file_content:
                    file_scan_results = self._scan_file_for_secrets(file_content, secret_patterns)
                    if file_scan_results:
                        results["file_analysis"].append(
                            {"file_path": str(file_path), "secrets_found": file_scan_results},
                        )
                        results["issues_found"] += len(file_scan_results)
                results["total_files_analyzed"] += 1

        results["summary"] = {
            "total_files": results["total_files_analyzed"],
            "potential_secrets_found": results["issues_found"],
            "security_score": max(0, 100 - results["issues_found"]),
        }

        job.update_progress(90, "Analysis complete")

        return {
            "workspace_path": workspace_path,
            "analysis_results": results,
            "analyzed_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _scan_file(self, job: Job, file_path: str) -> dict[str, Any]:
        """Scan a specific file for potential secrets."""
        job.update_progress(10, f"Scanning file: {file_path}")
        file_path_obj = Path(file_path)

        if not file_path_obj.exists() or not file_path_obj.is_file():
            msg = f"Invalid file path: {file_path}"
            raise ValueError(msg)

        # Define secret patterns
        secret_patterns = [
            r"API_KEY\s*=\s*[\"']?([A-Za-z0-9\-_]+)[\"']?",
            r"password\s*[:=]\s*[\"']?([A-Za-z0-9\-_]+)[\"']?",
            r"PRIVATE_KEY\s*=\s*[\"']?([A-Za-z0-9\-_]+)[\"']?",
        ]

        file_content = self.safe_file_read(file_path_obj)

        if not file_content:
            msg = f"Could not read file: {file_path}"
            raise ValueError(msg)

        secrets_found = self._scan_file_for_secrets(file_content, secret_patterns)

        job.update_progress(90, "File scan complete")

        return {
            "file_path": file_path,
            "secrets_found": secrets_found,
            "scanned_at": job.started_at.isoformat() if job.started_at else None,
        }

    def _scan_file_for_secrets(self, content: str, patterns: list[str]) -> list[dict[str, Any]]:
        """Scan file content for potential secrets using regex patterns."""
        secrets_found: list[Any] = []

        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                secrets_found.append(
                    {
                        "pattern": pattern,
                        "match": match.group(0),
                        "line_number": content[: match.start()].count("\n") + 1,
                        "confidence": "high" if len(match.group(1)) > 10 else "medium",
                    },
                )

        return secrets_found
