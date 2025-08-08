"""
Comprehensive unit tests for BaseAgentService class.

Tests initialization, job management, async operations, error handling,
and all core functionality with extensive mocking.
"""

import asyncio
from datetime import UTC
from datetime import datetime
import json
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

from app.agents.base_agent_service import BaseAgentService
from app.agents.base_agent_service import Job
from app.agents.base_agent_service import JobStatus
import pytest


class TestJob:
    """Test the Job class functionality."""

    def test_job_initialization_valid(self):
        """Test valid job initialization."""
        job_id = "test-job-123"
        job_type = "analysis"
        parameters = {"input": "test data", "mode": "full"}

        job = Job(job_id, job_type, parameters)

        assert job.job_id == job_id
        assert job.job_type == job_type
        assert job.parameters == parameters
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0
        assert job.result is None
        assert job.error is None
        assert isinstance(job.logs, list)
        assert len(job.logs) == 0
        assert job.started_at is None
        assert job.completed_at is None
        assert isinstance(job.created_at, datetime)

    def test_job_initialization_invalid_job_id(self):
        """Test job initialization with invalid job ID."""
        with pytest.raises(ValueError, match="Job ID must be a non-empty string"):
            Job("", "analysis", {})

        with pytest.raises(ValueError, match="Job ID must be a non-empty string"):
            Job(None, "analysis", {})

        with pytest.raises(ValueError, match="Job ID must be a non-empty string"):
            Job(123, "analysis", {})

    def test_job_initialization_invalid_job_type(self):
        """Test job initialization with invalid job type."""
        with pytest.raises(ValueError, match="Job type must be a non-empty string"):
            Job("test-id", "", {})

        with pytest.raises(ValueError, match="Job type must be a non-empty string"):
            Job("test-id", None, {})

    def test_job_initialization_invalid_parameters(self):
        """Test job initialization with invalid parameters."""
        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            Job("test-id", "analysis", None)

        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            Job("test-id", "analysis", "invalid")

    def test_job_defensive_copying(self):
        """Test that job makes defensive copies of parameters."""
        original_params = {"key": "value", "nested": {"item": 1}}
        job = Job("test-id", "analysis", original_params)

        # Modify original - should not affect job
        original_params["key"] = "modified"
        original_params["nested"]["item"] = 999

        assert job.parameters["key"] == "value"
        # Note: shallow copy means nested objects are still shared
        # This tests the current implementation, not deep copying

    def test_job_start_valid(self):
        """Test starting a job in pending status."""
        job = Job("test-id", "analysis", {})

        job.start()

        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
        assert isinstance(job.started_at, datetime)

    def test_job_start_invalid_status(self):
        """Test starting a job in invalid status."""
        job = Job("test-id", "analysis", {})
        job.status = JobStatus.RUNNING

        with pytest.raises(ValueError, match="Cannot start job in running status"):
            job.start()

    def test_job_complete_valid(self):
        """Test completing a running job."""
        job = Job("test-id", "analysis", {})
        job.start()

        result = {"output": "test result", "metrics": {"score": 95}}
        job.complete(result)

        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.progress == 100.0
        assert job.result == result
        assert job.result is not result  # Should be a copy

    def test_job_complete_invalid_status(self):
        """Test completing a job in invalid status."""
        job = Job("test-id", "analysis", {})

        with pytest.raises(ValueError, match="Cannot complete job in pending status"):
            job.complete({"result": "test"})

    def test_job_complete_invalid_result(self):
        """Test completing a job with invalid result."""
        job = Job("test-id", "analysis", {})
        job.start()

        with pytest.raises(ValueError, match="Result must be a dictionary"):
            job.complete("invalid result")

    def test_job_fail_valid(self):
        """Test failing a job."""
        job = Job("test-id", "analysis", {})
        job.start()

        error_msg = "Test error occurred"
        job.fail(error_msg)

        assert job.status == JobStatus.FAILED
        assert job.completed_at is not None
        assert job.error == error_msg

    def test_job_fail_invalid_error(self):
        """Test failing a job with invalid error."""
        job = Job("test-id", "analysis", {})

        with pytest.raises(ValueError, match="Error message must be a non-empty string"):
            job.fail("")

        with pytest.raises(ValueError, match="Error message must be a non-empty string"):
            job.fail(None)

    def test_job_cancel_valid(self):
        """Test cancelling a job."""
        job = Job("test-id", "analysis", {})

        job.cancel()

        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None

    def test_job_cancel_invalid_status(self):
        """Test cancelling a completed job."""
        job = Job("test-id", "analysis", {})
        job.start()
        job.complete({"result": "test"})

        with pytest.raises(ValueError, match="Cannot cancel job in completed status"):
            job.cancel()

    def test_job_add_log_valid(self):
        """Test adding log messages."""
        job = Job("test-id", "analysis", {})

        job.add_log("Starting analysis")
        job.add_log("Processing data")

        assert len(job.logs) == 2
        assert "Starting analysis" in job.logs[0]
        assert "Processing data" in job.logs[1]
        # Check timestamp format
        assert "[" in job.logs[0] and "]" in job.logs[0]

    def test_job_add_log_invalid(self):
        """Test adding invalid log messages."""
        job = Job("test-id", "analysis", {})

        # Should silently ignore invalid messages
        job.add_log("")
        job.add_log(None)
        job.add_log(123)

        assert len(job.logs) == 0

    def test_job_add_log_size_limit(self):
        """Test log size limit enforcement."""
        job = Job("test-id", "analysis", {})

        # Add exactly 1001 log entries to trigger the first truncation
        for i in range(1001):
            job.add_log(f"Log entry {i}")

        # Should be truncated to 500 entries after exceeding 1000
        assert len(job.logs) == 500
        # Check that the logs contain the expected entries (with timestamps)
        last_logs = [log for log in job.logs if "Log entry 1000" in log]
        assert len(last_logs) == 1

    def test_job_update_progress_valid(self):
        """Test updating job progress."""
        job = Job("test-id", "analysis", {})

        job.update_progress(50.0, "Halfway done")

        assert job.progress == 50.0
        assert len(job.logs) == 1
        assert "Halfway done" in job.logs[0]

    def test_job_update_progress_bounds(self):
        """Test progress bounds enforcement."""
        job = Job("test-id", "analysis", {})

        job.update_progress(-10.0)
        assert job.progress == 0.0

        job.update_progress(150.0)
        assert job.progress == 100.0

    def test_job_update_progress_invalid_type(self):
        """Test updating progress with invalid type."""
        job = Job("test-id", "analysis", {})

        with pytest.raises(ValueError, match="Progress must be a number"):
            job.update_progress("50")

    def test_job_get_duration(self):
        """Test getting job duration."""
        job = Job("test-id", "analysis", {})

        # Not started
        assert job.get_duration() is None

        # Started but not completed
        job.start()
        duration = job.get_duration()
        assert duration is not None
        assert duration >= 0

        # Completed
        job.complete({"result": "test"})
        duration = job.get_duration()
        assert duration is not None
        assert duration >= 0

    def test_job_to_dict(self):
        """Test converting job to dictionary."""
        job = Job("test-id", "analysis", {"input": "test"})
        job.start()
        job.add_log("Test log")
        job.update_progress(75.0)

        job_dict = job.to_dict()

        assert job_dict["job_id"] == "test-id"
        assert job_dict["job_type"] == "analysis"
        assert job_dict["parameters"] == {"input": "test"}
        assert job_dict["status"] == "running"
        assert job_dict["progress"] == 75.0
        assert job_dict["started_at"] is not None
        assert job_dict["created_at"] is not None
        assert isinstance(job_dict["logs"], list)
        assert job_dict["duration_seconds"] is not None


class TestJobStatus:
    """Test JobStatus enum."""

    def test_job_status_values(self):
        """Test job status enum values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"


class MockBaseAgentService(BaseAgentService):
    """Mock implementation of BaseAgentService for testing."""

    def __init__(self, agent_name: str = "test-agent", description: str = "Test agent"):
        super().__init__(agent_name, description)
        self.execute_job_calls = []
        self.validation_results = {}

    def register_tools(self) -> None:
        """Mock tool registration."""
        pass

    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Mock job execution."""
        self.execute_job_calls.append(job)

        # Simulate different job types
        if job.job_type == "failing_job":
            raise RuntimeError("Simulated job failure")
        elif job.job_type == "cancelled_job":
            raise asyncio.CancelledError()
        else:
            return {"status": "success", "job_type": job.job_type, "job_id": job.job_id}

    def validate_job_parameters(self, job_type: str, parameters: dict[str, Any]) -> bool:
        """Mock parameter validation."""
        # If we have explicit validation results, use them
        if job_type in self.validation_results:
            return self.validation_results[job_type]

        # Otherwise use the actual validation logic for testing
        if not isinstance(parameters, dict):
            return False

        required_params = self.get_required_parameters(job_type)

        for param in required_params:
            if param not in parameters:
                return False
            if parameters[param] is None:
                return False

        return True

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Mock required parameters."""
        return {
            "analysis": ["input", "mode"],
            "generation": ["prompt", "model"],
        }.get(job_type, [])


class TestBaseAgentService:
    """Test BaseAgentService functionality."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent service for testing."""
        return MockBaseAgentService()

    @pytest.fixture
    def mock_base_automation_agent(self):
        """Mock the BaseAutomationAgent constructor."""
        with patch("app.agents.base_agent_service.BaseAutomationAgent.__init__") as mock:
            mock.return_value = None
            yield mock

    def test_initialization_valid(self, mock_base_automation_agent):
        """Test valid agent initialization."""
        agent = MockBaseAgentService("test-agent", "Test description")

        assert agent.agent_name == "test-agent"
        assert agent.jobs == {}
        assert agent.max_concurrent_jobs == 5
        assert agent.running_jobs == 0
        assert agent.max_job_history == 1000
        assert agent.active_streams == {}

    def test_initialization_invalid_name(self, mock_base_automation_agent):
        """Test initialization with invalid agent name."""
        with pytest.raises(ValueError, match="Agent name must be a non-empty string"):
            MockBaseAgentService("", "description")

        with pytest.raises(ValueError, match="Agent name must be a non-empty string"):
            MockBaseAgentService(None, "description")

    def test_create_job_valid(self, mock_agent):
        """Test creating a valid job."""
        job_id = mock_agent.create_job("analysis", {"input": "test"})

        assert job_id in mock_agent.jobs
        job = mock_agent.jobs[job_id]
        assert job.job_type == "analysis"
        assert job.parameters == {"input": "test"}
        assert job.status == JobStatus.PENDING

    def test_create_job_invalid_type(self, mock_agent):
        """Test creating job with invalid type."""
        with pytest.raises(ValueError, match="Job type must be a non-empty string"):
            mock_agent.create_job("", {})

        with pytest.raises(ValueError, match="Job type must be a non-empty string"):
            mock_agent.create_job(None, {})

    def test_create_job_invalid_parameters(self, mock_agent):
        """Test creating job with invalid parameters."""
        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            mock_agent.create_job("analysis", None)

        with pytest.raises(ValueError, match="Parameters must be a dictionary"):
            mock_agent.create_job("analysis", "invalid")

    def test_create_job_cleanup_old_jobs(self, mock_agent):
        """Test automatic cleanup of old jobs."""
        mock_agent.max_job_history = 5

        # Create more jobs than the limit
        job_ids = []
        for i in range(7):
            job_id = mock_agent.create_job("analysis", {"index": i})
            job_ids.append(job_id)

        # Should trigger cleanup keeping only recent jobs
        assert len(mock_agent.jobs) <= 5

    def test_get_job_valid(self, mock_agent):
        """Test getting a valid job."""
        job_id = mock_agent.create_job("analysis", {"input": "test"})

        job = mock_agent.get_job(job_id)

        assert job is not None
        assert job.job_id == job_id

    def test_get_job_invalid(self, mock_agent):
        """Test getting invalid job."""
        assert mock_agent.get_job("nonexistent") is None
        assert mock_agent.get_job("") is None
        assert mock_agent.get_job(None) is None

    def test_get_job_status_valid(self, mock_agent):
        """Test getting job status."""
        job_id = mock_agent.create_job("analysis", {"input": "test"})

        response = mock_agent.get_job_status(job_id)

        assert response["status"] == "success"
        assert "job" in response
        assert response["job"]["job_id"] == job_id

    def test_get_job_status_invalid(self, mock_agent):
        """Test getting status of nonexistent job."""
        response = mock_agent.get_job_status("nonexistent")

        assert response["status"] == "error"
        assert "Job nonexistent not found" in response["error"]

    def test_cancel_job_valid(self, mock_agent):
        """Test cancelling a valid job."""
        job_id = mock_agent.create_job("analysis", {"input": "test"})

        response = mock_agent.cancel_job(job_id)

        assert response["status"] == "success"
        job = mock_agent.get_job(job_id)
        assert job.status == JobStatus.CANCELLED

    def test_cancel_job_invalid(self, mock_agent):
        """Test cancelling invalid job."""
        response = mock_agent.cancel_job("nonexistent")

        assert response["status"] == "error"
        assert "Job nonexistent not found" in response["error"]

    def test_cancel_job_invalid_status(self, mock_agent):
        """Test cancelling job in invalid status."""
        job_id = mock_agent.create_job("analysis", {"input": "test"})
        job = mock_agent.get_job(job_id)
        job.status = JobStatus.COMPLETED

        response = mock_agent.cancel_job(job_id)

        assert response["status"] == "error"

    @pytest.mark.asyncio
    async def test_execute_job_async_valid(self, mock_agent):
        """Test successful async job execution."""
        job_id = mock_agent.create_job("analysis", {"input": "test", "mode": "full"})

        response = await mock_agent.execute_job_async(job_id)

        assert response["status"] == "success"
        job = mock_agent.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        assert len(mock_agent.execute_job_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_job_async_nonexistent(self, mock_agent):
        """Test executing nonexistent job."""
        response = await mock_agent.execute_job_async("nonexistent")

        assert response["status"] == "error"
        assert "Job nonexistent not found" in response["error"]

    @pytest.mark.asyncio
    async def test_execute_job_async_max_concurrent(self, mock_agent):
        """Test max concurrent jobs limit."""
        mock_agent.running_jobs = mock_agent.max_concurrent_jobs

        job_id = mock_agent.create_job("analysis", {"input": "test"})
        response = await mock_agent.execute_job_async(job_id)

        assert response["status"] == "error"
        assert "Too many concurrent jobs" in response["error"]

    @pytest.mark.asyncio
    async def test_execute_job_async_validation_failure(self, mock_agent):
        """Test job execution with parameter validation failure."""
        mock_agent.validation_results["analysis"] = False

        job_id = mock_agent.create_job("analysis", {"input": "test"})
        response = await mock_agent.execute_job_async(job_id)

        assert response["status"] == "error"
        assert "Job parameter validation failed" in response["error"]
        job = mock_agent.get_job(job_id)
        assert job.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_job_async_job_failure(self, mock_agent):
        """Test job execution with job implementation failure."""
        job_id = mock_agent.create_job("failing_job", {"input": "test"})

        response = await mock_agent.execute_job_async(job_id)

        assert response["status"] == "error"
        assert "Job execution failed" in response["error"]
        job = mock_agent.get_job(job_id)
        assert job.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_job_async_cancellation(self, mock_agent):
        """Test job execution cancellation."""
        job_id = mock_agent.create_job("cancelled_job", {"input": "test"})

        response = await mock_agent.execute_job_async(job_id)

        assert response["status"] == "error"
        assert "Job was cancelled" in response["error"]
        job = mock_agent.get_job(job_id)
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_stream_job_progress_valid(self, mock_agent):
        """Test streaming job progress."""
        job_id = mock_agent.create_job("analysis", {"input": "test"})

        # Start the streaming
        stream = mock_agent.stream_job_progress(job_id)

        # Get first update (initial status)
        first_update = await stream.__anext__()
        assert first_update.startswith("data: ")

        # Parse JSON
        data = json.loads(first_update[6:-2])  # Remove "data: " and "\n\n"
        assert data["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_stream_job_progress_nonexistent(self, mock_agent):
        """Test streaming progress for nonexistent job."""
        stream = mock_agent.stream_job_progress("nonexistent")

        error_update = await stream.__anext__()
        assert "error" in error_update

    def test_stop_stream(self, mock_agent):
        """Test stopping a stream."""
        job_id = "test-job"
        mock_agent.active_streams[job_id] = True

        mock_agent.stop_stream(job_id)

        assert mock_agent.active_streams[job_id] is False

    @pytest.mark.asyncio
    async def test_generate_with_progress_valid(self, mock_agent):
        """Test content generation with progress tracking."""
        # Mock the get_model method
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated content"
        mock_model.generate_content.return_value = mock_response
        mock_agent.get_model = Mock(return_value=mock_model)

        result = await mock_agent.generate_with_progress("Test prompt")

        assert result == "Generated content"
        mock_model.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_progress_invalid_prompt(self, mock_agent):
        """Test generation with invalid prompt."""
        result = await mock_agent.generate_with_progress("")
        assert result is None

        result = await mock_agent.generate_with_progress(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_with_progress_with_job(self, mock_agent):
        """Test generation with job progress tracking."""
        job_id = mock_agent.create_job("generation", {"prompt": "test"})
        job = mock_agent.get_job(job_id)

        # Mock the get_model method
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated content"
        mock_model.generate_content.return_value = mock_response
        mock_agent.get_model = Mock(return_value=mock_model)

        result = await mock_agent.generate_with_progress("Test prompt", job=job)

        assert result == "Generated content"
        assert job.progress == 100.0
        assert len(job.logs) > 0

    @pytest.mark.asyncio
    async def test_generate_with_progress_callback(self, mock_agent):
        """Test generation with progress callback."""
        callback_calls = []

        def progress_callback(progress: float, message: str):
            callback_calls.append((progress, message))

        # Mock the get_model method
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated content"
        mock_model.generate_content.return_value = mock_response
        mock_agent.get_model = Mock(return_value=mock_model)

        result = await mock_agent.generate_with_progress(
            "Test prompt", progress_callback=progress_callback
        )

        assert result == "Generated content"
        assert len(callback_calls) > 0
        assert all(isinstance(call[0], int | float) for call in callback_calls)

    def test_validate_job_parameters_invalid_dict(self, mock_agent):
        """Test parameter validation with non-dict parameters."""
        result = mock_agent.validate_job_parameters("analysis", "not a dict")
        assert result is False

    def test_validate_job_parameters_missing_required(self, mock_agent):
        """Test parameter validation with missing required parameters."""
        # Reset validation results to use actual validation
        mock_agent.validation_results.clear()

        result = mock_agent.validate_job_parameters("analysis", {"mode": "full"})
        assert result is False  # Missing "input"

    def test_validate_job_parameters_none_values(self, mock_agent):
        """Test parameter validation with None values."""
        mock_agent.validation_results.clear()

        result = mock_agent.validate_job_parameters("analysis", {"input": None, "mode": "full"})
        assert result is False

    def test_get_required_parameters(self, mock_agent):
        """Test getting required parameters for job types."""
        params = mock_agent.get_required_parameters("analysis")
        assert params == ["input", "mode"]

        params = mock_agent.get_required_parameters("unknown")
        assert params == []

    def test_cleanup_old_jobs_by_count(self, mock_agent):
        """Test cleanup by count."""
        # Add some jobs
        for i in range(10):
            mock_agent.create_job("analysis", {"index": i})

        initial_count = len(mock_agent.jobs)
        mock_agent._cleanup_old_jobs(keep_recent=5)

        assert len(mock_agent.jobs) == 5
        assert len(mock_agent.jobs) < initial_count

    def test_cleanup_old_jobs_by_age(self, mock_agent):
        """Test cleanup by age."""
        # Create some jobs
        job_ids = []
        for i in range(3):
            job_id = mock_agent.create_job("analysis", {"index": i})
            job_ids.append(job_id)

        # Mark some as completed with old completion time
        from datetime import timedelta

        for job_id in job_ids[:2]:
            job = mock_agent.get_job(job_id)
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(UTC) - timedelta(hours=25)

        initial_count = len(mock_agent.jobs)
        mock_agent.cleanup_old_jobs(max_age_hours=24)

        # Should have removed the old completed jobs
        assert len(mock_agent.jobs) < initial_count

    def test_get_agent_stats(self, mock_agent):
        """Test getting agent statistics."""
        # Create jobs in different states
        job_id1 = mock_agent.create_job("analysis", {"test": 1})
        job_id2 = mock_agent.create_job("analysis", {"test": 2})
        job_id3 = mock_agent.create_job("analysis", {"test": 3})

        # Set different statuses
        mock_agent.get_job(job_id1).status = JobStatus.COMPLETED
        mock_agent.get_job(job_id2).status = JobStatus.FAILED
        mock_agent.get_job(job_id3).status = JobStatus.RUNNING

        stats = mock_agent.get_agent_stats()

        assert stats["agent_name"] == "test-agent"
        assert stats["total_jobs"] == 3
        assert stats["completed_jobs"] == 1
        assert stats["failed_jobs"] == 1
        assert stats["running_jobs"] == 1
        assert stats["cancelled_jobs"] == 0
        assert isinstance(stats["success_rate"], int | float)
        assert stats["max_concurrent_jobs"] == 5
        assert isinstance(stats["active_streams"], int)

    def test_list_jobs_no_filter(self, mock_agent):
        """Test listing jobs without filters."""
        # Create test jobs
        for i in range(3):
            mock_agent.create_job("analysis", {"index": i})

        jobs = mock_agent.list_jobs()

        assert len(jobs) == 3
        assert all(isinstance(job, dict) for job in jobs)

    def test_list_jobs_with_status_filter(self, mock_agent):
        """Test listing jobs with status filter."""
        job_id1 = mock_agent.create_job("analysis", {"test": 1})
        job_id2 = mock_agent.create_job("analysis", {"test": 2})

        # Set different statuses
        mock_agent.get_job(job_id1).status = JobStatus.COMPLETED
        mock_agent.get_job(job_id2).status = JobStatus.FAILED

        completed_jobs = mock_agent.list_jobs(status_filter=JobStatus.COMPLETED)
        failed_jobs = mock_agent.list_jobs(status_filter=JobStatus.FAILED)

        assert len(completed_jobs) == 1
        assert len(failed_jobs) == 1
        assert completed_jobs[0]["status"] == "completed"
        assert failed_jobs[0]["status"] == "failed"

    def test_list_jobs_with_pagination(self, mock_agent):
        """Test listing jobs with pagination."""
        # Create test jobs
        for i in range(10):
            mock_agent.create_job("analysis", {"index": i})

        # Test limit
        jobs = mock_agent.list_jobs(limit=5)
        assert len(jobs) == 5

        # Test offset
        jobs_page2 = mock_agent.list_jobs(limit=5, offset=5)
        assert len(jobs_page2) == 5

        # Should be different jobs
        job_ids_page1 = {job["job_id"] for job in jobs}
        job_ids_page2 = {job["job_id"] for job in jobs_page2}
        assert job_ids_page1.isdisjoint(job_ids_page2)
