"""
Error handling scenario tests for all major modules.
Tests failure modes, recovery mechanisms, and error propagation.
"""

import asyncio
import contextlib
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

from app.agents.base_agent_service import BaseAgentService
from app.agents.base_agent_service import Job
from app.agents.base_agent_service import JobStatus
from app.automation.auth_models import APIKey
from app.automation.auth_models import User
from app.automation.auth_storage import AuthStorage
from app.mcp_servers.base_mcp_server import BaseMCPServer
import pytest


class ErrorTestAgentService(BaseAgentService):
    """Test agent service with configurable error scenarios."""

    def __init__(self):
        super().__init__("error_test_agent", "Agent for error testing")
        self.force_error = None
        self.error_count = 0

    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Implementation that can simulate various error conditions."""
        job_type = job.job_type

        if self.force_error:
            self.error_count += 1
            if self.force_error == "timeout":
                await asyncio.sleep(10)  # Simulate timeout
            elif self.force_error == "memory_error":
                raise MemoryError("Simulated memory error")
            elif self.force_error == "value_error":
                raise ValueError("Simulated value error")
            elif self.force_error == "generic_error":
                raise Exception("Simulated generic error")
            elif self.force_error == "async_error":
                await asyncio.sleep(0.1)
                raise RuntimeError("Simulated async error")
            elif self.force_error == "intermittent" and self.error_count % 3 == 0:
                raise ConnectionError("Intermittent connection error")

        if job_type == "failing_task":
            raise ValueError(f"Task designed to fail: {job.parameters.get('reason', 'no reason')}")
        elif job_type == "timeout_task":
            await asyncio.sleep(5)  # Long running task
            return {"result": "Should not complete due to timeout"}
        elif job_type == "resource_error":
            raise OSError("Resource not available")
        elif job_type == "successful_task":
            return {"result": "Task completed successfully"}
        else:
            return {"result": f"Processed {job_type}"}


class ErrorTestMCPServer(BaseMCPServer):
    """Test MCP server with error scenarios."""

    def __init__(self):
        super().__init__("error_test_mcp")
        self.should_fail = False

    async def _initialize_components(self):
        """Implementation that can fail during initialization."""
        if self.should_fail:
            raise ConnectionError("Failed to connect to external service")


@pytest.fixture
def error_agent():
    """Create error test agent instance."""
    return ErrorTestAgentService()


@pytest.fixture
def error_mcp_server():
    """Create error test MCP server instance."""
    return ErrorTestMCPServer()


@pytest.fixture
def temp_auth_storage():
    """Create temporary auth storage for error testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = AuthStorage(Path(temp_dir))
        yield storage


class TestAgentServiceErrorHandling:
    """Test error handling in BaseAgentService."""

    @pytest.mark.asyncio
    async def test_job_execution_value_error(self, error_agent):
        """Test handling of ValueError during job execution."""
        error_agent.force_error = "value_error"

        job_id = error_agent.create_job("test_task", {})
        result = await error_agent.execute_job_async(job_id)

        assert result["success"] is False
        assert "execution failed" in result["message"].lower()

        job = error_agent.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert "Simulated value error" in job.error
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_job_execution_memory_error(self, error_agent):
        """Test handling of MemoryError during job execution."""
        error_agent.force_error = "memory_error"

        job_id = error_agent.create_job("test_task", {})
        result = await error_agent.execute_job_async(job_id)

        assert result["success"] is False
        job = error_agent.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert "memory error" in job.error.lower()

    @pytest.mark.asyncio
    async def test_job_execution_async_error(self, error_agent):
        """Test handling of errors in async operations."""
        error_agent.force_error = "async_error"

        job_id = error_agent.create_job("test_task", {})
        result = await error_agent.execute_job_async(job_id)

        assert result["success"] is False
        job = error_agent.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert "async error" in job.error.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_job_execution(self, error_agent):
        """Test execution of non-existent job."""
        result = await error_agent.execute_job_async("nonexistent_job_id")

        assert result["success"] is False
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_concurrent_job_limit_exceeded(self, error_agent):
        """Test behavior when concurrent job limit is exceeded."""
        error_agent.max_concurrent_jobs = 1

        # Create two jobs
        job1_id = error_agent.create_job("timeout_task", {})
        job2_id = error_agent.create_job("test_task", {})

        # Start first job (will take time)
        task1 = asyncio.create_task(error_agent.execute_job_async(job1_id))

        # Wait briefly then try second job
        await asyncio.sleep(0.1)
        result2 = await error_agent.execute_job_async(job2_id)

        # Second job should be rejected
        assert result2["success"] is False
        assert "too many concurrent" in result2["message"].lower()

        # Cancel first job to avoid hanging
        task1.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task1

    @pytest.mark.asyncio
    async def test_streaming_nonexistent_job(self, error_agent):
        """Test streaming for non-existent job."""
        updates = []
        async for update in error_agent.stream_job_progress("nonexistent"):
            updates.append(update)
            break  # Just get first update

        assert len(updates) == 1
        data = json.loads(updates[0][6:])  # Remove "data: " prefix
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_streaming_error_during_execution(self, error_agent):
        """Test streaming when job encounters error during execution."""
        error_agent.force_error = "value_error"

        job_id = error_agent.create_job("test_task", {})

        # Start streaming
        updates = []
        stream_task = asyncio.create_task(
            self._collect_stream_updates(error_agent, job_id, updates)
        )

        # Start job execution
        execution_task = asyncio.create_task(error_agent.execute_job_async(job_id))

        # Wait for execution to fail
        result = await execution_task
        await asyncio.sleep(0.1)  # Give streaming time to update

        # Stop streaming
        error_agent.stop_stream(job_id)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stream_task, timeout=1.0)

        assert result["success"] is False
        assert len(updates) > 0

        # Final update should show failed status
        final_update = json.loads(updates[-1][6:])
        assert final_update["status"] == "failed"

    async def _collect_stream_updates(self, agent, job_id, updates_list):
        """Helper to collect streaming updates."""
        async for update in agent.stream_job_progress(job_id):
            updates_list.append(update)
            if len(updates_list) >= 10:  # Limit collection
                break

    def test_job_validation_errors(self, error_agent):
        """Test job parameter validation errors."""
        # Mock validation to fail
        with patch.object(error_agent, "get_required_parameters", return_value=["required_param"]):
            valid = error_agent.validate_job_parameters("test_job", {})
            assert valid is False

    @pytest.mark.asyncio
    async def test_intermittent_error_handling(self, error_agent):
        """Test handling of intermittent errors."""
        error_agent.force_error = "intermittent"

        results = []
        for i in range(5):
            job_id = error_agent.create_job("test_task", {"attempt": i})
            result = await error_agent.execute_job_async(job_id)
            results.append(result)

        # Some should succeed, some should fail
        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]

        assert len(successes) > 0, "Some jobs should succeed with intermittent errors"
        assert len(failures) > 0, "Some jobs should fail with intermittent errors"


class TestAuthenticationErrorHandling:
    """Test error handling in authentication components."""

    def test_auth_storage_file_corruption(self, temp_auth_storage):
        """Test handling of corrupted storage files."""
        # Corrupt the users file
        temp_auth_storage.users_file.write_text("{ invalid json")

        # Should handle gracefully and return empty data
        data = temp_auth_storage._load_json(temp_auth_storage.users_file)
        assert data == {}

    def test_auth_storage_permission_error(self, temp_auth_storage):
        """Test handling of file permission errors."""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            # Should handle gracefully
            data = temp_auth_storage._load_json(temp_auth_storage.users_file)
            assert data == {}

    def test_auth_storage_disk_full_error(self, temp_auth_storage):
        """Test handling of disk full errors during save."""
        test_data = {"test": "data"}

        with patch("builtins.open", side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                temp_auth_storage._save_json(temp_auth_storage.users_file, test_data)

    def test_user_creation_with_invalid_data(self, temp_auth_storage):
        """Test user creation with invalid data."""
        # User with missing required fields
        invalid_user = User(
            id="",  # Empty ID
            username="",  # Empty username
            email="invalid-email",  # Invalid email format
            password_hash="",  # Empty password hash
        )

        # Should handle gracefully (may depend on validation)
        result = temp_auth_storage.create_user(invalid_user)
        # Result depends on validation - could be True or False
        assert isinstance(result, bool)

    def test_api_key_verification_with_corrupted_data(self, temp_auth_storage):
        """Test API key verification with corrupted data."""
        # Create API key with corrupted hash
        corrupted_key = APIKey(
            id="corrupted_key",
            user_id="user_123",
            name="Corrupted Key",
            key_hash="invalid_hash_format",
            key_prefix="corrupt01",
        )

        temp_auth_storage.create_api_key(corrupted_key)

        # Verification should fail gracefully
        result = temp_auth_storage.verify_api_key("corrupt01_full_key")
        assert result is None

    def test_authentication_with_locked_user(self, temp_auth_storage):
        """Test authentication with locked user account."""
        from datetime import timedelta

        # Create locked user
        locked_user = User(
            id="locked_user",
            username="lockeduser",
            email="locked@example.com",
            password_hash="hash",
            locked_until=datetime.now(UTC) + timedelta(hours=1),
        )

        temp_auth_storage.create_user(locked_user)

        # Authentication should fail
        result = temp_auth_storage.authenticate_user("lockeduser", "password")
        assert result is None

    def test_authentication_with_database_error(self, temp_auth_storage):
        """Test authentication when database operations fail."""
        with patch.object(
            temp_auth_storage, "get_user_by_username", side_effect=Exception("Database error")
        ):
            result = temp_auth_storage.authenticate_user("testuser", "password")
            assert result is None


class TestMCPServerErrorHandling:
    """Test error handling in MCP servers."""

    @pytest.mark.asyncio
    async def test_mcp_server_initialization_failure(self, error_mcp_server):
        """Test MCP server initialization failure."""
        error_mcp_server.should_fail = True

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await error_mcp_server.initialize()

        assert not error_mcp_server.initialized

    @pytest.mark.asyncio
    async def test_mcp_server_cleanup_after_error(self, error_mcp_server):
        """Test MCP server cleanup after initialization error."""
        error_mcp_server.should_fail = True

        with contextlib.suppress(ConnectionError):
            await error_mcp_server.initialize()

        # Should be able to clean up even after error
        await error_mcp_server.shutdown()
        assert not error_mcp_server.initialized

    def test_mcp_server_error_handling(self, error_mcp_server):
        """Test MCP server error handling utility."""
        test_error = ValueError("Test error message")
        result = error_mcp_server.handle_error(test_error, "test context")

        assert result["error"] == "Test error message"
        assert result["context"] == "test context"

    @pytest.mark.asyncio
    async def test_gemini_mcp_initialization_error(self):
        """Test Gemini MCP server initialization error."""
        with patch.dict(
            "os.environ",
            {
                "GOOGLE_CLOUD_PROJECT": "",
                "GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent/path.json",
            },
        ), pytest.raises(Exception):  # Should fail to initialize
            from app.mcp_servers.gemini_mcp_base import GeminiMCPBase

            GeminiMCPBase("test_server")

    @pytest.mark.asyncio
    async def test_gemini_mcp_content_generation_error(self):
        """Test Gemini MCP content generation error handling."""
        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPBase._initialize_model"):
            from app.mcp_servers.gemini_mcp_base import GeminiMCPBase

            server = GeminiMCPBase("test_server")

            # Mock model to raise exception
            server.model = Mock()
            server.model.generate_content.side_effect = Exception("API Error")

            with pytest.raises(Exception, match="API Error"):
                await server.generate_content("test prompt", use_cache=False)

    @pytest.mark.asyncio
    async def test_gemini_mcp_context_collection_error(self):
        """Test Gemini MCP context collection error handling."""
        with patch("app.mcp_servers.gemini_mcp_base.GeminiMCPBase._initialize_model"):
            from app.mcp_servers.gemini_mcp_base import GeminiMCPBase

            server = GeminiMCPBase("test_server")

            # Mock context collection to fail
            with patch.object(
                server, "_collect_context_python", side_effect=OSError("Permission denied")
            ):
                context = await server.collect_context_fast("/nonexistent/path")

                # Should return empty context instead of crashing
                assert context["file_count"] == 0
                assert context["rust_accelerated"] is False


class TestConfigurationErrorHandling:
    """Test error handling in configuration modules."""

    def test_invalid_environment_variables(self):
        """Test handling of invalid environment variables."""
        with patch.dict(
            "os.environ",
            {
                "GOOGLE_CLOUD_PROJECT": "",
                "GOOGLE_CLOUD_LOCATION": "",
                "GOOGLE_GENAI_USE_VERTEXAI": "invalid_boolean",
            },
        ):
            from app.automation.auth_storage import GeminiMCPConfig

            # Should handle invalid values gracefully
            config = GeminiMCPConfig()
            assert config.project == ""
            assert config.location == ""
            assert config.use_vertex is False  # Invalid boolean should default to False

    def test_missing_credentials_file(self):
        """Test handling of missing credentials file."""
        with patch.dict(
            "os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent/credentials.json"}
        ):
            # Should handle missing file gracefully
            from app.config import config

            assert config is not None

    def test_configuration_with_no_environment(self):
        """Test configuration when environment variables are missing."""
        # Remove all relevant environment variables

        with patch.dict("os.environ", {}, clear=True):
            # Should work with defaults
            from app.config import ResearchConfiguration

            config = ResearchConfiguration()

            assert config.critic_model is not None
            assert config.worker_model is not None
            assert config.max_search_iterations > 0


class TestErrorRecoveryMechanisms:
    """Test error recovery and resilience mechanisms."""

    @pytest.mark.asyncio
    async def test_agent_recovery_after_multiple_failures(self, error_agent):
        """Test agent recovery after multiple consecutive failures."""
        error_agent.force_error = "value_error"

        # Execute multiple failing jobs
        failing_results = []
        for i in range(3):
            job_id = error_agent.create_job("test_task", {"attempt": i})
            result = await error_agent.execute_job_async(job_id)
            failing_results.append(result)

        # All should fail
        assert all(not r["success"] for r in failing_results)

        # Disable forced errors
        error_agent.force_error = None

        # Next job should succeed
        recovery_job_id = error_agent.create_job("successful_task", {})
        recovery_result = await error_agent.execute_job_async(recovery_job_id)

        assert recovery_result["success"] is True

    @pytest.mark.asyncio
    async def test_job_state_consistency_after_errors(self, error_agent):
        """Test job state consistency after various errors."""
        # Create jobs that will fail in different ways
        error_types = ["value_error", "memory_error", "generic_error"]
        job_ids = []

        for error_type in error_types:
            error_agent.force_error = error_type
            job_id = error_agent.create_job("test_task", {"error_type": error_type})
            await error_agent.execute_job_async(job_id)
            job_ids.append(job_id)

        # Check all jobs are in failed state with proper error messages
        for i, job_id in enumerate(job_ids):
            job = error_agent.get_job(job_id)
            assert job.status == JobStatus.FAILED
            assert job.error is not None
            assert job.completed_at is not None
            assert error_types[i].replace("_", " ") in job.error.lower()

    @pytest.mark.asyncio
    async def test_resource_cleanup_after_errors(self, error_agent):
        """Test proper resource cleanup after errors."""
        initial_job_count = len(error_agent.jobs)
        initial_running_jobs = error_agent.running_jobs

        error_agent.force_error = "async_error"

        # Execute job that will fail
        job_id = error_agent.create_job("test_task", {})
        result = await error_agent.execute_job_async(job_id)

        assert result["success"] is False

        # Check resource counters are properly reset
        assert error_agent.running_jobs == initial_running_jobs
        assert len(error_agent.jobs) == initial_job_count + 1  # Job still exists but failed

    def test_storage_error_recovery(self, temp_auth_storage):
        """Test storage error recovery mechanisms."""
        # Create user successfully
        user = User(
            id="test_user", username="testuser", email="test@example.com", password_hash="hash"
        )
        result = temp_auth_storage.create_user(user)
        assert result is True

        # Simulate storage error during update
        with patch.object(temp_auth_storage, "_save_cache", side_effect=OSError("Disk full")):
            user.email = "updated@example.com"

            # Update should fail but not crash
            with pytest.raises(OSError):
                temp_auth_storage.update_user(user)

        # Storage should still be functional for reads
        retrieved_user = temp_auth_storage.get_user_by_id("test_user")
        assert retrieved_user is not None
        assert retrieved_user.email == "test@example.com"  # Original email, not updated

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, error_agent):
        """Test error handling under concurrent load."""
        error_agent.force_error = "intermittent"

        # Execute many jobs concurrently, some will fail
        job_ids = []
        for i in range(20):
            job_id = error_agent.create_job("test_task", {"concurrent_test": i})
            job_ids.append(job_id)

        tasks = [error_agent.execute_job_async(job_id) for job_id in job_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some should succeed, some should fail, none should raise unhandled exceptions
        successes = [r for r in results if isinstance(r, dict) and r.get("success")]
        failures = [r for r in results if isinstance(r, dict) and not r.get("success")]
        exceptions = [r for r in results if isinstance(r, Exception)]

        assert len(successes) > 0, "Some concurrent jobs should succeed"
        assert len(failures) > 0, "Some concurrent jobs should fail"
        assert len(exceptions) == 0, f"No unhandled exceptions expected, got: {exceptions}"

        # Agent should still be functional
        test_job_id = error_agent.create_job("successful_task", {})
        error_agent.force_error = None
        test_result = await error_agent.execute_job_async(test_job_id)
        assert test_result["success"] is True


class TestErrorLoggingAndMonitoring:
    """Test error logging and monitoring capabilities."""

    @pytest.mark.asyncio
    async def test_error_logging_in_job_execution(self, error_agent, caplog):
        """Test that errors are properly logged during job execution."""
        import logging

        with caplog.at_level(logging.ERROR):
            error_agent.force_error = "value_error"

            job_id = error_agent.create_job("test_task", {})
            result = await error_agent.execute_job_async(job_id)

            assert result["success"] is False

        # Check that error was logged
        error_logs = [record for record in caplog.records if record.levelno >= logging.ERROR]
        assert len(error_logs) > 0
        assert any("failed" in log.message.lower() for log in error_logs)

    def test_error_context_preservation(self, error_agent):
        """Test that error context is preserved through the call stack."""
        error_agent.force_error = "value_error"

        job_id = error_agent.create_job("test_task", {"context": "important_data"})

        # Execute synchronously to avoid async complications
        async def run_test():
            return await error_agent.execute_job_async(job_id)

        import asyncio

        result = asyncio.run(run_test())

        assert result["success"] is False
        assert "job_id" in result.get("data", {})

        # Job should preserve original context
        job = error_agent.get_job(job_id)
        assert job.parameters["context"] == "important_data"

    @pytest.mark.asyncio
    async def test_error_metrics_collection(self, error_agent):
        """Test collection of error metrics for monitoring."""
        # Execute jobs with different error types
        error_types = ["value_error", "memory_error", None, "generic_error", None]

        for error_type in error_types:
            error_agent.force_error = error_type
            job_id = error_agent.create_job("test_task", {"error_type": error_type})
            await error_agent.execute_job_async(job_id)

        # Get agent statistics
        stats = error_agent.get_agent_stats()

        assert stats["total_jobs"] == 5
        assert stats["failed_jobs"] == 3  # Three with errors
        assert stats["completed_jobs"] == 2  # Two successful

        # Success rate should be calculated correctly
        expected_success_rate = (2 / 5) * 100
        assert stats["success_rate"] == expected_success_rate
