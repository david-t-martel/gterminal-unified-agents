"""
Example unit tests demonstrating the unified testing framework.

This module shows how to use the framework's fixtures, helpers,
and utilities for effective testing.
"""

import asyncio
import time

import pytest

from tests.framework import AsyncTestHelper
from tests.framework import PerformanceAssertions
from tests.framework import PerformanceBenchmark
from tests.framework import TestHelper


class TestFrameworkExample:
    """Example tests using the unified testing framework."""

    def test_basic_fixture_usage(self, test_config, test_data_factory):
        """Test basic usage of framework fixtures."""
        # Test configuration fixture
        assert test_config.google_cloud_project == "test-project-id"
        assert test_config.environment == "test"

        # Test data factory
        request = test_data_factory.create_agent_request(
            message="Hello, world!", user_id="test-user-123"
        )

        assert request["message"] == "Hello, world!"
        assert request["user_id"] == "test-user-123"
        assert "session_id" in request
        assert "metadata" in request

    def test_helper_functions(self):
        """Test helper function usage."""
        # Test response structure validation
        valid_response = {
            "content": "Test response",
            "status": "success",
            "metadata": {"tokens_used": 50},
        }

        # This should not raise an exception
        TestHelper.assert_agent_response_valid(valid_response)

        # Test invalid response
        invalid_response = {
            "content": "Test response",
            # Missing required 'status' field
        }

        with pytest.raises(AssertionError, match="Required field 'status' missing"):
            TestHelper.assert_agent_response_valid(invalid_response)

    def test_dict_subset_comparison(self):
        """Test dictionary subset comparison utility."""
        full_dict = {
            "level1": {"level2": {"key1": "value1", "key2": "value2"}, "other": "data"},
            "top_level": "value",
        }

        subset = {"level1": {"level2": {"key1": "value1"}}}

        assert TestHelper.compare_dict_subset(subset, full_dict)

        # Test non-matching subset
        non_matching = {"level1": {"level2": {"key1": "different_value"}}}

        assert not TestHelper.compare_dict_subset(non_matching, full_dict)

    @pytest.mark.asyncio
    async def test_async_helpers(self):
        """Test async helper functions."""
        # Test wait for condition
        counter = {"value": 0}

        def increment_counter():
            counter["value"] += 1
            return counter["value"] >= 3

        # This should succeed quickly
        result = await AsyncTestHelper.wait_for_condition(
            increment_counter, timeout=1.0, interval=0.1
        )
        assert result is True
        assert counter["value"] >= 3

    @pytest.mark.asyncio
    async def test_performance_measurement(self):
        """Test performance measurement utilities."""

        async def mock_async_operation():
            await asyncio.sleep(0.1)  # Simulate work
            return "operation complete"

        performance_data = await AsyncTestHelper.measure_async_performance(mock_async_operation)

        assert "result" in performance_data
        assert performance_data["result"] == "operation complete"
        assert "response_time_ms" in performance_data
        assert performance_data["response_time_ms"] >= 100  # At least 100ms
        assert "timestamp" in performance_data

    @pytest.mark.performance
    async def test_load_simulation(self):
        """Test load simulation capabilities."""
        call_count = {"value": 0}

        async def mock_api_call():
            call_count["value"] += 1
            await asyncio.sleep(0.01)  # Small delay
            return f"response_{call_count['value']}"

        load_result = await AsyncTestHelper.simulate_load(
            mock_api_call, concurrent_requests=3, total_requests=10
        )

        assert load_result["success"] is True
        assert load_result["total_requests"] == 10
        assert load_result["successful_requests"] == 10
        assert load_result["error_count"] == 0
        assert "response_times" in load_result
        assert load_result["response_times"]["mean"] > 0

    def test_performance_benchmark_context(self):
        """Test performance benchmark context manager."""
        benchmark = PerformanceBenchmark("test_benchmark")

        with benchmark.measure_performance("test_operation"):
            time.sleep(0.1)  # Simulate work

        assert len(benchmark.results) == 1
        result = benchmark.results[0]
        assert result["operation"] == "test_operation"
        assert result["metrics"].response_time_ms >= 100

        # Test summary stats
        summary = benchmark.get_summary_stats()
        assert summary["total_operations"] == 1
        assert "response_times" in summary

    @pytest.mark.asyncio
    async def test_mock_integrations(self, mock_google_client, mock_redis_client):
        """Test mock integrations."""
        # Test Google client mock
        mock_instance = mock_google_client["instance"]
        response = await mock_instance.generate_content("Test prompt")
        assert response.text == "Test response"

        # Test Redis client mock
        await mock_redis_client.set("test_key", "test_value")
        mock_redis_client.set.assert_called_once_with("test_key", "test_value")

        mock_redis_client.get.return_value = "test_value"
        value = await mock_redis_client.get("test_key")
        assert value == "test_value"

    def test_performance_assertions(self):
        """Test performance assertion utilities."""
        from tests.framework.performance import PerformanceMetrics

        # Create test metrics
        good_metrics = PerformanceMetrics(
            response_time_ms=250.0,
            memory_usage_mb=100.0,
            cpu_usage_percent=50.0,
            memory_peak_mb=120.0,
            gc_collections=5,
            timestamp=time.time(),
        )

        # These should pass
        PerformanceAssertions.assert_response_time(good_metrics, 500.0)
        PerformanceAssertions.assert_memory_usage(good_metrics, 200.0)

        # Create bad metrics
        bad_metrics = PerformanceMetrics(
            response_time_ms=1500.0,  # Too high
            memory_usage_mb=600.0,  # Too high
            cpu_usage_percent=95.0,
            memory_peak_mb=650.0,
            gc_collections=50,
            timestamp=time.time(),
        )

        # These should fail
        with pytest.raises(AssertionError, match="Response time .* exceeds limit"):
            PerformanceAssertions.assert_response_time(bad_metrics, 1000.0)

        with pytest.raises(AssertionError, match="Memory usage .* exceeds limit"):
            PerformanceAssertions.assert_memory_usage(bad_metrics, 500.0)

    def test_temp_directory_fixture(self, temp_dir):
        """Test temporary directory fixture."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, test!")

        assert test_file.exists()
        assert test_file.read_text() == "Hello, test!"

    def test_security_context(self, security_test_context):
        """Test security testing context."""
        assert "safe_inputs" in security_test_context
        assert "malicious_inputs" in security_test_context
        assert "expected_sanitized" in security_test_context

        # Test that we have malicious inputs for testing
        malicious = security_test_context["malicious_inputs"]
        assert len(malicious) > 0
        assert any("script" in inp for inp in malicious)
        assert any("DROP TABLE" in inp for inp in malicious)

    def test_integration_config(self, integration_test_config):
        """Test integration test configuration."""
        assert "timeout_seconds" in integration_test_config
        assert "endpoints" in integration_test_config
        assert "expected_status_codes" in integration_test_config

        # Verify endpoint configurations
        endpoints = integration_test_config["endpoints"]
        assert "health" in endpoints
        assert "agent" in endpoints
        assert "mcp" in endpoints


class TestFrameworkPerformance:
    """Performance tests for the testing framework itself."""

    @pytest.mark.performance
    def test_fixture_creation_performance(self, test_data_factory):
        """Test that fixture creation is fast."""
        benchmark = PerformanceBenchmark("fixture_performance")

        with benchmark.measure_performance("create_agent_requests"):
            # Create many requests quickly
            requests = [
                test_data_factory.create_agent_request(
                    message=f"Test message {i}", user_id=f"user_{i}"
                )
                for i in range(100)
            ]

        assert len(requests) == 100

        # Should be very fast
        summary = benchmark.get_summary_stats()
        assert summary["response_times"]["mean_ms"] < 100  # Less than 100ms

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_async_helper_performance(self):
        """Test async helper performance."""

        async def fast_operation():
            return "fast_result"

        # Measure many calls
        load_result = await AsyncTestHelper.simulate_load(
            fast_operation, concurrent_requests=10, total_requests=100
        )

        assert load_result["success"] is True
        assert load_result["successful_requests"] == 100

        # Should have good throughput
        response_times = load_result["response_times"]
        assert response_times["mean"] < 10  # Less than 10ms average
        assert response_times["p95"] < 50  # P95 under 50ms


@pytest.mark.integration
class TestFrameworkIntegration:
    """Integration tests for framework components."""

    @pytest.mark.asyncio
    async def test_full_test_workflow(
        self, test_data_factory, mock_google_client, mock_redis_client
    ):
        """Test complete workflow using multiple framework components."""
        # Create test data
        request = test_data_factory.create_agent_request(
            message="Integration test", user_id="integration_user"
        )

        # Simulate agent processing
        mock_instance = mock_google_client["instance"]
        ai_response = await mock_instance.generate_content(request["message"])

        # Store in cache
        cache_key = f"response:{request['user_id']}"
        await mock_redis_client.set(cache_key, ai_response.text)

        # Verify cache
        mock_redis_client.get.return_value = ai_response.text
        cached_response = await mock_redis_client.get(cache_key)

        assert cached_response == "Test response"

        # Verify all interactions
        mock_instance.generate_content.assert_called_once_with("Integration test")
        mock_redis_client.set.assert_called_once_with(cache_key, "Test response")
        mock_redis_client.get.assert_called_once_with(cache_key)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
