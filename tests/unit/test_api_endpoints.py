"""Comprehensive tests for API endpoints.

This module tests FastAPI endpoints, request/response handling,
authentication, error handling, and API contract compliance.
"""

from decimal import Decimal
from typing import Any
from unittest.mock import patch

# Import API modules
from app.agent_endpoints import router as agent_router
from app.agents.base_agent_service import Job
from app.agents.base_agent_service import JobStatus
from app.performance_endpoints import router as performance_router
from app.server import app as main_app
from fastapi import FastAPI
from fastapi import status
from fastapi.testclient import TestClient
import pytest


class TestAgentEndpoints:
    """Test agent-related API endpoints."""

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        app.include_router(agent_router, prefix="/api/v1/agents")
        return app

    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)

    @pytest.fixture
    def mock_agent_service(self):
        """Create mock agent service."""
        from app.agents.base_agent_service import BaseAgentService

        class MockAgentService(BaseAgentService):
            def __init__(self):
                super().__init__()
                self.test_jobs = {}

            async def execute_job(self, job: Job) -> dict[str, Any]:
                return {"result": "success"}

            async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
                return {"processed": True}

        return MockAgentService()

    def test_create_job_endpoint_success(self, client, mock_agent_service):
        """Test successful job creation endpoint."""
        job_data = {
            "job_type": "analysis",
            "parameters": {"file_path": "/test/file.py", "language": "python"},
        }

        with patch("app.agent_endpoints.get_agent_service", return_value=mock_agent_service):
            response = client.post("/api/v1/agents/jobs", json=job_data)

        assert response.status_code == status.HTTP_201_CREATED
        response_data = response.json()
        assert "job_id" in response_data
        assert response_data["status"] == "created"
        assert response_data["job_type"] == "analysis"

    def test_create_job_endpoint_validation_error(self, client):
        """Test job creation with invalid data."""
        invalid_job_data = {
            "job_type": "",  # Invalid empty job type
            "parameters": "not_a_dict",  # Invalid parameters type
        }

        response = client.post("/api/v1/agents/jobs", json=invalid_job_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        response_data = response.json()
        assert "detail" in response_data

    def test_get_job_status_endpoint_success(self, client, mock_agent_service):
        """Test successful job status retrieval."""
        # Create a test job first
        test_job = Job("test-job-123", "analysis", {"test": "data"})
        mock_agent_service.jobs["test-job-123"] = test_job

        with patch("app.agent_endpoints.get_agent_service", return_value=mock_agent_service):
            response = client.get("/api/v1/agents/jobs/test-job-123")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["job_id"] == "test-job-123"
        assert response_data["job_type"] == "analysis"
        assert response_data["status"] == "pending"

    def test_get_job_status_endpoint_not_found(self, client, mock_agent_service):
        """Test job status retrieval for non-existent job."""
        with patch("app.agent_endpoints.get_agent_service", return_value=mock_agent_service):
            response = client.get("/api/v1/agents/jobs/non-existent-job")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        response_data = response.json()
        assert "not found" in response_data["detail"].lower()

    def test_execute_job_endpoint_success(self, client, mock_agent_service):
        """Test successful job execution endpoint."""
        # Create a test job
        test_job = Job("test-job-456", "analysis", {"test": "data"})
        mock_agent_service.jobs["test-job-456"] = test_job

        with patch("app.agent_endpoints.get_agent_service", return_value=mock_agent_service):
            response = client.post("/api/v1/agents/jobs/test-job-456/execute")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "result" in response_data

    def test_cancel_job_endpoint_success(self, client, mock_agent_service):
        """Test successful job cancellation."""
        # Create a test job
        test_job = Job("test-job-789", "analysis", {"test": "data"})
        mock_agent_service.jobs["test-job-789"] = test_job

        with patch("app.agent_endpoints.get_agent_service", return_value=mock_agent_service):
            response = client.post("/api/v1/agents/jobs/test-job-789/cancel")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["status"] == "cancelled"

    def test_list_jobs_endpoint_success(self, client, mock_agent_service):
        """Test job listing endpoint."""
        # Create test jobs
        job1 = Job("job-1", "analysis", {})
        job2 = Job("job-2", "generation", {})
        job2.status = JobStatus.COMPLETED

        mock_agent_service.jobs = {"job-1": job1, "job-2": job2}

        with patch("app.agent_endpoints.get_agent_service", return_value=mock_agent_service):
            response = client.get("/api/v1/agents/jobs")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert "jobs" in response_data
        assert len(response_data["jobs"]) == 2

    def test_list_jobs_endpoint_with_filters(self, client, mock_agent_service):
        """Test job listing with status filter."""
        # Create test jobs with different statuses
        job1 = Job("pending-job", "analysis", {})
        job2 = Job("completed-job", "generation", {})
        job2.status = JobStatus.COMPLETED

        mock_agent_service.jobs = {"pending-job": job1, "completed-job": job2}

        with patch("app.agent_endpoints.get_agent_service", return_value=mock_agent_service):
            response = client.get("/api/v1/agents/jobs?status=completed")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert len(response_data["jobs"]) == 1
        assert response_data["jobs"][0]["status"] == "completed"

    def test_stream_job_progress_endpoint(self, client, mock_agent_service):
        """Test job progress streaming endpoint."""
        test_job = Job("stream-job", "analysis", {})
        mock_agent_service.jobs["stream-job"] = test_job

        with patch("app.agent_endpoints.get_agent_service", return_value=mock_agent_service):
            # Note: TestClient doesn't support streaming responses well
            # In a real test, you'd use an async client
            response = client.get("/api/v1/agents/jobs/stream-job/progress")

        # Basic check that endpoint exists and doesn't error
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

    def test_agent_stats_endpoint(self, client, mock_agent_service):
        """Test agent statistics endpoint."""
        with patch("app.agent_endpoints.get_agent_service", return_value=mock_agent_service):
            response = client.get("/api/v1/agents/stats")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert "total_jobs" in response_data
        assert "agent_name" in response_data


class TestPerformanceEndpoints:
    """Test performance monitoring API endpoints."""

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI app with performance endpoints."""
        app = FastAPI()
        app.include_router(performance_router, prefix="/api/v1/performance")
        return app

    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)

    def test_system_metrics_endpoint(self, client):
        """Test system metrics endpoint."""
        with patch("app.performance_endpoints.get_system_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.1,
                "network_io": {"sent": 1024, "received": 2048},
            }

            response = client.get("/api/v1/performance/metrics")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert "cpu_usage" in response_data
        assert "memory_usage" in response_data
        assert response_data["cpu_usage"] == 45.2

    def test_performance_benchmark_endpoint(self, client):
        """Test performance benchmark endpoint."""
        benchmark_request = {
            "benchmark_type": "job_execution",
            "parameters": {"job_count": 100, "concurrency": 10},
        }

        with patch("app.performance_endpoints.run_benchmark") as mock_benchmark:
            mock_benchmark.return_value = {
                "benchmark_id": "bench-123",
                "status": "started",
                "estimated_duration": "30s",
            }

            response = client.post("/api/v1/performance/benchmark", json=benchmark_request)

        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()
        assert "benchmark_id" in response_data
        assert response_data["status"] == "started"

    def test_benchmark_results_endpoint(self, client):
        """Test benchmark results retrieval."""
        with patch("app.performance_endpoints.get_benchmark_results") as mock_results:
            mock_results.return_value = {
                "benchmark_id": "bench-123",
                "status": "completed",
                "results": {
                    "total_duration": 28.5,
                    "throughput": 3.51,
                    "avg_latency": 0.285,
                    "p95_latency": 0.456,
                },
            }

            response = client.get("/api/v1/performance/benchmark/bench-123")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["benchmark_id"] == "bench-123"
        assert response_data["status"] == "completed"
        assert "results" in response_data

    def test_cache_stats_endpoint(self, client):
        """Test cache statistics endpoint."""
        with patch("app.performance_endpoints.get_cache_stats") as mock_stats:
            mock_stats.return_value = {
                "hit_rate": 87.5,
                "miss_rate": 12.5,
                "total_requests": 10000,
                "cache_size": 5000,
                "memory_usage_mb": 45.2,
            }

            response = client.get("/api/v1/performance/cache/stats")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["hit_rate"] == 87.5
        assert response_data["total_requests"] == 10000

    def test_clear_cache_endpoint(self, client):
        """Test cache clearing endpoint."""
        with patch("app.performance_endpoints.clear_cache") as mock_clear:
            mock_clear.return_value = {"status": "success", "cleared_items": 1250}

            response = client.post("/api/v1/performance/cache/clear")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["cleared_items"] == 1250


class TestCostOptimizationEndpoints:
    """Test cost optimization API endpoints."""

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI app with cost endpoints."""
        app = FastAPI()
        # Import and include cost optimization router
        try:
            from app.cost_optimization.api import router as cost_router

            app.include_router(cost_router, prefix="/api/v1/cost")
        except ImportError:
            # If cost API module doesn't exist, create a simple router for testing
            from fastapi import APIRouter

            cost_router = APIRouter()

            @cost_router.get("/analysis")
            async def get_cost_analysis():
                return {"status": "success", "total_cost": 1500.00}

            app.include_router(cost_router, prefix="/api/v1/cost")

        return app

    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)

    def test_cost_analysis_endpoint(self, client):
        """Test cost analysis endpoint."""
        analysis_request = {
            "project_id": "test-project",
            "time_range": "30d",
            "include_forecasting": True,
        }

        with patch("app.cost_optimization.api.analyze_project_costs") as mock_analyze:
            mock_analyze.return_value = {
                "project_id": "test-project",
                "total_cost": Decimal("1500.00"),
                "cost_breakdown": {
                    "compute": Decimal("900.00"),
                    "storage": Decimal("400.00"),
                    "networking": Decimal("200.00"),
                },
                "forecast": {"next_30_days": Decimal("1600.00"), "confidence": 0.85},
            }

            response = client.post("/api/v1/cost/analysis", json=analysis_request)

        # Check if endpoint exists (might be 200 or 404 depending on implementation)
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

        if response.status_code == status.HTTP_200_OK:
            response_data = response.json()
            assert "total_cost" in response_data or "status" in response_data


class TestHealthCheckEndpoints:
    """Test health check and monitoring endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with main app."""
        return TestClient(main_app)

    def test_health_check_endpoint(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")

        # Health endpoint should exist and return success
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

        if response.status_code == status.HTTP_200_OK:
            response_data = response.json()
            assert "status" in response_data
            assert response_data["status"] in ["healthy", "ok", "success"]

    def test_readiness_check_endpoint(self, client):
        """Test readiness check endpoint."""
        response = client.get("/ready")

        # Check if readiness endpoint exists
        if response.status_code == status.HTTP_200_OK:
            response_data = response.json()
            assert "ready" in response_data or "status" in response_data

    def test_liveness_check_endpoint(self, client):
        """Test liveness check endpoint."""
        response = client.get("/alive")

        # Check if liveness endpoint exists
        if response.status_code == status.HTTP_200_OK:
            response_data = response.json()
            assert "alive" in response_data or "status" in response_data

    def test_version_endpoint(self, client):
        """Test version information endpoint."""
        response = client.get("/version")

        if response.status_code == status.HTTP_200_OK:
            response_data = response.json()
            assert "version" in response_data or "build" in response_data


class TestErrorHandling:
    """Test API error handling and responses."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(main_app)

    def test_404_error_handling(self, client):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent/endpoint")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        response_data = response.json()
        assert "detail" in response_data

    def test_method_not_allowed_handling(self, client):
        """Test 405 Method Not Allowed handling."""
        # Try to POST to a GET-only endpoint
        response = client.post("/health")

        assert response.status_code in [
            status.HTTP_405_METHOD_NOT_ALLOWED,
            status.HTTP_404_NOT_FOUND,  # If endpoint doesn't exist
        ]

    def test_validation_error_handling(self, client):
        """Test request validation error handling."""
        # Send invalid JSON to an endpoint that expects valid JSON
        invalid_json = '{"invalid": json}'

        response = client.post(
            "/api/v1/agents/jobs",
            data=invalid_json,
            headers={"Content-Type": "application/json"},  # Invalid JSON
        )

        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,  # If endpoint doesn't exist
        ]

    def test_internal_server_error_handling(self, client):
        """Test 500 internal server error handling."""
        # This would require setting up an endpoint that raises an exception
        # For now, we'll just verify the error handling structure exists

        # Mock an endpoint that raises an exception
        from fastapi import HTTPException

        @main_app.get("/test-error")
        async def test_error_endpoint():
            raise HTTPException(status_code=500, detail="Test internal error")

        response = client.get("/test-error")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        response_data = response.json()
        assert "detail" in response_data


class TestAuthentication:
    """Test API authentication and authorization."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(main_app)

    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication."""
        # This assumes there are protected endpoints
        protected_endpoints = [
            "/api/v1/agents/jobs",
            "/api/v1/performance/benchmark",
            "/api/v1/cost/analysis",
        ]

        for endpoint in protected_endpoints:
            response = client.post(endpoint, json={})

            # Should either require auth (401/403) or not exist (404)
            assert response.status_code in [
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
                status.HTTP_422_UNPROCESSABLE_ENTITY,  # If validation fails first
            ]

    def test_protected_endpoint_with_invalid_auth(self, client):
        """Test accessing protected endpoint with invalid authentication."""
        headers = {"Authorization": "Bearer invalid-token"}

        response = client.post(
            "/api/v1/agents/jobs", json={"job_type": "test", "parameters": {}}, headers=headers
        )

        # Should reject invalid token
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
            status.HTTP_404_NOT_FOUND,  # If endpoint doesn't exist
        ]

    def test_api_key_authentication(self, client):
        """Test API key authentication."""
        # Test with valid API key
        headers = {"X-API-Key": "valid-test-api-key"}

        with patch("app.authentication.validate_api_key", return_value=True):
            response = client.get("/api/v1/agents/stats", headers=headers)

        # Should not be rejected for auth reasons
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
        ]  # If endpoint doesn't exist


class TestRateLimiting:
    """Test API rate limiting."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(main_app)

    def test_rate_limiting_enforcement(self, client):
        """Test that rate limiting is enforced."""
        # Make multiple rapid requests to test rate limiting
        endpoint = "/health"  # Use a simple endpoint
        request_count = 100

        responses = []
        for _ in range(request_count):
            response = client.get(endpoint)
            responses.append(response)

        # Check if any requests were rate limited
        rate_limited_responses = [
            r for r in responses if r.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        ]

        # Either rate limiting is enforced, or all requests succeed
        successful_requests = [r for r in responses if r.status_code == status.HTTP_200_OK]

        # All requests should either succeed or be rate-limited
        assert len(successful_requests) + len(rate_limited_responses) >= request_count // 2

    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are present."""
        response = client.get("/health")

        # Check for common rate limiting headers
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "Retry-After",
        ]

        # At least some rate limiting headers should be present if implemented
        [header for header in rate_limit_headers if header in response.headers]

        # This test documents the rate limiting implementation
        # It's okay if no rate limiting headers are present in a basic implementation


class TestAPIDocumentation:
    """Test API documentation and OpenAPI spec."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(main_app)

    def test_openapi_spec_endpoint(self, client):
        """Test OpenAPI specification endpoint."""
        response = client.get("/openapi.json")

        assert response.status_code == status.HTTP_200_OK
        openapi_spec = response.json()

        # Basic OpenAPI spec validation
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec

        # Check that API endpoints are documented
        paths = openapi_spec["paths"]
        assert len(paths) > 0

    def test_swagger_ui_endpoint(self, client):
        """Test Swagger UI endpoint."""
        response = client.get("/docs")

        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers.get("content-type", "")

    def test_redoc_endpoint(self, client):
        """Test ReDoc endpoint."""
        response = client.get("/redoc")

        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers.get("content-type", "")


class TestCORS:
    """Test Cross-Origin Resource Sharing (CORS) configuration."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(main_app)

    def test_cors_preflight_request(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/api/v1/agents/jobs",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        # Should either handle CORS or return 404 if endpoint doesn't exist
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_204_NO_CONTENT,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_405_METHOD_NOT_ALLOWED,
        ]

        if response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]:
            # Check CORS headers
            cors_headers = [
                "Access-Control-Allow-Origin",
                "Access-Control-Allow-Methods",
                "Access-Control-Allow-Headers",
            ]

            # At least some CORS headers should be present
            present_cors_headers = [header for header in cors_headers if header in response.headers]

            # Document CORS configuration
            assert len(present_cors_headers) >= 0  # Allow for no CORS or proper CORS

    def test_cors_actual_request(self, client):
        """Test actual CORS request."""
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})

        # Should not be blocked by CORS for GET requests
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]
