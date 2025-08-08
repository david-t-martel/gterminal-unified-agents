"""
Integration tests for API endpoints.
Testing full request/response cycles with real components.
"""

from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.auth import get_current_user
from app.database import Base
from app.database import get_db
from app.main import app


class TestAPIEndpointsIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def test_client(self):
        """Create test client with mocked dependencies."""
        return TestClient(app)

    @pytest.fixture
    def test_db(self):
        """Create test database."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        def override_get_db():
            db = TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        return TestingSessionLocal()

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock = MagicMock()
        mock.get = MagicMock(return_value=None)
        mock.set = MagicMock(return_value=True)
        mock.delete = MagicMock(return_value=1)
        return mock

    @pytest.fixture
    def authenticated_client(self, test_client):
        """Create authenticated test client."""

        def override_auth():
            return {"user_id": "test_user", "email": "test@example.com"}

        app.dependency_overrides[get_current_user] = override_auth
        return test_client

    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_create_project(self, authenticated_client, test_db):
        """Test project creation endpoint."""
        project_data = {
            "name": "Test Project",
            "description": "Integration test project",
            "path": "/test/project",
            "language": "python",
            "framework": "fastapi",
        }

        response = authenticated_client.post("/api/projects", json=project_data)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Project"
        assert data["id"] is not None
        assert data["created_at"] is not None

    def test_get_projects(self, authenticated_client, test_db):
        """Test fetching projects list."""
        # Create test projects
        for i in range(3):
            project_data = {
                "name": f"Project {i}",
                "description": f"Description {i}",
                "path": f"/test/project{i}",
            }
            authenticated_client.post("/api/projects", json=project_data)

        response = authenticated_client.get("/api/projects")

        assert response.status_code == 200
        data = response.json()
        assert len(data["projects"]) == 3
        assert data["total"] == 3

    def test_get_project_by_id(self, authenticated_client, test_db):
        """Test fetching specific project."""
        # Create project
        project_data = {"name": "Specific Project", "path": "/specific"}
        create_response = authenticated_client.post("/api/projects", json=project_data)
        project_id = create_response.json()["id"]

        # Fetch project
        response = authenticated_client.get(f"/api/projects/{project_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Specific Project"
        assert data["id"] == project_id

    def test_update_project(self, authenticated_client, test_db):
        """Test project update."""
        # Create project
        project_data = {"name": "Original Name", "path": "/original"}
        create_response = authenticated_client.post("/api/projects", json=project_data)
        project_id = create_response.json()["id"]

        # Update project
        update_data = {"name": "Updated Name", "description": "Updated description"}
        response = authenticated_client.put(f"/api/projects/{project_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Updated description"

    def test_delete_project(self, authenticated_client, test_db):
        """Test project deletion."""
        # Create project
        project_data = {"name": "To Delete", "path": "/delete"}
        create_response = authenticated_client.post("/api/projects", json=project_data)
        project_id = create_response.json()["id"]

        # Delete project
        response = authenticated_client.delete(f"/api/projects/{project_id}")
        assert response.status_code == 204

        # Verify deletion
        get_response = authenticated_client.get(f"/api/projects/{project_id}")
        assert get_response.status_code == 404


class TestAgentEndpoints:
    """Test agent-related endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_agent_service(self):
        """Mock agent service."""
        with patch("app.api.agents.AgentService") as mock:
            service = mock.return_value
            service.create_agent = AsyncMock()
            service.get_agents = AsyncMock()
            service.execute_task = AsyncMock()
            yield service

    async def test_create_agent(self, client, mock_agent_service):
        """Test agent creation."""
        agent_data = {
            "type": "context_analysis",
            "name": "Test Agent",
            "config": {"model": "gemini-2.5-flash"},
        }

        mock_agent_service.create_agent.return_value = {
            "id": "agent_123",
            **agent_data,
            "status": "idle",
            "created_at": datetime.now().isoformat(),
        }

        response = client.post("/api/agents", json=agent_data)

        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "context_analysis"
        assert data["status"] == "idle"

    async def test_list_agents(self, client, mock_agent_service):
        """Test listing agents."""
        mock_agents = [
            {"id": "agent_1", "type": "context", "status": "idle"},
            {"id": "agent_2", "type": "codegen", "status": "running"},
            {"id": "agent_3", "type": "memory", "status": "idle"},
        ]

        mock_agent_service.get_agents.return_value = mock_agents

        response = client.get("/api/agents")

        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) == 3
        assert any(a["status"] == "running" for a in data["agents"])

    async def test_execute_agent_task(self, client, mock_agent_service):
        """Test task execution on agent."""
        task_data = {
            "agent_id": "agent_123",
            "task_type": "analyze",
            "parameters": {"path": "/test/project"},
        }

        mock_agent_service.execute_task.return_value = {
            "task_id": "task_456",
            "status": "completed",
            "result": {"analysis": "complete"},
        }

        response = client.post("/api/agents/execute", json=task_data)

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_456"
        assert data["status"] == "completed"

    async def test_get_agent_status(self, client, mock_agent_service):
        """Test getting agent status."""
        mock_agent_service.get_agent_status = AsyncMock(
            return_value={
                "id": "agent_123",
                "status": "running",
                "current_task": "analyzing",
                "memory_usage": "512MB",
                "uptime": "2h 15m",
            }
        )

        response = client.get("/api/agents/agent_123/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["current_task"] == "analyzing"


class TestWorkflowEndpoints:
    """Test workflow-related endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    async def test_create_workflow(self, client):
        """Test workflow creation."""
        workflow_data = {
            "name": "Test Workflow",
            "description": "Integration test workflow",
            "steps": [
                {"agent": "context", "action": "analyze"},
                {"agent": "codegen", "action": "generate"},
                {"agent": "test", "action": "validate"},
            ],
        }

        response = client.post("/api/workflows", json=workflow_data)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Workflow"
        assert len(data["steps"]) == 3

    async def test_execute_workflow(self, client):
        """Test workflow execution."""
        # Create workflow first
        workflow_data = {
            "name": "Execution Test",
            "steps": [{"agent": "context", "action": "analyze"}],
        }
        create_response = client.post("/api/workflows", json=workflow_data)
        workflow_id = create_response.json()["id"]

        # Execute workflow
        execution_data = {"workflow_id": workflow_id, "parameters": {}}
        response = client.post("/api/workflows/execute", json=execution_data)

        assert response.status_code in [200, 202]  # Accepted or OK
        data = response.json()
        assert "execution_id" in data
        assert data["status"] in ["started", "running", "queued"]

    async def test_get_workflow_status(self, client):
        """Test getting workflow execution status."""
        execution_id = "exec_123"

        with patch("app.api.workflows.get_execution_status") as mock_status:
            mock_status.return_value = {
                "execution_id": execution_id,
                "status": "completed",
                "progress": 100,
                "results": [{"step": 1, "result": "success"}],
            }

            response = client.get(f"/api/workflows/executions/{execution_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["progress"] == 100


class TestMCPEndpoints:
    """Test MCP server endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    async def test_list_mcp_servers(self, client):
        """Test listing MCP servers."""
        with patch("app.api.mcp.list_mcp_servers") as mock_list:
            mock_list.return_value = [
                {"name": "gemini-master-architect", "status": "running"},
                {"name": "code-generator-pro", "status": "running"},
                {"name": "memory-rag", "status": "stopped"},
            ]

            response = client.get("/api/mcp/servers")

            assert response.status_code == 200
            data = response.json()
            assert len(data["servers"]) == 3
            assert sum(1 for s in data["servers"] if s["status"] == "running") == 2

    async def test_start_mcp_server(self, client):
        """Test starting MCP server."""
        server_name = "memory-rag"

        with patch("app.api.mcp.start_mcp_server") as mock_start:
            mock_start.return_value = {"name": server_name, "status": "running", "pid": 12345}

            response = client.post(f"/api/mcp/servers/{server_name}/start")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"
            assert data["pid"] == 12345

    async def test_stop_mcp_server(self, client):
        """Test stopping MCP server."""
        server_name = "gemini-master-architect"

        with patch("app.api.mcp.stop_mcp_server") as mock_stop:
            mock_stop.return_value = {"name": server_name, "status": "stopped"}

            response = client.post(f"/api/mcp/servers/{server_name}/stop")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "stopped"

    async def test_mcp_server_health_check(self, client):
        """Test MCP server health check."""
        with patch("app.api.mcp.check_mcp_health") as mock_health:
            mock_health.return_value = {
                "healthy_servers": 2,
                "total_servers": 3,
                "servers": [
                    {"name": "server1", "healthy": True},
                    {"name": "server2", "healthy": True},
                    {"name": "server3", "healthy": False, "error": "Connection refused"},
                ],
            }

            response = client.get("/api/mcp/health")

            assert response.status_code == 200
            data = response.json()
            assert data["healthy_servers"] == 2
            assert data["total_servers"] == 3


class TestWebSocketEndpoints:
    """Test WebSocket endpoints for real-time updates."""

    @pytest.fixture
    async def websocket_client(self):
        """Create WebSocket test client."""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        return client

    async def test_websocket_connection(self, websocket_client):
        """Test WebSocket connection establishment."""
        with websocket_client.websocket_connect("/ws") as websocket:
            # Send initial message
            websocket.send_json({"type": "ping"})

            # Receive response
            data = websocket.receive_json()
            assert data["type"] == "pong"

    async def test_websocket_agent_updates(self, websocket_client):
        """Test receiving agent status updates via WebSocket."""
        with websocket_client.websocket_connect("/ws/agents") as websocket:
            # Subscribe to agent updates
            websocket.send_json({"type": "subscribe", "channel": "agent_status"})

            # Simulate agent status change
            with patch("app.websocket.broadcast_agent_status") as mock_broadcast:
                mock_broadcast.return_value = None

                # Trigger status update
                mock_broadcast({"agent_id": "agent_123", "status": "running", "task": "analyzing"})

            # Should receive update
            data = websocket.receive_json()
            assert data["type"] == "agent_status"

    async def test_websocket_workflow_progress(self, websocket_client):
        """Test workflow progress updates via WebSocket."""
        with websocket_client.websocket_connect("/ws/workflows") as websocket:
            # Subscribe to workflow updates
            websocket.send_json({"type": "subscribe", "workflow_id": "workflow_456"})

            # Simulate progress update
            with patch("app.websocket.send_workflow_progress") as mock_progress:
                mock_progress(
                    {"workflow_id": "workflow_456", "progress": 50, "current_step": "codegen"}
                )

            # Should receive progress update
            data = websocket.receive_json()
            assert data["progress"] == 50


class TestErrorHandling:
    """Test error handling in API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_404_not_found(self, client):
        """Test 404 error handling."""
        response = client.get("/api/nonexistent/endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_400_bad_request(self, client):
        """Test 400 error for invalid data."""
        invalid_data = {"invalid": "data"}
        response = client.post("/api/projects", json=invalid_data)
        assert response.status_code in [400, 422]  # Bad request or validation error

    def test_401_unauthorized(self, client):
        """Test 401 error for unauthorized access."""
        response = client.get("/api/protected/resource")
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    def test_500_internal_error(self, client):
        """Test 500 error handling."""
        with patch("app.api.projects.get_projects") as mock_get:
            mock_get.side_effect = Exception("Database error")

            response = client.get("/api/projects")
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data


class TestPagination:
    """Test pagination in list endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_pagination_parameters(self, client):
        """Test pagination with limit and offset."""
        response = client.get("/api/projects?limit=10&offset=20")
        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_pagination_defaults(self, client):
        """Test default pagination values."""
        response = client.get("/api/agents")
        assert response.status_code == 200
        data = response.json()
        # Should have default limit
        assert len(data["agents"]) <= 50  # Assuming 50 is default


class TestFiltering:
    """Test filtering in list endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_filter_by_status(self, client):
        """Test filtering by status."""
        response = client.get("/api/agents?status=running")
        assert response.status_code == 200
        data = response.json()
        # All returned agents should have running status
        assert all(a.get("status") == "running" for a in data.get("agents", []))

    def test_filter_by_type(self, client):
        """Test filtering by type."""
        response = client.get("/api/agents?type=context_analysis")
        assert response.status_code == 200
        data = response.json()
        # All returned agents should be of specified type
        assert all(a.get("type") == "context_analysis" for a in data.get("agents", []))

    def test_combined_filters(self, client):
        """Test combining multiple filters."""
        response = client.get("/api/projects?language=python&framework=fastapi")
        assert response.status_code == 200
        # Response should respect both filters


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
