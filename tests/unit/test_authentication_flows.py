"""
Unit tests for authentication flows and endpoints.
Tests login, logout, token management, and API key operations.
"""

from datetime import UTC
from datetime import datetime
from unittest.mock import Mock
from unittest.mock import patch

from app.automation.auth_endpoints import auth_router
from app.automation.auth_models import APIKey
from app.automation.auth_models import AuthContext
from app.automation.auth_models import User
from app.automation.auth_models import UserRole
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

# Create test FastAPI app
test_app = FastAPI()
test_app.include_router(auth_router)

client = TestClient(test_app)


@pytest.fixture
def sample_user():
    """Create sample user for testing."""
    return User(
        id="user_123",
        username="testuser",
        email="test@example.com",
        password_hash="hashed_password",
        role=UserRole.USER,
        permissions={"api:read", "api:write"},
    )


@pytest.fixture
def sample_admin_user():
    """Create sample admin user for testing."""
    return User(
        id="admin_123",
        username="adminuser",
        email="admin@example.com",
        password_hash="hashed_admin_password",
        role=UserRole.ADMIN,
        permissions={"admin:users", "admin:system", "api:read", "api:write"},
    )


@pytest.fixture
def sample_api_key():
    """Create sample API key for testing."""
    return APIKey(
        id="key_123",
        user_id="user_123",
        name="Test API Key",
        key_hash="hashed_key",
        key_prefix="testkey1",
        scopes={"api:read", "api:write"},
    )


@pytest.fixture
def mock_auth_storage():
    """Mock authentication storage."""
    with patch("app.automation.auth_endpoints.auth_storage") as mock_storage:
        yield mock_storage


@pytest.fixture
def mock_jwt_manager():
    """Mock JWT manager."""
    with patch("app.automation.auth_endpoints.jwt_manager") as mock_jwt:
        mock_jwt.create_token_pair.return_value = Mock(
            access_token="access_token_123",
            refresh_token="refresh_token_123",
            token_type="bearer",
            expires_in=3600,
        )
        mock_jwt.decode_token.return_value = {
            "sub": "user_123",
            "username": "testuser",
            "type": "refresh",
        }
        yield mock_jwt


@pytest.fixture
def mock_password_manager():
    """Mock password manager."""
    with patch("app.automation.auth_endpoints.password_manager") as mock_pm:
        mock_pm.hash_password.return_value = "hashed_password"
        mock_pm.verify_password.return_value = True
        yield mock_pm


@pytest.fixture
def mock_security_monitor():
    """Mock security monitor."""
    with patch("app.automation.auth_endpoints.security_monitor") as mock_monitor:
        yield mock_monitor


class TestLoginEndpoint:
    """Test login endpoint functionality."""

    def test_successful_login(
        self, mock_auth_storage, mock_jwt_manager, mock_security_monitor, sample_user
    ):
        """Test successful user login."""
        # Mock storage response
        mock_auth_storage.authenticate_user.return_value = sample_user

        # Login request
        login_data = {"username": "testuser", "password": "password123"}
        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 200
        data = response.json()

        assert data["access_token"] == "access_token_123"
        assert data["refresh_token"] == "refresh_token_123"
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == "testuser"
        assert data["user"]["email"] == "test@example.com"

        # Verify calls
        mock_auth_storage.authenticate_user.assert_called_once_with("testuser", "password123")
        mock_jwt_manager.create_token_pair.assert_called_once_with(sample_user)

    def test_invalid_credentials(self, mock_auth_storage, mock_security_monitor):
        """Test login with invalid credentials."""
        # Mock authentication failure
        mock_auth_storage.authenticate_user.return_value = None

        login_data = {"username": "testuser", "password": "wrongpassword"}
        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 401
        data = response.json()
        assert "Invalid username or password" in data["detail"]

        # Verify security logging
        mock_security_monitor.log_failed_auth.assert_called_once()

    def test_login_missing_credentials(self):
        """Test login with missing credentials."""
        # Missing password
        response = client.post("/auth/login", json={"username": "testuser"})
        assert response.status_code == 422

        # Missing username
        response = client.post("/auth/login", json={"password": "password123"})
        assert response.status_code == 422

        # Empty request
        response = client.post("/auth/login", json={})
        assert response.status_code == 422

    def test_login_service_error(self, mock_auth_storage):
        """Test login with service error."""
        # Mock storage to raise exception
        mock_auth_storage.authenticate_user.side_effect = Exception("Database error")

        login_data = {"username": "testuser", "password": "password123"}
        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 500
        data = response.json()
        assert "Authentication service error" in data["detail"]


class TestTokenRefreshEndpoint:
    """Test token refresh endpoint functionality."""

    def test_successful_token_refresh(self, mock_auth_storage, mock_jwt_manager, sample_user):
        """Test successful token refresh."""
        # Mock JWT decoding and user retrieval
        mock_auth_storage.get_user_by_id.return_value = sample_user

        refresh_data = {"refresh_token": "valid_refresh_token"}
        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == 200
        data = response.json()

        assert data["access_token"] == "access_token_123"
        assert data["refresh_token"] == "refresh_token_123"
        assert data["user"]["username"] == "testuser"

        # Verify calls
        mock_jwt_manager.decode_token.assert_called_once_with("valid_refresh_token")
        mock_auth_storage.get_user_by_id.assert_called_once_with("user_123")

    def test_invalid_refresh_token(self, mock_jwt_manager, mock_security_monitor):
        """Test refresh with invalid token."""
        # Mock invalid token
        mock_jwt_manager.decode_token.return_value = None

        refresh_data = {"refresh_token": "invalid_token"}
        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == 401
        data = response.json()
        assert "Invalid refresh token" in data["detail"]

        mock_security_monitor.log_failed_auth.assert_called_once()

    def test_refresh_token_wrong_type(self, mock_jwt_manager, mock_security_monitor):
        """Test refresh with access token instead of refresh token."""
        # Mock access token payload
        mock_jwt_manager.decode_token.return_value = {
            "sub": "user_123",
            "type": "access",
        }  # Wrong type

        refresh_data = {"refresh_token": "access_token"}
        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == 401
        data = response.json()
        assert "Invalid refresh token" in data["detail"]

    def test_refresh_token_inactive_user(
        self, mock_auth_storage, mock_jwt_manager, mock_security_monitor
    ):
        """Test refresh with inactive user."""
        # Mock inactive user
        inactive_user = User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            password_hash="hash",
            is_active=False,
        )
        mock_auth_storage.get_user_by_id.return_value = inactive_user

        refresh_data = {"refresh_token": "valid_refresh_token"}
        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == 401
        data = response.json()
        assert "User account is inactive" in data["detail"]

    def test_refresh_token_nonexistent_user(
        self, mock_auth_storage, mock_jwt_manager, mock_security_monitor
    ):
        """Test refresh with non-existent user."""
        mock_auth_storage.get_user_by_id.return_value = None

        refresh_data = {"refresh_token": "valid_refresh_token"}
        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == 401
        data = response.json()
        assert "User account is inactive" in data["detail"]


class TestLogoutEndpoint:
    """Test logout endpoint functionality."""

    def test_successful_logout(self, mock_jwt_manager):
        """Test successful logout."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="user_123", username="testuser", permissions=set()
            )

            # Mock request with Authorization header
            headers = {"Authorization": "Bearer access_token_123"}
            response = client.post("/auth/logout", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Successfully logged out"

            # Verify token blacklisting
            mock_jwt_manager.blacklist_token.assert_called_once_with("access_token_123")

    def test_logout_without_token(self):
        """Test logout without authentication token."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            # Mock authentication failure
            mock_auth.side_effect = HTTPException(status_code=401, detail="Not authenticated")

            response = client.post("/auth/logout")
            assert response.status_code == 401

    def test_logout_with_malformed_header(self, mock_jwt_manager):
        """Test logout with malformed Authorization header."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="user_123", username="testuser", permissions=set()
            )

            # Test with malformed header
            headers = {"Authorization": "InvalidFormat token"}
            response = client.post("/auth/logout", headers=headers)

            assert response.status_code == 200
            # Should not call blacklist_token with malformed header
            mock_jwt_manager.blacklist_token.assert_not_called()


class TestCurrentUserEndpoint:
    """Test current user info endpoint."""

    def test_get_current_user_info(self, mock_auth_storage, sample_user):
        """Test getting current user information."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="user_123", username="testuser", permissions={"api:read"}
            )
            mock_auth_storage.get_user_by_id.return_value = sample_user

            response = client.get("/auth/me")

            assert response.status_code == 200
            data = response.json()

            assert data["id"] == "user_123"
            assert data["username"] == "testuser"
            assert data["email"] == "test@example.com"
            assert data["is_active"] is True

    def test_get_current_user_not_found(self, mock_auth_storage):
        """Test getting current user when user not found."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="nonexistent_user", username="testuser", permissions=set()
            )
            mock_auth_storage.get_user_by_id.return_value = None

            response = client.get("/auth/me")

            assert response.status_code == 404
            data = response.json()
            assert "User not found" in data["detail"]


class TestUserManagementEndpoints:
    """Test user management endpoints (admin only)."""

    def test_create_user_success(self, mock_auth_storage, mock_password_manager, sample_admin_user):
        """Test successful user creation by admin."""
        with patch("app.automation.auth_endpoints.require_permission") as mock_perm:
            mock_perm.return_value = lambda: AuthContext(
                user_id="admin_123", username="adminuser", permissions={"admin:users"}
            )
            mock_auth_storage.create_user.return_value = True

            user_data = {
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "password123",
                "role": "user",
            }

            response = client.post("/auth/users", json=user_data)

            assert response.status_code == 200
            data = response.json()

            assert data["username"] == "newuser"
            assert data["email"] == "newuser@example.com"
            assert data["role"] == "user"

            # Verify password was hashed
            mock_password_manager.hash_password.assert_called_once_with("password123")
            mock_auth_storage.create_user.assert_called_once()

    def test_create_user_duplicate(self, mock_auth_storage, mock_password_manager):
        """Test creating user with duplicate username/email."""
        with patch("app.automation.auth_endpoints.require_permission") as mock_perm:
            mock_perm.return_value = lambda: AuthContext(
                user_id="admin_123", username="adminuser", permissions={"admin:users"}
            )
            mock_auth_storage.create_user.return_value = False  # Duplicate

            user_data = {
                "username": "existinguser",
                "email": "existing@example.com",
                "password": "password123",
                "role": "user",
            }

            response = client.post("/auth/users", json=user_data)

            assert response.status_code == 400
            data = response.json()
            assert "already exists" in data["detail"]

    def test_list_users(self, mock_auth_storage, sample_user):
        """Test listing users by admin."""
        with patch("app.automation.auth_endpoints.require_permission") as mock_perm:
            mock_perm.return_value = lambda: AuthContext(
                user_id="admin_123", username="adminuser", permissions={"admin:users"}
            )
            mock_auth_storage.list_users.return_value = [sample_user]

            response = client.get("/auth/users")

            assert response.status_code == 200
            data = response.json()

            assert len(data) == 1
            assert data[0]["username"] == "testuser"
            assert data[0]["email"] == "test@example.com"

    def test_delete_user_success(self, mock_auth_storage):
        """Test successful user deletion by admin."""
        with patch("app.automation.auth_endpoints.require_permission") as mock_perm:
            mock_perm.return_value = lambda: AuthContext(
                user_id="admin_123", username="adminuser", permissions={"admin:users"}
            )
            mock_auth_storage.delete_user.return_value = True

            response = client.delete("/auth/users/user_123")

            assert response.status_code == 200
            data = response.json()
            assert "deleted successfully" in data["message"]

            mock_auth_storage.delete_user.assert_called_once_with("user_123")

    def test_delete_user_self(self, mock_auth_storage):
        """Test admin cannot delete their own account."""
        with patch("app.automation.auth_endpoints.require_permission") as mock_perm:
            mock_perm.return_value = lambda: AuthContext(
                user_id="admin_123", username="adminuser", permissions={"admin:users"}
            )

            response = client.delete("/auth/users/admin_123")

            assert response.status_code == 400
            data = response.json()
            assert "Cannot delete your own account" in data["detail"]

    def test_delete_user_not_found(self, mock_auth_storage):
        """Test deleting non-existent user."""
        with patch("app.automation.auth_endpoints.require_permission") as mock_perm:
            mock_perm.return_value = lambda: AuthContext(
                user_id="admin_123", username="adminuser", permissions={"admin:users"}
            )
            mock_auth_storage.delete_user.return_value = False

            response = client.delete("/auth/users/nonexistent")

            assert response.status_code == 404
            data = response.json()
            assert "User not found" in data["detail"]


class TestAPIKeyManagementEndpoints:
    """Test API key management endpoints."""

    def test_create_api_key_success(self, mock_auth_storage):
        """Test successful API key creation."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="user_123", username="testuser", permissions={"api:read"}
            )

            # Mock APIKey.create_key method
            with patch("app.automation.auth_endpoints.APIKey.create_key") as mock_create:
                mock_api_key = Mock()
                mock_api_key.id = "key_123"
                mock_api_key.name = "Test Key"
                mock_api_key.key_prefix = "testkey1"
                mock_api_key.scopes = {"api:read"}
                mock_api_key.is_active = True
                mock_api_key.created_at = datetime.now(UTC)
                mock_api_key.last_used = None
                mock_api_key.expires_at = None
                mock_api_key.rate_limit = 1000
                mock_api_key.usage_count = 0

                mock_create.return_value = (mock_api_key, "testkey1_full_api_key")
                mock_auth_storage.create_api_key.return_value = True

                key_data = {"name": "Test Key", "scopes": ["api:read"], "expires_days": 30}

                response = client.post("/auth/api-keys", json=key_data)

                assert response.status_code == 200
                data = response.json()

                assert data["api_key"] == "testkey1_full_api_key"
                assert data["key_info"]["name"] == "Test Key"
                assert "Store this API key securely" in data["warning"]

    def test_create_api_key_storage_failure(self, mock_auth_storage):
        """Test API key creation with storage failure."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="user_123", username="testuser", permissions={"api:read"}
            )

            with patch("app.automation.auth_endpoints.APIKey.create_key") as mock_create:
                mock_create.return_value = (Mock(), "test_key")
                mock_auth_storage.create_api_key.return_value = False  # Storage failure

                key_data = {"name": "Test Key", "scopes": ["api:read"]}

                response = client.post("/auth/api-keys", json=key_data)

                assert response.status_code == 500
                data = response.json()
                assert "Failed to create API key" in data["detail"]

    def test_list_api_keys(self, mock_auth_storage, sample_api_key):
        """Test listing API keys for current user."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="user_123", username="testuser", permissions={"api:read"}
            )
            mock_auth_storage.list_api_keys.return_value = [sample_api_key]

            response = client.get("/auth/api-keys")

            assert response.status_code == 200
            data = response.json()

            assert len(data) == 1
            assert data[0]["name"] == "Test API Key"
            assert data[0]["key_prefix"] == "testkey1"

            # Verify only user's keys are requested
            mock_auth_storage.list_api_keys.assert_called_once_with(user_id="user_123")

    def test_delete_api_key_success(self, mock_auth_storage, sample_api_key):
        """Test successful API key deletion."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="user_123", username="testuser", permissions={"api:read"}
            )
            mock_auth_storage.get_api_key_by_id.return_value = sample_api_key
            mock_auth_storage.delete_api_key.return_value = True

            response = client.delete("/auth/api-keys/key_123")

            assert response.status_code == 200
            data = response.json()
            assert "deleted successfully" in data["message"]

    def test_delete_api_key_not_found(self, mock_auth_storage):
        """Test deleting non-existent API key."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="user_123", username="testuser", permissions={"api:read"}
            )
            mock_auth_storage.get_api_key_by_id.return_value = None

            response = client.delete("/auth/api-keys/nonexistent")

            assert response.status_code == 404
            data = response.json()
            assert "API key not found" in data["detail"]

    def test_delete_api_key_wrong_owner(self, mock_auth_storage):
        """Test deleting API key owned by another user."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="user_123",
                username="testuser",
                permissions={"api:read"},
                role=UserRole.USER,
            )

            # API key owned by different user
            other_user_key = APIKey(
                id="key_456",
                user_id="other_user",
                name="Other User Key",
                key_hash="hash",
                key_prefix="otherkey",
            )
            mock_auth_storage.get_api_key_by_id.return_value = other_user_key

            response = client.delete("/auth/api-keys/key_456")

            assert response.status_code == 403
            data = response.json()
            assert "Can only delete your own API keys" in data["detail"]

    def test_delete_api_key_admin_can_delete_any(self, mock_auth_storage):
        """Test admin can delete any API key."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="admin_123",
                username="adminuser",
                permissions={"admin:system"},
                role=UserRole.ADMIN,
            )

            # API key owned by different user
            other_user_key = APIKey(
                id="key_456",
                user_id="other_user",
                name="Other User Key",
                key_hash="hash",
                key_prefix="otherkey",
            )
            mock_auth_storage.get_api_key_by_id.return_value = other_user_key
            mock_auth_storage.delete_api_key.return_value = True

            response = client.delete("/auth/api-keys/key_456")

            assert response.status_code == 200
            data = response.json()
            assert "deleted successfully" in data["message"]


class TestSecurityEndpoints:
    """Test security and system endpoints."""

    def test_get_security_stats(self, mock_auth_storage):
        """Test getting security statistics."""
        with patch("app.automation.auth_endpoints.require_permission") as mock_perm:
            mock_perm.return_value = lambda: AuthContext(
                user_id="admin_123", username="adminuser", permissions={"admin:system"}
            )

            with patch("app.automation.auth_endpoints.get_auth_stats") as mock_stats:
                mock_stats.return_value = {
                    "total_requests": 1000,
                    "failed_attempts": 5,
                    "active_sessions": 10,
                }

                response = client.get("/auth/security/stats")

                assert response.status_code == 200
                data = response.json()

                assert data["total_requests"] == 1000
                assert data["failed_attempts"] == 5
                assert data["active_sessions"] == 10

    def test_get_audit_log(self):
        """Test getting audit log (placeholder implementation)."""
        with patch("app.automation.auth_endpoints.require_permission") as mock_perm:
            mock_perm.return_value = lambda: AuthContext(
                user_id="admin_123", username="adminuser", permissions={"admin:system"}
            )

            response = client.get("/auth/security/audit")

            assert response.status_code == 200
            data = response.json()
            assert "not yet implemented" in data["message"]

    def test_reset_suspicious_ips(self, mock_security_monitor):
        """Test resetting suspicious IPs."""
        with patch("app.automation.auth_endpoints.require_permission") as mock_perm:
            mock_perm.return_value = lambda: AuthContext(
                user_id="admin_123", username="adminuser", permissions={"admin:system"}
            )

            response = client.post("/auth/security/reset-suspicious")

            assert response.status_code == 200
            data = response.json()
            assert "cleared" in data["message"]

            # Verify security monitor was called
            mock_security_monitor.suspicious_ips.clear.assert_called_once()
            mock_security_monitor.failed_attempts.clear.assert_called_once()


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_auth_health_success(self, mock_auth_storage):
        """Test successful health check."""
        # Mock storage stats
        mock_auth_storage.get_storage_stats.return_value = {
            "total_users": 10,
            "active_users": 8,
            "total_api_keys": 15,
        }

        with patch("app.automation.auth_endpoints.validate_token_security") as mock_validate:
            mock_validate.return_value = {"algorithm": "HS256", "key_strength": "strong"}

            response = client.get("/auth/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "healthy"
            assert "storage_stats" in data
            assert "jwt_security" in data
            assert "timestamp" in data

    def test_auth_health_failure(self, mock_auth_storage):
        """Test health check with failure."""
        # Mock storage to raise exception
        mock_auth_storage.get_storage_stats.side_effect = Exception("Database connection failed")

        response = client.get("/auth/health")

        assert response.status_code == 503
        data = response.json()

        assert data["status"] == "unhealthy"
        assert "error" in data


class TestAuthenticationErrorHandling:
    """Test error handling in authentication flows."""

    def test_permission_denied(self):
        """Test permission denied scenarios."""
        # Try to access admin endpoint without permission
        response = client.get("/auth/users")
        assert response.status_code in {401, 403}

    def test_malformed_json_requests(self):
        """Test handling of malformed JSON requests."""
        # Invalid JSON for login
        response = client.post(
            "/auth/login", data="invalid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        # Login without username
        response = client.post("/auth/login", json={"password": "test"})
        assert response.status_code == 422

        # Create user without email
        with patch("app.automation.auth_endpoints.require_permission") as mock_perm:
            mock_perm.return_value = lambda: AuthContext(
                user_id="admin_123", username="adminuser", permissions={"admin:users"}
            )

            response = client.post(
                "/auth/users",
                json={
                    "username": "testuser",
                    "password": "password123",
                    # Missing email
                },
            )
            assert response.status_code == 422

    def test_invalid_token_formats(self):
        """Test handling of invalid token formats."""
        # Invalid Authorization header format
        headers = {"Authorization": "InvalidFormat"}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 401

        # Empty Authorization header
        headers = {"Authorization": ""}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 401


class TestDataValidation:
    """Test data validation in authentication endpoints."""

    def test_user_creation_validation(self):
        """Test user creation data validation."""
        with patch("app.automation.auth_endpoints.require_permission") as mock_perm:
            mock_perm.return_value = lambda: AuthContext(
                user_id="admin_123", username="adminuser", permissions={"admin:users"}
            )

            # Test with invalid email format
            response = client.post(
                "/auth/users",
                json={
                    "username": "testuser",
                    "email": "invalid-email",
                    "password": "password123",
                    "role": "user",
                },
            )
            assert response.status_code == 422

            # Test with invalid role
            response = client.post(
                "/auth/users",
                json={
                    "username": "testuser",
                    "email": "test@example.com",
                    "password": "password123",
                    "role": "invalid_role",
                },
            )
            assert response.status_code == 422

    def test_api_key_creation_validation(self):
        """Test API key creation data validation."""
        with patch("app.automation.auth_endpoints.get_current_active_user") as mock_auth:
            mock_auth.return_value = AuthContext(
                user_id="user_123", username="testuser", permissions={"api:read"}
            )

            # Test with invalid expires_days
            response = client.post(
                "/auth/api-keys",
                json={"name": "Test Key", "scopes": ["api:read"], "expires_days": -1},  # Invalid
            )
            assert response.status_code == 422

            # Test with empty name
            response = client.post("/auth/api-keys", json={"name": "", "scopes": ["api:read"]})
            assert response.status_code == 422
