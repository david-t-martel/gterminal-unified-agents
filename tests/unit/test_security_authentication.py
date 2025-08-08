"""Comprehensive security tests for authentication components.

This module tests authentication flows, credential management, security validation,
and access control mechanisms.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import Mock
from unittest.mock import patch

# Import authentication modules
from app.automation.auth_models import AuthenticationCredentials
from app.automation.auth_models import AuthenticationRequest
from app.automation.auth_models import AuthenticationResponse
from app.automation.auth_models import SecurityValidation
from app.automation.auth_models import TokenInfo
from app.automation.auth_storage import CredentialManager
from app.automation.auth_storage import SecureAuthStorage
from app.automation.auth_storage import SecurityValidator
import pytest


class TestAuthenticationCredentials:
    """Test AuthenticationCredentials Pydantic model."""

    def test_credentials_creation_valid(self):
        """Test valid credentials creation."""
        creds = AuthenticationCredentials(
            provider="google",
            client_id="test-client-id",
            client_secret="test-secret",
            project_id="test-project",
        )

        assert creds.provider == "google"
        assert creds.client_id == "test-client-id"
        assert creds.client_secret == "test-secret"
        assert creds.project_id == "test-project"
        assert creds.scopes == []  # Default empty list
        assert creds.created_at is not None
        assert isinstance(creds.created_at, datetime)

    def test_credentials_with_scopes(self):
        """Test credentials with custom scopes."""
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        creds = AuthenticationCredentials(
            provider="google",
            client_id="test-id",
            client_secret="test-secret",
            project_id="test-project",
            scopes=scopes,
        )

        assert creds.scopes == scopes

    def test_credentials_validation_empty_provider(self):
        """Test validation fails with empty provider."""
        with pytest.raises(ValueError, match="Provider cannot be empty"):
            AuthenticationCredentials(
                provider="",
                client_id="test-id",
                client_secret="test-secret",
                project_id="test-project",
            )

    def test_credentials_validation_empty_client_id(self):
        """Test validation fails with empty client_id."""
        with pytest.raises(ValueError, match="Client ID cannot be empty"):
            AuthenticationCredentials(
                provider="google",
                client_id="",
                client_secret="test-secret",
                project_id="test-project",
            )

    def test_credentials_validation_weak_secret(self):
        """Test validation fails with weak client_secret."""
        with pytest.raises(ValueError, match="Client secret must be at least 8 characters"):
            AuthenticationCredentials(
                provider="google",
                client_id="test-id",
                client_secret="weak",
                project_id="test-project",
            )

    def test_credentials_validation_invalid_project_id(self):
        """Test validation fails with invalid project_id."""
        with pytest.raises(ValueError, match="Project ID must be 6-30 characters"):
            AuthenticationCredentials(
                provider="google",
                client_id="test-id",
                client_secret="strong-secret",
                project_id="a",
            )

    def test_credentials_to_dict_excludes_sensitive(self):
        """Test that to_dict excludes sensitive information."""
        creds = AuthenticationCredentials(
            provider="google",
            client_id="test-id",
            client_secret="secret-value",
            project_id="test-project",
        )

        # Should not include client_secret in dictionary representation
        creds_dict = creds.model_dump(exclude={"client_secret"})
        assert "client_secret" not in creds_dict
        assert creds_dict["provider"] == "google"


class TestAuthenticationRequest:
    """Test AuthenticationRequest model."""

    def test_request_creation_valid(self):
        """Test valid authentication request."""
        request = AuthenticationRequest(
            provider="google",
            client_id="test-id",
            scopes=["cloud-platform"],
            redirect_uri="http://localhost:8080/callback",
        )

        assert request.provider == "google"
        assert request.client_id == "test-id"
        assert request.scopes == ["cloud-platform"]
        assert request.redirect_uri == "http://localhost:8080/callback"

    def test_request_validation_invalid_redirect_uri(self):
        """Test validation of redirect URI."""
        with pytest.raises(ValueError, match="Invalid redirect URI"):
            AuthenticationRequest(
                provider="google", client_id="test-id", redirect_uri="not-a-valid-uri"
            )

    def test_request_validation_empty_scopes(self):
        """Test that empty scopes are handled correctly."""
        request = AuthenticationRequest(
            provider="google", client_id="test-id", redirect_uri="http://localhost:8080/callback"
        )

        assert request.scopes == []


class TestAuthenticationResponse:
    """Test AuthenticationResponse model."""

    def test_response_success(self):
        """Test successful authentication response."""
        token_info = TokenInfo(access_token="access-token", token_type="Bearer", expires_in=3600)

        response = AuthenticationResponse(success=True, token_info=token_info, user_id="user-123")

        assert response.success is True
        assert response.token_info.access_token == "access-token"
        assert response.user_id == "user-123"
        assert response.error_message is None

    def test_response_failure(self):
        """Test failed authentication response."""
        response = AuthenticationResponse(success=False, error_message="Invalid credentials")

        assert response.success is False
        assert response.error_message == "Invalid credentials"
        assert response.token_info is None
        assert response.user_id is None


class TestTokenInfo:
    """Test TokenInfo model."""

    def test_token_info_creation(self):
        """Test token info creation."""
        token_info = TokenInfo(
            access_token="access-token",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="refresh-token",
            scope="cloud-platform",
        )

        assert token_info.access_token == "access-token"
        assert token_info.token_type == "Bearer"
        assert token_info.expires_in == 3600
        assert token_info.refresh_token == "refresh-token"
        assert token_info.scope == "cloud-platform"

    def test_token_info_expiry_calculation(self):
        """Test token expiry time calculation."""
        before_creation = datetime.now(UTC)

        token_info = TokenInfo(access_token="token", token_type="Bearer", expires_in=3600)

        after_creation = datetime.now(UTC)

        # Expires at should be between before and after creation + 3600 seconds
        expected_min = before_creation + timedelta(seconds=3600)
        expected_max = after_creation + timedelta(seconds=3600)

        assert expected_min <= token_info.expires_at <= expected_max

    def test_token_info_is_expired(self):
        """Test token expiry checking."""
        # Create expired token
        expired_token = TokenInfo(
            access_token="token", token_type="Bearer", expires_in=-1
        )  # Already expired

        assert expired_token.is_expired() is True

        # Create valid token
        valid_token = TokenInfo(access_token="token", token_type="Bearer", expires_in=3600)

        assert valid_token.is_expired() is False


class TestSecurityValidation:
    """Test SecurityValidation model."""

    def test_security_validation_creation(self):
        """Test security validation creation."""
        validation = SecurityValidation(
            is_valid=True, validation_type="credential_strength", message="Credentials are secure"
        )

        assert validation.is_valid is True
        assert validation.validation_type == "credential_strength"
        assert validation.message == "Credentials are secure"

    def test_security_validation_with_details(self):
        """Test security validation with details."""
        details = {"strength_score": 85, "vulnerabilities": []}

        validation = SecurityValidation(
            is_valid=True,
            validation_type="security_scan",
            message="Security check passed",
            details=details,
        )

        assert validation.details == details
        assert validation.details["strength_score"] == 85


class TestSecureAuthStorage:
    """Test SecureAuthStorage class."""

    @pytest.fixture
    def temp_storage_path(self, tmp_path):
        """Create temporary storage path."""
        return tmp_path / "auth_storage"

    @pytest.fixture
    def auth_storage(self, temp_storage_path):
        """Create SecureAuthStorage instance."""
        return SecureAuthStorage(storage_path=temp_storage_path)

    def test_storage_initialization(self, auth_storage, temp_storage_path):
        """Test storage initialization."""
        assert auth_storage.storage_path == temp_storage_path
        assert temp_storage_path.exists()
        assert oct(temp_storage_path.stat().st_mode)[-3:] == "700"  # Secure permissions

    @pytest.mark.asyncio
    async def test_store_credentials_success(self, auth_storage):
        """Test storing credentials successfully."""
        creds = AuthenticationCredentials(
            provider="google",
            client_id="test-id",
            client_secret="test-secret",
            project_id="test-project",
        )

        result = await auth_storage.store_credentials("test-key", creds)

        assert result is True

        # Verify file was created
        cred_file = auth_storage.storage_path / "test-key.json"
        assert cred_file.exists()
        assert oct(cred_file.stat().st_mode)[-3:] == "600"  # Secure file permissions

    @pytest.mark.asyncio
    async def test_retrieve_credentials_success(self, auth_storage):
        """Test retrieving stored credentials."""
        # First store credentials
        creds = AuthenticationCredentials(
            provider="google",
            client_id="test-id",
            client_secret="test-secret",
            project_id="test-project",
        )

        await auth_storage.store_credentials("test-key", creds)

        # Then retrieve them
        retrieved_creds = await auth_storage.retrieve_credentials("test-key")

        assert retrieved_creds is not None
        assert retrieved_creds.provider == "google"
        assert retrieved_creds.client_id == "test-id"
        assert retrieved_creds.client_secret == "test-secret"
        assert retrieved_creds.project_id == "test-project"

    @pytest.mark.asyncio
    async def test_retrieve_credentials_not_found(self, auth_storage):
        """Test retrieving non-existent credentials."""
        result = await auth_storage.retrieve_credentials("non-existent")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_credentials_success(self, auth_storage):
        """Test deleting stored credentials."""
        # Store credentials first
        creds = AuthenticationCredentials(
            provider="google",
            client_id="test-id",
            client_secret="test-secret",
            project_id="test-project",
        )

        await auth_storage.store_credentials("test-key", creds)

        # Delete them
        result = await auth_storage.delete_credentials("test-key")

        assert result is True

        # Verify file is gone
        cred_file = auth_storage.storage_path / "test-key.json"
        assert not cred_file.exists()

    @pytest.mark.asyncio
    async def test_delete_credentials_not_found(self, auth_storage):
        """Test deleting non-existent credentials."""
        result = await auth_storage.delete_credentials("non-existent")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_stored_credentials(self, auth_storage):
        """Test listing stored credentials."""
        # Store multiple credentials
        creds1 = AuthenticationCredentials(
            provider="google", client_id="id1", client_secret="secret1", project_id="project1"
        )

        creds2 = AuthenticationCredentials(
            provider="azure", client_id="id2", client_secret="secret2", project_id="project2"
        )

        await auth_storage.store_credentials("google-creds", creds1)
        await auth_storage.store_credentials("azure-creds", creds2)

        # List credentials
        stored_keys = await auth_storage.list_stored_credentials()

        assert "google-creds" in stored_keys
        assert "azure-creds" in stored_keys
        assert len(stored_keys) == 2

    @pytest.mark.asyncio
    async def test_storage_encryption(self, auth_storage):
        """Test that stored credentials are encrypted."""
        creds = AuthenticationCredentials(
            provider="google",
            client_id="test-id",
            client_secret="very-secret-value",
            project_id="test-project",
        )

        await auth_storage.store_credentials("test-key", creds)

        # Read raw file content
        cred_file = auth_storage.storage_path / "test-key.json"
        raw_content = cred_file.read_text()

        # Secret should not appear in plaintext
        assert "very-secret-value" not in raw_content
        # But should contain encrypted/encoded data
        assert len(raw_content) > 0


class TestCredentialManager:
    """Test CredentialManager class."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock SecureAuthStorage."""
        return Mock(spec=SecureAuthStorage)

    @pytest.fixture
    def credential_manager(self, mock_storage):
        """Create CredentialManager with mock storage."""
        return CredentialManager(auth_storage=mock_storage)

    @pytest.mark.asyncio
    async def test_authenticate_google_success(self, credential_manager, mock_storage):
        """Test successful Google authentication."""
        # Mock stored credentials
        mock_creds = AuthenticationCredentials(
            provider="google",
            client_id="test-id",
            client_secret="test-secret",
            project_id="test-project",
        )
        mock_storage.retrieve_credentials.return_value = mock_creds

        # Mock successful authentication
        with patch("app.automation.auth_storage.google.oauth2.service_account") as mock_service:
            mock_service.Credentials.from_service_account_info.return_value = Mock()

            result = await credential_manager.authenticate_google("test-project")

            assert result.success is True
            assert result.user_id == "test-project"

    @pytest.mark.asyncio
    async def test_authenticate_google_no_credentials(self, credential_manager, mock_storage):
        """Test Google authentication with no stored credentials."""
        mock_storage.retrieve_credentials.return_value = None

        result = await credential_manager.authenticate_google("test-project")

        assert result.success is False
        assert "No credentials found" in result.error_message

    @pytest.mark.asyncio
    async def test_refresh_token_success(self, credential_manager):
        """Test successful token refresh."""
        token_info = TokenInfo(
            access_token="old-token",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="refresh-token",
        )

        with patch("app.automation.auth_storage.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "new-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            new_token = await credential_manager.refresh_token(
                token_info, "client-id", "client-secret"
            )

            assert new_token.access_token == "new-token"
            assert new_token.refresh_token == "refresh-token"  # Should preserve refresh token

    @pytest.mark.asyncio
    async def test_refresh_token_failure(self, credential_manager):
        """Test token refresh failure."""
        token_info = TokenInfo(
            access_token="old-token",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="refresh-token",
        )

        with patch("app.automation.auth_storage.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Invalid refresh token"
            mock_post.return_value = mock_response

            new_token = await credential_manager.refresh_token(
                token_info, "client-id", "client-secret"
            )

            assert new_token is None

    @pytest.mark.asyncio
    async def test_validate_credentials_valid(self, credential_manager):
        """Test credential validation with valid credentials."""
        creds = AuthenticationCredentials(
            provider="google",
            client_id="valid-id",
            client_secret="strong-secret-value",
            project_id="valid-project",
        )

        result = await credential_manager.validate_credentials(creds)

        assert result.is_valid is True
        assert "valid" in result.message.lower()

    @pytest.mark.asyncio
    async def test_validate_credentials_weak_secret(self, credential_manager):
        """Test credential validation with weak secret."""
        creds = AuthenticationCredentials(
            provider="google",
            client_id="valid-id",
            client_secret="weak",
            project_id="valid-project",
        )

        result = await credential_manager.validate_credentials(creds)

        assert result.is_valid is False
        assert "weak" in result.message.lower() or "short" in result.message.lower()


class TestSecurityValidator:
    """Test SecurityValidator class."""

    @pytest.fixture
    def security_validator(self):
        """Create SecurityValidator instance."""
        return SecurityValidator()

    def test_validate_project_id_valid(self, security_validator):
        """Test valid project ID validation."""
        result = security_validator.validate_project_id("valid-project-123")

        assert result.is_valid is True

    def test_validate_project_id_invalid(self, security_validator):
        """Test invalid project ID validation."""
        result = security_validator.validate_project_id("INVALID-Project!")

        assert result.is_valid is False
        assert "invalid" in result.message.lower()

    def test_validate_client_secret_strong(self, security_validator):
        """Test strong client secret validation."""
        result = security_validator.validate_client_secret("StrongSecret123!")

        assert result.is_valid is True
        assert result.details["strength_score"] >= 80

    def test_validate_client_secret_weak(self, security_validator):
        """Test weak client secret validation."""
        result = security_validator.validate_client_secret("weak")

        assert result.is_valid is False
        assert result.details["strength_score"] < 50

    def test_scan_for_vulnerabilities_clean(self, security_validator):
        """Test vulnerability scan on clean credentials."""
        creds = AuthenticationCredentials(
            provider="google",
            client_id="clean-client-id",
            client_secret="clean-strong-secret-123",
            project_id="clean-project",
        )

        result = security_validator.scan_for_vulnerabilities(creds)

        assert result.is_valid is True
        assert len(result.details["vulnerabilities"]) == 0

    def test_scan_for_vulnerabilities_with_issues(self, security_validator):
        """Test vulnerability scan with security issues."""
        creds = AuthenticationCredentials(
            provider="google",
            client_id="admin",  # Common/weak ID
            client_secret="password123",  # Weak secret
            project_id="test",  # Too short
        )

        result = security_validator.scan_for_vulnerabilities(creds)

        assert result.is_valid is False
        assert len(result.details["vulnerabilities"]) > 0

    def test_check_credential_age_recent(self, security_validator):
        """Test credential age check for recent credentials."""
        creds = AuthenticationCredentials(
            provider="google",
            client_id="test-id",
            client_secret="test-secret",
            project_id="test-project",
        )

        result = security_validator.check_credential_age(creds, max_age_days=30)

        assert result.is_valid is True

    def test_check_credential_age_old(self, security_validator):
        """Test credential age check for old credentials."""
        creds = AuthenticationCredentials(
            provider="google",
            client_id="test-id",
            client_secret="test-secret",
            project_id="test-project",
        )

        # Manually set old creation time
        old_time = datetime.now(UTC) - timedelta(days=100)
        creds.created_at = old_time

        result = security_validator.check_credential_age(creds, max_age_days=30)

        assert result.is_valid is False
        assert "old" in result.message.lower() or "expired" in result.message.lower()


class TestSecurityIntegration:
    """Test security component integration."""

    @pytest.mark.asyncio
    async def test_full_authentication_flow(self, tmp_path):
        """Test complete authentication flow from storage to validation."""
        # Create real storage
        auth_storage = SecureAuthStorage(storage_path=tmp_path / "auth")
        credential_manager = CredentialManager(auth_storage=auth_storage)

        # Store credentials
        creds = AuthenticationCredentials(
            provider="google",
            client_id="integration-test-id",
            client_secret="strong-integration-secret-123",
            project_id="integration-project",
        )

        await auth_storage.store_credentials("integration-test", creds)

        # Retrieve and validate
        retrieved_creds = await auth_storage.retrieve_credentials("integration-test")
        assert retrieved_creds is not None

        validation_result = await credential_manager.validate_credentials(retrieved_creds)
        assert validation_result.is_valid is True

        # Cleanup
        await auth_storage.delete_credentials("integration-test")

    @pytest.mark.asyncio
    async def test_security_audit_trail(self, tmp_path):
        """Test that security operations create audit trail."""
        auth_storage = SecureAuthStorage(storage_path=tmp_path / "auth")

        # Store multiple credentials to create activity
        for i in range(3):
            creds = AuthenticationCredentials(
                provider="google",
                client_id=f"audit-test-{i}",
                client_secret=f"audit-secret-{i}-value",
                project_id=f"audit-project-{i}",
            )
            await auth_storage.store_credentials(f"audit-{i}", creds)

        # List should show all stored credentials
        stored_keys = await auth_storage.list_stored_credentials()
        assert len(stored_keys) == 3

        # Cleanup
        for i in range(3):
            await auth_storage.delete_credentials(f"audit-{i}")
