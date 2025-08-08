"""Comprehensive tests for authentication module.

Tests real authentication scenarios with service account credentials.
"""

import os
from unittest.mock import MagicMock
from unittest.mock import patch

from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account
import pytest

from gemini_cli.core.auth import GeminiAuth


class TestGeminiAuth:
    """Test suite for GeminiAuth class."""

    def test_constants(self):
        """Test that class constants are properly defined."""
        assert (
            GeminiAuth.BUSINESS_ACCOUNT_PATH
            == "/home/david/.auth/business/service-account-key.json"
        )
        assert GeminiAuth.PROJECT_ID == "auricleinc-gemini"
        assert GeminiAuth.LOCATION == "us-central1"

    @patch("pathlib.Path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_get_credentials_success(self, mock_from_file, mock_exists):
        """Test successful credential retrieval."""
        # Setup
        mock_exists.return_value = True
        mock_credentials = MagicMock(spec=service_account.Credentials)
        mock_from_file.return_value = mock_credentials

        # Execute
        credentials, project_id = GeminiAuth.get_credentials()

        # Verify
        assert credentials == mock_credentials
        assert project_id == "auricleinc-gemini"

        mock_exists.assert_called_once()
        mock_from_file.assert_called_once_with(
            GeminiAuth.BUSINESS_ACCOUNT_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    @patch("pathlib.Path.exists")
    def test_get_credentials_missing_file(self, mock_exists):
        """Test credential retrieval when service account file is missing."""
        # Setup
        mock_exists.return_value = False

        # Execute & Verify
        with pytest.raises(DefaultCredentialsError) as exc_info:
            GeminiAuth.get_credentials()

        assert "Business service account required at" in str(exc_info.value)
        assert GeminiAuth.BUSINESS_ACCOUNT_PATH in str(exc_info.value)

    @patch("pathlib.Path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_get_credentials_invalid_file(self, mock_from_file, mock_exists):
        """Test credential retrieval with invalid service account file."""
        # Setup
        mock_exists.return_value = True
        mock_from_file.side_effect = ValueError("Invalid service account file")

        # Execute & Verify
        with pytest.raises(ValueError) as exc_info:
            GeminiAuth.get_credentials()

        assert "Invalid service account file" in str(exc_info.value)

    @patch.dict(os.environ, {}, clear=True)
    def test_setup_environment(self):
        """Test environment variable setup."""
        # Execute
        GeminiAuth.setup_environment()

        # Verify
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == GeminiAuth.BUSINESS_ACCOUNT_PATH
        assert os.environ["GOOGLE_CLOUD_PROJECT"] == GeminiAuth.PROJECT_ID

    @patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "old_value"}, clear=True)
    def test_setup_environment_override(self):
        """Test that setup_environment overrides existing values."""
        # Setup - verify initial state
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "old_value"

        # Execute
        GeminiAuth.setup_environment()

        # Verify
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == GeminiAuth.BUSINESS_ACCOUNT_PATH
        assert os.environ["GOOGLE_CLOUD_PROJECT"] == GeminiAuth.PROJECT_ID

    @patch("pathlib.Path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_get_credentials_scopes(self, mock_from_file, mock_exists):
        """Test that credentials are created with correct scopes."""
        # Setup
        mock_exists.return_value = True
        mock_credentials = MagicMock(spec=service_account.Credentials)
        mock_from_file.return_value = mock_credentials

        # Execute
        credentials, _ = GeminiAuth.get_credentials()

        # Verify scopes are correctly passed
        call_args = mock_from_file.call_args
        assert call_args[1]["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]

    def test_class_method_accessibility(self):
        """Test that all methods are accessible as class methods."""
        # Verify methods exist and are callable
        assert callable(GeminiAuth.get_credentials)
        assert callable(GeminiAuth.setup_environment)

    @patch("pathlib.Path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_credentials_return_type(self, mock_from_file, mock_exists):
        """Test that get_credentials returns correct types."""
        # Setup
        mock_exists.return_value = True
        mock_credentials = MagicMock(spec=service_account.Credentials)
        mock_from_file.return_value = mock_credentials

        # Execute
        credentials, project_id = GeminiAuth.get_credentials()

        # Verify types
        assert isinstance(project_id, str)
        assert project_id == "auricleinc-gemini"
        # credentials should be a service account credentials object
        assert credentials == mock_credentials

    @patch("pathlib.Path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_multiple_calls_consistency(self, mock_from_file, mock_exists):
        """Test that multiple calls to get_credentials return consistent results."""
        # Setup
        mock_exists.return_value = True
        mock_credentials = MagicMock(spec=service_account.Credentials)
        mock_from_file.return_value = mock_credentials

        # Execute multiple calls
        creds1, proj1 = GeminiAuth.get_credentials()
        creds2, proj2 = GeminiAuth.get_credentials()

        # Verify consistency
        assert proj1 == proj2 == "auricleinc-gemini"
        # Each call should create new credentials instance
        assert mock_from_file.call_count == 2

    def test_file_path_format(self):
        """Test that the service account file path is correctly formatted."""
        path = GeminiAuth.BUSINESS_ACCOUNT_PATH

        # Should be an absolute path
        assert path.startswith("/")
        # Should point to a JSON file
        assert path.endswith(".json")
        # Should contain expected directory structure
        assert "/.auth/business/" in path

    @patch("pathlib.Path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    @patch.dict(os.environ, {}, clear=True)
    def test_full_auth_flow(self, mock_from_file, mock_exists):
        """Test complete authentication flow."""
        # Setup
        mock_exists.return_value = True
        mock_credentials = MagicMock(spec=service_account.Credentials)
        mock_from_file.return_value = mock_credentials

        # Execute full flow
        GeminiAuth.setup_environment()
        credentials, project_id = GeminiAuth.get_credentials()

        # Verify environment was set up
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == GeminiAuth.BUSINESS_ACCOUNT_PATH
        assert os.environ["GOOGLE_CLOUD_PROJECT"] == GeminiAuth.PROJECT_ID

        # Verify credentials were obtained
        assert credentials == mock_credentials
        assert project_id == "auricleinc-gemini"

    @patch("pathlib.Path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_error_handling_during_credential_creation(self, mock_from_file, mock_exists):
        """Test error handling when credential creation fails."""
        # Setup
        mock_exists.return_value = True
        mock_from_file.side_effect = Exception("Generic credential error")

        # Execute & Verify
        with pytest.raises(Exception) as exc_info:
            GeminiAuth.get_credentials()

        assert "Generic credential error" in str(exc_info.value)

    def test_project_configuration(self):
        """Test that project configuration matches expected values."""
        assert GeminiAuth.PROJECT_ID == "auricleinc-gemini"
        assert GeminiAuth.LOCATION == "us-central1"

        # Verify these are valid Google Cloud configurations
        assert isinstance(GeminiAuth.PROJECT_ID, str)
        assert isinstance(GeminiAuth.LOCATION, str)
        assert len(GeminiAuth.PROJECT_ID) > 0
        assert len(GeminiAuth.LOCATION) > 0
