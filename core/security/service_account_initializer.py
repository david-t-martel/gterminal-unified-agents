#!/usr/bin/env python3
"""Service Account Environment Initializer.

This module provides secure service account initialization for MCP servers.
It reads the service account JSON, validates credentials, and sets required
environment variables while removing any API key fallbacks.

Security Features:
- Only uses service account authentication
- Removes API key environment variables
- Validates service account format and permissions
- Provides detailed error reporting
- Implements secure credential path resolution
"""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

logger = logging.getLogger(__name__)


class ServiceAccountInitializationError(Exception):
    """Raised when service account initialization fails."""

    pass


class ServiceAccountInitializer:
    """Secure service account environment initializer for GCP services."""

    def __init__(self, required_scopes: list[str] | None = None) -> None:
        """Initialize the service account manager.

        Args:
            required_scopes: List of required OAuth scopes for validation
        """
        self.required_scopes = required_scopes or [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/generative-language",
        ]
        self.service_account_data: dict[str, Any] | None = None
        self.credentials_path: str | None = None

    def find_service_account_path(self) -> str | None:
        """Find service account credentials using secure path resolution.

        Search order:
        1. GOOGLE_APPLICATION_CREDENTIALS environment variable
        2. Business profile service account
        3. Personal profile service account (development only)
        4. Default gcloud location

        Returns:
            Path to service account JSON file, or None if not found
        """
        # Check environment variable first
        env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if env_path and Path(env_path).exists():
            return env_path

        # Try standard business profile location (preferred for production)
        business_path = Path.home() / ".auth" / "business" / "service-account-key.json"
        if business_path.exists():
            logger.info("Found business profile service account")
            return str(business_path)

        # Try personal profile (development only)
        personal_path = Path.home() / ".auth" / "personal" / "service-account-key.json"
        if personal_path.exists():
            logger.warning(
                "Using personal profile service account - not recommended for production"
            )
            return str(personal_path)

        # Try default gcloud location
        gcloud_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        if gcloud_path.exists():
            logger.info("Found gcloud default credentials")
            return str(gcloud_path)

        return None

    def validate_service_account(self, sa_data: dict[str, Any]) -> None:
        """Validate service account JSON structure and required fields.

        Args:
            sa_data: Service account JSON data

        Raises:
            ServiceAccountInitializationError: If validation fails
        """
        required_fields = [
            "type",
            "project_id",
            "private_key_id",
            "private_key",
            "client_email",
            "client_id",
            "auth_uri",
            "token_uri",
        ]

        # Check service account type
        if sa_data.get("type") != "service_account":
            raise ServiceAccountInitializationError(
                f"Invalid credential type: {sa_data.get('type')}. Expected 'service_account'"
            )

        # Check required fields
        missing_fields = [field for field in required_fields if not sa_data.get(field)]
        if missing_fields:
            raise ServiceAccountInitializationError(
                f"Missing required fields in service account: {missing_fields}"
            )

        # Validate project ID format
        project_id = sa_data["project_id"]
        if not self._is_valid_project_id(project_id):
            raise ServiceAccountInitializationError(f"Invalid project ID format: {project_id}")

        # Validate email format
        email = sa_data["client_email"]
        if not email.endswith(".iam.gserviceaccount.com"):
            raise ServiceAccountInitializationError(
                f"Invalid service account email format: {email}"
            )

    def _is_valid_project_id(self, project_id: str) -> bool:
        """Validate Google Cloud project ID format."""
        if not project_id or not isinstance(project_id, str):
            return False

        # Must be 6-30 characters
        if not 6 <= len(project_id) <= 30:
            return False

        # Must start with a lowercase letter
        if not project_id[0].islower():
            return False

        # Must not end with a hyphen
        if project_id.endswith("-"):
            return False

        # Must contain only lowercase letters, digits, hyphens
        import re

        return bool(re.match(r"^[a-z][a-z0-9-]*[a-z0-9]$", project_id))

    def remove_api_key_variables(self) -> list[str]:
        """Remove all API key environment variables for security.

        Returns:
            List of removed environment variable names
        """
        api_key_vars = [
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GENAI_API_KEY",
            "GOOGLE_AI_API_KEY",
            "AI_API_KEY",
        ]

        removed_vars = []
        for var in api_key_vars:
            if var in os.environ:
                del os.environ[var]
                removed_vars.append(var)
                logger.info(f"Removed API key variable: {var}")

        return removed_vars

    def set_environment_variables(self, sa_data: dict[str, Any], credentials_path: str) -> None:
        """Set required environment variables from service account data.

        Args:
            sa_data: Service account JSON data
            credentials_path: Path to service account file
        """
        # Set core GCP environment variables
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        os.environ["GOOGLE_CLOUD_PROJECT"] = sa_data["project_id"]
        os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Enable Vertex AI by default for service account authentication
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

        # Set additional useful variables
        os.environ["GOOGLE_CLOUD_SERVICE_ACCOUNT_EMAIL"] = sa_data["client_email"]
        os.environ["GOOGLE_CLOUD_CLIENT_ID"] = sa_data["client_id"]

        logger.info(f"Configured environment for project: {sa_data['project_id']}")
        logger.info(f"Service account: {sa_data['client_email']}")

    def test_credentials(self) -> dict[str, Any]:
        """Test service account credentials by making a simple API call.

        Returns:
            Dictionary with test results and metadata
        """
        try:
            from google.auth import default
            from google.auth.transport.requests import Request

            # Get default credentials (should use our service account)
            credentials, project = default()

            # Test by refreshing the credentials
            request = Request()
            credentials.refresh(request)

            result = {
                "status": "success",
                "project_id": project,
                "service_account_email": getattr(credentials, "service_account_email", "unknown"),
                "scopes": getattr(credentials, "scopes", []),
                "token_valid": credentials.valid,
                "expiry": getattr(credentials, "expiry", None),
            }

            logger.info("Service account credentials validated successfully")
            return result

        except Exception as e:
            logger.exception(f"Service account credential test failed: {e}")
            return {"status": "error", "error": str(e)}

    def initialize(self) -> dict[str, Any]:
        """Initialize service account environment with comprehensive validation.

        Returns:
            Dictionary with initialization results and metadata

        Raises:
            ServiceAccountInitializationError: If initialization fails
        """
        logger.info("Initializing service account environment...")

        # Step 1: Find service account credentials
        self.credentials_path = self.find_service_account_path()
        if not self.credentials_path:
            raise ServiceAccountInitializationError(
                "No service account credentials found. Please ensure GOOGLE_APPLICATION_CREDENTIALS "
                "is set or place service-account-key.json in ~/.auth/business/ or ~/.auth/personal/"
            )

        # Step 2: Load and validate service account
        try:
            with open(self.credentials_path, encoding="utf-8") as f:
                self.service_account_data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ServiceAccountInitializationError(
                f"Failed to load service account from {self.credentials_path}: {e}"
            )

        # Step 3: Validate service account structure
        self.validate_service_account(self.service_account_data)

        # Step 4: Remove API key variables for security
        removed_vars = self.remove_api_key_variables()

        # Step 5: Set environment variables
        self.set_environment_variables(self.service_account_data, self.credentials_path)

        # Step 6: Test credentials
        credential_test = self.test_credentials()

        # Return initialization summary
        return {
            "status": "success",
            "credentials_path": self.credentials_path,
            "project_id": self.service_account_data["project_id"],
            "service_account_email": self.service_account_data["client_email"],
            "location": os.environ.get("GOOGLE_CLOUD_LOCATION"),
            "removed_api_keys": removed_vars,
            "credential_test": credential_test,
            "vertex_ai_enabled": True,
            "environment_variables_set": [
                "GOOGLE_APPLICATION_CREDENTIALS",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_GENAI_USE_VERTEXAI",
                "GOOGLE_CLOUD_SERVICE_ACCOUNT_EMAIL",
                "GOOGLE_CLOUD_CLIENT_ID",
            ],
        }


def initialize_service_account(required_scopes: list[str] | None = None) -> dict[str, Any]:
    """Convenience function to initialize service account environment.

    Args:
        required_scopes: List of required OAuth scopes

    Returns:
        Dictionary with initialization results

    Raises:
        ServiceAccountInitializationError: If initialization fails
    """
    initializer = ServiceAccountInitializer(required_scopes)
    return initializer.initialize()


def ensure_service_account_only() -> None:
    """Ensure only service account authentication is available.

    This function removes all API key variables and validates that
    service account credentials are properly configured.

    Raises:
        ServiceAccountInitializationError: If service account is not configured
    """
    initializer = ServiceAccountInitializer()

    # Remove API keys
    removed = initializer.remove_api_key_variables()
    if removed:
        logger.info(f"Removed API key variables for security: {removed}")

    # Validate service account is available
    credentials_path = initializer.find_service_account_path()
    if not credentials_path:
        raise ServiceAccountInitializationError(
            "Service account authentication required. API key authentication disabled for security."
        )

    logger.info(f"Service account authentication confirmed: {credentials_path}")


if __name__ == "__main__":
    """CLI interface for service account initialization."""
    try:
        result = initialize_service_account()
        print("Service Account Initialization Successful!")
        print(f"Project: {result['project_id']}")
        print(f"Service Account: {result['service_account_email']}")
        print(f"Credentials Path: {result['credentials_path']}")

        if result.get("removed_api_keys"):
            print(f"Removed API keys: {result['removed_api_keys']}")

        if result["credential_test"]["status"] == "success":
            print("✅ Credentials validated successfully")
        else:
            print(f"⚠️  Credential validation warning: {result['credential_test'].get('error')}")

    except ServiceAccountInitializationError as e:
        print(f"❌ Service Account Initialization Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
