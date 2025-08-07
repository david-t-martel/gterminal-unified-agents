"""Business service account authentication only."""

import os

from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account


class GeminiAuth:
    """Strict business service account authentication."""

    BUSINESS_ACCOUNT_PATH = "/home/david/.auth/business/service-account-key.json"
    PROJECT_ID = "auricleinc-gemini"
    LOCATION = "us-central1"

    @classmethod
    def get_credentials(cls):
        """Get business service account credentials only."""
        if not os.path.exists(cls.BUSINESS_ACCOUNT_PATH):
            raise DefaultCredentialsError(
                f"Business service account required at {cls.BUSINESS_ACCOUNT_PATH}"
            )

        credentials = service_account.Credentials.from_service_account_file(
            cls.BUSINESS_ACCOUNT_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        return credentials, cls.PROJECT_ID

    @classmethod
    def setup_environment(cls):
        """Setup environment for Vertex AI."""
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cls.BUSINESS_ACCOUNT_PATH
        os.environ["GOOGLE_CLOUD_PROJECT"] = cls.PROJECT_ID
