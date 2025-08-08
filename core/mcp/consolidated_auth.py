"""
Consolidated Authentication System for MCP Server Framework

Provides unified authentication management for Google Cloud Platform services,
integrating with gcp-profile system and service account credentials.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any

from google.auth import default
from google.auth.credentials import Credentials
import google.auth.transport.requests
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


class AuthProfile(Enum):
    """Authentication profile types"""

    BUSINESS = "business"
    PERSONAL = "personal"


@dataclass
class AuthConfig:
    """Authentication configuration data"""

    profile: AuthProfile
    account: str
    project: str
    credentials_path: str | None = None
    api_keys: dict[str, str] | None = None


class ConsolidatedAuth:
    """
    Unified authentication system for MCP servers.

    Integrates with the gcp-profile system to provide consistent authentication
    across all MCP servers and agents in the fullstack system.
    """

    def __init__(self, profile: str | None = None):
        """
        Initialize consolidated authentication.

        Args:
            profile: Authentication profile ('business' or 'personal').
                    If None, auto-detects from environment.
        """
        self.profile = self._detect_profile(profile)
        self.config = self._load_profile_config()
        self._credentials: Credentials | None = None
        self._validated = False

    def _detect_profile(self, profile: str | None) -> AuthProfile:
        """Detect active authentication profile"""
        if profile:
            return AuthProfile(profile.lower())

        # Check environment variables set by gcp-profile
        if os.getenv("GCP_ACTIVE_PROFILE"):
            return AuthProfile(os.getenv("GCP_ACTIVE_PROFILE"))

        # Check for business credentials (default preference)
        business_creds = Path.home() / ".auth" / "business" / "service-account-key.json"
        if business_creds.exists():
            return AuthProfile.BUSINESS

        return AuthProfile.PERSONAL

    def _load_profile_config(self) -> AuthConfig:
        """Load configuration for the active profile"""
        profile_dir = Path.home() / ".gcp" / self.profile.value
        config_file = profile_dir / f"{self.profile.value}.env"

        if not config_file.exists():
            logger.warning(f"Profile config not found: {config_file}")
            return self._create_default_config()

        # Parse environment file
        config_data = {}
        with open(config_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    config_data[key.strip()] = value.strip().strip('"')

        # Load API keys from secure storage
        api_keys = self._load_api_keys()

        return AuthConfig(
            profile=self.profile,
            account=config_data.get("GOOGLE_ACCOUNT", ""),
            project=config_data.get("GOOGLE_CLOUD_PROJECT", ""),
            credentials_path=config_data.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
            api_keys=api_keys,
        )

    def _create_default_config(self) -> AuthConfig:
        """Create default configuration when profile config is missing"""
        if self.profile == AuthProfile.BUSINESS:
            return AuthConfig(
                profile=self.profile,
                account="david.martel@auricleinc.com",
                project="auricleinc-gemini",
                credentials_path=str(
                    Path.home() / ".auth" / "business" / "service-account-key.json"
                ),
            )
        else:
            return AuthConfig(
                profile=self.profile,
                account="davidmartel07@gmail.com",
                project="dtm-gemini-ai",
                api_keys=self._load_api_keys(),
            )

    def _load_api_keys(self) -> dict[str, str]:
        """Load API keys from environment and secure storage"""
        api_keys = {}

        # Standard API key environment variables
        key_vars = [
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "VERTEX_AI_API_KEY",
            "GENERATIVE_AI_API_KEY",
        ]

        for var in key_vars:
            if value := os.getenv(var):
                api_keys[var] = value

        return api_keys

    async def authenticate(self) -> Credentials:
        """
        Authenticate and return credentials.

        Returns:
            Google Cloud credentials object

        Raises:
            ValueError: If authentication fails
        """
        if self._credentials and self._validated:
            return self._credentials

        try:
            if self.config.credentials_path and Path(self.config.credentials_path).exists():
                # Service account authentication
                self._credentials = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path,
                    scopes=[
                        "https://www.googleapis.com/auth/cloud-platform",
                        "https://www.googleapis.com/auth/generative-language",
                    ],
                )
                logger.info(f"Authenticated with service account: {self.config.credentials_path}")

            else:
                # Default authentication (ADC)
                self._credentials, project = default(
                    scopes=[
                        "https://www.googleapis.com/auth/cloud-platform",
                        "https://www.googleapis.com/auth/generative-language",
                    ]
                )
                logger.info(f"Authenticated with default credentials for project: {project}")

            # Validate credentials
            await self._validate_credentials()
            self._validated = True

            return self._credentials

        except Exception as e:
            logger.exception(f"Authentication failed: {e}")
            raise ValueError(f"Authentication failed for profile {self.profile.value}: {e}")

    async def _validate_credentials(self):
        """Validate that credentials work with Google Cloud"""
        if not self._credentials:
            raise ValueError("No credentials to validate")

        try:
            request = google.auth.transport.requests.Request()
            self._credentials.refresh(request)
            logger.info("Credentials validation successful")

        except Exception as e:
            logger.exception(f"Credentials validation failed: {e}")
            raise

    def get_environment_vars(self) -> dict[str, str]:
        """
        Get environment variables for MCP server configuration.

        Returns:
            Dictionary of environment variables
        """
        env_vars = {
            "GOOGLE_CLOUD_PROJECT": self.config.project,
            "GCP_ACTIVE_PROFILE": self.profile.value,
        }

        if self.config.credentials_path:
            env_vars["GOOGLE_APPLICATION_CREDENTIALS"] = self.config.credentials_path

        if self.config.api_keys:
            env_vars.update(self.config.api_keys)

        return env_vars

    def get_auth_info(self) -> dict[str, Any]:
        """
        Get authentication information for debugging and monitoring.

        Returns:
            Dictionary with authentication details
        """
        return {
            "profile": self.profile.value,
            "account": self.config.account,
            "project": self.config.project,
            "credentials_path": self.config.credentials_path,
            "has_api_keys": bool(self.config.api_keys),
            "api_key_count": len(self.config.api_keys) if self.config.api_keys else 0,
            "validated": self._validated,
        }

    def sync_with_gcloud(self) -> bool:
        """
        Ensure gcloud configuration matches the active profile.

        Returns:
            True if sync was successful, False otherwise
        """
        try:
            import subprocess

            # Set gcloud account
            result = subprocess.run(
                ["gcloud", "config", "set", "account", self.config.account],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Failed to set gcloud account: {result.stderr}")
                return False

            # Set gcloud project
            result = subprocess.run(
                ["gcloud", "config", "set", "project", self.config.project],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Failed to set gcloud project: {result.stderr}")
                return False

            logger.info(f"Successfully synced gcloud with {self.profile.value} profile")
            return True

        except Exception as e:
            logger.exception(f"Failed to sync with gcloud: {e}")
            return False


# Singleton instance for global access
_global_auth: ConsolidatedAuth | None = None


def get_auth(profile: str | None = None) -> ConsolidatedAuth:
    """
    Get the global authentication instance.

    Args:
        profile: Authentication profile to use

    Returns:
        ConsolidatedAuth instance
    """
    global _global_auth

    if _global_auth is None or (profile and _global_auth.profile.value != profile):
        _global_auth = ConsolidatedAuth(profile)

    return _global_auth


async def authenticate(profile: str | None = None) -> Credentials:
    """
    Convenience function to authenticate with the specified profile.

    Args:
        profile: Authentication profile to use

    Returns:
        Google Cloud credentials
    """
    auth = get_auth(profile)
    return await auth.authenticate()
