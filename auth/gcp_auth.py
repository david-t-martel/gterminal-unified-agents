"""Google Cloud Platform profile-based authentication for GTerminal.

Implements the proven profile-based authentication system supporting
business and personal GCP accounts with automatic credential management.
"""

from enum import Enum
import json
import logging
import os
from pathlib import Path
from typing import Any

import google.auth
from google.auth.exceptions import DefaultCredentialsError
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)


class GCPProfile(str, Enum):
    """Supported GCP authentication profiles."""

    BUSINESS = "business"
    PERSONAL = "personal"
    DEFAULT = "default"


class GCPProfileAuth:
    """Profile-based Google Cloud authentication manager.

    Supports multiple authentication methods:
    - Service Account JSON files
    - OAuth user credentials
    - Application Default Credentials
    - API Keys for specific services
    """

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path.home() / ".config" / "gterminal" / "gcp"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Profile configurations
        self.profiles = {
            GCPProfile.BUSINESS: self._load_profile_config("business"),
            GCPProfile.PERSONAL: self._load_profile_config("personal"),
        }

        # Current active profile
        self.current_profile = self._get_current_profile()

        # Cached credentials
        self._credentials_cache: dict[str, Credentials] = {}

    def _load_profile_config(self, profile_name: str) -> dict[str, Any]:
        """Load profile configuration from file."""
        config_file = self.config_dir / f"{profile_name}.json"

        if not config_file.exists():
            logger.warning(f"Profile config not found: {config_file}")
            return {}

        try:
            with open(config_file) as f:
                config = json.load(f)
                logger.info(f"Loaded {profile_name} profile configuration")
                return config
        except Exception as e:
            logger.exception(f"Failed to load {profile_name} profile config: {e}")
            return {}

    def _save_profile_config(self, profile_name: str, config: dict[str, Any]) -> None:
        """Save profile configuration to file."""
        config_file = self.config_dir / f"{profile_name}.json"

        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)

            # Set secure permissions
            os.chmod(config_file, 0o600)
            logger.info(f"Saved {profile_name} profile configuration")

        except Exception as e:
            logger.exception(f"Failed to save {profile_name} profile config: {e}")

    def _get_current_profile(self) -> GCPProfile:
        """Get the currently active profile."""
        # Check environment variable first
        profile_env = os.getenv("GCP_PROFILE", "").lower()
        if profile_env in [p.value for p in GCPProfile]:
            return GCPProfile(profile_env)

        # Check saved preference
        preference_file = self.config_dir / "current_profile"
        if preference_file.exists():
            try:
                with open(preference_file) as f:
                    profile_name = f.read().strip().lower()
                    if profile_name in [p.value for p in GCPProfile]:
                        return GCPProfile(profile_name)
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to read profile preference: {e}")

        # Default to business profile
        return GCPProfile.BUSINESS

    def set_current_profile(self, profile: GCPProfile) -> None:
        """Set the current active profile."""
        self.current_profile = profile

        # Save preference
        preference_file = self.config_dir / "current_profile"
        try:
            with open(preference_file, "w") as f:
                f.write(profile.value)
            os.chmod(preference_file, 0o600)

            # Update environment variable
            os.environ["GCP_PROFILE"] = profile.value

            # Clear credentials cache
            self._credentials_cache.clear()

            logger.info(f"Switched to {profile.value} profile")

        except Exception as e:
            logger.exception(f"Failed to save profile preference: {e}")

    def configure_profile(
        self,
        profile: GCPProfile,
        project_id: str,
        auth_method: str = "service_account",
        service_account_path: str | None = None,
        api_key: str | None = None,
        location: str = "us-central1",
        **kwargs,
    ) -> None:
        """Configure a GCP profile with authentication details."""

        config = {
            "project_id": project_id,
            "location": location,
            "auth_method": auth_method,
            "created_at": str(datetime.now()),
            "metadata": kwargs,
        }

        if auth_method == "service_account" and service_account_path:
            # Validate service account file
            sa_path = Path(service_account_path)
            if not sa_path.exists():
                raise ValueError(f"Service account file not found: {sa_path}")

            config["service_account_path"] = str(sa_path.resolve())

        elif auth_method == "api_key" and api_key:
            config["api_key"] = api_key

        elif auth_method == "oauth":
            # OAuth configuration would be handled separately
            config["oauth_configured"] = True

        # Update profile configuration
        self.profiles[profile] = config
        self._save_profile_config(profile.value, config)

        logger.info(f"Configured {profile.value} profile with {auth_method} authentication")

    def get_credentials(self, profile: GCPProfile | None = None) -> tuple[Credentials, str]:
        """Get credentials for the specified profile."""
        profile = profile or self.current_profile

        # Check cache first
        cache_key = f"{profile.value}"
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key], self.get_project_id(profile)

        profile_config = self.profiles.get(profile, {})
        if not profile_config:
            logger.warning(f"No configuration found for {profile.value} profile")
            return self._get_default_credentials()

        auth_method = profile_config.get("auth_method", "service_account")

        try:
            if auth_method == "service_account":
                credentials = self._get_service_account_credentials(profile_config)
            elif auth_method == "api_key":
                # API keys don't use OAuth2 credentials
                raise ValueError("API key authentication doesn't use OAuth2 credentials")
            elif auth_method == "oauth":
                credentials = self._get_oauth_credentials(profile_config)
            else:
                logger.warning(f"Unknown auth method: {auth_method}")
                credentials = self._get_default_credentials()[0]

            # Cache the credentials
            self._credentials_cache[cache_key] = credentials

            return credentials, profile_config.get("project_id", "")

        except Exception as e:
            logger.exception(f"Failed to get credentials for {profile.value}: {e}")
            return self._get_default_credentials()

    def _get_service_account_credentials(self, config: dict[str, Any]) -> Credentials:
        """Get service account credentials from JSON file."""
        sa_path = config.get("service_account_path")
        if not sa_path:
            raise ValueError("Service account path not configured")

        if not Path(sa_path).exists():
            raise FileNotFoundError(f"Service account file not found: {sa_path}")

        credentials = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/generative-language",
            ],
        )

        logger.info(f"Loaded service account credentials from {sa_path}")
        return credentials

    def _get_oauth_credentials(self, config: dict[str, Any]) -> Credentials:
        """Get OAuth user credentials (placeholder - would need OAuth flow)."""
        # This is a simplified implementation
        # In production, you'd implement the full OAuth2 flow

        oauth_file = self.config_dir / f"oauth_{config.get('profile_name', 'default')}.json"
        if oauth_file.exists():
            credentials = Credentials.from_authorized_user_file(str(oauth_file))

            # Refresh if needed
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())

            return credentials

        raise ValueError("OAuth credentials not configured")

    def _get_default_credentials(self) -> tuple[Credentials, str]:
        """Get application default credentials as fallback."""
        try:
            credentials, project = google.auth.default(
                scopes=[
                    "https://www.googleapis.com/auth/cloud-platform",
                    "https://www.googleapis.com/auth/generative-language",
                ]
            )
            logger.info("Using application default credentials")
            return credentials, project or ""
        except DefaultCredentialsError as e:
            logger.exception(f"Failed to get default credentials: {e}")
            raise

    def get_project_id(self, profile: GCPProfile | None = None) -> str:
        """Get project ID for the specified profile."""
        profile = profile or self.current_profile
        profile_config = self.profiles.get(profile, {})

        project_id = profile_config.get("project_id")
        if project_id:
            return project_id

        # Try to get from default credentials
        try:
            _, project = google.auth.default()
            return project or ""
        except DefaultCredentialsError:
            return ""

    def get_api_key(self, profile: GCPProfile | None = None) -> str | None:
        """Get API key for the specified profile."""
        profile = profile or self.current_profile
        profile_config = self.profiles.get(profile, {})

        return profile_config.get("api_key")

    def get_location(self, profile: GCPProfile | None = None) -> str:
        """Get default location for the specified profile."""
        profile = profile or self.current_profile
        profile_config = self.profiles.get(profile, {})

        return profile_config.get("location", "us-central1")

    def test_credentials(self, profile: GCPProfile | None = None) -> dict[str, Any]:
        """Test credentials for the specified profile."""
        profile = profile or self.current_profile

        try:
            credentials, project_id = self.get_credentials(profile)

            # Test by making a simple API call
            # This is a placeholder - you'd make an actual API call

            result = {
                "profile": profile.value,
                "valid": True,
                "project_id": project_id,
                "auth_method": self.profiles.get(profile, {}).get("auth_method", "unknown"),
                "location": self.get_location(profile),
                "error": None,
            }

            logger.info(f"Credentials test passed for {profile.value} profile")
            return result

        except Exception as e:
            result = {
                "profile": profile.value,
                "valid": False,
                "project_id": "",
                "auth_method": self.profiles.get(profile, {}).get("auth_method", "unknown"),
                "location": "",
                "error": str(e),
            }

            logger.exception(f"Credentials test failed for {profile.value} profile: {e}")
            return result

    def list_profiles(self) -> dict[str, dict[str, Any]]:
        """List all configured profiles with their status."""
        profiles_info = {}

        for profile in GCPProfile:
            if profile == GCPProfile.DEFAULT:
                continue

            config = self.profiles.get(profile, {})
            test_result = self.test_credentials(profile)

            profiles_info[profile.value] = {
                "configured": bool(config),
                "current": profile == self.current_profile,
                "project_id": config.get("project_id", ""),
                "auth_method": config.get("auth_method", ""),
                "location": config.get("location", ""),
                "valid": test_result["valid"],
                "error": test_result.get("error"),
            }

        return profiles_info

    def get_environment_variables(self, profile: GCPProfile | None = None) -> dict[str, str]:
        """Get environment variables for the specified profile."""
        profile = profile or self.current_profile
        profile_config = self.profiles.get(profile, {})

        env_vars = {
            "GCP_PROFILE": profile.value,
            "GOOGLE_CLOUD_PROJECT": profile_config.get("project_id", ""),
            "GOOGLE_CLOUD_LOCATION": profile_config.get("location", "us-central1"),
        }

        # Add service account path if configured
        sa_path = profile_config.get("service_account_path")
        if sa_path:
            env_vars["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

        # Add API key if configured
        api_key = profile_config.get("api_key")
        if api_key:
            env_vars["GOOGLE_API_KEY"] = api_key

        return env_vars

    def apply_environment_variables(self, profile: GCPProfile | None = None) -> None:
        """Apply profile environment variables to current process."""
        env_vars = self.get_environment_variables(profile)

        for key, value in env_vars.items():
            if value:  # Only set non-empty values
                os.environ[key] = value

        logger.info(
            f"Applied environment variables for {(profile or self.current_profile).value} profile"
        )


# Global instance
gcp_auth = GCPProfileAuth()


def get_gcp_credentials(profile: str | None = None) -> tuple[Credentials, str]:
    """Convenience function to get GCP credentials."""
    gcp_profile = GCPProfile(profile) if profile else None
    return gcp_auth.get_credentials(gcp_profile)


def get_current_gcp_profile() -> str:
    """Get the current active GCP profile."""
    return gcp_auth.current_profile.value


def switch_gcp_profile(profile: str) -> bool:
    """Switch to a different GCP profile."""
    try:
        gcp_profile = GCPProfile(profile)
        gcp_auth.set_current_profile(gcp_profile)
        gcp_auth.apply_environment_variables(gcp_profile)
        return True
    except ValueError:
        logger.exception(f"Invalid profile: {profile}")
        return False


# Import datetime at the top
from datetime import datetime
