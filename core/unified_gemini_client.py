#!/usr/bin/env python3
"""Unified Gemini Client - Reliable authentication and access to Gemini models.

This module provides a simplified, reliable interface to Gemini models that:
1. Works with the existing GCP profile system
2. Provides fallback authentication mechanisms
3. Handles both Vertex AI and Google AI Studio access
4. Includes comprehensive error handling and retry logic
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

try:
    from google.auth import credentials as auth_credentials
    from google.oauth2 import service_account
    import vertexai
    from vertexai.generative_models import GenerativeModel

    HAS_VERTEX_AI = True
except ImportError:
    HAS_VERTEX_AI = False
    vertexai = None
    GenerativeModel = None
    auth_credentials = None
    service_account = None

try:
    import google.generativeai as genai

    HAS_GOOGLE_AI = True
except ImportError:
    HAS_GOOGLE_AI = False
    genai = None

logger = logging.getLogger(__name__)


class GeminiAuthenticationError(Exception):
    """Raised when Gemini authentication fails."""


class GeminiClient:
    """Unified client for accessing Gemini models with reliable authentication.

    This client automatically handles:
    - Service account authentication for Vertex AI
    - API key authentication for Google AI Studio
    - Profile switching between business and personal accounts
    - Fallback mechanisms when primary auth fails
    - Model initialization and configuration
    """

    # Profile configurations
    PROFILES = {
        "business": {
            "project_id": "auricleinc-gemini",
            "location": "us-central1",
            "service_account_path": "/home/david/.auth/business/service-account-key.json",
            "preferred_auth": "service_account",
            "use_vertex_ai": True,
            "fallback_api_key": None,
        },
        "personal": {
            "project_id": "dtm-gemini-ai",
            "location": "us-central1",
            "service_account_path": "/home/david/.auth/personal/client-secret.json",
            "preferred_auth": "api_key",
            "use_vertex_ai": False,
            "fallback_api_key": "/home/david/.gcp/personal/api_key.txt",
        },
    }

    # Default model configurations
    DEFAULT_MODELS = {
        "vertex_ai": "gemini-2.0-flash-exp",
        "google_ai": "gemini-2.0-flash-exp",
        "fallback": "gemini-1.5-pro",
    }

    def __init__(self, profile: str = "business", model_name: str | None = None) -> None:
        """Initialize the Gemini client.

        Args:
            profile: The profile to use ('business' or 'personal')
            model_name: Optional model name override

        """
        self.profile = profile
        self.model_name = model_name
        self.model = None
        self.auth_method = None
        self.project_id = None
        self.credentials = None

        # Validate profile
        if profile not in self.PROFILES:
            msg = f"Invalid profile: {profile}. Must be 'business' or 'personal'"
            raise ValueError(msg)

        self.profile_config = self.PROFILES[profile]
        logger.info(f"Initializing Gemini client with profile: {profile}")

    def authenticate(self) -> bool:
        """Authenticate with Google services using multiple fallback mechanisms.

        Returns:
            True if authentication successful, False otherwise

        """
        auth_methods = [
            ("service_account", self._authenticate_service_account),
            ("api_key", self._authenticate_api_key),
            ("environment", self._authenticate_environment),
            ("default", self._authenticate_default),
        ]

        # Try preferred method first
        preferred = self.profile_config["preferred_auth"]
        auth_methods = [m for m in auth_methods if m[0] == preferred] + [
            m for m in auth_methods if m[0] != preferred
        ]

        for method_name, auth_func in auth_methods:
            try:
                logger.info(f"Attempting authentication via: {method_name}")
                if auth_func():
                    self.auth_method = method_name
                    logger.info(f"✅ Authentication successful via: {method_name}")
                    return True
            except Exception as e:
                logger.debug(f"Authentication method {method_name} failed: {e}")
                continue

        logger.error("❌ All authentication methods failed")
        return False

    def _authenticate_service_account(self) -> bool:
        """Authenticate using service account key file."""
        if not HAS_VERTEX_AI:
            logger.debug("Vertex AI not available for service account auth")
            return False

        sa_path = Path(self.profile_config["service_account_path"])
        if not sa_path.exists():
            logger.debug(f"Service account key not found: {sa_path}")
            return False

        try:
            # Load service account credentials
            with open(sa_path) as f:
                key_data = json.load(f)

            self.credentials = service_account.Credentials.from_service_account_info(
                key_data,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            # Initialize Vertex AI
            vertexai.init(
                project=self.profile_config["project_id"],
                location=self.profile_config["location"],
                credentials=self.credentials,
            )

            self.project_id = self.profile_config["project_id"]
            logger.info(f"Service account auth successful for project: {self.project_id}")
            return True

        except Exception as e:
            logger.debug(f"Service account authentication failed: {e}")
            return False

    def _authenticate_api_key(self) -> bool:
        """Authenticate using API key for Google AI Studio."""
        if not HAS_GOOGLE_AI:
            logger.debug("Google AI not available for API key auth")
            return False

        api_key = None

        # Try to load API key from file
        api_key_file = self.profile_config.get("fallback_api_key")
        if api_key_file and Path(api_key_file).exists():
            try:
                api_key = Path(api_key_file).read_text().strip()
            except Exception as e:
                logger.debug(f"Failed to read API key file: {e}")

        # Try environment variables
        if not api_key:
            for var_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GENERATIVE_AI_API_KEY"]:
                api_key = os.getenv(var_name)
                if api_key:
                    logger.debug(f"Found API key in {var_name}")
                    break

        if not api_key:
            logger.debug("No API key found")
            return False

        try:
            # Configure Google AI with API key
            genai.configure(api_key=api_key)

            # Test the API key with a simple call
            models = list(genai.list_models())
            if models:
                logger.info("API key authentication successful")
                return True
            logger.debug("API key test failed - no models returned")
            return False

        except Exception as e:
            logger.debug(f"API key authentication failed: {e}")
            return False

    def _authenticate_environment(self) -> bool:
        """Authenticate using environment variables."""
        if not HAS_VERTEX_AI:
            return False

        project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            logger.debug("No project ID in environment")
            return False

        try:
            # Try default credentials with environment project
            vertexai.init(project=project_id, location="us-central1")
            self.project_id = project_id
            logger.info(f"Environment authentication successful for project: {project_id}")
            return True
        except Exception as e:
            logger.debug(f"Environment authentication failed: {e}")
            return False

    def _authenticate_default(self) -> bool:
        """Authenticate using Google default credentials."""
        if not HAS_VERTEX_AI:
            return False

        try:
            import google.auth

            credentials, project = google.auth.default()

            if not project:
                project = self.profile_config["project_id"]

            vertexai.init(
                project=project, location=self.profile_config["location"], credentials=credentials
            )

            self.project_id = project
            self.credentials = credentials
            logger.info(f"Default authentication successful for project: {project}")
            return True

        except Exception as e:
            logger.debug(f"Default authentication failed: {e}")
            return False

    def initialize_model(self, model_name: str | None = None) -> bool:
        """Initialize the Gemini model after authentication.

        Args:
            model_name: Optional model name override

        Returns:
            True if model initialization successful

        """
        if not model_name:
            model_name = self.model_name

        # Choose model based on auth method
        if not model_name:
            if self.auth_method in ["service_account", "environment", "default"]:
                model_name = self.DEFAULT_MODELS["vertex_ai"]
            else:
                model_name = self.DEFAULT_MODELS["google_ai"]

        try:
            if self.auth_method in ["service_account", "environment", "default"]:
                # Use Vertex AI
                if not HAS_VERTEX_AI:
                    msg = "Vertex AI not available"
                    raise GeminiAuthenticationError(msg)

                self.model = GenerativeModel(model_name)
                logger.info(f"Vertex AI model initialized: {model_name}")

            else:
                # Use Google AI Studio
                if not HAS_GOOGLE_AI:
                    msg = "Google AI not available"
                    raise GeminiAuthenticationError(msg)

                self.model = genai.GenerativeModel(model_name)
                logger.info(f"Google AI model initialized: {model_name}")

            return True

        except Exception as e:
            logger.exception(f"Model initialization failed: {e}")
            return False

    def get_model(self) -> Any:
        """Get the initialized model, authenticating and initializing if needed.

        Returns:
            The initialized Gemini model

        Raises:
            GeminiAuthenticationError: If authentication or initialization fails

        """
        if self.model is None:
            if not self.authenticate():
                msg = "Failed to authenticate with Google services"
                raise GeminiAuthenticationError(msg)

            if not self.initialize_model():
                msg = "Failed to initialize Gemini model"
                raise GeminiAuthenticationError(msg)

        return self.model

    async def generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using the Gemini model.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            Generated content as string

        """
        model = self.get_model()

        try:
            # Handle different model types
            if self.auth_method in ["service_account", "environment", "default"]:
                # Vertex AI model
                response = await model.generate_content_async(prompt, **kwargs)
                return response.text
            # Google AI model
            response = await model.generate_content_async(prompt, **kwargs)
            return response.text

        except Exception as e:
            logger.exception(f"Content generation failed: {e}")
            raise

    def generate_content_sync(self, prompt: str, **kwargs) -> str:
        """Generate content synchronously.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            Generated content as string

        """
        model = self.get_model()

        try:
            response = model.generate_content(prompt, **kwargs)
            return response.text
        except Exception as e:
            logger.exception(f"Sync content generation failed: {e}")
            raise

    def get_status(self) -> dict[str, Any]:
        """Get detailed status information about the client."""
        return {
            "profile": self.profile,
            "auth_method": self.auth_method,
            "project_id": self.project_id,
            "model_initialized": self.model is not None,
            "vertex_ai_available": HAS_VERTEX_AI,
            "google_ai_available": HAS_GOOGLE_AI,
            "service_account_exists": Path(self.profile_config["service_account_path"]).exists(),
            "api_key_available": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        }


def get_gemini_client(profile: str = "business", model_name: str | None = None) -> GeminiClient:
    """Get a configured Gemini client.

    Args:
        profile: Profile to use ('business' or 'personal')
        model_name: Optional model name override

    Returns:
        Configured GeminiClient instance

    """
    client = GeminiClient(profile=profile, model_name=model_name)

    # Attempt authentication immediately
    if not client.authenticate():
        logger.warning("Authentication failed during client creation")

    return client


# Convenience functions for backward compatibility
def get_business_client(model_name: str | None = None) -> GeminiClient:
    """Get a Gemini client configured for business use."""
    return get_gemini_client(profile="business", model_name=model_name)


def get_personal_client(model_name: str | None = None) -> GeminiClient:
    """Get a Gemini client configured for personal use."""
    return get_gemini_client(profile="personal", model_name=model_name)


async def test_gemini_access(profile: str = "business") -> dict[str, Any]:
    """Test Gemini access with comprehensive diagnostics.

    Args:
        profile: Profile to test

    Returns:
        Dictionary with test results

    """
    results = {
        "profile": profile,
        "authentication_successful": False,
        "model_initialization_successful": False,
        "content_generation_successful": False,
        "auth_method": None,
        "error_message": None,
        "status": {},
    }

    try:
        # Create client
        client = GeminiClient(profile=profile)

        # Test authentication
        if client.authenticate():
            results["authentication_successful"] = True
            results["auth_method"] = client.auth_method

            # Test model initialization
            if client.initialize_model():
                results["model_initialization_successful"] = True

                # Test content generation
                try:
                    response = client.generate_content_sync(
                        "Hello! Please respond with 'Gemini access test successful' to confirm you're working.",
                    )
                    if "successful" in response.lower():
                        results["content_generation_successful"] = True
                    else:
                        results["error_message"] = f"Unexpected response: {response}"
                except Exception as e:
                    results["error_message"] = f"Content generation failed: {e}"
            else:
                results["error_message"] = "Model initialization failed"
        else:
            results["error_message"] = "Authentication failed"

        results["status"] = client.get_status()

    except Exception as e:
        results["error_message"] = f"Test failed with exception: {e}"

    return results


if __name__ == "__main__":
    # Quick test when run directly
    import asyncio

    async def main() -> None:
        # Test business profile
        results = await test_gemini_access("business")
        for _key, _value in results.items():
            pass

        # Test personal profile
        results = await test_gemini_access("personal")
        for _key, _value in results.items():
            pass

    asyncio.run(main())
