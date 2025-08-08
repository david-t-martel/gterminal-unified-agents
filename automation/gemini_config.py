"""Centralized Gemini/Vertex AI configuration for automation agents.
Uses service account authentication for production use.
"""

import logging
import os
from pathlib import Path

import google.auth
from google.auth import credentials as auth_credentials
import vertexai
from vertexai.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL = "gemini-2.0-flash-exp"
DEFAULT_LOCATION = "us-central1"

# Cache for initialized models
_model_cache: dict[str, GenerativeModel] = {}


def get_credentials() -> tuple[auth_credentials.Credentials, str]:
    """Get Google Cloud credentials using service account.

    Returns:
        Tuple of (credentials, project_id)

    """
    # Clear any existing Windows-style paths
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        current_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        if current_path.startswith(("C:\\", "c:\\")):
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            logger.info("Cleared Windows-style credential path")

    # First try to get default credentials (service account)
    try:
        credentials, project_id = google.auth.default()
        logger.info(f"Using default credentials for project: {project_id}")
        return credentials, project_id
    except Exception as e:
        logger.info(f"Default credentials not found: {e}")

    # Try business profile service account
    business_sa_path = Path.home() / ".auth" / "business" / "service-account-key.json"
    if business_sa_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(business_sa_path)
        credentials, project_id = google.auth.default()
        logger.info(f"Using business service account for project: {project_id}")
        return credentials, project_id

    # Try personal profile service account
    personal_sa_path = Path.home() / ".auth" / "personal" / "service-account-key.json"
    if personal_sa_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(personal_sa_path)
        credentials, project_id = google.auth.default()
        logger.info(f"Using personal service account for project: {project_id}")
        return credentials, project_id

    msg = "No valid Google Cloud credentials found. Please set up service account authentication."
    raise ValueError(
        msg,
    )


def get_gemini_model(
    model_name: str | None = None,
    location: str | None = None,
) -> GenerativeModel:
    """Get initialized Gemini model using Vertex AI.

    Args:
        model_name: Model to use (defaults to gemini-2.0-flash-exp)
        location: GCP location (defaults to us-central1)

    Returns:
        Initialized GenerativeModel instance

    """
    model_name = model_name or DEFAULT_MODEL
    location = location or DEFAULT_LOCATION

    # Check cache
    cache_key = f"{model_name}:{location}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Get credentials and initialize Vertex AI
    credentials, project_id = get_credentials()

    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location, credentials=credentials)

    # Create model
    model = GenerativeModel(model_name)

    # Cache for reuse
    _model_cache[cache_key] = model

    logger.info(f"Initialized Gemini model: {model_name} in {location}")
    return model


def get_model_for_task(task_type: str) -> GenerativeModel:
    """Get the best model for a specific task type.

    Args:
        task_type: Type of task (code_review, test_generation, documentation, etc.)

    Returns:
        Initialized GenerativeModel instance

    """
    # Task-specific model selection
    task_models = {
        "code_review": "gemini-2.0-flash-exp",
        "test_generation": "gemini-2.0-flash-exp",
        "documentation": "gemini-2.0-flash-exp",
        "analysis": "gemini-2.0-flash-exp",
    }

    model_name = task_models.get(task_type, DEFAULT_MODEL)
    return get_gemini_model(model_name)
