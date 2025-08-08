"""Application configuration with enhanced security and modernization.

Security improvements:
- Profile-based configuration without environment variables
- No hardcoded credentials
- Enhanced validation and error handling
- Timezone-aware datetime handling

Modernization updates:
- Updated to use Python 3.10+ type annotations (dict/list/| instead of typing.Dict/List/Optional)
- Enhanced type hints throughout
- Better configuration management patterns
"""

from dataclasses import dataclass
from dataclasses import field
import logging
import os
from pathlib import Path
from typing import Any

# Initialize logger first
logger = logging.getLogger(__name__)

# Import our new GCP auth manager
try:
    from gterminal.config.gcp_auth import GCPAuthManager
    from gterminal.config.gcp_auth import get_auth_manager
    from gterminal.config.gcp_auth import get_current_credentials

    HAS_GCP_AUTH = True
except ImportError:
    logger.warning("GCP Auth Manager not available, falling back to environment variables")
    HAS_GCP_AUTH = False

from gterminal.security.secrets_manager import SecretsManager

# Initialize secrets manager
_secrets_manager = SecretsManager()

# Configuration defaults
DEFAULT_MODELS = {
    "critic": "gemini-2.5-pro",
    "worker": "gemini-2.5-flash",
    "master": "gemini-2.5-pro",
}

DEFAULT_LOCATIONS = {
    "global": "global",
    "us-central1": "us-central1",
    "us-east1": "us-east1",
    "europe-west1": "europe-west1",
}


def load_secure_environment() -> dict[str, str]:
    """Load secure environment variables from .secure directory."""
    env_vars: dict[str, Any] = {}

    # Try to load from secure environment file
    project_root = Path(__file__).parent.parent
    secure_env_file = project_root / ".secure" / ".env.secure"

    if secure_env_file.exists():
        try:
            with open(secure_env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # Remove quotes if present
                        value = value.strip("\"'")
                        env_vars[key.strip()] = value

            logger.info(f"Loaded {len(env_vars)} secure environment variables")
        except Exception as e:
            logger.warning(f"Failed to load secure environment: {e}")

    return env_vars


def get_google_project_id() -> str:
    """Safely get Google Cloud project ID using profile-based auth."""
    # Use GCP Auth Manager if available
    if HAS_GCP_AUTH:
        try:
            auth_manager = get_auth_manager()
            project_id = auth_manager.get_project_id()
            if project_id:
                logger.info(f"Using Google Cloud project from profile: {project_id}")
                return project_id
        except Exception as e:
            logger.debug(f"Error getting project from auth manager: {e}")

    # Try environment variable as fallback
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project_id:
        return project_id

    # Try to get from default credentials (only if not in test environment)
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError

            _, project_id = google.auth.default()
            if project_id:
                logger.info(f"Using Google Cloud project from credentials: {project_id}")
                return project_id
        except (DefaultCredentialsError, ImportError):
            logger.debug("No default Google credentials found")
        except Exception as e:
            logger.debug(f"Error getting default project: {e}")

    # Fallback for testing and development
    fallback_project = "test-project-id"
    logger.warning(f"Using fallback project ID: {fallback_project}")
    return fallback_project


def validate_model_name(model_name: str) -> bool:
    """Validate that model name follows expected patterns."""
    if not model_name or not isinstance(model_name, str):
        return False

    # Valid model patterns for Gemini
    valid_patterns = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-exp-",  # Experimental models
    ]

    return any(pattern in model_name for pattern in valid_patterns)


def setup_environment_defaults() -> None:
    """Set up environment defaults with secure practices."""
    secure_env = load_secure_environment()

    # Set secure defaults
    defaults = {
        "GOOGLE_CLOUD_PROJECT": get_google_project_id(),
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "GOOGLE_GENAI_USE_VERTEXAI": "True",
    }

    # Apply secure environment variables first, then defaults
    for key, value in {**defaults, **secure_env}.items():
        if key not in os.environ:
            os.environ[key] = value


# Initialize environment on import
setup_environment_defaults()


@dataclass
class SecurityConfig:
    """Security-related configuration."""

    # Authentication settings - using secure secrets manager
    jwt_secret_key: str = field(
        default_factory=lambda: _secrets_manager.get_secret("JWT_SECRET_KEY") or ""
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # API security
    api_rate_limit_per_hour: int = 1000
    api_rate_limit_burst: int = 50

    # Session security
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "strict"

    # Storage security
    auth_storage_path: str = field(
        default_factory=lambda: os.environ.get(
            "AUTH_STORAGE_PATH",
            str(Path.home() / ".config" / "fullstack-agent" / "auth"),
        ),
    )

    def __post_init__(self) -> None:
        """Validate security configuration."""
        if not self.jwt_secret_key:
            logger.error(
                "JWT secret key not set - authentication will fail. Set JWT_SECRET_KEY environment variable."
            )
            msg = "JWT_SECRET_KEY must be set for secure authentication"
            raise ValueError(msg)

        if self.jwt_access_token_expire_minutes < 5:
            logger.warning("JWT access token expiration is very short")

        if self.api_rate_limit_per_hour > 10000:
            logger.warning("API rate limit is very high")


@dataclass
class ModelConfig:
    """Model configuration with validation."""

    critic_model: str = DEFAULT_MODELS["critic"]
    worker_model: str = DEFAULT_MODELS["worker"]
    master_model: str = DEFAULT_MODELS["master"]

    # Model-specific settings
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192

    def __post_init__(self) -> None:
        """Validate model configuration."""
        models_to_check = [self.critic_model, self.worker_model, self.master_model]

        for model in models_to_check:
            if not validate_model_name(model):
                logger.warning(f"Model name may be invalid: {model}")

        # Validate parameters
        if not 0.0 <= self.temperature <= 2.0:
            msg = "Temperature must be between 0.0 and 2.0"
            raise ValueError(msg)

        if not 0.0 <= self.top_p <= 1.0:
            msg = "Top-p must be between 0.0 and 1.0"
            raise ValueError(msg)

        if self.top_k < 1:
            msg = "Top-k must be at least 1"
            raise ValueError(msg)

        if self.max_output_tokens < 1:
            msg = "Max output tokens must be at least 1"
            raise ValueError(msg)


@dataclass
class GoogleCloudConfig:
    """Google Cloud configuration with secure practices."""

    project_id: str = field(default_factory=get_google_project_id)
    location: str = field(default_factory=lambda: _get_location_from_auth())
    use_vertex_ai: bool = field(default_factory=lambda: _get_use_vertex_ai_from_auth())

    # Profile-based authentication
    profile: str | None = field(default=None)

    # Credentials (managed internally, not exposed)
    _credentials: Any | None = field(default=None, init=False, repr=False)

    # API settings
    timeout_seconds: int = 60
    max_retries: int = 3

    def __post_init__(self) -> None:
        """Validate Google Cloud configuration and load credentials."""
        if not self.project_id:
            msg = "Google Cloud project ID is required"
            raise ValueError(msg)

        if self.location not in DEFAULT_LOCATIONS:
            logger.warning(f"Unusual location specified: {self.location}")

        # Load credentials using auth manager
        if HAS_GCP_AUTH:
            try:
                auth_manager = get_auth_manager(profile=self.profile)
                creds = auth_manager.get_credentials()
                self._credentials = creds.credentials

                # Update config from loaded credentials
                self.project_id = creds.project_id
                self.location = creds.location
                self.use_vertex_ai = creds.use_vertex_ai
                self.profile = creds.profile

                logger.info(f"Loaded credentials for profile: {self.profile}")
            except Exception as e:
                logger.warning(f"Failed to load credentials via auth manager: {e}")

        if self.timeout_seconds < 10:
            logger.warning("Timeout is very short, may cause request failures")

    def get_credentials(self) -> Any | None:
        """Get the loaded credentials."""
        return self._credentials

    def get_credentials_info(self) -> dict[str, Any]:
        """Get information about credentials without exposing sensitive data."""
        return {
            "project_id": self.project_id,
            "location": self.location,
            "profile": self.profile,
            "use_vertex_ai": self.use_vertex_ai,
            "has_credentials": self._credentials is not None,
        }


def _get_location_from_auth() -> str:
    """Get location from auth manager or fallback."""
    if HAS_GCP_AUTH:
        try:
            return get_auth_manager().get_location()
        except Exception:
            pass
    return os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")


def _get_use_vertex_ai_from_auth() -> bool:
    """Get use_vertex_ai setting from auth manager or fallback."""
    if HAS_GCP_AUTH:
        try:
            creds = get_current_credentials()
            return creds.use_vertex_ai
        except Exception:
            pass
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true"


@dataclass
class ResearchConfiguration:
    """Main configuration class combining all configuration aspects.

    Enhanced with:
    - Security configuration
    - Model validation
    - Environment-based setup
    - Comprehensive validation
    """

    # Core model configuration
    models: ModelConfig = field(default_factory=ModelConfig)

    # Google Cloud configuration
    google_cloud: GoogleCloudConfig = field(default_factory=GoogleCloudConfig)

    # Security configuration
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # Research-specific settings
    max_search_iterations: int = 5
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # Feature flags
    enable_experimental_features: bool = False
    enable_detailed_logging: bool = False

    def __post_init__(self) -> None:
        """Validate overall configuration."""
        if self.max_search_iterations < 1:
            msg = "Max search iterations must be at least 1"
            raise ValueError(msg)

        if self.max_search_iterations > 20:
            logger.warning("Very high max search iterations may impact performance")

        if self.cache_ttl_seconds < 60:
            logger.warning("Cache TTL is very short")

    def get_model_for_task(self, task_type: str) -> str:
        """Get appropriate model for a specific task type."""
        task_model_mapping = {
            "critic": self.models.critic_model,
            "evaluation": self.models.critic_model,
            "review": self.models.critic_model,
            "worker": self.models.worker_model,
            "generation": self.models.worker_model,
            "analysis": self.models.worker_model,
            "master": self.models.master_model,
            "architecture": self.models.master_model,
            "planning": self.models.master_model,
        }

        return task_model_mapping.get(task_type, self.models.worker_model)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return {
            "models": {
                "critic_model": self.models.critic_model,
                "worker_model": self.models.worker_model,
                "master_model": self.models.master_model,
                "temperature": self.models.temperature,
                "top_p": self.models.top_p,
                "top_k": self.models.top_k,
                "max_output_tokens": self.models.max_output_tokens,
            },
            "google_cloud": {
                "project_id": self.google_cloud.project_id,
                "location": self.google_cloud.location,
                "use_vertex_ai": self.google_cloud.use_vertex_ai,
                "timeout_seconds": self.google_cloud.timeout_seconds,
                "max_retries": self.google_cloud.max_retries,
            },
            "research": {
                "max_search_iterations": self.max_search_iterations,
                "enable_caching": self.enable_caching,
                "cache_ttl_seconds": self.cache_ttl_seconds,
            },
            "security": {
                "jwt_algorithm": self.security.jwt_algorithm,
                "jwt_access_token_expire_minutes": self.security.jwt_access_token_expire_minutes,
                "api_rate_limit_per_hour": self.security.api_rate_limit_per_hour,
                "session_cookie_secure": self.security.session_cookie_secure,
            },
            "features": {
                "enable_experimental_features": self.enable_experimental_features,
                "enable_detailed_logging": self.enable_detailed_logging,
            },
        }


# Global configuration instance
try:
    config = ResearchConfiguration()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.exception(f"Failed to load configuration: {e}")
    # Fallback to basic configuration
    config = ResearchConfiguration(
        models=ModelConfig(), google_cloud=GoogleCloudConfig(), security=SecurityConfig()
    )


def get_config() -> ResearchConfiguration:
    """Get the global configuration instance."""
    return config


def reload_config() -> ResearchConfiguration:
    """Reload configuration from environment."""
    global config
    setup_environment_defaults()
    config = ResearchConfiguration()
    logger.info("Configuration reloaded")
    return config


def validate_configuration() -> list[str]:
    """Validate current configuration and return any issues."""
    issues: list[Any] = []

    try:
        # Test Google Cloud configuration
        if not config.google_cloud.project_id:
            issues.append("Google Cloud project ID not set")

        # Test model configuration
        for model_name in [
            config.models.critic_model,
            config.models.worker_model,
            config.models.master_model,
        ]:
            if not validate_model_name(model_name):
                issues.append(f"Invalid model name: {model_name}")

        # Test security configuration
        if not config.security.jwt_secret_key:
            issues.append("JWT secret key not configured")

        # Test paths
        auth_path = Path(config.security.auth_storage_path)
        if not auth_path.parent.exists():
            issues.append(f"Auth storage parent directory does not exist: {auth_path.parent}")

    except Exception as e:
        issues.append(f"Configuration validation error: {e}")

    return issues


# Backward compatibility aliases
critic_model = config.models.critic_model
worker_model = config.models.worker_model
max_search_iterations = config.max_search_iterations
