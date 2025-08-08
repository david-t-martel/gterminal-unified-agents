"""Centralized Configuration Management System for My Fullstack Agent.

This module provides a comprehensive, type-safe configuration management system that:
- Consolidates all scattered configuration files
- Provides environment-specific overrides
- Implements comprehensive validation
- Prevents configuration drift
- Integrates with CI/CD pipeline

Architecture:
- Base configuration classes using Pydantic v2
- Environment-specific configuration layers
- Configuration validation with detailed error reporting
- Secure credential management
- Hot-reload capability for development

Usage:
    from gterminal.config.centralized_config import get_config

    config = get_config()
    db_url = config.database.url
    api_timeout = config.performance.api_timeout
"""

from enum import Enum
import logging
import os
from pathlib import Path
import secrets
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
import yaml

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseConfig(BaseModel):
    """Database configuration with connection pooling."""

    model_config = ConfigDict(str_strip_whitespace=True)

    url: str = Field(
        default="postgresql://localhost:5432/my_fullstack_agent", description="Database URL"
    )
    pool_size: int = Field(default=10, ge=1, le=50, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")
    pool_pre_ping: bool = Field(default=True, description="Enable connection health checks")
    echo: bool = Field(default=False, description="Enable SQL query logging")

    # Environment-specific overrides
    pool_size_dev: int = Field(default=5, ge=1, le=20)
    pool_size_test: int = Field(default=2, ge=1, le=5)
    pool_size_prod: int = Field(default=20, ge=5, le=50)


class RedisConfig(BaseModel):
    """Redis configuration for caching and session storage."""

    model_config = ConfigDict(str_strip_whitespace=True)

    url: str = Field(default="redis://localhost:6379", description="Redis URL")
    password: SecretStr | None = Field(default=None, description="Redis password")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    max_connections: int = Field(default=20, ge=1, le=100, description="Max Redis connections")
    socket_timeout: int = Field(default=5, ge=1, le=30, description="Socket timeout in seconds")
    decode_responses: bool = Field(default=True, description="Decode Redis responses to strings")

    # Cache-specific settings
    default_ttl: int = Field(default=3600, ge=60, le=86400, description="Default TTL in seconds")
    key_prefix: str = Field(default="fullstack_agent:", description="Redis key prefix")


class SecurityConfig(BaseModel):
    """Security configuration with JWT and encryption settings."""

    model_config = ConfigDict(str_strip_whitespace=True)

    # JWT Configuration
    jwt_secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(64)),
        description="JWT secret key",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_access_token_expire_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Access token expiration in minutes",
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7, ge=1, le=30, description="Refresh token expiration in days"
    )

    # API Security
    api_rate_limit_per_hour: int = Field(
        default=1000, ge=100, le=10000, description="API rate limit per hour"
    )
    api_rate_limit_burst: int = Field(default=50, ge=10, le=200, description="API burst limit")

    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins",
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_allow_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed CORS methods",
    )

    # Session Security
    session_cookie_secure: bool = Field(default=True, description="Secure session cookies")
    session_cookie_httponly: bool = Field(default=True, description="HTTP-only session cookies")
    session_cookie_samesite: str = Field(default="strict", description="SameSite cookie policy")


class GoogleCloudConfig(BaseModel):
    """Google Cloud Platform configuration."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_id: str = Field(description="GCP Project ID")
    location: str = Field(default="us-central1", description="GCP region")
    use_vertex_ai: bool = Field(default=True, description="Use Vertex AI instead of Google AI")

    # Credentials (path only - never store actual credentials)
    credentials_path: str | None = Field(default=None, description="Path to service account key")

    # API Settings
    timeout_seconds: int = Field(default=60, ge=10, le=300, description="API timeout in seconds")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum API retries")

    @model_validator(mode="after")
    def validate_credentials_path(self) -> "GoogleCloudConfig":
        """Validate that credentials file exists if specified."""
        if self.credentials_path and not Path(self.credentials_path).exists():
            logger.warning(f"Credentials file not found: {self.credentials_path}")
        return self


class AIModelsConfig(BaseModel):
    """AI model configuration for different agents."""

    model_config = ConfigDict(str_strip_whitespace=True)

    # Gemini Models
    critic_model: str = Field(
        default="gemini-2.5-pro", description="Model for code review and criticism"
    )
    worker_model: str = Field(default="gemini-2.5-flash", description="Model for general tasks")
    master_model: str = Field(
        default="gemini-2.5-pro", description="Model for architecture and planning"
    )

    # Model Parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=40, ge=1, le=100, description="Top-k sampling")
    max_output_tokens: int = Field(default=8192, ge=1024, le=32768, description="Max output tokens")

    @model_validator(mode="after")
    def validate_model_names(self) -> "AIModelsConfig":
        """Validate model names follow expected patterns."""
        valid_patterns = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-exp-"]

        for model_field in ["critic_model", "worker_model", "master_model"]:
            model_name = getattr(self, model_field)
            if not any(pattern in model_name for pattern in valid_patterns):
                logger.warning(f"Model name may be invalid: {model_name}")

        return self


class MCPConfig(BaseModel):
    """Model Context Protocol server configuration."""

    model_config = ConfigDict(str_strip_whitespace=True)

    # Server Settings
    host: str = Field(default="0.0.0.0", description="MCP server host")
    port: int = Field(default=3000, ge=1024, le=65535, description="MCP server port")
    timeout: int = Field(
        default=30000, ge=5000, le=120000, description="MCP timeout in milliseconds"
    )
    max_connections: int = Field(
        default=100, ge=10, le=1000, description="Max concurrent connections"
    )

    # Individual MCP server ports
    code_reviewer_port: int = Field(default=3001, ge=1024, le=65535)
    workspace_analyzer_port: int = Field(default=3002, ge=1024, le=65535)
    documentation_port: int = Field(default=3003, ge=1024, le=65535)
    master_architect_port: int = Field(default=3004, ge=1024, le=65535)
    code_generator_port: int = Field(default=3005, ge=1024, le=65535)

    # Server Configuration
    servers: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "gemini-workspace-analyzer": {
                "command": "uv",
                "args": ["run", "python", "-m", "app.mcp_servers.gemini_workspace_analyzer"],
                "env": {"PYTHONPATH": "${workspaceFolder}", "LOG_LEVEL": "INFO"},
            },
            "secure-code-reviewer": {
                "command": "uv",
                "args": ["run", "python", "-m", "app.mcp_servers.security.secure_code_reviewer"],
                "env": {"PYTHONPATH": "${workspaceFolder}", "LOG_LEVEL": "INFO"},
            },
        },
    )


class PerformanceConfig(BaseModel):
    """Performance optimization configuration."""

    model_config = ConfigDict(str_strip_whitespace=True)

    # Cache Settings
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_default_timeout: int = Field(
        default=300, ge=60, le=3600, description="Default cache timeout"
    )
    cache_long_timeout: int = Field(
        default=3600, ge=300, le=86400, description="Long cache timeout"
    )
    memory_cache_max_size: int = Field(
        default=1000, ge=100, le=10000, description="Memory cache max items"
    )

    # Connection Pooling
    http_timeout: int = Field(default=30, ge=5, le=300, description="HTTP timeout in seconds")
    http_retries: int = Field(default=3, ge=1, le=10, description="HTTP retry attempts")
    http_backoff_factor: float = Field(
        default=0.3, ge=0.1, le=2.0, description="HTTP backoff factor"
    )
    max_connections: int = Field(default=100, ge=10, le=500, description="Max HTTP connections")
    connections_per_host: int = Field(default=30, ge=5, le=100, description="Connections per host")

    # Background Tasks
    celery_broker_url: str = Field(
        default="redis://localhost:6379/2", description="Celery broker URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/3", description="Celery result backend"
    )
    celery_task_serializer: str = Field(default="json", description="Celery task serializer")

    # Batch Processing
    enable_batch_processing: bool = Field(default=True, description="Enable batch processing")
    batch_size: int = Field(default=10, ge=1, le=100, description="Default batch size")
    max_batch_delay: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Max batch delay in seconds"
    )

    # Context Optimization
    max_context_size: int = Field(
        default=50000, ge=10000, le=200000, description="Max context size in tokens"
    )
    summarization_threshold: int = Field(
        default=40000,
        ge=5000,
        le=150000,
        description="Context summarization threshold",
    )


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""

    model_config = ConfigDict(str_strip_whitespace=True)

    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, ge=1024, le=65535, description="Metrics server port")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")

    # Health Checks
    health_check_timeout: int = Field(default=5, ge=1, le=30, description="Health check timeout")
    health_check_interval: int = Field(
        default=30, ge=10, le=300, description="Health check interval"
    )

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Application log level")
    log_format: str = Field(default="json", description="Log format (json|pretty)")
    log_file: str | None = Field(default=None, description="Log file path")

    # OpenTelemetry
    otel_endpoint: str | None = Field(default=None, description="OTEL exporter endpoint")
    otel_service_name: str = Field(default="my-fullstack-agent", description="OTEL service name")

    # Performance Budgets
    response_time_p95_target: int = Field(default=2000, description="P95 response time target (ms)")
    response_time_p95_threshold: int = Field(
        default=5000, description="P95 response time threshold (ms)"
    )
    cache_hit_rate_target: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Cache hit rate target"
    )
    error_rate_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Error rate threshold"
    )


class ServerConfig(BaseModel):
    """Web server configuration."""

    model_config = ConfigDict(str_strip_whitespace=True)

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, ge=1024, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=16, description="Number of worker processes")
    worker_class: str = Field(default="uvicorn.workers.UvicornWorker", description="Worker class")
    timeout_keep_alive: int = Field(default=2, ge=1, le=30, description="Keep-alive timeout")
    timeout_graceful_shutdown: int = Field(
        default=10, ge=5, le=60, description="Graceful shutdown timeout"
    )

    # Feature Flags
    enable_debug_endpoints: bool = Field(default=False, description="Enable debug endpoints")
    enable_admin_endpoints: bool = Field(default=False, description="Enable admin endpoints")
    enable_profiling: bool = Field(default=False, description="Enable profiling")


class TestingConfig(BaseModel):
    """Testing configuration with pytest settings."""

    model_config = ConfigDict(str_strip_whitespace=True)

    # Pytest Configuration
    test_paths: list[str] = Field(default=["tests"], description="Test discovery paths")
    python_files: str = Field(default="test_*.py *_test.py", description="Test file patterns")
    python_classes: str = Field(default="Test*", description="Test class patterns")
    python_functions: str = Field(default="test_*", description="Test function patterns")

    # Coverage Settings
    coverage_fail_under: int = Field(
        default=85, ge=50, le=100, description="Minimum coverage percentage"
    )
    coverage_source: list[str] = Field(default=["app"], description="Coverage source directories")
    coverage_omit: list[str] = Field(
        default=["*/tests/*", "*/test_*", "*/__pycache__/*", "*/migrations/*"],
        description="Coverage omit patterns",
    )

    # Test Execution
    asyncio_mode: str = Field(default="auto", description="Asyncio test mode")
    timeout: int = Field(default=300, ge=30, le=1800, description="Test timeout in seconds")
    maxfail: int = Field(default=10, ge=1, le=100, description="Maximum test failures")

    # Markers
    markers: dict[str, str] = Field(
        default_factory=lambda: {
            "unit": "Unit tests for individual components",
            "integration": "Integration tests between components",
            "performance": "Performance benchmarks and load tests",
            "slow": "Slow-running tests",
            "mcp": "MCP protocol and server tests",
            "security": "Security-focused tests",
            "memory": "Memory usage and efficiency tests",
        },
    )


class CentralizedConfig(BaseSettings):
    """Main centralized configuration class combining all configuration aspects."""

    model_config = SettingsConfigDict(
        env_file=[".env", ".env.local"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    app_name: str = Field(default="my-fullstack-agent", description="Application name")
    app_version: str = Field(default="0.3.0", description="Application version")

    # Configuration Sections
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig, description="Database configuration"
    )
    redis: RedisConfig = Field(default_factory=RedisConfig, description="Redis configuration")
    security: SecurityConfig = Field(
        default_factory=SecurityConfig, description="Security configuration"
    )
    google_cloud: GoogleCloudConfig = Field(description="Google Cloud configuration")
    ai_models: AIModelsConfig = Field(
        default_factory=AIModelsConfig, description="AI models configuration"
    )
    mcp: MCPConfig = Field(default_factory=MCPConfig, description="MCP server configuration")
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="Performance configuration"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring configuration"
    )
    server: ServerConfig = Field(default_factory=ServerConfig, description="Server configuration")
    testing: TestingConfig = Field(
        default_factory=TestingConfig, description="Testing configuration"
    )

    @model_validator(mode="after")
    def apply_environment_overrides(self) -> "CentralizedConfig":
        """Apply environment-specific configuration overrides."""
        if self.environment == Environment.DEVELOPMENT:
            self.debug = True
            self.monitoring.log_level = LogLevel.DEBUG
            self.server.workers = 1
            self.database.pool_size = self.database.pool_size_dev
            self.redis.max_connections = 5
            self.security.cors_origins.extend(["http://localhost:3000", "http://localhost:8080"])

        elif self.environment == Environment.TESTING:
            self.debug = False
            self.monitoring.log_level = LogLevel.WARNING
            self.database.pool_size = self.database.pool_size_test
            self.redis.max_connections = 2
            self.performance.enable_caching = False

        elif self.environment == Environment.STAGING:
            self.debug = True
            self.monitoring.log_level = LogLevel.DEBUG
            self.server.workers = 2
            self.database.pool_size = max(self.database.pool_size_dev, 5)
            self.server.enable_debug_endpoints = True
            self.server.enable_profiling = True

        elif self.environment == Environment.PRODUCTION:
            self.debug = False
            self.monitoring.log_level = LogLevel.INFO
            self.server.workers = min(self.server.workers * 2, 8)
            self.database.pool_size = self.database.pool_size_prod
            self.redis.max_connections = 20
            self.security.session_cookie_secure = True
            self.server.enable_debug_endpoints = False
            self.server.enable_admin_endpoints = False
            self.server.enable_profiling = False

        return self

    def to_pytest_ini(self) -> str:
        """Generate pytest.ini content from configuration."""
        markers_lines: list[Any] = []
        for marker, description in self.testing.markers.items():
            markers_lines.append(f"    {marker}: {description}")

        return f"""[tool:pytest]
minversion = 8.0
testpaths = {" ".join(self.testing.test_paths)}
python_files = {self.testing.python_files}
python_classes = {self.testing.python_classes}
python_functions = {self.testing.python_functions}

addopts =
    -ra
    --strict-markers
    --strict-config
    --cov={" --cov=".join(self.testing.coverage_source)}
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml:coverage.xml
    --cov-fail-under={self.testing.coverage_fail_under}
    --tb=short
    --maxfail={self.testing.maxfail}
    --timeout={self.testing.timeout}

asyncio_mode = {self.testing.asyncio_mode}

markers =
{chr(10).join(markers_lines)}

filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
    ignore::RuntimeWarning:google.*
    ignore::UserWarning:google.*

# Environment variables for testing
env =
    D:ENVIRONMENT={Environment.TESTING.value}
    D:DEBUG=false
    D:GOOGLE_CLOUD_PROJECT=test-project-id
    D:GOOGLE_GENAI_USE_VERTEXAI=false
    D:PYTEST_CURRENT_TEST=true
    D:REDIS_URL=redis://localhost:6379/1
"""

    def to_mcp_json(self) -> dict[str, Any]:
        """Generate .mcp.json content from configuration."""
        return {
            "mcpServers": {
                name: {
                    **server_config,
                    "env": {
                        **server_config.get("env", {}),
                        "GOOGLE_CLOUD_PROJECT": self.google_cloud.project_id,
                        "LOG_LEVEL": self.monitoring.log_level.value,
                    },
                }
                for name, server_config in self.mcp.servers.items()
            },
        }

    def to_docker_env(self) -> list[str]:
        """Generate Docker environment variables."""
        return [
            f"ENVIRONMENT={self.environment.value}",
            f"DEBUG={str(self.debug).lower()}",
            f"APP_NAME={self.app_name}",
            f"APP_VERSION={self.app_version}",
            f"LOG_LEVEL={self.monitoring.log_level.value}",
            f"UVICORN_HOST={self.server.host}",
            f"UVICORN_PORT={self.server.port}",
            f"UVICORN_WORKERS={self.server.workers}",
            f"DATABASE_URL={self.database.url}",
            f"REDIS_URL={self.redis.url}",
            f"GOOGLE_CLOUD_PROJECT={self.google_cloud.project_id}",
            f"GOOGLE_CLOUD_LOCATION={self.google_cloud.location}",
            f"MCP_SERVER_PORT={self.mcp.port}",
            f"ENABLE_METRICS={str(self.monitoring.enable_metrics).lower()}",
            f"METRICS_PORT={self.monitoring.metrics_port}",
            f"CACHE_TTL={self.redis.default_ttl}",
            f"HTTP_TIMEOUT={self.performance.http_timeout}",
            f"MAX_CONTEXT_SIZE={self.performance.max_context_size}",
        ]

    def validate_configuration(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues: list[Any] = []

        try:
            # Validate Google Cloud settings
            if not self.google_cloud.project_id:
                issues.append("Google Cloud project ID is required")

            # Validate port conflicts
            used_ports = {
                self.server.port,
                self.mcp.port,
                self.monitoring.metrics_port,
                self.mcp.code_reviewer_port,
                self.mcp.workspace_analyzer_port,
                self.mcp.documentation_port,
                self.mcp.master_architect_port,
                self.mcp.code_generator_port,
            }

            if len(used_ports) != 8:  # Should be 8 unique ports
                issues.append("Port conflicts detected in configuration")

            # Validate database URL format
            if not self.database.url.startswith(("postgresql://", "sqlite:///")):
                issues.append("Invalid database URL format")

            # Validate Redis URL format
            if not self.redis.url.startswith("redis://"):
                issues.append("Invalid Redis URL format")

            # Validate JWT secret in production
            if self.environment == Environment.PRODUCTION:
                if len(self.security.jwt_secret_key.get_secret_value()) < 32:
                    issues.append("JWT secret key too short for production")

            # Validate model names
            valid_patterns = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
            for model in [
                self.ai_models.critic_model,
                self.ai_models.worker_model,
                self.ai_models.master_model,
            ]:
                if not any(pattern in model for pattern in valid_patterns):
                    issues.append(f"Invalid model name: {model}")

        except Exception as e:
            issues.append(f"Configuration validation error: {e}")

        return issues


# Configuration loading and caching
_config_cache: CentralizedConfig | None = None
_config_file_path: Path | None = None


def load_config_from_yaml(config_path: str | Path) -> CentralizedConfig:
    """Load configuration from YAML file with environment variable interpolation."""
    config_path = Path(config_path)

    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    # Environment variable interpolation
    def interpolate_env_vars(obj) -> None:
        if isinstance(obj, dict):
            return {k: interpolate_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [interpolate_env_vars(item) for item in obj]
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.environ.get(env_var, obj)
        return obj

    interpolated_data = interpolate_env_vars(yaml_data)

    return CentralizedConfig(**interpolated_data)


def get_config(
    config_file: str | Path | None = None, force_reload: bool = False
) -> CentralizedConfig:
    """Get centralized configuration instance with caching."""
    global _config_cache, _config_file_path

    if config_file:
        config_file = Path(config_file)

    # Use cached config if available and no reload requested
    if not force_reload and _config_cache and (not config_file or config_file == _config_file_path):
        return _config_cache

    try:
        if config_file and config_file.exists():
            # Load from YAML file
            config = load_config_from_yaml(config_file)
            _config_file_path = config_file
        else:
            # Load from environment variables
            config = CentralizedConfig()
            _config_file_path = None

        # Validate configuration
        issues = config.validate_configuration()
        if issues:
            logger.warning(f"Configuration issues found: {issues}")

        # Cache the configuration
        _config_cache = config
        logger.info(f"Configuration loaded for environment: {config.environment.value}")

        return config

    except Exception as e:
        logger.exception(f"Failed to load configuration: {e}")
        # Return basic configuration as fallback
        return CentralizedConfig()


def reload_config(config_file: str | Path | None = None) -> CentralizedConfig:
    """Reload configuration from source."""
    return get_config(config_file=config_file, force_reload=True)


def validate_config(config: CentralizedConfig | None = None) -> list[str]:
    """Validate configuration and return list of issues."""
    if config is None:
        config = get_config()

    return config.validate_configuration()


# Export commonly used configurations for backward compatibility
def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config().database


def get_redis_config() -> RedisConfig:
    """Get Redis configuration."""
    return get_config().redis


def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return get_config().security


def get_ai_models_config() -> AIModelsConfig:
    """Get AI models configuration."""
    return get_config().ai_models


def get_performance_config() -> PerformanceConfig:
    """Get performance configuration."""
    return get_config().performance
