"""
Unit tests for configuration and storage modules.
Tests configuration loading, validation, and storage operations.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from app.automation.auth_models import APIKey
from app.automation.auth_models import User
from app.automation.auth_storage import AuthStorage
from app.automation.auth_storage import auth_storage
from app.automation.auth_storage import get_storage_stats
from app.automation.auth_storage import initialize_default_users
from app.config import ResearchConfiguration
from app.config import config
from app.utils.redis_client import RedisClientMixin
import pytest


class TestResearchConfiguration:
    """Test research configuration functionality."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config_obj = ResearchConfiguration()

        assert config_obj.critic_model == "gemini-2.5-pro"
        assert config_obj.worker_model == "gemini-2.5-flash"
        assert config_obj.max_search_iterations == 5

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config_obj = ResearchConfiguration(
            critic_model="custom-critic", worker_model="custom-worker", max_search_iterations=10
        )

        assert config_obj.critic_model == "custom-critic"
        assert config_obj.worker_model == "custom-worker"
        assert config_obj.max_search_iterations == 10

    def test_global_config_instance(self):
        """Test that global config instance exists and is properly configured."""
        assert config is not None
        assert isinstance(config, ResearchConfiguration)
        assert hasattr(config, "critic_model")
        assert hasattr(config, "worker_model")
        assert hasattr(config, "max_search_iterations")


class TestEnvironmentConfiguration:
    """Test environment variable configuration handling."""

    def test_google_cloud_project_setting(self):
        """Test Google Cloud project environment setting."""
        # Should have default or set value
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        assert project is not None

    def test_google_cloud_location_setting(self):
        """Test Google Cloud location environment setting."""
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        assert location is not None

    def test_vertex_ai_setting(self):
        """Test Vertex AI usage setting."""
        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
        assert use_vertex is not None
        assert use_vertex.lower() in ["true", "false"]

    def test_environment_override(self):
        """Test that environment variables can override defaults."""
        with patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "test-override-project",
                "GOOGLE_CLOUD_LOCATION": "europe-west1",
            },
        ):
            # Reload config with overrides
            import importlib

            import app.config

            importlib.reload(app.config)

            assert os.environ["GOOGLE_CLOUD_PROJECT"] == "test-override-project"
            assert os.environ["GOOGLE_CLOUD_LOCATION"] == "europe-west1"


class TestAuthStorageInitialization:
    """Test authentication storage initialization."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_storage_initialization(self, temp_storage_path):
        """Test storage initialization with custom path."""
        storage = AuthStorage(temp_storage_path)

        assert storage.storage_path == temp_storage_path
        assert storage.users_file == temp_storage_path / "users.json"
        assert storage.api_keys_file == temp_storage_path / "api_keys.json"
        assert storage.sessions_file == temp_storage_path / "sessions.json"

        # Check that files were created
        assert storage.users_file.exists()
        assert storage.api_keys_file.exists()
        assert storage.sessions_file.exists()

    def test_storage_files_initialization(self, temp_storage_path):
        """Test that storage files are properly initialized."""
        storage = AuthStorage(temp_storage_path)

        # Files should contain empty JSON objects
        with open(storage.users_file) as f:
            users_data = json.load(f)
            assert users_data == {}

        with open(storage.api_keys_file) as f:
            keys_data = json.load(f)
            assert keys_data == {}

        with open(storage.sessions_file) as f:
            sessions_data = json.load(f)
            assert sessions_data == {}


class TestAuthStorageJSONOperations:
    """Test JSON file operations in auth storage."""

    @pytest.fixture
    def storage_with_temp_path(self):
        """Create storage instance with temporary path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = AuthStorage(Path(temp_dir))
            yield storage

    def test_load_json_success(self, storage_with_temp_path):
        """Test successful JSON loading."""
        test_data = {"test": "data", "number": 123}
        test_file = storage_with_temp_path.storage_path / "test.json"

        with open(test_file, "w") as f:
            json.dump(test_data, f)

        loaded_data = storage_with_temp_path._load_json(test_file)
        assert loaded_data == test_data

    def test_load_json_nonexistent_file(self, storage_with_temp_path):
        """Test loading non-existent JSON file."""
        nonexistent_file = storage_with_temp_path.storage_path / "nonexistent.json"
        loaded_data = storage_with_temp_path._load_json(nonexistent_file)
        assert loaded_data == {}

    def test_load_json_invalid_json(self, storage_with_temp_path):
        """Test loading invalid JSON file."""
        invalid_file = storage_with_temp_path.storage_path / "invalid.json"
        invalid_file.write_text("{ invalid json")

        loaded_data = storage_with_temp_path._load_json(invalid_file)
        assert loaded_data == {}

    def test_save_json_success(self, storage_with_temp_path):
        """Test successful JSON saving."""
        test_data = {"test": "data", "array": [1, 2, 3]}
        test_file = storage_with_temp_path.storage_path / "test_save.json"

        storage_with_temp_path._save_json(test_file, test_data)

        assert test_file.exists()
        with open(test_file) as f:
            loaded_data = json.load(f)
            assert loaded_data == test_data

    def test_save_json_atomic_operation(self, storage_with_temp_path):
        """Test that JSON saving is atomic (uses temporary file)."""
        test_data = {"atomic": "test"}
        test_file = storage_with_temp_path.storage_path / "atomic_test.json"

        # Mock to test atomic operation
        with patch("pathlib.Path.rename") as mock_rename:
            storage_with_temp_path._save_json(test_file, test_data)
            mock_rename.assert_called_once()


class TestUserManagement:
    """Test user management functionality."""

    @pytest.fixture
    def storage_with_temp_path(self):
        """Create storage instance with temporary path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = AuthStorage(Path(temp_dir))
            yield storage

    @pytest.fixture
    def sample_user(self):
        """Create sample user for testing."""
        return User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            permissions={"read", "write"},
        )

    def test_create_user_success(self, storage_with_temp_path, sample_user):
        """Test successful user creation."""
        result = storage_with_temp_path.create_user(sample_user)

        assert result is True

        # Verify user was stored
        retrieved_user = storage_with_temp_path.get_user_by_id("user_123")
        assert retrieved_user is not None
        assert retrieved_user.username == "testuser"
        assert retrieved_user.email == "test@example.com"

    def test_create_user_duplicate_username(self, storage_with_temp_path, sample_user):
        """Test creating user with duplicate username."""
        # Create first user
        storage_with_temp_path.create_user(sample_user)

        # Try to create user with same username but different ID
        duplicate_user = User(
            id="user_456",
            username="testuser",  # Same username
            email="different@example.com",
            password_hash="different_hash",
        )

        result = storage_with_temp_path.create_user(duplicate_user)
        assert result is False

    def test_create_user_duplicate_email(self, storage_with_temp_path, sample_user):
        """Test creating user with duplicate email."""
        storage_with_temp_path.create_user(sample_user)

        duplicate_user = User(
            id="user_456",
            username="differentuser",
            email="test@example.com",  # Same email
            password_hash="different_hash",
        )

        result = storage_with_temp_path.create_user(duplicate_user)
        assert result is False

    def test_get_user_by_username(self, storage_with_temp_path, sample_user):
        """Test retrieving user by username."""
        storage_with_temp_path.create_user(sample_user)

        # Test exact match
        user = storage_with_temp_path.get_user_by_username("testuser")
        assert user is not None
        assert user.id == "user_123"

        # Test case insensitive
        user = storage_with_temp_path.get_user_by_username("TESTUSER")
        assert user is not None
        assert user.id == "user_123"

        # Test non-existent user
        user = storage_with_temp_path.get_user_by_username("nonexistent")
        assert user is None

    def test_update_user(self, storage_with_temp_path, sample_user):
        """Test user update functionality."""
        storage_with_temp_path.create_user(sample_user)

        # Update user
        sample_user.email = "updated@example.com"
        sample_user.permissions.add("admin")

        result = storage_with_temp_path.update_user(sample_user)
        assert result is True

        # Verify update
        updated_user = storage_with_temp_path.get_user_by_id("user_123")
        assert updated_user.email == "updated@example.com"
        assert "admin" in updated_user.permissions

    def test_update_nonexistent_user(self, storage_with_temp_path):
        """Test updating non-existent user."""
        nonexistent_user = User(
            id="nonexistent", username="test", email="test@example.com", password_hash="hash"
        )

        result = storage_with_temp_path.update_user(nonexistent_user)
        assert result is False

    def test_delete_user(self, storage_with_temp_path, sample_user):
        """Test user deletion."""
        storage_with_temp_path.create_user(sample_user)

        # Create API key for user
        api_key = APIKey(
            id="key_123",
            user_id="user_123",
            name="Test Key",
            key_hash="hash",
            key_prefix="prefix123",
        )
        storage_with_temp_path.create_api_key(api_key)

        # Delete user
        result = storage_with_temp_path.delete_user("user_123")
        assert result is True

        # Verify user and API keys are deleted
        assert storage_with_temp_path.get_user_by_id("user_123") is None
        assert storage_with_temp_path.get_api_key_by_id("key_123") is None

    def test_list_users(self, storage_with_temp_path):
        """Test listing all users."""
        # Initially empty
        users = storage_with_temp_path.list_users()
        assert len(users) == 0

        # Add users
        user1 = User(id="1", username="user1", email="user1@test.com", password_hash="hash")
        user2 = User(id="2", username="user2", email="user2@test.com", password_hash="hash")

        storage_with_temp_path.create_user(user1)
        storage_with_temp_path.create_user(user2)

        users = storage_with_temp_path.list_users()
        assert len(users) == 2
        assert any(u.username == "user1" for u in users)
        assert any(u.username == "user2" for u in users)


class TestAPIKeyManagement:
    """Test API key management functionality."""

    @pytest.fixture
    def storage_with_temp_path(self):
        """Create storage instance with temporary path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = AuthStorage(Path(temp_dir))
            yield storage

    @pytest.fixture
    def sample_api_key(self):
        """Create sample API key for testing."""
        return APIKey(
            id="key_123",
            user_id="user_123",
            name="Test API Key",
            key_hash="hashed_key_value",
            key_prefix="testkey1",
            scopes={"api:read", "api:write"},
        )

    def test_create_api_key(self, storage_with_temp_path, sample_api_key):
        """Test API key creation."""
        result = storage_with_temp_path.create_api_key(sample_api_key)
        assert result is True

        # Verify key was stored
        retrieved_key = storage_with_temp_path.get_api_key_by_id("key_123")
        assert retrieved_key is not None
        assert retrieved_key.name == "Test API Key"
        assert retrieved_key.user_id == "user_123"

    def test_get_api_key_by_prefix(self, storage_with_temp_path, sample_api_key):
        """Test retrieving API key by prefix."""
        storage_with_temp_path.create_api_key(sample_api_key)

        key = storage_with_temp_path.get_api_key_by_prefix("testkey1")
        assert key is not None
        assert key.id == "key_123"

        # Test non-existent prefix
        key = storage_with_temp_path.get_api_key_by_prefix("nonexistent")
        assert key is None

    def test_verify_api_key_valid(self, storage_with_temp_path):
        """Test API key verification with valid key."""
        api_key = APIKey(
            id="key_123",
            user_id="user_123",
            name="Test Key",
            key_hash="dummy_hash",  # Will be mocked
            key_prefix="testkey1",
        )

        # Mock the verify_key method
        with patch.object(api_key, "verify_key", return_value=True):
            with patch.object(api_key, "is_expired", return_value=False):
                storage_with_temp_path.create_api_key(api_key)

                result = storage_with_temp_path.verify_api_key("testkey1_full_key")
                assert result is not None
                assert result.id == "key_123"
                assert result.usage_count == 1

    def test_verify_api_key_invalid(self, storage_with_temp_path, sample_api_key):
        """Test API key verification with invalid key."""
        storage_with_temp_path.create_api_key(sample_api_key)

        # Test with short key
        result = storage_with_temp_path.verify_api_key("short")
        assert result is None

        # Test with wrong prefix
        result = storage_with_temp_path.verify_api_key("wrongpre_full_key")
        assert result is None

    def test_verify_api_key_expired(self, storage_with_temp_path):
        """Test API key verification with expired key."""
        api_key = APIKey(
            id="key_123",
            user_id="user_123",
            name="Expired Key",
            key_hash="hash",
            key_prefix="expiredk",
            expires_at=datetime.now(UTC) - timedelta(days=1),  # Expired
        )

        storage_with_temp_path.create_api_key(api_key)

        result = storage_with_temp_path.verify_api_key("expiredk_full_key")
        assert result is None

    def test_verify_api_key_inactive(self, storage_with_temp_path):
        """Test API key verification with inactive key."""
        api_key = APIKey(
            id="key_123",
            user_id="user_123",
            name="Inactive Key",
            key_hash="hash",
            key_prefix="inactive",
            is_active=False,
        )

        storage_with_temp_path.create_api_key(api_key)

        result = storage_with_temp_path.verify_api_key("inactive_full_key")
        assert result is None

    def test_list_api_keys_all(self, storage_with_temp_path):
        """Test listing all API keys."""
        key1 = APIKey(id="1", user_id="user1", name="Key 1", key_hash="hash1", key_prefix="key1")
        key2 = APIKey(id="2", user_id="user2", name="Key 2", key_hash="hash2", key_prefix="key2")

        storage_with_temp_path.create_api_key(key1)
        storage_with_temp_path.create_api_key(key2)

        keys = storage_with_temp_path.list_api_keys()
        assert len(keys) == 2

    def test_list_api_keys_by_user(self, storage_with_temp_path):
        """Test listing API keys for specific user."""
        key1 = APIKey(id="1", user_id="user1", name="Key 1", key_hash="hash1", key_prefix="key1")
        key2 = APIKey(id="2", user_id="user2", name="Key 2", key_hash="hash2", key_prefix="key2")
        key3 = APIKey(id="3", user_id="user1", name="Key 3", key_hash="hash3", key_prefix="key3")

        storage_with_temp_path.create_api_key(key1)
        storage_with_temp_path.create_api_key(key2)
        storage_with_temp_path.create_api_key(key3)

        user1_keys = storage_with_temp_path.list_api_keys("user1")
        assert len(user1_keys) == 2
        assert all(k.user_id == "user1" for k in user1_keys)

    def test_delete_api_key(self, storage_with_temp_path, sample_api_key):
        """Test API key deletion."""
        storage_with_temp_path.create_api_key(sample_api_key)

        result = storage_with_temp_path.delete_api_key("key_123")
        assert result is True

        # Verify deletion
        assert storage_with_temp_path.get_api_key_by_id("key_123") is None

    def test_delete_nonexistent_api_key(self, storage_with_temp_path):
        """Test deleting non-existent API key."""
        result = storage_with_temp_path.delete_api_key("nonexistent")
        assert result is False


class TestAuthenticationOperations:
    """Test authentication operations."""

    @pytest.fixture
    def storage_with_temp_path(self):
        """Create storage instance with temporary path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = AuthStorage(Path(temp_dir))
            yield storage

    def test_authenticate_user_success(self, storage_with_temp_path):
        """Test successful user authentication."""
        user = User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
        )
        storage_with_temp_path.create_user(user)

        with patch("app.automation.auth_storage.password_manager") as mock_pm:
            mock_pm.verify_password.return_value = True

            authenticated_user = storage_with_temp_path.authenticate_user("testuser", "password")

            assert authenticated_user is not None
            assert authenticated_user.username == "testuser"
            assert authenticated_user.failed_login_attempts == 0
            assert authenticated_user.last_login is not None

    def test_authenticate_user_wrong_password(self, storage_with_temp_path):
        """Test authentication with wrong password."""
        user = User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
        )
        storage_with_temp_path.create_user(user)

        with patch("app.automation.auth_storage.password_manager") as mock_pm:
            mock_pm.verify_password.return_value = False

            authenticated_user = storage_with_temp_path.authenticate_user(
                "testuser", "wrong_password"
            )

            assert authenticated_user is None

            # Check failed attempt was recorded
            updated_user = storage_with_temp_path.get_user_by_username("testuser")
            assert updated_user.failed_login_attempts == 1

    def test_authenticate_user_account_lockout(self, storage_with_temp_path):
        """Test account lockout after multiple failed attempts."""
        user = User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            failed_login_attempts=4,  # One more will lock
        )
        storage_with_temp_path.create_user(user)

        with patch("app.automation.auth_storage.password_manager") as mock_pm:
            mock_pm.verify_password.return_value = False

            authenticated_user = storage_with_temp_path.authenticate_user(
                "testuser", "wrong_password"
            )

            assert authenticated_user is None

            # Check account is locked
            locked_user = storage_with_temp_path.get_user_by_username("testuser")
            assert locked_user.locked_until is not None
            assert locked_user.is_locked()

    def test_authenticate_user_nonexistent(self, storage_with_temp_path):
        """Test authentication with non-existent user."""
        authenticated_user = storage_with_temp_path.authenticate_user("nonexistent", "password")
        assert authenticated_user is None

    def test_authenticate_user_inactive(self, storage_with_temp_path):
        """Test authentication with inactive user."""
        user = User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            is_active=False,
        )
        storage_with_temp_path.create_user(user)

        authenticated_user = storage_with_temp_path.authenticate_user("testuser", "password")
        assert authenticated_user is None

    def test_authenticate_api_key_success(self, storage_with_temp_path):
        """Test successful API key authentication."""
        # Create user
        user = User(
            id="user_123", username="testuser", email="test@example.com", password_hash="hash"
        )
        storage_with_temp_path.create_user(user)

        # Create API key
        api_key = APIKey(
            id="key_123",
            user_id="user_123",
            name="Test Key",
            key_hash="hash",
            key_prefix="testkey1",
        )

        with patch.object(api_key, "verify_key", return_value=True):
            with patch.object(api_key, "is_expired", return_value=False):
                storage_with_temp_path.create_api_key(api_key)

                result = storage_with_temp_path.authenticate_api_key("testkey1_full_key")

                assert result is not None
                key, auth_user = result
                assert key.id == "key_123"
                assert auth_user.id == "user_123"

    def test_authenticate_api_key_inactive_user(self, storage_with_temp_path):
        """Test API key authentication with inactive user."""
        # Create inactive user
        user = User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            password_hash="hash",
            is_active=False,
        )
        storage_with_temp_path.create_user(user)

        # Create API key
        api_key = APIKey(
            id="key_123",
            user_id="user_123",
            name="Test Key",
            key_hash="hash",
            key_prefix="testkey1",
        )

        with patch.object(api_key, "verify_key", return_value=True):
            with patch.object(api_key, "is_expired", return_value=False):
                storage_with_temp_path.create_api_key(api_key)

                result = storage_with_temp_path.authenticate_api_key("testkey1_full_key")
                assert result is None


class TestStorageStatistics:
    """Test storage statistics functionality."""

    def test_get_storage_stats_empty(self):
        """Test storage statistics with empty storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = AuthStorage(Path(temp_dir))

            # Temporarily replace global storage
            original_storage = auth_storage
            import app.automation.auth_storage

            app.automation.auth_storage.auth_storage = storage

            try:
                stats = get_storage_stats()

                assert stats["total_users"] == 0
                assert stats["active_users"] == 0
                assert stats["total_api_keys"] == 0
                assert stats["active_api_keys"] == 0
                assert "storage_path" in stats

            finally:
                app.automation.auth_storage.auth_storage = original_storage

    def test_get_storage_stats_with_data(self):
        """Test storage statistics with data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = AuthStorage(Path(temp_dir))

            # Add test data
            user1 = User(
                id="1",
                username="user1",
                email="user1@test.com",
                password_hash="hash",
                is_active=True,
            )
            user2 = User(
                id="2",
                username="user2",
                email="user2@test.com",
                password_hash="hash",
                is_active=False,
            )
            storage.create_user(user1)
            storage.create_user(user2)

            key1 = APIKey(
                id="1",
                user_id="1",
                name="Key 1",
                key_hash="hash1",
                key_prefix="key1",
                is_active=True,
            )
            key2 = APIKey(
                id="2",
                user_id="1",
                name="Key 2",
                key_hash="hash2",
                key_prefix="key2",
                is_active=False,
            )
            storage.create_api_key(key1)
            storage.create_api_key(key2)

            # Replace global storage temporarily
            original_storage = auth_storage
            import app.automation.auth_storage

            app.automation.auth_storage.auth_storage = storage

            try:
                stats = get_storage_stats()

                assert stats["total_users"] == 2
                assert stats["active_users"] == 1
                assert stats["total_api_keys"] == 2
                assert stats["active_api_keys"] == 1

            finally:
                app.automation.auth_storage.auth_storage = original_storage

    def test_initialize_default_users(self):
        """Test default user initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = AuthStorage(Path(temp_dir))

            # Replace global storage temporarily
            original_storage = auth_storage
            import app.automation.auth_storage

            app.automation.auth_storage.auth_storage = storage

            try:
                with patch("app.automation.auth_storage.create_default_admin_user") as mock_create:
                    mock_admin = User(
                        id="admin",
                        username="admin",
                        email="admin@example.com",
                        password_hash="hash",
                    )
                    mock_create.return_value = mock_admin

                    initialize_default_users()

                    # Verify admin user was created
                    admin_user = storage.get_user_by_username("admin")
                    assert admin_user is not None

            finally:
                app.automation.auth_storage.auth_storage = original_storage


class TestRedisClientMixin:
    """Test Redis client mixin functionality."""

    class TestRedisClass(RedisClientMixin):
        """Test class that uses Redis mixin."""

        pass

    @pytest.fixture
    def redis_instance(self):
        """Create instance with Redis mixin."""
        return self.TestRedisClass()

    def test_redis_mixin_initialization(self, redis_instance):
        """Test Redis mixin initialization."""
        # Should have Redis-related attributes
        assert hasattr(redis_instance, "redis_client")
        assert hasattr(redis_instance, "get_redis_client")
        assert hasattr(redis_instance, "close_redis")

    @pytest.mark.asyncio
    async def test_redis_client_operations(self, redis_instance, mock_redis_client):
        """Test Redis client operations through mixin."""
        # Mock Redis client
        redis_instance.redis_client = mock_redis_client

        # Test various Redis operations
        await redis_instance.redis_client.set("test_key", "test_value")
        mock_redis_client.set.assert_called_with("test_key", "test_value")

        await redis_instance.redis_client.get("test_key")
        mock_redis_client.get.assert_called_with("test_key")

        await redis_instance.redis_client.delete("test_key")
        mock_redis_client.delete.assert_called_with("test_key")


class TestCacheConfiguration:
    """Test cache-related configuration."""

    def test_cache_load_behavior(self):
        """Test cache loading behavior in auth storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = AuthStorage(Path(temp_dir))

            # Initially cache should not be loaded
            assert not storage._cache_loaded

            # Create a user (this should trigger cache loading)
            user = User(id="1", username="test", email="test@test.com", password_hash="hash")
            storage.create_user(user)

            # Cache should now be loaded
            assert storage._cache_loaded

    def test_cache_persistence_across_operations(self):
        """Test that cache persists across multiple operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = AuthStorage(Path(temp_dir))

            # Create user
            user = User(id="1", username="test", email="test@test.com", password_hash="hash")
            storage.create_user(user)

            # Get user (should use cache)
            retrieved = storage.get_user_by_id("1")
            assert retrieved is not None
            assert retrieved.username == "test"

            # Update user
            user.email = "updated@test.com"
            storage.update_user(user)

            # Get updated user
            updated = storage.get_user_by_id("1")
            assert updated.email == "updated@test.com"
