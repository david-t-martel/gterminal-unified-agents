"""Secure secrets management system.

Provides secure storage and retrieval of sensitive configuration data
with support for environment variables, file-based secrets, and external
secret management services.
"""

import base64
import json
import logging
import os
from pathlib import Path
import secrets
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecretsManager:
    """Secure secrets management with encryption and validation."""

    def __init__(self, master_key: str | None = None) -> None:
        """Initialize secrets manager.

        Args:
            master_key: Optional master key for encryption. If not provided,
                       will attempt to load from environment or generate.

        """
        self.master_key = master_key or self._get_or_create_master_key()
        self.cipher_suite = self._create_cipher_suite()
        self.secrets_cache: dict[str, Any] = {}

    def _get_or_create_master_key(self) -> str:
        """Get or create master encryption key."""
        # Try environment variable first
        master_key = os.getenv("SECRETS_MASTER_KEY")
        if master_key:
            return master_key

        # Try secure file
        key_file = Path.home() / ".config" / "fullstack-agent" / ".master_key"
        if key_file.exists():
            # Check file permissions
            stat_info = key_file.stat()
            if stat_info.st_mode & 0o077:
                logger.warning(
                    f"Master key file has insecure permissions: {oct(stat_info.st_mode)}"
                )

            return key_file.read_text().strip()

        # Generate new key
        logger.info("Generating new master key for secrets encryption")
        new_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()

        # Save to secure file
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(new_key)
        key_file.chmod(0o600)  # Secure permissions

        logger.info(f"Master key saved to: {key_file}")
        return new_key

    def _create_cipher_suite(self) -> Fernet:
        """Create cipher suite for encryption/decryption."""
        # Derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"fullstack-agent-salt",  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return Fernet(key)

    def get_secret(self, key: str, default: str | None = None) -> str | None:
        """Get secret value with fallback hierarchy.

        Attempts to retrieve secrets in order:
        1. Environment variable
        2. Encrypted secrets file
        3. Plain text secrets file (with warning)
        4. Default value

        Args:
            key: Secret key name
            default: Default value if not found

        Returns:
            Secret value or default

        """
        # Check cache first
        if key in self.secrets_cache:
            return self.secrets_cache[key]

        # Try environment variable
        env_value = os.getenv(key)
        if env_value:
            self.secrets_cache[key] = env_value
            return env_value

        # Try encrypted secrets file
        encrypted_value = self._get_from_encrypted_file(key)
        if encrypted_value:
            self.secrets_cache[key] = encrypted_value
            return encrypted_value

        # Try plain text secrets file (with warning)
        plain_value = self._get_from_plain_file(key)
        if plain_value:
            logger.warning(f"Secret '{key}' loaded from plain text file - consider encrypting")
            self.secrets_cache[key] = plain_value
            return plain_value

        # Return default
        if default is not None:
            logger.info(f"Using default value for secret: {key}")

        return default

    def set_secret(self, key: str, value: str, encrypt: bool = True) -> None:
        """Store secret value securely.

        Args:
            key: Secret key name
            value: Secret value
            encrypt: Whether to encrypt the value

        """
        if encrypt:
            self._save_to_encrypted_file(key, value)
        else:
            self._save_to_plain_file(key, value)
            logger.warning(f"Secret '{key}' saved in plain text - consider encrypting")

        # Update cache
        self.secrets_cache[key] = value

    def _get_from_encrypted_file(self, key: str) -> str | None:
        """Get secret from encrypted file."""
        secrets_file = self._get_secrets_file_path("encrypted_secrets.json")
        if not secrets_file.exists():
            return None

        try:
            encrypted_data = secrets_file.read_bytes()
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            secrets_dict = json.loads(decrypted_data.decode())
            return secrets_dict.get(key)
        except Exception as e:
            logger.exception(f"Failed to decrypt secrets file: {e}")
            return None

    def _save_to_encrypted_file(self, key: str, value: str) -> None:
        """Save secret to encrypted file."""
        secrets_file = self._get_secrets_file_path("encrypted_secrets.json")

        # Load existing secrets
        secrets_dict: dict[str, Any] = {}
        if secrets_file.exists():
            try:
                encrypted_data = secrets_file.read_bytes()
                decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                secrets_dict = json.loads(decrypted_data.decode())
            except Exception as e:
                logger.warning(f"Could not load existing secrets: {e}")

        # Update secrets
        secrets_dict[key] = value

        # Encrypt and save
        json_data = json.dumps(secrets_dict, indent=2).encode()
        encrypted_data = self.cipher_suite.encrypt(json_data)

        secrets_file.parent.mkdir(parents=True, exist_ok=True)
        secrets_file.write_bytes(encrypted_data)
        secrets_file.chmod(0o600)  # Secure permissions

    def _get_from_plain_file(self, key: str) -> str | None:
        """Get secret from plain text file."""
        secrets_file = self._get_secrets_file_path("secrets.json")
        if not secrets_file.exists():
            return None

        try:
            secrets_dict = json.loads(secrets_file.read_text())
            return secrets_dict.get(key)
        except Exception as e:
            logger.exception(f"Failed to read plain secrets file: {e}")
            return None

    def _save_to_plain_file(self, key: str, value: str) -> None:
        """Save secret to plain text file."""
        secrets_file = self._get_secrets_file_path("secrets.json")

        # Load existing secrets
        secrets_dict: dict[str, Any] = {}
        if secrets_file.exists():
            try:
                secrets_dict = json.loads(secrets_file.read_text())
            except Exception as e:
                logger.warning(f"Could not load existing plain secrets: {e}")

        # Update secrets
        secrets_dict[key] = value

        # Save
        secrets_file.parent.mkdir(parents=True, exist_ok=True)
        secrets_file.write_text(json.dumps(secrets_dict, indent=2))
        secrets_file.chmod(0o600)  # Secure permissions

    def _get_secrets_file_path(self, filename: str) -> Path:
        """Get path to secrets file."""
        # Try project .secure directory first
        project_root = Path(__file__).parent.parent.parent
        secure_dir = project_root / ".secure"
        if secure_dir.exists():
            return secure_dir / filename

        # Fall back to user config directory
        return Path.home() / ".config" / "fullstack-agent" / filename

    def validate_required_secrets(self, required_keys: list[str]) -> bool:
        """Validate that all required secrets are available.

        Args:
            required_keys: List of required secret keys

        Returns:
            True if all required secrets are available

        """
        missing_keys: list[Any] = []
        for key in required_keys:
            if not self.get_secret(key):
                missing_keys.append(key)

        if missing_keys:
            logger.error(f"Missing required secrets: {missing_keys}")
            return False

        return True

    def generate_secure_secret(self, length: int = 32) -> str:
        """Generate a cryptographically secure secret.

        Args:
            length: Length of secret in bytes

        Returns:
            Base64 encoded secure secret

        """
        return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode()

    def clear_cache(self) -> None:
        """Clear secrets cache."""
        self.secrets_cache.clear()
        logger.info("Secrets cache cleared")


# Global secrets manager instance
secrets_manager = SecretsManager()
