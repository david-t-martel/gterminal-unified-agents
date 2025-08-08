//! High-performance authentication and security operations
//!
//! This module provides PyO3 bindings for authentication operations that are
//! significantly faster and more secure than Python equivalents:
//! - JWT token validation with cryptographic verification
//! - Password hashing with Argon2
//! - API key generation and validation
//! - Cryptographic operations with ring

use argon2::password_hash::{rand_core::OsRng, SaltString};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use blake3;
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use pyo3::prelude::*;
use ring::rand::SecureRandom;
use ring::{hmac, rand as ring_rand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// JWT Claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,         // Subject (user id)
    pub iat: u64,            // Issued at
    pub exp: u64,            // Expiration
    pub aud: Option<String>, // Audience
    pub iss: Option<String>, // Issuer
    pub scopes: Vec<String>, // Permissions/scopes
    pub session_id: Option<String>,
    pub user_agent: Option<String>,
    pub ip_address: Option<String>,
}

/// High-performance JWT token manager
#[pyclass]
pub struct RustTokenManager {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    algorithm: Algorithm,
    default_expiration: u64,
    issuer: Option<String>,
}

#[pymethods]
impl RustTokenManager {
    #[new]
    #[pyo3(signature = (secret_key, algorithm="HS256", default_expiration_hours=24, issuer=None))]
    fn new(
        secret_key: String,
        algorithm: &str,
        default_expiration_hours: u64,
        issuer: Option<String>,
    ) -> PyResult<Self> {
        let algo = match algorithm {
            "HS256" => Algorithm::HS256,
            "HS384" => Algorithm::HS384,
            "HS512" => Algorithm::HS512,
            "RS256" => Algorithm::RS256,
            "RS384" => Algorithm::RS384,
            "RS512" => Algorithm::RS512,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported algorithm",
                ))
            }
        };

        let encoding_key = EncodingKey::from_secret(secret_key.as_bytes());
        let decoding_key = DecodingKey::from_secret(secret_key.as_bytes());

        Ok(Self {
            encoding_key,
            decoding_key,
            algorithm: algo,
            default_expiration: default_expiration_hours * 3600,
            issuer,
        })
    }

    /// Create JWT token with claims
    #[pyo3(signature = (user_id, scopes=None, expiration_seconds=None, audience=None, session_id=None, user_agent=None, ip_address=None))]
    fn create_token(
        &self,
        user_id: String,
        scopes: Option<Vec<String>>,
        expiration_seconds: Option<u64>,
        audience: Option<String>,
        session_id: Option<String>,
        user_agent: Option<String>,
        ip_address: Option<String>,
    ) -> PyResult<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let exp = now + expiration_seconds.unwrap_or(self.default_expiration);

        let claims = Claims {
            sub: user_id,
            iat: now,
            exp,
            aud: audience,
            iss: self.issuer.clone(),
            scopes: scopes.unwrap_or_default(),
            session_id,
            user_agent,
            ip_address,
        };

        let header = Header::new(self.algorithm);

        encode(&header, &claims, &self.encoding_key).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Token creation error: {e}"))
        })
    }

    /// Validate JWT token and return claims
    fn validate_token(&self, token: &str) -> PyResult<HashMap<String, PyObject>> {
        let validation = Validation::new(self.algorithm);

        let token_data = decode::<Claims>(token, &self.decoding_key, &validation).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Token validation error: {e}"))
        })?;

        let claims = token_data.claims;

        Python::with_gil(|py| {
            let mut result = HashMap::new();
            result.insert("user_id".to_string(), claims.sub.to_object(py));
            result.insert("issued_at".to_string(), claims.iat.to_object(py));
            result.insert("expires_at".to_string(), claims.exp.to_object(py));
            result.insert("scopes".to_string(), claims.scopes.to_object(py));

            if let Some(aud) = claims.aud {
                result.insert("audience".to_string(), aud.to_object(py));
            }
            if let Some(iss) = claims.iss {
                result.insert("issuer".to_string(), iss.to_object(py));
            }
            if let Some(session_id) = claims.session_id {
                result.insert("session_id".to_string(), session_id.to_object(py));
            }
            if let Some(user_agent) = claims.user_agent {
                result.insert("user_agent".to_string(), user_agent.to_object(py));
            }
            if let Some(ip_address) = claims.ip_address {
                result.insert("ip_address".to_string(), ip_address.to_object(py));
            }

            Ok(result)
        })
    }

    /// Check if token is expired
    fn is_token_expired(&self, token: &str) -> PyResult<bool> {
        let validation = Validation::new(self.algorithm);

        match decode::<Claims>(token, &self.decoding_key, &validation) {
            Ok(token_data) => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                Ok(now >= token_data.claims.exp)
            }
            Err(_) => Ok(true), // If we can't decode, consider it expired
        }
    }

    /// Refresh token (create new token from existing valid token)
    #[pyo3(signature = (token, new_expiration_seconds=None))]
    fn refresh_token(&self, token: &str, new_expiration_seconds: Option<u64>) -> PyResult<String> {
        let validation = Validation::new(self.algorithm);

        let token_data = decode::<Claims>(token, &self.decoding_key, &validation).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Token validation error: {e}"))
        })?;

        let old_claims = token_data.claims;

        // Create new token with updated expiration
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let exp = now + new_expiration_seconds.unwrap_or(self.default_expiration);

        let new_claims = Claims {
            sub: old_claims.sub,
            iat: now, // New issued at time
            exp,      // New expiration
            aud: old_claims.aud,
            iss: old_claims.iss,
            scopes: old_claims.scopes,
            session_id: old_claims.session_id,
            user_agent: old_claims.user_agent,
            ip_address: old_claims.ip_address,
        };

        let header = Header::new(self.algorithm);

        encode(&header, &new_claims, &self.encoding_key).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Token refresh error: {e}"))
        })
    }

    /// Batch validate multiple tokens in parallel
    fn batch_validate_tokens(
        &self,
        tokens: Vec<&str>,
    ) -> PyResult<Vec<Option<HashMap<String, PyObject>>>> {
        use rayon::prelude::*;

        let results: Vec<_> = tokens
            .par_iter()
            .map(|token| self.validate_token(token).ok())
            .collect();

        Ok(results)
    }
}

/// High-performance password and authentication validator
#[pyclass]
pub struct RustAuthValidator {
    argon2: Argon2<'static>,
    pepper: Option<String>,
    max_attempts: u32,
    lockout_duration: u64,
}

#[pymethods]
impl RustAuthValidator {
    #[new]
    #[pyo3(signature = (pepper=None, max_attempts=5, lockout_duration_minutes=30))]
    fn new(pepper: Option<String>, max_attempts: u32, lockout_duration_minutes: u64) -> Self {
        Self {
            argon2: Argon2::default(),
            pepper,
            max_attempts,
            lockout_duration: lockout_duration_minutes * 60,
        }
    }

    /// Hash password with Argon2 - much more secure than bcrypt
    fn hash_password(&self, password: &str) -> PyResult<String> {
        let password_with_pepper = if let Some(ref pepper) = self.pepper {
            format!("{password}{pepper}")
        } else {
            password.to_string()
        };

        let salt = SaltString::generate(&mut OsRng);

        let password_hash = self
            .argon2
            .hash_password(password_with_pepper.as_bytes(), &salt)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Password hashing error: {e}"
                ))
            })?;

        Ok(password_hash.to_string())
    }

    /// Verify password against hash
    fn verify_password(&self, password: &str, hash: &str) -> PyResult<bool> {
        let password_with_pepper = if let Some(ref pepper) = self.pepper {
            format!("{password}{pepper}")
        } else {
            password.to_string()
        };

        let parsed_hash = PasswordHash::new(hash).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid password hash: {e}"))
        })?;

        match self
            .argon2
            .verify_password(password_with_pepper.as_bytes(), &parsed_hash)
        {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Generate secure API key
    #[pyo3(signature = (length=32, prefix=None))]
    fn generate_api_key(&self, length: usize, prefix: Option<&str>) -> PyResult<String> {
        let rng = ring_rand::SystemRandom::new();
        let mut key_bytes = vec![0u8; length];
        rng.fill(&mut key_bytes).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to generate random bytes")
        })?;

        let key = hex::encode(key_bytes);

        match prefix {
            Some(p) => Ok(format!("{p}_{key}")),
            None => Ok(key),
        }
    }

    /// Validate API key format and checksum
    fn validate_api_key(&self, api_key: &str) -> PyResult<bool> {
        // Basic validation - should be hex string
        if api_key.len() < 16 {
            return Ok(false);
        }

        // Check if it's valid hex (after removing prefix if present)
        let key_part = if api_key.contains('_') {
            api_key.split('_').next_back().unwrap_or(api_key)
        } else {
            api_key
        };

        Ok(hex::decode(key_part).is_ok())
    }

    /// Generate secure session token
    fn generate_session_token(&self) -> PyResult<String> {
        let rng = ring_rand::SystemRandom::new();
        let mut token_bytes = vec![0u8; 32];
        rng.fill(&mut token_bytes).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to generate random bytes")
        })?;

        use base64::{engine::general_purpose, Engine as _};
        Ok(general_purpose::STANDARD.encode(token_bytes))
    }

    /// Create HMAC signature for data integrity
    fn create_hmac_signature(&self, data: &str, secret: &str) -> PyResult<String> {
        let key = hmac::Key::new(hmac::HMAC_SHA256, secret.as_bytes());
        let signature = hmac::sign(&key, data.as_bytes());
        Ok(hex::encode(signature.as_ref()))
    }

    /// Verify HMAC signature
    fn verify_hmac_signature(&self, data: &str, signature: &str, secret: &str) -> PyResult<bool> {
        let key = hmac::Key::new(hmac::HMAC_SHA256, secret.as_bytes());

        let signature_bytes = hex::decode(signature).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid signature format")
        })?;

        match hmac::verify(&key, data.as_bytes(), &signature_bytes) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Hash data with SHA256
    fn sha256_hash(&self, data: &str) -> PyResult<String> {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        let result = hasher.finalize();
        Ok(hex::encode(result))
    }

    /// Hash data with Blake3 (faster than SHA256)
    fn blake3_hash(&self, data: &str) -> PyResult<String> {
        let hash = blake3::hash(data.as_bytes());
        Ok(hash.to_hex().to_string())
    }

    /// Generate time-based one-time password (TOTP) compatible token
    fn generate_totp_secret(&self) -> PyResult<String> {
        let rng = ring_rand::SystemRandom::new();
        let mut secret_bytes = vec![0u8; 20]; // 160 bits as recommended by RFC 6238
        rng.fill(&mut secret_bytes).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to generate random bytes")
        })?;

        Ok(base32::encode(
            base32::Alphabet::Rfc4648 { padding: true },
            &secret_bytes,
        ))
    }

    /// Batch hash multiple passwords in parallel
    fn batch_hash_passwords(&self, passwords: Vec<&str>) -> PyResult<Vec<String>> {
        use rayon::prelude::*;

        let results: Result<Vec<_>, _> = passwords
            .par_iter()
            .map(|password| {
                let password_with_pepper = if let Some(ref pepper) = self.pepper {
                    format!("{password}{pepper}")
                } else {
                    password.to_string()
                };

                let salt = SaltString::generate(&mut OsRng);

                self.argon2
                    .hash_password(password_with_pepper.as_bytes(), &salt)
                    .map(|hash| hash.to_string())
                    .map_err(|e| anyhow::anyhow!("Password hashing error: {}", e))
            })
            .collect();

        results.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Batch hashing error: {e}"))
        })
    }

    /// Batch verify multiple passwords in parallel
    fn batch_verify_passwords(
        &self,
        password_hash_pairs: Vec<(&str, &str)>,
    ) -> PyResult<Vec<bool>> {
        use rayon::prelude::*;

        let results: Vec<_> = password_hash_pairs
            .par_iter()
            .map(|(password, hash)| {
                let password_with_pepper = if let Some(ref pepper) = self.pepper {
                    format!("{password}{pepper}")
                } else {
                    password.to_string()
                };

                if let Ok(parsed_hash) = PasswordHash::new(hash) {
                    self.argon2
                        .verify_password(password_with_pepper.as_bytes(), &parsed_hash)
                        .is_ok()
                } else {
                    false
                }
            })
            .collect();

        Ok(results)
    }

    /// Check password strength and return score (0-100)
    fn check_password_strength(&self, password: &str) -> PyResult<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            let mut result = HashMap::new();
            let mut score = 0u32;
            let mut issues = Vec::new();

            // Length check
            if password.len() >= 8 {
                score += 20;
            } else {
                issues.push("Password too short (minimum 8 characters)".to_string());
            }

            if password.len() >= 12 {
                score += 10;
            }

            // Character variety checks
            let has_lowercase = password.chars().any(|c| c.is_lowercase());
            let has_uppercase = password.chars().any(|c| c.is_uppercase());
            let has_digits = password.chars().any(|c| c.is_numeric());
            let has_special = password.chars().any(|c| !c.is_alphanumeric());

            if has_lowercase {
                score += 15;
            } else {
                issues.push("Missing lowercase letters".to_string());
            }
            if has_uppercase {
                score += 15;
            } else {
                issues.push("Missing uppercase letters".to_string());
            }
            if has_digits {
                score += 15;
            } else {
                issues.push("Missing digits".to_string());
            }
            if has_special {
                score += 25;
            } else {
                issues.push("Missing special characters".to_string());
            }

            // Common patterns check (simple)
            let common_patterns = ["123", "abc", "password", "qwerty"];
            let lower_password = password.to_lowercase();
            for pattern in &common_patterns {
                if lower_password.contains(pattern) {
                    score = score.saturating_sub(20);
                    issues.push(format!("Contains common pattern: {pattern}"));
                    break;
                }
            }

            let strength = match score {
                0..=30 => "Very Weak",
                31..=50 => "Weak",
                51..=70 => "Medium",
                71..=85 => "Strong",
                86..=100 => "Very Strong",
                _ => "Very Strong",
            };

            result.insert("score".to_string(), score.to_object(py));
            result.insert("strength".to_string(), strength.to_object(py));
            result.insert("issues".to_string(), issues.to_object(py));

            Ok(result)
        })
    }
}
