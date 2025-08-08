//! Path utilities module
//!
//! This module provides utilities for handling cross-platform path translation,
//! particularly between Windows Subsystem for Linux (WSL) and native Windows paths.
//! It enables seamless file operations across WSL and Windows boundaries.
//!
//! # Security
//!
//! This module includes path canonicalization and validation to prevent path traversal attacks.
//! All paths are validated to ensure they remain within allowed directories.
//!
//! # Performance Optimizations
//!
//! This module includes an LRU cache to avoid repeated subprocess calls to `wslpath`.
//! The cache significantly improves performance for repeated path translations.

use crate::error::{FsError, FsResult};
use dashmap::DashMap;
use lru::LruCache;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::{Duration, Instant};
use tokio::process::Command as TokioCommand;

/// Cache size for path translations
const PATH_CACHE_SIZE: usize = 1000;

/// Cache size for path validations
const PATH_VALIDATION_CACHE_SIZE: usize = 2000;

/// TTL for cached path validations (5 minutes)
const PATH_VALIDATION_TTL: Duration = Duration::from_secs(300);

/// Server-wide configurable root directory for security
/// This can be configured via environment variable or defaults to current directory
static ROOT_DIR: Lazy<PathBuf> = Lazy::new(|| {
    std::env::var("RUST_FS_ROOT")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("Failed to get current directory"))
});

/// Cache key for path translations
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct PathCacheKey {
    path: String,
    to_windows: bool,
}

/// Cache entry for path validation with TTL
#[derive(Clone, Debug)]
struct PathValidationEntry {
    resolved_path: PathBuf,
    timestamp: Instant,
}

impl PathValidationEntry {
    fn new(resolved_path: PathBuf) -> Self {
        Self {
            resolved_path,
            timestamp: Instant::now(),
        }
    }

    fn is_expired(&self) -> bool {
        self.timestamp.elapsed() > PATH_VALIDATION_TTL
    }
}

/// Global LRU cache for path translations
static PATH_CACHE: Lazy<RwLock<LruCache<PathCacheKey, String>>> =
    Lazy::new(|| RwLock::new(LruCache::new(NonZeroUsize::new(PATH_CACHE_SIZE).unwrap())));

/// Global cache for path validations with TTL using DashMap for better concurrency
static PATH_VALIDATION_CACHE: Lazy<DashMap<String, PathValidationEntry>> =
    Lazy::new(|| DashMap::new());

/// Converts between WSL and Windows path formats.
///
/// This function uses the `wslpath` utility to translate paths between WSL's
/// Unix-style paths (e.g., `/mnt/c/Users`) and Windows-style paths (e.g., `C:\Users`).
pub async fn translate_path(path_str: &str, to_windows: bool) -> FsResult<String> {
    let cache_key = PathCacheKey {
        path: path_str.to_string(),
        to_windows,
    };

    // Try to get from cache first
    {
        let cache = PATH_CACHE.read().unwrap();
        if let Some(cached_path) = cache.peek(&cache_key) {
            return Ok(cached_path.clone());
        }
    }

    // Not in cache, perform the translation
    let flag = if to_windows { "-w" } else { "-u" };

    let output = TokioCommand::new("wslpath")
        .arg(flag)
        .arg(path_str)
        .output()
        .await
        .map_err(|e| FsError::WslCommand(e.to_string()))?;

    if output.status.success() {
        let translated = String::from_utf8_lossy(&output.stdout).trim().to_string();

        // Cache the result
        {
            let mut cache = PATH_CACHE.write().unwrap();
            cache.put(cache_key, translated.clone());
        }

        Ok(translated)
    } else {
        Err(FsError::WslCommand(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ))
    }
}

/// Resolves and validates a path, handling WSL translation if needed.
///
/// This function intelligently detects when path translation is needed based on
/// the current platform and the path format. It automatically converts between
/// WSL and Windows paths to ensure operations work correctly across platforms.
pub async fn resolve_path(path_str: &str, path_is_wsl: bool) -> FsResult<PathBuf> {
    let resolved = if path_is_wsl && cfg!(target_os = "windows") {
        translate_path(path_str, true).await?
    } else if !path_is_wsl && cfg!(target_os = "linux") && path_str.starts_with("C:\\") {
        translate_path(path_str, false).await?
    } else {
        path_str.to_string()
    };

    Ok(PathBuf::from(resolved))
}

/// Resolves and validates a path with security checks to prevent path traversal.
///
/// This function performs critical security validation:
/// 1. Canonicalizes the path to resolve '..' and '.' components
/// 2. Ensures the canonical path remains within the allowed root directory
/// 3. Prevents access to files outside the configured boundary
pub async fn resolve_and_validate_path(path_str: &str) -> FsResult<PathBuf> {
    // Security check: reject paths with null bytes
    if path_str.contains('\0') {
        return Err(FsError::io("Path contains null byte".to_string()));
    }

    // Security check: reject empty paths
    if path_str.is_empty() {
        return Err(FsError::io("Empty path provided".to_string()));
    }

    // Security check: normalize the path first to prevent traversal attacks
    let path = normalize_path_str(path_str)?;
    let path_buf = PathBuf::from(&path);

    // For non-existent paths (e.g., creating new files), we need to validate the parent
    let canonical_path = if let Ok(canonical) = tokio::fs::canonicalize(&path_buf).await {
        canonical
    } else {
        // Path doesn't exist yet, validate parent directory instead
        if let Some(parent) = path_buf.parent() {
            // Security: Ensure parent path doesn't contain .. after normalization
            let parent_str = parent.to_string_lossy();
            if parent_str.contains("..") {
                return Err(FsError::ForbiddenPath(format!("Path traversal detected in parent: {}", path_str)));
            }

            // Try to canonicalize the parent
            if let Ok(canonical_parent) = tokio::fs::canonicalize(parent).await {
                // Ensure the parent is within the allowed root
                if !canonical_parent.starts_with(&*ROOT_DIR) {
                    return Err(FsError::ForbiddenPath(path_str.to_string()));
                }

                // Construct the full path by joining the canonical parent with the file name
                if let Some(file_name) = path_buf.file_name() {
                    // Security: Validate file name doesn't contain path separators
                    let file_name_str = file_name.to_string_lossy();
                    if file_name_str.contains('/') || file_name_str.contains('\\') {
                        return Err(FsError::io(format!("File name contains path separator: {}", file_name_str)));
                    }
                    canonical_parent.join(file_name)
                } else {
                    return Err(FsError::io(format!("Invalid path '{}': no file name", path_str)));
                }
            } else {
                // Parent doesn't exist either - walk up until we find an existing parent
                let mut current = parent.to_path_buf();
                let mut components_to_add = vec![path_buf.file_name().ok_or_else(||
                    FsError::io(format!("Invalid path '{}': no file name", path_str)))?.to_os_string()];

                loop {
                    if let Some(parent_path) = current.parent() {
                        if let Ok(canonical_parent) = tokio::fs::canonicalize(parent_path).await {
                            // Found an existing parent, check if it's within root
                            if !canonical_parent.starts_with(&*ROOT_DIR) {
                                return Err(FsError::ForbiddenPath(path_str.to_string()));
                            }

                            // Rebuild the path from the canonical parent
                            let mut result = canonical_parent;
                            for component in components_to_add.iter().rev() {
                                let component_str = component.to_string_lossy();
                                if component_str.contains('/') || component_str.contains('\\') {
                                    return Err(FsError::io(format!("Path component contains separator: {}", component_str)));
                                }
                                result = result.join(component);
                            }

                            break result;
                        } else {
                            // This parent doesn't exist either, keep looking
                            if let Some(file_name) = current.file_name() {
                                components_to_add.push(file_name.to_os_string());
                                current = parent_path.to_path_buf();
                            } else {
                                return Err(FsError::io(format!("Invalid path structure: {}", path_str)));
                            }
                        }
                    } else {
                        // Reached the root without finding an existing parent
                        return Err(FsError::io(format!("No valid parent directory found for: {}", path_str)));
                    }
                }
            }
        } else {
            // No parent directory, use the normalized path if it's within root
            let normalized = normalize_path(&path_buf);
            if !normalized.starts_with(&*ROOT_DIR) {
                return Err(FsError::ForbiddenPath(path_str.to_string()));
            }
            normalized
        }
    };

    // Final check: Ensure the resolved path is within the allowed root
    if !canonical_path.starts_with(&*ROOT_DIR) {
        return Err(FsError::ForbiddenPath(path_str.to_string()));
    }

    Ok(canonical_path)
}

/// Cached version of resolve_and_validate_path with TTL-based expiration
///
/// This function provides significant performance improvements for repeated path
/// operations by caching validated paths for 5 minutes. It maintains the same
/// security guarantees as resolve_and_validate_path.
pub async fn resolve_and_validate_path_cached(path_str: &str) -> FsResult<PathBuf> {
    // Check cache first - remove expired entries
    if let Some(entry) = PATH_VALIDATION_CACHE.get(path_str) {
        if !entry.is_expired() {
            return Ok(entry.resolved_path.clone());
        } else {
            // Entry is expired, remove it
            PATH_VALIDATION_CACHE.remove(path_str);
        }
    }

    // Not in cache or expired, perform validation
    let resolved_path = resolve_and_validate_path(path_str).await?;

    // Cache the result if cache isn't too full
    if PATH_VALIDATION_CACHE.len() < PATH_VALIDATION_CACHE_SIZE {
        PATH_VALIDATION_CACHE.insert(
            path_str.to_string(),
            PathValidationEntry::new(resolved_path.clone()),
        );
    }

    Ok(resolved_path)
}

/// Clean expired entries from the path validation cache
/// This should be called periodically to prevent memory leaks
pub fn cleanup_path_cache() {
    PATH_VALIDATION_CACHE.retain(|_, entry| !entry.is_expired());
}

/// Get cache statistics for monitoring
pub fn get_path_cache_stats() -> (usize, usize, usize) {
    let validation_cache_size = PATH_VALIDATION_CACHE.len();
    let translation_cache_size = {
        let cache = PATH_CACHE.read().unwrap();
        cache.len()
    };
    let max_validation_size = PATH_VALIDATION_CACHE_SIZE;

    (validation_cache_size, translation_cache_size, max_validation_size)
}

/// Normalize a path string to prevent directory traversal attacks
pub fn normalize_path_str(path_str: &str) -> FsResult<String> {
    // Security check: reject empty paths
    if path_str.is_empty() {
        return Err(FsError::io("Empty path provided".to_string()));
    }

    // Security check: reject paths with null bytes
    if path_str.contains('\0') {
        return Err(FsError::io("Path contains null byte".to_string()));
    }

    // Replace backslashes with forward slashes for consistent handling
    let normalized = path_str.replace('\\', "/");

    // Split into components and rebuild, removing .. and . components
    let parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();
    let mut stack: Vec<&str> = Vec::new();

    let is_absolute = path_str.starts_with('/') || (path_str.len() >= 2 && path_str.chars().nth(1) == Some(':'));

    for part in parts {
        match part {
            "." => continue,  // Skip current directory
            ".." => {
                // For absolute paths, prevent going above root
                if is_absolute && stack.is_empty() {
                    return Err(FsError::ForbiddenPath(format!("Path traversal attempt: {}", path_str)));
                }
                // For relative paths, only pop if we have components
                if !stack.is_empty() {
                    stack.pop();
                } else if !is_absolute {
                    // For relative paths, we need to track how many levels up
                    return Err(FsError::ForbiddenPath(format!("Path traversal attempt: {}", path_str)));
                }
            }
            _ => stack.push(part),
        }
    }

    // Reconstruct the path
    let result = if is_absolute {
        if cfg!(windows) && path_str.len() >= 2 && path_str.chars().nth(1) == Some(':') {
            // Windows absolute path like C:/
            format!("{}/{}", &path_str[0..2], stack.join("/"))
        } else {
            format!("/{}", stack.join("/"))
        }
    } else {
        stack.join("/")
    };

    // Final check: ensure no .. remains after normalization
    if result.contains("..") {
        return Err(FsError::ForbiddenPath(format!("Path traversal detected after normalization: {}", path_str)));
    }

    Ok(result)
}

/// Normalize a PathBuf by resolving . and .. components
fn normalize_path(path: &PathBuf) -> PathBuf {
    use std::path::Component;

    let mut components = vec![];

    for component in path.components() {
        match component {
            Component::ParentDir => {
                // Only pop if we have components and the last one isn't RootDir
                if !components.is_empty() {
                    if let Some(Component::RootDir) = components.last() {
                        // Don't pop the root directory
                    } else {
                        components.pop();
                    }
                }
            }
            Component::CurDir => {
                // Skip current directory markers
            }
            _ => {
                components.push(component);
            }
        }
    }

    components.iter().collect()
}

/// PyO3 wrapper for path utilities
#[pyclass]
pub struct RustPathUtils;

#[pymethods]
impl RustPathUtils {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Resolve and validate path (async)
    fn resolve_and_validate<'py>(
        &self,
        py: Python<'py>,
        path_str: String,
    ) -> PyResult<&'py PyAny> {
        pyo3_asyncio::tokio::return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            resolve_and_validate_path_cached(&path_str)
                .await
                .map(|p| p.to_string_lossy().to_string())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        })
    }

    /// Normalize path string
    fn normalize_path(&self, path_str: &str) -> PyResult<String> {
        normalize_path_str(path_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Get path cache statistics
    fn get_cache_stats(&self) -> (usize, usize, usize) {
        get_path_cache_stats()
    }

    /// Clean expired cache entries
    fn cleanup_cache(&self) {
        cleanup_path_cache();
    }

    /// Get root directory
    fn get_root_dir(&self) -> String {
        ROOT_DIR.to_string_lossy().to_string()
    }

    /// Set root directory (for testing)
    fn set_root_dir(&self, _path: &str) -> PyResult<()> {
        // This would require unsafe code to modify the static,
        // so for now we just return an error
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Root directory cannot be changed at runtime"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path_str() {
        assert_eq!(normalize_path_str("/home/user/file.txt").unwrap(), "/home/user/file.txt");
        assert_eq!(normalize_path_str("/home/user/../other/file.txt").unwrap(), "/home/other/file.txt");
        assert_eq!(normalize_path_str("/home/user/./file.txt").unwrap(), "/home/user/file.txt");

        // Should reject path traversal attempts
        assert!(normalize_path_str("/../etc/passwd").is_err());
        assert!(normalize_path_str("/home/../../etc/passwd").is_err());
    }

    #[test]
    fn test_path_validation_security() {
        // These should be rejected for security
        assert!(normalize_path_str("").is_err());
        assert!(normalize_path_str("file\0name").is_err());
        assert!(normalize_path_str("../../../etc/passwd").is_err());
    }

    #[tokio::test]
    async fn test_path_caching() {
        // This test would need a controlled environment to test properly
        // For now, just verify the cache stats function works
        let stats = get_path_cache_stats();
        assert!(stats.0 <= stats.2); // Used <= max
        assert!(stats.1 <= PATH_CACHE_SIZE); // Translation cache size reasonable
    }
}
