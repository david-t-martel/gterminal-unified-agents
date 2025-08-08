//! Enhanced TTL (Time-To-Live) Cache Implementation
//!
//! This module provides a high-performance TTL cache with statistics tracking
//! and PyO3 bindings for Python integration.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Statistics for cache performance monitoring
#[derive(Debug, Clone)]
#[pyclass]
pub struct CacheStats {
    #[pyo3(get)]
    pub hits: u64,
    #[pyo3(get)]
    pub misses: u64,
    #[pyo3(get)]
    pub total_entries: u64,
    #[pyo3(get)]
    pub expired_entries: u64,
}

#[pymethods]
impl CacheStats {
    #[new]
    pub fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            total_entries: 0,
            expired_entries: 0,
        }
    }

    /// Calculate hit ratio as percentage
    #[getter]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.total_entries = 0;
        self.expired_entries = 0;
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache entry with TTL support
#[derive(Debug, Clone)]
struct CacheEntry {
    value: String,
    expires_at: Instant,
}

impl CacheEntry {
    fn new(value: String, ttl: Duration) -> Self {
        Self {
            value,
            expires_at: Instant::now() + ttl,
        }
    }

    fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// Enhanced TTL Cache with statistics and performance monitoring
#[derive(Debug, Clone)]
#[pyclass]
pub struct EnhancedTtlCache {
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    stats: Arc<Mutex<CacheStats>>,
    default_ttl: Duration,
}

#[pymethods]
impl EnhancedTtlCache {
    /// Create a new TTL cache with default TTL in seconds
    #[new]
    pub fn new(default_ttl_seconds: u64) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CacheStats::new())),
            default_ttl: Duration::from_secs(default_ttl_seconds),
        }
    }

    /// Insert a value with default TTL
    pub fn set(&self, key: String, value: String) -> PyResult<()> {
        self.set_with_ttl(key, value, self.default_ttl.as_secs())
    }

    /// Insert a value with custom TTL in seconds
    pub fn set_with_ttl(&self, key: String, value: String, ttl_seconds: u64) -> PyResult<()> {
        let entry = CacheEntry::new(value, Duration::from_secs(ttl_seconds));

        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, entry);

            if let Ok(mut stats) = self.stats.lock() {
                stats.total_entries += 1;
            }
        }

        Ok(())
    }

    /// Get a value from the cache
    pub fn get(&self, key: &str) -> PyResult<Option<String>> {
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(entry) = cache.get(key) {
                if entry.is_expired() {
                    cache.remove(key);
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.misses += 1;
                        stats.expired_entries += 1;
                    }
                    return Ok(None);
                }

                if let Ok(mut stats) = self.stats.lock() {
                    stats.hits += 1;
                }

                return Ok(Some(entry.value.clone()));
            }
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.misses += 1;
        }

        Ok(None)
    }

    /// Remove a value from the cache
    pub fn remove(&self, key: &str) -> PyResult<bool> {
        if let Ok(mut cache) = self.cache.lock() {
            Ok(cache.remove(key).is_some())
        } else {
            Ok(false)
        }
    }

    /// Clear all entries from the cache
    pub fn clear(&self) -> PyResult<()> {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }

        if let Ok(mut stats) = self.stats.lock() {
            *stats = CacheStats::new();
        }

        Ok(())
    }

    /// Get current cache size
    #[getter]
    pub fn size(&self) -> PyResult<usize> {
        if let Ok(cache) = self.cache.lock() {
            Ok(cache.len())
        } else {
            Ok(0)
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> PyResult<CacheStats> {
        if let Ok(stats) = self.stats.lock() {
            Ok(stats.clone())
        } else {
            Ok(CacheStats::new())
        }
    }

    /// Clean up expired entries
    pub fn cleanup_expired(&self) -> PyResult<u64> {
        let mut removed_count = 0;

        if let Ok(mut cache) = self.cache.lock() {
            let expired_keys: Vec<String> = cache
                .iter()
                .filter_map(|(key, entry)| {
                    if entry.is_expired() {
                        Some(key.clone())
                    } else {
                        None
                    }
                })
                .collect();

            for key in &expired_keys {
                cache.remove(key);
                removed_count += 1;
            }

            if let Ok(mut stats) = self.stats.lock() {
                stats.expired_entries += removed_count;
            }
        }

        Ok(removed_count)
    }

    /// Check if key exists and is not expired
    pub fn contains_key(&self, key: &str) -> PyResult<bool> {
        if let Ok(cache) = self.cache.lock() {
            if let Some(entry) = cache.get(key) {
                return Ok(!entry.is_expired());
            }
        }
        Ok(false)
    }

    /// Get all non-expired keys
    pub fn keys(&self) -> PyResult<Vec<String>> {
        if let Ok(cache) = self.cache.lock() {
            let keys = cache
                .iter()
                .filter_map(|(key, entry)| {
                    if !entry.is_expired() {
                        Some(key.clone())
                    } else {
                        None
                    }
                })
                .collect();
            Ok(keys)
        } else {
            Ok(vec![])
        }
    }
}

impl Default for EnhancedTtlCache {
    fn default() -> Self {
        Self::new(3600) // Default to 1 hour TTL
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_cache_operations() {
        let cache = EnhancedTtlCache::new(10);

        // Test set and get
        assert!(cache.set("key1".to_string(), "value1".to_string()).is_ok());
        assert_eq!(cache.get("key1").unwrap(), Some("value1".to_string()));

        // Test non-existent key
        assert_eq!(cache.get("nonexistent").unwrap(), None);

        // Test remove
        assert_eq!(cache.remove("key1").unwrap(), true);
        assert_eq!(cache.get("key1").unwrap(), None);
        assert_eq!(cache.remove("key1").unwrap(), false);
    }

    #[test]
    fn test_ttl_expiration() {
        let cache = EnhancedTtlCache::new(1);

        // Set value with 1 second TTL
        cache.set_with_ttl("temp_key".to_string(), "temp_value".to_string(), 1).unwrap();
        assert_eq!(cache.get("temp_key").unwrap(), Some("temp_value".to_string()));

        // Wait for expiration
        thread::sleep(Duration::from_millis(1100));
        assert_eq!(cache.get("temp_key").unwrap(), None);
    }

    #[test]
    fn test_statistics() {
        let cache = EnhancedTtlCache::new(10);

        // Test hits and misses
        cache.set("key1".to_string(), "value1".to_string()).unwrap();
        cache.get("key1").unwrap(); // hit
        cache.get("nonexistent").unwrap(); // miss

        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_entries, 1);
        assert!(stats.hit_ratio() > 0.0);
    }

    #[test]
    fn test_cleanup_expired() {
        let cache = EnhancedTtlCache::new(1);

        // Add entries with short TTL
        cache.set_with_ttl("key1".to_string(), "value1".to_string(), 1).unwrap();
        cache.set_with_ttl("key2".to_string(), "value2".to_string(), 1).unwrap();

        // Wait for expiration
        thread::sleep(Duration::from_millis(1100));

        // Cleanup should remove expired entries
        let removed = cache.cleanup_expired().unwrap();
        assert_eq!(removed, 2);
    }
}
