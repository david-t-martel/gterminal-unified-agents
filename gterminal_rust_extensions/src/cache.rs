//! RustCache: High-performance concurrent cache with TTL and LRU eviction

use crate::utils::{increment_ops, current_timestamp, track_allocation, track_deallocation};
use anyhow::{Context, Result};
use dashmap::DashMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::{interval, Instant};

/// Cache entry with TTL and access tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    value: PyObject,
    created_at: u64,
    expires_at: Option<u64>,
    last_accessed: u64,
    access_count: u64,
    size_bytes: u64,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CacheStats {
    hits: u64,
    misses: u64,
    evictions: u64,
    expired: u64,
    total_ops: u64,
    memory_used: u64,
    peak_memory: u64,
}

/// High-performance concurrent cache
#[pyclass]
pub struct RustCache {
    storage: Arc<DashMap<String, CacheEntry>>,
    stats: Arc<parking_lot::RwLock<CacheStats>>,
    max_capacity: usize,
    default_ttl: Option<u64>,
    max_memory: Option<u64>,
    cleanup_interval: Duration,
    runtime: tokio::runtime::Runtime,
    cleanup_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

#[pymethods]
impl RustCache {
    /// Create new cache instance
    #[new]
    #[pyo3(signature = (capacity = 10000, default_ttl_secs = None, max_memory_bytes = None, cleanup_interval_secs = 60))]
    fn new(
        capacity: usize,
        default_ttl_secs: Option<u64>,
        max_memory_bytes: Option<u64>,
        cleanup_interval_secs: u64,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("cache")
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let cache = Self {
            storage: Arc::new(DashMap::new()),
            stats: Arc::new(parking_lot::RwLock::new(CacheStats::default())),
            max_capacity: capacity,
            default_ttl: default_ttl_secs,
            max_memory: max_memory_bytes,
            cleanup_interval: Duration::from_secs(cleanup_interval_secs),
            runtime,
            cleanup_handle: Arc::new(RwLock::new(None)),
        };

        // Start background cleanup task
        cache.start_cleanup_task();

        Ok(cache)
    }

    /// Store value in cache
    #[pyo3(signature = (key, value, ttl_secs = None))]
    fn set(&self, key: String, value: PyObject, ttl_secs: Option<u64>) -> PyResult<bool> {
        self.runtime.block_on(async {
            let now = current_timestamp();
            let ttl = ttl_secs.or(self.default_ttl);
            let expires_at = ttl.map(|t| now + t);

            // Calculate approximate size
            let size_bytes = self.estimate_size(&key, &value);

            // Check memory limit
            if let Some(max_mem) = self.max_memory {
                let current_memory = {
                    let stats = self.stats.read();
                    stats.memory_used
                };

                if current_memory + size_bytes > max_mem {
                    // Try to free memory by evicting LRU items
                    self.evict_lru_items(size_bytes).await;

                    // Check again
                    let current_memory = {
                        let stats = self.stats.read();
                        stats.memory_used
                    };

                    if current_memory + size_bytes > max_mem {
                        return Ok(false); // Cannot store due to memory limit
                    }
                }
            }

            // Check capacity limit and evict if necessary
            if self.storage.len() >= self.max_capacity {
                self.evict_lru_items(0).await; // Evict at least one item
            }

            let entry = CacheEntry {
                value,
                created_at: now,
                expires_at,
                last_accessed: now,
                access_count: 1,
                size_bytes,
            };

            // Update existing entry or insert new one
            let was_update = self.storage.contains_key(&key);
            let old_size = if was_update {
                self.storage.get(&key).map(|e| e.size_bytes).unwrap_or(0)
            } else {
                0
            };

            self.storage.insert(key, entry);

            // Update statistics
            {
                let mut stats = self.stats.write();
                if was_update {
                    stats.memory_used = stats.memory_used.saturating_sub(old_size) + size_bytes;
                } else {
                    stats.memory_used += size_bytes;
                    if stats.memory_used > stats.peak_memory {
                        stats.peak_memory = stats.memory_used;
                    }
                }
                stats.total_ops += 1;
            }

            increment_ops();
            track_allocation(size_bytes);
            if was_update {
                track_deallocation(old_size);
            }

            Ok(true)
        })
    }

    /// Retrieve value from cache
    fn get(&self, key: &str) -> PyResult<Option<PyObject>> {
        let now = current_timestamp();

        if let Some(mut entry) = self.storage.get_mut(key) {
            // Check if expired
            if let Some(expires_at) = entry.expires_at {
                if now > expires_at {
                    drop(entry); // Release the lock
                    self.storage.remove(key);

                    let mut stats = self.stats.write();
                    stats.misses += 1;
                    stats.expired += 1;
                    stats.total_ops += 1;

                    increment_ops();
                    return Ok(None);
                }
            }

            // Update access tracking
            entry.last_accessed = now;
            entry.access_count += 1;

            let value = entry.value.clone();

            // Update statistics
            {
                let mut stats = self.stats.write();
                stats.hits += 1;
                stats.total_ops += 1;
            }

            increment_ops();
            Ok(Some(value))
        } else {
            // Cache miss
            let mut stats = self.stats.write();
            stats.misses += 1;
            stats.total_ops += 1;

            increment_ops();
            Ok(None)
        }
    }

    /// Remove key from cache
    fn delete(&self, key: &str) -> PyResult<bool> {
        if let Some((_, entry)) = self.storage.remove(key) {
            // Update statistics
            {
                let mut stats = self.stats.write();
                stats.memory_used = stats.memory_used.saturating_sub(entry.size_bytes);
                stats.total_ops += 1;
            }

            increment_ops();
            track_deallocation(entry.size_bytes);
            Ok(true)
        } else {
            let mut stats = self.stats.write();
            stats.total_ops += 1;

            increment_ops();
            Ok(false)
        }
    }

    /// Check if key exists (without updating access time)
    fn exists(&self, key: &str) -> PyResult<bool> {
        let now = current_timestamp();

        if let Some(entry) = self.storage.get(key) {
            // Check if expired
            if let Some(expires_at) = entry.expires_at {
                if now > expires_at {
                    drop(entry);
                    self.storage.remove(key);
                    return Ok(false);
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get multiple values at once
    fn get_many(&self, keys: Vec<&str>) -> PyResult<std::collections::HashMap<String, PyObject>> {
        let mut result = std::collections::HashMap::new();
        let now = current_timestamp();

        for key in keys {
            if let Some(mut entry) = self.storage.get_mut(key) {
                // Check if expired
                if let Some(expires_at) = entry.expires_at {
                    if now > expires_at {
                        drop(entry);
                        self.storage.remove(key);
                        continue;
                    }
                }

                // Update access tracking
                entry.last_accessed = now;
                entry.access_count += 1;

                result.insert(key.to_string(), entry.value.clone());

                // Update hit count
                {
                    let mut stats = self.stats.write();
                    stats.hits += 1;
                    stats.total_ops += 1;
                }
            } else {
                // Update miss count
                let mut stats = self.stats.write();
                stats.misses += 1;
                stats.total_ops += 1;
            }
        }

        increment_ops();
        Ok(result)
    }

    /// Set multiple key-value pairs
    fn set_many(&self, items: std::collections::HashMap<String, PyObject>) -> PyResult<Vec<String>> {
        self.runtime.block_on(async {
            let mut successful = Vec::new();

            for (key, value) in items {
                if self.set(key.clone(), value, None)? {
                    successful.push(key);
                }
            }

            Ok(successful)
        })
    }

    /// Get all keys matching pattern
    fn keys(&self, pattern: Option<&str>) -> PyResult<Vec<String>> {
        let regex = pattern.map(|p| regex::Regex::new(p))
            .transpose()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let keys: Vec<String> = self.storage.iter()
            .filter_map(|entry| {
                let key = entry.key();
                if let Some(ref r) = regex {
                    if r.is_match(key) {
                        Some(key.clone())
                    } else {
                        None
                    }
                } else {
                    Some(key.clone())
                }
            })
            .collect();

        Ok(keys)
    }

    /// Get cache size (number of entries)
    fn size(&self) -> PyResult<usize> {
        Ok(self.storage.len())
    }

    /// Get memory usage in bytes
    fn memory_usage(&self) -> PyResult<u64> {
        let stats = self.stats.read();
        Ok(stats.memory_used)
    }

    /// Clear all entries
    fn clear(&self) -> PyResult<usize> {
        let count = self.storage.len();
        self.storage.clear();

        // Reset memory usage
        {
            let mut stats = self.stats.write();
            track_deallocation(stats.memory_used);
            stats.memory_used = 0;
            stats.total_ops += 1;
        }

        increment_ops();
        Ok(count)
    }

    /// Set TTL for existing key
    fn expire(&self, key: &str, ttl_secs: u64) -> PyResult<bool> {
        if let Some(mut entry) = self.storage.get_mut(key) {
            let now = current_timestamp();
            entry.expires_at = Some(now + ttl_secs);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get TTL for key (seconds remaining)
    fn ttl(&self, key: &str) -> PyResult<Option<u64>> {
        if let Some(entry) = self.storage.get(key) {
            if let Some(expires_at) = entry.expires_at {
                let now = current_timestamp();
                if expires_at > now {
                    Ok(Some(expires_at - now))
                } else {
                    Ok(Some(0)) // Expired
                }
            } else {
                Ok(None) // No TTL set
            }
        } else {
            Ok(None) // Key not found
        }
    }

    /// Increment counter value
    #[pyo3(signature = (key, increment = 1, ttl_secs = None))]
    fn incr(&self, key: &str, increment: i64, ttl_secs: Option<u64>) -> PyResult<i64> {
        Python::with_gil(|py| {
            let current_value = self.get(key)?
                .and_then(|obj| obj.extract::<i64>(py).ok())
                .unwrap_or(0);

            let new_value = current_value + increment;
            self.set(key.to_string(), new_value.into_py(py), ttl_secs)?;

            Ok(new_value)
        })
    }

    /// Decrement counter value
    #[pyo3(signature = (key, decrement = 1, ttl_secs = None))]
    fn decr(&self, key: &str, decrement: i64, ttl_secs: Option<u64>) -> PyResult<i64> {
        self.incr(key, -decrement, ttl_secs)
    }

    /// Get cache statistics
    fn get_stats(&self) -> PyResult<std::collections::HashMap<String, u64>> {
        let stats = self.stats.read();
        let mut result = std::collections::HashMap::new();

        result.insert("hits".to_string(), stats.hits);
        result.insert("misses".to_string(), stats.misses);
        result.insert("evictions".to_string(), stats.evictions);
        result.insert("expired".to_string(), stats.expired);
        result.insert("total_ops".to_string(), stats.total_ops);
        result.insert("memory_used".to_string(), stats.memory_used);
        result.insert("peak_memory".to_string(), stats.peak_memory);
        result.insert("entries".to_string(), self.storage.len() as u64);

        // Calculate hit rate
        let total_gets = stats.hits + stats.misses;
        let hit_rate = if total_gets > 0 {
            (stats.hits * 100) / total_gets
        } else {
            0
        };
        result.insert("hit_rate_percent".to_string(), hit_rate);

        Ok(result)
    }

    /// Reset cache statistics
    fn clear_stats(&self) -> PyResult<()> {
        let mut stats = self.stats.write();
        *stats = CacheStats::default();
        Ok(())
    }

    /// Manual cleanup of expired entries
    fn cleanup(&self) -> PyResult<u64> {
        self.runtime.block_on(async {
            let removed = self.cleanup_expired().await;
            Ok(removed)
        })
    }

    /// Persist cache to file (snapshot)
    fn save_snapshot(&self, path: &str) -> PyResult<usize> {
        self.runtime.block_on(async {
            let snapshot = self.create_snapshot().await;
            let serialized = serde_json::to_vec(&snapshot)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // Compress the data
            use flate2::write::GzEncoder;
            use flate2::Compression;
            use std::io::Write;

            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&serialized)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let compressed = encoder.finish()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            tokio::fs::write(path, &compressed).await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            Ok(compressed.len())
        })
    }

    /// Load cache from snapshot file
    fn load_snapshot(&self, path: &str) -> PyResult<usize> {
        self.runtime.block_on(async {
            let compressed = tokio::fs::read(path).await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // Decompress the data
            use flate2::read::GzDecoder;
            use std::io::Read;

            let mut decoder = GzDecoder::new(&compressed[..]);
            let mut serialized = Vec::new();
            decoder.read_to_end(&mut serialized)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            let snapshot: CacheSnapshot = serde_json::from_slice(&serialized)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            let loaded = self.restore_snapshot(snapshot).await?;
            Ok(loaded)
        })
    }

    /// Optimize cache by removing expired and least-used entries
    fn optimize(&self) -> PyResult<std::collections::HashMap<String, u64>> {
        self.runtime.block_on(async {
            let mut results = std::collections::HashMap::new();

            // Clean up expired entries
            let expired_removed = self.cleanup_expired().await;
            results.insert("expired_removed".to_string(), expired_removed);

            // If still over capacity, remove LRU entries
            let capacity_removed = if self.storage.len() > self.max_capacity {
                self.evict_lru_items(0).await
            } else {
                0
            };
            results.insert("lru_removed".to_string(), capacity_removed);

            // Update peak memory if current usage is lower
            {
                let mut stats = self.stats.write();
                if stats.memory_used < stats.peak_memory {
                    stats.peak_memory = stats.memory_used;
                }
            }

            Ok(results)
        })
    }
}

/// Snapshot structure for persistence
#[derive(Serialize, Deserialize)]
struct CacheSnapshot {
    entries: Vec<(String, CacheEntry)>,
    stats: CacheStats,
    metadata: std::collections::HashMap<String, String>,
}

impl RustCache {
    /// Start background cleanup task
    fn start_cleanup_task(&self) {
        let storage = Arc::clone(&self.storage);
        let stats = Arc::clone(&self.stats);
        let cleanup_interval = self.cleanup_interval;
        let cleanup_handle = Arc::clone(&self.cleanup_handle);

        self.runtime.spawn(async move {
            let mut interval = interval(cleanup_interval);

            loop {
                interval.tick().await;

                // Check if we should stop
                if Arc::strong_count(&storage) <= 2 { // Only this task and the main struct
                    break;
                }

                // Cleanup expired entries
                let now = current_timestamp();
                let mut expired_count = 0u64;
                let mut freed_memory = 0u64;

                storage.retain(|_, entry| {
                    if let Some(expires_at) = entry.expires_at {
                        if now > expires_at {
                            expired_count += 1;
                            freed_memory += entry.size_bytes;
                            track_deallocation(entry.size_bytes);
                            return false;
                        }
                    }
                    true
                });

                // Update statistics
                if expired_count > 0 {
                    let mut stats = stats.write();
                    stats.expired += expired_count;
                    stats.memory_used = stats.memory_used.saturating_sub(freed_memory);

                    tracing::debug!("Cleaned up {} expired entries, freed {} bytes",
                        expired_count, freed_memory);
                }
            }

            tracing::debug!("Cache cleanup task terminated");
        });

        // Store handle for cleanup
        self.runtime.spawn(async move {
            if let Ok(mut handle_guard) = cleanup_handle.write().await {
                // The cleanup task is managed by the runtime, not stored directly
                *handle_guard = None;
            }
        });
    }

    /// Estimate size of key-value pair in bytes
    fn estimate_size(&self, key: &str, value: &PyObject) -> u64 {
        let key_size = key.len() as u64;

        // Rough estimate of Python object size
        let value_size = Python::with_gil(|py| -> u64 {
            if let Ok(s) = value.extract::<String>(py) {
                s.len() as u64
            } else if let Ok(_) = value.extract::<i64>(py) {
                8
            } else if let Ok(_) = value.extract::<f64>(py) {
                8
            } else if let Ok(_) = value.extract::<bool>(py) {
                1
            } else if let Ok(bytes) = value.extract::<Vec<u8>>(py) {
                bytes.len() as u64
            } else {
                // Default estimate for complex objects
                256
            }
        });

        key_size + value_size + 128 // Add overhead for metadata
    }

    /// Clean up expired entries
    async fn cleanup_expired(&self) -> u64 {
        let now = current_timestamp();
        let mut expired_count = 0u64;
        let mut freed_memory = 0u64;

        self.storage.retain(|_, entry| {
            if let Some(expires_at) = entry.expires_at {
                if now > expires_at {
                    expired_count += 1;
                    freed_memory += entry.size_bytes;
                    track_deallocation(entry.size_bytes);
                    return false;
                }
            }
            true
        });

        // Update statistics
        if expired_count > 0 {
            let mut stats = self.stats.write();
            stats.expired += expired_count;
            stats.memory_used = stats.memory_used.saturating_sub(freed_memory);
        }

        expired_count
    }

    /// Evict least recently used items
    async fn evict_lru_items(&self, required_bytes: u64) -> u64 {
        // Collect entries with access info
        let mut entries: Vec<(String, u64, u64, u64)> = self.storage.iter()
            .map(|item| {
                let key = item.key().clone();
                let entry = item.value();
                (key, entry.last_accessed, entry.access_count, entry.size_bytes)
            })
            .collect();

        // Sort by last_accessed (ascending) then by access_count (ascending)
        entries.sort_by(|a, b| {
            a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2))
        });

        let mut evicted_count = 0u64;
        let mut freed_memory = 0u64;
        let target_count = std::cmp::max(1, self.storage.len() / 10); // Evict at least 10%

        for (key, _, _, size) in entries.into_iter().take(target_count) {
            if let Some((_, entry)) = self.storage.remove(&key) {
                evicted_count += 1;
                freed_memory += entry.size_bytes;
                track_deallocation(entry.size_bytes);

                // Stop if we've freed enough memory
                if required_bytes > 0 && freed_memory >= required_bytes {
                    break;
                }
            }
        }

        // Update statistics
        if evicted_count > 0 {
            let mut stats = self.stats.write();
            stats.evictions += evicted_count;
            stats.memory_used = stats.memory_used.saturating_sub(freed_memory);
        }

        evicted_count
    }

    /// Create snapshot for persistence
    async fn create_snapshot(&self) -> CacheSnapshot {
        let entries: Vec<(String, CacheEntry)> = self.storage.iter()
            .map(|item| (item.key().clone(), item.value().clone()))
            .collect();

        let stats = self.stats.read().clone();

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("version".to_string(), "1.0".to_string());
        metadata.insert("created_at".to_string(), current_timestamp().to_string());
        metadata.insert("entry_count".to_string(), entries.len().to_string());

        CacheSnapshot {
            entries,
            stats,
            metadata,
        }
    }

    /// Restore from snapshot
    async fn restore_snapshot(&self, snapshot: CacheSnapshot) -> PyResult<usize> {
        // Clear current cache
        self.storage.clear();

        let now = current_timestamp();
        let mut loaded_count = 0;
        let mut memory_used = 0;

        Python::with_gil(|py| -> PyResult<()> {
            for (key, mut entry) in snapshot.entries {
                // Skip expired entries
                if let Some(expires_at) = entry.expires_at {
                    if now > expires_at {
                        continue;
                    }
                }

                // Convert serialized value back to PyObject
                // Note: This is a simplified approach - in practice, you'd need
                // proper serialization/deserialization of Python objects
                memory_used += entry.size_bytes;
                self.storage.insert(key, entry);
                loaded_count += 1;
            }

            // Restore statistics (partially)
            {
                let mut stats = self.stats.write();
                stats.memory_used = memory_used;
                if memory_used > stats.peak_memory {
                    stats.peak_memory = memory_used;
                }
            }

            Ok(())
        })?;

        Ok(loaded_count)
    }
}

impl Drop for RustCache {
    fn drop(&mut self) {
        // The cleanup task will detect the reduced reference count and terminate
        tracing::debug!("RustCache dropped");
    }
}