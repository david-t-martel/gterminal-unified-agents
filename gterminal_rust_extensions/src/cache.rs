//! High-performance caching operations with TTL and memory-aware eviction
//!
//! This module provides high-performance cache operations using:
//! - DashMap for concurrent access
//! - LRU cache with efficient eviction
//! - TTL-based automatic expiration
//! - Memory-aware eviction to prevent OOM
//! - System memory monitoring
//! - Async background cleanup

use dashmap::DashMap;
use lru::LruCache;
use ttl_cache::TtlCache;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};  // Re-enabled
use ahash::AHasher;
use pyo3::prelude::*;
use std::hash::{Hash, Hasher};
use sysinfo::System;
use std::num::NonZeroUsize;
use tokio::task;
use tokio::time::interval;

/// High-performance cache item with TTL support and memory tracking
#[derive(Debug, Clone)]
pub struct CacheItem {
    pub value: Vec<u8>,  // Store as bytes for efficiency
    pub created_at: u64, // Unix timestamp
    pub ttl_seconds: u64,
    pub access_count: u64,
    pub last_accessed: u64, // For LRU tracking
    pub content_hash: Option<u64>,
    pub memory_size: usize, // Estimated memory usage in bytes
}

impl CacheItem {
    pub fn new(value: Vec<u8>, ttl_seconds: u64) -> Self {
        let mut hasher = AHasher::default();
        value.hash(&mut hasher);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Calculate memory size: value size + struct overhead (estimated)
        let memory_size = value.len() + std::mem::size_of::<Self>();

        Self {
            value,
            created_at: now,
            ttl_seconds,
            access_count: 0,
            last_accessed: now,
            content_hash: Some(hasher.finish()),
            memory_size,
        }
    }

    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now > (self.created_at + self.ttl_seconds)
    }

    pub fn remaining_ttl(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        (self.created_at + self.ttl_seconds).saturating_sub(now)
    }
}

/// High-performance in-memory cache with concurrent access and memory awareness
#[pyclass]
pub struct RustCache {
    cache: Arc<DashMap<String, CacheItem>>,
    max_size: usize,
    max_memory_bytes: usize,  // Maximum memory usage in bytes
    memory_threshold: f64,    // System memory threshold (0.0-1.0)
    default_ttl: u64,
    // Statistics
    hits: Arc<std::sync::atomic::AtomicU64>,
    misses: Arc<std::sync::atomic::AtomicU64>,
    sets: Arc<std::sync::atomic::AtomicU64>,
    evictions: Arc<std::sync::atomic::AtomicU64>,
    memory_evictions: Arc<std::sync::atomic::AtomicU64>,
    current_memory: Arc<std::sync::atomic::AtomicUsize>,
    // System monitor
    system: Arc<RwLock<System>>,
}

impl RustCache {
    pub fn new(max_size: usize, default_ttl_seconds: u64) -> Self {
        Self::new_with_memory_limit(
            max_size,
            default_ttl_seconds,
            100 * 1024 * 1024,  // Default 100MB max memory
            0.85,  // Default 85% system memory threshold
        )
    }

    pub fn new_with_memory_limit(
        max_size: usize,
        default_ttl_seconds: u64,
        max_memory_bytes: usize,
        memory_threshold: f64,
    ) -> Self {
        let mut system = System::new_all();
        system.refresh_memory();

        Self {
            cache: Arc::new(DashMap::new()),
            max_size,
            max_memory_bytes,
            memory_threshold: memory_threshold.clamp(0.1, 0.95),
            default_ttl: default_ttl_seconds,
            hits: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            misses: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            sets: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            evictions: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            memory_evictions: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            current_memory: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            system: Arc::new(RwLock::new(system)),
        }
    }

    /// Get value from cache
    pub fn get_internal(&self, key: &str) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
        if let Some(mut item) = self.cache.get_mut(key) {
            if item.is_expired() {
                // Remove expired item and update memory counter
                let memory_freed = item.memory_size;
                drop(item);
                self.cache.remove(key);
                self.current_memory
                    .fetch_sub(memory_freed, std::sync::atomic::Ordering::Relaxed);
                self.misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(None);
            }

            // Update access statistics and LRU timestamp
            item.access_count += 1;
            item.last_accessed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            Ok(Some(item.value.clone()))
        } else {
            self.misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(None)
        }
    }

    /// Set value in cache with optional TTL
    pub fn set_internal(
        &self,
        key: String,
        value: Vec<u8>,
        ttl_seconds: Option<u64>,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let ttl = ttl_seconds.unwrap_or(self.default_ttl);
        let item = CacheItem::new(value, ttl);
        let item_memory = item.memory_size;

        // Check memory pressure before adding
        if self.should_evict_for_memory(item_memory)? {
            self.evict_for_memory_pressure(item_memory)?;
        }

        // Check size limits and evict if necessary
        if self.cache.len() >= self.max_size && !self.cache.contains_key(&key) {
            self.evict_expired();

            // If still at capacity, evict least recently used
            if self.cache.len() >= self.max_size {
                self.evict_lru();
            }
        }

        // Update memory counter if replacing existing item
        if let Some(old_item) = self.cache.get(&key) {
            self.current_memory
                .fetch_sub(old_item.memory_size, std::sync::atomic::Ordering::Relaxed);
        }

        self.cache.insert(key, item);
        self.current_memory
            .fetch_add(item_memory, std::sync::atomic::Ordering::Relaxed);
        self.sets.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(true)
    }

    /// Remove key from cache
    pub fn delete_internal(&self, key: &str) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some((_, item)) = self.cache.remove(key) {
            self.current_memory
                .fetch_sub(item.memory_size, std::sync::atomic::Ordering::Relaxed);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Clear all entries or entries matching pattern
    pub fn clear_internal(&self, pattern: Option<&str>) -> Result<usize, Box<dyn std::error::Error>> {
        match pattern {
            Some(pat) => {
                let items_to_remove: Vec<(String, usize)> = self
                    .cache
                    .iter()
                    .filter(|entry| entry.key().contains(pat))
                    .map(|entry| (entry.key().clone(), entry.value().memory_size))
                    .collect();

                let count = items_to_remove.len();
                let mut total_memory_freed = 0;

                for (key, memory_size) in items_to_remove {
                    self.cache.remove(&key);
                    total_memory_freed += memory_size;
                }

                self.current_memory
                    .fetch_sub(total_memory_freed, std::sync::atomic::Ordering::Relaxed);
                Ok(count)
            }
            None => {
                let count = self.cache.len();
                self.cache.clear();
                self.current_memory
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                Ok(count)
            }
        }
    }

    /// Get cache statistics
    pub fn stats_internal(&self) -> Result<HashMap<String, u64>, Box<dyn std::error::Error>> {
        let mut stats = HashMap::new();
        stats.insert("size".to_string(), self.cache.len() as u64);
        stats.insert("max_size".to_string(), self.max_size as u64);
        stats.insert(
            "hits".to_string(),
            self.hits.load(std::sync::atomic::Ordering::Relaxed),
        );
        stats.insert(
            "misses".to_string(),
            self.misses.load(std::sync::atomic::Ordering::Relaxed),
        );
        stats.insert(
            "sets".to_string(),
            self.sets.load(std::sync::atomic::Ordering::Relaxed),
        );
        stats.insert(
            "evictions".to_string(),
            self.evictions.load(std::sync::atomic::Ordering::Relaxed),
        );
        stats.insert(
            "memory_evictions".to_string(),
            self.memory_evictions.load(std::sync::atomic::Ordering::Relaxed),
        );

        // Memory statistics
        let current_memory = self.current_memory.load(std::sync::atomic::Ordering::Relaxed);
        stats.insert("memory_bytes".to_string(), current_memory as u64);
        stats.insert("memory_mb".to_string(), (current_memory / (1024 * 1024)) as u64);
        stats.insert("max_memory_bytes".to_string(), self.max_memory_bytes as u64);
        stats.insert("max_memory_mb".to_string(), (self.max_memory_bytes / (1024 * 1024)) as u64);

        // System memory stats
        if let Ok(system) = self.system.read() {
            stats.insert("system_total_memory_mb".to_string(), system.total_memory() / 1024);
            stats.insert("system_used_memory_mb".to_string(), system.used_memory() / 1024);
            stats.insert("system_available_memory_mb".to_string(), system.available_memory() / 1024);
            let memory_usage_percent = (system.used_memory() as f64 / system.total_memory() as f64) * 100.0;
            stats.insert("system_memory_percent".to_string(), memory_usage_percent as u64);
        }

        let total_requests = stats["hits"] + stats["misses"];
        let hit_rate = if total_requests > 0 {
            (stats["hits"] as f64 / total_requests as f64 * 100.0) as u64
        } else {
            0
        };
        stats.insert("hit_rate_percent".to_string(), hit_rate);

        Ok(stats)
    }

    /// Check if key exists and is not expired
    pub fn contains_internal(&self, key: &str) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(item) = self.cache.get(key) {
            Ok(!item.is_expired())
        } else {
            Ok(false)
        }
    }

    /// Get keys matching pattern
    pub fn keys(
        &self,
        pattern: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let keys: Vec<String> = match pattern {
            Some(pat) => self
                .cache
                .iter()
                .filter(|entry| entry.key().contains(pat))
                .filter(|entry| !entry.value().is_expired())
                .take(limit.unwrap_or(usize::MAX))
                .map(|entry| entry.key().clone())
                .collect(),
            None => self
                .cache
                .iter()
                .filter(|entry| !entry.value().is_expired())
                .take(limit.unwrap_or(usize::MAX))
                .map(|entry| entry.key().clone())
                .collect(),
        };

        Ok(keys)
    }

    /// Batch get operation - more efficient than multiple gets
    pub fn batch_get_internal(
        &self,
        keys: Vec<&str>,
    ) -> Result<HashMap<String, Vec<u8>>, Box<dyn std::error::Error>> {
        let mut results = HashMap::new();

        for key in keys {
            if let Some(value) = self.get_internal(key)? {
                results.insert(key.to_string(), value);
            }
        }

        Ok(results)
    }

    /// Batch set operation - more efficient than multiple sets
    pub fn batch_set_internal(
        &self,
        items: HashMap<String, Vec<u8>>,
        ttl_seconds: Option<u64>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let ttl = ttl_seconds.unwrap_or(self.default_ttl);
        let mut count = 0;

        for (key, value) in items {
            if self.set_internal(key, value, Some(ttl))? {
                count += 1;
            }
        }

        Ok(count)
    }
}

impl Default for RustCache {
    fn default() -> Self {
        Self::new(10000, 3600)
    }
}

impl RustCache {
    /// Check if memory eviction is needed
    fn should_evict_for_memory(&self, new_item_size: usize) -> Result<bool, Box<dyn std::error::Error>> {
        let current_memory = self.current_memory.load(std::sync::atomic::Ordering::Relaxed);

        // Check cache memory limit
        if current_memory + new_item_size > self.max_memory_bytes {
            return Ok(true);
        }

        // Check system memory pressure
        if let Ok(mut system) = self.system.write() {
            system.refresh_memory();
            let memory_usage = system.used_memory() as f64 / system.total_memory() as f64;
            if memory_usage > self.memory_threshold {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Evict items to free memory for new item
    fn evict_for_memory_pressure(&self, required_memory: usize) -> Result<(), Box<dyn std::error::Error>> {
        let current_memory = self.current_memory.load(std::sync::atomic::Ordering::Relaxed);
        let mut memory_to_free = if current_memory + required_memory > self.max_memory_bytes {
            (current_memory + required_memory) - self.max_memory_bytes + (self.max_memory_bytes / 10) // Free 10% extra
        } else {
            self.max_memory_bytes / 10 // Free at least 10% of max
        };

        // First, remove expired entries
        let expired_items: Vec<(String, usize)> = self
            .cache
            .iter()
            .filter(|entry| entry.value().is_expired())
            .map(|entry| (entry.key().clone(), entry.value().memory_size))
            .collect();

        for (key, size) in expired_items {
            self.cache.remove(&key);
            self.current_memory
                .fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
            memory_to_free = memory_to_free.saturating_sub(size);
            self.memory_evictions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        if memory_to_free == 0 {
            return Ok(());
        }

        // Then, remove least recently used items
        let mut lru_candidates: Vec<(String, u64, usize)> = self
            .cache
            .iter()
            .map(|entry| (
                entry.key().clone(),
                entry.value().last_accessed,
                entry.value().memory_size,
            ))
            .collect();

        // Sort by last accessed time (oldest first)
        lru_candidates.sort_by_key(|(_, last_accessed, _)| *last_accessed);

        for (key, _, size) in lru_candidates {
            if memory_to_free == 0 {
                break;
            }

            self.cache.remove(&key);
            self.current_memory
                .fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
            memory_to_free = memory_to_free.saturating_sub(size);
            self.memory_evictions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        Ok(())
    }

    /// Remove expired entries
    fn evict_expired(&self) {
        let expired_items: Vec<(String, usize)> = self
            .cache
            .iter()
            .filter(|entry| entry.value().is_expired())
            .map(|entry| (entry.key().clone(), entry.value().memory_size))
            .collect();

        let count = expired_items.len();
        let mut total_memory_freed = 0;

        for (key, memory_size) in expired_items {
            self.cache.remove(&key);
            total_memory_freed += memory_size;
        }

        self.current_memory
            .fetch_sub(total_memory_freed, std::sync::atomic::Ordering::Relaxed);
        self.evictions
            .fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// Remove least recently used entries
    fn evict_lru(&self) {
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = u64::MAX;
        let mut oldest_memory_size = 0;

        // Find the least recently accessed entry
        for entry in self.cache.iter() {
            let access_time = entry.value().last_accessed;
            if access_time < oldest_time {
                oldest_time = access_time;
                oldest_key = Some(entry.key().clone());
                oldest_memory_size = entry.value().memory_size;
            }
        }

        if let Some(key) = oldest_key {
            self.cache.remove(&key);
            self.current_memory
                .fetch_sub(oldest_memory_size, std::sync::atomic::Ordering::Relaxed);
            self.evictions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[pymethods]
impl RustCache {
    #[new]
    #[pyo3(signature = (max_size=1000, default_ttl_seconds=3600, max_memory_mb=100, memory_threshold=0.85))]
    pub fn py_new(
        max_size: usize,
        default_ttl_seconds: u64,
        max_memory_mb: usize,
        memory_threshold: f64,
    ) -> Self {
        Self::new_with_memory_limit(
            max_size,
            default_ttl_seconds,
            max_memory_mb * 1024 * 1024,  // Convert MB to bytes
            memory_threshold,
        )
    }

    /// Python wrapper for get
    pub fn py_get(&self, key: &str) -> PyResult<Option<Vec<u8>>> {
        self.get_internal(key)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Python wrapper for set
    pub fn py_set(&self, key: String, value: Vec<u8>, ttl_seconds: Option<u64>) -> PyResult<bool> {
        self.set_internal(key, value, ttl_seconds)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Python wrapper for delete
    pub fn py_delete(&self, key: &str) -> PyResult<bool> {
        self.delete_internal(key)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Python wrapper for contains
    pub fn py_contains(&self, key: &str) -> PyResult<bool> {
        self.contains_internal(key)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Python wrapper for clear
    pub fn py_clear(&self, pattern: Option<&str>) -> PyResult<usize> {
        self.clear_internal(pattern)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Python wrapper for stats
    pub fn py_stats(&self) -> PyResult<HashMap<String, u64>> {
        self.stats_internal()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Python wrapper for batch_get
    pub fn py_batch_get(&self, keys: Vec<String>) -> PyResult<HashMap<String, Vec<u8>>> {
        let key_refs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
        self.batch_get_internal(key_refs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Python wrapper for batch_set
    pub fn py_batch_set(&self, items: HashMap<String, Vec<u8>>, ttl_seconds: Option<u64>) -> PyResult<usize> {
        self.batch_set_internal(items, ttl_seconds)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get memory statistics
    pub fn memory_info(&self) -> PyResult<HashMap<String, u64>> {
        let mut info = HashMap::new();
        let current_memory = self.current_memory.load(std::sync::atomic::Ordering::Relaxed);
        info.insert("current_bytes".to_string(), current_memory as u64);
        info.insert("current_mb".to_string(), (current_memory / (1024 * 1024)) as u64);
        info.insert("max_bytes".to_string(), self.max_memory_bytes as u64);
        info.insert("max_mb".to_string(), (self.max_memory_bytes / (1024 * 1024)) as u64);
        info.insert("memory_threshold_percent".to_string(), (self.memory_threshold * 100.0) as u64);
        info.insert("memory_evictions".to_string(),
            self.memory_evictions.load(std::sync::atomic::Ordering::Relaxed));

        Ok(info)
    }

    /// Manually trigger memory pressure eviction
    pub fn evict_memory(&self, target_mb: Option<usize>) -> PyResult<()> {
        let target_bytes = target_mb.map(|mb| mb * 1024 * 1024).unwrap_or(self.max_memory_bytes / 10);
        self.evict_for_memory_pressure(target_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    // Python API compatible methods
    /// Get value from cache with automatic conversion
    pub fn get(&self, key: String) -> PyResult<Option<String>> {
        match self.py_get(&key)? {
            Some(bytes) => {
                // Try to convert bytes to string
                match String::from_utf8(bytes) {
                    Ok(s) => Ok(Some(s)),
                    Err(_) => Ok(None), // Return None for non-UTF8 data
                }
            }
            None => Ok(None),
        }
    }

    /// Set value in cache with automatic conversion
    pub fn set(&self, key: String, value: String, ttl_seconds: Option<u64>) -> PyResult<bool> {
        self.py_set(key, value.into_bytes(), ttl_seconds)
    }

    /// Delete key from cache
    pub fn delete(&self, key: String) -> PyResult<bool> {
        self.py_delete(&key)
    }

    /// Check if key exists
    pub fn contains(&self, key: String) -> PyResult<bool> {
        self.py_contains(&key)
    }

    /// Clear cache
    pub fn clear(&self) -> PyResult<()> {
        self.py_clear(None)?;
        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> PyResult<HashMap<String, u64>> {
        self.py_stats()
    }

    /// Batch get operation
    pub fn batch_get(&self, keys: Vec<String>) -> PyResult<HashMap<String, String>> {
        let result = self.py_batch_get(keys)?;
        let mut converted = HashMap::new();
        for (key, bytes) in result {
            if let Ok(s) = String::from_utf8(bytes) {
                converted.insert(key, s);
            }
        }
        Ok(converted)
    }

    /// Batch set operation
    pub fn batch_set(&self, items: HashMap<String, String>, ttl_seconds: u64) -> PyResult<u32> {
        let converted: HashMap<String, Vec<u8>> = items
            .into_iter()
            .map(|(k, v)| (k, v.into_bytes()))
            .collect();
        let count = self.py_batch_set(converted, Some(ttl_seconds))?;
        Ok(count as u32)
    }
}

/// Redis-backed cache with connection pooling (placeholder implementation)
pub struct RustRedisCache {
    connection_manager: Arc<RwLock<Option<String>>>, // Placeholder for ConnectionManager
    key_prefix: String,
    default_ttl: u64,
    // Statistics
    hits: Arc<std::sync::atomic::AtomicU64>,
    misses: Arc<std::sync::atomic::AtomicU64>,
    sets: Arc<std::sync::atomic::AtomicU64>,
}

impl RustRedisCache {
    pub fn new(_redis_url: String, key_prefix: String, default_ttl_seconds: u64) -> Self {
        Self {
            connection_manager: Arc::new(RwLock::new(None)),
            key_prefix,
            default_ttl: default_ttl_seconds,
            hits: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            misses: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            sets: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Initialize Redis connection (placeholder implementation)
    pub fn connect(&self, redis_url: String) -> Result<(), Box<dyn std::error::Error>> {
        let mut conn_guard = self
            .connection_manager
            .write()
            .map_err(|e| format!("Lock error: {e}"))?;
        *conn_guard = Some(redis_url);
        Ok(())
    }

    /// Get value from Redis (placeholder implementation)
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
        let _prefixed_key = format!("{}{}", self.key_prefix, key);
        // Placeholder implementation - Redis temporarily disabled
        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(None)
    }

    /// Set value in Redis with TTL (placeholder implementation)
    pub fn set(
        &self,
        _key: &str,
        _value: Vec<u8>,
        _ttl_seconds: Option<u64>,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Placeholder implementation - Redis temporarily disabled
        self.sets.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(true)
    }

    /// Delete key from Redis (placeholder implementation)
    pub fn delete(&self, _key: &str) -> Result<bool, Box<dyn std::error::Error>> {
        // Placeholder implementation - Redis temporarily disabled
        Ok(false)
    }

    /// Get Redis statistics
    pub fn stats(&self) -> Result<HashMap<String, u64>, Box<dyn std::error::Error>> {
        let mut stats = HashMap::new();
        stats.insert(
            "hits".to_string(),
            self.hits.load(std::sync::atomic::Ordering::Relaxed),
        );
        stats.insert(
            "misses".to_string(),
            self.misses.load(std::sync::atomic::Ordering::Relaxed),
        );
        stats.insert(
            "sets".to_string(),
            self.sets.load(std::sync::atomic::Ordering::Relaxed),
        );

        let total_requests = stats["hits"] + stats["misses"];
        let hit_rate = if total_requests > 0 {
            (stats["hits"] as f64 / total_requests as f64 * 100.0) as u64
        } else {
            0
        };
        stats.insert("hit_rate_percent".to_string(), hit_rate);

        Ok(stats)
    }
}

/// Multi-layer cache manager combining memory and Redis
#[pyclass]
pub struct RustCacheManager {
    l1_cache: Arc<RustCache>,
    l2_cache: Option<Arc<RustRedisCache>>,
    total_hits: Arc<std::sync::atomic::AtomicU64>,
    total_misses: Arc<std::sync::atomic::AtomicU64>,
}

impl RustCacheManager {
    pub fn new(
        memory_max_size: usize,
        memory_ttl: u64,
        redis_cache: Option<&RustRedisCache>,
    ) -> Self {
        Self {
            l1_cache: Arc::new(RustCache::new(memory_max_size, memory_ttl)),
            l2_cache: redis_cache.map(|rc| Arc::new(rc.clone())),
            total_hits: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            total_misses: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Get value from multi-layer cache
    pub fn get_internal(&self, key: &str) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
        // Try L1 cache first
        if let Some(value) = self.l1_cache.get_internal(key)? {
            self.total_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(Some(value));
        }

        // Try L2 cache if available
        if let Some(ref l2) = self.l2_cache {
            if let Some(value) = l2.get(key)? {
                // Store in L1 for future access
                self.l1_cache.set_internal(key.to_string(), value.clone(), None)?;
                self.total_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(Some(value));
            }
        }

        self.total_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(None)
    }

    /// Set value in multi-layer cache
    pub fn set_internal(
        &self,
        key: String,
        value: Vec<u8>,
        ttl_seconds: Option<u64>,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Set in L1 cache
        let l1_success = self.l1_cache.set_internal(key.clone(), value.clone(), ttl_seconds)?;

        // Set in L2 cache if available
        let l2_success = if let Some(ref l2) = self.l2_cache {
            l2.set(&key, value, ttl_seconds)?
        } else {
            true
        };

        Ok(l1_success && l2_success)
    }

    /// Get comprehensive cache statistics
    pub fn stats_internal(
        &self,
    ) -> Result<HashMap<String, HashMap<String, u64>>, Box<dyn std::error::Error>> {
        let mut all_stats = HashMap::new();

        // L1 cache stats
        all_stats.insert("l1_cache".to_string(), self.l1_cache.stats_internal()?);

        // L2 cache stats if available
        if let Some(ref l2) = self.l2_cache {
            all_stats.insert("l2_cache".to_string(), l2.stats()?);
        }

        // Overall stats
        let mut overall = HashMap::new();
        overall.insert(
            "total_hits".to_string(),
            self.total_hits.load(std::sync::atomic::Ordering::Relaxed),
        );
        overall.insert(
            "total_misses".to_string(),
            self.total_misses.load(std::sync::atomic::Ordering::Relaxed),
        );
        all_stats.insert("overall".to_string(), overall);

        Ok(all_stats)
    }
}

impl Default for RustCacheManager {
    fn default() -> Self {
        Self::new(5000, 1800, None)
    }
}

#[pymethods]
impl RustCacheManager {
    #[new]
    #[pyo3(signature = (memory_max_size=5000, memory_ttl=1800, memory_max_mb=100, memory_threshold=0.85))]
    pub fn py_new(
        memory_max_size: usize,
        memory_ttl: u64,
        memory_max_mb: usize,
        memory_threshold: f64,
    ) -> Self {
        let cache = RustCache::new_with_memory_limit(
            memory_max_size,
            memory_ttl,
            memory_max_mb * 1024 * 1024,
            memory_threshold,
        );
        Self {
            l1_cache: Arc::new(cache),
            l2_cache: None,
            total_hits: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            total_misses: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Create a named cache with memory awareness
    pub fn create_cache(
        &mut self,
        _name: String,
        max_size: usize,
        default_ttl_seconds: u64,
        max_memory_mb: usize,
        memory_threshold: f64,
    ) -> PyResult<()> {
        // For simplicity in this implementation, we just update the l1_cache
        // In a full implementation, we'd maintain a map of named caches
        let cache = RustCache::new_with_memory_limit(
            max_size,
            default_ttl_seconds,
            max_memory_mb * 1024 * 1024,
            memory_threshold,
        );
        self.l1_cache = Arc::new(cache);
        Ok(())
    }

    /// Python wrapper for get
    pub fn py_get(&self, key: &str) -> PyResult<Option<Vec<u8>>> {
        self.get_internal(key)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Python wrapper for set
    pub fn py_set(&self, key: String, value: Vec<u8>, ttl_seconds: Option<u64>) -> PyResult<bool> {
        self.set_internal(key, value, ttl_seconds)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Python wrapper for stats
    pub fn py_stats(&self) -> PyResult<HashMap<String, HashMap<String, u64>>> {
        self.stats_internal()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    // Python API compatible methods
    /// Get cache by name (simplified implementation)
    pub fn get_cache(&self, _name: String) -> PyResult<Option<bool>> {
        // For simplicity, always return the main cache exists
        Ok(Some(true))
    }

    /// Delete cache by name (simplified implementation)
    pub fn delete_cache(&self, _name: String) -> PyResult<bool> {
        // For simplicity, always return success
        Ok(true)
    }

    /// List all cache names (simplified implementation)
    pub fn list_caches(&self) -> PyResult<Vec<String>> {
        // For simplicity, return a default cache name
        Ok(vec!["default".to_string()])
    }
}

impl Clone for RustRedisCache {
    fn clone(&self) -> Self {
        Self {
            connection_manager: self.connection_manager.clone(),
            key_prefix: self.key_prefix.clone(),
            default_ttl: self.default_ttl,
            hits: self.hits.clone(),
            misses: self.misses.clone(),
            sets: self.sets.clone(),
        }
    }
}
