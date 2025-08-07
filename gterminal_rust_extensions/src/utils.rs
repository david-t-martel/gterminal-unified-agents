//! Utility functions and types shared across components

use anyhow::{Context, Result};
use pyo3::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Global operation counter for performance tracking
static OPERATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Increment and return operation count
pub fn increment_ops() -> u64 {
    OPERATION_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Get current operation count
pub fn get_ops_count() -> u64 {
    OPERATION_COUNTER.load(Ordering::Relaxed)
}

/// Get current Unix timestamp in seconds
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Convert Python path to Rust PathBuf with validation
pub fn validate_path(path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    
    // Basic validation
    if path.to_string_lossy().is_empty() {
        return Err(anyhow::anyhow!("Path cannot be empty"));
    }
    
    // Convert to absolute path if relative
    let abs_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("Failed to get current directory")?
            .join(path)
    };
    
    // Canonicalize to resolve symlinks and normalize
    abs_path.canonicalize()
        .or_else(|_| Ok(abs_path)) // If canonicalize fails, return as-is
        .context("Path validation failed")
}

/// Safe file size check before operations
pub fn check_file_size(path: &Path, max_size: Option<u64>) -> Result<u64> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("Failed to get metadata for: {}", path.display()))?;
    
    let size = metadata.len();
    
    if let Some(max) = max_size {
        if size > max {
            return Err(anyhow::anyhow!(
                "File too large: {} bytes (max: {} bytes)", 
                size, max
            ));
        }
    }
    
    Ok(size)
}

/// Convert Duration to Python-compatible milliseconds
pub fn duration_to_millis(duration: Duration) -> u64 {
    duration.as_millis() as u64
}

/// Convert milliseconds to Duration
pub fn millis_to_duration(millis: u64) -> Duration {
    Duration::from_millis(millis)
}

/// Async-safe thread pool for CPU-intensive tasks
pub struct ThreadPool {
    pool: tokio::runtime::Runtime,
}

impl ThreadPool {
    pub fn new(threads: Option<usize>) -> Result<Self> {
        let threads = threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
        
        let pool = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            .thread_name("gterminal-worker")
            .enable_all()
            .build()
            .context("Failed to create thread pool")?;
        
        Ok(Self { pool })
    }
    
    pub fn spawn<F, T>(&self, future: F) -> tokio::task::JoinHandle<T>
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.pool.spawn(future)
    }
    
    pub fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        self.pool.block_on(future)
    }
}

/// Rate limiter for API calls and operations
#[derive(Debug)]
pub struct RateLimiter {
    max_requests: u32,
    window: Duration,
    requests: std::sync::Arc<parking_lot::Mutex<std::collections::VecDeque<std::time::Instant>>>,
}

impl RateLimiter {
    pub fn new(max_requests: u32, window_secs: u64) -> Self {
        Self {
            max_requests,
            window: Duration::from_secs(window_secs),
            requests: std::sync::Arc::new(parking_lot::Mutex::new(std::collections::VecDeque::new())),
        }
    }
    
    pub fn check_rate_limit(&self) -> bool {
        let now = std::time::Instant::now();
        let mut requests = self.requests.lock();
        
        // Remove old requests outside the window
        while let Some(&front) = requests.front() {
            if now.duration_since(front) > self.window {
                requests.pop_front();
            } else {
                break;
            }
        }
        
        // Check if we can make a new request
        if requests.len() < self.max_requests as usize {
            requests.push_back(now);
            true
        } else {
            false
        }
    }
    
    pub async fn wait_for_slot(&self) -> Result<()> {
        const MAX_WAIT: Duration = Duration::from_secs(60);
        const CHECK_INTERVAL: Duration = Duration::from_millis(100);
        
        let start = std::time::Instant::now();
        
        while !self.check_rate_limit() {
            if start.elapsed() > MAX_WAIT {
                return Err(anyhow::anyhow!("Rate limit wait timeout"));
            }
            tokio::time::sleep(CHECK_INTERVAL).await;
        }
        
        Ok(())
    }
}

/// Error handling utilities
pub trait ResultExt<T> {
    fn to_py_err(self) -> PyResult<T>;
}

impl<T> ResultExt<T> for Result<T> {
    fn to_py_err(self) -> PyResult<T> {
        self.map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }
}

/// Memory usage monitoring
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_allocated: u64,
    pub peak_allocated: u64,
    pub current_allocated: u64,
}

static MEMORY_TRACKER: std::sync::OnceLock<std::sync::Arc<parking_lot::Mutex<MemoryInfo>>> = 
    std::sync::OnceLock::new();

pub fn init_memory_tracking() {
    MEMORY_TRACKER.set(std::sync::Arc::new(parking_lot::Mutex::new(MemoryInfo {
        total_allocated: 0,
        peak_allocated: 0,
        current_allocated: 0,
    }))).ok();
}

pub fn track_allocation(size: u64) {
    if let Some(tracker) = MEMORY_TRACKER.get() {
        let mut info = tracker.lock();
        info.total_allocated += size;
        info.current_allocated += size;
        if info.current_allocated > info.peak_allocated {
            info.peak_allocated = info.current_allocated;
        }
    }
}

pub fn track_deallocation(size: u64) {
    if let Some(tracker) = MEMORY_TRACKER.get() {
        let mut info = tracker.lock();
        info.current_allocated = info.current_allocated.saturating_sub(size);
    }
}

pub fn get_memory_info() -> MemoryInfo {
    MEMORY_TRACKER
        .get()
        .map(|t| t.lock().clone())
        .unwrap_or_default()
}

impl Default for MemoryInfo {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            peak_allocated: 0,
            current_allocated: 0,
        }
    }
}

/// Configuration validation
pub fn validate_config<T: serde::de::DeserializeOwned>(
    config_str: &str,
    schema: Option<&str>,
) -> Result<T> {
    // Parse the configuration
    let config: T = serde_json::from_str(config_str)
        .context("Failed to parse configuration JSON")?;
    
    // Validate against schema if provided
    if let Some(schema_str) = schema {
        let schema = jsonschema::JSONSchema::compile(
            &serde_json::from_str::<serde_json::Value>(schema_str)?
        ).context("Failed to compile JSON schema")?;
        
        let instance = serde_json::from_str::<serde_json::Value>(config_str)?;
        
        if let Err(errors) = schema.validate(&instance) {
            let error_messages: Vec<String> = errors
                .map(|e| e.to_string())
                .collect();
            return Err(anyhow::anyhow!(
                "Configuration validation failed: {}", 
                error_messages.join(", ")
            ));
        }
    }
    
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_path_validation() {
        assert!(validate_path("/tmp").is_ok());
        assert!(validate_path("").is_err());
    }
    
    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(2, 1);
        assert!(limiter.check_rate_limit());
        assert!(limiter.check_rate_limit());
        assert!(!limiter.check_rate_limit()); // Should be rate limited
    }
    
    #[tokio::test]
    async fn test_thread_pool() {
        let pool = ThreadPool::new(Some(2)).unwrap();
        let handle = pool.spawn(async { 42 });
        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }
}