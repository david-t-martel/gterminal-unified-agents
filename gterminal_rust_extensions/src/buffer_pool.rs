//! Buffer pool for efficient memory reuse in file operations
//!
//! This module provides a thread-safe size-class buffer pool to reduce allocations
//! during file read/write operations. Different buffer sizes are used for optimal
//! memory usage based on operation requirements.

use dashmap::DashMap;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use std::sync::Mutex;

/// Size classes for buffers optimized for different use cases
pub const BUFFER_SIZES: &[usize] = &[
    4 * 1024,     // 4KB - Small files, metadata operations
    16 * 1024,    // 16KB - Medium files, text content
    64 * 1024,    // 64KB - Default size, good balance
    256 * 1024,   // 256KB - Large files, binary data
];

/// Default buffer size for file operations (64KB)
pub const DEFAULT_BUFFER_SIZE: usize = 64 * 1024;

/// Maximum number of buffers to keep per size class
const MAX_POOL_SIZE_PER_CLASS: usize = 16;

/// Size-class buffer pools using DashMap for better concurrent access
static BUFFER_POOLS: Lazy<DashMap<usize, Mutex<Vec<Vec<u8>>>>> =
    Lazy::new(|| {
        let pools = DashMap::new();
        for &size in BUFFER_SIZES {
            pools.insert(size, Mutex::new(Vec::new()));
        }
        pools
    });

/// Get the optimal buffer size for the given hint size
fn get_optimal_size(hint_size: usize) -> usize {
    for &size in BUFFER_SIZES {
        if hint_size <= size {
            return size;
        }
    }
    // For very large sizes, use the largest class
    *BUFFER_SIZES.last().unwrap()
}

/// Get a buffer from the pool or create a new one with specified size
pub fn get_buffer_sized(hint_size: usize) -> Vec<u8> {
    let optimal_size = get_optimal_size(hint_size);

    if let Some(pool_mutex) = BUFFER_POOLS.get(&optimal_size) {
        let mut pool = pool_mutex.lock().unwrap();
        if let Some(buffer) = pool.pop() {
            // Increment buffer pool reuses metric if telemetry is available
            return buffer;
        }
    }

    // Increment buffer pool allocations metric if telemetry is available
    Vec::with_capacity(optimal_size)
}

/// Get a buffer from the pool or create a new one (default size)
pub fn get_buffer() -> Vec<u8> {
    get_buffer_sized(DEFAULT_BUFFER_SIZE)
}

/// Return a buffer to the appropriate size-class pool for reuse
pub fn return_buffer(mut buffer: Vec<u8>) {
    buffer.clear();
    let capacity = buffer.capacity();
    let optimal_size = get_optimal_size(capacity);

    // Only return to pool if it matches one of our size classes
    if capacity >= optimal_size * 3 / 4 && capacity <= optimal_size {
        if let Some(pool_mutex) = BUFFER_POOLS.get(&optimal_size) {
            let mut pool = pool_mutex.lock().unwrap();
            if pool.len() < MAX_POOL_SIZE_PER_CLASS {
                pool.push(buffer);
            }
        }
    }
    // Otherwise, let the buffer be dropped
}

/// Get pool statistics for monitoring
pub fn pool_stats() -> Vec<(usize, usize, usize)> {
    let mut stats = Vec::new();
    for &size in BUFFER_SIZES {
        if let Some(pool_mutex) = BUFFER_POOLS.get(&size) {
            let pool = pool_mutex.lock().unwrap();
            stats.push((size, pool.len(), MAX_POOL_SIZE_PER_CLASS));
        }
    }
    stats
}

/// PyO3 wrapper for buffer pool operations
#[pyclass]
pub struct RustBufferPool;

#[pymethods]
impl RustBufferPool {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Get pool statistics
    fn get_stats(&self) -> Vec<(usize, usize, usize)> {
        pool_stats()
    }

    /// Get optimal buffer size for hint
    fn get_optimal_size(&self, hint_size: usize) -> usize {
        get_optimal_size(hint_size)
    }

    /// Clear all pools (for testing/cleanup)
    fn clear_pools(&self) {
        for pool_mutex in BUFFER_POOLS.iter() {
            let mut pool = pool_mutex.value().lock().unwrap();
            pool.clear();
        }
    }
}

/// Helper struct for RAII buffer management
pub struct BufferGuard {
    buffer: Option<Vec<u8>>,
}

impl BufferGuard {
    /// Create a new buffer guard with specified size hint
    pub fn new(hint_size: usize) -> Self {
        Self {
            buffer: Some(get_buffer_sized(hint_size)),
        }
    }

    /// Create a new buffer guard with default size
    pub fn new_default() -> Self {
        Self {
            buffer: Some(get_buffer()),
        }
    }

    /// Get a mutable reference to the buffer
    pub fn buffer(&mut self) -> &mut Vec<u8> {
        self.buffer.as_mut().expect("Buffer already taken")
    }

    /// Take the buffer out of the guard
    pub fn take(mut self) -> Vec<u8> {
        self.buffer.take().expect("Buffer already taken")
    }
}

impl Drop for BufferGuard {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            return_buffer(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_size_selection() {
        assert_eq!(get_optimal_size(1024), 4 * 1024);
        assert_eq!(get_optimal_size(4 * 1024), 4 * 1024);
        assert_eq!(get_optimal_size(5 * 1024), 16 * 1024);
        assert_eq!(get_optimal_size(64 * 1024), 64 * 1024);
        assert_eq!(get_optimal_size(1024 * 1024), 256 * 1024);
    }

    #[test]
    fn test_buffer_pool_operations() {
        // Clear pools for clean test
        let pool = RustBufferPool::new();
        pool.clear_pools();

        // Get initial stats
        let initial_stats = pool_stats();
        assert!(initial_stats.iter().all(|(_, used, _)| *used == 0));

        // Get and return a buffer
        let buffer = get_buffer_sized(4 * 1024);
        assert_eq!(buffer.capacity(), 4 * 1024);
        return_buffer(buffer);

        // Check that pool now has one buffer
        let stats = pool_stats();
        let small_pool_stat = stats.iter().find(|(size, _, _)| *size == 4 * 1024).unwrap();
        assert_eq!(small_pool_stat.1, 1);
    }

    #[test]
    fn test_buffer_guard() {
        let pool = RustBufferPool::new();
        pool.clear_pools();

        {
            let mut guard = BufferGuard::new(16 * 1024);
            let buffer = guard.buffer();
            assert_eq!(buffer.capacity(), 16 * 1024);
            buffer.extend_from_slice(b"test data");
        } // guard dropped here, buffer should be returned to pool

        // Check that pool received the buffer
        let stats = pool_stats();
        let medium_pool_stat = stats.iter().find(|(size, _, _)| *size == 16 * 1024).unwrap();
        assert_eq!(medium_pool_stat.1, 1);
    }
}
