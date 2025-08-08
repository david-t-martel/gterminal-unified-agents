//! PyO3 Module Definition for Python Integration
//!
//! This module defines the Python interface for the Rust extensions.

use pyo3::prelude::*;

// Import all the Rust implementations
use crate::auth::{RustAuthValidator, RustTokenManager};
use crate::fileops::{RustFileOps, RustSearchEngine};
use crate::utils::{RustPerformanceMetrics, RustResourceMonitor};

// Import new rust-fs integrated modules (only classes with #[pyclass])
use crate::execution::RustCommandExecutor;
use crate::buffer_pool::RustBufferPool;
use crate::advanced_search::RustAdvancedSearch;
use crate::path_utils::RustPathUtils;

/// Initialize the fullstack_agent_rust Python module
#[pymodule]
pub fn fullstack_agent_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Only add classes that have proper #[pyclass] and #[pymethods] annotations

    // Authentication components (these have #[pyclass] and #[pymethods])
    m.add_class::<RustAuthValidator>()?;
    m.add_class::<RustTokenManager>()?;

    // File operations components (these have #[pyclass] and #[pymethods])
    m.add_class::<RustFileOps>()?;
    m.add_class::<RustSearchEngine>()?;

    // Utility components (these have #[pyclass] and #[pymethods])
    m.add_class::<RustPerformanceMetrics>()?;
    m.add_class::<RustResourceMonitor>()?;

    // Core cache components
    m.add_class::<crate::cache::RustCache>()?;
    m.add_class::<crate::cache::RustCacheManager>()?;

    // Enhanced TTL cache
    m.add_class::<crate::ttl_cache_enhanced::EnhancedTtlCache>()?;
    m.add_class::<crate::ttl_cache_enhanced::CacheStats>()?;

    // JSON processing components
    m.add_class::<crate::json::RustJsonProcessor>()?;
    m.add_class::<crate::json::RustMessagePack>()?;

    // String processing components
    m.add_class::<crate::string_ops::RustStringOps>()?;

    // Async operations components
    m.add_class::<crate::async_ops::RustAsyncOps>()?;
    m.add_class::<crate::async_ops::TaskResult>()?;
    m.add_class::<crate::async_ops::TaskStatus>()?;

    // New rust-fs integrated components - all enabled with proper annotations
    m.add_class::<RustCommandExecutor>()?;
    m.add_class::<RustBufferPool>()?;
    m.add_class::<RustAdvancedSearch>()?;
    m.add_class::<RustPathUtils>()?;

    // Note: ExecResult, SearchResult, SearchMatch, FindResult are data structures
    // that don't need separate class registration as they're returned by methods

    // Fetch components temporarily disabled
    // m.add_class::<crate::fetch::RustFetchClient>()?;
    // m.add_class::<crate::fetch::PyFetchResponse>()?;
    // m.add_class::<crate::fetch::PyResponseMetrics>()?;

    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__description__",
        "High-performance Rust extensions for fullstack agent",
    )?;

    Ok(())
}
