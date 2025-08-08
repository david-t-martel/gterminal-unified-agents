//! High-Performance Rust Extensions for Fullstack Agent - PyO3 0.22 Compatible
//!
//! This module provides Rust implementations for performance-critical operations.

pub mod simple_core;
pub mod ttl_cache_enhanced;

// Re-export main types
pub use simple_core::RustCore;
pub use ttl_cache_enhanced::{EnhancedTtlCache, CacheStats};

use pyo3::prelude::*;

/// Initialize the fullstack_agent_rust Python module
#[pymodule]
fn fullstack_agent_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add core functionality
    m.add_class::<RustCore>()?;

    // Add enhanced TTL cache
    m.add_class::<EnhancedTtlCache>()?;
    m.add_class::<CacheStats>()?;

    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__description__",
        "High-performance Rust extensions for fullstack agent (PyO3 0.22)",
    )?;

    Ok(())
}
