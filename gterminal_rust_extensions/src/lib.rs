//! GTerminal Rust Extensions
//! High-performance Rust components for the GTerminal ReAct engine
//!
//! This library provides four main components:
//! - RustFileOps: Parallel file operations and watching
//! - RustCache: High-performance concurrent cache with TTL and LRU
//! - RustJsonProcessor: Fast JSON processing with validation and queries
//! - RustCommandExecutor: Secure process management and execution

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod cache;
pub mod command_executor;
pub mod file_ops;
pub mod json_processor;
pub mod utils;

use cache::RustCache;
use command_executor::RustCommandExecutor;
use file_ops::RustFileOps;
use json_processor::RustJsonProcessor;

/// Initialize tracing for debugging
#[pyfunction]
fn init_tracing(level: Option<&str>) -> PyResult<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    
    let level = level.unwrap_or("info");
    let filter = tracing_subscriber::EnvFilter::new(format!("gterminal_rust_extensions={}", level));
    
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(filter)
        .try_init()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to initialize tracing: {}", e)))?;
    
    Ok(())
}

/// Get version information
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get build information
#[pyfunction]
fn build_info() -> std::collections::HashMap<String, String> {
    let mut info = std::collections::HashMap::new();
    info.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
    info.insert("target".to_string(), env!("TARGET").to_string());
    info.insert("profile".to_string(), if cfg!(debug_assertions) { "debug".to_string() } else { "release".to_string() });
    info.insert("features".to_string(), "tokio,dashmap,serde,pyo3".to_string());
    info
}

/// Performance benchmark function
#[pyfunction]
fn benchmark_components(iterations: usize) -> PyResult<std::collections::HashMap<String, f64>> {
    let mut results = std::collections::HashMap::new();
    
    // Benchmark cache operations
    let start = std::time::Instant::now();
    let cache = RustCache::new(1000, Some(3600))?;
    for i in 0..iterations {
        cache.set(format!("key_{}", i), format!("value_{}", i).into())?;
    }
    for i in 0..iterations {
        let _ = cache.get(&format!("key_{}", i))?;
    }
    results.insert("cache_ops_per_sec".to_string(), iterations as f64 / start.elapsed().as_secs_f64());
    
    // Benchmark JSON processing
    let start = std::time::Instant::now();
    let processor = RustJsonProcessor::new()?;
    let test_json = serde_json::json!({
        "test": "data",
        "number": 42,
        "array": [1, 2, 3, 4, 5],
        "nested": {"key": "value"}
    }).to_string();
    
    for _ in 0..iterations {
        let _ = processor.parse(&test_json)?;
    }
    results.insert("json_parse_ops_per_sec".to_string(), iterations as f64 / start.elapsed().as_secs_f64());
    
    Ok(results)
}

/// Python module definition
#[pymodule]
fn gterminal_rust_extensions(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add classes
    m.add_class::<RustFileOps>()?;
    m.add_class::<RustCache>()?;
    m.add_class::<RustJsonProcessor>()?;
    m.add_class::<RustCommandExecutor>()?;
    
    // Add utility functions
    m.add_function(wrap_pyfunction!(init_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_components, m)?)?;
    
    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "GTerminal Team")?;
    m.add("__description__", "High-performance Rust extensions for GTerminal ReAct engine")?;
    
    Ok(())
}