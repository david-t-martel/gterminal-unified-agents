//! Simple Core Module - Minimal working Rust extensions for PyO3 0.22
//!
//! This provides basic functionality to ensure the build works before adding more features.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Simple core functionality
#[pyclass]
pub struct RustCore {
    version: String,
}

#[pymethods]
impl RustCore {
    #[new]
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Get version information
    #[getter]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Simple test function
    pub fn test_rust_integration(&self) -> String {
        "Rust extensions are working!".to_string()
    }

    /// Basic string operations
    pub fn reverse_string(&self, input: &str) -> String {
        input.chars().rev().collect()
    }

    /// Basic math operations
    pub fn add_numbers(&self, a: f64, b: f64) -> f64 {
        a + b
    }

    /// Work with Python dictionary
    pub fn process_dict(&self, data: HashMap<String, i32>) -> HashMap<String, i32> {
        data.into_iter()
            .map(|(k, v)| (k, v * 2))
            .collect()
    }
}

impl Default for RustCore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_core() {
        let core = RustCore::new();
        assert!(!core.version().is_empty());
        assert_eq!(core.test_rust_integration(), "Rust extensions are working!");
        assert_eq!(core.reverse_string("hello"), "olleh");
        assert_eq!(core.add_numbers(2.0, 3.0), 5.0);
    }

    #[test]
    fn test_dict_processing() {
        let core = RustCore::new();
        let mut input = HashMap::new();
        input.insert("a".to_string(), 1);
        input.insert("b".to_string(), 2);

        let result = core.process_dict(input);
        assert_eq!(result.get("a"), Some(&2));
        assert_eq!(result.get("b"), Some(&4));
    }
}
