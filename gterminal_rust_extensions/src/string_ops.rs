//! High-Performance String Processing Module
//!
//! This module provides Rust-based string operations with Python integration.
//! Focuses on regex operations, pattern matching, and string transformations.

use pyo3::prelude::*;
use regex::{Regex, RegexBuilder};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use globset::{Glob, GlobSet, GlobSetBuilder};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;

// Global regex cache for compiled patterns
static REGEX_CACHE: Lazy<Arc<Mutex<HashMap<String, Regex>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

/// High-performance string operations with regex caching
#[pyclass]
pub struct RustStringOps {
    // Multi-pattern matcher for efficient string searches
    aho_corasick: Option<AhoCorasick>,
    // Glob pattern matcher
    glob_set: Option<GlobSet>,
}

#[pymethods]
impl RustStringOps {
    #[new]
    pub fn new() -> Self {
        Self {
            aho_corasick: None,
            glob_set: None,
        }
    }

    /// Compile and cache a regex pattern for reuse
    pub fn compile_regex(&self, pattern: &str, case_insensitive: Option<bool>) -> PyResult<bool> {
        let mut cache = REGEX_CACHE.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Regex cache lock error: {}", e))
        })?;

        // Check if pattern is already cached
        let cache_key = if case_insensitive.unwrap_or(false) {
            format!("i:{}", pattern)
        } else {
            pattern.to_string()
        };

        if cache.contains_key(&cache_key) {
            return Ok(false); // Already cached
        }

        // Compile regex with options
        let regex = if case_insensitive.unwrap_or(false) {
            RegexBuilder::new(pattern)
                .case_insensitive(true)
                .build()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Regex compilation error: {}", e)))?
        } else {
            Regex::new(pattern)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Regex compilation error: {}", e)))?
        };

        cache.insert(cache_key, regex);
        Ok(true) // Newly compiled and cached
    }

    /// Find all matches of a regex pattern in text (uses cached patterns)
    pub fn regex_find_all(&self, pattern: &str, text: &str, case_insensitive: Option<bool>) -> PyResult<Vec<(usize, usize, String)>> {
        let cache_key = if case_insensitive.unwrap_or(false) {
            format!("i:{}", pattern)
        } else {
            pattern.to_string()
        };

        // First, try to get from cache
        let regex_result = {
            let cache = REGEX_CACHE.lock().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Regex cache lock error: {}", e))
            })?;
            cache.get(&cache_key).cloned()
        };

        // If not cached, compile and cache it
        let regex = match regex_result {
            Some(regex) => regex,
            None => {
                // Compile and cache the regex
                self.compile_regex(pattern, case_insensitive)?;

                // Get it from cache again
                let cache = REGEX_CACHE.lock().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Regex cache lock error: {}", e))
                })?;

                cache.get(&cache_key)
                    .cloned()
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to get or compile regex"))?
            }
        };

        let matches: Vec<(usize, usize, String)> = regex
            .find_iter(text)
            .map(|m| (m.start(), m.end(), m.as_str().to_string()))
            .collect();

        Ok(matches)
    }

    /// Replace all matches of a regex pattern with replacement text
    pub fn regex_replace_all(&self, pattern: &str, text: &str, replacement: &str, case_insensitive: Option<bool>) -> PyResult<String> {
        let cache_key = if case_insensitive.unwrap_or(false) {
            format!("i:{}", pattern)
        } else {
            pattern.to_string()
        };

        // First, try to get from cache
        let regex_result = {
            let cache = REGEX_CACHE.lock().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Regex cache lock error: {}", e))
            })?;
            cache.get(&cache_key).cloned()
        };

        // If not cached, compile and cache it
        let regex = match regex_result {
            Some(regex) => regex,
            None => {
                // Compile and cache the regex
                self.compile_regex(pattern, case_insensitive)?;

                // Get it from cache again
                let cache = REGEX_CACHE.lock().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Regex cache lock error: {}", e))
                })?;

                cache.get(&cache_key)
                    .cloned()
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to get or compile regex"))?
            }
        };

        Ok(regex.replace_all(text, replacement).to_string())
    }

    /// Set up multi-pattern matching with Aho-Corasick algorithm
    pub fn setup_multi_pattern(&mut self, patterns: Vec<String>, case_insensitive: Option<bool>) -> PyResult<()> {
        let ac = if case_insensitive.unwrap_or(false) {
            AhoCorasickBuilder::new()
                .ascii_case_insensitive(true)
                .build(patterns)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Aho-Corasick build error: {}", e)))?
        } else {
            AhoCorasick::new(patterns)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Aho-Corasick build error: {}", e)))?
        };

        self.aho_corasick = Some(ac);
        Ok(())
    }

    /// Find all matches using multi-pattern matching (very fast for many patterns)
    pub fn multi_pattern_find_all(&self, text: &str) -> PyResult<Vec<(usize, usize, usize, String)>> {
        let ac = self.aho_corasick.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Multi-pattern matcher not initialized. Call setup_multi_pattern first."))?;

        let matches: Vec<(usize, usize, usize, String)> = ac
            .find_iter(text)
            .map(|m| (m.start(), m.end(), m.pattern().as_usize(), text[m.start()..m.end()].to_string()))
            .collect();

        Ok(matches)
    }

    /// Set up glob pattern matching
    pub fn setup_glob_patterns(&mut self, patterns: Vec<String>) -> PyResult<()> {
        let mut builder = GlobSetBuilder::new();

        for pattern in patterns {
            let glob = Glob::new(&pattern)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid glob pattern '{}': {}", pattern, e)))?;
            builder.add(glob);
        }

        let glob_set = builder.build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Glob set build error: {}", e)))?;

        self.glob_set = Some(glob_set);
        Ok(())
    }

    /// Check if a string matches any of the configured glob patterns
    pub fn glob_match(&self, text: &str) -> PyResult<Vec<usize>> {
        let glob_set = self.glob_set.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Glob patterns not initialized. Call setup_glob_patterns first."))?;

        let matches: Vec<usize> = glob_set.matches(text).into_iter().collect();
        Ok(matches)
    }

    /// Fast string normalization and case conversion
    pub fn normalize_string(&self, text: &str, operation: &str) -> PyResult<String> {
        match operation {
            "lowercase" => Ok(text.to_lowercase()),
            "uppercase" => Ok(text.to_uppercase()),
            "trim" => Ok(text.trim().to_string()),
            "trim_whitespace" => Ok(text.chars().filter(|c| !c.is_whitespace()).collect()),
            "normalize_spaces" => {
                let normalized = text.split_whitespace().collect::<Vec<&str>>().join(" ");
                Ok(normalized)
            },
            "ascii_only" => {
                let ascii_only: String = text.chars().filter(|c| c.is_ascii()).collect();
                Ok(ascii_only)
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown normalization operation: {}. Available: lowercase, uppercase, trim, trim_whitespace, normalize_spaces, ascii_only", operation)
            ))
        }
    }

    /// Get statistics about cached regex patterns
    pub fn get_cache_stats(&self) -> PyResult<(usize, Vec<String>)> {
        let cache = REGEX_CACHE.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Regex cache lock error: {}", e))
        })?;

        let count = cache.len();
        let patterns: Vec<String> = cache.keys().cloned().collect();

        Ok((count, patterns))
    }

    /// Clear the regex pattern cache
    pub fn clear_regex_cache(&self) -> PyResult<usize> {
        let mut cache = REGEX_CACHE.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Regex cache lock error: {}", e))
        })?;

        let cleared_count = cache.len();
        cache.clear();

        Ok(cleared_count)
    }
}

impl Default for RustStringOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for string processing
pub mod string_utils {
    use super::*;

    /// Fast string distance calculation using Levenshtein distance
    pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 { return len2; }
        if len2 == 0 { return len1; }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        for i in 0..=len1 { matrix[i][0] = i; }
        for j in 0..=len2 { matrix[0][j] = j; }

        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i-1] == chars2[j-1] { 0 } else { 1 };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(matrix[i-1][j] + 1, matrix[i][j-1] + 1),
                    matrix[i-1][j-1] + cost
                );
            }
        }

        matrix[len1][len2]
    }

    /// Fast fuzzy string matching
    pub fn fuzzy_match_ratio(s1: &str, s2: &str) -> f64 {
        let max_len = std::cmp::max(s1.len(), s2.len());
        if max_len == 0 { return 1.0; }

        let distance = levenshtein_distance(s1, s2);
        1.0 - (distance as f64 / max_len as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_operations() {
        let string_ops = RustStringOps::new();

        // Test regex compilation and caching
        assert!(string_ops.compile_regex(r"\d+", None).unwrap());
        assert!(!string_ops.compile_regex(r"\d+", None).unwrap()); // Should be cached now

        // Test pattern matching
        let matches = string_ops.regex_find_all(r"\d+", "abc123def456", None).unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0], (3, 6, "123".to_string()));
        assert_eq!(matches[1], (9, 12, "456".to_string()));
    }

    #[test]
    fn test_multi_pattern_matching() {
        let mut string_ops = RustStringOps::new();

        // Setup patterns
        let patterns = vec!["foo".to_string(), "bar".to_string(), "baz".to_string()];
        string_ops.setup_multi_pattern(patterns, None).unwrap();

        // Test matching
        let matches = string_ops.multi_pattern_find_all("foo and bar but not baz").unwrap();
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_string_normalization() {
        let string_ops = RustStringOps::new();

        assert_eq!(string_ops.normalize_string("  HELLO WORLD  ", "trim").unwrap(), "HELLO WORLD");
        assert_eq!(string_ops.normalize_string("HELLO WORLD", "lowercase").unwrap(), "hello world");
        assert_eq!(string_ops.normalize_string("hello\n\nworld", "normalize_spaces").unwrap(), "hello world");
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(string_utils::levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(string_utils::levenshtein_distance("hello", "hello"), 0);
        assert_eq!(string_utils::levenshtein_distance("", "abc"), 3);
    }
}
