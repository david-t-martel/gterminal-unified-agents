//! Advanced search functionality with concurrent processing and regex caching
//!
//! This module provides high-performance search operations with:
//! - Concurrent file processing
//! - Regex pattern caching
//! - Glob pattern matching
//! - Path validation and security

use crate::error::{FsError, FsResult};
use crate::path_utils::{resolve_and_validate_path_cached, resolve_path};
use dashmap::DashMap;
use futures::stream::{self, StreamExt, TryStreamExt};
use globset::{Glob, GlobSetBuilder};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Maximum number of concurrent file operations
const CONCURRENT_OPS: usize = 16;

/// Maximum number of cached regex patterns
const MAX_REGEX_CACHE_SIZE: usize = 1000;

/// Global regex pattern cache for avoiding recompilation
static REGEX_CACHE: Lazy<DashMap<String, Arc<Regex>>> = Lazy::new(|| DashMap::with_capacity(100));

/// Get or compile a regex pattern from cache
fn get_cached_regex(pattern: &str) -> FsResult<Arc<Regex>> {
    // Check if pattern exists in cache
    if let Some(cached) = REGEX_CACHE.get(pattern) {
        return Ok(cached.clone());
    }

    // Compile new regex
    let regex = Regex::new(pattern).map_err(|e| FsError::InvalidRegex(e.to_string()))?;
    let arc_regex = Arc::new(regex);

    // Store in cache if not at capacity
    if REGEX_CACHE.len() < MAX_REGEX_CACHE_SIZE {
        REGEX_CACHE.insert(pattern.to_string(), arc_regex.clone());
    }

    Ok(arc_regex)
}

/// Response for search operations containing matched content
#[derive(Debug, Serialize, Deserialize)]
#[pyclass]
pub struct SearchResult {
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub matches: Vec<SearchMatch>,
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!("SearchResult(path='{}', matches={})", self.path, self.matches.len())
    }
}

/// Individual match within a file
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct SearchMatch {
    #[pyo3(get)]
    pub line_number: usize,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub byte_offset: Option<usize>,
}

#[pymethods]
impl SearchMatch {
    fn __repr__(&self) -> String {
        format!("SearchMatch(line={}, content='{:.50}...')",
                self.line_number,
                self.content.chars().take(50).collect::<String>())
    }
}

/// Response for find operations containing file paths
#[derive(Debug, Serialize, Deserialize)]
#[pyclass]
pub struct FindResult {
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub is_file: bool,
    #[pyo3(get)]
    pub size: Option<u64>,
}

#[pymethods]
impl FindResult {
    fn __repr__(&self) -> String {
        format!("FindResult(path='{}', is_file={}, size={:?})",
                self.path, self.is_file, self.size)
    }
}

/// Secure wrapper functions that validate paths before operations

/// Securely search for patterns with path validation
pub async fn search_secure(
    base_path_str: &str,
    pattern: &str,
    file_glob: Option<&str>,
    max_results: Option<usize>,
) -> FsResult<Vec<SearchResult>> {
    let base_path = resolve_and_validate_path_cached(base_path_str).await?;
    search(&base_path, pattern, file_glob, max_results).await
}

/// Securely find files with path validation
pub async fn find_secure(
    base_path_str: &str,
    glob_pattern: &str,
    max_results: Option<usize>,
) -> FsResult<Vec<FindResult>> {
    let base_path = resolve_and_validate_path_cached(base_path_str).await?;
    find(&base_path, glob_pattern, max_results).await
}

/// Securely replace text in files with path validation
pub async fn replace_in_files_secure(
    paths: Vec<String>,
    pattern: &str,
    replacement: &str,
    is_regex: bool,
) -> FsResult<usize> {
    // Validate all paths
    let mut validated_paths = Vec::new();
    for path_str in paths {
        let path = resolve_and_validate_path_cached(&path_str).await?;
        validated_paths.push(path);
    }

    // Convert Vec<PathBuf> to &[&Path]
    let path_refs: Vec<&Path> = validated_paths.iter().map(|p| p.as_path()).collect();
    replace_in_files(&path_refs, pattern, replacement, is_regex).await
}

/// Securely replace a block of text with path validation
pub async fn replace_block_secure(
    path_str: &str,
    start_line: usize,
    end_line: usize,
    new_content: &str,
) -> FsResult<()> {
    let path = resolve_and_validate_path_cached(path_str).await?;
    replace_block(&path, start_line, end_line, new_content).await
}

/// Searches for a pattern in files within a directory tree.
///
/// This function recursively searches through a directory tree, finding all files
/// that match the given glob pattern (if provided) and searching their contents
/// for matches against the provided regex pattern.
pub async fn search(
    base_path: &Path,
    pattern: &str,
    file_glob: Option<&str>,
    max_results: Option<usize>,
) -> FsResult<Vec<SearchResult>> {
    let base_path = resolve_path(base_path.to_str().unwrap_or(""), false).await?;
    let regex = get_cached_regex(pattern)?;
    let max_results = max_results.unwrap_or(10000);

    // Build glob matcher if pattern provided
    let glob_matcher = if let Some(glob_pattern) = file_glob {
        let glob = Glob::new(glob_pattern).map_err(|e| FsError::InvalidGlob(e.to_string()))?;
        Some(
            GlobSetBuilder::new()
                .add(glob)
                .build()
                .map_err(|e| FsError::InvalidGlob(e.to_string()))?,
        )
    } else {
        None
    };

    // Collect all files to search
    let mut files_to_search = Vec::new();
    let mut stack = vec![base_path.clone()];

    while let Some(current_path) = stack.pop() {
        let mut entries = tokio::fs::read_dir(&current_path)
            .await
            .map_err(|e| FsError::io(e.to_string()))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| FsError::io(e.to_string()))?
        {
            let path = entry.path();
            let metadata = entry
                .metadata()
                .await
                .map_err(|e| FsError::io(e.to_string()))?;

            if metadata.is_dir() {
                stack.push(path);
            } else if metadata.is_file() {
                // Check if file matches glob pattern
                if let Some(ref matcher) = glob_matcher {
                    if let Some(file_name) = path.file_name() {
                        let file_name_str = file_name.to_string_lossy();
                        if !matcher.is_match(file_name_str.as_ref()) {
                            continue;
                        }
                    }
                } else {
                    // Skip binary files if no glob pattern specified
                    if is_likely_binary(&path) {
                        continue;
                    }
                }
                files_to_search.push(path);

                if files_to_search.len() >= max_results * 10 {
                    // Limit files to search to prevent excessive resource usage
                    break;
                }
            }
        }
    }

    // Search files concurrently
    let results: Vec<_> = stream::iter(files_to_search)
        .map(|path| {
            let regex_clone = regex.clone();
            async move { search_file(&path, &regex_clone).await }
        })
        .buffer_unordered(CONCURRENT_OPS)
        .try_collect::<Vec<_>>()
        .await?
        .into_iter()
        .flatten()
        .take(max_results)
        .collect();

    Ok(results)
}

/// Search a single file for regex matches
async fn search_file(path: &Path, regex: &Regex) -> FsResult<Option<SearchResult>> {
    // Read the entire file as bytes first
    let content_bytes = tokio::fs::read(path)
        .await
        .map_err(|e| FsError::io(e.to_string()))?;

    // Convert to string lossily to handle non-UTF8 content
    let content = String::from_utf8_lossy(&content_bytes);

    let mut matches = Vec::new();
    let mut line_number = 1;
    let mut byte_offset = 0;

    // Process line by line
    for line in content.lines() {
        if regex.is_match(line) {
            matches.push(SearchMatch {
                line_number,
                content: line.to_string(),
                byte_offset: Some(byte_offset),
            });
        }
        line_number += 1;
        byte_offset += line.len() + 1; // +1 for newline
    }

    if matches.is_empty() {
        Ok(None)
    } else {
        Ok(Some(SearchResult {
            path: path.to_string_lossy().to_string(),
            matches,
        }))
    }
}

/// Check if a file is likely to be binary
fn is_likely_binary(path: &Path) -> bool {
    // Check common binary extensions
    if let Some(extension) = path.extension() {
        let ext = extension.to_string_lossy().to_lowercase();
        matches!(
            ext.as_str(),
            "exe" | "dll" | "so" | "dylib" | "a" | "o" | "lib" | "bin" | "dat" | "db" | "sqlite"
            | "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "ico" | "svg"
            | "mp3" | "mp4" | "avi" | "mov" | "wav" | "flac" | "ogg"
            | "zip" | "tar" | "gz" | "bz2" | "xz" | "7z" | "rar"
            | "pdf" | "doc" | "docx" | "xls" | "xlsx" | "ppt" | "pptx"
        )
    } else {
        false
    }
}

/// Finds files matching a glob pattern within a directory tree.
pub async fn find(
    base_path: &Path,
    glob_pattern: &str,
    max_results: Option<usize>,
) -> FsResult<Vec<FindResult>> {
    let base_path = resolve_path(base_path.to_str().unwrap_or(""), false).await?;
    let max_results = max_results.unwrap_or(10000);

    // Build glob matcher
    let glob = Glob::new(glob_pattern).map_err(|e| FsError::InvalidGlob(e.to_string()))?;
    let matcher = GlobSetBuilder::new()
        .add(glob)
        .build()
        .map_err(|e| FsError::InvalidGlob(e.to_string()))?;

    let mut results = Vec::new();
    let mut stack = vec![base_path.clone()];
    let mut processed = 0;

    while let Some(current_path) = stack.pop() {
        let mut entries = tokio::fs::read_dir(&current_path)
            .await
            .map_err(|e| FsError::io(e.to_string()))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| FsError::io(e.to_string()))?
        {
            let path = entry.path();
            let metadata = entry
                .metadata()
                .await
                .map_err(|e| FsError::io(e.to_string()))?;

            if metadata.is_dir() {
                stack.push(path.clone());
            }

            // Check if the full path or just the filename matches
            let matches = if let Ok(relative) = path.strip_prefix(&base_path) {
                matcher.is_match(relative.to_string_lossy().as_ref())
            } else if let Some(file_name) = path.file_name() {
                matcher.is_match(file_name.to_string_lossy().as_ref())
            } else {
                false
            };

            if matches {
                results.push(FindResult {
                    path: path.to_string_lossy().to_string(),
                    is_file: metadata.is_file(),
                    size: if metadata.is_file() { Some(metadata.len()) } else { None },
                });

                if results.len() >= max_results {
                    return Ok(results);
                }
            }

            processed += 1;
            if processed >= max_results * 100 {
                // Prevent excessive processing
                break;
            }
        }
    }

    Ok(results)
}

/// Replaces text in multiple files using either literal string matching or regex.
pub async fn replace_in_files(
    paths: &[&Path],
    pattern: &str,
    replacement: &str,
    is_regex: bool,
) -> FsResult<usize> {
    // Compile regex if needed
    let regex = if is_regex {
        Some(get_cached_regex(pattern)?)
    } else {
        None
    };

    // Convert paths to owned PathBufs to avoid lifetime issues
    let owned_paths: Vec<PathBuf> = paths.iter().map(|p| (*p).to_path_buf()).collect();

    // Process files concurrently
    let replacement_tasks: Vec<_> = stream::iter(owned_paths.into_iter())
        .map(|path| {
            let regex_clone = regex.clone();
            let replacement_clone = replacement.to_string();
            let pattern_clone = pattern.to_string();
            async move {
                let content = tokio::fs::read_to_string(&path)
                    .await
                    .map_err(|e| FsError::io(e.to_string()))?;

                let (new_content, count) = if let Some(re) = regex_clone {
                    // Regex replacement with capture group support
                    let mut local_count = 0;
                    let result = re.replace_all(&content, |caps: &regex::Captures| {
                        local_count += 1;
                        // Handle capture group replacements
                        let mut expanded = replacement_clone.clone();
                        for (i, mat) in caps.iter().enumerate() {
                            if let Some(m) = mat {
                                expanded = expanded.replace(&format!("${i}"), m.as_str());
                            }
                        }
                        expanded
                    });
                    (result.into_owned(), local_count)
                } else {
                    // Literal string replacement
                    let local_count = content.matches(&pattern_clone).count();
                    (
                        content.replace(&pattern_clone, &replacement_clone),
                        local_count,
                    )
                };

                if count > 0 {
                    tokio::fs::write(&path, new_content)
                        .await
                        .map_err(|e| FsError::io(e.to_string()))?;
                    Ok::<usize, FsError>(count)
                } else {
                    Ok::<usize, FsError>(0)
                }
            }
        })
        .buffer_unordered(CONCURRENT_OPS)
        .try_collect::<Vec<usize>>()
        .await?;

    Ok(replacement_tasks.iter().sum())
}

/// Replaces a block of lines in a file with new content.
pub async fn replace_block(
    path: &Path,
    start_line: usize,
    end_line: usize,
    new_content: &str,
) -> FsResult<()> {
    // Validate line range
    if start_line == 0 || end_line < start_line {
        return Err(FsError::InvalidLineRange);
    }

    // Read entire file content
    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| FsError::io(e.to_string()))?;

    // Split into lines while preserving line endings
    let lines: Vec<&str> = content.lines().collect();
    let mut new_lines: Vec<String> = Vec::new();

    // Check if line numbers are beyond file length
    if start_line > lines.len() || end_line > lines.len() {
        return Err(FsError::InvalidLineRange);
    }

    // Add lines before the block
    if start_line > 1 {
        new_lines.extend(
            lines[..start_line.saturating_sub(1)]
                .iter()
                .map(|&s| s.to_string()),
        );
    }

    // Add the replacement content
    new_lines.extend(new_content.lines().map(String::from));

    // Add lines after the block
    if end_line < lines.len() {
        new_lines.extend(lines[end_line..].iter().map(|&s| s.to_string()));
    }

    // Join with newlines and write back
    let final_content = new_lines.join("\n");
    tokio::fs::write(path, final_content)
        .await
        .map_err(|e| FsError::io(e.to_string()))
}

/// PyO3 wrapper for advanced search operations
#[pyclass]
pub struct RustAdvancedSearch;

#[pymethods]
impl RustAdvancedSearch {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Search for pattern in directory
    fn search<'py>(
        &self,
        py: Python<'py>,
        base_path: String,
        pattern: String,
        file_glob: Option<String>,
        max_results: Option<usize>,
    ) -> PyResult<&'py PyAny> {
        pyo3_asyncio::tokio::return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            search_secure(&base_path, &pattern, file_glob.as_deref(), max_results)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    /// Find files matching glob pattern
    fn find<'py>(
        &self,
        py: Python<'py>,
        base_path: String,
        glob_pattern: String,
        max_results: Option<usize>,
    ) -> PyResult<&'py PyAny> {
        pyo3_asyncio::tokio::return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            find_secure(&base_path, &glob_pattern, max_results)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    /// Replace text in files
    fn replace_in_files<'py>(
        &self,
        py: Python<'py>,
        paths: Vec<String>,
        pattern: String,
        replacement: String,
        is_regex: bool,
    ) -> PyResult<&'py PyAny> {
        pyo3_asyncio::tokio::return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            replace_in_files_secure(paths, &pattern, &replacement, is_regex)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    /// Get cache statistics
    fn get_regex_cache_stats(&self) -> (usize, usize) {
        (REGEX_CACHE.len(), MAX_REGEX_CACHE_SIZE)
    }

    /// Clear regex cache
    fn clear_regex_cache(&self) {
        REGEX_CACHE.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs;

    #[tokio::test]
    async fn test_regex_caching() {
        let pattern = r"\d+";

        // First access should compile and cache
        let regex1 = get_cached_regex(pattern).unwrap();

        // Second access should return cached version
        let regex2 = get_cached_regex(pattern).unwrap();

        // Should be the same Arc instance
        assert!(Arc::ptr_eq(&regex1, &regex2));
    }

    #[tokio::test]
    async fn test_search_functionality() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");

        fs::write(&test_file, "line 1\ntest pattern\nline 3\nanother test").await.unwrap();

        let results = search(temp_dir.path(), "test", None, None).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matches.len(), 2);
        assert_eq!(results[0].matches[0].line_number, 2);
        assert_eq!(results[0].matches[1].line_number, 4);
    }

    #[tokio::test]
    async fn test_find_functionality() {
        let temp_dir = TempDir::new().unwrap();

        fs::write(temp_dir.path().join("test.txt"), "content").await.unwrap();
        fs::write(temp_dir.path().join("other.log"), "content").await.unwrap();
        fs::create_dir(temp_dir.path().join("subdir")).await.unwrap();
        fs::write(temp_dir.path().join("subdir/nested.txt"), "content").await.unwrap();

        let results = find(temp_dir.path(), "*.txt", None).await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.path.contains("test.txt")));
        assert!(results.iter().any(|r| r.path.contains("nested.txt")));
    }
}
