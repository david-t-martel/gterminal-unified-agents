//! High-performance file I/O operations
//!
//! This module provides PyO3 bindings for file operations that are significantly
//! faster than Python equivalents:
//! - Memory-mapped file access for large files
//! - Parallel directory traversal
//! - Fast text search with regex and aho-corasick
//! - Efficient file hashing and comparison

use aho_corasick::AhoCorasick;
use blake3;
use ignore::WalkBuilder;
use memmap2::Mmap;
use pyo3::prelude::*;
// Removed pyo3_asyncio - using sync operations with PyO3 0.22
use rayon::prelude::*;
use regex::Regex;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tokio::fs as async_fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use walkdir::WalkDir;

// Import new integrated modules
use crate::buffer_pool::BufferGuard;
use crate::path_utils::resolve_and_validate_path_cached;

/// High-performance file operations
#[pyclass]
pub struct RustFileOps {
    max_file_size: usize,
    follow_symlinks: bool,
    respect_gitignore: bool,
}

#[pymethods]
impl RustFileOps {
    #[new]
    #[pyo3(signature = (max_file_size_mb=100, follow_symlinks=false, respect_gitignore=true))]
    fn new(max_file_size_mb: usize, follow_symlinks: bool, respect_gitignore: bool) -> Self {
        Self {
            max_file_size: max_file_size_mb * 1024 * 1024,
            follow_symlinks,
            respect_gitignore,
        }
    }

    /// Fast file reading with memory mapping for large files and path validation
    fn read_file<'py>(&self, py: Python<'py>, file_path: String) -> PyResult<&'py PyAny> {
        let max_size = self.max_file_size;

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            // Security: Validate and resolve path
            let validated_path = resolve_and_validate_path_cached(&file_path)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyPermissionError, _>(e.to_string()))?;

            if !validated_path.exists() {
                return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                    format!("File does not exist: {file_path}"),
                ));
            }

            let metadata = async_fs::metadata(&validated_path)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

            if metadata.len() > max_size as u64 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "File too large: {} bytes (max: {} bytes)",
                    metadata.len(),
                    max_size
                )));
            }

            if metadata.len() > 10 * 1024 * 1024 {
                // Use memory mapping for files > 10MB
                let file = std::fs::File::open(&validated_path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
                let mmap = unsafe { Mmap::map(&file) }
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
                Ok(String::from_utf8_lossy(&mmap).to_string())
            } else {
                // Use buffer pool for efficient small file reading
                let mut buffer_guard = BufferGuard::new(metadata.len() as usize);
                let buffer = buffer_guard.buffer();

                let mut file = async_fs::File::open(&validated_path)
                    .await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

                file.read_to_end(buffer)
                    .await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

                Ok(String::from_utf8_lossy(buffer).to_string())
            }
        })
    }

    /// Fast file writing with atomic operations and path validation
    fn write_file<'py>(
        &self,
        py: Python<'py>,
        file_path: String,
        content: String,
    ) -> PyResult<&'py PyAny> {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            // Security: Validate and resolve path
            let validated_path = resolve_and_validate_path_cached(&file_path)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyPermissionError, _>(e.to_string()))?;

            // Create parent directories if they don't exist
            if let Some(parent) = validated_path.parent() {
                async_fs::create_dir_all(parent).await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            }

            // Write to temporary file first for atomic operation
            let temp_path = format!("{}.tmp.{}", validated_path.to_string_lossy(), uuid::Uuid::new_v4());

            {
                let mut file = async_fs::File::create(&temp_path).await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
                file.write_all(content.as_bytes()).await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
                file.flush().await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            }

            // Atomic rename
            async_fs::rename(&temp_path, &validated_path).await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

            Ok(true)
        })
    }

    /// Fast directory listing with filtering
    #[pyo3(signature = (directory, pattern=None, recursive=false, max_depth=None))]
    fn list_directory(
        &self,
        directory: &str,
        pattern: Option<&str>,
        recursive: bool,
        max_depth: Option<usize>,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let path = Path::new(directory);
        if !path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("Directory does not exist: {directory}"),
            ));
        }

        let mut results = Vec::new();

        if recursive {
            let walker = if self.respect_gitignore {
                WalkBuilder::new(path)
                    .follow_links(self.follow_symlinks)
                    .build()
            } else {
                // Use walkdir for simpler cases
                let mut walk = WalkDir::new(path);
                if let Some(depth) = max_depth {
                    walk = walk.max_depth(depth);
                }
                if !self.follow_symlinks {
                    walk = walk.follow_links(false);
                }

                let entries: Vec<_> = walk.into_iter().filter_map(|e| e.ok()).collect();

                return Python::with_gil(|py| {
                    Ok(entries
                        .into_iter()
                        .filter_map(|entry| {
                            let file_name = entry.file_name().to_string_lossy().to_string();

                            // Apply pattern filter if specified
                            if let Some(pat) = pattern {
                                if !file_name.to_lowercase().contains(&pat.to_lowercase()) {
                                    return None;
                                }
                            }

                            self.create_file_info(py, entry.path()).ok()
                        })
                        .collect())
                });
            };

            // Handle gitignore-aware walker
            for entry in walker {
                match entry {
                    Ok(dir_entry) => {
                        let file_name = dir_entry.file_name().to_string_lossy().to_string();

                        // Apply pattern filter if specified
                        if let Some(pat) = pattern {
                            if !file_name.to_lowercase().contains(&pat.to_lowercase()) {
                                continue;
                            }
                        }

                        Python::with_gil(|py| {
                            if let Ok(info) = self.create_file_info(py, dir_entry.path()) {
                                results.push(info);
                            }
                        });
                    }
                    Err(_) => continue,
                }
            }
        } else {
            // Non-recursive listing
            let entries = fs::read_dir(path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

            Python::with_gil(|py| {
                for entry in entries {
                    if let Ok(dir_entry) = entry {
                        let file_name = dir_entry.file_name().to_string_lossy().to_string();

                        // Apply pattern filter if specified
                        if let Some(pat) = pattern {
                            if !file_name.to_lowercase().contains(&pat.to_lowercase()) {
                                continue;
                            }
                        }

                        if let Ok(info) = self.create_file_info(py, &dir_entry.path()) {
                            results.push(info);
                        }
                    }
                }
            });
        }

        Ok(results)
    }

    /// Fast file hashing with multiple algorithms
    #[pyo3(signature = (file_path, algorithm="blake3"))]
    fn hash_file(&self, file_path: &str, algorithm: &str) -> PyResult<String> {
        let path = Path::new(file_path);
        if !path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("File does not exist: {file_path}"),
            ));
        }

        let metadata = fs::metadata(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        if metadata.len() > self.max_file_size as u64 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "File too large: {} bytes",
                metadata.len()
            )));
        }

        match algorithm {
            "blake3" => {
                let contents = fs::read(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
                let hash = blake3::hash(&contents);
                Ok(hash.to_hex().to_string())
            }
            "sha256" => {
                let contents = fs::read(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
                let mut hasher = Sha256::new();
                hasher.update(&contents);
                Ok(hex::encode(hasher.finalize()))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported hash algorithm. Use 'blake3' or 'sha256'",
            )),
        }
    }

    /// Batch hash multiple files in parallel
    #[pyo3(signature = (file_paths, algorithm="blake3"))]
    fn batch_hash_files(
        &self,
        file_paths: Vec<&str>,
        algorithm: &str,
    ) -> PyResult<HashMap<String, String>> {
        let max_size = self.max_file_size;

        let results: HashMap<String, String> = file_paths
            .par_iter()
            .filter_map(|&path| {
                let file_path = Path::new(path);
                if !file_path.exists() {
                    return None;
                }

                let metadata = fs::metadata(file_path).ok()?;
                if metadata.len() > max_size as u64 {
                    return None;
                }

                let contents = fs::read(file_path).ok()?;

                let hash = match algorithm {
                    "blake3" => blake3::hash(&contents).to_hex().to_string(),
                    "sha256" => {
                        let mut hasher = Sha256::new();
                        hasher.update(&contents);
                        hex::encode(hasher.finalize())
                    }
                    _ => return None,
                };

                Some((path.to_string(), hash))
            })
            .collect();

        Ok(results)
    }

    /// Find duplicate files based on content hash
    fn find_duplicates(&self, directory: &str) -> PyResult<HashMap<String, Vec<String>>> {
        let mut hash_to_files: HashMap<String, Vec<String>> = HashMap::new();

        let walker = WalkDir::new(directory)
            .follow_links(self.follow_symlinks)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file());

        for entry in walker {
            let path = entry.path();

            if let Ok(metadata) = entry.metadata() {
                if metadata.len() > self.max_file_size as u64 {
                    continue;
                }
            } else {
                continue;
            }

            if let Ok(contents) = fs::read(path) {
                let hash = blake3::hash(&contents).to_hex().to_string();
                hash_to_files
                    .entry(hash)
                    .or_default()
                    .push(path.to_string_lossy().to_string());
            }
        }

        // Only return hashes with multiple files (duplicates)
        let duplicates: HashMap<String, Vec<String>> = hash_to_files
            .into_iter()
            .filter(|(_, files)| files.len() > 1)
            .collect();

        Ok(duplicates)
    }

    /// Get directory statistics
    fn directory_stats(&self, directory: &str) -> PyResult<HashMap<String, u64>> {
        let mut file_count = 0u64;
        let mut dir_count = 0u64;
        let mut total_size = 0u64;
        let mut largest_file = 0u64;

        let walker = WalkDir::new(directory)
            .follow_links(self.follow_symlinks)
            .into_iter()
            .filter_map(|e| e.ok());

        for entry in walker {
            if entry.file_type().is_file() {
                file_count += 1;
                if let Ok(metadata) = entry.metadata() {
                    let size = metadata.len();
                    total_size += size;
                    if size > largest_file {
                        largest_file = size;
                    }
                }
            } else if entry.file_type().is_dir() {
                dir_count += 1;
            }
        }

        let mut stats = HashMap::new();
        stats.insert("files".to_string(), file_count);
        stats.insert("directories".to_string(), dir_count);
        stats.insert("total_size".to_string(), total_size);
        stats.insert("largest_file".to_string(), largest_file);
        stats.insert(
            "average_file_size".to_string(),
            if file_count > 0 {
                total_size / file_count
            } else {
                0
            },
        );

        Ok(stats)
    }

    /// Copy file with progress tracking
    fn copy_file<'py>(
        &self,
        py: Python<'py>,
        source: String,
        destination: String,
    ) -> PyResult<&'py PyAny> {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let src_path = Path::new(&source);
            let dst_path = Path::new(&destination);

            if !src_path.exists() {
                return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                    format!("Source file does not exist: {source}"),
                ));
            }

            // Create parent directories if they don't exist
            if let Some(parent) = dst_path.parent() {
                async_fs::create_dir_all(parent)
                    .await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            }

            async_fs::copy(&src_path, &dst_path)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

            Ok(true)
        })
    }

    /// Move/rename file atomically
    fn move_file<'py>(
        &self,
        py: Python<'py>,
        source: String,
        destination: String,
    ) -> PyResult<&'py PyAny> {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let src_path = Path::new(&source);
            let dst_path = Path::new(&destination);

            if !src_path.exists() {
                return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                    format!("Source file does not exist: {source}"),
                ));
            }

            // Create parent directories if they don't exist
            if let Some(parent) = dst_path.parent() {
                async_fs::create_dir_all(parent)
                    .await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            }

            async_fs::rename(&src_path, &dst_path)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

            Ok(true)
        })
    }
}

impl RustFileOps {
    /// Create file information dictionary
    fn create_file_info(&self, py: Python<'_>, path: &Path) -> PyResult<HashMap<String, PyObject>> {
        let mut info = HashMap::new();

        info.insert(
            "path".to_string(),
            path.to_string_lossy().to_string().to_object(py),
        );
        info.insert(
            "name".to_string(),
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
                .to_object(py),
        );

        if let Ok(metadata) = fs::metadata(path) {
            info.insert("size".to_string(), metadata.len().to_object(py));
            info.insert("is_file".to_string(), metadata.is_file().to_object(py));
            info.insert("is_dir".to_string(), metadata.is_dir().to_object(py));
            info.insert(
                "readonly".to_string(),
                metadata.permissions().readonly().to_object(py),
            );

            if let Ok(created) = metadata.created() {
                let timestamp = created
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                info.insert("created".to_string(), timestamp.to_object(py));
            }

            if let Ok(modified) = metadata.modified() {
                let timestamp = modified
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                info.insert("modified".to_string(), timestamp.to_object(py));
            }
        }

        Ok(info)
    }
}

/// High-performance text search engine
#[pyclass]
pub struct RustSearchEngine {
    case_sensitive: bool,
    max_file_size: usize,
    max_results: usize,
}

#[pymethods]
impl RustSearchEngine {
    #[new]
    #[pyo3(signature = (case_sensitive=false, max_file_size_mb=10, max_results=10000))]
    fn new(case_sensitive: bool, max_file_size_mb: usize, max_results: usize) -> Self {
        Self {
            case_sensitive,
            max_file_size: max_file_size_mb * 1024 * 1024,
            max_results,
        }
    }

    /// Search for pattern in file using regex
    #[pyo3(signature = (file_path, pattern, context_lines=0))]
    fn search_file(
        &self,
        file_path: &str,
        pattern: &str,
        context_lines: usize,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let path = Path::new(file_path);
        if !path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("File does not exist: {file_path}"),
            ));
        }

        let metadata = fs::metadata(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        if metadata.len() > self.max_file_size as u64 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "File too large: {} bytes",
                metadata.len()
            )));
        }

        let regex = if self.case_sensitive {
            Regex::new(pattern)
        } else {
            Regex::new(&format!("(?i){pattern}"))
        }
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid regex: {e}"))
        })?;

        let file = fs::File::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let reader = BufReader::new(file);
        let lines: Vec<String> = reader
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let mut results = Vec::new();

        Python::with_gil(|py| {
            for (line_num, line) in lines.iter().enumerate() {
                if regex.is_match(line) {
                    let mut match_info = HashMap::new();
                    match_info.insert("line_number".to_string(), (line_num + 1).to_object(py));
                    match_info.insert("line".to_string(), line.to_object(py));
                    match_info.insert("file_path".to_string(), file_path.to_object(py));

                    // Add context lines if requested
                    if context_lines > 0 {
                        let start = line_num.saturating_sub(context_lines);
                        let end = std::cmp::min(line_num + context_lines + 1, lines.len());

                        let context: Vec<String> = lines[start..end].to_vec();
                        match_info.insert("context".to_string(), context.to_object(py));
                    }

                    // Extract matched groups
                    if let Some(captures) = regex.captures(line) {
                        let matches: Vec<String> = captures
                            .iter()
                            .filter_map(|m| m.map(|m| m.as_str().to_string()))
                            .collect();
                        match_info.insert("matches".to_string(), matches.to_object(py));
                    }

                    results.push(match_info);

                    if results.len() >= self.max_results {
                        break;
                    }
                }
            }
        });

        Ok(results)
    }

    /// Search for multiple patterns in directory using AhoCorasick
    #[pyo3(signature = (directory, patterns, file_extensions=None, recursive=true))]
    fn search_directory(
        &self,
        directory: &str,
        patterns: Vec<&str>,
        file_extensions: Option<Vec<&str>>,
        recursive: bool,
    ) -> PyResult<HashMap<String, Vec<HashMap<String, PyObject>>>> {
        let patterns_clone = patterns.clone(); // Clone before moving into AhoCorasick
        let ac = AhoCorasick::new(patterns).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("AhoCorasick error: {e}"))
        })?;

        let mut results: HashMap<String, Vec<HashMap<String, PyObject>>> = HashMap::new();
        let mut processed = 0;

        let walker = if recursive {
            WalkDir::new(directory).into_iter()
        } else {
            WalkDir::new(directory).max_depth(1).into_iter()
        };

        for entry in walker.filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();

            // Check file extension filter
            if let Some(ref extensions) = file_extensions {
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if !extensions.iter().any(|&e| e.to_lowercase() == ext_str) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Check file size
            if let Ok(metadata) = entry.metadata() {
                if metadata.len() > self.max_file_size as u64 {
                    continue;
                }
            } else {
                continue;
            }

            // Search file content
            if let Ok(content) = fs::read_to_string(path) {
                let mut file_matches = Vec::new();

                for mat in ac.find_iter(&content) {
                    // Find line number for the match
                    let line_start = content[..mat.start()]
                        .rfind('\n')
                        .map(|i| i + 1)
                        .unwrap_or(0);
                    let line_end = content[mat.end()..]
                        .find('\n')
                        .map(|i| mat.end() + i)
                        .unwrap_or(content.len());
                    let line = &content[line_start..line_end];
                    let line_num = content[..mat.start()].matches('\n').count() + 1;

                    Python::with_gil(|py| {
                        let mut match_info = HashMap::new();
                        match_info.insert(
                            "pattern_id".to_string(),
                            mat.pattern().as_usize().to_object(py),
                        );
                        match_info.insert(
                            "pattern".to_string(),
                            patterns_clone[mat.pattern().as_usize()].to_object(py),
                        );
                        match_info.insert("line_number".to_string(), line_num.to_object(py));
                        match_info.insert("line".to_string(), line.to_object(py));
                        match_info.insert("start".to_string(), mat.start().to_object(py));
                        match_info.insert("end".to_string(), mat.end().to_object(py));

                        file_matches.push(match_info);
                    });

                    if file_matches.len() >= self.max_results {
                        break;
                    }
                }

                if !file_matches.is_empty() {
                    results.insert(path.to_string_lossy().to_string(), file_matches);
                }
            }

            processed += 1;
            if processed >= 10000 {
                // Limit to prevent excessive resource usage
                break;
            }
        }

        Ok(results)
    }

    /// Fast line count for file
    fn count_lines(&self, file_path: &str) -> PyResult<usize> {
        let path = Path::new(file_path);
        if !path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("File does not exist: {file_path}"),
            ));
        }

        let metadata = fs::metadata(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        if metadata.len() > self.max_file_size as u64 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "File too large: {} bytes",
                metadata.len()
            )));
        }

        let content = fs::read_to_string(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        Ok(content.lines().count())
    }

    /// Replace text in file with backup
    #[pyo3(signature = (file_path, search_pattern, replacement, create_backup=true))]
    fn replace_in_file(
        &self,
        file_path: &str,
        search_pattern: &str,
        replacement: &str,
        create_backup: bool,
    ) -> PyResult<usize> {
        let path = Path::new(file_path);
        if !path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("File does not exist: {file_path}"),
            ));
        }

        let content = fs::read_to_string(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let regex = if self.case_sensitive {
            Regex::new(search_pattern)
        } else {
            Regex::new(&format!("(?i){search_pattern}"))
        }
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid regex: {e}"))
        })?;

        let new_content = regex.replace_all(&content, replacement);
        let replacement_count = regex.find_iter(&content).count();

        if replacement_count > 0 {
            // Create backup if requested
            if create_backup {
                let backup_path = format!("{file_path}.backup");
                fs::copy(path, &backup_path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            }

            // Write new content
            fs::write(path, new_content.as_bytes())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        }

        Ok(replacement_count)
    }
}
