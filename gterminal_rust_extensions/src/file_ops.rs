//! RustFileOps: High-performance file operations with async support and file watching

use crate::utils::{increment_ops, validate_path, check_file_size, ResultExt, track_allocation, track_deallocation};
use anyhow::{Context, Result};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, RwLock};
use walkdir::WalkDir;

/// File operation statistics
#[derive(Debug, Clone, Default)]
struct FileOpStats {
    reads: u64,
    writes: u64,
    deletes: u64,
    creates: u64,
    bytes_read: u64,
    bytes_written: u64,
}

/// File watcher handle for cleanup
pub struct WatcherHandle {
    _watcher: RecommendedWatcher,
    receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<Event>>>>,
    path: PathBuf,
}

/// High-performance file operations with async support
#[pyclass]
pub struct RustFileOps {
    runtime: tokio::runtime::Runtime,
    stats: Arc<RwLock<FileOpStats>>,
    watchers: Arc<RwLock<HashMap<String, WatcherHandle>>>,
    max_file_size: Option<u64>,
    parallel_threshold: usize,
}

#[pymethods]
impl RustFileOps {
    /// Create new RustFileOps instance
    #[new]
    #[pyo3(signature = (max_file_size = None, parallel_threshold = 10))]
    fn new(max_file_size: Option<u64>, parallel_threshold: usize) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .thread_name("file-ops")
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            runtime,
            stats: Arc::new(RwLock::new(FileOpStats::default())),
            watchers: Arc::new(RwLock::new(HashMap::new())),
            max_file_size,
            parallel_threshold,
        })
    }

    /// Read file content asynchronously
    #[pyo3(signature = (path, encoding = "utf-8", chunk_size = None))]
    fn read_file(
        &self,
        path: &str,
        encoding: &str,
        chunk_size: Option<usize>
    ) -> PyResult<String> {
        self.runtime.block_on(async {
            let path = validate_path(path).to_py_err()?;
            check_file_size(&path, self.max_file_size).to_py_err()?;

            let content = match chunk_size {
                Some(size) => self.read_file_chunked(&path, size).await,
                None => fs::read_to_string(&path).await
                    .with_context(|| format!("Failed to read file: {}", path.display())),
            }.to_py_err()?;

            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.reads += 1;
                stats.bytes_read += content.len() as u64;
            }

            increment_ops();
            Ok(content)
        })
    }

    /// Read file as bytes
    fn read_file_bytes(&self, path: &str) -> PyResult<Py<PyBytes>> {
        self.runtime.block_on(async {
            let path = validate_path(path).to_py_err()?;
            check_file_size(&path, self.max_file_size).to_py_err()?;

            let content = fs::read(&path).await
                .with_context(|| format!("Failed to read file: {}", path.display()))
                .to_py_err()?;

            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.reads += 1;
                stats.bytes_read += content.len() as u64;
            }

            increment_ops();
            Python::with_gil(|py| Ok(PyBytes::new(py, &content).into()))
        })
    }

    /// Write content to file asynchronously
    #[pyo3(signature = (path, content, create_dirs = true, backup = false))]
    fn write_file(
        &self,
        path: &str,
        content: &str,
        create_dirs: bool,
        backup: bool,
    ) -> PyResult<usize> {
        self.runtime.block_on(async {
            let path = validate_path(path).to_py_err()?;

            // Create parent directories if needed
            if create_dirs {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).await
                        .with_context(|| format!("Failed to create directories for: {}", path.display()))
                        .to_py_err()?;
                }
            }

            // Create backup if requested
            if backup && path.exists() {
                let backup_path = path.with_extension(
                    format!("{}.backup.{}",
                        path.extension().and_then(|s| s.to_str()).unwrap_or(""),
                        chrono::Utc::now().timestamp()
                    )
                );
                fs::copy(&path, &backup_path).await
                    .with_context(|| format!("Failed to create backup: {}", backup_path.display()))
                    .to_py_err()?;
            }

            // Write content
            fs::write(&path, content).await
                .with_context(|| format!("Failed to write file: {}", path.display()))
                .to_py_err()?;

            // Update statistics
            let bytes_written = content.len();
            {
                let mut stats = self.stats.write().await;
                stats.writes += 1;
                stats.bytes_written += bytes_written as u64;
            }

            increment_ops();
            track_allocation(bytes_written as u64);
            Ok(bytes_written)
        })
    }

    /// Write bytes to file
    fn write_file_bytes(&self, path: &str, content: &[u8]) -> PyResult<usize> {
        self.runtime.block_on(async {
            let path = validate_path(path).to_py_err()?;

            fs::write(&path, content).await
                .with_context(|| format!("Failed to write file: {}", path.display()))
                .to_py_err()?;

            // Update statistics
            let bytes_written = content.len();
            {
                let mut stats = self.stats.write().await;
                stats.writes += 1;
                stats.bytes_written += bytes_written as u64;
            }

            increment_ops();
            track_allocation(bytes_written as u64);
            Ok(bytes_written)
        })
    }

    /// Append content to file
    fn append_file(&self, path: &str, content: &str) -> PyResult<usize> {
        self.runtime.block_on(async {
            let path = validate_path(path).to_py_err()?;

            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .await
                .with_context(|| format!("Failed to open file for append: {}", path.display()))
                .to_py_err()?;

            file.write_all(content.as_bytes()).await
                .with_context(|| format!("Failed to append to file: {}", path.display()))
                .to_py_err()?;

            file.sync_all().await
                .with_context(|| format!("Failed to sync file: {}", path.display()))
                .to_py_err()?;

            let bytes_written = content.len();
            {
                let mut stats = self.stats.write().await;
                stats.writes += 1;
                stats.bytes_written += bytes_written as u64;
            }

            increment_ops();
            Ok(bytes_written)
        })
    }

    /// Delete file or directory
    #[pyo3(signature = (path, recursive = false))]
    fn delete(&self, path: &str, recursive: bool) -> PyResult<bool> {
        self.runtime.block_on(async {
            let path = validate_path(path).to_py_err()?;

            if !path.exists() {
                return Ok(false);
            }

            if path.is_file() {
                fs::remove_file(&path).await
                    .with_context(|| format!("Failed to delete file: {}", path.display()))
                    .to_py_err()?;
            } else if path.is_dir() {
                if recursive {
                    fs::remove_dir_all(&path).await
                        .with_context(|| format!("Failed to delete directory: {}", path.display()))
                        .to_py_err()?;
                } else {
                    fs::remove_dir(&path).await
                        .with_context(|| format!("Failed to delete directory: {}", path.display()))
                        .to_py_err()?;
                }
            }

            {
                let mut stats = self.stats.write().await;
                stats.deletes += 1;
            }

            increment_ops();
            Ok(true)
        })
    }

    /// Create directory
    #[pyo3(signature = (path, parents = true))]
    fn create_dir(&self, path: &str, parents: bool) -> PyResult<bool> {
        self.runtime.block_on(async {
            let path = validate_path(path).to_py_err()?;

            if path.exists() {
                return Ok(false);
            }

            if parents {
                fs::create_dir_all(&path).await
            } else {
                fs::create_dir(&path).await
            }
            .with_context(|| format!("Failed to create directory: {}", path.display()))
            .to_py_err()?;

            {
                let mut stats = self.stats.write().await;
                stats.creates += 1;
            }

            increment_ops();
            Ok(true)
        })
    }

    /// List directory contents with filtering
    #[pyo3(signature = (path, pattern = None, recursive = false, include_hidden = false))]
    fn list_dir(
        &self,
        path: &str,
        pattern: Option<&str>,
        recursive: bool,
        include_hidden: bool,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        self.runtime.block_on(async {
            let path = validate_path(path).to_py_err()?;

            let entries = if recursive {
                self.list_recursive(&path, pattern, include_hidden).await
            } else {
                self.list_single(&path, pattern, include_hidden).await
            }.to_py_err()?;

            increment_ops();
            Ok(entries)
        })
    }

    /// Copy file or directory
    #[pyo3(signature = (src, dst, overwrite = false))]
    fn copy(&self, src: &str, dst: &str, overwrite: bool) -> PyResult<u64> {
        self.runtime.block_on(async {
            let src_path = validate_path(src).to_py_err()?;
            let dst_path = validate_path(dst).to_py_err()?;

            if !overwrite && dst_path.exists() {
                return Err(pyo3::exceptions::PyFileExistsError::new_err(
                    format!("Destination exists: {}", dst_path.display())
                ));
            }

            let bytes_copied = if src_path.is_file() {
                fs::copy(&src_path, &dst_path).await
                    .with_context(|| format!("Failed to copy file: {} -> {}",
                        src_path.display(), dst_path.display()))
                    .to_py_err()?
            } else {
                self.copy_recursive(&src_path, &dst_path).await.to_py_err()?
            };

            increment_ops();
            Ok(bytes_copied)
        })
    }

    /// Move/rename file or directory
    fn move_file(&self, src: &str, dst: &str) -> PyResult<()> {
        self.runtime.block_on(async {
            let src_path = validate_path(src).to_py_err()?;
            let dst_path = validate_path(dst).to_py_err()?;

            // Create parent directories if needed
            if let Some(parent) = dst_path.parent() {
                fs::create_dir_all(parent).await
                    .with_context(|| format!("Failed to create directories for: {}", dst_path.display()))
                    .to_py_err()?;
            }

            fs::rename(&src_path, &dst_path).await
                .with_context(|| format!("Failed to move: {} -> {}",
                    src_path.display(), dst_path.display()))
                .to_py_err()?;

            increment_ops();
            Ok(())
        })
    }

    /// Get file/directory information
    fn get_info(&self, path: &str) -> PyResult<HashMap<String, PyObject>> {
        self.runtime.block_on(async {
            let path = validate_path(path).to_py_err()?;

            let metadata = fs::metadata(&path).await
                .with_context(|| format!("Failed to get metadata: {}", path.display()))
                .to_py_err()?;

            let mut info = HashMap::new();

            Python::with_gil(|py| {
                info.insert("path".to_string(), path.to_string_lossy().to_string().into_py(py));
                info.insert("size".to_string(), metadata.len().into_py(py));
                info.insert("is_file".to_string(), metadata.is_file().into_py(py));
                info.insert("is_dir".to_string(), metadata.is_dir().into_py(py));
                info.insert("is_symlink".to_string(), metadata.is_symlink().into_py(py));

                if let Ok(modified) = metadata.modified() {
                    if let Ok(duration) = modified.duration_since(SystemTime::UNIX_EPOCH) {
                        info.insert("modified".to_string(), duration.as_secs().into_py(py));
                    }
                }

                if let Ok(created) = metadata.created() {
                    if let Ok(duration) = created.duration_since(SystemTime::UNIX_EPOCH) {
                        info.insert("created".to_string(), duration.as_secs().into_py(py));
                    }
                }

                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let permissions = metadata.permissions();
                    info.insert("mode".to_string(), format!("{:o}", permissions.mode()).into_py(py));
                }
            });

            increment_ops();
            Ok(info)
        })
    }

    /// Start watching a path for changes
    #[pyo3(signature = (path, recursive = true))]
    fn watch_path(&self, path: &str, recursive: bool) -> PyResult<String> {
        let path = validate_path(path).to_py_err()?;
        let watch_id = uuid::Uuid::new_v4().to_string();

        let (tx, rx) = mpsc::unbounded_channel();

        let mut watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                match res {
                    Ok(event) => {
                        let _ = tx.send(event);
                    }
                    Err(e) => {
                        tracing::error!("Watch error: {}", e);
                    }
                }
            },
            Config::default(),
        ).to_py_err()?;

        let mode = if recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        watcher.watch(&path, mode).to_py_err()?;

        let handle = WatcherHandle {
            _watcher: watcher,
            receiver: Arc::new(RwLock::new(Some(rx))),
            path: path.clone(),
        };

        self.runtime.block_on(async {
            let mut watchers = self.watchers.write().await;
            watchers.insert(watch_id.clone(), handle);
        });

        Ok(watch_id)
    }

    /// Get pending file system events
    fn get_events(&self, watch_id: &str, timeout_ms: Option<u64>) -> PyResult<Vec<HashMap<String, PyObject>>> {
        self.runtime.block_on(async {
            let watchers = self.watchers.read().await;
            let handle = watchers.get(watch_id)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
                    format!("Watch ID not found: {}", watch_id)
                ))?;

            let mut receiver_guard = handle.receiver.write().await;
            let receiver = receiver_guard.as_mut()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                    "Watcher already closed"
                ))?;

            let mut events = Vec::new();
            let timeout = timeout_ms.map(|ms| Duration::from_millis(ms)).unwrap_or(Duration::from_millis(100));
            let deadline = tokio::time::Instant::now() + timeout;

            while tokio::time::Instant::now() < deadline {
                match tokio::time::timeout(Duration::from_millis(10), receiver.recv()).await {
                    Ok(Some(event)) => {
                        events.push(self.event_to_dict(event));
                        if events.len() >= 100 { // Prevent unbounded growth
                            break;
                        }
                    }
                    Ok(None) => break, // Channel closed
                    Err(_) => continue, // Timeout, try again
                }
            }

            Ok(events)
        })
    }

    /// Stop watching a path
    fn unwatch(&self, watch_id: &str) -> PyResult<bool> {
        self.runtime.block_on(async {
            let mut watchers = self.watchers.write().await;
            Ok(watchers.remove(watch_id).is_some())
        })
    }

    /// Batch file operations for improved performance
    fn batch_read(&self, paths: Vec<&str>) -> PyResult<HashMap<String, String>> {
        self.runtime.block_on(async {
            let mut results = HashMap::new();

            if paths.len() > self.parallel_threshold {
                // Parallel processing for large batches
                let futures: Vec<_> = paths.into_iter().map(|path| {
                    let path_owned = path.to_string();
                    async move {
                        let path_buf = match validate_path(&path_owned) {
                            Ok(p) => p,
                            Err(e) => return (path_owned, Err(e)),
                        };

                        let content = fs::read_to_string(&path_buf).await
                            .with_context(|| format!("Failed to read: {}", path_buf.display()));

                        (path_owned, content)
                    }
                }).collect();

                let batch_results = futures::future::join_all(futures).await;

                for (path, result) in batch_results {
                    match result {
                        Ok(content) => {
                            results.insert(path, content);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to read {}: {}", path, e);
                        }
                    }
                }
            } else {
                // Sequential processing for small batches
                for path in paths {
                    match self.read_file(path, "utf-8", None) {
                        Ok(content) => {
                            results.insert(path.to_string(), content);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to read {}: {}", path, e);
                        }
                    }
                }
            }

            increment_ops();
            Ok(results)
        })
    }

    /// Get operation statistics
    fn get_stats(&self) -> PyResult<HashMap<String, u64>> {
        self.runtime.block_on(async {
            let stats = self.stats.read().await;
            let mut result = HashMap::new();

            result.insert("reads".to_string(), stats.reads);
            result.insert("writes".to_string(), stats.writes);
            result.insert("deletes".to_string(), stats.deletes);
            result.insert("creates".to_string(), stats.creates);
            result.insert("bytes_read".to_string(), stats.bytes_read);
            result.insert("bytes_written".to_string(), stats.bytes_written);

            Ok(result)
        })
    }

    /// Clear statistics
    fn clear_stats(&self) -> PyResult<()> {
        self.runtime.block_on(async {
            let mut stats = self.stats.write().await;
            *stats = FileOpStats::default();
            Ok(())
        })
    }
}

impl RustFileOps {
    /// Read file in chunks for large files
    async fn read_file_chunked(&self, path: &Path, chunk_size: usize) -> Result<String> {
        let mut file = fs::File::open(path).await
            .with_context(|| format!("Failed to open file: {}", path.display()))?;

        let mut content = String::new();
        let mut buffer = vec![0u8; chunk_size];

        loop {
            let bytes_read = file.read(&mut buffer).await?;
            if bytes_read == 0 {
                break;
            }

            let chunk = String::from_utf8_lossy(&buffer[..bytes_read]);
            content.push_str(&chunk);
        }

        Ok(content)
    }

    /// List single directory
    async fn list_single(
        &self,
        path: &Path,
        pattern: Option<&str>,
        include_hidden: bool,
    ) -> Result<Vec<HashMap<String, PyObject>>> {
        let mut entries = Vec::new();
        let mut dir = fs::read_dir(path).await?;

        let pattern_regex = pattern.map(|p| regex::Regex::new(p)).transpose()?;

        while let Some(entry) = dir.next_entry().await? {
            let entry_path = entry.path();
            let name = entry_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            // Skip hidden files unless requested
            if !include_hidden && name.starts_with('.') {
                continue;
            }

            // Apply pattern filter
            if let Some(ref regex) = pattern_regex {
                if !regex.is_match(name) {
                    continue;
                }
            }

            let metadata = entry.metadata().await?;

            let mut info = HashMap::new();
            Python::with_gil(|py| {
                info.insert("path".to_string(), entry_path.to_string_lossy().to_string().into_py(py));
                info.insert("name".to_string(), name.into_py(py));
                info.insert("size".to_string(), metadata.len().into_py(py));
                info.insert("is_file".to_string(), metadata.is_file().into_py(py));
                info.insert("is_dir".to_string(), metadata.is_dir().into_py(py));
                info.insert("is_symlink".to_string(), metadata.is_symlink().into_py(py));
            });

            entries.push(info);
        }

        Ok(entries)
    }

    /// List directory recursively
    async fn list_recursive(
        &self,
        path: &Path,
        pattern: Option<&str>,
        include_hidden: bool,
    ) -> Result<Vec<HashMap<String, PyObject>>> {
        let mut entries = Vec::new();
        let pattern_regex = pattern.map(|p| regex::Regex::new(p)).transpose()?;

        for entry in WalkDir::new(path).into_iter() {
            let entry = entry?;
            let entry_path = entry.path();
            let name = entry_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            // Skip hidden files unless requested
            if !include_hidden && name.starts_with('.') {
                continue;
            }

            // Apply pattern filter
            if let Some(ref regex) = pattern_regex {
                if !regex.is_match(name) {
                    continue;
                }
            }

            let metadata = entry.metadata()?;

            let mut info = HashMap::new();
            Python::with_gil(|py| {
                info.insert("path".to_string(), entry_path.to_string_lossy().to_string().into_py(py));
                info.insert("name".to_string(), name.into_py(py));
                info.insert("size".to_string(), metadata.len().into_py(py));
                info.insert("is_file".to_string(), metadata.is_file().into_py(py));
                info.insert("is_dir".to_string(), metadata.is_dir().into_py(py));
                info.insert("is_symlink".to_string(), metadata.is_symlink().into_py(py));
                info.insert("depth".to_string(), entry.depth().into_py(py));
            });

            entries.push(info);
        }

        Ok(entries)
    }

    /// Copy directory recursively
    async fn copy_recursive(&self, src: &Path, dst: &Path) -> Result<u64> {
        let mut total_bytes = 0u64;

        fs::create_dir_all(dst).await?;

        for entry in WalkDir::new(src) {
            let entry = entry?;
            let src_path = entry.path();
            let rel_path = src_path.strip_prefix(src)?;
            let dst_path = dst.join(rel_path);

            if src_path.is_file() {
                if let Some(parent) = dst_path.parent() {
                    fs::create_dir_all(parent).await?;
                }
                let bytes = fs::copy(src_path, &dst_path).await?;
                total_bytes += bytes;
            } else if src_path.is_dir() {
                fs::create_dir_all(&dst_path).await?;
            }
        }

        Ok(total_bytes)
    }

    /// Convert notify Event to Python dict
    fn event_to_dict(&self, event: Event) -> HashMap<String, PyObject> {
        let mut dict = HashMap::new();

        Python::with_gil(|py| {
            dict.insert("kind".to_string(), format!("{:?}", event.kind).into_py(py));

            let paths: Vec<String> = event.paths.into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            dict.insert("paths".to_string(), paths.into_py(py));

            if let Some(tracker) = &event.attrs.tracker {
                dict.insert("tracker".to_string(), (*tracker as u64).into_py(py));
            }

            dict.insert("flag".to_string(), event.attrs.flag().map(|f| format!("{:?}", f)).into_py(py));
            dict.insert("info".to_string(), event.attrs.info().into_py(py));
            dict.insert("source".to_string(), event.attrs.source().map(|s| format!("{:?}", s)).into_py(py));
        });

        dict
    }
}

impl Drop for RustFileOps {
    fn drop(&mut self) {
        // Clean up any remaining watchers
        if let Ok(watchers) = self.watchers.try_read() {
            if !watchers.is_empty() {
                tracing::info!("Cleaning up {} file watchers", watchers.len());
            }
        }
    }
}