/*!
 * High-performance file watching engine for gterminal-filewatcher
 *
 * Provides real-time file monitoring with intelligent debouncing, batch processing,
 * and integrated tool execution for Python, TypeScript, Rust, and other file types.
 */

use crate::config::Config;
use crate::tools::ToolExecutor;
use crate::types::*;

use anyhow::{Context, Result};
use crossbeam::channel::{select, unbounded, Receiver, Sender};
use dashmap::DashMap;
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, trace, warn};
use walkdir::WalkDir;

/// High-performance file watching engine
pub struct WatchEngine {
    /// Project root directory
    project_path: PathBuf,

    /// Configuration
    config: Config,

    /// Tool executor
    tool_executor: Arc<ToolExecutor>,

    /// Debouncing cache (file_path -> last_event_time)
    debounce_cache: Arc<DashMap<PathBuf, Instant>>,

    /// Processing queue (file_path -> file_event)
    processing_queue: Arc<DashMap<PathBuf, FileEvent>>,

    /// Active processing jobs counter
    active_jobs: Arc<AtomicUsize>,

    /// Whether to enable auto-fixing
    auto_fix_enabled: bool,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Performance metrics
    metrics: Arc<PerformanceTracker>,

    /// Event broadcasters for real-time updates
    dashboard_tx: Option<Sender<DashboardUpdate>>,
}

/// Performance tracking for the watch engine
#[derive(Debug)]
pub struct PerformanceTracker {
    /// Total files processed
    pub files_processed: AtomicUsize,

    /// Total analysis time
    pub total_analysis_time_ms: AtomicUsize,

    /// Peak memory usage in bytes
    pub peak_memory_bytes: AtomicUsize,

    /// Cache hits
    pub cache_hits: AtomicUsize,

    /// Cache misses
    pub cache_misses: AtomicUsize,

    /// Errors encountered
    pub error_count: AtomicUsize,

    /// Start time for calculating rates
    pub start_time: Instant,
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self {
            files_processed: AtomicUsize::new(0),
            total_analysis_time_ms: AtomicUsize::new(0),
            peak_memory_bytes: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            error_count: AtomicUsize::new(0),
            start_time: Instant::now(),
        }
    }
}

impl PerformanceTracker {
    /// Get current performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        let elapsed = self.start_time.elapsed();
        let files_processed = self.files_processed.load(Ordering::Relaxed);
        let total_time_ms = self.total_analysis_time_ms.load(Ordering::Relaxed);
        let errors = self.error_count.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);

        let files_per_second = if elapsed.as_secs_f64() > 0.0 {
            files_processed as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let avg_time_per_file = if files_processed > 0 {
            Duration::from_millis(total_time_ms as u64 / files_processed as u64)
        } else {
            Duration::ZERO
        };

        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64
        } else {
            0.0
        };

        let error_rate = if files_processed > 0 {
            errors as f64 / files_processed as f64
        } else {
            0.0
        };

        PerformanceMetrics {
            total_files_processed: files_processed,
            total_analysis_time: Duration::from_millis(total_time_ms as u64),
            avg_time_per_file,
            files_per_second,
            peak_memory_usage: self.peak_memory_bytes.load(Ordering::Relaxed) as u64,
            cache_hit_rate,
            error_rate,
        }
    }

    /// Record file processing
    pub fn record_file_processed(&self, duration: Duration) {
        self.files_processed.fetch_add(1, Ordering::Relaxed);
        self.total_analysis_time_ms.fetch_add(duration.as_millis() as usize, Ordering::Relaxed);
    }

    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record error
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }
}

impl WatchEngine {
    /// Create a new watch engine
    pub fn new(project_path: PathBuf, config: Config, auto_fix_enabled: bool) -> Result<Self> {
        let tool_executor = Arc::new(ToolExecutor::new(config.clone())?);

        Ok(Self {
            project_path,
            config,
            tool_executor,
            debounce_cache: Arc::new(DashMap::with_capacity(1000)),
            processing_queue: Arc::new(DashMap::with_capacity(100)),
            active_jobs: Arc::new(AtomicUsize::new(0)),
            auto_fix_enabled,
            shutdown: Arc::new(AtomicBool::new(false)),
            metrics: Arc::new(PerformanceTracker::default()),
            dashboard_tx: None,
        })
    }

    /// Set dashboard update sender
    pub fn set_dashboard_sender(&mut self, sender: Sender<DashboardUpdate>) {
        self.dashboard_tx = Some(sender);
    }

    /// Start the watch engine
    pub async fn start(&mut self) -> Result<()> {
        info!("🚀 Starting gterminal-filewatcher engine");
        info!("📁 Project path: {}", self.project_path.display());
        info!("⚡ Auto-fix enabled: {}", self.auto_fix_enabled);

        // Perform initial scan
        self.initial_scan().await?;

        // Set up file system watcher
        let (tx, rx) = unbounded();
        let mut watcher = RecommendedWatcher::new(
            move |res| {
                if let Err(e) = tx.send(res) {
                    error!("Failed to send file system event: {}", e);
                }
            },
            notify::Config::default()
                .with_poll_interval(Duration::from_millis(50))
                .with_compare_contents(true), // More accurate change detection
        )
        .context("Failed to create file system watcher")?;

        watcher.watch(&self.project_path, RecursiveMode::Recursive)
            .with_context(|| format!("Failed to watch directory: {}", self.project_path.display()))?;

        // Start background processors
        self.spawn_processors().await;

        // Start performance monitoring
        self.spawn_performance_monitor().await;

        info!("✅ Watch engine started successfully");

        // Main event loop
        loop {
            select! {
                recv(rx) -> event => {
                    match event {
                        Ok(Ok(notify_event)) => {
                            self.handle_file_system_event(notify_event).await;
                        }
                        Ok(Err(e)) => {
                            warn!("File system watcher error: {}", e);
                            self.metrics.record_error();
                        }
                        Err(_) => {
                            info!("File system watcher channel closed");
                            break;
                        }
                    }
                }
            }

            if self.shutdown.load(Ordering::Relaxed) {
                break;
            }
        }

        info!("🛑 Watch engine stopped");
        Ok(())
    }

    /// Perform initial project scan
    async fn initial_scan(&self) -> Result<()> {
        info!("🔍 Starting initial project scan...");
        let start_time = Instant::now();

        // Collect files to scan
        let files: Vec<PathBuf> = WalkDir::new(&self.project_path)
            .into_iter()
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    if e.file_type().is_file() {
                        let path = e.path().to_path_buf();
                        if self.should_process_file(&path) {
                            Some(path)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
            })
            .collect();

        info!("📊 Found {} files to analyze", files.len());

        // Process files in parallel batches
        let batch_size = self.config.performance.batch_size;
        let batches: Vec<_> = files.chunks(batch_size).collect();

        // Process batches with controlled parallelism
        for (batch_idx, batch) in batches.iter().enumerate() {
            info!("🔄 Processing batch {}/{} ({} files)",
                batch_idx + 1, batches.len(), batch.len());

            let results: Vec<_> = batch
                .par_iter()
                .map(|file_path| {
                    let start = Instant::now();
                    let result = self.process_file_sync(file_path);
                    let duration = start.elapsed();
                    self.metrics.record_file_processed(duration);
                    result
                })
                .collect();

            // Log batch results
            let successful = results.iter().filter(|r| r.is_ok()).count();
            let failed = results.len() - successful;

            if failed > 0 {
                warn!("Batch {}: {} succeeded, {} failed", batch_idx + 1, successful, failed);
            } else {
                debug!("Batch {}: {} files processed successfully", batch_idx + 1, successful);
            }

            // Small delay between batches to prevent overwhelming the system
            if batch_idx + 1 < batches.len() {
                sleep(Duration::from_millis(10)).await;
            }
        }

        let duration = start_time.elapsed();
        info!("✅ Initial scan completed in {:?}", duration);
        self.print_performance_summary();

        Ok(())
    }

    /// Handle file system events
    async fn handle_file_system_event(&self, event: Event) {
        use notify::EventKind;

        // Filter relevant events
        let event_type = match event.kind {
            EventKind::Create(_) => FileEventType::Create,
            EventKind::Modify(_) => FileEventType::Modify,
            EventKind::Remove(_) => FileEventType::Delete,
            EventKind::Other => return, // Skip other events
            _ => return,
        };

        // Process each path in the event
        for path in event.paths {
            if !self.should_process_file(&path) {
                continue;
            }

            // Apply debouncing
            let now = Instant::now();
            let should_queue = if let Some(last_event) = self.debounce_cache.get(&path) {
                let time_since_last = now.duration_since(*last_event);
                if time_since_last >= Duration::from_millis(self.config.watch.debounce_ms) {
                    self.metrics.record_cache_miss();
                    true
                } else {
                    self.metrics.record_cache_hit();
                    false
                }
            } else {
                self.metrics.record_cache_miss();
                true
            };

            if should_queue {
                // Update debounce cache
                self.debounce_cache.insert(path.clone(), now);

                // Create file event
                let file_event = FileEvent::new(path.clone(), event_type);

                // Queue for processing
                self.processing_queue.insert(path.clone(), file_event.clone());

                debug!("📝 Queued file for processing: {:?} ({})", path, event_type);

                // Send dashboard update
                if let Some(ref tx) = self.dashboard_tx {
                    let update = DashboardUpdate {
                        update_type: DashboardUpdateType::FileChanged,
                        timestamp: chrono::Utc::now(),
                        data: DashboardData::FileChange {
                            file: path.clone(),
                            event_type,
                        },
                    };

                    if let Err(e) = tx.try_send(update) {
                        warn!("Failed to send dashboard update: {}", e);
                    }
                }
            }
        }
    }

    /// Check if a file should be processed
    fn should_process_file(&self, path: &Path) -> bool {
        // Check if path should be ignored
        if self.config.should_ignore_path(path) {
            return false;
        }

        // Check file extension
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            self.config.should_watch_extension(extension)
        } else {
            false
        }
    }

    /// Spawn background processors
    async fn spawn_processors(&self) {
        let processing_queue = self.processing_queue.clone();
        let active_jobs = self.active_jobs.clone();
        let shutdown = self.shutdown.clone();
        let max_jobs = self.config.performance.max_parallel_jobs;
        let process_interval = self.config.performance.process_interval_ms;
        let batch_size = self.config.performance.batch_size;
        let tool_executor = self.tool_executor.clone();
        let auto_fix = self.auto_fix_enabled;
        let dashboard_tx = self.dashboard_tx.clone();
        let metrics = self.metrics.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(process_interval));

            loop {
                interval.tick().await;

                if shutdown.load(Ordering::Relaxed) {
                    break;
                }

                let current_jobs = active_jobs.load(Ordering::Relaxed);
                if current_jobs >= max_jobs {
                    continue;
                }

                // Collect batch of files to process
                let mut batch: Vec<(PathBuf, FileEvent)> = Vec::new();
                let mut processed_keys = Vec::new();

                for entry in processing_queue.iter().take(batch_size) {
                    batch.push((entry.key().clone(), entry.value().clone()));
                    processed_keys.push(entry.key().clone());
                }

                // Remove processed items from queue
                for key in &processed_keys {
                    processing_queue.remove(key);
                }

                if !batch.is_empty() {
                    // Spawn batch processing task
                    let tool_executor_clone = tool_executor.clone();
                    let active_jobs_clone = active_jobs.clone();
                    let dashboard_tx_clone = dashboard_tx.clone();
                    let metrics_clone = metrics.clone();

                    tokio::spawn(async move {
                        active_jobs_clone.fetch_add(1, Ordering::Relaxed);

                        // Process batch in parallel
                        let results: Vec<_> = batch
                            .into_par_iter()
                            .map(|(path, _event)| {
                                let start = Instant::now();
                                let result = tool_executor_clone.analyze_file(&path, auto_fix);
                                let duration = start.elapsed();

                                metrics_clone.record_file_processed(duration);

                                match result {
                                    Ok(analysis_result) => {
                                        // Send dashboard update
                                        if let Some(ref tx) = dashboard_tx_clone {
                                            let update = DashboardUpdate {
                                                update_type: DashboardUpdateType::AnalysisCompleted,
                                                timestamp: chrono::Utc::now(),
                                                data: DashboardData::Analysis(analysis_result),
                                            };

                                            let _ = tx.try_send(update);
                                        }
                                        Ok(())
                                    }
                                    Err(e) => {
                                        metrics_clone.record_error();
                                        error!("Failed to analyze file {:?}: {}", path, e);
                                        Err(e)
                                    }
                                }
                            })
                            .collect();

                        let successful = results.iter().filter(|r| r.is_ok()).count();
                        let failed = results.len() - successful;

                        if failed > 0 {
                            warn!("Batch processing: {} succeeded, {} failed", successful, failed);
                        } else {
                            debug!("Batch processing: {} files processed successfully", successful);
                        }

                        active_jobs_clone.fetch_sub(1, Ordering::Relaxed);
                    });
                }
            }
        });
    }

    /// Spawn performance monitoring task
    async fn spawn_performance_monitor(&self) {
        let metrics = self.metrics.clone();
        let dashboard_tx = self.dashboard_tx.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30)); // Report every 30 seconds

            loop {
                interval.tick().await;

                if shutdown.load(Ordering::Relaxed) {
                    break;
                }

                let performance_metrics = metrics.get_metrics();

                info!("📊 Performance: {:.2} files/sec, {:.1}% cache hit rate, {:.1}% error rate",
                    performance_metrics.files_per_second,
                    performance_metrics.cache_hit_rate * 100.0,
                    performance_metrics.error_rate * 100.0
                );

                // Send dashboard update
                if let Some(ref tx) = dashboard_tx {
                    let update = DashboardUpdate {
                        update_type: DashboardUpdateType::StatusUpdate,
                        timestamp: chrono::Utc::now(),
                        data: DashboardData::Status(SystemStatus {
                            watcher_active: true,
                            files_watched: 0, // TODO: Track this
                            active_jobs: 0,   // TODO: Track this
                            queue_size: 0,    // TODO: Track this
                            resource_usage: ResourceUsage {
                                cpu_percent: 0.0,
                                memory_bytes: performance_metrics.peak_memory_usage,
                                memory_percent: 0.0,
                                disk_io: DiskIoStats {
                                    bytes_read: 0,
                                    bytes_written: 0,
                                    read_ops: 0,
                                    write_ops: 0,
                                },
                                network_io: None,
                            },
                            performance: performance_metrics,
                            tool_status: HashMap::new(), // TODO: Track tool status
                        }),
                    };

                    let _ = tx.try_send(update);
                }
            }
        });
    }

    /// Process a single file synchronously (for initial scan)
    fn process_file_sync(&self, file_path: &Path) -> Result<()> {
        trace!("🔍 Analyzing file: {}", file_path.display());

        let result = self.tool_executor.analyze_file(file_path, self.auto_fix_enabled)?;

        if result.has_critical_issues() {
            warn!("💥 Critical issues found in {}: {} issues",
                file_path.display(), result.issues.len());
        } else if !result.issues.is_empty() {
            debug!("⚠️  Issues found in {}: {} issues",
                file_path.display(), result.issues.len());
        }

        Ok(())
    }

    /// Analyze a single file (public API)
    pub async fn analyze_file(&mut self, file_path: &Path) -> Result<AnalysisResult> {
        if !self.should_process_file(file_path) {
            anyhow::bail!("File {:?} is not eligible for analysis", file_path);
        }

        let start_time = Instant::now();
        let result = self.tool_executor.analyze_file(file_path, self.auto_fix_enabled)?;
        let duration = start_time.elapsed();

        self.metrics.record_file_processed(duration);

        Ok(result)
    }

    /// Print performance summary
    fn print_performance_summary(&self) {
        let metrics = self.metrics.get_metrics();

        info!("📈 Performance Summary:");
        info!("  • Files processed: {}", metrics.total_files_processed);
        info!("  • Processing rate: {:.2} files/second", metrics.files_per_second);
        info!("  • Average time per file: {:?}", metrics.avg_time_per_file);
        info!("  • Cache hit rate: {:.1}%", metrics.cache_hit_rate * 100.0);
        info!("  • Error rate: {:.1}%", metrics.error_rate * 100.0);

        if metrics.peak_memory_usage > 0 {
            let mb = metrics.peak_memory_usage as f64 / 1_048_576.0;
            info!("  • Peak memory usage: {:.1} MB", mb);
        }
    }

    /// Get current system status
    pub fn get_system_status(&self) -> SystemStatus {
        let performance_metrics = self.metrics.get_metrics();

        SystemStatus {
            watcher_active: !self.shutdown.load(Ordering::Relaxed),
            files_watched: self.debounce_cache.len(),
            active_jobs: self.active_jobs.load(Ordering::Relaxed),
            queue_size: self.processing_queue.len(),
            resource_usage: ResourceUsage {
                cpu_percent: 0.0, // TODO: Implement CPU monitoring
                memory_bytes: performance_metrics.peak_memory_usage,
                memory_percent: 0.0,
                disk_io: DiskIoStats {
                    bytes_read: 0,
                    bytes_written: 0,
                    read_ops: 0,
                    write_ops: 0,
                },
                network_io: None,
            },
            performance: performance_metrics,
            tool_status: self.tool_executor.get_tool_status(),
        }
    }

    /// Shutdown the watch engine gracefully
    pub fn shutdown(&self) {
        info!("🛑 Initiating graceful shutdown");
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

// Cleanup on drop
impl Drop for WatchEngine {
    fn drop(&mut self) {
        self.shutdown();
        self.print_performance_summary();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    #[tokio::test]
    async fn test_watch_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = Config::default();

        let engine = WatchEngine::new(temp_dir.path().to_path_buf(), config, true);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_file_filtering() {
        let temp_dir = TempDir::new().unwrap();
        let config = Config::default();
        let engine = WatchEngine::new(temp_dir.path().to_path_buf(), config, true).unwrap();

        assert!(engine.should_process_file(Path::new("test.py")));
        assert!(engine.should_process_file(Path::new("src/main.rs")));
        assert!(!engine.should_process_file(Path::new("node_modules/test.js")));
        assert!(!engine.should_process_file(Path::new("test.exe")));
    }

    #[test]
    fn test_performance_tracker() {
        let tracker = PerformanceTracker::default();

        tracker.record_file_processed(Duration::from_millis(100));
        tracker.record_cache_hit();
        tracker.record_cache_miss();

        let metrics = tracker.get_metrics();
        assert_eq!(metrics.total_files_processed, 1);
        assert_eq!(metrics.cache_hit_rate, 0.5);
    }

    #[tokio::test]
    async fn test_initial_scan() {
        let temp_dir = TempDir::new().unwrap();
        let project_path = temp_dir.path();

        // Create test files
        fs::write(project_path.join("test.py"), "print('hello')").unwrap();
        fs::write(project_path.join("main.rs"), "fn main() {}").unwrap();
        fs::create_dir(project_path.join("src")).unwrap();
        fs::write(project_path.join("src/lib.py"), "def test(): pass").unwrap();

        let config = Config::default();
        let engine = WatchEngine::new(project_path.to_path_buf(), config, false);

        assert!(engine.is_ok());
        // Initial scan test would require mock tool executor
    }
}
