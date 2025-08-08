/*!
 * Core types and data structures for gterminal-filewatcher
 *
 * Defines file events, analysis results, tool outputs, and API responses
 * optimized for high-performance processing and real-time updates.
 */

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

/// File change event from the file system watcher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEvent {
    /// Path of the changed file
    pub path: PathBuf,

    /// Type of file system event
    pub event_type: FileEventType,

    /// Timestamp when the event occurred
    pub timestamp: DateTime<Utc>,

    /// File size in bytes (if available)
    pub size: Option<u64>,

    /// File extension
    pub extension: Option<String>,

    /// Whether this event should trigger analysis
    pub should_analyze: bool,
}

/// Types of file system events
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FileEventType {
    /// File was created
    Create,
    /// File was modified
    Modify,
    /// File was deleted
    Delete,
    /// File was renamed
    Rename,
    /// File attributes changed
    Chmod,
}

/// Result of analyzing a file with tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// File that was analyzed
    pub file_path: PathBuf,

    /// When the analysis started
    pub start_time: DateTime<Utc>,

    /// How long the analysis took
    pub duration: Duration,

    /// Issues found during analysis
    pub issues: Vec<Issue>,

    /// Tool outputs
    pub tool_results: HashMap<String, ToolResult>,

    /// Overall analysis status
    pub status: AnalysisStatus,

    /// Analysis statistics
    pub stats: AnalysisStats,

    /// Fixes that were applied
    pub fixes_applied: Vec<Fix>,
}

/// Individual issue found during analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    /// Tool that found this issue
    pub tool: String,

    /// Issue severity
    pub severity: IssueSeverity,

    /// Issue message
    pub message: String,

    /// Line number (if applicable)
    pub line: Option<u32>,

    /// Column number (if applicable)
    pub column: Option<u32>,

    /// Error code (if applicable)
    pub code: Option<String>,

    /// Suggested fix (if available)
    pub suggestion: Option<String>,

    /// Whether this issue can be auto-fixed
    pub auto_fixable: bool,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Information/note
    Info,
    /// Warning
    Warning,
    /// Error that should be fixed
    Error,
    /// Critical error that breaks functionality
    Critical,
}

/// Result of running a specific tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Tool name
    pub tool: String,

    /// Exit code from tool execution
    pub exit_code: i32,

    /// Standard output
    pub stdout: String,

    /// Standard error
    pub stderr: String,

    /// Execution duration
    pub duration: Duration,

    /// Whether the tool ran successfully
    pub success: bool,

    /// Issues found by this tool
    pub issues: Vec<Issue>,

    /// Fixes applied by this tool
    pub fixes_applied: Vec<Fix>,
}

/// Analysis status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnalysisStatus {
    /// Analysis completed successfully
    Success,
    /// Analysis completed with warnings
    Warning,
    /// Analysis failed
    Error,
    /// Analysis was skipped
    Skipped,
    /// Analysis is still running
    InProgress,
}

/// Analysis statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStats {
    /// Number of tools run
    pub tools_run: usize,

    /// Number of tools that succeeded
    pub tools_succeeded: usize,

    /// Number of tools that failed
    pub tools_failed: usize,

    /// Total issues found
    pub total_issues: usize,

    /// Issues by severity
    pub issues_by_severity: HashMap<IssueSeverity, usize>,

    /// Total fixes applied
    pub fixes_applied: usize,

    /// Lines of code analyzed (if available)
    pub lines_of_code: Option<usize>,
}

/// A fix that was applied to a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fix {
    /// Tool that applied the fix
    pub tool: String,

    /// Description of the fix
    pub description: String,

    /// Line number where fix was applied
    pub line: Option<u32>,

    /// Column number where fix was applied
    pub column: Option<u32>,

    /// Old text that was replaced (if available)
    pub old_text: Option<String>,

    /// New text that replaced the old (if available)
    pub new_text: Option<String>,

    /// Type of fix applied
    pub fix_type: FixType,
}

/// Types of fixes that can be applied
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FixType {
    /// Formatting fix (whitespace, style)
    Format,
    /// Import organization
    ImportSort,
    /// Lint rule fix
    Lint,
    /// Security vulnerability fix
    Security,
    /// Performance optimization
    Performance,
    /// Type annotation fix
    Type,
    /// Documentation fix
    Documentation,
    /// Other/custom fix
    Other,
}

/// Batch processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Files processed in this batch
    pub files: Vec<PathBuf>,

    /// When batch processing started
    pub start_time: DateTime<Utc>,

    /// Total processing duration
    pub duration: Duration,

    /// Results for each file
    pub results: Vec<AnalysisResult>,

    /// Overall batch statistics
    pub stats: BatchStats,
}

/// Batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    /// Number of files processed
    pub files_processed: usize,

    /// Number of files that had issues
    pub files_with_issues: usize,

    /// Number of files that were auto-fixed
    pub files_auto_fixed: usize,

    /// Total issues found across all files
    pub total_issues: usize,

    /// Total fixes applied across all files
    pub total_fixes: usize,

    /// Average processing time per file
    pub avg_processing_time: Duration,

    /// Files processed per second
    pub files_per_second: f64,
}

/// Real-time dashboard update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardUpdate {
    /// Update type
    pub update_type: DashboardUpdateType,

    /// Timestamp of the update
    pub timestamp: DateTime<Utc>,

    /// Update payload
    pub data: DashboardData,
}

/// Types of dashboard updates
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DashboardUpdateType {
    /// File was changed
    FileChanged,
    /// Analysis started
    AnalysisStarted,
    /// Analysis completed
    AnalysisCompleted,
    /// Batch processing update
    BatchUpdate,
    /// Tool execution update
    ToolUpdate,
    /// System status update
    StatusUpdate,
    /// Error occurred
    Error,
}

/// Dashboard update data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DashboardData {
    /// File change data
    FileChange {
        file: PathBuf,
        event_type: FileEventType,
    },
    /// Analysis result data
    Analysis(AnalysisResult),
    /// Batch result data
    Batch(BatchResult),
    /// Tool execution data
    Tool {
        tool: String,
        file: PathBuf,
        result: ToolResult,
    },
    /// System status data
    Status(SystemStatus),
    /// Error data
    Error {
        message: String,
        file: Option<PathBuf>,
        tool: Option<String>,
    },
}

/// System status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    /// Whether the watcher is active
    pub watcher_active: bool,

    /// Number of files being watched
    pub files_watched: usize,

    /// Number of active processing jobs
    pub active_jobs: usize,

    /// Size of processing queue
    pub queue_size: usize,

    /// System resource usage
    pub resource_usage: ResourceUsage,

    /// Performance metrics
    pub performance: PerformanceMetrics,

    /// Tool availability
    pub tool_status: HashMap<String, bool>,
}

/// System resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage percentage
    pub cpu_percent: f64,

    /// Memory usage in bytes
    pub memory_bytes: u64,

    /// Memory usage percentage
    pub memory_percent: f64,

    /// Disk I/O statistics
    pub disk_io: DiskIoStats,

    /// Network I/O statistics (if applicable)
    pub network_io: Option<NetworkIoStats>,
}

/// Disk I/O statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIoStats {
    /// Bytes read from disk
    pub bytes_read: u64,

    /// Bytes written to disk
    pub bytes_written: u64,

    /// Number of read operations
    pub read_ops: u64,

    /// Number of write operations
    pub write_ops: u64,
}

/// Network I/O statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIoStats {
    /// Bytes received
    pub bytes_received: u64,

    /// Bytes sent
    pub bytes_sent: u64,

    /// Number of connections
    pub connections: u32,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total files processed since startup
    pub total_files_processed: usize,

    /// Total analysis time
    pub total_analysis_time: Duration,

    /// Average time per file
    pub avg_time_per_file: Duration,

    /// Files processed per second
    pub files_per_second: f64,

    /// Peak memory usage
    pub peak_memory_usage: u64,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// Error rate
    pub error_rate: f64,
}

/// HTTP API request types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRequest {
    /// Request ID for tracking
    pub id: Option<String>,

    /// Request timestamp
    pub timestamp: DateTime<Utc>,

    /// Request data
    pub data: ApiRequestData,
}

/// API request data types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ApiRequestData {
    /// Analyze a specific file
    AnalyzeFile { file: PathBuf },

    /// Get system status
    GetStatus,

    /// Trigger auto-fix for file
    AutoFix { file: PathBuf },

    /// Run specific tool on file
    RunTool { tool: String, file: PathBuf },

    /// Get configuration
    GetConfig,

    /// Update configuration
    UpdateConfig { config: serde_json::Value },

    /// Pause/resume watching
    SetWatching { enabled: bool },

    /// Get performance metrics
    GetMetrics,
}

/// HTTP API response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    /// Request ID (if provided)
    pub id: Option<String>,

    /// Response timestamp
    pub timestamp: DateTime<Utc>,

    /// Whether the request succeeded
    pub success: bool,

    /// Response data
    pub data: Option<ApiResponseData>,

    /// Error message (if success = false)
    pub error: Option<String>,
}

/// API response data types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ApiResponseData {
    /// Analysis result
    Analysis(AnalysisResult),

    /// System status
    Status(SystemStatus),

    /// Configuration
    Config(serde_json::Value),

    /// Performance metrics
    Metrics(PerformanceMetrics),

    /// Simple success message
    Message(String),

    /// List of files
    Files(Vec<PathBuf>),
}

// Utility implementations
impl FileEvent {
    /// Create a new file event
    pub fn new(path: PathBuf, event_type: FileEventType) -> Self {
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_string());

        Self {
            path,
            event_type,
            timestamp: Utc::now(),
            size: None,
            extension,
            should_analyze: true,
        }
    }

    /// Get file extension without the dot
    pub fn extension(&self) -> Option<&str> {
        self.extension.as_deref()
    }
}

impl AnalysisResult {
    /// Create a new analysis result
    pub fn new(file_path: PathBuf) -> Self {
        Self {
            file_path,
            start_time: Utc::now(),
            duration: Duration::ZERO,
            issues: Vec::new(),
            tool_results: HashMap::new(),
            status: AnalysisStatus::InProgress,
            stats: AnalysisStats::default(),
            fixes_applied: Vec::new(),
        }
    }

    /// Mark analysis as completed and calculate final stats
    pub fn complete(&mut self) {
        self.duration = Utc::now().signed_duration_since(self.start_time)
            .to_std()
            .unwrap_or(Duration::ZERO);

        // Calculate final statistics
        self.stats.tools_run = self.tool_results.len();
        self.stats.tools_succeeded = self.tool_results.values()
            .filter(|r| r.success)
            .count();
        self.stats.tools_failed = self.stats.tools_run - self.stats.tools_succeeded;
        self.stats.total_issues = self.issues.len();
        self.stats.fixes_applied = self.fixes_applied.len();

        // Count issues by severity
        for issue in &self.issues {
            *self.stats.issues_by_severity.entry(issue.severity).or_insert(0) += 1;
        }

        // Determine overall status
        self.status = if self.stats.tools_failed > 0 {
            AnalysisStatus::Error
        } else if self.stats.total_issues > 0 {
            AnalysisStatus::Warning
        } else {
            AnalysisStatus::Success
        };
    }

    /// Check if analysis has critical issues
    pub fn has_critical_issues(&self) -> bool {
        self.issues.iter().any(|i| i.severity == IssueSeverity::Critical)
    }

    /// Get issues by severity
    pub fn issues_by_severity(&self, severity: IssueSeverity) -> Vec<&Issue> {
        self.issues.iter().filter(|i| i.severity == severity).collect()
    }
}

impl Default for AnalysisStats {
    fn default() -> Self {
        Self {
            tools_run: 0,
            tools_succeeded: 0,
            tools_failed: 0,
            total_issues: 0,
            issues_by_severity: HashMap::new(),
            fixes_applied: 0,
            lines_of_code: None,
        }
    }
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueSeverity::Info => write!(f, "INFO"),
            IssueSeverity::Warning => write!(f, "WARN"),
            IssueSeverity::Error => write!(f, "ERROR"),
            IssueSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl std::fmt::Display for FileEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileEventType::Create => write!(f, "CREATE"),
            FileEventType::Modify => write!(f, "MODIFY"),
            FileEventType::Delete => write!(f, "DELETE"),
            FileEventType::Rename => write!(f, "RENAME"),
            FileEventType::Chmod => write!(f, "CHMOD"),
        }
    }
}

// Serde custom serialization for Duration
mod duration_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

// Apply custom serialization to Duration fields
impl Serialize for AnalysisResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("AnalysisResult", 7)?;
        state.serialize_field("file_path", &self.file_path)?;
        state.serialize_field("start_time", &self.start_time)?;
        state.serialize_field("duration_ms", &self.duration.as_millis())?;
        state.serialize_field("issues", &self.issues)?;
        state.serialize_field("tool_results", &self.tool_results)?;
        state.serialize_field("status", &self.status)?;
        state.serialize_field("stats", &self.stats)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_event_creation() {
        let path = PathBuf::from("test.py");
        let event = FileEvent::new(path.clone(), FileEventType::Modify);

        assert_eq!(event.path, path);
        assert_eq!(event.event_type, FileEventType::Modify);
        assert_eq!(event.extension(), Some("py"));
        assert!(event.should_analyze);
    }

    #[test]
    fn test_analysis_result_completion() {
        let mut result = AnalysisResult::new(PathBuf::from("test.py"));

        result.issues.push(Issue {
            tool: "ruff".to_string(),
            severity: IssueSeverity::Error,
            message: "Test error".to_string(),
            line: Some(10),
            column: Some(5),
            code: Some("E001".to_string()),
            suggestion: None,
            auto_fixable: true,
        });

        result.complete();

        assert_eq!(result.status, AnalysisStatus::Warning);
        assert_eq!(result.stats.total_issues, 1);
        assert_eq!(result.stats.issues_by_severity[&IssueSeverity::Error], 1);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(IssueSeverity::Info < IssueSeverity::Warning);
        assert!(IssueSeverity::Warning < IssueSeverity::Error);
        assert!(IssueSeverity::Error < IssueSeverity::Critical);
    }

    #[test]
    fn test_serialization() {
        let event = FileEvent::new(PathBuf::from("test.py"), FileEventType::Create);
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: FileEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(event.path, deserialized.path);
        assert_eq!(event.event_type, deserialized.event_type);
    }
}
