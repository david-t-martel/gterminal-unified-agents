//! Error types module
//!
//! This module defines the error types used throughout the fullstack agent filesystem operations.
//! All errors are serializable for transmission over the MCP protocol and
//! provide detailed context about what went wrong.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Represents all possible errors that can occur in filesystem operations.
///
/// This enum provides comprehensive error handling for all operations in the
/// filesystem server. Each variant includes relevant context information to
/// help diagnose issues. All errors are serializable for the MCP protocol.
#[derive(Error, Debug, Serialize, Deserialize)]
pub enum FsError {
    /// Generic I/O error with description and optional context
    #[error("I/O error: {message}{}", .context.as_ref().map(|c| format!(" ({})", c)).unwrap_or_default())]
    Io {
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        context: Option<String>,
    },

    /// Path does not exist in the filesystem
    #[error("Path does not exist: {0}")]
    PathNotFound(String),

    /// Path already exists when trying to create
    #[error("Path already exists: {0}")]
    PathExists(String),

    /// Expected a file but found something else
    #[error("Path is not a file: {0}")]
    NotAFile(String),

    /// Expected a directory but found something else
    #[error("Path is not a directory: {0}")]
    NotADirectory(String),

    /// WSL path translation command failed
    #[error("WSL command failed: {0}")]
    WslCommand(String),

    /// JSON serialization or deserialization failed
    #[error("JSON serialization/deserialization error: {0}")]
    Json(String),

    /// Unknown or unsupported command
    #[error("Invalid command: '{0}'")]
    InvalidCommand(String),

    /// Required parameter missing from request
    #[error("Missing required parameter: '{0}' for command '{1}'")]
    MissingParameter(String, String),

    /// Invalid regular expression pattern
    #[error("Search pattern is not valid regex: {0}")]
    InvalidRegex(String),

    /// Invalid glob pattern for file matching
    #[error("Find pattern is not a valid glob: {0}")]
    InvalidGlob(String),

    /// Executable not found in system PATH
    #[error("Command not found in PATH: {0}")]
    CommandNotFound(String),

    /// Command execution exceeded timeout
    #[error("Command '{0}' timed out after {1} seconds")]
    ExecTimeout(String, u64),

    /// Invalid line range for block operations
    #[error("Invalid line range: start_line must be less than or equal to end_line")]
    InvalidLineRange,

    /// Path traversal attempt - accessing paths outside allowed directories
    #[error("Access to path outside permitted root: {0}")]
    ForbiddenPath(String),

    /// Command execution error
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Invalid input provided to a function
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Catch-all for unexpected errors
    #[error("An unknown error occurred")]
    Unknown,
}

/// Type alias for Results with FsError as the error type.
///
/// This alias simplifies function signatures throughout the codebase by
/// avoiding the need to specify the full `std::result::Result<T, FsError>` type.
pub type FsResult<T> = std::result::Result<T, FsError>;

impl FsError {
    /// Create an I/O error with context
    pub fn io_with_context<S: Into<String>, C: Into<String>>(message: S, context: C) -> Self {
        FsError::Io {
            message: message.into(),
            context: Some(context.into()),
        }
    }

    /// Create an I/O error without context
    pub fn io<S: Into<String>>(message: S) -> Self {
        FsError::Io {
            message: message.into(),
            context: None,
        }
    }

    /// Add context to an existing error
    pub fn with_context<C: Into<String>>(self, context: C) -> Self {
        match self {
            FsError::Io { message, context: _ } => FsError::Io {
                message,
                context: Some(context.into()),
            },
            _ => self, // Other error types don't support context yet
        }
    }
}

/// Extension trait for adding context to standard errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn context<C: Into<String>>(self, context: C) -> FsResult<T>;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: std::fmt::Display,
{
    fn context<C: Into<String>>(self, context: C) -> FsResult<T> {
        self.map_err(|e| FsError::io_with_context(e.to_string(), context))
    }
}

impl<T> ErrorContext<T> for Option<T> {
    fn context<C: Into<String>>(self, context: C) -> FsResult<T> {
        self.ok_or_else(|| FsError::io(context))
    }
}

/// Implement conversion from std::io::Error to FsError
impl From<std::io::Error> for FsError {
    fn from(err: std::io::Error) -> Self {
        FsError::io(err.to_string())
    }
}
