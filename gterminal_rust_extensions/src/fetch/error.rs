//! Error types for the fetch server

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, FetchError>;

#[derive(Error, Debug)]
pub enum FetchError {
    #[error("Network error: {0}")]
    Network(String),

    #[error("Request timeout after {0} seconds")]
    Timeout(u64),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("HTTP error: {status} - {message}")]
    Http { status: u16, message: String },

    #[error("Failed to decode response: {0}")]
    Decode(String),

    #[error("Too many redirects (max: {0})")]
    TooManyRedirects(usize),

    #[error("Authentication required")]
    AuthRequired,

    #[error("Rate limited: {0}")]
    RateLimit(String),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Proxy error: {0}")]
    Proxy(String),

    #[error("Invalid command: '{0}'")]
    InvalidCommand(String),

    #[error("Missing required parameter: '{0}' for command '{1}'")]
    MissingParameter(String, String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Other error: {0}")]
    Other(String),
}

impl Serialize for FetchError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("FetchError", 2)?;
        state.serialize_field("type", &format!("{self:?}"))?;
        state.serialize_field("message", &self.to_string())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for FetchError {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ErrorData {
            message: String,
        }

        let data = ErrorData::deserialize(deserializer)?;
        Ok(FetchError::Other(data.message))
    }
}

impl From<FetchError> for PyErr {
    fn from(err: FetchError) -> Self {
        match err {
            FetchError::Network(msg) => PyErr::new::<pyo3::exceptions::PyConnectionError, _>(msg),
            FetchError::Timeout(secs) => PyErr::new::<pyo3::exceptions::PyTimeoutError, _>(
                format!("Request timeout after {} seconds", secs)
            ),
            FetchError::InvalidUrl(msg) => PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid URL: {}", msg)
            ),
            FetchError::Http { status, message } => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("HTTP error: {} - {}", status, message)
            ),
            FetchError::Decode(msg) => PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to decode response: {}", msg)
            ),
            FetchError::TooManyRedirects(max) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Too many redirects (max: {})", max)
            ),
            FetchError::AuthRequired => PyErr::new::<pyo3::exceptions::PyPermissionError, _>(
                "Authentication required"
            ),
            FetchError::RateLimit(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Rate limited: {}", msg)
            ),
            FetchError::Config(msg) => PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid configuration: {}", msg)
            ),
            FetchError::Proxy(msg) => PyErr::new::<pyo3::exceptions::PyConnectionError, _>(
                format!("Proxy error: {}", msg)
            ),
            FetchError::InvalidCommand(cmd) => PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid command: '{}'", cmd)
            ),
            FetchError::MissingParameter(param, cmd) => PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Missing required parameter: '{}' for command '{}'", param, cmd)
            ),
            FetchError::Io(err) => PyErr::new::<pyo3::exceptions::PyIOError, _>(err.to_string()),
            FetchError::Reqwest(err) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()),
            FetchError::Json(err) => PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()),
            FetchError::Other(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg),
        }
    }
}
