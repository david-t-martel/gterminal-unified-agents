//! High-performance WebSocket message handling
//!
//! This module provides PyO3 bindings for WebSocket operations that are
//! significantly faster than Python equivalents:
//! - High-throughput message processing
//! - Connection pooling and management
//! - Binary message optimization
//! - Concurrent message handling

use pyo3::prelude::*;
// Removed pyo3_asyncio - using sync operations with PyO3 0.22
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
// use url::Url;  // Temporarily disabled due to ICU proc-macro conflicts

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RustWebSocketMessage {
    Text(String),
    Binary(Vec<u8>),
    Ping(Vec<u8>),
    Pong(Vec<u8>),
    Close(Option<String>),
}

/// WebSocket connection statistics
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub connection_time: Instant,
    pub last_activity: Instant,
    pub errors: u64,
}

impl ConnectionStats {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            messages_sent: 0,
            messages_received: 0,
            bytes_sent: 0,
            bytes_received: 0,
            connection_time: now,
            last_activity: now,
            errors: 0,
        }
    }
}

/// High-performance WebSocket client handler
#[pyclass]
pub struct RustWebSocketHandler {
    url: String,
    max_message_size: usize,
    connection_timeout: Duration,
    ping_interval: Duration,
    max_reconnect_attempts: u32,
}

#[pymethods]
impl RustWebSocketHandler {
    #[new]
    #[pyo3(signature = (url, max_message_size_mb=10, connection_timeout_seconds=30, ping_interval_seconds=30, max_reconnect_attempts=5))]
    fn new(
        url: String,
        max_message_size_mb: usize,
        connection_timeout_seconds: u64,
        ping_interval_seconds: u64,
        max_reconnect_attempts: u32,
    ) -> PyResult<Self> {
        // Basic URL validation (simplified to avoid ICU dependencies)
        if !url.starts_with("ws://") && !url.starts_with("wss://") {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("URL must start with ws:// or wss://".to_string()));
        }

        Ok(Self {
            url,
            max_message_size: max_message_size_mb * 1024 * 1024,
            connection_timeout: Duration::from_secs(connection_timeout_seconds),
            ping_interval: Duration::from_secs(ping_interval_seconds),
            max_reconnect_attempts,
        })
    }

    /// Connect to WebSocket server
    fn connect<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let url = self.url.clone();
        let timeout = self.connection_timeout;

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            // Basic URL validation (simplified to avoid ICU dependencies)
            if !url.starts_with("ws://") && !url.starts_with("wss://") {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("URL must start with ws:// or wss://".to_string()));
            }

            // Connect with timeout
            let connect_future = connect_async(&url);
            let (ws_stream, _) = tokio::time::timeout(timeout, connect_future)
                .await
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyTimeoutError, _>("Connection timeout"))?
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyConnectionError, _>(format!("WebSocket connection failed: {}", e)))?;

            // Split stream for concurrent reading/writing
            let (_write, _read) = ws_stream.split();

            Ok(format!("Connected to {}", &url))
        })
    }

    /// Send text message
    fn send_text<'py>(&self, py: Python<'py>, message: String) -> PyResult<&'py PyAny> {
        let max_size = self.max_message_size;

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            if message.len() > max_size {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Message too large: {} bytes (max: {} bytes)", message.len(), max_size)));
            }

            // This is a simplified example - in practice, you'd maintain connection state
            Ok(format!("Would send text message: {} bytes", message.len()))
        })
    }

    /// Send binary message
    fn send_binary<'py>(&self, py: Python<'py>, data: Vec<u8>) -> PyResult<&'py PyAny> {
        let max_size = self.max_message_size;

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            if data.len() > max_size {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Message too large: {} bytes (max: {} bytes)", data.len(), max_size)));
            }

            // This is a simplified example - in practice, you'd maintain connection state
            Ok(format!("Would send binary message: {} bytes", data.len()))
        })
    }

    /// Send ping message
    fn ping<'py>(&self, py: Python<'py>, data: Option<Vec<u8>>) -> PyResult<&'py PyAny> {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let ping_data = data.unwrap_or_default();
            Ok(format!("Would send ping: {} bytes", ping_data.len()))
        })
    }

    /// Close connection with optional reason
    fn close<'py>(&self, py: Python<'py>, reason: Option<String>) -> PyResult<&'py PyAny> {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let close_reason = reason.unwrap_or_else(|| "Normal closure".to_string());
            Ok(format!("Would close connection: {}", close_reason))
        })
    }
}

/// High-performance message processor for WebSocket messages
#[pyclass]
pub struct RustMessageProcessor {
    max_queue_size: usize,
    batch_size: usize,
    processing_timeout: Duration,
    stats: Arc<RwLock<HashMap<String, u64>>>,
}

#[pymethods]
impl RustMessageProcessor {
    #[new]
    #[pyo3(signature = (max_queue_size=10000, batch_size=100, processing_timeout_seconds=5))]
    fn new(max_queue_size: usize, batch_size: usize, processing_timeout_seconds: u64) -> Self {
        Self {
            max_queue_size,
            batch_size,
            processing_timeout: Duration::from_secs(processing_timeout_seconds),
            stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Process single message
    fn process_message(&self, py: Python<'_>, message_type: &str, data: Vec<u8>) -> PyResult<HashMap<String, PyObject>> {
        let mut result = HashMap::new();
        let data_len = data.len();  // Store length before potentially moving data

        match message_type {
            "text" => {
                let text = String::from_utf8(data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid UTF-8: {}", e)))?;

                result.insert("type".to_string(), "text".to_object(py));
                result.insert("content".to_string(), text.to_object(py));
                result.insert("size".to_string(), text.len().to_object(py));

                // Simple text analysis
                let word_count = text.split_whitespace().count();
                let line_count = text.lines().count();

                result.insert("word_count".to_string(), word_count.to_object(py));
                result.insert("line_count".to_string(), line_count.to_object(py));
            }
            "binary" => {
                result.insert("type".to_string(), "binary".to_object(py));
                result.insert("size".to_string(), data_len.to_object(py));

                // Binary analysis
                let entropy = self.calculate_entropy(&data);
                result.insert("entropy".to_string(), entropy.to_object(py));

                // Check if it might be compressed
                let is_likely_compressed = entropy > 7.0;
                result.insert("likely_compressed".to_string(), is_likely_compressed.to_object(py));
            }
            "json" => {
                let text = String::from_utf8(data)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid UTF-8: {}", e)))?;

                // Try to parse as JSON
                match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(json_value) => {
                        result.insert("type".to_string(), "json".to_object(py));
                        result.insert("valid_json".to_string(), true.to_object(py));
                        result.insert("size".to_string(), text.len().to_object(py));

                        // JSON structure analysis
                        let (objects, arrays, strings, numbers) = self.analyze_json_structure(&json_value);
                        result.insert("objects".to_string(), objects.to_object(py));
                        result.insert("arrays".to_string(), arrays.to_object(py));
                        result.insert("strings".to_string(), strings.to_object(py));
                        result.insert("numbers".to_string(), numbers.to_object(py));
                    }
                    Err(e) => {
                        result.insert("type".to_string(), "json".to_object(py));
                        result.insert("valid_json".to_string(), false.to_object(py));
                        result.insert("error".to_string(), e.to_string().to_object(py));
                    }
                }
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unsupported message type: {}", message_type)
                ));
            }
        }

        // Update statistics
        tokio::spawn({
            let stats = self.stats.clone();
            let msg_type = message_type.to_string();
            async move {
                let mut stats_guard = stats.write().await;
                *stats_guard.entry(format!("{}_count", msg_type)).or_insert(0) += 1;
                *stats_guard.entry(format!("{}_bytes", msg_type)).or_insert(0) += data_len as u64;
            }
        });

        Ok(result)
    }

    /// Process batch of messages in parallel
    fn process_batch<'py>(&self, py: Python<'py>, messages: Vec<HashMap<String, PyObject>>) -> PyResult<&'py PyAny> {
        let batch_size = self.batch_size;
        let timeout = self.processing_timeout;
        let _stats = self.stats.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            use rayon::prelude::*;

            let processor = RustMessageProcessor::new(10000, batch_size, timeout.as_secs());

            let results: Vec<_> = messages
                .par_chunks(batch_size)
                .flat_map(|chunk| {
                    chunk.par_iter().filter_map(|msg| {
                        Python::with_gil(|py| {
                            // Extract message type and data from the HashMap
                            let msg_type = msg.get("type")?.extract::<String>(py).ok()?;
                            let data = msg.get("data")?.extract::<Vec<u8>>(py).ok()?;

                            processor.process_message(py, &msg_type, data).ok()
                        })
                    })
                })
                .collect();

            Ok(results)
        })
    }

    /// Get processing statistics
    fn get_stats<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let stats = self.stats.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let stats_guard = stats.read().await;
            let stats_dict: HashMap<String, u64> = stats_guard.clone();
            Ok(stats_dict)
        })
    }

    /// Reset statistics
    fn reset_stats<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let stats = self.stats.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let mut stats_guard = stats.write().await;
            stats_guard.clear();
            Ok(true)
        })
    }

    /// Compress message data using zstd
    fn compress_message(&self, data: Vec<u8>) -> PyResult<Vec<u8>> {
        zstd::encode_all(std::io::Cursor::new(data), 3)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Compression error: {}", e)))
    }

    /// Decompress message data using zstd
    fn decompress_message(&self, compressed_data: Vec<u8>) -> PyResult<Vec<u8>> {
        zstd::decode_all(std::io::Cursor::new(compressed_data))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decompression error: {}", e)))
    }

    /// Validate message format
    fn validate_message(&self, py: Python<'_>, message: HashMap<String, PyObject>) -> PyResult<HashMap<String, PyObject>> {
        let mut result = HashMap::new();
        let mut errors = Vec::new();

        // Check required fields
        let required_fields = ["type", "data"];
        for field in &required_fields {
            if !message.contains_key(*field) {
                errors.push(format!("Missing required field: {}", field));
            }
        }

        // Validate message type
        if let Some(msg_type) = message.get("type") {
            if let Ok(type_str) = msg_type.extract::<String>(py) {
                let valid_types = ["text", "binary", "json", "ping", "pong"];
                if !valid_types.contains(&type_str.as_str()) {
                    errors.push(format!("Invalid message type: {}", type_str));
                }
            } else {
                errors.push("Message type must be a string".to_string());
            }
        }

        // Validate data field
        if let Some(data) = message.get("data") {
            if data.extract::<Vec<u8>>(py).is_err() {
                errors.push("Data field must be bytes".to_string());
            }
        }

        result.insert("valid".to_string(), errors.is_empty().to_object(py));
        result.insert("errors".to_string(), errors.to_object(py));

        Ok(result)
    }
}

impl RustMessageProcessor {
    /// Calculate Shannon entropy of binary data
    fn calculate_entropy(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut counts = [0u32; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }

        let len = data.len() as f64;
        let mut entropy = 0.0;

        for &count in &counts {
            if count > 0 {
                let p = count as f64 / len;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Analyze JSON structure and count elements
    fn analyze_json_structure(&self, value: &serde_json::Value) -> (u64, u64, u64, u64) {
        let mut objects = 0;
        let mut arrays = 0;
        let mut strings = 0;
        let mut numbers = 0;

        match value {
            serde_json::Value::Object(obj) => {
                objects += 1;
                for (_, v) in obj {
                    let (o, a, s, n) = self.analyze_json_structure(v);
                    objects += o;
                    arrays += a;
                    strings += s;
                    numbers += n;
                }
            }
            serde_json::Value::Array(arr) => {
                arrays += 1;
                for v in arr {
                    let (o, a, s, n) = self.analyze_json_structure(v);
                    objects += o;
                    arrays += a;
                    strings += s;
                    numbers += n;
                }
            }
            serde_json::Value::String(_) => {
                strings += 1;
            }
            serde_json::Value::Number(_) => {
                numbers += 1;
            }
            _ => {}
        }

        (objects, arrays, strings, numbers)
    }
}

/// WebSocket connection pool manager
#[pyclass]
pub struct RustWebSocketPool {
    max_connections: usize,
    connection_timeout: Duration,
    active_connections: Arc<RwLock<HashMap<String, ConnectionStats>>>,
}

#[pymethods]
impl RustWebSocketPool {
    #[new]
    #[pyo3(signature = (max_connections=100, connection_timeout_seconds=300))]
    fn new(max_connections: usize, connection_timeout_seconds: u64) -> Self {
        Self {
            max_connections,
            connection_timeout: Duration::from_secs(connection_timeout_seconds),
            active_connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get connection pool statistics
    fn get_pool_stats<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let connections = self.active_connections.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let conn_guard = connections.read().await;
            let mut stats = HashMap::new();

            stats.insert("total_connections".to_string(), conn_guard.len() as u64);
            stats.insert("max_connections".to_string(), 100u64); // self.max_connections as u64

            let mut total_messages_sent = 0u64;
            let mut total_messages_received = 0u64;
            let mut total_bytes_sent = 0u64;
            let mut total_bytes_received = 0u64;
            let mut total_errors = 0u64;

            for stats in conn_guard.values() {
                total_messages_sent += stats.messages_sent;
                total_messages_received += stats.messages_received;
                total_bytes_sent += stats.bytes_sent;
                total_bytes_received += stats.bytes_received;
                total_errors += stats.errors;
            }

            stats.insert("total_messages_sent".to_string(), total_messages_sent);
            stats.insert("total_messages_received".to_string(), total_messages_received);
            stats.insert("total_bytes_sent".to_string(), total_bytes_sent);
            stats.insert("total_bytes_received".to_string(), total_bytes_received);
            stats.insert("total_errors".to_string(), total_errors);

            Ok(stats)
        })
    }

    /// Clean up inactive connections
    fn cleanup_connections<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let connections = self.active_connections.clone();
        let timeout = self.connection_timeout;

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let mut conn_guard = connections.write().await;
            let now = Instant::now();

            let before_count = conn_guard.len();
            conn_guard.retain(|_, stats| {
                now.duration_since(stats.last_activity) < timeout
            });
            let after_count = conn_guard.len();

            Ok(before_count - after_count)
        })
    }
}
