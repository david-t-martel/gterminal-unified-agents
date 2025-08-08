//! High-performance HTTP fetch module with PyO3 bindings

pub mod client;
pub mod error;
pub mod types;

use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

pub use client::{Client, ClientBuilder, Response};
pub use error::{FetchError, Result};
pub use types::*;

/// Python-accessible fetch response
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyFetchResponse {
    #[pyo3(get)]
    pub url: String,
    #[pyo3(get)]
    pub status: u16,
    #[pyo3(get)]
    pub headers: HashMap<String, String>,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub text_content: String,
    #[pyo3(get)]
    pub backend: String,
    #[pyo3(get)]
    pub cached: bool,
    #[pyo3(get)]
    pub metrics: PyResponseMetrics,
}

#[pymethods]
impl PyFetchResponse {
    #[new]
    fn new(
        url: String,
        status: u16,
        headers: HashMap<String, String>,
        content: String,
        text_content: String,
        backend: String,
        cached: bool,
        metrics: PyResponseMetrics,
    ) -> Self {
        Self {
            url,
            status,
            headers,
            content,
            text_content,
            backend,
            cached,
            metrics,
        }
    }

    fn __repr__(&self) -> String {
        format!("PyFetchResponse(url='{}', status={}, backend='{}')",
                self.url, self.status, self.backend)
    }

    fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    fn is_redirect(&self) -> bool {
        self.status >= 300 && self.status < 400
    }

    fn is_client_error(&self) -> bool {
        self.status >= 400 && self.status < 500
    }

    fn is_server_error(&self) -> bool {
        self.status >= 500 && self.status < 600
    }
}

/// Python-accessible response metrics
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyResponseMetrics {
    #[pyo3(get)]
    pub total_time_ms: u64,
    #[pyo3(get)]
    pub dns_time_ms: Option<u64>,
    #[pyo3(get)]
    pub connect_time_ms: Option<u64>,
    #[pyo3(get)]
    pub tls_time_ms: Option<u64>,
    #[pyo3(get)]
    pub first_byte_time_ms: Option<u64>,
    #[pyo3(get)]
    pub download_time_ms: Option<u64>,
    #[pyo3(get)]
    pub size: u64,
    #[pyo3(get)]
    pub redirect_count: u32,
}

#[pymethods]
impl PyResponseMetrics {
    #[new]
    #[pyo3(signature = (total_time_ms, size, redirect_count, dns_time_ms=None, connect_time_ms=None, tls_time_ms=None, first_byte_time_ms=None, download_time_ms=None))]
    fn new(
        total_time_ms: u64,
        size: u64,
        redirect_count: u32,
        dns_time_ms: Option<u64>,
        connect_time_ms: Option<u64>,
        tls_time_ms: Option<u64>,
        first_byte_time_ms: Option<u64>,
        download_time_ms: Option<u64>,
    ) -> Self {
        Self {
            total_time_ms,
            dns_time_ms,
            connect_time_ms,
            tls_time_ms,
            first_byte_time_ms,
            download_time_ms,
            size,
            redirect_count,
        }
    }

    fn __repr__(&self) -> String {
        format!("PyResponseMetrics(total_time={}ms, size={} bytes)",
                self.total_time_ms, self.size)
    }
}

/// High-performance HTTP fetch client with PyO3 bindings
#[pyclass]
pub struct RustFetchClient {
    client: Client,
    max_concurrent: usize,
    default_timeout_ms: u64,
}

#[pymethods]
impl RustFetchClient {
    #[new]
    #[pyo3(signature = (max_concurrent = 10, timeout_ms = 30000, user_agent = None))]
    fn new(
        max_concurrent: usize,
        timeout_ms: u64,
        user_agent: Option<String>,
    ) -> PyResult<Self> {
        let mut builder = ClientBuilder::new();

        if let Some(ua) = user_agent {
            builder = builder.user_agent(ua);
        }

        builder = builder
            .connect_timeout(Duration::from_millis(timeout_ms))
            .pool_size(max_concurrent)
            .http2(true)
            .pooling(true);

        let client = builder.build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self {
            client,
            max_concurrent,
            default_timeout_ms: timeout_ms,
        })
    }

    /// Fetch a single URL
    #[pyo3(signature = (url, timeout_ms = None, headers = None, text_only = false))]
    fn fetch<'py>(
        &self,
        py: Python<'py>,
        url: String,
        timeout_ms: Option<u64>,
        headers: Option<HashMap<String, String>>,
        text_only: Option<bool>,
    ) -> PyResult<&'py PyAny> {
        let client = self.client.clone();
        let timeout = Duration::from_millis(timeout_ms.unwrap_or(self.default_timeout_ms));
        let text_only = text_only.unwrap_or(false);

        pyo3_asyncio::tokio::return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let start = Instant::now();
            let mut request = client.get(&url).timeout(timeout);

            if let Some(hdrs) = headers {
                for (key, value) in hdrs {
                    request = request.header(key, value);
                }
            }

            let response = request.send().await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let status = response.status().as_u16();
            let headers: HashMap<String, String> = response
                .headers()
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                .collect();

            let content = if text_only {
                response.text().await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            } else {
                let bytes = response.bytes().await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                String::from_utf8_lossy(&bytes).to_string()
            };

            let elapsed = start.elapsed();
            let size = content.len() as u64;

            let metrics = PyResponseMetrics {
                total_time_ms: elapsed.as_millis() as u64,
                dns_time_ms: None,
                connect_time_ms: None,
                tls_time_ms: None,
                first_byte_time_ms: None,
                download_time_ms: Some(elapsed.as_millis() as u64),
                size,
                redirect_count: 0,
            };

            Ok(PyFetchResponse {
                url,
                status,
                headers,
                content: content.clone(),
                text_content: content,
                backend: "rust-fetch".to_string(),
                cached: false,
                metrics,
            })
        })
    }

    /// Fetch multiple URLs concurrently
    #[pyo3(signature = (urls, timeout_ms = None, headers = None, text_only = false, fail_fast = false))]
    fn batch_fetch<'py>(
        &self,
        py: Python<'py>,
        urls: Vec<String>,
        timeout_ms: Option<u64>,
        headers: Option<HashMap<String, String>>,
        text_only: Option<bool>,
        fail_fast: Option<bool>,
    ) -> PyResult<&'py PyAny> {
        let client = self.client.clone();
        let max_concurrent = self.max_concurrent;
        let timeout = Duration::from_millis(timeout_ms.unwrap_or(self.default_timeout_ms));
        let text_only = text_only.unwrap_or(false);
        let fail_fast = fail_fast.unwrap_or(false);

        pyo3_asyncio::tokio::return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            use futures::stream::{self, StreamExt};

            let results: Vec<PyResult<PyFetchResponse>> = stream::iter(urls)
                .map(|url| {
                    let client = client.clone();
                    let headers = headers.clone();
                    async move {
                        let start = Instant::now();
                        let mut request = client.get(&url).timeout(timeout);

                        if let Some(hdrs) = headers {
                            for (key, value) in hdrs {
                                request = request.header(key, value);
                            }
                        }

                        match request.send().await {
                            Ok(response) => {
                                let status = response.status().as_u16();
                                let headers: HashMap<String, String> = response
                                    .headers()
                                    .iter()
                                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                                    .collect();

                                let content = if text_only {
                                    response.text().await
                                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                                } else {
                                    let bytes = response.bytes().await
                                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                                    String::from_utf8_lossy(&bytes).to_string()
                                };

                                let elapsed = start.elapsed();
                                let size = content.len() as u64;

                                let metrics = PyResponseMetrics {
                                    total_time_ms: elapsed.as_millis() as u64,
                                    dns_time_ms: None,
                                    connect_time_ms: None,
                                    tls_time_ms: None,
                                    first_byte_time_ms: None,
                                    download_time_ms: Some(elapsed.as_millis() as u64),
                                    size,
                                    redirect_count: 0,
                                };

                                Ok(PyFetchResponse {
                                    url,
                                    status,
                                    headers,
                                    content: content.clone(),
                                    text_content: content,
                                    backend: "rust-fetch".to_string(),
                                    cached: false,
                                    metrics,
                                })
                            }
                            Err(e) => {
                                if fail_fast {
                                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
                                } else {
                                    // Return error response for non-fail-fast mode
                                    let elapsed = start.elapsed();
                                    let metrics = PyResponseMetrics {
                                        total_time_ms: elapsed.as_millis() as u64,
                                        dns_time_ms: None,
                                        connect_time_ms: None,
                                        tls_time_ms: None,
                                        first_byte_time_ms: None,
                                        download_time_ms: Some(elapsed.as_millis() as u64),
                                        size: 0,
                                        redirect_count: 0,
                                    };

                                    Ok(PyFetchResponse {
                                        url,
                                        status: 0,
                                        headers: HashMap::new(),
                                        content: format!("ERROR: {}", e),
                                        text_content: format!("ERROR: {}", e),
                                        backend: "rust-fetch".to_string(),
                                        cached: false,
                                        metrics,
                                    })
                                }
                            }
                        }
                    }
                })
                .buffer_unordered(max_concurrent)
                .collect()
                .await;

            Ok(results.into_iter().collect::<std::result::Result<Vec<_>, PyErr>>()?)
        })
    }

    fn __repr__(&self) -> String {
        format!("RustFetchClient(max_concurrent={}, timeout={}ms)",
                self.max_concurrent, self.default_timeout_ms)
    }
}

impl From<ResponseMetrics> for PyResponseMetrics {
    fn from(metrics: ResponseMetrics) -> Self {
        Self {
            total_time_ms: metrics.total_time.as_millis() as u64,
            dns_time_ms: metrics.dns_time.map(|d| d.as_millis() as u64),
            connect_time_ms: metrics.connect_time.map(|d| d.as_millis() as u64),
            tls_time_ms: metrics.tls_time.map(|d| d.as_millis() as u64),
            first_byte_time_ms: metrics.first_byte_time.map(|d| d.as_millis() as u64),
            download_time_ms: metrics.download_time.map(|d| d.as_millis() as u64),
            size: metrics.size,
            redirect_count: metrics.redirect_count,
        }
    }
}
