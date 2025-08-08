//! Common types used throughout the fetch utility

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// HTTP authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Auth {
    /// HTTP Basic authentication
    Basic { username: String, password: String },
    /// Bearer token authentication
    Bearer { token: String },
    /// API key authentication
    ApiKey {
        key: String,
        value: String,
        #[serde(default)]
        location: ApiKeyLocation,
    },
    /// Custom authentication headers
    Custom { headers: HashMap<String, String> },
}

/// Location for API key placement
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ApiKeyLocation {
    #[default]
    Header,
    Query,
}

/// Request configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestConfig {
    /// Request timeout
    #[serde(with = "humantime_serde", default = "default_timeout")]
    pub timeout: Duration,

    /// Whether to follow redirects
    #[serde(default = "default_true")]
    pub follow_redirects: bool,

    /// Maximum number of redirects to follow
    #[serde(default = "default_max_redirects")]
    pub max_redirects: usize,

    /// Whether to verify SSL certificates
    #[serde(default = "default_true")]
    pub verify_ssl: bool,

    /// Compress request body
    #[serde(default)]
    pub compress_request: bool,

    /// Accept compressed responses
    #[serde(default = "default_true")]
    pub accept_encoding: bool,

    /// Retry configuration
    #[serde(default)]
    pub retry: RetryConfig,
}

impl Default for RequestConfig {
    fn default() -> Self {
        Self {
            timeout: default_timeout(),
            follow_redirects: true,
            max_redirects: default_max_redirects(),
            verify_ssl: true,
            compress_request: false,
            accept_encoding: true,
            retry: RetryConfig::default(),
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    #[serde(default = "default_zero")]
    pub max_retries: u32,

    /// Initial retry delay
    #[serde(with = "humantime_serde", default = "default_retry_delay")]
    pub initial_delay: Duration,

    /// Maximum retry delay
    #[serde(with = "humantime_serde", default = "default_max_retry_delay")]
    pub max_delay: Duration,

    /// Exponential backoff multiplier
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f64,

    /// Retry on connection errors
    #[serde(default = "default_true")]
    pub retry_on_connection_error: bool,

    /// Retry on timeout
    #[serde(default = "default_true")]
    pub retry_on_timeout: bool,

    /// Retry on 5xx status codes
    #[serde(default = "default_true")]
    pub retry_on_server_error: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 0,
            initial_delay: default_retry_delay(),
            max_delay: default_max_retry_delay(),
            backoff_multiplier: default_backoff_multiplier(),
            retry_on_connection_error: true,
            retry_on_timeout: true,
            retry_on_server_error: true,
        }
    }
}

/// Response metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetrics {
    /// Total request time
    pub total_time: Duration,
    /// DNS resolution time
    pub dns_time: Option<Duration>,
    /// TCP connection time
    pub connect_time: Option<Duration>,
    /// TLS handshake time
    pub tls_time: Option<Duration>,
    /// Time to first byte
    pub first_byte_time: Option<Duration>,
    /// Total download time
    pub download_time: Option<Duration>,
    /// Response body size in bytes
    pub size: u64,
    /// Number of redirects followed
    pub redirect_count: u32,
}

/// Client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// User agent string
    #[serde(default = "default_user_agent")]
    pub user_agent: String,

    /// Connection timeout
    #[serde(with = "humantime_serde", default = "default_connect_timeout")]
    pub connect_timeout: Duration,

    /// Connection pool size
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,

    /// Pool idle timeout
    #[serde(with = "humantime_serde", default = "default_pool_idle_timeout")]
    pub pool_idle_timeout: Duration,

    /// Enable HTTP/2
    #[serde(default = "default_true")]
    pub http2: bool,

    /// Enable connection pooling
    #[serde(default = "default_true")]
    pub pooling: bool,

    /// Proxy configuration
    #[serde(default)]
    pub proxy: Option<ProxyConfig>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            user_agent: default_user_agent(),
            connect_timeout: default_connect_timeout(),
            pool_size: default_pool_size(),
            pool_idle_timeout: default_pool_idle_timeout(),
            http2: true,
            pooling: true,
            proxy: None,
        }
    }
}

/// Proxy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    /// Proxy URL (http://proxy:port or socks5://proxy:port)
    pub url: String,

    /// Proxy authentication
    #[serde(default)]
    pub auth: Option<ProxyAuth>,

    /// Hostnames to exclude from proxy
    #[serde(default)]
    pub no_proxy: Vec<String>,
}

/// Proxy authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyAuth {
    pub username: String,
    pub password: String,
}

// Default value functions for serde
fn default_timeout() -> Duration {
    Duration::from_secs(30)
}

fn default_connect_timeout() -> Duration {
    Duration::from_secs(10)
}

fn default_true() -> bool {
    true
}

fn default_zero() -> u32 {
    0
}

fn default_max_redirects() -> usize {
    10
}

fn default_retry_delay() -> Duration {
    Duration::from_millis(100)
}

fn default_max_retry_delay() -> Duration {
    Duration::from_secs(30)
}

fn default_backoff_multiplier() -> f64 {
    2.0
}

fn default_user_agent() -> String {
    format!("my-fullstack-agent/{}", env!("CARGO_PKG_VERSION"))
}

fn default_pool_size() -> usize {
    16
}

fn default_pool_idle_timeout() -> Duration {
    Duration::from_secs(90)
}
