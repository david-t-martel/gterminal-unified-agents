//! Client builder implementation

use crate::fetch::error::{FetchError, Result};
use crate::fetch::types::{ClientConfig, ProxyConfig};
use reqwest::Client as ReqwestClient;
use std::sync::Arc;
use std::time::Duration;

use super::Client;

/// Builder for creating HTTP clients
pub struct ClientBuilder {
    config: ClientConfig,
}

impl ClientBuilder {
    /// Create a new client builder
    pub fn new() -> Self {
        Self {
            config: ClientConfig::default(),
        }
    }

    /// Set the user agent string
    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.config.user_agent = user_agent.into();
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.config.connect_timeout = timeout;
        self
    }

    /// Set connection pool size
    pub fn pool_size(mut self, size: usize) -> Self {
        self.config.pool_size = size;
        self
    }

    /// Set pool idle timeout
    pub fn pool_idle_timeout(mut self, timeout: Duration) -> Self {
        self.config.pool_idle_timeout = timeout;
        self
    }

    /// Enable or disable HTTP/2
    pub fn http2(mut self, enabled: bool) -> Self {
        self.config.http2 = enabled;
        self
    }

    /// Enable or disable connection pooling
    pub fn pooling(mut self, enabled: bool) -> Self {
        self.config.pooling = enabled;
        self
    }

    /// Set proxy configuration
    pub fn proxy(mut self, proxy: ProxyConfig) -> Self {
        self.config.proxy = Some(proxy);
        self
    }

    /// Build the client
    pub fn build(self) -> Result<Client> {
        let mut builder = ReqwestClient::builder()
            .user_agent(&self.config.user_agent)
            .connect_timeout(self.config.connect_timeout)
            .pool_idle_timeout(self.config.pool_idle_timeout)
            .http2_prior_knowledge()
            .http2_keep_alive_timeout(Duration::from_secs(30))
            .http2_keep_alive_interval(Duration::from_secs(10))
            .pool_max_idle_per_host(self.config.pool_size)
            .tcp_keepalive(Duration::from_secs(60));

        // Configure proxy if provided
        if let Some(proxy_config) = &self.config.proxy {
            let proxy = reqwest::Proxy::all(&proxy_config.url)
                .map_err(|e| FetchError::Config(format!("Invalid proxy URL: {}", e)))?;

            if let Some(auth) = &proxy_config.auth {
                builder = builder.proxy(proxy.basic_auth(&auth.username, &auth.password));
            } else {
                builder = builder.proxy(proxy);
            }
        }

        let inner = builder
            .build()
            .map_err(|e| FetchError::Config(format!("Failed to build client: {}", e)))?;

        Ok(Client {
            inner,
            config: Arc::new(self.config),
        })
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}
