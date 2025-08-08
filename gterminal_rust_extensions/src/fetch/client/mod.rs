//! HTTP client implementation

mod builder;

pub use builder::ClientBuilder;

use crate::fetch::error::{FetchError, Result};
use crate::fetch::types::{Auth, ClientConfig, RequestConfig};
use reqwest::{Client as ReqwestClient, Method};
use std::sync::Arc;
use std::time::Duration;

/// High-performance HTTP client with connection pooling
#[derive(Clone)]
pub struct Client {
    inner: ReqwestClient,
    config: Arc<ClientConfig>,
}

impl Client {
    /// Create a new client with default configuration
    pub fn new() -> Result<Self> {
        ClientBuilder::new().build()
    }

    /// Create a new client builder
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Get a reference to the inner reqwest client
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &ReqwestClient {
        &self.inner
    }

    /// Get the client configuration
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// Create a GET request
    pub fn get(&self, url: impl AsRef<str>) -> RequestBuilder {
        self.request(Method::GET, url)
    }

    /// Create a POST request
    pub fn post(&self, url: impl AsRef<str>) -> RequestBuilder {
        self.request(Method::POST, url)
    }

    /// Create a PUT request
    pub fn put(&self, url: impl AsRef<str>) -> RequestBuilder {
        self.request(Method::PUT, url)
    }

    /// Create a DELETE request
    pub fn delete(&self, url: impl AsRef<str>) -> RequestBuilder {
        self.request(Method::DELETE, url)
    }

    /// Create a PATCH request
    pub fn patch(&self, url: impl AsRef<str>) -> RequestBuilder {
        self.request(Method::PATCH, url)
    }

    /// Create a HEAD request
    pub fn head(&self, url: impl AsRef<str>) -> RequestBuilder {
        self.request(Method::HEAD, url)
    }

    /// Create a request with custom method
    pub fn request(&self, method: Method, url: impl AsRef<str>) -> RequestBuilder {
        RequestBuilder::new(self.inner.clone(), method, url.as_ref())
    }
}

/// Request builder for constructing HTTP requests
pub struct RequestBuilder {
    #[allow(dead_code)]
    client: ReqwestClient,
    builder: reqwest::RequestBuilder,
    config: RequestConfig,
}

impl RequestBuilder {
    fn new(client: ReqwestClient, method: Method, url: &str) -> Self {
        let builder = client.request(method, url);
        Self {
            client,
            builder,
            config: RequestConfig::default(),
        }
    }

    /// Set request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self.builder = self.builder.timeout(timeout);
        self
    }

    /// Set authentication
    pub fn auth(mut self, auth: Auth) -> Self {
        self.builder = match auth {
            Auth::Basic { username, password } => self.builder.basic_auth(username, Some(password)),
            Auth::Bearer { token } => self.builder.bearer_auth(token),
            Auth::ApiKey {
                key,
                value,
                location,
            } => match location {
                crate::fetch::types::ApiKeyLocation::Header => self.builder.header(key, value),
                crate::fetch::types::ApiKeyLocation::Query => self.builder.query(&[(key, value)]),
            },
            Auth::Custom { headers } => {
                let mut builder = self.builder;
                for (key, value) in headers {
                    builder = builder.header(key, value);
                }
                builder
            }
        };
        self
    }

    /// Add a header
    pub fn header(mut self, key: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        self.builder = self.builder.header(key.as_ref(), value.as_ref());
        self
    }

    /// Add multiple headers
    pub fn headers(
        mut self,
        headers: impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>,
    ) -> Self {
        for (key, value) in headers {
            self.builder = self.builder.header(key.as_ref(), value.as_ref());
        }
        self
    }

    /// Add query parameters
    pub fn query<T: serde::Serialize + ?Sized>(mut self, params: &T) -> Self {
        self.builder = self.builder.query(params);
        self
    }

    /// Set request body as JSON
    pub fn json<T: serde::Serialize + ?Sized>(mut self, json: &T) -> Self {
        self.builder = self.builder.json(json);
        self
    }

    /// Set request body as form data
    pub fn form<T: serde::Serialize + ?Sized>(mut self, form: &T) -> Self {
        self.builder = self.builder.form(form);
        self
    }

    /// Set request body as raw bytes
    pub fn body(mut self, body: impl Into<reqwest::Body>) -> Self {
        self.builder = self.builder.body(body);
        self
    }

    /// Set request body as multipart form
    pub fn multipart(mut self, form: reqwest::multipart::Form) -> Self {
        self.builder = self.builder.multipart(form);
        self
    }

    /// Set request configuration
    pub fn config(mut self, config: RequestConfig) -> Self {
        // Apply configuration to builder
        self.builder = self.builder.timeout(config.timeout);
        self.config = config;
        // Note: Other config options would be applied here
        self
    }

    /// Send the request and get response
    pub async fn send(self) -> Result<Response> {
        let response = self.builder.send().await.map_err(|e| {
            if e.is_timeout() {
                FetchError::Timeout(self.config.timeout.as_secs())
            } else if e.is_connect() {
                FetchError::Network(e.to_string())
            } else {
                FetchError::from(e)
            }
        })?;

        Ok(Response::new(response))
    }
}

/// HTTP response wrapper
pub struct Response {
    inner: reqwest::Response,
}

impl Response {
    fn new(inner: reqwest::Response) -> Self {
        Self { inner }
    }

    /// Get response status code
    pub fn status(&self) -> reqwest::StatusCode {
        self.inner.status()
    }

    /// Check if response is success (2xx)
    pub fn is_success(&self) -> bool {
        self.inner.status().is_success()
    }

    /// Get response headers
    pub fn headers(&self) -> &reqwest::header::HeaderMap {
        self.inner.headers()
    }

    /// Get response as text
    pub async fn text(self) -> Result<String> {
        self.inner
            .text()
            .await
            .map_err(|e| FetchError::Decode(e.to_string()))
    }

    /// Get response as JSON
    pub async fn json<T: serde::de::DeserializeOwned>(self) -> Result<T> {
        self.inner.json().await.map_err(|e| {
            if e.is_decode() {
                FetchError::Decode(e.to_string())
            } else {
                FetchError::from(e)
            }
        })
    }

    /// Get response as bytes
    pub async fn bytes(self) -> Result<bytes::Bytes> {
        self.inner
            .bytes()
            .await
            .map_err(|e| FetchError::Decode(e.to_string()))
    }

    /// Stream response chunks
    pub fn stream(self) -> impl futures::Stream<Item = Result<bytes::Bytes>> {
        use futures::StreamExt;

        self.inner
            .bytes_stream()
            .map(|result| result.map_err(|e| FetchError::Network(e.to_string())))
    }
}
