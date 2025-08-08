//! High-performance HTTP fetch library integrated into my-fullstack-agent

pub mod client;
pub mod error;
pub mod types;

pub use client::{Client, ClientBuilder, Response};
pub use error::{FetchError, Result};
pub use types::*;

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Fetch response data
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct FetchResponse {
    pub url: String,
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub content: String,
    pub text_content: String,
    pub backend: String,
    pub cached: bool,
    pub metrics: ResponseMetrics,
}

/// Fetch a single URL
pub async fn fetch_url(
    url: &str,
    config: Option<RequestConfig>,
) -> Result<FetchResponse> {
    let start = Instant::now();

    let client = Client::new()?;
    let mut request = client.get(url);

    if let Some(cfg) = config {
        request = request.timeout(cfg.timeout);
    }

    let response = request.send().await?;
    let status = response.status().as_u16();
    let headers: HashMap<String, String> = response
        .headers()
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();

    let content = response.text().await?;
    let elapsed = start.elapsed();

    Ok(FetchResponse {
        url: url.to_string(),
        status,
        headers,
        content: content.clone(),
        text_content: content,
        backend: "rust-fetch".to_string(),
        cached: false,
        metrics: ResponseMetrics {
            total_time: elapsed,
            dns_time: None,
            connect_time: None,
            tls_time: None,
            first_byte_time: None,
            download_time: Some(elapsed),
            size: 0, // Would be calculated from content length
            redirect_count: 0,
        },
    })
}

/// Fetch multiple URLs concurrently
pub async fn batch_fetch(
    urls: &[String],
    max_concurrent: usize,
    config: Option<RequestConfig>,
) -> Vec<Result<FetchResponse>> {
    use futures::stream::{self, StreamExt};

    let client = Client::new().unwrap_or_else(|_| panic!("Failed to create client"));

    stream::iter(urls)
        .map(|url| {
            let client = &client;
            let cfg = config.clone();
            async move {
                let start = Instant::now();
                let mut request = client.get(url);

                if let Some(config) = cfg {
                    request = request.timeout(config.timeout);
                }

                match request.send().await {
                    Ok(response) => {
                        let status = response.status().as_u16();
                        let headers: HashMap<String, String> = response
                            .headers()
                            .iter()
                            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                            .collect();

                        match response.text().await {
                            Ok(content) => {
                                let elapsed = start.elapsed();
                                Ok(FetchResponse {
                                    url: url.clone(),
                                    status,
                                    headers,
                                    content: content.clone(),
                                    text_content: content,
                                    backend: "rust-fetch".to_string(),
                                    cached: false,
                                    metrics: ResponseMetrics {
                                        total_time: elapsed,
                                        dns_time: None,
                                        connect_time: None,
                                        tls_time: None,
                                        first_byte_time: None,
                                        download_time: Some(elapsed),
                                        size: 0,
                                        redirect_count: 0,
                                    },
                                })
                            }
                            Err(e) => Err(FetchError::Decode(e.to_string())),
                        }
                    }
                    Err(e) => Err(FetchError::from(e)),
                }
            }
        })
        .buffer_unordered(max_concurrent)
        .collect()
        .await
}
