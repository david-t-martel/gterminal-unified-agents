/*!
 * HTTP API server for gterminal-filewatcher
 *
 * Provides REST endpoints for controlling the file watcher, triggering analysis,
 * getting status information, and integrating with development dashboards.
 */

use crate::config::Config;
use crate::types::*;

use anyhow::Result;
use axum::{
    extract::{Path as AxumPath, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    compression::CompressionLayer,
    trace::TraceLayer,
};
use tracing::{info, warn};

/// API server state
#[derive(Clone)]
pub struct ApiState {
    config: Config,
    // TODO: Add watch engine reference for real operations
}

/// API server for gterminal-filewatcher
pub struct ApiServer {
    config: Config,
}

/// Query parameters for file analysis
#[derive(Debug, Deserialize)]
pub struct AnalyzeQuery {
    /// Whether to enable auto-fix
    #[serde(default)]
    pub auto_fix: bool,
    /// Specific tools to run (comma-separated)
    pub tools: Option<String>,
    /// Verbose output
    #[serde(default)]
    pub verbose: bool,
}

/// Query parameters for status endpoint
#[derive(Debug, Deserialize)]
pub struct StatusQuery {
    /// Include performance metrics
    #[serde(default)]
    pub include_metrics: bool,
    /// Include tool status
    #[serde(default)]
    pub include_tools: bool,
}

/// Request body for configuration updates
#[derive(Debug, Deserialize)]
pub struct ConfigUpdateRequest {
    /// Updated configuration (partial)
    pub config: serde_json::Value,
    /// Whether to restart watching with new config
    #[serde(default)]
    pub restart: bool,
}

/// Response for health check endpoint
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub timestamp: chrono::DateTime<Utc>,
}

impl ApiServer {
    /// Create a new API server
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Start the API server
    pub async fn start(self, http_port: u16, _ws_port: u16) -> Result<()> {
        let state = ApiState {
            config: self.config.clone(),
        };

        let app = self.create_router(state);

        let addr = format!("{}:{}", self.config.server.host, http_port);
        info!("🌐 Starting HTTP API server on {}", addr);

        let listener = TcpListener::bind(&addr).await?;

        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Create the router with all endpoints
    fn create_router(&self, state: ApiState) -> Router {
        let cors = if self.config.server.cors_enabled {
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        } else {
            CorsLayer::permissive()
        };

        Router::new()
            // Health and status endpoints
            .route("/health", get(health_check))
            .route("/status", get(get_status))
            .route("/metrics", get(get_metrics))

            // File analysis endpoints
            .route("/analyze", post(analyze_project))
            .route("/analyze/*file_path", post(analyze_file))
            .route("/fix", post(auto_fix_project))
            .route("/fix/*file_path", post(auto_fix_file))

            // Tool management endpoints
            .route("/tools", get(list_tools))
            .route("/tools/:tool_name", post(run_specific_tool))
            .route("/tools/:tool_name/status", get(get_tool_status))

            // Configuration endpoints
            .route("/config", get(get_config))
            .route("/config", post(update_config))

            // File system endpoints
            .route("/files", get(list_watched_files))
            .route("/files/*file_path", get(get_file_info))

            // Control endpoints
            .route("/watch/start", post(start_watching))
            .route("/watch/stop", post(stop_watching))
            .route("/watch/restart", post(restart_watching))

            // Dashboard integration
            .route("/dashboard", get(get_dashboard_data))
            .route("/dashboard/update", post(update_dashboard))

            // WebSocket upgrade endpoint (for real-time updates)
            .route("/ws", get(websocket_handler))

            .layer(
                ServiceBuilder::new()
                    .layer(TraceLayer::new_for_http())
                    .layer(CompressionLayer::new())
                    .layer(cors)
            )
            .with_state(state)
    }
}

// Handler implementations

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    let response = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0, // TODO: Track actual uptime
        timestamp: Utc::now(),
    };

    Json(response)
}

/// Get system status
async fn get_status(
    Query(query): Query<StatusQuery>,
    State(_state): State<ApiState>,
) -> impl IntoResponse {
    // TODO: Get actual system status from watch engine
    let status = SystemStatus {
        watcher_active: true,
        files_watched: 0,
        active_jobs: 0,
        queue_size: 0,
        resource_usage: ResourceUsage {
            cpu_percent: 0.0,
            memory_bytes: 0,
            memory_percent: 0.0,
            disk_io: DiskIoStats {
                bytes_read: 0,
                bytes_written: 0,
                read_ops: 0,
                write_ops: 0,
            },
            network_io: None,
        },
        performance: PerformanceMetrics {
            total_files_processed: 0,
            total_analysis_time: std::time::Duration::ZERO,
            avg_time_per_file: std::time::Duration::ZERO,
            files_per_second: 0.0,
            peak_memory_usage: 0,
            cache_hit_rate: 0.0,
            error_rate: 0.0,
        },
        tool_status: HashMap::new(),
    };

    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Status(status)),
        error: None,
    })
}

/// Get performance metrics
async fn get_metrics(State(_state): State<ApiState>) -> impl IntoResponse {
    // TODO: Get actual metrics from watch engine
    let metrics = PerformanceMetrics {
        total_files_processed: 0,
        total_analysis_time: std::time::Duration::ZERO,
        avg_time_per_file: std::time::Duration::ZERO,
        files_per_second: 0.0,
        peak_memory_usage: 0,
        cache_hit_rate: 0.0,
        error_rate: 0.0,
    };

    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Metrics(metrics)),
        error: None,
    })
}

/// Analyze entire project
async fn analyze_project(
    Query(query): Query<AnalyzeQuery>,
    State(_state): State<ApiState>,
) -> impl IntoResponse {
    info!("🔍 API request: analyze project (auto_fix: {})", query.auto_fix);

    // TODO: Trigger project-wide analysis
    let response = ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message("Project analysis started".to_string())),
        error: None,
    };

    Json(response)
}

/// Analyze specific file
async fn analyze_file(
    AxumPath(file_path): AxumPath<String>,
    Query(query): Query<AnalyzeQuery>,
    State(_state): State<ApiState>,
) -> impl IntoResponse {
    info!("🔍 API request: analyze file {} (auto_fix: {})", file_path, query.auto_fix);

    let path = PathBuf::from(file_path);

    // TODO: Trigger file analysis with actual watch engine
    let analysis_result = AnalysisResult::new(path);

    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Analysis(analysis_result)),
        error: None,
    })
}

/// Auto-fix entire project
async fn auto_fix_project(State(_state): State<ApiState>) -> impl IntoResponse {
    info!("🔧 API request: auto-fix project");

    // TODO: Trigger project-wide auto-fix
    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message("Auto-fix started for project".to_string())),
        error: None,
    })
}

/// Auto-fix specific file
async fn auto_fix_file(
    AxumPath(file_path): AxumPath<String>,
    State(_state): State<ApiState>,
) -> impl IntoResponse {
    info!("🔧 API request: auto-fix file {}", file_path);

    // TODO: Trigger file auto-fix with actual watch engine
    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message(format!("Auto-fix started for {}", file_path))),
        error: None,
    })
}

/// List available tools
async fn list_tools(State(state): State<ApiState>) -> impl IntoResponse {
    let tools: Vec<String> = state.config.tools.keys().cloned().collect();

    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message(format!("Available tools: {}", tools.join(", ")))),
        error: None,
    })
}

/// Run a specific tool
async fn run_specific_tool(
    AxumPath(tool_name): AxumPath<String>,
    State(_state): State<ApiState>,
) -> impl IntoResponse {
    info!("🔧 API request: run tool {}", tool_name);

    // TODO: Run specific tool with actual watch engine
    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message(format!("Tool {} execution started", tool_name))),
        error: None,
    })
}

/// Get tool status
async fn get_tool_status(
    AxumPath(tool_name): AxumPath<String>,
    State(state): State<ApiState>,
) -> impl IntoResponse {
    let available = state.config.tools.contains_key(&tool_name);

    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message(
            if available {
                format!("Tool {} is available", tool_name)
            } else {
                format!("Tool {} is not configured", tool_name)
            }
        )),
        error: None,
    })
}

/// Get configuration
async fn get_config(State(state): State<ApiState>) -> impl IntoResponse {
    match serde_json::to_value(&state.config) {
        Ok(config_value) => Json(ApiResponse {
            id: None,
            timestamp: Utc::now(),
            success: true,
            data: Some(ApiResponseData::Config(config_value)),
            error: None,
        }),
        Err(e) => {
            warn!("Failed to serialize config: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    id: None,
                    timestamp: Utc::now(),
                    success: false,
                    data: None,
                    error: Some(format!("Failed to get configuration: {}", e)),
                })
            ).into_response()
        }
    }
}

/// Update configuration
async fn update_config(
    State(_state): State<ApiState>,
    Json(request): Json<ConfigUpdateRequest>,
) -> impl IntoResponse {
    info!("⚙️  API request: update configuration (restart: {})", request.restart);

    // TODO: Apply configuration updates
    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message("Configuration updated".to_string())),
        error: None,
    })
}

/// List watched files
async fn list_watched_files(State(_state): State<ApiState>) -> impl IntoResponse {
    // TODO: Get actual watched files from watch engine
    let files = vec![
        PathBuf::from("example.py"),
        PathBuf::from("src/main.rs"),
    ];

    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Files(files)),
        error: None,
    })
}

/// Get file information
async fn get_file_info(
    AxumPath(file_path): AxumPath<String>,
    State(_state): State<ApiState>,
) -> impl IntoResponse {
    info!("📄 API request: get file info for {}", file_path);

    let path = PathBuf::from(&file_path);

    // Check if file exists
    if !path.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(ApiResponse {
                id: None,
                timestamp: Utc::now(),
                success: false,
                data: None,
                error: Some(format!("File not found: {}", file_path)),
            })
        ).into_response();
    }

    // TODO: Get actual file analysis info
    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message(format!("File info for {}", file_path))),
        error: None,
    }).into_response()
}

/// Start watching
async fn start_watching(State(_state): State<ApiState>) -> impl IntoResponse {
    info!("▶️  API request: start watching");

    // TODO: Start actual file watching
    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message("File watching started".to_string())),
        error: None,
    })
}

/// Stop watching
async fn stop_watching(State(_state): State<ApiState>) -> impl IntoResponse {
    info!("⏹️  API request: stop watching");

    // TODO: Stop actual file watching
    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message("File watching stopped".to_string())),
        error: None,
    })
}

/// Restart watching
async fn restart_watching(State(_state): State<ApiState>) -> impl IntoResponse {
    info!("🔄 API request: restart watching");

    // TODO: Restart actual file watching
    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message("File watching restarted".to_string())),
        error: None,
    })
}

/// Get dashboard data
async fn get_dashboard_data(State(_state): State<ApiState>) -> impl IntoResponse {
    // TODO: Generate actual dashboard data
    let dashboard_data = serde_json::json!({
        "status": "active",
        "files_watched": 0,
        "recent_activity": [],
        "performance": {
            "files_per_second": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0
        }
    });

    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Config(dashboard_data)),
        error: None,
    })
}

/// Update dashboard (for external updates)
async fn update_dashboard(
    State(_state): State<ApiState>,
    Json(update): Json<serde_json::Value>,
) -> impl IntoResponse {
    info!("📊 API request: update dashboard");

    // TODO: Process dashboard update
    Json(ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(ApiResponseData::Message("Dashboard updated".to_string())),
        error: None,
    })
}

/// WebSocket handler for real-time updates
async fn websocket_handler() -> impl IntoResponse {
    // TODO: Implement WebSocket upgrade
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(ApiResponse {
            id: None,
            timestamp: Utc::now(),
            success: false,
            data: None,
            error: Some("WebSocket endpoint not yet implemented".to_string()),
        })
    )
}

// Utility functions for API responses
pub fn success_response(data: ApiResponseData) -> ApiResponse {
    ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: true,
        data: Some(data),
        error: None,
    }
}

pub fn error_response(message: &str) -> ApiResponse {
    ApiResponse {
        id: None,
        timestamp: Utc::now(),
        success: false,
        data: None,
        error: Some(message.to_string()),
    }
}

/// Custom error handling for the API
impl IntoResponse for ApiResponse {
    fn into_response(self) -> axum::response::Response {
        let status = if self.success {
            StatusCode::OK
        } else {
            StatusCode::BAD_REQUEST
        };

        let mut response = Json(self).into_response();
        *response.status_mut() = status;

        // Add content type header
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_endpoint() {
        let config = Config::default();
        let state = ApiState { config: config.clone() };
        let server = ApiServer::new(config);
        let app = server.create_router(state);

        let request = Request::builder()
            .uri("/health")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_status_endpoint() {
        let config = Config::default();
        let state = ApiState { config: config.clone() };
        let server = ApiServer::new(config);
        let app = server.create_router(state);

        let request = Request::builder()
            .uri("/status")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_config_endpoint() {
        let config = Config::default();
        let state = ApiState { config: config.clone() };
        let server = ApiServer::new(config);
        let app = server.create_router(state);

        let request = Request::builder()
            .uri("/config")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
