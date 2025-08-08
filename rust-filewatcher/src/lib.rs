/*!
 * gterminal-filewatcher library
 *
 * High-performance file watching and analysis library for development environments.
 * Provides real-time monitoring of source files with intelligent debouncing,
 * tool integration, and WebSocket-based dashboard updates.
 */

pub mod config;
pub mod engine;
pub mod server;
pub mod tools;
pub mod types;
pub mod websocket;

pub use config::Config;
pub use engine::WatchEngine;
pub use server::ApiServer;
pub use types::*;
pub use websocket::WebSocketServer;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration file names to search for
pub const CONFIG_FILES: &[&str] = &[
    "filewatcher.toml",
    ".filewatcher.toml",
    "pyproject.toml",
];

/// Initialize tracing with sensible defaults
pub fn init_tracing() -> anyhow::Result<()> {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,gterminal_filewatcher=debug"));

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_target(false)
                .with_thread_ids(false)
                .with_level(true)
                .with_ansi(true)
                .compact(),
        )
        .with(filter)
        .init();

    Ok(())
}

/// Create a default configuration for a project directory
pub fn create_default_config(project_path: &std::path::Path) -> anyhow::Result<Config> {
    Config::load_from_project(project_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_config_files() {
        assert!(!CONFIG_FILES.is_empty());
        assert!(CONFIG_FILES.contains(&"filewatcher.toml"));
    }
}
