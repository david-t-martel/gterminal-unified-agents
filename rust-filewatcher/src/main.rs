/*!
 * gterminal-filewatcher - High-performance file watching and analysis
 *
 * Monitors Python, TypeScript, Rust, JSON, and YAML files with minimal latency.
 * Triggers ruff LSP analysis and integrates with rufft-claude.sh for intelligent fixes.
 * Provides WebSocket and HTTP APIs for real-time development dashboard updates.
 */

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tokio::signal;
use tracing::{info, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod config;
mod engine;
mod server;
mod tools;
mod types;
mod websocket;

use config::Config;
use engine::WatchEngine;
use server::ApiServer;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(name = "gterminal-filewatcher")]
#[command(about = "High-performance filewatcher for gterminal project")]
#[command(version = "1.0.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start file watching with real-time analysis
    Watch {
        /// Project directory to watch
        #[arg(short, long, default_value = ".")]
        path: PathBuf,

        /// Enable WebSocket server for real-time updates
        #[arg(long, default_value_t = true)]
        websocket: bool,

        /// HTTP server port
        #[arg(long, default_value_t = 8767)]
        port: u16,

        /// WebSocket port
        #[arg(long, default_value_t = 8768)]
        ws_port: u16,

        /// Disable auto-fixes
        #[arg(long)]
        no_auto_fix: bool,
    },

    /// Run single file analysis
    Analyze {
        /// File to analyze
        file: PathBuf,

        /// Show detailed analysis
        #[arg(long)]
        verbose: bool,
    },

    /// Start HTTP server only
    Server {
        /// HTTP server port
        #[arg(short, long, default_value_t = 8767)]
        port: u16,

        /// Project directory
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
    },

    /// Validate configuration
    Config {
        /// Configuration file path
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Show current configuration
        #[arg(long)]
        show: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize high-performance tracing
    init_tracing()?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Watch {
            path,
            websocket,
            port,
            ws_port,
            no_auto_fix,
        } => {
            run_watcher(path, websocket, port, ws_port, !no_auto_fix).await
        }

        Commands::Analyze { file, verbose } => {
            run_analysis(file, verbose).await
        }

        Commands::Server { port, path } => {
            run_server_only(port, path).await
        }

        Commands::Config { file, show } => {
            handle_config(file, show).await
        }
    }
}

fn init_tracing() -> Result<()> {
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

async fn run_watcher(
    path: PathBuf,
    enable_websocket: bool,
    port: u16,
    ws_port: u16,
    auto_fix: bool,
) -> Result<()> {
    info!("🚀 Starting gterminal-filewatcher");
    info!("📁 Watching directory: {}", path.display());

    // Load configuration
    let config = Config::load_from_project(&path)?;

    // Create watch engine
    let mut engine = WatchEngine::new(path.clone(), config.clone(), auto_fix)?;

    // Start HTTP server
    let server_handle = if enable_websocket || port > 0 {
        let api_server = ApiServer::new(config.clone());
        Some(tokio::spawn(api_server.start(port, ws_port)))
    } else {
        None
    };

    // Handle graceful shutdown
    let shutdown_signal = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
        info!("🛑 Shutdown signal received");
    };

    // Run the watch engine
    tokio::select! {
        result = engine.start() => {
            if let Err(e) = result {
                tracing::error!("Watch engine failed: {}", e);
            }
        }
        _ = shutdown_signal => {
            info!("Shutting down gracefully...");
        }
    }

    // Clean shutdown
    if let Some(handle) = server_handle {
        handle.abort();
    }

    info!("✅ gterminal-filewatcher stopped");
    Ok(())
}

async fn run_analysis(file: PathBuf, verbose: bool) -> Result<()> {
    info!("🔍 Analyzing file: {}", file.display());

    let config = Config::load_from_project(&file.parent().unwrap_or(&PathBuf::from(".")))?;
    let mut engine = WatchEngine::new(PathBuf::from("."), config, false)?;

    let result = engine.analyze_file(&file).await?;

    if verbose {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("Analysis completed: {} issues found", result.issues.len());
        for issue in &result.issues {
            println!("  - {}: {}", issue.severity, issue.message);
        }
    }

    Ok(())
}

async fn run_server_only(port: u16, path: PathBuf) -> Result<()> {
    info!("🌐 Starting HTTP server on port {}", port);

    let config = Config::load_from_project(&path)?;
    let api_server = ApiServer::new(config);

    api_server.start(port, 0).await
}

async fn handle_config(file: Option<PathBuf>, show: bool) -> Result<()> {
    let config = if let Some(path) = file {
        Config::load_from_file(&path)?
    } else {
        Config::load_from_project(&PathBuf::from("."))?
    };

    if show {
        println!("{}", toml::to_string_pretty(&config)?);
    } else {
        println!("✅ Configuration is valid");
        println!("Watched extensions: {:?}", config.watch.extensions);
        println!("Tools configured: {:?}", config.tools.iter().map(|(k, _)| k).collect::<Vec<_>>());
    }

    Ok(())
}
