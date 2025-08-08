/*!
 * Configuration management for gterminal-filewatcher
 *
 * Handles project detection, tool configuration, and runtime settings
 * with support for multiple file types and environments.
 */

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// File watching configuration
    pub watch: WatchConfig,

    /// Tool configurations
    #[serde(default)]
    pub tools: HashMap<String, ToolConfig>,

    /// Server configuration
    pub server: ServerConfig,

    /// Performance settings
    pub performance: PerformanceConfig,

    /// Integration settings
    pub integration: IntegrationConfig,
}

/// File watching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchConfig {
    /// File extensions to watch
    pub extensions: Vec<String>,

    /// Directories to ignore
    pub ignore_dirs: Vec<String>,

    /// File patterns to ignore
    pub ignore_patterns: Vec<String>,

    /// Debounce delay in milliseconds
    pub debounce_ms: u64,

    /// Enable recursive watching
    #[serde(default = "default_true")]
    pub recursive: bool,
}

/// Tool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Tool executable path
    pub executable: String,

    /// Command line arguments
    #[serde(default)]
    pub args: Vec<String>,

    /// Working directory (relative to project root)
    #[serde(default)]
    pub working_dir: Option<String>,

    /// File extensions this tool handles
    pub extensions: Vec<String>,

    /// Enable auto-fix if supported
    #[serde(default)]
    pub auto_fix: bool,

    /// Tool priority (lower number = higher priority)
    #[serde(default)]
    pub priority: u8,

    /// Timeout for tool execution in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,

    /// Environment variables for tool execution
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// HTTP server host
    #[serde(default = "default_host")]
    pub host: String,

    /// Default HTTP port
    #[serde(default = "default_http_port")]
    pub http_port: u16,

    /// Default WebSocket port
    #[serde(default = "default_ws_port")]
    pub websocket_port: u16,

    /// Enable CORS
    #[serde(default = "default_true")]
    pub cors_enabled: bool,

    /// Request timeout in seconds
    #[serde(default = "default_request_timeout")]
    pub request_timeout: u64,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum parallel jobs
    #[serde(default = "default_max_jobs")]
    pub max_parallel_jobs: usize,

    /// Batch size for file processing
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Processing interval in milliseconds
    #[serde(default = "default_process_interval")]
    pub process_interval_ms: u64,

    /// Cache size for debouncing
    #[serde(default = "default_cache_size")]
    pub cache_size: usize,

    /// Enable memory optimization
    #[serde(default = "default_true")]
    pub memory_optimization: bool,
}

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Path to rufft-claude.sh script
    pub rufft_claude_script: Option<PathBuf>,

    /// Dashboard status file path
    pub dashboard_status_file: PathBuf,

    /// Enable MCP protocol integration
    #[serde(default)]
    pub mcp_enabled: bool,

    /// MCP server configuration
    #[serde(default)]
    pub mcp: McpConfig,

    /// Webhook URLs for notifications
    #[serde(default)]
    pub webhooks: Vec<String>,

    /// Enable notifications
    #[serde(default)]
    pub notifications_enabled: bool,
}

/// MCP protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    /// MCP server name
    pub name: String,

    /// MCP protocol version
    #[serde(default = "default_mcp_version")]
    pub version: String,

    /// Tools to expose via MCP
    #[serde(default)]
    pub exposed_tools: Vec<String>,
}

// Default value functions
fn default_true() -> bool { true }
fn default_timeout() -> u64 { 30 }
fn default_host() -> String { "127.0.0.1".to_string() }
fn default_http_port() -> u16 { 8767 }
fn default_ws_port() -> u16 { 8768 }
fn default_request_timeout() -> u64 { 30 }
fn default_max_jobs() -> usize { 8 }
fn default_batch_size() -> usize { 10 }
fn default_process_interval() -> u64 { 50 }
fn default_cache_size() -> usize { 1000 }
fn default_mcp_version() -> String { "2024-11-05".to_string() }

impl Default for Config {
    fn default() -> Self {
        Self {
            watch: WatchConfig::default(),
            tools: Self::default_tools(),
            server: ServerConfig::default(),
            performance: PerformanceConfig::default(),
            integration: IntegrationConfig::default(),
        }
    }
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            extensions: vec![
                "py".to_string(),
                "ts".to_string(),
                "tsx".to_string(),
                "js".to_string(),
                "jsx".to_string(),
                "rs".to_string(),
                "json".to_string(),
                "yaml".to_string(),
                "yml".to_string(),
                "toml".to_string(),
            ],
            ignore_dirs: vec![
                "node_modules".to_string(),
                "target".to_string(),
                "__pycache__".to_string(),
                ".git".to_string(),
                ".venv".to_string(),
                "venv".to_string(),
                "dist".to_string(),
                "build".to_string(),
                ".mypy_cache".to_string(),
                ".pytest_cache".to_string(),
                ".ruff_cache".to_string(),
                "htmlcov".to_string(),
                ".coverage".to_string(),
            ],
            ignore_patterns: vec![
                "*.pyc".to_string(),
                "*.pyo".to_string(),
                "*.log".to_string(),
                "*.tmp".to_string(),
                "*~".to_string(),
                ".DS_Store".to_string(),
            ],
            debounce_ms: 100,
            recursive: true,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            http_port: default_http_port(),
            websocket_port: default_ws_port(),
            cors_enabled: true,
            request_timeout: default_request_timeout(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_parallel_jobs: default_max_jobs(),
            batch_size: default_batch_size(),
            process_interval_ms: default_process_interval(),
            cache_size: default_cache_size(),
            memory_optimization: true,
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            rufft_claude_script: Some(PathBuf::from("scripts/rufft-claude.sh")),
            dashboard_status_file: PathBuf::from("dashboard_status.json"),
            mcp_enabled: false,
            mcp: McpConfig::default(),
            webhooks: Vec::new(),
            notifications_enabled: false,
        }
    }
}

impl Config {
    /// Load configuration from project directory
    pub fn load_from_project(project_path: &Path) -> Result<Self> {
        // Try to find configuration file
        let config_paths = [
            project_path.join("filewatcher.toml"),
            project_path.join(".filewatcher.toml"),
            project_path.join("pyproject.toml"),
        ];

        for config_path in &config_paths {
            if config_path.exists() {
                debug!("Found config file: {}", config_path.display());

                if config_path.file_name().unwrap() == "pyproject.toml" {
                    // Extract filewatcher config from pyproject.toml
                    return Self::load_from_pyproject(config_path);
                } else {
                    return Self::load_from_file(config_path);
                }
            }
        }

        info!("No config file found, using defaults with project detection");
        let mut config = Self::default();
        config.detect_project_tools(project_path)?;
        Ok(config)
    }

    /// Load configuration from a specific file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let config: Config = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;

        info!("Loaded configuration from: {}", path.display());
        Ok(config)
    }

    /// Load configuration from pyproject.toml
    fn load_from_pyproject(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read pyproject.toml: {}", path.display()))?;

        let pyproject: toml::Value = toml::from_str(&content)
            .with_context(|| format!("Failed to parse pyproject.toml: {}", path.display()))?;

        // Try to extract filewatcher configuration
        let mut config = if let Some(tool_section) = pyproject.get("tool") {
            if let Some(fw_section) = tool_section.get("filewatcher") {
                fw_section.clone().try_into()
                    .context("Failed to parse filewatcher configuration from pyproject.toml")?
            } else {
                Self::default()
            }
        } else {
            Self::default()
        };

        // Detect project tools from pyproject.toml
        config.detect_project_tools(path.parent().unwrap())?;

        info!("Loaded configuration from pyproject.toml: {}", path.display());
        Ok(config)
    }

    /// Detect project tools and update configuration
    fn detect_project_tools(&mut self, project_path: &Path) -> Result<()> {
        info!("🔍 Detecting project tools in: {}", project_path.display());

        // Detect Python tools
        if project_path.join("pyproject.toml").exists()
            || project_path.join("setup.py").exists()
            || project_path.join("requirements.txt").exists() {

            info!("📦 Python project detected");
            self.add_python_tools();
        }

        // Detect Node.js/TypeScript tools
        if project_path.join("package.json").exists() {
            info!("📦 Node.js/TypeScript project detected");
            self.add_nodejs_tools();
        }

        // Detect Rust tools
        if project_path.join("Cargo.toml").exists() {
            info!("📦 Rust project detected");
            self.add_rust_tools();
        }

        // Always add universal tools
        self.add_universal_tools();

        // Update integration paths relative to project
        if let Some(script_path) = &self.integration.rufft_claude_script {
            if !script_path.is_absolute() {
                self.integration.rufft_claude_script = Some(project_path.join(script_path));
            }
        }

        Ok(())
    }

    /// Add Python-specific tools
    fn add_python_tools(&mut self) {
        // Ruff - Python linter and formatter
        self.tools.insert("ruff".to_string(), ToolConfig {
            executable: "ruff".to_string(),
            args: vec!["check".to_string(), "--fix".to_string(), "--unsafe-fixes".to_string()],
            extensions: vec!["py".to_string()],
            auto_fix: true,
            priority: 1,
            timeout: 30,
            ..Default::default()
        });

        // Ruff format
        self.tools.insert("ruff-format".to_string(), ToolConfig {
            executable: "ruff".to_string(),
            args: vec!["format".to_string()],
            extensions: vec!["py".to_string()],
            auto_fix: true,
            priority: 2,
            timeout: 30,
            ..Default::default()
        });

        // MyPy - Static type checker
        self.tools.insert("mypy".to_string(), ToolConfig {
            executable: "mypy".to_string(),
            args: vec!["--show-error-codes".to_string()],
            extensions: vec!["py".to_string()],
            auto_fix: false,
            priority: 3,
            timeout: 60,
            ..Default::default()
        });
    }

    /// Add Node.js/TypeScript tools
    fn add_nodejs_tools(&mut self) {
        // TypeScript compiler
        self.tools.insert("tsc".to_string(), ToolConfig {
            executable: "npx".to_string(),
            args: vec!["tsc".to_string(), "--noEmit".to_string()],
            extensions: vec!["ts".to_string(), "tsx".to_string()],
            auto_fix: false,
            priority: 1,
            timeout: 60,
            ..Default::default()
        });

        // Biome or Prettier for formatting
        if which::which("biome").is_ok() {
            self.tools.insert("biome".to_string(), ToolConfig {
                executable: "biome".to_string(),
                args: vec!["format".to_string(), "--write".to_string()],
                extensions: vec!["js".to_string(), "jsx".to_string(), "ts".to_string(), "tsx".to_string(), "json".to_string()],
                auto_fix: true,
                priority: 2,
                timeout: 30,
                ..Default::default()
            });
        }
    }

    /// Add Rust tools
    fn add_rust_tools(&mut self) {
        // Clippy - Rust linter
        self.tools.insert("clippy".to_string(), ToolConfig {
            executable: "cargo".to_string(),
            args: vec!["clippy".to_string(), "--fix".to_string(), "--allow-dirty".to_string(), "--allow-staged".to_string()],
            extensions: vec!["rs".to_string()],
            auto_fix: true,
            priority: 1,
            timeout: 60,
            ..Default::default()
        });

        // Rustfmt - Rust formatter
        self.tools.insert("rustfmt".to_string(), ToolConfig {
            executable: "rustfmt".to_string(),
            args: vec![],
            extensions: vec!["rs".to_string()],
            auto_fix: true,
            priority: 2,
            timeout: 30,
            ..Default::default()
        });
    }

    /// Add universal tools
    fn add_universal_tools(&mut self) {
        // AST-grep for structural search and replace
        if which::which("ast-grep").is_ok() {
            self.tools.insert("ast-grep".to_string(), ToolConfig {
                executable: "ast-grep".to_string(),
                args: vec!["scan".to_string(), "--update-all".to_string()],
                extensions: vec!["py".to_string(), "js".to_string(), "ts".to_string(), "rs".to_string()],
                auto_fix: true,
                priority: 0, // Highest priority
                timeout: 30,
                ..Default::default()
            });
        }
    }

    /// Get default tool configurations
    fn default_tools() -> HashMap<String, ToolConfig> {
        HashMap::new() // Will be populated by project detection
    }

    /// Get tools for a specific file extension
    pub fn get_tools_for_extension(&self, extension: &str) -> Vec<&ToolConfig> {
        let mut tools: Vec<_> = self.tools
            .values()
            .filter(|tool| tool.extensions.contains(&extension.to_string()))
            .collect();

        // Sort by priority
        tools.sort_by_key(|tool| tool.priority);
        tools
    }

    /// Check if a path should be ignored
    pub fn should_ignore_path(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // Check ignore directories
        for ignore_dir in &self.watch.ignore_dirs {
            if path_str.contains(ignore_dir) {
                return true;
            }
        }

        // Check ignore patterns
        for pattern in &self.watch.ignore_patterns {
            if glob::Pattern::new(pattern)
                .map(|p| p.matches(&path_str))
                .unwrap_or(false) {
                return true;
            }
        }

        false
    }

    /// Check if a file extension should be watched
    pub fn should_watch_extension(&self, extension: &str) -> bool {
        self.watch.extensions.contains(&extension.to_string())
    }
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            executable: String::new(),
            args: Vec::new(),
            working_dir: None,
            extensions: Vec::new(),
            auto_fix: false,
            priority: 10,
            timeout: default_timeout(),
            env: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(!config.watch.extensions.is_empty());
        assert!(config.watch.extensions.contains(&"py".to_string()));
        assert!(config.watch.ignore_dirs.contains(&"node_modules".to_string()));
    }

    #[test]
    fn test_project_detection() {
        let temp_dir = TempDir::new().unwrap();
        let project_path = temp_dir.path();

        // Create a Python project
        std::fs::write(project_path.join("pyproject.toml"), "[project]\nname = \"test\"").unwrap();

        let config = Config::load_from_project(project_path).unwrap();
        assert!(config.tools.contains_key("ruff"));
        assert!(config.tools.contains_key("mypy"));
    }

    #[test]
    fn test_ignore_patterns() {
        let config = Config::default();

        assert!(config.should_ignore_path(Path::new("node_modules/test.js")));
        assert!(config.should_ignore_path(Path::new("target/debug/main")));
        assert!(!config.should_ignore_path(Path::new("src/main.py")));
    }

    #[test]
    fn test_extension_filtering() {
        let config = Config::default();

        assert!(config.should_watch_extension("py"));
        assert!(config.should_watch_extension("ts"));
        assert!(!config.should_watch_extension("exe"));
        assert!(!config.should_watch_extension("bin"));
    }
}
