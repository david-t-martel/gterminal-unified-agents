# gterminal-filewatcher

High-performance file watcher for the gterminal project with real-time analysis, intelligent auto-fixing, and WebSocket-based dashboard updates.

## Features

- **Ultra-fast file monitoring** with intelligent debouncing and batch processing
- **Multi-tool integration** supporting ruff, mypy, AST-grep, TypeScript, Rust, and more
- **Real-time dashboard updates** via WebSocket connections
- **Intelligent auto-fixing** with rufft-claude.sh integration
- **HTTP API** for programmatic control and status monitoring
- **Project auto-detection** for Python, Node.js, Rust, and Go projects
- **Resource-efficient** with configurable performance parameters

## Quick Start

### Installation

```bash
# Clone and install
git clone <repository>
cd rust-filewatcher
./scripts/install.sh

# Or build manually
cargo build --release
sudo cp target/release/gterminal-filewatcher /usr/local/bin/
```

### Basic Usage

```bash
# Watch current directory with auto-fix enabled
gterminal-filewatcher watch

# Watch specific directory
gterminal-filewatcher watch --path /path/to/project

# Start with HTTP API server
gterminal-filewatcher watch --port 8767 --ws-port 8768

# Analyze single file
gterminal-filewatcher analyze --file src/main.py

# Run HTTP server only
gterminal-filewatcher server --port 8767
```

## Configuration

Create a `filewatcher.toml` or `.filewatcher.toml` in your project root:

```toml
[watch]
extensions = ["py", "ts", "tsx", "js", "jsx", "rs", "json", "yaml", "yml"]
ignore_dirs = ["node_modules", "target", "__pycache__", ".git", ".venv"]
debounce_ms = 100
recursive = true

[server]
host = "127.0.0.1"
http_port = 8767
websocket_port = 8768
cors_enabled = true

[performance]
max_parallel_jobs = 8
batch_size = 10
process_interval_ms = 50
memory_optimization = true

[integration]
rufft_claude_script = "scripts/rufft-claude.sh"
dashboard_status_file = "dashboard_status.json"
mcp_enabled = true

# Tool configurations
[tools.ruff]
executable = "ruff"
args = ["check", "--output-format", "json"]
extensions = ["py"]
auto_fix = true
priority = 1
timeout = 30

[tools.mypy]
executable = "mypy"
args = ["--show-error-codes"]
extensions = ["py"]
auto_fix = false
priority = 3
timeout = 60
```

## HTTP API

The filewatcher provides a comprehensive REST API:

### Status and Control

- `GET /health` - Health check
- `GET /status` - System status and performance metrics
- `GET /metrics` - Detailed performance metrics
- `POST /watch/start` - Start file watching
- `POST /watch/stop` - Stop file watching
- `POST /watch/restart` - Restart file watching

### File Analysis

- `POST /analyze` - Analyze entire project
- `POST /analyze/{file_path}` - Analyze specific file
- `POST /fix` - Auto-fix entire project
- `POST /fix/{file_path}` - Auto-fix specific file

### Tool Management

- `GET /tools` - List available tools
- `POST /tools/{tool_name}` - Run specific tool
- `GET /tools/{tool_name}/status` - Get tool status

### Configuration

- `GET /config` - Get current configuration
- `POST /config` - Update configuration

### Dashboard Integration

- `GET /dashboard` - Get dashboard data
- `POST /dashboard/update` - Update dashboard
- `GET /ws` - WebSocket upgrade for real-time updates

## WebSocket API

Connect to `ws://localhost:8768/ws` for real-time updates:

### Message Types

```json
// Subscribe to specific update types
{
  "type": "Subscribe",
  "subscriptions": {
    "file_changes": true,
    "analysis_results": true,
    "system_status": true,
    "performance_metrics": false,
    "error_notifications": true,
    "path_filters": ["src/", "tests/"],
    "tool_filters": ["ruff", "mypy"]
  }
}

// Ping for connection health
{
  "type": "Ping",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Update Messages

```json
// File change notification
{
  "type": "Update",
  "update": {
    "update_type": "FileChanged",
    "timestamp": "2024-01-01T00:00:00Z",
    "data": {
      "file": "src/main.py",
      "event_type": "Modify"
    }
  }
}

// Analysis completion
{
  "type": "Update",
  "update": {
    "update_type": "AnalysisCompleted",
    "timestamp": "2024-01-01T00:00:00Z",
    "data": {
      "file_path": "src/main.py",
      "issues": [...],
      "fixes_applied": [...],
      "status": "Success"
    }
  }
}
```

## Tool Integration

### Supported Tools

- **ruff** - Python linter and formatter
- **mypy** - Python static type checker
- **ast-grep** - Structural search and replace
- **biome** - JavaScript/TypeScript formatter and linter
- **tsc** - TypeScript compiler
- **clippy** - Rust linter
- **rustfmt** - Rust formatter

### Custom Tools

Add custom tools to your configuration:

```toml
[tools.custom_linter]
executable = "my-linter"
args = ["--json", "--fix"]
extensions = ["py", "js"]
auto_fix = true
priority = 5
timeout = 45
working_dir = "tools/"

[tools.custom_linter.env]
CUSTOM_CONFIG = "/path/to/config"
```

## Performance Optimization

### Batch Processing

Files are processed in configurable batches for optimal performance:

```toml
[performance]
max_parallel_jobs = 8      # Concurrent analysis jobs
batch_size = 10           # Files per batch
process_interval_ms = 50  # Processing frequency
cache_size = 1000         # Debounce cache size
```

### Memory Management

- Configurable debounce caching to prevent excessive processing
- Intelligent queue management with size limits
- Optional memory optimization for resource-constrained environments

### Monitoring

- Real-time performance metrics via HTTP API and WebSocket
- Resource usage tracking (CPU, memory, disk I/O)
- Cache hit rates and error rates

## Integration with gterminal

### Automatic Integration

The installer creates integration scripts:

```bash
# Start filewatcher for gterminal project
./scripts/start-filewatcher.sh

# Manual integration with existing rufft-claude.sh
# (automatically added by installer)
```

### Dashboard Integration

The filewatcher updates `dashboard_status.json` with:

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "project": "gterminal",
  "status": {
    "watcher_active": true,
    "files_watched": 245,
    "active_jobs": 2,
    "queue_size": 0,
    "performance": {
      "files_per_second": 15.2,
      "cache_hit_rate": 0.85,
      "error_rate": 0.02
    },
    "recent_activity": [...]
  }
}
```

## Systemd Service

Create a systemd service for automatic startup:

```bash
# Created by installer
sudo systemctl enable gterminal-filewatcher
sudo systemctl start gterminal-filewatcher

# Check status
sudo systemctl status gterminal-filewatcher
```

## Development

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run -- watch
```

### Benchmarks

```bash
# Run performance benchmarks
cargo bench

# Profile with flamegraph
cargo install flamegraph
cargo flamegraph --bin gterminal-filewatcher -- watch --path ./test-project
```

## Troubleshooting

### Common Issues

1. **Tool not found errors**

   ```bash
   # Check tool availability
   which ruff mypy ast-grep

   # Install missing tools
   pip install ruff mypy
   npm install -g @biomejs/biome
   ```

2. **Permission denied**

   ```bash
   # Check file permissions
   ls -la /usr/local/bin/gterminal-filewatcher

   # Fix permissions
   sudo chmod +x /usr/local/bin/gterminal-filewatcher
   ```

3. **High resource usage**

   ```toml
   # Reduce parallel jobs
   [performance]
   max_parallel_jobs = 4
   batch_size = 5
   memory_optimization = true
   ```

4. **WebSocket connection issues**

   ```bash
   # Check if port is available
   netstat -ln | grep 8768

   # Test WebSocket connection
   wscat -c ws://localhost:8768/ws
   ```

### Debug Logging

```bash
# Enable debug logging
RUST_LOG=debug gterminal-filewatcher watch

# Enable trace logging for specific modules
RUST_LOG=gterminal_filewatcher::engine=trace gterminal-filewatcher watch

# Log to file
RUST_LOG=debug gterminal-filewatcher watch 2> filewatcher.log
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `cargo test` and `cargo clippy`
5. Submit a pull request

For bug reports and feature requests, please use the GitHub issue tracker.
