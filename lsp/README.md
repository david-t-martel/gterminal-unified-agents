# Ruff LSP Integration System

A comprehensive ruff LSP integration system that transforms ruff from a simple linter into a full AI-powered Python development assistant. This system maximizes ruff's capabilities as an LSP server with real-time diagnostics, intelligent fix suggestions, and seamless filewatcher integration.

## 🚀 Features

### Core LSP Integration

- **Persistent LSP Server**: Maintains a long-running connection to the ruff LSP server
- **Real-time Diagnostics**: Instant feedback as you type or save files
- **Advanced Code Actions**: Auto-import, refactoring, and optimization suggestions
- **Performance Monitoring**: Response time tracking and health monitoring
- **Multi-file Support**: Efficient handling of large codebases

### AI-Powered Intelligence

- **Claude Integration**: Context-aware fix suggestions using Claude's analysis
- **Confidence Scoring**: AI suggestions rated by confidence level
- **Batch Processing**: Analyze multiple files efficiently
- **Suggestion Caching**: Performance-optimized suggestion storage
- **Contextual Analysis**: Considers project structure and patterns

### Real-time Streaming

- **File Change Detection**: Automatic diagnostic refresh on file changes
- **WebSocket Integration**: Seamless connection with Rust filewatcher
- **Event Debouncing**: Intelligent batching to avoid excessive updates
- **Dashboard Integration**: Live visual feedback for development workflows
- **Multi-mode Streaming**: Auto, manual, and continuous streaming modes

### Dynamic Configuration

- **Project Analysis**: Automatic detection of project type and frameworks
- **Rule Optimization**: Intelligent rule selection based on codebase patterns
- **Framework-specific**: Tailored configurations for Django, FastAPI, etc.
- **Performance Tuning**: Optimized rule sets for different project sizes
- **A/B Testing**: Compare configuration effectiveness

## 📁 Architecture

```
gterminal/lsp/
├── __init__.py                 # Module exports and initialization
├── ruff_lsp_client.py         # Core LSP client with persistent connection
├── diagnostic_streamer.py     # Real-time diagnostic streaming system
├── ai_suggestion_engine.py    # Claude-powered fix suggestions
├── performance_monitor.py     # LSP performance monitoring and health checks
├── filewatcher_integration.py # WebSocket integration with Rust filewatcher
├── config_manager.py          # Dynamic ruff configuration management
├── demo.py                    # Comprehensive demonstration script
└── README.md                  # This documentation
```

## 🛠️ Installation & Setup

### Prerequisites

1. **Ruff with LSP support** (version 0.6.0+):

   ```bash
   pip install ruff>=0.6.0
   ```

2. **Claude CLI** for AI suggestions:

   ```bash
   # Follow Claude CLI installation instructions
   # Set CLAUDE_API_KEY environment variable
   ```

3. **Python dependencies**:

   ```bash
   pip install aiofiles aiohttp pydantic rich websockets tomli-w
   ```

4. **Optional: Rust filewatcher** for real-time integration:
   ```bash
   cd rust-filewatcher
   cargo build --release
   ```

### Quick Start

1. **Basic LSP client usage**:

   ```python
   from gterminal.lsp import RuffLSPClient, RuffLSPConfig

   config = RuffLSPConfig(workspace_root=Path("/path/to/project"))
   client = RuffLSPClient(config)

   await client.start()
   await client.open_document(Path("my_file.py"))
   diagnostics = client.get_diagnostics(Path("my_file.py"))
   await client.shutdown()
   ```

2. **Enhanced script integration**:

   ```bash
   # Enhanced rufft-claude.sh script with LSP capabilities
   ./scripts/rufft-claude.sh lsp-start
   ./scripts/rufft-claude.sh stream myfile.py auto
   ./scripts/rufft-claude.sh ai-suggest src/main.py
   ./scripts/rufft-claude.sh lsp-health
   ```

3. **Complete system demo**:
   ```bash
   cd gterminal/lsp
   python demo.py --workspace /path/to/project --mode full
   ```

## 🎯 Usage Examples

### LSP Server Management

```bash
# Start LSP server
./scripts/rufft-claude.sh lsp-start /path/to/workspace

# Check server status
./scripts/rufft-claude.sh lsp-status

# Perform health check
./scripts/rufft-claude.sh lsp-health

# Restart server
./scripts/rufft-claude.sh lsp-restart
```

### Real-time Diagnostic Streaming

```bash
# Auto-streaming with filewatcher integration
./scripts/rufft-claude.sh stream myfile.py auto

# Manual diagnostic check
./scripts/rufft-claude.sh diagnostics src/main.py

# Continuous monitoring
./scripts/rufft-claude.sh stream . continuous
```

### AI-Powered Suggestions

```bash
# Single file AI analysis
./scripts/rufft-claude.sh ai-suggest src/main.py

# Batch analysis for directory
./scripts/rufft-claude.sh ai-batch src/

# View performance metrics
./scripts/rufft-claude.sh metrics
```

### Python API Usage

```python
# Complete integration example
from gterminal.lsp import (
    RuffLSPClient, RuffLSPConfig,
    DiagnosticStreamer, DiagnosticStreamConfig,
    AISuggestionEngine, SuggestionRequest,
    FilewatcherIntegration, FilewatcherConfig,
    RuffConfigManager
)

async def main():
    # Dynamic configuration
    config_manager = RuffConfigManager(Path("."))
    analysis = await config_manager.analyze_project()
    ruff_config = await config_manager.generate_config()

    # LSP client setup
    lsp_config = RuffLSPConfig(
        workspace_root=Path("."),
        enable_ai_suggestions=True,
        enable_performance_monitoring=True
    )
    client = RuffLSPClient(lsp_config)
    await client.start()

    # AI suggestions
    ai_engine = AISuggestionEngine()
    # ... use the components

    await client.shutdown()
```

## ⚙️ Configuration

### LSP Client Configuration

```python
config = RuffLSPConfig(
    # Server settings
    server_cmd=["ruff", "server", "--preview"],
    workspace_root=Path("/project"),

    # Performance tuning
    diagnostic_debounce_ms=250,
    max_diagnostics_per_file=1000,

    # AI integration
    claude_model="haiku",
    enable_ai_suggestions=True,
    ai_confidence_threshold=0.7,

    # Monitoring
    enable_performance_monitoring=True,
    metrics_file=Path("lsp-metrics.json")
)
```

### Diagnostic Streaming Configuration

```python
stream_config = DiagnosticStreamConfig(
    # WebSocket connection
    filewatcher_host="localhost",
    filewatcher_port=8768,

    # File filtering
    file_extensions=[".py", ".pyi"],
    ignore_patterns=["__pycache__", ".venv"],

    # Event processing
    batch_size=10,
    event_debounce_ms=250,

    # Dashboard integration
    enable_dashboard_updates=True
)
```

### Dynamic Ruff Configuration

The system automatically analyzes your project and generates optimized ruff configurations:

- **Project Type Detection**: Web framework, CLI app, library, data science
- **Framework Recognition**: Django, FastAPI, Flask, Pandas, Pytest, etc.
- **Rule Optimization**: Performance-critical code, testing focus, security requirements
- **Per-file Ignores**: Smart ignore patterns for tests, migrations, config files

## 🔧 Advanced Features

### Performance Monitoring

The system includes comprehensive performance monitoring:

- **Response Times**: Average, P95, P99 response time tracking
- **Resource Usage**: Memory and CPU monitoring
- **Health Checks**: Automated health status assessment
- **Metrics Export**: JSON metrics for dashboard integration
- **Alert System**: Configurable thresholds and notifications

### Filewatcher Integration

Seamless integration with the Rust filewatcher:

- **WebSocket Connection**: Real-time file change notifications
- **Auto-reconnection**: Robust connection management
- **Event Filtering**: Smart filtering to reduce noise
- **Batch Processing**: Efficient handling of multiple changes
- **Bidirectional Communication**: Status updates to filewatcher

### AI Suggestion Engine

Claude-powered intelligent suggestions:

- **Context Analysis**: Considers entire file context and project structure
- **Confidence Scoring**: Rates suggestion quality and applicability
- **Batch Processing**: Efficient analysis of multiple files
- **Response Caching**: Performance optimization for repeated requests
- **Error Handling**: Graceful fallbacks for API failures

## 📊 Monitoring & Debugging

### LSP Health Monitoring

```bash
# Comprehensive health check
./scripts/rufft-claude.sh lsp-health

# Performance metrics
./scripts/rufft-claude.sh metrics

# Clean up resources
./scripts/rufft-claude.sh cleanup
```

### Debug Logging

Enable detailed logging for troubleshooting:

```python
config = RuffLSPConfig(
    log_level="DEBUG",
    trace="verbose"  # LSP trace level
)
```

### Performance Benchmarking

```python
# Benchmark configuration effectiveness
config_manager = RuffConfigManager(Path("."))
benchmark_results = await config_manager.benchmark_config()
print(f"Average check time: {benchmark_results['avg_check_time_ms']}ms")
```

## 🚨 Troubleshooting

### Common Issues

1. **LSP Server Won't Start**:

   - Check ruff installation: `ruff --version`
   - Verify workspace path exists
   - Check for existing ruff processes: `ps aux | grep ruff`

2. **WebSocket Connection Failed**:

   - Ensure Rust filewatcher is running: `cargo run -- watch`
   - Check port availability: `netstat -an | grep 8768`
   - Verify firewall settings

3. **AI Suggestions Not Working**:

   - Check Claude CLI: `claude --version`
   - Verify API key: `echo $CLAUDE_API_KEY`
   - Test with simple prompt: `claude "Hello"`

4. **High Response Times**:
   - Reduce `max_diagnostics_per_file`
   - Increase `diagnostic_debounce_ms`
   - Check system resources

### Debug Commands

```bash
# Check all components
./scripts/rufft-claude.sh lsp-health

# View detailed logs
tail -f /tmp/rufft-lsp.log

# Test individual components
python -m gterminal.lsp.demo --mode analysis-only --verbose
```

## 🔮 Roadmap

### Planned Features

- **IDE Integration**: VSCode extension and Neovim plugin
- **More AI Providers**: OpenAI, Anthropic API, local models
- **Advanced Caching**: Redis integration for distributed caching
- **Configuration A/B Testing**: Automated rule effectiveness testing
- **Team Analytics**: Code quality metrics and trends
- **Custom Rule Development**: Framework for project-specific rules

### Performance Improvements

- **Streaming Optimizations**: Reduce memory usage for large codebases
- **Parallel Processing**: Multi-threaded diagnostic processing
- **Incremental Analysis**: Only analyze changed portions of files
- **Smart Caching**: More sophisticated caching strategies

## 🤝 Contributing

Contributions are welcome! Please see the main project CONTRIBUTING.md for guidelines.

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -e .[dev]`
3. Run tests: `pytest tests/lsp/`
4. Run the demo: `python gterminal/lsp/demo.py --mode full`

## 📄 License

This project is licensed under the MIT License - see the main project LICENSE file for details.

## 🙏 Acknowledgments

- **Ruff Team**: For the excellent linter and LSP server
- **Anthropic**: For Claude AI capabilities
- **Rust Community**: For the high-performance filewatcher foundation
- **Python LSP Community**: For protocol standards and best practices
