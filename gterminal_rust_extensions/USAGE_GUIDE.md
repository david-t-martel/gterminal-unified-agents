# GTerminal Rust Extensions - Usage Guide

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install Python dependencies
pip install maturin
```

### 2. Build and Install

```bash
cd gterminal_rust_extensions

# Build and install in one step
python build.py --clean

# Or step by step
python build.py --clean --release --wheel
```

### 3. Test Installation

```bash
python example_usage.py
```

## Moving to GTerminal Directory

To use these extensions in your actual GTerminal project:

```bash
# Copy the entire directory to your GTerminal project
cp -r /home/david/agents/my-fullstack-agent/gterminal_rust_extensions /home/david/agents/gterminal/rust_extensions

# Or move it
mv /home/david/agents/my-fullstack-agent/gterminal_rust_extensions /home/david/agents/gterminal/rust_extensions

# Navigate to the new location
cd /home/david/agents/gterminal/rust_extensions

# Rebuild in new location
python build.py --clean --release
```

## Integration Patterns

### 1. Basic Integration

```python
# In your GTerminal Python code
from gterminal_rust_extensions import (
    RustFileOps, RustCache, RustJsonProcessor, RustCommandExecutor
)

class GTerminalReactEngine:
    def __init__(self):
        # High-performance file operations
        self.file_ops = RustFileOps(
            max_file_size=10*1024*1024,  # 10MB limit
            parallel_threshold=5         # Parallelize 5+ files
        )

        # Fast caching for context/results
        self.cache = RustCache(
            capacity=50000,
            default_ttl_secs=3600,       # 1 hour
            max_memory_bytes=500*1024*1024  # 500MB
        )

        # JSON processing for structured data
        self.json_processor = RustJsonProcessor(use_simd=True)

        # Secure command execution
        self.executor = RustCommandExecutor(
            max_processes=10,
            default_timeout_secs=300,
            rate_limit_per_minute=120
        )

        # Configure security
        self.executor.set_allowed_commands([
            "ls", "find", "grep", "cat", "head", "tail",
            "git", "python", "node", "npm", "cargo"
        ])
```

### 2. ReAct Pattern Integration

```python
class ReActAgent:
    def __init__(self):
        self.cache = RustCache(capacity=10000)
        self.file_ops = RustFileOps()
        self.json_processor = RustJsonProcessor()
        self.executor = RustCommandExecutor()

    def think(self, observation, context):
        # Check cache for similar contexts
        context_hash = hash(str(context))
        cached_thought = self.cache.get(f"thought:{context_hash}")
        if cached_thought:
            return cached_thought

        # Process observation with fast JSON operations
        if isinstance(observation, str) and observation.startswith('{'):
            structured_obs = self.json_processor.parse(observation)
        else:
            structured_obs = {"raw": observation}

        # Generate thought process
        thought = self._generate_thought(structured_obs, context)

        # Cache the result
        self.cache.set(f"thought:{context_hash}", thought, ttl_secs=1800)
        return thought

    def act(self, action_plan):
        # Execute actions using high-performance components
        action_type = action_plan.get("type")

        if action_type == "read_files":
            paths = action_plan["paths"]
            if len(paths) > 1:
                # Use batch read for multiple files
                results = self.file_ops.batch_read(paths)
            else:
                results = {paths[0]: self.file_ops.read_file(paths[0])}
            return results

        elif action_type == "execute_command":
            result = self.executor.execute(
                action_plan["command"],
                action_plan.get("args", []),
                timeout_secs=action_plan.get("timeout", 60)
            )
            return {
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "success": result["success"]
            }

        elif action_type == "process_json":
            data = action_plan["data"]
            if action_plan.get("query"):
                return self.json_processor.query(data, action_plan["query"])
            else:
                return self.json_processor.parse(data)
```

### 3. Advanced File Processing

```python
class FileProcessor:
    def __init__(self):
        self.file_ops = RustFileOps(parallel_threshold=10)
        self.json_processor = RustJsonProcessor()

    def process_project_directory(self, project_path):
        # List all relevant files
        python_files = self.file_ops.list_dir(
            project_path,
            pattern="*.py",
            recursive=True
        )

        js_files = self.file_ops.list_dir(
            project_path,
            pattern="*.js",
            recursive=True
        )

        # Batch read all files (automatically parallelized)
        all_files = [f["path"] for f in python_files + js_files]
        file_contents = self.file_ops.batch_read(all_files)

        # Process each file
        analysis_results = {}
        for file_path, content in file_contents.items():
            if file_path.endswith('.py'):
                analysis_results[file_path] = self.analyze_python_file(content)
            elif file_path.endswith('.js'):
                analysis_results[file_path] = self.analyze_js_file(content)

        return analysis_results

    def watch_for_changes(self, directory, callback):
        # Start watching for file changes
        watch_id = self.file_ops.watch_path(directory, recursive=True)

        try:
            while True:
                events = self.file_ops.get_events(watch_id, timeout_ms=1000)
                if events:
                    for event in events:
                        callback(event)
                time.sleep(0.1)
        finally:
            self.file_ops.unwatch(watch_id)
```

### 4. Caching Strategies

```python
class SmartCache:
    def __init__(self):
        # Multi-tier caching
        self.l1_cache = RustCache(
            capacity=1000,
            default_ttl_secs=300,    # 5 minutes - hot data
            max_memory_bytes=50*1024*1024
        )

        self.l2_cache = RustCache(
            capacity=10000,
            default_ttl_secs=3600,   # 1 hour - warm data
            max_memory_bytes=200*1024*1024
        )

    def get(self, key):
        # Check L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        # Check L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.set(key, value, ttl_secs=300)
            return value

        return None

    def set(self, key, value, importance="normal"):
        if importance == "high":
            # Store in both caches
            self.l1_cache.set(key, value, ttl_secs=600)  # 10 minutes
            self.l2_cache.set(key, value, ttl_secs=7200) # 2 hours
        else:
            # Store in L2 only
            self.l2_cache.set(key, value)
```

### 5. Command Pipeline

```python
class CommandPipeline:
    def __init__(self):
        self.executor = RustCommandExecutor(max_processes=5)
        self.executor.set_allowed_commands([
            "git", "find", "grep", "awk", "sed", "sort", "uniq", "wc"
        ])

    async def analyze_git_repository(self, repo_path):
        self.executor.set_working_directory(repo_path)

        # Run multiple commands in parallel
        commands = [
            ("git", ["log", "--oneline", "-n", "10"]),
            ("git", ["status", "--porcelain"]),
            ("find", [".", "-name", "*.py", "-type", "f"]),
            ("git", ["diff", "--stat"])
        ]

        # Start all commands asynchronously
        pids = []
        for cmd, args in commands:
            pid = self.executor.execute_async_bg(cmd, args)
            pids.append((pid, f"{cmd} {' '.join(args)}"))

        # Collect results
        results = {}
        for pid, description in pids:
            try:
                result = self.executor.wait_for_process(pid, timeout_secs=30)
                results[description] = {
                    "stdout": result["stdout"],
                    "success": result["success"]
                }
            except Exception as e:
                results[description] = {"error": str(e)}

        return results
```

## Performance Tips

### 1. File Operations

- Use `batch_read()` for multiple files (automatic parallelization)
- Set appropriate `parallel_threshold` based on your use case
- Use file watching instead of polling for real-time monitoring
- Set `max_file_size` to prevent memory issues with large files

### 2. Caching

- Choose appropriate TTL values based on data volatility
- Use `get_many()` and `set_many()` for batch operations
- Monitor hit rates with `get_stats()` and adjust capacity
- Use `optimize()` periodically to clean up expired entries
- Save snapshots for persistence across restarts

### 3. JSON Processing

- Enable SIMD for better performance on supported CPUs
- Use `stream_process()` for large arrays
- Register schemas once and reuse for validation
- Use `query()` instead of parsing entire documents for specific data

### 4. Command Execution

- Set appropriate timeout values to prevent hanging
- Use rate limiting to prevent resource exhaustion
- Configure allowed/blocked commands for security
- Use async execution for long-running commands
- Clean up completed processes regularly

### 5. Memory Management

- Monitor memory usage with component stats
- Set appropriate memory limits for caches
- Use memory tracking to identify leaks
- Clean up resources when done (watchers, processes)

## Monitoring and Debugging

### 1. Enable Comprehensive Logging

```python
from gterminal_rust_extensions import init_tracing
init_tracing("info")  # or "debug" for verbose output
```

### 2. Performance Monitoring

```python
def monitor_performance(components):
    while True:
        stats = {}
        if hasattr(components, 'file_ops'):
            stats['file_ops'] = components.file_ops.get_stats()
        if hasattr(components, 'cache'):
            stats['cache'] = components.cache.get_stats()
        if hasattr(components, 'json_processor'):
            stats['json'] = components.json_processor.get_stats()
        if hasattr(components, 'executor'):
            stats['executor'] = components.executor.get_stats()

        # Log or process stats
        print(f"Performance Stats: {stats}")
        time.sleep(60)  # Monitor every minute
```

### 3. Benchmarking

```python
from gterminal_rust_extensions import benchmark_components

# Run built-in benchmarks
results = benchmark_components(iterations=10000)
print("Benchmark Results:")
for component, ops_per_sec in results.items():
    print(f"  {component}: {ops_per_sec:,.0f} ops/sec")
```

## Troubleshooting

### Common Issues and Solutions

1. **Import Error**: Extension not found

   ```bash
   # Rebuild and reinstall
   python build.py --clean

   # Check if installed
   python -c "import gterminal_rust_extensions; print('OK')"
   ```

2. **Performance Issues**: Not seeing expected speedup

   ```python
   # Check SIMD is enabled
   processor = RustJsonProcessor(use_simd=True)

   # Verify parallel threshold
   file_ops = RustFileOps(parallel_threshold=5)  # Lower for more parallelization

   # Monitor actual performance
   from gterminal_rust_extensions import benchmark_components
   print(benchmark_components(1000))
   ```

3. **Memory Issues**: High memory usage

   ```python
   # Set memory limits
   cache = RustCache(max_memory_bytes=100*1024*1024)  # 100MB limit

   # Monitor usage
   from gterminal_rust_extensions import get_memory_info
   print(get_memory_info())

   # Optimize regularly
   cache.optimize()
   ```

4. **Security Issues**: Commands being blocked

   ```python
   # Check allowed commands
   executor = RustCommandExecutor()
   executor.set_allowed_commands(["git", "python", "node"])  # Add needed commands

   # Remove from blocked list
   executor.set_blocked_commands([])  # Clear blocked list if needed
   ```

## Next Steps

1. **Integration**: Move the extensions to your GTerminal directory
2. **Customization**: Modify the Rust code for specific needs
3. **Performance Testing**: Benchmark with your actual workloads
4. **Monitoring**: Set up performance monitoring in production
5. **Security**: Configure command restrictions for your environment

The extensions are designed to be drop-in replacements for Python equivalents with significant performance improvements. Start with the basic integration and gradually adopt more advanced features as needed.
