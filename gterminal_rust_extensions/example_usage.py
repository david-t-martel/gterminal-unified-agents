#!/usr/bin/env python3
"""
Example usage of GTerminal Rust Extensions

This script demonstrates how to use all four components:
- RustFileOps: File operations and watching
- RustCache: High-performance caching
- RustJsonProcessor: JSON processing and validation
- RustCommandExecutor: Secure command execution
"""

import json
import tempfile
import time
from pathlib import Path

# Import the Rust extensions (after building)
try:
    from gterminal_rust_extensions import (
        RustCache,
        RustCommandExecutor,
        RustFileOps,
        RustJsonProcessor,
        benchmark_components,
        build_info,
        init_tracing,
        version,
    )
except ImportError as e:
    print(f"Error importing Rust extensions: {e}")
    print("Make sure to build the extensions first with: maturin develop")
    exit(1)


def demo_file_ops():
    """Demonstrate RustFileOps functionality"""
    print("\n=== RustFileOps Demo ===")

    file_ops = RustFileOps(max_file_size=1024 * 1024, parallel_threshold=5)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.txt"

        # Write file
        content = "Hello, GTerminal Rust Extensions!\nThis is a test file."
        bytes_written = file_ops.write_file(str(test_file), content, create_dirs=True)
        print(f"✓ Written {bytes_written} bytes to {test_file}")

        # Read file
        read_content = file_ops.read_file(str(test_file))
        print(f"✓ Read content: {read_content[:50]}...")

        # Get file info
        info = file_ops.get_info(str(test_file))
        print(f"✓ File info: size={info['size']}, is_file={info['is_file']}")

        # Create directory
        new_dir = temp_path / "subdir"
        file_ops.create_dir(str(new_dir))
        print(f"✓ Created directory: {new_dir}")

        # List directory
        entries = file_ops.list_dir(str(temp_path), recursive=True)
        print(f"✓ Directory entries: {len(entries)} items")

        # Batch read (create multiple files first)
        test_files = {}
        for i in range(3):
            test_file_i = temp_path / f"test_{i}.txt"
            content_i = f"Test file {i} content"
            file_ops.write_file(str(test_file_i), content_i)
            test_files[str(test_file_i)] = content_i

        batch_results = file_ops.batch_read(list(test_files.keys()))
        print(f"✓ Batch read {len(batch_results)} files")

        # Copy file
        copy_target = temp_path / "test_copy.txt"
        bytes_copied = file_ops.copy(str(test_file), str(copy_target))
        print(f"✓ Copied {bytes_copied} bytes")

        # File watching (brief demo)
        watch_id = file_ops.watch_path(str(temp_path), recursive=True)
        print(f"✓ Started watching: {watch_id}")

        # Modify file to trigger events
        file_ops.append_file(str(test_file), "\nAppended line")

        # Check for events
        time.sleep(0.1)  # Give watcher time to detect changes
        events = file_ops.get_events(watch_id, timeout_ms=100)
        print(f"✓ Detected {len(events)} file system events")

        # Stop watching
        file_ops.unwatch(watch_id)
        print("✓ Stopped watching")

        # Get statistics
        stats = file_ops.get_stats()
        print(f"✓ File operations stats: {stats}")


def demo_cache():
    """Demonstrate RustCache functionality"""
    print("\n=== RustCache Demo ===")

    # Create cache with TTL and memory limit
    cache = RustCache(
        capacity=1000,
        default_ttl_secs=3600,  # 1 hour
        max_memory_bytes=1024 * 1024,  # 1 MB
        cleanup_interval_secs=30,
    )

    # Basic operations
    cache.set("key1", "value1")
    cache.set("key2", {"nested": {"data": [1, 2, 3]}})
    cache.set("key3", 42, ttl_secs=60)  # Custom TTL

    print(f"✓ Set 3 items, cache size: {cache.size()}")

    # Get operations
    value1 = cache.get("key1")
    value2 = cache.get("key2")
    missing = cache.get("nonexistent")

    print(f"✓ Retrieved: key1={value1}, key2={value2}, missing={missing}")

    # Batch operations
    batch_items = {f"batch_{i}": f"batch_value_{i}" for i in range(10)}
    successful = cache.set_many(batch_items)
    print(f"✓ Batch set {len(successful)} items")

    batch_results = cache.get_many([f"batch_{i}" for i in range(5)])
    print(f"✓ Batch get retrieved {len(batch_results)} items")

    # Counter operations
    cache.incr("counter", 5)
    cache.incr("counter", 10)
    counter_value = cache.get("counter")
    print(f"✓ Counter value: {counter_value}")

    # Pattern matching
    batch_keys = cache.keys("batch_*")
    print(f"✓ Found {len(batch_keys)} keys matching 'batch_*'")

    # TTL operations
    cache.expire("key1", 5)  # Expire in 5 seconds
    ttl = cache.ttl("key1")
    print(f"✓ TTL for key1: {ttl} seconds")

    # Memory and statistics
    memory_usage = cache.memory_usage()
    stats = cache.get_stats()
    print(f"✓ Memory usage: {memory_usage} bytes")
    print(
        f"✓ Cache stats: hit_rate={stats.get('hit_rate_percent', 0)}%, "
        f"hits={stats['hits']}, misses={stats['misses']}"
    )

    # Snapshot operations
    with tempfile.NamedTemporaryFile(suffix=".cache") as temp_file:
        snapshot_size = cache.save_snapshot(temp_file.name)
        print(f"✓ Saved snapshot: {snapshot_size} bytes")

        # Clear and reload
        cleared = cache.clear()
        print(f"✓ Cleared {cleared} entries")

        loaded = cache.load_snapshot(temp_file.name)
        print(f"✓ Loaded {loaded} entries from snapshot")

    # Optimization
    optimization_results = cache.optimize()
    print(f"✓ Cache optimization: {optimization_results}")


def demo_json_processor():
    """Demonstrate RustJsonProcessor functionality"""
    print("\n=== RustJsonProcessor Demo ===")

    processor = RustJsonProcessor(use_simd=True)

    # Sample JSON data
    sample_data = {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "active": True},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "active": False},
            {
                "id": 3,
                "name": "Charlie",
                "email": "charlie@example.com",
                "active": True,
            },
        ],
        "metadata": {
            "total": 3,
            "created_at": "2024-01-01T00:00:00Z",
            "version": "1.0",
        },
    }

    # Serialize to JSON
    json_str = processor.serialize(sample_data, pretty=True)
    print(f"✓ Serialized to JSON ({len(json_str)} bytes)")

    # Parse JSON
    parsed_data = processor.parse(json_str)
    print(f"✓ Parsed JSON: {type(parsed_data)}")

    # JSON Schema validation
    schema = {
        "type": "object",
        "required": ["users", "metadata"],
        "properties": {
            "users": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "name", "email"],
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                    },
                },
            },
            "metadata": {
                "type": "object",
                "required": ["total", "version"],
                "properties": {
                    "total": {"type": "integer"},
                    "version": {"type": "string"},
                },
            },
        },
    }

    processor.register_schema("user_data", json.dumps(schema))
    validation_errors = processor.validate(json_str, "user_data")
    print(f"✓ Schema validation: {len(validation_errors)} errors")

    # JSONPath queries
    active_users = processor.query(json_str, "users.*.name")  # Simplified query
    print(f"✓ Query results: {len(active_users)} items")

    # Transformations
    transform_spec = {
        "rename": {"metadata": "info"},
        "remove": ["created_at"],
        "add": {"processed_at": "2024-01-02T00:00:00Z"},
    }

    transformed = processor.transform(json_str, json.dumps(transform_spec))
    print(f"✓ Transformed JSON ({len(transformed)} bytes)")

    # Merge multiple JSON objects
    extra_data = '{"settings": {"theme": "dark", "notifications": true}}'
    merged = processor.merge([json_str, extra_data])
    print(f"✓ Merged JSON ({len(merged)} bytes)")

    # Flatten and unflatten
    flattened = processor.flatten(json_str, separator=".")
    print(f"✓ Flattened JSON ({len(flattened)} bytes)")

    unflattened = processor.unflatten(flattened, separator=".")
    print(f"✓ Unflattened JSON ({len(unflattened)} bytes)")

    # Extract specific keys
    extracted = processor.extract_keys(json_str, ["users", "metadata"])
    print(f"✓ Extracted keys: {len(extracted)} bytes")

    # Stream processing
    array_json = json.dumps([{"value": i} for i in range(10)])

    def double_value(item):
        if isinstance(item, dict) and "value" in item:
            return {"value": item["value"] * 2}
        return item

    processed_items = processor.stream_process(array_json, double_value)
    print(f"✓ Stream processed {len(processed_items)} items")

    # Diff comparison
    modified_data = sample_data.copy()
    modified_data["metadata"]["version"] = "1.1"
    modified_data["users"][0]["active"] = False

    modified_json = processor.serialize(modified_data)
    diff = processor.diff(json_str, modified_json)
    print(f"✓ JSON diff: {len(diff)} bytes")

    # Statistics
    stats = processor.get_stats()
    print(f"✓ JSON processor stats: {stats}")


def demo_command_executor():
    """Demonstrate RustCommandExecutor functionality"""
    print("\n=== RustCommandExecutor Demo ===")

    executor = RustCommandExecutor(
        max_processes=5, default_timeout_secs=30, rate_limit_per_minute=60
    )

    # Set allowed commands for security
    executor.set_allowed_commands(["echo", "ls", "pwd", "date", "uname"])

    # Set environment variables
    executor.set_env("TEST_VAR", "test_value")
    executor.set_env("CUSTOM_PATH", "/tmp")

    # Simple synchronous command execution
    result = executor.execute("echo", ["Hello, World!"])
    print(f"✓ Echo command: {result['stdout'].strip()}")
    print(f"  Exit code: {result['exit_code']}, Duration: {result['duration_ms']}ms")

    # Command with arguments
    result = executor.execute("ls", ["-la", "/tmp"], timeout_secs=10)
    print(f"✓ ls command: {len(result['stdout'])} bytes of output")

    # Environment variable test
    result = executor.execute("echo", ["$TEST_VAR"])
    print(f"✓ Environment variable: {result['stdout'].strip()}")

    # Asynchronous command execution
    pid = executor.execute_async_bg("sleep", ["2"], timeout_secs=5)
    print(f"✓ Started async command with PID: {pid}")

    # Monitor process
    status = executor.get_process_status(pid)
    print(f"  Initial status: {status}")

    # Wait for completion
    try:
        final_result = executor.wait_for_process(pid, timeout_secs=10)
        print(f"✓ Async command completed: {final_result['success']}")
        print(f"  Duration: {final_result['duration_ms']}ms")
    except Exception as e:
        print(f"✗ Error waiting for process: {e}")

    # List processes
    processes = executor.list_processes()
    print(f"✓ Active processes: {len(processes)}")

    # Long-running process with output streaming
    pid = executor.execute_async_bg("ping", ["-c", "3", "127.0.0.1"], timeout_secs=10)
    print(f"✓ Started ping command with PID: {pid}")

    # Read output in chunks
    import time

    time.sleep(1)  # Let it generate some output
    stdout_lines = executor.read_stdout(pid, max_lines=5)
    stderr_lines = executor.read_stderr(pid, max_lines=5)

    print(f"  Stdout lines: {len(stdout_lines)}")
    print(f"  Stderr lines: {len(stderr_lines)}")

    # Wait for ping to complete
    try:
        ping_result = executor.wait_for_process(pid, timeout_secs=15)
        print(f"✓ Ping completed successfully: {ping_result['success']}")
    except Exception as e:
        print(f"✗ Ping failed: {e}")

    # Test command restrictions
    try:
        result = executor.execute("rm", ["-rf", "/tmp/test"], check_allowed=True)
        print("✗ Security check failed - rm command should be blocked")
    except Exception as e:
        print(f"✓ Security check passed - blocked command: {e}")

    # Working directory test
    executor.set_working_directory("/tmp")
    result = executor.execute("pwd")
    print(f"✓ Working directory: {result['stdout'].strip()}")

    # Cleanup completed processes
    cleaned = executor.cleanup_completed()
    print(f"✓ Cleaned up {cleaned} completed processes")

    # Get statistics
    stats = executor.get_stats()
    print("✓ Command executor stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_performance():
    """Demonstrate performance capabilities"""
    print("\n=== Performance Demo ===")

    # Initialize tracing
    init_tracing("info")

    # Show build info
    build_info_data = build_info()
    print(f"✓ Build info: {build_info_data}")

    # Run benchmarks
    print("Running benchmarks...")
    benchmark_results = benchmark_components(iterations=1000)

    for component, ops_per_sec in benchmark_results.items():
        print(f"  {component}: {ops_per_sec:.2f} ops/sec")

    # Performance comparison with pure Python
    print("\nPerformance comparison:")

    # JSON processing comparison
    test_data = {"test": "data", "numbers": list(range(100))}

    # Rust JSON processor
    rust_processor = RustJsonProcessor()
    start_time = time.time()
    for _ in range(1000):
        json_str = rust_processor.serialize(test_data)
        rust_processor.parse(json_str)
    rust_time = time.time() - start_time

    # Python JSON
    import json as py_json

    start_time = time.time()
    for _ in range(1000):
        json_str = py_json.dumps(test_data)
        py_json.loads(json_str)
    python_time = time.time() - start_time

    speedup = python_time / rust_time if rust_time > 0 else 0
    print(
        f"  JSON processing: Rust={rust_time:.3f}s, Python={python_time:.3f}s, Speedup={speedup:.1f}x"
    )

    # Cache performance comparison
    rust_cache = RustCache(capacity=1000)

    # Rust cache
    start_time = time.time()
    for i in range(1000):
        rust_cache.set(f"key_{i}", f"value_{i}")
    for i in range(1000):
        rust_cache.get(f"key_{i}")
    rust_cache_time = time.time() - start_time

    # Python dict
    py_cache = {}
    start_time = time.time()
    for i in range(1000):
        py_cache[f"key_{i}"] = f"value_{i}"
    for i in range(1000):
        py_cache.get(f"key_{i}")
    python_cache_time = time.time() - start_time

    cache_speedup = python_cache_time / rust_cache_time if rust_cache_time > 0 else 0
    print(
        f"  Cache operations: Rust={rust_cache_time:.3f}s, Python={python_cache_time:.3f}s, Speedup={cache_speedup:.1f}x"
    )


def main():
    """Run all demonstrations"""
    print(f"GTerminal Rust Extensions v{version()}")
    print("=" * 60)

    try:
        demo_file_ops()
        demo_cache()
        demo_json_processor()
        demo_command_executor()
        demo_performance()

        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("The GTerminal Rust Extensions are working correctly.")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
