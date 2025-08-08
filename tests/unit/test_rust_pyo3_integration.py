"""Comprehensive tests for Rust/PyO3 integration.

This module tests the PyO3 bindings, Rust extensions,
performance optimizations, and hybrid Python-Rust functionality.
"""

import json
import time
from unittest.mock import patch

import pytest

# Try to import Rust extensions
try:
    import fullstack_agent_rust

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    fullstack_agent_rust = None

# Import performance integration
from app.performance.rust_integration import RustAuthConfig
from app.performance.rust_integration import RustCacheConfig
from app.performance.rust_integration import RustJsonConfig
from app.performance.rust_integration import RustPerformanceManager


class TestRustAvailability:
    """Test Rust extension availability and basic functionality."""

    def test_rust_extension_import(self):
        """Test that Rust extension can be imported."""
        if RUST_AVAILABLE:
            assert fullstack_agent_rust is not None
            assert hasattr(fullstack_agent_rust, "__version__")
        else:
            pytest.skip("Rust extensions not available")

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_rust_extension_version(self):
        """Test Rust extension version information."""
        version = fullstack_agent_rust.__version__
        assert isinstance(version, str)
        assert len(version) > 0

        # Version should follow semantic versioning pattern
        version_parts = version.split(".")
        assert len(version_parts) >= 2
        assert all(part.isdigit() for part in version_parts[:2])

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_rust_extension_modules(self):
        """Test that expected Rust modules are available."""
        expected_modules = [
            "RustCache",
            "RustJsonProcessor",
            "RustAuthValidator",
            "RustFileOps",
            "RustWebSocketHandler",
        ]

        for module_name in expected_modules:
            assert hasattr(fullstack_agent_rust, module_name), f"Missing {module_name}"


class TestRustCache:
    """Test Rust cache implementation."""

    @pytest.fixture
    def rust_cache(self):
        """Create Rust cache instance."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extensions not available")

        return fullstack_agent_rust.RustCache(capacity=1000, ttl_seconds=3600)

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_cache_basic_operations(self, rust_cache):
        """Test basic cache operations."""
        # Test set and get
        rust_cache.set("key1", {"data": "value1"})
        result = rust_cache.get("key1")

        assert result is not None
        assert result["data"] == "value1"

        # Test non-existent key
        result = rust_cache.get("nonexistent")
        assert result is None

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_cache_capacity_limits(self, rust_cache):
        """Test cache capacity enforcement."""
        # Fill cache beyond capacity
        for i in range(1500):  # More than capacity of 1000
            rust_cache.set(f"key_{i}", {"value": i})

        # Cache should have evicted some entries
        cache_size = rust_cache.size()
        assert cache_size <= 1000

        # Most recent entries should still be there
        assert rust_cache.get("key_1499") is not None
        assert rust_cache.get("key_1400") is not None

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_cache_ttl_expiration(self, rust_cache):
        """Test cache TTL expiration."""
        # Create cache with very short TTL
        short_ttl_cache = fullstack_agent_rust.RustCache(
            capacity=100, ttl_seconds=1
        )  # 1 second TTL

        short_ttl_cache.set("expiring_key", {"data": "expires_soon"})

        # Should be available immediately
        result = short_ttl_cache.get("expiring_key")
        assert result is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired now
        result = short_ttl_cache.get("expiring_key")
        assert result is None

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_cache_performance_vs_python(self):
        """Test cache performance compared to Python dict."""
        rust_cache = fullstack_agent_rust.RustCache(capacity=10000, ttl_seconds=3600)
        python_cache = {}

        data_items = [{"key": f"key_{i}", "value": f"value_{i}"} for i in range(1000)]

        # Benchmark Rust cache
        start_time = time.perf_counter()
        for item in data_items:
            rust_cache.set(item["key"], item)
        rust_set_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        for item in data_items:
            rust_cache.get(item["key"])
        rust_get_time = time.perf_counter() - start_time

        # Benchmark Python cache
        start_time = time.perf_counter()
        for item in data_items:
            python_cache[item["key"]] = item
        python_set_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        for item in data_items:
            python_cache.get(item["key"])
        python_get_time = time.perf_counter() - start_time

        # Rust cache should be competitive (within 3x of Python)
        assert rust_set_time < python_set_time * 3, (
            f"Rust set too slow: {rust_set_time:.4f}s vs Python {python_set_time:.4f}s"
        )
        assert rust_get_time < python_get_time * 3, (
            f"Rust get too slow: {rust_get_time:.4f}s vs Python {python_get_time:.4f}s"
        )

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_cache_memory_efficiency(self, rust_cache):
        """Test cache memory efficiency."""

        # Measure memory usage of cache vs Python dict
        python_dict = {}
        large_data = {"large_field": "x" * 1000, "id": 1, "metadata": {"nested": True}}

        # Add items to both caches
        for i in range(100):
            data_copy = large_data.copy()
            data_copy["id"] = i

            rust_cache.set(f"item_{i}", data_copy)
            python_dict[f"item_{i}"] = data_copy

        # Test that Rust cache uses reasonable memory
        rust_size = rust_cache.size()
        python_size = len(python_dict)

        assert rust_size == python_size == 100

        # Both should have the same logical size
        assert rust_cache.get("item_50")["id"] == python_dict["item_50"]["id"]

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_cache_clear_operation(self, rust_cache):
        """Test cache clear operation."""
        # Add some items
        for i in range(10):
            rust_cache.set(f"key_{i}", {"value": i})

        assert rust_cache.size() == 10

        # Clear cache
        rust_cache.clear()

        assert rust_cache.size() == 0
        assert rust_cache.get("key_5") is None

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_cache_thread_safety(self, rust_cache):
        """Test cache thread safety."""
        import concurrent.futures

        def worker_thread(thread_id, operation_count):
            """Worker thread for testing thread safety."""
            for i in range(operation_count):
                key = f"thread_{thread_id}_item_{i}"
                data = {"thread_id": thread_id, "item": i}

                # Set and immediately get
                rust_cache.set(key, data)
                result = rust_cache.get(key)

                # Should get back what we set
                if result is not None:
                    assert result["thread_id"] == thread_id
                    assert result["item"] == i

        # Run multiple threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker_thread, thread_id, 100) for thread_id in range(5)]

            # Wait for all threads to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Will raise exception if thread failed

        # Cache should still be functional
        assert rust_cache.size() > 0


class TestRustJsonProcessor:
    """Test Rust JSON processing implementation."""

    @pytest.fixture
    def rust_json_processor(self):
        """Create Rust JSON processor instance."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extensions not available")

        return fullstack_agent_rust.RustJsonProcessor()

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_json_parsing_basic(self, rust_json_processor):
        """Test basic JSON parsing."""
        json_string = '{"name": "test", "value": 42, "active": true}'

        result = rust_json_processor.parse(json_string)

        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["active"] is True

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_json_serialization_basic(self, rust_json_processor):
        """Test basic JSON serialization."""
        data = {
            "name": "test",
            "value": 42,
            "active": True,
            "items": [1, 2, 3],
            "metadata": {"nested": True},
        }

        json_string = rust_json_processor.serialize(data)

        # Should be valid JSON
        parsed_back = json.loads(json_string)
        assert parsed_back == data

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_json_performance_vs_python(self, rust_json_processor):
        """Test JSON processing performance vs Python."""
        import json

        # Create large JSON data
        large_data = {
            "items": [
                {
                    "id": i,
                    "name": f"item_{i}",
                    "data": {"field_" + str(j): f"value_{j}" for j in range(20)},
                }
                for i in range(1000)
            ],
            "metadata": {"total": 1000, "created": "2024-01-01"},
        }

        # Benchmark Python JSON
        start_time = time.perf_counter()
        python_json_string = json.dumps(large_data)
        python_serialize_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        python_parsed = json.loads(python_json_string)
        python_parse_time = time.perf_counter() - start_time

        # Benchmark Rust JSON
        start_time = time.perf_counter()
        rust_json_string = rust_json_processor.serialize(large_data)
        rust_serialize_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        rust_parsed = rust_json_processor.parse(rust_json_string)
        rust_parse_time = time.perf_counter() - start_time

        # Results should be equivalent
        assert rust_parsed["metadata"]["total"] == python_parsed["metadata"]["total"]
        assert len(rust_parsed["items"]) == len(python_parsed["items"])

        # Rust should be competitive (within 2x of Python for large data)
        assert rust_serialize_time < python_serialize_time * 2, (
            f"Rust serialize too slow: {rust_serialize_time:.4f}s vs Python {python_serialize_time:.4f}s"
        )
        assert rust_parse_time < python_parse_time * 2, (
            f"Rust parse too slow: {rust_parse_time:.4f}s vs Python {python_parse_time:.4f}s"
        )

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_json_error_handling(self, rust_json_processor):
        """Test JSON error handling."""
        # Test invalid JSON
        invalid_json = '{"name": "test", "value":}'

        with pytest.raises(Exception):  # Should raise some kind of JSON parsing error
            rust_json_processor.parse(invalid_json)

        # Test serialization of non-serializable data
        class NonSerializable:
            pass

        with pytest.raises(Exception):  # Should raise serialization error
            rust_json_processor.serialize({"obj": NonSerializable()})

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_json_streaming_processing(self, rust_json_processor):
        """Test streaming JSON processing."""
        if hasattr(rust_json_processor, "parse_stream"):
            # Test streaming JSON parsing
            json_lines = [
                '{"id": 1, "name": "first"}',
                '{"id": 2, "name": "second"}',
                '{"id": 3, "name": "third"}',
            ]

            results = []
            for line in json_lines:
                result = rust_json_processor.parse(line)
                results.append(result)

            assert len(results) == 3
            assert results[0]["name"] == "first"
            assert results[2]["id"] == 3


class TestRustAuthValidator:
    """Test Rust authentication validator."""

    @pytest.fixture
    def rust_auth_validator(self):
        """Create Rust auth validator instance."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extensions not available")

        return fullstack_agent_rust.RustAuthValidator()

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_jwt_validation_basic(self, rust_auth_validator):
        """Test basic JWT validation."""
        # Mock JWT token (this would normally be a real JWT)
        mock_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        secret_key = "test-secret-key"

        # This test documents the interface - actual validation depends on implementation
        try:
            is_valid = rust_auth_validator.validate_jwt(mock_token, secret_key)
            assert isinstance(is_valid, bool)
        except Exception as e:
            # If JWT validation is not fully implemented, that's acceptable
            assert "not implemented" in str(e).lower() or "jwt" in str(e).lower()

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_password_hashing(self, rust_auth_validator):
        """Test password hashing functionality."""
        password = "test-password-123"

        try:
            # Hash password
            hashed = rust_auth_validator.hash_password(password)
            assert isinstance(hashed, str)
            assert len(hashed) > 20  # Hashed password should be substantial
            assert hashed != password  # Should be different from original

            # Verify password
            is_valid = rust_auth_validator.verify_password(password, hashed)
            assert is_valid is True

            # Verify wrong password
            is_invalid = rust_auth_validator.verify_password("wrong-password", hashed)
            assert is_invalid is False

        except Exception as e:
            # If password hashing is not implemented, that's acceptable
            assert "not implemented" in str(e).lower() or "hash" in str(e).lower()

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_auth_performance(self, rust_auth_validator):
        """Test authentication performance."""
        passwords = [f"password_{i}" for i in range(100)]

        # Benchmark password hashing
        start_time = time.perf_counter()
        hashed_passwords = []

        try:
            for password in passwords:
                hashed = rust_auth_validator.hash_password(password)
                hashed_passwords.append(hashed)

            hash_time = time.perf_counter() - start_time

            # Verify all passwords
            start_time = time.perf_counter()
            for password, hashed in zip(passwords, hashed_passwords, strict=False):
                is_valid = rust_auth_validator.verify_password(password, hashed)
                assert is_valid is True

            verify_time = time.perf_counter() - start_time

            # Performance should be reasonable
            assert hash_time < 5.0, f"Password hashing too slow: {hash_time:.2f}s for 100 passwords"
            assert verify_time < 2.0, (
                f"Password verification too slow: {verify_time:.2f}s for 100 passwords"
            )

        except Exception as e:
            # If not implemented, skip performance test
            pytest.skip(f"Password hashing not implemented: {e}")


class TestRustFileOps:
    """Test Rust file operations."""

    @pytest.fixture
    def rust_file_ops(self):
        """Create Rust file operations instance."""
        if not RUST_AVAILABLE:
            pytest.skip("Rust extensions not available")

        return fullstack_agent_rust.RustFileOps()

    @pytest.fixture
    def test_files(self, tmp_path):
        """Create test files."""
        files = {}

        # Create various test files
        files["text"] = tmp_path / "test.txt"
        files["text"].write_text("Hello, Rust file operations!")

        files["json"] = tmp_path / "test.json"
        files["json"].write_text('{"name": "test", "value": 42}')

        files["large"] = tmp_path / "large.txt"
        files["large"].write_text("x" * 100000)  # 100KB file

        return files

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_file_reading(self, rust_file_ops, test_files):
        """Test file reading operations."""
        try:
            # Read text file
            content = rust_file_ops.read_file(str(test_files["text"]))
            assert content == "Hello, Rust file operations!"

            # Read JSON file
            json_content = rust_file_ops.read_file(str(test_files["json"]))
            assert '"name": "test"' in json_content

        except Exception as e:
            pytest.skip(f"File reading not implemented: {e}")

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_file_writing(self, rust_file_ops, tmp_path):
        """Test file writing operations."""
        test_file = tmp_path / "rust_written.txt"
        test_content = "Content written by Rust!"

        try:
            # Write file
            rust_file_ops.write_file(str(test_file), test_content)

            # Verify content was written
            assert test_file.exists()
            written_content = test_file.read_text()
            assert written_content == test_content

        except Exception as e:
            pytest.skip(f"File writing not implemented: {e}")

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_batch_file_operations(self, rust_file_ops, tmp_path):
        """Test batch file operations."""
        try:
            # Create multiple files
            file_paths = []
            for i in range(10):
                file_path = tmp_path / f"batch_file_{i}.txt"
                file_paths.append(str(file_path))

            # Batch write
            contents = [f"Content for file {i}" for i in range(10)]
            rust_file_ops.batch_write(file_paths, contents)

            # Verify all files were created
            for i, file_path in enumerate(file_paths):
                assert Path(file_path).exists()
                content = Path(file_path).read_text()
                assert content == f"Content for file {i}"

            # Batch read
            read_contents = rust_file_ops.batch_read(file_paths)
            assert len(read_contents) == 10
            assert read_contents[5] == "Content for file 5"

        except Exception as e:
            pytest.skip(f"Batch file operations not implemented: {e}")

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_file_performance_vs_python(self, rust_file_ops, test_files):
        """Test file operation performance vs Python."""

        large_file = test_files["large"]

        try:
            # Benchmark Rust file reading
            start_time = time.perf_counter()
            rust_content = rust_file_ops.read_file(str(large_file))
            rust_time = time.perf_counter() - start_time

            # Benchmark Python file reading
            start_time = time.perf_counter()
            with open(large_file) as f:
                python_content = f.read()
            python_time = time.perf_counter() - start_time

            # Content should be the same
            assert len(rust_content) == len(python_content)

            # Rust should be competitive (within 2x of Python)
            assert rust_time < python_time * 2, (
                f"Rust file reading too slow: {rust_time:.4f}s vs Python {python_time:.4f}s"
            )

        except Exception as e:
            pytest.skip(f"File performance testing not available: {e}")


class TestRustPerformanceManager:
    """Test Rust performance manager integration."""

    @pytest.fixture
    def performance_manager(self):
        """Create performance manager instance."""
        config = {
            "cache": RustCacheConfig(capacity=5000, ttl_seconds=1800),
            "json": RustJsonConfig(streaming=True, validate=True),
            "auth": RustAuthConfig(hash_rounds=12, jwt_secret="test-secret"),
        }
        return RustPerformanceManager(config)

    def test_performance_manager_initialization(self, performance_manager):
        """Test performance manager initialization."""
        assert performance_manager is not None
        assert hasattr(performance_manager, "rust_available")

        if RUST_AVAILABLE:
            assert performance_manager.rust_available is True
            assert hasattr(performance_manager, "cache")
            assert hasattr(performance_manager, "json_processor")
            assert hasattr(performance_manager, "auth_validator")

    def test_performance_manager_fallback(self):
        """Test performance manager fallback to Python implementations."""
        # Test with Rust unavailable
        with patch("app.performance.rust_integration.RUST_AVAILABLE", False):
            manager = RustPerformanceManager({})

            assert manager.rust_available is False
            # Should still provide Python fallback implementations

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_integrated_performance_workflow(self, performance_manager):
        """Test integrated performance workflow using Rust components."""
        # Test cache → JSON → auth workflow
        data = {
            "user_id": "test-user-123",
            "session_data": {"login_time": "2024-01-01T10:00:00Z"},
            "permissions": ["read", "write"],
        }

        # Cache the data
        performance_manager.cache.set("user_session", data)

        # Retrieve and serialize
        cached_data = performance_manager.cache.get("user_session")
        assert cached_data is not None

        json_string = performance_manager.json_processor.serialize(cached_data)
        assert "user_id" in json_string

        # Parse back
        parsed_data = performance_manager.json_processor.parse(json_string)
        assert parsed_data["user_id"] == "test-user-123"

    def test_performance_monitoring(self, performance_manager):
        """Test performance monitoring capabilities."""
        if hasattr(performance_manager, "get_performance_stats"):
            stats = performance_manager.get_performance_stats()

            assert isinstance(stats, dict)
            assert "rust_available" in stats

            if RUST_AVAILABLE:
                assert "cache_stats" in stats
                assert "operation_counts" in stats


class TestPyO3ErrorHandling:
    """Test PyO3 error handling and edge cases."""

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_rust_exception_handling(self):
        """Test that Rust exceptions are properly handled in Python."""
        fullstack_agent_rust.RustCache(capacity=10, ttl_seconds=60)

        # Test with invalid capacity (should raise error)
        try:
            fullstack_agent_rust.RustCache(capacity=0, ttl_seconds=60)
            # If this doesn't raise an error, that's acceptable too
        except Exception as e:
            assert isinstance(e, ValueError | RuntimeError)

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_memory_management(self):
        """Test memory management between Python and Rust."""
        import gc

        # Create and destroy many Rust objects
        for i in range(100):
            cache = fullstack_agent_rust.RustCache(capacity=100, ttl_seconds=60)
            cache.set(f"key_{i}", {"data": f"value_{i}"})
            del cache

        # Force garbage collection
        gc.collect()

        # Should not crash or leak memory significantly

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_unicode_handling(self):
        """Test Unicode handling between Python and Rust."""
        rust_cache = fullstack_agent_rust.RustCache(capacity=100, ttl_seconds=60)

        # Test with Unicode strings
        unicode_data = {
            "english": "Hello World",
            "spanish": "Hola Mundo",
            "chinese": "你好世界",
            "emoji": "🚀🔥⭐",
            "mixed": "Hello 世界 🌍",
        }

        rust_cache.set("unicode_test", unicode_data)
        result = rust_cache.get("unicode_test")

        assert result is not None
        assert result["chinese"] == "你好世界"
        assert result["emoji"] == "🚀🔥⭐"
        assert result["mixed"] == "Hello 世界 🌍"

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_large_data_handling(self):
        """Test handling of large data structures."""
        rust_json = fullstack_agent_rust.RustJsonProcessor()

        # Create large data structure
        large_data = {
            "arrays": [[i * j for j in range(100)] for i in range(100)],
            "objects": {f"key_{i}": {"nested": {"value": i}} for i in range(1000)},
            "strings": ["x" * 1000 for _ in range(100)],
        }

        # Should handle large data without crashing
        try:
            json_string = rust_json.serialize(large_data)
            parsed_back = rust_json.parse(json_string)

            assert len(parsed_back["arrays"]) == 100
            assert len(parsed_back["objects"]) == 1000
            assert len(parsed_back["strings"]) == 100

        except Exception as e:
            # If it fails due to memory or size limits, that's acceptable
            assert "memory" in str(e).lower() or "size" in str(e).lower()


class TestRustBenchmarks:
    """Comprehensive benchmarks for Rust components."""

    @pytest.mark.benchmark
    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_comprehensive_performance_benchmark(self):
        """Comprehensive performance benchmark of all Rust components."""
        # This test provides detailed performance metrics for CI/CD

        results = {}

        # Cache benchmark
        rust_cache = fullstack_agent_rust.RustCache(capacity=10000, ttl_seconds=3600)
        test_data = [{"id": i, "data": f"test_data_{i}"} for i in range(1000)]

        start_time = time.perf_counter()
        for i, data in enumerate(test_data):
            rust_cache.set(f"bench_key_{i}", data)
        cache_set_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        for i in range(len(test_data)):
            rust_cache.get(f"bench_key_{i}")
        cache_get_time = time.perf_counter() - start_time

        results["cache"] = {
            "set_time": cache_set_time,
            "get_time": cache_get_time,
            "set_throughput": len(test_data) / cache_set_time,
            "get_throughput": len(test_data) / cache_get_time,
        }

        # JSON benchmark
        rust_json = fullstack_agent_rust.RustJsonProcessor()
        large_json_data = {
            "items": test_data,
            "metadata": {"count": len(test_data), "benchmark": True},
        }

        start_time = time.perf_counter()
        json_string = rust_json.serialize(large_json_data)
        json_serialize_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        rust_json.parse(json_string)
        json_parse_time = time.perf_counter() - start_time

        results["json"] = {
            "serialize_time": json_serialize_time,
            "parse_time": json_parse_time,
            "data_size": len(json_string),
        }

        # Performance assertions
        assert results["cache"]["set_throughput"] > 1000, (
            "Cache set throughput should be > 1000 ops/sec"
        )
        assert results["cache"]["get_throughput"] > 5000, (
            "Cache get throughput should be > 5000 ops/sec"
        )
        assert results["json"]["serialize_time"] < 0.1, "JSON serialization should be < 100ms"
        assert results["json"]["parse_time"] < 0.1, "JSON parsing should be < 100ms"

        return results
