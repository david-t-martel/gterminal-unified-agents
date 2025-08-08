#!/usr/bin/env python3
"""Unit tests for performance components.

This test suite covers:
- CacheManager functionality and performance
- PerformanceMonitor execution profiling
- Memory and resource tracking
- Performance threshold monitoring
"""

import asyncio
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from performance.cache_manager import CacheManager
from performance.performance_monitor import ExecutionProfile
from performance.performance_monitor import PerformanceMetric
from performance.performance_monitor import PerformanceMonitor


class TestCacheManager:
    """Test cases for CacheManager."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_cache_manager_initialization(self):
        """Test CacheManager initialization."""
        cache = CacheManager(max_size=100, default_ttl=3600, cache_dir=self.temp_dir)

        assert cache.max_size == 100
        assert cache.default_ttl == 3600
        assert cache.cache_dir == self.temp_dir
        assert len(cache._memory_cache) == 0
        assert cache.stats["hits"] == 0

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = CacheManager(cache_dir=self.temp_dir)

        # Set a value
        await cache.set("test_key", "test_value", ttl=60)

        # Get the value
        result = await cache.get("test_key")

        assert result == "test_value"
        assert cache.stats["writes"] == 1
        assert cache.stats["hits"] == 1

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache expiration functionality."""
        cache = CacheManager(cache_dir=self.temp_dir)

        # Set a value with short TTL
        await cache.set("expire_key", "expire_value", ttl=1)

        # Should be available immediately
        result = await cache.get("expire_key")
        assert result == "expire_value"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be None after expiration
        result = await cache.get("expire_key")
        assert result is None
        assert cache.stats["misses"] > 0

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = CacheManager(max_size=2, cache_dir=self.temp_dir)

        # Fill cache to capacity
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Access key1 to make it more recently used
        await cache.get("key1")

        # Add third item, should evict key2 (least recently used)
        await cache.set("key3", "value3")

        # key1 and key3 should exist, key2 should be evicted
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") is None
        assert await cache.get("key3") == "value3"
        assert cache.stats["evictions"] > 0

    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Test cache deletion."""
        cache = CacheManager(cache_dir=self.temp_dir)

        await cache.set("delete_key", "delete_value")
        assert await cache.get("delete_key") == "delete_value"

        # Delete the key
        deleted = await cache.delete("delete_key")
        assert deleted is True

        # Should not be found
        assert await cache.get("delete_key") is None

        # Deleting non-existent key
        deleted = await cache.delete("non_existent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = CacheManager(cache_dir=self.temp_dir)

        # Add multiple items
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        assert len(cache._memory_cache) == 3

        # Clear cache
        await cache.clear()

        assert len(cache._memory_cache) == 0
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_persistent_cache(self):
        """Test persistent cache functionality."""
        cache = CacheManager(cache_dir=self.temp_dir)

        # Set value (will be stored persistently)
        await cache.set("persistent_key", {"data": "persistent_value"})

        # Clear memory cache but keep persistent
        cache._memory_cache.clear()
        cache._access_times.clear()

        # Should still be retrievable from persistent cache
        result = await cache.get("persistent_key")
        assert result == {"data": "persistent_value"}

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = CacheManager(cache_dir=self.temp_dir)

        # Generate some cache activity
        await cache.set("stats_key1", "value1")
        await cache.set("stats_key2", "value2")
        await cache.get("stats_key1")  # hit
        await cache.get("stats_key1")  # hit
        await cache.get("nonexistent")  # miss

        stats = cache.get_stats()

        assert stats["memory_cache_size"] == 2
        assert stats["total_hits"] == 2
        assert stats["total_misses"] == 1
        assert stats["total_writes"] == 2
        assert stats["hit_rate_percent"] > 0

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = CacheManager(cache_dir=self.temp_dir)

        # Add items with different TTLs
        await cache.set("short_ttl", "value1", ttl=1)
        await cache.set("long_ttl", "value2", ttl=100)

        # Wait for short TTL to expire
        await asyncio.sleep(1.1)

        # Cleanup expired entries
        expired_count = await cache.cleanup_expired()

        assert expired_count >= 1
        assert await cache.get("short_ttl") is None
        assert await cache.get("long_ttl") == "value2"

    def test_cache_repr(self):
        """Test cache string representation."""
        cache = CacheManager(max_size=50, cache_dir=self.temp_dir)

        repr_str = repr(cache)

        assert "CacheManager" in repr_str
        assert "0/50" in repr_str  # size/max_size
        assert "%" in repr_str  # hit rate


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""

    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor(max_profiles=500)

        assert monitor.max_profiles == 500
        assert len(monitor.execution_profiles) == 0
        assert len(monitor.global_metrics) == 0
        assert "execution_time_warning" in monitor.thresholds

    @pytest.mark.asyncio
    async def test_profile_execution_success(self):
        """Test successful execution profiling."""
        monitor = PerformanceMonitor()

        async with monitor.profile_execution("test_operation") as profile:
            await asyncio.sleep(0.1)  # Simulate work
            assert profile.operation_name == "test_operation"

        # Check that profile was recorded
        assert len(monitor.execution_profiles) == 1
        profile = monitor.execution_profiles[0]
        assert profile.operation_name == "test_operation"
        assert profile.success is True
        assert profile.duration is not None
        assert profile.duration >= 0.1

    @pytest.mark.asyncio
    async def test_profile_execution_failure(self):
        """Test execution profiling with exception."""
        monitor = PerformanceMonitor()

        with pytest.raises(ValueError):
            async with monitor.profile_execution("failing_operation") as profile:
                await asyncio.sleep(0.05)
                raise ValueError("Test error")

        # Check that profile was recorded despite failure
        assert len(monitor.execution_profiles) == 1
        profile = monitor.execution_profiles[0]
        assert profile.operation_name == "failing_operation"
        assert profile.success is False
        assert "Test error" in profile.error

    def test_record_metric(self):
        """Test recording custom metrics."""
        monitor = PerformanceMonitor()

        monitor.record_metric("api_calls", 42, "count", {"endpoint": "/test"})

        assert len(monitor.global_metrics) == 1
        metric = monitor.global_metrics[0]
        assert metric.name == "api_calls"
        assert metric.value == 42
        assert metric.unit == "count"
        assert metric.context["endpoint"] == "/test"

    @pytest.mark.asyncio
    async def test_execution_stats(self):
        """Test execution statistics generation."""
        monitor = PerformanceMonitor()

        # Create some execution profiles
        async with monitor.profile_execution("fast_op"):
            await asyncio.sleep(0.01)

        async with monitor.profile_execution("slow_op"):
            await asyncio.sleep(0.05)

        try:
            async with monitor.profile_execution("failed_op"):
                raise Exception("Test failure")
        except Exception:
            pass

        stats = monitor.get_execution_stats()

        assert stats["total_operations"] == 3
        assert stats["successful_operations"] == 2
        assert stats["failed_operations"] == 1
        assert stats["success_rate_percent"] > 60
        assert stats["execution_times"]["average_seconds"] > 0

    def test_recent_profiles(self):
        """Test getting recent execution profiles."""
        monitor = PerformanceMonitor()

        # Manually add some profiles
        for i in range(5):
            profile = ExecutionProfile(
                operation_name=f"op_{i}",
                start_time=time.time(),
                end_time=time.time() + 0.1,
                duration=0.1,
                success=True,
            )
            monitor._add_profile(profile)

        recent = monitor.get_recent_profiles(limit=3)

        assert len(recent) == 3
        # Should be in reverse order (most recent first)
        assert recent[0]["operation_name"] == "op_4"
        assert recent[1]["operation_name"] == "op_3"
        assert recent[2]["operation_name"] == "op_2"

    def test_performance_warnings_slow_execution(self):
        """Test detection of slow execution warnings."""
        monitor = PerformanceMonitor()

        # Add a slow operation profile
        slow_profile = ExecutionProfile(
            operation_name="slow_operation",
            start_time=time.time(),
            end_time=time.time() + 10,  # 10 seconds (exceeds warning threshold)
            duration=10.0,
            success=True,
        )
        monitor._add_profile(slow_profile)

        warnings = monitor.get_performance_warnings()

        assert len(warnings) > 0
        warning = warnings[0]
        assert warning["type"] == "slow_execution"
        assert warning["operation"] == "slow_operation"
        assert warning["duration_seconds"] == 10.0
        assert warning["severity"] in ["warning", "critical"]

    def test_performance_warnings_high_memory(self):
        """Test detection of high memory usage warnings."""
        monitor = PerformanceMonitor()

        # Add a high memory usage profile
        memory_profile = ExecutionProfile(
            operation_name="memory_intensive",
            start_time=time.time(),
            end_time=time.time() + 1,
            duration=1.0,
            memory_peak=600.0,  # 600MB (exceeds warning threshold)
            success=True,
        )
        monitor._add_profile(memory_profile)

        warnings = monitor.get_performance_warnings()

        assert len(warnings) > 0
        warning = warnings[0]
        assert warning["type"] == "high_memory_usage"
        assert warning["operation"] == "memory_intensive"
        assert warning["memory_mb"] == 600.0

    @patch("psutil.Process")
    def test_system_info(self, mock_process_class):
        """Test system information gathering."""
        # Mock psutil.Process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.memory_info.return_value = Mock(
            rss=104857600, vms=209715200
        )  # 100MB RSS, 200MB VMS
        mock_process.memory_percent.return_value = 5.0
        mock_process.cpu_percent.return_value = 25.0
        mock_process.num_threads.return_value = 8
        mock_process_class.return_value = mock_process

        # Mock system-level functions
        with patch("psutil.cpu_count", return_value=8):
            with patch("psutil.virtual_memory") as mock_vm:
                mock_vm.return_value = Mock(
                    total=17179869184, available=8589934592
                )  # 16GB total, 8GB available
                with patch("psutil.disk_usage") as mock_disk:
                    mock_disk.return_value = Mock(percent=75.0)

                    monitor = PerformanceMonitor()
                    system_info = monitor.get_system_info()

        assert system_info["process_id"] == 12345
        assert system_info["memory"]["rss_mb"] == 100.0
        assert system_info["memory"]["percent"] == 5.0
        assert system_info["cpu"]["percent"] == 25.0
        assert system_info["system"]["cpu_count"] == 8

    def test_clear_metrics(self):
        """Test clearing all metrics and profiles."""
        monitor = PerformanceMonitor()

        # Add some data
        monitor.record_metric("test_metric", 1.0)
        profile = ExecutionProfile("test_op", time.time())
        monitor._add_profile(profile)

        assert len(monitor.global_metrics) == 1
        assert len(monitor.execution_profiles) == 1

        # Clear all metrics
        monitor.clear_metrics()

        assert len(monitor.global_metrics) == 0
        assert len(monitor.execution_profiles) == 0

    @patch("psutil.Process")
    def test_memory_and_cpu_usage_methods(self, mock_process_class):
        """Test internal memory and CPU usage methods."""
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=52428800)  # 50MB
        mock_process.cpu_percent.return_value = 15.0
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        memory_usage = monitor._get_memory_usage()
        cpu_usage = monitor._get_cpu_usage()

        assert memory_usage == 50.0  # 50MB
        assert cpu_usage == 15.0

    @patch("psutil.Process")
    def test_memory_and_cpu_usage_exception_handling(self, mock_process_class):
        """Test exception handling in memory and CPU usage methods."""
        mock_process = Mock()
        mock_process.memory_info.side_effect = Exception("Process access error")
        mock_process.cpu_percent.side_effect = Exception("Process access error")
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        memory_usage = monitor._get_memory_usage()
        cpu_usage = monitor._get_cpu_usage()

        assert memory_usage == 0.0
        assert cpu_usage == 0.0

    def test_add_profile_size_limit(self):
        """Test profile list size limit enforcement."""
        monitor = PerformanceMonitor(max_profiles=3)

        # Add more profiles than the limit
        for i in range(5):
            profile = ExecutionProfile(f"op_{i}", time.time())
            monitor._add_profile(profile)

        # Should only keep the last 3 profiles
        assert len(monitor.execution_profiles) == 3
        assert monitor.execution_profiles[0].operation_name == "op_2"
        assert monitor.execution_profiles[2].operation_name == "op_4"

    def test_performance_monitor_repr(self):
        """Test PerformanceMonitor string representation."""
        monitor = PerformanceMonitor()

        # Add some data
        monitor.record_metric("test", 1)
        profile = ExecutionProfile("test", time.time())
        monitor._add_profile(profile)

        repr_str = repr(monitor)

        assert "PerformanceMonitor" in repr_str
        assert "profiles=1" in repr_str
        assert "metrics=1" in repr_str


class TestPerformanceMetric:
    """Test cases for PerformanceMetric dataclass."""

    def test_performance_metric_creation(self):
        """Test PerformanceMetric creation."""
        from datetime import datetime

        timestamp = datetime.now()
        metric = PerformanceMetric(
            name="test_metric",
            value=42.5,
            unit="ms",
            timestamp=timestamp,
            context={"operation": "test"},
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.unit == "ms"
        assert metric.timestamp == timestamp
        assert metric.context["operation"] == "test"


class TestExecutionProfile:
    """Test cases for ExecutionProfile dataclass."""

    def test_execution_profile_creation(self):
        """Test ExecutionProfile creation."""
        start_time = time.time()
        profile = ExecutionProfile(
            operation_name="test_operation", start_time=start_time, memory_start=100.0
        )

        assert profile.operation_name == "test_operation"
        assert profile.start_time == start_time
        assert profile.memory_start == 100.0
        assert profile.success is True  # default
        assert profile.end_time is None  # default
        assert len(profile.metrics) == 0  # default


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=performance", "--cov-report=term-missing"])
