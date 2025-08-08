//! Utility functions for performance monitoring and system operations
//!
//! This module provides PyO3 bindings for various utility operations:
//! - System information and monitoring
//! - Performance metrics collection
//! - Benchmarking operations
//! - Resource management

use pyo3::prelude::*;
// Removed pyo3_asyncio - using sync operations with PyO3 0.22
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::sync::RwLock;

/// Get comprehensive system information
#[pyfunction]
pub fn system_info(py: Python<'_>) -> PyResult<HashMap<String, PyObject>> {
    let mut sys = System::new_all();
    sys.refresh_all();

    let mut info = HashMap::new();

    // System information
    info.insert(
        "hostname".to_string(),
        System::host_name().unwrap_or_default().to_object(py),
    );
    info.insert(
        "kernel_version".to_string(),
        System::kernel_version().unwrap_or_default().to_object(py),
    );
    info.insert(
        "os_version".to_string(),
        System::long_os_version().unwrap_or_default().to_object(py),
    );
    info.insert(
        "architecture".to_string(),
        std::env::consts::ARCH.to_object(py),
    );

    // Memory information
    info.insert("total_memory".to_string(), sys.total_memory().to_object(py));
    info.insert("used_memory".to_string(), sys.used_memory().to_object(py));
    info.insert(
        "available_memory".to_string(),
        sys.available_memory().to_object(py),
    );
    info.insert("total_swap".to_string(), sys.total_swap().to_object(py));
    info.insert("used_swap".to_string(), sys.used_swap().to_object(py));

    // CPU information
    let cpus = sys.cpus();
    info.insert("cpu_count".to_string(), cpus.len().to_object(py));
    if let Some(cpu) = cpus.first() {
        info.insert("cpu_brand".to_string(), cpu.brand().to_object(py));
        info.insert("cpu_frequency".to_string(), cpu.frequency().to_object(py));
    }

    // System load average (Unix-like systems)
    let load_avg = System::load_average();
    let mut load_info = HashMap::new();
    load_info.insert("one".to_string(), load_avg.one.to_object(py));
    load_info.insert("five".to_string(), load_avg.five.to_object(py));
    load_info.insert("fifteen".to_string(), load_avg.fifteen.to_object(py));
    info.insert("load_average".to_string(), load_info.to_object(py));

    // Process count
    info.insert(
        "process_count".to_string(),
        sys.processes().len().to_object(py),
    );

    // Boot time
    info.insert("boot_time".to_string(), System::boot_time().to_object(py));

    // Current process information
    let current_pid = std::process::id();
    if let Some(process) = sys.process(sysinfo::Pid::from(current_pid as usize)) {
        let mut proc_info = HashMap::new();
        proc_info.insert("pid".to_string(), current_pid.to_object(py));
        proc_info.insert("name".to_string(), process.name().to_object(py));
        proc_info.insert("memory".to_string(), process.memory().to_object(py));
        proc_info.insert(
            "virtual_memory".to_string(),
            process.virtual_memory().to_object(py),
        );
        proc_info.insert("cpu_usage".to_string(), process.cpu_usage().to_object(py));
        proc_info.insert("start_time".to_string(), process.start_time().to_object(py));
        info.insert("current_process".to_string(), proc_info.to_object(py));
    }

    // Disk information - using static method
    let disks: Vec<HashMap<String, PyObject>> = sysinfo::Disks::new_with_refreshed_list()
        .iter()
        .map(|disk| {
            let mut disk_info = HashMap::new();
            disk_info.insert(
                "name".to_string(),
                disk.name().to_string_lossy().to_string().to_object(py),
            );
            disk_info.insert(
                "mount_point".to_string(),
                disk.mount_point()
                    .to_string_lossy()
                    .to_string()
                    .to_object(py),
            );
            disk_info.insert("total_space".to_string(), disk.total_space().to_object(py));
            disk_info.insert(
                "available_space".to_string(),
                disk.available_space().to_object(py),
            );
            disk_info.insert(
                "file_system".to_string(),
                disk.file_system()
                    .to_string_lossy()
                    .to_string()
                    .to_object(py),
            );
            disk_info.insert(
                "is_removable".to_string(),
                disk.is_removable().to_object(py),
            );
            disk_info
        })
        .collect();
    info.insert("disks".to_string(), disks.to_object(py));

    // Network interfaces - using static method
    let networks: Vec<HashMap<String, PyObject>> = sysinfo::Networks::new_with_refreshed_list()
        .iter()
        .map(|(interface_name, network)| {
            let mut net_info = HashMap::new();
            net_info.insert("interface".to_string(), interface_name.to_object(py));
            net_info.insert("received".to_string(), network.received().to_object(py));
            net_info.insert(
                "transmitted".to_string(),
                network.transmitted().to_object(py),
            );
            net_info.insert(
                "packets_received".to_string(),
                network.packets_received().to_object(py),
            );
            net_info.insert(
                "packets_transmitted".to_string(),
                network.packets_transmitted().to_object(py),
            );
            net_info.insert(
                "errors_on_received".to_string(),
                network.errors_on_received().to_object(py),
            );
            net_info.insert(
                "errors_on_transmitted".to_string(),
                network.errors_on_transmitted().to_object(py),
            );
            net_info
        })
        .collect();
    info.insert("networks".to_string(), networks.to_object(py));

    Ok(info)
}

/// Performance metrics collector
#[pyclass]
pub struct RustPerformanceMetrics {
    start_time: Instant,
    metrics: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    collection_interval: Duration,
}

#[pymethods]
impl RustPerformanceMetrics {
    #[new]
    #[pyo3(signature = (collection_interval_seconds=1))]
    fn new(collection_interval_seconds: u64) -> Self {
        Self {
            start_time: Instant::now(),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            collection_interval: Duration::from_secs(collection_interval_seconds),
        }
    }

    /// Record a metric value
    fn record_metric<'py>(
        &self,
        py: Python<'py>,
        name: String,
        value: f64,
    ) -> PyResult<&'py PyAny> {
        let metrics = self.metrics.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let mut metrics_guard = metrics.write().await;
            metrics_guard
                .entry(name)
                .or_insert_with(Vec::new)
                .push(value);
            Ok(true)
        })
    }

    /// Get metric statistics
    fn get_metric_stats<'py>(&self, py: Python<'py>, name: String) -> PyResult<&'py PyAny> {
        let metrics = self.metrics.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let metrics_guard = metrics.read().await;

            if let Some(values) = metrics_guard.get(&name) {
                if values.is_empty() {
                    return Ok(None::<HashMap<String, f64>>);
                }

                let count = values.len() as f64;
                let sum: f64 = values.iter().sum();
                let mean = sum / count;

                let mut sorted_values = values.clone();
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let min = sorted_values[0];
                let max = sorted_values[sorted_values.len() - 1];

                let median = if sorted_values.len() % 2 == 0 {
                    let mid = sorted_values.len() / 2;
                    (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
                } else {
                    sorted_values[sorted_values.len() / 2]
                };

                let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count;
                let std_dev = variance.sqrt();

                let p95_idx = ((count * 0.95) as usize).min(sorted_values.len() - 1);
                let p99_idx = ((count * 0.99) as usize).min(sorted_values.len() - 1);

                let mut stats = HashMap::new();
                stats.insert("count".to_string(), count);
                stats.insert("sum".to_string(), sum);
                stats.insert("mean".to_string(), mean);
                stats.insert("median".to_string(), median);
                stats.insert("min".to_string(), min);
                stats.insert("max".to_string(), max);
                stats.insert("std_dev".to_string(), std_dev);
                stats.insert("p95".to_string(), sorted_values[p95_idx]);
                stats.insert("p99".to_string(), sorted_values[p99_idx]);

                Ok(Some(stats))
            } else {
                Ok(None)
            }
        })
    }

    /// Get all metric names
    fn get_metric_names<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let metrics = self.metrics.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let metrics_guard = metrics.read().await;
            let names: Vec<String> = metrics_guard.keys().cloned().collect();
            Ok(names)
        })
    }

    /// Clear all metrics
    fn clear_metrics<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let metrics = self.metrics.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let mut metrics_guard = metrics.write().await;
            metrics_guard.clear();
            Ok(true)
        })
    }

    /// Get uptime in seconds
    fn get_uptime(&self) -> PyResult<f64> {
        Ok(self.start_time.elapsed().as_secs_f64())
    }

    /// Export metrics to dictionary
    fn export_metrics<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let metrics = self.metrics.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let metrics_guard = metrics.read().await;
            let exported: HashMap<String, Vec<f64>> = metrics_guard.clone();
            Ok(exported)
        })
    }
}

/// Benchmark operation performance
#[pyfunction]
pub fn benchmark_operation<'py>(
    py: Python<'py>,
    operation_name: String,
    iterations: usize,
) -> PyResult<&'py PyAny> {
    return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
        let mut timings = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();

            // Simulate operation - in practice, this would call the actual operation
            tokio::time::sleep(Duration::from_micros(1)).await;

            let duration = start.elapsed();
            timings.push(duration.as_nanos() as f64);
        }

        // Calculate statistics
        let count = timings.len() as f64;
        let sum: f64 = timings.iter().sum();
        let mean = sum / count;

        let mut sorted_timings = timings.clone();
        sorted_timings.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_timings[0];
        let max = sorted_timings[sorted_timings.len() - 1];

        let median = if sorted_timings.len() % 2 == 0 {
            let mid = sorted_timings.len() / 2;
            (sorted_timings[mid - 1] + sorted_timings[mid]) / 2.0
        } else {
            sorted_timings[sorted_timings.len() / 2]
        };

        let variance: f64 = timings.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count;
        let std_dev = variance.sqrt();

        let p95_idx = ((count * 0.95) as usize).min(sorted_timings.len() - 1);
        let p99_idx = ((count * 0.99) as usize).min(sorted_timings.len() - 1);

        let mut results = HashMap::new();
        results.insert("operation".to_string(), operation_name);
        results.insert("iterations".to_string(), (iterations as f64).to_string());
        results.insert("total_time_ns".to_string(), sum.to_string());
        results.insert("mean_time_ns".to_string(), mean.to_string());
        results.insert("median_time_ns".to_string(), median.to_string());
        results.insert("min_time_ns".to_string(), min.to_string());
        results.insert("max_time_ns".to_string(), max.to_string());
        results.insert("std_dev_ns".to_string(), std_dev.to_string());
        results.insert(
            "p95_time_ns".to_string(),
            sorted_timings[p95_idx].to_string(),
        );
        results.insert(
            "p99_time_ns".to_string(),
            sorted_timings[p99_idx].to_string(),
        );
        results.insert(
            "ops_per_second".to_string(),
            (1_000_000_000.0 / mean).to_string(),
        );

        Ok(results)
    })
}

/// Get performance metrics for the current system
#[pyfunction]
pub fn performance_metrics(py: Python<'_>) -> PyResult<HashMap<String, PyObject>> {
    let mut sys = System::new();
    sys.refresh_cpu_all();
    sys.refresh_memory();
    sys.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::All,
        sysinfo::ProcessRefreshKind::everything(),
    );

    let mut metrics = HashMap::new();

    // CPU metrics
    let cpu_usage: Vec<f32> = sys.cpus().iter().map(|cpu| cpu.cpu_usage()).collect();
    let avg_cpu_usage: f32 = cpu_usage.iter().sum::<f32>() / cpu_usage.len() as f32;

    metrics.insert("cpu_usage_percent".to_string(), avg_cpu_usage.to_object(py));
    metrics.insert("cpu_cores".to_string(), sys.cpus().len().to_object(py));

    // Memory metrics
    let memory_usage_percent = (sys.used_memory() as f64 / sys.total_memory() as f64) * 100.0;
    metrics.insert(
        "memory_usage_percent".to_string(),
        memory_usage_percent.to_object(py),
    );
    metrics.insert(
        "memory_used_mb".to_string(),
        (sys.used_memory() / 1024 / 1024).to_object(py),
    );
    metrics.insert(
        "memory_total_mb".to_string(),
        (sys.total_memory() / 1024 / 1024).to_object(py),
    );
    metrics.insert(
        "memory_available_mb".to_string(),
        (sys.available_memory() / 1024 / 1024).to_object(py),
    );

    // Swap metrics
    if sys.total_swap() > 0 {
        let swap_usage_percent = (sys.used_swap() as f64 / sys.total_swap() as f64) * 100.0;
        metrics.insert(
            "swap_usage_percent".to_string(),
            swap_usage_percent.to_object(py),
        );
        metrics.insert(
            "swap_used_mb".to_string(),
            (sys.used_swap() / 1024 / 1024).to_object(py),
        );
        metrics.insert(
            "swap_total_mb".to_string(),
            (sys.total_swap() / 1024 / 1024).to_object(py),
        );
    }

    // Load average
    let load_avg = System::load_average();
    metrics.insert("load_average_1m".to_string(), load_avg.one.to_object(py));
    metrics.insert("load_average_5m".to_string(), load_avg.five.to_object(py));
    metrics.insert(
        "load_average_15m".to_string(),
        load_avg.fifteen.to_object(py),
    );

    // Process metrics
    let process_count = sys.processes().len();
    metrics.insert("process_count".to_string(), process_count.to_object(py));

    // Current process metrics
    let current_pid = std::process::id();
    if let Some(process) = sys.process(sysinfo::Pid::from(current_pid as usize)) {
        metrics.insert(
            "current_process_memory_mb".to_string(),
            (process.memory() / 1024 / 1024).to_object(py),
        );
        metrics.insert(
            "current_process_cpu_percent".to_string(),
            process.cpu_usage().to_object(py),
        );
    }

    // Disk metrics
    let total_disk_space = 0u64;
    let available_disk_space = 0u64;

    // Note: disks() method changed in newer sysinfo versions
    // for disk in sys.disks() {
    //     total_disk_space += disk.total_space();
    //     available_disk_space += disk.available_space();
    // }

    if total_disk_space > 0 {
        let disk_usage_percent =
            ((total_disk_space - available_disk_space) as f64 / total_disk_space as f64) * 100.0;
        metrics.insert(
            "disk_usage_percent".to_string(),
            disk_usage_percent.to_object(py),
        );
        metrics.insert(
            "disk_total_gb".to_string(),
            (total_disk_space / 1024 / 1024 / 1024).to_object(py),
        );
        metrics.insert(
            "disk_available_gb".to_string(),
            (available_disk_space / 1024 / 1024 / 1024).to_object(py),
        );
    }

    // Timestamp
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    metrics.insert("timestamp".to_string(), timestamp.to_object(py));

    Ok(metrics)
}

/// Resource monitor that tracks system resources over time
#[pyclass]
pub struct RustResourceMonitor {
    start_time: Instant,
    samples: Arc<RwLock<Vec<HashMap<String, f64>>>>,
    max_samples: usize,
    collection_interval: Duration,
}

#[pymethods]
impl RustResourceMonitor {
    #[new]
    #[pyo3(signature = (max_samples=3600, collection_interval_seconds=1))]
    fn new(max_samples: usize, collection_interval_seconds: u64) -> Self {
        Self {
            start_time: Instant::now(),
            samples: Arc::new(RwLock::new(Vec::with_capacity(max_samples))),
            max_samples,
            collection_interval: Duration::from_secs(collection_interval_seconds),
        }
    }

    /// Start resource monitoring
    fn start_monitoring<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let samples = self.samples.clone();
        let max_samples = self.max_samples;
        let interval = self.collection_interval;

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let mut sys = System::new();

            loop {
                sys.refresh_all();

                let mut sample = HashMap::new();

                // Collect metrics
                let cpu_usage: Vec<f32> = sys.cpus().iter().map(|cpu| cpu.cpu_usage()).collect();
                let avg_cpu_usage: f32 = cpu_usage.iter().sum::<f32>() / cpu_usage.len() as f32;
                sample.insert("cpu_usage".to_string(), avg_cpu_usage as f64);

                let memory_usage = (sys.used_memory() as f64 / sys.total_memory() as f64) * 100.0;
                sample.insert("memory_usage".to_string(), memory_usage);

                let load_avg = System::load_average();
                sample.insert("load_average_1m".to_string(), load_avg.one);

                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
                sample.insert("timestamp".to_string(), timestamp);

                // Store sample
                {
                    let mut samples_guard = samples.write().await;
                    samples_guard.push(sample);

                    // Keep only the last max_samples
                    if samples_guard.len() > max_samples {
                        samples_guard.remove(0);
                    }
                }

                tokio::time::sleep(interval).await;
            }

            #[allow(unreachable_code)]
            Ok(())
        })
    }

    /// Get collected samples
    fn get_samples<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let samples = self.samples.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let samples_guard = samples.read().await;
            let collected_samples: Vec<HashMap<String, f64>> = samples_guard.clone();
            Ok(collected_samples)
        })
    }

    /// Get monitoring statistics
    fn get_monitoring_stats<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let samples = self.samples.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let samples_guard = samples.read().await;

            let mut stats = HashMap::new();
            stats.insert("sample_count".to_string(), samples_guard.len() as f64);
            stats.insert("max_samples".to_string(), 3600.0f64); // max_samples as f64
            stats.insert("uptime_seconds".to_string(), 0.0f64); // Would need access to start_time

            if !samples_guard.is_empty() {
                let first_timestamp = samples_guard[0].get("timestamp").unwrap_or(&0.0);
                let last_timestamp = samples_guard[samples_guard.len() - 1]
                    .get("timestamp")
                    .unwrap_or(&0.0);
                stats.insert(
                    "monitoring_duration".to_string(),
                    last_timestamp - first_timestamp,
                );
            }

            Ok(stats)
        })
    }

    /// Clear collected samples
    fn clear_samples<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let samples = self.samples.clone();

        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            let mut samples_guard = samples.write().await;
            samples_guard.clear();
            Ok(true)
        })
    }
}
