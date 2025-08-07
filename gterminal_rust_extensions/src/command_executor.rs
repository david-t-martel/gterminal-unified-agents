//! RustCommandExecutor: Secure command execution with process management

use crate::utils::{increment_ops, current_timestamp, ResultExt, RateLimiter};
use anyhow::{Context, Result};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::{Child, Command as AsyncCommand};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{timeout, sleep};

/// Process execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub duration: Duration,
    pub process_id: Option<u32>,
}

/// Running process information
#[derive(Debug)]
pub struct ProcessInfo {
    pub pid: u32,
    pub command: String,
    pub start_time: std::time::SystemTime,
    pub status: ProcessStatus,
    pub child: Option<Child>,
    pub stdout_receiver: Option<mpsc::UnboundedReceiver<String>>,
    pub stderr_receiver: Option<mpsc::UnboundedReceiver<String>>,
    pub kill_sender: Option<oneshot::Sender<()>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProcessStatus {
    Running,
    Completed,
    Failed,
    Killed,
    Timeout,
}

/// Command executor statistics
#[derive(Debug, Clone, Default)]
struct ExecutorStats {
    total_commands: u64,
    successful_commands: u64,
    failed_commands: u64,
    killed_commands: u64,
    timeout_commands: u64,
    total_execution_time_ms: u64,
}

/// Secure command executor with process management
#[pyclass]
pub struct RustCommandExecutor {
    runtime: tokio::runtime::Runtime,
    processes: Arc<RwLock<HashMap<u32, ProcessInfo>>>,
    stats: Arc<RwLock<ExecutorStats>>,
    allowed_commands: Arc<RwLock<Vec<String>>>,
    blocked_commands: Arc<RwLock<Vec<String>>>,
    rate_limiter: Option<RateLimiter>,
    max_processes: usize,
    default_timeout: Duration,
    working_directory: Option<std::path::PathBuf>,
    environment: Arc<RwLock<HashMap<String, String>>>,
}

#[pymethods]
impl RustCommandExecutor {
    /// Create new command executor
    #[new]
    #[pyo3(signature = (
        max_processes = 10,
        default_timeout_secs = 300,
        rate_limit_per_minute = None,
        working_directory = None
    ))]
    fn new(
        max_processes: usize,
        default_timeout_secs: u64,
        rate_limit_per_minute: Option<u32>,
        working_directory: Option<&str>,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .thread_name("cmd-executor")
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let wd = if let Some(dir) = working_directory {
            Some(std::path::PathBuf::from(dir))
        } else {
            None
        };

        let rate_limiter = rate_limit_per_minute.map(|limit| {
            RateLimiter::new(limit, 60) // Per minute
        });

        Ok(Self {
            runtime,
            processes: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ExecutorStats::default())),
            allowed_commands: Arc::new(RwLock::new(Vec::new())),
            blocked_commands: Arc::new(RwLock::new(Vec::new())),
            rate_limiter,
            max_processes,
            default_timeout: Duration::from_secs(default_timeout_secs),
            working_directory: wd,
            environment: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Execute command synchronously
    #[pyo3(signature = (
        command,
        args = None,
        timeout_secs = None,
        capture_output = true,
        check_allowed = true
    ))]
    fn execute(
        &self,
        command: &str,
        args: Option<Vec<&str>>,
        timeout_secs: Option<u64>,
        capture_output: bool,
        check_allowed: bool,
    ) -> PyResult<HashMap<String, PyObject>> {
        self.runtime.block_on(async {
            self.execute_async(command, args, timeout_secs, capture_output, check_allowed).await
        })
    }

    /// Execute command asynchronously (returns process ID)
    #[pyo3(signature = (
        command,
        args = None,
        timeout_secs = None,
        check_allowed = true
    ))]
    fn execute_async_bg(
        &self,
        command: &str,
        args: Option<Vec<&str>>,
        timeout_secs: Option<u64>,
        check_allowed: bool,
    ) -> PyResult<u32> {
        if check_allowed && !self.is_command_allowed(command)? {
            return Err(pyo3::exceptions::PyPermissionError::new_err(
                format!("Command not allowed: {}", command)
            ));
        }

        // Check rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if !limiter.check_rate_limit() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Rate limit exceeded"
                ));
            }
        }

        // Check max processes
        if self.processes.read().len() >= self.max_processes {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Maximum number of processes reached"
            ));
        }

        let timeout_duration = timeout_secs.map(Duration::from_secs)
            .unwrap_or(self.default_timeout);

        let command_str = format!("{} {}", command,
            args.as_ref().map(|a| a.join(" ")).unwrap_or_default());

        self.runtime.block_on(async {
            let (stdout_tx, stdout_rx) = mpsc::unbounded_channel();
            let (stderr_tx, stderr_rx) = mpsc::unbounded_channel();
            let (kill_tx, kill_rx) = oneshot::channel();

            let mut cmd = AsyncCommand::new(command);
            if let Some(ref args_vec) = args {
                cmd.args(args_vec);
            }

            // Set working directory
            if let Some(ref wd) = self.working_directory {
                cmd.current_dir(wd);
            }

            // Set environment variables
            {
                let env = self.environment.read();
                for (key, value) in env.iter() {
                    cmd.env(key, value);
                }
            }

            cmd.stdout(Stdio::piped())
               .stderr(Stdio::piped())
               .stdin(Stdio::null());

            let mut child = cmd.spawn()
                .with_context(|| format!("Failed to spawn command: {}", command))
                .to_py_err()?;

            let pid = child.id().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Failed to get process ID")
            })?;

            // Start output readers
            if let Some(stdout) = child.stdout.take() {
                let stdout_tx_clone = stdout_tx.clone();
                self.runtime.spawn(async move {
                    let mut reader = BufReader::new(stdout);
                    let mut line = String::new();
                    while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                        let _ = stdout_tx_clone.send(line.clone());
                        line.clear();
                    }
                });
            }

            if let Some(stderr) = child.stderr.take() {
                let stderr_tx_clone = stderr_tx.clone();
                self.runtime.spawn(async move {
                    let mut reader = BufReader::new(stderr);
                    let mut line = String::new();
                    while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                        let _ = stderr_tx_clone.send(line.clone());
                        line.clear();
                    }
                });
            }

            // Store process info
            let process_info = ProcessInfo {
                pid,
                command: command_str,
                start_time: std::time::SystemTime::now(),
                status: ProcessStatus::Running,
                child: Some(child),
                stdout_receiver: Some(stdout_rx),
                stderr_receiver: Some(stderr_rx),
                kill_sender: Some(kill_tx),
            };

            {
                let mut processes = self.processes.write();
                processes.insert(pid, process_info);
            }

            // Start timeout handler
            let processes_clone = Arc::clone(&self.processes);
            let stats_clone = Arc::clone(&self.stats);
            self.runtime.spawn(async move {
                tokio::select! {
                    _ = sleep(timeout_duration) => {
                        // Timeout - kill process
                        if let Some(mut process_info) = processes_clone.write().get_mut(&pid) {
                            process_info.status = ProcessStatus::Timeout;
                            if let Some(ref mut child) = process_info.child {
                                let _ = child.kill().await;
                            }
                        }

                        let mut stats = stats_clone.write();
                        stats.timeout_commands += 1;
                    }
                    _ = kill_rx => {
                        // Manual kill requested
                        if let Some(mut process_info) = processes_clone.write().get_mut(&pid) {
                            process_info.status = ProcessStatus::Killed;
                            if let Some(ref mut child) = process_info.child {
                                let _ = child.kill().await;
                            }
                        }

                        let mut stats = stats_clone.write();
                        stats.killed_commands += 1;
                    }
                }
            });

            increment_ops();
            Ok(pid)
        })
    }

    /// Get process status
    fn get_process_status(&self, pid: u32) -> PyResult<Option<String>> {
        let processes = self.processes.read();
        if let Some(process_info) = processes.get(&pid) {
            Ok(Some(format!("{:?}", process_info.status)))
        } else {
            Ok(None)
        }
    }

    /// Read stdout from running process
    #[pyo3(signature = (pid, max_lines = None))]
    fn read_stdout(&self, pid: u32, max_lines: Option<usize>) -> PyResult<Vec<String>> {
        self.runtime.block_on(async {
            let mut processes = self.processes.write();
            if let Some(process_info) = processes.get_mut(&pid) {
                if let Some(ref mut receiver) = process_info.stdout_receiver {
                    let mut lines = Vec::new();
                    let limit = max_lines.unwrap_or(100);

                    while lines.len() < limit {
                        match receiver.try_recv() {
                            Ok(line) => lines.push(line),
                            Err(_) => break,
                        }
                    }

                    Ok(lines)
                } else {
                    Ok(Vec::new())
                }
            } else {
                Err(pyo3::exceptions::PyKeyError::new_err(
                    format!("Process not found: {}", pid)
                ))
            }
        })
    }

    /// Read stderr from running process
    #[pyo3(signature = (pid, max_lines = None))]
    fn read_stderr(&self, pid: u32, max_lines: Option<usize>) -> PyResult<Vec<String>> {
        self.runtime.block_on(async {
            let mut processes = self.processes.write();
            if let Some(process_info) = processes.get_mut(&pid) {
                if let Some(ref mut receiver) = process_info.stderr_receiver {
                    let mut lines = Vec::new();
                    let limit = max_lines.unwrap_or(100);

                    while lines.len() < limit {
                        match receiver.try_recv() {
                            Ok(line) => lines.push(line),
                            Err(_) => break,
                        }
                    }

                    Ok(lines)
                } else {
                    Ok(Vec::new())
                }
            } else {
                Err(pyo3::exceptions::PyKeyError::new_err(
                    format!("Process not found: {}", pid)
                ))
            }
        })
    }

    /// Kill running process
    fn kill_process(&self, pid: u32) -> PyResult<bool> {
        self.runtime.block_on(async {
            let mut processes = self.processes.write();
            if let Some(process_info) = processes.get_mut(&pid) {
                if let Some(kill_sender) = process_info.kill_sender.take() {
                    let _ = kill_sender.send(());
                    Ok(true)
                } else {
                    Ok(false)
                }
            } else {
                Ok(false)
            }
        })
    }

    /// List running processes
    fn list_processes(&self) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let processes = self.processes.read();
        let mut result = Vec::new();

        Python::with_gil(|py| {
            for (pid, info) in processes.iter() {
                let mut process_dict = HashMap::new();

                process_dict.insert("pid".to_string(), (*pid).into_py(py));
                process_dict.insert("command".to_string(), info.command.clone().into_py(py));
                process_dict.insert("status".to_string(), format!("{:?}", info.status).into_py(py));

                if let Ok(duration) = info.start_time.elapsed() {
                    process_dict.insert("runtime_seconds".to_string(), duration.as_secs().into_py(py));
                }

                result.push(process_dict);
            }
        });

        Ok(result)
    }

    /// Wait for process to complete
    #[pyo3(signature = (pid, timeout_secs = None))]
    fn wait_for_process(&self, pid: u32, timeout_secs: Option<u64>) -> PyResult<HashMap<String, PyObject>> {
        self.runtime.block_on(async {
            let timeout_duration = timeout_secs.map(Duration::from_secs)
                .unwrap_or(Duration::from_secs(300));

            let start = Instant::now();

            // Poll for process completion
            loop {
                {
                    let mut processes = self.processes.write();
                    if let Some(process_info) = processes.get_mut(&pid) {
                        if let Some(ref mut child) = process_info.child {
                            match child.try_wait() {
                                Ok(Some(exit_status)) => {
                                    // Process completed
                                    process_info.status = if exit_status.success() {
                                        ProcessStatus::Completed
                                    } else {
                                        ProcessStatus::Failed
                                    };

                                    // Collect remaining output
                                    let mut stdout_lines = Vec::new();
                                    let mut stderr_lines = Vec::new();

                                    if let Some(ref mut stdout_rx) = process_info.stdout_receiver {
                                        while let Ok(line) = stdout_rx.try_recv() {
                                            stdout_lines.push(line);
                                        }
                                    }

                                    if let Some(ref mut stderr_rx) = process_info.stderr_receiver {
                                        while let Ok(line) = stderr_rx.try_recv() {
                                            stderr_lines.push(line);
                                        }
                                    }

                                    let duration = start.elapsed();

                                    // Update stats
                                    {
                                        let mut stats = self.stats.write();
                                        stats.total_commands += 1;
                                        stats.total_execution_time_ms += duration.as_millis() as u64;

                                        if exit_status.success() {
                                            stats.successful_commands += 1;
                                        } else {
                                            stats.failed_commands += 1;
                                        }
                                    }

                                    // Remove from active processes
                                    processes.remove(&pid);

                                    // Return result
                                    return Python::with_gil(|py| -> PyResult<HashMap<String, PyObject>> {
                                        let mut result = HashMap::new();
                                        result.insert("stdout".to_string(), stdout_lines.join("").into_py(py));
                                        result.insert("stderr".to_string(), stderr_lines.join("").into_py(py));
                                        result.insert("exit_code".to_string(), exit_status.code().into_py(py));
                                        result.insert("duration_ms".to_string(), duration.as_millis().into_py(py));
                                        result.insert("success".to_string(), exit_status.success().into_py(py));
                                        Ok(result)
                                    });
                                }
                                Ok(None) => {
                                    // Still running
                                }
                                Err(e) => {
                                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                        format!("Error checking process status: {}", e)
                                    ));
                                }
                            }
                        } else {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                "Process child handle not available"
                            ));
                        }
                    } else {
                        return Err(pyo3::exceptions::PyKeyError::new_err(
                            format!("Process not found: {}", pid)
                        ));
                    }
                }

                if start.elapsed() >= timeout_duration {
                    return Err(pyo3::exceptions::PyTimeoutError::new_err(
                        "Timeout waiting for process completion"
                    ));
                }

                sleep(Duration::from_millis(100)).await;
            }
        })
    }

    /// Set allowed commands (whitelist)
    fn set_allowed_commands(&self, commands: Vec<&str>) -> PyResult<()> {
        let mut allowed = self.allowed_commands.write();
        allowed.clear();
        allowed.extend(commands.into_iter().map(|s| s.to_string()));
        Ok(())
    }

    /// Set blocked commands (blacklist)
    fn set_blocked_commands(&self, commands: Vec<&str>) -> PyResult<()> {
        let mut blocked = self.blocked_commands.write();
        blocked.clear();
        blocked.extend(commands.into_iter().map(|s| s.to_string()));
        Ok(())
    }

    /// Set environment variable
    fn set_env(&self, key: &str, value: &str) -> PyResult<()> {
        let mut env = self.environment.write();
        env.insert(key.to_string(), value.to_string());
        Ok(())
    }

    /// Get environment variable
    fn get_env(&self, key: &str) -> PyResult<Option<String>> {
        let env = self.environment.read();
        Ok(env.get(key).cloned())
    }

    /// Set working directory
    fn set_working_directory(&mut self, path: &str) -> PyResult<()> {
        self.working_directory = Some(std::path::PathBuf::from(path));
        Ok(())
    }

    /// Get working directory
    fn get_working_directory(&self) -> PyResult<Option<String>> {
        Ok(self.working_directory.as_ref().map(|p| p.to_string_lossy().to_string()))
    }

    /// Get executor statistics
    fn get_stats(&self) -> PyResult<HashMap<String, u64>> {
        let stats = self.stats.read();
        let mut result = HashMap::new();

        result.insert("total_commands".to_string(), stats.total_commands);
        result.insert("successful_commands".to_string(), stats.successful_commands);
        result.insert("failed_commands".to_string(), stats.failed_commands);
        result.insert("killed_commands".to_string(), stats.killed_commands);
        result.insert("timeout_commands".to_string(), stats.timeout_commands);
        result.insert("total_execution_time_ms".to_string(), stats.total_execution_time_ms);
        result.insert("active_processes".to_string(), self.processes.read().len() as u64);

        // Calculate average execution time
        if stats.total_commands > 0 {
            result.insert("avg_execution_time_ms".to_string(),
                stats.total_execution_time_ms / stats.total_commands);
        }

        Ok(result)
    }

    /// Clear statistics
    fn clear_stats(&self) -> PyResult<()> {
        let mut stats = self.stats.write();
        *stats = ExecutorStats::default();
        Ok(())
    }

    /// Clean up completed processes
    fn cleanup_completed(&self) -> PyResult<usize> {
        let mut processes = self.processes.write();
        let initial_count = processes.len();

        processes.retain(|_, info| {
            matches!(info.status, ProcessStatus::Running)
        });

        let cleaned = initial_count - processes.len();
        Ok(cleaned)
    }
}

impl RustCommandExecutor {
    /// Execute command asynchronously (internal)
    async fn execute_async(
        &self,
        command: &str,
        args: Option<Vec<&str>>,
        timeout_secs: Option<u64>,
        capture_output: bool,
        check_allowed: bool,
    ) -> PyResult<HashMap<String, PyObject>> {
        if check_allowed && !self.is_command_allowed(command)? {
            return Err(pyo3::exceptions::PyPermissionError::new_err(
                format!("Command not allowed: {}", command)
            ));
        }

        // Check rate limit
        if let Some(ref limiter) = self.rate_limiter {
            limiter.wait_for_slot().await.to_py_err()?;
        }

        let timeout_duration = timeout_secs.map(Duration::from_secs)
            .unwrap_or(self.default_timeout);

        let start = Instant::now();

        let mut cmd = AsyncCommand::new(command);
        if let Some(ref args_vec) = args {
            cmd.args(args_vec);
        }

        // Set working directory
        if let Some(ref wd) = self.working_directory {
            cmd.current_dir(wd);
        }

        // Set environment variables
        {
            let env = self.environment.read();
            for (key, value) in env.iter() {
                cmd.env(key, value);
            }
        }

        if capture_output {
            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        }
        cmd.stdin(Stdio::null());

        let result = timeout(timeout_duration, cmd.output()).await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => {
                let success = output.status.success();

                // Update statistics
                {
                    let mut stats = self.stats.write();
                    stats.total_commands += 1;
                    stats.total_execution_time_ms += duration.as_millis() as u64;

                    if success {
                        stats.successful_commands += 1;
                    } else {
                        stats.failed_commands += 1;
                    }
                }

                increment_ops();

                Python::with_gil(|py| -> PyResult<HashMap<String, PyObject>> {
                    let mut result = HashMap::new();

                    if capture_output {
                        result.insert("stdout".to_string(),
                            String::from_utf8_lossy(&output.stdout).to_string().into_py(py));
                        result.insert("stderr".to_string(),
                            String::from_utf8_lossy(&output.stderr).to_string().into_py(py));
                    }

                    result.insert("exit_code".to_string(), output.status.code().into_py(py));
                    result.insert("success".to_string(), success.into_py(py));
                    result.insert("duration_ms".to_string(), duration.as_millis().into_py(py));

                    Ok(result)
                })
            }
            Ok(Err(e)) => {
                // Command failed to start
                {
                    let mut stats = self.stats.write();
                    stats.failed_commands += 1;
                    stats.total_commands += 1;
                }

                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Failed to execute command: {}", e)
                ))
            }
            Err(_) => {
                // Timeout
                {
                    let mut stats = self.stats.write();
                    stats.timeout_commands += 1;
                    stats.total_commands += 1;
                }

                Err(pyo3::exceptions::PyTimeoutError::new_err(
                    format!("Command timed out after {} seconds", timeout_duration.as_secs())
                ))
            }
        }
    }

    /// Check if command is allowed
    fn is_command_allowed(&self, command: &str) -> PyResult<bool> {
        // Check blacklist first
        {
            let blocked = self.blocked_commands.read();
            if !blocked.is_empty() {
                for blocked_cmd in blocked.iter() {
                    if command.contains(blocked_cmd) {
                        return Ok(false);
                    }
                }
            }
        }

        // Check whitelist if not empty
        {
            let allowed = self.allowed_commands.read();
            if !allowed.is_empty() {
                for allowed_cmd in allowed.iter() {
                    if command.starts_with(allowed_cmd) {
                        return Ok(true);
                    }
                }
                return Ok(false); // Not in whitelist
            }
        }

        // Default: allow if no restrictions set
        Ok(true)
    }
}

impl Drop for RustCommandExecutor {
    fn drop(&mut self) {
        // Kill all remaining processes
        let processes = self.processes.read();
        for (pid, _) in processes.iter() {
            let _ = self.kill_process(*pid);
        }
        tracing::debug!("RustCommandExecutor dropped, killed {} processes", processes.len());
    }
}