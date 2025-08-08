//! Secure command execution module
//!
//! This module provides secure command execution with:
//! - Allowlist-based command validation
//! - Input sanitization and validation
//! - Timeout enforcement
//! - Output size limits
//! - Environment variable isolation

use crate::error::{FsError, FsResult};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;

/// Result of command execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ExecResult {
    #[pyo3(get)]
    pub stdout: String,
    #[pyo3(get)]
    pub stderr: String,
    #[pyo3(get)]
    pub exit_code: i32,
}

#[pymethods]
impl ExecResult {
    fn __repr__(&self) -> String {
        format!(
            "ExecResult(exit_code={}, stdout_len={}, stderr_len={})",
            self.exit_code,
            self.stdout.len(),
            self.stderr.len()
        )
    }
}

/// Restricted set of allowed commands for security
static ALLOWED_COMMANDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    let commands = [
        // File system operations
        "ls", "cat", "grep", "find", "wc", "head", "tail", "echo", "date",
        "pwd", "whoami", "hostname", "uname", "df", "du", "ps", "top",
        "chmod", "chown", "chgrp", "mkdir", "rmdir", "rm", "cp", "mv", "ln",
        "touch", "test", "[", "true", "false", "sleep", "kill", "pkill",
        "file", "stat", "realpath", "dirname", "basename", "readlink",

        // Development tools
        "git", "cargo", "rustc", "npm", "node", "python", "python3", "pip", "pip3",
        "make", "cmake", "gcc", "g++", "clang", "clang++", "go", "ruby", "gem",
        "java", "javac", "mvn", "gradle", "dotnet", "php", "composer",
        "code", "vim", "nano", "emacs", "nvim", "subl", "atom",

        // Container and cloud tools
        "docker", "docker-compose", "kubectl", "terraform", "ansible", "vagrant",
        "podman", "buildah", "skopeo", "helm", "minikube", "kind",
        "gh", "glab", "hub", "az", "aws", "gcloud", "heroku", "flyctl",

        // Text processing
        "jq", "yq", "awk", "sed", "sort", "uniq", "cut", "tr", "tee", "paste",
        "column", "join", "comm", "diff", "patch", "xargs", "parallel",

        // Network tools (excluding dangerous ones like nc, ncat, telnet, ssh)
        "curl", "wget", "ping", "nslookup", "dig", "host", "traceroute", "netstat",
        "ss", "ip", "ifconfig", "route", "arp",
        "openssl", "gpg", "base64",

        // Archive tools
        "tar", "gzip", "gunzip", "zip", "unzip", "bzip2", "bunzip2", "xz", "unxz",
        "7z", "rar", "unrar", "zstd", "lz4", "pigz",

        // Checksum tools
        "md5sum", "sha1sum", "sha256sum", "sha512sum", "shasum", "cksum",

        // System tools
        "env", "printenv", "which", "whereis", "type", "command",
        "systemctl", "journalctl", "service", "lsof", "strace", "ltrace",
        "htop", "iotop", "vmstat", "iostat", "mpstat", "free", "uptime",
        "dmesg", "lsblk", "lspci", "lsusb", "lscpu", "lshw", "dmidecode",

        // Package managers
        "apt", "apt-get", "apt-cache", "dpkg", "yum", "dnf", "rpm", "zypper",
        "pacman", "brew", "snap", "flatpak", "nix", "apk", "emerge",

        // UV Python package manager
        "uv",

        // Additional useful commands
        "tree", "watch", "time", "timeout", "tty", "script", "screen", "tmux",
        "less", "more", "nl", "fmt", "fold", "expand", "unexpand", "pr",
        "od", "hexdump", "xxd", "strings", "split", "csplit", "shuf",
        "factor", "seq", "expr", "bc", "dc", "units", "cal", "date",
        "uuidgen", "mktemp", "yes", "rev", "tac", "paste", "look",
    ];
    commands.into_iter().collect()
});

/// Maximum output size (10MB)
const MAX_OUTPUT_SIZE: usize = 10 * 1024 * 1024;

/// Check if a string contains shell metacharacters that could be dangerous
fn contains_shell_metacharacters(s: &str) -> bool {
    // List of dangerous shell metacharacters
    const DANGEROUS_CHARS: &[char] = &[
        ';', '&', '|', '`', '$', '(', ')', '{', '}', '[', ']',
        '<', '>', '\n', '\r', '*', '?', '~', '!', '#', '%', '\\',
        '"', '\'', '\0', '\t', '\x0B', '\x0C', // Include all whitespace control chars
    ];

    // Check for dangerous patterns
    s.chars().any(|c| DANGEROUS_CHARS.contains(&c)) ||
    s.contains("..") ||  // Directory traversal
    s.contains("//") ||  // Double slashes can be problematic
    s.contains("\\\\") || // Double backslashes
    s.starts_with('-') || // Could be interpreted as command flag
    s.contains("$(") ||   // Command substitution
    s.contains("${") ||   // Variable expansion
    s.contains("[[") ||   // Bash conditional
    s.contains("]]") ||   // Bash conditional
    s.contains("&&") ||   // Command chaining
    s.contains("||") ||   // Command chaining
    s.contains(">>") ||   // Append redirect
    s.contains("<<") ||   // Here document
    s.contains("2>") ||   // Stderr redirect
    s.contains("&>") ||   // Combined redirect
    s.contains(">&") ||   // Redirect descriptor
    s.as_bytes().iter().any(|&b| b < 32 || b == 127) // Control characters
}

/// Find an executable in the system PATH
pub fn find_executable(command: &str) -> FsResult<std::path::PathBuf> {
    which::which(command)
        .map_err(|_| FsError::CommandNotFound(command.to_string()))
}

/// Check if a command is allowed based on security restrictions
pub fn is_command_allowed(command: &str) -> bool {
    let cmd = command.trim();

    // Security: Reject if command contains null bytes or control characters
    if cmd.is_empty() || cmd.as_bytes().iter().any(|&b| b == 0 || (b < 32 && b != 9) || b == 127) {
        return false;
    }

    // Security: Reject relative paths that try to escape
    if cmd.contains("../") || cmd.contains("..\\") {
        return false;
    }

    // Extract the base command name (handle both absolute and relative paths)
    let base_cmd = if cmd.contains('/') || cmd.contains('\\') {
        // It's a path, get the last component
        cmd.split(&['/', '\\'][..]).last().unwrap_or(cmd)
    } else {
        // It's just a command name
        cmd
    };

    let base_cmd_lower = base_cmd.to_lowercase();

    // Check if the command (or its base name) is in the allowed list
    // O(1) lookup instead of O(n) linear search
    ALLOWED_COMMANDS.contains(base_cmd_lower.as_str()) ||
    base_cmd_lower.split_whitespace().next()
        .map(|cmd| ALLOWED_COMMANDS.contains(cmd))
        .unwrap_or(false)
}

/// Execute a command with timeout and output limits
pub async fn exec_command(
    command: String,
    args: Vec<String>,
    timeout_ms: Option<u64>,
) -> FsResult<ExecResult> {
    // Security: Validate command doesn't contain shell metacharacters
    if contains_shell_metacharacters(&command) {
        return Err(FsError::InvalidInput(format!(
            "Command contains shell metacharacters: '{}'",
            command
        )));
    }

    // Security: Validate each argument
    for arg in &args {
        if contains_shell_metacharacters(arg) {
            return Err(FsError::InvalidInput(format!(
                "Argument contains shell metacharacters: '{}'",
                arg
            )));
        }
    }

    // Security check
    if !is_command_allowed(&command) {
        return Err(FsError::InvalidInput(format!(
            "Command '{}' is not in the allowed list for security reasons",
            command
        )));
    }

    let timeout_duration = Duration::from_millis(timeout_ms.unwrap_or(30000));

    let mut cmd = Command::new(&command);
    cmd.args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        // Security: Clear environment to prevent variable injection
        .env_clear()
        // Add back minimal safe environment
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", std::env::var("HOME").unwrap_or_default())
        .env("USER", std::env::var("USER").unwrap_or_default());

    let mut child = cmd.spawn()
        .map_err(|e| FsError::ExecutionError(format!("Failed to spawn command: {}", e)))?;

    let stdout = child.stdout.take()
        .ok_or_else(|| FsError::ExecutionError("Failed to capture stdout".to_string()))?;
    let stderr = child.stderr.take()
        .ok_or_else(|| FsError::ExecutionError("Failed to capture stderr".to_string()))?;

    let stdout_reader = BufReader::new(stdout);
    let stderr_reader = BufReader::new(stderr);

    let handle = tokio::spawn(async move {
        let mut stdout_lines = stdout_reader.lines();
        let mut stderr_lines = stderr_reader.lines();
        let mut stdout_output = Vec::new();
        let mut stderr_output = Vec::new();
        let mut total_size = 0;

        loop {
            tokio::select! {
                Ok(Some(line)) = stdout_lines.next_line() => {
                    total_size += line.len();
                    if total_size > MAX_OUTPUT_SIZE {
                        return Err(FsError::ExecutionError("Output size limit exceeded".to_string()));
                    }
                    stdout_output.push(line);
                }
                Ok(Some(line)) = stderr_lines.next_line() => {
                    total_size += line.len();
                    if total_size > MAX_OUTPUT_SIZE {
                        return Err(FsError::ExecutionError("Output size limit exceeded".to_string()));
                    }
                    stderr_output.push(line);
                }
                else => {
                    // Both streams are exhausted, wait for process to complete
                    let status = child.wait().await
                        .map_err(|e| FsError::ExecutionError(format!("Failed to wait for command: {}", e)))?;

                    return Ok(ExecResult {
                        stdout: stdout_output.join("\n"),
                        stderr: stderr_output.join("\n"),
                        exit_code: status.code().unwrap_or(-1),
                    });
                }
            }
        }
    });

    match timeout(timeout_duration, handle).await {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => Err(FsError::ExecutionError(format!("Task join error: {}", e))),
        Err(_) => {
            // Timeout occurred - attempt to kill the process
            // Note: The child process handle is moved into the spawned task,
            // so we can't kill it directly here. In a production system,
            // you'd want to restructure this to maintain process control.
            Err(FsError::ExecutionError(format!(
                "Command execution timed out after {} ms",
                timeout_ms.unwrap_or(30000)
            )))
        }
    }
}

/// PyO3 wrapper for secure command execution
#[pyclass]
pub struct RustCommandExecutor {
    timeout_ms: u64,
}

#[pymethods]
impl RustCommandExecutor {
    #[new]
    #[pyo3(signature = (timeout_ms=30000))]
    fn new(timeout_ms: u64) -> Self {
        Self { timeout_ms }
    }

    /// Execute a command with arguments
    fn execute_command<'py>(
        &self,
        py: Python<'py>,
        command: String,
        args: Vec<String>,
    ) -> PyResult<&'py PyAny> {
        let timeout = self.timeout_ms;
        pyo3_asyncio::tokio::return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("Async functionality temporarily disabled in PyO3 0.22"));// ASYNC DISABLED: {
            exec_command(command, args, Some(timeout))
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    /// Check if a command is allowed
    fn is_allowed(&self, command: &str) -> bool {
        is_command_allowed(command)
    }

    /// Find executable in PATH
    fn find_executable(&self, command: &str) -> PyResult<String> {
        find_executable(command)
            .map(|p| p.to_string_lossy().to_string())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(e.to_string()))
    }

    /// Get list of allowed commands (for debugging/inspection)
    fn get_allowed_commands(&self) -> Vec<String> {
        ALLOWED_COMMANDS.iter().map(|&s| s.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_allowed() {
        assert!(is_command_allowed("ls"));
        assert!(is_command_allowed("ls -la"));
        assert!(is_command_allowed("/bin/ls"));
        assert!(is_command_allowed("git status"));
        assert!(is_command_allowed("uv"));
        assert!(!is_command_allowed("nc"));
        assert!(!is_command_allowed("netcat"));
        assert!(!is_command_allowed("../../bin/evil"));
    }

    #[test]
    fn test_shell_metacharacters() {
        assert!(contains_shell_metacharacters("ls; rm -rf /"));
        assert!(contains_shell_metacharacters("cat $(echo test)"));
        assert!(contains_shell_metacharacters("ls | grep test"));
        assert!(!contains_shell_metacharacters("ls -la"));
        assert!(!contains_shell_metacharacters("python3 script.py"));
    }

    #[tokio::test]
    async fn test_execute_simple_command() {
        let result = exec_command("echo".to_string(), vec!["hello".to_string()], None).await;
        assert!(result.is_ok());

        let exec_result = result.unwrap();
        assert_eq!(exec_result.stdout.trim(), "hello");
        assert_eq!(exec_result.exit_code, 0);
    }

    #[tokio::test]
    async fn test_execute_disallowed_command() {
        let result = exec_command("netcat".to_string(), vec![], None).await;
        assert!(result.is_err());
        assert!(matches!(result, Err(FsError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_execute_with_timeout() {
        let result = exec_command(
            "sleep".to_string(),
            vec!["10".to_string()],
            Some(100)
        ).await;
        assert!(result.is_err());
        assert!(matches!(result, Err(FsError::ExecutionError(_))));
    }
}
