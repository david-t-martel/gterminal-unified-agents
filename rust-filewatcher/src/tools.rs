/*!
 * Tool integration module for gterminal-filewatcher
 *
 * Provides high-performance execution of analysis tools including ruff, mypy,
 * AST-grep, TypeScript compiler, and integration with rufft-claude.sh for
 * intelligent auto-fixing capabilities.
 */

use crate::config::{Config, ToolConfig};
use crate::types::*;

use anyhow::{Context, Result};
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tokio::process::Command as AsyncCommand;
use tracing::{debug, error, info, trace, warn};

/// Tool executor for running analysis tools
pub struct ToolExecutor {
    /// Configuration
    config: Config,

    /// Tool availability cache
    tool_availability: HashMap<String, bool>,

    /// Output parsers for different tools
    parsers: HashMap<String, Box<dyn OutputParser + Send + Sync>>,
}

/// Trait for parsing tool output into structured issues
pub trait OutputParser {
    /// Parse tool output into issues
    fn parse_output(&self, stdout: &str, stderr: &str, file_path: &Path) -> Vec<Issue>;

    /// Parse fixes from tool output
    fn parse_fixes(&self, stdout: &str, stderr: &str, file_path: &Path) -> Vec<Fix>;
}

/// Ruff output parser
pub struct RuffParser {
    /// Regex for parsing ruff JSON output
    json_regex: Regex,
}

/// MyPy output parser
pub struct MyPyParser {
    /// Regex for parsing mypy output lines
    line_regex: Regex,
}

/// AST-grep output parser
pub struct AstGrepParser {
    /// Regex for parsing ast-grep JSON output
    json_regex: Regex,
}

/// TypeScript compiler output parser
pub struct TypeScriptParser {
    /// Regex for parsing tsc output
    error_regex: Regex,
}

/// Generic text-based output parser
pub struct GenericParser {
    /// Tool name
    tool_name: String,
}

impl ToolExecutor {
    /// Create a new tool executor
    pub fn new(config: Config) -> Result<Self> {
        let mut tool_availability = HashMap::new();
        let mut parsers: HashMap<String, Box<dyn OutputParser + Send + Sync>> = HashMap::new();

        // Check tool availability
        for (tool_name, tool_config) in &config.tools {
            let available = Self::check_tool_availability(&tool_config.executable);
            tool_availability.insert(tool_name.clone(), available);

            if available {
                info!("✅ Tool available: {}", tool_name);
            } else {
                warn!("❌ Tool not available: {} ({})", tool_name, tool_config.executable);
            }
        }

        // Register parsers
        parsers.insert("ruff".to_string(), Box::new(RuffParser::new()?));
        parsers.insert("ruff-format".to_string(), Box::new(RuffParser::new()?));
        parsers.insert("mypy".to_string(), Box::new(MyPyParser::new()?));
        parsers.insert("ast-grep".to_string(), Box::new(AstGrepParser::new()?));
        parsers.insert("tsc".to_string(), Box::new(TypeScriptParser::new()?));

        Ok(Self {
            config,
            tool_availability,
            parsers,
        })
    }

    /// Check if a tool is available
    fn check_tool_availability(executable: &str) -> bool {
        Command::new("which")
            .arg(executable)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    /// Analyze a file with all applicable tools
    pub fn analyze_file(&self, file_path: &Path, auto_fix: bool) -> Result<AnalysisResult> {
        let mut result = AnalysisResult::new(file_path.to_path_buf());

        // Get file extension
        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        // Get applicable tools
        let tools = self.config.get_tools_for_extension(extension);

        if tools.is_empty() {
            debug!("No tools configured for extension: {}", extension);
            result.status = AnalysisStatus::Skipped;
            result.complete();
            return Ok(result);
        }

        info!("🔍 Analyzing {} with {} tools", file_path.display(), tools.len());

        // Run tools in priority order
        for tool_config in tools {
            if let Some(tool_name) = self.config.tools.iter()
                .find(|(_, config)| std::ptr::eq(*config, tool_config))
                .map(|(name, _)| name)
            {
                if !self.tool_availability.get(tool_name).unwrap_or(&false) {
                    debug!("Skipping unavailable tool: {}", tool_name);
                    continue;
                }

                match self.run_tool(tool_name, tool_config, file_path, auto_fix) {
                    Ok(tool_result) => {
                        // Collect issues from tool result
                        result.issues.extend(tool_result.issues.clone());
                        result.fixes_applied.extend(tool_result.fixes_applied.clone());
                        result.tool_results.insert(tool_name.clone(), tool_result);
                    }
                    Err(e) => {
                        error!("Tool {} failed for {}: {}", tool_name, file_path.display(), e);

                        // Create failed tool result
                        let tool_result = ToolResult {
                            tool: tool_name.clone(),
                            exit_code: -1,
                            stdout: String::new(),
                            stderr: e.to_string(),
                            duration: Duration::ZERO,
                            success: false,
                            issues: vec![Issue {
                                tool: tool_name.clone(),
                                severity: IssueSeverity::Error,
                                message: format!("Tool execution failed: {}", e),
                                line: None,
                                column: None,
                                code: None,
                                suggestion: None,
                                auto_fixable: false,
                            }],
                            fixes_applied: Vec::new(),
                        };

                        result.tool_results.insert(tool_name.clone(), tool_result);
                    }
                }
            }
        }

        // Run rufft-claude.sh integration if enabled and issues found
        if auto_fix && !result.issues.is_empty() {
            if let Some(ref script_path) = self.config.integration.rufft_claude_script {
                if script_path.exists() {
                    match self.run_rufft_claude(script_path, file_path, &result).await {
                        Ok(claude_result) => {
                            result.tool_results.insert("rufft-claude".to_string(), claude_result);
                        }
                        Err(e) => {
                            warn!("rufft-claude.sh failed: {}", e);
                        }
                    }
                }
            }
        }

        result.complete();

        info!("✅ Analysis complete: {} issues, {} fixes applied",
            result.issues.len(), result.fixes_applied.len());

        Ok(result)
    }

    /// Run a specific tool
    fn run_tool(&self, tool_name: &str, tool_config: &ToolConfig, file_path: &Path, auto_fix: bool) -> Result<ToolResult> {
        let start_time = Instant::now();

        debug!("🔧 Running tool: {} on {}", tool_name, file_path.display());

        let mut cmd = Command::new(&tool_config.executable);

        // Set working directory
        let working_dir = if let Some(ref wd) = tool_config.working_dir {
            self.config.integration.dashboard_status_file
                .parent()
                .unwrap()
                .join(wd)
        } else {
            file_path.parent().unwrap_or(Path::new(".")).to_path_buf()
        };

        cmd.current_dir(&working_dir);

        // Configure command based on tool
        self.configure_tool_command(&mut cmd, tool_name, tool_config, file_path, auto_fix)?;

        // Set environment variables
        for (key, value) in &tool_config.env {
            cmd.env(key, value);
        }

        // Set stdio
        cmd.stdout(Stdio::piped())
           .stderr(Stdio::piped());

        // Execute with timeout
        let output = std::thread::spawn(move || {
            cmd.output()
        }).join()
        .map_err(|_| anyhow::anyhow!("Tool execution thread panicked"))?
        .with_context(|| format!("Failed to execute tool: {}", tool_config.executable))?;

        let duration = start_time.elapsed();
        let exit_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let success = output.status.success();

        trace!("Tool {} completed in {:?} with exit code {}", tool_name, duration, exit_code);

        // Parse output
        let (issues, fixes_applied) = if let Some(parser) = self.parsers.get(tool_name) {
            let issues = parser.parse_output(&stdout, &stderr, file_path);
            let fixes = parser.parse_fixes(&stdout, &stderr, file_path);
            (issues, fixes)
        } else {
            // Use generic parser
            let parser = GenericParser { tool_name: tool_name.to_string() };
            let issues = parser.parse_output(&stdout, &stderr, file_path);
            let fixes = parser.parse_fixes(&stdout, &stderr, file_path);
            (issues, fixes)
        };

        Ok(ToolResult {
            tool: tool_name.to_string(),
            exit_code,
            stdout,
            stderr,
            duration,
            success,
            issues,
            fixes_applied,
        })
    }

    /// Configure command arguments for specific tools
    fn configure_tool_command(&self, cmd: &mut Command, tool_name: &str, tool_config: &ToolConfig, file_path: &Path, auto_fix: bool) -> Result<()> {
        match tool_name {
            "ruff" => {
                if tool_config.args.is_empty() {
                    cmd.args(&["check", "--output-format", "json"]);
                    if auto_fix {
                        cmd.args(&["--fix", "--unsafe-fixes"]);
                    }
                } else {
                    cmd.args(&tool_config.args);
                    if auto_fix && !tool_config.args.contains(&"--fix".to_string()) {
                        cmd.args(&["--fix", "--unsafe-fixes"]);
                    }
                }
                cmd.arg(file_path);
            }

            "ruff-format" => {
                if tool_config.args.is_empty() {
                    cmd.arg("format");
                    if auto_fix {
                        cmd.arg("--diff");
                    } else {
                        cmd.arg("--check");
                    }
                } else {
                    cmd.args(&tool_config.args);
                }
                cmd.arg(file_path);
            }

            "mypy" => {
                if tool_config.args.is_empty() {
                    cmd.args(&["--show-error-codes", "--no-error-summary"]);
                } else {
                    cmd.args(&tool_config.args);
                }
                cmd.arg(file_path);
            }

            "ast-grep" => {
                if tool_config.args.is_empty() {
                    cmd.args(&["scan", "--json"]);
                    if auto_fix {
                        cmd.arg("--update-all");
                    }
                } else {
                    cmd.args(&tool_config.args);
                }
                cmd.arg(file_path);
            }

            "tsc" => {
                if tool_config.args.is_empty() {
                    cmd.args(&["--noEmit", "--pretty", "false"]);
                } else {
                    cmd.args(&tool_config.args);
                }
                // TypeScript compiler works on project, not individual files
            }

            "clippy" => {
                // For clippy, we need to run cargo clippy
                if tool_config.args.is_empty() {
                    cmd.args(&["clippy", "--message-format", "json"]);
                    if auto_fix {
                        cmd.args(&["--fix", "--allow-dirty", "--allow-staged"]);
                    }
                } else {
                    cmd.args(&tool_config.args);
                }
            }

            "rustfmt" => {
                if tool_config.args.is_empty() {
                    if auto_fix {
                        // Format in place
                    } else {
                        cmd.arg("--check");
                    }
                } else {
                    cmd.args(&tool_config.args);
                }
                cmd.arg(file_path);
            }

            _ => {
                // Generic tool configuration
                cmd.args(&tool_config.args);
                cmd.arg(file_path);
            }
        }

        Ok(())
    }

    /// Run rufft-claude.sh integration
    async fn run_rufft_claude(&self, script_path: &Path, file_path: &Path, analysis_result: &AnalysisResult) -> Result<ToolResult> {
        let start_time = Instant::now();

        info!("🤖 Running rufft-claude.sh for intelligent fixes");

        // Prepare issue summary for rufft-claude
        let issue_summary = self.format_issues_for_claude(&analysis_result.issues);

        let mut cmd = AsyncCommand::new("bash");
        cmd.arg(script_path)
           .arg("fix")
           .arg(file_path)
           .arg(&issue_summary)
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        let output = cmd.output().await
            .with_context(|| format!("Failed to execute rufft-claude.sh: {}", script_path.display()))?;

        let duration = start_time.elapsed();
        let exit_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let success = output.status.success();

        // Parse Claude's response for applied fixes
        let fixes_applied = self.parse_claude_fixes(&stdout, file_path);

        Ok(ToolResult {
            tool: "rufft-claude".to_string(),
            exit_code,
            stdout,
            stderr,
            duration,
            success,
            issues: Vec::new(), // Claude doesn't report new issues
            fixes_applied,
        })
    }

    /// Format issues for Claude analysis
    fn format_issues_for_claude(&self, issues: &[Issue]) -> String {
        let mut summary = String::new();

        for issue in issues {
            summary.push_str(&format!(
                "{}: {} ({}:{}): {}\n",
                issue.severity,
                issue.tool,
                issue.line.unwrap_or(0),
                issue.column.unwrap_or(0),
                issue.message
            ));
        }

        summary
    }

    /// Parse Claude's fix output
    fn parse_claude_fixes(&self, output: &str, file_path: &Path) -> Vec<Fix> {
        let mut fixes = Vec::new();

        // Look for fix indicators in Claude's output
        for line in output.lines() {
            if line.contains("Fixed:") || line.contains("Applied:") {
                fixes.push(Fix {
                    tool: "rufft-claude".to_string(),
                    description: line.trim().to_string(),
                    line: None,
                    column: None,
                    old_text: None,
                    new_text: None,
                    fix_type: FixType::Other,
                });
            }
        }

        fixes
    }

    /// Get tool availability status
    pub fn get_tool_status(&self) -> HashMap<String, bool> {
        self.tool_availability.clone()
    }
}

// Parser implementations
impl RuffParser {
    pub fn new() -> Result<Self> {
        Ok(Self {
            json_regex: Regex::new(r"\{.*\}")?,
        })
    }
}

impl OutputParser for RuffParser {
    fn parse_output(&self, stdout: &str, _stderr: &str, file_path: &Path) -> Vec<Issue> {
        let mut issues = Vec::new();

        // Try parsing as JSON first
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(stdout) {
            if let Some(array) = json_value.as_array() {
                for item in array {
                    if let Some(issue) = self.parse_ruff_json_issue(item, file_path) {
                        issues.push(issue);
                    }
                }
            }
        } else {
            // Fallback to text parsing
            for line in stdout.lines() {
                if let Some(issue) = self.parse_ruff_text_line(line, file_path) {
                    issues.push(issue);
                }
            }
        }

        issues
    }

    fn parse_fixes(&self, stdout: &str, _stderr: &str, _file_path: &Path) -> Vec<Fix> {
        let mut fixes = Vec::new();

        // Look for fix indicators
        for line in stdout.lines() {
            if line.contains("fixed") || line.contains("Fixed") {
                fixes.push(Fix {
                    tool: "ruff".to_string(),
                    description: line.trim().to_string(),
                    line: None,
                    column: None,
                    old_text: None,
                    new_text: None,
                    fix_type: FixType::Lint,
                });
            }
        }

        fixes
    }
}

impl RuffParser {
    fn parse_ruff_json_issue(&self, json: &serde_json::Value, _file_path: &Path) -> Option<Issue> {
        Some(Issue {
            tool: "ruff".to_string(),
            severity: match json.get("severity")?.as_str()? {
                "error" => IssueSeverity::Error,
                "warning" => IssueSeverity::Warning,
                _ => IssueSeverity::Warning,
            },
            message: json.get("message")?.as_str()?.to_string(),
            line: json.get("location")?.get("row")?.as_u64().map(|n| n as u32),
            column: json.get("location")?.get("column")?.as_u64().map(|n| n as u32),
            code: json.get("code")?.as_str().map(|s| s.to_string()),
            suggestion: json.get("fix")?.get("message")?.as_str().map(|s| s.to_string()),
            auto_fixable: json.get("fix").is_some(),
        })
    }

    fn parse_ruff_text_line(&self, line: &str, _file_path: &Path) -> Option<Issue> {
        // Basic text parsing for ruff output
        if line.contains("error") || line.contains("warning") {
            Some(Issue {
                tool: "ruff".to_string(),
                severity: if line.contains("error") { IssueSeverity::Error } else { IssueSeverity::Warning },
                message: line.trim().to_string(),
                line: None,
                column: None,
                code: None,
                suggestion: None,
                auto_fixable: false,
            })
        } else {
            None
        }
    }
}

impl MyPyParser {
    pub fn new() -> Result<Self> {
        Ok(Self {
            line_regex: Regex::new(r"^(.+):(\d+):(\d+):\s+(error|warning|note):\s+(.+?)(?:\s+\[([^\]]+)\])?$")?,
        })
    }
}

impl OutputParser for MyPyParser {
    fn parse_output(&self, stdout: &str, _stderr: &str, _file_path: &Path) -> Vec<Issue> {
        let mut issues = Vec::new();

        for line in stdout.lines() {
            if let Some(captures) = self.line_regex.captures(line) {
                let severity = match captures.get(4).map(|m| m.as_str()) {
                    Some("error") => IssueSeverity::Error,
                    Some("warning") => IssueSeverity::Warning,
                    Some("note") => IssueSeverity::Info,
                    _ => IssueSeverity::Warning,
                };

                let issue = Issue {
                    tool: "mypy".to_string(),
                    severity,
                    message: captures.get(5).map(|m| m.as_str().to_string()).unwrap_or_default(),
                    line: captures.get(2).and_then(|m| m.as_str().parse().ok()),
                    column: captures.get(3).and_then(|m| m.as_str().parse().ok()),
                    code: captures.get(6).map(|m| m.as_str().to_string()),
                    suggestion: None,
                    auto_fixable: false,
                };

                issues.push(issue);
            }
        }

        issues
    }

    fn parse_fixes(&self, _stdout: &str, _stderr: &str, _file_path: &Path) -> Vec<Fix> {
        Vec::new() // MyPy doesn't apply fixes
    }
}

impl AstGrepParser {
    pub fn new() -> Result<Self> {
        Ok(Self {
            json_regex: Regex::new(r"\{.*\}")?,
        })
    }
}

impl OutputParser for AstGrepParser {
    fn parse_output(&self, stdout: &str, _stderr: &str, file_path: &Path) -> Vec<Issue> {
        let mut issues = Vec::new();

        // Parse JSON output from ast-grep
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(stdout) {
            if let Some(array) = json_value.as_array() {
                for item in array {
                    if let Some(issue) = self.parse_ast_grep_json(item, file_path) {
                        issues.push(issue);
                    }
                }
            }
        }

        issues
    }

    fn parse_fixes(&self, stdout: &str, _stderr: &str, _file_path: &Path) -> Vec<Fix> {
        let mut fixes = Vec::new();

        if stdout.contains("updated") || stdout.contains("fixed") {
            fixes.push(Fix {
                tool: "ast-grep".to_string(),
                description: "Applied AST-grep structural fixes".to_string(),
                line: None,
                column: None,
                old_text: None,
                new_text: None,
                fix_type: FixType::Lint,
            });
        }

        fixes
    }
}

impl AstGrepParser {
    fn parse_ast_grep_json(&self, json: &serde_json::Value, _file_path: &Path) -> Option<Issue> {
        Some(Issue {
            tool: "ast-grep".to_string(),
            severity: IssueSeverity::Warning,
            message: json.get("message")?.as_str()?.to_string(),
            line: json.get("range")?.get("start")?.get("line")?.as_u64().map(|n| n as u32),
            column: json.get("range")?.get("start")?.get("column")?.as_u64().map(|n| n as u32),
            code: json.get("ruleId")?.as_str().map(|s| s.to_string()),
            suggestion: json.get("fix")?.as_str().map(|s| s.to_string()),
            auto_fixable: json.get("fix").is_some(),
        })
    }
}

impl TypeScriptParser {
    pub fn new() -> Result<Self> {
        Ok(Self {
            error_regex: Regex::new(r"^(.+)\((\d+),(\d+)\):\s+(error|warning)\s+TS(\d+):\s+(.+)$")?,
        })
    }
}

impl OutputParser for TypeScriptParser {
    fn parse_output(&self, stdout: &str, stderr: &str, _file_path: &Path) -> Vec<Issue> {
        let mut issues = Vec::new();

        let output = if !stderr.is_empty() { stderr } else { stdout };

        for line in output.lines() {
            if let Some(captures) = self.error_regex.captures(line) {
                let severity = match captures.get(4).map(|m| m.as_str()) {
                    Some("error") => IssueSeverity::Error,
                    Some("warning") => IssueSeverity::Warning,
                    _ => IssueSeverity::Error,
                };

                let issue = Issue {
                    tool: "tsc".to_string(),
                    severity,
                    message: captures.get(6).map(|m| m.as_str().to_string()).unwrap_or_default(),
                    line: captures.get(2).and_then(|m| m.as_str().parse().ok()),
                    column: captures.get(3).and_then(|m| m.as_str().parse().ok()),
                    code: captures.get(5).map(|m| format!("TS{}", m.as_str())),
                    suggestion: None,
                    auto_fixable: false,
                };

                issues.push(issue);
            }
        }

        issues
    }

    fn parse_fixes(&self, _stdout: &str, _stderr: &str, _file_path: &Path) -> Vec<Fix> {
        Vec::new() // TypeScript compiler doesn't apply fixes
    }
}

impl OutputParser for GenericParser {
    fn parse_output(&self, stdout: &str, stderr: &str, _file_path: &Path) -> Vec<Issue> {
        let mut issues = Vec::new();

        // Simple generic parsing - look for error/warning keywords
        let output = if !stderr.is_empty() { stderr } else { stdout };

        for line in output.lines() {
            let lower_line = line.to_lowercase();
            if lower_line.contains("error") || lower_line.contains("warning") || lower_line.contains("fail") {
                let severity = if lower_line.contains("error") || lower_line.contains("fail") {
                    IssueSeverity::Error
                } else {
                    IssueSeverity::Warning
                };

                issues.push(Issue {
                    tool: self.tool_name.clone(),
                    severity,
                    message: line.trim().to_string(),
                    line: None,
                    column: None,
                    code: None,
                    suggestion: None,
                    auto_fixable: false,
                });
            }
        }

        issues
    }

    fn parse_fixes(&self, stdout: &str, _stderr: &str, _file_path: &Path) -> Vec<Fix> {
        let mut fixes = Vec::new();

        if stdout.contains("fixed") || stdout.contains("corrected") || stdout.contains("updated") {
            fixes.push(Fix {
                tool: self.tool_name.clone(),
                description: format!("{} applied fixes", self.tool_name),
                line: None,
                column: None,
                old_text: None,
                new_text: None,
                fix_type: FixType::Other,
            });
        }

        fixes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ruff_parser() {
        let parser = RuffParser::new().unwrap();
        let json_output = r#"[{"code": "E501", "message": "line too long", "location": {"row": 10, "column": 5}}]"#;

        let issues = parser.parse_output(json_output, "", Path::new("test.py"));
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].code, Some("E501".to_string()));
        assert_eq!(issues[0].line, Some(10));
    }

    #[test]
    fn test_mypy_parser() {
        let parser = MyPyParser::new().unwrap();
        let output = "test.py:10:5: error: Incompatible return value type  [return-value]";

        let issues = parser.parse_output(output, "", Path::new("test.py"));
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, IssueSeverity::Error);
        assert_eq!(issues[0].line, Some(10));
        assert_eq!(issues[0].column, Some(5));
        assert_eq!(issues[0].code, Some("return-value".to_string()));
    }

    #[test]
    fn test_tool_availability() {
        // This test assumes 'which' is available
        assert!(ToolExecutor::check_tool_availability("which"));
        assert!(!ToolExecutor::check_tool_availability("non_existent_tool_xyz"));
    }
}
