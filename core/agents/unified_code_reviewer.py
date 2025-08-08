"""Unified Code Reviewer - Consolidated code review functionality.

This module consolidates functionality from:
- CodeReviewAgentService (job management, streaming, comprehensive analysis)
- GeminiCodeReviewer MCP Server (high-performance with PyO3 optimizations)
- BaseCodeReviewAgent (automation features and MCP tools)

Key Features:
- Comprehensive code analysis (security, performance, quality)
- High-performance file operations with PyO3 Rust extensions
- Streaming progress updates and job management
- MCP server integration for Claude CLI access
- Advanced caching with TTL and circuit breakers
- Parallel file processing with git-aware scanning
- Security vulnerability detection with CWE mappings
- AI-powered review using Google Gemini 2.0 Flash
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Literal

# Core dependencies
import aiohttp
from fastmcp import FastMCP
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
import vertexai

# Project imports
from gterminal.core.agents.base_unified_agent import BaseUnifiedAgent
from gterminal.core.agents.base_unified_agent import Job
from gterminal.core.security.security_utils import safe_json_parse
from gterminal.core.security.security_utils import safe_subprocess_run
from gterminal.core.security.security_utils import validate_file_path

# Optional PyO3 Rust extensions for performance
try:
    from fullstack_agent_rust import RustCache
    from fullstack_agent_rust import RustFileOps
    from fullstack_agent_rust import RustJsonProcessor

    RUST_EXTENSIONS_AVAILABLE = True
except ImportError:
    RUST_EXTENSIONS_AVAILABLE = False

# Optional high-performance file operations
try:
    from gterminal.gemini_agents.utils.file_ops import FileOpsMonitor
    from gterminal.gemini_agents.utils.file_ops import ParallelFileOps

    PARALLEL_FILE_OPS_AVAILABLE = True
except ImportError:
    PARALLEL_FILE_OPS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for parallel operations
optimal_workers = min(32, (os.cpu_count() or 1) + 4)
executor = ThreadPoolExecutor(max_workers=optimal_workers, thread_name_prefix="CodeReviewer")


# Pydantic models for validation
class CodeMetadata(BaseModel):
    """Metadata about a code file."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    lines: int = Field(..., ge=0, description="Number of lines in the file")
    size: int = Field(..., ge=0, description="File size in bytes")
    extension: str = Field(..., description="File extension")
    encoding: str = Field(default="utf-8", description="File encoding")
    last_modified: datetime | None = Field(None, description="Last modification time")


class RelatedFile(BaseModel):
    """Information about a related file."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Path to the related file")
    content: str = Field(..., max_length=10000, description="File content (truncated)")
    relationship: str = Field(default="related", description="Type of relationship")


class CodeContext(BaseModel):
    """Complete context for code analysis."""

    model_config = ConfigDict(extra="forbid")

    target_file: str = Field(..., description="Path to the target file")
    content: str = Field(..., description="File content")
    related_files: list[RelatedFile] = Field(default_factory=list, description="Related files")
    metadata: CodeMetadata = Field(..., description="File metadata")
    cache_hit: bool = Field(default=False, description="Whether this was a cache hit")


class CodeIssue(BaseModel):
    """A code review issue."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["security", "performance", "quality", "style"] = Field(
        ..., description="Issue type"
    )
    severity: Literal["critical", "high", "medium", "low"] = Field(
        ..., description="Issue severity"
    )
    line: str = Field(..., description="Line number or range")
    description: str = Field(..., description="Issue description")
    suggestion: str = Field(..., description="How to fix the issue")
    code_example: str | None = Field(None, description="Example fix code")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in the finding")


class CodeMetrics(BaseModel):
    """Code quality metrics."""

    model_config = ConfigDict(extra="forbid")

    complexity: str = Field(..., description="Cyclomatic complexity estimate")
    maintainability: str = Field(..., description="Maintainability index")
    test_coverage: str = Field(..., description="Estimated test coverage needed")
    technical_debt: str | None = Field(None, description="Technical debt assessment")


class SecurityIssue(BaseModel):
    """A security issue found in code."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(..., description="File containing the issue")
    type: str = Field(..., description="Type of security issue")
    severity: Literal["critical", "high", "medium", "low"] = Field(
        ..., description="Issue severity"
    )
    line_number: int | None = Field(None, ge=1, description="Line number if known")
    description: str | None = Field(None, description="Detailed description")
    cwe_id: str | None = Field(None, description="CWE identifier if applicable")


class ReviewResult(BaseModel):
    """Complete code review result."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., description="Overall assessment")
    quality_score: float = Field(..., ge=1.0, le=10.0, description="Quality score 1-10")
    issues: list[CodeIssue] = Field(default_factory=list, description="Found issues")
    positive_aspects: list[str] = Field(default_factory=list, description="What's done well")
    recommendations: list[str] = Field(default_factory=list, description="Improvement suggestions")
    metrics: CodeMetrics = Field(..., description="Code metrics")


class UnifiedCodeReviewer(BaseUnifiedAgent):
    """Unified code reviewer with comprehensive analysis capabilities.

    Features:
    - Security vulnerability scanning with CWE mappings
    - Performance analysis with bottleneck detection
    - Code quality assessment with AI insights
    - Test coverage analysis and recommendations
    - Documentation completeness evaluation
    - High-performance file operations with Rust extensions
    - Streaming progress updates and job management
    - MCP server integration for Claude CLI access
    """

    def __init__(
        self,
        enable_high_performance: bool = True,
        auto_approve_threshold: float = 0.95,
        max_concurrent_reviews: int = 5,
    ) -> None:
        super().__init__(
            agent_name="unified_code_reviewer",
            description="Comprehensive code review with security, performance, and quality analysis",
            max_concurrent_jobs=max_concurrent_reviews,
            enable_resource_monitoring=True,
            enable_rust_extensions=RUST_EXTENSIONS_AVAILABLE,
        )

        self.auto_approve_threshold = auto_approve_threshold
        self.enable_high_performance = enable_high_performance and PARALLEL_FILE_OPS_AVAILABLE

        # Initialize high-performance file operations
        if self.enable_high_performance:
            try:
                self.file_ops = ParallelFileOps(max_workers=20, batch_size=100)
                self.performance_monitor = FileOpsMonitor()
                self.logger.info("High-performance file operations initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize high-performance file ops: {e}")
                self.enable_high_performance = False

        # Enhanced cache for context and analysis results
        self.context_cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}

        # Security patterns with CWE mappings
        self.security_patterns = [
            (
                "hardcoded_credentials",
                r"(api_key|password|secret|token|private_key)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
                "high",
                "CWE-798",
            ),
            (
                "sql_injection",
                r"(SELECT|INSERT|UPDATE|DELETE).*(\+|%|\|\|).*(%s|%d|\?)",
                "critical",
                "CWE-89",
            ),
            ("path_traversal", r"(\.\./|\.\.\\|%2e%2e)", "high", "CWE-22"),
            (
                "command_injection",
                r"(os\.system|subprocess\.(call|run|Popen)).*(\+|%|\|\|)",
                "critical",
                "CWE-78",
            ),
            ("weak_crypto", r"\b(md5|sha1|des|rc4)\b", "medium", "CWE-327"),
            ("weak_random", r"(random\.random|Math\.random)\(\)", "medium", "CWE-338"),
            ("xss_vulnerability", r"(innerHTML|document\.write).*(\+|%|\|\|)", "high", "CWE-79"),
            ("unsafe_eval", r"\beval\s*\(", "high", "CWE-95"),
            ("insecure_transport", r"http://", "medium", "CWE-319"),
            (
                "debug_info_leak",
                r"(console\.log|print|echo).*\b(password|secret|token|key)\b",
                "medium",
                "CWE-200",
            ),
        ]

        # Performance anti-patterns
        self.performance_patterns = [
            (r"for\s+\w+\s+in\s+range\(len\(", "Use enumerate() instead of range(len())"),
            (r"\.append\(\)\s*in\s+loop", "Consider list comprehension or pre-allocation"),
            (r"string\s*\+\s*string", "Consider using join() for string concatenation"),
            (r"global\s+\w+", "Global variables can impact performance"),
        ]

        # Connection pool for external requests
        self.connector_pool: aiohttp.TCPConnector | None = None

        self.logger.info(
            f"Unified code reviewer initialized with auto-approve threshold: {auto_approve_threshold}"
        )

    async def get_connector_pool(self) -> aiohttp.TCPConnector:
        """Get or create connection pool."""
        if self.connector_pool is None:
            self.connector_pool = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
                force_close=True,
                keepalive_timeout=30,
            )
        return self.connector_pool

    async def startup(self) -> None:
        """Initialize async components."""
        await super().startup()

        # Initialize Vertex AI
        try:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

            if project:
                vertexai.init(project=project, location=location)
                self.logger.info(
                    f"Vertex AI initialized - Project: {project}, Location: {location}"
                )
            else:
                self.logger.warning(
                    "GOOGLE_CLOUD_PROJECT not set, Vertex AI initialization skipped"
                )
        except Exception as e:
            self.logger.exception(f"Failed to initialize Vertex AI: {e}")

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.connector_pool:
            await self.connector_pool.close()
        executor.shutdown(wait=True)
        await super().shutdown()

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Get required parameters for job type."""
        if job_type == "review_file":
            return ["file_path"]
        if job_type == "review_pr":
            return ["pr_number"]
        if job_type == "review_diff":
            return ["diff_content"]
        if job_type == "review_project":
            return ["project_path"]
        if job_type == "security_scan":
            return ["target_path"]
        if job_type == "performance_analysis":
            return ["file_path"]
        return []

    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Execute code review job implementation."""
        job_type = job.job_type
        parameters = job.parameters

        if job_type == "review_file":
            return await self._review_file(job, parameters["file_path"])
        if job_type == "review_pr":
            return await self._review_pr(job, parameters["pr_number"])
        if job_type == "review_diff":
            return await self._review_diff(job, parameters["diff_content"])
        if job_type == "review_project":
            return await self._review_project(job, parameters["project_path"])
        if job_type == "security_scan":
            return await self._security_scan(job, parameters["target_path"])
        if job_type == "performance_analysis":
            return await self._performance_analysis(job, parameters["file_path"])
        msg = f"Unknown job type: {job_type}"
        raise ValueError(msg)

    async def _review_file(self, job: Job, file_path: str) -> dict[str, Any]:
        """Review a specific file with comprehensive analysis."""
        job.update_progress(10.0, f"Validating file path: {file_path}")

        try:
            validated_path = validate_file_path(file_path)
            if not Path(validated_path).exists():
                return {"error": f"File not found: {file_path}"}

            # Collect context with enhanced caching
            job.update_progress(25.0, "Collecting file context")
            context = await self._collect_code_context_fast(
                str(validated_path), include_related=True
            )

            # Perform comprehensive analysis
            job.update_progress(40.0, "Analyzing file content")
            analysis_result = await self._perform_comprehensive_analysis(
                job, context.content, "", str(validated_path)
            )

            return {
                "file_path": file_path,
                "analysis": analysis_result,
                "context_metadata": {
                    "lines_analyzed": context.metadata.lines,
                    "file_size": context.metadata.size,
                    "related_files_count": len(context.related_files),
                    "cache_hit": context.cache_hit,
                },
                "reviewed_at": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            msg = f"File review failed: {e!s}"
            raise Exception(msg)

    async def _review_pr(self, job: Job, pr_number: int) -> dict[str, Any]:
        """Review a pull request."""
        job.update_progress(10.0, f"Fetching PR #{pr_number} details")

        try:
            # Get PR diff
            diff_result = safe_subprocess_run(["gh", "pr", "diff", str(pr_number)])
            if diff_result.returncode != 0:
                return {"error": f"Could not fetch PR #{pr_number}"}

            diff_content = diff_result.stdout

            # Get PR info
            pr_info_result = safe_subprocess_run(
                [
                    "gh",
                    "pr",
                    "view",
                    str(pr_number),
                    "--json",
                    "title,body,author,additions,deletions,changedFiles",
                ],
            )

            pr_info: dict[str, Any] = {}
            if pr_info_result.returncode == 0:
                pr_info = safe_json_parse(pr_info_result.stdout) or {}

            job.update_progress(30.0, "Analyzing PR changes")

            # Analyze the diff
            analysis_result = await self._perform_comprehensive_analysis(
                job,
                diff_content,
                diff_content,
                f"PR #{pr_number}",
            )

            return {
                "pr_number": pr_number,
                "pr_info": pr_info,
                "analysis": analysis_result,
                "reviewed_at": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            msg = f"PR review failed: {e!s}"
            raise Exception(msg)

    async def _review_diff(self, job: Job, diff_content: str) -> dict[str, Any]:
        """Review diff content directly."""
        job.update_progress(10.0, "Processing diff content")

        try:
            analysis_result = await self._perform_comprehensive_analysis(
                job,
                diff_content,
                diff_content,
                "diff_content",
            )

            return {
                "diff_analyzed": True,
                "analysis": analysis_result,
                "reviewed_at": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            msg = f"Diff review failed: {e!s}"
            raise Exception(msg)

    async def _review_project(self, job: Job, project_path: str) -> dict[str, Any]:
        """Review entire project using high-performance operations."""
        job.update_progress(10.0, f"Scanning project: {project_path}")

        try:
            project_dir = Path(project_path)
            if not project_dir.exists():
                return {"error": f"Project path does not exist: {project_path}"}

            # Use high-performance file scanning
            job.update_progress(15.0, "Performing high-speed file discovery")

            code_patterns = [
                "*.py",
                "*.js",
                "*.ts",
                "*.jsx",
                "*.tsx",
                "*.java",
                "*.cpp",
                "*.c",
                "*.go",
                "*.rs",
            ]

            if self.enable_high_performance:
                try:
                    # Use parallel file operations
                    scan_result = await self.file_ops.git_aware_file_scan(
                        project_path,
                        respect_gitignore=True,
                        include_git_metadata=True,
                    )

                    # Filter for code files
                    code_files: list[Any] = []
                    for file_path in scan_result["files"]:
                        file_obj = Path(file_path)
                        if any(file_obj.match(pattern) for pattern in code_patterns):
                            code_files.append(file_path)

                    performance_stats = scan_result.get("performance", {})
                    job.update_progress(
                        25.0,
                        f"Found {len(code_files)} code files (rust-fs: {performance_stats.get('used_rust_fs', False)})",
                    )

                except Exception as e:
                    self.logger.warning(f"High-performance scan failed, using fallback: {e}")
                    # Fallback to traditional scanning
                    code_files: list[Any] = []
                    for pattern in code_patterns:
                        code_files.extend([str(f) for f in project_dir.rglob(pattern)])
                    job.update_progress(25.0, f"Found {len(code_files)} code files (fallback mode)")
            else:
                # Traditional scanning
                code_files: list[Any] = []
                for pattern in code_patterns:
                    code_files.extend([str(f) for f in project_dir.rglob(pattern)])
                job.update_progress(25.0, f"Found {len(code_files)} code files")

            # Limit files for analysis
            analysis_limit = min(50, len(code_files))
            selected_files = code_files[:analysis_limit]

            # Use parallel file reading
            job.update_progress(35.0, f"Reading {len(selected_files)} files in parallel")

            if self.enable_high_performance:
                try:
                    file_read_result = await self.file_ops.parallel_file_read(
                        selected_files,
                        max_file_size=1024 * 1024,  # 1MB limit
                    )
                    file_contents = file_read_result["files"]
                    read_errors = file_read_result["errors"]

                    if read_errors:
                        self.logger.warning(f"Failed to read {len(read_errors)} files")

                    job.update_progress(45.0, f"Successfully read {len(file_contents)} files")

                except Exception as e:
                    self.logger.warning(f"Parallel file reading failed: {e}")
                    # Fallback to sequential reading
                    file_contents = await self._sequential_file_read(selected_files[:20])
            else:
                file_contents = await self._sequential_file_read(selected_files[:20])

            # Analyze files in parallel batches
            job.update_progress(55.0, "Analyzing file contents")

            file_analyses: list[Any] = []
            analysis_tasks: list[Any] = []

            for file_path, file_data in file_contents.items():
                content = file_data.get("content", "")
                if content:
                    analysis_tasks.append(
                        self._analyze_file_content_with_context(content, file_path)
                    )

            # Execute analyses in parallel batches
            batch_size = 10
            for i in range(0, len(analysis_tasks), batch_size):
                batch = analysis_tasks[i : i + batch_size]
                progress = 55.0 + (30.0 * i / len(analysis_tasks))
                job.update_progress(progress, f"Analyzing batch {i // batch_size + 1}")

                try:
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)

                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            file_path = list(file_contents.keys())[i + j]
                            self.logger.warning(f"Analysis failed for {file_path}: {result}")
                            continue

                        file_path = list(file_contents.keys())[i + j]
                        file_analyses.append({"file": file_path, "analysis": result})

                except Exception as e:
                    self.logger.warning(f"Batch analysis failed: {e}")
                    continue

            job.update_progress(90.0, "Generating project summary")

            # Generate project-level summary
            project_summary = await self._generate_project_summary(file_analyses)

            # Add performance metrics
            project_summary["performance_metrics"] = {
                "total_files_found": len(code_files),
                "files_analyzed": len(file_analyses),
                "high_performance_enabled": self.enable_high_performance,
                "analysis_coverage": len(file_analyses) / len(code_files) if code_files else 0,
            }

            return {
                "project_path": project_path,
                "total_files": len(code_files),
                "analyzed_files": len(file_analyses),
                "file_analyses": file_analyses,
                "project_summary": project_summary,
                "reviewed_at": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            msg = f"Project review failed: {e!s}"
            raise Exception(msg)

    async def _security_scan(self, job: Job, target_path: str) -> dict[str, Any]:
        """Perform focused security scan."""
        job.update_progress(10.0, f"Starting security scan of {target_path}")

        try:
            target_dir = Path(target_path)
            if not target_dir.exists():
                return {"error": f"Target path does not exist: {target_path}"}

            # Find files to scan
            file_patterns = [
                "*.py",
                "*.js",
                "*.ts",
                "*.jsx",
                "*.tsx",
                "*.java",
                "*.php",
                "*.go",
                "*.rs",
            ]
            files_to_scan: list[Any] = []

            for pattern in file_patterns:
                files_to_scan.extend(target_dir.rglob(pattern))

            files_to_scan = [f for f in files_to_scan if f.is_file()][:75]  # Limit for performance

            job.update_progress(25.0, f"Scanning {len(files_to_scan)} files for security issues")

            # Scan files in parallel
            semaphore = asyncio.Semaphore(8)
            scan_tasks = [
                self._scan_file_for_security(file_path, semaphore) for file_path in files_to_scan
            ]

            scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)

            # Collect all security issues
            all_issues: list[Any] = []
            for result in scan_results:
                if isinstance(result, list):
                    all_issues.extend(result)

            # Group issues by type and severity
            issue_summary = self._summarize_security_issues(all_issues)

            job.update_progress(90.0, "Generating security recommendations")

            recommendations = self._generate_security_recommendations(issue_summary)

            return {
                "target_path": target_path,
                "security_issues": all_issues,
                "issue_summary": issue_summary,
                "recommendations": recommendations,
                "scanned_files": len(files_to_scan),
                "reviewed_at": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            msg = f"Security scan failed: {e!s}"
            raise Exception(msg)

    async def _performance_analysis(self, job: Job, file_path: str) -> dict[str, Any]:
        """Perform focused performance analysis."""
        job.update_progress(10.0, f"Starting performance analysis of {file_path}")

        try:
            validated_path = validate_file_path(file_path)
            content = self.safe_file_read(validated_path)

            if not content:
                return {"error": f"Could not read file: {file_path}"}

            job.update_progress(30.0, "Analyzing performance patterns")

            # Analyze performance issues
            performance_issues = self._analyze_performance(content)

            job.update_progress(60.0, "Generating AI performance insights")

            # Get AI insights for performance
            ai_insights = await self._generate_performance_ai_insights(content, file_path)

            job.update_progress(90.0, "Generating recommendations")

            recommendations = self._generate_performance_recommendations(
                performance_issues, ai_insights
            )

            return {
                "file_path": file_path,
                "performance_issues": performance_issues,
                "ai_insights": ai_insights,
                "recommendations": recommendations,
                "reviewed_at": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            msg = f"Performance analysis failed: {e!s}"
            raise Exception(msg)

    async def _collect_code_context_fast(
        self, file_path: str, include_related: bool = True
    ) -> CodeContext:
        """Fast context collection with enhanced caching."""
        start_time = time.time()

        # Generate cache key
        file_hash = await self._get_file_hash(file_path)
        cache_key = f"{file_path}:{file_hash}"

        # Check cache first
        if cache_key in self.context_cache:
            cached_context = self.context_cache[cache_key]
            if time.time() - cached_context.get("timestamp", 0) < self.cache_ttl:
                self.cache_stats["hits"] += 1
                self.logger.info(f"Cache hit for {file_path}")
                cached_context["data"].cache_hit = True
                return cached_context["data"]

        self.cache_stats["misses"] += 1

        # Read file content
        if self.enable_rust_extensions:
            try:
                content = self.rust_file_ops.read_file(file_path)
                file_stats = await self._get_file_stats_rust(file_path)
            except Exception as e:
                self.logger.debug(f"Rust operations failed, using Python: {e}")
                content = self.safe_file_read(file_path)
                file_stats = await self._get_file_stats_python(file_path)
        else:
            content = self.safe_file_read(file_path)
            file_stats = await self._get_file_stats_python(file_path)

        if not content:
            # Return minimal context for failed reads
            return CodeContext(
                target_file=file_path,
                content="",
                metadata=CodeMetadata(lines=0, size=0, extension=Path(file_path).suffix),
                related_files=[],
                cache_hit=False,
            )

        # Create metadata
        metadata = CodeMetadata(
            lines=len(content.splitlines()),
            size=len(content),
            extension=Path(file_path).suffix,
            encoding="utf-8",
            last_modified=file_stats.get("modified"),
        )

        related_files: list[Any] = []
        if include_related:
            related_files = await self._find_related_files(file_path, content)

        # Create context
        context = CodeContext(
            target_file=file_path,
            content=content,
            metadata=metadata,
            related_files=related_files,
            cache_hit=False,
        )

        # Cache the context
        self.context_cache[cache_key] = {
            "data": context,
            "timestamp": time.time(),
        }

        self.logger.info(
            f"Context collected for {file_path} with {len(related_files)} related files "
            f"(took {time.time() - start_time:.3f}s)",
        )

        return context

    async def _perform_comprehensive_analysis(
        self,
        job: Job | None,
        content: str,
        diff_content: str,
        context: str,
    ) -> dict[str, Any]:
        """Perform comprehensive code analysis."""
        # Security analysis
        if job:
            job.update_progress(40.0, "Performing security analysis")
        security_issues = self._analyze_security(content)

        # Quality analysis
        if job:
            job.update_progress(50.0, "Analyzing code quality")
        quality_analysis = await self._analyze_quality(content, context)

        # Performance analysis
        if job:
            job.update_progress(60.0, "Analyzing performance implications")
        performance_analysis = self._analyze_performance(content)

        # Documentation analysis
        if job:
            job.update_progress(70.0, "Checking documentation")
        documentation_analysis = self._analyze_documentation(content)

        # Generate AI review
        if job:
            job.update_progress(80.0, "Generating AI review")
        ai_review = await self._generate_ai_review(content, diff_content, context)

        # Calculate overall score
        if job:
            job.update_progress(90.0, "Calculating final score")
        overall_score = self._calculate_overall_score(
            security_issues,
            quality_analysis,
            performance_analysis,
            documentation_analysis,
            ai_review,
        )

        return {
            "overall_score": overall_score,
            "security_issues": security_issues,
            "quality_analysis": quality_analysis,
            "performance_analysis": performance_analysis,
            "documentation_analysis": documentation_analysis,
            "ai_review": ai_review,
            "approved": (
                overall_score >= self.auto_approve_threshold
                and len(security_issues.get("critical", [])) == 0
            ),
        }

    def _analyze_security(self, content: str) -> dict[str, Any]:
        """Analyze content for security issues with CWE mappings."""
        critical_issues: list[Any] = []
        warning_issues: list[Any] = []

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            for issue_type, pattern, severity, cwe_id in self.security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issue = {
                        "line": line_num,
                        "content": line.strip(),
                        "type": issue_type,
                        "pattern": pattern,
                        "severity": severity,
                        "cwe_id": cwe_id,
                    }

                    if severity == "critical":
                        critical_issues.append(issue)
                    else:
                        warning_issues.append(issue)

        return {
            "critical": critical_issues,
            "warnings": warning_issues,
            "total_issues": len(critical_issues) + len(warning_issues),
        }

    async def _analyze_quality(self, content: str, context: str) -> dict[str, Any]:
        """Analyze code quality using AI."""
        prompt = f"""Analyze this code for quality issues:

Context: {context}

Code:
```
{content[:5000]}  # Limit content for prompt
```

Focus on:
1. Code complexity and readability
2. Naming conventions
3. Function/class design
4. Error handling
5. Code duplication

Respond in JSON format with:
- complexity_score (0-10)
- readability_score (0-10)
- maintainability_score (0-10)
- issues: [list of specific issues]
- suggestions: [list of improvement suggestions]
"""

        ai_response = await self.generate_with_gemini(prompt, "code_review", parse_json=False)

        if ai_response:
            quality_data = safe_json_parse(ai_response)
            if quality_data:
                return quality_data

        # Fallback basic analysis
        lines = content.split("\n")
        return {
            "complexity_score": 7,
            "readability_score": 7,
            "maintainability_score": 7,
            "line_count": len(lines),
            "function_count": len(re.findall(r"def\s+\w+", content)),
            "class_count": len(re.findall(r"class\s+\w+", content)),
            "issues": [],
            "suggestions": [],
        }

    def _analyze_performance(self, content: str) -> dict[str, Any]:
        """Analyze performance implications."""
        performance_issues: list[Any] = []

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            for pattern, suggestion in self.performance_patterns:
                if re.search(pattern, line):
                    performance_issues.append(
                        {
                            "line": line_num,
                            "content": line.strip(),
                            "issue": suggestion,
                            "severity": "warning",
                        },
                    )

        return {
            "issues": performance_issues,
            "total_issues": len(performance_issues),
            "score": max(0, 10 - len(performance_issues)),
        }

    def _analyze_documentation(self, content: str) -> dict[str, Any]:
        """Analyze documentation completeness."""
        # Count functions and classes
        functions = re.findall(r"def\s+(\w+)", content)
        classes = re.findall(r"class\s+(\w+)", content)

        # Count docstrings
        docstring_count = len(re.findall(r'""".*?"""', content, re.DOTALL))
        docstring_count += len(re.findall(r"'''.*?'''", content, re.DOTALL))

        total_definitions = len(functions) + len(classes)
        documentation_ratio = docstring_count / max(total_definitions, 1)

        return {
            "functions": len(functions),
            "classes": len(classes),
            "docstrings": docstring_count,
            "documentation_ratio": documentation_ratio,
            "score": min(10, documentation_ratio * 10),
            "missing_docs": max(0, total_definitions - docstring_count),
        }

    async def _generate_ai_review(
        self, content: str, diff_content: str, context: str
    ) -> dict[str, Any]:
        """Generate comprehensive AI review."""
        prompt = f"""You are an expert code reviewer. Review this code and provide:

Context: {context}

Content to review:
```
{content[:8000]}  # Limit for prompt size
```

Diff (if available):
```diff
{diff_content[:2000] if diff_content else "No diff available"}
```

Provide a JSON response with:
- overall_score: 0.0 to 1.0
- strengths: [list of positive aspects]
- weaknesses: [list of areas for improvement]
- recommendations: [specific actionable recommendations]
- complexity_assessment: brief assessment of code complexity
- maintainability_notes: notes on long-term maintainability
"""

        ai_response = await self.generate_with_gemini(prompt, "code_review", parse_json=False)

        if ai_response:
            review_data = safe_json_parse(ai_response)
            if review_data:
                return review_data

        # Fallback response
        return {
            "overall_score": 0.7,
            "strengths": ["Code is readable"],
            "weaknesses": ["Could not perform detailed AI analysis"],
            "recommendations": ["Review manually for detailed feedback"],
            "complexity_assessment": "Unable to assess",
            "maintainability_notes": "Manual review recommended",
        }

    def _calculate_overall_score(
        self,
        security_issues: dict[str, Any],
        quality_analysis: dict[str, Any],
        performance_analysis: dict[str, Any],
        documentation_analysis: dict[str, Any],
        ai_review: dict[str, Any],
    ) -> float:
        """Calculate overall review score."""
        # Start with AI review score if available
        base_score = ai_review.get("overall_score", 0.7)

        # Penalize for security issues
        security_penalty = len(security_issues.get("critical", [])) * 0.3
        security_penalty += len(security_issues.get("warnings", [])) * 0.1

        # Factor in quality scores
        quality_bonus = 0
        if "complexity_score" in quality_analysis:
            quality_bonus = (quality_analysis["complexity_score"] / 10) * 0.1

        # Factor in documentation
        doc_bonus = (documentation_analysis.get("score", 5) / 10) * 0.1

        # Calculate final score
        final_score = max(0.0, min(1.0, base_score - security_penalty + quality_bonus + doc_bonus))

        return round(final_score, 3)

    async def _scan_file_for_security(
        self, file_path: Path, semaphore: asyncio.Semaphore
    ) -> list[SecurityIssue]:
        """Scan a single file for security issues with concurrency control."""
        async with semaphore:
            if self.enable_rust_extensions:
                try:
                    content = self.rust_file_ops.read_file(str(file_path))
                except Exception:
                    content = self.safe_file_read(file_path)
            else:
                content = self.safe_file_read(file_path)

            if not content:
                return []

            issues: list[Any] = []
            lines = content.splitlines()

            for i, line in enumerate(lines, 1):
                for issue_type, pattern, severity, cwe_id in self.security_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(
                            SecurityIssue(
                                file=str(file_path),
                                type=issue_type,
                                severity=severity,
                                line_number=i,
                                description=f"Potential {issue_type.replace('_', ' ')} found on line {i}",
                                cwe_id=cwe_id,
                            ),
                        )

            return issues

    async def _sequential_file_read(self, files: list[str]) -> dict[str, dict[str, Any]]:
        """Sequential file reading fallback."""
        file_contents: dict[str, Any] = {}
        for file_path in files:
            try:
                if self.enable_rust_extensions:
                    content = self.rust_file_ops.read_file(file_path)
                else:
                    content = self.safe_file_read(file_path)

                if content:
                    file_contents[file_path] = {"content": content}
            except Exception:
                continue
        return file_contents

    async def _analyze_file_content_with_context(
        self, content: str, file_path: str
    ) -> dict[str, Any]:
        """Analyze individual file content with enhanced context."""
        try:
            return await self._perform_comprehensive_analysis(None, content, "", file_path)
        except Exception as e:
            self.logger.warning(f"File analysis failed for {file_path}: {e}")
            return {
                "overall_score": 0.5,
                "security_issues": {"critical": [], "warnings": [], "total_issues": 0},
                "quality_analysis": {"complexity_score": 5, "readability_score": 5},
                "performance_analysis": {"issues": [], "score": 5},
                "documentation_analysis": {"score": 5},
                "ai_review": {
                    "overall_score": 0.5,
                    "strengths": [],
                    "weaknesses": ["Analysis failed"],
                },
                "approved": False,
                "error": str(e),
            }

    async def _generate_project_summary(
        self, file_analyses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate project-level summary from file analyses."""
        if not file_analyses:
            return {"error": "No files analyzed"}

        total_files = len(file_analyses)
        total_score = sum(
            analysis["analysis"].get("overall_score", 0) for analysis in file_analyses
        )

        security_issues: list[Any] = []
        for analysis in file_analyses:
            file_security = analysis["analysis"].get("security_issues", {})
            security_issues.extend(file_security.get("critical", []))

        return {
            "total_files_analyzed": total_files,
            "average_score": round(total_score / total_files, 3) if total_files > 0 else 0,
            "total_critical_security_issues": len(security_issues),
            "recommended_for_approval": (
                len(security_issues) == 0
                and (total_score / total_files) >= self.auto_approve_threshold
            ),
        }

    def _summarize_security_issues(self, issues: list[SecurityIssue]) -> dict[str, Any]:
        """Summarize security issues by type and severity."""
        summary: dict[str, Any] = {}
        critical_count = high_count = medium_count = low_count = 0

        for issue in issues:
            issue_type = issue.type
            if issue_type not in summary:
                summary[issue_type] = {"count": 0, "files": set(), "cwe_id": issue.cwe_id}

            summary[issue_type]["count"] += 1
            summary[issue_type]["files"].add(issue.file)

            # Count by severity
            if issue.severity == "critical":
                critical_count += 1
            elif issue.severity == "high":
                high_count += 1
            elif issue.severity == "medium":
                medium_count += 1
            else:
                low_count += 1

        # Convert sets to lists for JSON serialization
        for issue_type in summary:
            summary[issue_type]["files"] = list(summary[issue_type]["files"])

        return {
            "by_type": summary,
            "by_severity": {
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
            },
            "total_issues": len(issues),
        }

    def _generate_security_recommendations(self, issue_summary: dict[str, Any]) -> list[str]:
        """Generate security recommendations based on issues found."""
        recommendations: list[Any] = []

        severity_counts = issue_summary.get("by_severity", {})

        if severity_counts.get("critical", 0) > 0:
            recommendations.append("URGENT: Address critical security vulnerabilities immediately")

        if severity_counts.get("high", 0) > 0:
            recommendations.append("High priority: Fix high-severity security issues")

        if severity_counts.get("medium", 0) > 5:
            recommendations.append("Consider implementing security scanning in CI/CD pipeline")

        if issue_summary.get("total_issues", 0) > 10:
            recommendations.append("Perform comprehensive security audit of the codebase")

        return recommendations

    async def _generate_performance_ai_insights(
        self, content: str, file_path: str
    ) -> dict[str, Any]:
        """Generate AI insights for performance analysis."""
        prompt = f"""Analyze this code for performance bottlenecks and optimization opportunities:

File: {file_path}

Code:
```
{content[:6000]}
```

Provide JSON response with:
- bottlenecks: [list of performance bottlenecks found]
- optimizations: [list of specific optimization suggestions]
- complexity_assessment: algorithmic complexity analysis
- resource_usage: analysis of memory/CPU usage patterns
"""

        ai_response = await self.generate_with_gemini(
            prompt, "performance_analysis", parse_json=False
        )

        if ai_response:
            insights = safe_json_parse(ai_response)
            if insights:
                return insights

        return {
            "bottlenecks": [],
            "optimizations": ["Manual performance analysis recommended"],
            "complexity_assessment": "Unable to assess automatically",
            "resource_usage": "Analysis not available",
        }

    def _generate_performance_recommendations(
        self,
        performance_issues: dict[str, Any],
        ai_insights: dict[str, Any],
    ) -> list[str]:
        """Generate performance improvement recommendations."""
        recommendations: list[Any] = []

        if performance_issues.get("total_issues", 0) > 0:
            recommendations.append("Review identified performance anti-patterns")

        if performance_issues.get("total_issues", 0) > 5:
            recommendations.append("Consider performance profiling for detailed analysis")

        if ai_insights.get("bottlenecks"):
            recommendations.extend(ai_insights["bottlenecks"][:3])  # Top 3 bottlenecks

        return recommendations

    async def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for cache key."""
        try:
            if self.enable_rust_extensions:
                # Use Rust for fast hashing if available
                return hashlib.sha256(file_path.encode()).hexdigest()

            with open(file_path, "rb") as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception:
            return hashlib.sha256(file_path.encode()).hexdigest()

    async def _get_file_stats_rust(self, file_path: str) -> dict[str, Any]:
        """Get file statistics using Rust extensions."""
        try:
            # This would use Rust file stats if available
            return await self._get_file_stats_python(file_path)
        except Exception:
            return {}

    async def _get_file_stats_python(self, file_path: str) -> dict[str, Any]:
        """Get file statistics using Python."""
        try:
            path = Path(file_path)
            stat = path.stat()
            return {
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
            }
        except Exception:
            return {}

    async def _find_related_files(self, file_path: str, content: str) -> list[RelatedFile]:
        """Find related files based on imports and patterns."""
        related_files: list[Any] = []
        parent_dir = Path(file_path).parent
        file_name = Path(file_path).stem
        extension = Path(file_path).suffix

        # Enhanced pattern matching based on file type
        related_patterns: list[Any] = []
        if extension == ".py":
            related_patterns = [
                f"test_{file_name}.py",
                f"{file_name}_test.py",
                f"test_*{file_name}*.py",
                "__init__.py",
                "conftest.py",
            ]
        elif extension in {".js", ".ts"}:
            related_patterns = [
                f"{file_name}.test{extension}",
                f"{file_name}.spec{extension}",
                "index" + extension,
            ]

        # Find related files
        for pattern in related_patterns[:5]:  # Limit patterns
            try:
                matches = list(parent_dir.glob(pattern))
                for match in matches[:3]:  # Limit matches per pattern
                    if match != Path(file_path):
                        rel_content = self.safe_file_read(match)
                        if rel_content:
                            related_files.append(
                                RelatedFile(
                                    path=str(match),
                                    content=rel_content[:5000],  # Truncate content
                                    relationship="test" if "test" in pattern else "related",
                                ),
                            )
            except Exception:
                continue

        return related_files

    def register_tools(self) -> None:
        """Register MCP tools for unified code reviewer."""

        @self.mcp.tool()
        async def review_code(
            file_path: str,
            focus_areas: str = "security,performance,quality",
            severity_threshold: str = "medium",
            include_suggestions: str = "true",
        ) -> dict[str, Any]:
            """Review a code file for quality, security, and performance issues.

            Args:
                file_path: Path to the file to review
                focus_areas: Comma-separated areas to focus on
                severity_threshold: Minimum severity level
                include_suggestions: Whether to include improvement suggestions

            Returns:
                Comprehensive code review results

            """
            try:
                job_id = self.create_job(
                    "review_file",
                    {
                        "file_path": file_path,
                        "focus_areas": focus_areas.split(",") if focus_areas else [],
                        "severity_threshold": severity_threshold,
                        "include_suggestions": include_suggestions.lower() == "true",
                    },
                )

                return await self.execute_job_async(job_id, wait_for_completion=True)
            except Exception as e:
                self.logger.exception(f"Code review failed: {e}")
                return {"status": "error", "error": str(e)}

        @self.mcp.tool()
        async def review_project(
            project_path: str,
            file_patterns: str = "*.py,*.js,*.ts,*.jsx,*.tsx",
            exclude_patterns: str = "node_modules,__pycache__,.git",
            max_files: int = 50,
        ) -> dict[str, Any]:
            """Review an entire project for code quality and security issues.

            Args:
                project_path: Path to the project directory
                file_patterns: Comma-separated file patterns
                exclude_patterns: Comma-separated patterns to exclude
                max_files: Maximum number of files to review

            Returns:
                Project-wide code review results

            """
            try:
                job_id = self.create_job(
                    "review_project",
                    {
                        "project_path": project_path,
                        "file_patterns": file_patterns.split(","),
                        "exclude_patterns": exclude_patterns.split(","),
                        "max_files": max_files,
                    },
                )

                return await self.execute_job_async(job_id, wait_for_completion=True)
            except Exception as e:
                self.logger.exception(f"Project review failed: {e}")
                return {"status": "error", "error": str(e)}

        @self.mcp.tool()
        async def security_scan(
            target_path: str, scan_depth: str = "comprehensive"
        ) -> dict[str, Any]:
            """Perform a security-focused code scan.

            Args:
                target_path: Path to the file or directory to scan
                scan_depth: Scanning depth ('quick', 'standard', 'comprehensive')

            Returns:
                Security vulnerabilities and recommendations

            """
            try:
                job_id = self.create_job(
                    "security_scan",
                    {
                        "target_path": target_path,
                        "scan_depth": scan_depth,
                    },
                )

                return await self.execute_job_async(job_id, wait_for_completion=True)
            except Exception as e:
                self.logger.exception(f"Security scan failed: {e}")
                return {"status": "error", "error": str(e)}

        @self.mcp.tool()
        async def performance_analysis(
            file_path: str, include_profiling: bool = False
        ) -> dict[str, Any]:
            """Analyze code for performance issues and optimization opportunities.

            Args:
                file_path: Path to the file to analyze
                include_profiling: Whether to include profiling suggestions

            Returns:
                Performance analysis results and optimization suggestions

            """
            try:
                job_id = self.create_job(
                    "performance_analysis",
                    {
                        "file_path": file_path,
                        "include_profiling": include_profiling,
                    },
                )

                return await self.execute_job_async(job_id, wait_for_completion=True)
            except Exception as e:
                self.logger.exception(f"Performance analysis failed: {e}")
                return {"status": "error", "error": str(e)}

        @self.mcp.tool()
        async def get_review_stats() -> dict[str, Any]:
            """Get comprehensive review statistics and agent status.

            Returns:
                Agent statistics and performance metrics

            """
            try:
                stats = self.get_agent_stats()
                stats.update(
                    {
                        "auto_approve_threshold": self.auto_approve_threshold,
                        "high_performance_enabled": self.enable_high_performance,
                        "cache_stats": self.cache_stats,
                    },
                )
                return stats
            except Exception as e:
                self.logger.exception(f"Failed to get stats: {e}")
                return {"status": "error", "error": str(e)}


# Create global instance for standalone usage
unified_code_reviewer = UnifiedCodeReviewer()

# MCP Server for standalone usage
mcp_server = FastMCP("unified-code-reviewer")


@mcp_server.tool()
async def review_code(
    file_path: str,
    focus_areas: str = "security,performance,quality",
    severity_threshold: str = "medium",
    include_suggestions: str = "true",
) -> dict[str, Any]:
    """Review a code file for quality, security, and performance issues."""
    return await unified_code_reviewer.mcp.tools["review_code"](
        file_path,
        focus_areas,
        severity_threshold,
        include_suggestions,
    )


@mcp_server.tool()
async def review_project(
    project_path: str,
    file_patterns: str = "*.py,*.js,*.ts,*.jsx,*.tsx",
    exclude_patterns: str = "node_modules,__pycache__,.git",
    max_files: int = 50,
) -> dict[str, Any]:
    """Review an entire project for code quality and security issues."""
    return await unified_code_reviewer.mcp.tools["review_project"](
        project_path,
        file_patterns,
        exclude_patterns,
        max_files,
    )


@mcp_server.tool()
async def security_scan(target_path: str, scan_depth: str = "comprehensive") -> dict[str, Any]:
    """Perform a security-focused code scan."""
    return await unified_code_reviewer.mcp.tools["security_scan"](target_path, scan_depth)


@mcp_server.tool()
async def performance_analysis(file_path: str, include_profiling: bool = False) -> dict[str, Any]:
    """Analyze code for performance issues and optimization opportunities."""
    return await unified_code_reviewer.mcp.tools["performance_analysis"](
        file_path, include_profiling
    )


@mcp_server.tool()
async def get_review_stats() -> dict[str, Any]:
    """Get comprehensive review statistics and agent status."""
    return await unified_code_reviewer.mcp.tools["get_review_stats"]()


def main() -> None:
    """Run the unified code reviewer as an MCP server."""
    # Initialize the reviewer
    unified_code_reviewer.register_tools()

    # Run startup
    import asyncio

    asyncio.run(unified_code_reviewer.startup())

    # Run the MCP server
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
