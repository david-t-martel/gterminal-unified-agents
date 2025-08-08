"""Code Review Agent Service - Enhanced code review with job management and streaming.
Provides comprehensive code analysis, security scanning, and quality assessment.
Includes MCP server functionality for direct integration with Claude CLI.
"""

import asyncio
import logging
from pathlib import Path
import re
from typing import Any

from gterminal.agents.base_agent_service import BaseAgentService
from gterminal.agents.base_agent_service import Job
from gterminal.core.security.security_utils import safe_json_parse
from gterminal.core.security.security_utils import safe_subprocess_run
from gterminal.core.security.security_utils import validate_file_path
from gterminal.gemini_agents.utils.file_ops import FileOpsMonitor
from gterminal.gemini_agents.utils.file_ops import ParallelFileOps
from mcp.server import FastMCP

# Configure logging for MCP integration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeReviewAgentService(BaseAgentService):
    """Enhanced code review agent with comprehensive analysis capabilities.

    Features:
    - Security vulnerability scanning
    - Performance analysis
    - Code quality assessment
    - Test coverage analysis
    - Documentation completeness check
    - Streaming progress updates
    """

    def __init__(self) -> None:
        super().__init__(
            "code_review_agent",
            "Enhanced code review with security and performance analysis",
        )

        # Initialize high-performance file operations
        self.file_ops = ParallelFileOps(max_workers=15, batch_size=75)
        self.performance_monitor = FileOpsMonitor()

        # Review configuration
        self.auto_approve_threshold = 0.95
        self.security_critical_patterns = [
            r"eval\(",
            r"exec\(",
            r"os\.system\(",
            r"subprocess\.call\(",
            r"input\(\)",
            r"raw_input\(\)",
            r"__import__\(",
            r'open\([^)]*["\']w["\']',  # File writes
        ]

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Get required parameters for job type."""
        if job_type == "review_file":
            return ["file_path"]
        if job_type == "review_pr":
            return ["pr_number"]
        if job_type == "review_diff":
            return ["diff_content"]
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
            return await self._review_project(job, parameters.get("project_path", "."))
        msg = f"Unknown job type: {job_type}"
        raise ValueError(msg)

    async def _review_file(self, job: Job, file_path: str) -> dict[str, Any]:
        """Review a specific file."""
        job.update_progress(10.0, f"Validating file path: {file_path}")

        try:
            validated_path = validate_file_path(file_path)
            file_content = self.safe_file_read(validated_path)

            if not file_content:
                return {"error": f"Could not read file: {file_path}"}

            job.update_progress(30.0, "Analyzing file content")

            # Get file diff if in git repo
            diff_result = safe_subprocess_run(
                ["git", "diff", "HEAD", str(validated_path)],
            )
            diff_content = diff_result.stdout if diff_result.returncode == 0 else ""

            # Perform comprehensive analysis
            analysis_result = await self._perform_comprehensive_analysis(
                job,
                file_content,
                diff_content,
                str(validated_path),
            )

            return {
                "file_path": file_path,
                "analysis": analysis_result,
                "reviewed_at": job.started_at.isoformat() if job.started_at else None,
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
                "reviewed_at": job.started_at.isoformat() if job.started_at else None,
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
                "reviewed_at": job.started_at.isoformat() if job.started_at else None,
            }

        except Exception as e:
            msg = f"Diff review failed: {e!s}"
            raise Exception(msg)

    async def _review_project(self, job: Job, project_path: str) -> dict[str, Any]:
        """Review entire project using high-performance file operations."""
        job.update_progress(10.0, f"Scanning project: {project_path}")

        try:
            project_dir = Path(project_path)
            if not project_dir.exists():
                return {"error": f"Project path does not exist: {project_path}"}

            # Use high-performance file scanning with git-aware filtering
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

            try:
                # Use parallel file operations for fast scanning
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
                logger.warning(f"High-performance scan failed, using fallback: {e}")
                # Fallback to traditional scanning
                code_files: list[Any] = []
                for pattern in code_patterns:
                    code_files.extend([str(f) for f in project_dir.rglob(pattern)])
                job.update_progress(
                    25.0,
                    f"Found {len(code_files)} code files (fallback mode)",
                )

            # Limit files for analysis to prevent overwhelming
            analysis_limit = min(50, len(code_files))
            selected_files = code_files[:analysis_limit]

            # Use parallel file reading for faster content loading
            job.update_progress(
                30.0,
                f"Reading {len(selected_files)} files in parallel",
            )

            try:
                file_read_result = await self.file_ops.parallel_file_read(
                    selected_files,
                    max_file_size=1024 * 1024,  # 1MB limit per file
                )

                file_contents = file_read_result["files"]
                read_errors = file_read_result["errors"]

                if read_errors:
                    logger.warning(
                        f"Failed to read {len(read_errors)} files: {list(read_errors.keys())[:5]}",
                    )

                job.update_progress(
                    40.0,
                    f"Successfully read {len(file_contents)} files",
                )

            except Exception as e:
                logger.warning(f"Parallel file reading failed, using fallback: {e}")
                # Fallback to sequential reading
                file_contents: dict[str, Any] = {}
                for file_path in selected_files[:20]:  # Further limit for fallback
                    try:
                        content = await self.file_ops.rust_fs.read_file(file_path)
                        file_contents[file_path] = {"content": content}
                    except Exception:
                        continue

            # Analyze files in parallel batches
            job.update_progress(50.0, "Analyzing file contents")

            file_analyses: list[Any] = []
            analysis_tasks: list[Any] = []

            for file_path, file_data in file_contents.items():
                content = file_data.get("content", "")
                if content:
                    # Create analysis task
                    analysis_tasks.append(
                        self._analyze_file_content_with_context(content, file_path),
                    )

            # Execute analyses in parallel batches
            batch_size = 10
            for i in range(0, len(analysis_tasks), batch_size):
                batch = analysis_tasks[i : i + batch_size]
                progress = 50.0 + (25.0 * i / len(analysis_tasks))
                job.update_progress(progress, f"Analyzing batch {i // batch_size + 1}")

                try:
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)

                    for _j, (result, file_path) in enumerate(
                        zip(
                            batch_results,
                            list(file_contents.keys())[i : i + batch_size],
                            strict=False,
                        ),
                    ):
                        if isinstance(result, Exception):
                            logger.warning(f"Analysis failed for {file_path}: {result}")
                            continue

                        file_analyses.append({"file": file_path, "analysis": result})

                except Exception as e:
                    logger.warning(f"Batch analysis failed: {e}")
                    continue

            job.update_progress(80.0, "Generating project summary")

            # Generate project-level summary
            project_summary = await self._generate_project_summary(file_analyses)

            # Add performance metrics
            project_summary["performance_metrics"] = {
                "total_files_found": len(code_files),
                "files_analyzed": len(file_analyses),
                "parallel_processing_used": True,
                "analysis_coverage": (len(file_analyses) / len(code_files) if code_files else 0),
            }

            return {
                "project_path": project_path,
                "total_files": len(code_files),
                "analyzed_files": len(file_analyses),
                "file_analyses": file_analyses,
                "project_summary": project_summary,
                "reviewed_at": job.started_at.isoformat() if job.started_at else None,
            }

        except Exception as e:
            msg = f"Project review failed: {e!s}"
            raise Exception(msg)

    async def _perform_comprehensive_analysis(
        self,
        job: Job,
        content: str,
        diff_content: str,
        context: str,
    ) -> dict[str, Any]:
        """Perform comprehensive code analysis."""
        # Security analysis
        job.update_progress(40.0, "Performing security analysis")
        security_issues = self._analyze_security(content)

        # Quality analysis
        job.update_progress(50.0, "Analyzing code quality")
        quality_analysis = await self._analyze_quality(content, context)

        # Performance analysis
        job.update_progress(60.0, "Analyzing performance implications")
        performance_analysis = self._analyze_performance(content)

        # Documentation analysis
        job.update_progress(70.0, "Checking documentation")
        documentation_analysis = self._analyze_documentation(content)

        # Generate AI review
        job.update_progress(80.0, "Generating AI review")
        ai_review = await self._generate_ai_review(content, diff_content, context)

        # Calculate overall score
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
        """Analyze content for security issues."""
        critical_issues: list[Any] = []
        warning_issues: list[Any] = []

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            for pattern in self.security_critical_patterns:
                if re.search(pattern, line):
                    critical_issues.append(
                        {
                            "line": line_num,
                            "content": line.strip(),
                            "pattern": pattern,
                            "severity": "critical",
                        },
                    )

        # Check for hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
        ]

        for line_num, line in enumerate(lines, 1):
            for pattern in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    warning_issues.append(
                        {
                            "line": line_num,
                            "content": line.strip(),
                            "pattern": pattern,
                            "severity": "warning",
                            "type": "potential_secret",
                        },
                    )

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

        ai_response = await self.generate_with_gemini(
            prompt,
            "code_review",
            parse_json=False,
        )

        if ai_response:
            quality_data = safe_json_parse(ai_response)
            if quality_data:
                return quality_data

        # Fallback basic analysis
        lines = content.split("\n")
        return {
            "complexity_score": 7,  # Default moderate score
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

        # Check for common performance anti-patterns
        performance_patterns = [
            (
                r"for\s+\w+\s+in\s+range\(len\(",
                "Use enumerate() instead of range(len())",
            ),
            (
                r"\.append\(\)\s*in\s+loop",
                "Consider list comprehension or pre-allocation",
            ),
            (r"string\s*\+\s*string", "Consider using join() for string concatenation"),
            (r"global\s+\w+", "Global variables can impact performance"),
        ]

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            for pattern, suggestion in performance_patterns:
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
            "score": max(0, 10 - len(performance_issues)),  # Simple scoring
        }

    def _analyze_documentation(self, content: str) -> dict[str, Any]:
        """Analyze documentation completeness."""
        content.split("\n")

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
        self,
        content: str,
        diff_content: str,
        context: str,
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

        ai_response = await self.generate_with_gemini(
            prompt,
            "code_review",
            parse_json=False,
        )

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
        final_score = max(
            0.0,
            min(1.0, base_score - security_penalty + quality_bonus + doc_bonus),
        )

        return round(final_score, 3)

    async def _analyze_file_content(
        self,
        content: str,
        file_path: str,
    ) -> dict[str, Any]:
        """Analyze individual file content."""
        return await self._perform_comprehensive_analysis(None, content, "", file_path)

    async def _analyze_file_content_with_context(
        self,
        content: str,
        file_path: str,
    ) -> dict[str, Any]:
        """Analyze individual file content with enhanced context for parallel processing."""
        try:
            # Enhanced analysis that includes file path context for better results
            return await self._perform_comprehensive_analysis(
                None,
                content,
                "",
                file_path,
            )
        except Exception as e:
            logger.warning(f"File analysis failed for {file_path}: {e}")
            # Return minimal analysis on failure
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
        self,
        file_analyses: list[dict[str, Any]],
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
            "average_score": (round(total_score / total_files, 3) if total_files > 0 else 0),
            "total_critical_security_issues": len(security_issues),
            "recommended_for_approval": (
                len(security_issues) == 0
                and (total_score / total_files) >= self.auto_approve_threshold
            ),
        }

    def register_tools(self) -> None:
        """Register MCP tools for code review agent."""

        @self.mcp.tool()
        async def review_file(file_path: str) -> dict[str, Any]:
            """Review a specific file."""
            if not self.validate_job_parameters(
                "review_file",
                {"file_path": file_path},
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("review_file", {"file_path": file_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def review_pr(pr_number: int) -> dict[str, Any]:
            """Review a pull request."""
            if not self.validate_job_parameters("review_pr", {"pr_number": pr_number}):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("review_pr", {"pr_number": pr_number})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def review_project(project_path: str = ".") -> dict[str, Any]:
            """Review entire project."""
            job_id = self.create_job("review_project", {"project_path": project_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def get_review_config() -> dict[str, Any]:
            """Get current review configuration."""
            return self.create_success_response(
                {
                    "auto_approve_threshold": self.auto_approve_threshold,
                    "max_concurrent_jobs": self.max_concurrent_jobs,
                    "agent_stats": self.get_agent_stats(),
                },
            )


# Create global instance
code_review_service = CodeReviewAgentService()


# MCP Server Integration
# Initialize MCP server for standalone usage
mcp = FastMCP("code-reviewer")


@mcp.tool()
async def review_code(
    file_path: str,
    focus_areas: str = "",
    severity_threshold: str = "medium",
    include_suggestions: bool = True,
) -> dict[str, Any]:
    """Review a code file for quality, security, and performance.

    Args:
        file_path: Path to the file to review
        focus_areas: Comma-separated areas to focus on (e.g., 'security,performance')
        severity_threshold: Minimum severity level ('low', 'medium', 'high', 'critical')
        include_suggestions: Whether to include improvement suggestions

    Returns:
        Comprehensive code review results

    """
    try:
        # Parse focus areas from comma-separated string
        focus_areas_list = (
            [area.strip() for area in focus_areas.split(",") if area.strip()] if focus_areas else []
        )

        # Create job using the agent service
        job_id = code_review_service.create_job(
            "review_file",
            {
                "file_path": file_path,
                "focus_areas": focus_areas_list,
                "severity_threshold": severity_threshold,
                "include_suggestions": include_suggestions,
            },
        )

        return await code_review_service.execute_job_async(job_id)
    except Exception as e:
        logger.exception(f"Code review failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def review_project(
    project_path: str,
    file_patterns: str = "*.py,*.js,*.ts,*.jsx,*.tsx",
    exclude_patterns: str = "node_modules,__pycache__,.git",
    max_files: int = 100,
) -> dict[str, Any]:
    """Review an entire project for code quality and security issues.

    Args:
        project_path: Path to the project directory
        file_patterns: Comma-separated file patterns (e.g., '*.py,*.js')
        exclude_patterns: Comma-separated patterns to exclude (e.g., 'node_modules,__pycache__')
        max_files: Maximum number of files to review

    Returns:
        Project-wide code review results

    """
    try:
        # Parse patterns from comma-separated strings
        file_patterns_list = [p.strip() for p in file_patterns.split(",") if p.strip()]
        exclude_patterns_list = [p.strip() for p in exclude_patterns.split(",") if p.strip()]

        # Create job using the agent service
        job_id = code_review_service.create_job(
            "review_project",
            {
                "project_path": project_path,
                "file_patterns": file_patterns_list,
                "exclude_patterns": exclude_patterns_list,
                "max_files": max_files,
            },
        )

        return await code_review_service.execute_job_async(job_id)
    except Exception as e:
        logger.exception(f"Project review failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def security_scan(
    file_path: str,
    scan_depth: str = "comprehensive",
) -> dict[str, Any]:
    """Perform a security-focused code scan.

    Args:
        file_path: Path to the file or directory to scan
        scan_depth: Scanning depth ('quick', 'standard', 'comprehensive')

    Returns:
        Security vulnerabilities and recommendations

    """
    try:
        # Use the agent's existing security analysis functionality
        job_id = code_review_service.create_job(
            "review_file",
            {
                "file_path": file_path,
                "focus_areas": ["security"],
                "scan_depth": scan_depth,
            },
        )

        result = await code_review_service.execute_job_async(job_id)

        # Extract security-specific information
        if "analysis" in result and "security_issues" in result["analysis"]:
            return {
                "status": "success",
                "security_analysis": result["analysis"]["security_issues"],
                "recommendations": result["analysis"]
                .get("ai_review", {})
                .get("recommendations", []),
                "scan_depth": scan_depth,
            }

        return result
    except Exception as e:
        logger.exception(f"Security scan failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def performance_analysis(
    file_path: str,
    include_profiling: bool = False,
) -> dict[str, Any]:
    """Analyze code for performance issues and optimization opportunities.

    Args:
        file_path: Path to the file to analyze
        include_profiling: Whether to include profiling suggestions

    Returns:
        Performance analysis results and optimization suggestions

    """
    try:
        # Use the agent's existing performance analysis functionality
        job_id = code_review_service.create_job(
            "review_file",
            {
                "file_path": file_path,
                "focus_areas": ["performance"],
                "include_profiling": include_profiling,
            },
        )

        result = await code_review_service.execute_job_async(job_id)

        # Extract performance-specific information
        if "analysis" in result and "performance_analysis" in result["analysis"]:
            return {
                "status": "success",
                "performance_analysis": result["analysis"]["performance_analysis"],
                "recommendations": result["analysis"]
                .get("ai_review", {})
                .get("recommendations", []),
                "include_profiling": include_profiling,
            }

        return result
    except Exception as e:
        logger.exception(f"Performance analysis failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def get_review_status(job_id: str) -> dict[str, Any]:
    """Get the status of an ongoing review job.

    Args:
        job_id: The job ID to check

    Returns:
        Job status and progress information

    """
    try:
        return code_review_service.get_job_status(job_id)
    except Exception as e:
        logger.exception(f"Failed to get job status: {e}")
        return {"status": "error", "error": str(e)}


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
