"""Unified Workspace Analyzer Agent.

This consolidated agent combines functionality from:
- WorkspaceAnalyzerService (app/agents/workspace_analyzer_agent.py)
- GeminiWorkspaceAnalyzer MCP Server (app/mcp_servers/gemini_workspace_analyzer.py)
- GeminiOrchestrator workspace analysis capabilities

Features:
- Comprehensive project analysis with parallel processing
- High-performance file operations via rust-fs integration
- Google Gemini AI integration for intelligent insights
- MCP server compatibility for Claude CLI integration
- Advanced security scanning and performance analysis
- Technology stack detection and architecture recommendations
- Progress tracking and streaming results
- Context caching with TTL for performance optimization
- Git-aware analysis with .gitignore support
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
import fnmatch
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import aiohttp
import cachetools
from google.api_core import retry
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
import vertexai
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel

from gterminal.core.agents.base_unified_agent import BaseUnifiedAgent
from gterminal.core.agents.base_unified_agent import Job

# Try to import PyO3 Rust extensions for performance
try:
    from fullstack_agent_rust import RustCache
    from fullstack_agent_rust import RustFileOps
    from fullstack_agent_rust import RustJsonProcessor

    RUST_EXTENSIONS_AVAILABLE = True
except ImportError:
    RUST_EXTENSIONS_AVAILABLE = False
    logging.warning("PyO3 Rust extensions not available - using Python fallbacks")

# Configure logging
logger = logging.getLogger(__name__)


class WorkspaceAnalysisRequest(BaseModel):
    """Pydantic model for workspace analysis requests."""

    project_path: str = Field(..., description="Path to the project directory")
    analysis_depth: str = Field(default="standard", pattern="^(quick|standard|comprehensive)$")
    include_dependencies: bool = Field(default=True)
    include_tests: bool = Field(default=True)
    include_security: bool = Field(default=True)
    focus_areas: str = Field(default="", description="Comma-separated focus areas")
    max_files: int = Field(default=1000, ge=1, le=10000)


class FileSearchRequest(BaseModel):
    """Pydantic model for file search requests."""

    directory: str = Field(..., description="Directory to search")
    pattern: str = Field(default="*")
    exclude_patterns: str = Field(default="node_modules,.git,target,__pycache__")
    max_results: int = Field(default=1000, ge=1, le=5000)


class ContentSearchRequest(BaseModel):
    """Pydantic model for content search requests."""

    directory: str = Field(..., description="Directory to search")
    search_pattern: str = Field(..., description="Text pattern to search for")
    file_patterns: str = Field(default="*.py,*.js,*.ts,*.rs,*.go")
    max_results: int = Field(default=100, ge=1, le=1000)
    include_context: bool = Field(default=True)


class SecurityIssue(BaseModel):
    """Model for security issues found during analysis."""

    type: str
    severity: str  # low, medium, high, critical
    file_path: str
    line_number: int | None = None
    description: str
    recommendation: str
    cwe_id: str | None = None


class PerformanceBottleneck(BaseModel):
    """Model for performance bottlenecks identified."""

    type: str
    severity: str  # low, medium, high
    file_path: str
    line_number: int | None = None
    description: str
    optimization_suggestion: str
    estimated_impact: str


class AnalysisResult(BaseModel):
    """Comprehensive analysis result model."""

    project_path: str
    analysis_timestamp: float
    analysis_depth: str
    focus_areas: list[str]
    rust_performance: bool
    project_info: dict[str, Any]
    tech_stack: dict[str, Any]
    file_structure: dict[str, Any]
    dependencies: dict[str, Any]
    code_metrics: dict[str, Any]
    security_analysis: dict[str, Any]
    performance_analysis: dict[str, Any]
    ai_insights: dict[str, Any]
    recommendations: list[dict[str, Any]]
    improvement_suggestions: list[str]
    key_findings: list[str]


class UnifiedWorkspaceAnalyzer(BaseUnifiedAgent):
    """Unified workspace analyzer combining all workspace analysis capabilities.

    This agent consolidates:
    - WorkspaceAnalyzerService with comprehensive project analysis
    - GeminiWorkspaceAnalyzer MCP server with AI insights
    - High-performance rust-fs integration
    - Advanced security and performance scanning
    """

    def __init__(self) -> None:
        super().__init__(
            agent_name="unified_workspace_analyzer",
            description="Comprehensive project workspace analysis with AI insights",
        )

        # Initialize Rust extensions if available
        self.rust_file_ops = RustFileOps() if RUST_EXTENSIONS_AVAILABLE else None
        self.rust_cache = (
            RustCache(capacity=10000, ttl_seconds=3600) if RUST_EXTENSIONS_AVAILABLE else {}
        )
        self.rust_json = RustJsonProcessor() if RUST_EXTENSIONS_AVAILABLE else None

        # Technology stack detection patterns
        self.tech_patterns = {
            "python": [
                "*.py",
                "requirements.txt",
                "pyproject.toml",
                "setup.py",
                "Pipfile",
                "poetry.lock",
            ],
            "javascript": ["*.js", "*.ts", "package.json", "*.jsx", "*.tsx", "yarn.lock"],
            "java": ["*.java", "pom.xml", "build.gradle", "*.jar", "gradle.properties"],
            "csharp": ["*.cs", "*.csproj", "*.sln", "packages.config"],
            "go": ["*.go", "go.mod", "go.sum", "Gopkg.toml"],
            "rust": ["*.rs", "Cargo.toml", "Cargo.lock"],
            "php": ["*.php", "composer.json", "composer.lock"],
            "ruby": ["*.rb", "Gemfile", "Gemfile.lock"],
            "cpp": ["*.cpp", "*.c", "*.h", "*.hpp", "CMakeLists.txt", "Makefile"],
            "docker": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml", ".dockerignore"],
            "kubernetes": ["*.yaml", "*.yml", "helm/", "k8s/"],
            "terraform": ["*.tf", "*.tfvars", "terraform.tfstate"],
            "scala": ["*.scala", "build.sbt", "project/"],
            "swift": ["*.swift", "Package.swift", "*.xcodeproj"],
        }

        # Security scanning patterns
        self.security_patterns = {
            "hardcoded_secrets": [
                r'password\s*[=:]\s*["\'][^"\']{8,}["\']',
                r'api_key\s*[=:]\s*["\'][^"\']{20,}["\']',
                r'secret\s*[=:]\s*["\'][^"\']{10,}["\']',
                r'token\s*[=:]\s*["\'][^"\']{20,}["\']',
                r'private_key\s*[=:]\s*["\'][^"\']{50,}["\']',
            ],
            "sql_injection": [
                r'execute\s*\(\s*["\'].*%s.*["\']',
                r'query\s*\(\s*["\'].*\+.*["\']',
                r"SELECT.*WHERE.*=.*\+",
            ],
            "xss_patterns": [
                r"innerHTML\s*=.*\+",
                r"document\.write\s*\(",
                r"eval\s*\(",
            ],
        }

        # Performance bottleneck patterns
        self.performance_patterns = {
            "inefficient_loops": [
                (r"for\s+\w+\s+in\s+range\(len\(", "Use enumerate() instead of range(len())"),
                (r"while\s+True:", "Potential infinite loop detected"),
                (r'\.join\([\'"][\'\"]\)\s*\.join', "Multiple string joins can be optimized"),
            ],
            "blocking_operations": [
                (r"time\.sleep\(", "Blocking sleep call found"),
                (r"input\s*\(", "Blocking input call"),
                (r"requests\.get\((?!.*timeout)", "HTTP request without timeout"),
            ],
            "memory_issues": [
                (r"\.readlines\(\)", "Reading entire file into memory"),
                (r"pickle\.load\((?!.*\bwith\b)", "Pickle without context manager"),
            ],
        }

        # Google Vertex AI configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "auricleinc-gemini")
        self.location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        self.model_name = "gemini-2.0-flash-exp"

        # Initialize Vertex AI
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.gemini_model = GenerativeModel(
                model_name=self.model_name,
                generation_config=GenerationConfig(
                    max_output_tokens=8192,
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Vertex AI: {e}")
            self.gemini_model = None

        # Context cache with TTL (5 minutes)
        self.context_cache = cachetools.TTLCache(maxsize=100, ttl=300)

        # Thread pool for parallel I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=8)

        # HTTP session for external API calls
        self.http_session: aiohttp.ClientSession | None = None

        logger.info(
            f"UnifiedWorkspaceAnalyzer initialized with Rust extensions: {RUST_EXTENSIONS_AVAILABLE}"
        )

    async def initialize_http_session(self) -> None:
        """Initialize HTTP session with connection pooling."""
        if self.http_session is None or self.http_session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
                use_dns_cache=True,
            )
            self.http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=60, connect=10),
                headers={"User-Agent": "UnifiedWorkspaceAnalyzer/1.0"},
            )

    async def cleanup_resources(self) -> None:
        """Clean up resources."""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        self.thread_pool.shutdown(wait=True)

    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Execute workspace analysis job implementation."""
        job_type = job.job_type
        parameters = job.parameters

        try:
            if job_type == "analyze_workspace":
                return await self._analyze_workspace(job, parameters)
            if job_type == "find_files":
                return await self._find_files(job, parameters)
            if job_type == "search_content":
                return await self._search_content(job, parameters)
            if job_type == "analyze_dependencies":
                return await self._analyze_dependencies(job, parameters)
            if job_type == "analyze_security":
                return await self._analyze_security(job, parameters)
            if job_type == "analyze_performance":
                return await self._analyze_performance(job, parameters)
            if job_type == "get_project_overview":
                return await self._get_project_overview(job, parameters)
            msg = f"Unknown job type: {job_type}"
            raise ValueError(msg)

        except Exception as e:
            logger.exception(f"Job execution failed for {job_type}: {e}")
            return {"status": "error", "error": str(e)}

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Get required parameters for specific job types."""
        param_map = {
            "analyze_workspace": ["project_path"],
            "find_files": ["directory"],
            "search_content": ["directory", "search_pattern"],
            "analyze_dependencies": ["project_path"],
            "analyze_security": ["project_path"],
            "analyze_performance": ["project_path"],
            "get_project_overview": ["project_path"],
        }
        return param_map.get(job_type, [])

    async def _analyze_workspace(self, job: Job, params: dict[str, Any]) -> dict[str, Any]:
        """Comprehensive workspace analysis combining all analysis types."""
        try:
            # Validate input parameters
            request = WorkspaceAnalysisRequest(**params)

            job.update_progress(1.0, f"Starting comprehensive analysis of {request.project_path}")

            # Check cache first
            cache_key = self._get_cache_key(
                "workspace_analysis",
                request.project_path,
                request.analysis_depth,
                request.focus_areas,
            )
            if cache_key in self.context_cache:
                job.update_progress(100.0, "Returning cached analysis result")
                cached_result = self.context_cache[cache_key]
                cached_result["cached"] = True
                return cached_result

            project_path = Path(request.project_path)
            if not project_path.exists():
                return {
                    "status": "error",
                    "error": f"Project path does not exist: {request.project_path}",
                }

            focus_areas_list = (
                [area.strip() for area in request.focus_areas.split(",") if area.strip()]
                if request.focus_areas
                else []
            )

            # Initialize analysis result
            analysis_result = {
                "project_path": request.project_path,
                "analysis_timestamp": time.time(),
                "analysis_depth": request.analysis_depth,
                "focus_areas": focus_areas_list,
                "rust_performance": RUST_EXTENSIONS_AVAILABLE,
                "cached": False,
            }

            # Phase 1: Basic project information (5%)
            job.update_progress(5.0, "Gathering project information")
            analysis_result["project_info"] = await self._get_project_info(project_path)

            # Phase 2: Technology stack detection (15%)
            job.update_progress(15.0, "Detecting technology stack")
            analysis_result["tech_stack"] = await self._detect_tech_stack(
                project_path, request.max_files
            )

            # Phase 3: File structure analysis (25%)
            job.update_progress(25.0, "Analyzing file structure")
            analysis_result["file_structure"] = await self._analyze_file_structure(
                project_path, request.max_files
            )

            # Phase 4: Dependencies analysis (40%)
            if request.include_dependencies:
                job.update_progress(40.0, "Analyzing dependencies")
                analysis_result["dependencies"] = await self._analyze_project_dependencies(
                    project_path
                )
            else:
                analysis_result["dependencies"] = {"skipped": True}

            # Phase 5: Code metrics (55%)
            job.update_progress(55.0, "Calculating code metrics")
            analysis_result["code_metrics"] = await self._calculate_code_metrics(
                project_path, request.max_files
            )

            # Phase 6: Security analysis (70%)
            if request.include_security:
                job.update_progress(70.0, "Performing security analysis")
                analysis_result["security_analysis"] = await self._perform_security_analysis(
                    project_path
                )
            else:
                analysis_result["security_analysis"] = {"skipped": True}

            # Phase 7: Performance analysis (80%)
            job.update_progress(80.0, "Analyzing performance patterns")
            analysis_result["performance_analysis"] = await self._analyze_performance_patterns(
                project_path
            )

            # Phase 8: AI-powered insights with Gemini (90%)
            if self.gemini_model:
                job.update_progress(90.0, "Generating AI insights")
                analysis_result["ai_insights"] = await self._get_gemini_insights(
                    analysis_result, focus_areas_list
                )
            else:
                analysis_result["ai_insights"] = {"error": "Gemini AI not available"}

            # Phase 9: Generate recommendations and findings (95%)
            job.update_progress(95.0, "Generating recommendations")
            analysis_result["recommendations"] = self._generate_recommendations(analysis_result)
            analysis_result["improvement_suggestions"] = self._generate_improvement_suggestions(
                analysis_result
            )
            analysis_result["key_findings"] = self._extract_key_findings(analysis_result)

            # Cache the result
            self.context_cache[cache_key] = analysis_result

            job.update_progress(100.0, "Workspace analysis completed successfully")

            return {"status": "success", "analysis": analysis_result}

        except ValidationError as e:
            error_msg = f"Input validation failed: {e}"
            logger.exception(error_msg)
            return {"status": "error", "error": error_msg}
        except Exception as e:
            error_msg = f"Workspace analysis failed: {e}"
            logger.exception(error_msg)
            return {"status": "error", "error": error_msg}

    async def _find_files(self, job: Job, params: dict[str, Any]) -> dict[str, Any]:
        """High-performance file search using rust-fs or Python fallback."""
        try:
            request = FileSearchRequest(**params)

            job.update_progress(10.0, f"Starting file search in {request.directory}")

            # Check cache first
            cache_key = self._get_cache_key(
                "find_files", request.directory, request.pattern, request.exclude_patterns
            )
            if cache_key in self.context_cache:
                job.update_progress(100.0, "Returning cached file search result")
                return self.context_cache[cache_key]

            exclude_list = [p.strip() for p in request.exclude_patterns.split(",") if p.strip()]

            if self.rust_file_ops:
                # Use high-performance Rust implementation
                job.update_progress(50.0, "Performing high-speed file search with Rust")
                files = await self._run_in_thread(
                    self.rust_file_ops.find_files,
                    request.directory,
                    request.pattern,
                    max_results=request.max_results,
                )
                # Filter excludes
                filtered_files = [
                    f for f in files if not any(excl in str(f) for excl in exclude_list)
                ]
            else:
                # Python fallback
                job.update_progress(50.0, "Performing file search with Python fallback")
                filtered_files = await self._python_find_files(
                    request.directory,
                    request.pattern,
                    exclude_list,
                    request.max_results,
                )

            result = {
                "status": "success",
                "files": [str(f) for f in filtered_files[: request.max_results]],
                "total_count": len(filtered_files),
                "search_pattern": request.pattern,
                "exclude_patterns": exclude_list,
                "rust_powered": RUST_EXTENSIONS_AVAILABLE and self.rust_file_ops is not None,
                "cached": False,
            }

            # Cache the result
            self.context_cache[cache_key] = result

            job.update_progress(100.0, f"Found {len(filtered_files)} files")

            return result

        except Exception as e:
            error_msg = f"File search failed: {e}"
            logger.exception(error_msg)
            return {"status": "error", "error": error_msg}

    async def _search_content(self, job: Job, params: dict[str, Any]) -> dict[str, Any]:
        """High-performance content search using rust-fs or ripgrep fallback."""
        try:
            request = ContentSearchRequest(**params)

            job.update_progress(10.0, f"Starting content search in {request.directory}")

            # Check cache first
            cache_key = self._get_cache_key(
                "search_content",
                request.directory,
                request.search_pattern,
                request.file_patterns,
            )
            if cache_key in self.context_cache:
                job.update_progress(100.0, "Returning cached content search result")
                return self.context_cache[cache_key]

            if self.rust_file_ops:
                # Use high-performance Rust implementation
                job.update_progress(50.0, "Performing high-speed content search with Rust")
                results = await self._run_in_thread(
                    self.rust_file_ops.search_content,
                    request.directory,
                    request.search_pattern,
                    max_results=request.max_results,
                )
            else:
                # Python/ripgrep fallback
                job.update_progress(50.0, "Performing content search with Python fallback")
                results = await self._python_search_content(
                    request.directory,
                    request.search_pattern,
                    request.file_patterns,
                    request.max_results,
                )

            result = {
                "status": "success",
                "results": results[: request.max_results],
                "total_matches": len(results),
                "search_pattern": request.search_pattern,
                "file_patterns": request.file_patterns,
                "rust_powered": RUST_EXTENSIONS_AVAILABLE and self.rust_file_ops is not None,
                "cached": False,
            }

            # Cache the result
            self.context_cache[cache_key] = result

            job.update_progress(100.0, f"Found {len(results)} content matches")

            return result

        except Exception as e:
            error_msg = f"Content search failed: {e}"
            logger.exception(error_msg)
            return {"status": "error", "error": error_msg}

    async def _get_project_overview(self, job: Job, params: dict[str, Any]) -> dict[str, Any]:
        """Quick project overview combining structure analysis and AI insights."""
        try:
            project_path = params.get("project_path", ".")

            job.update_progress(10.0, f"Generating project overview for {project_path}")

            # Check cache first
            cache_key = self._get_cache_key("project_overview", project_path)
            if cache_key in self.context_cache:
                job.update_progress(100.0, "Returning cached project overview")
                return self.context_cache[cache_key]

            path = Path(project_path)
            if not path.exists():
                return {"status": "error", "error": f"Project path does not exist: {project_path}"}

            # Quick structural analysis
            job.update_progress(30.0, "Analyzing project structure")
            if self.rust_file_ops:
                stats = await self._run_in_thread(
                    self.rust_file_ops.get_project_stats, project_path
                )
            else:
                stats = await self._python_project_stats(project_path)

            # Quick AI overview
            job.update_progress(70.0, "Generating AI overview")
            if self.gemini_model:
                overview_prompt = f"""
                Analyze this project overview and provide a concise summary:

                Project Path: {project_path}
                Statistics: {json.dumps(stats, indent=2)}

                Please provide:
                1. Project type and main technology stack
                2. Key architectural patterns
                3. Overall code quality assessment (1-10 scale)
                4. Primary purpose/domain
                5. Notable strengths and areas for improvement

                Keep the response concise but informative.
                """

                ai_overview = await self._generate_content_with_retry(overview_prompt)
            else:
                ai_overview = "AI overview not available (Gemini not initialized)"

            result = {
                "status": "success",
                "project_path": project_path,
                "statistics": stats,
                "ai_overview": ai_overview,
                "rust_powered": RUST_EXTENSIONS_AVAILABLE and self.rust_file_ops is not None,
                "cached": False,
            }

            # Cache the result
            self.context_cache[cache_key] = result

            job.update_progress(100.0, "Project overview completed")

            return result

        except Exception as e:
            error_msg = f"Project overview failed: {e}"
            logger.exception(error_msg)
            return {"status": "error", "error": error_msg}

    async def _get_project_info(self, project_path: Path) -> dict[str, Any]:
        """Get basic project information including git status."""
        info = {
            "name": project_path.name,
            "absolute_path": str(project_path.absolute()),
            "size_mb": await self._calculate_directory_size(project_path),
            "is_git_repo": (project_path / ".git").exists(),
            "created_date": None,
            "last_modified": None,
        }

        # Git information
        if info["is_git_repo"]:
            git_info = await self._get_git_info(project_path)
            info.update(git_info)

        # Try to get creation/modification dates
        try:
            stat = project_path.stat()
            info["last_modified"] = stat.st_mtime
        except Exception:
            pass

        return info

    async def _detect_tech_stack(self, project_path: Path, max_files: int) -> dict[str, Any]:
        """Detect technology stack using high-performance operations."""
        detected_techs = {}

        try:
            # Use parallel file operations for faster tech stack detection
            search_tasks = []
            tech_mapping = {}

            # Create search tasks for all technology patterns
            for tech, patterns in self.tech_patterns.items():
                for pattern in patterns:
                    if self.rust_file_ops:
                        task = self._run_in_thread(
                            self.rust_file_ops.find_files,
                            str(project_path),
                            pattern,
                            max_results=min(50, max_files // 10),
                        )
                    else:
                        task = self._python_find_files(
                            str(project_path), pattern, [], min(50, max_files // 10)
                        )

                    search_tasks.append(task)
                    tech_mapping[len(search_tasks) - 1] = tech

            # Execute searches in parallel
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Process results
            for i, files in enumerate(search_results):
                if isinstance(files, Exception):
                    continue

                tech = tech_mapping[i]
                if files:  # Files found for this pattern
                    if tech not in detected_techs:
                        detected_techs[tech] = {
                            "detected": True,
                            "files": [],
                            "count": 0,
                        }

                    # Convert to relative paths and limit display
                    relative_files = []
                    for file_path in files:
                        try:
                            rel_path = str(Path(file_path).relative_to(project_path))
                            relative_files.append(rel_path)
                        except ValueError:
                            relative_files.append(str(file_path))

                    detected_techs[tech]["files"].extend(relative_files[:10])  # Limit display
                    detected_techs[tech]["count"] += len(files)

        except Exception as e:
            logger.warning(f"High-performance tech stack detection failed: {e}")
            # Fallback to simpler detection
            detected_techs = await self._fallback_tech_detection(project_path)

        # Detect frameworks and libraries
        frameworks = await self._detect_frameworks(project_path, detected_techs)

        return {
            "technologies": detected_techs,
            "frameworks": frameworks,
            "primary_language": self._determine_primary_language(detected_techs),
            "complexity_score": self._calculate_tech_complexity(detected_techs),
        }

    async def _analyze_file_structure(self, project_path: Path, max_files: int) -> dict[str, Any]:
        """Analyze project file structure using high-performance operations."""
        structure = {
            "total_files": 0,
            "total_directories": 0,
            "file_types": {},
            "large_files": [],
            "empty_directories": [],
            "structure_tree": {},
            "performance_stats": {},
        }

        try:
            if self.rust_file_ops:
                # Use rust-fs for high-performance directory scanning
                scan_result = await self._run_in_thread(
                    self.rust_file_ops.scan_directory,
                    str(project_path),
                    max_files=max_files,
                    include_git_metadata=True,
                )

                structure["total_files"] = scan_result.get("file_count", 0)
                structure["total_directories"] = scan_result.get("directory_count", 0)
                structure["performance_stats"] = scan_result.get("performance", {})

                # Analyze file types and sizes
                if "files" in scan_result:
                    file_analysis = await self._analyze_files_parallel(
                        scan_result["files"], project_path
                    )
                    structure.update(file_analysis)
            else:
                # Python fallback
                structure = await self._python_file_structure_analysis(project_path, max_files)

        except Exception as e:
            logger.warning(f"High-performance file scan failed, using fallback: {e}")
            structure = await self._python_file_structure_analysis(project_path, max_files)

        # Generate structure recommendations
        structure["recommendations"] = self._generate_structure_recommendations(structure)

        return structure

    async def _analyze_project_dependencies(self, project_path: Path) -> dict[str, Any]:
        """Analyze project dependencies across different technologies."""
        dependencies = {
            "python": await self._analyze_python_dependencies(project_path),
            "javascript": await self._analyze_js_dependencies(project_path),
            "java": await self._analyze_java_dependencies(project_path),
            "rust": await self._analyze_rust_dependencies(project_path),
            "go": await self._analyze_go_dependencies(project_path),
            "security_vulnerabilities": [],
            "outdated_packages": [],
            "dependency_tree_depth": 0,
        }

        # Security vulnerability scanning
        dependencies["security_vulnerabilities"] = await self._scan_dependency_vulnerabilities(
            project_path,
            dependencies,
        )

        return dependencies

    async def _calculate_code_metrics(self, project_path: Path, max_files: int) -> dict[str, Any]:
        """Calculate comprehensive code metrics."""
        metrics = {
            "lines_of_code": 0,
            "lines_of_comments": 0,
            "blank_lines": 0,
            "cyclomatic_complexity": 0,
            "functions": 0,
            "classes": 0,
            "files_analyzed": 0,
            "languages": {},
            "comment_ratio": 0.0,
            "code_ratio": 0.0,
            "maintainability_index": 0.0,
        }

        # Find code files using high-performance search
        code_extensions = [
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
            ".php",
            ".rb",
            ".scala",
            ".swift",
        ]
        code_files = []

        for ext in code_extensions:
            if self.rust_file_ops:
                files = await self._run_in_thread(
                    self.rust_file_ops.find_files,
                    str(project_path),
                    f"*{ext}",
                    max_results=max_files // len(code_extensions),
                )
            else:
                files = await self._python_find_files(
                    str(project_path),
                    f"*{ext}",
                    [],
                    max_files // len(code_extensions),
                )
            code_files.extend(files)

        metrics["files_analyzed"] = len(code_files)

        # Analyze files in batches for better performance
        batch_size = 50
        for i in range(0, len(code_files), batch_size):
            batch = code_files[i : i + batch_size]
            batch_metrics = await self._analyze_code_files_batch(batch, project_path)

            # Aggregate metrics
            for key in [
                "lines_of_code",
                "lines_of_comments",
                "blank_lines",
                "functions",
                "classes",
            ]:
                metrics[key] += batch_metrics.get(key, 0)

            # Merge language metrics
            for lang, lang_metrics in batch_metrics.get("languages", {}).items():
                if lang not in metrics["languages"]:
                    metrics["languages"][lang] = {"files": 0, "loc": 0}
                metrics["languages"][lang]["files"] += lang_metrics["files"]
                metrics["languages"][lang]["loc"] += lang_metrics["loc"]

        # Calculate ratios and derived metrics
        total_lines = (
            metrics["lines_of_code"] + metrics["lines_of_comments"] + metrics["blank_lines"]
        )
        if total_lines > 0:
            metrics["comment_ratio"] = metrics["lines_of_comments"] / total_lines
            metrics["code_ratio"] = metrics["lines_of_code"] / total_lines

            # Simple maintainability index calculation
            # Based on lines of code, cyclomatic complexity, and comment ratio
            complexity_penalty = max(0, metrics["cyclomatic_complexity"] - 100) * 0.01
            comment_bonus = metrics["comment_ratio"] * 20
            size_penalty = max(0, (total_lines - 10000) / 10000) * 10

            metrics["maintainability_index"] = max(
                0, min(100, 80 - complexity_penalty + comment_bonus - size_penalty)
            )

        return metrics

    async def _perform_security_analysis(self, project_path: Path) -> dict[str, Any]:
        """Perform comprehensive security analysis."""
        security_analysis = {
            "sensitive_files": [],
            "hardcoded_secrets": [],
            "security_vulnerabilities": [],
            "insecure_dependencies": [],
            "security_score": 10.0,  # Start with perfect score
            "cwe_mappings": {},
            "recommendations": [],
        }

        # Check for sensitive files
        sensitive_patterns = [
            "*.key",
            "*.pem",
            "*.p12",
            "*.jks",
            "*.keystore",
            ".env",
            ".env.*",
            "config.json",
            "secrets.*",
            "*.credentials",
            "id_rsa",
            "id_dsa",
            "*.pfx",
        ]

        for pattern in sensitive_patterns:
            if self.rust_file_ops:
                files = await self._run_in_thread(
                    self.rust_file_ops.find_files, str(project_path), pattern
                )
            else:
                files = await self._python_find_files(str(project_path), pattern, [], 100)

            for file_path in files:
                relative_path = str(Path(file_path).relative_to(project_path))
                risk_level = (
                    "high"
                    if any(ext in str(file_path) for ext in [".key", ".pem", ".p12"])
                    else "medium"
                )

                security_analysis["sensitive_files"].append(
                    {
                        "path": relative_path,
                        "type": "sensitive_file",
                        "risk": risk_level,
                    },
                )

        # Scan for hardcoded secrets in code files
        code_files = []
        for ext in [".py", ".js", ".ts", ".java", ".go", ".rs", ".php", ".rb"]:
            if self.rust_file_ops:
                files = await self._run_in_thread(
                    self.rust_file_ops.find_files,
                    str(project_path),
                    f"*{ext}",
                    max_results=200,
                )
            else:
                files = await self._python_find_files(str(project_path), f"*{ext}", [], 200)
            code_files.extend(files[:50])  # Limit per extension

        # Scan files for secrets patterns
        secret_scan_tasks = []
        for file_path in code_files:
            secret_scan_tasks.append(self._scan_file_for_secrets(file_path, project_path))

        secret_results = await asyncio.gather(*secret_scan_tasks, return_exceptions=True)

        for secrets_found in secret_results:
            if isinstance(secrets_found, Exception):
                continue
            security_analysis["hardcoded_secrets"].extend(secrets_found)

        # Scan for common security vulnerabilities
        vuln_scan_tasks = []
        for file_path in code_files:
            vuln_scan_tasks.append(self._scan_file_for_vulnerabilities(file_path, project_path))

        vuln_results = await asyncio.gather(*vuln_scan_tasks, return_exceptions=True)

        for vulns_found in vuln_results:
            if isinstance(vulns_found, Exception):
                continue
            security_analysis["security_vulnerabilities"].extend(vulns_found)

        # Calculate security score
        total_issues = (
            len(security_analysis["sensitive_files"])
            + len(security_analysis["hardcoded_secrets"]) * 2  # Secrets are weighted higher
            + len(security_analysis["security_vulnerabilities"])
            + len(security_analysis["insecure_dependencies"])
        )

        # Deduct points based on severity and count
        score_deduction = min(10.0, total_issues * 0.5)
        security_analysis["security_score"] = max(0.0, 10.0 - score_deduction)

        # Generate recommendations
        if security_analysis["hardcoded_secrets"]:
            security_analysis["recommendations"].append(
                "Remove hardcoded secrets and use environment variables or secret management systems",
            )
        if security_analysis["sensitive_files"]:
            security_analysis["recommendations"].append(
                "Review sensitive files and ensure they are properly secured and not in version control",
            )
        if security_analysis["security_vulnerabilities"]:
            security_analysis["recommendations"].append(
                "Address identified security vulnerabilities following secure coding practices",
            )

        return security_analysis

    async def _analyze_performance_patterns(self, project_path: Path) -> dict[str, Any]:
        """Analyze code for performance patterns and bottlenecks."""
        performance_analysis = {
            "bottlenecks": [],
            "optimization_opportunities": [],
            "performance_score": 8.0,  # Default good score
            "recommendations": [],
            "patterns_found": {},
        }

        # Find code files for analysis
        code_files = []
        for ext in [".py", ".js", ".ts", ".java", ".go", ".rs"]:
            if self.rust_file_ops:
                files = await self._run_in_thread(
                    self.rust_file_ops.find_files,
                    str(project_path),
                    f"*{ext}",
                    max_results=100,
                )
            else:
                files = await self._python_find_files(str(project_path), f"*{ext}", [], 100)
            code_files.extend(files[:30])  # Limit per extension

        # Analyze files for performance patterns
        pattern_scan_tasks = []
        for file_path in code_files:
            pattern_scan_tasks.append(
                self._scan_file_for_performance_patterns(file_path, project_path)
            )

        pattern_results = await asyncio.gather(*pattern_scan_tasks, return_exceptions=True)

        for patterns_found in pattern_results:
            if isinstance(patterns_found, Exception):
                continue
            performance_analysis["bottlenecks"].extend(patterns_found.get("bottlenecks", []))
            performance_analysis["optimization_opportunities"].extend(
                patterns_found.get("optimizations", [])
            )

        # Count pattern occurrences
        for bottleneck in performance_analysis["bottlenecks"]:
            pattern_type = bottleneck.get("type", "unknown")
            performance_analysis["patterns_found"][pattern_type] = (
                performance_analysis["patterns_found"].get(pattern_type, 0) + 1
            )

        # Calculate performance score
        total_issues = len(performance_analysis["bottlenecks"])
        high_severity_issues = len(
            [b for b in performance_analysis["bottlenecks"] if b.get("severity") == "high"]
        )

        score_deduction = min(8.0, total_issues * 0.3 + high_severity_issues * 0.5)
        performance_analysis["performance_score"] = max(0.0, 8.0 - score_deduction)

        # Generate recommendations
        if high_severity_issues > 0:
            performance_analysis["recommendations"].append(
                "Address high-severity performance bottlenecks first"
            )
        if total_issues > 10:
            performance_analysis["recommendations"].append(
                "Consider performance profiling for detailed analysis"
            )
        if "inefficient_loops" in performance_analysis["patterns_found"]:
            performance_analysis["recommendations"].append(
                "Review loop implementations for efficiency improvements"
            )
        if "blocking_operations" in performance_analysis["patterns_found"]:
            performance_analysis["recommendations"].append(
                "Replace blocking operations with async alternatives where possible",
            )

        return performance_analysis

    @retry.Retry(predicate=retry.if_exception_type(Exception))
    async def _generate_content_with_retry(self, prompt: str) -> str:
        """Generate content using Vertex AI with retry logic."""
        if not self.gemini_model:
            return "Gemini AI not available"

        try:
            response = await self._run_in_thread(lambda: self.gemini_model.generate_content(prompt))

            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            return "No content generated"

        except Exception as e:
            logger.exception(f"Content generation failed: {e}")
            raise

    async def _get_gemini_insights(
        self, analysis_result: dict[str, Any], focus_areas: list[str]
    ) -> dict[str, Any]:
        """Get AI-powered insights from Gemini analysis."""
        if not self.gemini_model:
            return {"error": "Gemini AI not available"}

        try:
            # Prepare context for Gemini
            context = self._prepare_gemini_context(analysis_result, focus_areas)

            analysis_prompt = f"""
            {context}

            Based on the comprehensive analysis above, provide detailed insights in JSON format:

            {{
                "architecture_assessment": {{
                    "quality_score": "1-10 with detailed explanation",
                    "patterns_identified": ["list of architectural patterns found"],
                    "strengths": ["key architectural strengths"],
                    "weaknesses": ["areas needing architectural improvement"]
                }},
                "code_quality_analysis": {{
                    "overall_score": "1-10 with explanation",
                    "maintainability": "assessment of code maintainability",
                    "readability": "assessment of code readability",
                    "consistency": "assessment of coding style consistency"
                }},
                "security_insights": {{
                    "risk_level": "low/medium/high overall risk assessment",
                    "critical_issues": ["most important security concerns"],
                    "compliance_notes": ["regulatory or standard compliance observations"]
                }},
                "performance_insights": {{
                    "efficiency_score": "1-10 with explanation",
                    "scalability_concerns": ["potential scalability bottlenecks"],
                    "optimization_priorities": ["prioritized performance improvements"]
                }},
                "technology_recommendations": {{
                    "current_stack_assessment": "evaluation of current technology choices",
                    "modernization_opportunities": ["suggestions for technology updates"],
                    "integration_improvements": ["ways to better integrate existing technologies"]
                }},
                "development_workflow": {{
                    "process_maturity": "assessment of development practices",
                    "tooling_recommendations": ["suggestions for development tooling"],
                    "best_practices": ["recommended best practices to adopt"]
                }},
                "strategic_recommendations": {{
                    "short_term_priorities": ["immediate actions to take"],
                    "medium_term_goals": ["goals for next 3-6 months"],
                    "long_term_vision": ["strategic direction recommendations"]
                }}
            }}
            """

            response = await self._generate_content_with_retry(analysis_prompt)

            # Try to parse as JSON, fallback to structured text
            try:
                if self.rust_json:
                    return self.rust_json.parse_json(response)
                return json.loads(response)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse AI response as JSON: {e}")
                return {"ai_insights_text": response}

        except Exception as e:
            logger.exception(f"Gemini insights generation failed: {e}")
            return {"error": f"AI insights generation failed: {e}"}

    def _prepare_gemini_context(
        self, analysis_result: dict[str, Any], focus_areas: list[str]
    ) -> str:
        """Prepare context for Gemini analysis."""
        context_parts = []

        context_parts.append("PROJECT ANALYSIS CONTEXT")
        context_parts.append(f"Project Path: {analysis_result.get('project_path', 'unknown')}")
        context_parts.append(f"Analysis Depth: {analysis_result.get('analysis_depth', 'standard')}")
        context_parts.append(
            f"Focus Areas: {', '.join(focus_areas) if focus_areas else 'General analysis'}"
        )
        context_parts.append(
            f"Rust Performance Enabled: {analysis_result.get('rust_performance', False)}"
        )
        context_parts.append("")

        # Add structural analysis results
        if "project_info" in analysis_result:
            context_parts.append("PROJECT INFORMATION:")
            context_parts.append(json.dumps(analysis_result["project_info"], indent=2))
            context_parts.append("")

        if "tech_stack" in analysis_result:
            context_parts.append("TECHNOLOGY STACK:")
            context_parts.append(json.dumps(analysis_result["tech_stack"], indent=2))
            context_parts.append("")

        if "code_metrics" in analysis_result:
            context_parts.append("CODE METRICS:")
            context_parts.append(json.dumps(analysis_result["code_metrics"], indent=2))
            context_parts.append("")

        if "security_analysis" in analysis_result and analysis_result["security_analysis"].get(
            "security_score"
        ):
            context_parts.append(
                f"SECURITY SCORE: {analysis_result['security_analysis']['security_score']}/10"
            )
            context_parts.append("")

        if "performance_analysis" in analysis_result and analysis_result[
            "performance_analysis"
        ].get(
            "performance_score",
        ):
            context_parts.append(
                f"PERFORMANCE SCORE: {analysis_result['performance_analysis']['performance_score']}/10",
            )
            context_parts.append("")

        context_parts.append("Please provide comprehensive insights focusing on:")
        context_parts.append("1. Architecture quality and design patterns")
        context_parts.append("2. Code organization and maintainability")
        context_parts.append("3. Security posture and risk assessment")
        context_parts.append("4. Performance characteristics and optimization opportunities")
        context_parts.append("5. Technology stack evaluation and modernization")
        context_parts.append("6. Development workflow and process maturity")
        context_parts.append("7. Strategic recommendations for improvement")

        return "\n".join(context_parts)

    def _generate_recommendations(self, analysis_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []

        # Security recommendations
        security_analysis = analysis_result.get("security_analysis", {})
        if security_analysis.get("security_score", 10) < 7:
            recommendations.append(
                {
                    "category": "security",
                    "priority": "high",
                    "title": "Improve Security Posture",
                    "description": "Security score below recommended threshold. Address identified vulnerabilities.",
                    "actions": security_analysis.get("recommendations", []),
                },
            )

        # Performance recommendations
        performance_analysis = analysis_result.get("performance_analysis", {})
        if len(performance_analysis.get("bottlenecks", [])) > 5:
            recommendations.append(
                {
                    "category": "performance",
                    "priority": "medium",
                    "title": "Address Performance Bottlenecks",
                    "description": "Multiple performance issues detected that could impact application responsiveness.",
                    "actions": performance_analysis.get("recommendations", []),
                },
            )

        # Architecture recommendations
        tech_stack = analysis_result.get("tech_stack", {})
        if tech_stack.get("complexity_score", 0) > 7:
            recommendations.append(
                {
                    "category": "architecture",
                    "priority": "medium",
                    "title": "Simplify Technology Stack",
                    "description": "High technology complexity may impact maintainability and development velocity.",
                    "actions": [
                        "Consider consolidating similar technologies",
                        "Evaluate necessity of each technology",
                    ],
                },
            )

        # Code quality recommendations
        code_metrics = analysis_result.get("code_metrics", {})
        if code_metrics.get("comment_ratio", 1) < 0.1:
            recommendations.append(
                {
                    "category": "maintainability",
                    "priority": "low",
                    "title": "Improve Documentation",
                    "description": "Low comment ratio suggests insufficient code documentation.",
                    "actions": [
                        "Add comments to complex functions",
                        "Document public APIs",
                        "Create architectural documentation",
                    ],
                },
            )

        # File structure recommendations
        file_structure = analysis_result.get("file_structure", {})
        if len(file_structure.get("large_files", [])) > 0:
            recommendations.append(
                {
                    "category": "maintainability",
                    "priority": "medium",
                    "title": "Refactor Large Files",
                    "description": "Large files detected that may impact maintainability.",
                    "actions": [
                        "Split large files into smaller modules",
                        "Extract common functionality",
                    ],
                },
            )

        return recommendations

    def _generate_improvement_suggestions(self, analysis_result: dict[str, Any]) -> list[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []

        # Extract suggestions from AI insights
        ai_insights = analysis_result.get("ai_insights", {})
        if isinstance(ai_insights, dict):
            # Try to extract structured suggestions
            for insights in ai_insights.values():
                if isinstance(insights, dict) and "recommendations" in insights:
                    if isinstance(insights["recommendations"], list):
                        suggestions.extend(insights["recommendations"])
                elif isinstance(insights, list):
                    suggestions.extend(insights)

        # Add suggestions based on analysis scores
        security_score = analysis_result.get("security_analysis", {}).get("security_score", 10)
        if security_score < 8:
            suggestions.append("Implement security scanning in CI/CD pipeline")
            suggestions.append("Review and update dependency security policies")

        performance_score = analysis_result.get("performance_analysis", {}).get(
            "performance_score", 10
        )
        if performance_score < 7:
            suggestions.append("Set up performance monitoring and alerting")
            suggestions.append("Consider implementing caching strategies")

        # Code quality suggestions
        maintainability_index = analysis_result.get("code_metrics", {}).get(
            "maintainability_index", 100
        )
        if maintainability_index < 70:
            suggestions.append("Refactor complex functions to improve readability")
            suggestions.append("Establish and enforce coding standards")

        return suggestions[:10]  # Limit to top 10 suggestions

    def _extract_key_findings(self, analysis_result: dict[str, Any]) -> list[str]:
        """Extract key findings from the comprehensive analysis."""
        findings = []

        # Project scale findings
        project_info = analysis_result.get("project_info", {})
        if "size_mb" in project_info:
            findings.append(f"Project size: {project_info['size_mb']} MB")

        # Technology findings
        tech_stack = analysis_result.get("tech_stack", {})
        if "primary_language" in tech_stack:
            findings.append(f"Primary language: {tech_stack['primary_language']}")

        tech_count = len(tech_stack.get("technologies", {}))
        if tech_count > 0:
            findings.append(f"Technologies detected: {tech_count}")

        # Code metrics findings
        code_metrics = analysis_result.get("code_metrics", {})
        if "lines_of_code" in code_metrics:
            findings.append(f"Lines of code: {code_metrics['lines_of_code']:,}")
        if "files_analyzed" in code_metrics:
            findings.append(f"Code files analyzed: {code_metrics['files_analyzed']}")

        # Security findings
        security_analysis = analysis_result.get("security_analysis", {})
        if "security_score" in security_analysis:
            findings.append(f"Security score: {security_analysis['security_score']}/10")

        # Performance findings
        performance_analysis = analysis_result.get("performance_analysis", {})
        if "performance_score" in performance_analysis:
            findings.append(f"Performance score: {performance_analysis['performance_score']}/10")

        bottleneck_count = len(performance_analysis.get("bottlenecks", []))
        if bottleneck_count > 0:
            findings.append(f"Performance bottlenecks identified: {bottleneck_count}")

        return findings

    # Helper methods for file operations and analysis

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run blocking function in thread pool."""
        return await asyncio.get_running_loop().run_in_executor(
            self.thread_pool, lambda: func(*args, **kwargs)
        )

    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments."""
        return f"{prefix}:{hash(str(args))}"

    async def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate directory size in MB."""
        if self.rust_file_ops:
            try:
                stats = await self._run_in_thread(
                    self.rust_file_ops.get_project_stats, str(directory)
                )
                return stats.get("size_mb", 0.0)
            except Exception:
                pass

        # Python fallback
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return round(total_size / (1024 * 1024), 2)

    async def _get_git_info(self, project_path: Path) -> dict[str, Any]:
        """Get Git repository information."""
        git_info = {}

        original_cwd = os.getcwd()
        try:
            os.chdir(project_path)

            # Get branch info
            try:
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if result.returncode == 0:
                    git_info["current_branch"] = result.stdout.strip()
            except Exception:
                pass

            # Get commit count
            try:
                result = subprocess.run(
                    ["git", "rev-list", "--count", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if result.returncode == 0:
                    git_info["total_commits"] = int(result.stdout.strip())
            except Exception:
                pass

            # Get remote info
            try:
                result = subprocess.run(
                    ["git", "remote", "-v"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if result.returncode == 0:
                    git_info["remotes"] = result.stdout.strip().split("\n")
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Error getting git info: {e}")
        finally:
            os.chdir(original_cwd)

        return git_info

    # Python fallback implementations

    async def _python_find_files(
        self,
        directory: str,
        pattern: str,
        exclude_list: list[str],
        max_results: int,
    ) -> list[str]:
        """Python implementation of file finding."""
        files = []
        try:
            for root, dirs, filenames in os.walk(directory):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not any(excl in d for excl in exclude_list)]

                for filename in filenames:
                    if fnmatch.fnmatch(filename, pattern):
                        full_path = str(Path(root) / filename)
                        if not any(excl in full_path for excl in exclude_list):
                            files.append(full_path)
                            if len(files) >= max_results:
                                return files
        except Exception as e:
            logger.exception(f"Python file search failed: {e}")

        return files

    async def _python_search_content(
        self,
        directory: str,
        pattern: str,
        file_patterns: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Python implementation of content search."""
        results = []
        file_patterns_list = [p.strip() for p in file_patterns.split(",")]

        try:
            for root, _dirs, files in os.walk(directory):
                for file in files:
                    if any(file.endswith(p.replace("*", "")) for p in file_patterns_list):
                        file_path = Path(root) / file
                        try:
                            with open(file_path, encoding="utf-8", errors="ignore") as f:
                                for line_num, line in enumerate(f, 1):
                                    if re.search(pattern, line, re.IGNORECASE):
                                        results.append(
                                            {
                                                "file": str(file_path),
                                                "line": line_num,
                                                "content": line.strip(),
                                            },
                                        )
                                        if len(results) >= max_results:
                                            return results
                        except (UnicodeDecodeError, PermissionError):
                            continue
        except Exception as e:
            logger.exception(f"Python content search failed: {e}")

        return results

    async def _python_project_stats(self, project_path: str) -> dict[str, Any]:
        """Python implementation of project statistics."""
        try:
            path = Path(project_path)
            if not path.exists():
                return {"error": "Project path does not exist"}

            files = list(path.rglob("*"))
            total_files = len([f for f in files if f.is_file()])
            total_dirs = len([f for f in files if f.is_dir()])

            # File type breakdown
            extensions = {}
            total_size = 0
            for file in files:
                if file.is_file():
                    ext = file.suffix.lower() or "no_extension"
                    extensions[ext] = extensions.get(ext, 0) + 1
                    with contextlib.suppress(Exception):
                        total_size += file.stat().st_size

            return {
                "total_files": total_files,
                "total_directories": total_dirs,
                "file_extensions": extensions,
                "project_size_bytes": total_size,
                "size_mb": round(total_size / (1024 * 1024), 2),
            }
        except Exception as e:
            return {"error": str(e)}

    async def _fallback_tech_detection(self, project_path: Path) -> dict[str, Any]:
        """Fallback technology detection using Python."""
        detected_techs = {}

        for tech, patterns in self.tech_patterns.items():
            files_found = []
            for pattern in patterns:
                files = list(project_path.rglob(pattern))
                files_found.extend([str(f.relative_to(project_path)) for f in files])

            if files_found:
                detected_techs[tech] = {
                    "detected": True,
                    "files": files_found[:10],  # Limit display
                    "count": len(files_found),
                }

        return detected_techs

    async def _python_file_structure_analysis(
        self, project_path: Path, max_files: int
    ) -> dict[str, Any]:
        """Python fallback for file structure analysis."""
        structure = {
            "total_files": 0,
            "total_directories": 0,
            "file_types": {},
            "large_files": [],
            "empty_directories": [],
            "performance_stats": {"used_rust_fs": False},
        }

        file_count = 0
        for root, dirs, files in os.walk(project_path):
            if file_count >= max_files:
                break

            root_path = Path(root)
            structure["total_directories"] += len(dirs)

            for file in files:
                if file_count >= max_files:
                    break

                file_path = root_path / file
                file_ext = file_path.suffix.lower()

                structure["total_files"] += 1
                file_count += 1

                # Count file types
                if file_ext:
                    structure["file_types"][file_ext] = structure["file_types"].get(file_ext, 0) + 1

                # Check for large files (>10MB)
                try:
                    file_size = file_path.stat().st_size
                    if file_size > 10 * 1024 * 1024:  # 10MB
                        structure["large_files"].append(
                            {
                                "path": str(file_path.relative_to(project_path)),
                                "size_mb": round(file_size / (1024 * 1024), 2),
                            },
                        )
                except Exception:
                    pass

            # Check for empty directories
            if not files and not dirs:
                with contextlib.suppress(ValueError):
                    structure["empty_directories"].append(str(root_path.relative_to(project_path)))

        return structure

    # More detailed analysis methods (continued...)

    async def _detect_frameworks(
        self, project_path: Path, detected_techs: dict[str, Any]
    ) -> dict[str, Any]:
        """Detect frameworks based on detected technologies."""
        frameworks = {}

        # Python frameworks
        if "python" in detected_techs:
            requirements_file = project_path / "requirements.txt"
            pyproject_file = project_path / "pyproject.toml"

            # Check requirements.txt
            if requirements_file.exists():
                try:
                    content = requirements_file.read_text()
                    if "django" in content.lower():
                        frameworks["django"] = {"type": "web_framework", "language": "python"}
                    if "flask" in content.lower():
                        frameworks["flask"] = {"type": "web_framework", "language": "python"}
                    if "fastapi" in content.lower():
                        frameworks["fastapi"] = {"type": "web_framework", "language": "python"}
                    if "pytorch" in content.lower():
                        frameworks["pytorch"] = {"type": "ml_framework", "language": "python"}
                    if "tensorflow" in content.lower():
                        frameworks["tensorflow"] = {"type": "ml_framework", "language": "python"}
                except Exception:
                    pass

            # Check pyproject.toml
            if pyproject_file.exists():
                try:
                    content = pyproject_file.read_text()
                    if "poetry" in content.lower():
                        frameworks["poetry"] = {"type": "package_manager", "language": "python"}
                except Exception:
                    pass

        # JavaScript frameworks
        if "javascript" in detected_techs:
            package_json = project_path / "package.json"
            if package_json.exists():
                try:
                    content = package_json.read_text()
                    package_data = (
                        self.rust_json.parse_json(content)
                        if self.rust_json
                        else json.loads(content)
                    )

                    deps = {
                        **package_data.get("dependencies", {}),
                        **package_data.get("devDependencies", {}),
                    }

                    if "react" in deps:
                        frameworks["react"] = {
                            "type": "frontend_framework",
                            "language": "javascript",
                        }
                    if "vue" in deps:
                        frameworks["vue"] = {"type": "frontend_framework", "language": "javascript"}
                    if "angular" in deps or "@angular/core" in deps:
                        frameworks["angular"] = {
                            "type": "frontend_framework",
                            "language": "javascript",
                        }
                    if "express" in deps:
                        frameworks["express"] = {
                            "type": "backend_framework",
                            "language": "javascript",
                        }
                    if "next" in deps or "nextjs" in deps:
                        frameworks["nextjs"] = {
                            "type": "fullstack_framework",
                            "language": "javascript",
                        }

                except Exception:
                    pass

        # Java frameworks
        if "java" in detected_techs:
            pom_xml = project_path / "pom.xml"
            build_gradle = project_path / "build.gradle"

            if pom_xml.exists():
                try:
                    content = pom_xml.read_text()
                    if "spring" in content.lower():
                        frameworks["spring"] = {"type": "backend_framework", "language": "java"}
                    if "hibernate" in content.lower():
                        frameworks["hibernate"] = {"type": "orm_framework", "language": "java"}
                except Exception:
                    pass

            if build_gradle.exists():
                try:
                    content = build_gradle.read_text()
                    if "spring" in content.lower():
                        frameworks["spring"] = {"type": "backend_framework", "language": "java"}
                except Exception:
                    pass

        return frameworks

    def _determine_primary_language(self, detected_techs: dict[str, Any]) -> str:
        """Determine the primary programming language."""
        if not detected_techs:
            return "unknown"

        # Count files by technology
        tech_counts = {
            tech: data.get("count", 0)
            for tech, data in detected_techs.items()
            if data.get("detected", False)
        }

        if not tech_counts:
            return "unknown"

        return max(tech_counts, key=tech_counts.get)

    def _calculate_tech_complexity(self, detected_techs: dict[str, Any]) -> int:
        """Calculate technology complexity score (1-10)."""
        detected_count = sum(1 for data in detected_techs.values() if data.get("detected", False))

        if detected_count <= 2:
            return 3  # Simple
        if detected_count <= 4:
            return 5  # Moderate
        if detected_count <= 6:
            return 7  # Complex
        return 9  # Very complex

    def _generate_structure_recommendations(self, structure: dict[str, Any]) -> list[str]:
        """Generate file structure recommendations."""
        recommendations = []

        if len(structure.get("large_files", [])) > 0:
            recommendations.append("Consider splitting large files for better maintainability")

        if len(structure.get("empty_directories", [])) > 5:
            recommendations.append("Clean up empty directories to reduce clutter")

        if structure.get("total_files", 0) > 1000:
            recommendations.append("Large project detected. Consider modularization")

        # File type recommendations
        file_types = structure.get("file_types", {})
        if ".pyc" in file_types:
            recommendations.append(
                "Add *.pyc to .gitignore to avoid committing compiled Python files"
            )

        if ".DS_Store" in file_types:
            recommendations.append(
                "Add .DS_Store to .gitignore to avoid committing macOS system files"
            )

        return recommendations

    # Dependency analysis methods for different languages

    async def _analyze_python_dependencies(self, project_path: Path) -> dict[str, Any]:
        """Analyze Python dependencies."""
        deps_info = {"packages": [], "outdated": [], "vulnerabilities": [], "files_found": []}

        # Check requirements.txt
        requirements_file = project_path / "requirements.txt"
        if requirements_file.exists():
            deps_info["files_found"].append("requirements.txt")
            try:
                content = requirements_file.read_text()
                packages = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip() and not line.startswith("#")
                ]
                deps_info["packages"] = packages[:50]  # Limit to first 50
            except Exception:
                pass

        # Check pyproject.toml
        pyproject_file = project_path / "pyproject.toml"
        if pyproject_file.exists():
            deps_info["files_found"].append("pyproject.toml")
            # Could parse TOML here if needed

        # Check Pipfile
        pipfile = project_path / "Pipfile"
        if pipfile.exists():
            deps_info["files_found"].append("Pipfile")

        return deps_info

    async def _analyze_js_dependencies(self, project_path: Path) -> dict[str, Any]:
        """Analyze JavaScript dependencies."""
        deps_info = {"packages": [], "outdated": [], "vulnerabilities": [], "files_found": []}

        package_json = project_path / "package.json"
        if package_json.exists():
            deps_info["files_found"].append("package.json")
            try:
                content = package_json.read_text()
                package_data = (
                    self.rust_json.parse_json(content) if self.rust_json else json.loads(content)
                )

                all_deps = {
                    **package_data.get("dependencies", {}),
                    **package_data.get("devDependencies", {}),
                }
                deps_info["packages"] = [f"{k}@{v}" for k, v in all_deps.items()]

            except Exception:
                pass

        # Check yarn.lock
        yarn_lock = project_path / "yarn.lock"
        if yarn_lock.exists():
            deps_info["files_found"].append("yarn.lock")

        # Check package-lock.json
        package_lock = project_path / "package-lock.json"
        if package_lock.exists():
            deps_info["files_found"].append("package-lock.json")

        return deps_info

    async def _analyze_java_dependencies(self, project_path: Path) -> dict[str, Any]:
        """Analyze Java dependencies."""
        deps_info = {"packages": [], "build_system": None, "files_found": []}

        # Check for Maven
        pom_xml = project_path / "pom.xml"
        if pom_xml.exists():
            deps_info["build_system"] = "maven"
            deps_info["files_found"].append("pom.xml")

        # Check for Gradle
        build_gradle = project_path / "build.gradle"
        if build_gradle.exists():
            deps_info["build_system"] = "gradle"
            deps_info["files_found"].append("build.gradle")

        build_gradle_kts = project_path / "build.gradle.kts"
        if build_gradle_kts.exists():
            deps_info["build_system"] = "gradle"
            deps_info["files_found"].append("build.gradle.kts")

        return deps_info

    async def _analyze_rust_dependencies(self, project_path: Path) -> dict[str, Any]:
        """Analyze Rust dependencies."""
        deps_info = {"packages": [], "files_found": []}

        cargo_toml = project_path / "Cargo.toml"
        if cargo_toml.exists():
            deps_info["files_found"].append("Cargo.toml")
            # Could parse TOML here if needed

        cargo_lock = project_path / "Cargo.lock"
        if cargo_lock.exists():
            deps_info["files_found"].append("Cargo.lock")

        return deps_info

    async def _analyze_go_dependencies(self, project_path: Path) -> dict[str, Any]:
        """Analyze Go dependencies."""
        deps_info = {"packages": [], "files_found": []}

        go_mod = project_path / "go.mod"
        if go_mod.exists():
            deps_info["files_found"].append("go.mod")

        go_sum = project_path / "go.sum"
        if go_sum.exists():
            deps_info["files_found"].append("go.sum")

        return deps_info

    async def _scan_dependency_vulnerabilities(
        self,
        project_path: Path,
        dependencies: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Scan for security vulnerabilities in dependencies."""
        return []

        # This would integrate with security scanning tools like:
        # - npm audit for JavaScript
        # - safety for Python
        # - OWASP Dependency Check for Java
        # For now, return empty list as placeholder

    # Security scanning methods

    async def _scan_file_for_secrets(
        self, file_path: str | Path, project_path: Path
    ) -> list[dict[str, Any]]:
        """Scan file for hardcoded secrets."""
        secrets = []

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    for pattern in self.security_patterns["hardcoded_secrets"]:
                        if re.search(pattern, line, re.IGNORECASE):
                            relative_path = str(Path(file_path).relative_to(project_path))
                            secrets.append(
                                {
                                    "file_path": relative_path,
                                    "line_number": line_num,
                                    "type": "hardcoded_secret",
                                    "severity": "high",
                                    "description": "Potential hardcoded secret detected",
                                    "recommendation": "Move secrets to environment variables or secure vault",
                                    "cwe_id": "CWE-798",
                                },
                            )
        except Exception:
            pass

        return secrets

    async def _scan_file_for_vulnerabilities(
        self, file_path: str | Path, project_path: Path
    ) -> list[dict[str, Any]]:
        """Scan file for security vulnerabilities."""
        vulnerabilities = []

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

                relative_path = str(Path(file_path).relative_to(project_path))

                # Check for SQL injection patterns
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.security_patterns["sql_injection"]:
                        if re.search(pattern, line, re.IGNORECASE):
                            vulnerabilities.append(
                                {
                                    "file_path": relative_path,
                                    "line_number": line_num,
                                    "type": "sql_injection",
                                    "severity": "high",
                                    "description": "Potential SQL injection vulnerability",
                                    "recommendation": "Use parameterized queries or prepared statements",
                                    "cwe_id": "CWE-89",
                                },
                            )

                # Check for XSS patterns
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.security_patterns["xss_patterns"]:
                        if re.search(pattern, line, re.IGNORECASE):
                            vulnerabilities.append(
                                {
                                    "file_path": relative_path,
                                    "line_number": line_num,
                                    "type": "xss",
                                    "severity": "medium",
                                    "description": "Potential XSS vulnerability",
                                    "recommendation": "Sanitize user input and use safe DOM manipulation",
                                    "cwe_id": "CWE-79",
                                },
                            )

        except Exception:
            pass

        return vulnerabilities

    # Performance analysis methods

    async def _scan_file_for_performance_patterns(
        self, file_path: str | Path, project_path: Path
    ) -> dict[str, Any]:
        """Scan file for performance patterns and bottlenecks."""
        result = {"bottlenecks": [], "optimizations": []}

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

                relative_path = str(Path(file_path).relative_to(project_path))

                # Check for inefficient loop patterns
                for line_num, line in enumerate(lines, 1):
                    for pattern, message in self.performance_patterns["inefficient_loops"]:
                        if re.search(pattern, line):
                            result["bottlenecks"].append(
                                {
                                    "file_path": relative_path,
                                    "line_number": line_num,
                                    "type": "inefficient_loop",
                                    "severity": "medium",
                                    "description": message,
                                    "optimization_suggestion": "Consider using more efficient iteration patterns",
                                    "estimated_impact": "Medium - could improve execution time",
                                },
                            )

                # Check for blocking operations
                for line_num, line in enumerate(lines, 1):
                    for pattern, message in self.performance_patterns["blocking_operations"]:
                        if re.search(pattern, line):
                            result["bottlenecks"].append(
                                {
                                    "file_path": relative_path,
                                    "line_number": line_num,
                                    "type": "blocking_operation",
                                    "severity": "high",
                                    "description": message,
                                    "optimization_suggestion": "Use async alternatives or proper timeout handling",
                                    "estimated_impact": "High - could cause application freezes",
                                },
                            )

                # Check for memory issues
                for line_num, line in enumerate(lines, 1):
                    for pattern, message in self.performance_patterns["memory_issues"]:
                        if re.search(pattern, line):
                            result["bottlenecks"].append(
                                {
                                    "file_path": relative_path,
                                    "line_number": line_num,
                                    "type": "memory_issue",
                                    "severity": "medium",
                                    "description": message,
                                    "optimization_suggestion": "Use streaming or chunked processing for large data",
                                    "estimated_impact": "Medium - could cause memory spikes",
                                },
                            )

        except Exception:
            pass

        return result

    async def _analyze_files_parallel(self, files: list[str], project_path: Path) -> dict[str, Any]:
        """Analyze files in parallel for types, sizes, etc."""
        file_analysis = {"file_types": {}, "large_files": []}

        # Process files in batches for better performance
        batch_size = 100
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]

            # Analyze batch
            for file_path in batch:
                try:
                    path = Path(file_path)

                    # Count file types
                    file_ext = path.suffix.lower()
                    if file_ext:
                        file_analysis["file_types"][file_ext] = (
                            file_analysis["file_types"].get(file_ext, 0) + 1
                        )

                    # Check for large files (>10MB)
                    if path.exists() and path.is_file():
                        file_size = path.stat().st_size
                        if file_size > 10 * 1024 * 1024:  # 10MB
                            try:
                                rel_path = str(path.relative_to(project_path))
                            except ValueError:
                                rel_path = str(path)

                            file_analysis["large_files"].append(
                                {
                                    "path": rel_path,
                                    "size_mb": round(file_size / (1024 * 1024), 2),
                                },
                            )

                except Exception:
                    continue

        return file_analysis

    async def _analyze_code_files_batch(
        self, files: list[str | Path], project_path: Path
    ) -> dict[str, Any]:
        """Analyze a batch of code files for metrics."""
        batch_metrics = {
            "lines_of_code": 0,
            "lines_of_comments": 0,
            "blank_lines": 0,
            "functions": 0,
            "classes": 0,
            "languages": {},
        }

        for file_path in files:
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                file_ext = Path(file_path).suffix
                file_metrics = self._analyze_single_file_metrics(content, file_ext)

                # Aggregate metrics
                for key in [
                    "lines_of_code",
                    "lines_of_comments",
                    "blank_lines",
                    "functions",
                    "classes",
                ]:
                    batch_metrics[key] += file_metrics.get(key, 0)

                # Language-specific metrics
                if file_ext not in batch_metrics["languages"]:
                    batch_metrics["languages"][file_ext] = {"files": 0, "loc": 0}
                batch_metrics["languages"][file_ext]["files"] += 1
                batch_metrics["languages"][file_ext]["loc"] += file_metrics.get("lines_of_code", 0)

            except Exception:
                continue

        return batch_metrics

    def _analyze_single_file_metrics(self, content: str, file_extension: str) -> dict[str, int]:
        """Analyze metrics for a single file."""
        lines = content.split("\n")
        metrics = {
            "lines_of_code": 0,
            "lines_of_comments": 0,
            "blank_lines": 0,
            "functions": 0,
            "classes": 0,
        }

        for line in lines:
            stripped = line.strip()
            if not stripped:
                metrics["blank_lines"] += 1
            elif stripped.startswith(("#", "//", "/*")):
                metrics["lines_of_comments"] += 1
            else:
                metrics["lines_of_code"] += 1

        # Count functions and classes based on file type
        if file_extension == ".py":
            metrics["functions"] = len([l for l in lines if l.strip().startswith("def ")])
            metrics["classes"] = len([l for l in lines if l.strip().startswith("class ")])
        elif file_extension in [".js", ".ts"]:
            metrics["functions"] = len([l for l in lines if "function" in l])
            metrics["classes"] = len([l for l in lines if l.strip().startswith("class ")])
        elif file_extension == ".java":
            metrics["functions"] = len(
                [l for l in lines if re.search(r"\b(public|private|protected).*\w+\s*\(", l)]
            )
            metrics["classes"] = len(
                [
                    l
                    for l in lines
                    if l.strip().startswith("class ") or l.strip().startswith("public class ")
                ],
            )

        return metrics

    def register_mcp_tools(self) -> None:
        """Register MCP tools for workspace analyzer."""

        @self.mcp.tool()
        async def analyze_workspace(
            project_path: str = ".",
            analysis_depth: str = "standard",
            include_dependencies: bool = True,
            include_tests: bool = True,
            focus_areas: str = "",
        ) -> dict[str, Any]:
            """Comprehensive workspace analysis."""
            job_id = self.create_job(
                "analyze_workspace",
                {
                    "project_path": project_path,
                    "analysis_depth": analysis_depth,
                    "include_dependencies": include_dependencies,
                    "include_tests": include_tests,
                    "focus_areas": focus_areas,
                },
            )
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def find_files(
            directory: str,
            pattern: str = "*",
            exclude_patterns: str = "node_modules,.git,target,__pycache__",
        ) -> dict[str, Any]:
            """High-performance file search."""
            job_id = self.create_job(
                "find_files",
                {"directory": directory, "pattern": pattern, "exclude_patterns": exclude_patterns},
            )
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def search_content(
            directory: str,
            search_pattern: str,
            file_patterns: str = "*.py,*.js,*.ts,*.rs,*.go",
            max_results: int = 100,
        ) -> dict[str, Any]:
            """High-performance content search."""
            job_id = self.create_job(
                "search_content",
                {
                    "directory": directory,
                    "search_pattern": search_pattern,
                    "file_patterns": file_patterns,
                    "max_results": max_results,
                },
            )
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def get_project_overview(project_path: str = ".") -> dict[str, Any]:
            """Quick project overview."""
            job_id = self.create_job("get_project_overview", {"project_path": project_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def clear_analysis_cache(cache_pattern: str = "*") -> dict[str, Any]:
            """Clear analysis cache."""
            try:
                cleared_count = 0
                total_count = len(self.context_cache)

                if cache_pattern == "*":
                    self.context_cache.clear()
                    cleared_count = total_count
                else:
                    keys_to_remove = []
                    for key in self.context_cache:
                        if fnmatch.fnmatch(key, cache_pattern):
                            keys_to_remove.append(key)

                    for key in keys_to_remove:
                        del self.context_cache[key]
                        cleared_count += 1

                return {
                    "status": "success",
                    "cleared_count": cleared_count,
                    "remaining_count": len(self.context_cache),
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

        @self.mcp.tool()
        async def get_analyzer_status() -> dict[str, Any]:
            """Get analyzer status and statistics."""
            return {
                "status": "success",
                "rust_extensions_available": RUST_EXTENSIONS_AVAILABLE,
                "gemini_ai_available": self.gemini_model is not None,
                "cache_size": len(self.context_cache),
                "supported_languages": list(self.tech_patterns.keys()),
                "agent_stats": self.get_agent_stats(),
            }


# Create global instance for MCP server usage
unified_workspace_analyzer = UnifiedWorkspaceAnalyzer()


# MCP Server entry point
def main() -> None:
    """Run as standalone MCP server."""
    import sys

    try:
        unified_workspace_analyzer.register_mcp_tools()
        logger.info("Starting UnifiedWorkspaceAnalyzer MCP server")
        unified_workspace_analyzer.mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        asyncio.run(unified_workspace_analyzer.cleanup_resources())


if __name__ == "__main__":
    main()
