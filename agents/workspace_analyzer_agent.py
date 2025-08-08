"""Workspace Analyzer Agent Service - Comprehensive project analysis and insights.
Provides detailed project architecture analysis, dependency mapping, and recommendations.
Includes MCP server functionality for direct integration with Claude CLI.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from gterminal.agents.base_agent_service import BaseAgentService
from gterminal.agents.base_agent_service import Job
from gterminal.core.security.security_utils import safe_json_parse
from gterminal.core.security.security_utils import safe_subprocess_run
from mcp.server import FastMCP

"FIXME: replace with rust utilities"
from gterminal.gemini_agents.utils.file_ops import FileOpsMonitor
from gterminal.gemini_agents.utils.file_ops import ParallelFileOps

# Configure logging for MCP integration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkspaceAnalyzerService(BaseAgentService):
    """Comprehensive workspace analysis service.

    Features:
    - Project structure analysis
    - Dependency mapping and vulnerability scanning
    - Code quality metrics
    - Technology stack identification
    - Architecture recommendations
    - Performance bottleneck identification
    - Security assessment
    """

    def __init__(self) -> None:
        super().__init__(
            "workspace_analyzer", "Comprehensive project analysis and architecture insights"
        )

        # Initialize high-performance file operations
        self.file_ops = ParallelFileOps(max_workers=20, batch_size=100)
        self.performance_monitor = FileOpsMonitor()

        # File patterns for different technologies
        self.tech_patterns = {
            "python": ["*.py", "requirements.txt", "pyproject.toml", "setup.py", "Pipfile"],
            "javascript": ["*.js", "*.ts", "package.json", "*.jsx", "*.tsx"],
            "java": ["*.java", "pom.xml", "build.gradle", "*.jar"],
            "csharp": ["*.cs", "*.csproj", "*.sln"],
            "go": ["*.go", "go.mod", "go.sum"],
            "rust": ["*.rs", "Cargo.toml", "Cargo.lock"],
            "php": ["*.php", "composer.json", "composer.lock"],
            "ruby": ["*.rb", "Gemfile", "Gemfile.lock"],
            "cpp": ["*.cpp", "*.c", "*.h", "*.hpp", "CMakeLists.txt", "Makefile"],
            "docker": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
            "kubernetes": ["*.yaml", "*.yml"],
            "terraform": ["*.tf", "*.tfvars"],
        }

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Get required parameters for job type."""
        if job_type in {"analyze_project", "analyze_dependencies", "analyze_architecture"}:
            return ["project_path"]
        return []

    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Execute workspace analysis job implementation."""
        job_type = job.job_type
        parameters = job.parameters

        if job_type == "analyze_project":
            return await self._analyze_project(job, parameters["project_path"])
        if job_type == "analyze_dependencies":
            return await self._analyze_dependencies(job, parameters["project_path"])
        if job_type == "analyze_architecture":
            return await self._analyze_architecture(job, parameters["project_path"])
        if job_type == "analyze_performance":
            return await self._analyze_performance(job, parameters["project_path"])
        msg = f"Unknown job type: {job_type}"
        raise ValueError(msg)

    async def _analyze_project(self, job: Job, project_path: str) -> dict[str, Any]:
        """Comprehensive project analysis."""
        job.update_progress(5.0, f"Starting analysis of {project_path}")

        project_dir = Path(project_path)
        if not project_dir.exists():
            return {"error": f"Project path does not exist: {project_path}"}

        analysis_result: dict[str, Any] = {}

        # Basic project info
        job.update_progress(10.0, "Gathering project information")
        analysis_result["project_info"] = await self._get_project_info(project_dir)

        # Technology stack detection
        job.update_progress(20.0, "Detecting technology stack")
        analysis_result["tech_stack"] = await self._detect_tech_stack(project_dir)

        # File structure analysis
        job.update_progress(30.0, "Analyzing file structure")
        analysis_result["file_structure"] = await self._analyze_file_structure(project_dir)

        # Dependency analysis
        job.update_progress(50.0, "Analyzing dependencies")
        analysis_result["dependencies"] = await self._analyze_project_dependencies(project_dir)

        # Code metrics
        job.update_progress(70.0, "Calculating code metrics")
        analysis_result["code_metrics"] = await self._calculate_code_metrics(project_dir)

        # Security analysis
        job.update_progress(80.0, "Performing security analysis")
        analysis_result["security_analysis"] = await self._analyze_security(project_dir)

        # Architecture recommendations
        job.update_progress(90.0, "Generating recommendations")
        analysis_result["recommendations"] = await self._generate_recommendations(analysis_result)

        job.update_progress(100.0, "Analysis complete")

        return {
            "project_path": project_path,
            "analysis": analysis_result,
            "analyzed_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _get_project_info(self, project_dir: Path) -> dict[str, Any]:
        """Get basic project information."""
        info = {
            "name": project_dir.name,
            "absolute_path": str(project_dir.absolute()),
            "size_mb": self._calculate_directory_size(project_dir),
            "is_git_repo": (project_dir / ".git").exists(),
            "created_date": None,
            "last_modified": None,
        }

        # Git information
        if info["is_git_repo"]:
            git_info = await self._get_git_info(project_dir)
            info.update(git_info)

        # Try to get creation/modification dates
        try:
            stat = project_dir.stat()
            info["last_modified"] = stat.st_mtime
        except Exception:
            pass

        return info

    async def _detect_tech_stack(self, project_dir: Path) -> dict[str, Any]:
        """Detect technology stack using high-performance file operations."""
        detected_techs: dict[str, Any] = {}

        try:
            # Use parallel file operations for faster tech stack detection
            tech_patterns_flat: list[Any] = []
            tech_mapping: dict[str, Any] = {}

            # Create flat list of patterns for parallel search
            for tech, patterns in self.tech_patterns.items():
                for pattern in patterns:
                    tech_patterns_flat.append(pattern)
                    tech_mapping[pattern] = tech

            # Perform parallel file search for all patterns
            search_tasks: list[Any] = []
            for pattern in tech_patterns_flat:
                search_tasks.append(
                    self.file_ops.rust_fs.find_files(project_dir, pattern, max_results=50)
                )

            # Execute searches in parallel
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Process results
            for pattern, files in zip(tech_patterns_flat, search_results, strict=False):
                if isinstance(files, Exception):
                    continue

                tech = tech_mapping[pattern]
                if files:  # Files found for this pattern
                    if tech not in detected_techs:
                        detected_techs[tech] = {
                            "detected": True,
                            "files": [],
                            "count": 0,
                        }

                    # Convert absolute paths to relative
                    relative_files: list[Any] = []
                    for file_path in files:
                        try:
                            rel_path = str(Path(file_path).relative_to(project_dir))
                            relative_files.append(rel_path)
                        except ValueError:
                            relative_files.append(file_path)

                    detected_techs[tech]["files"].extend(relative_files[:10])  # Limit display
                    detected_techs[tech]["count"] += len(files)

        except Exception as e:
            logger.warning(f"High-performance tech stack detection failed, using fallback: {e}")
            # Fallback to original method
            for tech, patterns in self.tech_patterns.items():
                files_found: list[Any] = []
                for pattern in patterns:
                    files = list(project_dir.rglob(pattern))
                    files_found.extend([str(f.relative_to(project_dir)) for f in files])

                if files_found:
                    detected_techs[tech] = {
                        "detected": True,
                        "files": files_found[:10],
                        "count": len(files_found),
                    }

        # Detect frameworks and libraries
        frameworks = await self._detect_frameworks(project_dir, detected_techs)

        return {
            "technologies": detected_techs,
            "frameworks": frameworks,
            "primary_language": self._determine_primary_language(detected_techs),
            "complexity_score": self._calculate_tech_complexity(detected_techs),
        }

    async def _analyze_file_structure(self, project_dir: Path) -> dict[str, Any]:
        """Analyze project file structure using high-performance rust-fs operations."""
        structure = {
            "total_files": 0,
            "total_directories": 0,
            "file_types": {},
            "large_files": [],
            "empty_directories": [],
            "structure_tree": {},
            "performance_stats": {},
        }

        # Use rust-fs for high-performance directory scanning
        try:
            # Perform git-aware scan to respect .gitignore
            scan_result = await self.file_ops.git_aware_file_scan(
                project_dir,
                respect_gitignore=True,
                include_git_metadata=True,
            )

            structure["total_files"] = scan_result["file_count"]
            structure["performance_stats"] = scan_result.get("performance", {})

            # Analyze file types and sizes in parallel
            if scan_result["files"]:
                file_analysis = await self._analyze_files_parallel(
                    scan_result["files"], project_dir
                )
                structure.update(file_analysis)

        except Exception as e:
            logger.warning(
                f"High-performance file scan failed, falling back to standard method: {e}"
            )
            # Fallback to original method
            structure = await self._analyze_file_structure_fallback(project_dir)

        # Generate structure recommendations
        structure["recommendations"] = self._generate_structure_recommendations(structure)

        return structure

    async def _analyze_project_dependencies(self, project_dir: Path) -> dict[str, Any]:
        """Analyze project dependencies."""
        dependencies = {
            "python": await self._analyze_python_dependencies(project_dir),
            "javascript": await self._analyze_js_dependencies(project_dir),
            "java": await self._analyze_java_dependencies(project_dir),
            "security_vulnerabilities": [],
            "outdated_packages": [],
            "dependency_tree_depth": 0,
        }

        # Security vulnerability scanning
        dependencies["security_vulnerabilities"] = await self._scan_vulnerabilities(
            project_dir, dependencies
        )

        return dependencies

    async def _calculate_code_metrics(self, project_dir: Path) -> dict[str, Any]:
        """Calculate various code metrics."""
        metrics = {
            "lines_of_code": 0,
            "lines_of_comments": 0,
            "blank_lines": 0,
            "cyclomatic_complexity": 0,
            "functions": 0,
            "classes": 0,
            "files_analyzed": 0,
            "languages": {},
        }

        # Find code files
        code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".php", ".rb"]
        code_files: list[Any] = []

        for ext in code_extensions:
            code_files.extend(project_dir.rglob(f"*{ext}"))

        metrics["files_analyzed"] = len(code_files)

        # Analyze each file
        for file_path in code_files[:100]:  # Limit to first 100 files
            try:
                content = self.safe_file_read(file_path)
                if content:
                    file_metrics = self._analyze_file_metrics(content, file_path.suffix)

                    # Aggregate metrics
                    metrics["lines_of_code"] += file_metrics["loc"]
                    metrics["lines_of_comments"] += file_metrics["comments"]
                    metrics["blank_lines"] += file_metrics["blank"]
                    metrics["functions"] += file_metrics["functions"]
                    metrics["classes"] += file_metrics["classes"]

                    # Language-specific metrics
                    lang = file_path.suffix
                    if lang not in metrics["languages"]:
                        metrics["languages"][lang] = {"files": 0, "loc": 0}
                    metrics["languages"][lang]["files"] += 1
                    metrics["languages"][lang]["loc"] += file_metrics["loc"]

            except Exception as e:
                self.logger.warning(f"Error analyzing {file_path}: {e}")

        # Calculate ratios
        total_lines = (
            metrics["lines_of_code"] + metrics["lines_of_comments"] + metrics["blank_lines"]
        )
        if total_lines > 0:
            metrics["comment_ratio"] = metrics["lines_of_comments"] / total_lines
            metrics["code_ratio"] = metrics["lines_of_code"] / total_lines

        return metrics

    async def _analyze_security(self, project_dir: Path) -> dict[str, Any]:
        """Perform security analysis."""
        security_analysis = {
            "sensitive_files": [],
            "hardcoded_secrets": [],
            "insecure_dependencies": [],
            "security_score": 8.0,  # Default good score
            "recommendations": [],
        }

        # Check for sensitive files
        sensitive_patterns = [
            "*.key",
            "*.pem",
            "*.p12",
            "*.jks",
            ".env",
            ".env.*",
            "config.json",
            "secrets.*",
        ]

        for pattern in sensitive_patterns:
            files = list(project_dir.rglob(pattern))
            for file_path in files:
                security_analysis["sensitive_files"].append(
                    {
                        "path": str(file_path.relative_to(project_dir)),
                        "type": "sensitive_file",
                        "risk": "high" if file_path.suffix in [".key", ".pem"] else "medium",
                    },
                )

        # Scan for hardcoded secrets in code files
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
        ]

        code_files = list(project_dir.rglob("*.py")) + list(project_dir.rglob("*.js"))
        for file_path in code_files[:50]:  # Limit scan
            try:
                content = self.safe_file_read(file_path)
                if content:
                    secrets_found = self._scan_file_for_secrets(content, secret_patterns)
                    for secret in secrets_found:
                        security_analysis["hardcoded_secrets"].append(
                            {
                                "file": str(file_path.relative_to(project_dir)),
                                "line": secret["line"],
                                "type": secret["type"],
                                "risk": "high",
                            },
                        )
            except Exception:
                pass

        # Calculate security score
        issues_count = (
            len(security_analysis["sensitive_files"])
            + len(security_analysis["hardcoded_secrets"])
            + len(security_analysis["insecure_dependencies"])
        )

        security_analysis["security_score"] = max(0, 10 - issues_count * 0.5)

        return security_analysis

    async def _generate_recommendations(self, analysis_result: dict[str, Any]) -> dict[str, Any]:
        """Generate architecture and improvement recommendations."""
        recommendations = {
            "architecture": [],
            "performance": [],
            "security": [],
            "maintainability": [],
            "priority": "medium",
        }

        # Architecture recommendations
        tech_stack = analysis_result.get("tech_stack", {})
        if len(tech_stack.get("technologies", {})) > 5:
            recommendations["architecture"].append(
                {
                    "type": "complexity",
                    "message": "Consider consolidating technology stack to reduce complexity",
                    "priority": "medium",
                },
            )

        # Security recommendations
        security_analysis = analysis_result.get("security_analysis", {})
        if security_analysis.get("security_score", 10) < 7:
            recommendations["security"].append(
                {
                    "type": "security_score",
                    "message": "Security score below recommended threshold. Review security findings.",
                    "priority": "high",
                },
            )

        # Performance recommendations
        file_structure = analysis_result.get("file_structure", {})
        if len(file_structure.get("large_files", [])) > 0:
            recommendations["performance"].append(
                {
                    "type": "large_files",
                    "message": "Large files detected. Consider optimization or splitting.",
                    "priority": "medium",
                },
            )

        # Code quality recommendations
        code_metrics = analysis_result.get("code_metrics", {})
        if code_metrics.get("comment_ratio", 0) < 0.1:
            recommendations["maintainability"].append(
                {
                    "type": "documentation",
                    "message": "Low comment ratio. Consider adding more documentation.",
                    "priority": "low",
                },
            )

        return recommendations

    async def _analyze_files_parallel(self, files: list[str], project_dir: Path) -> dict[str, Any]:
        """Analyze files in parallel for types, sizes, etc."""
        file_analysis = {"file_types": {}, "large_files": [], "total_directories": 0}

        # Get file stats in parallel batches
        batch_size = 50
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]

            # Get stats for batch
            stats_tasks: list[Any] = []
            for file_path in batch:
                stats_tasks.append(self.file_ops.rust_fs.get_file_stats(file_path))

            try:
                stats_results = await asyncio.gather(*stats_tasks, return_exceptions=True)

                for file_path, stats in zip(batch, stats_results, strict=False):
                    if isinstance(stats, Exception):
                        continue

                    # Count file types
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext:
                        file_analysis["file_types"][file_ext] = (
                            file_analysis["file_types"].get(file_ext, 0) + 1
                        )

                    # Check for large files (>10MB)
                    file_size = stats.get("size", 0)
                    if file_size > 10 * 1024 * 1024:  # 10MB
                        try:
                            rel_path = str(Path(file_path).relative_to(project_dir))
                        except ValueError:
                            rel_path = file_path

                        file_analysis["large_files"].append(
                            {
                                "path": rel_path,
                                "size_mb": round(file_size / (1024 * 1024), 2),
                            },
                        )

            except Exception as e:
                logger.warning(f"Error analyzing file batch: {e}")
                continue

        return file_analysis

    async def _analyze_file_structure_fallback(self, project_dir: Path) -> dict[str, Any]:
        """Fallback file structure analysis using standard Python operations."""
        structure = {
            "total_files": 0,
            "total_directories": 0,
            "file_types": {},
            "large_files": [],
            "empty_directories": [],
            "structure_tree": {},
            "performance_stats": {"used_rust_fs": False},
        }

        # Walk directory tree
        for root, dirs, files in os.walk(project_dir):
            root_path = Path(root)
            structure["total_directories"] += len(dirs)
            structure["total_files"] += len(files)

            # Analyze files
            for file in files:
                file_path = root_path / file
                file_ext = file_path.suffix.lower()

                # Count file types
                if file_ext:
                    structure["file_types"][file_ext] = structure["file_types"].get(file_ext, 0) + 1

                # Check for large files (>10MB)
                try:
                    file_size = file_path.stat().st_size
                    if file_size > 10 * 1024 * 1024:  # 10MB
                        structure["large_files"].append(
                            {
                                "path": str(file_path.relative_to(project_dir)),
                                "size_mb": round(file_size / (1024 * 1024), 2),
                            },
                        )
                except Exception:
                    pass

            # Check for empty directories
            if not files and not dirs:
                structure["empty_directories"].append(str(root_path.relative_to(project_dir)))

        return structure

    def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate directory size in MB."""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return round(total_size / (1024 * 1024), 2)

    async def _get_git_info(self, project_dir: Path) -> dict[str, Any]:
        """Get Git repository information."""
        git_info: dict[str, Any] = {}

        # Change to project directory for git commands
        original_cwd = os.getcwd()
        try:
            os.chdir(project_dir)

            # Get branch info
            branch_result = safe_subprocess_run(["git", "branch", "--show-current"])
            if branch_result.returncode == 0:
                git_info["current_branch"] = branch_result.stdout.strip()

            # Get commit count
            commit_result = safe_subprocess_run(["git", "rev-list", "--count", "HEAD"])
            if commit_result.returncode == 0:
                git_info["total_commits"] = int(commit_result.stdout.strip())

            # Get remote info
            remote_result = safe_subprocess_run(["git", "remote", "-v"])
            if remote_result.returncode == 0:
                git_info["remotes"] = remote_result.stdout.strip().split("\n")

        except Exception as e:
            self.logger.warning(f"Error getting git info: {e}")
        finally:
            os.chdir(original_cwd)

        return git_info

    async def _detect_frameworks(
        self, project_dir: Path, detected_techs: dict[str, Any]
    ) -> dict[str, Any]:
        """Detect frameworks based on detected technologies."""
        frameworks: dict[str, Any] = {}

        # Python frameworks
        if "python" in detected_techs:
            requirements_file = project_dir / "requirements.txt"
            if requirements_file.exists():
                content = self.safe_file_read(requirements_file)
                if content:
                    if "django" in content.lower():
                        frameworks["django"] = {"type": "web_framework", "language": "python"}
                    if "flask" in content.lower():
                        frameworks["flask"] = {"type": "web_framework", "language": "python"}
                    if "fastapi" in content.lower():
                        frameworks["fastapi"] = {"type": "web_framework", "language": "python"}

        # JavaScript frameworks
        if "javascript" in detected_techs:
            package_json = project_dir / "package.json"
            if package_json.exists():
                content = self.safe_file_read(package_json)
                if content:
                    package_data = safe_json_parse(content)
                    if package_data:
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
                            frameworks["vue"] = {
                                "type": "frontend_framework",
                                "language": "javascript",
                            }
                        if "angular" in deps:
                            frameworks["angular"] = {
                                "type": "frontend_framework",
                                "language": "javascript",
                            }
                        if "express" in deps:
                            frameworks["express"] = {
                                "type": "backend_framework",
                                "language": "javascript",
                            }

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
        if detected_count <= 5:
            return 6  # Moderate
        return 9  # Complex

    def _generate_structure_recommendations(self, structure: dict[str, Any]) -> list[str]:
        """Generate file structure recommendations."""
        recommendations: list[Any] = []

        if len(structure["large_files"]) > 0:
            recommendations.append("Consider splitting large files for better maintainability")

        if len(structure["empty_directories"]) > 5:
            recommendations.append("Clean up empty directories to reduce clutter")

        if structure["total_files"] > 1000:
            recommendations.append("Large project detected. Consider modularization")

        return recommendations

    async def _analyze_python_dependencies(self, project_dir: Path) -> dict[str, Any]:
        """Analyze Python dependencies."""
        deps_info = {"packages": [], "outdated": [], "vulnerabilities": []}

        requirements_file = project_dir / "requirements.txt"
        if requirements_file.exists():
            content = self.safe_file_read(requirements_file)
            if content:
                packages = [line.strip() for line in content.split("\n") if line.strip()]
                deps_info["packages"] = packages[:20]  # Limit to first 20

        return deps_info

    async def _analyze_js_dependencies(self, project_dir: Path) -> dict[str, Any]:
        """Analyze JavaScript dependencies."""
        deps_info = {"packages": [], "outdated": [], "vulnerabilities": []}

        package_json = project_dir / "package.json"
        if package_json.exists():
            content = self.safe_file_read(package_json)
            if content:
                package_data = safe_json_parse(content)
                if package_data:
                    all_deps = {
                        **package_data.get("dependencies", {}),
                        **package_data.get("devDependencies", {}),
                    }
                    deps_info["packages"] = [f"{k}@{v}" for k, v in all_deps.items()]

        return deps_info

    async def _analyze_java_dependencies(self, project_dir: Path) -> dict[str, Any]:
        """Analyze Java dependencies."""
        deps_info = {"packages": [], "outdated": [], "vulnerabilities": []}

        # Check for Maven
        pom_xml = project_dir / "pom.xml"
        if pom_xml.exists():
            deps_info["build_system"] = "maven"

        # Check for Gradle
        build_gradle = project_dir / "build.gradle"
        if build_gradle.exists():
            deps_info["build_system"] = "gradle"

        return deps_info

    async def _scan_vulnerabilities(
        self, project_dir: Path, dependencies: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Scan for security vulnerabilities in dependencies."""
        vulnerabilities: list[Any] = []

        # This would integrate with security scanning tools
        # For now, return empty list
        return vulnerabilities

    def _analyze_file_metrics(self, content: str, file_extension: str) -> dict[str, int]:
        """Analyze metrics for a single file."""
        lines = content.split("\n")

        metrics = {"loc": 0, "comments": 0, "blank": 0, "functions": 0, "classes": 0}

        for line in lines:
            stripped = line.strip()
            if not stripped:
                metrics["blank"] += 1
            elif stripped.startswith(("#", "//")):
                metrics["comments"] += 1
            else:
                metrics["loc"] += 1

        # Count functions and classes based on file type
        if file_extension == ".py":
            metrics["functions"] = len([l for l in lines if l.strip().startswith("def ")])
            metrics["classes"] = len([l for l in lines if l.strip().startswith("class ")])
        elif file_extension in [".js", ".ts"]:
            metrics["functions"] = len([l for l in lines if "function" in l])
            metrics["classes"] = len([l for l in lines if l.strip().startswith("class ")])

        return metrics

    def _scan_file_for_secrets(self, content: str, patterns: list[str]) -> list[dict[str, Any]]:
        """Scan file content for potential secrets."""
        import re

        secrets: list[Any] = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    secrets.append(
                        {"line": line_num, "type": "potential_secret", "pattern": pattern}
                    )

        return secrets

    async def _analyze_dependencies(self, job: Job, project_path: str) -> dict[str, Any]:
        """Focused dependency analysis."""
        job.update_progress(10.0, "Starting dependency analysis")

        project_dir = Path(project_path)
        if not project_dir.exists():
            return {"error": f"Project path does not exist: {project_path}"}

        dependencies = await self._analyze_project_dependencies(project_dir)

        job.update_progress(100.0, "Dependency analysis complete")

        return {
            "project_path": project_path,
            "dependencies": dependencies,
            "analyzed_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _analyze_architecture(self, job: Job, project_path: str) -> dict[str, Any]:
        """Focused architecture analysis."""
        job.update_progress(10.0, "Starting architecture analysis")

        project_dir = Path(project_path)
        if not project_dir.exists():
            return {"error": f"Project path does not exist: {project_path}"}

        # Get architecture-specific analysis
        job.update_progress(30.0, "Analyzing technology stack")
        tech_stack = await self._detect_tech_stack(project_dir)

        job.update_progress(60.0, "Analyzing file structure")
        file_structure = await self._analyze_file_structure(project_dir)

        job.update_progress(90.0, "Generating architecture recommendations")
        recommendations = await self._generate_architecture_recommendations(
            tech_stack, file_structure
        )

        job.update_progress(100.0, "Architecture analysis complete")

        return {
            "project_path": project_path,
            "tech_stack": tech_stack,
            "file_structure": file_structure,
            "recommendations": recommendations,
            "analyzed_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _analyze_performance(self, job: Job, project_path: str) -> dict[str, Any]:
        """Focused performance analysis."""
        job.update_progress(10.0, "Starting performance analysis")

        project_dir = Path(project_path)
        if not project_dir.exists():
            return {"error": f"Project path does not exist: {project_path}"}

        performance_analysis = {
            "bottlenecks": [],
            "optimizations": [],
            "resource_usage": {},
            "recommendations": [],
        }

        job.update_progress(50.0, "Analyzing code for performance issues")

        # Analyze code files for performance patterns
        code_files = list(project_dir.rglob("*.py")) + list(project_dir.rglob("*.js"))

        for file_path in code_files[:50]:  # Limit analysis
            content = self.safe_file_read(file_path)
            if content:
                bottlenecks = self._identify_performance_bottlenecks(content, str(file_path))
                performance_analysis["bottlenecks"].extend(bottlenecks)

        job.update_progress(80.0, "Generating performance recommendations")

        # Generate recommendations
        performance_analysis["recommendations"] = self._generate_performance_recommendations(
            performance_analysis["bottlenecks"],
        )

        job.update_progress(100.0, "Performance analysis complete")

        return {
            "project_path": project_path,
            "performance_analysis": performance_analysis,
            "analyzed_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_architecture_recommendations(
        self,
        tech_stack: dict[str, Any],
        file_structure: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate architecture-specific recommendations."""
        recommendations: list[Any] = []

        # Technology complexity recommendations
        if tech_stack.get("complexity_score", 0) > 7:
            recommendations.append(
                {
                    "category": "architecture",
                    "priority": "medium",
                    "title": "Reduce Technology Complexity",
                    "description": "Multiple technologies detected. Consider consolidating to improve maintainability.",
                },
            )

        # File structure recommendations
        if len(file_structure.get("large_files", [])) > 5:
            recommendations.append(
                {
                    "category": "structure",
                    "priority": "high",
                    "title": "Large Files Detected",
                    "description": "Multiple large files found. Consider breaking them into smaller modules.",
                },
            )

        return recommendations

    def _identify_performance_bottlenecks(
        self, content: str, file_path: str
    ) -> list[dict[str, Any]]:
        """Identify potential performance bottlenecks in code."""
        import re

        bottlenecks: list[Any] = []
        lines = content.split("\n")

        # Performance anti-patterns
        patterns = [
            (r"for\s+\w+\s+in\s+range\(len\(", "Use enumerate() instead of range(len())"),
            (r"while\s+True:", "Potential infinite loop detected"),
            (r"time\.sleep\(", "Blocking sleep call found"),
            (r"\.join\(\)\s*in\s+loop", "String concatenation in loop"),
        ]

        for line_num, line in enumerate(lines, 1):
            for pattern, message in patterns:
                if re.search(pattern, line):
                    bottlenecks.append(
                        {
                            "file": file_path,
                            "line": line_num,
                            "issue": message,
                            "code": line.strip(),
                            "severity": "medium",
                        },
                    )

        return bottlenecks

    def _generate_performance_recommendations(self, bottlenecks: list[dict[str, Any]]) -> list[str]:
        """Generate performance improvement recommendations."""
        recommendations: list[Any] = []

        if len(bottlenecks) > 0:
            recommendations.append("Review identified performance bottlenecks")

        if len(bottlenecks) > 10:
            recommendations.append("Consider performance profiling for detailed analysis")

        return recommendations

    def register_tools(self) -> None:
        """Register MCP tools for workspace analyzer."""

        @self.mcp.tool()
        async def analyze_project(project_path: str = ".") -> dict[str, Any]:
            """Comprehensive project analysis."""
            if not self.validate_job_parameters("analyze_project", {"project_path": project_path}):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("analyze_project", {"project_path": project_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def analyze_dependencies(project_path: str = ".") -> dict[str, Any]:
            """Analyze project dependencies."""
            if not self.validate_job_parameters(
                "analyze_dependencies", {"project_path": project_path}
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("analyze_dependencies", {"project_path": project_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def analyze_architecture(project_path: str = ".") -> dict[str, Any]:
            """Analyze project architecture."""
            if not self.validate_job_parameters(
                "analyze_architecture", {"project_path": project_path}
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("analyze_architecture", {"project_path": project_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def get_analyzer_stats() -> dict[str, Any]:
            """Get analyzer statistics."""
            return self.create_success_response(
                {
                    "agent_stats": self.get_agent_stats(),
                    "supported_technologies": list(self.tech_patterns.keys()),
                },
            )


# Create global instance
workspace_analyzer_service = WorkspaceAnalyzerService()


# MCP Server Integration
# Initialize MCP server for standalone usage
mcp = FastMCP("workspace-analyzer")


@mcp.tool()
async def analyze_workspace(
    project_path: str,
    analysis_depth: int = 3,
    include_dependencies: bool = True,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Analyze a project workspace for structure, architecture, and quality.

    Args:
        project_path: Path to the project directory
        analysis_depth: Maximum directory depth to analyze (not used in current implementation)
        include_dependencies: Whether to analyze project dependencies
        include_tests: Whether to include test analysis (not used in current implementation)

    Returns:
        Comprehensive workspace analysis results

    """
    try:
        # Create job using the agent service
        job_id = workspace_analyzer_service.create_job(
            "analyze_project",
            {
                "project_path": project_path,
                "analysis_depth": analysis_depth,
                "include_dependencies": include_dependencies,
                "include_tests": include_tests,
            },
        )

        return await workspace_analyzer_service.execute_job_async(job_id)
    except Exception as e:
        logger.exception(f"Workspace analysis failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def analyze_dependencies(
    project_path: str,
    check_vulnerabilities: bool = True,
    check_licenses: bool = False,
) -> dict[str, Any]:
    """Analyze project dependencies for vulnerabilities and issues.

    Args:
        project_path: Path to the project directory
        check_vulnerabilities: Whether to check for security vulnerabilities
        check_licenses: Whether to check license compatibility (not used in current implementation)

    Returns:
        Dependency analysis results

    """
    try:
        # Create job using the agent service
        job_id = workspace_analyzer_service.create_job(
            "analyze_dependencies",
            {
                "project_path": project_path,
                "check_vulnerabilities": check_vulnerabilities,
                "check_licenses": check_licenses,
            },
        )

        return await workspace_analyzer_service.execute_job_async(job_id)
    except Exception as e:
        logger.exception(f"Dependency analysis failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def analyze_architecture(
    project_path: str,
    generate_diagrams: bool = True,
    suggest_improvements: bool = True,
) -> dict[str, Any]:
    """Analyze project architecture and provide recommendations.

    Args:
        project_path: Path to the project directory
        generate_diagrams: Whether to generate architecture diagrams (not used in current implementation)
        suggest_improvements: Whether to suggest architectural improvements

    Returns:
        Architecture analysis and recommendations

    """
    try:
        # Create job using the agent service
        job_id = workspace_analyzer_service.create_job(
            "analyze_architecture",
            {
                "project_path": project_path,
                "generate_diagrams": generate_diagrams,
                "suggest_improvements": suggest_improvements,
            },
        )

        return await workspace_analyzer_service.execute_job_async(job_id)
    except Exception as e:
        logger.exception(f"Architecture analysis failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def analyze_code_metrics(
    project_path: str,
    include_complexity: bool = True,
    include_coverage: bool = False,
) -> dict[str, Any]:
    """Calculate code metrics for a project.

    Args:
        project_path: Path to the project directory
        include_complexity: Whether to calculate complexity metrics
        include_coverage: Whether to analyze test coverage (not implemented)

    Returns:
        Code metrics and quality indicators

    """
    try:
        # Use the project analysis which includes code metrics
        job_id = workspace_analyzer_service.create_job(
            "analyze_project",
            {
                "project_path": project_path,
                "include_complexity": include_complexity,
                "include_coverage": include_coverage,
            },
        )

        result = await workspace_analyzer_service.execute_job_async(job_id)

        # Extract code metrics from the full analysis
        if "analysis" in result and "code_metrics" in result["analysis"]:
            return {
                "status": "success",
                "code_metrics": result["analysis"]["code_metrics"],
                "project_path": project_path,
            }

        return result
    except Exception as e:
        logger.exception(f"Metrics calculation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def find_security_issues(
    project_path: str,
    scan_configs: bool = True,
    scan_secrets: bool = True,
) -> dict[str, Any]:
    """Find security issues in a project.

    Args:
        project_path: Path to the project directory
        scan_configs: Whether to scan configuration files
        scan_secrets: Whether to scan for hardcoded secrets

    Returns:
        Security issues and vulnerabilities found

    """
    try:
        # Use the project analysis which includes security analysis
        job_id = workspace_analyzer_service.create_job(
            "analyze_project",
            {
                "project_path": project_path,
                "scan_configs": scan_configs,
                "scan_secrets": scan_secrets,
            },
        )

        result = await workspace_analyzer_service.execute_job_async(job_id)

        # Extract security analysis from the full analysis
        if "analysis" in result and "security_analysis" in result["analysis"]:
            return {
                "status": "success",
                "security_analysis": result["analysis"]["security_analysis"],
                "project_path": project_path,
            }

        return result
    except Exception as e:
        logger.exception(f"Security scan failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def get_analysis_status(job_id: str) -> dict[str, Any]:
    """Get the status of an ongoing analysis job.

    Args:
        job_id: The job ID to check

    Returns:
        Job status and progress information

    """
    try:
        return workspace_analyzer_service.get_job_status(job_id)
    except Exception as e:
        logger.exception(f"Failed to get job status: {e}")
        return {"status": "error", "error": str(e)}


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
