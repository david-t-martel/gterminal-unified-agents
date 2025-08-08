"""Documentation Generator Agent Service - Automated documentation generation and maintenance.
Creates comprehensive documentation including API docs, README files, and code documentation.
Includes MCP server functionality for direct integration with Claude CLI.
Features high-performance web fetching with integrated Rust extensions.
"""

import json
import logging
from pathlib import Path
import re
from typing import Any

from gterminal.agents.base_agent_service import BaseAgentService
from gterminal.agents.base_agent_service import Job
from gterminal.core.security.security_utils import safe_json_parse
from gterminal.gemini_agents.utils.rust_fetch_client import RustFetchClient
from mcp.server import FastMCP

# Configure logging for MCP integration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentationGeneratorService(BaseAgentService):
    """Comprehensive documentation generation service.

    Features:
    - API documentation generation from code
    - README file creation and updates
    - Code documentation (docstrings) generation
    - Architecture documentation
    - User guides and tutorials
    - Changelog generation
    - Multi-format output (Markdown, HTML, PDF)
    """

    def __init__(self) -> None:
        super().__init__(
            "documentation_generator",
            "Automated documentation generation and maintenance",
        )

        # Documentation templates
        self.templates = {
            "readme": self._get_readme_template(),
            "api_docs": self._get_api_docs_template(),
            "contributing": self._get_contributing_template(),
            "changelog": self._get_changelog_template(),
        }

        # Supported documentation formats
        self.formats = ["markdown", "html", "rst", "pdf"]

        # High-performance HTTP client for fetching external documentation
        self._fetch_client: RustFetchClient = None

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Get required parameters for job type."""
        if job_type in {
            "generate_api_docs",
            "generate_readme",
            "generate_code_docs",
            "update_documentation",
        }:
            return ["project_path"]
        return []

    async def _get_fetch_client(self) -> RustFetchClient:
        """Get or create the high-performance fetch client."""
        if self._fetch_client is None:
            self._fetch_client = RustFetchClient(
                max_concurrent=5,  # Conservative for documentation sites
                timeout=30,
                cache_ttl=7200,  # Cache docs for 2 hours
                user_agent="my-fullstack-agent-docs/1.0",
            )
        return self._fetch_client

    async def _fetch_external_documentation(
        self, urls: list[str], base_url: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch external documentation URLs with high performance.

        Args:
            urls: List of URLs or paths to fetch
            base_url: Base URL to resolve relative paths

        Returns:
            List of fetched documentation responses

        """
        # Build full URLs from paths if base_url provided
        full_urls: list[Any] = []
        for url in urls:
            if url.startswith(("http://", "https://")):
                full_urls.append(url)
            elif base_url:
                from urllib.parse import urljoin

                full_urls.append(urljoin(base_url, url))
            else:
                full_urls.append(url)

        fetch_client = await self._get_fetch_client()

        try:
            async with fetch_client:
                results = await fetch_client.batch_fetch(
                    urls=full_urls,
                    text_only=True,  # Documentation is usually text
                    fail_fast=False,  # Continue even if some URLs fail
                )

                logger.info(
                    f"Successfully fetched {len([r for r in results if r.get('status', 0) < 400])} "
                    f"out of {len(full_urls)} documentation URLs",
                )

                return results

        except Exception as e:
            logger.exception(f"Failed to fetch external documentation: {e}")
            return []

    async def _fetch_dependency_documentation(
        self,
        dependencies: list[str],
        package_registry: str = "pypi",
    ) -> dict[str, dict[str, Any]]:
        """Fetch documentation for project dependencies.

        Args:
            dependencies: List of dependency names
            package_registry: Package registry (pypi, npm, etc.)

        Returns:
            Dictionary mapping dependency names to their documentation

        """
        doc_urls: list[Any] = []

        if package_registry == "pypi":
            # PyPI API URLs for package information
            doc_urls = [f"https://pypi.org/pypi/{dep}/json" for dep in dependencies]
        elif package_registry == "npm":
            # NPM registry URLs
            doc_urls = [f"https://registry.npmjs.org/{dep}" for dep in dependencies]
        else:
            logger.warning(f"Unsupported package registry: {package_registry}")
            return {}

        fetch_client = await self._get_fetch_client()

        try:
            async with fetch_client:
                results = await fetch_client.batch_fetch(
                    urls=doc_urls,
                    text_only=False,  # JSON responses
                    fail_fast=False,
                )

                dependency_docs: dict[str, Any] = {}
                for i, result in enumerate(results):
                    if result.get("status", 0) == 200:
                        try:
                            data = json.loads(result.get("content", "{}"))
                            dependency_docs[dependencies[i]] = {
                                "info": data.get("info", {}),
                                "urls": data.get("urls", {}),
                                "home_page": data.get("info", {}).get("home_page", ""),
                                "docs_url": data.get("info", {}).get("docs_url", ""),
                                "summary": data.get("info", {}).get("summary", ""),
                            }
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse JSON for dependency: {dependencies[i]}"
                            )
                    else:
                        logger.warning(f"Failed to fetch info for dependency: {dependencies[i]}")

                return dependency_docs

        except Exception as e:
            logger.exception(f"Failed to fetch dependency documentation: {e}")
            return {}

    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Execute documentation generation job implementation."""
        job_type = job.job_type
        parameters = job.parameters

        if job_type == "generate_api_docs":
            return await self._generate_api_docs(job, parameters["project_path"])
        if job_type == "generate_readme":
            return await self._generate_readme(job, parameters["project_path"])
        if job_type == "generate_code_docs":
            return await self._generate_code_docs(job, parameters["project_path"])
        if job_type == "update_documentation":
            return await self._update_documentation(job, parameters["project_path"])
        if job_type == "generate_user_guide":
            return await self._generate_user_guide(job, parameters["project_path"])
        msg = f"Unknown job type: {job_type}"
        raise ValueError(msg)

    async def _generate_api_docs(self, job: Job, project_path: str) -> dict[str, Any]:
        """Generate API documentation from code."""
        job.update_progress(10.0, f"Starting API documentation generation for {project_path}")

        project_dir = Path(project_path)
        if not project_dir.exists():
            return {"error": f"Project path does not exist: {project_path}"}

        api_docs = {
            "endpoints": [],
            "models": [],
            "authentication": {},
            "examples": [],
            "generated_files": [],
        }

        # Discover Python API files
        job.update_progress(20.0, "Discovering API files")
        api_files = self._find_api_files(project_dir)

        # Extract API endpoints
        job.update_progress(40.0, "Extracting API endpoints")
        for file_path in api_files:
            endpoints = await self._extract_api_endpoints(file_path)
            api_docs["endpoints"].extend(endpoints)

        # Extract data models
        job.update_progress(60.0, "Extracting data models")
        model_files = self._find_model_files(project_dir)
        for file_path in model_files:
            models = await self._extract_data_models(file_path)
            api_docs["models"].extend(models)

        # Generate OpenAPI specification
        job.update_progress(80.0, "Generating OpenAPI specification")
        openapi_spec = await self._generate_openapi_spec(api_docs, project_dir)

        # Generate documentation files
        job.update_progress(90.0, "Writing documentation files")
        docs_dir = project_dir / "docs" / "api"
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Write API documentation
        api_md_path = docs_dir / "api.md"
        api_content = await self._generate_api_markdown(api_docs)
        if self.safe_file_write(api_md_path, api_content):
            api_docs["generated_files"].append(str(api_md_path))

        # Write OpenAPI spec
        if openapi_spec:
            openapi_path = docs_dir / "openapi.json"
            if self.safe_file_write(openapi_path, json.dumps(openapi_spec, indent=2)):
                api_docs["generated_files"].append(str(openapi_path))

        job.update_progress(100.0, "API documentation generation complete")

        return {
            "project_path": project_path,
            "api_docs": api_docs,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_readme(self, job: Job, project_path: str) -> dict[str, Any]:
        """Generate or update README file."""
        job.update_progress(10.0, f"Starting README generation for {project_path}")

        project_dir = Path(project_path)
        if not project_dir.exists():
            return {"error": f"Project path does not exist: {project_path}"}

        # Analyze project to gather information
        job.update_progress(30.0, "Analyzing project structure")
        project_info = await self._analyze_project_for_readme(project_dir)

        # Generate README content
        job.update_progress(60.0, "Generating README content")
        readme_content = await self._generate_readme_content(project_info)

        # Write README file
        job.update_progress(90.0, "Writing README file")
        readme_path = project_dir / "README.md"

        # Backup existing README if it exists
        if readme_path.exists():
            backup_path = self.create_backup(readme_path)
            if backup_path:
                job.add_log(f"Created backup: {backup_path}")

        success = self.safe_file_write(readme_path, readme_content)

        job.update_progress(100.0, "README generation complete")

        return {
            "project_path": project_path,
            "readme_path": str(readme_path),
            "success": success,
            "content_sections": self._get_readme_sections(),
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_code_docs(self, job: Job, project_path: str) -> dict[str, Any]:
        """Generate code documentation (docstrings)."""
        job.update_progress(10.0, f"Starting code documentation generation for {project_path}")

        project_dir = Path(project_path)
        if not project_dir.exists():
            return {"error": f"Project path does not exist: {project_path}"}

        docs_generated = {
            "files_processed": 0,
            "functions_documented": 0,
            "classes_documented": 0,
            "modules_documented": 0,
            "updated_files": [],
        }

        # Find Python files
        job.update_progress(20.0, "Finding Python files")
        python_files = list(project_dir.rglob("*.py"))
        exclude_patterns = ["__pycache__", ".git", "venv", "env", "node_modules"]
        python_files = [
            f for f in python_files if not any(pattern in str(f) for pattern in exclude_patterns)
        ]

        # Process files
        total_files = len(python_files)
        for i, file_path in enumerate(python_files[:50]):  # Limit to first 50 files
            progress = 20.0 + (60.0 * i / min(total_files, 50))
            job.update_progress(progress, f"Processing {file_path.name}")

            try:
                result = await self._add_docstrings_to_file(file_path)
                if result["updated"]:
                    docs_generated["updated_files"].append(str(file_path))
                    docs_generated["functions_documented"] += result["functions"]
                    docs_generated["classes_documented"] += result["classes"]

                docs_generated["files_processed"] += 1

            except Exception as e:
                job.add_log(f"Error processing {file_path}: {e!s}")

        # Generate module documentation
        job.update_progress(85.0, "Generating module documentation")
        module_docs = await self._generate_module_docs(project_dir)
        docs_generated["modules_documented"] = len(module_docs)

        job.update_progress(100.0, "Code documentation generation complete")

        return {
            "project_path": project_path,
            "documentation_stats": docs_generated,
            "module_docs": module_docs,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _update_documentation(self, job: Job, project_path: str) -> dict[str, Any]:
        """Update existing documentation."""
        job.update_progress(10.0, f"Starting documentation update for {project_path}")

        project_dir = Path(project_path)
        if not project_dir.exists():
            return {"error": f"Project path does not exist: {project_path}"}

        update_results = {"files_updated": [], "files_created": [], "errors": []}

        # Update README if exists
        job.update_progress(30.0, "Updating README")
        readme_path = project_dir / "README.md"
        if readme_path.exists():
            try:
                updated_readme = await self._update_existing_readme(readme_path, project_dir)
                if updated_readme:
                    update_results["files_updated"].append(str(readme_path))
            except Exception as e:
                update_results["errors"].append(f"README update failed: {e!s}")

        # Update CHANGELOG
        job.update_progress(50.0, "Updating CHANGELOG")
        changelog_path = project_dir / "CHANGELOG.md"
        try:
            changelog_updated = await self._update_changelog(project_dir)
            if changelog_updated:
                if changelog_path.exists():
                    update_results["files_updated"].append(str(changelog_path))
                else:
                    update_results["files_created"].append(str(changelog_path))
        except Exception as e:
            update_results["errors"].append(f"CHANGELOG update failed: {e!s}")

        # Update API documentation
        job.update_progress(70.0, "Updating API documentation")
        try:
            api_result = await self._generate_api_docs(job, project_path)
            if "api_docs" in api_result:
                update_results["files_updated"].extend(
                    api_result["api_docs"].get("generated_files", [])
                )
        except Exception as e:
            update_results["errors"].append(f"API docs update failed: {e!s}")

        job.update_progress(100.0, "Documentation update complete")

        return {
            "project_path": project_path,
            "update_results": update_results,
            "updated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_user_guide(self, job: Job, project_path: str) -> dict[str, Any]:
        """Generate user guide documentation."""
        job.update_progress(10.0, f"Starting user guide generation for {project_path}")

        project_dir = Path(project_path)
        if not project_dir.exists():
            return {"error": f"Project path does not exist: {project_path}"}

        # Analyze project for user guide content
        job.update_progress(30.0, "Analyzing project for user guide")
        project_analysis = await self._analyze_project_for_guide(project_dir)

        # Generate user guide content
        job.update_progress(60.0, "Generating user guide content")
        guide_content = await self._generate_guide_content(project_analysis)

        # Write user guide
        job.update_progress(90.0, "Writing user guide")
        docs_dir = project_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        guide_path = docs_dir / "user-guide.md"

        success = self.safe_file_write(guide_path, guide_content)

        job.update_progress(100.0, "User guide generation complete")

        return {
            "project_path": project_path,
            "guide_path": str(guide_path),
            "success": success,
            "sections_generated": len(guide_content.split("##")) - 1,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    def _find_api_files(self, project_dir: Path) -> list[Path]:
        """Find API-related files in the project."""
        api_files: list[Any] = []
        common_api_patterns = [
            "**/api/**/*.py",
            "**/routes/**/*.py",
            "**/endpoints/**/*.py",
            "**/views/**/*.py",
            "**/*api*.py",
            "**/*router*.py",
            "**/*endpoint*.py",
        ]

        for pattern in common_api_patterns:
            api_files.extend(project_dir.glob(pattern))

        return list(set(api_files))  # Remove duplicates

    def _find_model_files(self, project_dir: Path) -> list[Path]:
        """Find data model files in the project."""
        model_files: list[Any] = []
        model_patterns = [
            "**/models/**/*.py",
            "**/schemas/**/*.py",
            "**/*model*.py",
            "**/*schema*.py",
        ]

        for pattern in model_patterns:
            model_files.extend(project_dir.glob(pattern))

        return list(set(model_files))

    async def _extract_api_endpoints(self, file_path: Path) -> list[dict[str, Any]]:
        """Extract API endpoints from a Python file."""
        endpoints: list[Any] = []

        content = self.safe_file_read(file_path)
        if not content:
            return endpoints

        # Look for FastAPI/Flask route decorators
        lines = content.split("\n")
        current_endpoint = None

        for i, line in enumerate(lines):
            stripped_line = line.strip()

            # FastAPI routes
            if stripped_line.startswith(("@app.", "@router.")):
                method_match = re.search(
                    r"@(?:app|router)\.(get|post|put|delete|patch)", stripped_line
                )
                if method_match:
                    method = method_match.group(1).upper()
                    path_match = re.search(r'["\']([^"\']+)["\']', stripped_line)
                    path = path_match.group(1) if path_match else "/"

                    # Get function name from next non-empty line
                    func_name = ""
                    for j in range(i + 1, min(i + 5, len(lines))):
                        func_line = lines[j].strip()
                        if func_line and func_line.startswith("def "):
                            func_name = func_line.split("(")[0].replace("def ", "")
                            break

                    current_endpoint = {
                        "method": method,
                        "path": path,
                        "function": func_name,
                        "file": str(file_path),
                        "line": i + 1,
                        "description": "",
                        "parameters": [],
                        "responses": {},
                    }

            # Extract docstring for current endpoint
            elif current_endpoint and stripped_line.startswith('"""'):
                docstring_start = i
                docstring_end = i
                for j in range(i + 1, len(lines)):
                    if '"""' in lines[j]:
                        docstring_end = j
                        break

                if docstring_end > docstring_start:
                    docstring = "\n".join(lines[docstring_start : docstring_end + 1])
                    current_endpoint["description"] = self._clean_docstring(docstring)

                endpoints.append(current_endpoint)
                current_endpoint = None

        return endpoints

    async def _extract_data_models(self, file_path: Path) -> list[dict[str, Any]]:
        """Extract data models from a Python file."""
        models: list[Any] = []

        content = self.safe_file_read(file_path)
        if not content:
            return models

        # Look for Pydantic models or dataclasses
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped_line = line.strip()

            # Pydantic BaseModel
            if stripped_line.startswith("class ") and (
                "BaseModel" in stripped_line or "Schema" in stripped_line
            ):
                class_name = stripped_line.split("(")[0].replace("class ", "").strip()

                model = {
                    "name": class_name,
                    "type": "pydantic",
                    "file": str(file_path),
                    "line": i + 1,
                    "fields": [],
                    "description": "",
                }

                # Extract fields
                for j in range(i + 1, min(i + 50, len(lines))):
                    field_line = lines[j].strip()
                    if not field_line or field_line.startswith("class "):
                        break

                    # Field definition
                    if ":" in field_line and not field_line.startswith("def "):
                        field_parts = field_line.split(":")
                        field_name = field_parts[0].strip()
                        field_type = field_parts[1].strip() if len(field_parts) > 1 else "Any"

                        model["fields"].append(
                            {
                                "name": field_name,
                                "type": field_type,
                                "required": "Optional" not in field_type,
                            },
                        )

                models.append(model)

        return models

    async def _generate_openapi_spec(
        self, api_docs: dict[str, Any], project_dir: Path
    ) -> dict[str, Any] | None:
        """Generate OpenAPI specification from extracted API data."""
        if not api_docs["endpoints"]:
            return None

        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": project_dir.name.replace("-", " ").replace("_", " ").title(),
                "version": "1.0.0",
                "description": f"API documentation for {project_dir.name}",
            },
            "paths": {},
            "components": {"schemas": {}},
        }

        # Add endpoints
        for endpoint in api_docs["endpoints"]:
            path = endpoint["path"]
            method = endpoint["method"].lower()

            if path not in spec["paths"]:
                spec["paths"][path] = {}

            spec["paths"][path][method] = {
                "summary": endpoint["function"].replace("_", " ").title(),
                "description": endpoint["description"] or f"{method.upper()} {path}",
                "responses": {"200": {"description": "Successful response"}},
            }

        # Add schemas from models
        for model in api_docs["models"]:
            schema = {"type": "object", "properties": {}, "required": []}

            for field in model["fields"]:
                schema["properties"][field["name"]] = {
                    "type": self._convert_python_type_to_openapi(field["type"])
                }
                if field["required"]:
                    schema["required"].append(field["name"])

            spec["components"]["schemas"][model["name"]] = schema

        return spec

    async def _generate_api_markdown(self, api_docs: dict[str, Any]) -> str:
        """Generate API documentation in Markdown format."""
        content = ["# API Documentation\n"]

        if api_docs["endpoints"]:
            content.append("## Endpoints\n")

            for endpoint in api_docs["endpoints"]:
                content.append(f"### {endpoint['method']} {endpoint['path']}\n")

                if endpoint["description"]:
                    content.append(f"{endpoint['description']}\n")

                content.append(f"**Function:** `{endpoint['function']}`\n")
                content.append(f"**File:** `{endpoint['file']}`\n")
                content.append("")

        if api_docs["models"]:
            content.append("## Data Models\n")

            for model in api_docs["models"]:
                content.append(f"### {model['name']}\n")

                if model["description"]:
                    content.append(f"{model['description']}\n")

                if model["fields"]:
                    content.append("**Fields:**\n")
                    for field in model["fields"]:
                        required = " (required)" if field["required"] else " (optional)"
                        content.append(f"- `{field['name']}`: {field['type']}{required}")

                content.append(f"\n**File:** `{model['file']}`\n")
                content.append("")

        return "\n".join(content)

    async def _analyze_project_for_readme(self, project_dir: Path) -> dict[str, Any]:
        """Analyze project to gather information for README."""
        info = {
            "name": project_dir.name,
            "description": "",
            "technologies": [],
            "features": [],
            "installation": [],
            "usage": [],
            "api_endpoints": [],
            "has_tests": False,
            "has_docs": False,
            "license": None,
        }

        # Check for common files
        if (project_dir / "requirements.txt").exists() or (project_dir / "pyproject.toml").exists():
            info["technologies"].append("Python")
            info["installation"].append("pip install -r requirements.txt")

        if (project_dir / "package.json").exists():
            info["technologies"].append("Node.js")
            info["installation"].append("npm install")

        if (project_dir / "Cargo.toml").exists():
            info["technologies"].append("Rust")
            info["installation"].append("cargo build")

        # Check for tests
        test_dirs = ["tests", "test", "__tests__"]
        info["has_tests"] = any((project_dir / test_dir).exists() for test_dir in test_dirs)

        # Check for documentation
        docs_dirs = ["docs", "documentation"]
        info["has_docs"] = any((project_dir / docs_dir).exists() for docs_dir in docs_dirs)

        # Check for license
        license_files = ["LICENSE", "LICENSE.txt", "LICENSE.md"]
        for license_file in license_files:
            if (project_dir / license_file).exists():
                info["license"] = license_file
                break

        # Extract description from setup.py or package.json
        await self._extract_project_description(project_dir, info)

        return info

    async def _generate_readme_content(self, project_info: dict[str, Any]) -> str:
        """Generate README content from project information."""
        template = self.templates["readme"]

        # Use AI to enhance the README
        enhancement_prompt = f"""Generate a comprehensive README.md for a project with these details:

Project Name: {project_info["name"]}
Technologies: {", ".join(project_info["technologies"])}
Has Tests: {project_info["has_tests"]}
Has Documentation: {project_info["has_docs"]}
Installation Steps: {project_info["installation"]}

Create a professional README with:
1. Project title and description
2. Features list
3. Installation instructions
4. Usage examples
5. API documentation (if applicable)
6. Testing information
7. Contributing guidelines
8. License information

Make it engaging and informative. Return only the README content in Markdown format."""

        ai_readme = await self.generate_with_gemini(enhancement_prompt, "documentation")

        if ai_readme and len(ai_readme) > 500:  # Ensure we got substantial content
            return ai_readme

        # Fallback to template-based generation
        return template.format(
            project_name=project_info["name"].replace("-", " ").replace("_", " ").title(),
            description=project_info.get("description", "A comprehensive software project."),
            technologies=(
                ", ".join(project_info["technologies"])
                if project_info["technologies"]
                else "Various technologies"
            ),
            installation_steps=(
                "\n".join([f"```bash\n{step}\n```" for step in project_info["installation"]])
                if project_info["installation"]
                else "Installation instructions coming soon."
            ),
            has_tests="✅ Yes" if project_info["has_tests"] else "❌ No",
            has_docs="✅ Yes" if project_info["has_docs"] else "❌ No",
        )

    async def _add_docstrings_to_file(self, file_path: Path) -> dict[str, Any]:
        """Add docstrings to functions and classes in a Python file."""
        content = self.safe_file_read(file_path)
        if not content:
            return {"updated": False, "functions": 0, "classes": 0}

        lines = content.split("\n")
        updated_lines: list[Any] = []
        functions_documented = 0
        classes_documented = 0
        file_updated = False

        i = 0
        while i < len(lines):
            line = lines[i]
            updated_lines.append(line)

            # Check for function or class definition
            stripped = line.strip()
            if (stripped.startswith(("def ", "class "))) and not stripped.endswith(":"):
                # Multi-line definition, find the end
                while i + 1 < len(lines) and not lines[i].strip().endswith(":"):
                    i += 1
                    updated_lines.append(lines[i])

            if stripped.startswith(("def ", "class ")):
                # Check if next non-empty line is a docstring
                has_docstring = False
                next_idx = i + 1
                while next_idx < len(lines) and not lines[next_idx].strip():
                    next_idx += 1

                if next_idx < len(lines) and lines[next_idx].strip().startswith('"""'):
                    has_docstring = True

                if not has_docstring:
                    # Generate docstring using AI
                    context = "\n".join(lines[max(0, i - 2) : min(len(lines), i + 10)])
                    docstring = await self._generate_docstring(context, stripped)

                    if docstring:
                        # Add docstring with proper indentation
                        indent = len(line) - len(line.lstrip())
                        docstring_lines = docstring.split("\n")
                        for doc_line in docstring_lines:
                            updated_lines.append(" " * (indent + 4) + doc_line)

                        if stripped.startswith("def "):
                            functions_documented += 1
                        else:
                            classes_documented += 1
                        file_updated = True

            i += 1

        # Write updated content if changes were made
        if file_updated:
            updated_content = "\n".join(updated_lines)
            self.safe_file_write(file_path, updated_content)

        return {
            "updated": file_updated,
            "functions": functions_documented,
            "classes": classes_documented,
        }

    async def _generate_docstring(self, context: str, definition: str) -> str | None:
        """Generate docstring for a function or class."""
        prompt = f"""Generate a concise docstring for this Python code:

Context:
```python
{context}
```

Generate only the docstring content (without triple quotes) for the definition:
{definition}

Make it brief but informative, including:
- What the function/class does
- Parameters (if any)
- Returns (if applicable)
- Brief example if helpful

Keep it under 5 lines."""

        docstring = await self.generate_with_gemini(prompt, "documentation")

        if docstring:
            # Clean and format the docstring
            docstring = docstring.strip()
            if not docstring.startswith('"""'):
                docstring = f'"""\n{docstring}\n"""'
            return docstring

        return None

    async def _generate_module_docs(self, project_dir: Path) -> list[dict[str, Any]]:
        """Generate documentation for Python modules."""
        module_docs: list[Any] = []

        python_files = list(project_dir.rglob("*.py"))

        for file_path in python_files[:20]:  # Limit to first 20 files
            if any(exclude in str(file_path) for exclude in ["__pycache__", ".git", "venv"]):
                continue

            content = self.safe_file_read(file_path)
            if content:
                module_info = {
                    "module": str(file_path.relative_to(project_dir)),
                    "functions": [],
                    "classes": [],
                    "imports": [],
                }

                lines = content.split("\n")
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("def "):
                        func_name = stripped.split("(")[0].replace("def ", "")
                        module_info["functions"].append(func_name)
                    elif stripped.startswith("class "):
                        class_name = stripped.split("(")[0].replace("class ", "").replace(":", "")
                        module_info["classes"].append(class_name)
                    elif stripped.startswith(("import ", "from ")):
                        module_info["imports"].append(stripped)

                if module_info["functions"] or module_info["classes"]:
                    module_docs.append(module_info)

        return module_docs

    def _get_readme_template(self) -> str:
        """Get README template."""
        return """# {project_name}

{description}

## 🚀 Features

- Modern architecture and design patterns
- Comprehensive testing suite
- Detailed documentation
- Easy installation and setup

## 🛠 Technologies

{technologies}

## 📦 Installation

{installation_steps}

## 🚀 Usage

```bash
# Basic usage example
python main.py
```

## 📖 Documentation

- **API Documentation**: See `/docs/api/` for detailed API documentation
- **User Guide**: Check `/docs/user-guide.md` for comprehensive usage guide

## 🧪 Testing

Tests: {has_tests}

```bash
# Run tests
pytest
```

## 📚 Documentation

Documentation: {has_docs}

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to all contributors
- Built with modern development practices
- Powered by AI-assisted development tools
"""

    def _get_api_docs_template(self) -> str:
        """Get API documentation template."""
        return """# API Documentation

This document provides comprehensive information about the API endpoints, data models, and usage examples.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

This API uses JWT token authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

{endpoints}

## Data Models

{models}

## Error Responses

All endpoints return errors in the following format:

```json
{
  "error": "Error description",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Rate Limiting

- 100 requests per minute per IP
- 1000 requests per hour per authenticated user
"""

    def _get_contributing_template(self) -> str:
        """Get contributing guidelines template."""
        return """# Contributing Guidelines

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Development Setup

1. Fork the repository
2. Clone your fork
3. Install dependencies
4. Run tests to ensure everything works

## Code Standards

- Follow PEP 8 for Python code
- Add tests for new features
- Update documentation as needed
- Use meaningful commit messages

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Add/update tests
4. Update documentation
5. Submit pull request

## Reporting Issues

Please use the issue tracker to report bugs or request features.
"""

    def _get_changelog_template(self) -> str:
        """Get changelog template."""
        return """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features coming soon

### Changed
- Updates and improvements

### Fixed
- Bug fixes

## [1.0.0] - {date}

### Added
- Initial release
- Core functionality
- Basic documentation
"""

    def _clean_docstring(self, docstring: str) -> str:
        """Clean and format docstring."""
        # Remove triple quotes
        docstring = docstring.replace('"""', "").strip()
        # Remove excessive whitespace
        lines = [line.strip() for line in docstring.split("\n") if line.strip()]
        return " ".join(lines)

    def _convert_python_type_to_openapi(self, python_type: str) -> str:
        """Convert Python type annotation to OpenAPI type."""
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "dict": "object",
            "list": "array",
            "List": "array",
            "Dict": "object",
        }

        # Clean up the type string
        clean_type = python_type.split("[")[0].strip()
        return type_mapping.get(clean_type, "string")

    async def _extract_project_description(self, project_dir: Path, info: dict[str, Any]) -> None:
        """Extract project description from various sources."""
        # Try package.json
        package_json = project_dir / "package.json"
        if package_json.exists():
            content = self.safe_file_read(package_json)
            if content:
                data = safe_json_parse(content)
                if data and "description" in data:
                    info["description"] = data["description"]
                    return

        # Try pyproject.toml
        pyproject_toml = project_dir / "pyproject.toml"
        if pyproject_toml.exists():
            content = self.safe_file_read(pyproject_toml)
            if content and "description" in content:
                # Simple extraction - would need proper TOML parsing for production
                for line in content.split("\n"):
                    if line.strip().startswith("description"):
                        desc = line.split("=", 1)[1].strip().strip('"').strip("'")
                        info["description"] = desc
                        return

        # Fallback
        info["description"] = f"A {info['name']} project"

    async def _update_existing_readme(self, readme_path: Path, project_dir: Path) -> bool:
        """Update existing README with new information."""
        current_content = self.safe_file_read(readme_path)
        if not current_content:
            return False

        # Analyze current README and project
        project_info = await self._analyze_project_for_readme(project_dir)

        # Generate updated content while preserving custom sections
        updated_content = await self._merge_readme_content(current_content, project_info)

        if updated_content != current_content:
            return self.safe_file_write(readme_path, updated_content)

        return False

    async def _merge_readme_content(
        self, current_content: str, project_info: dict[str, Any]
    ) -> str:
        """Merge current README with updated project information."""
        # For now, return updated content
        # In production, this would intelligently merge sections
        return await self._generate_readme_content(project_info)

    async def _update_changelog(self, project_dir: Path) -> bool:
        """Update CHANGELOG.md with recent changes."""
        changelog_path = project_dir / "CHANGELOG.md"

        # Get git log for recent changes
        self._get_recent_git_changes(project_dir)

        template = self.templates["changelog"].format(date=str(datetime.now(timezone.utc).date()))

        return self.safe_file_write(changelog_path, template)

    def _get_recent_git_changes(self, project_dir: Path) -> list[str]:
        """Get recent git changes for changelog."""
        try:
            result = self.run_command(["git", "log", "--oneline", "--since=1 week ago"])
            if result and result.returncode == 0:
                return result.stdout.strip().split("\n")
        except Exception:
            pass
        return []

    async def _analyze_project_for_guide(self, project_dir: Path) -> dict[str, Any]:
        """Analyze project for user guide generation."""
        analysis = {
            "name": project_dir.name,
            "main_modules": [],
            "entry_points": [],
            "configuration": [],
            "examples": [],
        }

        # Find main entry points
        main_files = ["main.py", "app.py", "__main__.py", "run.py"]
        for main_file in main_files:
            if (project_dir / main_file).exists():
                analysis["entry_points"].append(main_file)

        # Find configuration files
        config_files = ["config.py", "settings.py", ".env", "config.json"]
        for config_file in config_files:
            if (project_dir / config_file).exists():
                analysis["configuration"].append(config_file)

        return analysis

    async def _generate_guide_content(self, analysis: dict[str, Any]) -> str:
        """Generate user guide content."""
        prompt = f"""Create a comprehensive user guide for a project with these characteristics:

Project Name: {analysis["name"]}
Entry Points: {analysis["entry_points"]}
Configuration Files: {analysis["configuration"]}

Generate a user guide with:
1. Getting Started section
2. Configuration guide
3. Usage examples
4. Common tasks
5. Troubleshooting
6. FAQ

Make it practical and user-friendly. Return in Markdown format."""

        ai_guide = await self.generate_with_gemini(prompt, "documentation")

        if ai_guide and len(ai_guide) > 300:
            return ai_guide

        # Fallback template
        return f"""# {analysis["name"]} User Guide

## Getting Started

Welcome to {analysis["name"]}! This guide will help you get up and running quickly.

## Installation

Follow the installation instructions in the README.md file.

## Configuration

Configure the application by editing these files:
{chr(10).join(f"- {config}" for config in analysis["configuration"])}

## Usage

### Basic Usage

Run the application using:
```bash
python {analysis["entry_points"][0] if analysis["entry_points"] else "main.py"}
```

### Advanced Usage

For advanced features and configuration options, see the API documentation.

## Troubleshooting

### Common Issues

1. **Installation problems**: Make sure you have the correct Python version
2. **Configuration errors**: Check your configuration files for syntax errors
3. **Runtime issues**: Check the logs for detailed error messages

## FAQ

**Q: How do I get started?**
A: Follow the installation instructions and run the basic usage example.

**Q: Where can I find more help?**
A: Check the documentation or open an issue on GitHub.
"""

    def _get_readme_sections(self) -> list[str]:
        """Get list of README sections that were generated."""
        return [
            "Title and Description",
            "Features",
            "Technologies",
            "Installation",
            "Usage",
            "Documentation",
            "Testing",
            "Contributing",
            "License",
            "Acknowledgments",
        ]

    def register_tools(self) -> None:
        """Register MCP tools for documentation generator."""

        @self.mcp.tool()
        async def generate_api_docs(project_path: str = ".") -> dict[str, Any]:
            """Generate API documentation from code."""
            if not self.validate_job_parameters(
                "generate_api_docs", {"project_path": project_path}
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("generate_api_docs", {"project_path": project_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def generate_readme(project_path: str = ".") -> dict[str, Any]:
            """Generate or update README file."""
            if not self.validate_job_parameters("generate_readme", {"project_path": project_path}):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("generate_readme", {"project_path": project_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def generate_code_docs(project_path: str = ".") -> dict[str, Any]:
            """Generate code documentation (docstrings)."""
            if not self.validate_job_parameters(
                "generate_code_docs", {"project_path": project_path}
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("generate_code_docs", {"project_path": project_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def update_documentation(project_path: str = ".") -> dict[str, Any]:
            """Update existing documentation."""
            if not self.validate_job_parameters(
                "update_documentation", {"project_path": project_path}
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("update_documentation", {"project_path": project_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def get_documentation_stats() -> dict[str, Any]:
            """Get documentation generator statistics."""
            return self.create_success_response(
                {
                    "agent_stats": self.get_agent_stats(),
                    "supported_formats": self.formats,
                    "template_count": len(self.templates),
                },
            )


# Create global instance
documentation_generator_service = DocumentationGeneratorService()


# MCP Server Integration
# Initialize MCP server for standalone usage
mcp = FastMCP("documentation-generator")


@mcp.tool()
async def generate_documentation(
    source_path: str,
    doc_type: str = "api",
    output_format: str = "markdown",
    include_examples: bool = True,
) -> dict[str, Any]:
    """Generate documentation for code or APIs.

    Args:
        source_path: Path to the source code or project
        doc_type: Type of documentation ('api', 'code', 'readme', 'user-guide')
        output_format: Output format ('markdown', 'html', 'sphinx', 'openapi') - not used in current implementation
        include_examples: Whether to include usage examples - not used in current implementation

    Returns:
        Generated documentation content

    """
    try:
        # Map doc_type to job_type
        job_type_mapping = {
            "api": "generate_api_docs",
            "code": "generate_code_docs",
            "readme": "generate_readme",
            "user-guide": "generate_user_guide",
        }

        job_type = job_type_mapping.get(doc_type, "generate_api_docs")

        job_id = documentation_generator_service.create_job(
            job_type,
            {
                "project_path": source_path,
                "doc_type": doc_type,
                "output_format": output_format,
                "include_examples": include_examples,
            },
        )

        return await documentation_generator_service.execute_job_async(job_id)
    except Exception as e:
        logger.exception(f"Documentation generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def generate_api_docs(
    source_path: str,
    framework: str = "fastapi",
    include_schemas: bool = True,
    include_examples: bool = True,
) -> dict[str, Any]:
    """Generate API documentation from code.

    Args:
        source_path: Path to the API source code
        framework: API framework ('fastapi', 'flask', 'django', 'express') - not used in current implementation
        include_schemas: Whether to include request/response schemas - not used in current implementation
        include_examples: Whether to include usage examples - not used in current implementation

    Returns:
        API documentation content

    """
    try:
        job_id = documentation_generator_service.create_job(
            "generate_api_docs",
            {
                "project_path": source_path,
                "framework": framework,
                "include_schemas": include_schemas,
                "include_examples": include_examples,
            },
        )

        return await documentation_generator_service.execute_job_async(job_id)
    except Exception as e:
        logger.exception(f"API documentation generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def generate_readme(
    project_path: str,
    template_style: str = "comprehensive",
    include_badges: bool = True,
    include_installation: bool = True,
    include_contributing: bool = True,
) -> dict[str, Any]:
    """Generate or update a README file for a project.

    Args:
        project_path: Path to the project directory
        template_style: README style ('minimal', 'standard', 'comprehensive') - not used in current implementation
        include_badges: Whether to include status badges - not used in current implementation
        include_installation: Whether to include installation instructions - not used in current implementation
        include_contributing: Whether to include contribution guidelines - not used in current implementation

    Returns:
        Generated README content

    """
    try:
        job_id = documentation_generator_service.create_job(
            "generate_readme",
            {
                "project_path": project_path,
                "template_style": template_style,
                "include_badges": include_badges,
                "include_installation": include_installation,
                "include_contributing": include_contributing,
            },
        )

        return await documentation_generator_service.execute_job_async(job_id)
    except Exception as e:
        logger.exception(f"README generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def generate_code_docs(
    file_path: str,
    doc_style: str = "google",
    include_types: bool = True,
    include_examples: bool = False,
) -> dict[str, Any]:
    """Generate documentation for code functions and classes.

    Args:
        file_path: Path to the code file
        doc_style: Documentation style ('google', 'sphinx', 'numpy', 'jsdoc') - not used in current implementation
        include_types: Whether to include type annotations - not used in current implementation
        include_examples: Whether to include usage examples - not used in current implementation

    Returns:
        Code with generated documentation

    """
    try:
        job_id = documentation_generator_service.create_job(
            "generate_code_docs",
            {
                "project_path": file_path,
                "doc_style": doc_style,
                "include_types": include_types,
                "include_examples": include_examples,
            },
        )

        return await documentation_generator_service.execute_job_async(job_id)
    except Exception as e:
        logger.exception(f"Code documentation generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def generate_user_guide(
    project_path: str,
    audience: str = "developer",
    include_tutorials: bool = True,
    include_faq: bool = True,
) -> dict[str, Any]:
    """Generate a user guide for a project.

    Args:
        project_path: Path to the project directory
        audience: Target audience ('developer', 'end-user', 'admin') - not used in current implementation
        include_tutorials: Whether to include tutorials - not used in current implementation
        include_faq: Whether to include FAQ section - not used in current implementation

    Returns:
        User guide content

    """
    try:
        job_id = documentation_generator_service.create_job(
            "generate_user_guide",
            {
                "project_path": project_path,
                "audience": audience,
                "include_tutorials": include_tutorials,
                "include_faq": include_faq,
            },
        )

        return await documentation_generator_service.execute_job_async(job_id)
    except Exception as e:
        logger.exception(f"User guide generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def update_documentation(
    doc_path: str,
    source_path: str,
    preserve_custom_sections: bool = True,
) -> dict[str, Any]:
    """Update existing documentation based on code changes.

    Args:
        doc_path: Path to the existing documentation - used as project_path
        source_path: Path to the updated source code - not used in current implementation
        preserve_custom_sections: Whether to preserve manually added sections - not used in current implementation

    Returns:
        Updated documentation content

    """
    try:
        job_id = documentation_generator_service.create_job(
            "update_documentation",
            {
                "project_path": doc_path,
                "source_path": source_path,
                "preserve_custom_sections": preserve_custom_sections,
            },
        )

        return await documentation_generator_service.execute_job_async(job_id)
    except Exception as e:
        logger.exception(f"Documentation update failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def get_documentation_status(job_id: str) -> dict[str, Any]:
    """Get the status of an ongoing documentation generation job.

    Args:
        job_id: The job ID to check

    Returns:
        Job status and progress information

    """
    try:
        return documentation_generator_service.get_job_status(job_id)
    except Exception as e:
        logger.exception(f"Failed to get job status: {e}")
        return {"status": "error", "error": str(e)}


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
