"""Unified Documentation Generator Agent.

This agent consolidates functionality from:
- app/automation/documentation_agent.py (FastMCP automation agent)
- app/agents/gemini_orchestrator.py (DocumentationAgent class)
- Related documentation generation capabilities

Key features:
- Comprehensive documentation generation for code, APIs, and projects
- Multiple output formats (Markdown, HTML, RST, JSON)
- AI-powered content generation using Google Gemini
- FastMCP server integration for Claude CLI access
- PyO3 Rust extensions for high-performance operations
- Batch processing for large codebases
- Intelligent content structuring and organization
- Multiple documentation styles (Google, Sphinx, NumPy, etc.)

Usage:
    # As standalone agent
    generator = UnifiedDocumentationGenerator()
    result = await generator.generate_documentation("/path/to/code.py")

    # As MCP server for Claude CLI
    python -m app.core.agents.unified_documentation_generator
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import aiohttp
import cachetools
from fastmcp import FastMCP
from pydantic import BaseModel
from pydantic import Field
import vertexai
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel

from gterminal.core.agents.base_unified_agent import BaseUnifiedAgent
from gterminal.core.agents.base_unified_agent import UnifiedJob

# Import PyO3 Rust extensions for performance
try:
    from fullstack_agent_rust import RustCache
    from fullstack_agent_rust import RustFileOps
    from fullstack_agent_rust import RustJsonProcessor

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("PyO3 Rust extensions not available - falling back to Python implementations")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("unified-documentation-generator")

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "auricleinc-gemini")
LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
DEFAULT_DOC_STYLE = os.getenv("DOC_STYLE", "google")

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Thread pool for parallel operations
thread_pool = ThreadPoolExecutor(max_workers=8)


class DocumentationType(Enum):
    """Documentation generation types."""

    MODULE = "module"
    API = "api"
    CLASS = "class"
    FUNCTION = "function"
    README = "readme"
    USER_GUIDE = "user_guide"
    ARCHITECTURE = "architecture"
    CHANGELOG = "changelog"


class OutputFormat(Enum):
    """Supported output formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    RST = "rst"
    JSON = "json"
    SPHINX = "sphinx"


class DocumentationStyle(Enum):
    """Documentation styles."""

    GOOGLE = "google"
    SPHINX = "sphinx"
    NUMPY = "numpy"
    JSDOC = "jsdoc"
    PYDOC = "pydoc"


@dataclass
class DocumentationRequest:
    """Documentation generation request."""

    source_path: str
    output_path: str | None = None
    doc_type: DocumentationType = DocumentationType.MODULE
    output_format: OutputFormat = OutputFormat.MARKDOWN
    style: DocumentationStyle = DocumentationStyle.GOOGLE
    include_examples: bool = True
    include_tests: bool = True
    include_private: bool = False
    max_depth: int = 5


class DocumentationContext(BaseModel):
    """Context for documentation generation."""

    project_root: str = Field(..., description="Project root directory")
    source_files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    style_config: dict[str, Any] = Field(default_factory=dict)


class GeneratedDocumentation(BaseModel):
    """Generated documentation result."""

    content: str = Field(..., description="Generated documentation content")
    file_path: str | None = Field(default=None)
    doc_type: str = Field(...)
    format: str = Field(...)
    style: str = Field(...)
    metadata: dict[str, Any] = Field(default_factory=dict)
    performance_metrics: dict[str, float] = Field(default_factory=dict)


class Job(BaseModel):
    """Documentation generation job."""

    id: str = Field(..., description="Unique job identifier")
    status: str = Field(default="pending")
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    message: str = Field(default="")
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = Field(default=None)
    completed_at: float | None = Field(default=None)
    result: dict[str, Any] | None = Field(default=None)
    error: str | None = Field(default=None)


class CircuitBreaker:
    """Circuit breaker for API call resilience."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        # half_open
        return True

    def record_success(self) -> None:
        """Record successful execution."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class UnifiedDocumentationGenerator(BaseUnifiedAgent):
    """Unified documentation generator consolidating all documentation functionality.

    This class combines:
    - Document generation from automation/documentation_agent.py
    - Agent orchestration from gemini_orchestrator.py
    - High-performance PyO3 Rust integration
    - FastMCP server capabilities
    - Comprehensive job management
    - Circuit breaker patterns for resilience
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the unified documentation generator."""
        super().__init__(
            "unified_documentation_generator",
            "Comprehensive documentation generation with AI-powered content",
        )

        self.config = config or {}
        self.active_jobs: set[str] = set()
        self.circuit_breaker = CircuitBreaker()

        # Initialize caching
        if RUST_AVAILABLE:
            self.cache = RustCache(capacity=1000, ttl_seconds=3600)
            self.file_ops = RustFileOps()
            self.json_processor = RustJsonProcessor()
            logger.info("Using PyO3 Rust extensions for high performance")
        else:
            self.cache = cachetools.TTLCache(maxsize=1000, ttl=3600)
            logger.info("Using Python fallback implementations")

        # Initialize Gemini model
        try:
            self.model = GenerativeModel(
                model_name=MODEL_NAME,
                generation_config=GenerationConfig(
                    max_output_tokens=8192, temperature=0.1, top_p=0.8, top_k=40
                ),
            )
            logger.info(f"Initialized Gemini model: {MODEL_NAME}")
        except Exception as e:
            logger.exception(f"Failed to initialize Gemini model: {e}")
            raise

        # HTTP session for external requests
        self.session: aiohttp.ClientSession | None = None

        # Documentation style configuration
        self.doc_style = DEFAULT_DOC_STYLE
        self.default_audience = "developers"

        # Code patterns for documentation extraction
        self.code_patterns = {
            "python": {
                "functions": r"def\s+(\w+)\s*\([^)]*\):",
                "classes": r"class\s+(\w+)(?:\([^)]*\))?:",
                "docstrings": r'"""([^"]*)"""',
                "imports": r"^(?:from\s+\S+\s+)?import\s+(.+)$",
            },
            "javascript": {
                "functions": r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
                "classes": r"class\s+(\w+)(?:\s+extends\s+\w+)?",
                "exports": r"(?:export\s+(?:default\s+)?(?:function|class|const)\s+(\w+)|module\.exports\s*=)",
            },
            "rust": {
                "functions": r"(?:pub\s+)?fn\s+(\w+)",
                "structs": r"(?:pub\s+)?struct\s+(\w+)",
                "traits": r"(?:pub\s+)?trait\s+(\w+)",
                "modules": r"mod\s+(\w+)",
            },
        }

        logger.info("UnifiedDocumentationGenerator initialized successfully")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            self.session = aiohttp.ClientSession(
                connector=connector, timeout=aiohttp.ClientTimeout(total=60)
            )
        return self.session

    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        import uuid

        return str(uuid.uuid4())

    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key."""
        return f"{prefix}:{hash(str(args))}"

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run blocking function in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(thread_pool, lambda: func(*args, **kwargs))

    def get_supported_job_types(self) -> list[str]:
        """Return supported job types."""
        return [
            "generate_readme",
            "generate_api_docs",
            "generate_code_docs",
            "generate_user_guide",
            "generate_changelog",
            "comprehensive_docs",
        ]

    async def execute_job(self, job: UnifiedJob) -> Any:
        """Execute documentation generation job with progress tracking."""
        job_type = job.job_type
        parameters = job.parameters

        job.update_progress(5.0, f"Starting {job_type}")

        try:
            if job_type == "generate_readme":
                return await self._generate_readme(job, parameters.get("project_path", "."))
            if job_type == "generate_api_docs":
                return await self._generate_api_docs(job, parameters.get("source_path", "."))
            if job_type == "generate_code_docs":
                return await self._generate_code_docs(job, parameters["file_path"])
            if job_type == "generate_user_guide":
                return await self._generate_user_guide(job, parameters.get("project_path", "."))
            if job_type == "generate_changelog":
                return await self._generate_changelog(job, parameters.get("project_path", "."))
            if job_type == "comprehensive_docs":
                return await self._generate_comprehensive_docs(
                    job, parameters.get("project_path", ".")
                )
            msg = f"Unsupported job type: {job_type}"
            raise ValueError(msg)

        except Exception as e:
            logger.exception(f"Job {job.job_id} failed: {e!s}")
            raise

    async def generate_documentation(
        self,
        source_path: str,
        output_path: str | None = None,
        doc_type: DocumentationType = DocumentationType.MODULE,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        style: DocumentationStyle = DocumentationStyle.GOOGLE,
        include_examples: bool = True,
        include_tests: bool = True,
        include_private: bool = False,
        max_depth: int = 5,
    ) -> GeneratedDocumentation:
        """Generate comprehensive documentation for source code.

        Args:
            source_path: Path to source file or directory
            output_path: Optional output file path
            doc_type: Type of documentation to generate
            output_format: Output format (markdown, html, rst, etc.)
            style: Documentation style (google, sphinx, numpy)
            include_examples: Whether to include usage examples
            include_tests: Whether to include test documentation
            include_private: Whether to document private members
            max_depth: Maximum directory depth to process

        Returns:
            GeneratedDocumentation object with content and metadata

        """
        start_time = time.time()

        try:
            # Create documentation request
            request = DocumentationRequest(
                source_path=source_path,
                output_path=output_path,
                doc_type=doc_type,
                output_format=output_format,
                style=style,
                include_examples=include_examples,
                include_tests=include_tests,
                include_private=include_private,
                max_depth=max_depth,
            )

            # Analyze source structure
            logger.info(f"Analyzing source structure: {source_path}")
            structure = await self._analyze_source_structure(source_path)

            if "error" in structure:
                msg = f"Source analysis failed: {structure['error']}"
                raise ValueError(msg)

            # Generate documentation prompt
            prompt = self._prepare_documentation_prompt(request, structure)

            # Generate documentation content
            logger.info("Generating documentation content with Gemini")
            content = await self._generate_content_with_retry(prompt)

            # Post-process content based on format
            if request.output_format == OutputFormat.JSON:
                try:
                    if RUST_AVAILABLE:
                        # Validate JSON with Rust processor
                        self.json_processor.parse_json(content)
                    else:
                        json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("Generated content is not valid JSON, wrapping it")
                    content = json.dumps({"documentation": content}, indent=2)

            # Save to file if output path specified
            final_output_path = output_path
            if output_path:
                if RUST_AVAILABLE:
                    self.file_ops.write_file(output_path, content)
                else:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(content)
                logger.info(f"Documentation saved to: {output_path}")

            # Calculate performance metrics
            duration = time.time() - start_time

            result = GeneratedDocumentation(
                content=content,
                file_path=final_output_path,
                doc_type=doc_type.value,
                format=output_format.value,
                style=style.value,
                metadata={
                    "source_path": source_path,
                    "structure": structure,
                    "generation_time": time.time(),
                    "rust_powered": RUST_AVAILABLE,
                },
                performance_metrics={
                    "duration_seconds": duration,
                    "content_length": len(content),
                    "structure_complexity": structure.get("complexity_score", 0),
                },
            )

            logger.info(f"Documentation generation completed in {duration:.2f}s")
            return result

        except Exception as e:
            logger.exception(f"Documentation generation failed: {e}")
            raise

    async def _analyze_source_structure(self, source_path: str) -> dict[str, Any]:
        """Analyze source code structure."""
        cache_key = self._get_cache_key("source_structure", source_path)

        # Check cache first
        cached = self.cache.get(cache_key) if RUST_AVAILABLE else self.cache.get(cache_key)

        if cached:
            return cached

        try:
            path = Path(source_path)
            structure = {
                "type": "unknown",
                "classes": [],
                "functions": [],
                "imports": [],
                "docstring": None,
                "complexity_score": 0,
                "lines_of_code": 0,
            }

            if path.is_file() and path.suffix == ".py":
                if RUST_AVAILABLE:
                    content = self.file_ops.read_file(str(path))
                else:
                    with open(path, encoding="utf-8") as f:
                        content = f.read()

                structure = await self._analyze_python_file(content)
                structure["lines_of_code"] = len(content.splitlines())

            elif path.is_dir():
                structure = await self._analyze_directory(path)

            # Cache result
            if RUST_AVAILABLE:
                self.cache.set(cache_key, structure)
            else:
                self.cache[cache_key] = structure

            return structure

        except Exception as e:
            logger.exception(f"Failed to analyze source structure: {e}")
            return {"error": str(e)}

    async def _analyze_python_file(self, content: str) -> dict[str, Any]:
        """Analyze Python file structure."""
        import ast

        try:
            tree = ast.parse(content)

            structure = {
                "type": "python_module",
                "classes": [],
                "functions": [],
                "imports": [],
                "docstring": ast.get_docstring(tree),
                "complexity_score": 0,
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    structure["classes"].append(
                        {
                            "name": node.name,
                            "docstring": ast.get_docstring(node),
                            "methods": [
                                n.name for n in node.body if isinstance(n, ast.FunctionDef)
                            ],
                            "line_number": node.lineno,
                        },
                    )
                elif isinstance(node, ast.FunctionDef):
                    if not any(
                        node.lineno > cls.get("line_number", 0) for cls in structure["classes"]
                    ):
                        structure["functions"].append(
                            {
                                "name": node.name,
                                "docstring": ast.get_docstring(node),
                                "args": [arg.arg for arg in node.args.args],
                                "line_number": node.lineno,
                            },
                        )
                elif isinstance(node, ast.Import | ast.ImportFrom):
                    if isinstance(node, ast.Import):
                        structure["imports"].extend([alias.name for alias in node.names])
                    else:
                        module = node.module or ""
                        structure["imports"].append(module)

            # Calculate complexity score
            structure["complexity_score"] = len(structure["classes"]) * 2 + len(
                structure["functions"]
            )

            return structure

        except SyntaxError as e:
            logger.warning(f"Syntax error in Python file: {e}")
            return {"error": f"Syntax error: {e}"}
        except Exception as e:
            logger.exception(f"Failed to analyze Python file: {e}")
            return {"error": str(e)}

    async def _analyze_directory(self, path: Path) -> dict[str, Any]:
        """Analyze directory structure."""
        structure = {
            "type": "directory",
            "files": [],
            "subdirectories": [],
            "python_modules": 0,
            "total_files": 0,
        }

        try:
            for item in path.iterdir():
                if item.is_file():
                    structure["files"].append(
                        {"name": item.name, "extension": item.suffix, "size": item.stat().st_size},
                    )
                    structure["total_files"] += 1
                    if item.suffix == ".py":
                        structure["python_modules"] += 1
                elif item.is_dir() and not item.name.startswith("."):
                    structure["subdirectories"].append(item.name)

            return structure

        except Exception as e:
            logger.exception(f"Failed to analyze directory: {e}")
            return {"error": str(e)}

    async def _generate_content_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generate content using Gemini with retry logic."""
        if not self.circuit_breaker.can_execute():
            msg = "Circuit breaker is open - too many failures"
            raise Exception(msg)

        for attempt in range(max_retries):
            try:
                response = await self._run_in_thread(self.model.generate_content, prompt)

                if response.candidates and response.candidates[0].content.parts:
                    content = response.candidates[0].content.parts[0].text
                    self.circuit_breaker.record_success()
                    return content
                msg = "No content generated"
                raise ValueError(msg)

            except Exception as e:
                logger.warning(f"Content generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.circuit_breaker.record_failure()
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff

        msg = "Failed to generate content after retries"
        raise Exception(msg)

    def _prepare_documentation_prompt(
        self, request: DocumentationRequest, structure: dict[str, Any]
    ) -> str:
        """Prepare prompt for documentation generation."""
        style_guidelines = {
            DocumentationStyle.GOOGLE: """
            Use Google docstring style:
            - Brief description on first line
            - Args: section with parameter descriptions
            - Returns: section with return value description
            - Raises: section if applicable
            - Example: section with usage examples
            """,
            DocumentationStyle.SPHINX: """
            Use Sphinx/RST style:
            - :param param_name: Parameter description
            - :type param_name: Parameter type
            - :returns: Return description
            - :rtype: Return type
            - :raises ExceptionType: Exception description
            """,
            DocumentationStyle.NUMPY: """
            Use NumPy docstring style:
            - Parameters section with type and description
            - Returns section with type and description
            - Examples section with code samples
            """,
        }

        guidelines = style_guidelines.get(
            request.style, style_guidelines[DocumentationStyle.GOOGLE]
        )

        format_instructions = {
            OutputFormat.MARKDOWN: "Format output as clean, well-structured Markdown",
            OutputFormat.HTML: "Format output as semantic HTML with proper tags",
            OutputFormat.RST: "Format output as reStructuredText",
            OutputFormat.JSON: "Format output as structured JSON",
        }

        format_instruction = format_instructions.get(
            request.output_format, format_instructions[OutputFormat.MARKDOWN]
        )

        return f"""
        You are an expert technical documentation writer. Generate comprehensive {request.doc_type.value} documentation.

        DOCUMENTATION STYLE:
        {guidelines}

        OUTPUT FORMAT:
        {format_instruction}

        SOURCE STRUCTURE:
        {json.dumps(structure, indent=2)}

        REQUIREMENTS:
        1. Clear, concise explanations
        2. Complete parameter and return value documentation
        3. {"Include practical usage examples" if request.include_examples else "No examples needed"}
        4. {"Include test scenarios" if request.include_tests else "Skip test documentation"}
        5. {"Include private methods/attributes" if request.include_private else "Only document public API"}
        6. Maintain consistency with {request.style.value} style
        7. Ensure professional tone and accuracy

        Generate the documentation now:
        """

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()

        thread_pool.shutdown(wait=True)
        logger.info("UnifiedDocumentationGenerator cleanup completed")


# Global instance for MCP tools
_generator_instance: UnifiedDocumentationGenerator | None = None


async def get_generator() -> UnifiedDocumentationGenerator:
    """Get or create global generator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = UnifiedDocumentationGenerator()
    return _generator_instance


# FastMCP Tools for Claude CLI Integration


@mcp.tool()
async def generate_documentation(
    source_path: str,
    output_path: str = "",
    doc_type: str = "module",
    output_format: str = "markdown",
    style: str = "google",
    include_examples: bool = True,
    include_tests: bool = True,
    include_private: bool = False,
) -> dict[str, Any]:
    """Generate comprehensive documentation for source code.

    Args:
        source_path: Path to source file or directory to document
        output_path: Optional output file path (empty for content only)
        doc_type: Documentation type (module, api, class, function, readme, user_guide)
        output_format: Output format (markdown, html, rst, json)
        style: Documentation style (google, sphinx, numpy)
        include_examples: Whether to include usage examples
        include_tests: Whether to include test documentation
        include_private: Whether to document private members

    Returns:
        Generated documentation with metadata

    """
    try:
        generator = await get_generator()

        # Convert string enums
        doc_type_enum = DocumentationType(doc_type)
        format_enum = OutputFormat(output_format)
        style_enum = DocumentationStyle(style)

        result = await generator.generate_documentation(
            source_path=source_path,
            output_path=output_path if output_path else None,
            doc_type=doc_type_enum,
            output_format=format_enum,
            style=style_enum,
            include_examples=include_examples,
            include_tests=include_tests,
            include_private=include_private,
        )

        return {
            "status": "success",
            "content": result.content,
            "file_path": result.file_path,
            "doc_type": result.doc_type,
            "format": result.format,
            "style": result.style,
            "metadata": result.metadata,
            "performance_metrics": result.performance_metrics,
        }

    except Exception as e:
        logger.exception(f"Documentation generation failed: {e}")
        return {"status": "error", "error": str(e), "source_path": source_path}


@mcp.tool()
async def get_generator_status() -> dict[str, Any]:
    """Get unified documentation generator status and metrics.

    Returns:
        Generator status information

    """
    try:
        generator = await get_generator()

        return {
            "status": "success",
            "rust_available": RUST_AVAILABLE,
            "model_name": MODEL_NAME,
            "project_id": PROJECT_ID,
            "location": LOCATION,
            "active_jobs": len(generator.active_jobs),
            "circuit_breaker_state": generator.circuit_breaker.state,
            "cache_size": generator.cache.size() if RUST_AVAILABLE else len(generator.cache),
            "supported_formats": [f.value for f in OutputFormat],
            "supported_styles": [s.value for s in DocumentationStyle],
            "supported_doc_types": [t.value for t in DocumentationType],
        }

    except Exception as e:
        logger.exception(f"Failed to get generator status: {e}")
        return {"status": "error", "error": str(e)}


async def cleanup() -> None:
    """Clean up resources on shutdown."""
    global _generator_instance
    if _generator_instance:
        await _generator_instance.cleanup()
        _generator_instance = None


def main() -> None:
    """Main entry point for MCP server."""
    import signal
    import sys

    def signal_handler(signum, frame) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.run(cleanup())
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting Unified Documentation Generator MCP Server")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Project: {PROJECT_ID}")
    logger.info(f"Location: {LOCATION}")
    logger.info(f"Rust Extensions: {RUST_AVAILABLE}")

    try:
        # Run the MCP server
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")
    except Exception as e:
        logger.exception(f"Server error: {e}")
    finally:
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
