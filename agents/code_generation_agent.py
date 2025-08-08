"""Code Generation Agent Service - Automated code generation and scaffolding.
Generates code from specifications, templates, and architectural designs.

Enhanced with JSON RPC 2.0 compliance for standardized request/response patterns.
"""

import ast
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from gterminal.agents.base_agent_service import BaseAgentService
from gterminal.agents.base_agent_service import Job
from gterminal.agents.rpc_parameter_models import GenerateApiParams
from gterminal.agents.rpc_parameter_models import GenerateCodeParams
from gterminal.agents.rpc_parameter_models import GenerateModelsParams
from gterminal.core.rpc.models import SessionContext
from gterminal.core.rpc.patterns import RpcAgentMixin
from gterminal.core.rpc.patterns import rpc_method
from gterminal.utils.rust_extensions import RUST_CORE_AVAILABLE
from gterminal.utils.rust_extensions import EnhancedTtlCache
from gterminal.utils.rust_extensions import RustCore
from gterminal.utils.rust_extensions import test_rust_integration


class CodeGenerationService(BaseAgentService, RpcAgentMixin):
    """Comprehensive code generation service.

    Features:
    - API endpoint generation from specifications
    - Data model generation (Pydantic, SQLAlchemy)
    - Frontend component generation (React, Vue)
    - Test case generation
    - Database migration generation
    - Configuration file generation
    - Boilerplate and scaffolding generation
    - Code refactoring and transformation
    """

    def __init__(self) -> None:
        super().__init__("code_generator", "Automated code generation and scaffolding")

        # Initialize Rust components for high-performance operations
        self.rust_available = RUST_CORE_AVAILABLE
        self.rust_core = None
        self.rust_cache = None

        if self.rust_available:
            try:
                self.rust_core = RustCore()
                # Use Rust cache for code generation templates and results (30 minutes TTL)
                self.rust_cache = EnhancedTtlCache(1800)
                logging.info("Code Generator initialized with Rust acceleration")

                # Test Rust integration for performance metrics
                rust_status = test_rust_integration()
                logging.info(f"Code generator Rust integration: {rust_status}")

            except Exception as e:
                logging.warning(f"Failed to initialize Rust components in CodeGen: {e}")
                self.rust_core = None
                self.rust_cache = None
                self.rust_available = False
        else:
            logging.info("Code Generator using Python fallbacks")

        # Template cache metrics
        self._template_cache_hits = 0
        self._template_cache_misses = 0

        # Code templates for different technologies
        self.templates = {
            "fastapi_endpoint": self._get_fastapi_endpoint_template(),
            "pydantic_model": self._get_pydantic_model_template(),
            "react_component": self._get_react_component_template(),
            "python_test": self._get_python_test_template(),
            "sqlalchemy_model": self._get_sqlalchemy_model_template(),
            "docker_compose": self._get_docker_compose_template(),
            "github_workflow": self._get_github_workflow_template(),
        }

        # Supported languages and frameworks
        self.supported_frameworks = {
            "backend": ["fastapi", "django", "flask", "express", "spring"],
            "frontend": ["react", "vue", "angular", "svelte"],
            "database": ["sqlalchemy", "mongoose", "prisma"],
            "testing": ["pytest", "jest", "junit"],
        }

    def get_required_parameters(self, job_type: str) -> list[str]:
        """Get required parameters for job type."""
        if job_type == "generate_code":
            return ["specification"]
        if job_type == "generate_api":
            return ["api_specification"]
        if job_type == "generate_models":
            return ["model_specification"]
        if job_type == "generate_tests":
            return ["code_path"]
        return []

    async def _execute_job_implementation(self, job: Job) -> dict[str, Any]:
        """Execute code generation job implementation."""
        job_type = job.job_type
        parameters = job.parameters

        if job_type == "generate_code":
            return await self._generate_code(job, parameters["specification"])
        if job_type == "generate_api":
            return await self._generate_api(job, parameters["api_specification"])
        if job_type == "generate_models":
            return await self._generate_models(job, parameters["model_specification"])
        if job_type == "generate_tests":
            return await self._generate_tests(job, parameters["code_path"])
        if job_type == "generate_frontend":
            return await self._generate_frontend(
                job,
                parameters["frontend_specification"],
            )
        if job_type == "generate_migration":
            return await self._generate_migration(
                job,
                parameters["migration_specification"],
            )
        if job_type == "scaffold_project":
            return await self._scaffold_project(
                job,
                parameters["project_specification"],
            )
        msg = f"Unknown job type: {job_type}"
        raise ValueError(msg)

    # JSON RPC 2.0 Compliant Methods

    @rpc_method(
        method_name="generate_code",
        timeout_seconds=300,
        validate_params=True,
        log_performance=True,
    )
    async def generate_code_rpc(
        self,
        params: GenerateCodeParams,
        session: SessionContext | None = None,
    ) -> dict[str, Any]:
        """Generate code from specification with RPC compliance.

        Args:
            params: Code generation parameters
            session: Optional session context

        Returns:
            Code generation result
        """
        job = Job(
            job_id=f"rpc_generate_code_{datetime.now(UTC).timestamp()}",
            job_type="generate_code",
            parameters={"specification": params.specification},
        )

        return await self._generate_code(job, params.specification)

    @rpc_method(
        method_name="generate_api",
        timeout_seconds=300,
        validate_params=True,
        log_performance=True,
    )
    async def generate_api_rpc(
        self,
        params: GenerateApiParams,
        session: SessionContext | None = None,
    ) -> dict[str, Any]:
        """Generate API endpoints with RPC compliance.

        Args:
            params: API generation parameters
            session: Optional session context

        Returns:
            API generation result
        """
        job = Job(
            job_id=f"rpc_generate_api_{datetime.now(UTC).timestamp()}",
            job_type="generate_api",
            parameters={"api_specification": params.api_specification},
        )

        return await self._generate_api(job, params.api_specification)

    @rpc_method(
        method_name="generate_models",
        timeout_seconds=180,
        validate_params=True,
        log_performance=True,
    )
    async def generate_models_rpc(
        self,
        params: GenerateModelsParams,
        session: SessionContext | None = None,
    ) -> dict[str, Any]:
        """Generate data models with RPC compliance.

        Args:
            params: Model generation parameters
            session: Optional session context

        Returns:
            Model generation result
        """
        job = Job(
            job_id=f"rpc_generate_models_{datetime.now(UTC).timestamp()}",
            job_type="generate_models",
            parameters={"model_specification": params.model_specification},
        )

        return await self._generate_models(job, params.model_specification)

    async def _generate_code(
        self,
        job: Job,
        specification: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate code from general specification."""
        job.update_progress(10.0, "Analyzing code specification")

        code_generation_result = {
            "generated_files": [],
            "warnings": [],
            "suggestions": [],
            "total_lines_generated": 0,
        }

        # Parse specification
        job.update_progress(20.0, "Parsing specification")
        parsed_spec = await self._parse_specification(specification)

        # Generate different types of code based on specification
        if "api_endpoints" in parsed_spec:
            job.update_progress(40.0, "Generating API endpoints")
            api_result = await self._generate_api_code(parsed_spec["api_endpoints"])
            code_generation_result["generated_files"].extend(api_result["files"])

        if "data_models" in parsed_spec:
            job.update_progress(60.0, "Generating data models")
            model_result = await self._generate_model_code(parsed_spec["data_models"])
            code_generation_result["generated_files"].extend(model_result["files"])

        if "frontend_components" in parsed_spec:
            job.update_progress(80.0, "Generating frontend components")
            frontend_result = await self._generate_frontend_code(
                parsed_spec["frontend_components"],
            )
            code_generation_result["generated_files"].extend(frontend_result["files"])

        # Write generated files
        job.update_progress(90.0, "Writing generated files")
        await self._write_generated_files(code_generation_result["generated_files"])

        # Calculate metrics
        code_generation_result["total_lines_generated"] = sum(
            len(file["content"].split("\n")) for file in code_generation_result["generated_files"]
        )

        job.update_progress(100.0, "Code generation complete")

        return {
            "specification": specification,
            "generation_result": code_generation_result,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_api(
        self,
        job: Job,
        api_specification: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate API endpoints from specification."""
        job.update_progress(10.0, "Analyzing API specification")

        api_generation_result = {
            "endpoints": [],
            "models": [],
            "generated_files": [],
            "openapi_spec": None,
        }

        # Extract endpoints from specification
        job.update_progress(30.0, "Extracting endpoint definitions")
        endpoints = api_specification.get("endpoints", [])

        # Generate endpoint code
        job.update_progress(50.0, "Generating endpoint code")
        for endpoint in endpoints:
            endpoint_code = await self._generate_endpoint_code(endpoint)
            if endpoint_code:
                api_generation_result["endpoints"].append(endpoint_code)

        # Generate data models
        job.update_progress(70.0, "Generating data models")
        models = api_specification.get("models", [])
        for model in models:
            model_code = await self._generate_model_code_single(model)
            if model_code:
                api_generation_result["models"].append(model_code)

        # Generate OpenAPI specification
        job.update_progress(85.0, "Generating OpenAPI specification")
        api_generation_result["openapi_spec"] = await self._generate_openapi_specification(
            api_specification
        )

        # Create file structure
        job.update_progress(95.0, "Creating file structure")
        api_generation_result["generated_files"] = await self._create_api_file_structure(
            api_generation_result
        )

        job.update_progress(100.0, "API generation complete")

        return {
            "api_specification": api_specification,
            "api_generation": api_generation_result,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_models(
        self,
        job: Job,
        model_specification: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate data models from specification."""
        job.update_progress(10.0, "Analyzing model specification")

        model_generation_result = {
            "pydantic_models": [],
            "sqlalchemy_models": [],
            "generated_files": [],
            "relationships": [],
        }

        models = model_specification.get("models", [])
        model_type = model_specification.get("type", "pydantic")

        # Generate models
        job.update_progress(40.0, f"Generating {model_type} models")
        for model in models:
            if model_type == "pydantic":
                model_code = await self._generate_pydantic_model(model)
                model_generation_result["pydantic_models"].append(model_code)
            elif model_type == "sqlalchemy":
                model_code = await self._generate_sqlalchemy_model(model)
                model_generation_result["sqlalchemy_models"].append(model_code)

        # Generate relationships
        job.update_progress(70.0, "Generating model relationships")
        relationships = model_specification.get("relationships", [])
        for relationship in relationships:
            rel_code = await self._generate_relationship_code(relationship, model_type)
            model_generation_result["relationships"].append(rel_code)

        # Create files
        job.update_progress(90.0, "Creating model files")
        model_generation_result["generated_files"] = await self._create_model_files(
            model_generation_result,
            model_type,
        )

        job.update_progress(100.0, "Model generation complete")

        return {
            "model_specification": model_specification,
            "model_generation": model_generation_result,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_tests(self, job: Job, code_path: str) -> dict[str, Any]:
        """Generate tests for existing code."""
        job.update_progress(10.0, f"Analyzing code at {code_path}")

        test_generation_result = {
            "test_files": [],
            "coverage_targets": [],
            "generated_files": [],
            "test_frameworks": [],
        }

        code_path_obj = Path(code_path)
        if not code_path_obj.exists():
            return {"error": f"Code path does not exist: {code_path}"}

        # Find Python files to test
        job.update_progress(30.0, "Finding code files to test")
        if code_path_obj.is_file():
            python_files = [code_path_obj]
        else:
            python_files = list(code_path_obj.rglob("*.py"))
            # Exclude test files and common directories
            python_files = [
                f
                for f in python_files
                if not any(
                    exclude in str(f)
                    for exclude in ["test_", "_test.py", "__pycache__", "venv", ".git"]
                )
            ]

        # Generate tests for each file
        job.update_progress(50.0, "Generating test cases")
        for i, file_path in enumerate(python_files[:20]):  # Limit to first 20 files
            progress = 50.0 + (40.0 * i / min(len(python_files), 20))
            job.update_progress(progress, f"Generating tests for {file_path.name}")

            test_code = await self._generate_test_for_file(file_path)
            if test_code:
                test_generation_result["test_files"].append(
                    {
                        "source_file": str(file_path),
                        "test_file": str(file_path.parent / f"test_{file_path.name}"),
                        "test_code": test_code,
                    },
                )

        # Create test files
        job.update_progress(95.0, "Writing test files")
        test_generation_result["generated_files"] = await self._write_test_files(
            test_generation_result["test_files"],
        )

        job.update_progress(100.0, "Test generation complete")

        return {
            "code_path": code_path,
            "test_generation": test_generation_result,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _generate_frontend(
        self,
        job: Job,
        frontend_specification: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate frontend components from specification."""
        job.update_progress(10.0, "Analyzing frontend specification")

        frontend_generation_result = {
            "components": [],
            "pages": [],
            "styles": [],
            "generated_files": [],
        }

        framework = frontend_specification.get("framework", "react")
        components = frontend_specification.get("components", [])

        # Generate components
        job.update_progress(40.0, f"Generating {framework} components")
        for component in components:
            component_code = await self._generate_frontend_component(
                component,
                framework,
            )
            if component_code:
                frontend_generation_result["components"].append(component_code)

        # Generate pages
        job.update_progress(70.0, "Generating pages")
        pages = frontend_specification.get("pages", [])
        for page in pages:
            page_code = await self._generate_frontend_page(page, framework)
            if page_code:
                frontend_generation_result["pages"].append(page_code)

        # Create files
        job.update_progress(90.0, "Creating frontend files")
        frontend_generation_result["generated_files"] = await self._create_frontend_files(
            frontend_generation_result, framework
        )

        job.update_progress(100.0, "Frontend generation complete")

        return {
            "frontend_specification": frontend_specification,
            "frontend_generation": frontend_generation_result,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _scaffold_project(
        self,
        job: Job,
        project_specification: dict[str, Any],
    ) -> dict[str, Any]:
        """Scaffold entire project structure."""
        job.update_progress(10.0, "Analyzing project specification")

        scaffold_result = {
            "project_structure": {},
            "generated_files": [],
            "configuration_files": [],
            "documentation_files": [],
        }

        project_type = project_specification.get("type", "fullstack")
        project_name = project_specification.get("name", "new_project")

        # Create project structure
        job.update_progress(30.0, "Creating project structure")
        scaffold_result["project_structure"] = await self._create_project_structure(
            project_name,
            project_type,
        )

        # Generate configuration files
        job.update_progress(50.0, "Generating configuration files")
        config_files = await self._generate_config_files(project_specification)
        scaffold_result["configuration_files"] = config_files

        # Generate boilerplate code
        job.update_progress(70.0, "Generating boilerplate code")
        boilerplate_files = await self._generate_boilerplate_code(project_specification)
        scaffold_result["generated_files"].extend(boilerplate_files)

        # Generate documentation
        job.update_progress(85.0, "Generating documentation")
        doc_files = await self._generate_project_documentation(project_specification)
        scaffold_result["documentation_files"] = doc_files

        # Write all files
        job.update_progress(95.0, "Writing project files")
        await self._write_project_files(scaffold_result, project_name)

        job.update_progress(100.0, "Project scaffolding complete")

        return {
            "project_specification": project_specification,
            "scaffold_result": scaffold_result,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    async def _parse_specification(
        self,
        specification: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse and normalize specification."""
        return {
            "api_endpoints": specification.get("api_endpoints", []),
            "data_models": specification.get("data_models", []),
            "frontend_components": specification.get("frontend_components", []),
            "tests": specification.get("tests", []),
            "configuration": specification.get("configuration", {}),
        }

    async def _generate_api_code(
        self,
        endpoints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate API endpoint code."""
        files: list[Any] = []

        for endpoint in endpoints:
            code = await self._generate_endpoint_code(endpoint)
            if code:
                files.append(
                    {
                        "path": f"api/{endpoint['name']}.py",
                        "content": code,
                        "type": "endpoint",
                    },
                )

        return {"files": files}

    async def _generate_endpoint_code(self, endpoint: dict[str, Any]) -> str | None:
        """Generate code for a single API endpoint with Rust acceleration."""
        # Generate cache key for this specific endpoint
        cache_key = f"endpoint_code:{endpoint.get('method', 'GET')}:{endpoint.get('path', '/')}"

        # Try Rust cache first for performance
        if self.rust_cache:
            try:
                cached_code = self.rust_cache.get(cache_key)
                if cached_code:
                    self._template_cache_hits += 1
                    logging.info(f"Using cached endpoint code for {cache_key}")
                    return cached_code
                else:
                    self._template_cache_misses += 1
            except Exception as e:
                logging.warning(f"Rust cache error in endpoint generation: {e}")

        template = self.templates["fastapi_endpoint"]

        # Use AI to generate endpoint implementation
        prompt = f"""Generate a FastAPI endpoint implementation for:

Endpoint: {endpoint.get("method", "GET")} {endpoint.get("path", "/")}
Description: {endpoint.get("description", "API endpoint")}
Parameters: {endpoint.get("parameters", [])}
Response: {endpoint.get("response", {})}

Generate complete FastAPI endpoint code with:
1. Proper type hints
2. Request/response models
3. Error handling
4. Documentation strings
5. Validation

Return only the Python code."""

        ai_code = await self.generate_with_gemini(prompt, "code_generation")

        generated_code = None
        if ai_code and len(ai_code) > 50:
            generated_code = ai_code
        else:
            # Fallback to template
            generated_code = template.format(
                endpoint_name=endpoint.get("name", "example"),
                method=endpoint.get("method", "GET").lower(),
                path=endpoint.get("path", "/"),
                description=endpoint.get("description", "API endpoint"),
            )

        # Cache the result using Rust cache for future requests
        if generated_code and self.rust_cache:
            try:
                # Cache for 30 minutes (1800 seconds)
                self.rust_cache.set_with_ttl(cache_key, generated_code, 1800)
                logging.info(f"Cached endpoint code for {cache_key}")
            except Exception as e:
                logging.warning(f"Failed to cache endpoint code: {e}")

        return generated_code

    async def _generate_model_code_single(self, model: dict[str, Any]) -> str | None:
        """Generate code for a single data model."""
        template = self.templates["pydantic_model"]

        # Use AI to generate model
        prompt = f"""Generate a Pydantic model for:

Model Name: {model.get("name", "ExampleModel")}
Fields: {model.get("fields", [])}
Description: {model.get("description", "Data model")}

Generate complete Pydantic model with:
1. Proper field types and validation
2. Optional/required field handling
3. Field descriptions
4. Example values
5. Model configuration

Return only the Python code."""

        ai_code = await self.generate_with_gemini(prompt, "code_generation")

        if ai_code and len(ai_code) > 30:
            return ai_code

        # Fallback to template
        fields_code = ""
        for field in model.get("fields", []):
            field_name = field.get("name", "field")
            field_type = field.get("type", "str")
            field_desc = field.get("description", "")
            fields_code += f"    {field_name}: {field_type}  # {field_desc}\n"

        return template.format(
            model_name=model.get("name", "ExampleModel"),
            fields=fields_code,
            description=model.get("description", "Data model"),
        )

    async def _generate_pydantic_model(self, model: dict[str, Any]) -> dict[str, Any]:
        """Generate Pydantic model with metadata."""
        code = await self._generate_model_code_single(model)

        return {
            "name": model.get("name", "ExampleModel"),
            "code": code,
            "fields": model.get("fields", []),
            "type": "pydantic",
        }

    async def _generate_sqlalchemy_model(self, model: dict[str, Any]) -> dict[str, Any]:
        """Generate SQLAlchemy model with metadata."""
        template = self.templates["sqlalchemy_model"]

        # Generate SQLAlchemy model code
        prompt = f"""Generate a SQLAlchemy model for:

Model Name: {model.get("name", "ExampleModel")}
Table Name: {model.get("table_name", model.get("name", "example").lower())}
Fields: {model.get("fields", [])}

Generate complete SQLAlchemy model with:
1. Proper column types and constraints
2. Primary keys and foreign keys
3. Relationships if specified
4. Table name configuration
5. Proper imports

Return only the Python code."""

        ai_code = await self.generate_with_gemini(prompt, "code_generation")

        if ai_code and "class" in ai_code:
            code = ai_code
        else:
            # Fallback to template
            fields_code = ""
            for field in model.get("fields", []):
                field_name = field.get("name", "field")
                field_type = self._convert_to_sqlalchemy_type(field.get("type", "str"))
                constraints = field.get("constraints", [])

                constraint_str = ""
                if "primary_key" in constraints:
                    constraint_str += ", primary_key=True"
                if "nullable" in constraints and not constraints["nullable"]:
                    constraint_str += ", nullable=False"

                fields_code += f"    {field_name} = Column({field_type}{constraint_str})\n"

            code = template.format(
                model_name=model.get("name", "ExampleModel"),
                table_name=model.get(
                    "table_name",
                    model.get("name", "example").lower(),
                ),
                fields=fields_code,
            )

        return {
            "name": model.get("name", "ExampleModel"),
            "code": code,
            "fields": model.get("fields", []),
            "type": "sqlalchemy",
        }

    async def _generate_test_for_file(self, file_path: Path) -> str | None:
        """Generate test code for a Python file."""
        content = self.safe_file_read(file_path)
        if not content:
            return None

        # Parse the Python file to extract functions and classes
        try:
            tree = ast.parse(content)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        except Exception:
            return None

        if not functions and not classes:
            return None

        # Use AI to generate comprehensive tests
        prompt = f"""Generate comprehensive pytest tests for this Python file:

File: {file_path.name}
Functions: {functions}
Classes: {classes}

Code sample:
```python
{content[:2000]}  # First 2000 chars
```

Generate complete test file with:
1. Import statements
2. Test fixtures if needed
3. Test functions for each public function/method
4. Edge case testing
5. Mock usage where appropriate
6. Proper assertions

Return only the Python test code."""

        ai_test_code = await self.generate_with_gemini(prompt, "code_generation")

        if ai_test_code and "def test_" in ai_test_code:
            return ai_test_code

        # Fallback basic test generation
        test_code = f"""import pytest
from {file_path.stem} import *

"""

        for func in functions[:10]:  # Limit to first 10 functions
            test_code += f"""
def test_{func}():
    \"\"\"Test {func} function.\"\"\"
    # TODO: Implement test for {func}
    assert True  # Placeholder

"""

        return test_code

    async def _generate_frontend_component(
        self,
        component: dict[str, Any],
        framework: str,
    ) -> dict[str, Any] | None:
        """Generate frontend component code."""
        if framework.lower() == "react":
            return await self._generate_react_component(component)
        if framework.lower() == "vue":
            return await self._generate_vue_component(component)
        return None

    async def _generate_react_component(
        self,
        component: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate React component."""
        template = self.templates["react_component"]

        # Use AI to generate React component
        prompt = f"""Generate a React functional component for:

Component Name: {component.get("name", "ExampleComponent")}
Props: {component.get("props", [])}
Description: {component.get("description", "React component")}
Features: {component.get("features", [])}

Generate complete React component with:
1. TypeScript interfaces for props
2. Proper state management with hooks
3. Event handlers
4. JSX structure
5. CSS-in-JS or className usage
6. Component documentation

Return only the React component code."""

        ai_code = await self.generate_with_gemini(prompt, "code_generation")

        if ai_code and ("function" in ai_code or "const" in ai_code):
            code = ai_code
        else:
            # Fallback to template
            props_interface = ""
            if component.get("props"):
                props_list: list[Any] = []
                for prop in component.get("props", []):
                    prop_name = prop.get("name", "prop")
                    prop_type = prop.get("type", "string")
                    optional = "?" if prop.get("optional", False) else ""
                    props_list.append(f"  {prop_name}{optional}: {prop_type}")
                props_interface = (
                    f"interface {component.get('name', 'Example')}Props {{\n"
                    + "\n".join(props_list)
                    + "\n}"
                )

            code = template.format(
                component_name=component.get("name", "ExampleComponent"),
                props_interface=props_interface,
                description=component.get("description", "React component"),
            )

        return {
            "name": component.get("name", "ExampleComponent"),
            "code": code,
            "framework": "react",
            "type": "component",
        }

    def _convert_to_sqlalchemy_type(self, python_type: str) -> str:
        """Convert Python type to SQLAlchemy column type."""
        type_mapping = {
            "str": "String",
            "int": "Integer",
            "float": "Float",
            "bool": "Boolean",
            "datetime": "DateTime",
            "date": "Date",
            "text": "Text",
            "json": "JSON",
        }
        return type_mapping.get(python_type, "String")

    # Template methods
    def _get_fastapi_endpoint_template(self) -> str:
        """Get FastAPI endpoint template."""
        return '''from fastapi import APIRouter, HTTPException, Depends
from typing import Any, Optional, Union
from pydantic import BaseModel

router = APIRouter()

class {endpoint_name}Response(BaseModel):
    """Response model for {endpoint_name} endpoint."""
    message: str
    data: dict[str, Any] = None

@router.{method}("{path}")
async def {endpoint_name}() -> {endpoint_name}Response:
    """
    {description}
    """
    try:
        # TODO: Implement endpoint logic
        return {endpoint_name}Response(
            message="Success",
            data={{"example": "data"}}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''

    def _get_pydantic_model_template(self) -> str:
        """Get Pydantic model template."""
        return '''from pydantic import BaseModel, Field
from typing import
from datetime import datetime, timezone

class {model_name}(BaseModel):
    """
    {description}
    """
{fields}

    class Config:
        from_attributes = True
        json_encoders = {{
            datetime: lambda v: v.isoformat()
        }}
'''

    def _get_react_component_template(self) -> str:
        """Get React component template."""
        return """import React from 'react';

{props_interface}

/**
 * {description}
 */
const {component_name}: React.FC<{component_name}Props> = (props) => {{
  return (
    <div className="{component_name}">
      <h2>{component_name}</h2>
      <p>Component implementation goes here</p>
    </div>
  );
}};

export default {component_name};
"""

    def _get_python_test_template(self) -> str:
        """Get Python test template."""
        return '''import pytest
from unittest.mock import Mock, patch

def test_example() -> None:
    """Test example function."""
    # Arrange

    # Act

    # Assert
    assert True

@pytest.fixture
def sample_data() -> None:
    """Sample test data."""
    return {{"key": "value"}}
'''

    def _get_sqlalchemy_model_template(self) -> str:
        """Get SQLAlchemy model template."""
        return '''from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class {model_name}(Base):
    """
    SQLAlchemy model for {model_name}
    """
    __tablename__ = '{table_name}'

{fields}
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
'''

    def _get_docker_compose_template(self) -> str:
        """Get Docker Compose template."""
        return """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/dbname
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: dbname
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
"""

    def _get_github_workflow_template(self) -> str:
        """Get GitHub workflow template."""
        return """name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
"""

    # Additional helper methods for code generation
    async def _write_generated_files(self, files: list[dict[str, Any]]) -> None:
        """Write generated files to disk."""
        for file_info in files:
            file_path = Path(file_info["path"])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.safe_file_write(file_path, file_info["content"])

    async def _generate_model_code(
        self,
        models: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate model code from list of models."""
        files: list[Any] = []
        for model in models:
            code = await self._generate_model_code_single(model)
            if code:
                files.append(
                    {
                        "path": f"models/{model.get('name', 'model').lower()}.py",
                        "content": code,
                        "type": "model",
                    },
                )
        return {"files": files}

    async def _generate_frontend_code(
        self,
        components: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate frontend component code."""
        files: list[Any] = []
        for component in components:
            code = await self._generate_react_component(component)
            if code:
                files.append(
                    {
                        "path": f"components/{component.get('name', 'Component')}.tsx",
                        "content": code["code"],
                        "type": "component",
                    },
                )
        return {"files": files}

    async def _generate_openapi_specification(
        self,
        api_spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate OpenAPI specification."""
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": api_spec.get("title", "Generated API"),
                "version": api_spec.get("version", "1.0.0"),
                "description": api_spec.get("description", "Auto-generated API"),
            },
            "paths": {},
            "components": {"schemas": {}},
        }

        # Add paths from endpoints
        for endpoint in api_spec.get("endpoints", []):
            path = endpoint.get("path", "/")
            method = endpoint.get("method", "GET").lower()

            if path not in openapi_spec["paths"]:
                openapi_spec["paths"][path] = {}

            openapi_spec["paths"][path][method] = {
                "summary": endpoint.get("description", "API endpoint"),
                "responses": {"200": {"description": "Success"}},
            }

        return openapi_spec

    async def _create_api_file_structure(
        self,
        api_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Create API file structure."""
        files: list[Any] = []

        # Main API router file
        router_content = """from fastapi import APIRouter

router = APIRouter()

# Import endpoint routers here
"""
        files.append({"path": "api/__init__.py", "content": "", "type": "init"})

        files.append(
            {"path": "api/router.py", "content": router_content, "type": "router"},
        )

        # Individual endpoint files
        for endpoint in api_result["endpoints"]:
            files.append(
                {
                    "path": f"api/endpoints/{endpoint.get('name', 'endpoint')}.py",
                    "content": endpoint,
                    "type": "endpoint",
                },
            )

        return files

    async def _generate_relationship_code(
        self,
        relationship: dict[str, Any],
        model_type: str,
    ) -> str:
        """Generate relationship code between models."""
        if model_type == "sqlalchemy":
            return f"""# Relationship: {relationship.get("name", "relationship")}
# {relationship.get("description", "Model relationship")}
"""
        return ""

    async def _create_model_files(
        self,
        model_result: dict[str, Any],
        model_type: str,
    ) -> list[dict[str, Any]]:
        """Create model files."""
        files: list[Any] = []

        if model_type == "pydantic":
            for model in model_result["pydantic_models"]:
                files.append(
                    {
                        "path": f"models/{model['name'].lower()}.py",
                        "content": model["code"],
                        "type": "pydantic_model",
                    },
                )
        elif model_type == "sqlalchemy":
            for model in model_result["sqlalchemy_models"]:
                files.append(
                    {
                        "path": f"models/{model['name'].lower()}.py",
                        "content": model["code"],
                        "type": "sqlalchemy_model",
                    },
                )

        return files

    async def _write_test_files(self, test_files: list[dict[str, Any]]) -> list[str]:
        """Write test files to disk."""
        written_files: list[Any] = []

        for test_info in test_files:
            test_path = Path(test_info["test_file"])
            test_path.parent.mkdir(parents=True, exist_ok=True)

            if self.safe_file_write(test_path, test_info["test_code"]):
                written_files.append(str(test_path))

        return written_files

    async def _generate_frontend_page(
        self,
        page: dict[str, Any],
        framework: str,
    ) -> dict[str, Any] | None:
        """Generate frontend page code."""
        if framework.lower() == "react":
            # Generate React page component
            template = """import React from 'react';

const {page_name}: React.FC = () => {{
  return (
    <div className="{page_class}">
      <h1>{page_title}</h1>
      <p>{page_description}</p>
    </div>
  );
}};

export default {page_name};
"""

            code = template.format(
                page_name=page.get("name", "Page"),
                page_class=page.get("name", "page").lower(),
                page_title=page.get("title", "Page Title"),
                page_description=page.get("description", "Page description"),
            )

            return {"name": page.get("name", "Page"), "code": code, "type": "page"}

        return None

    async def _create_frontend_files(
        self,
        frontend_result: dict[str, Any],
        framework: str,
    ) -> list[dict[str, Any]]:
        """Create frontend files."""
        files: list[Any] = []

        # Component files
        for component in frontend_result["components"]:
            ext = "tsx" if framework == "react" else "vue"
            files.append(
                {
                    "path": f"src/components/{component['name']}.{ext}",
                    "content": component["code"],
                    "type": "component",
                },
            )

        # Page files
        for page in frontend_result["pages"]:
            ext = "tsx" if framework == "react" else "vue"
            files.append(
                {
                    "path": f"src/pages/{page['name']}.{ext}",
                    "content": page["code"],
                    "type": "page",
                },
            )

        return files

    async def _create_project_structure(
        self,
        project_name: str,
        project_type: str,
    ) -> dict[str, Any]:
        """Create project directory structure."""
        structure = {"directories": [], "files": []}

        if project_type == "fullstack":
            structure["directories"] = [
                f"{project_name}/backend/app",
                f"{project_name}/backend/app/api",
                f"{project_name}/backend/app/models",
                f"{project_name}/backend/app/services",
                f"{project_name}/backend/tests",
                f"{project_name}/frontend/src",
                f"{project_name}/frontend/src/components",
                f"{project_name}/frontend/src/pages",
                f"{project_name}/frontend/public",
                f"{project_name}/docs",
            ]
        elif project_type == "backend":
            structure["directories"] = [
                f"{project_name}/app",
                f"{project_name}/app/api",
                f"{project_name}/app/models",
                f"{project_name}/app/services",
                f"{project_name}/tests",
                f"{project_name}/docs",
            ]

        return structure

    async def _generate_config_files(
        self,
        project_spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate configuration files."""
        config_files: list[Any] = []

        # Requirements.txt for Python projects
        if project_spec.get("backend_framework") in ["fastapi", "django", "flask"]:
            requirements = """fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
alembic==1.13.0
pytest==7.4.3
pytest-cov==4.1.0
"""
            config_files.append(
                {
                    "path": "requirements.txt",
                    "content": requirements,
                    "type": "dependencies",
                },
            )

        # Package.json for frontend
        if project_spec.get("frontend_framework") in ["react", "vue", "angular"]:
            package_json = {
                "name": project_spec.get("name", "frontend-app"),
                "version": "1.0.0",
                "dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0"},
                "scripts": {
                    "start": "react-scripts start",
                    "build": "react-scripts build",
                    "test": "react-scripts test",
                },
            }

            config_files.append(
                {
                    "path": "frontend/package.json",
                    "content": json.dumps(package_json, indent=2),
                    "type": "package_config",
                },
            )

        # Docker Compose
        config_files.append(
            {
                "path": "docker-compose.yml",
                "content": self.templates["docker_compose"],
                "type": "docker",
            },
        )

        return config_files

    async def _generate_boilerplate_code(
        self,
        project_spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate boilerplate code files."""
        files: list[Any] = []

        # Main application file
        if project_spec.get("backend_framework") == "fastapi":
            main_content = """from fastapi import FastAPI
from gterminal.api.router import router

app = FastAPI(title="Generated API", version="1.0.0")

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
            files.append(
                {"path": "app/main.py", "content": main_content, "type": "main"},
            )

        return files

    async def _generate_project_documentation(
        self,
        project_spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate project documentation files."""
        doc_files: list[Any] = []

        # README.md
        readme_content = f"""# {project_spec.get("name", "Project")}

{project_spec.get("description", "Generated project description")}

## Setup

1. Install dependencies
2. Configure environment variables
3. Run the application

## API Documentation

API documentation is available at `/docs` when running the application.

## Testing

Run tests with:
```bash
pytest
```
"""

        doc_files.append(
            {"path": "README.md", "content": readme_content, "type": "documentation"},
        )

        return doc_files

    async def _write_project_files(
        self,
        scaffold_result: dict[str, Any],
        project_name: str,
    ) -> None:
        """Write all project files to disk."""
        # Create directories
        for directory in scaffold_result["project_structure"]["directories"]:
            Path(directory).mkdir(parents=True, exist_ok=True)

        # Write all files
        all_files = (
            scaffold_result["generated_files"]
            + scaffold_result["configuration_files"]
            + scaffold_result["documentation_files"]
        )

        for file_info in all_files:
            file_path = Path(project_name) / file_info["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.safe_file_write(file_path, file_info["content"])

    async def _generate_vue_component(
        self,
        component: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate Vue.js component."""
        template = """<template>
  <div class="{component_class}">
    <h2>{{ title }}</h2>
    <p>{description}</p>
  </div>
</template>

<script>
export default {{
  name: '{component_name}',
  props: {{
    title: {{
      type: String,
      default: '{component_name}'
    }}
  }},
  data() {{
    return {{
      // Component data
    }}
  }},
  methods: {{
    // Component methods
  }}
}}
</script>

<style scoped>
.{component_class} {{
  padding: 20px;
}}
</style>
"""

        code = template.format(
            component_name=component.get("name", "ExampleComponent"),
            component_class=component.get("name", "example").lower(),
            description=component.get("description", "Vue component"),
        )

        return {
            "name": component.get("name", "ExampleComponent"),
            "code": code,
            "framework": "vue",
            "type": "component",
        }

    async def _generate_migration(
        self,
        job: Job,
        migration_spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate database migration."""
        job.update_progress(10.0, "Analyzing migration specification")

        migration_result = {
            "migration_file": "",
            "migration_code": "",
            "rollback_code": "",
        }

        # Generate Alembic migration
        migration_name = migration_spec.get("name", "auto_migration")

        migration_template = '''"""Add {migration_name}

Revision ID: {revision_id}
Revises:
Create Date: {create_date}

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '{revision_id}'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    {upgrade_code}

def downgrade() -> None:
    {downgrade_code}
'''

        # Generate migration code based on specification
        changes = migration_spec.get("changes", [])
        upgrade_statements: list[Any] = []
        downgrade_statements: list[Any] = []

        for change in changes:
            if change["type"] == "create_table":
                table_name = change["table_name"]
                upgrade_statements.append(f"op.create_table('{table_name}', ...)")
                downgrade_statements.append(f"op.drop_table('{table_name}')")

        import uuid

        migration_code = migration_template.format(
            migration_name=migration_name,
            revision_id=str(uuid.uuid4())[:12],
            create_date=datetime.now(UTC).isoformat(),
            upgrade_code=("\n    ".join(upgrade_statements) if upgrade_statements else "pass"),
            downgrade_code=(
                "\n    ".join(downgrade_statements) if downgrade_statements else "pass"
            ),
        )

        migration_result["migration_code"] = migration_code
        migration_result["migration_file"] = f"alembic/versions/{migration_name}.py"

        job.update_progress(100.0, "Migration generation complete")

        return {
            "migration_specification": migration_spec,
            "migration_result": migration_result,
            "generated_at": job.started_at.isoformat() if job.started_at else None,
        }

    def register_tools(self) -> None:
        """Register MCP tools for code generator."""

        @self.mcp.tool()
        async def generate_code(specification: dict[str, Any]) -> dict[str, Any]:
            """Generate code from general specification."""
            if not self.validate_job_parameters(
                "generate_code",
                {"specification": specification},
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("generate_code", {"specification": specification})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def generate_api(api_specification: dict[str, Any]) -> dict[str, Any]:
            """Generate API endpoints from specification."""
            if not self.validate_job_parameters(
                "generate_api",
                {"api_specification": api_specification},
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job(
                "generate_api",
                {"api_specification": api_specification},
            )
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def generate_models(
            model_specification: dict[str, Any],
        ) -> dict[str, Any]:
            """Generate data models from specification."""
            if not self.validate_job_parameters(
                "generate_models",
                {"model_specification": model_specification},
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job(
                "generate_models",
                {"model_specification": model_specification},
            )
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def generate_tests(code_path: str) -> dict[str, Any]:
            """Generate tests for existing code."""
            if not self.validate_job_parameters(
                "generate_tests",
                {"code_path": code_path},
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job("generate_tests", {"code_path": code_path})
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def scaffold_project(
            project_specification: dict[str, Any],
        ) -> dict[str, Any]:
            """Scaffold entire project structure."""
            if not self.validate_job_parameters(
                "scaffold_project",
                {"project_specification": project_specification},
            ):
                return self.create_error_response("Invalid parameters")

            job_id = self.create_job(
                "scaffold_project",
                {"project_specification": project_specification},
            )
            return await self.execute_job_async(job_id)

        @self.mcp.tool()
        async def get_supported_frameworks() -> dict[str, Any]:
            """Get supported frameworks and templates."""
            return self.create_success_response(
                {
                    "supported_frameworks": self.supported_frameworks,
                    "available_templates": list(self.templates.keys()),
                    "agent_stats": self.get_agent_stats(),
                },
            )


# Create global instance
code_generation_service = CodeGenerationService()
