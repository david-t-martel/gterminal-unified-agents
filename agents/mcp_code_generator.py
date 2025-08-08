#!/usr/bin/env python3
"""MCP Server for Code Generation Agent.

Provides MCP server interface for the Code Generation Agent service.
"""

import logging
from typing import Any

from gterminal.agents.code_generation_agent import CodeGenerationService
from mcp.server import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("code-generator")

# Initialize the agent service
agent_service = CodeGenerationService()


@mcp.tool()
async def generate_code(
    specification: str,
    language: str = "python",
    framework: str | None = None,
    include_tests: bool = True,
    include_docs: bool = True,
) -> dict[str, Any]:
    """Generate code from specifications.

    Args:
        specification: Code specifications and requirements
        language: Programming language ('python', 'javascript', 'typescript', 'java', 'go', 'rust')
        framework: Framework to use (e.g., 'fastapi', 'react', 'django')
        include_tests: Whether to generate test cases
        include_docs: Whether to include documentation

    Returns:
        Generated code with tests and documentation

    """
    try:
        request = {
            "specification": specification,
            "language": language,
            "framework": framework,
            "include_tests": include_tests,
            "include_docs": include_docs,
        }

        return await agent_service.generate_code(request)
    except Exception as e:
        logger.exception(f"Code generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def generate_project(
    project_name: str,
    project_type: str,
    language: str = "python",
    include_ci: bool = True,
    include_docker: bool = True,
) -> dict[str, Any]:
    """Generate a complete project scaffold.

    Args:
        project_name: Name of the project
        project_type: Type of project ('web-api', 'cli-tool', 'library', 'web-app')
        language: Programming language
        include_ci: Whether to include CI/CD configuration
        include_docker: Whether to include Docker configuration

    Returns:
        Generated project structure and files

    """
    try:
        request = {
            "project_name": project_name,
            "project_type": project_type,
            "language": language,
            "include_ci": include_ci,
            "include_docker": include_docker,
        }

        return await agent_service.generate_project_scaffold(request)
    except Exception as e:
        logger.exception(f"Project generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def generate_api_endpoint(
    specification: str,
    framework: str = "fastapi",
    include_validation: bool = True,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Generate API endpoint implementation.

    Args:
        specification: Endpoint specifications (method, path, request/response)
        framework: API framework ('fastapi', 'flask', 'django', 'express')
        include_validation: Whether to include input validation
        include_tests: Whether to generate tests

    Returns:
        Generated API endpoint code

    """
    try:
        request = {
            "specification": specification,
            "generation_type": "api-endpoint",
            "framework": framework,
            "include_validation": include_validation,
            "include_tests": include_tests,
        }

        return await agent_service.generate_api_endpoint(request)
    except Exception as e:
        logger.exception(f"API endpoint generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def generate_data_model(
    specification: str,
    model_type: str = "pydantic",
    include_validation: bool = True,
    include_serialization: bool = True,
) -> dict[str, Any]:
    """Generate data model classes.

    Args:
        specification: Data model specifications
        model_type: Model type ('pydantic', 'sqlalchemy', 'django-orm', 'typescript')
        include_validation: Whether to include validation rules
        include_serialization: Whether to include serialization methods

    Returns:
        Generated data model code

    """
    try:
        request = {
            "specification": specification,
            "generation_type": "data-model",
            "model_type": model_type,
            "include_validation": include_validation,
            "include_serialization": include_serialization,
        }

        return await agent_service.generate_data_model(request)
    except Exception as e:
        logger.exception(f"Data model generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def generate_tests(
    code_path: str,
    test_framework: str = "pytest",
    coverage_target: int = 80,
    include_edge_cases: bool = True,
) -> dict[str, Any]:
    """Generate test cases for existing code.

    Args:
        code_path: Path to the code to test
        test_framework: Test framework ('pytest', 'unittest', 'jest', 'mocha')
        coverage_target: Target code coverage percentage
        include_edge_cases: Whether to include edge case tests

    Returns:
        Generated test code

    """
    try:
        request = {
            "code_path": code_path,
            "generation_type": "tests",
            "test_framework": test_framework,
            "coverage_target": coverage_target,
            "include_edge_cases": include_edge_cases,
        }

        return await agent_service.generate_tests(request)
    except Exception as e:
        logger.exception(f"Test generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def generate_frontend_component(
    specification: str,
    framework: str = "react",
    styling: str = "css",
    include_tests: bool = True,
) -> dict[str, Any]:
    """Generate frontend component code.

    Args:
        specification: Component specifications and requirements
        framework: Frontend framework ('react', 'vue', 'angular', 'svelte')
        styling: Styling approach ('css', 'sass', 'styled-components', 'tailwind')
        include_tests: Whether to generate component tests

    Returns:
        Generated component code

    """
    try:
        request = {
            "specification": specification,
            "generation_type": "frontend-component",
            "framework": framework,
            "styling": styling,
            "include_tests": include_tests,
        }

        return await agent_service.generate_frontend_component(request)
    except Exception as e:
        logger.exception(f"Component generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def generate_migration(
    from_version: str,
    to_version: str,
    migration_type: str = "database",
    rollback_script: bool = True,
) -> dict[str, Any]:
    """Generate migration scripts.

    Args:
        from_version: Source version or state
        to_version: Target version or state
        migration_type: Type of migration ('database', 'api', 'data')
        rollback_script: Whether to include rollback script

    Returns:
        Migration scripts with rollback options

    """
    try:
        request = {
            "from_version": from_version,
            "to_version": to_version,
            "generation_type": "migration",
            "migration_type": migration_type,
            "rollback_script": rollback_script,
        }

        return await agent_service.generate_migration(request)
    except Exception as e:
        logger.exception(f"Migration generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def get_generation_status(job_id: str) -> dict[str, Any]:
    """Get the status of an ongoing generation job.

    Args:
        job_id: The job ID to check

    Returns:
        Job status and progress information

    """
    try:
        return await agent_service.get_job_status(job_id)
    except Exception as e:
        logger.exception(f"Failed to get job status: {e}")
        return {"status": "error", "error": str(e)}


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
