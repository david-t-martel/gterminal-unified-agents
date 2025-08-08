"""Pydantic parameter models for JSON RPC 2.0 compliance.

This module defines parameter models for all agent methods that will be
converted to RPC compliance, providing type safety and automatic validation.
"""

from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import validator


# Code Generation Agent Parameters
class GenerateCodeParams(BaseModel):
    """Parameters for code generation tasks."""

    specification: dict[str, Any] = Field(description="Code generation specification")
    language: str = Field(default="python", description="Target programming language")
    template_name: str | None = Field(default=None, description="Template to use for generation")
    output_directory: str | None = Field(
        default=None, description="Directory to write generated code"
    )
    include_tests: bool = Field(default=True, description="Whether to generate test files")

    @validator("specification")
    def validate_specification(cls, v):
        if not isinstance(v, dict) or not v:
            raise ValueError("Specification must be a non-empty dictionary")
        return v


class GenerateApiParams(BaseModel):
    """Parameters for API generation tasks."""

    api_specification: dict[str, Any] = Field(description="API specification")
    framework: str = Field(default="fastapi", description="API framework to use")
    include_openapi: bool = Field(default=True, description="Generate OpenAPI documentation")
    authentication_type: str | None = Field(default=None, description="Authentication method")

    @validator("api_specification")
    def validate_api_specification(cls, v):
        if not isinstance(v, dict) or not v:
            raise ValueError("API specification must be a non-empty dictionary")
        return v


class GenerateModelsParams(BaseModel):
    """Parameters for data model generation."""

    model_specification: dict[str, Any] = Field(description="Model specification")
    model_type: str = Field(default="pydantic", description="Type of models to generate")
    include_relationships: bool = Field(default=True, description="Include model relationships")
    validation_rules: dict[str, Any] | None = Field(default=None, description="Validation rules")


# Workspace Analyzer Agent Parameters
class AnalyzeProjectParams(BaseModel):
    """Parameters for project analysis."""

    project_path: str = Field(description="Path to the project to analyze")
    depth: int = Field(default=3, ge=1, le=10, description="Analysis depth")
    include_patterns: list[str] = Field(
        default_factory=lambda: ["*.py", "*.js", "*.ts"],
        description="File patterns to include",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["node_modules/*", "venv/*", "__pycache__/*"],
        description="File patterns to exclude",
    )
    analyze_dependencies: bool = Field(default=True, description="Analyze project dependencies")
    analyze_security: bool = Field(default=True, description="Perform security analysis")

    @validator("project_path")
    def validate_project_path(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Project path must be a non-empty string")
        return v


class AnalyzeDependenciesParams(BaseModel):
    """Parameters for dependency analysis."""

    project_path: str = Field(description="Path to the project")
    include_dev_dependencies: bool = Field(
        default=False, description="Include development dependencies"
    )
    check_vulnerabilities: bool = Field(default=True, description="Check for known vulnerabilities")
    suggest_updates: bool = Field(default=True, description="Suggest dependency updates")


class AnalyzeArchitectureParams(BaseModel):
    """Parameters for architecture analysis."""

    project_path: str = Field(description="Path to the project")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    generate_diagrams: bool = Field(default=False, description="Generate architecture diagrams")
    include_metrics: bool = Field(default=True, description="Include code metrics")


# Master Architect Agent Parameters
class DesignSystemParams(BaseModel):
    """Parameters for system design tasks."""

    requirements: dict[str, Any] = Field(description="System requirements specification")
    architecture_style: str | None = Field(default=None, description="Preferred architecture style")
    scale_requirements: dict[str, Any] | None = Field(
        default=None, description="Scalability requirements"
    )
    technology_constraints: list[str] | None = Field(
        default=None, description="Technology constraints"
    )

    @validator("requirements")
    def validate_requirements(cls, v):
        if not isinstance(v, dict) or not v:
            raise ValueError("Requirements must be a non-empty dictionary")
        return v


class RecommendTechnologiesParams(BaseModel):
    """Parameters for technology recommendation."""

    project_type: str = Field(description="Type of project (web, mobile, desktop, etc.)")
    requirements: dict[str, Any] = Field(description="Technical requirements")
    team_expertise: list[str] | None = Field(
        default=None, description="Team's technology expertise"
    )
    budget_constraints: str | None = Field(default=None, description="Budget constraints")
    timeline: str | None = Field(default=None, description="Project timeline")


class AnalyzeArchitectureParams(BaseModel):
    """Parameters for architecture analysis by master architect."""

    project_path: str = Field(description="Path to the project")
    focus_areas: list[str] | None = Field(
        default=None, description="Specific areas to focus analysis on"
    )
    include_recommendations: bool = Field(
        default=True, description="Include improvement recommendations"
    )
    generate_report: bool = Field(default=True, description="Generate detailed report")


# Documentation Generator Agent Parameters
class GenerateApiDocsParams(BaseModel):
    """Parameters for API documentation generation."""

    project_path: str = Field(description="Path to the project")
    output_format: str = Field(default="markdown", description="Output format for documentation")
    include_examples: bool = Field(default=True, description="Include usage examples")
    auto_detect_endpoints: bool = Field(default=True, description="Auto-detect API endpoints")


class GenerateReadmeParams(BaseModel):
    """Parameters for README generation."""

    project_path: str = Field(description="Path to the project")
    include_badges: bool = Field(default=True, description="Include status badges")
    include_installation: bool = Field(
        default=True, description="Include installation instructions"
    )
    include_examples: bool = Field(default=True, description="Include usage examples")
    style: str = Field(default="comprehensive", description="Documentation style")


class GenerateCodeDocsParams(BaseModel):
    """Parameters for code documentation generation."""

    project_path: str = Field(description="Path to the project")
    output_directory: str | None = Field(
        default=None, description="Output directory for documentation"
    )
    format_type: str = Field(default="sphinx", description="Documentation format")
    include_private: bool = Field(default=False, description="Document private methods")


# Code Review Agent Parameters
class ReviewFileParams(BaseModel):
    """Parameters for file review tasks."""

    file_path: str = Field(description="Path to file to review")
    review_type: str = Field(default="comprehensive", description="Type of review to perform")
    focus_areas: list[str] | None = Field(default=None, description="Specific areas to focus on")
    severity_threshold: str = Field(
        default="medium", description="Minimum severity level to report"
    )

    @validator("file_path")
    def validate_file_path(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("File path must be a non-empty string")
        return v


class ReviewPrParams(BaseModel):
    """Parameters for pull request review."""

    pr_number: int = Field(description="Pull request number", gt=0)
    repository: str | None = Field(default=None, description="Repository identifier")
    focus_areas: list[str] | None = Field(default=None, description="Areas to focus review on")
    auto_approve: bool = Field(default=False, description="Auto-approve if no issues found")


class ReviewDiffParams(BaseModel):
    """Parameters for diff review."""

    diff_content: str = Field(description="Diff content to review")
    context_files: list[str] | None = Field(default=None, description="Additional context files")
    review_level: str = Field(default="thorough", description="Level of review thoroughness")


# Gemini Server Agent Parameters
class CodeAnalysisParams(BaseModel):
    """Parameters for code analysis using Gemini."""

    code: str = Field(description="Code to analyze")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis")
    language: str | None = Field(default=None, description="Programming language")
    focus_areas: list[str] | None = Field(default=None, description="Specific focus areas")

    @validator("code")
    def validate_code(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Code must be a non-empty string")
        return v


class SystemCommandParams(BaseModel):
    """Parameters for system command execution."""

    command: str = Field(description="System command to execute")
    working_directory: str | None = Field(default=None, description="Working directory")
    timeout: int = Field(default=30, ge=1, le=300, description="Command timeout in seconds")
    capture_output: bool = Field(default=True, description="Capture command output")

    @validator("command")
    def validate_command(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Command must be a non-empty string")
        return v


class ContentGenerationParams(BaseModel):
    """Parameters for content generation using Gemini."""

    prompt: str = Field(description="Generation prompt")
    content_type: str = Field(default="text", description="Type of content to generate")
    max_length: int | None = Field(default=None, description="Maximum content length")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Generation temperature")

    @validator("prompt")
    def validate_prompt(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Prompt must be a non-empty string")
        return v


class FileProcessingParams(BaseModel):
    """Parameters for file processing tasks."""

    file_path: str = Field(description="Path to file to process")
    operation: str = Field(description="Processing operation to perform")
    options: dict[str, Any] | None = Field(default=None, description="Processing options")
    output_path: str | None = Field(default=None, description="Output file path")


# Production Ready Agent Parameters
class ProductionChecklistParams(BaseModel):
    """Parameters for production readiness checklist."""

    project_path: str = Field(description="Path to the project")
    environment: str = Field(default="production", description="Target environment")
    check_categories: list[str] | None = Field(default=None, description="Categories to check")
    generate_report: bool = Field(default=True, description="Generate detailed report")


class SecurityAuditParams(BaseModel):
    """Parameters for security audit."""

    project_path: str = Field(description="Path to the project")
    audit_depth: str = Field(default="comprehensive", description="Depth of security audit")
    include_dependencies: bool = Field(default=True, description="Audit dependencies")
    generate_fixes: bool = Field(default=True, description="Generate security fixes")


class PerformanceAnalysisParams(BaseModel):
    """Parameters for performance analysis."""

    project_path: str = Field(description="Path to the project")
    analysis_type: str = Field(default="comprehensive", description="Type of performance analysis")
    benchmark_against: str | None = Field(default=None, description="Benchmark comparison target")
    generate_optimizations: bool = Field(
        default=True, description="Generate optimization suggestions"
    )
