#!/usr/bin/env python3
"""Code Generation Tools - Tools for generating and refactoring code."""

import logging
from pathlib import Path
from typing import Any

from vertexai.generative_models import GenerativeModel

from gterminal.core.tools.registry import BaseTool
from gterminal.core.tools.registry import ToolParameter
from gterminal.core.tools.registry import ToolResult

logger = logging.getLogger(__name__)


class GenerateCodeTool(BaseTool):
    """Tool for generating code using Gemini."""

    def __init__(self, model: GenerativeModel | None = None) -> None:
        super().__init__(
            name="generate_code",
            description="Generate code based on requirements",
            category="generation",
        )
        self.model = model
        self._model_name = "gemini-2.0-flash-exp"

    def _get_model(self) -> GenerativeModel:
        """Get the model instance, creating it if necessary."""
        if self.model is None:
            self.model = GenerativeModel(self._model_name)
        return self.model

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="description",
                type="string",
                description="Description of what to generate",
                required=True,
            ),
            ToolParameter(
                name="language",
                type="string",
                description="Programming language",
                required=False,
                default="python",
            ),
            ToolParameter(
                name="output_path",
                type="string",
                description="Path to save generated code",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="context",
                type="string",
                description="Additional context or requirements",
                required=False,
                default="",
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            description = params["description"]
            language = params.get("language", "python")
            output_path = params.get("output_path")
            context = params.get("context", "")

            prompt = f"""Generate {language} code for the following requirement:

Description: {description}

Additional Context: {context}

Requirements:
1. Write clean, well-documented code
2. Follow best practices for {language}
3. Include appropriate error handling
4. Add docstrings/comments explaining the code
5. Make it production-ready

Generate only the code without any markdown formatting or explanations."""

            response = await self._get_model().generate_content_async(prompt)
            generated_code = response.text

            # Save to file if path provided
            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as f:
                    f.write(generated_code)

            return ToolResult(
                success=True,
                data={
                    "code": generated_code,
                    "language": language,
                    "saved_to": str(output_path) if output_path else None,
                    "lines": len(generated_code.splitlines()),
                },
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class RefactorCodeTool(BaseTool):
    """Tool for refactoring existing code."""

    def __init__(self, model: GenerativeModel | None = None) -> None:
        super().__init__(
            name="refactor_code",
            description="Refactor code to improve quality",
            category="generation",
        )
        self.model = model
        self._model_name = "gemini-2.0-flash-exp"

    def _get_model(self) -> GenerativeModel:
        """Get the model instance, creating it if necessary."""
        if self.model is None:
            self.model = GenerativeModel(self._model_name)
        return self.model

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to file to refactor",
                required=True,
            ),
            ToolParameter(
                name="improvements",
                type="list",
                description="Types of improvements (readability, performance, security)",
                required=False,
                default=["readability", "performance"],
            ),
            ToolParameter(
                name="preserve_functionality",
                type="boolean",
                description="Ensure functionality is preserved",
                required=False,
                default=True,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            file_path = Path(params["file_path"])
            improvements = params.get("improvements", ["readability", "performance"])
            preserve_functionality = params.get("preserve_functionality", True)

            if not file_path.exists():
                return ToolResult(success=False, data=None, error=f"File not found: {file_path}")

            with file_path.open(encoding="utf-8") as f:
                original_code = f.read()

            prompt = f"""Refactor the following code to improve {", ".join(improvements)}:

```
{original_code}
```

Requirements:
1. {"Preserve exact functionality" if preserve_functionality else "Improve functionality if possible"}
2. Focus on: {", ".join(improvements)}
3. Maintain or improve code documentation
4. Follow language best practices
5. Ensure the refactored code is production-ready

Generate only the refactored code without any markdown formatting or explanations."""

            response = await self._get_model().generate_content_async(prompt)
            refactored_code = response.text

            # Create backup
            backup_path = file_path.with_suffix(file_path.suffix + ".backup")
            with backup_path.open("w", encoding="utf-8") as f:
                f.write(original_code)

            # Write refactored code
            with file_path.open("w", encoding="utf-8") as f:
                f.write(refactored_code)

            return ToolResult(
                success=True,
                data={
                    "file": str(file_path),
                    "backup": str(backup_path),
                    "improvements": improvements,
                    "original_lines": len(original_code.splitlines()),
                    "refactored_lines": len(refactored_code.splitlines()),
                },
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GenerateTestsTool(BaseTool):
    """Tool for generating tests for code."""

    def __init__(self, model: GenerativeModel | None = None) -> None:
        super().__init__(
            name="generate_tests",
            description="Generate tests for existing code",
            category="generation",
        )
        self.model = model
        self._model_name = "gemini-2.0-flash-exp"

    def _get_model(self) -> GenerativeModel:
        """Get the model instance, creating it if necessary."""
        if self.model is None:
            self.model = GenerativeModel(self._model_name)
        return self.model

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to file to test",
                required=True,
            ),
            ToolParameter(
                name="framework",
                type="string",
                description="Test framework (pytest, unittest)",
                required=False,
                default="pytest",
            ),
            ToolParameter(
                name="coverage_target",
                type="integer",
                description="Target test coverage percentage",
                required=False,
                default=85,
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            file_path = Path(params["file_path"])
            framework = params.get("framework", "pytest")
            coverage_target = params.get("coverage_target", 85)

            if not file_path.exists():
                return ToolResult(success=False, data=None, error=f"File not found: {file_path}")

            with file_path.open(encoding="utf-8") as f:
                code = f.read()

            prompt = f"""Generate comprehensive tests for the following code using {framework}:

```
{code}
```

Requirements:
1. Use {framework} framework
2. Aim for at least {coverage_target}% code coverage
3. Include unit tests for all functions/methods
4. Include edge cases and error conditions
5. Add appropriate test fixtures and mocks where needed
6. Include docstrings explaining what each test does
7. Follow {framework} best practices

Generate only the test code without any markdown formatting or explanations."""

            response = await self._get_model().generate_content_async(prompt)
            test_code = response.text

            # Save test file
            test_file = file_path.parent / f"test_{file_path.name}"
            with test_file.open("w", encoding="utf-8") as f:
                f.write(test_code)

            return ToolResult(
                success=True,
                data={
                    "test_file": str(test_file),
                    "source_file": str(file_path),
                    "framework": framework,
                    "test_lines": len(test_code.splitlines()),
                },
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class GenerateBoilerplateTool(BaseTool):
    """Tool for generating project boilerplate."""

    def __init__(self, model: GenerativeModel | None = None) -> None:
        super().__init__(
            name="generate_boilerplate",
            description="Generate project boilerplate and templates",
            category="generation",
        )
        self.model = model
        self._model_name = "gemini-2.0-flash-exp"

    def _get_model(self) -> GenerativeModel:
        """Get the model instance, creating it if necessary."""
        if self.model is None:
            self.model = GenerativeModel(self._model_name)
        return self.model

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="project_type",
                type="string",
                description="Type of project (api, cli, web, library)",
                required=True,
            ),
            ToolParameter(
                name="language",
                type="string",
                description="Programming language",
                required=False,
                default="python",
            ),
            ToolParameter(
                name="output_dir",
                type="string",
                description="Output directory",
                required=True,
            ),
            ToolParameter(
                name="features",
                type="list",
                description="Features to include",
                required=False,
                default=[],
            ),
        ]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            project_type = params["project_type"]
            language = params.get("language", "python")
            output_dir = Path(params["output_dir"])
            features = params.get("features", [])

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            files_created = []

            # Generate appropriate boilerplate based on project type
            if project_type == "api" and language == "python":
                # FastAPI boilerplate
                files_to_create = {
                    "main.py": self._generate_fastapi_main(),
                    "requirements.txt": self._generate_requirements(["fastapi", "uvicorn"]),
                    "Dockerfile": self._generate_dockerfile(language),
                    ".gitignore": self._generate_gitignore(language),
                    "README.md": self._generate_readme(project_type, language),
                }
            elif project_type == "cli" and language == "python":
                # CLI boilerplate
                files_to_create = {
                    "cli.py": self._generate_cli_main(),
                    "requirements.txt": self._generate_requirements(["click", "rich"]),
                    ".gitignore": self._generate_gitignore(language),
                    "README.md": self._generate_readme(project_type, language),
                }
            else:
                # Generic boilerplate
                files_to_create = {
                    "main.py": "#!/usr/bin/env python3\n\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()",
                    ".gitignore": self._generate_gitignore(language),
                    "README.md": self._generate_readme(project_type, language),
                }

            # Create files
            for filename, content in files_to_create.items():
                file_path = output_dir / filename
                with file_path.open("w", encoding="utf-8") as f:
                    f.write(content)
                files_created.append(str(file_path))

            return ToolResult(
                success=True,
                data={
                    "project_type": project_type,
                    "language": language,
                    "output_dir": str(output_dir),
                    "files_created": files_created,
                    "features": features,
                },
            )

        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def _generate_fastapi_main(self) -> str:
        return """#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="API", version="0.1.0")

class HealthResponse(BaseModel):
    status: str
    version: str

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", version="0.1.0")

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

    def _generate_cli_main(self) -> str:
        return """#!/usr/bin/env python3
import click
from rich.console import Console

console = Console()

@click.group()
def cli():
    \"\"\"CLI Application\"\"\"
    pass

@cli.command()
@click.option('--name', default='World', help='Name to greet')
def hello(name):
    \"\"\"Say hello\"\"\"
    console.print(f"[green]Hello, {name}![/green]")

if __name__ == '__main__':
    cli()
"""

    def _generate_requirements(self, packages: list[str]) -> str:
        return "\n".join(packages) + "\n"

    def _generate_dockerfile(self, language: str) -> str:
        if language == "python":
            return """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
"""
        return "# Add Dockerfile content for " + language

    def _generate_gitignore(self, language: str) -> str:
        if language == "python":
            return """__pycache__/
*.py[cod]
*$py.class
*.so
.env
.venv/
env/
venv/
ENV/
.coverage
htmlcov/
.pytest_cache/
"""
        return "# Add .gitignore for " + language

    def _generate_readme(self, project_type: str, language: str) -> str:
        return f"""# {project_type.title()} Project

A {project_type} application built with {language}.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the application
python main.py
```

## Development

```bash
# Run tests
pytest

# Format code
black .
```
"""
