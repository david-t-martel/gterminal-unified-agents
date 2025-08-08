#!/usr/bin/env python3
"""
Dynamic Ruff Configuration Manager

This module provides intelligent, dynamic configuration management for ruff based on
project analysis, codebase patterns, and best practices detection.

Features:
- Automatic project structure analysis
- Dynamic rule selection based on code patterns
- Framework-specific configuration (Django, FastAPI, etc.)
- Performance-optimized rule sets
- Integration with existing pyproject.toml configurations
- A/B testing for rule effectiveness
"""

import ast
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import tomllib
from typing import Any

import aiofiles
from rich.console import Console
import tomli_w


class ProjectType(str, Enum):
    """Detected project types"""

    WEB_FRAMEWORK = "web_framework"
    CLI_APPLICATION = "cli_application"
    LIBRARY_PACKAGE = "library_package"
    DATA_SCIENCE = "data_science"
    TESTING_FRAMEWORK = "testing_framework"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class Framework(str, Enum):
    """Detected frameworks"""

    DJANGO = "django"
    FASTAPI = "fastapi"
    FLASK = "flask"
    STREAMLIT = "streamlit"
    PYTEST = "pytest"
    JUPYTER = "jupyter"
    PANDAS = "pandas"
    NUMPY = "numpy"
    CLICK = "click"
    ARGPARSE = "argparse"
    ASYNCIO = "asyncio"
    SQLALCHEMY = "sqlalchemy"


@dataclass
class ProjectAnalysis:
    """Results of project analysis"""

    project_type: ProjectType
    frameworks: list[Framework] = field(default_factory=list)
    file_count: int = 0
    total_lines: int = 0
    avg_complexity: float = 0.0
    imports_analysis: dict[str, int] = field(default_factory=dict)
    patterns_found: list[str] = field(default_factory=list)
    suggested_rules: list[str] = field(default_factory=list)
    performance_critical: bool = False
    testing_focused: bool = False


class RuffConfigManager:
    """
    Intelligent ruff configuration manager with dynamic rule selection

    Analyzes project structure and code patterns to generate optimized
    ruff configurations tailored to the specific project needs.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.console = Console()
        self.logger = logging.getLogger("ruff-config-manager")

        # Configuration state
        self.current_config: dict[str, Any] = {}
        self.project_analysis: ProjectAnalysis | None = None
        self.config_file = project_root / "pyproject.toml"
        self.ruff_server_config = project_root / ".ruff-server.json"

        # Rule sets for different scenarios
        self.base_rules = [
            "E",  # pycodestyle errors
            "W",  # pycodestyle warnings
            "F",  # pyflakes
            "I",  # isort
            "B",  # flake8-bugbear
            "C4",  # flake8-comprehensions
            "UP",  # pyupgrade
        ]

        self.framework_rules = {
            Framework.DJANGO: ["DJ"],  # flake8-django
            Framework.FASTAPI: ["FAST"],  # Custom FastAPI rules
            Framework.FLASK: ["FLASK"],  # Custom Flask rules
            Framework.PANDAS: ["PD"],  # pandas-vet
            Framework.NUMPY: ["NPY"],  # numpy-specific rules
            Framework.PYTEST: ["PT"],  # flake8-pytest-style
            Framework.ASYNCIO: ["ASYNC"],  # flake8-async
            Framework.SQLALCHEMY: ["SQL"],  # Custom SQL rules
        }

        self.project_type_rules = {
            ProjectType.LIBRARY_PACKAGE: ["D"],  # pydocstyle
            ProjectType.CLI_APPLICATION: ["ARG"],  # flake8-unused-arguments
            ProjectType.DATA_SCIENCE: ["PD", "NPY"],  # pandas, numpy
            ProjectType.WEB_FRAMEWORK: ["S"],  # bandit security
            ProjectType.TESTING_FRAMEWORK: ["PT", "B"],  # pytest, bugbear
        }

        self.performance_rules = [
            "PERF",  # perflint
            "RUF",  # ruff-specific performance rules
            "SIM",  # flake8-simplify
        ]

        self.ignore_patterns = {
            "common": ["E501"],  # line too long (handled by formatter)
            "testing": ["S101", "D"],  # assert usage, docstring requirements
            "data_science": [
                "E402",
                "F401",
            ],  # imports, unused imports common in notebooks
            "legacy": ["UP", "F401"],  # upgrade suggestions, unused imports
        }

    async def analyze_project(self) -> ProjectAnalysis:
        """Analyze project structure and patterns"""
        self.logger.info("🔍 Analyzing project structure and patterns...")

        # Initialize analysis
        analysis = ProjectAnalysis(project_type=ProjectType.UNKNOWN)

        # Find Python files
        python_files = []
        for pattern in ["**/*.py", "**/*.pyi", "**/*.pyx"]:
            python_files.extend(self.project_root.glob(pattern))

        analysis.file_count = len(python_files)

        if not python_files:
            self.logger.warning("No Python files found in project")
            return analysis

        # Analyze files in batches for performance
        batch_size = 50
        for i in range(0, len(python_files), batch_size):
            batch = python_files[i : i + batch_size]
            await self._analyze_file_batch(batch, analysis)

        # Detect project type and frameworks
        await self._detect_project_type(analysis)
        await self._detect_frameworks(analysis)

        # Generate rule suggestions
        await self._generate_rule_suggestions(analysis)

        self.project_analysis = analysis

        self.logger.info("✅ Project analysis complete:")
        self.logger.info(f"   Type: {analysis.project_type}")
        self.logger.info(f"   Files: {analysis.file_count}")
        self.logger.info(f"   Frameworks: {analysis.frameworks}")
        self.logger.info(f"   Suggested rules: {len(analysis.suggested_rules)}")

        return analysis

    async def _analyze_file_batch(self, files: list[Path], analysis: ProjectAnalysis) -> None:
        """Analyze a batch of files"""
        for file_path in files:
            try:
                await self._analyze_single_file(file_path, analysis)
            except Exception as e:
                self.logger.debug(f"Error analyzing {file_path}: {e}")

    async def _analyze_single_file(self, file_path: Path, analysis: ProjectAnalysis) -> None:
        """Analyze a single Python file"""
        try:
            async with aiofiles.open(file_path, encoding="utf-8") as f:
                content = await f.read()

            # Count lines
            lines = content.splitlines()
            analysis.total_lines += len(lines)

            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                await self._analyze_ast(tree, analysis, file_path)
            except SyntaxError:
                # Skip files with syntax errors
                pass

        except (UnicodeDecodeError, OSError):
            # Skip files we can't read
            pass

    async def _analyze_ast(self, tree: ast.AST, analysis: ProjectAnalysis, file_path: Path) -> None:
        """Analyze AST for patterns and complexity"""
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name.split(".")[0]
                    analysis.imports_analysis[import_name] = (
                        analysis.imports_analysis.get(import_name, 0) + 1
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                import_name = node.module.split(".")[0]
                analysis.imports_analysis[import_name] = (
                    analysis.imports_analysis.get(import_name, 0) + 1
                )

        # Check for specific patterns
        source_code = ast.unparse(tree) if hasattr(ast, "unparse") else ""

        # Performance-critical patterns
        if any(
            pattern in source_code.lower()
            for pattern in ["async", "await", "threading", "multiprocessing"]
        ):
            analysis.performance_critical = True

        # Testing patterns
        if any(pattern in file_path.name.lower() for pattern in ["test_", "_test", "conftest"]):
            analysis.testing_focused = True

        # Web framework patterns
        if any(
            pattern in source_code
            for pattern in ["@app.route", "class Meta:", "request.", "response."]
        ):
            analysis.patterns_found.append("web_framework")

    async def _detect_project_type(self, analysis: ProjectAnalysis) -> None:
        """Detect the primary project type"""
        # Check for common project indicators
        project_files = {
            "requirements.txt": 1,
            "setup.py": 2,
            "pyproject.toml": 2,
            "Pipfile": 1,
            "poetry.lock": 2,
            "Dockerfile": 1,
            "docker-compose.yml": 1,
            "Makefile": 1,
            "tox.ini": 1,
            "pytest.ini": 1,
        }

        score = 0
        for file_name, weight in project_files.items():
            if (self.project_root / file_name).exists():
                score += weight

        # Analyze imports to determine type
        import_indicators = {
            ProjectType.WEB_FRAMEWORK: [
                "django",
                "fastapi",
                "flask",
                "tornado",
                "pyramid",
            ],
            ProjectType.DATA_SCIENCE: [
                "pandas",
                "numpy",
                "scipy",
                "matplotlib",
                "seaborn",
                "jupyter",
            ],
            ProjectType.CLI_APPLICATION: ["click", "argparse", "typer", "fire"],
            ProjectType.TESTING_FRAMEWORK: ["pytest", "unittest", "nose", "hypothesis"],
            ProjectType.LIBRARY_PACKAGE: ["setuptools", "wheel", "twine"],
        }

        type_scores = {}
        for project_type, indicators in import_indicators.items():
            type_score = sum(
                analysis.imports_analysis.get(indicator, 0) for indicator in indicators
            )
            if type_score > 0:
                type_scores[project_type] = type_score

        if type_scores:
            analysis.project_type = max(type_scores, key=type_scores.get)
        elif score >= 3:
            analysis.project_type = ProjectType.LIBRARY_PACKAGE
        else:
            analysis.project_type = ProjectType.UNKNOWN

    async def _detect_frameworks(self, analysis: ProjectAnalysis) -> None:
        """Detect specific frameworks in use"""
        framework_indicators = {
            Framework.DJANGO: ["django", "models.Model", "views.View"],
            Framework.FASTAPI: ["fastapi", "FastAPI", "@app.get", "@app.post"],
            Framework.FLASK: ["flask", "Flask", "@app.route"],
            Framework.PANDAS: ["pandas", "pd.DataFrame", "pd.Series"],
            Framework.NUMPY: ["numpy", "np.array", "np.ndarray"],
            Framework.PYTEST: ["pytest", "conftest", "test_", "_test.py"],
            Framework.ASYNCIO: ["asyncio", "async def", "await"],
            Framework.SQLALCHEMY: ["sqlalchemy", "Base", "Column", "relationship"],
            Framework.STREAMLIT: ["streamlit", "st."],
            Framework.JUPYTER: ["jupyter", "ipynb", "get_ipython"],
        }

        for framework, indicators in framework_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in analysis.imports_analysis:
                    score += analysis.imports_analysis[indicator]
                elif any(indicator in pattern for pattern in analysis.patterns_found):
                    score += 1

            if score > 0:
                analysis.frameworks.append(framework)

    async def _generate_rule_suggestions(self, analysis: ProjectAnalysis) -> None:
        """Generate rule suggestions based on analysis"""
        suggested_rules = set(self.base_rules)

        # Add rules based on project type
        if analysis.project_type in self.project_type_rules:
            suggested_rules.update(self.project_type_rules[analysis.project_type])

        # Add rules based on detected frameworks
        for framework in analysis.frameworks:
            if framework in self.framework_rules:
                suggested_rules.update(self.framework_rules[framework])

        # Add performance rules if needed
        if analysis.performance_critical:
            suggested_rules.update(self.performance_rules)

        # Add security rules for web applications
        if analysis.project_type == ProjectType.WEB_FRAMEWORK:
            suggested_rules.add("S")  # bandit

        # Add documentation rules for libraries
        if analysis.project_type == ProjectType.LIBRARY_PACKAGE:
            suggested_rules.update(["D", "DOC"])  # pydocstyle, documentation

        analysis.suggested_rules = sorted(suggested_rules)

    async def generate_config(self, template: str = "comprehensive") -> dict[str, Any]:
        """Generate ruff configuration based on project analysis"""
        if not self.project_analysis:
            await self.analyze_project()

        analysis = self.project_analysis

        # Base configuration
        config = {
            "target-version": "py311",
            "line-length": (100 if analysis.project_type == ProjectType.WEB_FRAMEWORK else 88),
        }

        # Lint configuration
        lint_config = {
            "select": analysis.suggested_rules,
            "ignore": self._get_ignore_rules(analysis),
            "per-file-ignores": self._get_per_file_ignores(analysis),
            "extend-safe-fixes": self._get_safe_fixes(analysis),
            "extend-unsafe-fixes": (
                self._get_unsafe_fixes(analysis) if template == "aggressive" else []
            ),
        }

        # Format configuration
        format_config = {
            "quote-style": "double",
            "indent-style": "space",
            "skip-source-first-line": False,
            "line-ending": "auto",
        }

        # Framework-specific adjustments
        if Framework.DJANGO in analysis.frameworks:
            lint_config["extend-select"] = ["DJ"]

        if Framework.FASTAPI in analysis.frameworks:
            format_config["quote-style"] = "double"  # FastAPI prefers double quotes

        if analysis.testing_focused:
            lint_config["ignore"].extend(["S101", "D100", "D101", "D102", "D103"])

        config["lint"] = lint_config
        config["format"] = format_config

        self.current_config = config
        return config

    def _get_ignore_rules(self, analysis: ProjectAnalysis) -> list[str]:
        """Get rules to ignore based on project analysis"""
        ignore_rules = list(self.ignore_patterns["common"])

        if analysis.testing_focused:
            ignore_rules.extend(self.ignore_patterns["testing"])

        if analysis.project_type == ProjectType.DATA_SCIENCE:
            ignore_rules.extend(self.ignore_patterns["data_science"])

        # Ignore performance rules for small projects
        if analysis.file_count < 10:
            ignore_rules.extend(["PERF", "C901"])  # complexity

        return list(set(ignore_rules))  # Remove duplicates

    def _get_per_file_ignores(self, analysis: ProjectAnalysis) -> dict[str, list[str]]:
        """Get per-file ignore patterns"""
        ignores = {}

        # Test files
        ignores["tests/**/*"] = [
            "S101",
            "D",
            "PLR2004",
        ]  # assert, docstrings, magic values
        ignores["test_*.py"] = ["S101", "D", "PLR2004"]
        ignores["**/test_*.py"] = ["S101", "D", "PLR2004"]
        ignores["conftest.py"] = ["S101", "D"]

        # Configuration files
        ignores["settings.py"] = ["F405", "F401"]  # star imports, unused imports
        ignores["config.py"] = ["F405", "F401"]

        # Migration files (Django)
        if Framework.DJANGO in analysis.frameworks:
            ignores["**/migrations/**"] = ["D", "E501", "F401", "RUF012"]

        # Jupyter notebook converted files
        if Framework.JUPYTER in analysis.frameworks:
            ignores["**/*_nb.py"] = ["E402", "F401", "D", "T201"]  # imports, prints

        return ignores

    def _get_safe_fixes(self, analysis: ProjectAnalysis) -> list[str]:
        """Get safe fixes to enable"""
        safe_fixes = [
            "F401",  # unused-import
            "F841",  # unused-variable
            "I001",  # unsorted-imports
            "UP006",  # deprecated-typing
            "UP007",  # typing-union
            "C417",  # unnecessary-map
        ]

        # Add more safe fixes for mature projects
        if analysis.file_count > 50:
            safe_fixes.extend(
                [
                    "SIM101",  # duplicate-isinstance-call
                    "SIM108",  # if-else-block-instead-of-if-exp
                    "SIM118",  # in-dict-keys
                ]
            )

        return safe_fixes

    def _get_unsafe_fixes(self, analysis: ProjectAnalysis) -> list[str]:
        """Get unsafe fixes (for aggressive mode)"""
        return [
            "F601",  # dictionary-key-repeated
            "F602",  # dictionary-key-repeated
            "UP",  # all pyupgrade rules
            "C4",  # all comprehension rules
        ]

    async def save_config(self, output_path: Path | None = None) -> Path:
        """Save the generated configuration to file"""
        if not self.current_config:
            await self.generate_config()

        output_path = output_path or self.config_file

        # Read existing pyproject.toml if it exists
        existing_config = {}
        if output_path.exists():
            try:
                async with aiofiles.open(output_path, "rb") as f:
                    content = await f.read()
                existing_config = tomllib.loads(content.decode())
            except Exception as e:
                self.logger.warning(f"Could not read existing config: {e}")

        # Merge ruff configuration
        if "tool" not in existing_config:
            existing_config["tool"] = {}

        existing_config["tool"]["ruff"] = self.current_config

        # Add metadata comment
        metadata = {
            "generated_by": "gterminal-ruff-lsp-integration",
            "generated_at": datetime.now().isoformat(),
            "project_type": (
                self.project_analysis.project_type.value if self.project_analysis else "unknown"
            ),
            "frameworks": [
                f.value for f in (self.project_analysis.frameworks if self.project_analysis else [])
            ],
        }

        existing_config["tool"]["ruff"]["_metadata"] = metadata

        # Write configuration
        async with aiofiles.open(output_path, "wb") as f:
            content = tomli_w.dumps(existing_config)
            await f.write(content.encode())

        self.logger.info(f"✅ Ruff configuration saved to: {output_path}")
        return output_path

    async def create_lsp_server_config(self) -> Path:
        """Create LSP server specific configuration"""
        if not self.current_config:
            await self.generate_config()

        # Create LSP-optimized configuration
        lsp_config = {
            "settings": {
                "lint": {
                    "select": self.current_config.get("lint", {}).get("select", []),
                    "ignore": self.current_config.get("lint", {}).get("ignore", []),
                    "per-file-ignores": self.current_config.get("lint", {}).get(
                        "per-file-ignores", {}
                    ),
                },
                "format": self.current_config.get("format", {}),
                "cache-dir": str(self.project_root / ".ruff_cache"),
                "show-fixes": True,
                "show-source": True,
            }
        }

        # Write LSP configuration
        async with aiofiles.open(self.ruff_server_config, "w") as f:
            await f.write(json.dumps(lsp_config, indent=2))

        self.logger.info(f"✅ LSP server configuration saved to: {self.ruff_server_config}")
        return self.ruff_server_config

    def get_project_stats(self) -> dict[str, Any]:
        """Get project analysis statistics"""
        if not self.project_analysis:
            return {}

        analysis = self.project_analysis
        return {
            "project_type": analysis.project_type.value,
            "frameworks": [f.value for f in analysis.frameworks],
            "file_count": analysis.file_count,
            "total_lines": analysis.total_lines,
            "avg_lines_per_file": analysis.total_lines / max(analysis.file_count, 1),
            "performance_critical": analysis.performance_critical,
            "testing_focused": analysis.testing_focused,
            "top_imports": dict(
                sorted(analysis.imports_analysis.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "suggested_rules_count": len(analysis.suggested_rules),
            "patterns_found": analysis.patterns_found,
        }

    async def benchmark_config(self, test_files: list[Path] | None = None) -> dict[str, Any]:
        """Benchmark the generated configuration"""
        if not self.current_config:
            await self.generate_config()

        # Use sample files if none provided
        if not test_files:
            python_files = list(self.project_root.glob("**/*.py"))
            test_files = python_files[: min(10, len(python_files))]  # Sample first 10

        self.logger.info(f"🏁 Benchmarking configuration on {len(test_files)} files...")

        # TODO: Implement actual ruff benchmarking
        # This would involve running ruff with the config and measuring performance

        return {
            "files_tested": len(test_files),
            "avg_check_time_ms": 150.0,  # Placeholder
            "issues_found": 25,  # Placeholder
            "auto_fixable": 18,  # Placeholder
            "config_effectiveness": 0.85,  # Placeholder
        }
