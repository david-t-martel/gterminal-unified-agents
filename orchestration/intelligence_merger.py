#!/usr/bin/env python3
"""Intelligence Merger - Extract and merge best elements from multiple files.

This component focuses on INTELLIGENT CONSOLIDATION:
1. Analyze multiple similar files
2. Extract the BEST elements from each
3. Merge into a single, superior consolidated file
4. Preserve valuable optimizations especially from /app/
"""

import ast
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodeElement:
    """Represents a code element that can be extracted and merged."""

    element_type: str  # function, class, import, constant, etc.
    name: str
    content: str
    source_file: str
    quality_score: float
    dependencies: set[str]
    optimization_indicators: list[str]
    docstring: str | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class MergeAnalysis:
    """Analysis of what should be merged and how."""

    best_elements: list[CodeElement]
    duplicate_elements: list[list[CodeElement]]
    missing_dependencies: set[str]
    consolidation_strategy: str
    estimated_lines_saved: int


class IntelligenceMerger:
    """Intelligent code merger that extracts best elements and creates superior consolidated files.

    Unlike simple file deletion, this preserves and enhances the best parts of multiple files.
    """

    def __init__(self) -> None:
        self.optimization_keywords = {
            "async",
            "await",
            "asyncio",
            "concurrent",
            "threading",
            "multiprocessing",
            "cache",
            "lru_cache",
            "functools",
            "optimize",
            "performance",
            "fast",
            "efficient",
            "PyO3",
            "Rust",
            "uvloop",
            "cython",
            "numpy",
            "pandas",
        }

        self.quality_indicators = {
            "error_handling": ["try:", "except:", "finally:", "raise", "logging"],
            "documentation": ['"""', "'''", "docstring", "@param", "@return"],
            "type_hints": [":", "->", "typing", "Union", "Optional", "List", "Dict"],
            "testing": ["assert", "test_", "mock", "patch", "pytest", "unittest"],
            "modern_python": ['f"', "pathlib", "dataclass", "enum", "contextlib"],
        }

    def analyze_file(self, file_path: Path) -> list[CodeElement]:
        """Analyze a Python file and extract its code elements."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)

            elements: list[Any] = []

            # Extract top-level elements
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    element = self._analyze_function(node, content, str(file_path))
                    elements.append(element)
                elif isinstance(node, ast.ClassDef):
                    element = self._analyze_class(node, content, str(file_path))
                    elements.append(element)
                elif isinstance(node, ast.Import | ast.ImportFrom):
                    element = self._analyze_import(node, content, str(file_path))
                    elements.append(element)

            return elements

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.exception(f"Error analyzing {file_path}: {e}")
            return []

    def _analyze_function(self, node: ast.FunctionDef, content: str, file_path: str) -> CodeElement:
        """Analyze a function definition."""
        func_content = ast.get_source_segment(content, node) or ""

        # Calculate quality score
        quality_score = self._calculate_quality_score(func_content)

        # Find optimization indicators
        optimizations = [kw for kw in self.optimization_keywords if kw in func_content]

        # Extract dependencies
        dependencies = self._extract_dependencies(node)

        # Get docstring
        docstring = ast.get_docstring(node)

        return CodeElement(
            element_type="function",
            name=node.name,
            content=func_content,
            source_file=file_path,
            quality_score=quality_score,
            dependencies=dependencies,
            optimization_indicators=optimizations,
            docstring=docstring,
            line_start=node.lineno,
            line_end=node.end_lineno,
        )

    def _analyze_class(self, node: ast.ClassDef, content: str, file_path: str) -> CodeElement:
        """Analyze a class definition."""
        class_content = ast.get_source_segment(content, node) or ""

        quality_score = self._calculate_quality_score(class_content)
        optimizations = [kw for kw in self.optimization_keywords if kw in class_content]
        dependencies = self._extract_dependencies(node)
        docstring = ast.get_docstring(node)

        return CodeElement(
            element_type="class",
            name=node.name,
            content=class_content,
            source_file=file_path,
            quality_score=quality_score,
            dependencies=dependencies,
            optimization_indicators=optimizations,
            docstring=docstring,
            line_start=node.lineno,
            line_end=node.end_lineno,
        )

    def _analyze_import(
        self, node: ast.Import | ast.ImportFrom, content: str, file_path: str
    ) -> CodeElement:
        """Analyze an import statement."""
        import_content = ast.get_source_segment(content, node) or ""

        # Imports get quality score based on what they import
        quality_score = 1.0
        if any(kw in import_content for kw in self.optimization_keywords):
            quality_score += 2.0

        if isinstance(node, ast.ImportFrom):
            name = f"from {node.module}" if node.module else "from ."
        else:
            name = "import " + ", ".join(alias.name for alias in node.names)

        return CodeElement(
            element_type="import",
            name=name,
            content=import_content,
            source_file=file_path,
            quality_score=quality_score,
            dependencies=set(),
            optimization_indicators=[],
            line_start=node.lineno,
            line_end=node.end_lineno,
        )

    def _calculate_quality_score(self, content: str) -> float:
        """Calculate quality score for code content."""
        score = 1.0  # Base score

        # Check for quality indicators
        for indicators in self.quality_indicators.values():
            found = sum(1 for indicator in indicators if indicator in content)
            if found > 0:
                score += found * 0.5

        # Bonus for optimization indicators
        optimization_count = sum(1 for kw in self.optimization_keywords if kw in content)
        score += optimization_count * 1.0

        # Bonus for comprehensive error handling
        if all(indicator in content for indicator in ["try:", "except:", "logging"]):
            score += 2.0

        # Penalty for very short or very long functions
        lines = len(content.split("\n"))
        if lines < 3:
            score *= 0.5  # Too short
        elif lines > 100:
            score *= 0.8  # Might be too complex

        return score

    def _extract_dependencies(self, node: ast.AST) -> set[str]:
        """Extract dependencies from an AST node."""
        dependencies = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.add(child.id)
            elif isinstance(child, ast.Attribute):
                dependencies.add(child.attr)

        # Filter out built-ins and keywords
        builtins = {"print", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple"}
        keywords = {
            "self",
            "cls",
            "return",
            "yield",
            "if",
            "else",
            "for",
            "while",
            "with",
            "def",
            "class",
        }

        return dependencies - builtins - keywords

    def find_consolidation_opportunities(self, file_paths: list[Path]) -> dict[str, list[Path]]:
        """Find files that should be consolidated together."""
        opportunities: dict[str, Any] = {}

        # Group by similar names/purposes
        for file_path in file_paths:
            file_name = file_path.name.lower()

            # Identify file categories
            if "agent" in file_name:
                opportunities.setdefault("agents", []).append(file_path)
            elif "mcp" in file_name:
                opportunities.setdefault("mcp_servers", []).append(file_path)
            elif "test" in file_name:
                opportunities.setdefault("tests", []).append(file_path)
            elif any(word in file_name for word in ["config", "settings", "env"]):
                opportunities.setdefault("configuration", []).append(file_path)
            elif any(word in file_name for word in ["util", "helper", "common", "shared"]):
                opportunities.setdefault("utilities", []).append(file_path)
            elif any(word in file_name for word in ["performance", "cache", "optimize"]):
                opportunities.setdefault("performance", []).append(file_path)
            else:
                opportunities.setdefault("miscellaneous", []).append(file_path)

        # Only return groups with multiple files
        return {category: files for category, files in opportunities.items() if len(files) > 1}

    def analyze_consolidation_group(self, file_paths: list[Path]) -> MergeAnalysis:
        """Analyze a group of files for intelligent consolidation."""
        all_elements: list[Any] = []

        # Analyze all files
        for file_path in file_paths:
            elements = self.analyze_file(file_path)
            all_elements.extend(elements)

        # Group elements by name and type
        element_groups: dict[str, Any] = {}
        for element in all_elements:
            key = (element.element_type, element.name)
            element_groups.setdefault(key, []).append(element)

        # Find best version of each element
        best_elements: list[Any] = []
        duplicate_groups: list[Any] = []

        for elements in element_groups.values():
            if len(elements) == 1:
                best_elements.append(elements[0])
            else:
                # Multiple versions - find the best one
                best = max(elements, key=lambda e: e.quality_score)
                best_elements.append(best)
                duplicate_groups.append(elements)

        # Calculate potential savings
        total_lines = sum(len(e.content.split("\n")) for e in all_elements)
        kept_lines = sum(len(e.content.split("\n")) for e in best_elements)
        lines_saved = total_lines - kept_lines

        # Determine consolidation strategy
        if any("app/" in elem.source_file for elem in best_elements):
            strategy = "preserve_app_optimizations"
        elif any(elem.optimization_indicators for elem in best_elements):
            strategy = "preserve_performance_optimizations"
        else:
            strategy = "merge_similar_functionality"

        return MergeAnalysis(
            best_elements=best_elements,
            duplicate_elements=duplicate_groups,
            missing_dependencies=set(),  # TODO: implement dependency analysis
            consolidation_strategy=strategy,
            estimated_lines_saved=lines_saved,
        )

    def create_consolidated_file(
        self, analysis: MergeAnalysis, output_path: Path, file_group_name: str
    ) -> str:
        """Create a consolidated file from the merge analysis."""
        # Sort elements by type for better organization
        imports = [e for e in analysis.best_elements if e.element_type == "import"]
        classes = [e for e in analysis.best_elements if e.element_type == "class"]
        functions = [e for e in analysis.best_elements if e.element_type == "function"]

        # Build consolidated content
        content_parts = [
            '"""',
            f"Consolidated {file_group_name.title()} Module",
            "",
            "This file was intelligently created by merging the best elements from:",
        ]

        # List source files
        source_files = {e.source_file for e in analysis.best_elements}
        for source_file in sorted(source_files):
            content_parts.append(f"- {source_file}")

        content_parts.extend(
            [
                "",
                f"Consolidation strategy: {analysis.consolidation_strategy}",
                f"Lines saved: {analysis.estimated_lines_saved}",
                f"Created: {datetime.now().isoformat()}",
                '"""',
                "",
            ],
        )

        # Add imports
        if imports:
            content_parts.append("# Consolidated imports")
            for imp in imports:
                content_parts.append(imp.content)
            content_parts.append("")

        # Add classes
        if classes:
            content_parts.append("# Consolidated classes")
            for cls in classes:
                content_parts.extend(
                    [
                        f"# From: {cls.source_file}",
                        f"# Quality score: {cls.quality_score:.1f}",
                        cls.content,
                        "",
                    ],
                )

        # Add functions
        if functions:
            content_parts.append("# Consolidated functions")
            for func in functions:
                content_parts.extend(
                    [
                        f"# From: {func.source_file}",
                        f"# Quality score: {func.quality_score:.1f}",
                        func.content,
                        "",
                    ],
                )

        consolidated_content = "\n".join(content_parts)

        # Write consolidated file
        output_path.write_text(consolidated_content, encoding="utf-8")

        return f"Created consolidated file: {output_path} ({len(analysis.best_elements)} elements merged)"

    def intelligent_consolidation(self, project_root: Path) -> dict[str, Any]:
        """Perform intelligent consolidation on the entire project."""
        results = {
            "consolidated_files": [],
            "files_marked_for_deletion": [],
            "optimizations_preserved": [],
            "total_lines_saved": 0,
            "summary": "",
        }

        # Find all Python files
        python_files = list(project_root.rglob("*.py"))

        # Exclude certain directories
        excluded_dirs = {".venv", "__pycache__", ".git", "node_modules"}
        python_files = [f for f in python_files if not any(exc in str(f) for exc in excluded_dirs)]

        logger.info(f"Found {len(python_files)} Python files for consolidation analysis")

        # Find consolidation opportunities
        opportunities = self.find_consolidation_opportunities(python_files)

        logger.info(f"Found {len(opportunities)} consolidation opportunities")

        # Process each opportunity
        for group_name, file_group in opportunities.items():
            if len(file_group) < 2:
                continue

            logger.info(f"Analyzing {group_name} group with {len(file_group)} files")

            # Analyze the group
            analysis = self.analyze_consolidation_group(file_group)

            # Create consolidated file
            output_path = project_root / "app" / f"consolidated_{group_name}.py"
            output_path.parent.mkdir(exist_ok=True)

            self.create_consolidated_file(analysis, output_path, group_name)

            results["consolidated_files"].append(
                {
                    "group": group_name,
                    "output_file": str(output_path),
                    "source_files": [str(f) for f in file_group],
                    "elements_merged": len(analysis.best_elements),
                    "lines_saved": analysis.estimated_lines_saved,
                    "strategy": analysis.consolidation_strategy,
                },
            )

            # Mark original files for deletion (but don't delete yet)
            results["files_marked_for_deletion"].extend([str(f) for f in file_group])

            # Track preserved optimizations
            optimized_elements = [e for e in analysis.best_elements if e.optimization_indicators]
            for element in optimized_elements:
                results["optimizations_preserved"].append(
                    {
                        "element": f"{element.element_type}:{element.name}",
                        "source": element.source_file,
                        "optimizations": element.optimization_indicators,
                    },
                )

            results["total_lines_saved"] += analysis.estimated_lines_saved

        # Create summary
        results["summary"] = (
            f"Intelligent consolidation completed: "
            f"{len(results['consolidated_files'])} consolidated files created, "
            f"{len(results['files_marked_for_deletion'])} files marked for deletion, "
            f"{results['total_lines_saved']} lines saved, "
            f"{len(results['optimizations_preserved'])} optimizations preserved"
        )

        logger.info(results["summary"])
        return results
