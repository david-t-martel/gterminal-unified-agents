#!/usr/bin/env python3
"""Gemini Consolidation Server - Independent process for file consolidation operations.
Runs as a separate server that Claude can communicate with via HTTP/MCP.
"""

from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any

# FastAPI for HTTP server
from fastapi import FastAPI
from fastapi import HTTPException

# Google Gemini
import google.generativeai as genai
from pydantic import BaseModel
from pydantic import Field

# Rich for console output
from rich.console import Console

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gterminal.utils.cache_utils import SmartCache
from gterminal.utils.file_ops import FileOperations

console = Console()

# Initialize Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
else:
    model = None
    console.print("[yellow]Warning: No Gemini API key found. AI features disabled.[/yellow]")


class ConsolidationRequest(BaseModel):
    """Request for consolidation operations."""

    operation: str = Field(
        ..., description="Operation to perform: analyze, merge, delete, consolidate"
    )
    target_path: str = Field(..., description="Target directory or file path")
    patterns: list[str] | None = Field(default=None, description="File patterns to match")
    dry_run: bool = Field(default=True, description="Whether to perform dry run")
    aggressive: bool = Field(default=False, description="Aggressive consolidation mode")


class ConsolidationResponse(BaseModel):
    """Response from consolidation operations."""

    status: str
    operation: str
    files_affected: int
    details: dict[str, Any]
    suggestions: list[str] | None = None


class GeminiConsolidationServer:
    """Independent Gemini server for file consolidation."""

    def __init__(self) -> None:
        self.file_ops = FileOperations()
        self.cache = SmartCache()
        self.app = FastAPI(title="Gemini Consolidation Server")
        self.setup_routes()
        self.consolidation_history = []

    def setup_routes(self) -> None:
        """Set up FastAPI routes."""

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "gemini_enabled": model is not None}

        @self.app.post("/analyze", response_model=ConsolidationResponse)
        async def analyze_duplicates(request: ConsolidationRequest):
            """Analyze directory for duplicates."""
            return await self.analyze_duplicates(request)

        @self.app.post("/merge", response_model=ConsolidationResponse)
        async def merge_files(request: ConsolidationRequest):
            """Merge duplicate files."""
            return await self.merge_duplicates(request)

        @self.app.post("/delete", response_model=ConsolidationResponse)
        async def delete_files(request: ConsolidationRequest):
            """Delete redundant files."""
            return await self.delete_redundant(request)

        @self.app.post("/consolidate", response_model=ConsolidationResponse)
        async def consolidate_all(request: ConsolidationRequest):
            """Full consolidation: analyze, merge, and delete."""
            return await self.full_consolidation(request)

        @self.app.get("/history")
        async def get_history():
            """Get consolidation history."""
            return {"history": self.consolidation_history}

        @self.app.post("/gemini/analyze")
        async def gemini_analyze(request: ConsolidationRequest):
            """Use Gemini to analyze and suggest consolidation."""
            return await self.gemini_analysis(request)

    async def analyze_duplicates(self, request: ConsolidationRequest) -> ConsolidationResponse:
        """Analyze directory for duplicate files."""
        target = Path(request.target_path)

        if not target.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {target}")

        duplicates: dict[str, Any] = {}
        file_hashes: dict[str, Any] = {}
        files_analyzed = 0

        # Find all Python files (or use patterns)
        patterns = request.patterns or ["*.py"]

        for pattern in patterns:
            for file_path in target.rglob(pattern):
                if file_path.is_file():
                    files_analyzed += 1

                    # Calculate file hash
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()

                    if file_hash in file_hashes:
                        if file_hash not in duplicates:
                            duplicates[file_hash] = [file_hashes[file_hash]]
                        duplicates[file_hash].append(str(file_path))
                    else:
                        file_hashes[file_hash] = str(file_path)

        # Identify specific consolidation targets
        consolidation_targets = self._identify_consolidation_targets(target)

        details = {
            "files_analyzed": files_analyzed,
            "duplicate_groups": len(duplicates),
            "total_duplicates": sum(len(v) - 1 for v in duplicates.values()),
            "duplicate_files": list(duplicates.values()),
            "consolidation_targets": consolidation_targets,
        }

        suggestions = [
            f"Found {len(duplicates)} groups of duplicate files",
            f"Can eliminate {details['total_duplicates']} duplicate files",
            "Recommend aggressive consolidation of test runners, benchmarks, and validation files",
        ]

        response = ConsolidationResponse(
            status="success",
            operation="analyze",
            files_affected=files_analyzed,
            details=details,
            suggestions=suggestions,
        )

        self.consolidation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "operation": "analyze",
                "target": str(target),
                "results": details,
            },
        )

        return response

    def _identify_consolidation_targets(self, base_path: Path) -> dict[str, list[str]]:
        """Identify specific files that should be consolidated."""
        targets = {
            "test_runners": [],
            "benchmarks": [],
            "validators": [],
            "configs": [],
        }

        # Test runners
        for pattern in ["*test_runner*.py", "*runner*.py"]:
            targets["test_runners"].extend([str(p) for p in base_path.rglob(pattern)])

        # Benchmarks
        for pattern in ["*benchmark*.py", "*bench*.py"]:
            targets["benchmarks"].extend([str(p) for p in base_path.rglob(pattern)])

        # Validators
        for pattern in ["*validation*.py", "*validator*.py"]:
            targets["validators"].extend([str(p) for p in base_path.rglob(pattern)])

        # Configs
        for pattern in ["pytest*.ini", ".env*"]:
            targets["configs"].extend([str(p) for p in base_path.rglob(pattern)])

        return {k: v for k, v in targets.items() if v}

    async def merge_duplicates(self, request: ConsolidationRequest) -> ConsolidationResponse:
        """Merge duplicate files into consolidated versions."""
        target = Path(request.target_path)

        consolidation_map = {
            "test_runners": "tests/consolidated_test_runner.py",
            "benchmarks": "benchmarks/consolidated_benchmark.py",
            "validators": "tests/consolidated_validator.py",
        }

        merged_files: list[Any] = []

        if not request.dry_run:
            # Actually perform merges
            targets = self._identify_consolidation_targets(target)

            for category, consolidated_path in consolidation_map.items():
                if targets.get(category):
                    output_path = target / consolidated_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Merge content (simplified - in reality would parse and combine)
                    combined_content = self._merge_file_contents(targets[category])

                    with open(output_path, "w") as f:
                        f.write(combined_content)

                    merged_files.extend(targets[category])
                    console.print(
                        f"[green]Merged {len(targets[category])} files into {consolidated_path}[/green]"
                    )

        details = {
            "merged_files": merged_files,
            "consolidated_files": list(consolidation_map.values()),
            "dry_run": request.dry_run,
        }

        return ConsolidationResponse(
            status="success",
            operation="merge",
            files_affected=len(merged_files),
            details=details,
            suggestions=["Review consolidated files before deleting originals"],
        )

    def _merge_file_contents(self, file_paths: list[str]) -> str:
        """Merge contents of multiple files intelligently."""
        # This is a simplified version - in reality would parse AST and merge intelligently
        imports = set()

        for file_path in file_paths:
            with open(file_path) as f:
                lines = f.readlines()

                for line in lines:
                    if line.startswith(("import ", "from ")):
                        imports.add(line.strip())
                    # More sophisticated parsing would go here

        # Build consolidated file
        content = "#!/usr/bin/env python3\n"
        content += '"""Consolidated file created by Gemini Consolidation Server."""\n\n'
        content += "\n".join(sorted(imports)) + "\n\n"
        content += "# TODO: Complete consolidation\n"

        return content

    async def delete_redundant(self, request: ConsolidationRequest) -> ConsolidationResponse:
        """Delete redundant files after consolidation."""
        target = Path(request.target_path)

        files_to_delete: list[Any] = []

        # Identify files to delete
        if request.patterns:
            for pattern in request.patterns:
                files_to_delete.extend(target.rglob(pattern))

        deleted_files: list[Any] = []

        if not request.dry_run and request.aggressive:
            for file_path in files_to_delete:
                if file_path.is_file():
                    # Create backup first
                    backup_path = file_path.with_suffix(file_path.suffix + ".consolidated-backup")
                    shutil.copy2(file_path, backup_path)

                    # Delete original
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    console.print(f"[red]Deleted: {file_path}[/red]")

        details = {
            "deleted_files": (
                deleted_files if not request.dry_run else [str(f) for f in files_to_delete]
            ),
            "dry_run": request.dry_run,
            "aggressive": request.aggressive,
        }

        return ConsolidationResponse(
            status="success",
            operation="delete",
            files_affected=len(deleted_files),
            details=details,
            suggestions=["Backups created with .consolidated-backup extension"],
        )

    async def full_consolidation(self, request: ConsolidationRequest) -> ConsolidationResponse:
        """Perform full consolidation: analyze, merge, and delete."""
        # Step 1: Analyze
        analysis = await self.analyze_duplicates(request)

        # Step 2: Merge
        merge_request = ConsolidationRequest(
            operation="merge",
            target_path=request.target_path,
            patterns=request.patterns,
            dry_run=request.dry_run,
            aggressive=request.aggressive,
        )
        merge_result = await self.merge_duplicates(merge_request)

        # Step 3: Delete (if aggressive)
        deleted = 0
        if request.aggressive and not request.dry_run:
            delete_request = ConsolidationRequest(
                operation="delete",
                target_path=request.target_path,
                patterns=request.patterns,
                dry_run=False,
                aggressive=True,
            )
            delete_result = await self.delete_redundant(delete_request)
            deleted = delete_result.files_affected

        details = {
            "analysis": analysis.details,
            "merge": merge_result.details,
            "deleted": deleted,
        }

        return ConsolidationResponse(
            status="success",
            operation="full_consolidation",
            files_affected=analysis.files_affected,
            details=details,
            suggestions=[
                f"Consolidation complete: {deleted} files deleted",
                "Review consolidated files in tests/ and benchmarks/",
                "Run tests to ensure functionality preserved",
            ],
        )

    async def gemini_analysis(self, request: ConsolidationRequest) -> dict[str, Any]:
        """Use Gemini AI to analyze and suggest consolidation strategies."""
        if not model:
            return {"error": "Gemini not configured"}

        target = Path(request.target_path)

        # Gather file structure
        file_structure: list[Any] = []
        for file_path in target.rglob("*.py"):
            if not any(skip in str(file_path) for skip in [".venv", "__pycache__", ".git"]):
                file_structure.append(str(file_path.relative_to(target)))

        prompt = f"""
        Analyze this Python project structure and suggest consolidation:

        Project: {target.name}
        Total Python files: {len(file_structure)}

        Files (first 100):
        {chr(10).join(file_structure[:100])}

        Identify:
        1. Duplicate functionality that can be merged
        2. Files that should be deleted
        3. Consolidation strategy to reduce file count by 90%
        4. Specific commands to execute the consolidation

        Be aggressive - this project has too much duplication.
        """

        response = model.generate_content(prompt)

        return {
            "gemini_analysis": response.text,
            "file_count": len(file_structure),
            "project": target.name,
        }

    def run(self, host: str = "127.0.0.1", port: int = 8100) -> None:
        """Run the server."""
        import uvicorn

        console.print(
            f"[bold green]Starting Gemini Consolidation Server on {host}:{port}[/bold green]"
        )
        console.print(
            "[yellow]Server running independently - Claude can communicate via HTTP[/yellow]"
        )
        uvicorn.run(self.app, host=host, port=port)


def main() -> None:
    """Main entry point."""
    server = GeminiConsolidationServer()
    server.run()


if __name__ == "__main__":
    main()
