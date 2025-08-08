#!/usr/bin/env python3
"""Auto-Claude FastAPI Endpoints.

Provides REST API endpoints for enhanced TypeScript-aware auto-claude functionality.
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from gterminal.auto_claude_enhanced import AutoClaudeConfig
from gterminal.auto_claude_enhanced import DiagnosticResult
from gterminal.auto_claude_enhanced import EnhancedAutoClaude

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["auto-claude"])

# Active auto-claude sessions
active_sessions: dict[str, EnhancedAutoClaude] = {}


class AutoClaudeAnalysisRequest(BaseModel):
    """Request model for project analysis."""

    project_root: str
    files: list[str] | None = None
    languages: list[str] | None = None
    enable_typescript: bool = True
    enable_ai_fixes: bool = True
    backend_url: str = "http://127.0.0.1:8100"


class AutoClaudeFixRequest(BaseModel):
    """Request model for fixing issues."""

    session_id: str
    auto_fix: bool = True
    preview_only: bool = False
    max_fixes: int = 50


class ProjectAnalysisResponse(BaseModel):
    """Response model for project analysis."""

    session_id: str
    total_issues: int
    fixable_issues: int
    issues_by_language: dict[str, int]
    issues_by_severity: dict[str, int]
    diagnostics: list[DiagnosticResult]
    status: str = "completed"


class FixResponse(BaseModel):
    """Response model for fix operations."""

    session_id: str
    success: bool
    files_modified: list[str]
    diagnostics_fixed: int
    diagnostics_remaining: int
    errors: list[str] = Field(default_factory=list)
    status: str = "completed"


@router.post("/analyze", response_model=ProjectAnalysisResponse)
async def analyze_project(request: AutoClaudeAnalysisRequest):
    """Analyze a project for code issues across multiple languages with TypeScript support.

    This endpoint:
    1. Creates an enhanced auto-claude session
    2. Analyzes the project for issues in Python, TypeScript, JavaScript, Rust
    3. Returns detailed diagnostics with fix capabilities
    4. Maintains session for subsequent fix operations
    """
    try:
        # Create auto-claude configuration
        config = AutoClaudeConfig(
            project_root=request.project_root,
            backend_url=request.backend_url,
            enable_ai_fixes=request.enable_ai_fixes,
            enable_cross_project_learning=True,
            pre_commit_integration=True,
        )

        # Create enhanced auto-claude instance
        auto_claude = EnhancedAutoClaude(config)

        # Initialize session with backend
        session_id = await auto_claude.initialize_session()
        if not session_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize session with unified Gemini server",
            )

        # Store session for later use
        active_sessions[session_id] = auto_claude

        # Analyze project
        diagnostics = await auto_claude.analyze_project(request.files)

        # Process diagnostics for response
        issues_by_language: dict[str, Any] = {}
        issues_by_severity: dict[str, Any] = {}
        fixable_count = 0

        for diagnostic in diagnostics:
            # Count by language
            language = auto_claude._detect_language(diagnostic.file_path) or "unknown"
            issues_by_language[language] = issues_by_language.get(language, 0) + 1

            # Count by severity
            issues_by_severity[diagnostic.severity] = (
                issues_by_severity.get(diagnostic.severity, 0) + 1
            )

            # Count fixable issues
            if diagnostic.fixable:
                fixable_count += 1

        logger.info(f"Analysis completed for session {session_id}: {len(diagnostics)} issues found")

        return ProjectAnalysisResponse(
            session_id=session_id,
            total_issues=len(diagnostics),
            fixable_issues=fixable_count,
            issues_by_language=issues_by_language,
            issues_by_severity=issues_by_severity,
            diagnostics=diagnostics,
        )

    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e!s}")


@router.post("/fix", response_model=FixResponse)
async def fix_issues(request: AutoClaudeFixRequest):
    """Fix code issues using automatic tools and AI-powered fixes.

    This endpoint:
    1. Retrieves the active auto-claude session
    2. Applies automatic fixes using standard tools (prettier, eslint, black, ruff)
    3. Uses AI fixes for complex issues via the unified Gemini server
    4. Returns results of fix operations
    """
    try:
        # Get active session
        if request.session_id not in active_sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session {request.session_id} not found or expired",
            )

        auto_claude = active_sessions[request.session_id]

        # Get current diagnostics
        diagnostics = await auto_claude.analyze_project()

        if not diagnostics:
            return FixResponse(
                session_id=request.session_id,
                success=True,
                files_modified=[],
                diagnostics_fixed=0,
                diagnostics_remaining=0,
                status="no_issues_found",
            )

        if request.preview_only:
            # Preview mode - show what would be fixed
            fixable_diagnostics = [d for d in diagnostics if d.fixable]
            return FixResponse(
                session_id=request.session_id,
                success=True,
                files_modified=[d.file_path for d in fixable_diagnostics],
                diagnostics_fixed=0,
                diagnostics_remaining=len(diagnostics),
                status="preview_completed",
            )

        # Apply fixes
        fix_result = await auto_claude.fix_issues(diagnostics)

        logger.info(
            f"Fix completed for session {request.session_id}: {fix_result.diagnostics_fixed} fixed"
        )

        return FixResponse(
            session_id=request.session_id,
            success=fix_result.success,
            files_modified=fix_result.files_modified,
            diagnostics_fixed=fix_result.diagnostics_fixed,
            diagnostics_remaining=fix_result.diagnostics_remaining,
            errors=fix_result.errors,
        )

    except Exception as e:
        logger.exception(f"Fix operation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fix operation failed: {e!s}")


@router.get("/sessions")
async def list_sessions():
    """List active auto-claude sessions."""
    sessions: list[Any] = []
    for session_id, auto_claude in active_sessions.items():
        sessions.append(
            {
                "session_id": session_id,
                "project_root": auto_claude.config.project_root,
                "backend_url": auto_claude.config.backend_url,
                "created": "active",  # Could add timestamp tracking
            },
        )

    return {"active_sessions": len(sessions), "sessions": sessions}


@router.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    """Close an active auto-claude session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    auto_claude = active_sessions[session_id]
    await auto_claude.close()
    del active_sessions[session_id]

    return {"message": f"Session {session_id} closed successfully"}


@router.post("/install-hooks")
async def install_pre_commit_hooks(project_root: str):
    """Install enhanced pre-commit hooks with TypeScript support."""
    try:
        config = AutoClaudeConfig(project_root=project_root)
        auto_claude = EnhancedAutoClaude(config)

        success = await auto_claude.install_pre_commit_hooks()

        if success:
            return {"message": "Pre-commit hooks installed successfully"}
        raise HTTPException(status_code=500, detail="Failed to install pre-commit hooks")

    except Exception as e:
        logger.exception(f"Hook installation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hook installation failed: {e!s}")


@router.post("/pre-commit-check")
async def run_pre_commit_check(project_root: str):
    """Run pre-commit checks for staged files."""
    try:
        config = AutoClaudeConfig(project_root=project_root)
        auto_claude = EnhancedAutoClaude(config)

        success = await auto_claude.run_pre_commit_checks()

        return {
            "success": success,
            "message": ("Pre-commit checks passed" if success else "Pre-commit checks failed"),
        }

    except Exception as e:
        logger.exception(f"Pre-commit check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pre-commit check failed: {e!s}")


class TypeScriptAnalysisRequest(BaseModel):
    """Request model for TypeScript-specific analysis."""

    project_root: str
    files: list[str] | None = None
    create_tsconfig: bool = True


@router.post("/typescript/analyze")
async def analyze_typescript(request: TypeScriptAnalysisRequest):
    """Analyze TypeScript files specifically.

    This endpoint provides TypeScript-specific analysis including:
    - Type checking with tsc
    - Code quality with ESLint
    - Formatting with Prettier
    - Custom TypeScript configuration creation
    """
    try:
        from gterminal.auto_claude_enhanced import TypeScriptAnalyzer

        project_root = Path(request.project_root)
        analyzer = TypeScriptAnalyzer(project_root)

        # Analyze TypeScript files
        diagnostics = await analyzer.analyze_typescript(request.files)

        # Group diagnostics by source
        diagnostics_by_source: dict[str, Any] = {}
        for diagnostic in diagnostics:
            source = diagnostic.source
            if source not in diagnostics_by_source:
                diagnostics_by_source[source] = []
            diagnostics_by_source[source].append(diagnostic)

        return {
            "total_issues": len(diagnostics),
            "issues_by_source": {k: len(v) for k, v in diagnostics_by_source.items()},
            "fixable_issues": sum(1 for d in diagnostics if d.fixable),
            "diagnostics": diagnostics,
            "tsconfig_found": analyzer.tsconfig_path is not None,
            "tsconfig_path": (str(analyzer.tsconfig_path) if analyzer.tsconfig_path else None),
        }

    except Exception as e:
        logger.exception(f"TypeScript analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"TypeScript analysis failed: {e!s}")


@router.get("/health")
async def health_check():
    """Health check endpoint for auto-claude service."""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "service": "enhanced-auto-claude",
        "features": [
            "multi-language-support",
            "typescript-analysis",
            "ai-powered-fixes",
            "pre-commit-integration",
            "cross-project-learning",
        ],
    }


# Cleanup function to be called on shutdown
async def cleanup_sessions() -> None:
    """Clean up all active sessions."""
    for session_id, auto_claude in active_sessions.items():
        try:
            await auto_claude.close()
        except Exception as e:
            logger.exception(f"Error closing session {session_id}: {e}")

    active_sessions.clear()
    logger.info("All auto-claude sessions cleaned up")
