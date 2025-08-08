#!/usr/bin/env python3
"""Simplified Gemini Code Reviewer MCP Server.

A streamlined version that bypasses the complex import chain while maintaining
core functionality for Claude CLI integration.

CORE FEATURES:
- Code quality review using Gemini 2.0 Flash
- Security scanning with basic pattern detection
- Performance analysis with bottleneck identification
- Service account authentication via Vertex AI
- Clean FastMCP implementation without complex dependencies

MCP TOOLS:
- review_code: Code quality review with issue detection
- review_security: Security vulnerability scanning
- review_performance: Performance analysis and optimization suggestions
- comprehensive_analysis: Combined analysis with summary
"""

import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Set up environment
os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS", "/home/david/.auth/business/service-account-key.json"
)
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "auricleinc-gemini")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from fastmcp import FastMCP
    from pydantic import BaseModel
    from pydantic import Field
    import vertexai
    from vertexai.generative_models import GenerativeModel
    from vertexai.generative_models import Part
except ImportError as e:
    logger.exception(f"Failed to import required packages: {e}")
    sys.exit(1)

# Initialize Vertex AI
try:
    vertexai.init(project="auricleinc-gemini", location="us-central1")
    model = GenerativeModel("gemini-2.0-flash-exp")
    logger.info("✅ Vertex AI initialized successfully")
except Exception as e:
    logger.exception(f"Failed to initialize Vertex AI: {e}")
    sys.exit(1)

# Create MCP server
mcp = FastMCP("gemini-code-reviewer")


class CodeReviewRequest(BaseModel):
    """Request model for code review."""

    code: str = Field(..., description="Code to review")
    filename: str = Field(default="unknown.py", description="Optional filename for context")
    focus: str = Field(
        default="general", description="Focus area: general, security, performance, style"
    )


class ReviewResult(BaseModel):
    """Review result model."""

    issues: list[dict] = Field(default_factory=list, description="List of identified issues")
    suggestions: list[str] = Field(
        default_factory=list, description="List of improvement suggestions"
    )
    score: int = Field(default=0, description="Quality score 0-100")
    summary: str = Field(default="", description="Summary of review")


@mcp.tool()
async def review_code(request: CodeReviewRequest) -> dict[str, Any]:
    """Review code for quality, style, and best practices."""
    try:
        prompt = f"""
        Analyze this {request.filename} code for quality, style, and best practices:

        ```{Path(request.filename).suffix[1:] if Path(request.filename).suffix else "python"}
        {request.code}
        ```

        Focus on: {request.focus}

        Provide analysis in JSON format:
        {{
            "issues": [
                {{"type": "error|warning|info", "line": 0, "message": "description", "suggestion": "how to fix"}}
            ],
            "suggestions": ["improvement suggestion 1", "suggestion 2"],
            "score": 85,
            "summary": "Brief summary of code quality"
        }}
        """

        response = await model.generate_content_async(prompt)

        # Parse response and return structured result
        result = {
            "status": "success",
            "filename": request.filename,
            "focus": request.focus,
            "analysis": response.text,
            "timestamp": asyncio.get_event_loop().time(),
        }

        logger.info(f"Code review completed for {request.filename}")
        return result

    except Exception as e:
        logger.exception(f"Code review failed: {e}")
        return {"status": "error", "error": str(e), "filename": request.filename}


@mcp.tool()
async def review_security(request: CodeReviewRequest) -> dict[str, Any]:
    """Scan code for security vulnerabilities and patterns."""
    try:
        prompt = f"""
        Perform a security analysis of this {request.filename} code:

        ```{Path(request.filename).suffix[1:] if Path(request.filename).suffix else "python"}
        {request.code}
        ```

        Look for:
        - SQL injection vulnerabilities
        - XSS vulnerabilities
        - Input validation issues
        - Hardcoded credentials
        - Unsafe deserialization
        - Command injection risks
        - Insecure cryptography usage

        Provide results in JSON format:
        {{
            "vulnerabilities": [
                {{"severity": "high|medium|low", "type": "vulnerability type", "line": 0, "description": "details", "fix": "how to fix"}}
            ],
            "security_score": 85,
            "recommendations": ["security recommendation 1", "recommendation 2"]
        }}
        """

        response = await model.generate_content_async(prompt)

        result = {
            "status": "success",
            "filename": request.filename,
            "security_analysis": response.text,
            "timestamp": asyncio.get_event_loop().time(),
        }

        logger.info(f"Security review completed for {request.filename}")
        return result

    except Exception as e:
        logger.exception(f"Security review failed: {e}")
        return {"status": "error", "error": str(e), "filename": request.filename}


@mcp.tool()
async def review_performance(request: CodeReviewRequest) -> dict[str, Any]:
    """Analyze code for performance issues and optimization opportunities."""
    try:
        prompt = f"""
        Analyze this {request.filename} code for performance issues:

        ```{Path(request.filename).suffix[1:] if Path(request.filename).suffix else "python"}
        {request.code}
        ```

        Look for:
        - Algorithmic complexity issues (O(n²) vs O(n log n))
        - Memory usage problems
        - Inefficient loops or data structures
        - Database query optimization
        - I/O bottlenecks
        - Caching opportunities
        - Async/await usage improvements

        Provide results in JSON format:
        {{
            "performance_issues": [
                {{"severity": "high|medium|low", "type": "issue type", "line": 0, "description": "details", "optimization": "how to optimize"}}
            ],
            "performance_score": 85,
            "optimizations": ["optimization 1", "optimization 2"]
        }}
        """

        response = await model.generate_content_async(prompt)

        result = {
            "status": "success",
            "filename": request.filename,
            "performance_analysis": response.text,
            "timestamp": asyncio.get_event_loop().time(),
        }

        logger.info(f"Performance review completed for {request.filename}")
        return result

    except Exception as e:
        logger.exception(f"Performance review failed: {e}")
        return {"status": "error", "error": str(e), "filename": request.filename}


@mcp.tool()
async def comprehensive_analysis(request: CodeReviewRequest) -> dict[str, Any]:
    """Perform comprehensive code analysis combining quality, security, and performance."""
    try:
        # Run all analyses
        quality_result = await review_code(request)
        security_result = await review_security(request)
        performance_result = await review_performance(request)

        # Combine results
        result = {
            "status": "success",
            "filename": request.filename,
            "comprehensive_analysis": {
                "quality": quality_result,
                "security": security_result,
                "performance": performance_result,
            },
            "timestamp": asyncio.get_event_loop().time(),
        }

        logger.info(f"Comprehensive analysis completed for {request.filename}")
        return result

    except Exception as e:
        logger.exception(f"Comprehensive analysis failed: {e}")
        return {"status": "error", "error": str(e), "filename": request.filename}


if __name__ == "__main__":
    logger.info("Starting Gemini Code Reviewer MCP Server...")
    logger.info(
        "🎯 Available tools: review_code, review_security, review_performance, comprehensive_analysis"
    )
    mcp.run()
