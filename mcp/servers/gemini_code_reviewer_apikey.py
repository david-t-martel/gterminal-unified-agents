#!/usr/bin/env python3
"""Gemini Code Reviewer MCP Server - API Key Version.

Uses Google AI SDK with API key instead of Vertex AI to bypass
authentication and import issues.

CORE FEATURES:
- Code quality review using Gemini 2.0 Flash
- Security scanning with pattern detection
- Performance analysis with optimization suggestions
- API key authentication (no service account needed)
- Clean FastMCP implementation

MCP TOOLS:
- review_code: Code quality review
- review_security: Security vulnerability scanning
- review_performance: Performance analysis
- comprehensive_analysis: Combined analysis
"""

import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Set up environment from gcp-profile
os.environ.setdefault("LOG_LEVEL", "INFO")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from fastmcp import FastMCP
    import google.generativeai as genai
    from pydantic import BaseModel
    from pydantic import Field
except ImportError as e:
    logger.exception(f"Failed to import required packages: {e}")
    sys.exit(1)

# Get API key from environment (set by gcp-profile)
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("No Google API key found. Run 'gcp-profile business' first.")
    sys.exit(1)

# Configure Google AI SDK
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    logger.info("✅ Google AI SDK configured successfully")
except Exception as e:
    logger.exception(f"Failed to configure Google AI SDK: {e}")
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


@mcp.tool()
async def review_code(request: CodeReviewRequest) -> dict[str, Any]:
    """Review code for quality, style, and best practices."""
    try:
        file_ext = Path(request.filename).suffix[1:] if Path(request.filename).suffix else "python"

        prompt = f"""
        Analyze this {request.filename} code for quality, style, and best practices:

        ```{file_ext}
        {request.code}
        ```

        Focus on: {request.focus}

        Provide analysis in this JSON structure:
        {{
            "issues": [
                {{"type": "error|warning|info", "line": 0, "message": "description", "suggestion": "how to fix"}}
            ],
            "suggestions": ["improvement suggestion 1", "suggestion 2"],
            "score": 85,
            "summary": "Brief summary of code quality"
        }}

        Be specific about line numbers when possible and provide actionable suggestions.
        """

        response = await asyncio.to_thread(model.generate_content, prompt)

        result = {
            "status": "success",
            "filename": request.filename,
            "focus": request.focus,
            "analysis": response.text,
            "timestamp": asyncio.get_event_loop().time(),
        }

        logger.info(f"✅ Code review completed for {request.filename}")
        return result

    except Exception as e:
        logger.exception(f"❌ Code review failed: {e}")
        return {"status": "error", "error": str(e), "filename": request.filename}


@mcp.tool()
async def review_security(request: CodeReviewRequest) -> dict[str, Any]:
    """Scan code for security vulnerabilities and patterns."""
    try:
        file_ext = Path(request.filename).suffix[1:] if Path(request.filename).suffix else "python"

        prompt = f"""
        Perform a comprehensive security analysis of this {request.filename} code:

        ```{file_ext}
        {request.code}
        ```

        Look for these security issues:
        - SQL injection vulnerabilities
        - Cross-site scripting (XSS) risks
        - Input validation problems
        - Hardcoded passwords/secrets
        - Unsafe deserialization
        - Command injection vulnerabilities
        - Cryptographic weaknesses
        - Path traversal risks
        - Authentication/authorization flaws

        Provide results in this JSON structure:
        {{
            "vulnerabilities": [
                {{"severity": "critical|high|medium|low", "type": "vulnerability type", "line": 0, "cwe": "CWE-89", "description": "detailed explanation", "fix": "specific fix recommendation"}}
            ],
            "security_score": 85,
            "recommendations": ["security recommendation 1", "recommendation 2"],
            "summary": "Overall security assessment"
        }}

        Include CWE identifiers where applicable and be specific about remediation.
        """

        response = await asyncio.to_thread(model.generate_content, prompt)

        result = {
            "status": "success",
            "filename": request.filename,
            "security_analysis": response.text,
            "timestamp": asyncio.get_event_loop().time(),
        }

        logger.info(f"🔒 Security review completed for {request.filename}")
        return result

    except Exception as e:
        logger.exception(f"❌ Security review failed: {e}")
        return {"status": "error", "error": str(e), "filename": request.filename}


@mcp.tool()
async def review_performance(request: CodeReviewRequest) -> dict[str, Any]:
    """Analyze code for performance issues and optimization opportunities."""
    try:
        file_ext = Path(request.filename).suffix[1:] if Path(request.filename).suffix else "python"

        prompt = f"""
        Analyze this {request.filename} code for performance bottlenecks and optimization opportunities:

        ```{file_ext}
        {request.code}
        ```

        Focus on:
        - Algorithmic complexity (Big O analysis)
        - Memory usage patterns
        - Loop efficiency and data structure choices
        - Database query optimization
        - I/O operations and caching opportunities
        - Async/await usage for concurrency
        - Resource management and cleanup
        - Hot path optimizations

        Provide results in this JSON structure:
        {{
            "performance_issues": [
                {{"severity": "critical|high|medium|low", "type": "performance issue", "line": 0, "complexity": "O(n²)", "description": "detailed analysis", "optimization": "specific optimization strategy"}}
            ],
            "performance_score": 85,
            "optimizations": ["optimization opportunity 1", "optimization 2"],
            "summary": "Overall performance assessment"
        }}

        Include complexity analysis and quantify improvements where possible.
        """

        response = await asyncio.to_thread(model.generate_content, prompt)

        result = {
            "status": "success",
            "filename": request.filename,
            "performance_analysis": response.text,
            "timestamp": asyncio.get_event_loop().time(),
        }

        logger.info(f"⚡ Performance review completed for {request.filename}")
        return result

    except Exception as e:
        logger.exception(f"❌ Performance review failed: {e}")
        return {"status": "error", "error": str(e), "filename": request.filename}


@mcp.tool()
async def comprehensive_analysis(request: CodeReviewRequest) -> dict[str, Any]:
    """Perform comprehensive code analysis combining quality, security, and performance."""
    try:
        logger.info(f"🔍 Starting comprehensive analysis for {request.filename}")

        # Run all analyses concurrently for better performance
        results = await asyncio.gather(
            review_code(request),
            review_security(request),
            review_performance(request),
            return_exceptions=True,
        )

        quality_result, security_result, performance_result = results

        # Handle any exceptions
        if isinstance(quality_result, Exception):
            logger.error(f"Quality analysis failed: {quality_result}")
            quality_result = {"status": "error", "error": str(quality_result)}

        if isinstance(security_result, Exception):
            logger.error(f"Security analysis failed: {security_result}")
            security_result = {"status": "error", "error": str(security_result)}

        if isinstance(performance_result, Exception):
            logger.error(f"Performance analysis failed: {performance_result}")
            performance_result = {"status": "error", "error": str(performance_result)}

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

        logger.info(f"🎯 Comprehensive analysis completed for {request.filename}")
        return result

    except Exception as e:
        logger.exception(f"❌ Comprehensive analysis failed: {e}")
        return {"status": "error", "error": str(e), "filename": request.filename}


if __name__ == "__main__":
    logger.info("🚀 Starting Gemini Code Reviewer MCP Server (API Key Version)")
    logger.info(
        "🔧 Available tools: review_code, review_security, review_performance, comprehensive_analysis"
    )
    logger.info(f"🔑 Using API key: {api_key[:12]}...")
    mcp.run()
