"""MCP Gemini Integration - Client for accessing existing MCP Gemini servers.

This module provides access to the production-ready MCP Gemini servers that contain
the actual AI analysis implementations, ensuring that unified agents use real
functionality instead of placeholder code.

INTEGRATES WITH:
- gemini_code_reviewer.py (MCP server)
- gemini_workspace_analyzer.py (MCP server)
- gemini_master_architect.py (MCP server)
- Other MCP Gemini implementations

PROVIDES:
- Direct function calls to MCP server tools
- Connection pooling and error handling
- Result caching and optimization
- Async integration for unified agents
"""

import asyncio
import builtins
import contextlib
import json
import logging
import os
import tempfile
from typing import Any

from gterminal.core.security.security_utils import safe_json_parse
from gterminal.core.security.security_utils import safe_subprocess_run

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPGeminiClient:
    """Client for accessing MCP Gemini servers with production AI analysis.

    Provides direct access to the sophisticated MCP server implementations
    that contain real Gemini AI integration, ensuring unified agents use
    actual analysis capabilities rather than stub implementations.
    """

    def __init__(self) -> None:
        self.mcp_servers = {
            "code_reviewer": "app.mcp_servers.gemini_code_reviewer",
            "workspace_analyzer": "app.mcp_servers.gemini_workspace_analyzer",
            "master_architect": "app.mcp_servers.gemini_master_architect",
            "documentation_generator": "app.mcp_servers.gemini_documentation_generator",
        }

        # Connection pool for MCP server processes
        self._server_processes = {}
        self._connection_lock = asyncio.Lock()

        logger.info("Initialized MCP Gemini Client")

    async def review_code_comprehensive(
        self,
        file_path: str,
        focus_areas: str = "security,performance,quality",
        include_suggestions: str = "true",
        severity_threshold: str = "medium",
    ) -> dict[str, Any]:
        """Call the production gemini_code_reviewer MCP server for comprehensive analysis.

        This uses the actual MCP server with Gemini 2.0 Flash integration,
        advanced caching, PyO3 Rust bindings, and sophisticated analysis.
        """
        try:
            # Call the actual MCP server tool
            result = await self._call_mcp_tool(
                "code_reviewer",
                "review_code",
                {
                    "file_path": file_path,
                    "focus_areas": focus_areas,
                    "include_suggestions": include_suggestions,
                    "severity_threshold": severity_threshold,
                },
            )

            return {
                "status": "success",
                "result": result,
                "source": "mcp_gemini_code_reviewer",
            }

        except Exception as e:
            logger.exception(f"MCP code review failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "source": "mcp_gemini_code_reviewer",
            }

    async def review_security_comprehensive(
        self,
        directory: str,
        file_patterns: str = "*.py,*.js,*.ts,*.java,*.rs,*.go",
        scan_depth: str = "comprehensive",
    ) -> dict[str, Any]:
        """Call the production gemini_code_reviewer security scan functionality.

        Uses the sophisticated security analysis with CWE mappings,
        threat detection, and pattern recognition from the MCP server.
        """
        try:
            result = await self._call_mcp_tool(
                "code_reviewer",
                "review_security",
                {
                    "directory": directory,
                    "file_patterns": file_patterns,
                    "scan_depth": scan_depth,
                },
            )

            return {
                "status": "success",
                "result": result,
                "source": "mcp_gemini_code_reviewer",
            }

        except Exception as e:
            logger.exception(f"MCP security scan failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "source": "mcp_gemini_code_reviewer",
            }

    async def comprehensive_analysis(
        self,
        target_path: str,
        analysis_types: str = "code_quality,security,performance",
        file_patterns: str = "*.py,*.js,*.ts,*.rs,*.go",
        max_files: str = "50",
    ) -> dict[str, Any]:
        """Call the production comprehensive analysis from gemini_code_reviewer.

        Uses the full AI-powered analysis combining multiple review types
        with advanced pattern detection and Gemini insights.
        """
        try:
            result = await self._call_mcp_tool(
                "code_reviewer",
                "comprehensive_analysis",
                {
                    "target_path": target_path,
                    "analysis_types": analysis_types,
                    "file_patterns": file_patterns,
                    "max_files": max_files,
                },
            )

            return {
                "status": "success",
                "result": result,
                "source": "mcp_gemini_code_reviewer",
            }

        except Exception as e:
            logger.exception(f"MCP comprehensive analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "source": "mcp_gemini_code_reviewer",
            }

    async def analyze_workspace_comprehensive(
        self,
        project_path: str,
        analysis_depth: str = "comprehensive",
        include_dependencies: str = "true",
        include_tests: str = "true",
    ) -> dict[str, Any]:
        """Call the production gemini_workspace_analyzer for full project analysis.

        Uses PyO3 Rust integration, advanced caching, and Gemini AI analysis
        for comprehensive workspace insights.
        """
        try:
            result = await self._call_mcp_tool(
                "workspace_analyzer",
                "analyze_workspace",
                {
                    "project_path": project_path,
                    "analysis_depth": analysis_depth,
                    "include_dependencies": include_dependencies,
                    "include_tests": include_tests,
                },
            )

            return {
                "status": "success",
                "result": result,
                "source": "mcp_gemini_workspace_analyzer",
            }

        except Exception as e:
            logger.exception(f"MCP workspace analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "source": "mcp_gemini_workspace_analyzer",
            }

    async def search_workspace_content(
        self,
        directory: str,
        search_pattern: str,
        file_patterns: str = "*.py,*.js,*.ts,*.rs,*.go",
        max_results: int = 100,
    ) -> dict[str, Any]:
        """Call the workspace analyzer content search functionality.

        Uses high-performance search with parallel processing and
        intelligent result ranking.
        """
        try:
            result = await self._call_mcp_tool(
                "workspace_analyzer",
                "search_workspace_content",
                {
                    "directory": directory,
                    "search_pattern": search_pattern,
                    "file_patterns": file_patterns,
                    "max_results": str(max_results),
                },
            )

            return {
                "status": "success",
                "result": result,
                "source": "mcp_gemini_workspace_analyzer",
            }

        except Exception as e:
            logger.exception(f"MCP content search failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "source": "mcp_gemini_workspace_analyzer",
            }

    async def get_project_overview(self, project_path: str) -> dict[str, Any]:
        """Call workspace analyzer for high-level project overview.

        Provides technology stack detection, architecture analysis,
        and project health assessment.
        """
        try:
            result = await self._call_mcp_tool(
                "workspace_analyzer",
                "get_project_overview",
                {"project_path": project_path},
            )

            return {
                "status": "success",
                "result": result,
                "source": "mcp_gemini_workspace_analyzer",
            }

        except Exception as e:
            logger.exception(f"MCP project overview failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "source": "mcp_gemini_workspace_analyzer",
            }

    async def analyze_architecture_patterns(
        self,
        project_path: str,
        analysis_depth: str = "comprehensive",
    ) -> dict[str, Any]:
        """Call gemini_master_architect for architecture pattern analysis.

        Uses sophisticated pattern recognition and architectural
        recommendations from the master architect MCP server.
        """
        try:
            result = await self._call_mcp_tool(
                "master_architect",
                "analyze_architecture",
                {"project_path": project_path, "analysis_depth": analysis_depth},
            )

            return {
                "status": "success",
                "result": result,
                "source": "mcp_gemini_master_architect",
            }

        except Exception as e:
            logger.exception(f"MCP architecture analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "source": "mcp_gemini_master_architect",
            }

    async def generate_documentation_ai(
        self,
        source_path: str,
        doc_type: str = "readme",
        target_audience: str = "developers",
        include_examples: str = "true",
    ) -> dict[str, Any]:
        """Call documentation generator MCP server for AI-powered docs.

        Uses advanced template system and Gemini AI for generating
        comprehensive, context-aware documentation.
        """
        try:
            result = await self._call_mcp_tool(
                "documentation_generator",
                "generate_documentation",
                {
                    "source_path": source_path,
                    "doc_type": doc_type,
                    "target_audience": target_audience,
                    "include_examples": include_examples,
                },
            )

            return {
                "status": "success",
                "result": result,
                "source": "mcp_gemini_documentation_generator",
            }

        except Exception as e:
            logger.exception(f"MCP documentation generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "source": "mcp_gemini_documentation_generator",
            }

    async def _call_mcp_tool(
        self, server_name: str, tool_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Internal method to call MCP server tools.

        This creates a temporary script that imports and calls the MCP server
        tool directly, then parses the JSON result.
        """
        async with self._connection_lock:
            try:
                server_module = self.mcp_servers.get(server_name)
                if not server_module:
                    msg = f"Unknown MCP server: {server_name}"
                    raise ValueError(msg)

                # Create temporary script to call MCP tool
                script_content = f"""
import sys
import asyncio
import json
import os

# Set environment variables for Vertex AI
os.environ.setdefault('GOOGLE_CLOUD_PROJECT', 'auricleinc-gemini')
os.environ.setdefault('VERTEX_AI_LOCATION', 'us-central1')

# Import the MCP server module
from {server_module} import {tool_name}

async def call_tool():
    try:
        # Call the tool with parameters
        result = await {tool_name}(**{json.dumps(parameters)})
        print("MCP_RESULT_START")
        print(json.dumps(result, default=str))
        print("MCP_RESULT_END")
    except Exception as e:
        print("MCP_ERROR_START")
        print(json.dumps({{"error": str(e)}}))
        print("MCP_ERROR_END")

if __name__ == "__main__":
    asyncio.run(call_tool())
"""

                # Write script to temporary file
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(script_content)
                    script_path = f.name

                try:
                    # Execute the script
                    result = safe_subprocess_run(
                        ["python", script_path], timeout=300
                    )  # 5 minute timeout

                    output = result.stdout

                    # Parse result from output
                    if "MCP_RESULT_START" in output:
                        start_idx = output.find("MCP_RESULT_START") + len("MCP_RESULT_START\n")
                        end_idx = output.find("MCP_RESULT_END")
                        result_json = output[start_idx:end_idx].strip()
                        return safe_json_parse(result_json)

                    if "MCP_ERROR_START" in output:
                        start_idx = output.find("MCP_ERROR_START") + len("MCP_ERROR_START\n")
                        end_idx = output.find("MCP_ERROR_END")
                        error_json = output[start_idx:end_idx].strip()
                        error_data = safe_json_parse(error_json)
                        raise Exception(error_data.get("error", "Unknown MCP error"))

                    msg = f"Unexpected MCP output: {output}"
                    raise Exception(msg)

                finally:
                    # Clean up temporary file
                    with contextlib.suppress(builtins.BaseException):
                        os.unlink(script_path)

            except Exception as e:
                logger.exception(f"MCP tool call failed: {e}")
                raise

    async def health_check(self) -> dict[str, Any]:
        """Check health of all MCP Gemini servers.

        Returns status information for connection health monitoring.
        """
        health_status = {"overall": "healthy", "servers": {}, "errors": []}

        # Test each server with a simple call
        for server_name in self.mcp_servers:
            try:
                if server_name == "code_reviewer":
                    # Simple health check - this should be fast
                    await self._call_mcp_tool(server_name, "get_cache_stats", {})
                    health_status["servers"][server_name] = "healthy"

                elif server_name == "workspace_analyzer":
                    # Simple test with current directory
                    await self._call_mcp_tool(
                        server_name, "get_project_overview", {"project_path": "."}
                    )
                    health_status["servers"][server_name] = "healthy"

                else:
                    health_status["servers"][server_name] = "not_tested"

            except Exception as e:
                health_status["servers"][server_name] = "unhealthy"
                health_status["errors"].append(f"{server_name}: {e!s}")
                health_status["overall"] = "degraded"

        return health_status

    async def cleanup(self) -> None:
        """Cleanup MCP client resources."""
        # Clean up any server processes
        for process in self._server_processes.values():
            try:
                process.terminate()
                await asyncio.sleep(0.1)
                if process.poll() is None:
                    process.kill()
            except:
                pass

        self._server_processes.clear()
        logger.info("MCP Gemini Client cleanup completed")


# Global instance
mcp_gemini_client = MCPGeminiClient()
