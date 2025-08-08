#!/usr/bin/env python3
"""Gemini Server Agent MCP Server - Unified Integration.

MCP server implementation that integrates with the GeminiUnifiedServer for comprehensive
AI agent capabilities with Claude CLI integration.

Features:
- FastMCP framework compliance for MCP Inspector validation
- Integration with GeminiUnifiedServer for session management
- Rust extensions for high-performance operations
- Authentication and cost optimization
- Multi-agent orchestration capabilities
- Real-time monitoring and metrics

All tools use proper type hints and follow MCP protocol standards.
"""

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import GeminiUnifiedServer
# MCP Framework
from fastmcp import FastMCP

from gterminal.gemini_unified_server import TaskPriority
from gterminal.gemini_unified_server import TaskRequest
from gterminal.gemini_unified_server import UnifiedGeminiServer

# Import Rust extensions with fallback handling
try:
    from gterminal.utils.rust_extensions import RUST_EXTENSIONS_AVAILABLE
    from gterminal.utils.rust_extensions import EnhancedAuthValidator
    from gterminal.utils.rust_extensions import EnhancedFileOps
    from gterminal.utils.rust_extensions import EnhancedJsonProcessor
    from gterminal.utils.rust_extensions import EnhancedRustCache
    from gterminal.utils.rust_extensions import get_performance_metrics
    from gterminal.utils.rust_extensions import get_system_info

    rust_available = RUST_EXTENSIONS_AVAILABLE
except ImportError:
    rust_available = False

    # Fallback classes for when Rust extensions aren't available
    class MockEnhanced:
        def __init__(self, *args, **kwargs) -> None:
            pass

    EnhancedFileOps = MockEnhanced
    EnhancedRustCache = MockEnhanced
    EnhancedJsonProcessor = MockEnhanced
    EnhancedAuthValidator = MockEnhanced

    def get_system_info() -> None:
        return {"available": False, "message": "Rust extensions not available"}

    def get_performance_metrics() -> None:
        return {"available": False, "message": "Rust extensions not available"}


# Initialize MCP server
mcp = FastMCP("gemini-server-agent")

# Global server instance
unified_server: UnifiedGeminiServer | None = None

# Initialize Rust extensions
rust_file_ops = EnhancedFileOps() if rust_available else None
rust_cache = EnhancedRustCache(capacity=10000, ttl_seconds=3600) if rust_available else None
rust_json = EnhancedJsonProcessor() if rust_available else None
rust_auth = EnhancedAuthValidator() if rust_available else None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


async def get_server() -> UnifiedGeminiServer:
    """Get or initialize the unified Gemini server."""
    global unified_server

    if unified_server is None:
        unified_server = UnifiedGeminiServer()
        await unified_server.start_async_components()
        logger.info("🤖 GeminiUnifiedServer initialized successfully")

    return unified_server


@mcp.tool()
async def analyze_project(
    project_path: str,
    analysis_depth: str = "comprehensive",
    include_dependencies: bool = True,
    include_security: bool = True,
    output_format: str = "json",
) -> dict[str, Any]:
    """Analyze project structure, dependencies, and code quality.

    Args:
        project_path: Path to the project directory to analyze
        analysis_depth: Analysis depth (quick, standard, comprehensive)
        include_dependencies: Whether to analyze project dependencies
        include_security: Whether to include security analysis
        output_format: Output format (json, markdown, summary)

    Returns:
        Comprehensive project analysis with structure, metrics, and recommendations

    """
    try:
        server = await get_server()

        # Create analysis task
        task = TaskRequest(
            task_type="analysis",
            instruction=f"Analyze project at {project_path} with {analysis_depth} depth. "
            f"Include dependencies: {include_dependencies}, security: {include_security}",
            target_path=project_path,
            options={
                "analysis_depth": analysis_depth,
                "include_dependencies": include_dependencies,
                "include_security": include_security,
                "output_format": output_format,
            },
            priority=TaskPriority.HIGH,
        )

        # Process with session management
        result = await server.process_task_with_session(task)

        # Use Rust extensions for file operations if available
        if rust_available and rust_file_ops:
            try:
                project_stats = rust_file_ops.get_directory_stats(project_path)
                result["project_stats"] = project_stats
            except Exception as e:
                logger.warning(f"Rust file operations failed: {e}")

        return {
            "success": True,
            "task_type": "analyze_project",
            "project_path": project_path,
            "analysis": result,
            "rust_enhanced": rust_available,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Project analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_type": "analyze_project",
            "project_path": project_path,
        }


@mcp.tool()
async def generate_code(
    instruction: str,
    language: str = "python",
    framework: str = "",
    style: str = "production",
    include_tests: bool = True,
    include_docs: bool = True,
) -> dict[str, Any]:
    """Generate code based on natural language instructions.

    Args:
        instruction: Natural language description of code to generate
        language: Programming language (python, typescript, rust, go)
        framework: Framework to use (fastapi, react, django, etc.)
        style: Code style (production, prototype, minimal)
        include_tests: Whether to generate unit tests
        include_docs: Whether to include documentation

    Returns:
        Generated code with tests, documentation, and implementation guidance

    """
    try:
        server = await get_server()

        # Enhanced instruction with context
        enhanced_instruction = f"""
        Generate {language} code for: {instruction}

        Requirements:
        - Language: {language}
        - Framework: {framework or "standard library"}
        - Style: {style}
        - Include tests: {include_tests}
        - Include documentation: {include_docs}
        - Use modern best practices and type hints
        - Include proper error handling
        - Follow security best practices
        """

        task = TaskRequest(
            task_type="code_generation",
            instruction=enhanced_instruction,
            options={
                "language": language,
                "framework": framework,
                "style": style,
                "include_tests": include_tests,
                "include_docs": include_docs,
            },
            priority=TaskPriority.NORMAL,
        )

        result = await server.process_task_with_session(task)

        return {
            "success": True,
            "task_type": "generate_code",
            "instruction": instruction,
            "language": language,
            "framework": framework,
            "generated_code": result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Code generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_type": "generate_code",
            "instruction": instruction,
        }


@mcp.tool()
async def consolidate_code(
    target_directory: str,
    file_patterns: str = "*.py,*.ts,*.js,*.rs",
    dry_run: bool = True,
    backup_files: bool = True,
    consolidation_strategy: str = "intelligent",
) -> dict[str, Any]:
    """Consolidate duplicate code and remove redundant files using Rust extensions.

    Args:
        target_directory: Directory to scan for consolidation opportunities
        file_patterns: Comma-separated file patterns to analyze
        dry_run: Whether to only analyze without making changes
        backup_files: Whether to create backups before consolidation
        consolidation_strategy: Strategy (aggressive, intelligent, conservative)

    Returns:
        Consolidation analysis and results with duplicate file identification

    """
    try:
        server = await get_server()

        # Parse file patterns
        patterns = [p.strip() for p in file_patterns.split(",") if p.strip()]

        task = TaskRequest(
            task_type="consolidation",
            instruction=f"Consolidate code in {target_directory} using {consolidation_strategy} strategy. "
            f"Patterns: {patterns}. Dry run: {dry_run}",
            target_path=target_directory,
            options={
                "file_patterns": patterns,
                "dry_run": dry_run,
                "backup_files": backup_files,
                "consolidation_strategy": consolidation_strategy,
            },
            priority=TaskPriority.HIGH,
        )

        result = await server.process_task_with_session(task)

        # Use Rust extensions for high-performance file operations
        if rust_available and rust_file_ops:
            try:
                # Find duplicate files using Rust
                duplicates = rust_file_ops.find_duplicate_files(target_directory, patterns)
                result["rust_duplicate_analysis"] = duplicates

                # Get file statistics
                stats = rust_file_ops.get_directory_stats(target_directory)
                result["directory_stats"] = stats

            except Exception as e:
                logger.warning(f"Rust file operations failed: {e}")

        return {
            "success": True,
            "task_type": "consolidate_code",
            "target_directory": target_directory,
            "file_patterns": patterns,
            "dry_run": dry_run,
            "consolidation_result": result,
            "rust_enhanced": rust_available,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Code consolidation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_type": "consolidate_code",
            "target_directory": target_directory,
        }


@mcp.tool()
async def generate_documentation(
    target_path: str,
    doc_type: str = "comprehensive",
    output_format: str = "markdown",
    include_api_docs: bool = True,
    include_examples: bool = True,
    include_architecture: bool = True,
) -> dict[str, Any]:
    """Generate comprehensive documentation for code or projects.

    Args:
        target_path: Path to file or directory to document
        doc_type: Documentation type (api, user, developer, comprehensive)
        output_format: Output format (markdown, rst, html, pdf)
        include_api_docs: Whether to generate API documentation
        include_examples: Whether to include usage examples
        include_architecture: Whether to include architecture diagrams

    Returns:
        Generated documentation with multiple formats and comprehensive coverage

    """
    try:
        server = await get_server()

        task = TaskRequest(
            task_type="documentation",
            instruction=f"Generate {doc_type} documentation for {target_path} in {output_format} format. "
            f"Include API docs: {include_api_docs}, examples: {include_examples}, "
            f"architecture: {include_architecture}",
            target_path=target_path,
            options={
                "doc_type": doc_type,
                "output_format": output_format,
                "include_api_docs": include_api_docs,
                "include_examples": include_examples,
                "include_architecture": include_architecture,
            },
            priority=TaskPriority.NORMAL,
        )

        result = await server.process_task_with_session(task)

        return {
            "success": True,
            "task_type": "generate_documentation",
            "target_path": target_path,
            "doc_type": doc_type,
            "output_format": output_format,
            "documentation": result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Documentation generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_type": "generate_documentation",
            "target_path": target_path,
        }


@mcp.tool()
async def review_code(
    code_input: str,
    review_type: str = "comprehensive",
    focus_areas: str = "security,performance,maintainability",
    severity_threshold: str = "medium",
    include_suggestions: bool = True,
) -> dict[str, Any]:
    """Perform comprehensive code review with security and performance analysis.

    Args:
        code_input: Code content or file path to review
        review_type: Review type (security, performance, style, comprehensive)
        focus_areas: Comma-separated focus areas
        severity_threshold: Minimum severity to report (low, medium, high, critical)
        include_suggestions: Whether to include improvement suggestions

    Returns:
        Detailed code review with issues, metrics, and actionable recommendations

    """
    try:
        server = await get_server()

        # Parse focus areas
        focus_list = [area.strip() for area in focus_areas.split(",") if area.strip()]

        # Determine if input is file path or code content
        is_file_path = len(code_input.splitlines()) == 1 and Path(code_input).exists()

        instruction = f"""
        Perform {review_type} code review focusing on: {focus_list}
        Severity threshold: {severity_threshold}
        Include suggestions: {include_suggestions}

        {"File path: " + code_input if is_file_path else "Code content provided"}
        """

        task = TaskRequest(
            task_type="analysis",  # Using analysis task type for code review
            instruction=instruction,
            target_path=code_input if is_file_path else None,
            options={
                "review_type": review_type,
                "focus_areas": focus_list,
                "severity_threshold": severity_threshold,
                "include_suggestions": include_suggestions,
                "is_file_path": is_file_path,
                "code_content": code_input if not is_file_path else None,
            },
            priority=TaskPriority.HIGH,
        )

        result = await server.process_task_with_session(task)

        return {
            "success": True,
            "task_type": "review_code",
            "review_type": review_type,
            "focus_areas": focus_list,
            "severity_threshold": severity_threshold,
            "code_review": result,
            "is_file_path": is_file_path,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Code review failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_type": "review_code",
            "review_type": review_type,
        }


@mcp.tool()
async def orchestrate_agents(
    task_description: str,
    agents_required: str = "analyzer,generator,reviewer",
    execution_mode: str = "sequential",
    coordination_strategy: str = "intelligent",
    timeout_minutes: int = 30,
) -> dict[str, Any]:
    """Orchestrate multiple AI agents for complex multi-step tasks.

    Args:
        task_description: Description of the complex task requiring multiple agents
        agents_required: Comma-separated list of agent types needed
        execution_mode: Execution mode (sequential, parallel, adaptive)
        coordination_strategy: How agents coordinate (simple, intelligent, autonomous)
        timeout_minutes: Maximum time for orchestration to complete

    Returns:
        Multi-agent orchestration results with individual agent outputs and coordination

    """
    try:
        server = await get_server()

        # Parse required agents
        agent_list = [agent.strip() for agent in agents_required.split(",") if agent.strip()]

        # Create orchestration task
        task = TaskRequest(
            task_type="orchestration",  # New task type for multi-agent coordination
            instruction=f"""
            Multi-agent orchestration task: {task_description}

            Required agents: {agent_list}
            Execution mode: {execution_mode}
            Coordination strategy: {coordination_strategy}

            Please coordinate the following agent capabilities:
            - analyzer: Project and code analysis
            - generator: Code and documentation generation
            - reviewer: Code review and quality assessment
            - consolidator: Code consolidation and cleanup
            - architect: System design and recommendations
            """,
            options={
                "agents_required": agent_list,
                "execution_mode": execution_mode,
                "coordination_strategy": coordination_strategy,
                "timeout_minutes": timeout_minutes,
            },
            priority=TaskPriority.HIGH,
            timeout_seconds=timeout_minutes * 60,
        )

        # Process orchestration task
        result = await server.process_task_with_session(task)

        # Track orchestration metrics
        return {
            "success": True,
            "task_type": "orchestrate_agents",
            "task_description": task_description,
            "agents_required": agent_list,
            "execution_mode": execution_mode,
            "coordination_strategy": coordination_strategy,
            "orchestration_result": result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Agent orchestration failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_type": "orchestrate_agents",
            "task_description": task_description,
        }


@mcp.tool()
async def get_system_status(
    include_performance: bool = True,
    include_sessions: bool = True,
    include_rust_metrics: bool = True,
) -> dict[str, Any]:
    """Get comprehensive system status including server health and performance metrics.

    Args:
        include_performance: Whether to include performance metrics
        include_sessions: Whether to include session information
        include_rust_metrics: Whether to include Rust extension metrics

    Returns:
        Complete system status with health, performance, and operational metrics

    """
    try:
        server = await get_server()

        # Basic server status
        status = {
            "server_name": "GeminiUnifiedServer",
            "status": "operational",
            "model": "gemini-2.0-flash-exp",
            "authentication": "service_account",
            "timestamp": datetime.now().isoformat(),
        }

        if include_sessions:
            status["sessions"] = {
                "total_sessions": len(server.sessions),
                "active_keep_alive": len(server.keep_alive_tasks),
                "session_details": [
                    {
                        "session_id": sid,
                        "created_at": session.created_at.isoformat(),
                        "last_activity": session.last_activity.isoformat(),
                        "conversation_length": len(session.conversation_history),
                        "active_tasks": len(session.active_tasks),
                    }
                    for sid, session in list(server.sessions.items())[
                        :10
                    ]  # Limit to 10 most recent
                ],
            }

        if include_performance:
            status["performance"] = {
                "active_tasks": len(server.active_tasks),
                "queued_tasks": server.task_queue.qsize(),
                "executor_threads": server.executor._max_workers,
                "uptime_seconds": "running",  # Simplified uptime
            }

        if include_rust_metrics and rust_available:
            try:
                rust_status = get_system_info()
                rust_perf = get_performance_metrics()

                status["rust_extensions"] = {
                    "available": True,
                    "system_info": rust_status,
                    "performance_metrics": rust_perf,
                    "cache_stats": rust_cache.get_stats() if rust_cache else None,
                }
            except Exception as e:
                status["rust_extensions"] = {"available": True, "error": str(e)}
        else:
            status["rust_extensions"] = {"available": False}

        return {
            "success": True,
            "system_status": status,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"System status check failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@mcp.tool()
async def manage_sessions(
    action: str, session_id: str = "", session_data: str = "{}"
) -> dict[str, Any]:
    """Manage server sessions for persistent conversations and state.

    Args:
        action: Action to perform (create, get, list, delete, cleanup)
        session_id: Session ID for get/delete actions
        session_data: JSON string of session data for create/update

    Returns:
        Session management results with session information and status

    """
    try:
        server = await get_server()

        if action == "create":
            new_session_id = server.create_session(session_id or None)
            return {
                "success": True,
                "action": "create",
                "session_id": new_session_id,
                "message": "Session created successfully",
            }

        if action == "get":
            if not session_id:
                return {"success": False, "error": "Session ID required for get action"}

            if session_id in server.sessions:
                session = server.sessions[session_id]
                return {
                    "success": True,
                    "action": "get",
                    "session_id": session_id,
                    "session_data": {
                        "created_at": session.created_at.isoformat(),
                        "last_activity": session.last_activity.isoformat(),
                        "conversation_length": len(session.conversation_history),
                        "active_tasks": len(session.active_tasks),
                        "context_cache_size": len(session.context_cache),
                    },
                }
            return {"success": False, "error": "Session not found"}

        if action == "list":
            sessions: list[Any] = []
            for sid, session in server.sessions.items():
                sessions.append(
                    {
                        "session_id": sid,
                        "created_at": session.created_at.isoformat(),
                        "last_activity": session.last_activity.isoformat(),
                        "conversation_length": len(session.conversation_history),
                        "active_tasks": len(session.active_tasks),
                    },
                )

            return {
                "success": True,
                "action": "list",
                "total_sessions": len(sessions),
                "sessions": sessions,
            }

        if action == "delete":
            if not session_id:
                return {
                    "success": False,
                    "error": "Session ID required for delete action",
                }

            if session_id in server.sessions:
                # Cancel keep-alive task if exists
                for task in server.keep_alive_tasks:
                    if task.get_name() == session_id:
                        task.cancel()
                        server.keep_alive_tasks.remove(task)
                        break

                # Delete session
                del server.sessions[session_id]
                server.save_sessions()

                return {
                    "success": True,
                    "action": "delete",
                    "session_id": session_id,
                    "message": "Session deleted successfully",
                }
            return {"success": False, "error": "Session not found"}

        if action == "cleanup":
            server.cleanup_old_sessions()
            return {
                "success": True,
                "action": "cleanup",
                "message": "Old sessions cleaned up successfully",
            }

        return {"success": False, "error": f"Unknown action: {action}"}

    except Exception as e:
        logger.exception(f"Session management failed: {e}")
        return {"success": False, "error": str(e), "action": action}


# Server initialization and health check
@mcp.tool()
async def health_check() -> dict[str, Any]:
    """Perform comprehensive health check of all system components.

    Returns:
        Complete health status of server, Rust extensions, and dependencies

    """
    try:
        server = await get_server()

        health_status = {
            "overall_status": "healthy",
            "server": {
                "status": "operational",
                "model_available": server.model is not None,
                "sessions": len(server.sessions),
                "active_tasks": len(server.active_tasks),
            },
            "rust_extensions": {
                "available": rust_available,
                "status": "operational" if rust_available else "unavailable",
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Test Gemini model
        try:
            await server.model.generate_content_async("Health check test")
            health_status["server"]["model_test"] = "passed"
        except Exception as e:
            health_status["server"]["model_test"] = f"failed: {e}"
            health_status["overall_status"] = "degraded"

        # Test Rust extensions
        if rust_available:
            try:
                system_info = get_system_info()
                health_status["rust_extensions"]["test"] = "passed"
                health_status["rust_extensions"]["details"] = system_info
            except Exception as e:
                health_status["rust_extensions"]["test"] = f"failed: {e}"
                health_status["overall_status"] = "degraded"

        return {"success": True, "health_check": health_status}

    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        return {"success": False, "error": str(e), "overall_status": "unhealthy"}


if __name__ == "__main__":
    logger.info("🚀 Starting Gemini Server Agent MCP Server")
    logger.info(f"🔧 Rust extensions available: {rust_available}")
    logger.info("🔗 MCP Inspector compliant with FastMCP framework")
    mcp.run()
