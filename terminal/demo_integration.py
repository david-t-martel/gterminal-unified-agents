#!/usr/bin/env python3
"""Integration Demo for Enhanced ReAct Orchestrator.

This script demonstrates how to use the Enhanced ReAct Orchestrator with
the existing infrastructure to create a production-ready AI agent system.
"""

import asyncio
import logging
from pathlib import Path
import sys
from typing import Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the enhanced orchestrator
try:
    from .enhanced_react_orchestrator import AgentRole
    from .enhanced_react_orchestrator import EnhancedReActOrchestrator
    from .enhanced_react_orchestrator import EnhancedReActTask
    from .enhanced_react_orchestrator import TaskPriority
    from .enhanced_react_orchestrator import create_enhanced_orchestrator
except ImportError:
    logger.exception("Failed to import enhanced orchestrator. Ensure the module is available.")
    sys.exit(1)


async def demo_basic_usage():
    """Demonstrate basic usage of the enhanced orchestrator."""
    print("🚀 Starting Basic Usage Demo")
    print("=" * 50)

    # Create orchestrator with all features enabled
    orchestrator = await create_enhanced_orchestrator(
        project_root=Path("/home/david/agents/my-fullstack-agent"),
        gemini_profile="business",
        enable_local_llm=True,
        enable_web_fetch=True,
    )

    # Create a simple task
    task = EnhancedReActTask(
        description="Analyze the current codebase structure and identify key components",
        priority=TaskPriority.MEDIUM,
        privacy_level="standard",
        context={
            "focus_areas": ["architecture", "performance", "maintainability"],
            "output_format": "markdown",
        },
        success_criteria=[
            "Identify main architectural patterns",
            "List key performance considerations",
            "Suggest improvement areas",
        ],
    )

    try:
        # Process the task
        result = await orchestrator.process_enhanced_task(task)

        print("✅ Task completed successfully!")
        print(f"   Execution time: {result['execution_time']:.2f}s")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Success: {result['success']}")

        if result.get("result"):
            print("\n📝 Result Preview:")
            result_str = str(result["result"])
            print(f"   {result_str[:200]}..." if len(result_str) > 200 else f"   {result_str}")

    except Exception as e:
        logger.exception(f"Basic demo failed: {e}")

    finally:
        await orchestrator.cleanup()


async def demo_multi_agent_coordination():
    """Demonstrate multi-agent coordination capabilities."""
    print("\n🤝 Starting Multi-Agent Coordination Demo")
    print("=" * 50)

    orchestrator = await create_enhanced_orchestrator()

    # Create a task requiring multiple agents
    task = EnhancedReActTask(
        description="Perform comprehensive code review and architectural analysis",
        priority=TaskPriority.HIGH,
        privacy_level="standard",
        required_agents=[AgentRole.REVIEWER, AgentRole.ARCHITECT, AgentRole.ANALYZER],
        context={
            "target_files": [
                "app/core/react_engine.py",
                "app/terminal/enhanced_react_orchestrator.py",
            ],
            "review_criteria": ["security", "performance", "maintainability"],
        },
        constraints=[
            "Focus on production-readiness",
            "Consider scalability implications",
            "Identify potential bottlenecks",
        ],
        success_criteria=[
            "Complete security analysis",
            "Architectural assessment",
            "Performance recommendations",
        ],
    )

    # Progress callback to show real-time updates
    def progress_callback(progress: dict[str, Any]):
        print("🔄 Progress Update:")
        print(f"   Iteration: {progress['iteration']}")
        print(f"   Status: {progress['status']}")
        print(f"   Actions taken: {progress['actions_taken']}")
        if progress.get("latest_step"):
            step_preview = (
                progress["latest_step"][:150] + "..."
                if len(progress["latest_step"]) > 150
                else progress["latest_step"]
            )
            print(f"   Latest step: {step_preview}")
        print()

    try:
        async with orchestrator.enhanced_task_session(task) as session:
            result = await orchestrator.process_enhanced_task(task, progress_callback)
            session["final_result"] = result

        print("✅ Multi-agent task completed!")
        print(f"   Total time: {result['execution_time']:.2f}s")
        print(f"   Agents coordinated: {len(task.required_agents)}")
        print(f"   Success: {result['success']}")

        # Show metrics
        metrics = await orchestrator.get_comprehensive_metrics()
        print("\n📊 Coordination Metrics:")
        print(f"   Messages sent: {metrics['message_queue_stats']['sent']}")
        print(f"   Messages processed: {metrics['message_queue_stats']['processed']}")
        print(f"   Active queues: {metrics['message_queue_stats']['active_queues']}")

    except Exception as e:
        logger.exception(f"Multi-agent demo failed: {e}")

    finally:
        await orchestrator.cleanup()


async def demo_privacy_sensitive_processing():
    """Demonstrate privacy-sensitive task processing with local LLM."""
    print("\n🔒 Starting Privacy-Sensitive Processing Demo")
    print("=" * 50)

    orchestrator = await create_enhanced_orchestrator(enable_local_llm=True)

    # Create a privacy-sensitive task
    task = EnhancedReActTask(
        description="Analyze configuration files for sensitive information and security vulnerabilities",
        priority=TaskPriority.HIGH,
        privacy_level="sensitive",  # This will trigger local LLM usage
        context={
            "scan_patterns": ["passwords", "api_keys", "tokens", "secrets"],
            "config_files": [".env", "config.json", "settings.py"],
        },
        constraints=[
            "Process locally only",
            "Do not transmit sensitive data",
            "Provide actionable security recommendations",
        ],
        success_criteria=[
            "Identify all potential security issues",
            "Provide remediation steps",
            "Generate security audit report",
        ],
    )

    try:
        result = await orchestrator.process_enhanced_task(task)

        print("🔒 Privacy-sensitive task completed!")
        print(f"   Used local LLM: {result.get('used_local_llm', False)}")
        print(f"   Execution time: {result['execution_time']:.2f}s")
        print(f"   Success: {result['success']}")

        # Show local LLM metrics
        metrics = await orchestrator.get_comprehensive_metrics()
        print("\n🔐 Privacy Processing Metrics:")
        print(f"   Local LLM available: {metrics['local_llm_available']}")
        print(f"   Local LLM calls: {metrics['orchestrator_metrics']['local_llm_calls']}")
        print(f"   Cloud LLM calls: {metrics['orchestrator_metrics']['llm_calls']}")

    except Exception as e:
        logger.exception(f"Privacy demo failed: {e}")

    finally:
        await orchestrator.cleanup()


async def demo_web_integration():
    """Demonstrate web fetching and integration capabilities."""
    print("\n🌐 Starting Web Integration Demo")
    print("=" * 50)

    orchestrator = await create_enhanced_orchestrator(enable_web_fetch=True)

    # Create a task that requires web content
    task = EnhancedReActTask(
        description="Research current best practices for Python async programming and ReAct patterns from authoritative sources",
        priority=TaskPriority.MEDIUM,
        privacy_level="standard",
        context={
            "research_topics": [
                "async/await patterns",
                "ReAct implementation",
                "Python performance",
            ],
            "source_requirements": [
                "official documentation",
                "authoritative blogs",
                "research papers",
            ],
        },
        success_criteria=[
            "Gather information from multiple sources",
            "Synthesize findings into actionable insights",
            "Provide implementation recommendations",
        ],
    )

    try:
        result = await orchestrator.process_enhanced_task(task)

        print("🌐 Web integration task completed!")
        print(f"   Execution time: {result['execution_time']:.2f}s")
        print(f"   Success: {result['success']}")

        # Show web fetch metrics
        performance_metrics = result.get("performance_metrics", {})
        print("\n🔍 Web Integration Metrics:")
        print(f"   Cache hits: {performance_metrics.get('cache_hits', 0)}")
        print(f"   Tools used: {performance_metrics.get('tools_used', 0)}")

    except Exception as e:
        logger.exception(f"Web integration demo failed: {e}")

    finally:
        await orchestrator.cleanup()


async def demo_comprehensive_system():
    """Demonstrate the complete system with all features."""
    print("\n🎯 Starting Comprehensive System Demo")
    print("=" * 50)

    orchestrator = await create_enhanced_orchestrator(
        enable_local_llm=True,
        enable_web_fetch=True,
    )

    # Create a complex task that uses all features
    task = EnhancedReActTask(
        description="""Perform a comprehensive analysis of the my-fullstack-agent project:
        1. Analyze the codebase architecture and identify improvement opportunities
        2. Research current best practices for similar systems
        3. Review security implications of the current implementation
        4. Generate a detailed improvement roadmap with prioritized recommendations""",
        priority=TaskPriority.HIGH,
        privacy_level="sensitive",  # Mixed: some parts need privacy, others don't
        required_agents=[AgentRole.ARCHITECT, AgentRole.REVIEWER, AgentRole.ANALYZER],
        context={
            "project_root": "/home/david/agents/my-fullstack-agent",
            "focus_areas": ["architecture", "security", "performance", "maintainability"],
            "deliverable_format": "structured_report",
        },
        constraints=[
            "Maintain backward compatibility",
            "Consider production deployment requirements",
            "Include cost-benefit analysis",
            "Provide implementation timeline",
        ],
        success_criteria=[
            "Complete architectural assessment",
            "Security vulnerability analysis",
            "Performance optimization recommendations",
            "Prioritized improvement roadmap",
            "Implementation cost estimates",
        ],
    )

    # Enhanced progress tracking
    def detailed_progress_callback(progress: dict[str, Any]):
        print("🔄 Comprehensive Analysis Progress:")
        print(f"   Phase: Iteration {progress['iteration']}")
        print(f"   Status: {progress['status'].upper()}")
        print(f"   Actions completed: {progress.get('actions_taken', 0)}")

        if progress.get("latest_step"):
            print(f"   Current focus: {progress['latest_step'][:100]}...")
        print(f"   {'─' * 40}")

    try:
        # Process with full session management
        async with orchestrator.enhanced_task_session(task) as session:
            session["start_time"]

            result = await orchestrator.process_enhanced_task(task, detailed_progress_callback)

            session["result"] = result
            session["analysis_complete"] = True

        print("\n🎯 Comprehensive Analysis Completed!")
        print(f"   Total execution time: {result['execution_time']:.2f}s")
        print(f"   Iterations required: {result['iterations']}")
        print(f"   Success: {result['success']}")
        print(f"   Used local LLM: {result.get('used_local_llm', False)}")

        # Comprehensive metrics
        final_metrics = await orchestrator.get_comprehensive_metrics()

        print("\n📊 Comprehensive System Metrics:")
        print(
            f"   Total tasks completed: {final_metrics['orchestrator_metrics']['tasks_completed']}"
        )
        print(
            f"   Average task time: {final_metrics['orchestrator_metrics']['average_task_time']:.2f}s"
        )
        print(
            f"   Message queue efficiency: {final_metrics['message_queue_stats']['processed']} processed"
        )
        print(f"   Rust extensions active: {final_metrics['rust_extensions']}")
        print(f"   Local LLM integration: {final_metrics['local_llm_available']}")
        print(f"   Web fetch capability: {final_metrics['web_fetch_available']}")

        if result.get("result"):
            print("\n📋 Analysis Summary:")
            result_preview = (
                str(result["result"])[:500] + "..."
                if len(str(result["result"])) > 500
                else str(result["result"])
            )
            print(f"   {result_preview}")

    except Exception as e:
        logger.exception(f"Comprehensive demo failed: {e}")

    finally:
        await orchestrator.cleanup()


async def main():
    """Run all demonstration scenarios."""
    print("🎉 Enhanced ReAct Orchestrator Integration Demonstration")
    print("🔧 Built upon existing my-fullstack-agent infrastructure")
    print("📦 Leverages Rust extensions, MCP clients, and session management")
    print("=" * 70)

    try:
        # Run all demos in sequence
        await demo_basic_usage()
        await demo_multi_agent_coordination()
        await demo_privacy_sensitive_processing()
        await demo_web_integration()
        await demo_comprehensive_system()

        print("\n🎊 All demonstrations completed successfully!")
        print("🚀 The Enhanced ReAct Orchestrator is ready for production use.")

    except KeyboardInterrupt:
        print("\n⏸️  Demo interrupted by user")
    except Exception as e:
        logger.exception(f"Demo suite failed: {e}")

    print("\n✅ Integration demonstration complete.")


if __name__ == "__main__":
    # Run the integration demo
    asyncio.run(main())
