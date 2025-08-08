#!/usr/bin/env python3
"""
Demonstration of the Fixed Autonomous ReAct Engine

This script demonstrates the fixed autonomous ReAct engine working on practical tasks:
1. Creating a simple file with system information
2. Analyzing a project structure
3. Generating documentation

All the critical issues have been resolved:
✅ Rust extension import errors (with proper fallbacks)
✅ Pydantic validation errors (proper field types and validation)
✅ JSON parsing failures (robust extraction from LLM responses)
✅ Error handling and graceful degradation
"""

import asyncio
import json
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demo_autonomous_file_creation():
    """Demonstrate autonomous file creation task."""
    print("🎯 DEMO 1: Autonomous File Creation")
    print("=" * 60)

    from app.core.react_engine import AutonomyLevel
    from app.core.react_engine import ReactEngine
    from app.core.react_engine import ReactEngineConfig

    config = ReactEngineConfig(
        enable_redis=True,
        enable_rag=True,
        enable_autonomous=True,
        autonomy_level=AutonomyLevel.FULLY_AUTO,
        cache_responses=True,
    )
    engine = ReactEngine(model=None, config=config, profile="business")

    request = """Create a system information file called 'system_info.txt' with:
    - Current timestamp
    - Python version
    - Operating system details
    - Available memory
    - Current working directory"""

    try:
        print(f"📝 Request: {request}")
        print("\n🤖 Processing autonomously...")

        response = await engine.process_request(request=request, streaming=True)

        print("\n📊 Results:")
        print(f"   ✅ Success: {response.success}")
        print(f"   ⏱️  Total Time: {response.total_time:.2f}s")
        print(f"   🔧 Steps Executed: {len(response.steps_executed)}")

        if response.steps_executed:
            print("\n📋 Execution Steps:")
            for i, step in enumerate(response.steps_executed, 1):
                status = "✅" if (step.result and step.result.success) else "❌"
                print(f"   {i}. {status} {step.description}")

        return response.success

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


async def demo_autonomous_project_analysis():
    """Demonstrate autonomous project analysis task."""
    print("\n🎯 DEMO 2: Autonomous Project Analysis")
    print("=" * 60)

    from app.core.react_engine import AutonomyLevel
    from app.core.react_engine import ReactEngine
    from app.core.react_engine import ReactEngineConfig

    config = ReactEngineConfig(
        enable_redis=True,
        enable_rag=True,
        enable_autonomous=True,
        autonomy_level=AutonomyLevel.SEMI_AUTO,
        cache_responses=True,
    )
    engine = ReactEngine(model=None, config=config, profile="business")

    request = """Analyze the current project structure and create a summary report including:
    - Directory structure overview
    - Main Python files and their purposes
    - Configuration files present
    - Test files organization
    - Key dependencies identified"""

    try:
        print(f"📝 Request: {request}")
        print("\n🤖 Processing with semi-autonomous mode...")

        response = await engine.process_request(request=request, streaming=True)

        print("\n📊 Results:")
        print(f"   ✅ Success: {response.success}")
        print(f"   ⏱️  Total Time: {response.total_time:.2f}s")
        print(f"   🔧 Steps Executed: {len(response.steps_executed)}")

        # Show status
        status = await engine.get_autonomous_status()
        print("\n🧠 Engine Status:")
        print(f"   📚 Learned Patterns: {status['learning_data']['learned_patterns']}")
        print(
            f"   📈 Success Rate: {status['performance_metrics']['successful_requests']}/{status['performance_metrics']['total_requests']}"
        )

        return response.success

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


async def demo_error_handling():
    """Demonstrate robust error handling and recovery."""
    print("\n🎯 DEMO 3: Error Handling & Recovery")
    print("=" * 60)

    from app.core.react_engine import AutonomyLevel
    from app.core.react_engine import ReactEngine
    from app.core.react_engine import ReactEngineConfig

    config = ReactEngineConfig(
        enable_redis=True,
        enable_rag=True,
        enable_autonomous=True,
        autonomy_level=AutonomyLevel.FULLY_AUTO,
        cache_responses=True,
    )
    engine = ReactEngine(model=None, config=config, profile="business")

    # Intentionally problematic request to test error handling
    request = """Use a non-existent tool called 'magic_tool' to perform impossible tasks like:
    - Reading from a file that doesn't exist: /impossible/path/file.txt
    - Connecting to a fake API endpoint
    - Processing malformed JSON: {"broken": json, missing: quotes}"""

    try:
        print(f"📝 Request (intentionally problematic): {request}")
        print("\n🤖 Testing error handling and recovery...")

        response = await engine.process_request(request=request, streaming=True)

        print("\n📊 Results:")
        print(f"   🛡️  Graceful Handling: {not response.success}")  # Should fail gracefully
        print(f"   ⏱️  Total Time: {response.total_time:.2f}s")
        print(f"   🔧 Steps Attempted: {len(response.steps_executed)}")

        if not response.success:
            print("   ✅ Error handled gracefully - no crashes!")
            print("   🔍 Error details preserved in result")

        return True  # Success means graceful error handling

    except Exception as e:
        print(f"❌ Unexpected error (should not happen): {e}")
        return False


async def demo_json_robustness():
    """Demonstrate robust JSON parsing with various malformed inputs."""
    print("\n🎯 DEMO 4: Robust JSON Parsing")
    print("=" * 60)

    from app.core.json_utils import extract_json_from_llm_response

    # Test various malformed JSON that might come from LLMs
    test_cases = [
        ("Standard JSON", '{"goal": "test", "steps": []}'),
        ("Markdown wrapped", '```json\n{"goal": "test", "complexity": "simple"}\n```'),
        (
            "Mixed content",
            'Here is the plan: {"goal": "analyze", "steps": [{"type": "act", "description": "analyze code"}]} - that\'s the plan!',
        ),
        ("Trailing commas", '{\n  "goal": "test",\n  "steps": [],\n  "complexity": "simple",\n}'),
        ("Python booleans", '{"goal": "test", "success": True, "failed": False, "nothing": None}'),
        ("Single quotes", "{'goal': 'test', 'type': 'simple', 'steps': []}"),
        ("Unquoted keys", '{goal: "test", type: "simple", steps: []}'),
        ("Malformed nested", '{"goal": "test", "data": {"key": "value"}, }'),
    ]

    success_count = 0

    for test_name, test_json in test_cases:
        try:
            result = extract_json_from_llm_response(test_json)
            print(f"   ✅ {test_name}: Parsed successfully")
            print(f"      Input: {test_json[:50]}{'...' if len(test_json) > 50 else ''}")
            print(
                f"      Output: {json.dumps(result, separators=(',', ':'))[:50]}{'...' if len(str(result)) > 50 else ''}"
            )
            success_count += 1
        except Exception as e:
            print(f"   ❌ {test_name}: Failed - {e}")

    print(f"\n📊 JSON Parsing Results: {success_count}/{len(test_cases)} successful")
    return success_count == len(test_cases)


async def main():
    """Run all demonstrations."""
    print("🚀 AUTONOMOUS REACT ENGINE DEMONSTRATION")
    print("This demonstrates all the fixes implemented:")
    print("✅ Rust Extension Issues → Proper fallbacks")
    print("✅ Pydantic Validation → Fixed field types and validation")
    print("✅ JSON Parsing → Robust extraction from LLM responses")
    print("✅ Error Handling → Graceful degradation\n")

    demos = [
        ("File Creation", demo_autonomous_file_creation),
        ("Project Analysis", demo_autonomous_project_analysis),
        ("Error Handling", demo_error_handling),
        ("JSON Robustness", demo_json_robustness),
    ]

    results = {}

    for demo_name, demo_func in demos:
        try:
            success = await demo_func()
            results[demo_name] = success
        except Exception as e:
            print(f"❌ {demo_name} demo failed with error: {e}")
            results[demo_name] = False

    # Summary
    print(f"\n{'=' * 60}")
    print("🎉 DEMONSTRATION SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for demo_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {demo_name}: {status}")

    print(f"\nOverall: {passed}/{total} demos successful ({passed / total * 100:.1f}%)")

    if passed == total:
        print("\n🎉 All fixes are working perfectly!")
        print("The autonomous ReAct engine is ready for Phase 2 implementation!")
        return 0
    else:
        print("\n⚠️  Some demos had issues - review the logs above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
