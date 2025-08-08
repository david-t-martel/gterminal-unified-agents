#!/usr/bin/env python3
"""Test function calling in gterminal Gemini client."""

import asyncio
import logging
from pathlib import Path
import sys

# Add gterminal to path
sys.path.insert(0, str(Path(__file__).parent))

from gemini_cli.core.client import GeminiClient
from gemini_cli.core.react_engine import SimpleReactEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_direct_function_calling():
    """Test direct function calling with Gemini client."""
    logger.info("🧪 Testing direct function calling in GeminiClient")

    client = GeminiClient("gemini-2.0-flash-exp")

    # Create a mock tool for testing
    class MockTool:
        @property
        def name(self):
            return "test_tool"

        @property
        def description(self):
            return "A test tool for validating function calling"

        async def execute(self, params):
            return {"result": f"Executed with params: {params}"}

    tools = [MockTool()]

    prompt = """
    Please use the test_tool to demonstrate function calling capability.
    Call it with some test parameters to show that function calling is working.
    """

    try:
        response = await client.process_with_tools(prompt, tools)
        logger.info(f"Response: {response}")

        if "Function calls made:" in response:
            logger.info("✅ Function calling is ENABLED and working!")
            return True
        else:
            logger.warning("⚠️ Function calling may not be triggered")
            return False
    except Exception as e:
        logger.exception(f"❌ Test failed: {e}")
        return False


async def test_react_engine_with_function_calling():
    """Test ReAct engine with function calling enabled."""
    logger.info("\n🧪 Testing ReAct engine with function calling")

    engine = SimpleReactEngine("gemini-2.0-flash-exp")

    # Test with a simple file operation request
    request = "List all Python files in the current directory"

    try:
        response = await engine.process(request)
        logger.info(f"ReAct Response: {response}")

        # Check execution summary
        summary = engine.get_execution_summary()
        logger.info(f"Execution Summary: {summary}")

        return True
    except Exception as e:
        logger.exception(f"❌ ReAct test failed: {e}")
        return False


async def main():
    """Run all function calling tests."""
    logger.info("🚀 Starting function calling tests for gterminal")
    logger.info("=" * 60)

    # Test 1: Direct function calling
    test1_result = await test_direct_function_calling()

    # Test 2: ReAct engine with function calling
    test2_result = await test_react_engine_with_function_calling()

    # Summary
    logger.info("\n📊 Test Summary")
    logger.info("=" * 60)
    logger.info(f"Direct function calling: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    logger.info(f"ReAct engine: {'✅ PASSED' if test2_result else '❌ FAILED'}")

    if test1_result and test2_result:
        logger.info("\n🎉 All tests passed! Function calling is enabled.")
        return True
    else:
        logger.error("\n❌ Some tests failed. Function calling needs more work.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
