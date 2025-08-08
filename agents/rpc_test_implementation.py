"""Test implementation for JSON RPC 2.0 compliant agents.

This script demonstrates how to use the RPC-enabled agents with proper
request/response patterns and parameter validation.
"""

import asyncio
from datetime import UTC
from datetime import datetime
import logging

# Import the RPC-enabled agents
from gterminal.agents.code_generation_agent import CodeGenerationService
from gterminal.agents.master_architect_agent import MasterArchitectService
from gterminal.agents.rpc_parameter_models import AnalyzeProjectParams
from gterminal.agents.rpc_parameter_models import DesignSystemParams
from gterminal.agents.rpc_parameter_models import GenerateCodeParams
from gterminal.agents.workspace_analyzer_agent import WorkspaceAnalyzerService

# Import RPC infrastructure
from gterminal.core.rpc.models import RpcRequest
from gterminal.core.rpc.models import SessionContext
from gterminal.core.rpc.patterns import method_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_code_generation_rpc():
    """Test Code Generation Agent with RPC compliance."""
    print("=== Testing Code Generation Agent (JSON RPC 2.0) ===")

    agent = CodeGenerationService()
    await agent.startup()

    try:
        # Test generate_code RPC method
        params = GenerateCodeParams(
            specification={
                "type": "api_service",
                "name": "UserService",
                "endpoints": [
                    {
                        "method": "GET",
                        "path": "/users",
                        "description": "List all users",
                    },
                    {
                        "method": "POST",
                        "path": "/users",
                        "description": "Create new user",
                    },
                ],
                "models": [
                    {
                        "name": "User",
                        "fields": {"id": "int", "name": "str", "email": "str"},
                    }
                ],
            },
            language="python",
            include_tests=True,
        )

        # Create RPC request
        request = RpcRequest(method="generate_code", params=params, id="test_gen_code_1")

        # Execute RPC method
        response = await agent.handle_rpc_request(request)

        print(f"✅ Code Generation RPC Response: {response.result is not None}")
        print(f"   Agent: {response.agent_name}")
        print(f"   Execution time: {response.execution_time_ms}ms")

        # Test API generation
        api_params = GenerateApiParams(
            api_specification={
                "name": "Product API",
                "version": "1.0.0",
                "endpoints": [
                    {
                        "method": "GET",
                        "path": "/products",
                        "description": "List products",
                    },
                    {
                        "method": "POST",
                        "path": "/products",
                        "description": "Create product",
                    },
                ],
            },
            framework="fastapi",
            include_openapi=True,
        )

        api_request = RpcRequest(method="generate_api", params=api_params, id="test_gen_api_1")

        api_response = await agent.handle_rpc_request(api_request)
        print(f"✅ API Generation RPC Response: {api_response.result is not None}")

    except Exception as e:
        print(f"❌ Code Generation RPC Error: {e}")
    finally:
        await agent.shutdown()


async def test_workspace_analyzer_rpc():
    """Test Workspace Analyzer Agent with RPC compliance."""
    print("\n=== Testing Workspace Analyzer Agent (JSON RPC 2.0) ===")

    agent = WorkspaceAnalyzerService()
    await agent.startup()

    try:
        # Test analyze_project RPC method
        params = AnalyzeProjectParams(
            project_path="/home/david/agents/my-fullstack-agent",
            depth=2,
            include_patterns=["*.py", "*.json"],
            analyze_dependencies=True,
            analyze_security=False,  # Skip security to speed up test
        )

        request = RpcRequest(method="analyze_project", params=params, id="test_analyze_1")

        response = await agent.handle_rpc_request(request)

        print(f"✅ Project Analysis RPC Response: {response.result is not None}")
        print(f"   Agent: {response.agent_name}")
        print(f"   Execution time: {response.execution_time_ms}ms")

    except Exception as e:
        print(f"❌ Workspace Analysis RPC Error: {e}")
    finally:
        await agent.shutdown()


async def test_master_architect_rpc():
    """Test Master Architect Agent with RPC compliance."""
    print("\n=== Testing Master Architect Agent (JSON RPC 2.0) ===")

    agent = MasterArchitectService()
    await agent.startup()

    try:
        # Test design_system RPC method
        params = DesignSystemParams(
            requirements={
                "type": "web_application",
                "scale": "medium",
                "users": "1000-10000",
                "features": ["user_authentication", "data_storage", "api_integration"],
                "constraints": ["budget_conscious", "fast_development"],
            },
            architecture_style="microservices",
        )

        request = RpcRequest(method="design_system", params=params, id="test_design_1")

        response = await agent.handle_rpc_request(request)

        print(f"✅ System Design RPC Response: {response.result is not None}")
        print(f"   Agent: {response.agent_name}")
        print(f"   Execution time: {response.execution_time_ms}ms")

    except Exception as e:
        print(f"❌ System Design RPC Error: {e}")
    finally:
        await agent.shutdown()


async def test_rpc_method_registry():
    """Test the RPC method registry functionality."""
    print("\n=== Testing RPC Method Registry ===")

    # List all registered RPC methods
    methods = method_registry.list_methods()
    print(f"📋 Registered RPC Methods ({len(methods)}):")

    for method_name in methods:
        method_info = method_registry.get_method(method_name)
        if method_info:
            print(f"   • {method_name} ({method_info['agent_name']})")
            print(
                f"     Config: timeout={method_info['config'].timeout_seconds}s, "
                f"validation={method_info['config'].validate_params}"
            )


async def test_session_management():
    """Test session context management."""
    print("\n=== Testing Session Management ===")

    agent = CodeGenerationService()
    await agent.startup()

    try:
        # Create a session
        session = SessionContext(
            session_id="test_session_123",
            agent_name=agent.agent_name,
            created_at=datetime.now(UTC),
            last_activity=datetime.now(UTC),
        )

        # Test with session context
        params = GenerateCodeParams(
            specification={
                "type": "simple_function",
                "name": "calculate_tax",
                "description": "Calculate tax amount",
            }
        )

        request = RpcRequest(
            method="generate_code",
            params=params,
            id="test_session_1",
            session_id=session.session_id,
        )

        response = await agent.handle_rpc_request(request)

        print(f"✅ Session-based RPC Response: {response.result is not None}")
        print(f"   Session ID: {response.session_id}")

    except Exception as e:
        print(f"❌ Session Management Error: {e}")
    finally:
        await agent.shutdown()


async def run_rpc_compliance_tests():
    """Run comprehensive RPC compliance tests."""
    print("🚀 JSON RPC 2.0 Agent Compliance Testing")
    print("=" * 50)

    start_time = datetime.now(UTC)

    # Run all tests
    await test_code_generation_rpc()
    await test_workspace_analyzer_rpc()
    await test_master_architect_rpc()
    await test_rpc_method_registry()
    await test_session_management()

    end_time = datetime.now(UTC)
    duration = (end_time - start_time).total_seconds()

    print(f"\n🏁 Testing completed in {duration:.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_rpc_compliance_tests())
