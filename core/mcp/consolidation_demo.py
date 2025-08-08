"""
MCP Consolidation Framework Demo

Demonstrates the integrated functionality of the MCP consolidation system
including authentication, configuration management, security, and server lifecycle.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from gterminal.core.mcp.config_manager import MCPConfigManager
from gterminal.core.mcp.config_manager import ServerType
from gterminal.core.mcp.consolidated_auth import ConsolidatedAuth
from gterminal.core.mcp.security_manager import SecurityLevel
from gterminal.core.mcp.security_manager import SecurityManager
from gterminal.core.mcp.server_registry import ServerRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_authentication():
    """Demonstrate authentication functionality"""
    print("\n🔐 Authentication Demo")
    print("=" * 50)

    # Initialize authentication
    auth = ConsolidatedAuth()

    print(f"Active Profile: {auth.profile.value}")
    print(f"Account: {auth.config.account}")
    print(f"Project: {auth.config.project}")

    # Get environment variables for MCP servers
    env_vars = auth.get_environment_vars()
    print(f"Environment Variables: {list(env_vars.keys())}")

    # Test authentication
    try:
        await auth.authenticate()
        print("✅ Authentication successful")

        # Show auth info
        auth_info = auth.get_auth_info()
        print(f"Auth Info: {auth_info}")

    except Exception as e:
        print(f"❌ Authentication failed: {e}")


def demo_configuration():
    """Demonstrate configuration management"""
    print("\n⚙️ Configuration Management Demo")
    print("=" * 50)

    # Initialize config manager
    config_manager = MCPConfigManager()

    # Load existing configuration
    config_path = Path(__file__).parent.parent.parent.parent / ".mcp.json"

    try:
        config_data = config_manager.load_config(config_path)
        print(f"✅ Loaded configuration with {len(config_data.get('mcpServers', {}))} servers")

        # Show server configurations
        servers = config_manager.get_all_servers()
        for name, server in servers.items():
            print(f"  📡 {name}: {server.command} {' '.join(server.args[:2])}")

        # Show security configuration
        security_config = config_manager.get_security_config()
        if security_config:
            print(f"🔒 Security Mode: {security_config.authentication_mode}")

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")

        # Demonstrate creating a template configuration
        print("\n📝 Creating template configuration...")

        template_server = config_manager.create_server_template(ServerType.CODE_REVIEWER)
        config_manager.add_server(template_server)
        print(f"✅ Created template for {template_server.name}")


def demo_security():
    """Demonstrate security management"""
    print("\n🛡️ Security Management Demo")
    print("=" * 50)

    # Initialize security manager
    security_manager = SecurityManager()

    # Set policies for different server types
    security_manager.set_policy("gemini-code-reviewer", SecurityLevel.SECURE)
    security_manager.set_policy("unified-mcp-gateway", SecurityLevel.GATEWAY)
    security_manager.set_policy("unified-monitoring-server", SecurityLevel.MONITORING)

    # Test command validation
    test_commands = [
        ("echo", ["hello world"]),
        ("uv", ["run", "python", "-m", "app.mcp_servers.gemini_code_reviewer"]),
        ("rm", ["-rf", "/"]),  # Should be blocked
        ("sudo", ["rm", "file"]),  # Should be blocked
    ]

    for command, args in test_commands:
        is_allowed, reason = security_manager.validate_command(
            "gemini-code-reviewer", command, args
        )
        status = "✅" if is_allowed else "❌"
        print(f"  {status} {command} {' '.join(args)} - {reason or 'Allowed'}")

    # Test path validation
    test_paths = [
        "/home/david/agents/my-fullstack-agent/app",  # Should be allowed
        "/etc/passwd",  # Should be blocked
        "/home/david/.ssh/id_rsa",  # Should be blocked
    ]

    print("\nPath Access Validation:")
    for path in test_paths:
        is_allowed, reason = security_manager.validate_path_access(
            "gemini-code-reviewer", path, "read"
        )
        status = "✅" if is_allowed else "❌"
        print(f"  {status} {path} - {reason or 'Allowed'}")

    # Generate security report
    report = security_manager.get_security_report()
    print("\n📊 Security Report:")
    print(f"  Servers managed: {report['total_servers_managed']}")
    print(f"  Total violations: {report['total_violations']}")
    print(f"  Policies configured: {report['policies_configured']}")


async def demo_server_registry():
    """Demonstrate server registry and lifecycle management"""
    print("\n🖥️ Server Registry Demo")
    print("=" * 50)

    # Initialize all components
    auth = ConsolidatedAuth()
    config_manager = MCPConfigManager()
    security_manager = SecurityManager()

    # Load configuration if available
    config_path = Path(__file__).parent.parent.parent.parent / ".mcp.json"

    try:
        config_manager.load_config(config_path)
    except Exception:
        # Create a simple test server if config loading fails
        from gterminal.core.mcp.config_manager import ServerConfig

        test_server = ServerConfig(
            name="test-echo-server",
            command="echo",
            args=["MCP Server Running"],
            env={"LOG_LEVEL": "INFO"},
        )
        config_manager.add_server(test_server)

    # Initialize server registry
    registry = ServerRegistry(config_manager, security_manager, auth)

    # Register servers from configuration
    servers = config_manager.get_all_servers()
    for server in servers.values():
        registry.register_server(server)

    print(f"✅ Registered {len(servers)} servers")

    # Show server status
    all_status = registry.get_all_server_status()
    print(f"📊 Total servers: {all_status['total_servers']}")
    print(f"🟢 Running: {all_status['running_servers']}")
    print(f"🔴 Failed: {all_status['failed_servers']}")

    # Note: We won't actually start servers in this demo to avoid conflicts
    print("(Server startup skipped in demo mode)")


async def main():
    """Run all demonstrations"""
    print("🚀 MCP Consolidation Framework Demo")
    print("=" * 70)

    try:
        await demo_authentication()
        demo_configuration()
        demo_security()
        await demo_server_registry()

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
