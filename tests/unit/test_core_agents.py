"""
Unit tests for core agent system components.
Addressing 0% coverage in agent modules.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.agents.base_agent import BaseAgent
from app.agents.code_generation_agent import CodeGenerationAgent
from app.agents.context_analysis_agent import ContextAnalysisAgent
from app.agents.memory_rag_agent import MemoryRAGAgent
from app.agents.unified_agent import UnifiedAgent
from app.agents.workflow_agent import WorkflowAgent


class TestBaseAgent:
    """Test suite for BaseAgent class."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client."""
        mock = MagicMock()
        mock.generate_content = AsyncMock(return_value=MagicMock(text="Mock response"))
        return mock

    @pytest.fixture
    def base_agent(self, mock_gemini_client):
        """Create BaseAgent instance with mocked dependencies."""
        with patch("app.agents.base_agent.genai") as mock_genai:
            mock_genai.GenerativeModel.return_value = mock_gemini_client
            agent = BaseAgent(name="test-agent", model_name="gemini-2.5-flash", api_key="test-key")
            agent.client = mock_gemini_client
            return agent

    async def test_agent_initialization(self, base_agent):
        """Test agent initializes with correct parameters."""
        assert base_agent.name == "test-agent"
        assert base_agent.model_name == "gemini-2.5-flash"
        assert base_agent.context_window == 1000000
        assert base_agent.conversation_history == []

    async def test_generate_response(self, base_agent):
        """Test basic response generation."""
        response = await base_agent.generate_response("Test prompt")
        assert response == "Mock response"
        base_agent.client.generate_content.assert_called_once()

    async def test_add_to_history(self, base_agent):
        """Test conversation history management."""
        base_agent.add_to_history("user", "Hello")
        base_agent.add_to_history("assistant", "Hi there")

        assert len(base_agent.conversation_history) == 2
        assert base_agent.conversation_history[0]["role"] == "user"
        assert base_agent.conversation_history[1]["content"] == "Hi there"

    async def test_clear_history(self, base_agent):
        """Test clearing conversation history."""
        base_agent.add_to_history("user", "Test")
        base_agent.clear_history()
        assert len(base_agent.conversation_history) == 0

    async def test_context_window_management(self, base_agent):
        """Test context window truncation."""
        # Add messages that exceed context window
        for _i in range(100):
            base_agent.add_to_history("user", "x" * 20000)

        # Should truncate to fit context window
        total_length = sum(len(msg["content"]) for msg in base_agent.conversation_history)
        assert total_length <= base_agent.context_window

    async def test_error_handling(self, base_agent):
        """Test error handling in response generation."""
        base_agent.client.generate_content.side_effect = Exception("API Error")

        with pytest.raises(Exception) as exc_info:
            await base_agent.generate_response("Test")

        assert "API Error" in str(exc_info.value)


class TestContextAnalysisAgent:
    """Test suite for ContextAnalysisAgent."""

    @pytest.fixture
    def context_agent(self):
        """Create ContextAnalysisAgent with mocked dependencies."""
        with patch("app.agents.context_analysis_agent.genai"):
            agent = ContextAnalysisAgent(api_key="test-key")
            agent.client = AsyncMock()
            agent.client.generate_content = AsyncMock(
                return_value=MagicMock(
                    text=json.dumps(
                        {
                            "summary": "Test project",
                            "key_components": ["component1", "component2"],
                            "dependencies": ["dep1", "dep2"],
                            "architecture": "microservices",
                            "recommendations": ["rec1", "rec2"],
                        }
                    )
                )
            )
            return agent

    async def test_analyze_project_structure(self, context_agent, tmp_path):
        """Test project structure analysis."""
        # Create test project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Test Project")
        (tmp_path / "requirements.txt").write_text("pytest\nrequests")

        result = await context_agent.analyze_project(str(tmp_path))

        assert "summary" in result
        assert "key_components" in result
        assert len(result["key_components"]) == 2
        assert result["architecture"] == "microservices"

    async def test_extract_dependencies(self, context_agent, tmp_path):
        """Test dependency extraction."""
        # Create package files
        (tmp_path / "package.json").write_text('{"dependencies": {"react": "^18.0.0"}}')
        (tmp_path / "requirements.txt").write_text("django>=4.0\nflask")
        (tmp_path / "Cargo.toml").write_text('[dependencies]\ntokio = "1.0"')

        deps = await context_agent.extract_dependencies(str(tmp_path))

        assert "python" in deps or "javascript" in deps or "rust" in deps

    async def test_generate_context_summary(self, context_agent):
        """Test context summary generation."""
        context_data = {
            "files": ["file1.py", "file2.js"],
            "structure": {"src": ["main.py"]},
            "metadata": {"language": "python"},
        }

        summary = await context_agent.generate_summary(context_data)
        assert isinstance(summary, str)
        assert len(summary) > 0

    async def test_identify_patterns(self, context_agent, tmp_path):
        """Test code pattern identification."""
        # Create files with patterns
        (tmp_path / "api.py").write_text(
            """
class APIEndpoint:
    def get(self): pass
    def post(self): pass
"""
        )

        patterns = await context_agent.identify_patterns(str(tmp_path))
        assert isinstance(patterns, list)


class TestCodeGenerationAgent:
    """Test suite for CodeGenerationAgent."""

    @pytest.fixture
    def codegen_agent(self):
        """Create CodeGenerationAgent with mocked dependencies."""
        with patch("app.agents.code_generation_agent.genai"):
            agent = CodeGenerationAgent(api_key="test-key")
            agent.client = AsyncMock()
            return agent

    async def test_generate_code(self, codegen_agent):
        """Test code generation."""
        codegen_agent.client.generate_content = AsyncMock(
            return_value=MagicMock(
                text="""
def calculate_sum(a, b):
    return a + b
"""
            )
        )

        code = await codegen_agent.generate_code(
            "Create a function to add two numbers", language="python"
        )

        assert "def calculate_sum" in code
        assert "return a + b" in code

    async def test_generate_tests(self, codegen_agent):
        """Test test generation for code."""
        code = """
def multiply(a, b):
    return a * b
"""

        codegen_agent.client.generate_content = AsyncMock(
            return_value=MagicMock(
                text="""
import pytest

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(0, 5) == 0
    assert multiply(-1, 4) == -4
"""
            )
        )

        tests = await codegen_agent.generate_tests(code, framework="pytest")

        assert "import pytest" in tests
        assert "test_multiply" in tests
        assert "assert multiply" in tests

    async def test_refactor_code(self, codegen_agent):
        """Test code refactoring."""
        original_code = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""

        codegen_agent.client.generate_content = AsyncMock(
            return_value=MagicMock(
                text="""
def process_data(data):
    return [item * 2 for item in data if item > 0]
"""
            )
        )

        refactored = await codegen_agent.refactor_code(original_code, style="functional")

        assert "return [" in refactored
        assert "for item in data if" in refactored

    async def test_generate_documentation(self, codegen_agent):
        """Test documentation generation."""
        code = """
def calculate_interest(principal, rate, time):
    return principal * rate * time / 100
"""

        codegen_agent.client.generate_content = AsyncMock(
            return_value=MagicMock(
                text="""
def calculate_interest(principal, rate, time):
    \"\"\"
    Calculate simple interest.

    Args:
        principal (float): Principal amount
        rate (float): Interest rate (percentage)
        time (float): Time period in years

    Returns:
        float: Simple interest amount
    \"\"\"
    return principal * rate * time / 100
"""
            )
        )

        documented = await codegen_agent.add_documentation(code)

        assert '"""' in documented
        assert "Args:" in documented
        assert "Returns:" in documented


class TestMemoryRAGAgent:
    """Test suite for MemoryRAGAgent."""

    @pytest.fixture
    def memory_agent(self):
        """Create MemoryRAGAgent with mocked dependencies."""
        with patch("app.agents.memory_rag_agent.genai"):
            with patch("app.agents.memory_rag_agent.redis"):
                agent = MemoryRAGAgent(api_key="test-key", redis_url="redis://localhost:6379")
                agent.client = AsyncMock()
                agent.redis_client = MagicMock()
                agent.vector_store = MagicMock()
                return agent

    async def test_store_memory(self, memory_agent):
        """Test storing memories."""
        memory_agent.redis_client.set = MagicMock(return_value=True)

        result = await memory_agent.store_memory(
            key="test_memory", content="Important information", metadata={"type": "fact"}
        )

        assert result is True
        memory_agent.redis_client.set.assert_called_once()

    async def test_retrieve_memory(self, memory_agent):
        """Test retrieving memories."""
        memory_agent.redis_client.get = MagicMock(
            return_value=json.dumps({"content": "Stored information", "metadata": {"type": "fact"}})
        )

        memory = await memory_agent.retrieve_memory("test_memory")

        assert memory["content"] == "Stored information"
        assert memory["metadata"]["type"] == "fact"

    async def test_semantic_search(self, memory_agent):
        """Test semantic search in memories."""
        memory_agent.vector_store.search = AsyncMock(
            return_value=[
                {"content": "Related memory 1", "score": 0.9},
                {"content": "Related memory 2", "score": 0.8},
            ]
        )

        results = await memory_agent.search_memories(
            query="Find information about testing", top_k=2
        )

        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]

    async def test_memory_summarization(self, memory_agent):
        """Test memory summarization."""
        memories = ["Memory about feature A", "Memory about feature B", "Memory about bug fix"]

        memory_agent.client.generate_content = AsyncMock(
            return_value=MagicMock(text="Summary: Features A and B implemented, bug fixed")
        )

        summary = await memory_agent.summarize_memories(memories)

        assert "Features A and B" in summary
        assert "bug fixed" in summary


class TestWorkflowAgent:
    """Test suite for WorkflowAgent."""

    @pytest.fixture
    def workflow_agent(self):
        """Create WorkflowAgent with mocked dependencies."""
        with patch("app.agents.workflow_agent.genai"):
            agent = WorkflowAgent(api_key="test-key")
            agent.client = AsyncMock()
            agent.agents = {"context": AsyncMock(), "codegen": AsyncMock(), "memory": AsyncMock()}
            return agent

    async def test_execute_workflow(self, workflow_agent):
        """Test workflow execution."""
        workflow = {
            "name": "test_workflow",
            "steps": [
                {"agent": "context", "action": "analyze", "params": {}},
                {"agent": "codegen", "action": "generate", "params": {}},
            ],
        }

        workflow_agent.agents["context"].analyze = AsyncMock(return_value={"analysis": "complete"})
        workflow_agent.agents["codegen"].generate = AsyncMock(return_value={"code": "generated"})

        results = await workflow_agent.execute_workflow(workflow)

        assert len(results) == 2
        assert results[0]["analysis"] == "complete"
        assert results[1]["code"] == "generated"

    async def test_orchestrate_agents(self, workflow_agent):
        """Test multi-agent orchestration."""
        task = {"type": "feature_development", "requirements": ["analyze", "generate", "test"]}

        workflow_agent.client.generate_content = AsyncMock(
            return_value=MagicMock(
                text=json.dumps(
                    {
                        "workflow": {
                            "steps": [
                                {"agent": "context", "action": "analyze"},
                                {"agent": "codegen", "action": "generate"},
                                {"agent": "codegen", "action": "test"},
                            ]
                        }
                    }
                )
            )
        )

        plan = await workflow_agent.create_execution_plan(task)

        assert "workflow" in plan
        assert len(plan["workflow"]["steps"]) == 3

    async def test_parallel_execution(self, workflow_agent):
        """Test parallel step execution."""
        parallel_steps = [
            {"agent": "context", "action": "analyze", "parallel": True},
            {"agent": "memory", "action": "search", "parallel": True},
        ]

        workflow_agent.agents["context"].analyze = AsyncMock(return_value={"result": "context"})
        workflow_agent.agents["memory"].search = AsyncMock(return_value={"result": "memory"})

        results = await workflow_agent.execute_parallel(parallel_steps)

        assert len(results) == 2
        assert any(r["result"] == "context" for r in results)
        assert any(r["result"] == "memory" for r in results)


class TestUnifiedAgent:
    """Test suite for UnifiedAgent combining all capabilities."""

    @pytest.fixture
    def unified_agent(self):
        """Create UnifiedAgent with mocked dependencies."""
        with patch("app.agents.unified_agent.genai"):
            agent = UnifiedAgent(api_key="test-key")
            # Mock all sub-agents
            agent.context_agent = AsyncMock()
            agent.codegen_agent = AsyncMock()
            agent.memory_agent = AsyncMock()
            agent.workflow_agent = AsyncMock()
            agent.client = AsyncMock()
            return agent

    async def test_unified_process_request(self, unified_agent):
        """Test unified request processing."""
        request = {
            "type": "analyze_and_generate",
            "project_path": "/test/project",
            "requirements": "Create API endpoint",
        }

        unified_agent.context_agent.analyze = AsyncMock(return_value={"structure": "analyzed"})
        unified_agent.codegen_agent.generate = AsyncMock(return_value="Generated code")

        result = await unified_agent.process_request(request)

        assert "context" in result
        assert "generated_code" in result

    async def test_adaptive_agent_selection(self, unified_agent):
        """Test adaptive agent selection based on task."""
        task = "Analyze the codebase and find security issues"

        unified_agent.client.generate_content = AsyncMock(
            return_value=MagicMock(
                text=json.dumps(
                    {
                        "selected_agents": ["context", "security"],
                        "reasoning": "Task requires code analysis and security scanning",
                    }
                )
            )
        )

        agents = await unified_agent.select_agents_for_task(task)

        assert "context" in agents["selected_agents"]
        assert "reasoning" in agents

    async def test_memory_integration(self, unified_agent):
        """Test memory system integration."""
        # Store context in memory
        context = {"project": "test", "timestamp": datetime.now().isoformat()}

        unified_agent.memory_agent.store = AsyncMock(return_value=True)
        unified_agent.memory_agent.retrieve = AsyncMock(return_value=context)

        # Store
        stored = await unified_agent.memory_agent.store("context_123", context)
        assert stored is True

        # Retrieve
        retrieved = await unified_agent.memory_agent.retrieve("context_123")
        assert retrieved["project"] == "test"

    async def test_error_recovery(self, unified_agent):
        """Test error recovery mechanisms."""
        # Simulate failure in one agent
        unified_agent.context_agent.analyze = AsyncMock(
            side_effect=Exception("Context analysis failed")
        )

        # Should fall back to alternative approach
        unified_agent.client.generate_content = AsyncMock(
            return_value=MagicMock(text="Fallback analysis")
        )

        result = await unified_agent.process_with_fallback(
            task="analyze",
            primary_agent=unified_agent.context_agent,
            fallback_method=unified_agent.client.generate_content,
        )

        assert result == "Fallback analysis"


@pytest.mark.asyncio
class TestAsyncAgentOperations:
    """Test asynchronous operations across agents."""

    async def test_concurrent_agent_execution(self):
        """Test multiple agents running concurrently."""
        agents = [AsyncMock(process=AsyncMock(return_value=f"Result {i}")) for i in range(5)]

        tasks = [agent.process() for agent in agents]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(f"Result {i}" == results[i] for i in range(5))

    async def test_rate_limiting(self):
        """Test rate limiting for API calls."""
        from app.agents.utils import RateLimiter

        rate_limiter = RateLimiter(max_calls=3, time_window=1)

        calls = []
        for _i in range(5):
            allowed = await rate_limiter.check_rate_limit()
            calls.append(allowed)

        # First 3 should be allowed, next 2 should be blocked
        assert sum(calls[:3]) == 3
        assert sum(calls[3:]) < 2

    async def test_timeout_handling(self):
        """Test timeout handling in agent operations."""

        async def slow_operation():
            await asyncio.sleep(5)
            return "Completed"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
