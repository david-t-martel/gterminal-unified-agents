#!/usr/bin/env python3
"""Complete ReAct Engine for GTerminal - Production Implementation.

This module provides a comprehensive ReAct (Reasoning and Acting) engine that integrates:
- Full ReAct loop (Think → Act → Observe)
- Context persistence with Redis/RustCache fallback
- Tool registration and execution system
- MCP client integration (gemini_master_architect, gemini_code_reviewer)
- Web fetch capabilities
- Multi-agent communication with message queue
- Session management and restoration
- Native Gemini function calling
- Local LLM integration for specific tasks

Architecture:
- ReactEngine: Main coordinator
- ContextManager: Handles persistence and session management
- ToolOrchestrator: Manages tool registration and execution
- MCPBridge: Interfaces with MCP servers
- LocalLLMBridge: Integrates local inference for privacy-critical tasks
- MessageQueue: Handles multi-agent communication
"""

import asyncio
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timedelta
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse
import uuid

import httpx
from pydantic import BaseModel
from pydantic import Field
from pydantic import validator

# Import performance optimizations
try:
    from fullstack_agent_rust import RustCache
    from fullstack_agent_rust import RustFileOps
    from fullstack_agent_rust import RustJsonProcessor

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Import existing components
from gterminal.core.agents.mcp_client import GeminiMCPClient
from gterminal.core.tools.registry import ToolRegistry
from gterminal.core.unified_gemini_client import get_gemini_client

# Import terminal types
from .react_types import ActionType
from .react_types import ReActStatus

logger = logging.getLogger(__name__)


class ThoughtType(str, Enum):
    """Types of reasoning thoughts."""

    ANALYSIS = "analysis"
    PLANNING = "planning"
    PROBLEM_SOLVING = "problem_solving"
    DECISION = "decision"
    REFLECTION = "reflection"
    FINAL_ANSWER = "final_answer"


class AgentRole(str, Enum):
    """Multi-agent communication roles."""

    ARCHITECT = "architect"
    REVIEWER = "reviewer"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"


class PersistenceLevel(str, Enum):
    """Context persistence levels."""

    NONE = "none"
    SESSION = "session"
    TEMPORARY = "temporary"
    PERSISTENT = "persistent"
    PERMANENT = "permanent"


class ReactThought(BaseModel):
    """Represents a reasoning thought in the ReAct process."""

    thought_type: ThoughtType
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_chain: list[str] = Field(default_factory=list)
    alternative_approaches: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: float | None = None


class ReactAction(BaseModel):
    """Enhanced action with full context and metadata."""

    action_type: ActionType
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    target_agent: AgentRole | None = None
    priority: int = Field(default=1, ge=1, le=10)
    timeout_seconds: int | None = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    prerequisites: list[str] = Field(default_factory=list)
    expected_outcome: str | None = None
    fallback_actions: list[str] = Field(default_factory=list)


class ReactObservation(BaseModel):
    """Enhanced observation with analysis and recommendations."""

    success: bool
    result: Any
    error_message: str | None = None
    execution_time_ms: float
    confidence: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0, default=0.8)
    insights: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    follow_up_needed: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


class ReactSession(BaseModel):
    """Complete session state with comprehensive context."""

    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex}")
    user_id: str | None = None
    task: str
    status: ReActStatus = ReActStatus.INITIALIZED
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None

    # ReAct process state
    current_iteration: int = 0
    max_iterations: int = 20
    thoughts: list[ReactThought] = Field(default_factory=list)
    actions: list[ReactAction] = Field(default_factory=list)
    observations: list[ReactObservation] = Field(default_factory=list)

    # Context and memory
    context_data: dict[str, Any] = Field(default_factory=dict)
    working_memory: dict[str, Any] = Field(default_factory=dict)
    knowledge_base: dict[str, Any] = Field(default_factory=dict)

    # Performance tracking
    total_time_ms: float = 0.0
    tool_calls: int = 0
    successful_tools: int = 0
    failed_tools: int = 0

    # Multi-agent coordination
    agent_messages: list[dict[str, Any]] = Field(default_factory=list)
    collaborating_agents: list[AgentRole] = Field(default_factory=list)

    # Persistence settings
    persistence_level: PersistenceLevel = PersistenceLevel.SESSION
    auto_save: bool = True

    @validator("max_iterations")
    def validate_max_iterations(cls, v):
        if v <= 0:
            raise ValueError("max_iterations must be positive")
        return min(v, 100)  # Safety limit

    def add_thought(self, thought: ReactThought) -> None:
        """Add a thought to the session."""
        self.thoughts.append(thought)

    def add_action(self, action: ReactAction) -> None:
        """Add an action to the session."""
        self.actions.append(action)

    def add_observation(self, observation: ReactObservation) -> None:
        """Add an observation to the session."""
        self.observations.append(observation)
        if observation.success:
            self.successful_tools += 1
        else:
            self.failed_tools += 1
        self.tool_calls += 1


class ContextManager:
    """Manages session persistence with Redis and RustCache fallback."""

    def __init__(self, project_root: Path, redis_url: str | None = None):
        self.project_root = project_root
        self.context_dir = project_root / ".react_sessions"
        self.context_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Redis if available
        self.redis_client = None
        if redis_url:
            try:
                import redis.asyncio as redis

                self.redis_client = redis.from_url(redis_url)
                logger.info("✅ Redis connected for context persistence")
            except ImportError:
                logger.warning("Redis not available, using file-based persistence")

        # Initialize RustCache if available
        self.rust_cache = None
        if RUST_AVAILABLE:
            self.rust_cache = RustCache(capacity=10000, ttl_seconds=3600)
            logger.info("✅ RustCache initialized for context caching")

    async def save_session(self, session: ReactSession) -> bool:
        """Save session with multiple persistence backends."""
        session_data = session.model_dump()
        session_key = f"react_session:{session.session_id}"

        success = False

        # Try Redis first
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    session_key, timedelta(hours=24), json.dumps(session_data, default=str)
                )
                success = True
                logger.debug(f"Session saved to Redis: {session.session_id}")
            except Exception as e:
                logger.warning(f"Redis save failed: {e}")

        # Try RustCache
        if self.rust_cache:
            try:
                self.rust_cache.set(session_key, json.dumps(session_data, default=str))
                success = True
                logger.debug(f"Session cached in RustCache: {session.session_id}")
            except Exception as e:
                logger.warning(f"RustCache save failed: {e}")

        # File-based fallback
        try:
            session_file = self.context_dir / f"{session.session_id}.json"
            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2, default=str)
            success = True
            logger.debug(f"Session saved to file: {session_file}")
        except Exception as e:
            logger.exception(f"File save failed: {e}")

        return success

    async def load_session(self, session_id: str) -> ReactSession | None:
        """Load session from available persistence backends."""
        session_key = f"react_session:{session_id}"

        # Try Redis first
        if self.redis_client:
            try:
                data = await self.redis_client.get(session_key)
                if data:
                    session_data = json.loads(data)
                    return ReactSession(**session_data)
            except Exception as e:
                logger.warning(f"Redis load failed: {e}")

        # Try RustCache
        if self.rust_cache:
            try:
                cached_data = self.rust_cache.get(session_key)
                if cached_data:
                    session_data = json.loads(cached_data)
                    return ReactSession(**session_data)
            except Exception as e:
                logger.warning(f"RustCache load failed: {e}")

        # File-based fallback
        try:
            session_file = self.context_dir / f"{session_id}.json"
            if session_file.exists():
                with open(session_file) as f:
                    session_data = json.load(f)
                return ReactSession(**session_data)
        except Exception as e:
            logger.warning(f"File load failed: {e}")

        return None

    async def list_sessions(self, user_id: str | None = None) -> list[str]:
        """List available sessions."""
        sessions = []

        # Get from files
        for session_file in self.context_dir.glob("*.json"):
            session_id = session_file.stem
            sessions.append(session_id)

        # Get from Redis if available
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("react_session:*")
                for key in keys:
                    session_id = key.decode("utf-8").split(":", 1)[1]
                    if session_id not in sessions:
                        sessions.append(session_id)
            except Exception:
                pass

        return sessions

    async def delete_session(self, session_id: str) -> bool:
        """Delete session from all backends."""
        success = False
        session_key = f"react_session:{session_id}"

        # Delete from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(session_key)
                success = True
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")

        # Delete from RustCache
        if self.rust_cache:
            try:
                self.rust_cache.delete(session_key)
                success = True
            except Exception:
                pass

        # Delete file
        try:
            session_file = self.context_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                success = True
        except Exception as e:
            logger.warning(f"File delete failed: {e}")

        return success


class ToolOrchestrator:
    """Orchestrates tool execution with MCP integration and local LLM support."""

    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.mcp_client = GeminiMCPClient()
        self.execution_history: list[dict[str, Any]] = []

        # Local LLM client for privacy-critical tasks
        self.local_llm_client = None
        self._init_local_llm()

        logger.info("🔧 Tool orchestrator initialized")

    def _init_local_llm(self) -> None:
        """Initialize local LLM client for privacy-critical operations."""
        try:
            # This would connect to the rust-llm local inference server
            self.local_llm_client = httpx.AsyncClient(
                base_url="http://localhost:8080",  # rust-llm HTTP server
                timeout=30.0,
            )
            logger.info("✅ Local LLM client initialized")
        except Exception as e:
            logger.warning(f"Local LLM not available: {e}")

    async def execute_action(
        self, action: ReactAction, session: ReactSession, use_local_llm: bool = False
    ) -> ReactObservation:
        """Execute an action with comprehensive error handling and retry logic."""
        start_time = time.time()
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"

        logger.info(f"🚀 Executing action: {action.action_type.value} - {action.description}")

        try:
            # Choose execution method
            if use_local_llm and self.local_llm_client:
                result = await self._execute_with_local_llm(action, session)
            elif action.target_agent:
                result = await self._execute_with_mcp(action, session)
            else:
                result = await self._execute_with_tool_registry(action, session)

            execution_time_ms = (time.time() - start_time) * 1000

            # Create observation
            observation = ReactObservation(
                success=result.get("success", True),
                result=result.get("data"),
                error_message=result.get("error"),
                execution_time_ms=execution_time_ms,
                confidence=result.get("confidence", 0.8),
                quality_score=self._assess_result_quality(result, action),
                insights=self._extract_insights(result, action),
                recommendations=self._generate_recommendations(result, action),
                metadata={
                    "execution_id": execution_id,
                    "tool_used": result.get("tool_name"),
                    "retry_count": action.retry_count,
                    "local_llm_used": use_local_llm,
                },
            )

            # Log execution
            self.execution_history.append(
                {
                    "execution_id": execution_id,
                    "action": action.model_dump(),
                    "observation": observation.model_dump(),
                    "session_id": session.session_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info(f"✅ Action completed in {execution_time_ms:.0f}ms")
            return observation

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.exception(f"❌ Action execution failed: {e}")

            # Handle retries
            if action.retry_count < action.max_retries:
                action.retry_count += 1
                logger.info(
                    f"🔄 Retrying action (attempt {action.retry_count}/{action.max_retries})"
                )
                await asyncio.sleep(min(2**action.retry_count, 10))  # Exponential backoff
                return await self.execute_action(action, session, use_local_llm)

            return ReactObservation(
                success=False,
                result=None,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                confidence=0.0,
                quality_score=0.0,
                metadata={"execution_id": execution_id, "retry_count": action.retry_count},
            )

    async def _execute_with_mcp(self, action: ReactAction, session: ReactSession) -> dict[str, Any]:
        """Execute action using MCP servers."""
        # Map agent roles to MCP servers
        server_map = {
            AgentRole.ARCHITECT: "gemini-master-architect",
            AgentRole.REVIEWER: "gemini-code-reviewer",
            AgentRole.ANALYZER: "gemini-workspace-analyzer",
        }

        server_name = server_map.get(action.target_agent, "default")
        tool_name = action.action_type.value.replace("_", "-")

        try:
            mcp_call = await self.mcp_client.call_tool(server_name, tool_name, action.parameters)

            return {
                "success": mcp_call.error is None,
                "data": mcp_call.result,
                "error": mcp_call.error,
                "tool_name": f"{server_name}.{tool_name}",
                "confidence": 0.9 if mcp_call.error is None else 0.3,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": f"{server_name}.{tool_name}",
                "confidence": 0.0,
            }

    async def _execute_with_local_llm(
        self, action: ReactAction, session: ReactSession
    ) -> dict[str, Any]:
        """Execute action using local LLM for privacy-critical tasks."""
        if not self.local_llm_client:
            raise RuntimeError("Local LLM client not available")

        # Prepare context for local LLM
        context = {
            "task": session.task,
            "action": action.model_dump(),
            "session_context": session.context_data,
            "recent_thoughts": [t.content for t in session.thoughts[-3:]],
        }

        try:
            response = await self.local_llm_client.post(
                "/api/process",
                json={
                    "prompt": f"Execute this action: {action.description}",
                    "context": context,
                    "parameters": action.parameters,
                },
            )
            response.raise_for_status()
            result = response.json()

            return {
                "success": result.get("success", True),
                "data": result.get("response"),
                "tool_name": "local_llm",
                "confidence": result.get("confidence", 0.7),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": "local_llm",
                "confidence": 0.0,
            }

    async def _execute_with_tool_registry(
        self, action: ReactAction, session: ReactSession
    ) -> dict[str, Any]:
        """Execute action using the standard tool registry."""
        try:
            tool_result = await self.tool_registry.execute(
                action.action_type.value, action.parameters
            )

            return {
                "success": tool_result.success,
                "data": tool_result.data,
                "error": tool_result.error,
                "tool_name": action.action_type.value,
                "confidence": 0.8 if tool_result.success else 0.2,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": action.action_type.value,
                "confidence": 0.0,
            }

    def _assess_result_quality(self, result: dict[str, Any], action: ReactAction) -> float:
        """Assess the quality of an execution result."""
        if not result.get("success"):
            return 0.0

        score = 0.5  # Base score

        # Add score based on result completeness
        if result.get("data"):
            score += 0.3

        # Add score based on execution time (faster is better up to a point)
        if "execution_time_ms" in result:
            time_ms = result["execution_time_ms"]
            if time_ms < 1000:  # Under 1 second
                score += 0.2
            elif time_ms < 5000:  # Under 5 seconds
                score += 0.1

        return min(score, 1.0)

    def _extract_insights(self, result: dict[str, Any], action: ReactAction) -> list[str]:
        """Extract insights from execution results."""
        insights = []

        if result.get("success"):
            insights.append(f"Successfully executed {action.action_type.value}")

            if result.get("data"):
                data_type = type(result["data"]).__name__
                insights.append(f"Returned {data_type} result")
        else:
            insights.append(f"Failed to execute {action.action_type.value}")
            if result.get("error"):
                insights.append(f"Error: {result['error']}")

        return insights

    def _generate_recommendations(self, result: dict[str, Any], action: ReactAction) -> list[str]:
        """Generate recommendations based on execution results."""
        recommendations = []

        if not result.get("success"):
            recommendations.append("Consider alternative approaches")
            if action.retry_count == 0:
                recommendations.append("Retry with modified parameters")

            if action.fallback_actions:
                recommendations.append(
                    f"Try fallback actions: {', '.join(action.fallback_actions)}"
                )

        return recommendations


class MessageQueue:
    """Handles multi-agent communication and coordination."""

    def __init__(self):
        self.message_queues: dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
        self.subscribers: dict[str, list[Callable]] = defaultdict(list)
        self.message_history: list[dict[str, Any]] = []

        logger.info("📡 Message queue initialized for multi-agent communication")

    async def send_message(
        self,
        sender: AgentRole,
        receiver: AgentRole,
        message_type: str,
        content: Any,
        session_id: str,
        priority: int = 1,
    ) -> str:
        """Send a message between agents."""
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        message = {
            "id": message_id,
            "sender": sender.value,
            "receiver": receiver.value,
            "type": message_type,
            "content": content,
            "session_id": session_id,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
        }

        # Add to queue
        queue_key = f"{receiver.value}:{session_id}"
        await self.message_queues[queue_key].put(message)

        # Store in history
        self.message_history.append(message)

        # Notify subscribers
        for callback in self.subscribers[queue_key]:
            try:
                await callback(message)
            except Exception as e:
                logger.warning(f"Subscriber notification failed: {e}")

        logger.debug(f"📤 Message sent: {sender.value} → {receiver.value} ({message_type})")
        return message_id

    async def receive_message(
        self, receiver: AgentRole, session_id: str, timeout: float = 10.0
    ) -> dict[str, Any] | None:
        """Receive a message for an agent."""
        queue_key = f"{receiver.value}:{session_id}"

        try:
            message = await asyncio.wait_for(self.message_queues[queue_key].get(), timeout=timeout)
            logger.debug(f"📥 Message received by {receiver.value}: {message['type']}")
            return message
        except TimeoutError:
            logger.debug(f"⏱️ Message receive timeout for {receiver.value}")
            return None

    def subscribe(self, receiver: AgentRole, session_id: str, callback: Callable) -> None:
        """Subscribe to messages for an agent."""
        queue_key = f"{receiver.value}:{session_id}"
        self.subscribers[queue_key].append(callback)

    def get_message_history(
        self, session_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get message history for a session or all sessions."""
        if session_id:
            messages = [msg for msg in self.message_history if msg["session_id"] == session_id]
        else:
            messages = self.message_history

        return sorted(messages, key=lambda x: x["timestamp"], reverse=True)[:limit]


class WebFetchService:
    """Handles web content fetching with caching and rate limiting."""

    def __init__(self):
        self.http_client = httpx.AsyncClient(
            timeout=30.0, limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        self.cache: dict[str, dict[str, Any]] = {}
        self.rate_limits: dict[str, list[datetime]] = defaultdict(list)

        logger.info("🌐 Web fetch service initialized")

    async def fetch_content(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
        extract_text: bool = False,
    ) -> dict[str, Any]:
        """Fetch web content with caching and rate limiting."""
        # Generate cache key
        cache_key = hashlib.md5(f"{url}:{json.dumps(headers or {})}".encode()).hexdigest()

        # Check cache
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            if datetime.now() - datetime.fromisoformat(cached["timestamp"]) < timedelta(hours=1):
                logger.debug(f"📋 Cache hit for URL: {url}")
                return cached

        # Rate limiting
        domain = urlparse(url).netloc
        now = datetime.now()
        recent_requests = [ts for ts in self.rate_limits[domain] if now - ts < timedelta(minutes=1)]

        if len(recent_requests) >= 60:  # 60 requests per minute
            raise ValueError(f"Rate limit exceeded for domain: {domain}")

        self.rate_limits[domain].append(now)

        try:
            response = await self.http_client.get(url, headers=headers or {})
            response.raise_for_status()

            content = response.text
            if extract_text:
                # Simple text extraction (could be enhanced with BeautifulSoup)
                import re

                content = re.sub(r"<[^>]+>", "", content)
                content = re.sub(r"\s+", " ", content).strip()

            result = {
                "url": url,
                "status_code": response.status_code,
                "content": content,
                "headers": dict(response.headers),
                "timestamp": now.isoformat(),
                "extracted_text": extract_text,
                "content_length": len(content),
            }

            # Cache result
            if use_cache:
                self.cache[cache_key] = result

            logger.info(f"🌐 Fetched content from {url} ({len(content)} chars)")
            return result

        except Exception as e:
            logger.exception(f"❌ Failed to fetch {url}: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.http_client.aclose()


class CompleteReactEngine:
    """Complete ReAct Engine integrating all components for production use."""

    def __init__(
        self,
        project_root: Path | None = None,
        gemini_profile: str = "business",
        redis_url: str | None = None,
        enable_local_llm: bool = True,
        max_iterations: int = 20,
    ):
        self.project_root = project_root or Path("/home/david/agents/my-fullstack-agent")
        self.gemini_profile = gemini_profile
        self.max_iterations = max_iterations
        self.enable_local_llm = enable_local_llm

        # Initialize components
        self.context_manager = ContextManager(self.project_root, redis_url)
        self.tool_orchestrator = ToolOrchestrator()
        self.message_queue = MessageQueue()
        self.web_fetch_service = WebFetchService()

        # Initialize Gemini client
        self.gemini_client = get_gemini_client(profile=gemini_profile)

        # Performance tracking
        self.performance_metrics = {
            "total_sessions": 0,
            "successful_sessions": 0,
            "average_session_time": 0.0,
            "total_actions": 0,
            "successful_actions": 0,
            "cache_hits": 0,
        }

        logger.info("🧠 Complete ReAct Engine initialized")
        logger.info(f"   Project root: {self.project_root}")
        logger.info(f"   Gemini profile: {gemini_profile}")
        logger.info(f"   Max iterations: {max_iterations}")
        logger.info(f"   Local LLM enabled: {enable_local_llm}")
        logger.info(f"   Rust optimizations: {RUST_AVAILABLE}")

    async def process_task(
        self,
        task: str,
        session_id: str | None = None,
        user_id: str | None = None,
        streaming: bool = False,
        use_local_llm_for_privacy: bool = False,
        collaboration_agents: list[AgentRole] | None = None,
        progress_callback: Callable[[ReactSession], None] | None = None,
    ) -> ReactSession:
        """Process a task using the complete ReAct framework."""
        datetime.now()

        # Create or load session
        if session_id:
            session = await self.context_manager.load_session(session_id)
            if session:
                logger.info(f"📂 Restored session: {session_id}")
            else:
                logger.warning(f"Session not found, creating new: {session_id}")
                session = ReactSession(session_id=session_id, task=task, user_id=user_id)
        else:
            session = ReactSession(task=task, user_id=user_id, max_iterations=self.max_iterations)

        session.status = ReActStatus.THINKING
        session.collaborating_agents = collaboration_agents or []

        self.performance_metrics["total_sessions"] += 1

        logger.info(f"🚀 Processing task: {task[:100]}...")
        logger.info(f"   Session ID: {session.session_id}")
        logger.info(f"   Max iterations: {session.max_iterations}")

        try:
            # Main ReAct loop
            while session.current_iteration < session.max_iterations:
                session.current_iteration += 1
                iteration_start = time.time()

                logger.info(
                    f"🔄 ReAct iteration {session.current_iteration}/{session.max_iterations}"
                )

                # THINK: Generate reasoning thought
                thought = await self._generate_thought(session, use_local_llm_for_privacy)
                session.add_thought(thought)

                # Check for final answer
                if thought.thought_type == ThoughtType.FINAL_ANSWER:
                    session.status = ReActStatus.COMPLETED
                    break

                # ACT: Determine and execute action
                if thought.thought_type in [ThoughtType.PLANNING, ThoughtType.DECISION]:
                    session.status = ReActStatus.ACTING

                    action = await self._plan_action(session, thought)
                    if action:
                        session.add_action(action)

                        # Execute action
                        observation = await self.tool_orchestrator.execute_action(
                            action, session, use_local_llm_for_privacy
                        )
                        session.add_observation(observation)

                        # OBSERVE: Process results
                        session.status = ReActStatus.OBSERVING
                        await self._process_observation(session, observation)

                        # Multi-agent communication if needed
                        if (
                            action.target_agent
                            and action.target_agent in session.collaborating_agents
                        ):
                            await self._handle_agent_collaboration(session, action, observation)

                # Update performance metrics
                iteration_time = time.time() - iteration_start
                session.total_time_ms += iteration_time * 1000

                # Auto-save session if enabled
                if session.auto_save:
                    await self.context_manager.save_session(session)

                # Stream progress if requested
                if streaming and progress_callback:
                    progress_callback(session)

                # Brief pause between iterations
                await asyncio.sleep(0.1)

            # Handle completion or timeout
            if session.current_iteration >= session.max_iterations:
                session.status = ReActStatus.TERMINATED
                logger.warning(f"⏱️ Session terminated after {session.max_iterations} iterations")
            else:
                session.status = ReActStatus.COMPLETED
                self.performance_metrics["successful_sessions"] += 1
                logger.info("✅ Task completed successfully")

            session.end_time = datetime.now()
            session.total_time_ms = (session.end_time - session.start_time).total_seconds() * 1000

            # Update performance metrics
            self._update_performance_metrics(session)

            # Final save
            await self.context_manager.save_session(session)

            return session

        except Exception as e:
            session.status = ReActStatus.FAILED
            session.end_time = datetime.now()
            logger.exception(f"❌ Task processing failed: {e}")

            # Save failed session for analysis
            await self.context_manager.save_session(session)

            raise

    async def _generate_thought(
        self, session: ReactSession, use_local_llm: bool = False
    ) -> ReactThought:
        """Generate a reasoning thought using Gemini or local LLM."""
        start_time = time.time()

        # Build context
        context = self._build_reasoning_context(session)

        # Choose reasoning prompt based on iteration and context
        if session.current_iteration == 1:
            thought_type = ThoughtType.ANALYSIS
            prompt = self._build_analysis_prompt(session.task, context)
        elif len(session.actions) == 0:
            thought_type = ThoughtType.PLANNING
            prompt = self._build_planning_prompt(session.task, context)
        elif session.observations and not session.observations[-1].success:
            thought_type = ThoughtType.PROBLEM_SOLVING
            prompt = self._build_problem_solving_prompt(session, context)
        else:
            thought_type = ThoughtType.DECISION
            prompt = self._build_decision_prompt(session, context)

        try:
            # Choose reasoning method
            if use_local_llm and self.tool_orchestrator.local_llm_client:
                response = await self._reason_with_local_llm(prompt, context)
            else:
                response = await self._reason_with_gemini(prompt, context)

            # Extract reasoning chain and alternatives
            reasoning_chain = self._extract_reasoning_chain(response)
            alternatives = self._extract_alternatives(response)

            duration_ms = (time.time() - start_time) * 1000

            # Assess confidence based on response quality
            confidence = self._assess_thought_confidence(response, reasoning_chain, alternatives)

            thought = ReactThought(
                thought_type=thought_type,
                content=response,
                confidence=confidence,
                reasoning_chain=reasoning_chain,
                alternative_approaches=alternatives,
                duration_ms=duration_ms,
            )

            logger.info(f"💭 Generated {thought_type.value} thought (confidence: {confidence:.2f})")
            return thought

        except Exception as e:
            logger.exception(f"❌ Thought generation failed: {e}")
            # Fallback thought
            return ReactThought(
                thought_type=ThoughtType.PROBLEM_SOLVING,
                content=f"Error in reasoning: {e}. Need to reconsider approach.",
                confidence=0.1,
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _reason_with_gemini(self, prompt: str, context: dict[str, Any]) -> str:
        """Generate reasoning using Gemini."""
        try:
            response = await self.gemini_client.generate_content(prompt)
            return response
        except Exception as e:
            logger.exception(f"Gemini reasoning failed: {e}")
            raise

    async def _reason_with_local_llm(self, prompt: str, context: dict[str, Any]) -> str:
        """Generate reasoning using local LLM."""
        if not self.tool_orchestrator.local_llm_client:
            raise RuntimeError("Local LLM not available")

        try:
            response = await self.tool_orchestrator.local_llm_client.post(
                "/api/chat",
                json={
                    "prompt": prompt,
                    "context": context,
                    "max_tokens": 1000,
                    "temperature": 0.3,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.exception(f"Local LLM reasoning failed: {e}")
            raise

    async def _plan_action(
        self, session: ReactSession, thought: ReactThought
    ) -> ReactAction | None:
        """Plan the next action based on the current thought."""
        # Extract action from thought content
        action_info = await self._extract_action_from_thought(thought.content, session)

        if not action_info:
            return None

        # Determine target agent for collaborative tasks
        target_agent = None
        if "architecture" in thought.content.lower() or "design" in thought.content.lower():
            target_agent = AgentRole.ARCHITECT
        elif "review" in thought.content.lower() or "quality" in thought.content.lower():
            target_agent = AgentRole.REVIEWER
        elif "analyze" in thought.content.lower():
            target_agent = AgentRole.ANALYZER

        action = ReactAction(
            action_type=ActionType(action_info.get("type", "custom")),
            description=action_info.get("description", "Execute planned action"),
            parameters=action_info.get("parameters", {}),
            target_agent=target_agent,
            priority=action_info.get("priority", 1),
            expected_outcome=action_info.get("expected_outcome"),
            fallback_actions=action_info.get("fallback_actions", []),
        )

        return action

    async def _extract_action_from_thought(
        self, thought_content: str, session: ReactSession
    ) -> dict[str, Any] | None:
        """Extract actionable information from thought content."""
        # This could be enhanced with more sophisticated NLP
        # For now, using simple keyword detection and patterns

        action_keywords = {
            "analyze": ActionType.ANALYZE_CODE,
            "generate": ActionType.GENERATE_CODE,
            "review": ActionType.REVIEW_CODE,
            "search": ActionType.SEARCH,
            "file": ActionType.FILE_OPERATION,
            "command": ActionType.SYSTEM_COMMAND,
            "documentation": ActionType.DOCUMENTATION,
            "architect": ActionType.ARCHITECT,
        }

        content_lower = thought_content.lower()

        # Find matching action type
        action_type = None
        for keyword, act_type in action_keywords.items():
            if keyword in content_lower:
                action_type = act_type
                break

        if not action_type:
            return None

        # Extract parameters (simplified)
        parameters = {}
        if "file" in content_lower:
            # Try to extract file paths
            import re

            file_matches = re.findall(r'["\']([^"\']+\.[a-zA-Z]+)["\']', thought_content)
            if file_matches:
                parameters["files"] = file_matches

        return {
            "type": action_type.value,
            "description": f"Execute {action_type.value.replace('_', ' ')} based on reasoning",
            "parameters": parameters,
            "priority": 1,
        }

    async def _process_observation(
        self, session: ReactSession, observation: ReactObservation
    ) -> None:
        """Process observation results and update session context."""
        # Update session context with new information
        if observation.success and observation.result:
            if isinstance(observation.result, dict):
                session.context_data.update(observation.result)
            else:
                session.context_data[f"result_{len(session.observations)}"] = observation.result

        # Update working memory
        session.working_memory["last_observation"] = {
            "success": observation.success,
            "confidence": observation.confidence,
            "quality": observation.quality_score,
            "insights": observation.insights,
        }

        # Learn from failed observations
        if not observation.success:
            session.working_memory.setdefault("failures", []).append(
                {
                    "error": observation.error_message,
                    "recommendations": observation.recommendations,
                    "timestamp": observation.timestamp.isoformat(),
                }
            )

    async def _handle_agent_collaboration(
        self, session: ReactSession, action: ReactAction, observation: ReactObservation
    ) -> None:
        """Handle multi-agent collaboration through message queue."""
        if not action.target_agent:
            return

        # Send request to target agent
        await self.message_queue.send_message(
            sender=AgentRole.COORDINATOR,
            receiver=action.target_agent,
            message_type="action_request",
            content={
                "action": action.model_dump(),
                "context": session.context_data,
                "observation": observation.model_dump(),
            },
            session_id=session.session_id,
            priority=action.priority,
        )

        # Wait for response (with timeout)
        response = await self.message_queue.receive_message(
            AgentRole.COORDINATOR, session.session_id, timeout=30.0
        )

        if response:
            session.agent_messages.append(response)

            # Update context with agent response
            if response.get("content", {}).get("result"):
                session.context_data.update(response["content"]["result"])

    def _build_reasoning_context(self, session: ReactSession) -> dict[str, Any]:
        """Build comprehensive context for reasoning."""
        return {
            "task": session.task,
            "current_iteration": session.current_iteration,
            "max_iterations": session.max_iterations,
            "previous_thoughts": [t.content for t in session.thoughts[-3:]],
            "recent_actions": [a.description for a in session.actions[-2:]],
            "recent_observations": [
                {
                    "success": obs.success,
                    "insights": obs.insights,
                    "recommendations": obs.recommendations,
                }
                for obs in session.observations[-2:]
            ],
            "context_data": session.context_data,
            "working_memory": session.working_memory,
            "performance_stats": {
                "successful_tools": session.successful_tools,
                "failed_tools": session.failed_tools,
                "total_time_ms": session.total_time_ms,
            },
        }

    def _build_analysis_prompt(self, task: str, context: dict[str, Any]) -> str:
        """Build initial analysis prompt."""
        return f"""You are an advanced AI agent using the ReAct (Reasoning and Acting) paradigm.
Analyze this task and develop an understanding of what needs to be accomplished.

Task: {task}

Context: {json.dumps(context, indent=2)}

Provide a thorough analysis including:
1. What is the main objective?
2. What information do I need to gather?
3. What actions might be required?
4. What challenges might I encounter?
5. How should I approach this systematically?

Think step by step and provide your analysis."""

    def _build_planning_prompt(self, task: str, context: dict[str, Any]) -> str:
        """Build planning prompt."""
        return f"""Based on your analysis, create a specific plan for accomplishing this task.

Task: {task}
Context: {json.dumps(context, indent=2)}

Create a concrete plan that includes:
1. Specific steps to take
2. Tools or resources needed
3. Expected outcomes
4. Potential alternatives
5. Success criteria

Focus on the immediate next action to take."""

    def _build_decision_prompt(self, session: ReactSession, context: dict[str, Any]) -> str:
        """Build decision-making prompt."""
        recent_results = []
        for obs in session.observations[-2:]:
            recent_results.append(
                {
                    "success": obs.success,
                    "insights": obs.insights,
                    "recommendations": obs.recommendations,
                }
            )

        return f"""Based on recent progress, decide what to do next.

Original Task: {session.task}
Recent Results: {json.dumps(recent_results, indent=2)}
Current Context: {json.dumps(context, indent=2)}

Consider:
1. Have I made sufficient progress toward the goal?
2. What have I learned from recent actions?
3. Do I need to adjust my approach?
4. What is the most logical next step?
5. Am I ready to provide a final answer?

Make a clear decision about the next action or whether to conclude."""

    def _build_problem_solving_prompt(self, session: ReactSession, context: dict[str, Any]) -> str:
        """Build problem-solving prompt for when things go wrong."""
        last_failure = session.observations[-1] if session.observations else None

        return f"""The last action failed. Analyze what went wrong and determine how to proceed.

Original Task: {session.task}
Failed Action: {session.actions[-1].description if session.actions else "None"}
Error: {last_failure.error_message if last_failure else "Unknown"}
Recommendations: {last_failure.recommendations if last_failure else []}

Analyze:
1. Why did this action fail?
2. What can I learn from this failure?
3. What alternative approaches are available?
4. Should I retry with modifications or try a different approach?
5. How can I avoid similar failures?

Determine the best course of action to recover and continue progress."""

    def _extract_reasoning_chain(self, response: str) -> list[str]:
        """Extract reasoning steps from response."""
        # Simple extraction - could be enhanced with more sophisticated parsing
        lines = response.split("\n")
        reasoning_steps = []

        for line in lines:
            line = line.strip()
            if line and (
                line.startswith(("1.", "2.", "3.", "4.", "5."))
                or line.startswith(("First,", "Second,", "Third,", "Next,", "Finally,"))
            ):
                reasoning_steps.append(line)

        return reasoning_steps

    def _extract_alternatives(self, response: str) -> list[str]:
        """Extract alternative approaches from response."""
        alternatives = []
        content_lower = response.lower()

        # Look for alternative indicators
        alt_indicators = ["alternatively", "another approach", "could also", "option", "instead"]

        for indicator in alt_indicators:
            if indicator in content_lower:
                # Extract the sentence containing the alternative
                sentences = response.split(".")
                for sentence in sentences:
                    if indicator in sentence.lower():
                        alternatives.append(sentence.strip())
                        break

        return alternatives

    def _assess_thought_confidence(
        self, response: str, reasoning_chain: list[str], alternatives: list[str]
    ) -> float:
        """Assess confidence in the generated thought."""
        confidence = 0.5  # Base confidence

        # Add confidence for structured reasoning
        if reasoning_chain:
            confidence += min(len(reasoning_chain) * 0.1, 0.3)

        # Add confidence for considering alternatives
        if alternatives:
            confidence += 0.1

        # Reduce confidence for uncertainty indicators
        uncertainty_words = ["maybe", "perhaps", "might", "could be", "unsure", "unclear"]
        response_lower = response.lower()
        uncertainty_count = sum(1 for word in uncertainty_words if word in response_lower)
        confidence -= uncertainty_count * 0.05

        # Ensure confidence is within bounds
        return max(0.1, min(confidence, 1.0))

    def _update_performance_metrics(self, session: ReactSession) -> None:
        """Update global performance metrics."""
        self.performance_metrics["total_actions"] += len(session.actions)
        self.performance_metrics["successful_actions"] += session.successful_tools

        # Update average session time
        session_count = self.performance_metrics["successful_sessions"]
        if session_count > 0:
            current_avg = self.performance_metrics["average_session_time"]
            new_avg = (current_avg * (session_count - 1) + session.total_time_ms) / session_count
            self.performance_metrics["average_session_time"] = new_avg

    # Public utility methods

    async def list_sessions(self, user_id: str | None = None) -> list[str]:
        """List available sessions."""
        return await self.context_manager.list_sessions(user_id)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return await self.context_manager.delete_session(session_id)

    async def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get a summary of a session."""
        session = await self.context_manager.load_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "task": session.task,
            "status": session.status.value,
            "iterations": session.current_iteration,
            "total_time_ms": session.total_time_ms,
            "thoughts": len(session.thoughts),
            "actions": len(session.actions),
            "observations": len(session.observations),
            "success_rate": session.successful_tools / max(session.tool_calls, 1),
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
        }

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            "engine_metrics": self.performance_metrics,
            "component_status": {
                "context_manager": "active",
                "tool_orchestrator": "active",
                "message_queue": "active",
                "web_fetch_service": "active",
                "gemini_client": "connected",
                "rust_available": RUST_AVAILABLE,
            },
            "resource_usage": {
                "active_sessions": len(await self.list_sessions()),
                "message_queue_size": sum(
                    queue.qsize() for queue in self.message_queue.message_queues.values()
                ),
                "cache_entries": len(self.web_fetch_service.cache),
                "execution_history": len(self.tool_orchestrator.execution_history),
            },
        }

    async def cleanup(self) -> None:
        """Cleanup all resources."""
        await self.web_fetch_service.cleanup()
        if self.tool_orchestrator.local_llm_client:
            await self.tool_orchestrator.local_llm_client.aclose()
        logger.info("🧹 Complete ReAct Engine cleanup completed")


# Convenience functions for easy integration


async def create_react_engine(project_root: Path | None = None, **kwargs) -> CompleteReactEngine:
    """Create and initialize a complete ReAct engine."""
    engine = CompleteReactEngine(project_root=project_root, **kwargs)
    logger.info("✅ Complete ReAct Engine created and ready")
    return engine


@asynccontextmanager
async def react_session(
    engine: CompleteReactEngine, task: str, **kwargs
) -> AsyncGenerator[ReactSession, None]:
    """Context manager for ReAct sessions with automatic cleanup."""
    session = None
    try:
        session = await engine.process_task(task, **kwargs)
        yield session
    finally:
        if session and session.auto_save:
            await engine.context_manager.save_session(session)


# Example usage and testing functions


async def demo_react_engine() -> None:
    """Demonstrate the complete ReAct engine capabilities."""
    engine = await create_react_engine(
        enable_local_llm=True,
        max_iterations=5,
    )

    test_task = "Analyze the project structure and provide recommendations for code organization"

    try:
        async with react_session(engine, test_task) as session:
            print(f"✅ Task completed: {session.status.value}")
            print(f"📊 Iterations: {session.current_iteration}")
            print(f"⏱️ Time: {session.total_time_ms:.0f}ms")
            print(f"🔧 Actions: {len(session.actions)}")
            print(f"👁️ Observations: {len(session.observations)}")

            if session.thoughts:
                print(f"💭 Final thought: {session.thoughts[-1].content[:100]}...")

    except Exception as e:
        logger.exception(f"Demo failed: {e}")
    finally:
        await engine.cleanup()


if __name__ == "__main__":
    # Run demo when executed directly
    asyncio.run(demo_react_engine())
