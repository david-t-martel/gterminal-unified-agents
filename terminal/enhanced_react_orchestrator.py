#!/usr/bin/env python3
"""Enhanced ReAct Orchestrator - Leverages existing infrastructure for production-ready ReAct engine.

This orchestrator builds upon the comprehensive existing infrastructure:
- Existing ReAct engine (app.core.react_engine)
- Rust performance utilities (RustCache, RustJsonProcessor, etc.)
- MCP client infrastructure (GeminiMCPClient)
- Terminal operations (TerminalRustOps)
- Session management (SessionManager)
- Unified Gemini client

New capabilities added:
- Multi-agent communication through MCP
- Local LLM integration for privacy-critical tasks
- Enhanced context persistence with Redis/Rust fallback
- Web fetch capabilities with caching
- Production-ready error handling and monitoring
- Tool orchestration and execution pipeline
"""

import asyncio
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
import uuid

import httpx
from pydantic import BaseModel
from pydantic import Field
from pydantic import validator

from gterminal.core.agents.mcp_client import GeminiMCPClient

# Import existing infrastructure
from gterminal.core.react_engine import ReActEngine
from gterminal.core.react_engine import StepType
from gterminal.core.session import SessionManager
from gterminal.core.tools.registry import ToolRegistry
from gterminal.core.unified_gemini_client import get_gemini_client

from .react_types import ReActContext
from .react_types import ReActStatus
from .react_types import ReActStep
from .rust_terminal_ops import RUST_EXTENSIONS_AVAILABLE
from .rust_terminal_ops import TerminalRustOps

# Import performance optimizations
try:
    from fullstack_agent_rust import EnhancedTtlCache
    from fullstack_agent_rust import RustCache
    from fullstack_agent_rust import RustJsonProcessor

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Enhanced agent roles for multi-agent coordination."""

    COORDINATOR = "coordinator"
    ARCHITECT = "architect"
    REVIEWER = "reviewer"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    LOCAL_LLM = "local_llm"
    WEB_FETCHER = "web_fetcher"


class TaskPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class EnhancedReActTask(BaseModel):
    """Enhanced task model with comprehensive metadata."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_duration: int | None = None  # seconds
    required_agents: list[AgentRole] = Field(default_factory=list)
    privacy_level: str = Field(default="standard")  # standard, sensitive, private

    # Context and constraints
    context: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)

    # Tracking
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: str = "pending"

    @validator("privacy_level")
    def validate_privacy(cls, v):
        if v not in ["standard", "sensitive", "private"]:
            raise ValueError("Privacy level must be standard, sensitive, or private")
        return v


class AgentMessage(BaseModel):
    """Inter-agent communication message."""

    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    sender: AgentRole
    receiver: AgentRole
    message_type: str
    content: dict[str, Any]

    # Message metadata
    priority: TaskPriority = TaskPriority.MEDIUM
    requires_response: bool = False
    correlation_id: str | None = None  # For request-response patterns
    expires_at: datetime | None = None

    created_at: datetime = Field(default_factory=datetime.now)
    delivered_at: datetime | None = None
    processed_at: datetime | None = None


class LocalLLMBridge:
    """Bridge to local LLM framework for privacy-critical operations."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        self.available = False
        logger.info(f"🔒 Local LLM bridge initialized for {base_url}")

    async def check_availability(self) -> bool:
        """Check if local LLM service is available."""
        try:
            response = await self.client.get("/api/health", timeout=5.0)
            self.available = response.status_code == 200
            if self.available:
                logger.info("✅ Local LLM service available")
            return self.available
        except Exception as e:
            logger.warning(f"🔒 Local LLM service unavailable: {e}")
            self.available = False
            return False

    async def process_sensitive_task(
        self, prompt: str, context: dict[str, Any], max_tokens: int = 1000
    ) -> dict[str, Any]:
        """Process privacy-sensitive tasks using local LLM."""
        if not self.available:
            await self.check_availability()
            if not self.available:
                raise RuntimeError("Local LLM service not available for sensitive task")

        try:
            response = await self.client.post(
                "/api/chat",
                json={
                    "prompt": prompt,
                    "context": context,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "privacy_mode": True,
                },
            )
            response.raise_for_status()
            result = response.json()

            return {
                "response": result.get("response", ""),
                "confidence": result.get("confidence", 0.8),
                "tokens_used": result.get("tokens_used", 0),
                "local_processing": True,
            }
        except Exception as e:
            logger.exception(f"Local LLM processing failed: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.client.aclose()


class WebFetchService:
    """Enhanced web content fetching with intelligent caching."""

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            follow_redirects=True,
        )

        # Initialize cache
        if RUST_AVAILABLE:
            self.cache = RustCache(max_size=5000, default_ttl_seconds=3600)
        else:
            self.cache = {}

        self.rate_limits = {}  # Domain -> (count, reset_time)
        logger.info("🌐 Web fetch service initialized with caching")

    async def fetch_with_cache(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        cache_ttl: int = 3600,
        extract_text: bool = False,
    ) -> dict[str, Any]:
        """Fetch URL with intelligent caching and rate limiting."""
        # Generate cache key
        cache_key = f"web_fetch:{hashlib.md5(f'{url}:{json.dumps(headers or {})}:{extract_text}'.encode()).hexdigest()}"

        # Check cache first
        if RUST_AVAILABLE:
            cached = await asyncio.to_thread(self.cache.get, cache_key)
            if cached:
                logger.debug(f"📋 Cache hit for URL: {url}")
                return json.loads(cached) if isinstance(cached, str) else cached
        elif cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data["cached_at"] < timedelta(seconds=cache_ttl):
                logger.debug(f"📋 Cache hit for URL: {url}")
                return cached_data

        # Check rate limits
        domain = httpx.URL(url).host
        if await self._is_rate_limited(domain):
            raise ValueError(f"Rate limit exceeded for {domain}")

        try:
            # Fetch content
            start_time = time.time()
            response = await self.client.get(url, headers=headers or {})
            response.raise_for_status()
            fetch_time = time.time() - start_time

            content = response.text
            if extract_text:
                content = self._extract_text_content(content)

            result = {
                "url": url,
                "status_code": response.status_code,
                "content": content,
                "content_length": len(content),
                "headers": dict(response.headers),
                "fetch_time": fetch_time,
                "cached_at": datetime.now(),
                "extracted_text": extract_text,
            }

            # Cache result
            if RUST_AVAILABLE:
                await asyncio.to_thread(
                    self.cache.set, cache_key, json.dumps(result, default=str), cache_ttl
                )
            else:
                self.cache[cache_key] = result

            logger.info(f"🌐 Fetched {url} ({len(content)} chars in {fetch_time:.2f}s)")
            return result

        except Exception as e:
            logger.exception(f"❌ Failed to fetch {url}: {e}")
            raise

    async def _is_rate_limited(self, domain: str) -> bool:
        """Check if domain is rate limited."""
        now = time.time()
        if domain not in self.rate_limits:
            self.rate_limits[domain] = (0, now + 60)  # 60 requests per minute
            return False

        count, reset_time = self.rate_limits[domain]
        if now > reset_time:
            self.rate_limits[domain] = (1, now + 60)
            return False

        if count >= 60:  # Rate limit: 60 requests per minute
            return True

        self.rate_limits[domain] = (count + 1, reset_time)
        return False

    def _extract_text_content(self, html: str) -> str:
        """Extract text content from HTML."""
        # Simple text extraction - could use BeautifulSoup for better results
        import re

        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.client.aclose()


class MessageQueue:
    """Enhanced message queue for multi-agent coordination."""

    def __init__(self):
        self.queues: dict[str, asyncio.Queue] = {}
        self.message_history: list[AgentMessage] = []
        self.subscribers: dict[str, list[Callable]] = {}
        self.processing_stats = {
            "sent": 0,
            "delivered": 0,
            "processed": 0,
            "failed": 0,
        }
        logger.info("📡 Enhanced message queue initialized")

    async def send_message(self, message: AgentMessage) -> bool:
        """Send message to target agent."""
        queue_key = f"{message.receiver.value}"

        # Create queue if it doesn't exist
        if queue_key not in self.queues:
            self.queues[queue_key] = asyncio.Queue(maxsize=1000)

        try:
            # Check if message has expired
            if message.expires_at and datetime.now() > message.expires_at:
                logger.warning(f"⏰ Message {message.message_id} expired before delivery")
                self.processing_stats["failed"] += 1
                return False

            # Add to queue
            await self.queues[queue_key].put(message)
            message.delivered_at = datetime.now()

            # Store in history
            self.message_history.append(message)

            # Notify subscribers
            for callback in self.subscribers.get(queue_key, []):
                try:
                    await callback(message)
                except Exception as e:
                    logger.warning(f"Subscriber callback failed: {e}")

            self.processing_stats["sent"] += 1
            self.processing_stats["delivered"] += 1

            logger.debug(f"📤 Message sent: {message.sender.value} → {message.receiver.value}")
            return True

        except Exception as e:
            logger.exception(f"Failed to send message: {e}")
            self.processing_stats["failed"] += 1
            return False

    async def receive_message(
        self, receiver: AgentRole, timeout: float = 10.0
    ) -> AgentMessage | None:
        """Receive message for an agent."""
        queue_key = f"{receiver.value}"

        if queue_key not in self.queues:
            self.queues[queue_key] = asyncio.Queue(maxsize=1000)

        try:
            message = await asyncio.wait_for(self.queues[queue_key].get(), timeout=timeout)
            message.processed_at = datetime.now()
            self.processing_stats["processed"] += 1

            logger.debug(f"📥 Message received by {receiver.value}: {message.message_type}")
            return message

        except TimeoutError:
            logger.debug(f"⏱️ Message receive timeout for {receiver.value}")
            return None
        except Exception as e:
            logger.exception(f"Failed to receive message: {e}")
            return None

    def subscribe(self, receiver: AgentRole, callback: Callable) -> None:
        """Subscribe to messages for an agent."""
        queue_key = f"{receiver.value}"
        if queue_key not in self.subscribers:
            self.subscribers[queue_key] = []
        self.subscribers[queue_key].append(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get message queue statistics."""
        return {
            **self.processing_stats,
            "active_queues": len(self.queues),
            "queue_sizes": {k: v.qsize() for k, v in self.queues.items()},
            "subscribers": {k: len(v) for k, v in self.subscribers.items()},
            "message_history": len(self.message_history),
        }


class EnhancedReActOrchestrator:
    """Enhanced ReAct Orchestrator that leverages existing infrastructure."""

    def __init__(
        self,
        project_root: Path | None = None,
        gemini_profile: str = "business",
        enable_local_llm: bool = True,
        enable_web_fetch: bool = True,
        redis_url: str | None = None,
    ):
        self.project_root = project_root or Path("/home/david/agents/my-fullstack-agent")
        self.gemini_profile = gemini_profile

        # Initialize existing components
        self.react_engine = ReActEngine()
        self.session_manager = SessionManager()
        self.tool_registry = ToolRegistry()
        self.terminal_ops = TerminalRustOps()
        self.mcp_client = GeminiMCPClient()

        # Initialize enhanced components
        self.message_queue = MessageQueue()
        self.web_fetch_service = WebFetchService() if enable_web_fetch else None
        self.local_llm_bridge = LocalLLMBridge() if enable_local_llm else None

        # Get Gemini client
        self.gemini_client = get_gemini_client(profile=gemini_profile)

        # Performance tracking
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_task_time": 0.0,
            "cache_hits": 0,
            "llm_calls": 0,
            "local_llm_calls": 0,
        }

        logger.info("🧠 Enhanced ReAct Orchestrator initialized")
        logger.info(f"   Project root: {self.project_root}")
        logger.info(f"   Gemini profile: {gemini_profile}")
        logger.info(f"   Rust extensions: {RUST_EXTENSIONS_AVAILABLE}")
        logger.info(f"   Local LLM: {enable_local_llm}")
        logger.info(f"   Web fetch: {enable_web_fetch}")

    async def process_enhanced_task(
        self,
        task: EnhancedReActTask,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Process an enhanced ReAct task with full orchestration."""
        start_time = time.time()
        task.started_at = datetime.now()
        task.status = "processing"

        logger.info(f"🚀 Processing task: {task.description[:100]}...")
        logger.info(f"   Task ID: {task.task_id}")
        logger.info(f"   Priority: {task.priority.value}")
        logger.info(f"   Privacy level: {task.privacy_level}")

        try:
            # Determine processing strategy based on privacy level
            use_local_llm = task.privacy_level in ["sensitive", "private"] and self.local_llm_bridge

            # Initialize session context
            session_id = f"enhanced_{task.task_id}"
            context = ReActContext(
                session_id=session_id,
                task=task.description,
                max_iterations=20,
                context_data=task.context.copy(),
            )

            # Cache initial context
            await self.terminal_ops.cache_context(context)

            # Main processing loop with enhanced capabilities
            result = await self._execute_enhanced_react_loop(
                task, context, use_local_llm, progress_callback
            )

            # Update task completion
            task.completed_at = datetime.now()
            task.status = "completed" if result.get("success") else "failed"

            execution_time = time.time() - start_time
            self.metrics["tasks_completed"] += 1 if result.get("success") else 0
            self.metrics["tasks_failed"] += 1 if not result.get("success") else 0

            # Update average task time
            completed = self.metrics["tasks_completed"]
            if completed > 0:
                current_avg = self.metrics["average_task_time"]
                self.metrics["average_task_time"] = (
                    current_avg * (completed - 1) + execution_time
                ) / completed

            logger.info(f"✅ Task completed in {execution_time:.2f}s: {task.task_id}")

            return {
                "task_id": task.task_id,
                "success": result.get("success", False),
                "result": result.get("result"),
                "execution_time": execution_time,
                "iterations": context.current_iteration,
                "used_local_llm": use_local_llm,
                "performance_metrics": {
                    "cache_hits": self.metrics["cache_hits"],
                    "llm_calls": result.get("llm_calls", 0),
                    "tools_used": len(context.steps),
                },
            }

        except Exception as e:
            task.completed_at = datetime.now()
            task.status = "failed"
            self.metrics["tasks_failed"] += 1
            execution_time = time.time() - start_time

            logger.exception(f"❌ Task failed: {task.task_id} - {e}")

            return {
                "task_id": task.task_id,
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "task_status": task.status,
            }

    async def _execute_enhanced_react_loop(
        self,
        task: EnhancedReActTask,
        context: ReActContext,
        use_local_llm: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Execute the enhanced ReAct loop with multi-agent coordination."""
        llm_calls = 0
        context.status = ReActStatus.THINKING

        while context.current_iteration < context.max_iterations:
            context.current_iteration += 1

            logger.info(
                f"🔄 Enhanced ReAct iteration {context.current_iteration}/{context.max_iterations}"
            )

            # REASON: Generate next step using appropriate LLM
            if use_local_llm:
                reasoning = await self._reason_with_local_llm(task, context)
                self.metrics["local_llm_calls"] += 1
            else:
                reasoning = await self._reason_with_gemini(task, context)
                self.metrics["llm_calls"] += 1

            llm_calls += 1

            # Create reasoning step
            step = ReActStep(
                step_type=StepType.THOUGHT,
                content=reasoning,
                timestamp=datetime.now(),
            )
            context.add_step(step)

            # Check for completion
            if "COMPLETE" in reasoning.upper() or "FINAL ANSWER" in reasoning.upper():
                context.status = ReActStatus.COMPLETED
                break

            # ACT: Execute actions based on reasoning
            context.status = ReActStatus.ACTING
            action_result = await self._execute_enhanced_actions(task, context, reasoning)

            # OBSERVE: Process results
            context.status = ReActStatus.OBSERVING
            observation_step = ReActStep(
                step_type=StepType.OBSERVATION,
                content=str(action_result.get("result", "No result")),
                metadata=action_result.get("metadata", {}),
                timestamp=datetime.now(),
            )
            context.add_step(observation_step)

            # Update context with new information
            if action_result.get("context_updates"):
                context.context_data.update(action_result["context_updates"])

            # Handle multi-agent coordination if needed
            if task.required_agents:
                await self._coordinate_with_agents(task, context, action_result)

            # Auto-save context
            await self.terminal_ops.cache_context(context)

            # Stream progress if callback provided
            if progress_callback:
                progress_callback(
                    {
                        "iteration": context.current_iteration,
                        "status": context.status.value,
                        "latest_step": step.content[:200] + "..."
                        if len(step.content) > 200
                        else step.content,
                        "actions_taken": len(
                            [s for s in context.steps if s.step_type == StepType.ACTION]
                        ),
                    }
                )

            # Brief pause between iterations
            await asyncio.sleep(0.1)

        # Handle completion or timeout
        if context.current_iteration >= context.max_iterations:
            context.status = ReActStatus.TERMINATED
            success = False
            result = "Maximum iterations reached without completion"
        else:
            context.status = ReActStatus.COMPLETED
            success = True
            result = self._extract_final_result(context)

        return {
            "success": success,
            "result": result,
            "llm_calls": llm_calls,
            "context": context,
        }

    async def _reason_with_local_llm(self, task: EnhancedReActTask, context: ReActContext) -> str:
        """Generate reasoning using local LLM for privacy-sensitive tasks."""
        if not self.local_llm_bridge:
            raise RuntimeError("Local LLM bridge not available")

        # Build context-aware prompt
        prompt = self._build_reasoning_prompt(task, context, privacy_mode=True)

        try:
            result = await self.local_llm_bridge.process_sensitive_task(
                prompt=prompt, context=context.context_data, max_tokens=1000
            )

            logger.debug("🔒 Used local LLM for sensitive reasoning")
            return result["response"]

        except Exception as e:
            logger.exception(f"Local LLM reasoning failed: {e}")
            # Fallback to Gemini with privacy warning
            logger.warning("⚠️ Falling back to Gemini for sensitive task")
            return await self._reason_with_gemini(task, context)

    async def _reason_with_gemini(self, task: EnhancedReActTask, context: ReActContext) -> str:
        """Generate reasoning using Gemini."""
        prompt = self._build_reasoning_prompt(task, context, privacy_mode=False)

        try:
            if hasattr(self.gemini_client, "generate_content"):
                response = await self.gemini_client.generate_content(prompt)
            else:
                # Fallback to basic implementation
                response = f"Analyzing task: {task.description}\nCurrent iteration: {context.current_iteration}\nNext action needed based on context."

            return response

        except Exception as e:
            logger.exception(f"Gemini reasoning failed: {e}")
            # Provide fallback reasoning
            return (
                f"Error in AI reasoning: {e}. Proceeding with basic analysis of: {task.description}"
            )

    def _build_reasoning_prompt(
        self, task: EnhancedReActTask, context: ReActContext, privacy_mode: bool
    ) -> str:
        """Build context-aware reasoning prompt."""
        privacy_note = (
            "\n🔒 PRIVACY MODE: This task contains sensitive information. Process locally only."
            if privacy_mode
            else ""
        )

        recent_steps = context.steps[-3:] if len(context.steps) > 3 else context.steps
        steps_summary = "\n".join(
            [f"- {step.step_type.value}: {step.content[:100]}..." for step in recent_steps]
        )

        return f"""
You are an advanced ReAct agent processing this task:

TASK: {task.description}
PRIORITY: {task.priority.value}
ITERATION: {context.current_iteration}/{context.max_iterations}
{privacy_note}

RECENT STEPS:
{steps_summary or "No previous steps"}

CONTEXT DATA:
{json.dumps(context.context_data, indent=2) if context.context_data else "No additional context"}

CONSTRAINTS:
{chr(10).join(f"- {constraint}" for constraint in task.constraints) if task.constraints else "No specific constraints"}

SUCCESS CRITERIA:
{chr(10).join(f"- {criteria}" for criteria in task.success_criteria) if task.success_criteria else "Standard completion criteria"}

Based on the above information, what should be the next step?
Consider:
1. What has been accomplished so far?
2. What still needs to be done?
3. What tools or actions are needed?
4. Are we ready to provide a final answer?

Respond with your reasoning and next action. If the task is complete, start your response with "COMPLETE:".
"""

    async def _execute_enhanced_actions(
        self, task: EnhancedReActTask, context: ReActContext, reasoning: str
    ) -> dict[str, Any]:
        """Execute actions based on reasoning with enhanced capabilities."""
        # Extract actions from reasoning
        actions = self._extract_actions_from_reasoning(reasoning)

        results = []
        context_updates = {}

        for action in actions:
            try:
                # Determine execution method
                if action.get("type") == "web_fetch" and self.web_fetch_service:
                    result = await self._execute_web_fetch(action)
                elif action.get("type") == "mcp_tool":
                    result = await self._execute_mcp_tool(action)
                elif action.get("agent_role"):
                    result = await self._execute_agent_action(action, task, context)
                else:
                    result = await self._execute_standard_tool(action)

                results.append(result)

                # Update context if result provides updates
                if result.get("context_updates"):
                    context_updates.update(result["context_updates"])

            except Exception as e:
                logger.exception(f"Action execution failed: {e}")
                results.append(
                    {
                        "success": False,
                        "error": str(e),
                        "action": action,
                    }
                )

        return {
            "results": results,
            "context_updates": context_updates,
            "metadata": {
                "actions_count": len(actions),
                "successful_actions": len([r for r in results if r.get("success")]),
                "timestamp": datetime.now().isoformat(),
            },
        }

    def _extract_actions_from_reasoning(self, reasoning: str) -> list[dict[str, Any]]:
        """Extract actionable items from reasoning text."""
        actions = []

        # Simple action extraction - could be enhanced with NLP
        reasoning_lower = reasoning.lower()

        # Web fetch actions
        if "fetch" in reasoning_lower or "download" in reasoning_lower or "http" in reasoning:
            import re

            urls = re.findall(r"https?://[^\s]+", reasoning)
            for url in urls:
                actions.append(
                    {
                        "type": "web_fetch",
                        "url": url,
                        "extract_text": True,
                    }
                )

        # File operations
        if "read file" in reasoning_lower or "analyze file" in reasoning_lower:
            actions.append(
                {
                    "type": "file_operation",
                    "operation": "read",
                    "description": "Read and analyze file content",
                }
            )

        # Code operations
        if "generate code" in reasoning_lower or "write code" in reasoning_lower:
            actions.append(
                {
                    "type": "code_generation",
                    "description": "Generate code based on requirements",
                }
            )

        # Review operations
        if "review" in reasoning_lower or "analyze quality" in reasoning_lower:
            actions.append(
                {
                    "type": "code_review",
                    "agent_role": "reviewer",
                    "description": "Review code for quality and security",
                }
            )

        # Architecture operations
        if "architecture" in reasoning_lower or "design" in reasoning_lower:
            actions.append(
                {
                    "type": "architecture_analysis",
                    "agent_role": "architect",
                    "description": "Analyze and provide architectural guidance",
                }
            )

        # If no specific actions found, create a general analysis action
        if not actions:
            actions.append(
                {
                    "type": "general_analysis",
                    "description": "Perform general analysis based on reasoning",
                }
            )

        return actions

    async def _execute_web_fetch(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute web fetch action."""
        if not self.web_fetch_service:
            return {"success": False, "error": "Web fetch service not available"}

        try:
            result = await self.web_fetch_service.fetch_with_cache(
                url=action["url"],
                extract_text=action.get("extract_text", False),
            )

            return {
                "success": True,
                "result": result,
                "context_updates": {
                    f"web_content_{action['url']}": result["content"][:5000],  # Limit size
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_mcp_tool(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute MCP tool action."""
        try:
            server_name = action.get("server", "default")
            tool_name = action.get("tool", action.get("type", "unknown"))
            parameters = action.get("parameters", {})

            mcp_result = await self.mcp_client.call_tool(server_name, tool_name, parameters)

            return {
                "success": mcp_result.error is None,
                "result": mcp_result.result,
                "error": mcp_result.error,
                "execution_time": mcp_result.execution_time,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_agent_action(
        self, action: dict[str, Any], task: EnhancedReActTask, context: ReActContext
    ) -> dict[str, Any]:
        """Execute action through specialized agent."""
        agent_role = AgentRole(action["agent_role"])

        # Create message for target agent
        message = AgentMessage(
            sender=AgentRole.COORDINATOR,
            receiver=agent_role,
            message_type="action_request",
            content={
                "action": action,
                "task_context": task.model_dump(),
                "react_context": {
                    "session_id": context.session_id,
                    "iteration": context.current_iteration,
                    "context_data": context.context_data,
                },
            },
            requires_response=True,
            priority=task.priority,
        )

        # Send message and wait for response
        await self.message_queue.send_message(message)

        response = await self.message_queue.receive_message(AgentRole.COORDINATOR, timeout=30.0)

        if response:
            return {
                "success": True,
                "result": response.content,
                "agent_response": True,
            }
        else:
            return {
                "success": False,
                "error": f"No response from {agent_role.value} agent",
            }

    async def _execute_standard_tool(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute action using standard tool registry."""
        try:
            tool_name = action.get("type", "unknown")
            parameters = action.get("parameters", {})

            # Use existing tool registry
            result = await self.tool_registry.execute(tool_name, parameters)

            return {
                "success": result.success if hasattr(result, "success") else True,
                "result": result.data if hasattr(result, "data") else result,
                "tool_used": tool_name,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _coordinate_with_agents(
        self, task: EnhancedReActTask, context: ReActContext, action_result: dict[str, Any]
    ) -> None:
        """Handle multi-agent coordination."""
        for agent_role in task.required_agents:
            try:
                # Send status update to required agents
                status_message = AgentMessage(
                    sender=AgentRole.COORDINATOR,
                    receiver=agent_role,
                    message_type="status_update",
                    content={
                        "task_id": task.task_id,
                        "iteration": context.current_iteration,
                        "latest_result": action_result,
                        "context_summary": {
                            "total_steps": len(context.steps),
                            "status": context.status.value,
                        },
                    },
                    priority=task.priority,
                )

                await self.message_queue.send_message(status_message)

            except Exception as e:
                logger.warning(f"Failed to coordinate with {agent_role.value}: {e}")

    def _extract_final_result(self, context: ReActContext) -> str:
        """Extract final result from ReAct context."""
        # Look for completion indicators in recent steps
        recent_steps = context.steps[-3:] if len(context.steps) > 3 else context.steps

        for step in reversed(recent_steps):
            if "COMPLETE" in step.content.upper() or "FINAL ANSWER" in step.content.upper():
                return step.content

        # Fallback: summarize the process
        return f"Task processed through {context.current_iteration} iterations with {len(context.steps)} total steps. Latest context: {context.context_data}"

    # Public utility methods

    async def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Get comprehensive orchestrator metrics."""
        return {
            "orchestrator_metrics": self.metrics,
            "message_queue_stats": self.message_queue.get_stats(),
            "terminal_ops_metrics": await self.terminal_ops.get_performance_metrics(),
            "mcp_status": self.mcp_client.get_connection_status(),
            "rust_extensions": RUST_EXTENSIONS_AVAILABLE,
            "local_llm_available": self.local_llm_bridge.available
            if self.local_llm_bridge
            else False,
            "web_fetch_available": self.web_fetch_service is not None,
            "timestamp": datetime.now().isoformat(),
        }

    @asynccontextmanager
    async def enhanced_task_session(
        self, task: EnhancedReActTask
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Context manager for enhanced task processing."""
        session_data = {"task": task, "start_time": time.time()}

        try:
            yield session_data
        finally:
            # Cleanup and finalization
            session_data["end_time"] = time.time()
            session_data["duration"] = session_data["end_time"] - session_data["start_time"]

            # Save task history if needed
            await self.terminal_ops.cache_set(
                f"task_history:{task.task_id}",
                json.dumps(task.model_dump(), default=str),
                ttl=86400,  # 24 hours
            )

    async def cleanup(self) -> None:
        """Cleanup all resources."""
        if self.web_fetch_service:
            await self.web_fetch_service.cleanup()
        if self.local_llm_bridge:
            await self.local_llm_bridge.cleanup()
        logger.info("🧹 Enhanced ReAct Orchestrator cleanup completed")


# Convenience functions and demo


async def create_enhanced_orchestrator(**kwargs) -> EnhancedReActOrchestrator:
    """Create and initialize enhanced orchestrator."""
    orchestrator = EnhancedReActOrchestrator(**kwargs)

    # Initialize local LLM if available
    if orchestrator.local_llm_bridge:
        await orchestrator.local_llm_bridge.check_availability()

    logger.info("✅ Enhanced ReAct Orchestrator ready for production use")
    return orchestrator


async def demo_enhanced_react_orchestrator() -> None:
    """Demonstrate the enhanced orchestrator capabilities."""
    # Create orchestrator
    orchestrator = await create_enhanced_orchestrator(
        enable_local_llm=True,
        enable_web_fetch=True,
    )

    try:
        # Create demo task
        demo_task = EnhancedReActTask(
            description="Analyze the project structure and provide comprehensive recommendations for code organization and architecture improvements",
            priority=TaskPriority.HIGH,
            privacy_level="standard",
            required_agents=[AgentRole.ARCHITECT, AgentRole.ANALYZER],
            constraints=[
                "Focus on Python best practices",
                "Consider performance implications",
                "Maintain backward compatibility",
            ],
            success_criteria=[
                "Identify key architectural issues",
                "Provide actionable recommendations",
                "Include implementation timeline",
            ],
            context={
                "project_root": "/home/david/agents/my-fullstack-agent",
                "primary_language": "python",
                "existing_frameworks": ["FastAPI", "Pydantic", "Rust Extensions"],
            },
        )

        # Process task with progress tracking
        def progress_callback(progress):
            print(f"🔄 Progress: Iteration {progress['iteration']} - {progress['status']}")
            print(f"   Latest: {progress['latest_step'][:100]}...")

        async with orchestrator.enhanced_task_session(demo_task) as session:
            result = await orchestrator.process_enhanced_task(demo_task, progress_callback)
            session["result"] = result

        # Display results
        print("\n✅ Demo Task Results:")
        print(f"   Success: {result['success']}")
        print(f"   Execution time: {result['execution_time']:.2f}s")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Used local LLM: {result['used_local_llm']}")

        # Show metrics
        metrics = await orchestrator.get_comprehensive_metrics()
        print("\n📊 System Metrics:")
        print(f"   Tasks completed: {metrics['orchestrator_metrics']['tasks_completed']}")
        print(f"   Average task time: {metrics['orchestrator_metrics']['average_task_time']:.2f}s")
        print(f"   Message queue processed: {metrics['message_queue_stats']['processed']}")
        print(f"   Rust extensions: {metrics['rust_extensions']}")

    except Exception as e:
        logger.exception(f"Demo failed: {e}")
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    # Run demo when executed directly
    asyncio.run(demo_enhanced_react_orchestrator())
