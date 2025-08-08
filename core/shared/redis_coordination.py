#!/usr/bin/env python3
"""Redis Coordination System for Multi-Agent Communication.

This module provides Redis-based coordination and communication between agents:
- Inter-agent message passing and coordination
- Shared state management and synchronization
- Task queue management and distribution
- Real-time event broadcasting
- Agent registration and discovery
- Resource locking and coordination

Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Agent A       │───▶│  Redis Broker    │◀───│   Agent B       │
│  (Code Review)  │    │  (Coordination)  │    │ (Workspace)     │
└─────────────────┘    │                  │    └─────────────────┘
                       │  • Messages      │    ┌─────────────────┐
┌─────────────────┐    │  • State         │◀───│   Agent C       │
│   Agent D       │───▶│  • Tasks         │    │ (Architecture)  │
│ (Documentation) │    │  • Events        │    └─────────────────┘
└─────────────────┘    │  • Locks         │
                       └──────────────────┘
"""

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
import time
from typing import Any
import uuid

import aioredis
from pydantic import BaseModel
from pydantic import Field

# Configure logging
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_TIMEOUT = int(os.getenv("REDIS_TIMEOUT", "30"))

# Redis key prefixes
AGENT_PREFIX = "agent:"
MESSAGE_PREFIX = "msg:"
TASK_PREFIX = "task:"
EVENT_PREFIX = "event:"
LOCK_PREFIX = "lock:"
STATE_PREFIX = "state:"


class MessageType(str, Enum):
    """Message types for inter-agent communication."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""

    id: str
    from_agent: str
    to_agent: str | None  # None for broadcast
    message_type: MessageType
    payload: dict[str, Any]
    timestamp: float
    correlation_id: str | None = None
    expires_at: float | None = None


@dataclass
class AgentTask:
    """Task structure for distributed task management."""

    id: str
    agent_type: str
    task_type: str
    payload: dict[str, Any]
    status: TaskStatus
    created_at: float
    updated_at: float
    assigned_to: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    priority: int = 0  # Higher number = higher priority
    max_retries: int = 3
    retry_count: int = 0


class AgentInfo(BaseModel):
    """Agent registration information."""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Agent type (code-reviewer, workspace-analyzer, etc.)")
    capabilities: list[str] = Field(default_factory=list, description="Agent capabilities")
    status: str = Field(default="active", description="Agent status")
    last_heartbeat: float = Field(default_factory=time.time, description="Last heartbeat timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RedisCoordination:
    """Redis-based coordination system for multi-agent communication."""

    def __init__(
        self, agent_id: str, agent_type: str, capabilities: list[str] | None = None
    ) -> None:
        """Initialize Redis coordination.

        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (code-reviewer, workspace-analyzer, etc.)
            capabilities: List of capabilities this agent provides

        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities or []

        self._redis: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._message_handlers: dict[MessageType, list[Callable]] = {
            msg_type: [] for msg_type in MessageType
        }
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._message_processor_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Connect to Redis and initialize coordination."""
        try:
            self._redis = aioredis.from_url(
                REDIS_URL, db=REDIS_DB, socket_timeout=REDIS_TIMEOUT, decode_responses=True
            )

            # Test connection
            await self._redis.ping()
            logger.info(f"Connected to Redis at {REDIS_URL}")

            # Register agent
            await self._register_agent()

            # Set up pub/sub
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(
                f"{MESSAGE_PREFIX}{self.agent_id}",  # Direct messages
                f"{MESSAGE_PREFIX}broadcast",  # Broadcast messages
                f"{EVENT_PREFIX}*",  # Event notifications
            )

            # Start background tasks
            self._running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._message_processor_task = asyncio.create_task(self._message_processor())

            logger.info(f"Agent {self.agent_id} coordination initialized")

        except Exception as e:
            logger.exception(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis and cleanup."""
        self._running = False

        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._message_processor_task:
            self._message_processor_task.cancel()

        # Unregister agent
        if self._redis:
            await self._unregister_agent()

        # Close pub/sub
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()

        # Close Redis connection
        if self._redis:
            await self._redis.close()

        logger.info(f"Agent {self.agent_id} disconnected from coordination")

    async def _register_agent(self) -> None:
        """Register this agent in the coordination system."""
        agent_info = AgentInfo(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            capabilities=self.capabilities,
            status="active",
            last_heartbeat=time.time(),
        )

        await self._redis.hset(
            f"{AGENT_PREFIX}registry", self.agent_id, json.dumps(agent_info.dict())
        )

        logger.info(f"Registered agent {self.agent_id} of type {self.agent_type}")

    async def _unregister_agent(self) -> None:
        """Unregister this agent from the coordination system."""
        await self._redis.hdel(f"{AGENT_PREFIX}registry", self.agent_id)
        logger.info(f"Unregistered agent {self.agent_id}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to indicate agent is alive."""
        while self._running:
            try:
                await self._redis.hset(
                    f"{AGENT_PREFIX}registry",
                    self.agent_id,
                    json.dumps(
                        {
                            "agent_id": self.agent_id,
                            "agent_type": self.agent_type,
                            "capabilities": self.capabilities,
                            "status": "active",
                            "last_heartbeat": time.time(),
                        },
                    ),
                )
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Heartbeat failed: {e}")
                await asyncio.sleep(5)

    async def _message_processor(self) -> None:
        """Process incoming messages from Redis pub/sub."""
        while self._running:
            try:
                message = await self._pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Message processing error: {e}")
                await asyncio.sleep(1)

    async def _handle_message(self, redis_message: dict[str, Any]) -> None:
        """Handle incoming Redis message."""
        try:
            channel = redis_message["channel"]
            data = json.loads(redis_message["data"])

            if channel.startswith(MESSAGE_PREFIX):
                # Inter-agent message
                message = AgentMessage(**data)

                # Check if message is expired
                if message.expires_at and time.time() > message.expires_at:
                    logger.debug(f"Discarding expired message {message.id}")
                    return

                # Route to appropriate handlers
                handlers = self._message_handlers.get(message.message_type, [])
                for handler in handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.exception(f"Message handler error: {e}")

            elif channel.startswith(EVENT_PREFIX):
                # Event notification
                await self._handle_event(data)

        except Exception as e:
            logger.exception(f"Failed to handle message: {e}")

    async def _handle_event(self, event_data: dict[str, Any]) -> None:
        """Handle event notifications."""
        logger.info(f"Received event: {event_data}")

    def add_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Add a message handler for a specific message type."""
        self._message_handlers[message_type].append(handler)

    async def send_message(
        self,
        to_agent: str | None,
        message_type: MessageType,
        payload: dict[str, Any],
        correlation_id: str | None = None,
        expires_in_seconds: int | None = None,
    ) -> str:
        """Send a message to another agent or broadcast.

        Args:
            to_agent: Target agent ID (None for broadcast)
            message_type: Type of message
            payload: Message payload
            correlation_id: Optional correlation ID for request/response
            expires_in_seconds: Message expiration time

        Returns:
            Message ID

        """
        message = AgentMessage(
            id=str(uuid.uuid4()),
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            payload=payload,
            timestamp=time.time(),
            correlation_id=correlation_id,
            expires_at=time.time() + expires_in_seconds if expires_in_seconds else None,
        )

        # Determine channel
        channel = (
            f"{MESSAGE_PREFIX}broadcast" if to_agent is None else f"{MESSAGE_PREFIX}{to_agent}"
        )

        # Send message
        await self._redis.publish(channel, json.dumps(asdict(message)))

        logger.debug(f"Sent {message_type} message {message.id} to {to_agent or 'broadcast'}")
        return message.id

    async def request(
        self, to_agent: str, payload: dict[str, Any], timeout_seconds: int = 30
    ) -> dict[str, Any]:
        """Send a request and wait for response.

        Args:
            to_agent: Target agent ID
            payload: Request payload
            timeout_seconds: Request timeout

        Returns:
            Response payload

        """
        correlation_id = str(uuid.uuid4())
        response_future = asyncio.Future()

        # Set up response handler
        async def response_handler(message: AgentMessage) -> None:
            if (
                message.correlation_id == correlation_id
                and message.message_type == MessageType.RESPONSE
            ) and not response_future.done():
                response_future.set_result(message.payload)

        self.add_message_handler(MessageType.RESPONSE, response_handler)

        try:
            # Send request
            await self.send_message(
                to_agent=to_agent,
                message_type=MessageType.REQUEST,
                payload=payload,
                correlation_id=correlation_id,
                expires_in_seconds=timeout_seconds,
            )

            # Wait for response
            return await asyncio.wait_for(response_future, timeout=timeout_seconds)

        except TimeoutError:
            msg = f"Request to {to_agent} timed out after {timeout_seconds} seconds"
            raise TimeoutError(msg)
        finally:
            # Remove handler
            self._message_handlers[MessageType.RESPONSE].remove(response_handler)

    async def respond(self, original_message: AgentMessage, payload: dict[str, Any]) -> None:
        """Send a response to a request message."""
        await self.send_message(
            to_agent=original_message.from_agent,
            message_type=MessageType.RESPONSE,
            payload=payload,
            correlation_id=original_message.correlation_id,
        )

    async def broadcast(self, payload: dict[str, Any]) -> str:
        """Broadcast a message to all agents."""
        return await self.send_message(
            to_agent=None, message_type=MessageType.BROADCAST, payload=payload
        )

    async def notify(self, to_agent: str, payload: dict[str, Any]) -> str:
        """Send a notification to a specific agent."""
        return await self.send_message(
            to_agent=to_agent, message_type=MessageType.NOTIFICATION, payload=payload
        )

    async def get_agent_list(self) -> dict[str, AgentInfo]:
        """Get list of all registered agents."""
        agents_data = await self._redis.hgetall(f"{AGENT_PREFIX}registry")
        agents: dict[str, Any] = {}

        for agent_id, agent_json in agents_data.items():
            try:
                agent_info = AgentInfo(**json.loads(agent_json))
                agents[agent_id] = agent_info
            except Exception as e:
                logger.warning(f"Failed to parse agent info for {agent_id}: {e}")

        return agents

    async def find_agents_by_capability(self, capability: str) -> list[AgentInfo]:
        """Find agents that have a specific capability."""
        agents = await self.get_agent_list()
        return [agent for agent in agents.values() if capability in agent.capabilities]

    async def create_task(
        self, agent_type: str, task_type: str, payload: dict[str, Any], priority: int = 0
    ) -> str:
        """Create a task for distributed processing.

        Args:
            agent_type: Type of agent that should handle this task
            task_type: Specific task type
            payload: Task payload
            priority: Task priority (higher = more important)

        Returns:
            Task ID

        """
        task = AgentTask(
            id=str(uuid.uuid4()),
            agent_type=agent_type,
            task_type=task_type,
            payload=payload,
            status=TaskStatus.PENDING,
            created_at=time.time(),
            updated_at=time.time(),
            priority=priority,
        )

        # Store task
        await self._redis.hset(f"{TASK_PREFIX}all", task.id, json.dumps(asdict(task)))

        # Add to priority queue for the agent type
        await self._redis.zadd(f"{TASK_PREFIX}queue:{agent_type}", {task.id: priority})

        logger.info(f"Created task {task.id} for {agent_type}")
        return task.id

    async def claim_task(self, agent_type: str) -> AgentTask | None:
        """Claim the highest priority task for this agent type.

        Args:
            agent_type: Agent type

        Returns:
            Task if available, None otherwise

        """
        # Get highest priority task
        result = await self._redis.zpopmax(f"{TASK_PREFIX}queue:{agent_type}")
        if not result:
            return None

        task_id, _ = result[0]

        # Get full task data
        task_json = await self._redis.hget(f"{TASK_PREFIX}all", task_id)
        if not task_json:
            return None

        task = AgentTask(**json.loads(task_json))

        # Mark as running and assign to this agent
        task.status = TaskStatus.RUNNING
        task.assigned_to = self.agent_id
        task.updated_at = time.time()

        await self._redis.hset(f"{TASK_PREFIX}all", task.id, json.dumps(asdict(task)))

        logger.info(f"Claimed task {task.id}")
        return task

    async def complete_task(
        self,
        task_id: str,
        result: dict[str, Any],
        status: TaskStatus = TaskStatus.COMPLETED,
    ) -> None:
        """Mark a task as completed with results.

        Args:
            task_id: Task ID
            result: Task result
            status: Final task status

        """
        task_json = await self._redis.hget(f"{TASK_PREFIX}all", task_id)
        if not task_json:
            logger.warning(f"Task {task_id} not found")
            return

        task = AgentTask(**json.loads(task_json))
        task.status = status
        task.result = result
        task.updated_at = time.time()

        await self._redis.hset(f"{TASK_PREFIX}all", task.id, json.dumps(asdict(task)))

        logger.info(f"Completed task {task_id} with status {status}")

    async def acquire_lock(self, resource: str, timeout_seconds: int = 30) -> bool:
        """Acquire a distributed lock for a resource.

        Args:
            resource: Resource identifier
            timeout_seconds: Lock timeout

        Returns:
            True if lock acquired, False otherwise

        """
        lock_key = f"{LOCK_PREFIX}{resource}"
        lock_value = f"{self.agent_id}:{time.time()}"

        # Try to acquire lock
        result = await self._redis.set(lock_key, lock_value, nx=True, ex=timeout_seconds)

        acquired = bool(result)
        if acquired:
            logger.debug(f"Acquired lock for {resource}")
        else:
            logger.debug(f"Failed to acquire lock for {resource}")

        return acquired

    async def release_lock(self, resource: str) -> bool:
        """Release a distributed lock for a resource.

        Args:
            resource: Resource identifier

        Returns:
            True if lock released, False if not held by this agent

        """
        lock_key = f"{LOCK_PREFIX}{resource}"
        current_value = await self._redis.get(lock_key)

        if current_value and current_value.startswith(f"{self.agent_id}:"):
            await self._redis.delete(lock_key)
            logger.debug(f"Released lock for {resource}")
            return True

        logger.debug(f"Cannot release lock for {resource} - not held by this agent")
        return False

    async def set_shared_state(
        self, key: str, value: dict[str, Any], ttl_seconds: int | None = None
    ) -> None:
        """Set shared state accessible by all agents."""
        state_key = f"{STATE_PREFIX}{key}"
        await self._redis.set(state_key, json.dumps(value), ex=ttl_seconds)

    async def get_shared_state(self, key: str) -> dict[str, Any] | None:
        """Get shared state."""
        state_key = f"{STATE_PREFIX}{key}"
        value = await self._redis.get(state_key)
        return json.loads(value) if value else None

    async def publish_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event to all interested agents."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
            "source": self.agent_id,
        }

        await self._redis.publish(f"{EVENT_PREFIX}{event_type}", json.dumps(event))
        logger.debug(f"Published event {event_type}")


# Global coordination instance
_coordination: RedisCoordination | None = None


async def get_coordination(
    agent_id: str, agent_type: str, capabilities: list[str] | None = None
) -> RedisCoordination:
    """Get or create global coordination instance."""
    global _coordination

    if _coordination is None:
        _coordination = RedisCoordination(agent_id, agent_type, capabilities)
        await _coordination.connect()

    return _coordination


async def cleanup_coordination() -> None:
    """Cleanup global coordination instance."""
    global _coordination

    if _coordination:
        await _coordination.disconnect()
        _coordination = None
