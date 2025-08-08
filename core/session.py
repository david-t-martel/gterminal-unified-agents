#!/usr/bin/env python3
"""Session Management - Maintains state across requests and interfaces."""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles
import aiofiles.os
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

logger = logging.getLogger(__name__)


class Interaction(BaseModel):
    """Represents a single interaction in a session."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = {}


class Session(BaseModel):
    """Represents a user session with history and context."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    interactions: list[Interaction] = []
    context: dict[str, Any] = {}
    active_tools: list[str] = []
    websocket: Any | None = None  # WebSocket connection if connected

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_interaction(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add an interaction to the session history."""
        interaction = Interaction(role=role, content=content, metadata=metadata or {})
        self.interactions.append(interaction)
        self.last_activity = datetime.now()

    def get_recent_interactions(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get recent interactions for context building."""
        recent = self.interactions[-limit:] if self.interactions else []
        return [
            {
                "role": i.role,
                "content": i.content[:200],  # Truncate for context
                "timestamp": i.timestamp.isoformat(),
            }
            for i in recent
        ]

    def update_context(self, key: str, value: Any) -> None:
        """Update session context."""
        self.context[key] = value
        self.last_activity = datetime.now()

    def clear_interactions(self) -> None:
        """Clear interaction history while preserving context."""
        self.interactions = []
        self.last_activity = datetime.now()


class SessionPersistence:
    """Handles session persistence to disk."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir or Path("/home/david/agents/my-fullstack-agent/.sessions")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, session: Session) -> None:
        """Save session to disk using async I/O with atomic writes."""
        try:
            session_file = self.storage_dir / f"{session.id}.json"
            temp_file = self.storage_dir / f"{session.id}.json.tmp"

            # Prepare data for JSON serialization
            data = {
                "id": session.id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "interactions": [
                    {
                        "role": i.role,
                        "content": i.content,
                        "timestamp": i.timestamp.isoformat(),
                        "metadata": i.metadata,
                    }
                    for i in session.interactions
                ],
                "context": session.context,
                "active_tools": session.active_tools,
            }

            # Serialize JSON to string first to avoid blocking in the async context
            json_data = json.dumps(data, indent=2)

            # Write to temporary file first for atomic operation
            async with aiofiles.open(temp_file, mode="w", encoding="utf-8") as f:
                await f.write(json_data)

            # Atomically rename temp file to actual file
            # This ensures we don't have partially written files
            await aiofiles.os.rename(temp_file, session_file)

            logger.debug(f"Saved session {session.id}")

        except Exception as e:
            logger.exception(f"Failed to save session {session.id}: {e}")
            # Clean up temp file if it exists
            try:
                if temp_file.exists():
                    await aiofiles.os.remove(temp_file)
            except Exception:
                pass

    async def load(self, session_id: str) -> Session | None:
        """Load session from disk using async I/O."""
        try:
            session_file = self.storage_dir / f"{session_id}.json"

            if not session_file.exists():
                return None

            # Read file asynchronously
            async with aiofiles.open(session_file, encoding="utf-8") as f:
                json_data = await f.read()

            # Parse JSON (this is CPU-bound, not I/O-bound, so it's ok to do synchronously)
            data = json.loads(json_data)

            # Reconstruct session
            session = Session(
                id=data["id"],
                created_at=datetime.fromisoformat(data["created_at"]),
                last_activity=datetime.fromisoformat(data["last_activity"]),
                context=data.get("context", {}),
                active_tools=data.get("active_tools", []),
            )

            # Reconstruct interactions
            for interaction_data in data.get("interactions", []):
                interaction = Interaction(
                    role=interaction_data["role"],
                    content=interaction_data["content"],
                    timestamp=datetime.fromisoformat(interaction_data["timestamp"]),
                    metadata=interaction_data.get("metadata", {}),
                )
                session.interactions.append(interaction)

            logger.debug(f"Loaded session {session_id}")
            return session

        except Exception as e:
            logger.exception(f"Failed to load session {session_id}: {e}")
            return None

    async def list_sessions(self) -> list[str]:
        """List all saved session IDs."""
        try:
            session_files = self.storage_dir.glob("*.json")
            return [f.stem for f in session_files]
        except Exception as e:
            logger.exception(f"Failed to list sessions: {e}")
            return []

    async def delete(self, session_id: str) -> None:
        """Delete a saved session using async I/O."""
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            if session_file.exists():
                await aiofiles.os.remove(session_file)
                logger.debug(f"Deleted session {session_id}")
        except Exception as e:
            logger.exception(f"Failed to delete session {session_id}: {e}")


class SessionManager:
    """Manages active sessions and persistence."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.sessions: dict[str, Session] = {}
        self.persistence = SessionPersistence(storage_dir)
        logger.info("Session Manager initialized")

    def get_or_create(self, session_id: str | None = None) -> Session:
        """Get existing session or create new one (synchronous version)."""
        if not session_id:
            session_id = str(uuid4())

        if session_id not in self.sessions:
            # Create new session without trying to load from disk in sync context
            self.sessions[session_id] = Session(id=session_id)
            logger.info(f"Created new session {session_id}")

        return self.sessions[session_id]

    async def get_or_create_async(self, session_id: str | None = None) -> Session:
        """Get existing session or create new one (async version with persistence)."""
        if not session_id:
            session_id = str(uuid4())

        if session_id not in self.sessions:
            # Try to load from disk first
            saved_session = await self.persistence.load(session_id)

            if saved_session:
                self.sessions[session_id] = saved_session
                logger.info(f"Loaded existing session {session_id}")
            else:
                self.sessions[session_id] = Session(id=session_id)
                logger.info(f"Created new session {session_id}")

        return self.sessions[session_id]

    async def save(self, session_id: str) -> None:
        """Save session to persistent storage."""
        if session_id in self.sessions:
            await self.persistence.save(self.sessions[session_id])

    async def save_all(self) -> None:
        """Save all active sessions."""
        for session_id in self.sessions:
            await self.save(session_id)

    def remove(self, session_id: str) -> None:
        """Remove session from active sessions."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Removed session {session_id} from active sessions")

    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> None:
        """Clean up old inactive sessions."""
        now = datetime.now()
        sessions_to_remove: list[Any] = []

        for session_id, session in self.sessions.items():
            age_hours = (now - session.last_activity).total_seconds() / 3600
            if age_hours > max_age_hours:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            await self.save(session_id)  # Save before removing
            self.remove(session_id)
            logger.info(f"Cleaned up old session {session_id}")

    def get_active_sessions(self) -> list[dict[str, Any]]:
        """Get information about active sessions."""
        return [
            {
                "id": session.id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "interaction_count": len(session.interactions),
                "has_websocket": session.websocket is not None,
            }
            for session in self.sessions.values()
        ]
