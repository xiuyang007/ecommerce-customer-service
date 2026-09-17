from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import monotonic

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import Settings


class UnknownSessionError(Exception):
    pass


class SessionBusyError(Exception):
    pass


@dataclass
class SessionHandle:
    session_id: str
    history: list[BaseMessage]


@dataclass
class _Session:
    messages: list[BaseMessage]
    updated_at: float
    busy: bool = False


class InMemorySessionStore:
    """Small single-process store for the Chapter 01 demo."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()

    async def begin(self, session_id: str | None) -> SessionHandle:
        async with self._lock:
            self._purge_expired()
            if session_id is None:
                session_id = self._new_session_id()
                if len(self._sessions) >= self._settings.max_session_count:
                    oldest_id = min(self._sessions, key=lambda key: self._sessions[key].updated_at)
                    del self._sessions[oldest_id]
                session = _Session(messages=[], updated_at=monotonic())
                self._sessions[session_id] = session
            else:
                session = self._sessions.get(session_id)
                if session is None:
                    raise UnknownSessionError(session_id)

            if session.busy:
                raise SessionBusyError(session_id)
            session.busy = True
            session.updated_at = monotonic()
            return SessionHandle(session_id=session_id, history=list(session.messages))

    async def commit(self, handle: SessionHandle, user_text: str, assistant_text: str) -> None:
        async with self._lock:
            session = self._sessions.get(handle.session_id)
            if session is None:
                return
            session.messages.extend(
                [HumanMessage(content=user_text), AIMessage(content=assistant_text)]
            )
            session.busy = False
            session.updated_at = monotonic()

    async def abort(self, handle: SessionHandle) -> None:
        async with self._lock:
            session = self._sessions.get(handle.session_id)
            if session is not None:
                session.busy = False
                session.updated_at = monotonic()

    async def get_messages(self, session_id: str) -> list[BaseMessage]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise UnknownSessionError(session_id)
            return list(session.messages)

    def _purge_expired(self) -> None:
        deadline = monotonic() - self._settings.session_ttl_seconds
        expired = [
            key
            for key, value in self._sessions.items()
            if value.updated_at < deadline and not value.busy
        ]
        for key in expired:
            del self._sessions[key]

    @staticmethod
    def _new_session_id() -> str:
        import uuid

        return f"s_{uuid.uuid4().hex}"
