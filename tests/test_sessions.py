from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.store.sessions import InMemorySessionStore, SessionBusyError, UnknownSessionError


@pytest.mark.asyncio
async def test_session_commits_only_completed_turns() -> None:
    store = InMemorySessionStore(Settings())
    handle = await store.begin(None)
    with pytest.raises(SessionBusyError):
        await store.begin(handle.session_id)

    await store.abort(handle)
    handle = await store.begin(handle.session_id)
    await store.commit(handle, "问题", "回答")
    messages = await store.get_messages(handle.session_id)
    assert [message.content for message in messages] == ["问题", "回答"]


@pytest.mark.asyncio
async def test_unknown_session_is_not_created_silently() -> None:
    store = InMemorySessionStore(Settings())
    with pytest.raises(UnknownSessionError):
        await store.begin("missing")
