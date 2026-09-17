from __future__ import annotations

import json
from collections.abc import AsyncIterator


def sse_event(event: str, payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {encoded}\n\n"


async def as_sse(events: AsyncIterator[tuple[str, dict]]) -> AsyncIterator[str]:
    async for event, payload in events:
        yield sse_event(event, payload)
