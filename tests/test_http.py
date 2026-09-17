from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import AIMessageChunk
from pydantic import SecretStr

from app.config import Settings
from app.main import app
from app.schemas import AfterSalesExtraction
from app.services.chat import ModelProvider
from app.store.sessions import InMemorySessionStore


class FakeStructuredRunnable:
    async def ainvoke(self, messages):
        return AfterSalesExtraction(
            order_id="A-1",
            request_type="exchange",
            expected_solution="换新",
        )


class FakeModel:
    def __init__(self):
        self.calls = []

    async def astream(self, messages):
        self.calls.append(messages)
        yield AIMessageChunk(content="您好，")
        yield AIMessageChunk(content="我会帮您整理问题。")

    def with_structured_output(self, schema, method, include_raw=False):
        assert schema is AfterSalesExtraction
        assert method in {"function_calling", "json_mode", "json_schema"}
        return FakeStructuredRunnable()


class FakeProvider:
    def __init__(self, model):
        self.model = model

    def get(self):
        return self.model


@pytest.fixture
def fake_provider():
    original_provider = app.state.provider
    original_sessions = app.state.sessions
    model = FakeModel()
    app.state.provider = FakeProvider(model)
    app.state.sessions = InMemorySessionStore(Settings())
    yield model
    app.state.provider = original_provider
    app.state.sessions = original_sessions


@pytest.mark.asyncio
async def test_chat_stream_and_second_turn_context(fake_provider) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat/stream",
            json={"message": "第一轮问题"},
        )
        assert response.status_code == 200
        assert "event: meta" in response.text
        assert "event: delta" in response.text
        assert "event: done" in response.text
        meta_line = next(
            line[6:]
            for line in response.text.splitlines()
            if line.startswith("data: {\"session_id")
        )
        session_id = json.loads(meta_line)["session_id"]

        second = await client.post(
            "/api/v1/chat/stream",
            json={"session_id": session_id, "message": "第二轮问题"},
        )
        assert second.status_code == 200
        sent_messages = fake_provider.calls[1]
        contents = [message.content for message in sent_messages]
        assert "第一轮问题" in contents
        assert "第二轮问题" in contents


@pytest.mark.asyncio
async def test_after_sales_endpoint_returns_fixed_json(fake_provider) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/after-sales/extract",
            json={"text": "订单 A-1，我想换新。"},
        )
    assert response.status_code == 200
    assert response.json() == {
        "order_id": "A-1",
        "request_type": "exchange",
        "expected_solution": "换新",
    }


@pytest.mark.asyncio
async def test_missing_llm_key_is_rejected_before_sse_starts() -> None:
    original_provider = app.state.provider
    original_sessions = app.state.sessions
    app.state.provider = ModelProvider(Settings(llm_api_key=SecretStr("")))
    app.state.sessions = InMemorySessionStore(Settings())
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/chat/stream", json={"message": "你好"})
        assert response.status_code == 503
        assert "LLM_API_KEY" in response.json()["detail"]
    finally:
        app.state.provider = original_provider
        app.state.sessions = original_sessions
    


@pytest.mark.asyncio
async def test_browser_client_is_served_at_root() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "电商智能客服" in response.text
    assert "api/v1/chat/stream" in response.text
    assert "售后提取" in response.text
    assert "localStorage" in response.text


