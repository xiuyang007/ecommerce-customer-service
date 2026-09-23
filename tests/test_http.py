from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import AIMessageChunk
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.main as main_module
from app.config import Settings
from app.main import app
from app.models import Base, Conversation, FAQ, Message, Ticket
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
    original_repository = app.state.repository
    model = FakeModel()
    app.state.provider = FakeProvider(model)
    app.state.sessions = InMemorySessionStore(Settings())
    app.state.repository = None
    yield model
    app.state.provider = original_provider
    app.state.sessions = original_sessions
    app.state.repository = original_repository


@pytest.fixture
async def sqlite_session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with session_factory() as session:
        session.add(FAQ(question="退货政策是什么", answer="7 天内可申请。", category="policy"))
        conversation = Conversation(session_id="s_database_test", status="open")
        session.add(conversation)
        await session.flush()
        session.add(Message(conversation_id=conversation.id, role="user", content="测试消息"))
        session.add(
            Ticket(
                ticket_id="T202609190001",
                conversation_id=conversation.id,
                description="测试工单",
                ticket_type="other",
                status="pending",
            )
        )
        await session.commit()
    yield session_factory
    await engine.dispose()


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
    original_repository = app.state.repository
    app.state.provider = ModelProvider(Settings(llm_api_key=SecretStr("")))
    app.state.sessions = InMemorySessionStore(Settings())
    app.state.repository = None
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/chat/stream", json={"message": "你好"})
        assert response.status_code == 503
        assert "LLM_API_KEY" in response.json()["detail"]
    finally:
        app.state.provider = original_provider
        app.state.sessions = original_sessions
        app.state.repository = original_repository


@pytest.mark.asyncio
async def test_database_tables_returns_six_business_tables(
    monkeypatch,
    sqlite_session_factory,
) -> None:
    monkeypatch.setattr(main_module, "SessionFactory", sqlite_session_factory)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/database/tables?limit=5")
    assert response.status_code == 200
    tables = response.json()["tables"]
    assert [table["name"] for table in tables] == [
        "faq",
        "conversations",
        "messages",
        "tickets",
        "low_confidence_questions",
        "faith_cases",
    ]
    counts = {table["name"]: table["total"] for table in tables}
    assert counts == {
        "faq": 1,
        "conversations": 1,
        "messages": 1,
        "tickets": 1,
        "low_confidence_questions": 0,
        "faith_cases": 0,
    }
    assert tables[3]["rows"][0]["ticket_id"] == "T202609190001"


@pytest.mark.asyncio
async def test_database_tables_rejects_invalid_limit() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/database/tables?limit=0")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_browser_client_is_served_at_root() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "电商智能客服" in response.text
    assert "api/v1/chat/stream" in response.text
    assert "tool-badge" in response.text
    assert "数据表" in response.text
    assert "citation-ref" in response.text
    assert "已反馈" in response.text
