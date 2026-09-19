from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models import Base, FAQ
from app.repositories import SqlAlchemyChatRepository
from app.services.tool_chat import ToolChatService


class FakeProvider:
    def __init__(self, model):
        self.model = model

    def get(self):
        return self.model


class FakeToolPlanningModel:
    def __init__(self, plans):
        self.plans = list(plans)
        self.plan_index = 0
        self.final_calls = []

    def bind_tools(self, tools):
        assert {tool.name for tool in tools} == {
            "query_order",
            "query_product",
            "query_logistics",
            "query_faq",
            "create_ticket",
        }
        return self

    async def ainvoke(self, messages):
        plan = self.plans[min(self.plan_index, len(self.plans) - 1)]
        self.plan_index += 1
        return plan

    async def astream(self, messages):
        self.final_calls.append(messages)
        yield AIMessageChunk(content="物流")
        yield AIMessageChunk(content="已更新")


@pytest.fixture
async def repository():
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
        await session.commit()
    yield SqlAlchemyChatRepository(session_factory)
    await engine.dispose()


@pytest.mark.asyncio
async def test_tool_chat_executes_one_tool_and_persists_trace(repository) -> None:
    model = FakeToolPlanningModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_logistics",
                        "args": {"order_id": "1001"},
                        "id": "call_logistics_1",
                    }
                ],
            ),
            AIMessage(content=""),
        ]
    )
    service = ToolChatService(Settings(), FakeProvider(model), repository)
    prepared = await service.prepare(None, "订单 1001 的物流到哪了")

    events = [event async for event in service.stream(prepared)]
    names = [name for name, _ in events]
    assert names[:3] == ["meta", "tool_status", "tool_result"]
    assert "delta" in names and names[-1] == "done"
    assert events[1][1]["tool"] == "query_logistics"
    assert events[2][1]["ok"] is True
    assert events[-1][1]["tool_used"] is True

    rows = await repository.list_messages(prepared.conversation_id)
    assert [row.role for row in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[1].tool_name == "query_logistics"
    assert rows[2].tool_result["order_id"] == "1001"


@pytest.mark.asyncio
async def test_tool_chat_no_tool_path_streams_normally(repository) -> None:
    model = FakeToolPlanningModel([AIMessage(content="")])
    service = ToolChatService(Settings(), FakeProvider(model), repository)
    prepared = await service.prepare(None, "你好")
    events = [event async for event in service.stream(prepared)]
    assert [name for name, _ in events] == ["meta", "delta", "delta", "done"]
    assert events[-1][1]["tool_used"] is False
