from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.models import Base, FAQ
from app.repositories import SqlAlchemyChatRepository
from app.services.tool_executor import ToolExecutor
from app.services.tools import RegisteredTool, ToolRegistry


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
        session.add_all(
            [
                FAQ(question="退货政策是什么", answer="签收后 7 天内可申请退货。", category="policy"),
                FAQ(question="如何申请换货", answer="在订单详情中提交申请。", category="after_sales"),
            ]
        )
        await session.commit()
    yield SqlAlchemyChatRepository(session_factory)
    await engine.dispose()


@pytest.mark.asyncio
async def test_query_faq_hits_and_intentional_miss(repository) -> None:
    conversation = await repository.ensure_conversation(None)
    registry = ToolRegistry.for_conversation(repository, conversation.id)
    faq_tool = registry.get("query_faq")
    assert faq_tool is not None

    hit = await faq_tool.tool.ainvoke({"keyword": "退货政策"})
    miss = await faq_tool.tool.ainvoke({"keyword": "邮费"})
    assert hit and hit[0]["question"] == "退货政策是什么"
    assert miss == []


@pytest.mark.asyncio
async def test_create_ticket_persists_ticket(repository) -> None:
    conversation = await repository.ensure_conversation(None)
    registry = ToolRegistry.for_conversation(repository, conversation.id)
    ticket_tool = registry.get("create_ticket")
    assert ticket_tool is not None

    result = await ticket_tool.tool.ainvoke(
        {"description": "耳机左边没有声音", "ticket_type": "exchange"}
    )
    assert result["status"] == "pending"
    assert result["ticket_type"] == "exchange"
    assert result["ticket_id"].startswith("T")


@pytest.mark.asyncio
async def test_tool_executor_reports_unknown_tool(repository) -> None:
    conversation = await repository.ensure_conversation(None)
    registry = ToolRegistry.for_conversation(repository, conversation.id)
    execution = await ToolExecutor(registry).execute(
        {"name": "missing_tool", "id": "call_1", "args": {}}
    )
    assert execution.ok is False
    assert execution.error and execution.error["type"] == "unknown_tool"


@pytest.mark.asyncio
async def test_query_logistics_returns_single_tool_result(repository) -> None:
    conversation = await repository.ensure_conversation(None)
    registry = ToolRegistry.for_conversation(repository, conversation.id)
    execution = await ToolExecutor(registry).execute(
        {"name": "query_logistics", "id": "call_2", "args": {"order_id": "1001"}}
    )
    assert execution.ok is True
    assert execution.result["order_id"] == "1001"
    assert execution.result["latest"]


@pytest.mark.asyncio
async def test_tool_executor_retries_retryable_failure() -> None:
    attempts = 0

    @tool
    async def flaky_query(keyword: str) -> dict:
        """Test tool that succeeds on the second attempt."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return {"keyword": keyword}

    registry = ToolRegistry([RegisteredTool(flaky_query, timeout_seconds=1, retryable=True)])
    execution = await ToolExecutor(registry).execute(
        {"name": "flaky_query", "id": "call_3", "args": {"keyword": "test"}}
    )
    assert execution.ok is True
    assert execution.attempts == 2


@pytest.mark.asyncio
async def test_tool_executor_reports_timeout() -> None:
    @tool
    async def slow_query(keyword: str) -> dict:
        """Test tool that exceeds its timeout."""
        await asyncio.sleep(0.05)
        return {"keyword": keyword}

    registry = ToolRegistry([RegisteredTool(slow_query, timeout_seconds=0.001, retryable=False)])
    execution = await ToolExecutor(registry).execute(
        {"name": "slow_query", "id": "call_4", "args": {"keyword": "test"}}
    )
    assert execution.ok is False
    assert execution.error and execution.error["type"] == "timeout"
