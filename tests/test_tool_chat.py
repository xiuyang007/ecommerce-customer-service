from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models import Base, FAQ
from app.rag.schemas import EvidenceAssessment, QueryAnalysis, RetrievalResult, RetrievedChunk
from app.repositories import SqlAlchemyChatRepository
from app.services.tool_chat import ToolChatService


class FakeProvider:
    def __init__(self, model):
        self.model = model

    def get(self):
        return self.model


class FakeToolPlanningModel:
    def __init__(self, plans, expected_tools: set[str] | None = None):
        self.plans = list(plans)
        self.plan_index = 0
        self.final_calls = []
        self.expected_tools = expected_tools

    def bind_tools(self, tools):
        expected = self.expected_tools or {
            "query_order",
            "query_product",
            "query_logistics",
            "query_faq",
            "create_ticket",
        }
        assert {tool.name for tool in tools} == expected
        return self

    async def ainvoke(self, messages):
        plan = self.plans[min(self.plan_index, len(self.plans) - 1)]
        self.plan_index += 1
        return plan

    async def astream(self, messages):
        self.final_calls.append(messages)
        yield AIMessageChunk(content="物流")
        yield AIMessageChunk(content="已更新")


class FakeRetrievalService:
    async def search(self, query, *, category=None, strategy="hybrid_rerank", limit=None):
        return RetrievalResult(
            query=query,
            analysis=QueryAnalysis(normalized_query=query),
            strategy=strategy,
            chunks=[
                RetrievedChunk(
                    chunk_id="policy-return#1",
                    document_id="policy-return",
                    text="签收后 7 天内可以申请退货。",
                    category="policy",
                    section_path=["售后政策", "退货"],
                    source_uri="knowledge://policy/return#1",
                    score=0.99,
                    citation_id=1,
                )
            ],
        )

    async def assess_evidence(self, query, chunks, extra_evidence=""):
        return EvidenceAssessment(sufficient=True, confidence=0.95, reason="证据直接支持")

    @staticmethod
    def prompt_order(chunks):
        return chunks

    @staticmethod
    def extract_citation_ids(answer):
        return [1]


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


@pytest.mark.asyncio
async def test_tool_chat_executes_multiple_steps_then_converges(repository) -> None:
    model = FakeToolPlanningModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_logistics",
                        "args": {"order_id": "1001"},
                        "id": "call_step_1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_faq",
                        "args": {"keyword": "退货政策"},
                        "id": "call_step_2",
                    }
                ],
            ),
            AIMessage(content=""),
        ]
    )
    service = ToolChatService(Settings(tool_max_steps=3), FakeProvider(model), repository)
    prepared = await service.prepare(None, "物流和退货政策分别是什么")
    events = [event async for event in service.stream(prepared)]
    assert [name for name, _ in events[:5]] == [
        "meta",
        "tool_status",
        "tool_result",
        "tool_status",
        "tool_result",
    ]
    assert events[1][1]["step"] == 1
    assert events[3][1]["step"] == 2
    assert events[-1][1]["tool_steps"] == 2
    assert events[-1][1]["finish_reason"] == "stop"
    rows = await repository.list_messages(prepared.conversation_id)
    assert [row.role for row in rows] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_tool_chat_deduplicates_identical_tool_calls(repository) -> None:
    duplicate_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_faq",
                "args": {"keyword": "邮费"},
                "id": "call_duplicate",
            }
        ],
    )
    model = FakeToolPlanningModel(
        [
            duplicate_call,
            AIMessage(
                content="",
                tool_calls=[{**duplicate_call.tool_calls[0], "id": "call_duplicate_2"}],
            ),
        ]
    )
    service = ToolChatService(Settings(tool_max_steps=3), FakeProvider(model), repository)
    prepared = await service.prepare(None, "邮费是多少")
    events = [event async for event in service.stream(prepared)]
    assert [name for name, _ in events].count("tool_status") == 1
    assert [name for name, _ in events].count("tool_result") == 1
    assert events[-1][0] == "done"


@pytest.mark.asyncio
async def test_tool_chat_emits_sources_for_knowledge_retrieval(repository) -> None:
    expected = {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
        "search_knowledge",
    }
    model = FakeToolPlanningModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge",
                        "args": {"query": "退货政策是什么"},
                        "id": "call_rag_1",
                    }
                ],
            ),
            AIMessage(content=""),
        ],
        expected_tools=expected,
    )
    service = ToolChatService(
        Settings(tool_max_steps=3),
        FakeProvider(model),
        repository,
        FakeRetrievalService(),
    )
    prepared = await service.prepare(None, "退货政策是什么")
    events = [event async for event in service.stream(prepared)]
    names = [name for name, _ in events]
    assert "tool_status" in names
    assert "sources" in names
    sources = next(payload for name, payload in events if name == "sources")
    assert sources["citations"][0]["chunk_id"] == "policy-return#1"
