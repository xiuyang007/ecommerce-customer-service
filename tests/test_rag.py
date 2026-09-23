from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.models import Base, Conversation
from app.rag.retrieval import KnowledgeRetrievalService
from app.rag.schemas import QueryAnalysis, RetrievedChunk
from app.repositories import SqlAlchemyChatRepository


class FakeQueryUnderstanding:
    async def analyze(self, query: str) -> QueryAnalysis:
        return QueryAnalysis(
            normalized_query=query,
            keywords=[query],
            synonyms=["同义词"],
            category=None,
            confidence=0.9,
        )


class FakeEmbedding:
    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    def unload(self) -> None:
        return None


class FakeReranker:
    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int):
        for index, chunk in enumerate(chunks):
            chunk.score = 1.0 - index / 100
        return chunks[:top_k]

    def unload(self) -> None:
        return None


class FakeStore:
    def __init__(self):
        self.calls: list[str] = []

    def _chunks(self):
        return [
            RetrievedChunk(
                chunk_id="policy-return#1",
                document_id="policy-return",
                text="签收后 7 天内可以申请退货。",
                category="policy",
                section_path=["售后政策", "退货"],
                source_uri="knowledge://policy/return#1",
            ),
            RetrievedChunk(
                chunk_id="policy-return#2",
                document_id="policy-return",
                text="运费承担方式以店铺审核为准。",
                category="policy",
                section_path=["售后政策", "退货", "运费"],
                source_uri="knowledge://policy/return#2",
            ),
        ]

    def dense_search(self, vector, limit, category):
        self.calls.append("dense")
        return self._chunks()[:limit]

    def bm25_search(self, query, limit, category):
        self.calls.append("bm25")
        return self._chunks()[:limit]

    def hybrid_search(self, vector, query, limit, category):
        self.calls.append("hybrid")
        return self._chunks()[:limit]


@pytest.mark.asyncio
async def test_four_retrieval_strategies_assign_citations() -> None:
    store = FakeStore()
    service = KnowledgeRetrievalService(
        Settings(),
        model_provider=None,
        store=store,
        embedding=FakeEmbedding(),
        reranker=FakeReranker(),
        query_understanding=FakeQueryUnderstanding(),
    )
    for strategy in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        result = await service.search("退货政策是什么", strategy=strategy, limit=2)
        assert [chunk.citation_id for chunk in result.chunks] == [1, 2]
        assert result.chunks[0].chunk_id == "policy-return#1"


@pytest.mark.asyncio
async def test_low_confidence_and_faith_case_persistence() -> None:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with factory() as session:
        session.add(Conversation(session_id="s_test", status="open"))
        await session.commit()
    repository = SqlAlchemyChatRepository(factory)
    conversation = await repository.ensure_conversation("s_test")

    row = await repository.add_low_confidence_question(
        conversation_id=conversation.id,
        raw_question="这个问题知识库没有",
        source="retrieval_low_conf",
        reason="检索结果为空",
    )
    assert row.id is not None

    first = await repository.upsert_faith_case(
        eval_id="A01",
        bucket="A_policy",
        query="退货政策是什么",
        strategy="hybrid_rerank",
        answer="错误答案",
        reason="证据冲突",
        citations=[],
        judge_model="test-judge",
    )
    second = await repository.upsert_faith_case(
        eval_id="A01",
        bucket="A_policy",
        query="退货政策是什么",
        strategy="hybrid_rerank",
        answer="修正后仍错误",
        reason="再次编造",
        citations=[{"n": 1, "chunk_id": "policy-return#1"}],
        judge_model="test-judge",
    )
    assert first.id == second.id
    assert second.seen_count == 2
    assert second.status == "未解决"
    await engine.dispose()
