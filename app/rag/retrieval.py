from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings
from app.rag.embedding import EmbeddingService
from app.rag.milvus_store import MilvusKnowledgeStore
from app.rag.query_understanding import QueryUnderstandingService
from app.rag.reranker import RerankerService
from app.rag.schemas import EvidenceAssessment, RetrievalResult, RetrievedChunk


class KnowledgeRetrievalService:
    def __init__(
        self,
        settings: Settings,
        model_provider,
        *,
        store: MilvusKnowledgeStore | None = None,
        embedding: EmbeddingService | None = None,
        reranker: RerankerService | None = None,
        query_understanding: QueryUnderstandingService | None = None,
    ):
        self.settings = settings
        self.model_provider = model_provider
        self.store = store or MilvusKnowledgeStore(settings)
        self.embedding = embedding or EmbeddingService(settings)
        self.reranker = reranker or RerankerService(settings)
        self.query_understanding = query_understanding or QueryUnderstandingService(model_provider, settings)

    async def search(
        self,
        query: str,
        *,
        category: str | None = None,
        strategy: str = "hybrid_rerank",
        limit: int | None = None,
    ) -> RetrievalResult:
        top_k = limit or self.settings.rag_final_top_k
        analysis = await self.query_understanding.analyze(query)
        effective_category = category or (
            analysis.category if analysis.confidence >= 0.75 else None
        )
        search_text = " ".join([analysis.normalized_query, *analysis.keywords, *analysis.synonyms])

        if strategy == "dense":
            self.reranker.unload()
            vector = (await self.embedding.embed_dense([analysis.normalized_query]))[0]
            chunks = self.store.dense_search(vector, top_k, effective_category)
        elif strategy == "bm25":
            self.embedding.unload()
            self.reranker.unload()
            chunks = self.store.bm25_search(search_text, top_k, effective_category)
        elif strategy == "hybrid":
            self.reranker.unload()
            vector = (await self.embedding.embed_dense([analysis.normalized_query]))[0]
            chunks = self.store.hybrid_search(vector, search_text, top_k, effective_category)
        elif strategy == "hybrid_rerank":
            self.reranker.unload()
            vector = (await self.embedding.embed_dense([analysis.normalized_query]))[0]
            candidates = self.store.hybrid_search(
                vector,
                search_text,
                max(self.settings.rag_dense_top_k, self.settings.rag_bm25_top_k),
                effective_category,
            )
            self.embedding.unload()
            ranked = await self.reranker.rerank(query, candidates, top_k)
            chunks = [chunk for chunk in ranked if chunk.score >= self.settings.rag_min_rerank_score]
        else:
            raise ValueError(f"unsupported retrieval strategy: {strategy}")

        for index, chunk in enumerate(chunks, start=1):
            chunk.citation_id = index
        return RetrievalResult(query=query, analysis=analysis, strategy=strategy, chunks=chunks)

    async def assess_evidence(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        extra_evidence: str = "",
    ) -> EvidenceAssessment:
        if not chunks and not extra_evidence:
            return EvidenceAssessment(
                sufficient=False,
                confidence=0.0,
                reason="检索结果为空",
                missing="知识库中没有候选证据",
            )
        evidence = "\n\n".join(
            f"[{chunk.citation_id}] {chunk.text}" for chunk in chunks
        )
        if extra_evidence:
            evidence = f"{evidence}\n\n其他工具结果：\n{extra_evidence}".strip()
        prompt = [
            SystemMessage(
                content=(
                    "判断给定检索证据是否足以回答用户问题。只输出 JSON："
                    "sufficient、confidence、reason、missing。"
                    "如果证据能回答问题的核心，即使缺少扩展细节也判 sufficient=true，"
                    "缺失细节写入 missing；如果证据说明答案取决于地区、订单、商品或活动条件，"
                    "只要能够据此向用户解释规则，也判 sufficient=true；"
                    "只有证据与问题主题无关、互相冲突或无法支持核心结论时才判 false。"
                    "证据只能支持明确写出的内容，不能根据常识补全。"
                )
            ),
            HumanMessage(content=f"问题：{query}\n\n证据：\n{evidence}"),
        ]
        try:
            structured = self.model_provider.get().with_structured_output(
                EvidenceAssessment,
                method=self.settings.llm_structured_method,
                include_raw=True,
            )
            result = await structured.ainvoke(prompt)
            parsed = result.get("parsed") if isinstance(result, dict) else result
            if parsed is None:
                raise ValueError("evidence assessment returned no parsed result")
            return parsed if isinstance(parsed, EvidenceAssessment) else EvidenceAssessment.model_validate(parsed)
        except Exception:
            # Retrieval itself remains usable; the failure becomes an explicit low-confidence result.
            return EvidenceAssessment(
                sufficient=False,
                confidence=0.0,
                reason="证据充分性自评失败",
                missing="无法确认现有证据是否足够",
            )

    @staticmethod
    def prompt_order(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        front = chunks[::2]
        back = list(reversed(chunks[1::2]))
        return front + back

    @staticmethod
    def extract_citation_ids(answer: str) -> list[int]:
        return [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
