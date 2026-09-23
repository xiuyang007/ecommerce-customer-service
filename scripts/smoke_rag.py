from __future__ import annotations

import argparse
import asyncio

from app.config import get_settings
from app.rag.retrieval import KnowledgeRetrievalService
from app.services.chat import ModelProvider


async def run() -> None:
    settings = get_settings()
    service = KnowledgeRetrievalService(settings, ModelProvider(settings))

    bm25 = await service.search("X1001 左耳没声音怎么办", strategy="bm25", limit=10)
    assert "product-x1001#1" in [chunk.chunk_id for chunk in bm25.chunks]
    print("[PASS] BM25 exact model recall")

    hybrid = await service.search("X1001 左耳没声音怎么办", strategy="hybrid_rerank", limit=10)
    assert "product-x1001#1" in [chunk.chunk_id for chunk in hybrid.chunks]
    assert all(chunk.citation_id > 0 for chunk in hybrid.chunks)
    print("[PASS] hybrid + rerank citations")

    filtered = await service.search(
        "左耳没声音怎么办",
        category="product",
        strategy="bm25",
        limit=10,
    )
    assert filtered.chunks and all(chunk.category == "product" for chunk in filtered.chunks)
    print("[PASS] metadata category filter")

    negative = await service.search("火星地址支持配送吗", strategy="hybrid_rerank", limit=10)
    assessment = await service.assess_evidence("火星地址支持配送吗", negative.chunks)
    assert assessment.sufficient is False
    print("[PASS] low-confidence rejection gate")
    print("ALL_RAG_CHECKS_PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 04 RAG smoke checks")
    parser.parse_args()
    asyncio.run(run())


if __name__ == "__main__":
    main()
