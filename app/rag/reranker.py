from __future__ import annotations

import asyncio
import gc
import os
import threading

from app.config import Settings
from app.rag.schemas import RetrievedChunk


class RerankerService:
    """Lazy bge-reranker-v2-m3 cross-encoder."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        with self._lock:
            if self._model is None:
                os.environ.setdefault("HF_HOME", self.settings.hf_home)
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(
                    self.settings.reranker_model,
                    use_fp16=self.settings.reranker_device != "cpu",
                    devices=[self.settings.reranker_device],
                    max_length=512,
                    normalize=True,
                )
        return self._model

    def _rerank_sync(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = self._load_model()
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = model.compute_score(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        ranked = []
        for chunk, score in zip(chunks, scores, strict=False):
            chunk.score = float(score)
            ranked.append(chunk)
        return sorted(ranked, key=lambda item: item.score, reverse=True)

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        ranked = await asyncio.to_thread(self._rerank_sync, query, chunks)
        return ranked[:top_k]

    def unload(self) -> None:
        self._model = None
        gc.collect()
