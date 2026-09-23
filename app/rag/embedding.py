from __future__ import annotations

import asyncio
import gc
import os
import threading

from app.config import Settings


class EmbeddingService:
    """Lazy bge-m3 dense encoder. Model loading is kept out of module import."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        with self._lock:
            if self._model is None:
                os.environ.setdefault("HF_HOME", self.settings.hf_home)
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(
                    self.settings.embedding_model,
                    use_fp16=self.settings.embedding_device != "cpu",
                    devices=[self.settings.embedding_device],
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
        return self._model

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        result = model.encode(texts, batch_size=1, max_length=512)
        return [list(map(float, vector)) for vector in result["dense_vecs"]]

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode_sync, texts)

    def unload(self) -> None:
        self._model = None
        gc.collect()
