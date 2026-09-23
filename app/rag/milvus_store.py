from __future__ import annotations

import re

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from app.config import Settings
from app.rag.schemas import RetrievedChunk

SAFE_FILTER_VALUE = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]+$")


class MilvusKnowledgeStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.token = settings.milvus_token.get_secret_value() or None
        self._client_instance = None

    def _client(self) -> MilvusClient:
        if self._client_instance is None:
            self._client_instance = MilvusClient(uri=self.settings.milvus_uri, token=self.token)
        return self._client_instance

    def ensure_collection(self, dimension: int, recreate: bool = False) -> None:
        name = self.settings.milvus_collection
        client = self._client()
        if client.has_collection(name):
            if not recreate:
                client.load_collection(name)
                return
            client.drop_collection(name)

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field("document_id", DataType.VARCHAR, max_length=128)
        schema.add_field("text", DataType.VARCHAR, max_length=8192, enable_analyzer=True, analyzer_params={"type": "chinese"})
        schema.add_field("question", DataType.VARCHAR, max_length=512)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("section_path", DataType.JSON)
        schema.add_field("source_uri", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
        )
        client.create_collection(collection_name=name, schema=schema, index_params=index_params)
        client.load_collection(name)

    def upsert(self, rows: list[dict]) -> None:
        if rows:
            client = self._client()
            client.upsert(collection_name=self.settings.milvus_collection, data=rows)
            client.flush(self.settings.milvus_collection)

    def dense_search(self, vector: list[float], limit: int, category: str | None) -> list[RetrievedChunk]:
        result = self._client().search(
            collection_name=self.settings.milvus_collection,
            data=[vector],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE"},
            limit=limit,
            filter=self._filter_expr(category),
            output_fields=self._output_fields(),
        )
        return self._parse_hits(result)

    def bm25_search(self, query: str, limit: int, category: str | None) -> list[RetrievedChunk]:
        result = self._client().search(
            collection_name=self.settings.milvus_collection,
            data=[query],
            anns_field="sparse_vector",
            search_params={"metric_type": "BM25"},
            limit=limit,
            filter=self._filter_expr(category),
            output_fields=self._output_fields(),
        )
        return self._parse_hits(result)

    def hybrid_search(
        self,
        vector: list[float],
        query: str,
        limit: int,
        category: str | None,
    ) -> list[RetrievedChunk]:
        expr = self._filter_expr(category)
        dense_req = AnnSearchRequest(
            data=[vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit,
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=limit,
            expr=expr,
        )
        result = self._client().hybrid_search(
            collection_name=self.settings.milvus_collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=self.settings.rag_rrf_k),
            limit=limit,
            output_fields=self._output_fields(),
        )
        return self._parse_hits(result)

    def delete_by_source(self, source_uri: str) -> None:
        self._client().delete(
            collection_name=self.settings.milvus_collection,
            filter=self._filter_expr(source_uri, field="source_uri"),
        )

    @staticmethod
    def _output_fields() -> list[str]:
        return [
            "chunk_id",
            "document_id",
            "text",
            "question",
            "category",
            "section_path",
            "source_uri",
            "chunk_index",
        ]

    @staticmethod
    def _filter_expr(value: str | None, field: str = "category") -> str:
        if not value:
            return ""
        if not SAFE_FILTER_VALUE.match(value):
            raise ValueError(f"unsafe metadata filter value: {value}")
        return f'{field} == "{value}"'

    @staticmethod
    def _parse_hits(result) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for hits in result:
            for hit in hits:
                entity = hit.get("entity", {})
                chunks.append(
                    RetrievedChunk(
                        chunk_id=entity.get("chunk_id", ""),
                        document_id=entity.get("document_id", ""),
                        text=entity.get("text", ""),
                        question=entity.get("question") or None,
                        category=entity.get("category"),
                        section_path=list(entity.get("section_path") or []),
                        source_uri=entity.get("source_uri"),
                        chunk_index=int(entity.get("chunk_index") or 0),
                        score=float(hit.get("distance") or 0.0),
                    )
                )
        return chunks
