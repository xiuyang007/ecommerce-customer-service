from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.rag.embedding import EmbeddingService
from app.rag.milvus_store import MilvusKnowledgeStore


async def ingest(path: Path, recreate: bool) -> None:
    settings = get_settings()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    embedding = EmbeddingService(settings)
    vectors = await embedding.embed_dense([row["answer"] for row in rows])
    store = MilvusKnowledgeStore(settings)
    store.ensure_collection(len(vectors[0]), recreate=recreate)
    data = [
        {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "text": row["answer"],
            "question": row["question"],
            "category": row["category"],
            "section_path": row["section_path"],
            "source_uri": row["source_uri"],
            "chunk_index": row["chunk_index"],
            "dense_vector": vector,
        }
        for row, vector in zip(rows, vectors, strict=True)
    ]
    store.upsert(data)
    print(f"ingested_chunks={len(data)} collection={settings.milvus_collection}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge chunks into Milvus")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "knowledge" / "chunks.jsonl",
    )
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    asyncio.run(ingest(args.path, args.recreate))


if __name__ == "__main__":
    main()
