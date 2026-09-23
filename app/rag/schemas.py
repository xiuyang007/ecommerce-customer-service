from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QueryAnalysis(BaseModel):
    normalized_query: str = Field(description="标准化后的标准问法")
    keywords: list[str] = Field(default_factory=list, description="用于 BM25 的关键词")
    synonyms: list[str] = Field(default_factory=list, description="仅用于检索扩展的同义词")
    category: str | None = Field(default=None, description="可选的品类过滤条件")
    confidence: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_keys(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if not data.get("normalized_query"):
            for key in (
                "standard_query",
                "query",
                "normalized",
                "original_query",
                "rewritten_query",
                "rewritten",
                "query_rewrite",
            ):
                if data.get(key):
                    data["normalized_query"] = data[key]
                    break
        data.setdefault("keywords", [])
        data.setdefault("synonyms", [])
        if isinstance(data["keywords"], str):
            data["keywords"] = [data["keywords"]]
        elif isinstance(data["keywords"], dict):
            data["keywords"] = [str(key) for key in data["keywords"]]
        if isinstance(data["synonyms"], dict):
            flattened = []
            for key, values in data["synonyms"].items():
                flattened.append(str(key))
                if isinstance(values, list):
                    flattened.extend(str(value) for value in values)
                else:
                    flattened.append(str(values))
            data["synonyms"] = flattened
        elif isinstance(data["synonyms"], str):
            data["synonyms"] = [data["synonyms"]]
        return data


class EvidenceAssessment(BaseModel):
    sufficient: bool = Field(description="现有证据是否足以回答")
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason: str = Field(default="", description="判断依据")
    missing: str | None = Field(default=None, description="缺少的信息")

    @model_validator(mode="before")
    @classmethod
    def normalize_assessment_keys(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "sufficient" not in data:
            for key in ("evidence_sufficient", "enough", "is_sufficient"):
                if key in data:
                    data["sufficient"] = data[key]
                    break
        if isinstance(data.get("missing"), list):
            data["missing"] = "、".join(str(item) for item in data["missing"])
        return data


class FaithfulnessJudgment(BaseModel):
    faithful: bool = Field(description="答案是否完全由给定证据支持")
    violations: list[str] = Field(default_factory=list, description="证据不足或编造的句子")
    reason: str = Field(default="", description="裁判理由")


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    category: str | None
    section_path: list[str]
    source_uri: str | None
    chunk_index: int = 0
    question: str | None = None
    score: float = 0.0
    citation_id: int = 0

    def to_citation(self) -> dict:
        return {
            "n": self.citation_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "section_path": self.section_path,
            "question": self.question,
            "answer": self.text,
            "source_uri": self.source_uri,
            "category": self.category,
            "score": self.score,
        }

    @classmethod
    def from_citation(cls, value: dict) -> "RetrievedChunk":
        return cls(
            chunk_id=str(value.get("chunk_id") or ""),
            document_id=str(value.get("document_id") or ""),
            text=str(value.get("answer") or ""),
            question=value.get("question"),
            category=value.get("category"),
            section_path=list(value.get("section_path") or []),
            source_uri=value.get("source_uri"),
            score=float(value.get("score") or 0.0),
            citation_id=int(value.get("n") or 0),
        )


@dataclass
class RetrievalResult:
    query: str
    analysis: QueryAnalysis
    strategy: str
    chunks: list[RetrievedChunk] = field(default_factory=list)

    def to_tool_payload(self) -> dict:
        return {
            "query": self.query,
            "normalized_query": self.analysis.normalized_query,
            "strategy": self.strategy,
            "citation_count": len(self.chunks),
            "citations": [chunk.to_citation() for chunk in self.chunks],
        }


RetrievalStrategy = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
