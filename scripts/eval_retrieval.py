from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.db import SessionFactory
from app.rag.retrieval import KnowledgeRetrievalService
from app.rag.schemas import FaithfulnessJudgment
from app.repositories import SqlAlchemyChatRepository
from app.services.chat import ModelProvider, content_to_text

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


async def generate_answer(provider: ModelProvider, query: str, chunks) -> str:
    evidence = "\n\n".join(f"[{chunk.citation_id}] {chunk.text}" for chunk in chunks)
    response = await provider.get().ainvoke(
        [
            SystemMessage(
                content=(
                    "只能根据给定证据回答，并用 [n] 标注引用。"
                    "证据不足时必须明确拒答，不能补充证据之外的结论。"
                )
            ),
            HumanMessage(content=f"问题：{query}\n\n证据：\n{evidence or '无'}"),
        ]
    )
    return content_to_text(response.content).strip()


async def judge_faithfulness(provider: ModelProvider, query: str, answer: str, chunks) -> FaithfulnessJudgment:
    evidence = "\n\n".join(f"[{chunk.citation_id}] {chunk.text}" for chunk in chunks)
    structured = provider.get().with_structured_output(
        FaithfulnessJudgment,
        method=get_settings().llm_structured_method,
        include_raw=True,
    )
    result = await structured.ainvoke(
        [
            SystemMessage(content="判断答案中每个事实是否有证据支持，只输出 JSON。"),
            HumanMessage(content=f"问题：{query}\n答案：{answer}\n证据：\n{evidence or '无'}"),
        ]
    )
    parsed = result.get("parsed") if isinstance(result, dict) else result
    return parsed if isinstance(parsed, FaithfulnessJudgment) else FaithfulnessJudgment.model_validate(parsed)


async def evaluate(
    dataset: Path,
    report_path: Path,
    *,
    with_generation: bool,
    top_k: int,
) -> int:
    settings = get_settings()
    provider = ModelProvider(settings)
    retrieval = KnowledgeRetrievalService(settings, provider)
    repository = SqlAlchemyChatRepository(SessionFactory)
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    report: dict = {"strategies": {}, "cases": []}

    analyses = {
        case["eval_id"]: await retrieval.query_understanding.analyze(case["query"])
        for case in cases
    }
    vectors = await retrieval.embedding.embed_dense(
        [analyses[case["eval_id"]].normalized_query for case in cases]
    )
    search_texts = {
        case["eval_id"]: " ".join(
            [
                analyses[case["eval_id"]].normalized_query,
                *analyses[case["eval_id"]].keywords,
                *analyses[case["eval_id"]].synonyms,
            ]
        )
        for case in cases
    }
    categories = {
        case["eval_id"]: (
            analyses[case["eval_id"]].category
            if analyses[case["eval_id"]].confidence >= 0.75
            else None
        )
        for case in cases
    }

    dense_results = {}
    bm25_results = {}
    hybrid_results = {}
    for case, vector in zip(cases, vectors, strict=True):
        eval_id = case["eval_id"]
        dense_results[eval_id] = retrieval.store.dense_search(
            vector, max(top_k, 20), categories[eval_id]
        )
        bm25_results[eval_id] = retrieval.store.bm25_search(
            search_texts[eval_id], max(top_k, 20), categories[eval_id]
        )
        hybrid_results[eval_id] = retrieval.store.hybrid_search(
            vector,
            search_texts[eval_id],
            max(top_k, 20),
            categories[eval_id],
        )
    retrieval.embedding.unload()
    rerank_results = {}
    for case in cases:
        eval_id = case["eval_id"]
        ranked = await retrieval.reranker.rerank(
            case["query"],
            hybrid_results[eval_id],
            max(top_k, 20),
        )
        rerank_results[eval_id] = [
            chunk for chunk in ranked if chunk.score >= settings.rag_min_rerank_score
        ]
    retrieval.reranker.unload()
    result_sets = {
        "dense": dense_results,
        "bm25": bm25_results,
        "hybrid": hybrid_results,
        "hybrid_rerank": rerank_results,
    }

    for strategy in STRATEGIES:
        totals = defaultdict(float)
        buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        count = 0
        for case in cases:
            chunks = result_sets[strategy][case["eval_id"]]
            retrieved = [chunk.chunk_id for chunk in chunks]
            relevant = set(case["ground_truth_chunk_ids"])
            correct_rejection = False
            if not case["should_answer"]:
                assessment = await retrieval.assess_evidence(case["query"], chunks[:top_k])
                correct_rejection = not assessment.sufficient
            row = {
                "eval_id": case["eval_id"],
                "query": case["query"],
                "query_type": case["query_type"],
                "strategy": strategy,
                "retrieved": retrieved,
                "recall@5": recall_at_k(retrieved, relevant, 5),
                "recall@10": recall_at_k(retrieved, relevant, 10),
                "recall@20": recall_at_k(retrieved, relevant, 20),
                "mrr": reciprocal_rank(retrieved, relevant),
                "correct_rejection": correct_rejection,
                "answer": "",
                "faithful": None,
                "judge_reason": "",
            }
            if with_generation:
                row["answer"] = await generate_answer(provider, case["query"], chunks[:top_k])
                judgment = await judge_faithfulness(provider, case["query"], row["answer"], chunks[:top_k])
                row["faithful"] = judgment.faithful
                row["judge_reason"] = judgment.reason
                if not judgment.faithful:
                    await repository.upsert_faith_case(
                        eval_id=case["eval_id"],
                        bucket=case["query_type"],
                        query=case["query"],
                        strategy=strategy,
                        answer=row["answer"],
                        reason=judgment.reason or "; ".join(judgment.violations),
                        citations=[chunk.to_citation() for chunk in chunks[:top_k]],
                        judge_model=settings.llm_model,
                    )
            report["cases"].append(row)
            if case["should_answer"]:
                count += 1
                for metric in ("recall@5", "recall@10", "recall@20", "mrr"):
                    totals[metric] += row[metric]
                    buckets[case["query_type"]][metric] += row[metric]
            else:
                totals["correct_rejection"] += int(row["correct_rejection"])
                buckets[case["query_type"]]["correct_rejection"] += int(row["correct_rejection"])

        metrics = {
            metric: value / max(count, 1)
            for metric, value in totals.items()
            if metric != "correct_rejection"
        }
        if any(not case["should_answer"] for case in cases):
            negative_count = sum(1 for case in cases if not case["should_answer"])
            metrics["correct_rejection"] = totals["correct_rejection"] / negative_count
        report["strategies"][strategy] = {
            "metrics": metrics,
            "buckets": {
                bucket: {
                    metric: value / (
                        sum(
                            1
                            for case in cases
                            if case["query_type"] == bucket and case["should_answer"]
                        )
                        if metric != "correct_rejection"
                        else max(
                            1,
                            sum(
                                1
                                for case in cases
                                if case["query_type"] == bucket and not case["should_answer"]
                            ),
                        )
                    )
                    for metric, value in values.items()
                }
                for bucket, values in buckets.items()
            },
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Chapter 04 Retrieval Report", "", "| strategy | Recall@5 | Recall@10 | Recall@20 | MRR | Faithfulness |", "|---|---:|---:|---:|---:|---:|"]
    for strategy, value in report["strategies"].items():
        metrics = value["metrics"]
        faithful_rows = [row for row in report["cases"] if row["strategy"] == strategy and row["faithful"] is not None]
        faithfulness = (
            sum(1 for row in faithful_rows if row["faithful"]) / len(faithful_rows)
            if faithful_rows
            else None
        )
        lines.append(
            f"| {strategy} | {metrics.get('recall@5', 0):.4f} | {metrics.get('recall@10', 0):.4f} | "
            f"{metrics.get('recall@20', 0):.4f} | {metrics.get('mrr', 0):.4f} | "
            f"{'N/A' if faithfulness is None else f'{faithfulness:.4f}'} |"
        )
    report_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_json={report_path}")
    print(f"report_markdown={report_path.with_suffix('.md')}")
    return 0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate Chapter 04 retrieval strategies")
    parser.add_argument("--dataset", type=Path, default=root / "evals" / "ch04" / "retrieval_queries.jsonl")
    parser.add_argument("--report", type=Path, default=root / "evals" / "reports" / "ch04_retrieval_report.json")
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(evaluate(args.dataset, args.report, with_generation=args.with_generation, top_k=args.top_k)))


if __name__ == "__main__":
    main()
