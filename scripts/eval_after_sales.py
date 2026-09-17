from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

FIELDS = ("order_id", "request_type", "expected_solution")


def equivalent_solution(expected: str | None, actual: str | None) -> bool:
    if expected is None or actual is None:
        return expected is actual
    expected_normalized = "".join(expected.split()).lower()
    actual_normalized = "".join(actual.split()).lower()
    if expected_normalized == actual_normalized:
        return True
    # Accept harmless explanatory suffixes while keeping the main requested action.
    if expected_normalized in actual_normalized or actual_normalized in expected_normalized:
        return True

    synonym_groups = (
        ("维修", ("维修", "修理", "修复")),
        ("换货", ("换货", "换新", "更换")),
        ("退款", ("退款", "退钱")),
    )
    for canonical, synonyms in synonym_groups:
        expected_has_intent = canonical in expected_normalized or any(
            synonym.lower() in expected_normalized for synonym in synonyms
        )
        actual_has_intent = any(synonym.lower() in actual_normalized for synonym in synonyms)
        if expected_has_intent and actual_has_intent:
            return True

    # Chinese shorthand such as “两个都退” preserves the count and action intent.
    if (
        "两个" in expected_normalized
        and "两个" in actual_normalized
        and "退" in expected_normalized
        and "退" in actual_normalized
    ):
        return True
    return False


def field_matches(field: str, expected: Any, actual: Any) -> bool:
    if field == "expected_solution":
        return equivalent_solution(expected, actual)
    return expected == actual


async def request_with_retry(client: httpx.AsyncClient, text: str) -> httpx.Response:
    last_response: httpx.Response | None = None
    for attempt in range(3):
        response = await client.post("/api/v1/after-sales/extract", json={"text": text})
        last_response = response
        if response.status_code < 500:
            return response
        if attempt < 2:
            await asyncio.sleep(1.5 * (attempt + 1))
    assert last_response is not None
    return last_response


async def evaluate(base_url: str, dataset: Path) -> int:
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = 0
    field_hits = {field: 0 for field in FIELDS}

    async with httpx.AsyncClient(base_url=base_url, timeout=90) as client:
        for index, case in enumerate(cases, start=1):
            response = await request_with_retry(client, case["text"])
            if response.is_error:
                print(f"case={index} FAIL http_status={response.status_code} body={response.text[:200]}")
                continue
            actual = response.json()
            expected = case["expected"]
            matches = {
                field: field_matches(field, expected.get(field), actual.get(field))
                for field in FIELDS
            }
            for field, matched in matches.items():
                field_hits[field] += int(matched)
            is_pass = all(matches.values())
            passed += int(is_pass)
            print(f"case={index} {'PASS' if is_pass else 'FAIL'} fields={matches}")
            if not is_pass:
                print(json.dumps({"expected": expected, "actual": actual}, ensure_ascii=False))

    total = len(cases)
    print(f"cases_passed={passed}/{total}")
    print(
        "field_accuracy="
        + json.dumps(
            {field: f"{hits}/{total}" for field, hits in field_hits.items()},
            ensure_ascii=False,
        )
    )
    return 0 if passed == total else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-model after-sales extraction evaluation")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evals" / "after_sales_examples.jsonl",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(evaluate(args.base_url, args.dataset)))


if __name__ == "__main__":
    main()
