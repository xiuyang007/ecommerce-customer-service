from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass

import httpx


@dataclass
class StreamResult:
    tools: list[tuple[str, bool, object]]
    answer: str
    done: dict | None
    sources: list[dict]


def parse_sse(text: str) -> StreamResult:
    event = ""
    tools: list[tuple[str, bool, object]] = []
    parts: list[str] = []
    done: dict | None = None
    sources: list[dict] = []
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            payload = json.loads(line[5:].strip())
            if event == "delta":
                parts.append(payload.get("text", ""))
            elif event == "tool_result":
                tools.append((payload.get("tool", ""), bool(payload.get("ok")), payload.get("result")))
            elif event == "error":
                raise AssertionError(payload.get("message", "stream error"))
            elif event == "done":
                done = payload
            elif event == "sources":
                sources = list(payload.get("citations") or [])
    return StreamResult(tools=tools, answer="".join(parts).strip(), done=done, sources=sources)


async def ask(client: httpx.AsyncClient, message: str, session_id: str | None = None) -> tuple[str, StreamResult]:
    response = await client.post(
        "/api/v1/chat/stream",
        json={"session_id": session_id, "message": message},
    )
    response.raise_for_status()
    result = parse_sse(response.text)
    for line in response.text.splitlines():
        if line.startswith("data:") and '"session_id"' in line:
            return json.loads(line[5:].strip())["session_id"], result
    raise AssertionError("session_id missing from meta event")


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"[PASS] {name}{f' - {detail}' if detail else ''}")


def low_confidence_marker(result: StreamResult) -> bool:
    return bool(result.done and result.done.get("low_confidence") is True)


async def run(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=90) as client:
        session_id, result = await ask(client, "订单 1001 的物流到哪了")
        check("order logistics tool", result.tools and result.tools[0][0] == "query_logistics", str(result.tools))
        check("order logistics result", bool(result.answer), result.answer[:100])
        check("order logistics done", bool(result.done and result.done.get("tool_used") is True))

        _, faq = await ask(client, "退货政策是什么")
        check(
            "faq tool",
            faq.tools and faq.tools[0][0] in {"query_faq", "search_knowledge"},
            str(faq.tools),
        )
        check("faq result", bool(faq.answer) and low_confidence_marker(faq) is False, faq.answer[:100])

        _, miss = await ask(client, "邮费是多少")
        check(
            "shipping fee tool",
            miss.tools and miss.tools[0][0] in {"query_faq", "search_knowledge"},
            str(miss.tools),
        )
        check("shipping fee answer", bool(miss.answer) and low_confidence_marker(miss) is False, miss.answer[:160])

        _, multi = await ask(client, "请先查询订单 1001 的物流，再查询退货政策，最后综合回答。")
        tool_names = [name for name, _, _ in multi.tools]
        check(
            "multi-step tool sequence",
            "query_logistics" in tool_names and "search_knowledge" in tool_names,
            str(multi.tools),
        )
        check(
            "multi-step convergence",
            bool(multi.done and multi.done.get("tool_steps", 0) >= 2 and multi.answer),
            str(multi.done),
        )

        _, rag = await ask(client, "X1001 左耳没声音怎么办")
        rag_tools = [name for name, _, _ in rag.tools]
        check("knowledge retrieval tool", "search_knowledge" in rag_tools, str(rag.tools))
        check("knowledge citations", bool(rag.sources), str(rag.sources[:2]))
        check("knowledge answer", "[1]" in rag.answer, rag.answer[:120])

        _, low = await ask(client, "火星地址支持配送吗")
        check(
            "low-confidence refusal",
            bool(low.done and low.done.get("low_confidence") is True and low.answer),
            str(low.done),
        )

    print("ALL_TOOL_CHECKS_PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 02 tool-call smoke checks")
    parser.add_argument("--base-url", default="http://127.0.0.1:8767")
    args = parser.parse_args()
    asyncio.run(run(args.base_url))


if __name__ == "__main__":
    main()
