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


def parse_sse(text: str) -> StreamResult:
    event = ""
    tools: list[tuple[str, bool, object]] = []
    parts: list[str] = []
    done: dict | None = None
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
    return StreamResult(tools=tools, answer="".join(parts).strip(), done=done)


async def ask(
    client: httpx.AsyncClient,
    message: str,
    session_id: str | None = None,
) -> tuple[str, StreamResult]:
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


async def run(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=90) as client:
        _, logistics = await ask(client, "订单 1001 的物流到哪了")
        check(
            "物流工具",
            bool(logistics.tools and logistics.tools[0][0] == "query_logistics"),
            str(logistics.tools),
        )
        check("物流回答", bool(logistics.answer), logistics.answer[:100])
        check(
            "单轮收敛",
            bool(logistics.done and logistics.done.get("tool_used") is True),
            str(logistics.done),
        )

        _, faq = await ask(client, "退货政策是什么")
        check("FAQ 工具", bool(faq.tools and faq.tools[0][0] == "query_faq"), str(faq.tools))
        check("FAQ 回答", bool(faq.answer), faq.answer[:100])

        _, miss = await ask(client, "邮费是多少")
        check("漏召回工具", bool(miss.tools and miss.tools[0][0] == "query_faq"), str(miss.tools))
        check(
            "漏召回按空结果作答",
            bool(miss.answer and miss.tools[0][2] == []),
            f"{miss.tools} answer={miss.answer[:100]}",
        )

        _, order = await ask(client, "查询订单 1001")
        check("订单工具", bool(order.tools and order.tools[0][0] == "query_order"), str(order.tools))

        _, product = await ask(client, "查询商品 P1001")
        check(
            "商品工具",
            bool(product.tools and product.tools[0][0] == "query_product"),
            str(product.tools),
        )

        _, ticket = await ask(
            client,
            "请创建人工工单，问题描述是耳机左边没有声音，工单类型是 exchange。",
        )
        check(
            "工单工具",
            bool(ticket.tools and ticket.tools[0][0] == "create_ticket"),
            str(ticket.tools),
        )

        tables_response = await client.get("/api/v1/database/tables?limit=5")
        tables_response.raise_for_status()
        tables = {item["name"]: item for item in tables_response.json()["tables"]}
        check(
            "工具链路落库",
            tables["conversations"]["total"] >= 1
            and tables["messages"]["total"] >= 1
            and tables["tickets"]["total"] >= 1,
            str({name: item["total"] for name, item in tables.items()}),
        )

    print("ALL_TOOL_CHECKS_PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 02 tool-call smoke checks")
    parser.add_argument("--base-url", default="http://127.0.0.1:8767")
    args = parser.parse_args()
    asyncio.run(run(args.base_url))


if __name__ == "__main__":
    main()
