from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass

import httpx


@dataclass
class SseResult:
    session_id: str | None
    answer: str
    events: list[str]


def parse_sse(text: str) -> SseResult:
    current_event = ""
    session_id: str | None = None
    answer_parts: list[str] = []
    events: list[str] = []
    for line in text.splitlines():
        if line.startswith("event:"):
            current_event = line[6:].strip()
            events.append(current_event)
        elif line.startswith("data:"):
            payload = json.loads(line[5:].strip())
            if current_event == "meta":
                session_id = payload.get("session_id")
            elif current_event == "delta":
                answer_parts.append(payload.get("text", ""))
            elif current_event == "error":
                raise AssertionError(payload.get("message", "stream error"))
    return SseResult(session_id=session_id, answer="".join(answer_parts).strip(), events=events)


def check(name: str, condition: bool, details: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed: {details}")
    print(f"[PASS] {name}{f' — {details}' if details else ''}")


async def run(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=90) as client:
        health = await client.get("/health")
        check("健康检查", health.status_code == 200 and health.json().get("status") == "ok")

        page = await client.get("/")
        check(
            "浏览器页面",
            page.status_code == 200
            and "电商智能客服" in page.text
            and "售后提取" in page.text
            and "localStorage" in page.text,
        )

        first_response = await client.post(
            "/api/v1/chat/stream",
            json={"message": "我买的耳机左边没有声音，想换货。请简短回复。"},
        )
        first = parse_sse(first_response.text)
        check(
            "第一轮 SSE",
            first_response.status_code == 200
            and first.session_id is not None
            and first.answer
            and first.events[0] == "meta"
            and "delta" in first.events
            and first.events[-1] == "done",
            f"events={first.events}",
        )

        second_response = await client.post(
            "/api/v1/chat/stream",
            json={
                "session_id": first.session_id,
                "message": "请只回复一句，确认你记得我刚才的换货诉求。",
            },
        )
        second = parse_sse(second_response.text)
        context_signal = any(word in second.answer for word in ("换货", "换新", "更换", "耳机", "记得"))
        check(
            "第二轮上下文",
            second_response.status_code == 200
            and second.session_id == first.session_id
            and second.answer
            and context_signal
            and second.events[-1] == "done",
            f"answer={second.answer[:100]}",
        )

        extraction_response = await client.post(
            "/api/v1/after-sales/extract",
            json={"text": "订单 A202609130001，耳机左边没有声音，我想换一个新的。"},
        )
        extraction = extraction_response.json()
        check(
            "售后结构化提取",
            extraction_response.status_code == 200
            and extraction.get("order_id") == "A202609130001"
            and extraction.get("request_type") == "exchange"
            and "换" in (extraction.get("expected_solution") or ""),
            json.dumps(extraction, ensure_ascii=False),
        )

    print("[PASS] 全自动 API 冒烟验证完成")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local API smoke checks")
    parser.add_argument("--base-url", default="http://127.0.0.1:8767")
    args = parser.parse_args()
    asyncio.run(run(args.base_url))


if __name__ == "__main__":
    main()
