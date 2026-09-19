from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from langchain_core.tools import BaseTool, tool

from app.repositories import SqlAlchemyChatRepository


def _seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _order_data(order_id: str) -> dict:
    product_names = ("无线耳机", "机械键盘", "保温杯", "运动鞋")
    statuses = ("paid", "shipped", "delivered")
    status = statuses[_seed(f"order:{order_id}") % len(statuses)]
    return {
        "order_id": order_id,
        "status": status,
        "product_name": product_names[_seed(f"product:{order_id}") % len(product_names)],
        "amount": f"{199 + _seed(f'amount:{order_id}') % 1800}.00",
        "created_at": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
    }


@dataclass(frozen=True)
class RegisteredTool:
    tool: BaseTool
    timeout_seconds: float
    retryable: bool


class ToolRegistry:
    def __init__(self, tools: list[RegisteredTool]):
        self._tools = {registered.tool.name: registered for registered in tools}

    @property
    def tools(self) -> list[BaseTool]:
        return [registered.tool for registered in self._tools.values()]

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    @classmethod
    def for_conversation(
        cls,
        repository: SqlAlchemyChatRepository,
        conversation_id: int,
    ) -> "ToolRegistry":
        @tool
        async def query_order(order_id: str) -> dict:
            """查询订单基础信息，参数是订单号。"""
            return _order_data(order_id)

        @tool
        async def query_product(product_id: str) -> dict:
            """查询商品基础信息，参数是商品编号。"""
            names = ("无线耳机", "机械键盘", "保温杯", "运动鞋")
            return {
                "product_id": product_id,
                "name": names[_seed(f"name:{product_id}") % len(names)],
                "price": f"{99 + _seed(f'price:{product_id}') % 900}.00",
                "stock": 5 + _seed(f"stock:{product_id}") % 100,
                "on_sale": True,
            }

        @tool
        async def query_logistics(order_id: str) -> dict:
            """查询订单物流轨迹，参数是订单号。"""
            states = ("in_transit", "delivered", "out_for_delivery")
            state = states[_seed(f"logistics:{order_id}") % len(states)]
            latest = {
                "in_transit": "包裹已到达武汉转运中心",
                "delivered": "包裹已签收",
                "out_for_delivery": "包裹正在派送中",
            }[state]
            now = datetime.now()
            return {
                "order_id": order_id,
                "carrier": "顺丰速运",
                "tracking_number": f"SF{1000000000 + _seed(order_id) % 8999999999}",
                "status": state,
                "latest": latest,
                "timeline": [
                    {"time": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M"), "event": "已发货"},
                    {"time": now.strftime("%Y-%m-%d %H:%M"), "event": latest},
                ],
            }

        @tool
        async def query_faq(keyword: str) -> list[dict]:
            """按关键词查询常见问题，参数是用户问题中的关键词。"""
            rows = await repository.query_faq(keyword)
            return [
                {"question": row.question, "answer": row.answer, "category": row.category}
                for row in rows
            ]

        @tool
        async def create_ticket(
            description: str,
            ticket_type: Literal[
                "refund",
                "return_refund",
                "exchange",
                "repair",
                "other",
            ],
        ) -> dict:
            """创建人工工单，参数是问题描述和工单类型。"""
            ticket = await repository.create_ticket(
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
            return {
                "ticket_id": ticket.ticket_id,
                "ticket_type": ticket.ticket_type,
                "status": ticket.status,
            }

        return cls(
            [
                RegisteredTool(query_order, timeout_seconds=6, retryable=True),
                RegisteredTool(query_product, timeout_seconds=6, retryable=True),
                RegisteredTool(query_logistics, timeout_seconds=6, retryable=True),
                RegisteredTool(query_faq, timeout_seconds=6, retryable=True),
                RegisteredTool(create_ticket, timeout_seconds=6, retryable=False),
            ]
        )
