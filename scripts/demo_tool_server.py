from __future__ import annotations

import json

import uvicorn
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.db import SessionFactory
from app.main import app
from app.repositories import SqlAlchemyChatRepository


class DeterministicToolModel:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        text = str(messages[-1].content)
        if "物流" in text:
            return AIMessage(
                content="",
                tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_demo_logistics"}],
            )
        if "退货政策" in text:
            return AIMessage(
                content="",
                tool_calls=[{"name": "query_faq", "args": {"keyword": "退货政策"}, "id": "call_demo_faq"}],
            )
        if "邮费" in text:
            return AIMessage(
                content="",
                tool_calls=[{"name": "query_faq", "args": {"keyword": "邮费"}, "id": "call_demo_faq_miss"}],
            )
        return AIMessage(content="")

    async def astream(self, messages):
        tool_message = next((message for message in reversed(messages) if isinstance(message, ToolMessage)), None)
        if tool_message is not None:
            try:
                result = json.loads(tool_message.content)
            except Exception:
                result = tool_message.content
            if isinstance(result, list):
                answer = result[0]["answer"] if result else "没有查到对应的常见问题，请以店铺官方规则为准。"
            elif isinstance(result, dict):
                answer = f"订单 {result.get('order_id', '')} 最新物流：{result.get('latest', '暂无信息')}。"
            else:
                answer = "工具没有返回可用结果，请以平台信息为准。"
        else:
            answer = "您好，我可以帮您查询订单、商品、物流和常见问题。"
        for part in answer:
            yield AIMessageChunk(content=part)


class DeterministicProvider:
    def __init__(self):
        self.model = DeterministicToolModel()

    def get(self):
        return self.model


app.state.provider = DeterministicProvider()
app.state.repository = SqlAlchemyChatRepository(SessionFactory)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8767)
