from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.config import Settings
from app.prompts import CHAT_PROMPT
from app.repositories import SqlAlchemyChatRepository
from app.services.chat import ModelProvider, content_to_text
from app.services.context import trim_history
from app.services.tool_executor import ToolExecution, ToolExecutor
from app.services.tools import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class PreparedToolChat:
    conversation_id: int
    session_id: str
    messages: list[BaseMessage]
    request_id: str
    model: Any
    registry: ToolRegistry
    executor: ToolExecutor
    user_message: str


def _stored_messages_to_langchain(messages: list[Any]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        if message.role == "user":
            converted.append(HumanMessage(content=message.content or ""))
        elif message.role == "assistant":
            if message.tool_call_id and message.tool_name:
                converted.append(
                    AIMessage(
                        content=message.content or "",
                        tool_calls=[
                            {
                                "name": message.tool_name,
                                "args": message.tool_arguments or {},
                                "id": message.tool_call_id,
                            }
                        ],
                    )
                )
            else:
                converted.append(AIMessage(content=message.content or ""))
        elif message.role == "tool":
            converted.append(
                ToolMessage(
                    content=json.dumps(message.tool_result, ensure_ascii=False)
                    if message.tool_result is not None
                    else (message.content or ""),
                    tool_call_id=message.tool_call_id or "",
                    name=message.tool_name,
                )
            )
    return converted


class ToolChatService:
    """Single-step function-calling chat: execute at most one tool, then answer."""

    def __init__(
        self,
        settings: Settings,
        provider: ModelProvider,
        repository: SqlAlchemyChatRepository,
    ):
        self.settings = settings
        self.provider = provider
        self.repository = repository

    async def prepare(self, session_id: str | None, message: str) -> PreparedToolChat:
        model = self.provider.get()
        conversation = await self.repository.ensure_conversation(session_id)
        history_rows = await self.repository.list_messages(conversation.id)
        history = _stored_messages_to_langchain(history_rows)
        await self.repository.add_message(conversation.id, "user", message)
        trimmed = trim_history(history, message, self.settings)
        prompt_value = await CHAT_PROMPT.ainvoke({"history": trimmed, "message": message})
        registry = ToolRegistry.for_conversation(self.repository, conversation.id)
        return PreparedToolChat(
            conversation_id=conversation.id,
            session_id=conversation.session_id,
            messages=prompt_value.to_messages(),
            request_id=f"r_{uuid.uuid4().hex}",
            model=model,
            registry=registry,
            executor=ToolExecutor(registry),
            user_message=message,
        )

    async def stream(
        self,
        prepared: PreparedToolChat,
        _user_message: str | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        yield "meta", {
            "session_id": prepared.session_id,
            "conversation_id": prepared.conversation_id,
            "request_id": prepared.request_id,
        }
        try:
            planner = prepared.model.bind_tools(prepared.registry.tools)
            messages: list[BaseMessage] = list(prepared.messages)
            plan = await planner.ainvoke(messages)
            tool_calls = getattr(plan, "tool_calls", []) or []
            if not tool_calls:
                async for event in self._stream_final_answer(
                    prepared,
                    messages,
                    tool_used=False,
                    tool_steps=0,
                ):
                    yield event
                return
            if len(tool_calls) > 1:
                yield "error", {
                    "code": "multiple_tool_calls",
                    "message": "本轮只允许调用一个工具",
                }
                return

            tool_call = tool_calls[0]
            name = tool_call.get("name", "")
            tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex}")
            arguments = tool_call.get("args", {}) or {}
            tool_call = {**tool_call, "id": tool_call_id, "args": arguments}
            yield "tool_status", {
                "status": "running",
                "step": 1,
                "tool": name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
            }
            await self.repository.add_message(
                prepared.conversation_id,
                "assistant",
                content_to_text(getattr(plan, "content", "")),
                tool_name=name,
                tool_call_id=tool_call_id,
                tool_arguments=arguments,
            )

            execution = await prepared.executor.execute(tool_call)
            await self.repository.add_message(
                prepared.conversation_id,
                "tool",
                content=json.dumps(
                    execution.result if execution.ok else execution.error,
                    ensure_ascii=False,
                ),
                tool_name=name,
                tool_call_id=tool_call_id,
                tool_result=execution.result if execution.ok else execution.error,
            )
            yield "tool_result", self._tool_result_payload(execution)
            messages.extend(
                [
                    AIMessage(
                        content=content_to_text(getattr(plan, "content", "")),
                        tool_calls=[tool_call],
                    ),
                    ToolMessage(
                        content=json.dumps(
                            execution.result if execution.ok else execution.error,
                            ensure_ascii=False,
                        ),
                        tool_call_id=tool_call_id,
                        name=name,
                    ),
                ]
            )
            async for event in self._stream_final_answer(
                prepared,
                messages,
                tool_used=True,
                tool_steps=1,
            ):
                yield event
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("tool chat failed")
            yield "error", {"code": "chat_error", "message": "模型暂时不可用，请稍后重试"}

    async def _stream_final_answer(
        self,
        prepared: PreparedToolChat,
        messages: list[BaseMessage],
        *,
        tool_used: bool,
        tool_steps: int,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        parts: list[str] = []
        async for chunk in prepared.model.astream(messages):
            text = content_to_text(getattr(chunk, "content", ""))
            if not text:
                continue
            parts.append(text)
            yield "delta", {"text": text}
        answer = "".join(parts).strip()
        await self.repository.add_message(prepared.conversation_id, "assistant", answer)
        yield "done", {
            "finish_reason": "stop",
            "tool_used": tool_used,
            "tool_steps": tool_steps,
        }

    @staticmethod
    def _tool_result_payload(execution: ToolExecution) -> dict[str, Any]:
        return {
            "step": 1,
            "tool": execution.name,
            "tool_call_id": execution.tool_call_id,
            "ok": execution.ok,
            "result": execution.result,
            "error": execution.error,
            "attempts": execution.attempts,
            "duration_ms": execution.duration_ms,
        }
