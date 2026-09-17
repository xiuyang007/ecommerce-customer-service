from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.prompts import AFTER_SALES_PROMPT, CHAT_PROMPT
from app.schemas import AfterSalesExtraction
from app.services.context import trim_history
from app.store.sessions import InMemorySessionStore, SessionHandle

logger = logging.getLogger(__name__)


class ModelConfigurationError(Exception):
    pass


class StructuredOutputError(Exception):
    pass


class ModelProvider:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._chat_model: ChatOpenAI | None = None

    def get(self) -> ChatOpenAI:
        if not self._settings.llm_api_key.get_secret_value():
            raise ModelConfigurationError("LLM_API_KEY is not configured")
        if self._chat_model is None:
            self._chat_model = ChatOpenAI(
                api_key=self._settings.llm_api_key.get_secret_value(),
                base_url=self._settings.llm_base_url,
                model=self._settings.llm_model,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
        return self._chat_model


@dataclass
class PreparedChat:
    handle: SessionHandle
    messages: list[BaseMessage]
    request_id: str
    model: Any


class ChatService:
    def __init__(self, settings: Settings, sessions: InMemorySessionStore, provider: ModelProvider):
        self.settings = settings
        self.sessions = sessions
        self.provider = provider

    async def prepare(self, session_id: str | None, message: str) -> PreparedChat:
        # Validate configuration before StreamingResponse sends the HTTP headers.
        model = self.provider.get()
        handle = await self.sessions.begin(session_id)
        try:
            history = trim_history(handle.history, message, self.settings)
            prompt_value = await CHAT_PROMPT.ainvoke({"history": history, "message": message})
            return PreparedChat(
                handle=handle,
                messages=prompt_value.to_messages(),
                request_id=f"r_{uuid.uuid4().hex}",
                model=model,
            )
        except Exception:
            await self.sessions.abort(handle)
            raise

    async def stream(
        self,
        prepared: PreparedChat,
        user_message: str,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        parts: list[str] = []
        completed = False
        try:
            yield "meta", {"session_id": prepared.handle.session_id, "request_id": prepared.request_id}
            async for chunk in prepared.model.astream(prepared.messages):
                text = content_to_text(getattr(chunk, "content", ""))
                if not text:
                    continue
                parts.append(text)
                yield "delta", {"text": text}

            answer = "".join(parts).strip()
            await self.sessions.commit(prepared.handle, user_message, answer)
            completed = True
            yield "done", {"finish_reason": "stop"}
        except asyncio.CancelledError:
            await self.sessions.abort(prepared.handle)
            raise
        except Exception:
            await self.sessions.abort(prepared.handle)
            logger.exception("chat stream failed")
            yield "error", {"code": "upstream_error", "message": "模型暂时不可用，请稍后重试"}
        finally:
            if not completed:
                await self.sessions.abort(prepared.handle)

    async def extract_after_sales(self, text: str) -> AfterSalesExtraction:
        model = self.provider.get()
        prompt_value = await AFTER_SALES_PROMPT.ainvoke({"text": text})
        structured = model.with_structured_output(
            AfterSalesExtraction,
            method=self.settings.llm_structured_method,
            include_raw=True,
        )
        result = await structured.ainvoke(prompt_value.to_messages())
        if isinstance(result, dict) and "parsed" in result:
            parsed = result.get("parsed")
            if parsed is not None:
                return parsed if isinstance(parsed, AfterSalesExtraction) else AfterSalesExtraction.model_validate(parsed)
            raw = result.get("raw")
            tool_calls = getattr(raw, "tool_calls", []) or []
            if tool_calls and isinstance(tool_calls[0], dict):
                return AfterSalesExtraction.model_validate(tool_calls[0].get("args", {}))
            raise StructuredOutputError("structured model returned no parsed result or tool call")
        if isinstance(result, AfterSalesExtraction):
            return result
        return AfterSalesExtraction.model_validate(result)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks)
    return ""
