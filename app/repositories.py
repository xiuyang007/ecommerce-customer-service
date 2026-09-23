from __future__ import annotations

import secrets
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import Conversation, FAQ, FaithCase, LowConfidenceQuestion, Message, Ticket
from app.store.sessions import UnknownSessionError


class SqlAlchemyChatRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def ensure_conversation(self, session_id: str | None) -> Conversation:
        name = session_id or f"s_{secrets.token_hex(16)}"
        async with self._session_factory() as session:
            existing = await session.scalar(
                select(Conversation).where(Conversation.session_id == name)
            )
            if existing is not None:
                return existing
            if session_id is not None:
                raise UnknownSessionError(session_id)
            conversation = Conversation(session_id=name, status="open")
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return conversation

    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str | None = None,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_arguments: dict | None = None,
        tool_result: dict | list | str | None = None,
    ) -> Message:
        async with self._session_factory() as session:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_arguments=tool_arguments,
                tool_result=tool_result,
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)
            return message

    async def list_messages(self, conversation_id: int) -> list[Message]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id.asc())
            )
            return list(result.all())

    async def query_faq(self, keyword: str, limit: int = 5) -> list[FAQ]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(FAQ)
                .where(
                    FAQ.question.contains(keyword, autoescape=True)
                    | FAQ.answer.contains(keyword, autoescape=True)
                )
                .order_by(FAQ.id.asc())
                .limit(limit)
            )
            return list(result.all())

    async def create_ticket(
        self,
        conversation_id: int,
        description: str,
        ticket_type: str,
    ) -> Ticket:
        async with self._session_factory() as session:
            ticket = Ticket(
                ticket_id=self._new_ticket_id(),
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
                status="pending",
            )
            session.add(ticket)
            await session.commit()
            await session.refresh(ticket)
            return ticket

    async def add_low_confidence_question(
        self,
        *,
        conversation_id: int | None,
        raw_question: str,
        source: str,
        reason: str | None,
    ) -> LowConfidenceQuestion:
        async with self._session_factory() as session:
            row = LowConfidenceQuestion(
                conversation_id=conversation_id,
                raw_question=raw_question,
                source=source,
                reason=reason,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row

    async def upsert_faith_case(
        self,
        *,
        eval_id: str,
        bucket: str,
        query: str,
        strategy: str,
        answer: str,
        reason: str,
        citations: list | None,
        judge_model: str | None,
    ) -> FaithCase:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(FaithCase).where(FaithCase.eval_id == eval_id).with_for_update()
            )
            now = datetime.now()
            if row is None:
                row = FaithCase(
                    eval_id=eval_id,
                    bucket=bucket,
                    query=query,
                    strategy=strategy,
                    answer=answer,
                    reason=reason,
                    citations=citations,
                    judge_model=judge_model,
                    status="未解决",
                    seen_count=1,
                    first_seen_at=now,
                    last_seen_at=now,
                )
                session.add(row)
            else:
                row.bucket = bucket
                row.query = query
                row.strategy = strategy
                row.answer = answer
                row.reason = reason
                row.citations = citations
                row.judge_model = judge_model
                row.seen_count += 1
                row.last_seen_at = now
                row.status = "未解决"
            await session.commit()
            await session.refresh(row)
            return row

    @staticmethod
    def _new_ticket_id() -> str:
        return f"T{datetime.now():%Y%m%d}{secrets.token_hex(4).upper()}"
