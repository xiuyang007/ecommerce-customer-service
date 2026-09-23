from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, inspect as sqlalchemy_inspect, select
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.db import SessionFactory
from app.models import Conversation, FAQ, FaithCase, LowConfidenceQuestion, Message, Ticket
from app.rag.retrieval import KnowledgeRetrievalService
from app.repositories import SqlAlchemyChatRepository
from app.schemas import AfterSalesExtractRequest, ChatRequest
from app.services.chat import ChatService, ModelConfigurationError, ModelProvider
from app.services.context import ContextTooLargeError
from app.services.sse import as_sse
from app.services.tool_chat import ToolChatService
from app.store.sessions import InMemorySessionStore, SessionBusyError, UnknownSessionError

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")
app.state.sessions = InMemorySessionStore(settings)
app.state.provider = ModelProvider(settings)
app.state.repository = SqlAlchemyChatRepository(SessionFactory)
app.state.retrieval_service = KnowledgeRetrievalService(settings, app.state.provider)


def get_chat_service(request: Request):
    if request.app.state.repository is not None:
        return ToolChatService(
            settings,
            request.app.state.provider,
            request.app.state.repository,
            request.app.state.retrieval_service,
        )
    return ChatService(settings, request.app.state.sessions, request.app.state.provider)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.post("/api/v1/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    service = get_chat_service(request)
    try:
        prepared = await service.prepare(payload.session_id, payload.message)
    except UnknownSessionError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found") from exc
    except SessionBusyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="session is busy") from exc
    except ContextTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except ModelConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.exception("database unavailable")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="database unavailable") from exc

    return StreamingResponse(
        as_sse(service.stream(prepared, payload.message)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/after-sales/extract")
async def extract_after_sales(payload: AfterSalesExtractRequest, request: Request) -> JSONResponse:
    service = ChatService(settings, request.app.state.sessions, request.app.state.provider)
    try:
        result = await service.extract_after_sales(payload.text)
    except ModelConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("after-sales extraction failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="structured extraction failed",
        ) from exc
    return JSONResponse(result.model_dump())


def _serialize_database_value(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ")
    if isinstance(value, Decimal):
        return str(value)
    return value


@app.get("/api/v1/database/tables")
async def database_tables(limit: int = Query(default=20, ge=1, le=100)) -> JSONResponse:
    models = (FAQ, Conversation, Message, Ticket, LowConfidenceQuestion, FaithCase)
    tables = []
    try:
        async with SessionFactory() as session:
            for model in models:
                primary_key = sqlalchemy_inspect(model).primary_key[0]
                total = await session.scalar(select(func.count()).select_from(model))
                rows = (
                    await session.scalars(
                        select(model).order_by(primary_key.desc()).limit(limit)
                    )
                ).all()
                column_names = list(sqlalchemy_inspect(model).columns.keys())
                tables.append(
                    {
                        "name": model.__tablename__,
                        "columns": column_names,
                        "total": total or 0,
                        "rows": [
                            {
                                column: _serialize_database_value(getattr(row, column))
                                for column in column_names
                            }
                            for row in rows
                        ],
                    }
                )
    except SQLAlchemyError as exc:
        logger.exception("database table preview failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="database unavailable") from exc
    return JSONResponse({"tables": tables})


# Serve the lightweight browser client after API routes so /api/* keeps priority.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="frontend")
