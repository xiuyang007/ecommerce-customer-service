from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.schemas import AfterSalesExtractRequest, ChatRequest
from app.services.chat import ChatService, ModelConfigurationError, ModelProvider
from app.services.context import ContextTooLargeError
from app.services.sse import as_sse
from app.store.sessions import InMemorySessionStore, SessionBusyError, UnknownSessionError

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")
app.state.sessions = InMemorySessionStore(settings)
app.state.provider = ModelProvider(settings)


def get_chat_service(request: Request) -> ChatService:
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
    service = get_chat_service(request)
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


WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="frontend")
