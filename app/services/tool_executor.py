from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from pydantic import BaseModel

from app.services.tools import RegisteredTool, ToolRegistry


class UnknownToolError(Exception):
    pass


@dataclass
class ToolExecution:
    name: str
    tool_call_id: str
    arguments: dict[str, Any]
    ok: bool
    result: Any = None
    error: dict[str, str] | None = None
    attempts: int = 1
    duration_ms: int = 0


def json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(self, tool_call: dict[str, Any]) -> ToolExecution:
        name = tool_call.get("name", "")
        tool_call_id = tool_call.get("id", "")
        arguments = tool_call.get("args", {}) or {}
        registered = self.registry.get(name)
        if registered is None:
            return ToolExecution(
                name=name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                ok=False,
                error={"type": "unknown_tool", "message": f"未知工具: {name}"},
            )

        started = time.perf_counter()
        attempts = 0
        max_attempts = 2 if registered.retryable else 1
        try:
            validated = registered.tool.args_schema.model_validate(arguments)
            payload = validated.model_dump()
        except Exception as exc:
            return ToolExecution(
                name=name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                ok=False,
                error={"type": "validation_error", "message": str(exc)},
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        last_error: Exception | None = None
        while attempts < max_attempts:
            attempts += 1
            try:
                result = await asyncio.wait_for(
                    registered.tool.ainvoke(payload),
                    timeout=registered.timeout_seconds,
                )
                return ToolExecution(
                    name=name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    ok=True,
                    result=json_safe(result),
                    attempts=attempts,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            except Exception as exc:
                last_error = exc
                if attempts >= max_attempts:
                    break

        error_type = "timeout" if isinstance(last_error, TimeoutError) else "execution_error"
        return ToolExecution(
            name=name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            ok=False,
            error={"type": error_type, "message": str(last_error or "工具执行失败")},
            attempts=attempts,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
