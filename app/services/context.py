from __future__ import annotations

import json
import math
from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from app.config import Settings
from app.prompts import CUSTOMER_SERVICE_SYSTEM_PROMPT


class ContextTooLargeError(Exception):
    pass


def estimate_text_tokens(text: str) -> int:
    """Conservative portable estimate; it is not provider billing usage."""

    return max(1, math.ceil(len(text) / 2))


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def estimate_message_tokens(message: BaseMessage) -> int:
    return estimate_text_tokens(message_text(message)) + 4


def trim_history(history: Sequence[BaseMessage], current_message: str, settings: Settings) -> list[BaseMessage]:
    """Keep recent complete user/assistant turns under the configured budget."""

    system_budget = estimate_text_tokens(CUSTOMER_SERVICE_SYSTEM_PROMPT) + 4
    current_budget = estimate_text_tokens(current_message) + 4
    available = (
        settings.context_window_tokens
        - settings.output_reserved_tokens
        - settings.context_safety_tokens
        - system_budget
        - current_budget
    )
    if available <= 0:
        raise ContextTooLargeError("current message exceeds the configured context budget")

    turns: list[tuple[BaseMessage, BaseMessage]] = []
    for index in range(0, len(history) - 1, 2):
        turns.append((history[index], history[index + 1]))

    selected_reversed: list[BaseMessage] = []
    used = 0
    # Stored history is complete turns. Walking backwards ensures newest context wins.
    for human, assistant in reversed(turns):
        cost = estimate_message_tokens(human) + estimate_message_tokens(assistant)
        if used + cost > available:
            break
        selected_reversed.extend([assistant, human])
        used += cost

    return list(reversed(selected_reversed))
