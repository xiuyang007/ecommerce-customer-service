from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class AfterSalesExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=16000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class AfterSalesExtraction(BaseModel):
    """Only information explicitly present in the customer's description."""

    order_id: str | None = Field(default=None, description="Order number, or null when absent or ambiguous")
    request_type: Literal[
        "refund",
        "return_refund",
        "exchange",
        "repair",
        "other",
        "unknown",
    ] = Field(
        description=(
            "Request category: return_refund means the customer does not want the item or asks to return it "
            "and get money back; refund means a refund request without a clear return context."
        )
    )
    expected_solution: str | None = Field(
        default=None,
        description="The solution explicitly requested by the customer, or null when absent",
    )

    @field_validator("order_id", "expected_solution", mode="before")
    @classmethod
    def normalize_null_like_values(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {
            "",
            "null",
            "none",
            "n/a",
            "na",
            "未提及",
            "无",
            "没有",
        }:
            return None
        return value

    @model_validator(mode="after")
    def normalize_enum_solution(self) -> "AfterSalesExtraction":
        if self.expected_solution:
            canonical = {
                "refund": "退款",
                "return_refund": "退货退款",
                "exchange": "换货",
                "repair": "维修",
            }.get(self.expected_solution.strip().lower())
            if canonical:
                self.expected_solution = canonical
        return self
