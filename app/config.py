from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ecommerce-customer-service"
    app_env: str = "dev"

    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_max_tokens: int = Field(default=800, ge=1)
    llm_structured_method: str = Field(default="function_calling", pattern="^(json_schema|json_mode|function_calling)$")

    database_url: str = "mysql+asyncmy://app:app_password@127.0.0.1:13306/ecommerce_support?charset=utf8mb4"
    database_echo: bool = False
    tool_max_steps: int = Field(default=3, ge=1, le=10)

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: SecretStr = SecretStr("")
    milvus_collection: str = "knowledge_chunks"
    embedding_model: str = "D:/models/huggingface/bge-m3"
    embedding_device: str = "cpu"
    reranker_model: str = "D:/models/huggingface/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    rag_dense_top_k: int = Field(default=50, ge=1, le=500)
    rag_bm25_top_k: int = Field(default=50, ge=1, le=500)
    rag_rrf_k: int = Field(default=60, ge=1, le=1000)
    rag_final_top_k: int = Field(default=10, ge=1, le=100)
    rag_min_rerank_score: float = Field(default=0.1, ge=0, le=1)
    hf_home: str = "D:/models/huggingface"

    context_window_tokens: int = Field(default=8192, ge=512)
    output_reserved_tokens: int = Field(default=800, ge=1)
    context_safety_tokens: int = Field(default=256, ge=0)
    max_message_chars: int = Field(default=8000, ge=1)
    session_ttl_seconds: int = Field(default=3600, ge=60)
    max_session_count: int = Field(default=1000, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
