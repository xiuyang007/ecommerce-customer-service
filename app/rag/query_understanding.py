from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings
from app.rag.schemas import QueryAnalysis

logger = logging.getLogger(__name__)


class QueryUnderstandingService:
    def __init__(self, model_provider, settings: Settings):
        self.model_provider = model_provider
        self.settings = settings

    async def analyze(self, query: str) -> QueryAnalysis:
        prompt = [
            SystemMessage(
                content=(
                    "你是电商客服检索系统的查询理解模块。只输出 JSON，不要回答用户问题。"
                    "将口语问题改写为标准问法，提取 BM25 关键词，并给只用于检索的同义词。"
                    "normalized_query 必须保持原问题范围，只做去口语和同义归一，不得加入原问题未提及的主题。"
                    "category 只能是 policy、logistics、refund、after_sales、invoice、product 或 null；"
                    "具体型号、参数或产品故障使用 product；只有用户明确问退换修流程时才用 after_sales；"
                    "无法高置信判断时返回 null。"
                )
            ),
            HumanMessage(content=query),
        ]
        try:
            response = await self.model_provider.get().ainvoke(prompt)
            content = response.content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:].strip()
            return QueryAnalysis.model_validate(json.loads(content))
        except Exception:
            logger.exception("query understanding failed; using raw query fallback")
            return QueryAnalysis(
                normalized_query=query,
                keywords=[query],
                synonyms=[],
                category=None,
                confidence=0.3,
            )
