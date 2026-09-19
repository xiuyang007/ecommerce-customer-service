from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models import Base, FAQ

settings = get_settings()
engine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_pre_ping=True,
)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        yield session


async def initialize_database() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with SessionFactory() as session:
        existing = await session.scalar(select(FAQ.id).limit(1))
        if existing is None:
            session.add_all(
                [
                    FAQ(question="退货政策是什么", answer="签收后 7 天内可申请退货，商品需保持完好并符合平台规则。", category="policy"),
                    FAQ(question="如何申请换货", answer="请在订单详情中提交换货申请，并提供订单号、商品问题和期望方案。", category="after_sales"),
                    FAQ(question="发票怎么开", answer="可在订单详情中申请电子发票，具体开票时间以平台规则为准。", category="invoice"),
                    FAQ(question="物流多久到货", answer="常规地区通常 2 到 5 天送达，偏远地区时间可能更长。", category="logistics"),
                    FAQ(question="退款多久到账", answer="退款审核通过后通常 1 到 7 个工作日到账，具体以支付渠道为准。", category="refund"),
                ]
            )
            await session.commit()
