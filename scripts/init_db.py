from __future__ import annotations

import asyncio

from app.db import engine, initialize_database


async def main() -> None:
    await initialize_database()
    await engine.dispose()
    print("database_initialized")


if __name__ == "__main__":
    asyncio.run(main())
