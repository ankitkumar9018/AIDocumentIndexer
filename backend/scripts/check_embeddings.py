#!/usr/bin/env python3
"""Check entity embedding status."""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.db.database import get_async_session_factory
from backend.db.models import Entity
from sqlalchemy import select, func

async def check_embeddings():
    session_factory = get_async_session_factory()
    async with session_factory() as db:
        # Count entities with embeddings
        result = await db.execute(
            select(func.count(Entity.id)).where(Entity.embedding.isnot(None))
        )
        count_with = result.scalar_one()

        # Count entities without embeddings
        result2 = await db.execute(
            select(func.count(Entity.id)).where(Entity.embedding.is_(None))
        )
        count_without = result2.scalar_one()

        # Count entities with valid names but no embeddings
        result3 = await db.execute(
            select(func.count(Entity.id)).where(
                Entity.embedding.is_(None),
                Entity.name.isnot(None),
                Entity.name != ''
            )
        )
        count_missing_with_names = result3.scalar_one()

        print(f"Entities WITH embeddings: {count_with}")
        print(f"Entities WITHOUT embeddings: {count_without}")
        print(f"Entities needing backfill (valid names, no embedding): {count_missing_with_names}")

        return count_with, count_without, count_missing_with_names

if __name__ == "__main__":
    asyncio.run(check_embeddings())
