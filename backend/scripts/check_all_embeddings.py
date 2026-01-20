#!/usr/bin/env python3
"""Check embedding status across all tables."""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.db.database import get_async_session_factory
from backend.db.models import Entity, Chunk
from sqlalchemy import select, func, text

async def check_all_embeddings():
    session_factory = get_async_session_factory()
    async with session_factory() as db:
        # Check entities
        result = await db.execute(
            select(func.count(Entity.id)).where(Entity.embedding.isnot(None))
        )
        entities_with = result.scalar_one()

        result = await db.execute(
            select(func.count(Entity.id)).where(Entity.embedding.is_(None))
        )
        entities_without = result.scalar_one()

        # Check chunks
        result = await db.execute(
            select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None))
        )
        chunks_with = result.scalar_one()

        result = await db.execute(
            select(func.count(Chunk.id)).where(Chunk.embedding.is_(None))
        )
        chunks_without = result.scalar_one()

        # Check if documents table has embedding column
        try:
            result = await db.execute(text("""
                SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL
            """))
            docs_with = result.scalar_one()

            result = await db.execute(text("""
                SELECT COUNT(*) FROM documents WHERE embedding IS NULL
            """))
            docs_without = result.scalar_one()
        except Exception:
            # Documents table doesn't have embedding column or doesn't exist
            docs_with = 0
            docs_without = 0

        print("=" * 70)
        print("EMBEDDING STATUS ACROSS ALL TABLES")
        print("=" * 70)
        print(f"\n📊 ENTITIES:")
        print(f"   ✅ With embeddings:    {entities_with:>6}")
        print(f"   ❌ Without embeddings: {entities_without:>6}")
        print(f"   📈 Total:              {entities_with + entities_without:>6}")

        print(f"\n📄 CHUNKS:")
        print(f"   ✅ With embeddings:    {chunks_with:>6}")
        print(f"   ❌ Without embeddings: {chunks_without:>6}")
        print(f"   📈 Total:              {chunks_with + chunks_without:>6}")

        print(f"\n📚 DOCUMENTS:")
        print(f"   ✅ With embeddings:    {docs_with:>6}")
        print(f"   ❌ Without embeddings: {docs_without:>6}")
        print(f"   📈 Total:              {docs_with + docs_without:>6}")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        total_with = entities_with + chunks_with + docs_with
        total_without = entities_without + chunks_without + docs_without
        total = total_with + total_without

        if total > 0:
            percentage = (total_with / total) * 100
        else:
            percentage = 0

        print(f"Total embeddings:     {total_with:>6}")
        print(f"Missing embeddings:   {total_without:>6}")
        print(f"Overall coverage:     {percentage:>5.1f}%")

        print("\n" + "=" * 70)

        if entities_without > 1:  # More than 1 (the empty name entity)
            print("⚠️  ACTION REQUIRED: Run entity backfill")
            print("   python backend/scripts/backfill_entity_embeddings.py")
        else:
            print("✅ Entities: All done!")

        if chunks_without > 0:
            print("⚠️  ACTION REQUIRED: Re-index documents to generate chunk embeddings")
            print("   Documents need to be re-uploaded via API or UI")
        else:
            print("✅ Chunks: All done!")

        if docs_without > 0:
            print("⚠️  ACTION REQUIRED: Re-index documents to generate document embeddings")
            print("   Documents need to be re-uploaded via API or UI")
        else:
            print("✅ Documents: All done!")

        print("=" * 70)

if __name__ == "__main__":
    asyncio.run(check_all_embeddings())
