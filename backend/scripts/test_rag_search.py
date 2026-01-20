#!/usr/bin/env python3
"""Test if RAG search actually returns results."""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.db.database import get_async_session_factory
from backend.db.models import Chunk
from sqlalchemy import select, func

async def test_rag_status():
    session_factory = get_async_session_factory()
    async with session_factory() as db:
        # Count total chunks
        result = await db.execute(select(func.count(Chunk.id)))
        total_chunks = result.scalar_one()

        # Count chunks with embeddings (required for vector search)
        result = await db.execute(
            select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None))
        )
        chunks_with_embeddings = result.scalar_one()

        print("=" * 70)
        print("RAG SEARCH CAPABILITY TEST")
        print("=" * 70)
        print(f"\nTotal chunks in database:     {total_chunks:>6}")
        print(f"Chunks with embeddings:       {chunks_with_embeddings:>6}")
        print(f"Chunks WITHOUT embeddings:    {total_chunks - chunks_with_embeddings:>6}")

        print("\n" + "=" * 70)
        print("DIAGNOSIS")
        print("=" * 70)

        if chunks_with_embeddings == 0:
            print("\n❌ RAG SEARCH IS **NOT WORKING**")
            print("\nWhy:")
            print("  - Vector/semantic search requires embeddings")
            print("  - 0 chunks have embeddings")
            print("  - Queries will return empty results or fall back to keyword search")
            print("\nWhat you might have seen:")
            print("  - Chat returns generic responses (not using your documents)")
            print("  - Search shows no results")
            print("  - System uses LLM's general knowledge instead of your docs")

            print("\n" + "=" * 70)
            print("TO FIX")
            print("=" * 70)
            print("\nYou MUST re-index documents to generate embeddings:")
            print("\nOption 1: Create a backfill script (RECOMMENDED)")
            print("  I can create a script to generate embeddings for existing chunks")
            print("  This preserves your documents and just adds embeddings")
            print("\nOption 2: Re-upload documents via UI/API")
            print("  Delete and re-upload all documents")
            print("  New embeddings will be generated automatically")

        else:
            coverage = (chunks_with_embeddings / total_chunks) * 100
            print(f"\n✅ RAG SEARCH IS PARTIALLY WORKING ({coverage:.1f}% coverage)")
            print(f"\n  - {chunks_with_embeddings} chunks are searchable")
            print(f"  - {total_chunks - chunks_with_embeddings} chunks are NOT searchable")

        print("\n" + "=" * 70)

if __name__ == "__main__":
    asyncio.run(test_rag_status())
