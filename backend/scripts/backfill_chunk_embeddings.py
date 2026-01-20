#!/usr/bin/env python3
"""
Backfill embeddings for chunks that don't have them.

This script generates embeddings for all chunks with NULL embeddings using
the configured embedding provider (Ollama, OpenAI, HuggingFace, etc.).

Usage:
    python backend/scripts/backfill_chunk_embeddings.py [--dry-run] [--batch-size N]
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv(project_root / ".env")

from backend.db.database import get_async_session_factory
from backend.db.models import Chunk
from backend.services.embeddings import EmbeddingService
from sqlalchemy import select, func
import structlog

logger = structlog.get_logger()


async def backfill_chunk_embeddings(
    dry_run: bool = False,
    batch_size: int = 50
):
    """
    Generate embeddings for all chunks without them.

    Args:
        dry_run: If True, only count chunks without embeddings
        batch_size: Number of chunks to process per batch
    """
    session_factory = get_async_session_factory()

    # Get embedding provider from environment
    embedding_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")

    # For Ollama, prefer OLLAMA_EMBEDDING_MODEL over DEFAULT_EMBEDDING_MODEL
    if embedding_provider.lower() == "ollama":
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", None)
        if not embedding_model:
            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", None)
        logger.info("Using Ollama embedding model", model=embedding_model)
    else:
        embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", None)

    logger.info(
        "Embedding configuration",
        provider=embedding_provider,
        model=embedding_model or "default"
    )

    # Create embedding service
    embedding_service = EmbeddingService(
        provider=embedding_provider,
        model=embedding_model
    )

    # Get embedding dimension
    test_embedding = embedding_service.embed_texts(["test"])
    embedding_dim = len(test_embedding[0]) if test_embedding else 0
    logger.info(f"Embedding dimension: {embedding_dim}D")

    async with session_factory() as db:
        # Count chunks without embeddings
        result = await db.execute(
            select(func.count(Chunk.id)).where(
                Chunk.embedding.is_(None)
            )
        )
        total_without_embeddings = result.scalar_one()

        # Count chunks with embeddings
        result = await db.execute(
            select(func.count(Chunk.id)).where(
                Chunk.embedding.isnot(None)
            )
        )
        total_with_embeddings = result.scalar_one()

        total_chunks = total_without_embeddings + total_with_embeddings

        print("\n" + "=" * 70)
        print("CHUNK EMBEDDING BACKFILL")
        print("=" * 70)
        print(f"Provider:                     {embedding_provider}")
        print(f"Model:                        {embedding_model or 'default'}")
        print(f"Embedding dimension:          {embedding_dim}D")
        print(f"\nTotal chunks:                 {total_chunks:>6}")
        print(f"Chunks with embeddings:       {total_with_embeddings:>6}")
        print(f"Chunks WITHOUT embeddings:    {total_without_embeddings:>6}")

        if dry_run:
            print("\n" + "=" * 70)
            print("DRY RUN - No changes will be made")
            print("=" * 70)
            return

        if total_without_embeddings == 0:
            print("\n✅ All chunks already have embeddings!")
            print("=" * 70)
            return

        # Ask for confirmation
        print("\n" + "=" * 70)
        print("CONFIRMATION")
        print("=" * 70)
        print(f"This will generate {embedding_dim}D embeddings for {total_without_embeddings} chunks")
        print(f"using {embedding_provider} ({embedding_model or 'default model'})")
        print(f"\nProcessing will happen in batches of {batch_size} chunks.")

        confirm = input("\nProceed? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

        print("\n" + "=" * 70)
        print("PROCESSING")
        print("=" * 70)

        processed_count = 0
        error_count = 0

        while True:
            # Get batch of chunks without embeddings
            result = await db.execute(
                select(Chunk).where(
                    Chunk.embedding.is_(None)
                ).limit(batch_size)
            )
            chunks: List[Chunk] = list(result.scalars().all())

            if not chunks:
                break

            try:
                # Extract texts from chunks
                texts = [chunk.content for chunk in chunks]

                # Generate embeddings
                logger.info(f"Generating embeddings for batch of {len(chunks)} chunks...")
                embeddings = embedding_service.embed_texts(texts)

                if len(embeddings) != len(chunks):
                    logger.error(
                        f"Embedding count mismatch: got {len(embeddings)}, expected {len(chunks)}"
                    )
                    error_count += len(chunks)
                    continue

                # Update chunks with embeddings
                for chunk, embedding in zip(chunks, embeddings):
                    if embedding and len(embedding) == embedding_dim:
                        chunk.embedding = embedding
                    else:
                        logger.warning(
                            f"Invalid embedding for chunk {chunk.id}",
                            expected_dim=embedding_dim,
                            got_dim=len(embedding) if embedding else 0
                        )
                        error_count += 1

                # Commit batch
                await db.commit()

                processed_count += len(chunks)
                progress = (processed_count / total_without_embeddings) * 100
                print(f"Progress: {processed_count}/{total_without_embeddings} ({progress:.1f}%)")

            except Exception as e:
                logger.error(
                    "Error processing batch",
                    error=str(e),
                    batch_size=len(chunks)
                )
                await db.rollback()
                error_count += len(chunks)

                # Continue to next batch
                continue

        print("\n" + "=" * 70)
        print("COMPLETE")
        print("=" * 70)
        print(f"Successfully processed: {processed_count} chunks")
        if error_count > 0:
            print(f"Errors encountered:     {error_count} chunks")
        print("=" * 70)

        # Verify final state
        result = await db.execute(
            select(func.count(Chunk.id)).where(
                Chunk.embedding.isnot(None)
            )
        )
        final_with_embeddings = result.scalar_one()

        result = await db.execute(
            select(func.count(Chunk.id)).where(
                Chunk.embedding.is_(None)
            )
        )
        final_without_embeddings = result.scalar_one()

        print("\nFINAL STATE:")
        print(f"Chunks with embeddings:    {final_with_embeddings:>6}")
        print(f"Chunks WITHOUT embeddings: {final_without_embeddings:>6}")

        if final_without_embeddings == 0:
            print("\n✅ SUCCESS! All chunks now have embeddings.")
            print("RAG search is now fully functional!")
        else:
            print(f"\n⚠️  {final_without_embeddings} chunks still need embeddings.")
            print("You may need to run this script again.")

        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for chunks without them"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done, don't make changes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of chunks to process per batch (default: 50)"
    )

    args = parser.parse_args()

    asyncio.run(backfill_chunk_embeddings(
        dry_run=args.dry_run,
        batch_size=args.batch_size
    ))
