#!/usr/bin/env python3
"""
Generate additional embeddings with a different provider.

This script allows you to generate embeddings from a second (or third) provider
while keeping existing embeddings intact. This enables instant provider switching
without re-indexing.

Usage:
    # Generate OpenAI embeddings while keeping Ollama
    python backend/scripts/generate_additional_embeddings.py \
        --provider openai \
        --model text-embedding-3-small \
        --dimension 768

    # Generate high-quality OpenAI embeddings
    python backend/scripts/generate_additional_embeddings.py \
        --provider openai \
        --model text-embedding-3-large \
        --dimension 3072

    # Set as primary (will be used for queries)
    python backend/scripts/generate_additional_embeddings.py \
        --provider openai \
        --model text-embedding-3-small \
        --dimension 768 \
        --set-primary
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import List, Optional

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
from sqlalchemy import select, func, and_, exists
import structlog

logger = structlog.get_logger()

# Import multi-embedding models if they exist
try:
    from backend.db.models_multi_embedding import ChunkEmbedding, get_or_create_embedding
    HAS_MULTI_EMBEDDING = True
except ImportError:
    HAS_MULTI_EMBEDDING = False
    logger.warning("Multi-embedding models not found. Run migration first.")


async def generate_additional_embeddings(
    target_provider: str,
    target_model: Optional[str] = None,
    target_dimension: Optional[int] = None,
    set_as_primary: bool = False,
    batch_size: int = 50,
    dry_run: bool = False
):
    """
    Generate embeddings using a different provider while keeping existing ones.

    Args:
        target_provider: Provider to use (openai, ollama, huggingface, etc.)
        target_model: Model name (optional, uses provider default if not specified)
        target_dimension: Target dimension (optional, auto-detected for OpenAI v3)
        set_as_primary: If True, set these embeddings as primary for queries
        batch_size: Number of chunks to process per batch
        dry_run: If True, only show what would be done
    """
    if not HAS_MULTI_EMBEDDING:
        print("\n❌ ERROR: Multi-embedding tables not found!")
        print("\nPlease run the migration first:")
        print("  alembic revision --autogenerate -m 'add_multi_embedding_support'")
        print("  alembic upgrade head")
        return

    session_factory = get_async_session_factory()

    # Initialize target embedding service
    logger.info(
        "Initializing embedding service",
        provider=target_provider,
        model=target_model or "default"
    )

    embedding_service = EmbeddingService(
        provider=target_provider,
        model=target_model
    )

    # Get actual dimension from embedding service
    test_embedding = embedding_service.embed_texts(["test"])
    actual_dimension = len(test_embedding[0]) if test_embedding else 0

    # If user specified dimension and it's OpenAI v3, verify it matches
    if target_dimension and target_dimension != actual_dimension:
        if "text-embedding-3" in (target_model or ""):
            print(f"\n⚠️  WARNING: Requested {target_dimension}D but got {actual_dimension}D")
            print(f"OpenAI v3 models may not support {target_dimension}D reduction.")
            print("Continuing with actual dimension...")
        target_dimension = actual_dimension
    else:
        target_dimension = actual_dimension

    # Get actual model name from service
    if not target_model:
        if target_provider == "ollama":
            target_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        else:
            target_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

    logger.info(
        "Target embedding configuration",
        provider=target_provider,
        model=target_model,
        dimension=target_dimension
    )

    async with session_factory() as db:
        # Count total chunks
        result = await db.execute(select(func.count(Chunk.id)))
        total_chunks = result.scalar_one()

        # Count chunks that already have this embedding
        result = await db.execute(
            select(func.count(Chunk.id))
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
            .where(
                ChunkEmbedding.provider == target_provider,
                ChunkEmbedding.model == target_model,
                ChunkEmbedding.dimension == target_dimension
            )
        )
        chunks_with_target = result.scalar_one()

        chunks_needed = total_chunks - chunks_with_target

        print("\n" + "=" * 70)
        print("ADDITIONAL EMBEDDING GENERATION")
        print("=" * 70)
        print(f"Target provider:              {target_provider}")
        print(f"Target model:                 {target_model}")
        print(f"Target dimension:             {target_dimension}D")
        print(f"Set as primary:               {'Yes' if set_as_primary else 'No'}")
        print(f"\nTotal chunks:                 {total_chunks:>6}")
        print(f"Already have target embedding:{chunks_with_target:>6}")
        print(f"Need to generate:             {chunks_needed:>6}")

        if dry_run:
            print("\n" + "=" * 70)
            print("DRY RUN - No changes will be made")
            print("=" * 70)
            return

        if chunks_needed == 0:
            print("\n✅ All chunks already have this embedding!")
            if set_as_primary:
                print("\nUpdating primary flag...")
                await db.execute(
                    ChunkEmbedding.__table__.update()
                    .where(
                        ChunkEmbedding.provider == target_provider,
                        ChunkEmbedding.model == target_model,
                        ChunkEmbedding.dimension == target_dimension
                    )
                    .values(is_primary=True)
                )
                # Unset other primary embeddings
                await db.execute(
                    ChunkEmbedding.__table__.update()
                    .where(
                        and_(
                            ChunkEmbedding.is_primary == True,
                            ~and_(
                                ChunkEmbedding.provider == target_provider,
                                ChunkEmbedding.model == target_model,
                                ChunkEmbedding.dimension == target_dimension
                            )
                        )
                    )
                    .values(is_primary=False)
                )
                await db.commit()
                print("✅ Primary flag updated!")
            print("=" * 70)
            return

        # Ask for confirmation
        print("\n" + "=" * 70)
        print("CONFIRMATION")
        print("=" * 70)
        print(f"This will generate {target_dimension}D embeddings for {chunks_needed} chunks")
        print(f"using {target_provider} ({target_model})")

        if target_provider == "openai":
            cost_estimate = (chunks_needed * 500) / 1_000_000 * 0.02  # Assuming 500 tokens avg, $0.02 per 1M
            print(f"\nEstimated cost: ${cost_estimate:.2f}")

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

        # Find chunks without this specific embedding
        # Use subquery to exclude chunks that already have the target embedding
        chunks_to_process = (
            select(Chunk)
            .where(
                ~exists(
                    select(1)
                    .where(
                        ChunkEmbedding.chunk_id == Chunk.id,
                        ChunkEmbedding.provider == target_provider,
                        ChunkEmbedding.model == target_model,
                        ChunkEmbedding.dimension == target_dimension
                    )
                )
            )
        )

        # Process in batches
        offset = 0
        while True:
            # Get batch of chunks
            result = await db.execute(
                chunks_to_process
                .limit(batch_size)
                .offset(offset)
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
                    offset += batch_size
                    continue

                # Store embeddings in chunk_embeddings table
                for chunk, embedding in zip(chunks, embeddings):
                    if embedding and len(embedding) == target_dimension:
                        await get_or_create_embedding(
                            db=db,
                            chunk_id=chunk.id,
                            provider=target_provider,
                            model=target_model,
                            dimension=target_dimension,
                            embedding=embedding,
                            set_as_primary=set_as_primary
                        )

                        # Also update primary embedding in chunks table if requested
                        if set_as_primary:
                            chunk.embedding = embedding
                            chunk.embedding_provider = target_provider
                            chunk.embedding_model = target_model
                            chunk.embedding_dimension = target_dimension
                    else:
                        logger.warning(
                            f"Invalid embedding for chunk {chunk.id}",
                            expected_dim=target_dimension,
                            got_dim=len(embedding) if embedding else 0
                        )
                        error_count += 1

                # Commit batch
                await db.commit()

                processed_count += len(chunks)
                progress = (processed_count / chunks_needed) * 100 if chunks_needed > 0 else 100
                print(f"Progress: {processed_count}/{chunks_needed} ({progress:.1f}%)")

            except Exception as e:
                logger.error(
                    "Error processing batch",
                    error=str(e),
                    batch_size=len(chunks)
                )
                await db.rollback()
                error_count += len(chunks)

            offset += batch_size

        print("\n" + "=" * 70)
        print("COMPLETE")
        print("=" * 70)
        print(f"Successfully processed: {processed_count} chunks")
        if error_count > 0:
            print(f"Errors encountered:     {error_count} chunks")

        if set_as_primary:
            print(f"\n✅ {target_provider}/{target_model} is now the PRIMARY embedding")
            print("   All queries will use this embedding by default")
        else:
            print(f"\n✅ {target_provider}/{target_model} stored as ALTERNATIVE embedding")
            print("   You can switch to it later with --set-primary")

        print("=" * 70)

        # Show current embedding status
        print("\n📊 CURRENT EMBEDDING STATUS:")
        result = await db.execute(
            select(
                ChunkEmbedding.provider,
                ChunkEmbedding.model,
                ChunkEmbedding.dimension,
                func.count(ChunkEmbedding.id).label('count'),
                func.sum(ChunkEmbedding.is_primary.cast(func.Integer)).label('primary_count')
            )
            .group_by(ChunkEmbedding.provider, ChunkEmbedding.model, ChunkEmbedding.dimension)
        )

        for row in result:
            primary_marker = " ⭐ PRIMARY" if row.primary_count > 0 else ""
            print(f"  {row.provider}/{row.model} ({row.dimension}D): {row.count} chunks{primary_marker}")

        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate additional embeddings with a different provider"
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["ollama", "openai", "huggingface", "cohere", "voyage", "mistral"],
        help="Target embedding provider"
    )
    parser.add_argument(
        "--model",
        help="Target model name (optional, uses provider default if not specified)"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        help="Target dimension (optional, auto-detected for most providers)"
    )
    parser.add_argument(
        "--set-primary",
        action="store_true",
        help="Set these embeddings as primary (used for queries)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of chunks to process per batch (default: 50)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done, don't make changes"
    )

    args = parser.parse_args()

    asyncio.run(generate_additional_embeddings(
        target_provider=args.provider,
        target_model=args.model,
        target_dimension=args.dimension,
        set_as_primary=args.set_primary,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    ))
