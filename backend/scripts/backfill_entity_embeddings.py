#!/usr/bin/env python3
"""
AIDocumentIndexer - Entity Embedding Backfill Script
=====================================================

Generates embeddings for all entities that don't have them yet.

This script should be run after enabling Phase 15 KG optimizations to backfill
embeddings for entities that were created before the auto-generation feature
was added.

Usage:
    python backend/scripts/backfill_entity_embeddings.py [--batch-size N] [--dry-run]

Options:
    --batch-size N    Number of entities to process per batch (default: 100)
    --dry-run         Show what would be done without making changes
    --verbose         Enable verbose logging
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_async_session_factory
from backend.db.models import Entity
from backend.services.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


async def count_entities_without_embeddings(db: AsyncSession) -> int:
    """
    Count how many entities need embeddings.

    Args:
        db: Database session

    Returns:
        Count of entities without embeddings (with valid names)
    """
    result = await db.execute(
        select(func.count(Entity.id)).where(
            Entity.embedding.is_(None),
            Entity.name.isnot(None),
            Entity.name != ''
        )
    )
    count = result.scalar_one()
    return count


async def backfill_entity_embeddings(
    db: AsyncSession,
    batch_size: int = 100,
    dry_run: bool = False,
    verbose: bool = False
) -> dict:
    """
    Generate embeddings for all entities without them.

    Args:
        db: Database session
        batch_size: Number of entities to process per batch
        dry_run: If True, don't actually update the database
        verbose: If True, log detailed progress

    Returns:
        Dictionary with statistics:
        - total_processed: Total entities processed
        - total_updated: Total entities updated with embeddings
        - batches: Number of batches processed
        - failed: Number of failures
    """
    logger.info("Starting entity embedding backfill", batch_size=batch_size, dry_run=dry_run)

    # Get embedding provider from environment (supports local Ollama, HuggingFace, etc.)
    embedding_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")

    logger.debug(
        "Environment variables",
        DEFAULT_LLM_PROVIDER=embedding_provider,
        DEFAULT_EMBEDDING_MODEL=os.getenv("DEFAULT_EMBEDDING_MODEL"),
        OLLAMA_EMBEDDING_MODEL=os.getenv("OLLAMA_EMBEDDING_MODEL"),
    )

    # For Ollama, prefer OLLAMA_EMBEDDING_MODEL over DEFAULT_EMBEDDING_MODEL
    # This allows provider-specific configuration
    if embedding_provider.lower() == "ollama":
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", None)
        if not embedding_model:
            # Fall back to DEFAULT_EMBEDDING_MODEL if OLLAMA_EMBEDDING_MODEL not set
            embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", None)
        logger.debug("Using Ollama embedding model", model=embedding_model)
    else:
        # For other providers, use DEFAULT_EMBEDDING_MODEL
        embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", None)

    logger.info(
        "Using embedding service",
        provider=embedding_provider,
        model=embedding_model or f"default for {embedding_provider}"
    )

    embedding_service = EmbeddingService(
        provider=embedding_provider,
        model=embedding_model
    )

    stats = {
        "total_processed": 0,
        "total_updated": 0,
        "batches": 0,
        "failed": 0,
        "start_time": datetime.utcnow(),
    }

    # Process in batches until no more entities without embeddings
    while True:
        # Find entities without embeddings AND with valid names
        result = await db.execute(
            select(Entity).where(
                Entity.embedding.is_(None),
                Entity.name.isnot(None),
                Entity.name != ''
            ).limit(batch_size)
        )
        entities: List[Entity] = list(result.scalars().all())

        if not entities:
            logger.info("No more entities to process")
            break

        stats["batches"] += 1
        stats["total_processed"] += len(entities)

        logger.info(
            f"Processing batch {stats['batches']}",
            entities_in_batch=len(entities),
            total_processed=stats["total_processed"],
        )

        # Generate text for embedding (name + description + aliases)
        texts = []
        for entity in entities:
            text_parts = [entity.name]
            if entity.description:
                text_parts.append(entity.description)
            if entity.aliases:
                text_parts.extend(entity.aliases[:3])  # Limit aliases
            texts.append(" ".join(text_parts))

        if verbose:
            logger.debug(
                "Sample entity texts",
                samples=texts[:3] if len(texts) > 3 else texts
            )

        try:
            # Batch embed
            embeddings = embedding_service.embed_texts(texts)

            # Update entities
            updated_count = 0
            for entity, embedding in zip(entities, embeddings):
                if embedding and not all(v == 0 for v in embedding):
                    if not dry_run:
                        entity.embedding = embedding
                    updated_count += 1

                    if verbose and updated_count <= 3:
                        logger.debug(
                            "Generated embedding",
                            entity_name=entity.name,
                            entity_type=entity.entity_type.value,
                            embedding_dimension=len(embedding),
                        )
                else:
                    stats["failed"] += 1
                    logger.warning(
                        "Failed to generate embedding",
                        entity_name=entity.name,
                        entity_id=str(entity.id),
                    )

            stats["total_updated"] += updated_count

            if not dry_run:
                await db.commit()
                logger.info(
                    "Batch committed",
                    updated=updated_count,
                    failed=len(entities) - updated_count,
                )
            else:
                logger.info(
                    "Batch processed (dry-run, no changes made)",
                    would_update=updated_count,
                    would_fail=len(entities) - updated_count,
                )

        except Exception as e:
            logger.error(
                "Batch processing failed",
                error=str(e),
                batch_number=stats["batches"],
            )
            stats["failed"] += len(entities)
            if not dry_run:
                await db.rollback()

    # Calculate statistics
    stats["end_time"] = datetime.utcnow()
    stats["duration_seconds"] = (stats["end_time"] - stats["start_time"]).total_seconds()
    stats["entities_per_second"] = stats["total_processed"] / stats["duration_seconds"] if stats["duration_seconds"] > 0 else 0

    return stats


async def main():
    """Main entry point for the backfill script."""
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for entities that don't have them"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of entities to process per batch (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG)
        )

    logger.info(
        "Entity Embedding Backfill Script",
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # Check how many entities need embeddings
    session_factory = get_async_session_factory()
    async with session_factory() as db:
        try:
            count = await count_entities_without_embeddings(db)
            logger.info(
                "Entities without embeddings",
                count=count,
                estimated_batches=(count + args.batch_size - 1) // args.batch_size,
            )

            if count == 0:
                logger.info("All entities already have embeddings!")
                return

            if args.dry_run:
                logger.info("DRY RUN MODE - No changes will be made")

            # Confirm before proceeding (unless dry-run)
            if not args.dry_run:
                response = input(f"\nProcess {count} entities? This will make database changes. [y/N]: ")
                if response.lower() != 'y':
                    logger.info("Backfill cancelled by user")
                    return

            # Run backfill
            stats = await backfill_entity_embeddings(
                db=db,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )

            # Print final statistics
            logger.info(
                "Backfill complete",
                total_processed=stats["total_processed"],
                total_updated=stats["total_updated"],
                failed=stats["failed"],
                batches=stats["batches"],
                duration_seconds=round(stats["duration_seconds"], 2),
                entities_per_second=round(stats["entities_per_second"], 2),
            )

            if args.dry_run:
                logger.info(
                    "DRY RUN COMPLETE - No changes were made. Run without --dry-run to apply changes."
                )
            else:
                logger.info(
                    "SUCCESS - All entity embeddings have been generated!"
                )
        except Exception as e:
            logger.error(f"Error in main: {e}")
            raise


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
