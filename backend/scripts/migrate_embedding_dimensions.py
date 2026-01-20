#!/usr/bin/env python3
"""
AIDocumentIndexer - Embedding Dimension Migration
==================================================

Clears existing embeddings when switching between providers with different dimensions.

This script is useful when:
- Switching from OpenAI (1536D) to Ollama (768D)
- Switching from Ollama (768D) to OpenAI (1536D)
- Switching to any other provider with different dimensions

The database schema will automatically adapt to the new dimension based on
DEFAULT_LLM_PROVIDER in your .env file.

Usage:
    python backend/scripts/migrate_embedding_dimensions.py [--dry-run]

Options:
    --dry-run    Show what would be done without making changes
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import structlog
from sqlalchemy import text

from backend.db.database import get_async_session_factory
from backend.db.models import get_embedding_dimension

logger = structlog.get_logger(__name__)


async def migrate_embedding_dimensions(dry_run: bool = False):
    """
    Clear all existing embeddings in preparation for dimension change.

    This clears embeddings from:
    - Chunk embeddings
    - Document embeddings
    - Entity embeddings
    - Query cache embeddings

    The database schema will automatically use the new dimension from
    DEFAULT_LLM_PROVIDER when the application restarts.

    Args:
        dry_run: If True, show what would be done without making changes

    Returns:
        Dictionary with migration statistics
    """
    new_dimension = get_embedding_dimension()
    provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")

    logger.info(
        "Starting embedding dimension migration",
        dry_run=dry_run,
        new_dimension=new_dimension,
        provider=provider
    )

    session_factory = get_async_session_factory()
    async with session_factory() as db:
        try:
            # Check current database type
            result = await db.execute(text("SELECT sqlite_version()"))
            sqlite_version = result.scalar_one()
            logger.info("Database detected", type="SQLite", version=sqlite_version)

            stats = {
                "chunks_cleared": 0,
                "documents_cleared": 0,
                "entities_cleared": 0,
                "queries_cleared": 0,
            }

            # Count existing embeddings in each table
            result = await db.execute(text("""
                SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL
            """))
            chunks_count = result.scalar_one()
            stats["chunks_cleared"] = chunks_count

            result = await db.execute(text("""
                SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL
            """))
            docs_count = result.scalar_one()
            stats["documents_cleared"] = docs_count

            result = await db.execute(text("""
                SELECT COUNT(*) FROM entities WHERE embedding IS NOT NULL
            """))
            entities_count = result.scalar_one()
            stats["entities_cleared"] = entities_count

            result = await db.execute(text("""
                SELECT COUNT(*) FROM llm_response_cache WHERE query_embedding IS NOT NULL
            """))
            queries_count = result.scalar_one()
            stats["queries_cleared"] = queries_count

            total = chunks_count + docs_count + entities_count + queries_count

            logger.info(
                "Found existing embeddings",
                chunks=chunks_count,
                documents=docs_count,
                entities=entities_count,
                queries=queries_count,
                total=total
            )

            if total == 0:
                logger.info("No embeddings to clear")
                return {
                    "success": True,
                    "message": "No embeddings found, nothing to clear",
                    **stats
                }

            if not dry_run:
                # Clear all embeddings
                logger.info("Clearing chunk embeddings...")
                await db.execute(text("""
                    UPDATE chunks SET embedding = NULL WHERE embedding IS NOT NULL
                """))

                logger.info("Clearing document embeddings...")
                await db.execute(text("""
                    UPDATE documents SET embedding = NULL WHERE embedding IS NOT NULL
                """))

                logger.info("Clearing entity embeddings...")
                await db.execute(text("""
                    UPDATE entities SET embedding = NULL WHERE embedding IS NOT NULL
                """))

                logger.info("Clearing query cache embeddings...")
                await db.execute(text("""
                    UPDATE llm_response_cache SET query_embedding = NULL WHERE query_embedding IS NOT NULL
                """))

                await db.commit()
                logger.info("Successfully cleared all embeddings")
            else:
                logger.info("DRY RUN: Would clear all embeddings")

            return {
                "success": True,
                "database_type": "SQLite",
                "sqlite_version": sqlite_version,
                "new_dimension": new_dimension,
                "provider": provider,
                **stats,
                "message": f"Ready for {new_dimension}D embeddings ({provider} provider)"
            }

        except Exception as e:
            logger.error("Migration failed", error=str(e))
            return {"success": False, "error": str(e)}


async def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description="Clear embeddings when switching providers/dimensions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    new_dimension = get_embedding_dimension()

    logger.info(
        "Embedding Dimension Migration",
        dry_run=args.dry_run,
        provider=provider,
        new_dimension=new_dimension
    )

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    else:
        print("\n" + "=" * 70)
        print("⚠️  WARNING: This will clear ALL existing embeddings")
        print("=" * 70)
        print(f"Provider: {provider}")
        print(f"New dimension: {new_dimension}D")
        print("\nAfter clearing, you will need to:")
        print("1. Re-index all documents")
        print("2. Run entity embedding backfill")
        print("=" * 70)
        response = input("\nContinue? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Migration cancelled by user")
            return

    # Run migration
    result = await migrate_embedding_dimensions(dry_run=args.dry_run)

    # Print result
    if result["success"]:
        logger.info("Migration successful", **{k: v for k, v in result.items() if k != "success"})

        if args.dry_run:
            logger.info("DRY RUN COMPLETE - Run without --dry-run to apply changes")
        else:
            logger.info(f"✅ SUCCESS - Database ready for {new_dimension}D embeddings!")
            logger.info("Next steps:")
            logger.info("1. Re-index documents to generate new embeddings")
            logger.info("2. Run: python backend/scripts/backfill_entity_embeddings.py")
    else:
        logger.error("Migration failed", error=result.get("error"))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
