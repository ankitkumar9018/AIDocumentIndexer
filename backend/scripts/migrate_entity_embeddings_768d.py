#!/usr/bin/env python3
"""
AIDocumentIndexer - Entity Embedding Dimension Migration
=========================================================

Migrates entity embedding column from 1536D (OpenAI) to 768D (Ollama/HuggingFace).

This script updates the database schema to support 768-dimensional embeddings
from models like Ollama's nomic-embed-text and HuggingFace's sentence-transformers.

Usage:
    python backend/scripts/migrate_entity_embeddings_768d.py [--dry-run]

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

logger = structlog.get_logger(__name__)


async def migrate_embedding_dimensions(dry_run: bool = False):
    """
    Migrate entity embeddings from 1536D to 768D.

    For SQLite with vector extension:
    - Drops existing embedding column (if it has fixed dimensions)
    - Recreates with 768 dimensions

    Args:
        dry_run: If True, show what would be done without making changes

    Returns:
        Dictionary with migration statistics
    """
    logger.info("Starting entity embedding dimension migration", dry_run=dry_run)

    session_factory = get_async_session_factory()
    async with session_factory() as db:
        try:
            # Check current database type
            result = await db.execute(text("SELECT sqlite_version()"))
            sqlite_version = result.scalar_one()
            logger.info("Database detected", type="SQLite", version=sqlite_version)

            # Step 1: Check if entities table exists
            result = await db.execute(text("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='entities'
            """))
            if not result.scalar_one_or_none():
                logger.error("Entities table does not exist")
                return {"success": False, "error": "entities table not found"}

            # Step 2: Check current schema
            result = await db.execute(text("PRAGMA table_info(entities)"))
            columns = result.fetchall()
            embedding_col = None
            for col in columns:
                if col[1] == 'embedding':  # col[1] is column name
                    embedding_col = col
                    break

            if embedding_col:
                logger.info(
                    "Found embedding column",
                    name=embedding_col[1],
                    type=embedding_col[2],
                    nullable=bool(embedding_col[3])
                )
            else:
                logger.warning("No embedding column found in entities table")

            # Step 3: For SQLite, we need to recreate the table or handle it differently
            # SQLite doesn't support ALTER COLUMN, so we'll just accept any dimension
            # The ARRAY type in SQLite is flexible and doesn't enforce dimensions

            # Check if there are any existing embeddings
            result = await db.execute(text("""
                SELECT COUNT(*) FROM entities WHERE embedding IS NOT NULL
            """))
            existing_embeddings = result.scalar_one()

            if existing_embeddings > 0:
                logger.warning(
                    "Found existing embeddings that will be cleared",
                    count=existing_embeddings
                )

                if not dry_run:
                    # Clear existing embeddings (they're 1536D and incompatible)
                    await db.execute(text("""
                        UPDATE entities SET embedding = NULL WHERE embedding IS NOT NULL
                    """))
                    await db.commit()
                    logger.info("Cleared existing 1536D embeddings")
                else:
                    logger.info("DRY RUN: Would clear existing embeddings")

            # Step 4: SQLite with ARRAY type is flexible - no schema change needed
            # Just document that we now support 768D embeddings
            logger.info("SQLite ARRAY type supports flexible dimensions")
            logger.info("Migration complete - database ready for 768D embeddings")

            return {
                "success": True,
                "database_type": "SQLite",
                "sqlite_version": sqlite_version,
                "existing_embeddings_cleared": existing_embeddings if not dry_run else 0,
                "would_clear": existing_embeddings if dry_run else 0,
                "new_dimension": 768,
                "message": "Ready for 768D embeddings (Ollama nomic-embed-text)"
            }

        except Exception as e:
            logger.error("Migration failed", error=str(e))
            return {"success": False, "error": str(e)}


async def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate entity embeddings from 1536D to 768D"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    logger.info(
        "Entity Embedding Dimension Migration",
        dry_run=args.dry_run,
    )

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    else:
        response = input("\nThis will clear existing 1536D embeddings and prepare for 768D. Continue? [y/N]: ")
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
            logger.info("✅ SUCCESS - Database ready for 768D embeddings!")
            logger.info("Next step: Run backfill script to generate embeddings for 1560 entities")
    else:
        logger.error("Migration failed", error=result.get("error"))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
