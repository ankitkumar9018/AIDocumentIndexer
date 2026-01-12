#!/usr/bin/env python3
"""
Complete Data Cleanup Script
=============================

This script completely clears all documents and embeddings from both
ChromaDB and SQLite, allowing for a fresh start with proper multi-tenant
metadata.

Usage:
    python -m backend.scripts.cleanup_all_data

    # Dry run (no changes)
    python -m backend.scripts.cleanup_all_data --dry-run

WARNING: This will delete ALL documents and embeddings! Use with caution.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import structlog
from sqlalchemy import delete, select, func

from backend.db.database import async_session_context
from backend.db.models import Document, Chunk
from backend.services.vectorstore_local import get_chroma_vector_store

logger = structlog.get_logger(__name__)


async def get_sqlite_stats():
    """Get counts from SQLite."""
    async with async_session_context() as db:
        doc_count = await db.scalar(select(func.count(Document.id)))
        chunk_count = await db.scalar(select(func.count(Chunk.id)))
        return {"documents": doc_count or 0, "chunks": chunk_count or 0}


def get_chroma_stats():
    """Get counts from ChromaDB."""
    try:
        chroma_store = get_chroma_vector_store()
        collection = chroma_store._collection
        count = collection.count()
        return {"embeddings": count}
    except Exception as e:
        logger.error("Failed to get Chroma stats", error=str(e))
        return {"embeddings": 0, "error": str(e)}


async def clear_sqlite(dry_run: bool = False):
    """Clear all documents and chunks from SQLite."""
    async with async_session_context() as db:
        if dry_run:
            return

        # Delete chunks first (foreign key constraint)
        await db.execute(delete(Chunk))

        # Delete documents
        await db.execute(delete(Document))

        await db.commit()


def clear_chroma(dry_run: bool = False):
    """Clear all embeddings from ChromaDB."""
    try:
        chroma_store = get_chroma_vector_store()
        collection = chroma_store._collection

        if dry_run:
            return collection.count()

        # Get all IDs and delete them
        all_data = collection.get()
        if all_data["ids"]:
            collection.delete(ids=all_data["ids"])

        return len(all_data["ids"]) if all_data["ids"] else 0
    except Exception as e:
        logger.error("Failed to clear Chroma", error=str(e))
        return 0


async def main():
    parser = argparse.ArgumentParser(
        description="Clear all documents and embeddings for fresh start"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't make changes, just report what would be done",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Complete Data Cleanup Script")
    print("=" * 60)

    # Get current stats
    print("\nCurrent data state:")

    sqlite_stats = await get_sqlite_stats()
    print(f"  SQLite Documents: {sqlite_stats['documents']}")
    print(f"  SQLite Chunks:    {sqlite_stats['chunks']}")

    chroma_stats = get_chroma_stats()
    print(f"  ChromaDB Embeddings: {chroma_stats['embeddings']}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***")
        print("\nWould delete:")
        print(f"  - {sqlite_stats['documents']} documents from SQLite")
        print(f"  - {sqlite_stats['chunks']} chunks from SQLite")
        print(f"  - {chroma_stats['embeddings']} embeddings from ChromaDB")
        return

    # Confirmation prompt
    if not args.force:
        print("\n" + "!" * 60)
        print("WARNING: This will DELETE ALL data!")
        print("!" * 60)
        confirm = input("\nType 'DELETE ALL' to confirm: ")
        if confirm != "DELETE ALL":
            print("Aborted.")
            return

    # Perform cleanup
    print("\nClearing data...")

    print("  Clearing ChromaDB embeddings...")
    chroma_deleted = clear_chroma(dry_run=False)
    print(f"    Deleted {chroma_deleted} embeddings")

    print("  Clearing SQLite documents and chunks...")
    await clear_sqlite(dry_run=False)
    print(f"    Deleted {sqlite_stats['chunks']} chunks")
    print(f"    Deleted {sqlite_stats['documents']} documents")

    # Verify cleanup
    print("\nVerifying cleanup...")
    sqlite_stats_after = await get_sqlite_stats()
    chroma_stats_after = get_chroma_stats()

    print(f"  SQLite Documents: {sqlite_stats_after['documents']}")
    print(f"  SQLite Chunks:    {sqlite_stats_after['chunks']}")
    print(f"  ChromaDB Embeddings: {chroma_stats_after['embeddings']}")

    if (sqlite_stats_after['documents'] == 0 and
        sqlite_stats_after['chunks'] == 0 and
        chroma_stats_after['embeddings'] == 0):
        print("\n✓ Cleanup complete! All data has been cleared.")
        print("\nYou can now re-upload your documents with proper organization metadata.")
    else:
        print("\n⚠ Warning: Some data may not have been deleted. Check logs for errors.")


if __name__ == "__main__":
    asyncio.run(main())
