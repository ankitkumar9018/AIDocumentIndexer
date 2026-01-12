#!/usr/bin/env python3
"""
Update Existing Chroma Embeddings with Organization Metadata
=============================================================

This script updates existing embeddings in ChromaDB with organization_id,
uploaded_by_id, and is_private metadata from the SQLite Document records.

This is needed because older embeddings were created before multi-tenant
metadata was stored in Chroma.

Usage:
    python -m backend.scripts.update_chroma_metadata

    # Dry run (no changes)
    python -m backend.scripts.update_chroma_metadata --dry-run

    # Verbose output
    python -m backend.scripts.update_chroma_metadata --verbose
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import structlog
from sqlalchemy import select

from backend.db.database import async_session_context
from backend.db.models import Document
from backend.services.vectorstore_local import get_chroma_vector_store

logger = structlog.get_logger(__name__)


async def get_documents_with_org_info() -> Dict[str, Dict[str, Any]]:
    """
    Get all documents from SQLite with their organization metadata.

    Returns:
        Dict mapping document_id to {organization_id, uploaded_by_id, is_private}
    """
    async with async_session_context() as db:
        query = select(
            Document.id,
            Document.organization_id,
            Document.uploaded_by_id,
            Document.is_private,
            Document.filename,
        )
        result = await db.execute(query)
        rows = result.all()

        doc_info = {}
        for row in rows:
            doc_id = str(row[0])
            doc_info[doc_id] = {
                "organization_id": str(row[1]) if row[1] else "",
                "uploaded_by_id": str(row[2]) if row[2] else "",
                "is_private": bool(row[3]) if row[3] is not None else False,
                "filename": row[4] or "",
            }

        return doc_info


def update_chroma_metadata(
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Update Chroma embeddings with organization metadata from SQLite.

    Args:
        dry_run: If True, don't make changes, just report what would be done
        verbose: Print detailed progress

    Returns:
        Dict with counts: {updated, skipped, errors, total}
    """
    stats = {
        "total": 0,
        "updated": 0,
        "already_has_org": 0,
        "no_doc_record": 0,
        "errors": 0,
    }

    # Get Chroma store
    chroma_store = get_chroma_vector_store()
    collection = chroma_store._collection

    # Get all embeddings from Chroma
    print("Fetching all embeddings from ChromaDB...")
    all_data = collection.get(include=["metadatas"])

    if not all_data["ids"]:
        print("No embeddings found in ChromaDB.")
        return stats

    stats["total"] = len(all_data["ids"])
    print(f"Found {stats['total']} embeddings in ChromaDB")

    # Get document info from SQLite
    print("Fetching document metadata from SQLite...")
    doc_info = asyncio.get_event_loop().run_until_complete(get_documents_with_org_info())
    print(f"Found {len(doc_info)} documents in SQLite")

    # Group embeddings by document_id for batch updates
    updates_by_doc: Dict[str, List[str]] = {}  # doc_id -> [chunk_ids]

    for i, (chunk_id, metadata) in enumerate(zip(all_data["ids"], all_data["metadatas"])):
        doc_id = metadata.get("document_id", "")

        # Check if already has organization metadata
        if metadata.get("organization_id"):
            stats["already_has_org"] += 1
            if verbose:
                print(f"  [{i+1}/{stats['total']}] {chunk_id[:8]}... already has org metadata")
            continue

        # Check if we have doc info for this document
        if doc_id not in doc_info:
            stats["no_doc_record"] += 1
            if verbose:
                print(f"  [{i+1}/{stats['total']}] {chunk_id[:8]}... no document record found for {doc_id[:8]}...")
            continue

        # Add to updates
        if doc_id not in updates_by_doc:
            updates_by_doc[doc_id] = []
        updates_by_doc[doc_id].append(chunk_id)

    # Perform updates
    print(f"\nUpdating {sum(len(ids) for ids in updates_by_doc.values())} embeddings across {len(updates_by_doc)} documents...")

    for doc_id, chunk_ids in updates_by_doc.items():
        info = doc_info[doc_id]

        if verbose:
            print(f"  Updating {len(chunk_ids)} chunks for doc {info['filename'][:30]}...")

        if dry_run:
            stats["updated"] += len(chunk_ids)
            continue

        try:
            # Update metadata for all chunks of this document
            # Chroma requires updating metadata for each chunk individually
            for chunk_id in chunk_ids:
                # Get existing metadata
                existing = collection.get(ids=[chunk_id], include=["metadatas"])
                if existing["metadatas"]:
                    new_metadata = {**existing["metadatas"][0]}
                    new_metadata["organization_id"] = info["organization_id"]
                    new_metadata["uploaded_by_id"] = info["uploaded_by_id"]
                    new_metadata["is_private"] = info["is_private"]

                    collection.update(
                        ids=[chunk_id],
                        metadatas=[new_metadata],
                    )

            stats["updated"] += len(chunk_ids)

        except Exception as e:
            logger.error(
                "Failed to update chunks for document",
                doc_id=doc_id,
                error=str(e),
            )
            stats["errors"] += len(chunk_ids)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Update existing Chroma embeddings with organization metadata"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't make changes, just report what would be done",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Chroma Metadata Update Script")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    stats = update_chroma_metadata(
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Total embeddings:        {stats['total']}")
    print(f"  Updated:                 {stats['updated']}")
    print(f"  Already had org:         {stats['already_has_org']}")
    print(f"  No document record:      {stats['no_doc_record']}")
    print(f"  Errors:                  {stats['errors']}")

    if stats['no_doc_record'] > 0 and stats['updated'] == 0:
        print("\n" + "=" * 60)
        print("WARNING: Document ID Mismatch Detected")
        print("=" * 60)
        print("""
The embeddings in Chroma have document_ids that don't exist in the
SQLite database. This can happen if:

1. Documents were deleted and re-uploaded with new IDs
2. The database was reset without clearing Chroma
3. Documents were created before the Document model was used

Options:
- Re-upload documents to create matching records with org metadata
- Clear Chroma and re-process all documents
- The post-retrieval filtering will still work using SQLite data

Going forward, new documents will have proper metadata in both
SQLite and Chroma for organization filtering.
""")

    if args.dry_run:
        print("\n*** This was a dry run. Run without --dry-run to apply changes. ***")


if __name__ == "__main__":
    main()
