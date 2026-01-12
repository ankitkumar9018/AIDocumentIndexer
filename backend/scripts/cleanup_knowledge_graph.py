#!/usr/bin/env python3
"""
Knowledge Graph Maintenance Script
====================================

This script provides maintenance utilities for the knowledge graph:
- Cleanup orphan entities (entities with no document mentions)
- Cleanup orphan relationships (relationships with deleted entities)
- Rebuild knowledge graph for specific documents
- Analyze graph statistics

Usage:
    # Analyze graph status
    python -m backend.scripts.cleanup_knowledge_graph analyze

    # Cleanup orphan entities (dry run)
    python -m backend.scripts.cleanup_knowledge_graph cleanup --dry-run

    # Cleanup orphan entities (for real)
    python -m backend.scripts.cleanup_knowledge_graph cleanup

    # Rebuild graph for specific document
    python -m backend.scripts.cleanup_knowledge_graph rebuild --document-id <uuid>

    # Rebuild entire knowledge graph
    python -m backend.scripts.cleanup_knowledge_graph rebuild --all

Best Practices (from research):
- After deleting documents, run 'cleanup' to remove orphan entities
- Entities shared across multiple documents are only orphaned when ALL
  referencing documents are deleted
- Rebuilding the graph re-extracts entities from document chunks using LLM
- Regular maintenance (weekly/monthly) keeps the graph clean and performant
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Set
from uuid import UUID

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import structlog
from sqlalchemy import select, func, delete, and_, not_, exists

from backend.db.database import async_session_context
from backend.db.models import Entity, EntityMention, EntityRelation, Document, Chunk

logger = structlog.get_logger(__name__)


async def analyze_graph():
    """Analyze knowledge graph statistics."""
    print("=" * 60)
    print("Knowledge Graph Analysis")
    print("=" * 60)

    async with async_session_context() as db:
        # Entity counts
        total_entities = await db.scalar(select(func.count(Entity.id)))
        print(f"\nTotal Entities: {total_entities}")

        # Entity breakdown by type
        from backend.db.models import EntityType
        print("\nEntities by Type:")
        for etype in EntityType:
            count = await db.scalar(
                select(func.count(Entity.id))
                .where(Entity.entity_type == etype)
            )
            if count > 0:
                print(f"  {etype.value}: {count}")

        # Relationship counts
        total_relations = await db.scalar(select(func.count(EntityRelation.id)))
        print(f"\nTotal Relationships: {total_relations}")

        # Mention counts
        total_mentions = await db.scalar(select(func.count(EntityMention.id)))
        print(f"Total Mentions: {total_mentions}")

        # Documents with graph data
        docs_with_mentions = await db.scalar(
            select(func.count(func.distinct(EntityMention.document_id)))
        )
        total_docs = await db.scalar(select(func.count(Document.id)))
        print(f"\nDocuments with KG data: {docs_with_mentions}/{total_docs}")

        # Find orphan entities (no mentions)
        orphan_subquery = (
            select(EntityMention.entity_id)
            .distinct()
        )
        orphan_count = await db.scalar(
            select(func.count(Entity.id))
            .where(not_(Entity.id.in_(orphan_subquery)))
        )
        print(f"\nOrphan Entities (no mentions): {orphan_count}")

        if orphan_count > 0:
            # Show some examples
            orphan_result = await db.execute(
                select(Entity.name, Entity.entity_type)
                .where(not_(Entity.id.in_(orphan_subquery)))
                .limit(10)
            )
            print("  Examples:")
            for name, etype in orphan_result.all():
                print(f"    - {name} ({etype.value})")
            if orphan_count > 10:
                print(f"    ... and {orphan_count - 10} more")

        # Find orphan relationships (source or target deleted)
        valid_entity_ids = select(Entity.id)
        orphan_relations = await db.scalar(
            select(func.count(EntityRelation.id))
            .where(
                not_(
                    and_(
                        EntityRelation.source_entity_id.in_(valid_entity_ids),
                        EntityRelation.target_entity_id.in_(valid_entity_ids),
                    )
                )
            )
        )
        print(f"Orphan Relationships: {orphan_relations}")

        # Graph density
        if total_entities > 0:
            avg_relations_per_entity = total_relations / total_entities
            avg_mentions_per_entity = total_mentions / total_entities
            print(f"\nAvg Relations per Entity: {avg_relations_per_entity:.2f}")
            print(f"Avg Mentions per Entity: {avg_mentions_per_entity:.2f}")


async def cleanup_orphans(dry_run: bool = True):
    """Clean up orphan entities and relationships."""
    print("=" * 60)
    print("Knowledge Graph Cleanup")
    if dry_run:
        print("*** DRY RUN - No changes will be made ***")
    print("=" * 60)

    async with async_session_context() as db:
        # Step 1: Find orphan entities
        mentioned_entity_ids = select(EntityMention.entity_id).distinct()
        orphan_entities = await db.execute(
            select(Entity.id, Entity.name, Entity.entity_type)
            .where(not_(Entity.id.in_(mentioned_entity_ids)))
        )
        orphan_list = orphan_entities.all()

        print(f"\nOrphan Entities to delete: {len(orphan_list)}")
        if orphan_list:
            for eid, name, etype in orphan_list[:10]:
                print(f"  - {name} ({etype.value})")
            if len(orphan_list) > 10:
                print(f"  ... and {len(orphan_list) - 10} more")

        orphan_ids = [e[0] for e in orphan_list]

        # Step 2: Delete relationships involving orphan entities
        if orphan_ids:
            orphan_relations = await db.scalar(
                select(func.count(EntityRelation.id))
                .where(
                    EntityRelation.source_entity_id.in_(orphan_ids) |
                    EntityRelation.target_entity_id.in_(orphan_ids)
                )
            )
            print(f"Relationships to delete (involve orphans): {orphan_relations}")

            if not dry_run:
                await db.execute(
                    delete(EntityRelation)
                    .where(
                        EntityRelation.source_entity_id.in_(orphan_ids) |
                        EntityRelation.target_entity_id.in_(orphan_ids)
                    )
                )

        # Step 3: Delete orphan entities
        if orphan_ids and not dry_run:
            await db.execute(
                delete(Entity).where(Entity.id.in_(orphan_ids))
            )

        # Step 4: Clean up any remaining orphan relationships
        valid_entity_ids = select(Entity.id)
        remaining_orphan_relations = await db.scalar(
            select(func.count(EntityRelation.id))
            .where(
                not_(
                    and_(
                        EntityRelation.source_entity_id.in_(valid_entity_ids),
                        EntityRelation.target_entity_id.in_(valid_entity_ids),
                    )
                )
            )
        )

        if remaining_orphan_relations > 0:
            print(f"Additional orphan relationships to delete: {remaining_orphan_relations}")

            if not dry_run:
                # Delete relationships where either end doesn't exist
                await db.execute(
                    delete(EntityRelation)
                    .where(
                        not_(
                            and_(
                                EntityRelation.source_entity_id.in_(valid_entity_ids),
                                EntityRelation.target_entity_id.in_(valid_entity_ids),
                            )
                        )
                    )
                )

        if not dry_run:
            await db.commit()
            print("\n✓ Cleanup complete!")
        else:
            print("\n*** Run without --dry-run to apply changes ***")


async def rebuild_document_graph(document_id: str):
    """Rebuild knowledge graph for a specific document."""
    print(f"\nRebuilding graph for document: {document_id}")

    async with async_session_context() as db:
        # Verify document exists
        doc = await db.scalar(
            select(Document).where(Document.id == UUID(document_id))
        )
        if not doc:
            print(f"Document not found: {document_id}")
            return

        print(f"Document: {doc.original_filename or doc.filename}")

        # Delete existing mentions for this document
        deleted_mentions = await db.execute(
            delete(EntityMention).where(EntityMention.document_id == UUID(document_id))
        )
        print(f"Deleted existing mentions: {deleted_mentions.rowcount}")

        await db.commit()

        # Re-process document for graph
        from backend.services.knowledge_graph import get_knowledge_graph_service
        kg_service = await get_knowledge_graph_service(db)

        stats = await kg_service.process_document_for_graph(UUID(document_id))

        print(f"Extracted: {stats.get('entities', 0)} entities, "
              f"{stats.get('relations', 0)} relations, "
              f"{stats.get('mentions', 0)} mentions")

        await db.commit()
        print("✓ Document graph rebuilt!")


async def rebuild_all_graphs():
    """Rebuild knowledge graph for all documents."""
    print("=" * 60)
    print("Rebuilding Entire Knowledge Graph")
    print("=" * 60)

    async with async_session_context() as db:
        # Get all completed documents
        docs_result = await db.execute(
            select(Document.id, Document.original_filename)
            .where(Document.processing_status == "COMPLETED")
        )
        docs = docs_result.all()

        print(f"Documents to process: {len(docs)}")

        # Clear existing graph data
        print("\nClearing existing graph data...")
        await db.execute(delete(EntityMention))
        await db.execute(delete(EntityRelation))
        await db.execute(delete(Entity))
        await db.commit()
        print("✓ Graph cleared")

        # Rebuild for each document
        from backend.services.knowledge_graph import get_knowledge_graph_service

        total_stats = {"entities": 0, "relations": 0, "mentions": 0}

        for i, (doc_id, filename) in enumerate(docs, 1):
            print(f"\n[{i}/{len(docs)}] Processing: {filename or doc_id}")

            # Need fresh session for each doc due to LLM calls
            async with async_session_context() as doc_db:
                kg_service = await get_knowledge_graph_service(doc_db)

                try:
                    stats = await kg_service.process_document_for_graph(doc_id)
                    for key in total_stats:
                        total_stats[key] += stats.get(key, 0)
                    await doc_db.commit()
                    print(f"  → {stats.get('entities', 0)} entities, "
                          f"{stats.get('relations', 0)} relations")
                except Exception as e:
                    print(f"  → Error: {e}")

        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total Entities: {total_stats['entities']}")
        print(f"  Total Relations: {total_stats['relations']}")
        print(f"  Total Mentions: {total_stats['mentions']}")
        print("✓ Knowledge graph rebuild complete!")


async def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Maintenance Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyze command
    subparsers.add_parser("analyze", help="Analyze graph statistics")

    # Cleanup command
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Clean up orphan entities and relationships"
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )

    # Rebuild command
    rebuild_parser = subparsers.add_parser(
        "rebuild",
        help="Rebuild knowledge graph from documents"
    )
    rebuild_parser.add_argument(
        "--document-id",
        type=str,
        help="Rebuild graph for specific document UUID"
    )
    rebuild_parser.add_argument(
        "--all",
        action="store_true",
        help="Rebuild entire knowledge graph (WARNING: slow, uses LLM)"
    )

    args = parser.parse_args()

    if args.command == "analyze":
        await analyze_graph()
    elif args.command == "cleanup":
        await cleanup_orphans(dry_run=args.dry_run)
    elif args.command == "rebuild":
        if args.document_id:
            await rebuild_document_graph(args.document_id)
        elif args.all:
            confirm = input("This will use LLM for all documents. Continue? (y/N): ")
            if confirm.lower() == "y":
                await rebuild_all_graphs()
            else:
                print("Aborted.")
        else:
            print("Specify --document-id <uuid> or --all")


if __name__ == "__main__":
    asyncio.run(main())
