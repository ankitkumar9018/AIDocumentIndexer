#!/usr/bin/env python3
"""
Rebuild ChromaDB Index from SQLite Chunks

This script reads all document chunks from SQLite and re-creates
the ChromaDB embeddings index.
"""

import asyncio
import os
import sys
import shutil
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("DATABASE_URL", "sqlite:////Users/ankit/ankit_private/programs/mandala/AIDocumentIndexer/aidocindexer.db")


async def rebuild_chroma():
    """Rebuild ChromaDB index from SQLite chunks."""
    from sqlalchemy import select
    from backend.db.database import async_session_context
    from backend.db.models import Chunk, Document, ProcessingStatus
    from backend.services.llm import get_embeddings

    # First, completely remove old ChromaDB data
    chroma_dir = Path("./data/chroma")
    if chroma_dir.exists():
        print(f"Removing old ChromaDB data at {chroma_dir}...")
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # Now import and initialize fresh ChromaDB
    from backend.services.vectorstore_local import get_chroma_vector_store
    print("Initializing fresh ChromaDB...")
    vectorstore = get_chroma_vector_store()

    print("Initializing embedding service...")
    embeddings = get_embeddings()

    # Get all completed documents with chunks
    async with async_session_context() as db:
        # Get all chunks with their documents
        query = (
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.processing_status == ProcessingStatus.COMPLETED)
        )
        result = await db.execute(query)
        rows = result.all()

        print(f"Found {len(rows)} chunks to index")

        if not rows:
            print("No chunks found!")
            return

        # Process in batches
        batch_size = 50
        total_indexed = 0

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]

            # Prepare batch data
            chunk_ids = []
            documents = []
            metadatas = []

            for chunk, doc in batch:
                chunk_ids.append(str(chunk.id))
                documents.append(chunk.content or "")

                # Build metadata - ChromaDB doesn't accept None values
                meta = {
                    "document_id": str(chunk.document_id),
                    "document_filename": doc.original_filename or doc.filename or "unknown",
                    "chunk_index": chunk.chunk_index or 0,
                    "token_count": chunk.token_count or 0,
                }
                # Only add optional fields if they have values
                if chunk.page_number is not None:
                    meta["page_number"] = chunk.page_number
                if chunk.section_title:
                    meta["section_title"] = chunk.section_title
                if doc.tags and len(doc.tags) > 0:
                    meta["collection"] = doc.tags[0]
                metadatas.append(meta)

            # Generate embeddings
            print(f"Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")
            try:
                batch_embeddings = await embeddings.aembed_documents(documents)
            except Exception as e:
                print(f"Error embedding batch: {e}")
                continue

            # Add to ChromaDB
            try:
                vectorstore._collection.add(
                    ids=chunk_ids,
                    embeddings=batch_embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                total_indexed += len(batch)
                print(f"  Indexed {total_indexed}/{len(rows)} chunks")
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}")
                continue

        print(f"\nDone! Indexed {total_indexed} chunks into ChromaDB")

        # Verify
        count = vectorstore._collection.count()
        print(f"ChromaDB collection count: {count}")


if __name__ == "__main__":
    asyncio.run(rebuild_chroma())
