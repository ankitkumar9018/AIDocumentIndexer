"""
Migration script: Add source tracking columns to documents table.

New columns:
  - source_url (VARCHAR 2000, nullable)
  - source_type (VARCHAR 50, nullable)
  - is_stored_locally (BOOLEAN, default TRUE)
  - upload_source_info (JSON, nullable)

Run: python -m scripts.migrate_source_tracking
Or:  python scripts/migrate_source_tracking.py

In development mode (APP_ENV=development), these columns are created
automatically by SQLAlchemy's Base.metadata.create_all on startup.
This script is for production deployments where auto-create is disabled.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text


async def migrate():
    from backend.db.database import get_async_engine

    engine = get_async_engine()

    columns_to_add = [
        ("source_url", "VARCHAR(2000)", "NULL"),
        ("source_type", "VARCHAR(50)", "NULL"),
        ("is_stored_locally", "BOOLEAN", "TRUE"),
        ("upload_source_info", "JSONB", "NULL"),
    ]

    async with engine.begin() as conn:
        for col_name, col_type, default in columns_to_add:
            # Check if column exists
            result = await conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'documents' AND column_name = :col"
                ),
                {"col": col_name},
            )
            exists = result.scalar_one_or_none()

            if exists:
                print(f"  Column '{col_name}' already exists, skipping.")
                continue

            # Add column
            default_clause = f" DEFAULT {default}" if default != "NULL" else ""
            sql = f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}{default_clause}"
            await conn.execute(text(sql))
            print(f"  Added column '{col_name}' ({col_type}, default={default})")

    print("\nMigration complete!")


if __name__ == "__main__":
    print("Running source tracking migration...")
    asyncio.run(migrate())
