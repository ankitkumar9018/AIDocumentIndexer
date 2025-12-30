"""Add EasyOCR settings

Revision ID: 20251230_005
Revises: 20251230_004
Create Date: 2025-12-30 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20251230_005'
down_revision: Union[str, None] = '20251230_004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add EasyOCR settings to system_settings table."""

    # Check if system_settings table exists
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if 'system_settings' not in inspector.get_table_names():
        print("system_settings table does not exist, skipping EasyOCR settings insertion")
        return

    # Get the connection
    conn = op.get_bind()

    # Check if settings already exist
    result = conn.execute(
        sa.text("SELECT COUNT(*) FROM system_settings WHERE key LIKE 'ocr.easyocr%'")
    )
    count = result.scalar()

    if count > 0:
        print(f"EasyOCR settings already exist ({count} rows), skipping insertion")
        return

    # Update ocr.provider description to include easyocr
    conn.execute(
        sa.text("""
            UPDATE system_settings
            SET description = 'OCR provider (paddleocr, easyocr, tesseract, auto)'
            WHERE key = 'ocr.provider'
        """)
    )

    # Insert EasyOCR settings
    conn.execute(
        sa.text("""
            INSERT INTO system_settings (key, value, category, value_type, description, created_at, updated_at)
            VALUES
                ('ocr.easyocr.languages', '["en"]', 'ocr', 'json', 'List of language codes for EasyOCR', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('ocr.easyocr.use_gpu', 'true', 'ocr', 'boolean', 'Use GPU acceleration for EasyOCR (if available)', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """)
    )

    print("✓ EasyOCR settings added successfully")


def downgrade() -> None:
    """Remove EasyOCR settings."""

    conn = op.get_bind()

    # Restore original ocr.provider description
    conn.execute(
        sa.text("""
            UPDATE system_settings
            SET description = 'OCR provider (paddleocr, tesseract, auto)'
            WHERE key = 'ocr.provider'
        """)
    )

    # Remove EasyOCR settings
    conn.execute(
        sa.text("DELETE FROM system_settings WHERE key LIKE 'ocr.easyocr%'")
    )

    print("✓ EasyOCR settings removed")
