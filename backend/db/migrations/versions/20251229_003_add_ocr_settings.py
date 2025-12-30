"""Add OCR settings to SystemSettings table

Revision ID: 003_ocr_settings
Revises: 002_ai_optimization
Create Date: 2025-12-29

Adds OCR configuration settings:
- ocr.provider: OCR provider selection (paddleocr, tesseract, auto)
- ocr.paddle.variant: Model variant (server=accurate, mobile=fast)
- ocr.paddle.languages: List of language codes
- ocr.paddle.model_dir: Model storage directory
- ocr.paddle.auto_download: Auto-download on startup
- ocr.tesseract.fallback_enabled: Fallback to Tesseract
"""
import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '003_ocr_settings'
down_revision: Union[str, None] = '002_ai_optimization'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add OCR settings to system_settings table."""

    # Check if system_settings table exists
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if 'system_settings' not in tables:
        # SystemSettings table doesn't exist yet, skip
        return

    # Insert OCR settings
    # Note: Using raw SQL for compatibility with both SQLite and PostgreSQL
    connection = op.get_bind()

    settings = [
        (
            'ocr.provider',
            json.dumps('paddleocr'),
            'ocr',
            'string',
            'OCR provider (paddleocr, tesseract, auto)'
        ),
        (
            'ocr.paddle.variant',
            json.dumps('server'),
            'ocr',
            'string',
            'PaddleOCR model variant (server=accurate, mobile=fast)'
        ),
        (
            'ocr.paddle.languages',
            json.dumps(['en', 'de']),
            'ocr',
            'json',
            'List of language codes for OCR'
        ),
        (
            'ocr.paddle.model_dir',
            json.dumps('./data/paddle_models'),
            'ocr',
            'string',
            'Directory for PaddleOCR model cache'
        ),
        (
            'ocr.paddle.auto_download',
            json.dumps(True),
            'ocr',
            'boolean',
            'Auto-download models on startup'
        ),
        (
            'ocr.tesseract.fallback_enabled',
            json.dumps(True),
            'ocr',
            'boolean',
            'Fall back to Tesseract if PaddleOCR fails'
        ),
    ]

    # Check if settings already exist and insert if not
    for key, value, category, value_type, description in settings:
        # Check if setting exists
        result = connection.execute(
            sa.text("SELECT COUNT(*) FROM system_settings WHERE key = :key"),
            {"key": key}
        ).scalar()

        if result == 0:
            # Insert new setting
            connection.execute(
                sa.text("""
                    INSERT INTO system_settings (key, value, category, value_type, description)
                    VALUES (:key, :value, :category, :value_type, :description)
                """),
                {
                    "key": key,
                    "value": value,
                    "category": category,
                    "value_type": value_type,
                    "description": description,
                }
            )


def downgrade() -> None:
    """Remove OCR settings from system_settings table."""

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if 'system_settings' not in tables:
        return

    # Delete OCR settings
    connection = op.get_bind()
    connection.execute(
        sa.text("DELETE FROM system_settings WHERE category = 'ocr'")
    )
