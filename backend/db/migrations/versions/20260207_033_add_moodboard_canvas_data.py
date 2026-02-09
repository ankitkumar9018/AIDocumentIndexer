"""add canvas_data to moodboard

Revision ID: 033_add_moodboard_canvas_data
Revises: d360f63b4f02
Create Date: 2026-02-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '033_add_moodboard_canvas_data'
down_revision: Union[str, None] = 'd360f63b4f02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('mood_boards', sa.Column('canvas_data', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('mood_boards', 'canvas_data')
