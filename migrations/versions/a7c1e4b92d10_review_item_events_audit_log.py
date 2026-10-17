"""review_item_events: append-only audit log of reviewer/admin actions

Revision ID: a7c1e4b92d10
Revises: 5bc30c06b065
Create Date: 2026-10-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7c1e4b92d10'
down_revision: Union[str, Sequence[str], None] = '5bc30c06b065'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema. Additive only: one new table, no change to existing ones."""
    op.create_table('review_item_events',
    sa.Column('event_id', sa.String(), nullable=False),
    sa.Column('review_item_id', sa.String(), nullable=False),
    sa.Column('event_type', sa.String(), nullable=False),
    sa.Column('actor_user_id', sa.String(), nullable=False),
    sa.Column('actor_role', sa.String(), nullable=True),
    sa.Column('from_status', sa.String(), nullable=True),
    sa.Column('to_status', sa.String(), nullable=True),
    sa.Column('from_assignee_user_id', sa.String(), nullable=True),
    sa.Column('to_assignee_user_id', sa.String(), nullable=True),
    sa.Column('overrode_recommendation', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('event_id')
    )
    op.create_index(
        'ix_review_item_events_item_created',
        'review_item_events',
        ['review_item_id', 'created_at'],
        unique=False,
    )
    op.create_index(
        'ix_review_item_events_actor_created',
        'review_item_events',
        ['actor_user_id', 'created_at'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_review_item_events_actor_created', table_name='review_item_events')
    op.drop_index('ix_review_item_events_item_created', table_name='review_item_events')
    op.drop_table('review_item_events')
