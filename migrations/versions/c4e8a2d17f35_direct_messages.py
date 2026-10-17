"""direct_messages: internal admin <-> reviewer text messages

Revision ID: c4e8a2d17f35
Revises: b3d9f0a1c7e2
Create Date: 2026-10-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4e8a2d17f35'
down_revision: Union[str, Sequence[str], None] = 'b3d9f0a1c7e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema. Additive only: one new table, no change to existing ones."""
    op.create_table('direct_messages',
    sa.Column('message_id', sa.String(), nullable=False),
    sa.Column('reviewer_user_id', sa.String(), nullable=False),
    sa.Column('sender_user_id', sa.String(), nullable=False),
    sa.Column('sender_role', sa.String(), nullable=False),
    sa.Column('body', sa.Text(), nullable=False),
    sa.Column('client_message_id', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('read_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('message_id'),
    sa.UniqueConstraint('sender_user_id', 'client_message_id', name='uq_direct_messages_sender_client_id')
    )
    op.create_index('ix_direct_messages_conversation_created', 'direct_messages', ['reviewer_user_id', 'created_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_direct_messages_conversation_created', table_name='direct_messages')
    op.drop_table('direct_messages')
