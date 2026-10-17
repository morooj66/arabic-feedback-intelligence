"""account_requests: self-service account requests awaiting admin approval

Revision ID: d5f1b3c8a2e6
Revises: c4e8a2d17f35
Create Date: 2026-10-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5f1b3c8a2e6'
down_revision: Union[str, Sequence[str], None] = 'c4e8a2d17f35'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema. Additive only: one new table, no change to existing ones."""
    op.create_table('account_requests',
    sa.Column('request_id', sa.String(), nullable=False),
    sa.Column('email', sa.String(), nullable=False),
    sa.Column('display_name', sa.String(), nullable=False),
    sa.Column('password_hash', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('decided_at', sa.DateTime(), nullable=True),
    sa.Column('decided_by_user_id', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('request_id'),
    sa.UniqueConstraint('email')
    )
    op.create_index('ix_account_requests_status_created', 'account_requests', ['status', 'created_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_account_requests_status_created', table_name='account_requests')
    op.drop_table('account_requests')
