"""password_reset_tokens: single-use, hashed, short-lived self-service reset tokens

Revision ID: b3d9f0a1c7e2
Revises: a7c1e4b92d10
Create Date: 2026-10-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3d9f0a1c7e2'
down_revision: Union[str, Sequence[str], None] = 'a7c1e4b92d10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema. Additive only: one new table, no change to existing ones."""
    op.create_table('password_reset_tokens',
    sa.Column('token_hash', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('used_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('token_hash')
    )
    op.create_index('ix_password_reset_tokens_user_created', 'password_reset_tokens', ['user_id', 'created_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_password_reset_tokens_user_created', table_name='password_reset_tokens')
    op.drop_table('password_reset_tokens')
