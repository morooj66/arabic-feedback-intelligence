"""m5e: agent_execution_traces

Revision ID: 5bc30c06b065
Revises: d34a7c3c9094
Create Date: 2026-09-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5bc30c06b065'
down_revision: Union[str, Sequence[str], None] = 'd34a7c3c9094'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('agent_execution_traces',
    sa.Column('execution_id', sa.String(), nullable=False),
    sa.Column('review_item_id', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('grounding_mode', sa.String(), nullable=False),
    sa.Column('official_retrieval_outcome', sa.String(), nullable=False),
    sa.Column('official_result_count', sa.Integer(), nullable=False),
    sa.Column('official_grounding_available', sa.Boolean(), nullable=False),
    sa.Column('style_retrieval_outcome', sa.String(), nullable=False),
    sa.Column('style_result_count', sa.Integer(), nullable=False),
    sa.Column('logical_generation_count', sa.Integer(), nullable=False),
    sa.Column('refinement_attempted', sa.Boolean(), nullable=False),
    sa.Column('initial_validation_passed', sa.Boolean(), nullable=False),
    sa.Column('initial_failure_code', sa.String(), nullable=True),
    sa.Column('final_validation_passed', sa.Boolean(), nullable=False),
    sa.Column('final_failure_code', sa.String(), nullable=True),
    sa.Column('generation_status', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('execution_id')
    )
    op.create_index(
        'ix_agent_execution_traces_review_item_created',
        'agent_execution_traces',
        ['review_item_id', 'created_at'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_agent_execution_traces_review_item_created', table_name='agent_execution_traces')
    op.drop_table('agent_execution_traces')
