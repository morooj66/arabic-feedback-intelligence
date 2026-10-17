"""Demo seed uses the current generation pipeline, not canned rg-v5 replies."""

from __future__ import annotations

from sqlalchemy.orm import Session

from config import settings
from rcs.review.demo_seed import _STALE_CANNED, _TEST_PIPELINE_DRAFTS, seed_demo_db
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine


import pytest

pytestmark = pytest.mark.usefixtures("legacy_eligibility_policy")


def test_demo_seed_uses_pipeline_drafts_not_stale_canned(tmp_path):
    path = seed_demo_db(tmp_path / "demo.sqlite", live=False)
    engine = get_engine(path)
    with Session(engine) as session:
        thanks = session.get(ReviewItem, "8742")
        complaint = session.get(ReviewItem, "8743")
        incomplete = session.get(ReviewItem, "8801")
        spam = session.get(ReviewItem, "9001")
        sent = session.get(ReviewItem, "9100")
        edited = session.get(ReviewItem, "8745")
        assert thanks.suggested_final_reply == _TEST_PIPELINE_DRAFTS["8742"]
        assert thanks.suggested_final_reply != _STALE_CANNED
        assert thanks.generation_prompt_version == settings.RESPONSE_GENERATION_PROMPT_VERSION
        assert thanks.generation_model_name == settings.RESPONSE_GENERATION_MODEL_NAME
        assert complaint.suggested_final_reply == _TEST_PIPELINE_DRAFTS["8743"]
        assert incomplete.suggested_final_reply is None
        assert incomplete.review_status == "REVIEW_ONLY"
        assert spam.reviewer_action == "NO_RESPONSE_NEEDED"
        assert spam.final_reply is None
        assert sent.reviewer_action == "SEND_AS_IS"
        assert sent.final_reply == _TEST_PIPELINE_DRAFTS["9100"]
        assert sent.outbound_status == "PENDING"
        assert sent.review_status != "SENT"
        assert edited.reviewer_action == "EDIT_AND_SEND"
        assert edited.suggested_final_reply == _TEST_PIPELINE_DRAFTS["8745"]
        assert edited.final_reply != edited.suggested_final_reply
        assert edited.outbound_status == "PENDING"
        assert thanks.generation_routing_decision == "GENERATE_DRAFT"


def test_demo_seed_records_pipeline_refresh_ids(tmp_path):
    seed_demo_db(tmp_path / "demo.sqlite", live=False)
    assert seed_demo_db.last_refreshed_comment_ids == ["8742", "8743", "8744", "8745", "8746", "9100"]
