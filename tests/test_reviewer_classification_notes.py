"""UAT remediation: the reviewer sees the contextual conflicts already stored with the classification (display-only), and nothing else leaks."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
from sqlalchemy.orm import Session

from rcs.review.auth import create_user
from rcs.review.dto import parse_context_conflicts, to_detail
from rcs.review.http_api import serve_in_thread
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db


# ---- parsing --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("reason,expected", [
    (None, ()),
    ("", ()),
    ("frozen_model_inference", ()),
    ("frozen_model_inference; context_conflicts=intent:calm_correction_predicted_complaint", ("intent:calm_correction_predicted_complaint",)),
    ("x; context_conflicts=topic:platform_failure_not_technical_topic,sentiment:calm_content_predicted_negative; later=ignored",
     ("topic:platform_failure_not_technical_topic", "sentiment:calm_content_predicted_negative")),
    ("context_conflicts=intent:ok_one, <script>alert(1)</script>:x ,notadimension:foo_bar,intent:", ("intent:ok_one",)),     # only well-formed tokens survive
    ("context_conflicts=" + "intent:" + "a" * 200, ()),
])
def test_parse_context_conflicts(reason, expected):
    assert parse_context_conflicts(reason) == expected


def test_detail_carries_only_the_parsed_codes_not_the_raw_reason():
    item = ReviewItem(comment_id="c1", source_file="f.xlsx", source_row_index=1, review_status="DRAFT_READY",
                      ai_classification_reason="experimental_local_v2_pilot_inference; context_conflicts=intent:question_not_predicted_inquiry")
    d = to_detail(item)
    assert d.context_conflicts == ("intent:question_not_predicted_inquiry",)
    assert "experimental" not in json.dumps(d.context_conflicts)
    assert not hasattr(d, "classification_reason") and not hasattr(d, "ai_classification_reason")


# ---- HTTP: exposure and reviewer isolation --------------------------------------------------------------------------------------
def _get(base, path, cookie):
    req = urllib.request.Request(base + path, headers={"Cookie": cookie})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _login(base, email, password):
    req = urllib.request.Request(base + "/api/login", data=json.dumps({"email": email, "password": password}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.headers["Set-Cookie"].split(";")[0]


@pytest.fixture()
def env(tmp_path):
    engine = get_engine(tmp_path / "notes.sqlite"); init_db(engine)
    a = create_user(engine, email="a@x.org", password="reviewer-a-pass-1", display_name="A", role="REVIEWER")
    b = create_user(engine, email="b@x.org", password="reviewer-b-pass-1", display_name="B", role="REVIEWER")
    with Session(engine) as s:
        s.add(ReviewItem(comment_id="with-conflicts", source_file="f.xlsx", source_row_index=1, original_comment="الموقع لا يفتح", review_status="REVIEW_ONLY", reviewer_user_id=a.user_id,
                         primary_intent="SUGGESTION", topic_subtype="OFF_TOPIC_PERSONAL", ai_confidence=0.26,
                         ai_classification_reason="frozen_model_inference; context_conflicts=topic:platform_failure_not_technical_topic,intent:question_not_predicted_inquiry"))
        s.add(ReviewItem(comment_id="plain", source_file="f.xlsx", source_row_index=2, original_comment="شكرا", review_status="DRAFT_READY", reviewer_user_id=a.user_id,
                         ai_classification_reason="frozen_model_inference"))
        s.add(ReviewItem(comment_id="bs-item", source_file="f.xlsx", source_row_index=3, original_comment="x", review_status="DRAFT_READY", reviewer_user_id=b.user_id,
                         ai_classification_reason="frozen_model_inference; context_conflicts=intent:praise_not_predicted_thanks"))
        s.commit()
    srv, base = serve_in_thread(engine)
    yield base
    srv.shutdown()


def test_assigned_reviewer_sees_conflicts_and_nothing_internal(env):
    cookie = _login(env, "a@x.org", "reviewer-a-pass-1")
    status, body = _get(env, "/api/items/with-conflicts", cookie)
    assert status == 200
    assert body["context_conflicts"] == ["topic:platform_failure_not_technical_topic", "intent:question_not_predicted_inquiry"]
    text = json.dumps(body, ensure_ascii=False)
    assert "frozen_model_inference" not in text and "ai_classification_reason" not in text and "classification_reason" not in text
    assert "visitor_name" not in body and "visitor_email" not in body                    # existing reviewer-facing privacy rule unchanged
    assert body["primary_intent"] == "SUGGESTION" and body["topic_subtype"] == "OFF_TOPIC_PERSONAL"      # predictions are untouched


def test_item_without_conflicts_returns_an_empty_list(env):
    cookie = _login(env, "a@x.org", "reviewer-a-pass-1")
    status, body = _get(env, "/api/items/plain", cookie)
    assert status == 200 and body["context_conflicts"] == []


def test_reviewer_isolation_is_preserved(env):
    cookie_a = _login(env, "a@x.org", "reviewer-a-pass-1")
    status, body = _get(env, "/api/items/bs-item", cookie_a)                              # belongs to reviewer B
    assert status in (403, 404) and "context_conflicts" not in body
    assert _get(env, "/api/items/with-conflicts", "rcs_session=forged")[0] == 401
