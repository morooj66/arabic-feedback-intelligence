from __future__ import annotations

import datetime as dt
import os

import pytest

from rcs.review.clock import iso_app, now_naive, parse_range_bound
from rcs.review.identity import (
    DemoIdentityDisabled,
    ReviewerIdentity,
    demo_microsoft_identity,
    is_production_env,
    migrate_demo_session_identity,
)

# NOTE: demo_microsoft_identity()/migrate_demo_session_identity() are used
# only by scripts/serve_reviewer_api.py --seed and the offline local-demo
# package builder (rcs.review.demo_seed, rcs.review.local_demo_package) —
# never by the real, authenticated application (rcs.review.http_api),
# which is real email/password + server-side sessions as of Phase 6. See
# tests/test_auth.py for the real authentication tests.


def test_demo_microsoft_identity_is_development_only(monkeypatch):
    monkeypatch.delenv("RCS_ENV", raising=False)
    monkeypatch.delenv("RCS_DEMO_REVIEWER_NAME", raising=False)
    ident = demo_microsoft_identity()
    assert ident.display_name == "مراجع"
    assert ident.reviewer_id == "demo:reviewer"
    monkeypatch.setenv("RCS_ENV", "production")
    assert is_production_env()
    with pytest.raises(DemoIdentityDisabled):
        demo_microsoft_identity()


def test_stale_demo_cookie_name_is_migrated_in_development(monkeypatch):
    monkeypatch.delenv("RCS_ENV", raising=False)
    monkeypatch.delenv("RCS_DEMO_REVIEWER_NAME", raising=False)
    stale = ReviewerIdentity("demo:reviewer", "مروج محمد")
    migrated = migrate_demo_session_identity(stale)
    assert migrated.display_name == "مراجع"
    other = ReviewerIdentity("demo:someone@example.org", "مروج محمد")
    assert migrate_demo_session_identity(other).display_name == "مروج محمد"
    monkeypatch.setenv("RCS_ENV", "production")
    assert migrate_demo_session_identity(stale).display_name == "مروج محمد"


def test_clock_treats_naive_as_riyadh():
    naive = dt.datetime(2026, 8, 26, 10, 0, 0)
    stamp = iso_app(naive)
    assert stamp is not None
    assert stamp.endswith("+03:00")
    assert now_naive(dt.datetime(2026, 8, 26, 7, 0, tzinfo=dt.timezone.utc)) == dt.datetime(2026, 8, 26, 10, 0, 0)
    start = parse_range_bound("2026-08-01", end_of_day=False)
    end = parse_range_bound("2026-08-01", end_of_day=True)
    assert start == dt.datetime(2026, 8, 1, 0, 0, 0)
    assert end == dt.datetime(2026, 8, 1, 23, 59, 59)
    assert start <= end


def test_canonical_taxonomy_labels_match_settings():
    from rcs.review.taxonomy_labels import INTENT_AR, SENTIMENT_AR, TOPIC_AR, assert_canonical_coverage

    assert_canonical_coverage()
    assert len(INTENT_AR) + len(TOPIC_AR) + len(SENTIMENT_AR) == 15
    assert "WORD_OF_YEAR" not in TOPIC_AR
