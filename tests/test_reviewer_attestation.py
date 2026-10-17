"""Reviewer attestation of unverifiable factual statements: permission-controlled, exact-text, durably audited, and unable to bypass URL /
commitment / policy / security checks. Production defaults; no network."""

from __future__ import annotations

import datetime as dt
import inspect
import json
import urllib.error
import urllib.request

import pytest

from config import settings
from rcs.response_generation.schema import GenerationResult
from rcs.review.audit_events import EVENT_EDIT_AND_SEND, EVENT_EDIT_AND_SEND_ATTESTED, EVENT_SEND_AS_IS, EVENT_SEND_AS_IS_ATTESTED, EVENT_TYPES
from rcs.review.auth import create_user
from rcs.review.http_api import serve_in_thread
from rcs.review.identity import ReviewerIdentity
from rcs.review.repositories.contracts import ReviewItemRepository
from rcs.review.repositories.mongo import MongoReviewItemRepository
from rcs.review.repositories.postgres import PostgresReviewItemRepository
from rcs.review.repository import get_engine, init_db, upsert_review_items
from rcs.review.service import FactAttestation, ReviewWorkflowService
from rcs.review.workflow import AttestationNotPermitted, AttestationRequired, DraftValidationError

COMMENT = "ما معنى كلمة سوار"
FACT = "معنى كلمة «سوار» هو حلية تلبس في المعصم."          # unverifiable lexical fact -> attestable
FACT_CODES = ("unsupported_definition",)

REVIEWER = ReviewerIdentity("u-rev", "مراجع", user_id="u-rev", role="REVIEWER")
ADMIN = ReviewerIdentity("u-adm", "مدير", user_id="u-adm", role="ADMIN")
NO_ROLE = ReviewerIdentity("legacy", "قديم")


def _engine(tmp_path):
    engine = get_engine(tmp_path / "att.sqlite")
    init_db(engine)
    return engine


def _seed(engine, comment_id="c1", draft=None, status_text=COMMENT):
    now = dt.datetime(2026, 1, 15, 12, 0, 0)
    upsert_review_items(engine, [{
        "comment_id": comment_id, "source_file": "t.xlsx", "source_row_index": 1, "original_comment": status_text, "ingested_at": now,
        "comment_created_at": now, "review_status": "CLASSIFIED", "ready_to_send": False, "primary_intent": "INQUIRY", "topic_subtype": "GENERAL",
        "ai_primary_intent": "INQUIRY", "ai_topic_subtype": "GENERAL", "ai_confidence": 0.5, "row_version": 1,
    }])
    svc = ReviewWorkflowService(engine)
    if draft is None:
        svc.apply_generation_result(GenerationResult(
            comment_id=comment_id, routing_decision="REVIEW_ONLY", generation_status="SKIPPED", suggested_response=None, review_required=True,
            reason_code="factual_question_without_grounding", provider=None, model_name=None, prompt_version=None, generated_at="2026-08-25T12:00:00+00:00"))
    else:
        svc.apply_generation_result(GenerationResult(
            comment_id=comment_id, routing_decision="GENERATE_DRAFT", generation_status="SUCCEEDED", suggested_response=draft, review_required=True,
            reason_code="x", provider="gemini", model_name="m", prompt_version="rg-v14.2", generated_at="2026-08-25T12:00:00+00:00"))
    return svc


def _types(svc, cid="c1"):
    return [e.event_type for e in svc.item_history(cid)]


ATT = FactAttestation(confirmed=True, finding_codes=FACT_CODES)


# ----------------------------------------------------------------------------------------- safe reviewer-authored facts
def test_unverifiable_fact_without_attestation_is_refused_and_nothing_changes(tmp_path):
    svc = _seed(_engine(tmp_path))
    before = svc.repo.get("c1")
    with pytest.raises(AttestationRequired) as exc:
        svc.edit_and_send("c1", REVIEWER, FACT)
    assert exc.value.finding_codes == FACT_CODES
    after = svc.repo.get("c1")
    assert (after.review_status, after.row_version, after.final_reply, after.reviewer_action) == (before.review_status, before.row_version, None, None)
    assert _types(svc) == []                                    # a refused attempt writes no event


def test_attested_fact_is_sent_exactly_as_written_and_audited(tmp_path):
    svc = _seed(_engine(tmp_path))
    detail = svc.edit_and_send("c1", REVIEWER, FACT, attestation=ATT)
    assert detail.review_status == "EDITED_AND_APPROVED" and detail.final_response == FACT      # no rewriting
    events = svc.item_history("c1")
    assert [e.event_type for e in events] == [EVENT_EDIT_AND_SEND_ATTESTED]
    e = events[0]
    assert e.actor_user_id == "u-rev" and e.actor_role == "REVIEWER" and e.created_at is not None
    assert e.to_status == "EDITED_AND_APPROVED" and e.from_status in ("REVIEW_ONLY", "CLASSIFIED")


def test_reply_needing_no_attestation_records_a_plain_event_even_if_a_confirmation_is_sent(tmp_path):
    svc = _seed(_engine(tmp_path))
    svc.edit_and_send("c1", REVIEWER, "نشكر لكم سؤالكم، ولا تتوفر لدينا معلومة موثوقة عن معنى هذه الكلمة.", attestation=ATT)
    assert _types(svc) == [EVENT_EDIT_AND_SEND]                  # never a misleading "attested" record


def test_verdict_and_number_claims_are_attestable_too(tmp_path):
    svc = _seed(_engine(tmp_path), "c1", status_text="أرى أن التعريف خاطئ")
    text = "التعريف الوارد في المعجم صحيح، وعدد معانيه 3."
    with pytest.raises(AttestationRequired) as exc:
        svc.edit_and_send("c1", REVIEWER, text)
    assert set(exc.value.finding_codes) == {"unsupported_correctness_claim", "unsupported_number"}
    svc.edit_and_send("c1", REVIEWER, text, attestation=FactAttestation(True, tuple(exc.value.finding_codes)))
    assert _types(svc) == [EVENT_EDIT_AND_SEND_ATTESTED]


# -------------------------------------------------------------------------------------------- missing / wrong attestation
@pytest.mark.parametrize(
    "att",
    [
        None,
        FactAttestation(confirmed=False, finding_codes=FACT_CODES),
        FactAttestation(confirmed=True, finding_codes=()),                                         # blanket confirmation
        FactAttestation(confirmed=True, finding_codes=("unsupported_number",)),                    # confirms something else
        FactAttestation(confirmed=True, finding_codes=FACT_CODES + ("unsupported_correctness_claim",)),   # stale / over-broad
    ],
)
def test_missing_or_mismatched_attestation_is_refused(tmp_path, att):
    svc = _seed(_engine(tmp_path))
    with pytest.raises(AttestationRequired):
        svc.edit_and_send("c1", REVIEWER, FACT, attestation=att)
    assert svc.repo.get("c1").final_reply is None and _types(svc) == []


# ------------------------------------------------------------------------------------ attestation can never bypass the rest
@pytest.mark.parametrize(
    "text,expected",
    [
        (FACT + " راجعوا https://example.org/help", "unapproved_url"),
        (FACT + " سنقوم بإضافتها قريبا.", "unsupported_commitment"),
        (FACT + " تمت إضافة الكلمة إلى المعجم.", "unsupported_commitment"),
        (FACT + " تخضع الاقتراحات لمراجعة لجنة متخصصة.", "unsupported_policy_claim"),
        (FACT + " اتصلوا على 0501234567", "pii_in_draft"),
        (FACT + " حسب confidence النظام", "internal_leak"),
        ("تم تحويل ملاحظتك إلى فريق المحتوى. " + FACT, "unsupported_operational_claim"),
    ],
)
def test_attestation_cannot_bypass_hard_checks(tmp_path, text, expected):
    svc = _seed(_engine(tmp_path))
    with pytest.raises(DraftValidationError) as exc:
        svc.edit_and_send("c1", ADMIN, text, attestation=ATT)               # even an ADMIN with a matching confirmation
    assert expected in exc.value.blocking
    assert svc.repo.get("c1").final_reply is None and _types(svc) == []


def test_hard_findings_win_over_attestable_ones_and_attestation_is_not_consulted(tmp_path):
    svc = _seed(_engine(tmp_path))
    with pytest.raises(DraftValidationError):
        svc.edit_and_send("c1", REVIEWER, FACT + " سنصلح الخلل قريبا.")      # no attestation supplied: still the hard error, not 'attestation required'


# ------------------------------------------------------------------------------------------------------- role enforcement
@pytest.mark.parametrize("who", [NO_ROLE, ReviewerIdentity("x", "x", user_id="x", role="VIEWER")])
def test_roles_outside_the_allow_list_cannot_attest(tmp_path, who):
    svc = _seed(_engine(tmp_path))
    with pytest.raises(AttestationNotPermitted):
        svc.edit_and_send("c1", who, FACT, attestation=ATT)
    assert svc.repo.get("c1").final_reply is None and _types(svc) == []


def test_role_allow_list_is_configurable(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FACT_ATTESTATION_ROLES", frozenset({"ADMIN"}))
    svc = _seed(_engine(tmp_path))
    with pytest.raises(AttestationNotPermitted):
        svc.edit_and_send("c1", REVIEWER, FACT, attestation=ATT)
    svc.edit_and_send("c1", ADMIN, FACT, attestation=ATT)
    assert _types(svc) == [EVENT_EDIT_AND_SEND_ATTESTED]


def test_attestation_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FACT_ATTESTATION_ENABLED", False)
    svc = _seed(_engine(tmp_path))
    with pytest.raises(AttestationNotPermitted):
        svc.edit_and_send("c1", ADMIN, FACT, attestation=ATT)


def test_defaults_are_conservative():
    assert settings.FACT_ATTESTATION_ENABLED is True and settings.FACT_ATTESTATION_ROLES == frozenset({"REVIEWER", "ADMIN"})
    assert "unapproved_url" not in __import__("rcs.response_generation.validate", fromlist=["x"]).ATTESTABLE_FINDINGS


# --------------------------------------------------------------------------------------- send_as_is enforces the same policy
def test_send_as_is_of_a_stale_draft_with_an_unverifiable_fact_needs_attestation(tmp_path):
    svc = _seed(_engine(tmp_path), draft=FACT)
    with pytest.raises(AttestationRequired):
        svc.send_as_is("c1", REVIEWER)
    assert svc.repo.get("c1").review_status == "DRAFT_READY"
    svc.send_as_is("c1", REVIEWER, attestation=ATT)
    assert _types(svc) == [EVENT_SEND_AS_IS_ATTESTED]
    assert svc.repo.get("c1").final_reply == FACT


def test_send_as_is_enforces_hard_blocks_and_roles_like_edit_and_send(tmp_path):
    svc = _seed(_engine(tmp_path), draft="شكرًا لكم، سنقوم بمراجعة الملاحظة وإضافتها.")
    with pytest.raises(DraftValidationError):
        svc.send_as_is("c1", ADMIN, attestation=ATT)
    svc2 = _seed(_engine(tmp_path / "sub") if (tmp_path / "sub").mkdir() is None else None, draft=FACT)
    with pytest.raises(AttestationNotPermitted):
        svc2.send_as_is("c1", NO_ROLE, attestation=ATT)


def test_clean_draft_sends_as_is_with_a_plain_event(tmp_path):
    svc = _seed(_engine(tmp_path), draft="نشكر لكم اهتمامكم بالمعجم.")
    svc.send_as_is("c1", REVIEWER, attestation=ATT)
    assert _types(svc) == [EVENT_SEND_AS_IS]


def test_with_an_official_source_configured_only_grounding_independent_checks_rerun_on_stored_drafts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "OFFICIAL_KNOWLEDGE_SOURCE_PATH", "/nonexistent/corpus.json")
    svc = _seed(_engine(tmp_path), draft=FACT)                    # was checked WITH grounding when generated
    svc.send_as_is("c1", REVIEWER)
    assert _types(svc) == [EVENT_SEND_AS_IS]
    svc2 = _seed(_engine(tmp_path / "s2") if (tmp_path / "s2").mkdir() is None else None, draft="سنقوم بمراجعة الملاحظة قريبا.")
    with pytest.raises(DraftValidationError):
        svc2.send_as_is("c1", REVIEWER)


# --------------------------------------------------------------------------------------------------------- audit persistence
def test_attested_event_is_durable_across_a_restart(tmp_path):
    engine = _engine(tmp_path)
    svc = _seed(engine)
    svc.edit_and_send("c1", REVIEWER, FACT, attestation=ATT)
    del svc, engine
    svc2 = ReviewWorkflowService(get_engine(tmp_path / "att.sqlite"))
    events = svc2.item_history("c1")
    assert [e.event_type for e in events] == [EVENT_EDIT_AND_SEND_ATTESTED]
    assert events[0].actor_user_id == "u-rev"
    assert svc2.repo.get("c1").final_reply == FACT


def test_event_is_written_in_the_same_transaction_as_the_action(tmp_path, monkeypatch):
    from rcs.review.repositories import postgres

    svc = _seed(_engine(tmp_path))

    def boom(*a, **k):
        raise RuntimeError("audit store down")

    monkeypatch.setattr(postgres, "_add_event", boom)
    with pytest.raises(RuntimeError):
        svc.edit_and_send("c1", REVIEWER, FACT, attestation=ATT)
    monkeypatch.undo()
    item = svc.repo.get("c1")
    assert item.final_reply is None and item.reviewer_action is None             # no attested send without its audit record


def test_event_types_registered_and_backends_share_the_contract():
    assert EVENT_EDIT_AND_SEND_ATTESTED in EVENT_TYPES and EVENT_SEND_AS_IS_ATTESTED in EVENT_TYPES
    for cls in (ReviewItemRepository, PostgresReviewItemRepository, MongoReviewItemRepository):
        for name in ("send_as_is", "edit_and_send"):
            assert "attested" in inspect.signature(getattr(cls, name)).parameters, (cls.__name__, name)


def test_audit_event_carries_no_reply_text(tmp_path):
    svc = _seed(_engine(tmp_path))
    svc.edit_and_send("c1", REVIEWER, FACT, attestation=ATT)
    blob = json.dumps([e.__dict__ for e in svc.item_history("c1")], default=str, ensure_ascii=False)
    assert "سوار" not in blob and "المعصم" not in blob


# ---------------------------------------------------------------------------------------------------------------- HTTP
def _json(url, *, method="GET", data=None, cookie=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json", **({"Cookie": cookie} if cookie else {})}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), resp.headers
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8")), err.headers


def _http_setup(tmp_path):
    from sqlalchemy.orm import Session

    from rcs.review.models import ReviewItem

    engine = _engine(tmp_path)
    reviewer = create_user(engine, email="r@example.org", password="passwordR123", display_name="R", role="REVIEWER")
    create_user(engine, email="a@example.org", password="passwordA123", display_name="A", role="ADMIN")
    _seed(engine)
    with Session(engine) as s:
        s.get(ReviewItem, "c1").reviewer_user_id = reviewer.user_id
        s.commit()
    server, base = serve_in_thread(engine)
    login = lambda e, p: _json(f"{base}/api/login", method="POST", data={"email": e, "password": p})[2].get("Set-Cookie").split(";")[0]
    return server, base, login


def test_http_flow_requires_attestation_then_succeeds_and_admin_sees_the_event(tmp_path):
    server, base, login = _http_setup(tmp_path)
    try:
        cookie = login("r@example.org", "passwordR123")
        ver = _json(f"{base}/api/items/c1", cookie=cookie)[1]["row_version"]
        status, payload, _ = _json(f"{base}/api/items/c1/edit-and-send", method="POST", cookie=cookie, data={"expected_version": ver, "edited_response": FACT})
        assert status == 422 and payload["error"] == "attestation_required" and payload["attestable_findings"] == list(FACT_CODES)
        assert payload["draft_preserved"] is True and payload["sent"] is False
        status, check, _ = _json(f"{base}/api/items/c1/validate-reply", method="POST", cookie=cookie, data={"reply": FACT})
        assert status == 200 and check["attestation_required"] is True and check["attestation_permitted"] is True and check["blocking_findings"] == []
        status, bad, _ = _json(f"{base}/api/items/c1/edit-and-send", method="POST", cookie=cookie,
                               data={"expected_version": ver, "edited_response": FACT + " https://example.org/x", "attestation": {"confirmed": True, "finding_codes": list(FACT_CODES)}})
        assert status == 422 and bad["error"] == "draft_validation_failed" and "unapproved_url" in bad["blocking_findings"]
        status, ok, _ = _json(f"{base}/api/items/c1/edit-and-send", method="POST", cookie=cookie,
                              data={"expected_version": ver, "edited_response": FACT, "attestation": {"confirmed": True, "finding_codes": list(FACT_CODES)}})
        assert status == 200 and ok["review_status"] == "EDITED_AND_APPROVED" and ok["final_response"] == FACT
        admin = login("a@example.org", "passwordA123")
        status, hist, _ = _json(f"{base}/api/admin/items/c1/history", cookie=admin)
        assert status == 200 and [e["event_type"] for e in hist["events"]] == [EVENT_EDIT_AND_SEND_ATTESTED]
        assert hist["events"][0]["actor_role"] == "REVIEWER"
    finally:
        server.shutdown()


def test_http_role_not_permitted_returns_403(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FACT_ATTESTATION_ROLES", frozenset({"ADMIN"}))
    server, base, login = _http_setup(tmp_path)
    try:
        cookie = login("r@example.org", "passwordR123")
        ver = _json(f"{base}/api/items/c1", cookie=cookie)[1]["row_version"]
        status, payload, _ = _json(f"{base}/api/items/c1/edit-and-send", method="POST", cookie=cookie,
                                   data={"expected_version": ver, "edited_response": FACT, "attestation": {"confirmed": True, "finding_codes": list(FACT_CODES)}})
        assert status == 403 and payload["error"] == "attestation_not_permitted" and payload["sent"] is False
        assert _json(f"{base}/api/items/c1/validate-reply", method="POST", cookie=cookie, data={"reply": FACT})[1]["attestation_permitted"] is False
    finally:
        server.shutdown()


def test_http_unauthenticated_cannot_attest(tmp_path):
    server, base, login = _http_setup(tmp_path)
    try:
        status, _, _ = _json(f"{base}/api/items/c1/edit-and-send", method="POST", data={"edited_response": FACT, "attestation": {"confirmed": True, "finding_codes": list(FACT_CODES)}})
        assert status == 401
    finally:
        server.shutdown()


# ------------------------------------------------------- both send actions enforce the identical policy (parity matrix)
PARITY = [
    ("clean", "نشكر لكم اهتمامكم بالمعجم.", None, None),
    ("fact-no-attestation", FACT, None, AttestationRequired),
    ("fact-wrong-codes", FACT, FactAttestation(True, ("unsupported_number",)), AttestationRequired),
    ("fact-attested", FACT, ATT, "attested"),
    ("promise", "سنقوم بمراجعة الملاحظة قريبا.", ATT, DraftValidationError),
    ("url", "راجعوا https://example.org/x", ATT, DraftValidationError),
    ("policy", "تخضع الاقتراحات لمراجعة لجنة متخصصة.", ATT, DraftValidationError),
    ("pii", "اتصلوا على 0501234567", ATT, DraftValidationError),
]


@pytest.mark.parametrize("name,text,att,expected", PARITY, ids=[p[0] for p in PARITY])
@pytest.mark.parametrize("who", [REVIEWER, NO_ROLE], ids=["reviewer", "no-role"])
def test_send_as_is_and_edit_and_send_reach_the_same_decision(tmp_path, name, text, att, expected, who):
    outcomes = []
    for i, action in enumerate(("as_is", "edit")):
        sub = tmp_path / action
        sub.mkdir()
        svc = _seed(_engine(sub), draft=text if action == "as_is" else None)
        try:
            if action == "as_is":
                svc.send_as_is("c1", who, attestation=att)
            else:
                svc.edit_and_send("c1", who, text, attestation=att)
            outcomes.append(("attested" if _types(svc)[-1].endswith("ATTESTED") else "ok"))
        except (AttestationRequired, AttestationNotPermitted, DraftValidationError) as exc:
            outcomes.append(type(exc).__name__)
    assert outcomes[0] == outcomes[1], (name, who.role, outcomes)
    if who is REVIEWER and expected is not None:
        assert outcomes[0] == ("attested" if expected == "attested" else expected.__name__)
    if who is NO_ROLE and expected in (AttestationRequired, "attested"):
        assert outcomes[0] == "AttestationNotPermitted"
