"""
Phase M5A.3 — validation-gate refactor parity tests.

Proves rcs.response_generation.validate's new gate structure
(GateResult / ValidationSummary / run_validation_gates) produces byte-
for-byte identical outcomes to the exact pre-refactor validate_draft
body -- reproduced verbatim below as `_reference_validate_draft`, a
frozen copy of the linear implementation that existed before this
phase (see the M5A.2 conversation history / git blame for the
original). This is the strongest form of parity proof: not "the new
code looks right," but "the new code and the old code agree on every
case tried," including deliberately multi-violation drafts that pin
down gate ORDER, not just individual outcomes.

No new pattern, threshold, or check is introduced anywhere in this
file. If a case here ever needs a new constant to express, that would
itself be scope creep for M5A.3.
"""

from __future__ import annotations

import re

import pytest

from config import settings
from rcs.classification.redact import redact_phone_numbers
from rcs.response_generation.service import GenerationRequest, generate_suggested_response
from rcs.response_generation.validate import (
    GateResult,
    ValidationSummary,
    _GENERIC_EXACT_DRAFTS,
    _LEAK_PATTERNS,
    _UNSUPPORTED_ACTION_PATTERNS,
    run_validation_gates,
    validate_draft,
)
from rcs.response_intelligence.orchestrator import generate_response_with_orchestrator

from test_response_generation import FakeGeminiClient, _draft_json  # noqa: E402 -- reuse the existing fake

_ARABIC_RE = re.compile(r"[؀-ۿ]")
_EMPTY_GROUNDING = {"", "(none — do not invent facts)"}


def _reference_normalize_exact(text: str) -> str:
    return " ".join(text.strip().split())


def _reference_grounding_is_empty(trusted_grounding_text: str | None) -> bool:
    return _reference_normalize_exact(trusted_grounding_text or "") in _EMPTY_GROUNDING


def _reference_validate_draft(text, *, trusted_grounding_text=None):
    """Frozen, verbatim copy of validate_draft's pre-M5A.3 linear body
    -- the oracle every new-code result is checked against below. Do
    NOT "clean up" or refactor this copy; its whole purpose is to stay
    exactly what the old code did, forever, as a comparison baseline."""
    if text is None or not str(text).strip():
        return False, "empty_output"

    draft = text.strip()
    if len(draft) < settings.RESPONSE_DRAFT_MIN_CHARS:
        return False, "too_short"
    if len(draft) > settings.RESPONSE_DRAFT_MAX_CHARS:
        return False, "excessive_length"

    arabic_count = len(_ARABIC_RE.findall(draft))
    if arabic_count < settings.RESPONSE_DRAFT_MIN_ARABIC_LETTERS:
        return False, "missing_arabic"

    for pattern in _LEAK_PATTERNS:
        if pattern.search(draft):
            return False, "internal_leak"

    if redact_phone_numbers(draft) != draft:
        return False, "pii_in_draft"

    if _reference_normalize_exact(draft) in _GENERIC_EXACT_DRAFTS:
        return False, "generic_boilerplate"

    grounding = trusted_grounding_text or ""
    for pattern in _UNSUPPORTED_ACTION_PATTERNS:
        if not pattern.search(draft):
            continue
        if _reference_grounding_is_empty(grounding) or not pattern.search(grounding):
            return False, "unsupported_operational_claim"

    return True, None


THANKS_DRAFT = "شكرًا لكلماتك الطيبة عن العمل على المعجم."
GENERIC_BOILERPLATE = "أهلًا بك، نشكرك على تواصلك واهتمامك بالمعجم."
UNSUPPORTED_CLAIM = "تم تدوين ملاحظتك وسيتم إصلاح المشكلة خلال 24 ساعة."


def _req(**overrides) -> GenerationRequest:
    base = dict(
        comment_id="c1",
        comment_text="شكرًا لكم على هذا العمل الرائع",
        primary_intent="THANKS_PRAISE",
        topic_subtype="GENERAL",
        sentiment="POSITIVE",
        confidence=0.9,
    )
    base.update(overrides)
    return GenerationRequest(**base)


# --- 1. Valid draft passes --------------------------------------------------


def test_valid_draft_passes_all_gates():
    """Ambiguous/Insufficient-Context Response Handling milestone added a
    7th gate, `visitor_intent_attribution`, appended last -- see
    validate.py's `_visitor_intent_attribution_gate`. Reply-safety hardening appended an 8th, `claim_gates` (rcs.response_generation.claim_gates)."""
    summary = run_validation_gates(THANKS_DRAFT)
    assert summary.passed is True
    assert summary.first_failure is None
    assert len(summary.evaluated_gates) == 9
    assert all(g.passed for g in summary.evaluated_gates)
    assert [g.gate_name for g in summary.evaluated_gates] == [
        "content_shape", "internal_leak", "pii", "boilerplate", "evaluator_voice", "unsupported_claim",
        "visitor_intent_attribution", "pronoun", "claim_gates",
    ]
    assert validate_draft(THANKS_DRAFT) == (True, None)


# --- 2. Every existing failure reason reproduced exactly --------------------


_SINGLE_VIOLATION_CASES = [
    pytest.param(None, {}, "empty_output", id="none_text"),
    pytest.param("   ", {}, "empty_output", id="blank_text"),
    pytest.param("قصير", {}, "too_short", id="too_short"),
    pytest.param("شكرًا " + ("ممتاز " * 200), {}, "excessive_length", id="excessive_length"),
    pytest.param("Thank you for your message to our team today.", {}, "missing_arabic", id="missing_arabic"),
    pytest.param("شكرًا، نموذج MARBERTv2 واثق بنسبة confidence عالية.", {}, "internal_leak", id="internal_leak"),
    pytest.param("شكرًا لتواصلكم على 0501234567 مع المعجم.", {}, "pii_in_draft", id="pii_in_draft"),
    pytest.param(GENERIC_BOILERPLATE, {}, "generic_boilerplate", id="generic_boilerplate"),
    pytest.param(UNSUPPORTED_CLAIM, {}, "unsupported_operational_claim", id="unsupported_claim_no_grounding"),
]


@pytest.mark.parametrize("draft,kwargs,expected_reason", _SINGLE_VIOLATION_CASES)
def test_failure_reason_matches_reference_exactly(draft, kwargs, expected_reason):
    reference = _reference_validate_draft(draft, **kwargs)
    current = validate_draft(draft, **kwargs)
    assert reference == current == (False, expected_reason)


# --- 3. First-failure ordering parity: multi-violation drafts ---------------


_MULTI_VIOLATION_CASES = [
    # too_short AND would-be internal leak if it were long enough to reach that gate --
    # content_shape gate (order 1) must win over internal_leak (order 2).
    pytest.param("CAMeL", {}, "too_short", id="short_wins_over_leak"),
    # excessive_length AND contains a leak pattern -- content_shape (length check)
    # must still win since it runs before the leak gate.
    pytest.param(
        "شكرًا " + ("ممتاز " * 200) + " MARBERTv2 confidence",
        {},
        "excessive_length",
        id="excessive_length_wins_over_leak",
    ),
    # internal_leak AND pii_in_draft both present -- internal_leak (order 2) must
    # win over pii (order 3).
    pytest.param(
        "شكرًا لكم، نموذج MARBERTv2 يتواصل معكم على 0501234567 دائما وبكل احترام.",
        {},
        "internal_leak",
        id="leak_wins_over_pii",
    ),
    # unsupported_operational_claim AND generic_boilerplate exact match is not
    # combinable (boilerplate requires an EXACT normalized match, which would not
    # also contain a claim pattern) -- instead prove boilerplate (order 4) wins
    # over unsupported_claim (order 5) using a draft that is boilerplate-exact
    # but would also be long/leak-free/pii-free, and confirm claim patterns in
    # the SAME text never get reached because boilerplate short-circuits first.
    pytest.param(GENERIC_BOILERPLATE, {}, "generic_boilerplate", id="boilerplate_wins_when_also_would_pass_claim_gate"),
]


@pytest.mark.parametrize("draft,kwargs,expected_reason", _MULTI_VIOLATION_CASES)
def test_first_failure_ordering_parity(draft, kwargs, expected_reason):
    reference = _reference_validate_draft(draft, **kwargs)
    current = validate_draft(draft, **kwargs)
    assert reference == current == (False, expected_reason)
    # cross-check against the gate runner directly: exactly one failing gate
    # reported, and it is the same one, and no gate after it was evaluated.
    summary = run_validation_gates(draft, **kwargs)
    assert summary.passed is False
    assert summary.first_failure.reason_code == expected_reason
    assert summary.evaluated_gates[-1] is summary.first_failure


def test_evaluated_gates_stops_at_first_failure_content_shape():
    """A content_shape failure must mean ONLY that one gate ran -- proving
    the runner truly short-circuits rather than accumulating results."""
    summary = run_validation_gates("قصير")
    assert len(summary.evaluated_gates) == 1
    assert summary.evaluated_gates[0].gate_name == "content_shape"


def test_evaluated_gates_stops_at_internal_leak():
    summary = run_validation_gates("شكرًا لكم، نموذج MARBERTv2 يتواصل معكم دائما وبكل احترام.")
    names = [g.gate_name for g in summary.evaluated_gates]
    assert names == ["content_shape", "internal_leak"]


# --- 4. trusted_grounding_text behavior parity ------------------------------


def test_grounding_allows_matching_operational_claim():
    grounding = "تم تحويل الملاحظة إلى فريق المحتوى."
    draft = "تم تحويل ملاحظتك إلى فريق المحتوى."
    reference = _reference_validate_draft(draft, trusted_grounding_text=grounding)
    current = validate_draft(draft, trusted_grounding_text=grounding)
    assert reference == current == (True, None)


def test_grounding_none_treated_same_as_empty_string():
    reference_none = _reference_validate_draft(UNSUPPORTED_CLAIM, trusted_grounding_text=None)
    reference_empty = _reference_validate_draft(UNSUPPORTED_CLAIM, trusted_grounding_text="")
    current_none = validate_draft(UNSUPPORTED_CLAIM, trusted_grounding_text=None)
    current_empty = validate_draft(UNSUPPORTED_CLAIM, trusted_grounding_text="")
    assert reference_none == reference_empty == current_none == current_empty == (False, "unsupported_operational_claim")


def test_grounding_placeholder_string_treated_as_empty():
    placeholder = "(none — do not invent facts)"
    reference = _reference_validate_draft(UNSUPPORTED_CLAIM, trusted_grounding_text=placeholder)
    current = validate_draft(UNSUPPORTED_CLAIM, trusted_grounding_text=placeholder)
    assert reference == current == (False, "unsupported_operational_claim")


def test_grounding_present_but_not_matching_claim_still_fails():
    grounding = "المعجم يستقبل ملاحظات عامة عن جودة الشرح."
    reference = _reference_validate_draft(UNSUPPORTED_CLAIM, trusted_grounding_text=grounding)
    current = validate_draft(UNSUPPORTED_CLAIM, trusted_grounding_text=grounding)
    assert reference == current == (False, "unsupported_operational_claim")


# --- 5. validate_draft public compatibility ---------------------------------


def test_validate_draft_signature_and_return_shape_unchanged():
    result = validate_draft(THANKS_DRAFT, trusted_grounding_text=None)
    assert isinstance(result, tuple)
    assert len(result) == 2
    ok, why = result
    assert isinstance(ok, bool)
    assert why is None or isinstance(why, str)


def test_leak_patterns_constant_still_importable_from_validate_module():
    """scripts/run_response_quality_eval_30.py imports _LEAK_PATTERNS
    directly from this module -- confirming that import path still
    resolves is a real backward-compatibility requirement, not a nice-
    to-have."""
    from rcs.response_generation.validate import _LEAK_PATTERNS as imported

    assert imported is _LEAK_PATTERNS
    assert len(imported) > 0


# --- 6. generate_suggested_response behavior unchanged ----------------------


def test_generate_suggested_response_validation_failure_unchanged():
    client = FakeGeminiClient([_draft_json(UNSUPPORTED_CLAIM)])
    result = generate_suggested_response(
        _req(comment_text="الموقع لا يعمل", primary_intent="COMPLAINT"), client, max_retries=0
    )
    assert result.generation_status == "FAILED_REVIEW_ONLY"
    assert result.routing_decision == "REVIEW_ONLY"
    assert result.suggested_response is None
    assert result.reason_code == "unsupported_operational_claim"
    assert result.review_required is True
    assert client.call_count == 1


def test_generate_suggested_response_valid_draft_unchanged():
    client = FakeGeminiClient([_draft_json(THANKS_DRAFT)])
    result = generate_suggested_response(_req(), client)
    assert result.generation_status == "SUCCEEDED"
    assert result.suggested_response == THANKS_DRAFT
    assert client.call_count == 1


# --- 7. orchestrator behavior unchanged -------------------------------------


def test_orchestrator_validation_failure_parity():
    """Phase M5D note: this must use a NON-refinement-eligible failure
    (pii_in_draft) rather than the original GENERIC_BOILERPLATE fixture.
    Since M5D, "generic_boilerplate" is a refinement-eligible reason
    code -- the orchestrator now deliberately makes a second, bounded
    Gemini call for it that a direct `generate_suggested_response` call
    never makes, so strict parity no longer holds for THAT code (see
    tests/test_m5d_refinement.py for the intentional-divergence proof).
    Parity still holds, and is proven here, for a reason code that is
    NOT refinement-eligible."""
    pii_draft = "شكرًا لتواصلكم على 0501234567 مع المعجم."
    client_direct = FakeGeminiClient([_draft_json(pii_draft)])
    client_orchestrated = FakeGeminiClient([_draft_json(pii_draft)])
    result_direct = generate_suggested_response(_req(), client_direct, max_retries=0)
    result_orchestrated = generate_response_with_orchestrator(_req(), client_orchestrated, max_retries=0)
    d1 = result_direct.to_dict()
    d2 = result_orchestrated.to_dict()
    d1.pop("generated_at")
    d2.pop("generated_at")
    assert d1 == d2
    assert d1["reason_code"] == "pii_in_draft"
    assert client_direct.call_count == client_orchestrated.call_count == 1


# --- 8. validation failures do not cause Gemini transport retries ----------


def test_validation_failure_makes_exactly_one_gemini_call_even_with_retries_available():
    client = FakeGeminiClient([_draft_json(UNSUPPORTED_CLAIM)])  # would raise if called again
    result = generate_suggested_response(_req(), client, max_retries=2)
    assert result.generation_status == "FAILED_REVIEW_ONLY"
    assert client.call_count == 1  # a valid schema response that fails validation is never transport-retried


# --- 9. no LLM evaluator call exists -----------------------------------------


def test_no_llm_evaluator_in_validation_module():
    """Static guard: validate.py must not import or call anything
    Gemini/LLM-shaped -- every gate here is a deterministic regex/length
    check, not a model judgment."""
    import ast
    import inspect

    import rcs.response_generation.validate as validate_module

    tree = ast.parse(inspect.getsource(validate_module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    banned = ("genai", "gemini", "llm", "openai", "langgraph", "agentcore")
    for name in imported_names:
        lowered = name.lower()
        assert not any(b in lowered for b in banned), f"unexpected LLM-evaluator-shaped import: {name!r}"


# --- 10. no new policy/tone/RAG behavior exists ------------------------------


def test_no_new_gate_concepts_defined():
    """Static guard: only the audited gates exist -- no Policy, Tone,
    Grounding-extension, or Routing gate. Originally asserted exactly
    five gates (M5A.3's structural refactor added no new gate). rg-v9
    (2026-09) intentionally added ONE new, narrow, evidence-anchored
    gate (`_evaluator_voice_gate`, see its own docstring) after the
    first real human review found detached-evaluator-voice openings in
    live output -- this assertion was updated for that one deliberate
    addition, not loosened generally. The Ambiguous/Insufficient-Context
    Response Handling milestone added a second such gate,
    `_visitor_intent_attribution_gate` -- again one narrow,
    evidence-anchored addition (see its own docstring for the real
    audit evidence), not a loosening of this guard."""
    import inspect

    import rcs.response_generation.validate as validate_module

    gate_functions = [
        name for name, obj in inspect.getmembers(validate_module)
        if inspect.isfunction(obj) and name.endswith("_gate")
    ]
    assert set(gate_functions) == {
        "_content_shape_gate", "_internal_leak_gate", "_pii_gate", "_boilerplate_gate",
        "_evaluator_voice_gate", "_unsupported_claim_gate", "_visitor_intent_attribution_gate", "_pronoun_gate",
    }


def test_validation_module_has_no_retrieval_or_memory_imports():
    import ast
    import inspect

    import rcs.response_generation.validate as validate_module

    tree = ast.parse(inspect.getsource(validate_module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    banned = ("tool", "retriev", "memory", "embedding", "vector")
    for name in imported_names:
        lowered = name.lower()
        assert not any(b in lowered for b in banned), f"unexpected M5A.4+ import: {name!r}"
