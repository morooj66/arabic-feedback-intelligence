"""
Tests for the extended composition instruction that closes a detached-
topic-preamble register gap: an inquiry reply that opens with a
sentence-initial topic-relating clause ("بخصوص/بشأن/يتعلق بـ/يتصل بـ +
a restatement of the question") before engaging the point is the same
detached, summary-shaped register the composition instruction already
forbids under other connector words.

These tests deliberately check the BEHAVIORAL distinction the prompt
now draws, not a banned-word list: the same connector words remain
explicitly valid when used naturally inside a sentence that is already
engaging the point rather than opening by relating/labeling it.
"""

from __future__ import annotations

from config import settings
from rcs.response_generation.prompt import SYSTEM_INSTRUCTIONS


def test_prompt_names_the_additional_connector_verb_family():
    for phrase in ("يتعلق تساؤلكم بـ", "يتصل استفساركم بـ"):
        assert phrase in SYSTEM_INSTRUCTIONS


def test_prompt_states_the_underlying_behavior_not_a_word_ban():
    assert "SENTENCE-OPENING labeling function, not these particular words" in SYSTEM_INSTRUCTIONS


def test_prompt_gives_a_valid_natural_use_example_with_the_same_words():
    """The exact same connector word ("بشأن") must appear in an example
    the prompt explicitly calls fine, proving this is a register
    distinction, not a lexical ban."""
    assert 'نرحب باستفساركم بشأن' in SYSTEM_INSTRUCTIONS
    assert 'نقدّر استفساركم اللغوي بشأن' in SYSTEM_INSTRUCTIONS


def test_prompt_ties_the_new_guidance_to_the_existing_rg_v10_shape():
    """This must read as an EXTENSION of the existing named shape, not a
    freestanding new rule -- proves no redesign occurred."""
    assert "identical forbidden shape under different connector words" in SYSTEM_INSTRUCTIONS


def test_no_fixed_opening_or_abrupt_reply_is_required():
    """Expressive flexibility must remain explicit: no single required
    opening, and replies are not forced to be abrupt."""
    assert "Vary openings and closings naturally across replies" in SYSTEM_INSTRUCTIONS
    assert "no fixed reply shape or" in SYSTEM_INSTRUCTIONS or "no fixed order" in SYSTEM_INSTRUCTIONS


def test_prompt_version_is_rg_v13():
    assert settings.RESPONSE_GENERATION_PROMPT_VERSION == "rg-v14.4"


def test_no_new_validator_pattern_for_this_register_issue():
    """This is a prompt-only clarification -- the deterministic
    validator gained no new pattern for it (a bounded phrase check here
    would risk exactly the false positives on legitimate mid-sentence
    use this change is careful to avoid)."""
    from rcs.response_generation.validate import _UNSUPPORTED_ACTION_PATTERNS

    assert len(_UNSUPPORTED_ACTION_PATTERNS) == 45
    joined = "|".join(p.pattern for p in _UNSUPPORTED_ACTION_PATTERNS)
    for banned_addition in ("يتعلق", "يتصل", "بخصوص", "بشأن"):
        assert banned_addition not in joined


def test_no_new_gate_or_reason_code_was_added():
    import inspect

    import rcs.response_generation.validate as validate_module
    from rcs.response_generation.refinement import REFINEMENT_ELIGIBLE_REASON_CODES

    gate_functions = {
        name for name, obj in inspect.getmembers(validate_module)
        if inspect.isfunction(obj) and name.endswith("_gate")
    }
    assert gate_functions == {
        "_content_shape_gate", "_internal_leak_gate", "_pii_gate", "_boilerplate_gate",
        "_evaluator_voice_gate", "_unsupported_claim_gate", "_visitor_intent_attribution_gate", "_pronoun_gate",
    }
    # The reply-safety milestone later ADDED refinable codes (see tests/test_reply_grounding_and_refinement.py for the exact set); this milestone's own
    # set must still be contained in it.
    assert frozenset(
        {
            "unsupported_operational_claim",
            "generic_boilerplate",
            "empty_output",
            "too_short",
            "excessive_length",
            "missing_arabic",
            "evaluator_voice_opening",
            "unsupported_visitor_intent_claim",
        }
    ) <= REFINEMENT_ELIGIBLE_REASON_CODES


def test_prior_named_shapes_and_hard_rules_untouched():
    """rg-v9/rg-v10/rg-v11/rg-v12 wording must remain exactly as it was
    -- this is an extension, not a rewrite."""
    assert 'حول/بخصوص ما طرحتموه/عرضتموه بشأن' in SYSTEM_INSTRUCTIONS
    assert "ملاحظة دقيقة" in SYSTEM_INSTRUCTIONS
    assert "Do not attribute a specific action, request, or proposal to the visitor" in SYSTEM_INSTRUCTIONS
    assert (
        "never state or imply that you, this draft, or the institution has performed, "
        "is performing, or will perform that action"
    ) in SYSTEM_INSTRUCTIONS
