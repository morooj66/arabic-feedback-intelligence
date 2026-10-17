"""Matching helpers behind the base-model error analysis (normalisation + expression families)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "offline_eval" / "error_analysis"))

import common as C  # noqa: E402


def test_normalisation_removes_diacritics_and_unifies_letter_variants():
    assert C.norm_ar("يَفْتَـح") == "يفتح"
    assert C.norm_ar("أَحمد إلى آخر") == C.norm_ar("احمد الي اخر")
    assert C.norm_ar("مشكلة") == C.norm_ar("مشكله")                       # ta-marbuta / ha unified


@pytest.mark.parametrize("text,expected", [
    ("الموقع ما يفتح عندي", True), ("الصفحة ما تظهر", True), ("الموقع يَعْلَق", True), ("حاولت وما ضبط", True),
    ("كل ما أبحث ما يطلع لي شي", True), ("فيه مشكلة في البحث", True), ("النتيجة غلط", True),
    ("كيف أبحث عن كلمة؟", False), ("أتمنى إضافة خاصية", False), ("شكرا لكم", False),
])
def test_failure_expression_families_match_reports_not_neutral_text(text, expected):
    assert C.any_failure(text) is expected


def test_contrast_cues_separate_how_to_and_feature_requests_from_failure_reports():
    assert C.hits("كيف أسجل دخول؟", C.CONTRAST["howto_question"])
    assert C.hits("ياليت يكون فيه تطبيق", C.CONTRAST["feature_request"])
    assert not C.hits("الموقع ما يفتح", C.CONTRAST["howto_question"])
    assert C.hits("هل البحث يشتغل بدون تشكيل؟", C.CONTRAST["howto_question"]) and C.hits("البحث يشتغل", ["يشتغل"])


def test_analysis_modules_never_import_the_blind_evaluator_or_fit_on_blind_labels():
    for name in ("intent_cv_experiment.py", "sentiment_diagnostics.py"):
        src = (Path(C.__file__).parent / name).read_text(encoding="utf-8")
        fit_lines = [l for l in src.splitlines() if ".fit(" in l]
        assert fit_lines and not any("blind" in l.lower() for l in fit_lines)      # nothing is fitted on blind data
    assert "load_blind" not in (Path(C.__file__).parent / "intent_cv_experiment.py").read_text(encoding="utf-8")
