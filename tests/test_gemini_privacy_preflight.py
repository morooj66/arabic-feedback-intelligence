"""Outbound-Gemini privacy preflight pinned as regression tests (synthetic fixtures only; capture is at the transport seam, AFTER the final privacy boundary).
Every identifier / personal-name / link / handle fixture must be absent from the outbound prompt, and dictionary content must survive."""
import pytest

from scripts.offline_eval.reply_quality import privacy_preflight as PF

BY_CAT = {c: (t, b) for c, t, b in PF.FIXTURES}
PRESERVE = {c: (t, k) for c, t, k in PF.PRESERVE}


@pytest.mark.parametrize("category", list(BY_CAT))
def test_nothing_identifying_reaches_the_model_seam(category):
    text, banned = BY_CAT[category]
    prompt = PF.capture(text)
    assert [b for b in banned if b in prompt] == []


@pytest.mark.parametrize("category", list(PRESERVE))
def test_dictionary_content_survives_sanitisation(category):
    text, keep = PRESERVE[category]
    prompt = PF.capture(text)
    assert [k for k in keep if k not in prompt] == []


def test_entry_title_and_internal_entry_id_do_not_carry_identifiers():
    for cat, title, banned in PF.TITLE_FIXTURES:
        assert not [b for b in banned if b in PF.capture("أقترح إضافة خاصية البحث بالجذر في المعجم", title=title)], cat
    assert "ENTRY-SYN-1" not in PF.capture("أقترح إضافة خاصية البحث بالجذر", title="كلمة")


def test_the_preflight_script_reports_pass():
    assert PF.main() == 0
