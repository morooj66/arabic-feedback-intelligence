"""Privacy preflight for the Gemini classification evaluation, reviewed with SYNTHETIC adversarial comments (all names, numbers and handles are fictional).

Finding recorded by these tests: the production sanitiser alone lets most name-bearing adversarial comments through; the harness residual-risk screen blocks
many more but is a HEURISTIC, so the hash-approved human review of the exact inputs remains mandatory. Battery 3 was NOT used to tune the screen and still leaks."""
from __future__ import annotations

import json
import os
import re
import stat

import pytest

from rcs.response_generation.outbound_privacy import sanitize_text
from scripts.offline_eval.gemini_classification import compare_real as cr
from scripts.offline_eval.gemini_classification import real_data as rd
from scripts.offline_eval.gemini_classification import sanitized_inputs as si
from scripts.offline_eval.gemini_classification import synthetic_cases as sc
from scripts.offline_eval.gemini_classification.client import MockGeminiClient, extract_comment
from scripts.offline_eval.gemini_classification.privacy import LiveApproval, RealDataNotApproved, preflight
from scripts.offline_eval.gemini_classification.runner import Record

BATTERY_1 = [
 # (category, text, fragments that must NOT reach Gemini)
 ("bare_name_signature_line", "أقترح إضافة كلمة جديدة إلى المعجم\nنورة", ["نورة"]),
 ("bare_name_after_thanks", "شكرا لكم على الجهد المبذول يا ريم", ["ريم"]),
 ("greeting_with_name", "السلام عليكم أنا سلطان وأقترح إضافة كلمة", ["سلطان"]),
 ("my_name_is_single", "اسمي ماجد وأقترح تعديل التعريف", ["ماجد"]),
 ("my_name_full", "اسمي ماجد بن سعد الدوسري وأقترح تعديل التعريف", ["ماجد", "الدوسري"]),
 ("kunya_abu", "شكرا يا أبا خالد على الرد السريع", ["خالد"]),
 ("kunya_umm", "تقول أم فهد إن الكلمة تنطق هكذا", ["فهد"]),
 ("nasab_chain", "الكلمة ذكرها فهد بن ناصر بن عبدالعزيز في كتابه", ["ناصر بن عبدالعزيز"]),
 ("relative_named_grandmother", "جدتي هيفاء تستخدم هذه الكلمة دائما", ["هيفاء"]),
 ("relative_named_son", "ابني ياسر سألني عن معنى الكلمة", ["ياسر"]),
 ("colleague_named", "زميلي عبدالإله يقول إن التعريف خطأ", ["عبدالإله"]),
 ("title_plus_name", "ذكر الدكتور عبدالله الغامدي أن الكلمة دخيلة", ["الغامدي"]),
 ("signature_dash", "أرجو مراجعة المدخل\n- منيرة العتيبي", ["منيرة", "العتيبي"]),
 ("signature_regards_ar", "أرجو مراجعة المدخل. مع خالص التقدير: فواز", ["فواز"]),
 ("latin_signature", "Please review this entry. Thanks, Layla Hassan", ["Layla Hassan"]),
 ("latin_bare_name", "Great dictionary, regards Omar", ["Omar"]),
 ("handle_without_at", "تابعوني على انستغرام sara_test_account", ["sara_test_account"]),
 ("handle_platform_word", "حسابي في سناب: layla.test99", ["layla.test99"]),
 ("email_obfuscated_at_dot", "راسلوني على sara dot test at example dot invalid", ["example dot invalid", "sara dot test"]),
 ("email_obfuscated_paren", "بريدي sara.test(at)example.invalid", ["example.invalid"]),
 ("phone_spelled_digits", "اتصلوا صفر خمسة صفر صفر صفر صفر صفر صفر ثلاثة", ["صفر خمسة صفر"]),
 ("phone_arabic_indic_spaced", "رقمي ٠٥٠ ٠٠٠ ٠٠٠٣", ["٠٥٠ ٠٠٠ ٠٠٠٣"]),
 ("phone_dots", "رقمي 050.000.0003 للتواصل", ["050.000.0003"]),
 ("phone_plus_966_spaces", "واتساب +966 50 000 0003", ["50 000 0003"]),
 ("id_number_no_cue", "الرقم 1000000003 يخصني", ["1000000003"]),
 ("address_street", "أسكن في حي الياسمين شارع الأمير سلطان بالرياض", ["حي الياسمين"]),
 ("employer_role", "أعمل مهندسا في شركة الأفق الفني بجدة", ["شركة الأفق"]),
 ("name_in_url_path", "رابط ملفي example.invalid/u/layla-hassan", ["layla-hassan"]),
 ("diacritised_self_intro", "أَنَا مَاجِد وَأَقْتَرِحُ إِضَافَةَ كَلِمَةٍ", ["مَاجِد"]),
 ("elongated_self_intro", "أنـــا فهــــد وأقترح إضافة كلمة", ["فهد", "فهــــد"]),
 ("name_after_hamza_variant", "انا عبدالرحمن اقترح اضافة كلمة", ["عبدالرحمن"]),
 ("third_party_in_passing", "قال لي صديقي بندر إن المعجم ينقصه كلمة", ["بندر"]),
 ("child_name_sensitive", "ابنتي لمى عمرها ٧ سنوات تحب المعجم", ["لمى"]),
 ("health_circumstance", "أنا مريض بالسرطان وأتمنى إضافة كلمة", ["السرطان"]),
 ("mixed_name_phone_email", "أنا ريم الحربي 0500000009 sara.test@example.invalid", ["ريم", "الحربي", "0500000009", "sara.test"]),
]
PRESERVE = [
 ("headword_is_a_name", "أقترح إضافة كلمة نورة إلى المعجم مع شرح معناها", ["كلمة نورة"]),
 ("what_does_name_mean", "ما معنى اسم أراس وما أصله", ["أراس"]),
 ("historical_figure_quoted", "قال الجاحظ في كتاب الحيوان إن الكلمة قديمة", ["الجاحظ"]),
 ("plain_suggestion", "أقترح إضافة خاصية البحث بالجذر", ["أقترح إضافة خاصية البحث بالجذر"]),
]

BATTERY_2 = [
 ("dash_name_single_line", "شكرا لكم — عبدالمجيد", ["عبدالمجيد"]),
 ("from_brother_name", "تحياتي لكم من الأخ يزيد", ["يزيد"]),
 ("role_plus_name", "هذا رأيي أنا الباحث سعود", ["سعود"]),
 ("supervisor_named", "كتب لي المشرف فيصل ملاحظة على الشرح", ["فيصل"]),
 ("phone_whatsapp_cue", "محتاج مساعدة ابعثوا لي على الواتس 0500000012", ["0500000012"]),
 ("from_city_my_name", "أنا من حائل واسمي رنا", ["رنا"]),
 ("contact_person_named", "ممكن تتواصلون مع لولوة بخصوص التعديل", ["لولوة"]),
 ("agreed_with_title", "تم الاتفاق مع الأستاذ رائد على التعديل", ["رائد"]),
 ("sent_by_latin", "Please add this word. Sent by: Tariq", ["Tariq"]),
 ("sender_label_ar", "المرسل: سمير", ["سمير"]),
 ("handle_at_inline", "حسابي @sara_x للتواصل", ["sara_x"]),
 ("phone_mixed_digits_words", "جوالي صفر٥٠٠٠٠٠٠١٣", ["٥٠٠٠٠٠٠١٣"]),
 ("insta_dot_handle", "my insta is sara.test", ["sara.test"]),
 ("headword_and_relative", "كلمة ريما اسم بنت أختي ريما", ["ريما"]),
 ("honorific_name", "شكرا أخي الكريم محمد على الرد", ["محمد"]),
 ("names_list_after_colon", "يسعدني أن أشكر فريق العمل: ندى ومها", ["ندى", "مها"]),
 ("bare_single_name", "عبدالله", ["عبدالله"]),
 ("affiliation_student", "أنا طالب دراسات عليا في جامعة الملك سعود وأقترح إضافة كلمة", ["جامعة الملك سعود"]),
 ("greeting_to_named_staff", "السلام عليكم الأخت هند أرجو مراجعة المدخل", ["هند"]),
 ("name_before_wa", "شكرا ليلى وجزاكم الله خيرا", ["ليلى"]),
]

BATTERY_3 = [
 ("closing_variant_full_name", "مع تحيات هيفاء المطيري", ["هيفاء", "المطيري"]),
 ("writer_of_message", "كاتب هذه الرسالة غازي", ["غازي"]),
 ("name_label_ar", "الاسم الكريم: بدر", ["بدر"]),
 ("send_result_to_named", "ارسلوا النتيجة للأخ تركي", ["تركي"]),
 ("thanks_to_named_ms", "شكرا للأستاذة سميرة على التعاون", ["سميرة"]),
 ("cousin_named", "ابنة عمي مها تقول الكلمة", ["مها"]),
 ("elder_named", "أخبرني عنها العم سالم", ["سالم"]),
 ("email_obfuscated_at_dot_com", "contact me: layla at example dot com", ["layla at example"]),
 ("phone_spelled_long", "رقمي تسعة ستة ستة خمسة صفر صفر صفر صفر صفر ثلاثة", ["تسعة ستة ستة"]),
 ("twitter_handle_label", "تويتر: @layla_x", ["layla_x"]),
 ("visited_with_daughter", "زرت المعجم مع ابنتي فرح", ["فرح"]),
 ("headword_and_sister_name", "كلمة سندس هي اسم أختي وأريد معناها", ["سندس"]),
 ("department_head_named", "مدير القسم السيد أنس قال لي", ["أنس"]),
 ("dr_abbrev_signature", "وفقكم الله — د. ماهر", ["ماهر"]),
]


def _leaks(battery, fn):
    return {c for c, t, secrets in battery if (r := fn(t)) is not None and any(s in r for s in secrets)}


def _production(text):
    return sanitize_text(text, names=True).text


def _harness(text):
    p = preflight(text)
    return p.text if p.eligible else None


def test_batteries_one_and_two_have_no_leak_through_the_harness_preflight():
    assert _leaks(BATTERY_1, _harness) == set() and _leaks(BATTERY_2, _harness) == set()


def test_the_production_sanitiser_alone_is_not_enough():
    assert len(_leaks(BATTERY_1, _production)) >= 10 and len(_leaks(BATTERY_2, _production)) >= 10      # documented: rule-based masking misses bare names


# Known residual risk, measured on a battery that was not used to build the screen. This list may only shrink.
KNOWN_RESIDUAL_BATTERY_3 = {"closing_variant_full_name", "writer_of_message", "send_result_to_named", "elder_named", "dr_abbrev_signature"}


def test_untuned_battery_still_leaks_which_is_why_human_review_is_mandatory():
    harness, production = _leaks(BATTERY_3, _harness), _leaks(BATTERY_3, _production)
    assert harness <= KNOWN_RESIDUAL_BATTERY_3 and len(harness) < len(production)
    assert len(harness) >= 1          # if this ever reads 0, re-validate on a NEW untuned battery before trusting it


@pytest.mark.parametrize("cat,text,keep", PRESERVE)
def test_dictionary_content_survives(cat, text, keep):
    p = preflight(text)
    assert p.eligible and all(k in p.text for k in keep)


def test_benign_synthetic_comments_are_not_over_blocked():
    blocked = [r.record_id for r in sc.records() if not preflight(r.text).eligible]
    assert len(blocked) <= 8 and {"s031", "s036", "s050"} <= set(blocked)          # kinship story, abuse, injection token; plus bare words
    for rid in ("s001", "s006", "s012", "s016", "s020", "s026", "s052", "s053"):
        assert rid not in blocked


def test_masking_must_not_depend_on_diacritics_or_tatweel():
    assert preflight("أَنَا مَاجِد وَأَقْتَرِحُ إِضَافَةَ كَلِمَةٍ").eligible is False
    assert preflight("أنـــا فهــــد وأقترح إضافة كلمة").eligible is False


# ----------------------------------------------------------------------------------------------------------------- hash-approved inputs gate
def _real_sets():
    gold, recs = sc.gold_rows(), sc.records()
    def mk(name, lo, hi):
        rs = [Record(r.record_id, r.text, provenance="real", dataset=name) for r in recs[lo:hi]]
        return rd.EvalSet(name, rs, gold[lo:hi], sc.degrade(gold[lo:hi], flip_rate=0.5, seed=lo))
    return [mk("human_test", 0, 27), mk("blind_natural", 27, 54)]


def _approval(hashes, calls=200):
    return LiveApproval("owner", "2026-10-16", "test", calls, 5.0, ("human_test", "blind_natural"), tuple(sorted(hashes.items())))


def test_build_writes_owner_only_files_and_a_stable_hash(tmp_path):
    recs = [Record(r.record_id, r.text, provenance="real", dataset="human_test") for r in sc.records()]
    info = si.build(recs, "human_test", tmp_path)
    again = si.build(recs, "human_test", tmp_path)
    assert info["sha256"] == again["sha256"] and info["sent_candidates"] + info["quarantined"] == len(recs) and info["quarantined"] >= 1
    for name in info["files"][:2]:
        assert stat.S_IMODE(os.stat(tmp_path / name).st_mode) == 0o600
    assert info["sha256"] in (tmp_path / "human_test_INSPECTION.md").read_text(encoding="utf-8")
    q = json.loads((tmp_path / "human_test_quarantined.json").read_text(encoding="utf-8"))
    assert all(v.startswith(("residual_risk", "empty")) or v for v in q.values())
    body = (tmp_path / "human_test_sanitized_inputs.jsonl").read_text(encoding="utf-8")
    assert "0500000046" not in body and "example.invalid" not in body and "الزهراني" not in body


def test_run_live_sends_only_the_hash_approved_sanitised_text(tmp_path):
    sets = _real_sets()
    hashes = {s.name: si.build(s.records, s.name, tmp_path)["sha256"] for s in sets}
    client = MockGeminiClient()
    res = cr.run_live(_approval(hashes), client=client, sets=_real_sets(), out_dir=tmp_path / "out", inputs_dir=tmp_path, sleep=lambda s: None)
    sent = sum(info["gemini_ok"] for info in res["sets"].values())
    assert sent >= 1 and client.calls >= sent
    assert all(("0500000046" not in p and "example.invalid" not in p and "الزهراني" not in p) for p in client.prompts)
    quarantined = {k for n in hashes for k in json.loads((tmp_path / f"{n}_quarantined.json").read_text(encoding="utf-8"))}
    sent_comments = {extract_comment(p) for p in client.prompts}
    withheld = [r for s in _real_sets() for r in s.records if r.record_id in quarantined]
    assert withheld and all(r.text not in sent_comments for r in withheld)


def test_a_changed_input_file_blocks_every_call(tmp_path):
    sets = _real_sets()
    hashes = {s.name: si.build(s.records, s.name, tmp_path)["sha256"] for s in sets}
    p = si.jsonl_path("human_test", tmp_path)
    p.write_text(p.read_text(encoding="utf-8") + json.dumps({"id": "zz", "sanitized_text": "نص مضاف"}, ensure_ascii=False) + "\n", encoding="utf-8")
    client = MockGeminiClient()
    with pytest.raises(si.InputHashMismatch):
        cr.run_live(_approval(hashes), client=client, sets=_real_sets(), out_dir=tmp_path / "out", inputs_dir=tmp_path, sleep=lambda s: None)
    assert client.calls == 0


def test_real_data_without_an_approved_hash_is_refused_before_any_call(tmp_path):
    client = MockGeminiClient()
    with pytest.raises(RealDataNotApproved):
        cr.run_live(_approval({"blind_natural": "a" * 64}), client=client, sets=_real_sets(), out_dir=tmp_path, inputs_dir=tmp_path, sleep=lambda s: None)
    assert client.calls == 0
