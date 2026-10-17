"""PRIVACY PREFLIGHT for the Gemini request: builds the EXACT outbound prompt through the production path (dto reviewer-facing redaction -> GenerationRequest ->
run_agent_pipeline -> generate_once_with_draft -> build_generation_prompt) with a capturing client at the transport seam, using ONLY local synthetic fixtures.
No network, no Gemini, no database. Exit code 1 if any fixture leaks. Output: data/evaluation/reply_quality/privacy_preflight.json (fixture categories + verdicts only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from rcs.response_generation.service import GenerationRequest  # noqa: E402
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata  # noqa: E402
from rcs.response_intelligence.orchestrator import run_agent_pipeline  # noqa: E402
from rcs.response_intelligence.retrieval.contracts import SUCCESS_EMPTY, ApprovedResponseMemoryRetrievalResult, OfficialKnowledgeRetrievalResult  # noqa: E402
from rcs.review.dto import _reviewer_facing_text  # noqa: E402
from scripts.offline_eval.response_agent.fakes import FakeApprovedResponseMemoryRetriever, FakeGeminiClient, FakeOfficialKnowledgeRetriever  # noqa: E402

OUT = ROOT / "data" / "evaluation" / "reply_quality"; OUT.mkdir(parents=True, exist_ok=True)
SAFE = json.dumps({"suggested_response": "نشكر لكم اقتراحكم إضافة خاصية البحث بالجذر."}, ensure_ascii=False)

# (category, comment text, strings that MUST NOT reach the model). Everything below is synthetic.
FIXTURES = [
    ("phone_local", "أقترح إضافة خاصية البحث بالجذر وللتواصل 0501234567 شكرا", ["0501234567"]),
    ("phone_international", "أقترح إضافة خاصية البحث +966 50 123 4567 شكرا", ["50 123 4567", "966"]),
    ("phone_arabic_indic_digits", "أقترح إضافة خاصية البحث اتصلوا ٠٥٠١٢٣٤٥٦٧", ["٠٥٠١٢٣٤٥٦٧"]),
    ("phone_dashes", "أقترح إضافة خاصية البحث 050-123-4567 شكرا", ["050-123-4567", "123-4567"]),
    ("national_id_10_digits", "أقترح إضافة خاصية البحث هويتي 1012345678", ["1012345678"]),
    ("email", "أقترح إضافة خاصية البحث راسلوني على sara.test@example.com", ["sara.test@example.com", "example.com"]),
    ("email_uppercase", "أقترح إضافة خاصية البحث SARA.TEST@EXAMPLE.ORG", ["SARA.TEST@EXAMPLE.ORG"]),
    ("name_self_identification_ar", "أنا محمد بن عبدالله العتيبي وأقترح إضافة خاصية البحث بالجذر", ["محمد بن عبدالله العتيبي", "العتيبي"]),
    ("name_my_name_is_ar", "اسمي فاطمة الزهراني وأقترح إضافة خاصية البحث بالجذر", ["فاطمة الزهراني", "الزهراني"]),
    ("name_signature_ar", "أقترح إضافة خاصية البحث بالجذر. تحياتي خالد السبيعي", ["خالد السبيعي", "السبيعي"]),
    ("name_english", "My name is John Smith and I suggest adding root search", ["John Smith"]),
    ("url_or_handle", "أقترح إضافة خاصية البحث تابعوني على @sara_test_account وhttps://example.org/profile/sara", ["@sara_test_account", "example.org/profile"]),
    ("iban_like", "أقترح إضافة خاصية البحث حسابي SA0380000000608010167519", ["SA0380000000608010167519"]),
]
FIXTURES += [
    ("name_strong_intro_maak", "معك أحمد السالم وأقترح إضافة خاصية البحث بالجذر", ["أحمد السالم", "السالم"]),
    ("name_kinship_nasab", "جدتي نورة بنت سعد بن فهد تقول الكلمة هكذا وأقترح إضافتها", ["نورة بنت سعد", "سعد بن فهد"]),
    ("name_brother_intro", "اخوكم عبدالرحمن يقترح إضافة خاصية البحث بالجذر", ["عبدالرحمن"]),
    ("name_formal_signature", "أقترح إضافة خاصية البحث بالجذر. مع تحياتي: سلطان الدوسري", ["سلطان الدوسري", "الدوسري"]),
    ("name_first_person_frequent_in_corpus", "أنا نوف وأقترح إضافة خاصية البحث بالجذر", ["نوف"]),
    ("name_first_person_second", "أنا هيثم وأتمنى إضافة خاصية البحث بالجذر", ["هيثم"]),
    ("name_kinship_frequent_in_corpus", "أمي موضي تقول هذه الكلمة بمعنى آخر وأقترح إضافته", ["موضي"]),
    ("name_friend_kinship", "صديقي عبدالعزيز يقول الكلمة هكذا وأقترح إضافة المعنى", ["عبدالعزيز"]),
    ("name_latin_regards", "I suggest adding root search. Regards, Sarah Johnson", ["Sarah Johnson"]),
    ("name_latin_i_am", "I am Ahmed Khan and I suggest adding root search", ["Ahmed Khan"]),
    ("email_glued_to_arabic_conjunction", "أقترح إضافة خاصية البحث وsara.test@example.com للتواصل", ["sara.test@example.com"]),
    ("url_glued_without_scheme", "أقترح إضافة خاصية البحث وexample.org/profile/sara للمزيد", ["example.org/profile"]),
    ("handle_glued_to_arabic_conjunction", "أقترح إضافة خاصية البحث و@sara_test_account للتواصل", ["@sara_test_account"]),
    ("url_www", "أقترح إضافة خاصية البحث www.example.org/page", ["www.example.org"]),
    ("name_and_phone_together", "أنا ريم الحربي وجوالي 0555000111 وأقترح إضافة خاصية البحث", ["ريم الحربي", "الحربي", "0555000111"]),
]

# Dictionary content that must SURVIVE sanitisation (meaning preservation): (category, text, fragments that must remain)
PRESERVE = [
    ("headword_that_is_a_name", "أقترح إضافة كلمة محمد إلى المعجم مع شرح معناها", ["كلمة محمد"]),
    ("labelled_headword_name", "الكلمة: نوف اسم علم مؤنث وأقترح إضافتها", ["الكلمة: نوف"]),
    ("first_person_verb_suggest", "أنا أقترح استخدام مفردة الشابكة بدل الانترنت", ["أقترح استخدام مفردة الشابكة"]),
    ("first_person_verb_see", "أنا أرى أن تعريف كلمة سوار غير دقيق", ["أرى أن تعريف كلمة سوار"]),
    ("first_person_role", "أنا طالب دراسات عليا وأتمنى إضافة مصطلحات قانونية", ["طالب دراسات عليا"]),
    ("kinship_predicate", "أمي تقول هذه الكلمة دائما بمعنى آخر وأقترح إضافته", ["تقول هذه الكلمة"]),
    ("latin_vocabulary", "أرجو إضافة كلمة toilet إلى المعجم العربي", ["toilet"]),
    ("historical_source", "قال ابن منظور في لسان العرب إن الكلمة بهذا المعنى", ["ابن منظور", "لسان العرب"]),
    ("single_link_nasab_citation", "ورد في كلام عبدالله بن عباس رضي الله عنهما شاهد على هذا المعنى", ["عبدالله بن عباس"]),
    ("diacritised_example", "كدية: الكَلِمَةُ الجَدِيدَةُ تُضَافُ إِلَى المُعْجَمِ بَعْدَ المُرَاجَعَةِ", ["الكَلِمَةُ الجَدِيدَةُ"]),
]

TITLE_FIXTURES = [
    ("title_with_phone", "كلمة 0501234567", ["0501234567"]),
    ("title_with_email", "كلمة sara.test@example.com", ["sara.test@example.com"]),
]


def capture(comment: str, title: str | None = None) -> str:
    client = FakeGeminiClient([SAFE])
    req = GenerationRequest(comment_id="synthetic", comment_text=_reviewer_facing_text(comment) or "", primary_intent="SUGGESTION", topic_subtype="DICTIONARY_SUGGESTION",
                            sentiment="NEUTRAL", confidence=0.5, identifying_metadata=IdentifyingMetadata(title, "ENTRY-SYN-1") if title else None)
    run_agent_pipeline(req, client=client,
                       official_knowledge_retriever=FakeOfficialKnowledgeRetriever(OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)),
                       approved_response_retriever=FakeApprovedResponseMemoryRetriever(ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)))
    assert client.call_count >= 1, "fixture did not reach the model seam"
    return "\n".join(client.prompts)


def main() -> int:
    results = []
    for cat, text, banned in FIXTURES:
        prompt = capture(text)
        leaked = [b for b in banned if b in prompt]
        results.append({"category": cat, "passed": not leaked, "leaked_fragments": len(leaked)})
    for cat, title, banned in TITLE_FIXTURES:
        prompt = capture("أقترح إضافة خاصية البحث بالجذر في المعجم", title=title)
        leaked = [b for b in banned if b in prompt]
        results.append({"category": cat, "passed": not leaked, "leaked_fragments": len(leaked)})
    for cat, text, keep in PRESERVE:
        prompt = capture(text)
        lost = [k for k in keep if k not in prompt]
        results.append({"category": "preserved:" + cat, "passed": not lost, "leaked_fragments": 0, "lost_fragments": len(lost)})
    # the prompt must never carry the internal entry_id
    results.append({"category": "internal_entry_id_not_rendered", "passed": "ENTRY-SYN-1" not in capture("أقترح إضافة خاصية البحث بالجذر", title="كلمة"), "leaked_fragments": 0})
    failed = [r["category"] for r in results if not r["passed"]]
    summary = {"verdict": "PASS" if not failed else "FAIL", "failed_categories": failed, "results": results,
               "redaction_implemented_in_path": ["phone numbers (>= 7 digits, any digit script)", "e-mail addresses"],
               "note": "synthetic fixtures only; nothing was sent anywhere"}
    (OUT / "privacy_preflight.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
