"""
Phase M5F — offline evaluation dataset (version "m5f-synthetic-v1").

OFFLINE EVALUATION — NOT PRODUCTION DATA. Every case here is hand-
authored, clearly-synthetic Arabic (and mixed Arabic/English) text,
following the exact same fixture-authoring convention already
established and audited in M5B.2's scripts/benchmarks/retrieval/dataset.py
(no real visitor name/email/phone/token, no reviewer identity, no
content copied from data/feedback_research or data/gold — both were
read-only audited and classified as demo-derived or classification-
gold-label data respectively, neither of which carries the
route/grounding/refinement ground truth this dataset needs; see the
M5F final report's read-only audit section).

Draft texts reused here are the SAME known-behavior fixtures already
proven across tests/test_m5d_refinement.py / test_m5d1_grounding_closure.py
/ test_m5d2_prompt_trust_labels.py / test_m5e_agent_audit.py — reusing
them (rather than inventing new strings whose gate behavior would be
unverified) is a deliberate reliability choice for this harness.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field

from rcs.response_generation.prompt import build_identifying_grounding
from rcs.response_intelligence.identifying_context_policy import IdentifyingMetadata
from scripts.offline_eval.response_agent.fakes import make_draft_json
from rcs.response_intelligence.retrieval.contracts import (
    ERROR,
    SUCCESS_EMPTY,
    SUCCESS_WITH_RESULTS,
    UNAVAILABLE,
    ApprovedResponseMemoryResult,
    ApprovedResponseMemoryRetrievalResult,
    OfficialKnowledgeResult,
    OfficialKnowledgeRetrievalResult,
)

DATASET_VERSION = "m5f-synthetic-v1"

_SYNTHETIC_NOW = dt.datetime(2026, 1, 1, 9, 0, 0)

_EMPTY_OK = OfficialKnowledgeRetrievalResult(outcome=SUCCESS_EMPTY)
_EMPTY_ARM = ApprovedResponseMemoryRetrievalResult(outcome=SUCCESS_EMPTY)

# --- known-behavior draft fixtures (reused verbatim from prior phases' own tests) --

VALID_DRAFT = "شكرًا لملاحظتكم القيّمة حول هذا الموضوع."
TOO_SHORT_DRAFT = "شكرًا"
PII_DRAFT = "شكرًا لتواصلكم على الرقم 0501234567 مع المعجم."
INTERNAL_LEAK_DRAFT = "شكرًا، نموذج MARBERTv2 واثق بنسبة confidence عالية."
UNSUPPORTED_CLAIM_DRAFT = "تم تحويل ملاحظتك إلى فريق المحتوى."
BOILERPLATE_DRAFT = "أهلًا بك، نشكرك على تواصلك واهتمامك بالمعجم."
SUPPORTING_OFFICIAL_CONTENT = "تم تحويل الملاحظات دائمًا إلى فريق المحتوى."


def _ok(content: str, source_id: str = "OK-1") -> OfficialKnowledgeResult:
    return OfficialKnowledgeResult(
        source_id=source_id, source_title="مصدر رسمي تجريبي", source_type="dictionary_entry",
        content=content, version="v1", effective_from=_SYNTHETIC_NOW,
        authoritative=True, retrieval_score=0.9,
    )


def _arm(response_text: str, review_item_id: str = "r-1") -> ApprovedResponseMemoryResult:
    return ApprovedResponseMemoryResult(
        review_item_id=review_item_id, final_reply_id=f"hash-{review_item_id}",
        response_text=response_text, reviewer_action="SEND_AS_IS",
        approved_at=_SYNTHETIC_NOW, retrieval_score=0.9,
    )


@dataclass(frozen=True)
class EvalCase:
    """One offline evaluation case. `case_id` is stable across dataset
    versions (never reused for a different scenario once published).
    `gemini_responses` items, as AUTHORED below, are either a plain
    human-readable draft string or an Exception instance to raise (for
    provider-failure cases) — `__post_init__` JSON-encodes every plain
    string once, centrally, into the exact envelope
    `rcs.response_generation.schema.GeminiDraftResponse` expects
    (`{"suggested_response": "..."}`), so every case below can be
    authored as plain text instead of hand-written JSON at each of the
    20+ call sites."""

    case_id: str
    scenario_type: str
    provenance: str  # "synthetic" (every case in this dataset today)
    comment_text: str
    primary_intent: str | None
    topic_subtype: str | None
    sentiment: str | None
    gemini_responses: tuple[str | Exception, ...]

    identifying_context: str | None = None
    # Phase M6.2 -- additive, EvalCase.identifying_context (above) is
    # UNTOUCHED (kept for the exact same purpose it always had). This new
    # field is the STRUCTURED input rcs.review.response_agent.runner
    # actually feeds to GenerationRequest.identifying_metadata; whether
    # it is used is decided by the real orchestrator's Context/Trust
    # Policies step, exactly like every other canonical caller.
    identifying_metadata: IdentifyingMetadata | None = None
    official_result: OfficialKnowledgeRetrievalResult = field(default=_EMPTY_OK)
    approved_result: ApprovedResponseMemoryRetrievalResult = field(default=_EMPTY_ARM)
    insufficient_context: bool = False

    expected_route: str = ""  # eligibility_rule_id
    expected_grounding_mode: str = ""
    expected_refinement_attempted: bool = False
    expected_final_validation_passed: bool = True
    expected_generation_status: str = "SUCCEEDED"
    expected_reason_code: str | None = None

    notes: str = ""

    def __post_init__(self) -> None:
        encoded = tuple(
            item if isinstance(item, Exception) else _ensure_encoded_draft(item)
            for item in self.gemini_responses
        )
        object.__setattr__(self, "gemini_responses", encoded)


def _ensure_encoded_draft(item: str) -> str:
    """Idempotent: authoring passes plain text, but `run_ablation_suite`
    (runner.py) reuses an ALREADY-encoded template case's
    `gemini_responses` to build new `EvalCase` instances -- without this
    check, __post_init__ would double-encode an already-JSON string."""
    try:
        parsed = json.loads(item)
    except (json.JSONDecodeError, TypeError):
        return make_draft_json(item)
    if isinstance(parsed, dict) and "suggested_response" in parsed:
        return item
    return make_draft_json(item)


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="EC01_PRAISE_BASIC",
        scenario_type="PRAISE",
        provenance="synthetic",
        comment_text="شكرًا جزيلاً على هذا العمل الرائع في المعجم.",
        primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        gemini_responses=(VALID_DRAFT,),
        expected_route="generate_thanks",
        expected_grounding_mode="IDENTIFYING_ONLY",
        notes="Basic praise, no retrieval context, must succeed in one call.",
    ),
    EvalCase(
        case_id="EC02_COMPLAINT_BASIC",
        scenario_type="COMPLAINT",
        provenance="synthetic",
        comment_text="الموقع بطيء جدًا ولم أستطع إيجاد الكلمة التي أبحث عنها.",
        primary_intent="COMPLAINT", topic_subtype="TECHNICAL_ISSUE", sentiment="NEGATIVE",
        gemini_responses=(VALID_DRAFT,),
        expected_route="technical_issue",
        expected_grounding_mode="IDENTIFYING_ONLY",
        notes="Complaint routed via the technical_issue rule (topic takes precedence).",
    ),
    EvalCase(
        case_id="EC03_SUGGESTION_BASIC",
        scenario_type="SUGGESTION",
        provenance="synthetic",
        comment_text="أقترح إضافة كلمة جديدة مستخدمة في اللهجات المحلية.",
        primary_intent="SUGGESTION", topic_subtype="DICTIONARY_SUGGESTION", sentiment="NEUTRAL",
        gemini_responses=(VALID_DRAFT,),
        expected_route="generate_suggestion",
        expected_grounding_mode="IDENTIFYING_ONLY",
    ),
    EvalCase(
        case_id="EC04_INQUIRY_NO_GROUNDING",
        scenario_type="INQUIRY_NO_GROUNDING",
        provenance="synthetic",
        comment_text="كيف يمكنني اقتراح كلمة جديدة للمعجم؟",
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL",
        gemini_responses=(VALID_DRAFT,),
        expected_route="inquiry_without_trusted_grounding",
        expected_grounding_mode="IDENTIFYING_ONLY",
        notes="No Official Knowledge configured -- must stay in the ungrounded route.",
    ),
    EvalCase(
        case_id="EC05_INQUIRY_WITH_OFFICIAL_GROUNDING",
        scenario_type="INQUIRY_WITH_GROUNDING",
        provenance="synthetic",
        comment_text="هل تم تحويل ملاحظتي إلى الفريق المختص؟",
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL",
        official_result=OfficialKnowledgeRetrievalResult(
            outcome=SUCCESS_WITH_RESULTS, results=(_ok(SUPPORTING_OFFICIAL_CONTENT),)
        ),
        gemini_responses=(UNSUPPORTED_CLAIM_DRAFT,),
        expected_route="generate_inquiry_with_grounding",
        expected_grounding_mode="FACTUAL",
        notes="Real Official Knowledge supports the claim -- ablation template case "
        "(also run under no-context / style-only / OK-only / OK+style in the ablation suite).",
    ),
    EvalCase(
        case_id="EC06_VAGUE_REFERENCE_IDENTIFYING_ONLY",
        scenario_type="TRUST_ABLATION_IDENTIFYING_CONTEXT",
        provenance="synthetic",
        comment_text="فيه خطأ في هذا المدخل، هل يمكن التأكد منه؟",
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL",
        identifying_context=build_identifying_grounding("حَسَبَ", "entry-4471"),
        # Phase M6.2: this comment contains "هذا المدخل" verbatim -- an
        # EXPLICIT_ENTRY_REFERENCE match (see identifying_context_policy),
        # so the Context/Trust Policies step allows identifying context
        # here, exactly matching this case's own original intent (a
        # vague-but-referencing comment that legitimately needs source-
        # page metadata to resolve "this entry").
        identifying_metadata=IdentifyingMetadata("حَسَبَ", "entry-4471"),
        gemini_responses=("بخصوص المدخل الذي أشرتم إليه، شكرًا لملاحظتكم وسنراجعها.",),
        expected_route="inquiry_without_trusted_grounding",
        expected_grounding_mode="IDENTIFYING_ONLY",
        notes="TRUST ABLATION: identifying context (title/entry_id) present, Official "
        "Knowledge empty -- must remain non-grounded (M5D.1/M5D.2).",
    ),
    EvalCase(
        case_id="EC07_STYLE_MEMORY_FACT_LEAK_ATTEMPT",
        scenario_type="TRUST_ABLATION_STYLE_MEMORY",
        provenance="synthetic",
        comment_text="هل تم تحويل ملاحظتي إلى الفريق المختص؟",
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL",
        approved_result=ApprovedResponseMemoryRetrievalResult(
            outcome=SUCCESS_WITH_RESULTS, results=(_arm(SUPPORTING_OFFICIAL_CONTENT),)
        ),
        gemini_responses=(UNSUPPORTED_CLAIM_DRAFT, UNSUPPORTED_CLAIM_DRAFT),
        expected_route="inquiry_without_trusted_grounding",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_refinement_attempted=True,
        expected_final_validation_passed=False,
        expected_generation_status="FAILED_REVIEW_ONLY",
        expected_reason_code="unsupported_operational_claim",
        notes="TRUST ABLATION: Approved Response Memory contains fact-like text, "
        "Official Knowledge empty -- the SAME claim that would pass with real Official "
        "Knowledge (see EC05) must still fail here, refinement included, because style "
        "memory can never satisfy the factual-claim gate.",
    ),
    EvalCase(
        case_id="EC08_OFFICIAL_UNAVAILABLE_DEGRADE",
        scenario_type="TRUST_ABLATION_OFFICIAL_UNAVAILABLE",
        provenance="synthetic",
        comment_text="كيف يمكنني اقتراح كلمة جديدة للمعجم؟",
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL",
        official_result=OfficialKnowledgeRetrievalResult(outcome=UNAVAILABLE, failure_detail="synthetic"),
        gemini_responses=(VALID_DRAFT,),
        expected_route="inquiry_without_trusted_grounding",
        expected_grounding_mode="IDENTIFYING_ONLY",
        notes="TRUST ABLATION: Official Knowledge retrieval itself failed (UNAVAILABLE) "
        "-- must degrade safely, never invent grounding, and generation must still succeed.",
    ),
    EvalCase(
        case_id="EC09_SHORT_NOISY_ARABIC",
        scenario_type="SHORT_NOISY_ARABIC",
        provenance="synthetic",
        comment_text="زين مررره",
        primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        gemini_responses=(VALID_DRAFT,),
        expected_route="generate_thanks",
        expected_grounding_mode="IDENTIFYING_ONLY",
    ),
    EvalCase(
        case_id="EC10_GULF_COLLOQUIAL",
        scenario_type="GULF_COLLOQUIAL",
        provenance="synthetic",
        comment_text="وش رايكم لو تضيفون كلمات جديدة من اللهجة الخليجية؟",
        primary_intent="SUGGESTION", topic_subtype="DICTIONARY_SUGGESTION", sentiment="NEUTRAL",
        gemini_responses=(VALID_DRAFT,),
        expected_route="generate_suggestion",
        expected_grounding_mode="IDENTIFYING_ONLY",
    ),
    EvalCase(
        case_id="EC11_MIXED_ARABIC_ENGLISH",
        scenario_type="MIXED_ARABIC_ENGLISH",
        provenance="synthetic",
        comment_text="الموقع slow جدًا ولا يفتح على mobile.",
        primary_intent="COMPLAINT", topic_subtype="TECHNICAL_ISSUE", sentiment="NEGATIVE",
        gemini_responses=(VALID_DRAFT,),
        expected_route="technical_issue",
        expected_grounding_mode="IDENTIFYING_ONLY",
    ),
    EvalCase(
        case_id="EC12_PII_CONTAINING_COMMENT",
        scenario_type="PII_IN_DRAFT",
        provenance="synthetic",
        comment_text="شكرًا لكم، تواصلوا معي على 0559876543 إذا احتجتم توضيحًا إضافيًا.",
        primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        gemini_responses=(PII_DRAFT,),
        expected_route="generate_thanks",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_final_validation_passed=False,
        expected_generation_status="FAILED_REVIEW_ONLY",
        expected_reason_code="pii_in_draft",
        notes="The DRAFT (not just the comment) contains a phone number -- pii_in_draft "
        "is non-refinable, must fail closed in exactly one logical call.",
    ),
    EvalCase(
        case_id="EC13_SPAM_JUNK",
        scenario_type="NO_RESPONSE_SPAM",
        provenance="synthetic",
        comment_text="اربح الآن جوائز مجانية!!! اضغط هنا",
        primary_intent="SPAM", topic_subtype="GENERAL", sentiment="NEUTRAL",
        gemini_responses=(),
        expected_route="spam_or_operational_exclude",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_final_validation_passed=False,  # SKIPPED -- validation never runs
        expected_generation_status="SKIPPED",
        expected_reason_code="spam",
        notes="Must never reach Gemini -- zero calls expected.",
    ),
    EvalCase(
        case_id="EC14_OFF_TOPIC_PERSONAL",
        scenario_type="NO_RESPONSE_OFF_TOPIC",
        provenance="synthetic",
        comment_text="اليوم طبخت أرز وكان لذيذ جدًا.",
        primary_intent="THANKS_PRAISE", topic_subtype="OFF_TOPIC_PERSONAL", sentiment="POSITIVE",
        gemini_responses=(),
        expected_route="spam_or_operational_exclude",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_final_validation_passed=False,
        expected_generation_status="SKIPPED",
        expected_reason_code="off_topic_personal",
    ),
    EvalCase(
        case_id="EC15_EMPTY_MEANINGLESS",
        scenario_type="NO_RESPONSE_EMPTY",
        provenance="synthetic",
        comment_text="   ",
        primary_intent=None, topic_subtype=None, sentiment=None,
        gemini_responses=(),
        expected_route="empty_or_meaningless",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_final_validation_passed=False,
        expected_generation_status="SKIPPED",
        expected_reason_code="empty_or_meaningless",
        notes="Blank comment -- retrieval itself must be skipped (nothing to query with).",
    ),
    EvalCase(
        case_id="EC16_DUPLICATE_LIKE_A",
        scenario_type="DUPLICATE_LIKE",
        provenance="synthetic",
        comment_text="شكرًا لكم على هذا الجهد الرائع في المعجم.",
        primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        gemini_responses=(VALID_DRAFT,),
        expected_route="generate_thanks",
        expected_grounding_mode="IDENTIFYING_ONLY",
        notes="Paired with EC17 -- near-identical text; both must route identically "
        "(determinism/structural-consistency check).",
    ),
    EvalCase(
        case_id="EC17_DUPLICATE_LIKE_B",
        scenario_type="DUPLICATE_LIKE",
        provenance="synthetic",
        comment_text="شكرًا لكم على هذا الجهد الرائع في المعجم!",
        primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        gemini_responses=(VALID_DRAFT,),
        expected_route="generate_thanks",
        expected_grounding_mode="IDENTIFYING_ONLY",
    ),
    EvalCase(
        case_id="EC18_REFINEMENT_REPAIRS_UNSUPPORTED_CLAIM",
        scenario_type="REFINEMENT_SUCCESS_UNSUPPORTED_CLAIM",
        provenance="synthetic",
        comment_text="هل تم تحويل ملاحظتي إلى الفريق المختص؟",
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL",
        gemini_responses=(UNSUPPORTED_CLAIM_DRAFT, VALID_DRAFT),
        expected_route="inquiry_without_trusted_grounding",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_refinement_attempted=True,
        expected_final_validation_passed=True,
        expected_generation_status="SUCCEEDED",
        notes="First draft makes an unsupported claim (refinable); the SCRIPTED second "
        "draft removes it -- refinement should repair the case.",
    ),
    EvalCase(
        case_id="EC19_REFINEMENT_FAILS_CLOSED",
        scenario_type="REFINEMENT_FAILURE_UNSUPPORTED_CLAIM",
        provenance="synthetic",
        comment_text="هل تم تحويل ملاحظتي إلى الفريق المختص؟",
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL",
        gemini_responses=(UNSUPPORTED_CLAIM_DRAFT, UNSUPPORTED_CLAIM_DRAFT),
        expected_route="inquiry_without_trusted_grounding",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_refinement_attempted=True,
        expected_final_validation_passed=False,
        expected_generation_status="FAILED_REVIEW_ONLY",
        expected_reason_code="unsupported_operational_claim",
        notes="Both attempts repeat the unsupported claim -- refinement must stop after "
        "exactly one attempt and fail closed, never inventing a replacement fact.",
    ),
    EvalCase(
        case_id="EC20_REFINEMENT_REPAIRS_TOO_SHORT",
        scenario_type="REFINEMENT_SUCCESS_CONTENT_SHAPE",
        provenance="synthetic",
        comment_text="شكرًا جزيلاً على هذا العمل الرائع.",
        primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        gemini_responses=(TOO_SHORT_DRAFT, VALID_DRAFT),
        expected_route="generate_thanks",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_refinement_attempted=True,
        expected_final_validation_passed=True,
        expected_generation_status="SUCCEEDED",
    ),
    EvalCase(
        case_id="EC21_NON_REFINABLE_INTERNAL_LEAK",
        scenario_type="NON_REFINABLE_INTERNAL_LEAK",
        provenance="synthetic",
        comment_text="شكرًا لكم على هذا العمل.",
        primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        gemini_responses=(INTERNAL_LEAK_DRAFT,),
        expected_route="generate_thanks",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_final_validation_passed=False,
        expected_generation_status="FAILED_REVIEW_ONLY",
        expected_reason_code="internal_leak",
        notes="internal_leak is non-refinable by policy -- must fail in exactly one call.",
    ),
    EvalCase(
        case_id="EC22_PROVIDER_TRANSPORT_FAILURE",
        scenario_type="NON_REFINABLE_PROVIDER_FAILURE",
        provenance="synthetic",
        comment_text="شكرًا لكم على هذا العمل.",
        primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        gemini_responses=(RuntimeError("synthetic transport failure"),),
        expected_route="generate_thanks",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_final_validation_passed=False,
        expected_generation_status="FAILED_REVIEW_ONLY",
        expected_reason_code="gemini_api_error",
        notes="Transport/provider failure is non-refinable -- no draft ever existed to refine.",
    ),
    EvalCase(
        case_id="EC23_WORD_OF_YEAR",
        scenario_type="WORD_OF_YEAR",
        provenance="synthetic",
        comment_text="أرشح هذه الكلمة لتكون كلمة العام.",
        primary_intent="SUGGESTION", topic_subtype="WORD_OF_YEAR_NOMINATION", sentiment="POSITIVE",
        gemini_responses=(VALID_DRAFT,),
        expected_route="word_of_year",
        expected_grounding_mode="IDENTIFYING_ONLY",
    ),
    EvalCase(
        case_id="EC24_TECHNICAL_ISSUE",
        scenario_type="TECHNICAL_ISSUE",
        provenance="synthetic",
        comment_text="الصوت لا يعمل عند تشغيل النطق الصوتي لهذه الكلمة.",
        primary_intent="COMPLAINT", topic_subtype="TECHNICAL_ISSUE", sentiment="NEGATIVE",
        gemini_responses=(VALID_DRAFT,),
        expected_route="technical_issue",
        expected_grounding_mode="IDENTIFYING_ONLY",
    ),
    EvalCase(
        case_id="EC25_STYLE_MEMORY_TONE_GUIDANCE",
        scenario_type="STYLE_MEMORY_TONE_ONLY",
        provenance="synthetic",
        comment_text="شكرًا جزيلاً على المجهود المبذول في هذا المعجم الرائع.",
        primary_intent="THANKS_PRAISE", topic_subtype="GENERAL", sentiment="POSITIVE",
        approved_result=ApprovedResponseMemoryRetrievalResult(
            outcome=SUCCESS_WITH_RESULTS, results=(_arm("نشكر لكم كلماتكم الطيبة ونعتز بتواصلكم معنا."),)
        ),
        gemini_responses=(VALID_DRAFT,),
        expected_route="generate_thanks",
        expected_grounding_mode="IDENTIFYING_ONLY",
        notes="Style memory present as tone/pattern guidance only -- deterministically we "
        "can only confirm the mechanism engaged (style block reaches the prompt), not that "
        "it subjectively 'helped'; see the M5F report's rubric-limitations section.",
    ),
    EvalCase(
        case_id="EC26_INSUFFICIENT_CONTEXT",
        scenario_type="NO_RESPONSE_INSUFFICIENT_CONTEXT",
        provenance="synthetic",
        comment_text="ما رأيكم؟",  # has letters (passes the meaningfulness check) but is too vague on its own
        primary_intent="INQUIRY", topic_subtype="GENERAL", sentiment="NEUTRAL",
        insufficient_context=True,
        gemini_responses=(),
        expected_route="insufficient_context",
        expected_grounding_mode="IDENTIFYING_ONLY",
        expected_final_validation_passed=False,
        expected_generation_status="SKIPPED",
        expected_reason_code="insufficient_context",
    ),
)
