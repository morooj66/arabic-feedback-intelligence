"""Guardrails on Gemini drafts. Fail closed — never invent a replacement draft.

Phase M5A.3: the same checks that always lived here are now organized as
five explicit, named gates run in a fixed order by run_validation_gates,
instead of one long function body. This is a structural refactor only —
every regex/constant/threshold below is unchanged, evaluated in the same
order, and validate_draft (the pre-existing public function every caller
already uses) still returns the exact same (bool, reason_code | None)
shape it always has. No new check, no new pattern, no new tone/policy/
grounding rule was added.

Schema validation (GeminiDraftResponse.model_validate_json) happens
BEFORE this module is ever reached, in
rcs.response_generation.service.generate_suggested_response — that is a
response-PARSING boundary (is this valid JSON matching our schema?), not
a content-validation boundary (is this parsed draft safe/sane to show a
reviewer?). The two stay separate on purpose; nothing here re-parses or
re-validates the JSON shape.

Constant ownership: every pattern/threshold stays exactly where it was
in this file — including `_LEAK_PATTERNS`, which scripts/run_response_
quality_eval_30.py imports directly today (`from rcs.response_generation
.validate import _LEAK_PATTERNS`). Moving it elsewhere would either break
that import or create a second copy; neither is acceptable, so the gate
functions below are thin wrappers around logic that lives here, not a
new module that reaches back in for the same regexes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from config import settings
from rcs.classification.redact import redact_phone_numbers
from rcs.response_generation import claim_gates

_ARABIC_RE = re.compile(r"[؀-ۿ]")
_EMPTY_GROUNDING = {"", "(none — do not invent facts)"}

_LEAK_PATTERNS = (
    re.compile(r"\bMARBERT", re.I),
    re.compile(r"CAMeL", re.I),
    re.compile(r"TF-?IDF", re.I),
    re.compile(r"LogisticRegression", re.I),
    re.compile(r"Macro-?F1", re.I),
    re.compile(r"human_test", re.I),
    re.compile(r"GENERATE_DRAFT"),
    re.compile(r"REVIEW_ONLY"),
    re.compile(r"NO_RESPONSE_RECOMMENDED"),
    re.compile(r"gemini-\d", re.I),
    re.compile(r"prompt_version", re.I),
    re.compile(r"classification_source", re.I),
    re.compile(r"GEMINI_API_KEY"),
    re.compile(r"SYSTEM_INSTRUCTIONS"),
    re.compile(r"UNTRUSTED VISITOR COMMENT"),
    re.compile(r"\bconfidence\b", re.I),
)

# Operational claims that require an explicit matching fact in trusted grounding.
_UNSUPPORTED_ACTION_PATTERNS = (
    re.compile(r"تم تدوين"),
    re.compile(r"تم تحويل"),
    re.compile(r"تم نقل"),
    re.compile(r"تم رفع"),
    re.compile(r"تم استلام"),
    re.compile(r"تم أخذ.{0,40}بعين الاعتبار"),
    re.compile(r"سيتم التواصل"),
    re.compile(r"سيتم إصلاح"),
    re.compile(r"سيتم تنفيذ"),
    re.compile(r"سيتم النظر"),
    re.compile(r"سيراجعها المختص"),
    re.compile(r"للفريق المختص"),
    re.compile(r"على المختصين"),
    re.compile(r"سيطلع عليها الفريق"),
    re.compile(r"وصلنا إلى الفريق"),
    re.compile(r"خلال\s*\d+"),
    re.compile(r"خلال\s*(ساعة|يوم|أيام|أسبوع)"),
    re.compile(r"القنوات?\s*الرسمية"),
    re.compile(r"official channels?", re.I),
    re.compile(r"يمكنك التواصل"),
    re.compile(r"التواصل عبر"),
    re.compile(r"الإجراءات المعتمدة"),
    re.compile(r"أطر الشراكة"),
    # Same category as above (an org-action claim needing an explicit
    # matching fact in grounding), extended with execution/commitment
    # phrasings for action verbs the original list -- evidence-anchored
    # to earlier complaint/inquiry review findings -- never covered
    # (send content, contact, cooperate, add/correct/approve an entry).
    re.compile(r"تم إرسال"),
    re.compile(r"سيتم إرسال"),
    re.compile(r"سنرسل"),
    re.compile(r"سوف نرسل"),
    re.compile(r"تم التواصل"),
    re.compile(r"سنتواصل"),
    re.compile(r"سوف نتواصل"),
    re.compile(r"تم قبول"),
    re.compile(r"سيتم قبول"),
    re.compile(r"سنقبل"),
    re.compile(r"تم اعتماد"),
    re.compile(r"سيتم اعتماد"),
    re.compile(r"سنعتمد"),
    re.compile(r"تم إضافة"),
    re.compile(r"سيتم إضافة"),
    re.compile(r"سنضيف"),
    re.compile(r"تم تعديل"),
    re.compile(r"سيتم تعديل"),
    re.compile(r"سنعدل"),
    re.compile(r"تم التعاون"),
    re.compile(r"سيتم التعاون"),
    re.compile(r"سنتعاون"),
)

_GENERIC_EXACT_DRAFTS = frozenset(
    {
        "أهلًا بك، نشكرك على تواصلك واهتمامك بالمعجم.",
        "أهلاً بك، نشكرك على تواصلك واهتمامك بالمعجم.",
        "أهلًا بك نشكرك على تواصلك واهتمامك بالمعجم.",
        "نسعد بتواصلك معنا ونقدر اهتمامك بالمعجم.",
        "ونسعد دائمًا بتواصلك.",
    }
)


def _normalize_exact(text: str) -> str:
    return " ".join(text.strip().split())


def _grounding_is_empty(trusted_grounding_text: str | None) -> bool:
    return _normalize_exact(trusted_grounding_text or "") in _EMPTY_GROUNDING


# --- typed gate result (Phase M5A.3) ----------------------------------------


@dataclass(frozen=True)
class GateResult:
    """One gate's outcome. `reason_code` is the exact same string
    validate_draft has always returned for that failure -- nothing here
    invents a new vocabulary, it just names which check produced the
    existing code."""

    gate_name: str
    passed: bool
    reason_code: str | None = None


@dataclass(frozen=True)
class ValidationSummary:
    """`evaluated_gates` holds only the gates actually run -- the runner
    stops at the first failure (see run_validation_gates), so this is
    never all five unless every gate passed. No remediation hints, no
    provenance, no reasoning trace: those belong to later phases, if
    ever."""

    passed: bool
    first_failure: GateResult | None
    evaluated_gates: tuple[GateResult, ...]


# --- individual gates, in the exact order validate_draft always ran them ---


def _content_shape_gate(text: str | None) -> GateResult:
    """Verbatim: empty/blank check, then length bounds, then minimum
    Arabic-letter count -- the same four checks and the same order
    validate_draft always ran first, now grouped under one gate name
    because they are all "is this draft shaped like a usable reply"
    checks, per the M5A.3 grouping."""
    if text is None or not str(text).strip():
        return GateResult("content_shape", False, "empty_output")
    draft = text.strip()
    if len(draft) < settings.RESPONSE_DRAFT_MIN_CHARS:
        return GateResult("content_shape", False, "too_short")
    if len(draft) > settings.RESPONSE_DRAFT_MAX_CHARS:
        return GateResult("content_shape", False, "excessive_length")
    arabic_count = len(_ARABIC_RE.findall(draft))
    if arabic_count < settings.RESPONSE_DRAFT_MIN_ARABIC_LETTERS:
        return GateResult("content_shape", False, "missing_arabic")
    return GateResult("content_shape", True)


def _internal_leak_gate(draft: str) -> GateResult:
    """Verbatim: the same _LEAK_PATTERNS loop, same order, same
    single reason_code regardless of which pattern matched -- exactly
    as before (the original never distinguished which leak pattern
    fired, only that one did)."""
    for pattern in _LEAK_PATTERNS:
        if pattern.search(draft):
            return GateResult("internal_leak", False, "internal_leak")
    return GateResult("internal_leak", True)


def _pii_gate(draft: str) -> GateResult:
    """Verbatim: the exact current PII check (phone numbers only, via
    the same redact_phone_numbers already used for prompt-input
    redaction). Not extended to email or any other PII type in this
    phase -- that would be new coverage, not a structural refactor."""
    if redact_phone_numbers(draft) != draft:
        return GateResult("pii", False, "pii_in_draft")
    return GateResult("pii", True)


def _boilerplate_gate(draft: str) -> GateResult:
    """Reject known canned drafts and rg-v14.2's evidence-backed stacked
    courtesy shape under the existing recoverable boilerplate reason code."""
    if _normalize_exact(draft) in _GENERIC_EXACT_DRAFTS:
        return GateResult("boilerplate", False, "generic_boilerplate")
    # The focused rg-v14.1 live regression produced two replies with stacked
    # ceremonial reactions despite the prompt's at-most-one rule. Keep this
    # deliberately small: count only first-person institutional courtesy
    # markers observed in the evaluation family.
    courtesy_markers = (
        "نقدّر",
        "نقدر",
        "نشكر",
        "نعتز",
        "يسرنا",
        "نرحب",
        "يؤسفنا",
        "نأسف",
        "شكرًا",
        "شكرا",
    )
    matches = sum(draft.count(marker) for marker in courtesy_markers)
    if matches > 1:
        return GateResult("boilerplate", False, "generic_boilerplate")
    return GateResult("boilerplate", True)


# Phase rg-v9 (2026-09): the literal openings observed in the first real
# human review of 10 live visitor comments, where the draft opened by
# characterizing the visitor's comment itself (a detached, third-person
# "evaluation" register) instead of replying to the visitor. Deliberately
# NOT a general blacklist: only checked against the draft's OPENING, so
# the same words used naturally elsewhere in a reply addressed to the
# visitor (e.g. "نقدر ملاحظتكم بشأن...") never match.
_EVALUATOR_VOICE_OPENINGS = (
    "ملاحظة دقيقة",
    "ملاحظة بالغة الدقة",
    "رأي اصطلاحي",
    "مساهمة معجمية قيمة",
    "تحليل عميق",
    "اقتراح مهم",
)


def _evaluator_voice_gate(draft: str) -> GateResult:
    """Narrow, deterministic, evidence-anchored: fails only when the
    draft's OPENING is one of the exact detached-evaluation phrases
    above. See `_EVALUATOR_VOICE_OPENINGS`'s comment for why this is
    scope-limited to the opening rather than a whole-text blacklist."""
    normalized = _normalize_exact(draft)
    for opening in _EVALUATOR_VOICE_OPENINGS:
        if normalized.startswith(opening):
            return GateResult("evaluator_voice", False, "evaluator_voice_opening")
    return GateResult("evaluator_voice", True)


def _unsupported_claim_gate(draft: str, trusted_grounding_text: str | None) -> GateResult:
    """Verbatim: the same _UNSUPPORTED_ACTION_PATTERNS loop and the
    same grounding-must-state-the-same-fact check. This is today's only
    grounding-related check -- named clearly here as such, but its
    behavior (including what happens when trusted_grounding_text is
    None, which _grounding_is_empty treats as empty) is unchanged."""
    grounding = trusted_grounding_text or ""
    for pattern in _UNSUPPORTED_ACTION_PATTERNS:
        if not pattern.search(draft):
            continue
        if _grounding_is_empty(grounding) or not pattern.search(grounding):
            return GateResult("unsupported_claim", False, "unsupported_operational_claim")
    return GateResult("unsupported_claim", True)


def _strip_diacritics(text: str) -> str:
    """Unicode category Mn (Arabic tashkeel) only -- same normalization
    concept as identifying_context_policy._strip_arabic_diacritics, but
    duplicated here (not imported) because rcs.response_intelligence
    depends on rcs.response_generation, not the other way around; a
    reverse import would create a layering cycle."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


# --- Visitor intent/action attribution guard --------------------------
#
# Audit evidence (scripts/offline_eval/identifying_context_mechanism/
# results/identifying_context_mechanism_cases.jsonl, eval_id
# "anshita_la_safiya-NONE"): comment "أنشطة لا صفية" (a bare phrase, no
# stated request) with NO identifying context at all still produced
# "نشكر إسهامكم باقتراح تركيب «أنشطة لا صفية»." -- a draft asserting the
# visitor proposed adding the phrase, though the comment never said so.
# Root cause traced to prompt.py's classification-signal framing and
# situation hints (fixed there); this gate is a narrow, general,
# closed-vocabulary BACKSTOP, not a rule about any one comment: it fails
# a draft only when it attributes a specific action (a suggestion, a
# request, or a correction) to the visitor via one of a small set of
# general "your-<noun>" possessive phrases or 2nd-person-plural past-
# tense verbs, AND the visitor's own comment text contains none of the
# matching family's lexical evidence. A generic, non-specific
# acknowledgement (e.g. "نقدر مشاركتكم") is never flagged -- only claims
# of a concrete action the visitor is said to have taken.
_SUGGESTION_ATTRIBUTION = ("اقتراحكم", "اقتراحك", "مقترحكم", "مقترحك", "اقترحتم")
_ACTION_ATTRIBUTION_TO_EVIDENCE: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        _SUGGESTION_ATTRIBUTION,   # evidence below is kept for documentation / backward compatibility; the gate uses comment_expresses_suggestion()
        (
            "اقترح",
            "أقترح",
            "نقترح",
            "يقترح",
            "تقترح",
            "اقتراح",
            "مقترح",
            "أتمنى إضافة",
            "من المفيد",
            "توفير خيار",
        ),
    ),
    (
        ("طلبكم", "طلبك", "طلبتم"),
        ("أرجو", "الرجاء", "يرجى", "نرجو", "أطلب", "نطلب", "طلب"),
    ),
    (
        ("تصحيحكم", "تصحيحك", "صححتم"),
        ("خطأ", "غلط", "تصحيح", "الصحيح"),
    ),
    # rg-v14.2 live evidence: a bare word was answered with "نشكر اهتمامكم
    # بالإثراء المعجمي". This narrow family checks only that invented motive;
    # it does not reject generic acknowledgment of a substantive contribution.
    (
        (
            "اهتمامكم بالإثراء المعجمي",
            "اهتمامك بالإثراء المعجمي",
            "حرصكم على الإثراء المعجمي",
            "حرصك على الإثراء المعجمي",
        ),
        ("إثراء", "اهتمام", "أهتم", "مهتم", "مساهمة", "إسهام"),
    ),
)


# --- Evidence that the visitor actually SUGGESTED something (family 1 of the attribution table) -------------------------------------------------------
#
# Live rg-v14.2 pilot evidence: "نشكركم على مقترحكم المتعلق بالأمثلة الموثقة" was rejected for the comment "أتمنى لو كان المثال المعطى مثالا حقيقيا من مصدر مكتوب ..."
# and "جزاكم الله خيرًا ... وعلى اقتراحكم بشأن إضافة الأمثلة التطبيقية" for "... تحية طيبة، وددت أن أشير إلى خطأ إملائي في أحد الأمثلة الواردة في المادة": both comments DO suggest something, but only through a
# wish ("أتمنى لو") or a polite request ("ممكن ان يضاف"), which the old substring list ("اقترح", "أتمنى إضافة", ...) did not recognise.
# The evidence is now recognised by expression IN CONTEXT, never by an isolated keyword:
#   explicit      اقترح / أقترح / نقترح / اقتراح / مقترح ...
#   wish          أتمنى (لو | أن | إضافة | توفير | تحسين ...), ياليت / ليت      - NOT a courtesy wish ("أتمنى لكم التوفيق", "أتمنى أن تكونوا بخير")
#   polite ask    ممكن / يمكن / من الممكن (أن) + an add / change verb (يضاف، تضيفون، إضافة، تعديل ...)  - NOT "كيف يمكنني ..." and NOT "هل يمكن البحث ...؟"
#   request       أرجو / نرجو / الرجاء / يرجى + (إضافة، تضمين، توفير، تعديل ...)
#   should        يجب / ينبغي / يفضل / من الأفضل / من المفيد + an add / change expression
#   imperative    أضيفوا / زيدوا / حسّنوا / عدّلوا / صحّحوا / وفّروا
# A match preceded (within two words) by a negator (لا، لم، لن، ما، ليس، مو، غير) is NOT evidence: "لا أقترح ..." / "غير ممكن إضافة ..." state a stance, not a suggestion.
# A comment with none of these expressions (a bare word, a question, a complaint) still cannot be thanked for a "suggestion": the protection against invented requests is unchanged.
_ADD_OR_CHANGE = r"(?:اضافه|إضافه|اضافة|إضافة|يضاف|تضاف|يضيف|تضيف\w*|تضمين|يتضمن|تتضمن|توفير|توفر|تعديل|تحسين|تصحيح|حذف|زياده|تغيير|تطوير|اتاحه|إتاحه|ادراج|إدراج|وضع|تقديم|عمل|عرض|ابراز|إبراز|اظهار|إظهار|توضيح|شرح|ذكر|نشر|ترتيب|تنظيم|دمج|فصل)"
_SUGGESTION_EXPRESSIONS = tuple(
    claim_gates._rx(p)
    for p in (
        r"(?:اقترح|أقترح|نقترح|يقترح|تقترح)\w*",
        r"اقتراح\w*",
        r"مقترح\w*",
        r"(?:اتمنى|أتمنى|نتمنى|إتمنى)\s+(?:لو|ان|أن|ان\s+يكون|\S*" + _ADD_OR_CHANGE + r"|وجود|ان\s+يتم)(?!\s+(?:\S+\s+){0,2}?(?:بخير|التوفيق|يوفقكم|وفقكم|الصحه|السعاده|النجاح|تكونوا|تكون))",
        r"(?<!\S)(?:ياليت|يا\s+ليت|ليت)(?!\S)",
        r"(?<!\S)(?:ممكن|يمكن|من\s+الممكن|هل\s+يمكن|هل\s+ممكن)\s+(?:ان\s+|أن\s+)?(?:\S+\s+){0,1}?" + _ADD_OR_CHANGE,
        r"(?<!\S)(?:ارجو|أرجو|نرجو|الرجاء|يرجى|من\s+فضلكم)\s+(?:\S+\s+){0,2}?" + _ADD_OR_CHANGE,
        r"(?<!\S)(?:يجب|ينبغي|لابد|لا\s+بد|يفضل|من\s+الافضل|من\s+الأفضل|الافضل)\s+(?:ان\s+|أن\s+)?(?:\S+\s+){0,2}?" + _ADD_OR_CHANGE,
        r"من\s+المفيد",                                   # "it would be useful ..." is itself the recommendation (kept from the previous evidence list)
        r"(?<!\S)(?:اضيفوا|أضيفوا|ضيفوا|زيدوا|حسنوا|عدلوا|صححوا|وفروا|اضف|أضف)(?!\S)",
        r"توفير\s+خيار",
    )
)
_NEGATED_BEFORE = claim_gates._rx(r"(?:^|\s)(?:لا|لم|لن|ما|ليس|مو|غير)\s+(?:\S+\s+)?$")


def comment_expresses_suggestion(comment_text: str | None) -> bool:
    """True when the visitor's own text contains a genuine, non-negated suggestion / wish / polite request / recommendation (see the table above)."""
    text = claim_gates.normalize(comment_text)
    if not text:
        return False
    for rx in _SUGGESTION_EXPRESSIONS:
        for m in rx.finditer(text):
            if not _NEGATED_BEFORE.search(text[max(0, m.start() - 24): m.start()]):
                return True
    return False


# --- Evidence that the visitor actually ASKED something (question-attribution family) -------------------------------------------------------------------
#
# A draft that thanks the visitor for "سؤالكم / استفساركم / تساؤلكم" (or says "استفسرتم / سألتم") credits them with a question. The visitor's own text must contain one:
#   strong   an interrogative word (هل، كيف، لماذا، متى، أين، كم، ماذا، وش، ايش ...), a "ما" question phrase (ما معنى، ما هو، ما الفرق ...), or an INDIRECT request for an
#            explanation / meaning with no question mark ("أرجو توضيح ...", "أريد معرفة ...", "اشرحوا لي", "لا أعرف معنى ...", "أبحث عن معنى ...", "لدي سؤال")
#   weak     only a question mark. A question mark proves that SOMETHING was asked, not WHAT: a generic "نشكركم على استفساركم" is allowed, a topic-specific
#            "استفساركم عن/حول/بشأن ..." is not.
# A negated cue ("لا أسأل عن ...") is not evidence. Whatever the evidence, a quoted term in the draft («...») must occur in the comment: the draft may not name a topic the
# visitor never mentioned.
_ASK_EXPRESSIONS = tuple(
    claim_gates._rx(p)
    for p in (
        r"(?<!\S)(?:هل|كيف|لماذا|ليش|ليه|متى|اين|أين|وين|فين|كم|ماذا|شنو|وش|ايش|إيش|مين|منو|علاش)(?!\S)",
        r"(?<!\S)ما\s+(?:معنى|معني|هو|هي|الفرق|المقصود|المراد|حكم|أصل|اصل|الصواب|تعريف|اسم|مرادف|جمع|مفرد|سبب|دلالة|معناه|معناها|الدليل)",
        r"(?<!\S)(?:ماذا|وش|ايش|إيش)\s+(?:تعني|يعني|معنى|معني)",
        r"(?<!\S)(?:ارجو|أرجو|نرجو|اريد|أريد|اود|أود|ابغى|أبغى|ابغي|ابي|أبي|احتاج|أحتاج|ارغب|أرغب|يرجى|الرجاء)\s+(?:\S+\s+){0,2}?"
        r"(?:معرفة|معرفه|توضيح|شرح|بيان|تفسير|ايضاح|إيضاح|افادة|إفادة|اجابة|إجابة|معنى|معني|الاستفسار|السؤال)",
        r"(?<!\S)(?:اشرحوا|اشرح|وضحوا|وضح|فسروا|بينوا|اخبروني|أخبروني|افيدوني|أفيدوني|عرفوني)(?!\S)",
        r"(?<!\S)قولوا\s+لي",
        r"(?<!\S)(?:اسأل|أسأل|استفسر|أستفسر|نستفسر|اتساءل|أتساءل)(?!\S)",
        r"(?<!\S)و?(?:استفسار|سؤال|تساؤل)(?!\S)",
        r"(?:سؤالي|استفساري|تساؤلي|(?:لدي|عندي)\s+(?:سؤال|استفسار|تساؤل))",
        r"(?:لا|ما|لم)\s+(?:أعرف|اعرف|أفهم|افهم|فهمت|ادري|أدري)\s+(?:معنى|معني|ما|كيف|لماذا|\S+)",
        r"(?:غير\s+واضح\s+لي|مو\s+فاهم|مب\s+فاهم)",
        r"(?:ابحث|أبحث|بحثت)\s+عن\s+(?:معنى|معني|تعريف|شرح)",
    )
)
_QUESTION_ATTRIBUTION = ("سؤالكم", "سؤالك", "استفساركم", "استفسارك", "تساؤلكم", "تساؤلك", "استفسرتم", "سألتم", "تساءلتم")
_TOPIC_AFTER_QUESTION = claim_gates._rx(r"(?:سؤالكم|سؤالك|استفساركم|استفسارك|تساؤلكم|تساؤلك|استفسرتم|سألتم|تساءلتم)\s+(?:عن|حول|بشأن|بشان|بخصوص|في)\s+\S+")
_QUOTED = re.compile(r"[«\"“]([^»\"”]{2,60})[»\"”]")


def comment_inquiry_evidence(comment_text: str | None) -> str | None:
    """'strong' (a genuine question or indirect request for an explanation, with or without a question mark), 'weak' (a question mark only) or None."""
    text = claim_gates.normalize(comment_text)
    if not text:
        return None
    for rx in _ASK_EXPRESSIONS:
        for m in rx.finditer(text):
            if not _NEGATED_BEFORE.search(text[max(0, m.start() - 24): m.start()]):
                return "strong"
    return "weak" if ("؟" in text or "?" in text) else None


def _question_attribution_supported(draft: str, comment_text: str) -> bool:
    evidence = comment_inquiry_evidence(comment_text)
    if evidence is None:
        return False
    d = claim_gates.normalize(draft)
    if evidence == "weak" and _TOPIC_AFTER_QUESTION.search(d):
        return False                                                  # a question mark does not tell WHAT was asked
    c = claim_gates.normalize(comment_text)
    for term in _QUOTED.findall(draft):
        core = re.sub(r"^(?:ال|لل|بال|وال|فال)", "", claim_gates.normalize(term)).strip()
        if core and core not in c:
            return False                                                  # the draft names a term the visitor never mentioned
    return True


# A thanks that credits the VISITOR's act with a first-person-plural possessive ("نشكركم على مشاركتنا مقترحاتكم": "for OUR sharing your suggestions") says the opposite of
# what was meant (مشاركتكم). Only these visitor-act nouns directly after "على" are matched, so valid constructions such as "تزويدنا بـ" or "تواصلكم معنا" are untouched.
_WRONG_POSSESSIVE = claim_gates._rx(r"(?<!\S)على\s+(?:مشاركتنا|اقتراحنا|مقترحنا|ملاحظتنا|تنبيهنا|مساهمتنا|إرسالنا|تواصلنا|اهتمامنا)(?![\u0621-\u064a])")


def _pronoun_gate(draft: str) -> GateResult:
    if _WRONG_POSSESSIVE.search(claim_gates.normalize(draft)):
        return GateResult("pronoun", False, "wrong_pronoun_construction")
    return GateResult("pronoun", True)


def _visitor_intent_attribution_gate(draft: str, comment_text: str | None) -> GateResult:
    """No-op (always passes) when `comment_text` is unavailable --
    correctness here requires comparing the draft's claim against the
    actual visitor text, so a caller that never supplies it (e.g. the
    dev-inspect arbitrary-draft-paste tool, or offline-eval scripts that
    predate this parameter) gets byte-identical behavior to before this
    gate existed."""
    if comment_text is None:
        return GateResult("visitor_intent_attribution", True)
    normalized_comment = _strip_diacritics(comment_text)
    for attribution_phrases, evidence_tokens in _ACTION_ATTRIBUTION_TO_EVIDENCE:
        if not any(phrase in draft for phrase in attribution_phrases):
            continue
        if attribution_phrases is _SUGGESTION_ATTRIBUTION:
            supported = comment_expresses_suggestion(comment_text)          # in-context expressions, negation aware (was: bare substring list)
        else:
            supported = any(token in normalized_comment for token in evidence_tokens)
        if not supported:
            return GateResult("visitor_intent_attribution", False, "unsupported_visitor_intent_claim")
    stripped = _strip_diacritics(draft)
    if any(p in stripped for p in _QUESTION_ATTRIBUTION) and not _question_attribution_supported(draft, comment_text):
        return GateResult("visitor_intent_attribution", False, "unsupported_visitor_intent_claim")
    return GateResult("visitor_intent_attribution", True)


def _claim_gates(
    draft: str, trusted_grounding_text: str | None, comment_text: str | None, *, include_assertion_gate: bool = True
) -> GateResult:
    """Structural claim gates (rcs.response_generation.claim_gates): URLs, commitments, policies, numbers, definitions, correctness verdicts.
    Run on normalised text so inflection / spelling variants are caught; fail closed when grounding is empty."""
    failed = claim_gates.first_failing_claim_gate(
        draft, comment_text=comment_text, grounding=trusted_grounding_text, include_assertion_gate=include_assertion_gate
    )
    if failed is None:
        return GateResult("claim_gates", True)
    gate, reason = failed
    return GateResult(gate, False, reason)


def run_validation_gates(
    text: str | None,
    *,
    trusted_grounding_text: str | None = None,
    comment_text: str | None = None,
) -> ValidationSummary:
    """Fixed order, no dynamic registration, no reflection, stops at the
    first failing gate -- the exact first-failure semantics
    validate_draft has always had. Order: content_shape -> internal_leak
    -> pii -> boilerplate -> evaluator_voice -> unsupported_claim ->
    visitor_intent_attribution (the original order, plus evaluator_voice
    inserted alongside boilerplate as of rg-v9, plus
    visitor_intent_attribution appended last -- all three added gates
    are "wrong shape/content of reply" checks, evaluated after the
    original checks so existing failure precedence is unchanged)."""
    evaluated: list[GateResult] = []

    shape = _content_shape_gate(text)
    evaluated.append(shape)
    if not shape.passed:
        return ValidationSummary(False, shape, tuple(evaluated))

    draft = text.strip()  # safe: content_shape_gate already proved this is non-empty

    for gate_fn in (_internal_leak_gate, _pii_gate, _boilerplate_gate, _evaluator_voice_gate):
        result = gate_fn(draft)
        evaluated.append(result)
        if not result.passed:
            return ValidationSummary(False, result, tuple(evaluated))

    claim_result = _unsupported_claim_gate(draft, trusted_grounding_text)
    evaluated.append(claim_result)
    if not claim_result.passed:
        return ValidationSummary(False, claim_result, tuple(evaluated))

    attribution_result = _visitor_intent_attribution_gate(draft, comment_text)
    evaluated.append(attribution_result)
    if not attribution_result.passed:
        return ValidationSummary(False, attribution_result, tuple(evaluated))

    pronoun_result = _pronoun_gate(draft)
    evaluated.append(pronoun_result)
    if not pronoun_result.passed:
        return ValidationSummary(False, pronoun_result, tuple(evaluated))

    # Structural claim gates run LAST so every pre-existing gate keeps its failure precedence and reason code.
    structural_result = _claim_gates(draft, trusted_grounding_text, comment_text)
    evaluated.append(structural_result)
    if not structural_result.passed:
        return ValidationSummary(False, structural_result, tuple(evaluated))

    return ValidationSummary(True, None, tuple(evaluated))


def validate_draft(
    text: str | None,
    *,
    trusted_grounding_text: str | None = None,
    comment_text: str | None = None,
) -> tuple[bool, str | None]:
    """Unchanged public contract: (True, None) on success, (False,
    reason_code) on the first failing gate. Every existing caller
    (rcs.response_generation.service, offline eval scripts, tests)
    keeps working against this exact signature and return shape --
    run_validation_gates is the new internal structure; this function
    is now a one-line facade over it. `comment_text` is a new, optional,
    keyword-only parameter (default None): every existing call site that
    never passes it gets exactly its previous behavior (see
    `_visitor_intent_attribution_gate`)."""
    summary = run_validation_gates(
        text, trusted_grounding_text=trusted_grounding_text, comment_text=comment_text
    )
    if summary.passed:
        return True, None
    return False, summary.first_failure.reason_code


# Findings a reviewer MAY attest (lexical / factual statements that cannot be checked against approved grounding). Everything else is
# never attestable: URLs, institutional commitments and promises, institutional policies / procedures / durations / channels, operational
# claims, internal leaks and personal data, empty or over-long text.
ATTESTABLE_FINDINGS = frozenset({"unsupported_definition", "unsupported_correctness_claim", "unsupported_number"})


@dataclass(frozen=True)
class ReplyFindings:
    blocking: tuple[str, ...]
    attestable: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.blocking and not self.attestable


def collect_reply_findings(
    text: str | None, *, trusted_grounding_text: str | None = None, comment_text: str | None = None
) -> ReplyFindings:
    """ALL findings for reviewer-written (or stale generated) text, split into never-attestable (`blocking`) and attestable. Same gates as
    `validate_reviewer_edit`, but collects instead of stopping at the first failure. Pure; never rewrites `text`."""
    if text is None or not str(text).strip():
        return ReplyFindings(("empty_output",), ())
    draft = text.strip()
    found: list[str] = []
    if len(draft) > settings.RESPONSE_DRAFT_MAX_CHARS:
        found.append("excessive_length")
    for gate_fn in (_internal_leak_gate, _pii_gate, lambda d: _unsupported_claim_gate(d, trusted_grounding_text)):
        r = gate_fn(draft)
        if not r.passed and r.reason_code:
            found.append(r.reason_code)
    found.extend(
        claim_gates.all_failing_claim_gates(draft, comment_text=comment_text, grounding=trusted_grounding_text, include_assertion_gate=False)
    )
    found = list(dict.fromkeys(found))
    return ReplyFindings(
        tuple(c for c in found if c not in ATTESTABLE_FINDINGS), tuple(c for c in found if c in ATTESTABLE_FINDINGS)
    )


def validate_reviewer_edit(
    text: str | None,
    *,
    trusted_grounding_text: str | None = None,
    comment_text: str | None = None,
) -> tuple[bool, str | None]:
    """Safety gates for a reply a HUMAN wrote or edited, evaluated before any send action. Same safety gates as generated drafts
    (non-empty, length ceiling, internal leak, PII, unsupported operational claims, structural claim gates). Style gates (boilerplate,
    evaluator voice, visitor-intent attribution) and the minimum-length / Arabic-letter-count shape checks are NOT applied: they judge
    model prose, not a reviewer's deliberate wording. The text is never rewritten; the caller gets a reason code and must ask the
    reviewer to correct it and re-validate."""
    if text is None or not str(text).strip():
        return False, "empty_output"
    draft = text.strip()
    if len(draft) > settings.RESPONSE_DRAFT_MAX_CHARS:
        return False, "excessive_length"
    for gate_fn in (_internal_leak_gate, _pii_gate):
        r = gate_fn(draft)
        if not r.passed:
            return False, r.reason_code
    r = _unsupported_claim_gate(draft, trusted_grounding_text)
    if not r.passed:
        return False, r.reason_code
    r = _claim_gates(draft, trusted_grounding_text, comment_text, include_assertion_gate=False)
    if not r.passed:
        return False, r.reason_code
    return True, None
