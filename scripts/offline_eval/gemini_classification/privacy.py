"""Privacy preflight for the classification request. Reuses the production outbound privacy boundary (rcs.response_generation.outbound_privacy):
names and identifiers are sanitised, uncertain text is BLOCKED (the row is then ineligible for every system being compared), and the final prompt is
re-checked by guard_prompt() immediately before any call. Also gates real data behind a written approval."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rcs.response_generation.outbound_privacy import OutboundPrivacyError, guard_prompt, sanitize_text


class RealDataNotApproved(PermissionError):
    pass


@dataclass(frozen=True)
class LiveApproval:
    """A written authorisation to send sanitised REAL comments to Gemini. Nothing in this preparation phase creates one."""

    approver: str
    approved_on: str
    scope: str
    max_calls: int
    max_usd: float
    datasets: tuple[str, ...] = ()
    input_sha256: tuple[tuple[str, str], ...] = ()      # (dataset, sha256 of the human-reviewed sanitised inputs)

    def validate(self) -> None:
        if not (self.approver and self.approved_on and self.scope):
            raise RealDataNotApproved("approval is incomplete")
        if self.max_calls <= 0 or self.max_usd <= 0:
            raise RealDataNotApproved("approval must set positive call and cost caps")


@dataclass(frozen=True)
class PreflightResult:
    eligible: bool
    text: str | None
    blocked_reason: str | None = None
    actions: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------------------- residual-risk screen
# The production sanitiser is rule-based: it masks identifiers and names introduced by an identity cue, but it cannot know a bare name. A synthetic
# adversarial battery (tests/test_gemini_classification_privacy.py) showed 17 of 35 cases passing it. This SECOND screen is part of the evaluation
# harness only (production is untouched): anything that still looks like it may carry a name, handle, contact detail or workplace after sanitising
# is BLOCKED -- losing a row is cheap, a leak is not. Blocked rows leave every compared system, so comparisons stay on identical rows.
_DIAC = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_W = r"[\u0600-\u06FF]+"
_KUNYA = re.compile(rf"(?<![\u0600-\u06FF])(أبو|ابو|أبا|ابا|أبي|ابي|أم|ام)\s+({_W})")
_NASAB = re.compile(rf"(?<![\u0600-\u06FF])(بن|ابن|بنت|آل)\s+({_W})")
_KIN = re.compile(r"(?<![\u0600-\u06FF])(جدتي|جدي|امي|أمي|ابي|أبي|اخي|أخي|اختي|أختي|ابني|ابنتي|بنتي|زوجتي|زوجي|خالي|خالتي|عمي|عمتي|صديقي|صديقتي|زميلي|زميلتي|جاري|جارتي|ولدي)(?![\u0600-\u06FF])")
_TITLE = re.compile(rf"(?<![\u0600-\u06FF])(الدكتور|الدكتورة|دكتور|الاستاذ|الأستاذ|الاستاذة|الأستاذة|المهندس|المهندسة|الشيخ|الشيخة|السيد|السيدة|الأمير|الأميرة|الملك)\s+({_W})")
_VOCATIVE_OK = {"ليت", "رب", "الله", "سلام", "اخوان", "إخوان", "جماعة", "اهل", "أهل", "فريق", "قوم", "سيدي", "عزيزي", "اخي", "أخي", "الله", "لها", "له", "هذا", "هذه", "لكم", "حبذا"}
_VOCATIVE = re.compile(rf"(?<![\u0600-\u06FF])يا\s+({_W})")
_CLOSING = re.compile(rf"(تحياتي|التقدير|التحية|الاحترام|تقديري|شكري|المخلص|المخلصة|اخوكم|أخوكم|اختكم|أختكم|مرسله|مرسلها)\s*[:،,\-–]?\s*({_W})(\s+{_W}){{0,2}}\s*$")
_LAST_LINE_OK = {"شكرا", "شكراً", "شكرا لكم", "جزاكم الله خيرا", "وفقكم الله", "بارك الله فيكم", "تحياتي", "مع الشكر", "مع خالص التقدير", "والسلام"}
_ROLE_NAME = re.compile(rf"(?<![\u0600-\u06FF])(الأخ|الاخ|الأخت|الاخت|الأخوة|المشرف|المشرفة|المدير|المديرة|الباحث|الباحثة|الموظف|الموظفة|المسؤول|المسؤولة|الزميل|الزميلة)\s+(?!الكريم|الفاضل)({_W})")
_SENDER = re.compile(r"(المرسل|المرسلة|مرسل من|من طرف|sent by|from|regards|best|thanks|cheers)\s*[:\-,]?\s*[A-Za-z\u0600-\u06FF]+", re.I)
_DASH_SIGNATURE = re.compile(rf"\s[—–\-]\s*{_W}(\s+{_W})?\s*$")
_COLON_NAME_LIST = re.compile(rf":\s*{_W}\s+و\s*{_W}\s*$|:\s*{_W}\s*$")
_THANKS_OK = "لكم|لك|جزيلا|جزيلاً|كثيرا|كثيراً|جدا|جداً|على|من|يا|و|الله|عليكم|لكما|للقائمين|للجميع|جميعا|بارك|وبارك|مرة|ولكم|وجزاكم|لهذا|لهذه|لهم|لها|له|إلى|الى|والله|وحياكم|كل|فريق"
_THANKS_NAME = re.compile(r"(?<![\u0600-\u06FF])(شكرا|شكراً|شكر|ممنون|مشكور|مشكورة)\s+(?!(?:" + _THANKS_OK + r")(?![\u0600-\u06FF]))(" + _W + ")")
_CONTACT_WITH = re.compile(r"(تواصل\w*|اتصل\w*|كلم\w*|راسل\w*|بلغ\w*)\s+(?:مع|ب|لـ)?\s*(?:مع\s+)?(" + _W + ")")
_HANDLE_LIKE = re.compile(r"(?<![\w])[A-Za-z][A-Za-z0-9]*[._][A-Za-z0-9._-]*[A-Za-z0-9]")
_URL_PATH = re.compile(r"[A-Za-z0-9-]+\.[A-Za-z0-9.-]+/\S+")
_OBF_EMAIL = re.compile(r"(\(|\[|\{)\s*(at|dot)\s*(\)|\]|\})|\b(at|dot)\b\s*\w+\s*\b(dot)\b|\b\w+\s+dot\s+\w+", re.I)
_DIGIT_WORDS = r"(صفر|واحد|اثنين|اثنان|ثنتين|ثلاثة|ثلاث|اربعة|أربعة|اربع|خمسة|خمس|ستة|ست|سبعة|سبع|ثمانية|ثماني|تسعة|تسع)"
_SPELLED_NUMBER = re.compile(rf"{_DIGIT_WORDS}(\s+{_DIGIT_WORDS}){{3,}}")
_WORKPLACE = re.compile(rf"(أعمل|اعمل|أشتغل|اشتغل|موظف|موظفة|وظيفتي|طالب|طالبة)\s+.{{0,40}}?(شركة|مؤسسة|جامعة|مدرسة|وزارة|بنك|مستشفى|كلية)\s+{_W}")
_CHILD_AGE = re.compile(r"(عمره|عمرها|عمري)\s*[\d\u0660-\u0669]+")
_LATIN_NAME = re.compile(r"(?<![A-Za-z])[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?")


def _short_signature_line(raw: str) -> bool:
    """Checked on the RAW text: the production sanitiser collapses line breaks, which would hide a name on its own last line."""
    lines = [ln for ln in _DIAC.sub("", raw or "").strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    last = lines[-1].strip(" -–—*_.:،")
    return bool(last) and len(last.split()) <= 3 and last not in _LAST_LINE_OK and "[" not in last


def residual_screen(sanitised: str) -> list[str]:
    """Reasons (stable codes, no text) why sanitised text should still not be sent. Empty list = no residual-risk cue found."""
    s = _DIAC.sub("", sanitised)
    reasons: list[str] = []
    for code, rx in (("kunya", _KUNYA), ("nasab", _NASAB), ("kinship_term", _KIN), ("title_plus_name", _TITLE), ("closing_plus_name", _CLOSING),
                     ("role_plus_name", _ROLE_NAME), ("sender_label", _SENDER), ("dash_signature", _DASH_SIGNATURE), ("trailing_colon_name", _COLON_NAME_LIST), ("thanks_plus_name", _THANKS_NAME), ("contact_person", re.compile(r"(تواصل\w*|اتصل\w*|كلم\w*|راسل\w*)\s+مع\s+" + _W)),
                     ("handle_like_token", _HANDLE_LIKE), ("url_path", _URL_PATH), ("obfuscated_email", _OBF_EMAIL), ("spelled_phone_number", _SPELLED_NUMBER),
                     ("workplace", _WORKPLACE), ("age_of_person", _CHILD_AGE)):
        if rx.search(s):
            reasons.append(code)
    for m in _VOCATIVE.finditer(s):
        if m.group(1) not in _VOCATIVE_OK:
            reasons.append("vocative_name")
            break
    if len(s.split()) <= 1:
        reasons.append("single_token_text")
    if re.search(r"[\u0600-\u06FF]", s) and _LATIN_NAME.search(re.sub(r"\[[^\]]*\]", "", s)):
        reasons.append("latin_capitalised_word")
    return reasons


def preflight(text: str | None) -> PreflightResult:
    res = sanitize_text(text, names=True)
    if res.text is None:
        return PreflightResult(False, None, res.blocked_reason or "blocked", dict(res.actions))
    # differential check: masking must not depend on diacritics/tatweel (a known way a self-introduction slips past the cue rules)
    plain = sanitize_text(_DIAC.sub("", text or ""), names=True)
    if plain.text is None or _DIAC.sub("", res.text) != plain.text:
        return PreflightResult(False, None, "residual_risk:normalisation_sensitive_masking", dict(res.actions))
    reasons = residual_screen(res.text)
    if _short_signature_line(text or ""):
        reasons.append("short_signature_line")
    if reasons:
        return PreflightResult(False, None, "residual_risk:" + ",".join(reasons), dict(res.actions))
    return PreflightResult(True, res.text, None, dict(res.actions))


def egress_guard(prompt: str) -> str:
    """Final check on the exact outbound prompt. Raises OutboundPrivacyError (no request is made) if anything identifying remains."""
    return guard_prompt(prompt)


def require_provenance(provenance: str, approval: LiveApproval | None, dataset: str = "") -> None:
    if provenance == "synthetic":
        return
    if approval is None:
        raise RealDataNotApproved("real visitor data requires an explicit LiveApproval; none was supplied")
    approval.validate()
    if dataset and approval.datasets and dataset not in approval.datasets:
        raise RealDataNotApproved(f"dataset {dataset!r} is not covered by the approval")
    if dataset and dataset not in dict(approval.input_sha256):
        raise RealDataNotApproved(f"no approved sanitised-input hash for dataset {dataset!r}")


__all__ = ["LiveApproval", "OutboundPrivacyError", "PreflightResult", "RealDataNotApproved", "egress_guard", "preflight", "require_provenance"]
