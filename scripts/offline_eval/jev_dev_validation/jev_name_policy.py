"""Context-aware sanitisation policy for text that may leave the machine. LOCAL ONLY (pure functions + a cached corpus lookup).

POLICY (agreed): never send visitor name / email / phone / identifier FIELDS, and send only the minimum comment content,
sanitised. An Arabic (or Latin) name appearing INSIDE a comment is not automatically personal information:

  MASK  ->  [NAME]   names that identify the visitor or a private individual:
                      * self-introduction   "أنا محمد وأقترح ..." -> "أنا [NAME] وأقترح ..." ; "اسمي ..." ; "معك ..." ; "مع حضراتكم ..."
                      * a named relative with a nasab chain  "جدتي ساره بنت X بن Y" -> "جدتي [NAME]"
                      * multi-link genealogies  "X بن Y بن Z"
                      * Latin names after an identity cue (my name is / I am / regards / Mr / Dr) or next to an e-mail/URL/ID
  KEEP                names used as dictionary / linguistic content:
                      * the subject of a meaning/origin question  "ما معنى اسم محمد؟"
                      * a proposed word or headword  "أقترح إضافة كلمة محمد" ; "الكلمة: X"
                      * a single historical / literary figure cited as a source (a lone "ابن X")
                      * Latin terms that are vocabulary ("Watch party", "science communication")
  REMOVE              contact-form boilerplate that is only visitor metadata: "الجوال: [ID], نص الرسالة:", "رقم الجوال: [ID]"
  QUARANTINE (fail closed)  residual e-mail/URL/long digits/@, a name glued to a number, sensitive personal circumstances
                      (health, money/benefits, marriage/legal) told about a private person, or text that is only identity.

Rules cannot be perfect: unusual names after a cue can still be missed, and a lexical name can occasionally be masked. A human
review of the output stays mandatory before any transmission (see docs/evaluations/jev_privacy_review_checklist.md).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jev_common  # noqa: E402,F401
from rcs.classification.external_redaction import (  # noqa: E402
    _ADDRESS, _EMAIL, _HANDLE, _ID_WITH_CUE, _LONG_DIGITS, _SIGNATURE, _URL,
)

NAME = "[NAME]"
IDENTITY_MASK_ACTIONS = ("first_person_name", "self_introduction_name", "kinship_name", "kinship_name_with_nasab",
                         "multi_link_genealogy", "signature_name", "name_glued_to_identifier", "latin_name_after_identity_cue",
                         "latin_name_next_to_contact_pattern")
TOK = r"[؀-ۿ]+"
_DIAC = re.compile(r"[ً-ٰٟـ]")

# ---- contact-form boilerplate (visitor metadata, not comment content)
_FORM_PREFIX = re.compile(r"^\s*(?:الجوال\s*:\s*(?:\[ID\]|null|\d+)\s*,?\s*نص\s+الرسالة\s*:|محتوي\s+الرسالة\s*:|نص\s+الرسالة\s*:)\s*")
_FORM_PHONE_TAIL = re.compile(r"\s*(?:رقم\s+)?الجوال\s*:?\s*(?:\[ID\]|null|\+?\d[\d\s().-]*)?(?=\s|$)")  # the phone FIELD label (+ value), anywhere
# the English form variant appends "Phone Number: <value>" (or "Test Phone ...") to the visitor's text: metadata, not content
_FORM_PHONE_EN = re.compile(r"(?i)\s*(?:\b(?:test\s+)?phone\s+number\b|\btest\s+phone\b)\s*:?\s*(?:\[ID\]|null|\+?\d[\d\s().-]*)?")

# ---- lexical-protection cues: a name right after these is the SUBJECT of the comment, never masked
_LEX_PREFIX = re.compile(r"(?:معنى\s+(?:اسم\s+)?|اسم\s+|أصل\s+(?:اسم\s+)?|اشتقاق\s+(?:اسم\s+)?|كلمة\s+|الكلمة\s*:?\s*|لفظ(?:ة)?\s+|مصطلح\s+|لقب\s+)$")

_STRONG_INTRO = re.compile(rf"(?P<cue>(?:اسمي|إسمي|معك|معكم|اخوكم|أخوكم|اختكم|أختكم|مع\s+حضراتكم|مع\s+تحياتي|تحياتي)\s*[:،,-]?\s*)(?P<n>{TOK}(?:\s+{TOK}){{0,2}})")
_ANA = re.compile(rf"(?<![؀-ۿ])(?P<cue>(?:أنا|انا|اني|إني)\s+)(?P<n>{TOK})")
_KIN = re.compile(rf"(?<![؀-ۿ])(?P<cue>(?:جدتي|جدي|امي|أمي|اختي|أختي|اخي|أخي|عمي|عمتي|خالي|خالتي|زوجتي|زوجي|ابني|بنتي|والدي|والدتي|ابوي|أبوي|أخوي)\s+)(?P<n>{TOK})")
_NASAB_LINKS = re.compile(rf"(?P<n>{TOK}(?:\s+(?:بنت|بن|ابن)\s+{TOK}){{2,5}})")
_KIN_NASAB = re.compile(rf"(?P<cue>(?:جدتي|جدي|امي|أمي|اختي|أختي|عمتي|خالتي|والدتي|ابوي|أبوي)\s+)(?P<n>{TOK}\s+(?:بنت|بن|ابن)\s+{TOK}(?:\s+(?:بنت|بن|ابن)\s+{TOK}){{0,4}})")
_NASAB_TAIL = re.compile(rf"(?P<n>(?:بنت|بن|ابن)\s+{TOK}(?:\s+(?:بنت|بن|ابن|ال)\s+{TOK}){{0,4}})")
_ID_GLUED = re.compile(rf"\[ID\]\s*(?P<n>{TOK}(?:\s+{TOK}){{0,3}})")
_LATIN_IDENT = re.compile(r"(?i)\b(?P<cue>my\s+name\s+is|i\s+am|i'm|regards|sincerely|best|thanks|mr\.?|mrs\.?|ms\.?|dr\.?)\s*,?\s+(?P<n>[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)")
_LATIN_NEXT_TO_CONTACT = re.compile(r"(?P<n>[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\s*(?=\[(?:EMAIL|URL|ID)\])")
_LATIN_BIGRAM = re.compile(r"(?<![A-Za-z])[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2}(?![A-Za-z])")

# ---- sensitive personal circumstances about a private person (fail closed)
_SENSITIVE = re.compile(r"(?:الإسكان\s+التنموي|مستحق(?:ه|ة)?\s+(?:منزل|سكن|اعانة|إعانة)|إعانة|الضمان\s+الاجتماعي|راتبي|ديوني|"
                        r"طلاق|انفصل\s+عنها|انفصلت|زوجته\s+التي|سجن|محكمة|قضية\s+ضد|سرطان|مرضي|مريض(?:ة)?\s+بـ)")

_STOP = {"من", "في", "على", "عن", "إلى", "الى", "الذي", "التي", "هذا", "هذه", "ذلك", "هو", "هي", "لا", "ما", "لم", "لن", "قد", "كل",
         "أريد", "اريد", "أود", "اود", "أرجو", "ارجو", "أقترح", "اقترح", "أتمنى", "اتمنى", "اعرف", "أعرف", "احب", "أحب", "اقول", "أقول",
         "بس", "مع", "او", "أو", "ثم", "كان", "كنت", "عندي", "لدي", "طالب", "طالبة", "مستخدم", "زائر"}

_corpus_cache: dict | None = None


def norm(t: str) -> str:
    return _DIAC.sub("", t).translate(str.maketrans("أإآٱىة", "اااايه"))


def corpus_stats() -> dict:
    """Token statistics over the full local source corpus, used ONLY to tell common predicates from rare (name-like) tokens.
    Never printed or persisted. Falls back to empty statistics (=> more masking) if the corpus is unavailable."""
    global _corpus_cache
    if _corpus_cache is not None:
        return _corpus_cache
    stats = {"after_ana": Counter(), "after_kin": Counter(), "strong_names": Counter()}
    raw_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "historical-export.xlsx"
    try:
        from config import settings
        from rcs.ingest.loader import load_raw
        raw = load_raw(raw_path, settings.SOURCE_COLUMNS)
        for t in raw["original_comment"].fillna("").astype(str):
            for m in _ANA.finditer(t):
                stats["after_ana"][norm(m.group("n"))] += 1
            for m in _KIN.finditer(t):
                stats["after_kin"][norm(m.group("n"))] += 1
            for m in re.finditer(rf"(?:اسمي|إسمي|معك)\s*[:،,-]?\s*({TOK})", t):
                stats["strong_names"][norm(m.group(1))] += 1
    except Exception:  # corpus unavailable: keep empty stats (conservative)
        pass
    _corpus_cache = stats
    return stats


@dataclass
class Result:
    text: str | None
    actions: Counter = field(default_factory=Counter)
    quarantine_reason: str | None = None


def _protected(text: str, start: int) -> bool:
    return bool(_LEX_PREFIX.search(text[max(0, start - 24):start]))


def _mask_span(text: str, m: re.Match, group: str = "n") -> tuple[str, int]:
    s, e = m.span(group)
    return text[:s] + NAME + text[e:], s + len(NAME) - e


def _stopset() -> set[str]:
    return {norm(x) for x in _STOP}


def _trim_intro(n: str, max_tokens: int = 3) -> int:
    """Length (in chars) of the leading part of `n` that is a name: up to 3 tokens, stopping at function words, verbs
    from the stop list and conjunction-prefixed tokens (so 'اسمي سعد الغامدي أقترح كلمة' masks only 'سعد الغامدي')."""
    stats = corpus_stats()
    end, count = 0, 0
    for m in re.finditer(TOK, n):
        tok = norm(m.group(0))
        if count >= max_tokens or tok in _stopset() or (m.group(0).startswith("و") and stats["strong_names"][tok] == 0):
            break
        end, count = m.end(), count + 1
    return end


def _sub_names(text: str, rx: re.Pattern, actions: Counter, key: str, accept=lambda m: True, trim=None) -> str:
    out, offset = text, 0
    for m in list(rx.finditer(text)):
        if _protected(text, m.start("n")) or not accept(m):
            continue
        s, e = m.span("n")
        if trim is not None:
            length = trim(m.group("n"))
            if length <= 0:
                continue
            e = s + length
        out = out[: s + offset] + NAME + out[e + offset:]
        offset += len(NAME) - (e - s)
        actions[key] += 1
    return out


def sanitize(text: str) -> Result:
    res = Result(text=None)
    a = res.actions
    t = " ".join(str(text).split()) if text is not None else ""
    if not t.strip():
        res.quarantine_reason = "empty"
        return res
    t2 = _FORM_PREFIX.sub("", t)
    t2 = _FORM_PHONE_TAIL.sub("", t2)
    t2 = _FORM_PHONE_EN.sub(" ", t2)
    if t2 != t:
        a["contact_form_boilerplate_removed"] += 1
    t = t2.strip()

    # structural identifiers (same detectors as the production redactor; names are handled by the policy below)
    for rx, rep, key in ((_EMAIL, "[EMAIL]", "email"), (_URL, "[URL]", "url"), (_HANDLE, "[HANDLE]", "handle"),
                         (_ID_WITH_CUE, "[ID]", "id_with_cue"), (_LONG_DIGITS, "[ID]", "long_digits"), (_ADDRESS, "[ADDRESS]", "address")):
        t, n = rx.subn(rep, t)
        if n:
            a[key] += n
    t, n = _SIGNATURE.subn("\n" + NAME, t)
    if n:
        a["signature_name"] += n
    t = " ".join(t.split())

    if _SENSITIVE.search(t):
        res.quarantine_reason = "sensitive_personal_circumstance"
        return res

    stats = corpus_stats()
    # identity masking
    t = _sub_names(t, _KIN_NASAB, a, "kinship_name_with_nasab")
    t = _sub_names(t, _STRONG_INTRO, a, "self_introduction_name", trim=_trim_intro)
    t = _sub_names(t, _ANA, a, "first_person_name",
                   accept=lambda m: norm(m.group("n")) not in _stopset() and not m.group("n").startswith("و")
                   and stats["after_ana"][norm(m.group("n"))] <= 2 and len(m.group("n")) >= 3)
    t = _sub_names(t, _KIN, a, "kinship_name",
                   accept=lambda m: stats["after_kin"][norm(m.group("n"))] <= 2 and norm(m.group("n")) not in _stopset())
    t = _sub_names(t, _NASAB_LINKS, a, "multi_link_genealogy")
    t, n = _ID_GLUED.subn(lambda m: "[ID] " + NAME, t)
    a["name_glued_to_identifier"] += n
    t, n = _LATIN_IDENT.subn(lambda m: m.group(0).replace(m.group("n"), NAME), t)
    a["latin_name_after_identity_cue"] += n
    t, n = _LATIN_NEXT_TO_CONTACT.subn(NAME, t)
    a["latin_name_next_to_contact_pattern"] += n
    t = " ".join(t.split())

    # a leftover nasab fragment ("بنت X" / "بن X") directly after a mask is still part of the identity
    t = re.sub(rf"{re.escape(NAME)}\s+(?:بنت|بن|ابن)\s+{TOK}(?:\s+(?:بنت|بن|ابن)\s+{TOK})*", NAME, t)

    # residual fail-closed checks
    if _EMAIL.search(t) or _URL.search(t) or "@" in t.replace("[EMAIL]", ""):
        res.quarantine_reason = "residual_contact_pattern"
        return res
    if _LONG_DIGITS.search(re.sub(r"\[[A-Z]+\]", "", t)):
        res.quarantine_reason = "residual_long_digits"
        return res
    if re.search(r"\[ID\][؀-ۿA-Za-z]", t):
        res.quarantine_reason = "name_glued_to_identifier"
        return res
    letters = re.sub(r"\[[A-Z]+\]", "", t)
    if len(re.findall(r"[؀-ۿA-Za-z]", letters)) < 2:
        res.quarantine_reason = "only_identity_or_empty_content"
        return res
    kept_latin = _LATIN_BIGRAM.findall(t)
    if kept_latin:
        a["latin_term_kept_as_lexical_content"] += len(kept_latin)
    res.text = t
    return res
