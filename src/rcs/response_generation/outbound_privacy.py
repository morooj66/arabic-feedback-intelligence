"""Privacy boundary for EVERY outbound Gemini response-generation request (initial draft, retry, refinement).

`guard_prompt(prompt)` is applied to the fully assembled prompt immediately before `client.generate` (see service._call_with_retry), so no caller, prompt
version or retry path can bypass it. The fixed system instructions are trusted and untouched; every VARIABLE block is sanitised in place:

    UNTRUSTED VISITOR COMMENT, IDENTIFYING CONTEXT (entry title), STYLE REFERENCE EXAMPLES, PREVIOUS DRAFT   -> full sanitisation (identifiers + personal names)
    OFFICIAL KNOWLEDGE (approved text)                                                                         -> identifiers only (e-mail, long digit runs, handles)

Policy (same as the project's agreed external-text policy, reimplemented here because the tooling version lives under scripts/):
  MASK   phone numbers, e-mail addresses, external links, social handles, id-like digit runs, addresses, contact-form boilerplate, and names that identify the
         VISITOR or a private person: self-introduction ("أنا ..." / "اسمي ..." / "معك ..."), signatures ("تحياتي ..."), named relatives, nasab chains,
         Latin names after an identity cue ("My name is ...", "Regards ...").
  KEEP   dictionary content: a name that is the subject of a meaning/origin question, a proposed headword ("أقترح إضافة كلمة محمد"), "الكلمة: X", lone
         historical/literary figures, Latin vocabulary. First-person VERBS and common predicates after "أنا" are not names.
  BLOCK  (fail closed, no external request is made): sensitive personal circumstances about a private person (health, money/benefits, legal, marriage),
         identifiers that survive masking, a name glued to an identifier, or text that is only identity.
Deterministic rules cannot be perfect: an unusual name after no identity cue can still be missed, and an unusual predicate after a cue can be masked
(privacy-first). The accepted residual risks are listed in docs/evaluations/outbound_privacy.md.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from rcs.classification.external_redaction import _ADDRESS, _ID_WITH_CUE, _LONG_DIGITS
from rcs.classification.redact import redact_phone_numbers
from rcs.response_generation.claim_gates import APPROVED_URLS

# Identifier patterns that also match when glued to a preceding Arabic letter or conjunction ("وhttps://...", "وsara@x.com", "و@handle"), which the shared
# external_redaction detectors miss because \b / \w treat Arabic letters as word characters.
_EMAIL = re.compile(r"(?<![A-Za-z0-9_.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.I)
_URL = re.compile(r"(?i)(?:https?://|www\.)\S+|(?<![A-Za-z0-9_-])(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|info|io|me|co|app|sa|ly|tv)\b(?:/\S*)?")
_HANDLE = re.compile(r"(?<![A-Za-z0-9_@])@[A-Za-z0-9_][A-Za-z0-9_.-]{1,30}")

NAME = "[name removed]"
LINK = "[link removed]"
HANDLE = "[handle removed]"
EMAIL = "[email removed]"
IDENT = "[id removed]"
ADDRESS = "[address removed]"
TOK = r"[؀-ۿ]+"
_DIAC = re.compile(r"[ً-ٰٟـ]")


class OutboundPrivacyError(RuntimeError):
    """The text could not be shown to be safe for an external model. No request is made."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Sanitized:
    text: str | None
    actions: Counter = field(default_factory=Counter)
    blocked_reason: str | None = None


def norm(t: str) -> str:
    return _DIAC.sub("", t).translate(str.maketrans("أإآٱىة", "اااايه"))


# ----------------------------------------------------------------------------------- what is NOT a name after a cue
_FUNCTION_AND_PREDICATES = """
من في على عن الى إلى الذي التي هذا هذه ذلك هو هي لا ما لم لن قد كل مع او أو ثم كان كانت كنت عندي لدي عنده عندها لها له
اريد أريد اود أود ارجو أرجو اقترح أقترح اتمنى أتمنى اعرف أعرف احب أحب اقول أقول بس طالب طالبة مستخدم زائر باحث باحثة معلم معلمة مترجم مترجمة دارس دارسة مهتم مهتمة
ارى أرى اعتقد أعتقد اظن أظن اجد أجد ارغب أرغب ابحث أبحث اكتب أكتب اعمل أعمل ادرس أدرس استخدم أستخدم اشكر أشكر اطلب أطلب اسأل أسأل افضل أفضل احتاج أحتاج
استطيع أستطيع اقدم أقدم ارسل أرسل اضيف أضيف اعلم أعلم ابغى أبغى ابي أبي اشوف أشوف احس أحس احترم أحترم اخاف أخاف اقصد أقصد اكمل أكمل اقف أقف اجلس أجلس
اسجل أسجل اشهد أشهد اوصل أوصل اثبت أثبت اقولها أقولها اعربها أعربها اتابع أتابع استعمل أستعمل اتحدث أتحدث اقرأ أقرأ اقرا اتعلم أتعلم اعلق أعلق اشارك أشارك
اطلع أطلع الاحظ ألاحظ لاحظت لاحظ وجدت قرأت قرات بحثت استخدمت اقترحت اتمنى ملاحظ مندهش متفاجئ سعيد سعيدة مبسوط مبسوطة محظوظ محظوظة راضي راضية
يقول يقولها تقول تقولها يسمي تسمي تسميها اسمها اسمه ام اب مره كل دايما دائما حاليا اذا الله حفظه حفظها رحمه رحمها الي اللي مو مدري مادري فقط بدون عشان
زوجتي زوجي امي أمي ابوي أبوي اختي أختي اخي أخي عمي عمتي خالي خالتي جدي جدتي والدي والدتي ابني بنتي
""".split()
_PREDICATES = {norm(x) for x in _FUNCTION_AND_PREDICATES}

_LEX_PREFIX = re.compile(r"(?:معنى\s+(?:اسم\s+)?|اسم\s+|أصل\s+(?:اسم\s+)?|اشتقاق\s+(?:اسم\s+)?|كلمة\s+|الكلمة\s*:?\s*|لفظ(?:ة)?\s+|مصطلح\s+|لقب\s+|مفردة\s+|المفردة\s*:?\s*)$")
_STRONG_INTRO = re.compile(rf"(?P<cue>(?:اسمي|إسمي|معك|معكم|اخوكم|أخوكم|اختكم|أختكم|مع\s+حضراتكم|مع\s+تحياتي|تحياتي|مع\s+التحية|مع\s+خالص\s+التحية|المخلص|الموقّع)\s*[:،,-]?\s*)(?P<n>{TOK}(?:\s+{TOK}){{0,3}})")
_ANA = re.compile(rf"(?<![؀-ۿ])(?P<cue>(?:أنا|انا|اني|إني)\s+)(?P<n>{TOK}(?:\s+{TOK}){{0,3}})")
_KIN = re.compile(rf"(?<![؀-ۿ])(?P<cue>(?:جدتي|جدي|امي|أمي|اختي|أختي|اخي|أخي|عمي|عمتي|خالي|خالتي|زوجتي|زوجي|ابني|بنتي|والدي|والدتي|ابوي|أبوي|أخوي|صديقي|صديقتي|زميلي|زميلتي|جاري|جارتي)\s+)(?P<n>{TOK})")
_NASAB_LINKS = re.compile(rf"(?P<n>{TOK}(?:\s+(?:بنت|بن|ابن)\s+{TOK}){{2,5}})")
_KIN_NASAB = re.compile(rf"(?P<cue>(?:جدتي|جدي|امي|أمي|اختي|أختي|عمتي|خالتي|والدتي|ابوي|أبوي)\s+)(?P<n>{TOK}\s+(?:بنت|بن|ابن)\s+{TOK}(?:\s+(?:بنت|بن|ابن)\s+{TOK}){{0,4}})")
_ID_GLUED = re.compile(rf"\[id removed\]\s*(?P<n>{TOK}(?:\s+{TOK}){{0,3}})")
_LATIN_IDENT = re.compile(r"(?i)\b(?P<cue>my\s+name\s+is|i\s+am|i'm|name\s*:|regards|sincerely|best|thanks|mr\.?|mrs\.?|ms\.?|dr\.?)\s*,?\s+(?P<n>[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})")
_LATIN_NEXT_TO_CONTACT = re.compile(r"(?P<n>[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\s*(?=\[(?:email|link|handle|id) removed\])")
_FORM_PREFIX = re.compile(r"^\s*(?:الجوال\s*:\s*(?:\[[^\]]*\]|null|\d+)\s*,?\s*نص\s+الرسالة\s*:|محتوي\s+الرسالة\s*:|نص\s+الرسالة\s*:)\s*")
_FORM_PHONE_TAIL = re.compile(r"\s*(?:رقم\s+)?الجوال\s*:?\s*(?:\[[^\]]*\]|null|\+?\d[\d\s().-]*)?(?=\s|$)")
_FORM_PHONE_EN = re.compile(r"(?i)\s*(?:\b(?:test\s+)?phone\s+number\b|\btest\s+phone\b)\s*:?\s*(?:\[[^\]]*\]|null|\+?\d[\d\s().-]*)?")
_SENSITIVE = re.compile(r"(?:الإسكان\s+التنموي|مستحق(?:ه|ة)?\s+(?:منزل|سكن|اعانة|إعانة)|إعانة|الضمان\s+الاجتماعي|راتبي|ديوني|طلاق|انفصل\s+عنها|انفصلت|"
                        r"زوجته\s+التي|سجن|محكمة|قضية\s+ضد|سرطان|مرضي|مريض(?:ة)?\s+بـ)")
_RESIDUAL = (_EMAIL, _URL, _HANDLE, _LONG_DIGITS, _ID_WITH_CUE)


def _protected(text: str, start: int) -> bool:
    return bool(_LEX_PREFIX.search(text[max(0, start - 24):start]))


def _conjunction_prefixed_predicate(raw: str) -> bool:
    """'واقترح' / 'وأتمنى' / 'والله' (conjunction + predicate) are not names; 'وليد' / 'وفاء' (a name that merely starts with waw) are."""
    return raw.startswith("و") and len(raw) > 2 and norm(raw[1:]) in _PREDICATES


_CONNECTORS = {"بن", "بنت", "ابن", "ال", "آل", "ابو", "أبو", "ام", "أم"}
_DOMAIN_WORDS = {norm(x) for x in "اضافه إضافة كلمه كلمة مفرده مفردة معني معنى تعديل تصحيح خاصيه خاصية ميزه ميزة اقتراح ملاحظه ملاحظة تعريف مثال المعجم القاموس الموقع التطبيق البحث".split()}


def _name_prefix_len(n: str, max_tokens: int = 4) -> int:
    """Characters of the leading part of `n` that are a name. The first token is a name unless it is a predicate; further tokens only continue the name when they
    look like a family name (article-prefixed, a nasab connector or the token after one, or a plain token that is not a verb-shaped / dictionary word). Stops at
    predicates, function words and conjunction-prefixed tokens, so 'اخوكم عبدالرحمن يقترح إضافة خاصية' masks only the name."""
    end, count, after_connector = 0, 0, False
    for m in re.finditer(TOK, n):
        raw, tok = m.group(0), norm(m.group(0))
        if count >= max_tokens or tok in _PREDICATES or _conjunction_prefixed_predicate(raw) or (count > 0 and raw.startswith("و")):
            break
        if count > 0:
            is_connector = tok in {norm(c) for c in _CONNECTORS}
            family_like = raw.startswith("ال") or after_connector or is_connector or (len(raw) >= 3 and raw[0] not in "يتنأا" and tok not in _DOMAIN_WORDS)
            if not family_like or tok in _DOMAIN_WORDS:
                break
            after_connector = is_connector
        end, count = m.end(), count + 1
    return end


def _sub_names(text: str, rx: re.Pattern, actions: Counter, key: str, *, trim=None, accept=None) -> str:
    out, offset = text, 0
    for m in list(rx.finditer(text)):
        if _protected(text, m.start("n")) or (accept is not None and not accept(m)):
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


def sanitize_text(text: str | None, *, names: bool = True) -> Sanitized:
    """Identifiers always; personal names when `names` (visitor-authored or model-echoed text). Returns blocked_reason instead of text when unsafe."""
    res = Sanitized(text=None)
    a = res.actions
    t = " ".join(str(text).split()) if text is not None else ""
    if not t.strip():
        res.blocked_reason = "empty"
        return res
    # already-applied placeholders (e.g. "[phone number removed]" from the earlier prompt-build redaction) must survive the form-boilerplate patterns
    held: list[str] = []

    def _hold(m: re.Match) -> str:
        held.append(m.group(0))
        return f"\x00{len(held) - 1}\x00"

    t = re.sub(r"\[[a-z ]+ removed\]", _hold, t)
    t2 = _FORM_PREFIX.sub("", t)
    t2 = _FORM_PHONE_TAIL.sub("", t2)
    t2 = _FORM_PHONE_EN.sub(" ", t2)
    if t2 != t:
        a["contact_form_boilerplate_removed"] += 1
    t = t2.strip()
    before = t
    t = redact_phone_numbers(t) or ""
    t = re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], t) if held else t
    held = []
    if t != before:
        a["phone"] += 1
    for rx, rep, key in ((_EMAIL, EMAIL, "email"), (_URL, LINK, "link"), (_HANDLE, HANDLE, "handle"), (_ID_WITH_CUE, IDENT, "id_with_cue"),
                         (_LONG_DIGITS, IDENT, "long_digits"), (_ADDRESS, ADDRESS, "address")):
        t, n = rx.subn(rep, t)
        if n:
            a[key] += n
    # (the shared _SIGNATURE detector is deliberately NOT used: it lists "الموقع" - "the website" - as a signature cue and would erase any short comment that
    #  starts with it. Signatures are handled by _STRONG_INTRO below.)
    t = " ".join(t.split())
    if names and _SENSITIVE.search(t):
        res.blocked_reason = "sensitive_personal_circumstance"
        return res
    if names:
        t = _sub_names(t, _KIN_NASAB, a, "kinship_name_with_nasab")
        t = _sub_names(t, _STRONG_INTRO, a, "self_introduction_or_signature_name", trim=_name_prefix_len)
        t = _sub_names(t, _ANA, a, "first_person_name", trim=_name_prefix_len,
                       accept=lambda m: len(m.group("n").split()[0]) >= 2 and norm(m.group("n").split()[0]) not in _PREDICATES and not _conjunction_prefixed_predicate(m.group("n").split()[0]))
        t = _sub_names(t, _KIN, a, "kinship_name", accept=lambda m: norm(m.group("n")) not in _PREDICATES and not _conjunction_prefixed_predicate(m.group("n")))
        t = _sub_names(t, _NASAB_LINKS, a, "multi_link_genealogy")
        t, n = _ID_GLUED.subn(lambda m: IDENT + " " + NAME, t)
        a["name_glued_to_identifier"] += n
        t, n = _LATIN_IDENT.subn(lambda m: m.group(0).replace(m.group("n"), NAME), t)
        a["latin_name_after_identity_cue"] += n
        t, n = _LATIN_NEXT_TO_CONTACT.subn(NAME, t)
        a["latin_name_next_to_contact"] += n
        t = " ".join(t.split())
        # a nasab fragment right after a mask is still part of the identity ("[name removed] بن X")
        t = re.sub(rf"{re.escape(NAME)}(?:\s+(?:بنت|بن|ابن)\s+{TOK})+", NAME, t)
    # fail closed on anything that survived masking
    probe = re.sub(r"\[[a-z ]+ removed\]", "", t)
    if any(rx.search(probe) for rx in _RESIDUAL) or "@" in probe:
        res.blocked_reason = "residual_identifier"
        return res
    if re.search(r"\[id removed\][؀-ۿA-Za-z]", t):
        res.blocked_reason = "name_glued_to_identifier"
        return res
    if len(re.findall(r"[؀-ۿA-Za-z]", re.sub(r"\[[a-z ]+ removed\]", "", t))) < 2:
        res.blocked_reason = "only_identity_or_empty_content"
        return res
    res.text = t
    return res


# ------------------------------------------------------------------------------------------------- prompt-level boundary
_BLOCK = re.compile(
    r"(?P<head>--- (?P<name>UNTRUSTED VISITOR COMMENT|IDENTIFYING CONTEXT|STYLE REFERENCE EXAMPLES|PREVIOUS DRAFT|OFFICIAL KNOWLEDGE)[^\n]*---\n)"
    r"(?P<body>.*?)(?P<tail>\n--- END (?P=name)[^\n]*---)",
    re.S,
)
_FULL_SANITISE = {"UNTRUSTED VISITOR COMMENT", "IDENTIFYING CONTEXT", "STYLE REFERENCE EXAMPLES", "PREVIOUS DRAFT"}


def guard_prompt(prompt: str) -> str:
    """The final egress check. Returns the prompt with every variable block sanitised, or raises OutboundPrivacyError (no request is made).
    Idempotent: sanitising an already sanitised prompt changes nothing."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise OutboundPrivacyError("empty_prompt")
    seen = set()

    def repl(m: re.Match) -> str:
        name, body = m.group("name"), m.group("body")
        seen.add(name)
        if name in _FULL_SANITISE:
            if not body.strip() or body.strip().startswith("(none"):
                return m.group(0)
            r = sanitize_text(body, names=True)
            if r.text is None:
                raise OutboundPrivacyError(f"{name.lower().replace(' ', '_')}:{r.blocked_reason}")
            return m.group("head") + r.text + m.group("tail")
        r = sanitize_text(body, names=False)                 # approved knowledge: identifiers only, lexical content untouched
        if r.text is None and r.blocked_reason != "empty":
            raise OutboundPrivacyError(f"official_knowledge:{r.blocked_reason}")
        return m.group("head") + (r.text or body) + m.group("tail")

    out = _BLOCK.sub(repl, prompt)
    if "UNTRUSTED VISITOR COMMENT" not in seen:
        raise OutboundPrivacyError("visitor_block_missing")        # cannot verify what is being sent
    # whole-prompt assertion: outside the approved-knowledge block and the single approved URL no identifier may remain
    probe = _BLOCK.sub(lambda m: "" if m.group("name") == "OFFICIAL KNOWLEDGE" else m.group(0), out)
    for u in APPROVED_URLS:
        probe = probe.replace(u, "")
    probe = re.sub(r"\[[a-z ]+ removed\]", "", probe)
    if any(rx.search(probe) for rx in (_EMAIL, _URL, _HANDLE, _LONG_DIGITS)):
        raise OutboundPrivacyError("residual_identifier_in_prompt")
    return out
