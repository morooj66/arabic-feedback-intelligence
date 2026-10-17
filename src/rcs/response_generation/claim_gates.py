"""Deterministic claim gates for Arabic replies (reply-safety hardening).

The original validator (`validate._UNSUPPORTED_ACTION_PATTERNS`) is a closed phrase list. These gates detect the same categories by
STRUCTURE on a normalised text (diacritics/tatweel removed, alef variants unified, Arabic-Indic digits mapped) so spelling and
inflection variants are caught, and they add the categories the phrase list never covered:

  unapproved_url               any URL / domain that is not the approved dictionary homepage or present in the official grounding
  unsupported_commitment       future / progressive / past-tense action claims (we will / are working on / have done / the specialised team will),
                               hand-off claims and timelines, unless the OFFICIAL KNOWLEDGE states the same wording
  unsupported_policy_claim     institutional rules / procedures / criteria / durations stated as fact without grounding
  unsupported_number           a number in the reply that is neither in the visitor's comment nor in the grounding
  unsupported_definition       a dictionary definition / etymology / grammar assertion not backed by grounding (the defining clause must
                               overlap the grounding); echoing the visitor's own words with an attribution ("ذكرتم أن ...") is allowed
  unsupported_assertion        (ungrounded mode only) a declarative sentence that is not thanks / apology / a request or question to the visitor /
                               a limitation statement / a receipt acknowledgement / an echo of the visitor's words - an allow-list over speech acts
  unsupported_correctness_claim  a statement that a disputed entry / the visitor's claim is correct or incorrect without grounding

A sentence that states a LIMITATION ("لا تتوفر لدينا معلومة موثوقة ...") is exempt from the policy/definition/correctness gates: saying that
information is unavailable is the required safe behaviour, not a claim.

Pure functions. No I/O, no model, no network. All gates fail closed when grounding is empty.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from config import settings

APPROVED_URLS = (settings.REFERENCE_HOME_URL,)

_EMPTY_GROUNDING = {"", "(none — do not invent facts)"}

_ALEF = str.maketrans("أإآٱ", "اااا")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_MATCH = str.maketrans("ةى", "هي")   # ta-marbuta / alef-maksura folded for MATCHING only; the reply text itself is never altered
def _rx(pattern: str) -> "re.Pattern[str]":
    """Compile a pattern written in natural spelling against the FOLDED text (ta-marbuta->ha, alef-maksura->ya) that normalize() produces."""
    return re.compile(pattern.translate(_MATCH))


_PUNCT = re.compile(r"[،؛؟,.;:!?()\[\]{}\"'«»“”\-–—_/\\|*]+")
_SENT_SPLIT = re.compile(r"[.!؟?؛;،,\n]+")


def normalize(text: Optional[str]) -> str:
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn").replace("ـ", "")
    return " ".join(s.translate(_ALEF).translate(_DIGITS).translate(_MATCH).lower().split())


def _tokens(norm: str) -> list[str]:
    return [t for t in _PUNCT.sub(" ", norm).split() if t]


def _has_grounding(grounding: Optional[str]) -> bool:
    return normalize(grounding) not in _EMPTY_GROUNDING


def _words(norm: str) -> str:
    """Normalised text with punctuation replaced by spaces, padded for whole-word phrase search."""
    return " " + " ".join(_tokens(norm)) + " "


def _contains(words_padded: str, phrase: str) -> bool:
    return (" " + normalize(phrase) + " ") in words_padded


# ------------------------------------------------------------------------------------------------------------ URL gate
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.(?:gov|com|org|net|edu|info|io|me|sa|app|co|ly)\b(?:/\S*)?",
    re.I,
)


def unapproved_url(draft: str, grounding: Optional[str]) -> bool:
    g = (grounding or "").lower() if _has_grounding(grounding) else ""
    for m in _URL_RE.finditer(draft):
        found = m.group(0).rstrip(".,;:!?)»”\"'").lower()
        if any(found.rstrip("/") == u.rstrip("/").lower() for u in APPROVED_URLS):
            continue
        if g and found in g:
            continue
        return True
    return False


# ------------------------------------------------------------------------------------------------------ commitment gate
# Future / progressive action stems, matched on the token after the prefix  س/سوف + (ي|ن|ت).
_ACTION_STEMS = (
    "قوم", "عمل", "طلع", "اطلع", "صل", "صلح", "حرص", "سعي", "سعى", "بذل", "ضمن", "تاكد", "تابع", "راجع", "عالج", "حل", "نظر", "درس", "ضيف", "اضيف", "ضاف",
    "حدث", "طور", "عدل", "صحح", "اعتمد", "ادرج", "ادخل", "رسل", "ارسل", "تواصل", "حول", "احيل", "رفع", "اتخذ", "اخذ", "ناقش", "بحث", "وفر",
    "لبي", "اعمل", "قبل", "رد ", "ابلغ", "اخطر", "نشر", "غير", "ضبط", "اصحح", "تحقق", "فحص", "رصد", "عرض", "يتم", "تتم", "فعل", "وصل", "اتصل",
    "نظم", "رتب", "اقر", "قر",
)
_FUTURE_PREFIX_ALLOWED_AFTER_SAWF = _rx(r"^[ينت]\w{2,}$")
_SEEN_NOUNS = {"سنه", "سنوات", "سنين", "سند", "سندات", "سنابل", "سنتي", "سنتر", "سنام", "سنان", "سنه", "سنوي", "سنويه", "سنده", "سنبله"}
_BENIGN_FUTURE = {"يسعدنا", "نسعد", "سنسعد", "سيسعدنا", "سنكون", "سيكون", "ستكون", "سنظل", "سيظل", "ستظل", "سنبقى", "سيبقى", "ستبقى", "ستجد", "ستجدون", "سترون", "ستلاحظ", "سنقدر", "سنرحب"}

_PROGRESSIVE_PHRASES = (
    "نعمل على", "نعمل حاليا", "نعمل الان", "نحن نعمل", "نقوم", "سنقوم", "سيقوم", "يقوم الفريق", "يعمل الفريق", "يعمل المختصون", "يعمل المختصين", "تعمل الجهات",
    "جاري العمل", "جار العمل", "جاري العمل على", "قيد المراجعه", "قيد الدراسه", "قيد المعالجه", "قيد التنفيذ", "قيد النظر", "تحت الدراسه",
    "تحت المراجعه", "تحت المعالجه", "نسعى الى", "نسعى ل", "نسعى جاهدين", "نبذل", "نحرص على", "حرصنا على", "نتعهد", "نعد", "نعدكم", "نؤكد لكم",
    "نلتزم", "نتعهد", "نكافئ", "نؤكد لكم", "نتابع", "نراجع", "ندرس", "ننظر", "ننظر فيه", "نعالج", "نصلح", "نصحح", "نعدل", "نضيف", "نحدث", "نطور", "نعتمد", "نرسل", "نتواصل", "نبلغ",
    "نحيل", "نحول", "نرفع", "نتخذ", "نأخذ", "ناخذ", "نضمن", "نتاكد", "سنضمن", "نوفر", "نقبل", "نعتزم", "ننوي", "في طريقنا",
)
_PAST_ACTIONS = (
    "تم", "تمت", "تمّ", "قمنا", "لقد قمنا", "اتخذنا", "احلنا", "رفعنا", "اضفنا", "صححنا", "عدلنا", "ارسلنا", "حولنا", "راجعنا", "تواصلنا", "اعتمدنا",
    "دوننا", "سجلنا", "وثقنا", "ابلغنا", "نقلنا", "عالجنا", "اصلحنا", "حللنا", "حدثنا", "اخذنا", "ادرجنا", "قبلنا", "درسنا", "نظرنا", "ناقشنا",
    "وجهنا", "اوصلنا", "حذفنا", "غيرنا", "نشرنا", "فعلنا", "ضبطنا", "فحصنا", "تحققنا", "اطلعنا", "رفعناها", "رفعناه", "بلغنا", "اعدنا", "اجرينا",
    "انجزنا", "ابلغنا", "اتممنا", "انهينا",
)
_HANDOFF_PHRASES = (
    "الفريق المختص", "الفريق المعني", "الجهات المختصه", "الجهه المختصه", "الجهات المعنيه", "الجهه المعنيه", "الاداره المعنيه", "الاداره المختصه",
    "القسم المختص", "المختصين", "المختصون", "المسؤولين", "المسؤولون", "الادارة العليا", "اللجنه", "لجنه مختصه",
    "ستحال", "سيحال", "سنحيل", "احالة", "رفعها الى", "رفعه الى", "الفريق التقني", "فريق الدعم",
)
_TIMELINES = (
    "قريبا", "في اقرب وقت", "باقرب وقت", "في اقرب فرصه", "باقرب فرصه", "لاحقا", "في التحديث القادم", "التحديث القادم", "الاصدار القادم",
    "النسخه القادمه", "في القريب العاجل", "عما قريب", "مستقبلا", "بشكل عاجل", "على وجه السرعه", "فورا", "في اقرب الاوقات", "خلال",
    "بعد ايام", "الاسبوع القادم", "الشهر القادم", "في الايام القادمه", "في المستقبل القريب",
)
# Phrases that contain a flagged substring but are NOT commitments.
_COMMIT_SAFE = ("خلال تواصلكم", "خلال استخدامكم", "خلال زيارتكم", "خلال بحثكم", "من خلال")


def _future_commitments(words_padded: str, tokens: list[str]) -> Optional[str]:
    for i, t in enumerate(tokens):
        if t in _BENIGN_FUTURE:
            continue
        if t == "سوف" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt not in _BENIGN_FUTURE and _FUTURE_PREFIX_ALLOWED_AFTER_SAWF.match(nxt):
                return f"سوف {nxt}"
            continue
        if len(t) >= 4 and t.startswith("سن") and t not in _SEEN_NOUNS:
            return t                                       # "سن..." = first-person-plural future ("we will ..."): always a commitment
        if len(t) >= 5 and t[0] == "س" and t[1] in "ينت":
            rest = t[2:]
            if any(rest.startswith(stem.strip()) for stem in _ACTION_STEMS if stem.strip()):
                return t
    return None


def _first_phrase(words_padded: str, phrases: tuple[str, ...]) -> Optional[str]:
    for p in phrases:
        if _contains(words_padded, p):
            return p
    return None


def _commitment_phrase(words_padded: str, tokens: list[str]) -> Optional[str]:
    for safe in _COMMIT_SAFE:
        words_padded = words_padded.replace(" " + normalize(safe) + " ", " ")
    fut = _future_commitments(words_padded, tokens)
    if fut:
        return fut
    # "سيتم" / "ستتم" / "يتم" + anything: passive institutional action.
    for t in tokens:
        if t in ("سيتم", "ستتم", "تم", "تمت", "سيتم،"):
            return t
    for group in (_PROGRESSIVE_PHRASES, _PAST_ACTIONS, _HANDOFF_PHRASES, _TIMELINES):
        hit = _first_phrase(words_padded, group)
        if hit:
            return hit
    return None


_DELIVERY = _rx(r"(?:وصل|وصلت|وصلنا|وصلتنا|ارسل|ارسلت|ارسلنا|رفع|رفعت|رفعنا|احيل|احلنا|نقل|نقلت|نقلنا)\w*\s+(?:\S+\s+){0,4}?(?:الي|الى|لدي|عند)\s+(?:ال)?(?:فريق|اداره|قسم|مختص|جهه|جهات|مسؤول|لجنه|معجم|مشرف)")


def unsupported_commitment(draft: str, grounding: Optional[str]) -> bool:
    norm = normalize(draft)
    hit = _commitment_phrase(_words(norm), _tokens(norm))
    if hit is None:
        m = _DELIVERY.search(" ".join(_tokens(norm)))
        hit = m.group(0) if m else None
    if hit is None:
        return False
    if _has_grounding(grounding) and (" " + normalize(hit) + " ") in _words(normalize(grounding)):
        return False
    return True


# ------------------------------------------------------------------------------------------ limitation (exempt) sentences
_LIMITATION = (
    "لا تتوفر", "لا يتوفر", "لا تتوافر", "لا يتوافر", "لا نملك", "ليس لدينا", "ليست لدينا", "لا تتاح", "لا يتاح", "غير متوفر", "غير متوفره", "غير متاح",
    "لا نستطيع", "لا يمكننا", "لا يمكن", "لا يسعنا", "غير قادرين", "لا نتمكن", "تعذر علينا", "يتعذر علينا", "لا نتوفر", "لا تتوفر لدينا", "لا نمتلك", "لا تتيح لنا",
)


def _is_limitation(sentence_norm: str) -> bool:
    w = _words(sentence_norm).replace(" ولا ", " لا ")
    return any(_contains(w, p) for p in _LIMITATION)


def _sentences(draft: str) -> list[str]:
    return [normalize(s) for s in _SENT_SPLIT.split(draft) if s.strip()]


# --------------------------------------------------------------------------------------------------------- policy gate
_POLICY_TERMS = (
    "سياسه", "سياسات", "سياسة المعجم", "اجراءات", "الاجراءات", "ضوابط", "الضوابط", "معايير", "المعايير",
    "شروط", "الشروط", "لوائح", "اللوائح", "انظمه", "الانظمه", "مراحل", "مرحله", "تخضع", "يخضع", "يشترط", "تشترط", "يتطلب", "تتطلب", "يستغرق", "تستغرق",
    "مده", "المده", "اولويه", "الاولويه", "ساعات الدوام", "اوقات الدوام", "يقبل المعجم", "لا يقبل المعجم", "تقبل المنصه", "يقبل الموقع", "تقبل المنصة",
    "يتم قبول", "نقبل الاقتراحات", "نستقبل الاقتراحات", "تستقبل المنصه", "يستقبل المعجم", "لجنه", "لجان", "اللجنه", "اعتماد الكلمات", "اعتماد الكلمه",
    "نظام المراجعه", "تتم مراجعه", "تتم المراجعه", "تراجع الاقتراحات", "تدقق", "تُدقق", "تقييم الاقتراحات", 
    "القنوات الرسميه", "قنوات رسميه", "رسميه",
    "الدوام", "مجانا", "مجاني", "مدفوع", "رسوم", "حق الملكيه", "حقوق النشر", "ترخيص", "رخصه", "رخصة",
)
_DURATION_RE = _rx(
    r"(?:\d+|يوم|يومين|يومان|ثلاث\w*|اربع\w*|خمس\w*|ست\w*|سبع\w*|ثمان\w*|تسع\w*|عشر\w*|اسبوع\w*|شهر\w*)\s*"
    r"(?:ايام|يوم|اسابيع|اسبوع|اسبوعين|اسبوعان|شهر|اشهر|شهور|ساعات|ساعه|ساعة|دقائق|دقيقه|سنوات|سنه|عاما|اعوام|ايام عمل)"
)


_DIRECTING = _rx(
    r"(?:مراجعه|زياره|استخدام|الاطلاع علي|الاتصال|اتصلوا|التواصل|راجعوا|راسلوا|مراسله|ابحثوا في|ارجعوا الي)\s+(?:\S+\s+){0,3}?"
    r"(?:قسم|صفحه|الصفحه|الاسئله الشائعه|المساعده|الدعم|الهاتف|البريد|الايميل|الواتساب|نموذج|رقم|خدمه العملاء|مركز|الاتصال|الموقع الرسمي)"
    r"|(?:ستجد|ستجدون|ستجدين)\b|عبر\s+(?:البريد|الهاتف|الواتساب|الايميل|الاتصال|تطبيق)"
)


def unsupported_policy_claim(draft: str, grounding: Optional[str], *, check_directing: bool = True) -> bool:
    g_words = _words(normalize(grounding)) if _has_grounding(grounding) else ""
    if check_directing and not g_words:
        for sent in _sentences(draft):
            if not _is_limitation(sent) and _DIRECTING.search(" ".join(_tokens(sent))):
                return True                                # points the visitor to a channel / section / page nothing in the system can vouch for
    for sent in _sentences(draft):
        if _is_limitation(sent):
            continue
        w = _words(sent)
        for term in _POLICY_TERMS:
            if _contains(w, term):
                if g_words and _contains(g_words, term):
                    continue
                return True
        if _DURATION_RE.search(sent) and not (g_words and _DURATION_RE.search(g_words)):
            return True
    return False


def unsupported_number(draft: str, comment: Optional[str], grounding: Optional[str]) -> bool:
    text = _URL_RE.sub(" ", draft)
    nums = re.findall(r"\d+(?:[.,]\d+)?", normalize(text))
    if not nums:
        return False
    allowed = set(re.findall(r"\d+(?:[.,]\d+)?", normalize(comment) + " " + (normalize(grounding) if _has_grounding(grounding) else "")))
    return any(n not in allowed for n in nums)


# ----------------------------------------------------------------------------------------------------- definition gate
_STOP = {"في", "من", "على", "الى", "عن", "ان", "او", "ثم", "هذا", "هذه", "ذلك", "التي", "الذي", "هو", "هي", "ما", "لا", "كل", "قد", "مع", "او", "بعد", "قبل"}
_DEF_TRIGGERS = _rx(
    r"(?:^|\s)(?:تعني|يعني|معناها|معناه|المقصود(?:\s+ب\S+)?|تدل على|يدل على|تطلق على|يطلق على|تستخدم ل\S*|يستخدم ل\S*|تستعمل ل?\S*|يستعمل ل?\S*|"
    r"اصلها|اصله|مشتقه من|مشتق من|جذرها|جذره|ضدها|ضده|مرادفها|مرادفه)\s+(.+)$"
)
_DEF_FORMS = (
    _rx(r"(?:^|\s)معنى\s+(?:كلمه\s+|لفظ\S*\s+|الكلمه\s+)?\S+(?:\s+\S+){0,2}?\s+(?:هو|هي)\s+(.+)$"),
    _rx(r"(?:^|\s)تعريف\s+(?:كلمه\s+)?\S+(?:\s+\S+){0,2}?\s+(?:هو|هي)\s+(.+)$"),
    _rx(r"(?:^|\s)(?:كلمه|لفظ\S*)\s+\S+\s+(?:هي|هو)\s+(.+)$"),
    _rx(r"(?:^|\s)(?:هي|هو)\s+((?:اسم|فعل|صفه|مصدر|جمع|حرف|ظرف)\b.*)$"),
    _rx(r"(?:^|\s)(?:اصل|جذر|مرادف|ضد|اشتقاق)\s+(?:كلمه|لفظ\S*)\s+\S+\s+(.+)$"),
    _rx(r"(?:^|\s)(?:تدل|يدل)\s+\S+\s+على\s+(.+)$"),
)
_INTERROGATIVES = {"ماذا", "هل", "كيف", "لماذا", "اي", "ايه", "اين", "متى"}
_ATTRIBUTION = ("تذكرون", "ذكرتم", "تقولون", "قلتم", "ترون", "تشيرون", "اشرتم", "تفيدون", "افدتم", "ورد في تعليقكم", "بحسب ما ذكرتم", "تقصدون", "تسالون",
                "تستفسرون", "تطلبون", "طلبتم", "ما ذكرتموه", "كما ذكرتم", "حسب ما ذكرتم", "تعليقكم")


_NOMINAL_PREFIX = {"توضيح", "بيان", "شرح", "ايضاح", "تفسير", "ذكر", "ايراد", "معرفه", "اضافه", "تعريف", "طلب", "ذكره", "ذكرها", "وتوضيح", "وبيان", "وشرح", "وايضاح", "وذكر", "وايراد", "ومعرفه", "واضافه", "لتوضيح", "لبيان", "لشرح", "بتوضيح", "ببيان", "بشرح", "ان", "عن", "حول", "في", "من", "على"}


def _nominal_reference(sent: str, start: int) -> bool:
    before = sent[:start].split()
    return bool(before) and before[-1] in _NOMINAL_PREFIX


_SUFFIXES = ("كم", "هم", "ها", "نا", "ات", "ون", "ين", "ه", "ي")


def _light_stem(t: str) -> str:
    """Very light Arabic stemming for OVERLAP comparison only (never for matching claims): drop a conjunction / preposition prefix, the
    article, and a pronoun / plural suffix, so 'واستفساركم' and 'استفسار' compare equal."""
    if len(t) > 4 and t[0] in "وف":
        t = t[1:]
    if len(t) > 4 and t[0] in "بلك":
        t = t[1:]
    if t.startswith("ال") and len(t) > 4:
        t = t[2:]
    for suf in _SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[: -len(suf)]
            break
    return t


def _content_tokens(s: str) -> set[str]:
    out = set()
    for t in _tokens(s):
        t = _light_stem(t)
        if len(t) >= 3 and t not in _STOP:
            out.add(t)
    return out


def unsupported_definition(draft: str, comment: Optional[str], grounding: Optional[str]) -> bool:
    g_tokens = _content_tokens(normalize(grounding)) if _has_grounding(grounding) else set()
    c_tokens = _content_tokens(normalize(comment))
    for sent in (" ".join(_tokens(x)) for x in _sentences(draft)):
        if _is_limitation(sent):
            continue
        clauses = []
        if any(q in _tokens(sent) for q in _INTERROGATIVES):
            continue                                       # a question to the visitor ("ماذا تعني بكلمة ...") is not an assertion
        m = _DEF_TRIGGERS.search(sent)
        if m and _nominal_reference(sent, m.start()):
            m = None                                       # "توضيح معناها" restates the visitor's own request; it asserts nothing
        if m:
            clause = m.group(1).strip()
            if not clause.startswith(("ان ", "انه", "انها", "انكم", "انني", "اننا")):   # "يعني أن ..." is an implication, not a definition
                clauses.append(clause)
        for pat in _DEF_FORMS:
            mm = pat.search(sent)
            if mm:
                clauses.append(mm.group(1).strip())
        for clause in clauses:
            toks = _content_tokens(clause)
            if not toks:
                continue
            attributed = any(a in sent for a in _ATTRIBUTION)
            if attributed and len(toks & c_tokens) / len(toks) >= 0.7:
                continue
            if g_tokens and len(toks & g_tokens) / len(toks) >= 0.6:
                continue
            return True
    return False


# ------------------------------------------------------------------------------------------------ correctness (dispute) gate
_VERDICT = r"(?:غير\s+)?(?:صحيح\w*|سليم\w*|دقيق\w*|خاطئ\w*|خطا|غلط|صائب\w*|مصيب\w*|وجيه\w*|مقبول\w*|مرفوض\w*)"
_CORRECTNESS = (
    _rx(r"(?:التعريف|المعنى|الشرح|الوارد|المدخل|الاملاء|الكتابه|الصياغه|الضبط|النطق|المعلومه|الكلمه|الاشتقاق|الجمع)\s+(?:الوارد\s+|المذكور\s+)?(?:\S+\s+){0,3}?" + _VERDICT),
    _rx(r"(?:ملاحظتكم|اقتراحكم|رايكم|ما ذكرتم(?:وه)?|ما ذكرتموه|تصويبكم|تعليقكم|طرحكم)\s+(?:هي\s+|هو\s+)?(?:\S+\s+){0,2}?" + _VERDICT),
    _rx(r"(?:الصواب|الصحيح)\s+(?:هو|هي|ان|ما)\b"),
    _rx(r"(?:ملاحظه|تنبيه|توضيح|اقتراح|مداخله|استدراك|تصويب|راي|تعليق|طرح)\s+(?:لغوي\w*\s+)?(?:دقيق|دقيقه|صحيح|صحيحه|وجيه|وجيهه|سديد|سديده|في محلها|في محله)\b"),
    _rx(r"(?:^|\s)(?:انتم\s+)?(?:محق|محقون|محقين|معكم الحق|الحق معكم|اصبتم|اصبت)(?:\s|$)"),
    _rx(r"(?:اكد|اكدنا|اثبت|اثبتنا|ثبت)\s+(?:\S+\s+){0,2}?(?:صحه|صحته|صحتها|خطا|دقه|دقته)"),
    _rx(r"لا\s+(?:يوجد|توجد)\s+(?:اي\s+)?خطا|ليس\s+(?:هناك|فيه)\s+(?:اي\s+)?خطا|لا\s+غبار"),
    _rx(r"(?:مدخلنا|تعريفنا|المعجم)\s+(?:\S+\s+){0,2}?(?:صحيح|سليم|دقيق)"),
)


# --- Unverified ADMISSION of an error (live rg-v14.2 pilot #2 "نعتذر عن الخطأ الوارد في تشكيل عنوان البطاقة", #3 "نعتذر عن عدم الدقة التي أشرتم إليها") ---------------
# An apology (or an admission verb) whose OBJECT is an error in the dictionary content, or a technical failure, concedes something nothing in the system has verified.
# It is the same family as a verdict ("the entry is wrong"), so it is reported as `unsupported_correctness_claim` (refinable once; attestable by a permitted reviewer).
# Narrow on purpose: it fires only on  <apology | admission verb> ... <error / failure noun> within one clause, with no hedge that qualifies THAT noun. Neutral
# acknowledgment ("نقدّر ملاحظتكم بشأن الخطأ"), empathy for the visitor's experience ("نعتذر عمّا واجهتموه من صعوبة", "نعتذر عن الإزعاج") and apologies about the visitor's own
# inability ("عدم تمكنكم من ...") are untouched. A hedge counts only when it directly qualifies the noun ("إن كان هناك خطأ", "أي خطأ", "الخطأ المحتمل"); a "قد"/"إذا"/"لو" elsewhere
# in the sentence exempts nothing.
_APOLOGY_TRIGGERS = {normalize(x) for x in ("نعتذر", "نأسف", "أعتذر", "يؤسفنا", "آسفون", "أسفنا", "نتأسف", "عذرا")}
_ADMISSION_TRIGGERS = {normalize(x) for x in ("نقر", "نعترف", "نؤكد", "تبين", "اتضح", "ثبت", "تأكدنا", "نسلم")}
_DICTIONARY_ERROR_NOUNS = tuple(normalize(x).split() for x in (
    "الخطأ", "خطأ", "الأخطاء", "أخطاء", "الغلط", "غلط", "عدم الدقة", "عدم دقة", "عدم صحة", "عدم الصحة", "القصور", "النقص", "الهفوة", "هفوة", "التقصير"))
_TECHNICAL_FAILURE_NOUNS = tuple(normalize(x).split() for x in (
    "تعطل", "التعطل", "عطل", "العطل", "الأعطال", "أعطال", "خلل", "الخلل", "توقف", "التوقف", "فشل", "الفشل", "بطء", "البطء", "انقطاع", "الانقطاع",
    "عدم عمل", "عدم اشتغال", "عدم ظهور", "عدم فتح", "المشكلة التقنية", "مشكلة تقنية"))
_ERROR_NOUNS = _DICTIONARY_ERROR_NOUNS + _TECHNICAL_FAILURE_NOUNS
_EXISTENCE_VERBS = {normalize(x) for x in ("كان", "يكون", "وقع", "يقع", "حدث", "يحدث", "ورد", "وجد", "يوجد", "تسبب", "كنا")}
_HEDGE_WORDS = {normalize(x) for x in ("اذا", "إذا", "لو", "لعل", "ربما", "أي", "اي")}
_HEDGE_AFTER_NOUN = {normalize(x) for x in ("محتمل", "المحتمل", "محتملة", "المحتملة", "مزعوم", "المزعوم", "مزعومة", "المزعومة", "مفترض", "المفترض")}
_WINDOW = 10


def _noun_at(toks: list[str], j: int) -> int:
    """Length (in tokens) of an error / failure noun starting at toks[j] (also behind a one-letter prefix: بالخطأ, للخطأ, والخطأ), else 0."""
    variants = [toks[j]]
    if len(toks[j]) > 3 and toks[j][0] in "وفبلك":
        variants.append(toks[j][1:])
    if toks[j].startswith("لل"):
        variants.append("ال" + toks[j][2:])
    for noun in _ERROR_NOUNS:
        first = noun[0]
        if first in variants and toks[j + 1: j + len(noun)] == noun[1:]:
            return len(noun)
    return 0


def _hedge_qualifies_noun(toks: list[str], j: int, n: int, trig: int, admission: bool) -> bool:
    wide = toks[max(trig + 1, j - 5): j]                                                          # "إذا كان في الشرح عدم دقة": a conditional clause may sit up to 5 words before the noun
    for k, w in enumerate(wide):
        if (w in _HEDGE_WORDS or w in ("ان", "قد")) and k + 1 < len(wide) and wide[k + 1] in _EXISTENCE_VERBS:
            return True
    before = toks[max(trig + 1, j - 3): j]
    for k, w in enumerate(before):
        if w in _HEDGE_WORDS:
            return True
        if w == "ان" and k + 1 < len(before) and before[k + 1] in _EXISTENCE_VERBS:              # "إن كان / إن وقع" (folded "إن" = "ان")
            return True
        if w == "قد" and any(x in _EXISTENCE_VERBS for x in before[k + 1:]):                     # "قد يكون / قد وقع" directly before the noun
            return True
    if j + n < len(toks) and toks[j + n] in _HEDGE_AFTER_NOUN:                                    # "الخطأ المحتمل"
        return True
    if admission:                                                                                 # a conditional admission verb: "إذا تبين وجود خطأ"
        return any(w in ("اذا", "لو", "عند", "حال") for w in toks[max(0, trig - 2): trig])
    return False


def unverified_admission_of_error(sentence_norm: str) -> bool:
    toks = _tokens(sentence_norm)
    for i, t in enumerate(toks):
        u = t[1:] if len(t) > 3 and t[0] in "وف" else t                                            # "ونعتذر" / "فنعتذر"
        admission = u in _ADMISSION_TRIGGERS
        if not (u in _APOLOGY_TRIGGERS or t in _APOLOGY_TRIGGERS or admission):
            continue
        j = i + 1
        while j < min(len(toks), i + 1 + _WINDOW):
            n = _noun_at(toks, j)
            if n:
                if not _hedge_qualifies_noun(toks, j, n, i, admission):
                    return True
                j += n
            else:
                j += 1
    return False


def unsupported_correctness_claim(draft: str, grounding: Optional[str]) -> bool:
    if _has_grounding(grounding):
        return False                     # the claim might be supported; the definition gate still checks content overlap
    for sent in _sentences(draft):
        if _is_limitation(sent):
            continue
        if any(p.search(sent) for p in _CORRECTNESS) or unverified_admission_of_error(sent):
            return True
    return False



# ---------------------------------------------------------------------------------------- assertion gate (ungrounded mode)
# With NO official grounding the only safe speech acts are: thanks / appreciation / blessing, apology and empathy, a request or question
# TO the visitor, a statement that information is unavailable, a receipt acknowledgement, and an echo of the visitor's own words.
# Every other declarative sentence asserts something about the institution, the platform or a word that nothing in the system can back.
# This is an allow-list over speech acts (not a phrase blacklist), so unseen wordings of invented facts are still caught.
_ACK_STEMS = (
    "شكر", "نشكر", "شاكر", "الشكر", "يشكر", "يسعد", "تسعد", "امتن", "نامل", "نرجو", "في حال رغبتكم", "في حال رغبتك", "ان رغبتم", "يسرنا", "نتطلع", "تقدير", "نقدر", "تقدر", "نثمن", "تثمن", "يسعدنا", "نسعد", "نرحب", "ترحيب", "نعتز", "جزاكم", "جزاك", "بارك", "الله", "وفقكم", "حفظكم",
    "اكرمكم", "اسعدكم", "عفوا", "اهلا", "مرحبا", "تحيه", "السلام", "نتمني", "نتطلع", "محل تقدير", "ندعو", "دعاءكم", "دعائكم", "الله يعطيكم", "الله يجزيكم",
    "الله يحفظكم", "كلمه طيبه", "كلماتكم", "تسعدنا", "يسرنا", "سعدنا",
)
_APOLOGY_EMPATHY_STEMS = ("نعتذر", "ناسف", "عذرا", "اسفين", "نتفهم", "نفهم", "تفهمنا", "نقدر ظروفكم", "ندرك", "نشعر بقلقكم")
_REQUEST_STEMS = ("نرجو", "يرجي", "ارجو", "نامل منكم", "تفضلوا", "زودونا", "زودنا", "وضحوا", "اذكروا", "ارسلوا", "نطلب منكم", "نود معرفه", "نحتاج الي", "اخبرونا", "اكتبوا لنا")
_PERMISSION_REQUEST = re.compile(r"(?:يمكنكم|بامكانكم|بوسعكم|ممكن)\s+(?:\S+\s+){0,2}?(?:تزويدنا|اخبارنا|ابلاغنا|توضيح|ذكر|ارسال|مشاركتنا|اطلاعنا|كتابه)")
_RECEIPT_RE = re.compile(r"(?:^|\s)و?وصل\w*\s+(?:ملاحظ|استفسار|سؤال|اقتراح|رسال|تعليق|مسال|طلب|مقترح)\w*")
_RECEIPT = ("وصلتنا", "وصلت رسالتكم", "وصلت ملاحظتكم", "وصلنا تعليقكم", "اطلعنا علي", "قراناها", "قرانا", "نشير الي", "نود الاشاره الي")


_PREAMBLE = ("بخصوص", "بشان", "حول", "فيما يخص", "فيما يتعلق", "بالنسبه ل", "تعقيبا علي", "ردا علي", "جوابا علي", "اشارة الي")


def _is_preamble_only(toks: list[str]) -> bool:
    """'بخصوص المدخل الذي أشرتم إليه' - a topic-introducing fragment with no predicate of its own."""
    if not toks or not any(toks[0] == normalize(p).split()[0] and " ".join(toks).startswith(normalize(p)) for p in _PREAMBLE):
        return False
    return not any(t.startswith(("سن", "سي", "ست", "نعمل", "نقوم", "يتم", "تم")) for t in toks)


def _continues_thanks(toks: list[str]) -> bool:
    """'... وعلى توضيحكم للفارق' / 'واهتمامكم بإثراء المحتوى' - a coordinated object of the previous (safe) thanks clause."""
    if not toks:
        return False
    first = toks[0]
    return first in ("وعلي", "وعن", "وكذلك") or (first.startswith("و") and first.endswith(("كم", "كما")) and len(first) > 4)


_ATTRIBUTION_NORM = tuple(normalize(a) for a in _ATTRIBUTION)


_MA_QUESTION_SECOND = {"هو", "هي", "هذا", "هذه", "ذا", "معني", "سبب", "الفرق", "الفارق", "المقصود", "المراد", "حكم", "اصل", "اسم"}


def _has_interrogative(toks: list[str]) -> bool:
    """Interrogative word anywhere ("هل", "كيف", ... also behind a waw: "وهل"), or a clause-initial "ما" + noun/pronoun ("ما الكلمة ...", "ما هو ...", "ما معنى ...").
    A bare "ما" elsewhere is a negator ("ما يفتح") and is NOT a question word."""
    if any(t in _INTERROGATIVES or (len(t) > 2 and t[0] == "و" and t[1:] in _INTERROGATIVES) for t in toks):
        return True
    return len(toks) >= 2 and toks[0] == "ما" and (toks[1].startswith("ال") or toks[1] in _MA_QUESTION_SECOND)


def _continues_question(toks: list[str]) -> bool:
    """The second half of a compound question: "هل تقصدون X، أم تريدون Y؟" / "..., أو في Y؟". Only the coordinators أم / أو qualify, and only directly after a clause
    that is itself a question; a declarative clause after a comma ("..., علماً أن المعجم يعتمد ...") is NOT a continuation and is still judged on its own."""
    return bool(toks) and toks[0] in ("ام", "او")


def _is_safe_speech_act(sent: str, comment_tokens: set[str]) -> bool:
    w = _words(sent)
    toks = _tokens(sent)
    if _is_preamble_only(toks):
        return True
    if "approvedurl" in toks:
        return True                                        # the one approved navigation URL (substituted by unsupported_assertion)
    if _is_limitation(sent):
        return True
    if _has_interrogative(toks):
        return True
    for stems in (_ACK_STEMS, _APOLOGY_EMPATHY_STEMS, _REQUEST_STEMS):
        for stem in stems:
            stem = normalize(stem)
            if " " in stem:
                if (" " + stem) in w:
                    return True
            elif any(t.startswith(stem) or t.startswith("و" + stem) or t.startswith("ل" + stem) or t.startswith("ب" + stem) for t in toks):
                return True
    if any(_contains(w, r) for r in _RECEIPT) or _RECEIPT_RE.search(" ".join(toks)):
        return True
    if any(a in " ".join(toks) for a in _ATTRIBUTION_NORM):
        ct0 = _content_tokens(sent)
        if ct0 and (len(ct0 & comment_tokens) >= 2 or len(ct0 & comment_tokens) / len(ct0) >= 0.3):
            return True                                    # restates what the visitor said (a verdict on it is caught by the correctness gate)
    if _PERMISSION_REQUEST.search(" ".join(toks)):
        return True
    ct = _content_tokens(sent)
    if len(ct) >= 2 and len(ct & comment_tokens) / len(ct) >= 0.6:
        return True                                        # echo of the visitor's own words
    return len(toks) <= 2                                  # fragments such as "وعليكم السلام" carry no claim


def unsupported_assertion(draft: str, comment: Optional[str], grounding: Optional[str]) -> bool:
    if _has_grounding(grounding):
        return False                                       # grounded replies are checked by overlap in the definition / policy gates
    ct = _content_tokens(normalize(comment))
    for u in APPROVED_URLS:
        draft = re.sub(re.escape(u), " APPROVEDURL ", draft, flags=re.I)
    prev_safe, prev_question = False, False
    for sent in _sentences(draft):
        s = " ".join(_tokens(sent))
        if not s:
            continue
        toks = s.split()
        is_question = _has_interrogative(toks)
        safe = _is_safe_speech_act(s, ct) or (prev_safe and _continues_thanks(toks)) or (prev_question and _continues_question(toks))
        if not safe:
            return True
        prev_safe, prev_question = safe, is_question or (prev_question and _continues_question(toks))
    return False

# ------------------------------------------------------------------------------------------------------------- driver
def first_failing_claim_gate(
    draft: str, *, comment_text: Optional[str], grounding: Optional[str], include_assertion_gate: bool = True
) -> Optional[tuple[str, str]]:
    """Returns (gate_name, reason_code) for the first failing claim gate, or None. Fixed order. `include_assertion_gate=False` is used for
    reviewer-written text: a human may state facts they know; the category gates (promises, policies, definitions, verdicts, URLs, numbers)
    still apply, the "ungrounded replies may only thank / apologise / ask" allow-list does not."""
    if unapproved_url(draft, grounding):
        return "unapproved_url", "unapproved_url"
    if unsupported_commitment(draft, grounding):
        return "unsupported_commitment", "unsupported_commitment"
    if unsupported_policy_claim(draft, grounding, check_directing=include_assertion_gate):
        return "unsupported_policy_claim", "unsupported_policy_claim"
    if unsupported_number(draft, comment_text, grounding):
        return "unsupported_number", "unsupported_number"
    if unsupported_definition(draft, comment_text, grounding):
        return "unsupported_definition", "unsupported_definition"
    if unsupported_correctness_claim(draft, grounding):
        return "unsupported_correctness_claim", "unsupported_correctness_claim"
    if include_assertion_gate and unsupported_assertion(draft, comment_text, grounding):
        return "unsupported_assertion", "unsupported_assertion"
    return None


def all_failing_claim_gates(
    draft: str, *, comment_text: Optional[str], grounding: Optional[str], include_assertion_gate: bool = True
) -> list[str]:
    """Every failing claim-gate reason code (not only the first), in the fixed gate order. Used to separate findings a reviewer may
    attest from findings that can never be attested."""
    out: list[str] = []
    if unapproved_url(draft, grounding):
        out.append("unapproved_url")
    if unsupported_commitment(draft, grounding):
        out.append("unsupported_commitment")
    if unsupported_policy_claim(draft, grounding, check_directing=include_assertion_gate):
        out.append("unsupported_policy_claim")
    if unsupported_number(draft, comment_text, grounding):
        out.append("unsupported_number")
    if unsupported_definition(draft, comment_text, grounding):
        out.append("unsupported_definition")
    if unsupported_correctness_claim(draft, grounding):
        out.append("unsupported_correctness_claim")
    if include_assertion_gate and unsupported_assertion(draft, comment_text, grounding):
        out.append("unsupported_assertion")
    return out


REASON_MESSAGES_AR = {
    "unapproved_url": "يحتوي الرد على رابط غير معتمد.",
    "unsupported_commitment": "يتضمن الرد وعدًا أو التزامًا أو إجراءً (مثل: سنقوم / جارٍ العمل / تم / الفريق المختص / قريبًا) لا يدعمه مصدر معتمد.",
    "unsupported_policy_claim": "يتضمن الرد سياسة أو إجراءً أو شرطًا أو مدة لا يدعمها مصدر معتمد.",
    "unsupported_number": "يتضمن الرد رقمًا غير وارد في تعليق الزائر أو في مصدر معتمد.",
    "unsupported_definition": "يتضمن الرد تعريفًا أو معلومة معجمية لا يدعمها مصدر معتمد.",
    "unsupported_correctness_claim": "يحكم الرد بصحة أو خطأ مدخل أو ادعاء، أو يعتذر عن خطأ أو عطل لم يُتحقق منه، دون مصدر معتمد.",
    "unsupported_assertion": "يتضمن الرد معلومة أو تقريرًا عن المعجم أو المنصة أو الكلمة لا يدعمه مصدر معتمد (يُسمح فقط بالشكر والاعتذار وطلب التوضيح وبيان عدم توفر المعلومة).",
    "unsupported_operational_claim": "يتضمن الرد ادعاء إجراء تشغيلي غير مدعوم.",
    "internal_leak": "يكشف الرد معلومات داخلية للنظام.",
    "pii_in_draft": "يحتوي الرد على بيانات شخصية.",
    "empty_output": "الرد فارغ.",
}
