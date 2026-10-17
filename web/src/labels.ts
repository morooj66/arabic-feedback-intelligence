export const STATUS_LABEL: Record<string, string> = {
  DRAFT_READY: "بانتظار المراجعة",
  REVIEW_ONLY: "بانتظار المراجعة",
  IN_REVIEW: "قيد المراجعة",
  APPROVED: "تمت الموافقة",
  EDITED_AND_APPROVED: "تمت الموافقة",
  NO_REPLY: "لا يحتاج إلى رد",
  // Model suggestion only — pending human review, distinct from the final NO_REPLY decision.
  NO_RESPONSE_RECOMMENDED: "مقترح: لا يحتاج إلى رد (بانتظار القرار)",
  SENDING: "جاري الإرسال",
  SENT: "تم الإرسال",
  SEND_FAILED: "فشل الإرسال",
};

// Shared status → badge-color mapping (visual reconciliation milestone)
// — used by every table/list that shows review_status, so the same
// status is never given a different color on different pages. Green =
// resolved/pending-default, orange = actively in review, red = a real
// failure needing attention. Never color-only: STATUS_LABEL text is
// always rendered alongside this class.
export function statusBadgeClass(status: string): string {
  if (status === "IN_REVIEW") return "badge badge-review";
  if (status === "APPROVED" || status === "EDITED_AND_APPROVED" || status === "SENT") return "badge badge-ok";
  if (status === "NO_REPLY") return "badge badge-no";
  if (status === "SEND_FAILED") return "badge badge-fail";
  return "badge badge-pending";
}

export const INTENT_LABEL: Record<string, string> = {
  THANKS_PRAISE: "شكر وثناء",
  COMPLAINT: "شكوى",
  SUGGESTION: "اقتراح",
  INQUIRY: "استفسار",
  SPAM: "غير ذي صلة",
};

export const TOPIC_LABEL: Record<string, string> = {
  GENERAL: "عام",
  TECHNICAL_ISSUE: "مشكلة تقنية",
  WORD_OF_YEAR_NOMINATION: "ترشيح كلمة العام",
  OFF_TOPIC_PERSONAL: "خارج الموضوع",
  WORD_SUBMISSION: "اقتراح مفردة",
  ENTRY_CORRECTION: "تصحيح مدخل",
  DICTIONARY_SUGGESTION: "اقتراح معجمي",
};

export const SENTIMENT_LABEL: Record<string, string> = {
  POSITIVE: "إيجابي",
  NEGATIVE: "سلبي",
  NEUTRAL: "محايد",
};

export const PRIORITY_LABEL: Record<number, string> = {
  1: "عاجل",
  2: "متوسط",
  3: "روتيني",
};

export const ROLE_LABEL: Record<string, string> = {
  REVIEWER: "مراجع",
  ADMIN: "مسؤول النظام",
};

// The "final decision" a reviewer recorded — used by Completed Reviews'
// decision filter. Matches config/settings.py::WORKFLOW_REVIEWER_ACTIONS
// exactly (the operational vocabulary, not the legacy six-action one).
export const REVIEWER_ACTION_LABEL: Record<string, string> = {
  SEND_AS_IS: "اعتماد كما هو",
  EDIT_AND_SEND: "تعديل واعتماد",
  NO_RESPONSE_NEEDED: "لا يحتاج رد",
};

export const NO_RESPONSE_REASONS = [
  "لا يتطلب ردًا",
  "محتوى غير ذي صلة",
  "تكرار",
  "أخرى",
];

export const MAX_RESPONSE_CHARS = 600;

export function outboundBanner(status: string | null, errorCode: string | null, sent: boolean): string | null {
  if (sent || status === "SENT") return "تم الإرسال";
  if (errorCode === "CONNECTOR_NOT_CONFIGURED" || status === "FAILED") {
    return "تم اعتماد الرد — بانتظار تكامل الإرسال";
  }
  if (status === "PENDING" || status === "SENDING") {
    return "تم اعتماد الرد — بانتظار تكامل الإرسال";
  }
  return null;
}

// Reviewer-facing notes about a classification. Display-only: they never carry a confidence score, never change a label and never affect routing.
// Unknown codes fall back to one generic sentence - the raw code is never shown.
export const CONTEXT_CONFLICT_FALLBACK = "تنبيه آلي: قد لا يطابق التصنيف المعروض نص التعليق.";
export const CONTEXT_CONFLICT_LABEL: Record<string, string> = {
  "intent:calm_correction_predicted_complaint": "صُنّف التعليق كشكوى، لكن صياغته تبدو تصحيحًا هادئًا دون نبرة استياء.",
  "intent:dissatisfaction_not_predicted_complaint": "تظهر في التعليق نبرة استياء، ولم يُصنَّف كشكوى.",
  "intent:question_not_predicted_inquiry": "يتضمن التعليق سؤالًا، ولم يُصنَّف كاستفسار.",
  "intent:praise_not_predicted_thanks": "يتضمن التعليق شكرًا أو ثناءً، ولم يُصنَّف كذلك.",
  "intent:thanks_with_substantive_content": "صُنّف التعليق كشكر، لكنه يتضمن أيضًا طلبًا أو سؤالًا أو ملاحظة تستحق الاهتمام.",
  "topic:platform_failure_not_technical_topic": "يصف التعليق عطلًا في الموقع أو التطبيق، ولم يُصنَّف موضوعه مشكلة تقنية.",
  "topic:entry_error_report_not_entry_correction": "يبدو أن التعليق يبلّغ عن خطأ في مدخل، ولم يُصنَّف موضوعه تصحيح مدخل.",
  "topic:platform_feature_idea_not_dictionary_suggestion": "يقترح التعليق ميزة للمنصة، ولم يُصنَّف موضوعه اقتراحًا للمعجم.",
  "topic:dictionary_request_predicted_off_topic": "يتضمن التعليق طلبًا أو سؤالًا عن المعجم، لكنه صُنّف خارج الموضوع.",
  "sentiment:negative_experience_not_negative": "تظهر في التعليق نبرة سلبية، ولم تُصنَّف المشاعر سلبية.",
  "sentiment:calm_content_predicted_negative": "صُنّفت المشاعر سلبية، لكن صياغة التعليق تبدو هادئة.",
  "sentiment:negated_negative_predicted_negative": "يتضمن التعليق نفيًا يعكس معنى الكلمة، فقد لا تطابق المشاعر المصنفة نبرته.",
  "sentiment:negated_positive_predicted_positive": "يتضمن التعليق نفيًا يعكس معنى الكلمة، فقد لا تطابق المشاعر المصنفة نبرته.",
  "sentiment:praise_with_suggestion_predicted_negative": "يجمع التعليق بين ثناء واقتراح، وصُنّفت مشاعره سلبية.",
};

// Why no draft was generated, for the reason codes the eligibility step actually emits; anything else shows nothing.
export const NO_DRAFT_REASON_LABEL: Record<string, string> = {
  uncertain_exclusion_requires_review: "التصنيف الآلي يرجّح استبعاد التعليق لكنه غير مؤكد، لذا لم تُولَّد مسودة ويلزم قرارك.",
  uncertain_exclusion_review: "التصنيف الآلي يرجّح استبعاد التعليق لكنه غير مؤكد، لذا لم تُولَّد مسودة ويلزم قرارك.",
  factual_question_without_grounding: "يطلب التعليق معلومة لغوية ولا يتوفر مصدر رسمي معتمد للإجابة، لذا لم تُولَّد مسودة.",
};
