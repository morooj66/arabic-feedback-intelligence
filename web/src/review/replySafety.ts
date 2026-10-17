/** Arabic copy for reply-safety findings. Backend finding codes are NEVER shown to users; unknown codes fall back to a generic sentence. */

/** Claims a permitted reviewer may confirm (the server decides; this only supplies wording). */
const ATTESTABLE_TEXT: Record<string, string> = {
  unsupported_definition: "تعريف أو معلومة معجمية عن كلمة لا يمكن التحقق منها من مصدر معتمد.",
  unsupported_correctness_claim: "حكم بصحة مدخل أو ادعاء أو بخطئه لا يمكن التحقق منه من مصدر معتمد.",
  unsupported_number: "رقم أو إحصاء غير وارد في تعليق الزائر ولا في مصدر معتمد.",
};

/** Restrictions that confirmation can never lift. */
const BLOCKING_TEXT: Record<string, string> = {
  unapproved_url: "الرد يحتوي على رابط غير معتمد. احذف الرابط، أو استخدم الرابط الرسمي المعتمد فقط.",
  unsupported_commitment: "الرد يتضمن وعدًا أو التزامًا أو ادعاء إجراء (مثل: سنقوم، تم، قريبًا، الفريق المختص). أعد صياغته دون وعود.",
  unsupported_operational_claim: "الرد يتضمن وعدًا أو التزامًا أو ادعاء إجراء (مثل: سنقوم، تم، قريبًا، الفريق المختص). أعد صياغته دون وعود.",
  unsupported_policy_claim: "الرد يذكر سياسة أو إجراءً أو مدة أو قناة تواصل غير معتمدة.",
  internal_leak: "الرد يتضمن معلومات داخلية للنظام لا يجوز إرسالها.",
  pii_in_draft: "الرد يتضمن بيانات شخصية (مثل رقم هاتف).",
  empty_output: "الرد فارغ.",
  excessive_length: "الرد أطول من الحد المسموح.",
};

const GENERIC = "لم يجتز الرد فحص السلامة التلقائي.";

export function attestableText(code: string): string {
  return ATTESTABLE_TEXT[code] || "معلومة واقعية لا يمكن التحقق منها من مصدر معتمد.";
}

export function blockingText(code: string): string {
  return BLOCKING_TEXT[code] || GENERIC;
}

/** De-duplicated Arabic sentences for a list of blocking codes (several codes can share one sentence). */
export function blockingTexts(codes: string[]): string[] {
  return Array.from(new Set(codes.map(blockingText)));
}

export function sameCodes(a: string[], b: string[]): boolean {
  return a.length === b.length && [...a].sort().join("|") === [...b].sort().join("|");
}

export const SAFETY_COPY = {
  attestTitle: "يتضمن الرد معلومات لا يمكن التحقق منها من مصدر معتمد",
  attestBody: "لا يتوفر للنظام مصدر معتمد للتحقق من العبارات التالية. إن كنت متأكدًا من صحتها فيمكنك تأكيدها، وسيُسجَّل تأكيدك باسمك. لن يُغيَّر نص ردّك.",
  attestCheckbox: "أؤكد صحة المعلومات المذكورة أعلاه وأتحمل مسؤولية إرسالها",
  attestSend: "تأكيد وإرسال",
  attestEdit: "تعديل الرد",
  stale: "تغيّرت نتائج الفحص منذ آخر مرة؛ راجع القائمة المحدّثة وأكّدها من جديد.",
  deniedTitle: "لا تملك صلاحية تأكيد هذه المعلومات",
  deniedBody: "لا يمكنك إرسال عبارات واقعية غير موثقة. عدّل الرد بحذفها، أو أحِل التعليق إلى مراجع لديه الصلاحية. لم يُرسل شيء، ونص ردّك محفوظ كما هو.",
  blockedTitle: "لا يمكن إرسال هذا الرد",
  blockedBody: "لم يُرسل شيء، ونص ردّك محفوظ كما هو. هذا القيد لا يمكن تجاوزه بالتأكيد؛ عدّل الرد ثم أعد المحاولة.",
  checking: "جاري التحقق من الرد…",
} as const;
