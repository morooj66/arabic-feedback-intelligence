/** Fixed Arabic UI copy. Never interpolate raw service/provider exceptions. */
export const STATE_COPY = {
  loading: "جاري التحميل…",
  serviceUnavailable: "تعذر الاتصال بالخدمة حاليًا. يرجى المحاولة مرة أخرى بعد قليل.",
  permissionDenied: "غير مصرح لك بالوصول إلى هذه الصفحة.",
  assignedEmpty: { title: "لا توجد تعليقات مسندة لك حاليًا", description: "ستظهر هنا التعليقات الجديدة عند إسنادها لك." },
  reviewerCaughtUp: { title: "اكتملت جميع المراجعات", description: "لا توجد تعليقات أخرى بانتظار المراجعة حاليًا." },
  noResults: { title: "لا توجد نتائج", description: "لا توجد تعليقات مطابقة للفلاتر المحددة." },
  queueLoading: "جاري تحميل القائمة…",
  queueFailed: "تعذر تحميل القائمة.",
  mutationFailed: "تعذر تنفيذ الإجراء. تحقق من الرد ثم أعد المحاولة.",
  versionConflict: "تعارض في الحفظ: حدّث العنصر ثم أعد المحاولة.",
  messagesEmpty: { title: "لا توجد محادثات معروضة", description: "لم تُفعّل خدمة الرسائل الداخلية بعد." },
  messagesUnavailable: { title: "التراسل الداخلي غير متاح بعد", description: "يتطلب عرض المحادثات وإرسال الرسائل ربط واجهة رسائل فعلية في الخادم. لا تُعرض رسائل وهمية ولا تُحفظ مسودات محليًا." },
} as const;
