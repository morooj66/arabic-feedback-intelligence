"""Invented demo comments and the marker-stamped workbook the demo mode accepts. Every sentence is made up for this project (no real visitor text)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

WORDS = ["سوار", "غيمة", "مدينة", "رحلة", "نافذة", "حديقة", "كتاب", "ميزان", "طريق", "قلم", "جسر", "ضياء"]
TEMPLATES = [
    "شكرًا لكم على هذا المعجم المفيد وعلى جهودكم الواضحة",
    "جزاكم الله خيرًا على سهولة البحث وجودة الشروح",
    "الصفحة لا تفتح من الهاتف وتظهر رسالة خطأ عند البحث عن {w}",
    "رابط الصوت لا يعمل عندي في صفحة كلمة {w}",
    "المعنى المذكور لكلمة {w} غير دقيق وأرجو مراجعته",
    "أقترح إضافة خاصية البحث بحسب الجذر لتسهيل الوصول إلى الكلمات",
    "أتمنى إضافة أمثلة تطبيقية أكثر لكل مادة في المعجم",
    "أقترح إضافة كلمة {w} إلى المعجم لأنها شائعة بين الناس",
    "أرشح كلمة {w} لتكون كلمة العام لأنها تعبّر عن المرحلة",
    "ما معنى كلمة {w} وكيف تُنطق بالشكل الصحيح",
    "هل كلمة {w} فصيحة وما أصلها في اللغة",
    "اشتروا منتجاتنا الآن بأسعار مخفضة وعروض حصرية",
]


def demo_rows(n: int = 36) -> list[dict]:
    t0 = dt.datetime(2026, 1, 1, 9, 0)
    return [{"original_comment": f"{TEMPLATES[i % len(TEMPLATES)].format(w=WORDS[i % len(WORDS)])} (عينة تجريبية رقم {i + 1})", "original_title": None, "entry_id": None, "lemma": None,
             "visitor_name": f"زائر تجريبي {i + 1}", "visitor_email": f"demo{i + 1}@example.invalid", "original_type": "ملاحظة",
             "comment_created_at": t0 + dt.timedelta(hours=i), "platform": "web"} for i in range(n)]


def write_demo_workbook(path: Path, n: int = 36) -> Path:
    from rcs.demo import workbook
    from rcs.review.excel_review_flow import write_fixture_workbook

    out = write_fixture_workbook(Path(path), demo_rows(n))
    workbook.stamp(out)
    return out
