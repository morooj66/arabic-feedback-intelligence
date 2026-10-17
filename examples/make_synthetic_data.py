"""Generates the synthetic example data (deterministic). Every sentence is invented for this project: no real visitor text, names, e-mail addresses or phone numbers.

    python examples/make_synthetic_data.py        # writes examples/synthetic_labeled_comments.csv and examples/synthetic_comments.xlsx

The labelled CSV has labels by construction (each template belongs to one intent/topic/sentiment) with ~20% deliberately flipped labels to mimic annotator disagreement, so it is only useful to demonstrate the training and evaluation
code paths. Any score computed on it says nothing about real-world accuracy."""
from __future__ import annotations

import csv
import datetime as dt
import itertools
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE.parent)]

WORDS = ["سوار", "غيمة", "مدينة", "رحلة", "نافذة", "حديقة", "كتاب", "ميزان", "طريق", "قلم", "جسر", "ضياء"]
# (intent, topic, sentiment, templates). {w} is replaced by a word from WORDS.
SPEC = [
    ("THANKS_PRAISE", "GENERAL", "POSITIVE", ["شكرًا لكم على هذا المعجم المفيد وعلى جهودكم الواضحة", "جزاكم الله خيرًا على سهولة البحث وجودة الشروح", "عمل رائع ومفيد جدًا، بارك الله فيكم"]),
    ("COMPLAINT", "TECHNICAL_ISSUE", "NEGATIVE", ["الصفحة لا تفتح من الهاتف وتظهر رسالة خطأ عند البحث عن {w}", "رابط الصوت لا يعمل عندي في صفحة كلمة {w}", "الموقع بطيء جدًا ويتوقف عند فتح نتائج البحث"]),
    ("COMPLAINT", "ENTRY_CORRECTION", "NEGATIVE", ["المعنى المذكور لكلمة {w} غير دقيق وأرجو مراجعته", "وجدت خطأً إملائيًا في المثال الوارد في مادة {w}", "الشرح الموجود لكلمة {w} ناقص ولا يغطي المعنى الشائع"]),
    ("SUGGESTION", "DICTIONARY_SUGGESTION", "NEUTRAL", ["أقترح إضافة خاصية البحث بحسب الجذر لتسهيل الوصول إلى الكلمات", "أتمنى إضافة أمثلة تطبيقية أكثر لكل مادة في المعجم", "هل يمكن إضافة قائمة بالكلمات المرتبطة بكل مادة"]),
    ("SUGGESTION", "WORD_SUBMISSION", "NEUTRAL", ["أقترح إضافة كلمة {w} إلى المعجم لأنها شائعة بين الناس", "أرجو النظر في إضافة كلمة {w} بمعناها المتداول"]),
    ("SUGGESTION", "WORD_OF_YEAR_NOMINATION", "POSITIVE", ["أرشح كلمة {w} لتكون كلمة العام لأنها تعبّر عن المرحلة", "ترشيحي لكلمة العام هو كلمة {w}"]),
    ("INQUIRY", "GENERAL", "NEUTRAL", ["ما معنى كلمة {w} وكيف تُنطق بالشكل الصحيح", "هل كلمة {w} فصيحة وما أصلها في اللغة", "ما الفرق بين كلمة {w} وكلمة قريبة منها في المعنى"]),
    ("SPAM", "OFF_TOPIC_PERSONAL", "NEUTRAL", ["اشتروا منتجاتنا الآن بأسعار مخفضة وعروض حصرية", "أبحث عن وظيفة مناسبة، هل لديكم فرص شاغرة", "تعرّفوا على أفضل خدمات الاستضافة الرخيصة"]),
]


INTENTS = sorted({s[0] for s in SPEC})
TOPICS = sorted({s[1] for s in SPEC})
SENTIMENTS = sorted({s[2] for s in SPEC})


def rows(n_per_template: int = 9, seed: int = 7, noise: float = 0.2):
    """`noise` = share of labels deliberately flipped per task, mimicking annotator disagreement (the real data's dominant problem), so the demo does not show a
    misleading perfect score."""
    rnd = random.Random(seed)
    out = []
    for intent, topic, sent, templates in SPEC:
        for t in templates:
            for _ in range(n_per_template):
                i, tp, s = intent, topic, sent
                if rnd.random() < noise:
                    i = rnd.choice(INTENTS)
                if rnd.random() < noise:
                    tp = rnd.choice(TOPICS)
                if rnd.random() < noise:
                    s = rnd.choice(SENTIMENTS)
                out.append((t.format(w=rnd.choice(WORDS)), i, tp, s))
    rnd.shuffle(out)
    return out


def main() -> None:
    data = rows()
    with open(HERE / "synthetic_labeled_comments.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comment_id", "text", "intent", "topic", "sentiment"])
        for i, (text, intent, topic, sent) in enumerate(data, 1):
            w.writerow([f"syn-{i:04d}", text, intent, topic, sent])
    from rcs.review.excel_review_flow import write_fixture_workbook

    rnd = random.Random(11)
    t0 = dt.datetime(2026, 1, 1, 9, 0)
    sample = []
    for i, (text, *_rest) in enumerate(rnd.sample(data, 40), 1):
        if i % 9 == 0:
            text += " ويمكنكم التواصل معي على 0500000000"          # reserved/placeholder number: exercises the PII redaction path
        if i % 11 == 0:
            text += f" أو عبر demo{i}@example.invalid"
        sample.append({"original_comment": text, "original_title": None, "entry_id": None, "lemma": None, "visitor_name": f"زائر تجريبي {i}", "visitor_email": f"demo{i}@example.invalid",
                       "original_type": "ملاحظة", "comment_created_at": t0 + dt.timedelta(hours=i), "platform": "web"})
    out = write_fixture_workbook(HERE / "synthetic_comments.xlsx", sample)
    from rcs.demo.workbook import stamp

    stamp(out)                                                   # carries the marker the hosted demo mode requires for uploads
    print(f"wrote {len(data)} labelled rows and {len(sample)} workbook rows")


if __name__ == "__main__":
    main()
