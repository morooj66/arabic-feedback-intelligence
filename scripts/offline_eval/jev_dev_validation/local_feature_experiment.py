"""Local (no network) experiment: do guide-derived cue features and source metadata help on the hard classes?

Targets: COMPLAINT (intent); WORD_SUBMISSION, ENTRY_CORRECTION, DICTIONARY_SUGGESTION (topic).

Pre-declared BEFORE any scoring (so the cue lists cannot be tuned to the validation errors): the lexicon below is taken from
the "signal" columns of LABELING_GUIDE.md plus plain spelling variants. The feature sets, the models and the selection rule
are fixed here; the feature set is chosen by 4-fold cross-validation INSIDE train; the validation set is scored once.
Nothing here prints or stores comment text. Gold is only read.

Run:  python scripts/offline_eval/jev_dev_validation/local_feature_experiment.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jev_common  # noqa: E402,F401
import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix, hstack  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import jev_metrics as M  # noqa: E402
from config import settings  # noqa: E402
from jev_common import GOLD_CSV, OUT_DIR  # noqa: E402
from rcs.ingest.loader import load_raw  # noqa: E402

RAW_XLSX = Path(__file__).resolve().parents[3] / "data" / "raw" / "historical-export.xlsx"

# ---- guide-derived lexicon (pre-declared) ---------------------------------------------------------------------
LEX = {
    "error_claim": ["خطأ", "خطا", "خاطئ", "خاطئة", "غلط", "غير صحيح", "غير صحيحة", "تصويب", "تصحيح", "الصواب", "مكرر", "تكرار",
                    "ناقص", "لا يتناسب", "لا معنى", "لا يمت", "اضطرب", "مضطرب"],
    "word_submission": ["أقترح كلمة", "اقترح كلمة", "إضافة كلمة", "اضافة كلمة", "كلمة جديدة", "الكلمة:", "الكلمة :", "المعنى:", "المعنى :",
                        "معناها", "المثال", "اللهجة"],
    "complaint_tone": ["لا يعمل", "لا يفتح", "سيء", "سيئة", "للأسف", "مشكلة", "لا يظهر", "لا يمكن", "ما حد", "اهمال", "إهمال", "تأخر"],
    "suggestion_cue": ["أتمنى", "اتمنى", "أقترح", "اقترح", "أرجو", "ارجو", "ياليت", "يرجى", "رجاء", "حبذا"],
    "feature_request": ["خاصية", "ميزة", "تطبيق", "البحث حسب", "تضيفون", "تحسين", "نسخة", "pdf", "PDF"],
    "question_cue": ["؟", "?", "كيف", "هل ", "أين", "لماذا", "ماذا", "متى", "ما معنى", "ما هو"],
}
LEX_KEYS = list(LEX)
TASKS = ("intent", "topic")
TARGETS = {"intent": ["COMPLAINT"], "topic": ["WORD_SUBMISSION", "ENTRY_CORRECTION", "DICTIONARY_SUGGESTION"]}
GOLD_COL = {"intent": "gold_primary_intent", "topic": "gold_topic_subtype"}
FEATURE_SETS = {"A text": (), "B +type": ("type",), "C +lemma": ("lemma",), "D +cues": ("cues",), "E +type+lemma+cues": ("type", "lemma", "cues")}


def load_split(split: str, few_shot: bool) -> list[dict]:
    rows = []
    with open(GOLD_CSV, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != split or r["gold_annotation_status"] != "LABELED" or not all(r[c] for c in GOLD_COL.values()):
                continue
            if not few_shot and str(r.get("is_few_shot_example", "")).lower() in ("true", "1", "yes"):
                continue
            rows.append(r)
    return rows


def attach_meta(rows: list[dict]) -> None:
    raw = load_raw(RAW_XLSX, settings.SOURCE_COLUMNS)
    meta = {r.comment_id: (r.original_type, str(r.lemma).strip() not in ("", "nan", "None")) for r in raw.itertuples()}
    for r in rows:
        r["original_type"], r["has_lemma"] = meta.get(r["comment_id"], ("UNKNOWN", False))


def cue_vector(text: str) -> list[float]:
    t = " ".join(text.split())
    return [min(3.0, float(sum(t.count(w) for w in LEX[k]))) for k in LEX_KEYS]


class Model:
    def __init__(self, groups, C):
        self.groups, self.C = groups, C

    def _extra(self, rows, fit=False):
        parts = []
        if "type" in self.groups:
            if fit:
                self.types_ = sorted({r["original_type"] for r in rows})
            parts.append(csr_matrix(np.array([[1.0 if r["original_type"] == t else 0.0 for t in self.types_] for r in rows])))
        if "lemma" in self.groups:
            parts.append(csr_matrix(np.array([[2.0 if r["has_lemma"] else 0.0] for r in rows])))
        if "cues" in self.groups:
            parts.append(csr_matrix(np.array([cue_vector(r["text"]) for r in rows])))
        return parts

    def _x(self, rows, fit=False):
        texts = [r["text"] for r in rows]
        if fit:
            self.v_ = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)
            base = self.v_.fit_transform(texts)
        else:
            base = self.v_.transform(texts)
        return hstack([base] + self._extra(rows, fit)).tocsr()

    def fit(self, rows, y):
        self.clf_ = LogisticRegression(C=self.C, solver="liblinear", class_weight="balanced", max_iter=2000)
        self.clf_.fit(self._x(rows, True), y)
        return self

    def predict(self, rows):
        return list(self.clf_.predict(self._x(rows)))


def cv_oof(rows, y, groups, C, k=4):
    pred = [None] * len(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=0).split(rows, y):
            m = Model(groups, C).fit([rows[i] for i in tr], [y[i] for i in tr])
            for i, p in zip(te, m.predict([rows[i] for i in te])):
                pred[i] = p
    return pred


def target_f1(y, pred, task):
    pc = M.per_class(y, pred, M.TASK_LABELS[task])
    return sum(pc[c]["f1"] for c in TARGETS[task]) / len(TARGETS[task]), pc


def main() -> None:
    train = load_split("train", few_shot=True)
    val_all = load_split("val", few_shot=False)
    attach_meta(train)
    attach_meta(val_all)
    for r in train:
        r["text"] = r["comment_text"]
    gold = json.loads((OUT_DIR / "gold_labels.local.json").read_text())
    idmap = json.loads((OUT_DIR / "id_map.local.json").read_text())
    sanit = {json.loads(l)["eval_id"]: json.loads(l)["text"] for l in (OUT_DIR / "sanitized_inputs.jsonl").read_text(encoding="utf-8").splitlines()}
    by_cid = {r["comment_id"]: r for r in val_all}
    ids = list(gold)
    val = []
    for i in ids:
        r = dict(by_cid[idmap[i]])
        r["text"] = sanit[i]  # the same sanitized text the Jev/RCS baselines saw
        val.append(r)
    jev = {json.loads(l)["eval_id"]: json.loads(l) for l in (OUT_DIR / "jev_predictions.jsonl").read_text().splitlines()}
    rcs = json.loads((OUT_DIR / "rcs_predictions.json").read_text())
    out = {"n_train": len(train), "n_val": len(val), "lexicon_keys": LEX_KEYS, "tasks": {}}
    L = ["# Local feature experiment (development data; not independent accuracy)", "",
         "Cue lexicon pre-declared from LABELING_GUIDE.md before scoring. Feature set chosen by 4-fold CV inside train; validation (94) scored once.", ""]
    # transparent rule baseline for WS vs EC (no learning)
    for name, rows in (("train", train), ("val", val)):
        sub = [r for r in rows if r["gold_topic_subtype"] in ("WORD_SUBMISSION", "ENTRY_CORRECTION")]
        def rule(r):
            cv = dict(zip(LEX_KEYS, cue_vector(r["text"])))
            return "ENTRY_CORRECTION" if (r["has_lemma"] or cv["error_claim"] > 0) else "WORD_SUBMISSION"
        acc = sum(rule(r) == r["gold_topic_subtype"] for r in sub) / len(sub)
        ec = [r for r in sub if r["gold_topic_subtype"] == "ENTRY_CORRECTION"]
        ws = [r for r in sub if r["gold_topic_subtype"] == "WORD_SUBMISSION"]
        out.setdefault("rule_ws_vs_ec", {})[name] = {"n": len(sub), "accuracy": acc,
                                                     "ec_recall": sum(rule(r) == "ENTRY_CORRECTION" for r in ec) / len(ec),
                                                     "ws_recall": sum(rule(r) == "WORD_SUBMISSION" for r in ws) / len(ws)}
    L += ["## Transparent rule for WORD_SUBMISSION vs ENTRY_CORRECTION (no learning): ENTRY_CORRECTION if lemma present OR an error-claim cue, else WORD_SUBMISSION", "",
          "| split | WS/EC rows | accuracy | EC recall | WS recall |", "|---|---|---|---|---|"]
    for name, v in out["rule_ws_vs_ec"].items():
        L.append(f"| {name} | {v['n']} | {v['accuracy']:.2f} | {v['ec_recall']:.2f} | {v['ws_recall']:.2f} |")
    L.append("")
    for task in TASKS:
        y = [r[GOLD_COL[task]] for r in train]
        L += [f"## {task}: train CV (out-of-fold) mean F1 over target classes {TARGETS[task]}", "", "| feature set | C | CV target-F1 | CV macro-F1 |", "|---|---|---|---|"]
        best = None
        for fname, groups in FEATURE_SETS.items():
            for C in (1.0, 10.0):
                pred = cv_oof(train, y, groups, C)
                tf, pc = target_f1(y, pred, task)
                mf = M.macro_f1(pc, only_supported=True)
                L.append(f"| {fname} | {C} | {tf:.3f} | {mf:.3f} |")
                if best is None or tf > best[0]:
                    best = (tf, fname, groups, C)
        L += ["", f"Chosen by CV: **{best[1]}**, C={best[3]}.", ""]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chosen = Model(best[2], best[3]).fit(train, y)
            textonly = Model((), best[3]).fit(train, y)
        g = [gold[i][task] for i in ids]
        cands = {"Frozen RCS": [rcs[i][task] for i in ids], "Jev v1": [jev[i][task] for i in ids],
                 "LR text only": textonly.predict(val), f"LR {best[1]} (CV-chosen)": chosen.predict(val)}
        L += [f"### {task}: validation (n={len(ids)}), scored once", "", "| candidate | accuracy | macro-F1 | " + " | ".join(f"{c} P/R/F1" for c in TARGETS[task]) + " |",
              "|---|---|---|" + "---|" * len(TARGETS[task])]
        out["tasks"][task] = {"chosen": best[1], "C": best[3], "val": {}}
        for name, pred in cands.items():
            s = M.score_task(task, g, pred)
            cells = []
            for c in TARGETS[task]:
                v = s["per_class"][c]
                cells.append(f"{v['precision']:.2f}/{v['recall']:.2f}/{v['f1']:.2f} (n={v['support']})")
            L.append(f"| {name} | {s['accuracy']:.3f} | {s['macro_f1_supported_classes']:.3f} | " + " | ".join(cells) + " |")
            out["tasks"][task]["val"][name] = {"accuracy": s["accuracy"], "macro_f1": s["macro_f1_supported_classes"],
                                               "targets": {c: s["per_class"][c] for c in TARGETS[task]}}
        L.append("")
    (OUT_DIR / "local_feature_experiment_report.md").write_text("\n".join(L), encoding="utf-8")
    (OUT_DIR / "local_feature_experiment.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print("wrote local_feature_experiment_report.md")


if __name__ == "__main__":
    main()
