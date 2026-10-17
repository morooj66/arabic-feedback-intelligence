"""LOCAL error analysis + targeted classification experiments on already-exposed development data.

No network, no paid API, no production model change, no database write. Gold labels are only read.

Protocol (so gains are not mistaken for independent accuracy):
  * Candidates are TRAINED on the development `train` split only.
  * Hyper-parameters / thresholds are chosen by cross-validation INSIDE `train`.
  * The chosen candidate is then scored ONCE on the 94 sanitized development-validation records,
    next to the frozen-RCS and Jev baselines.
  * The sealed human_test rows (split == 'test') are never read: the CSV reader drops them on sight.

Run:  python scripts/offline_eval/jev_dev_validation/experiments.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix, hstack  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import jev_common  # noqa: E402,F401  (puts src/ and the repo root on sys.path)
import jev_metrics as M  # noqa: E402
from config import settings  # noqa: E402
from jev_common import GOLD_CSV, OUT_DIR  # noqa: E402
from rcs.ingest.loader import load_raw  # noqa: E402

RAW_XLSX = Path(__file__).resolve().parents[3] / "data" / "raw" / "historical-export.xlsx"
TASKS = ("intent", "topic", "sentiment")
GOLD_COL = {"intent": "gold_primary_intent", "topic": "gold_topic_subtype", "sentiment": "gold_sentiment"}
ENTRY_TYPES = ("الإبلاغ عن مدخل", "ما أصل هذه الكلمة وهل وردت في المعاجم القديمة بالمعنى نفسه", "التعليق على المدخل")


# --------------------------------------------------------------------------- data


def load_split(split: str, include_few_shot: bool = False) -> list[dict]:
    """Stream the gold CSV keeping ONLY `split` rows (train or val); every other row is dropped unread."""
    assert split in ("train", "val")
    rows = []
    with open(GOLD_CSV, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != split:
                continue
            if r["gold_annotation_status"] != "LABELED" or not all(r[c] for c in GOLD_COL.values()):
                continue
            if not include_few_shot and str(r.get("is_few_shot_example", "")).lower() in ("true", "1", "yes"):
                continue
            rows.append(r)
    return rows


def attach_types(rows: list[dict]) -> None:
    raw = load_raw(RAW_XLSX, settings.SOURCE_COLUMNS)
    t = dict(zip(raw.comment_id, raw.original_type))
    for r in rows:
        r["original_type"] = t.get(r["comment_id"]) or "UNKNOWN"


# ----------------------------------------------------------------- feature models


class TextTypeLR:
    """TF-IDF (word 1-2 + char_wb 2-5) logistic regression, optionally with the form type one-hot."""

    def __init__(self, C: float = 3.0, use_type: bool = False):
        self.C, self.use_type = C, use_type

    def _type_matrix(self, types, fit=False):
        if fit:
            self.types_ = sorted(set(types))
        idx = {t: i for i, t in enumerate(self.types_)}
        m = np.zeros((len(types), len(self.types_)))
        for n, t in enumerate(types):
            if t in idx:
                m[n, idx[t]] = 1.0
        return csr_matrix(m)

    def _x(self, texts, types, fit=False):
        if fit:
            self.w_ = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            self.c_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=40000)
            parts = [self.w_.fit_transform(texts), self.c_.fit_transform(texts)]
        else:
            parts = [self.w_.transform(texts), self.c_.transform(texts)]
        if self.use_type:
            parts.append(self._type_matrix(types, fit) * 2.0)
        return hstack(parts).tocsr()

    def fit(self, texts, types, y):
        self.clf_ = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        self.clf_.fit(self._x(texts, types, fit=True), y)
        return self

    def predict(self, texts, types):
        return list(self.clf_.predict(self._x(texts, types)))

    def predict_proba(self, texts, types):
        return self.clf_.predict_proba(self._x(texts, types)), list(self.clf_.classes_)


def cv_macro_f1(rows, task, make, k=4, seed=0):
    texts = [r["comment_text"] for r in rows]
    types = [r["original_type"] for r in rows]
    y = [r[GOLD_COL[task]] for r in rows]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    pred = [None] * len(y)
    proba = [None] * len(y)
    classes = None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in skf.split(texts, y):
            m = make().fit([texts[i] for i in tr], [types[i] for i in tr], [y[i] for i in tr])
            P, classes = m.predict_proba([texts[i] for i in te], [types[i] for i in te])
            for row, i in enumerate(te):
                proba[i] = dict(zip(classes, P[row]))
                pred[i] = classes[int(P[row].argmax())]
    pc = M.per_class(y, pred, M.TASK_LABELS[task])
    return M.macro_f1(pc, only_supported=True), M.accuracy(y, pred), pred, proba


# --------------------------------------------------------------------- analyses


def type_table(rows, task):
    tab: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        tab[r["original_type"]][r[GOLD_COL[task]]] += 1
    return {t: dict(c) for t, c in sorted(tab.items(), key=lambda kv: -sum(kv[1].values()))}


def lookup_model(train_rows, task):
    """original_type -> (majority label, share, n) learned on train only."""
    out = {}
    tab = type_table(train_rows, task)
    for t, c in tab.items():
        lab, n = max(c.items(), key=lambda kv: kv[1])
        out[t] = (lab, n / sum(c.values()), sum(c.values()))
    return out


def duplicate_conflicts(rows):
    seen: dict[str, list] = defaultdict(list)
    for r in rows:
        key = " ".join(r["comment_text"].split()).strip().lower()
        if len(key) >= 3:
            seen[key].append(r)
    res = {t: 0 for t in TASKS}
    groups = 0
    for g in seen.values():
        if len(g) < 2:
            continue
        groups += 1
        for t in TASKS:
            if len({x[GOLD_COL[t]] for x in g}) > 1:
                res[t] += 1
    return {"duplicate_text_groups": groups, "groups_with_conflicting_gold": res}


def flag_stats(gold_lab, flags, target):
    tp = sum(1 for g, f in zip(gold_lab, flags) if g == target and f)
    fp = sum(1 for g, f in zip(gold_lab, flags) if g != target and f)
    fn = sum(1 for g, f in zip(gold_lab, flags) if g == target and not f)
    return {"flagged": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "recall": tp / (tp + fn) if tp + fn else None, "precision": tp / (tp + fp) if tp + fp else None}


# ------------------------------------------------------------------------- main


def main() -> None:
    train, val_all = load_split("train", include_few_shot=True), load_split("val")
    attach_types(train)
    attach_types(val_all)
    gold = json.loads((OUT_DIR / "gold_labels.local.json").read_text())
    idmap = json.loads((OUT_DIR / "id_map.local.json").read_text())
    jev = {json.loads(l)["eval_id"]: json.loads(l) for l in (OUT_DIR / "jev_predictions.jsonl").read_text().splitlines()}
    rcs = json.loads((OUT_DIR / "rcs_predictions.json").read_text())
    texts_val = {json.loads(l)["eval_id"]: json.loads(l)["text"] for l in
                 (OUT_DIR / "sanitized_inputs.jsonl").read_text(encoding="utf-8").splitlines()}
    by_cid = {r["comment_id"]: r for r in val_all}
    ids = list(gold)
    vtypes = [by_cid[idmap[i]]["original_type"] for i in ids]
    vtexts = [texts_val[i] for i in ids]
    for i in ids:
        assert by_cid[idmap[i]][GOLD_COL["topic"]] == gold[i]["topic"], "gold drift"  # gold preserved
    report: dict = {"n_train": len(train), "n_val": len(ids), "tasks": {}}
    L: list[str] = ["# Local classification experiments (development data only)", "",
                    "**Not independent accuracy.** Candidates are trained on `train`; settings are chosen by cross-validation "
                    "inside `train`; the 94-record development-validation set (already used for model selection and for the Jev "
                    "comparison) is scored once. Treat gains as indicative, not as production accuracy.", ""]

    # ---- A. error analysis
    L += ["## A. Error analysis", ""]
    L += [f"train rows used: {len(train)}; validation rows: {len(ids)}", ""]
    dup = duplicate_conflicts(train + val_all)
    report["duplicates"] = dup
    L += [f"Identical texts (normalised) in train+val: {dup['duplicate_text_groups']} groups; groups whose gold labels "
          f"DISAGREE: {dup['groups_with_conflicting_gold']}", ""]
    for task in ("topic", "intent"):
        tt = type_table(train + val_all, task)
        report.setdefault("type_tables", {})[task] = tt
        L += [f"### original_type x gold {task} (train+val)", "", "| original_type | n | distribution |", "|---|---|---|"]
        for t, c in tt.items():
            n = sum(c.values())
            L.append(f"| {t} | {n} | " + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])) + " |")
        L.append("")
    # shared vs model-specific errors
    L += ["### Who is wrong: taxonomy boundary vs model", ""]
    L += ["| Task | Jev wrong | RCS wrong | both wrong | both wrong, SAME label | both wrong, DIFFERENT label | only Jev | only RCS |", "|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        jw = {i for i in ids if jev[i][t] != gold[i][t]}
        rw = {i for i in ids if rcs[i][t] != gold[i][t]}
        both = jw & rw
        same = {i for i in both if jev[i][t] == rcs[i][t]}
        L.append(f"| {t} | {len(jw)} | {len(rw)} | {len(both)} | {len(same)} | {len(both - same)} | {len(jw - rw)} | {len(rw - jw)} |")
        report["tasks"].setdefault(t, {})["errors"] = dict(jev=len(jw), rcs=len(rw), both=len(both), both_same_label=len(same))
    L += ["", "Reading: 'both wrong with the SAME wrong label' is the strongest signal of a label/taxonomy-boundary problem "
          "(two unrelated systems independently reach the same answer); 'only one wrong' points at that model.", ""]
    pairs = Counter()
    for i in ids:
        for t in TASKS:
            if jev[i][t] != gold[i][t] and rcs[i][t] == jev[i][t]:
                pairs[(t, gold[i][t], jev[i][t])] += 1
    L += ["Top shared-error pairs (gold -> both-predicted): " + "; ".join(f"{t}: {g}->{p} x{n}" for (t, g, p), n in pairs.most_common(8)), ""]

    # ---- B. original_type as auxiliary signal (learned on train only)
    L += ["## B. original_type as a Topic signal (lookup learned on train, scored on validation)", ""]
    look = lookup_model(train, "topic")
    L += ["| original_type | train n | majority gold topic | share |", "|---|---|---|---|"]
    for t, (lab, sh, n) in sorted(look.items(), key=lambda kv: -kv[1][2]):
        L.append(f"| {t} | {n} | {lab} | {sh:.2f} |")
    L.append("")
    lk_pred = {i: (look.get(vt, ("GENERAL", 0, 0))[0]) for i, vt in zip(ids, vtypes)}
    gt = [gold[i]["topic"] for i in ids]
    cands_topic: dict[str, list[str]] = {
        "RCS (frozen)": [rcs[i]["topic"] for i in ids],
        "Jev (v1 prompt)": [jev[i]["topic"] for i in ids],
        "type-lookup only": [lk_pred[i] for i in ids],
    }

    def override(base, lab_types, min_share=0.7, min_n=10):
        out = []
        for p, vt in zip(base, vtypes):
            lab, sh, n = look.get(vt, (None, 0, 0))
            out.append(lab if (lab and sh >= min_share and n >= min_n and p in ("WORD_SUBMISSION", "ENTRY_CORRECTION")) else p)
        return out

    cands_topic["RCS + type override (WORD/ENTRY only)"] = override(cands_topic["RCS (frozen)"], vtypes)
    cands_topic["Jev + type override (WORD/ENTRY only)"] = override(cands_topic["Jev (v1 prompt)"], vtypes)

    # ---- C. trained candidates (train only; C by CV in train)
    L += ["## C. Trained candidates (train-only fit; C chosen by 4-fold CV inside train)", ""]
    cv_rows = []
    oof: dict[tuple, tuple] = {}
    chosen: dict[tuple, float] = {}
    for task in TASKS:
        for use_type in (False, True):
            best = None
            for C in (1.0, 10.0):
                f1, acc, oof_pred, oof_proba = cv_macro_f1(train, task, lambda C=C, u=use_type: TextTypeLR(C=C, use_type=u))
                cv_rows.append((task, use_type, C, f1, acc))
                if best is None or f1 > best[0]:
                    best = (f1, C)
                    oof[(task, use_type)] = (oof_pred, oof_proba)
            chosen[(task, use_type)] = best[1]
    L += ["| Task | features | C | CV macro-F1 (train) | CV accuracy (train) |", "|---|---|---|---|---|"]
    for task, u, C, f1, acc in cv_rows:
        mark = " *" if chosen[(task, u)] == C else ""
        L.append(f"| {task} | {'text+type' if u else 'text'} | {C}{mark} | {f1:.3f} | {acc:.3f} |")
    L += ["", "(* = chosen by CV macro-F1.)", ""]
    ttexts = [r["comment_text"] for r in train]
    ttypes = [r["original_type"] for r in train]
    all_preds: dict[str, dict[str, list[str]]] = {"topic": cands_topic, "intent": {}, "sentiment": {}}
    all_preds["intent"]["RCS (frozen)"] = [rcs[i]["intent"] for i in ids]
    all_preds["intent"]["Jev (v1 prompt)"] = [jev[i]["intent"] for i in ids]
    all_preds["sentiment"]["RCS (frozen)"] = [rcs[i]["sentiment"] for i in ids]
    all_preds["sentiment"]["Jev (v1 prompt)"] = [jev[i]["sentiment"] for i in ids]
    import warnings
    for task in TASKS:
        y = [r[GOLD_COL[task]] for r in train]
        for use_type in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = TextTypeLR(C=chosen[(task, use_type)], use_type=use_type).fit(ttexts, ttypes, y)
            all_preds[task][f"LR {'text+type' if use_type else 'text'} (trained on train)"] = m.predict(vtexts, vtypes)

    # ---- D. fusion / flag experiments (a-priori rules, not tuned on validation)
    jev_neg = [jev[i]["sentiment"] == "NEGATIVE" for i in ids]
    rcs_neg = [rcs[i]["sentiment"] == "NEGATIVE" for i in ids]
    all_preds["sentiment"]["NEGATIVE if either RCS or Jev says NEGATIVE"] = [
        "NEGATIVE" if (a or b) else (jev[i]["sentiment"]) for a, b, i in zip(jev_neg, rcs_neg, ids)]

    # rare-class thresholds, chosen on train out-of-fold probabilities ONLY (never on validation)
    L_thr = ["## D2. Rare-class probability thresholds (chosen on train CV out-of-fold, applied once to validation)", "",
             "| Task | class | feature set | threshold (CV-chosen) | OOF F1 at threshold | OOF recall | OOF precision |", "|---|---|---|---|---|---|---|"]
    thr_models: dict[tuple, TextTypeLR] = {}
    for task, cls in (("intent", "COMPLAINT"), ("topic", "TECHNICAL_ISSUE"), ("sentiment", "NEGATIVE")):
        for use_type in (False, True):
            _, oof_proba = oof[(task, use_type)]
            y = [r[GOLD_COL[task]] for r in train]
            best = None
            for t in [x / 100 for x in range(10, 75, 5)]:
                f = [pr[cls] >= t for pr in oof_proba]
                st = flag_stats(y, f, cls)
                pr_, rc_ = st["precision"] or 0.0, st["recall"] or 0.0
                f1 = 2 * pr_ * rc_ / (pr_ + rc_) if pr_ + rc_ else 0.0
                if best is None or f1 > best[0]:
                    best = (f1, t, rc_, pr_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = TextTypeLR(C=chosen[(task, use_type)], use_type=use_type).fit(ttexts, ttypes, y)
            P, classes = m.predict_proba(vtexts, vtypes)
            ci = classes.index(cls)
            base = [classes[int(r.argmax())] for r in P]
            name = f"LR {'text+type' if use_type else 'text'} + {cls} threshold {best[1]:.2f}"
            all_preds[task][name] = [cls if P[n][ci] >= best[1] else base[n] for n in range(len(base))]
            L_thr.append(f"| {task} | {cls} | {'text+type' if use_type else 'text'} | {best[1]:.2f} | {best[0]:.2f} | {best[2]:.2f} | {best[3]:.2f} |")
    L += L_thr + [""]
    # out-of-fold per-class view on train (a larger sample than the 94, but also the design sample)
    L += ["## D3. Train cross-validation per-class (out-of-fold; larger sample, but used to choose settings)", ""]
    for task in TASKS:
        y = [r[GOLD_COL[task]] for r in train]
        for use_type in (False, True):
            pred, _ = oof[(task, use_type)]
            pc = M.per_class(y, pred, M.TASK_LABELS[task])
            L += [f"**{task} - LR {'text+type' if use_type else 'text'} (OOF, n={len(y)})**", "", "| class | P | R | F1 | support |", "|---|---|---|---|---|"]
            for lab, v in pc.items():
                if v["support"]:
                    L.append(f"| {lab} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} |")
            L.append("")
    # ---- scoring
    L += ["## D. Validation scores (94 records; same records for every row)", ""]
    for task in TASKS:
        g = [gold[i][task] for i in ids]
        L += [f"### {task}", "", "| Candidate | Accuracy | Macro-F1 (supported) | 95% CI macro-F1 |", "|---|---|---|---|"]
        report["tasks"].setdefault(task, {})["scores"] = {}
        for name, pred in all_preds[task].items():
            s = M.score_task(task, g, pred)
            ci = s["ci95"]["macro_f1"]
            L.append(f"| {name} | {s['accuracy']:.3f} | {s['macro_f1_supported_classes']:.3f} | {ci[0]:.2f}-{ci[1]:.2f} |")
            report["tasks"][task]["scores"][name] = {"accuracy": s["accuracy"], "macro_f1": s["macro_f1_supported_classes"],
                                                     "per_class": s["per_class"], "pred": pred}
        L.append("")
    # per-class + confusion for the candidates that matter
    thr_names = lambda task: [n for n in all_preds[task] if "threshold" in n]
    focus = {"topic": ["RCS (frozen)", "Jev (v1 prompt)", "type-lookup only", "RCS + type override (WORD/ENTRY only)",
                       "LR text (trained on train)", "LR text+type (trained on train)"] + thr_names("topic"),
             "intent": ["RCS (frozen)", "Jev (v1 prompt)", "LR text (trained on train)", "LR text+type (trained on train)"] + thr_names("intent"),
             "sentiment": ["RCS (frozen)", "Jev (v1 prompt)", "LR text (trained on train)",
                           "NEGATIVE if either RCS or Jev says NEGATIVE"] + thr_names("sentiment")}
    L += ["## E. Per-class P/R/F1 and confusion for the key candidates", ""]
    for task in TASKS:
        labs = M.TASK_LABELS[task]
        g = [gold[i][task] for i in ids]
        for name in focus[task]:
            pred = all_preds[task][name]
            pc = M.per_class(g, pred, labs)
            L += [f"**{task} - {name}**", "", "| class | P | R | F1 | support | predicted |", "|---|---|---|---|---|---|"]
            for lab in labs:
                v = pc[lab]
                if v["support"] or v["predicted"]:
                    L.append(f"| {lab} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} | {v['predicted']} |")
            cm = M.confusion(g, pred, labs, labs)
            cols = list(labs) + ["OTHER"]
            L += ["", "confusion (rows gold, cols predicted): " + ", ".join(c[:10] for c in cols), "```"]
            for gl in labs:
                if any(cm[gl].values()):
                    L.append(f"{gl[:22]:22} " + " ".join(f"{cm[gl][c]:3d}" for c in cols))
            L += ["```", ""]
    # complaint / technical-issue escalation flags
    L += ["## F. Escalation flags (priority bump) - recall vs reviewer workload", ""]
    gi = [gold[i]["intent"] for i in ids]
    gtp = [gold[i]["topic"] for i in ids]
    flags = {
        "RCS says COMPLAINT": [rcs[i]["intent"] == "COMPLAINT" for i in ids],
        "Jev says COMPLAINT": [jev[i]["intent"] == "COMPLAINT" for i in ids],
        "LR text+type says COMPLAINT": [p == "COMPLAINT" for p in all_preds["intent"]["LR text+type (trained on train)"]],
        "Jev COMPLAINT or LR COMPLAINT": [jev[i]["intent"] == "COMPLAINT" or p == "COMPLAINT"
                                          for i, p in zip(ids, all_preds["intent"]["LR text+type (trained on train)"])],
    }
    for name in thr_names("intent"):
        flags[f"{name}: flag COMPLAINT"] = [p == "COMPLAINT" for p in all_preds["intent"][name]]
        flags[f"Jev COMPLAINT or [{name}]"] = [jev[i]["intent"] == "COMPLAINT" or p == "COMPLAINT"
                                                for i, p in zip(ids, all_preds["intent"][name])]
    L += ["| Flag (target = gold COMPLAINT, 9 rows) | flagged | TP | FP | FN | recall | precision |", "|---|---|---|---|---|---|---|"]
    report["flags"] = {}
    for n, f in flags.items():
        s = flag_stats(gi, f, "COMPLAINT")
        report["flags"][n] = s
        L.append(f"| {n} | {s['flagged']} | {s['tp']} | {s['fp']} | {s['fn']} | {s['recall']:.2f} | {(s['precision'] or 0):.2f} |")
    tflags = {
        "RCS says TECHNICAL_ISSUE": [rcs[i]["topic"] == "TECHNICAL_ISSUE" for i in ids],
        "Jev says TECHNICAL_ISSUE": [jev[i]["topic"] == "TECHNICAL_ISSUE" for i in ids],
        "LR text+type says TECHNICAL_ISSUE": [p == "TECHNICAL_ISSUE" for p in all_preds["topic"]["LR text+type (trained on train)"]],
    }
    for name in thr_names("topic"):
        tflags[f"{name}: flag TECHNICAL_ISSUE"] = [p == "TECHNICAL_ISSUE" for p in all_preds["topic"][name]]
    L += ["", "| Flag (target = gold TECHNICAL_ISSUE, 4 rows) | flagged | TP | FP | FN | recall | precision |", "|---|---|---|---|---|---|---|"]
    for n, f in tflags.items():
        s = flag_stats(gtp, f, "TECHNICAL_ISSUE")
        report["flags"][n] = s
        L.append(f"| {n} | {s['flagged']} | {s['tp']} | {s['fp']} | {s['fn']} | {s['recall']:.2f} | {(s['precision'] or 0):.2f} |")
    L.append("")
    (OUT_DIR / "experiments_report.md").write_text("\n".join(L), encoding="utf-8")
    (OUT_DIR / "experiments.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print("wrote", OUT_DIR / "experiments_report.md")


if __name__ == "__main__":
    main()
