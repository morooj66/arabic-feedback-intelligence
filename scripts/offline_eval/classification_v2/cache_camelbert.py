"""Run the FROZEN pretrained CAMeLBERT sentiment model locally (inference only, cached HF weights, no network) over the 563 development
rows and cache the class probabilities. No training, no labels used."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import json
from scripts.offline_eval.classification_v2.common import OUT, load_dev
from rcs.training.gold_splits import LabeledExample
from rcs.training.sentiment_pretrained import evaluate_pretrained_sentiment

rows = load_dev()
ex = [LabeledExample(comment_id=r["comment_id"], text=r["comment_text"], label="NEUTRAL", annotation_stage="dev", split="dev") for r in rows]
res = evaluate_pretrained_sentiment(ex, model_id="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment", batch_size=16)
labels = list(res.label_order) if hasattr(res, "label_order") else None
out = {"labels": labels, "comment_ids": list(res.comment_ids), "probabilities": res.probabilities, "y_pred": list(res.y_pred)}
(OUT / "camelbert_probs_dev563.json").write_text(json.dumps(out))
import csv, numpy as np
syn = [s["text"] for s in csv.DictReader(open(OUT.parents[1] / "evaluation/error_analysis/synthetic_augmentation_candidates.csv", encoding="utf-8"))]
sres = evaluate_pretrained_sentiment([LabeledExample(comment_id=f"syn{i}", text=t, label="NEUTRAL", annotation_stage="syn", split="syn") for i, t in enumerate(syn)],
                                     model_id="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment", batch_size=16)
np.save(OUT / "camelbert_probs_synthetic36.npy", np.asarray(sres.probabilities, dtype=float))
print("cached", len(out["comment_ids"]), "labels", labels)
