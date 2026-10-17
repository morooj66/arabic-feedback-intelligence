"""Unsupervised sentence embeddings (multilingual-e5-small, cached local weights, no network, no labels) of the 563 development rows.
Mean-pooled, L2-normalised, 'passage: ' prefix (the adapter's embed_passages), max_length 64 - the adapter already validated in the repository. Used as frozen FEATURES only."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from scripts.offline_eval.classification_v2.common import OUT, load_dev
from rcs.response_intelligence.retrieval.embeddings import E5EmbeddingAdapter

rows = load_dev()
ad = E5EmbeddingAdapter(); ad.load()
fn = ad.embed_passages
import csv
syn = [s["text"] for s in csv.DictReader(open(OUT.parents[1] / "evaluation/error_analysis/synthetic_augmentation_candidates.csv", encoding="utf-8"))]
np.save(OUT / "e5_small_synthetic36.npy", np.asarray(fn(syn), dtype=float))
E = np.asarray(fn([r["comment_text"] for r in rows]), dtype=float)
np.save(OUT / "e5_small_dev563.npy", E)
print(E.shape, float(np.linalg.norm(E, axis=1).mean()))
