"""Loaders for the two approved held-out sets. They read LABELS, IDS and (for the later authorised run) the comment text; nothing here prints or
logs text, and nothing runs on real data until compare_real.py is given a written approval.

human_test     gold CSV rows with split == 'test' and status LABELED (68); frozen intent/sentiment from the saved test predictions, frozen topic by local inference
blind_natural  the 45 'natural' records of blind_eval_v1 (the 30 targeted records are diagnostics only); frozen predictions from the saved blind evaluation
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .evaluate import GoldRow, Prediction
from .runner import Record

ROOT = Path(__file__).resolve().parents[3]
GOLD_CSV = ROOT / "data" / "gold" / "gold_dataset_final_training.csv"
FINAL_TEST = ROOT / "data" / "models" / "final_test"
BLIND = ROOT / "data" / "evaluation" / "blind_eval_v1"


@dataclass
class EvalSet:
    name: str
    records: list[Record]
    gold: list[GoldRow]
    frozen: dict[str, Prediction]


def counts_only(name: str, gold_csv: Path = GOLD_CSV, blind: Path = BLIND) -> dict:
    """Dry-run helper: counts from NON-text columns only (the comment-text column is never parsed)."""
    if name == "human_test":
        import pandas as pd

        df = pd.read_csv(gold_csv, usecols=["split", "gold_annotation_status"], dtype=str, encoding="utf-8-sig", keep_default_na=False)
        return {"rows": int(((df["split"] == "test") & (df["gold_annotation_status"] == "LABELED")).sum())}
    with open(blind / "coordinator_only" / "blind_eval_mapping.local.csv", newline="", encoding="utf-8") as fh:
        return {"rows": sum(1 for r in csv.DictReader(fh) if r["component"] == "natural")}


def load_human_test(frozen_topic: Mapping[str, str], gold_csv: Path = GOLD_CSV, final_test: Path = FINAL_TEST) -> EvalSet:
    with open(gold_csv, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test" and r["gold_annotation_status"] == "LABELED"]
    intent = json.loads((final_test / "intent_tfidf_logistic_test_preds.json").read_text())
    senti = json.loads((final_test / "sentiment_camelbert_test_preds.json").read_text())
    pi = dict(zip(intent["comment_ids"], intent["y_pred"]))
    ps = dict(zip(senti["comment_ids"], senti["y_pred"]))
    records = [Record(r["comment_id"], r["comment_text"], provenance="real", dataset="human_test") for r in rows]
    gold = [GoldRow(r["comment_id"], r["gold_primary_intent"], r["gold_topic_subtype"], r["gold_sentiment"]) for r in rows]
    frozen = {i: Prediction(pi[i], frozen_topic[i], ps[i]) for i in pi if i in ps and i in frozen_topic}
    return EvalSet("human_test", records, gold, frozen)


def load_blind_natural(blind: Path = BLIND) -> EvalSet:
    labels = json.loads((blind / "returned" / "labels_verbatim_from_numbers.json").read_text(encoding="utf-8"))
    with open(blind / "coordinator_only" / "blind_eval_mapping.local.csv", newline="", encoding="utf-8") as fh:
        comp = {r["record_id"]: r["component"] for r in csv.DictReader(fh)}
    preds = json.loads((blind / "results" / "rcs_predictions.json").read_text(encoding="utf-8"))
    nat = [x for x in labels if comp.get(x["record_id"]) == "natural"]
    records = [Record(x["record_id"], x["comment_text"], provenance="real", dataset="blind_natural") for x in nat]
    gold = [GoldRow(x["record_id"], x["intent"], x["topic"], x["sentiment"], bool(x.get("insufficient_context"))) for x in nat]
    frozen = {x["record_id"]: Prediction(preds[x["record_id"]]["intent"], preds[x["record_id"]]["topic"], preds[x["record_id"]]["sentiment"]) for x in nat if x["record_id"] in preds}
    return EvalSet("blind_natural", records, gold, frozen)
