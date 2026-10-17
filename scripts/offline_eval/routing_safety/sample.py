"""
Long-comment routing-safety evaluation — deterministic sample builder.

EVALUATION ONLY. Read-only against the source Excel and the frozen
model artifacts. Writes exactly one local JSONL file under
scripts/offline_eval/routing_safety/results/. Never touches
PostgreSQL/Supabase/MongoDB, never writes review_items, never modifies
the source workbook, never changes classification/routing/prompt code.

--- "Long", defined deterministically before sampling ------------------

A comment is "long" when its Topic-tokenizer token count (untruncated,
UBC-NLP/MARBERTv2's own tokenizer) exceeds `LONG_TOKEN_THRESHOLD` = 256
-- more than DOUBLE the frozen Topic model's actual inference window
(128, from FROZEN_MODEL_SELECTION.json). This guarantees that for every
sampled comment, less than half of its tokenized content can possibly
reach the model in a single window, which is the exact condition the
routing-safety question is about. The threshold is fixed in code, not
tuned after seeing results.

--- Sampling ----------------------------------------------------------

`SAMPLE_SEED` (fixed, printed below) drives `random.Random(SAMPLE_SEED)
.sample(...)` over the SORTED list of qualifying comment_ids (sorting
first makes the population order deterministic regardless of any
non-deterministic pandas iteration order). No comment is chosen or
excluded based on its predicted class or on whether it "looks
problematic" -- selection happens BEFORE any topic/intent/sentiment
prediction is computed for the sampled set.

--- Privacy ------------------------------------------------------------

The output JSONL stores a SHORT, REDACTED preview (first 80 characters
of `redact_for_generation(original_comment)`) for human-review display
only -- never the full raw comment, never visitor_name/visitor_email
(load_raw's columns for those are never even read here), never a raw
Gemini prompt (no Gemini call happens in this module at all).
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from config import settings  # noqa: E402
from rcs.classification.frozen_inference import predict_intent, predict_sentiment  # noqa: E402
from rcs.classification import frozen_inference as _frozen_inference  # noqa: E402
from rcs.ingest.loader import load_raw  # noqa: E402
from rcs.response_generation.eligibility import GENERATE_DRAFT  # noqa: E402
from rcs.response_generation.prompt import redact_for_generation  # noqa: E402
from rcs.response_intelligence.inspector import inspect_eligibility  # noqa: E402

SOURCE_FILE = _ROOT / "data" / "raw" / "historical-export.xlsx"
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "routing_safety_sample.jsonl"

LONG_TOKEN_THRESHOLD = 256
SAMPLE_SIZE = 25
SAMPLE_SEED = 42

_PREVIEW_CHARS = 80


@dataclass(frozen=True)
class SampledCaseRecord:
    eval_id: str
    source_row_index: int
    comment_preview: str
    total_topic_tokens: int
    topic_prediction: str
    topic_confidence: float
    topic_second_best: Optional[str]
    topic_second_best_confidence: Optional[float]
    intent_prediction: str
    intent_confidence: float
    sentiment_prediction: str
    routing_decision: str
    routing_reason_code: str
    would_call_gemini: bool
    window_beginning_topic: Optional[str] = None
    window_beginning_confidence: Optional[float] = None
    window_middle_topic: Optional[str] = None
    window_middle_confidence: Optional[float] = None
    window_ending_topic: Optional[str] = None
    window_ending_confidence: Optional[float] = None
    window_averaged_top_topic: Optional[str] = None
    window_averaged_top_confidence: Optional[float] = None


def _topic_tokenizer():
    from rcs.training.hf_loading import load_tokenizer

    human_test = _frozen_inference._load_human_test_module()
    checkpoint = human_test.find_topic_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError("Frozen MARBERTv2 topic checkpoint not found -- see frozen_inference.py")
    return load_tokenizer(str(checkpoint)), checkpoint, human_test


def _topic_distribution_batch(texts: list[str], comment_ids: list[str], human_test, checkpoint) -> dict[str, list[tuple[str, float]]]:
    """Reuses the REAL evaluate_frozen_topic (same function
    frozen_inference.predict_topic calls internally) to get the FULL
    class-probability distribution per text, not just the argmax --
    needed for "second-best Topic" and for the window comparisons.
    Returns {comment_id: [(label, prob), ...]} sorted by prob desc."""
    from rcs.training.gold_splits import LabeledExample

    examples = [
        LabeledExample(comment_id=cid, text=text, label="GENERAL", annotation_stage="routing_safety_eval", split="eval")
        for cid, text in zip(comment_ids, texts)
    ]
    result = human_test.evaluate_frozen_topic(examples, checkpoint)
    out = {}
    for cid, probs in zip(result["comment_ids"], result["probabilities"]):
        ranked = sorted(zip(result["labels"], probs), key=lambda kv: -kv[1])
        out[cid] = ranked
    return out


def build_sample() -> list[SampledCaseRecord]:
    df = load_raw(SOURCE_FILE, settings.SOURCE_COLUMNS)
    df = df[df["original_comment"].notna()]

    tokenizer, checkpoint, human_test = _topic_tokenizer()

    # --- Step 1: define the "long" population, deterministically -------
    token_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        text = str(row["original_comment"])
        token_counts[str(row["comment_id"])] = len(tokenizer(text, truncation=False)["input_ids"])

    long_ids = sorted(cid for cid, n in token_counts.items() if n > LONG_TOKEN_THRESHOLD)
    print(f"population: {len(df)} comments, {len(long_ids)} exceed {LONG_TOKEN_THRESHOLD} tokens")

    # --- Step 2: deterministic seeded sample, chosen before any topic/
    # intent/sentiment prediction is computed for these comments --------
    rng = random.Random(SAMPLE_SEED)
    k = min(SAMPLE_SIZE, len(long_ids))
    sampled_ids = set(rng.sample(long_ids, k))
    sampled_rows = df[df["comment_id"].isin(sampled_ids)]
    print(f"sampled {len(sampled_rows)} of {len(long_ids)} qualifying comments (seed={SAMPLE_SEED})")

    texts = [str(r["original_comment"]) for _, r in sampled_rows.iterrows()]
    cids = [str(r["comment_id"]) for _, r in sampled_rows.iterrows()]

    # --- Step 3: production-identical predictions (single beginning-
    # window Topic call, full-text Intent, full-text Sentiment) ---------
    topic_dists = _topic_distribution_batch(texts, cids, human_test, checkpoint)
    intent_preds = dict(zip(cids, predict_intent(texts)))
    sentiment_preds = predict_sentiment(cids, texts)

    records: list[SampledCaseRecord] = []
    off_topic_texts: list[str] = []
    off_topic_cids: list[str] = []
    off_topic_windows: dict[str, dict[str, str]] = {}

    for _, row in sampled_rows.iterrows():
        cid = str(row["comment_id"])
        text = str(row["original_comment"])
        ranked = topic_dists[cid]
        top_label, top_prob = ranked[0]
        second_label, second_prob = (ranked[1] if len(ranked) > 1 else (None, None))
        intent_label, intent_conf = intent_preds[cid]
        sentiment_label, _sentiment_conf = sentiment_preds[cid]

        eligibility = inspect_eligibility(
            comment_text=text, primary_intent=intent_label, topic_subtype=top_label, sentiment=sentiment_label,
            confidence=intent_conf,
        )

        if top_label == "OFF_TOPIC_PERSONAL":
            off_topic_cids.append(cid)
            off_topic_texts.append(text)
            # ~4-5 chars/token observed for this tokenizer on this corpus
            # (see the prior investigation) -- 500 chars is a generous
            # buffer that keeps each slice's OWN tokenization under 128
            # tokens without further internal truncation for most cases.
            off_topic_windows[cid] = {
                "middle": text[max(0, len(text) // 2 - 250): len(text) // 2 + 250],
                "ending": text[-500:],
            }

        records.append(
            SampledCaseRecord(
                eval_id=cid,
                source_row_index=int(row["source_row_index"]),
                comment_preview=redact_for_generation(text)[:_PREVIEW_CHARS],
                total_topic_tokens=token_counts[cid],
                topic_prediction=top_label,
                topic_confidence=round(top_prob, 4),
                topic_second_best=second_label,
                topic_second_best_confidence=(round(second_prob, 4) if second_prob is not None else None),
                intent_prediction=intent_label,
                intent_confidence=round(intent_conf, 4),
                sentiment_prediction=sentiment_label,
                routing_decision=eligibility.routing_decision,
                routing_reason_code=eligibility.reason_code,
                would_call_gemini=(eligibility.routing_decision == GENERATE_DRAFT),
            )
        )

    # --- Step 4: multi-window signals, ONLY for OFF_TOPIC_PERSONAL cases -
    text_by_cid = {cid: text for cid, text in zip(cids, texts)}
    if off_topic_cids:
        window_texts, window_keys = [], []
        for cid in off_topic_cids:
            window_texts.append(text_by_cid[cid])
            window_keys.append((cid, "beginning"))
        for cid in off_topic_cids:
            window_texts.append(off_topic_windows[cid]["middle"])
            window_keys.append((cid, "middle"))
        for cid in off_topic_cids:
            window_texts.append(off_topic_windows[cid]["ending"])
            window_keys.append((cid, "ending"))
        window_ids = [f"{cid}::{tag}" for cid, tag in window_keys]
        window_dists = _topic_distribution_batch([str(t) for t in window_texts], window_ids, human_test, checkpoint)

        by_cid: dict[str, dict[str, tuple[str, float]]] = {cid: {} for cid in off_topic_cids}
        for (cid, tag), wid in zip(window_keys, window_ids):
            by_cid[cid][tag] = window_dists[wid][0]

        for rec in records:
            if rec.eval_id in by_cid:
                windows = by_cid[rec.eval_id]
                avg: dict[str, float] = {}
                for tag in ("beginning", "middle", "ending"):
                    label, _ = windows[tag]
                    avg[label] = avg.get(label, 0.0) + 1.0 / 3.0
                avg_top = max(avg, key=avg.get)
                idx = records.index(rec)
                records[idx] = SampledCaseRecord(
                    **{
                        **asdict(rec),
                        "window_beginning_topic": windows["beginning"][0],
                        "window_beginning_confidence": round(windows["beginning"][1], 4),
                        "window_middle_topic": windows["middle"][0],
                        "window_middle_confidence": round(windows["middle"][1], 4),
                        "window_ending_topic": windows["ending"][0],
                        "window_ending_confidence": round(windows["ending"][1], 4),
                        "window_averaged_top_topic": avg_top,
                        "window_averaged_top_confidence": round(avg[avg_top], 4),
                    }
                )

    return records


def write_sample() -> Path:
    records = build_sample()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    return RESULTS_PATH


if __name__ == "__main__":
    path = write_sample()
    print(f"wrote {path}")
