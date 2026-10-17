"""Frozen RCS predictions on the SAME sanitized inputs. Local inference only: no network, no cost,
no model change (loads the frozen artifacts exactly as production does)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jev_common import OUT_DIR  # noqa: E402
from rcs.classification import frozen_inference  # noqa: E402


def main() -> None:
    rows = [json.loads(l) for l in (OUT_DIR / "sanitized_inputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [r["eval_id"] for r in rows]
    texts = [r["text"] for r in rows]
    intent = frozen_inference.predict_intent(texts)
    topic = frozen_inference.predict_topic(ids, texts)
    sent = frozen_inference.predict_sentiment(ids, texts)
    out = {
        i: {"intent": intent[n][0], "topic": topic[i][0], "sentiment": sent[i][0]}
        for n, i in enumerate(ids)
    }
    (OUT_DIR / "rcs_predictions.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {len(out)} frozen-RCS predictions")


if __name__ == "__main__":
    main()
