"""Paths and constants shared by the Jev development-validation tooling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GOLD_CSV = ROOT / "data" / "gold" / "gold_dataset_final_training.csv"
HUMAN_TEST_EXCLUSION = ROOT / "data" / "gold" / "human_test_exclusion_ids.json"
OUT_DIR = ROOT / "data" / "evaluation" / "jev_dev_validation"

# Previously exposed DEVELOPMENT splits only. 'val' was the original Jev evaluation set (it motivated the v2
# prompt). 'train' is the frozen models' own training data, so it is only used for ZERO-SHOT Jev comparisons
# (never to score the frozen RCS models). 'test' is the sealed human_test set and is never selectable.
ALLOWED_SPLITS = ("val", "train")
DEFAULT_SPLITS = ("val",)
V2_HOLDOUT_DIR = ROOT / "data" / "evaluation" / "jev_v2_holdout"
OBSERVED_USD_PER_REQUEST = 0.007824 / 94  # provider-reported, v1 prompt, 94 real requests (2026-10-14)

# Measured on the 2026-10-13 exploratory run (docs/evaluations/rcs_vs_jev_2026-10-13):
# 30 requests, 42,118 input + 6,250 output tokens, $0.001768956 total.
PRIOR_RUN = {"requests": 30, "input_tokens": 42118, "output_tokens": 6250, "usd": 0.001768956}
BLENDED_USD_PER_TOKEN = PRIOR_RUN["usd"] / (PRIOR_RUN["input_tokens"] + PRIOR_RUN["output_tokens"])
COST_SAFETY_FACTOR = 2.0
