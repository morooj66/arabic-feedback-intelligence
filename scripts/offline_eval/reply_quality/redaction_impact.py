"""How much does the outbound sanitiser change REAL comments, and does it destroy meaning? Offline, local, no Gemini. Runs sanitize_text over all raw visitor comments
and compares with the project's reference name policy (jev_name_policy). Prints/stores aggregates and masked-context samples only - never the masked names."""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/offline_eval/jev_dev_validation"))
import jev_name_policy as NP  # noqa: E402
import pandas as pd  # noqa: E402
from config import settings  # noqa: E402
from rcs.ingest.loader import load_raw  # noqa: E402
from rcs.response_generation.outbound_privacy import NAME, sanitize_text  # noqa: E402

OUT = ROOT / "data/evaluation/reply_quality"; OUT.mkdir(parents=True, exist_ok=True)
raw = load_raw(ROOT / "data/raw/historical-export.xlsx", settings.SOURCE_COLUMNS)
texts = [str(t) for t in raw["original_comment"].fillna("").tolist() if str(t).strip()]
TOKR = re.compile(r"[؀-ۿA-Za-z]+")
acts, blocked, changed, ret, heavy, masked_comments = Counter(), Counter(), 0, [], 0, 0
mine_only, jev_only, both_names = 0, 0, 0
samples = []
random.seed(11)
for t in texts:
    r = sanitize_text(t, names=True)
    j = NP.sanitize(t)
    if r.text is None:
        blocked[r.blocked_reason] += 1
        continue
    a = r.actions
    acts.update(a)
    orig_tok = TOKR.findall(t)
    new_tok = [w for w in TOKR.findall(r.text) if w not in ("name", "removed", "phone", "number", "link", "handle", "email", "id", "address")]
    kept = len(set(orig_tok) & set(new_tok)) / max(1, len(set(orig_tok)))
    if r.text != " ".join(t.split()):
        changed += 1
    sem = [k for k, v in a.items() if v and k != "contact_form_boilerplate_removed"]
    if sem:                                   # comments where a name / link / handle / e-mail / phone / id was actually masked (not just form boilerplate dropped)
        masked_comments += 1; ret.append(kept)
        if kept < 0.7: heavy += 1
    name_mask_mine = NAME in r.text
    name_mask_jev = j.text is not None and "[NAME]" in j.text
    if name_mask_mine and not name_mask_jev: mine_only += 1
    if name_mask_jev and not name_mask_mine: jev_only += 1
    if name_mask_mine and name_mask_jev: both_names += 1
    if name_mask_mine and not name_mask_jev and len(samples) < 400: samples.append(r.text)
summary = {"comments_processed": len(texts), "changed_share": round(changed / len(texts), 3), "blocked_share": round(sum(blocked.values()) / len(texts), 3),
           "blocked_reasons": dict(blocked), "actions": dict(acts.most_common()),
           "comments_with_a_real_mask_(name/link/handle/email/phone/id)": masked_comments, "mean_token_retention_in_those": round(sum(ret) / max(1, len(ret)), 3), "of_those_losing_over_30pct_of_distinct_tokens": heavy,
           "name_masking_vs_reference_policy": {"both_mask_a_name": both_names, "only_this_sanitiser": mine_only, "only_reference_policy": jev_only}}
(OUT / "redaction_impact.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
(OUT / "redaction_impact_samples.json").write_text(json.dumps(random.sample(samples, min(25, len(samples))), ensure_ascii=False, indent=1))
print(json.dumps(summary, indent=1, ensure_ascii=False))
