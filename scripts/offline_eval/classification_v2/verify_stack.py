"""Task 1: verify the selected v2 artifacts (I10 intent, S5 sentiment), their dependencies, feature extraction and reproducibility.
Offline (cached local weights). Writes data/evaluation/classification_v2/stack_verification.json. Trains nothing, changes nothing."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn

from scripts.offline_eval.classification_v2.common import OUT, load_dev  # sets sys.path (src, root)
from config import settings
from rcs.classification.candidate_v2 import verify_artifact
from rcs.classification.v2_stack import DEFAULT_ARTIFACT_DIR, V2Stack

report = {"python": sys.version.split()[0], "platform": platform.platform(), "sklearn": sklearn.__version__, "numpy": np.__version__}
import torch, transformers  # noqa: E402
report.update(torch=torch.__version__, transformers=transformers.__version__)
decision = json.load(open(OUT / "decision.json"))
art = {}
for task, key in (("intent", "intent"), ("sentiment", "sentiment"), ("topic", "topic")):
    d = DEFAULT_ARTIFACT_DIR / task
    verify_artifact(d)                                                               # raises on any mismatch
    sha = hashlib.sha256((d / "model.joblib").read_bytes()).hexdigest()
    cfg = json.loads((d / "config.json").read_text()); repro = json.loads((d / "reproducibility.json").read_text())
    art[task] = dict(sha256_manifest_ok=True, model_sha256=sha, matches_study_decision=(sha == decision[key]["model_sha256"]), variant=cfg["variant"],
                     status=cfg["status"], extra_features=cfg["extra_features"], trained_sklearn=repro["sklearn"], runtime_sklearn=sklearn.__version__,
                     sklearn_matches=(repro["sklearn"] == sklearn.__version__), labels_equal_taxonomy=(
                         set(cfg["labels"]) == set({"intent": settings.PRIMARY_INTENTS, "sentiment": settings.SENTIMENTS, "topic": settings.TOPIC_SUBTYPES}[task])))
report["artifacts"] = art

# feature-extraction provenance (pinned e5 revision present in the local cache; CAMeL snapshot recorded)
hub = Path.home() / ".cache/huggingface/hub"
from rcs.response_intelligence.retrieval.embeddings import E5_MODEL_REVISION  # noqa: E402
e5_snap = hub / "models--intfloat--multilingual-e5-small/snapshots"
cam_snap = hub / "models--CAMeL-Lab--bert-base-arabic-camelbert-da-sentiment/snapshots"
report["feature_models"] = dict(e5_pinned_revision=E5_MODEL_REVISION, e5_pinned_revision_cached=(e5_snap / E5_MODEL_REVISION).exists(),
                                camel_cached_snapshots=sorted(p.name for p in cam_snap.iterdir()) if cam_snap.exists() else [])

# reproducibility: live features vs the features used in training/evaluation, and predictions from both routes
rows = load_dev(); texts = [r["comment_text"] for r in rows]
E5c = np.load(OUT / "e5_small_dev563.npy")
CAMc = np.log(np.clip(np.array(json.load(open(OUT / "camelbert_probs_dev563.json"))["probabilities"]), 1e-6, 1))
st = V2Stack()
st.load()
X = st.features.get(texts)
report["feature_parity"] = dict(e5_max_abs_diff=float(np.abs(X[:, :384] - E5c).max()), camel_logprob_max_abs_diff=float(np.abs(X[:, 384:] - CAMc).max()),
                                e5_cosine_min=float(np.min(np.sum(X[:, :384] * E5c, 1) / (np.linalg.norm(X[:, :384], axis=1) * np.linalg.norm(E5c, axis=1)))))
cached = np.hstack([E5c, CAMc])
rep = {}
for task, taxonomy in (("intent", settings.PRIMARY_INTENTS), ("sentiment", settings.SENTIMENTS)):
    m = st.models[task]
    P_cached = m.predict_proba(texts, extra=cached); P_live = m.predict_proba(texts, extra=X)
    live = st.predict_intent(texts) if task == "intent" else list(st.predict_sentiment([r["comment_id"] for r in rows], texts).values())
    lab_c = [m.labels[k] for k in P_cached.argmax(1)]
    rep[task] = dict(label_agreement_cached_vs_live=float(np.mean([a == b[0] for a, b in zip(lab_c, live)])), max_abs_proba_diff=float(np.abs(P_cached - P_live).max()),
                     all_labels_in_taxonomy=all(b[0] in taxonomy for b in live), all_conf_in_unit_interval=all(0.0 <= b[1] <= 1.0 for b in live),
                     deterministic=bool(np.allclose(P_live, m.predict_proba(texts, extra=X))), distribution=dict(zip(*np.unique([b[0] for b in live], return_counts=True))))
    rep[task]["distribution"] = {k: int(v) for k, v in rep[task]["distribution"].items()}
report["prediction_parity"] = rep
report["ok"] = bool(all(a["sha256_manifest_ok"] and a["matches_study_decision"] and a["labels_equal_taxonomy"] for a in art.values())
                    and report["feature_models"]["e5_pinned_revision_cached"] and report["feature_parity"]["e5_max_abs_diff"] < 1e-3
                    and report["feature_parity"]["camel_logprob_max_abs_diff"] < 1e-2 and all(v["label_agreement_cached_vs_live"] >= 0.99 for v in rep.values()))
(OUT / "stack_verification.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
