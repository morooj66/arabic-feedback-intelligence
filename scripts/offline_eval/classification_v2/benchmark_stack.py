"""Task 2: runtime performance of the v2 stack vs the frozen stack on existing local development comments (offline, cached weights).
Each scenario runs in a fresh subprocess so cold-start time and peak RSS (resource.ru_maxrss) are not contaminated by earlier work.
Usage: python -m scripts.offline_eval.classification_v2.benchmark_stack            (orchestrates all scenarios, writes benchmark_results.json)
       python -m scripts.offline_eval.classification_v2.benchmark_stack --scenario NAME   (internal)"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import argparse
import json
import resource
import statistics
import subprocess
import sys
import time

from scripts.offline_eval.classification_v2.common import OUT, load_dev


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def texts_n(n):
    rows = load_dev()
    seen, out = set(), []
    for r in rows:
        if r["comment_text"] not in seen:
            seen.add(r["comment_text"]); out.append(r)
        if len(out) >= n:
            break
    return out


def scenario_v2_cold_and_warm():
    from rcs.classification.v2_stack import V2Stack
    t0 = time.perf_counter(); st = V2Stack(); st.load_artifacts(); t_art = time.perf_counter() - t0
    t0 = time.perf_counter(); st.features.load(); t_models = time.perf_counter() - t0
    rows = texts_n(300); T = [r["comment_text"] for r in rows]; ids = [r["comment_id"] for r in rows]
    t0 = time.perf_counter(); st.predict_intent(T[:1]); st.predict_sentiment(ids[:1], T[:1]); first = time.perf_counter() - t0
    res = dict(cold_artifact_load_s=round(t_art, 3), cold_model_load_s=round(t_models, 3), cold_first_prediction_s=round(first, 3),
               cold_total_to_first_prediction_s=round(t_art + t_models + first, 3), peak_rss_mb_after_load=rss_mb())
    # warm single-comment latency: intent then sentiment on a NEW comment (feature work happens once, shared)
    shared, unshared = [], []
    for i in range(1, 101):
        t0 = time.perf_counter(); st.predict_intent([T[i]]); st.predict_sentiment([ids[i]], [T[i]]); shared.append(time.perf_counter() - t0)
    for i in range(101, 201):
        t0 = time.perf_counter(); st.predict_intent([T[i]]); st.features._cache.clear(); st.predict_sentiment([ids[i]], [T[i]]); unshared.append(time.perf_counter() - t0)
    cached = []
    for i in range(1, 101):
        t0 = time.perf_counter(); st.predict_intent([T[i]]); st.predict_sentiment([ids[i]], [T[i]]); cached.append(time.perf_counter() - t0)
    ms = lambda xs: dict(median_ms=round(statistics.median(xs) * 1000, 1), p95_ms=round(pct(xs, .95) * 1000, 1))
    res["warm_single_comment_intent_plus_sentiment"] = dict(shared_features=ms(shared), features_recomputed_for_second_task=ms(unshared), repeated_text_cache_hit=ms(cached))
    batch = {}
    for bs in (8, 32, 128):
        st.features._cache.clear()
        t0 = time.perf_counter(); st.predict_intent(T[:bs]); st.predict_sentiment(ids[:bs], T[:bs]); dt = time.perf_counter() - t0
        batch[str(bs)] = dict(total_s=round(dt, 3), per_comment_ms=round(dt / bs * 1000, 1))
    res["batch_intent_plus_sentiment_cold_cache"] = batch
    res["feature_cache_stats"] = st.features.stats
    res["peak_rss_mb_end"] = rss_mb()
    return res


def scenario_stack(which):
    """Whole classification call sequence as excel_review_flow performs it: intent, topic, sentiment for one batch."""
    from config import settings
    settings.CLASSIFICATION_MODE = "local_v2_pilot" if which == "pilot" else "legacy_rcs"
    from rcs.classification import runtime
    out = {}
    for bs in (32,):
        rows = texts_n(bs); T = [r["comment_text"] for r in rows]; ids = [r["comment_id"] for r in rows]
        for rep in range(3):
            t0 = time.perf_counter(); runtime.predict_intent(T); t1 = time.perf_counter(); runtime.predict_topic(ids, T); t2 = time.perf_counter(); runtime.predict_sentiment(ids, T); t3 = time.perf_counter()
            out[f"batch{bs}_call{rep + 1}"] = dict(intent_s=round(t1 - t0, 2), topic_s=round(t2 - t1, 2), sentiment_s=round(t3 - t2, 2), total_s=round(t3 - t0, 2))
    out["peak_rss_mb"] = rss_mb()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario")
    a = ap.parse_args()
    if a.scenario:
        fn = {"v2": scenario_v2_cold_and_warm, "pilot_stack": lambda: scenario_stack("pilot"), "frozen_stack": lambda: scenario_stack("frozen")}[a.scenario]
        print("RESULT " + json.dumps(fn())); return
    results = {}
    for s in ("v2", "frozen_stack", "pilot_stack"):
        p = subprocess.run([sys.executable, "-m", "scripts.offline_eval.classification_v2.benchmark_stack", "--scenario", s], capture_output=True, text=True, cwd=str(OUT.parents[2]))
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        results[s] = json.loads(line[-1][7:]) if line else {"error": (p.stderr or p.stdout)[-600:]}
        print(s, "done", flush=True)
    import platform
    results["environment"] = dict(platform=platform.platform(), python=sys.version.split()[0], note="Apple-silicon laptop; torch picks MPS when available. Single run per scenario; figures are indicative.")
    (OUT / "benchmark_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
