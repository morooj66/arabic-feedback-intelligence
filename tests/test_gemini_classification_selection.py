"""Hybrid composition, the pre-registered selection rule, the real-data loaders and the gated comparison runner -- all on synthetic data / a mock client."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.offline_eval.gemini_classification import compare_real as cr
from scripts.offline_eval.gemini_classification import prompt as pm
from scripts.offline_eval.gemini_classification import real_data as rd
from scripts.offline_eval.gemini_classification import selection as sel
from scripts.offline_eval.gemini_classification import synthetic_cases as sc
from scripts.offline_eval.gemini_classification import taxonomy as tx
from scripts.offline_eval.gemini_classification.client import MockGeminiClient
from scripts.offline_eval.gemini_classification.evaluate import Prediction, evaluate_systems
from scripts.offline_eval.gemini_classification.hybrids import SYSTEMS, all_systems, compose, composed_from_winners
from scripts.offline_eval.gemini_classification.privacy import LiveApproval, RealDataNotApproved, preflight


def P(i, t, s): return Prediction(i, t, s)


# ----------------------------------------------------------------------------------------------------------------- hybrids
def test_compose_takes_each_task_from_its_source_and_keeps_only_common_rows():
    f = {"a": P("SPAM", "GENERAL", "NEUTRAL"), "b": P("INQUIRY", "GENERAL", "NEUTRAL")}
    g = {"a": P("SUGGESTION", "WORD_SUBMISSION", "POSITIVE")}
    out = compose(SYSTEMS["hybrid_topic_gemini"], f, g)
    assert set(out) == {"a"} and (out["a"].intent, out["a"].topic, out["a"].sentiment) == ("SPAM", "WORD_SUBMISSION", "NEUTRAL")
    full = all_systems(f, g)
    assert set(full) == set(SYSTEMS) and set(full["frozen"]) == {"a", "b"} and set(full["gemini"]) == {"a"}
    assert composed_from_winners({"intent": "gemini", "topic": "frozen"}) == {"intent": "gemini", "topic": "frozen", "sentiment": "frozen"}
    assert len(SYSTEMS) == 4                                   # a small, fixed number of combinations


# ----------------------------------------------------------------------------------------------------------------- selection rule
def _set_reports(frozen_flip, gem_flip, seed_f=1, seed_g=2, n_resamples=300):
    gold = sc.gold_rows()
    f = sc.degrade(gold, flip_rate=frozen_flip, seed=seed_f)
    g = sc.degrade(gold, flip_rate=gem_flip, seed=seed_g)
    return evaluate_systems(gold, all_systems(f, g), baseline="frozen", n_resamples=n_resamples, seed=5)


GOOD_Q = {"records": 54, "schema_failure_rate": 0.0, "api_failure_rate": 0.0, "latency_seconds": {"p50": 3.0, "p95": 5.0}, "total_cost_usd": 0.06}


def test_clear_gemini_win_on_both_sets_selects_gemini_for_every_task():
    reports = {"human_test": _set_reports(0.6, 0.0), "blind_natural": _set_reports(0.6, 0.0, 3, 4)}
    d = sel.select(reports, {"human_test": GOOD_Q, "blind_natural": GOOD_Q})
    assert {t: x["decision"] for t, x in d["tasks"].items()} == {"intent": "GEMINI", "topic": "GEMINI", "sentiment": "GEMINI"}
    assert d["architecture"] == "gemini" and d["inconclusive_tasks"] == []


def test_gemini_worse_or_equal_keeps_the_frozen_classifier():
    worse = {"human_test": _set_reports(0.0, 0.6), "blind_natural": _set_reports(0.0, 0.6, 3, 4)}
    d = sel.select(worse, {"human_test": GOOD_Q, "blind_natural": GOOD_Q})
    assert all(x["decision"] == "FROZEN" for x in d["tasks"].values()) and d["architecture"] == "frozen"
    mixed = {"human_test": _set_reports(0.6, 0.0), "blind_natural": _set_reports(0.0, 0.6, 3, 4)}     # better on one set, worse on the other
    assert all(x["decision"] == "FROZEN" for x in sel.select(mixed, {"human_test": GOOD_Q, "blind_natural": GOOD_Q})["tasks"].values())
    same = {"human_test": _set_reports(0.0, 0.0), "blind_natural": _set_reports(0.0, 0.0, 3, 4)}
    d = sel.select(same, {"a": GOOD_Q})
    assert all(x["decision"] == "FROZEN" and x["reason"] == "no measured gain" for x in d["tasks"].values())


def test_small_unproven_gain_is_inconclusive_not_a_win():
    rep = _set_reports(0.0, 0.0)
    rep["systems"]["gemini"]["intent"]["macro_f1_supported_classes"] = rep["systems"]["frozen"]["intent"]["macro_f1_supported_classes"] + 0.05
    rep["paired_vs_baseline"]["gemini"]["intent"]["macro_f1_supported_classes"]["ci95"] = (-0.1, 0.2)
    out = sel.decide_task("intent", {"s": rep}, {"ok": True})
    assert out["decision"] == "INCONCLUSIVE"
    rep["paired_vs_baseline"]["gemini"]["intent"]["macro_f1_supported_classes"]["ci95"] = (0.01, 0.2)
    assert sel.decide_task("intent", {"s": rep}, {"ok": True})["decision"] == "GEMINI"


@pytest.mark.parametrize("q", [
    {**GOOD_Q, "schema_failure_rate": 0.05}, {**GOOD_Q, "api_failure_rate": 0.03}, {**GOOD_Q, "latency_seconds": {"p50": 3.0, "p95": 40.0}}, {**GOOD_Q, "total_cost_usd": 54.0},
])
def test_reliability_latency_or_cost_failures_block_gemini(q):
    reports = {"human_test": _set_reports(0.6, 0.0)}
    d = sel.select(reports, {"human_test": q})
    assert all(x["decision"] == "FROZEN" and "reliability" in x["reason"] for x in d["tasks"].values()) and d["architecture"] == "frozen"


def test_a_large_per_class_regression_blocks_gemini():
    rep = _set_reports(0.5, 0.0)
    cls = rep["systems"]["gemini"]["intent"]["per_class"]
    fcls = rep["systems"]["frozen"]["intent"]["per_class"]
    target = next(i for i, c in enumerate(fcls) if c["support"] >= 5)
    fcls[target]["f1"] = 0.9
    cls[target]["f1"] = 0.4
    assert sel.decide_task("intent", {"s": rep}, {"ok": True})["decision"] == "FROZEN"


def test_routing_guard_reverts_a_pair_that_adds_false_exclusions():
    reports = {"human_test": _set_reports(0.6, 0.0), "blind_natural": _set_reports(0.6, 0.0, 3, 4)}
    reports["human_test"]["systems"]["hybrid_intent_topic_gemini"]["routing"]["false_exclusion"] = reports["human_test"]["systems"]["frozen"]["routing"]["false_exclusion"] + 2
    reports["human_test"]["systems"]["hybrid_topic_gemini"]["routing"]["false_exclusion"] = 99
    d = sel.select(reports, {"human_test": GOOD_Q, "blind_natural": GOOD_Q})
    assert d["winners"]["intent"] == d["winners"]["topic"] == "frozen" and d["notes"] and d["routing_guard"]["ok"] is False


def test_gemini_intent_with_frozen_topic_is_not_allowed_because_routing_was_not_measured_jointly():
    gold = sc.gold_rows()
    wrong = {"THANKS_PRAISE": "SPAM", "SPAM": "INQUIRY", "INQUIRY": "SUGGESTION", "SUGGESTION": "COMPLAINT", "COMPLAINT": "THANKS_PRAISE"}
    frozen = {g.record_id: P(wrong[g.intent], g.topic, g.sentiment) for g in gold}          # only the intent is wrong
    gemini = {g.record_id: P(g.intent, g.topic, g.sentiment) for g in gold}
    reports = {"s": evaluate_systems(gold, all_systems(frozen, gemini), baseline="frozen", n_resamples=200, seed=1)}
    d = sel.select(reports, {"s": GOOD_Q})
    assert d["tasks"]["intent"]["decision"] == "GEMINI" and d["tasks"]["topic"]["decision"] == "FROZEN"
    assert d["winners"]["intent"] == "frozen" and d["architecture"] == "frozen" and any("not evaluated jointly" in n for n in d["notes"])


# ----------------------------------------------------------------------------------------------------------------- loaders (synthetic fixture files)
def _write_fixture(tmp: Path):
    gold = tmp / "gold.csv"
    with open(gold, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["comment_id", "comment_text", "gold_primary_intent", "gold_topic_subtype", "gold_sentiment", "split", "gold_annotation_status"])
        w.writeheader()
        for i, (intent, topic, senti, status) in enumerate([("SPAM", "GENERAL", "NEUTRAL", "LABELED"), ("INQUIRY", "GENERAL", "NEUTRAL", "LABELED"), ("", "", "", "INSUFFICIENT_CONTEXT")]):
            w.writerow({"comment_id": f"t{i}", "comment_text": f"نص تجريبي {i}", "gold_primary_intent": intent, "gold_topic_subtype": topic, "gold_sentiment": senti, "split": "test", "gold_annotation_status": status})
        w.writerow({"comment_id": "v0", "comment_text": "x", "gold_primary_intent": "SPAM", "gold_topic_subtype": "GENERAL", "gold_sentiment": "NEUTRAL", "split": "val", "gold_annotation_status": "LABELED"})
    ft = tmp / "final_test"
    ft.mkdir()
    (ft / "intent_tfidf_logistic_test_preds.json").write_text(json.dumps({"comment_ids": ["t0", "t1"], "y_true": ["SPAM", "INQUIRY"], "y_pred": ["SPAM", "SUGGESTION"], "labels": []}))
    (ft / "sentiment_camelbert_test_preds.json").write_text(json.dumps({"comment_ids": ["t0", "t1"], "y_true": ["NEUTRAL", "NEUTRAL"], "y_pred": ["NEUTRAL", "NEGATIVE"], "labels": []}))
    b = tmp / "blind"
    for d in ("returned", "coordinator_only", "results"):
        (b / d).mkdir(parents=True)
    (b / "returned" / "labels_verbatim_from_numbers.json").write_text(json.dumps([
        {"record_id": "BE-001", "comment_text": "نص", "intent": "SPAM", "topic": "GENERAL", "sentiment": "NEUTRAL", "insufficient_context": False},
        {"record_id": "BE-002", "comment_text": "نص", "intent": "COMPLAINT", "topic": "TECHNICAL_ISSUE", "sentiment": "NEGATIVE", "insufficient_context": False}], ensure_ascii=False))
    (b / "coordinator_only" / "blind_eval_mapping.local.csv").write_text("record_id,comment_id,component\nBE-001,x,natural\nBE-002,y,targeted\n")
    (b / "results" / "rcs_predictions.json").write_text(json.dumps({"BE-001": {"intent": "SPAM", "topic": "GENERAL", "sentiment": "NEUTRAL"}, "BE-002": {"intent": "INQUIRY", "topic": "GENERAL", "sentiment": "NEUTRAL"}}))
    return gold, ft, b


def test_loaders_select_exactly_the_approved_rows(tmp_path):
    gold, ft, blind = _write_fixture(tmp_path)
    assert rd.counts_only("human_test", gold, blind) == {"rows": 2} and rd.counts_only("blind_natural", gold, blind) == {"rows": 1}
    ht = rd.load_human_test({"t0": "GENERAL", "t1": "GENERAL"}, gold, ft)
    assert [r.record_id for r in ht.records] == ["t0", "t1"] and all(r.provenance == "real" and r.dataset == "human_test" for r in ht.records)
    assert ht.frozen["t1"].intent == "SUGGESTION" and ht.frozen["t1"].sentiment == "NEGATIVE" and ht.gold[1].intent == "INQUIRY"
    bn = rd.load_blind_natural(blind)                                       # the targeted (rare-class) record is excluded
    assert [r.record_id for r in bn.records] == ["BE-001"] and set(bn.frozen) == {"BE-001"}


# ----------------------------------------------------------------------------------------------------------------- gated runner
def test_prompt_freeze_matches_and_detects_edits(monkeypatch):
    assert pm.verify_frozen_prompt() == pm.prompt_sha256()
    monkeypatch.setattr(pm, "SYSTEM_INSTRUCTION", pm.SYSTEM_INSTRUCTION + " extra")
    with pytest.raises(RuntimeError):
        pm.verify_frozen_prompt()


def test_approval_file_must_cover_both_datasets_and_set_caps(tmp_path):
    f = tmp_path / "a.json"
    base = {"approver": "owner", "approved_on": "2026-10-16", "scope": "final", "max_calls": 150, "max_usd": 2.0, "datasets": ["human_test", "blind_natural"], "input_sha256": {"human_test": "a" * 64, "blind_natural": "b" * 64},
            "legal_basis": "institutional approval ref", "paid_tier_confirmed": True, "independent_reviewer_read_inputs": True}
    f.write_text(json.dumps(base))
    assert cr.load_approval(f).max_calls == 150
    f.write_text(json.dumps({**base, "datasets": ["human_test"]}))
    with pytest.raises(RealDataNotApproved):
        cr.load_approval(f)
    f.write_text(json.dumps({**base, "max_usd": 0}))
    with pytest.raises(RealDataNotApproved):
        cr.load_approval(f)
    for missing_key, bad in (("legal_basis", ""), ("paid_tier_confirmed", False), ("independent_reviewer_read_inputs", False)):
        f.write_text(json.dumps({**base, missing_key: bad}))
        with pytest.raises(RealDataNotApproved):
            cr.load_approval(f)
    f.write_text(json.dumps({**base, "input_sha256": {"human_test": "a" * 64}}))             # no reviewed-input hash for the second dataset
    with pytest.raises(RealDataNotApproved):
        cr.load_approval(f)


def test_frozen_topic_reproduction_check_aborts_before_any_call():
    gold = sc.gold_rows()
    bad = rd.EvalSet("human_test", sc.records(), gold, {g.record_id: P(g.intent, "GENERAL", g.sentiment) for g in gold})
    with pytest.raises(RuntimeError):
        cr.check_frozen_topic_reproduces(bad)


def test_run_live_end_to_end_with_a_mock_client_and_synthetic_sets(tmp_path):
    gold, recs = sc.gold_rows(), sc.records()
    half = len(gold) // 2
    sets = [rd.EvalSet("human_test", recs[:half], gold[:half], sc.degrade(gold[:half], flip_rate=0.5, seed=1)),
            rd.EvalSet("blind_natural", recs[half:], gold[half:], sc.degrade(gold[half:], flip_rate=0.5, seed=2))]
    client = MockGeminiClient()
    approval = LiveApproval("owner", "2026-10-16", "test", max_calls=120, max_usd=5.0, datasets=("human_test", "blind_natural"))
    res = cr.run_live(approval, client=client, sets=sets, out_dir=tmp_path, sleep=lambda s: None)
    n_ok = sum(1 for r in recs if preflight(r.text).eligible)
    assert client.calls == n_ok + 30 and res["stability"]["calls"] == 30 and res["stability"]["all_three_agreement"] == 1.0
    assert set(res["reports"]) == {"human_test", "blind_natural"} and set(res["reports"]["human_test"]["systems"]) == set(SYSTEMS)
    assert isinstance(res["decision"]["architecture"], str)
    assert set(res["decision"]["tasks"]) == set(tx.TASKS) and res["prompt_sha256"] == pm.prompt_sha256()
    report_text = (tmp_path / "real_comparison_report.json").read_text(encoding="utf-8")
    assert not re.search(r"[؀-ۿ]{3,}", report_text) and "cost_per_1000" in report_text


def test_run_live_respects_the_call_cap(tmp_path):
    gold, recs = sc.gold_rows(), sc.records()
    sets = [rd.EvalSet("human_test", recs[:20], gold[:20], sc.degrade(gold[:20], flip_rate=0.5, seed=1)), rd.EvalSet("blind_natural", recs[20:40], gold[20:40], sc.degrade(gold[20:40], flip_rate=0.5, seed=2))]
    client = MockGeminiClient()
    res = cr.run_live(LiveApproval("o", "d", "s", max_calls=25, max_usd=5.0, datasets=("human_test", "blind_natural")), client=client, sets=sets, out_dir=tmp_path, sleep=lambda s: None)
    assert client.calls <= 25 and res["sets"]["blind_natural"]["gemini_ok"] < 20


@pytest.mark.skipif(not rd.GOLD_CSV.exists(), reason="needs the private gold CSV (not shipped)")
def test_dry_run_reads_no_text_and_reports_the_planned_sizes(capsys):
    assert cr.main([]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["human_test"]["rows"] == 68 and out["blind_natural"]["rows"] == 45 and out["live_calls_planned"] == {"final": 113, "stability": 30}
    assert not re.search(r"[؀-ۿ]", json.dumps(out, ensure_ascii=False))
