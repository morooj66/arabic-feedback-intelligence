"""Jev development-validation tooling (scripts/offline_eval/jev_dev_validation).

Nothing here touches the network: the executor is exercised with a fake transport,
and every refusal path asserts the transport was never called.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1] / "scripts" / "offline_eval" / "jev_dev_validation"
sys.path.insert(0, str(_PKG))

import jev_metrics as M  # noqa: E402
import prepare_inputs as P  # noqa: E402
import run_eval as R  # noqa: E402
from config import settings  # noqa: E402
from jev_protocol import JevProtocolError, build_request, estimate_cost, validate_response  # noqa: E402
from jev_redaction import Quarantined, redact_strict  # noqa: E402
from rcs.classification.jev_criteria import QUESTIONS  # noqa: E402

SEALED_ID = "f" * 64


# ------------------------------------------------------------------ redaction


def test_redaction_removes_emails_phones_titled_and_nasab_names():
    r = redact_strict("تواصلوا معي على a.b@test.com والأستاذ خالد العتيبي ممتاز وأبو سعد أيضًا 0555123456")
    assert "@" not in r.text and "0555123456" not in r.text
    assert "خالد" not in r.text and "العتيبي" not in r.text and "سعد" not in r.text
    assert "[PERSON]" in r.text and r.replacements


def test_redaction_quarantines_name_like_latin_and_contact_solicitation():
    with pytest.raises(Quarantined) as q:
        redact_strict("شكرا John Smith على المعجم")
    assert q.value.reason == "latin_name_like_token"
    with pytest.raises(Quarantined) as q:
        redact_strict("كلمة جميلة واتساب 0555")
    assert q.value.reason in ("contact_solicitation_with_digits", "production_redactor_failed_closed")


def test_human_cleared_latin_names_are_replaced_not_sent():
    r = redact_strict("شكرا John Smith على المعجم", release_latin_names=True)
    assert "John" not in r.text and "Smith" not in r.text and "[PERSON]" in r.text


def test_redaction_is_idempotent_and_keeps_plain_text():
    plain = "أقترح إضافة خاصية البحث حسب التصنيف"
    assert redact_strict(plain).text == plain
    once = redact_strict("الدكتور محمد الأحمد شكرا").text
    assert redact_strict(once).text == once


# --------------------------------------------------- split / sealed-id guards


def _write_gold(tmp_path: Path, rows: list[dict]) -> Path:
    cols = ["comment_id", "comment_text", "gold_primary_intent", "gold_topic_subtype", "gold_sentiment",
            "gold_annotation_status", "split", "is_few_shot_example"]
    path = tmp_path / "gold.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def _row(cid, text, split="val", intent="SUGGESTION", topic="GENERAL", sent="NEUTRAL", status="LABELED", few="False"):
    return dict(comment_id=cid, comment_text=text, gold_primary_intent=intent, gold_topic_subtype=topic,
                gold_sentiment=sent, gold_annotation_status=status, split=split, is_few_shot_example=few)


def test_loader_never_retains_test_or_train_rows(tmp_path):
    path = _write_gold(tmp_path, [
        _row("v1", "تعليق التحقق الأول"),
        _row("t1", "SEALED-TEST-TEXT-SENTINEL", split="test"),
        _row("tr1", "TRAIN-TEXT-SENTINEL", split="train"),
    ])
    df = P.load_dev_rows(path)
    assert list(df["comment_id"]) == ["v1"]
    blob = df.to_csv()
    assert "SENTINEL" not in blob


def test_sealed_human_test_id_in_selection_is_a_hard_error(tmp_path):
    df = P.load_dev_rows(_write_gold(tmp_path, [_row(SEALED_ID, "نص")]))
    with pytest.raises(SystemExit):
        P.select_rows(df, {SEALED_ID})


def test_selection_keeps_only_complete_labeled_non_few_shot_rows(tmp_path):
    df = P.load_dev_rows(_write_gold(tmp_path, [
        _row("ok", "نص جيد"),
        _row("insufficient", "نص", status="INSUFFICIENT_CONTEXT", intent="", topic="", sent=""),
        _row("fewshot", "نص مثال", few="True"),
        _row("nolabel", "نص", intent=""),
    ]))
    assert list(P.select_rows(df, set())["comment_id"]) == ["ok"]


def test_build_quarantines_and_keeps_gold_out_of_payloads(tmp_path):
    df = P.load_dev_rows(_write_gold(tmp_path, [
        _row("a", "أقترح خاصية بحث جديدة"),
        _row("b", "شكرا John Smith"),
        _row("c", "تواصل معي a@b.com", intent="COMPLAINT", topic="TECHNICAL_ISSUE", sent="NEGATIVE"),
    ]))
    res = P.build(df, set())
    assert [k["eval_id"] for k in res["kept"]] == ["dev-0001", "dev-0003"]
    assert res["quarantined"] == [{"eval_id": "dev-0002", "reason": "latin_name_like_token"}]
    out = P.write_outputs(res, tmp_path / "out")
    plan = out["plan"]
    assert plan["requests"] == 2 and plan["calls_executed"] == 0
    inputs = (tmp_path / "out" / "sanitized_inputs.jsonl").read_text(encoding="utf-8")
    assert "a@b.com" not in inputs and "John" not in inputs
    assert "gold" not in inputs.lower() and "COMPLAINT" not in inputs  # labels never in the inputs file
    assert (tmp_path / "out" / "gold_labels.local.json").exists()
    assert plan["sanitized_inputs_sha256"] == hashlib.sha256((tmp_path / "out" / "sanitized_inputs.jsonl").read_bytes()).hexdigest()


# ------------------------------------------------------------ protocol / payload


def test_request_holds_only_sanitized_text_and_frozen_taxonomy():
    req = build_request("نص معقم")
    assert set(req) == {"model", "state", "questions"} and req["state"] == {"visitor_comment_text": "نص معقم"}
    assert set(req["questions"]) == {"intent", "topic", "sentiment"}
    assert set(req["questions"]["intent"]["criteria"]) == set(settings.PRIMARY_INTENTS)
    assert set(req["questions"]["topic"]["criteria"]) == set(settings.TOPIC_SUBTYPES)
    assert set(req["questions"]["sentiment"]["criteria"]) == set(settings.SENTIMENTS)
    assert "comment_id" not in json.dumps(req)


def test_topic_criteria_follow_the_labeling_guide_distinctions():
    topic = QUESTIONS["topic"][1]
    assert "غير موجود" in topic["WORD_SUBMISSION"] and "تصحيح" in topic["WORD_SUBMISSION"]
    assert "موجود أصلًا" in topic["ENTRY_CORRECTION"]
    assert "منصة" in topic["DICTIONARY_SUGGESTION"] and "ليست كلمة" in topic["DICTIONARY_SUGGESTION"]
    assert "كلمة العام" in topic["WORD_OF_YEAR_NOMINATION"] and "عرضًا" in topic["WORD_OF_YEAR_NOMINATION"]
    assert "أولى من «عام»" in topic["OFF_TOPIC_PERSONAL"]
    assert "تفضيل" not in QUESTIONS["intent"][1]["COMPLAINT"] and "يُفضَّل الاقتراح" in QUESTIONS["intent"][1]["COMPLAINT"]


def _good_response():
    def ans(labels, pick):
        n = len(labels)
        return {"type": "choice", "choice": pick, "confidence": 0.9,
                "probabilities": {l: (0.9 if l == pick else 0.1 / (n - 1)) for l in labels}}
    return {
        "provider": settings.JEV_TOPIC_EXPECTED_PROVIDER, "model": settings.JEV_TOPIC_EXPECTED_SNAPSHOT,
        "answers": {"intent": ans(settings.PRIMARY_INTENTS, "SUGGESTION"), "topic": ans(settings.TOPIC_SUBTYPES, "GENERAL"),
                    "sentiment": ans(settings.SENTIMENTS, "NEUTRAL")},
    }


def test_response_validation_accepts_exact_shape_and_rejects_everything_else():
    out = validate_response(_good_response())
    assert {k: v[0] for k, v in out.items()} == {"intent": "SUGGESTION", "topic": "GENERAL", "sentiment": "NEUTRAL"}
    bad = []
    r = _good_response(); r["provider"] = "Other"; bad.append(r)
    r = _good_response(); r["model"] = "other"; bad.append(r)
    r = _good_response(); r["answers"]["intent"]["choice"] = "DUA_BLESSING"; bad.append(r)
    r = _good_response(); del r["answers"]["sentiment"]; bad.append(r)
    r = _good_response(); r["answers"]["extra"] = r["answers"]["topic"]; bad.append(r)
    r = _good_response(); r["answers"]["topic"]["probabilities"]["GENERAL"] = 0.0; bad.append(r)
    for b in bad:
        with pytest.raises(JevProtocolError):
            validate_response(b)


def test_cost_estimate_is_proportional_and_has_upper_bound():
    one = estimate_cost([build_request("نص")])
    many = estimate_cost([build_request("نص")] * 10)
    assert many["requests"] == 10 and many["estimated_usd"] == pytest.approx(10 * one["estimated_usd"], rel=0.02)
    assert many["upper_bound_usd"] == pytest.approx(2 * many["estimated_usd"], rel=0.01)


# --------------------------------------------------------------------- executor


def _inputs(tmp_path, n=3):
    path = tmp_path / "sanitized_inputs.jsonl"
    path.write_text("".join(json.dumps({"eval_id": f"dev-{i:04d}", "text": f"أقترح إضافة خاصية رقم {i}"},
                                       ensure_ascii=False) + "\n" for i in range(1, n + 1)), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _args(tmp_path, path, **kw):
    base = dict(inputs=str(path), predictions_out=str(tmp_path / "preds.jsonl"), limit=0, execute=False,
                approve_sha256="", max_requests=0, max_usd=0.0)
    base.update(kw)
    return argparse.Namespace(**base)


class _Transport:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def __call__(self, url, headers, body, timeout):
        self.calls += 1
        if self.fail_at == self.calls:
            return {"provider": "Other"}
        payload = json.loads(body)
        assert set(payload) == {"model", "state", "questions"}  # nothing but sanitized text + questions
        return _good_response()


def test_dry_run_sends_nothing_and_reports_exact_count(tmp_path):
    path, _ = _inputs(tmp_path)
    t = _Transport()
    out = R.run(_args(tmp_path, path), transport=t)
    assert t.calls == 0 and out["calls_executed"] == 0 and out["requests"] == 3 and out["mode"] == "dry-run"


@pytest.mark.parametrize("kw,env", [
    (dict(approve_sha256="wrong", max_requests=5, max_usd=1.0), {"OPENROUTER_API_KEY": "k"}),
    (dict(max_requests=0, max_usd=1.0), {"OPENROUTER_API_KEY": "k"}),
    (dict(max_requests=5, max_usd=0.0), {"OPENROUTER_API_KEY": "k"}),
    (dict(max_requests=2, max_usd=1.0), {"OPENROUTER_API_KEY": "k"}),       # 3 rows > cap
    (dict(max_requests=5, max_usd=1e-9), {"OPENROUTER_API_KEY": "k"}),      # cost cap
    (dict(max_requests=5, max_usd=1.0), {}),                                # no key
])
def test_execute_refuses_without_every_guard_and_never_calls(tmp_path, kw, env):
    path, digest = _inputs(tmp_path)
    kw = dict(kw)
    kw.setdefault("approve_sha256", digest)
    t = _Transport()
    with pytest.raises(SystemExit):
        R.run(_args(tmp_path, path, execute=True, **kw), transport=t, env=env)
    assert t.calls == 0


def test_execute_with_all_guards_makes_exactly_n_calls_and_writes_predictions(tmp_path):
    path, digest = _inputs(tmp_path)
    t = _Transport()
    out = R.run(_args(tmp_path, path, execute=True, approve_sha256=digest, max_requests=3, max_usd=1.0),
                transport=t, env={"OPENROUTER_API_KEY": "k"})
    assert t.calls == 3 and out["calls_executed"] == 3
    lines = (tmp_path / "preds.jsonl").read_text().splitlines()
    assert [json.loads(l)["eval_id"] for l in lines] == ["dev-0001", "dev-0002", "dev-0003"]


def test_execute_stops_at_first_failure_without_retry(tmp_path):
    path, digest = _inputs(tmp_path)
    t = _Transport(fail_at=2)
    out = R.run(_args(tmp_path, path, execute=True, approve_sha256=digest, max_requests=3, max_usd=1.0),
                transport=t, env={"OPENROUTER_API_KEY": "k"})
    assert t.calls == 2 and out["calls_executed"] == 1 and out["stopped_at"] == "dev-0002"


def test_executor_aborts_before_any_call_if_inputs_are_not_redaction_fixed_points(tmp_path):
    path = tmp_path / "sanitized_inputs.jsonl"
    path.write_text(json.dumps({"eval_id": "dev-0001", "text": "راسلني على me@example.com"}) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    t = _Transport()
    with pytest.raises(SystemExit):
        R.run(_args(tmp_path, path, execute=True, approve_sha256=digest, max_requests=1, max_usd=1.0),
              transport=t, env={"OPENROUTER_API_KEY": "k"})
    assert t.calls == 0


# ---------------------------------------------------------------------- metrics


def test_metrics_known_values():
    gold = {"1": dict(intent="COMPLAINT", topic="TECHNICAL_ISSUE", sentiment="NEGATIVE"),
            "2": dict(intent="COMPLAINT", topic="ENTRY_CORRECTION", sentiment="NEUTRAL"),
            "3": dict(intent="SPAM", topic="WORD_SUBMISSION", sentiment="NEUTRAL"),
            "4": dict(intent="INQUIRY", topic="DICTIONARY_SUGGESTION", sentiment="POSITIVE")}
    pred = {"1": dict(intent="COMPLAINT", topic="TECHNICAL_ISSUE", sentiment="NEGATIVE"),
            "2": dict(intent="SUGGESTION", topic="WORD_SUBMISSION", sentiment="NEUTRAL"),
            "3": dict(intent="SPAM", topic="WORD_SUBMISSION", sentiment="NEGATIVE"),
            "4": dict(intent="INQUIRY", topic="GENERAL", sentiment="POSITIVE")}
    res = M.evaluate(gold, pred)
    assert res["rows_scored"] == 4 and res["tasks"]["intent"]["accuracy"] == 0.75
    assert res["complaint_recall"]["hits"] == 1 and res["complaint_recall"]["support"] == 2
    assert res["technical_issue_recall"]["recall"] == 1.0
    pc = res["tasks"]["intent"]["per_class"]["COMPLAINT"]
    assert (pc["precision"], pc["recall"], pc["support"]) == (1.0, 0.5, 2)
    assert res["tasks"]["intent"]["per_class"]["SUGGESTION"]["precision"] == 0.0
    conf = res["suggestion_subtype_confusion"]
    assert conf["ENTRY_CORRECTION"]["WORD_SUBMISSION"] == 1 and conf["DICTIONARY_SUGGESTION"]["OTHER"] == 1
    assert res["sentiment_confusion"]["NEUTRAL"]["NEGATIVE"] == 1
    # macro over supported classes ignores a zero-support class; macro over all classes does not
    t = res["tasks"]["topic"]
    assert t["macro_f1_supported_classes"] >= t["macro_f1_all_classes"]


def test_metrics_missing_predictions_are_excluded_not_imputed():
    gold = {"1": dict(intent="SPAM", topic="GENERAL", sentiment="NEUTRAL"),
            "2": dict(intent="SPAM", topic="GENERAL", sentiment="NEUTRAL")}
    res = M.evaluate(gold, {"1": dict(intent="SPAM", topic="GENERAL", sentiment="NEUTRAL")})
    assert res["rows_scored"] == 1 and res["rows_missing_prediction"] == 1


def test_wilson_interval_contains_point_estimate_and_is_wide_for_tiny_n():
    lo, hi = M.wilson(0, 4)
    assert lo == 0.0 and hi > 0.4
    lo, hi = M.wilson(9, 9)
    assert hi == pytest.approx(1.0) and 0.65 < lo < 0.75


# ------------------------------------------------------- candidate v2 (prepared only)


def test_v2_candidate_changes_only_the_intended_criteria_and_keeps_the_frozen_taxonomy():
    import jev_criteria_candidate_v2 as V2

    v1, v2 = build_request("نص"), build_request("نص", "v2")
    assert set(v1) == set(v2) == {"model", "state", "questions"} and v1["state"] == v2["state"]
    for task in ("intent", "topic", "sentiment"):
        c1, c2 = v1["questions"][task]["criteria"], v2["questions"][task]["criteria"]
        assert set(c1) == set(c2) == set(QUESTIONS[task][2])  # taxonomy untouched
        changed = {k for k in c1 if c1[k] != c2[k]}
        assert changed == set(V2.CHANGED[task])
        assert v1["questions"][task]["instructions"] == v2["questions"][task]["instructions"]
    with pytest.raises(ValueError):
        build_request("نص", "v3")


def test_v2_dry_run_sends_nothing(tmp_path):
    path, _ = _inputs(tmp_path)
    t = _Transport()
    out = R.run(_args(tmp_path, path, criteria="v2"), transport=t)
    assert t.calls == 0 and out["requests"] == 3 and out["mode"] == "dry-run"


# ---------------------------------------------- holdout preparation / v1-v2 comparison


def test_train_mode_selects_train_rows_but_still_never_test_rows(tmp_path):
    path = _write_gold(tmp_path, [
        _row("v1", "تعليق التحقق", split="val"),
        _row("tr1", "تعليق تدريب أول", split="train"),
        _row("t1", "SEALED-TEST-SENTINEL", split="test"),
    ])
    df = P.load_dev_rows(path, splits=("train",))
    assert list(df["comment_id"]) == ["tr1"]
    with pytest.raises(SystemExit):  # the sealed split cannot even be requested
        P.load_dev_rows(path, splits=("test",))
    with pytest.raises(SystemExit):
        P.load_dev_rows(path, splits=("train", "test"))


def test_holdout_build_excludes_validation_texts_and_repeats(tmp_path):
    df = P.load_dev_rows(_write_gold(tmp_path, [
        _row("a", "أقترح إضافة خاصية بحث", split="train"),
        _row("b", "نص ظهر في التحقق من قبل", split="train"),
        _row("c", "أقترح إضافة خاصية بحث", split="train"),   # repeat of a
        _row("d", "نص مختلف تماما", split="train"),
    ]), splits=("train",))
    excluded = frozenset({" ".join("نص ظهر في التحقق من قبل".split()).lower()})
    res = P.build(df, set(), splits=("train",), prefix="tr", exclude_texts=excluded)
    assert [k["eval_id"] for k in res["kept"]] == ["tr-0001", "tr-0004"]
    assert res["dup_excluded"] == {"identical_to_excluded_set": 1, "repeated_within_set": 1}
    plan = P.write_outputs(res, tmp_path / "hold", versions=("v1", "v2"), splits=("train",))["plan"]
    assert plan["total_requests_all_versions"] == 4 and plan["calls_executed"] == 0
    assert set(plan["versions"]) == {"v1", "v2"} and plan["observed_rate_estimate_usd"] > 0


def _mk(gold_topics, p1_topics, p2_topics, intents=None):
    gold, p1, p2 = {}, {}, {}
    for n, (g, a, b) in enumerate(zip(gold_topics, p1_topics, p2_topics)):
        i = f"tr-{n:04d}"
        gi = (intents or ["SUGGESTION"] * len(gold_topics))[n]
        gold[i] = dict(intent=gi, topic=g, sentiment="NEUTRAL")
        p1[i] = dict(intent=gi, topic=a, sentiment="NEUTRAL")
        p2[i] = dict(intent=gi, topic=b, sentiment="NEUTRAL")
    return gold, p1, p2


def test_compare_flags_promising_when_the_preregistered_topic_criterion_holds():
    import compare_v1_v2 as C
    ws, ec = "WORD_SUBMISSION", "ENTRY_CORRECTION"
    gold = [ws] * 20 + [ec] * 10
    v1 = [ec] * 15 + [ws] * 5 + [ec] * 10          # v1: WS recall 0.25, EC recall 1.0
    v2 = [ws] * 16 + [ec] * 4 + [ec] * 9 + [ws]     # v2: WS recall 0.80, EC recall 0.90
    res = C.compare(*_mk(gold, v1, v2))
    t = res["targets"]
    assert t["topic:WORD_SUBMISSION"]["d_recall"] == pytest.approx(0.55)
    assert t["topic:ENTRY_CORRECTION"]["d_recall"] == pytest.approx(-0.10)
    assert res["verdict"]["C1_topic_boundary"] and res["verdict"]["C3_no_broad_regression"]
    assert res["verdict"]["verdict"] == "PROMISING"
    assert res["tasks"]["topic"]["d_acc_ci"][0] > 0
    assert res["tasks"]["sentiment"]["acc_v1"] == res["tasks"]["sentiment"]["acc_v2"] == 1.0
    assert res["flips"]["sentiment"]["changed_rate"] == 0.0
    assert "PROMISING" in C.to_markdown(res)


def test_compare_rejects_a_broad_regression_and_reports_not_evaluable_classes():
    import compare_v1_v2 as C
    gold = ["GENERAL"] * 40
    v1 = ["GENERAL"] * 40
    v2 = ["WORD_SUBMISSION"] * 20 + ["GENERAL"] * 20       # v2 breaks half of a previously perfect task
    res = C.compare(*_mk(gold, v1, v2))
    assert res["verdict"]["verdict"] == "REJECT" and not res["verdict"]["C3_no_broad_regression"]
    assert len(res["flips"]["topic"]["v1_right_v2_wrong"]) == 20
    assert res["targets"]["topic:TECHNICAL_ISSUE"]["support"] == 0
    assert "not evaluable" in C.to_markdown(res)


def test_adjudication_entries_are_valid_and_never_touch_gold_inputs():
    import adjudication as A
    A.validate_entries()
    assert {e[2] for e in A.E} == {A.LE, A.AM, A.ME}
    for _, task, _, rule, proposed, _ in A.E:
        assert rule in A.RULES
        assert proposed is None or proposed == A.IC or proposed in M.TASK_LABELS[task]
    assert not hasattr(A, "write_gold")  # the module has no code path that writes Gold


# ------------------------------- privacy hardening after the 2026-10-15 canary (bare Arabic names)


@pytest.mark.parametrize("text,reason", [
    ("0555123456ما أصل هذه الكلمة وهل وردت في المعاجم القديمة بالمعنى نفسه", None),                # digits glued to a name: production redactor or the ID rule
    ("[ID]غازي تركي", "id_adjacent_name_like_token"),
    ("سمعتها من جدتي فلانة بنت [PERSON] بن [PERSON]", "genealogy_chain_may_leave_given_names"),
    ("أنا [PERSON] الوعيل من قسم اللغة", "first_person_name_leftover"),
])
def test_new_fail_closed_rules_quarantine_instead_of_sending(text, reason):
    with pytest.raises(Quarantined) as q:
        redact_strict(text)
    if reason:
        assert q.value.reason == reason


def test_manual_quarantine_removes_rows_a_human_flagged_even_when_rules_pass(tmp_path):
    df = P.load_dev_rows(_write_gold(tmp_path, [
        _row("a", "أقترح خاصية بحث جديدة", split="train"),
        _row("b", "نص يبدو سليما للقواعد", split="train"),
    ]), splits=("train",))
    res = P.build(df, set(), splits=("train",), prefix="tr", manual_quarantine=frozenset({"tr-0002"}))
    assert [k["eval_id"] for k in res["kept"]] == ["tr-0001"]
    assert res["quarantined"] == [{"eval_id": "tr-0002", "reason": "manual_review_possible_personal_data"}]
    assert "tr-0002" not in res["gold"]


# ------------------------------------------------------------------ privacy_review tooling


def _synthetic_dirs(tmp_path, texts=None, quarantined=("tr-0009",), manual=("tr-0009",)):
    import privacy_review as PR
    val, hold, audit = tmp_path / "val", tmp_path / "hold", tmp_path / "audit"
    val.mkdir(); hold.mkdir()
    texts = texts or {"tr-0002": "أقترح إضافة خاصية بحث", "tr-0003": "الكلمة ومعناها شرح مفصل", "tr-0004": "نص عادي آخر هنا"}
    meta = {"id": "gen-x", "usage": {"input_tokens": 1, "output_tokens": 1, "cost": 0.0001}}
    (val / "sanitized_inputs.jsonl").write_text(
        "".join(json.dumps({"eval_id": f"dev-{n:04d}", "text": f"نص التحقق {n}"}, ensure_ascii=False) + "\n" for n in (1, 2)), encoding="utf-8")
    (val / "jev_canary_predictions.jsonl").write_text(
        json.dumps({"eval_id": "dev-0001", "provider_meta": meta}) + "\n", encoding="utf-8")
    (val / "jev_rest_predictions.jsonl").write_text(
        json.dumps({"eval_id": "dev-0002", "provider_meta": meta}) + "\n", encoding="utf-8")
    old = {**texts, "tr-0001": "نص قديم", "tr-0006": "اسمحيط فريدزلق", "tr-0009": "نص مستبعد"}
    order = ["tr-0001", "tr-0002", "tr-0003", "tr-0004", "tr-0006"]
    (hold / "sanitized_inputs.v0_approved_7047d9b0.jsonl").write_text(
        "".join(json.dumps({"eval_id": i, "text": old[i]}, ensure_ascii=False) + "\n" for i in order + ["tr-0009"]), encoding="utf-8")
    (hold / "jev_v1_canary.jsonl").write_text(
        "".join(json.dumps({"eval_id": i, "provider_meta": meta}) + "\n" for i in order), encoding="utf-8")
    (hold / "jev_v2_canary.jsonl").write_text(
        "".join(json.dumps({"eval_id": i, "provider_meta": meta}) + "\n" for i in order[:4]), encoding="utf-8")
    inputs = hold / "sanitized_inputs.jsonl"
    inputs.write_text("".join(json.dumps({"eval_id": i, "text": t}, ensure_ascii=False) + "\n" for i, t in texts.items()), encoding="utf-8")
    (hold / "request_plan.json").write_text(json.dumps({"sanitized_inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest()}))
    (hold / "quarantined.json").write_text(json.dumps([{"eval_id": q, "reason": "x"} for q in quarantined]))
    (hold / "manual_quarantine.local.json").write_text(json.dumps(list(manual)))
    return PR, val, hold, audit


def test_ledger_records_every_request_without_any_text_and_locks_the_directory(tmp_path):
    PR, val, hold, audit = _synthetic_dirs(tmp_path)
    summary = PR.build_ledger(audit, val, hold)
    assert summary["requests_total"] == 2 + 5 + 5 and summary["requests_successful"] == 11 and summary["contains_text"] is False
    ledger = (audit / "transmission_ledger.json").read_text(encoding="utf-8")
    for secret in ("نص التحقق", "أقترح إضافة خاصية بحث", "نص قديم"):
        assert secret not in ledger                      # the ledger never holds comment text
    assert "sent_provider_http_403" in ledger            # the 5th v2 request is recorded as sent-and-refused
    assert oct((audit).stat().st_mode & 0o777) == "0o700"
    assert oct((audit / "transmission_ledger.json").stat().st_mode & 0o777) == "0o400"
    assert (audit / "transmitted_payload_texts.jsonl").exists()
    assert not (hold / "sanitized_inputs.v0_approved_7047d9b0.jsonl").exists()   # moved into the locked directory


def test_verify_passes_clean_inputs_and_fails_each_unsafe_condition(tmp_path, monkeypatch):
    PR, val, hold, audit = _synthetic_dirs(tmp_path)
    PR.build_ledger(audit, val, hold)
    monkeypatch.setattr(PR, "corpus_document_frequency", lambda: __import__("collections").Counter())
    monkeypatch.setattr(PR, "RAW_XLSX", tmp_path / "missing.xlsx")
    good = PR.verify(hold, val, audit)
    assert good["all_automated_checks_passed"] and good["counts"]["ready"] == 3
    assert "cannot guarantee" in good["disclaimer"]

    def rewrite(texts, **kw):
        (hold / "sanitized_inputs.jsonl").write_text("".join(json.dumps({"eval_id": i, "text": t}, ensure_ascii=False) + "\n" for i, t in texts.items()), encoding="utf-8")
        plan = {"sanitized_inputs_sha256": hashlib.sha256((hold / "sanitized_inputs.jsonl").read_bytes()).hexdigest()}
        (hold / "request_plan.json").write_text(json.dumps(plan))
    base = {"tr-0002": "أقترح إضافة خاصية بحث", "tr-0003": "نص عادي"}
    rewrite({**base, "tr-0009": "نص مستبعد"})                      # a quarantined id slipped back in
    assert not PR.verify(hold, val, audit)["checks"]["no_quarantined_id_in_inputs"]
    rewrite({**base, "tr-0006": "نص"})                              # a confirmed-exposed id slipped back in
    assert not PR.verify(hold, val, audit)["checks"]["known_exposed_holdout_ids_excluded"]
    rewrite({**base, "tr-0010": "راسلني a.b@mail.com"})            # structural identifier
    r = PR.verify(hold, val, audit)
    assert not r["checks"]["no_structural_identifier_patterns"] and "tr-0010" in r["flagged_ids"]["email"]
    rewrite({**base, "tr-0011": "[ID]فلان"})
    assert not PR.verify(hold, val, audit)["checks"]["no_structural_identifier_patterns"]
    rewrite(base)
    (hold / "request_plan.json").write_text(json.dumps({"sanitized_inputs_sha256": "0" * 64}))   # approval hash mismatch
    assert not PR.verify(hold, val, audit)["checks"]["hash_matches_request_plan"]


def test_verify_output_never_contains_comment_text(tmp_path, monkeypatch):
    PR, val, hold, audit = _synthetic_dirs(tmp_path)
    PR.build_ledger(audit, val, hold)
    monkeypatch.setattr(PR, "corpus_document_frequency", lambda: __import__("collections").Counter())
    monkeypatch.setattr(PR, "RAW_XLSX", tmp_path / "missing.xlsx")
    out = json.dumps(PR.verify(hold, val, audit), ensure_ascii=False)
    for t in ("أقترح إضافة خاصية بحث", "الكلمة ومعناها شرح مفصل", "نص عادي آخر هنا"):
        assert t not in out


def test_quarantine_command_rejects_unknown_ids(tmp_path):
    PR, val, hold, audit = _synthetic_dirs(tmp_path)
    with pytest.raises(SystemExit):
        PR.add_quarantine(["tr-9999"], hold)


# ------------------------------------------------- context-aware name policy (identity vs lexical content)


import jev_name_policy as NP  # noqa: E402
import recover_quarantined as RQ  # noqa: E402
import candidate_ws_ec_layer as CL  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("أنا محمد وأقترح إضافة كلمة", "أنا [NAME] وأقترح إضافة كلمة"),             # self-introduction: identity -> masked
    ("ما معنى اسم محمد؟", "ما معنى اسم محمد؟"),                                  # lexical subject -> kept
    ("أقترح إضافة كلمة محمد", "أقترح إضافة كلمة محمد"),                          # proposed word -> kept
    ("اسمي سعد الغامدي أقترح كلمة جديدة", "اسمي [NAME] أقترح كلمة جديدة"),      # the span stops before the verb
    ("سمعتها من جدتي ساره بنت فلان بن علان وهي تقول هذا", "سمعتها من جدتي [NAME] وهي تقول هذا"),
    ("أنا أقترح إضافة كلمة", "أنا أقترح إضافة كلمة"),                            # a verb after 'أنا' is not a name
    ("انا احب سلاف", "انا احب سلاف"),                                            # a named public figure is not visitor identity
    ("ابن سينا قال في القانون", "ابن سينا قال في القانون"),                      # a cited historical figure stays
    ("Watch party is a common term", "Watch party is a common term"),             # Latin vocabulary stays
    ("my name is John Smith and I suggest a word", "my name is [NAME] and I suggest a word"),
])
def test_name_policy_masks_identity_but_keeps_lexical_names(raw, expected):
    r = NP.sanitize(raw)
    assert r.text == expected and r.quarantine_reason is None


def test_name_policy_removes_form_metadata_and_contact_identifiers():
    assert NP.sanitize("الجوال: 0555123456, نص الرسالة: أقترح كلمة سِجِّيل").text == "أقترح كلمة سِجِّيل"
    assert NP.sanitize("محتوي الرسالة: معنى كلمة اساسات رقم الجوال: 0555123456").text == "معنى كلمة اساسات"
    r = NP.sanitize("كلمة جميلة Phone Number: 0555123456")                       # English form label + value = metadata
    assert r.text == "كلمة جميلة" and "Phone" not in r.text
    assert NP.sanitize("راسلوني a.b@mail.com").text == "راسلوني [EMAIL]"


def test_name_policy_fails_closed_on_sensitive_or_identity_only_content():
    assert NP.sanitize("مستحقه منزل من ٨ سنوات الإسكان التنموي").quarantine_reason == "sensitive_personal_circumstance"
    assert NP.sanitize("[ID]غازي تركي").text is None                              # a name glued to a number and nothing else
    assert NP.sanitize("Test Phone Number: 123456789").quarantine_reason in ("only_identity_or_empty_content", "empty")


def test_recovery_requires_evidence_that_the_identity_context_was_handled():
    handled = NP.sanitize("أنا محمد وأقترح إضافة كلمة")
    plain = NP.sanitize("كلمة ومعناها شرح")
    assert RQ.decide("manual_review_possible_personal_data", handled)[0] == "RECOVERED"
    assert RQ.decide("manual_review_possible_personal_data", plain) == ("STILL_QUARANTINED", "human_flagged_name_but_no_identity_mask_applied")
    assert RQ.decide("latin_name_like_token", plain)[0] == "RECOVERED"             # a blunt-rule false positive is recoverable
    assert RQ.decide("latin_name_like_token", NP.sanitize("مستحقه منزل الإسكان التنموي"))[0] == "STILL_QUARANTINED"


def test_ws_ec_layer_only_touches_its_two_classes_and_follows_the_frozen_spec():
    assert CL.layer("GENERAL", "هذا خطأ", True) == "GENERAL"                       # other classes are never changed
    assert CL.layer("DICTIONARY_SUGGESTION", "كلمة", False) == "DICTIONARY_SUGGESTION"
    assert CL.layer("ENTRY_CORRECTION", "كلمة ومعناها", False) == "WORD_SUBMISSION"  # no lemma, no error claim
    assert CL.layer("WORD_SUBMISSION", "التعريف غير صحيح", False) == "ENTRY_CORRECTION"  # explicit error claim
    assert CL.layer("WORD_SUBMISSION", "كلمة ومعناها", True) == "ENTRY_CORRECTION"       # attached to an existing entry (lemma)
    assert CL.SPEC["learned_parameters"] == 0 and len(CL.SPEC_SHA256) == 64
    assert CL.SPEC_SHA256 == __import__("hashlib").sha256(json.dumps(CL.SPEC, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def test_external_request_never_carries_visitor_fields_or_local_metadata():
    blob = json.dumps(build_request("نص معقم"), ensure_ascii=False)
    for banned in ("visitor_name", "visitor_email", "email", "phone", "mobile", "comment_id", "lemma", "original_type", "entry_id", "platform"):
        assert banned not in blob
