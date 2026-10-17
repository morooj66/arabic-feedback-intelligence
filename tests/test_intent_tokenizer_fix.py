"""Intent tokenizer-fix investigation (scripts/offline_eval/intent_tokenizer_fix): mechanism, isolation, and reproduction of the frozen baseline."""
from __future__ import annotations

import pytest

from scripts.offline_eval.intent_tokenizer_fix import evaluate as ev


def test_default_token_pattern_drops_fully_diacritised_words():
    word = "يَفْتَحُ"
    assert ev._TOKEN.findall(word) == []                     # the confirmed defect: a fully diacritised word yields no token
    assert ev._TOKEN.findall(ev.strip_diacritics(word)) == ["يفتح"]
    assert ev.strip_diacritics("كلـــمة") == "كلمة"


def test_candidates_differ_from_the_frozen_pipeline_only_by_the_preprocessor():
    base = ev.make_pipeline(None)
    for name, fn in ev.PREPROCESSORS.items():
        pipe = ev.make_pipeline(fn)
        a, b = base.named_steps["tfidf"].get_params(), pipe.named_steps["tfidf"].get_params()
        assert {k: v for k, v in a.items() if k != "preprocessor"} == {k: v for k, v in b.items() if k != "preprocessor"}, name
        assert base.named_steps["clf"].get_params() == pipe.named_steps["clf"].get_params()


@pytest.mark.skipif(not ev.GOLD.exists(), reason="needs the private gold CSV (not shipped)")
def test_only_train_and_val_rows_are_used_and_the_frozen_score_is_reproduced():
    rows = ev.load_rows()
    assert {r["split"] for r in rows} == {"train", "val"}                       # the sealed test split is never loaded
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    pipe = ev.make_pipeline(None).fit([r["comment_text"] for r in train], [r["gold_primary_intent"] for r in train])
    from scripts.offline_eval.gemini_classification import metrics as m

    got = m.task_report(ev.INTENTS, [r["gold_primary_intent"] for r in val], pipe.predict([r["comment_text"] for r in val]).tolist())["macro_f1_all_classes"]
    assert got == pytest.approx(ev.FROZEN_VAL_MACRO_F1, abs=1e-9)
