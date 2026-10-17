#!/usr/bin/env python3
"""
Launch the Arabic RTL human-labeling wizard for the 40-row B.2 calibration
sample. Read-only on the canonical CSV and frozen b2-v2/b2-v3 predictions.

Usage:
  PYTHONPATH='src:.' .venv/bin/python scripts/run_calibration_wizard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr  # noqa: E402

from rcs.evaluation.calibration_labeling import (  # noqa: E402
    INTENT_OPTIONS,
    SECONDARY_INTENT_OPTIONS,
    SENTIMENT_OPTIONS,
    SUBCATEGORY_OPTIONS,
    TOPIC_OPTIONS,
    count_completed_rows,
    current_row,
    finalize_calibration_review,
    go_previous_row,
    is_review_complete,
    load_or_init_review_state,
    row_to_form_labels,
    save_and_advance_row,
    save_current_row,
    topic_requires_subcategory,
)

RTL_CSS = """
.gradio-container, .gradio-container * {
    direction: rtl;
    text-align: right;
}
#comment_box textarea {
    direction: rtl;
    text-align: right;
    font-size: 1.1rem;
    line-height: 1.6;
}
#progress_box {
    font-size: 1.05rem;
    font-weight: 600;
}
"""

INTENT_LABELS = [label for label, _ in INTENT_OPTIONS]
TOPIC_LABELS = [label for label, _ in TOPIC_OPTIONS]
SENTIMENT_LABELS = [label for label, _ in SENTIMENT_OPTIONS]
SUBCATEGORY_LABELS = [label for label, _ in SUBCATEGORY_OPTIONS]
SECONDARY_LABELS = [label for label, _ in SECONDARY_INTENT_OPTIONS]
INSUFFICIENT_LABEL = next(label for label, code in INTENT_OPTIONS if code == "INSUFFICIENT_CONTEXT")
SAVE_OK_MESSAGE = "تم الحفظ."

# Gradio output contract: 7 rendered UI values + status + state + show_secondary.
RENDER_OUTPUT_COUNT = 7
HANDLER_OUTPUT_COUNT = 10
UI_OUTPUT_COUNT = 8


def _progress_text(state: dict) -> str:
    position = state["current_index"] + 1
    total = len(state["rows"])
    completed = count_completed_rows(state)
    return f"التعليق {position} من {total}  |  مكتمل: {completed} من {total}"


def _render_row(state: dict, show_secondary: bool):
    row = current_row(state)
    labels = row_to_form_labels(row)
    topic_visible = labels["intent_label"] not in ("", INSUFFICIENT_LABEL)
    topic_code = next(
        (code for label, code in TOPIC_OPTIONS if label == labels["topic_label"]),
        "",
    )
    subcategory_visible = topic_visible and topic_requires_subcategory(topic_code)
    return (
        _progress_text(state),
        row["comment_text"],
        gr.update(value=labels["intent_label"] or None),
        gr.update(visible=topic_visible, value=labels["topic_label"] or None),
        gr.update(visible=topic_visible, value=labels["sentiment_label"] or None),
        gr.update(visible=subcategory_visible, value=labels["subcategory_label"] or None),
        gr.update(visible=show_secondary, value=labels["secondary_intent_label"] or None),
    )


def _pack_response(state: dict, show_secondary: bool, status: str = ""):
    if not status and is_review_complete(state):
        status = "اكتملت جميع التعليقات الـ40. يمكنك إنهاء المراجعة لحفظ التقرير."
    return _render_row(state, show_secondary) + (status, state, show_secondary)


def on_intent_change(intent_label: str):
    insufficient = intent_label == INSUFFICIENT_LABEL
    return (
        gr.update(visible=not insufficient, value=None),
        gr.update(visible=not insufficient, value=None),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
    )


def on_topic_change(topic_label: str):
    if not topic_label:
        return gr.update(visible=False, value=None)
    topic_code = next((code for label, code in TOPIC_OPTIONS if label == topic_label), "")
    return gr.update(visible=topic_requires_subcategory(topic_code), value=None)


def on_save_next(
    state: dict,
    show_secondary: bool,
    intent_label: str,
    topic_label: str,
    sentiment_label: str,
    subcategory_label: str,
    secondary_intent_label: str,
):
    state, error = save_and_advance_row(
        state,
        intent_label=intent_label or "",
        topic_label=topic_label or "",
        sentiment_label=sentiment_label or "",
        subcategory_label=subcategory_label or "",
        secondary_intent_label=secondary_intent_label or "",
    )
    if error:
        return _pack_response(state, show_secondary, error)
    return _pack_response(state, False, SAVE_OK_MESSAGE)


def on_save_only(
    state: dict,
    show_secondary: bool,
    intent_label: str,
    topic_label: str,
    sentiment_label: str,
    subcategory_label: str,
    secondary_intent_label: str,
):
    state, error = save_current_row(
        state,
        intent_label=intent_label or "",
        topic_label=topic_label or "",
        sentiment_label=sentiment_label or "",
        subcategory_label=subcategory_label or "",
        secondary_intent_label=secondary_intent_label or "",
    )
    if error:
        return _pack_response(state, show_secondary, error)
    return _pack_response(state, show_secondary, SAVE_OK_MESSAGE)


def on_prev(state: dict, show_secondary: bool):
    state = go_previous_row(state)
    return _pack_response(state, False, "")


def on_show_secondary(_show_secondary: bool):
    return True, gr.update(visible=True)


def on_finalize(state: dict, show_secondary: bool):
    if not is_review_complete(state):
        return _pack_response(
            state,
            show_secondary,
            "أكمل جميع التعليقات الـ40 قبل إنهاء المراجعة.",
        )
    _, _, _, report_text = finalize_calibration_review(state)
    return _pack_response(
        state,
        show_secondary,
        "تم إنشاء ملفات المراجعة والتقرير.\n\n" + report_text,
    )


def build_app(initial_state: dict) -> gr.Blocks:
    with gr.Blocks(css=RTL_CSS, title="معالج تصنيف العيّنة — B.2") as demo:
        gr.Markdown(
            "# معالج تصنيف العيّنة (40 تعليق)\n"
            "التصنيف البشري هو المرجع النهائي. توقعات Gemini مخفية عنك لتجنب التحيز."
        )
        state = gr.State(initial_state)
        show_secondary = gr.State(False)

        progress = gr.Markdown(elem_id="progress_box")
        status = gr.Markdown()
        comment = gr.Textbox(
            label="نص التعليق",
            lines=8,
            interactive=False,
            elem_id="comment_box",
        )

        intent = gr.Radio(
            choices=INTENT_LABELS,
            label="ما الغرض الأساسي من التعليق؟",
            value=None,
        )
        topic = gr.Radio(
            choices=TOPIC_LABELS,
            label="التعليق عن إيش؟",
            visible=False,
            value=None,
        )
        sentiment = gr.Radio(
            choices=SENTIMENT_LABELS,
            label="نبرة التعليق؟",
            visible=False,
            value=None,
        )
        subcategory = gr.Radio(
            choices=SUBCATEGORY_LABELS,
            label="نوع التصحيح / المشكلة",
            visible=False,
            value=None,
        )

        with gr.Row():
            toggle_secondary = gr.Button("هل توجد نية ثانية واضحة؟")
        secondary = gr.Radio(
            choices=SECONDARY_LABELS,
            label="النية الثانية",
            visible=False,
            value=None,
        )

        with gr.Row():
            prev_btn = gr.Button("السابق")
            save_btn = gr.Button("حفظ")
            next_btn = gr.Button("حفظ والتالي", variant="primary")
            finalize_btn = gr.Button("إنهاء المراجعة وإنشاء التقرير")

        ui_outputs = [
            progress,
            comment,
            intent,
            topic,
            sentiment,
            subcategory,
            secondary,
            status,
        ]
        state_outputs = [state, show_secondary]
        outputs = ui_outputs + state_outputs

        demo.load(
            fn=lambda s, ss: _pack_response(s, ss),
            inputs=[state, show_secondary],
            outputs=outputs,
        )

        intent.change(
            fn=on_intent_change,
            inputs=[intent],
            outputs=[topic, sentiment, subcategory, secondary],
        )
        topic.change(fn=on_topic_change, inputs=[topic], outputs=[subcategory])

        toggle_secondary.click(
            fn=on_show_secondary,
            inputs=[show_secondary],
            outputs=[show_secondary, secondary],
        )

        prev_btn.click(fn=on_prev, inputs=[state, show_secondary], outputs=outputs)
        save_btn.click(
            fn=on_save_only,
            inputs=[state, show_secondary, intent, topic, sentiment, subcategory, secondary],
            outputs=outputs,
        )
        next_btn.click(
            fn=on_save_next,
            inputs=[state, show_secondary, intent, topic, sentiment, subcategory, secondary],
            outputs=outputs,
        )
        finalize_btn.click(
            fn=on_finalize,
            inputs=[state, show_secondary],
            outputs=outputs,
        )

    return demo


def main() -> None:
    initial_state = load_or_init_review_state()
    demo = build_app(initial_state)
    demo.launch(server_name="127.0.0.1")


if __name__ == "__main__":
    main()
