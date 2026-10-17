#!/usr/bin/env python3
"""
Launch the Arabic RTL human-labeling wizard for Phase B.3 — remaining
ai_assisted_train rows only. Shows b2-v3 Gemini suggestions as assistance;
human gold is written only on explicit save or accept-AI click.

Usage:
  PYTHONPATH='src:.' .venv/bin/python scripts/run_b3_wizard.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr  # noqa: E402

from rcs.evaluation.b3_labeling import (  # noqa: E402
    accept_ai_suggestion_row,
    ai_suggestion_to_form_labels,
    count_status_breakdown,
    finalize_b3_review,
    go_previous_b3_row,
    load_or_init_b3_review_state,
    save_and_advance_b3_row,
    save_current_b3_row,
)
from rcs.evaluation.calibration_labeling import (  # noqa: E402
    INTENT_OPTIONS,
    MANUAL_EDIT_GUIDANCE_MD,
    SECONDARY_INTENT_OPTIONS,
    SENTIMENT_OPTIONS,
    SUBCATEGORY_OPTIONS,
    TOPIC_OPTIONS,
    current_row,
    is_review_complete,
    row_to_form_labels,
    topic_requires_subcategory,
)

RTL_CSS = """
.gradio-container, .gradio-container * {
    direction: rtl;
    text-align: right;
}
.block, .form, .gap {
    gap: 0.35rem !important;
}
#progress_wrap, #progress_wrap * {
    background: transparent !important;
    color: inherit !important;
}
#ai_card_wrap, #ai_card_wrap > div, #ai_card_wrap .prose {
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
}
#status_box {
    min-height: 1rem;
    margin: 0.15rem 0 !important;
    padding: 0 !important;
    font-size: 0.9rem;
}
#comment_box label {
    font-weight: 700;
    margin-bottom: 0.15rem !important;
}
#comment_box textarea {
    direction: rtl;
    text-align: right;
    font-size: 1.2rem !important;
    line-height: 1.7 !important;
    min-height: 6rem !important;
    max-height: 10rem !important;
    padding: 0.9rem 1rem !important;
    border-radius: 10px !important;
    background: #111827 !important;
    color: #f9fafb !important;
    border: 1px solid #374151 !important;
}
#decision_row {
    margin: 0.5rem 0 !important;
    gap: 0.5rem !important;
}
#decision_row button {
    flex: 1 1 0;
    min-height: 3.1rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 0.55rem 0.65rem !important;
}
#decision_row button.primary {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #1d4ed8 !important;
}
#decision_row button.secondary {
    background: #334155 !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
}
#decision_row button.stop {
    background: #7f1d1d !important;
    color: #fef2f2 !important;
    border: 1px solid #991b1b !important;
}
#nav_row {
    margin-top: 0.35rem !important;
    gap: 0.45rem !important;
}
#nav_row button {
    min-height: 2.5rem !important;
    font-weight: 600 !important;
}
#manual_form_col {
    margin: 0.35rem 0 0 0 !important;
    padding: 0.5rem 0.65rem !important;
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
    background: #111827 !important;
}
#manual_form_col .wrap, #manual_form_col fieldset {
    gap: 0.25rem !important;
    padding: 0.35rem 0 !important;
}
#manual_form_col label span {
    font-size: 0.95rem !important;
}
#finalize_row {
    margin-top: 0.35rem !important;
}
"""

INTENT_LABELS = [label for label, _ in INTENT_OPTIONS]
TOPIC_LABELS = [label for label, _ in TOPIC_OPTIONS]
SENTIMENT_LABELS = [label for label, _ in SENTIMENT_OPTIONS]
SUBCATEGORY_LABELS = [label for label, _ in SUBCATEGORY_OPTIONS]
SECONDARY_LABELS = [label for label, _ in SECONDARY_INTENT_OPTIONS]
INSUFFICIENT_LABEL = next(label for label, code in INTENT_OPTIONS if code == "INSUFFICIENT_CONTEXT")
SAVE_OK_MESSAGE = "تم الحفظ."
ACCEPT_OK_MESSAGE = "تم اعتماد اقتراح الذكاء وحفظه."
INSUFFICIENT_OK_MESSAGE = "تم حفظ: النص غير كافٍ للحكم."

RENDER_OUTPUT_COUNT = 9
HANDLER_OUTPUT_COUNT = 13


def _progress_html(state: dict) -> str:
    stats = count_status_breakdown(state)
    position = state["current_index"] + 1
    total = stats["total"]
    completed = stats["completed"]
    pct = int(round((completed / total) * 100)) if total else 0
    return (
        '<div id="progress_panel" style="margin:0 0 0.4rem 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'color:#e5e7eb;font-size:0.92rem;font-weight:600;margin-bottom:0.35rem;">'
        f'<span>التعليق {position} من {total}</span>'
        f'<span>مكتمل {completed} | متبقٍ {stats["remaining"]}</span>'
        "</div>"
        '<div style="background:#374151;border-radius:999px;height:8px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:#3b82f6;border-radius:999px;"></div>'
        "</div>"
        "</div>"
    )


def _ai_suggestion_card_html(row: dict) -> str:
    labels = ai_suggestion_to_form_labels(row)
    conf = html.escape(row.get("ai_suggested_confidence", "") or "—")

    def _cell(label: str, value: str) -> str:
        safe_label = html.escape(label)
        safe_value = html.escape(value or "—")
        return (
            '<div style="display:flex;flex-wrap:wrap;gap:0.35rem 0.55rem;align-items:baseline;">'
            f'<span style="color:#94a3b8;font-size:0.84rem;">{safe_label}</span>'
            f'<span style="color:#ffffff;font-weight:700;font-size:0.96rem;">{safe_value}</span>'
            "</div>"
        )

    return (
        '<div id="ai_card" style="background:#0f172a;color:#f1f5f9;'
        'border:1px solid #475569;border-radius:10px;padding:0.7rem 0.9rem;'
        'line-height:1.45;margin:0.25rem 0 0.45rem 0;">'
        '<div style="font-weight:700;font-size:0.92rem;color:#93c5fd;margin-bottom:0.45rem;">'
        "اقتراح Gemini (b2-v3) — للمساعدة فقط"
        "</div>"
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.3rem 0.9rem;">'
        f"{_cell('الغرض', labels['intent_label'])}"
        f"{_cell('الموضوع', labels['topic_label'])}"
        f"{_cell('النبرة', labels['sentiment_label'])}"
        f"{_cell('الثقة', conf)}"
        "</div>"
        '<div style="margin-top:0.45rem;font-size:0.78rem;color:#94a3b8;">'
        "لن يُنسخ إلى الذهب إلا بالضغط على «اعتماد اقتراح الذكاء»."
        "</div>"
        "</div>"
    )


def _field_update(*, visible: bool, value: str | None = None):
    payload: dict = {"visible": visible}
    if value is not None:
        payload["value"] = value
    return gr.update(**payload)


def _render_row(state: dict, show_secondary: bool, show_manual_form: bool):
    row = current_row(state)
    labels = row_to_form_labels(row)
    topic_visible = labels["intent_label"] not in ("", INSUFFICIENT_LABEL)
    topic_code = next(
        (code for label, code in TOPIC_OPTIONS if label == labels["topic_label"]),
        "",
    )
    subcategory_visible = topic_visible and topic_requires_subcategory(topic_code)
    form_visible = show_manual_form
    return (
        _progress_html(state),
        row["comment_text"],
        _ai_suggestion_card_html(row),
        gr.update(visible=form_visible),
        _field_update(visible=form_visible, value=labels["intent_label"] or None),
        _field_update(
            visible=form_visible and topic_visible,
            value=labels["topic_label"] or None,
        ),
        _field_update(
            visible=form_visible and topic_visible,
            value=labels["sentiment_label"] or None,
        ),
        _field_update(
            visible=form_visible and subcategory_visible,
            value=labels["subcategory_label"] or None,
        ),
        _field_update(
            visible=form_visible and show_secondary,
            value=labels["secondary_intent_label"] or None,
        ),
    )


def _pack_response(
    state: dict,
    show_secondary: bool,
    show_manual_form: bool,
    status: str = "",
):
    if not status and is_review_complete(state):
        stats = count_status_breakdown(state)
        status = (
            f"اكتملت جميع التعليقات ({stats['total']}). "
            "يمكنك إنهاء المراجعة لاستيراد الذهب إلى الملف الأساسي."
        )
    return _render_row(state, show_secondary, show_manual_form) + (
        status,
        state,
        show_secondary,
        show_manual_form,
    )


def on_intent_change(intent_label: str, show_manual_form: bool):
    insufficient = intent_label == INSUFFICIENT_LABEL
    show = show_manual_form and not insufficient
    return (
        _field_update(visible=show, value=None),
        _field_update(visible=show, value=None),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
    )


def on_topic_change(topic_label: str, show_manual_form: bool):
    if not topic_label or not show_manual_form:
        return gr.update(visible=False, value=None)
    topic_code = next((code for label, code in TOPIC_OPTIONS if label == topic_label), "")
    return gr.update(visible=topic_requires_subcategory(topic_code), value=None)


def on_show_manual_form(state: dict, show_secondary: bool, _show_manual_form: bool):
    return _pack_response(state, show_secondary, True, "")


def _resolve_form_labels(
    state: dict,
    intent_label: str,
    topic_label: str,
    sentiment_label: str,
    subcategory_label: str,
    secondary_intent_label: str,
) -> tuple[str, str, str, str, str]:
    """Use saved row gold when Gradio radios are empty (e.g. after Accept AI)."""
    saved = row_to_form_labels(current_row(state))
    return (
        intent_label or saved["intent_label"] or "",
        topic_label or saved["topic_label"] or "",
        sentiment_label or saved["sentiment_label"] or "",
        subcategory_label or saved["subcategory_label"] or "",
        secondary_intent_label or saved["secondary_intent_label"] or "",
    )


def on_save_next(
    state: dict,
    show_secondary: bool,
    show_manual_form: bool,
    intent_label: str,
    topic_label: str,
    sentiment_label: str,
    subcategory_label: str,
    secondary_intent_label: str,
):
    intent_label, topic_label, sentiment_label, subcategory_label, secondary_intent_label = (
        _resolve_form_labels(
            state,
            intent_label,
            topic_label,
            sentiment_label,
            subcategory_label,
            secondary_intent_label,
        )
    )
    state, error = save_and_advance_b3_row(
        state,
        intent_label=intent_label,
        topic_label=topic_label,
        sentiment_label=sentiment_label,
        subcategory_label=subcategory_label,
        secondary_intent_label=secondary_intent_label,
    )
    if error:
        return _pack_response(state, show_secondary, show_manual_form, error)
    return _pack_response(state, False, False, SAVE_OK_MESSAGE)


def on_save_only(
    state: dict,
    show_secondary: bool,
    show_manual_form: bool,
    intent_label: str,
    topic_label: str,
    sentiment_label: str,
    subcategory_label: str,
    secondary_intent_label: str,
):
    intent_label, topic_label, sentiment_label, subcategory_label, secondary_intent_label = (
        _resolve_form_labels(
            state,
            intent_label,
            topic_label,
            sentiment_label,
            subcategory_label,
            secondary_intent_label,
        )
    )
    state, error = save_current_b3_row(
        state,
        intent_label=intent_label,
        topic_label=topic_label,
        sentiment_label=sentiment_label,
        subcategory_label=subcategory_label,
        secondary_intent_label=secondary_intent_label,
    )
    if error:
        return _pack_response(state, show_secondary, show_manual_form, error)
    return _pack_response(state, show_secondary, show_manual_form, SAVE_OK_MESSAGE)


def on_insufficient(state: dict, show_secondary: bool, _show_manual_form: bool):
    state, error = save_and_advance_b3_row(
        state,
        intent_label=INSUFFICIENT_LABEL,
    )
    if error:
        return _pack_response(state, show_secondary, False, error)
    return _pack_response(state, False, False, INSUFFICIENT_OK_MESSAGE)


def on_accept_ai(state: dict, show_secondary: bool, _show_manual_form: bool):
    state, error = accept_ai_suggestion_row(state)
    if error:
        return _pack_response(state, show_secondary, False, error)
    return _pack_response(state, False, False, ACCEPT_OK_MESSAGE)


def on_prev(state: dict, _show_secondary: bool, _show_manual_form: bool):
    state = go_previous_b3_row(state)
    return _pack_response(state, False, False, "")


def on_show_secondary(show_secondary: bool, show_manual_form: bool):
    return True, gr.update(visible=show_manual_form)


def on_finalize(state: dict, show_secondary: bool, show_manual_form: bool):
    if not is_review_complete(state):
        stats = count_status_breakdown(state)
        return _pack_response(
            state,
            show_secondary,
            show_manual_form,
            f"أكمل جميع التعليقات قبل الإنهاء. متبقٍ: {stats['remaining']}.",
        )
    _, _, summary = finalize_b3_review(state)
    return _pack_response(state, show_secondary, show_manual_form, summary)


def build_app(initial_state: dict) -> gr.Blocks:
    with gr.Blocks(css=RTL_CSS, title="معالج مراجعة B.3", fill_height=False) as demo:
        state = gr.State(initial_state)
        show_secondary = gr.State(False)
        show_manual_form = gr.State(False)

        progress = gr.HTML(elem_id="progress_wrap")
        comment = gr.Textbox(
            label="نص التعليق",
            lines=4,
            max_lines=5,
            interactive=False,
            elem_id="comment_box",
            show_copy_button=False,
        )
        ai_panel = gr.HTML(elem_id="ai_card_wrap")

        with gr.Row(elem_id="decision_row"):
            accept_ai_btn = gr.Button("✓ اعتماد اقتراح الذكاء", elem_classes=["primary"])
            edit_btn = gr.Button("✎ تعديل التصنيف", elem_classes=["secondary"])
            insufficient_btn = gr.Button("؟ النص غير كافٍ للحكم", elem_classes=["stop"])

        with gr.Column(visible=False, elem_id="manual_form_col") as manual_form:
            gr.Markdown("**تعديل التصنيف**")
            gr.Markdown(MANUAL_EDIT_GUIDANCE_MD, elem_id="manual_edit_guidance")
            intent = gr.Radio(
                choices=INTENT_LABELS,
                label="ما الغرض الأساسي من التعليق؟",
                visible=False,
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
            toggle_secondary = gr.Button("هل توجد نية ثانية واضحة؟", size="sm")
            secondary = gr.Radio(
                choices=SECONDARY_LABELS,
                label="النية الثانية",
                visible=False,
                value=None,
            )

        status = gr.Markdown(elem_id="status_box")
        with gr.Row(elem_id="nav_row"):
            prev_btn = gr.Button("السابق")
            save_btn = gr.Button("حفظ")
            next_btn = gr.Button("حفظ والتالي", variant="primary")

        with gr.Row(elem_id="finalize_row"):
            finalize_btn = gr.Button("إنهاء المراجعة واستيراد الذهب", size="sm")

        ui_outputs = [
            progress,
            comment,
            ai_panel,
            manual_form,
            intent,
            topic,
            sentiment,
            subcategory,
            secondary,
            status,
        ]
        state_outputs = [state, show_secondary, show_manual_form]
        outputs = ui_outputs + state_outputs

        demo.load(
            fn=lambda s, ss, sm: _pack_response(s, ss, sm),
            inputs=[state, show_secondary, show_manual_form],
            outputs=outputs,
        )

        edit_btn.click(
            fn=on_show_manual_form,
            inputs=[state, show_secondary, show_manual_form],
            outputs=outputs,
        )

        intent.change(
            fn=on_intent_change,
            inputs=[intent, show_manual_form],
            outputs=[topic, sentiment, subcategory, secondary],
        )
        topic.change(
            fn=on_topic_change,
            inputs=[topic, show_manual_form],
            outputs=[subcategory],
        )

        toggle_secondary.click(
            fn=on_show_secondary,
            inputs=[show_secondary, show_manual_form],
            outputs=[show_secondary, secondary],
        )

        accept_ai_btn.click(
            fn=on_accept_ai,
            inputs=[state, show_secondary, show_manual_form],
            outputs=outputs,
        )
        insufficient_btn.click(
            fn=on_insufficient,
            inputs=[state, show_secondary, show_manual_form],
            outputs=outputs,
        )
        prev_btn.click(
            fn=on_prev,
            inputs=[state, show_secondary, show_manual_form],
            outputs=outputs,
        )
        save_btn.click(
            fn=on_save_only,
            inputs=[
                state,
                show_secondary,
                show_manual_form,
                intent,
                topic,
                sentiment,
                subcategory,
                secondary,
            ],
            outputs=outputs,
        )
        next_btn.click(
            fn=on_save_next,
            inputs=[
                state,
                show_secondary,
                show_manual_form,
                intent,
                topic,
                sentiment,
                subcategory,
                secondary,
            ],
            outputs=outputs,
        )
        finalize_btn.click(
            fn=on_finalize,
            inputs=[state, show_secondary, show_manual_form],
            outputs=outputs,
        )

    return demo


def main() -> None:
    initial_state = load_or_init_b3_review_state()
    stats = count_status_breakdown(initial_state)
    if stats["total"] == 0:
        raise SystemExit("No pending ai_assisted_train rows — B.3 review may already be complete.")
    demo = build_app(initial_state)
    demo.launch(server_name="127.0.0.1")


if __name__ == "__main__":
    main()
