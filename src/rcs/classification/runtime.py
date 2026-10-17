"""Configurable classifier composition with legacy behavior as the default."""

from __future__ import annotations

from config import settings
from rcs.classification import frozen_inference
from rcs.classification import topic_boundary_layer
from rcs.classification.jev_topic import JevTopicAdapter


class ClassificationModeError(RuntimeError):
    pass


def classification_mode() -> str:
    mode = settings.CLASSIFICATION_MODE
    if mode not in settings.CLASSIFICATION_MODES:
        raise ClassificationModeError(
            f"unsupported classification mode {mode!r}; expected one of {settings.CLASSIFICATION_MODES}"
        )
    if mode == "local_v2_pilot":
        from rcs.review.identity import is_production_env

        if is_production_env():
            raise ClassificationModeError("local_v2_pilot is a local-only pilot mode and is refused when RCS_ENV=production")
    return mode


def predict_intent(texts: list[str]) -> list[tuple[str, float]]:
    # Mode is consulted only for the opt-in pilot; legacy / hybrid behaviour (and an unset or invalid mode on this call) is exactly as before.
    if settings.CLASSIFICATION_MODE == "local_v2_pilot":
        classification_mode()
        from rcs.classification import v2_stack

        return v2_stack.get_stack().predict_intent(texts)
    return frozen_inference.predict_intent(texts)


def predict_sentiment(comment_ids: list[str], texts: list[str]) -> dict[str, tuple[str, float]]:
    if settings.CLASSIFICATION_MODE == "local_v2_pilot":
        classification_mode()
        from rcs.classification import v2_stack

        return v2_stack.get_stack().predict_sentiment(comment_ids, texts)
    return frozen_inference.predict_sentiment(comment_ids, texts)


def predict_topic(
    comment_ids: list[str],
    texts: list[str],
    *,
    adapter: JevTopicAdapter | None = None,
) -> dict[str, tuple[str, float]]:
    mode = classification_mode()
    if mode in ("legacy_rcs", "local_v2_pilot"):          # the pilot keeps the frozen MARBERTv2 topic model
        return frozen_inference.predict_topic(comment_ids, texts)
    if mode == "hybrid_jev_topic":
        provider = adapter or JevTopicAdapter(api_key=settings.OPENROUTER_API_KEY)
        return provider.predict(comment_ids, texts)
    raise ClassificationModeError(f"unreachable classification mode {mode!r}")


def apply_optional_topic_layer(
    topic_preds: dict[str, tuple[str, float]],
    comment_ids: list[str],
    texts: list[str],
    lemmas: list | None,
) -> tuple[dict[str, tuple[str, float]], bool]:
    """Opt-in (settings.TOPIC_BOUNDARY_LAYER_ENABLED, default False). When disabled the SAME object is returned untouched."""
    if not settings.TOPIC_BOUNDARY_LAYER_ENABLED:
        return topic_preds, False
    return topic_boundary_layer.apply_boundary_layer(topic_preds, comment_ids, texts, lemmas), True


def with_layer_provenance(prov: dict[str, str], layer_applied: bool) -> dict[str, str]:
    if not layer_applied:
        return prov
    suffix = topic_boundary_layer.PROVENANCE_SUFFIX
    return {**prov, "source": prov["source"] + suffix, "model_name": prov["model_name"] + suffix,
            "prompt_version": prov["prompt_version"] + suffix}


def provenance() -> dict[str, str]:
    mode = classification_mode()
    if mode == "legacy_rcs":
        return {
            "reason": "frozen_model_inference",
            "source": "FROZEN_TFIDF_LOGREG+MARBERTV2+CAMELBERT",
            "model_name": "tfidf_logistic_intent+marbertv2_topic+camelbert_sentiment",
            "prompt_version": "frozen-2026-08-25",
        }
    if mode == "local_v2_pilot":
        return {
            "reason": "experimental_local_v2_pilot_inference",
            "source": "V2_I10_INTENT+FROZEN_MARBERTV2_TOPIC+V2_S5_SENTIMENT",
            "model_name": "classification_v2_i10_intent+marbertv2_topic+classification_v2_s5_sentiment",
            "prompt_version": "classification-v2-local-pilot",
        }
    return {
        "reason": "experimental_hybrid_jev_topic_inference",
        "source": "FROZEN_TFIDF_LOGREG+JEV_TOPIC+CAMELBERT",
        "model_name": "tfidf_logistic_intent+vendor_jev_1_13_topic+camelbert_sentiment",
        "prompt_version": "hybrid-jev-topic-experimental-v1",
    }
