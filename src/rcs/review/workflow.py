"""Explicit review-item state machine. Invalid transitions fail closed."""

from __future__ import annotations

from dataclasses import dataclass

from config import settings

# (from_status, action) -> to_status
TRANSITIONS: dict[tuple[str, str], str] = {
    ("CLASSIFIED", "apply_generate_draft"): "DRAFT_READY",
    ("CLASSIFIED", "apply_review_only"): "REVIEW_ONLY",
    ("CLASSIFIED", "apply_no_response_recommended"): "NO_RESPONSE_RECOMMENDED",
    ("UNPROCESSED", "apply_generate_draft"): "DRAFT_READY",
    ("UNPROCESSED", "apply_review_only"): "REVIEW_ONLY",
    ("UNPROCESSED", "apply_no_response_recommended"): "NO_RESPONSE_RECOMMENDED",
    ("DRAFT_READY", "claim"): "IN_REVIEW",
    ("REVIEW_ONLY", "claim"): "IN_REVIEW",
    ("IN_REVIEW", "claim"): "IN_REVIEW",
    ("DRAFT_READY", "SEND_AS_IS"): "APPROVED",
    ("IN_REVIEW", "SEND_AS_IS"): "APPROVED",
    ("DRAFT_READY", "EDIT_AND_SEND"): "EDITED_AND_APPROVED",
    ("REVIEW_ONLY", "EDIT_AND_SEND"): "EDITED_AND_APPROVED",
    ("IN_REVIEW", "EDIT_AND_SEND"): "EDITED_AND_APPROVED",
    ("DRAFT_READY", "NO_RESPONSE_NEEDED"): "NO_REPLY",
    ("REVIEW_ONLY", "NO_RESPONSE_NEEDED"): "NO_REPLY",
    ("IN_REVIEW", "NO_RESPONSE_NEEDED"): "NO_REPLY",
    ("NO_RESPONSE_RECOMMENDED", "NO_RESPONSE_NEEDED"): "NO_REPLY",
    # Reviewer override of a model's no-response recommendation: claiming
    # it (also done implicitly by EDIT_AND_SEND) moves it to IN_REVIEW.
    ("NO_RESPONSE_RECOMMENDED", "claim"): "IN_REVIEW",
    ("APPROVED", "dispatch"): "SENDING",
    ("EDITED_AND_APPROVED", "dispatch"): "SENDING",
    ("SEND_FAILED", "dispatch"): "SENDING",
    ("SENDING", "mark_sent"): "SENT",
    ("SENDING", "mark_failed"): "SEND_FAILED",
}


class WorkflowError(Exception):
    """Base workflow failure."""


class InvalidTransition(WorkflowError):
    pass


class AttestationRequired(WorkflowError):
    """The reply contains factual claims that cannot be verified against approved grounding. A permitted reviewer may send it only with an
    explicit attestation naming exactly these findings. Nothing was changed or sent."""

    def __init__(self, finding_codes: tuple[str, ...], messages_ar: tuple[str, ...] = ()):
        super().__init__("attestation_required: " + ",".join(finding_codes))
        self.finding_codes = tuple(finding_codes)
        self.messages_ar = tuple(messages_ar)


class AttestationNotPermitted(WorkflowError):
    """The acting user's role may not attest factual claims (or attestation is switched off)."""

    def __init__(self, finding_codes: tuple[str, ...] = ()):
        super().__init__("attestation_not_permitted")
        self.finding_codes = tuple(finding_codes)


class DraftValidationError(WorkflowError):
    """A reply (reviewer-edited, or a stale generated draft) failed the deterministic reply-safety gates. Nothing was changed or sent;
    the stored draft and item state are untouched and the reviewer may correct the text and submit it again (re-validation)."""

    def __init__(self, reason_code: str, message_ar: str, blocking: tuple[str, ...] = (), attestable: tuple[str, ...] = ()):
        super().__init__(f"draft_validation_failed: {reason_code}")
        self.reason_code = reason_code
        self.message_ar = message_ar
        self.blocking = tuple(blocking) or (reason_code,)
        self.attestable = tuple(attestable)


class VersionConflict(WorkflowError):
    pass


class ItemNotFound(WorkflowError):
    pass


class Forbidden(WorkflowError):
    """Raised when an authenticated REVIEWER attempts to view or act on an
    item that isn't assigned to them (including items assigned to no one
    yet) — see http_api.py's _require_item_access. Never raised for ADMIN,
    who retains global access."""

    pass


@dataclass(frozen=True)
class Transition:
    from_status: str
    action: str
    to_status: str


def next_status(current: str, action: str) -> str:
    key = (current, action)
    if key not in TRANSITIONS:
        raise InvalidTransition(f"illegal transition: {current} + {action}")
    dest = TRANSITIONS[key]
    if dest not in settings.REVIEW_STATUSES:
        raise InvalidTransition(f"unknown destination status {dest}")
    return dest
