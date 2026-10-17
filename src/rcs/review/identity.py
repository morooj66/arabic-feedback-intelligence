"""Provider-agnostic reviewer identity. No passwords. Entra ID can map later."""

from __future__ import annotations

import os
from dataclasses import dataclass


class DemoIdentityDisabled(RuntimeError):
    """Raised when demo-session identity is requested in a production env."""


@dataclass(frozen=True)
class ReviewerIdentity:
    reviewer_id: str
    display_name: str | None = None
    # Phase 6: the authenticated users.user_id, when this identity came
    # from a real login. None for pre-Phase-6/demo identities — never
    # backfilled onto historical rows. See ReviewItem.reviewer_user_id.
    user_id: str | None = None
    # ADMIN / REVIEWER for an authenticated user; recorded on audit events.
    role: str | None = None

    def __post_init__(self) -> None:
        if not str(self.reviewer_id).strip():
            raise ValueError("reviewer_id is required")


def is_production_env() -> bool:
    return os.environ.get("RCS_ENV", "").strip().lower() in {"production", "prod"}


def demo_microsoft_identity() -> ReviewerIdentity:
    """Development/demo Microsoft-button identity only. Not a production account."""
    if is_production_env():
        raise DemoIdentityDisabled("demo_identity_disabled")
    # Phase 4: no hardcoded personal name in the demo identity default —
    # a neutral placeholder until Phase 6 real accounts exist. Still
    # overridable via env var for anyone who wants a named local demo.
    reviewer_id = os.environ.get("RCS_DEMO_REVIEWER_ID", "demo:reviewer").strip() or "demo:reviewer"
    name = os.environ.get("RCS_DEMO_REVIEWER_NAME", "مراجع").strip() or "مراجع"
    return ReviewerIdentity(reviewer_id, name)


def migrate_demo_session_identity(ident: ReviewerIdentity) -> ReviewerIdentity:
    """Refresh a stale development Microsoft-demo cookie name. Production is unchanged."""
    if is_production_env():
        return ident
    try:
        current = demo_microsoft_identity()
    except DemoIdentityDisabled:
        return ident
    if ident.reviewer_id != current.reviewer_id:
        return ident
    if ident.display_name == current.display_name:
        return ident
    return current
