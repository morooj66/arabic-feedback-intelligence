"""
Outbound send boundary.

Response generation must not import dictionary platform APIs.
A later connector implements `OutboundSender` without changing this layer.
"""

from __future__ import annotations

from typing import Protocol


class OutboundSender(Protocol):
    def send_approved_response(self, comment_id: str, response_text: str) -> None:
        """Deliver an already-approved reply. Not called by generation."""


class UnconfiguredOutboundSender:
    """Placeholder until the real visitor-channel contract is known."""

    def send_approved_response(self, comment_id: str, response_text: str) -> None:
        raise NotImplementedError(
            "Visitor send connector is not implemented. Response generation "
            "does not send. Wire a dictionary platform adapter later."
        )


def automatic_ack_allowed() -> bool:
    from config import settings

    return bool(settings.AUTOMATIC_ACK_ENABLED)
