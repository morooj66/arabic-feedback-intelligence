"""
Outbound send contract for approved replies.

Response generation still never sends. Reviewer workflow calls this only
after SEND_AS_IS / EDIT_AND_SEND. A real dictionary-platform HTTP connector is not implemented.

Future connector (not wired to production send):

  receive/import comment
    → comment_id (stable, never rewritten)
    → source_reference (workbook row / channel id)

  send approved final response
    → source_reference
    → final_response
    → idempotency_key
    → structured OutboundSendResult

Until a documented visitor-comment write API exists, UnconfiguredOutboundSender
returns CONNECTOR_NOT_CONFIGURED and review_status must stay PENDING (not SENT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from rcs.response_generation.outbound import UnconfiguredOutboundSender as _LegacyUnconfigured


@dataclass(frozen=True)
class InboundComment:
    """Import/receive boundary. Excel ingest already fills these."""

    comment_id: str
    source_reference: str | None
    comment_text: str | None = None


@dataclass(frozen=True)
class OutboundDispatchRequest:
    """Send boundary. Production adapter is not connected."""

    comment_id: str
    source_reference: str | None
    final_response: str
    idempotency_key: str


@dataclass(frozen=True)
class OutboundSendResult:
    ok: bool
    comment_id: str
    idempotency_key: str
    external_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class OutboundSender(Protocol):
    def send_response(
        self,
        comment_id: str,
        source_reference: str | None,
        final_response: str,
        idempotency_key: str,
    ) -> OutboundSendResult: ...


class UnconfiguredOutboundSender(_LegacyUnconfigured):
    """Production send is not configured. Structured result for dispatch."""

    def send_response(
        self,
        comment_id: str,
        source_reference: str | None,
        final_response: str,
        idempotency_key: str,
    ) -> OutboundSendResult:
        return OutboundSendResult(
            ok=False,
            comment_id=comment_id,
            idempotency_key=idempotency_key,
            error_code="CONNECTOR_NOT_CONFIGURED",
            error_message=(
                "dictionary platform send connector is not configured. "
                "Approved text is kept; dispatch must not drop final_response."
            ),
        )


class RecordingOutboundSender:
    """Test/mock adapter. Does not call any network."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[dict] = []

    def send_response(
        self,
        comment_id: str,
        source_reference: str | None,
        final_response: str,
        idempotency_key: str,
    ) -> OutboundSendResult:
        self.calls.append(
            {
                "comment_id": comment_id,
                "source_reference": source_reference,
                "final_response": final_response,
                "idempotency_key": idempotency_key,
            }
        )
        if self.fail:
            return OutboundSendResult(
                ok=False,
                comment_id=comment_id,
                idempotency_key=idempotency_key,
                error_code="MOCK_SEND_FAILED",
                error_message="mock adapter forced failure",
            )
        return OutboundSendResult(
            ok=True,
            comment_id=comment_id,
            idempotency_key=idempotency_key,
            external_id=f"mock-{idempotency_key[:12]}",
        )
