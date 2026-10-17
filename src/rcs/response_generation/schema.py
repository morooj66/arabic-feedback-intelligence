"""Structured contracts for response generation (not classification)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

from config import settings

RoutingDecision = Literal["GENERATE_DRAFT", "REVIEW_ONLY", "NO_RESPONSE_RECOMMENDED"]
GenerationStatus = Literal["SKIPPED", "SUCCEEDED", "FAILED_REVIEW_ONLY"]

assert set(RoutingDecision.__args__) == set(settings.RESPONSE_ELIGIBILITY_DECISIONS)


class GeminiDraftResponse(BaseModel):
    """What Gemini is allowed to return. Routing is never in this payload."""

    model_config = {"extra": "forbid"}

    suggested_response: str = Field(min_length=1)


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "suggested_response": {
            "type": "string",
            "description": "Arabic visitor-facing draft only. No JSON keys besides this. No internal terms.",
        },
    },
    "required": ["suggested_response"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class GenerationResult:
    comment_id: str
    routing_decision: str
    generation_status: str
    suggested_response: Optional[str]
    review_required: bool
    reason_code: str
    provider: Optional[str]
    model_name: Optional[str]
    prompt_version: Optional[str]
    generated_at: str
    future_reviewer_actions: tuple[str, ...] = field(default_factory=tuple)
    eligibility_rule_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "comment_id": self.comment_id,
            "routing_decision": self.routing_decision,
            "generation_status": self.generation_status,
            "suggested_response": self.suggested_response,
            "review_required": self.review_required,
            "reason_code": self.reason_code,
            "provider": self.provider,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "generated_at": self.generated_at,
            "future_reviewer_actions": list(self.future_reviewer_actions),
            "eligibility_rule_id": self.eligibility_rule_id,
            "error": self.error,
        }
