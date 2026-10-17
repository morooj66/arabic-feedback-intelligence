"""Token and cost accounting from the verified price sheet (pricing.json). Thinking tokens are billed as output tokens."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

_PRICE_FILE = Path(__file__).with_name("pricing.json")


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    thought_tokens: int = 0
    estimated: bool = False  # True when the tokens were estimated locally rather than reported by the API


class PricingTable:
    def __init__(self, data: dict | None = None):
        self.data = data if data is not None else json.loads(_PRICE_FILE.read_text(encoding="utf-8"))
        self.unit = float(self.data["unit_tokens"])

    def _tier(self, on: dt.date) -> dict:
        for t in self.data["tiers"]:
            if dt.date.fromisoformat(t["valid_from"]) <= on <= dt.date.fromisoformat(t["valid_to"]):
                return t
        raise ValueError(f"no price tier covers {on}")

    def cost(self, usage: Usage, on: dt.date | None = None) -> float:
        t = self._tier(on or dt.date.today())
        billed_output = usage.output_tokens + usage.thought_tokens
        return usage.input_tokens / self.unit * t["input_per_unit"] + billed_output / self.unit * t["output_per_unit"]

    def worst_case(self, calls: int, input_tokens_per_call: int, max_output_tokens_per_call: int, on: dt.date | None = None) -> float:
        """Upper bound: every call uses the full output cap (thinking + answer) -- the real figure is lower."""
        return calls * self.cost(Usage(input_tokens_per_call, max_output_tokens_per_call, 0), on)


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Local estimate used only for mock runs (measured on the 10-call smoke test: about 4.3 characters per token for this prompt); real usage comes from the API."""
    return max(1, int(len(text) / chars_per_token) + 1)
