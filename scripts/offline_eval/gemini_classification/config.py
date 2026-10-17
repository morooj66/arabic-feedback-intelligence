"""Generation settings, expressed exactly as the Interactions API accepts them (verified against the official docs on 2026-10-16).

Determinism notes (from the official documentation):
* temperature: Google recommends leaving Gemini 3 models at the default 1.0 and warns that lowering it can degrade results, so it is NOT set;
* seed: a documented generation_config field; reproducibility is best-effort, which is why a separate stability stage measures repeat agreement;
* thinking_level: minimal/low/medium/high for gemini-3.6-flash; thinking tokens are billed as OUTPUT, so a fixed low level bounds cost;
* max_output_tokens caps output (and therefore the worst-case cost of one call).
"""
from __future__ import annotations

from dataclasses import dataclass

MODEL_ID = "gemini-3.6-flash"


@dataclass(frozen=True)
class GenerationSettings:
    model_id: str = MODEL_ID
    seed: int = 20261016
    thinking_level: str = "minimal"
    max_output_tokens: int = 512
    temperature: float | None = None  # None = provider default (1.0), per Google's guidance for Gemini 3

    def to_generation_config(self) -> dict:
        cfg: dict = {"seed": self.seed, "thinking_level": self.thinking_level, "max_output_tokens": self.max_output_tokens}
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        return cfg

    def validate(self) -> None:
        if self.thinking_level not in ("minimal", "low", "medium", "high"):
            raise ValueError("thinking_level must be minimal, low, medium or high")
        if not (16 <= self.max_output_tokens <= 4096):
            raise ValueError("max_output_tokens out of the expected range")
