"""Runs a ClassifierClient over evaluation records with privacy preflight, bounded retries, strict validation, token/cost accounting and hard caps.

Never opens a database, never writes anywhere except an optional call log (ids, hashes, outcomes and counts only -- no comment text).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .client import ClassifierClient, ClientError
from .config import GenerationSettings
from .pricing import PricingTable, Usage
from .privacy import LiveApproval, OutboundPrivacyError, egress_guard, preflight, require_provenance
from .prompt import PROMPT_VERSION, build_prompt
from .schema import RESPONSE_SCHEMA, Classification, SchemaError, parse_and_validate

OK, SCHEMA_FAILURE, API_FAILURE, PRIVACY_BLOCKED, BUDGET_STOPPED = "OK", "SCHEMA_FAILURE", "API_FAILURE", "PRIVACY_BLOCKED", "BUDGET_STOPPED"


@dataclass(frozen=True)
class Record:
    record_id: str
    text: str
    provenance: str = "synthetic"      # "synthetic" | "real"
    dataset: str = ""


@dataclass
class CallOutcome:
    record_id: str
    status: str
    prediction: Classification | None = None
    failure_code: str | None = None
    attempts: int = 0
    latency_seconds: float = 0.0
    usage: Usage = field(default_factory=lambda: Usage(0, 0, 0))
    cost_usd: float = 0.0


@dataclass(frozen=True)
class Budget:
    max_calls: int
    max_usd: float


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RunSummary:
    outcomes: list[CallOutcome]
    calls_made: int
    total_cost_usd: float
    stopped_by_budget: bool

    def counts(self) -> dict:
        c: dict = {}
        for o in self.outcomes:
            c[o.status] = c.get(o.status, 0) + 1
        return c


class ClassificationRunner:
    def __init__(
        self,
        client: ClassifierClient,
        settings: GenerationSettings | None = None,
        *,
        budget: Budget,
        pricing: PricingTable | None = None,
        approval: LiveApproval | None = None,
        max_api_retries: int = 2,
        max_schema_retries: int = 1,
        backoff_base_seconds: float = 0.5,
        backoff_max_seconds: float = 4.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        today: dt.date | None = None,
        call_log: Path | None = None,
    ):
        self.client = client
        self.settings = settings or GenerationSettings()
        self.settings.validate()
        self.budget = budget
        self.pricing = pricing or PricingTable()
        self.approval = approval
        self.max_api_retries = max_api_retries
        self.max_schema_retries = max_schema_retries
        self.backoff_base = backoff_base_seconds
        self.backoff_max = backoff_max_seconds
        self._sleep = sleep
        self._clock = clock
        self._today = today
        self.call_log = call_log
        self.calls_made = 0
        self.total_cost = 0.0

    # ----------------------------------------------------------------------------------------------------------------- helpers
    def _check_budget(self) -> None:
        if self.calls_made >= self.budget.max_calls:
            raise BudgetExceeded("call cap reached")
        if self.total_cost >= self.budget.max_usd:
            raise BudgetExceeded("cost cap reached")

    def _log(self, rec: Record, outcome: CallOutcome) -> None:
        if self.call_log is None:
            return
        line = {
            "record_id": rec.record_id,
            "text_sha256": hashlib.sha256(rec.text.encode("utf-8")).hexdigest()[:16],
            "status": outcome.status, "failure_code": outcome.failure_code, "attempts": outcome.attempts,
            "latency_s": round(outcome.latency_seconds, 3), "input_tokens": outcome.usage.input_tokens,
            "output_tokens": outcome.usage.output_tokens, "thought_tokens": outcome.usage.thought_tokens,
            "cost_usd": round(outcome.cost_usd, 6), "prompt_version": PROMPT_VERSION, "model": self.settings.model_id, "seed": self.settings.seed,
        }
        self.call_log.parent.mkdir(parents=True, exist_ok=True)
        with self.call_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _backoff(self, attempt: int) -> float:
        return min(self.backoff_max, self.backoff_base * (2 ** attempt))

    # ----------------------------------------------------------------------------------------------------------------- one record
    def classify_one(self, rec: Record) -> CallOutcome:
        require_provenance(rec.provenance, self.approval, rec.dataset)
        pre = preflight(rec.text)
        if not pre.eligible:
            out = CallOutcome(rec.record_id, PRIVACY_BLOCKED, failure_code=pre.blocked_reason)
            self._log(rec, out)
            return out
        try:
            prompt = egress_guard(build_prompt(pre.text))
        except OutboundPrivacyError as exc:
            out = CallOutcome(rec.record_id, PRIVACY_BLOCKED, failure_code=f"egress:{exc.reason}")
            self._log(rec, out)
            return out

        out = CallOutcome(rec.record_id, API_FAILURE)
        api_failures = schema_failures = 0
        while True:
            self._check_budget()
            start = self._clock()
            try:
                self.calls_made += 1
                resp = self.client.classify(prompt, RESPONSE_SCHEMA, self.settings)
            except ClientError as exc:
                out.attempts += 1
                out.latency_seconds += self._clock() - start
                out.failure_code = exc.code
                if exc.retryable and api_failures < self.max_api_retries:
                    self._sleep(self._backoff(api_failures))
                    api_failures += 1
                    continue
                out.status = API_FAILURE
                break
            out.attempts += 1
            out.latency_seconds += self._clock() - start
            out.usage = Usage(out.usage.input_tokens + resp.usage.input_tokens, out.usage.output_tokens + resp.usage.output_tokens,
                              out.usage.thought_tokens + resp.usage.thought_tokens, resp.usage.estimated)
            call_cost = self.pricing.cost(resp.usage, self._today)
            out.cost_usd += call_cost
            self.total_cost += call_cost
            try:
                out.prediction = parse_and_validate(resp.text)
                out.status, out.failure_code = OK, None
                break
            except SchemaError as exc:
                out.failure_code = exc.code
                if schema_failures < self.max_schema_retries:
                    schema_failures += 1
                    continue
                out.status = SCHEMA_FAILURE
                break
        self._log(rec, out)
        return out

    # ----------------------------------------------------------------------------------------------------------------- many records
    def run(self, records: Sequence[Record]) -> RunSummary:
        outcomes: list[CallOutcome] = []
        stopped = False
        for rec in records:
            try:
                outcomes.append(self.classify_one(rec))
            except BudgetExceeded:
                stopped = True
                outcomes.append(CallOutcome(rec.record_id, BUDGET_STOPPED, failure_code="budget"))
                continue
        return RunSummary(outcomes, self.calls_made, self.total_cost, stopped)
