"""Offline stand-in for the LLM used ONLY in demo mode: deterministic, local, no network. Returns short, generic, safe Arabic acknowledgements chosen by simple keyword
rules from the visitor comment found in the prompt. It never promises anything, never contains links/contacts and never invents facts (the normal validation gates
still run on its output; note the repository's own rule of at most one courtesy marker per reply)."""

from __future__ import annotations

import json
import re

from rcs.demo.classifier import intent_of

_COMMENT = re.compile(r"--- UNTRUSTED VISITOR COMMENT[^\n]*---\n(.*?)\n--- END UNTRUSTED VISITOR COMMENT", re.S)
REPLIES = {
    "THANKS_PRAISE": "نشكركم على كلماتكم الطيبة، ويسعدنا أن يكون المعجم مفيدًا لكم.",
    "COMPLAINT": "نشكركم على تنبيهكم إلى هذه الملاحظة المتعلقة بالمعجم.",
    "SUGGESTION": "نشكركم على اقتراحكم الذي يخص تطوير المعجم.",
    "INQUIRY": "نشكركم على سؤالكم وتواصلكم مع المعجم.",
}
DEFAULT = "شكرًا لكم على ملاحظتكم."


class DemoProvider:
    calls = 0

    def generate(self, prompt: str, response_schema: dict) -> str:
        DemoProvider.calls += 1
        m = _COMMENT.search(prompt)
        intent = intent_of(m.group(1))[0] if m else None
        return json.dumps({"suggested_response": REPLIES.get(intent, DEFAULT)}, ensure_ascii=False)
