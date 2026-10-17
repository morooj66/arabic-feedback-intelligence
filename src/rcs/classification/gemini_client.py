"""
Thin wrapper around the Gemini structured-output call. This is the ONLY
file in the project that talks to the Gemini API.

`google.generativeai` is imported lazily, inside the method that actually
needs it — not at module import time. That means:
  - importing this module never requires the package to be installed or
    an API key to be set (tests never trigger it, since they inject a
    fake client instead of constructing GeminiClient), and
  - constructing a GeminiClient without ever calling `.generate()` is
    always safe, e.g. for wiring/config checks.

No retry logic lives here — retries are the caller's (classifier.py)
concern, so this class stays a single, simple, mockable seam: one method,
one job, one real network call per invocation.
"""

from __future__ import annotations


class GeminiClient:
    def __init__(self, api_key: str | None, model_name: str):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Set it in the environment (see .env.example) "
                "before constructing a live GeminiClient."
            )
        self._api_key = api_key
        self._model_name = model_name
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is None:
            import google.generativeai as genai  # lazy import, see module docstring

            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self._model_name)
        return self._model

    def generate(self, prompt: str, response_schema: dict) -> str:
        """
        Sends one structured-output request. Returns the raw text of the
        model's response (expected to be a JSON string per
        response_mime_type="application/json"). Raises on any
        network/API-level failure — callers are responsible for catching
        and falling back safely (see classifier.py).
        """
        model = self._ensure_model()
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            },
        )
        return response.text
