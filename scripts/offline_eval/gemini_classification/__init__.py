"""Offline evaluation harness for a Gemini-based Arabic visitor-comment classifier (candidate "gc-v1").

ISOLATED from the production pipeline: nothing here is imported by src/, no database is opened, no frozen model is touched, and no
network call is made unless a LiveGeminiClient is explicitly constructed with a written approval (see client.py) -- which this
preparation phase never does. Taxonomy values are read from config.settings (read-only) and validated for exact equality.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
