"""Isolated SYNTHETIC-DATA demo mode.

Everything demo-specific lives in this package. Production code imports only `rcs.demo.state` (a boolean that is False unless `rcs.demo.activation.activate()` ran, which
requires the full lock in `rcs.demo.guard`). A normal or production start never imports anything else from here."""
