"""
Phase M5F — offline evaluation harness for the Response Agent
(rcs.response_intelligence.orchestrator.run_agent_pipeline).

OFFLINE EVALUATION — NOT PRODUCTION DATA. Everything under this package
is read-only with respect to production: it never touches a real
database, never calls the real Gemini API by default, and never writes
to review_items/review_sessions/processing_jobs/agent_execution_traces.

Not imported by the served application. Isolated the same way
scripts/benchmarks/retrieval/ was in M5B.2 — a self-contained script
tree, not an installed package.
"""
