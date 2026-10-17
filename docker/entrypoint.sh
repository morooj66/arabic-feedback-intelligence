#!/bin/sh
# Fetches the frozen private HF artifacts (checksum-verified, safe to
# re-run — see scripts/fetch_frozen_models.py) before starting the
# unmodified server. Fails the container start (non-zero exit) on any
# missing/corrupt/checksum-mismatched artifact, rather than serving a
# broken deployment.
set -e

python scripts/fetch_frozen_models.py
exec python scripts/serve_reviewer_api.py "$@"
