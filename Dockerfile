# Host-neutral production image for the Reviewer API backend.
#
# Runs the exact same application as scripts/serve_reviewer_api.py /
# render.yaml — no Oracle-specific (or any host-specific) code. Multi-arch:
# build for linux/amd64 and linux/arm64 (Oracle Ampere A1) from this same
# Dockerfile via `docker buildx build --platform linux/amd64,linux/arm64`.
# torch, transformers, and psycopg[binary] all ship prebuilt manylinux
# wheels for both architectures, so no source compilation is needed on
# either platform.
FROM python:3.13-slim

WORKDIR /app

# ca-certificates: needed for HTTPS to the Hugging Face Hub, Supabase, and
# Gemini. Nothing else — no compiler toolchain needed for arm64 or amd64.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt .
# torch from plain PyPI resolves to a CUDA-enabled build on Linux (pulling
# multi-GB nvidia-cuda-*/cudnn packages) even on this CPU-only arm64/amd64
# target. Installing the CPU-only wheel first — from PyTorch's own CPU
# index, which publishes both linux/amd64 and linux/arm64 builds — makes
# the second install a no-op for torch (already satisfied) while pulling
# every other package normally from PyPI.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements-server.txt

# Only what the running application needs — no tests, docs, training
# scripts' data, or dev tooling. See .dockerignore for the full exclusion
# list (visitor data, databases, uploads, model binaries, secrets, etc.
# are never part of the build context in the first place).
COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY migrations/ migrations/
COPY alembic.ini .
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Non-root: /app/data holds the fetched frozen models, the operational
# SQLite fallback DB, and temporary Excel uploads — all written at
# runtime, so the app's own user needs to own it. Pre-creating the
# subdirectories here (not just /app/data) matters if data/models is
# later bind-mounted at container start: Docker auto-creates a missing
# mount-point directory as root, which would otherwise silently block
# the non-root user from writing to data/db or data/uploads.
RUN useradd --create-home --uid 10001 rcs \
    && mkdir -p /app/data/db /app/data/uploads /app/data/models \
    && chown -R rcs:rcs /app
USER rcs

ENV PYTHONUNBUFFERED=1 \
    RCS_ENV=production \
    PORT=8765

EXPOSE 8765

# /api/health needs no auth and no database write — safe, cheap check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8765') + '/api/health', timeout=3)"

# entrypoint.sh fetches the frozen private HF artifacts (checksum-verified
# against config/frozen_models_manifest.json — see scripts/fetch_frozen_models.py)
# before execing the unmodified server. HF_TOKEN/DATABASE_URL/GEMINI_API_KEY/
# etc. are read from the container's runtime environment only (docker run -e
# / --env-file) — never present in any image layer.
ENTRYPOINT ["/app/entrypoint.sh"]
