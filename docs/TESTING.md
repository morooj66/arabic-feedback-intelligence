# Testing

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt
HF_HUB_OFFLINE=1 python -m pytest -q -rs tests
cd web && npm ci && npm test && npx tsc --noEmit && npm run build
```
- Offline: no test calls an LLM or any external service; time-dependent and concurrency behaviour is tested deterministically where possible (e.g. `tests/test_assign_race_deterministic.py` forces the interleaving).
- Tests that need things not in this repository (frozen classifier files, restricted data, a live MongoDB, a git work tree) **skip with an explicit reason** (`tests/conftest.py`).
- `scripts/e2e/` has Playwright journeys (reviewer flow, export, full workflow, messaging, authentication, demo mode) against a real backend with a mocked provider and blocked external sockets.
- Passing means the workflow behaves as specified on synthetic data. It does not demonstrate model accuracy, reply quality, PostgreSQL/MongoDB production behaviour or load capacity.


## Data layer on PostgreSQL
The suites run on SQLite by default. `RCS_TEST_POSTGRES_URL=postgresql+psycopg://…/<throw-away database>` runs the same suites on PostgreSQL (one isolated schema per test). Use a disposable database, never an institutional one; the tests never read `.env`.
