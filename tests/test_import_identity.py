"""Regression tests for the Excel same-filename import collision (synthetic fixtures only).

Root cause: comment_id = sha256(source_file|row_index) and source_file = the file NAME, so a different workbook uploaded under an existing name produced
the SAME ids as the earlier batch. Processed rows are locked (new rows silently skipped), unprocessed rows were overwritten, and the job still said "succeeded".
"""
from __future__ import annotations

import datetime as dt
import hashlib
import time
import urllib.parse

import pytest
from sqlalchemy.orm import Session

from config import settings
from rcs.ingest.loader import _comment_id
from rcs.review import excel_review_flow as erf
from rcs.review.excel_review_flow import import_subset, process_uploaded_excel_batch, run_content_deduplication, write_fixture_workbook
from rcs.review.excel_upload import save_uploaded_workbook
from rcs.review.import_identity import ImportIdentityCollision, assert_no_collision, check_identity, content_fingerprint, resolve_source_file, unique_name
from rcs.review import import_identity as ii
from rcs.review.models import ReviewItem
from rcs.review.repository import get_engine, init_db

T0 = dt.datetime(2026, 8, 1, 9, 0, 0)


def rows(prefix: str, n: int, *, texts: list[str] | None = None):
    out = []
    for i in range(n):
        out.append({
            "original_comment": texts[i] if texts else f"{prefix} أقترح إضافة كلمة جديدة رقم {i} إلى المعجم لفائدة الباحثين",
            "original_title": None, "entry_id": None, "lemma": None, "visitor_name": f"زائر{i}", "visitor_email": f"v{i}@example.invalid",
            "original_type": "ملاحظة", "comment_created_at": T0 + dt.timedelta(minutes=i), "platform": "web",
        })
    return out


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(ii, "_claims", {})
    engine = get_engine(tmp_path / "db.sqlite")
    init_db(engine)

    class E:
        pass

    e = E()
    e.engine, e.tmp = engine, tmp_path

    def upload(name, data_rows):
        src = write_fixture_workbook(tmp_path / f"src_{hashlib.md5(repr(data_rows).encode()).hexdigest()[:6]}.xlsx", data_rows)
        saved = save_uploaded_workbook(name, src.read_bytes())
        return saved.path

    e.upload = upload

    def ingest(path, **kw):
        final = resolve_source_file(engine, path)
        if "row_indices" not in kw:
            kw.setdefault("limit", 1000)                                                              # import_subset defaults to the first 6 meaningful rows
        return final, import_subset(engine, final, **kw)

    e.ingest = ingest
    e.rows_for = lambda sf: _rows_for(engine, sf)
    return e


def _rows_for(engine, source_file):
    with Session(engine) as s:
        return {r.comment_id: (r.original_comment, r.review_status, r.source_row_index) for r in s.query(ReviewItem).filter_by(source_file=source_file).all()}


# ------------------------------------------------------------------------------------------------------------------------ distinct batches
def test_different_files_with_the_same_filename_become_distinct_batches(env):
    a_path, c_path = env.upload("batch_A.xlsx", rows("ألف", 6)), env.upload("batch_A.xlsx", rows("جيم", 8))
    a_final, a_res = env.ingest(a_path)
    assert a_final.name == "batch_A.xlsx" and a_res["inserted"] == 6                              # first upload keeps its name (backward compatible ids)
    before = env.rows_for("batch_A.xlsx")
    c_final, c_res = env.ingest(c_path)
    assert c_final.name == unique_name("batch_A.xlsx", content_fingerprint(c_final)) and c_final.name != "batch_A.xlsx"
    assert c_res["inserted"] == 8 and c_res["skipped"] == 0                                        # the regression: previously 0 inserted, success reported
    assert len(env.rows_for(c_final.name)) == 8 and env.rows_for("batch_A.xlsx") == before        # batch A untouched
    assert not set(env.rows_for(c_final.name)) & set(before)                                        # no shared ids


def test_existing_ids_are_never_changed(env):
    a_final, _ = env.ingest(env.upload("batch.xlsx", rows("ألف", 5)))
    assert set(env.rows_for("batch.xlsx")) == {_comment_id("batch.xlsx", i) for i in range(5)}      # historical id scheme untouched


# ------------------------------------------------------------------------------------------------------------------------ idempotency
def test_identical_reupload_is_idempotent_for_both_batches(env):
    a, c = rows("ألف", 6), rows("جيم", 8)
    env.ingest(env.upload("b.xlsx", a))
    c_final, _ = env.ingest(env.upload("b.xlsx", c))
    snapshot = {sf: env.rows_for(sf) for sf in ("b.xlsx", c_final.name)}
    for _ in range(2):
        a2_final, a2 = env.ingest(env.upload("b.xlsx", a))
        c2_final, c2 = env.ingest(env.upload("b.xlsx", c))
        assert a2_final.name == "b.xlsx" and c2_final.name == c_final.name
        assert a2["inserted"] == 0 and c2["inserted"] == 0
    assert {sf: env.rows_for(sf) for sf in snapshot} == snapshot
    with Session(env.engine) as s:
        assert s.query(ReviewItem).count() == 14                                                     # no duplicates created


def test_resume_of_a_partially_imported_batch_keeps_its_name(env):
    path = env.upload("p.xlsx", rows("ألف", 6))
    final, res = env.ingest(path, row_indices=[0, 1, 2])
    assert res["inserted"] == 3
    again, res2 = env.ingest(env.upload("p.xlsx", rows("ألف", 6)))
    assert again.name == "p.xlsx" and res2["inserted"] + res2["updated"] == 6 and len(env.rows_for("p.xlsx")) == 6


def test_same_comments_resaved_with_new_file_metadata_are_the_same_batch(env):
    data = rows("ألف", 4)
    env.ingest(env.upload("m.xlsx", data))
    time.sleep(1.1)                                                                                  # a re-saved workbook has different bytes (embedded timestamps)
    again = env.upload("m.xlsx", data)
    again.write_bytes(write_fixture_workbook(env.tmp / "resaved.xlsx", data).read_bytes())
    final, res = env.ingest(again)
    assert final.name == "m.xlsx" and res["inserted"] == 0


def test_a_workbook_with_extra_rows_appended_extends_the_same_batch(env):
    env.ingest(env.upload("e.xlsx", rows("ألف", 3)))
    final, res = env.ingest(env.upload("e.xlsx", rows("ألف", 5)))             # rows 0-2 identical, 3-4 new
    assert final.name == "e.xlsx" and res["inserted"] == 2 and len(env.rows_for("e.xlsx")) == 5


# ------------------------------------------------------------------------------------------------------------------------ within-batch duplicates
def test_duplicate_comments_inside_one_batch_follow_the_existing_policy(env):
    same = "تعليق مكرر تماما للتجربة داخل نفس الدفعة الأولى"
    data = rows("x", 4, texts=[same, same, "تعليق آخر مختلف تماما عن السابق", same])
    final, res = env.ingest(env.upload("d.xlsx", data))
    assert res["inserted"] == 4                                                                      # every source row is kept (ids are per row)
    ids = list(env.rows_for(final.name))
    out = run_content_deduplication(env.engine, ids)
    assert out["duplicate_content"] == 2 and out["canonical"] == 2                                   # lowest row index is canonical, the others DUPLICATE_CONTENT
    again, res2 = env.ingest(env.upload("d.xlsx", data))
    assert again.name == "d.xlsx" and res2["inserted"] == 0
    assert sorted(v[1] for v in env.rows_for("d.xlsx").values()).count("DUPLICATE_CONTENT") == 2


# ------------------------------------------------------------------------------------------------------------------------ accounting of expected skips
def _stub_pipeline(monkeypatch, engine):
    class Out:
        def __init__(self, ids):
            self.comment_ids = ids

    def fake(engine_, source_path, *, file_name=None, row_indices=None, client=None, **kw):
        res = import_subset(engine_, source_path, row_indices=row_indices)
        return Out(res["comment_ids"])

    monkeypatch.setattr(erf, "process_uploaded_excel", fake)


def test_expected_skips_are_reported_accurately(env, monkeypatch):
    _stub_pipeline(monkeypatch, env.engine)
    data = rows("ألف", 5, texts=["قصير", "ok", "تعليق كاف الطول للاستيراد الأول", "تعليق ثان كاف الطول للاستيراد", "تعليق ثالث كاف الطول للاستيراد"])
    path = resolve_source_file(env.engine, env.upload("s.xlsx", data))
    first = process_uploaded_excel_batch(env.engine, path, batch_size=10)
    assert (first.total_file_rows, first.batch_processed, first.invalid_rows_skipped, first.already_imported, first.remaining) == (5, 3, 2, 0, 0)
    second = process_uploaded_excel_batch(env.engine, path, batch_size=10)                          # identical re-upload / resume: nothing new, and it says so
    assert (second.batch_processed, second.already_imported, second.invalid_rows_skipped, second.remaining) == (0, 3, 2, 0)
    assert len(env.rows_for("s.xlsx")) == 3                                                          # the two short rows were never ingested


# ------------------------------------------------------------------------------------------------------------------------ unexpected collisions fail clearly
def test_collision_in_the_ingest_step_fails_before_any_write(env, monkeypatch):
    _stub_pipeline(monkeypatch, env.engine)
    env.ingest(env.upload("k.xlsx", rows("ألف", 4)))
    before = env.rows_for("k.xlsx")
    other = env.upload("k.xlsx", rows("جيم", 4))                                                    # different content, same name, resolution bypassed
    assert check_identity(env.engine, other, "k.xlsx").state == "conflict"
    with pytest.raises(ImportIdentityCollision):
        assert_no_collision(env.engine, other)
    with pytest.raises(ImportIdentityCollision):
        process_uploaded_excel_batch(env.engine, other, batch_size=10)
    assert env.rows_for("k.xlsx") == before                                                          # unprocessed rows were NOT overwritten


def test_both_names_occupied_by_other_content_refuses_to_import(env):
    env.ingest(env.upload("z.xlsx", rows("ألف", 3)))
    target = env.upload("z.xlsx", rows("جيم", 3))
    alt = unique_name("z.xlsx", content_fingerprint(target))
    squatter = env.upload(alt, rows("دال", 3))
    squat_final, _ = env.ingest(squatter)                                                            # different content squats on the content-addressed name
    assert squat_final.name == alt
    with pytest.raises(ImportIdentityCollision):
        resolve_source_file(env.engine, target)


def test_two_unprocessed_uploads_with_the_same_name_do_not_collide(env):
    first = resolve_source_file(env.engine, env.upload("q.xlsx", rows("ألف", 3)))                   # nothing imported yet, name only claimed
    second = resolve_source_file(env.engine, env.upload("q.xlsx", rows("جيم", 3)))
    assert first.name == "q.xlsx" and second.name != "q.xlsx"


# ------------------------------------------------------------------------------------------------------------------------ end to end (HTTP + background job + real frozen pipeline)
@pytest.mark.usefixtures("legacy_eligibility_policy")
def test_batch_c_regression_over_http_and_the_job_runner(tmp_path, monkeypatch):
    import json
    import urllib.error
    import urllib.request

    from rcs.review.auth import create_user
    from rcs.review.http_api import serve_in_thread
    from rcs.review.sessions import get_job

    monkeypatch.setattr(settings, "EXCEL_UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(ii, "_claims", {})
    engine = get_engine(tmp_path / "r.sqlite")
    init_db(engine)
    server, base = serve_in_thread(engine)

    def req(path, *, method="GET", data=None, raw=None, cookie=None, filename=None):
        headers = {"Cookie": cookie} if cookie else {}
        if raw is not None:
            headers.update({"Content-Type": "application/octet-stream", "X-Filename": urllib.parse.quote(filename)})
            body = raw
        else:
            headers["Content-Type"] = "application/json"
            body = None if data is None else json.dumps(data).encode()
        r = urllib.request.Request(base + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}"), resp.headers
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}"), e.headers

    def xlsx(data_rows):
        p = tmp_path / f"w_{hashlib.md5(repr(data_rows).encode()).hexdigest()[:6]}.xlsx"
        write_fixture_workbook(p, data_rows)
        return p.read_bytes()

    def run_batch(content):
        s, up, _ = req("/api/excel/upload", method="POST", raw=xlsx(content), cookie=cookie, filename="batch_A.xlsx")
        assert s == 200 and up["valid"]
        s, sess, _ = req("/api/excel/sessions", method="POST", data={"upload_token": up["upload_token"]}, cookie=cookie)
        assert s == 200, sess
        job = get_job(engine, sess["job_id"])
        deadline = time.time() + 180
        while job.status == "running" and time.time() < deadline:
            time.sleep(0.3)
            job = get_job(engine, sess["job_id"])
        assert job.status == "succeeded", job.error_message
        return sess["session"], job

    try:
        create_user(engine, email="admin@test.local", password="testpassword123", display_name="مدير", role="ADMIN")
        _, _, h = req("/api/login", method="POST", data={"email": "admin@test.local", "password": "testpassword123"})
        cookie = h.get("Set-Cookie").split(";")[0]
        a_rows, c_rows = rows("ألف", 6), rows("جيم", 8)
        a_session, a_job = run_batch(a_rows)
        c_session, c_job = run_batch(c_rows)                                                          # the confirmed batch C case: 8 different rows, same name
        assert a_session["source_file"] == "batch_A.xlsx" and c_session["source_file"] != "batch_A.xlsx"
        assert a_session["original_filename"] == c_session["original_filename"] == "batch_A.xlsx"    # the admin still sees the name they uploaded
        assert len(_rows_for(engine, a_session["source_file"])) == 6 and len(_rows_for(engine, c_session["source_file"])) == 8
        assert c_job.rows_done == 8 and c_job.rows_total == 8                                          # accurate progress, not "succeeded with 0 rows"
        s, full, _ = req(f"/api/excel/sessions/{c_session['session_id']}", cookie=cookie)
        assert s == 200 and (full.get("session") or full)["counts"]["total"] == 8
        a2_session, _ = run_batch(a_rows)                                                              # identical re-upload of A
        assert a2_session["source_file"] == "batch_A.xlsx"
        with Session(engine) as sess_db:
            assert sess_db.query(ReviewItem).count() == 14                                             # still 6 + 8, nothing duplicated
    finally:
        server.shutdown()
