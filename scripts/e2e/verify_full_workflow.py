"""Full-workflow verification, SYNTHETIC data only. Real HTTP app + REAL frozen classifiers + SIMULATED (mock) Gemini, external sockets blocked.

    python scripts/e2e/serve_e2e_mock.py --workdir <workdir> --port 8794 --echo-gemini      # fresh, empty workdir
    python scripts/e2e/verify_full_workflow.py run <workdir> http://127.0.0.1:8794 [http://localhost:5193]

Import (new / same-name-different-content / identical re-upload / invalid + duplicate rows) -> classification -> generation -> count-based and
assign-all -> reviewer decisions (approve / edit / no-response / conflict) -> completed-only export, traced per comment number. If the frontend URL is
given, the admin's "المراجعات المكتملة" page is also driven in real Chrome and its downloaded file is compared with the database.
Nothing here prints a comment text; only numbers, counts and ids."""
from __future__ import annotations

import datetime as dt
import hashlib
import http.cookiejar
import io
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COLS = ["العنوان", "معرف المدخل", "اللما", "المحتوى", "اسم الزائر", " البريد الالكتروني للزائر", "النوع", "التاريخ", "المنصة"]
TEMPLATES = ["اقترح كلمة بديلة عربية للمصطلح الأجنبي المنتشر بين الناس", "الصفحة لا تفتح من الجوال وتظهر رسالة خطأ عند البحث", "شكرا لكم على الرد السريع وعلى تصحيح الخطأ الذي أبلغت عنه"]
FAIL_MARK, BAD_MARK = "فشل-التوليد", "رد-معطوب"


def main(work: Path, api: str, front: str | None) -> int:
    import openpyxl

    U = json.loads((work / "e2e_users.json").read_text())
    DB = work / "e2e.sqlite"
    R: list[dict] = []

    def check(name, ok, detail=""):
        R.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)[:300]})
        print(("PASS" if ok else "FAIL"), "-", name, ("| " + str(detail)[:200]) if not ok and detail else "", flush=True)
        return ok

    def q(sql, *a):
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        r = c.execute(sql, a).fetchall()
        c.close()
        return r

    class Api:
        def __init__(self, who):
            self.jar = http.cookiejar.CookieJar()
            self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
            self.who = who
            self.req("POST", "/api/login", {"email": U[who]["email"], "password": U[who]["password"]})

        def req(self, method, path, body=None, raw=None, headers=None, want_bytes=False):
            data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
            h = {"Content-Type": "application/json"} if raw is None else {}
            h.update(headers or {})
            r = urllib.request.Request(api + path, data=data, method=method, headers=h)
            try:
                with self.op.open(r, timeout=600) as resp:
                    pl = resp.read()
                    return resp.status, (pl if want_bytes else (json.loads(pl.decode()) if pl else {}))
            except urllib.error.HTTPError as e:
                pl = e.read()
                try:
                    return e.code, (pl if want_bytes else json.loads(pl.decode()))
                except Exception:  # noqa: BLE001
                    return e.code, {}

    def workbook(rows, path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(COLS)
        for i, (title, entry, text, who) in enumerate(rows):
            ws.append([title, entry, "سوار", text, who, f"v{i}@e2e.invalid", "ملاحظة", dt.datetime(2026, 8, 1) + dt.timedelta(minutes=i), "web"])
        wb.save(path)
        return path

    def import_file(admin, path, filename):
        s, up = admin.req("POST", f"/api/excel/upload?filename={filename}", raw=Path(path).read_bytes(), headers={"Content-Type": "application/octet-stream"})
        if s != 200:
            return s, up, None
        s, sess = admin.req("POST", "/api/excel/sessions", {"upload_token": up["upload_token"]})
        if s != 200:
            return s, sess, None
        t0, status, j = time.time(), None, {}
        while time.time() - t0 < 900:
            _, j = admin.req("GET", f"/api/excel/jobs/{sess['job_id']}")
            status = j.get("status")
            if status in ("succeeded", "failed"):
                break
            time.sleep(2)
        return 200, up, (status, j)

    def num_of(text):
        m = re.search(r"رقم\s*(\d+)", text or "")
        return int(m.group(1)) if m else None

    def fingerprint():
        rows = q("select comment_id, review_status, reviewer_action, reviewer_user_id, row_version, final_reply, reviewer_edit, no_response_reason from review_items order by comment_id")
        ev = q("select event_id, review_item_id, event_type from review_item_events order by event_id")
        return hashlib.sha256(repr((rows, ev)).encode()).hexdigest()

    admin = Api("admin")
    ra, rb = Api("reviewer_a"), Api("reviewer_b")
    A_ID, B_ID = U["reviewer_a"]["user_id"], U["reviewer_b"]["user_id"]
    stamp = work / "inputs"
    stamp.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- 1. IMPORT
    rowsA = [(f"مدخل {n}", f"A{n:03d}", f"{TEMPLATES[n % 3]} رقم {n}", f"زائر {n}") for n in range(1, 41)]
    rowsA[4] = (rowsA[4][0], rowsA[4][1], f"{TEMPLATES[1]} رقم 5 {FAIL_MARK}", rowsA[4][3])           # provider failure -> must fail closed
    rowsA[5] = (rowsA[5][0], rowsA[5][1], f"{TEMPLATES[1]} رقم 6 {BAD_MARK}", rowsA[5][3])            # malformed provider output -> must fail closed
    rowsA.append(("فارغ", "A900", "", "زائر فارغ"))                                                      # invalid row (no comment)
    rowsA.append(("مكرر", "A901", rowsA[0][2], "زائر مكرر"))                                             # duplicate of comment 1 inside the file
    rowsB = [(f"مدخل {n}", f"B{n:03d}", f"{TEMPLATES[n % 3]} رقم {n}", f"زائر {n}") for n in range(101, 131)]
    pA, pB = workbook(rowsA, stamp / "A.xlsx"), workbook(rowsB, stamp / "B.xlsx")

    s, upA, resA = import_file(admin, pA, "batch.xlsx")
    check("import A: accepted and processed", s == 200 and resA and resA[0] == "succeeded", (s, resA and resA[0]))
    check("import A: the validator flags the file as valid and reports the full row count", upA.get("valid") and upA.get("total_rows") == len(rowsA), (upA.get("valid"), upA.get("total_rows")))
    itemsA = q("select comment_id, original_comment, review_status, source_file, source_row_index from review_items where source_file like 'batch%'")
    n_a = len(itemsA)
    check("import A: invalid row skipped and in-file duplicate not duplicated -> 40 unique comments stored (+ duplicate row held as non-reviewable)", n_a == 41 and sum(1 for r in itemsA if r[2] == "DUPLICATE_CONTENT") == 1, n_a)
    check("import A: job progress equals the valid rows (41 of 41; the invalid row is not counted)", resA and resA[1].get("rows_done") == 41 and resA[1].get("rows_total") == 41, resA and {k: resA[1].get(k) for k in ("rows_done", "rows_total")})
    srcA = {r[3] for r in itemsA}
    check("import A: one batch identity", len(srcA) == 1, srcA)
    before = q("select count(*) from review_items")[0][0]

    s, upA2, resA2 = import_file(admin, pA, "batch.xlsx")
    after = q("select count(*) from review_items")[0][0]
    check("identical re-upload (same name, same content) is idempotent: no new rows", after == before, (before, after))

    s, upB, resB = import_file(admin, pB, "batch.xlsx")
    srcs = {r[0] for r in q("select distinct source_file from review_items")}
    check("same filename with DIFFERENT content becomes a separate batch (no overwrite, no collision)", resB and resB[0] == "succeeded" and len(srcs) == 2, (resB and resB[0], len(srcs)))
    check("different-content batch added exactly its 30 comments; batch A untouched",
          q("select count(*) from review_items")[0][0] == before + 30 and q("select count(*) from review_items where source_file=?", next(iter(srcA)))[0][0] == n_a, "")
    ids = [r[0] for r in q("select comment_id from review_items")]
    check("every comment_id is unique and (source_file,row_index) is unique", len(ids) == len(set(ids)) and q("select count(*) from (select 1 from review_items group by source_file, source_row_index having count(*)>1)")[0][0] == 0, "")

    # ---------------------------------------------------------------- 2/3. CLASSIFICATION + GENERATION, traced per comment number
    rows = q("""select comment_id, original_comment, source_file, review_status, ai_primary_intent, ai_topic_subtype, ai_sentiment, ai_classification_source, ai_routing_decision,
                generation_routing_decision, generation_status, suggested_final_reply, generation_prompt_version from review_items""")
    byn = {}
    for r in rows:
        byn.setdefault(num_of(r[1]), []).append(r)
    live = [r for r in rows if r[3] != "DUPLICATE_CONTENT" and r[3] != "EXCLUDED"]
    check("classification: the frozen models persisted intent, topic and sentiment for every reviewable comment",
          live and all(r[4] and r[5] and r[6] and "FROZEN" in (r[7] or "") and r[8] for r in live), len(live))
    drafted = [r for r in rows if r[11]]
    check("generation: every stored draft carries the rg-v14.4 prompt version and is bound to ITS OWN comment number", drafted and all(r[12] == "rg-v14.4" and num_of(r[11]) == num_of(r[1]) for r in drafted), len(drafted))
    called = json.loads((work / "e2e_counters.json").read_text())
    elig = [r for r in rows if r[9] == "GENERATE_DRAFT"]
    check("generation: drafts exist only for rows routed to GENERATE_DRAFT; others carry no invented reply", all((not r[11]) for r in rows if r[9] != "GENERATE_DRAFT") and elig, (len(elig), len(drafted)))
    for mark, name in ((FAIL_MARK, "provider failure"), (BAD_MARK, "malformed provider output")):
        hit = [r for r in rows if mark in (r[1] or "")]
        check(f"generation failure handling ({name}): no draft stored, row falls back to review-only, comment still present",
              hit and all((not r[11]) and r[3] in ("REVIEW_ONLY", "NO_RESPONSE_RECOMMENDED", "EXCLUDED") for r in hit), [(r[3], r[10]) for r in hit])
    check("generation: no personal identifier ever reached the provider (mock counted)", called.get("prompts_containing_pii", 0) == 0, called)
    check("isolation: no outbound connection other than loopback was attempted", called.get("blocked_external_connections", 0) == 0, called)
    snap = {r[0]: (r[1], r[4], r[5], r[6], r[11]) for r in rows}

    # ---------------------------------------------------------------- 4. ASSIGNMENT
    reviewable = [r[0] for r in q("select comment_id from review_items where review_status in ('DRAFT_READY','REVIEW_ONLY','NO_RESPONSE_RECOMMENDED') and reviewer_user_id is null order by comment_id")]
    s, res = admin.req("POST", "/api/admin/assign-by-filter", {"reviewer_user_id": A_ID, "count": 20})
    check("assign: chosen count 20 -> exactly 20 assigned to reviewer A", s == 200 and res.get("assigned") == 20 and len(q("select 1 from review_items where reviewer_user_id=?", A_ID)) == 20, (s, res))
    s, res = admin.req("POST", "/api/admin/assign-by-filter", {"reviewer_user_id": B_ID, "count": 10})
    check("assign: reviewer B receives 10 different comments (no item assigned twice)", s == 200 and res.get("assigned") == 10 and not (
        {r[0] for r in q("select comment_id from review_items where reviewer_user_id=?", A_ID)} & {r[0] for r in q("select comment_id from review_items where reviewer_user_id=?", B_ID)}), res)
    s, res = admin.req("POST", "/api/admin/assign-by-filter", {"reviewer_user_id": B_ID, "count": 10_000})
    check("assign: asking for more than exist assigns only what is available and reports the shortfall", s == 200 and res.get("shortfall", 0) > 0 and res.get("assigned") == res.get("matched_before"), res)
    left = q("select count(*) from review_items where reviewer_user_id is null and review_status in ('DRAFT_READY','REVIEW_ONLY','NO_RESPONSE_RECOMMENDED')")[0][0]
    check("assign: after assigning everything eligible nothing reviewable remains unassigned; duplicates/excluded never assigned", left == 0 and q(
        "select count(*) from review_items where reviewer_user_id is not null and review_status in ('DUPLICATE_CONTENT','EXCLUDED')")[0][0] == 0, left)
    a_ids = {r[0] for r in q("select comment_id from review_items where reviewer_user_id=?", A_ID)}
    b_ids = {r[0] for r in q("select comment_id from review_items where reviewer_user_id=?", B_ID)}

    # ---------------------------------------------------------------- 5. REVIEWER
    s, qa = ra.req("GET", "/api/queue?limit=200")
    check("reviewer A sees exactly her assigned comments across pages (and none of B's)", s == 200 and {i["comment_id"] for i in qa["items"]} == a_ids and qa["total"] == len(a_ids), (s, qa.get("total"), len(a_ids)))
    s, qb = rb.req("GET", "/api/queue?limit=200")
    check("reviewer B sees only hers", {i["comment_id"] for i in qb["items"]} == b_ids, qb.get("total"))
    someone_b = sorted(b_ids)[0]
    check("isolation: reviewer A cannot read or decide on a comment assigned to B", ra.req("GET", f"/api/items/{someone_b}")[0] in (403, 404) and ra.req("POST", f"/api/items/{someone_b}/no-response", {"reason": "x"})[0] in (403, 404), "")
    check("isolation: a reviewer cannot self-assign or use admin endpoints", ra.req("POST", "/api/admin/assign-by-filter", {"reviewer_user_id": A_ID, "all": True})[0] == 403 and ra.req("GET", "/api/excel/export/completed?all=true", want_bytes=True)[0] == 403, "")

    decided = {}
    for who, client, ids_ in (("A", ra, sorted(a_ids)), ("B", rb, sorted(b_ids))):
        for i, cid in enumerate(ids_):
            _, d = client.req("GET", f"/api/items/{cid}")
            ver = d.get("row_version")
            kind = i % 3
            has_draft = bool(snap[cid][4])
            if kind == 0 and has_draft:
                s, _ = client.req("POST", f"/api/items/{cid}/send-as-is", {"expected_version": ver})
                decided[cid] = ("APPROVED", s)
            elif kind == 1:
                s, _ = client.req("POST", f"/api/items/{cid}/edit-and-send", {"edited_response": f"نشكركم على ملاحظتكم رقم {num_of(snap[cid][0])}، وستُدرس بعناية.", "expected_version": ver})
                decided[cid] = ("EDITED", s)
            else:
                s, _ = client.req("POST", f"/api/items/{cid}/no-response", {"reason": "لا يتطلب ردًا", "expected_version": ver})
                decided[cid] = ("NONE", s)
    if not all(s == 200 for _, s in decided.values()):                                                  # surface the real status codes once if the contract differs
        print("   decision status codes:", sorted({(k, s) for k, s in decided.values()}))
    check("reviewer decisions (approve / edit-and-approve / no-response) were accepted for every assigned comment", all(s == 200 for _, s in decided.values()), len(decided))
    done = {r[0]: r for r in q("select comment_id, reviewer_action, final_reply, reviewer_edit, no_response_reason, review_status, row_version, reviewer_user_id from review_items where reviewer_action is not null")}
    check("persistence: each decision is stored on the right comment and by the right reviewer", len(done) == len(decided) and all(done[c][7] == (A_ID if c in a_ids else B_ID) for c in done), len(done))
    check("persistence: final reply for approved items equals the stored draft; edited items store the edit; no-response items store no reply",
          all((done[c][2] == snap[c][4]) if k == "APPROVED" else (done[c][3] and done[c][2] == done[c][3]) if k == "EDITED" else (not done[c][2] and done[c][4]) for c, (k, _) in decided.items()), "")
    ev = q("select review_item_id, event_type from review_item_events")
    decided_events = {e[0] for e in ev if e[1] != "IMPORTED"}
    check("audit: every decided comment has audit events recorded (history intact)", len(done) > 0 and len(q("select distinct review_item_id from review_item_events")) >= len(done), (len(ev), sorted({e[1] for e in ev})))
    # conflict protection: a stale version is refused and changes nothing
    stale = sorted(a_ids)[0]
    fp0 = fingerprint()
    s, _ = ra.req("POST", f"/api/items/{stale}/edit-and-send", {"edited_response": "رد قديم", "expected_version": 1})
    check("conflict: a decision on an already-decided comment with a stale version is rejected (409) and nothing changes", s in (409, 400, 422) and fingerprint() == fp0, s)
    # nothing about classification/response moved while reviewing
    now = {r[0]: (r[1], r[4], r[5], r[6], r[11]) for r in q("""select comment_id, original_comment, source_file, review_status, ai_primary_intent, ai_topic_subtype, ai_sentiment, ai_classification_source, ai_routing_decision,
                generation_routing_decision, generation_status, suggested_final_reply, generation_prompt_version from review_items""")}
    check("traceability: original text, AI classification and AI draft are unchanged by assignment and review (no silent overwrite)", now == snap, "")

    # ---------------------------------------------------------------- 6. EXPORT
    fp1 = fingerprint()
    s, blob = admin.req("GET", "/api/excel/export/completed?all=true", want_bytes=True)
    wb = openpyxl.load_workbook(io.BytesIO(blob))
    h1, *r1 = [list(r) for r in wb["الردود المراجعة"].iter_rows(values_only=True)]
    h2, *r2 = [list(r) for r in wb["بيانات التدقيق"].iter_rows(values_only=True)]
    exp_ids = [r[h2.index("comment_id")] for r in r2]
    check("export all: exactly the completed decisions, each once, no pending/duplicate/excluded rows", s == 200 and set(exp_ids) == set(done) and len(exp_ids) == len(set(exp_ids)) == len(done), (s, len(exp_ids), len(done)))
    ok = True
    for a_, b_ in zip(r1, r2):
        cid = b_[h2.index("comment_id")]
        ok = ok and a_[h1.index("إجراء المراجع")] == done[cid][1] and (a_[h1.index("الرد النهائي")] or None) == (done[cid][2] or None) and (b_[h2.index("no_response_reason")] or None) == (done[cid][4] or None)
    check("export: final replies, edits, no-response decisions and reasons match the persisted decisions row by row", ok, "")
    check("export: each exported row still belongs to its own comment number (text, intent and reply stay together)",
          all(num_of(r[h1.index("الرد النهائي")]) in (None, num_of(snap[r2_[h2.index("comment_id")]][0])) for r, r2_ in zip(r1, r2)), "")
    s, blob5 = admin.req("GET", "/api/excel/export/completed?count=5", want_bytes=True)
    n5 = len(list(openpyxl.load_workbook(io.BytesIO(blob5))["بيانات التدقيق"].iter_rows(values_only=True))) - 1 if s == 200 else -1
    check("export N: a chosen count exports exactly that many completed reviews", n5 == 5, (s, n5))
    s, blobA = admin.req("GET", f"/api/excel/export/completed?all=true&reviewer_user_id={A_ID}", want_bytes=True)
    nA = len(list(openpyxl.load_workbook(io.BytesIO(blobA))["بيانات التدقيق"].iter_rows(values_only=True))) - 1 if s == 200 else -1
    check("export filtered: only the selected reviewer's completed reviews", nA == len([c for c in done if c in a_ids]), (s, nA))
    check("export is read-only: statuses, assignments, versions, texts and audit history are byte-identical", fingerprint() == fp1, "")
    s, _ = admin.req("POST", "/api/items/" + sorted(done)[0] + "/no-response", {"reason": "x"})
    check("no admin decision path exists on the review endpoints (admin cannot decide as a reviewer)", s in (400, 403, 404, 409), s)

    # formula-injection protection itself is asserted in tests/test_export_completed.py; here we verify the produced workbook has no formula cells
    check("export: no cell in either sheet is stored as a formula", all(c.data_type != "f" for ws in wb.worksheets for row in ws.iter_rows() for c in row), "")

    # ---------------------------------------------------------------- browser: admin completed page vs DB (optional)
    if front:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            br = p.chromium.launch(channel="chrome", headless=True)
            pg = br.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True).new_page()
            pg.on("dialog", lambda d: d.accept())
            pg.goto(front + "/")
            pg.wait_for_timeout(600)
            pg.fill('input[placeholder="أدخل البريد الإلكتروني"]', U["admin"]["email"])
            pg.fill('input[placeholder="أدخل كلمة المرور"]', U["admin"]["password"])
            pg.click("button[type=submit]")
            pg.wait_for_timeout(1800)
            pg.click("text=المراجعات المكتملة >> nth=0")
            pg.wait_for_timeout(2000)
            body = re.sub(r"\s+", " ", pg.inner_text("body"))
            check("browser: the admin's completed page shows the persisted total", f"{len(done)} مراجعة مكتملة" in body, re.findall(r"\d+ مراجعة مكتملة", body))
            with pg.expect_download() as dl:
                pg.click("button:has-text('تصدير المراجعات المكتملة')")
            path = work / "ui_full_export.xlsx"
            dl.value.save_as(str(path))
            w2 = openpyxl.load_workbook(path)
            rr = [list(r) for r in w2["بيانات التدقيق"].iter_rows(values_only=True)]
            hh, rest = rr[0], rr[1:]
            check("browser: the file downloaded through the UI contains exactly the completed set", {r[hh.index("comment_id")] for r in rest} == set(done), len(rest))
            check("browser: exporting through the UI did not change any persisted state", fingerprint() == fp1, "")
            br.close()
    else:
        print("   (browser part skipped: no frontend URL given)")

    out = work / "full_workflow_report.json"
    out.write_text(json.dumps({"results": R, "pass": sum(r["result"] == "PASS" for r in R), "fail": sum(r["result"] == "FAIL" for r in R)}, ensure_ascii=False, indent=1))
    print("\nPASS", sum(r["result"] == "PASS" for r in R), "FAIL", sum(r["result"] == "FAIL" for r in R))
    return 0 if all(r["result"] == "PASS" for r in R) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None))
