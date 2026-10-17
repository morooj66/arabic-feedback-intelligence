"""Synthetic-data DEMO MODE end to end (real Chrome -> web UI -> the real server entry point in demo mode). Synthetic data only.

    # prepare (operator step): DATABASE_URL=sqlite:///<dir>/rcs_demo_local.sqlite python scripts/demo/init_demo_environment.py ... --credentials-out <dir>/credentials.json
    # serve:  RCS_DEMO_MODE=synthetic-demo RCS_DEMO_ACK=... DATABASE_URL=... python scripts/serve_reviewer_api.py --port 8797   (optionally with ML libraries unimportable)
    python scripts/e2e/verify_demo_mode.py run <dir> http://localhost:5198
Credentials are read from <dir>/credentials.json and never printed."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import tempfile
import urllib.request
from pathlib import Path


def main(env: Path, base: str) -> int:
    import openpyxl
    from playwright.sync_api import sync_playwright

    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"), str(Path(__file__).resolve().parents[2])]
    from rcs.demo import synthetic
    from rcs.review.excel_review_flow import write_fixture_workbook

    C = json.loads((env / "credentials.json").read_text())
    admin_email = next(e for e, v in C.items() if v["role"] == "ADMIN")
    rev_emails = [e for e, v in C.items() if v["role"] == "REVIEWER"]
    DB = env / "rcs_demo_local.sqlite"
    R: list[dict] = []

    def check(name, ok, detail=""):
        R.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)[:300]})
        print(("PASS" if ok else "FAIL"), "-", name, ("| " + str(detail)[:200]) if not ok and detail else "", flush=True)

    def q(sql, *a):
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); r = c.execute(sql, a).fetchall(); c.close(); return r

    def body(pg):
        return re.sub(r"\s+", " ", pg.inner_text("body"))

    def new_page(br):
        pg = br.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True).new_page()
        pg.on("dialog", lambda d: d.accept())
        return pg

    def login(pg, email, password):
        pg.goto(base + "/"); pg.wait_for_timeout(900)
        pg.fill('input[placeholder="أدخل البريد الإلكتروني"]', email); pg.fill('input[placeholder="أدخل كلمة المرور"]', password)
        pg.click("button[type=submit]"); pg.wait_for_timeout(2200)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rows = synthetic.demo_rows(8)
        for i, r in enumerate(rows):
            r["original_comment"] = r["original_comment"].replace("عينة تجريبية رقم", "عينة إضافية رقم")
        plain = write_fixture_workbook(td / "plain.xlsx", rows)                      # NOT stamped: must be refused
        stamped = write_fixture_workbook(td / "stamped.xlsx", rows)
        from rcs.demo import workbook
        workbook.stamp(stamped)

        with sync_playwright() as p:
            br = p.chromium.launch(channel="chrome", headless=True)

            # ------------------------------------------------------------ login page: invite-only, honest, labelled
            anon = new_page(br); anon.goto(base + "/"); anon.wait_for_timeout(1500)
            check("demo: the instance labels itself as a synthetic-data demo", "نسخة تجريبية" in body(anon) and "بيانات اصطناعية" in body(anon), body(anon)[:120])
            check("demo: the original login page is unchanged in structure (e-mail, password, forgot, Microsoft button)",
                  anon.query_selector('input[placeholder="أدخل البريد الإلكتروني"]') is not None and "نسيت كلمة المرور؟" in body(anon) and anon.query_selector("button.btn-ms") is not None, "")
            check("demo: invite-only: no 'new account request' link is offered", "طلب حساب جديد" not in body(anon), "")
            check("demo: Microsoft sign-in is visible but disabled", anon.query_selector("button.btn-ms").is_disabled() and anon.query_selector("a.btn-ms") is None, "")
            st = anon.evaluate("""async () => { const post = (p, b) => fetch(p, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(b)}).then(r => r.status);
                return { signup: await post('/api/account-requests', {email:'x@example.invalid', display_name:'x', password:'long-enough-1'}), reset: await post('/api/password-reset/request', {email:'a@example.invalid'}),
                         queue: (await fetch('/api/queue')).status, users: (await fetch('/api/admin/users')).status, export: (await fetch('/api/excel/export/completed?all=true')).status } }""")
            check("demo: sign-up and password reset are unavailable (503) and every protected endpoint answers 401 without a session", st == {"signup": 503, "reset": 503, "queue": 401, "users": 401, "export": 401}, st)
            bad = []
            for email, pw in [("admin@example.invalid", "admin"), ("demo.admin@demo.invalid", "password"), (admin_email, "wrong-password-1"), ("admin", "admin123")]:
                pg = new_page(br); login(pg, email, pw)
                bad.append(pg.query_selector('input[placeholder="أدخل كلمة المرور"]') is not None)
            check("demo: no default, guessable or wrong credentials work (no bypass)", all(bad), bad)

            # ------------------------------------------------------------ admin: dashboard, upload policy, import
            adm = new_page(br); login(adm, admin_email, C[admin_email]["password"])
            check("admin: signs in with the operator-issued credentials and sees the admin dashboard seeded with synthetic comments",
                  "الإسناد" in body(adm) and "نظرة عامة" in body(adm) and q("select count(*) from review_items")[0][0] == 36, body(adm)[:100])
            before = q("select count(*) from review_items")[0][0]
            adm.click("text=استيراد ملف جديد"); adm.wait_for_timeout(1200)
            adm.set_input_files("input[type=file]", str(plain)); adm.wait_for_timeout(3000)
            check("demo: a workbook WITHOUT the synthetic-data marker is refused in the UI and nothing is imported", "اصطناع" in body(adm) and q("select count(*) from review_items")[0][0] == before, body(adm)[-200:])
            adm.goto(base + "/"); adm.wait_for_timeout(1500)
            if adm.query_selector('input[placeholder="أدخل كلمة المرور"]') is not None:
                login(adm, admin_email, C[admin_email]["password"])
            adm.click("text=استيراد ملف جديد"); adm.wait_for_timeout(1200)
            adm.set_input_files("input[type=file]", str(stamped))
            import time
            t0 = time.time()
            while time.time() - t0 < 90 and q("select count(*) from review_items")[0][0] < before + 8:
                adm.wait_for_timeout(1500)
            check("demo: a marker-stamped synthetic workbook is imported through the UI (classification by the rule stand-in, drafts by the local generator)", q("select count(*) from review_items")[0][0] == before + 8, q("select count(*) from review_items")[0][0])
            check("demo: provenance says DEMO_RULES (never mistaken for the frozen models) and drafts come from the local generator",
                  {r[0] for r in q("select distinct ai_classification_source from review_items")} == {"DEMO_RULES_SYNTHETIC_ONLY"}
                  and all((r[0] or "") in ("", None) or len(r[0]) < 80 for r in q("select suggested_final_reply from review_items")) and q("select count(*) from review_items where suggested_final_reply is not null and suggested_final_reply != ''")[0][0] > 10, "")

            # ------------------------------------------------------------ assignment
            adm.click("text=الإسناد >> nth=0"); adm.wait_for_timeout(2500)
            opts = adm.eval_on_selector_all('select[aria-label="المراجع للإسناد بالعدد"] option', "els => els.map(e => [e.value, e.textContent])")
            r1 = C[rev_emails[0]]["user_id"]
            adm.select_option('select[aria-label="المراجع للإسناد بالعدد"]', value=r1); adm.wait_for_timeout(400)
            adm.fill('input[aria-label="عدد التعليقات المطلوب إسنادها"]', "9"); adm.click('button:has-text("إسناد العدد")'); adm.wait_for_timeout(2500)
            check("admin: count-based assignment of 9 comments to reviewer 1", q("select count(*) from review_items where reviewer_user_id=?", r1)[0][0] == 9, opts[:2])

            # ------------------------------------------------------------ reviewer 1: isolation + decisions
            rv = new_page(br); login(rv, rev_emails[0], C[rev_emails[0]]["password"])
            check("reviewer 1: sees only reviewer navigation", "المسند لي" in body(rv) and "الإسناد" not in body(rv) and "التحليلات" not in body(rv), body(rv)[:100])
            qa = rv.evaluate("fetch('/api/queue?limit=200').then(r=>r.json()).then(j=>[j.total, j.items.map(i=>i.comment_id)])")
            mine = {r[0] for r in q("select comment_id from review_items where reviewer_user_id=?", r1)}
            check("reviewer 1: the queue is exactly the 9 assigned comments", qa[0] == 9 and set(qa[1]) == mine, qa[0])
            st = rv.evaluate("""async () => ({ users: (await fetch('/api/admin/users')).status, assign: (await fetch('/api/admin/assign-by-filter', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({reviewer_user_id:'x', all:true})})).status, export: (await fetch('/api/excel/export/completed?all=true')).status, signup: (await fetch('/api/admin/account-requests')).status })""")
            check("reviewer 1: admin endpoints are refused by the backend (403)", st == {"users": 403, "assign": 403, "export": 403, "signup": 403}, st)
            rv2 = new_page(br); login(rv2, rev_emails[1], C[rev_emails[1]]["password"])
            q2 = rv2.evaluate("fetch('/api/queue?limit=200').then(r=>r.json()).then(j=>j.total)")
            check("reviewer 2: sees none of reviewer 1's comments and cannot open them", q2 == 0 and rv2.evaluate("(id)=>fetch('/api/items/'+id).then(r=>r.status)", sorted(mine)[0]) in (403, 404), q2)

            rv.click('button:has-text("بدء المراجعة")'); rv.wait_for_timeout(2500)

            def approve_enabled():
                b = rv.query_selector('button:has-text("اعتماد كما هو")'); return b is not None and b.is_enabled()

            kinds = []
            for kind in ("approve", "edit", "noreply"):
                if kind == "approve":
                    for _ in range(9):
                        if approve_enabled(): break
                        nxt = rv.query_selector('button:has-text("التالي")')
                        if nxt is None or nxt.is_disabled(): break
                        nxt.click(); rv.wait_for_timeout(800)
                    if not approve_enabled(): kinds.append("approve-unavailable"); continue
                    rv.click('button:has-text("اعتماد كما هو")'); kinds.append("approve")
                elif kind == "edit":
                    rv.click('button:has-text("تعديل الرد")'); rv.wait_for_timeout(400); rv.fill("textarea", "نشكركم على تواصلكم الكريم مع المعجم."); rv.wait_for_timeout(300)
                    rv.click('button:has-text("تعديل واعتماد")'); kinds.append("edit")
                else:
                    rv.click('button:has-text("لا يحتاج رد")'); rv.wait_for_timeout(700); rv.click('button:has-text("تأكيد")'); kinds.append("noreply")
                rv.wait_for_timeout(2800)
            acts = {r[0]: r[1] for r in q("select reviewer_action, count(*) from review_items where reviewer_action is not null group by 1")}
            check("reviewer 1: approve / edit-and-approve / no-response decisions were persisted", kinds == ["approve", "edit", "noreply"] and acts == {"SEND_AS_IS": 1, "EDIT_AND_SEND": 1, "NO_RESPONSE_NEEDED": 1}, (kinds, acts))

            # ------------------------------------------------------------ messaging + export
            adm.click("button:has-text('الرسائل') >> nth=0"); adm.wait_for_timeout(1500)
            adm.click(f".messages-thread-item:has-text('{rev_emails[0].split('@')[0]}')"); adm.wait_for_timeout(800)
            adm.fill("#message-draft", "رسالة تجريبية من المسؤول."); adm.click("button:has-text('إرسال')"); adm.wait_for_timeout(1500)
            rv.click("button:has-text('الرسائل') >> nth=0"); rv.wait_for_timeout(1800)
            check("messaging: the reviewer receives the admin's message", "رسالة تجريبية من المسؤول" in body(rv), "")
            adm.click("text=المراجعات المكتملة >> nth=0"); adm.wait_for_timeout(2500)
            with adm.expect_download() as dl:
                adm.click("button:has-text('تصدير المراجعات المكتملة')")
            path = td / "export.xlsx"; dl.value.save_as(str(path))
            wb = openpyxl.load_workbook(path)
            h2, *r2 = [list(r) for r in wb["بيانات التدقيق"].iter_rows(values_only=True)]
            check("export: completed-only (exactly the 3 decided comments)", len(r2) == 3 and {r[h2.index("comment_id")] for r in r2} == {r[0] for r in q("select comment_id from review_items where reviewer_action is not null")}, len(r2))

            # ------------------------------------------------------------ no real provider, no real data
            check("synthetic only: every stored comment is an invented 'عينة' sample", q("select count(*) from review_items where original_comment not like '%عينة%'")[0][0] == 0, "")
            from rcs.demo.provider import DEFAULT, REPLIES
            canned = set(REPLIES.values()) | {DEFAULT}
            drafts = [r[0] for r in q("select suggested_final_reply from review_items where suggested_final_reply is not null and suggested_final_reply != ''")]
            check("no real provider: every generated draft is one of the local generator's canned replies (an LLM would not produce only these)", drafts and all(d in canned for d in drafts), len(drafts))
            br.close()
    out = env / "demo_mode_report.json"
    out.write_text(json.dumps({"results": R, "pass": sum(r["result"] == "PASS" for r in R), "fail": sum(r["result"] == "FAIL" for r in R)}, ensure_ascii=False, indent=1))
    print("\nPASS", sum(r["result"] == "PASS" for r in R), "FAIL", sum(r["result"] == "FAIL" for r in R))
    return 0 if all(r["result"] == "PASS" for r in R) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[2]), sys.argv[3]))
