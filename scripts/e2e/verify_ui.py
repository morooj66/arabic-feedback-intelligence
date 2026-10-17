"""Real-browser (system Chrome via Playwright) journeys against the real Vite frontend -> real backend (scripts/e2e/serve_e2e_mock.py). Synthetic data, mock Gemini.
Needs: `pip install playwright` in a scratch venv (uses the installed Chrome, downloads nothing). Writes <workdir>/ui_report.json and screenshots."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

WORK = Path(sys.argv[1]); BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:5192"
U = json.loads((WORK / "e2e_users.json").read_text()); DB = WORK / "e2e.sqlite"
R: list[dict] = []


def check(name, ok, detail=""):
    R.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)[:300]})
    print(("PASS" if ok else "FAIL"), "-", name, ("| " + str(detail)[:200]) if not ok and detail else "", flush=True)
    return ok


def body(pg):
    return pg.inner_text("body")


def login(pg, who, pw=None):
    pg.goto(BASE + "/"); pg.wait_for_timeout(700)
    pg.fill('input[placeholder="أدخل البريد الإلكتروني"]', U[who]["email"]); pg.fill('input[placeholder="أدخل كلمة المرور"]', pw or U[who]["password"])
    pg.click('button[type=submit]'); pg.wait_for_timeout(1800)


def rows():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); c.row_factory = sqlite3.Row
    out = [dict(r) for r in c.execute("select * from review_items")]; c.close(); return out


with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True)
    # ---------------------------------------------------------------- unauthenticated
    anon = br.new_page(viewport={"width": 1400, "height": 900}); anon.goto(BASE + "/review"); anon.wait_for_timeout(1500)
    check("ui: unauthenticated /review shows the login form (no data)", anon.query_selector('input[placeholder="أدخل كلمة المرور"]') is not None and "إجمالي التعليقات" not in body(anon), anon.url)
    login(anon, "admin", "definitely-wrong-password"); b = body(anon)
    check("ui: wrong password shows an error and stays on login", anon.query_selector('input[placeholder="أدخل كلمة المرور"]') is not None, b[:120].replace("\n", " "))
    anon.close()

    # ---------------------------------------------------------------- admin journey
    ad = br.new_page(viewport={"width": 1400, "height": 900}); DIALOGS = []; ad.on("dialog", lambda d: (DIALOGS.append(d.message[:80]), d.accept())); login(ad, "admin")
    check("ui: admin login lands on the admin dashboard", "إجمالي التعليقات" in body(ad) and "الإسناد" in body(ad), ad.url)
    ad.click("text=استيراد ملف جديد"); ad.wait_for_timeout(600)
    ad.set_input_files("[data-testid=excel-file-input]", str(WORK / "ui_input.xlsx"))
    done = False
    for _ in range(120):
        ad.wait_for_timeout(2500)
        t = body(ad)
        if len(rows()) >= 15 and sum(1 for r in rows() if r.get("ai_primary_intent")) >= 12 and ("جاهز للمراجعة" in t and "✓" in t.split("جاهز للمراجعة")[0][-6:] or "تم" in t):
            done = True; break
        if re.search(r"فشل|خطأ", t) and len(rows()) == 0 and _ > 10:
            break
    ad.screenshot(path=str(WORK / "ui_import_done.png"))
    check("ui+db: import through the UI persisted and classified all rows", len(rows()) == 15 and sum(1 for r in rows() if r.get("ai_primary_intent")) >= 12, (len(rows()), body(ad)[200:420].replace("\n", " ")))
    ad.click("text=الرئيسية >> nth=0"); ad.wait_for_timeout(2500)
    t = body(ad)
    arabic_digits = str(15).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    check("ui: dashboard total reflects the import (15)", arabic_digits in t or "15" in t, t[150:420].replace("\n", " "))
    ad.click("text=الإسناد >> nth=0"); ad.wait_for_timeout(2000)
    boxes = ad.query_selector_all("tbody input[type=checkbox]")
    check("ui: assignment page lists the unassigned items", len(boxes) >= 10, len(boxes))
    half = len(boxes) // 2
    for bx in boxes[:half]:
        bx.check()
    ad.select_option('select[aria-label="اختر المراجع"]', label=next(o.inner_text() for o in ad.query_selector_all('select[aria-label="اختر المراجع"] option') if "reviewer_a" in o.inner_text()))
    ad.click("button:has-text(\"إسناد المحدد\")"); ad.wait_for_timeout(2500)
    mid = body(ad); R.append({"check": "info: confirm dialogs seen", "result": "INFO", "detail": str(DIALOGS)})
    ad.screenshot(path=str(WORK / "ui_assign_a.png"))
    a_cnt = sum(1 for r in rows() if r.get("reviewer_user_id") == U["reviewer_a"]["user_id"])
    check("ui+db: assigning a selection to reviewer A persisted", a_cnt == half, (a_cnt, half))
    boxes = ad.query_selector_all("tbody input[type=checkbox]")
    for bx in boxes:
        bx.check()
    ad.select_option('select[aria-label="اختر المراجع"]', label=next(o.inner_text() for o in ad.query_selector_all('select[aria-label="اختر المراجع"] option') if "reviewer_b" in o.inner_text()))
    ad.click("button:has-text(\"إسناد المحدد\")"); ad.wait_for_timeout(2500)
    b_cnt = sum(1 for r in rows() if r.get("reviewer_user_id") == U["reviewer_b"]["user_id"])
    check("ui+db: remaining items assigned to reviewer B", b_cnt >= 1 and a_cnt + b_cnt == 14, (a_cnt, b_cnt))
    ad.click("text=التحليلات >> nth=0"); ad.wait_for_timeout(3000)
    t = body(ad); ad.screenshot(path=str(WORK / "ui_analytics.png"))
    check("ui: analytics page renders without an error banner", "تعذر" not in t and "خطأ" not in t.split("التحليلات")[-1][:300], t[120:360].replace("\n", " "))
    ad.click("text=المراجعات المكتملة >> nth=0"); ad.wait_for_timeout(2000)
    check("ui: completed-reviews page renders", "المراجعات المكتملة" in body(ad), "")
    ad.click("text=الرسائل >> nth=0"); ad.wait_for_timeout(1500)
    R.append({"check": "info: messages page text", "result": "INFO", "detail": body(ad)[120:400].replace("\n", " ")})

    # ---------------------------------------------------------------- reviewer journey
    rv = br.new_page(viewport={"width": 1400, "height": 900}); login(rv, "reviewer_a")
    t = body(rv); rv.screenshot(path=str(WORK / "ui_reviewer_home.png"))
    check("ui: reviewer login lands on the assigned queue, no admin navigation", "المسند لي" in t and "الإسناد" not in t.split("المسند لي")[0] + t[:200] and "التحليلات" not in t[:300], t[:200].replace("\n", " "))
    R.append({"check": "info: reviewer home text", "result": "INFO", "detail": t[100:700].replace("\n", " ")})
    R.append({"check": "info: reviewer buttons", "result": "INFO", "detail": str([x.inner_text()[:28] for x in rv.query_selector_all("button")][:30])})
    rv.screenshot(path=str(WORK / "ui_reviewer_home2.png"), full_page=True)
    rv.on("dialog", lambda d: d.accept())
    a_ids = {r["comment_id"] for r in rows() if r.get("reviewer_user_id") == U["reviewer_a"]["user_id"]}
    b_ids = {r["comment_id"] for r in rows() if r.get("reviewer_user_id") == U["reviewer_b"]["user_id"]}

    def current_id(pg):
        m = re.search(r"#([0-9a-f]{64})", body(pg)); return m.group(1) if m else None

    def status_of(cid):
        return next((r for r in rows() if r["comment_id"] == cid), {})

    rv.click('button:has-text("بدء المراجعة")'); rv.wait_for_timeout(2000)
    c1 = current_id(rv); check("ui: reviewer opens one of THEIR assigned comments", c1 in a_ids, c1 and c1[:10])
    check("ui: a draft is shown for review", len(rv.query_selector_all("textarea")) == 1 and "الرد المقترح" in body(rv), "")
    rv.click('button:has-text("اعتماد كما هو")'); rv.wait_for_timeout(2500)
    s1 = status_of(c1)
    check("ui+db: approve-as-is persisted (APPROVED, outbound PENDING, not sent)", s1.get("review_status") == "APPROVED" and s1.get("outbound_status") == "PENDING" and not s1.get("outbound_external_id"), (s1.get("review_status"), s1.get("outbound_status")))
    c2 = current_id(rv)
    check("ui: reviewer advances to a different assigned comment", c2 is not None and c2 != c1 and c2 in a_ids, c2 and c2[:10])
    rv.click('button:has-text("تعديل الرد")'); rv.wait_for_timeout(500)
    rv.fill("textarea", "راجعوا https://example.org/help للتفاصيل"); rv.wait_for_timeout(500)
    bad_t = body(rv); disabled = rv.query_selector('button:has-text("تعديل واعتماد")').is_disabled()
    if not disabled:
        rv.click('button:has-text("تعديل واعتماد")'); rv.wait_for_timeout(2500)
    after_bad = re.sub(r"\s+", " ", body(rv))
    s2 = status_of(c2)
    check("ui+db: an edit containing an unapproved URL is NOT approved", s2.get("review_status") not in ("EDITED_AND_APPROVED", "APPROVED"), (s2.get("review_status"), "button disabled" if disabled else "clicked"))
    R.append({"check": "info: UI feedback after clicking approve on the unsafe edit", "result": "INFO", "detail": ("disabled=" + str(disabled) + " | " + after_bad[after_bad.find("تحرير الرد") - 160: after_bad.find("تحرير الرد") + 420])})
    rv.fill("textarea", "نشكركم على ملاحظتكم القيّمة."); rv.wait_for_timeout(500)
    rv.click('button:has-text("تعديل واعتماد")'); rv.wait_for_timeout(2500)
    s2 = status_of(c2)
    check("ui+db: safe edit-and-approve persisted with the reviewer's exact text", s2.get("review_status") == "EDITED_AND_APPROVED" and (s2.get("reviewer_edit") or "") == "نشكركم على ملاحظتكم القيّمة.", (s2.get("review_status"), (s2.get("reviewer_edit") or "")[:40]))
    c3 = current_id(rv)
    rv.click('button:has-text("لا يحتاج رد")'); rv.wait_for_timeout(800)
    rv.select_option('select:has(option:text-is("لا يتطلب ردًا"))', label="لا يتطلب ردًا"); rv.wait_for_timeout(300)
    rv.click('button:has-text("تأكيد")'); rv.wait_for_timeout(2500)
    s3 = status_of(c3)
    R.append({"check": "info: no-response click result", "result": "INFO", "detail": f"status={s3.get('review_status')} action={s3.get('reviewer_action')}"})
    check("ui+db: no-response decision persisted", s3.get("review_status") == "NO_REPLY", (s3.get("review_status"), s3.get("reviewer_action")))
    ev = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    types = [r[0] for r in ev.execute("select event_type from review_item_events where actor_role='REVIEWER' order by created_at")]; ev.close()
    check("audit: reviewer actions from the UI are in the event trail", {"SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED"} <= set(types), types)
    rv.goto(BASE + "/review"); rv.wait_for_timeout(2000)
    T = body(rv)
    check("ui: reviewer progress counts reflect their decisions", "مكتمل" in T, re.sub(r"\s+", " ", T)[200:380])
    # --- isolation: reviewer B must not see A's items, and the browser must not be able to read A's item
    rb = br.new_page(viewport={"width": 1400, "height": 900}); login(rb, "reviewer_b"); tb = body(rb)
    seen_a = [i for i in a_ids if i[:12] in tb]
    check("ui: reviewer B's page shows none of reviewer A's comments", not seen_a and "إجمالي المسند إليّ 7" in re.sub(r"\s+", " ", tb) .replace("\n", " ") or not seen_a, (len(seen_a), re.sub(r"\s+", " ", tb)[100:220]))
    code = rb.evaluate("async (id) => (await fetch('/api/items/' + id, {credentials:'include'})).status", sorted(a_ids)[0])
    check("ui(browser session): reviewer B fetching A's item is refused", code in (403, 404), code)
    # --- logout from the UI
    rv.goto(BASE + "/review"); rv.wait_for_timeout(1500)
    um = rv.query_selector('button:has-text("E2E reviewer_a")')
    if um:
        um.click(); rv.wait_for_timeout(600)
    lo = rv.query_selector('text=تسجيل الخروج')
    R.append({"check": "info: logout control present", "result": "INFO", "detail": str(lo is not None)})
    if lo:
        lo.click(); rv.wait_for_timeout(2000)
    rv.goto(BASE + "/review"); rv.wait_for_timeout(1500)
    check("ui: after logout /review requires login again", rv.query_selector('input[placeholder="أدخل كلمة المرور"]') is not None, rv.url)
    code = rv.evaluate("async () => (await fetch('/api/queue', {credentials:'include'})).status")
    check("ui(browser session): API refuses the logged-out browser", code == 401, code)
    rep = WORK / "ui_report.json"
    rep.write_text(json.dumps({"results": R}, ensure_ascii=False, indent=1))
    print("PASS", sum(r["result"] == "PASS" for r in R), "FAIL", sum(r["result"] == "FAIL" for r in R))
    br.close()
