"""Admin <-> reviewer messaging end-to-end (real Chrome -> real Vite frontend -> real backend). SYNTHETIC users and messages only.

    python scripts/e2e/serve_e2e_mock.py --workdir <fresh workdir> --port 8794
    python scripts/e2e/verify_messaging_flow.py run <workdir> http://localhost:5193
Two independent browser contexts (admin, reviewer A) plus a third (reviewer B) for isolation."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path


def main(work: Path, base: str) -> int:
    from playwright.sync_api import sync_playwright

    U = json.loads((work / "e2e_users.json").read_text())
    DB = work / "e2e.sqlite"
    R: list[dict] = []

    def check(name, ok, detail=""):
        R.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)[:300]})
        print(("PASS" if ok else "FAIL"), "-", name, ("| " + str(detail)[:200]) if not ok and detail else "", flush=True)

    def q(sql, *a):
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); r = c.execute(sql, a).fetchall(); c.close(); return r

    def login(br, who):
        pg = br.new_context(viewport={"width": 1400, "height": 900}).new_page()
        pg.on("dialog", lambda d: d.accept())
        pg.goto(base + "/"); pg.wait_for_timeout(600)
        pg.fill('input[placeholder="أدخل البريد الإلكتروني"]', U[who]["email"]); pg.fill('input[placeholder="أدخل كلمة المرور"]', U[who]["password"])
        pg.click("button[type=submit]"); pg.wait_for_timeout(1800)
        return pg

    def open_messages(pg):
        pg.click("button:has-text('الرسائل') >> nth=0"); pg.wait_for_timeout(1500)

    def body(pg):
        return re.sub(r"\s+", " ", pg.inner_text("body"))

    def send(pg, text):
        pg.fill("#message-draft", text); pg.click("button:has-text('إرسال')"); pg.wait_for_timeout(1500)

    A, B = U["reviewer_a"]["user_id"], U["reviewer_b"]["user_id"]
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True)
        adm, ra, rb = login(br, "admin"), login(br, "reviewer_a"), login(br, "reviewer_b")

        open_messages(adm)
        check("admin: the real Messages page replaces the unavailable placeholder", "التراسل الداخلي غير متاح" not in body(adm) and "محادثات العمل" in body(adm), "")
        names = [t.inner_text() for t in adm.query_selector_all(".messages-thread-item")]
        check("admin: lists a conversation for every reviewer, none with messages yet", len(names) == 2 and all("لا توجد رسائل بعد" in n for n in names), len(names))
        adm.click(f".messages-thread-item:has-text('E2E reviewer_a')"); adm.wait_for_timeout(800)
        check("admin: the composer cannot send an empty or whitespace-only message", adm.is_disabled("button:has-text('إرسال')"), "")
        adm.fill("#message-draft", "   "); check("admin: whitespace-only stays disabled", adm.is_disabled("button:has-text('إرسال')"), "")
        send(adm, "مرحبًا، يرجى إعطاء الأولوية للتعليقات الأقدم.")
        check("admin: the sent message appears in the conversation with sender and time", "مرحبًا، يرجى إعطاء الأولوية للتعليقات الأقدم." in body(adm) and "مسؤول النظام" in body(adm), "")
        check("admin: the composer is cleared after sending", adm.input_value("#message-draft") == "", "")
        adm.click(".messages-thread-item:has-text('E2E reviewer_b')"); adm.wait_for_timeout(800)          # looking at another conversation, so A's reply will stay unread

        open_messages(ra)
        check("reviewer A: opens the same page and sees the admin message flagged as new", "مرحبًا، يرجى إعطاء الأولوية" in body(ra) and "جديدة" in body(ra), "")
        check("reviewer A: sees no list of other reviewers", len(ra.query_selector_all(".messages-thread-item")) == 1 and "E2E reviewer_b" not in body(ra), "")
        ra.wait_for_timeout(600)
        check("the opened message was marked read in the database (by the reviewer, not the sender)", q("select count(*) from direct_messages where sender_role='ADMIN' and read_at is not null")[0][0] == 1, "")
        send(ra, "تم، سأبدأ بالأقدم.")
        check("reviewer A: the reply appears with reviewer identity", "تم، سأبدأ بالأقدم." in body(ra), "")

        ra.reload(); ra.wait_for_timeout(1500); open_messages(ra)
        txt = body(ra)
        check("refresh persistence (reviewer): both messages survive a reload, in order, none flagged new", txt.index("مرحبًا، يرجى") < txt.index("تم، سأبدأ") and "جديدة" not in txt, "")

        adm.reload(); adm.wait_for_timeout(1500); open_messages(adm)
        row = adm.query_selector(".messages-thread-item:has-text('E2E reviewer_a')")
        check("admin: the reviewer's reply shows as an unread count on that conversation after a reload", row is not None and "1 غير مقروءة" in row.inner_text() and "تم، سأبدأ بالأقدم." in row.inner_text(), row and row.inner_text())
        check("admin: the other reviewer's conversation shows no unread", "غير مقروءة" not in adm.query_selector(".messages-thread-item:has-text('E2E reviewer_b')").inner_text(), "")
        adm.click(".messages-thread-item:has-text('E2E reviewer_a')"); adm.wait_for_timeout(1500)
        check("admin: opening the conversation shows the full history and the new reply", "مرحبًا، يرجى" in body(adm) and "تم، سأبدأ بالأقدم." in body(adm) and "جديدة" in body(adm), "")
        adm.wait_for_timeout(500)
        check("opening cleared the unread state in the database", q("select count(*) from direct_messages where sender_role='REVIEWER' and read_at is null")[0][0] == 0, "")

        open_messages(rb)
        check("reviewer B: sees none of A's messages", "مرحبًا، يرجى" not in body(rb) and "تم، سأبدأ" not in body(rb) and "لا توجد رسائل بعد" in body(rb), "")
        st = rb.evaluate("""async (a) => ({
            read: (await fetch('/api/messages/conversation?reviewer_user_id=' + a, {credentials:'include'})).status,
            send: (await fetch('/api/messages/send', {method:'POST', credentials:'include', headers:{'Content-Type':'application/json'}, body: JSON.stringify({reviewer_user_id:a, body:'x', client_message_id:'forged-id-0001'})})).status,
            mark: (await fetch('/api/messages/read', {method:'POST', credentials:'include', headers:{'Content-Type':'application/json'}, body: JSON.stringify({reviewer_user_id:a})})).status,
            list: (await fetch('/api/messages/conversations', {credentials:'include'})).status })""", A)
        check("reviewer B: direct API access to A's conversation is refused (read/send/mark 403) and the admin list is forbidden", st == {"read": 403, "send": 403, "mark": 403, "list": 403}, st)
        anon = br.new_context().new_page(); anon.goto(base + "/"); anon.wait_for_timeout(500)
        check("unauthenticated: every messaging endpoint answers 401", anon.evaluate("""async () => (await Promise.all(['/api/messages/conversations','/api/messages/unread','/api/messages/conversation'].map(p => fetch(p).then(r => r.status)))).join(',')""") == "401,401,401", "")

        dup = ra.evaluate("""async () => { const call = () => fetch('/api/messages/send', {method:'POST', credentials:'include', headers:{'Content-Type':'application/json'}, body: JSON.stringify({body:'مرة واحدة', client_message_id:'dup-test-id-01'})}).then(r => r.json()); const rs = await Promise.all([call(), call(), call(), call()]); return rs.map(r => r.message_id) }""")
        check("duplicate protection: four simultaneous identical requests create exactly one stored message", len(set(dup)) == 1 and q("select count(*) from direct_messages where client_message_id='dup-test-id-01'")[0][0] == 1, dup)

        # returning to reviewing loses nothing: the Messages state is separate from review work
        before = q("select count(*), sum(row_version) from review_items")
        ra.click("button:has-text('المسند لي') >> nth=0"); ra.wait_for_timeout(1200); open_messages(ra)
        check("reviewer A: returning to the review list and back keeps the conversation; review data is untouched by messaging", "تم، سأبدأ بالأقدم." in body(ra) and q("select count(*), sum(row_version) from review_items") == before, "")

        rows = q("select sender_role, sender_user_id, reviewer_user_id, created_at, read_at from direct_messages order by created_at, message_id")
        check("database: sender identity and conversation key are correct and timestamps are present", [(r[0], r[1], r[2]) for r in rows[:2]] == [("ADMIN", U["admin"]["user_id"], A), ("REVIEWER", A, A)] and all(r[3] for r in rows), rows[:2])
        check("database: no message exists for reviewer B's conversation", q("select count(*) from direct_messages where reviewer_user_id=?", B)[0][0] == 0, "")
        br.close()
    out = work / "messaging_flow_report.json"
    out.write_text(json.dumps({"results": R, "pass": sum(r["result"] == "PASS" for r in R), "fail": sum(r["result"] == "FAIL" for r in R)}, ensure_ascii=False, indent=1))
    print("\nPASS", sum(r["result"] == "PASS" for r in R), "FAIL", sum(r["result"] == "FAIL" for r in R))
    return 0 if all(r["result"] == "PASS" for r in R) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[2]), sys.argv[3]))
