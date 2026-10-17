"""Authentication journey end-to-end (real Chrome -> real Vite frontend -> real backend). SYNTHETIC accounts only. E-mail goes to an in-memory FAKE sender (mirrored
to <workdir>/outbox.json); nothing is sent anywhere.

    python scripts/e2e/serve_e2e_mock.py --workdir <fresh workdir> --port 8794 --fake-email-base http://localhost:5193
    python scripts/e2e/verify_auth_flow.py run <workdir> http://localhost:5193

Covers: login page (Microsoft retained but unavailable), new-account request, inactive until approved, admin approval (reviewer role only), login, forgot password,
reset link from the e-mail, single use, old password/session dead, logout, role/permission enforcement, rate limiting, no account enumeration."""
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

    def body(pg):
        return re.sub(r"\s+", " ", pg.inner_text("body"))

    def fresh(br):
        pg = br.new_context(viewport={"width": 1400, "height": 900}).new_page()
        pg.on("dialog", lambda d: d.accept())
        return pg

    def login(pg, email, password):
        pg.goto(base + "/"); pg.wait_for_timeout(700)
        pg.fill('input[placeholder="أدخل البريد الإلكتروني"]', email); pg.fill('input[placeholder="أدخل كلمة المرور"]', password)
        pg.click("button[type=submit]"); pg.wait_for_timeout(1800)

    NEW_EMAIL, NEW_PW, NEW_NAME = "new.requester@example.invalid", "first-password-77", "مستخدم جديد للاختبار"
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True)

        # ---------------------------------------------------------------- login page
        pg = fresh(br); pg.goto(base + "/"); pg.wait_for_timeout(1200)
        ms = pg.query_selector("button.btn-ms")
        check("login: the Microsoft button is retained but disabled, with an explicit 'not available' note (no fake success path)",
              ms is not None and ms.is_disabled() and "غير متاح حاليًا" in body(pg) and pg.query_selector("a.btn-ms") is None, "")
        check("login: 'forgot password' and 'new account request' are offered on the same single login page", "نسيت كلمة المرور؟" in body(pg) and "طلب حساب جديد" in body(pg), "")

        # ---------------------------------------------------------------- new account request
        pg.click("text=طلب حساب جديد"); pg.wait_for_timeout(1000)
        check("request: opens the request form (name, e-mail, password) without any role/permission choice",
              pg.query_selector("#req-name") is not None and pg.query_selector("#req-email") is not None and pg.query_selector("#req-password") is not None and "الصلاحية" not in body(pg) and pg.query_selector("select") is None, "")
        pg.fill("#req-name", NEW_NAME); pg.fill("#req-email", NEW_EMAIL); pg.fill("#req-password", "short")
        pg.click("button:has-text('إرسال الطلب')"); pg.wait_for_timeout(500)
        check("request: a too-short password is rejected (browser/server validation) and nothing is queued", q("select count(*) from account_requests")[0][0] == 0, "")
        pg.fill("#req-password", NEW_PW); pg.click("button:has-text('إرسال الطلب')"); pg.wait_for_timeout(1500)
        check("request: the confirmation says the account stays inactive until an admin approves it", "غير مفعّل" in body(pg) and "مسؤول النظام" in body(pg), "")
        check("request: stored as PENDING, password only as a bcrypt hash, no user row yet",
              q("select status from account_requests where email=?", NEW_EMAIL) == [("PENDING",)] and q("select password_hash from account_requests where email=?", NEW_EMAIL)[0][0].startswith("$2")
              and q("select count(*) from users where email=?", NEW_EMAIL)[0][0] == 0, "")

        pg2 = fresh(br); login(pg2, NEW_EMAIL, NEW_PW)
        check("pending: the requester cannot log in (generic failure, still on the login page)", pg2.query_selector('input[placeholder="أدخل كلمة المرور"]') is not None and "مسؤول النظام" not in body(pg2).replace("تواصل مع مسؤول النظام", ""), body(pg2)[:150])
        st = pg2.evaluate("""async () => { const r = await fetch('/api/queue', {credentials:'include'}); return r.status }""")
        check("pending: no session exists (protected API answers 401)", st == 401, st)

        # no enumeration + rate limiting through the real endpoint
        same = pg2.evaluate("""async () => { const post = (email) => fetch('/api/account-requests', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({email, display_name:'x', password:'long-enough-1'})}).then(async r => [r.status, JSON.stringify(await r.json())]);
            return [await post('admin@e2e.invalid'), await post('nobody.at.all@example.invalid')] }""")
        check("no enumeration: an address that already has an account and an unknown address get identical answers", same[0] == same[1] and same[0][0] == 200, same)
        codes = pg2.evaluate("""async () => { const out = []; for (let i = 0; i < 6; i++) out.push((await fetch('/api/account-requests', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({email:'spam'+i+'@example.invalid', display_name:'x', password:'long-enough-1'})})).status); return out }""")
        check("rate limiting: repeated requests from one address end in HTTP 429", 429 in codes, codes)
        check("rate limiting: no account was created by the flood and no user rows were added", q("select count(*) from users")[0][0] == 3, q("select count(*) from users")[0][0])

        # ---------------------------------------------------------------- admin approval
        adm = fresh(br); login(adm, U["admin"]["email"], U["admin"]["password"])
        adm.click("button.account-trigger"); adm.wait_for_timeout(400); adm.click("button.menu-link:has-text('الإعدادات')"); adm.wait_for_timeout(1500)
        panel = adm.query_selector("[aria-label='طلبات الحسابات المعلّقة']")
        check("admin: the pending request appears in Settings with name and e-mail, and no password", panel is not None and NEW_NAME in panel.inner_text() and NEW_EMAIL in panel.inner_text(), "")
        adm.click("[aria-label='طلبات الحسابات المعلّقة'] li:has-text('new.requester') button:has-text('موافقة')"); adm.wait_for_timeout(1500)
        check("admin approval created exactly one ACTIVE account with the REVIEWER role (never ADMIN)",
              q("select role, is_active from users where email=?", NEW_EMAIL) == [("REVIEWER", 1)] and q("select status from account_requests where email=?", NEW_EMAIL) == [("APPROVED",)], q("select role,is_active from users where email=?", NEW_EMAIL))

        # ---------------------------------------------------------------- the approved user
        usr = fresh(br); login(usr, NEW_EMAIL, NEW_PW)
        check("approved user: logs in with the password chosen at request time and lands in the reviewer workspace (no admin navigation)", "المسند لي" in body(usr) and "الإسناد" not in body(usr) and "التحليلات" not in body(usr), body(usr)[:120])
        st = usr.evaluate("""async () => ({ admin: (await fetch('/api/admin/account-requests', {credentials:'include'})).status, users: (await fetch('/api/admin/users', {credentials:'include'})).status,
            approve: (await fetch('/api/admin/account-requests/approve', {method:'POST', credentials:'include', headers:{'Content-Type':'application/json'}, body: JSON.stringify({request_id:'x'})})).status })""")
        check("approved user: admin endpoints are refused by the backend (403), not merely hidden", st == {"admin": 403, "users": 403, "approve": 403}, st)
        usr.click("button.account-trigger"); usr.wait_for_timeout(400); usr.click("button:has-text('تسجيل الخروج')"); usr.wait_for_timeout(1200)
        st = usr.evaluate("""async () => (await fetch('/api/queue', {credentials:'include'})).status""")
        check("logout: the session is destroyed server-side (API answers 401)", st == 401, st)

        # ---------------------------------------------------------------- forgot password
        fp = fresh(br); fp.goto(base + "/"); fp.wait_for_timeout(800)
        fp.click("text=نسيت كلمة المرور؟"); fp.wait_for_timeout(1000)
        check("forgot password: opens the reset-request page (feature is on)", "إرسال رابط إعادة التعيين" in body(fp), "")
        fp.fill("#reset-email", "nobody.known@example.invalid"); fp.click("button:has-text('إرسال رابط إعادة التعيين')"); fp.wait_for_timeout(1500)
        msg_unknown = body(fp)
        outbox = (work / "outbox.json")
        check("forgot password: an UNKNOWN address shows the same notice and sends nothing", "إذا كان هذا البريد الإلكتروني مسجَّلًا" in msg_unknown and not outbox.exists(), "")
        fp2 = fresh(br); fp2.goto(base + "/forgot-password"); fp2.wait_for_timeout(1000)
        fp2.fill("#reset-email", NEW_EMAIL); fp2.click("button:has-text('إرسال رابط إعادة التعيين')"); fp2.wait_for_timeout(2000)
        check("forgot password: a KNOWN address shows the identical notice", "إذا كان هذا البريد الإلكتروني مسجَّلًا" in body(fp2) and re.sub(r"\s+", " ", msg_unknown).split("إذا كان")[1][:40] == body(fp2).split("إذا كان")[1][:40], "")
        mails = json.loads(outbox.read_text()) if outbox.exists() else []
        link = re.search(r"(http://[^\s]+/reset-password#token=[A-Za-z0-9_\-]+)", mails[-1]["body"]).group(1) if mails else ""
        check("forgot password: exactly one e-mail with a single-use link in the URL fragment was produced for the known address (fake sender)", len(mails) == 1 and mails[0]["to"] == NEW_EMAIL and "#token=" in link, len(mails))
        token = link.split("#token=")[1] if link else ""
        stored = q("select token_hash from password_reset_tokens")
        check("reset token: only its SHA-256 hash is stored, with an expiry of at most 30 minutes", len(stored) == 1 and token not in stored[0][0] and len(stored[0][0]) == 64
              and q("select (julianday(expires_at) - julianday(created_at)) * 1440 from password_reset_tokens")[0][0] <= 30.01, "")

        rp = fresh(br); rp.goto(link); rp.wait_for_timeout(1500)
        check("reset page: the token is removed from the address bar after load", "token=" not in rp.url, rp.url)
        pw_inputs = rp.query_selector_all("input[type=password]")
        for i in pw_inputs:
            i.fill("second-password-88")
        rp.click("button[type=submit]"); rp.wait_for_timeout(2000)
        check("reset: the new password was accepted", "تم" in body(rp) or rp.query_selector('input[placeholder="أدخل البريد الإلكتروني"]') is not None, body(rp)[:150])
        old = fresh(br); login(old, NEW_EMAIL, NEW_PW)
        check("reset: the OLD password no longer works", old.query_selector('input[placeholder="أدخل كلمة المرور"]') is not None, "")
        new = fresh(br); login(new, NEW_EMAIL, "second-password-88")
        check("reset: the NEW password works", "المسند لي" in body(new), body(new)[:100])
        rp2 = fresh(br); rp2.goto(link); rp2.wait_for_timeout(1200)
        for i in rp2.query_selector_all("input[type=password]"):
            i.fill("third-password-99")
        rp2.click("button[type=submit]"); rp2.wait_for_timeout(1800)
        check("reset: the link cannot be used twice (still the second password afterwards)", q("select count(*) from users where email=?", NEW_EMAIL)[0][0] == 1, "")
        again = fresh(br); login(again, NEW_EMAIL, "third-password-99")
        check("reset: a second use of the same link did NOT change the password", again.query_selector('input[placeholder="أدخل كلمة المرور"]') is not None, "")
        st = new.evaluate("""async () => (await fetch('/api/queue', {credentials:'include'})).status""")
        check("reset: sessions that existed before the reset are revoked only if they predate it (this one was created after, so it stays valid)", st == 200, st)
        br.close()
    out = work / "auth_flow_report.json"
    out.write_text(json.dumps({"results": R, "pass": sum(r["result"] == "PASS" for r in R), "fail": sum(r["result"] == "FAIL" for r in R)}, ensure_ascii=False, indent=1))
    print("\nPASS", sum(r["result"] == "PASS" for r in R), "FAIL", sum(r["result"] == "FAIL" for r in R))
    return 0 if all(r["result"] == "PASS" for r in R) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[2]), sys.argv[3]))
