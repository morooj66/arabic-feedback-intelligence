"""Reviewer workflow end-to-end (real Chrome via Playwright -> real Vite frontend -> real backend). SYNTHETIC data only; no Gemini, no external network.

    python scripts/e2e/verify_reviewer_flow.py seed  <workdir>                        # creates <workdir>/e2e.sqlite + e2e_users.json and the synthetic review items
    python scripts/e2e/serve_e2e_mock.py --workdir <workdir> --port 8794             # (reuses the seeded database and users)
    python scripts/e2e/verify_reviewer_flow.py run   <workdir> http://localhost:5193 # Vite dev server proxying /api to the backend above

Needs Playwright in a scratch venv with the system Chrome (nothing is downloaded). Writes <workdir>/reviewer_flow_report.json; prints PASS/FAIL per check.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import secrets
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

N_R1, N_R2, N_FREE, N_DONE_R1 = 30, 10, 5, 2


def seed(work: Path) -> None:
    from sqlalchemy.orm import Session

    from rcs.review.auth import ADMIN, REVIEWER, create_user
    from rcs.review.models import ReviewItem
    from rcs.review.persistence import build_repository_bundle
    from rcs.review.repository import get_engine, init_db

    work.mkdir(parents=True, exist_ok=True)
    bundle, engine = build_repository_bundle(engine_factory=lambda: get_engine(work / "e2e.sqlite"))
    init_db(engine)
    users = {}
    for key, role in (("admin", ADMIN), ("reviewer_a", REVIEWER), ("reviewer_b", REVIEWER)):
        pw = secrets.token_urlsafe(12)
        u = create_user(bundle, email=f"{key}@flow.invalid", password=pw, display_name=f"Flow {key}", role=role)
        users[key] = {"email": f"{key}@flow.invalid", "password": pw, "user_id": u.user_id, "role": role}
    (work / "e2e_users.json").write_text(json.dumps(users))
    t0 = dt.datetime(2026, 8, 1, 9, 0, 0)
    intents = ["SUGGESTION", "INQUIRY", "COMPLAINT", "THANKS_PRAISE"]
    topics = ["WORD_SUBMISSION", "GENERAL", "TECHNICAL_ISSUE", "GENERAL"]

    def add(s, i, prefix, owner, *, status="DRAFT_READY", action=None, draft=True):
        k = i % 4
        s.add(ReviewItem(
            comment_id=f"{prefix}{i:03d}", source_file="flow.xlsx", source_row_index=i, original_comment=f"تعليق اختباري رقم {prefix}{i:03d} حول إضافة الأمثلة إلى المعجم",
            review_status=status, ai_primary_intent=intents[k], primary_intent=intents[k], ai_topic_subtype=topics[k], topic_subtype=topics[k], sentiment="NEUTRAL",
            ai_suggested_final_reply=("نشكركم على ملاحظتكم القيّمة." if draft else None), suggested_final_reply=("نشكركم على ملاحظتكم القيّمة." if draft else None),
            reviewer_user_id=owner, reviewer_action=action, review_priority=(i % 3) + 1, ingested_at=t0 + dt.timedelta(minutes=i), comment_created_at=t0 + dt.timedelta(minutes=i),
            row_version=1, ready_to_send=action is not None))
    with Session(engine) as s:
        for i in range(N_R1):
            add(s, i, "a", users["reviewer_a"]["user_id"], draft=(i % 10 != 9))             # a009, a019, a029: no draft (REVIEW_ONLY style)
            if i % 10 == 9:
                s.flush(); s.query(ReviewItem).filter_by(comment_id=f"a{i:03d}").update({"review_status": "REVIEW_ONLY"})
        for i in range(N_DONE_R1):
            add(s, 100 + i, "d", users["reviewer_a"]["user_id"], status="APPROVED", action="SEND_AS_IS")
        for i in range(N_R2):
            add(s, i, "b", users["reviewer_b"]["user_id"])
        for i in range(N_FREE):
            add(s, i, "u", None)
        s.commit()
    print("seeded:", {"reviewer_a_pending": N_R1, "reviewer_a_completed": N_DONE_R1, "reviewer_b": N_R2, "unassigned": N_FREE})


def run(work: Path, base: str) -> int:
    from playwright.sync_api import sync_playwright

    U = json.loads((work / "e2e_users.json").read_text())
    DB = work / "e2e.sqlite"
    R: list[dict] = []

    def check(name, ok, detail=""):
        R.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)[:300]})
        print(("PASS" if ok else "FAIL"), "-", name, ("| " + str(detail)[:200]) if not ok and detail else "", flush=True)
        return ok

    def row(cid):
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); c.row_factory = sqlite3.Row
        r = c.execute("select * from review_items where comment_id=?", (cid,)).fetchone(); c.close()
        return dict(r) if r else None

    def scalar(sql, *a):
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); v = c.execute(sql, a).fetchone()[0]; c.close(); return v

    def login(pg, who):
        pg.goto(base + "/"); pg.wait_for_timeout(600)
        pg.fill('input[placeholder="أدخل البريد الإلكتروني"]', U[who]["email"]); pg.fill('input[placeholder="أدخل كلمة المرور"]', U[who]["password"])
        pg.click("button[type=submit]"); pg.wait_for_timeout(1600)

    def body(pg):
        return re.sub(r"\s+", " ", pg.inner_text("body"))

    def current_id(pg):
        m = re.search(r"#([a-z]\d{3})", body(pg))
        return m.group(1) if m else None

    def position(pg):
        m = re.search(r"(\d+) من (\d+)", pg.inner_text(".focus-nav"))
        return (int(m.group(1)), int(m.group(2))) if m else None

    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True)
        ctx = br.new_context(viewport={"width": 1400, "height": 900}); rv = ctx.new_page(); rv.on("dialog", lambda d: d.accept())
        login(rv, "reviewer_a")
        t = body(rv)
        check("login: reviewer lands on the assigned list, no admin navigation", "المسند لي" in t and "الإسناد" not in t[:300] and "التحليلات" not in t[:300], t[:160])
        check("assigned only: totals reflect only reviewer A's comments", f"إجمالي المسند إليّ {N_R1 + N_DONE_R1}" in t and f"بانتظار المراجعة {N_R1}" in t and "مكتمل 2" in t, re.sub(r"\s+", " ", t)[100:260])
        check("assigned only: no comment of reviewer B or an unassigned one is listed", not re.search(r"#[bu]\d{3}", t), "")
        api_total = rv.evaluate("fetch('/api/queue?limit=200').then(r=>r.json()).then(j=>j.total)")
        check("backend: queue total equals the assigned pending count", api_total == N_R1, api_total)

        # -------- filters
        rv.fill('input[aria-label="بحث"]', "a01"); rv.wait_for_timeout(1200)
        t = body(rv); ids = set(re.findall(r"#(a\d{3})", t))
        check("filter: search narrows the list to the matching comments", ids and all(i.startswith("a01") for i in ids) and "قائمة المراجعة (10)" in t, (sorted(ids)[:4], re.findall(r"قائمة المراجعة \(\d+\)", t)))
        rv.fill('input[aria-label="بحث"]', ""); rv.wait_for_timeout(800)
        # -------- start a review session from the filtered list and walk forward through ALL of it
        rv.fill('input[aria-label="بحث"]', "رقم a0"); rv.wait_for_timeout(1200)             # a000..a029 all contain "رقم a0" -> 30 matches, 4 pages of 8
        t = body(rv)
        check("filter: the working set is larger than one page", "قائمة المراجعة (30)" in t, re.findall(r"قائمة المراجعة \(\d+\)", t))
        rv.click('button:has-text("بدء المراجعة")'); rv.wait_for_timeout(2000)
        first = current_id(rv)
        check("session: starts on the first comment of the filtered list", first is not None and first.startswith("a"), first)
        check("classification is display-only: no intent/topic/sentiment control in the review screen",
              not rv.query_selector_all('select:has(option:text-is("اقتراح"))') and len(rv.query_selector_all("textarea")) == 1 and "تغيير التصنيف" not in body(rv), "")
        seen = [first]
        for _ in range(40):
            nxt = rv.query_selector('button[aria-label="التعليق التالي"]')
            if nxt is None or nxt.is_disabled():
                break
            nxt.click(); rv.wait_for_timeout(900)
            cid = current_id(rv)
            if cid in seen:
                break
            seen.append(cid)
        check("navigation: 'next' reaches every comment of the filtered list (30), across page boundaries", len(seen) == 30 and len(set(seen)) == 30, (len(seen), seen[-3:]))
        pos = position(rv)
        check("navigation: the position indicator counts the whole filtered list", pos == (30, 30), pos)
        back = [current_id(rv)]
        for _ in range(40):
            prv = rv.query_selector('button[aria-label="التعليق السابق"]')
            if prv is None or prv.is_disabled():
                break
            prv.click(); rv.wait_for_timeout(900); back.append(current_id(rv))
        check("navigation: 'previous' walks all the way back to the first comment", back[-1] == first and len(set(back)) == 30, (len(back), back[-1], first))

        # -------- decisions
        c1 = current_id(rv)
        rv.click('button:has-text("اعتماد كما هو")'); rv.wait_for_timeout(2500)
        s1 = row(c1)
        check("approve: persisted as APPROVED with the draft as the final reply", s1["review_status"] == "APPROVED" and s1["reviewer_action"] == "SEND_AS_IS" and s1["final_reply"] == "نشكركم على ملاحظتكم القيّمة.", (s1["review_status"], s1["final_reply"]))
        c2 = current_id(rv)
        check("approve: the screen moved on to the next pending comment", c2 not in (None, c1) and row(c2)["review_status"] in ("DRAFT_READY", "IN_REVIEW", "REVIEW_ONLY"), (c1, c2))
        rv.click('button:has-text("تعديل الرد")'); rv.wait_for_timeout(400)
        rv.fill("textarea", "نشكركم على تواصلكم، وقد سُجِّلت ملاحظتكم للدراسة."); rv.wait_for_timeout(300)
        rv.click('button:has-text("تعديل واعتماد")'); rv.wait_for_timeout(2500)
        s2 = row(c2)
        check("edit and approve: persisted with the reviewer's exact text", s2["review_status"] == "EDITED_AND_APPROVED" and s2["reviewer_edit"] == "نشكركم على تواصلكم، وقد سُجِّلت ملاحظتكم للدراسة." and s2["suggested_final_reply"] == "نشكركم على ملاحظتكم القيّمة.", (s2["review_status"], s2["reviewer_edit"]))
        c3 = current_id(rv)
        rv.click('button:has-text("لا يحتاج رد")'); rv.wait_for_timeout(700)
        rv.select_option('select:has(option:text-is("لا يتطلب ردًا"))', label="لا يتطلب ردًا"); rv.wait_for_timeout(300)
        rv.click('button:has-text("تأكيد")'); rv.wait_for_timeout(2500)
        s3 = row(c3)
        check("no response needed: persisted as NO_REPLY with its reason", s3["review_status"] == "NO_REPLY" and s3["reviewer_action"] == "NO_RESPONSE_NEEDED" and (s3["no_response_reason"] or ""), (s3["review_status"], s3["no_response_reason"]))
        for cid in (c1, c2, c3):
            check(f"classification untouched by the decision ({cid})", row(cid)["primary_intent"] == row(cid)["ai_primary_intent"] and row(cid)["topic_subtype"] == row(cid)["ai_topic_subtype"], "")

        # -------- progress and completed vs pending, then refresh and resume
        decided = N_DONE_R1 + 3
        rv.reload(); rv.wait_for_timeout(2500)
        t = body(rv)
        check("refresh: decisions survived and the counters are correct", f"بانتظار المراجعة {N_R1 - 3}" in t and f"مكتمل {decided}" in t, re.sub(r"\s+", " ", t)[100:300])
        pend_ids = set(rv.evaluate("fetch('/api/queue?limit=200&filter=pending').then(r=>r.json()).then(j=>j.items.map(i=>i.comment_id))"))
        done_ids = set(rv.evaluate("fetch('/api/queue?limit=200&filter=completed').then(r=>r.json()).then(j=>j.items.map(i=>i.comment_id))"))
        check("pending and completed are disjoint and add up", not pend_ids & done_ids and len(pend_ids) == N_R1 - 3 and len(done_ids) == decided and {c1, c2, c3} <= done_ids, (len(pend_ids), len(done_ids)))
        rv.goto(base + "/review"); rv.wait_for_timeout(2000)
        rv.click('button:has-text("بدء المراجعة")'); rv.wait_for_timeout(2000)
        resumed = current_id(rv)
        check("resume: after a refresh the session restarts on a still-pending comment, never a decided one", resumed is not None and resumed not in {c1, c2, c3} and row(resumed)["review_status"] in ("DRAFT_READY", "IN_REVIEW", "REVIEW_ONLY"), resumed)
        # claimed-but-undecided comments (opened earlier while navigating) remain pending
        claimed = scalar("select count(*) from review_items where reviewer_user_id=? and review_status='IN_REVIEW'", U["reviewer_a"]["user_id"])
        check("claimed but undecided comments stay in the pending count", claimed >= 1 and scalar("select count(*) from review_items where reviewer_user_id=? and reviewer_action is null", U["reviewer_a"]["user_id"]) == N_R1 - 3, claimed)

        # -------- concurrent changes: the same reviewer in two tabs
        tab2 = ctx.new_page(); tab2.on("dialog", lambda d: d.accept()); tab2.goto(base + "/review"); tab2.wait_for_timeout(2000)
        tab2.click('button:has-text("بدء المراجعة")'); tab2.wait_for_timeout(2000)
        same = current_id(tab2)
        check("concurrency: both tabs opened the same pending comment", same == resumed, (same, resumed))
        X, Y = "نشكركم على ملاحظتكم الأولى وسنأخذها بالاعتبار.", "رد مختلف من التبويب الثاني للتجربة."
        rv.click('button:has-text("تعديل الرد")'); rv.wait_for_timeout(300); rv.fill("textarea", X); rv.wait_for_timeout(300)
        rv.click('button:has-text("تعديل واعتماد")'); rv.wait_for_timeout(2500)
        tab2.click('button:has-text("تعديل الرد")'); tab2.wait_for_timeout(300); tab2.fill("textarea", Y); tab2.wait_for_timeout(300)
        tab2.click('button:has-text("تعديل واعتماد")'); tab2.wait_for_timeout(2500)
        s4 = row(same)
        check("concurrency: the first decision stands and the stale tab cannot overwrite it", s4["review_status"] == "EDITED_AND_APPROVED" and s4["reviewer_edit"] == X, (s4["review_status"], s4["reviewer_edit"]))
        tt = body(tab2)
        check("concurrency: the stale tab tells the reviewer instead of failing silently", bool(re.search(r"تغيّر|تغير|محدّث|حدّث|حُدّث|سابق|مراجع آخر|اعتُمد|تعذر|خطأ|متزامن|نسخة", tt)), tt[:240])
        events = scalar("select count(*) from review_item_events where review_item_id=? and event_type in ('SEND_AS_IS','EDIT_AND_SEND')", same)
        check("concurrency: exactly one decision event was recorded", events == 1, events)
        tab2.close()

        # -------- a comment reassigned away while it is open
        victim = current_id(rv)
        if victim:
            c = sqlite3.connect(DB); c.execute("update review_items set reviewer_user_id=? where comment_id=?", (U["reviewer_b"]["user_id"], victim)); c.commit(); c.close()
            code = rv.evaluate("(id)=>fetch('/api/items/'+id+'/send-as-is',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(r=>r.status)", victim)
            check("reassigned away: the former reviewer is refused by the backend", code in (403, 404), code)
            check("reassigned away: nothing was decided", row(victim)["reviewer_action"] is None, row(victim)["review_status"])

        # -------- backend authorization from a browser session
        rb = br.new_context(viewport={"width": 1400, "height": 900}).new_page(); login(rb, "reviewer_b")
        tb = body(rb)
        others = [i for i in re.findall(r"#([au]\d{3})", tb) if i != victim]                  # the comment reassigned to B above is legitimately theirs now
        check("reviewer B sees only their own comments (including the one just reassigned to them)", not others and (victim is None or victim in re.findall(r"#([a-z]\d{3})", tb)), others[:3])
        a_item = "a005"
        codes = rb.evaluate("""async (id)=>{const out={};
            out.get=(await fetch('/api/items/'+id)).status;
            for (const a of ['claim','send-as-is','no-response','edit-and-send']) out[a]=(await fetch('/api/items/'+id+'/'+a,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reason:'x',edited_response:'x'})})).status;
            out.hist=(await fetch('/api/admin/items/'+id+'/history')).status; out.assign=(await fetch('/api/admin/assign',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({comment_ids:[id],reviewer_user_id:'x'})})).status;
            out.export=(await fetch('/api/excel/export')).status; return out;}""", a_item)
        check("authorization: reviewer B is refused on every read/write of reviewer A's comment and on admin endpoints", all(v in (403, 404) for v in codes.values()), codes)
        before = row(a_item)
        check("authorization: nothing changed on A's comment", before["reviewer_user_id"] == U["reviewer_a"]["user_id"] and before["reviewer_action"] is None, before["review_status"])
        anon = br.new_context().new_page(); anon.goto(base + "/"); anon.wait_for_timeout(500)
        acode = anon.evaluate("fetch('/api/queue').then(r=>r.status)")
        check("authorization: an unauthenticated browser gets 401", acode == 401, acode)
        br.close()
    out = work / "reviewer_flow_report.json"
    out.write_text(json.dumps({"results": R, "pass": sum(r["result"] == "PASS" for r in R), "fail": sum(r["result"] == "FAIL" for r in R)}, ensure_ascii=False, indent=1))
    print("\nPASS", sum(r["result"] == "PASS" for r in R), "FAIL", sum(r["result"] == "FAIL" for r in R))
    return 0 if all(r["result"] == "PASS" for r in R) else 1


if __name__ == "__main__":
    mode, workdir = sys.argv[1], Path(sys.argv[2])
    if mode == "seed":
        seed(workdir)
    else:
        sys.exit(run(workdir, sys.argv[3] if len(sys.argv) > 3 else "http://localhost:5193"))
