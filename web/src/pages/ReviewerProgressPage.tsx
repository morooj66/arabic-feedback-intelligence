import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ReviewerSummaryRow } from "../api/types";

function pct(row: ReviewerSummaryRow): number {
  return row.assigned_total > 0 ? Math.round((row.completed / row.assigned_total) * 100) : 0;
}

export function ReviewerProgressPage() {
  const [reviewers, setReviewers] = useState<ReviewerSummaryRow[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .adminReviewers()
      .then((res) => {
        if (!cancelled) setReviewers(res.reviewers);
      })
      .catch(() => {
        if (!cancelled) setError("تعذر تحميل بيانات المراجعين.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="reports-shell" dir="rtl">
      <header className="reports-head">
        <p>متابعة توزيع الإسناد والتقدم لكل مراجع — للاطلاع فقط، ولا يشمل إنشاء أو تعطيل الحسابات.</p>
      </header>

      {error ? <p className="reports-note">{error}</p> : null}

      <section className="reports-panel">
        <h2>المراجعون</h2>
        {reviewers === null && !error ? (
          <p className="reports-empty">جاري التحميل…</p>
        ) : reviewers && reviewers.length === 0 ? (
          <p className="reports-empty">لا يوجد مراجعون نشطون حاليًا.</p>
        ) : reviewers ? (
          <div className="data-table-wrap">
            <table className="data-table" aria-label="تقدم المراجعين">
              <thead>
                <tr>
                  <th>المراجع</th>
                  <th>البريد الإلكتروني</th>
                  <th>إجمالي المسند</th>
                  <th>قيد الإنجاز</th>
                  <th>مكتمل</th>
                  <th>نسبة الإنجاز</th>
                </tr>
              </thead>
              <tbody>
                {reviewers.map((r) => (
                  <tr key={r.user_id}>
                    <td>{r.display_name}</td>
                    <td>{r.email}</td>
                    <td>{r.assigned_total}</td>
                    <td>{r.pending}</td>
                    <td>{r.completed}</td>
                    <td>
                      <div className="progress-cell">
                        <div className="progress-track">
                          <div className="progress-fill" style={{ width: `${pct(r)}%` }} />
                        </div>
                        <span>{pct(r)}٪</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>
    </div>
  );
}
