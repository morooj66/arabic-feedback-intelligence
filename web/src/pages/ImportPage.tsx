import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { api } from "../api/client";
import type { ApiError, ExcelJob, ExcelSession } from "../api/types";

type Stage = "loading" | "empty" | "uploading" | "starting" | "processing" | "ready" | "completed";

const UNSUPPORTED_TYPE_MESSAGE = "نوع الملف غير مدعوم — الملفات المدعومة حاليًا: .xlsx فقط";
const POLL_INTERVAL_MS = 1500;

const STAGE_ORDER = ["رفع الملف", "تهيئة البيانات", "تصنيف التعليقات", "إنشاء الردود المقترحة", "جاهز للمراجعة"];

export function ImportPage(props: { onOpenReview: (sourceFile: string) => void; onProcessed?: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [stage, setStage] = useState<Stage>("loading");
  const [fileName, setFileName] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadDone, setUploadDone] = useState(false);
  const [session, setSession] = useState<ExcelSession | null>(null);
  const [job, setJob] = useState<ExcelJob | null>(null);
  const [error, setError] = useState("");
  const [downloading, setDownloading] = useState(false);

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  useEffect(() => stopPolling, []);

  useEffect(() => {
    let cancelled = false;
    api
      .activeExcelSession()
      .then((res) => {
        if (cancelled) return;
        if (!res.active) {
          setStage("empty");
          return;
        }
        setSession(res.active);
        setFileName(res.active.original_filename);
        if (res.active.status === "completed") {
          setStage("completed");
        } else if (res.active.status === "ready") {
          setStage("ready");
        } else if (res.active.latest_job_id) {
          setStage("processing");
          pollJob(res.active.latest_job_id);
        } else {
          setStage("empty");
        }
      })
      .catch(() => !cancelled && setStage("empty"));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function pollJob(jobId: string) {
    stopPolling();
    const tick = async () => {
      try {
        const current = await api.getExcelJob(jobId);
        setJob(current);
        if (current.status === "succeeded") {
          stopPolling();
          const refreshed = await api.getExcelSession(current.session_id);
          setSession(refreshed);
          setStage(refreshed.status === "completed" ? "completed" : "ready");
          props.onProcessed?.();
        } else if (current.status === "failed") {
          stopPolling();
          setError(current.error_message ? `تعذّرت المعالجة: ${current.error_message}` : "تعذّرت معالجة الملف.");
          // Stage deliberately stays "processing" — the stage checklist
          // freezes at wherever it stopped, with the error + retry action
          // shown alongside it (see the render section below).
        }
      } catch {
        // transient poll failure — try again on the next tick rather than surfacing an error immediately
      }
    };
    void tick();
    pollRef.current = setInterval(tick, POLL_INTERVAL_MS);
  }

  function pickFile() {
    inputRef.current?.click();
  }

  function resetToEmpty() {
    stopPolling();
    setSession(null);
    setJob(null);
    setFileName("");
    setUploadProgress(0);
    setUploadDone(false);
    setError("");
    setStage("empty");
  }

  async function onFileSelected(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    setError("");
    setFileName(file.name);
    setUploadProgress(0);
    setUploadDone(false);

    if (!file.name.toLowerCase().endsWith(".xlsx")) {
      setError(UNSUPPORTED_TYPE_MESSAGE);
      setStage("empty");
      return;
    }

    setStage("uploading");
    try {
      const uploaded = await api.uploadExcelWithProgress(file, setUploadProgress);
      if (!uploaded.valid) {
        setError(uploaded.errors.join("، ") || "الملف غير صالح للمعالجة.");
        setStage("empty");
        return;
      }
      setUploadDone(true);
      setStage("starting");
      const created = await api.createExcelSession(uploaded.upload_token);
      setSession(created.session);
      setStage("processing");
      pollJob(created.job_id);
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr.code === "unsupported_file_type") {
        setError(UNSUPPORTED_TYPE_MESSAGE);
      } else if (apiErr.code === "file_too_large") {
        setError("حجم الملف يتجاوز الحد المسموح به");
      } else if (apiErr.code === "empty_upload" || apiErr.code === "empty_body") {
        setError("الملف المرفوع فارغ");
      } else if (apiErr.code === "invalid_workbook") {
        setError(apiErr.message || "الملف لا يحتوي على البيانات أو الأعمدة المتوقعة.");
      } else {
        setError(apiErr.message ? `تعذّر رفع الملف: ${apiErr.message}` : "تعذّر رفع الملف. حاول مرة أخرى.");
      }
      setStage("empty");
    }
  }

  async function retryProcessing() {
    if (!session) return;
    setError("");
    try {
      const resumed = await api.resumeExcelSession(session.session_id);
      pollJob(resumed.job_id);
    } catch (err) {
      const apiErr = err as ApiError;
      setError(apiErr.message ? `تعذّر استئناف المعالجة: ${apiErr.message}` : "تعذّر استئناف المعالجة.");
    }
  }

  async function downloadResults() {
    if (!session) return;
    setDownloading(true);
    setError("");
    try {
      await api.downloadExcelExport(session.source_file);
    } catch (err) {
      const apiErr = err as ApiError;
      setError(apiErr.status === 409 ? "لا توجد نتائج مراجعة جاهزة للتصدير" : "تعذّر تنزيل ملف النتائج.");
    } finally {
      setDownloading(false);
    }
  }

  const stageIndex = job?.current_stage ? STAGE_ORDER.indexOf(job.current_stage) : -1;
  const pct = job?.rows_total ? Math.min(100, Math.round((job.rows_done / job.rows_total) * 100)) : null;

  return (
    <div className="settings-shell" dir="rtl">
      <header className="settings-head">
        <p>
          استورد ملف تعليقات الزوار لتهيئته للمراجعة. سيقوم النظام بمعالجة التعليقات وتصنيفها وإنشاء الردود
          المقترحة تلقائيًا.
        </p>
      </header>

      {error ? <p className="login-error import-error">{error}</p> : null}

      {stage === "loading" ? <section className="settings-card">جاري التحقق من وجود ملف نشط…</section> : null}

      {stage === "empty" ? (
        <section className="settings-card">
          <p className="settings-kicker">استيراد ملف Excel</p>
          <h2>استيراد ملف Excel</h2>
          <input
            ref={inputRef}
            type="file"
            accept=".xlsx"
            hidden
            data-testid="excel-file-input"
            onChange={onFileSelected}
          />
          <div className="import-file-row">
            <button type="button" className="btn-primary" onClick={pickFile}>
              استيراد ملف Excel
            </button>
          </div>
        </section>
      ) : null}

      {stage === "uploading" || stage === "starting" ? (
        <section className="settings-card">
          <p className="settings-kicker">الملف</p>
          <h2>{fileName}</h2>
          {!uploadDone ? (
            <>
              <p className="settings-help">جاري رفع الملف</p>
              <div className="import-progress-track">
                <div className="import-progress-fill" style={{ width: `${Math.round(uploadProgress * 100)}%` }} />
              </div>
            </>
          ) : (
            <p className="settings-help">✓ تم رفع الملف بنجاح</p>
          )}
        </section>
      ) : null}

      {stage === "processing" ? (
        <section className="settings-card">
          <p className="settings-kicker">الملف</p>
          <h2>{fileName}</h2>
          <ul className="import-stage-list" aria-label="مراحل المعالجة">
            {STAGE_ORDER.map((label, i) => {
              const state = i < stageIndex ? "done" : i === stageIndex ? "current" : "pending";
              return (
                <li key={label} data-state={state}>
                  <span className="import-stage-marker">{state === "done" ? "✓" : state === "current" ? "●" : "○"}</span>
                  <span>{label}</span>
                </li>
              );
            })}
          </ul>
          {job?.rows_total ? (
            <>
              <div className="import-progress-track">
                <div className="import-progress-fill" style={{ width: `${pct}%` }} />
              </div>
              <p className="settings-help">
                تمت معالجة {job.rows_done} من {job.rows_total} ({pct}%)
              </p>
            </>
          ) : null}
        </section>
      ) : null}

      {(stage === "ready" || stage === "completed") && session ? (
        <section className="settings-card">
          {stage === "completed" ? (
            <p className="settings-kicker">✓ اكتملت مراجعة الملف</p>
          ) : session.counts.completed > 0 ? (
            <p className="settings-kicker">الملف الحالي</p>
          ) : (
            <p className="settings-kicker">✓ الملف جاهز للمراجعة</p>
          )}
          <h2>{session.original_filename}</h2>
          <div className="dash-kpis dash-kpis--home import-summary-grid" aria-label="ملخص الملف">
            <article className="dash-kpi">
              <span>إجمالي التعليقات</span>
              <strong>{session.counts.canonical_total}</strong>
            </article>
            <article className="dash-kpi">
              <span>المستبعد</span>
              <strong>{session.counts.excluded}</strong>
            </article>
            <article className="dash-kpi">
              <span>تمت مراجعته</span>
              <strong>{session.counts.completed}</strong>
            </article>
            <article className="dash-kpi">
              <span>المتبقي</span>
              <strong>{session.counts.remaining}</strong>
            </article>
          </div>
          <div className="import-actions">
            {stage === "ready" ? (
              <button type="button" className="btn-primary" onClick={() => props.onOpenReview(session.source_file)}>
                {session.counts.completed > 0 ? "متابعة المراجعة" : "ابدأ المراجعة"}
              </button>
            ) : null}
            <button
              type="button"
              className="btn-quiet"
              onClick={downloadResults}
              disabled={downloading || session.counts.completed === 0}
            >
              {downloading ? "جاري التنزيل…" : "تنزيل ملف الردود"}
            </button>
            <button type="button" className="btn-ghost" onClick={resetToEmpty}>
              استيراد ملف جديد
            </button>
          </div>
        </section>
      ) : null}

      {error && session && stage === "processing" ? (
        <section className="settings-card">
          <button type="button" className="btn-quiet" onClick={retryProcessing}>
            إعادة المحاولة
          </button>
        </section>
      ) : null}
    </div>
  );
}
