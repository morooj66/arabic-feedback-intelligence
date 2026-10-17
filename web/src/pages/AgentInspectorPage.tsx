import { useEffect, useState } from "react";
import { api } from "../api/client";

type JsonRecord = Record<string, unknown>;

function PassFail(props: { pass: boolean | null | undefined; labelPass?: string; labelFail?: string }) {
  if (props.pass === null || props.pass === undefined) {
    return <span className="badge badge-pending">—</span>;
  }
  return props.pass ? (
    <span className="badge badge-ok">{props.labelPass || "PASS"}</span>
  ) : (
    <span className="badge badge-fail">{props.labelFail || "FAIL"}</span>
  );
}

function JsonBlock(props: { data: unknown }) {
  return (
    <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: 12, background: "var(--panel-2, #f4f4f5)", padding: 8, borderRadius: 6 }}>
      {JSON.stringify(props.data, null, 2)}
    </pre>
  );
}

function Panel(props: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ border: "1px solid var(--border, #ddd)", borderRadius: 8, padding: 12, marginBottom: 12 }}>
      <h3 style={{ marginTop: 0 }}>{props.title}</h3>
      {props.children}
    </section>
  );
}

export function AgentInspectorPage() {
  const [architecture, setArchitecture] = useState<JsonRecord | null>(null);
  const [config, setConfig] = useState<JsonRecord | null>(null);
  const [commentText, setCommentText] = useState("شكرًا جزيلاً على هذا العمل الرائع في المعجم.");
  const [primaryIntent, setPrimaryIntent] = useState("");
  const [topicSubtype, setTopicSubtype] = useState("");
  const [sentiment, setSentiment] = useState("");
  const [draftText, setDraftText] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);

  const [classificationResult, setClassificationResult] = useState<JsonRecord | null>(null);
  const [eligibilityResult, setEligibilityResult] = useState<JsonRecord | null>(null);
  const [retrievalResult, setRetrievalResult] = useState<JsonRecord | null>(null);
  const [groundingResult, setGroundingResult] = useState<JsonRecord | null>(null);
  const [validationResult, setValidationResult] = useState<JsonRecord | null>(null);
  const [fullRunResult, setFullRunResult] = useState<JsonRecord | null>(null);

  useEffect(() => {
    api.inspectorArchitecture().then(setArchitecture).catch(() => {});
    api.inspectorConfig().then(setConfig).catch(() => {});
  }, []);

  async function run<T>(name: string, fn: () => Promise<T>, onDone: (r: T) => void) {
    setBusy(name);
    setError("");
    try {
      const result = await fn();
      onDone(result);
    } catch (e: any) {
      setError(e?.message || "request_failed");
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="workspace workspace--wide">
      <h2>فاحص الوكيل (Agent Inspector) — أداة تطوير/إدارة فقط</h2>
      <p style={{ opacity: 0.8 }}>
        وضع اختبار آمن (dry-run): لا يكتب أبدًا على review_items، ولا يُنشئ إسناد مراجع، ولا يُرسل ردًا فعليًا. كل تنفيذ كامل
        (Full Agent E2E Test) يستدعي Gemini الحقيقي (استدعاء فعلي له تكلفة) لكنه لا يمس بيانات الإنتاج إطلاقًا؛ سجل التتبع
        الخاص به معزول في الذاكرة فقط ولا يختلط بسجلات M5E الحقيقية.
      </p>

      <Panel title="بنية الوكيل (Agent Architecture)">
        {architecture ? <JsonBlock data={architecture} /> : <span>...جارٍ التحميل</span>}
      </Panel>

      <Panel title="إعدادات الوكيل (Agent Configuration)">
        {config ? <JsonBlock data={config} /> : <span>...جارٍ التحميل</span>}
      </Panel>

      <Panel title="مُدخل الاختبار (Test Input)">
        <label>
          نص التعليق (Comment text)
          <textarea value={commentText} onChange={(e) => setCommentText(e.target.value)} rows={3} style={{ width: "100%" }} />
        </label>
        <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
          <input placeholder="primary_intent (اختياري)" value={primaryIntent} onChange={(e) => setPrimaryIntent(e.target.value)} />
          <input placeholder="topic_subtype (اختياري)" value={topicSubtype} onChange={(e) => setTopicSubtype(e.target.value)} />
          <input placeholder="sentiment (اختياري)" value={sentiment} onChange={(e) => setSentiment(e.target.value)} />
        </div>
        {error ? <div className="empty" style={{ marginTop: 8 }}>{error}</div> : null}
      </Panel>

      <Panel title="اختبار التصنيف (Classification Test)">
        <button
          disabled={busy !== null}
          onClick={() => run("classification", () => api.inspectorClassification(commentText), setClassificationResult)}
        >
          {busy === "classification" ? "...جارٍ التشغيل" : "تشغيل"}
        </button>
        {classificationResult ? <JsonBlock data={classificationResult} /> : null}
      </Panel>

      <Panel title="اختبار الأهلية والتوجيه (Eligibility / Routing Test)">
        <button
          disabled={busy !== null}
          onClick={() =>
            run(
              "eligibility",
              () =>
                api.inspectorEligibility({
                  comment_text: commentText,
                  primary_intent: primaryIntent || null,
                  topic_subtype: topicSubtype || null,
                  sentiment: sentiment || null,
                }),
              setEligibilityResult
            )
          }
        >
          {busy === "eligibility" ? "...جارٍ التشغيل" : "تشغيل"}
        </button>
        {eligibilityResult ? (
          <>
            <div>
              routing_decision: <strong>{String(eligibilityResult.routing_decision)}</strong> | rule_id:{" "}
              {String(eligibilityResult.rule_id)}
            </div>
            <JsonBlock data={eligibilityResult} />
          </>
        ) : null}
      </Panel>

      <Panel title="اختبار الاسترجاع (Retrieval Test)">
        <button
          disabled={busy !== null}
          onClick={() =>
            run(
              "retrieval",
              () =>
                api.inspectorRetrieval({
                  comment_text: commentText,
                  primary_intent: primaryIntent || null,
                  topic_subtype: topicSubtype || null,
                }),
              setRetrievalResult
            )
          }
        >
          {busy === "retrieval" ? "...جارٍ التشغيل" : "تشغيل"}
        </button>
        {retrievalResult ? (
          <>
            <div>Official Knowledge: {String(retrievalResult.official_outcome)} (عدد: {String(retrievalResult.official_result_count)})</div>
            <div>Approved Response Memory — STYLE ONLY: {String(retrievalResult.approved_outcome)} (عدد: {String(retrievalResult.approved_result_count)})</div>
            <JsonBlock data={retrievalResult} />
          </>
        ) : null}
      </Panel>

      <Panel title="اختبار التأصيل (Grounding Test)">
        <button
          disabled={busy !== null}
          onClick={() =>
            run(
              "grounding",
              () => api.inspectorGrounding({ identifying_context: null, official_knowledge_available: false }),
              setGroundingResult
            )
          }
        >
          تشغيل (بدون معرفة رسمية)
        </button>{" "}
        <button
          disabled={busy !== null}
          onClick={() =>
            run(
              "grounding",
              () =>
                api.inspectorGrounding({
                  identifying_context: null,
                  official_knowledge_available: true,
                  official_content: "نص معرفة رسمية تجريبي للاختبار فقط.",
                }),
              setGroundingResult
            )
          }
        >
          تشغيل (مع معرفة رسمية تجريبية)
        </button>
        {groundingResult ? <JsonBlock data={groundingResult} /> : null}
      </Panel>

      <Panel title="اختبار التحقق (Validation Test) — أدخل نص مسودة يدويًا">
        <textarea value={draftText} onChange={(e) => setDraftText(e.target.value)} rows={2} style={{ width: "100%" }} placeholder="نص المسودة للاختبار" />
        <div>
          <button disabled={busy !== null} onClick={() => run("validation", () => api.inspectorValidation({ draft_text: draftText }), setValidationResult)}>
            {busy === "validation" ? "...جارٍ التشغيل" : "تشغيل"}
          </button>
        </div>
        {validationResult ? (
          <>
            <div>
              النتيجة: <PassFail pass={validationResult.passed as boolean} />
            </div>
            <JsonBlock data={validationResult} />
          </>
        ) : null}
      </Panel>

      <Panel title="اختبار الوكيل الكامل (Full Agent E2E Test) — يستدعي Gemini الحقيقي">
        <button disabled={busy !== null} onClick={() => run("full-run", () => api.inspectorFullRun({
          comment_text: commentText,
          primary_intent: primaryIntent || null,
          topic_subtype: topicSubtype || null,
          sentiment: sentiment || null,
        }), setFullRunResult)}>
          {busy === "full-run" ? "...جارٍ التشغيل" : "تشغيل الاختبار الكامل"}
        </button>
        {fullRunResult ? (
          <>
            {(() => {
              const stages = (fullRunResult.stages || {}) as JsonRecord;
              const meta = (fullRunResult.metadata || {}) as JsonRecord;
              return (
                <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
                  <div>Grounding mode: <strong>{String(meta.grounding_mode)}</strong></div>
                  <div>Draft 1: {String(stages.draft_1 ?? "—")}</div>
                  <div>
                    Draft 1 validation: <PassFail pass={meta.initial_validation_passed as boolean} /> {meta.initial_failure_code ? `(${meta.initial_failure_code})` : ""}
                  </div>
                  <div>Refinement attempted: {meta.refinement_attempted ? "نعم" : "لا"}</div>
                  {meta.refinement_attempted ? (
                    <>
                      <div>Draft 2: {String(stages.draft_2 ?? "—")}</div>
                      <div>
                        Draft 2 validation: <PassFail pass={meta.final_validation_passed as boolean} /> {meta.final_failure_code ? `(${meta.final_failure_code})` : ""}
                      </div>
                    </>
                  ) : null}
                  <div>logical_generation_count: {String(meta.logical_generation_count)} (الحد الأقصى المسموح: 2)</div>
                  <div>generation_status: <strong>{String(meta.generation_status)}</strong></div>
                </div>
              );
            })()}
            <JsonBlock data={fullRunResult} />
          </>
        ) : null}
      </Panel>
    </div>
  );
}
