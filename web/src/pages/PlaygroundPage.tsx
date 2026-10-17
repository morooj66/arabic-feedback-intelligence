import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";

// Internal AI Evaluation Playground (admin-only, disabled by default on the
// server). Two workflows:
//   Explore  - run custom comments and inspect predictions; nothing is scored.
//   Evaluate - enter expected labels and explicitly VERIFY them; only verified
//              cases are ever scored or counted toward metrics.
// Nothing is saved unless the "save" button is pressed.

type Rec = Record<string, any>;
type Workflow = "explore" | "evaluate" | "results";

const SAMPLE_VARIANTS = [
  "التطبيق ممتاز",
  "التطبيق ممتاز 😊",
  "التطبيق ممتاز 😡",
  "التطبيق مو ممتاز",
  "التطبيق ممتاز لكن البحث بطيء",
  "Good application",
];

const EXPECTED_FIELDS: { key: string; label: string; taxonomyKey: string; agentOnly?: boolean }[] = [
  { key: "intent", label: "النية (Intent)", taxonomyKey: "intents" },
  { key: "topic", label: "الموضوع (Topic)", taxonomyKey: "topics" },
  { key: "sentiment", label: "المشاعر (Sentiment)", taxonomyKey: "sentiments" },
  { key: "routing", label: "التوجيه (Routing)", taxonomyKey: "routing_decisions" },
  { key: "response_decision", label: "قرار الوكيل (Agent decision)", taxonomyKey: "response_decisions", agentOnly: true },
];

const VERDICT_STYLE: Record<string, string> = {
  PASS: "badge badge-ok",
  FAIL: "badge badge-fail",
  NOT_EVALUATED: "badge badge-pending",
  NEEDS_HUMAN_REVIEW: "badge badge-pending",
  ERROR: "badge badge-fail",
};

interface VariantDraft {
  text: string;
  status: "UNVERIFIED" | "VERIFIED" | "NEEDS_HUMAN_REVIEW";
  expected: Record<string, string>;
  notes: string;
}

const emptyDraft = (text = ""): VariantDraft => ({ text, status: "UNVERIFIED", expected: {}, notes: "" });

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
const num = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? "—" : v.toFixed(d));

function Panel(props: { title: string; children: React.ReactNode; tone?: "warn" }) {
  return (
    <section
      style={{
        border: `1px solid ${props.tone === "warn" ? "#c0841a" : "var(--border, #ddd)"}`,
        background: props.tone === "warn" ? "rgba(192,132,26,.08)" : undefined,
        borderRadius: 8,
        padding: 12,
        marginBottom: 12,
      }}
    >
      <h3 style={{ marginTop: 0 }}>{props.title}</h3>
      {props.children}
    </section>
  );
}

function ModeBadge(props: { mode: string | null | undefined }) {
  if (!props.mode) return null;
  const mock = props.mode === "mock";
  return (
    <span
      className="badge"
      style={{ background: mock ? "#5b6b7a" : "#b3261e", color: "#fff", padding: "2px 8px", borderRadius: 6, fontWeight: 700 }}
    >
      {mock ? "MOCK" : "LIVE"}
    </span>
  );
}

function Dim(props: { children: React.ReactNode }) {
  return <span style={{ opacity: 0.7, fontSize: 12 }}>{props.children}</span>;
}

function PredictionRow(props: { label: string; item: Rec | null | undefined; error?: boolean }) {
  return (
    <tr>
      <td>{props.label}</td>
      <td>
        {props.item ? (
          <>
            <strong dir="ltr">{props.item.label}</strong>
            <br />
            <Dim>الثقة (confidence, top class): {pct(props.item.confidence)}</Dim>
          </>
        ) : (
          <Dim>{props.error ? "فشل — انظر الأخطاء" : "—"}</Dim>
        )}
      </td>
    </tr>
  );
}

function AgentRunView(props: { run: Rec; title: string }) {
  const { run } = props;
  return (
    <div style={{ border: "1px dashed var(--border, #bbb)", borderRadius: 6, padding: 8, marginTop: 8 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <strong>{props.title}</strong>
        <ModeBadge mode={run.agent_mode} />
        <span
          className="badge"
          data-testid={`path-${run.label_source}`}
          style={{ background: run.production_equivalent ? "#2e6b3a" : "#8a5a00", color: "#fff", padding: "2px 8px", borderRadius: 6 }}
        >
          {run.production_equivalent ? "مكافئ للإنتاج (production-equivalent)" : "اختبار عزل — ليس مسار الإنتاج"}
        </span>
        <Dim>التسميات المستخدمة: {run.label_source === "verified" ? "موثّقة يدويًا (verified)" : "تنبؤات المصنِّف (predicted)"}</Dim>
      </div>
      <Dim>{run.mode_banner}</Dim>
      {run.decision ? (
        <ul style={{ margin: "6px 0", paddingInlineStart: 18 }}>
          <li dir="ltr">routing_decision: <strong>{run.decision.routing_decision}</strong></li>
          <li dir="ltr">generation_status: {run.decision.generation_status}</li>
          <li dir="ltr">reason_code: {run.decision.reason_code}</li>
          <li dir="ltr">eligibility_rule_id: {run.decision.eligibility_rule_id}</li>
          <li dir="ltr">classification_routing: {run.classification_routing_decision ?? "—"}</li>
        </ul>
      ) : null}
      {run.tool_calls?.length ? (
        <details>
          <summary>استدعاءات الأدوات (Tool calls) — {run.tool_calls.length}</summary>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>{JSON.stringify(run.tool_calls, null, 2)}</pre>
        </details>
      ) : null}
      {run.drafts && (run.drafts.draft_1 || run.drafts.draft_2) ? (
        <details>
          <summary>المسودات (Drafts)</summary>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>{JSON.stringify(run.drafts, null, 2)}</pre>
        </details>
      ) : null}
      <div style={{ marginTop: 6 }}>
        <strong>الرد المولَّد:</strong>{" "}
        {run.response ? (
          <>
            <span dir="auto" style={{ display: "block", padding: 6, background: "var(--panel-2, #f4f4f5)", borderRadius: 4 }}>{run.response}</span>
            {run.agent_mode === "mock" ? <Dim>نص تجريبي ثابت — ليس مخرجات Gemini ولا يدل على جودة الرد.</Dim> : null}
          </>
        ) : (
          <Dim>لا يوجد رد</Dim>
        )}
      </div>
      <Dim>
        زمن التنفيذ: {run.execution_time_ms ?? "—"} ms · طلبات النموذج اللغوي الفعلية: {run.llm_calls}
      </Dim>
      {run.fallback ? (
        <div role="alert" style={{ color: "#b3261e", marginTop: 4 }}>
          بديل/فشل مغلق (fallback): {run.fallback.kind} — {run.fallback.detail}
        </div>
      ) : null}
      {run.error ? (
        <div role="alert" style={{ color: "#b3261e" }} dir="ltr">
          {run.error.stage}: {run.error.type} — {run.error.message}
        </div>
      ) : null}
    </div>
  );
}

function VariantResultCard(props: { variant: Rec; draft: VariantDraft; verdict: Rec | null; showExpected: boolean }) {
  const { variant: v, draft, verdict } = props;
  const cls = v.classification;
  const errs: Rec[] = v.errors || [];
  return (
    <article style={{ border: "1px solid var(--border, #ddd)", borderRadius: 8, padding: 10, marginBottom: 10 }} data-testid={`variant-result-${v.index}`}>
      <header style={{ display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
        <strong dir="auto">#{v.index + 1} — {draft.text}</strong>
        {verdict ? <span className={VERDICT_STYLE[verdict.verdict] || "badge"} data-testid={`verdict-${v.index}`}>{verdict.verdict}</span> : null}
      </header>
      <Dim>
        {v.preprocessing.changed ? "تم تطبيع النص (normalize_text). " : ""}
        {v.preprocessing.prefilter_reason ? `المرشّح المسبق: ${v.preprocessing.prefilter_reason} — ${v.preprocessing.production_outcome}` : v.preprocessing.production_outcome}
      </Dim>
      <table style={{ width: "100%", marginTop: 6 }}>
        <tbody>
          <PredictionRow label="النية (Intent)" item={cls?.intent} error={!!cls} />
          <PredictionRow label="الموضوع (Topic)" item={cls?.topic} error={!!cls} />
          <PredictionRow label="المشاعر (Sentiment)" item={cls?.sentiment} error={!!cls} />
          <tr>
            <td>التوجيه (Routing)</td>
            <td dir="ltr">{v.routing ? <>{v.routing.decision}{v.routing.reason_code ? ` (${v.routing.reason_code})` : ""}</> : <Dim>—</Dim>}</td>
          </tr>
          <tr>
            <td>الأهلية — معاينة قبل الاسترجاع</td>
            <td dir="ltr">{v.eligibility_preview ? <>{v.eligibility_preview.routing_decision} · {v.eligibility_preview.rule_id}</> : <Dim>—</Dim>}</td>
          </tr>
        </tbody>
      </table>
      {props.showExpected && verdict && verdict.fields ? (
        <table style={{ width: "100%", marginTop: 8, borderTop: "1px solid var(--border, #ddd)" }} aria-label="expected-vs-actual">
          <thead>
            <tr><th>الحقل</th><th>المتوقع (يدوي)</th><th>الفعلي (النموذج)</th><th></th></tr>
          </thead>
          <tbody>
            {Object.entries(verdict.fields as Rec).map(([k, f]: [string, Rec]) => (
              <tr key={k}>
                <td>{k}</td>
                <td dir="ltr" style={{ background: "rgba(40,90,200,.10)" }}>{f.expected}</td>
                <td dir="ltr" style={{ background: "rgba(120,120,120,.12)" }}>{f.actual ?? "—"}</td>
                <td>{f.evaluated ? (f.match ? "✓" : "✗") : "؟"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {v.pipeline_status && !v.pipeline_status.classification_complete ? (
        <div role="alert" data-testid="partial-classification" style={{ border: "1px solid #b3261e", borderRadius: 6, padding: 6, marginTop: 6, color: "#b3261e" }}>
          <strong>تصنيف ناقص (partial classification):</strong> فشلت المراحل: <span dir="ltr">{v.pipeline_status.failed_stages.join(", ")}</span>. نتائج المصنِّفات الناجحة معروضة أعلاه،
          لكنها ليست مكافئة للإنتاج.
        </div>
      ) : null}
      {v.agent_skipped ? (
        <div role="alert" data-testid="agent-skipped" style={{ border: "1px solid #b3261e", borderRadius: 6, padding: 6, marginTop: 6 }}>
          <strong>لم يُشغَّل الوكيل (Agent not run):</strong> <span dir="ltr">{v.agent_skipped.detail}</span>
        </div>
      ) : null}
      {Object.entries(v.agent_runs || {}).map(([src, run]) => (
        <AgentRunView key={src} run={run as Rec} title={src === "verified" ? "الوكيل بتسميات موثّقة (B)" : "الوكيل بتنبؤات المصنِّف (A)"} />
      ))}
      {v.fixed_label_comparison ? (
        <div style={{ marginTop: 6 }}>
          <strong>مقارنة A / B:</strong>{" "}
          {v.fixed_label_comparison.decision_differs ? "القرار يختلف (خطأ المصنِّف يغيّر سلوك الوكيل)" : "القرار نفسه"} ·{" "}
          {v.fixed_label_comparison.rule_differs ? "القاعدة تختلف" : "القاعدة نفسها"}
          <br />
          <Dim>{v.fixed_label_comparison.note}</Dim>
        </div>
      ) : null}
      <Dim>
        الزمن (ms): {JSON.stringify(v.timings_ms)}
      </Dim>
      {errs.length ? (
        <ul role="alert" style={{ color: "#b3261e", margin: "6px 0", paddingInlineStart: 18 }} dir="ltr">
          {errs.map((e, i) => (
            <li key={i}>{e.stage}: {e.type} — {e.message}</li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}

function ComparisonTable(props: { run: Rec; drafts: VariantDraft[] }) {
  const agentKey = (v: Rec) => v.agent_runs?.predicted ?? v.agent_runs?.verified;
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", fontSize: 13 }} aria-label="variant-comparison">
        <thead>
          <tr>
            <th>#</th><th>النص</th><th>Intent</th><th>Topic</th><th>Sentiment</th><th>Routing</th><th>Agent</th>
          </tr>
        </thead>
        <tbody>
          {props.run.variants.map((v: Rec) => {
            const c = v.classification;
            const cell = (it: Rec | null | undefined) => (it ? `${it.label} (${pct(it.confidence)})` : "—");
            const a = agentKey(v);
            return (
              <tr key={v.index}>
                <td>{v.index + 1}</td>
                <td dir="auto">{props.drafts[v.index]?.text}</td>
                <td dir="ltr">{cell(c?.intent)}</td>
                <td dir="ltr">{cell(c?.topic)}</td>
                <td dir="ltr">{cell(c?.sentiment)}</td>
                <td dir="ltr">{v.routing?.decision ?? "—"}</td>
                <td dir="ltr">{a?.decision ? `${a.decision.routing_decision} [${a.agent_mode}]` : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ConfusionMatrix(props: { cm: Rec }) {
  const { labels, matrix } = props.cm;
  const used = labels.map((_: string, i: number) => matrix[i].some((x: number) => x > 0) || matrix.some((r: number[]) => r[i] > 0));
  return (
    <div style={{ overflowX: "auto" }}>
      <Dim>الصفوف: المتوقع الموثّق · الأعمدة: تنبؤ النموذج</Dim>
      <table style={{ fontSize: 12 }}>
        <thead>
          <tr><th></th>{labels.map((l: string, i: number) => (used[i] ? <th key={l} dir="ltr">{l}</th> : null))}</tr>
        </thead>
        <tbody>
          {labels.map((l: string, r: number) =>
            used[r] ? (
              <tr key={l}>
                <th dir="ltr">{l}</th>
                {labels.map((c: string, ci: number) => (used[ci] ? <td key={c} style={{ textAlign: "center", fontWeight: r === ci ? 700 : 400 }}>{matrix[r][ci]}</td> : null))}
              </tr>
            ) : null,
          )}
        </tbody>
      </table>
    </div>
  );
}

function MetricsView(props: { metrics: Rec }) {
  const m = props.metrics;
  return (
    <div>
      <p data-testid="case-counts">
        الحالات: {m.case_counts.total} · موثّقة (Verified): <strong>{m.case_counts.verified}</strong> · غير موثّقة: {m.case_counts.unverified} · تحتاج مراجعة بشرية: {m.case_counts.needs_human_review}
      </p>
      {m.notes.map((n: string) => (
        <Dim key={n}>{n}<br /></Dim>
      ))}
      {Object.entries(m.classification as Rec).map(([task, t]: [string, Rec]) => (
        <Panel key={task} title={`${task} — n=${t.n} (الأصناف المدعومة: ${t.supported_classes})`}>
          {t.sufficient ? (
            <p data-testid={`metrics-${task}`}>
              Accuracy {pct(t.accuracy)} · Macro-F1 (supported classes) {num(t.macro_f1_supported_classes)} · Weighted-F1 {num(t.weighted_f1)}
            </p>
          ) : (
            <p role="status" data-testid={`withheld-${task}`}>الدرجات المجمّعة محجوبة: {t.withheld_reason}</p>
          )}
          <table style={{ fontSize: 12 }}>
            <thead><tr><th>Class</th><th>Support</th><th>Predicted</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
            <tbody>
              {t.per_class.filter((c: Rec) => c.support > 0 || c.predicted > 0).map((c: Rec) => (
                <tr key={c.label}>
                  <td dir="ltr">{c.label}{c.low_support ? " ⚠ low support" : ""}</td>
                  <td>{c.support}</td><td>{c.predicted}</td>
                  <td>{num(c.precision)}</td><td>{num(c.recall)}</td><td>{num(c.f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {t.n > 0 ? <ConfusionMatrix cm={t.confusion_matrix} /> : null}
          <Dim>{t.caveat}</Dim>
        </Panel>
      ))}
      <Panel title="التوجيه (Routing) وقرارات الوكيل">
        <p>التوجيه: n={m.routing.n} · صحيح={m.routing.correct} {m.routing.withheld_reason ? `— ${m.routing.withheld_reason}` : `· ${pct(m.routing.accuracy)}`}</p>
        {(["mock", "live"] as const).map((mode) => (
          <p key={mode} data-testid={`agent-${mode}`}>
            <ModeBadge mode={mode} /> قرار الوكيل: n={m.agent[mode].n} · متفق={m.agent[mode].correct} <Dim>{m.agent[mode].note}</Dim>
          </p>
        ))}
      </Panel>
    </div>
  );
}

export function PlaygroundPage() {
  const [status, setStatus] = useState<Rec | null>(null);
  const [statusError, setStatusError] = useState("");
  const [workflow, setWorkflow] = useState<Workflow>("explore");
  const [drafts, setDrafts] = useState<VariantDraft[]>([emptyDraft("")]);
  const [mode, setMode] = useState<"classification_only" | "full_pipeline">("classification_only");
  const [agentMode, setAgentMode] = useState<"mock" | "live">("mock");
  const [liveConfirmed, setLiveConfirmed] = useState(false);
  const [fixedLabels, setFixedLabels] = useState(false);
  const [forceClassify, setForceClassify] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [run, setRun] = useState<Rec | null>(null);
  const [stale, setStale] = useState(false);
  const [verdicts, setVerdicts] = useState<Record<number, Rec>>({});
  const [savedIds, setSavedIds] = useState<Record<number, string>>({});
  const [metrics, setMetrics] = useState<Rec | null>(null);
  const [cases, setCases] = useState<Rec[]>([]);

  useEffect(() => {
    api.playgroundStatus().then(setStatus).catch((e: any) => setStatusError(e?.message || "request_failed"));
  }, []);

  const maxVariants: number = status?.limits?.max_variants ?? 10;
  const maxChars: number = status?.limits?.max_chars ?? 1000;
  const taxonomy: Rec = status?.taxonomy ?? {};
  const evaluating = workflow === "evaluate";

  const verifiedFixed = useMemo(
    () => drafts.map((d) => (d.status === "VERIFIED" && d.expected.intent && d.expected.topic && d.expected.sentiment ? { verified: true, labels: { intent: d.expected.intent, topic: d.expected.topic, sentiment: d.expected.sentiment } } : null)),
    [drafts],
  );

  function updateDraft(i: number, patch: Partial<VariantDraft>) {
    setDrafts((ds) => ds.map((d, j) => (j === i ? { ...d, ...patch } : d)));
    if (patch.text !== undefined) setStale(true);
  }

  async function doRun() {
    setBusy(true);
    setError("");
    setVerdicts({});
    setSavedIds({});
    try {
      const useFixed = evaluating && fixedLabels && mode === "full_pipeline";
      const body: Rec = {
        variants: drafts.map((d) => d.text),
        mode,
        agent_mode: agentMode,
        live_confirmed: agentMode === "live" && liveConfirmed,
        force_classify_prefiltered: forceClassify,
        label_source: useFixed ? "both" : "predicted",
        fixed_labels: useFixed ? verifiedFixed : undefined,
      };
      const result = await api.playgroundRun(body);
      setRun(result);
      setStale(false);
      if (evaluating) {
        const next: Record<number, Rec> = {};
        for (let i = 0; i < drafts.length; i++) {
          const d = drafts[i];
          if (d.status === "UNVERIFIED" && Object.keys(d.expected).length === 0) continue;
          next[i] = await api.playgroundCompare({ run_id: result.run_id, variant_index: i, status: d.status, expected: pruned(d.expected) });
        }
        setVerdicts(next);
      }
    } catch (e: any) {
      setError(e?.message || "request_failed");
    } finally {
      setBusy(false);
    }
  }

  async function saveCase(i: number) {
    if (!run) return;
    setError("");
    try {
      const d = drafts[i];
      const res = await api.playgroundSaveCase({ run_id: run.run_id, variant_index: i, status: d.status, expected: pruned(d.expected), notes: d.notes, case_id: savedIds[i] });
      setSavedIds((s) => ({ ...s, [i]: res.case.case_id }));
    } catch (e: any) {
      setError(e?.message || "request_failed");
    }
  }

  async function loadResults() {
    setError("");
    try {
      const [m, c] = await Promise.all([api.playgroundMetrics(), api.playgroundCases()]);
      setMetrics(m);
      setCases(c.cases || []);
    } catch (e: any) {
      setError(e?.message || "request_failed");
    }
  }

  useEffect(() => {
    if (workflow === "results" && status?.enabled) void loadResults();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workflow, status?.enabled]);

  async function deleteCase(id: string) {
    await api.playgroundDeleteCase(id);
    await loadResults();
  }

  if (statusError) {
    return <div className="workspace workspace--wide"><h2>ساحة التقييم</h2><div className="empty" role="alert">{statusError}</div></div>;
  }
  if (!status) return <div className="workspace workspace--wide"><span>...جارٍ التحميل</span></div>;
  if (!status.enabled) {
    return (
      <div className="workspace workspace--wide">
        <h2>ساحة التقييم التفاعلية (Evaluation Playground)</h2>
        <Panel title="معطّلة" tone="warn">
          <p data-testid="playground-disabled">
            هذه الأداة معطّلة افتراضيًا، ولا تعمل أبدًا في بيئة الإنتاج. للتشغيل محليًا للتطوير فقط، شغّل الخادم مع المتغير
            <code dir="ltr"> RCS_PLAYGROUND_ENABLED=1</code>.
          </p>
          <Dim>السبب: {status.disabled_reason}</Dim>
        </Panel>
      </div>
    );
  }

  const live = agentMode === "live" && mode === "full_pipeline";
  const runDisabled = busy || drafts.every((d) => !d.text.trim()) || (live && !liveConfirmed) || drafts.some((d) => d.text.length > maxChars);

  return (
    <div className="workspace workspace--wide">
      <h2>ساحة التقييم التفاعلية (Evaluation Playground) — أداة تطوير داخلية</h2>
      <p style={{ opacity: 0.8 }}>
        لا تكتب في قواعد بيانات الإنتاج ولا في بيانات gold ولا تُرسل أي رد. لا يُحفظ أي نص تلقائيًا؛ الحفظ يتم فقط عند الضغط على «حفظ الحالة».
        استخدم نصوصًا مخصصة فقط — لا تلصق تعليقات زوار حقيقية.
      </p>

      <div role="tablist" style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        {(["explore", "evaluate", "results"] as Workflow[]).map((w) => (
          <button key={w} role="tab" aria-selected={workflow === w} className={workflow === w ? "btn-primary" : "btn-quiet"} onClick={() => setWorkflow(w)}>
            {w === "explore" ? "استكشاف (Explore)" : w === "evaluate" ? "تقييم (Evaluate)" : "النتائج (Results)"}
          </button>
        ))}
      </div>

      {error ? <div className="empty" role="alert" style={{ marginBottom: 8 }}>{error}</div> : null}

      {workflow === "results" ? (
        metrics ? <MetricsView metrics={metrics} /> : <span>...جارٍ التحميل</span>
      ) : (
        <>
          <Panel title={evaluating ? "التعليقات والتسميات المتوقعة (Evaluate)" : "التعليقات (Explore) — لا يتم احتساب أي مقاييس"}>
            {evaluating ? (
              <Dim>
                لا تُحتسب التسميات المتوقعة كحقيقة مرجعية إلا عند اختيار الحالة «VERIFIED» صراحةً. الحالات «UNVERIFIED» و«NEEDS_HUMAN_REVIEW» لا تُقيَّم ولا تدخل في المقاييس.
              </Dim>
            ) : null}
            {drafts.map((d, i) => (
              <div key={i} style={{ borderBottom: "1px solid var(--border, #eee)", paddingBottom: 8, marginBottom: 8 }}>
                <div style={{ display: "flex", gap: 8 }}>
                  <textarea
                    aria-label={`comment-${i + 1}`}
                    dir="auto"
                    rows={2}
                    value={d.text}
                    maxLength={maxChars + 200}
                    onChange={(e) => updateDraft(i, { text: e.target.value })}
                    style={{ flex: 1 }}
                  />
                  <div>
                    <Dim>{d.text.length}/{maxChars}</Dim>
                    <br />
                    <button type="button" className="btn-quiet" onClick={() => setDrafts((ds) => (ds.length > 1 ? ds.filter((_, j) => j !== i) : ds))} aria-label={`remove-${i + 1}`}>حذف</button>
                  </div>
                </div>
                {evaluating ? (
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 4, alignItems: "center" }}>
                    {EXPECTED_FIELDS.filter((f) => !f.agentOnly || mode === "full_pipeline").map((f) => (
                      <label key={f.key} style={{ background: "rgba(40,90,200,.08)", padding: "2px 6px", borderRadius: 4 }}>
                        <Dim>المتوقع — {f.label}</Dim>
                        <select
                          aria-label={`expected-${f.key}-${i + 1}`}
                          value={d.expected[f.key] || ""}
                          onChange={(e) => updateDraft(i, { expected: { ...d.expected, [f.key]: e.target.value } })}
                        >
                          <option value="">— غير محدد —</option>
                          {(taxonomy[f.taxonomyKey] || []).map((v: string) => <option key={v} value={v}>{v}</option>)}
                        </select>
                      </label>
                    ))}
                    <label>
                      <Dim>حالة الحالة</Dim>
                      <select aria-label={`status-${i + 1}`} value={d.status} onChange={(e) => updateDraft(i, { status: e.target.value as VariantDraft["status"] })}>
                        <option value="UNVERIFIED">UNVERIFIED (غير موثّقة)</option>
                        <option value="VERIFIED">VERIFIED (وثّقتُها يدويًا)</option>
                        <option value="NEEDS_HUMAN_REVIEW">NEEDS_HUMAN_REVIEW (غامضة)</option>
                      </select>
                    </label>
                    <input aria-label={`notes-${i + 1}`} placeholder="ملاحظات" value={d.notes} onChange={(e) => updateDraft(i, { notes: e.target.value })} />
                  </div>
                ) : null}
              </div>
            ))}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button type="button" className="btn-quiet" disabled={drafts.length >= maxVariants} onClick={() => { setDrafts((ds) => [...ds, emptyDraft("")]); setStale(true); }}>
                + إضافة صيغة ({drafts.length}/{maxVariants})
              </button>
              <button type="button" className="btn-quiet" onClick={() => { setDrafts(SAMPLE_VARIANTS.slice(0, maxVariants).map(emptyDraft)); setStale(true); }}>
                تحميل أمثلة الصيغ
              </button>
            </div>
          </Panel>

          <Panel title="وضع التنفيذ (Execution mode)">
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <label><input type="radio" name="pg-mode" checked={mode === "classification_only"} onChange={() => { setMode("classification_only"); setFixedLabels(false); }} /> Classification Only</label>
              <label><input type="radio" name="pg-mode" checked={mode === "full_pipeline"} onChange={() => setMode("full_pipeline")} /> Full Pipeline: Classification → Agent → Response</label>
            </div>
            {mode === "full_pipeline" ? (
              <div style={{ marginTop: 8 }}>
                <strong>وضع الوكيل (Agent mode): </strong>
                <label><input type="radio" name="pg-agent" checked={agentMode === "mock"} onChange={() => { setAgentMode("mock"); setLiveConfirmed(false); }} /> MOCK (الافتراضي — مجاني، بدون شبكة)</label>{" "}
                <label><input type="radio" name="pg-agent" checked={agentMode === "live"} onChange={() => setAgentMode("live")} /> LIVE Gemini</label>
                {agentMode === "mock" ? <div><Dim>الرد التجريبي نص ثابت: يختبر التوجيه والأدوات والتحقق فقط، ولا يدل على جودة ردود Gemini الحقيقية.</Dim></div> : null}
                {agentMode === "live" ? (
                  <div role="alert" style={{ border: "2px solid #b3261e", borderRadius: 6, padding: 8, marginTop: 6 }} data-testid="live-warning">
                    <strong>تحذير التكلفة والخصوصية:</strong> سيُرسل النص الذي كتبته إلى خدمة Gemini الخارجية (استدعاء مدفوع). لا تُدخل بيانات زوار حقيقية.
                    {status.live_configured ? null : <div><strong>لم يتم ضبط مفتاح Gemini على الخادم — ستفشل المحاولة.</strong></div>}
                    {evaluating && fixedLabels ? <div>اختبار التسميات الثابتة يضاعف عدد الاستدعاءات (A + B).</div> : null}
                    <label style={{ display: "block", marginTop: 6 }}>
                      <input type="checkbox" checked={liveConfirmed} onChange={(e) => setLiveConfirmed(e.target.checked)} /> أفهم التكلفة وأن النص سيغادر جهازي
                    </label>
                  </div>
                ) : null}
                {evaluating ? (
                  <label style={{ display: "block", marginTop: 8 }}>
                    <input type="checkbox" checked={fixedLabels} onChange={(e) => setFixedLabels(e.target.checked)} /> اختبار التسميات الثابتة: قارن الوكيل بتنبؤات المصنِّف (A) مقابل التسميات الموثّقة يدويًا (B) — يلزم VERIFIED مع Intent/Topic/Sentiment
                  </label>
                ) : null}
              </div>
            ) : null}
            <label style={{ display: "block", marginTop: 8 }}>
              <input type="checkbox" checked={forceClassify} onChange={(e) => setForceClassify(e.target.checked)} /> صنّف حتى لو استبعده المرشّح المسبق (الإنتاج لا يصنّفه أبدًا)
            </label>
            <div style={{ marginTop: 10 }}>
              <button type="button" className="btn-primary" disabled={runDisabled} onClick={() => void doRun()}>
                {busy ? "...جارٍ التشغيل (قد يستغرق تحميل النماذج بضع ثوانٍ)" : "تشغيل"}
              </button>
            </div>
          </Panel>

          {run ? (
            <Panel title="النتائج">
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                {run.mode === "full_pipeline" ? <ModeBadge mode={run.agent_mode} /> : null}
                <Dim>
                  الزمن الكلي {run.total_ms} ms · دفعة {run.batch_size} · زمن المصنِّفات (يشمل تحميل النماذج): {run.classification_timings_ms ? JSON.stringify(run.classification_timings_ms) : "—"}
                </Dim>
              </div>
              {run.mode_banner ? <div role="status" style={{ fontWeight: 700 }}>{run.mode_banner}</div> : null}
              {stale ? <div role="status" className="empty">تم تعديل النص بعد التشغيل — أعد التشغيل قبل الحفظ.</div> : null}
              {run.variants.length > 1 ? <ComparisonTable run={run} drafts={drafts} /> : null}
              {run.variants.map((v: Rec) => (
                <div key={v.index}>
                  <VariantResultCard variant={v} draft={drafts[v.index] ?? emptyDraft("")} verdict={verdicts[v.index] ?? null} showExpected={evaluating} />
                  {evaluating ? (
                    <button type="button" className="btn-quiet" disabled={stale || drafts[v.index]?.status === "VERIFIED" && Object.keys(pruned(drafts[v.index].expected)).length === 0} onClick={() => void saveCase(v.index)}>
                      {savedIds[v.index] ? `تم الحفظ (${savedIds[v.index]}) — حفظ نسخة جديدة` : "حفظ الحالة"}
                    </button>
                  ) : null}
                </div>
              ))}
            </Panel>
          ) : null}

          <Panel title="ملاحظات التطابق مع الإنتاج (Production parity)">
            <ul style={{ margin: 0, paddingInlineStart: 18 }}>
              {(status.parity_notes || []).map((n: string) => <li key={n}><Dim>{n}</Dim></li>)}
            </ul>
          </Panel>
        </>
      )}

      {workflow === "results" && cases.length ? (
        <Panel title="الحالات المحفوظة">
          <table style={{ width: "100%", fontSize: 13 }}>
            <thead><tr><th>ID</th><th>النص</th><th>الحالة</th><th>الحكم</th><th>الوضع</th><th></th></tr></thead>
            <tbody>
              {cases.map((c) => (
                <tr key={c.case_id}>
                  <td dir="ltr">{c.case_id}</td>
                  <td dir="auto">{c.text}</td>
                  <td dir="ltr">{c.status}</td>
                  <td><span className={VERDICT_STYLE[c.verdict?.verdict] || "badge"}>{c.verdict?.verdict}</span></td>
                  <td>{c.result?.agent_mode ? <ModeBadge mode={c.result.agent_mode} /> : "—"}</td>
                  <td><button type="button" className="btn-quiet" onClick={() => void deleteCase(c.case_id)}>حذف</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      ) : null}
    </div>
  );
}

function pruned(expected: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(expected).filter(([, v]) => v));
}
