import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { AnalyticsRange, AnalyticsSnapshot } from "../api/types";
import { INTENT_LABEL, SENTIMENT_LABEL, TOPIC_LABEL } from "../labels";
import { canonicalChartRows } from "../analytics/display";
import { formatArabicDate, formatTrendBucket, formatYmd } from "../time/appTime";
import { formatDurationAr } from "../time/relative";
import "./AnalyticsPage.css";

const number = new Intl.NumberFormat("ar-SA");
const percent = new Intl.NumberFormat("ar-SA", { style: "percent", maximumFractionDigits: 1 });
const share = (n: number, total: number) => total > 0 ? percent.format(n / total) : "—";
const RANGES: { key: AnalyticsRange; label: string }[] = [
  { key: "today", label: "اليوم" }, { key: "week", label: "آخر ٧ أيام" },
  { key: "month", label: "هذا الشهر" }, { key: "custom", label: "نطاق مخصص" },
];
const OUTCOMES = [
  { key: "SEND_AS_IS", label: "اعتماد دون تعديل", note: "من قرارات المراجعة البشرية" },
  { key: "EDIT_AND_SEND", label: "تعديل ثم اعتماد", note: "من قرارات المراجعة البشرية" },
  { key: "NO_RESPONSE_NEEDED", label: "بدون رد", note: "قرار بشري بعد المراجعة" },
];

export function AnalyticsPage() {
  const [range, setRange] = useState<AnalyticsRange>("week");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [data, setData] = useState<AnalyticsSnapshot | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [revision, setRevision] = useState(0);
  const customInvalid = range === "custom" && Boolean(start && end && start > end);
  const customIncomplete = range === "custom" && (!start || !end);
  const valid = !customInvalid && !customIncomplete;

  useEffect(() => {
    if (window.scrollY !== 0) window.scrollTo(0, 0);
  }, []);
  useEffect(() => {
    setData(null);
    if (!valid) return;
    let cancelled = false;
    setState("loading");
    api.analytics({ range, start: range === "custom" ? start : undefined, end: range === "custom" ? end : undefined })
      .then(snap => { if (!cancelled) { setData(snap); setState("ready"); } })
      .catch(() => { if (!cancelled) setState("error"); });
    return () => { cancelled = true; };
  }, [range, start, end, revision, valid]);

  function chooseRange(next: AnalyticsRange) {
    if (next === range) return;
    setData(null); setRange(next);
    if (next === "custom") { setStart(start || formatYmd()); setEnd(end || formatYmd()); }
  }
  function reset() { setData(null); setRange("week"); setStart(""); setEnd(""); setRevision(n => n + 1); }

  const outcomeTotal = data ? OUTCOMES.reduce((sum, row) => sum + (data.outcomes[row.key] || 0), 0) : 0;
  const intents = canonicalChartRows("intent", data?.classification.intent);
  const intentTotal = intents.reduce((sum, row) => sum + row.count, 0);
  const suggestions = intents.find(row => row.key === "SUGGESTION")?.count || 0;
  const unknownLabels = data && (Boolean(data.data_quality?.note) || (["intent", "topic", "sentiment"] as const).some(dim => {
    const keys = new Set(canonicalChartRows(dim, []).map(row => row.key));
    return data.classification[dim].some(row => !keys.has(row.key));
  }));
  const context = data ? `${data.range_start ? formatArabicDate(data.range_start.slice(0, 10)) : "بداية البيانات المتاحة"} — ${formatArabicDate(data.range_end.slice(0, 10))}` : RANGES.find(r => r.key === range)?.label;

  return <div className="analytics-workspace" dir="rtl">
    <header className="analytics-heading"><div><h2>التحليلات</h2><p>نظرة شاملة على حجم التعليقات، سير المراجعة، ونتائج الردود.</p></div><span className="analytics-eyebrow">المراجعة البشرية · صوت الزائر</span></header>
    <section className="analytics-panel analytics-filters" aria-label="تصفية التحليلات">
      <div className="analytics-filter-top"><div role="group" aria-label="النطاق الزمني">{RANGES.map(r => <button key={r.key} type="button" aria-pressed={range === r.key} onClick={() => chooseRange(r.key)}>{r.label}</button>)}</div><button type="button" className="analytics-text-button" disabled={!valid || state === "loading"} onClick={() => { setData(null); setRevision(n => n + 1); }}>تحديث البيانات</button></div>
      {range === "custom" ? <div className="analytics-custom"><label><span>من تاريخ</span><input aria-label="من تاريخ" type="date" value={start} aria-invalid={customInvalid || customIncomplete} aria-describedby={!valid ? "analytics-range-error" : undefined} onChange={e => { setData(null); setStart(e.target.value); }} /></label><label><span>إلى تاريخ</span><input aria-label="إلى تاريخ" type="date" value={end} aria-invalid={customInvalid || customIncomplete} aria-describedby={!valid ? "analytics-range-error" : undefined} onChange={e => { setData(null); setEnd(e.target.value); }} /></label></div> : null}
      <div className="analytics-filter-context"><p>الفترة: <strong>{context}</strong><span> · بتوقيت الرياض</span></p>{range !== "week" ? <button type="button" className="analytics-text-button" onClick={reset}>إعادة ضبط الفترة</button> : null}</div>
      {!valid ? <p className="analytics-error-text" id="analytics-range-error" role="alert">{customInvalid ? "تاريخ البداية يجب أن يسبق تاريخ النهاية أو يساويه." : "اختر تاريخ البداية والنهاية لعرض التحليلات."}</p> : null}
    </section>

    {valid && state === "loading" ? <div className="analytics-panel analytics-state" role="status"><span className="analytics-loading-mark" aria-hidden="true" /><h3>جاري تحميل التحليلات…</h3><p>تُحدّث مؤشرات المراجعة وصوت الزائر للفترة المختارة.</p></div> : null}
    {valid && state === "error" ? <div className="analytics-panel analytics-state" role="alert"><h3>تعذر تحميل التحليلات.</h3><p>الفترة المختارة محفوظة. حاول مجددًا.</p><button type="button" onClick={() => setRevision(n => n + 1)}>إعادة المحاولة</button></div> : null}
    {valid && state === "ready" && data ? <>
      <section className="analytics-kpis" aria-label="ملخص الفترة">
        <Kpi label="التعليقات الواردة" value={number.format(data.kpis.incoming)} note="بحسب تاريخ ورود التعليق في الفترة" />
        <Kpi label="إجمالي المراجعات البشرية" value={number.format(outcomeTotal)} note="قرارات بشرية مسجّلة خلال الفترة" />
        <Kpi label="متوسط زمن المراجعة" value={data.kpis.avg_review_seconds == null ? "غير متاح" : formatDurationAr(data.kpis.avg_review_seconds)} note={data.kpis.avg_review_sample > 0 ? `من ${number.format(data.kpis.avg_review_sample)} مراجعة ذات توقيت مكتمل` : "لا تتوفر مراجعات ذات توقيت مكتمل"} />
      </section>
      <section className="analytics-section" aria-labelledby="analytics-operations-title">
        <header className="analytics-section-heading"><span className="analytics-section-marker" aria-hidden="true">01</span><div><h3 id="analytics-operations-title">جودة وعملية المراجعة</h3><p>كيف تسير المراجعة البشرية، وما حجم التعديل على الردود؟</p></div></header>
        <section className="analytics-panel" aria-labelledby="analytics-quality-title">
          <div className="analytics-panel-heading"><h4 id="analytics-quality-title">جودة الردود والمراجعة</h4><span>قرارات الفترة المختارة</span></div>
          <div className="analytics-outcomes">{OUTCOMES.map((row, i) => {
            const count = data.outcomes[row.key] || 0;
            return <article key={row.key} className={`analytics-outcome analytics-tone-${i}`}><h5>{row.label}</h5><div><strong>{number.format(count)}</strong><span>{share(count, outcomeTotal)}</span></div><div className="analytics-meter" aria-hidden="true"><i style={{ width: `${outcomeTotal ? count / outcomeTotal * 100 : 0}%` }} /></div><p>{row.note}</p></article>;
          })}</div>
          <p className="analytics-footnote">{outcomeTotal ? `النسب من ${number.format(outcomeTotal)} قرار بشري مسجّل؛ لا تشمل الإغلاق الآلي دون رد، ولا تمثل تقييمًا للمراجعين.` : "لا توجد قرارات مراجعة بشرية مسجلة في هذه الفترة؛ النسب غير متاحة."}</p>
        </section>
        <section className="analytics-panel" aria-labelledby="analytics-trend-title">
          <div className="analytics-panel-heading"><div><h4 id="analytics-trend-title">اتجاه التعليقات والمراجعة</h4><p>الوارد حسب تاريخ الإضافة، والمراجعات حسب تاريخ إتمامها.</p></div><span>خلال الفترة</span></div>
          {data.trend.some(p => p.incoming || p.reviewed) ? <TrendChart key={data.range_start + data.range_end} points={data.trend} /> : <p className="analytics-empty">لا توجد حركة مؤرخة لعرضها في هذه الفترة.</p>}
          <p className="analytics-footnote">قد تتضمن المراجعات تعليقات وردت قبل الفترة؛ الخطان لا يمثلان نسبة إنجاز لمجموعة واحدة.</p>
        </section>
        <details className="analytics-panel analytics-queue"><summary>حالة قائمة الانتظار <span>لقطة حالية · خارج مرشح الفترة</span></summary><p className="analytics-footnote">تعكس العناصر المعلقة الآن، بصرف النظر عن فترة التقرير أعلاه.</p><div className="analytics-queue-kpis"><Kpi label="بانتظار المراجعة" value={number.format(data.kpis.pending)} note="إجمالي المعلق حاليًا" /><Kpi label="أولوية عاجلة" value={number.format(data.kpis.urgent_pending)} note="من العناصر المعلقة حاليًا" /><Kpi label="التعليقات المتأخرة" value={number.format(data.kpis.overdue_pending)} note="معلقة منذ أكثر من ٧ أيام" /></div>{data.queue_health.dated > 0 ? <><div className="analytics-age-bars">{[{label:"أقل من يوم",n:data.queue_health.under_day},{label:"١–٢ يوم",n:data.queue_health.one_two},{label:"٣–٧ أيام",n:data.queue_health.three_seven},{label:"أكثر من ٧ أيام",n:data.queue_health.over_seven}].map(row => <div key={row.label}><p>{row.label}<strong>{number.format(row.n)}</strong></p><div className="analytics-meter" aria-hidden="true"><i style={{width:`${row.n / data.queue_health.dated * 100}%`}} /></div></div>)}</div><p className="analytics-footnote">أعمار {number.format(data.queue_health.dated)} تعليقًا معلقًا ذي تاريخ ورود متاح.{data.queue_health.oldest_pending_age_seconds != null ? ` أقدمها منذ ${formatDurationAr(data.queue_health.oldest_pending_age_seconds)}.` : ""}</p></> : <p className="analytics-empty">{data.kpis.pending ? "لا تتوفر تواريخ كافية لحساب أعمار العناصر المعلقة." : "لا توجد تعليقات بانتظار المراجعة حاليًا."}</p>}</details>
      </section>

      <section className="analytics-section" aria-labelledby="analytics-voice-title">
        <header className="analytics-section-heading"><span className="analytics-section-marker" aria-hidden="true">02</span><div><h3 id="analytics-voice-title">صوت الزائر والرؤى</h3><p>ما الذي يقوله الزوار، وما الموضوعات التي يتحدثون عنها؟</p></div></header>
        <article className="analytics-suggestions"><div><h4>التعليقات المصنفة كاقتراحات</h4><p>تعليقات صُنّفت ضمن نية «اقتراح» خلال الفترة المحددة.</p></div><div className="analytics-suggestion-value"><strong>{number.format(suggestions)}</strong><span>{share(suggestions, intentTotal)} من النوايا المصنفة</span></div></article>
        <section aria-labelledby="analytics-class-title"><div className="analytics-class-heading"><h4 id="analytics-class-title">تحليل التعليقات</h4><p>توزيعات مستقلة للتعليقات الواردة خلال الفترة، وليست تقاطعًا بين النية والموضوع.</p></div><div className="analytics-distributions"><Dist title="النية" dimension="intent" rows={data.classification.intent} labels={INTENT_LABEL} /><Dist title="الموضوع" dimension="topic" rows={data.classification.topic} labels={TOPIC_LABEL} /></div></section>
        {unknownLabels ? <p className="analytics-data-note">توجد قيم تصنيف خارج التصنيف المعتمد؛ استُبعدت من العرض ومن حساب النسب، دون دمجها في فئات أخرى.</p> : null}
        <details className="analytics-panel analytics-sentiment"><summary>نبرة التعليق <span>تفصيل إضافي من التصنيف المسجّل</span></summary><Dist title="توزيع النبرة" dimension="sentiment" rows={data.classification.sentiment} labels={SENTIMENT_LABEL} /></details>
      </section>
    </> : null}
  </div>;
}

function Kpi({ label, value, note }: { label: string; value: string; note: string }) {
  return <article className="analytics-kpi"><h3>{label}</h3><strong>{value}</strong><p>{note}</p></article>;
}

function Dist({ title, dimension, rows, labels }: { title: string; dimension: "intent" | "topic" | "sentiment"; rows: { key: string; count: number }[]; labels: Record<string, string> }) {
  const chartRows = canonicalChartRows(dimension, rows);
  const total = chartRows.reduce((sum, row) => sum + row.count, 0);
  return <section className="analytics-panel analytics-dist" data-testid={`class-dist-${dimension}`} aria-label={`توزيع ${title}`}><div className="analytics-panel-heading"><h4>{title}</h4><span>{number.format(total)} تعليقًا مصنفًا</span></div>
    {!total ? <p className="analytics-empty">لا توجد تصنيفات معتمدة لهذا البعد في الفترة المختارة.</p> : null}
    <ul>{chartRows.map(row => <li key={row.key}><div className="analytics-dist-labels"><span>{labels[row.key]}</span><strong>{number.format(row.count)} <small>({share(row.count, total)})</small></strong></div><div className="analytics-meter" aria-hidden="true"><i style={{width:`${total ? row.count / total * 100 : 0}%`}} /></div></li>)}</ul>
    <p className="analytics-footnote">النسب من التعليقات ذات التصنيف المعتمد لهذا البعد فقط.</p>
  </section>;
}

function TrendChart({ points }: { points: AnalyticsSnapshot["trend"] }) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  useEffect(() => {
    if (!ref.current || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(entries => setWidth(Math.max(240, entries[0].contentRect.width)));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  const h = 210, left = 38, right = width - 12, top = 16, bottom = h - 18;
  const max = Math.max(1, ...points.flatMap(p => [p.incoming, p.reviewed]));
  // Earliest bucket at the right: explicit RTL chronological progression.
  const x = (i: number) => points.length === 1 ? (left + right) / 2 : right - i / (points.length - 1) * (right - left);
  const y = (n: number) => bottom - n / max * (bottom - top);
  const line = (key: "incoming" | "reviewed") => points.map((p,i) => `${x(i)},${y(p[key])}`).join(" ");
  const totals = points.reduce((t,p) => ({ incoming:t.incoming+p.incoming,reviewed:t.reviewed+p.reviewed }),{incoming:0,reviewed:0});
  return <div ref={ref} className="analytics-trend">
    <div className="analytics-legend"><span><i className="analytics-line-solid" aria-hidden="true" />الوارد</span><span><i className="analytics-line-dashed" aria-hidden="true" />المراجعات المؤرخة</span></div>
    <svg viewBox={`0 0 ${width} ${h}`} role="img" aria-label={`اتجاه الوارد والمراجعة: ${number.format(totals.incoming)} وارد، ${number.format(totals.reviewed)} مراجعة مؤرخة. القيم التفصيلية متاحة في الجدول أدناه.`}>
      {(max < 2 ? [0, max] : [0, Math.ceil(max / 2), max]).map(t => <g key={t}><line x1={left} x2={right} y1={y(t)} y2={y(t)} stroke="#e5ece7" /><text x={left-8} y={y(t)+4} textAnchor="end" direction="ltr" fontSize="12" fill="#68766e">{number.format(t)}</text></g>)}
      <polyline points={line("incoming")} fill="none" stroke="#2f6b4f" strokeWidth="2.5" />
      <polyline points={line("reviewed")} fill="none" stroke="#78857c" strokeWidth="2.5" strokeDasharray="6 5" />
      {points.length === 1 ? <><circle cx={x(0)} cy={y(points[0].incoming)} r="4" fill="#2f6b4f" /><circle cx={x(0)} cy={y(points[0].reviewed)} r="4" fill="#78857c" /></> : null}
    </svg>
    <div className="analytics-trend-dates"><span>{formatTrendBucket(points[0].bucket)}</span>{points.length > 1 ? <span>{formatTrendBucket(points[points.length-1].bucket)}</span> : null}</div>
    <details className="analytics-trend-values"><summary>عرض القيم الزمنية</summary><div className="analytics-values-scroll" tabIndex={0} role="region" aria-label="القيم الزمنية"><table><caption>القيم المسجلة لكل فترة زمنية</caption><thead><tr><th scope="col">الفترة</th><th scope="col">الوارد</th><th scope="col">المراجعات المؤرخة</th></tr></thead><tbody>{points.map(p => <tr key={p.bucket}><th scope="row">{formatTrendBucket(p.bucket)}</th><td>{number.format(p.incoming)}</td><td>{number.format(p.reviewed)}</td></tr>)}</tbody></table></div></details>
  </div>;
}
