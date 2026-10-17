import { useEffect, useRef, useState } from "react";
import { PRIORITY_LABEL, SENTIMENT_LABEL } from "../labels";

/** Optional list filters shared by the reviewer queue, Assignment and Completed Reviews. */
export type ExtraFilterValue = {
  sentiments: string[];
  priorities: string[];
  createdFrom: string;
  createdTo: string;
};

export const EMPTY_EXTRA: ExtraFilterValue = { sentiments: [], priorities: [], createdFrom: "", createdTo: "" };

export function extraFilterCount(v: ExtraFilterValue): number {
  return v.sentiments.length + v.priorities.length + (v.createdFrom ? 1 : 0) + (v.createdTo ? 1 : 0);
}

export function extraFilterLabels(v: ExtraFilterValue): string[] {
  return [
    ...v.sentiments.map(k => `المشاعر: ${SENTIMENT_LABEL[k] || k}`),
    ...v.priorities.map(k => `الأولوية: ${PRIORITY_LABEL[Number(k)] || k}`),
    ...(v.createdFrom ? [`من تاريخ التعليق: ${v.createdFrom}`] : []),
    ...(v.createdTo ? [`إلى تاريخ التعليق: ${v.createdTo}`] : []),
  ];
}

/** Query-string params understood by /api/queue and /api/admin/assignable-items. */
export function extraFilterParams(v: ExtraFilterValue) {
  return {
    sentiments: v.sentiments,
    priorities: v.priorities,
    createdFrom: v.createdFrom || undefined,
    createdTo: v.createdTo || undefined,
  };
}

function OptionsChip({ label, options, selected, onChange }: {
  label: string;
  options: { key: string; label: string }[];
  selected: string[];
  onChange: (next: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, []);
  return (
    <div className="classification-filter" ref={ref}>
      <button type="button" className={`chip ${selected.length ? "active" : ""}`} onClick={() => setOpen(!open)} aria-expanded={open}>
        {label}{selected.length ? ` (${selected.length})` : ""}
      </button>
      {open ? (
        <div className="classification-filter-panel" role="dialog" aria-label={`تصفية حسب ${label}`}>
          <div className="classification-filter-group">
            {options.map(o => (
              <label key={o.key} className="classification-filter-option">
                <input
                  type="checkbox"
                  checked={selected.includes(o.key)}
                  onChange={() => onChange(selected.includes(o.key) ? selected.filter(k => k !== o.key) : [...selected, o.key])}
                />
                {o.label}
              </label>
            ))}
          </div>
          {selected.length ? (
            <button type="button" className="linkish classification-filter-clear" onClick={() => onChange([])}>مسح</button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

const SENTIMENT_OPTIONS = Object.keys(SENTIMENT_LABEL).map(key => ({ key, label: SENTIMENT_LABEL[key] }));
const PRIORITY_OPTIONS = Object.keys(PRIORITY_LABEL).map(key => ({ key, label: PRIORITY_LABEL[Number(key)] }));

/** Sentiment, priority and comment-date-range filters. Presentation only; state lives in the page. */
export function ExtraFilters({ value, onChange }: { value: ExtraFilterValue; onChange: (next: ExtraFilterValue) => void }) {
  return (
    <div className="extra-filters">
      <div><span>المشاعر</span><OptionsChip label="المشاعر" options={SENTIMENT_OPTIONS} selected={value.sentiments} onChange={sentiments => onChange({ ...value, sentiments })} /></div>
      <div><span>الأولوية</span><OptionsChip label="الأولوية" options={PRIORITY_OPTIONS} selected={value.priorities} onChange={priorities => onChange({ ...value, priorities })} /></div>
      <label><span>من تاريخ التعليق</span><input type="date" aria-label="من تاريخ التعليق" value={value.createdFrom} max={value.createdTo || undefined} onChange={e => onChange({ ...value, createdFrom: e.target.value })} /></label>
      <label><span>إلى تاريخ التعليق</span><input type="date" aria-label="إلى تاريخ التعليق" value={value.createdTo} min={value.createdFrom || undefined} onChange={e => onChange({ ...value, createdTo: e.target.value })} /></label>
    </div>
  );
}
