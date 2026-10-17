import { useEffect, useRef, useState } from "react";
import { INTENT_LABEL, TOPIC_LABEL } from "../labels";

const INTENT_KEYS = Object.keys(INTENT_LABEL);
const TOPIC_KEYS = Object.keys(TOPIC_LABEL);

/**
 * Multi-select classification filter (primary_intent x topic_subtype).
 * SUGGESTION ("اقتراح", a primary_intent) and DICTIONARY_SUGGESTION
 * ("اقتراح معجمي", a topic_subtype) are deliberately rendered in two
 * separate groups so they are never presented as interchangeable.
 */
export function ClassificationFilter({
  selectedIntents,
  selectedTopics,
  onChange,
}: {
  selectedIntents: string[];
  selectedTopics: string[];
  onChange: (intents: string[], topics: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const count = selectedIntents.length + selectedTopics.length;

  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, []);

  function toggle(list: string[], value: string): string[] {
    return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
  }

  return (
    <div className="classification-filter" ref={ref}>
      <button type="button" className={`chip ${count ? "active" : ""}`} onClick={() => setOpen(!open)} aria-expanded={open}>
        التصنيف{count ? ` (${count})` : ""}
      </button>
      {open ? (
        <div className="classification-filter-panel" role="dialog" aria-label="تصفية حسب التصنيف">
          <div className="classification-filter-group">
            <p className="classification-filter-heading">النية</p>
            {INTENT_KEYS.map((key) => (
              <label key={key} className="classification-filter-option">
                <input
                  type="checkbox"
                  checked={selectedIntents.includes(key)}
                  onChange={() => onChange(toggle(selectedIntents, key), selectedTopics)}
                />
                {INTENT_LABEL[key]}
              </label>
            ))}
          </div>
          <div className="classification-filter-group">
            <p className="classification-filter-heading">الموضوع</p>
            {TOPIC_KEYS.map((key) => (
              <label key={key} className="classification-filter-option">
                <input
                  type="checkbox"
                  checked={selectedTopics.includes(key)}
                  onChange={() => onChange(selectedIntents, toggle(selectedTopics, key))}
                />
                {TOPIC_LABEL[key]}
              </label>
            ))}
          </div>
          {count ? (
            <button type="button" className="linkish classification-filter-clear" onClick={() => onChange([], [])}>
              مسح التصنيف
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
