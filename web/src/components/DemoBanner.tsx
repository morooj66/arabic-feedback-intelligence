import { useEffect, useState } from "react";
import { api } from "../api/client";
import "./DemoBanner.css";

/** Renders ONLY when the server reports that it is a synthetic-data demo instance (health.demo). A small fixed pill: it does not move or restyle anything else. */
export function DemoBanner() {
  const [demo, setDemo] = useState(false);
  useEffect(() => {
    let alive = true;
    Promise.resolve(api.health?.())
      .then((status) => { if (alive) setDemo(Boolean(status?.demo)); })
      .catch(() => { if (alive) setDemo(false); });
    return () => { alive = false; };
  }, []);
  if (!demo) return null;
  return <div className="demo-banner" role="note" dir="rtl">نسخة تجريبية — بيانات اصطناعية فقط. لا تُدخل بيانات حقيقية.</div>;
}
