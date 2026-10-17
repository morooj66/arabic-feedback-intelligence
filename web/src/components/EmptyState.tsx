import type { ReactNode } from "react";
import { SystemState } from "./SystemState";

/**
 * Shared, restrained empty-state pattern used everywhere a list/queue
 * can legitimately have nothing in it — new admin, caught-up reviewer,
 * no filter matches. Institutional tone: a plain statement of fact,
 * never celebratory language ("أحسنت!") and never a fake metric.
 */
export function EmptyState({
  title,
  description,
  action,
  kind = "empty",
}: {
  title: string;
  description?: string;
  action?: ReactNode;
  kind?: "empty" | "noResults" | "unavailable";
}) {
  return (
    <SystemState kind={kind} className="empty-state">
      <p className="empty-state-title">{title}</p>
      {description ? <p className="empty-state-desc">{description}</p> : null}
      {action ? <div className="empty-state-action">{action}</div> : null}
    </SystemState>
  );
}
