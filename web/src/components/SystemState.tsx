import type { ReactNode } from "react";

export type SystemStateKind = "loading" | "empty" | "noResults" | "unavailable" | "permission" | "error" | "success";

/** Semantic state wrapper. Deliberately adds no styling or behavior to locked screens.
 * Children must be curated UI copy, never an API exception's message/detail.
 * Use announce=false when the same message is already announced in a modal.
 */
export function SystemState({ kind, as: Tag = "div", className, children, announce = true }: {
  kind: SystemStateKind;
  as?: "div" | "p";
  className?: string;
  children: ReactNode;
  announce?: boolean;
}) {
  const urgent = kind === "error" || kind === "permission";
  return <Tag className={className} data-system-state={kind} role={announce ? urgent ? "alert" : "status" : undefined} aria-live={announce ? urgent ? "assertive" : "polite" : "off"} aria-atomic="true" dir="rtl">{children}</Tag>;
}
