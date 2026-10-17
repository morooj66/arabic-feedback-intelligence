import { INTENT_LABEL, SENTIMENT_LABEL, TOPIC_LABEL } from "../labels";

export const CANONICAL_INTENT_KEYS = ["THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM"] as const;
export const CANONICAL_TOPIC_KEYS = [
  "WORD_SUBMISSION",
  "ENTRY_CORRECTION",
  "DICTIONARY_SUGGESTION",
  "TECHNICAL_ISSUE",
  "WORD_OF_YEAR_NOMINATION",
  "OFF_TOPIC_PERSONAL",
  "GENERAL",
] as const;
export const CANONICAL_SENTIMENT_KEYS = ["POSITIVE", "NEGATIVE", "NEUTRAL"] as const;

export function classificationShare(count: number, total: number): number {
  return total ? count / total : 0;
}

export function outcomeHover(label: string, count: number, total: number): string {
  const pct = Math.round(classificationShare(count, total) * 100);
  return `${label}\n${count} تعليقات\n${pct}٪`;
}

export function classHover(label: string, count: number, total: number): string {
  const pct = Math.round(classificationShare(count, total) * 100);
  return `${label}\n${count}\n${pct}٪`;
}

export function trendHover(bucketLabel: string, incoming: number, reviewed: number): string {
  return `${bucketLabel}\nالتعليقات الواردة: ${incoming}\nتمت مراجعتها: ${reviewed}`;
}

export function canonicalChartRows(
  dimension: "intent" | "topic" | "sentiment",
  rows: { key: string; count: number }[] | undefined,
): { key: string; count: number }[] {
  const keys =
    dimension === "intent" ? CANONICAL_INTENT_KEYS : dimension === "topic" ? CANONICAL_TOPIC_KEYS : CANONICAL_SENTIMENT_KEYS;
  const counts = new Map((rows || []).map((row) => [row.key, row.count]));
  return keys.map((key) => ({ key, count: counts.get(key) ?? 0 }));
}

export function canonicalClassCount(): number {
  return CANONICAL_INTENT_KEYS.length + CANONICAL_TOPIC_KEYS.length + CANONICAL_SENTIMENT_KEYS.length;
}

export function arabicClassLabel(dimension: "intent" | "topic" | "sentiment", key: string): string {
  const map = dimension === "intent" ? INTENT_LABEL : dimension === "topic" ? TOPIC_LABEL : SENTIMENT_LABEL;
  return map[key] || key;
}
