import type { QueueItem, WorkflowCounts } from "../api/types";

const DAY_MS = 86400000;

export type AgeBucketKey = "under_day" | "one_two" | "three_seven" | "over_seven";

export const AGE_BUCKETS: { key: AgeBucketKey; label: string }[] = [
  { key: "under_day", label: "أقل من يوم" },
  { key: "one_two", label: "1–2 يوم" },
  { key: "three_seven", label: "3–7 أيام" },
  { key: "over_seven", label: "أكثر من 7 أيام" },
];

export function ageBucket(iso: string | null | undefined, now = Date.now()): AgeBucketKey | null {
  if (!iso) return null;
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return null;
  const days = (now - then) / DAY_MS;
  if (days < 1) return "under_day";
  if (days < 3) return "one_two";
  if (days < 8) return "three_seven";
  return "over_seven";
}

export function queueHealth(items: QueueItem[], now = Date.now()): { counts: Record<AgeBucketKey, number>; dated: number } {
  const counts: Record<AgeBucketKey, number> = {
    under_day: 0,
    one_two: 0,
    three_seven: 0,
    over_seven: 0,
  };
  let dated = 0;
  for (const item of items) {
    const bucket = ageBucket(item.ingested_at || item.comment_created_at, now);
    if (!bucket) continue;
    counts[bucket] += 1;
    dated += 1;
  }
  return { counts, dated };
}

export function reviewOutcomes(byStatus: Record<string, number> | undefined): {
  sendAsIs: number;
  edited: number;
  noReply: number;
  total: number;
} {
  const src = byStatus || {};
  const sendAsIs = src.APPROVED || 0;
  const edited = src.EDITED_AND_APPROVED || 0;
  const noReply = src.NO_REPLY || 0;
  return { sendAsIs, edited, noReply, total: sendAsIs + edited + noReply };
}

export function reviewedCount(counts: WorkflowCounts): number {
  return Math.max(0, counts.all - counts.pending);
}
