import type { QueueItem } from "../api/types";

function waitStamp(item: QueueItem): string {
  return item.ingested_at || item.comment_created_at || "";
}

/** Highest existing review_priority first; oldest waiting within the same level. */
export function orderStartingList(items: QueueItem[]): QueueItem[] {
  return [...items].sort((a, b) => {
    const pa = a.review_priority ?? 99;
    const pb = b.review_priority ?? 99;
    if (pa !== pb) return pa - pb;
    return waitStamp(a).localeCompare(waitStamp(b));
  });
}
