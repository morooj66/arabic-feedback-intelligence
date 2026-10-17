import { APP_TIMEZONE, parseAppInstant } from "./appTime";

const MINUTE = 60;
const HOUR = 3600;
const DAY = 86400;
const WEEK = 604800;

export function formatRelativeAr(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "—";
  const then = parseAppInstant(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const seconds = Math.max(0, Math.floor((now - then) / 1000));
  if (seconds < 45) return "منذ لحظات";
  if (seconds < HOUR) {
    const n = Math.max(1, Math.round(seconds / MINUTE));
    return n === 1 ? "منذ دقيقة" : n === 2 ? "منذ دقيقتين" : `منذ ${n} دقيقة`;
  }
  if (seconds < DAY) {
    const n = Math.max(1, Math.round(seconds / HOUR));
    return n === 1 ? "منذ ساعة" : n === 2 ? "منذ ساعتين" : `منذ ${n} ساعات`;
  }
  if (seconds < WEEK) {
    const n = Math.max(1, Math.round(seconds / DAY));
    if (n === 1) return "منذ يوم";
    if (n === 2) return "منذ يومين";
    return `منذ ${n} أيام`;
  }
  const weeks = Math.round(seconds / WEEK);
  if (weeks === 1) return "منذ أسبوع";
  if (weeks === 2) return "منذ أسبوعين";
  return `منذ ${weeks} أسابيع`;
}

export function formatDurationAr(seconds: number | null | undefined): string {
  if (seconds == null || seconds < 0) return "غير متاح حاليًا";
  if (seconds < 60) return "أقل من دقيقة";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return minutes === 1 ? "دقيقة واحدة" : `${minutes} دقيقة`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return hours === 1 ? "ساعة واحدة" : `${hours} ساعات`;
  const days = Math.round(hours / 24);
  return days === 1 ? "يوم واحد" : `${days} أيام`;
}

export function isOverdue(iso: string | null | undefined, now = Date.now()): boolean {
  if (!iso) return false;
  const then = parseAppInstant(iso).getTime();
  return now - then > 7 * DAY * 1000;
}

export function formatAbsoluteAr(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = parseAppInstant(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return new Intl.DateTimeFormat("ar-SA", {
    timeZone: APP_TIMEZONE,
    day: "numeric",
    month: "long",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(d);
}
