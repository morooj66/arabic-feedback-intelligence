export const APP_TIMEZONE = "Asia/Riyadh";

export function hourInAppTz(at: Date = new Date()): number {
  const hour = new Intl.DateTimeFormat("en-GB", {
    timeZone: APP_TIMEZONE,
    hour: "numeric",
    hour12: false,
  }).format(at);
  return Number.parseInt(hour, 10);
}

export function greetingPrefix(at: Date = new Date()): string {
  return hourInAppTz(at) < 12 ? "صباح الخير" : "مساء الخير";
}

export function firstName(name: string): string {
  return name.trim().split(/\s+/)[0] || name;
}

export function parseAppInstant(iso: string): Date {
  if (/[zZ]|[+-]\d{2}:\d{2}$/.test(iso)) return new Date(iso);
  return new Date(`${iso}+03:00`);
}

export function formatYmd(at: Date = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: APP_TIMEZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(at);
  const year = parts.find((p) => p.type === "year")?.value;
  const month = parts.find((p) => p.type === "month")?.value;
  const day = parts.find((p) => p.type === "day")?.value;
  return `${year}-${month}-${day}`;
}

export function formatArabicDate(isoDate: string): string {
  if (!isoDate) return "";
  const d = parseAppInstant(`${isoDate}T12:00:00+03:00`);
  if (Number.isNaN(d.getTime())) return isoDate;
  return new Intl.DateTimeFormat("ar-SA", {
    timeZone: APP_TIMEZONE,
    day: "numeric",
    month: "long",
    year: "numeric",
  }).format(d);
}

export function formatTrendBucket(bucket: string): string {
  const hourly = /^\d{4}-\d{2}-\d{2} \d{2}$/.test(bucket);
  const iso = hourly ? `${bucket.replace(" ", "T")}:00:00+03:00` : `${bucket}T12:00:00+03:00`;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return bucket;
  if (hourly) {
    return new Intl.DateTimeFormat("ar-SA", {
      timeZone: APP_TIMEZONE,
      day: "numeric",
      month: "long",
      hour: "numeric",
      minute: "2-digit",
    }).format(d);
  }
  return new Intl.DateTimeFormat("ar-SA", {
    timeZone: APP_TIMEZONE,
    day: "numeric",
    month: "long",
    year: "numeric",
  }).format(d);
}
