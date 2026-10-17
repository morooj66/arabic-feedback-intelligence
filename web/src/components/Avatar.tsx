/**
 * Neutral institutional initial avatar. Deliberately never infers
 * gender from a name — an initial derived purely from the first
 * character(s) of `displayName`, nothing else. If a real authorized
 * profile image becomes available later, add an optional `photoUrl`
 * prop here rather than inventing one now.
 */
export function initials(displayName: string): string {
  const first = displayName.trim().split(/\s+/)[0] || "";
  return first[0] || "م";
}

export function Avatar({ displayName, size = "md" }: { displayName: string; size?: "sm" | "md" }) {
  return (
    <span className={`avatar avatar-${size}`} aria-hidden="true">
      {initials(displayName)}
    </span>
  );
}
