/** Reviewer composer enablement. Do not key off a separate `original` copy. */

const COMPOSER_EDITABLE = new Set(["DRAFT_READY", "REVIEW_ONLY", "IN_REVIEW", "NO_RESPONSE_RECOMMENDED"]);

export function isComposerEditable(status: string | undefined | null): boolean {
  return Boolean(status && COMPOSER_EDITABLE.has(status));
}

export function composerActionFlags(
  allowedActions: string[] | undefined,
  suggestedResponse: string | null | undefined,
  draft: string,
): { suggestion: string; dirty: boolean; canSendAsIs: boolean; canEditSend: boolean; canNoReply: boolean } {
  const suggestion = (suggestedResponse || "").trim();
  const text = draft.trim();
  const dirty = text !== suggestion;
  const allowed = allowedActions || [];
  return {
    suggestion,
    dirty,
    canSendAsIs: allowed.includes("SEND_AS_IS") && !dirty && suggestion.length > 0,
    canEditSend: allowed.includes("EDIT_AND_SEND") && text.length > 0 && dirty,
    canNoReply: allowed.includes("NO_RESPONSE_NEEDED"),
  };
}
