import { describe, expect, it } from "vitest";
import { composerActionFlags, isComposerEditable } from "./composerActions";

describe("composerActionFlags", () => {
  it("disables both send actions when there is no Gemini draft and the editor is empty", () => {
    const flags = composerActionFlags(["EDIT_AND_SEND", "NO_RESPONSE_NEEDED"], null, "");
    expect(flags.canSendAsIs).toBe(false);
    expect(flags.canEditSend).toBe(false);
    expect(flags.canNoReply).toBe(true);
  });

  it("enables EDIT_AND_SEND as soon as the reviewer types a manual reply without a Gemini draft", () => {
    const flags = composerActionFlags(["EDIT_AND_SEND", "NO_RESPONSE_NEEDED"], null, "رد يدوي");
    expect(flags.canSendAsIs).toBe(false);
    expect(flags.canEditSend).toBe(true);
  });

  it("does not treat a stale original-style copy as the gate: empty suggestion + non-empty draft is dirty", () => {
    const flags = composerActionFlags(["EDIT_AND_SEND", "NO_RESPONSE_NEEDED"], "", "  رد  ");
    expect(flags.dirty).toBe(true);
    expect(flags.canEditSend).toBe(true);
  });

  it("keeps SEND_AS_IS for an unchanged Gemini draft and EDIT_AND_SEND after an edit", () => {
    const draft = "شكرًا لتواصلكم معنا.";
    const unchanged = composerActionFlags(["SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED"], draft, draft);
    expect(unchanged.canSendAsIs).toBe(true);
    expect(unchanged.canEditSend).toBe(false);
    const edited = composerActionFlags(["SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED"], draft, draft + " تعديل");
    expect(edited.canSendAsIs).toBe(false);
    expect(edited.canEditSend).toBe(true);
    const restored = composerActionFlags(["SEND_AS_IS", "EDIT_AND_SEND", "NO_RESPONSE_NEEDED"], draft, draft);
    expect(restored.canSendAsIs).toBe(true);
    expect(restored.canEditSend).toBe(false);
  });

  it("allows typing on DRAFT_READY / REVIEW_ONLY / IN_REVIEW only", () => {
    expect(isComposerEditable("REVIEW_ONLY")).toBe(true);
    expect(isComposerEditable("DRAFT_READY")).toBe(true);
    expect(isComposerEditable("IN_REVIEW")).toBe(true);
    expect(isComposerEditable("NO_RESPONSE_RECOMMENDED")).toBe(true);
    expect(isComposerEditable("NO_REPLY")).toBe(false);
    expect(isComposerEditable("EDITED_AND_APPROVED")).toBe(false);
  });
});
