import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { PlaygroundPage } from "./PlaygroundPage";

vi.mock("../api/client", () => ({
  api: {
    playgroundStatus: vi.fn(),
    playgroundRun: vi.fn(),
    playgroundCompare: vi.fn(),
    playgroundSaveCase: vi.fn(),
    playgroundCases: vi.fn(),
    playgroundDeleteCase: vi.fn(),
    playgroundMetrics: vi.fn(),
  },
}));

const TAXONOMY = {
  intents: ["THANKS_PRAISE", "COMPLAINT", "SUGGESTION", "INQUIRY", "SPAM"],
  topics: ["WORD_SUBMISSION", "ENTRY_CORRECTION", "DICTIONARY_SUGGESTION", "TECHNICAL_ISSUE", "WORD_OF_YEAR_NOMINATION", "OFF_TOPIC_PERSONAL", "GENERAL"],
  sentiments: ["POSITIVE", "NEGATIVE", "NEUTRAL"],
  routing_decisions: ["AUTO_REPLY_CANDIDATE", "ACKNOWLEDGE_THEN_REVIEW", "REVIEW_ONLY", "EXCLUDE"],
  response_decisions: ["GENERATE_DRAFT", "REVIEW_ONLY", "NO_RESPONSE_RECOMMENDED"],
};
const enabledStatus = { enabled: true, limits: { max_variants: 10, max_chars: 1000 }, live_configured: true, taxonomy: TAXONOMY, parity_notes: ["note-1"] };

function runResult(withAgent = false) {
  return {
    run_id: "run-1", mode: withAgent ? "full_pipeline" : "classification_only", agent_mode: withAgent ? "mock" : null,
    mode_banner: withAgent ? "MOCK – scripted placeholder text, NOT Gemini output. Says nothing about real response quality." : null,
    batch_size: 1, total_ms: 5, classification_timings_ms: { total: 3 },
    variants: [{
      index: 0, text_length: 5,
      preprocessing: { normalized_text: "x", changed: false, prefilter_reason: null, production_outcome: "CLASSIFIED" },
      classification: { intent: { label: "SUGGESTION", confidence: 0.8 }, topic: { label: "GENERAL", confidence: 0.7 }, sentiment: { label: "NEUTRAL", confidence: 0.9 } },
      routing: { decision: "ACKNOWLEDGE_THEN_REVIEW", reason_code: null },
      eligibility_preview: { routing_decision: "GENERATE_DRAFT", rule_id: "generate_suggestion" },
      agent_runs: withAgent ? { predicted: { label_source: "predicted", agent_mode: "mock", mode_banner: "MOCK banner", decision: { routing_decision: "GENERATE_DRAFT", generation_status: "SUCCEEDED", reason_code: "r", eligibility_rule_id: "generate_suggestion" }, tool_calls: [], drafts: null, response: "رد تجريبي", llm_calls: 1, execution_time_ms: 2, fallback: null, error: null, classification_routing_decision: "ACKNOWLEDGE_THEN_REVIEW" } } : {},
      timings_ms: { preprocess: 0.1 }, errors: [],
    }],
  };
}

async function openEnabled() {
  vi.mocked(api.playgroundStatus).mockResolvedValue(enabledStatus);
  render(<PlaygroundPage />);
  await screen.findByRole("heading", { name: /ساحة التقييم التفاعلية/ });
}

beforeEach(() => {
  vi.resetAllMocks();
});

describe("Playground page", () => {
  it("shows a disabled notice and exposes no run controls when the server has it disabled", async () => {
    vi.mocked(api.playgroundStatus).mockResolvedValue({ enabled: false, disabled_reason: "RCS_PLAYGROUND_ENABLED_not_set", taxonomy: null });
    render(<PlaygroundPage />);
    expect(await screen.findByTestId("playground-disabled")).toHaveTextContent("RCS_PLAYGROUND_ENABLED=1");
    expect(screen.queryByRole("button", { name: "تشغيل" })).not.toBeInTheDocument();
    expect(api.playgroundRun).not.toHaveBeenCalled();
  });

  it("explore workflow runs predictions without any expected labels, verdicts or metrics calls", async () => {
    await openEnabled();
    vi.mocked(api.playgroundRun).mockResolvedValue(runResult());
    await userEvent.type(screen.getByLabelText("comment-1"), "التطبيق ممتاز");
    await userEvent.click(screen.getByRole("button", { name: "تشغيل" }));
    const card = await screen.findByTestId("variant-result-0");
    expect(within(card).getByText("SUGGESTION")).toBeInTheDocument();
    expect(within(card).getByText(/80\.0%/)).toBeInTheDocument();
    expect(screen.queryByLabelText("expected-intent-1")).not.toBeInTheDocument();
    expect(screen.queryByTestId("verdict-0")).not.toBeInTheDocument();
    expect(api.playgroundCompare).not.toHaveBeenCalled();
    expect(api.playgroundMetrics).not.toHaveBeenCalled();
    expect(vi.mocked(api.playgroundRun).mock.calls[0][0]).toMatchObject({ mode: "classification_only", agent_mode: "mock", live_confirmed: false });
  });

  it("defaults the Agent to MOCK and labels mock output as not Gemini", async () => {
    await openEnabled();
    vi.mocked(api.playgroundRun).mockResolvedValue(runResult(true));
    await userEvent.type(screen.getByLabelText("comment-1"), "التطبيق ممتاز");
    await userEvent.click(screen.getByLabelText(/Full Pipeline/));
    expect(screen.getByLabelText(/MOCK/)).toBeChecked();
    expect(screen.queryByTestId("live-warning")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "تشغيل" }));
    await screen.findByTestId("variant-result-0");
    expect(screen.getAllByText(/NOT Gemini output/).length).toBeGreaterThan(0);
    expect(screen.getByText(/ليس مخرجات Gemini/)).toBeInTheDocument();
    expect(vi.mocked(api.playgroundRun).mock.calls[0][0]).toMatchObject({ mode: "full_pipeline", agent_mode: "mock", live_confirmed: false });
  });

  it("live mode requires an explicit cost/privacy confirmation before the run button enables", async () => {
    await openEnabled();
    await userEvent.type(screen.getByLabelText("comment-1"), "التطبيق ممتاز");
    await userEvent.click(screen.getByLabelText(/Full Pipeline/));
    await userEvent.click(screen.getByLabelText(/LIVE Gemini/));
    expect(screen.getByTestId("live-warning")).toHaveTextContent("التكلفة والخصوصية");
    expect(screen.getByRole("button", { name: "تشغيل" })).toBeDisabled();
    await userEvent.click(screen.getByLabelText(/أفهم التكلفة/));
    expect(screen.getByRole("button", { name: "تشغيل" })).toBeEnabled();
    vi.mocked(api.playgroundRun).mockResolvedValue({ ...runResult(true), agent_mode: "live" });
    await userEvent.click(screen.getByRole("button", { name: "تشغيل" }));
    await waitFor(() => expect(api.playgroundRun).toHaveBeenCalled());
    expect(vi.mocked(api.playgroundRun).mock.calls[0][0]).toMatchObject({ agent_mode: "live", live_confirmed: true });
  });

  it("switching back to mock clears the live confirmation", async () => {
    await openEnabled();
    await userEvent.click(screen.getByLabelText(/Full Pipeline/));
    await userEvent.click(screen.getByLabelText(/LIVE Gemini/));
    await userEvent.click(screen.getByLabelText(/أفهم التكلفة/));
    await userEvent.click(screen.getByLabelText(/MOCK \(/));
    await userEvent.click(screen.getByLabelText(/LIVE Gemini/));
    expect(screen.getByLabelText(/أفهم التكلفة/)).not.toBeChecked();
  });

  it("enforces the variant limit and can load the example variations", async () => {
    await openEnabled();
    await userEvent.click(screen.getByRole("button", { name: "تحميل أمثلة الصيغ" }));
    expect(screen.getByLabelText("comment-6")).toHaveValue("Good application");
    for (let i = 0; i < 4; i++) await userEvent.click(screen.getByRole("button", { name: /إضافة صيغة/ }));
    expect(screen.getByRole("button", { name: /إضافة صيغة/ })).toBeDisabled();
  });

  it("evaluate workflow: only explicitly VERIFIED expected labels are compared; actual and expected are visibly distinct", async () => {
    await openEnabled();
    await userEvent.click(screen.getByRole("tab", { name: /تقييم/ }));
    vi.mocked(api.playgroundRun).mockResolvedValue(runResult());
    vi.mocked(api.playgroundCompare).mockResolvedValue({
      verdict: "FAIL", status: "VERIFIED",
      fields: { intent: { expected: "INQUIRY", actual: "SUGGESTION", evaluated: true, match: false } },
    });
    await userEvent.type(screen.getByLabelText("comment-1"), "هل يوجد بحث؟");
    await userEvent.selectOptions(screen.getByLabelText("expected-intent-1"), "INQUIRY");
    await userEvent.selectOptions(screen.getByLabelText("status-1"), "VERIFIED");
    await userEvent.click(screen.getByRole("button", { name: "تشغيل" }));
    expect(await screen.findByTestId("verdict-0")).toHaveTextContent("FAIL");
    const table = screen.getByLabelText("expected-vs-actual");
    expect(within(table).getByText("المتوقع (يدوي)")).toBeInTheDocument();
    expect(within(table).getByText("الفعلي (النموذج)")).toBeInTheDocument();
    expect(vi.mocked(api.playgroundCompare).mock.calls[0][0]).toMatchObject({ status: "VERIFIED", expected: { intent: "INQUIRY" }, run_id: "run-1", variant_index: 0 });
  });

  it("evaluate workflow never compares a variant with no expected labels and UNVERIFIED status", async () => {
    await openEnabled();
    await userEvent.click(screen.getByRole("tab", { name: /تقييم/ }));
    vi.mocked(api.playgroundRun).mockResolvedValue(runResult());
    await userEvent.type(screen.getByLabelText("comment-1"), "التطبيق ممتاز");
    await userEvent.click(screen.getByRole("button", { name: "تشغيل" }));
    await screen.findByTestId("variant-result-0");
    expect(api.playgroundCompare).not.toHaveBeenCalled();
    expect(screen.queryByTestId("verdict-0")).not.toBeInTheDocument();
  });

  it("saving is explicit: nothing is saved by running, and the save button sends the server run reference only", async () => {
    await openEnabled();
    await userEvent.click(screen.getByRole("tab", { name: /تقييم/ }));
    vi.mocked(api.playgroundRun).mockResolvedValue(runResult());
    vi.mocked(api.playgroundSaveCase).mockResolvedValue({ case: { case_id: "pg-abc" } });
    await userEvent.type(screen.getByLabelText("comment-1"), "التطبيق ممتاز");
    await userEvent.selectOptions(screen.getByLabelText("status-1"), "NEEDS_HUMAN_REVIEW");
    await userEvent.click(screen.getByRole("button", { name: "تشغيل" }));
    await screen.findByTestId("variant-result-0");
    expect(api.playgroundSaveCase).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "حفظ الحالة" }));
    expect(vi.mocked(api.playgroundSaveCase).mock.calls[0][0]).toMatchObject({ run_id: "run-1", variant_index: 0, status: "NEEDS_HUMAN_REVIEW" });
    expect(vi.mocked(api.playgroundSaveCase).mock.calls[0][0]).not.toHaveProperty("result");
    expect(await screen.findByText(/pg-abc/)).toBeInTheDocument();
  });

  it("when a classifier fails the page shows the failing stage and the skipped Agent, never an Agent result", async () => {
    await openEnabled();
    const base = runResult(true);
    const v = base.variants[0] as Record<string, any>;
    v.classification.sentiment = null;
    v.agent_runs = {};
    v.pipeline_status = { classification_complete: false, failed_stages: ["classification.sentiment"], production_equivalent: false, note: "partial" };
    v.agent_skipped = { label_source: "predicted", reason: "classification_incomplete", failed_stages: ["classification.sentiment"], detail: "Agent not run because classification failed at classification.sentiment. First diagnostic: classification.sentiment: OSError - weights missing", production_equivalent: false };
    v.errors = [{ stage: "classification.sentiment", type: "OSError", message: "weights missing" }];
    vi.mocked(api.playgroundRun).mockResolvedValue(base);
    await userEvent.type(screen.getByLabelText("comment-1"), "التطبيق ممتاز");
    await userEvent.click(screen.getByLabelText(/Full Pipeline/));
    await userEvent.click(screen.getByRole("button", { name: "تشغيل" }));
    const card = await screen.findByTestId("variant-result-0");
    expect(within(card).getByTestId("partial-classification")).toHaveTextContent("classification.sentiment");
    expect(within(card).getByTestId("agent-skipped")).toHaveTextContent("OSError - weights missing");
    expect(within(card).getByText("SUGGESTION")).toBeInTheDocument(); // successful classifier output is preserved
    expect(within(card).queryByTestId("path-predicted")).not.toBeInTheDocument();
    expect(within(card).queryByText("رد تجريبي")).not.toBeInTheDocument();
  });

  it("labels fixed-label Agent runs as isolation tests and predicted-label runs as production-equivalent", async () => {
    await openEnabled();
    const base = runResult(true);
    const v = base.variants[0] as Record<string, any>;
    v.agent_runs.predicted.production_equivalent = true;
    v.agent_runs.verified = { ...v.agent_runs.predicted, label_source: "verified", production_equivalent: false };
    vi.mocked(api.playgroundRun).mockResolvedValue(base);
    await userEvent.type(screen.getByLabelText("comment-1"), "التطبيق ممتاز");
    await userEvent.click(screen.getByLabelText(/Full Pipeline/));
    await userEvent.click(screen.getByRole("button", { name: "تشغيل" }));
    const card = await screen.findByTestId("variant-result-0");
    expect(within(card).getByTestId("path-predicted")).toHaveTextContent("production-equivalent");
    expect(within(card).getByTestId("path-verified")).toHaveTextContent("ليس مسار الإنتاج");
  });

  it("results tab shows withheld aggregates and sample sizes instead of misleading scores", async () => {
    await openEnabled();
    const task = (n: number) => ({ n, supported_classes: 1, sufficient: false, withheld_reason: `only ${n} verified labeled case(s); need at least 10`, per_class: [], confusion_matrix: { labels: [], matrix: [] }, caveat: "c" });
    vi.mocked(api.playgroundMetrics).mockResolvedValue({
      case_counts: { total: 3, verified: 2, unverified: 1, needs_human_review: 0 }, notes: ["n"],
      classification: { intent: task(2), topic: task(0), sentiment: task(0) },
      routing: { n: 0, correct: 0, accuracy: null, withheld_reason: "only 0 verified routing case(s); need 10" },
      agent: { mock: { n: 0, correct: 0, note: "plumbing" }, live: { n: 0, correct: 0, note: "decision" } },
    });
    vi.mocked(api.playgroundCases).mockResolvedValue({ cases: [] });
    await userEvent.click(screen.getByRole("tab", { name: /النتائج/ }));
    expect(await screen.findByTestId("withheld-intent")).toHaveTextContent("need at least 10");
    expect(screen.queryByTestId("metrics-intent")).not.toBeInTheDocument();
    expect(screen.getByTestId("case-counts")).toHaveTextContent("موثّقة (Verified): 2");
  });
});
