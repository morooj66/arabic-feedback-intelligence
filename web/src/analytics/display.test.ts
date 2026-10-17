import { describe, expect, it } from "vitest";
import {
  arabicClassLabel,
  canonicalChartRows,
  canonicalClassCount,
  CANONICAL_INTENT_KEYS,
  CANONICAL_TOPIC_KEYS,
  classHover,
  outcomeHover,
  trendHover,
} from "./display";

describe("analytics display", () => {
  it("covers the canonical 15 classes and no legacy topic", () => {
    expect(canonicalClassCount()).toBe(15);
    expect(CANONICAL_INTENT_KEYS).toHaveLength(5);
    expect(CANONICAL_TOPIC_KEYS).toHaveLength(7);
    expect(CANONICAL_TOPIC_KEYS).not.toContain("WORD_OF_YEAR");
    expect(arabicClassLabel("topic", "WORD_OF_YEAR_NOMINATION")).toBe("ترشيح كلمة العام");
    expect(arabicClassLabel("intent", "SPAM")).toBe("غير ذي صلة");
    expect(arabicClassLabel("sentiment", "NEUTRAL")).toBe("محايد");
  });

  it("drops WORD_OF_YEAR from the topic chart without merging its count", () => {
    const rows = canonicalChartRows("topic", [
      { key: "GENERAL", count: 2 },
      { key: "WORD_OF_YEAR", count: 9 },
      { key: "WORD_OF_YEAR_NOMINATION", count: 1 },
    ]);
    expect(rows.map((row) => row.key)).toEqual([...CANONICAL_TOPIC_KEYS]);
    expect(rows).toHaveLength(7);
    expect(rows.some((row) => row.key === "WORD_OF_YEAR")).toBe(false);
    expect(rows.find((row) => row.key === "WORD_OF_YEAR_NOMINATION")?.count).toBe(1);
    expect(rows.find((row) => row.key === "TECHNICAL_ISSUE")?.count).toBe(0);
    expect(rows.map((row) => arabicClassLabel("topic", row.key))).toEqual([
      "اقتراح مفردة",
      "تصحيح مدخل",
      "اقتراح معجمي",
      "مشكلة تقنية",
      "ترشيح كلمة العام",
      "خارج الموضوع",
      "عام",
    ]);
  });

  it("maps hover payloads from real counts", () => {
    expect(outcomeHover("تعديل ثم إرسال", 1, 3)).toBe("تعديل ثم إرسال\n1 تعليقات\n33٪");
    expect(classHover("شكر وثناء", 0, 6)).toBe("شكر وثناء\n0\n0٪");
    expect(trendHover("٢٦ أغسطس", 5, 3)).toContain("التعليقات الواردة: 5");
    expect(trendHover("٢٦ أغسطس", 5, 3)).toContain("تمت مراجعتها: 3");
  });
});
