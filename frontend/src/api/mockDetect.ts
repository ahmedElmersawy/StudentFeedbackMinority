/**
 * MOCK minority detection — deterministic placeholder for class demos.
 *
 * TODO: Replace with `fetch("/api/detect-minority", { method: "POST", ... })`
 * backed by your Python `feedback_core` + transformers pipeline.
 */

import type { DetectionResponse, DetectionRow, ParsedDataset, RowLabel, TrendPoint } from "../types";
import { detectTheme, topThemes } from "../lib/themes";
import { looksLikeDate } from "../lib/autoColumns";

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function labelRow(feedback: string): { label: RowLabel; confidence: number } {
  const h = hash(feedback) % 100;
  const lower = feedback.toLowerCase();
  const stressed =
    lower.includes("bad") ||
    lower.includes("poor") ||
    lower.includes("never") ||
    lower.includes("worst") ||
    lower.includes("!!");
  const long = feedback.length > 180;
  if (stressed && long && h < 55) return { label: "minority", confidence: 0.72 + (h % 20) / 100 };
  if (h < 8) return { label: "needs_review", confidence: 0.55 + (h % 15) / 100 };
  return { label: "majority", confidence: 0.78 + (h % 18) / 100 };
}

export async function runMockMinorityDetection(
  dataset: ParsedDataset,
  feedbackColumns: string[],
  ratingColumn: string | null,
  dateColumn: string | null,
): Promise<DetectionResponse> {
  await new Promise((r) => setTimeout(r, 600));

  const rows: DetectionRow[] = dataset.rows.map((r, i) => {
    const feedback = feedbackColumns.map((c) => String(r[c] ?? "")).join(" ").trim();
    const id =
      String(r["id"] ?? r["student_id"] ?? r["submission_id"] ?? r["ID"] ?? "").trim() ||
      `row-${i + 1}`;
    const { label, confidence } = labelRow(feedback);
    return {
      id,
      feedback: feedback.slice(0, 2000),
      label,
      confidence,
      theme: detectTheme(feedback),
    };
  });

  const majority = rows.filter((x) => x.label === "majority");
  const minority = rows.filter((x) => x.label === "minority");
  const needs = rows.filter((x) => x.label === "needs_review");

  let ratingByLabel: DetectionResponse["summary"]["ratingByLabel"];
  if (ratingColumn && dataset.headers.includes(ratingColumn)) {
    ratingByLabel = (["majority", "minority", "needs_review"] as const).map((lab) => {
      const nums: number[] = [];
      rows.forEach((row, i) => {
        if (row.label !== lab) return;
        const raw = dataset.rows[i]?.[ratingColumn];
        const n = parseFloat(String(raw ?? "").replace(",", "."));
        if (Number.isFinite(n)) nums.push(n);
      });
      const subset = rows.filter((x) => x.label === lab);
      const avg = nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
      return { label: lab, avgRating: Math.round(avg * 100) / 100, count: subset.length };
    });
  }

  let trend: TrendPoint[] | undefined;
  if (dateColumn && dataset.headers.includes(dateColumn)) {
    const byMonth = new Map<string, { minority: number; majority: number }>();
    rows.forEach((row, i) => {
      const raw = String(dataset.rows[i]?.[dateColumn] ?? "");
      if (!looksLikeDate(raw)) return;
      const d = new Date(raw);
      const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
      if (!byMonth.has(key)) byMonth.set(key, { minority: 0, majority: 0 });
      const b = byMonth.get(key)!;
      if (row.label === "minority") b.minority += 1;
      else b.majority += 1;
    });
    trend = [...byMonth.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([period, v]) => ({ period, minority: v.minority, majority: v.majority }));
    if (!trend.length) trend = undefined;
  }

  const total = rows.length;
  const minorityPct = total ? Math.round((minority.length / total) * 1000) / 10 : 0;

  return {
    rows,
    summary: {
      total,
      majority: majority.length,
      minority: minority.length,
      needsReview: needs.length,
      minorityPct,
      majorityThemes: topThemes(majority.map((x) => x.feedback)),
      minorityThemes: topThemes(minority.map((x) => x.feedback)),
      ratingByLabel,
      trend,
    },
  };
}
