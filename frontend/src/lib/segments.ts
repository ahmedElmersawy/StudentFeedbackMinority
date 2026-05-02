import type { ParsedDataset } from "../types";

const SEGMENT_PALETTE = ["#7c6cf5", "#2dd4bf", "#fbbf24", "#c084fc", "#fb7185", "#38bdf8", "#a3e635", "#f472b6"];

export function guessSegmentColumn(dataset: ParsedDataset, feedbackCols: string[]): string | null {
  let best: { h: string; score: number } | null = null;
  for (const h of dataset.headers) {
    if (feedbackCols.includes(h)) continue;
    const vals = new Set(
      dataset.rows.map((r) => String(r[h] ?? "").trim()).filter((s) => s.length > 0 && s.length < 80),
    );
    const n = vals.size;
    if (n < 2 || n > 18) continue;
    const score = n >= 3 && n <= 12 ? 80 - Math.abs(n - 6) : 40 - Math.abs(n - 8);
    if (!best || score > best.score) best = { h, score };
  }
  return best?.h ?? null;
}

export type SegmentSlice = { label: string; count: number; pct: number; color: string };

export function segmentBreakdown(
  dataset: ParsedDataset,
  column: string | null,
): SegmentSlice[] {
  if (!column || !dataset.headers.includes(column)) return [];
  const counts = new Map<string, number>();
  for (const r of dataset.rows) {
    const v = String(r[column] ?? "").trim() || "—";
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const total = dataset.rows.length || 1;
  return [...counts.entries()]
    .map(([label, count]) => ({
      label: label.length > 28 ? `${label.slice(0, 26)}…` : label,
      count,
      pct: Math.round((count / total) * 1000) / 10,
      color: "",
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8)
    .map((s, i) => ({ ...s, color: SEGMENT_PALETTE[i % SEGMENT_PALETTE.length] }));
}
