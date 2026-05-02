/**
 * Client-side heuristics to suggest feedback / rating / date columns.
 * Mirrors the spirit of Python `feedback_core.auto_detect_text_columns` for the SPA.
 */

const FEEDBACK_HINTS = [
  "feedback",
  "comment",
  "review",
  "text",
  "opinion",
  "response",
  "description",
  "teaching",
  "course",
  "exam",
  "lab",
  "library",
  "extra",
  "open",
];

const RATING_HINTS = ["rating", "score", "stars", "satisfaction", "overall"];

const DATE_HINTS = ["date", "time", "timestamp", "submitted", "created"];

function meanLen(values: string[]): number {
  const lens = values.map((s) => s.length).filter((n) => n > 0);
  if (!lens.length) return 0;
  return lens.reduce((a, b) => a + b, 0) / lens.length;
}

function scoreFeedbackColumn(header: string, samples: string[]): number {
  const h = header.toLowerCase();
  let s = meanLen(samples);
  if (s < 8) return 0;
  for (const kw of FEEDBACK_HINTS) {
    if (h.includes(kw)) s += 40;
  }
  if (h === "id" || h.endsWith("_id")) return 0;
  return s;
}

export function suggestFeedbackColumns(
  headers: string[],
  rows: Record<string, string>[],
  max = 10,
): string[] {
  const scored = headers.map((h) => {
    const samples = rows.slice(0, 200).map((r) => String(r[h] ?? ""));
    return { h, sc: scoreFeedbackColumn(h, samples) };
  });
  scored.sort((a, b) => b.sc - a.sc);
  return scored.filter((x) => x.sc > 0).map((x) => x.h).slice(0, max);
}

export function suggestRatingColumn(headers: string[]): string | null {
  for (const h of headers) {
    const low = h.toLowerCase();
    if (RATING_HINTS.some((k) => low.includes(k))) return h;
  }
  return null;
}

export function suggestDateColumn(headers: string[]): string | null {
  for (const h of headers) {
    const low = h.toLowerCase();
    if (DATE_HINTS.some((k) => low.includes(k))) return h;
  }
  return null;
}

export function looksLikeDate(value: string): boolean {
  const v = value.trim();
  if (!v) return false;
  const t = Date.parse(v);
  return !Number.isNaN(t);
}
