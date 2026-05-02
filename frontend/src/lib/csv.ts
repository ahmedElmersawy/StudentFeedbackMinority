import Papa from "papaparse";
import type { ParsedDataset } from "../types";
import { suggestDateColumn, suggestFeedbackColumns, suggestRatingColumn } from "./autoColumns";

export function parseCsvFile(file: File): Promise<ParsedDataset> {
  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (res) => {
        const rows = res.data as Record<string, string>[];
        const headers = res.meta.fields ?? Object.keys(rows[0] ?? {});
        if (!headers.length) {
          reject(new Error("CSV has no headers."));
          return;
        }
        resolve({
          headers,
          rows,
          suggestedFeedbackColumns: suggestFeedbackColumns(headers, rows),
          suggestedRatingColumn: suggestRatingColumn(headers),
          suggestedDateColumn: suggestDateColumn(headers),
        });
      },
      error: (err) => reject(err),
    });
  });
}

export function validateFeedbackColumns(
  headers: string[],
  selected: string[],
): string | null {
  if (!selected.length) return "Select at least one feedback column.";
  for (const c of selected) {
    if (!headers.includes(c)) return `Unknown column: ${c}`;
  }
  return null;
}
