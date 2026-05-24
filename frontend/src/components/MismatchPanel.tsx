import { useMemo } from "react";
import type { AnalysisResult, ResultRow } from "../types";

interface Props {
  result: AnalysisResult;
}

function downloadCsv(rows: ResultRow[], filename: string) {
  const cols = ["text", "prediction", "confidence", "mismatch_type"];
  const header = cols.join(",");
  const lines = rows.map((r) =>
    cols.map((c) => `"${String(r[c] ?? "").replace(/"/g, '""')}"`).join(",")
  );
  const blob = new Blob([header + "\n" + lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
}

function MismatchCard({ row, index }: { row: ResultRow; index: number }) {
  const text = ((row.text ?? row.feedback ?? "") as string);
  const isHigh = row.mismatch_type === "HIGH_MISMATCH";
  const conf = typeof row.confidence === "number" ? row.confidence : null;

  return (
    <div
      className={`rounded-xl border p-4 ${
        isHigh
          ? "border-orange-500/25 bg-orange-500/[0.06]"
          : "border-purple-500/25 bg-purple-500/[0.06]"
      }`}
    >
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span
          className={`rounded-full px-2 py-0.5 text-[11px] font-bold ${
            isHigh
              ? "bg-orange-500/20 text-orange-300"
              : "bg-purple-500/20 text-purple-300"
          }`}
        >
          {row.mismatch_type as string}
        </span>
        {row.prediction && (
          <span className="text-xs text-slate-500">
            Predicted: <span className="text-slate-300">{row.prediction as string}</span>
          </span>
        )}
        {conf != null && (
          <span className="text-xs text-slate-500">
            Confidence: <span className="text-slate-300">{(conf * 100).toFixed(1)}%</span>
          </span>
        )}
        <span className="ml-auto text-[10px] text-slate-600">#{index + 1}</span>
      </div>
      <p className="line-clamp-3 text-xs leading-relaxed text-slate-400">{text}</p>
    </div>
  );
}

export function MismatchPanel({ result }: Props) {
  const allMismatches = useMemo(
    () => result.rows.filter((r) => Boolean(r.mismatch_flag)),
    [result.rows],
  );
  const highMismatches = allMismatches.filter((r) => r.mismatch_type === "HIGH_MISMATCH");
  const reverseMismatches = allMismatches.filter((r) => r.mismatch_type === "REVERSE_MISMATCH");

  if (allMismatches.length === 0) {
    return (
      <div className="rounded-2xl border border-white/[0.08] bg-[#141928] p-8 text-center">
        <p className="text-sm text-slate-500">
          No mismatches detected.
          {result.summary.total > 0 && " (This dataset may not have numeric rating columns.)"}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* KPIs */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="relative overflow-hidden rounded-xl border border-white/[0.08] bg-[#141928] px-4 py-3">
          <div className="absolute inset-x-0 top-0 h-0.5 bg-orange-400" />
          <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-slate-500">Total Mismatches</p>
          <p className="mt-1 text-xl font-bold text-slate-100">{allMismatches.length}</p>
          <p className="mt-0.5 text-[11px] text-slate-500">
            {result.summary.mismatch_pct}% of dataset
          </p>
        </div>
        <div className="relative overflow-hidden rounded-xl border border-white/[0.08] bg-[#141928] px-4 py-3">
          <div className="absolute inset-x-0 top-0 h-0.5 bg-orange-400" />
          <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-slate-500">High Mismatch</p>
          <p className="mt-1 text-xl font-bold text-slate-100">{highMismatches.length}</p>
          <p className="mt-0.5 text-[11px] text-slate-500">High rating + negative text</p>
        </div>
        <div className="relative overflow-hidden rounded-xl border border-white/[0.08] bg-[#141928] px-4 py-3">
          <div className="absolute inset-x-0 top-0 h-0.5 bg-purple-400" />
          <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-slate-500">Reverse Mismatch</p>
          <p className="mt-1 text-xl font-bold text-slate-100">{reverseMismatches.length}</p>
          <p className="mt-0.5 text-[11px] text-slate-500">Low rating + positive text</p>
        </div>
      </div>

      {/* Explanation */}
      <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3 text-xs text-slate-500">
        <p>
          <strong className="text-orange-300">HIGH_MISMATCH</strong> — rating ≥ 3.8 but text
          classified as Negative or Neutral. Student may be masking frustration in ratings.
        </p>
        <p className="mt-1">
          <strong className="text-purple-300">REVERSE_MISMATCH</strong> — rating ≤ 2.5 but text
          classified as Positive. Student may be hedging on the Likert scale.
        </p>
        <p className="mt-1 text-slate-600">These rows are often the most actionable feedback for instructors.</p>
      </div>

      {/* Lists */}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-white/[0.08] bg-[#141928] p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
              High Mismatch
            </h3>
            <button
              onClick={() => downloadCsv(highMismatches, "high_mismatch.csv")}
              className="rounded-lg border border-white/[0.1] px-2 py-1 text-[10px] text-slate-500 hover:text-slate-300"
            >
              CSV
            </button>
          </div>
          <div className="space-y-3 max-h-[480px] overflow-y-auto pr-1">
            {highMismatches.slice(0, 50).map((row, i) => (
              <MismatchCard key={i} row={row} index={i} />
            ))}
            {highMismatches.length === 0 && (
              <p className="text-sm text-slate-600">None detected.</p>
            )}
          </div>
        </div>

        <div className="rounded-2xl border border-white/[0.08] bg-[#141928] p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
              Reverse Mismatch
            </h3>
            <button
              onClick={() => downloadCsv(reverseMismatches, "reverse_mismatch.csv")}
              className="rounded-lg border border-white/[0.1] px-2 py-1 text-[10px] text-slate-500 hover:text-slate-300"
            >
              CSV
            </button>
          </div>
          <div className="space-y-3 max-h-[480px] overflow-y-auto pr-1">
            {reverseMismatches.slice(0, 50).map((row, i) => (
              <MismatchCard key={i} row={row} index={i} />
            ))}
            {reverseMismatches.length === 0 && (
              <p className="text-sm text-slate-600">None detected.</p>
            )}
          </div>
        </div>
      </div>

      {/* Download all */}
      <div className="flex justify-end">
        <button
          onClick={() => downloadCsv(allMismatches, "all_mismatches.csv")}
          className="rounded-xl border border-white/[0.1] px-4 py-2 text-sm font-medium text-slate-400 hover:text-slate-200"
        >
          Download all mismatches as CSV
        </button>
      </div>
    </div>
  );
}
