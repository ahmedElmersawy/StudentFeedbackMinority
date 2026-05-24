import { useMemo, useState } from "react";
import type { AnalysisResult, ResultRow } from "../types";

interface Props {
  result: AnalysisResult;
}

interface ReviewState {
  [index: number]: "approved" | "rejected" | null;
}

function downloadCsv(rows: ResultRow[], corrections: ReviewState, filename: string) {
  const cols = ["text", "prediction", "confidence", "human_label"];
  const header = cols.join(",");
  const lines = rows.map((r, i) => {
    const humanLabel = corrections[i] === "approved"
      ? r.prediction
      : corrections[i] === "rejected"
      ? "REJECTED"
      : "";
    return ["text", "prediction", "confidence", "human_label"].map((c) => {
      const v = c === "human_label" ? humanLabel : String(r[c] ?? "");
      return `"${v.replace(/"/g, '""')}"`;
    }).join(",");
  });
  const blob = new Blob([header + "\n" + lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
}

function confColor(c: number) {
  if (c >= 0.65) return "text-amber-400";
  return "text-rose-400";
}

export function ReviewQueue({ result }: Props) {
  const reviewRows = useMemo(
    () => result.rows.filter((r) => Boolean(r.needs_review)),
    [result.rows],
  );
  const [corrections, setCorrections] = useState<ReviewState>({});

  const reviewed = Object.keys(corrections).length;
  const approved = Object.values(corrections).filter((v) => v === "approved").length;
  const rejected = Object.values(corrections).filter((v) => v === "rejected").length;

  const setLabel = (index: number, label: "approved" | "rejected" | null) => {
    setCorrections((prev) => ({ ...prev, [index]: label }));
  };

  if (reviewRows.length === 0) {
    return (
      <div className="rounded-2xl border border-white/[0.08] bg-[#141928] p-8 text-center">
        <p className="text-sm text-slate-500">
          No rows flagged for review — all predictions are above the confidence threshold.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Stats */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: "Needs review", value: reviewRows.length, accent: "bg-amber-400" },
          { label: "% of dataset", value: `${result.summary.review_pct}%`, accent: "bg-amber-400" },
          { label: "Reviewed", value: `${reviewed}/${reviewRows.length}`, accent: "bg-indigo-400" },
          { label: "Corrections", value: rejected, accent: "bg-rose-400" },
        ].map((kpi) => (
          <div key={kpi.label} className="relative overflow-hidden rounded-xl border border-white/[0.08] bg-[#141928] px-4 py-3">
            <div className={`absolute inset-x-0 top-0 h-0.5 ${kpi.accent}`} />
            <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-slate-500">{kpi.label}</p>
            <p className="mt-1 text-xl font-bold text-slate-100">{kpi.value}</p>
          </div>
        ))}
      </div>

      {/* Info */}
      <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3 text-xs text-slate-500">
        Predictions below the confidence threshold ({result.summary.avg_confidence > 0 ? "0.65" : "0.65"}) are queued here for
        human review. Use the thumbs up/down buttons to accept or correct labels. Export the queue with corrections for retraining.
      </div>

      {/* Review list */}
      <div className="rounded-2xl border border-white/[0.08] bg-[#141928] p-5">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
            Low-Confidence Predictions
          </h3>
          <button
            onClick={() => downloadCsv(reviewRows, corrections, "review_queue.csv")}
            className="rounded-lg border border-white/[0.1] px-3 py-1 text-xs text-slate-400 hover:text-slate-200"
          >
            Export queue CSV
          </button>
        </div>

        {/* Progress bar */}
        {reviewRows.length > 0 && (
          <div className="mb-4 space-y-1">
            <div className="flex justify-between text-[10px] text-slate-600">
              <span>{reviewed} reviewed</span>
              <span>{reviewRows.length - reviewed} remaining</span>
            </div>
            <div className="h-1 overflow-hidden rounded-full bg-white/[0.06]">
              <div
                className="h-full rounded-full bg-indigo-500 transition-all duration-300"
                style={{ width: `${Math.round((reviewed / Math.max(1, reviewRows.length)) * 100)}%` }}
              />
            </div>
          </div>
        )}

        <ul className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
          {reviewRows.map((row, i) => {
            const text = ((row.text ?? row.feedback ?? "") as string);
            const conf = typeof row.confidence === "number" ? row.confidence : null;
            const pred = (row.prediction ?? "") as string;
            const state = corrections[i] ?? null;

            return (
              <li
                key={i}
                className={`rounded-xl border p-4 transition-colors ${
                  state === "approved"
                    ? "border-emerald-500/30 bg-emerald-500/[0.05]"
                    : state === "rejected"
                    ? "border-rose-500/30 bg-rose-500/[0.05]"
                    : "border-white/[0.07] bg-white/[0.02]"
                }`}
              >
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  {pred && (
                    <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] font-semibold text-amber-300">
                      {pred}
                    </span>
                  )}
                  {conf != null && (
                    <span className={`text-xs font-medium ${confColor(conf)}`}>
                      {(conf * 100).toFixed(1)}% confidence
                    </span>
                  )}
                  <span className="ml-auto text-[10px] text-slate-600">#{i + 1}</span>
                </div>

                <p className="mb-3 line-clamp-3 text-xs leading-relaxed text-slate-400">{text}</p>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setLabel(i, state === "approved" ? null : "approved")}
                    className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors ${
                      state === "approved"
                        ? "bg-emerald-500 text-white"
                        : "border border-white/[0.1] text-slate-500 hover:border-emerald-500/50 hover:text-emerald-300"
                    }`}
                  >
                    <span>👍</span> Correct
                  </button>
                  <button
                    onClick={() => setLabel(i, state === "rejected" ? null : "rejected")}
                    className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors ${
                      state === "rejected"
                        ? "bg-rose-500 text-white"
                        : "border border-white/[0.1] text-slate-500 hover:border-rose-500/50 hover:text-rose-300"
                    }`}
                  >
                    <span>👎</span> Wrong
                  </button>
                  {state && (
                    <span className={`ml-auto text-[11px] font-semibold ${state === "approved" ? "text-emerald-400" : "text-rose-400"}`}>
                      {state === "approved" ? "✓ Accepted" : "✗ Flagged for correction"}
                    </span>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      {/* Summary actions */}
      {reviewed > 0 && (
        <div className="flex items-center justify-between rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3">
          <span className="text-xs text-slate-500">
            {approved} accepted · {rejected} flagged as wrong
          </span>
          <button
            onClick={() => downloadCsv(reviewRows, corrections, "review_corrections.csv")}
            className="rounded-lg bg-indigo-500 px-4 py-1.5 text-xs font-semibold text-white hover:bg-indigo-400"
          >
            Export corrections for retraining
          </button>
        </div>
      )}
    </div>
  );
}
