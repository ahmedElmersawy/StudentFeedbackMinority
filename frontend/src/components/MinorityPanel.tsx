import { useMemo } from "react";
import type { AnalysisResult, ResultRow } from "../types";

interface Props {
  result: AnalysisResult;
  onDownloadCsv?: () => void;
}

const CATEGORY_COLORS: Record<string, string> = {
  Mental_Health: "#fb7185",
  Financial_Hardship: "#f97316",
  International_Student: "#a78bfa",
  First_Generation: "#34d399",
  Disability_Accessibility: "#38bdf8",
  Racial_Ethnic_Minority: "#fbbf24",
  Gender_Identity: "#e879f9",
  International_Academic_Background: "#6ee7b7",
  Caregiver: "#f472b6",
  Statistical_Outlier_Only: "#64748b",
};

function categoryColor(cat: string): string {
  return CATEGORY_COLORS[cat] ?? "#818cf8";
}

function downloadCsv(rows: ResultRow[], filename: string) {
  const cols = ["text", "prediction", "confidence", "minority_category", "is_outlier", "is_minority_cluster", "cluster_id"];
  const header = cols.join(",");
  const lines = rows.map((r) =>
    cols.map((c) => {
      const v = String(r[c] ?? "").replace(/"/g, '""');
      return `"${v}"`;
    }).join(",")
  );
  const blob = new Blob([header + "\n" + lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
}

export function MinorityPanel({ result, onDownloadCsv }: Props) {
  const minorityRows = useMemo(
    () => result.rows.filter((r) => Boolean(r.is_minority_pattern)),
    [result.rows],
  );

  const catBreakdown = result.summary.minority_category_breakdown ?? {};
  const catEntries = Object.entries(catBreakdown).sort((a, b) => b[1] - a[1]);
  const total = result.summary.total;

  return (
    <div className="space-y-5">
      {/* Summary */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: "Minority rows", value: result.summary.minority, accent: "bg-rose-500" },
          { label: "% of dataset", value: `${result.summary.minority_pct}%`, accent: "bg-amber-400" },
          { label: "Outliers (IsoForest)", value: result.rows.filter((r) => r.is_outlier).length, accent: "bg-indigo-400" },
          { label: "Small clusters", value: result.rows.filter((r) => r.is_minority_cluster).length, accent: "bg-teal-400" },
        ].map((kpi) => (
          <div key={kpi.label} className="relative overflow-hidden rounded-xl border border-white/[0.08] bg-[#141928] px-4 py-3">
            <div className={`absolute inset-x-0 top-0 h-0.5 ${kpi.accent}`} />
            <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-slate-500">{kpi.label}</p>
            <p className="mt-1 text-xl font-bold tabular-nums text-slate-100">{kpi.value}</p>
          </div>
        ))}
      </div>

      {/* Category breakdown chart */}
      {catEntries.length > 0 && (
        <div className="rounded-2xl border border-white/[0.08] bg-[#141928] p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
              Experiential Category Breakdown
            </h3>
            <span className="text-xs text-slate-600">{minorityRows.length} flagged rows</span>
          </div>
          <div className="space-y-2.5">
            {catEntries.map(([cat, count]) => {
              const pct = Math.round((count / Math.max(1, minorityRows.length)) * 100);
              const color = categoryColor(cat);
              return (
                <div key={cat}>
                  <div className="mb-1 flex justify-between text-xs">
                    <span className="text-slate-300">{cat.replace(/_/g, " ")}</span>
                    <span className="font-semibold" style={{ color }}>{count} · {pct}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-white/[0.06]">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${pct}%`, background: color }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Minority row list */}
      <div className="rounded-2xl border border-white/[0.08] bg-[#141928] p-5">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
            Flagged Feedback
          </h3>
          <button
            onClick={() => onDownloadCsv ? onDownloadCsv() : downloadCsv(minorityRows, "minority_patterns.csv")}
            className="rounded-lg border border-white/[0.1] px-3 py-1 text-xs text-slate-400 hover:text-slate-200"
          >
            Download CSV
          </button>
        </div>

        <ul className="space-y-3 max-h-[520px] overflow-y-auto pr-1">
          {minorityRows.slice(0, 100).map((row, i) => {
            const text = ((row.text ?? row.feedback ?? "") as string);
            const cats = (row.minority_category as string | undefined) ?? "";
            const isOutlier = Boolean(row.is_outlier);
            const isCluster = Boolean(row.is_minority_cluster);

            return (
              <li
                key={i}
                className="rounded-xl border border-amber-500/20 bg-amber-500/[0.05] p-3 text-sm"
              >
                <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
                  {isOutlier && (
                    <span className="rounded-full bg-rose-500/20 px-2 py-0.5 text-[10px] font-semibold text-rose-300">
                      outlier
                    </span>
                  )}
                  {isCluster && (
                    <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[10px] font-semibold text-amber-300">
                      small cluster
                    </span>
                  )}
                  {cats.split("|").filter(Boolean).map((cat) => (
                    <span
                      key={cat}
                      className="rounded-full px-2 py-0.5 text-[10px] font-semibold"
                      style={{ background: `${categoryColor(cat)}20`, color: categoryColor(cat) }}
                    >
                      {cat.replace(/_/g, " ")}
                    </span>
                  ))}
                  {row.prediction && (
                    <span className="ml-auto text-[10px] text-slate-500">{row.prediction as string}</span>
                  )}
                </div>
                <p className="line-clamp-3 text-slate-400 text-xs leading-relaxed">{text}</p>
              </li>
            );
          })}
          {minorityRows.length === 0 && (
            <li className="py-6 text-center text-sm text-slate-600">No minority patterns detected.</li>
          )}
          {minorityRows.length > 100 && (
            <li className="py-3 text-center text-xs text-slate-600">
              Showing first 100 of {minorityRows.length}. Download CSV for full list.
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}
