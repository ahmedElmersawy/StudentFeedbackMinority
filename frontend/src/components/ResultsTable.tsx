import type { DetectionRow } from "../types";

export function ResultsTable({ rows, variant = "light" }: { rows: DetectionRow[]; variant?: "light" | "dark" }) {
  const show = rows.slice(0, 150);
  const dark = variant === "dark";
  return (
    <div
      className={
        dark
          ? "overflow-hidden rounded-2xl border border-white/[0.08] bg-[#141928] shadow-[0_1px_0_rgba(255,255,255,0.04)_inset]"
          : "overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm"
      }
    >
      <div className={dark ? "border-b border-white/[0.06] bg-[#0f1117] px-4 py-3" : "border-b border-slate-100 bg-slate-50 px-4 py-3"}>
        <h3 className={dark ? "font-semibold text-slate-100" : "font-semibold text-slate-800"}>Labeled results</h3>
        <p className={dark ? "text-xs text-slate-500" : "text-xs text-slate-500"}>
          Showing first {show.length} of {rows.length} rows
        </p>
      </div>
      <div className="max-h-[480px] overflow-auto">
        <table className="w-full text-left text-sm">
          <thead className={dark ? "sticky top-0 z-[1] border-b border-white/[0.06] bg-[#141928] text-xs uppercase text-slate-500" : "sticky top-0 bg-white shadow-sm"}>
            <tr className={dark ? "" : "border-b border-slate-200 text-xs uppercase text-slate-500"}>
              <th className="px-4 py-3 font-medium">ID</th>
              <th className="px-4 py-3 font-medium">Feedback</th>
              <th className="px-4 py-3 font-medium">Label</th>
              <th className="px-4 py-3 font-medium">Confidence</th>
              <th className="px-4 py-3 font-medium">Theme</th>
            </tr>
          </thead>
          <tbody>
            {show.map((r, i) => (
              <tr
                key={`${r.id}-${i}`}
                className={dark ? "border-b border-white/[0.05] hover:bg-white/[0.03]" : "border-b border-slate-100 hover:bg-slate-50/80"}
              >
                <td className={dark ? "max-w-[100px] truncate px-4 py-2 font-mono text-xs text-slate-500" : "max-w-[100px] truncate px-4 py-2 font-mono text-xs text-slate-600"}>
                  {r.id}
                </td>
                <td className={dark ? "max-w-md px-4 py-2 text-slate-300" : "max-w-md px-4 py-2 text-slate-700"}>
                  {r.feedback.slice(0, 160)}
                  {r.feedback.length > 160 ? "…" : ""}
                </td>
                <td className="px-4 py-2">
                  <span
                    className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                      r.label === "majority"
                        ? dark
                          ? "bg-teal-500/20 text-teal-300"
                          : "bg-emerald-100 text-emerald-800"
                        : r.label === "minority"
                          ? dark
                            ? "bg-rose-500/20 text-rose-300"
                            : "bg-amber-100 text-amber-900"
                          : dark
                            ? "bg-amber-500/15 text-amber-200"
                            : "bg-violet-100 text-violet-800"
                    }`}
                  >
                    {r.label}
                  </span>
                </td>
                <td className={dark ? "px-4 py-2 tabular-nums text-slate-500" : "px-4 py-2 tabular-nums text-slate-600"}>{r.confidence.toFixed(2)}</td>
                <td className={dark ? "px-4 py-2 text-slate-500" : "px-4 py-2 text-slate-600"}>{r.theme}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
