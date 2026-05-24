import { useMemo, useState } from "react";
import type { ResultRow } from "../types";

interface Props {
  rows: ResultRow[];
  onDownloadCsv?: () => void;
}

type SortKey = "confidence" | "prediction" | "is_minority_pattern" | "mismatch_flag";
type SortDir = "asc" | "desc";

function cn(...xs: (string | false | undefined)[]) {
  return xs.filter(Boolean).join(" ");
}

function confColor(c: number) {
  if (c >= 0.8) return "text-emerald-400";
  if (c >= 0.65) return "text-amber-400";
  return "text-rose-400";
}

function confBg(c: number) {
  if (c >= 0.8) return "bg-emerald-500/15";
  if (c >= 0.65) return "bg-amber-500/15";
  return "bg-rose-500/15";
}

function predColor(p: string) {
  const l = p.toLowerCase();
  if (l.startsWith("positive")) return "text-emerald-300 bg-emerald-500/10";
  if (l.startsWith("negative")) return "text-rose-300 bg-rose-500/10";
  return "text-amber-300 bg-amber-500/10";
}

const PAGE_SIZE = 50;

export function ResultsTable({ rows, onDownloadCsv }: Props) {
  const [search, setSearch] = useState("");
  const [filterPred, setFilterPred] = useState<string>("all");
  const [filterMinority, setFilterMinority] = useState<"all" | "minority" | "non-minority">("all");
  const [sortKey, setSortKey] = useState<SortKey>("confidence");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(0);

  const predictions = useMemo(
    () => ["all", ...Array.from(new Set(rows.map((r) => r.prediction ?? "").filter(Boolean)))],
    [rows],
  );

  const filtered = useMemo(() => {
    let r = rows;
    if (search.trim()) {
      const q = search.toLowerCase();
      r = r.filter((row) => ((row.text ?? row.feedback ?? "") as string).toLowerCase().includes(q));
    }
    if (filterPred !== "all") {
      r = r.filter((row) => row.prediction === filterPred);
    }
    if (filterMinority === "minority") {
      r = r.filter((row) => Boolean(row.is_minority_pattern));
    } else if (filterMinority === "non-minority") {
      r = r.filter((row) => !row.is_minority_pattern);
    }
    return [...r].sort((a, b) => {
      const av = (a[sortKey] as number) ?? 0;
      const bv = (b[sortKey] as number) ?? 0;
      return sortDir === "asc" ? av - bv : bv - av;
    });
  }, [rows, search, filterPred, filterMinority, sortKey, sortDir]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageRows = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("desc"); }
    setPage(0);
  };

  const SortBtn = ({ k, label }: { k: SortKey; label: string }) => (
    <button
      onClick={() => toggleSort(k)}
      className="flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500 hover:text-slate-300"
    >
      {label}
      {sortKey === k && <span className="text-indigo-400">{sortDir === "asc" ? "↑" : "↓"}</span>}
    </button>
  );

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Search feedback text…"
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(0); }}
          className="min-w-[200px] flex-1 rounded-lg border border-white/[0.1] bg-white/[0.04] px-3 py-1.5 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-400"
        />
        <select
          value={filterPred}
          onChange={(e) => { setFilterPred(e.target.value); setPage(0); }}
          className="rounded-lg border border-white/[0.1] bg-[#0f1117] px-3 py-1.5 text-sm text-slate-300"
        >
          {predictions.map((p) => (
            <option key={p} value={p}>{p === "all" ? "All predictions" : p}</option>
          ))}
        </select>
        <select
          value={filterMinority}
          onChange={(e) => { setFilterMinority(e.target.value as typeof filterMinority); setPage(0); }}
          className="rounded-lg border border-white/[0.1] bg-[#0f1117] px-3 py-1.5 text-sm text-slate-300"
        >
          <option value="all">All rows</option>
          <option value="minority">Minority only</option>
          <option value="non-minority">Non-minority only</option>
        </select>
        <span className="ml-auto text-xs text-slate-600">{filtered.length.toLocaleString()} rows</span>
        {onDownloadCsv && (
          <button
            onClick={onDownloadCsv}
            className="rounded-lg border border-white/[0.1] px-3 py-1.5 text-xs font-medium text-slate-400 hover:text-slate-200"
          >
            Download CSV
          </button>
        )}
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-2xl border border-white/[0.08]">
        <table className="w-full text-sm">
          <thead className="border-b border-white/[0.08] bg-white/[0.03]">
            <tr>
              <th className="px-4 py-3 text-left"><SortBtn k="prediction" label="Prediction" /></th>
              <th className="px-4 py-3 text-left"><SortBtn k="confidence" label="Confidence" /></th>
              <th className="px-4 py-3 text-left"><SortBtn k="is_minority_pattern" label="Minority" /></th>
              <th className="px-4 py-3 text-left"><SortBtn k="mismatch_flag" label="Mismatch" /></th>
              <th className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                Category
              </th>
              <th className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                Feedback
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/[0.04]">
            {pageRows.map((row, i) => {
              const text = ((row.text ?? row.feedback ?? "") as string);
              const conf = typeof row.confidence === "number" ? row.confidence : null;
              const pred = (row.prediction ?? "") as string;
              const isMinority = Boolean(row.is_minority_pattern);
              const isMismatch = Boolean(row.mismatch_flag);
              const cats = (row.minority_category as string | undefined) ?? "";

              return (
                <tr key={i} className={cn("transition-colors hover:bg-white/[0.02]", isMinority && "bg-amber-500/[0.03]")}>
                  <td className="px-4 py-3">
                    {pred && (
                      <span className={cn("inline-flex rounded-full px-2 py-0.5 text-[11px] font-semibold", predColor(pred))}>
                        {pred}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    {conf != null && (
                      <span className={cn("inline-flex rounded-full px-2 py-0.5 text-[11px] font-semibold", confBg(conf), confColor(conf))}>
                        {(conf * 100).toFixed(1)}%
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    {isMinority && (
                      <span className="rounded-full bg-rose-500/15 px-2 py-0.5 text-[11px] font-semibold text-rose-300">
                        minority
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    {isMismatch && (
                      <span className={cn(
                        "rounded-full px-2 py-0.5 text-[11px] font-semibold",
                        row.mismatch_type === "HIGH_MISMATCH"
                          ? "bg-orange-500/15 text-orange-300"
                          : "bg-purple-500/15 text-purple-300",
                      )}>
                        {row.mismatch_type as string}
                      </span>
                    )}
                  </td>
                  <td className="max-w-[140px] px-4 py-3">
                    {cats && (
                      <div className="flex flex-wrap gap-1">
                        {cats.split("|").filter(Boolean).map((cat) => (
                          <span key={cat} className="rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] text-indigo-300">
                            {cat.replace(/_/g, " ")}
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                  <td className="max-w-[400px] px-4 py-3">
                    <p className="line-clamp-2 text-xs text-slate-400">{text}</p>
                  </td>
                </tr>
              );
            })}
            {pageRows.length === 0 && (
              <tr>
                <td colSpan={6} className="py-10 text-center text-sm text-slate-600">
                  No rows match the current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="rounded-lg border border-white/[0.1] px-3 py-1.5 text-xs text-slate-400 disabled:opacity-30"
          >
            ← Prev
          </button>
          <span className="text-xs text-slate-600">Page {page + 1} of {totalPages}</span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className="rounded-lg border border-white/[0.1] px-3 py-1.5 text-xs text-slate-400 disabled:opacity-30"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}
