import { useMemo, useState } from "react";
import type { ResultRow } from "../types";

interface Props {
  rows: ResultRow[];
  onDownloadCsv?: () => void;
}

type SortKey = "confidence" | "prediction" | "is_minority_pattern" | "mismatch_flag";

function confBadge(c: number) {
  if (c >= 0.8) return <span className="badge badge-green">{(c*100).toFixed(1)}%</span>;
  if (c >= 0.65) return <span className="badge badge-amber">{(c*100).toFixed(1)}%</span>;
  return <span className="badge badge-red">{(c*100).toFixed(1)}%</span>;
}

function predBadge(p: string) {
  const l = p.toLowerCase();
  const cls = l.includes("positive") ? "badge-green" : l.includes("negative") ? "badge-red" : l.includes("minority") ? "badge-amber" : l.includes("suggestion") ? "badge-blue" : "badge-slate";
  return <span className={`badge ${cls}`} style={{ fontSize: 10 }}>{p.replace(/_/g, " ")}</span>;
}

const PAGE_SIZE = 50;

export function ResultsTable({ rows, onDownloadCsv }: Props) {
  const [search,        setSearch]        = useState("");
  const [filterPred,    setFilterPred]    = useState("all");
  const [filterMinority,setFilterMinority]= useState<"all"|"minority"|"non-minority">("all");
  const [sortKey,       setSortKey]       = useState<SortKey>("confidence");
  const [sortAsc,       setSortAsc]       = useState(false);
  const [page,          setPage]          = useState(0);

  const predictions = useMemo(
    () => ["all", ...Array.from(new Set(rows.map(r => r.prediction ?? "").filter(Boolean)))],
    [rows]
  );

  const filtered = useMemo(() => {
    let r = rows;
    if (search.trim()) {
      const q = search.toLowerCase();
      r = r.filter(row => ((row.text ?? row.feedback ?? "") as string).toLowerCase().includes(q));
    }
    if (filterPred !== "all")  r = r.filter(row => row.prediction === filterPred);
    if (filterMinority === "minority")     r = r.filter(row => Boolean(row.is_minority_pattern));
    if (filterMinority === "non-minority") r = r.filter(row => !row.is_minority_pattern);
    return [...r].sort((a, b) => {
      const av = (a[sortKey] as number) ?? 0;
      const bv = (b[sortKey] as number) ?? 0;
      return sortAsc ? av - bv : bv - av;
    });
  }, [rows, search, filterPred, filterMinority, sortKey, sortAsc]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageRows   = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  function toggleSort(k: SortKey) {
    if (sortKey === k) setSortAsc(a => !a);
    else { setSortKey(k); setSortAsc(false); }
    setPage(0);
  }

  function SortTh({ k, children }: { k: SortKey; children: React.ReactNode }) {
    return (
      <th style={{ cursor: "pointer", userSelect: "none" }} onClick={() => toggleSort(k)}>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          {children}
          {sortKey === k && <span style={{ fontSize: 9, color: "var(--brand)" }}>{sortAsc ? "↑" : "↓"}</span>}
        </span>
      </th>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Toolbar */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
        <div className="search-bar" style={{ flex: 1, minWidth: 200, maxWidth: 340 }}>
          <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="var(--text-muted)" strokeWidth="1.5"><circle cx="6" cy="6" r="4"/><path d="M10 10l4 4"/></svg>
          <input
            placeholder="Search feedback text…"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
          />
        </div>

        <select
          className="select-field"
          style={{ width: "auto", minWidth: 180 }}
          value={filterPred}
          onChange={e => { setFilterPred(e.target.value); setPage(0); }}
        >
          {predictions.map(p => (
            <option key={p} value={p}>{p === "all" ? "All predictions" : p.replace(/_/g, " ")}</option>
          ))}
        </select>

        <select
          className="select-field"
          style={{ width: "auto" }}
          value={filterMinority}
          onChange={e => { setFilterMinority(e.target.value as typeof filterMinority); setPage(0); }}
        >
          <option value="all">All rows</option>
          <option value="minority">Minority only</option>
          <option value="non-minority">Non-minority only</option>
        </select>

        <span style={{ fontSize: 13, color: "var(--text-muted)", marginLeft: "auto" }}>
          {filtered.length.toLocaleString()} rows
        </span>

        {onDownloadCsv && (
          <button className="btn btn-secondary btn-sm" onClick={onDownloadCsv}>
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 12V4M4 8l4 4 4-4M2 14h12"/></svg>
            Download CSV
          </button>
        )}
      </div>

      {/* Table */}
      <div className="card" style={{ overflowX: "auto" }}>
        <table className="data-table">
          <thead>
            <tr>
              <SortTh k="prediction">Prediction</SortTh>
              <SortTh k="confidence">Confidence</SortTh>
              <SortTh k="is_minority_pattern">Minority</SortTh>
              <SortTh k="mismatch_flag">Mismatch</SortTh>
              <th>Category</th>
              <th>Feedback</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, i) => {
              const text       = ((row.text ?? row.feedback ?? "") as string);
              const conf       = typeof row.confidence === "number" ? row.confidence : null;
              const pred       = (row.prediction ?? "") as string;
              const isMinority = Boolean(row.is_minority_pattern);
              const isMismatch = Boolean(row.mismatch_flag);
              const cats       = (row.minority_category as string | undefined) ?? "";

              return (
                <tr key={i} style={{ background: isMinority ? "#fffbeb" : undefined }}>
                  <td>{pred && predBadge(pred)}</td>
                  <td>{conf != null && confBadge(conf)}</td>
                  <td>
                    {isMinority && <span className="badge badge-amber">minority</span>}
                  </td>
                  <td>
                    {isMismatch && (
                      <span className={`badge ${row.mismatch_type === "HIGH_MISMATCH" ? "badge-orange" : "badge-violet"}`}>
                        {row.mismatch_type as string}
                      </span>
                    )}
                  </td>
                  <td style={{ maxWidth: 160 }}>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                      {cats.split("|").filter(Boolean).map(cat => (
                        <span key={cat} className="badge badge-slate" style={{ fontSize: 10 }}>
                          {cat.replace(/_/g, " ")}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td style={{ maxWidth: 400 }}>
                    <p style={{ margin: 0, fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5,
                      display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                      {text}
                    </p>
                  </td>
                </tr>
              );
            })}
            {pageRows.length === 0 && (
              <tr>
                <td colSpan={6} style={{ textAlign: "center", padding: "32px", color: "var(--text-muted)", fontSize: 13 }}>
                  No rows match the current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
          <button className="btn btn-secondary btn-sm" disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Prev</button>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Page {page + 1} of {totalPages}</span>
          <button className="btn btn-secondary btn-sm" disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>Next →</button>
        </div>
      )}
    </div>
  );
}
