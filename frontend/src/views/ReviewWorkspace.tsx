import { useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisResult, ResultRow } from "../types";

interface Props { result: AnalysisResult }

interface ReviewState { [i: number]: "approved" | "rejected" | null }

function downloadCsv(rows: ResultRow[], corrections: ReviewState, filename: string) {
  const cols = ["text", "prediction", "confidence", "human_label"];
  const lines = rows.map((r, i) => {
    const hl = corrections[i] === "approved" ? r.prediction : corrections[i] === "rejected" ? "REJECTED" : "";
    return cols.map(c => `"${String(c === "human_label" ? hl ?? "" : r[c] ?? "").replace(/"/g, '""')}"`).join(",");
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([[cols.join(","), ...lines].join("\n")], { type: "text/csv" }));
  a.download = filename;
  a.click();
}

export function ReviewWorkspace({ result }: Props) {
  const reviewRows = useMemo(
    () => result.rows.filter(r => Boolean(r.needs_review)),
    [result.rows]
  );
  const [corrections, setCorrections] = useState<ReviewState>({});
  const [cursor, setCursor] = useState(0);
  const [filter, setFilter]  = useState<"all" | "approved" | "rejected" | "pending">("all");
  const [page, setPage]      = useState(0);
  const PAGE = 15;

  const reviewed = Object.values(corrections).filter(v => v != null).length;
  const approved = Object.values(corrections).filter(v => v === "approved").length;
  const rejected = Object.values(corrections).filter(v => v === "rejected").length;

  const filtered = useMemo(() => {
    if (filter === "all") return reviewRows;
    if (filter === "approved") return reviewRows.filter((_, i) => corrections[i] === "approved");
    if (filter === "rejected") return reviewRows.filter((_, i) => corrections[i] === "rejected");
    return reviewRows.filter((_, i) => !corrections[i]);
  }, [reviewRows, corrections, filter]);

  const totalPages = Math.ceil(filtered.length / PAGE);
  const pageRows   = filtered.slice(page * PAGE, (page + 1) * PAGE);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      const globalIdx = pageRows[cursor % PAGE] ? reviewRows.indexOf(pageRows[cursor % PAGE]) : -1;
      if (e.key === "y" || e.key === "Y") { if (globalIdx >= 0) setCorrections(p => ({ ...p, [globalIdx]: p[globalIdx] === "approved" ? null : "approved" })); }
      if (e.key === "n" || e.key === "N") { if (globalIdx >= 0) setCorrections(p => ({ ...p, [globalIdx]: p[globalIdx] === "rejected" ? null : "rejected" })); }
      if (e.key === "j" || e.key === "ArrowDown") setCursor(c => Math.min(c + 1, pageRows.length - 1));
      if (e.key === "k" || e.key === "ArrowUp")   setCursor(c => Math.max(c - 1, 0));
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [cursor, pageRows, reviewRows]);

  if (reviewRows.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon" style={{ background: "var(--success-bg)" }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="1.5">
            <circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-5"/>
          </svg>
        </div>
        <h3 style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", margin: "0 0 6px" }}>All predictions above threshold</h3>
        <p style={{ fontSize: 13, color: "var(--text-muted)" }}>No rows need human review.</p>
      </div>
    );
  }

  const curConf = result.summary.avg_confidence;
  const pctDone = Math.round((reviewed / reviewRows.length) * 100);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: "var(--text-primary)", margin: 0 }}>Review Workspace</h1>
          <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>
            {reviewRows.length.toLocaleString()} low-confidence predictions · confidence threshold {curConf > 0 ? "0.65" : "0.65"}
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {reviewed > 0 && (
            <button className="btn btn-primary btn-sm" onClick={() => downloadCsv(reviewRows, corrections, "review_corrections.csv")}>
              Export {reviewed} Corrections
            </button>
          )}
          <button className="btn btn-secondary btn-sm" onClick={() => downloadCsv(reviewRows, corrections, "review_queue.csv")}>
            Export Queue
          </button>
        </div>
      </div>

      {/* Progress + stats */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 16 }}>
        <div className="card" style={{ padding: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--text-muted)", marginBottom: 6 }}>
                <span>{reviewed} of {reviewRows.length} reviewed</span>
                <span style={{ fontWeight: 700, color: "var(--brand)" }}>{pctDone}%</span>
              </div>
              <div className="progress-track" style={{ height: 8 }}>
                <div className="progress-bar" style={{ width: `${pctDone}%`, transition: "width 0.4s ease" }} />
              </div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 20 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 22, fontWeight: 800, color: "var(--success)", fontVariantNumeric: "tabular-nums" }}>{approved}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Accepted</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 22, fontWeight: 800, color: "var(--danger)", fontVariantNumeric: "tabular-nums" }}>{rejected}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Rejected</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 22, fontWeight: 800, color: "var(--text-muted)", fontVariantNumeric: "tabular-nums" }}>{reviewRows.length - reviewed}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Remaining</div>
            </div>
          </div>
        </div>

        {/* Keyboard hint */}
        <div className="card" style={{ padding: 20, minWidth: 200 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 10 }}>Keyboard Shortcuts</div>
          {[
            ["Y", "Accept prediction"],
            ["N", "Reject prediction"],
            ["J / ↓", "Next record"],
            ["K / ↑", "Previous record"],
          ].map(([key, action]) => (
            <div key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <kbd style={{ background: "var(--surface-3)", border: "1px solid var(--border-2)", borderRadius: 4, padding: "2px 7px", fontSize: 11, fontFamily: "monospace", color: "var(--text-primary)", boxShadow: "0 1px 0 var(--border-2)" }}>
                {key}
              </kbd>
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{action}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Filter tabs */}
      <div style={{ display: "flex", gap: 0, borderBottom: "2px solid var(--border)" }}>
        {([
          { id: "all",      label: `All (${reviewRows.length})` },
          { id: "pending",  label: `Pending (${reviewRows.length - reviewed})` },
          { id: "approved", label: `Accepted (${approved})` },
          { id: "rejected", label: `Rejected (${rejected})` },
        ] as const).map(tab => (
          <button
            key={tab.id}
            className={`tab-btn${filter === tab.id ? " active" : ""}`}
            onClick={() => { setFilter(tab.id); setPage(0); }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Review list */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {pageRows.map((row, localI) => {
          const globalIdx = reviewRows.indexOf(row);
          const state     = corrections[globalIdx] ?? null;
          const text      = ((row.text ?? row.feedback ?? "") as string);
          const conf      = typeof row.confidence === "number" ? row.confidence : null;
          const pred      = (row.prediction ?? "") as string;
          const isCurrent = cursor === localI;

          return (
            <div
              key={globalIdx}
              className="card"
              style={{
                border: `2px solid ${state === "approved" ? "var(--success)" : state === "rejected" ? "var(--danger)" : isCurrent ? "var(--brand)" : "var(--border)"}`,
                background: state === "approved" ? "var(--success-bg)" : state === "rejected" ? "var(--danger-bg)" : isCurrent ? "var(--brand-light)" : "var(--surface)",
                transition: "all 0.15s",
                cursor: "pointer",
              }}
              onClick={() => setCursor(localI)}
            >
              <div style={{ padding: 16 }}>
                {/* Meta row */}
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10, flexWrap: "wrap" }}>
                  {pred && (
                    <span className="badge badge-amber">{pred.replace(/_/g, " ")}</span>
                  )}
                  {conf != null && (
                    <span style={{ fontSize: 12, color: conf >= 0.65 ? "var(--warning)" : "var(--danger)", fontWeight: 500, fontVariantNumeric: "tabular-nums" }}>
                      {(conf * 100).toFixed(1)}% confidence
                    </span>
                  )}
                  {state && (
                    <span className={`badge ${state === "approved" ? "badge-green" : "badge-red"}`}>
                      {state === "approved" ? "✓ Accepted" : "✗ Rejected"}
                    </span>
                  )}
                  <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-muted)" }}>#{globalIdx + 1}</span>
                </div>

                {/* Text */}
                <p style={{ fontSize: 13, color: "var(--text-primary)", lineHeight: 1.7, margin: "0 0 14px" }}>
                  {text}
                </p>

                {/* Actions */}
                <div style={{ display: "flex", gap: 8 }}>
                  <button
                    className={`btn btn-sm ${state === "approved" ? "btn-primary" : "btn-secondary"}`}
                    style={state === "approved" ? { background: "var(--success)", boxShadow: "none" } : {}}
                    onClick={e => { e.stopPropagation(); setCorrections(p => ({ ...p, [globalIdx]: p[globalIdx] === "approved" ? null : "approved" })); }}
                  >
                    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M2 8l4 4 8-8"/></svg>
                    Correct
                  </button>
                  <button
                    className={`btn btn-sm ${state === "rejected" ? "btn-danger" : "btn-secondary"}`}
                    onClick={e => { e.stopPropagation(); setCorrections(p => ({ ...p, [globalIdx]: p[globalIdx] === "rejected" ? null : "rejected" })); }}
                  >
                    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 3l10 10M13 3L3 13"/></svg>
                    Wrong Label
                  </button>
                </div>
              </div>
            </div>
          );
        })}
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
