import { useMemo } from "react";
import type { AnalysisResult, ResultRow } from "../types";

interface Props { result: AnalysisResult }

function downloadCsv(rows: ResultRow[], filename: string) {
  const cols   = ["text", "prediction", "confidence", "mismatch_type"];
  const csv    = [cols.join(","), ...rows.map(r => cols.map(c => `"${String(r[c] ?? "").replace(/"/g, '""')}"`).join(","))].join("\n");
  const a      = document.createElement("a");
  a.href       = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download   = filename;
  a.click();
}

function MismatchCard({ row, index }: { row: ResultRow; index: number }) {
  const text   = ((row.text ?? row.feedback ?? "") as string);
  const isHigh = row.mismatch_type === "HIGH_MISMATCH";
  const conf   = typeof row.confidence === "number" ? row.confidence : null;

  return (
    <div style={{
      border: `1px solid ${isHigh ? "#fed7aa" : "#e9d5ff"}`,
      background: isHigh ? "#fff7ed" : "#faf5ff",
      borderRadius: 10, padding: 14, marginBottom: 8,
    }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span className={`badge ${isHigh ? "badge-orange" : "badge-violet"}`}>
          {row.mismatch_type as string}
        </span>
        {row.prediction && (
          <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            Label: <strong>{row.prediction as string}</strong>
          </span>
        )}
        {conf != null && (
          <span style={{ fontSize: 12, color: "var(--text-secondary)", fontVariantNumeric: "tabular-nums" }}>
            {(conf * 100).toFixed(1)}% conf
          </span>
        )}
        <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-muted)" }}>#{index + 1}</span>
      </div>
      <p style={{ fontSize: 13, color: "var(--text-primary)", lineHeight: 1.65, margin: 0,
        display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
        {text}
      </p>
    </div>
  );
}

export function MismatchPanel({ result }: Props) {
  const all     = useMemo(() => result.rows.filter(r => Boolean(r.mismatch_flag)), [result.rows]);
  const high    = all.filter(r => r.mismatch_type === "HIGH_MISMATCH");
  const reverse = all.filter(r => r.mismatch_type === "REVERSE_MISMATCH");

  if (all.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--brand)" strokeWidth="1.5">
            <circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-5"/>
          </svg>
        </div>
        <h3 style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", margin: "0 0 6px" }}>No mismatches detected</h3>
        <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
          {result.summary.total > 0 ? "Dataset may not have numeric rating columns." : "Upload a dataset first."}
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
        {[
          { label: "Total mismatches", value: all.length,    pct: result.summary.mismatch_pct, color: "#c2410c" },
          { label: "High mismatch",    value: high.length,   pct: null,                        color: "var(--warning)" },
          { label: "Reverse mismatch", value: reverse.length,pct: null,                        color: "var(--brand)" },
        ].map(k => (
          <div key={k.label} className="card" style={{ padding: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--text-muted)", marginBottom: 8 }}>{k.label}</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: k.color, fontVariantNumeric: "tabular-nums" }}>{k.value.toLocaleString()}</div>
            {k.pct != null && <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>{k.pct}% of dataset</div>}
          </div>
        ))}
      </div>

      {/* Explanation */}
      <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 10, padding: "14px 18px", fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7 }}>
        <strong style={{ color: "#c2410c" }}>HIGH_MISMATCH</strong> — numeric rating ≥ 3.8 but text classified as Negative/Neutral. Student may be masking frustration. &nbsp;·&nbsp;
        <strong style={{ color: "var(--brand)" }}>REVERSE_MISMATCH</strong> — rating ≤ 2.5 but text classified as Positive. Student may be hedging on the Likert scale.
      </div>

      {/* Two column lists */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {[
          { title: "High Mismatch", rows: high, filename: "high_mismatch.csv" },
          { title: "Reverse Mismatch", rows: reverse, filename: "reverse_mismatch.csv" },
        ].map(col => (
          <div key={col.title} className="card">
            <div className="card-header">
              <div className="card-title">{col.title}</div>
              <button className="btn btn-secondary btn-sm" onClick={() => downloadCsv(col.rows, col.filename)}>CSV</button>
            </div>
            <div className="card-body" style={{ maxHeight: 480, overflowY: "auto" }}>
              {col.rows.length === 0
                ? <p style={{ color: "var(--text-muted)", fontSize: 13 }}>None detected.</p>
                : col.rows.slice(0, 50).map((row, i) => <MismatchCard key={i} row={row} index={i} />)
              }
            </div>
          </div>
        ))}
      </div>

      <div style={{ textAlign: "right" }}>
        <button className="btn btn-secondary btn-sm" onClick={() => downloadCsv(all, "all_mismatches.csv")}>
          Download all {all.length} mismatches
        </button>
      </div>
    </div>
  );
}
