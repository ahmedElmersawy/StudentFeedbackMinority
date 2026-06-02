import { useMemo, useState } from "react";
import type { AnalysisResult, ResultRow } from "../types";

interface Props {
  result: AnalysisResult;
  onDownloadCsv?: () => void;
}

const CAT_COLORS: Record<string, string> = {
  Mental_Health: "#dc2626",              Financial_Hardship: "#c2410c",
  International_Student: "#7c3aed",      First_Generation: "#059669",
  Disability_Accessibility: "#0284c7",   Racial_Ethnic_Minority: "#d97706",
  Gender_Identity: "#9333ea",            International_Academic_Background: "#0891b2",
  Caregiver: "#db2777",                  Statistical_Outlier_Only: "#6b7280",
  Negative_Peer_Flag: "#dc2626",         Suggestion_Flag: "#2563eb",
  Mixed_Signal_Pattern: "#d97706",
};

function catColor(cat: string) { return CAT_COLORS[cat] ?? "#7c3aed"; }

function ExpandableRow({ row, index }: { row: ResultRow; index: number }) {
  const [open, setOpen] = useState(false);
  const text    = ((row.text ?? row.feedback ?? "") as string);
  const cats    = (row.minority_category as string | undefined) ?? "";
  const isOut   = Boolean(row.is_outlier);
  const isClust = Boolean(row.is_minority_cluster);
  const conf    = typeof row.confidence === "number" ? row.confidence : null;

  return (
    <div
      className="card"
      style={{
        border: "1px solid #fde68a",
        background: "#fffbeb",
        marginBottom: 8,
        overflow: "hidden",
      }}
    >
      <button
        style={{ width: "100%", display: "flex", alignItems: "flex-start", gap: 12, padding: 14, background: "none", border: "none", cursor: "pointer", textAlign: "left" }}
        onClick={() => setOpen(o => !o)}
      >
        {/* Badges */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, flexShrink: 0, maxWidth: 200 }}>
          {isOut   && <span className="badge badge-red" style={{ fontSize: 10 }}>outlier</span>}
          {isClust && <span className="badge badge-amber" style={{ fontSize: 10 }}>cluster</span>}
          {cats.split("|").filter(Boolean).map(cat => (
            <span key={cat} className="badge" style={{ fontSize: 10, background: `${catColor(cat)}18`, color: catColor(cat) }}>
              {cat.replace(/_/g, " ")}
            </span>
          ))}
        </div>
        {/* Text */}
        <p style={{ flex: 1, fontSize: 13, color: "var(--text-primary)", lineHeight: 1.6, margin: 0, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: open ? undefined : 2, WebkitBoxOrient: "vertical" }}>
          {text}
        </p>
        {/* Right */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4, flexShrink: 0 }}>
          {conf != null && <span style={{ fontSize: 11, color: "var(--text-muted)", fontVariantNumeric: "tabular-nums" }}>{(conf*100).toFixed(0)}%</span>}
          <svg style={{ transform: open ? "rotate(180deg)" : "", transition: "transform 0.15s" }} width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="var(--text-muted)" strokeWidth="1.5">
            <path d="M2 4l4 4 4-4"/>
          </svg>
        </div>
      </button>
      {open && (
        <div style={{ borderTop: "1px solid #fde68a", padding: "12px 14px", background: "white" }}>
          <p style={{ fontSize: 13, color: "var(--text-primary)", lineHeight: 1.7, margin: "0 0 8px" }}>{text}</p>
          {row.prediction && (
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              Label: <strong>{row.prediction as string}</strong> · Row #{index + 1}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function MinorityPanel({ result, onDownloadCsv }: Props) {
  const minorityRows = useMemo(() => result.rows.filter(r => Boolean(r.is_minority_pattern)), [result.rows]);
  const [search, setSearch]  = useState("");

  const filtered = useMemo(() => {
    if (!search.trim()) return minorityRows;
    const q = search.toLowerCase();
    return minorityRows.filter(r =>
      ((r.text ?? r.feedback ?? "") as string).toLowerCase().includes(q) ||
      ((r.minority_category ?? "") as string).toLowerCase().includes(q)
    );
  }, [minorityRows, search]);

  const catBreakdown = result.summary.minority_category_breakdown ?? {};
  const catEntries   = Object.entries(catBreakdown).sort((a, b) => b[1] - a[1]);
  const maxCat       = Math.max(...catEntries.map(([, v]) => v), 1);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        {[
          { label: "Minority rows",       value: result.summary.minority,                             color: "var(--warning)" },
          { label: "% of dataset",        value: `${result.summary.minority_pct}%`,                  color: "var(--warning)" },
          { label: "Statistical outliers",value: result.rows.filter(r => r.is_outlier).length,       color: "var(--brand)" },
          { label: "Small clusters",      value: result.rows.filter(r => r.is_minority_cluster).length, color: var_info() },
        ].map(k => (
          <div key={k.label} className="card" style={{ padding: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--text-muted)", marginBottom: 8 }}>{k.label}</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: k.color, fontVariantNumeric: "tabular-nums" }}>
              {typeof k.value === "number" ? k.value.toLocaleString() : k.value}
            </div>
          </div>
        ))}
      </div>

      {/* Category bars */}
      {catEntries.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Experiential Category Breakdown</div>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{minorityRows.length.toLocaleString()} flagged rows</span>
          </div>
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {catEntries.map(([cat, count]) => {
              const color = catColor(cat);
              const pct   = Math.round((count / Math.max(1, minorityRows.length)) * 100);
              const w     = Math.round((count / maxCat) * 100);
              return (
                <div key={cat}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 5 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, flexShrink: 0 }} />
                      <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>{cat.replace(/_/g, " ")}</span>
                    </div>
                    <span style={{ color, fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>{count.toLocaleString()} · {pct}%</span>
                  </div>
                  <div className="progress-track" style={{ height: 8 }}>
                    <div className="progress-bar" style={{ width: `${w}%`, background: color }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Flagged rows */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">Flagged Feedback</div>
          <div style={{ display: "flex", gap: 8 }}>
            <div className="search-bar" style={{ width: 200 }}>
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="var(--text-muted)" strokeWidth="1.5"><circle cx="6" cy="6" r="4"/><path d="M10 10l4 4"/></svg>
              <input placeholder="Search…" value={search} onChange={e => setSearch(e.target.value)} />
            </div>
            {onDownloadCsv && (
              <button className="btn btn-secondary btn-sm" onClick={onDownloadCsv}>
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 12V4M4 8l4 4 4-4M2 14h12"/></svg>
                CSV
              </button>
            )}
          </div>
        </div>
        <div className="card-body" style={{ maxHeight: 540, overflowY: "auto" }}>
          {filtered.length === 0 ? (
            <div style={{ padding: "24px 0", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
              {search ? "No results for your search." : "No minority patterns detected."}
            </div>
          ) : (
            <>
              {filtered.slice(0, 100).map((row, i) => <ExpandableRow key={i} row={row} index={i} />)}
              {filtered.length > 100 && (
                <p style={{ textAlign: "center", fontSize: 12, color: "var(--text-muted)", padding: "8px 0" }}>
                  Showing 100 of {filtered.length.toLocaleString()} — download CSV for full list.
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function var_info() { return "var(--info)"; }
