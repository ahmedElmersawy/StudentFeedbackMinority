import { useEffect, useMemo, useState } from "react";
import type { AnalysisResult } from "../types";
import { fetchLabels, downloadLabelCsv } from "../api/client";
import type { LabelStat } from "../api/client";
import { SparkBar, ConfidenceBar, labelColor } from "../components/Charts";

interface Props {
  jobId: string;
  result: AnalysisResult;
  onToast: (msg: string) => void;
}

type SortKey = "count" | "pct" | "confidence" | "minority" | "label";

function LabelDetailPanel({ stat, jobId, total, onClose, onToast }: {
  stat: LabelStat; jobId: string; total: number; onClose: () => void; onToast: (m: string) => void;
}) {
  const color = labelColor(stat.label, 0);

  return (
    <div className="card animate-scale-in" style={{ position: "sticky", top: 0 }}>
      <div className="card-header">
        <div>
          <div className="card-title" style={{ color }}>{stat.label.replace(/_/g, " ")}</div>
          <div className="card-subtitle">{stat.count.toLocaleString()} records · {stat.pct}% of dataset</div>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={onClose}>✕</button>
      </div>
      <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {/* Metrics */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(200px,1fr))", gap: 12 }}>
          {[
            { label: "Records",       value: stat.count.toLocaleString(),                         color: "var(--text-primary)" },
            { label: "% of Dataset",  value: `${stat.pct}%`,                                      color: "var(--brand)" },
            { label: "Avg Confidence",value: `${(stat.avg_confidence * 100).toFixed(1)}%`,        color: stat.avg_confidence >= 0.8 ? "var(--success)" : "var(--warning)" },
            { label: "Minority Rows", value: stat.minority_count.toLocaleString(),                 color: stat.minority_count > 0 ? "var(--warning)" : "var(--text-muted)" },
          ].map(m => (
            <div key={m.label} style={{ background: "var(--surface-2)", borderRadius: 10, padding: "14px 16px", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 500, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.06em" }}>{m.label}</div>
              <div style={{ fontSize: 22, fontWeight: 800, color: m.color, fontVariantNumeric: "tabular-nums" }}>{m.value}</div>
            </div>
          ))}
        </div>

        {/* Confidence bar */}
        <div>
          <div style={{ fontSize: 12, fontWeight: 500, color: "var(--text-secondary)", marginBottom: 8 }}>Average Confidence</div>
          <div className="progress-track" style={{ height: 8 }}>
            <div className="progress-bar" style={{
              width: `${stat.avg_confidence * 100}%`,
              background: stat.avg_confidence >= 0.8 ? "var(--success)" : stat.avg_confidence >= 0.65 ? "var(--warning)" : "var(--danger)",
            }} />
          </div>
        </div>

        {/* Distribution pill */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ flex: 1, height: 24, borderRadius: 6, overflow: "hidden", background: "var(--surface-3)", position: "relative" }}>
            <div style={{ position: "absolute", inset: 0, width: `${stat.pct}%`, background: `${color}30`, borderRight: `3px solid ${color}` }} />
          </div>
          <span style={{ fontSize: 12, fontWeight: 600, color, minWidth: 48, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
            {stat.pct}%
          </span>
        </div>

        {/* Actions */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <button
            className="btn btn-primary"
            onClick={() => { downloadLabelCsv(jobId, stat.label); onToast(`Downloading ${stat.label} CSV…`); }}
          >
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 12V4M4 8l4 4 4-4M2 14h12"/></svg>
            Download CSV ({stat.count.toLocaleString()} rows)
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => {
              const data = JSON.stringify({ label: stat.label, count: stat.count, pct: stat.pct, avg_confidence: stat.avg_confidence }, null, 2);
              const a = document.createElement("a");
              a.href = URL.createObjectURL(new Blob([data], { type: "application/json" }));
              a.download = `${stat.label}_stats.json`;
              a.click();
              onToast("Downloading JSON stats…");
            }}
          >
            Export Stats JSON
          </button>
        </div>
      </div>
    </div>
  );
}

export function LabelExplorer({ jobId, result, onToast }: Props) {
  const [stats, setStats]   = useState<LabelStat[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch]   = useState("");
  const [sort, setSort]       = useState<SortKey>("count");
  const [asc, setAsc]         = useState(false);
  const [selected, setSelected] = useState<LabelStat | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchLabels(jobId)
      .then(d => setStats(d.labels))
      .catch(() => setStats([]))
      .finally(() => setLoading(false));
  }, [jobId]);

  const total = result.summary.total;
  const maxCount = Math.max(...stats.map(s => s.count), 1);

  const sorted = useMemo(() => {
    const filtered = stats.filter(s => s.label.toLowerCase().includes(search.toLowerCase()));
    return [...filtered].sort((a, b) => {
      let cmp = 0;
      if (sort === "count")      cmp = b.count            - a.count;
      if (sort === "pct")        cmp = b.pct              - a.pct;
      if (sort === "confidence") cmp = b.avg_confidence   - a.avg_confidence;
      if (sort === "minority")   cmp = b.minority_count   - a.minority_count;
      if (sort === "label")      cmp = a.label.localeCompare(b.label);
      return asc ? -cmp : cmp;
    });
  }, [stats, search, sort, asc]);

  function toggleSort(k: SortKey) {
    if (sort === k) setAsc(a => !a);
    else { setSort(k); setAsc(false); }
  }

  function SortTh({ k, children }: { k: SortKey; children: React.ReactNode }) {
    const active = sort === k;
    return (
      <th
        style={{ cursor: "pointer", userSelect: "none", whiteSpace: "nowrap" }}
        onClick={() => toggleSort(k)}
      >
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          {children}
          <span style={{ fontSize: 9, color: active ? "var(--brand)" : "transparent" }}>
            {asc ? "↑" : "↓"}
          </span>
        </span>
      </th>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: "var(--text-primary)", margin: 0 }}>Label Explorer</h1>
          <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>
            {stats.length} labels · {total.toLocaleString()} total records
          </p>
        </div>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => {
            stats.forEach(s => downloadLabelCsv(jobId, s.label));
            onToast("Downloading all label CSVs…");
          }}
        >
          Download All Labels
        </button>
      </div>

      {/* Summary chips */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {["Positive", "Negative", "Minority", "Suggestion", "Self"].map(group => {
          const matching = stats.filter(s => s.label.toLowerCase().includes(group.toLowerCase()));
          if (matching.length === 0) return null;
          const total2 = matching.reduce((s, m) => s + m.count, 0);
          return (
            <span key={group} className="badge badge-slate" style={{ fontSize: 12, padding: "4px 10px" }}>
              {group}: {total2.toLocaleString()}
            </span>
          );
        })}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: selected ? "minmax(0,1fr) min(320px,100%)" : "1fr", gap: 16, alignItems: "start" }}>
        {/* Table */}
        <div className="card">
          {/* Toolbar */}
          <div className="card-header">
            <div className="search-bar" style={{ flex: 1, maxWidth: 320 }}>
              <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="var(--text-muted)" strokeWidth="1.5">
                <circle cx="6" cy="6" r="4"/><path d="M10 10l4 4"/>
              </svg>
              <input
                placeholder="Search labels…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{sorted.length} labels</span>
          </div>

          {loading ? (
            <div className="card-body">
              {[...Array(8)].map((_, i) => (
                <div key={i} style={{ display: "flex", gap: 16, alignItems: "center", padding: "10px 0", borderBottom: "1px solid var(--border)" }}>
                  <div className="skeleton" style={{ width: 160, height: 14 }} />
                  <div className="skeleton" style={{ flex: 1, height: 14 }} />
                  <div className="skeleton" style={{ width: 60, height: 14 }} />
                </div>
              ))}
            </div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <SortTh k="label">Label</SortTh>
                    <SortTh k="count">Records</SortTh>
                    <th>Distribution</th>
                    <SortTh k="pct">Share</SortTh>
                    <SortTh k="confidence">Avg Conf</SortTh>
                    <SortTh k="minority">Minority</SortTh>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.map(stat => {
                    const color  = labelColor(stat.label, 0);
                    const isSelected = selected?.label === stat.label;
                    return (
                      <tr
                        key={stat.label}
                        style={{ cursor: "pointer", background: isSelected ? "var(--brand-light)" : undefined }}
                        onClick={() => setSelected(isSelected ? null : stat)}
                      >
                        <td>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, flexShrink: 0 }} />
                            <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-primary)" }}>
                              {stat.label.replace(/_/g, " ")}
                            </span>
                          </div>
                        </td>
                        <td style={{ fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>
                          {stat.count.toLocaleString()}
                        </td>
                        <td style={{ minWidth: 120, maxWidth: 180 }}>
                          <SparkBar value={stat.count} max={maxCount} color={color} height={18} />
                        </td>
                        <td style={{ fontVariantNumeric: "tabular-nums" }}>
                          <span className="badge badge-slate">{stat.pct}%</span>
                        </td>
                        <td style={{ minWidth: 120 }}>
                          <ConfidenceBar value={stat.avg_confidence} />
                        </td>
                        <td>
                          {stat.minority_count > 0 ? (
                            <span className="badge badge-amber">{stat.minority_count}</span>
                          ) : (
                            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>—</span>
                          )}
                        </td>
                        <td onClick={e => e.stopPropagation()}>
                          <button
                            className="btn btn-ghost btn-sm"
                            data-tip="Download CSV"
                            onClick={() => { downloadLabelCsv(jobId, stat.label); onToast(`Downloading ${stat.label}…`); }}
                          >
                            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 12V4M4 8l4 4 4-4M2 14h12"/></svg>
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                  {sorted.length === 0 && (
                    <tr>
                      <td colSpan={7} style={{ textAlign: "center", padding: "32px 16px", color: "var(--text-muted)" }}>
                        No labels match your search.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Detail panel */}
        {selected && (
          <LabelDetailPanel
            stat={selected}
            jobId={jobId}
            total={total}
            onClose={() => setSelected(null)}
            onToast={onToast}
          />
        )}
      </div>
    </div>
  );
}
